/*
 * Copyright 2026 CEA (www.cea.fr)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

/**
 * @file block_write.h
 * @brief The chunk-aligned write (docs/gpu_design.md, 7.3 mitigation 2, and section 13,
 *        "relaxed publication"): a block of threads writes a(i) = f(i) over a part of one
 *        chunk, with no device-scope acquire or release.
 *
 * @details Stage 1, experimental (see chunk_desc.h), from the prototype
 *          (gpu/proto/zero_any_device.h, block_write_relaxed). Measured there: first write
 *          1.83x a flat write on the GH200, 3.4x on the MI300A (S = 1024), against 2.2x and
 *          15x with the acquire/release publication.
 *
 *          Writer::write, called by every thread of a block for chunk c and [lo, hi):
 *          - The descriptor is loaded relaxed. A buffer of an earlier kernel (settled) was
 *            ordered by the kernel boundary: the block writes into it.
 *          - A compressed chunk: the block votes; if every value written is the constant,
 *            nothing happens. Otherwise it takes a buffer from the arena. A block that
 *            covers the whole chunk fills it (values, constant elsewhere) and publishes it
 *            by a relaxed CAS. A part of the chunk: it publishes the unfilled buffer tagged
 *            fresh, then writes its elements only.
 *          - A fresh buffer (published by another block of the same kernel, or of an earlier
 *            one not settled yet): the block writes its elements and sets their bits in the
 *            chunk's `written` bitmap (one relaxed atomic or per 32 elements).
 *          - After the kernel, settle_fresh puts the constant where no bit is set, clears the
 *            bitmap and removes the tag. Read kernels never see a tag.
 *
 *          The blocks of one kernel write disjoint elements (the caller's contract: no
 *          element written twice in a kernel), so nothing has to order one block's stores
 *          before another's; the kernel boundary orders them for the readers.
 *
 *          Arena exhausted: the block writes nothing in that chunk and records its span; the
 *          host grows the arena and writes the span again with the same f (Array::finish:
 *          f must be a function of i, the same in both passes). A lost CAS (two blocks on
 *          one compressed chunk) parks the loser's buffer in its warp's spare slot, which the
 *          warp's next materialisation reuses (the one it evicts goes to a lost list); the
 *          host gives both back.
 *
 *          Writer::assign, from any thread (docs/gpu_assign.md): the same fresh buffers and
 *          bitmap, one element at a time, no vote, no wait; the arena exhausted, the value goes
 *          to the append-buffer, which Array::finish() replays (the last append of each
 *          element, unless a direct write came after it).
 */

#include <cstdint> // std::int64_t
#include <cstdlib> // std::abort (Writer::assign / load called on the host)

#include "chunk/device/arena.h"      // ArenaRef
#include "chunk/device/atomics.h"    // backoff, cas_ptr, exchange_ptr, fetch_add, fetch_or, load, load_ptr
#include "chunk/device/chunk_desc.h" // ChunkDesc, Variant, constant_of, folds_to, busy_mark, is_fresh, tag_fresh, untag
#include "chunk/device/warp.h" // ballot, group_by_key, lane_id, global_warp_id, active_lanes, shuffle_ptr, sync_lanes
#include "chunk/internal/host_device.h" // CHUNK_HD
#include "chunk/internal/same_value.h"  // same_value

namespace shrink_array::device
{

/// @brief A part [lo, hi) of chunk c that a block could not write (arena exhausted).
struct Span
{
    std::int64_t c;
    std::int64_t lo;
    std::int64_t hi;
};

/// @brief The device-side counters of the write path, read back by the host.
struct WriteCounters
{
    unsigned long long n_lost;     ///< entries of the lost list (past its capacity: an error)
    unsigned long long n_failed;   ///< entries of the failed-span list (past its capacity: an error)
    unsigned long long cas_lost;   ///< publications that lost their CAS
    unsigned long long n_append;   ///< entries of the append-buffer (past its capacity: writes lost)
    unsigned long long claim_wait; ///< assign() calls that waited for another thread's claim
    unsigned long long claim_spin; ///< waits given up at the bound (the write was appended)
    /// Fresh buffers the phase published, their chunks in Writer::fresh_list (past its capacity:
    /// the settle scans every chunk). finish() settles only when nonzero.
    unsigned long long fresh;
};

/// Polls of a claimed descriptor before an assign() gives up and appends (docs/gpu_assign.md
/// 3.3): far beyond a claim's duration (three device-scope atomics, no pre-fill); what it
/// bounds is a claim held by a lane of the same AMD wavefront on a divergent path, which cannot
/// progress while this lane polls. 2^14 polls (about 1 ms) gave up on 0.3% of the waits on the
/// MI300A, where those atomics are slow: 2^16, the prototype's bound.
inline constexpr unsigned int claim_spin_max = 1U << 16;

/**
 * @brief What a write kernel gets of an array: trivially copyable, passed by value.
 * @tparam T element type, trivially copyable
 * @tparam S chunk size, a power of two, a multiple of 32
 * @tparam V the variant (chunk_desc.h)
 */
template <typename T, unsigned int S, Variant V = Variant::zero_any>
struct Writer
{
    static_assert(S >= 32 && (S & (S - 1)) == 0, "Writer: S is a power of two, 32 at least");

    /// Words of the `written` bitmap per chunk.
    static constexpr unsigned int words = S / 32;

    ChunkDesc<T, V>*   desc = nullptr;
    std::int64_t       n    = 0;
    ArenaRef<T, S>     arena{};
    unsigned*          written         = nullptr; ///< words per chunk, one bit per element
    T**                lost            = nullptr; ///< buffers of lost CAS
    unsigned long long lost_capacity   = 0;
    Span*              failed          = nullptr; ///< spans not written (arena exhausted)
    unsigned long long failed_capacity = 0;
    WriteCounters*     counters        = nullptr;
    T**                spare           = nullptr; ///< a parked buffer per slot (slot: the warp, see park)
    unsigned long long spare_slots     = 0;       ///< a power of two
    std::int64_t*      append_index    = nullptr; ///< append-buffer: element indices
    T*                 append_value    = nullptr; ///< append-buffer: values
    unsigned long long append_capacity = 0;
    std::int64_t*      fresh_list      = nullptr; ///< the chunks of the fresh publications, in order
    unsigned long long fresh_capacity  = 0;
    /// One bit per chunk: an element of it was appended in this phase (set before the entry,
    /// cleared by the replay). Only those chunks append their constant and look the
    /// append-buffer up (assign_device 1., load_device).
    unsigned* appended = nullptr;
    /// zero_or_value: the array's second value (T{} while none is declared), the only value
    /// other than T{} a whole-chunk write folds to (fold_whole); the other variants ignore it.
    T second{};

    /**
     * @brief a(i) = v, from any thread of a write kernel (docs/gpu_assign.md): no
     *        device-scope acquire or release, no wait. A buffer of an earlier kernel: a
     *        store. A compressed chunk: nothing if v is its constant, else a buffer published
     *        empty and tagged fresh by a relaxed CAS. A fresh buffer: the element, then its bit
     *        in `written`. The arena exhausted: the append-buffer, replayed by
     *        Array::finish().
     * @details CHUNK_HD so that a `__host__ __device__` lambda (KOKKOS_LAMBDA) calls it; its
     *          body exists in device code only, a host call aborts. Not a branch inside the
     *          caller's lambda: a lambda whose device body differs from its host one under
     *          KOKKOS_IF_ON_DEVICE failed to launch (Kokkos 5.3, nvcc 13.1: "invalid resource
     *          handle").
     */
    CHUNK_HD void assign(std::int64_t i, T v) const noexcept;

    /// @brief a(i) in a write kernel (for `a(i) += v`): the constant of a compressed chunk, an
    ///        unwritten element of a fresh buffer included. What this thread wrote, or a thread
    ///        of its block before a __syncthreads(), it reads back -- an append included: once
    ///        the arena has run out in the kernel, an element that would read the constant is
    ///        looked up in the append-buffer first, from its end (slow, until finish()).
    [[nodiscard]] CHUNK_HD T load(std::int64_t i) const noexcept;

    /**
     * @brief a(i) = f(i) for every i of [@p lo, @p hi), a part of chunk @p c; called by every
     *        thread of a block of @p BLOCK threads. f: `T operator()(std::int64_t) const`.
     * @details S % BLOCK == 0 and BLOCK % warp_size == 0. Uses shared memory and
     *          __syncthreads: a block calls it with the same (c, lo, hi).
     */
    template <int BLOCK, typename F>
    __device__ void write(std::int64_t c, std::int64_t lo, std::int64_t hi, const F& f) const noexcept;

  private:
    template <int BLOCK>
    [[nodiscard]] __device__ bool
    fold_whole(ChunkDesc<T, V>& d, std::int64_t base, std::int64_t valid_hi, const T* new_vals) const noexcept;
    __device__ void            assign_device(std::int64_t i, T v) const noexcept;
    [[nodiscard]] __device__ T load_device(std::int64_t i) const noexcept;
    template <int BLOCK>
    __device__ void
    write_marked(T* buf, std::int64_t c, std::int64_t lo, std::int64_t hi, const T* new_vals) const noexcept;
    template <int BLOCK>
    __device__ void
    write_plain(T* buf, std::int64_t c, std::int64_t lo, std::int64_t hi, const T* new_vals) const noexcept;
    __device__ T**                spare_slot() const noexcept;
    __device__ T*                 take_buffer(std::int64_t c) const noexcept;
    __device__ void               park(T* buf) const noexcept;
    __device__ void               fail(std::int64_t c, std::int64_t lo, std::int64_t hi) const noexcept;
    __device__ T*                 publish(std::int64_t c, ChunkDesc<T, V>& d, T constant) const noexcept;
    __device__ void               fill_constant(T* buf, T constant) const noexcept;
    __device__ T*                 wait_claim(ChunkDesc<T, V>& d, T* found) const noexcept;
    __device__ T* materialise_grouped(std::int64_t c, ChunkDesc<T, V>& d, T* seen, T constant) const noexcept;
    __device__ void               append(std::int64_t i, T v) const noexcept;
    __device__ void               mark_written(std::int64_t c, unsigned int e) const noexcept;
    __device__ void               note_fresh(std::int64_t c) const noexcept;
    [[nodiscard]] __device__ T    appended_or(std::int64_t i, T otherwise) const noexcept;
    [[nodiscard]] __device__ bool chunk_appended(std::int64_t c) const noexcept;
};

namespace kernels
{

/// @brief a(i) = f(i) for i in [lo, hi): one block per chunk, grid-stride over the chunks.
template <int BLOCK, typename T, unsigned int S, Variant V, typename F>
__global__ void __launch_bounds__(BLOCK) for_each_chunk(Writer<T, S, V> w, std::int64_t lo, std::int64_t hi, F f)
{
    const std::int64_t first = lo / S;
    const std::int64_t last  = (hi + S - 1) / S;
    for (std::int64_t c = first + blockIdx.x; c < last; c += gridDim.x)
    {
        const std::int64_t clo = c * S > lo ? c * S : lo;
        const std::int64_t chi = (c + 1) * S < hi ? (c + 1) * S : hi;
        w.template write<BLOCK>(c, clo, chi, f);
    }
}

/// @brief The spans of @p spans written again (after the arena grew): one block per span.
template <int BLOCK, typename T, unsigned int S, Variant V, typename F>
__global__ void __launch_bounds__(BLOCK) rewrite_spans(Writer<T, S, V> w, const Span* spans, unsigned long long n, F f)
{
    for (unsigned long long k = blockIdx.x; k < n; k += gridDim.x)
    {
        w.template write<BLOCK>(spans[k].c, spans[k].lo, spans[k].hi, f);
    }
}

/// @brief Chunk @p c, if its buffer is fresh: the constant where no bit is set, the bitmap
///        cleared, the tag removed. Every lane of a warp calls it with the same c.
template <typename T, unsigned int S, Variant V>
__device__ void settle_chunk(const Writer<T, S, V>& w, std::int64_t c, int lane)
{
    ChunkDesc<T, V>& d   = w.desc[c];
    T* const         buf = d.buf;
    if (is_filled(buf))
    {
        if (lane == 0)
        {
            d.buf = untag(buf); // every element written: the mark only
        }
        return; // uniform per warp
    }
    if (!is_fresh(buf))
    {
        return; // uniform per warp
    }
    T* const        p        = untag(buf);
    const T         constant = constant_of(d);
    unsigned* const bits     = w.written + c * Writer<T, S, V>::words;
    for (unsigned int e = static_cast<unsigned int>(lane); e < S; e += warp_size)
    {
        if (((bits[e >> 5] >> (e & 31)) & 1U) == 0)
        {
            p[e] = constant;
        }
    }
    sync_lanes(full_warp); // every lane has read the bitmap
    for (unsigned int k = static_cast<unsigned int>(lane); k < Writer<T, S, V>::words; k += warp_size)
    {
        bits[k] = 0;
    }
    if (lane == 0)
    {
        d.buf = p;
    }
}

/// @brief Every fresh buffer of the array, found by a scan of the descriptors: one warp per
///        chunk, grid-stride (the fresh list overflowed, or the count is not known here).
template <typename T, unsigned int S, Variant V>
__global__ void settle_fresh(Writer<T, S, V> w, std::int64_t n_chunks)
{
    const int          lane  = lane_id();
    const std::int64_t warps = static_cast<std::int64_t>(gridDim.x) * (blockDim.x / warp_size);
    const std::int64_t first = (blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x) / warp_size;
    for (std::int64_t c = first; c < n_chunks; c += warps)
    {
        settle_chunk(w, c, lane);
    }
}

/// @brief The fresh buffers of the fresh list's first @p count entries: one warp per entry.
template <typename T, unsigned int S, Variant V>
__global__ void settle_listed(Writer<T, S, V> w, unsigned long long count)
{
    const int                lane  = lane_id();
    const unsigned long long warps = static_cast<unsigned long long>(gridDim.x) * (blockDim.x / warp_size);
    const unsigned long long first =
        (blockIdx.x * static_cast<unsigned long long>(blockDim.x) + threadIdx.x) / warp_size;
    for (unsigned long long k = first; k < count; k += warps)
    {
        settle_chunk(w, w.fresh_list[k], lane);
    }
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        // The list emptied for the next phase, here and not by a memset after the launch (a
        // blocking call of about 13 us on the MI300A, five per phase of kokkos_shrink_bench).
        // No note_fresh() runs in this kernel; the next phase's kernels come after it.
        w.counters->fresh = 0;
    }
}

/// @brief The lost list and the spare slots back to the arena (after DeviceArena::normalise()).
template <typename T, unsigned int S, Variant V>
__global__ void reclaim_lost(Writer<T, S, V> w, unsigned long long n)
{
    const unsigned long long stride = static_cast<unsigned long long>(gridDim.x) * blockDim.x;
    const unsigned long long first  = blockIdx.x * static_cast<unsigned long long>(blockDim.x) + threadIdx.x;
    for (unsigned long long k = first; k < n; k += stride)
    {
        w.arena.give_back(w.lost[k]);
    }
    for (unsigned long long k = first; k < w.spare_slots; k += stride)
    {
        if (w.spare[k] != nullptr)
        {
            w.arena.give_back(w.spare[k]);
            w.spare[k] = nullptr;
        }
    }
}

/// @brief Entries [0, @p n) of the append-buffer back to index -1 (no entry). A template: a
///        kernel defined in a header (a plain __global__ cannot be inline).
template <typename Index>
__global__ void reset_append(Index* index, unsigned long long n)
{
    const unsigned long long stride = static_cast<unsigned long long>(gridDim.x) * blockDim.x;
    for (unsigned long long k = blockIdx.x * static_cast<unsigned long long>(blockDim.x) + threadIdx.x; k < n;
         k += stride)
    {
        index[k] = -1;
    }
}

/// @brief The replay of an append-buffer (docs/gpu_assign.md 3.4), deduplicated by the host
///        (the last append of each element): an entry whose element is marked in a fresh
///        buffer is skipped, a direct write came after it. Before settle_fresh.
template <typename T, unsigned int S, Variant V>
__global__ void replay_appends(Writer<T, S, V> w, const std::int64_t* index, const T* value, unsigned long long n)
{
    const unsigned long long stride = static_cast<unsigned long long>(gridDim.x) * blockDim.x;
    for (unsigned long long k = blockIdx.x * static_cast<unsigned long long>(blockDim.x) + threadIdx.x; k < n;
         k += stride)
    {
        const std::int64_t i   = index[k];
        const std::int64_t c   = i / S;
        const auto         e   = static_cast<unsigned int>(i % S);
        T* const           buf = load_ptr(&w.desc[c].buf);
        if (is_fresh(buf) && ((load(&w.written[c * Writer<T, S, V>::words + (e >> 5)]) >> (e & 31)) & 1U) != 0)
        {
            continue;
        }
        w.assign(i, value[k]);
    }
}

} // namespace kernels

// ---------------------------------------------------------------------------
// Writer

template <typename T, unsigned int S, Variant V>
template <int BLOCK, typename F>
__device__ void Writer<T, S, V>::write(std::int64_t c, std::int64_t lo, std::int64_t hi, const F& f) const noexcept
{
    static_assert(S % BLOCK == 0 && BLOCK % warp_size == 0, "Writer::write: BLOCK divides S, a multiple of the warp");
    constexpr int      per_thread = static_cast<int>(S) / BLOCK; // elements per thread
    __shared__ T*      s_buf;
    ChunkDesc<T, V>&   d    = desc[c];
    const std::int64_t base = c * S;
    const int          t    = static_cast<int>(threadIdx.x);

    __syncthreads(); // every thread has read s_buf of the previous call
    if (t == 0)
    {
        // An assign() claim of the same kernel: wait for its publication, as assign() does.
        T* seen = load_ptr(&d.buf);
        for (unsigned int spins = 0; seen == busy_mark<T>() && spins < claim_spin_max; ++spins)
        {
            backoff();
            seen = load_ptr(&d.buf);
        }
        if (is_filled(seen))
        {
            seen = load_ptr_acquire(&d.buf); // an assign() of this kernel filled it: its pre-fill first
        }
        s_buf = seen;
    }
    __syncthreads();
    T* const buf = s_buf;
    if (buf == busy_mark<T>())
    {
        if (t == 0)
        {
            fail(c, lo, hi); // written again by finish(f)
        }
        return;
    }
    T new_vals[per_thread];
    for (int j = 0; j < per_thread; ++j)
    {
        const std::int64_t i = base + t + j * BLOCK;
        new_vals[j]          = i >= lo && i < hi ? f(i) : constant_of(d);
    }
    if (buf != nullptr)
    {
        if (is_fresh(buf))
        {
            write_marked<BLOCK>(untag(buf), c, lo, hi, new_vals); // another block's, this kernel or unsettled
        }
        else
        {
            write_plain<BLOCK>(untag(buf), c, lo, hi, new_vals); // an earlier kernel's, or filled by an assign()
        }
        return;
    }
    // Compressed: the vote first (the constant onto its own chunk costs nothing).
    bool differs = false;
    for (int j = 0; j < per_thread; ++j)
    {
        differs = differs || !same_value(new_vals[j], constant_of(d));
    }
    if (__syncthreads_or(differs ? 1 : 0) == 0)
    {
        return;
    }
    const std::int64_t valid_hi = base + S < n ? base + S : n;
    const bool         whole    = lo <= base && hi >= valid_hi;
    if constexpr (V != Variant::zero)
    {
        if (whole && fold_whole<BLOCK>(d, base, valid_hi, new_vals))
        {
            return; // the chunk reads its new value, no buffer (docs/gpu_design.md 7.3)
        }
    }
    if (t == 0)
    {
        s_buf = take_buffer(c);
    }
    __syncthreads();
    T* const mine = s_buf;
    if (mine == nullptr)
    {
        if (t == 0)
        {
            fail(c, lo, hi);
        }
        return;
    }
    if (whole)
    {
        for (int j = 0; j < per_thread; ++j)
        {
            mine[t + j * BLOCK] = new_vals[j]; // values, the constant past n
        }
    }
    __syncthreads(); // every thread has read s_buf
    if (t == 0)
    {
        T* const published = whole ? mine : tag_fresh(mine);
        T*       found     = cas_ptr(&d.buf, static_cast<T*>(nullptr), published);
        if (found != nullptr)
        {
            fetch_add(&counters->cas_lost, 1ULL);
            park(mine);
            if (is_filled(found))
            {
                found = load_ptr_acquire(&d.buf); // filled by an assign(): its pre-fill first
            }
        }
        else if (!whole)
        {
            note_fresh(c);
        }
        s_buf = found != nullptr ? found : published;
    }
    __syncthreads();
    T* const winner = s_buf;
    if (winner == mine)
    {
        return; // the whole chunk, written
    }
    if (winner == busy_mark<T>())
    {
        if (t == 0)
        {
            fail(c, lo, hi); // an assign() claimed it meanwhile: written again by finish(f)
        }
        return;
    }
    if (is_fresh(winner))
    {
        write_marked<BLOCK>(untag(winner), c, lo, hi, new_vals);
    }
    else
    {
        write_plain<BLOCK>(untag(winner), c, lo, hi, new_vals); // a whole-chunk writer or a filling assign() overlapped
    }
}

/**
 * @details zero_any and zero_or_value, a compressed chunk the block writes whole (every
 *          element below n): if its new values are one value (zero_or_value: T{} or the second
 *          value, folds_to), the chunk takes it as its constant and no buffer -- the
 *          vote before materialising of docs/gpu_design.md 7.3, to a new value and not only to
 *          the chunk's own. Without it a write of constants by bands (kokkos_shrink_bench's
 *          init, shrink_bench's fill) materialised every chunk it changed, and the arena kept
 *          that peak (MI300A, 5 x 8 GiB: 20 GB of arena for 1.5 GB held after compress()).
 *          The constant is changed under the claim (busy_mark()), as assign() materialises: a
 *          claim or a publication of the same kernel found there instead makes it false, and
 *          the caller materialises as before. Every thread of the block calls it.
 */
template <typename T, unsigned int S, Variant V>
template <int BLOCK>
[[nodiscard]] __device__ bool Writer<T, S, V>::fold_whole(ChunkDesc<T, V>& d,
                                                          std::int64_t     base,
                                                          std::int64_t     valid_hi,
                                                          const T*         new_vals) const noexcept
{
    constexpr int per_thread = static_cast<int>(S) / BLOCK;
    // Raw storage: a __shared__ T may not have a constructor (std::complex); T is trivially copyable.
    __shared__ alignas(T) unsigned char s_value[sizeof(T)];
    __shared__ int                      s_folded;
    T&                                  value = *reinterpret_cast<T*>(s_value);
    const int                           t     = static_cast<int>(threadIdx.x);
    if (t == 0)
    {
        value = new_vals[0]; // element base: whole, so base >= lo
    }
    __syncthreads();
    bool other = false;
    for (int j = 0; j < per_thread; ++j)
    {
        other = other || (base + t + j * BLOCK < valid_hi && !same_value(new_vals[j], value));
    }
    if (__syncthreads_or(other ? 1 : 0) != 0 || !folds_to<V>(value, second))
    {
        return false; // value is shared: the whole block returns
    }
    if (t == 0)
    {
        s_folded = 0;
        if (cas_ptr(&d.buf, static_cast<T*>(nullptr), busy_mark<T>()) == nullptr)
        {
            d.constant = value;
            (void)exchange_ptr(&d.buf, static_cast<T*>(nullptr));
            s_folded = 1;
        }
    }
    __syncthreads();
    return s_folded != 0;
}

/// @details The 32 lanes of a warp, or half a wavefront, write 32 consecutive elements
///          (BLOCK is a multiple of the warp size): one atomic or per 32 elements.
template <typename T, unsigned int S, Variant V>
template <int BLOCK>
__device__ void Writer<T, S, V>::write_marked(T*           buf,
                                              std::int64_t c,
                                              std::int64_t lo,
                                              std::int64_t hi,
                                              const T*     new_vals) const noexcept
{
    constexpr int      per_thread = static_cast<int>(S) / BLOCK;
    const std::int64_t base       = c * S;
    const int          t          = static_cast<int>(threadIdx.x);
    const int          lane       = lane_id();
    // Every lane of the warp is here (the whole block calls it, BLOCK a multiple of the warp
    // size): the full mask, not active_lanes(), which f's divergence before the call could
    // narrow and drop the bits of the lanes it leaves out.
    const LaneMask warp = full_warp;
    for (int j = 0; j < per_thread; ++j)
    {
        const int          e        = t + j * BLOCK;
        const std::int64_t i        = base + e;
        const bool         in_range = i >= lo && i < hi;
        if (in_range)
        {
            buf[e] = new_vals[j];
        }
        const LaneMask m = ballot(warp, in_range);
        if ((lane & 31) == 0)
        {
            const auto bits = static_cast<unsigned>(m >> lane);
            if (bits != 0)
            {
                fetch_or(&written[c * words + static_cast<unsigned>(e >> 5)], bits);
            }
        }
    }
}

template <typename T, unsigned int S, Variant V>
template <int BLOCK>
__device__ void
Writer<T, S, V>::write_plain(T* buf, std::int64_t c, std::int64_t lo, std::int64_t hi, const T* new_vals) const noexcept
{
    constexpr int      per_thread = static_cast<int>(S) / BLOCK;
    const std::int64_t base       = c * S;
    for (int j = 0; j < per_thread; ++j)
    {
        const int e = static_cast<int>(threadIdx.x) + j * BLOCK;
        if (base + e >= lo && base + e < hi)
        {
            buf[e] = new_vals[j];
        }
    }
}

/// @details The warp's slot: the buffer of a lost publication waits there for the warp's next
///          materialisation. Two warps sharing a slot only exchange atomically.
template <typename T, unsigned int S, Variant V>
__device__ T** Writer<T, S, V>::spare_slot() const noexcept
{
    return spare + (global_warp_id() & (spare_slots - 1)); // any block shape: a caller's kernel
}

/// @brief A buffer for chunk @p c: one from the arena, else the warp's spare one once the arena
///        is exhausted; nullptr when neither has one.
/// @details The arena first: the spare slot is almost always empty, and trying it first cost
///          every materialisation one more device-scope atomic (MI300A, 8 GiB: the first
///          chunk-aligned write +13% / +7% at S = 256 / 1024 against 407674ad5). The parked
///          buffers go back to the arena at Array::finish() anyway.
template <typename T, unsigned int S, Variant V>
__device__ T* Writer<T, S, V>::take_buffer(std::int64_t c) const noexcept
{
    T* const mine = arena.take(c);
    return mine != nullptr || spare_slots == 0 ? mine : exchange_ptr(spare_slot(), static_cast<T*>(nullptr));
}

/// @brief The buffer of a lost publication into the warp's spare slot; the one it evicts, if
///        any, to the lost list. Both go back to the arena at Array::finish().
template <typename T, unsigned int S, Variant V>
__device__ void Writer<T, S, V>::park(T* buf) const noexcept
{
    T* const evicted = spare_slots != 0 ? exchange_ptr(spare_slot(), buf) : buf;
    if (evicted == nullptr)
    {
        return;
    }
    const unsigned long long k = fetch_add(&counters->n_lost, 1ULL);
    if (k < lost_capacity)
    {
        lost[k] = evicted;
    }
}

/// @details The claim's winner takes a buffer from the arena (not the spare slot: one atomic
///          less before the publication, which every other writer of the chunk waits for) and
///          publishes it -- or releases the claim when the arena is exhausted. Returns what it
///          published (nullptr: the caller appends).
///
///          HIP: empty, tagged fresh, by a relaxed exchange; every write into it then sets its
///          bit of the `written` bitmap (mark_written, one atomic per word and wavefront).
///
///          CUDA: filled with the constant first (one lane, 16-byte stores) and tagged filled,
///          by a release exchange that the writers acquire: a write into it is a plain store,
///          no bit. The bits were half of a first write by assign() (RTX A1000, device_bench:
///          a grid-stride first write 22.8 ms with them, 11.0 ms without), and CUDA cannot
///          aggregate them per warp (materialise_grouped). The settle only removes the mark.
///          Not on HIP: acquire and release cost the MI300A 300x to 600x a flat write in the
///          prototype's protocol.
template <typename T, unsigned int S, Variant V>
__device__ T* Writer<T, S, V>::publish(std::int64_t c, ChunkDesc<T, V>& d, T constant) const noexcept
{
    T* const mine = arena.take(c);
#if defined(CHUNK_DEVICE_HIP)
    (void)constant; // an empty buffer, its unwritten elements take the constant at the settle
    T* const publ = mine != nullptr ? tag_fresh(mine) : nullptr;
    (void)exchange_ptr(&d.buf, publ);
#else
    if (mine != nullptr)
    {
        // T{} as a literal where the constant is T{} (zero always, zero_any and zero_or_value
        // mostly): stores from the zero register. The same zeros from a register that holds
        // them took a first write by assign() 60 % longer, and 1.0 as a literal as long (RTX
        // A1000, device_bench, S = 1024: 13.8 ms against 22.6 and 21.5 ms).
        if (same_value(constant, T{}))
        {
            fill_constant(mine, T{});
        }
        else
        {
            fill_constant(mine, constant);
        }
    }
    T* const publ = mine != nullptr ? tag_filled(mine) : nullptr;
    (void)exchange_ptr_release(&d.buf, publ); // the pre-fill before the pointer
#endif
    if (publ != nullptr)
    {
        note_fresh(c); // after the publication: no waiter waits for it; the settle removes the mark
    }
    return publ;
}

/// @details One thread, 16-byte stores where a buffer's size allows them (S * sizeof(T) a
///          multiple of 16: every geometry of the library's builds).
template <typename T, unsigned int S, Variant V>
__device__ void Writer<T, S, V>::fill_constant(T* buf, T constant) const noexcept
{
    if constexpr ((static_cast<std::size_t>(S) * sizeof(T)) % 16 == 0 && (sizeof(T) == 4 || sizeof(T) == 8))
    {
        // The 16-byte word built from the constant's bits in registers: through a local array
        // and a memcpy, the fill of a runtime constant cost a first write by assign() 60 % more
        // than T{}'s (zero_any against zero, RTX A1000, S = 1024).
        uint4 word;
        if constexpr (sizeof(T) == 8)
        {
            unsigned long long bits = 0;
            __builtin_memcpy(&bits, &constant, sizeof(bits)); // a scalar: kept in registers (bit_cast crashed nvcc 13.3)
            const auto lo   = static_cast<unsigned int>(bits);
            const auto hi   = static_cast<unsigned int>(bits >> 32);
            word            = make_uint4(lo, hi, lo, hi);
        }
        else
        {
            unsigned int bits = 0;
            __builtin_memcpy(&bits, &constant, sizeof(bits));
            word            = make_uint4(bits, bits, bits, bits);
        }
        constexpr unsigned int per = static_cast<unsigned int>(16 / sizeof(T));
        uint4* const           out = reinterpret_cast<uint4*>(buf);
        for (unsigned int k = 0; k < S / per; ++k)
        {
            out[k] = word;
        }
    }
    else
    {
        for (unsigned int e = 0; e < S; ++e)
        {
            buf[e] = constant;
        }
    }
}

/// @details Polls the descriptor (relaxed loads, a short pause) while another thread's claim
///          holds it, up to claim_spin_max. Returns the chunk's buffer (fresh-tagged, or
///          untagged from a whole-chunk Writer::write), or nullptr: the claim was released
///          (arena exhausted) or the wait given up; the caller appends.
template <typename T, unsigned int S, Variant V>
__device__ T* Writer<T, S, V>::wait_claim(ChunkDesc<T, V>& d, T* found) const noexcept
{
    if (found != busy_mark<T>())
    {
        return found;
    }
    fetch_add(&counters->claim_wait, 1ULL);
    for (unsigned int spins = 0; found == busy_mark<T>(); ++spins)
    {
        if (spins == claim_spin_max)
        {
            fetch_add(&counters->claim_spin, 1ULL);
            return nullptr;
        }
        backoff();
        found = load_ptr(&d.buf);
    }
    return found;
}

/**
 * @details The claim first, the buffer after (docs/gpu_assign.md 3.3).
 *
 *          HIP: aggregated per wavefront. The active lanes that need chunk @p c elect a leader,
 *          and the others get its result by a shuffle (a grid-stride first write: one claim or
 *          one wait per wavefront instead of one per lane). Three phases, as the prototype
 *          learned: every leader tries its claim (a relaxed CAS from nullptr to busy_mark());
 *          the winners publish; only then do the others wait. The lanes of a wavefront run in
 *          lockstep: a leader waiting in the same call as a winner would hold back that
 *          winner's publication, and every writer of its chunk behind it (the MI300A gave up
 *          waits that way). In lockstep the active mask is exactly the lanes of this call.
 *
 *          CUDA: per lane, no warp collective. Under independent thread scheduling (Volta and
 *          later) the lanes that reach assign() need not be converged: the lanes on a
 *          materialised chunk write and go on to their next grid-stride iteration while their
 *          neighbours are still in this call. The grouped version hung there on every run
 *          (RTX A1000, device_assign_validation step 1b: scattered writes into materialised
 *          and compressed chunks mixed): two subsets of one warp, with __activemask() masks
 *          naming each other, waited for each other, one in a __syncwarp, the other in a
 *          shuffle. Per lane, the claim's CAS still elects one writer per chunk, the winner
 *          publishes without waiting for anything, and a waiter polls only the descriptor:
 *          with independent thread scheduling the winner progresses even when the waiter is a
 *          lane of its own warp.
 *
 *          What the grouping saved is kept without a collective: the first active lane claims
 *          first, and the other lanes, in a block after it, load the descriptor again and
 *          claim only a chunk still compressed. On a grid-stride first write the warp makes
 *          one CAS instead of 32 on one address (RTX A1000, device_bench 0.5 GiB: 30 ms per
 *          lane, 22-23 ms with the hint, as with the grouping). active_lanes() is only a hint
 *          there: no sync depends on it, and a wrong one costs a CAS, not a wait.
 *
 *          @p seen: what this lane loaded (nullptr or busy_mark()).
 */
template <typename T, unsigned int S, Variant V>
__device__ T*
Writer<T, S, V>::materialise_grouped(std::int64_t c, ChunkDesc<T, V>& d, T* seen, T constant) const noexcept
{
#if defined(CHUNK_DEVICE_HIP)
    const LaneMask active = active_lanes();
    LaneMask       group  = 0;
    int            leader = 0;
    group_by_key(c, active, group, leader);
    const bool lead  = lane_id() == leader;
    T*         found = seen;
    bool       won   = false;
    if (lead && seen == nullptr)
    {
        found = cas_ptr(&d.buf, static_cast<T*>(nullptr), busy_mark<T>());
        won   = found == nullptr;
    }
    T* got = nullptr;
    if (won)
    {
        got = publish(c, d, constant);
    }
    sync_lanes(active); // every claim of this call published before any lane waits
    if (lead && !won)
    {
        got = wait_claim(d, found);
    }
    return shuffle_ptr(active, got, leader);
#else
    // A hint only, never a collective's mask: the first active lane claims first, the others
    // look again after it (the second block runs once the first has reconverged) and claim
    // only a chunk still compressed -- one CAS per warp on a grid-stride first write instead
    // of one per lane. A wrong hint costs a CAS, never a wait on another lane.
    const bool first = lane_id() == first_lane(active_lanes());
    T*         found = seen;
    bool       won   = false;
    if (first && seen == nullptr)
    {
        found = cas_ptr(&d.buf, static_cast<T*>(nullptr), busy_mark<T>());
        won   = found == nullptr;
    }
    if (!first && seen == nullptr)
    {
        found = load_ptr(&d.buf);
        if (found == nullptr)
        {
            found = cas_ptr(&d.buf, static_cast<T*>(nullptr), busy_mark<T>());
            won   = found == nullptr;
        }
    }
    T* got = nullptr;
    if (won)
    {
        got = publish(c, d, constant);
    }
    if (!won)
    {
        if (found == busy_mark<T>())
        {
            found = load_ptr(&d.buf); // a lane of this warp may have just published it
        }
        got = wait_claim(d, found);
        if (is_filled(got))
        {
            got = load_ptr_acquire(&d.buf); // its publisher's pre-fill before this lane's writes
        }
    }
    return got;
#endif
}

/// @details Chunk @p c, just published fresh, into the fresh list: settle() then visits the
///          listed chunks only, not every chunk of the array (a scan of a million descriptors
///          and its launch, per array and per phase, on the MI300A's 2^30 elements). Past the
///          list's capacity -- the arena's buffers, which a phase cannot publish more of -- the
///          count alone goes on and the settle scans.
template <typename T, unsigned int S, Variant V>
__device__ void Writer<T, S, V>::note_fresh(std::int64_t c) const noexcept
{
    const unsigned long long k = fetch_add(&counters->fresh, 1ULL);
    if (k < fresh_capacity)
    {
        fresh_list[k] = c;
    }
}

/// @details The chunk's `appended` bit first, then the entry: this thread's later assigns and
///          loads of the chunk see the bit (one address), and so do the other threads after a
///          __syncthreads() or a kernel boundary, as for the entry itself.
template <typename T, unsigned int S, Variant V>
__device__ void Writer<T, S, V>::append(std::int64_t i, T v) const noexcept
{
    const std::int64_t c = i / S;
    fetch_or(&appended[c >> 5], 1U << (c & 31));
    const unsigned long long k = fetch_add(&counters->n_append, 1ULL);
    if (k < append_capacity)
    {
        append_index[k] = i;
        append_value[k] = v;
    }
}

template <typename T, unsigned int S, Variant V>
CHUNK_HD void Writer<T, S, V>::assign(std::int64_t i, T v) const noexcept
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    assign_device(i, v);
#else
    (void)i;
    (void)v;
    std::abort(); // a host thread: the write path runs in device kernels only
#endif
}

template <typename T, unsigned int S, Variant V>
CHUNK_HD T Writer<T, S, V>::load(std::int64_t i) const noexcept
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return load_device(i);
#else
    (void)i;
    std::abort(); // a host thread: the write path runs in device kernels only
#endif
}

/// @details Element @p e of fresh chunk @p c written: its bit in the chunk's `written`
///          bitmap, which settle() reads after the phase (the kernel's end orders the two).
///
///          HIP: one atomic per 32-bit word of the bitmap and per wavefront. The lanes that set
///          bits of one word -- a grid-stride rewrite: 64 consecutive elements, two or three
///          words -- elect their lowest lane, which ORs the group's bits in one fetch_or; 32
///          ballots, uniform over the wavefront, give each group its bits exactly. In lockstep
///          the active lanes are exactly this call's (no lane is elsewhere), so these
///          collectives are safe there; at most four groups are formed, the lanes past them
///          (scattered writes: a word each) OR their own bit as before. The MI300A's
///          device-scope atomics cross its XCDs: a window of 10^7 elements rewritten by
///          assign() took 634 us against 50 us flat, one fetch_or per element (rocprofv3,
///          kokkos_shrink_bench, an arena of half the chunks).
///
///          CUDA: one fetch_or per element. The lanes that reach assign() need not be converged
///          under independent thread scheduling, and a collective over __activemask()'s lanes
///          deadlocked there (materialise_grouped).
template <typename T, unsigned int S, Variant V>
__device__ void Writer<T, S, V>::mark_written(std::int64_t c, unsigned int e) const noexcept
{
    const std::int64_t w   = c * words + static_cast<std::int64_t>(e >> 5);
    const unsigned int bit = 1U << (e & 31);
#if defined(CHUNK_DEVICE_HIP)
    const LaneMask active  = active_lanes();
    const int      lane    = lane_id();
    LaneMask       pending = active;
    LaneMask       group   = 0;
    int            leader  = -1;
    for (int round = 0; round < 4 && pending != 0; ++round) // pending: the same on every lane
    {
        const int          l  = first_lane(pending);
        const std::int64_t lw = shuffle(active, w, l);
        const LaneMask     g  = ballot(active, ((pending >> lane) & 1U) != 0 && w == lw);
        if (((g >> lane) & 1U) != 0)
        {
            group  = g;
            leader = l;
        }
        pending &= ~g;
    }
    unsigned int bits = 0;
    for (unsigned int b = 0; b < 32; ++b) // uniform: every active lane takes part
    {
        const LaneMask m = ballot(active, (e & 31) == b);
        bits |= (m & group) != 0 ? 1U << b : 0U;
    }
    if (leader < 0)
    {
        fetch_or(&written[w], bit); // past the four groups: its own bit
    }
    else if (lane == leader)
    {
        fetch_or(&written[w], bits);
    }
#else
    fetch_or(&written[w], bit);
#endif
}

template <typename T, unsigned int S, Variant V>
__device__ void Writer<T, S, V>::assign_device(std::int64_t i, T v) const noexcept
{
    const std::int64_t c   = i / S;
    const auto         e   = static_cast<unsigned int>(i % S);
    ChunkDesc<T, V>&   d   = desc[c];
    T*                 buf = load_ptr(&d.buf);
    if (buf == nullptr || buf == busy_mark<T>())
    {
        const T constant = constant_of(d); // a compressed chunk's: no writer changes it in a phase
        if (same_value(v, constant))
        {
            // 1. The constant onto its chunk: nothing (a claimed chunk still reads it) -- unless
            //    the chunk appended in this phase: an earlier append of this element by this
            //    thread (its own bit, so seen here) would win at the replay. The constant is
            //    appended after it, the last append wins (kokkos_shrink_bench, MI300A:
            //    `a2 = .5; a2 = 1.0` on chunks of 1.0, the arena exhausted, wrong checksums from
            //    run to run). Not the phase's append count: one wait given up made every write of
            //    a constant append, millions on windows of 10^7 elements (GH200: the append-buffer
            //    overflowed, the rewrite 591x flat).
            if (chunk_appended(c))
            {
                append(i, v);
            }
            return;
        }
        buf = materialise_grouped(c, d, buf, constant); // 2.
        if (buf == nullptr)
        {
            append(i, v);
            return;
        }
    }
    if (is_fresh(buf))
    {
        untag(buf)[e] = v; // 3. the element, then its bit
        mark_written(c, e);
        return;
    }
    if (is_filled(buf))
    {
        // 3'. CUDA: filled in this phase, every element valid. Acquired (a relaxed load saw it,
        //     or the claim's winner is this lane: then no load is needed, the reload is cheap).
        //     A chunk that appended in this phase appends again: no bit tells the replay that a
        //     direct write came after an append of the same element, the last append wins.
        if (chunk_appended(c))
        {
            append(i, v);
            return;
        }
        T* const p = untag(load_ptr_acquire(&d.buf));
        p[e]       = v;
        return;
    }
    buf[e] = v; // 4. a buffer of an earlier kernel
}

template <typename T, unsigned int S, Variant V>
[[nodiscard]] __device__ T Writer<T, S, V>::load_device(std::int64_t i) const noexcept
{
    const std::int64_t     c   = i / S;
    const auto             e   = static_cast<unsigned int>(i % S);
    const ChunkDesc<T, V>& d   = desc[c];
    T* const               buf = load_ptr(&d.buf);
    if (buf == nullptr || buf == busy_mark<T>())
    {
        // claimed: no element written yet but appends, only in a chunk that appended
        return chunk_appended(c) ? appended_or(i, constant_of(d)) : constant_of(d);
    }
    if (is_fresh(buf))
    {
        // device::load: the member load(i) hides the atomic load inside the class.
        const bool set = ((device::load(&written[c * words + (e >> 5)]) >> (e & 31)) & 1U) != 0;
        return set ? untag(buf)[e] : chunk_appended(c) ? appended_or(i, constant_of(d)) : constant_of(d);
    }
    if (is_filled(buf))
    {
        // CUDA: filled in this phase (its pre-fill acquired); a chunk that appended keeps its
        // writes in the append-buffer (assign_device 3').
        const T x = untag(load_ptr_acquire(&d.buf))[e];
        return chunk_appended(c) ? appended_or(i, x) : x;
    }
    return buf[e]; // a buffer of an earlier kernel: no append to its chunk in this one
}

/// @details The append-buffer's unused entries hold index -1 (Array resets them after each
///          replay), so an entry being written by another thread reads as no entry or as its
///          new index, never as an earlier phase's.
template <typename T, unsigned int S, Variant V>
[[nodiscard]] __device__ T Writer<T, S, V>::appended_or(std::int64_t i, T otherwise) const noexcept
{
    const unsigned long long n = device::load(&counters->n_append);
    for (unsigned long long k = n < append_capacity ? n : append_capacity; k > 0; --k)
    {
        if (append_index[k - 1] == i)
        {
            return append_value[k - 1];
        }
    }
    return otherwise;
}

/// @brief Whether an element of chunk @p c was appended in this phase (append()).
template <typename T, unsigned int S, Variant V>
[[nodiscard]] __device__ bool Writer<T, S, V>::chunk_appended(std::int64_t c) const noexcept
{
    return ((device::load(&appended[c >> 5]) >> (c & 31)) & 1U) != 0;
}

template <typename T, unsigned int S, Variant V>
__device__ void Writer<T, S, V>::fail(std::int64_t c, std::int64_t lo, std::int64_t hi) const noexcept
{
    const unsigned long long k = fetch_add(&counters->n_failed, 1ULL);
    if (k < failed_capacity)
    {
        failed[k] = Span{c, lo, hi};
    }
}

} // namespace shrink_array::device
