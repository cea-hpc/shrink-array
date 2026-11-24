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

#include "chunk/device/arena.h"        // ArenaRef
#include "chunk/device/atomics.h"      // backoff, cas_ptr, exchange_ptr, fetch_add, fetch_or, load, load_ptr
#include "chunk/device/chunk_desc.h"   // ChunkDesc, busy_mark, is_fresh, tag_fresh, untag
#include "chunk/device/warp.h"         // ballot, lane_id, active_lanes, sync_lanes
#include "chunk/internal/same_value.h" // same_value

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
};

/// Polls of a claimed descriptor before an assign() gives up and appends (docs/gpu_assign.md
/// 3.3): far beyond a claim's duration (two atomics, no pre-fill); what it bounds is a claim
/// held by a lane of the same AMD wavefront on a divergent path, which cannot progress while
/// this lane polls.
inline constexpr unsigned int claim_spin_max = 1U << 14;

/**
 * @brief What a write kernel gets of an array: trivially copyable, passed by value.
 * @tparam T element type, trivially copyable
 * @tparam S chunk size, a power of two, a multiple of 32
 */
template <typename T, unsigned int S>
struct Writer
{
    static_assert(S >= 32 && (S & (S - 1)) == 0, "Writer: S is a power of two, 32 at least");

    /// Words of the `written` bitmap per chunk.
    static constexpr unsigned int words = S / 32;

    ChunkDesc<T>*      desc = nullptr;
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

    /**
     * @brief a(i) = v, from any thread of a write kernel (docs/gpu_assign.md): no
     *        device-scope acquire or release, no wait. A buffer of an earlier kernel: a
     *        store. A compressed chunk: nothing if v is its constant, else a buffer published
     *        empty and tagged fresh by a relaxed CAS. A fresh buffer: the element, then its bit
     *        in `written`. The arena exhausted: the append-buffer, replayed by
     *        Array::finish().
     */
    __device__ void assign(std::int64_t i, T v) const noexcept;

    /// @brief a(i) in a write kernel (for `a(i) += v`): the constant of a compressed chunk, an
    ///        unwritten element of a fresh buffer included. What this thread wrote, or a thread
    ///        of its block before a __syncthreads(), it reads back -- an append included: once
    ///        the arena has run out in the kernel, an element that would read the constant is
    ///        looked up in the append-buffer first, from its end (slow, until finish()).
    [[nodiscard]] __device__ T load(std::int64_t i) const noexcept;

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
    __device__ void
    write_marked(T* buf, std::int64_t c, std::int64_t lo, std::int64_t hi, const T* new_vals) const noexcept;
    template <int BLOCK>
    __device__ void
    write_plain(T* buf, std::int64_t c, std::int64_t lo, std::int64_t hi, const T* new_vals) const noexcept;
    __device__ T**             spare_slot() const noexcept;
    __device__ T*              take_buffer(std::int64_t c) const noexcept;
    __device__ void            park(T* buf) const noexcept;
    __device__ void            fail(std::int64_t c, std::int64_t lo, std::int64_t hi) const noexcept;
    __device__ T*              materialise(std::int64_t c, ChunkDesc<T>& d, T* seen) const noexcept;
    __device__ void            append(std::int64_t i, T v) const noexcept;
    [[nodiscard]] __device__ T appended_or(std::int64_t i, T otherwise) const noexcept;
};

namespace kernels
{

/// @brief a(i) = f(i) for i in [lo, hi): one block per chunk, grid-stride over the chunks.
template <int BLOCK, typename T, unsigned int S, typename F>
__global__ void __launch_bounds__(BLOCK) for_each_chunk(Writer<T, S> w, std::int64_t lo, std::int64_t hi, F f)
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
template <int BLOCK, typename T, unsigned int S, typename F>
__global__ void __launch_bounds__(BLOCK) rewrite_spans(Writer<T, S> w, const Span* spans, unsigned long long n, F f)
{
    for (unsigned long long k = blockIdx.x; k < n; k += gridDim.x)
    {
        w.template write<BLOCK>(spans[k].c, spans[k].lo, spans[k].hi, f);
    }
}

/// @brief Each fresh buffer: the constant where no bit is set, the bitmap cleared, the tag
///        removed. One warp per chunk, grid-stride.
template <typename T, unsigned int S>
__global__ void settle_fresh(Writer<T, S> w, std::int64_t n_chunks)
{
    const int          lane  = lane_id();
    const std::int64_t warps = static_cast<std::int64_t>(gridDim.x) * (blockDim.x / warp_size);
    const std::int64_t first = (blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x) / warp_size;
    for (std::int64_t c = first; c < n_chunks; c += warps)
    {
        ChunkDesc<T>& d   = w.desc[c];
        T* const      buf = d.buf;
        if (!is_fresh(buf))
        {
            continue; // uniform per warp
        }
        T* const        p        = untag(buf);
        const T         constant = d.constant;
        unsigned* const bits     = w.written + c * Writer<T, S>::words;
        for (unsigned int e = static_cast<unsigned int>(lane); e < S; e += warp_size)
        {
            if (((bits[e >> 5] >> (e & 31)) & 1U) == 0)
            {
                p[e] = constant;
            }
        }
        sync_lanes(full_warp); // every lane has read the bitmap
        for (unsigned int k = static_cast<unsigned int>(lane); k < Writer<T, S>::words; k += warp_size)
        {
            bits[k] = 0;
        }
        if (lane == 0)
        {
            d.buf = p;
        }
    }
}

/// @brief The lost list and the spare slots back to the arena (after DeviceArena::normalise()).
template <typename T, unsigned int S>
__global__ void reclaim_lost(Writer<T, S> w, unsigned long long n)
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
template <typename T, unsigned int S>
__global__ void replay_appends(Writer<T, S> w, const std::int64_t* index, const T* value, unsigned long long n)
{
    const unsigned long long stride = static_cast<unsigned long long>(gridDim.x) * blockDim.x;
    for (unsigned long long k = blockIdx.x * static_cast<unsigned long long>(blockDim.x) + threadIdx.x; k < n;
         k += stride)
    {
        const std::int64_t i   = index[k];
        const std::int64_t c   = i / S;
        const auto         e   = static_cast<unsigned int>(i % S);
        T* const           buf = load_ptr(&w.desc[c].buf);
        if (is_fresh(buf) && ((load(&w.written[c * Writer<T, S>::words + (e >> 5)]) >> (e & 31)) & 1U) != 0)
        {
            continue;
        }
        w.assign(i, value[k]);
    }
}

} // namespace kernels

// ---------------------------------------------------------------------------
// Writer

template <typename T, unsigned int S>
template <int BLOCK, typename F>
__device__ void Writer<T, S>::write(std::int64_t c, std::int64_t lo, std::int64_t hi, const F& f) const noexcept
{
    static_assert(S % BLOCK == 0 && BLOCK % warp_size == 0, "Writer::write: BLOCK divides S, a multiple of the warp");
    constexpr int      per_thread = static_cast<int>(S) / BLOCK; // elements per thread
    __shared__ T*      s_buf;
    ChunkDesc<T>&      d    = desc[c];
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
        new_vals[j]          = i >= lo && i < hi ? f(i) : d.constant;
    }
    if (buf != nullptr)
    {
        if (is_fresh(buf))
        {
            write_marked<BLOCK>(untag(buf), c, lo, hi, new_vals); // another block's, this kernel or unsettled
        }
        else
        {
            write_plain<BLOCK>(buf, c, lo, hi, new_vals); // an earlier kernel's
        }
        return;
    }
    // Compressed: the vote first (the constant onto its own chunk costs nothing).
    bool differs = false;
    for (int j = 0; j < per_thread; ++j)
    {
        differs = differs || !same_value(new_vals[j], d.constant);
    }
    if (__syncthreads_or(differs ? 1 : 0) == 0)
    {
        return;
    }
    const std::int64_t valid_hi = base + S < n ? base + S : n;
    const bool         whole    = lo <= base && hi >= valid_hi;
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
        T* const found     = cas_ptr(&d.buf, static_cast<T*>(nullptr), published);
        if (found != nullptr)
        {
            fetch_add(&counters->cas_lost, 1ULL);
            park(mine);
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
        write_plain<BLOCK>(winner, c, lo, hi, new_vals); // a whole-chunk writer overlapped: outside the contract
    }
}

/// @details The 32 lanes of a warp, or half a wavefront, write 32 consecutive elements
///          (BLOCK is a multiple of the warp size): one atomic or per 32 elements.
template <typename T, unsigned int S>
template <int BLOCK>
__device__ void
Writer<T, S>::write_marked(T* buf, std::int64_t c, std::int64_t lo, std::int64_t hi, const T* new_vals) const noexcept
{
    constexpr int      per_thread = static_cast<int>(S) / BLOCK;
    const std::int64_t base       = c * S;
    const int          t          = static_cast<int>(threadIdx.x);
    const int          lane       = lane_id();
    const LaneMask     warp       = active_lanes();
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

template <typename T, unsigned int S>
template <int BLOCK>
__device__ void
Writer<T, S>::write_plain(T* buf, std::int64_t c, std::int64_t lo, std::int64_t hi, const T* new_vals) const noexcept
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
template <typename T, unsigned int S>
__device__ T** Writer<T, S>::spare_slot() const noexcept
{
    const unsigned long long thread = blockIdx.x * static_cast<unsigned long long>(blockDim.x) + threadIdx.x;
    return spare + ((thread / warp_size) & (spare_slots - 1));
}

/// @brief A buffer for chunk @p c: the warp's spare one, else one from the arena; nullptr when
///        the arena is exhausted.
template <typename T, unsigned int S>
__device__ T* Writer<T, S>::take_buffer(std::int64_t c) const noexcept
{
    T* const mine = spare_slots != 0 ? exchange_ptr(spare_slot(), static_cast<T*>(nullptr)) : nullptr;
    return mine != nullptr ? mine : arena.take(c);
}

/// @brief The buffer of a lost publication into the warp's spare slot; the one it evicts, if
///        any, to the lost list. Both go back to the arena at Array::finish().
template <typename T, unsigned int S>
__device__ void Writer<T, S>::park(T* buf) const noexcept
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

/// @details The claim first, the buffer after (docs/gpu_assign.md 3.3): a relaxed CAS from
///          nullptr to busy_mark(); its winner alone takes a buffer and publishes it empty,
///          tagged fresh (a relaxed exchange), or releases the claim when the arena is
///          exhausted. No pre-fill: every writer marks its elements. The others poll the
///          descriptor (relaxed) until the publication, up to claim_spin_max. @p seen: what the
///          caller loaded (nullptr or busy_mark()). Returns the chunk's buffer (fresh-tagged,
///          or untagged from a whole-chunk Writer::write), or nullptr: the caller appends.
template <typename T, unsigned int S>
__device__ T* Writer<T, S>::materialise(std::int64_t c, ChunkDesc<T>& d, T* seen) const noexcept
{
    T* const busy  = busy_mark<T>();
    T*       found = seen;
    if (seen == nullptr)
    {
        found = cas_ptr(&d.buf, static_cast<T*>(nullptr), busy);
        if (found == nullptr)
        {
            T* const mine = take_buffer(c);
            T* const publ = mine != nullptr ? tag_fresh(mine) : nullptr;
            (void)exchange_ptr(&d.buf, publ); // publish, or release the claim (arena exhausted)
            return publ;
        }
    }
    if (found == busy)
    {
        fetch_add(&counters->claim_wait, 1ULL);
        for (unsigned int spins = 0; found == busy; ++spins)
        {
            if (spins == claim_spin_max)
            {
                fetch_add(&counters->claim_spin, 1ULL);
                return nullptr;
            }
            backoff();
            found = load_ptr(&d.buf);
        }
    }
    return found; // nullptr: the claim was released, the arena is exhausted
}

template <typename T, unsigned int S>
__device__ void Writer<T, S>::append(std::int64_t i, T v) const noexcept
{
    const unsigned long long k = fetch_add(&counters->n_append, 1ULL);
    if (k < append_capacity)
    {
        append_index[k] = i;
        append_value[k] = v;
    }
}

template <typename T, unsigned int S>
__device__ void Writer<T, S>::assign(std::int64_t i, T v) const noexcept
{
    const std::int64_t c   = i / S;
    const auto         e   = static_cast<unsigned int>(i % S);
    ChunkDesc<T>&      d   = desc[c];
    T*                 buf = load_ptr(&d.buf);
    if (buf == nullptr || buf == busy_mark<T>())
    {
        if (same_value(v, d.constant))
        {
            return; // 1. the constant onto its chunk: nothing (a claimed chunk still reads it)
        }
        buf = materialise(c, d, buf); // 2.
        if (buf == nullptr)
        {
            append(i, v);
            return;
        }
    }
    if (is_fresh(buf))
    {
        untag(buf)[e] = v; // 3. the element, then its bit
        fetch_or(&written[c * words + (e >> 5)], 1U << (e & 31));
        return;
    }
    buf[e] = v; // 4. a buffer of an earlier kernel
}

template <typename T, unsigned int S>
[[nodiscard]] __device__ T Writer<T, S>::load(std::int64_t i) const noexcept
{
    const std::int64_t  c   = i / S;
    const auto          e   = static_cast<unsigned int>(i % S);
    const ChunkDesc<T>& d   = desc[c];
    T* const            buf = load_ptr(&d.buf);
    if (buf == nullptr || buf == busy_mark<T>())
    {
        return appended_or(i, d.constant); // claimed: no element written yet but appends
    }
    if (is_fresh(buf))
    {
        // device::load: the member load(i) hides the atomic load inside the class.
        const bool set = ((device::load(&written[c * words + (e >> 5)]) >> (e & 31)) & 1U) != 0;
        return set ? untag(buf)[e] : appended_or(i, d.constant);
    }
    return buf[e]; // a buffer of an earlier kernel: no append to its chunk in this one
}

/// @details The append-buffer's unused entries hold index -1 (Array resets them after each
///          replay), so an entry being written by another thread reads as no entry or as its
///          new index, never as an earlier phase's.
template <typename T, unsigned int S>
[[nodiscard]] __device__ T Writer<T, S>::appended_or(std::int64_t i, T otherwise) const noexcept
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

template <typename T, unsigned int S>
__device__ void Writer<T, S>::fail(std::int64_t c, std::int64_t lo, std::int64_t hi) const noexcept
{
    const unsigned long long k = fetch_add(&counters->n_failed, 1ULL);
    if (k < failed_capacity)
    {
        failed[k] = Span{c, lo, hi};
    }
}

} // namespace shrink_array::device
