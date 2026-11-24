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
 *          one compressed chunk) parks the loser's buffer in a list the host gives back.
 */

#include <cstdint> // std::int64_t

#include "chunk/device/arena.h"        // ArenaRef
#include "chunk/device/atomics.h"      // cas_ptr, fetch_add, fetch_or, load_ptr
#include "chunk/device/chunk_desc.h"   // ChunkDesc, is_fresh, tag_fresh, untag
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
    unsigned long long n_lost;   ///< entries of the lost list (past its capacity: an error)
    unsigned long long n_failed; ///< entries of the failed-span list (past its capacity: an error)
    unsigned long long cas_lost; ///< publications that lost their CAS
};

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
    write_marked(T* buf, std::int64_t c, std::int64_t lo, std::int64_t hi, const T* vals) const noexcept;
    template <int BLOCK>
    __device__ void write_plain(T* buf, std::int64_t c, std::int64_t lo, std::int64_t hi, const T* vals) const noexcept;
    __device__ void park(T* buf) const noexcept;
    __device__ void fail(std::int64_t c, std::int64_t lo, std::int64_t hi) const noexcept;
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
        sync_lanes(active_lanes()); // every lane has read the bitmap
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

/// @brief The lost list back to the arena (after DeviceArena::normalise()).
template <typename T, unsigned int S>
__global__ void reclaim_lost(Writer<T, S> w, unsigned long long n)
{
    const unsigned long long stride = static_cast<unsigned long long>(gridDim.x) * blockDim.x;
    for (unsigned long long k = blockIdx.x * static_cast<unsigned long long>(blockDim.x) + threadIdx.x; k < n;
         k += stride)
    {
        w.arena.give_back(w.lost[k]);
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
        s_buf = load_ptr(&d.buf);
    }
    __syncthreads();
    T* const buf = s_buf;
    T        vals[per_thread];
    for (int j = 0; j < per_thread; ++j)
    {
        const std::int64_t i = base + t + j * BLOCK;
        vals[j]              = i >= lo && i < hi ? f(i) : d.constant;
    }
    if (buf != nullptr)
    {
        if (is_fresh(buf))
        {
            write_marked<BLOCK>(untag(buf), c, lo, hi, vals); // another block's, this kernel or unsettled
        }
        else
        {
            write_plain<BLOCK>(buf, c, lo, hi, vals); // an earlier kernel's
        }
        return;
    }
    // Compressed: the vote first (the constant onto its own chunk costs nothing).
    bool differs = false;
    for (int j = 0; j < per_thread; ++j)
    {
        differs = differs || !same_value(vals[j], d.constant);
    }
    if (__syncthreads_or(differs ? 1 : 0) == 0)
    {
        return;
    }
    const std::int64_t valid_hi = base + S < n ? base + S : n;
    const bool         whole    = lo <= base && hi >= valid_hi;
    if (t == 0)
    {
        s_buf = arena.take(c);
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
            mine[t + j * BLOCK] = vals[j]; // values, the constant past n
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
    if (is_fresh(winner))
    {
        write_marked<BLOCK>(untag(winner), c, lo, hi, vals);
    }
    else
    {
        write_plain<BLOCK>(winner, c, lo, hi, vals); // a whole-chunk writer overlapped: outside the contract
    }
}

/// @details The 32 lanes of a warp, or half a wavefront, write 32 consecutive elements
///          (BLOCK is a multiple of the warp size): one atomic or per 32 elements.
template <typename T, unsigned int S>
template <int BLOCK>
__device__ void
Writer<T, S>::write_marked(T* buf, std::int64_t c, std::int64_t lo, std::int64_t hi, const T* vals) const noexcept
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
            buf[e] = vals[j];
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
Writer<T, S>::write_plain(T* buf, std::int64_t c, std::int64_t lo, std::int64_t hi, const T* vals) const noexcept
{
    constexpr int      per_thread = static_cast<int>(S) / BLOCK;
    const std::int64_t base       = c * S;
    for (int j = 0; j < per_thread; ++j)
    {
        const int e = static_cast<int>(threadIdx.x) + j * BLOCK;
        if (base + e >= lo && base + e < hi)
        {
            buf[e] = vals[j];
        }
    }
}

template <typename T, unsigned int S>
__device__ void Writer<T, S>::park(T* buf) const noexcept
{
    const unsigned long long k = fetch_add(&counters->n_lost, 1ULL);
    if (k < lost_capacity)
    {
        lost[k] = buf;
    }
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
