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
 * @file array.h
 * @brief device::Array, the host side of a device zero_any array so far: its descriptors,
 *        its arena, and the write phases of the chunk-aligned path (block_write.h).
 *
 * @details Stage 1, experimental (see chunk_desc.h). What it does now:
 *          - view(): the read path (view.h), for read kernels, between write phases.
 *          - parallel_for_chunks(lo, hi, f): a(i) = f(i) for i in [lo, hi), one block per
 *            chunk, then finish(f).
 *          - writer() and finish(f): for a kernel of the caller's that calls Writer::write
 *            itself (several spans per kernel, the same f in every call), then finish(f):
 *            fresh buffers settled, spans the arena could not serve written again after it
 *            grew, lost buffers given back.
 *          - compress(): the fold-back pass (compress.h), between phases.
 *          Next: the host <-> device conversion.
 */

#include <cstddef>     // std::size_t
#include <cstdint>     // std::int64_t
#include <stdexcept>   // std::runtime_error
#include <string>      // std::to_string
#include <type_traits> // std::is_trivially_copyable_v
#include <vector>      // std::vector

#include "chunk/device/arena.h"       // DeviceArena
#include "chunk/device/backend.h"     // allocate, release, zero, copy, synchronize
#include "chunk/device/block_write.h" // Writer, Span, WriteCounters, kernels
#include "chunk/device/chunk_desc.h"  // ChunkDesc
#include "chunk/device/compress.h"    // kernels::compress_chunks
#include "chunk/device/view.h"        // View

namespace shrink_array::device
{

namespace kernels
{

/// @brief Every descriptor {nullptr, @p constant}.
template <typename T>
__global__ void init_desc(ChunkDesc<T>* desc, std::int64_t n_chunks, T constant)
{
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t c = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x; c < n_chunks; c += stride)
    {
        desc[c].buf      = nullptr;
        desc[c].constant = constant;
    }
}

/// @brief Every element into a flat buffer.
template <typename T, unsigned int S>
__global__ void expand(View<T, S> v, T* out)
{
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x; i < v.size(); i += stride)
    {
        out[i] = v(i);
    }
}

/// @brief Chunks holding a buffer.
template <typename T>
__global__ void count_materialised(const ChunkDesc<T>* desc, std::int64_t n_chunks, unsigned long long* out)
{
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    unsigned long long mine   = 0;
    for (std::int64_t c = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x; c < n_chunks; c += stride)
    {
        mine += desc[c].buf != nullptr ? 1 : 0;
    }
    fetch_add(out, mine);
}

} // namespace kernels

/**
 * @brief A device array of @p T in chunks of @p S, every element @p constant at first.
 * @tparam T element type, trivially copyable
 * @tparam S chunk size, a power of two, 32 at least (S = 1024 on GPU, docs/gpu_design.md 13)
 */
template <typename T, unsigned int S>
class Array
{
    static_assert(std::is_trivially_copyable_v<T>, "device::Array: trivially copyable elements");

  public:
    /// @brief @p n elements reading @p constant; an arena of @p arena_buffers buffers to start.
    Array(std::int64_t n, unsigned long long arena_buffers, T constant = T{});
    ~Array();
    Array(const Array&)            = delete;
    Array& operator=(const Array&) = delete;
    Array(Array&&)                 = delete;
    Array& operator=(Array&&)      = delete;

    /// @brief For a read kernel, between write phases.
    [[nodiscard]] View<T, S> view() const noexcept
    {
        return View<T, S>{desc_, n_};
    }

    /// @brief For a write kernel of the caller's that calls Writer::write; finish(f) after it.
    [[nodiscard]] Writer<T, S> writer() const noexcept;

    /// @brief a(i) = f(i) for i in [@p lo, @p hi) by the chunk-aligned loop, then finish(f).
    ///        f: `T operator()(std::int64_t) const`, a function of i.
    template <int BLOCK = 256, typename F>
    void parallel_for_chunks(std::int64_t lo, std::int64_t hi, const F& f, unsigned int grid = 4096);

    /// @brief The end of a write phase whose kernels called Writer::write with @p f: settles
    ///        the fresh buffers, writes the spans the arena could not serve again after
    ///        growing it (as often as needed), gives the lost buffers back. Waits.
    template <int BLOCK = 256, typename F>
    void finish(const F& f);

    /// @brief The compression pass (section 7), between phases: every materialised chunk
    ///        whose elements below size() are one value folds back to it, its buffer back to
    ///        the arena. Returns the chunks folded back. Waits.
    unsigned long long compress();

    /// @brief Elements.
    [[nodiscard]] std::int64_t size() const noexcept
    {
        return n_;
    }

    /// @brief Chunks.
    [[nodiscard]] std::int64_t n_chunks() const noexcept
    {
        return n_chunks_;
    }

    /// @brief Chunks holding a buffer (waits).
    [[nodiscard]] unsigned long long materialised() const;

    /// @brief Publications that lost their CAS since the array was made.
    [[nodiscard]] unsigned long long cas_lost() const;

    /// @brief Every element, on the host (waits).
    [[nodiscard]] std::vector<T> to_host() const;

    /// @brief The arena.
    [[nodiscard]] DeviceArena<T, S>& arena() noexcept
    {
        return arena_;
    }

  private:
    void settle();
    void reclaim_lost();

    std::int64_t       n_;
    std::int64_t       n_chunks_;
    ChunkDesc<T>*      desc_     = nullptr;
    unsigned*          written_  = nullptr;
    Span*              failed_   = nullptr;
    T**                lost_     = nullptr;
    WriteCounters*     counters_ = nullptr;
    unsigned long long failed_capacity_;
    unsigned long long lost_capacity_;
    DeviceArena<T, S>  arena_;
};

// ---------------------------------------------------------------------------

/// @details The failed-span list holds two spans per chunk (a chunk-aligned loop fails once
///          per chunk at most; a caller's kernel may have more spans per chunk, past that the
///          phase throws). The lost list holds one buffer per chunk: a lost CAS needs two
///          blocks on one compressed chunk.
template <typename T, unsigned int S>
Array<T, S>::Array(std::int64_t n, unsigned long long arena_buffers, T constant)
    : n_(n)
    , n_chunks_((n + S - 1) / S)
    , failed_capacity_(2 * static_cast<unsigned long long>(n_chunks_) + 1)
    , lost_capacity_(static_cast<unsigned long long>(n_chunks_) + 1)
    , arena_(arena_buffers)
{
    const auto nc = static_cast<std::size_t>(n_chunks_);
    desc_         = allocate<ChunkDesc<T>>(nc);
    written_      = allocate<unsigned>(nc * Writer<T, S>::words);
    failed_       = allocate<Span>(static_cast<std::size_t>(failed_capacity_));
    lost_         = allocate<T*>(static_cast<std::size_t>(lost_capacity_));
    counters_     = allocate<WriteCounters>(1);
    zero(written_, nc * Writer<T, S>::words);
    zero(counters_, 1);
    if (n_chunks_ > 0)
    {
        kernels::init_desc<<<256, 256>>>(desc_, n_chunks_, constant);
    }
    synchronize();
}

template <typename T, unsigned int S>
Array<T, S>::~Array()
{
    release(desc_);
    release(written_);
    release(failed_);
    release(lost_);
    release(counters_);
}

template <typename T, unsigned int S>
[[nodiscard]] Writer<T, S> Array<T, S>::writer() const noexcept
{
    Writer<T, S> w;
    w.desc            = desc_;
    w.n               = n_;
    w.arena           = arena_.ref();
    w.written         = written_;
    w.lost            = lost_;
    w.lost_capacity   = lost_capacity_;
    w.failed          = failed_;
    w.failed_capacity = failed_capacity_;
    w.counters        = counters_;
    return w;
}

template <typename T, unsigned int S>
template <int BLOCK, typename F>
void Array<T, S>::parallel_for_chunks(std::int64_t lo, std::int64_t hi, const F& f, unsigned int grid)
{
    if (hi > lo)
    {
        const std::int64_t chunks = (hi + S - 1) / S - lo / S;
        const auto         blocks = static_cast<unsigned int>(chunks < grid ? chunks : grid);
        kernels::for_each_chunk<BLOCK><<<blocks, BLOCK>>>(writer(), lo, hi, f);
    }
    finish<BLOCK>(f);
}

/// @details A failed span's chunk was compressed and needed a buffer: at most one buffer per
///          failed span, plus a quarter for the CAS the second pass may lose. The spans are
///          read from a copy: the second pass may fail again into the emptied list.
template <typename T, unsigned int S>
template <int BLOCK, typename F>
void Array<T, S>::finish(const F& f)
{
    synchronize();
    settle();
    for (;;)
    {
        WriteCounters c{};
        copy(&c, counters_, 1);
        if (c.n_failed > failed_capacity_)
        {
            throw std::runtime_error("shrink_array::device::Array::finish: " + std::to_string(c.n_failed) +
                                     " failed spans, past the capacity of the list");
        }
        if (c.n_failed == 0)
        {
            break;
        }
        const unsigned long long n_spans = c.n_failed;
        reclaim_lost();
        arena_.grow(n_spans + n_spans / 4 + 1);
        Span* spans = allocate<Span>(static_cast<std::size_t>(n_spans));
        copy(spans, failed_, static_cast<std::size_t>(n_spans));
        copy(&c, counters_, 1);
        c.n_failed = 0;
        copy(counters_, &c, 1);
        const auto blocks = static_cast<unsigned int>(n_spans < 4096 ? n_spans : 4096);
        kernels::rewrite_spans<BLOCK><<<blocks, BLOCK>>>(writer(), spans, n_spans, f);
        synchronize();
        release(spans);
        settle();
    }
    reclaim_lost();
}

template <typename T, unsigned int S>
void Array<T, S>::settle()
{
    if (n_chunks_ > 0)
    {
        const std::int64_t warps_per_block = 256 / warp_size;
        const std::int64_t blocks          = (n_chunks_ + warps_per_block - 1) / warps_per_block;
        kernels::settle_fresh<<<static_cast<unsigned int>(blocks < 65536 ? blocks : 65536), 256>>>(writer(), n_chunks_);
    }
    synchronize();
}

template <typename T, unsigned int S>
void Array<T, S>::reclaim_lost()
{
    WriteCounters c{};
    copy(&c, counters_, 1);
    if (c.n_lost > lost_capacity_)
    {
        throw std::runtime_error("shrink_array::device::Array: " + std::to_string(c.n_lost) +
                                 " lost buffers, past the capacity of the list");
    }
    if (c.n_lost > 0)
    {
        arena_.normalise();
        kernels::reclaim_lost<<<64, 256>>>(writer(), c.n_lost);
        synchronize();
        c.n_lost = 0;
        copy(counters_, &c, 1);
    }
}

/// @details One warp per chunk (the prototype: 85% to 102% of the stream read at S = 1024 on
///          the GH200 and the MI300A when every chunk folds back), at most 65536 blocks of 256
///          threads, grid-stride beyond.
template <typename T, unsigned int S>
unsigned long long Array<T, S>::compress()
{
    synchronize();
    settle(); // a fresh buffer left by a kernel without finish() would fold on unwritten elements
    unsigned long long* d = allocate<unsigned long long>(1);
    zero(d, 1);
    arena_.normalise();
    if (n_chunks_ > 0)
    {
        const std::int64_t warps_per_block = 256 / warp_size;
        const std::int64_t blocks          = (n_chunks_ + warps_per_block - 1) / warps_per_block;
        kernels::compress_chunks<T, S>
            <<<static_cast<unsigned int>(blocks < 65536 ? blocks : 65536), 256>>>(desc_, n_, arena_.ref(), d);
    }
    synchronize();
    unsigned long long folded = 0;
    copy(&folded, d, 1);
    release(d);
    return folded;
}

template <typename T, unsigned int S>
[[nodiscard]] unsigned long long Array<T, S>::materialised() const
{
    unsigned long long* d = allocate<unsigned long long>(1);
    zero(d, 1);
    kernels::count_materialised<<<256, 256>>>(desc_, n_chunks_, d);
    synchronize();
    unsigned long long h = 0;
    copy(&h, d, 1);
    release(d);
    return h;
}

template <typename T, unsigned int S>
[[nodiscard]] unsigned long long Array<T, S>::cas_lost() const
{
    WriteCounters c{};
    copy(&c, counters_, 1);
    return c.cas_lost;
}

template <typename T, unsigned int S>
[[nodiscard]] std::vector<T> Array<T, S>::to_host() const
{
    std::vector<T> out(static_cast<std::size_t>(n_));
    T*             d = allocate<T>(out.size());
    if (n_ > 0)
    {
        kernels::expand<<<1024, 256>>>(view(), d);
    }
    synchronize();
    copy(out.data(), d, out.size());
    release(d);
    return out;
}

} // namespace shrink_array::device
