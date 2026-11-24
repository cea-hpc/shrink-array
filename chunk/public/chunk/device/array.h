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
 *          - assign(state) and image(): the host <-> device conversion, through the
 *            ArrayState of chunk_exchange.h (chunk/internal/chunk_state.h). A program calls
 *            export_array() / import_array() on its host array in a C++ translation unit
 *            and assign() / image() in its device one: nvcc cannot compile the host arrays'
 *            headers (docs/gpu_design.md, section 14). Only the materialised chunks move,
 *            in batches of 64 MiB; a compressed chunk is its descriptor.
 */

#include <cstddef>     // std::size_t
#include <cstdint>     // std::int64_t
#include <memory>      // std::unique_ptr, std::make_unique_for_overwrite
#include <stdexcept>   // std::runtime_error
#include <string>      // std::to_string
#include <type_traits> // std::is_trivially_copyable_v
#include <vector>      // std::vector

#include "chunk/device/arena.h"         // DeviceArena
#include "chunk/device/backend.h"       // allocate, release, zero, copy, synchronize
#include "chunk/device/block_write.h"   // Writer, Span, WriteCounters, kernels
#include "chunk/device/chunk_desc.h"    // ChunkDesc
#include "chunk/device/compress.h"      // kernels::compress_chunks
#include "chunk/device/view.h"          // View
#include "chunk/internal/chunk_state.h" // ArrayState, ArrayImage, ChunkKind, detail::motif_element
#include "chunk/internal/same_value.h"  // same_value

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

/// @brief Every buffer back to the arena (after normalise()), every descriptor compressed.
template <typename T, unsigned int S>
__global__ void release_all(ChunkDesc<T>* desc, std::int64_t n_chunks, ArenaRef<T, S> arena)
{
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t c = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x; c < n_chunks; c += stride)
    {
        if (desc[c].buf != nullptr)
        {
            arena.give_back(untag(desc[c].buf));
            desc[c].buf = nullptr;
        }
    }
}

/// @brief Entry k: a buffer for chunk chunks[k], filled with staging[k * S, (k + 1) * S),
///        published. One block per entry, grid-stride; @p fail set if the arena runs out.
template <typename T, unsigned int S>
__global__ void upload_chunks(ChunkDesc<T>*       desc,
                              ArenaRef<T, S>      arena,
                              const std::int64_t* chunks,
                              const T*            staging,
                              unsigned long long  count,
                              unsigned long long* fail)
{
    __shared__ T* buf;
    for (unsigned long long k = blockIdx.x; k < count; k += gridDim.x)
    {
        __syncthreads(); // every thread has read buf of the previous entry
        if (threadIdx.x == 0)
        {
            buf = arena.take(chunks[k]);
        }
        __syncthreads();
        if (buf == nullptr)
        {
            if (threadIdx.x == 0)
            {
                fetch_add(fail, 1ULL);
            }
            continue;
        }
        for (unsigned int e = threadIdx.x; e < S; e += blockDim.x)
        {
            buf[e] = staging[k * S + e];
        }
        if (threadIdx.x == 0)
        {
            desc[chunks[k]].buf = buf;
        }
    }
}

/// @brief Entry k: the buffer of chunk chunks[k] into staging[k * S, (k + 1) * S).
template <typename T, unsigned int S>
__global__ void
gather_chunks(const ChunkDesc<T>* desc, const std::int64_t* chunks, T* staging, unsigned long long count)
{
    for (unsigned long long k = blockIdx.x; k < count; k += gridDim.x)
    {
        const T* buf = desc[chunks[k]].buf;
        for (unsigned int e = threadIdx.x; e < S; e += blockDim.x)
        {
            staging[k * S + e] = buf[e];
        }
    }
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
    /// Threads per block of the write kernels: 256, or S below it.
    static constexpr int default_block = static_cast<int>(S < 256 ? S : 256);

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
    template <int BLOCK = default_block, typename F>
    void parallel_for_chunks(std::int64_t lo, std::int64_t hi, const F& f, unsigned int grid = 4096);

    /// @brief The end of a write phase whose kernels called Writer::write with @p f: settles
    ///        the fresh buffers, writes the spans the arena could not serve again after
    ///        growing it (as often as needed), gives the lost buffers back. Waits.
    template <int BLOCK = default_block, typename F>
    void finish(const F& f);

    /// @brief The compression pass (section 7), between phases: every materialised chunk
    ///        whose elements below size() are one value folds back to it, its buffer back to
    ///        the arena. Returns the chunks folded back. Waits.
    unsigned long long compress();

    /// @brief Makes this array the one @p state describes (export_array() of a host array
    ///        of size() elements at chunk size S): what it held goes back to the arena, the
    ///        arena grows if the materialised chunks need it. Between phases. Waits.
    /// @throws std::runtime_error if @p state has another size or chunk size.
    void assign(const ArrayState<T>& state);

    /// @brief This array as an ArrayState that owns the materialised chunks' elements, for
    ///        import_array() on a host array. A compressed chunk is first_value if its
    ///        constant is T{}, second_value otherwise. Between phases. Waits.
    [[nodiscard]] ArrayImage<T> image() const;

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
    void                      settle();
    void                      reclaim_lost();
    [[nodiscard]] std::size_t batch_chunks() const noexcept
    {
        constexpr std::size_t batch_bytes = std::size_t{64} << 20;
        return batch_bytes / (S * sizeof(T)) > 0 ? batch_bytes / (S * sizeof(T)) : 1;
    }

    std::int64_t   n_;
    std::int64_t   n_chunks_;
    ChunkDesc<T>*  desc_     = nullptr;
    unsigned*      written_  = nullptr;
    Span*          failed_   = nullptr;
    T**            lost_     = nullptr;
    WriteCounters* counters_ = nullptr;
    /// One counter for compress() and materialised(), allocated once instead of a device
    /// allocation and free per call.
    unsigned long long* scratch_ = nullptr;
    unsigned long long  failed_capacity_;
    unsigned long long  lost_capacity_;
    DeviceArena<T, S>   arena_;
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
    scratch_      = allocate<unsigned long long>(1);
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
    release(scratch_);
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
    unsigned long long* d = scratch_;
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
    return folded;
}

/// @details The materialised and motif chunks are filled on the host into a staging buffer
///          (a motif expanded), copied in batches, and each batch's chunks take a buffer from
///          the arena on the device; the tail of the last chunk holds T{}.
template <typename T, unsigned int S>
void Array<T, S>::assign(const ArrayState<T>& state)
{
    const auto nc = static_cast<std::size_t>(n_chunks_);
    if (state.chunk_size != S || state.size != static_cast<std::size_t>(n_) || state.chunks.size() < nc)
    {
        throw std::runtime_error("shrink_array::device::Array::assign: a state of " + std::to_string(state.size) +
                                 " elements at chunk size " + std::to_string(state.chunk_size) + " for an array of " +
                                 std::to_string(n_) + " at " + std::to_string(S));
    }
    synchronize();
    settle();
    arena_.normalise();
    if (n_chunks_ > 0)
    {
        kernels::release_all<T, S><<<256, 256>>>(desc_, n_chunks_, arena_.ref());
    }
    synchronize();
    std::vector<ChunkDesc<T>> h(nc);
    std::vector<std::int64_t> mats;
    for (std::size_t c = 0; c < nc; ++c)
    {
        const ChunkState<T>& cs       = state.chunks[c];
        const bool           constant = cs.kind == ChunkKind::first_value || cs.kind == ChunkKind::second_value;
        h[c]                          = ChunkDesc<T>{nullptr, constant ? cs.value : T{}};
        if (!constant)
        {
            mats.push_back(static_cast<std::int64_t>(c));
        }
    }
    copy(desc_, h.data(), nc);
    if (arena_.buffers() < mats.size())
    {
        arena_.grow(mats.size() - arena_.buffers()); // every buffer is free now
    }
    const std::size_t   batch   = batch_chunks();
    const auto          staging = std::make_unique_for_overwrite<T[]>(batch * S);
    T*                  d_stage = allocate<T>(batch * S);
    std::int64_t*       d_idx   = allocate<std::int64_t>(batch);
    unsigned long long* d_fail  = allocate<unsigned long long>(1);
    zero(d_fail, 1);
    for (std::size_t first = 0; first < mats.size(); first += batch)
    {
        const std::size_t count = mats.size() - first < batch ? mats.size() - first : batch;
        for (std::size_t k = 0; k < count; ++k)
        {
            const auto           c     = static_cast<std::size_t>(mats[first + k]);
            const ChunkState<T>& cs    = state.chunks[c];
            const std::size_t    valid = state.size - c * S < S ? state.size - c * S : S;
            T* const             out   = staging.get() + k * S;
            for (std::size_t e = 0; e < S; ++e)
            {
                out[e] = e >= valid                           ? T{}
                         : cs.kind == ChunkKind::materialised ? cs.data[e]
                                                              : detail::motif_element(cs.data[e], cs.value);
            }
        }
        copy(d_stage, staging.get(), count * S);
        copy(d_idx, mats.data() + first, count);
        kernels::upload_chunks<T, S><<<static_cast<unsigned int>(count < 65535 ? count : 65535), 256>>>(
            desc_, arena_.ref(), d_idx, d_stage, count, d_fail);
        synchronize();
    }
    unsigned long long failed = 0;
    copy(&failed, d_fail, 1);
    release(d_stage);
    release(d_idx);
    release(d_fail);
    if (failed != 0)
    {
        throw std::runtime_error("shrink_array::device::Array::assign: the arena ran out");
    }
}

template <typename T, unsigned int S>
[[nodiscard]] ArrayImage<T> Array<T, S>::image() const
{
    const auto nc = static_cast<std::size_t>(n_chunks_);
    synchronize();
    std::vector<ChunkDesc<T>> h(nc);
    copy(h.data(), desc_, nc);
    ArrayImage<T> img;
    img.state.size       = static_cast<std::size_t>(n_);
    img.state.chunk_size = S;
    img.state.chunks.resize(nc);
    std::vector<std::int64_t> mats;
    for (std::size_t c = 0; c < nc; ++c)
    {
        if (is_fresh(h[c].buf))
        {
            throw std::runtime_error("shrink_array::device::Array::image: a write phase without finish()");
        }
        if (h[c].buf == nullptr)
        {
            const bool first = same_value(h[c].constant, T{});
            img.state.chunks[c] =
                ChunkState<T>{first ? ChunkKind::first_value : ChunkKind::second_value, h[c].constant, nullptr};
        }
        else
        {
            mats.push_back(static_cast<std::int64_t>(c));
        }
    }
    img.elements              = std::make_unique_for_overwrite<T[]>(mats.size() * S);
    const std::size_t batch   = batch_chunks();
    T*                d_stage = allocate<T>(batch * S);
    std::int64_t*     d_idx   = allocate<std::int64_t>(batch);
    for (std::size_t first = 0; first < mats.size(); first += batch)
    {
        const std::size_t count = mats.size() - first < batch ? mats.size() - first : batch;
        copy(d_idx, mats.data() + first, count);
        kernels::gather_chunks<T, S>
            <<<static_cast<unsigned int>(count < 65535 ? count : 65535), 256>>>(desc_, d_idx, d_stage, count);
        synchronize();
        copy(img.elements.get() + first * S, d_stage, count * S);
    }
    release(d_stage);
    release(d_idx);
    for (std::size_t k = 0; k < mats.size(); ++k)
    {
        img.state.chunks[static_cast<std::size_t>(mats[k])] =
            ChunkState<T>{ChunkKind::materialised, T{}, img.elements.get() + k * S};
    }
    return img;
}

template <typename T, unsigned int S>
[[nodiscard]] unsigned long long Array<T, S>::materialised() const
{
    zero(scratch_, 1);
    kernels::count_materialised<<<256, 256>>>(desc_, n_chunks_, scratch_);
    synchronize();
    unsigned long long h = 0;
    copy(&h, scratch_, 1);
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
