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
 * @brief device::Array, the host side of a device zero_any or zero array: its descriptors,
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
 *          - writer() and finish(): for a kernel of the caller's that calls Writer::assign
 *            (docs/gpu_assign.md), then finish(): the append-buffer replayed, fresh buffers
 *            settled, spare and lost buffers given back.
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
#include <algorithm>   // std::stable_sort
#include <memory>      // std::unique_ptr, std::make_unique_for_overwrite
#include <numeric>     // std::iota
#include <stdexcept>   // std::runtime_error, std::invalid_argument
#include <string>      // std::to_string
#include <type_traits> // std::is_trivially_copyable_v, std::is_same_v
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

/// @brief Every descriptor {nullptr, @p constant} (zero: {nullptr}).
template <typename T, Variant V>
__global__ void init_desc(ChunkDesc<T, V>* desc, std::int64_t n_chunks, T constant)
{
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t c = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x; c < n_chunks; c += stride)
    {
        desc[c].buf = nullptr;
        if constexpr (V == Variant::zero_any)
        {
            desc[c].constant = constant;
        }
    }
}

/// @brief Every element into a flat buffer.
template <typename T, unsigned int S, Variant V>
__global__ void expand(View<T, S, V> v, T* out)
{
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x; i < v.size(); i += stride)
    {
        out[i] = v(i);
    }
}

/// @brief Chunks holding a buffer.
template <typename T, Variant V>
__global__ void count_materialised(const ChunkDesc<T, V>* desc, std::int64_t n_chunks, unsigned long long* out)
{
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    unsigned long long mine   = 0;
    for (std::int64_t c = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x; c < n_chunks; c += stride)
    {
        mine += desc[c].buf != nullptr ? 1 : 0;
    }
    fetch_add(out, mine);
}

/// @brief Chunks of @p chunks with no buffer, added to @p out.
template <typename T, Variant V>
__global__ void
count_compressed(const ChunkDesc<T, V>* desc, const std::int64_t* chunks, unsigned long long m, unsigned long long* out)
{
    const unsigned long long stride = static_cast<unsigned long long>(gridDim.x) * blockDim.x;
    unsigned long long       mine   = 0;
    for (unsigned long long k = blockIdx.x * static_cast<unsigned long long>(blockDim.x) + threadIdx.x; k < m;
         k += stride)
    {
        mine += desc[chunks[k]].buf == nullptr ? 1 : 0;
    }
    fetch_add(out, mine);
}

/// @brief Every buffer back to the arena (after normalise()), every descriptor compressed.
template <typename T, unsigned int S, Variant V>
__global__ void release_all(ChunkDesc<T, V>* desc, std::int64_t n_chunks, ArenaRef<T, S> arena)
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
template <typename T, unsigned int S, Variant V>
__global__ void upload_chunks(ChunkDesc<T, V>*    desc,
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
template <typename T, unsigned int S, Variant V>
__global__ void
gather_chunks(const ChunkDesc<T, V>* desc, const std::int64_t* chunks, T* staging, unsigned long long count)
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

/// Arrays whose counters one gather_counters launch copies (finish_all()).
inline constexpr int gather_max = 16;

/// @brief Where gather_counters reads and writes: n pairs.
template <typename C>
struct CounterGather
{
    const C* src[gather_max];
    C*       dst[gather_max];
    int      n;
};

/// @brief Copies the counters of up to gather_max arrays into pinned host memory: one launch
///        instead of a copy per array (on the MI300A an asynchronous copy is about 5 us of host
///        time and a blit kernel each).
template <typename C>
__global__ void gather_counters(CounterGather<C> g)
{
    for (int k = static_cast<int>(threadIdx.x); k < g.n; k += static_cast<int>(blockDim.x))
    {
        *g.dst[k] = *g.src[k];
    }
}

} // namespace kernels

/// @brief An array's counters on the device and their pinned host copy (finish_all()).
struct CounterSlot
{
    const WriteCounters* device;
    WriteCounters*       host;
};

/**
 * @brief A device array of @p T in chunks of @p S, every element @p constant at first.
 * @tparam T element type, trivially copyable
 * @tparam S chunk size, a power of two, 32 at least (S = 1024 on GPU, docs/gpu_design.md 13)
 * @tparam V the variant (chunk_desc.h): zero_any, or zero (the constant is T{}, a chunk
 *           folds back only to it, and the descriptor is its pointer alone)
 */
template <typename T, unsigned int S, Variant V = Variant::zero_any>
class Array
{
    static_assert(std::is_trivially_copyable_v<T>, "device::Array: trivially copyable elements");

  public:
    /// Threads per block of the write kernels: 256, or S below it.
    static constexpr int default_block = static_cast<int>(S < 256 ? S : 256);

    /// Spare slots of the write path: more than the warps resident on an H100 or an MI300A.
    static constexpr unsigned long long spare_slots = 1ULL << 14;

    /// @brief @p n elements reading @p constant; an arena of @p arena_buffers buffers to start;
    ///        an append-buffer of @p append_capacity entries (0: 1% of the arena's elements,
    ///        4096 at least), for Writer::assign when the arena runs out.
    /// @throws std::invalid_argument in the zero variant if @p constant is not T{}.
    Array(std::int64_t n, unsigned long long arena_buffers, T constant = T{}, unsigned long long append_capacity = 0);
    ~Array();
    Array(const Array&)            = delete;
    Array& operator=(const Array&) = delete;
    Array(Array&&)                 = delete;
    Array& operator=(Array&&)      = delete;

    /// @brief For a read kernel, between write phases.
    [[nodiscard]] View<T, S, V> view() const noexcept
    {
        return View<T, S, V>{desc_, n_};
    }

    /// @brief For a write kernel of the caller's that calls Writer::write (then finish(f)) or
    ///        Writer::assign (then finish()).
    [[nodiscard]] Writer<T, S, V> writer() const noexcept;

    /// @brief a(i) = f(i) for i in [@p lo, @p hi) by the chunk-aligned loop, then finish(f).
    ///        f: `T operator()(std::int64_t) const`, a function of i.
    template <int BLOCK = default_block, typename F>
    void parallel_for_chunks(std::int64_t lo, std::int64_t hi, const F& f, unsigned int grid = 4096);

    /// @brief The end of a write phase whose kernels called Writer::write with @p f: settles
    ///        the fresh buffers, writes the spans the arena could not serve again after
    ///        growing it (as often as needed), gives the lost buffers back. Waits.
    template <int BLOCK = default_block, typename F>
    void finish(const F& f);

    /// @brief The end of a write phase whose kernels called Writer::assign: replays the
    ///        append-buffer (growing the arena), settles the fresh buffers, gives the spare
    ///        and lost buffers back. Waits.
    /// @throws std::runtime_error if the append-buffer overflowed (writes lost), or if a
    ///         Writer::write span failed (it needs finish(f)).
    void finish();

    /// @brief For finish_all(): where the counters are and where their copy goes.
    [[nodiscard]] CounterSlot counter_slot() const noexcept
    {
        return {counters_, seen_};
    }

    /// @brief finish() from the counters finish_all() copied into counter_slot().host, after a
    ///        synchronize(). Waits only for the steps that need it (appends, failed spans,
    ///        parked buffers).
    void finish_end();

    /// @brief The compression pass (section 7), between phases: every materialised chunk
    ///        whose elements below size() are one value folds back to it, its buffer back to
    ///        the arena. Returns the chunks folded back. Waits.
    unsigned long long compress();

    /// @brief Makes this array the one @p state describes (export_array() of a host array
    ///        of size() elements at chunk size S): what it held goes back to the arena, the
    ///        arena grows if the materialised chunks need it. Between phases. Waits. In the
    ///        zero variant a chunk compressed to another value than T{} gets a buffer.
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

    /// @brief The write path's counters since the array was made (cas_lost, claim_wait,
    ///        claim_spin; the list counters are reset by finish()).
    [[nodiscard]] WriteCounters write_counters() const;

    /// @brief Every element, on the host (waits).
    [[nodiscard]] std::vector<T> to_host() const;

    /// @brief The arena.
    [[nodiscard]] DeviceArena<T, S>& arena() noexcept
    {
        return arena_;
    }

  private:
    struct NoFunction
    {
    };
    template <int BLOCK, typename F>
    void finish_impl(const F* f);
    template <int BLOCK, typename F>
    void                             finish_from(const WriteCounters& first, const F* f);
    void                             replay_appends();
    [[nodiscard]] unsigned long long count_compressed(const std::vector<std::int64_t>& chunks);
    void                             fit_lost_list();
    void                             grow_arena(unsigned long long more);
    void                             settle();
    void                             settle_listed(unsigned long long count);
    void                             reclaim_lost();
    [[nodiscard]] std::size_t        batch_chunks() const noexcept
    {
        constexpr std::size_t batch_bytes = std::size_t{64} << 20;
        return batch_bytes / (S * sizeof(T)) > 0 ? batch_bytes / (S * sizeof(T)) : 1;
    }

    std::int64_t     n_;
    std::int64_t     n_chunks_;
    ChunkDesc<T, V>* desc_       = nullptr;
    unsigned*        written_    = nullptr;
    Span*            failed_     = nullptr;
    T**              lost_       = nullptr;
    std::int64_t*    fresh_list_ = nullptr; ///< as many entries as the arena has buffers (fit_lost_list)
    WriteCounters*   counters_   = nullptr;
    WriteCounters*   seen_       = nullptr; ///< pinned host copy of the counters (finish_all)
    /// One counter for compress() and materialised(), allocated once instead of a device
    /// allocation and free per call.
    unsigned long long* scratch_ = nullptr;
    unsigned long long  failed_capacity_;
    unsigned long long  lost_capacity_;
    T**                 spare_        = nullptr;
    std::int64_t*       append_index_ = nullptr;
    T*                  append_value_ = nullptr;
    unsigned long long  append_capacity_;
    unsigned long long  cas_lost_seen_ = 0; ///< cas_lost when finish() last gave the parked buffers back
    DeviceArena<T, S>   arena_;
};

// ---------------------------------------------------------------------------

/// @details The failed-span list holds two spans per chunk (a chunk-aligned loop fails once
///          per chunk at most; a caller's kernel may have more spans per chunk, past that the
///          phase throws). The lost list holds one buffer per buffer of the arena: a buffer is
///          lost at most once per take (fit_lost_list() follows the arena's growth).
template <typename T, unsigned int S, Variant V>
Array<T, S, V>::Array(std::int64_t n, unsigned long long arena_buffers, T constant, unsigned long long append_capacity)
    : n_(n)
    , n_chunks_((n + S - 1) / S)
    , failed_capacity_(2 * static_cast<unsigned long long>(n_chunks_) + 1)
    , lost_capacity_(0)
    , append_capacity_(append_capacity != 0                ? append_capacity
                       : arena_buffers * S / 100 > 4096ULL ? arena_buffers * S / 100
                                                           : 4096ULL)
    , arena_(arena_buffers)
{
    if (V == Variant::zero && !same_value(constant, T{}))
    {
        throw std::invalid_argument("shrink_array::device::Array: the zero variant reads T{}, no other constant");
    }
    const auto nc = static_cast<std::size_t>(n_chunks_);
    desc_         = allocate<ChunkDesc<T, V>>(nc);
    written_      = allocate<unsigned>(nc * Writer<T, S, V>::words);
    failed_       = allocate<Span>(static_cast<std::size_t>(failed_capacity_));
    counters_     = allocate<WriteCounters>(1);
    seen_         = allocate_pinned<WriteCounters>(1);
    scratch_      = allocate<unsigned long long>(1);
    spare_        = allocate<T*>(spare_slots);
    append_index_ = allocate<std::int64_t>(static_cast<std::size_t>(append_capacity_));
    append_value_ = allocate<T>(static_cast<std::size_t>(append_capacity_));
    zero(written_, nc * Writer<T, S, V>::words);
    zero(counters_, 1);
    zero(spare_, spare_slots);
    kernels::reset_append<<<256, 256>>>(append_index_, append_capacity_);
    fit_lost_list();
    if (n_chunks_ > 0)
    {
        kernels::init_desc<<<256, 256>>>(desc_, n_chunks_, constant);
    }
    synchronize();
}

template <typename T, unsigned int S, Variant V>
Array<T, S, V>::~Array()
{
    release(desc_);
    release(written_);
    release(failed_);
    release(lost_);
    release(fresh_list_);
    release(counters_);
    release_pinned(seen_);
    release(scratch_);
    release(spare_);
    release(append_index_);
    release(append_value_);
}

template <typename T, unsigned int S, Variant V>
[[nodiscard]] Writer<T, S, V> Array<T, S, V>::writer() const noexcept
{
    Writer<T, S, V> w;
    w.desc            = desc_;
    w.n               = n_;
    w.arena           = arena_.ref();
    w.written         = written_;
    w.lost            = lost_;
    w.lost_capacity   = lost_capacity_;
    w.fresh_list      = fresh_list_;
    w.fresh_capacity  = lost_capacity_;
    w.failed          = failed_;
    w.failed_capacity = failed_capacity_;
    w.counters        = counters_;
    w.spare           = spare_;
    w.spare_slots     = spare_slots;
    w.append_index    = append_index_;
    w.append_value    = append_value_;
    w.append_capacity = append_capacity_;
    return w;
}

template <typename T, unsigned int S, Variant V>
template <int BLOCK, typename F>
void Array<T, S, V>::parallel_for_chunks(std::int64_t lo, std::int64_t hi, const F& f, unsigned int grid)
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
template <typename T, unsigned int S, Variant V>
template <int BLOCK, typename F>
void Array<T, S, V>::finish(const F& f)
{
    finish_impl<BLOCK>(&f);
}

template <typename T, unsigned int S, Variant V>
void Array<T, S, V>::finish()
{
    finish_impl<default_block>(static_cast<const NoFunction*>(nullptr));
}

/// @details The appends first: their replay skips the elements marked in a fresh buffer, and
///          settle_fresh clears the marks (docs/gpu_assign.md 3.4, 3.5).
/// @details One read of the counters first: a phase that left no append, no failed span, no lost
///          or parked buffer (cas_lost unchanged) and published no fresh buffer has nothing to
///          finish -- a phase that only wrote into materialised chunks, or the constant onto its
///          chunks. The full path read them, synchronised and scanned every chunk some eight
///          times: about 115 us a call on the T1000, which made the small windows of
///          kokkos_shrink_bench 12x to 38x their flat time.
template <typename T, unsigned int S, Variant V>
template <int BLOCK, typename F>
void Array<T, S, V>::finish_impl(const F* f)
{
    synchronize();
    WriteCounters first{};
    copy(&first, counters_, 1);
    finish_from<BLOCK>(first, f);
}

template <typename T, unsigned int S, Variant V>
void Array<T, S, V>::finish_end()
{
    const WriteCounters first = *seen_;
    finish_from<default_block>(first, static_cast<const NoFunction*>(nullptr));
}

/// @details Every step from @p first, the counters read once after the phase's kernels.
template <typename T, unsigned int S, Variant V>
template <int BLOCK, typename F>
void Array<T, S, V>::finish_from(const WriteCounters& first, const F* f)
{
    const bool parked = first.n_lost != 0 || first.cas_lost != cas_lost_seen_;
    if (first.n_append == 0 && first.n_failed == 0 && !parked && first.fresh == 0)
    {
        return;
    }
    // Each step from that one read: a phase that only published fresh buffers (the writes of
    // kokkos_shrink_bench's windows) costs one copy, the settle and one synchronisation, not
    // the five copies, three kernels and three synchronisations of every step in turn (MI300A:
    // 37x to 57x the flat time for those phases).
    if (first.n_append != 0)
    {
        replay_appends(); // its assign() calls may publish fresh buffers: settled just below
    }
    if (first.n_append != 0)
    {
        settle(); // the replay may have published fresh buffers: not counted in `first`
    }
    else if (first.fresh != 0)
    {
        settle_listed(first.fresh);
    }
    while (first.n_failed != 0) // left by its break once the spans are all written
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
        if constexpr (std::is_same_v<F, NoFunction>)
        {
            throw std::runtime_error("shrink_array::device::Array::finish(): " + std::to_string(n_spans) +
                                     " Writer::write spans to write again: call finish(f)");
        }
        reclaim_lost();
        grow_arena(n_spans + n_spans / 4 + 1);
        Span* spans = allocate<Span>(static_cast<std::size_t>(n_spans));
        copy(spans, failed_, static_cast<std::size_t>(n_spans));
        copy(&c, counters_, 1);
        c.n_failed = 0;
        copy(counters_, &c, 1);
        const auto blocks = static_cast<unsigned int>(n_spans < 4096 ? n_spans : 4096);
        if constexpr (!std::is_same_v<F, NoFunction>)
        {
            kernels::rewrite_spans<BLOCK><<<blocks, BLOCK>>>(writer(), spans, n_spans, *f);
        }
        synchronize();
        release(spans);
        settle();
    }
    if (parked || first.n_failed != 0 || first.n_append != 0)
    {
        reclaim_lost(); // the lost list and the spare slots back to the arena
        WriteCounters c{};
        copy(&c, counters_, 1);
        cas_lost_seen_ = c.cas_lost;
    }
}

template <typename T, unsigned int S, Variant V>
void Array<T, S, V>::settle()
{
    if (n_chunks_ > 0)
    {
        const std::int64_t warps_per_block = 256 / warp_size;
        const std::int64_t blocks          = (n_chunks_ + warps_per_block - 1) / warps_per_block;
        kernels::settle_fresh<<<static_cast<unsigned int>(blocks < 65536 ? blocks : 65536), 256>>>(writer(), n_chunks_);
    }
    synchronize();
    zero(&counters_->fresh, 1); // every fresh buffer settled (a device address, not read here)
}

/// @details The chunks of the first @p count fresh publications only, from the fresh list (a
///          scan if it overflowed). No synchronisation: the next kernels, the library's and
///          Kokkos's (its streams wait for the default one), run after it on the device, and a
///          read on the host copies after it.
template <typename T, unsigned int S, Variant V>
void Array<T, S, V>::settle_listed(unsigned long long count)
{
    if (count > lost_capacity_)
    {
        settle();
        return;
    }
    const unsigned long long warps_per_block = 256 / warp_size;
    const unsigned long long blocks          = (count + warps_per_block - 1) / warps_per_block;
    kernels::settle_listed<<<static_cast<unsigned int>(blocks < 65536 ? blocks : 65536), 256>>>(writer(), count);
}

template <typename T, unsigned int S, Variant V>
void Array<T, S, V>::reclaim_lost()
{
    WriteCounters c{};
    copy(&c, counters_, 1);
    if (c.n_lost > lost_capacity_)
    {
        throw std::runtime_error("shrink_array::device::Array: " + std::to_string(c.n_lost) +
                                 " lost buffers, past the capacity of the list");
    }
    arena_.normalise();
    kernels::reclaim_lost<<<64, 256>>>(writer(), c.n_lost); // the spare slots too
    synchronize();
    if (c.n_lost > 0)
    {
        c.n_lost = 0;
        copy(counters_, &c, 1);
    }
}

/// @details By @p more buffers, or a quarter of the arena if that is more: a phase that misses a
///          few buffers would otherwise add an arena each time, and an array whose chunks stay
///          materialised (the zero variant under constants other than T{}) reached max_arenas
///          after about 200 small phases (kokkos_shrink_bench). Geometric, the arenas stay
///          O(log) of the growth. Between phases.
template <typename T, unsigned int S, Variant V>
void Array<T, S, V>::grow_arena(unsigned long long more)
{
    const unsigned long long quarter = arena_.buffers() / 4;
    arena_.grow(more > quarter ? more : quarter);
    fit_lost_list();
}

/// @details The lost list holds as many buffers as the arena: called after every growth,
///          between phases.
template <typename T, unsigned int S, Variant V>
void Array<T, S, V>::fit_lost_list()
{
    if (lost_capacity_ >= arena_.buffers() && lost_ != nullptr)
    {
        return;
    }
    release(lost_);
    release(fresh_list_);
    lost_capacity_ = arena_.buffers() > 0 ? arena_.buffers() : 1;
    lost_          = allocate<T*>(static_cast<std::size_t>(lost_capacity_));
    fresh_list_    = allocate<std::int64_t>(static_cast<std::size_t>(lost_capacity_));
}

/// @brief Chunks of @p chunks whose descriptor holds no buffer (between phases).
template <typename T, unsigned int S, Variant V>
[[nodiscard]] unsigned long long Array<T, S, V>::count_compressed(const std::vector<std::int64_t>& chunks)
{
    if (chunks.empty())
    {
        return 0;
    }
    std::int64_t* const d_chunks = allocate<std::int64_t>(chunks.size());
    copy(d_chunks, chunks.data(), chunks.size());
    zero(scratch_, 1);
    kernels::count_compressed<<<256, 256>>>(desc_, d_chunks, static_cast<unsigned long long>(chunks.size()), scratch_);
    synchronize();
    unsigned long long h = 0;
    copy(&h, scratch_, 1);
    release(d_chunks);
    return h;
}

/// @details Deduplicated on the host: the append-buffer only fills when the arena runs out or a
///          wait is given up, it is small. For each element the last append (the highest index:
///          one thread's appends are in its program order); the arena grows only if the chunks
///          still compressed need more buffers than it has free (one claim, one buffer per
///          chunk). The replay may append again (to the emptied buffer): repeat.
template <typename T, unsigned int S, Variant V>
void Array<T, S, V>::replay_appends()
{
    for (;;)
    {
        WriteCounters c{};
        copy(&c, counters_, 1);
        if (c.n_append == 0)
        {
            return;
        }
        if (c.n_append > append_capacity_)
        {
            throw std::runtime_error(
                "shrink_array::device::Array::finish: " + std::to_string(c.n_append - append_capacity_) +
                " writes lost (append-buffer full: a larger one, or a larger arena)");
        }
        const auto                m = static_cast<std::size_t>(c.n_append);
        std::vector<std::int64_t> idx(m);
        const auto                val = std::make_unique_for_overwrite<T[]>(m);
        copy(idx.data(), append_index_, m);
        copy(val.get(), append_value_, m);
        kernels::reset_append<<<256, 256>>>(append_index_, m); // before the replay appends again
        std::vector<std::size_t> order(m);
        std::iota(order.begin(), order.end(), std::size_t{0});
        std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return idx[a] < idx[b]; });
        std::vector<std::int64_t> last_idx;
        std::vector<std::int64_t> chunks;
        const auto                last_val = std::make_unique_for_overwrite<T[]>(m);
        for (std::size_t k = 0; k < m; ++k)
        {
            const std::size_t j = order[k];
            if (k + 1 < m && idx[order[k + 1]] == idx[j])
            {
                continue; // a later append of the same element
            }
            if (chunks.empty() || chunks.back() != idx[j] / S)
            {
                chunks.push_back(idx[j] / S);
            }
            last_val[last_idx.size()] = val[j];
            last_idx.push_back(idx[j]);
        }
        reclaim_lost();
        // Room for the chunks still compressed: an append whose wait was given up (MI300A) or
        // whose chunk was published later finds a buffer, and growing for it piled up arenas.
        const unsigned long long needed = count_compressed(chunks);
        const unsigned long long free   = arena_.buffers() - arena_.in_use();
        if (needed > free)
        {
            grow_arena(needed - free + needed / 4 + 1);
        }
        copy(&c, counters_, 1);
        c.n_append = 0;
        copy(counters_, &c, 1);
        const std::size_t   u       = last_idx.size();
        std::int64_t* const d_index = allocate<std::int64_t>(u);
        T* const            d_value = allocate<T>(u);
        copy(d_index, last_idx.data(), u);
        copy(d_value, last_val.get(), u);
        kernels::replay_appends<T, S, V><<<256, 256>>>(writer(), d_index, d_value, u);
        synchronize();
        release(d_index);
        release(d_value);
    }
}

/// @details One warp per chunk (the prototype: 85% to 102% of the stream read at S = 1024 on
///          the GH200 and the MI300A when every chunk folds back), at most 65536 blocks of 256
///          threads, grid-stride beyond.
template <typename T, unsigned int S, Variant V>
unsigned long long Array<T, S, V>::compress()
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
        kernels::compress_chunks<T, S, V>
            <<<static_cast<unsigned int>(blocks < 65536 ? blocks : 65536), 256>>>(desc_, n_, arena_.ref(), d);
    }
    synchronize();
    unsigned long long folded = 0;
    copy(&folded, d, 1);
    return folded;
}

/// @details The materialised and motif chunks are filled on the host into a staging buffer
///          (a motif expanded), copied in batches, and each batch's chunks take a buffer from
///          the arena on the device; the tail of the last chunk holds T{}. In the zero
///          variant a chunk compressed to another value than T{} (a zero_any or zero_or_value
///          host array) is materialised too, every element that value.
template <typename T, unsigned int S, Variant V>
void Array<T, S, V>::assign(const ArrayState<T>& state)
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
        kernels::release_all<T, S, V><<<256, 256>>>(desc_, n_chunks_, arena_.ref());
    }
    synchronize();
    std::vector<ChunkDesc<T, V>> h(nc);
    std::vector<std::int64_t>    mats;
    for (std::size_t c = 0; c < nc; ++c)
    {
        const ChunkState<T>& cs       = state.chunks[c];
        const bool           constant = cs.kind == ChunkKind::first_value || cs.kind == ChunkKind::second_value;
        h[c].buf                      = nullptr;
        if constexpr (V == Variant::zero_any)
        {
            h[c].constant = constant ? cs.value : T{};
        }
        // zero: a chunk compressed to another value than T{} is materialised, filled with it.
        if (!constant || (V == Variant::zero && !same_value(cs.value, T{})))
        {
            mats.push_back(static_cast<std::int64_t>(c));
        }
    }
    copy(desc_, h.data(), nc);
    if (arena_.buffers() < mats.size())
    {
        arena_.grow(mats.size() - arena_.buffers()); // every buffer is free now
        fit_lost_list();
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
                         : cs.kind == ChunkKind::motif        ? detail::motif_element(cs.data[e], cs.value)
                                                              : cs.value; // zero: a constant other than T{}
            }
        }
        copy(d_stage, staging.get(), count * S);
        copy(d_idx, mats.data() + first, count);
        kernels::upload_chunks<T, S, V><<<static_cast<unsigned int>(count < 65535 ? count : 65535), 256>>>(
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

template <typename T, unsigned int S, Variant V>
[[nodiscard]] ArrayImage<T> Array<T, S, V>::image() const
{
    const auto nc = static_cast<std::size_t>(n_chunks_);
    synchronize();
    std::vector<ChunkDesc<T, V>> h(nc);
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
            const T    value = constant_of(h[c]);
            const bool first = same_value(value, T{});
            img.state.chunks[c] =
                ChunkState<T>{first ? ChunkKind::first_value : ChunkKind::second_value, value, nullptr};
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
        kernels::gather_chunks<T, S, V>
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

template <typename T, unsigned int S, Variant V>
[[nodiscard]] unsigned long long Array<T, S, V>::materialised() const
{
    zero(scratch_, 1);
    kernels::count_materialised<<<256, 256>>>(desc_, n_chunks_, scratch_);
    synchronize();
    unsigned long long h = 0;
    copy(&h, scratch_, 1);
    return h;
}

template <typename T, unsigned int S, Variant V>
[[nodiscard]] WriteCounters Array<T, S, V>::write_counters() const
{
    WriteCounters c{};
    copy(&c, counters_, 1);
    return c;
}

template <typename T, unsigned int S, Variant V>
[[nodiscard]] unsigned long long Array<T, S, V>::cas_lost() const
{
    WriteCounters c{};
    copy(&c, counters_, 1);
    return c.cas_lost;
}

template <typename T, unsigned int S, Variant V>
[[nodiscard]] std::vector<T> Array<T, S, V>::to_host() const
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

/// @brief finish() of every array of @p a with one wait for all: their counters copied into
///        pinned host memory by one kernel (per gather_max arrays), one synchronize(), each
///        array finished from its copy. A phase that wrote several arrays (kokkos_shrink_bench's
///        five) waited once per array, a synchronisation and a blocking copy each: most of such
///        a phase's time when its kernels are small.
template <typename... A>
void finish_all(A&... a)
{
    const CounterSlot slots[] = {a.counter_slot()...};
    constexpr int     n       = static_cast<int>(sizeof...(A));
    for (int k = 0; k < n; k += kernels::gather_max)
    {
        kernels::CounterGather<WriteCounters> g{};
        g.n = n - k < kernels::gather_max ? n - k : kernels::gather_max;
        for (int j = 0; j < g.n; ++j)
        {
            g.src[j] = slots[k + j].device;
            g.dst[j] = slots[k + j].host;
        }
        kernels::gather_counters<<<1, 32>>>(g);
    }
    synchronize();
    (a.finish_end(), ...);
}

} // namespace shrink_array::device
