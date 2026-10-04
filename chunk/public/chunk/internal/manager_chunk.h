/*
 * Copyright 2024-2026 CEA (www.cea.fr)
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

#include <algorithm>     // std::max
#include <map>           // std::map
#include <vector>        // std::vector
#include <tuple>         // std::tuple, std::get
#include <cstdio>        // fprintf (pool breakdown)
#include <unordered_map> // std::unordered_map
#include <cstdint>       // pool keys (drain_reuse_list_for_size, keyed lists)
#include <type_traits>   // remove_reference_t / remove_pointer_t (same)
#include <functional>    // std::plus (parked_blocks)
#include <ranges>        // fold_left, views::transform / values (parked_blocks)

#include "chunk/bit.h"                        // is_bool_like_v: bool and Bit share the bool path
#include "chunk/chunk_def.h"                  // ChunkDef::size_min_buffer_reuse
#include "chunk/config.h"                     // build_config::use_openmp, chunk_config::chunk_dynamic
#include "chunk/internal/chunk_size_policy.h" // thread_chunk_size, set_thread_chunk_size
#include "chunk/internal/TChunk.h"            // TChunkComp, TChunk2CompMap, TChunk2AComp
#include "chunk/PMP_CHUNK.h"                  // CHUNK_PMP_THREAD_CAPACITY (max_thr)
#include "chunk/internal/pool_ledger.h"       // CHUNK_LEDGER_DISCARD (drain_reuse_list_for_size)
#include "chunk/type_name.h"                  // type_name<T> (pool breakdown)
#include "fast_allocators/internal/svalloc.h" // svfree (drain_reuse_list_for_size)

#include "chunk/internal/freepath_counters.h" // CHUNK_FREEPATH_INC (CHUNK_ARENA_REPORT)

namespace shrink_array
{
namespace internal
{
/// @brief One repeat seen by the teardown dedup phase: skip @p ptr when freeing bucket @p bucket.
/// @see ManagerChunkParallelImpl::clear_list_block_free_dedup
template <typename T>
struct TeardownRepeat
{
    std::size_t bucket;
    T*          ptr;
};
} // namespace internal

/**
 * @brief The CHUNK_DYNAMIC-only half of the manager interface.
 *
 * @details Runtime geometry adds one verb, and only that build has it. Selected
 *          by chunk_config / build_config constants that mirror the macros
 *          exactly, so the surface is chosen by the type system rather than by
 *          the preprocessor -- the same policy split the chunk classes use
 *          (ChunkAllocPolicy, ChunkDeferredOpsPolicy, DeferredReadGate).
 *
 * @tparam IsDynamic Whether runtime geometry is compiled in.
 */
template <bool IsDynamic = build_config::chunk_dynamic>
struct ManagerChunkDynInterface; // primary left undefined: only the 2 specializations exist

template <>
struct ManagerChunkDynInterface<true>
{
    virtual ~ManagerChunkDynInterface() = default;

    /// @brief Declare a runtime chunk geometry to this type's allocators.
    virtual void register_chunk_size(unsigned int size, int block_size) = 0;
};

/// @brief Static-geometry builds: nothing to declare.
template <>
struct ManagerChunkDynInterface<false>
{
    virtual ~ManagerChunkDynInterface() = default;
};

/**
 * @brief The allocator / free-list half of the manager interface.
 *
 * @details The two modes do not offer a narrower and a wider surface, they
 *          offer DIFFERENT ones: OpenMP has per-thread allocators, per-thread
 *          reuse lists and a deferred-op queue to flush; the serial build has a
 *          single allocator and none of the rest. That is why this is two
 *          specializations rather than one set with no-ops.
 *
 * @tparam UseOpenMP Whether the parallel surface is the one compiled in.
 */
template <bool UseOpenMP = build_config::use_openmp>
struct ManagerChunkParallelInterface; // primary left undefined: only the 2 specializations exist

template <>
struct ManagerChunkParallelInterface<true>
{
    virtual ~ManagerChunkParallelInterface() = default;

    // --- per-thread allocator and free-list surface ---
    /// @brief Set every rank's allocator geometry, from the calling thread:
    ///        the allocators are slots indexed by rank, and a slot's slab is
    ///        only taken at that rank's first allocation. Sequential.
    virtual void init_allocators(int block_size) const = 0;
    /// @brief Create this type's array of per-thread chunk allocators
    ///        (fast-alloc builds; a no-op otherwise).
    virtual void alloc_chunk_allocator_thr() = 0;

    /// @brief Size the per-rank free-lists of released buffers to the team.
    virtual void resize_all_list_block_free_thr() = 0;
    /// @brief Apply this type's pending deferred write ops, repeatedly until a
    ///        pass changes nothing (a replay can enqueue another op).
    /// @param in_parallel  True when called from inside a parallel region.
    /// @param have_flushed Set (never cleared) when at least one op was applied.
    virtual void flush_all_deferred(bool in_parallel, bool& have_flushed) = 0;
    /// @brief After the last flush: destroy the per-thread op maps and clear
    ///        every array's dirty bitset, so the next phase starts clean.
    virtual void destroy_after_flush(bool in_parallel) = 0;

    /// @brief Destroy the per-thread free-list containers themselves (teardown).
    virtual void delete_list_block_chunk_free_itself() = 0;

    /// @brief Would load_balance_list_block_free() be worth calling?
    [[nodiscard]] virtual bool need_load_balance_list_block_free() const = 0;
    /// @brief Redistribute released buffers across the per-thread free-lists
    ///        (a thread freeing far more than it allocates would hoard them).
    virtual void load_balance_list_block_free() = 0;

    // --- teardown of the reuse lists, in phases (see manager_chunk_inline.h) ---
    // AllTypeChunk::clear_list_block_free_thr() drives them: prepare() outside
    // the region, then inside ONE region and separated by barriers: screen()
    // per bucket, sort() per bucket, dedup() per address slice, thr() per
    // bucket. Every per-bucket call is independent of the others in its phase,
    // so a thread may take several buckets (team smaller than the count).

    /// @brief Size the teardown buckets. Sequential, before the region.
    /// @return the bucket count the per-bucket phases stride over.
    virtual std::size_t clear_list_block_free_prepare() = 0;

    /// @brief Drain rank @p bucket's reuse lists into its bucket. Out-of-range is a no-op.
    virtual void clear_list_block_free_screen(int bucket) = 0;

    /// @brief Sort bucket @p bucket and drop its own repeats. Out-of-range is a no-op.
    virtual void clear_list_block_free_sort(int bucket) = 0;

    /// @brief Record the cross-bucket repeats within address slice @p slice of @p nslices.
    virtual void clear_list_block_free_dedup(int slice, int nslices) = 0;

    /// @brief Free bucket @p bucket, minus the recorded repeats. Out-of-range is a no-op.
    /// @note One rank per thread when the team allows: a block is cheapest to
    ///       release through the arena it came from.
    virtual void clear_list_block_free_thr(int bucket) = 0;
};

/// @brief Serial builds: one allocator, no per-thread free lists, nothing to flush.
template <>
struct ManagerChunkParallelInterface<false>
{
    virtual ~ManagerChunkParallelInterface() = default;

    /// @brief Set the single allocator's geometry.
    virtual void init_allocator(int block_size) const = 0;
    /// @brief Destroy the single allocator (teardown).
    virtual void delete_chunk_allocator() = 0;
};

/**
 * @brief Type-erased interface to a per-(variant, element-type) chunk manager.
 *
 * @details ManagerChunk is a template on both the chunk variant and the element
 *          type, so there is one distinct manager class per combination. Code
 *          that must act on ALL of them at once -- AllTypeChunk, which walks its
 *          registry to initialise allocators, run compression passes or tear
 *          everything down -- cannot name those types, so it holds
 *          ManagerChunkInt pointers and dispatches through these virtuals.
 *
 *          Which methods exist depends on the build: the OpenMP and serial
 *          builds expose different allocator and free-list surfaces (per-thread
 *          vs single), and the dynamic-geometry build adds register_chunk_size().
 *          Every implementation lives in ManagerChunk below.
 */
struct ManagerChunkInt : public virtual ManagerChunkDynInterface<>, public virtual ManagerChunkParallelInterface<>
{
    ~ManagerChunkInt() override = default;

    // --- present in every build ---
    /// @brief Release this type's allocation blocks.
    virtual void destroy_block_chunk() = 0;
    /// @brief Size the per-rank candidate lists to the team.
    virtual void resize_chunk_to_compress_potentially() const = 0;
    /// @brief Compress every compressible chunk of this type now, ignoring the
    ///        candidate lists.
    virtual void force_chunk_to_compress() const = 0;
    /// @brief Does any of the @p n_thr per-rank candidate lists hold a chunk?
    [[nodiscard]] virtual bool have_chunk_to_compress_potentially(int n_thr) const = 0;
    /// @brief Drain this type's compression-candidate backlog.
    /// @param apply true: compress the candidates found constant; false: walk
    ///        the lists without compressing (see AllTypeChunk's).
    virtual void compress_chunk_to_compress_potentially(bool apply) = 0;

    /// @name The single sweep region (AllTypeChunk::compress_chunk_to_compress_potentially)
    /// The pass compress_chunk_to_compress_potentially() makes, split so that
    /// every type runs in ONE region: prepare serially, then each thread runs
    /// sweep_rank() for its rank, then the mop-up of the ranks past the team
    /// if sweep_mops_up(). Outside a region only; no-ops without fast block
    /// compression, like the pass itself.
    ///@{
    /// @brief Serial, before the region: snapshot what sweep_rank() walks.
    virtual void sweep_prepare() = 0;
    /// @brief Rank @p thr's share of the pass.
    virtual void sweep_rank(int thr, bool apply) = 0;
    /// @brief Are the ranks past the team swept too, serially, after the region?
    [[nodiscard]] virtual bool sweep_mops_up(bool apply) const = 0;
    ///@}

    /// Bytes parked in this manager's reuse pools (printed with CHUNK_STATS).
    /// Common to every build: the reuse pools are not an OpenMP-only surface,
    /// and AllTypeChunk reaches this through a ManagerChunkInt*.
    [[nodiscard]] virtual std::size_t reuse_pool_bytes_int() const = 0;
    /// Return the pools' excess; see ManagerChunk::trim_reuse_pools.
    virtual std::size_t trim_reuse_pools_int(std::size_t keep_per_thread) = 0;
    /// Print the pools' breakdown by geometry; see
    /// ManagerChunk::print_reuse_pool_breakdown.
    virtual void print_reuse_pool_breakdown_int(const char* label) const = 0;

    // The build-dependent halves live in the three interface policies above.
};

/**
 * @brief The CHUNK_DYNAMIC-only half of the manager IMPLEMENTATION.
 *
 * @details Carries the override, so ManagerChunk needs no conditional
 *          declaration for a base virtual that exists only in that build --
 *          an `override` cannot be written conditionally, which is why the
 *          declaration has to move with the interface half rather than be
 *          guarded in place. Inherits the interface policy VIRTUALLY, so it
 *          shares the one ManagerChunkInt already brings in.
 *
 *          The body reaches ManagerChunk's own members through a CRTP
 *          down-cast: max_thr is a per-instance const, not a static.
 */
template <template <typename> typename CHUNK_T, typename T, bool IsDynamic = build_config::chunk_dynamic>
struct ManagerChunkDynImpl; // primary left undefined: only the 2 specializations exist

/// @brief Dynamic-geometry builds: implements register_chunk_size().
template <template <typename> typename CHUNK_T, typename T>
struct ManagerChunkDynImpl<CHUNK_T, T, true> : public virtual ManagerChunkDynInterface<true>
{
    void register_chunk_size(unsigned int size, int block_size) override;
};

/// @brief Static-geometry builds: nothing to implement.
template <template <typename> typename CHUNK_T, typename T>
struct ManagerChunkDynImpl<CHUNK_T, T, false> : public virtual ManagerChunkDynInterface<false>
{
};

/**
 * @brief The allocator / free-list half of the manager IMPLEMENTATION.
 *
 * @details Carries the overrides of ManagerChunkParallelInterface, so
 *          ManagerChunk needs no conditional declaration for base virtuals
 *          that exist in one mode only. @see ManagerChunkDynImpl.
 *
 *          The bodies belong to ManagerChunk's state, so they reach it through
 *          `mgr` (statics) and a CRTP down-cast (per-instance members such as
 *          max_thr) -- ManagerChunk befriends the policies for that.
 */
template <template <typename> class CHUNK_T, typename T, bool UseOpenMP = build_config::use_openmp>
class ManagerChunkParallelImpl; // primary left undefined: only the 2 specializations exist

/// @brief OpenMP builds: the per-rank allocator and free-list overrides.
template <template <typename> class CHUNK_T, typename T>
class ManagerChunkParallelImpl<CHUNK_T, T, true> : public virtual ManagerChunkParallelInterface<true>
{
  private:
    void init_allocators(int block_size) const override;
    void alloc_chunk_allocator_thr() override;

    void resize_all_list_block_free_thr() override;
    void load_balance_list_block_free() override;
    void flush_all_deferred(bool in_parallel, bool& have_flushed) override;
    void destroy_after_flush(bool in_parallel) override;

    void delete_list_block_chunk_free_itself() override;

    [[nodiscard]] bool need_load_balance_list_block_free() const override;
    std::size_t        clear_list_block_free_prepare() override;
    void               clear_list_block_free_screen(int bucket) override;
    void               clear_list_block_free_sort(int bucket) override;
    void               clear_list_block_free_dedup(int slice, int nslices) override;
    void               clear_list_block_free_thr(int bucket) override;
};

/// @brief Serial builds: a single allocator, and nothing per-thread to manage.
template <template <typename> class CHUNK_T, typename T>
class ManagerChunkParallelImpl<CHUNK_T, T, false> : public virtual ManagerChunkParallelInterface<false>
{
  private:
    void init_allocator(int block_size) const override;
    void delete_chunk_allocator() override;
};

/**
 * @brief Owns the shared, per-(variant, element-type) state behind every array.
 *
 * @details One instance exists per @c CHUNK_T / @c T combination, reachable
 *          statically. It holds three families of shared state:
 *
 *          - **Buffer reuse free-lists.** Chunk buffers released by a teardown
 *            are parked here for the next allocation instead of going back to
 *            the allocator. In dynamic builds they are keyed BY GEOMETRY
 *            (list_block_free_thr_by_size_), because a buffer physically sized
 *            for one geometry must never be popped at another -- see
 *            get_list_block_free_thr_for_size().
 *          - **Compression candidate lists.** Chunks a write left eligible for
 *            compression, queued per thread and processed in batches. Since the
 *            store is shared across arrays, a destroyed array must remove its
 *            own entries -- see the remove_from_compress_list_* helpers.
 *          - **Allocator plumbing and counters** used by the statistics and the
 *            size advisor.
 *
 * @tparam CHUNK_T chunk variant template (Chunk, Chunk2 or Chunk2A)
 * @tparam T       stored element type
 */
template <template <typename> class CHUNK_T, typename T>
class ManagerChunk final : public virtual ManagerChunkInt,
                           public ManagerChunkDynImpl<CHUNK_T, T>,
                           public ManagerChunkParallelImpl<CHUNK_T, T>
{
    // The implementation policies carry overrides whose bodies belong to
    // this class's state (max_thr is a per-instance const, not a static), and
    // they are BASES, so protected access does not reach it -- hence friendship
    // rather than a public accessor, which would widen the surface for everyone.
    template <template <typename> class, typename, bool>
    friend struct ManagerChunkDynImpl;
    template <template <typename> class, typename, bool>
    friend class ManagerChunkParallelImpl;

  private:
    using p_vector_char = std::vector<char>*;
    using inf_array_t   = std::tuple<T, int, p_vector_char>;

    struct alignas(128) PerThreadFreeList
    {
        std::vector<T*> list;
    };

    static constexpr bool is_bool     = is_bool_like_v<T>; // bool and Bit (bit.h)
    static constexpr bool is_not_bool = !is_bool;

    void               destroy_block_chunk() override;
    void               force_chunk_to_compress() const override;
    [[nodiscard]] bool have_chunk_to_compress_potentially(int n_thr) const override;
    void               compress_chunk_to_compress_potentially(bool apply) override;
    void               sweep_prepare() override;
    void               sweep_rank(int thr, bool apply) override;
    [[nodiscard]] bool sweep_mops_up(bool apply) const override;

    [[nodiscard]] std::size_t reuse_pool_bytes_int() const override
    {
        return reuse_pool_bytes();
    }
    std::size_t trim_reuse_pools_int(std::size_t keep) override
    {
        return trim_reuse_pools(keep);
    }
    void print_reuse_pool_breakdown_int(const char* label) const override
    {
        print_reuse_pool_breakdown(label);
    }

    /// @brief Per-key reuse budget (sequential: at array creation / rechunk).
    static void set_size_buffer_reuse(unsigned int key, int budget)
    {
        if (key != 0)
        {
            size_buffer_reuse_by_key_[key] = std::max(budget, ChunkDef::size_min_buffer_reuse);
        }
    }

    /// @brief Size the legacy (unsized) reuse list to @p max_thr threads.
    static void resize_list_block_free_thr(int max_thr)
    {
        list_block_free_thr.resize(max_thr);
    }
    /**
     * @brief Ensure a per-thread reuse list exists for geometry @p size.
     * @param size    geometry to register
     * @param max_thr number of per-thread slots to create
     * @note Sequential: call before any parallel region parks buffers at @p size.
     * @note A no-op in static-geometry builds: the by-size map is declared in
     *       every build (see list_block_free_thr_by_size_) but stays empty there,
     *       so the body can be selected with `if constexpr`.
     */
    static void register_free_list_size([[maybe_unused]] unsigned int size, [[maybe_unused]] int max_thr)
    {
        if constexpr (chunk_config::chunk_dynamic)
        {
            const auto it = list_block_free_thr_by_size_.find(size);
            if (it == list_block_free_thr_by_size_.end())
            {
                list_block_free_thr_by_size_[size].resize(max_thr);
            }
        }
    }

    /// @brief Create the per-rank reuse lists of (@p key, @p size) if they do
    ///        not exist yet; key 0 is the shared pool's (register_free_list_size()).
    /// @note Every build (static geometry has one size, but many keys).
    ///       Sequential: call at the array's creation, before any parallel
    ///       region parks a block of that array.
    static void register_free_list_key(unsigned int key, unsigned int size, int max_thr)
    {
        if (key == 0)
        {
            return; // the shared pool is register_free_list_size()'s
        }
        const auto it = list_block_free_thr_keyed_.find(keyed_index(key, size));
        if (it == list_block_free_thr_keyed_.end())
        {
            list_block_free_thr_keyed_[keyed_index(key, size)].resize(max_thr);
        }
    }

    /**
     * @brief Print what the reuse pools hold, broken down by geometry.
     *
     * @details reuse_pool_bytes() gives a total; this says WHERE it sits, which
     *          is what distinguishes the three ways a re-chunk can leave buffers
     *          behind:
     *          - everything under the LEGACY store: the buffers were parked
     *            through a path that does not key on geometry, and
     *            drain_reuse_list_for_size() -- which only walks the by-size map
     *            -- can never reach them. This is what happened on v1 before
     *            set_active_chunk_size() started registering the starting
     *            geometry.
     *          - concentrated on the OLD size: the drain did not run, or ran
     *            against the wrong key.
     *          - spread across every size: the drain runs but frees only part of
     *            what it should -- e.g. one thread rank out of max_thr.
     *
     *          Measured on chunk2a at 64 threads with a forced re-chunk to 256:
     *          19677 MB parked out of a 31100 MB footprint, against 16730 MB
     *          total with no re-chunk at all. The re-chunk ADDS 14 GB, and this
     *          is the breakdown that says which of the three it is.
     *
     * @param label prefix for the printed lines
     */
    static void print_reuse_pool_breakdown(const char* label) noexcept
    {
        const std::size_t legacy_blocks = parked_blocks(list_block_free_thr);

        if (legacy_blocks != 0)
        {
            std::fprintf(stderr,
                         "[POOLS] %s %-22s legacy store        : %9zu block(s), %8.1f MB"
                         " (sized by the ambient %u)\n",
                         label, type_name<T>().data(), legacy_blocks,
                         (double)(legacy_blocks * sizeof(T) * CHUNK_T<T>::storage_units(thread_chunk_size())) / 1e6,
                         thread_chunk_size());
        }

        if constexpr (chunk_config::chunk_dynamic)
        {
            for (const auto& kv : list_block_free_thr_by_size_)
            {
                const std::size_t blocks          = parked_blocks(kv.second);
                const auto        non_empty_ranks = static_cast<std::size_t>(
                    std::ranges::count_if(kv.second, [](const auto& ptf) { return !ptf.list.empty(); }));
                if (blocks == 0)
                {
                    continue;
                }
                std::fprintf(stderr,
                             "[POOLS] %s %-22s size %-6u        : %9zu block(s), %8.1f MB"
                             " across %zu/%zu rank(s)\n",
                             label, type_name<T>().data(), kv.first, blocks,
                             (double)(blocks * sizeof(T) * CHUNK_T<T>::storage_units(kv.first)) / 1e6, non_empty_ranks,
                             kv.second.size());
            }
        }
    }

    /**
     * @brief Bytes parked in this type's reuse pools, every rank, geometry and
     *        key included (printed with CHUNK_STATS).
     *
     * @details Buffers a chunk released but that were kept for reuse rather than
     *          returned. They are live memory no CHUNK_STATS counter reports:
     *          `list bytes` counts reuse EVENTS, not what is sitting in the pool
     *          at the end. Printed with the stats so a run's footprint can be
     *          split into data, chunk metadata and retention.
     *
     *          A re-chunk should leave the array at the memory of an array that
     *          was BORN at the new geometry. Measured on shrink_bench: 64
     *          everywhere = 128.0 MB, 256 everywhere = 122.9 MB, but 64 followed
     *          by a re-chunk to 256 = 169.2 MB. malloc_trim() does not move it,
     *          so the 46 MB gap is live memory the library still owns -- and the
     *          reuse pools are where old-geometry buffers go. This counts them.
     */
    [[nodiscard]] static std::size_t reuse_pool_bytes() noexcept
    {
        std::size_t total =
            parked_blocks(list_block_free_thr) * sizeof(T) * CHUNK_T<T>::storage_units(thread_chunk_size());

        if constexpr (chunk_config::chunk_dynamic)
        {
            for (const auto& [size, ranks] : list_block_free_thr_by_size_)
            {
                total += parked_blocks(ranks) * sizeof(T) * CHUNK_T<T>::storage_units(size);
            }
        }
        return total;
    }

    /**
     * @brief Return the reuse pools' excess to the allocator, keeping a margin.
     *
     * @details The pools never shrink on their own. A buffer a chunk releases is
     *          parked for reuse, and the list grows to the high-water mark of
     *          simultaneously-free buffers and stays there. Measured on v1 dyn at
     *          64 threads, once the starting geometry was registered so the
     *          buffers were keyed at all: 29953 MB parked out of a 38523 MB
     *          footprint -- 78% of the process. 35 million double blocks at one
     *          geometry is not reuse, it is accumulation.
     *
     *          So: keep what the next materialisations plausibly need, hand the
     *          rest back. @p keep_per_thread is that working margin, per thread
     *          AND per geometry -- a cap, not a target. A list holding less is
     *          left alone and nothing is pre-allocated.
     *
     * @param keep_per_thread buffers to keep in each (geometry, rank) list
     * @return bytes returned to the allocator
     *
     * @warning std-alloc only, and a no-op otherwise. With the fast allocator a
     *          freed value slab is never munmap'd (see the op/value aliasing FIX
     *          in manager_chunk_inline.h), so handing buffers back there just
     *          moves them between the arena's own lists without releasing a
     *          page. Doing nothing is honest; doing it anyway would be pure cost.
     *
     * @warning Timing is the caller's business, and it matters: trim too eagerly
     *          and the next materialisation re-allocates what was just freed.
     *          The natural point is after a compression pass, when many buffers
     *          land in the pools at once -- the caller knows that, this does not.
     */
    static std::size_t trim_reuse_pools(std::size_t keep_per_thread) noexcept
    {
        if constexpr (chunk_config::use_fast_alloc)
        {
            (void)keep_per_thread;
            return 0;
        }
        else
        {
            std::size_t freed = 0;

            const auto trim_one = [&freed, keep_per_thread](auto& lst, std::size_t elem_count) noexcept
            {
                if (lst.size() <= keep_per_thread)
                {
                    return;
                }
                while (lst.size() > keep_per_thread)
                {
                    T* block = lst.back();
                    lst.pop_back();
                    CHUNK_LEDGER_DISCARD(block, "trim_reuse_pools");
                    svfree(block);
                    freed += CHUNK_T<T>::storage_units(elem_count) * sizeof(T);
                }
                // A vector that shrank keeps its capacity, so a pool that peaked
                // once holds on to the pointer array for good. Hand that back too.
                lst.shrink_to_fit();
            };

            for (auto& ptf : list_block_free_thr)
            {
                trim_one(ptf.list, thread_chunk_size());
            }

            if constexpr (chunk_config::chunk_dynamic)
            {
                for (auto& kv : list_block_free_thr_by_size_)
                {
                    for (auto& ptf : kv.second)
                    {
                        trim_one(ptf.list, kv.first);
                    }
                }
            }

            return freed;
        }
    }

    /// @brief Set the candidate-list length that triggers a compression pass.
    /// @details Decides when trigger_compression_if_needed() flags a rank's
    ///          candidate list, hence how often compression runs at all.
    static void set_size_max_number_of_chunk_to_check_compress(int v)
    {
        size_max_number_of_chunk_to_check_compress = v;
    }

    /**
     * @brief Drop from @p v every queued candidate that belongs to @p num_tab.
     *
     * @details The one place that knows a candidate tuple names its owning array
     *          in field <2>. The three remove_from_compress_list_*_by_num_tab
     *          below spelled this erase-remove out individually; only their
     *          TRAVERSAL genuinely differs (a flat per-rank vector for Chunk and
     *          Chunk2A, a per-rank map of second-value buckets for Chunk2), so
     *          that is all they keep.
     *
     * @tparam Vec any sequence of candidate tuples.
     * @param v       the sequence to filter, modified in place.
     * @param num_tab tab index whose candidates are removed.
     */
    template <typename Vec>
    static void erase_candidates_of_tab(Vec& v, int num_tab) noexcept
    {
        std::erase_if(v, [num_tab](const auto& tup) { return std::get<2>(tup) == num_tab; });
    }

    /// @brief Map key of list_block_free_thr_keyed_: (@p key << 32) | @p size.
    static constexpr std::uint64_t keyed_index(unsigned int key, unsigned int size) noexcept
    {
        return (static_cast<std::uint64_t>(key) << 32U) | size;
    }

    const int max_thr = CHUNK_PMP_THREAD_CAPACITY;

    /// @brief Chunk2: the second-value lists sweep_rank() drains, snapshotted
    ///        serially by sweep_prepare() as the per-type pass always did.
    std::vector<std::tuple<T, int, int>*> sweep_inf_arrays_;

    /// @brief Slab byte budget seen at the last register_chunk_size(); what
    ///        register_alloc_key() hands to the keyed allocator sets.
    static inline int registered_block_size_ = 0;

    // Serial counterparts of the *_thr stores below: no per-thread indirection.
    //
    // Declared in EVERY build -- empty and never touched in OpenMP ones -- for the
    // same reason as list_block_free_thr_by_size_ below: they are members of the
    // current instantiation, so the DISCARDED branch of an
    // `if constexpr (build_config::use_openmp)` still name-looks them up and they
    // have to exist to be named. Declaring them unconditionally is what lets the
    // accessors and the remove_from_compress_list_* helpers select with
    // `if constexpr`. The static cost is three empty containers
    // per (variant, type). Only the serial getters below expose them.
    static inline TChunk2CompMap<CHUNK_T, T> chunk2_to_compress_potentially;

    static inline TChunkComp<CHUNK_T, T> chunk_to_compress_potentially;

    static inline long somme_chunk_create{0};

    static inline int                                   size_buffer_reuse;
    static inline std::unordered_map<unsigned int, int> size_buffer_reuse_by_key_; ///< see get_size_buffer_reuse(key)

    static inline int size_max_number_of_chunk_to_check_compress;

    static inline std::vector<PerThreadFreeList> list_block_free_thr;
    // Multi-size: a separate reuse free-list per (size, thread). A reused block
    // must match the active chunk size, so blocks are never shared across sizes.
    // Written only at register_chunk_size (sequential); read per-alloc/free.
    //
    // Declared in EVERY build -- an empty map in static ones -- so the accessors
    // can select with `if constexpr (build_config::chunk_dynamic)`. A
    // discarded if-constexpr branch is still name-looked-up, and this
    // is a member of the current instantiation, so it has to exist to be named.
    // The static cost is one empty std::map per (variant, type), never touched.
    static inline std::map<unsigned int, std::vector<PerThreadFreeList>> list_block_free_thr_by_size_;
    /// @brief Reuse lists of the arrays with their own slab affinity: one
    ///        vector of per-rank lists per (affinity key, size), key != 0.
    ///        Key 0 -- the rank's shared pool -- stays on
    ///        list_block_free_thr_by_size_ / list_block_free_thr, untouched.
    ///        A list (thr, K, size) only ever holds blocks from K's slabs: an
    ///        own-rank free parks its block here and the same rank reuses it
    ///        for K; a foreign-rank block parks under ITS key (slab->owner_key)
    ///        and is either reused by this rank for that same array or routed
    ///        to its owner by the drain. Nothing has to be filtered at pop.
    static inline std::map<std::uint64_t, std::vector<PerThreadFreeList>> list_block_free_thr_keyed_;

    /**
     * @brief Teardown hand-off buffer: pointers cleared for release, by rank.
     *
     * @details STATIC on purpose. The teardown phases (clear_list_block_free_*)
     *          are called by every thread of one parallel region opened in
     *          AllTypeChunk, each on its own buckets or slice; a local would
     *          be private to each thread, so a bucket screened by one thread
     *          would be invisible to the one that frees it.
     *
     *          Only live during teardown; cleared as it is consumed.
     */
    static inline std::vector<std::vector<T*>> teardown_to_free_;

    /**
     * @brief Cross-bucket repeats found by clear_list_block_free_dedup(), one
     *        list per address slice.
     * @details Written by the dedup phase (each slice its own list, no
     *          sharing), read by the free phase after the barrier. Empty on any
     *          sane run.
     */
    static inline std::vector<std::vector<internal::TeardownRepeat<T>>> teardown_repeats_;

  public:
    /**
     * @brief Captures the thread capacity once, at construction.
     *
     * @note max_thr is the thread capacity (CHUNK_PMP_THREAD_CAPACITY), fixed
     *       for the process: every per-thread container this manager sizes
     *       holds a slot for every rank that may run, including a team raised
     *       after the manager was created, up to that capacity.
     */
    ManagerChunk()           = default;
    ~ManagerChunk() override = default;

    ManagerChunk(const ManagerChunk& ref) = delete;

    void resize_chunk_to_compress_potentially() const override;

    // Where the build-dependent overrides went, so a reader looking for one here
    // is not left thinking it does not exist:
    //   register_chunk_size()                 -> ManagerChunkDynImpl
    //   the allocator / free-list surface     -> ManagerChunkParallelImpl

    /// @brief Recompute how many buffers each thread may keep parked for reuse.
    static void compute_size_buffer_reuse();

    /// @brief Recompute the candidate-list length that triggers a compression pass.
    static void compute_size_max_number_of_chunk_to_check_compress();

    /// @brief Add @p count to the running total of chunks created (statistics).
    static void increm_somme_chunk_create(int count) noexcept
    {
        somme_chunk_create += count;
    }
    /// @brief Subtract @p count from the running total of chunks created (statistics).
    static void decrem_somme_chunk_create(int count) noexcept
    {
        somme_chunk_create -= count;
    }
    /// @brief Chunks created minus chunks destroyed -- what the compression
    ///        threshold is computed from, so it must track the chunks alive.
    [[nodiscard]] static long get_somme_chunk_create() noexcept
    {
        return somme_chunk_create;
    }

    /// @brief Buffers parked in @p ranks, one reuse list per rank.
    [[nodiscard]] static std::size_t parked_blocks(const std::vector<PerThreadFreeList>& ranks) noexcept
    {
        return std::ranges::fold_left(ranks | std::views::transform([](const auto& ptf) { return ptf.list.size(); }),
                                      0uz, std::plus{});
    }

    /// @brief Buffers parked for reuse, every rank, every geometry.
    [[nodiscard]] static std::size_t reuse_pool_blocks() noexcept
    {
        std::size_t blocks = parked_blocks(list_block_free_thr);
        if constexpr (chunk_config::chunk_dynamic)
        {
            for (const auto& ranks : list_block_free_thr_by_size_ | std::views::values)
            {
                blocks += parked_blocks(ranks);
            }
        }
        return blocks;
    }

    /// @brief Buffers parked for reuse at geometry @p size, every rank (per-size
    ///        lists only: the legacy list is not keyed by geometry).
    [[nodiscard]] static std::size_t reuse_pool_blocks_at([[maybe_unused]] unsigned int size) noexcept
    {
        std::size_t blocks = 0;
        if constexpr (chunk_config::chunk_dynamic)
        {
            const auto it = list_block_free_thr_by_size_.find(size);
            if (it != list_block_free_thr_by_size_.end())
            {
                blocks = parked_blocks(it->second);
            }
        }
        return blocks;
    }
    /// @brief Maximum number of buffers a thread keeps parked for reuse.
    [[nodiscard]] static int get_size_buffer_reuse() noexcept
    {
        return size_buffer_reuse;
    }
    /// @brief Reuse budget of the list (thr, key): for an array with its own
    ///        affinity key, its own chunk count over (factor_div x threads) --
    ///        the share of THAT array a rank can expect to recycle -- set at
    ///        the array's creation by set_size_buffer_reuse(key, ...). Key 0
    ///        and unknown keys fall back to the global budget.
    [[nodiscard]] static int get_size_buffer_reuse(unsigned int key) noexcept
    {
        if (key != 0)
        {
            const auto it = size_buffer_reuse_by_key_.find(key);
            if (it != size_buffer_reuse_by_key_.end())
            {
                return it->second;
            }
        }
        return size_buffer_reuse;
    }

    /// @brief Everything an array with its own affinity key needs, in one call
    ///        (sequential, from the array's rebuild_storage once nb_chunk is
    ///        known): its allocator set at (key, size), its per-rank reuse
    ///        lists, and their budget -- nb_chunk / (factor_div x threads),
    ///        the array's own share per rank rather than the process-wide
    ///        somme_chunk_create the global budget derives from.
    /// @note Dynamic OpenMP fast-alloc builds only; a no-op elsewhere (the
    ///       static registry has no key dimension yet).
    /// @return true when the keyed allocator set, lists and budget now exist;
    ///         false when they could not be created (no slab budget yet, or a
    ///         build without the dynamic registry) -- the caller must then stay
    ///         on key 0 rather than route to a set that does not exist.
    [[nodiscard]] static bool register_alloc_key(unsigned int key, unsigned int size, int nb_chunk);

    /// @brief Reuse list of (thr_rank, key) at the ambient geometry. Key 0 is
    ///        get_list_block_free_thr(thr_rank); an unregistered key falls back
    ///        to the shared pool rather than indexing blindly.
    static std::vector<T*>& get_list_block_free_thr(int thr_rank, unsigned int key) noexcept
    {
        if (key != 0)
        {
            const auto it = list_block_free_thr_keyed_.find(keyed_index(key, thread_chunk_size()));
            if (it != list_block_free_thr_keyed_.end() && static_cast<size_t>(thr_rank) < it->second.size())
            {
                return it->second[thr_rank].list;
            }
        }
        return get_list_block_free_thr(thr_rank);
    }
    /// @brief Number of blocks parked in rank @p thr_rank's reuse list for
    ///        @p key at the ambient geometry.
    [[nodiscard]] static size_t get_list_block_free_thr_size(int thr_rank, unsigned int key) noexcept
    {
        if (key != 0)
        {
            const auto it = list_block_free_thr_keyed_.find(keyed_index(key, thread_chunk_size()));
            if (it != list_block_free_thr_keyed_.end() && static_cast<size_t>(thr_rank) < it->second.size())
            {
                return it->second[thr_rank].list.size();
            }
        }
        return get_list_block_free_thr_size(thr_rank);
    }
    /// @brief Number of buffers currently parked for @p thr_rank at the ambient geometry.
    [[nodiscard]] static size_t get_list_block_free_thr_size(int thr_rank) noexcept
    {
        // Bounds-checked on purpose: every reuse container is only indexable
        // once resize_all_list_block_free_thr() has sized it to max_thr, and that is
        // NOT guaranteed at every call site. In fast-alloc builds nothing ever
        // parks in the legacy store, so it can still be empty when the first
        // chunk is allocated; in dynamic builds a geometry born from a re-chunk
        // has no per-size list, so get_list_block_free_thr() falls back to that
        // same (possibly empty) legacy store.
        //
        // Reading the size through operator[] on an empty vector is silent UB
        // in a normal build and aborts under _GLIBCXX_ASSERTIONS with
        // "Assertion '__n < this->size()' failed" -- which is exactly how this
        // surfaced from ChunkAllocPolicy::allocate_chunk_internal. Answering 0
        // for a slot that does not exist is both correct (no buffer is
        // reusable) and what every caller already handles.
        if (static_cast<size_t>(thr_rank) >= list_block_free_thr.size())
        {
            if constexpr (chunk_config::chunk_dynamic)
            {
                const auto it = list_block_free_thr_by_size_.find(thread_chunk_size());
                if (it != list_block_free_thr_by_size_.end() && static_cast<size_t>(thr_rank) < it->second.size())
                {
                    return it->second[thr_rank].list.size();
                }
            }
            return 0;
        }
        return get_list_block_free_thr(thr_rank).size();
    }
    /// @brief The whole legacy (unsized) per-thread reuse list container.
    static std::vector<PerThreadFreeList>& get_list_block_free_thr() noexcept
    {
        return list_block_free_thr;
    }

    /**
     * @brief Reuse list for @p thr_rank at the AMBIENT geometry.
     * @param thr_rank owning thread's rank
     * @return the per-thread reuse list for thread_chunk_size()
     * @warning Correct only when the ambient size is the buffer's own geometry.
     *          At teardown or recycle, use get_list_block_free_thr_for_size().
     */
    static std::vector<T*>& get_list_block_free_thr(int thr_rank) noexcept
    {
        if constexpr (chunk_config::chunk_dynamic)
        {
            const unsigned int sz = thread_chunk_size();
            const auto         it = list_block_free_thr_by_size_.find(sz);
            // The per-size vector is sized at register_runtime_chunk_size()
            // time; a thread rank beyond it would be the same out-of-bounds
            // operator[] as on the legacy store, so fall back rather than
            // index blindly.
            if (it != list_block_free_thr_by_size_.end() && static_cast<size_t>(thr_rank) < it->second.size())
            {
                return it->second[thr_rank].list;
            }
            // fallback: legacy single store (single-size / unregistered)
        }
        return list_block_free_thr[thr_rank].list;
    }

    /**
     * @brief Reuse list for an EXPLICIT geometry, ignoring the ambient size.
     *
     * @details A buffer must be parked in, and popped from, the list for ITS OWN
     *          geometry. Parking a size-A buffer while the ambient
     *          thread_chunk_size() is B lands it in list[B]; a later size-B pop
     *          then reuses a physically-A buffer, and init_block()'s memset at B
     *          runs past its end -- an ASan heap-buffer-overflow.
     *
     * @param thr_rank owning thread's rank
     * @param sz       the buffer's real geometry
     * @return the per-thread reuse list for @p sz (the legacy single list in
     *         single-size builds, or when @p sz was never registered)
     * @note Callers that know the buffer's real geometry -- teardown, recycle --
     *       must use this rather than get_list_block_free_thr().
     */
    static std::vector<T*>& get_list_block_free_thr_for_size([[maybe_unused]] int          thr_rank,
                                                             [[maybe_unused]] unsigned int sz)
    {
        if constexpr (chunk_config::chunk_dynamic)
        {
            const auto it = list_block_free_thr_by_size_.find(sz);
            if (it != list_block_free_thr_by_size_.end())
            {
                return it->second[thr_rank].list;
            }
        }

        return list_block_free_thr[thr_rank].list;
    }

    /**
     * @brief Empty the by-geometry reuse list for @p size, across all threads
     *        (dynamic builds; a no-op otherwise).
     *
     * @details Called by rechunk_to() in all three variants before allocating at
     *          a new geometry: a buffer physically sized for a DIFFERENT
     *          geometry left in this list would overflow init_block()'s memset
     *          once popped at @p size.
     *
     *          How the buffers are released depends on the allocator. Under
     *          std-alloc they were svcalloc'd, so each is svfree'd (they are
     *          PARKED in the ledger, hence the DISCARD transitions). Under
     *          fast-alloc they belong to immortal arenas and are returned to
     *          their OWNING thread's slab -- free_elt_pow2() is only safe on an
     *          owned block, so foreign ones stay parked as a bounded residual.
     *
     * @param size geometry whose reuse list is drained
     * @warning Only the per-size list is drained, looked up directly. It
     *          deliberately does NOT fall back to the unsized legacy list: in a
     *          serial build that list is never resized, so indexing it would be
     *          out of bounds. Teardown drains the legacy list separately, in
     *          clear_list_block_free_thr().
     */
    static void drain_reuse_list_for_size([[maybe_unused]] unsigned int size) noexcept
    {
        // A static on ManagerChunk<CHUNK_T, T>: it drains ONE element type --
        // the one belonging to the re-chunked array. The other types keep
        // their pools regardless.
        if constexpr (chunk_config::chunk_dynamic)
        {
            // Drain ONLY the per-size reuse list for `size`, looked up directly. If it
            // was never registered (e.g. the array's INITIAL geometry, set at
            // construction without going through set_chunk_size/register_runtime_chunk_size),
            // nothing was ever parked in a per-size list for it -> return. Do NOT fall
            // back to the unsized legacy list via get_list_block_free_thr_for_size:
            // in a non-OMP build that list is not resized, so list_block_free_thr[thr]
            // is an out-of-bounds operator[] -> SIGSEGV/assert at the first re-chunk.
            // Teardown's clear_list_block_free_thr drains the legacy list separately.
            // One routine per list set: the shared per-size list, then every
            // keyed list (per-array affinity) at that same size -- a re-chunk
            // or a destruction leaves its old-geometry blocks in the array's
            // own lists, which must empty for the array's slabs to empty.
            const auto drain_one = [&](std::vector<PerThreadFreeList>& per_thr)
            {
                const auto n_thr = static_cast<int>(per_thr.size()); // iterate by ITS size, never max_thr

                if constexpr (!chunk_config::use_fast_alloc)
                {
                    // std-alloc: parked buffers were svcalloc'd -> svfree each.
                    for (int thr = 0; thr < n_thr; ++thr)
                    {
                        auto& lst = per_thr[thr].list;
                        for (T* b : lst)
                        {
                            CHUNK_LEDGER_DISCARD(b, "rechunk_to.drain_reuse");
                            svfree(b);
                        }
                        lst.clear();
                    }
                }
                else
                {
                    // fast-alloc: free each block through its OWNING thread's slab
                    // (free_elt_pow2 is only safe on an owned block). Owner-routed
                    // drain, in three passes.
                    //
                    // The previous shape -- a parallel pass where rank T offered its
                    // own list to its own allocator, then a SERIAL owner-routing
                    // sweep offering every leftover block to every rank in turn --
                    // was calibrated on 7553 orphans (3.9 MB). It is quadratic:
                    // is_free_able_elt() walks the allocator's slab chains (linear in
                    // slabs), the sweep repeats it for every rank, and after a v1
                    // re-chunk of a 10^9-double array most of the 11M parked blocks
                    // are orphans (clear_st parks in the FREEING thread's list, the
                    // refill reads spans owned by other ranks). perf at 64 threads:
                    // 39% in the OpenMP barrier, the drain on one thread underneath.
                    //
                    // Ownership is not ambiguous, only expensive to ask the allocator
                    // for: every slab is reachable from exactly one rank's chains,
                    // and addr_to_block() maps a pointer to its slab in O(1). So:
                    //   1. serial: allocator instance -> rank (n_thr entries); a slab
                    //      names its instance in its header (TCAllocator::owner);
                    //   2. parallel over q: rank q walks ITS list, buckets each
                    //      block under its owner (row q of a q x r matrix -- no two
                    //      ranks write the same cell), keeps what has no known owner;
                    //   3. parallel over r: rank r frees every bucket[*][r] into
                    //      ITS OWN allocator, the only mutation of a slab chain, and
                    //      always by the rank that owns it.
                    // Blocks whose slab is unknown to every chain (bypass_slabs(),
                    // or a slab already torn down) stay parked as before: an
                    // unreachable block is worse than a retained one.
                    using alloc_t = std::remove_reference_t<decltype(CHUNK_T<T>::get_chunk_allocator(0))>;
                    // Every slab names its owning allocator instance AND that
                    // instance's rank (TCAllocator::owner / owner_rank, written at
                    // creation), whatever affinity key the instance serves. So a
                    // block routes on two field reads: rank r's bucket, then
                    // owner->free_elt_pow2 on rank r. No map, no chain walk, and
                    // the keyed allocator sets need no special case.
                    alloc_t* geometry = nullptr; // any instance of this size: slab_of() needs its geometry only
                    {
                        const unsigned int saved_sz = shrink_array::thread_chunk_size();
                        shrink_array::set_thread_chunk_size(size);
                        geometry = &CHUNK_T<T>::get_chunk_allocator(0, 0u);
                        shrink_array::set_thread_chunk_size(saved_sz);
                    }
                    if (geometry != nullptr)
                    {
                        std::vector<std::vector<std::vector<T*>>> bucket(
                            static_cast<std::size_t>(n_thr),
                            std::vector<std::vector<T*>>(static_cast<std::size_t>(n_thr)));

                        const auto sort_own_list = [&](int rank) noexcept
                        {
                            if (rank < 0 || rank >= n_thr)
                            {
                                return;
                            }
                            auto& lst = per_thr[rank].list;
                            if (lst.empty())
                            {
                                return;
                            }
                            auto&       row  = bucket[static_cast<std::size_t>(rank)];
                            std::size_t kept = 0;
                            for (std::size_t i = 0; i < lst.size(); ++i)
                            {
                                T*          block = lst[i];
                                const auto* slab  = geometry->slab_of(block);
                                if (slab && slab->owner && slab->owner_rank >= 0 && slab->owner_rank < n_thr)
                                {
                                    row[static_cast<std::size_t>(slab->owner_rank)].push_back(block);
                                }
                                else
                                {
                                    lst[kept++] = block; // no owner claims it: stays parked
                                }
                            }
                            lst.resize(kept);
                        };
                        const auto free_own_buckets = [&](int rank) noexcept
                        {
                            if (rank < 0 || rank >= n_thr)
                            {
                                return;
                            }
                            // Route to the `size` allocator: get_chunk_allocator keys on
                            // the per-thread active size. Freeing a size-`size` block
                            // through the wrong-size allocator corrupts that slab's
                            // free-list -> SIGSEGV walking it at teardown.
                            // Every block in bucket[*][r] belongs to an instance of
                            // rank r (its slab said so); free it through THAT instance.
                            for (int q = 0; q < n_thr; ++q)
                            {
                                auto& cell = bucket[static_cast<std::size_t>(q)][static_cast<std::size_t>(rank)];
                                for (T* b : cell)
                                {
                                    CHUNK_FREEPATH_INC(freed);
                                    CHUNK_LEDGER_DISCARD(b, "rechunk_to.drain_reuse.owner");
                                    auto* slab  = geometry->slab_of(b);
                                    auto* owner = static_cast<alloc_t*>(const_cast<void*>(slab->owner));
                                    owner->free_elt_pow2(b, slab);
                                }
                                cell.clear();
                                cell.shrink_to_fit();
                            }
                        };

                        if constexpr (chunk_config::not_use_compression_with_ompt)
                        {
// Serial call sites (the refill's own region has closed); a
// no-op region in a non-OMP build. Skipped under OMPT, where
// opening a region would fire compression callbacks that
// must not run here.
#pragma omp parallel
                            {
                                sort_own_list(CHUNK_PMP_THREAD_RANK);
                            }
                            // Ranks a smaller team did not cover (cheap when empty).
                            for (int q = 0; q < n_thr; ++q)
                                sort_own_list(q);
#pragma omp parallel
                            {
                                free_own_buckets(CHUNK_PMP_THREAD_RANK);
                            }
                            for (int r = 0; r < n_thr; ++r)
                                free_own_buckets(r);
                        }
                        else
                        {
                            for (int q = 0; q < n_thr; ++q)
                                sort_own_list(q);
                            for (int r = 0; r < n_thr; ++r)
                                free_own_buckets(r);
                        }
                    }
                }
            }; // drain_one
            if (const auto it = list_block_free_thr_by_size_.find(size); it != list_block_free_thr_by_size_.end())
            {
                drain_one(it->second);
            }
            for (auto& kv : list_block_free_thr_keyed_)
            {
                if (static_cast<unsigned int>(kv.first & 0xffffffffu) == size)
                {
                    drain_one(kv.second);
                }
            }
        }
    }

    /// @brief Candidate-list length that triggers a compression pass.
    [[nodiscard]] static int get_size_max_number_of_chunk_to_check_compress() noexcept
    {
        return size_max_number_of_chunk_to_check_compress;
    }
    // Serial-only overload, and NOT merely unused under OpenMP: the store it
    // returns is declared in every build, so this WOULD compile there -- but
    // it would hand back a permanently empty list while the real candidates
    // sit in the per-thread store. Not existing at all is the point, which the
    // constraint expresses without a preprocessor guard: under OpenMP the
    // overload is not viable, and a call to it fails to compile.
    /// @brief The Chunk2A candidate list (serial builds: a single one).
    static auto& get_chunk2A_to_compress_potentially() noexcept
        requires(!build_config::use_openmp)
    {
        return chunk2A_to_compress_potentially;
    }
    /// @brief Rank @p thr's Chunk2A candidate list (the single one in a
    ///        serial build).
    static auto& get_chunk2A_to_compress_potentially([[maybe_unused]] int thr) noexcept
    {
        if constexpr (build_config::use_openmp)
        {
            return chunk2A_to_compress_potentially_thr[thr];
        }
        else
        {
            return chunk2A_to_compress_potentially;
        }
    }

    /// @brief Is no Chunk2A chunk queued for compression, on any thread?
    [[nodiscard]] static bool is_chunk2A_to_compress_potentially_empty() noexcept
    {
        if constexpr (build_config::use_openmp)
        {
            return chunk2A_to_compress_potentially_thr.empty();
        }
        else
        {
            return chunk2A_to_compress_potentially.empty();
        }
    }

    /**
     * @brief Drop every queued Chunk2A candidate pointing into [@p lo, @p hi).
     *
     * @details Teardown removal by CHUNK-POINTER RANGE, the robust counterpart of
     *          remove_from_compress_list_chunk2a_by_num_tab(). Candidate tuples
     *          hold &chunk_list[j] pointers, so filtering on the array's own
     *          chunk_list span is num_tab-agnostic and guarantees no pointer into
     *          this array survives its teardown.
     *
     *          The by-num_tab form can MISS entries whose num_tab was shifted by
     *          a rechunk identity swap; the leftover Chunk2A* then dangles once
     *          chunk_list is destroyed, and a later compression pass dereferences
     *          it and double-frees its buffer (a corrupted doubly-linked list
     *          under std-alloc).
     *
     * @param lo first address of the array's chunk_list span
     * @param hi one past its last address
     * @note In plain fast-block-compression builds (GNU, no OMPT) this cleanup is
     *       the ONLY one: the built-in vec_tup.clear() runs only under OMPT.
     */
    static void remove_from_compress_list_chunk2a_by_ptr_range(const void* lo, const void* hi) noexcept
    {
        const auto in_range = [lo, hi](const auto& tup)
        {
            const void* ptr = static_cast<const void*>(std::get<0>(tup));
            return ptr >= lo && ptr < hi;
        };
        if constexpr (build_config::use_openmp)
        {
            const auto nthr = static_cast<int>(chunk2A_to_compress_potentially_thr.size());
            for (int thr = 0; thr < nthr; ++thr)
            {
                std::erase_if(chunk2A_to_compress_potentially_thr[thr], in_range);
            }
        }
        else
        {
            std::erase_if(chunk2A_to_compress_potentially, in_range);
        }
    }

    /**
     * @brief Drop every queued Chunk2A candidate of array @p num_tab, in every
     *        rank's list (ArrayChunk2A::clear_core and rechunk_to).
     * @param num_tab tab index whose candidates are removed
     * @note Filtering by num_tab rather than clearing wholesale keeps other live
     *       arrays' candidates intact. Prefer
     *       remove_from_compress_list_chunk2a_by_ptr_range() at teardown: see its
     *       documentation for the case this form can miss.
     */
    static void remove_from_compress_list_chunk2a_by_num_tab(int num_tab) noexcept
    {
        if constexpr (build_config::use_openmp)
        {
            const auto nthr = static_cast<int>(chunk2A_to_compress_potentially_thr.size());
            for (int thr = 0; thr < nthr; ++thr)
            {
                erase_candidates_of_tab(chunk2A_to_compress_potentially_thr[thr], num_tab);
            }
        }
        else
        {
            erase_candidates_of_tab(chunk2A_to_compress_potentially, num_tab);
        }
    }

    /**
     * @brief Chunk2 analogue: drop every queued Chunk2 candidate for @p num_tab.
     *
     * @details Walks all threads and all second-value buckets of
     *          map_chunk2_to_compress_potentially_thr and removes the matching
     *          tuples WITHOUT compressing anything. Called from
     *          ArrayChunk2::clear_core() so that a torn-down or re-chunked
     *          array's Chunk2* cannot dangle in the shared store and be walked by
     *          a later compression pass.
     *
     * @param num_tab tab index whose candidates are removed
     * @note OpenMP builds only. The serial store is a map<T, vector<Chunk2*>>,
     *       which carries no num_tab; it is drained instead by AllTypeChunk's
     *       terminal sweep in clear_core().
     */
    static void remove_from_compress_list_chunk2_by_num_tab([[maybe_unused]] int num_tab) noexcept
    {
        // The store's type is dependent (TChunk2CompMap is a per-thread vector of
        // maps under OpenMP, a plain map otherwise), so the serial branch is
        // discarded before instantiation.
        if constexpr (build_config::use_openmp)
        {
            const auto nthr = static_cast<int>(map_chunk2_to_compress_potentially_thr.size());
            for (int thr = 0; thr < nthr; ++thr)
            {
                // Chunk2 alone has this extra level: its candidates are bucketed
                // by second value, so every bucket of every rank must be walked.
                for (const auto& kv : map_chunk2_to_compress_potentially_thr[thr])
                {
                    const auto& sp = kv.second;
                    if (!sp)
                    {
                        continue;
                    }
                    erase_candidates_of_tab(*sp, num_tab);
                }
            }
        }
    }

    /// @brief Drop every queued Chunk (v1) candidate of array @p num_tab
    ///        (ArrayChunk::clear_core and rechunk_to).
    // Plain-Chunk (v1) analogue: remove every queued Chunk candidate belonging to
    // `num_tab` from chunk_to_compress_potentially_thr (a flat per-thread vector
    // of (Chunk*, num_chunk, num_tab) tuples — v1 has no second_value keying).
    // Used by ArrayChunk::clear_core and rechunk_to. fill_list_to_compress scans
    // at the AMBIENT geometry (alloc_chunk_count()), so a candidate left in the
    // list across a geometry change is later scanned at the wrong size
    // (over-scan on shrink -> heap-overflow, under-scan on grow -> mis-compress);
    // removing by num_tab avoids that without forcing compression. No sweep runs
    // in v1's clear(), but a mid-refill make_block_compression can, hence the
    // rechunk removals. Filtering by num_tab keeps other live arrays intact.
    static void remove_from_compress_list_chunk_by_num_tab(int num_tab) noexcept
    {
        if constexpr (build_config::use_openmp)
        {
            const auto nthr = static_cast<int>(chunk_to_compress_potentially_thr.size());
            for (int thr = 0; thr < nthr; ++thr)
            {
                erase_candidates_of_tab(chunk_to_compress_potentially_thr[thr], num_tab);
            }
        }
        else
        {
            erase_candidates_of_tab(chunk_to_compress_potentially, num_tab);
        }
    }

    static inline TChunk2CompMap<CHUNK_T, T>  map_chunk2_to_compress_potentially_thr;
    static inline TChunk2CompThr<CHUNK_T, T>  chunk2_to_compress_potentially_thr;
    static inline TChunk2ACompThr<CHUNK_T, T> chunk2A_to_compress_potentially_thr;

    static inline TChunkCompThr<CHUNK_T, T> chunk_to_compress_potentially_thr;

    static inline TChunk2AComp<CHUNK_T, T> chunk2A_to_compress_potentially;
};

} // namespace shrink_array

#include "chunk/internal/manager_chunk_inline.h" // the template bodies