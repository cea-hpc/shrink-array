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
#include <utility>       // std::pair (sweep_empty_slabs)
#include <atomic>        // TeamDrainScratch's rank flags
#include <memory>        // std::unique_ptr (the same flags)
#include <map>           // std::map
#include <vector>        // std::vector
#include <tuple>         // std::tuple, std::get
#include <unordered_map> // std::unordered_map
#include <cstdint>       // pool keys (drain_reuse_list_for_size, keyed lists)
#include <type_traits>   // remove_reference_t / remove_pointer_t (same)

#include "chunk/bit.h"                        // is_bool_like_v: bool and Bit share the bool path
#include "chunk/zero_chunk_def.h"             // ChunkDef::size_min_buffer_reuse
#include "chunk/zero_any_chunk_def.h"         // zero_any::ChunkDef: the reuse budget constants
#include "chunk/config.h"                     // build_config::threaded, chunk_config::chunk_dynamic
#include "chunk/internal/chunk_size_policy.h" // thread_chunk_size, set_thread_chunk_size
#include "chunk/log.h"                        // CHUNK_LOG_ALWAYS (pool breakdown)
#include "chunk/internal/TChunk.h"            // TChunkComp, TChunk2CompMap, TChunk2AComp
#include "chunk/PMP_CHUNK.h"                  // CHUNK_PMP_THREAD_CAPACITY (max_thr), pmp::team
#include "chunk/internal/pool_ledger.h"       // CHUNK_LEDGER_DISCARD (drain_reuse_list_for_size)
#include "chunk/type_name.h"                  // type_name<T> (pool breakdown)
#include "fast_allocators/internal/svalloc.h" // svfree (drain_reuse_list_for_size)

#include "chunk/internal/freepath_counters.h" // CHUNK_FREEPATH_INC (arena=info)
#include "chunk/internal/immortal.h"          // Immortal / LazyImmortal: statics safe for global arrays
#include "chunk/internal/second_value_info.h" // SecondValueInfo (InfArrayT)

namespace shrink_array
{
// Forward declarations breaking the include cycle chunk_func.h -> zero_chunk.h -> zero_proxy.h ->
// THIS file -> chunk_func.h: when chunk_func.h is the entry point, its
// templates are not yet declared here (#pragma once skips it), and the explicit
// load_balance_list_block_free_thr<T>() calls below need a declaration. Defined in chunk_func.h.
template <typename T>
bool need_load_balance_list_block_free_thr();
template <typename T>
void load_balance_list_block_free_thr();

// Forward declaration: only named in contexts that depend on T. The definition
// (chunk/internal/compression_manager.h) comes with manager_chunk_inline.h.
template <typename T, typename ChunkT>
struct CompressionManager;

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
 * @details Selected by build_config constants rather than the preprocessor -- the
 *          same policy split the chunk classes use (ChunkAllocPolicy,
 *          ChunkDeferredOpsPolicy, DeferredReadGate).
 *
 * @tparam IsDynamic Whether runtime geometry is compiled in.
 */
template <bool IsDynamic = build_config::chunk_dynamic>
struct ManagerChunkDynInterface; // primary left undefined: only the 2 specializations exist

/// @brief Dynamic-geometry builds: register_chunk_size().
template <>
struct ManagerChunkDynInterface<true>
{
    ManagerChunkDynInterface()          = default;
    virtual ~ManagerChunkDynInterface() = default;
    // An interface: never copied or moved (no slicing).
    ManagerChunkDynInterface(const ManagerChunkDynInterface&)            = delete;
    ManagerChunkDynInterface& operator=(const ManagerChunkDynInterface&) = delete;
    ManagerChunkDynInterface(ManagerChunkDynInterface&&)                 = delete;
    ManagerChunkDynInterface& operator=(ManagerChunkDynInterface&&)      = delete;

    /// @brief Declare a runtime chunk geometry to this type's allocators.
    virtual void register_chunk_size(unsigned int size, int block_size) = 0;
};

/// @brief Static-geometry builds: nothing to declare.
template <>
struct ManagerChunkDynInterface<false>
{
    ManagerChunkDynInterface()          = default;
    virtual ~ManagerChunkDynInterface() = default;
    // An interface: never copied or moved (no slicing).
    ManagerChunkDynInterface(const ManagerChunkDynInterface&)            = delete;
    ManagerChunkDynInterface& operator=(const ManagerChunkDynInterface&) = delete;
    ManagerChunkDynInterface(ManagerChunkDynInterface&&)                 = delete;
    ManagerChunkDynInterface& operator=(ManagerChunkDynInterface&&)      = delete;
};

/**
 * @brief The allocator / free-list half of the manager interface.
 *
 * @details The two modes offer DIFFERENT surfaces (OpenMP: per-thread allocators,
 *          reuse lists, deferred ops; serial: one allocator), hence two
 *          specializations rather than one set with no-ops.
 *
 * @tparam Threaded Whether the parallel surface is the one compiled in.
 */
template <bool Threaded = build_config::threaded>
struct ManagerChunkParallelInterface; // primary left undefined: only the 2 specializations exist

/// @brief OpenMP builds: per-thread allocators, reuse lists and deferred ops.
template <>
struct ManagerChunkParallelInterface<true>
{
    ManagerChunkParallelInterface()          = default;
    virtual ~ManagerChunkParallelInterface() = default;
    // An interface: never copied or moved (no slicing).
    ManagerChunkParallelInterface(const ManagerChunkParallelInterface&)            = delete;
    ManagerChunkParallelInterface& operator=(const ManagerChunkParallelInterface&) = delete;
    ManagerChunkParallelInterface(ManagerChunkParallelInterface&&)                 = delete;
    ManagerChunkParallelInterface& operator=(ManagerChunkParallelInterface&&)      = delete;

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
    ManagerChunkParallelInterface()          = default;
    virtual ~ManagerChunkParallelInterface() = default;
    // An interface: never copied or moved (no slicing).
    ManagerChunkParallelInterface(const ManagerChunkParallelInterface&)            = delete;
    ManagerChunkParallelInterface& operator=(const ManagerChunkParallelInterface&) = delete;
    ManagerChunkParallelInterface(ManagerChunkParallelInterface&&)                 = delete;
    ManagerChunkParallelInterface& operator=(ManagerChunkParallelInterface&&)      = delete;

    /// @brief Set the single allocator's geometry.
    virtual void init_allocator(int block_size) const = 0;
    /// @brief Destroy the single allocator (teardown).
    virtual void delete_chunk_allocator() = 0;
};

/**
 * @brief Type-erased interface to a per-(variant, element-type) chunk manager.
 *
 * @details One ManagerChunk class exists per (variant, type); AllTypeChunk, which
 *          acts on all of them (allocator init, compression passes, teardown),
 *          holds ManagerChunkInt pointers and dispatches through these virtuals.
 *          The build-dependent methods come from the two interface policies above.
 */
struct ManagerChunkInt : public virtual ManagerChunkDynInterface<>, public virtual ManagerChunkParallelInterface<>
{
    ManagerChunkInt()           = default;
    ~ManagerChunkInt() override = default;
    // An interface: never copied or moved (no slicing).
    ManagerChunkInt(const ManagerChunkInt&)            = delete;
    ManagerChunkInt& operator=(const ManagerChunkInt&) = delete;
    ManagerChunkInt(ManagerChunkInt&&)                 = delete;
    ManagerChunkInt& operator=(ManagerChunkInt&&)      = delete;

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

    /// @brief Bytes parked in this manager's reuse pools (printed with CHUNK_STATS).
    /// @details Every build: the reuse pools are not OpenMP-only.
    [[nodiscard]] virtual std::size_t reuse_pool_bytes_int() const = 0;
    /// @brief Return the pools' excess; see ManagerChunk::trim_reuse_pools.
    virtual std::size_t trim_reuse_pools_int(std::size_t keep_per_thread) = 0;
    /// @brief Print the pools' breakdown by geometry; see
    ///        ManagerChunk::print_reuse_pool_breakdown.
    virtual void print_reuse_pool_breakdown_int(const char* label) const = 0;

    // The build-dependent halves live in the three interface policies above.
};

/**
 * @brief The CHUNK_DYNAMIC-only half of the manager IMPLEMENTATION.
 *
 * @details Carries the override, because an `override` of a virtual that exists
 *          in one build only cannot be written conditionally in ManagerChunk.
 *          Inherits the interface VIRTUALLY, to share the one ManagerChunkInt
 *          brings in. The body reaches ManagerChunk's members (max_thr is
 *          per-instance) through a CRTP down-cast.
 */
template <template <typename> typename ChunkT, typename T, bool IsDynamic = build_config::chunk_dynamic>
struct ManagerChunkDynImpl; // primary left undefined: only the 2 specializations exist

/// @brief Dynamic-geometry builds: implements register_chunk_size().
template <template <typename> typename ChunkT, typename T>
struct ManagerChunkDynImpl<ChunkT, T, true> : public virtual ManagerChunkDynInterface<true>
{
    void register_chunk_size(unsigned int size, int block_size) override;
};

/// @brief Static-geometry builds: nothing to implement.
template <template <typename> typename ChunkT, typename T>
struct ManagerChunkDynImpl<ChunkT, T, false> : public virtual ManagerChunkDynInterface<false>
{
};

/**
 * @brief The allocator / free-list half of the manager IMPLEMENTATION.
 *
 * @details Carries the mode-specific overrides; @see ManagerChunkDynImpl. The
 *          bodies reach ManagerChunk's state through `mgr` (statics) and a CRTP
 *          down-cast (max_thr); ManagerChunk befriends the policies for that.
 */
template <template <typename> class ChunkT, typename T, bool Threaded = build_config::threaded>
class ManagerChunkParallelImpl; // primary left undefined: only the 2 specializations exist

/// @brief OpenMP builds: the per-rank allocator and free-list overrides.
template <template <typename> class ChunkT, typename T>
class ManagerChunkParallelImpl<ChunkT, T, true> : public virtual ManagerChunkParallelInterface<true>
{
  private:
    void init_allocators(int block_size) const override;
    void alloc_chunk_allocator_thr() override;

    void resize_all_list_block_free_thr() override;
    /// @brief Redistribute released buffers across the per-thread free-lists.
    /// @details A thread that frees far more than it allocates would otherwise hoard
    ///          buffers its neighbours need. Delegates to the free function that owns the policy.
    void load_balance_list_block_free() override
    {
        load_balance_list_block_free_thr<T>();
    }
    void flush_all_deferred(bool in_parallel, bool& have_flushed) override;
    void destroy_after_flush(bool in_parallel) override;

    void delete_list_block_chunk_free_itself() override;

    /// @brief Is the imbalance between per-thread free-lists worth a redistribution pass?
    /// @return True when load_balance_list_block_free() would be worth calling.
    [[nodiscard]] bool need_load_balance_list_block_free() const override
    {
        return need_load_balance_list_block_free_thr<T>();
    }
    std::size_t clear_list_block_free_prepare() override;
    void        clear_list_block_free_screen(int bucket) override;
    void        clear_list_block_free_sort(int bucket) override;
    void        clear_list_block_free_dedup(int slice, int nslices) override;
    void        clear_list_block_free_thr(int bucket) override;
};

/// @brief Serial builds: a single allocator, and nothing per-thread to manage.
template <template <typename> class ChunkT, typename T>
class ManagerChunkParallelImpl<ChunkT, T, false> : public virtual ManagerChunkParallelInterface<false>
{
  private:
    void init_allocator(int block_size) const override;
    /// @brief Destroy this type's single chunk allocator (serial build).
    void delete_chunk_allocator() override
    {
        // std-alloc: allocator_data is NullValueAllocator, destroy() is a no-op.
        ChunkT<T>::allocator_data->destroy();
    }
};

/**
 * @brief Owns the shared, per-(variant, element-type) state behind every array.
 *
 * @details One instance exists per @c ChunkT / @c T combination, reachable
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
 * @tparam ChunkT chunk variant template (zero, zero_one_value or zero_any)
 * @tparam T       stored element type
 */
template <template <typename> class ChunkT, typename T>
class ManagerChunk final : public virtual ManagerChunkInt,
                           public ManagerChunkDynImpl<ChunkT, T>,
                           public ManagerChunkParallelImpl<ChunkT, T>
{
    // The implementation policies are BASES and need this class's state, which
    // protected access does not reach: friendship, not a public accessor.
    template <template <typename> class, typename, bool>
    friend struct ManagerChunkDynImpl;
    template <template <typename> class, typename, bool>
    friend class ManagerChunkParallelImpl;

  private:
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

    /// @brief Per-key reuse budget (sequential: at array creation / rechunk),
    ///        floored at zero::ChunkDef::size_min_buffer_reuse buffers.
    static void set_size_buffer_reuse(unsigned int key, int budget)
    {
        if (key != 0)
        {
            (*size_buffer_reuse_by_key_)[key] = std::max(budget, zero::ChunkDef::size_min_buffer_reuse);
        }
    }

    /// @brief Divisor of the proportional reuse budget: a rank parks up to
    ///        chunks / (this * threads) buffers
    ///        (zero_any::ChunkDef::factor_div_somme_chunk_buffer_reuse).
    /// @details The chunks counted are all the chunks created, compressed
    ///          ones included: at 99 % compression this parks about twice the
    ///          buffers actually materialised. Halved or doubled, and with the
    ///          floors removed or set to 4 MiB per rank, it changed neither time
    ///          nor memory (Milan, three variants at dyn, 64 threads, October
    ///          2026): the former CHUNK_REUSE_SHARE_DIV and CHUNK_REUSE_FLOOR_KB.
    [[nodiscard]] static constexpr long reuse_share_div() noexcept
    {
        return static_cast<long>(zero_any::ChunkDef::factor_div_somme_chunk_buffer_reuse);
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
     * @note A no-op in static-geometry builds (see list_block_free_thr_by_size_).
     */
    static void register_free_list_size([[maybe_unused]] unsigned int size, [[maybe_unused]] int max_thr);

    /// @brief Create the per-rank reuse lists of (@p key, @p size) if they do
    ///        not exist yet; key 0 is the shared pool's (register_free_list_size()).
    /// @note Every build (static geometry has one size, but many keys).
    ///       Sequential: call at the array's creation, before any parallel
    ///       region parks a block of that array.
    static void register_free_list_key(unsigned int key, unsigned int size, int max_thr);

    /**
     * @brief Print what the reuse pools hold, broken down by geometry.
     *
     * @details Where the bytes sit tells how a re-chunk left buffers behind:
     *          - under the LEGACY store: parked by a path that does not key on
     *            geometry, which drain_reuse_list_for_size() never reaches;
     *          - on the OLD size: the drain did not run, or used the wrong key;
     *          - across every size: the drain frees only part of what it should
     *            (e.g. one rank out of max_thr).
     *
     * @param label prefix for the printed lines
     */
    static void print_reuse_pool_breakdown(const char* label) noexcept;

    /**
     * @brief Bytes parked in this type's reuse pools, every rank, geometry and
     *        key included (printed with CHUNK_STATS).
     *
     * @details Live memory no other CHUNK_STATS counter reports (`list bytes`
     *          counts reuse EVENTS). Old-geometry buffers left here are the usual
     *          gap between a re-chunked array and one born at the new geometry.
     */
    [[nodiscard]] static std::size_t reuse_pool_bytes() noexcept;

    /**
     * @brief Return the reuse pools' excess to the allocator, keeping a margin.
     *
     * @details The pools never shrink on their own: a list stays at its
     *          high-water mark, which can be most of the process footprint.
     *          @p keep_per_thread is a cap per thread AND per geometry, not a
     *          target: a shorter list is left alone, nothing is pre-allocated.
     *
     *          Fast allocator: the parked buffers go back to their slabs
     *          (release_parked_buffers()), then every slab left empty is given
     *          back to the system (sweep_empty_slabs()) -- the fast allocator's
     *          malloc_trim. Outside a parallel region only (a no-op inside):
     *          the slab chains are touched with no lock.
     *
     * @param keep_per_thread buffers to keep in each (geometry, rank) list
     * @return bytes returned to the allocator
     *
     * @warning Timing is the caller's: trimming too eagerly makes the next
     *          materialisation re-allocate. The natural point is after a
     *          compression pass.
     */
    static std::size_t trim_reuse_pools(std::size_t keep_per_thread) noexcept;

    /**
     * @brief Call @p func(size, lists) for every reuse-list set of this type
     *        whose geometry is known: the legacy lists in a static build (every
     *        buffer has the static size), the per-size lists in a dynamic one,
     *        and the keyed lists. The legacy lists of a dynamic build mix
     *        geometries and are skipped.
     */
    template <typename F>
    static void for_each_sized_reuse_set(F&& func);

    /**
     * @brief Fast allocator: give each rank's parked buffers past the first
     *        @p keep_per_thread back to the slab they came from.
     * @details Each buffer goes to the allocator instance its slab names
     *          (owner), whatever rank parked it: the drain's routing
     *          (drain_free_rank), done serially. A buffer no slab claims stays
     *          parked. Serial, outside any parallel region: the owners' slab
     *          chains are modified with no lock.
     * @return The bytes given back.
     */
    static std::size_t release_parked_buffers([[maybe_unused]] std::size_t keep_per_thread) noexcept;

    /**
     * @brief Fast allocator: give back to the system (madvise) the slabs of
     *        this type that hold no block -- every rank, the shared pool
     *        (key 0) and every keyed set, at every registered geometry, into
     *        ONE sweep.
     * @details Freed buffers return to their slab, and a slab is only given
     *          back by a sweep: without one, the memory of a compression stays
     *          resident until the next re-chunk or destruction. Serial, at a
     *          quiescent point (no allocation or free in flight): an empty slab
     *          must BE empty (fast_alloc::EmptySlabSweep). A no-op inside a
     *          parallel region, and in a build without OpenMP.
     * @return Number of slabs handed to the sweep.
     */
    static std::size_t sweep_empty_slabs() noexcept;

    /// @brief Set the candidate-list length that triggers a compression pass.
    /// @details Decides when trigger_compression_if_needed() flags a rank's
    ///          candidate list, hence how often compression runs at all.
    static void set_size_max_number_of_chunk_to_check_compress(int max_chunks)
    {
        size_max_number_of_chunk_to_check_compress = max_chunks;
    }

    /**
     * @brief Drop from @p candidates every queued candidate that belongs to @p num_tab.
     *
     * @details The one place that knows a candidate tuple names its owning array
     *          in field <2>; the remove_from_compress_list_*_by_num_tab helpers
     *          differ only in their traversal.
     *
     * @tparam Vec any sequence of candidate tuples.
     * @param candidates       the sequence to filter, modified in place.
     * @param num_tab tab index whose candidates are removed.
     */
    template <typename Vec>
    static void erase_candidates_of_tab(Vec& candidates, int num_tab) noexcept
    {
        std::erase_if(candidates, [num_tab](const auto& tup) { return std::get<2>(tup) == num_tab; });
    }

    /// @brief Map key of list_block_free_thr_keyed_: (@p key << 32) | @p size.
    static constexpr std::uint64_t keyed_index(unsigned int key, unsigned int size) noexcept
    {
        return (static_cast<std::uint64_t>(key) << 32U) | size;
    }

    const int max_thr_ = CHUNK_PMP_THREAD_CAPACITY;

    /// @brief zero_one_value: the second-value lists sweep_rank() drains, snapshotted
    ///        serially by sweep_prepare().
    std::vector<SecondValueInfo<T>*> sweep_inf_arrays_;

    /// @brief Slab byte budget seen at the last register_chunk_size(); what
    ///        register_alloc_key() hands to the keyed allocator sets.
    static inline int registered_block_size_ = 0;

    // Serial counterpart of chunk_to_compress_potentially_thr (likewise
    // chunk2a_to_compress_potentially). Declared in EVERY build, empty under
    // OpenMP, for the reason given at list_block_free_thr_by_size_.
    static inline constinit internal::Immortal<TChunkComp<ChunkT, T>> chunk_to_compress_potentially_storage_{};
    static inline TChunkComp<ChunkT, T>& chunk_to_compress_potentially = chunk_to_compress_potentially_storage_.value;

    static inline long somme_chunk_create{0};

    static inline int size_buffer_reuse;
    static inline constinit internal::LazyImmortal<std::unordered_map<unsigned int, int>>
        size_buffer_reuse_by_key_; ///< see get_size_buffer_reuse(key)

    static inline int size_max_number_of_chunk_to_check_compress;

    static inline constinit internal::Immortal<std::vector<PerThreadFreeList>> list_block_free_thr_storage_{};
    static inline std::vector<PerThreadFreeList>& list_block_free_thr = list_block_free_thr_storage_.value;
    // Multi-size: a separate reuse free-list per (size, thread); a reused block
    // must match the active chunk size. Written only at register_chunk_size
    // (sequential); read per-alloc/free.
    //
    // Declared in EVERY build (empty in static ones) so accessors can select with
    // `if constexpr`: a discarded branch still name-looks up members of the
    // current instantiation, so they must exist.
    static inline constinit internal::LazyImmortal<std::map<unsigned int, std::vector<PerThreadFreeList>>>
        list_block_free_thr_by_size_;
    /// @brief Reuse lists of the arrays with their own slab affinity: per-rank
    ///        lists per (affinity key, size), key != 0 (key 0 stays on
    ///        list_block_free_thr_by_size_ / list_block_free_thr).
    ///        Invariant: list (thr, K, size) only holds blocks of K's slabs -- a
    ///        foreign-rank block parks under ITS key (slab->owner_key) -- so
    ///        nothing has to be filtered at pop.
    static inline constinit internal::LazyImmortal<std::map<std::uint64_t, std::vector<PerThreadFreeList>>>
        list_block_free_thr_keyed_;

    /**
     * @brief Teardown hand-off buffer: pointers cleared for release, by rank.
     *
     * @details STATIC on purpose: the teardown phases run on different threads
     *          of one region, so a bucket screened by one thread must be visible
     *          to the one that frees it. Live during teardown only.
     */
    static inline constinit internal::Immortal<std::vector<std::vector<T*>>> teardown_to_free_storage_{};
    static inline std::vector<std::vector<T*>>& teardown_to_free_ = teardown_to_free_storage_.value;

    /**
     * @brief Cross-bucket repeats found by clear_list_block_free_dedup(), one
     *        list per address slice.
     * @details Written by the dedup phase (one list per slice), read by the free
     *          phase after the barrier. Normally empty.
     */
    static inline constinit internal::Immortal<std::vector<std::vector<internal::TeardownRepeat<T>>>>
                                                                         teardown_repeats_storage_{};
    static inline std::vector<std::vector<internal::TeardownRepeat<T>>>& teardown_repeats_ =
        teardown_repeats_storage_.value;

  public:
    /**
     * @brief Captures the thread capacity once, at construction.
     *
     * @note max_thr is the process-wide thread capacity: every per-thread
     *       container has a slot for any rank, even of a team raised later.
     */
    ManagerChunk()           = default;
    ~ManagerChunk() override = default;

    /**
     * @brief Size the static per-rank structures of (ChunkT, T)
     *        (see CompressionManager::ensure_initialised).
     * @details Called by AllTypeChunk::register_manager(), before the manager
     *          is published: the fan-outs and the first array come after it.
     */
    static void init_static_state()
    {
        CompressionManager<T, ChunkT<T>>::ensure_initialised();
        (void)ChunkT<T>::get_manager(); // built now, outside any parallel region of the type
    }

    ManagerChunk(const ManagerChunk&)            = delete;
    ManagerChunk& operator=(const ManagerChunk&) = delete;
    ManagerChunk(ManagerChunk&&)                 = delete;
    ManagerChunk& operator=(ManagerChunk&&)      = delete;

    void resize_chunk_to_compress_potentially() const override;

    // Build-dependent overrides live in the policies:
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

    /// @brief Buffers parked for reuse, every rank, every geometry.
    [[nodiscard]] static std::size_t reuse_pool_blocks() noexcept;

    /// @brief Buffers parked for reuse at geometry @p size, every rank (per-size
    ///        lists only: the legacy list is not keyed by geometry).
    [[nodiscard]] static std::size_t reuse_pool_blocks_at([[maybe_unused]] unsigned int size) noexcept;
    /// @brief Maximum number of buffers a thread keeps parked for reuse.
    [[nodiscard]] static int get_size_buffer_reuse() noexcept
    {
        return size_buffer_reuse;
    }
    /// @brief Reuse budget of the list (thr, key): the array's own share per rank,
    ///        set by set_size_buffer_reuse(key, ...); key 0 and unknown keys get
    ///        the global budget.
    [[nodiscard]] static int get_size_buffer_reuse(unsigned int key) noexcept;

    /// @brief Everything an array with its own affinity key needs, in one call
    ///        (sequential, from rebuild_storage once nb_chunk is known): its
    ///        allocator set at (key, size), its per-rank reuse lists, and their
    ///        budget nb_chunk / (factor_div x threads).
    /// @note Dynamic OpenMP fast-alloc builds only; a no-op elsewhere.
    /// @return true when the keyed set, lists and budget exist; false otherwise
    ///         (no slab budget yet, or no dynamic registry) -- the caller must
    ///         then stay on key 0.
    [[nodiscard]] static bool register_alloc_key(unsigned int key, unsigned int size, int nb_chunk);

    /**
     * @brief Resolve the reuse list of (@p thr_rank, @p key) at the ambient
     *        geometry ONCE, or report that no slot exists.
     *
     * @details Hot-path form of get_list_block_free_thr_size(thr_rank, key) then
     *          get_list_block_free_thr(thr_rank, key), which repeat the map
     *          lookups. Same resolution order:
     *          - keyed list (key != 0, registered, thr_rank in range);
     *          - per-size list (dynamic builds, registered, thr_rank in range);
     *          - legacy store (thr_rank in range);
     *          - otherwise nullptr -- the slot was never sized (see
     *            get_list_block_free_thr_size(int) for why that happens).
     *
     * @param thr_rank owning thread's rank
     * @param key      affinity key (0 = the rank's shared pool)
     * @return the list, or nullptr when no slot exists for @p thr_rank
     * @note The pointer is valid until the next registration, resize or
     *       teardown of the reuse stores -- use it immediately, never cache it.
     */
    [[nodiscard]] static std::vector<T*>* find_list_block_free_thr(int thr_rank, unsigned int key) noexcept;

    /// @brief Reuse list of (thr_rank, key) at the ambient geometry. Key 0 is
    ///        get_list_block_free_thr(thr_rank); an unregistered key falls back
    ///        to the shared pool rather than indexing blindly.
    static std::vector<T*>& get_list_block_free_thr(int thr_rank, unsigned int key) noexcept;
    /// @brief Number of blocks parked in rank @p thr_rank's reuse list for
    ///        @p key at the ambient geometry (0 when no slot exists).
    [[nodiscard]] static size_t get_list_block_free_thr_size(int thr_rank, unsigned int key) noexcept
    {
        const auto* const list = find_list_block_free_thr(thr_rank, key);
        return list != nullptr ? list->size() : 0;
    }
    /// @brief Number of buffers currently parked for @p thr_rank at the ambient geometry.
    [[nodiscard]] static size_t get_list_block_free_thr_size(int thr_rank) noexcept;
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
    static std::vector<T*>& get_list_block_free_thr(int thr_rank) noexcept;

    /**
     * @brief Reuse list for an EXPLICIT geometry, ignoring the ambient size.
     *
     * @details A buffer must be parked in, and popped from, the list for ITS OWN
     *          geometry: a size-A buffer parked while the ambient size is B would
     *          later be popped at B, and init_block()'s memset would overrun it.
     *
     * @param thr_rank owning thread's rank
     * @param sz       the buffer's real geometry
     * @return the per-thread reuse list for @p sz (the legacy single list in
     *         single-size builds, or when @p sz was never registered)
     * @note Callers that know the buffer's real geometry -- teardown, recycle --
     *       must use this rather than get_list_block_free_thr().
     */
    static std::vector<T*>& get_list_block_free_thr_for_size([[maybe_unused]] int          thr_rank,
                                                             [[maybe_unused]] unsigned int sz);

    /**
     * @brief Empty the by-geometry reuse list for @p size, across all threads
     *        (dynamic builds; a no-op otherwise).
     *
     * @details Called by rechunk_to() (all variants) before allocating at a new
     *          geometry. Std-alloc: each buffer is svfree'd (parked in the
     *          ledger, hence DISCARD). Fast-alloc: each is returned to its OWNING
     *          rank's slab (free_elt_pow2() is only safe on an owned block);
     *          blocks with no known owner stay parked.
     *
     * @param size geometry whose reuse list is drained
     * @warning Only the per-size lists are drained. Do not fall back to the
     *          legacy list: a serial build never sizes it (out of bounds).
     *          Teardown drains it in clear_list_block_free_thr().
     */
    static void drain_reuse_list_for_size([[maybe_unused]] unsigned int size) noexcept;

    /// @brief Owner buckets of a drain: [sorting rank][owning rank].
    using DrainBuckets = std::vector<std::vector<std::vector<T*>>>;

    /**
     * @brief Drain pass 1 for one rank: sort rank @p rank's parked blocks into
     *        row @p rank of @p bucket by owning rank; blocks no slab claims stay
     *        parked. Touches rank @p rank's list and row only.
     * @param per_thr the per-rank lists of parked blocks; rank @p rank's is
     *        sorted, and keeps only the blocks no slab claims.
     * @param n_thr team size: owning ranks outside [0, n_thr) are not sorted.
     * @param rank the sorting rank; out of [0, @p n_thr) does nothing.
     * @param geometry the allocator whose slab_of() finds each block's slab
     *        and owner.
     * @param bucket the drain's [sorting rank][owning rank] buckets.
     * @param col_dirty when given (the team drain), flag o is raised for every
     *        owning rank o that receives a block: column o needs a free pass.
     */
    template <typename Alloc>
    static void drain_sort_rank(std::vector<PerThreadFreeList>& per_thr,
                                int                             n_thr,
                                int                             rank,
                                Alloc*                          geometry,
                                DrainBuckets&                   bucket,
                                std::atomic<unsigned char>*     col_dirty = nullptr) noexcept;

    /**
     * @brief Drain pass 2 for one rank: free column @p rank of @p bucket into
     *        rank @p rank's own allocator instances -- the only mutation of a
     *        slab chain, always by the rank that owns it.
     * @param n_thr team size: the rows of @p bucket read.
     * @param rank the owning rank, whose column is freed; out of [0, @p n_thr)
     *        does nothing.
     * @param geometry the allocator whose slab_of() finds each block's slab;
     *        the block is freed into the instance that slab names as owner.
     * @param bucket the drain's [sorting rank][owning rank] buckets, filled by
     *        drain_sort_rank().
     * @param release_cells give the cells' capacity back (a one-off drain) or
     *        keep it (drain_reuse_list_for_size_team(), reused per slice).
     */
    template <typename Alloc>
    static void
    drain_free_rank(int n_thr, int rank, Alloc* geometry, DrainBuckets& bucket, bool release_cells) noexcept;

    /// @brief What drain_reuse_list_for_size_team() keeps from one slice
    ///        boundary to the next: the owner buckets (grown once, then reused
    ///        with their capacity) and the list sets found for the size.
    struct TeamDrainScratch
    {
        DrainBuckets                                 bucket;
        std::vector<std::vector<PerThreadFreeList>*> sets;
        void*                                        geometry = nullptr;
        /// Per rank, sized like @c bucket: held = a team member is this rank
        /// (it sorts and frees its own row and column); dirty = column r got
        /// blocks in the current set. The single-thread tails visit only
        /// unheld, dirty ranks: the matrix is capacity x capacity whatever the
        /// team, and walking it all on one thread costs milliseconds per slice.
        std::unique_ptr<std::atomic<unsigned char>[]> held;
        std::unique_ptr<std::atomic<unsigned char>[]> dirty;
        std::size_t                                   n_flags = 0;
    };

    /**
     * @brief drain_reuse_list_for_size() run by the WHOLE @p team: every thread
     *        of the team must call it, at the same point (it shares work and
     *        meets barriers on @p team).
     *
     * @details Same owner-routed drain and ownership discipline, but inside the
     *          caller's region (the serial drain opens two regions per list set,
     *          and runs on one thread under OMPT), with the bucket matrix
     *          allocated once per re-chunk. Used at the zero refill's slice
     *          boundaries (rechunk_refill), where the serial drain cost several ms
     *          each. Ranks no team member holds are swept by one thread.
     * @pre A quiescent point for the lists of @p size: no thread parks into
     *      them or pops from them meanwhile (the refill's slice barrier).
     *      Fast-alloc OpenMP dynamic builds; a no-op otherwise.
     * @param team the team running the drain (the refill's)
     * @param size the geometry whose reuse lists are drained
     * @param scratch    scratch kept across calls (the bucket matrix, the flags)
     */
    static void drain_reuse_list_for_size_team([[maybe_unused]] pmp::Team&        team,
                                               [[maybe_unused]] unsigned int      size,
                                               [[maybe_unused]] TeamDrainScratch& scratch) noexcept;

    /// @brief Candidate-list length that triggers a compression pass.
    [[nodiscard]] static int get_size_max_number_of_chunk_to_check_compress() noexcept
    {
        return size_max_number_of_chunk_to_check_compress;
    }
    // Serial-only on purpose: under OpenMP it would return the always-empty
    // serial store while the candidates sit per thread, so the constraint makes
    // a call fail to compile there.
    /// @brief The zero_any candidate list (serial builds: a single one).
    static auto& get_chunk2a_to_compress_potentially() noexcept
        requires(!build_config::threaded)
    {
        return chunk2a_to_compress_potentially;
    }
    /// @brief Rank @p thr's zero_any candidate list (the single one in a
    ///        serial build).
    static auto& get_chunk2a_to_compress_potentially([[maybe_unused]] int thr) noexcept;

    /// @brief Is no zero_any chunk queued for compression, on any thread?
    [[nodiscard]] static bool is_chunk2a_to_compress_potentially_empty() noexcept;

    /**
     * @brief Drop every queued zero_any candidate pointing into [@p lo, @p hi).
     *
     * @details Teardown removal by CHUNK-POINTER RANGE: candidates hold
     *          &chunk_list_[j], so filtering on the array's chunk_list_ span
     *          guarantees no pointer into it survives teardown. The by-num_tab
     *          form can MISS entries whose num_tab a rechunk identity swap
     *          shifted; the dangling zero_any::Chunk* would then be double-freed by a
     *          later compression pass.
     *
     * @param lo first address of the array's chunk_list_ span
     * @param hi one past its last address
     * @note In plain fast-block-compression builds (GNU, no OMPT) this cleanup is
     *       the ONLY one: the built-in vec_tup.clear() runs only under OMPT.
     */
    static void remove_from_compress_list_chunk2a_by_ptr_range(const void* lo, const void* hi) noexcept;

    /**
     * @brief Drop every queued zero_any candidate of array @p num_tab, in every
     *        rank's list (zero_any::Array::clear_core and rechunk_to).
     * @param num_tab tab index whose candidates are removed
     * @note Keeps other arrays' candidates. At teardown prefer
     *       remove_from_compress_list_chunk2a_by_ptr_range() (see why there).
     */
    static void remove_from_compress_list_chunk2a_by_num_tab(int num_tab) noexcept;

    /**
     * @brief zero_one_value analogue: drop every queued zero_one_value candidate for @p num_tab.
     *
     * @details Removes the matching tuples from every rank and second-value
     *          bucket, WITHOUT compressing. Called from zero_one_value::Array::clear_core()
     *          so no zero_one_value::Chunk* of a torn-down or re-chunked array dangles there.
     *
     * @param num_tab tab index whose candidates are removed
     * @note OpenMP builds only: a serial build queues no zero_one_value candidate (fast
     *       block compression is OpenMP-only), so there is nothing to drop.
     */
    static void remove_from_compress_list_chunk2_by_num_tab([[maybe_unused]] int num_tab) noexcept;

    /// @brief Drop every queued zero candidate of array @p num_tab
    ///        (zero::Array::clear_core and rechunk_to).
    /// @details Removes this array's candidates without compressing, and keeps
    ///          other arrays' intact.
    static void remove_from_compress_list_chunk_by_num_tab(int num_tab) noexcept;

    /// @brief zero_one_value's candidate lists: per rank, one list per second value.
    static inline constinit internal::Immortal<TChunk2CompMap<ChunkT, T>>
                                             map_chunk2_to_compress_potentially_thr_storage_{};
    static inline TChunk2CompMap<ChunkT, T>& map_chunk2_to_compress_potentially_thr =
        map_chunk2_to_compress_potentially_thr_storage_.value;

    /**
     * @brief Rank @p thr's candidate list for second value @p key, or null
     *        when the key has none (OpenMP builds: the per-rank store).
     * @details A lookup, never operator[]: a miss would insert a null entry, from
     *          the write path, into a map other ranks read (have_chunk_to_compress,
     *          the rank-0 emptiness probe).
     */
    template <typename Key>
    [[nodiscard]] static auto* compress_candidates(int thr, const Key& key) noexcept
    {
        auto&      by_value = map_chunk2_to_compress_potentially_thr[thr];
        const auto it       = by_value.find(key);
        return it != by_value.end() ? it->second.get() : nullptr;
    }
    /// @brief zero_one_value: per rank, one list per second-value index, copied by
    ///        zero_one_value::Array into map_chunk2_to_compress_potentially_thr when it
    ///        registers a second value.
    static inline constinit internal::Immortal<TChunk2CompThr<ChunkT, T>> chunk2_to_compress_potentially_thr_storage_{};
    static inline TChunk2CompThr<ChunkT, T>&                              chunk2_to_compress_potentially_thr =
        chunk2_to_compress_potentially_thr_storage_.value;
    /// @brief zero_any's per-rank candidate lists.
    static inline constinit internal::Immortal<TChunk2ACompThr<ChunkT, T>>
                                              chunk2a_to_compress_potentially_thr_storage_{};
    static inline TChunk2ACompThr<ChunkT, T>& chunk2a_to_compress_potentially_thr =
        chunk2a_to_compress_potentially_thr_storage_.value;
    /// @brief zero's per-rank candidate lists.
    static inline constinit internal::Immortal<TChunkCompThr<ChunkT, T>> chunk_to_compress_potentially_thr_storage_{};
    static inline TChunkCompThr<ChunkT, T>&                              chunk_to_compress_potentially_thr =
        chunk_to_compress_potentially_thr_storage_.value;
    /// @brief zero_any's single candidate list of a serial build.
    static inline constinit internal::Immortal<TChunk2AComp<ChunkT, T>> chunk2a_to_compress_potentially_storage_{};
    static inline TChunk2AComp<ChunkT, T>&                              chunk2a_to_compress_potentially =
        chunk2a_to_compress_potentially_storage_.value;
};

} // namespace shrink_array

#include "chunk/internal/manager_chunk_inline.h" // the template bodies