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
 * @file reuse_pools.h
 * @brief ReusePools: the buffer reuse pools of one (variant, element type),
 *        a base of ManagerChunk.
 *
 * Its bodies are in reuse_pools_inline.h, which manager_chunk.h includes once
 * ManagerChunk is complete.
 */

#include <algorithm>     // std::max
#include <atomic>        // TeamDrainScratch's rank flags
#include <cstddef>       // std::size_t
#include <cstdint>       // pool keys (drain_reuse_list_for_size, keyed lists)
#include <map>           // std::map
#include <memory>        // std::unique_ptr (the same flags)
#include <unordered_map> // std::unordered_map
#include <vector>        // std::vector

#include "chunk/zero_chunk_def.h"     // ChunkDef::size_min_buffer_reuse
#include "chunk/zero_any_chunk_def.h" // zero_any::ChunkDef: the reuse budget constants
#include "chunk/PMP_CHUNK.h"          // pmp::Team (drain_reuse_list_for_size_team)
#include "chunk/internal/immortal.h"  // Immortal / LazyImmortal: statics safe for global arrays

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
 * @brief The buffer reuse pools of one (variant, element type), shared by
 *        all its arrays: a base of ManagerChunk, static state only.
 *
 * @details Chunk buffers released by a compression or a teardown are parked
 *          here for the next materialisation instead of going back to the
 *          allocator, per rank. In dynamic builds they are keyed BY GEOMETRY
 *          (list_block_free_thr_by_size_), because a buffer physically sized
 *          for one geometry must never be popped at another -- see
 *          get_list_block_free_thr_for_size(); the arrays with their own slab
 *          affinity have their own lists (list_block_free_thr_keyed_). The
 *          pools also hold the reuse budgets, the drains of a re-chunk, the
 *          trims, and the hand-off buffers of the teardown
 *          (ManagerChunkParallelImpl::clear_list_block_free_*).
 *
 *          The members reached through ManagerChunk only -- by its bodies
 *          and its policies, which it befriends -- are protected.
 *
 * @tparam ChunkT chunk variant template (zero, zero_one_value or zero_any)
 * @tparam T      stored element type
 */
template <template <typename> class ChunkT, typename T>
class ReusePools
{
  protected:
    struct alignas(128) PerThreadFreeList
    {
        std::vector<T*> list;
    };

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

    /// @brief Map key of list_block_free_thr_keyed_: (@p key << 32) | @p size.
    static constexpr std::uint64_t keyed_index(unsigned int key, unsigned int size) noexcept
    {
        return (static_cast<std::uint64_t>(key) << 32U) | size;
    }

    static inline int size_buffer_reuse;
    static inline constinit internal::LazyImmortal<std::unordered_map<unsigned int, int>>
        size_buffer_reuse_by_key_; ///< see get_size_buffer_reuse(key)

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
};

} // namespace shrink_array
