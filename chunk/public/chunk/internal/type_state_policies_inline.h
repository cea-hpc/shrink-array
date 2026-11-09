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
 * @file type_state_policies_inline.h
 * @brief Out-of-line bodies of TypeStateDynImpl and TypeStateParallelImpl:
 *        the per-rank allocators, the deferred flush, and the teardown of the
 *        reuse lists in four phases. Included by type_state.h once
 *        TypeState is complete.
 */

#include <algorithm>     // std::max, std::ranges, std::ranges::push_heap, std::ranges::sort
#include <atomic>        // std::atomic, std::memory_order
#include <cstddef>       // std::size_t
#include <functional>    // std::less
#include <unordered_set> // std::unordered_set
#include <vector>        // std::vector

#include "chunk/internal/type_state.h"           // TypeState complete: the policies reach its state
#include "chunk/config.h"                        // build_config, chunk_config
#include "chunk/internal/invariant_check.h"      // CHUNK_CHECK_FAILED
#include "chunk/log.h"                           // CHUNK_LOG_FIRST_N, CHUNK_LOG
#include "fast_allocators/internal/svalloc.h"    // svfree
#include "chunk/internal/PMP_CHUNK.h"            // CHUNK_PMP_*, pmp::
#include "chunk/internal/chunk_size_policy.h"    // alloc_chunk_count, cst_buffer_size
#include "chunk/internal/value_slab.h"           // value_slab_request
#include "chunk/internal/compression_manager.h"  // CompressionManager
#include "chunk/internal/deferred_op_pool.h"     // fast_alloc::ArenaKind
#include "chunk/internal/deferred_ops_storage.h" // DeferredOpsStorage
#include "chunk/internal/pool_ledger.h"          // CHUNK_LEDGER_*
#include "chunk/internal/zero_chunk.h"           // Chunk<T>: the ChunkT<T>:: calls
#include "chunk/internal/zero_one_value_chunk.h" // zero_one_value::Chunk<T>: the ChunkT<T>:: calls
#include "chunk/internal/zero_any_chunk.h"       // zero_any::Chunk<T>: the ChunkT<T>:: calls
#include "chunk/zero_array.h"                    // ArrayChunkCore complete: ArrayType::clear_all_dirty_bitsets
#include "chunk/internal/array_chunk_for.h"      // ArrayChunkForT
#include "chunk/internal/concept_chunk.h"        // IsChunk, IsZeroOneValue, IsZeroAnyFamily
#include "chunk/internal/chunk_func.h"           // load_balance_list_block_free_thr (the policies' header declares it)

namespace shrink_array
{
/// @brief Re-entrancy guard of the deferred flush (see flush_all_deferred).
// Defined (once) in type_state.cpp; flush_all_deferred's body lives in this
// header, so every TU that instantiates it needs the declaration.
extern std::atomic<bool> g_need_flush_deferred;

// The definitions below are members of the mode-specific policies, so the
// specialization itself selects the mode: <..., true> is only ever
// instantiated under OpenMP, <..., false> only without it.

/**
 * @brief Sizes the per-rank free-lists of released buffers to the current team.
 */
template <template <typename> typename ChunkT, typename T>
void TypeStateParallelImpl<ChunkT, T, true>::resize_all_list_block_free_thr()
{
    using Mgr        = TypeState<ChunkT, T>;
    const auto& self = static_cast<const Mgr&>(*this);

    Mgr::resize_list_block_free_thr(self.max_thr_);
}

/**
 * @brief Release the deferred-op storage once its ops have been applied.
 *
 * Destroys the per-thread op maps and clears every array's dirty bitset, so the
 * next phase starts from a clean slate.
 *
 * @param in_parallel True when called from inside a parallel region, where the
 *        destruction has to be serialised with `omp single`.
 *
 * @note It first walks the maps for ops that were enqueued but never drained,
 *       in EVERY build. It does not abort on them: it reports (an error of
 *       the log's channel drain, the first 16 times), then applies and clears them. Skipping the walk would
 *       destroy the maps with the ops in them: a write lost without a word,
 *       its block never recycled.
 *       The walk is one empty() test per rank when nothing is left.
 */
template <template <typename> typename ChunkT, typename T>
void TypeStateParallelImpl<ChunkT, T, true>::destroy_after_flush(bool in_parallel)
{
    using Storage = DeferredOpsStorage<typename ChunkT<T>::PerThreadMap>;
    auto& storage = Storage::instance();

    {
        // Defensive drain, every build: log leftovers (enqueued but never drained by
        // flush_deferred), then execute them and clear the maps. Do not just destroy the
        // maps: the writes they hold would be dropped silently.
        auto* arr = storage.get();
        if (arr)
        {
            const int max_thr = CHUNK_PMP_RANK_BOUND;

            // Drain arr[rank] fully. op_ptr->destroy() recycles the op block into the
            // thread_local free_list of the thread that RUNS this lambda, so run it on
            // the thread owning the rank: the block recycles locally instead of being
            // dropped as foreign. Draining on the owner is the only way to keep it local.
            const auto drain_index = [&](int rank)
            {
                if (rank < 0 || rank >= max_thr || arr[rank].empty())
                {
                    return;
                }

                // Should never happen: report, capped.
                CHUNK_LOG_FIRST_N(16, error, drain,
                                  "destroy_after_flush: thr={} has {} chunks with deferred ops (draining defensively)",
                                  rank, arr[rank].size());

                for (auto& [chunk_ptr, ops_map_ptr] : arr[rank])
                {
                    if (!ops_map_ptr)
                    {
                        continue;
                    }
                    for (auto& [num, op_ptr] : *ops_map_ptr)
                    {
                        op_ptr->write_body();
                        op_ptr->destroy();
                    }
                    ops_map_ptr->clear();
                }
                arr[rank].clear();
            };

            if constexpr (chunk_config::not_use_compression_with_ompt)
            {
                // Route-to-owner drain. destroy_after_flush(false) is only called from serial context
                // (finalize_parallel_phase_after_flush runs after the parallel region closes), so this
                // region does not nest. Each thread drains only its own index (disjoint, race-free) and
                // recycles into its own pool. Compiled out under OMPT: opening a region would fire the
                // parallel_end compression callback, which must not run from a teardown drain.
                pmp::team(pmp::Options{}, [&](pmp::Team& /*team*/) { drain_index(CHUNK_PMP_THREAD_RANK); });
            }

            // Serial mop-up: the OMPT build (no region above) and any index the team did
            // not cover. Foreign blocks may be dropped here: a bounded, safe fallback.
            for (int t = 0; t < max_thr; ++t)
            {
                drain_index(t);
            }
        }
    }
    using ArrayType = ArrayChunkForT<ChunkT<T>>;

    if (in_parallel)
    {
        pmp::current_team().single(
            [&]
            {
                storage.destroy();

                ArrayType::clear_all_dirty_bitsets();
            });
    }
    else
    {
        storage.destroy();

        ArrayType::clear_all_dirty_bitsets();
    }
}

/**
 * @brief Apply this type's pending deferred write ops.
 *
 * Clears `g_need_flush_deferred` for the duration and restores it on
 * exit, so a writer that runs meanwhile does not queue work this pass will miss.
 *
 * In-region, each thread drains its own rank and returns. Out of region it
 * opens its own parallel region and drains REPEATEDLY: replaying an op can
 * enqueue another, so it loops until a pass changes nothing.
 *
 * @param in_parallel  True when called from inside a parallel region.
 * @param have_flushed Set (never cleared) when at least one op was applied, so
 *                     the caller knows whether buffers need reclaiming.
 *
 * @warning The drain loop is capped at 1024 passes. Reaching the cap means ops
 *          are being re-enqueued as fast as they are drained; it reports
 *          (CHUNK_CHECK_FAILED) and gives up rather than hanging.
 * @note Under any_check, and outside aggressive mode, it also audits whether
 *       candidates remain after the final flush (compress=info). A few are benign; a large or
 *       recurring residual is not.
 */
template <template <typename> typename ChunkT, typename T>
void TypeStateParallelImpl<ChunkT, T, true>::flush_all_deferred(bool in_parallel, bool& have_flushed)
{
    using Mgr        = TypeState<ChunkT, T>;
    const auto& self = static_cast<const Mgr&>(*this);

    using Storage = DeferredOpsStorage<typename ChunkT<T>::PerThreadMap>;
    auto& storage = Storage::instance();

    // The global flag keeps the barrier callback out of the SERIAL flush's own nested drain
    // regions; only that flush may toggle it. Do not toggle it in-region: every thread would
    // store into one line at every barrier, and a thread reading it mid-pass would skip its
    // OWN flush.
    const bool toggle_flag = !in_parallel;
    if (toggle_flag)
    {
        g_need_flush_deferred.store(false, std::memory_order_relaxed);
    }
    if (in_parallel)
    {
        const int thr_rank = CHUNK_PMP_THREAD_RANK;
        // Own-rank early-out: pending_ops_rank(r) is raised by rank r's own thread when it
        // enqueues (ops are always enqueued on the writer's rank) and lowered only after an
        // op is drained or erased. So a zero read here, by rank r's thread, cannot hide an op
        // it queued; a stale non-zero only takes the locked path. Skips a great many empty
        // flushes.
        if (CompressionManager<T, ChunkT<T>>::pending_ops_rank(thr_rank) != 0)
        {
            have_flushed |= CompressionManager<T, ChunkT<T>>::flush_deferred(thr_rank);
        }
    }
    else
    {
        bool local_flushed = false;

        // Drain until ALL op_maps are empty BEFORE storage.destroy(). One flush_deferred pass
        // leaves re-enqueues (a write_body can enqueue a new op); destroying the storage with
        // them would leave op_ptrs into freed slabs (use-after-free). At this synchronisation
        // point re-enqueues converge; the bound only guards against non-termination.
        constexpr int max_drain_passes = 1024;
        bool          any_left         = true;
        int           pass             = 0;
        for (; any_left && pass < max_drain_passes; ++pass)
        {
            // No op queued on any rank: no region. Exact here -- outside any team, after its
            // join, every enqueue's increment is visible to this acquire sum, and every op in a
            // map is counted (increment per insertion, decrement only after removal). Skipping
            // saves a fork/join of the whole team per type and per pass.
            if (!CompressionManager<T, ChunkT<T>>::any_pending_ops())
            {
                break;
            }

            int  team         = 0;
            bool pass_flushed = pmp::team_any(pmp::Options{},
                                              [&](pmp::Team& /*team*/, bool& flushed)
                                              {
                                                  const int thr_rank = CHUNK_PMP_THREAD_RANK;
                                                  if (thr_rank == 0)
                                                  {
                                                      team = CHUNK_PMP_NUM_THREADS;
                                                  }
                                                  flushed |= CompressionManager<T, ChunkT<T>>::flush_deferred(thr_rank);
                                              });
            // The ranks this region's team does not have: a team opened with num_threads(n)
            // above omp_get_max_threads(), or one that shrank since, leaves ops on ranks >=
            // `team`, which storage.destroy() would discard (lost writes). Same mop-up as
            // the compression sweeps.
            for (int r = team; r < CHUNK_PMP_RANK_BOUND; ++r)
            {
                pass_flushed |= CompressionManager<T, ChunkT<T>>::flush_deferred(r);
            }
            local_flushed |= pass_flushed;
            any_left = pass_flushed;
        }

        if (pass >= max_drain_passes)
        {
            CHUNK_CHECK_FAILED(drain,
                               "flush_all_deferred hit {} passes without settling -- possible runaway re-enqueue",
                               max_drain_passes);
        }

        have_flushed |= local_flushed;

        // ── Post-flush audit: did we miss any block to compress? ──────────────
        // Outside aggressive mode the candidate set should be empty after the final drain;
        // if not, the compression pass (OMPT hook or GNU equivalent) skipped blocks. Check
        // builds only (under aggressive compression the set churns legitimately). Non-fatal:
        // a few stragglers are tolerable, a persistent residual is not.
        if constexpr (build_config::any_check)
        {
            if (!chunk_config::aggressive_compression()) // run-time switch: chunk/config.h
            {
                if (self.have_chunk_to_compress_potentially(CHUNK_PMP_RANK_BOUND))
                {
                    CHUNK_LOG(info, compress,
                              "audit: blocks left to compress after final flush (non-aggressive) -- compression pass "
                              "skipped some blocks. A few is benign; a recurring/large residual is not.");
                }
            }
        }

        // storage.destroy() frees every op still queued, unapplied. If the counters say any
        // remain, those writes are about to be lost: report it in every build.
        if (CompressionManager<T, ChunkT<T>>::any_pending_ops()) [[unlikely]]
        {
            CHUNK_CHECK_FAILED(drain,
                               "flush_all_deferred: deferred ops still queued after {} pass(es) -- "
                               "storage.destroy() discards them: their writes are LOST",
                               pass);
        }

        storage.destroy();
    }

    if (toggle_flag)
    {
        g_need_flush_deferred.store(true, std::memory_order_release);
    }
}

/**
 * @brief Create this type's array of per-thread chunk allocators.
 * @note Fast-alloc OpenMP builds only; a no-op everywhere else, since the other
 *       modes have no per-thread allocator to build.
 */
template <template <typename> typename ChunkT, typename T>
void TypeStateParallelImpl<ChunkT, T, true>::alloc_chunk_allocator_thr()
{
    using Mgr        = TypeState<ChunkT, T>;
    const auto& self = static_cast<const Mgr&>(*this);

    // std-alloc: allocator_data is NullValueAllocator (no per-thread array to build).
    if constexpr (build_config::threaded && build_config::use_fast_alloc)
    {
        // Built whole: an allocator is not movable.
        *ChunkT<T>::allocator_data = std::vector<TCTypeCAllocator<T>>(static_cast<std::size_t>(self.max_thr_));
    }
}

/**
 * @brief Set every rank's chunk allocator geometry.
 *
 * @param block_size Slab size in bytes (ParamChunk::block_size), divided by
 *        8 / sizeof(T) for element types narrower than 8 bytes, so that a
 *        slab holds about as many elements whatever the type.
 * @note Fast-alloc builds only. Sequential, from whichever thread registers
 *       the manager: an allocator is a slot indexed by rank, not a thread
 *       local, and init() only records the geometry -- the slab is taken at
 *       the rank's first allocation. A type is registered by its first
 *       array, from any context.
 */
template <template <typename> typename ChunkT, typename T>
void TypeStateParallelImpl<ChunkT, T, true>::init_allocators([[maybe_unused]] int block_size) const
{
    // std-alloc: no per-thread slab allocator to init. Guard the whole body:
    // get_chunk_allocator() has no return on its std branch (it std::abort()s),
    // so instantiating this call in a std build would fail return-type deduction.
    if constexpr (build_config::use_fast_alloc)
    {
        using Mgr         = TypeState<ChunkT, T>;
        const int max_thr = static_cast<const Mgr&>(*this).max_thr_;

        // Also recorded here: an array may be created before any runtime size is registered,
        // and register_alloc_key() needs the slab budget.
        Mgr::registered_block_size_ = block_size;

        // One slab rule for every allocator of T (value_slab.h).
        const auto slab_bytes = static_cast<uintptr_t>(
            value_slab_request<T>(block_size, ChunkT<T>::storage_units(shrink_array::cst_buffer_size) * sizeof(T)));

        for (int thr_rank = 0; thr_rank < max_thr; ++thr_rank)
        {
            // -1 = never munmap a value slab: its address could be reobtained by the op pool's
            // fast_alloc::map_anonymous() and overwrite a live op (or vice-versa). Value slabs
            // are long-lived, so the retained memory is modest.
            auto& allocator = ChunkT<T>::get_chunk_allocator(thr_rank);
            allocator.set_rank(thr_rank); // stamped into every slab as owner_rank
            allocator.init(ChunkT<T>::storage_units(shrink_array::alloc_chunk_count()), slab_bytes, -1);
        }
    }
}

/**
 * @brief Initialise this type's single chunk allocator (serial build).
 * @param block_size Slab size in bytes (ParamChunk::block_size).
 * @note The OpenMP counterpart is init_allocators(), one allocator per rank.
 */
template <template <typename> typename ChunkT, typename T>
void TypeStateParallelImpl<ChunkT, T, false>::init_allocator(int block_size) const
{
    // std-alloc: allocator_data is NullValueAllocator, init()/set_arena() are no-ops.
    // -1 = never munmap a value slab, and the Value arena rather than the Op
    // default, so a value pool never shares slabs with the deferred-op pool
    // (see register_chunk_size_serial in chunk_base.h; this is the static /
    // unregistered-size fallback allocator).
    ChunkT<T>::allocator_data->set_arena(fast_alloc::ArenaKind::value);
    ChunkT<T>::allocator_data->init(ChunkT<T>::storage_units(shrink_array::alloc_chunk_count()), block_size, -1);
}

/**
 * @brief Declare a runtime chunk geometry to this type's allocators.
 *
 * Two things happen, and BOTH are needed: the fast allocator gets an allocator
 * for that element count, and the manager gets a per-size reuse free-list slot.
 * The second is why this is called even in std-alloc builds, where there is no
 * allocator to create.
 *
 * @param size       Chunk size, in elements, being registered.
 * @param block_size Slab size in bytes of that size's allocators.
 *
 * @warning An array built at a size never registered here reports that size
 *          from chunk_size() but draws its buffers from whatever pool WAS
 *          primed, so every read past that width returns another chunk's data
 *          -- silently. See dyn_geometry_validation, which guards exactly this.
 */
template <template <typename> typename ChunkT, typename T>
void TypeStateDynImpl<ChunkT, T, true>::register_chunk_size(unsigned int size, [[maybe_unused]] int block_size)
{
    using Mgr                   = TypeState<ChunkT, T>;
    const int this_thr          = static_cast<const Mgr&>(*this).max_thr_;
    Mgr::registered_block_size_ = block_size; // for register_alloc_key()

    if constexpr (build_config::use_fast_alloc)
    {
        if constexpr (build_config::threaded)
        {
            // Per-thread chunk allocators for `size`.
            ChunkT<T>::register_chunk_size(size, this_thr, block_size);
        }
        else
        {
            // The single per-size allocator; old and new sizes coexist in
            // allocator_pool_serial_ during rechunk_refill.
            ChunkT<T>::register_chunk_size_serial(size, block_size);
        }
    }
    // BOTH modes, BOTH alloc backends: the per-size reuse free-list needs a slot for `size`.
    // Without it the by-size lookup falls back to the (non-OMP: unsized) legacy list:
    // drain_reuse_list_for_size reads a garbage vector and buffers of different geometries
    // get mixed (overflow in init_block). max_thr is CHUNK_PMP_MAX_THREADS (1 in non-OMP).
    Mgr::register_free_list_size(size, this_thr);
}

// Teardown of the reuse lists, in four phases inside ONE parallel region
// (AllTypeChunk::clear_list_block_free_thr opens it, once for every type):
//
//   prepare  sequential, before the region: size the buckets.
//   screen   one bucket per call: move rank r's legacy list and its slot in
//            every per-size list into teardown_to_free_[r]. No shared state.
//   sort     one bucket per call: sort, drop the repeats WITHIN the bucket.
//   dedup    one address slice per call: walk every bucket's sub-range for
//            that slice and record the repeats ACROSS buckets.
//   free     one bucket per call: release everything not recorded as a repeat.
//
// Why the repeats matter: across a rechunk a buffer can be parked in BOTH the legacy list and a
// per-size list (clear_core() switches destination once a per-size list exists), and freeing
// it twice corrupts the heap. The sort and dedup phases catch every repeat -- same rank or
// not -- before the first free, report it, and free it once.
// Compile with -DCHUNK_NO_FREE_GUARD to skip the sort and dedup phases.
namespace internal
{
/// @brief Pointer order: std::less is the one comparison guaranteed total on pointers.
template <typename T>
struct PtrLess
{
    /// @brief Compare two pointers with std::less.
    /// @return true when @p lhs orders before @p rhs.
    static bool operator()(const T* lhs, const T* rhs) noexcept
    {
        return std::less<const T*>{}(lhs, rhs);
    }
};
} // namespace internal

/**
 * @brief Size the teardown buckets (sequential, before the region).
 * @return the number of buckets; the region strides its ranks over it.
 * @details One bucket per rank, and a bucket for every slot either collection
 *          holds beyond max_thr, so that nothing is left unscreened by a rank
 *          that has no owner.
 * @post TypeState::teardown_to_free_ has that many empty buckets and
 *       teardown_repeats_ is empty.
 */
template <template <typename> typename ChunkT, typename T>
std::size_t TypeStateParallelImpl<ChunkT, T, true>::clear_list_block_free_prepare()
{
    using Mgr        = TypeState<ChunkT, T>;
    const auto& self = static_cast<const Mgr&>(*this);

    auto nb = static_cast<std::size_t>(std::max(1, self.max_thr_));
    nb      = std::max(nb, Mgr::get_list_block_free_thr().size());
    if constexpr (build_config::chunk_dynamic)
    {
        for (const auto& size_kv : *Mgr::list_block_free_thr_by_size_)
        {
            nb = std::max(nb, size_kv.second.size());
        }
    }

    for (const auto& key_kv : *Mgr::list_block_free_thr_keyed_)
    {
        nb = std::max(nb, key_kv.second.size());
    }
    Mgr::teardown_to_free_.assign(nb, std::vector<T*>{});
    // One repeat list per address slice, and the slices are the team, whose
    // size is only known inside the region: max_thr bounds it.
    Mgr::teardown_repeats_.assign(std::max(nb, static_cast<std::size_t>(CHUNK_PMP_MAX_THREADS)),
                                  std::vector<internal::TeardownRepeat<T>>{});
    return nb;
}

/**
 * @brief Screen ONE rank's reuse lists into its teardown bucket.
 *
 * @param bucket rank whose lists are drained; out of range is a no-op.
 *
 * @details Meant to run inside the region, one bucket per call, any thread.
 *          Touches only that rank's slot in each collection and that rank's
 *          bucket, so the calls are independent.
 *
 * @warning The default list and the per-size lists must BOTH be drained. The
 *          size-aware accessor returns only a per-size list and leaves the
 *          default one untouched, which leaks the buffers recovered through
 *          recup_value -- hence the raw `.list` access below rather than that
 *          overload. The per-size lists exist only in dynamic-geometry builds.
 * @note Only queues anything when NOT using the fast allocator: there the
 *       blocks belong to the arena and are released with it. The lists are
 *       emptied either way.
 * @pre clear_list_block_free_prepare() has run.
 */
template <template <typename> typename ChunkT, typename T>
void TypeStateParallelImpl<ChunkT, T, true>::clear_list_block_free_screen(int bucket)
{
    using Mgr     = TypeState<ChunkT, T>;
    auto& to_free = Mgr::teardown_to_free_;

    if (bucket < 0 || static_cast<std::size_t>(bucket) >= to_free.size())
    {
        return;
    }
    const auto bucket_idx = static_cast<std::size_t>(bucket);
    auto&      out        = to_free[bucket_idx];

    // RAW default list container, NOT the size-aware overload. It may be smaller than max_thr
    // (see allocate_chunk_internal): an unsized slot has nothing to drain.
    if (auto& legacy = Mgr::get_list_block_free_thr(); bucket_idx < legacy.size())
    {
        auto& lbf = legacy[bucket_idx].list;
        // std-alloc (svalloc/malloc) owns each block, so it must be freed on
        // teardown; the fast allocator's arena reclaims in bulk, nothing to do.
        if constexpr (!build_config::use_fast_alloc)
        {
            out.insert(out.end(), lbf.begin(), lbf.end());
        }
        lbf.clear();
    }

    // Also drain the SIZED reuse free-lists (clear_core parks buffers here, keyed by
    // size_policy_.size()). drain_reuse_list_for_size runs only at a rechunk, so at program
    // end the final geometry's buffers would leak otherwise. All variants.
    if constexpr (build_config::chunk_dynamic)
    {
        for (auto& size_kv : *Mgr::list_block_free_thr_by_size_)
        {
            if (bucket_idx >= size_kv.second.size())
            {
                continue;
            }
            auto& lst = size_kv.second[bucket_idx].list;
            if constexpr (!build_config::use_fast_alloc)
            {
                out.insert(out.end(), lst.begin(), lst.end());
            }
            lst.clear();
        }
    }
    // The keyed lists (per-array affinity) are screened the same way: a block
    // parked there is a block like any other, just filed under its array.
    for (auto& key_kv : *Mgr::list_block_free_thr_keyed_)
    {
        if (bucket_idx >= key_kv.second.size())
        {
            continue;
        }
        auto& lst = key_kv.second[bucket_idx].list;
        if constexpr (!build_config::use_fast_alloc)
        {
            out.insert(out.end(), lst.begin(), lst.end());
        }
        lst.clear();
    }
}

/**
 * @brief Sort ONE bucket and drop the repeats it holds by itself.
 * @param bucket bucket to sort; out of range is a no-op.
 * @details Inside the region, one bucket per call. A same-rank repeat (legacy
 *          list AND per-size list of the same rank, the usual double-free
 *          shape) ends up adjacent after the sort and is erased here. The
 *          bucket is left sorted, which is what the dedup phase relies on.
 * @pre clear_list_block_free_screen() has run for EVERY bucket (barrier).
 */
template <template <typename> typename ChunkT, typename T>
void TypeStateParallelImpl<ChunkT, T, true>::clear_list_block_free_sort([[maybe_unused]] int bucket)
{
    if constexpr (!build_config::free_guard)
    {
        return;
    }

    using Mgr     = TypeState<ChunkT, T>;
    auto& to_free = Mgr::teardown_to_free_;

    if (bucket < 0 || static_cast<std::size_t>(bucket) >= to_free.size())
    {
        return;
    }
    auto& blocks = to_free[static_cast<std::size_t>(bucket)];

    std::ranges::sort(blocks, internal::PtrLess<T>{});
    for (std::size_t i = 1; i < blocks.size(); ++i)
    {
        if (blocks[i] == blocks[i - 1])
        {
            CHUNK_CHECK_FAILED(pool, "double free: {} queued twice for free in bucket {} -- NOT freeing again",
                               static_cast<void*>(blocks[i]), bucket);
        }
    }
    blocks.erase(std::ranges::unique(blocks).begin(), blocks.end());
}

/**
 * @brief Find the repeats ACROSS buckets within ONE slice of the address space.
 *
 * @param slice   which slice this call handles, in [0, nslices)
 * @param nslices how many slices the address space is cut into (the team size)
 *
 * @details Inside the region, one slice per call. The slice bounds are taken
 *          from a regular sample of every (sorted) bucket, so the slices hold
 *          about the same number of pointers whatever the address layout; each
 *          call then k-way merges the buckets' sub-ranges for its slice and
 *          records every pointer already produced by a lower bucket. Only
 *          reads the buckets -- the repeats go to teardown_repeats_[slice],
 *          private to this call -- so the slices are independent, and the
 *          free phase applies them after the barrier.
 * @pre clear_list_block_free_sort() has run for EVERY bucket (barrier).
 */
template <template <typename> typename ChunkT, typename T>
void TypeStateParallelImpl<ChunkT, T, true>::clear_list_block_free_dedup([[maybe_unused]] int slice,
                                                                         [[maybe_unused]] int nslices)
{
    if constexpr (!build_config::free_guard)
    {
        return;
    }

    using Mgr           = TypeState<ChunkT, T>;
    const auto& to_free = Mgr::teardown_to_free_;
    const auto  less    = internal::PtrLess<T>{};

    if (slice < 0 || nslices <= 0 || slice >= nslices)
    {
        return;
    }
    if (static_cast<std::size_t>(slice) >= Mgr::teardown_repeats_.size())
    {
        // A team larger than max_thr: prepare() could not size for it.
        CHUNK_LOG(warn, pool, "teardown dedup: slice {} of {} has no repeat list, skipped", slice, nslices);
        return;
    }
    const std::size_t nfree = to_free.size();
    if (nfree < 2)
    {
        return; // a single bucket has no cross-bucket repeat, and sort already did it
    }

    // Slice bounds: nslices-1 samples per bucket, sorted, one splitter every
    // nslices-th sample. Every call computes the same splitters from the same
    // (read-only) input, which beats a single + barrier to share them.
    std::vector<T*> samples;
    samples.reserve(nfree * static_cast<std::size_t>(nslices));
    for (const auto& block : to_free)
    {
        for (int j = 1; j < nslices; ++j)
        {
            if (!block.empty())
            {
                samples.push_back(
                    block[block.size() * static_cast<std::size_t>(j) / static_cast<std::size_t>(nslices)]);
            }
        }
    }
    std::ranges::sort(samples, less);

    const auto splitter = [&](int split) -> T*
    {
        // Splitter s bounds slices s-1 | s; there are nslices-1 of them.
        const std::size_t idx = samples.size() * static_cast<std::size_t>(split) / static_cast<std::size_t>(nslices);
        return samples[idx];
    };
    const bool has_lo = slice > 0 && !samples.empty();
    const bool has_hi = slice + 1 < nslices && !samples.empty();
    T* const   lo     = has_lo ? splitter(slice) : nullptr;
    T* const   hi     = has_hi ? splitter(slice + 1) : nullptr;

    // Heads of the k-way merge (k = nfree): (pointer, bucket, position), min-heap on
    // (pointer, bucket) so that a pointer present in two buckets comes out
    // lower bucket first -- the repeat is then always the higher one.
    struct Head
    {
        T*          p;
        std::size_t bucket;
        std::size_t pos;
        std::size_t end;
    };
    // after(a, b): a comes out of the min-heap after b -- pointer order, then bucket. The
    // heap algorithms build a max-heap on their comparator: this one, "greater", makes it
    // a min-heap.
    const auto after = [&](const Head& a, const Head& b)
    { return less(b.p, a.p) || (a.p == b.p && a.bucket > b.bucket); };

    std::vector<Head> heap;
    heap.reserve(nfree);
    for (std::size_t bi = 0; bi < nfree; ++bi)
    {
        const auto& blocks = to_free[bi];
        const auto  first  = has_lo ? std::ranges::lower_bound(blocks, lo, less) : blocks.begin();
        const auto  last   = has_hi ? std::lower_bound(first, blocks.end(), hi, less) : blocks.end();
        if (first != last)
        {
            heap.push_back(Head{*first, bi, static_cast<std::size_t>(first - blocks.begin()),
                                static_cast<std::size_t>(last - blocks.begin())});
        }
    }
    std::ranges::make_heap(heap, after);

    auto&       repeats     = Mgr::teardown_repeats_[static_cast<std::size_t>(slice)];
    const T*    prev        = nullptr;
    std::size_t prev_bucket = 0;
    bool        have        = false;
    while (!heap.empty())
    {
        std::ranges::pop_heap(heap, after);
        Head head = heap.back();
        heap.pop_back();

        if (have && head.p == prev)
        {
            // Same pointer as the one just produced from a lower bucket: this
            // one is the repeat, skipped when ITS bucket is freed.
            CHUNK_CHECK_FAILED(pool, "double free: {} queued for free in buckets {} and {} -- NOT freeing again",
                               static_cast<void*>(head.p), prev_bucket, head.bucket);
            repeats.push_back(internal::TeardownRepeat<T>{head.bucket, head.p});
        }
        prev        = head.p;
        prev_bucket = head.bucket;
        have        = true;

        if (++head.pos < head.end)
        {
            head.p = to_free[head.bucket][head.pos];
            heap.push_back(head);
            std::ranges::push_heap(heap, after);
        }
    }
}

/**
 * @brief Free ONE bucket.
 *
 * @param bucket bucket to release; out of range is a no-op.
 *
 * @details Inside the region, one bucket per call, by the thread whose rank it
 *          is when the team is large enough. That is the point of the split,
 *          not just the sharing: glibc gives each thread its own arena and a
 *          block is cheapest to release through the one it came from (64
 *          ranks releasing through a single arena contend on its lock).
 *
 *          Skips the pointers the dedup phase recorded against this bucket.
 *          That list is empty on any sane run, so the set is built only when
 *          it is not.
 * @pre clear_list_block_free_dedup() has run for EVERY slice (barrier).
 */
template <template <typename> typename ChunkT, typename T>
void TypeStateParallelImpl<ChunkT, T, true>::clear_list_block_free_thr(int bucket)
{
    using Mgr     = TypeState<ChunkT, T>;
    auto& to_free = Mgr::teardown_to_free_;

    if (bucket < 0 || static_cast<std::size_t>(bucket) >= to_free.size())
    {
        return;
    }
    const auto bucket_idx = static_cast<std::size_t>(bucket);
    auto&      blocks     = to_free[bucket_idx];

    std::unordered_set<T*> skip;
    for (const auto& per_slice : Mgr::teardown_repeats_)
    {
        for (const auto& rep : per_slice)
        {
            if (rep.bucket == bucket_idx)
            {
                skip.insert(rep.ptr);
            }
        }
    }

    // Tell the ledger before each svfree, or every block released here stays parked in the
    // ledger and floods the ledger report (ledger=info) with false positives.
    if (skip.empty())
    {
        for (const auto* elm : blocks)
        {
            CHUNK_LEDGER_DISCARD(elm, "clear_list_block_free_thr.drain");
            fast_alloc::svfree(elm);
        }
    }
    else
    {
        for (auto* elm : blocks)
        {
            if (!skip.contains(elm))
            {
                CHUNK_LEDGER_DISCARD(elm, "clear_list_block_free_thr.drain");
                fast_alloc::svfree(elm);
            }
        }
    }
    blocks.clear();
}

/**
 * @brief Destroy the per-thread free-list containers themselves (teardown).
 * @note Comes after clear_list_block_free_thr(), which empties them: this drops the
 *       containers, not their contents.
 */
template <template <typename> typename ChunkT, typename T>
void TypeStateParallelImpl<ChunkT, T, true>::delete_list_block_chunk_free_itself()
{
    using Mgr = TypeState<ChunkT, T>;

    if constexpr (IsSecondValueFamily<ChunkT, T>)
    {
        Mgr::list_block_free_thr.clear();
    }
    else
    {
        ChunkT<T>::Manager::get_list_block_free_thr(0).clear();
    }
}

} // namespace shrink_array
