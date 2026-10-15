/*
 * Copyright 2025-2026 CEA (www.cea.fr)
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

// Template method definitions for ManagerChunk<CHUNK_T, T>. Kept in a dedicated
// inline header (included at the end of chunk/internal/manager_chunk.h) so that every
// (CHUNK_T, T) the consumer uses auto-instantiates at its use site.

#include <atomic>        // std::atomic, std::memory_order
#include <cstdio>        // std::fprintf, std::fflush
#include <cstdlib>       // std::getenv (the trace switches)
#include <algorithm>     // std::max, std::ranges::sort, std::ranges::push_heap
#include <functional>    // std::less
#include <tuple>         // std::tuple (the memory estimate)
#include <type_traits>   // std::decay_t, std::conditional_t
#include <unordered_set> // std::unordered_set
#include <vector>        // std::vector

#include "chunk/internal/manager_chunk.h"     // the declarations these bodies belong to
#include "chunk/config.h"                     // build_config::use_fast_alloc, chunk_config::use_fast_block_compression
#include "chunk/internal/invariant_check.h" // CHUNK_INVARIANT_FAILED
#include "fast_allocators/internal/svalloc.h" // svfree
#include "chunk/PMP_CHUNK.h"                  // CHUNK_PMP_THREAD_RANK, CHUNK_PMP_MAX_THREADS, pmp::team / team_any / current_team
#include "chunk/internal/chunk_size_policy.h" // alloc_chunk_count
#include "chunk/internal/compression_manager.h"  // CompressionManager (the barrier flushes)
#include "chunk/internal/deferred_op_pool.h"     // fast_alloc::ArenaKind (its std-alloc stand-in too)
#include "chunk/internal/deferred_ops_storage.h" // DeferredOpsStorage
#include "chunk/internal/pool_ledger.h"          // CHUNK_LEDGER_DISCARD
#include "chunk/chunk_std.h"                     // Chunk<T>: the CHUNK_T<T>:: calls
#include "chunk/chunk2_std.h"                    // Chunk2<T>: the CHUNK_T<T>:: calls
#include "chunk/chunk2A_std.h"                   // Chunk2A<T>: the CHUNK_T<T>:: calls
#include "chunk/chunk_def.h"                     // ChunkDef
#include "chunk/chunk2_def.h"                    // Chunk2Def
#include "chunk/chunk2A_def.h"                   // Chunk2ADef
#include "chunk/array_chunk.h"                   // ArrayChunkCore complete: ArrayType::clear_all_dirty_bitsets
#include "chunk/internal/array_chunk_for.h"      // ArrayChunkFor_t
#include "chunk/internal/concept_chunk.h"        // IsChunk, IsChunk2, IsChunk2A, IsChunk2Or2A
#include "chunk/global_stats_chunk_list.h"       // GlobalStatsChunkList
#include "chunk/type_name.h"                     // type_name<T>
#include "chunk/internal/chunk_func.h"           // load_balance_list_block_free_thr
#include "chunk/singleton.h"                     // SingletonChk

namespace shrink_array
{
// Forward declarations breaking the include cycle chunk_func.h -> chunk_std.h -> proxy.h ->
// manager_chunk.h -> THIS file -> chunk_func.h: when chunk_func.h is the entry point, its
// templates are not yet declared here (#pragma once skips it), and the explicit
// load_balance_list_block_free_thr<T>() calls below need a declaration. Defined in chunk_func.h.
template <typename T>
bool need_load_balance_list_block_free_thr();
template <typename T>
void load_balance_list_block_free_thr();
} // namespace shrink_array

/**
 * @file manager_chunk_inline.h
 * @brief Out-of-line bodies of ManagerChunk, the per-type chunk manager.
 *
 * One ManagerChunk exists per (chunk template, element type) pair, and it owns
 * everything that is per-type rather than per-array: the allocators, the
 * per-thread free-lists of released buffers, and the compression-candidate
 * lists. AllTypeChunk walks the whole set of them through the ManagerChunkInt
 * interface; this file is where the work actually happens.
 *
 * Almost every body here has the same shape: an `if constexpr` ladder on which
 * chunk variant CHUNK_T is. That is not incidental — the three variants keep
 * their candidates in genuinely different containers:
 *  - **Chunk** : one list per rank;
 *  - **Chunk2**: a map per rank, KEYED BY SECOND VALUE, so anything that walks
 *    the candidates has to iterate the array's registered second values first
 *    (and `bool`, whose second value is always `true`, takes a separate branch);
 *  - **Chunk2A**: one list per rank again, since its second value is per chunk.
 *
 * @warning have_chunk_to_compress_potentially() must mirror
 *          compress_chunk_to_compress_potentially()'s dispatch EXACTLY. They are
 *          the predicate and the action of the same decision: if the predicate
 *          reports "nothing to compress" for a variant the action would have
 *          handled, the between-phase pass never runs for it and all of its
 *          compression falls on the opportunistic in-loop path.
 *
 * @see manager_chunk.h for ManagerChunkInt, the type-erased interface, and for
 *      the free-list containers these bodies manipulate.
 * @see all_type_chunk.h for the registry that drives every manager at once.
 */

namespace shrink_array
{

/// @brief Re-entrancy guard of the deferred flush (see flush_all_deferred).
// Defined (once) in manager_chunk.cpp; flush_all_deferred's body lives in this
// header, so every TU that instantiates it needs the declaration.
extern std::atomic<bool> g_need_flush_deferred;
/**
 * @brief Compress every compressible chunk of this type, ignoring the candidate lists.
 *
 * The unconditional sweep: it does not ask whether anything was queued, it runs
 * the variant's block compression directly. Used where the caller wants the
 * memory back now rather than at the next phase boundary.
 *
 * @note Chunk and Chunk2A take a proxy argument (null here: no write triggered
 *       this), Chunk2 does not — hence the three-way `if constexpr`.
 */
template <template <typename> typename CHUNK_T, typename T>
void ManagerChunk<CHUNK_T, T>::force_chunk_to_compress() const
{
    // The calling thread's rank, in or out of a region: no region is opened.
    const int thr = CHUNK_PMP_THREAD_RANK;

    if constexpr (chunk_config::use_fast_block_compression && IsChunk<CHUNK_T, T>)
    {
        CHUNK_T<T>::make_block_compression(thr, nullptr);
    }

    if constexpr (IsChunk2A<CHUNK_T, T>)
    {
        CHUNK_T<T>::make_block_compression(thr, nullptr);
    }
    else if constexpr (chunk_config::use_fast_block_compression && IsChunk2<CHUNK_T, T>)
    {
        CHUNK_T<T>::make_block_compression(thr);
    }
}

/**
 * @brief Sizes this type's compression-candidate containers to the current team.
 *
 * Which containers exist depends on the variant (see the file header): a plain
 * per-rank list for Chunk and Chunk2A, plus the per-second-value map for
 * Chunk2. A list keeps its capacity across the passes' clear(), so nothing
 * pre-reserves it.
 */
template <template <typename> typename CHUNK_T, typename T>
void ManagerChunk<CHUNK_T, T>::resize_chunk_to_compress_potentially() const
{
    const int nthr = CHUNK_PMP_THREAD_CAPACITY; // every rank that may run, as the manager's max_thr

    // --- Standard Chunk case ---
    if constexpr (chunk_config::use_fast_block_compression && IsChunk<CHUNK_T, T>)
    {
        // Resize the list of chunks potentially needing compression
        chunk_to_compress_potentially_thr.resize(nthr);
    }

    // --- Complex types (Chunk2/2A) multi-threading logic ---
    if constexpr (build_config::threaded)
    {
        // Specific to Chunk2: Dual-value compression with static maps
        if constexpr (IsChunk2<CHUNK_T, T>)
        {
            map_chunk2_to_compress_potentially_thr.resize(nthr);
            chunk2_to_compress_potentially_thr.resize(nthr);
        }

        // Specific to Chunk2A: Per-chunk secondary value compression
        if constexpr (IsChunk2A<CHUNK_T, T>)
        {
            chunk2A_to_compress_potentially_thr.resize(nthr);
        }
    }
}

/**
 * @brief Is there anything queued for compression for this type?
 *
 * Cheap gate for the callers that would otherwise open a parallel region for
 * nothing. For Chunk2 it has to ask once per registered second value, since
 * that is what keys the candidate map.
 *
 * @param n_thr   Number of per-rank lists to examine.
 * @return True as soon as one list holds a candidate.
 *
 * @warning This predicate and compress_chunk_to_compress_potentially() must agree
 *          branch for branch — see the @warning in the file header.
 */
template <template <typename> typename CHUNK_T, typename T>
bool ManagerChunk<CHUNK_T, T>::have_chunk_to_compress_potentially(int n_thr) const
{
    // Mirror compress_chunk_to_compress_potentially's dispatch exactly (chunk2 is handled
    // whenever use_fast_block_compression is on, GNU included, not only under OMPT): see the
    // file header's @warning.
    if constexpr (chunk_config::use_fast_block_compression)
    {
        if constexpr (IsChunk2<CHUNK_T, T>)
        {
            if constexpr (is_not_bool)
            {
                const auto& sh_inf           = *Proxy2<T>::share_info_array;
                const auto& v_all_second_val = sh_inf.get_all_second_value();
                return std::ranges::any_of(v_all_second_val, [n_thr](const auto& second_value)
                                           { return CHUNK_T<T>::have_chunk_to_compress(second_value, n_thr); });
            }
            else
            {
                return CHUNK_T<T>::have_chunk_to_compress_bool(n_thr);
            }
        }
        else
        {
            return CHUNK_T<T>::have_chunk_to_compress(n_thr);
        }
    }

    return false;
}

/**
 * @brief Drain this type's compression-candidate backlog.
 *
 * The action whose predicate is have_chunk_to_compress_potentially(). Chunk2 is
 * driven once per registered second value; the other variants have a single
 * entry point.
 *
 * @param apply true: compress the candidates found constant; false: walk the
 *        lists without compressing (the dying array's purge).
 */
template <template <typename> typename CHUNK_T, typename T>
void ManagerChunk<CHUNK_T, T>::compress_chunk_to_compress_potentially([[maybe_unused]] bool apply)
{
    if constexpr (chunk_config::use_fast_block_compression)
    {
        if constexpr (IsChunk2<CHUNK_T, T>)
        {
            if constexpr (is_not_bool)
            {
                const auto& sh_inf          = *Proxy2<T>::share_info_array;
                const auto& v_all_inf_array = sh_inf.get_all_inf_array();
                for (const auto& inf_array : v_all_inf_array)
                {
                    CHUNK_T<T>::compress_chunk_to_compress(*inf_array, apply);
                }
            }
            else
            {
                CHUNK_T<T>::compress_chunk_to_compress_bool(apply);
            }
        }
        else
        {
            CHUNK_T<T>::compress_chunk_to_compress(apply);
        }
    }
}

/// @brief Chunk2: snapshot the second-value lists the sweep drains.
template <template <typename> typename CHUNK_T, typename T>
void ManagerChunk<CHUNK_T, T>::sweep_prepare()
{
    if constexpr (chunk_config::use_fast_block_compression && IsChunk2<CHUNK_T, T>)
    {
        if constexpr (is_not_bool)
        {
            const auto v = Proxy2<T>::share_info_array->get_all_inf_array();
            sweep_inf_arrays_.assign(v.begin(), v.end());
        }
    }
}

/// @brief Rank @p thr's share of compress_chunk_to_compress_potentially().
template <template <typename> typename CHUNK_T, typename T>
void ManagerChunk<CHUNK_T, T>::sweep_rank([[maybe_unused]] int thr, [[maybe_unused]] bool apply)
{
    if constexpr (chunk_config::use_fast_block_compression)
    {
        if constexpr (IsChunk2<CHUNK_T, T>)
        {
            if constexpr (is_not_bool)
            {
                for (const auto* inf_array : sweep_inf_arrays_)
                {
                    CHUNK_T<T>::compress_sweep_rank(thr, *inf_array, apply);
                }
            }
            else
            {
                CHUNK_T<T>::compress_sweep_rank_bool(thr, apply);
            }
        }
        else if constexpr (IsChunk2A<CHUNK_T, T>)
        {
            CHUNK_T<T>::compress_sweep_rank(thr, apply);
        }
        else
        {
            CHUNK_T<T>::compress_sweep_rank(thr); // Chunk: always compresses
        }
    }
}

/// @brief Chunk2 always swept the ranks past the team; Chunk2A in one pass;
///        Chunk never.
template <template <typename> typename CHUNK_T, typename T>
bool ManagerChunk<CHUNK_T, T>::sweep_mops_up([[maybe_unused]] bool apply) const
{
    if constexpr (chunk_config::use_fast_block_compression)
    {
        if constexpr (IsChunk2<CHUNK_T, T>)
        {
            return true;
        }
        else if constexpr (IsChunk2A<CHUNK_T, T>)
        {
            return CHUNK_T<T>::compress_sweep_mops_up(apply);
        }
    }
    return false;
}

// The definitions below are members of the mode-specific policies, so the
// specialization itself selects the mode: <..., true> is only ever
// instantiated under OpenMP, <..., false> only without it.

/**
 * @brief Sizes the per-rank free-lists of released buffers to the current team.
 */
template <template <typename> typename CHUNK_T, typename T>
void ManagerChunkParallelImpl<CHUNK_T, T, true>::resize_all_list_block_free_thr()
{
    using mgr        = ManagerChunk<CHUNK_T, T>;
    const auto& self = static_cast<const mgr&>(*this);

    mgr::resize_list_block_free_thr(self.max_thr);
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
 *       in EVERY build. It does not abort on them: it reports (stderr, the
 *       first 16 times), then applies and clears them. Skipping the walk would
 *       destroy the maps with the ops in them: a write lost without a word,
 *       its block never recycled.
 *       The walk is one empty() test per rank when nothing is left.
 */
template <template <typename> typename CHUNK_T, typename T>
void ManagerChunkParallelImpl<CHUNK_T, T, true>::destroy_after_flush(bool in_parallel)
{
    using Storage = DeferredOpsStorage<typename CHUNK_T<T>::per_thread_map>;
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
                static std::atomic<int> reports{0};
                const int               report = reports.fetch_add(1, std::memory_order_relaxed);
                if (report < 16)
                {
                    std::fprintf(stderr,
                                 "[LEAK] destroy_after_flush: thr=%d has %zu chunks with deferred ops (draining "
                                 "defensively)%s\n",
                                 rank, arr[rank].size(), report == 15 ? " -- further reports suppressed" : "");
                }

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
pmp::team(pmp::options{}, [&](pmp::Team& /*team*/) { drain_index(CHUNK_PMP_THREAD_RANK); });
            }

            // Serial mop-up: the OMPT build (no region above) and any index the team did
            // not cover. Foreign blocks may be dropped here: a bounded, safe fallback.
            for (int t = 0; t < max_thr; ++t)
            {
                drain_index(t);
            }

            std::fflush(stderr);
        }
    }
    using ArrayType = ArrayChunkFor_t<CHUNK_T<T>>;

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
 *          are being re-enqueued as fast as they are drained; it reports on
 *          stderr and gives up rather than hanging.
 * @note Under any_check, and outside aggressive mode, it also audits whether
 *       candidates remain after the final flush. A few are benign; a large or
 *       recurring residual is not.
 */
template <template <typename> typename CHUNK_T, typename T>
void ManagerChunkParallelImpl<CHUNK_T, T, true>::flush_all_deferred(bool in_parallel, bool& have_flushed)
{
    using mgr        = ManagerChunk<CHUNK_T, T>;
    const auto& self = static_cast<const mgr&>(*this);

    using Storage = DeferredOpsStorage<typename CHUNK_T<T>::per_thread_map>;
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
        if (CompressionManager<T, CHUNK_T<T>>::pending_ops_rank(thr_rank) != 0)
        {
            have_flushed |= CompressionManager<T, CHUNK_T<T>>::flush_deferred(thr_rank);
        }
    }
    else
    {
        bool local_flushed = false;

        // Drain until ALL op_maps are empty BEFORE storage.destroy(). One flush_deferred pass
        // leaves re-enqueues (a write_body can enqueue a new op); destroying the storage with
        // them would leave op_ptrs into freed slabs (use-after-free). At this synchronisation
        // point re-enqueues converge; the bound only guards against non-termination.
        constexpr int kMaxDrainPasses = 1024;
        bool          any_left        = true;
        int           pass            = 0;
        for (; any_left && pass < kMaxDrainPasses; ++pass)
        {
            // No op queued on any rank: no region. Exact here -- outside any team, after its
            // join, every enqueue's increment is visible to this acquire sum, and every op in a
            // map is counted (increment per insertion, decrement only after removal). Skipping
            // saves a fork/join of the whole team per type and per pass.
            if (!CompressionManager<T, CHUNK_T<T>>::any_pending_ops())
            {
                break;
            }

            int  team         = 0;
            bool pass_flushed = pmp::team_any(pmp::options{},
                                              [&](pmp::Team& /*team*/, bool& flushed)
                                              {
                                                  const int thr_rank = CHUNK_PMP_THREAD_RANK;
                                                  if (thr_rank == 0)
                                                  {
                                                      team = CHUNK_PMP_NUM_THREADS;
                                                  }
                                                  flushed |=
                                                      CompressionManager<T, CHUNK_T<T>>::flush_deferred(thr_rank);
                                              });
            // The ranks this region's team does not have: a team opened with num_threads(n)
            // above omp_get_max_threads(), or one that shrank since, leaves ops on ranks >=
            // `team`, which storage.destroy() would discard (lost writes). Same mop-up as
            // the compression sweeps.
            for (int r = team; r < CHUNK_PMP_RANK_BOUND; ++r)
            {
                pass_flushed |= CompressionManager<T, CHUNK_T<T>>::flush_deferred(r);
            }
            local_flushed |= pass_flushed;
            any_left = pass_flushed;
        }

        if (pass >= kMaxDrainPasses)
        {
            std::fprintf(stderr,
                         "[DRAIN] flush_all_deferred hit %d passes without settling — "
                         "possible runaway re-enqueue\n",
                         kMaxDrainPasses);
            std::fflush(stderr);
        }

        have_flushed |= local_flushed;

        // ── Post-flush audit: did we miss any block to compress? ──────────────
        // Outside aggressive mode the candidate set should be empty after the final drain;
        // if not, the compression pass (OMPT hook or GNU equivalent) skipped blocks. Check
        // builds only (under aggressive compression the set churns legitimately). Non-fatal:
        // a few stragglers are tolerable, a persistent residual is not.
        if constexpr (build_config::any_check && !chunk_config::have_aggressive_compression)
        {
            if (self.have_chunk_to_compress_potentially(CHUNK_PMP_RANK_BOUND))
            {
                std::fprintf(stderr, "[COMPRESS-AUDIT] blocks left to compress after final flush "
                                     "(non-aggressive) — compression pass skipped some blocks. "
                                     "A few is benign; a recurring/large residual is not.\n");
                std::fflush(stderr);
            }
        }

        // storage.destroy() frees every op still queued, unapplied. If the counters say any
        // remain, those writes are about to be lost: report it in every build.
        if (CompressionManager<T, CHUNK_T<T>>::any_pending_ops()) [[unlikely]]
        {
            if constexpr (build_config::any_check)
            {
                CHUNK_INVARIANT_FAILED("[DRAIN] flush_all_deferred: deferred ops still queued after {} pass(es) -- "
                                       "storage.destroy() would discard them: their writes would be LOST",
                                       pass);
            }
            std::fprintf(stderr,
                         "[DRAIN] flush_all_deferred: deferred ops still queued after %d pass(es) -- "
                         "storage.destroy() discards them: their writes are LOST\n",
                         pass);
            std::fflush(stderr);
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
template <template <typename> typename CHUNK_T, typename T>
void ManagerChunkParallelImpl<CHUNK_T, T, true>::alloc_chunk_allocator_thr()
{
    using mgr        = ManagerChunk<CHUNK_T, T>;
    const auto& self = static_cast<const mgr&>(*this);

    // std-alloc: allocator_data is NullValueAllocator (no per-thread array to build).
    if constexpr (build_config::threaded && build_config::use_fast_alloc)
    {
        // Built whole: an allocator is not movable.
        *CHUNK_T<T>::allocator_data = std::vector<TCTypeCAllocator<T>>(static_cast<std::size_t>(self.max_thr));
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
template <template <typename> typename CHUNK_T, typename T>
void ManagerChunkParallelImpl<CHUNK_T, T, true>::init_allocators([[maybe_unused]] int block_size) const
{
    // std-alloc: no per-thread slab allocator to init. Guard the whole body:
    // get_chunk_allocator() has no return on its std branch (it std::abort()s),
    // so instantiating this call in a std build would fail return-type deduction.
    if constexpr (build_config::use_fast_alloc)
    {
        using mgr         = ManagerChunk<CHUNK_T, T>;
        const int max_thr = static_cast<const mgr&>(*this).max_thr;

        // Also recorded here: an array may be created before any runtime size is registered,
        // and register_alloc_key() needs the slab budget.
        mgr::registered_block_size_ = block_size;

        constexpr int v = std::max(static_cast<int>(8 / sizeof(T)), 1);

        for (int thr_rank = 0; thr_rank < max_thr; ++thr_rank)
        {
            // -1 = never munmap a value slab: its address could be reobtained by the op pool's
            // fast_alloc::map_anonymous() and overwrite a live op (or vice-versa). Value slabs
            // are long-lived, so the retained memory is modest.
            auto& allocator = CHUNK_T<T>::get_chunk_allocator(thr_rank);
            allocator.set_rank(thr_rank); // stamped into every slab as owner_rank
            allocator.init(CHUNK_T<T>::storage_units(shrink_array::alloc_chunk_count()), block_size / v, -1);
        }
    }
}

/**
 * @brief Initialise this type's single chunk allocator (serial build).
 * @param block_size Slab size in bytes (ParamChunk::block_size).
 * @note The OpenMP counterpart is init_allocators(), one allocator per rank.
 */
template <template <typename> typename CHUNK_T, typename T>
void ManagerChunkParallelImpl<CHUNK_T, T, false>::init_allocator(int block_size) const
{
    // std-alloc: allocator_data is NullValueAllocator, init()/set_arena() are no-ops.
    // -1 = never munmap a value slab, and the Value arena rather than the Op
    // default, so a value pool never shares slabs with the deferred-op pool
    // (see register_chunk_size_serial in chunk_base.h; this is the static /
    // unregistered-size fallback allocator).
    CHUNK_T<T>::allocator_data->set_arena(fast_alloc::ArenaKind::Value);
    CHUNK_T<T>::allocator_data->init(CHUNK_T<T>::storage_units(shrink_array::alloc_chunk_count()), block_size, -1);
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
template <template <typename> typename CHUNK_T, typename T>
void ManagerChunkDynImpl<CHUNK_T, T, true>::register_chunk_size(unsigned int size, [[maybe_unused]] int block_size)
{
    using mgr                   = ManagerChunk<CHUNK_T, T>;
    const int this_thr          = static_cast<const mgr&>(*this).max_thr;
    mgr::registered_block_size_ = block_size; // for register_alloc_key()

    if constexpr (build_config::use_fast_alloc)
    {
        if constexpr (build_config::threaded)
        {
            // Per-thread chunk allocators for `size`.
            CHUNK_T<T>::register_chunk_size(size, this_thr, block_size);
        }
        else
        {
            // The single per-size allocator; old and new sizes coexist in
            // allocator_pool_serial_ during rechunk_refill.
            CHUNK_T<T>::register_chunk_size_serial(size, block_size);
        }
    }
    // BOTH modes, BOTH alloc backends: the per-size reuse free-list needs a slot for `size`.
    // Without it the by-size lookup falls back to the (non-OMP: unsized) legacy list:
    // drain_reuse_list_for_size reads a garbage vector and buffers of different geometries
    // get mixed (overflow in init_block). max_thr is CHUNK_PMP_MAX_THREADS (1 in non-OMP).
    mgr::register_free_list_size(size, this_thr);
}

/**
 * @brief Destroy this type's single chunk allocator (serial build).
 */
template <template <typename> typename CHUNK_T, typename T>
void ManagerChunkParallelImpl<CHUNK_T, T, false>::delete_chunk_allocator()
{
    // std-alloc: allocator_data is NullValueAllocator, destroy() is a no-op.
    CHUNK_T<T>::allocator_data->destroy();
}

template <template <typename> typename CHUNK_T, typename T>
void ManagerChunk<CHUNK_T, T>::init_static_state()
{
    CompressionManager<T, CHUNK_T<T>>::ensure_initialised();
    (void)CHUNK_T<T>::get_manager(); // built now, outside any parallel region of the type
}

/**
 * @brief Release this type's allocation blocks.
 * @note Serial builds only. Under OpenMP the per-thread allocators own the
 *       blocks and are torn down through their own path, so the body is empty
 *       there.
 */
template <template <typename> typename CHUNK_T, typename T>
void ManagerChunk<CHUNK_T, T>::destroy_block_chunk()
{
    // allocator_data is declared in every build (chunk_base.h) and is a dependent name, so the
    // discarded branch is never instantiated. std-alloc: destroy() is a no-op.
    if constexpr (build_config::not_threaded)
    {
        CHUNK_T<T>::allocator_data->destroy();
    }
}

/**
 * @brief Redistribute released buffers across the per-thread free-lists.
 *
 * A thread that frees far more than it allocates would otherwise hoard buffers
 * its neighbours need. Delegates to the free function that owns the policy.
 */
template <template <typename> typename CHUNK_T, typename T>
void ManagerChunkParallelImpl<CHUNK_T, T, true>::load_balance_list_block_free()
{
    load_balance_list_block_free_thr<T>();
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
    bool operator()(const T* lhs, const T* rhs) const noexcept
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
 * @post ManagerChunk::teardown_to_free_ has that many empty buckets and
 *       teardown_repeats_ is empty.
 */
template <template <typename> typename CHUNK_T, typename T>
std::size_t ManagerChunkParallelImpl<CHUNK_T, T, true>::clear_list_block_free_prepare()
{
    using mgr        = ManagerChunk<CHUNK_T, T>;
    const auto& self = static_cast<const mgr&>(*this);

    auto nb = static_cast<std::size_t>(std::max(1, self.max_thr));
    nb      = std::max(nb, mgr::get_list_block_free_thr().size());
    if constexpr (build_config::chunk_dynamic)
    {
        for (const auto& size_kv : *mgr::list_block_free_thr_by_size_)
        {
            nb = std::max(nb, size_kv.second.size());
        }
    }

    for (const auto& key_kv : *mgr::list_block_free_thr_keyed_)
    {
        nb = std::max(nb, key_kv.second.size());
    }
    mgr::teardown_to_free_.assign(nb, std::vector<T*>{});
    // One repeat list per address slice, and the slices are the team, whose
    // size is only known inside the region: max_thr bounds it.
    mgr::teardown_repeats_.assign(std::max(nb, static_cast<std::size_t>(CHUNK_PMP_MAX_THREADS)),
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
template <template <typename> typename CHUNK_T, typename T>
void ManagerChunkParallelImpl<CHUNK_T, T, true>::clear_list_block_free_screen(int bucket)
{
    using mgr     = ManagerChunk<CHUNK_T, T>;
    auto& to_free = mgr::teardown_to_free_;

    if (bucket < 0 || static_cast<std::size_t>(bucket) >= to_free.size())
    {
        return;
    }
    const auto bucket_idx = static_cast<std::size_t>(bucket);
    auto&      out        = to_free[bucket_idx];

    // RAW default list container, NOT the size-aware overload. It may be smaller than max_thr
    // (see allocate_chunk_internal): an unsized slot has nothing to drain.
    if (auto& legacy = mgr::get_list_block_free_thr(); bucket_idx < legacy.size())
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
        for (auto& size_kv : *mgr::list_block_free_thr_by_size_)
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
    for (auto& key_kv : *mgr::list_block_free_thr_keyed_)
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
template <template <typename> typename CHUNK_T, typename T>
void ManagerChunkParallelImpl<CHUNK_T, T, true>::clear_list_block_free_sort([[maybe_unused]] int bucket)
{
    if constexpr (!build_config::free_guard)
    {
        return;
    }

    using mgr     = ManagerChunk<CHUNK_T, T>;
    auto& to_free = mgr::teardown_to_free_;

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
            std::fprintf(stderr,
                         "[DOUBLE-FREE] %p queued twice for free in bucket %d "
                         "-- NOT freeing again\n",
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
template <template <typename> typename CHUNK_T, typename T>
void ManagerChunkParallelImpl<CHUNK_T, T, true>::clear_list_block_free_dedup([[maybe_unused]] int slice,
                                                                             [[maybe_unused]] int nslices)
{
    if constexpr (!build_config::free_guard)
    {
        return;
    }

    using mgr           = ManagerChunk<CHUNK_T, T>;
    const auto& to_free = mgr::teardown_to_free_;
    const auto  less    = internal::PtrLess<T>{};

    if (slice < 0 || nslices <= 0 || slice >= nslices)
    {
        return;
    }
    if (static_cast<std::size_t>(slice) >= mgr::teardown_repeats_.size())
    {
        // A team larger than max_thr: prepare() could not size for it.
        std::fprintf(stderr, "[DOUBLE-FREE] teardown dedup: slice %d of %d has no repeat list, skipped\n", slice,
                     nslices);
        return;
    }
    const std::size_t k = to_free.size();
    if (k < 2)
    {
        return; // a single bucket has no cross-bucket repeat, and sort already did it
    }

    // Slice bounds: nslices-1 samples per bucket, sorted, one splitter every
    // nslices-th sample. Every call computes the same splitters from the same
    // (read-only) input, which beats a single + barrier to share them.
    std::vector<T*> samples;
    samples.reserve(k * static_cast<std::size_t>(nslices));
    for (const auto& b : to_free)
    {
        for (int j = 1; j < nslices; ++j)
        {
            if (!b.empty())
            {
                samples.push_back(b[b.size() * static_cast<std::size_t>(j) / static_cast<std::size_t>(nslices)]);
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

    // Heads of the k-way merge: (pointer, bucket, position), min-heap on
    // (pointer, bucket) so that a pointer present in two buckets comes out
    // lower bucket first -- the repeat is then always the higher one.
    struct Head
    {
        T*          p;
        std::size_t bucket;
        std::size_t pos;
        std::size_t end;
    };
    const auto greater = [&](const Head& lhs, const Head& rhs)
    { return less(rhs.p, lhs.p) || (lhs.p == rhs.p && lhs.bucket > rhs.bucket); };

    std::vector<Head> heap;
    heap.reserve(k);
    for (std::size_t bi = 0; bi < k; ++bi)
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
    std::ranges::make_heap(heap, greater);

    auto&       repeats     = mgr::teardown_repeats_[static_cast<std::size_t>(slice)];
    const T*    prev        = nullptr;
    std::size_t prev_bucket = 0;
    bool        have        = false;
    while (!heap.empty())
    {
        std::ranges::pop_heap(heap, greater);
        Head head = heap.back();
        heap.pop_back();

        if (have && head.p == prev)
        {
            // Same pointer as the one just produced from a lower bucket: this
            // one is the repeat, skipped when ITS bucket is freed.
            std::fprintf(stderr,
                         "[DOUBLE-FREE] %p queued for free in buckets %zu and %zu "
                         "-- NOT freeing again\n",
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
            std::ranges::push_heap(heap, greater);
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
template <template <typename> typename CHUNK_T, typename T>
void ManagerChunkParallelImpl<CHUNK_T, T, true>::clear_list_block_free_thr(int bucket)
{
    using mgr     = ManagerChunk<CHUNK_T, T>;
    auto& to_free = mgr::teardown_to_free_;

    if (bucket < 0 || static_cast<std::size_t>(bucket) >= to_free.size())
    {
        return;
    }
    const auto bucket_idx = static_cast<std::size_t>(bucket);
    auto&      blocks     = to_free[bucket_idx];

    std::unordered_set<T*> skip;
    for (const auto& per_slice : mgr::teardown_repeats_)
    {
        for (const auto& rep : per_slice)
        {
            if (rep.bucket == bucket_idx)
            {
                skip.insert(rep.ptr);
            }
        }
    }

    // Tell the ledger before each svfree, or every block released here stays PARKED in the
    // ledger and floods CHUNK_LEDGER_REPORT with false positives.
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
template <template <typename> typename CHUNK_T, typename T>
void ManagerChunkParallelImpl<CHUNK_T, T, true>::delete_list_block_chunk_free_itself()
{
    using mgr = ManagerChunk<CHUNK_T, T>;

    if constexpr (IsChunk2Or2A<CHUNK_T, T>)
    {
        mgr::list_block_free_thr.clear();
    }
    else
    {
        CHUNK_T<T>::manager_chunk::get_list_block_free_thr(0).clear();
    }
}

/**
 * @brief Is the imbalance between per-thread free-lists worth a redistribution pass?
 * @return True when load_balance_list_block_free() would be worth calling.
 */
template <template <typename> typename CHUNK_T, typename T>
bool ManagerChunkParallelImpl<CHUNK_T, T, true>::need_load_balance_list_block_free() const
{
    return need_load_balance_list_block_free_thr<T>();
}

template <template <typename> typename CHUNK_T, typename T>
bool ManagerChunk<CHUNK_T, T>::register_alloc_key([[maybe_unused]] unsigned int key,
                                                  [[maybe_unused]] unsigned int size,
                                                  [[maybe_unused]] int          nb_chunk)
{
    if constexpr (build_config::chunk_dynamic && build_config::use_fast_alloc && build_config::threaded)
    {
        // Without a slab budget there is nothing to build the keyed allocator
        // set from. Report the failure: the caller must NOT keep the key, or
        // every allocation of that array would route to a set that does not
        // exist and fall back to the legacy single allocator.
        if (key == 0 || registered_block_size_ == 0)
        {
            return false;
        }
        // The per-rank sets are sized for every rank that may ever run (the
        // capacity); the budget is shared by the team that runs now.
        const int capacity = CHUNK_PMP_THREAD_CAPACITY;
        const int max_thr  = CHUNK_PMP_MAX_THREADS;
        CHUNK_T<T>::register_chunk_size(size, capacity, registered_block_size_, key);
        register_free_list_key(key, size, capacity);
        // The array's own share per rank, where the global budget uses the
        // process-wide somme_chunk_create.
        set_size_buffer_reuse(key, static_cast<int>(nb_chunk / (reuse_share_div() * max_thr)), size);
        if (std::getenv("CHUNK_AFFINITY_TRACE") != nullptr)
        {
            std::fprintf(stderr, "[AFFINITY] key=%u size=%u nb_chunk=%d budget=%d threads=%d\n", key, size, nb_chunk,
                         get_size_buffer_reuse(key), max_thr);
            std::fflush(stderr);
        }
        return true;
    }
    else
    {
        return false;
    }
}

// compute_size_buffer_reuse (documented in manager_chunk.h): derives the reuse
// budget from the number of chunks created so far, so a program that allocates
// more keeps proportionally more buffers parked for reuse.
template <template <typename> typename CHUNK_T, typename T>
void ManagerChunk<CHUNK_T, T>::compute_size_buffer_reuse()
{
    // Chunk2ADef's reuse constants serve every variant (chunk2A_def.h);
    // CHUNK_REUSE_SHARE_DIV / CHUNK_REUSE_FLOOR_KB override them.
    size_buffer_reuse = static_cast<int>(somme_chunk_create / (reuse_share_div() * CHUNK_PMP_MAX_THREADS));
    size_buffer_reuse =
        std::max(size_buffer_reuse, reuse_floor_buffers(alloc_chunk_count(), Chunk2ADef::size_min_buffer_reuse));
}

// compute_size_max_number_of_chunk_to_check_compress (documented in
// manager_chunk.h): sets the threshold past which a rank's candidate list is
// flagged dangerous -- the szm that ProxyBase::should_trigger_compression
// compares against.
template <template <typename> typename CHUNK_T, typename T>
void ManagerChunk<CHUNK_T, T>::compute_size_max_number_of_chunk_to_check_compress()
{
    auto& global_stats = SingletonChk<GlobalStatsChunkList>::getRef();

    // Take the tuning constants from THIS manager's own variant: they differ a great deal
    // (see ChunkDef / Chunk2Def / Chunk2ADef). The value is divided by the thread count, so at
    // high thread counts the floor IS the threshold. Another variant's higher floor means a
    // rank's list almost never reaches it: compression almost never runs during execution and
    // every materialisation allocates afresh.
    using VariantDef = std::conditional_t<IsChunk2A<CHUNK_T, T>, Chunk2ADef,
                                          std::conditional_t<IsChunk2<CHUNK_T, T>, Chunk2Def, ChunkDef>>;

    int threshold = somme_chunk_create / (VariantDef::factor_div_nb_chunk_for_compress * CHUNK_PMP_MAX_THREADS);
    threshold     = std::max(threshold, VariantDef::size_min_list_chunk_to_check_compress);

    set_size_max_number_of_chunk_to_check_compress(threshold);
    const size_t element_size = sizeof(std::tuple<CHUNK_T<T>*, int, int>);

    global_stats.record(type_name<CHUNK_T<T>>(), somme_chunk_create, threshold, element_size);
}

} // namespace shrink_array
