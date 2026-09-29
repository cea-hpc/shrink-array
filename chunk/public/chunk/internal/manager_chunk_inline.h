#pragma once

// Template method definitions for ManagerChunk<CHUNK_T, T>. Kept in a dedicated
// inline header (included at the end of chunk/internal/manager_chunk.h) so that every
// (CHUNK_T, T) the consumer uses auto-instantiates at its use site. Bodies are
// the CURRENT branch versions (with the kMaxDrainPasses drain and the defensive
// destroy_after_flush drain) — NOT the older gen-dev-27 versions.

#include <atomic>
#include <cstdio>
#include <cstdlib> // std::getenv (the trace switches)
#include <print>   // std::println (DEBUG_SET_SIZE)
#include <algorithm>
#include <functional>
#include <tuple> // std::tuple (the memory estimate)
#include <type_traits>
#include <unordered_set>
#include <vector>

#include "chunk/internal/manager_chunk.h" // the declarations these bodies belong to
#include "chunk/config.h"
#include "chunk/internal/svalloc.h" // svfree
#include "chunk/PMP_CHUNK.h"
#include "chunk/internal/chunk_size_policy.h"
#include "chunk/internal/compression_manager.h"  // CompressionManager (the barrier flushes)
#include "chunk/internal/deferred_op_pool.h"     // fast_alloc::ArenaKind (its std-alloc stand-in too)
#include "chunk/internal/deferred_ops_storage.h" // DeferredOpsStorage
#include "chunk/internal/pool_ledger.h"          // CHUNK_LEDGER_DISCARD
#include "chunk/chunk_std.h"
#include "chunk/chunk2_std.h"
#include "chunk/chunk2A_std.h"
#include "chunk/chunk_def.h"
#include "chunk/chunk2_def.h"
#include "chunk/chunk2A_def.h"
#include "chunk/array_chunk.h"
#include "chunk/array_chunk2.h"
#include "chunk/internal/array_chunk_for.h"
#include "chunk/internal/concept_chunk.h"
#include "chunk/global_stats_chunk_list.h"
#include "chunk/type_name.h"
#include "chunk/internal/chunk_func.h"
#include "chunk/singleton.h"

namespace shrink_array
{
// Forward declarations breaking the include cycle
// chunk_func.h -> chunk_std.h -> proxy.h -> manager_chunk.h -> THIS file ->
// chunk_func.h: when chunk_func.h is the ENTRY point, its include of
// chunk_std.h re-enters this file while chunk_func.h's own templates are not
// yet declared (#pragma once skips the in-progress header), and the explicit
// load_balance_list_block_free_thr<T>() calls below need a visible template
// declaration. Definitions live in chunk_func.h.
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
 *          compression falls on the opportunistic in-loop path. The inline
 *          comment on the predicate records the GNU build where that happened.
 *
 * @see manager_chunk.h for ManagerChunkInt, the type-erased interface, and for
 *      the free-list containers these bodies manipulate.
 * @see all_type_chunk.h for the registry that drives every manager at once.
 */

namespace shrink_array
{

// Defined (once) in ManagerChunk.cpp. Declared here because flush_all_deferred's
// body now lives in this header and is compiled in TUs that lack the definition.
extern std::atomic<bool> __global_need_flush_deferred;
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
    const auto lbd = [&]
    {
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
    };

    lbd();
    // (A CHUNK_PMP_IN_PARALLEL_REGION-conditional dispatch was tried here but hit
    //  an llvm OpenMP runtime bug; the plain lbd() above is used instead.)
}

/**
 * @brief Sizes this type's compression-candidate containers to the current team.
 *
 * Which containers exist depends on the variant (see the file header): a plain
 * per-rank list for Chunk and Chunk2A, plus the per-second-value map for
 * Chunk2. A list keeps its capacity across the passes' clear(), so nothing
 * pre-reserves it (the per-rank "last size" history that once did, at every
 * parallel-begin under OMPT, was a no-op and is gone).
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
    if constexpr (build_config::use_openmp)
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
 *          branch for branch — see the @warning in the file header, and the
 *          inline comment below, which records what a disagreement cost.
 */
template <template <typename> typename CHUNK_T, typename T>
bool ManagerChunk<CHUNK_T, T>::have_chunk_to_compress_potentially(int n_thr) const
{
    // Mirror compress_chunk_to_compress_potentially's dispatch exactly: it handles
    // chunk2 whenever use_fast_block_compression is on (GNU embedded included),
    // not only under OMPT. This predicate must agree, otherwise in GNU it would
    // report "nothing to compress" for chunk2 and the between-phase compression
    // pass would never run for it (the opportunistic in-loop path would carry
    // all of it, with its non-deterministic stalls).
    if constexpr (chunk_config::use_fast_block_compression)
    {
        if constexpr (IsChunk2<CHUNK_T, T>)
        {
            if constexpr (is_not_bool)
            {
                const auto& sh_inf           = Proxy2<T>::share_info_array;
                const auto& v_all_second_val = sh_inf.get_all_second_value();
                for (const auto& second_value : v_all_second_val)
                {
                    if (CHUNK_T<T>::have_chunk_to_compress(second_value, n_thr))
                        return true;
                }
                return false;
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
                const auto& sh_inf          = Proxy2<T>::share_info_array;
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
 * @note Under `build_config::any_check` this first walks the maps looking for
 *       ops that were enqueued but never drained. It does NOT abort on them: it
 *       reports, then applies and clears them, deliberately, so a run can
 *       continue far enough to show whether the leftovers were a harmless race
 *       or real corruption upstream.
 */
template <template <typename> typename CHUNK_T, typename T>
void ManagerChunkParallelImpl<CHUNK_T, T, true>::destroy_after_flush(bool in_parallel)
{
    using Storage = DeferredOpsStorage<typename CHUNK_T<T>::per_thread_map>;
    auto& storage = Storage::instance();

    if constexpr (build_config::any_check)
    {
        // Defensive drain: log any leftovers (they represent ops that were
        // enqueued via to_be_queued_for_deferred_processing but never
        // drained by flush_deferred). Then execute them and clear the maps
        // rather than aborting. This is a diagnostic mode: it lets the
        // run continue so we can see whether the downstream data behavior
        // is correct (leftovers were harmless races) or corrupt (real bug
        // upstream).
        auto* arr = storage.get();
        if (arr)
        {
            const int max_thr = CHUNK_PMP_RANK_BOUND;

            // Drain arr[t] fully. Crucially, op_ptr->destroy() recycles the op
            // block into the free_list of whichever thread PHYSICALLY RUNS this
            // lambda (free_list()/allocator() are thread_local). So if we run it
            // on the thread that currently owns rank t, the block recycles
            // LOCALLY — no foreign block, no "[RECYCLE] dropping FOREIGN block"
            // drop, no per-phase leak. There is no way to push into another
            // thread's thread_local free_list, so draining-on-owner is the only
            // route that keeps recycling local.
            const auto drain_index = [&](int rank)
            {
                if (rank < 0 || rank >= max_thr || arr[rank].empty())
                    return;

                std::fprintf(
                    stderr,
                    "[LEAK] destroy_after_flush: thr=%d has %zu chunks with deferred ops (draining defensively)\n",
                    rank, arr[rank].size());

                for (auto& [chunk_ptr, ops_map_ptr] : arr[rank])
                {
                    if (!ops_map_ptr)
                        continue;
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
// Route-to-owner drain. destroy_after_flush(false) is only ever
// called in serial context (in_parallel == false at every call
// site — verified: finalize_parallel_phase_after_flush runs AFTER
// the enclosing #pragma omp parallel closes), so opening a region
// here does not nest. Each thread drains ONLY its own index
// (disjoint arr[t]) → data-race-free, and destroy() recycles into
// that thread's own pool. Compiled out under OMPT, where opening a
// region would fire the parallel_end compression callback
// that must not run from a teardown drain.
#pragma omp parallel
                {
                    drain_index(CHUNK_PMP_THREAD_RANK);
                }
            }

            // Serial mop-up. Covers: (a) the OMPT build, where the parallel
            // block above is compiled out; (b) any index not covered by the
            // team when team_size < max_thr. Indices already drained above are
            // now empty → skipped. Here the calling thread runs any remaining
            // foreign ops, so recycle() may drop foreign blocks exactly as
            // before — the same bounded, safe fallback as the original code.
            for (int t = 0; t < max_thr; ++t)
                drain_index(t);

            std::fflush(stderr);
        }
    }
    using ArrayType = ArrayChunkFor_t<CHUNK_T<T>>;

    if (in_parallel)
    {
#pragma omp single // nowait // useless risk ?
        {
            storage.destroy();

            ArrayType::clear_all_dirty_bitsets();
        }
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
 * Clears `__global_need_flush_deferred` for the duration and restores it on
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

    // The global flag keeps the barrier callback out of the SERIAL flush's own
    // nested drain regions; only that flush may toggle it. When the in-region
    // flush toggled it too, 64 threads stored into one line at every barrier,
    // and a thread reading it mid-pass skipped its OWN flush.
    const bool toggle_flag = !in_parallel;
    if (toggle_flag)
        __global_need_flush_deferred.store(false, std::memory_order_relaxed);
    if (in_parallel)
    {
        const int thr_rank = CHUNK_PMP_THREAD_RANK;
        // Own-rank early-out: pending_ops_rank(r) is raised by rank r's own
        // thread in the enqueue call that inserts each new op, and lowered
        // only after an op is drained or erased; ops are always enqueued on
        // the writer's rank. So a zero read HERE, by rank r's thread, cannot hide an op this
        // thread queued. A stale non-zero (a concurrent cross-rank drain) only
        // takes the locked path below. 6.3 M flush attempts at 64 threads,
        // none of which applied an op, is what this skips.
        if (CompressionManager<T, CHUNK_T<T>>::pending_ops_rank(thr_rank) != 0)
            have_flushed |= CompressionManager<T, CHUNK_T<T>>::flush_deferred(thr_rank);
    }
    else
    {
        bool local_flushed = false;

        // FIX (deferred use-after-free at reset): drain until ALL op_maps are
        // empty BEFORE storage.destroy(). A single flush_deferred pass does one
        // swap-drain and leaves reentrant re-enqueues (a write_body can enqueue a
        // new compression op) in op_map. If we destroy the storage now, those
        // surviving op_ptrs reference freed slabs -> use-after-free / SEGV when a
        // later flush reads them. We are at a synchronization point (end of
        // phase, application writes finished), so re-enqueues converge; loop with
        // a generous safety bound to avoid any pathological non-termination.
        constexpr int kMaxDrainPasses = 1024;
        bool          any_left        = true;
        int           pass            = 0;
        for (; any_left && pass < kMaxDrainPasses; ++pass)
        {
            // No op queued on any rank: no region. Exact here -- we are outside
            // any team, after its join, so every enqueue's increment is
            // visible to this acquire sum; and every op in a map is counted
            // (one insertion site, increment per new entry; decrements only
            // after removal). A skipped pass would only have dropped chunk-set
            // keys and a "may hold ops" hint. Was: one fork/join of the whole
            // team per type and per pass regardless -- 92% of the run's
            // parallel regions at 64 threads.
            if (!CompressionManager<T, CHUNK_T<T>>::any_pending_ops())
                break;

            bool pass_flushed = false;
#pragma omp parallel reduction(|| : pass_flushed)
            {
                const int thr_rank = CHUNK_PMP_THREAD_RANK;
                pass_flushed |= CompressionManager<T, CHUNK_T<T>>::flush_deferred(thr_rank);
            }
            local_flushed |= pass_flushed;
            // Another pass produced flushes? then re-enqueues may still exist.
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
        // After the end-of-phase drain, in a NON-aggressive build, every block
        // that became constant should have been picked up and compressed — the
        // "potentially to compress" set should be empty. If haveChunkToCompress-
        // Potentially() still reports work, the compression pass (OMPT hook or
        // its GNU _g equivalent) let some blocks slip through. Debug/check-only
        // (any_check), and only meaningful outside aggressive mode: under
        // have_agressive_compression the set is legitimately churning (permanent
        // compression) and the check would false-positive. Non-fatal by design —
        // this is an observability probe, not a correctness gate: a few
        // stragglers are tolerable (opportunistic in-loop compression can lag),
        // a persistent residual across runs means the pass structurally skips
        // blocks. We report presence; the caller can escalate if it recurs.
        if constexpr (build_config::any_check && !chunk_config::have_agressive_compression)
        {
            if (self.have_chunk_to_compress_potentially(CHUNK_PMP_RANK_BOUND))
            {
                std::fprintf(stderr, "[COMPRESS-AUDIT] blocks left to compress after final flush "
                                     "(non-aggressive) — compression pass skipped some blocks. "
                                     "A few is benign; a recurring/large residual is not.\n");
                std::fflush(stderr);
            }
        }

        storage.destroy();
    }

    if (toggle_flag)
        __global_need_flush_deferred.store(true, std::memory_order_release);
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
    if constexpr (build_config::use_openmp && build_config::use_fast_alloc)
    {
        // Built whole: an allocator is not movable.
        CHUNK_T<T>::allocator_data = std::vector<TCTypeCAllocator<T>>(static_cast<std::size_t>(self.max_thr));
    }
}

/**
 * @brief Set every rank's chunk allocator geometry.
 *
 * @param block_size Number of chunks per allocation block, scaled down for wide
 *        element types so a block stays a comparable number of BYTES rather
 *        than of elements.
 * @note Fast-alloc builds only. Sequential, from whichever thread registers
 *       the manager: an allocator is a slot indexed by rank, not a thread
 *       local, and init() only records the geometry -- the slab is taken at
 *       the rank's first allocation. This used to run inside a parallel
 *       region, one call per rank, at ParamChunk::init_indexes(); a type is
 *       now registered by its first array, from any context.
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

        // Also recorded here, not only in register_chunk_size(): an array may be
        // created before any runtime size is registered, and register_alloc_key()
        // needs the slab budget to build its keyed allocator set.
        mgr::registered_block_size_ = block_size;

        constexpr int v = std::max(static_cast<int>(8 / sizeof(T)), 1);

        for (int thr_rank = 0; thr_rank < max_thr; ++thr_rank)
        {
            // FIX (op/value slab aliasing): -1 = never munmap a value slab. A freed
            // value slab's address could be reobtained by the op pool's
            // sctk_mmap(nullptr) and overwrite a live op (or vice-versa). Value slabs
            // are long-lived (data), so few become empty and retained memory is modest.
            auto& allocator = CHUNK_T<T>::get_chunk_allocator(thr_rank);
            allocator.set_rank(thr_rank); // stamped into every slab as owner_rank
            allocator.init(CHUNK_T<T>::storage_units(shrink_array::alloc_chunk_count()), block_size / v, -1);
        }
    }
}

/**
 * @brief Initialise this type's single chunk allocator (serial build).
 * @param block_size Number of chunks per allocation block.
 * @note The OpenMP counterpart is init_allocators(), one allocator per rank.
 */
template <template <typename> typename CHUNK_T, typename T>
void ManagerChunkParallelImpl<CHUNK_T, T, false>::init_allocator(int block_size) const
{
    // std-alloc: allocator_data is NullValueAllocator, init()/set_arena() are no-ops.
    // FIX (op/value slab aliasing): -1 = never munmap. See OpenMP variant above.
    // FIX (op/value slab aliasing, arena routing): tag this pool Value so it draws
    // from value_arena(), not the Op-arena default -- see register_chunk_size_serial
    // in chunk_base.h for the full rationale (same bug, this is the static/
    // non-CHUNK_DYNAMIC and unregistered-size-fallback allocator).
    CHUNK_T<T>::allocator_data.set_arena(fast_alloc::ArenaKind::Value);
    CHUNK_T<T>::allocator_data.init(CHUNK_T<T>::storage_units(shrink_array::alloc_chunk_count()), block_size, -1);
}

/**
 * @brief Declare a runtime chunk geometry to this type's allocators.
 *
 * Two things happen, and BOTH are needed: the fast allocator gets an allocator
 * for that element count, and the manager gets a per-size reuse free-list slot.
 * The second is why this is called even in std-alloc builds, where there is no
 * allocator to create.
 *
 * Declared once (manager_chunk.h, under CHUNK_DYNAMIC), so it is defined once
 * too: the OpenMP and serial builds used to carry a copy each, differing only
 * in which allocator call they make, with the shared free-list rationale
 * written out twice in different words.
 *
 * @param size       Chunk size, in elements, being registered.
 * @param block_size Number of chunks per allocation block for that size.
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
        if constexpr (build_config::use_openmp)
        {
            // Per-thread chunk allocators for `size`.
            CHUNK_T<T>::register_chunk_size(size, this_thr, block_size);
        }
        else
        {
            // The single per-size allocator, so a runtime re-chunk can allocate
            // at the new element size. Populated for the creation size here and
            // again for each new size at re-chunk, so old and new coexist in
            // allocator_pool_serial_ during rechunk_refill.
            CHUNK_T<T>::register_chunk_size_serial(size, block_size);
        }
    }
    // BOTH modes, BOTH alloc backends: the per-size reuse free-list needs a slot
    // for `size`. Without it the by-size lookup falls back to the (in non-OMP,
    // unsized) legacy list -> drain_reuse_list_for_size dereferences a garbage
    // vector (SIGSEGV at the first re-chunk) and buffers of different geometries
    // get mixed (overflow in init_block). max_thr is CHUNK_PMP_MAX_THREADS
    // (1 in non-OMP), matching every by-size iterator.
    mgr::register_free_list_size(size, this_thr);
}

/**
 * @brief Destroy this type's single chunk allocator (serial build).
 */
template <template <typename> typename CHUNK_T, typename T>
void ManagerChunkParallelImpl<CHUNK_T, T, false>::delete_chunk_allocator()
{
    // std-alloc: allocator_data is NullValueAllocator, destroy() is a no-op.
    CHUNK_T<T>::allocator_data.destroy();
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
    // Serial builds own their single allocator here; the OpenMP ones tear the
    // per-thread allocators down through their own path. allocator_data is
    // declared in every build (chunk_base.h), and it is a dependent name, so the
    // discarded branch is never instantiated.
    // std-alloc: allocator_data is NullValueAllocator, destroy() is a no-op.
    if constexpr (build_config::not_use_openmp)
    {
        CHUNK_T<T>::allocator_data.destroy();
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
// Why the repeats matter (double-free hunt): a block parked in BOTH the legacy
// list and a per-size list (clear_core() switches destination once
// register_runtime_chunk_size has created a per-size list for that geometry,
// so a buffer can be filed on either side across a rechunk) is freed twice:
// glibc aborts with "corrupted double-linked list" in _int_free. The sort and
// dedup phases catch every such repeat -- same rank or not -- BEFORE the first
// free, report each one, and free it once. This used to be a single
// unordered_set fed sequentially by one thread; sorting per bucket and merging
// per address slice does the same job with every thread busy.
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
        for (const auto& size_kv : mgr::list_block_free_thr_by_size_)
            nb = std::max(nb, size_kv.second.size());
    }

    for (const auto& key_kv : mgr::list_block_free_thr_keyed_)
        nb = std::max(nb, key_kv.second.size());
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
        return;
    const auto bucket_idx = static_cast<std::size_t>(bucket);
    auto&      out        = to_free[bucket_idx];

    // The legacy store is not guaranteed to be sized to max_thr (see
    // allocate_chunk_internal): a slot that was never sized simply has nothing
    // to drain.
    // RAW default list container, NOT the size-aware overload.
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

    // Also drain the SIZED reuse free-lists. The array teardown (clear_core)
    // parks OLD/live-geometry buffers here, keyed by
    // size_policy_.size(). drain_reuse_list_for_size releases a given size at
    // each rechunk, but at PROGRAM END no rechunk runs, so the FINAL
    // geometry's buffers sit in this map unfreed -> ASan direct-leak of one
    // array's worth of chunks. Applies to v1 as much as to chunk2/2A -- v1
    // once lacked it and leaked its final geometry on every re-chunked dyn run.
    if constexpr (build_config::chunk_dynamic)
    {
        for (auto& size_kv : mgr::list_block_free_thr_by_size_)
        {
            if (bucket_idx >= size_kv.second.size())
                continue;
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
    for (auto& key_kv : mgr::list_block_free_thr_keyed_)
    {
        if (bucket_idx >= key_kv.second.size())
            continue;
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
        return;

    using mgr     = ManagerChunk<CHUNK_T, T>;
    auto& to_free = mgr::teardown_to_free_;

    if (bucket < 0 || static_cast<std::size_t>(bucket) >= to_free.size())
        return;
    auto& blocks = to_free[static_cast<std::size_t>(bucket)];

    std::ranges::sort(blocks, internal::PtrLess<T>{});
    for (std::size_t i = 1; i < blocks.size(); ++i)
    {
        if (blocks[i] == blocks[i - 1])
            std::fprintf(stderr,
                         "[DOUBLE-FREE] %p queued twice for free in bucket %d "
                         "-- NOT freeing again\n",
                         static_cast<void*>(blocks[i]), bucket);
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
        return;

    using mgr           = ManagerChunk<CHUNK_T, T>;
    const auto& to_free = mgr::teardown_to_free_;
    const auto  less    = internal::PtrLess<T>{};

    if (slice < 0 || nslices <= 0 || slice >= nslices)
        return;
    if (static_cast<std::size_t>(slice) >= mgr::teardown_repeats_.size())
    {
        // A team larger than max_thr: prepare() could not size for it.
        std::fprintf(stderr, "[DOUBLE-FREE] teardown dedup: slice %d of %d has no repeat list, skipped\n", slice,
                     nslices);
        return;
    }
    const std::size_t k = to_free.size();
    if (k < 2)
        return; // a single bucket has no cross-bucket repeat, and sort already did it

    // Slice bounds: nslices-1 samples per bucket, sorted, one splitter every
    // nslices-th sample. Every call computes the same splitters from the same
    // (read-only) input, which beats a single + barrier to share them.
    std::vector<T*> samples;
    samples.reserve(k * static_cast<std::size_t>(nslices));
    for (const auto& b : to_free)
    {
        for (int j = 1; j < nslices; ++j)
            if (!b.empty())
                samples.push_back(b[b.size() * static_cast<std::size_t>(j) / static_cast<std::size_t>(nslices)]);
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
            heap.push_back(Head{*first, bi, static_cast<std::size_t>(first - blocks.begin()),
                                static_cast<std::size_t>(last - blocks.begin())});
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
 *          block is cheapest to release through the one it came from. Teardown
 *          on this workload put __lll_lock_wake_private high in the profile --
 *          64 ranks releasing through a single arena.
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
        return;
    const auto bucket_idx = static_cast<std::size_t>(bucket);
    auto&      blocks     = to_free[bucket_idx];

    std::unordered_set<T*> skip;
    for (const auto& per_slice : mgr::teardown_repeats_)
    {
        for (const auto& rep : per_slice)
            if (rep.bucket == bucket_idx)
                skip.insert(rep.ptr);
    }

    // Tell the ledger before each svfree: this exit drain was the only free
    // site in the manager without a hook, so every block it released stayed
    // PARKED in the ledger -- millions of false positives in the
    // CHUNK_LEDGER_REPORT output that masked any real leak.
    if (skip.empty())
    {
        for (auto* elm : blocks)
        {
            CHUNK_LEDGER_DISCARD(elm, "clear_list_block_free_thr.drain");
            svfree(elm);
        }
    }
    else
    {
        for (auto* elm : blocks)
            if (!skip.contains(elm))
            {
                CHUNK_LEDGER_DISCARD(elm, "clear_list_block_free_thr.drain");
                svfree(elm);
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

// compute_size_buffer_reuse (documented in manager_chunk.h): derives the reuse
// budget from the number of chunks created so far, so a program that allocates
// more keeps proportionally more buffers parked for reuse.
template <template <typename> typename CHUNK_T, typename T>
bool ManagerChunk<CHUNK_T, T>::register_alloc_key([[maybe_unused]] unsigned int key,
                                                  [[maybe_unused]] unsigned int size,
                                                  [[maybe_unused]] int          nb_chunk)
{
    if constexpr (build_config::chunk_dynamic && build_config::use_fast_alloc && build_config::use_openmp)
    {
        // Without a slab budget there is nothing to build the keyed allocator
        // set from. Report the failure: the caller must NOT keep the key, or
        // every allocation of that array would route to a set that does not
        // exist and fall back to the legacy single allocator.
        if (key == 0 || registered_block_size_ == 0)
            return false;
        // The per-rank sets are sized for every rank that may ever run (the
        // capacity); the budget is shared by the team that runs now.
        const int capacity = CHUNK_PMP_THREAD_CAPACITY;
        const int max_thr  = CHUNK_PMP_MAX_THREADS;
        CHUNK_T<T>::register_chunk_size(size, capacity, registered_block_size_, key);
        register_free_list_key(key, size, capacity);
        // The array's own share per rank, where the global budget uses the
        // process-wide somme_chunk_create.
        set_size_buffer_reuse(key, nb_chunk / (Chunk2ADef::factor_div_somme_chunk_buffer_resuse * max_thr));
        if (std::getenv("CHUNK_AFFINITY_TRACE"))
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
template <template <typename> typename CHUNK_T, typename T>
void ManagerChunk<CHUNK_T, T>::compute_size_buffer_reuse()
{
    // to change Chunk2ADef...
    size_buffer_reuse =
        static_cast<int>(somme_chunk_create /
                         (static_cast<long>(Chunk2ADef::factor_div_somme_chunk_buffer_resuse) * CHUNK_PMP_MAX_THREADS));
    size_buffer_reuse = std::max(size_buffer_reuse, Chunk2ADef::size_min_buffer_reuse);
}

/// @brief Print @p count and the size count * elem_size, in the largest unit reached (Go, Mo, Ko or bytes).
/// @param label Prefix of the printed line.
/// @param count Number of elements.
/// @param elem_size Size of one element, in bytes.
inline void print_mem_estimate(const char* label, std::size_t count, std::size_t elem_size)
{
    const long double bytes = static_cast<long double>(count) * static_cast<long double>(elem_size);

    const long double kib = bytes / 1024.0L;
    const long double mib = kib / 1024.0L;
    const long double gib = mib / 1024.0L;

    // {:g}: the six significant digits std::cout printed.
    if (gib >= 1.0L)
        std::println("{} : {} elements, approx {:g} Go", label, count, gib);
    else if (mib >= 1.0L)
        std::println("{} : {} elements, approx {:g} Mo", label, count, mib);
    else if (kib >= 1.0L)
        std::println("{} : {} elements, approx {:g} Ko", label, count, kib);
    else
        std::println("{} : {} elements, approx {:g} bytes", label, count, bytes);
}

// compute_size_max_number_of_chunk_to_check_compress (documented in
// manager_chunk.h): sets the threshold past which a rank's candidate list is
// flagged dangerous -- the szm that ProxyBase::should_trigger_compression
// compares against.
template <template <typename> typename CHUNK_T, typename T>
void ManagerChunk<CHUNK_T, T>::compute_size_max_number_of_chunk_to_check_compress()
{
    auto& global_stats = SingletonChk<GlobalStatsChunkList>::getRef();

    // Take the tuning constants from THIS manager's own variant.
    //
    // Both were spelled Chunk2Def:: in a function templated on CHUNK_T, so every
    // variant was tuned with Chunk2's numbers. At the levels this tree is
    // configured with they differ a great deal:
    //
    //     variant    floor   divisor
    //     Chunk        100        25
    //     Chunk2      2000        20
    //     Chunk2A      500        50
    //
    // A v1 manager therefore used a floor of 2000 instead of 100 -- twenty times
    // too high -- and Chunk2A 2000 instead of 500. The floor is not a detail: the
    // computed value is divided by the thread count, so at 64 threads it
    // collapses onto the floor and the floor IS the threshold. A rank's candidate
    // list had to reach 2000 entries before trigger_compression_if_needed() would
    // flag it, so compression almost never ran during execution, almost no buffer
    // was ever returned to the reuse pools, and every materialisation allocated
    // afresh.
    //
    // Measured on v1 dyn, 64 threads, before this fix: 66844665 fresh
    // allocations against 788282 compressions -- 85 to 1 -- and 33 GB resident
    // with the pools empty throughout the run and filled only at teardown.
    //
    // The `if constexpr (IsChunk2<...>)` left commented out just below says
    // someone had already seen this.
    using VariantDef = std::conditional_t<IsChunk2A<CHUNK_T, T>, Chunk2ADef,
                                          std::conditional_t<IsChunk2<CHUNK_T, T>, Chunk2Def, ChunkDef>>;

    int threshold = somme_chunk_create / (VariantDef::factor_div_nb_chunk_for_compress * CHUNK_PMP_MAX_THREADS);
    threshold     = std::max(threshold, VariantDef::size_min_list_chunk_to_check_compress);

    set_size_max_number_of_chunk_to_check_compress(threshold);
    if constexpr (build_config::debug_set_size)
    {
        std::println("somme_chunk_create : {}", somme_chunk_create);
        std::println("VariantDef::factor_div_nb_chunk_for_compress * CHUNK_PMP_MAX_THREADS : {}",
                     VariantDef::factor_div_nb_chunk_for_compress * CHUNK_PMP_MAX_THREADS);

        const int max_chunks = get_size_max_number_of_chunk_to_check_compress();
        std::println("size_max_number_of_chunk_to_check_compress : {}", max_chunks);

        if constexpr (build_config::use_openmp)
        {
            std::println("\n=== Estimated memory footprint per container ===");
            std::println("Threads (CHUNK_PMP_MAX_THREADS): {}", CHUNK_PMP_MAX_THREADS);
            std::println("Total estimated memory = above values × number of threads");

            if constexpr (IsChunk2<CHUNK_T, T>)
            {
                using Chunk2Cont = std::decay_t<decltype(chunk2_to_compress_potentially_thr[0])>;
                print_mem_estimate("chunk2_to_compress_potentially_thr (per thread)", max_chunks,
                                   sizeof(typename Chunk2Cont::value_type));
            }
            else if constexpr (IsChunk2A<CHUNK_T, T>)
            {
                using Chunk2ACont = std::decay_t<decltype(chunk2A_to_compress_potentially_thr[0])>;

                print_mem_estimate("chunk2A_to_compress_potentially_thr (per thread)", max_chunks,
                                   sizeof(typename Chunk2ACont::value_type));
            }
            else
            {
                using ChunkCont = std::decay_t<decltype(chunk_to_compress_potentially_thr[0])>;

                print_mem_estimate("chunk_to_compress_potentially_thr (per thread)", max_chunks,
                                   sizeof(typename ChunkCont::value_type));
            };
        }
    }
    const size_t element_size = sizeof(std::tuple<CHUNK_T<T>*, int, int>);

    global_stats.record(type_name<CHUNK_T<T>>(), somme_chunk_create, threshold, element_size);
}

} // namespace shrink_array
