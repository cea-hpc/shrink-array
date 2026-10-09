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

// Out-of-line bodies of the CHUNK_DYNAMIC re-chunk algorithm
// (RechunkSupport<Derived, true>). Included at the end of array_chunk_base.h,
// after the class, so the CRTP self() down-cast sees a complete Derived and so
// the heavy includes below (compression manager, stats) stay out of the
// container declaration itself.

#include <algorithm>   // std::min / std::max (CHUNK_RECHUNK_SLICES)
#include <atomic>      // std::atomic<long> n_folded (CHUNK_RECHUNK_TRACE)
#include <chrono>      // CHUNK_RECHUNK_TRACE phase timers
#include <ratio>       // std::milli (the same timers)
#include <cstddef>     // std::size_t
#include <cstdio>      // std::fprintf (CHUNK_RECHUNK_TRACE)
#include <cstring>     // std::memcpy / std::memcmp (refill bulk read)
#include <cstdlib>     // std::getenv (CHUNK_RECHUNK_TRACE, CHUNK_RECHUNK_SCHEDULE)
#include <optional>    // the affinity scope around tmp
#include <type_traits> // std::is_same_v, the v1 bool guard
#include <string_view> // CHUNK_RECHUNK_SCHEDULE parsing
#include <utility>     // std::swap (the array swap)
#include <vector>      // the refill buffers

#include "chunk/internal/in_t.h"                // in_t<T>: the element snapshots
#include "chunk/internal/array_chunk_base.h"    // ArrayChunkBase: self().set_chunk_size, invalidate_proxies
#include "chunk/internal/array_chunk_core.h"    // alloc_affinity_detail::RechunkStorageScope
#include "chunk/internal/array_chunk_rechunk.h" // RechunkSupport, the class these bodies belong to
#include "chunk/array_chunk_range_ops.h"        // copy_range -- the public bulk fill
#include "chunk/internal/chunk_stats.h"         // increm_chunk_rechunk* (CHUNK_STATS)
#include "chunk/internal/compression_manager.h" // CompressionManager<>::FlushReplayGuard in rechunk_refill
#include "chunk/PMP_CHUNK.h"                    // get_thr_rank_fast
#include "chunk/internal/env_parse.h"           // env_number / env_flag (checked, not atoi)
#include "chunk/internal/exact_rss.h"           // exact_rss_kib (rss_mib_now)
#include "chunk/vector_range_ops.h"             // assume_drained
#include "chunk/config.h"                       // build_config::use_openmp, chunk_config::use_chunk2a
#include "chunk/chunk_traits.h"                 // Proxy2A<T>
#include "chunk/internal/chunk_size_policy.h"   // set_thread_chunk_size / set_active_chunk_size
#include "chunk/bit.h"                          // is_bool_like_v (refill bulk read gate)
#include "chunk/internal/invariant_check.h"     // CHUNK_INVARIANT_FAILED (refill bulk read check)

namespace shrink_array::internal
{
/// CHUNK_RECHUNK_TRACE (read once, set = on): per-phase timings, memory and
/// occupancy of every re-chunk. Read here once rather than by a getenv() at
/// each of the sites below.
inline const bool g_rechunk_trace = std::getenv("CHUNK_RECHUNK_TRACE") != nullptr;

/// CHUNK_RECHUNK_RECYCLE (default on; 0 = off): at the v1 refill's slice
/// boundaries, hand the slabs the old geometry has just emptied to the value
/// arena for the next slice (TCSlabPool::recycle_empty_slabs), instead of
/// madvise'ing them. Off = the sweep of before, for comparison.
inline const bool g_rechunk_recycle = []() noexcept
{
    const char* env = std::getenv("CHUNK_RECHUNK_RECYCLE");
    return env == nullptr || env[0] == '\0' || env[0] != '0';
}();

/// CHUNK_RECHUNK_ADVISE_EACH_SLICE (default off; 1 = on): with slab recycling,
/// madvise the recycled slabs a slice did not take at EVERY slice boundary, as
/// before, rather than once at the end of the refill.
inline const bool g_rechunk_advise_each_slice = []() noexcept
{
    const char* env = std::getenv("CHUNK_RECHUNK_ADVISE_EACH_SLICE");
    return env != nullptr && env[0] != '\0' && env[0] != '0';
}();

/// CHUNK_RECHUNK_TRACE_MEM (read once, set = on): the MEMORY side of the trace --
/// [RECHUNK-RSS] (exact RSS, a page-table walk per read), the new geometry's
/// occupancy, the allocator census. Kept apart from CHUNK_RECHUNK_TRACE so the
/// phase times are not measured with these probes inside them (they cost
/// seconds per run at tens of GB and 128 threads).
inline const bool g_rechunk_trace_mem = std::getenv("CHUNK_RECHUNK_TRACE_MEM") != nullptr;

/// CHUNK_RECHUNK_TRACE: the v1 refill's inter-slice costs for the re-chunk in
/// progress (ms), reset by rechunk_to() and printed in its phases line.
/// Written by the master between the slice regions only.
inline double g_slice_drain_ms   = 0.0;
inline double g_slice_collect_ms = 0.0;
inline double g_slice_madvise_ms = 0.0;

/// CHUNK_RECHUNK_TRACE: where the v1 refill's slice regions spend their wall
/// time, summed over the slices of the re-chunk in progress (ms), reset by
/// rechunk_to() and printed on its "slices" line. Instants on steady_clock
/// (one system-wide counter), relative to the master just before the region:
///   fork      the LAST thread's entry into the region
///   loop      the last end of a thread's rebuild loop, minus fork
///   barrier   the master past the closing barrier, minus the last loop end
///   boundary  the slice-boundary drain + sweep (~ drain + collect + madvise)
///   join      the master out of the region, minus the end of the boundary
/// work max / avg: one thread's rebuild loop, max and mean over the team
/// (their gap is the imbalance); setup max: guard + buffer before the loop.
/// Written by the master outside the regions only.
inline int    g_slice_count        = 0;
inline double g_slice_fork_ms      = 0.0;
inline double g_slice_loop_ms      = 0.0;
inline double g_slice_barrier_ms   = 0.0;
inline double g_slice_boundary_ms  = 0.0;
inline double g_slice_join_ms      = 0.0;
inline double g_slice_work_max_ms  = 0.0;
inline double g_slice_work_avg_ms  = 0.0;
inline double g_slice_setup_max_ms = 0.0;

/// @brief Resident set size of this process in MiB, or -1 if unreadable.
/// @details EXACT (smaps_rollup, see exact_rss.h): statm lags by gigabytes at
///          64 threads. Only the CHUNK_RECHUNK_TRACE paths read it (per phase
///          of a re-chunk, per slice).
inline long rss_mib_now() noexcept
{
    const long kib = shrink_array::exact_rss_kib();
    return kib >= 0 ? kib / 1024 : -1;
}

/// @brief The rebuild loop's schedule: CHUNK_RECHUNK_SCHEDULE="kind[,chunk]"
///        (static|dynamic|guided|auto, chunk >= 1 or omitted), read once;
///        guided,16 when unset or malformed: a few % faster than dynamic,64 on
///        the 64-thread bench, where dynamic,256/1024 were 2-3x slower on
///        small re-chunks. CHUNK_RECHUNK_SCHEDULE=dynamic,64 restores the old
///        schedule.
///        Serial builds parse it too; chunk_pmp_set_schedule() is then a no-op.
struct RebuildSchedule
{
    chunk_pmp_schedule schedule{chunk_pmp_sched::sched_guided, 16};
    char               text[32]{"guided,16"};
};

/// @brief The rebuild loop's schedule, parsed once from CHUNK_RECHUNK_SCHEDULE ("kind[,chunk]").
/// @return The parsed schedule, or the default guided,16 when the variable is unset or not understood.
inline const RebuildSchedule& rebuild_schedule() noexcept
{
    static const RebuildSchedule schedule = []
    {
        RebuildSchedule parsed;
        const char*     env = std::getenv("CHUNK_RECHUNK_SCHEDULE");
        if (env == nullptr || env[0] == '\0')
        {
            return parsed;
        }
        char            kind[16] = {};
        int             chunk    = 0;
        const int       nfields  = std::sscanf(env, "%15[a-z],%d", kind, &chunk);
        chunk_pmp_sched k        = chunk_pmp_sched::sched_static;
        bool            ok       = nfields >= 1 && chunk >= 0;
        if (ok)
        {
            const std::string_view kv(kind);
            if (kv == "static")
            {
                k = chunk_pmp_sched::sched_static;
            }
            else if (kv == "dynamic")
            {
                k = chunk_pmp_sched::sched_dynamic;
            }
            else if (kv == "guided")
            {
                k = chunk_pmp_sched::sched_guided;
            }
            else if (kv == "auto")
            {
                k = chunk_pmp_sched::sched_auto;
            }
            else
            {
                ok = false;
            }
        }
        if (!ok)
        {
            std::fprintf(stderr,
                         "[RECHUNK] CHUNK_RECHUNK_SCHEDULE=\"%s\" not understood "
                         "(kind[,chunk], kind = static|dynamic|guided|auto): keeping guided,16\n",
                         env);
            return parsed;
        }
        parsed.schedule = {k, chunk}; // chunk 0: the runtime's default for this kind
        std::snprintf(parsed.text, sizeof parsed.text, "%s,%d", kind, chunk);
        return parsed;
    }();
    return schedule;
}
} // namespace shrink_array::internal

namespace shrink_array
{

template <typename Derived>
void RechunkSupport<Derived, true>::rechunk_to(unsigned int new_size)
{
    const unsigned int old_size = self().size_policy_.size();
    // Captured before the swap below hands `self` tmp's key: the allocator set
    // holding the OLD geometry's blocks is (old_key, old_size), and it is the
    // one whose slabs must empty for the re-chunk to give memory back.
    const unsigned int old_key = self().alloc_key();
    if (new_size == old_size || new_size == 0)
    {
        return;
    }

    // The index split is a shift and a mask: a size that is not a power of
    // two (48: shift 5, mask 47) sends two elements to one slot, and the
    // refill below -- which assumes one geometry tiles the other exactly --
    // frees a buffer still in use. Refused, with a message in EVERY build:
    // a caller bug, on a cold path.
    if (!is_valid_chunk_size(new_size))
    {
        std::fprintf(stderr, "[RECHUNK] rechunk_to(%u) refused: a chunk size must be a power of two\n", new_size);
        std::fflush(stderr);
        return;
    }

    // SAFETY: the shared constant buffers (cst_val1 / cst_val2) are sized
    // cst_buffer_size (the dynamic-mode ceiling). A chunk wider than that
    // would read those buffers out of bounds, so refuse the re-chunk.
    // Raise CHUNK_DYN_MAX_SIZE if a legitimate geometry needs to be larger.
    if (new_size > cst_buffer_size)
    {
        if constexpr (build_config::any_check)
        {
            std::fprintf(stderr,
                         "[RECHUNK] rechunk_to(%u) refused: exceeds cst_buffer_size=%u "
                         "(raise CHUNK_DYN_MAX_SIZE)\n",
                         new_size, shrink_array::cst_buffer_size);
            std::fflush(stderr);
        }
        return;
    }

    // Serial fast-alloc builds re-chunk too: the per-size allocator pool
    // (allocator_pool_serial_, populated by register_chunk_size_serial via
    // set_chunk_size) routes each allocation to the right element size, so the
    // old and new geometries coexist safely for the duration of the refill.

    const ind_type total = self().m_size;
    if (total <= 0)
    {
        self().set_chunk_size(new_size); // legal on an empty array
        return;
    }

    // A real geometry change is now committed: every no-op / refusal exit
    // (same size, size 0, invalid or too wide a size, empty array) is above
    // this point, and nothing below aborts. Count it for the run's stats.
    ++shrink_array::g_rechunk_count;

    // Drain THIS array's deferred WRITE ops at the OLD geometry, before
    // the t_active switch.
    // CHUNK_RECHUNK_TRACE=1: per-phase wall time, to see where a slow
    // re-chunk spends it (tmp creation, refill, flush+swap, tmp teardown).
    const bool trace_phases = internal::g_rechunk_trace;
    const auto t_now        = [] { return std::chrono::steady_clock::now(); };
    const auto t_ms         = [](auto start, auto stop)
    { return std::chrono::duration<double, std::milli>(stop - start).count(); };
    const auto t0                  = t_now();
    internal::g_slice_drain_ms     = 0.0; // filled by rechunk_refill's slice boundaries
    internal::g_slice_collect_ms   = 0.0;
    internal::g_slice_madvise_ms   = 0.0;
    internal::g_slice_count        = 0; // filled by rechunk_refill's slice regions
    internal::g_slice_fork_ms      = 0.0;
    internal::g_slice_loop_ms      = 0.0;
    internal::g_slice_barrier_ms   = 0.0;
    internal::g_slice_boundary_ms  = 0.0;
    internal::g_slice_join_ms      = 0.0;
    internal::g_slice_work_max_ms  = 0.0;
    internal::g_slice_work_avg_ms  = 0.0;
    internal::g_slice_setup_max_ms = 0.0;
    std::chrono::steady_clock::time_point t1_out;
    std::chrono::steady_clock::time_point t2_out;
    std::chrono::steady_clock::time_point t3_out;
    std::chrono::steady_clock::time_point t4_out;
    // CHUNK_RECHUNK_TRACE: process RSS at each phase boundary, next to the
    // wall times. The slice trace inside rechunk_refill brackets the refill;
    // this brackets everything around it, where the remaining part of the
    // peak was measured to sit.
    // Exact RSS only under CHUNK_RECHUNK_TRACE_MEM: each read walks the page
    // tables, and they sit inside the timed phases.
    const auto rss_now    = [&] { return internal::g_rechunk_trace_mem ? internal::rss_mib_now() : -1L; };
    const long r_entry    = rss_now();
    long       r_flush    = -1;
    long       r_pre_tmp  = -1;
    long       r_refill   = -1;
    long       r_swap     = -1;
    long       r_teardown = -1;

    self().flush_pending_deferred_ops();
    r_flush = rss_now();

    // Remove THIS array's pending compression CANDIDATES at the OLD
    // geometry before t_active changes. fill_list_to_compress scans at the
    // AMBIENT geometry (alloc_chunk_count()), NOT the chunk's own size; a
    // perform_compression during tmp's refill (t_active == new_size) would
    // otherwise scan these OLD-size chunks at the NEW size -> over-scan on
    // shrink (heap-overflow) / under-scan on grow (mis-compression).
    // Content-neutral: the snapshot below reads materialised values.
    self().rechunk_drop_candidates();

    // No dense O(n) snapshot: the refill below streams old -> new chunk by
    // chunk, so the compressed store is never materialised whole.

    // Variant state captured BEFORE the geometry switch (Chunk2's
    // array-level second_value; T{} / ignored for Chunk & Chunk2A).
    const in_t<T> state = self().rechunk_capture_state();

    // 2) build the fresh array at S'. Set the active geometry around its
    //    construction + refill only, then restore.
    // g_active is saved/restored (see the matching restore below, right
    // before tmp is destroyed) since only rechunk_to() ever touches it.
    // t_active is NOT saved here: it is a thread-local that every write
    // path self-corrects to its own array's chunk_size() right before
    // materialising, so a value captured here would already be stale and
    // arbitrary relative to old_size by the time it would be restored --
    // see the restore site for the full rationale.
    const unsigned int saved_g_active = active_chunk_size();

    set_active_chunk_size(new_size);
    set_thread_chunk_size(new_size);
    register_runtime_chunk_size(new_size); // primes S' pools

    // Drain the S' reuse free-list BEFORE anything allocates at S' (a
    // buffer of a different geometry parked here would overflow init).
    self().rechunk_drain_reuse_list(new_size);

    {
        t1_out    = t_now();
        r_pre_tmp = rss_now();
        // The new storage takes its own allocator set (lazy affinity: an array
        // gets one at its first re-chunk, see assign_alloc_key()).
        std::optional<alloc_affinity_detail::RechunkStorageScope> affinity_scope(std::in_place);
        t_birth_at_active_size = true; // exactly S', whatever the element type
        Derived tmp(total);            // born at S', own num_tab, primed pool
        t_birth_at_active_size = false;
        affinity_scope.reset();

        // Seed variant state into tmp BEFORE refill (Chunk2 second_value,
        // so its compression decisions use the same value).
        tmp.rechunk_seed_fresh(state);

        // 3) Refill tmp from *this (OLD -> NEW geometry): rechunk_refill()
        //    rebuilds chunk by chunk and skips the spans still at the first
        //    constant (only materialised chunks are written).
        // CHUNK_STATS: one geometry change, plus the per-chunk work it costs
        // (counted inside rebuild_one below).
        increm_chunk_rechunk<chunk_type>();
        t2_out = t_now();
        self().rechunk_refill(tmp, old_size, new_size);
        t3_out   = t_now();
        r_refill = rss_now();

        tmp.flush_pending_deferred_ops(); // materialise tmp's deferred writes

        // tmp's refill CANDIDATES are deliberately KEPT. Dropped, every rebuilt
        // chunk holding a constant stays materialised until teardown -- a
        // Chunk2/2A chunk only re-enters the candidate list on a write that
        // transitions it to a constant, which never happens to a chunk that
        // already holds one -- hundreds of MB to ~1 GB per re-chunk on the
        // bench.
        //
        // No geometry-mismatch hazard: every candidate tuple carries the
        // chunk's own geometry (field <3>, captured at enqueue) and all three
        // variants scan and compress at that size, not at the ambient
        // alloc_chunk_count().
        //
        // What makes keeping them correct across the swap below:
        //  - the tuples point into tmp.chunk_list's storage, which the
        //    std::swap hands to *this unchanged (buffer swap, no move);
        //  - they carry tmp's num_tab, and rechunk_swap_identity() makes that
        //    *this's identity (member swap for Chunk/Chunk2A, registry swap for
        //    Chunk2), so the sweep's reset_tab_inside_list_to_compress(num_tab,
        //    num_chunk) lands on the row allocated for tmp at new_size;
        //  - tmp's own destruction removes candidates by ITS post-swap num_tab
        //    (= *this's old identity, already emptied by rechunk_drop_candidates()
        //    above) and, for 2A, by the OLD chunk_list's pointer range --
        //    neither touches these.
        // The next end-of-phase sweep then scans them with the variant's own
        // compression logic and folds the constant ones back. Nothing is
        // synthesised here.

        // Re-drain THIS array's deferred ops before swapping storage out.
        // FORCED: the ops are keyed by the OLD chunk_list's addresses, which are
        // about to be destroyed with tmp. any_pending_ops() has been seen to
        // read 0 here while ops were still queued, so the unforced flush left
        // them behind to be replayed later against a recycled, wrong-size chunk
        // -- the coexistence heap-buffer-overflow. Flush every chunk while the
        // old geometry is still in place and the replay is correct.
        self().flush_pending_deferred_ops(/*force=*/true);

        // 4) SWAP the common storage. nb_tab is static and NOT swapped.
        //    num_tab + registry (+ any variant members) go through the
        //    identity hook: Chunk / Chunk2A swap the num_tab member, Chunk2
        //    resolves num_tab from the registry and must NOT swap it.
        using std::swap;
        swap(self().chunk_list, tmp.chunk_list);
        // The new storage was built under tmp's own affinity key (its slabs);
        // the key travels with the storage. self keeps tmp's key from here on,
        // tmp takes the old one and dies with it: its clear_core() drains that
        // key's lists and the old slabs empty.
        swap(self().alloc_key_, tmp.alloc_key_);
        self().swap_dirty_bitset(tmp);
        swap(self().m_size, tmp.m_size);
        swap(self().nb_chunk, tmp.nb_chunk);
        swap(self().size_policy_, tmp.size_policy_);
        self().invalidate_proxies();

        const bool ok = self().rechunk_swap_identity(tmp);
        if constexpr (build_config::any_check)
        {
            if (!ok)
            {
                std::fprintf(stderr, "[RECHUNK] identity swap failed (this or tmp "
                                     "not registered) -- rechunk aborted mid-swap\n");
                std::fflush(stderr);
            }
        }
        (void)ok;

        // Variant finalisation after the swap (Chunk2 re-owns its
        // map_inf_array entry so it does not dangle when tmp dies).
        self().rechunk_reown_after_swap(tmp, state);

        // Restore the active geometry BEFORE tmp is destroyed, so tmp (now
        // OLD geometry) frees its blocks into the OLD-size pool.
        //
        // t_active is restored to old_size, NOT to a value saved at entry:
        // every write path re-sets t_active to ITS OWN array's chunk_size()
        // right before materialising (see Proxy2::ensure_writable_storage()
        // and its siblings), so a value saved at entry may belong to an
        // unrelated array. Restoring it would route free_elt_pow2() below
        // (through get_chunk_allocator(), which keys on the ambient) to the
        // WRONG size pool. old_size is reliable: captured once, from this
        // array's own size_policy_. g_active is touched only by rechunk_to()
        // (set_active_chunk_size()), so saved_g_active is a trustworthy
        // baseline.
        set_active_chunk_size(saved_g_active);
        set_thread_chunk_size(old_size);

        t4_out = t_now();
        r_swap = rss_now();
        // tmp destroyed here: clear_core drops tmp's candidates by num_tab
        // and parks its OLD-geometry blocks by size. Without the end-of-life
        // census: tmp's death is not an exit (ArrayChunkBase::exit_census_).
        tmp.exit_census_ = false;
        // ... and its chunk_list buffer becomes the next re-chunk's (see
        // ArrayChunkCore::release_chunk_list), reset right after the brace.
        alloc_affinity_detail::g_releasing_rechunk_storage = true;
    }
    alloc_affinity_detail::g_releasing_rechunk_storage = false;
    const auto t5                                      = t_now(); // tmp destroyed
    r_teardown                                         = rss_now();

    // Free the OLD-geometry blocks tmp just parked into the old-size reuse
    // list. Nothing pops them again -- this array (and, in a global
    // re-chunk, every array) has moved off old_size -- so without this they
    // accumulate as pure RETENTION across successive re-chunks: memory
    // grows by ~one array's worth of buffers per geometry change. Returns them
    // to the OS under std-alloc (under fast alloc, the slab sweep below does);
    // also removes stale-geometry buffers so a later pop can never grab a
    // wrong-size block.
    self().rechunk_drain_reuse_list(old_size);
    const auto t5b = t_now(); // reuse-list drain done; the empty-slab sweep follows
    // The array has left (old_key, old_size) for good: once the drain above
    // has run its slabs there hold nothing, but manage_full_empty() only
    // inspects a slab at the instant a block is freed, so slabs that became
    // empty unobserved stay resident. Sweep them now -- key 0 too: the shared
    // per-type pool every array below CHUNK_ALLOC_AFFINITY_MIN_BYTES falls
    // back to, where most of that residue sat (GBs at 64 threads).
    //
    // Sweeping a SHARED pool means trusting nn_ref, which the hot path may
    // not do: manage_full_empty() fires on every free, concurrently, where a
    // transient zero is possible. This runs at the serial, quiescent end of a
    // re-chunk, with no concurrent free in flight, so an empty slab is empty.
    // Emptiness is the whole claim -- a key-0 slab mixes several arrays'
    // blocks and no ownership is assumed.
    if constexpr (build_config::use_fast_alloc && build_config::use_openmp)
    {
        const unsigned int saved_sz = thread_chunk_size();
        set_thread_chunk_size(old_size);
        // Every rank, both keys, into ONE sweep: the abandoned set's slabs were
        // carved back to back across ranks, so one madvise per contiguous run
        // instead of one per slab (each one a TLB shootdown on every CPU).
        typename std::remove_reference_t<decltype(chunk_type::get_chunk_allocator(0, 0u))>::empty_slab_sweep sweep;
        for (int r = 0; r < CHUNK_PMP_RANK_BOUND; ++r)
        {
            if (old_key != 0)
            {
                chunk_type::get_chunk_allocator(r, old_key).collect_empty_slabs(sweep);
            }
            chunk_type::get_chunk_allocator(r, 0u).collect_empty_slabs(sweep);
        }
        sweep.flush();
        set_thread_chunk_size(saved_sz);
    }
    if (internal::g_rechunk_trace_mem) // a census: memory side of the trace
    {
        // Occupancy of the allocator set the array just left. Its slabs hold
        // nothing but this array's old-geometry blocks, so once the drain has
        // run they should be empty and their pages given back. A non-zero
        // "live" here is the whole question: blocks of that geometry are still
        // out somewhere, which is why no slab can empty and why the re-chunk's
        // memory never comes back.
        // Reported for the array's OWN key and, separately, for key 0 -- the
        // shared per-type pool every array falls back to below the affinity
        // threshold. A key-0 slab mixes several arrays' blocks and cannot
        // empty on one array's re-chunk, by construction.
        const auto scan = [&](unsigned int key, std::size_t& slabs, std::size_t& live, std::size_t& empty)
        {
            if constexpr (build_config::use_fast_alloc && build_config::use_openmp)
            {
                const unsigned int saved = thread_chunk_size();
                set_thread_chunk_size(old_size);
                for (int r = 0; r < CHUNK_PMP_RANK_BOUND; ++r)
                {
                    const auto& allocator = chunk_type::get_chunk_allocator(r, key);
                    for (const auto* node : {allocator.block, allocator.block_full})
                    {
                        for (; node != nullptr; node = node->next)
                        {
                            ++slabs;
                            live += static_cast<std::size_t>(node->nn_ref);
                            if (node->nn_ref == 0)
                            {
                                ++empty;
                            }
                        }
                    }
                }
                set_thread_chunk_size(saved);
            }
        };
        std::size_t slabs       = 0;
        std::size_t live_blocks = 0;
        std::size_t empty_slabs = 0;
        scan(old_key, slabs, live_blocks, empty_slabs);
        std::size_t s0 = 0;
        std::size_t l0 = 0;
        std::size_t e0 = 0;
        if (old_key != 0)
        {
            scan(0u, s0, l0, e0);
        }
        const std::size_t blk = static_cast<std::size_t>(old_size) * sizeof(T);
        std::fprintf(stderr,
                     "[RECHUNK-POOL] S=%u key=%u after drain: slabs=%zu  live_blocks=%zu (%zu MB)  "
                     "empty_slabs=%zu | key=0 (shared pool): slabs=%zu  live_blocks=%zu (%zu MB)  "
                     "empty_slabs=%zu\n",
                     old_size, old_key, slabs, live_blocks, (live_blocks * blk) >> 20U, empty_slabs, s0, l0,
                     (l0 * blk) >> 20U, e0);
        std::fflush(stderr);
    }
    if (trace_phases)
    {
        const auto t6 = t_now();
        std::fprintf(stderr,
                     "[RECHUNK] %u -> %u phases (ms): prologue %.0f | tmp create %.0f | refill %.0f | "
                     "of which slice drain %.0f | of which slice collect %.0f | of which slice madvise %.0f | "
                     "flush+swap %.0f | tmp teardown %.0f | old-pool drain %.0f | slab sweep %.0f | total %.0f\n",
                     old_size, new_size, t_ms(t0, t1_out), t_ms(t1_out, t2_out), t_ms(t2_out, t3_out),
                     internal::g_slice_drain_ms, internal::g_slice_collect_ms, internal::g_slice_madvise_ms,
                     t_ms(t3_out, t4_out), t_ms(t4_out, t5), t_ms(t5, t5b), t_ms(t5b, t6), t_ms(t0, t6));
        if (internal::g_slice_count > 0)
        {
            std::fprintf(stderr,
                         "[RECHUNK] %u -> %u slices %d (ms): fork %.1f | loop %.1f | work max %.1f | "
                         "work avg %.1f | setup max %.1f | barrier %.1f | boundary %.1f | join %.1f\n",
                         old_size, new_size, internal::g_slice_count, internal::g_slice_fork_ms,
                         internal::g_slice_loop_ms, internal::g_slice_work_max_ms, internal::g_slice_work_avg_ms,
                         internal::g_slice_setup_max_ms, internal::g_slice_barrier_ms, internal::g_slice_boundary_ms,
                         internal::g_slice_join_ms);
        }
    }
    if (internal::g_rechunk_trace_mem)
    {
        const long r_end = rss_now();
        std::fprintf(stderr,
                     "[RECHUNK-RSS] %u -> %u (MB): entry %ld | after flush %ld | before tmp %ld | after refill %ld | "
                     "after swap %ld | after teardown %ld | after sweep %ld   (end-entry %+ld)\n",
                     old_size, new_size, r_entry, r_flush, r_pre_tmp, r_refill, r_swap, r_teardown, r_end,
                     r_end - r_entry);
    }
}

template <typename Derived>
void RechunkSupport<Derived, true>::rechunk_refill(Derived& tmp, int old_size, int new_size)
{
    // Level-2 chunk-level rebuild, for each of the three chunk variants: every
    // non-cst1 new chunk is ARMED by a real proxy write (the write path's
    // compression bookkeeping -- set_ptr_is_inside_chunk_list_to_compress
    // etc.) and the rest RAW-filled, which copy_range() does (bool, bit-packed,
    // element by element); cst1 spans are skipped. ~13x faster than the tiled
    // transcription (else branch), which remains for a build that selects no
    // chunk variant.
    // Materialisation is tracked with is_cst_val1() (values == the cst1
    // sentinel), which is universal and index-free -- unlike
    // written()/is_cst_val() on chunk2, which need the second_value index. While
    // it is filled a chunk only goes cst1 -> real buffer (never cst2, which is a
    // compression result), so "still cst1" == "not yet materialised" holds for
    // every variant.
    if constexpr (chunk_config::use_chunk || chunk_config::use_chunk2 || chunk_config::use_chunk2a)
    {
        const ind_type total = self().m_size;
        const int      n_new = tmp.nb_chunk;
        // Geometry is per-array (this.size_=old, tmp.size_=new), so operator[]
        // maps correctly without the global. Only alloc_chunk_count()
        // (= thread_chunk_size(), TLS) needs the size, set per thread below.
        // Seed the process-wide global ONCE (final value = new); NEVER set the
        // global inside the loop — it would race under parallelism.
        set_active_chunk_size(new_size);

        // Rebuild one new chunk c into its own buffer. Pure-read of *this (old)
        // + armed write of tmp (new). Independent across c: each new chunk is
        // written by exactly one caller, *this reads are const, and the write
        // path partitions allocator/compression state by thread rank — so this
        // is safe to run concurrently over disjoint c.
        std::atomic<long> n_folded{0};    // spans the rebuild tried to fold back
        std::atomic<long> n_installed{0}; // ... of which 2A installed directly (no buffer)
        std::atomic<long> n_refused{0};   // ... of which compress() declined: the span
                                          // merged constants that DIFFER, so no
                                          // single-constant form can hold it
        // Chunk2's cst2 predicate and its compress() both need the array's
        // second-value registry entry, and both run per chunk. The lookup goes
        // through the shared info registry, so it is hoisted here: once per
        // re-chunk instead of once per chunk. `ind_old2` selects the SOURCE
        // array's second value (the scan below reads *this), `inf_new2` the
        // fresh array's (the fold writes tmp) -- rechunk_seed_fresh() has
        // already re-declared it on tmp, above.
        // Deduced, not named: only ArrayChunk2 has inf_array_t, so spelling the
        // type here would instantiate it for every variant. The discarded
        // branch of an `if constexpr` inside the lambda is never instantiated,
        // hence the immediately-invoked form.
        // NESTED, not `use_chunk2 && Derived::is_not_bool`: `&&` short-circuits
        // EVALUATION, never NAME LOOKUP. Derived::is_not_bool is still looked up
        // when the condition is instantiated, so a v1 build -- where ArrayChunk
        // has no such member -- failed to compile the condition itself, before
        // ever reaching the discarded branch. Nesting is what makes the inner
        // test unreachable for a variant that cannot spell it.
        [[maybe_unused]] const int ind_old2 = [&]
        {
            if constexpr (chunk_config::use_chunk2)
            {
                if constexpr (Derived::is_not_bool)
                {
                    return self().get_ind_second_value();
                }
                else
                {
                    return 0;
                }
            }
            else
            {
                return 0;
            }
        }();
        [[maybe_unused]] const auto inf_new2 = [&]
        {
            if constexpr (chunk_config::use_chunk2)
            {
                if constexpr (Derived::is_not_bool)
                {
                    return tmp.get_info_second_value();
                }
                else
                {
                    return 0;
                }
            }
            else
            {
                return 0;
            }
        }();
        // Fining only (v1, OpenMP): how many new chunks read one old chunk,
        // and how many have finished with each -- see the release below.
        [[maybe_unused]] const int                     fine_ratio = old_size > new_size ? old_size / new_size : 1;
        [[maybe_unused]] std::vector<std::atomic<int>> old_done;
        if constexpr (chunk_config::use_chunk && build_config::use_openmp)
        {
            if (old_size > new_size)
            {
                old_done = std::vector<std::atomic<int>>(static_cast<std::size_t>(self().nb_chunk));
            }
        }
        // Per-thread fold counters: ++ on the shared atomics above, per chunk and
        // by every thread, kept one cache line of the caller's frame bouncing --
        // the line the lambda also reads n / new_size from at each call. Merged
        // once per thread and slice after the loop.
        struct RebuildTally
        {
            long folded    = 0;
            long refused   = 0;
            long installed = 0; // 2A: constant installed without a buffer
        };
        const auto rebuild_one = [&](int new_chunk, std::vector<T>& buf, RebuildTally& tally)
        {
            const ind_type base = static_cast<ind_type>(new_chunk) * static_cast<ind_type>(new_size);
            if (base >= total)
            {
                return;
            }
            const ind_type end = (base + new_size) < total ? (base + new_size) : total;
            const ind_type len = end - base;

            const int old_first = static_cast<int>(base / old_size);
            const int old_last  = static_cast<int>((end - 1) / old_size);
            bool      all_cst1  = true;
            // 2A: every source chunk is a compressed constant (cst1 or cst2), so
            // the new chunk, once filled, holds only constants and can be folded
            // back immediately (see below). Not tracked for the other variants:
            // Chunk2's is_cst_val2() needs the second-value index, v1 has cst1
            // only (covered by all_cst1).
            bool all_folded = true;
            // 2A: every source is cst2 at the SAME shared array (same value):
            // the new chunk is that constant, known without filling it.
            [[maybe_unused]] bool     any_cst1_src = false;
            [[maybe_unused]] bool     same_cst2    = true;
            [[maybe_unused]] const T* cst2_src     = nullptr;
            for (int oc = old_first; oc <= old_last; ++oc)
            {
                const auto& och = self().chunk_list[static_cast<std::size_t>(oc)];
                if constexpr (chunk_config::use_chunk2a)
                {
                    if (och.is_cst_val1())
                    {
                        any_cst1_src = true;
                    }
                }
                if (!och.is_cst_val1())
                {
                    all_cst1 = false;
                    if constexpr (chunk_config::use_chunk2a)
                    {
                        if (!och.is_cst_val2())
                        {
                            all_folded = false;
                        }
                        else if (cst2_src == nullptr)
                        {
                            cst2_src = och.values_load();
                        }
                        else if (och.values_load() != cst2_src)
                        {
                            same_cst2 = false;
                        }
                    }
                    else if constexpr (chunk_config::use_chunk2)
                    {
                        // Nested for the same reason as ind_old2 above.
                        if constexpr (Derived::is_not_bool)
                        {
                            // Same test as 2A, with the second-value index
                            // Chunk2 needs and 2A carries implicitly.
                            if (!och.is_cst_val2(ind_old2))
                            {
                                all_folded = false;
                            }
                        }
                        else
                        {
                            all_folded = false; // bool: no second value
                        }
                    }
                    else
                    {
                        all_folded = false;
                    }
                }
            }
            if (all_cst1)
            {
                // Nothing to rebuild: the whole span was still the compressible
                // constant, so the new chunk stays compressed (work avoided).
                increm_chunk_rechunk_skipped<chunk_type>();
                return; // fresh cst1, leave it
            }
            increm_chunk_rechunk_rebuilt<chunk_type>();

            // 2A: all sources are the same second constant -> install it.
            // Copying it into a fresh buffer, scanning it back and releasing
            // the buffer (the fold below) gives the same state, but the pages
            // those allocations touch stay with the process (hundreds of MB per
            // re-chunk on the 64-thread bench). This is what a type-2
            // compression commits (find_or_set on the chunk's second-value
            // slot, type table, values pointer), minus the buffer; tmp's chunk
            // is reachable by no other thread until the swap, so no CAS.
            if constexpr (chunk_config::use_chunk2a && !std::is_same_v<T, bool>)
            {
                if (all_folded && same_cst2 && !any_cst1_src && cst2_src != nullptr)
                {
                    set_thread_chunk_size(old_size);
                    // auto: SecondValueRef is not declared in every TU that
                    // instantiates this file (array_chunk2.cpp), and the name
                    // would be looked up even in a discarded branch.
                    const T    v   = static_cast<T>(std::as_const(self())[base]);
                    auto&      nch = tmp.chunk_list[static_cast<std::size_t>(new_chunk)];
                    const auto psv = Proxy2A<T>::BaseSpec::get_tab_second_value(tmp.num_tab, new_chunk);
                    char*      ptc = &Proxy2A<T>::tab_type_cst_val[tmp.num_tab][new_chunk];
                    (void)nch.find_or_set_ind_cst_val2(v, psv);
                    chunk_type::set_cst_val_to_cst_val2(ptc);
                    nch.init_values_cst_val(ptc, psv);
                    ++tally.folded;
                    ++tally.installed;
                    return;
                }
            }

            // read *this at OLD; TLS is per-thread
            set_thread_chunk_size(old_size);
            // Bulk read, one values_load() per SOURCE chunk instead of the full
            // read protocol per element (const operator[]: index split,
            // sync_chunk, a ReadSection with its light_fence -- a full seq_cst
            // fence without asymmetric Dekker --, acquire load, 2A range test;
            // none of it vectorises).
            //
            // Sound because the source is QUIESCENT for the whole refill:
            // rechunk_to() flushed its deferred ops (nothing for sync_chunk to
            // pull) and dropped its compression candidates
            // (rechunk_drop_candidates), and nobody writes *this meanwhile, so
            // no pass can claim, compress or release one of its chunks -- what
            // the ReadSection guards against. The only release is v1's
            // release_st below, coordinated by old_done: a source chunk is
            // freed once every span reading it is done.
            //
            // Per source chunk, the same decision element_at_unguarded() makes
            // per element, on the same single load: a 2A cst2 pointer names an
            // identifier region that was never written -> fill with the
            // constant; anything else (materialised buffer, cst1 sentinel,
            // Chunk2's filled cst2 array) is readable as is -> memcpy.
            // bool / Bit keep the per-element path (bit-packed storage).
            if constexpr (std::is_trivially_copyable_v<T> && !is_bool_like_v<T>)
            {
                std::size_t k = 0;
                for (int oc = old_first; oc <= old_last; ++oc)
                {
                    const auto&    och     = self().chunk_list[static_cast<std::size_t>(oc)];
                    const ind_type oc_base = static_cast<ind_type>(oc) * static_cast<ind_type>(old_size);
                    const ind_type lo      = base > oc_base ? base : oc_base;
                    const ind_type oc_end  = oc_base + static_cast<ind_type>(old_size);
                    const ind_type hi      = end < oc_end ? end : oc_end;
                    const auto     n       = static_cast<std::size_t>(hi - lo);
                    const auto     off     = static_cast<std::size_t>(lo - oc_base);
                    const T* const vals    = och.values_load();
                    if constexpr (chunk_config::use_chunk2a)
                    {
                        if (och.is_cst_val2_ptr_of(vals))
                        {
                            std::fill_n(buf.begin() + static_cast<std::ptrdiff_t>(k), n,
                                        och.second_value_from_ptr(vals));
                            k += n;
                            continue;
                        }
                    }
                    // raw-read: refill bulk read -- quiescent source, see above
                    std::memcpy(buf.data() + k, vals + off, n * sizeof(T));
                    k += n;
                }
                // Check builds: the bulk read must agree with the per-element
                // path. Sampled (first, last, every 64th): this is a check, not
                // a second copy. Bytewise, so only where bytes ARE the value:
                // scalars (floats included -- both paths copy the same bits,
                // NaN and -0.0 alike) and classes without padding.
                if constexpr (build_config::any_check &&
                              (!std::is_class_v<T> || std::has_unique_object_representations_v<T>))
                {
                    if (k != static_cast<std::size_t>(len)) [[unlikely]]
                    {
                        CHUNK_INVARIANT_FAILED("rechunk refill bulk read: new chunk {} copied {} elements, span is {}",
                                               new_chunk, k, static_cast<std::size_t>(len));
                    }
                    const auto check_at = [&](ind_type j)
                    {
                        const T ref = static_cast<T>(std::as_const(self())[base + j]);
                        if (std::memcmp(&ref, &buf[static_cast<std::size_t>(j)], sizeof(T)) != 0) [[unlikely]]
                        {
                            CHUNK_INVARIANT_FAILED(
                                "rechunk refill bulk read: new chunk {} element {} (global {}) differs "
                                "from the per-element read ({} -> {})",
                                new_chunk, static_cast<long long>(j), static_cast<long long>(base + j), old_size,
                                new_size);
                        }
                    };
                    for (ind_type j = 0; j < len; j += 64)
                    {
                        check_at(j);
                    }
                    check_at(len - 1);
                }
            }
            else
            {
                for (ind_type k = 0; k < len; ++k)
                {
                    buf[static_cast<std::size_t>(k)] = static_cast<T>(std::as_const(self())[base + k]);
                }
            }

            // Release the source chunks this span has just consumed, NOW rather
            // than at tmp's destruction.
            //
            // rechunk_to() builds tmp at S' while *this still holds every chunk
            // at S, so the two coexist until the swap and the peak is close to
            // the sum -- which is what makes @dyn cost roughly twice its static
            // counterpart in the bench, even though the steady state barely
            // moves.
            //
            // A source chunk is done for good once every element of it has been
            // transcribed. Sizes are powers of two, so the boundaries align
            // exactly. When coarsening, this span covers every chunk of
            // [old_first, old_last] whole, and releases them. When fining,
            // old_size / new_size new chunks read the same old one, on
            // whichever threads the schedule gives them: the LAST to finish
            // releases it, counted per old chunk (old_done). Never "the last
            // new chunk BY INDEX": a schedule block boundary can fall between
            // c and c+1, and the source is then freed while the thread holding
            // the previous index still reads it.
            //
            // Released while the TLS is still old_size, deliberately:
            // free_block_to_pool() routes by the AMBIENT geometry, so freeing a
            // S-sized buffer under S' would park it in the wrong pool.
            // v1 only: Chunk2/Chunk2A's clear_st() take the variant's compression
            // state (num_compress_value, ptr_type_cst, ptr_second_value), which
            // this loop has no business synthesising -- releasing their buffers
            // early would need the matching teardown of that state. Chunk<T>'s
            // clear_st(thr, n_elems) frees the buffer and marks the chunk cst1,
            // which is exactly what "already transcribed" means here.
            if constexpr (chunk_config::use_chunk && build_config::use_openmp)
            {
                const int thr = get_thr_rank_fast();
                for (int oc = old_first; oc <= old_last; ++oc)
                {
                    if (old_size > new_size)
                    {
                        // Fining: old chunk oc is read by fine_ratio new chunks;
                        // release it with the last of them to finish, whoever
                        // that is.
                        if (old_done[static_cast<std::size_t>(oc)].fetch_add(1, std::memory_order_acq_rel) + 1 !=
                            fine_ratio)
                        {
                            continue;
                        }
                    }
                    else
                    {
                        const ind_type oc_end = static_cast<ind_type>(oc + 1) * static_cast<ind_type>(old_size);
                        if (oc_end > end)
                        {
                            continue; // still needed by the next new chunk
                        }
                    }
                    auto& och = self().chunk_list[static_cast<std::size_t>(oc)];
                    // release_st, not clear_st: a release once copied, not a
                    // compression (CHUNK_STATS counted every rebuilt chunk as one).
                    if (!och.is_cst_val1())
                    {
                        och.release_st(thr, old_size);
                    }
                }
            }

            // Fill tmp's new chunk c through the public bulk API. copy_range()
            // is the fast path (the per-element proxy path costs ~13x more): it
            // takes one decision per chunk, arms with real proxy writes, stores
            // the rest straight into the buffer, and restores the zero counter
            // the bulk store bypasses -- with the same variant split, bool
            // (bit-packed) going per element.
            //
            // assume_drained: rechunk_to() has already flushed this array's
            // deferred ops, and this lambda runs inside the parallel loop below,
            // where a per-chunk whole-array drain would be both wrong and
            // pointless.
            set_thread_chunk_size(new_size);
            copy_range(tmp, static_cast<std::size_t>(base), static_cast<std::size_t>(end), buf.begin(), assume_drained);

            // Fold back NOW what was already compressed.
            //
            // copy_range materialises the new chunk whatever its content. When
            // every source chunk was a compressed constant (all_folded), the
            // buffer just written holds nothing but constants, and leaving it
            // for the next end-of-phase sweep keeps the WHOLE array's cst2 share
            // materialised until then (it doubled the process RSS on the full
            // bench).
            //
            // So run the variant's own compress() on the chunk right here, on
            // this thread's rank, while the TLS is still new_size (clear_int_st
            // frees the buffer into the pool routed by the ambient geometry).
            // It is exactly what the sweep would do: scan, then clear_int_st
            // with the variant's state -- nothing synthesised. The transient
            // is bounded to one chunk per thread. If the constants differ
            // across the span the scan refuses and the chunk stays materialised,
            // which is then the truth. The candidate the arming write queued
            // remains in the list; the sweep finds a cst1/cst2 chunk and skips
            // it (compress() guards on is_not_cst_val1(); the scan on a cst2
            // chunk reads the constant and re-derives the same state).
            //
            // Wired for 2A and Chunk2. The two compress() signatures differ --
            // 2A locates the chunk by (num_chunk, num_tab), Chunk2 takes the
            // array's inf_array record -- but the fold is the same: re-scan and
            // clear_int_st with the variant's own state, nothing synthesised.
            // bool registers no second value (get_ind_second_value aborts) and
            // is bit-packed; it never takes this path.
            if constexpr (chunk_config::use_chunk2a)
            {
                if (all_folded)
                {
                    ++tally.folded;
                    auto& nch = tmp.chunk_list[static_cast<std::size_t>(new_chunk)];
                    nch.compress(get_thr_rank_fast(), new_chunk, tmp.num_tab,
                                 /*apply=*/true, ChunkElems{static_cast<unsigned>(new_size)});
                    // Every source chunk WAS a constant here, so a refusal means
                    // the span merged constants that differ -- the genuine cost
                    // of coarsening, the one the forecast has to carry.
                    if (!nch.is_cst_val1() && !nch.is_cst_val2())
                    {
                        ++tally.refused;
                    }
                }
            }
            // No v1 branch. v1 has one compressible constant, zero, and the
            // spans whose sources were all cst1 already took the early return
            // above; scanning every other rebuilt chunk for zeros folded almost
            // none of them (~0.02%) for a full buffer scan each. The
            // end-of-phase sweep does that work later, on the chunks that have
            // actually become zero. Chunk<T>::compress() keeps the guard that a
            // second visit to a cst1 chunk must not park the zero sentinel --
            // load-bearing for the sweep regardless of this loop.
            else if constexpr (chunk_config::use_chunk2)
            {
                // Nested for the same reason as ind_old2 above; bool has no
                // second value to fold to, so it simply does nothing here.
                //
                // AND gated on use_fast_block_compression: this four-argument
                // compress() lives in Chunk2FastCompPolicy<T, Derived, Enabled>,
                // which Chunk2 inherits as
                // Chunk2FastCompPolicy<..., chunk_config::use_fast_block_compression>.
                // That constant IS use_openmp: in a SERIAL build the overload
                // does not exist (and there is no deferred compression to fold).
                if constexpr (chunk_config::use_fast_block_compression && Derived::is_not_bool)
                {
                    if (all_folded)
                    {
                        ++tally.folded;
                        auto& nch = tmp.chunk_list[static_cast<std::size_t>(new_chunk)];
                        nch.compress(get_thr_rank_fast(), inf_new2,
                                     /*apply=*/true, static_cast<unsigned int>(new_size));
                        if (!nch.is_cst_val1() && !nch.is_cst_val2(inf_new2.ind))
                        {
                            ++tally.refused;
                        }
                    }
                }
            }
        };

        // Parallel rebuild: each thread takes disjoint new chunks and drives the
        // write path on its own rank; the schedule (runtime, see
        // internal::rebuild_schedule()) balances cst1-skip (cheap) vs
        // materialise (costly). Geometry is per-array so no global race; only
        // per-thread TLS + per-rank allocator/compression state are touched.
        //
        // The OMPT parallel-end finalize is suppressed (g_in_rechunk_rebuild,
        // below): this region's end must NOT flush/compress the half-written
        // array (it corrupts grow regions that span several old chunks).
        // rechunk_to flushes at a quiescent point after the refill.
        //
        // SLICED rebuild. Inside one region v1 releases each source chunk as
        // soon as it is transcribed (release_st in rebuild_one), but a block
        // freed by a rank that does not own its slab is only PARKED in that
        // rank's reuse list -- walking the owner's slab chains from inside a
        // parallel region is what the owner-routed drain exists to avoid. So no
        // slab empties and no page comes back until the drain that follows the
        // WHOLE rebuild, while the new geometry is materialised in full:
        // peak = old + new.
        //
        // Draining at each slice boundary -- a serial point, the region has
        // closed -- routes that slice's parked blocks home, empties the old
        // slabs of the range just consumed and gives their pages back BEFORE
        // the next slice materialises, bounding the peak to ~old + new/slices.
        // rebuild_one folds each new chunk back to a constant when it can, so
        // a drained slice leaves little behind.
        //
        // CHUNK_RECHUNK_SLICES sets the count (default 16; 1 = one pass). v1 +
        // fast alloc + OpenMP only: Chunk2/Chunk2A do not release their source
        // chunks during the copy (their clear_st carries compression state this
        // loop has no business synthesising), so they have nothing to give back
        // between slices and the extra regions would be pure cost.
        int n_slices = 1;
        if constexpr (chunk_config::use_chunk && build_config::use_fast_alloc && build_config::use_openmp)
        {
            // 16: at 64 threads it cuts the peak above the pre-re-chunk level
            // from ~6.9 GB (1 slice) to ~1.7 GB; 32 gains little more. What
            // remains (~750 MB) is the two geometries coexisting at slab
            // granularity, insensitive to slicing by construction. Rebuilding
            // in slab order was measured and ruled out: index order is already
            // slab-local to within a factor of 1.4-2.
            n_slices = 16;
            n_slices = std::max(1, shrink_array::env_number<int>("CHUNK_RECHUNK_SLICES", n_slices));
            // Not worth slicing an array too small to empty a slab per slice.
            n_slices = std::min(n_slices, std::max(1, n_new / 4096));
        }
        // The affinity key of the set being emptied. `tmp` carries the new one;
        // rechunk_to() swaps them only after this call returns, so self()'s key
        // is still the old geometry's here.
        const unsigned int old_key_refill = self().alloc_key();

        // Slice boundaries run INSIDE the slice's region (see the loop): the
        // team drain's scratch lives across them, and so do the trace timers.
        using manager_t = Traits::manager_type;
        typename manager_t::TeamDrainScratch team_drain;
        // Slab recycling at the slice boundaries (internal::g_rechunk_recycle):
        // the ranks the team holds (each recycles its own slabs, one thread
        // the others'), and an address buffer per rank, reused. Sized at the
        // first boundary.
        std::vector<unsigned char>      recycle_held;
        std::vector<std::vector<void*>> recycle_scratch;
        using slice_clock   = std::chrono::steady_clock;
        const auto slice_ms = [](slice_clock::time_point a, slice_clock::time_point b)
        { return std::chrono::duration<double, std::milli>(b - a).count(); };
        slice_clock::time_point slice_t0{}; // written by the master, read after barriers
        slice_clock::time_point slice_t1{}; // written in a single, read after its barrier

        // Held across the WHOLE sliced loop, drains included. The flag also
        // suppresses the OMPT parallel-end finalize, which must not flush or
        // compress a half-rebuilt array -- and a slice boundary is precisely
        // "half rebuilt". The drain is serial under OMPT, so it opens no region
        // of its own and the flag costs it nothing.
        // The rebuild loop is schedule(runtime): set the chosen schedule on this
        // task (the slice regions inherit it) and restore the caller's after
        // the slices. See internal::rebuild_schedule(). Both calls are
        // no-ops in a serial build (PMP_CHUNK.h).
        const chunk_pmp_schedule caller_schedule = chunk_pmp_get_schedule();
        if constexpr (build_config::use_openmp)
        {
            const auto& rs = internal::rebuild_schedule();
            chunk_pmp_set_schedule(rs.schedule);
            if (internal::g_rechunk_trace)
            {
                std::fprintf(stderr, "[RECHUNK] rebuild schedule: %s\n", rs.text);
            }
        }
        g_in_rechunk_rebuild.store(true, std::memory_order_release);
        // Its own knob: the per-slice lines would bury the summary that
        // CHUNK_RECHUNK_TRACE is asked for. The only view of the peak's SHAPE:
        // a saw-tooth means the drains return memory and the coexistence is
        // inside a slice, a monotonic climb that they return nothing.
        const bool slice_trace = std::getenv("CHUNK_RECHUNK_TRACE_SLICES") != nullptr;
        const long rss_start   = slice_trace ? internal::rss_mib_now() : -1;
        if (slice_trace)
        {
            std::fprintf(stderr, "[RECHUNK-SLICE] 0/%d start       rss=%ld MB\n", n_slices, rss_start);
        }
        // CHUNK_RECHUNK_TRACE: the slice regions' wall time, split (see
        // internal::g_slice_fork_ms). Shared by the team; ns since sp_a.
        const bool sp_on = internal::g_rechunk_trace;
        struct SliceProbe
        {
            std::atomic<long long> entry_max{0};
            std::atomic<long long> loop_end_max{0};
            std::atomic<long long> work_max{0};
            std::atomic<long long> work_sum{0};
            std::atomic<long long> setup_max{0};
            std::atomic<int>       n{0};
        } sp;
        const auto sp_ns = [](slice_clock::time_point a, slice_clock::time_point b) noexcept -> long long
        { return std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count(); };
        const auto sp_max = [](std::atomic<long long>& a, long long v) noexcept
        {
            long long cur = a.load(std::memory_order_relaxed);
            while (v > cur && !a.compare_exchange_weak(cur, v, std::memory_order_relaxed))
            {
            }
        };
        slice_clock::time_point sp_a{};
        slice_clock::time_point sp_b{}; // master, past the closing barrier
        slice_clock::time_point sp_d{}; // master, past the boundary
        if (sp_on)
        {
            internal::g_slice_count = n_slices;
        }
        for (int sl = 0; sl < n_slices; ++sl)
        {
            const int c0 = static_cast<int>((static_cast<long>(n_new) * sl) / n_slices);
            const int c1 = static_cast<int>((static_cast<long>(n_new) * (sl + 1)) / n_slices);
            if (sp_on)
            {
                sp.entry_max.store(0, std::memory_order_relaxed);
                sp.loop_end_max.store(0, std::memory_order_relaxed);
                sp.work_max.store(0, std::memory_order_relaxed);
                sp.work_sum.store(0, std::memory_order_relaxed);
                sp.setup_max.store(0, std::memory_order_relaxed);
                sp.n.store(0, std::memory_order_relaxed);
                sp_a = slice_clock::now();
            }
#pragma omp parallel
            {
                const slice_clock::time_point sp_e = sp_on ? slice_clock::now() : slice_clock::time_point{};
                // Apply the rebuild's arming writes DIRECTLY, never via the
                // deferred path: a chunk reads cst1 until it is materialised,
                // so deferred arming writes would fill tmp's op map, and a
                // rehash frees bucket storage an enqueued op still targets
                // (heap-use-after-free at the flush). Under in_flush_replay each
                // write goes straight through write_body (materialise + store,
                // no enqueue): the first arming write materialises, the raw
                // fill does the rest. Safe here -- each thread rebuilds
                // DISJOINT new chunks, so the bypassed gate guards nothing.
                // Per-thread (thread_local), so it lives inside the region.
                const typename CompressionManager<T, chunk_type>::FlushReplayGuard rechunk_replay_guard;
                std::vector<T> buf(static_cast<std::size_t>(new_size));
                RebuildTally   tally; // this thread's fold counts for this slice
                                      // schedule(runtime): CHUNK_RECHUNK_SCHEDULE, default guided,16
                                      // (internal::rebuild_schedule).
                const slice_clock::time_point sp_w0 = sp_on ? slice_clock::now() : slice_clock::time_point{};
                // nowait + the explicit barrier below: the same closing barrier,
                // with each thread's loop end observable (CHUNK_RECHUNK_TRACE).
#pragma omp for schedule(runtime) nowait
                for (int c = c0; c < c1; ++c)
                {
                    rebuild_one(c, buf, tally);
                }
                if (sp_on)
                {
                    const auto sp_w1 = slice_clock::now();
                    sp_max(sp.entry_max, sp_ns(sp_a, sp_e));
                    sp_max(sp.loop_end_max, sp_ns(sp_a, sp_w1));
                    sp_max(sp.work_max, sp_ns(sp_w0, sp_w1));
                    sp_max(sp.setup_max, sp_ns(sp_e, sp_w0));
                    sp.work_sum.fetch_add(sp_ns(sp_w0, sp_w1), std::memory_order_relaxed);
                    sp.n.fetch_add(1, std::memory_order_relaxed);
                }
                // One RMW per thread and slice, not one per folded chunk; and
                // none at all unless the [RECHUNK] census, their only reader,
                // is on.
                if (internal::g_rechunk_trace_mem)
                {
                    n_folded.fetch_add(tally.folded, std::memory_order_relaxed);
                    n_refused.fetch_add(tally.refused, std::memory_order_relaxed);
                    n_installed.fetch_add(tally.installed, std::memory_order_relaxed);
                }
#pragma omp barrier
#pragma omp masked
                {
                    if (sp_on)
                    {
                        sp_b = slice_clock::now();
                    }
                }

                // Slice boundary, INSIDE the region (the omp for's barrier has
                // closed the slice): send this slice's parked old-geometry
                // blocks home, then give back the slabs they just emptied. Only
                // blocks already released by release_st are touched -- the
                // source chunks the remaining slices still have to read are
                // untouched. The whole team drains (each rank its own list and
                // its own slabs) and one thread sweeps: on the master alone,
                // between two regions, a boundary costs ~5 ms at 64 threads on
                // Grace, 15 boundaries per re-chunk. n_slices > 1 only in these
                // builds (see above).
                if constexpr (build_config::use_fast_alloc && build_config::use_openmp)
                {
                    if (sl + 1 < n_slices)
                    {
#pragma omp masked
                        {
                            if (internal::g_rechunk_trace)
                            {
                                slice_t0 = slice_clock::now();
                            }
                        }
                        manager_t::drain_reuse_list_for_size_team(static_cast<unsigned int>(old_size), team_drain);
                        if (internal::g_rechunk_recycle)
                        {
                            // The slabs this drain emptied go to the value
                            // arena for the next slice, which allocates about
                            // as much at the new size (same slab size): their
                            // resident pages are reused, where a madvise gave
                            // them to the kernel and the next slice faulted
                            // fresh zero pages back (madvise alone: ~2 s of a
                            // 128-thread run on Grace). Each rank recycles its
                            // own slabs -- the drain's owner discipline -- and
                            // the arena hands a rank its own back first (NUMA).
                            // The set and the safety argument are the sweep's
                            // (the else branch): the old storage's set, a
                            // quiescent point, a slab found empty is empty.
                            // The trace books the recycling as "collect" and
                            // the leftovers' madvise as "madvise".
#pragma omp single
                            {
                                const auto sc0 =
                                    internal::g_rechunk_trace ? slice_clock::now() : slice_clock::time_point{};
                                if (internal::g_rechunk_trace)
                                {
                                    internal::g_slice_drain_ms += slice_ms(slice_t0, sc0);
                                }
                                const auto n_ranks = static_cast<std::size_t>(CHUNK_PMP_RANK_BOUND);
                                if (recycle_held.size() != n_ranks)
                                {
                                    recycle_held.assign(n_ranks, 0);
                                    recycle_scratch.resize(n_ranks);
                                }
                                std::fill(recycle_held.begin(), recycle_held.end(), static_cast<unsigned char>(0));
                                // The previous boundary's slabs this slice did
                                // not take: back to the kernel now only under
                                // CHUNK_RECHUNK_ADVISE_EACH_SLICE=1. By default
                                // they wait for the end of the refill (one
                                // pass, below the slice loop): what they hold
                                // is memory the OLD geometry released, so the
                                // RSS stays at its entry level plus whatever
                                // the new geometry needs beyond the old one --
                                // the same bound as when they are advised
                                // between slices, when they are refaulted. And
                                // the 15 serial madvise passes of a re-chunk
                                // (one TLB shootdown on every CPU per run)
                                // become one.
                                if (internal::g_rechunk_advise_each_slice)
                                {
                                    set_thread_chunk_size(static_cast<unsigned int>(old_size));
                                    chunk_type::get_chunk_allocator(0, old_key_refill).advise_recycled_slabs();
                                    set_thread_chunk_size(static_cast<unsigned int>(new_size));
                                }
                                if (internal::g_rechunk_trace)
                                {
                                    slice_t1 = slice_clock::now();
                                    internal::g_slice_madvise_ms += slice_ms(sc0, slice_t1);
                                }
                            } // implicit barrier: recycle_held is clear
                            {
                                // Each thread unlinks its own rank's empty
                                // slabs, in parallel and without the arena's
                                // lock; the single below hands them over. Every
                                // thread calling recycle_batch() itself queued
                                // the whole team on that one mutex.
                                const int me = CHUNK_PMP_THREAD_RANK;
                                if (me >= 0 && static_cast<std::size_t>(me) < recycle_held.size())
                                {
                                    recycle_held[static_cast<std::size_t>(me)] = 1;
                                    const unsigned int saved_sz =
                                        thread_chunk_size(); // this thread's, as the slice left it
                                    set_thread_chunk_size(static_cast<unsigned int>(old_size));
                                    chunk_type::get_chunk_allocator(me, old_key_refill)
                                        .detach_empty_slabs(recycle_scratch[static_cast<std::size_t>(me)]);
                                    set_thread_chunk_size(saved_sz);
                                }
                            }
#pragma omp barrier
#pragma omp single
                            {
                                set_thread_chunk_size(static_cast<unsigned int>(old_size));
                                for (std::size_t r = 0; r < recycle_held.size(); ++r)
                                {
                                    auto& allocator =
                                        chunk_type::get_chunk_allocator(static_cast<int>(r), old_key_refill);
                                    if (recycle_held[r] == 0)
                                    {
                                        // A rank no team member holds (a
                                        // smaller team than the rank bound).
                                        allocator.detach_empty_slabs(recycle_scratch[r]);
                                    }
                                    // One uncontended lock per rank, each
                                    // rank's slabs under its own rank (NUMA).
                                    allocator.recycle_detached(recycle_scratch[r]);
                                }
                                set_thread_chunk_size(static_cast<unsigned int>(new_size));
                                if (internal::g_rechunk_trace)
                                {
                                    internal::g_slice_collect_ms += slice_ms(slice_t1, slice_clock::now());
                                }
                            } // implicit barrier: the next slice allocates from the recycled slabs
                        }
                        else
                        {
#pragma omp single
                            {
                                const auto sc0 =
                                    internal::g_rechunk_trace ? slice_clock::now() : slice_clock::time_point{};
                                if (internal::g_rechunk_trace)
                                {
                                    internal::g_slice_drain_ms += slice_ms(slice_t0, sc0);
                                }
                                set_thread_chunk_size(static_cast<unsigned int>(old_size));
                                // The old storage's allocator set -- the shared
                                // pool (key 0) too when it lives there, which is
                                // every first re-chunk under lazy affinity:
                                // skipping it held every released block's slab
                                // (GBs) until the final sweep. Safe for the
                                // final sweep's reason: a serial, quiescent
                                // point, so a slab found empty is empty.
                                // One sweep across ranks: one madvise per
                                // contiguous run.
                                typename std::remove_reference_t<decltype(chunk_type::get_chunk_allocator(
                                    0, 0u))>::empty_slab_sweep sweep;
                                for (int r = 0; r < CHUNK_PMP_RANK_BOUND; ++r)
                                {
                                    chunk_type::get_chunk_allocator(r, old_key_refill).collect_empty_slabs(sweep);
                                }
                                const auto sc1 =
                                    internal::g_rechunk_trace ? slice_clock::now() : slice_clock::time_point{};
                                sweep.flush();
                                if (internal::g_rechunk_trace)
                                {
                                    internal::g_slice_collect_ms += slice_ms(sc0, sc1);
                                    internal::g_slice_madvise_ms += slice_ms(sc1, slice_clock::now());
                                }
                                set_thread_chunk_size(static_cast<unsigned int>(new_size));
                            } // implicit barrier: the next slice starts on emptied slabs
                        }
                    }
                }
#pragma omp masked
                {
                    if (sp_on)
                    {
                        sp_d = slice_clock::now();
                    }
                }
            }
            if (sp_on)
            {
                const auto   sp_post = slice_clock::now();
                const double e_ms    = static_cast<double>(sp.entry_max.load(std::memory_order_relaxed)) * 1e-6;
                const double l_ms    = static_cast<double>(sp.loop_end_max.load(std::memory_order_relaxed)) * 1e-6;
                const int    n_thr   = std::max(1, sp.n.load(std::memory_order_relaxed));
                internal::g_slice_fork_ms += e_ms;
                internal::g_slice_loop_ms += l_ms - e_ms;
                internal::g_slice_barrier_ms += slice_ms(sp_a, sp_b) - l_ms;
                internal::g_slice_boundary_ms += slice_ms(sp_b, sp_d);
                internal::g_slice_join_ms += slice_ms(sp_d, sp_post);
                internal::g_slice_work_max_ms +=
                    static_cast<double>(sp.work_max.load(std::memory_order_relaxed)) * 1e-6;
                internal::g_slice_work_avg_ms +=
                    static_cast<double>(sp.work_sum.load(std::memory_order_relaxed)) * 1e-6 / n_thr;
                internal::g_slice_setup_max_ms +=
                    static_cast<double>(sp.setup_max.load(std::memory_order_relaxed)) * 1e-6;
            }
            if (slice_trace && sl + 1 < n_slices)
            {
                const long rss_mib = internal::rss_mib_now();
                std::fprintf(stderr, "[RECHUNK-SLICE] %d/%d after-drain rss=%ld MB  (%+ld since start)\n", sl + 1,
                             n_slices, rss_mib, rss_mib - rss_start);
            }
        }
        if (slice_trace)
        {
            const long rss_mib = internal::rss_mib_now();
            std::fprintf(stderr, "[RECHUNK-SLICE] %d/%d end         rss=%ld MB  (%+ld since start)\n", n_slices,
                         n_slices, rss_mib, rss_mib - rss_start);
        }
        if constexpr (build_config::use_fast_alloc && build_config::use_openmp)
        {
            // The last boundary's recycled slabs the last slice did not take:
            // back to the kernel, as the sweep would have done.
            if (internal::g_rechunk_recycle && n_slices > 1)
            {
                const auto a0 = internal::g_rechunk_trace ? slice_clock::now() : slice_clock::time_point{};
                set_thread_chunk_size(static_cast<unsigned int>(old_size));
                chunk_type::get_chunk_allocator(0, old_key_refill).advise_recycled_slabs();
                set_thread_chunk_size(static_cast<unsigned int>(new_size));
                if (internal::g_rechunk_trace)
                {
                    internal::g_slice_madvise_ms += slice_ms(a0, slice_clock::now());
                }
            }
        }
        g_in_rechunk_rebuild.store(false, std::memory_order_release);
        chunk_pmp_set_schedule(caller_schedule); // the caller's, untouched
        if (internal::g_rechunk_trace_mem)       // a serial count inside the refill phase
        {
            // What the new geometry actually HOLDS, counted on tmp before the
            // swap. "folded" says how many spans the rebuild could fold; this
            // says how many chunks ended up carrying a real buffer, which is
            // the data memory the re-chunk leaves behind. The two answer
            // different questions and the gap between them is the diagnosis:
            //  - few folded, many materialised -> the fold is misfiring (wrong
            //    second-value index, or the span's sources were not constants);
            //  - many folded, many materialised -> genuine merging cost: two
            //    neighbours carrying DIFFERENT constants make a chunk that no
            //    single-constant representation can hold, and it must stay
            //    materialised. That cost is real and belongs in the decider's
            //    data term.
            //  - many folded, few materialised -> the residue is not in the
            //    chunks; look at the pools (CHUNK_ARENA_REPORT=1).
            long n_mat = 0;
            for (int c = 0; c < n_new; ++c)
            {
                const auto& ch         = tmp.chunk_list[static_cast<std::size_t>(c)];
                bool        compressed = ch.is_cst_val1();
                if constexpr (chunk_config::use_chunk2a)
                {
                    compressed = compressed || ch.is_cst_val2();
                }
                else if constexpr (chunk_config::use_chunk2)
                {
                    // Nested, like the four sites above: `&&` short-circuits
                    // evaluation, never name lookup, so Derived::is_not_bool is
                    // still looked up for a v1 Derived that has no such member.
                    if constexpr (Derived::is_not_bool)
                    {
                        compressed = compressed || ch.is_cst_val2(inf_new2.ind);
                    }
                }
                if (!compressed)
                {
                    ++n_mat;
                }
            }
            std::fprintf(stderr,
                         "[RECHUNK] %d -> %d: %d new chunks, %ld folded back at rebuild (%ld installed directly), "
                         "%ld still materialised (%ld MB of buffers), %ld refused merged-constant spans\n",
                         old_size, new_size, n_new, n_folded.load(), n_installed.load(), n_mat,
                         (n_mat * static_cast<long>(new_size) * static_cast<long>(sizeof(T))) >> 20, n_refused.load());
        }
    }
    else
    {
        // bool chunks (bit-packed) and anything not covered above: tiled
        // element transcription through the proxy (no raw-fill).
        const ind_type     total = self().m_size;
        constexpr ind_type TILE  = ind_type{1} << 20;
        const ind_type     bsz   = total < TILE ? total : TILE;
        std::vector<T>     buf(static_cast<std::size_t>(bsz));
        for (ind_type base = 0; base < total; base += bsz)
        {
            const ind_type len = (total - base) < bsz ? (total - base) : bsz;
            set_active_chunk_size(old_size);
            set_thread_chunk_size(old_size);
            for (ind_type k = 0; k < len; ++k)
            {
                buf[static_cast<std::size_t>(k)] = static_cast<T>(self()[base + k]);
            }
            set_active_chunk_size(new_size);
            set_thread_chunk_size(new_size);
            for (ind_type k = 0; k < len; ++k)
            {
                tmp[base + k] = static_cast<T>(buf[static_cast<std::size_t>(k)]);
            }
        }
    }
}

} // namespace shrink_array
