#pragma once

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
#include <cstdlib>     // std::getenv (CHUNK_POOL_BREAKDOWN, CHUNK_RECHUNK_TRACE, CHUNK_RECHUNK_SLICES)
#include <type_traits> // std::is_same_v, the v1 bool guard
#include <string_view> // CHUNK_RECHUNK_SCHEDULE parsing
#include <utility>     // std::swap (the array swap)
#include <vector>      // the refill buffers

#include "chunk/internal/in_t.h" // in_t<T>: the element snapshots
#include "chunk/internal/array_chunk_base.h"
#include "chunk/internal/array_chunk_rechunk.h"
#include "chunk/array_chunk_range_ops.h"        // copy_range -- the public bulk fill
#include "chunk/internal/chunk_stats.h"         // increm_chunk_rechunk* (CHUNK_STATS)
#include "chunk/internal/compression_manager.h" // CompressionManager<>::FlushReplayGuard in rechunk_refill
#include "chunk/PMP_CHUNK.h"                    // get_thr_rank_fast
#include "chunk/internal/env_parse.h"           // env_number / env_flag (checked, not atoi)
#include "chunk/internal/exact_rss.h"           // exact_rss_kib (rss_mib_now)
#include "chunk/vector_range_ops.h"             // assume_drained
#include "chunk/config.h"
#include "chunk/chunk_traits.h"
#include "chunk/internal/chunk_size_policy.h" // set_thread_chunk_size / set_active_chunk_size

namespace shrink_array::internal
{
/// A/B switch: CHUNK_RECHUNK_INSTALL_LEGACY=1 (read once) makes the Chunk2A
/// rebuild copy and fold spans whose sources are all the same second constant,
/// as before, instead of installing that constant directly.
inline const bool g_install_legacy = []
{
    const char* env = std::getenv("CHUNK_RECHUNK_INSTALL_LEGACY");
    return env != nullptr && env[0] != '\0' && env[0] != '0';
}();

/// @brief Resident set size of this process in MiB, or -1 if unreadable.
/// @details EXACT (smaps_rollup, see exact_rss.h): statm, used until Sept '26,
///          lags by gigabytes at 64 threads. Only the CHUNK_RECHUNK_TRACE
///          paths read it (per phase of a re-chunk, per slice).
inline long rss_mib_now() noexcept
{
    const long kib = shrink_array::exact_rss_kib();
    return kib >= 0 ? kib / 1024 : -1;
}

/// @brief The rebuild loop's schedule: CHUNK_RECHUNK_SCHEDULE="kind[,chunk]"
///        (static|dynamic|guided|auto, chunk >= 1 or omitted), read once;
///        guided,16 when unset or malformed. Measured at 64 threads
///        (shrink_bench_chunk2_dyn --mode perf): whole bench -3.8% vs the
///        historical dynamic,64 (34.68 -> 33.37 s), rebuild phases -7%
///        (median); dynamic,256/1024 were 2-3x slower on small rechunks.
///        CHUNK_RECHUNK_SCHEDULE=dynamic,64 restores the old schedule.
///        Serial builds parse it too; chunk_pmp_set_schedule() is then a no-op.
struct RebuildSchedule
{
    chunk_pmp_schedule schedule{chunk_pmp_sched::sched_guided, 16};
    char               text[32]{"guided,16"};
};

inline const RebuildSchedule& rebuild_schedule() noexcept
{
    static const RebuildSchedule schedule = []
    {
        RebuildSchedule parsed;
        const char*     env = std::getenv("CHUNK_RECHUNK_SCHEDULE");
        if (env == nullptr || env[0] == '\0')
            return parsed;
        char            kind[16] = {};
        int             chunk    = 0;
        const int       nfields  = std::sscanf(env, "%15[a-z],%d", kind, &chunk);
        chunk_pmp_sched k        = chunk_pmp_sched::sched_static;
        bool            ok       = nfields >= 1 && chunk >= 0;
        if (ok)
        {
            const std::string_view kv(kind);
            if (kv == "static")
                k = chunk_pmp_sched::sched_static;
            else if (kv == "dynamic")
                k = chunk_pmp_sched::sched_dynamic;
            else if (kv == "guided")
                k = chunk_pmp_sched::sched_guided;
            else if (kv == "auto")
                k = chunk_pmp_sched::sched_auto;
            else
                ok = false;
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

/// @copydoc shrink_array::print_all_reuse_pool_breakdown
/// Re-declared here rather than including all_type_chunk.h, which would close an
/// include cycle through array_chunk2A.h. Canonical declaration lives there.
void print_all_reuse_pool_breakdown(const char* label);

template <typename Derived>
void RechunkSupport<Derived, true>::rechunk_to(unsigned int new_size)
{
    const unsigned int old_size = self().size_policy_.size();
    // Captured before the swap below hands `self` tmp's key: the allocator set
    // holding the OLD geometry's blocks is (old_key, old_size), and it is the
    // one whose slabs must empty for the re-chunk to give memory back.
    const unsigned int old_key = self().alloc_key();
    if (new_size == old_size || new_size == 0)
        return;

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

    // Serial fast-alloc re-chunk is now supported: the per-size allocator pool
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
    // (same size, size 0, serial fast-alloc, empty array) is above this
    // point, and nothing below aborts. Count it for the run's stats.
    ++shrink_array::g_rechunk_count;

    // Drain THIS array's deferred WRITE ops at the OLD geometry, before
    // the t_active switch.
    // CHUNK_RECHUNK_TRACE=1: per-phase wall time, to see where a slow
    // re-chunk spends it (tmp creation, refill, flush+swap, tmp teardown).
    static const bool trace_phases = std::getenv("CHUNK_RECHUNK_TRACE") != nullptr;
    const auto        t_now        = [] { return std::chrono::steady_clock::now(); };
    const auto        t_ms         = [](auto start, auto stop)
    { return std::chrono::duration<double, std::milli>(stop - start).count(); };
    const auto                            t0 = t_now();
    std::chrono::steady_clock::time_point t1_out;
    std::chrono::steady_clock::time_point t2_out;
    std::chrono::steady_clock::time_point t3_out;
    std::chrono::steady_clock::time_point t4_out;
    // CHUNK_RECHUNK_TRACE: process RSS at each phase boundary, next to the
    // wall times. The slice trace inside rechunk_refill brackets the refill;
    // this brackets everything around it, where the remaining part of the
    // peak was measured to sit.
    const auto rss_now = [&] { return trace_phases ? internal::rss_mib_now() : -1L; };
    const long r_entry = rss_now();
    long       r_flush = -1, r_pre_tmp = -1, r_refill = -1, r_swap = -1, r_teardown = -1;

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

    // (No dense O(n) snapshot: the refill below streams old->new in
    // bounded tiles — see step 3 — instead of materialising the whole
    // logical array, which cost O(n*sizeof(T)) RAM and defeated the
    // compressed store.)

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
        Derived tmp(total); // born at S', own num_tab, primed pool

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

        // tmp's refill CANDIDATES are deliberately KEPT.
        //
        // They used to be dropped here (tmp.rechunk_drop_candidates(), same
        // geometry-mismatch rationale as the OLD-geometry removal above), which
        // "leaves *this's blocks materialised but correct". Measured cost of
        // that on shrink_bench chunk2 dyn (--divisor 20, 4 re-chunks 64->128):
        // 781 456 queued candidates never scanned, every new chunk rebuilt from
        // an all-cst2 span left materialised until teardown -- a Chunk2/2A chunk
        // only re-enters the candidate list on a write that transitions it to a
        // constant, which never happens to a chunk that already holds one --
        // and the run ended at 2 176 MB against 1 623 MB without re-chunking.
        // On the full bench (64 threads, 10^9 elements) one re-chunk of
        // tab_short cost +985 MB in steady state, chunk2a.
        //
        // The geometry-mismatch hazard the drop guarded against no longer
        // exists: every candidate tuple carries the chunk's own geometry
        // (field <3>, captured at enqueue) and all three variants scan and
        // compress at that size, not at the ambient alloc_chunk_count().
        //
        // What makes keeping them correct across the swap below:
        //  - the tuples point into tmp.chunk_list's storage, which the
        //    std::swap hands to *this unchanged (buffer swap, no move);
        //  - they carry tmp's num_tab, and rechunk_swap_identity() makes that
        //    *this's identity (member swap for Chunk/Chunk2A, registry swap for
        //    Chunk2), so the sweep's reset_tab_inside_list_to_compress(num_tab,
        //    num_chunk) lands on the row allocated for tmp at new_size;
        //  - tmp's own destruction removes candidates by ITS post-swap num_tab
        //    (= *this's old identity, already emptied at line ~94) and, for 2A,
        //    by the OLD chunk_list's pointer range -- neither touches these.
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
        // t_active is restored to old_size, NOT the saved_t_active captured
        // at entry: t_active is a thread-local that every write path now
        // self-corrects to ITS OWN array's chunk_size() right before
        // materialising (see Proxy2::ensure_writable_storage() and its
        // siblings), so by the time this rechunk_to() call ran, it could
        // already reflect whatever UNRELATED array was written last --
        // with no guaranteed relationship to old_size. Restoring to that
        // stale, effectively-arbitrary value routes free_elt_pow2() below
        // (via get_chunk_allocator(), which keys on the ambient) to the
        // WRONG size pool -- same failure family as sl_contains(list,
        // node) assertions / SIGSEGVs, just triggered from the free side.
        // old_size is reliable: captured once, from this array's own
        // self().size_policy_, before anything below could have disturbed it.
        // g_active is unaffected by this: only rechunk_to() ever touches
        // it (via set_active_chunk_size()), never a per-write self-
        // correction, so saved_g_active stays a trustworthy baseline.
        set_active_chunk_size(saved_g_active);
        set_thread_chunk_size(old_size);

        t4_out = t_now();
        r_swap = rss_now();
        // tmp destroyed here: clear_core drops tmp's candidates by num_tab
        // and parks its OLD-geometry blocks by size.
    }
    const auto t5 = t_now(); // tmp destroyed
    r_teardown    = rss_now();

    // Free the OLD-geometry blocks tmp just parked into the old-size reuse
    // list. Nothing pops them again — this array (and, in a global
    // re-chunk, every array) has moved off old_size — so without this they
    // accumulate as pure RETENTION across successive re-chunks: memory
    // grows by ~one array's worth of buffers per geometry change (observed
    // 20 GB after repeated re-chunks). Returns them to the OS under
    // std-alloc; also removes stale-geometry buffers so a later pop can
    // never grab a wrong-size block. NOTE: with an immortal-arena fast
    // allocator these pages cannot be returned — retention is inherent
    // there until the allocator gains per-size recycling.
    // TEMPORARY (re-chunk retention hunt): what do the pools hold on either side
    // of the drain? Forcing a re-chunk to 256 on chunk2a at 64 threads left
    // 19677 MB parked out of a 31100 MB footprint, against 16730 MB with no
    // re-chunk at all -- the move ADDS 14 GB. Printing the breakdown here says
    // whether the drain never ran, ran against the wrong key, or ran on a store
    // it cannot reach. Enable with CHUNK_POOL_BREAKDOWN=1.
    const bool pool_dbg = std::getenv("CHUNK_POOL_BREAKDOWN") != nullptr;
    if (pool_dbg)
        print_all_reuse_pool_breakdown("before drain");

    self().rechunk_drain_reuse_list(old_size);

    if (pool_dbg)
    {
        std::fprintf(stderr, "[POOLS] drained old_size=%d, new geometry is %d\n", old_size, new_size);
        print_all_reuse_pool_breakdown("after  drain");
    }
    // The array has left (old_key, old_size) for good: its slabs there hold
    // nothing any more once the drain above has run, but manage_full_empty()
    // only inspects a slab at the instant a block is freed, so the ones that
    // became empty without that being observed stay resident. Sweep them now.
    // Measured at 64 threads: 23 362 / 18 300 / 8 600 / 18 261 empty slabs with
    // zero live blocks after the four re-chunks of iteration 1000, and no
    // madvise at all before this.
    // Key 0 is swept too. It is the shared per-type pool every array below
    // CHUNK_ALLOC_AFFINITY_MIN_BYTES falls back to, it was never visited, and
    // it is where the memory was: measured at 64 threads, iteration 1000, after
    // the four re-chunks -- double 2106 empty slabs, complex 1480, short 192,
    // all with zero live blocks and all still resident. That is the bulk of the
    // 2.4 GB the process kept while the chunk-level accounting balanced.
    //
    // Sweeping a SHARED pool means trusting nn_ref again, which is what the
    // hot path lost the right to do. The difference is the context, not the
    // counter: manage_full_empty() fired on every free, concurrently, where a
    // transient zero is a real possibility; this runs at the serial, quiescent
    // point at the end of a re-chunk, with no concurrent free in flight, so an
    // empty slab is empty. Emptiness is also the whole claim -- a key-0 slab
    // mixes several arrays' blocks and we make no assumption about ownership.
    if constexpr (build_config::use_fast_alloc && build_config::use_openmp)
    {
        const unsigned int saved_sz = thread_chunk_size();
        set_thread_chunk_size(old_size);
        for (int r = 0; r < CHUNK_PMP_RANK_BOUND; ++r)
        {
            if (old_key != 0)
                chunk_type::get_chunk_allocator(r, old_key).reclaim_empty_slabs_now();
            chunk_type::get_chunk_allocator(r, 0u).reclaim_empty_slabs_now();
        }
        set_thread_chunk_size(saved_sz);
    }
    if (std::getenv("CHUNK_RECHUNK_TRACE") || pool_dbg)
    {
        // Occupancy of the allocator set the array just left. Its slabs hold
        // nothing but this array's old-geometry blocks, so once the drain has
        // run they should be empty and their pages given back. A non-zero
        // "live" here is the whole question: blocks of that geometry are still
        // out somewhere, which is why no slab can empty and why the re-chunk's
        // memory never comes back.
        // Reported for the array's OWN key and, separately, for key 0 -- the
        // shared per-type pool every array falls back to below the affinity
        // threshold. The drain and the slab sweep only ever visit the array's
        // own key, so anything of the old geometry that lives in key 0 is
        // never routed home and never gives its pages back. A slab there
        // mixes several arrays' blocks and cannot empty on one array's
        // re-chunk, by construction.
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
                        for (; node != nullptr; node = node->next)
                        {
                            ++slabs;
                            live += static_cast<std::size_t>(node->nn_ref);
                            if (node->nn_ref == 0)
                                ++empty;
                        }
                }
                set_thread_chunk_size(saved);
            }
        };
        std::size_t slabs = 0, live_blocks = 0, empty_slabs = 0;
        scan(old_key, slabs, live_blocks, empty_slabs);
        std::size_t s0 = 0, l0 = 0, e0 = 0;
        if (old_key != 0)
            scan(0u, s0, l0, e0);
        const std::size_t blk = static_cast<std::size_t>(old_size) * sizeof(T);
        std::fprintf(stderr,
                     "[RECHUNK-POOL] S=%u key=%u after drain: slabs=%zu  live_blocks=%zu (%zu MB)  "
                     "empty_slabs=%zu | key=0 (shared, never swept): slabs=%zu  live_blocks=%zu (%zu MB)  "
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
                     "flush+swap %.0f | tmp teardown %.0f | old-pool drain %.0f | total %.0f\n",
                     old_size, new_size, t_ms(t0, t1_out), t_ms(t1_out, t2_out), t_ms(t2_out, t3_out),
                     t_ms(t3_out, t4_out), t_ms(t4_out, t5), t_ms(t5, t6), t_ms(t0, t6));
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
    // transcription (108s -> 8s), validated correct at 1/8/64 threads + real
    // runs. The tiled transcription (else branch) remains for a build that
    // selects no chunk variant.
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
        // re-chunk instead of 15.6 M times. `ind_old2` selects the SOURCE
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
                    return self().get_ind_second_value();
                else
                    return 0;
            }
            else
                return 0;
        }();
        [[maybe_unused]] const auto inf_new2 = [&]
        {
            if constexpr (chunk_config::use_chunk2)
            {
                if constexpr (Derived::is_not_bool)
                    return tmp.get_info_second_value();
                else
                    return 0;
            }
            else
                return 0;
        }();
        // Fining only (v1, OpenMP): how many new chunks read one old chunk,
        // and how many have finished with each -- see the release below.
        [[maybe_unused]] const int                     fine_ratio = old_size > new_size ? old_size / new_size : 1;
        [[maybe_unused]] std::vector<std::atomic<int>> old_done;
        if constexpr (chunk_config::use_chunk && build_config::use_openmp)
        {
            if (old_size > new_size)
                old_done = std::vector<std::atomic<int>>(static_cast<std::size_t>(self().nb_chunk));
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
                return;
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
                        any_cst1_src = true;
                }
                if (!och.is_cst_val1())
                {
                    all_cst1 = false;
                    if constexpr (chunk_config::use_chunk2a)
                    {
                        if (!och.is_cst_val2())
                            all_folded = false;
                        else if (cst2_src == nullptr)
                            cst2_src = och.values_load();
                        else if (och.values_load() != cst2_src)
                            same_cst2 = false;
                    }
                    else if constexpr (chunk_config::use_chunk2)
                    {
                        // Nested for the same reason as ind_old2 above.
                        if constexpr (Derived::is_not_bool)
                        {
                            // Same test as 2A, with the second-value index
                            // Chunk2 needs and 2A carries implicitly.
                            if (!och.is_cst_val2(ind_old2))
                                all_folded = false;
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
            // those allocations touch stay with the process: 130-640 MB per
            // 128 -> 256 re-chunk on the 64-thread bench, which turned the
            // forecast gains into losses. This is what a type-2 compression
            // commits (find_or_set on the chunk's second-value slot, type
            // table, values pointer), minus the buffer; tmp's chunk is
            // reachable by no other thread until the swap, so no CAS.
            if constexpr (chunk_config::use_chunk2a && !std::is_same_v<T, bool>)
            {
                if (all_folded && same_cst2 && !any_cst1_src && cst2_src != nullptr && !internal::g_install_legacy)
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

            // read *this at OLD (const operator[] = pure read); TLS is per-thread
            set_thread_chunk_size(old_size);
            for (ind_type k = 0; k < len; ++k)
                buf[static_cast<std::size_t>(k)] = static_cast<T>(std::as_const(self())[base + k]);

            // Release the source chunks this span has just consumed, NOW rather
            // than at tmp's destruction.
            //
            // rechunk_to() builds tmp at S' while *this still holds every chunk
            // at S, so the two coexist until the swap and the peak is close to
            // the sum -- which is what makes @dyn cost roughly twice its static
            // counterpart in the bench (chunk2a: 6984 MB @256 vs 17028 MB @dyn
            // at 64 threads), even though the steady state barely moves.
            //
            // A source chunk is done for good once every element of it has been
            // transcribed. Sizes are powers of two, so the boundaries align
            // exactly. When coarsening, this span covers every chunk of
            // [old_first, old_last] whole, and releases them. When fining,
            // old_size / new_size new chunks read the same old one, on
            // whichever threads the schedule gives them: the LAST to finish
            // releases it, counted per old chunk (old_done). The release used
            // to be the job of the last new chunk BY INDEX, "exactly one c, no
            // coordination" -- which freed the source while the thread holding
            // the previous index was still reading it. Any schedule whose block
            // boundary can fall between c and c+1 hit it: guided,16 (the
            // default since Sept '26), guided,64 and static gave 952 to 1560
            // wrong elements on rechunk_validation's 512 -> 256; dynamic,64,
            // dynamic,16 and static,16 keep the pair on one thread and hid it.
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
                            continue;
                    }
                    else
                    {
                        const ind_type oc_end = static_cast<ind_type>(oc + 1) * static_cast<ind_type>(old_size);
                        if (oc_end > end)
                            continue; // still needed by the next new chunk
                    }
                    auto& och = self().chunk_list[static_cast<std::size_t>(oc)];
                    // release_st, not clear_st: a release once copied, not a
                    // compression (CHUNK_STATS counted every rebuilt chunk as one).
                    if (!och.is_cst_val1())
                        och.release_st(thr, old_size);
                }
            }

            // Fill tmp's new chunk c through the public bulk API.
            //
            // This used to be a hand-written arm + raw-fill, kept here because
            // the per-element proxy path costs 13x more (108s -> 8s measured).
            // copy_range() IS that fast path, made public and tested: it takes
            // one decision per chunk, arms with real proxy writes, stores the
            // rest straight into the buffer, and restores the zero counter the
            // bulk store bypasses. It also carries the same variant split --
            // bool (bit-packed) goes per element -- which no longer has to be
            // restated here.
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
            // for the next end-of-phase sweep means the WHOLE array's cst2 share
            // sits materialised until then: measured on the full bench (64
            // threads, 10^9 elements, chunk2a), the four 64->128 re-chunks of
            // iteration 1000 took the process from 10.2 GB to 21.6 GB before
            // the sweep -- tab1 alone is 72% cst2, 5.7 GB of transient buffers.
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
            // array's inf_array tuple -- but the fold is the same: re-scan and
            // clear_int_st with the variant's own state, nothing synthesised.
            // Without it Chunk2 left its whole cst2 share materialised until
            // the next end-of-phase sweep: 72.6% of tab1's chunks, measured at
            // +10.1 GB after the four re-chunks of iteration 1000 (64 threads,
            // 10^9 elements) where 2A, already folding, GAINED 674 MB.
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
                        ++tally.refused;
                }
            }
            // No v1 branch. v1 has one compressible constant, zero, and the
            // spans whose sources were all cst1 already took the early return
            // above; every other rebuilt chunk holds the writes the prologue's
            // flush just materialised. A branch that scanned each of them and
            // folded the uniformly-zero ones was measured at 64 threads,
            // iteration 1000: 0 folded on tab1 / tab_short / tab_complex, 1392
            // on tab2 -- 0.02% -- for a full buffer scan on 6 M chunks. The
            // end-of-phase sweep does the same work later, on the chunks that
            // have actually become zero by then. What that branch DID leave
            // behind is the guard in Chunk<T>::compress(): a second visit to a
            // cst1 chunk must not park the zero sentinel, and that guard is
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
                // That constant IS use_openmp, so in a SERIAL build the overload
                // does not exist and this branch failed to compile -- chunk2 +
                // serial + dyn, the one combination of {chunk, chunk2, chunk2a} x
                // {serial, OpenMP} nobody had built. A serial run has no deferred
                // compression to fold anyway.
                if constexpr (chunk_config::use_fast_block_compression && Derived::is_not_bool)
                {
                    if (all_folded)
                    {
                        ++tally.folded;
                        auto& nch = tmp.chunk_list[static_cast<std::size_t>(new_chunk)];
                        nch.compress(get_thr_rank_fast(), inf_new2,
                                     /*apply=*/true, static_cast<unsigned int>(new_size));
                        if (!nch.is_cst_val1() && !nch.is_cst_val2(std::get<1>(inf_new2)))
                            ++tally.refused;
                    }
                }
            }
        };

        // Parallel rebuild: each thread takes disjoint new chunks and drives the
        // write path on its own rank; dynamic schedule balances cst1-skip (cheap)
        // vs materialise (costly). Geometry is per-array so no global race; only
        // per-thread TLS + per-rank allocator/compression state are touched.
        //
        // Suppress the OMPT parallel-end finalize: this region's end must NOT
        // flush/compress the half-written array (it corrupts grow regions that
        // span several old chunks). rechunk_to flushes at a quiescent point
        // after refill. Set on the encountering thread, around the region only.
        // SLICED rebuild. Inside one region v1 releases each source chunk as
        // soon as it is transcribed (clear_st in rebuild_one), but a block
        // freed by a rank that does not own its slab is only PARKED in that
        // rank's reuse list -- walking the owner's slab chains from inside a
        // parallel region is exactly what the owner-routed drain exists to
        // avoid. So no slab empties and no page comes back until the drain that
        // follows the WHOLE rebuild, while the new geometry is materialised in
        // full: peak = old + new. Measured +6.4 GB at iteration 1000 of the
        // 64-thread bench, the two storages of tab1 coexisting.
        //
        // Draining at each slice boundary -- a serial point, the region has
        // closed -- routes that slice's parked blocks home, empties the old
        // slabs of the range just consumed and gives their pages back BEFORE
        // the next slice materialises, bounding the peak to ~old + new/slices.
        // rebuild_one folds each new chunk back to a constant when it can, so
        // a drained slice leaves little behind.
        //
        // CHUNK_RECHUNK_SLICES sets the count (default 8; 1 = one pass, the
        // previous behaviour). v1 + fast alloc + OpenMP only: Chunk2/Chunk2A do
        // not release their source chunks during the copy (their clear_st
        // carries compression state this loop has no business synthesising), so
        // they have nothing to give back between slices and the extra regions
        // would be pure cost.
        int n_slices = 1;
        if constexpr (chunk_config::use_chunk && build_config::use_fast_alloc && build_config::use_openmp)
        {
            // A/B at 64 threads, peak above the pre-re-chunk level:
            //   1 slice  +6918 MB
            //   16       +1703 MB   highest slice boundary 32435, intra-slice term 834 MB
            //   32       +1335 MB   highest slice boundary 32526, intra-slice term 359 MB
            // Doubling halves the intra-slice term, as 1/n predicts, but the
            // boundary maximum does not move: ~755 MB of accumulated
            // coexistence is insensitive to slicing by construction -- it comes
            // from blocks of one slab being scattered across the index, not
            // from slice size. 16 is where both curves are still clearly
            // favourable; 32 traded 368 MB of peak for what looked like 266 MB
            // of steady state on a single run.
            // Reordering the rebuild by slab was measured and ruled out.
            // [RECHUNK-LOCALITY], since retired, sampled every 64th new chunk on
            // all four arrays: a new chunk's two source buffers share a slab
            // 98.2-99.8% of the time, and one slice of 32 touches 4.2-6.2% of
            // the slabs where a perfectly slab-local order would touch 3.1%.
            // Index order is therefore already local to within a factor of
            // 1.4-2, so rebuilding in slab order could not recover the ~755 MB
            // floor: that floor is the two geometries coexisting at slab
            // granularity, not blocks scattered across the index.
            n_slices = 16;
            n_slices = std::max(1, shrink_array::env_number<int>("CHUNK_RECHUNK_SLICES", n_slices));
            // Not worth slicing an array too small to empty a slab per slice.
            n_slices = std::min(n_slices, std::max(1, n_new / 4096));
        }
        // The affinity key of the set being emptied. `tmp` carries the new one;
        // rechunk_to() swaps them only after this call returns, so self()'s key
        // is still the old geometry's here.
        const unsigned int old_key_refill = self().alloc_key();

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
            if (std::getenv("CHUNK_RECHUNK_TRACE"))
                std::fprintf(stderr, "[RECHUNK] rebuild schedule: %s\n", rs.text);
        }
        g_in_rechunk_rebuild = true;
        // Its own knob: one statm read per slice is nothing, but 18 to 34 lines
        // per array bury the summary that CHUNK_RECHUNK_TRACE is asked for.
        // This is the only view of the peak's SHAPE -- saw-tooth means the
        // drains return memory and the coexistence is inside a slice, a
        // monotonic climb would mean they return nothing -- and it will be
        // wanted again if Chunk2/2A ever release their sources during the copy
        // and become sliceable.
        const bool slice_trace = std::getenv("CHUNK_RECHUNK_TRACE_SLICES") != nullptr;
        const long rss_start   = slice_trace ? internal::rss_mib_now() : -1;
        if (slice_trace)
            std::fprintf(stderr, "[RECHUNK-SLICE] 0/%d start       rss=%ld MB\n", n_slices, rss_start);
        for (int sl = 0; sl < n_slices; ++sl)
        {
            const int c0 = static_cast<int>((static_cast<long>(n_new) * sl) / n_slices);
            const int c1 = static_cast<int>((static_cast<long>(n_new) * (sl + 1)) / n_slices);
#pragma omp parallel
            {
                // Apply the rebuild's arming writes DIRECTLY, never via the deferred
                // path. Without this, tmp[base+kw]=... enqueues a deferred op into tmp's
                // per-chunk op_map; the arming loop below writes EVERY element while the
                // chunk still reads cst1 (a deferred write never materialises in place),
                // so the op_map fills and a rehash (increase_size) frees its old bucket
                // storage while an already-enqueued op still targets it -> heap-use-after-
                // free when that op is later flushed (ASan: WRITE in assign_impl's deferred
                // body, freed by unordered_dense increase_size, all under rechunk_refill).
                // in_flush_replay routes each write straight through write_body
                // (materialise + store, no enqueue): the first arming write materialises,
                // the while-loop exits, and the raw-fill does the rest. Safe here -- each
                // thread rebuilds DISJOINT new chunks, so the bypassed Dekker/gate guard
                // nothing. Per-thread (thread_local), so it lives inside the region.
                typename CompressionManager<T, chunk_type>::FlushReplayGuard rechunk_replay_guard;
                std::vector<T>                                               buf(static_cast<std::size_t>(new_size));
                RebuildTally tally; // this thread's fold counts for this slice
                                    // schedule(runtime): CHUNK_RECHUNK_SCHEDULE, default guided,16
                                    // (internal::rebuild_schedule).
#pragma omp for schedule(runtime)
                for (int c = c0; c < c1; ++c)
                    rebuild_one(c, buf, tally);
                // One RMW per thread and slice, not one per folded chunk.
                n_folded.fetch_add(tally.folded, std::memory_order_relaxed);
                n_refused.fetch_add(tally.refused, std::memory_order_relaxed);
                n_installed.fetch_add(tally.installed, std::memory_order_relaxed);
            }
            if (sl + 1 < n_slices)
            {
                // Serial point: send this slice's parked old-geometry blocks
                // home, then give back the slabs they just emptied. Only blocks
                // already released by clear_st are touched -- the source chunks
                // the remaining slices still have to read are untouched.
                set_thread_chunk_size(static_cast<unsigned int>(old_size));
                self().rechunk_drain_reuse_list(static_cast<unsigned int>(old_size));
                if constexpr (build_config::use_fast_alloc && build_config::use_openmp)
                {
                    if (old_key_refill != 0)
                        for (int r = 0; r < CHUNK_PMP_RANK_BOUND; ++r)
                            chunk_type::get_chunk_allocator(r, old_key_refill).reclaim_empty_slabs_now();
                }
                set_thread_chunk_size(static_cast<unsigned int>(new_size));
                if (slice_trace)
                {
                    const long rss_mib = internal::rss_mib_now();
                    std::fprintf(stderr, "[RECHUNK-SLICE] %d/%d after-drain rss=%ld MB  (%+ld since start)\n", sl + 1,
                                 n_slices, rss_mib, rss_mib - rss_start);
                }
            }
        }
        if (slice_trace)
        {
            const long rss_mib = internal::rss_mib_now();
            std::fprintf(stderr, "[RECHUNK-SLICE] %d/%d end         rss=%ld MB  (%+ld since start)\n", n_slices,
                         n_slices, rss_mib, rss_mib - rss_start);
        }
        g_in_rechunk_rebuild = false;
        chunk_pmp_set_schedule(caller_schedule); // the caller's, untouched
        if (std::getenv("CHUNK_RECHUNK_TRACE"))
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
            //    data term, which today predicts a couple of MB.
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
                        compressed = compressed || ch.is_cst_val2(std::get<1>(inf_new2));
                }
                if (!compressed)
                    ++n_mat;
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
                buf[static_cast<std::size_t>(k)] = static_cast<T>(self()[base + k]);
            set_active_chunk_size(new_size);
            set_thread_chunk_size(new_size);
            for (ind_type k = 0; k < len; ++k)
                tmp[base + k] = static_cast<T>(buf[static_cast<std::size_t>(k)]);
        }
    }
}

} // namespace shrink_array
