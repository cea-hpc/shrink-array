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

/**
 * @file param_chunk.cpp
 * @brief Bodies of the ParamChunk façade: startup wiring and the exit report.
 *
 * Three things live here, and they are unrelated to each other beyond sharing
 * the façade:
 *
 *  - **Startup.** init_indexes() sizes the allocation block from the thread
 *    count, publishes it, and asks AllTypeChunk to build whatever the build
 *    needs -- per-thread structures and allocators under OpenMP, a single
 *    allocator otherwise. That half is one verb with two bodies,
 *    init_thread_structures(), on ParamChunkParallelInterface rather than on
 *    ParamChunk itself: each body calls AllTypeChunk verbs the other build
 *    does not declare.
 *  - **The geometry registration hook**, for dynamic builds.
 *  - **ChunkStats()**, the instrumentation report, which is most of the file.
 *    It is a presentation routine: it reduces the per-thread counters and
 *    prints them. The counting happens elsewhere.
 *
 * @note Everything here is sequential-context. Nothing in this file is safe to
 *       call from inside a parallel region.
 */

//
// Created by ballereaup on 07/05/24.
//

#include <algorithm> // std::max
#include <chrono>    // steady_clock, the gate rates (CHUNK_STATS report only)
#include <cstddef>   // std::size_t (CHUNK_STATS report only: the retention)
#include <iomanip>   // std::setprecision (CHUNK_STATS report only: the ratios)
#include <iostream>  // std::cout, std::cerr
#include <mutex>     // std::lock_guard (CHUNK_STATS report only: the size distribution)
#include <utility>   // std::cmp_greater
#include <unistd.h>  // sysconf(_SC_PAGESIZE): page-aware slab floor

#include "chunk/config.h"                              // build_config::chunk_dynamic, use_openmp
#include "chunk/singleton.h"                           // SingletonChk
#include "chunk/internal/chunk_size_config.h"          // __chunk_size
#include "chunk/internal/chunk_size_policy.h"          // active_chunk_size
#include "chunk/internal/chunk_stats.h"                // per-variant stats reduce/print
#include "chunk/param_chunk.h"                         // the declarations defined here
#include "chunk/PMP_CHUNK.h"                           // CHUNK_PMP_MAX_THREADS
#include "chunk/internal/write_compress_sync_policy.h" // dekker::protocol_name (CHUNK_STATS report only)
#include "chunk/all_type_chunk.h"                      // AllTypeChunk

namespace shrink_array
{

/// @brief Allocation block (slab) size in bytes, computed once by init_indexes().
int ParamChunk::block_size;
// No #ifdef _OPENMP around these two bodies, and the reason is the AllT
// parameter of ParamChunkParallelInterface (see param_chunk.h, which carries
// the full rationale): the bodies are TEMPLATES, so only the one this build
// selects is instantiated -- by the single explicit instantiation at the end of
// the pair -- and the other is never compiled. Both reach the façade through
// SingletonChk<AllT>, never SingletonChk<AllTypeChunk>: the calls MUST stay
// dependent, or phase-1 lookup checks them against the concrete type at
// definition time and the body of the build we are not in fails to compile.

/// @copydoc ParamChunkParallelInterface<true>::init_thread_structures
template <typename AllT>
void ParamChunkParallelInterface<true, AllT>::init_thread_structures()
{
    if constexpr (build_config::use_fast_alloc)
    {
        std::cout << "Using \"fast allocator\" for chunks"
                  << "  [build: reclaim_empty_slabs=" << (build_config::reclaim_empty_slabs ? "ON" : "OFF")
                  << ", alloc_affinity_min_bytes=" << (build_config::alloc_affinity_min_bytes >> 20U) << "MiB]\n";
    }
}

// The one instantiation of the pair, and the whole point of the AllT parameter:
// the body of the OTHER thread model is left uninstantiated rather than
// #ifdef'd out. Mirrors all_type_chunk.cpp's
// `template class AllTypeChunkParallelInterface<AllTypeChunk, build_config::use_openmp>;`.
template struct ParamChunkParallelInterface<build_config::use_openmp, AllTypeChunk>;

/**
 * @copydoc ParamChunk::init_indexes
 *
 * @details The block size is a budget divided by the thread count -- 2 MiB of
 *          chunks per process, or 15 MiB with @p big_chunk_block, split across
 *          the ranks and floored at 240 KiB so a large team does not end up
 *          with blocks too small to amortise anything.
 *
 * @warning A chunk larger than one block cannot be allocated at all; that is
 *          what the warning here is about, and it is only a warning because the
 *          geometry is a compile-time constant the caller cannot fix at run time.
 */
void ParamChunk::init_indexes(bool big_chunk_block)
{
    auto& ens_type_chunk = SingletonChk<AllTypeChunk>::getRef();

    int sz = 1024 * 1024 * 2;
    if (big_chunk_block)
    {
        sz = 1024 * 1024 * 15;
    }

    constexpr int sz_min = 1024 * 16 * 15;

    sz /= CHUNK_PMP_MAX_THREADS; // taking into account the number of threads

    sz = std::max(sz_min, sz);
    // Page-aware floor. An emptied slab gives back all its pages but the last
    // (trailing back-pointer), so the retained share is one page per slab:
    // 256 KiB slabs on 64 KiB pages keep 25% -- measured at 64 threads after
    // four re-chunks, ~60 000 emptied slabs kept ~3.7 GB. Sixteen pages per
    // slab caps that at ~6% (1 MiB slabs on 64 KiB pages; on 4 KiB pages the
    // 240 KiB floor above already dominates). The cost is one partly-filled
    // slab per (rank, array, size): 64 x 13 x 1 MiB worst case.
    if (const long ps = ::sysconf(_SC_PAGESIZE); ps > 0)
    {
        sz = std::max(sz, static_cast<int>(16 * ps));
    }

    set_block_size(sz);

    if (std::cmp_greater(__chunk_size, sz))
    {
        std::cerr << "Caution: chunk sizes can't be larger than one page.\n";
    }

    // The per-type structures (candidate lists, per-rank lists and
    // allocators) are built when each type registers its manager -- at its
    // first array, or through init_indexes<Types>() -- with the block size
    // published here. Nothing per type is left at this point.
    init_thread_structures(); // the allocator banner

    if constexpr (chunk_config::stats)
    {
        init_all_stats();
    }

    ens_type_chunk.mark_ready(block_size);

    // Register the STARTING geometry, now that the managers exist.
    //
    // g_active_chunk_size is initialised statically to __chunk_size
    // (chunk_size_policy.h), so the size every array is born with never goes
    // through set_active_chunk_size() and would never reach the per-size
    // free-list map: a run that never re-chunks would have an EMPTY map,
    // get_list_block_free_thr_for_size() would fall back to the geometry-blind
    // legacy store for every buffer, and drain_reuse_list_for_size() -- which
    // only walks the map -- could never reach them again (v1 dyn, 64 threads:
    // 30 GB parked there, out of a 51 GB footprint).
    //
    // Idempotent and sequential; the size is already in g_active_chunk_size.
    if constexpr (build_config::chunk_dynamic)
    {
        register_runtime_chunk_size(active_chunk_size());
    }
}

/// @copydoc ParamChunk::force_compression
void ParamChunk::force_compression()
{
    // Declared in every build so call sites need no guard; it simply has nothing
    // to force when the build has no fast block compression.
    if constexpr (chunk_config::use_fast_block_compression)
    {
        const auto& ens_type_chunk = SingletonChk<AllTypeChunk>::getRef();

        ens_type_chunk.force_chunk_to_compress();
    }
}

#ifdef CHUNK_STATS

/**
 * @copydoc ParamChunk::ChunkStats
 *
 * @details Two passes over the variants, and the first exists only to decide
 *          what the legend should say: a legend describing counters that never
 *          appear -- sync on an uncontended run, re-chunk in a static build --
 *          is noise, so the totals are reduced first and each legend line is
 *          emitted only if something will stand behind it.
 *
 * @note Presentation only. Every number printed here is reduced from counters
 *       maintained elsewhere; nothing is measured in this function.
 */
void ParamChunk::ChunkStats()
{
    // Per-variant report: reduce each variant's per-thread counters and print a
    // block for chunk / chunk2 / chunk2a. Variants with no activity are skipped.

    // Legend: printed BEFORE the blocks, but filtered on what those blocks will
    // actually contain -- a legend describing counters that never appear (sync on
    // an uncontended run, rechunk in a static build) is just noise. So reduce every
    // variant first, then emit only the lines that have something behind them.
    ChunkStatsThr all{};
    for (int v = 0; v < CHUNK_STATS_N_VARIANT; ++v)
    {
        const ChunkStatsThr stats = reduce_chunk_stats(v);
        all.alloc += stats.alloc;
        all.write += stats.write;
        all.read += stats.read;
        all.read_cst += stats.read_cst;
        all.read_sync += stats.read_sync;
        all.read_sync_cross += stats.read_sync_cross;
        all.create += stats.create;
        all.list += stats.list;
        all.bytes_alloc += stats.bytes_alloc;
        all.bytes_reused += stats.bytes_reused;
        all.bytes_freed += stats.bytes_freed;
        all.teardown_clear += stats.teardown_clear;
        all.compress += stats.compress;
        all.still_compressed_at_exit += stats.still_compressed_at_exit;
        all.release_park += stats.release_park;
        all.release_pool += stats.release_pool;
        all.release_free += stats.release_free;
        all.compress_attempt += stats.compress_attempt;
        all.mark_ok += stats.mark_ok;
        all.mark_refused += stats.mark_refused;
        all.scan_read += stats.scan_read;
        all.rechunk += stats.rechunk;
        all.gate_start += stats.gate_start;
        all.gate_start_flush_yield += stats.gate_start_flush_yield;
        all.gate_start_yield_late += stats.gate_start_yield_late;
        all.gate_start_deferred += stats.gate_start_deferred;
        all.gate_flush += stats.gate_flush;
        all.gate_driver += stats.gate_driver;
        all.gate_deferred += stats.gate_deferred;
        all.gate_wait += stats.gate_wait;
        all.gate_wait_yields += stats.gate_wait_yields;
        all.gate_mark_hit += stats.gate_mark_hit;
        all.rechunk_rebuilt += stats.rechunk_rebuilt;
        all.rechunk_skipped += stats.rechunk_skipped;
    }

    std::cout << "\n===== CHUNK_STATS (instrumentation counters, -DCHUNK_STATS) =====\n";
    if constexpr (build_config::use_openmp)
    {
        std::cout << "  write/compress handshake : " << dekker::protocol_name() << "\n";
    }
    std::cout << "  alloc          blocks obtained from a fresh allocation\n";
    if (all.list != 0)
    {
        std::cout << "  list           blocks taken from the reuse list instead (alloc+list = create)\n";
    }
    // Unconditional on purpose, despite sitting under the `if` above until the
    // indentation was corrected: the per-variant block prints a `bytes` line
    // under BOTH alloc and list, and bytes_alloc can be non-zero on a run that
    // never reused anything.
    std::cout << "    bytes        the same, in MB -- comparable across geometries\n"
                 "                 (dyn builds), where a raw block count is not\n";
    std::cout << "  write          element writes (a write of the compressible value is not counted)\n"
                 "    hit/materialize of those, how many needed to materialise a\n"
                 "                 compressed chunk first vs write straight into one\n"
                 "  read           element reads by the application, split into:\n"
                 "   constant      served straight from the constant: the chunk was\n"
                 "                 compressed, so no data was touched -- this is the\n"
                 "                 share the compression actually saves on reads\n"
                 "   materialised  served from a real buffer (normal memory access)\n";
    if (all.read_sync != 0)
    {
        std::cout << "   sync          accesses that had to enter the slow synchronisation\n"
                     "                 path (deferred ops had to be flushed first)\n"
                     "    cross-rank   of those, the ones that needed the expensive walk\n"
                     "                 across every rank rather than just their own\n";
    }
    std::cout << "  read/write     reads per write: the access profile of the run\n"
                 "  create         chunks materialised (a buffer was attached)\n"
                 "  compress (during exec) chunks compressed: a materialised buffer\n"
                 "                 was released, observed WHILE the program ran\n";
    if (all.still_compressed_at_exit != 0)
    {
        std::cout << "  still compressed (at exit) chunks found at the constant when\n"
                     "                 each array was destroyed -- includes chunks\n"
                     "                 never written at all, NOT just real compressions\n";
    }
    if (all.mark_ok != 0 || all.mark_refused != 0)
    {
        std::cout << "  mark           writes that flagged a chunk as a compression candidate:\n"
                     "   queued        the flag went 0->1, the chunk was queued\n"
                     "   refused       it was already 1, so nothing was queued. A rate\n"
                     "                 near 100% with a flat queued count means the flag\n"
                     "                 is never reset after the sweep\n";
    }
    if (all.compress_attempt != 0)
    {
        std::cout << "   attempts      chunks scanned as compression candidates, and how\n"
                     "                 many of those scans paid off\n"
                     "   scan reads    element reads performed BY those scans; counted apart\n"
                     "                 from `read` so they do not distort the hit rate\n";
    }
    std::cout << "  teardown_clear chunks swept by the end-of-life clear pass (NOT a\n"
                 "                 compression); non-zero only once the arrays are cleared\n";
    if (all.rechunk != 0)
    {
        std::cout << "  rechunk        geometry changes, and the work they cost:\n"
                     "   rebuilt       new chunks actually rebuilt from the old ones\n"
                     "   skipped       new chunks left compressed -- nothing to rebuild\n";
    }
    if (all.gate_start != 0 || all.gate_flush != 0 || all.gate_driver != 0)
    {
        std::cout << "  gate           the write/compress handshake's RARE side, with its\n"
                     "                 rate over the run's wall time -- each of these is one\n"
                     "                 heavy barrier in an asymmetric Dekker:\n"
                     "   start         compression start() entered (yield: gave way to a\n"
                     "                 flush; deferred: gave up waiting for writers)\n"
                     "   flush         announce_flush() calls\n"
                     "   driver        writers sent to the slow path\n";
    }
    std::cout << "=================================================================\n";

    for (int v = 0; v < CHUNK_STATS_N_VARIANT; ++v)
    {
        const ChunkStatsThr stats = reduce_chunk_stats(v);
        // Skip a variant that saw no activity at all (e.g. chunk2 in a chunk-only
        // binary): printing an all-zero block would just be noise.
        const bool any = stats.alloc != 0 || stats.write != 0 || stats.read != 0 || stats.read_cst != 0 ||
                         stats.create != 0 || stats.list != 0 || stats.teardown_clear != 0 || stats.compress != 0 ||
                         stats.compress_attempt != 0 || stats.rechunk != 0 || stats.read_sync != 0 ||
                         stats.mark_ok != 0 || stats.mark_refused != 0 || stats.gate_start != 0 ||
                         stats.gate_flush != 0 || stats.gate_driver != 0;
        if (!any)
        {
            continue;
        }
        std::cout << "\n----- CHUNK_STATS [" << chunk_variant_name[v] << "] -----\n";
        std::cout << "  alloc          = " << stats.alloc << "\n";
        if (stats.bytes_alloc != 0)
        {
            std::cout << "    bytes        = " << (stats.bytes_alloc / (1024L * 1024)) << " MB\n";
        }
        if (stats.list != 0)
        {
            std::cout << "  list           = " << stats.list << "\n";
        }
        if (stats.bytes_reused != 0)
        {
            std::cout << "    bytes        = " << (stats.bytes_reused / (1024L * 1024)) << " MB\n";
        }
        const long   reads_all = stats.read + stats.read_cst;
        const long   total_rw  = stats.write + reads_all;
        const double pct_w =
            total_rw != 0 ? 100.0 * static_cast<double>(stats.write) / static_cast<double>(total_rw) : 0.0;
        const double pct_r =
            total_rw != 0 ? 100.0 * static_cast<double>(reads_all) / static_cast<double>(total_rw) : 0.0;
        std::cout << "  write          = " << stats.write << "  (" << std::fixed << std::setprecision(1) << pct_w
                  << "% of r/w)\n";
        // Write-side counterpart to the read hit-rate below: of all writes, how
        // many landed straight in an already-materialised buffer (cheap) vs
        // how many first had to materialise a compressed chunk (`create` --
        // the same event already counted under `alloc`/`list` above).
        // Materialisation is only ever write-triggered (never by a read), so
        // this derivation needs no extra instrumentation: write_materialize
        // is exactly `create`, and write_hit is the remainder.
        if (stats.write != 0)
        {
            const long   write_materialize = stats.create;
            const long   write_hit         = stats.write - write_materialize;
            const double pct_hit           = 100.0 * static_cast<double>(write_hit) / static_cast<double>(stats.write);
            std::cout << "    hit          = " << write_hit << "  (" << std::fixed << std::setprecision(1) << pct_hit
                      << "% straight into an existing buffer)\n";
            std::cout << "    materialize  = " << write_materialize << "  (" << (100.0 - pct_hit)
                      << "% had to materialise first)\n";
        }
        std::cout << "  read           = " << reads_all << "  (" << pct_r << "% of r/w)\n";
        // Compression hit rate on reads: share served straight from the constant
        // (no data touched) vs from a materialised buffer. This is what the
        // compression actually buys on the read path.
        if (reads_all != 0)
        {
            const double hit = 100.0 * static_cast<double>(stats.read_cst) / static_cast<double>(reads_all);
            std::cout << "    constant     = " << stats.read_cst << "  (" << hit << "% hit)\n";
            std::cout << "    materialised = " << stats.read << "  (" << (100.0 - hit) << "%)\n";
        }
        // Read-side synchronisation cost: how many accesses had to enter the slow
        // sync path at all, and how many of those needed the expensive cross-rank
        // walk rather than settling on their own rank.
        if (stats.read_sync != 0)
        {
            std::cout << "    sync         = " << stats.read_sync;
            if (reads_all != 0)
            {
                std::cout << "  (" << std::fixed << std::setprecision(1)
                          << 100.0 * static_cast<double>(stats.read_sync) / static_cast<double>(reads_all)
                          << "% of reads)" << std::defaultfloat;
            }
            std::cout << "\n";
            std::cout << "     cross-rank  = " << stats.read_sync_cross << "  (" << std::fixed << std::setprecision(1)
                      << 100.0 * static_cast<double>(stats.read_sync_cross) / static_cast<double>(stats.read_sync)
                      << "% of sync)\n"
                      << std::defaultfloat;
        }
        if (stats.write != 0)
        {
            std::cout << "  read/write     = " << std::setprecision(2)
                      << static_cast<double>(reads_all) / static_cast<double>(stats.write) << " reads per write\n";
        }
        std::cout << std::defaultfloat;
        std::cout << "  create         = " << stats.create << "\n";
        // Candidate marking: a write that finds the chunk's flag at 0 flips it and
        // queues the chunk (mark_ok); one that finds it already at 1 does not
        // (mark_refused). Refusals are normal while a chunk sits in the list, but a
        // refusal rate near 100% with a flat mark_ok means the flag is never reset
        // after the sweep -- the chunk can then never be re-queued, and compression
        // silently stops for it.
        if (stats.mark_ok != 0 || stats.mark_refused != 0)
        {
            const long marks = stats.mark_ok + stats.mark_refused;
            std::cout << "  mark           = " << marks << "\n";
            std::cout << "    queued       = " << stats.mark_ok << "  (" << std::fixed << std::setprecision(1)
                      << 100.0 * static_cast<double>(stats.mark_ok) / static_cast<double>(marks) << "%)\n";
            std::cout << "    refused      = " << stats.mark_refused << "  ("
                      << 100.0 * static_cast<double>(stats.mark_refused) / static_cast<double>(marks)
                      << "% already flagged)\n"
                      << std::defaultfloat;
        }
        // `compress (during exec)` counts REAL compressions observed WHILE the
        // program runs: a materialised chunk (created by a write) transitioning
        // back to the constant. It does NOT include chunks that were simply never
        // written -- a freshly built array starts every chunk at the constant, and
        // most stay there for the whole run without a single write touching them.
        // `still compressed (at exit)`, below, is the unrelated number: how many
        // chunks ARE at the constant when each array is destroyed, whatever the
        // reason (born that way, or genuinely compressed). The two can differ by
        // orders of magnitude on a run with few writes -- that is expected, not a
        // counting bug.
        std::cout << "  compress (during exec) = " << stats.compress << "\n";
        if (stats.bytes_freed != 0)
        {
            std::cout << "    bytes        = " << (stats.bytes_freed / (1024L * 1024)) << " MB\n";
        }
        if (stats.still_compressed_at_exit != 0)
        {
            std::cout << "  still compressed (at exit) = " << stats.still_compressed_at_exit << "\n";
        }
        // Scan yield: how many of the chunks examined as compression candidates
        // actually got compressed, and what those scans cost in element reads
        // (scan_read is kept out of `read` so the read hit-rate stays about the
        // application's own accesses).
        if (stats.compress_attempt != 0)
        {
            std::cout << "    attempts     = " << stats.compress_attempt << "  (" << std::fixed << std::setprecision(1)
                      << 100.0 * static_cast<double>(stats.compress) / static_cast<double>(stats.compress_attempt)
                      << "% succeed)\n"
                      << std::defaultfloat;
            std::cout << "    scan reads   = " << stats.scan_read << "\n";
        }
        // teardown_clear: chunks swept by the end-of-life clear_st pass -- NOT a
        // compression (compress counts those). Normally 0 in a perf run.
        std::cout << "  teardown_clear = " << stats.teardown_clear << "\n";
        std::cout << "  released       = " << stats.released
                  << "  (buffers released without a compression: re-chunk sources, teardown sweeps)\n";
        if (stats.gate_start != 0 || stats.gate_flush != 0 || stats.gate_driver != 0)
        {
            const double secs =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - __chunk_stats_t0).count();
            const auto rate = [secs](long count) { return secs > 0 ? static_cast<double>(count) / secs : 0.0; };
            std::cout << std::fixed << std::setprecision(0);
            std::cout << "  gate start     = " << stats.gate_start << "  (" << rate(stats.gate_start) << "/s; yield "
                      << stats.gate_start_flush_yield << " of which late " << stats.gate_start_yield_late
                      << ", deferred " << stats.gate_start_deferred << ")\n";
            std::cout << "  gate flush     = " << stats.gate_flush << "  (" << rate(stats.gate_flush) << "/s)\n";
            std::cout << "  gate driver    = " << stats.gate_driver << "  (" << rate(stats.gate_driver) << "/s)\n";
            std::cout << "  gate deferred  = " << stats.gate_deferred << "  (" << rate(stats.gate_deferred)
                      << "/s; writes parked in a rank queue while a compressor was in flight)\n";
            std::cout << "  gate wait      = " << stats.gate_wait << "  (" << rate(stats.gate_wait)
                      << "/s; writers that found the gate shut, " << stats.gate_wait_yields
                      << " yields of 1000 pauses; " << stats.gate_mark_hit << " on a chunk marked by a pass)\n";
            std::cout << "  gate wall time = " << std::setprecision(1) << secs << " s\n" << std::defaultfloat;
        }
        // Re-chunk counters exist only in a dynamic build (a static geometry
        // never re-chunks), so the block is compiled out elsewhere.
        if constexpr (build_config::chunk_dynamic)
        {
            if (stats.rechunk != 0)
            {
                const long touched = stats.rechunk_rebuilt + stats.rechunk_skipped;
                std::cout << "  rechunk        = " << stats.rechunk << "\n";
                std::cout << "    rebuilt      = " << stats.rechunk_rebuilt;
                if (touched != 0)
                {
                    std::cout << "  (" << std::fixed << std::setprecision(1)
                              << 100.0 * static_cast<double>(stats.rechunk_rebuilt) / static_cast<double>(touched)
                              << "% of new chunks)" << std::defaultfloat;
                }
                std::cout << "\n";
                std::cout << "    skipped      = " << stats.rechunk_skipped << "\n";
            }

            // End-of-life chunk-size distribution: how many of this variant's
            // arrays are currently at each size, across their whole lifetime
            // (one entry recorded per array, at its own teardown -- see
            // record_chunk_size_at_exit). Dyn-only: a static build never
            // re-chunks, so every array is always at the same compile-time
            // size and this has nothing to say.
            {
                std::lock_guard<std::mutex> lock(shrink_array::__chunk_size_dist_mutex[v]);
                const auto&                 dist = shrink_array::__chunk_size_dist[v];
                if (!dist.empty())
                {
                    std::cout << "  size dist      arrays at each chunk size, at their own teardown:\n";
                    for (const auto& [size, count] : dist)
                    {
                        std::cout << "    S=" << size << "          = " << count << "\n";
                    }
                }
            }
        }
        std::cout << "-------------------------------------\n";
    }

    // ── Retention, not counted by any of the above ───────────────────────────
    // The counters above measure EVENTS (allocations, reuses, compressions).
    // They say nothing about what is still held when the run ends, and that is
    // where a dyn build and its static twin diverged while every event counter
    // matched to within 0.05%: same alloc, same create, same still-compressed,
    // 2.8 GB apart. These two lines split that residue: buffers parked in the
    // reuse pools, and the per-(size, rank) value arenas that only exist in a
    // dynamic build.
    {
        const auto&       etc  = SingletonChk<AllTypeChunk>::getRef();
        const std::size_t pool = etc.reuse_pool_bytes();
        std::cout << "\n----- CHUNK_RETENTION -----\n";
        std::cout << "  reuse pools    = " << (pool / (std::size_t{1024} * 1024)) << " MB parked at exit\n";
    }
}

#else
/// @copydoc ParamChunk::ChunkStats
/// @note Builds without CHUNK_STATS keep no counters, so there is nothing to
///       report; the function still exists so call sites need no guard.
void ParamChunk::ChunkStats() {}
#endif

// Defined for EVERY dyn build (matching its unconditional declaration in
// param_chunk.h), with the OMP/FAST_ALLOC specifics guarded INSIDE the body.
// Outside _OPENMP && FAST_ALLOC there is nothing to register:
//   - non-OMP fast-alloc: get_chunk_allocator() routes to the single
//     allocator_data and never consults allocator_pool_;
//   - STD_ALLOC: no slab allocators at all;
//   - get_list_block_free_thr() falls back to the legacy single store when a
//     size is unregistered, so skipping register_free_list_size is safe too.
// No #ifdef CHUNK_DYNAMIC around this body either: like the two
// init_thread_structures() above, it is a TEMPLATE (see the AllT note in
// param_chunk.h) and a static build simply never instantiates it. Same
// constraint too -- SingletonChk<AllT>, never SingletonChk<AllTypeChunk>, or
// phase-1 lookup would check register_chunk_size against the concrete facade at
// definition time and reject it in a static build, where only
// AllTypeChunkDynInterface<Derived, false> is inherited.
/// @copydoc ParamChunkDynInterface<true>::register_chunk_size
template <typename AllT>
void ParamChunkDynInterface<true, AllT>::register_chunk_size([[maybe_unused]] unsigned int size)
{
    // Must run in BOTH OMP and non-OMP dynamic builds: without it a non-OMP
    // dynamic build has no per-size reuse
    // free-list slot -> SIGSEGV in drain_reuse_list_for_size at the first
    // re-chunk. The manager's register_chunk_size is compiled for every dynamic
    // build (guarded by if constexpr internally), so this is safe non-OMP.
    auto& ens_type_chunk = SingletonChk<AllT>::getRef();
    // Register this size with the per-type managers. In fast-alloc this also
    // creates the per-thread chunk allocators; in std-alloc there are no such
    // allocators, but we STILL must run this so the per-size reuse free-lists
    // (list_block_free_thr_by_size_) get a slot for `size`. Without it the
    // by-size map stays empty, every get_list_block_free_thr[_for_size] falls
    // back to the single legacy list, and buffers of DIFFERENT geometries share
    // one list: a size-A buffer parked there is later popped for a size-B chunk
    // and overflows in init_block's memset (ASan heap-buffer-overflow).
    ens_type_chunk.register_chunk_size(size, ParamChunk::block_size);
}

// The one instantiation of the pair: in a static build this is the EMPTY
// <false, AllTypeChunk>, and the body above is never compiled.
template struct ParamChunkDynInterface<build_config::chunk_dynamic, AllTypeChunk>;

// Registration hook, defined in EVERY build so `if constexpr
// (build_config::chunk_dynamic)` call sites link in both modes.
/**
 * @brief Build-agnostic wrapper over ParamChunk::register_chunk_size().
 * @param size the runtime chunk size to declare; ignored in a static build.
 * @details Defined in EVERY build so an `if constexpr (chunk_dynamic)` call
 *          site links either way, and unconditional inside: in a static build
 *          ParamChunk inherits ParamChunkDynInterface<false, AllTypeChunk>,
 *          whose register_chunk_size is a no-op -- so no `#ifdef
 *          CHUNK_DYNAMIC` here, and also no compile-time rejection of the call
 *          in a static build; see the warning on that specialisation.
 */
void register_runtime_chunk_size(unsigned int size)
{
    ParamChunk::register_chunk_size(size);
}

} // namespace shrink_array
