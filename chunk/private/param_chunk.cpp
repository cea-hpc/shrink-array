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
 *  - **Startup**: init_indexes() sizes and publishes the allocation block;
 *    init_thread_structures() lives on ParamChunkParallelInterface, one body
 *    per thread model.
 *  - **The geometry registration hook**, for dynamic builds.
 *  - **chunk_stats()**, the instrumentation report (presentation only).
 *
 * @note Sequential context only: nothing here may be called in a parallel region.
 */

#include <algorithm> // std::max
#include <chrono>    // steady_clock, the gate rates (CHUNK_STATS report only)
#include <cstddef>   // std::size_t (CHUNK_STATS report only: the retention)
#include <iomanip>   // std::setprecision (CHUNK_STATS report only: the ratios)
#include <iostream>  // std::cout
#include <mutex>     // std::scoped_lock (CHUNK_STATS report only: the size distribution)
#include <utility>   // std::cmp_greater
#include <unistd.h>  // sysconf(_SC_PAGESIZE): page-aware slab floor

#include "chunk/config.h"                              // build_config::chunk_dynamic, use_openmp
#include "chunk/singleton.h"                           // SingletonChk
#include "chunk/internal/chunk_size_config.h"          // static_chunk_size
#include "chunk/internal/chunk_size_policy.h"          // active_chunk_size
#include "chunk/internal/chunk_stats.h"                // per-variant stats reduce/print
#include "chunk/param_chunk.h"                         // the declarations defined here
#include "chunk/PMP_CHUNK.h"                           // CHUNK_PMP_MAX_THREADS
#include "chunk/internal/write_compress_sync_policy.h" // dekker::protocol_name (CHUNK_STATS report only)
#include "chunk/all_type_chunk.h"                      // AllTypeChunk
#include "chunk/log.h"                                 // CHUNK_LOG

namespace shrink_array
{

/// @brief Allocation block (slab) size in bytes, computed once by init_indexes().
int ParamChunk::block_size;
// No #ifdef _OPENMP: the bodies are TEMPLATES on AllT (see param_chunk.h), and
// only the one this build instantiates below is compiled. Use
// SingletonChk<AllT>, never SingletonChk<AllTypeChunk>: the calls must stay
// dependent, or phase-1 lookup rejects the other build's body.

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

// The one instantiation: the other thread model's body stays uninstantiated
// (as AllTypeChunkParallelInterface in all_type_chunk.cpp).
template struct ParamChunkParallelInterface<build_config::use_openmp, AllTypeChunk>;

/**
 * @copydoc ParamChunk::init_indexes
 *
 * @details Block size = 2 MiB (15 MiB with @p big_chunk_block) divided by the
 *          thread count, floored at 240 KiB and at sixteen pages.
 *
 * @warning A chunk larger than one block cannot be allocated; only a warning,
 *          as the caller cannot fix the compile-time geometry at run time.
 */
void ParamChunk::init_indexes(bool big_chunk_block)
{
    auto& ens_type_chunk = SingletonChk<AllTypeChunk>::get_ref();

    int sz = 1024 * 1024 * 2;
    if (big_chunk_block)
    {
        sz = 1024 * 1024 * 15;
    }

    constexpr int sz_min = 1024 * 16 * 15;

    sz /= CHUNK_PMP_MAX_THREADS; // taking into account the number of threads

    sz = std::max(sz_min, sz);
    // Page-aware floor. An emptied slab keeps its last page (trailing
    // back-pointer), so with few pages per slab the retained share is large
    // (a quarter with 256 KiB slabs on 64 KiB pages). Sixteen pages cap it at
    // ~6%; the cost is one partly-filled slab per (rank, array, size).
    if (const long ps = ::sysconf(_SC_PAGESIZE); ps > 0)
    {
        sz = std::max(sz, static_cast<int>(16 * ps));
    }

    set_block_size(sz);

    if (std::cmp_greater(static_chunk_size, sz))
    {
        CHUNK_LOG(warn, general, "chunk sizes can't be larger than one page");
    }

    // Per-type structures are built when each type registers its manager (first
    // array, or init_indexes<Types>()), with the block size published here.
    init_thread_structures(); // the allocator banner

    if constexpr (chunk_config::stats)
    {
        init_all_stats();
    }

    ens_type_chunk.mark_ready(block_size);

    // Register the STARTING geometry, now that the managers exist. It is set
    // statically (chunk_size_policy.h), never through set_active_chunk_size(),
    // so without this the per-size free-list map stays empty: every buffer goes
    // to the geometry-blind legacy store, which drain_reuse_list_for_size()
    // never reaches (a large share of the footprint stays parked). Idempotent.
    if constexpr (build_config::chunk_dynamic)
    {
        register_runtime_chunk_size(active_chunk_size());
    }
}

/// @copydoc ParamChunk::force_compression
void ParamChunk::force_compression()
{
    // Declared in every build; a no-op without fast block compression.
    if constexpr (chunk_config::use_fast_block_compression)
    {
        const auto& ens_type_chunk = SingletonChk<AllTypeChunk>::get_ref();

        ens_type_chunk.force_chunk_to_compress();
    }
}

#if CHUNK_OPT_STATS

/**
 * @copydoc ParamChunk::chunk_stats
 *
 * @details Two passes: the first reduces the totals so that each legend line
 *          is printed only when some counter will stand behind it.
 *
 * @note Presentation only: the counters are maintained elsewhere.
 */
void ParamChunk::chunk_stats()
{
    // Totals first, to filter the legend; then one block per active variant.
    ChunkStatsThr all{};
    for (int v = 0; v < chunk_stats_n_variant; ++v)
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
    // Unconditional: the per-variant block prints `bytes` under both alloc and
    // list, and bytes_alloc can be non-zero on a run that never reused anything.
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

    for (int v = 0; v < chunk_stats_n_variant; ++v)
    {
        const ChunkStatsThr stats = reduce_chunk_stats(v);
        // Skip a variant with no activity at all.
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
        // Writes into an existing buffer vs writes that materialised first.
        // Only writes materialise, so write_materialize is exactly `create`.
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
        // Read hit rate: share served straight from the constant.
        if (reads_all != 0)
        {
            const double hit = 100.0 * static_cast<double>(stats.read_cst) / static_cast<double>(reads_all);
            std::cout << "    constant     = " << stats.read_cst << "  (" << hit << "% hit)\n";
            std::cout << "    materialised = " << stats.read << "  (" << (100.0 - hit) << "%)\n";
        }
        // Reads that entered the slow sync path, and of those the cross-rank ones.
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
        // mark_ok: flag 0 -> 1, chunk queued; mark_refused: already 1. Refusals
        // near 100% with a flat mark_ok mean the flag is never reset after the
        // sweep, and compression silently stops for those chunks.
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
        // `compress (during exec)`: real compressions of a materialised chunk
        // during the run. `still compressed (at exit)`: chunks at the constant at
        // destruction, never-written ones included. They may differ by orders
        // of magnitude on a run with few writes; that is not a counting bug.
        std::cout << "  compress (during exec) = " << stats.compress << "\n";
        if (stats.bytes_freed != 0)
        {
            std::cout << "    bytes        = " << (stats.bytes_freed / (1024L * 1024)) << " MB\n";
        }
        if (stats.still_compressed_at_exit != 0)
        {
            std::cout << "  still compressed (at exit) = " << stats.still_compressed_at_exit << "\n";
        }
        // Scan yield and cost (scan_read is kept out of `read`, which counts the
        // application's own accesses).
        if (stats.compress_attempt != 0)
        {
            std::cout << "    attempts     = " << stats.compress_attempt << "  (" << std::fixed << std::setprecision(1)
                      << 100.0 * static_cast<double>(stats.compress) / static_cast<double>(stats.compress_attempt)
                      << "% succeed)\n"
                      << std::defaultfloat;
            std::cout << "    scan reads   = " << stats.scan_read << "\n";
        }
        // teardown_clear: end-of-life clear_st sweep, NOT a compression.
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
        // Dynamic builds only: a static geometry never re-chunks.
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

            // Chunk size of each array at its own teardown (record_chunk_size_at_exit).
            {
                const std::scoped_lock lock(shrink_array::__chunk_size_dist_mutex[v]);
                const auto&            dist = shrink_array::__chunk_size_dist[v];
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
    // The counters above measure EVENTS; what is still held at the end is
    // separate, and can differ widely between builds whose event counters
    // match: here, the buffers parked in the reuse pools.
    {
        const auto&       etc  = SingletonChk<AllTypeChunk>::get_ref();
        const std::size_t pool = etc.reuse_pool_bytes();
        std::cout << "\n----- CHUNK_RETENTION -----\n";
        std::cout << "  reuse pools    = " << (pool / (std::size_t{1024} * 1024)) << " MB parked at exit\n";
    }
}

#else
/// @copydoc ParamChunk::chunk_stats
/// @note No counters without CHUNK_STATS; kept so call sites need no guard.
void ParamChunk::chunk_stats() {}
#endif

// Defined for every dynamic build; no #ifdef CHUNK_DYNAMIC: a TEMPLATE on AllT,
// never instantiated by a static build. Same rule as init_thread_structures():
// SingletonChk<AllT>, never SingletonChk<AllTypeChunk>, or phase-1 lookup
// rejects it in a static build.
/// @copydoc ParamChunkDynInterface<true>::register_chunk_size
template <typename AllT>
void ParamChunkDynInterface<true, AllT>::register_chunk_size([[maybe_unused]] unsigned int size)
{
    // Must run in every dynamic build, OpenMP or not, fast-alloc or std-alloc:
    // besides the per-thread allocators (fast-alloc), it gives the per-size
    // reuse free-lists (list_block_free_thr_by_size_) a slot for `size`.
    // Without it, buffers of DIFFERENT geometries share the legacy list and a
    // smaller buffer is popped for a larger chunk (overflow in init_block), or
    // drain_reuse_list_for_size finds no slot at the first re-chunk.
    auto& ens_type_chunk = SingletonChk<AllT>::get_ref();
    ens_type_chunk.register_chunk_size(size, ParamChunk::block_size);
}

// The one instantiation of the pair: in a static build this is the EMPTY
// <false, AllTypeChunk>, and the body above is never compiled.
template struct ParamChunkDynInterface<build_config::chunk_dynamic, AllTypeChunk>;

/**
 * @brief Build-agnostic wrapper over ParamChunk::register_chunk_size().
 * @param size the runtime chunk size to declare; ignored in a static build.
 * @details Defined in EVERY build so an `if constexpr (chunk_dynamic)` call
 *          site links either way; in a static build the call is the no-op of
 *          ParamChunkDynInterface<false, AllTypeChunk>.
 */
void register_runtime_chunk_size(unsigned int size)
{
    ParamChunk::register_chunk_size(size);
}

} // namespace shrink_array
