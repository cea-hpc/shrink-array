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

/**
 * @file chunk_stats.h
 * @brief Optional instrumentation counters (enabled with -DCHUNK_STATS). OFF
 *        by default: every increment/query below compiles to nothing, so
 *        there is no cost in a normal build.
 *
 * Design (see the redesign notes):
 *   - PER VARIANT: counters are kept separately for chunk / chunk2 / chunk2a
 *     (indexed by \<ChunkType\>::stat_variant == 0 / 1 / 2), so the
 *     end-of-run report shows one block per variant.
 *   - PER THREAD: each thread bumps its OWN slot (indexed by the current
 *     thread rank), so the hot path needs NO atomics and suffers no false
 *     sharing (each slot is cache-line aligned). A reduction over threads is
 *     done once, at display time.
 *   - Increments are TEMPLATED on the chunk type: increm_chunk_read<C>()
 *     routes to __chunk_stats[C::stat_variant][rank]. Every call site
 *     already has the chunk type to hand (Derived / ChunkType), so this is
 *     a drop-in.
 */

#include <chrono>  // __chunk_stats_t0, the rates in print_chunk_stats
#include <complex> // std::complex (ChunkStatsZeroComparable)
#include <cstdio>
#include <map>
#include <mutex>
#include <concepts>          // ChunkStatsZeroComparable
#include "chunk/PMP_CHUNK.h" // get_thr_rank_fast / CHUNK_PMP_MAX_THREADS
#include "chunk/config.h"    // build_config::chunk_dynamic

namespace shrink_array
{

/// @brief Number of chunk variants tracked: chunk (0), chunk2 (1), chunk2a (2).
inline constexpr int CHUNK_STATS_N_VARIANT = 3;

/// @brief Upper bound on threads. Generous fixed size so the counter table
///        is a plain static array (no init-order/alloc concerns). Override
///        with -DCHUNK_STATS_MAX_THR.
#ifndef CHUNK_STATS_MAX_THR
    #define CHUNK_STATS_MAX_THR 1024
#endif

/// @brief Display name for each variant, indexed by stat_variant.
inline constexpr const char* chunk_variant_name[CHUNK_STATS_N_VARIANT] = {"chunk", "chunk2", "chunk2a"};

#ifdef CHUNK_STATS

/**
 * @brief Per-(variant, thread) event counters. One cache line per slot, so
 *        concurrent writers from different threads never share a line.
 */
struct alignas(64) ChunkStatsThr
{
    long alloc           = 0;
    long write           = 0;
    long read            = 0; ///< reads served from a materialised buffer
    long read_cst        = 0; ///< reads served straight from the constant (compressed chunk)
    long read_sync       = 0; ///< accesses that had to enter the slow sync path
    long read_sync_cross = 0; ///< of those, the ones needing the cross-rank walk
    long create          = 0;
    long list            = 0;
    long bytes_alloc     = 0; ///< bytes from FRESH allocations (alloc events)
    long bytes_reused    = 0; ///< bytes handed out from the reuse list (list events)
    long bytes_freed     = 0; ///< bytes returned by a real compression (compress events)
    long teardown_clear  = 0; ///< chunks swept by clear_st at teardown (NOT a compression)
    /// chunks found at the constant when the array is destroyed (born-compressed +
    /// real compressions; see print_compression_stats)
    long still_compressed_at_exit = 0;
    long compress                 = 0; ///< successful compressions (a block was released)
    long release_park             = 0; ///< compress breakdown: buffers parked by clear_st
    long release_pool             = 0; ///< compress breakdown: buffers returned via free_block_to_pool
    long release_free             = 0; ///< compress breakdown: buffers freed via free_values
    long compress_attempt         = 0; ///< chunks scanned as compression candidates
    long mark_ok                  = 0; ///< writes that newly marked a chunk as candidate (CAS 0->1)
    long mark_refused             = 0; ///< writes refused because the chunk was ALREADY flagged
    long scan_read                = 0; ///< element reads done BY those scans (not application reads)
    long rechunk                  = 0; ///< rechunk_to() operations (geometry changes)
    long rechunk_rebuilt          = 0; ///< new chunks actually rebuilt by those rechunks
    long rechunk_skipped          = 0; ///< new chunks left compressed (nothing to rebuild)
    /// Write/compress handshake, the rare side: how often the code that pays
    /// the heavy barrier of an asymmetric Dekker would run. Each gate_start,
    /// gate_flush and gate_driver is ONE membarrier() in that design, so their
    /// rate against the run's wall time is what decides whether the design
    /// pays. Counted in every build mode; the asymmetric protocol is not there yet.
    long gate_start             = 0; ///< CompressionManager::start() entered
    long gate_start_flush_yield = 0; ///< ... of which yielded to a flush (either check)
    long gate_start_yield_late  = 0; ///< ... of those, at the re-check AFTER the writer wait (hard held, writers deferred for nothing)
    long gate_start_deferred    = 0; ///< ... of which gave up after the writer-wait budget
    long gate_flush             = 0; ///< announce_flush() calls (flush_deferred, flush_chunk_all_ranks)
    long gate_deferred          = 0; ///< writes deferred (enqueue_deferred_write): a compressor was in flight
    long gate_wait              = 0; ///< wait_out_compressor() calls (a writer found the gate shut)
    long gate_mark_hit          = 0; ///< ... of which because ITS chunk was marked by a compressor's pass
    long gate_wait_yields       = 0; ///< ... yields spent there (one per 1000 pauses)
    long gate_driver            = 0; ///< run_slow_path() drivers (a writer sent to the slow path)
};

/// @brief Program start, for the per-second rates print_chunk_stats() shows
///        next to the gate counters. Static init: as early as the counters.
inline const std::chrono::steady_clock::time_point __chunk_stats_t0 = std::chrono::steady_clock::now();

/// @brief [variant][thread] counter table. Defined once in chunk_std.cpp.
extern ChunkStatsThr __chunk_stats[CHUNK_STATS_N_VARIANT][CHUNK_STATS_MAX_THR];

/**
 * @brief Current thread's counter slot for a variant.
 * @param variant Chunk variant index (see stat_variant / chunk_variant_name).
 * @return Reference to this thread's slot. Bounds-guarded: a rank past the
 *         table falls back to slot 0 rather than corrupting memory.
 *
 * The rank is the cached one (get_thr_rank_fast), as on the write path: an
 * increment runs on every read and write, and the raw CHUNK_PMP_THREAD_RANK
 * is an omp_get_thread_num() call through the PLT plus a __tls_get_addr in
 * libomp -- 10% of a stats-build profile at 8 threads, charged to whatever
 * the counters were meant to measure.
 */
inline ChunkStatsThr& __chunk_stats_slot(int variant) noexcept
{
    int r = get_thr_rank_fast();
    if (r < 0 || r >= CHUNK_STATS_MAX_THR)
        r = 0;
    return __chunk_stats[variant][r];
}

// ---- increments (templated on the chunk type; no atomics) ------------------

/// @brief Count one fresh allocation.
template <typename C>
inline void increm_chunk_alloc()
{
    __chunk_stats_slot(C::stat_variant).alloc++;
}
/// @brief Count one read served from a materialised buffer.
template <typename C>
void increm_chunk_read()
{
    __chunk_stats_slot(C::stat_variant).read++;
}
/// @brief Count one read served straight from the constant (compressed chunk).
template <typename C>
void increm_chunk_read_cst()
{
    __chunk_stats_slot(C::stat_variant).read_cst++;
}
/// @brief Count one access that had to enter the slow sync path.
template <typename C>
void increm_chunk_read_sync()
{
    __chunk_stats_slot(C::stat_variant).read_sync++;
}
/// @brief Count one of those slow-sync accesses that needed the cross-rank walk.
template <typename C>
void increm_chunk_read_sync_cross()
{
    __chunk_stats_slot(C::stat_variant).read_sync_cross++;
}
/// @brief Count one chunk creation.
template <typename C>
void increm_chunk_create()
{
    __chunk_stats_slot(C::stat_variant).create++;
}
/// @brief Count one block handed out from the reuse list.
template <typename C>
void increm_chunk_list()
{
    __chunk_stats_slot(C::stat_variant).list++;
}
/// @brief Add `b` bytes from a fresh allocation (alloc event).
/// @param b Bytes allocated.
template <typename C>
void add_chunk_bytes_alloc(long b)
{
    __chunk_stats_slot(C::stat_variant).bytes_alloc += b;
}
/// @brief Add `b` bytes handed out from the reuse list (list event).
/// @param b Bytes reused.
template <typename C>
void add_chunk_bytes_reused(long b)
{
    __chunk_stats_slot(C::stat_variant).bytes_reused += b;
}
/// @brief Add `b` bytes returned by a real compression (compress event).
/// @param b Bytes freed.
template <typename C>
void add_chunk_bytes_freed(long b)
{
    __chunk_stats_slot(C::stat_variant).bytes_freed += b;
}
/// @brief Count one chunk swept by clear_st at teardown (NOT a compression).
template <typename C>
void increm_chunk_teardown_clear()
{
    __chunk_stats_slot(C::stat_variant).teardown_clear++;
}
/// @brief Add `nb` chunks found at the constant when the array is destroyed.
/// @param nb Number of still-compressed chunks found.
template <typename C>
void add_chunk_still_compressed_at_exit(long nb)
{
    __chunk_stats_slot(C::stat_variant).still_compressed_at_exit += nb;
}
/// @brief Count one successful compression (a block was released).
template <typename C>
void increm_chunk_compress()
{
    __chunk_stats_slot(C::stat_variant).compress++;
}
/// @brief Compress breakdown: count one buffer parked by clear_st.
template <typename C>
void increm_chunk_release_park()
{
    __chunk_stats_slot(C::stat_variant).release_park++;
}
/// @brief Compress breakdown: count one buffer returned via free_block_to_pool.
template <typename C>
void increm_chunk_release_pool()
{
    __chunk_stats_slot(C::stat_variant).release_pool++;
}
/// @brief Compress breakdown: count one buffer freed via free_values.
template <typename C>
void increm_chunk_release_free()
{
    __chunk_stats_slot(C::stat_variant).release_free++;
}
/// @brief Count one chunk scanned as a compression candidate.
template <typename C>
void increm_chunk_compress_attempt()
{
    __chunk_stats_slot(C::stat_variant).compress_attempt++;
}
/// @brief Count one write that newly marked a chunk as a candidate (CAS 0->1).
template <typename C>
void increm_chunk_mark_ok()
{
    __chunk_stats_slot(C::stat_variant).mark_ok++;
}
/// @brief Count one write refused because the chunk was ALREADY flagged.
template <typename C>
void increm_chunk_mark_refused()
{
    __chunk_stats_slot(C::stat_variant).mark_refused++;
}
/// @brief Count one CompressionManager::start() entered.
template <typename C>
void increm_chunk_gate_start()
{
    __chunk_stats_slot(C::stat_variant).gate_start++;
}
/// @brief Count one start() that yielded to a flush.
template <typename C>
void increm_chunk_gate_start_flush_yield()
{
    __chunk_stats_slot(C::stat_variant).gate_start_flush_yield++;
}
/// @brief Count one start() that yielded at the re-check, after the writer wait.
template <typename C>
void increm_chunk_gate_start_yield_late()
{
    __chunk_stats_slot(C::stat_variant).gate_start_yield_late++;
}
/// @brief Count one start() that gave up after the writer-wait budget.
template <typename C>
void increm_chunk_gate_start_deferred()
{
    __chunk_stats_slot(C::stat_variant).gate_start_deferred++;
}
/// @brief Count one announce_flush().
template <typename C>
void increm_chunk_gate_flush()
{
    __chunk_stats_slot(C::stat_variant).gate_flush++;
}
/// @brief Count one run_slow_path() driver.
template <typename C>
void increm_chunk_gate_driver()
{
    __chunk_stats_slot(C::stat_variant).gate_driver++;
}
/// @brief Count one write deferred to a per-rank queue.
template <typename C>
void increm_chunk_gate_deferred()
{
    __chunk_stats_slot(C::stat_variant).gate_deferred++;
}
/// @brief Count one writer-side wait for a compressor, and the yields it spent.
template <typename C>
void increm_chunk_gate_wait(int yields, bool mark_hit)
{
    auto& s = __chunk_stats_slot(C::stat_variant);
    s.gate_wait++;
    s.gate_wait_yields += yields;
    if (mark_hit)
        s.gate_mark_hit++;
}
/// @brief Count one element read done BY a compression-candidate scan (not an application read).
template <typename C>
void increm_chunk_scan_read()
{
    __chunk_stats_slot(C::stat_variant).scan_read++;
}
/// @brief Count @p n element reads done by one scan (ChunkBase::scan_constants and friends).
template <typename C>
void add_chunk_scan_read(long n)
{
    __chunk_stats_slot(C::stat_variant).scan_read += n;
}
/// @brief Count one rechunk_to() operation (geometry change).
template <typename C>
void increm_chunk_rechunk()
{
    __chunk_stats_slot(C::stat_variant).rechunk++;
}
/// @brief Count one new chunk actually rebuilt by a rechunk.
template <typename C>
void increm_chunk_rechunk_rebuilt()
{
    __chunk_stats_slot(C::stat_variant).rechunk_rebuilt++;
}
/// @brief Count one new chunk left compressed by a rechunk (nothing to rebuild).
template <typename C>
void increm_chunk_rechunk_skipped()
{
    __chunk_stats_slot(C::stat_variant).rechunk_skipped++;
}

/// @brief Count one write, unconditionally (see the value-aware overload below).
template <typename C>
void increm_chunk_write()
{
    __chunk_stats_slot(C::stat_variant).write++;
}

/**
 * @brief Whether V can be compared to a zero-initialised V, so
 *        increm_chunk_write(V) can skip counting a write that is itself a
 *        compressible zero.
 * @details Element types differ widely here: plain arithmetic types compare
 *          to 0, std::complex needs a complex zero (no
 *          operator!=(complex,int)), and V can even be a Proxy
 *          (proxy-to-proxy assign), which is neither default-constructible
 *          nor comparable to 0.
 * @tparam V Type being written.
 */
template <typename V>
concept ChunkStatsZeroComparable = requires(const V& a) {
    { a != V(0) } -> std::convertible_to<bool>;
};

/**
 * @brief Count a write, skipping values that are zero (the library
 *        compresses zeros, so a zero write does not materialise anything).
 *        The zero test is only applied when it is well-formed for V (see
 *        ChunkStatsZeroComparable); anything we cannot test is simply counted.
 * @tparam C Chunk type (selects the variant slot).
 * @tparam V Type of the value being written.
 * @param v Value being written.
 */
template <typename C, typename V>
void increm_chunk_write(V v)
{
    if constexpr (ChunkStatsZeroComparable<V>)
    {
        if (v != V(0))
            __chunk_stats_slot(C::stat_variant).write++;
    }
    else
    {
        __chunk_stats_slot(C::stat_variant).write++;
    }
}

// batch variants (a clear/compress sweep touches nb blocks at once)

/// @brief Add `nb` chunks swept by clear_st at teardown, in one batch.
/// @param nb Number of chunks swept.
template <typename C>
void add_chunk_teardown_clear(long nb)
{
    __chunk_stats_slot(C::stat_variant).teardown_clear += nb;
}
/// @brief Add `nb` successful compressions, in one batch (a sweep touching
///        several blocks at once).
/// @param nb Number of chunks compressed.
template <typename C>
void add_chunk_compress(long nb)
{
    __chunk_stats_slot(C::stat_variant).compress += nb;
}

// ---- reduction / lifecycle -------------------------------------------------

/// @brief Reset every (variant, thread) counter slot to zero.
inline void init_all_stats() noexcept
{
    for (int v = 0; v < CHUNK_STATS_N_VARIANT; ++v)
        for (int t = 0; t < CHUNK_STATS_MAX_THR; ++t)
            __chunk_stats[v][t] = ChunkStatsThr{};
}

/**
 * @brief Dyn-only, end-of-life state: how many arrays are currently at each
 *        chunk size, per variant ([chunk size] -> [array count]).
 * @details Unlike the event counters above (per-thread, lock-free), this is
 *          read/written only when an array is destroyed (rare: at most a
 *          handful of times per run for this bench), so a mutex-protected
 *          map is fine -- no hot-path contention. Static builds
 *          (chunk_dynamic == false) never touch this: every array has the
 *          same, fixed, compile-time size, so a distribution has nothing to
 *          say.
 */
inline std::mutex __chunk_size_dist_mutex[CHUNK_STATS_N_VARIANT];
/// @brief See __chunk_size_dist_mutex.
inline std::map<unsigned int, long> __chunk_size_dist[CHUNK_STATS_N_VARIANT];

/**
 * @brief Record one array's current chunk size in the end-of-life size
 *        distribution (__chunk_size_dist). No-op on static builds or when
 *        stats are disabled.
 * @tparam C Chunk type (selects the variant slot).
 * @param size The array's current chunk size.
 */
template <typename C>
inline void record_chunk_size_at_exit(unsigned int size) noexcept
{
    if constexpr (build_config::chunk_dynamic && chunk_config::stats)
    {
        std::lock_guard<std::mutex> lock(__chunk_size_dist_mutex[C::stat_variant]);
        ++__chunk_size_dist[C::stat_variant][size];
    }
}

/**
 * @brief Totals for one variant, summed over all threads.
 * @param variant Chunk variant index (see stat_variant / chunk_variant_name).
 * @return A ChunkStatsThr holding the sum of every thread's slot for that variant.
 */
inline ChunkStatsThr reduce_chunk_stats(int variant) noexcept
{
    ChunkStatsThr acc{};
    const int     nthr = CHUNK_PMP_MAX_THREADS < CHUNK_STATS_MAX_THR ? CHUNK_PMP_MAX_THREADS : CHUNK_STATS_MAX_THR;
    for (int t = 0; t < nthr; ++t)
    {
        const ChunkStatsThr& s = __chunk_stats[variant][t];
        acc.alloc += s.alloc;
        acc.write += s.write;
        acc.read += s.read;
        acc.read_cst += s.read_cst;
        acc.read_sync += s.read_sync;
        acc.read_sync_cross += s.read_sync_cross;
        acc.create += s.create;
        acc.list += s.list;
        acc.bytes_alloc += s.bytes_alloc;
        acc.bytes_reused += s.bytes_reused;
        acc.bytes_freed += s.bytes_freed;
        acc.teardown_clear += s.teardown_clear;
        acc.still_compressed_at_exit += s.still_compressed_at_exit;
        acc.compress += s.compress;
        acc.release_park += s.release_park;
        acc.release_pool += s.release_pool;
        acc.release_free += s.release_free;
        acc.compress_attempt += s.compress_attempt;
        acc.mark_ok += s.mark_ok;
        acc.mark_refused += s.mark_refused;
        acc.scan_read += s.scan_read;
        acc.rechunk += s.rechunk;
        acc.rechunk_rebuilt += s.rechunk_rebuilt;
        acc.rechunk_skipped += s.rechunk_skipped;
        acc.gate_start += s.gate_start;
        acc.gate_start_flush_yield += s.gate_start_flush_yield;
        acc.gate_start_yield_late += s.gate_start_yield_late;
        acc.gate_start_deferred += s.gate_start_deferred;
        acc.gate_flush += s.gate_flush;
        acc.gate_driver += s.gate_driver;
        acc.gate_deferred += s.gate_deferred;
        acc.gate_wait += s.gate_wait;
        acc.gate_wait_yields += s.gate_wait_yields;
        acc.gate_mark_hit += s.gate_mark_hit;
    }
    return acc;
}

// record_chunk_size_at_exit() and reduce_chunk_stats() have no no-op mirror
// below on purpose, not by omission: every call to the former is already
// wrapped in `if constexpr (chunk_config::stats)` at its 3 call sites (see
// array_chunk*_inline.h), so it is simply never instantiated when stats are
// off; and the latter returns ChunkStatsThr by value, a type that itself
// only exists in this #ifdef branch, so its only caller (param_chunk.cpp) is
// necessarily already gated behind its own `#ifdef CHUNK_STATS`.

#else // !CHUNK_STATS

/**
 * @brief No-op mirrors of every counter function above, active when
 *        CHUNK_STATS is undefined: same names and signatures, empty bodies,
 *        so every call site compiles to nothing at zero cost.
 */
template <typename C>
void increm_chunk_alloc()
{
}
template <typename C>
void increm_chunk_read()
{
}
template <typename C>
void increm_chunk_read_cst()
{
}
template <typename C>
void increm_chunk_read_sync()
{
}
template <typename C>
void increm_chunk_read_sync_cross()
{
}
template <typename C>
void increm_chunk_create()
{
}
template <typename C>
void increm_chunk_list()
{
}
template <typename C>
void add_chunk_bytes_alloc(long)
{
}
template <typename C>
void add_chunk_bytes_reused(long)
{
}
template <typename C>
void add_chunk_bytes_freed(long)
{
}
template <typename C>
void increm_chunk_teardown_clear()
{
}
template <typename C>
void add_chunk_still_compressed_at_exit(long)
{
}
template <typename C>
void increm_chunk_compress()
{
}
template <typename C>
void increm_chunk_release_park()
{
}
template <typename C>
void increm_chunk_release_pool()
{
}
template <typename C>
void increm_chunk_release_free()
{
}
template <typename C>
void increm_chunk_compress_attempt()
{
}
template <typename C>
void increm_chunk_mark_ok()
{
}
template <typename C>
void increm_chunk_mark_refused()
{
}
template <typename C>
void increm_chunk_scan_read()
{
}
template <typename C>
void increm_chunk_gate_start()
{
}
template <typename C>
void increm_chunk_gate_start_flush_yield()
{
}
template <typename C>
void increm_chunk_gate_start_yield_late()
{
}
template <typename C>
void increm_chunk_gate_start_deferred()
{
}
template <typename C>
void increm_chunk_gate_flush()
{
}
template <typename C>
void increm_chunk_gate_driver()
{
}
template <typename C>
void increm_chunk_gate_deferred()
{
}
template <typename C>
void increm_chunk_gate_wait(int, bool)
{
}
template <typename C>
void add_chunk_scan_read(long)
{
}
template <typename C>
void increm_chunk_rechunk()
{
}
template <typename C>
void increm_chunk_rechunk_rebuilt()
{
}
template <typename C>
void increm_chunk_rechunk_skipped()
{
}
template <typename C>
void increm_chunk_write()
{
}
template <typename C, typename V>
void increm_chunk_write(V)
{
}
template <typename C>
void add_chunk_teardown_clear(long)
{
}
template <typename C>
void add_chunk_compress(long)
{
}
inline void init_all_stats() noexcept {}

#endif

} // namespace shrink_array
