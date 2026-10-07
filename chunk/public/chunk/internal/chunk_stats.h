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
 *     (indexed by \<ChunkType\>\::stat_variant == 0 / 1 / 2), so the
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

// All of these serve the CHUNK_STATS half of the file only.
#include <array>    // std::array
#include <chrono>   // __chunk_stats_t0, the rates ParamChunk::ChunkStats() prints
#include <map>      // std::map
#include <mutex>    // std::mutex, std::lock_guard
#include <concepts> // ChunkStatsZeroComparable

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
    /// buffers released WITHOUT a compression (a re-chunk's old storage, the
    /// teardown sweeps); their bytes are in bytes_freed too, so the live
    /// figures (alloc + reused - freed, create - compress - released) stay
    /// right across re-chunks
    long released = 0;
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
    long gate_start_yield_late =
        0; ///< ... of those, at the re-check AFTER the writer wait (hard held, writers deferred for nothing)
    long gate_start_deferred = 0; ///< ... of which gave up after the writer-wait budget
    long gate_flush          = 0; ///< announce_flush() calls (flush_deferred, flush_chunk_all_ranks)
    long gate_deferred       = 0; ///< writes deferred (enqueue_deferred_write): a compressor was in flight
    long gate_wait           = 0; ///< wait_out_compressor() calls (a writer found the gate shut)
    long gate_mark_hit       = 0; ///< ... of which because ITS chunk was marked by a compressor's pass
    long gate_wait_yields    = 0; ///< ... yields spent there (one per 1000 pauses)
    long gate_driver         = 0; ///< run_slow_path() drivers (a writer sent to the slow path)
};

/// @brief Program start, for the per-second rates ParamChunk::ChunkStats() shows
///        next to the gate counters. Static init: as early as the counters.
inline const std::chrono::steady_clock::time_point __chunk_stats_t0 = std::chrono::steady_clock::now();

/// @brief [variant][thread] counter table. Defined once in chunk_std.cpp.
extern std::array<std::array<ChunkStatsThr, CHUNK_STATS_MAX_THR>, CHUNK_STATS_N_VARIANT> __chunk_stats;

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
    int rank = get_thr_rank_fast();
    if (rank < 0 || rank >= CHUNK_STATS_MAX_THR)
    {
        rank = 0;
    }
    return __chunk_stats[variant][rank];
}

// ---- increments (templated on the chunk type; no atomics) ------------------

/// @brief Count one fresh allocation.
template <typename C>
inline void increm_chunk_alloc() noexcept
{
    __chunk_stats_slot(C::stat_variant).alloc++;
}
/// @brief Count one read served from a materialised buffer.
template <typename C>
void increm_chunk_read() noexcept
{
    __chunk_stats_slot(C::stat_variant).read++;
}
/// @brief Count one read served straight from the constant (compressed chunk).
template <typename C>
void increm_chunk_read_cst() noexcept
{
    __chunk_stats_slot(C::stat_variant).read_cst++;
}
/// @brief Count one access that had to enter the slow sync path.
template <typename C>
void increm_chunk_read_sync() noexcept
{
    __chunk_stats_slot(C::stat_variant).read_sync++;
}
/// @brief Count one of those slow-sync accesses that needed the cross-rank walk.
template <typename C>
void increm_chunk_read_sync_cross() noexcept
{
    __chunk_stats_slot(C::stat_variant).read_sync_cross++;
}
/// @brief Count one chunk creation.
template <typename C>
void increm_chunk_create() noexcept
{
    __chunk_stats_slot(C::stat_variant).create++;
}
/// @brief Count one block handed out from the reuse list.
template <typename C>
void increm_chunk_list() noexcept
{
    __chunk_stats_slot(C::stat_variant).list++;
}
/// @brief Add @p bytes bytes from a fresh allocation (alloc event).
/// @param bytes Bytes allocated.
template <typename C>
void add_chunk_bytes_alloc(long bytes) noexcept
{
    __chunk_stats_slot(C::stat_variant).bytes_alloc += bytes;
}
/// @brief Add @p bytes bytes handed out from the reuse list (list event).
/// @param bytes Bytes reused.
template <typename C>
void add_chunk_bytes_reused(long bytes) noexcept
{
    __chunk_stats_slot(C::stat_variant).bytes_reused += bytes;
}
/// @brief Add @p bytes bytes returned by a real compression (compress event).
/// @param bytes Bytes freed.
template <typename C>
void add_chunk_bytes_freed(long bytes) noexcept
{
    __chunk_stats_slot(C::stat_variant).bytes_freed += bytes;
}
/// @brief Count one chunk swept by clear_st at teardown (NOT a compression).
template <typename C>
void increm_chunk_teardown_clear() noexcept
{
    __chunk_stats_slot(C::stat_variant).teardown_clear++;
}
/// @brief Count @p n buffers released WITHOUT a compression, @p bytes in all.
/// @param n     Buffers released.
/// @param bytes Their total size in bytes.
template <typename C>
void add_chunk_released(long n, long bytes) noexcept
{
    __chunk_stats_slot(C::stat_variant).released += n;
    __chunk_stats_slot(C::stat_variant).bytes_freed += bytes;
}
/// @brief Add `nb` chunks found at the constant when the array is destroyed.
/// @param nb Number of still-compressed chunks found.
template <typename C>
void add_chunk_still_compressed_at_exit(long nb) noexcept
{
    __chunk_stats_slot(C::stat_variant).still_compressed_at_exit += nb;
}
/// @brief Count one successful compression (a block was released).
template <typename C>
void increm_chunk_compress() noexcept
{
    __chunk_stats_slot(C::stat_variant).compress++;
}
/// @brief Compress breakdown: count one buffer parked by clear_st.
template <typename C>
void increm_chunk_release_park() noexcept
{
    __chunk_stats_slot(C::stat_variant).release_park++;
}
/// @brief Compress breakdown: count one buffer returned via free_block_to_pool.
template <typename C>
void increm_chunk_release_pool() noexcept
{
    __chunk_stats_slot(C::stat_variant).release_pool++;
}
/// @brief Compress breakdown: count one buffer freed via free_values.
template <typename C>
void increm_chunk_release_free() noexcept
{
    __chunk_stats_slot(C::stat_variant).release_free++;
}
/// @brief Count one chunk scanned as a compression candidate.
template <typename C>
void increm_chunk_compress_attempt() noexcept
{
    __chunk_stats_slot(C::stat_variant).compress_attempt++;
}
/// @brief Count one write that newly marked a chunk as a candidate (CAS 0->1).
template <typename C>
void increm_chunk_mark_ok() noexcept
{
    __chunk_stats_slot(C::stat_variant).mark_ok++;
}
/// @brief Count one write refused because the chunk was ALREADY flagged.
template <typename C>
void increm_chunk_mark_refused() noexcept
{
    __chunk_stats_slot(C::stat_variant).mark_refused++;
}
/// @brief Count one CompressionManager::start() entered.
template <typename C>
void increm_chunk_gate_start() noexcept
{
    __chunk_stats_slot(C::stat_variant).gate_start++;
}
/// @brief Count one start() that yielded to a flush.
template <typename C>
void increm_chunk_gate_start_flush_yield() noexcept
{
    __chunk_stats_slot(C::stat_variant).gate_start_flush_yield++;
}
/// @brief Count one start() that yielded at the re-check, after the writer wait.
template <typename C>
void increm_chunk_gate_start_yield_late() noexcept
{
    __chunk_stats_slot(C::stat_variant).gate_start_yield_late++;
}
/// @brief Count one start() that gave up after the writer-wait budget.
template <typename C>
void increm_chunk_gate_start_deferred() noexcept
{
    __chunk_stats_slot(C::stat_variant).gate_start_deferred++;
}
/// @brief Count one announce_flush().
template <typename C>
void increm_chunk_gate_flush() noexcept
{
    __chunk_stats_slot(C::stat_variant).gate_flush++;
}
/// @brief Count one run_slow_path() driver.
template <typename C>
void increm_chunk_gate_driver() noexcept
{
    __chunk_stats_slot(C::stat_variant).gate_driver++;
}
/// @brief Count one write deferred to a per-rank queue.
template <typename C>
void increm_chunk_gate_deferred() noexcept
{
    __chunk_stats_slot(C::stat_variant).gate_deferred++;
}
/// @brief Count one writer-side wait for a compressor, and the yields it spent.
template <typename C>
void increm_chunk_gate_wait(int yields, bool mark_hit) noexcept
{
    auto& slot = __chunk_stats_slot(C::stat_variant);
    slot.gate_wait++;
    slot.gate_wait_yields += yields;
    if (mark_hit)
    {
        slot.gate_mark_hit++;
    }
}
/// @brief Count one element read done BY a compression-candidate scan (not an application read).
template <typename C>
void increm_chunk_scan_read() noexcept
{
    __chunk_stats_slot(C::stat_variant).scan_read++;
}
/// @brief Count @p count element reads done by one scan (ChunkBase::scan_constants and friends).
template <typename C>
void add_chunk_scan_read(long count) noexcept
{
    __chunk_stats_slot(C::stat_variant).scan_read += count;
}
/// @brief Count one rechunk_to() operation (geometry change).
template <typename C>
void increm_chunk_rechunk() noexcept
{
    __chunk_stats_slot(C::stat_variant).rechunk++;
}
/// @brief Count one new chunk actually rebuilt by a rechunk.
template <typename C>
void increm_chunk_rechunk_rebuilt() noexcept
{
    __chunk_stats_slot(C::stat_variant).rechunk_rebuilt++;
}
/// @brief Count one new chunk left compressed by a rechunk (nothing to rebuild).
template <typename C>
void increm_chunk_rechunk_skipped() noexcept
{
    __chunk_stats_slot(C::stat_variant).rechunk_skipped++;
}

/// @brief Count one write, unconditionally (see the value-aware overload below).
template <typename C>
void increm_chunk_write() noexcept
{
    __chunk_stats_slot(C::stat_variant).write++;
}

/**
 * @brief Whether V can be compared to a zero-initialised V, so
 *        increm_chunk_write(V) can skip counting a write that is itself a
 *        compressible zero.
 * @details The zero is V{}, the value the library compresses: 0 for an
 *          arithmetic type, a complex zero, the all-zero aggregate. Not
 *          V(0): for an aggregate (Point3, Pair16) it initialises the first
 *          member only (-Wmissing-field-initializers), and std::complex has
 *          no operator!=(complex, int). V can even be a Proxy (proxy-to-proxy
 *          assign), which is not default-constructible: not zero-comparable,
 *          every such write is counted.
 * @tparam V Type being written.
 */
template <typename V>
concept ChunkStatsZeroComparable = requires(const V& value) {
    { value != V{} } -> std::convertible_to<bool>;
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
void increm_chunk_write(V v) noexcept
{
    if constexpr (ChunkStatsZeroComparable<V>)
    {
        if (v != V{})
        {
            __chunk_stats_slot(C::stat_variant).write++;
        }
    }
    else
    {
        __chunk_stats_slot(C::stat_variant).write++;
    }
}

// ---- reduction / lifecycle -------------------------------------------------

/// @brief Reset every (variant, thread) counter slot to zero.
inline void init_all_stats() noexcept
{
    for (auto& per_variant : __chunk_stats)
    {
        for (ChunkStatsThr& slot : per_variant)
        {
            slot = ChunkStatsThr{};
        }
    }
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
inline std::array<std::mutex, CHUNK_STATS_N_VARIANT> __chunk_size_dist_mutex;
/// @brief See __chunk_size_dist_mutex.
inline std::array<std::map<unsigned int, long>, CHUNK_STATS_N_VARIANT> __chunk_size_dist;

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
        const ChunkStatsThr& slot = __chunk_stats[variant][t];
        acc.alloc += slot.alloc;
        acc.write += slot.write;
        acc.read += slot.read;
        acc.read_cst += slot.read_cst;
        acc.read_sync += slot.read_sync;
        acc.read_sync_cross += slot.read_sync_cross;
        acc.create += slot.create;
        acc.list += slot.list;
        acc.bytes_alloc += slot.bytes_alloc;
        acc.bytes_reused += slot.bytes_reused;
        acc.bytes_freed += slot.bytes_freed;
        acc.teardown_clear += slot.teardown_clear;
        acc.released += slot.released;
        acc.still_compressed_at_exit += slot.still_compressed_at_exit;
        acc.compress += slot.compress;
        acc.release_park += slot.release_park;
        acc.release_pool += slot.release_pool;
        acc.release_free += slot.release_free;
        acc.compress_attempt += slot.compress_attempt;
        acc.mark_ok += slot.mark_ok;
        acc.mark_refused += slot.mark_refused;
        acc.scan_read += slot.scan_read;
        acc.rechunk += slot.rechunk;
        acc.rechunk_rebuilt += slot.rechunk_rebuilt;
        acc.rechunk_skipped += slot.rechunk_skipped;
        acc.gate_start += slot.gate_start;
        acc.gate_start_flush_yield += slot.gate_start_flush_yield;
        acc.gate_start_yield_late += slot.gate_start_yield_late;
        acc.gate_start_deferred += slot.gate_start_deferred;
        acc.gate_flush += slot.gate_flush;
        acc.gate_driver += slot.gate_driver;
        acc.gate_deferred += slot.gate_deferred;
        acc.gate_wait += slot.gate_wait;
        acc.gate_wait_yields += slot.gate_wait_yields;
        acc.gate_mark_hit += slot.gate_mark_hit;
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
///@{
template <typename C>
void increm_chunk_alloc() noexcept
{
}
template <typename C>
void increm_chunk_read() noexcept
{
}
template <typename C>
void increm_chunk_read_cst() noexcept
{
}
template <typename C>
void increm_chunk_read_sync() noexcept
{
}
template <typename C>
void increm_chunk_read_sync_cross() noexcept
{
}
template <typename C>
void increm_chunk_create() noexcept
{
}
template <typename C>
void increm_chunk_list() noexcept
{
}
template <typename C>
void add_chunk_bytes_alloc(long) noexcept
{
}
template <typename C>
void add_chunk_bytes_reused(long) noexcept
{
}
template <typename C>
void add_chunk_bytes_freed(long) noexcept
{
}
template <typename C>
void increm_chunk_teardown_clear() noexcept
{
}
template <typename C>
void add_chunk_released(long, long) noexcept
{
}
template <typename C>
void add_chunk_still_compressed_at_exit(long) noexcept
{
}
template <typename C>
void increm_chunk_compress() noexcept
{
}
template <typename C>
void increm_chunk_release_park() noexcept
{
}
template <typename C>
void increm_chunk_release_pool() noexcept
{
}
template <typename C>
void increm_chunk_release_free() noexcept
{
}
template <typename C>
void increm_chunk_compress_attempt() noexcept
{
}
template <typename C>
void increm_chunk_mark_ok() noexcept
{
}
template <typename C>
void increm_chunk_mark_refused() noexcept
{
}
template <typename C>
void increm_chunk_scan_read() noexcept
{
}
template <typename C>
void increm_chunk_gate_start() noexcept
{
}
template <typename C>
void increm_chunk_gate_start_flush_yield() noexcept
{
}
template <typename C>
void increm_chunk_gate_start_yield_late() noexcept
{
}
template <typename C>
void increm_chunk_gate_start_deferred() noexcept
{
}
template <typename C>
void increm_chunk_gate_flush() noexcept
{
}
template <typename C>
void increm_chunk_gate_driver() noexcept
{
}
template <typename C>
void increm_chunk_gate_deferred() noexcept
{
}
template <typename C>
void increm_chunk_gate_wait(int, bool) noexcept
{
}
template <typename C>
void add_chunk_scan_read(long) noexcept
{
}
template <typename C>
void increm_chunk_rechunk() noexcept
{
}
template <typename C>
void increm_chunk_rechunk_rebuilt() noexcept
{
}
template <typename C>
void increm_chunk_rechunk_skipped() noexcept
{
}
template <typename C>
void increm_chunk_write() noexcept
{
}
template <typename C, typename V>
void increm_chunk_write(V) noexcept
{
}
inline void init_all_stats() noexcept {}
///@}

#endif

} // namespace shrink_array
