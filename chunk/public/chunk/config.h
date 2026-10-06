#pragma once


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

#include <cstddef> // std::size_t
#include <cstdint> // std::int64_t
#include <atomic>  // g_ompt_suspended

#include "chunk/internal/define_mm_protect.h" // MM_PROTECT (the switch, off by default)
#include "chunk/chunk_define.h"               // FAST_ALLOC

namespace build_config
{

#ifdef MM_PROTECT
inline constexpr bool use_mm_protect = true;
#else
inline constexpr bool use_mm_protect = false;
#endif

// Runtime-configurable chunk geometry (DynamicChunkSize). Prefer this constexpr
// over raw #ifdef CHUNK_DYNAMIC: use `if constexpr (build_config::chunk_dynamic)`
// wherever the guarded code is a statement whose symbols exist in both modes.
// Member/type/virtual declarations that only make sense in dynamic mode still
// need the #ifdef (they must not exist at all otherwise).
#ifdef CHUNK_DYNAMIC
inline constexpr bool chunk_dynamic = true;
#else
inline constexpr bool chunk_dynamic = false;
#endif
/// @brief Per-array slab affinity: an array whose chunk metadata alone weighs
///        more than this gets its own allocator set (rank x size), so its
///        buffers share slabs with no other array and a re-chunk or a
///        destruction empties whole slabs (madvise'd back). Below it the rank's
///        shared pool is used: with ~240 KB slabs a small array would pay one
///        slab per rank for nothing. 0 disables affinity altogether.
#ifndef CHUNK_ALLOC_AFFINITY_MIN_BYTES
    #define CHUNK_ALLOC_AFFINITY_MIN_BYTES (std::size_t(16) << 20U)
#endif
inline constexpr std::size_t alloc_affinity_min_bytes =
    CHUNK_ALLOC_AFFINITY_MIN_BYTES; // -D to override (0 = off; tiny = every array, for tests)

/// @brief Mirror of alloc_config::reclaim_empty_slabs, from the same macro
///        (CHUNK_NO_RECLAIM_EMPTY_SLABS, defined only by the CMake option
///        CHUNK_RECLAIM_EMPTY_SLABS=OFF, a global add_definitions, so every TU
///        sees the same value). On by default, as the original.
///
/// Duplicated deliberately. The original lives in
/// fast_allocators/internal/alloc_config.h, a header a std-alloc build never pulls in,
/// so naming alloc_config:: outside a fast-alloc TU does not compile -- and
/// `if constexpr (build_config::use_fast_alloc)` does NOT rescue it: the name
/// is not dependent on the enclosing template's parameters, so it is looked up
/// when that template is DEFINED, discarded branch or not (gcc says so with
/// -Wtemplate-body). This mirror is what lets such a site be a plain
/// `if constexpr` with no \#ifdef, exactly as alloc_affinity_min_bytes above.
#ifdef CHUNK_NO_RECLAIM_EMPTY_SLABS
inline constexpr bool reclaim_empty_slabs = false;
#else
inline constexpr bool reclaim_empty_slabs = true;
#endif

#ifdef _OPENMP
inline constexpr bool use_openmp = true;
#else
inline constexpr bool use_openmp = false;
#endif

// Derived — not_use_openmp kept for call-site readability
inline constexpr bool not_use_openmp = !use_openmp;

// ── Read-side visibility ────────────────────────────────────────────────────
// The deferred-op visibility toll lives at the READ entry points, not the WRITE
// operators: writes do not flush; the pure-read proxy entries pull
// materialisation on demand (ChunkBase::reader_sync); a WAW guard discards a
// superseded same-offset op still queued on the writer's own rank before a
// direct write; and a writer drains its OWN rank once that rank's backlog
// crosses deferred_drain_threshold.
inline constexpr bool sync_on_read = true;

// Backlog bound: a writer drains its own rank when that rank's pending-op
// count crosses this threshold. Too small re-creates the per-write toll; too
// big lets the backlog grow memory. -D tunable.
#ifndef CHUNK_DEFERRED_DRAIN_THRESHOLD
    #define CHUNK_DEFERRED_DRAIN_THRESHOLD 4096
#endif
inline constexpr std::int64_t deferred_drain_threshold = CHUNK_DEFERRED_DRAIN_THRESHOLD;

// TEST KNOB -- force the deferred-write path. With the per-chunk marks and
// the bounded wait, a bench or a fuzz replay defers next to nothing (gate
// deferred = 0 in the stats): the enqueue, the backlog drain, the read gate, the
// WAW guard and the cross-rank replays run only when a compressor happens to
// be holding the chunk at the very instant of a write. N > 0 sends one write
// in N of every thread down that path with no compressor in sight (from the
// chunk's point of view exactly "a compressor had it marked"), alternating the
// two exits: check #1 (no writer flag raised) and check #2 (flag raised then
// dropped). A build with -DCHUNK_FORCE_DEFER_EVERY=7 must pass the file
// cross-check bit for bit -- that is the test. 0 (the default) compiles the
// knob out; ignored without OpenMP (no deferred path there).
#ifndef CHUNK_FORCE_DEFER_EVERY
    #define CHUNK_FORCE_DEFER_EVERY 0
#endif
inline constexpr unsigned force_defer_every = use_openmp ? CHUNK_FORCE_DEFER_EVERY : 0u;

#ifdef FAST_ALLOC
inline constexpr bool use_fast_alloc = true;
inline constexpr bool use_std_alloc  = false;
#else
inline constexpr bool use_std_alloc  = true;
inline constexpr bool use_fast_alloc = false;
#endif

#ifdef DEBUG
inline constexpr bool is_debug = true;
#else
inline constexpr bool is_debug = false;
#endif

// Independent from is_debug — can be combined.
// Cost: a few extra atomic loads / pointer comparisons per write.
// Use to hunt rare MT bugs that don't reproduce in -O0 -g builds.

#ifdef CHECK_OPTIM
inline constexpr bool check_optim = true;
#else
inline constexpr bool check_optim = false;
#endif

// Convenience predicate: any kind of debug or runtime-check active
inline constexpr bool any_check = is_debug || check_optim;

#ifdef HAVE_OMPT
inline constexpr bool has_ompt = true;
#else
inline constexpr bool has_ompt = false;
#endif

// Derived — has_not_ompt kept for call-site readability
inline constexpr bool has_not_ompt = !has_ompt;

#ifdef BZIP2
inline constexpr bool has_bzip2 = true;
#else
inline constexpr bool has_bzip2 = false;
#endif

#ifdef DO_CHECK
inline constexpr bool do_check = true;
#else
inline constexpr bool do_check = false;
#endif

// Single source of the deferred op / ctx pool slab size, in bytes
// (deferred_op_pool.h; CHUNK_DEFERRED_POOL_KB overrides it at run time).
// 64 KiB, measured: 2 MiB slabs were slower for the same peak RSS
// (shrink_bench_chunk_dyn --mode perf, 64 / 128 threads, 2-4%).
// std::size_t{64}: the product in size_t, not in int then widened
// (bugprone-implicit-widening-of-multiplication-result).
inline constexpr std::size_t deferred_pool_granularity = std::size_t{64} * 1024;

inline constexpr bool verbose = false;

// --- Diagnostic switches: -D on the command line, or the CMake option that sets it ---
#ifdef PERF
/// Perf build: ArrayChunk2's diagnostic scan of missed compressions is skipped.
inline constexpr bool perf_build = true;
#else
inline constexpr bool perf_build = false;
#endif

#ifdef CHUNK_POOL_NO_REUSE
/// Deferred-op pool without free-list: every block a real new/delete (ASan / TSan).
inline constexpr bool pool_no_reuse = true;
#else
inline constexpr bool pool_no_reuse = false;
#endif

#ifdef CHUNK_ALIGN_CHECK
/// Abort on a slab block under-aligned for its type.
inline constexpr bool align_check = true;
#else
inline constexpr bool align_check = false;
#endif

#ifdef RECHUNK_LEDGERTRACE
/// Trace every pool-ledger state transition on stderr.
inline constexpr bool rechunk_ledger_trace = true;
#else
inline constexpr bool rechunk_ledger_trace = false;
#endif

#ifdef CHUNK_NO_FREE_GUARD
/// The teardown double-free guard (sort + dedup of the blocks to free).
inline constexpr bool free_guard = false;
#else
inline constexpr bool free_guard = true;
#endif

#ifdef DEBUG_SET_SIZE
/// Print the compression thresholds and the lists' memory estimate.
inline constexpr bool debug_set_size = true;
#else
inline constexpr bool debug_set_size = false;
#endif

#ifdef CHUNK_SECOND_VALUE_INDEXED
/// Chunk2A: second values of more than 4 bytes stored as 4-byte identifiers.
inline constexpr bool second_value_indexed = true;
#else
inline constexpr bool second_value_indexed = false;
#endif

} // namespace build_config

/// Number of threads currently suspending the OMPT callbacks (they run only
/// while it is zero). A COUNTER, not a flag: several threads suspend them at
/// once (compress_chunk_to_compress runs on every thread of a region), and a
/// flag would let the FIRST to finish resume them while the others are still
/// compressing. Atomic: it is written and read concurrently. See OMPTcc<true>.
///
/// On a cache line of its own (OmptSuspendCounter): every OMPT callback of
/// every thread loads it, so a neighbour written often would turn those
/// loads into misses. alignas on the VARIABLE would only align its start;
/// alignas on the TYPE also makes sizeof 128, so nothing follows it on the
/// line. Derived from std::atomic<int>, constructors included: every use is
/// unchanged.
struct alignas(128) OmptSuspendCounter : public std::atomic<int>
{
    using std::atomic<int>::atomic;
};
extern OmptSuspendCounter g_ompt_suspended;


/**
 * @namespace chunk_config
 * @brief Represents the internal logic of the compression engine (The "How").
 */
namespace chunk_config
{
using namespace build_config;

// --- Array Type Selection ---
/// True when built with -DCHUNK: the Chunk variant is the selected array type.
#ifdef CHUNK
inline constexpr bool use_chunk = true;
#else
inline constexpr bool use_chunk = false;
#endif

/// True when built with -DCHUNK2: the Chunk2 variant is the selected array type.
#ifdef CHUNK2
inline constexpr bool use_chunk2 = true;
#else
inline constexpr bool use_chunk2 = false;
#endif

/// True when built with -DCHUNK2A: the Chunk2A variant is the selected array type.
#ifdef CHUNK2A
inline constexpr bool use_chunk2a = true;
#else
inline constexpr bool use_chunk2a = false;
#endif

inline constexpr bool is_shrink_enabled = use_chunk || use_chunk2 || use_chunk2a; ///< A chunked variant is selected.
inline constexpr bool use_chunk2or2a    = use_chunk2 || use_chunk2a;              ///< Chunk2 or Chunk2A is selected.

/**
 * @brief Chunk2's second-value index for an array that has declared none.
 *
 * @details An ArrayChunk2 registers (T{}, this index) at construction and
 *          keeps it until set_second_value(). Every consumer of the index
 *          answers "no second constant" for it: get_cst_val2() is nullptr,
 *          is_cst_val2() false, written() reduces to "not cst_val1", and the
 *          compression pass compresses only the runs of first_value.
 */
inline constexpr int no_second_value_index = -1;

// --- Compression Engine Logic ---
// Compression is always the embedded, incremental one. The manual mode
// (HAVE_CHUNK_MANUAL_COMPRESSION, where compression happened only in explicit
// compress_int() sweeps) was retired; see the tag manual-compression-last.

/**
 * @brief Aggressive compression mode toggle.
 *
 * false (default, "release" mode): the deferred compression sweep is
 * triggered explicitly at end-of-parallel-phase boundaries via
 * compress_chunk_to_compress_potentially(true). Fewer, more grouped and more
 * predictable compression events → faster in production, but exercises
 * fewer concurrent code paths.
 *
 * true ("have_aggressive_compression" mode): the sweep is left to run at
 * scattered, unpredictable points during the parallel section. Slower, but
 * stresses the concurrent compression
 * code paths → better MT-bug detection.
 *
 * Set via -DHAVE_AGGRESSIVE_COMPRESSION=ON at CMake configure time.
 */
#ifdef HAVE_AGGRESSIVE_COMPRESSION
inline constexpr bool have_aggressive_compression = true;
#else
inline constexpr bool have_aggressive_compression = false;
#endif

/**
 * SHRINK_SMALL_CHUNK: chunk size vs read-gate precision trade-off.
 *
 * Default (undefined): each chunk carries a per-chunk pending_ops_ counter
 * (std::atomic<int32_t>). Because ChunkBase is 8-byte aligned (the `values`
 * pointer), the 4-byte counter costs 8 bytes with padding, so e.g.
 * sizeof(Chunk2A<double>) is 24. In exchange the read path gates precisely on
 * THIS chunk's pending count and keeps clean-chunk reads off flush_chunk_all_
 * ranks — ~10% faster at high thread counts (measured: the coarse gate spun
 * ~11% of samples in rank_flush_acquire's CAS).
 *
 * Defined (-DSHRINK_SMALL_CHUNK=ON): drop the counter -> chunks stay at 16
 * bytes, at the cost of the coarse any_pending_ops() gate (any pending op
 * anywhere makes a dirty-chunk read flush). Choose this when chunk memory
 * footprint matters more than the ~10% throughput.
 *
 * A true 16-byte + precise-gate build is not cleanly achievable: the counter
 * cannot be bit-packed into the aligned `values` pointer (it is a count, not a
 * flag) and an external per-array counter would need a chunk*->index lookup on
 * the flush side (which only holds the pointer).
 */
#ifdef SHRINK_SMALL_CHUNK
inline constexpr bool use_precise_pending_gate = false;
#else
inline constexpr bool use_precise_pending_gate = true;
#endif

/**
 * @brief Per-chunk exclusion for the compressor's pass (ChunkBase::steal_mark_).
 *
 * With it, a writer is kept off a chunk only while a compressor has that chunk
 * marked, and the compressor's writer window (CompressionManager::start())
 * waits only for the writers whose flag points to a marked chunk -- nothing
 * global on the write path, no steering of the other writers. Same
 * condition as the precise pending gate: the mark uses the padding the
 * pending_ops_ counter leaves, and neither exists without OpenMP or under
 * SHRINK_SMALL_CHUNK -- those builds gate every writer on hard_inflight_ for
 * the whole pass, as before.
 */
inline constexpr bool use_chunk_steal_mark = build_config::use_openmp && use_precise_pending_gate;

/**
 * @brief Tuning Levels:
 * Level 1: Small buffers, aggressive flushing/compression (Debug/Race detection)
 * Level 2: Standard production settings
 * Level 3: Large buffers, minimal management overhead (Max performance)
 */
#ifndef SHRINK_LEVEL
    #define SHRINK_LEVEL 3 // Default to standard
#endif

/// SHRINK_LEVEL (1, 2 or 3, default 3) as a constexpr; selects Chunk2Def's thresholds.
inline constexpr int tuning_level = SHRINK_LEVEL;
// Debug ring log: two independent switches.
//   RING_LOG_DEBUG  -> ALL events, including hot-path (FLUSH/COMPRESS/STORE_VALUE).
//                      Very heavy: can mask timing-dependent bugs.
//   RING_LOG_OP     -> ONLY the 4 op-lifecycle events (ALLOC/ENQUEUE/FREE/FLUSH_ONE).
//                      Light: a handful of events per compression, does not mask.
//#define RING_LOG_DEBUG
//#define RING_LOG_OP      // op-lifecycle tracing only (light)

// The ring buffer + dump_ring_for_addr + ring_log_op must exist if EITHER flag
// is on. RING_LOG_ANY drives the buffer compilation in ring_log_debug.cpp/.h.
#if defined(RING_LOG_DEBUG) || defined(RING_LOG_OP)
    #define RING_LOG_ANY
#endif

/// True under RING_LOG_DEBUG: ring-log every event, hot path included.
#ifdef RING_LOG_DEBUG
inline constexpr bool ring_log_debug = true;
#else
inline constexpr bool ring_log_debug = false;
#endif

/// True under RING_LOG_OP: ring-log only the op-lifecycle events.
#ifdef RING_LOG_OP
inline constexpr bool ring_log_op_only = true;
#else
inline constexpr bool ring_log_op_only = false;
#endif

/**
 * @brief Equivalent to HAVE_EMBEDDED_FAST_BLOCK_COMPRESSION.
 * True under OpenMP, the only mode where block-level deferred writes
 * are needed.
 * This is the primary compression-mode flag from which the two
 * aliases below are derived.
 */
inline constexpr bool use_fast_block_compression = use_openmp;

// A ThreadSanitizer build, whichever compiler spells it. TSan does not model
// membarrier(), so such a build keeps the symmetric handshake below.
#if defined(__SANITIZE_THREAD__)
    #define CHUNK_TSAN_BUILD 1
#elif defined(__has_feature)
    #if __has_feature(thread_sanitizer)
        #define CHUNK_TSAN_BUILD 1
    #endif
#endif
/// True in a ThreadSanitizer build (detected above): selects the symmetric handshake TSan can model.
#ifdef CHUNK_TSAN_BUILD
inline constexpr bool tsan_build = true;
#else
inline constexpr bool tsan_build = false;
#endif

/**
 * @brief The write/compress handshake's shape (WriteCompressSyncPolicy).
 *
 * @details Two sides must never run together: an element WRITE inside its
 *          critical section, and a COMPRESSOR stealing that chunk's buffer
 *          (or a cross-rank flush). The writes are the hot path -- billions
 *          per run -- and the other side is rare; on the reference bench it
 *          did not run once (the CHUNK_STATS gate counters).
 *
 *          - symmetric (false): both sides announce with a seq_cst store and a
 *            full fence, then look at the other. Correct everywhere, but the
 *            writer pays `stlr` + `dmb ish` + `ldar` on AArch64 (an `mfence`
 *            on x86) for EVERY element write: ~70% of
 *            write_with_optional_lock_impl in the profile.
 *          - asymmetric (true): the writer's announce is a plain store and a
 *            COMPILER barrier only; the rare side, after its own seq_cst
 *            announce, calls membarrier(MEMBARRIER_CMD_PRIVATE_EXPEDITED),
 *            which runs a full memory barrier on every CPU executing a thread
 *            of this process before returning. Each writer's store->load pair
 *            is then split by that barrier, or entirely before, or entirely
 *            after it -- and in every case one of the two sides sees the
 *            other. The pattern JVM safepoints and .NET's
 *            FlushProcessWriteBuffers rely on; see announce_write() for the
 *            argument and heavy_barrier() for the call.
 *
 *          The asymmetric shape is THE shape of every build that can (no
 *          macro, no CMake option). Measured on shrink_bench chunk2a_256, 6
 *          x86 threads: symmetric +7%, asymmetric -7% against the previous
 *          write engine -- the symmetric shape pays the writers' bounded wait
 *          during a compression's scan with nothing in return; the asymmetric
 *          one removes the fence from every write. It relies on the rare side
 *          being rare: an engine that sends every write of a "dangerous" rank
 *          through the slow path (defer, flush, try to compress -- one
 *          announce each) turns membarriers into a feedback loop (4.7 s ->
 *          76 s on that bench). Linux only (the syscall), never under
 *          ThreadSanitizer (it does not model membarrier and would report
 *          races that are not there): those two are the only symmetric builds
 *          left. An asymmetric build still falls back to the symmetric shape
 *          AT RUN TIME if membarrier registration fails (kernel < 4.14,
 *          seccomp); print_chunk_stats() says which one ran.
 */
#if defined(__linux__) && !defined(CHUNK_TSAN_BUILD)
inline constexpr bool asymmetric_dekker = true;
#else
inline constexpr bool asymmetric_dekker = false;
#endif

/**
 * @brief Alias for use_fast_block_compression.
 * Equivalent to HAVE_EMBEDDED_FAST_BLOCK_COMPRESSION.
 * Both macros are always set/cleared together by the _OPENMP block
 * in chunk_define.h, so they are interchangeable. The name is kept
 * for clarity at call sites that reason about the counter mechanism.
 */
inline constexpr bool use_compression_without_counter = use_fast_block_compression;

/**
 * @brief Complement of use_fast_block_compression.
 * Equivalent to HAVE_EMBEDDED_COMPRESSION_WITH_COUNTER.
 * Active in serial builds only.
 */
inline constexpr bool use_compression_with_counter = !use_fast_block_compression;

/**
 * @brief Compression driven by OMPT callbacks.
 * Equivalent to HAVE_EMBEDDED_COMPRESSION_WITH_OMPT, so that the OMPT-only
 * paths can be selected with `if constexpr` instead of the preprocessor.
 */
inline constexpr bool use_compression_with_ompt     = has_ompt;
inline constexpr bool not_use_compression_with_ompt = !has_ompt; ///< Complement of use_compression_with_ompt.

// --- Stats ---
/// True under CHUNK_STATS: per-thread chunk counters are collected and reported.
#ifdef CHUNK_STATS
inline constexpr bool stats = true;
#else
inline constexpr bool stats = false;
#endif

/// @brief Run-time switch of the OMPT-driven compression: the compression
///        drivers turn it off around their own passes (has_ompt builds; this
///        primary is the always-off stand-in for the others).
template <bool Enabled>
class OMPTcc
{
  public:
    /// @brief No switch to move.
    static void update([[maybe_unused]] bool state) noexcept {}

  private:
    /// @brief Always off.
    static bool is_enabled() noexcept
    {
        return false;
    }
    /// @brief Complement of is_enabled().
    static bool is_not_enabled() noexcept
    {
        return !is_enabled();
    }
};

/// @brief The real switch, behind the process-wide ::g_ompt_suspended.
template <>
class OMPTcc<true>
{
  public:
    /// @brief Set the switch.
    /// @details update(false) suspends the callbacks, update(true) resumes
    ///          them; they stay suspended while ANY thread holds a suspension.
    ///          The resume never takes the count below zero, so an extra
    ///          update(true) is a no-op, as it was with the former flag.
    ///          relaxed: a switch, publishing no data.
    static void update(bool state) noexcept
    {
        if (!state)
        {
            ::g_ompt_suspended.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        int cur = ::g_ompt_suspended.load(std::memory_order_relaxed);
        while (cur > 0 && !::g_ompt_suspended.compare_exchange_weak(cur, cur - 1, std::memory_order_relaxed))
        {
        }
    }

  private:
    /// @brief Read the switch.
    static bool is_enabled() noexcept
    {
        return ::g_ompt_suspended.load(std::memory_order_relaxed) == 0;
    }
    /// @brief Complement of is_enabled().
    static bool is_not_enabled() noexcept
    {
        return !is_enabled();
    }
};

/// @brief Run-time switch of the NaN check on written values (do_check
///        builds; this primary is the always-off stand-in for the others).
template <bool Enabled>
class CheckNaN
{
  public:
    /// @brief No switch to move.
    static void update([[maybe_unused]] bool state) noexcept {}
    /// @brief No check.
    static void maybe([[maybe_unused]] auto val, [[maybe_unused]] long i) noexcept {}

  private:
    /// @brief Always off.
    static constexpr bool is_enabled() noexcept
    {
        return false;
    }
};

/// @brief The real switch: maybe() checks while it is on.
template <>
class CheckNaN<true>
{
  public:
    /// @brief Set the switch.
    static void update(bool state) noexcept
    {
        check_nan = state;
    }
    /// @brief global_check_nan(@p val, @p i) if the switch is on.
    static void maybe(auto val, long i) noexcept
    {
        if (check_nan)
        {
            global_check_nan(val, i);
        }
    }

  private:
    /// @brief Compiled in (the switch may still be off at run time).
    static constexpr bool is_enabled() noexcept
    {
        return true;
    }

    static inline bool check_nan{true}; ///< the switch, on by default
};

} // namespace chunk_config
// Rechunk rebuild suppresses the OMPT parallel-end finalize. The chunk-level
// rebuild (array_chunk_base.h) drives the deferred write path inside its OWN
// #pragma omp parallel; that region's end fires on_parallel_end, which would
// flush_all_deferred + compress MID-REBUILD, before every new chunk is
// consistently written — corrupting grow regions that span several old chunks.
// rechunk_to flushes properly afterwards at a quiescent point, so the rebuild
// sets this true around its parallel region to skip the premature finalize.
// WRITTEN by the master only, outside any region (store, release). READ by
// the OMPT callbacks (load, acquire): on_parallel_end on the master, but
// on_sync_region on EVERY thread of the team. The fork and the closing
// barrier order the reads made inside the region; the one that could cross
// the master's `false` -- a worker leaving the closing implicit barrier -- is
// filtered by on_sync_region's early return, a guard that exists for another
// reason. Atomic so that this does not rest on it (it was a plain bool,
// documented as master-only).
namespace shrink_array
{
inline std::atomic<bool> g_in_rechunk_rebuild{false};

// Monotonic count of committed rechunk_to() calls (ones that actually change the
// geometry — early-returns for no-op/empty/refused rechunks do NOT count). Bumped
// in ArrayChunkBase::rechunk_to just before the rebuild. Process-wide; used by
// benchmarks/tools to report how many re-chunks a run performed. Not reset
// automatically — a harness that wants a per-phase delta snapshots it itself.
inline unsigned long g_rechunk_count = 0;
} // namespace shrink_array
