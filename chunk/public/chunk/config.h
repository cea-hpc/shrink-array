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

#include <cstddef>
#include <cstdint>

#include "chunk/internal/define_mm_protect.h"
#include "chunk/chunk_define.h"

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
    #define CHUNK_ALLOC_AFFINITY_MIN_BYTES (std::size_t(16) << 20)
#endif
inline constexpr std::size_t alloc_affinity_min_bytes =
    CHUNK_ALLOC_AFFINITY_MIN_BYTES; // -D to override (0 = off; tiny = every array, for tests)

/**
 * @brief Second values carry an identifier, not a filled array. ON by default.
 *
 * @details A chunk compressed to a second constant points into a region that is
 *          never written: the pointer names an identifier, and the value comes
 *          from the value table. is_cst_val2() stays a range test on the
 *          pointer and id_of_ptr() is arithmetic -- neither dereferences it --
 *          so the read path pays one well-predicted range test and the value
 *          costs 8 bytes.
 *
 *          The representation it replaced gave each distinct value a shared
 *          array FILLED with it, so `values[i]` returned the constant with no
 *          test at all. That property is what kept the chunk header at 16
 *          bytes, and it was bought twice over: 32 KiB per value in dyn, so the
 *          2 GiB region held 65 536 of them and a quantised field exhausted it
 *          and aborted; and a fill of cst_buffer_size elements per
 *          registration, with the page faults behind it.
 *
 *          Measured (chunk2a static 256, 2000 iterations, 64 threads, one
 *          seed): 17 503 G instructions against 18 990 G, -7.8%, IPC 5.75
 *          against 5.61, system time 199 s against 233 s. The branch costs less
 *          than the fill it removes, which is the opposite of what was expected
 *          when it was written.
 *
 *          -DCHUNK_CST_VAL2_FILLED_ARRAYS restores the arrays: for a workload
 *          whose read path dominates its registrations, or to bisect a
 *          suspected regression against the representation every release before
 *          this one used.
 *
 * @warning The failure mode of this code is a WRONG VALUE, not a crash: a site
 *          that reads a value out of the buffer, or loads `values` twice around
 *          a test, returns something plausible. Re-run rechunk_validation_*, a
 *          TSan pass and a --mode check run whenever it is touched.
 */
#ifdef CHUNK_CST_VAL2_FILLED_ARRAYS
inline constexpr bool cst_val2_no_arrays = false;
#else
inline constexpr bool cst_val2_no_arrays = true;
#endif

/// @brief Mirror of alloc_config::reclaim_empty_slabs, from the same CMake
///        option (CHUNK_RECLAIM_EMPTY_SLABS, a global add_definitions, so every
///        TU sees the same value).
///
/// Duplicated deliberately. The original lives in
/// fast_allocators/alloc_config.h, a header a std-alloc build never pulls in,
/// so naming alloc_config:: outside a fast-alloc TU does not compile -- and
/// `if constexpr (build_config::use_fast_alloc)` does NOT rescue it: the name
/// is not dependent on the enclosing template's parameters, so it is looked up
/// when that template is DEFINED, discarded branch or not (gcc says so with
/// -Wtemplate-body). This mirror is what lets such a site be a plain
/// `if constexpr` with no #ifdef, exactly as alloc_affinity_min_bytes above.
#ifdef CHUNK_RECLAIM_EMPTY_SLABS
inline constexpr bool reclaim_empty_slabs = true;
#else
inline constexpr bool reclaim_empty_slabs = false;
#endif

#ifdef _OPENMP
inline constexpr bool use_openmp = true;
#else
inline constexpr bool use_openmp = false;
#endif

// Derived — not_use_openmp kept for call-site readability
inline constexpr bool not_use_openmp = !use_openmp;

// ── sync_on_read (permanent) ────────────────────────────────────────────────
// The deferred-op visibility toll lives at the READ entry points, not the WRITE
// operators: writes no longer flush (P1); the pure-read proxy entries pull
// materialisation on demand (ChunkBase::reader_sync); a WAW guard discards a
// superseded same-offset op still queued on the writer's own rank before a
// direct write (P2); and a writer drains its OWN rank once that rank's backlog
// crosses deferred_drain_threshold (P3). This is now the permanent behaviour;
// the historical write-side-flush mode (the old OFF default) has been retired.
// sync_on_read is now the permanent behaviour (read-side visibility toll: P1/P2/P3).
// The historical write-side-flush mode has been retired; this is always ON. The
// 16 `if constexpr (build_config::sync_on_read)` sites therefore always take the
// ON branch (the OFF branches are dead and elided by the compiler; a later pass
// collapses them).
inline constexpr bool sync_on_read = true;

// P3 backlog bound: a writer drains its own rank when that rank's pending-op
// count crosses this threshold (only under sync_on_read). Too small re-creates
// the per-write toll; too big lets the backlog grow memory. -D tunable.
#ifndef CHUNK_DEFERRED_DRAIN_THRESHOLD
    #define CHUNK_DEFERRED_DRAIN_THRESHOLD 4096
#endif
inline constexpr std::int64_t deferred_drain_threshold = CHUNK_DEFERRED_DRAIN_THRESHOLD;

// TEST KNOB -- force the deferred-write path. With the per-chunk marks and
// the bounded wait, a bench or a fuzz replay defers next to nothing (gate
// deferred = 0 in the stats): the enqueue, the P3 drain, the read gate, the
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
#else
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
constexpr bool check_optim = true;
#else
constexpr bool check_optim = false;
#endif

// Convenience predicate: any kind of debug or runtime-check active
constexpr bool any_check = is_debug || check_optim;

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

// std::size_t{2}, not 2: the product is otherwise computed in int and then
// widened (bugprone-implicit-widening-of-multiplication-result). It fits in an
// int today; expressing the type once means it still will if the value grows.
constexpr std::size_t deferred_pool_granularity = std::size_t{2} * 1024 * 1024;

static constexpr bool verbose = false;

} // namespace build_config

extern bool __global_ompt;


/**
 * @namespace chunk_config
 * @brief Represents the internal logic of the compression engine (The "How").
 */
namespace chunk_config
{
using namespace build_config;

// --- Array Type Selection ---
#ifdef CHUNK
inline constexpr bool use_chunk = true;
#else
inline constexpr bool use_chunk = false;
#endif

#ifdef CHUNK2
inline constexpr bool use_chunk2 = true;
#else
inline constexpr bool use_chunk2 = false;
#endif

#ifdef CHUNK2A
inline constexpr bool use_chunk2a = true;
#else
inline constexpr bool use_chunk2a = false;
#endif

inline constexpr bool is_shrink_enabled = use_chunk || use_chunk2 || use_chunk2a;
inline constexpr bool use_chunk2or2a    = use_chunk2 || use_chunk2a;

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
 * true ("have_agressive_compression" mode): the sweep is left to run at
 * scattered, unpredictable points during the parallel section (its
 * historical behavior). Slower, but stresses the concurrent compression
 * code paths → better MT-bug detection.
 *
 * Set via -DHAVE_AGRESSIVE_COMPRESSION=ON at CMake configure time.
 */
#ifdef HAVE_AGRESSIVE_COMPRESSION
inline constexpr bool have_agressive_compression = true;
#else
inline constexpr bool have_agressive_compression = false;
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

inline constexpr int tuning_level = SHRINK_LEVEL;
// Level 1: 64KB (Stress allocation logic)
// Level 2: 2MB (Standard)
// Level 3: 16MB (High Throughput)
constexpr size_t deferred_pool_granularity = (tuning_level == 1)   ? 64 * 1024
                                             : (tuning_level == 3) ? 16 * 1024 * 1024
                                                                   : 2 * 1024 * 1024;
// SEGV hunt: two independent switches.
//   RING_LOG_DEBUG  -> ALL events, including hot-path (FLUSH/COMPRESS/STORE_VALUE).
//                      Very heavy: masks the heisenbug. Leave OFF for the hunt.
//   RING_LOG_OP     -> ONLY the 4 op-lifecycle events (ALLOC/ENQUEUE/FREE/FLUSH_ONE).
//                      Light: a handful of events per compression, does not mask.
//#define RING_LOG_DEBUG
//#define RING_LOG_OP      // SEGV hunt: op-lifecycle tracing only (light)

// The ring buffer + dump_ring_for_addr + ring_log_op must exist if EITHER flag
// is on. RING_LOG_ANY drives the buffer compilation in ring_log_debug.cpp/.h.
#if defined(RING_LOG_DEBUG) || defined(RING_LOG_OP)
    #define RING_LOG_ANY
#endif

#ifdef RING_LOG_DEBUG
inline constexpr bool ring_log_debug = true;
#else
inline constexpr bool ring_log_debug = false;
#endif

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
 *          THE DEFAULT (-DCHUNK_ASYMMETRIC_DEKKER, CMake option of the same
 *          name, ON) since Sept '26. It was opt-in for a while, because the
 *          "rare" side was not rare: a rank whose candidate list was flagged
 *          dangerous sent every write through the slow path, each of which
 *          deferred the write, flushed it back at once (an announce) and
 *          tried to compress (an announce, yielded 98% of the time to one of
 *          those flushes). At 8 threads that was 14 000 announces/s with
 *          fences and, with membarriers, a feedback loop at 1 000 000
 *          announces/s: shrink_bench chunk2a_256 went from 4.7 s to 76 s.
 *          The write engine no longer does that (write_with_optional_lock_impl:
 *          the dangerous rank writes, then drives; a writer waits a compressor
 *          out instead of deferring), and the rare side is a few hundred
 *          announces per second. Measured then, same bench, 6 threads pinned
 *          to x86 P-cores, three interleaved runs each:
 *            before the engine change            3.18 - 3.27 s
 *            new engine, symmetric               3.40 - 3.55 s  (+7%)
 *            new engine, asymmetric              2.96 - 3.15 s  (-7%)
 *          The symmetric shape pays the writers' bounded wait during a
 *          compression's scan with nothing in return; the asymmetric one
 *          removes the fence from every write and covers it. Linux only (the
 *          syscall), never under ThreadSanitizer (it does not model membarrier
 *          and would report races that are not there). An asymmetric build
 *          still falls back to the symmetric shape AT RUN TIME if membarrier
 *          registration fails (kernel < 4.14, seccomp); print_chunk_stats()
 *          says which one ran.
 */
#if defined(CHUNK_ASYMMETRIC_DEKKER) && defined(__linux__) && !defined(CHUNK_TSAN_BUILD)
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
inline constexpr bool not_use_compression_with_ompt = !has_ompt;

// --- Stats ---
#ifdef CHUNK_STATS
inline constexpr bool stats = true;
#else
inline constexpr bool stats = false;
#endif

/// @brief Run-time switch of the OMPT-driven compression: the compression
///        drivers turn it off around their own passes (has_ompt builds; this
///        primary is the always-off stand-in for the others).
template <bool Enabled>
struct OMPTcc
{
    /// @brief No switch to move.
    static void update([[maybe_unused]] bool state) {}
    /// @brief Always off.
    static bool is_enabled()
    {
        return false;
    }
    /// @brief Complement of is_enabled().
    static bool is_not_enabled()
    {
        return !is_enabled();
    }
};

/// @brief The real switch, behind the process-wide ::__global_ompt.
template <>
struct OMPTcc<true>
{
    /// @brief Set the switch.
    static void update(bool state)
    {
        ::__global_ompt = state;
    }
    /// @brief Read the switch.
    static bool is_enabled()
    {
        return ::__global_ompt;
    }
    /// @brief Complement of is_enabled().
    static bool is_not_enabled()
    {
        return !is_enabled();
    }
};

/// @brief Run-time switch of the NaN check on written values (do_check
///        builds; this primary is the always-off stand-in for the others).
template <bool Enabled>
struct CheckNaN
{
    /// @brief No switch to move.
    static void update([[maybe_unused]] bool state) {}
    /// @brief No check.
    static void maybe([[maybe_unused]] auto val, [[maybe_unused]] long i) {}
    /// @brief Always off.
    static constexpr bool is_enabled()
    {
        return false;
    }
};

/// @brief The real switch: maybe() checks while it is on.
template <>
struct CheckNaN<true>
{
    static inline bool check_nan{true}; ///< the switch, on by default

    /// @brief Set the switch.
    static void update(bool state)
    {
        check_nan = state;
    }
    /// @brief global_check_nan(@p val, @p i) if the switch is on.
    static void maybe(auto val, long i)
    {
        if (check_nan)
            global_check_nan(val, i);
    }
    /// @brief Compiled in (the switch may still be off at run time).
    static constexpr bool is_enabled()
    {
        return true;
    }
};

} // namespace chunk_config
// Rechunk rebuild suppresses the OMPT parallel-end finalize. The chunk-level
// rebuild (array_chunk_base.h) drives the deferred write path inside its OWN
// #pragma omp parallel; that region's end fires on_parallel_end, which would
// flush_all_deferred + compress MID-REBUILD, before every new chunk is
// consistently written — corrupting grow regions that span several old chunks.
// rechunk_to flushes properly afterwards at a quiescent point, so the rebuild
// sets this true around its parallel region to skip the premature finalize.
// Master-only (set/read on the encountering thread around the region).
namespace shrink_array
{
inline bool g_in_rechunk_rebuild = false;

// Monotonic count of committed rechunk_to() calls (ones that actually change the
// geometry — early-returns for no-op/empty/refused rechunks do NOT count). Bumped
// in ArrayChunkBase::rechunk_to just before the rebuild. Process-wide; used by
// benchmarks/tools to report how many re-chunks a run performed. Not reset
// automatically — a harness that wants a per-phase delta snapshots it itself.
inline unsigned long g_rechunk_count = 0;
} // namespace shrink_array
