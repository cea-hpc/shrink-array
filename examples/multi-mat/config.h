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

#pragma once

/**
 * @file config.h
 * @brief Every compile-time switch of the multi-mat example, as constants.
 *
 * The only file of the example that reads a preprocessor symbol. Everything
 * else tests the `constexpr bool`s below with `if constexpr`, so the
 * variants of the program (compression mode, OpenMP or not, interactive
 * input, stats) are ordinary code that the compiler checks in every build.
 *
 * ## The compression mode
 *
 * The chunk library compresses from its write path in every OpenMP build: a
 * write that closes a constant run queues its chunk, and a thread whose
 * list fills runs a pass over the candidates. What a pass needs from the
 * application is the PHASES (chunk/manage_ompt.h): a write that lands on a
 * chunk the pass holds is deferred, and somebody has to replay the deferred
 * writes before they are read -- at the barriers -- and to drain the
 * candidates left when the region ends. Two ways to provide them, two
 * executables from the same sources:
 *
 *  - manual (default, every compiler): the application makes the phase
 *    calls itself -- the four one-line wrappers of this file, and nothing
 *    else -- and compresses its field arrays by explicit compress() calls
 *    every freq_compress iterations (ComputeLoops.h). No OMPT tool in this
 *    binary (ompt_off.cpp);
 *
 *  - embedded (MULTIMAT_EMBEDDED_COMPRESSION): no call at all, the parallel
 *    loops are plain OpenMP. The OMPT tool of manage_ompt.cpp makes the
 *    phase calls from the runtime's callbacks -- implicit-task begin,
 *    barrier end, parallel end -- and the wrappers below compile to
 *    nothing. That tool is what the mode stands on, so a build without
 *    OMPT is refused here rather than compiled to a version that loses the
 *    deferred writes: CHUNK_OPT_OMPT is set by the top-level CMakeLists for
 *    Clang and IntelLLVM (OMP=ON, not under TSan); gcc's libgomp has no
 *    OMPT.
 *
 * The presence fraction (a Chunk2 array) is compressed by that engine in
 * both modes: its compress() is an audit pass, not the compressor.
 */

#include "chunk/PMP_CHUNK.h"   // the OpenMP shims of the library (serial build: 1 thread, rank 0)
#include "chunk/config.h"      // build_config::use_openmp, chunk_config::stats
#include "chunk/manage_ompt.h" // shrink_array::begin_parallel_phase, finalize_barrier_phase

#if defined(MULTIMAT_EMBEDDED_COMPRESSION) && !CHUNK_OPT_OMPT
    #error \
        "multi-mat-embedded rests on the OMPT tool of the chunk library, which this build has not (CHUNK_OPT_OMPT): build with Clang or IntelLLVM, OMP=ON, no TSan -- gcc's libgomp has no OMPT"
#endif

// `#pragma forceinline` is the classic Intel compiler's; gcc and clang warn
// "ignoring #pragma forceinline" on every use and inline the lambda anyway at
// -O3. Kept for icpc, empty elsewhere. A pragma has no constexpr form.
#if defined(__INTEL_COMPILER) && !defined(__INTEL_LLVM_COMPILER)
    #define PRAGMA_FORCEINLINE _Pragma("forceinline")
#else
    #define PRAGMA_FORCEINLINE
#endif

namespace multi_mat
{

// ── Switches ─────────────────────────────────────────────────────────────

/// @brief The library compresses on its own, through the OMPT tool of
///        manage_ompt.cpp; the program calls nothing.
#ifdef MULTIMAT_EMBEDDED_COMPRESSION
inline constexpr bool embedded_compression = true;
#else
inline constexpr bool embedded_compression = false;
#endif

/// @brief The parameters are asked on stdin (MULTIMAT_INTERACTIVE) rather
///        than taken from their defaults (Simu.h).
#ifdef MULTIMAT_INTERACTIVE
inline constexpr bool interactive_input = true;
#else
inline constexpr bool interactive_input = false;
#endif

/// @brief OpenMP build (the library's own flag: `_OPENMP`).
inline constexpr bool use_openmp = build_config::use_openmp;

/// @brief The library keeps its per-variant counters (CHUNK_STATS) and can
///        print them at exit.
inline constexpr bool chunk_stats = chunk_config::stats;

// ── OpenMP shims ─────────────────────────────────────────────────────────
// Through the library's, which are 1 / 0 / false in a serial build, so the
// example never names an omp_* function itself.

/// @brief Threads the runtime would give a parallel region.
[[nodiscard]] inline int max_threads() noexcept
{
    return CHUNK_PMP_MAX_THREADS;
}

/// @brief Rank of the calling thread in its team (0 outside a region).
[[nodiscard]] inline int thread_rank() noexcept
{
    return CHUNK_PMP_THREAD_RANK;
}

/// @brief Is the caller inside a parallel region?
[[nodiscard]] inline bool in_parallel_region() noexcept
{
    return CHUNK_PMP_IN_PARALLEL_REGION;
}

// ── The manual phase calls ───────────────────────────────────────────────
// Elided in the embedded executable, where the OMPT tool makes them.

/// @brief Inside the region, by every thread, before the first chunk write.
inline void begin_parallel_phase()
{
    if constexpr (!embedded_compression)
    {
        shrink_array::begin_parallel_phase();
    }
}

/// @brief After a barrier, before the reads that must see the writes of
///        before it: every thread replays its deferred writes, then a
///        barrier again so no thread reads while another still replays.
inline void barrier_phase()
{
    if constexpr (!embedded_compression)
    {
        shrink_array::finalize_barrier_phase();
#pragma omp barrier
    }
}

/// @brief Last thing inside the region: the deferred writes, replayed by
///        the team.
inline void end_parallel_phase_in_region()
{
    if constexpr (!embedded_compression)
    {
        shrink_array::finalize_parallel_phase_only_flush();
    }
}

/// @brief First thing after the region: the pools reset and the candidates
///        left over compressed. Sequential: the drain opens parallel
///        regions of its own.
inline void end_parallel_phase_after_region()
{
    if constexpr (!embedded_compression)
    {
        shrink_array::finalize_parallel_phase_after_flush();
    }
}

} // namespace multi_mat
