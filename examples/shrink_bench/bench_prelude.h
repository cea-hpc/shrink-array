/*
 * Copyright 2024-2026 CEA (www.cea.fr)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */

#pragma once

// Central include-order / variant-macro prelude for the shrink_bench bench.
//
// Every bench TU must include this header first (before anything else that
// pulls chunk/* transitively). It:
//   1. defines BZIP2 by default (unless already defined by the build system),
//   2. pulls the base chunk headers in the correct order,
//   3. derives SHRINK_ARRAY / CHECK_COMPRESSION / NEED_COMPRESSION from the
//      variant macros (CHUNK / CHUNK2 / CHUNK2A) and the mode macros
//      (DEBUG / CHECK),
//   4. pulls the variant-specific array_chunk* headers,
//   5. defines the complex_double alias used across TUs.
//
// The variant macros (CHUNK / CHUNK2 / CHUNK2A) MUST be defined identically
// on every TU of a given executable target. In practice they are defined
// PUBLIC on the chunk_vN library target so nothing else has to worry.

// BZip2 support: on by default. Define -DNO_BZIP2 at build time to disable.
#if !defined(BZIP2) && !defined(NO_BZIP2)
    #define BZIP2
#endif

#ifdef BZIP2
    #include <bzlib.h>
#endif

#ifdef _OPENMP
    #include <omp.h>
#endif

// Standard library headers that chunk/* transitively assumes but does not
// always #include itself. Pulling them in here (before chunk/*) makes the
// bench TUs portable across libstdc++ (looser transitive includes) and
// libc++ (stricter). If chunk/* gets its own hygiene pass later, this
// block becomes redundant and can be trimmed.
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>
#include <cstdlib> // std::getenv / std::atol (BENCH_SCHED_BLOCK)

#include "bench_int.h" // Int / UInt: the bench index type (32- or 64-bit)

#include "chunk/chunk_define.h"
#include "chunk/term_color.h"
#include "chunk/PMP_CHUNK.h"
#include "chunk/manage_ompt.h"
#include "chunk/singleton.h"
#include "chunk/global_stats_chunk_list.h"

#if (defined CHUNK || defined CHUNK2 || defined CHUNK2A)
    #define SHRINK_ARRAY
#endif

// ── The bench's work-sharing loops: `simd` on the std::vector baseline only ──
//
// A loop over a shrink array is not vectorisable: operator[] hands back a
// proxy whose assignment is a handshake with the compressor, a possible
// deferred op, an atomic or two. `omp simd` on such a loop would promise the
// compiler that its iterations are independent, which they are not, and the
// compiler is entitled to act on the promise. The std::vector baseline is
// plain memory (tab_bool is a std::vector<short> there, not the bit proxy),
// so there the same loops are exactly what `simd` is for -- and clang 22 does
// vectorise them (the multi-mat example uses the same split, LoopsSimd vs
// Loops, with the same rationale).
//
// One iteration space, one schedule(static), the same on every loop of a
// phase: that is what lets the loops run nowait (bench_phase_compute.cpp). A
// plain schedule(static) keeps its partition under `for simd` -- only the
// `simd:` schedule MODIFIER rounds chunks to the vector width -- so the
// thread-to-iteration mapping the nowait argument relies on is unchanged.
//
// The reduction variant carries the accumulators' reduction clause in BOTH
// builds: a worksharing reduction must be on a variable shared in the
// region (OpenMP forbids a `for simd` reduction on a private one, e.g. a
// reduction variable of the enclosing `parallel`), so the region declares
// none and each accumulating loop combines its partial into the shared
// accumulator itself -- legal with nowait, whole after the next barrier.
//
// Note on numerics: a `simd` loop calling cos/sin/... only vectorises with a
// vector libm (-fveclib, see chunk_enable_fast_math in examples/CMakeLists.txt),
// which the check target does not get -- those loops stay scalar there, and
// the check-mode checksums stay comparable with a shrink build's.
//
// BENCH_OMP_FOR_STATIC_NOWAIT_SCALAR is the same worksharing loop with no
// `simd` in either build: for the loops that store into tab_bool. In the
// baseline that is a std::vector<bool>, a bit-packed proxy store LLVM cannot
// vectorise -- and, in the big outlined region the bench's phases are, one
// such loop under `simd` also leaves the std::complex copy-assignments of
// its sibling loops as out-of-line calls, so those fail too. The clause
// bought nothing there but a -Wpass-failed per loop; the auto-vectoriser
// still runs on these loops exactly as before. Keep `simd` on the loops that
// do vectorise (the compute kernel, the plain fills) so that the warning
// keeps meaning something for them.
#define BENCH_PRAGMA_STR(x) _Pragma(#x)
#define BENCH_OMP_FOR_STATIC_NOWAIT_SCALAR _Pragma("omp for schedule(static) nowait")
#define BENCH_OMP_FOR_STATIC_NOWAIT_RED_SCALAR(...) BENCH_PRAGMA_STR(omp for schedule(static) nowait reduction(+ : __VA_ARGS__))
#ifdef SHRINK_ARRAY
    #define BENCH_OMP_FOR_STATIC_NOWAIT _Pragma("omp for schedule(static) nowait")
    #define BENCH_OMP_FOR_STATIC_NOWAIT_RED(...) BENCH_PRAGMA_STR(omp for schedule(static) nowait reduction(+ : __VA_ARGS__))
#else
    #define BENCH_OMP_FOR_STATIC_NOWAIT _Pragma("omp for simd schedule(static) nowait")
    #define BENCH_OMP_FOR_STATIC_NOWAIT_RED(...) BENCH_PRAGMA_STR(omp for simd schedule(static) nowait reduction(+ : __VA_ARGS__))
#endif

// ── Round-robin, chunk-aligned schedule for index-dependent loops ───────────
//
// schedule(static) hands each thread ONE contiguous slice, so a loop whose
// work per element depends on the index (heavy only above some threshold, or
// on the last 10%) loads a few adjacent threads. The _BLOCK_ macros deal
// blocks of sched_blk iterations round-robin instead, over [sched_base, end):
// sched_base is start rounded DOWN to a multiple of the block, and the loop
// skips the indices below start, so every block boundary is a multiple of the
// block size. Blocks SMALLER than a chunk are fine and faster here: a chunk is
// then written by several threads, but the finer round-robin balances better,
// and that dominates (see the measurements below). Every loop of a region that reads back what another wrote
// must use the SAME sched_blk and the SAME bounds: that is the condition under
// which OpenMP gives a thread the same iterations in each.
//
// BENCH_SCHED_BLOCK (environment, read once): unset -> 64; N -> N rounded up
// to a power of two; 0 -> the previous contiguous slices (no alignment).
// 64 threads, --mode perf, 5 runs each:
//                    0 (slices)   64       128      256      512      1024
//   chunk2_256        18.33 s   9.33 s   9.48 s  10.07 s
//   chunk2a_dyn                12.02 s  12.10 s  12.56 s  13.69 s  16.12 s
// (the 0 column predates later library fixes; compare within a row.)
//
// The macros name a local `sched_blk`; declare it with bench_sched_for().
#define BENCH_OMP_FOR_BLOCK_NOWAIT_SCALAR _Pragma("omp for schedule(static, sched_blk) nowait")
#define BENCH_OMP_FOR_BLOCK_NOWAIT_RED_SCALAR(...) BENCH_PRAGMA_STR(omp for schedule(static, sched_blk) nowait reduction(+ : __VA_ARGS__))
#ifdef SHRINK_ARRAY
    #define BENCH_OMP_FOR_BLOCK_NOWAIT_RED(...) BENCH_PRAGMA_STR(omp for schedule(static, sched_blk) nowait reduction(+ : __VA_ARGS__))
#else
    #define BENCH_OMP_FOR_BLOCK_NOWAIT_RED(...) BENCH_PRAGMA_STR(omp for simd schedule(static, sched_blk) nowait reduction(+ : __VA_ARGS__))
#endif

/// @brief BENCH_SCHED_BLOCK, read once, rounded up to a power of two; 0 keeps
///        the contiguous slices. Default 64.
inline Int bench_sched_block_env() noexcept
{
    static const Int v = []() -> Int
    {
        const char* e = std::getenv("BENCH_SCHED_BLOCK");
        const long  b = (e && e[0]) ? std::atol(e) : 64L;
        if (b <= 0)
            return 0;
        long p = 1;
        while (p < b && p < (1L << 30))
            p <<= 1;
        return static_cast<Int>(p);
    }();
    return v;
}

/// @brief Block size and aligned start of a [start, end) worksharing loop.
struct BenchSched
{
    Int blk;  ///< iterations per block (the schedule's chunk size)
    Int base; ///< first iteration: start rounded down to a multiple of blk
};

/// @brief The schedule of a [start, end) loop; call it BEFORE the parallel
///        region, so that every thread uses the same values.
inline BenchSched bench_sched_for(Int start, Int end) noexcept
{
    const Int b = bench_sched_block_env();
    if (b > 0)
        return {b, start - start % b};
    const Int t = CHUNK_PMP_MAX_THREADS;
    return {std::max<Int>(1, (end - start + t - 1) / t), start}; // contiguous slices
}

// No longer gated on CHECK: there is no CHECK macro any more, the mode is a
// runtime flag (--mode, see bench_config.h). A debug build therefore always
// carries the compression verification code; a --mode perf run simply does not
// execute it, which costs nothing it did not already cost.
#if defined SHRINK_ARRAY && defined DEBUG
    #define CHECK_COMPRESSION
#endif

#if (defined SHRINK_ARRAY) && (defined CHECK_COMPRESSION)
    #define NEED_COMPRESSION
#endif

#include "chunk/param_chunk.h"
#include "chunk/config.h"
#include "chunk/type_name.h"
#include "chunk/chunk_std.h"

#if defined CHUNK
    #include "chunk/array_chunk.h"
#elif defined CHUNK2
    #include "chunk/array_chunk2.h"
    #include "chunk/debug.h"
#elif defined CHUNK2A
    #include "chunk/array_chunk2.h"
    #include "chunk/array_chunk2A.h"
#else
    #include <vector>
#endif

using complex_double = std::complex<double>;

// Assumed cache line size (used for padding). Kept here rather than in
// bench_config.h because it's a hardware/ABI-level constant, not a bench
// mode knob.
inline constexpr std::size_t CACHE_LINE_SIZE = 64;
