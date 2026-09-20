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

// Bench-mode configuration. Defines whether we run in --check or --perf mode
// and the derived sizing / frequency constants. All functions are constexpr
// (or fall back to plain functions if INPUT_DATA is set, in which case the
// sizes become runtime-configurable via std::cin).

#include <cstdlib> // std::getenv / std::atoi (CHUNK_BENCH_MEM_PERIOD)

#include "bench_prelude.h"

// PRINT_MEM is enabled by default to preserve historical behaviour.
#ifndef PRINT_MEM
    #define PRINT_MEM
#endif

// Bench-level compile-time flags, exposed as `inline constexpr bool` so
// that `if constexpr (...)` can be used in shrink_bench.cpp instead of
// `#ifdef` around runtime code.
#ifdef PRINT_MEM
inline constexpr bool print_mem = true;
#else
inline constexpr bool print_mem = false;
#endif

#ifdef TEST_CAPTUR_BUG
inline constexpr bool test_captur_bug = true;
#else
inline constexpr bool test_captur_bug = false;
#endif

// INPUT_DATA: the io mode is asked for on std::cin at start-up instead of being
// derived from --mode. A hand-edited developer switch, never set by CMake.
#ifdef INPUT_DATA
inline constexpr bool input_data = true;
#else
inline constexpr bool input_data = false;
#endif

// CHECK_COMPRESSION / NEED_COMPRESSION, both computed in bench_prelude.h:
// CHECK_COMPRESSION == SHRINK_ARRAY && DEBUG, and NEED_COMPRESSION is currently
// its exact synonym. They are mirrored separately anyway -- bench_prelude.h is
// free to make them diverge again, and the call sites say which one they mean.
#ifdef CHECK_COMPRESSION
inline constexpr bool check_compression = true;
#else
inline constexpr bool check_compression = false;
#endif

#ifdef NEED_COMPRESSION
inline constexpr bool need_compression = true;
#else
inline constexpr bool need_compression = false;
#endif

// BZIP2 (bench_prelude.h: on unless -DNO_BZIP2). Distinct from
// build_config::has_bzip2, which mirrors the same macro but is the LIBRARY's
// view of it; keep the bench sites on this one.
#ifdef BZIP2
inline constexpr bool have_bzip2 = true;
#else
inline constexpr bool have_bzip2 = false;
#endif

// NOT mirrored here, on purpose -- chunk/config.h already exposes them, and a
// second spelling would be one more thing to keep in step:
//   SHRINK_ARRAY    -> chunk_config::is_shrink_enabled
//   CHUNK/2/2A      -> chunk_config::use_chunk / use_chunk2 / use_chunk2a
//   RING_LOG_DEBUG  -> chunk_config::ring_log_debug
//   CHUNK_DYNAMIC   -> build_config::chunk_dynamic
//   _OPENMP         -> build_config::use_openmp (and CHUNK_PMP_MAX_THREADS)
//   HAVE_OMPT       -> build_config::has_ompt
//
// What a mirror can and cannot buy -- the rules that decided every guard in
// this directory, and the ones in chunk/ that were left alone:
//
// In a NON-template, `if constexpr` still type-checks AND odr-uses the
// discarded branch. It may replace a directive only when BOTH branches name
// entities that exist and LINK in every configuration. That is a question of
// fact, checkable: all_type_chunk.h declares print_all_reuse_pool_breakdown
// unguarded and `nm` finds it in libchunk_vector.a, so bench_teardown.cpp's
// guard could go; fast_allocators' report_all_arenas is not even on the
// std-alloc include path, so all_type_chunk.cpp's cannot.
//
// In a TEMPLATE it is different, and this is the exception that retired eleven
// directives from bench_io.h: a discarded branch whose expressions are
// DEPENDENT on the template's parameters is never instantiated. Hence
// `template <typename VEC> read_check_binary_file` guarding chunk-only calls on
// data_check with `if constexpr`, and compress_one(auto&) in
// bench_phase_compress.cpp -- one abbreviated function template whose body is
// genuinely discarded for std::vector.
//
// Two traps inside a template:
//   - a NON-dependent name is looked up when the template is DEFINED, discarded
//     branch or not. `if constexpr (build_config::use_fast_alloc)` does not let
//     you write alloc_config::x in a std-alloc build; the fix is a mirror that
//     exists everywhere (build_config::reclaim_empty_slabs).
//   - `&&` short-circuits EVALUATION, never NAME LOOKUP. `if constexpr (A &&
//     Derived::m)` is ill-formed for a Derived without `m` even when A is false
//     -- NEST the two tests instead (array_chunk_rechunk_inline.h).
//
// Five shapes that stay #ifdef whatever you do, because no `if constexpr` can
// select them:
//   1. an #include                     (BZIP2 -> <bzlib.h>)
//   2. a type                          (the four tab_bool declarations)
//   3. a `using namespace`             (shrink_bench.cpp)
//   4. a member only one variant has   (chunk_base.h's pending_ops_)
//   5. the DEFINITION of an object whose TYPE is itself conditional --
//      chunk_std.cpp's `ChunkStatsThr __chunk_stats[3][1024]`, where
//      ChunkStatsThr exists only under CHUNK_STATS. 576 KB of BSS, measured
//      with nm, that a non-stats build has no reason to carry.
//
// And two that stay for judgement rather than for the language: a deliberate
// size decision (4 again), and platform or compiler detection (__GLIBC__,
// __GNUC__, __NVCOMPILER, aarch64) -- there is nothing to mirror there.

// CONSTEXPR is now always empty, and that is the whole of the check/perf fusion.
//
// It used to be `constexpr` unless INPUT_DATA was defined, and the bench mode
// was two compile-time booleans fed by the CHECK / PERF macros. That forced TWO
// executables per (variant, geometry) -- the same 2400-line translation unit
// compiled twice, 21 s and 2.2 GB each -- for a difference of a few integers:
// the io mode, two workload sizes, an iteration stop, two frequencies and two
// booleans. Everything below is now an ordinary runtime function, the mode is
// chosen by --mode, and CMake builds ONE target per (variant, geometry).
//
// Kept as a macro rather than deleted: it marks, at each of its call sites in
// shrink_bench.cpp, the values that USED to be compile-time constants. Reading
// `CONSTEXPR bool stats = have_stats();` still says "this is fixed for the whole
// run, decided once at startup", which is exactly the contract.
#define CONSTEXPR

/// @brief The bench mode, fixed at startup by --mode and constant thereafter.
enum class BenchMode
{
    check, ///< small sizes, verification and stats on -- the default
    perf   ///< large sizes, verification off
};

/// @brief Current mode. Written ONCE, by set_bench_mode() during argument
///        parsing, before any accessor below is read.
inline BenchMode g_bench_mode = BenchMode::check;

inline bool mode_check = true;  ///< g_bench_mode == check (mirror, kept for readability)
inline bool mode_perf  = false; ///< g_bench_mode == perf

/**
 * @brief Select the bench mode. Call once, from argument parsing.
 * @param m the mode --mode asked for.
 * @warning Must run BEFORE the first get_ or have_ accessor call: those used to be
 *          compile-time constants and their callers still treat them as fixed
 *          for the whole run.
 */
inline void set_bench_mode(BenchMode m)
{
    g_bench_mode = m;
    mode_check   = (m == BenchMode::check);
    mode_perf    = (m == BenchMode::perf);
}

inline int get_io_mode()
{
    // The two branches only ever differed by the check-mode value (2 reads the
    // dump back and verifies it, which needs a shrink build; the baseline
    // writes it, 1). Nothing here names a variant API, so no directive.
    if (mode_check)
        return chunk_config::is_shrink_enabled ? 2 : 1;
    if (mode_perf)
        return 0;
    return -1;
}

inline int get_size_interval(int multiplier = 1, int divisor = 1)
{
    int base_size = -1;
    if (mode_check)
        base_size = 20000;
    else if (mode_perf)
        base_size = 100000;

    return (base_size * multiplier) / divisor;
}

inline int get_size_data(int multiplier = 1, int divisor = 1)
{
    long base_size = -1; // long to avoid overflow on large data sets
    if (mode_check)
        base_size = 5000000;
    else if (mode_perf)
        base_size = 1000000000;

    return static_cast<int>((base_size * multiplier) / divisor);
}

inline int get_iter_stop(int iter_max)
{
    if (mode_check)
        return 33;
    else if (mode_perf)
        return iter_max;
    return -1;
}

inline int get_freq_raz(int iter_max)
{
    if (mode_check)
        return 4;
    else if (mode_perf)
    {
        int ret = iter_max / 10;
        if (ret == 0)
        {
            ret = 1;
        }
        return ret;
    }
    return -1;
}

inline int get_freq_conso_mem()
{
    // CHUNK_BENCH_MEM_PERIOD=<n>: print the memory line every n iterations,
    // whatever the mode. =1 turns the ten iterations before an advisor tick
    // from a blind spot into a curve, which is what locating a transient
    // peak needs. Decided once at startup, like every other CONSTEXPR site.
    if (const char* e = std::getenv("CHUNK_BENCH_MEM_PERIOD"))
    {
        const int n = std::atoi(e);
        if (n > 0)
            return n;
    }
    if (mode_check)
        return 1;
    else if (mode_perf)
        return 10;
    return -1;
}

inline bool have_stats()
{
    if (mode_check)
        return true;
    else if (mode_perf)
        return false;
    return false;
}

inline bool have_check_sum()
{
    return have_stats();
}