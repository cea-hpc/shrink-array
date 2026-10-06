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

#include <cmath>  // std::floor, std::isfinite
#include <cstdio> // std::snprintf

#include <algorithm> // std::min, std::max
#include <iostream>  // std::cerr
#include <limits>    // std::numeric_limits
#include <string>    // std::string

// Bench-mode configuration. Defines whether we run in --check or --perf mode
// and the derived sizing / frequency constants. All functions are constexpr
// (or fall back to plain functions if INPUT_DATA is set, in which case the
// sizes become runtime-configurable via std::cin).

#include <cstdlib> // std::getenv

#include "bench_prelude.h"            // Int, build_config::, CHUNK_PMP_MAX_THREADS
#include "chunk/internal/env_parse.h" // env_number / arg_number (checked, not atoi)

// PRINT_MEM is enabled by default.
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

// CONSTEXPR is always empty: the bench mode is a runtime choice (--mode), so
// ONE target is built per (variant, geometry) instead of two (check / perf).
//
// Kept as a macro rather than deleted: it marks, at each of its call sites in
// shrink_bench.cpp, values fixed for the whole run, decided once at startup.
// Reading `CONSTEXPR const bool stats = have_stats();` still says exactly that.
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
 * @warning Must run BEFORE the first get_ or have_ accessor call: their callers
 *          treat them as fixed for the whole run.
 */
inline void set_bench_mode(BenchMode m)
{
    // An int64 build (BENCH_INT64) checks against the same data_tab* files as
    // the 32-bit one: they hold element values and a checksum, nothing sized by
    // Int, and the index type changes no value the bench computes. A mismatch
    // there is a real 32/64-bit divergence -- in the library's index path, or
    // an index expression of the bench that overflows at 32 bits.
    g_bench_mode = m;
    mode_check   = (m == BenchMode::check);
    mode_perf    = (m == BenchMode::perf);
}

/// @brief I/O mode of the run.
/// @return check mode: 2 in a shrink build, 1 otherwise; perf mode: 0; no mode chosen: -1.
inline Int get_io_mode()
{
    // The two branches only ever differed by the check-mode value (2 reads the
    // dump back and verifies it, which needs a shrink build; the baseline
    // writes it, 1). Nothing here names a variant API, so no directive.
    if (mode_check)
    {
        return chunk_config::is_shrink_enabled ? 2 : 1;
    }
    if (mode_perf)
    {
        return 0;
    }
    return -1;
}

/// @brief Per-mode base of the interval size, before --multiply / --divisor.
inline long long get_size_base_interval()
{
    if (mode_check)
    {
        return 20000;
    }
    if (mode_perf)
    {
        return 100000;
    }
    return -1;
}

/// @brief Per-mode base of the array size, before --multiply / --divisor.
inline long long get_size_base_data()
{
    if (mode_check)
    {
        return 5000000;
    }
    if (mode_perf)
    {
        return 1000000000;
    }
    return -1;
}

/// @brief floor(base * multiplier / divisor), in 64-bit: --multiply is real.
inline long long scaled_size(long long base, double multiplier, Int divisor)
{
    return static_cast<long long>(std::floor(static_cast<double>(base) * multiplier / static_cast<double>(divisor)));
}

/// @brief Interval size: the per-mode base scaled by @p multiplier / @p divisor (scaled_size()).
/// @param multiplier --multiply factor.
/// @param divisor --divisor value.
/// @return The scaled size, converted to Int.
inline Int get_size_interval(double multiplier = 1.0, Int divisor = 1)
{
    return static_cast<Int>(scaled_size(get_size_base_interval(), multiplier, divisor));
}

/// @brief Array size: the per-mode base scaled by @p multiplier / @p divisor (scaled_size()).
/// @param multiplier --multiply factor.
/// @param divisor --divisor value.
/// @return The scaled size, converted to Int.
inline Int get_size_data(double multiplier = 1.0, Int divisor = 1)
{
    return static_cast<Int>(scaled_size(get_size_base_data(), multiplier, divisor));
}

/**
 * @brief Refuse factors that push the bench's Int arithmetic past its range.
 * @details The largest Int the bench computes is below
 *          max(2 * sz + 65536, sz + max(interval, 2 * iter_max, 65536)):
 *            - tab_float and tab_char are CREATED at 2 * sz (shrink_bench.cpp,
 *              resized to sz afterwards): the largest term. It is what made
 *              --mode perf --multiply 2 create a float array of 4e9 elements
 *              wrapped to a negative size (std::bad_alloc);
 *            - end0 = start0 + size, BEFORE its clip to sz (start0 <= sz,
 *              size <= interval);
 *            - i + 2 * k in the compute phase (i < sz, k < iter_max);
 *            - end0 + 256, the tail loop;
 *            - the library's Int indices (ind_type = Int): the element count,
 *              and chunk count x chunk size, up to sz + one chunk (65536
 *              covers any geometry the rechunk reaches).
 *          Beyond that the bench and the library need 64-bit indices, which
 *          is what the int64 declination is (bench_int.h): `Int` is `long`
 *          there, so this same bound is computed against LONG_MAX and a
 *          `--multiply` the 32-bit build refuses simply passes.
 *          Also refuses a factor that leaves an empty interval. Exits(1)
 *          with the largest valid --multiply for this --divisor.
 */
inline void check_size_limits(double multiplier, Int divisor, Int iter_max)
{
    // %.10g: the default stream precision printed 1.9999999 as "2".
    const auto real = [](double v)
    {
        char b[32];
        std::snprintf(b, sizeof b, "%.10g", v);
        return std::string(b);
    };
    if (!std::isfinite(multiplier) || multiplier <= 0.0 || divisor < 1)
    {
        std::cerr << "error: --multiply must be a real > 0 and --divisor an integer >= 1 (got " << real(multiplier)
                  << ", " << divisor << ")\n";
        std::exit(1);
    }
    const long long base_data = get_size_base_data();
    const long long base_itv  = get_size_base_interval();
    if (base_data <= 0 || base_itv <= 0)
    {
        return; // no --mode chosen: nothing is allocated from these sizes
    }

    if (scaled_size(base_itv, multiplier, divisor) < 1)
    {
        std::cerr << "error: --multiply " << real(multiplier) << " --divisor " << divisor
                  << " leaves an empty interval (the interval base is " << base_itv << ")\n";
        std::exit(1);
    }

    const long long margin = std::max({2LL * iter_max, 65536LL});
    const auto      reach  = [&](double m)
    {
        const long long sz_m = scaled_size(base_data, m, divisor);
        return std::max((2 * sz_m) + 65536LL, sz_m + std::max(scaled_size(base_itv, m, divisor), margin));
    };

    constexpr long long int_max = std::numeric_limits<Int>::max();
    if (reach(multiplier) <= int_max)
    {
        return;
    }

    // Each term of the bound, solved for m (the floors only lower the reach):
    //   2 * sz + 65536   <= int_max  ->  m <= (int_max - 65536) * d / (2 * base_data)
    //   sz + margin      <= int_max  ->  m <= (int_max - margin) * d / base_data
    //   sz + interval    <= int_max  ->  m <= int_max * d / (base_data + base_itv)
    const double d     = static_cast<double>(divisor);
    const double top   = static_cast<double>(int_max); // 2^63 in the int64 build: the loop below re-checks exactly
    double       m_max = std::min({(top - 65536.0) * d / (2.0 * static_cast<double>(base_data)),
                                   (top - static_cast<double>(margin)) * d / static_cast<double>(base_data),
                                   top * d / static_cast<double>(base_data + base_itv)});
    m_max              = std::floor(m_max * 1e6) / 1e6; // printed to 6 decimals: round DOWN
    while (m_max > 0.0 && reach(m_max) > int_max)
    {
        m_max -= 1e-6;
    }

    char buf[64];
    std::snprintf(buf, sizeof buf, "%.6f", m_max);
    std::cerr << "error: --multiply " << real(multiplier) << " --divisor " << divisor << " gives "
              << scaled_size(base_data, multiplier, divisor) << " elements; the bench's Int arithmetic would reach "
              << reach(multiplier) << " > the Int range (" << int_max << ").\n";
    if (m_max > 0.0)
    {
        std::cerr << "       Largest valid --multiply for --divisor " << divisor << " (iter_max " << iter_max
                  << "): " << buf << "\n";
    }
    else
    {
        std::cerr << "       No --multiply fits with --divisor " << divisor << ": raise --divisor.\n";
    }
    std::cerr << "       Beyond that the bench and the library (ind_type = Int) need 64-bit indices:\n"
                 "       rebuild with -DSHRINK_BENCH_INT64=ON and run the int64 executable\n"
                 "       (shrink_bench_<variant>_int64_<1024|dyn>, perf only).\n";
    std::exit(1);
}

/// @brief Iteration count of the run.
/// @param iter_max Iteration count used in perf mode.
/// @return 33 in check mode, @p iter_max in perf mode, -1 with no mode chosen.
inline Int get_iter_stop(Int iter_max)
{
    if (mode_check)
    {
        return 33;
    }
    else if (mode_perf)
    {
        return iter_max;
    }
    return -1;
}

/// @brief Period, in iterations, of the reset step.
/// @param iter_max Iteration count, used in perf mode.
/// @return 4 in check mode, max(iter_max / 10, 1) in perf mode, -1 with no mode chosen.
inline Int get_freq_raz(Int iter_max)
{
    if (mode_check)
    {
        return 4;
    }
    else if (mode_perf)
    {
        Int ret = iter_max / 10;
        if (ret == 0)
        {
            ret = 1;
        }
        return ret;
    }
    return -1;
}

/// @brief Period, in iterations, of the memory line.
/// @return CHUNK_BENCH_MEM_PERIOD when set and positive; otherwise 1 in check mode, 10 in perf mode, -1 with no
///         mode chosen.
inline Int get_freq_conso_mem()
{
    // CHUNK_BENCH_MEM_PERIOD=<n>: print the memory line every n iterations,
    // whatever the mode. =1 turns the ten iterations before an advisor tick
    // from a blind spot into a curve, which is what locating a transient
    // peak needs. Decided once at startup, like every other CONSTEXPR site.
    if (const Int n = shrink_array::env_number<Int>("CHUNK_BENCH_MEM_PERIOD", 0); n > 0)
    {
        return n;
    }
    if (mode_check)
    {
        return 1;
    }
    else if (mode_perf)
    {
        return 10;
    }
    return -1;
}

/// @brief Whether statistics are printed: true in check mode only.
inline bool have_stats()
{
    if (mode_check)
    {
        return true;
    }
    else if (mode_perf)
    {
        return false;
    }
    return false;
}

/// @brief Whether checksums are computed: same answer as have_stats().
inline bool have_check_sum()
{
    return have_stats();
}