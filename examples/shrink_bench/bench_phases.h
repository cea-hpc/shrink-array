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
 * @file bench_phases.h
 * @brief The OMPT phase-transition wrappers, and the bench phases themselves.
 *
 * Shared by the workload's parallel regions, each in its own translation
 * unit: every one opens with begin_parallel_phase_g() and closes with the
 * matching finalize. `inline` (no anonymous namespace) keeps the "no library
 * header change" property across TUs.
 */

#pragma once

#include <chrono> // std::chrono::steady_clock
#include <cmath>  // std::sqrt
#include <vector> // std::vector

#include "bench_array.h" // BenchArrays
#include "bench_par.h"   // bench_par::region_ends_phase

// Guarded inline wrappers around the OMPT phase transitions. Semantics are
// strictly identical to the manual `if constexpr (build_config::has_not_ompt)
// { ... }` blocks they replace: the `if constexpr` is evaluated at compile
// time, so under `has_ompt` the body is elided and the call becomes a no-op
// that the optimizer folds away. Kept in an anonymous namespace so linkage
// is local and no library header change is required.
//
// The TBB bench (bench_par::region_ends_phase): bench_par::region() begins and
// ends the phase itself, as OMPT does; the wrappers are no-ops there too.
inline void begin_parallel_phase_g()
{
    if constexpr (build_config::has_not_ompt && !bench_par::region_ends_phase)
    {
        shrink_array::begin_parallel_phase();
    }
}

/// @brief Without OMPT: shrink_array::finalize_parallel_phase_only_flush(); no-op otherwise.
/// @details OpenMP only: the split end needs a barrier of the application's
///          team, which only OpenMP gives the library (manage_ompt.h). The
///          other backends end the phase after the region, in one call
///          (finalize_parallel_phase_after_flush_g()).
inline void finalize_parallel_phase_only_flush_g()
{
    if constexpr (build_config::has_not_ompt && build_config::use_openmp && !bench_par::region_ends_phase)
    {
        shrink_array::finalize_parallel_phase_only_flush();
    }
}

/// @brief Without OMPT: shrink_array::finalize_parallel_phase_after_flush(); no-op otherwise.
/// @details Std and serial backends: shrink_array::finalize_parallel_phase(),
///          the serial end of the phase contract -- it drains every rank's
///          queue itself, after the region's team has joined.
inline void finalize_parallel_phase_after_flush_g()
{
    if constexpr (bench_par::region_ends_phase)
    {
        // bench_par::region() has ended the phase.
    }
    else if constexpr (build_config::has_not_ompt && build_config::use_openmp)
    {
        shrink_array::finalize_parallel_phase_after_flush();
    }
    else if constexpr (build_config::has_not_ompt)
    {
        shrink_array::finalize_parallel_phase();
    }
}

/**
 * @brief Fill every array with the workload's starting pattern.
 *
 * @param bench_arr the arrays, by reference.
 * @param start the timing cursors main() carries across phases; this phase
 *        stamps them (start, end, elapsed_seconds) from inside an `omp
 *        single`, exactly as it did when it was inline in main().
 * @param end see start.
 * @param elapsed_seconds see start.
 *
 * @details Moved VERBATIM out of main(): one `omp parallel` region holding a
 *          dozen `omp for nowait` loops, 200 lines, the largest block of the
 *          workload after the compute loop. An ORDINARY function compiled once
 *          -- not a template the caller would re-instantiate, which is what
 *          makes the split worth anything for build time.
 */
void bench_phase_fill(BenchArrays&                           bench_arr,
                      std::chrono::steady_clock::time_point& start,
                      std::chrono::steady_clock::time_point& end,
                      std::chrono::duration<double>&         elapsed_seconds);

/// @brief Store sqrt(@p value) at index @p index of @p tab. For a zero::Array, the raw reference tab(index) is used only
///        when the chunk is written, operator[] otherwise.
/// @param tab Destination array.
/// @param index Element index.
/// @param value Value whose square root is stored.
template <class Arr, typename T>
void neutral_sqrt_assign(Arr& tab, Int index, T value)
{
    // zero: the raw reference tab(index) is valid on a MATERIALISED chunk only. The
    // write that computed value (tab2[index] = fmod(...)) can itself have compressed
    // the chunk -- a 0 completing an all-zero chunk, which the serial (counter)
    // build folds at once -- and the raw store then went into the shared
    // constant (harmless for sqrt(0) = 0, an CHUNK_INVARIANT_FAILED in checked
    // builds; seed-dependent). Same value either way, so the reference data
    // do not change.
    if constexpr (chunk_config::use_zero)
    {
        if (tab.written(index))
        {
            tab(index) = std::sqrt(value); // discarded quand Arr=std::vector : never typed
        }
        else
        {
            tab[index] = std::sqrt(value);
        }
    }
    else
    {
        tab[index] = std::sqrt(value);
    }
}

/**
 * @brief One iteration of the main compute loop, over [start0, end0).
 *
 * @param bench_arr the arrays.
 * @param iter the iteration index; the workload varies its pattern with iter % 16.
 * @param start0 start of the window this iteration writes.
 * @param end0 end of that window.
 * @param tab_nba1 the per-iteration thresholds main() precomputed; read at
 *        [iter], never written here.
 * @param tab_nba2 see tab_nba1.
 * @param val_sum the running accumulators (val_sum, long_val_sum, v_real,
 *        v_im), ADDED TO.
 * @param long_val_sum see val_sum.
 * @param v_real see val_sum.
 * @param v_im see val_sum.
 *
 * @details Moved VERBATIM out of main(): 392 lines, six `omp for` loops inside a
 *          single `omp parallel reduction` region -- the largest block of the
 *          bench.
 * @note The reduction is done on zero-initialised LOCALS and added to the
 *       caller's accumulators on return. Strictly the same arithmetic as
 *       reducing straight into them (which is what the inline version did), and
 *       it keeps references out of the reduction clause.
 */
void bench_phase_compute(BenchArrays&            bench_arr,
                         Int                     iter,
                         Int                     start0,
                         Int                     end0,
                         const std::vector<Int>& tab_nba1,
                         const std::vector<Int>& tab_nba2,
                         double&                 val_sum,
                         long&                   long_val_sum,
                         double&                 v_real,
                         double&                 v_im);

/**
 * @brief The struct arrays' starting pattern (bench_phase_struct.cpp): runs of
 *        the null element, of the second value, of another constant, then one
 *        interval of varied elements. Its own region, after bench_phase_fill.
 */
void bench_phase_struct_fill(BenchArrays& bench_arr);

/**
 * @brief The struct arrays' compute window: a run of nulls, a run of the
 *        second value, the rest updated from its value, then a read sweep
 *        into @p point_sum. Same arguments as bench_phase_compute.
 */
void bench_phase_struct_compute(BenchArrays&            bench_arr,
                                Int                     iter,
                                Int                     start0,
                                Int                     end0,
                                const std::vector<Int>& tab_nba1,
                                const std::vector<Int>& tab_nba2,
                                double&                 point_sum);

/// @brief The struct arrays' share of the raz sweep: [start0, end0) back to T{}.
void bench_phase_struct_raz(BenchArrays& bench_arr, Int start0, Int end0);

/**
 * @brief tab_mask's starting pattern (bench_phase_mask.cpp): whole segments of
 *        each constant, of random masks and of scattered flags
 *        (bench_mask_initial). Its own region, after bench_phase_fill.
 */
void bench_phase_mask_fill(BenchArrays& bench_arr);

/**
 * @brief tab_mask's iteration @p iter: flips in one mixed block in 16, a constant
 *        segment in 32 turned whole to the other constant, then a read sweep of
 *        the blocks written into @p mask_count.
 */
void bench_phase_mask_compute(BenchArrays& bench_arr, Int iter, long& mask_count);

/**
 * @brief tab_conn's starting mesh (bench_phase_conn.cpp): every row in its first
 *        numbering. Its own region, after bench_phase_fill.
 */
void bench_phase_conn_fill(BenchArrays& bench_arr);

/**
 * @brief tab_conn's iteration @p iter: band iter % bench_conn_bands read into
 *        @p conn_sum (the gather), then renumbered (its node order rotated, or
 *        put back).
 */
void bench_phase_conn_compute(BenchArrays& bench_arr, Int iter, long& conn_sum);

/**
 * @brief Stress (a): a team one thread SHORT of the default, over [beg_s, end_s).
 * @details Moved verbatim. Runs on ~1/5 of iterations (disjoint k-modulo gate in
 *          main()), enough to surface rare bugs without dominating the baseline.
 */
void bench_phase_stress_teams(BenchArrays& bench_arr, Int team, Int beg_s, Int end_s);

/**
 * @brief Stress (b): `schedule(dynamic, grain)` over [beg_s, end_s).
 * @details Moved verbatim. Scatters iterations so several threads land on the
 *          SAME chunk at once -- it is what exercises the CAS / Dekker paths that
 *          stay idle under the default static schedule.
 */
void bench_phase_stress_dynamic(BenchArrays& bench_arr, Int grain, Int beg_s, Int end_s);

/**
 * @brief Stress (d): fill_range_mt from every thread, over [beg_s, end_s).
 * @details 64-element blocks dealt round-robin, so chunks are shared between
 *          threads; eight arrays, every constant outcome plus a plain value
 *          alternating between neighbouring blocks. @p iter picks which of the
 *          two value sets runs, so chunks go back and forth across the run.
 *          See bench_phase_ranges.cpp.
 */
void bench_phase_range_mt(BenchArrays& bench_arr, Int iter, Int beg_s, Int end_s);

/// @name The four sibling range-fill regions
/// Same shape, different values and closing bound. See bench_phase_ranges.cpp.
/// @{
void bench_phase_range_a(BenchArrays& bench_arr, Int beg, Int iter);
void bench_phase_range_b(BenchArrays& bench_arr, Int beg, Int iter);
void bench_phase_range_c(BenchArrays& bench_arr, Int beg, Int seg_end);
void bench_phase_range_d(BenchArrays& bench_arr, Int beg, Int seg_end);
/// @}

/**
 * @brief Compress all thirteen arrays, one `omp single nowait` each.
 *
 * @note Brackets itself with the OMPT phase update: a caller cannot forget
 *       what it need not write.
 * @details The thirteen compress calls go through the compress_one(auto&)
 *          template, so the body needs no #ifdef. `single nowait` per array
 *          rather than a loop: the arrays have thirteen DIFFERENT types, so
 *          there is nothing to loop over -- the region hands one array to
 *          whichever thread arrives, and nowait lets the next start
 *          immediately.
 * @note Does strictly nothing in the std::vector baseline build.
 */
void bench_phase_compress_all(BenchArrays& bench_arr);

/**
 * @brief The I/O decisions main() takes once, at startup, and that the
 *        verification pass reads.
 *
 * @note no_io and no_compress are the raw `const char*` main() computes from the
 *       flag or the environment; non-null means "on". Kept as-is rather than
 *       normalised to bool, so the moved code stays verbatim.
 */
struct BenchIoFlags
{
    const char* no_io;       ///< non-null: skip file I/O entirely
    const char* no_compress; ///< non-null: write/read uncompressed
    Int         io_res;      ///< get_io_mode(): 0 none, 1 write, 2 read-and-check
    bool        read_trace;
    bool        write_trace;
    bool        checksum_only = false; ///< --checksum_only: write the checksum files alone
};

/**
 * @brief Write or re-read-and-check every array's reference file.
 *
 * @param ind the iteration index, or -1 for the pass that runs before the loop.
 *
 * @details Moved VERBATIM out of main(), and the two lambdas had to travel
 *          together: lbd_check is GENERIC (`const auto& tab`) because it is
 *          called with all thirteen array types, so it cannot cross a
 *          translation-unit boundary on its own. Inside this file it stays a
 *          generic lambda, instantiated once here instead of in main() -- and
 *          the entry point below is an ordinary function.
 */
void bench_check_all(BenchArrays& bench_arr, Int ind, const BenchIoFlags& io);

/**
 * @brief Print the per-array value census and the chunk filling rates.
 *
 * @param stats whether the run collects them at all -- have_stats(), true in
 *        check mode, false in perf. Passed rather than read here, so this file
 *        stays clear of the mode machinery.
 *
 * @details Moved VERBATIM, and again as a GROUP: lbd_stats calls
 *          lbd_fill_values and lbd_fill_chunk, both generic over the array type
 *          since they are applied to all eleven arrays. None of the three could
 *          cross a translation-unit boundary alone. They stay generic lambdas
 *          inside this file; the entry point is an ordinary function.
 */
void bench_stats(BenchArrays& bench_arr, bool stats);

/**
 * @brief The re-chunk sweep: find the next run in tab1, rewrite it.
 *
 * @param bench_arr the arrays.
 * @param iter the iteration index; k % 4, % 8 and % 16 pick which of the four
 *        bench_phase_range_* patterns gets written.
 * @param size the window this iteration works in (size, start0, end0).
 *        Printed.
 * @param start0 where the scan begins -- which then deliberately runs to sz,
 *        not to end0.
 * @param end0 see size.
 *
 * @details Moved VERBATIM out of main(), where it was a seventy-line nest of
 *          modulo tests around four near-identical scans. It is the DISPATCHER
 *          of the four sibling range fills, which is why it lives in their
 *          file: side by side, it becomes visible that the four differ only by
 *          the value written and by the sentinel the scan stops on.
 */
void bench_phase_raz_sweep(BenchArrays& bench_arr, Int iter, Int size, Int start0, Int end0);

/**
 * @brief Print the thirteen checksums.
 * @param check_sum have_check_sum(): false in perf mode, where nothing is read
 *        back and the census would only cost time.
 * @details Moved verbatim, with the variadic print_checksums helper it needs --
 *          which main() had no other use for.
 */
void bench_print_checksums(BenchArrays& bench_arr, bool check_sum);

/**
 * @brief Validate the compiled chunk geometry, then start the library.
 *
 * @details Moved VERBATIM out of main(). Aborts -- deliberately, and loudly --
 *          on a geometry the build cannot honour: a null chunk size, one that
 *          is neither a power of two nor a multiple of 64, or one outside
 *          [size_min, size_max]. Then
 *          ParamChunk::init_indexes(), which must run BEFORE any array exists,
 *          hence the call staying exactly where it was in main().
 * @note Empty in a std::vector build: the whole body sat under #ifdef
 *       SHRINK_ARRAY, and the guard moved with it.
 */
void bench_init_library();

/**
 * @brief Release the arrays, report, and shut the library down.
 *
 * @param test_stl passed straight through to the --test_stl self-checks, which
 *        run just before the teardown proper.
 *
 * @details Moved VERBATIM out of the tail of main(), and the ORDER inside is
 *          load-bearing, which is the reason it travels as one block rather
 *          than as a handful of calls main() could reorder by accident:
 *            1. tab_*.clear() -- clear() runs clear_core(), which is what feeds
 *               the teardown_clear counter;
 *            2. only THEN chunk_stats(), or the counter always reads 0;
 *            3. the compression check, the pool breakdown, and last the
 *               library's own pool release.
 */
void bench_teardown(BenchArrays& bench_arr, bool test_stl);
