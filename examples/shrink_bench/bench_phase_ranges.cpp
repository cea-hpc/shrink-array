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

/// @file bench_phase_ranges.cpp
/// @brief The four sibling range-fill regions of the re-chunk sweep.
///
/// Four `omp parallel` regions of ~28 lines, near-identical in shape: write a
/// fixed pattern over [beg, end) across the thirteen arrays. They differ only by
/// the values written and by which bound closes the range. Inline in main() they
/// read as four anonymous walls of assignments; here they are four named calls.
/// Bodies are VERBATIM, bound names included -- hence the parameter called `i`
/// on two of them.

#include <complex>
#include <iostream>

#include "bench_phases.h"

void bench_phase_range_a(BenchArrays& bench_arr, int beg, int i)
{
    // Aliases so the body below is VERBATIM (see bench_phase_fill.cpp).
    auto& tab1 = bench_arr.tab1;
    auto& tab2 = bench_arr.tab2;
    auto& tab3 = bench_arr.tab3;
    auto& tab4 = bench_arr.tab4;
    auto& tab5 = bench_arr.tab5;

    auto& tab_int     = bench_arr.tab_int;
    auto& tab_bool    = bench_arr.tab_bool;
    auto& tab_complex = bench_arr.tab_complex;
    auto& tab_float   = bench_arr.tab_float;
    auto& tab_short   = bench_arr.tab_short;
    auto& tab_char    = bench_arr.tab_char;
    auto& tab_long    = bench_arr.tab_long;
    auto& tab_ulong   = bench_arr.tab_ulong;

#pragma omp parallel
    {
        begin_parallel_phase_g();
BENCH_OMP_FOR_STATIC_NOWAIT_SCALAR
        for (int j = beg; j < i; ++j)
        {
            tab1[j] = 0;
            tab2[j] = second_value_d;
            tab3[j] = 0;
            tab4[j] = 0;
            tab5[j] = second_value_d2;

            tab_float[j] = second_value_f0;
            tab_char[j]  = 0;
            tab_short[j] = 0;

            tab_int[j]  = 0;
            tab_bool[j] = false;

            tab_complex[j] = zero_complex;

            tab_long[j]  = 0L;
            tab_ulong[j] = 0UL;
        }

        finalize_parallel_phase_only_flush_g();
    }

    finalize_parallel_phase_after_flush_g();
}

void bench_phase_range_b(BenchArrays& bench_arr, int beg, int i)
{
    // Aliases so the body below is VERBATIM (see bench_phase_fill.cpp).
    auto& tab1 = bench_arr.tab1;
    auto& tab2 = bench_arr.tab2;
    auto& tab3 = bench_arr.tab3;
    auto& tab4 = bench_arr.tab4;
    auto& tab5 = bench_arr.tab5;

    auto& tab_int     = bench_arr.tab_int;
    auto& tab_bool    = bench_arr.tab_bool;
    auto& tab_complex = bench_arr.tab_complex;
    auto& tab_float   = bench_arr.tab_float;
    auto& tab_short   = bench_arr.tab_short;
    auto& tab_char    = bench_arr.tab_char;
    auto& tab_long    = bench_arr.tab_long;
    auto& tab_ulong   = bench_arr.tab_ulong;

#pragma omp parallel
    {
        begin_parallel_phase_g();
BENCH_OMP_FOR_STATIC_NOWAIT_SCALAR
        for (int j = beg; j < i; ++j)
        {
            tab1[j] = second_value_d;

            tab2[j] = second_value_d;
            tab3[j] = second_value_d;

            tab4[j] = second_value_d1;
            tab5[j] = second_value_d2;

            tab_float[j] = second_value_f1;
            tab_short[j] = second_value_s1;
            tab_char[j]  = second_value_c1;

            tab_int[j]  = second_value_i;
            tab_bool[j] = second_value_b;

            tab_complex[j] = second_value_c;

            tab_long[j]  = second_value_l;
            tab_ulong[j] = second_value_ul;
        }

        finalize_parallel_phase_only_flush_g();
    }

    finalize_parallel_phase_after_flush_g();
}

void bench_phase_range_c(BenchArrays& bench_arr, int beg, int seg_end)
{
    // Aliases so the body below is VERBATIM (see bench_phase_fill.cpp).
    auto& tab1 = bench_arr.tab1;
    auto& tab2 = bench_arr.tab2;
    auto& tab3 = bench_arr.tab3;
    auto& tab4 = bench_arr.tab4;
    auto& tab5 = bench_arr.tab5;

    auto& tab_int     = bench_arr.tab_int;
    auto& tab_bool    = bench_arr.tab_bool;
    auto& tab_complex = bench_arr.tab_complex;
    auto& tab_float   = bench_arr.tab_float;
    auto& tab_short   = bench_arr.tab_short;
    auto& tab_char    = bench_arr.tab_char;
    auto& tab_long    = bench_arr.tab_long;
    auto& tab_ulong   = bench_arr.tab_ulong;

#pragma omp parallel
    {
        begin_parallel_phase_g();
BENCH_OMP_FOR_STATIC_NOWAIT_SCALAR
        for (int j = beg; j < seg_end; ++j)
        {
            tab1[j] = second_value_d0;

            tab2[j] = second_value_d0;
            tab3[j] = second_value_d0;
            tab4[j] = 2 * second_value_d0;
            tab5[j] = 3 * second_value_d0;

            tab_float[j] = second_value_f0;
            tab_short[j] = 3 * second_value_s0;
            tab_char[j]  = 4 * second_value_c0;

            tab_int[j]  = other_second_value_i;
            tab_bool[j] = second_value_b;

            tab_complex[j] = other_second_value_c;

            tab_long[j]  = 2L * second_value_l;
            tab_ulong[j] = 2UL * second_value_ul;
        }

        finalize_parallel_phase_only_flush_g();
    }

    finalize_parallel_phase_after_flush_g();
}

void bench_phase_range_d(BenchArrays& bench_arr, int beg, int seg_end)
{
    // Aliases so the body below is VERBATIM (see bench_phase_fill.cpp).
    auto& tab1 = bench_arr.tab1;
    auto& tab2 = bench_arr.tab2;
    auto& tab3 = bench_arr.tab3;
    auto& tab4 = bench_arr.tab4;
    auto& tab5 = bench_arr.tab5;

    auto& tab_int     = bench_arr.tab_int;
    auto& tab_bool    = bench_arr.tab_bool;
    auto& tab_complex = bench_arr.tab_complex;
    auto& tab_float   = bench_arr.tab_float;
    auto& tab_short   = bench_arr.tab_short;
    auto& tab_char    = bench_arr.tab_char;
    auto& tab_long    = bench_arr.tab_long;
    auto& tab_ulong   = bench_arr.tab_ulong;

#pragma omp parallel
    {
        begin_parallel_phase_g();
BENCH_OMP_FOR_STATIC_NOWAIT_SCALAR
        for (int j = beg; j < seg_end; ++j)
        {
            tab1[j] = second_value_d;
            tab2[j] = second_value_d;
            tab3[j] = second_value_d;
            tab4[j] = 2 * second_value_d0;
            tab5[j] = 3 * second_value_d0;

            tab_float[j] = second_value_f0;
            tab_short[j] = 3 * second_value_s0;
            tab_char[j]  = 4 * second_value_c0;

            tab_int[j]  = second_value_i;
            tab_bool[j] = second_value_b;

            tab_complex[j] = second_value_c;

            tab_long[j]  = second_value_l;
            tab_ulong[j] = second_value_ul;
        }

        finalize_parallel_phase_only_flush_g();
    }

    finalize_parallel_phase_after_flush_g();
}

void bench_phase_raz_sweep(BenchArrays& bench_arr, int k, int size, int start0, int end0)
{
    // Aliases so the body below is VERBATIM (see bench_phase_fill.cpp). Only
    // tab1 is scanned here; the four range fills reach the rest through `a`,
    // which the moved code calls bench_arrays.
    const auto& tab1         = bench_arr.tab1;
    const int   sz           = bench_arr.sz;
    auto&       bench_arrays = bench_arr;

    std::cout << "interval " << ": [" << start0 << ", " << end0 << "]" << " size : " << size << "\n";

    int i;

    for (i = start0; i < sz; ++i)
    {
        if (tab1[i] != 0)
        {
            break;
        }
    }

    const int deb = i;

    if (k % 4 == 0)
    {
        if (k % 8 == 0)
        {
            const int beg = i;
            while (i < sz && tab1[i] != 0 && tab1[i] != second_value_d)
            {
                i++;
            }

            bench_phase_range_a(bench_arrays, beg, i);
            std::cout << "set : " << i - beg << " values to : 0" << "\n";
        }
        else if (k % 16 == 0)
        {
            const int beg = i;
            while (i < sz && tab1[i] != 0 && tab1[i] != second_value_d)
            {
                i++;
            }

            bench_phase_range_b(bench_arrays, beg, i);

            std::cout << "set : " << i - beg << " values to : " << second_value_d << "\n";
        }
        else
        {
            const int beg = i;
            while (i < sz && tab1[i] != 0 && tab1[i] != second_value_d0)
            {
                i++;
            }
            const int seg_end = i;

            bench_phase_range_c(bench_arrays, beg, seg_end);

            std::cout << "set : " << seg_end - beg << " values to : " << second_value_d0 << "\n";
        }
    }
    else
    {
        const int beg = i;
        while (i < sz && tab1[i] != 0)
        {
            i++;
        }

        const int seg_end = i;

        bench_phase_range_d(bench_arrays, beg, seg_end);

        std::cout << "set : " << seg_end - deb << " values to : " << second_value_d << "\n";
    }
}
