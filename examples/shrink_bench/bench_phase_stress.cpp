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

/// @file bench_phase_stress.cpp
/// @brief The two per-iteration stress regions. See bench_phases.h.

#include <algorithm>
#include <cmath>
#include <complex>
#include <iostream>

#include "bench_phases.h"
#include "bench_utils.h"

void bench_phase_stress_teams(BenchArrays& bench_arr, [[maybe_unused]] int team, int beg_s, int end_s)
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
    auto& tab_long    = bench_arr.tab_long;
    auto& tab_ulong   = bench_arr.tab_ulong;

#pragma omp parallel num_threads(team - 1)
    {
        begin_parallel_phase_g();
BENCH_OMP_FOR_STATIC_NOWAIT_SCALAR
        for (int j = beg_s; j < end_s; ++j)
        {
            tab1[j] = second_value_d;
            tab2[j] = 0;
            tab3[j] = second_value_d;
            tab4[j] = second_value_d1;
            tab5[j] = second_value_d2;

            tab_int[j]     = second_value_i;
            tab_bool[j]    = second_value_b;
            tab_complex[j] = second_value_c;

            tab_long[j]  = second_value_l;
            tab_ulong[j] = second_value_ul;
        }
        finalize_parallel_phase_only_flush_g();
    }
    finalize_parallel_phase_after_flush_g();
}

void bench_phase_stress_dynamic(BenchArrays& bench_arr, [[maybe_unused]] int grain, int beg_s, int end_s)
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
#pragma omp for schedule(dynamic, grain) nowait
        for (int j = beg_s; j < end_s; ++j)
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
