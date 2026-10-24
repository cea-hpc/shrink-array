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

/// @file bench_phase_fill.cpp
/// @brief The initial fill phase. See bench_phases.h for why it lives here.

#include <algorithm> // std::min
#include <cmath>     // std::round, std::sin (tab_quant)
#include <chrono>    // std::chrono::steady_clock
#include <print>     // std::println

#include "bench_phases.h" // bench_phase_fill (defined here), BenchArrays

void bench_phase_fill(BenchArrays&                           bench_arr,
                      std::chrono::steady_clock::time_point& start,
                      std::chrono::steady_clock::time_point& end,
                      std::chrono::duration<double>&         elapsed_seconds)
{
    // Local aliases so the body below is VERBATIM -- same names, same lines,
    // same comments as when it sat in main(). A behaviour difference is
    // therefore an extraction bug, never a rewrite.
    auto& tab1 = bench_arr.tab1;
    auto& tab2 = bench_arr.tab2;
    auto& tab3 = bench_arr.tab3;
    auto& tab4 = bench_arr.tab4;
    auto& tab5 = bench_arr.tab5;

    auto& tab_int       = bench_arr.tab_int;
    auto& tab_bool      = bench_arr.tab_bool;
    auto& tab_bool_bits = bench_arr.tab_bool_bits;
    auto& tab_complex   = bench_arr.tab_complex;
    auto& tab_float     = bench_arr.tab_float;
    auto& tab_short     = bench_arr.tab_short;
    auto& tab_char      = bench_arr.tab_char;
    auto& tab_long      = bench_arr.tab_long;
    auto& tab_ulong     = bench_arr.tab_ulong;
    auto& tab_quant     = bench_arr.tab_quant;

    const Int sz       = bench_arr.sz;
    const Int interval = bench_arr.interval;

    bench_par::region(
        [&]([[maybe_unused]] bench_par::Team& team)
        {
            begin_parallel_phase_g();

            team.for_static<Int>(
                0, 256,
                [&](const Int i) __attribute__((always_inline))
                {
                    if (i % 3 == 0)
                    {
                        tab1[i] = second_value_d;
                        tab2[i] = std::sin(i);
                        tab3[i] = 0;
                    }
                    else
                    {
                        tab1[i] = 0;
                        tab2[i] = 0;
                        tab3[i] = std::sin(i);
                    }

                    tab4[i] = 0;
                    tab5[i] = 0;

                    tab_float[i] = 0;
                    tab_char[i]  = 0;

                    tab_short[i] = 0;

                    tab_int[i] = 0;

                    tab_bool[i]      = false;
                    tab_bool_bits[i] = false;
                    tab_complex[i]   = zero_complex;
                },
                bench_par::nowait);

            bench_par::for_static_vec(team, 256, sz / 4,
                                      [&](const Int i) __attribute__((always_inline)) { tab_short[i] = 0; });

            bench_par::for_static_vec(team, sz / 4, sz / 2, [&](const Int i) __attribute__((always_inline))
                                      { tab_short[i] = second_value_s0; });

            team.for_static<Int>(
                256, sz / 2,
                [&](const Int i) __attribute__((always_inline))
                {
                    tab1[i] = second_value_d;
                    tab2[i] = second_value_d;
                    tab3[i] = 0;
                    tab4[i] = 0;
                    tab5[i] = 0;

                    tab_int[i]       = 0;
                    tab_bool[i]      = false;
                    tab_bool_bits[i] = false;

                    tab_float[i] = 0;
                    tab_char[i]  = 0;

                    tab_complex[i] = zero_complex;
                },
                bench_par::nowait);

            Int begin_i = sz / 2;
            Int end_i   = sz - (sz / 3);

            team.single(
                [&]
                {
                    end             = std::chrono::steady_clock::now();
                    elapsed_seconds = end - start;
                    start           = end;
                    std::println("\nElapsed time init zero loops : {:.6f}s", elapsed_seconds.count());
                },
                bench_par::nowait);

            team.for_static<Int>(
                begin_i, end_i,
                [&](const Int i) __attribute__((always_inline))
                {
                    tab1[i] = second_value_d;
                    tab2[i] = 0;
                    tab3[i] = second_value_d;

                    tab4[i] = second_value_d1;
                    tab5[i] = second_value_d2;

                    tab_int[i]       = second_value_i;
                    tab_bool[i]      = second_value_b;
                    tab_bool_bits[i] = second_value_b;

                    tab_complex[i] = second_value_c;

                    tab_long[i]  = second_value_l;
                    tab_ulong[i] = second_value_ul;
                    // One level per BAND of bench_quant_band elements, spread over the
                    // array: this is the line that exercises the registry, at up to
                    // 2 * bench_double_bands + 1 distinct second values instead of the
                    // one every other array declares. The level must be constant over
                    // a whole chunk: sin(i) of the element index changes at every
                    // element (sin of consecutive integers jumps around), so no chunk
                    // was ever constant and the registry saw nothing from this array.
                    // A fixed band, not chunk_size(): the same data in every build
                    // (std::vector reference, 64 / 256 / 1024, dyn), so the checksums
                    // still compare; 1024 is a multiple of each of those geometries.
                    constexpr Int bench_quant_band = 1024;
                    tab_quant[i] =
                        std::round(std::sin(static_cast<double>(i / bench_quant_band)) * bench_double_bands) /
                        bench_double_bands;
                },
                bench_par::nowait);

            begin_i = end_i;
            end_i   = sz - (sz / 4);

            team.for_static<Int>(
                begin_i, end_i,
                [&](const Int i) __attribute__((always_inline))
                {
                    tab1[i] = second_value_d0;
                    tab2[i] = second_value_d0;
                    tab3[i] = second_value_d0;

                    tab4[i]      = 2 * second_value_d0;
                    tab5[i]      = 3 * second_value_d0;
                    tab_float[i] = 3 * second_value_f0;

                    tab_int[i]       = other_second_value_i;
                    tab_bool[i]      = second_value_b;
                    tab_bool_bits[i] = second_value_b;

                    tab_complex[i] = other_second_value_c;

                    tab_char[i] = 2 * second_value_c0;
                },
                bench_par::nowait);

            team.single(
                [&]
                {
                    end             = std::chrono::steady_clock::now();
                    elapsed_seconds = end - start;
                    start           = end;
                    std::println("\nElapsed time init 2nd values loops : {:.6f}s", elapsed_seconds.count());
                },
                bench_par::nowait);

            begin_i = end_i;
            end_i   = std::min(begin_i + interval, sz);

            team.for_static<Int>(
                begin_i, end_i,
                [&](const Int i) __attribute__((always_inline))
                {
                    tab1[i] = 2;
                    tab2[i] = 2;
                    tab3[i] = 1;
                    tab4[i] = 1;
                    tab5[i] = second_value_d2 + 10;

                    tab1[i] *= 2;
                    tab2[i] /= 2;
                    tab3[i] += 1;

                    tab4[i] -= 1;
                    tab5[i] -= 10;
                    tab_char[i] += 1;
                    tab_float[i] = second_value_f0 + 1;

                    tab_int[i]       = 0;
                    tab_bool[i]      = false;
                    tab_bool_bits[i] = false;
                    tab_complex[i]   = ComplexDouble(1, 2);
                },
                bench_par::nowait);

            begin_i = end_i;
            end_i   = std::min(begin_i + interval, sz);

            team.for_static<Int>(
                begin_i, end_i,
                [&](const Int i) __attribute__((always_inline))
                {
                    tab1[i] = 1;
                    tab2[i] = 2;
                    tab3[i] = 2;

                    tab4[i]          = second_value_d1;
                    tab5[i]          = second_value_d2;
                    tab_char[i]      = second_value_c1;
                    tab_float[i]     = second_value_f1;
                    tab_int[i]       = 0;
                    tab_bool[i]      = false;
                    tab_bool_bits[i] = false;
                    tab_complex[i]   = zero_complex;
                },
                bench_par::nowait);

            team.for_static<Int>(
                end_i, sz,
                [&](const Int i) __attribute__((always_inline))
                {
                    tab1[i] = 0;
                    tab2[i] = 0;
                    tab3[i] = 0;
                    tab4[i] = 0;
                    tab5[i] = 0;

                    tab_int[i]       = 0;
                    tab_bool[i]      = false;
                    tab_bool_bits[i] = false;
                    tab_complex[i]   = zero_complex;
                },
                bench_par::nowait);

            // Scalar: LLVM reports this one "uncountable" under `simd`, with the
            // char store alone as well (the short fills above vectorise). A
            // zeroing loop the idiom pass turns into memset anyway.
            team.for_static<Int>(
                end_i, 2 * sz,
                [&](const Int i) __attribute__((always_inline))
                {
                    tab_char[i]  = 0;
                    tab_float[i] = 0;
                },
                bench_par::nowait);

            team.single(
                [&]
                {
                    end             = std::chrono::steady_clock::now();
                    elapsed_seconds = end - start;
                    start           = end;
                    std::println("\nElapsed time other loops : {:.6f}s", elapsed_seconds.count());
                },
                bench_par::nowait);

            finalize_parallel_phase_only_flush_g();
        });

    finalize_parallel_phase_after_flush_g();
}
