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
#include <cmath>     // std::round, std::sin (array_quant)
#include <chrono>    // std::chrono::steady_clock
#include <print>     // std::println

#include "bench_phases.h" // bench_phase_fill (defined here), BenchArrays

void bench_phase_fill(BenchArrays&                           bench_arr,
                      std::chrono::steady_clock::time_point& start,
                      std::chrono::steady_clock::time_point& end,
                      std::chrono::duration<double>&         elapsed_seconds)
{
    // Short local names for the arrays, shared by every phase file.
    auto& arr1 = bench_arr.arr1;
    auto& arr2 = bench_arr.arr2;
    auto& arr3 = bench_arr.arr3;
    auto& arr4 = bench_arr.arr4;
    auto& arr5 = bench_arr.arr5;

    auto& array_int       = bench_arr.array_int;
    auto& array_bool      = bench_arr.array_bool;
    auto& array_bool_bits = bench_arr.array_bool_bits;
    auto& array_complex   = bench_arr.array_complex;
    auto& array_float     = bench_arr.array_float;
    auto& array_short     = bench_arr.array_short;
    auto& array_char      = bench_arr.array_char;
    auto& array_long      = bench_arr.array_long;
    auto& array_ulong     = bench_arr.array_ulong;
    auto& array_quant     = bench_arr.array_quant;

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
                        arr1[i] = second_value_d;
                        arr2[i] = std::sin(i);
                        arr3[i] = 0;
                    }
                    else
                    {
                        arr1[i] = 0;
                        arr2[i] = 0;
                        arr3[i] = std::sin(i);
                    }

                    arr4[i] = 0;
                    arr5[i] = 0;

                    array_float[i] = 0;
                    array_char[i]  = 0;

                    array_short[i] = 0;

                    array_int[i] = 0;

                    array_bool[i]      = false;
                    array_bool_bits[i] = false;
                    array_complex[i]   = zero_complex;
                },
                bench_par::nowait);

            bench_par::for_static_vec(team, 256, sz / 4,
                                      [&](const Int i) __attribute__((always_inline)) { array_short[i] = 0; });

            bench_par::for_static_vec(team, sz / 4, sz / 2, [&](const Int i) __attribute__((always_inline))
                                      { array_short[i] = second_value_s0; });

            team.for_static<Int>(
                256, sz / 2,
                [&](const Int i) __attribute__((always_inline))
                {
                    arr1[i] = second_value_d;
                    arr2[i] = second_value_d;
                    arr3[i] = 0;
                    arr4[i] = 0;
                    arr5[i] = 0;

                    array_int[i]       = 0;
                    array_bool[i]      = false;
                    array_bool_bits[i] = false;

                    array_float[i] = 0;
                    array_char[i]  = 0;

                    array_complex[i] = zero_complex;
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
                    arr1[i] = second_value_d;
                    arr2[i] = 0;
                    arr3[i] = second_value_d;

                    arr4[i] = second_value_d1;
                    arr5[i] = second_value_d2;

                    array_int[i]       = second_value_i;
                    array_bool[i]      = second_value_b;
                    array_bool_bits[i] = second_value_b;

                    array_complex[i] = second_value_c;

                    array_long[i]  = second_value_l;
                    array_ulong[i] = second_value_ul;
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
                    array_quant[i] =
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
                    arr1[i] = second_value_d0;
                    arr2[i] = second_value_d0;
                    arr3[i] = second_value_d0;

                    arr4[i]        = 2 * second_value_d0;
                    arr5[i]        = 3 * second_value_d0;
                    array_float[i] = 3 * second_value_f0;

                    array_int[i]       = other_second_value_i;
                    array_bool[i]      = second_value_b;
                    array_bool_bits[i] = second_value_b;

                    array_complex[i] = other_second_value_c;

                    array_char[i] = 2 * second_value_c0;
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
                    arr1[i] = 2;
                    arr2[i] = 2;
                    arr3[i] = 1;
                    arr4[i] = 1;
                    arr5[i] = second_value_d2 + 10;

                    arr1[i] *= 2;
                    arr2[i] /= 2;
                    arr3[i] += 1;

                    arr4[i] -= 1;
                    arr5[i] -= 10;
                    array_char[i] += 1;
                    array_float[i] = second_value_f0 + 1;

                    array_int[i]       = 0;
                    array_bool[i]      = false;
                    array_bool_bits[i] = false;
                    array_complex[i]   = ComplexDouble(1, 2);
                },
                bench_par::nowait);

            begin_i = end_i;
            end_i   = std::min(begin_i + interval, sz);

            team.for_static<Int>(
                begin_i, end_i,
                [&](const Int i) __attribute__((always_inline))
                {
                    arr1[i] = 1;
                    arr2[i] = 2;
                    arr3[i] = 2;

                    arr4[i]            = second_value_d1;
                    arr5[i]            = second_value_d2;
                    array_char[i]      = second_value_c1;
                    array_float[i]     = second_value_f1;
                    array_int[i]       = 0;
                    array_bool[i]      = false;
                    array_bool_bits[i] = false;
                    array_complex[i]   = zero_complex;
                },
                bench_par::nowait);

            team.for_static<Int>(
                end_i, sz,
                [&](const Int i) __attribute__((always_inline))
                {
                    arr1[i] = 0;
                    arr2[i] = 0;
                    arr3[i] = 0;
                    arr4[i] = 0;
                    arr5[i] = 0;

                    array_int[i]       = 0;
                    array_bool[i]      = false;
                    array_bool_bits[i] = false;
                    array_complex[i]   = zero_complex;
                },
                bench_par::nowait);

            // Scalar: LLVM reports this one "uncountable" under `simd`, with the
            // char store alone as well (the short fills above vectorise). A
            // zeroing loop the idiom pass turns into memset anyway.
            team.for_static<Int>(
                end_i, 2 * sz,
                [&](const Int i) __attribute__((always_inline))
                {
                    array_char[i]  = 0;
                    array_float[i] = 0;
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
