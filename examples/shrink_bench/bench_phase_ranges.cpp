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
/// the values written and by which bound closes the range. Two of them take
/// that bound as `i`.

#include <algorithm> // std::min (bench_phase_range_mt)
#include <cstddef>   // std::size_t (bench_phase_range_mt)
#include <iostream>  // std::cout

#include "bench_phases.h" // bench_phase_range_* (defined here), BenchArrays

void bench_phase_range_a(BenchArrays& bench_arr, Int beg, Int iter)
{
    // Short local names for the arrays (see bench_phase_fill.cpp).
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

    bench_par::region(
        [&]([[maybe_unused]] bench_par::Team& team)
        {
            begin_parallel_phase_g();
            team.for_static<Int>(
                beg, iter,
                [&](const Int j) __attribute__((always_inline))
                {
                    arr1[j] = 0;
                    arr2[j] = second_value_d;
                    arr3[j] = 0;
                    arr4[j] = 0;
                    arr5[j] = second_value_d2;

                    array_float[j] = second_value_f0;
                    array_char[j]  = 0;
                    array_short[j] = 0;

                    array_int[j]       = 0;
                    array_bool[j]      = false;
                    array_bool_bits[j] = false;

                    array_complex[j] = zero_complex;

                    array_long[j]  = 0L;
                    array_ulong[j] = 0UL;
                },
                bench_par::nowait);

            finalize_parallel_phase_only_flush_g();
        });

    finalize_parallel_phase_after_flush_g();
}

void bench_phase_range_b(BenchArrays& bench_arr, Int beg, Int iter)
{
    // Short local names for the arrays (see bench_phase_fill.cpp).
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

    bench_par::region(
        [&]([[maybe_unused]] bench_par::Team& team)
        {
            begin_parallel_phase_g();
            team.for_static<Int>(
                beg, iter,
                [&](const Int j) __attribute__((always_inline))
                {
                    arr1[j] = second_value_d;

                    arr2[j] = second_value_d;
                    arr3[j] = second_value_d;

                    arr4[j] = second_value_d1;
                    arr5[j] = second_value_d2;

                    array_float[j] = second_value_f1;
                    array_short[j] = second_value_s1;
                    array_char[j]  = second_value_c1;

                    array_int[j]       = second_value_i;
                    array_bool[j]      = second_value_b;
                    array_bool_bits[j] = second_value_b;

                    array_complex[j] = second_value_c;

                    array_long[j]  = second_value_l;
                    array_ulong[j] = second_value_ul;
                },
                bench_par::nowait);

            finalize_parallel_phase_only_flush_g();
        });

    finalize_parallel_phase_after_flush_g();
}

void bench_phase_range_c(BenchArrays& bench_arr, Int beg, Int seg_end)
{
    // Short local names for the arrays (see bench_phase_fill.cpp).
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

    bench_par::region(
        [&]([[maybe_unused]] bench_par::Team& team)
        {
            begin_parallel_phase_g();
            team.for_static<Int>(
                beg, seg_end,
                [&](const Int j) __attribute__((always_inline))
                {
                    arr1[j] = second_value_d0;

                    arr2[j] = second_value_d0;
                    arr3[j] = second_value_d0;
                    arr4[j] = 2 * second_value_d0;
                    arr5[j] = 3 * second_value_d0;

                    array_float[j] = second_value_f0;
                    array_short[j] = 3 * second_value_s0;
                    array_char[j]  = 4 * second_value_c0;

                    array_int[j]       = other_second_value_i;
                    array_bool[j]      = second_value_b;
                    array_bool_bits[j] = second_value_b;

                    array_complex[j] = other_second_value_c;

                    array_long[j]  = 2L * second_value_l;
                    array_ulong[j] = 2UL * second_value_ul;
                },
                bench_par::nowait);

            finalize_parallel_phase_only_flush_g();
        });

    finalize_parallel_phase_after_flush_g();
}

void bench_phase_range_d(BenchArrays& bench_arr, Int beg, Int seg_end)
{
    // Short local names for the arrays (see bench_phase_fill.cpp).
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

    bench_par::region(
        [&]([[maybe_unused]] bench_par::Team& team)
        {
            begin_parallel_phase_g();
            team.for_static<Int>(
                beg, seg_end,
                [&](const Int j) __attribute__((always_inline))
                {
                    arr1[j] = second_value_d;
                    arr2[j] = second_value_d;
                    arr3[j] = second_value_d;
                    arr4[j] = 2 * second_value_d0;
                    arr5[j] = 3 * second_value_d0;

                    array_float[j] = second_value_f0;
                    array_short[j] = 3 * second_value_s0;
                    array_char[j]  = 4 * second_value_c0;

                    array_int[j]       = second_value_i;
                    array_bool[j]      = second_value_b;
                    array_bool_bits[j] = second_value_b;

                    array_complex[j] = second_value_c;

                    array_long[j]  = second_value_l;
                    array_ulong[j] = second_value_ul;
                },
                bench_par::nowait);

            finalize_parallel_phase_only_flush_g();
        });

    finalize_parallel_phase_after_flush_g();
}

void bench_phase_reset_sweep(BenchArrays& bench_arr, Int iter, Int size, Int start0, Int end0)
{
    // Short local names for the arrays (see bench_phase_fill.cpp). Only
    // arr1 is scanned here; the four range fills reach the rest through
    // bench_arrays.
    const auto& arr1         = bench_arr.arr1;
    const Int   sz           = bench_arr.sz;
    auto&       bench_arrays = bench_arr;

    std::cout << "interval " << ": [" << start0 << ", " << end0 << "]" << " size : " << size << "\n";

    Int pos = start0;

    for (; pos < sz; ++pos)
    {
        if (arr1[pos] != 0)
        {
            break;
        }
    }

    const Int begin = pos;

    if (iter % 4 == 0)
    {
        if (iter % 8 == 0)
        {
            const Int beg = pos;
            while (pos < sz && arr1[pos] != 0 && arr1[pos] != second_value_d)
            {
                pos++;
            }

            bench_phase_range_a(bench_arrays, beg, pos);
            std::cout << "set : " << pos - beg << " values to : 0" << "\n";
        }
        else if (iter % 16 == 0)
        {
            const Int beg = pos;
            while (pos < sz && arr1[pos] != 0 && arr1[pos] != second_value_d)
            {
                pos++;
            }

            bench_phase_range_b(bench_arrays, beg, pos);

            std::cout << "set : " << pos - beg << " values to : " << second_value_d << "\n";
        }
        else
        {
            const Int beg = pos;
            while (pos < sz && arr1[pos] != 0 && arr1[pos] != second_value_d0)
            {
                pos++;
            }
            const Int seg_end = pos;

            bench_phase_range_c(bench_arrays, beg, seg_end);

            std::cout << "set : " << seg_end - beg << " values to : " << second_value_d0 << "\n";
        }
    }
    else
    {
        const Int beg = pos;
        while (pos < sz && arr1[pos] != 0)
        {
            pos++;
        }

        const Int seg_end = pos;

        bench_phase_range_d(bench_arrays, beg, seg_end);

        std::cout << "set : " << seg_end - begin << " values to : " << second_value_d << "\n";
    }
}

void bench_phase_range_mt(BenchArrays& bench_arr, Int iter, Int beg_s, Int end_s)
{
    auto& arr1            = bench_arr.arr1;
    auto& arr2            = bench_arr.arr2;
    auto& arr3            = bench_arr.arr3;
    auto& array_int       = bench_arr.array_int;
    auto& array_bool      = bench_arr.array_bool;
    auto& array_bool_bits = bench_arr.array_bool_bits;
    auto& array_complex   = bench_arr.array_complex;
    auto& array_float     = bench_arr.array_float;
    auto& array_long      = bench_arr.array_long;

    if (end_s <= beg_s)
    {
        return;
    }

    // Smaller than a chunk at every geometry: several threads per chunk.
    constexpr Int blk  = 64;
    const Int     nblk = (end_s - beg_s + blk - 1) / blk;
    // Every other run swaps the constants, so the same chunks go from one
    // constant to the other (and back) across the run.
    const bool swap = ((iter / 5) & 1) != 0;

    bench_par::region(
        [&]([[maybe_unused]] bench_par::Team& team)
        {
            begin_parallel_phase_g();
            team.for_static_chunks<Int>(
                0, nblk, 1,
                [&](const Int b) __attribute__((always_inline))
                {
                    const auto lo = static_cast<std::size_t>(beg_s + (b * blk));
                    const auto hi = static_cast<std::size_t>(std::min(beg_s + ((b + 1) * blk), end_s));

                    // shrink_array:: spelled out: this file has no using-directive, and
                    // the std::vector baseline resolves to vector_range_ops.h.
                    shrink_array::fill_range_mt(arr1, lo, hi, swap ? 0. : second_value_d);
                    shrink_array::fill_range_mt(arr2, lo, hi, swap ? second_value_d : 0.);
                    // A plain value alternating between neighbouring blocks: chunks
                    // stay materialised, and are NOT queued (they cannot be uniform).
                    shrink_array::fill_range_mt(arr3, lo, hi, (b & 1) ? 3.5 : 2.5);
                    shrink_array::fill_range_mt(array_int, lo, hi, swap ? 0 : second_value_i);
                    shrink_array::fill_range_mt(array_float, lo, hi, swap ? second_value_f1 : second_value_f0);
                    shrink_array::fill_range_mt(array_long, lo, hi, swap ? 0L : second_value_l);
                    shrink_array::fill_range_mt(array_complex, lo, hi, swap ? zero_complex : second_value_c);
                    // bool: the element path on every variant; array_bool_bits is
                    // bit-packed in a zero_or_value build. The baseline's two are
                    // std::vector<short>.
                    shrink_array::fill_range_mt(array_bool, lo, hi, !swap);
                    shrink_array::fill_range_mt(array_bool_bits, lo, hi, !swap);
                },
                bench_par::nowait);

            finalize_parallel_phase_only_flush_g();
        });

    finalize_parallel_phase_after_flush_g();
}
