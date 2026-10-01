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

#include <algorithm> // std::min (bench_phase_range_mt)
#include <cstddef>   // std::size_t (bench_phase_range_mt)
#include <print>

#include "bench_phases.h"

void bench_phase_range_a(BenchArrays& bench_arr, Int beg, Int i)
{
    // Aliases so the body below is VERBATIM (see bench_phase_fill.cpp).
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

#pragma omp parallel
    {
        begin_parallel_phase_g();
        BENCH_OMP_FOR_STATIC_NOWAIT_SCALAR
        for (Int j = beg; j < i; ++j)
        {
            tab1[j] = 0;
            tab2[j] = second_value_d;
            tab3[j] = 0;
            tab4[j] = 0;
            tab5[j] = second_value_d2;

            tab_float[j] = second_value_f0;
            tab_char[j]  = 0;
            tab_short[j] = 0;

            tab_int[j]       = 0;
            tab_bool[j]      = false;
            tab_bool_bits[j] = false;

            tab_complex[j] = zero_complex;

            tab_long[j]  = 0L;
            tab_ulong[j] = 0UL;
        }

        finalize_parallel_phase_only_flush_g();
    }

    finalize_parallel_phase_after_flush_g();
}

void bench_phase_range_b(BenchArrays& bench_arr, Int beg, Int i)
{
    // Aliases so the body below is VERBATIM (see bench_phase_fill.cpp).
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

#pragma omp parallel
    {
        begin_parallel_phase_g();
        BENCH_OMP_FOR_STATIC_NOWAIT_SCALAR
        for (Int j = beg; j < i; ++j)
        {
            tab1[j] = second_value_d;

            tab2[j] = second_value_d;
            tab3[j] = second_value_d;

            tab4[j] = second_value_d1;
            tab5[j] = second_value_d2;

            tab_float[j] = second_value_f1;
            tab_short[j] = second_value_s1;
            tab_char[j]  = second_value_c1;

            tab_int[j]       = second_value_i;
            tab_bool[j]      = second_value_b;
            tab_bool_bits[j] = second_value_b;

            tab_complex[j] = second_value_c;

            tab_long[j]  = second_value_l;
            tab_ulong[j] = second_value_ul;
        }

        finalize_parallel_phase_only_flush_g();
    }

    finalize_parallel_phase_after_flush_g();
}

void bench_phase_range_c(BenchArrays& bench_arr, Int beg, Int seg_end)
{
    // Aliases so the body below is VERBATIM (see bench_phase_fill.cpp).
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

#pragma omp parallel
    {
        begin_parallel_phase_g();
        BENCH_OMP_FOR_STATIC_NOWAIT_SCALAR
        for (Int j = beg; j < seg_end; ++j)
        {
            tab1[j] = second_value_d0;

            tab2[j] = second_value_d0;
            tab3[j] = second_value_d0;
            tab4[j] = 2 * second_value_d0;
            tab5[j] = 3 * second_value_d0;

            tab_float[j] = second_value_f0;
            tab_short[j] = 3 * second_value_s0;
            tab_char[j]  = 4 * second_value_c0;

            tab_int[j]       = other_second_value_i;
            tab_bool[j]      = second_value_b;
            tab_bool_bits[j] = second_value_b;

            tab_complex[j] = other_second_value_c;

            tab_long[j]  = 2L * second_value_l;
            tab_ulong[j] = 2UL * second_value_ul;
        }

        finalize_parallel_phase_only_flush_g();
    }

    finalize_parallel_phase_after_flush_g();
}

void bench_phase_range_d(BenchArrays& bench_arr, Int beg, Int seg_end)
{
    // Aliases so the body below is VERBATIM (see bench_phase_fill.cpp).
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

#pragma omp parallel
    {
        begin_parallel_phase_g();
        BENCH_OMP_FOR_STATIC_NOWAIT_SCALAR
        for (Int j = beg; j < seg_end; ++j)
        {
            tab1[j] = second_value_d;
            tab2[j] = second_value_d;
            tab3[j] = second_value_d;
            tab4[j] = 2 * second_value_d0;
            tab5[j] = 3 * second_value_d0;

            tab_float[j] = second_value_f0;
            tab_short[j] = 3 * second_value_s0;
            tab_char[j]  = 4 * second_value_c0;

            tab_int[j]       = second_value_i;
            tab_bool[j]      = second_value_b;
            tab_bool_bits[j] = second_value_b;

            tab_complex[j] = second_value_c;

            tab_long[j]  = second_value_l;
            tab_ulong[j] = second_value_ul;
        }

        finalize_parallel_phase_only_flush_g();
    }

    finalize_parallel_phase_after_flush_g();
}

void bench_phase_raz_sweep(BenchArrays& bench_arr, Int k, Int size, Int start0, Int end0)
{
    // Aliases so the body below is VERBATIM (see bench_phase_fill.cpp). Only
    // tab1 is scanned here; the four range fills reach the rest through `a`,
    // which the moved code calls bench_arrays.
    const auto& tab1         = bench_arr.tab1;
    const Int   sz           = bench_arr.sz;
    auto&       bench_arrays = bench_arr;

    std::println("interval : [{}, {}] size : {}", start0, end0, size);

    Int i = start0;

    for (; i < sz; ++i)
    {
        if (tab1[i] != 0)
        {
            break;
        }
    }

    const Int deb = i;

    if (k % 4 == 0)
    {
        if (k % 8 == 0)
        {
            const Int beg = i;
            while (i < sz && tab1[i] != 0 && tab1[i] != second_value_d)
            {
                i++;
            }

            bench_phase_range_a(bench_arrays, beg, i);
            std::println("set : {} values to : 0", i - beg);
        }
        else if (k % 16 == 0)
        {
            const Int beg = i;
            while (i < sz && tab1[i] != 0 && tab1[i] != second_value_d)
            {
                i++;
            }

            bench_phase_range_b(bench_arrays, beg, i);

            std::println("set : {} values to : {:g}", i - beg, second_value_d);
        }
        else
        {
            const Int beg = i;
            while (i < sz && tab1[i] != 0 && tab1[i] != second_value_d0)
            {
                i++;
            }
            const Int seg_end = i;

            bench_phase_range_c(bench_arrays, beg, seg_end);

            std::println("set : {} values to : {:g}", seg_end - beg, second_value_d0);
        }
    }
    else
    {
        const Int beg = i;
        while (i < sz && tab1[i] != 0)
        {
            i++;
        }

        const Int seg_end = i;

        bench_phase_range_d(bench_arrays, beg, seg_end);

        std::println("set : {} values to : {:g}", seg_end - deb, second_value_d);
    }
}

void bench_phase_range_mt(BenchArrays& bench_arr, Int k, Int beg_s, Int end_s)
{
    auto& tab1          = bench_arr.tab1;
    auto& tab2          = bench_arr.tab2;
    auto& tab3          = bench_arr.tab3;
    auto& tab_int       = bench_arr.tab_int;
    auto& tab_bool      = bench_arr.tab_bool;
    auto& tab_bool_bits = bench_arr.tab_bool_bits;
    auto& tab_complex   = bench_arr.tab_complex;
    auto& tab_float     = bench_arr.tab_float;
    auto& tab_long      = bench_arr.tab_long;

    if (end_s <= beg_s)
    {
        return;
    }

    // Smaller than a chunk at every geometry: several threads per chunk.
    constexpr Int kBlk = 64;
    const Int     nblk = (end_s - beg_s + kBlk - 1) / kBlk;
    // Every other run swaps the constants, so the same chunks go from one
    // constant to the other (and back) across the run.
    const bool swap = ((k / 5) & 1) != 0;

#pragma omp parallel
    {
        begin_parallel_phase_g();
#pragma omp for schedule(static, 1) nowait
        for (Int b = 0; b < nblk; ++b)
        {
            const auto lo = static_cast<std::size_t>(beg_s + (b * kBlk));
            const auto hi = static_cast<std::size_t>(std::min(beg_s + ((b + 1) * kBlk), end_s));

            // shrink_array:: spelled out: this file has no using-directive, and
            // the std::vector baseline resolves to vector_range_ops.h.
            shrink_array::fill_range_mt(tab1, lo, hi, swap ? 0. : second_value_d);
            shrink_array::fill_range_mt(tab2, lo, hi, swap ? second_value_d : 0.);
            // A plain value alternating between neighbouring blocks: chunks
            // stay materialised, and are NOT queued (they cannot be uniform).
            shrink_array::fill_range_mt(tab3, lo, hi, (b & 1) ? 3.5 : 2.5);
            shrink_array::fill_range_mt(tab_int, lo, hi, swap ? 0 : second_value_i);
            shrink_array::fill_range_mt(tab_float, lo, hi, swap ? second_value_f1 : second_value_f0);
            shrink_array::fill_range_mt(tab_long, lo, hi, swap ? 0L : second_value_l);
            shrink_array::fill_range_mt(tab_complex, lo, hi, swap ? zero_complex : second_value_c);
            // bool: the element path on every variant; tab_bool_bits is
            // bit-packed in a chunk2 build. The baseline's two are
            // std::vector<short>.
            shrink_array::fill_range_mt(tab_bool, lo, hi, !swap);
            shrink_array::fill_range_mt(tab_bool_bits, lo, hi, !swap);
        }

        finalize_parallel_phase_only_flush_g();
    }

    finalize_parallel_phase_after_flush_g();
}
