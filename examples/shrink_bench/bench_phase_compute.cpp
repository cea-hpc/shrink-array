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

/// @file bench_phase_compute.cpp
/// @brief The main compute loop's parallel region. See bench_phases.h.

#include <cstdlib> // std::getenv
#include <cmath>   // std::cos, std::sin, std::sqrt
#include <complex> // ComplexDouble (std::complex) arithmetic
#include <limits>  // std::numeric_limits
#include <vector>  // std::vector

#include <algorithm> // std::min / std::max (block bounds)

#include "bench_phases.h"             // bench_phase_compute (defined here), BenchArrays
#include "bench_utils.h"              // linear_correlation, exponential_correlation
#include "chunk/internal/env_parse.h" // env_number / arg_number (checked, not atoi)

// Inner loop of the four-accumulator loop: a simd reduction on the std::vector
// baseline (keeps it vectorised), nothing on shrink builds (the read gate has
// side effects). See bench_phase_compute.
#ifdef SHRINK_ARRAY
    #define BENCH_OMP_SIMD_RED(...)
#else
    #define BENCH_OMP_SIMD_RED(...) BENCH_PRAGMA_STR(omp simd reduction(+ : __VA_ARGS__))
#endif

// Distinct second values the unsigned long array may register, as a mask over
// the index. CHUNK_BENCH_ULONG_BANDS sets the count (default 4096, the order a
// real workload reaches); 0 means no bound, i.e. one value per 256-index band
// for the whole array, which is what the registry stress runs want.
static const unsigned long bench_ulong_mask = []
{
    const long bands = shrink_array::env_number<long>("CHUNK_BENCH_ULONG_BANDS", 4096);
    if (bands <= 0)
    {
        return ~0UL; // unbounded
    }
    // Bands are 256 indices wide (the |0xFF, &~0x0F arithmetic below), so the
    // mask has to cover bands * 256 indices, rounded up to a power of two.
    const unsigned long span = static_cast<unsigned long>(bands) * 256UL;
    unsigned long       mask = 1;
    while (mask < span)
    {
        mask <<= 1U;
    }
    return mask - 1UL;
}();

void bench_phase_compute(BenchArrays&            bench_arr,
                         Int                     iter,
                         Int                     start0,
                         Int                     end0,
                         const std::vector<Int>& array_nba1,
                         const std::vector<Int>& array_nba2,
                         double&                 val_sum_out,
                         long&                   long_val_sum_out,
                         double&                 v_real_out,
                         double&                 v_im_out)
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

    const Int sz = bench_arr.sz;

    // Zero-initialised: the reduction accumulates here, then lands on the
    // caller's running totals at the end of the function.
    double val_sum      = 0;
    long   long_val_sum = 0;
    double v_real       = 0;
    double v_im         = 0;

    // The accumulators are SHARED in the region; each loop that adds to them
    // adds a per-thread partial into them atomically at the end of the thread's
    // own portion (BENCH_FOR_*_RED_SCALAR and BENCH_RED, bench_par.h; the
    // four-accumulator loop by hand). A partial of a nowait loop is in at the
    // end of the thread's portion; the value is whole after the region's
    // barrier, which is where it is read.
    // Block size and aligned start of the [start0, end0) loops' schedule
    // (bench_sched_for, bench_prelude.h). Computed once, before the region:
    // every thread must see the same values, or the same-thread guarantee is
    // void. The pragma macros name sched_blk.
    const BenchSched sched      = bench_sched_for(start0, end0);
    const Int        sched_blk  = sched.blk;
    const Int        sched_base = sched.base;

    bench_par::region(
        [&]([[maybe_unused]] bench_par::Team& team)
        {
            begin_parallel_phase_g();

            const Int dec = iter % 16;
            const Int deb = std::max<Int>(start0 - 512, 0);

            // nowait on this loop and the next: they write DISJOINT ranges
            // ([deb, start0) here, [end0, end0 + 256) below) that nothing in this
            // region reads back, so a thread done with its slice may go on. The
            // three loops after them stay barriered: the [start0, end0) loop
            // writes what the reduction loops then read, and the last one
            // rewrites the tail its own p = 0 pass produced.
            // array_complex peeled out of the loop below. std::complex<double> is an
            // interleaved {re, im} pair, so a store over consecutive i needs
            // stride-2 interleaving, which the vectoriser will not build from a
            // value made of two separate libm calls -- and its presence kept the
            // whole loop scalar. This loop is a pure writer: the branches pick WHAT
            // to write from the index alone and nothing is read back, so the same
            // values land at the same indices, and only the iteration order over
            // two independent arrays changes.
            //
            // Scalar (this loop and the next): a few hundred elements, and libm
            // calls inside index-selected branches -- "control flow cannot be
            // substituted for a select", the calls may not be speculated.
            team.for_static<Int>(
                deb, start0,
                [&](const Int i) __attribute__((always_inline))
                {
                    const Int offset = i - deb;

                    if (offset < 60 + dec)
                    {
                        array_complex[i] = ComplexDouble(std::sin(i + iter), std::cos(i + iter));
                    }
                    else if (offset < 120 + dec)
                    {
                        array_complex[i] = zero_complex;
                    }
                    else if (offset < 240 + dec)
                    {
                        array_complex[i] = second_value_c;
                    }
                    else if (offset < 360 + dec)
                    {
                        array_complex[i] = zero_complex;
                    }
                    else
                    {
                        array_complex[i] = second_value_c;
                    }
                },
                bench_par::nowait);

            team.for_static<Int>(
                deb, start0,
                [&](const Int i) __attribute__((always_inline))
                {
                    const Int offset = i - deb;

                    if (offset < 60 + dec)
                    {
                        arr1[i]      = 0;
                        arr2[i]      = second_value_d;
                        arr3[i]      = std::tan(i + iter);
                        arr4[i]      = 0;
                        arr5[i]      = std::sin(i + iter);
                        array_int[i] = 0;

                        array_float[offset] = static_cast<float>(std::cos(i - iter));
                        array_short[offset] = 0;
                        array_char[i]       = 0;
                    }
                    else if (offset < 120 + dec)
                    {
                        arr1[i]             = second_value_d;
                        arr2[i]             = 0;
                        arr3[i]             = 0;
                        arr4[i]             = second_value_d1;
                        arr5[i]             = 0;
                        array_int[i]        = 1;
                        array_float[offset] = 0;
                        array_short[offset] = 1;
                        array_char[i]       = 0;
                    }
                    else if (offset < 240 + dec)
                    {
                        arr1[i]             = 0;
                        arr2[i]             = std::sin(i + iter);
                        arr3[i]             = second_value_d;
                        arr4[i]             = 0;
                        arr5[i]             = second_value_d2;
                        array_int[i]        = static_cast<int>(offset);
                        array_float[offset] = second_value_f0;
                        array_short[offset] = second_value_s0;
                        array_char[i]       = second_value_c0;
                    }
                    else if (offset < 360 + dec)
                    {
                        arr1[i]             = std::cos(i + iter);
                        arr2[i]             = 0;
                        arr3[i]             = 0;
                        arr4[i]             = 0;
                        arr5[i]             = 0;
                        array_int[i]        = static_cast<int>(offset + 1);
                        array_float[offset] = 0;
                        array_short[offset] = 0;
                        array_char[i]       = 0;
                    }
                    else
                    {
                        arr1[i]             = 0;
                        arr2[i]             = second_value_d;
                        arr3[i]             = second_value_d;
                        arr4[i]             = second_value_d1;
                        arr5[i]             = second_value_d2;
                        array_float[offset] = second_value_f1;
                        array_short[offset] = second_value_s2;
                        array_char[i]       = 0;
                        array_int[i]        = 0;
                    }
                },
                bench_par::nowait);

            // No barrier plus flush of the deferred writes (finalize_barrier_phase)
            // after either nowait loop: that is exactly what the OMPT build does at
            // an implicit barrier -- and a nowait loop has none.
            // Both builds now flush these two loops' writes at the next loop's
            // barrier, which is before anything reads them.
            //
            // Scalar: 256 elements, and interpolation() carries early exits
            // ("cannot vectorize early exit loop with more than one early exit").
            team.for_static<Int>(
                end0, std::min(end0 + 256, sz),
                [&](const Int i) __attribute__((always_inline))
                {
                    if (i % 3 == 0)
                    {
                        arr1[i] = second_value_d;
                        arr2[i] = std::sin(i + iter);
                        arr3[i] = 0;
                        arr4[i] = second_value_d1;
                        arr5[i] = std::cos(i - iter);

                        array_complex[i] = zero_complex;
                        array_int[i]     = 0;

                        array_float[i] = static_cast<float>(std::tan(i));
                        array_short[i] = 0;
                        array_char[i]  = static_cast<unsigned char>(interpolation(i, sz, 128));
                    }
                    else
                    {
                        arr1[i]          = 0;
                        arr2[i]          = 0;
                        arr3[i]          = std::sin(i + iter);
                        arr4[i]          = 0;
                        arr5[i]          = std::cos(i + (2 * iter));
                        array_complex[i] = ComplexDouble(std::tan(i + (2 * iter)), std::cos(i - iter));
                        array_int[i]     = static_cast<int>(i - iter);
                        array_float[i]   = 0;
                        array_short[i]   = static_cast<short>(interpolation(i, sz, std::numeric_limits<short>::max()));
                        array_char[i]    = 0;
                    }
                },
                bench_par::nowait);

            // To simulate behavior of a code where we read more than we write

            // nowait here too, on BOTH branches, and schedule(static) spelt out on
            // this loop and on the reduction loops that read what it writes: with
            // the same iteration count and an explicit static schedule, OpenMP
            // guarantees the same thread gets the same iterations in both loops
            // (the default schedule is implementation-defined, so it must be
            // written). Each thread therefore reads back only what IT wrote, and a
            // write of its own that is still deferred is what the library's read
            // gate resolves (DeferredReadGate: a read of a chunk with pending ops
            // drains them first). No cross-thread dependency, no barrier, and no
            // flush after it -- same reasoning as the two nowait loops above.
            if (iter % 4 != 0)
            {
                // To simulate behavior of a code where we read more than we write

                team.for_static_chunks<Int>(
                    sched_base, end0, bench_par::block(sched_blk),
                    [&](const Int i) __attribute__((always_inline))
                    {
                        if (i < start0) // alignment head: same trip count in every loop
                        {
                            return;
                        }
                        arr1[i] = std::sqrt(double(i));

                        array_float[i] += 1;

                        array_complex[i] = ComplexDouble(std::sqrt(double(i + 1)), std::sqrt(double(i - 1)));

                        if (i > array_nba1[iter])
                        {
                            arr2[i]       = std::cos(double(2 * i));
                            array_char[i] = static_cast<unsigned char>(
                                exponential_correlation(i, sz, std::numeric_limits<char>::max(), 2));
                            array_short[i] = static_cast<short>(
                                exponential_correlation(i, sz, std::numeric_limits<short>::max(), 2));
                        }

                        if (i > array_nba2[iter])
                        {
                            arr3[i] = std::sin(arr1[i] + arr2[i]);
                            arr4[i] *= arr5[i] + 1;
                            array_bool[i]      = true;
                            array_bool_bits[i] = true;
                            array_int[i]       = static_cast<int>(i + 1);

                            array_int[i] = static_cast<int>(std::floor(log10(static_cast<double>(array_int[i]))));
                            array_int[i] *=
                                array_int[i]; // Before the previous line if i bigger than sqrt(soit 2^31 − 1)
                            // (46340) --> overflow

                            array_int[i]--;
                            array_int[i] = -array_int[i];

                            ++array_int[i];
                            array_int[i] = std::abs(array_int[i]);

                            array_long[i] = static_cast<long>(i);
                            array_long[i] += 1L;
                            array_long[i] *= 2L;
                            --array_long[i];

                            // Bounded to CHUNK_BENCH_ULONG_BANDS distinct values. The
                            // band arithmetic below maps each 256-index band to one
                            // constant, so writing i directly gave the array a NEW
                            // second value per written range -- 214 300 of them at
                            // 4000 iterations, growing with the run. That is not what
                            // a solver does: it registers the constants it actually
                            // holds, a few thousand at most. Masking first keeps the
                            // write identical in shape -- same operations, same type,
                            // a band still maps to one value -- and stops the bench
                            // from manufacturing the very registry pressure it was
                            // then blamed for. 0 restores the unbounded behaviour.
                            array_ulong[i] = static_cast<unsigned long>(i) & bench_ulong_mask;
                            array_ulong[i] |= 0xFFUL;
                            array_ulong[i] &= ~0x0FUL;

                            array_float[i] -= 1.5f;
                            array_char[i] =
                                static_cast<unsigned char>(linear_correlation(i, sz, std::numeric_limits<char>::max()));
                            array_short[i] =
                                static_cast<short>(linear_correlation(i, sz, std::numeric_limits<short>::max()));
                            array_complex[i] = ComplexDouble(arr1[i], arr2[i]);

                            array_complex[i] = 3. * array_complex[i];
                            array_complex[i] = array_complex[i] + ComplexDouble(1., 3.);
                            array_complex[i] *= 2.;
                            array_complex[i] = ComplexDouble(2., 3.) * array_complex[i];
                            array_complex[i] = 2. / array_complex[i];
                            array_complex[i] = 2. - array_complex[i];
                            array_complex[i] = array_complex[i] - ComplexDouble(4., 3.);
                            array_complex[i] = -array_complex[i];
                        }
                    },
                    bench_par::nowait);
            }
            else
            {
                team.for_static_chunks<Int>(
                    sched_base, end0, bench_par::block(sched_blk),
                    [&](const Int i) __attribute__((always_inline))
                    {
                        if (i < start0) // alignment head: same trip count in every loop
                        {
                            return;
                        }
                        array_int[i]       = second_value_i;
                        array_bool[i]      = second_value_b;
                        array_bool_bits[i] = second_value_b;

                        arr2[i] = .5;
                        arr3[i] = 2;
                        arr1[i] = arr2[i] * arr3[i];
                        arr2[i] = second_value_d;
                        arr3[i] = second_value_d;

                        arr4[i] = second_value_d1;
                        arr5[i] = second_value_d2;
                        arr1[i] *= second_value_d;

                        array_float[i] = second_value_f1;
                        array_char[i]  = second_value_c1;
                        array_short[i] = second_value_s1;

                        array_complex[i] = second_value_c;
                    },
                    bench_par::nowait);
            }

            const Int interval_loc = end0 - start0;
            const Int interval10   = interval_loc / 10;

            const Int beg_loc = start0 + (4 * interval10);
            const Int end_loc = start0 + (5 * interval10);

            const auto beg_loc0 =
                static_cast<Int>(static_cast<double>(start0) + (4.8 * static_cast<double>(interval10)));

            // The per-thread partials of the 20 passes below, merged ONCE after
            // them: an atomic merge per pass (80 per thread per call, every thread
            // at the same moment, on the four neighbouring sums of main) was 5.7 %
            // of a TBB run at 160 threads -- the double ones are CAS loops.
            double p_val_sum      = 0;
            long   p_long_val_sum = 0;
            double p_v_real       = 0;
            double p_v_im         = 0;
            for (Int p = 0; p < 20; ++p)
            {
                const auto val = static_cast<double>(p);

                // One update per accumulator per element, the branch as a select:
                // the form the vectoriser recognises as a reduction (the previous
                // shape -- val_sum updated in both arms of an if, by += then -= --
                // was "value that could not be identified as reduction"). Same
                // arithmetic per element; only the rounding order inside val_sum
                // changes, and val_sum is printed, not compared. On the
                // std::vector baseline this loop now vectorises (width 4, cos /
                // sin / tan through the vector libm, atan scalarised); on a shrink
                // build it is the same scalar loop as before.
                // No reduction clause: GCC merges a multi-variable reduction inside
                // GOMP_atomic_start(), ONE global libgomp mutex, which 64 threads hit
                // together at the end of this nowait loop (2.9x slower than libomp).
                // Per-thread partials (declared before the passes), merged by
                // lock-free atomics after them. The blocks are walked explicitly --
                // block b of sched_blk iterations from sched_base goes to thread
                // b mod T, the mapping of the region's schedule(static, sched_blk)
                // loops -- so the inner loop can be an omp simd reduction on the
                // std::vector baseline.
                const Int n_blocks = (end0 - sched_base + sched_blk - 1) / sched_blk;
                team.for_static_chunks<Int>(
                    0, n_blocks, 1,
                    [&](const Int blk) __attribute__((always_inline))
                    {
                        const Int lo = std::max(start0, sched_base + (blk * sched_blk));
                        const Int hi = std::min(end0, sched_base + ((blk + 1) * sched_blk));
                        BENCH_OMP_SIMD_RED(p_val_sum, p_long_val_sum, p_v_real, p_v_im)
                        for (Int i = lo; i < hi; ++i)
                        {
                            const double t1 = arr1[i];
                            const double t2 = arr2[i];
                            const double t3 = arr3[i];
                            const double t5 = arr5[i];

                            const ComplexDouble valc = array_complex[i];
                            const bool          flag = array_bool[i];
                            const Int           ti   = array_int[i];

                            const double common = std::cos(t1) + std::sin(t2) + std::atan(t3) + std::cos(val);
                            const double arm    = flag ? (std::cos(t1) + std::sin(t2) + std::atan(t3) + std::cos(val)) -
                                                             (std::tan(t3) + std::cos(t5))
                                                       : (std::sin(t1) + std::atan(t2) + std::cos(t3) + std::sin(val)) -
                                                             (std::sin(t3) + std::atan(t5));

                            p_val_sum += common + arm;
                            p_v_real += std::cos(valc.real()) - array_float[i];
                            p_v_im += std::sin(valc.imag()) - array_short[i] - array_char[i];
                            p_long_val_sum += ti > 0 ? static_cast<long>(ti) : 0L;
                        }
                    },
                    bench_par::nowait);
            }
            bench_par::atomic_add(val_sum, p_val_sum);
            bench_par::atomic_add(long_val_sum, p_long_val_sum);
            bench_par::atomic_add(v_real, p_v_real);
            bench_par::atomic_add(v_im, p_v_im);

            // No barrier before the last loop, and none between its two passes:
            // same iteration space, same static schedule, so its p = 1 pass reads
            // back the tail its own p = 0 pass wrote, and the array_bool / array_int it
            // sets are read by the reductions above on the same slice. The whole
            // region is now barrier-free: every thread works its own slice from
            // the first loop to the last, and the deferred writes it leaves behind
            // are resolved by the library's read gate or by the region's end.
            for (Int p = 0; p < 2; ++p)
            {
                const auto val = static_cast<double>(p);

                bench_par::for_static_chunks_sum(
                    team, sched_base, end0, bench_par::block(sched_blk), val_sum,
                    [&](const Int i, auto& val_sum_part) __attribute__((always_inline))
                    {
                        if (i < start0) // alignment head: same trip count in every loop
                        {
                            return;
                        }
                        val_sum_part += std::cos(arr1[i]) + std::sin(arr2[i]) + std::atan(arr3[i]) + std::cos(val);
                        val_sum_part += std::sin(arr4[i]) * std::cos(arr5[i]);
                        val_sum_part -= array_float[i] + array_short[i] - array_char[i];
                        if (i > end0 - interval10)
                        {
                            arr1[i] = (arr2[i] * arr3[i] * std::sin(arr1[i])) + std::cos(val);
                            arr1[i] += arr2[i] - std::exp(std::fmod(static_cast<double>(arr3[i]), 700.0));
                            arr1[i] -= arr3[i] + std::tan(arr2[i]);
                            arr1[i] += 1;
                            --arr2[i];
                            arr2[i] = -arr2[i];
                            arr2[i]++;
                            arr2[i] *= arr3[i];
                            arr2[i] /= arr1[i];
                            ++arr4[i];
                            arr5[i] *= 2;
                            array_float[i] /= 1.01f;
                            array_float[i] -= .5f;

                            array_complex[i] /= ComplexDouble(2, 3);
                            array_complex[i] -= arr1[i];
                            array_complex[i] *= arr1[i];

                            array_complex[i] = ComplexDouble(3., 2.) * arr1[i] + arr2[i] / ComplexDouble(2., 2.);
                            array_complex[i] = arr1[i] - ComplexDouble(1., 2.) + arr3[i] + ComplexDouble(3., 1.);

                            array_complex[i] *= arr1[i];

                            // --- bound accumulators to keep the stress workload
                            // numerically finite (values are a checksum; exact
                            // magnitude irrelevant). Prevents FE_OVERFLOW traps on
                            // the benign divergence of exp / repeated *=. Memory
                            // and rechunk behaviour are unchanged (same writes).
                            arr1[i] = std::fmod(static_cast<double>(arr1[i]), 1e6);
                            arr2[i] = std::fmod(static_cast<double>(arr2[i]), 1e6);
                            arr3[i] = std::fmod(static_cast<double>(arr3[i]), 1e6);
                            arr5[i] = std::fmod(static_cast<double>(arr5[i]), 1e6);
                            {
                                const ComplexDouble cb = array_complex[i];
                                array_complex[i] = ComplexDouble(std::fmod(cb.real(), 1e6), std::fmod(cb.imag(), 1e6));
                            }

                            if (arr2[i] >= 0)
                            {
                                neutral_sqrt_assign(arr2, i, static_cast<double>(arr2[i]));
                            }
                            array_bool[i]      = true;
                            array_bool_bits[i] = true;
                            array_int[i]       = static_cast<int>(i);
                        }

                        if (i > beg_loc && i < end_loc)
                        {
                            arr2[i] = 0;
                            arr3[i] = 1;

                            arr4[i] = 0;
                            arr5[i] = 1;

                            array_int[i] = static_cast<int>(i & 0xF);
                            array_int[i] |= 0x1;
                            array_int[i] ^= 0x3;
                            array_int[i] <<= 2;
                            array_int[i] >>= 1;
                            array_int[i] = (array_int[i] & 0xF) | 0x2;
                            array_int[i] %= 5; // same value as array_int[i] % 5: the only %= of the bench
                            array_int[i] = ~array_int[i];
                            array_int[i] = 1;

                            if (i > beg_loc0)
                            {
                                array_int[i] = (array_bool[i] ? 1 : 0);

                                array_float[i] = array_int[i] + 0.5f;

                                array_int[i] = static_cast<int>(array_float[i] * 2);

                                if (array_int[i] == 0)
                                {
                                    array_bool[i]      = true;
                                    array_bool_bits[i] = true;
                                }
                                if (array_int[i] != 1)
                                {
                                    array_bool[i]      = true;
                                    array_bool_bits[i] = true;
                                }
                                if (array_int[i] < 10)
                                {
                                    array_bool[i]      = true;
                                    array_bool_bits[i] = true;
                                }
                                if (array_int[i] <= 100)
                                {
                                    array_bool[i]      = true;
                                    array_bool_bits[i] = true;
                                }

                                const bool flag = array_bool[i];
                                if (!flag)
                                {
                                    array_int[i]++;
                                }
                                if (array_bool[i] && array_int[i] > 0)
                                {
                                    --array_int[i];
                                }
                                if (array_bool[i] || array_int[i] < 0)
                                {
                                    array_int[i]--;
                                }

                                array_float[i] = -array_float[i];
                                array_int[i]   = array_int[i] + 1;
                                array_float[i] = array_float[i] - 0.5f;
                                array_int[i]   = array_int[i] * 2;
                                array_int[i]   = array_int[i] / 3;
                                array_int[i]   = array_bool[i] ? (array_float[i] > 0 ? 1 : 2) : 0;
                                array_float[i] = array_int[i] + 0.5f;
                                array_int[i]   = static_cast<int>(array_float[i] * arr1[i]);
                            }
                            else
                            {
                                array_complex[i] = zero_complex;

                                array_float[i] = 0;
                                array_char[i]  = 0;
                                array_short[i] = 0;
                            }
                        }
                    });
            }

            finalize_parallel_phase_only_flush_g();
        });
    finalize_parallel_phase_after_flush_g();

    val_sum_out += val_sum;
    long_val_sum_out += long_val_sum;
    v_real_out += v_real;
    v_im_out += v_im;
}
