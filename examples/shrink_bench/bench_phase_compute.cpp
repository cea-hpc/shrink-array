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

#include <cstdlib> // std::getenv / std::atol (CHUNK_BENCH_ULONG_BANDS)
#include <cmath>
#include <complex>
#include <limits>
#include <vector>

#include <algorithm> // std::min / std::max (block bounds)

#include "bench_phases.h"
#include "bench_utils.h" // linear_correlation, exponential_correlation

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
static const unsigned long bench_ulong_mask = []() -> unsigned long
{
    long bands = 4096;
    if (const char* e = std::getenv("CHUNK_BENCH_ULONG_BANDS"))
        bands = std::atol(e);
    if (bands <= 0)
        return ~0UL;                      // unbounded
    // Bands are 256 indices wide (the |0xFF, &~0x0F arithmetic below), so the
    // mask has to cover bands * 256 indices, rounded up to a power of two.
    unsigned long span = static_cast<unsigned long>(bands) * 256UL;
    unsigned long m    = 1;
    while (m < span)
        m <<= 1;
    return m - 1UL;
}();

void bench_phase_compute(BenchArrays&            bench_arr,
                         int                     k,
                         int                     start0,
                         int                     end0,
                         const std::vector<int>& tab_nba1,
                         const std::vector<int>& tab_nba2,
                         double&                 val_sum_out,
                         long&                   long_val_sum_out,
                         double&                 v_real_out,
                         double&                 v_im_out)
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

    const int sz = bench_arr.sz;

    // Zero-initialised: the reduction accumulates here, then lands on the
    // caller's running totals at the end of the function.
    double val_sum      = 0;
    long   long_val_sum = 0;
    double v_real       = 0;
    double v_im         = 0;

// The accumulators are SHARED in the region, and each loop that adds to them
// carries its own reduction clause (BENCH_OMP_FOR_STATIC_NOWAIT_RED): a
// worksharing reduction must be on a shared variable, and that is the form
// `for simd` accepts on the std::vector baseline. A reduction on a nowait
// loop is complete for the thread at the end of its own portion; the value
// is whole after the region's barrier, which is where it is read.
    // Block size and aligned start of the [start0, end0) loops' schedule
    // (bench_sched_for, bench_prelude.h). Computed once, before the region:
    // every thread must see the same values, or the same-thread guarantee is
    // void. The pragma macros name sched_blk.
    const BenchSched sched      = bench_sched_for(start0, end0);
    const int        sched_blk  = sched.blk;
    const int        sched_base = sched.base;

#pragma omp parallel
    {
        begin_parallel_phase_g();

        const int dec = k % 16;
        const int deb = std::max(start0 - 512, 0);

        // nowait on this loop and the next: they write DISJOINT ranges
        // ([deb, start0) here, [end0, end0 + 256) below) that nothing in this
        // region reads back, so a thread done with its slice may go on. The
        // three loops after them stay barriered: the [start0, end0) loop
        // writes what the reduction loops then read, and the last one
        // rewrites the tail its own p = 0 pass produced.
        // tab_complex peeled out of the loop below. std::complex<double> is an
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
        BENCH_OMP_FOR_STATIC_NOWAIT_SCALAR
        for (int i = deb; i < start0; ++i)
        {
            const int j = i - deb;

            if (j < 60 + dec)
                tab_complex[i] = complex_double(std::sin(i + k), std::cos(i + k));
            else if (j < 120 + dec)
                tab_complex[i] = zero_complex;
            else if (j < 240 + dec)
                tab_complex[i] = second_value_c;
            else if (j < 360 + dec)
                tab_complex[i] = zero_complex;
            else
                tab_complex[i] = second_value_c;
        }

        BENCH_OMP_FOR_STATIC_NOWAIT_SCALAR
        for (int i = deb; i < start0; ++i)
        {
            const int j = i - deb;

            if (j < 60 + dec)
            {
                tab1[i]    = 0;
                tab2[i]    = second_value_d;
                tab3[i]    = std::tan(i + k);
                tab4[i]    = 0;
                tab5[i]    = std::sin(i + k);
                tab_int[i] = 0;

                tab_float[j] = static_cast<float>(std::cos(i - k));
                tab_short[j] = 0;
                tab_char[i]  = 0;
            }
            else if (j < 120 + dec)
            {
                tab1[i]      = second_value_d;
                tab2[i]      = 0;
                tab3[i]      = 0;
                tab4[i]      = second_value_d1;
                tab5[i]      = 0;
                tab_int[i]   = 1;
                tab_float[j] = 0;
                tab_short[j] = 1;
                tab_char[i]  = 0;
            }
            else if (j < 240 + dec)
            {
                tab1[i]      = 0;
                tab2[i]      = std::sin(i + k);
                tab3[i]      = second_value_d;
                tab4[i]      = 0;
                tab5[i]      = second_value_d2;
                tab_int[i]   = j;
                tab_float[j] = second_value_f0;
                tab_short[j] = second_value_s0;
                tab_char[i]  = second_value_c0;
            }
            else if (j < 360 + dec)
            {
                tab1[i]      = std::cos(i + k);
                tab2[i]      = 0;
                tab3[i]      = 0;
                tab4[i]      = 0;
                tab5[i]      = 0;
                tab_int[i]   = j + 1;
                tab_float[j] = 0;
                tab_short[j] = 0;
                tab_char[i]  = 0;
            }
            else
            {
                tab1[i]      = 0;
                tab2[i]      = second_value_d;
                tab3[i]      = second_value_d;
                tab4[i]      = second_value_d1;
                tab5[i]      = second_value_d2;
                tab_float[j] = second_value_f1;
                tab_short[j] = second_value_s2;
                tab_char[i]  = 0;
                tab_int[i]   = 0;
            }
        }

        // No finalize_barrier_phase_g() after either nowait loop: that hook is
        // a barrier plus a flush of the deferred writes, i.e. exactly what the
        // OMPT build does at an implicit barrier -- and a nowait loop has none.
        // Both builds now flush these two loops' writes at the next loop's
        // barrier, which is before anything reads them.
        //
        // Scalar: 256 elements, and interpolation() carries early exits
        // ("cannot vectorize early exit loop with more than one early exit").
        BENCH_OMP_FOR_STATIC_NOWAIT_SCALAR
        for (int i = end0; i < std::min(end0 + 256, sz); ++i)
        {
            if (i % 3 == 0)
            {
                tab1[i] = second_value_d;
                tab2[i] = std::sin(i + k);
                tab3[i] = 0;
                tab4[i] = second_value_d1;
                tab5[i] = std::cos(i - k);

                tab_complex[i] = zero_complex;
                tab_int[i]     = 0;

                tab_float[i] = static_cast<float>(std::tan(i));
                tab_short[i] = 0;
                tab_char[i]  = interpolation(i, sz, 128);
            }
            else
            {
                tab1[i]        = 0;
                tab2[i]        = 0;
                tab3[i]        = std::sin(i + k);
                tab4[i]        = 0;
                tab5[i]        = std::cos(i + 2 * k);
                tab_complex[i] = complex_double(std::tan(i + 2 * k), std::cos(i - k));
                tab_int[i]     = i - k;
                tab_float[i]   = 0;
                tab_short[i]   = interpolation(i, sz, std::numeric_limits<short>::max());
                tab_char[i]    = 0;
            }
        }

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
        // finalize_barrier_phase_g() after it -- same reasoning as the two
        // nowait loops above.
        if (k % 4 != 0)
        {
            // To simulate behavior of a code where we read more than we write

            BENCH_OMP_FOR_BLOCK_NOWAIT_SCALAR
            for (int i = sched_base; i < end0; ++i)
            {
                if (i < start0) // alignment head: same trip count in every loop
                    continue;
                tab1[i] = std::sqrt(double(i));

                tab_float[i] += 1;

                tab_complex[i] = complex_double(std::sqrt(double(i + 1)), std::sqrt(double(i - 1)));

                if (i > tab_nba1[k])
                {
                    tab2[i]      = std::cos(double(2 * i));
                    tab_char[i]  = exponential_correlation(i, sz, std::numeric_limits<char>::max(), 2);
                    tab_short[i] = exponential_correlation(i, sz, std::numeric_limits<short>::max(), 2);
                }

                if (i > tab_nba2[k])
                {
                    tab3[i] = std::sin(tab1[i] + tab2[i]);
                    tab4[i] *= tab5[i] + 1;
                    tab_bool[i] = true;
                    tab_int[i]  = i + 1;

                    tab_int[i] = static_cast<int>(std::floor(log10(static_cast<double>(tab_int[i]))));
                    tab_int[i] *= tab_int[i]; // Before the previous line if i bigger than sqrt(soit 2^31 − 1)
                    // (46340) --> overflow

                    tab_int[i]--;
                    tab_int[i] = -tab_int[i];

                    ++tab_int[i];
                    tab_int[i] = std::abs(tab_int[i]);

                    tab_long[i] = static_cast<long>(i);
                    tab_long[i] += 1L;
                    tab_long[i] *= 2L;
                    --tab_long[i];

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
                    tab_ulong[i] = static_cast<unsigned long>(i) & bench_ulong_mask;
                    tab_ulong[i] |= 0xFFUL;
                    tab_ulong[i] &= ~0x0FUL;

                    tab_float[i] -= 1.5f;
                    tab_char[i]    = linear_correlation(i, sz, std::numeric_limits<char>::max());
                    tab_short[i]   = linear_correlation(i, sz, std::numeric_limits<short>::max());
                    tab_complex[i] = complex_double(tab1[i], tab2[i]);

                    tab_complex[i] = 3. * tab_complex[i];
                    tab_complex[i] = tab_complex[i] + complex_double(1., 3.);
                    tab_complex[i] *= 2.;
                    tab_complex[i] = complex_double(2., 3.) * tab_complex[i];
                    tab_complex[i] = 2. / tab_complex[i];
                    tab_complex[i] = 2. - tab_complex[i];
                    tab_complex[i] = tab_complex[i] - complex_double(4., 3.);
                    tab_complex[i] = -tab_complex[i];
                }
            }
        }
        else
        {
            BENCH_OMP_FOR_BLOCK_NOWAIT_SCALAR
            for (int i = sched_base; i < end0; ++i)
            {
                if (i < start0) // alignment head: same trip count in every loop
                    continue;
                tab_int[i]  = second_value_i;
                tab_bool[i] = second_value_b;

                tab2[i] = .5;
                tab3[i] = 2;
                tab1[i] = tab2[i] * tab3[i];
                tab2[i] = second_value_d;
                tab3[i] = second_value_d;

                tab4[i] = second_value_d1;
                tab5[i] = second_value_d2;
                tab1[i] *= second_value_d;

                tab_float[i] = second_value_f1;
                tab_char[i]  = second_value_c1;
                tab_short[i] = second_value_s1;

                tab_complex[i] = second_value_c;
            }
        }

        const int interval_loc = end0 - start0;
        const int interval10   = interval_loc / 10;

        const int beg_loc = start0 + 4 * interval10;
        const int end_loc = start0 + 5 * interval10;

        const int beg_loc0 = start0 + 4.8 * interval10;

        for (int p = 0; p < 20; ++p)
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
            // Per-thread partials, merged by lock-free atomics below. The blocks
            // are walked explicitly -- block b of sched_blk iterations from
            // sched_base goes to thread b mod T, the mapping of the region's
            // schedule(static, sched_blk) loops -- so the inner loop can be an
            // omp simd reduction on the std::vector baseline.
            double    p_val_sum      = 0;
            long      p_long_val_sum = 0;
            double    p_v_real       = 0;
            double    p_v_im         = 0;
            const int n_blocks       = (end0 - sched_base + sched_blk - 1) / sched_blk;
#pragma omp for schedule(static, 1) nowait
            for (int blk = 0; blk < n_blocks; ++blk)
            {
                const int lo = std::max(start0, sched_base + blk * sched_blk);
                const int hi = std::min(end0, sched_base + (blk + 1) * sched_blk);
                BENCH_OMP_SIMD_RED(p_val_sum, p_long_val_sum, p_v_real, p_v_im)
                for (int i = lo; i < hi; ++i)
                {
                    const double t1 = tab1[i];
                    const double t2 = tab2[i];
                    const double t3 = tab3[i];
                    const double t5 = tab5[i];

                    const complex_double valc = tab_complex[i];
                    const bool           b    = tab_bool[i];
                    const int            ti   = tab_int[i];

                    const double common = std::cos(t1) + std::sin(t2) + std::atan(t3) + std::cos(val);
                    const double arm =
                        b ? (std::cos(t1) + std::sin(t2) + std::atan(t3) + std::cos(val)) - (std::tan(t3) + std::cos(t5))
                          : (std::sin(t1) + std::atan(t2) + std::cos(t3) + std::sin(val)) - (std::sin(t3) + std::atan(t5));

                    p_val_sum += common + arm;
                    p_v_real += std::cos(valc.real()) - tab_float[i];
                    p_v_im += std::sin(valc.imag()) - tab_short[i] - tab_char[i];
                    p_long_val_sum += ti > 0 ? static_cast<long>(ti) : 0L;
                }
            }
#pragma omp atomic
            val_sum += p_val_sum;
#pragma omp atomic
            long_val_sum += p_long_val_sum;
#pragma omp atomic
            v_real += p_v_real;
#pragma omp atomic
            v_im += p_v_im;
        }

        // No barrier before the last loop, and none between its two passes:
        // same iteration space, same static schedule, so its p = 1 pass reads
        // back the tail its own p = 0 pass wrote, and the tab_bool / tab_int it
        // sets are read by the reductions above on the same slice. The whole
        // region is now barrier-free: every thread works its own slice from
        // the first loop to the last, and the deferred writes it leaves behind
        // are resolved by the library's read gate or by the region's end.
        for (int p = 0; p < 2; ++p)
        {
            const double val = static_cast<double>(p);

            BENCH_OMP_FOR_BLOCK_NOWAIT_RED_SCALAR(val_sum)
            for (int i = sched_base; i < end0; ++i)
            {
                if (i < start0) // alignment head: same trip count in every loop
                    continue;
                val_sum += std::cos(tab1[i]) + std::sin(tab2[i]) + std::atan(tab3[i]) + std::cos(val);
                val_sum += std::sin(tab4[i]) * std::cos(tab5[i]);
                val_sum -= tab_float[i] + tab_short[i] - tab_char[i];
                if (i > end0 - interval10)
                {
                    tab1[i] = tab2[i] * tab3[i] * std::sin(tab1[i]) + std::cos(val);
                    tab1[i] += tab2[i] - std::exp(std::fmod(static_cast<double>(tab3[i]), 700.0));
                    tab1[i] -= tab3[i] + std::tan(tab2[i]);
                    tab1[i] += 1;
                    --tab2[i];
                    tab2[i] = -tab2[i];
                    tab2[i]++;
                    tab2[i] *= tab3[i];
                    tab2[i] /= tab1[i];
                    ++tab4[i];
                    tab5[i] *= 2;
                    tab_float[i] /= 1.01f;
                    tab_float[i] -= .5f;

                    tab_complex[i] /= complex_double(2, 3);
                    tab_complex[i] -= tab1[i];
                    tab_complex[i] *= tab1[i];

                    tab_complex[i] = complex_double(3., 2.) * tab1[i] + tab2[i] / complex_double(2., 2.);
                    tab_complex[i] = tab1[i] - complex_double(1., 2.) + tab3[i] + complex_double(3., 1.);

                    tab_complex[i] *= tab1[i];

                    // --- bound accumulators to keep the stress workload
                    // numerically finite (values are a checksum; exact
                    // magnitude irrelevant). Prevents FE_OVERFLOW traps on
                    // the benign divergence of exp / repeated *=. Memory
                    // and rechunk behaviour are unchanged (same writes).
                    tab1[i] = std::fmod(static_cast<double>(tab1[i]), 1e6);
                    tab2[i] = std::fmod(static_cast<double>(tab2[i]), 1e6);
                    tab3[i] = std::fmod(static_cast<double>(tab3[i]), 1e6);
                    tab5[i] = std::fmod(static_cast<double>(tab5[i]), 1e6);
                    {
                        const complex_double _cb = tab_complex[i];
                        tab_complex[i] = complex_double(std::fmod(_cb.real(), 1e6), std::fmod(_cb.imag(), 1e6));
                    }

                    if (tab2[i] >= 0)
                    {
                        neutral_sqrt_assign(tab2, i, tab2[i]);
                    }
                    tab_bool[i] = true;
                    tab_int[i]  = i;
                }

                if (i > beg_loc && i < end_loc)
                {
                    tab2[i] = 0;
                    tab3[i] = 1;

                    tab4[i] = 0;
                    tab5[i] = 1;

                    tab_int[i] = i & 0xF;
                    tab_int[i] |= 0x1;
                    tab_int[i] ^= 0x3;
                    tab_int[i] <<= 2;
                    tab_int[i] >>= 1;
                    tab_int[i] = (tab_int[i] & 0xF) | 0x2;
                    tab_int[i] = tab_int[i] % 5;
                    tab_int[i] = ~tab_int[i];
                    tab_int[i] = 1;

                    if (i > beg_loc0)
                    {
                        tab_int[i] = (tab_bool[i] ? 1 : 0);

                        tab_float[i] = tab_int[i] + 0.5f;

                        tab_int[i] = static_cast<int>(tab_float[i] * 2);

                        if (tab_int[i] == 0)
                        {
                            tab_bool[i] = true;
                        }
                        if (tab_int[i] != 1)
                        {
                            tab_bool[i] = true;
                        }
                        if (tab_int[i] < 10)
                        {
                            tab_bool[i] = true;
                        }
                        if (tab_int[i] <= 100)
                        {
                            tab_bool[i] = true;
                        }

                        const bool b = tab_bool[i];
                        if (!b)
                        {
                            tab_int[i]++;
                        }
                        if (tab_bool[i] && tab_int[i] > 0)
                        {
                            --tab_int[i];
                        }
                        if (tab_bool[i] || tab_int[i] < 0)
                        {
                            tab_int[i]--;
                        }

                        tab_float[i] = -tab_float[i];
                        tab_int[i]   = tab_int[i] + 1;
                        tab_float[i] = tab_float[i] - 0.5f;
                        tab_int[i]   = tab_int[i] * 2;
                        tab_int[i]   = tab_int[i] / 3;
                        tab_int[i]   = tab_bool[i] ? (tab_float[i] > 0 ? 1 : 2) : 0;
                        tab_float[i] = tab_int[i] + 0.5f;
                        tab_int[i]   = static_cast<int>(tab_float[i] * tab1[i]);
                    }
                    else
                    {
                        tab_complex[i] = zero_complex;

                        tab_float[i] = 0;
                        tab_char[i]  = 0;
                        tab_short[i] = 0;
                    }
                }
            }
        }

        finalize_parallel_phase_only_flush_g();
    }
    finalize_parallel_phase_after_flush_g();

    val_sum_out += val_sum;
    long_val_sum_out += long_val_sum;
    v_real_out += v_real;
    v_im_out += v_im;
}
