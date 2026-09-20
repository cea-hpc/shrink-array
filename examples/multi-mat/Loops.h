
/*
 * Copyright 2024-2025 CEA (www.cea.fr)
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

#pragma once

#include "PMP.h"

// `#pragma forceinline` is the classic Intel compiler's; gcc and clang warn
// "ignoring #pragma forceinline" on every use and inline the lambda anyway at
// -O3. Kept for icpc, empty elsewhere.
#if defined(__INTEL_COMPILER) && !defined(__INTEL_LLVM_COMPILER)
    #define PRAGMA_FORCEINLINE _Pragma("forceinline")
#else
    #define PRAGMA_FORCEINLINE
#endif

class Loops
{
  public:
    Loops()  = default;
    ~Loops() = default;

    template <typename LBD>
    static void omp_for_nowait_simd(int kdeb, int kfin, const LBD& lbd)
    {
#pragma omp for schedule(static) nowait
        for (int k = kdeb; k < kfin; ++k)
        {
            PRAGMA_FORCEINLINE
            lbd(k);
        }
    }

    template <typename LBD, typename LL>
    static void omp_simd_foreach_int(const LL& ll, const LBD& lbd, int nn)
    {
#pragma omp for schedule(static)
        for (int k = 0; k < nn; ++k)
        {
            const int num = ll[k];
            PRAGMA_FORCEINLINE
            lbd(num);
        }
    }

    template <typename LBD, typename LL>
    static void omp_simd_foreach_nowait_int(const LL& ll, const LBD& lbd, int nn)
    {
#pragma omp for schedule(static) nowait
        for (int k = 0; k < nn; ++k)
        {
            const int num = ll[k];
            PRAGMA_FORCEINLINE
            lbd(num);
        }
    }

    template <typename LBD, typename LL>
    void omp_simd_foreach(const LL& ll, const LBD& lbd) const
    {
        omp_simd_foreach_int(ll, lbd, ll.size());
    }

    template <typename LBD, typename LL>
    void omp_simd_foreach_nowait(const LL& ll, const LBD& lbd) const
    {
        omp_simd_foreach_nowait_int(ll, lbd, ll.size());
    }

    template <typename LBD, typename LL, typename T>
    static void omp_foreach_simd_red_sum(const LL& ll, T& red, const LBD& lbd) noexcept
    {
        const int nn = ll.size();

#ifndef __clang__
        const auto [kdeb, kfin] = get_parallel_loop_boundaries(PMP_THREAD_RANK, PMP_MAX_THREADS, nn);
#else
        int kdeb, kfin;
        std::tie(kdeb, kfin) = get_parallel_loop_boundaries(PMP_THREAD_RANK, PMP_MAX_THREADS, nn);
#endif


        for (int k = kdeb; k < kfin; ++k)
        {
            const int nnn = ll[k];
            PRAGMA_FORCEINLINE
            lbd(nnn, red);
        }

#pragma omp barrier
    }
};

class LoopsSimd
{
  public:
    LoopsSimd()  = default;
    ~LoopsSimd() = default;

    template <typename LBD, typename LL>
    static void omp_simd_foreach_int(const LL& ll, const LBD& lbd, int nn)
    {
#pragma omp for simd        schedule(static)
        for (int k = 0; k < nn; ++k)
        {
            const int num = ll[k];
            PRAGMA_FORCEINLINE
            lbd(num);
        }
           }

           template <typename LBD, typename LL>
           static void omp_simd_foreach_nowait_int(const LL& ll, const LBD& lbd, int nn)
           {
#pragma omp for simd nowait schedule(static)
        for (int k = 0; k < nn; ++k)
        {
            const int num = ll[k];
            PRAGMA_FORCEINLINE
            lbd(num);
        }
    }

    template <typename LBD, typename LL>
    void omp_simd_foreach(const LL& ll, const LBD& lbd) const
    {
        omp_simd_foreach_int(ll, lbd, ll.size());
    }

    template <typename LBD, typename LL>
    void omp_simd_foreach_nowait(const LL& ll, const LBD& lbd) const
    {
        omp_simd_foreach_nowait_int(ll, lbd, ll.size());
    }

    template <typename LBD>
    static void omp_for_nowait_simd(int kdeb, int kfin, const LBD& lbd)
    {
#pragma omp for simd nowait schedule(static)
        for (int k = kdeb; k < kfin; ++k)
        {
            PRAGMA_FORCEINLINE
            lbd(k);
        }
    }

    template <typename LBD, typename LL, typename T>
    void omp_foreach_simd_red_sum(const LL& ll, T& red, const LBD& lbd) const noexcept
    {
        const int nn = ll.size();

#ifndef __clang__
        const auto [kdeb, kfin] = get_parallel_loop_boundaries(PMP_THREAD_RANK, PMP_MAX_THREADS, nn);
#else
        int kdeb, kfin;
        std::tie(kdeb, kfin) = get_parallel_loop_boundaries(PMP_THREAD_RANK, PMP_MAX_THREADS, nn);
#endif
        T redl = red;
#pragma omp simd reduction(+ : redl)
        for (int k = kdeb; k < kfin; ++k)
        {
            const int nnn = ll[k];
            PRAGMA_FORCEINLINE
            lbd(nnn, redl);
        }

        red = redl;

#pragma omp barrier
    }
};
