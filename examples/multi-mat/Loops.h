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

#pragma once

/**
 * @file Loops.h
 * @brief The two loop policies of the example: scalar for the chunk arrays,
 *        `omp simd` for the std::vector ones.
 *
 * Every loop of ComputeLoops.h is written once, against a policy object;
 * MultiMatAlgo picks the policy per material. Same names, same shapes, and
 * the one difference: LoopsSimd adds `simd` to the OpenMP directives, which
 * vectorises the std::vector path and would only cost on the chunk path
 * (a chunk element access is a proxy, not a lane).
 */

#include <tuple>

#include "config.h"
#include "PMP.h"

/// @brief Loop policy of the chunk materials: work-shared, never vectorised.
struct Loops
{
    /// @brief `lbd(k)` for k in [kdeb, kfin), shared by the team, no barrier.
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

    /// @brief `lbd(ll[k])` for the first @p nn entries, shared by the team.
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

    /// @brief omp_simd_foreach_int() without the barrier at the end.
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

    /// @brief `lbd(i)` for every index i of the indirection @p ll.
    template <typename LBD, typename LL>
    void omp_simd_foreach(const LL& ll, const LBD& lbd) const
    {
        omp_simd_foreach_int(ll, lbd, static_cast<int>(ll.size()));
    }

    /// @brief omp_simd_foreach() without the barrier at the end.
    template <typename LBD, typename LL>
    void omp_simd_foreach_nowait(const LL& ll, const LBD& lbd) const
    {
        omp_simd_foreach_nowait_int(ll, lbd, static_cast<int>(ll.size()));
    }

    /**
     * @brief `lbd(ll[k], red)` over the thread's share of @p ll, then a
     *        barrier.
     * @details The range is split by hand (PMP.h) so that @p red, the
     *          caller's per-thread accumulator, is touched by one thread
     *          only; the caller reduces across threads.
     */
    template <typename LBD, typename LL, typename T>
    static void omp_foreach_simd_red_sum(const LL& ll, T& red, const LBD& lbd) noexcept
    {
        const int nn = static_cast<int>(ll.size());

        int kdeb             = 0;
        int kfin             = 0;
        std::tie(kdeb, kfin) = get_parallel_loop_boundaries(multi_mat::thread_rank(), multi_mat::max_threads(), nn);

        for (int k = kdeb; k < kfin; ++k)
        {
            const int nnn = ll[k];
            PRAGMA_FORCEINLINE
            lbd(nnn, red);
        }

#pragma omp barrier
    }
};

/// @brief Loop policy of the std::vector materials: work-shared and
///        vectorised (`omp simd`).
struct LoopsSimd
{
    /// @copydoc Loops::omp_simd_foreach_int
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

    /// @copydoc Loops::omp_simd_foreach_nowait_int
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

    /// @copydoc Loops::omp_simd_foreach
    template <typename LBD, typename LL>
    void omp_simd_foreach(const LL& ll, const LBD& lbd) const
    {
        omp_simd_foreach_int(ll, lbd, static_cast<int>(ll.size()));
    }

    /// @copydoc Loops::omp_simd_foreach_nowait
    template <typename LBD, typename LL>
    void omp_simd_foreach_nowait(const LL& ll, const LBD& lbd) const
    {
        omp_simd_foreach_nowait_int(ll, lbd, static_cast<int>(ll.size()));
    }

    /// @copydoc Loops::omp_for_nowait_simd
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

    /**
     * @copydoc Loops::omp_foreach_simd_red_sum
     * @note The accumulation goes through a local under `omp simd
     *       reduction`, so the lanes reduce before the thread does.
     */
    template <typename LBD, typename LL, typename T>
    void omp_foreach_simd_red_sum(const LL& ll, T& red, const LBD& lbd) const noexcept
    {
        const int nn = static_cast<int>(ll.size());

        int kdeb             = 0;
        int kfin             = 0;
        std::tie(kdeb, kfin) = get_parallel_loop_boundaries(multi_mat::thread_rank(), multi_mat::max_threads(), nn);

        T redl = red;
#pragma omp simd            reduction(+ : redl)
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
