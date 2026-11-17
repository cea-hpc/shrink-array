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
 * @file PMP.h
 * @brief Static partition of a loop between the threads of a team.
 *
 * The reduction loops of Loops.h split their index range by hand rather
 * than through `omp for`: the SIMD reduction inside then sees a contiguous
 * range per thread. The OpenMP queries come from config.h.
 */

#include <utility> // std::pair, std::make_pair

#include "config.h" // multi_mat::in_parallel_region

/**
 * @brief The share of @p n iterations of thread @p trank among
 *        @p n_max_tasks, as a half-open range.
 * @details The first `n % n_max_tasks` threads take one more iteration.
 *          Fewer iterations than threads: thread 0 takes them all.
 * @param trank        Rank of the calling thread.
 * @param n_max_tasks Size of the team.
 * @param n           Iteration count.
 * @param[out] kbegin    First iteration of the thread.
 * @param[out] kend    One past its last.
 */
inline void get_parallel_loop_boundaries_int(int trank, int n_max_tasks, int n, int& kbegin, int& kend)
{
    if (n > n_max_tasks)
    {
        const int quot = n / n_max_tasks;
        const int rem  = n % n_max_tasks;
        if (trank < rem)
        {
            const int n_cells_spec = quot + 1;
            kbegin                 = trank * n_cells_spec;
            kend                   = kbegin + n_cells_spec;
        }
        else
        {
            const int n_cells_spec = quot;
            kbegin                 = rem * (quot + 1) + (trank - rem) * n_cells_spec;
            kend                   = kbegin + n_cells_spec;
        }
    }
    else
    {
        if (trank)
        {
            kbegin = kend = 0;
        }
        else
        {
            kbegin = 0;
            kend   = n;
        }
    }
}

/**
 * @brief get_parallel_loop_boundaries_int() inside a parallel region, the
 *        whole range outside one.
 * @copydetails get_parallel_loop_boundaries_int
 */
inline void get_parallel_loop_boundaries(const int trank, const int n_max_tasks, const int n, int& kbegin, int& kend)
{
    if (multi_mat::in_parallel_region())
    {
        get_parallel_loop_boundaries_int(trank, n_max_tasks, n, kbegin, kend);
    }
    else
    {
        kbegin = 0;
        kend   = n;
    }
}

/// @brief get_parallel_loop_boundaries() returning the pair {kbegin, kend}.
inline std::pair<int, int> get_parallel_loop_boundaries(const int trank, const int n_max_tasks, const int n)
{
    int kbegin = 0;
    int kend   = 0;
    get_parallel_loop_boundaries(trank, n_max_tasks, n, kbegin, kend);
    return std::make_pair(kbegin, kend);
}
