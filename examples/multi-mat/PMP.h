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

#include <utility>

#include "config.h"

/**
 * @brief The share of @p nb iterations of thread @p trank among
 *        @p nb_max_tasks, as a half-open range.
 * @details The first `nb % nb_max_tasks` threads take one more iteration.
 *          Fewer iterations than threads: thread 0 takes them all.
 * @param trank        Rank of the calling thread.
 * @param nb_max_tasks Size of the team.
 * @param nb           Iteration count.
 * @param[out] kdeb    First iteration of the thread.
 * @param[out] kfin    One past its last.
 */
inline void get_parallel_loop_boundaries_int(int trank, int nb_max_tasks, int nb, int& kdeb, int& kfin)
{
    if (nb > nb_max_tasks)
    {
        const int Q = nb / nb_max_tasks;
        const int R = nb % nb_max_tasks;
        if (trank < R)
        {
            const int nb_cells_spec = Q + 1;
            kdeb                    = trank * nb_cells_spec;
            kfin                    = kdeb + nb_cells_spec;
        }
        else
        {
            const int nb_cells_spec = Q;
            kdeb                    = R * (Q + 1) + (trank - R) * nb_cells_spec;
            kfin                    = kdeb + nb_cells_spec;
        }
    }
    else
    {
        if (trank)
        {
            kdeb = kfin = 0;
        }
        else
        {
            kdeb = 0;
            kfin = nb;
        }
    }
}

/**
 * @brief get_parallel_loop_boundaries_int() inside a parallel region, the
 *        whole range outside one.
 * @copydetails get_parallel_loop_boundaries_int
 */
inline void get_parallel_loop_boundaries(const int trank, const int nb_max_tasks, const int nb, int& kdeb, int& kfin)
{
    if (multi_mat::in_parallel_region())
    {
        get_parallel_loop_boundaries_int(trank, nb_max_tasks, nb, kdeb, kfin);
    }
    else
    {
        kdeb = 0;
        kfin = nb;
    }
}

/// @brief get_parallel_loop_boundaries() returning the pair {kdeb, kfin}.
inline std::pair<int, int> get_parallel_loop_boundaries(const int trank, const int nb_max_tasks, const int nb)
{
    int kdeb, kfin;
    get_parallel_loop_boundaries(trank, nb_max_tasks, nb, kdeb, kfin);
    return std::make_pair(kdeb, kfin);
}
