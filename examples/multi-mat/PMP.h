
/*
* Copyright 2024 CEA (www.cea.fr)
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

#include <utility>

#ifdef _OPENMP
#include <omp.h>

#define PMP_MAX_THREADS omp_get_max_threads()
#define PMP_THREAD_RANK omp_get_thread_num()
#define PMP_IN_PARALLEL_REGION omp_in_parallel()

#else

#define PMP_MAX_THREADS 1
#define PMP_THREAD_RANK 1
#define PMP_IN_PARALLEL_REGION false

#endif


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

inline void get_parallel_loop_boundaries(const int trank, const int nb_max_tasks, const int nb, int& kdeb, int& kfin)
{
    if (PMP_IN_PARALLEL_REGION)
    {
        get_parallel_loop_boundaries_int(trank, nb_max_tasks, nb, kdeb, kfin);
    }
    else
    {
        kdeb = 0;
        kfin = nb;
    }
}
inline std::pair<int, int> get_parallel_loop_boundaries(const int trank, const int nb_max_tasks, const int nb)
{
    int kdeb, kfin;

    if (PMP_IN_PARALLEL_REGION)
    {
        get_parallel_loop_boundaries_int(trank, nb_max_tasks, nb, kdeb, kfin);
    }
    else
    {
        kdeb = 0;
        kfin = nb;
    }

    return std::make_pair(kdeb, kfin);
}
