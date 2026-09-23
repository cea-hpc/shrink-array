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

#include <cassert>
#include "config.h"

namespace shrink_array
{
inline thread_local int __fast_omp_rank = -1;
}

#ifdef _OPENMP
    #include <algorithm>

    #include <omp.h>

inline int get_thr_rank_fast() noexcept
{
    // If OMPT is working, this value is already set at the start of the region.
    // No guards, no OpenMP library calls, just a direct TLS read.
    int r = ::shrink_array::__fast_omp_rank;

    // Fallback if OMPT is disabled or not supported by the runtime
    if (r == -1) [[unlikely]]
    {
        r = ::shrink_array::__fast_omp_rank = omp_get_thread_num();
    }
    if constexpr (build_config::any_check)
    {
        assert(::shrink_array::__fast_omp_rank == omp_get_thread_num());
    }
    return r;
}

    #ifndef CHUNK_PMP_THREAD_RANK

inline int chunk_pmp_thread_rank() noexcept
{
    return omp_get_thread_num();
}

        #define CHUNK_PMP_THREAD_RANK chunk_pmp_thread_rank()

    #endif

    #ifndef CHUNK_PMP_MAX_THREADS
        #define CHUNK_PMP_MAX_THREADS omp_get_max_threads()
    #endif

    /// Size of the team CURRENTLY running, as opposed to the maximum available.
    /// Needed by code that strides over buckets by thread rank inside a parallel
    /// region; calling omp_get_num_threads() directly breaks the serial build,
    /// which is what this header exists to prevent.
    #ifndef CHUNK_PMP_NUM_THREADS
        #define CHUNK_PMP_NUM_THREADS omp_get_num_threads()
    #endif

    #ifndef CHUNK_PMP_IN_PARALLEL_REGION
        #define CHUNK_PMP_IN_PARALLEL_REGION omp_in_parallel()
    #endif

#else

inline int get_thr_rank_fast() noexcept
{
    return 0;
}
    #ifndef CHUNK_PMP_THREAD_RANK
        #define CHUNK_PMP_THREAD_RANK 0
    #endif

    #ifndef CHUNK_PMP_MAX_THREADS
        #define CHUNK_PMP_MAX_THREADS 1
    #endif

    /// @copydoc CHUNK_PMP_NUM_THREADS
    #ifndef CHUNK_PMP_NUM_THREADS
        #define CHUNK_PMP_NUM_THREADS 1
    #endif

    #ifndef CHUNK_PMP_IN_PARALLEL_REGION
        #define CHUNK_PMP_IN_PARALLEL_REGION false
    #endif

#endif

inline int get_size_max_tab_omp(int nn)
{
#ifdef _OPENMP
    const int max_thr = CHUNK_PMP_IN_PARALLEL_REGION ? CHUNK_PMP_MAX_THREADS : 1;
    if (nn > max_thr && max_thr > 1)
    {
        const int pp = nn / std::max(1, max_thr) + max_thr;
        return pp;
    }
    else
    {
        return nn;
    }

#else
    return nn;
#endif
}

inline int get_size_max_tab_omp(int nn, [[maybe_unused]] int nb_static)
{
#ifdef _OPENMP
    const int max_thr = CHUNK_PMP_IN_PARALLEL_REGION ? CHUNK_PMP_MAX_THREADS : 1;

    if (nn > max_thr && max_thr > 1)
    {
        int       pp  = nn / max_thr;
        const int nbs = pp / nb_static;

        pp = (nbs + 1) * nb_static + max_thr;

        return pp;
    }
    else
    {
        return nn;
    }

#else
    return nn;
#endif
}