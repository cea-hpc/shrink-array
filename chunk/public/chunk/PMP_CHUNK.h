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
    #include <atomic>
    #include <cstdlib> // std::getenv, std::atoi

    #include <omp.h>

    #include "chunk/internal/invariant_check.h" // INVARIANT_FAILED

namespace shrink_array
{
/**
 * @brief How many thread ranks the per-thread structures are sized for.
 * @details Every per-rank structure of the library -- the chunk allocators,
 *          the compression-candidate lists, the deferred-op maps, the writer
 *          and flush flags -- is indexed by the OpenMP thread rank and sized
 *          ONCE, when it is first built. It used to be sized by
 *          omp_get_max_threads() at that moment, so an omp_set_num_threads()
 *          raising the team after the arrays were created indexed past the
 *          end, silently. The capacity is fixed at the first call:
 *          max(omp_get_max_threads(), omp_get_num_procs()), raised by
 *          CHUNK_THREAD_CAPACITY if set. A rank at or past it stops the program
 *          with an invariant failure (chunk_pmp_checked_rank) instead of an
 *          out-of-bounds access. OpenMP gives no real upper bound
 *          (omp_get_thread_limit() is usually INT_MAX), hence the cores as the
 *          default ceiling and the variable for oversubscription.
 */
[[nodiscard]] inline int chunk_thread_capacity() noexcept
{
    static const int capacity = []
    {
        int cap = std::max(omp_get_max_threads(), omp_get_num_procs());
        if (const char* env = std::getenv("CHUNK_THREAD_CAPACITY"); env != nullptr)
            cap = std::max(cap, std::atoi(env));
        return cap;
    }();
    return capacity;
}

namespace pmp_detail
{
/// One past the highest rank ever cached (chunk_pmp_checked_rank); starts at 0.
inline std::atomic<int> g_rank_high_water{0};
} // namespace pmp_detail

/**
 * @brief Validate a rank against the capacity and record it; returns it.
 * @details Called where a rank is obtained or cached (begin_parallel_phase,
 *          the fallback of get_thr_rank_fast, chunk_pmp_thread_rank), not on
 *          every access. The high-water update is a CAS only when a rank grows
 *          past every rank seen so far.
 */
inline int chunk_pmp_checked_rank(int rank) noexcept
{
    if (rank >= chunk_thread_capacity()) [[unlikely]]
        INVARIANT_FAILED("thread rank {} but the per-thread structures hold {}: the OpenMP team grew past the thread "
                         "capacity fixed when the arrays were set up. Set OMP_NUM_THREADS (or call "
                         "omp_set_num_threads) to the largest team before creating the arrays, or raise "
                         "CHUNK_THREAD_CAPACITY.",
                         rank, chunk_thread_capacity());
    int seen = pmp_detail::g_rank_high_water.load(std::memory_order_relaxed);
    while (rank >= seen &&
           !pmp_detail::g_rank_high_water.compare_exchange_weak(seen, rank + 1, std::memory_order_seq_cst))
    {
    }
    return rank;
}

/**
 * @brief Upper bound of a loop over thread ranks that looks for work
 *        (candidate lists, deferred ops, writer flags).
 * @details omp_get_max_threads(), as these loops always used, but never below
 *          one past the highest rank seen (a team that shrank still has state
 *          in its upper ranks) and never above the capacity the structures
 *          were sized for. Without a team change it is exactly the former
 *          bound. Loops that SET an entry for every rank use the capacity.
 */
[[nodiscard]] inline int chunk_rank_bound() noexcept
{
    const int seen = pmp_detail::g_rank_high_water.load(std::memory_order_seq_cst);
    return std::min(chunk_thread_capacity(), std::max(omp_get_max_threads(), seen));
}
} // namespace shrink_array

inline int get_thr_rank_fast() noexcept
{
    // If OMPT is working, this value is already set at the start of the region.
    // No guards, no OpenMP library calls, just a direct TLS read.
    int rank = ::shrink_array::__fast_omp_rank;

    // Fallback if OMPT is disabled or not supported by the runtime
    if (rank == -1) [[unlikely]]
    {
        rank = ::shrink_array::__fast_omp_rank = ::shrink_array::chunk_pmp_checked_rank(omp_get_thread_num());
    }
    if constexpr (build_config::any_check)
    {
        assert(::shrink_array::__fast_omp_rank == omp_get_thread_num());
    }
    return rank;
}

    #ifndef CHUNK_PMP_THREAD_RANK

inline int chunk_pmp_thread_rank() noexcept
{
    return ::shrink_array::chunk_pmp_checked_rank(omp_get_thread_num());
}

        #define CHUNK_PMP_THREAD_RANK chunk_pmp_thread_rank()

    #endif

    #ifndef CHUNK_PMP_MAX_THREADS
        #define CHUNK_PMP_MAX_THREADS omp_get_max_threads()
    #endif

    /// Ranks the per-thread structures are sized for: size them with this,
    /// and loop over every rank with it when SETTING an entry per rank.
    #ifndef CHUNK_PMP_THREAD_CAPACITY
        #define CHUNK_PMP_THREAD_CAPACITY ::shrink_array::chunk_thread_capacity()
    #endif

    /// Bound of a loop over ranks that LOOKS for work: see chunk_rank_bound().
    #ifndef CHUNK_PMP_RANK_BOUND
        #define CHUNK_PMP_RANK_BOUND ::shrink_array::chunk_rank_bound()
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

    /// @copydoc CHUNK_PMP_THREAD_CAPACITY
    #ifndef CHUNK_PMP_THREAD_CAPACITY
        #define CHUNK_PMP_THREAD_CAPACITY 1
    #endif

    /// @copydoc CHUNK_PMP_RANK_BOUND
    #ifndef CHUNK_PMP_RANK_BOUND
        #define CHUNK_PMP_RANK_BOUND 1
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