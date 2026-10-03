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

#include "chunk/config.h" // build_config::any_check

namespace shrink_array
{
/// @brief Thread-local cache of the calling thread's rank in the active team;
///        -1 until get_thr_rank_fast() or begin_parallel_phase() fills it.
inline thread_local int g_fast_omp_rank = -1;

/// @brief The loop schedules a `schedule(runtime)` loop can be given -- the
///        values of omp_sched_t (OpenMP 5.x: static 1, dynamic 2, guided 3,
///        auto 4), in a type that exists in a serial build too.
enum class chunk_pmp_sched : int
{
    sched_static  = 1,
    sched_dynamic = 2,
    sched_guided  = 3,
    sched_auto    = 4,
};

/// @brief A runtime schedule: its kind and chunk (0: the runtime's default).
struct chunk_pmp_schedule
{
    chunk_pmp_sched kind  = chunk_pmp_sched::sched_static; ///< Schedule kind (omp_sched_t value).
    int             chunk = 0;                             ///< Chunk size; 0 lets the runtime pick its default.
};
} // namespace shrink_array

#ifdef _OPENMP
    #include <algorithm> // std::max, std::min
    #include <atomic>    // std::atomic

    #include <omp.h> // omp_get_thread_num, omp_get_max_threads, omp_in_parallel

    #include "chunk/internal/invariant_check.h" // INVARIANT_FAILED
    #include "chunk/internal/env_parse.h"       // CHUNK_THREAD_CAPACITY

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
        cap     = std::max(cap, shrink_array::env_number<int>("CHUNK_THREAD_CAPACITY", 0));
        return cap;
    }();
    return capacity;
}

namespace internal
{
/// One past the highest rank ever cached (chunk_pmp_checked_rank); starts at 0.
inline std::atomic<int> g_rank_high_water{0};
} // namespace internal

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
    {
        INVARIANT_FAILED("thread rank {} but the per-thread structures hold {}: the OpenMP team grew past the thread "
                         "capacity fixed when the arrays were set up. Set OMP_NUM_THREADS (or call "
                         "omp_set_num_threads) to the largest team before creating the arrays, or raise "
                         "CHUNK_THREAD_CAPACITY.",
                         rank, chunk_thread_capacity());
    }
    int seen = internal::g_rank_high_water.load(std::memory_order_relaxed);
    while (rank >= seen &&
           !internal::g_rank_high_water.compare_exchange_weak(seen, rank + 1, std::memory_order_seq_cst))
    {
    }
    return rank;
}

/**
 * @brief The rank the library indexes its per-thread structures with: the
 *        thread's rank in the one ACTIVE team.
 * @details omp_get_thread_num() is the rank in the INNERMOST team. Outside any
 *          nesting that is the same thing. In a nested region that is
 *          serialised (a team of one: the default, max-active-levels 1) it is
 *          0 for every thread, so thread k of the outer team would share rank
 *          0's candidate list and deferred-op queue with the real rank 0 --
 *          unprotected, those are per-rank by design. The active team's rank
 *          is found with omp_get_ancestor_thread_num. Two ACTIVE levels (inner
 *          teams of several threads) are not supported: ranks of different
 *          inner teams collide, so it is an invariant failure.
 */
[[nodiscard]] inline int chunk_pmp_library_rank() noexcept
{
    const int level = omp_get_level();
    if (level <= 1) [[likely]]
    {
        return omp_get_thread_num();
    }
    if (omp_get_active_level() > 1) [[unlikely]]
    {
        INVARIANT_FAILED("nested ACTIVE parallel regions (active level {}) are not supported by the chunk arrays: "
                         "per-thread ranks of different inner teams would collide. Keep the inner regions "
                         "serialised (OMP_MAX_ACTIVE_LEVELS=1, the default) or do not touch the arrays in them.",
                         omp_get_active_level());
    }
    for (int l = level; l >= 1; --l)
    {
        if (omp_get_team_size(l) > 1)
        {
            return omp_get_ancestor_thread_num(l);
        }
    }
    return 0; // every level serialised: a single thread
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
    const int seen = internal::g_rank_high_water.load(std::memory_order_seq_cst);
    return std::min(chunk_thread_capacity(), std::max(omp_get_max_threads(), seen));
}
/// @brief The calling task's run-sched-var (omp_get_schedule).
[[nodiscard]] inline chunk_pmp_schedule chunk_pmp_get_schedule() noexcept
{
    omp_sched_t kind  = omp_sched_static;
    int         chunk = 0;
    omp_get_schedule(&kind, &chunk);
    // As is, modifier bits included (omp_sched_monotonic): the enum's int
    // holds them, so a get / set round trip restores the caller's exactly.
    return {static_cast<chunk_pmp_sched>(kind), chunk};
}

/// @brief Set the calling task's run-sched-var (omp_set_schedule): the
///        schedule of the `schedule(runtime)` loops it then opens.
inline void chunk_pmp_set_schedule(chunk_pmp_schedule schedule) noexcept
{
    omp_set_schedule(static_cast<omp_sched_t>(schedule.kind), schedule.chunk);
}

/// @brief The calling thread's number in its current team (omp_get_thread_num),
///        unchecked: for diagnostics that must work from anywhere, where
///        CHUNK_PMP_THREAD_RANK would check nesting and the rank bound.
[[nodiscard]] inline int chunk_pmp_team_rank() noexcept
{
    return omp_get_thread_num();
}

/// @brief The team size of the next parallel regions (omp_set_num_threads).
inline void chunk_pmp_set_num_threads(int n) noexcept
{
    omp_set_num_threads(n);
}

/// @brief How many nested levels may be active (omp_set_max_active_levels).
inline void chunk_pmp_set_max_active_levels(int levels) noexcept
{
    omp_set_max_active_levels(levels);
}
} // namespace shrink_array

/// @brief Rank of the calling thread in the active OpenMP team, read from the thread-local cache.
/// @return The cached rank; filled from the OpenMP library on first use when the cache holds -1.
inline int get_thr_rank_fast() noexcept
{
    // If OMPT is working, this value is already set at the start of the region.
    // No guards, no OpenMP library calls, just a direct TLS read.
    int rank = ::shrink_array::g_fast_omp_rank;

    // Fallback if OMPT is disabled or not supported by the runtime
    if (rank == -1) [[unlikely]]
    {
        rank = ::shrink_array::g_fast_omp_rank =
            ::shrink_array::chunk_pmp_checked_rank(::shrink_array::chunk_pmp_library_rank());
    }
    if constexpr (build_config::any_check)
    {
        // A stale cache (a rank left over from another region) is exactly
        // what the per-rank queues cannot survive: fail in release check
        // builds too, where an assert is compiled out.
        if (rank != ::shrink_array::chunk_pmp_library_rank()) [[unlikely]]
        {
            INVARIANT_FAILED("stale thread-rank cache: {} but this thread is rank {} of the active team", rank,
                             ::shrink_array::chunk_pmp_library_rank());
        }
    }
    return rank;
}

    #ifndef CHUNK_PMP_THREAD_RANK

/// @brief Rank of the calling thread, queried from the OpenMP library and checked.
/// @return chunk_pmp_checked_rank() of chunk_pmp_library_rank().
inline int chunk_pmp_thread_rank() noexcept
{
    return ::shrink_array::chunk_pmp_checked_rank(::shrink_array::chunk_pmp_library_rank());
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

namespace shrink_array
{
/// @copydoc chunk_pmp_get_schedule -- serial build: no runtime schedule.
[[nodiscard]] inline chunk_pmp_schedule chunk_pmp_get_schedule() noexcept
{
    return {};
}
/// @copydoc chunk_pmp_set_schedule -- serial build: nothing to set.
inline void chunk_pmp_set_schedule(chunk_pmp_schedule) noexcept {}
/// @copydoc chunk_pmp_team_rank -- serial build: always 0.
[[nodiscard]] inline int chunk_pmp_team_rank() noexcept
{
    return 0;
}
/// @copydoc chunk_pmp_set_num_threads -- serial build: no team to size.
inline void chunk_pmp_set_num_threads(int) noexcept {}
/// @copydoc chunk_pmp_set_max_active_levels -- serial build: no levels.
inline void chunk_pmp_set_max_active_levels(int) noexcept {}
} // namespace shrink_array

/// @brief Serial build: the calling thread's rank, always 0.
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

/// @brief Size of a per-thread table for @p nn elements.
/// @param nn Number of elements.
/// @return nn / max_threads + max_threads inside a parallel region when nn exceeds the thread count (>1),
///         otherwise @p nn.
inline int get_size_max_tab_omp(int nn)
{
#ifdef _OPENMP
    const int max_thr = CHUNK_PMP_IN_PARALLEL_REGION ? CHUNK_PMP_MAX_THREADS : 1;
    if (nn > max_thr && max_thr > 1)
    {
        const int pp = (nn / std::max(1, max_thr)) + max_thr;
        return pp;
    }
#endif
    return nn;
}

/// @brief Size of a per-thread table for @p nn elements, the per-thread share rounded to static blocks.
/// @param nn Number of elements.
/// @param nb_static Block size the per-thread share (nn / max_threads) is rounded up past.
/// @return ((nn / max_threads) / nb_static + 1) * nb_static + max_threads inside a parallel region when nn
///         exceeds the thread count (>1), otherwise @p nn.
inline int get_size_max_tab_omp(int nn, [[maybe_unused]] int nb_static)
{
#ifdef _OPENMP
    const int max_thr = CHUNK_PMP_IN_PARALLEL_REGION ? CHUNK_PMP_MAX_THREADS : 1;

    if (nn > max_thr && max_thr > 1)
    {
        int       pp  = nn / max_thr;
        const int nbs = pp / nb_static;

        pp = ((nbs + 1) * nb_static) + max_thr;

        return pp;
    }
#endif
    return nn;
}