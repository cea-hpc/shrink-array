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

#include <algorithm> // std::max (get_size_max_tab_omp)

#include "chunk/config.h"                         // build_config::any_check, build_config::threaded
#include "fast_allocators/internal/pmp_loops.h"   // fast_alloc::pmp: the loops and teams re-exported below
#include "fast_allocators/internal/pmp_runtime.h" // fast_alloc::pmp: the runtime queries, FAST_ALLOC_PMP_THREADED

/// @brief The library's threading backend (docs/threading_backends.md): the
///        data-parallel loops (use B), the teams (use C) and the runtime
///        queries (use A). One implementation, in fast_allocators/internal/
///        pmp_loops.h and pmp_runtime.h, shared with the allocator. The
///        CHUNK_PMP_* macros below are the library's names for the queries.
namespace shrink_array::pmp
{
using ::fast_alloc::pmp::active_level;
using ::fast_alloc::pmp::backend;
using ::fast_alloc::pmp::BackendKind;
using ::fast_alloc::pmp::current_team;
using ::fast_alloc::pmp::enter_application_phase;
using ::fast_alloc::pmp::in_application_phase;
using ::fast_alloc::pmp::leave_application_phase;
using ::fast_alloc::pmp::nowait;
using ::fast_alloc::pmp::Options;
using ::fast_alloc::pmp::parallel_for;
using ::fast_alloc::pmp::parallel_for_local;
using ::fast_alloc::pmp::parallel_sum;
using ::fast_alloc::pmp::SchedKind;
using ::fast_alloc::pmp::Schedule;
using ::fast_alloc::pmp::team;
using ::fast_alloc::pmp::Team;
using ::fast_alloc::pmp::team_any;
using ::fast_alloc::pmp::team_sum;
using ::fast_alloc::pmp::thread_rank;
using ::fast_alloc::pmp::threaded;
using ::fast_alloc::pmp::WaitMode;

static_assert(build_config::threaded == threaded,
              "chunk: the build's threading flag (CHUNK_OPT_THREADED) and the backend pmp_runtime.h selects disagree");
} // namespace shrink_array::pmp

namespace shrink_array
{
/// @brief Thread-local cache of the calling thread's rank in the active team;
///        -1 until get_thr_rank_fast() or begin_parallel_phase() fills it.
///        Not used by the std backend (chunk_rank_cache()).
inline thread_local int g_fast_omp_rank = -1;

/// @brief The calling thread's rank cache: g_fast_omp_rank, or, on the std backend, the
///        std backend's (pmp::rank_cache(), which it clears when a team changes the
///        thread's rank, as the OMPT implicit-task callback clears g_fast_omp_rank).
[[nodiscard]] inline int& chunk_rank_cache() noexcept
{
    if constexpr (::fast_alloc::pmp::backend == ::fast_alloc::pmp::BackendKind::std_threads)
    {
        return ::fast_alloc::pmp::rank_cache();
    }
    else
    {
        return ::shrink_array::g_fast_omp_rank;
    }
}

/// @brief The loop schedules a `schedule(runtime)` loop can be given -- the
///        values of omp_sched_t (OpenMP 5.x: static 1, dynamic 2, guided 3,
///        auto 4), in a type that exists in a serial build too. `int` like
///        omp_sched_t, which it is cast to and from.
// NOLINTNEXTLINE(performance-enum-size)
enum class ChunkPmpSched : int
{
    sched_static  = 1,
    sched_dynamic = 2,
    sched_guided  = 3,
    sched_auto    = 4,
};

/// @brief A runtime schedule: its kind and chunk (0: the runtime's default).
struct ChunkPmpSchedule
{
    ChunkPmpSched kind  = ChunkPmpSched::sched_static; ///< Schedule kind (omp_sched_t value).
    int           chunk = 0;                           ///< Chunk size; 0 lets the runtime pick its default.
};
} // namespace shrink_array

// The threaded backends (OpenMP today): the rank checks, the capacity and the
// queries, all through fast_alloc::pmp. Stays a directive: the serial branch
// below defines the same names as constants.
#if FAST_ALLOC_PMP_THREADED
    #include <atomic> // std::atomic

    #include "chunk/internal/invariant_check.h" // CHUNK_INVARIANT_FAILED
    #include "chunk/internal/env_parse.h"       // CHUNK_THREAD_CAPACITY

namespace shrink_array
{
/**
 * @brief How many thread ranks the per-thread structures are sized for.
 * @details Every per-rank structure (allocators, candidate lists, deferred-op maps,
 *          writer and flush flags) is sized ONCE when first built; sizing by
 *          omp_get_max_threads() would let a later omp_set_num_threads() index past
 *          the end, silently. Fixed at the first call: max(omp_get_max_threads(),
 *          omp_get_num_procs(), capacity_floor), raised by CHUNK_THREAD_CAPACITY. A rank
 *          at or past it is an invariant failure (chunk_pmp_checked_rank). OpenMP gives
 *          no real upper bound (omp_get_thread_limit() is usually INT_MAX), hence the
 *          cores as default -- and a floor, so that a team larger than the node's cores
 *          (a `num_threads()` clause, a component that picks its own team, a job
 *          granted a few cores) does not stop the program.
 *
 *          What a rank of capacity costs: the per-type, per-rank structures, about
 *          1 KiB per element type (measured: 2.8 KiB per rank for three types, RSS,
 *          from 2 to 4096 ranks), built once; the loops that look for work stop at the
 *          highest rank seen (chunk_rank_bound), not at the capacity. Eight arrays of
 *          32 MiB, each with its own allocator set: the same RSS from 2 to 1024.
 *
 *          Std backend: also the capacity of the rank registry the application's
 *          threads draw from (configured here, before its first use).
 */
inline constexpr int capacity_floor = 256;

[[nodiscard]] inline int chunk_thread_capacity() noexcept
{
    static const int capacity = []
    {
        int cap = std::max({::fast_alloc::pmp::max_threads(), ::fast_alloc::pmp::num_procs(), capacity_floor});
        cap     = std::max(cap, shrink_array::env_number<int>("CHUNK_THREAD_CAPACITY", 0));
        if constexpr (::fast_alloc::pmp::backend == ::fast_alloc::pmp::BackendKind::std_threads)
        {
            // false when the registry already exists (the allocator asked a rank
            // first): it was then sized by the same rule, see thread_rank_registry().
            (void)::fast_alloc::pmp::configure_thread_rank_registry(cap);
        }
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
        CHUNK_INVARIANT_FAILED(
            "thread rank {} but the per-thread structures hold {}: the OpenMP team grew past the thread "
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
 * @details omp_get_thread_num() is the rank in the INNERMOST team: in a serialised
 *          nested region it is 0 for every thread, which would make outer threads share
 *          rank 0's (unprotected, per-rank) structures. So the active team's rank comes
 *          from omp_get_ancestor_thread_num. Two ACTIVE levels are not supported (ranks of
 *          different inner teams collide): invariant failure.
 *
 *          Std backend: pmp::thread_rank() -- the rank in the active team of the std backend,
 *          or, in none, the thread's registered rank (the application's threads and the
 *          thread that drives the library between teams). A full registry (more
 *          threads alive in the library than chunk_thread_capacity()) is an invariant
 *          failure, as a rank past the capacity is.
 */
[[nodiscard]] inline int chunk_pmp_library_rank() noexcept
{
    namespace rt = ::fast_alloc::pmp;
    if constexpr (rt::backend == rt::BackendKind::std_threads)
    {
        if (rt::active_level() > 1) [[unlikely]]
        {
            CHUNK_INVARIANT_FAILED(
                "nested ACTIVE teams (active level {}) are not supported by the chunk arrays: per-thread "
                "ranks of different inner teams would collide. Keep max_active_levels at 1 (the default).",
                rt::active_level());
        }
        (void)chunk_thread_capacity(); // sizes the registry before its first rank
        const int rank = rt::thread_rank();
        if (rank < 0) [[unlikely]]
        {
            CHUNK_INVARIANT_FAILED(
                "std backend: no rank left for this thread -- {} threads already hold one, the capacity "
                "the per-thread structures were sized for. Raise CHUNK_THREAD_CAPACITY, or let threads "
                "that stopped using the arrays exit (or call pmp::release_registered_rank()).",
                rt::thread_rank_registry().capacity());
        }
        return rank;
    }
    const int level = rt::level();
    if (level <= 1) [[likely]]
    {
        return rt::team_rank();
    }
    if (rt::active_level() > 1) [[unlikely]]
    {
        CHUNK_INVARIANT_FAILED(
            "nested ACTIVE parallel regions (active level {}) are not supported by the chunk arrays: "
            "per-thread ranks of different inner teams would collide. Keep the inner regions "
            "serialised (OMP_MAX_ACTIVE_LEVELS=1, the default) or do not touch the arrays in them.",
            rt::active_level());
    }
    for (int l = level; l >= 1; --l)
    {
        if (rt::team_size_at(l) > 1)
        {
            return rt::ancestor_rank(l);
        }
    }
    return 0; // every level serialised: a single thread
}

/**
 * @brief Upper bound of a loop over thread ranks that looks for work
 *        (candidate lists, deferred ops, writer flags).
 * @details omp_get_max_threads(), but never below one past the highest rank seen
 *          (a team that shrank still has state in its upper ranks) and never above
 *          the capacity. Loops that SET an entry for every rank use the capacity.
 */
[[nodiscard]] inline int chunk_rank_bound() noexcept
{
    const int seen = internal::g_rank_high_water.load(std::memory_order_seq_cst);
    return std::min(chunk_thread_capacity(), std::max(::fast_alloc::pmp::max_threads(), seen));
}
/// @brief The calling task's run-sched-var (omp_get_schedule).
[[nodiscard]] inline ChunkPmpSchedule chunk_pmp_get_schedule() noexcept
{
    const auto s = ::fast_alloc::pmp::get_runtime_schedule();
    // As is, modifier bits included (omp_sched_monotonic): the enum's int
    // holds them, so a get / set round trip restores the caller's exactly.
    return {.kind = static_cast<ChunkPmpSched>(s.kind), .chunk = s.chunk};
}

/// @brief Set the calling task's run-sched-var (omp_set_schedule): the
///        schedule of the `schedule(runtime)` loops it then opens.
inline void chunk_pmp_set_schedule(ChunkPmpSchedule schedule) noexcept
{
    ::fast_alloc::pmp::set_runtime_schedule({.kind = static_cast<int>(schedule.kind), .chunk = schedule.chunk});
}

/// @brief The calling thread's number in its current team (omp_get_thread_num),
///        unchecked: for diagnostics that must work from anywhere, where
///        CHUNK_PMP_THREAD_RANK would check nesting and the rank bound.
[[nodiscard]] inline int chunk_pmp_team_rank() noexcept
{
    return ::fast_alloc::pmp::team_rank();
}

/// @brief The team size of the next parallel regions (omp_set_num_threads).
inline void chunk_pmp_set_num_threads(int n) noexcept
{
    ::fast_alloc::pmp::set_num_threads(n);
}

/// @brief How many nested levels may be active (omp_set_max_active_levels).
inline void chunk_pmp_set_max_active_levels(int levels) noexcept
{
    ::fast_alloc::pmp::set_max_active_levels(levels);
}

/// @brief Rank of the calling thread in the active OpenMP team, read from the thread-local cache.
/// @return The cached rank; filled from the OpenMP library on first use when the cache holds -1.
inline int get_thr_rank_fast() noexcept
{
    // If OMPT is working, this value is already set at the start of the region.
    // No guards, no OpenMP library calls, just a direct TLS read.
    int& cache = ::shrink_array::chunk_rank_cache();
    int  rank  = cache;

    // Fallback if OMPT is disabled or not supported by the runtime
    if (rank == -1) [[unlikely]]
    {
        rank = cache = ::shrink_array::chunk_pmp_checked_rank(::shrink_array::chunk_pmp_library_rank());
    }
    if constexpr (build_config::any_check)
    {
        // A stale cache (a rank left over from another region) breaks the per-rank
        // queues: fail in release check builds too.
        if (rank != ::shrink_array::chunk_pmp_library_rank()) [[unlikely]]
        {
            CHUNK_INVARIANT_FAILED("stale thread-rank cache: {} but this thread is rank {} of the active team", rank,
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

        #define CHUNK_PMP_THREAD_RANK ::shrink_array::chunk_pmp_thread_rank()

    #endif
} // namespace shrink_array

    #ifndef CHUNK_PMP_MAX_THREADS
        #define CHUNK_PMP_MAX_THREADS ::fast_alloc::pmp::max_threads()
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
    /// Use it rather than omp_get_num_threads(), which breaks the serial build.
    #ifndef CHUNK_PMP_NUM_THREADS
        #define CHUNK_PMP_NUM_THREADS ::fast_alloc::pmp::team_size()
    #endif

    #ifndef CHUNK_PMP_IN_PARALLEL_REGION
        #define CHUNK_PMP_IN_PARALLEL_REGION ::fast_alloc::pmp::in_parallel()
    #endif

    /// Nesting level of the calling thread (omp_get_level): 0 outside any
    /// region, 1 in a region, 2 in a region nested in it (serialised or not).
    #ifndef CHUNK_PMP_LEVEL
        #define CHUNK_PMP_LEVEL ::fast_alloc::pmp::level()
    #endif

#else

namespace shrink_array
{
/// @copydoc chunk_pmp_get_schedule -- serial build: no runtime schedule.
[[nodiscard]] inline ChunkPmpSchedule chunk_pmp_get_schedule() noexcept
{
    return {};
}
/// @copydoc chunk_pmp_set_schedule -- serial build: nothing to set.
inline void chunk_pmp_set_schedule(ChunkPmpSchedule) noexcept {}
/// @copydoc chunk_pmp_team_rank -- serial build: always 0.
[[nodiscard]] inline int chunk_pmp_team_rank() noexcept
{
    return 0;
}
/// @copydoc chunk_pmp_set_num_threads -- serial build: no team to size.
inline void chunk_pmp_set_num_threads(int) noexcept {}
/// @copydoc chunk_pmp_set_max_active_levels -- serial build: no levels.
inline void chunk_pmp_set_max_active_levels(int) noexcept {}

/// @brief Serial build: the calling thread's rank, always 0.
inline int get_thr_rank_fast() noexcept
{
    return 0;
}
} // namespace shrink_array
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

    /// @copydoc CHUNK_PMP_LEVEL
    #ifndef CHUNK_PMP_LEVEL
        #define CHUNK_PMP_LEVEL 0
    #endif

#endif

namespace shrink_array
{
/// @brief Size of a per-thread table for @p nn elements.
/// @param nn Number of elements.
/// @return nn / max_threads + max_threads inside a parallel region when nn exceeds the thread count (>1),
///         otherwise @p nn.
inline int get_size_max_tab_omp(int nn)
{
    if constexpr (build_config::threaded)
    {
        const int max_thr = CHUNK_PMP_IN_PARALLEL_REGION ? CHUNK_PMP_MAX_THREADS : 1;
        if (nn > max_thr && max_thr > 1)
        {
            const int pp = (nn / std::max(1, max_thr)) + max_thr;
            return pp;
        }
    }
    return nn;
}

/// @brief Size of a per-thread table for @p nn elements, the per-thread share rounded to static blocks.
/// @param nn Number of elements.
/// @param nb_static Block size the per-thread share (nn / max_threads) is rounded up past.
/// @return ((nn / max_threads) / nb_static + 1) * nb_static + max_threads inside a parallel region when nn
///         exceeds the thread count (>1), otherwise @p nn.
inline int get_size_max_tab_omp(int nn, [[maybe_unused]] int nb_static)
{
    if constexpr (build_config::threaded)
    {
        const int max_thr = CHUNK_PMP_IN_PARALLEL_REGION ? CHUNK_PMP_MAX_THREADS : 1;

        if (nn > max_thr && max_thr > 1)
        {
            int       pp  = nn / max_thr;
            const int nbs = pp / nb_static;

            pp = ((nbs + 1) * nb_static) + max_thr;

            return pp;
        }
    }
    return nn;
}

} // namespace shrink_array
