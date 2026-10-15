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

// The runtime queries of the library's threading backend (docs/threading_backends.md,
// use A): which backend the build runs on, the calling thread's rank and team, the
// thread counts, the nesting level, the schedule of the `schedule(runtime)` loops --
// and the registry that gives a dense rank to a thread no team numbers.
//
// The one place that asks the threading runtime these questions: chunk's
// PMP_CHUNK.h (CHUNK_PMP_*), pmp_alloc.h (FAST_ALLOC_PMP_*) and slab_arena.h
// go through it. The loops and teams are in pmp_loops.h.
//
// Backends: OpenMP (the default when the build compiles with OpenMP: every
// query is the omp_* call of the same name, inlined), pool (std::jthread
// workers, FAST_ALLOC_PMP_POOL=1, which the targets of a CHUNK_THREADS=pool
// build add: the queries read the calling thread's frames, pmp_pool_state.h)
// and serial (one thread, rank 0, no region).

#include <atomic> // std::atomic (the rank registry)
#include <memory> // std::unique_ptr (the registry's slots)
#include <thread> // std::thread::hardware_concurrency (the registry's default capacity)

/// 1 for the pool backend (FAST_ALLOC_PMP_POOL=1 on the command line, which the
/// CMake targets of a CHUNK_THREADS=pool build add), 0 otherwise.
#if defined(FAST_ALLOC_PMP_POOL) && FAST_ALLOC_PMP_POOL
    #define FAST_ALLOC_PMP_BACKEND_POOL 1
#else
    #define FAST_ALLOC_PMP_BACKEND_POOL 0
#endif

#if defined(_OPENMP) && FAST_ALLOC_PMP_BACKEND_POOL
    #error "FAST_ALLOC_PMP_POOL=1 (the pool backend) in a file compiled with OpenMP: one backend per build"
#endif

#ifdef _OPENMP
    #include <omp.h> // omp_get_thread_num, omp_get_max_threads, omp_in_parallel, omp_get_level, ...
#elif FAST_ALLOC_PMP_BACKEND_POOL
    #include "fast_allocators/internal/pmp_pool_state.h" // pool::tls, pool::settings: what the queries read
#endif

/// 1 when the build's backend runs threads (OpenMP, pool), 0 for the serial
/// backend: for the declaration-level switches, where an `if constexpr` cannot
/// sit. Statements test fast_alloc::pmp::threaded instead.
#if defined(_OPENMP) || FAST_ALLOC_PMP_BACKEND_POOL
    #define FAST_ALLOC_PMP_THREADED 1
#else
    #define FAST_ALLOC_PMP_THREADED 0
#endif

namespace fast_alloc::pmp
{
/// @brief The threading backends a build can run the library on.
enum class backend_kind : int
{
    serial, ///< One thread: rank 0, no region, no team.
    openmp, ///< OpenMP: the queries are the omp_* calls, the teams OpenMP regions.
    pool,   ///< std::jthread workers of the library's own (pmp_pool.h).
};

/// @brief The backend of this build (fixed at compile time).
inline constexpr backend_kind backend = FAST_ALLOC_PMP_BACKEND_POOL ? backend_kind::pool
                                        : FAST_ALLOC_PMP_THREADED   ? backend_kind::openmp
                                                                    : backend_kind::serial;

/// @brief Whether the backend runs threads: false only for the serial backend.
inline constexpr bool threaded = backend != backend_kind::serial;

// ---------------------------------------------------------------------------
// The queries. OpenMP: the omp_* call of the same meaning. Serial: one thread.
// ---------------------------------------------------------------------------

/// @brief Team size of the next region opened without num_threads (omp_get_max_threads).
[[nodiscard]] inline int max_threads() noexcept
{
#ifdef _OPENMP
    return omp_get_max_threads();
#elif FAST_ALLOC_PMP_BACKEND_POOL
    return pool::nthreads_var();
#else
    return 1;
#endif
}

/// @brief Processors the backend may run threads on (omp_get_num_procs).
[[nodiscard]] inline int num_procs() noexcept
{
#ifdef _OPENMP
    return omp_get_num_procs();
#elif FAST_ALLOC_PMP_BACKEND_POOL
    const int n = static_cast<int>(std::thread::hardware_concurrency());
    return n > 0 ? n : 1;
#else
    return 1;
#endif
}

/// @brief The calling thread's number in its INNERMOST team (omp_get_thread_num).
[[nodiscard]] inline int team_rank() noexcept
{
#ifdef _OPENMP
    return omp_get_thread_num();
#elif FAST_ALLOC_PMP_BACKEND_POOL
    return pool::tls().innermost().rank;
#else
    return 0;
#endif
}

/// @brief Size of the calling thread's innermost team (omp_get_num_threads).
[[nodiscard]] inline int team_size() noexcept
{
#ifdef _OPENMP
    return omp_get_num_threads();
#elif FAST_ALLOC_PMP_BACKEND_POOL
    return pool::tls().innermost().size;
#else
    return 1;
#endif
}

/// @brief Whether the calling thread is inside an ACTIVE region (omp_in_parallel).
[[nodiscard]] inline bool in_parallel() noexcept
{
#ifdef _OPENMP
    return omp_in_parallel() != 0;
#elif FAST_ALLOC_PMP_BACKEND_POOL
    return pool::tls().active_level > 0;
#else
    return false;
#endif
}

/// @brief Nesting level of the calling thread, serialised regions included
///        (omp_get_level): 0 outside any region.
[[nodiscard]] inline int level() noexcept
{
#ifdef _OPENMP
    return omp_get_level();
#elif FAST_ALLOC_PMP_BACKEND_POOL
    return pool::tls().level;
#else
    return 0;
#endif
}

/// @brief Nesting level counting only the ACTIVE regions (omp_get_active_level).
[[nodiscard]] inline int active_level() noexcept
{
#ifdef _OPENMP
    return omp_get_active_level();
#elif FAST_ALLOC_PMP_BACKEND_POOL
    return pool::tls().active_level;
#else
    return 0;
#endif
}

/// @brief Size of the team at nesting level @p lvl of the calling thread
///        (omp_get_team_size); -1 for a level it is not in.
[[nodiscard]] inline int team_size_at(int lvl) noexcept
{
#ifdef _OPENMP
    return omp_get_team_size(lvl);
#elif FAST_ALLOC_PMP_BACKEND_POOL
    const auto& s = pool::tls();
    if (lvl == 0)
    {
        return 1;
    }
    if (lvl < 0 || lvl > s.level)
    {
        return -1;
    }
    return lvl <= pool::kMaxFrames ? s.frames[lvl - 1].size : 1;
#else
    return lvl == 0 ? 1 : -1;
#endif
}

/// @brief The calling thread's ancestor's number at nesting level @p lvl
///        (omp_get_ancestor_thread_num); -1 for a level it is not in.
[[nodiscard]] inline int ancestor_rank(int lvl) noexcept
{
#ifdef _OPENMP
    return omp_get_ancestor_thread_num(lvl);
#elif FAST_ALLOC_PMP_BACKEND_POOL
    const auto& s = pool::tls();
    if (lvl == 0)
    {
        return 0;
    }
    if (lvl < 0 || lvl > s.level)
    {
        return -1;
    }
    return lvl <= pool::kMaxFrames ? s.frames[lvl - 1].rank : 0;
#else
    return lvl == 0 ? 0 : -1;
#endif
}

/// @brief The team size of the next regions opened without num_threads (omp_set_num_threads).
inline void set_num_threads([[maybe_unused]] int n) noexcept
{
#ifdef _OPENMP
    omp_set_num_threads(n);
#elif FAST_ALLOC_PMP_BACKEND_POOL
    pool::settings().nthreads.store(n > 0 ? n : 0, std::memory_order_relaxed);
#endif
}

/// @brief How many nested levels may be active (omp_set_max_active_levels).
inline void set_max_active_levels([[maybe_unused]] int levels) noexcept
{
#ifdef _OPENMP
    omp_set_max_active_levels(levels);
#elif FAST_ALLOC_PMP_BACKEND_POOL
    pool::settings().max_active_levels.store(levels > 0 ? levels : 0, std::memory_order_relaxed);
#endif
}

/// @brief A runtime schedule: the omp_sched_t value as an int (static 1, dynamic 2,
///        guided 3, auto 4, modifier bits included) and its chunk (0: the default).
struct runtime_schedule
{
    int kind  = 1; ///< omp_sched_t value; 1 = static.
    int chunk = 0; ///< Chunk size; 0 lets the runtime pick its default.
};

/// @brief The calling task's schedule of the `schedule(runtime)` loops (omp_get_schedule).
[[nodiscard]] inline runtime_schedule get_runtime_schedule() noexcept
{
#ifdef _OPENMP
    omp_sched_t kind  = omp_sched_static;
    int         chunk = 0;
    omp_get_schedule(&kind, &chunk);
    return {.kind = static_cast<int>(kind), .chunk = chunk};
#elif FAST_ALLOC_PMP_BACKEND_POOL
    return {.kind  = pool::settings().sched_kind.load(std::memory_order_relaxed),
            .chunk = pool::settings().sched_chunk.load(std::memory_order_relaxed)};
#else
    return {};
#endif
}

/// @brief Set the calling task's schedule of the `schedule(runtime)` loops (omp_set_schedule).
inline void set_runtime_schedule([[maybe_unused]] runtime_schedule s) noexcept
{
#ifdef _OPENMP
    omp_set_schedule(static_cast<omp_sched_t>(s.kind), s.chunk);
#elif FAST_ALLOC_PMP_BACKEND_POOL
    pool::settings().sched_kind.store(s.kind, std::memory_order_relaxed);
    pool::settings().sched_chunk.store(s.chunk, std::memory_order_relaxed);
#endif
}

// ---------------------------------------------------------------------------
// The rank registry: a dense rank for a thread no team numbers.
// ---------------------------------------------------------------------------

/**
 * @brief Dense ranks [0, capacity) handed to threads, the lowest free one
 *        first, and given back for reuse.
 * @details For the threads of a backend that does not number them (the
 *          application's std::thread or TBB workers, under the pool backend):
 *          the per-rank structures of the library want a small dense index,
 *          bounded by the capacity they were sized for. A thread takes a rank
 *          at its first registered_rank() and gives it back when it exits, so
 *          a program that keeps starting threads never runs out while no more
 *          than @c capacity of them are alive at once.
 *
 *          Lock-free: one atomic flag per rank. acquire() is a scan from 0,
 *          paid once per thread; release() a store.
 */
class RankRegistry
{
  public:
    /// @param capacity ranks it can hand out at once (at least 1)
    explicit RankRegistry(int capacity)
        : capacity_(capacity > 0 ? capacity : 1), taken_(std::make_unique<std::atomic<bool>[]>(capacity_))
    {
        for (int r = 0; r < capacity_; ++r)
        {
            taken_[r].store(false, std::memory_order_relaxed);
        }
    }

    RankRegistry(const RankRegistry&)            = delete;
    RankRegistry& operator=(const RankRegistry&) = delete;
    RankRegistry(RankRegistry&&)                 = delete;
    RankRegistry& operator=(RankRegistry&&)      = delete;
    ~RankRegistry()                              = default;

    /// @brief The ranks it can hand out at once.
    [[nodiscard]] int capacity() const noexcept
    {
        return capacity_;
    }

    /// @brief Take the lowest free rank; -1 when all @c capacity are taken.
    [[nodiscard]] int acquire() noexcept
    {
        for (int r = 0; r < capacity_; ++r)
        {
            if (!taken_[r].load(std::memory_order_relaxed) && !taken_[r].exchange(true, std::memory_order_acq_rel))
            {
                in_use_.fetch_add(1, std::memory_order_relaxed);
                int seen = high_water_.load(std::memory_order_relaxed);
                while (r >= seen && !high_water_.compare_exchange_weak(seen, r + 1, std::memory_order_relaxed))
                {
                }
                return r;
            }
        }
        return -1;
    }

    /// @brief Give @p rank back (a rank acquire() returned, once).
    void release(int rank) noexcept
    {
        if (rank >= 0 && rank < capacity_)
        {
            in_use_.fetch_sub(1, std::memory_order_relaxed);
            taken_[rank].store(false, std::memory_order_release);
        }
    }

    /// @brief Ranks currently taken.
    [[nodiscard]] int in_use() const noexcept
    {
        return in_use_.load(std::memory_order_relaxed);
    }

    /// @brief One past the highest rank ever handed out (0 before the first).
    [[nodiscard]] int high_water() const noexcept
    {
        return high_water_.load(std::memory_order_relaxed);
    }

  private:
    int                                  capacity_;
    std::unique_ptr<std::atomic<bool>[]> taken_;
    std::atomic<int>                     in_use_{0};
    std::atomic<int>                     high_water_{0};
};

namespace detail
{
/// Capacity the process-wide registry is built with (0: the default).
inline std::atomic<int> g_registry_capacity{0};
/// Set once the process-wide registry exists: its capacity is fixed.
inline std::atomic<bool> g_registry_built{false};
} // namespace detail

/**
 * @brief Fix the capacity of thread_rank_registry(), before its first use.
 * @return false when the registry already exists (its capacity stays).
 */
inline bool configure_thread_rank_registry(int capacity) noexcept
{
    if (detail::g_registry_built.load(std::memory_order_acquire))
    {
        return false;
    }
    detail::g_registry_capacity.store(capacity, std::memory_order_release);
    return true;
}

/**
 * @brief The process-wide registry registered_rank() draws from.
 * @details Built at its first call, with the capacity of
 *          configure_thread_rank_registry() or, by default, the hardware
 *          threads. Immortal (never destroyed): the threads give their rank
 *          back from their thread_local destructors, which can run after the
 *          static destructors at exit.
 */
[[nodiscard]] inline RankRegistry& thread_rank_registry()
{
    static RankRegistry* const registry = []
    {
        int cap = detail::g_registry_capacity.load(std::memory_order_acquire);
        if (cap <= 0)
        {
            cap = static_cast<int>(std::thread::hardware_concurrency());
        }
        auto* r = new RankRegistry(cap); // NOLINT(cppcoreguidelines-owning-memory): immortal
        detail::g_registry_built.store(true, std::memory_order_release);
        return r;
    }();
    return *registry;
}

namespace detail
{
/// The calling thread's registered rank (-1: none), given back when the thread exits.
struct RegisteredRank
{
    int rank = -1;

    RegisteredRank()                                 = default;
    RegisteredRank(const RegisteredRank&)            = delete;
    RegisteredRank& operator=(const RegisteredRank&) = delete;
    RegisteredRank(RegisteredRank&&)                 = delete;
    RegisteredRank& operator=(RegisteredRank&&)      = delete;
    ~RegisteredRank()
    {
        if (rank >= 0)
        {
            thread_rank_registry().release(rank);
        }
    }
};

[[nodiscard]] inline RegisteredRank& registered_rank_slot() noexcept
{
    thread_local RegisteredRank slot;
    return slot;
}
} // namespace detail

/**
 * @brief The calling thread's rank in thread_rank_registry(): taken at the
 *        first call, the same afterwards, given back when the thread exits
 *        (or at release_registered_rank()).
 * @return The rank, or -1 when the registry is full: the caller reports it (a
 *         thread past the capacity the per-rank structures were sized for). A
 *         later call tries again.
 */
[[nodiscard]] inline int registered_rank() noexcept
{
    auto& slot = detail::registered_rank_slot();
    if (slot.rank < 0) [[unlikely]]
    {
        slot.rank = thread_rank_registry().acquire();
    }
    return slot.rank;
}

/// @brief Give the calling thread's registered rank back before it exits; the
///        next registered_rank() takes one again.
inline void release_registered_rank() noexcept
{
    auto& slot = detail::registered_rank_slot();
    if (slot.rank >= 0)
    {
        thread_rank_registry().release(slot.rank);
        slot.rank = -1;
    }
}

} // namespace fast_alloc::pmp
