/*
 * Copyright 2026 CEA (www.cea.fr)
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

// The state of the std backend (docs/threading_backends.md, step 7) that its
// runtime queries read: the calling thread's stack of team frames, and the
// process-wide settings OpenMP calls its ICVs (team size of the next team,
// active levels allowed, the schedule of the run-time loops). The teams
// themselves are in pmp_std.h; the queries in pmp_runtime.h.
//
// Standard library only.

#include <array>   // std::array (the frame stack)
#include <atomic>  // std::atomic (the settings)
#include <chrono>  // std::chrono::steady_clock (the spin budget)
#include <cstdlib> // std::getenv, std::strtol (CHUNK_NUM_THREADS, OMP_NUM_THREADS, CHUNK_PMP_SPIN_US)
#include <thread>  // std::thread::hardware_concurrency, std::this_thread::yield
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    #include <intrin.h> // _mm_pause (cpu_relax)
#endif

namespace fast_alloc::pmp::std_backend
{
struct TeamCtx; // pmp_std.h

/// @brief One level of team nesting, as the calling thread sees it.
struct Frame
{
    TeamCtx* ctx  = nullptr; ///< The team's shared state; nullptr for a team of one.
    int      rank = 0;       ///< The thread's rank in it.
    int      size = 1;       ///< Its size.
};

/// @brief Levels whose team the queries can still name; deeper ones count, as a team of one.
inline constexpr int max_frames = 16;

/// @brief The calling thread's nesting: the frames of the teams it is in, innermost last.
struct ThreadState
{
    std::array<Frame, max_frames> frames{};         ///< frames[l - 1]: the team of level l.
    int                           level        = 0; ///< Teams the thread is in, serialised ones included.
    int                           active_level = 0; ///< Those of more than one thread.
    /// The thread's rank in its innermost ACTIVE team (the one of more than one
    /// thread); -1 in none, where the thread's rank is its registered one
    /// (pmp_runtime.h, thread_rank()).
    int active_rank = -1;
    /// The library's cache of thread_rank() (chunk's get_thr_rank_fast): -1
    /// until filled; a team's entry clears it and its exit restores it, since
    /// the rank changes there (OpenMP's OMPT implicit-task callback does the same).
    int rank_cache = -1;

    /// @brief The innermost team's frame; a team of one outside any.
    [[nodiscard]] Frame innermost() const noexcept
    {
        return (level > 0 && level <= max_frames) ? frames[level - 1] : Frame{};
    }
};

/// @brief The calling thread's state.
[[nodiscard]] inline ThreadState& tls() noexcept
{
    thread_local ThreadState state;
    return state;
}

/// @brief Enters a team on the calling thread for its scope, leaves it after.
class FrameGuard
{
  public:
    FrameGuard(TeamCtx* ctx, int rank, int size) noexcept
    {
        auto& state = tls();
        if (state.level < max_frames)
        {
            state.frames[state.level] = Frame{.ctx = ctx, .rank = rank, .size = size};
        }
        ++state.level;
        active_     = size > 1;
        prev_rank_  = state.active_rank;
        prev_cache_ = state.rank_cache;
        if (active_)
        {
            ++state.active_level;
            state.active_rank = rank;
        }
        state.rank_cache = -1;
    }
    FrameGuard(const FrameGuard&)            = delete;
    FrameGuard& operator=(const FrameGuard&) = delete;
    FrameGuard(FrameGuard&&)                 = delete;
    FrameGuard& operator=(FrameGuard&&)      = delete;
    ~FrameGuard()
    {
        auto& state = tls();
        --state.level;
        if (active_)
        {
            --state.active_level;
        }
        state.active_rank = prev_rank_;
        state.rank_cache  = prev_cache_;
    }

  private:
    bool active_     = false;
    int  prev_rank_  = -1; ///< active_rank before the team: restored at its end.
    int  prev_cache_ = -1; ///< rank_cache before the team: still right after it.
};

/// @brief The process-wide settings (OpenMP's ICVs): set by the queries'
///        setters, read when a team opens.
struct Settings
{
    std::atomic<int> nthreads{0};          ///< Size of the next team; 0: default_threads().
    std::atomic<int> max_active_levels{1}; ///< Nested teams beyond this run as a team of one.
    std::atomic<int> sched_kind{1};        ///< omp_sched_t value of the run-time loops (1: static).
    std::atomic<int> sched_chunk{0};       ///< Their chunk (0: the default).
    /// Application threads, which no team of the std backend numbers, may be in the
    /// library: set by the first begin_parallel_phase() made outside any team,
    /// cleared by finalize_parallel_phase() (the phase contract,
    /// docs/threading_backends.md). While it is set, in_parallel() is true
    /// everywhere and no team of more than one thread opens: its workers'
    /// ranks would be those of the application's threads.
    std::atomic<bool> application_phase{false};
};

/// @brief The settings (one instance per process, never destroyed).
[[nodiscard]] inline Settings& settings() noexcept
{
    static Settings* const instance = new Settings; // NOLINT(cppcoreguidelines-owning-memory): immortal
    return *instance;
}

/// @brief The leading number of environment variable @p name, 0 when unset or not a number.
[[nodiscard]] inline int env_threads(const char* name) noexcept
{
    const char* value = std::getenv(name); // NOLINT(concurrency-mt-unsafe): read once, at start-up
    if (value == nullptr)
    {
        return 0;
    }
    char*      end    = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    return (end != value && parsed > 0 && parsed < 65536) ? static_cast<int>(parsed) : 0;
}

/// @brief The default team size: CHUNK_NUM_THREADS, else OMP_NUM_THREADS (its
///        first level), else the hardware threads.
[[nodiscard]] inline int default_threads() noexcept
{
    static const int threads = []
    {
        int requested = env_threads("CHUNK_NUM_THREADS");
        if (requested == 0)
        {
            requested = env_threads("OMP_NUM_THREADS");
        }
        if (requested == 0)
        {
            requested = static_cast<int>(std::thread::hardware_concurrency());
        }
        return requested > 0 ? requested : 1;
    }();
    return threads;
}

/// @brief The size of the next team opened without a size (omp_get_max_threads).
[[nodiscard]] inline int nthreads_var() noexcept
{
    const int nthreads = settings().nthreads.load(std::memory_order_relaxed);
    return nthreads > 0 ? nthreads : default_threads();
}

/// @brief How long a waiting thread of the backend spins before it sleeps, in
///        ns: CHUNK_PMP_SPIN_US (microseconds, default 1000; 0: sleep at once).
/// @details OpenMP's blocktime, for the barriers, singles and team hand-offs
///          of the std backend. A futex sleep and wake-up costs tens of
///          microseconds per thread, and a barrier of a large team wakes them
///          in a cascade (~0.4 ms at 160 threads, against ~20 us spinning):
///          spinning wins whenever the wait is shorter than the budget, which
///          is the case of the library's teams (the re-chunk's slice
///          boundaries: a few barriers every few milliseconds). The budget
///          bounds the CPU a waiting thread burns, for a long wait or an
///          oversubscribed machine.
[[nodiscard]] inline long long spin_budget_ns() noexcept
{
    static const long long budget = []
    {
        long long   us    = 1000;
        const char* value = std::getenv("CHUNK_PMP_SPIN_US"); // NOLINT(concurrency-mt-unsafe): read once
        if (value != nullptr)
        {
            char*      end    = nullptr;
            const long parsed = std::strtol(value, &end, 10);
            if (end != value && parsed >= 0 && parsed <= 10'000'000)
            {
                us = parsed;
            }
        }
        return us * 1000;
    }();
    return budget;
}

/// @brief A hint to the core that the thread spins (x86 pause, AArch64 yield).
inline void cpu_relax() noexcept
{
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#elif defined(__aarch64__) || defined(__arm__)
    asm volatile("yield" ::: "memory");
#elif defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    _mm_pause();
#endif
}

/// @brief The hardware threads (at least 1), read once.
[[nodiscard]] inline int hardware_threads() noexcept
{
    static const int threads = []
    {
        const auto n = static_cast<int>(std::thread::hardware_concurrency());
        return n > 0 ? n : 1;
    }();
    return threads;
}

/**
 * @brief Spin until @p done() is true or the spin budget is spent.
 * @param team_threads Threads of the team the caller waits with: more than
 *        the hardware threads, and the caller sleeps at once (a spinning
 *        waiter would hold the core of a thread it waits for -- libgomp
 *        throttles its spin count the same way).
 * @return @p done()'s last value: false means the caller must sleep.
 * @details Every 256 rounds the clock is read and the thread yields: when
 *          other threads want the cores (another runtime's idle workers, a
 *          busy machine), the one being waited for gets a turn.
 */
template <typename Done>
[[nodiscard]] bool spin_until(int team_threads, Done&& done) noexcept
{
    const long long budget = team_threads > hardware_threads() ? 0 : spin_budget_ns();
    if (budget <= 0)
    {
        return done();
    }
    const auto start = std::chrono::steady_clock::now();
    for (unsigned i = 1;; ++i)
    {
        if (done())
        {
            return true;
        }
        cpu_relax();
        if ((i & 255U) == 0U)
        {
            if (std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start)
                    .count() >= budget)
            {
                return done();
            }
            std::this_thread::yield();
        }
    }
}
} // namespace fast_alloc::pmp::std_backend
