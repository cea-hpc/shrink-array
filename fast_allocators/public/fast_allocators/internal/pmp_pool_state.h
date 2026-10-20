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

// The state of the pool backend (docs/threading_backends.md, step 7) that its
// runtime queries read: the calling thread's stack of team frames, and the
// process-wide settings OpenMP calls its ICVs (team size of the next team,
// active levels allowed, the schedule of the run-time loops). The teams
// themselves are in pmp_pool.h; the queries in pmp_runtime.h.
//
// Standard library only.

#include <array>   // std::array (the frame stack)
#include <atomic>  // std::atomic (the settings)
#include <cstdlib> // std::getenv, std::strtol (CHUNK_NUM_THREADS, OMP_NUM_THREADS)
#include <thread>  // std::thread::hardware_concurrency (the default team size)

namespace fast_alloc::pmp::pool
{
struct TeamCtx; // pmp_pool.h

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
        auto& s = tls();
        if (s.level < max_frames)
        {
            s.frames[s.level] = Frame{.ctx = ctx, .rank = rank, .size = size};
        }
        ++s.level;
        active_     = size > 1;
        prev_rank_  = s.active_rank;
        prev_cache_ = s.rank_cache;
        if (active_)
        {
            ++s.active_level;
            s.active_rank = rank;
        }
        s.rank_cache = -1;
    }
    FrameGuard(const FrameGuard&)            = delete;
    FrameGuard& operator=(const FrameGuard&) = delete;
    FrameGuard(FrameGuard&&)                 = delete;
    FrameGuard& operator=(FrameGuard&&)      = delete;
    ~FrameGuard()
    {
        auto& s = tls();
        --s.level;
        if (active_)
        {
            --s.active_level;
        }
        s.active_rank = prev_rank_;
        s.rank_cache  = prev_cache_;
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
    /// Application threads, which no team of the pool numbers, may be in the
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
    static Settings* const s = new Settings; // NOLINT(cppcoreguidelines-owning-memory): immortal
    return *s;
}

/// @brief The leading number of environment variable @p name, 0 when unset or not a number.
[[nodiscard]] inline int env_threads(const char* name) noexcept
{
    const char* v = std::getenv(name); // NOLINT(concurrency-mt-unsafe): read once, at start-up
    if (v == nullptr)
    {
        return 0;
    }
    char*      end = nullptr;
    const long n   = std::strtol(v, &end, 10);
    return (end != v && n > 0 && n < 65536) ? static_cast<int>(n) : 0;
}

/// @brief The default team size: CHUNK_NUM_THREADS, else OMP_NUM_THREADS (its
///        first level), else the hardware threads.
[[nodiscard]] inline int default_threads() noexcept
{
    static const int n = []
    {
        int k = env_threads("CHUNK_NUM_THREADS");
        if (k == 0)
        {
            k = env_threads("OMP_NUM_THREADS");
        }
        if (k == 0)
        {
            k = static_cast<int>(std::thread::hardware_concurrency());
        }
        return k > 0 ? k : 1;
    }();
    return n;
}

/// @brief The size of the next team opened without a size (omp_get_max_threads).
[[nodiscard]] inline int nthreads_var() noexcept
{
    const int n = settings().nthreads.load(std::memory_order_relaxed);
    return n > 0 ? n : default_threads();
}
} // namespace fast_alloc::pmp::pool
