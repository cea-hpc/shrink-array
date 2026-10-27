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

// The std backend of the library's threading (docs/threading_backends.md,
// step 7): the loops and teams of pmp_loops.h on std::jthread workers of the
// library's own. Included by pmp_loops.h only, after its shared types
// (options, schedule, WaitMode), in a build with FAST_ALLOC_PMP_STD=1.
//
// The model is OpenMP's, so the call sites need no change:
//   - a team: the calling thread is rank 0, workers 1..n-1 of a process-wide
//     pool (worker k is always rank k: the static loops of successive teams
//     give a rank the same block, so the pages land where they are used --
//     first touch, as with OpenMP's thread reuse);
//   - the worksharing (for_static, for_dynamic, for_runtime, single) is met by
//     every thread of the team in the same order; each thread counts the
//     constructs it met, and a construct that needs shared state (a dynamic
//     loop's next block, a single's claim) finds it by that count;
//   - barrier(): a std::barrier of the team; the end of a team is a join;
//   - a team opened inside a team (nested) runs as a team of one, as OpenMP's
//     default (max_active_levels 1); one team of the pool at a time: a second
//     thread opening one meanwhile waits for it.
//
// Standard library only.

#ifndef FAST_ALLOC_PMP_LOOPS_TYPES
    #error "pmp_std.h is included by pmp_loops.h, after the types it shares with the other backends"
#endif

#include <barrier>       // std::barrier (Team::barrier)
#include <cstdint>       // std::uint64_t (the pool's job word)
#include <exception>     // std::terminate (a body that throws)
#include <memory>        // std::unique_ptr (the construct slots)
#include <mutex>         // std::mutex, std::scoped_lock (the slots; one team at a time)
#include <thread>        // std::jthread (the workers)
#include <unordered_map> // std::unordered_map (the construct slots)
#include <vector>        // std::vector (the workers, the per-rank counts and results)

#include "fast_allocators/internal/pmp_std_state.h" // std_backend::tls, FrameGuard, settings, nthreads_var

namespace fast_alloc::pmp
{
namespace std_backend
{
/// @brief The shared state of one worksharing construct of a team: a dynamic
///        loop's next block, a single's claim.
struct Slot
{
    std::atomic<long long> next{0};        ///< Next unclaimed offset of a dynamic loop.
    std::atomic<bool>      claimed{false}; ///< A single: some thread took it.
    int                    left = 0;       ///< Threads that have not left it yet (under TeamCtx::mutex).
};

/// @brief What the threads of one team share.
struct TeamCtx
{
    explicit TeamCtx(int n) : size(n), bar(n), met(static_cast<std::size_t>(n), 0) {}

    const int                                                size;  ///< Threads in the team.
    std::barrier<>                                           bar;   ///< Team::barrier().
    std::mutex                                               mutex; ///< Guards slots.
    std::unordered_map<std::uint64_t, std::unique_ptr<Slot>> slots; ///< The constructs some thread is in.
    std::vector<std::uint64_t>                               met;   ///< met[rank]: constructs rank has entered.

    /// @brief Enter the next construct of @p rank; its slot, and its number in @p seq.
    Slot& enter(int rank, std::uint64_t& seq);

    /// @brief Leave construct @p seq; the last thread out frees its slot.
    void leave(std::uint64_t seq);
};

/**
 * @brief The process-wide workers: worker k runs rank k + 1 of every team
 *        of more than k + 1 threads.
 * @details A job is published as one word, (generation << 16) | team size, so
 *          a worker that is not in the team reads nothing else; the job's
 *          function is read only by the workers in it, which the dispatcher
 *          waits for before it writes the next one. Immortal: the workers sleep
 *          on the word until the process exits.
 */
class Pool
{
  public:
    /// @brief The pool (built at the first team, never destroyed).
    [[nodiscard]] static Pool& get()
    {
        static Pool* const p = new Pool; // NOLINT(cppcoreguidelines-owning-memory): immortal, see above
        return *p;
    }

    Pool(const Pool&)            = delete;
    Pool& operator=(const Pool&) = delete;
    Pool(Pool&&)                 = delete;
    Pool& operator=(Pool&&)      = delete;
    ~Pool()                      = default;

    /// @brief Run @p job(rank) for every rank of [0, @p n): rank 0 on the
    ///        calling thread, the others on the workers; return when all are done.
    template <typename Job>
    void run(int n, Job& job);

  private:
    Pool() = default;

    void call(int rank) noexcept;

    void worker_loop(int rank, std::uint64_t seen) noexcept;

    std::mutex                 team_mutex_; ///< One team at a time.
    std::vector<std::jthread>  workers_;    ///< Worker k: rank k + 1.
    std::atomic<std::uint64_t> word_{0};    ///< (generation << 16) | team size.
    std::atomic<int>           pending_{0}; ///< Workers of the current team not done yet.
    void (*fn_)(void*, int) = nullptr;      ///< The current job (read by its workers only).
    void* arg_              = nullptr;      ///< Its argument.
};

/// @brief [b, e) of @p rank in a static block split of n iterations over @p nt
///        threads (libgomp's: the first n % nt ranks get one more).
inline void static_block(long long n, int nt, int rank, long long& b, long long& e) noexcept
{
    const long long q = n / nt;
    const long long r = n % nt;
    if (rank < r)
    {
        b = rank * (q + 1);
        e = b + q + 1;
    }
    else
    {
        b = (r * (q + 1)) + ((rank - r) * q);
        e = b + q;
    }
}

/**
 * @brief Open a team of @p nt threads (fewer: a team of one, see below) and
 *        run @p body(Team&) on each.
 * @details A team of one when @p on is false, @p nt is 1, the calling
 *          thread is already in max_active_levels active teams, or an
 *          application phase is open (application threads hold the ranks the
 *          workers would take); it still counts as a level, as OpenMP's
 *          serialised regions do.
 */
template <typename TeamT, typename Body>
void open_team(bool on, int nt, Body& body)
{
    const auto& s = tls();
    if (!on || nt <= 1 || s.active_level >= settings().max_active_levels.load(std::memory_order_relaxed) ||
        settings().application_phase.load(std::memory_order_acquire))
    {
        const FrameGuard frame(nullptr, 0, 1);
        TeamT            t(nullptr, 0);
        body(t);
        return;
    }
    TeamCtx ctx(nt);
    auto    job = [&ctx, &body, nt](int rank)
    {
        const FrameGuard frame(&ctx, rank, nt);
        TeamT            t(&ctx, rank);
        body(t);
    };
    Pool::get().run(nt, job);
}
} // namespace std_backend

/**
 * @brief The team a thread runs in, as seen from that thread (see the OpenMP
 *        backend's in pmp_loops.h). Std backend: the team's shared state and
 *        the thread's rank in it; no state for a team of one.
 */
class Team
{
  public:
    Team() = default;
    /// @param ctx  the team's shared state (nullptr: a team of one)
    /// @param rank the calling thread's rank in it
    Team(std_backend::TeamCtx* ctx, int rank) noexcept : ctx_(ctx), rank_(rank) {}

    /// @brief The calling thread's rank in this team, in [0, size()).
    [[nodiscard]] int rank() const noexcept
    {
        return rank_;
    }

    /// @brief The number of threads in this team.
    [[nodiscard]] int size() const noexcept
    {
        return ctx_ != nullptr ? ctx_->size : 1;
    }

    /// @brief One contiguous block per thread (`schedule(static)`).
    template <typename Index, typename Body>
    void for_static(Index first, Index last, Body&& body, WaitMode w = WaitMode::wait);

    /// @brief Round-robin blocks of @p chunk iterations (`schedule(static, chunk)`).
    template <typename Index, typename Body>
    void for_static_chunks(Index first, Index last, int chunk, Body&& body, WaitMode w = WaitMode::wait);

    /// @brief for_static(): the same share. No OpenMP here to say `simd` to: the
    ///        loop is the auto-vectoriser's, as in the OpenMP backend without it.
    template <typename Index, typename Body>
    void for_static_simd(Index first, Index last, Body&& body, WaitMode w = WaitMode::wait)
    {
        for_static(first, last, std::forward<Body>(body), w);
    }

    /// @brief for_static_chunks(): the same blocks (see for_static_simd()).
    template <typename Index, typename Body>
    void for_static_chunks_simd(Index first, Index last, int chunk, Body&& body, WaitMode w = WaitMode::wait)
    {
        for_static_chunks(first, last, chunk, std::forward<Body>(body), w);
    }

    /// @brief Blocks of @p chunk iterations, first come first served (`schedule(dynamic, chunk)`).
    template <typename Index, typename Body>
    void for_dynamic(Index first, Index last, int chunk, Body&& body, WaitMode w = WaitMode::wait);

    /// @brief The run-time schedule (set_runtime_schedule; `schedule(runtime)`):
    ///        static (chunk 0: blocks), dynamic, guided as dynamic, auto as static.
    template <typename Index, typename Body>
    void for_runtime(Index first, Index last, Body&& body, WaitMode w = WaitMode::wait);

    /// @brief Wait until every thread of the team gets here.
    void barrier() noexcept
    {
        if (ctx_ != nullptr)
        {
            ctx_->bar.arrive_and_wait();
        }
    }

    /// @brief Run @p f on ONE thread of the team, whichever gets here first.
    template <typename F>
    void single(F&& f, WaitMode w = WaitMode::wait);

    /// @brief Run @p f on the team's rank 0 only, no barrier.
    template <typename F>
    void masked(F&& f)
    {
        if (rank_ == 0)
        {
            f();
        }
    }

  private:
    void end_construct(WaitMode w) noexcept
    {
        if (w == WaitMode::wait)
        {
            barrier();
        }
    }

    std_backend::TeamCtx* ctx_  = nullptr;
    int                   rank_ = 0;
};

/// @brief The team the calling thread runs in: inside a team, the innermost;
///        outside any, a team of one.
[[nodiscard]] inline Team current_team() noexcept
{
    const std_backend::Frame f = std_backend::tls().innermost();
    return Team{f.ctx, f.rank};
}

/// @brief Run @p body(team) on every thread of a new team.
template <typename Body>
void team(const Options& opt, Body&& body)
{
    std_backend::open_team<Team>(opt.use_team, detail::team_size(opt), body);
}

namespace detail
{
/// @brief One thread's share of a loop of @p opt's schedule, on @p t.
template <typename Index, typename Body>
void share(Team& t, Index first, Index last, const Options& opt, Body&& body)
{
    switch (opt.sched.kind)
    {
        case SchedKind::dynamic:
            t.for_dynamic(first, last, chunk_of(opt.sched), std::forward<Body>(body), nowait);
            return;
        case SchedKind::static_chunks:
            t.for_static_chunks(first, last, chunk_of(opt.sched), std::forward<Body>(body), nowait);
            return;
        case SchedKind::static_blocks:
            break;
    }
    t.for_static(first, last, std::forward<Body>(body), nowait);
}
} // namespace detail

/// @brief Run @p body(i) for every i in [@p first, @p last), on a team.
template <typename Index, typename Body>
void parallel_for(Index first, Index last, const Options& opt, Body&& body)
{
    static_assert(std::is_integral_v<Index>, "the loop index is an integer");
    team(opt, [&](Team& t) { detail::share(t, first, last, opt, body); });
}

/// @brief Run @p body(i, local) for every i in [@p first, @p last), on a team,
///        with local = @p make_local() taken once per thread.
template <typename Index, typename MakeLocal, typename Body>
void parallel_for_local(Index first, Index last, const Options& opt, MakeLocal&& make_local, Body&& body)
{
    static_assert(std::is_integral_v<Index>, "the loop index is an integer");
    team(opt,
         [&](Team& t)
         {
             decltype(auto) local = make_local();
             detail::share(t, first, last, opt, [&](Index i) { body(i, local); });
         });
}

/// @brief Sum @p N counters over [@p first, @p last), on a team (each thread
///        into its own copy, the copies summed in rank order).
template <typename T, std::size_t N, typename Index, typename Body>
[[nodiscard]] std::array<T, N> parallel_sum(Index first, Index last, const Options& opt, Body&& body)
{
    static_assert(std::is_integral_v<Index>, "the loop index is an integer");
    static_assert(std::is_arithmetic_v<T> && N > 0, "N arithmetic counters");
    std::vector<std::array<T, N>> parts(static_cast<std::size_t>(detail::team_size(opt)));
    team(opt,
         [&](Team& t)
         {
             T acc[N] = {}; // NOLINT(cppcoreguidelines-avoid-c-arrays): the body's `T (&)[N]`
             detail::share(t, first, last, opt, [&](Index i) { body(i, acc); });
             for (std::size_t k = 0; k < N; ++k)
             {
                 parts[static_cast<std::size_t>(t.rank())][k] = acc[k];
             }
         });
    std::array<T, N> sums{};
    for (const auto& p : parts)
    {
        for (std::size_t k = 0; k < N; ++k)
        {
            sums[k] += p[k];
        }
    }
    return sums;
}

/// @brief Run @p body(team, acc) on every thread of a new team, each thread
///        with its own acc[0..N) (zero to start); return their sums.
template <typename T, std::size_t N, typename Body>
[[nodiscard]] std::array<T, N> team_sum(const Options& opt, Body&& body)
{
    static_assert(std::is_arithmetic_v<T> && N > 0, "N arithmetic counters");
    std::vector<std::array<T, N>> parts(static_cast<std::size_t>(detail::team_size(opt)));
    team(opt,
         [&](Team& t)
         {
             T acc[N] = {}; // NOLINT(cppcoreguidelines-avoid-c-arrays): the body's `T (&)[N]`
             body(t, acc);
             for (std::size_t k = 0; k < N; ++k)
             {
                 parts[static_cast<std::size_t>(t.rank())][k] = acc[k];
             }
         });
    std::array<T, N> sums{};
    for (const auto& p : parts)
    {
        for (std::size_t k = 0; k < N; ++k)
        {
            sums[k] += p[k];
        }
    }
    return sums;
}

/// @brief Run @p body(team, flag) on every thread of a new team, each thread
///        with its own flag (false to start); return whether any set it.
template <typename Body>
[[nodiscard]] bool team_any(const Options& opt, Body&& body)
{
    std::vector<char> flags(static_cast<std::size_t>(detail::team_size(opt)), 0);
    team(opt,
         [&](Team& t)
         {
             bool flag = false;
             body(t, flag);
             flags[static_cast<std::size_t>(t.rank())] = flag ? 1 : 0;
         });
    for (const char f : flags)
    {
        if (f != 0)
        {
            return true;
        }
    }
    return false;
}

} // namespace fast_alloc::pmp

#include "fast_allocators/internal/pmp_std_inline.h" // the template bodies
