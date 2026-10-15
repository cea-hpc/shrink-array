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

// The pool backend of the library's threading (docs/threading_backends.md,
// step 7): the loops and teams of pmp_loops.h on std::jthread workers of the
// library's own. Included by pmp_loops.h only, after its shared types
// (options, schedule, wait_mode), in a build with FAST_ALLOC_PMP_POOL=1.
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
    #error "pmp_pool.h is included by pmp_loops.h, after the types it shares with the other backends"
#endif

#include <barrier>       // std::barrier (Team::barrier)
#include <cstdint>       // std::uint64_t (the pool's job word)
#include <exception>     // std::terminate (a body that throws)
#include <memory>        // std::unique_ptr (the construct slots)
#include <mutex>         // std::mutex, std::scoped_lock (the slots; one team at a time)
#include <thread>        // std::jthread (the workers)
#include <unordered_map> // std::unordered_map (the construct slots)
#include <vector>        // std::vector (the workers, the per-rank counts and results)

#include "fast_allocators/internal/pmp_pool_state.h" // pool::tls, FrameGuard, settings, nthreads_var

namespace fast_alloc::pmp
{
namespace pool
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
    Slot& enter(int rank, std::uint64_t& seq)
    {
        seq = met[static_cast<std::size_t>(rank)]++;
        const std::scoped_lock lock(mutex);
        auto&                  s = slots[seq];
        if (!s)
        {
            s       = std::make_unique<Slot>();
            s->left = size;
        }
        return *s;
    }

    /// @brief Leave construct @p seq; the last thread out frees its slot.
    void leave(std::uint64_t seq)
    {
        const std::scoped_lock lock(mutex);
        const auto             it = slots.find(seq);
        if (--it->second->left == 0)
        {
            slots.erase(it);
        }
    }
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
    void run(int n, Job& job)
    {
        const std::scoped_lock lock(team_mutex_);
        while (static_cast<int>(workers_.size()) < n - 1)
        {
            const int           r     = static_cast<int>(workers_.size()) + 1;
            const std::uint64_t start = word_.load(std::memory_order_acquire);
            workers_.emplace_back([this, r, start] { worker_loop(r, start); });
        }
        fn_  = [](void* j, int rank) { (*static_cast<Job*>(j))(rank); };
        arg_ = &job;
        pending_.store(n - 1, std::memory_order_relaxed);
        const std::uint64_t gen = (word_.load(std::memory_order_relaxed) >> 16) + 1;
        word_.store((gen << 16) | static_cast<std::uint64_t>(n), std::memory_order_release);
        word_.notify_all();
        call(0);
        for (int p = pending_.load(std::memory_order_acquire); p != 0; p = pending_.load(std::memory_order_acquire))
        {
            pending_.wait(p, std::memory_order_acquire);
        }
    }

  private:
    Pool() = default;

    void call(int rank) noexcept
    {
        try
        {
            fn_(arg_, rank);
        }
        catch (...)
        {
            std::terminate(); // as OpenMP: an exception must not leave a region
        }
    }

    void worker_loop(int rank, std::uint64_t seen) noexcept
    {
        for (;;)
        {
            word_.wait(seen, std::memory_order_acquire);
            seen = word_.load(std::memory_order_acquire);
            if (rank < static_cast<int>(seen & 0xFFFFU))
            {
                call(rank);
                if (pending_.fetch_sub(1, std::memory_order_acq_rel) == 1)
                {
                    pending_.notify_one();
                }
            }
        }
    }

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
 * @details A team of one when @p on is false, @p nt is 1, or the calling
 *          thread is already in max_active_levels active teams; it still
 *          counts as a level, as OpenMP's serialised regions do.
 */
template <typename TeamT, typename Body>
void open_team(bool on, int nt, Body& body)
{
    const auto& s = tls();
    if (!on || nt <= 1 || s.active_level >= settings().max_active_levels.load(std::memory_order_relaxed))
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
} // namespace pool

/**
 * @brief The team a thread runs in, as seen from that thread (see the OpenMP
 *        backend's in pmp_loops.h). Pool backend: the team's shared state and
 *        the thread's rank in it; no state for a team of one.
 */
class Team
{
  public:
    Team() = default;
    /// @param ctx  the team's shared state (nullptr: a team of one)
    /// @param rank the calling thread's rank in it
    Team(pool::TeamCtx* ctx, int rank) noexcept : ctx_(ctx), rank_(rank) {}

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
    void for_static(Index first, Index last, Body&& body, wait_mode w = wait_mode::wait)
    {
        static_assert(std::is_integral_v<Index>, "the loop index is an integer");
        long long b = 0;
        long long e = static_cast<long long>(last) - static_cast<long long>(first);
        if (ctx_ != nullptr && e > 0)
        {
            pool::static_block(e, ctx_->size, rank_, b, e);
        }
        for (long long k = b; k < e; ++k)
        {
            body(static_cast<Index>(first + static_cast<Index>(k)));
        }
        end_construct(w);
    }

    /// @brief Round-robin blocks of @p chunk iterations (`schedule(static, chunk)`).
    template <typename Index, typename Body>
    void for_static_chunks(Index first, Index last, int chunk, Body&& body, wait_mode w = wait_mode::wait)
    {
        static_assert(std::is_integral_v<Index>, "the loop index is an integer");
        const long long n  = static_cast<long long>(last) - static_cast<long long>(first);
        const long long c  = chunk > 0 ? chunk : 1;
        const long long nt = size();
        for (long long b = rank_ * c; b < n; b += nt * c)
        {
            const long long e = b + c < n ? b + c : n;
            for (long long k = b; k < e; ++k)
            {
                body(static_cast<Index>(first + static_cast<Index>(k)));
            }
        }
        end_construct(w);
    }

    /// @brief Blocks of @p chunk iterations, first come first served (`schedule(dynamic, chunk)`).
    template <typename Index, typename Body>
    void for_dynamic(Index first, Index last, int chunk, Body&& body, wait_mode w = wait_mode::wait)
    {
        static_assert(std::is_integral_v<Index>, "the loop index is an integer");
        const long long n = static_cast<long long>(last) - static_cast<long long>(first);
        const long long c = chunk > 0 ? chunk : 1;
        if (ctx_ == nullptr)
        {
            for (long long k = 0; k < n; ++k)
            {
                body(static_cast<Index>(first + static_cast<Index>(k)));
            }
            return;
        }
        std::uint64_t seq  = 0;
        pool::Slot&   slot = ctx_->enter(rank_, seq);
        for (long long b = slot.next.fetch_add(c, std::memory_order_relaxed); b < n;
             b           = slot.next.fetch_add(c, std::memory_order_relaxed))
        {
            const long long e = b + c < n ? b + c : n;
            for (long long k = b; k < e; ++k)
            {
                body(static_cast<Index>(first + static_cast<Index>(k)));
            }
        }
        ctx_->leave(seq);
        end_construct(w);
    }

    /// @brief The run-time schedule (set_runtime_schedule; `schedule(runtime)`):
    ///        static (chunk 0: blocks), dynamic, guided as dynamic, auto as static.
    template <typename Index, typename Body>
    void for_runtime(Index first, Index last, Body&& body, wait_mode w = wait_mode::wait)
    {
        const auto& s     = pool::settings();
        const int   kind  = s.sched_kind.load(std::memory_order_relaxed) & 0x7FFFFFFF; // monotonic bit off
        const int   chunk = s.sched_chunk.load(std::memory_order_relaxed);
        if (kind == 2 || kind == 3)
        {
            for_dynamic(first, last, chunk, std::forward<Body>(body), w);
        }
        else if (kind == 1 && chunk > 0)
        {
            for_static_chunks(first, last, chunk, std::forward<Body>(body), w);
        }
        else
        {
            for_static(first, last, std::forward<Body>(body), w);
        }
    }

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
    void single(F&& f, wait_mode w = wait_mode::wait)
    {
        if (ctx_ == nullptr)
        {
            f();
            return;
        }
        std::uint64_t seq  = 0;
        pool::Slot&   slot = ctx_->enter(rank_, seq);
        if (!slot.claimed.exchange(true, std::memory_order_acq_rel))
        {
            f();
        }
        ctx_->leave(seq);
        end_construct(w);
    }

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
    void end_construct(wait_mode w) noexcept
    {
        if (w == wait_mode::wait)
        {
            barrier();
        }
    }

    pool::TeamCtx* ctx_  = nullptr;
    int            rank_ = 0;
};

/// @brief The team the calling thread runs in: inside a team, the innermost;
///        outside any, a team of one.
[[nodiscard]] inline Team current_team() noexcept
{
    const pool::Frame f = pool::tls().innermost();
    return Team{f.ctx, f.rank};
}

/// @brief Run @p body(team) on every thread of a new team.
template <typename Body>
void team(const options& opt, Body&& body)
{
    pool::open_team<Team>(opt.use_team, detail::team_size(opt), body);
}

namespace detail
{
/// @brief One thread's share of a loop of @p opt's schedule, on @p t.
template <typename Index, typename Body>
void share(Team& t, Index first, Index last, const options& opt, Body&& body)
{
    switch (opt.sched.kind)
    {
        case sched_kind::dynamic:
            t.for_dynamic(first, last, chunk_of(opt.sched), std::forward<Body>(body), nowait);
            return;
        case sched_kind::static_chunks:
            t.for_static_chunks(first, last, chunk_of(opt.sched), std::forward<Body>(body), nowait);
            return;
        case sched_kind::static_blocks:
            break;
    }
    t.for_static(first, last, std::forward<Body>(body), nowait);
}
} // namespace detail

/// @brief Run @p body(i) for every i in [@p first, @p last), on a team.
template <typename Index, typename Body>
void parallel_for(Index first, Index last, const options& opt, Body&& body)
{
    static_assert(std::is_integral_v<Index>, "the loop index is an integer");
    team(opt, [&](Team& t) { detail::share(t, first, last, opt, body); });
}

/// @brief Run @p body(i, local) for every i in [@p first, @p last), on a team,
///        with local = @p make_local() taken once per thread.
template <typename Index, typename MakeLocal, typename Body>
void parallel_for_local(Index first, Index last, const options& opt, MakeLocal&& make_local, Body&& body)
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
[[nodiscard]] std::array<T, N> parallel_sum(Index first, Index last, const options& opt, Body&& body)
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
[[nodiscard]] std::array<T, N> team_sum(const options& opt, Body&& body)
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
[[nodiscard]] bool team_any(const options& opt, Body&& body)
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
