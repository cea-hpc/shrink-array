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

/**
 * @file bench_par.h
 * @brief The bench's parallel regions and worksharing loops: one source for
 *        the OpenMP and std backends (docs/threading_backends.md).
 *
 * A region is bench_par::region(body), body(Team& t) run by every thread of a
 * team of the library's backend (pmp::team): OpenMP, a `#pragma omp parallel`
 * whose body calls the lambda; std, a team of the backend's std::jthreads;
 * serial, the calling thread. The TBB bench (BENCH_PAR_TBB, a std build with
 * CHUNK_WITH_TBB): one TBB task per virtual rank, which never waits for
 * another -- the arrays written from TBB's threads, as a TBB program would.
 *
 * Inside it, the loops are written as a header macro followed by the body:
 *
 *     BENCH_FOR_STATIC_NOWAIT_SCALAR(t, i, deb, start0)
 *     {
 *         tab[i] = ...;
 *     }
 *
 * - OpenMP: the macro is the pragma the loop had, then
 *   `for (Int i = (b); i < (e); ++i)`. The worksharing the compiler sees is
 *   the same as before: orphaned `omp for` constructs binding to the region.
 * - std / serial: the macro is a loop over the calling thread's share,
 *   computed from t.rank() and t.size() exactly as libgomp shares the same
 *   schedule out. A rank gets the same iterations in every loop of the same
 *   bounds and schedule -- what the nowait reasoning of the phases relies on
 *   (bench_phase_compute.cpp) -- and the same blocks from one region to the
 *   next (the std backend's worker k is always rank k: first touch).
 *
 * The schedules: _STATIC (`schedule(static)`, one block per thread), _BLOCK
 * (`schedule(static, sched_blk)`, a local named sched_blk:
 * bench_sched_for(), bench_prelude.h), _ROUND_ROBIN (`schedule(static, 1)`).
 * Every loop is NOWAIT, every single() too: the regions have no barrier,
 * which is what lets the TBB bench run them as tasks (below).
 *
 * Reductions (_RED): the body adds to BENCH_RED(v), not to v: each thread adds
 * into a private partial, BENCH_RED(v), which goes into v atomically once the
 * thread's share is done -- whole after the next barrier or the end of the
 * region. Every build, OpenMP included: an orphaned `omp for reduction(+ : v)`
 * in the body of the region's lambda, v a capture by reference, is refused by
 * clang ("reduction variable must be shared"), accepted by gcc.
 *
 * The macros name the loop's variable, so the body is the same text in every
 * build; `continue` works, `break` is refused by OpenMP anyway.
 */

#pragma once

#include <algorithm>     // std::min (the block schedule's last block)
#include <atomic>        // std::atomic_ref (the reductions' partials, atomic_add)
#include <cstddef>       // std::size_t (the TBB thread count)
#include <cstdint>       // std::uint64_t (the TBB regions' construct numbers)
#include <cstdlib>       // std::abort (a construct the TBB regions do not have)
#include <cstdio>        // std::fputs (its message)
#include <memory>        // std::unique_ptr (the TBB regions' construct slots)
#include <mutex>         // std::mutex (critical(), the TBB regions' slots)
#include <unordered_map> // std::unordered_map (the TBB regions' construct slots)
#include <utility>       // std::forward

#include "bench_int.h"       // Int
#include "chunk/PMP_CHUNK.h" // shrink_array::pmp::team, pmp::Team, pmp::Options, pmp::backend

#if defined(BENCH_PAR_TBB) && BENCH_PAR_TBB
    #include <oneapi/tbb/blocked_range.h>  // oneapi::tbb::blocked_range (the virtual ranks)
    #include <oneapi/tbb/global_control.h> // oneapi::tbb::global_control (the thread count)
    #include <oneapi/tbb/parallel_for.h>   // oneapi::tbb::parallel_for, static_partitioner
    #include <oneapi/tbb/task_arena.h>     // oneapi::tbb::task_arena
#endif

namespace bench_par
{
/// @brief Team::single(f, nowait): no barrier after it.
inline constexpr auto nowait = shrink_array::pmp::nowait;

#if defined(BENCH_PAR_TBB) && BENCH_PAR_TBB

    #ifdef _OPENMP
        #error "BENCH_PAR_TBB: the TBB bench is not an OpenMP program (CHUNK_THREADS=std)"
    #endif
static_assert(shrink_array::pmp::backend == shrink_array::pmp::BackendKind::std_threads,
              "the TBB bench needs the std backend: TBB's threads write the arrays (CHUNK_THREADS=std)");

/// @brief false: the regions are TBB tasks, not a team of the library -- a
///        team-wide pass of the library (compress()) cannot run inside one.
inline constexpr bool team_regions = false;

namespace detail
{
/// @brief What the tasks of one region share: the state of its constructs
///        (a single's claim, a dynamic loop's next block), found by number.
class RegionCtx
{
  public:
    struct Slot
    {
        std::atomic<long long> next{0};
        bool                   claimed = false; ///< under the mutex
    };

    /// @brief The slot of construct @p seq (made by the first task to get there).
    Slot& slot(std::uint64_t seq)
    {
        const std::scoped_lock lock(mutex_);
        auto&                  s = slots_[seq];
        if (!s)
        {
            s = std::make_unique<Slot>();
        }
        return *s;
    }

    /// @brief Claim construct @p seq's single: true for the first task only.
    bool claim(std::uint64_t seq)
    {
        const std::scoped_lock lock(mutex_);
        auto&                  s = slots_[seq];
        if (!s)
        {
            s = std::make_unique<Slot>();
        }
        return !std::exchange(s->claimed, true);
    }

  private:
    std::mutex                                               mutex_;
    std::unordered_map<std::uint64_t, std::unique_ptr<Slot>> slots_;
};

[[noreturn]] inline void refuse(const char* what) noexcept
{
    std::fputs(what, stderr);
    std::abort();
}
} // namespace detail

/**
 * @brief The region as one of its tasks sees it: a virtual rank in [0, size()).
 * @details A region of the TBB bench is @c size() TBB tasks, one per rank, that
 *          never wait for each other: the bench's regions are barrier-free
 *          (every worksharing loop and single is nowait), so a task that runs
 *          after another on the same thread loses nothing but time -- and no
 *          task blocks one that has not started, which is how a barrier deadlocks
 *          TBB. Hence no barrier() here, and the constructs nowait only:
 *          single(), for_dynamic(), and the static loops, whose shares come
 *          from the virtual rank as pmp::Team's do from the thread's.
 */
class Team
{
  public:
    Team(detail::RegionCtx* ctx, int rank, int size) noexcept : ctx_(ctx), rank_(rank), size_(size) {}

    [[nodiscard]] int rank() const noexcept
    {
        return rank_;
    }
    [[nodiscard]] int size() const noexcept
    {
        return size_;
    }

    /// @brief Run @p f in ONE task of the region (nowait only).
    template <typename F>
    void single(F&& f, shrink_array::pmp::WaitMode w)
    {
        if (w != shrink_array::pmp::WaitMode::nowait)
        {
            detail::refuse("bench_par: a single with a barrier in a TBB region (only nowait)\n");
        }
        if (ctx_->claim(met_++))
        {
            f();
        }
    }

    /// @brief One contiguous block per rank, libgomp's split (`schedule(static)`; nowait only).
    template <typename Index, typename Body>
    void for_static(Index first, Index last, Body&& body, shrink_array::pmp::WaitMode w)
    {
        require_nowait(w, "bench_par: a static loop with a barrier in a TBB region (only nowait)\n");
        const long long n = static_cast<long long>(last) - static_cast<long long>(first);
        if (n <= 0)
        {
            return;
        }
        const long long q = n / size_;
        const long long m = n % size_;
        const long long b = rank_ < m ? rank_ * (q + 1) : (m * (q + 1)) + ((rank_ - m) * q);
        const long long e = b + q + (rank_ < m ? 1 : 0);
        for (long long k = b; k < e; ++k)
        {
            body(static_cast<Index>(first + static_cast<Index>(k)));
        }
    }

    /// @brief for_static(): the same share (no OpenMP in a TBB bench).
    template <typename Index, typename Body>
    void for_static_simd(Index first, Index last, Body&& body, shrink_array::pmp::WaitMode w)
    {
        for_static(first, last, std::forward<Body>(body), w);
    }

    /// @brief Blocks of @p chunk iterations, block k to rank k % size()
    ///        (`schedule(static, chunk)`; nowait only).
    template <typename Index, typename Body>
    void for_static_chunks(Index first, Index last, int chunk, Body&& body, shrink_array::pmp::WaitMode w)
    {
        require_nowait(w, "bench_par: a static loop with a barrier in a TBB region (only nowait)\n");
        const long long n  = static_cast<long long>(last) - static_cast<long long>(first);
        const long long c  = chunk > 0 ? chunk : 1;
        const long long nt = size_;
        for (long long b = rank_ * c; b < n; b += nt * c)
        {
            const long long e = b + c < n ? b + c : n;
            for (long long k = b; k < e; ++k)
            {
                body(static_cast<Index>(first + static_cast<Index>(k)));
            }
        }
    }

    /// @brief for_static_chunks(): the same blocks (no OpenMP in a TBB bench).
    template <typename Index, typename Body>
    void for_static_chunks_simd(Index first, Index last, int chunk, Body&& body, shrink_array::pmp::WaitMode w)
    {
        for_static_chunks(first, last, chunk, std::forward<Body>(body), w);
    }

    /// @brief Blocks of @p chunk iterations, first come first served (nowait only).
    template <typename Index, typename Body>
    void for_dynamic(Index first, Index last, int chunk, Body&& body, shrink_array::pmp::WaitMode w)
    {
        if (w != shrink_array::pmp::WaitMode::nowait)
        {
            detail::refuse("bench_par: a dynamic loop with a barrier in a TBB region (only nowait)\n");
        }
        const long long n    = static_cast<long long>(last) - static_cast<long long>(first);
        const long long c    = chunk > 0 ? chunk : 1;
        auto&           slot = ctx_->slot(met_++);
        for (long long b = slot.next.fetch_add(c, std::memory_order_relaxed); b < n;
             b           = slot.next.fetch_add(c, std::memory_order_relaxed))
        {
            const long long e = b + c < n ? b + c : n;
            for (long long k = b; k < e; ++k)
            {
                body(static_cast<Index>(first + static_cast<Index>(k)));
            }
        }
    }

  private:
    static void require_nowait(shrink_array::pmp::WaitMode w, const char* what) noexcept
    {
        if (w != shrink_array::pmp::WaitMode::nowait)
        {
            detail::refuse(what);
        }
    }

    detail::RegionCtx* ctx_;
    int                rank_;
    int                size_;
    std::uint64_t      met_ = 0; ///< constructs this task has met (the slots' numbers)
};

namespace detail
{
/// @brief The bench's TBB threads: CHUNK_PMP_MAX_THREADS of them (the
///        backend's team size: CHUNK_NUM_THREADS, else OMP_NUM_THREADS, as the
///        scripts set it), the calling thread among them. Immortal.
inline oneapi::tbb::task_arena& arena()
{
    static const int                    n = CHUNK_PMP_MAX_THREADS;
    static oneapi::tbb::global_control* gc =
        new oneapi::tbb::global_control(oneapi::tbb::global_control::max_allowed_parallelism, // NOLINT: immortal
                                        static_cast<std::size_t>(n));
    static oneapi::tbb::task_arena* const a = new oneapi::tbb::task_arena(n); // NOLINT: immortal
    (void)gc;
    return *a;
}
} // namespace detail

/// @brief Run @p body(Team&) once per virtual rank of [0, @p num_threads), as TBB tasks.
template <typename Body>
void region(int num_threads, Body&& body)
{
    const int         n = num_threads > 0 ? num_threads : CHUNK_PMP_MAX_THREADS;
    detail::RegionCtx ctx;
    detail::arena().execute(
        [&]
        {
            oneapi::tbb::parallel_for(
                oneapi::tbb::blocked_range<int>(0, n, 1),
                [&](const oneapi::tbb::blocked_range<int>& r)
                {
                    for (int k = r.begin(); k != r.end(); ++k)
                    {
                        Team t(&ctx, k, n);
                        body(t);
                    }
                },
                oneapi::tbb::static_partitioner{});
        });
}

/// @brief region() with CHUNK_PMP_MAX_THREADS virtual ranks.
template <typename Body>
void region(Body&& body)
{
    region(0, std::forward<Body>(body));
}

#else

using Team = shrink_array::pmp::Team;

/// @brief true: a region is a team of the library's backend, inside which the
///        library's team-wide passes (compress()) share the work.
inline constexpr bool team_regions = true;

/// @brief Run @p body(Team&) on every thread of a new team of the backend's default size.
template <typename Body>
void region(Body&& body)
{
    shrink_array::pmp::team(shrink_array::pmp::Options{}, std::forward<Body>(body));
}

/// @brief Run @p body(Team&) on every thread of a new team of @p num_threads threads.
template <typename Body>
void region(int num_threads, Body&& body)
{
    shrink_array::pmp::team(shrink_array::pmp::Options{.num_threads = num_threads}, std::forward<Body>(body));
}

#endif

/// @brief The calling thread's share [lo, hi) of [@p first, @p last) under
///        `schedule(static)`: libgomp's split, the first (e - b) % size ranks
///        one iteration more.
[[nodiscard]] inline Int static_lo(const Team& team, Int first, Int last) noexcept
{
    const Int count = last - first;
    if (count <= 0)
    {
        return first;
    }
    const Int nt   = team.size();
    const Int rank = team.rank();
    const Int quot = count / nt;
    const Int rem  = count % nt;
    return first + (rank < rem ? rank * (quot + 1) : (rem * (quot + 1)) + ((rank - rem) * quot));
}

/// @brief One past the end of static_lo()'s share.
[[nodiscard]] inline Int static_hi(const Team& team, Int first, Int last) noexcept
{
    const Int count = last - first;
    if (count <= 0)
    {
        return first;
    }
    const Int nt   = team.size();
    const Int rank = team.rank();
    return static_lo(team, first, last) + (count / nt) + (rank < count % nt ? 1 : 0);
}

/// @brief v += x, atomically, whatever the backend: std::atomic_ref is atomic
///        against OpenMP's threads as against the std backend's (it compiles
///        to the same instruction as `omp atomic`).
template <typename T>
inline void atomic_add(T& target, T delta) noexcept
{
    std::atomic_ref<T>(target).fetch_add(delta, std::memory_order_relaxed);
}

/// @brief A block size for Team::for_static_chunks (an int): the bench's
///        sched_blk (bench_sched_for(), an Int, a power of two or a slice).
[[nodiscard]] inline int block(Int sched_blk) noexcept
{
    return static_cast<int>(sched_blk);
}

// The bodies given to the helpers below, and the bench's loop bodies, are
// lambdas marked __attribute__((always_inline)) (after the parameters; the
// [[gnu::always_inline]] spelling there applies to the type and is ignored):
// gcc -O3 otherwise leaves a big body of a big region out of line, a call per
// iteration where the BENCH_FOR_* macros had the body in the loop.

/**
 * @brief A reduction inside a region: @p body(i, part) adds into the calling
 *        thread's partial for its share of [@p first, @p last) under `schedule(static)`,
 *        then the partial goes into @p sum atomically. No barrier: @p sum is whole
 *        after the next barrier or the end of the region.
 * @details Not a `reduction(+ : v)` clause (an orphaned `omp for reduction` on
 *          a lambda's capture is refused by clang), not pmp::team_sum (a region
 *          of its own, where the bench's phases are one barrier-free region).
 */
template <typename T, typename Body>
void for_static_sum(Team& team, Int first, Int last, T& sum, Body&& body)
{
    T part{};
    team.for_static<Int>(
        first, last, [&](const Int i) __attribute__((always_inline)) { body(i, part); }, shrink_array::pmp::nowait);
    atomic_add(sum, part);
}

/// @brief for_static_sum() over round-robin blocks of @p chunk iterations
///        (`schedule(static, chunk)`).
template <typename T, typename Body>
void for_static_chunks_sum(Team& team, Int first, Int last, int chunk, T& sum, Body&& body)
{
    T part{};
    team.for_static_chunks<Int>(
        first, last, chunk, [&](const Int i) __attribute__((always_inline)) { body(i, part); },
        shrink_array::pmp::nowait);
    atomic_add(sum, part);
}

/// @brief The lock of critical(): one for the whole bench, as an unnamed
///        `omp critical`.
[[nodiscard]] inline std::mutex& critical_mutex() noexcept
{
    static std::mutex mtx;
    return mtx;
}

/// @brief Run @p func under the bench's one lock (an unnamed `omp critical`).
template <typename F>
void critical(F&& func)
{
    const std::scoped_lock lock(critical_mutex());
    func();
}

/// @brief A thread's partial of a reduction (every build, see the file
///        comment): added into the shared variable when the loop's share is done.
template <typename T>
class Partial
{
  public:
    explicit Partial(T& shared) noexcept : shared_(shared) {}
    Partial(const Partial&)            = delete;
    Partial& operator=(const Partial&) = delete;
    Partial(Partial&&)                 = delete;
    Partial& operator=(Partial&&)      = delete;
    ~Partial()
    {
        atomic_add(shared_, value);
    }

    /// @brief True once: the loop runs its share, then the partial goes in.
    [[nodiscard]] bool once() noexcept
    {
        return !done_ && (done_ = true);
    }

    T value{}; ///< the thread's partial

  private:
    T&   shared_;
    bool done_ = false;
};
} // namespace bench_par

// clang-format off
#ifdef _OPENMP

    #define BENCH_FOR_STATIC_NOWAIT(t, i, b, e) BENCH_OMP_FOR_STATIC_NOWAIT for (Int i = (b); i < (e); ++i)
    #define BENCH_FOR_STATIC_NOWAIT_SCALAR(t, i, b, e) BENCH_OMP_FOR_STATIC_NOWAIT_SCALAR for (Int i = (b); i < (e); ++i)
    // sched_blk copied into a local first: clang 19 and 20 crash on a chunk size that is a
    // capture of the region's lambda (segfault in EmitOMPWorksharingLoop).
    #define BENCH_FOR_BLOCK_NOWAIT_SCALAR(t, i, b, e) \
        if (const Int bench_blk_ = sched_blk; true) _Pragma("omp for schedule(static, bench_blk_) nowait") for (Int i = (b); i < (e); ++i)
    #define BENCH_FOR_ROUND_ROBIN_NOWAIT(t, i, b, e) _Pragma("omp for schedule(static, 1) nowait") for (Int i = (b); i < (e); ++i)
    // A per-thread partial, not a reduction clause: see the file comment.
    #define BENCH_FOR_STATIC_NOWAIT_RED_SCALAR(t, i, b, e, v) \
        for (::bench_par::Partial<decltype(v)> v##_part_(v); v##_part_.once();) BENCH_FOR_STATIC_NOWAIT_SCALAR(t, i, b, e)
    #define BENCH_FOR_BLOCK_NOWAIT_RED_SCALAR(t, i, b, e, v) \
        for (::bench_par::Partial<decltype(v)> v##_part_(v); v##_part_.once();) BENCH_FOR_BLOCK_NOWAIT_SCALAR(t, i, b, e)
    #define BENCH_RED(v) v##_part_.value

#else

    // The calling thread's share, as libgomp deals the same schedule out.
    #define BENCH_FOR_STATIC_NOWAIT(t, i, b, e) \
        for (Int i = ::bench_par::static_lo((t), (b), (e)), i##_end_ = ::bench_par::static_hi((t), (b), (e)); i < i##_end_; ++i)
    #define BENCH_FOR_STATIC_NOWAIT_SCALAR(t, i, b, e) BENCH_FOR_STATIC_NOWAIT(t, i, b, e)
    // schedule(static, sched_blk): block k of sched_blk iterations from b goes to rank k % size.
    #define BENCH_FOR_BLOCK_NOWAIT_SCALAR(t, i, b, e)                                                               \
        for (Int i##_blk_ = (b) + (static_cast<Int>((t).rank()) * sched_blk); i##_blk_ < (e);                     \
             i##_blk_ += static_cast<Int>((t).size()) * sched_blk)                                                \
            for (Int i = i##_blk_, i##_end_ = std::min<Int>(i##_blk_ + sched_blk, (e)); i < i##_end_; ++i)
    #define BENCH_FOR_ROUND_ROBIN_NOWAIT(t, i, b, e) \
        for (Int i = (b) + static_cast<Int>((t).rank()); i < (e); i += static_cast<Int>((t).size()))
    #define BENCH_FOR_STATIC_NOWAIT_RED_SCALAR(t, i, b, e, v) \
        for (::bench_par::Partial<decltype(v)> v##_part_(v); v##_part_.once();) BENCH_FOR_STATIC_NOWAIT(t, i, b, e)
    #define BENCH_FOR_BLOCK_NOWAIT_RED_SCALAR(t, i, b, e, v) \
        for (::bench_par::Partial<decltype(v)> v##_part_(v); v##_part_.once();) BENCH_FOR_BLOCK_NOWAIT_SCALAR(t, i, b, e)
    #define BENCH_RED(v) v##_part_.value

#endif
// clang-format on
