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

// The threading backend of the two libraries (docs/threading_backends.md):
// the call sites say WHAT runs in parallel, this file says HOW.
//
// Use B, one-shot data-parallel loops over an index range, run at a serial
// point on a team of their own:
//
//   parallel_for(first, last, opt, body)            body(i)
//   parallel_for_local(first, last, opt, make, body) body(i, local), local = make() once per thread
//   parallel_sum<T, N>(first, last, opt, body)      body(i, acc), acc[0..N) summed over the threads
//
// Use C, team regions with worksharing and synchronisation inside:
//
//   team(opt, body)                                 body(Team&), run by every thread of a new team
//   team_sum<T, N>(opt, body)                       body(Team&, acc), acc[0..N) summed over the team
//   team_any(opt, body)                             body(Team&, flag): did any thread set its flag?
//   current_team()                                  the team the calling thread runs in (a team of
//                                                   one outside any region)
//   Team: rank(), size(), for_static(), for_static_chunks(), for_dynamic(), for_runtime(),
//         barrier(), single(), masked(); for_static_simd() and for_static_chunks_simd(),
//         the same shares as `omp for simd` (std and serial: the auto-vectoriser's loop)
//
// Every thread of a team must meet the same Team calls in the same order (the
// OpenMP rule for worksharing and barriers).
//
// OpenMP backend: each loop is ONE `#pragma omp
// parallel for` (parallel_for_local: a `parallel` region holding an `omp
// for`), each team a `#pragma omp parallel`, and the Team methods the orphaned
// constructs (`omp for`, `barrier`, `single`, `masked`), so they bind to the
// innermost enclosing region -- the library's own team, or the caller's. The
// clauses are the ones the call sites had before, so the OMPT callbacks the
// chunk library hangs its phase ends on (parallel_end, the barriers) fire
// exactly as they did.
//
// Std backend (FAST_ALLOC_PMP_STD=1, CMake CHUNK_THREADS=std): the same
// API on std::jthread workers of the library's own, with OpenMP's model
// (pmp_std.h).
//
// Serial backend: everything runs on the calling thread, a team of one.
//
// Standard library and <omp.h> only: chunk's std-alloc build installs this
// file with svalloc.h and singleton.h (chunk/CMakeLists.txt).

#include <array>       // std::array (parallel_sum's result)
#include <cstddef>     // std::size_t
#include <type_traits> // std::is_integral_v
#include <utility>     // std::forward (the std backend's loops)

#ifdef _OPENMP
    #include <omp.h> // omp_get_max_threads (the default team size), omp_get_thread_num, omp_get_num_threads
#endif

#include "fast_allocators/internal/pmp_runtime.h" // FAST_ALLOC_PMP_BACKEND_STD, max_threads (the std backend's team size)

namespace fast_alloc::pmp
{

/// @brief How the iterations are shared out over the team.
enum class SchedKind : int
{
    static_blocks, ///< one contiguous block per thread (`schedule(static)`)
    static_chunks, ///< round-robin blocks of `chunk` iterations (`schedule(static, chunk)`)
    dynamic,       ///< blocks of `chunk` iterations, first come first served (`schedule(dynamic, chunk)`)
};

/// @brief A loop schedule: its kind and, for the chunked ones, the block size.
struct Schedule
{
    SchedKind kind  = SchedKind::static_blocks; ///< How the iterations are shared out.
    int       chunk = 0;                        ///< Block size (chunked kinds; below 1 means 1).

    /// @brief One contiguous block per thread: first touch where the static
    ///        per-chunk loops will use the data. The default.
    [[nodiscard]] static constexpr Schedule static_blocks() noexcept
    {
        return {};
    }
    /// @brief Round-robin blocks of @p chunk iterations.
    [[nodiscard]] static constexpr Schedule static_chunks(int chunk) noexcept
    {
        return {.kind = SchedKind::static_chunks, .chunk = chunk};
    }
    /// @brief Blocks of @p chunk iterations, taken by whichever thread is free.
    [[nodiscard]] static constexpr Schedule dynamic(int chunk) noexcept
    {
        return {.kind = SchedKind::dynamic, .chunk = chunk};
    }
};

/// @brief Whether a worksharing construct ends with a barrier.
enum class WaitMode : bool
{
    wait,   ///< every thread waits at the end (the default)
    nowait, ///< no barrier at the end: a thread goes on as soon as its share is done
};

/// @brief Shorthand for WaitMode::nowait.
inline constexpr WaitMode nowait = WaitMode::nowait;

/// @brief How a loop runs.
struct Options
{
    /// false: no team. OpenMP: the region still opens, with one thread (the
    /// `if` clause), so its OMPT callbacks fire as for any region.
    bool     use_team    = true;
    int      num_threads = 0;  ///< Team size; 0 or less: the backend's default.
    Schedule sched       = {}; ///< How the iterations are shared out.
};

namespace detail
{
/// @brief The team size to ask for: @p opt's, or the backend's default.
[[nodiscard]] inline int team_size([[maybe_unused]] const Options& opt) noexcept
{
#ifdef _OPENMP
    return opt.num_threads > 0 ? opt.num_threads : omp_get_max_threads();
#elif FAST_ALLOC_PMP_BACKEND_STD
    return opt.num_threads > 0 ? opt.num_threads : max_threads();
#else
    return 1;
#endif
}

/// @brief The block size of a chunked schedule (at least 1).
[[nodiscard]] constexpr int chunk_of(const Schedule& sched) noexcept
{
    return sched.chunk > 0 ? sched.chunk : 1;
}
} // namespace detail
} // namespace fast_alloc::pmp

// clang-format off
// (the OpenMP / serial bodies below keep their directive indentation)
#if FAST_ALLOC_PMP_BACKEND_STD
// The std backend's loops, Team and teams.
#define FAST_ALLOC_PMP_LOOPS_TYPES 1
#include "fast_allocators/internal/pmp_std.h" // parallel_for ... team_any, Team, current_team on the std backend
#else
// clang-format on

namespace fast_alloc::pmp
{

/**
 * @brief Run @p body(i) for every i in [@p first, @p last), on a team.
 * @param first first index
 * @param last  one past the last index
 * @param opt   team, size and schedule
 * @param body  callable `void(Index)`; iterations must be independent
 */
template <typename Index, typename Body>
void parallel_for(Index first, Index last, [[maybe_unused]] const Options& opt, Body&& body)
{
    static_assert(std::is_integral_v<Index>, "the loop index is an integer");
    #ifdef _OPENMP
    const bool on    = opt.use_team;
    const int  nt    = detail::team_size(opt);
    const int  chunk = detail::chunk_of(opt.sched);
    switch (opt.sched.kind)
    {
        case SchedKind::dynamic:
        #pragma omp parallel for schedule(dynamic, chunk) if (on) num_threads(nt)
            for (Index i = first; i < last; ++i)
            {
                body(i);
            }
            return;
        case SchedKind::static_chunks:
        #pragma omp parallel for schedule(static, chunk) if (on) num_threads(nt)
            for (Index i = first; i < last; ++i)
            {
                body(i);
            }
            return;
        case SchedKind::static_blocks:
            break;
    }
        #pragma omp parallel for schedule(static) if (on) num_threads(nt)
    for (Index i = first; i < last; ++i)
    {
        body(i);
    }
    #else
    for (Index i = first; i < last; ++i)
    {
        body(i);
    }
    #endif
}

/**
 * @brief Run @p body(i, local) for every i in [@p first, @p last), on a team,
 *        with local = @p make_local() taken once per thread.
 * @details For loops whose threads each need their own state (the reuse list
 *          of the rank that releases a buffer, say).
 * @param first      first index
 * @param last       one past the last index
 * @param opt        team, size and schedule
 * @param make_local callable `Local&()` (or returning a value), called once per thread
 * @param body       callable `void(Index, Local&)`; iterations must be independent
 */
template <typename Index, typename MakeLocal, typename Body>
void parallel_for_local(Index                           first,
                        Index                           last,
                        [[maybe_unused]] const Options& opt,
                        MakeLocal&&                     make_local,
                        Body&&                          body)
{
    static_assert(std::is_integral_v<Index>, "the loop index is an integer");
    #ifdef _OPENMP
    const bool      on    = opt.use_team;
    const int       nt    = detail::team_size(opt);
    const int       chunk = detail::chunk_of(opt.sched);
    const SchedKind kind  = opt.sched.kind;
        #pragma omp parallel if (on) num_threads(nt)
    {
        decltype(auto) local = make_local();
        // Every thread takes the same branch: each meets the same `omp for`.
        switch (kind)
        {
            case SchedKind::dynamic:
        #pragma omp for schedule(dynamic, chunk)
                for (Index i = first; i < last; ++i)
                {
                    body(i, local);
                }
                break;
            case SchedKind::static_chunks:
        #pragma omp for schedule(static, chunk)
                for (Index i = first; i < last; ++i)
                {
                    body(i, local);
                }
                break;
            case SchedKind::static_blocks:
        #pragma omp for schedule(static)
                for (Index i = first; i < last; ++i)
                {
                    body(i, local);
                }
                break;
        }
    }
    #else
    decltype(auto) local = make_local();
    for (Index i = first; i < last; ++i)
    {
        body(i, local);
    }
    #endif
}

/**
 * @brief Sum @p N counters over [@p first, @p last), on a team: @p body(i, acc)
 *        adds iteration i's share to acc[0..N), each thread into its own copy
 *        (zero to start), and the copies are summed.
 * @details A `reduction(+ : ...)` loop. Write the body with `auto& acc` and
 *          index it: `[&](int j, auto& acc) { if (...) { ++acc[0]; } }`.
 * @tparam T arithmetic type of the counters
 * @tparam N number of counters
 * @param first first index
 * @param last  one past the last index
 * @param opt   team, size and schedule
 * @param body  callable `void(Index, T (&)[N])`; iterations independent but for the sums
 * @return the N sums
 */
template <typename T, std::size_t N, typename Index, typename Body>
[[nodiscard]] std::array<T, N> parallel_sum(Index first, Index last, [[maybe_unused]] const Options& opt, Body&& body)
{
    static_assert(std::is_integral_v<Index>, "the loop index is an integer");
    static_assert(std::is_arithmetic_v<T> && N > 0, "N arithmetic counters");
    T acc[N] = {}; // NOLINT(cppcoreguidelines-avoid-c-arrays): an OpenMP array-section reduction
    #ifdef _OPENMP
    const bool on    = opt.use_team;
    const int  nt    = detail::team_size(opt);
    const int  chunk = detail::chunk_of(opt.sched);
    switch (opt.sched.kind)
    {
        case SchedKind::dynamic:
        #pragma omp parallel for schedule(dynamic, chunk) reduction(+ : acc[ : N]) if (on) num_threads(nt)
            for (Index i = first; i < last; ++i)
            {
                body(i, acc);
            }
            break;
        case SchedKind::static_chunks:
        #pragma omp parallel for schedule(static, chunk) reduction(+ : acc[ : N]) if (on) num_threads(nt)
            for (Index i = first; i < last; ++i)
            {
                body(i, acc);
            }
            break;
        case SchedKind::static_blocks:
        #pragma omp parallel for schedule(static) reduction(+ : acc[ : N]) if (on) num_threads(nt)
            for (Index i = first; i < last; ++i)
            {
                body(i, acc);
            }
            break;
    }
    #else
    for (Index i = first; i < last; ++i)
    {
        body(i, acc);
    }
    #endif
    std::array<T, N> sums{};
    for (std::size_t k = 0; k < N; ++k)
    {
        sums[k] = acc[k];
    }
    return sums;
}

/**
 * @brief The team a thread runs in, as seen from that thread: its rank, the
 *        team size, and the worksharing and synchronisation shared with the
 *        other threads.
 * @details Given to the body of team() / team_sum(), or got from
 *          current_team() by code that may run inside the library's team or
 *          inside the caller's region. OpenMP backend: no state, every method
 *          is an orphaned construct binding to the innermost enclosing region.
 */
class Team
{
  public:
    /// @brief The calling thread's rank in this team, in [0, size()).
    [[nodiscard]] int rank() const noexcept
    {
    #ifdef _OPENMP
        return omp_get_thread_num();
    #else
        return 0;
    #endif
    }

    /// @brief The number of threads in this team.
    [[nodiscard]] int size() const noexcept
    {
    #ifdef _OPENMP
        return omp_get_num_threads();
    #else
        return 1;
    #endif
    }

    /**
     * @brief Share [@p first, @p last) out over the team, one contiguous
     *        block per thread (`omp for schedule(static)`).
     * @param first first index
     * @param last  one past the last index
     * @param body  called as body(i), once per index
     * @param wait_mode WaitMode::nowait: no barrier at the end.
     */
    template <typename Index, typename Body>
    void for_static(Index first, Index last, Body&& body, [[maybe_unused]] WaitMode wait_mode = WaitMode::wait);

    /**
     * @brief for_static() as `omp for simd schedule(static)`: the same share,
     *        the loop vectorised as `simd` asks. Serial build: for_static().
     * @details For loops that do vectorise (plain stores into a std::vector,
     *          a compute kernel). Not for loops through the arrays' proxies:
     *          there `simd` fails (-Wpass-failed) and can leave the sibling
     *          loops of the same outlined region unvectorised too.
     * @param wait_mode WaitMode::nowait: no barrier at the end.
     */
    template <typename Index, typename Body>
    void for_static_simd(Index first, Index last, Body&& body, [[maybe_unused]] WaitMode wait_mode = WaitMode::wait);

    /**
     * @brief Share [@p first, @p last) out over the team in blocks of
     *        @p chunk iterations, round-robin: block k from @p first to rank
     *        k % size() (`omp for schedule(static, chunk)`).
     * @details A rank gets the same blocks in every loop of the same bounds and
     *          chunk: two nowait loops in a row need no barrier between them
     *          for a rank to read in the second what it wrote in the first.
     *          The chunk is a local of this function: clang 19 and 20 crash
     *          (EmitOMPWorksharingLoop) on a chunk that is a capture of the
     *          region's lambda.
     * @param chunk the block size (below 1 means 1)
     * @param wait_mode     WaitMode::nowait: no barrier at the end.
     */
    template <typename Index, typename Body>
    void for_static_chunks(Index                     first,
                           Index                     last,
                           int                       chunk,
                           Body&&                    body,
                           [[maybe_unused]] WaitMode wait_mode = WaitMode::wait);

    /**
     * @brief for_static_chunks() as `omp for simd schedule(static, chunk)`:
     *        the same blocks (no `simd:` modifier, which would round them),
     *        the loop vectorised. Serial build: for_static_chunks().
     * @param chunk the block size (below 1 means 1)
     * @param wait_mode     WaitMode::nowait: no barrier at the end.
     */
    template <typename Index, typename Body>
    void for_static_chunks_simd(Index                     first,
                                Index                     last,
                                int                       chunk,
                                Body&&                    body,
                                [[maybe_unused]] WaitMode wait_mode = WaitMode::wait);

    /**
     * @brief Share [@p first, @p last) out over the team in blocks of
     *        @p chunk iterations, first come first served
     *        (`omp for schedule(dynamic, chunk)`).
     * @param first first index
     * @param last  one past the last index
     * @param chunk iterations handed out at a time
     * @param body  called as body(i), once per index
     * @param wait_mode WaitMode::nowait: no barrier at the end.
     */
    template <typename Index, typename Body>
    void
    for_dynamic(Index first, Index last, int chunk, Body&& body, [[maybe_unused]] WaitMode wait_mode = WaitMode::wait);

    /**
     * @brief Share [@p first, @p last) out over the team with the calling
     *        task's run-time schedule (`omp for schedule(runtime)`: OMP_SCHEDULE,
     *        or what omp_set_schedule set -- the rechunk refill's guided,16).
     * @param first first index
     * @param last  one past the last index
     * @param body  called as body(i), once per index
     * @param wait_mode WaitMode::nowait: no barrier at the end.
     */
    template <typename Index, typename Body>
    void for_runtime(Index first, Index last, Body&& body, [[maybe_unused]] WaitMode wait_mode = WaitMode::wait);

    /// @brief Wait until every thread of the team gets here (`omp barrier`).
    static void barrier() noexcept
    {
    #ifdef _OPENMP
        #pragma omp barrier
    #endif
    }

    /**
     * @brief Run @p func on ONE thread of the team, whichever gets here first
     *        (`omp single`).
     * @param func called once, without argument
     * @param wait_mode WaitMode::wait (the default): the others wait for it at the
     *          end; nowait: they go on.
     */
    template <typename F>
    void single(F&& func, [[maybe_unused]] WaitMode wait_mode = WaitMode::wait);

    /// @brief Run @p func on the team's rank 0 only, no barrier (`omp masked`).
    template <typename F>
    void masked(F&& func);
};

/// @brief The team the calling thread runs in: inside a region, that region's
///        (the innermost); outside any, a team of one.
[[nodiscard]] inline Team current_team() noexcept
{
    return Team{};
}

/**
 * @brief Run @p body(team) on every thread of a new team.
 * @param opt  use_team (false: a team of one) and num_threads; sched unused
 * @param body callable `void(Team&)`
 */
template <typename Body>
void team(const Options& opt, Body&& body)
{
    #ifdef _OPENMP
    const bool on = opt.use_team;
    const int  nt = detail::team_size(opt);
        #pragma omp parallel if (on) num_threads(nt)
    {
        Team team;
        body(team);
    }
    #else
    (void)opt;
    Team t;
    body(t);
    #endif
}

/**
 * @brief Run @p body(team, acc) on every thread of a new team, each thread
 *        with its own acc[0..N) (zero to start); return their sums
 *        (`omp parallel reduction(+ : acc[:N])`).
 * @param opt  use_team (false: a team of one) and num_threads; sched unused
 * @param body callable `void(Team&, T (&)[N])`
 */
template <typename T, std::size_t N, typename Body>
[[nodiscard]] std::array<T, N> team_sum(const Options& opt, Body&& body)
{
    static_assert(std::is_arithmetic_v<T> && N > 0, "N arithmetic counters");
    T acc[N] = {}; // NOLINT(cppcoreguidelines-avoid-c-arrays): an OpenMP array-section reduction
    #ifdef _OPENMP
    const bool on = opt.use_team;
    const int  nt = detail::team_size(opt);
        #pragma omp parallel if (on) num_threads(nt) reduction(+ : acc[ : N])
    {
        Team team;
        body(team, acc);
    }
    #else
    (void)opt;
    Team t;
    body(t, acc);
    #endif
    std::array<T, N> sums{};
    for (std::size_t k = 0; k < N; ++k)
    {
        sums[k] = acc[k];
    }
    return sums;
}

/**
 * @brief Run @p body(team, flag) on every thread of a new team, each thread
 *        with its own flag (false to start); return whether any set it
 *        (`omp parallel reduction(|| : flag)`).
 * @param opt  use_team (false: a team of one) and num_threads; sched unused
 * @param body callable `void(Team&, bool&)`
 */
template <typename Body>
[[nodiscard]] bool team_any(const Options& opt, Body&& body)
{
    bool flag = false;
    #ifdef _OPENMP
    const bool on = opt.use_team;
    const int  nt = detail::team_size(opt);
        #pragma omp parallel if (on) num_threads(nt) reduction(|| : flag)
    {
        Team team;
        body(team, flag);
    }
    #else
    (void)opt;
    Team t;
    body(t, flag);
    #endif
    return flag;
}

} // namespace fast_alloc::pmp

// clang-format off
#endif // FAST_ALLOC_PMP_BACKEND_STD
// clang-format on

#include "fast_allocators/internal/pmp_loops_inline.h" // the template bodies
