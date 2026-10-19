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
 *        the OpenMP and pool backends (docs/threading_backends.md).
 *
 * A region is bench_par::region(body), body(Team& t) run by every thread of a
 * team of the library's backend (pmp::team): OpenMP, a `#pragma omp parallel`
 * whose body calls the lambda; pool, a team of the pool's std::jthreads;
 * serial, the calling thread.
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
 * - pool / serial: the macro is a loop over the calling thread's share,
 *   computed from t.rank() and t.size() exactly as libgomp shares the same
 *   schedule out. A rank gets the same iterations in every loop of the same
 *   bounds and schedule -- what the nowait reasoning of the phases relies on
 *   (bench_phase_compute.cpp) -- and the same blocks from one region to the
 *   next (the pool's worker k is always rank k: first touch).
 *
 * The schedules: _STATIC (`schedule(static)`, one block per thread), _BLOCK
 * (`schedule(static, sched_blk)`, a local named sched_blk:
 * bench_sched_for(), bench_prelude.h), _ROUND_ROBIN (`schedule(static, 1)`).
 * A loop without NOWAIT ends with a barrier in OpenMP; in the other builds
 * write BENCH_END_FOR(t) after its body.
 *
 * Reductions (_RED): the body adds to BENCH_RED(v), not to v. OpenMP: the
 * pragma keeps its `reduction(+ : v)` on the shared v, and BENCH_RED(v) is v.
 * Other builds: each thread adds into a private partial, BENCH_RED(v), which
 * goes into v atomically once the thread's share is done -- whole after the
 * next barrier or the end of the region, as with OpenMP.
 *
 * The macros name the loop's variable, so the body is the same text in every
 * build; `continue` works, `break` is refused by OpenMP anyway.
 */

#pragma once

#include <algorithm> // std::min (the block schedule's last block)
#include <atomic>    // std::atomic_ref (the reductions' partials, atomic_add)
#include <mutex>     // std::mutex (critical())
#include <utility>   // std::forward

#include "bench_int.h"       // Int
#include "chunk/PMP_CHUNK.h" // shrink_array::pmp::team, pmp::Team, pmp::options

namespace bench_par
{
using Team = shrink_array::pmp::Team;

/// @brief Team::single(f, nowait): no barrier after it.
inline constexpr auto nowait = shrink_array::pmp::nowait;

/// @brief Run @p body(Team&) on every thread of a new team of the backend's default size.
template <typename Body>
void region(Body&& body)
{
    shrink_array::pmp::team(shrink_array::pmp::options{}, std::forward<Body>(body));
}

/// @brief Run @p body(Team&) on every thread of a new team of @p num_threads threads.
template <typename Body>
void region(int num_threads, Body&& body)
{
    shrink_array::pmp::team(shrink_array::pmp::options{.num_threads = num_threads}, std::forward<Body>(body));
}

/// @brief The calling thread's share [lo, hi) of [@p b, @p e) under
///        `schedule(static)`: libgomp's split, the first (e - b) % size ranks
///        one iteration more.
[[nodiscard]] inline Int static_lo(const Team& t, Int b, Int e) noexcept
{
    const Int n = e - b;
    if (n <= 0)
    {
        return b;
    }
    const Int nt = t.size();
    const Int r  = t.rank();
    const Int q  = n / nt;
    const Int m  = n % nt;
    return b + (r < m ? r * (q + 1) : (m * (q + 1)) + ((r - m) * q));
}

/// @brief One past the end of static_lo()'s share.
[[nodiscard]] inline Int static_hi(const Team& t, Int b, Int e) noexcept
{
    const Int n = e - b;
    if (n <= 0)
    {
        return b;
    }
    const Int nt = t.size();
    const Int r  = t.rank();
    return static_lo(t, b, e) + (n / nt) + (r < n % nt ? 1 : 0);
}

/// @brief v += x, atomically (`omp atomic` with OpenMP).
template <typename T>
inline void atomic_add(T& v, T x) noexcept
{
#ifdef _OPENMP
    #pragma omp atomic
    v += x;
#else
    std::atomic_ref<T>(v).fetch_add(x, std::memory_order_relaxed);
#endif
}

/// @brief The lock of critical(): one for the whole bench, as an unnamed
///        `omp critical`.
[[nodiscard]] inline std::mutex& critical_mutex() noexcept
{
    static std::mutex m;
    return m;
}

/// @brief Run @p f under the bench's one lock (an unnamed `omp critical`).
template <typename F>
void critical(F&& f)
{
    const std::scoped_lock lock(critical_mutex());
    f();
}

/// @brief A thread's partial of a reduction (builds without OpenMP): added
///        into the shared variable when the loop's share is done.
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
    #define BENCH_FOR_STATIC(t, i, b, e) _Pragma("omp for schedule(static)") for (Int i = (b); i < (e); ++i)
    #define BENCH_FOR_BLOCK_NOWAIT_SCALAR(t, i, b, e) BENCH_OMP_FOR_BLOCK_NOWAIT_SCALAR for (Int i = (b); i < (e); ++i)
    #define BENCH_FOR_ROUND_ROBIN_NOWAIT(t, i, b, e) _Pragma("omp for schedule(static, 1) nowait") for (Int i = (b); i < (e); ++i)
    #define BENCH_FOR_STATIC_NOWAIT_RED_SCALAR(t, i, b, e, v) BENCH_OMP_FOR_STATIC_NOWAIT_RED_SCALAR(v) for (Int i = (b); i < (e); ++i)
    #define BENCH_FOR_BLOCK_NOWAIT_RED_SCALAR(t, i, b, e, v) BENCH_OMP_FOR_BLOCK_NOWAIT_RED_SCALAR(v) for (Int i = (b); i < (e); ++i)
    #define BENCH_RED(v) v
    #define BENCH_END_FOR(t) static_cast<void>(0)

#else

    // The calling thread's share, as libgomp deals the same schedule out.
    #define BENCH_FOR_STATIC_NOWAIT(t, i, b, e) \
        for (Int i = ::bench_par::static_lo((t), (b), (e)), i##_end_ = ::bench_par::static_hi((t), (b), (e)); i < i##_end_; ++i)
    #define BENCH_FOR_STATIC_NOWAIT_SCALAR(t, i, b, e) BENCH_FOR_STATIC_NOWAIT(t, i, b, e)
    #define BENCH_FOR_STATIC(t, i, b, e) BENCH_FOR_STATIC_NOWAIT(t, i, b, e)
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
    #define BENCH_END_FOR(t) (t).barrier()

#endif
// clang-format on
