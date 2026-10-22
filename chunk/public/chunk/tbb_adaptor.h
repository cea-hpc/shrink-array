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

/**
 * @file tbb_adaptor.h
 * @brief shrink_array::tbb::parallel_for: oneTBB's parallel_for with the phase
 *        contract already in place (docs/threading_backends.md, section 2).
 *
 * A TBB application writes the arrays from TBB's threads, which no runtime of
 * the library numbers. The phase contract every backend supports is the
 * serial one: begin_parallel_phase() on every thread that writes, before its
 * first write, then finalize_parallel_phase() once, from one thread, after the
 * writers stopped. TBB gives no hook at the start of a task on a worker, so
 * the adaptor places both calls itself:
 *
 * @code
 * shrink_array::tbb::parallel_for(oneapi::tbb::blocked_range<long>(0, n),
 *                                 [&](const oneapi::tbb::blocked_range<long>& r)
 *                                 {
 *                                     for (long i = r.begin(); i != r.end(); ++i)
 *                                         a[i] = f(i);
 *                                 });
 * // here: the deferred writes are applied, the compression pass has run
 * @endcode
 *
 * is
 *
 * @code
 * oneapi::tbb::parallel_for(range, [&](auto& r) { begin_parallel_phase(); body(r); });
 * finalize_parallel_phase();
 * @endcode
 *
 * begin_parallel_phase() runs before every piece of the range a thread
 * executes (a thread that runs several pieces calls it again: same rank,
 * same queue, allowed within the phase), and finalize_parallel_phase() once
 * oneapi::tbb::parallel_for has returned -- every task done -- also when the
 * body threw (the exception then goes on to the caller).
 *
 * The forms of oneapi::tbb::parallel_for, the same arguments:
 *  - a range and a body, then optionally a partitioner
 *    (oneapi::tbb::static_partitioner keeps a piece on the same thread from
 *    one call to the next: first touch) and a task_group_context;
 *  - first, last [, step], a function of the index, then optionally a
 *    partitioner and a context. The adaptor runs it on a blocked_range of
 *    the iteration numbers, so begin_parallel_phase() runs once per piece,
 *    not once per index.
 *
 * Backends (CHUNK_THREADS):
 *  - pool: each TBB thread that writes takes a rank of the registry
 *    (pmp::registered_rank()), the application phase is open while the
 *    tasks run (in_parallel() true, no team of the pool), and the flush and
 *    the compression pass of finalize_parallel_phase() run on the pool.
 *    The registry must hold every TBB thread alive at once and the calling
 *    thread: its capacity is that of chunk_thread_capacity() (the hardware
 *    threads by default, CHUNK_THREAD_CAPACITY to raise it) -- enough for
 *    TBB's default arena; a task_arena wider than the machine needs
 *    CHUNK_THREAD_CAPACITY.
 *  - serial: the library is not thread-safe, so the body runs on the
 *    calling thread over the whole range (the index form: every index in
 *    order), without TBB.
 *  - openmp: refused at compile time. OpenMP numbers only its own threads:
 *    every TBB thread would be rank 0 and share one queue.
 *
 * One phase at a time: two adaptor calls running at once (from two threads,
 * or one nested in the body of the other) share the application phase, and
 * the first to return would finalize under the other's writers. A nested
 * oneapi::tbb::parallel_for inside the body is fine: its tasks are part of the
 * same phase -- but call begin_parallel_phase() at the top of its body, or use
 * the adaptor's body wrapper, phase_body().
 *
 * Header-only, and the library does not depend on TBB: the program that
 * includes this header compiles and links with oneTBB (TBB::tbb).
 */

#include <concepts>    // std::integral, std::invocable
#include <stdexcept>   // std::invalid_argument (a step that is not positive)
#include <type_traits> // std::is_same_v
#include <utility>     // std::forward

#include <oneapi/tbb/blocked_range.h> // oneapi::tbb::blocked_range (the index forms)
#include <oneapi/tbb/parallel_for.h>  // oneapi::tbb::parallel_for

#include "chunk/PMP_CHUNK.h"   // pmp::backend, pmp::BackendKind
#include "chunk/manage_ompt.h" // begin_parallel_phase, finalize_parallel_phase

namespace shrink_array::tbb
{
namespace detail
{
/// @brief True; a dependent false for the OpenMP backend, so the refusal fires
///        only when the adaptor is called, not when the header is included.
template <typename T>
inline constexpr bool backend_supported = pmp::backend != pmp::BackendKind::openmp || std::is_same_v<T, void>;

/// @brief The message of that refusal.
#define SHRINK_TBB_ADAPTOR_OPENMP_REFUSED                                                                             \
    "shrink_array::tbb::parallel_for needs the pool or the serial backend (CHUNK_THREADS=pool): the OpenMP backend "  \
    "numbers only OpenMP's threads, and every TBB thread would share rank 0"

/// @brief Ends the phase when it goes out of scope: after the tasks, also
///        when the body threw.
struct PhaseEnd
{
    PhaseEnd()                           = default;
    PhaseEnd(const PhaseEnd&)            = delete;
    PhaseEnd& operator=(const PhaseEnd&) = delete;
    PhaseEnd(PhaseEnd&&)                 = delete;
    PhaseEnd& operator=(PhaseEnd&&)      = delete;
    ~PhaseEnd()
    {
        finalize_parallel_phase();
    }
};

/// @brief The body of the index forms: iteration numbers [r.begin(), r.end())
///        of the loop first, first + step, ... < last.
template <typename Index, typename Function>
struct IndexBody
{
    Index           first;
    Index           step;
    const Function& f;

    void operator()(const oneapi::tbb::blocked_range<Index>& r) const
    {
        for (Index k = r.begin(); k != r.end(); ++k)
        {
            f(static_cast<Index>(first + (k * step)));
        }
    }
};

/// @brief Iterations of first, first + step, ... < last (0 when empty).
template <std::integral Index>
[[nodiscard]] Index iterations(Index first, Index last, Index step) noexcept
{
    return last > first ? static_cast<Index>(((last - first) + (step - 1)) / step) : Index(0);
}
} // namespace detail

/**
 * @brief A body that calls begin_parallel_phase() before every piece it runs,
 *        then @p body on it.
 * @details What the adaptor gives oneapi::tbb::parallel_for; for the body of a
 *          oneapi::tbb::parallel_for nested in an adaptor call, or of another
 *          TBB algorithm run between begin and finalize by hand. Holds a
 *          reference: @p body must outlive the algorithm.
 */
template <typename Body>
struct PhaseBody
{
    const Body& body;

    template <typename Range>
    void operator()(Range& r) const
    {
        begin_parallel_phase();
        body(r);
    }
};

/// @brief PhaseBody of @p body.
template <typename Body>
[[nodiscard]] PhaseBody<Body> phase_body(const Body& body) noexcept
{
    return PhaseBody<Body>{body};
}

/**
 * @brief oneapi::tbb::parallel_for(range, body, extra...) as one phase of the
 *        library: begin_parallel_phase() before every piece, then
 *        finalize_parallel_phase() once all are done.
 * @param range a TBB range (blocked_range, blocked_range2d, ...)
 * @param body  called with each piece, `body(Range&)` as TBB calls it
 * @param extra the partitioner and / or the task_group_context of
 *              oneapi::tbb::parallel_for, passed on as they are
 * @details Serial backend: body(a copy of @p range), on the calling thread.
 */
template <typename Range, typename Body, typename... Extra>
    requires(!std::integral<Range>)
void parallel_for(const Range& range, const Body& body, Extra&&... extra)
{
    static_assert(detail::backend_supported<Body>, SHRINK_TBB_ADAPTOR_OPENMP_REFUSED);
    if constexpr (pmp::backend == pmp::BackendKind::serial)
    {
        Range whole = range;
        body(whole);
        ((void)extra, ...);
    }
    else
    {
        const detail::PhaseEnd end;
        oneapi::tbb::parallel_for(range, shrink_array::tbb::phase_body(body), std::forward<Extra>(extra)...);
    }
}

/**
 * @brief oneapi::tbb::parallel_for(first, last, step, f, extra...) as one
 *        phase of the library: f(i) for i = first, first + step, ... < last.
 * @param first first index
 * @param last  the indices stay below it
 * @param f     called as f(i), once per index
 * @param step  positive, as oneapi::tbb::parallel_for requires (std::invalid_argument otherwise)
 * @param extra partitioner and / or task_group_context
 * @details Run on a blocked_range of the iteration numbers (grain 1, as
 *          TBB's own index form): begin_parallel_phase() once per piece.
 *          Serial backend: every index in order, on the calling thread.
 */
template <std::integral Index, typename Function, typename... Extra>
    requires std::invocable<const Function&, Index>
void parallel_for(Index first, Index last, Index step, const Function& f, Extra&&... extra)
{
    static_assert(detail::backend_supported<Function>, SHRINK_TBB_ADAPTOR_OPENMP_REFUSED);
    if (step <= Index(0))
    {
        throw std::invalid_argument("shrink_array::tbb::parallel_for: step must be positive"); // as TBB
    }
    const Index n = detail::iterations(first, last, step);
    if constexpr (pmp::backend == pmp::BackendKind::serial)
    {
        for (Index k = 0; k < n; ++k)
        {
            f(static_cast<Index>(first + (k * step)));
        }
        ((void)extra, ...);
    }
    else
    {
        if (n == 0)
        {
            return; // no task, no phase (TBB's index form does nothing either)
        }
        // Qualified: unqualified, argument-dependent lookup would also find
        // oneapi::tbb::parallel_for (blocked_range is TBB's), which wins the
        // overload resolution -- and the phase would neither begin nor end.
        shrink_array::tbb::parallel_for(oneapi::tbb::blocked_range<Index>(Index(0), n),
                                        detail::IndexBody<Index, Function>{.first = first, .step = step, .f = f},
                                        std::forward<Extra>(extra)...);
    }
}

/// @brief The index form with step 1: f(i) for i in [@p first, @p last).
template <std::integral Index, typename Function, typename... Extra>
    requires std::invocable<const Function&, Index>
void parallel_for(Index first, Index last, const Function& f, Extra&&... extra)
{
    shrink_array::tbb::parallel_for(first, last, Index(1), f, std::forward<Extra>(extra)...);
}

#undef SHRINK_TBB_ADAPTOR_OPENMP_REFUSED

} // namespace shrink_array::tbb
