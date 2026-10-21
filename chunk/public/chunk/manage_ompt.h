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
 * @file manage_ompt.h
 * @brief Parallel-phase transition hooks of the chunk library: the phase
 *        contract (docs/threading_backends.md, section 2).
 *
 * The library defers two kinds of work to well-defined points of the OpenMP
 * lifecycle rather than doing it inline at each write:
 *  - **deferred write ops**: writes issued on a compressed chunk are queued
 *    per-thread and applied later, at a point where no other thread writes;
 *  - **deferred compression**: chunks that became compressible are collected
 *    as "candidates" and compressed in bulk between phases.
 *
 * The functions below are those points. The contract, whatever runs the
 * application's threads:
 *  - a **phase** is the stretch between two serial points, during which the
 *    application's threads write the arrays concurrently;
 *  - **begin**: every thread that writes calls begin_parallel_phase() before
 *    its first write of the phase (it caches the thread's rank);
 *  - **end**: ONE thread calls finalize_parallel_phase() once every writer has
 *    stopped (the team has joined): it drains the deferred queues of every
 *    rank itself, then runs the compression pass on the library's backend. No
 *    barrier inside the application's threads -- the shape every backend
 *    supports.
 *  - Two more shapes need a team barrier in the application's region, so the
 *    OpenMP backend alone has them: the split end
 *    (finalize_parallel_phase_only_flush() by every thread inside the region,
 *    finalize_parallel_phase_after_flush() after it) and the barrier
 *    transition (finalize_barrier_phase()).
 *
 * Who calls them, selected at build time:
 *  - `CHUNK_OPT_OMPT` (an OMPT build): manage_ompt.cpp registers an OMPT
 *    tool and calls them from the runtime callbacks -- the automatic
 *    implementation of the contract for OpenMP applications; the application
 *    code has nothing to do (see `build_config::has_ompt`);
 *  - otherwise the application calls them explicitly around its parallel
 *    regions (see the `*_g()` wrappers of examples/shrink_bench, guarded by
 *    `build_config::has_not_ompt` so exactly one of the two paths is active),
 *    or lets parallel_phase() do it.
 *
 * All of them are no-ops unless `build_config::threaded &&
 * chunk_config::is_shrink_enabled` (`build_config::use_openmp` for the split
 * end and the barrier transition), so they can be called unconditionally from
 * serial or vector-mode builds.
 *
 * parallel_phase() (at the end) places begin_parallel_phase() and
 * finalize_parallel_phase() itself: nothing left to forget.
 *
 * One top-level OpenMP team at a time. The deferred-op queues are indexed
 * by the rank IN the team: two teams started concurrently by different
 * host threads would share ranks 0..n-1, hence queues, and one team's
 * phase end would replay the other's writes in the middle of its region.
 * Refused once an array exists, in every build (phase_guard::teams).
 *
 * @see all_type_chunk.cpp for the definitions.
 */

#include <utility> // std::forward

#include "chunk/config.h"    // build_config::has_not_ompt
#include "chunk/PMP_CHUNK.h" // pmp::team (parallel_phase's region)

namespace shrink_array
{

/**
 * @brief Entry point of a parallel phase, called by **every** thread of the team.
 *
 * Caches the OpenMP rank of the calling thread into the thread-local
 * `g_fast_omp_rank`, so the per-thread deferred queues can be indexed without
 * paying an `omp_get_thread_num()` call on each write.
 *
 * @note Must run inside the parallel region, before any chunk write.
 */
extern void begin_parallel_phase();

/**
 * @brief End of a parallel phase, called from **outside** the parallel region.
 *
 * Flushes every thread's deferred write queue, resets the deferred-op pools,
 * then drains the deferred-compression backlog (skipped in aggressive mode,
 * `chunk_config::aggressive_compression()`, which compresses in-loop only).
 *
 * All-in-one; use finalize_parallel_phase_only_flush() /
 * finalize_parallel_phase_after_flush() when the flush must happen while the
 * region is still open.
 *
 * @note Called inside a parallel region (a parallel_phase() nested in a phase:
 *       a component's, a solver's), it does nothing: the enclosing phase is not
 *       over, and its own end finalizes. Under OMPT likewise, a nested region's
 *       end is not a phase boundary; only the outermost one is.
 */
extern void finalize_parallel_phase();

/**
 * @brief First half of the split phase-end, called from **inside** the parallel region.
 *
 * Meets the team on its barrier (pmp::current_team().barrier(): the `omp for`
 * loops are typically `nowait`, this guarantees every thread has finished
 * enqueueing), then each thread drains its **own** queue, by the rank that is
 * only stable while the region is open. OpenMP backend only: the barrier is
 * the application's team's.
 *
 * Must be paired with finalize_parallel_phase_after_flush(), called once the
 * region is closed.
 */
extern void finalize_parallel_phase_only_flush();

/**
 * @brief Second half of the split phase-end, called from **outside** the parallel region.
 *
 * Runs the post-flush cleanup that cannot happen inside the region: destroys
 * the buffers released by the flush, resets the deferred-op pools and drains
 * the deferred-compression backlog (again, unless
 * `chunk_config::aggressive_compression()`).
 *
 * @pre finalize_parallel_phase_only_flush() has run inside the region that
 *      just ended.
 */
extern void finalize_parallel_phase_after_flush();

/**
 * @brief Phase transition on a plain barrier, called from **inside** the parallel region.
 *
 * Only flushes the deferred write queues (per thread, in-region rank) so values
 * written before the barrier are visible after it. No pool reset, no
 * compression drain.
 *
 * @note The barrier itself is the caller's responsibility (the OMPT path gets
 *       it from the runtime's sync-region callback, the manual path emits an
 *       `omp barrier` in its wrapper).
 * @return True when at least one deferred op was applied (for the OMPT
 *         sync-region statistics; callers may ignore it).
 */
extern bool finalize_barrier_phase();

/**
 * @brief Once per process: warn if this build relies on the OMPT tool and the
 *        OpenMP runtime did not load it.
 *
 * In an OMPT build (CHUNK_OPT_OMPT) the phase boundaries run from the OMPT
 * callbacks only; if the runtime did not load the tool (OMP_TOOL=disabled,
 * another tool first in OMP_TOOL_LIBRARIES, ompt_start_tool not found) they
 * silently never happen. This forces the runtime's initialisation, then prints
 * one line to stderr if the tool is not active (aborts with CHUNK_REQUIRE_OMPT=1).
 *
 * Called by the AllTypeChunk constructor. A no-op without the OMPT tool.
 *
 * Weak: a library built NO_OMPT_TOOL (the program ships its own tool, e.g.
 * examples/multi-mat/manage_ompt.cpp) has no definition, and the call is skipped.
 */
__attribute__((weak)) extern void check_ompt_tool_loaded();

/**
 * @brief A parallel region with the phase protocol already in place:
 *        begin_parallel_phase() by every thread before @p body,
 *        finalize_parallel_phase() once the region is over.
 * @details A team of the backend (pmp::team). With OpenMP, the same as
 *          @code
 *          #pragma omp parallel num_threads(num_threads)
 *          {
 *              begin_parallel_phase();
 *              body();
 *          }
 *          finalize_parallel_phase();
 *          @endcode
 *          with both calls left out under OMPT, where the runtime makes them.
 *          The worksharing goes inside @p body, orphaned: `#pragma omp for`
 *          with its schedule, `nowait`, a reduction on a variable declared
 *          outside... Compiled without OpenMP, @p body runs once.
 * @param num_threads team size (the num_threads clause); 0: the runtime's.
 * @param body        what every thread of the team runs.
 */
template <typename Body>
void parallel_phase(int num_threads, Body&& body)
{
    auto team_body = [&body]
    {
        if constexpr (build_config::has_not_ompt)
        {
            begin_parallel_phase();
        }
        body();
    };
    // num_threads 0 (or less): the backend's default team (OpenMP: omp_get_max_threads()).
    pmp::team(pmp::options{.num_threads = num_threads}, [&team_body](pmp::Team& /*team*/) { team_body(); });
    if constexpr (build_config::has_not_ompt)
    {
        finalize_parallel_phase();
    }
}

/// @brief parallel_phase() with the runtime's team size.
template <typename Body>
void parallel_phase(Body&& body)
{
    parallel_phase(0, std::forward<Body>(body));
}

} // namespace shrink_array