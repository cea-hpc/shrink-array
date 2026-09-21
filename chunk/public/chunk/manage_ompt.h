//
// Created by ballereaup on 21/04/2026.
//

#pragma once

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
 * @file manage_ompt.h
 * @brief Parallel-phase transition hooks of the chunk library.
 *
 * The library defers two kinds of work to well-defined points of the OpenMP
 * lifecycle rather than doing it inline at each write:
 *  - **deferred write ops**: writes issued on a compressed chunk are queued
 *    per-thread and applied later, at a point where no other thread writes;
 *  - **deferred compression**: chunks that became compressible are collected
 *    as "candidates" and compressed in bulk between phases.
 *
 * The functions below are those points. They come in two flavours, selected at
 * build time:
 *  - `HAVE_EMBEDDED_COMPRESSION_WITH_OMPT`: manage_ompt.cpp registers an OMPT
 *    tool and calls them from the runtime callbacks; the application code has
 *    nothing to do (see `build_config::has_ompt`);
 *  - otherwise the application calls them explicitly around its parallel
 *    regions (see the `*_g()` wrappers of examples/shrink_bench, guarded by
 *    `build_config::has_not_ompt` so exactly one of the two paths is active).
 *
 * All of them are no-ops unless `build_config::use_openmp &&
 * chunk_config::is_shrink_enabled`, so they can be called unconditionally from
 * serial or vector-mode builds.
 *
 * @see all_type_chunk.cpp for the definitions.
 */

namespace shrink_array
{

/**
 * @brief Entry point of a parallel phase, called by **every** thread of the team.
 *
 * Caches the OpenMP rank of the calling thread into the thread-local
 * `__fast_omp_rank`, so the per-thread deferred queues can be indexed without
 * paying an `omp_get_thread_num()` call on each write.
 *
 * @note Must run inside the parallel region, before any chunk write.
 */
extern void begin_parallel_phase();

/**
 * @brief End of a parallel phase, called from **outside** the parallel region.
 *
 * Flushes the deferred write queues of all threads (parallel flush, no
 * enclosing barrier needed since the region is already joined), resets the
 * deferred-op pools, then drains the deferred-compression backlog. That drain
 * is skipped in aggressive mode (`chunk_config::have_agressive_compression`),
 * where compression is left entirely to the opportunistic in-loop path.
 *
 * This is the "all-in-one" variant; use the
 * @ref finalize_parallel_phase_only_flush / @ref finalize_parallel_phase_after_flush
 * pair instead when the flush must happen while the region is still open.
 */
extern void finalize_parallel_phase();

/**
 * @brief First half of the split phase-end, called from **inside** the parallel region.
 *
 * Emits an `omp barrier` — the surrounding `omp for` loops are typically
 * `nowait`, so this is what guarantees every thread has finished enqueueing —
 * then each thread drains its **own** deferred queue, using the rank that is
 * only stable while the region is open.
 *
 * Must be paired with @ref finalize_parallel_phase_after_flush, called once the
 * region is closed.
 */
extern void finalize_parallel_phase_only_flush();

/**
 * @brief Second half of the split phase-end, called from **outside** the parallel region.
 *
 * Runs the post-flush cleanup that cannot happen inside the region: destroys
 * the buffers released by the flush, resets the deferred-op pools and drains
 * the deferred-compression backlog (again, unless
 * `chunk_config::have_agressive_compression`).
 *
 * @pre @ref finalize_parallel_phase_only_flush has run inside the region that
 *      just ended.
 */
extern void finalize_parallel_phase_after_flush();

/**
 * @brief Phase transition on a plain barrier, called from **inside** the parallel region.
 *
 * Lighter than @ref finalize_parallel_phase: it only flushes the deferred write
 * queues (per-thread, using the in-region rank) so that the values written
 * before the barrier are visible after it. No pool reset, no compression
 * drain — the region is not over.
 *
 * @note The barrier itself is the caller's responsibility (the OMPT path gets
 *       it from the runtime's sync-region callback, the manual path emits an
 *       `omp barrier` in its wrapper).
 * @return True when at least one deferred op was applied (for the OMPT
 *         sync-region statistics; callers may ignore it).
 */
extern bool finalize_barrier_phase();

} // namespace shrink_array