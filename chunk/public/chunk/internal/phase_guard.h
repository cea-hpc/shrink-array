/*
 * Copyright 2024-2026 CEA (www.cea.fr)
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
 * @file phase_guard.h
 * @brief Check-build guard of the parallel-phase protocol (manage_ompt.h).
 *
 * Without OMPT the library does not see region boundaries. A write can be
 * DEFERRED to the writing thread's queue (once the chunk's candidate list is
 * flagged dangerous), and it is the application that drains those queues
 * between regions: begin_parallel_phase() by every thread inside the region,
 * finalize_parallel_phase() (or the only_flush / after_flush pair) outside
 * it. Skip the brackets and a deferred write of one region lands AFTER a
 * direct write of the next one -- values silently wrong, only on the chunks
 * that happened to be deferred, only at the thread counts that push a list
 * past the threshold. compress_roundtrip found it that way on Chunk2A<bool>.
 *
 * Measured (8 threads, forced deferrals, team size changing from region to
 * region): finalize forgotten between regions replays a freed deferred op
 * (DEFERRED_LIFECYCLE: op magic poisoned); begin and finalize both forgotten
 * leave wrong values with no message in a release build. With a fixed team
 * both went unnoticed -- the trap only springs when the team changes.
 *
 * The guard is a generation counter: every phase end bumps it, every
 * begin_parallel_phase() stamps the calling thread with the current value.
 * Two checks:
 *  - finalize forgotten: a begin_parallel_phase() by a thread whose stamp is
 *    ALREADY current -- it began in this phase and no phase end came since
 *    (check_begin()). Deterministic: it does not wait for a write to be
 *    deferred, which a normal run almost never does. Only at the outermost
 *    level: a region nested in it runs on the same thread, in the same
 *    phase, and may call begin again.
 *  - begin forgotten: a write about to be DEFERRED inside a parallel region
 *    by a thread whose stamp is not current (check_write(), at the enqueue).
 *    Not on every write: a write needs no begin as such (the rank cache
 *    fills itself), and an array created and used inside an `omp single`
 *    without the brackets is legitimate (region_storage_validation). What
 *    needs the brackets is a write left queued in a rank's queue.
 *
 * Check builds only (build_config::any_check), and only where the protocol
 * is manual (build_config::has_not_ompt): under OMPT the runtime callbacks
 * make the calls themselves. Costs nothing elsewhere. shrink_array::
 * parallel_phase() (manage_ompt.h) makes both calls for the program.
 */

#include <atomic> // std::atomic, std::memory_order_relaxed

#include "chunk/PMP_CHUNK.h"                // CHUNK_PMP_IN_PARALLEL_REGION, CHUNK_PMP_LEVEL, chunk_pmp_team_rank
#include "chunk/config.h"                   // build_config::any_check, build_config::has_not_ompt
#include "chunk/internal/invariant_check.h" // CHUNK_INVARIANT_FAILED

namespace shrink_array::phase_guard
{
/// @brief Phase generation: bumped by every phase end, read by begin and by the check.
inline std::atomic<unsigned> generation{1};

/// @brief The generation this thread's last begin_parallel_phase() saw.
inline thread_local unsigned armed_for = 0;

/// @brief begin_parallel_phase(): stamp the calling thread with the current generation.
inline void arm() noexcept
{
    armed_for = generation.load(std::memory_order_relaxed);
}

/// @brief Phase end: open a new generation, so a thread that does not re-arm shows as stale.
inline void next_phase() noexcept
{
    generation.fetch_add(1, std::memory_order_relaxed);
}

/// @brief Is the calling thread armed for the current phase?
[[nodiscard]] inline bool armed() noexcept
{
    return armed_for == generation.load(std::memory_order_relaxed);
}

/// @brief Whether the check applies in this build: manual protocol, check build.
inline constexpr bool active = build_config::any_check && build_config::use_openmp && build_config::has_not_ompt;

/// @brief begin_parallel_phase(): fail if the calling thread already began in
///        this phase (finalize forgotten), then stamp it.
/// @details Outside any region (level 0) it does nothing: there is no region
///          to bracket. In a nested region it only stamps (see the file).
inline void check_begin() noexcept
{
    if constexpr (active)
    {
        const int level = CHUNK_PMP_LEVEL;
        if (level == 0)
        {
            return;
        }
        if (level == 1 && armed()) [[unlikely]]
        {
            CHUNK_INVARIANT_FAILED(
                "[PHASE PROTOCOL] begin_parallel_phase() twice in the same phase on thread {}: the region before "
                "this one did not end with finalize_parallel_phase() (or the only_flush / after_flush pair), so its "
                "deferred writes are still queued and replay after this region's -- wrong values, or a freed op "
                "replayed when the team size changes. Call finalize_parallel_phase() after every region that "
                "called begin_parallel_phase(), or let shrink_array::parallel_phase() do both.",
                chunk_pmp_team_rank());
        }
        arm();
    }
}

/// @brief A write about to be deferred: fail if it runs in a parallel region
///        on a thread that did not call begin_parallel_phase() in this phase.
/// @param chunk the chunk written, @param num the element (for the message).
inline void check_write(const void* chunk, long num) noexcept
{
    if constexpr (active)
    {
        if (CHUNK_PMP_IN_PARALLEL_REGION && !armed()) [[unlikely]]
        {
            CHUNK_INVARIANT_FAILED(
                "[PHASE PROTOCOL] deferred write in a parallel region by thread {}, which did not call "
                "begin_parallel_phase() since the last finalize_parallel_phase(): without OMPT every thread of a "
                "region that writes calls begin_parallel_phase() before its first write, and "
                "finalize_parallel_phase() runs after the region, or this write replays after a later region's "
                "writes -- or let shrink_array::parallel_phase() do both. chunk={} num={}",
                chunk_pmp_team_rank(), chunk, num);
        }
    }
}
} // namespace shrink_array::phase_guard
