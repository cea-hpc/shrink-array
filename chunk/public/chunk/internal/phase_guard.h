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
 * The guard is a generation counter: every phase end bumps it, every
 * begin_parallel_phase() stamps the calling thread with the current value,
 * and the point where a write becomes deferred checks that the stamp is
 * current. A stale stamp means the region was entered without
 * begin_parallel_phase(), or the previous phase was never finalized.
 *
 * Check builds only (build_config::any_check), and only where the protocol
 * is manual (build_config::has_not_ompt): under OMPT the runtime callbacks
 * make the calls themselves. Costs nothing elsewhere.
 */

#include <atomic>

#include "chunk/config.h"

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
} // namespace shrink_array::phase_guard
