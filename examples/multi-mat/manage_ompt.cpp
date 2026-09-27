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
 * @file manage_ompt.cpp
 * @brief The OMPT tool of multi-mat-embedded: what the embedded compression
 *        needs from the OpenMP runtime, and nothing more.
 *
 * The chunk library compresses from its write path; what it asks of the
 * application is the PHASES of chunk/manage_ompt.h -- the deferred writes
 * replayed before they are read, the candidates left over drained when the
 * region ends. This file makes those calls from the runtime's callbacks, so
 * the application source has none:
 *
 *   ompt_callback_implicit_task  (begin)  -> begin_parallel_phase()
 *        every thread of the team, before its first write: the rank cache
 *   ompt_callback_sync_region    (end)    -> finalize_barrier_phase()
 *        every thread, leaving a barrier: its deferred writes, replayed
 *        before the reads that follow the barrier
 *   ompt_callback_parallel_end            -> finalize_parallel_phase()
 *        the encountering thread, the team joined: the deferred writes of
 *        every thread, the pools reset, the candidates compressed
 *
 * finalize_parallel_phase() opens parallel regions of its own (the replay
 * is done by a team), whose end reaches this same callback: a guard makes
 * those inner ends, and the barriers inside them, no-ops.
 *
 * The library ships an equivalent tool (chunk/private/manage_ompt.cpp) in
 * every variant; the variant this example links (chunk_v1v2_<size>) is
 * built without it, so that this file is the one the runtime finds --
 * self-contained, and readable next to the loops it serves. Compiled in
 * the embedded executable alone (CMakeLists.txt); the manual one defines
 * the switch the library reads in ompt_off.cpp, and has no tool.
 */

#include <atomic>
#include <cstdio>

#include <omp-tools.h>

#include "chunk/config.h"
#include "chunk/manage_ompt.h"

/// @brief The suspension counter the library reads and updates
///        (chunk_config::OMPTcc, declared in chunk/config.h): the callbacks
///        below run only while it is zero, which it is at start.
std::atomic<int> g_ompt_suspended{0}; // 0 = callbacks active

namespace
{

/// @brief Set while finalize_parallel_phase() runs: the regions it opens
///        end through on_parallel_end too, and their barriers through
///        on_sync_region. Not on_implicit_task: the threads of those
///        regions need their rank cache like any other (a check build
///        asserts on a stale one).
std::atomic<bool> in_phase_end{false};

void on_implicit_task(ompt_scope_endpoint_t endpoint, ompt_data_t*, ompt_data_t*, unsigned int, unsigned int, int)
{
    if (g_ompt_suspended.load(std::memory_order_relaxed) != 0)
        return;
    if (endpoint == ompt_scope_begin)
        shrink_array::begin_parallel_phase();
}

void on_sync_region(ompt_sync_region_t, ompt_scope_endpoint_t endpoint, ompt_data_t*, ompt_data_t*, const void*)
{
    if (g_ompt_suspended.load(std::memory_order_relaxed) != 0 || in_phase_end.load(std::memory_order_relaxed))
        return;
    if (endpoint == ompt_scope_end)
        shrink_array::finalize_barrier_phase();
}

void on_parallel_end(ompt_data_t*, ompt_data_t*, int, const void*)
{
    if (g_ompt_suspended.load(std::memory_order_relaxed) != 0 || in_phase_end.load(std::memory_order_relaxed))
        return;
    in_phase_end.store(true, std::memory_order_relaxed);
    shrink_array::finalize_parallel_phase();
    in_phase_end.store(false, std::memory_order_relaxed);
}

int ompt_initialize(ompt_function_lookup_t lookup, int, ompt_data_t*)
{
    auto set_callback = reinterpret_cast<ompt_set_callback_t>(lookup("ompt_set_callback"));
    if (set_callback == nullptr)
    {
        std::fputs("multi-mat: ompt_set_callback not found, the embedded compression has no phases\n", stderr);
        return 0; // the tool is dropped
    }
    set_callback(ompt_callback_implicit_task, reinterpret_cast<ompt_callback_t>(on_implicit_task));
    set_callback(ompt_callback_sync_region, reinterpret_cast<ompt_callback_t>(on_sync_region));
    set_callback(ompt_callback_parallel_end, reinterpret_cast<ompt_callback_t>(on_parallel_end));
    std::puts("multi-mat: OMPT tool registered (implicit_task, sync_region, parallel_end)");
    return 1;
}

void ompt_finalize(ompt_data_t*) {}

} // namespace

/// @brief The entry point the OpenMP runtime looks up at start-up.
ompt_start_tool_result_t* ompt_start_tool(unsigned int, const char*)
{
    static ompt_start_tool_result_t result = {&ompt_initialize, &ompt_finalize, {0}};
    return &result;
}
