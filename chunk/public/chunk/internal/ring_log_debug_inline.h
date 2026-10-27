/*
 * Copyright 2025-2026 CEA (www.cea.fr)
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
#include "chunk/ring_log_debug.h" // the class these bodies belong to

namespace shrink_array
{

// ---------------------------------------------------------------------------
// shrink_array::ScopedRingLog<BeginEvent, EndEvent, Manual>
// ---------------------------------------------------------------------------

template <EventCode BeginEvent, EventCode EndEvent, bool Manual>
ScopedRingLog<BeginEvent, EndEvent, Manual>::ScopedRingLog(uint32_t payload_value) noexcept
{
    if constexpr (chunk_config::ring_log_debug)
    {
        tid_     = get_tid();
        payload_ = payload_value;
        ended_   = false;
        ring_log_start(tid_, BeginEvent, payload_);
    }
}

template <EventCode BeginEvent, EventCode EndEvent, bool Manual>
ScopedRingLog<BeginEvent, EndEvent, Manual>::~ScopedRingLog() noexcept
{
    if constexpr (chunk_config::ring_log_debug)
    {
        if constexpr (!Manual)
        {
            ring_log_end(tid_, EndEvent, payload_); // end, not start
        }
        else
        {
            // Manual mode: stop (in check builds only) if the user forgot to call .end()
            if constexpr (build_config::any_check)
            {
                if (!ended_)
                {
                    CHUNK_INVARIANT_FAILED("ScopedRingLog<Manual>: end() not called before destruction "
                                           "(BEGIN_EVENT={})",
                                           static_cast<int>(BeginEvent));
                }
            }
        }
    }
}

template <EventCode BeginEvent, EventCode EndEvent, bool Manual>
void ScopedRingLog<BeginEvent, EndEvent, Manual>::end() noexcept
{
    if constexpr (chunk_config::ring_log_debug && Manual)
    {
        if (!ended_)
        {
            ring_log_end(tid_, EndEvent, payload_);
            ended_ = true;
        }
    }
}

} // namespace shrink_array
