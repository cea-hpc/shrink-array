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

// CHUNK_LOG=rechunk=debug: where rechunk_to()'s "tmp teardown" phase spends its
// time. rechunk_to() arms the probe just before tmp (the OLD storage, after the
// swap) is destroyed; the teardown path marks its steps; rechunk_to() prints the
// split. Disarmed, a mark is one thread-local load and a branch, on a cold path.
//
// Thread-local: only the thread running rechunk_to() records, and any other
// array torn down meanwhile on another thread leaves this split alone. A step
// that runs inside a team region is timed from outside it by that thread.

#include <array>   // std::array (the per-step times)
#include <chrono>  // std::chrono::steady_clock
#include <cstddef> // std::size_t

namespace shrink_array::internal
{
/// @brief The steps of a teardown, in the order they run (zero_any::ArrayImpl).
enum class TeardownStep : std::size_t
{
    entry,      ///< from the swap to clear_core(): invalidate_proxies, the check-build scan
    flush,      ///< flush_pending_deferred_ops(force)
    candidates, ///< removal from the compression candidate list (by num_tab, by address range)
    terminal,   ///< pool counters, terminal compression sweep, check-build audit
    park,       ///< the team region: scan for written chunks, park their buffers
    registry,   ///< inside it: the second-value registry release (Holder::clear)
    tables,     ///< release_chunk_list() + clear_tab_is_int() (per-chunk metadata tables)
    key_drain,  ///< drain_reuse_list_for_size() of an array with its own allocator key
    erase,      ///< share_info_array->erase (the identity registry)
    list_unmap, ///< the chunk_list_ buffer freed (unmapped from 1 MiB on)
    count       ///< number of steps
};

/// @brief The steps' names, as printed (same order as TeardownStep).
inline constexpr std::array<const char*, static_cast<std::size_t>(TeardownStep::count)> teardown_step_names{
    "entry", "flush", "candidates", "terminal", "park", "registry", "tables", "key drain", "erase", "list unmap"};

/// @brief The per-thread teardown split (see the top of this file).
struct TeardownProbe
{
    using Clock = std::chrono::steady_clock; ///< One system-wide counter.

    bool                                                              on = false; ///< armed by rechunk_to()
    Clock::time_point                                                 last{};     ///< the previous mark
    std::array<double, static_cast<std::size_t>(TeardownStep::count)> ms{};       ///< per step, in ms

    /// @brief Arm: zero the steps, start the clock.
    void start() noexcept
    {
        ms.fill(0.0);
        last = Clock::now();
        on   = true;
    }

    /// @brief Charge the time since the previous mark to @p step.
    void mark(TeardownStep step) noexcept
    {
        if (!on)
        {
            return;
        }
        const auto now = Clock::now();
        ms[static_cast<std::size_t>(step)] += std::chrono::duration<double, std::milli>(now - last).count();
        last = now;
    }

    /// @brief Move @p moved_ms already charged to @p from over to @p to (a
    ///        sub-step timed inside a step).
    void transfer(TeardownStep from, TeardownStep to, double moved_ms) noexcept
    {
        if (!on)
        {
            return;
        }
        ms[static_cast<std::size_t>(from)] -= moved_ms;
        ms[static_cast<std::size_t>(to)] += moved_ms;
    }

    /// @brief Disarm.
    /// @return the instant of the last mark
    Clock::time_point stop() noexcept
    {
        on = false;
        return last;
    }
};

/// @brief This thread's probe.
inline TeardownProbe& teardown_probe() noexcept
{
    static thread_local TeardownProbe probe;
    return probe;
}
} // namespace shrink_array::internal
