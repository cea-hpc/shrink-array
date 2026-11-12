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
 * @file array_chunk_rechunk_trace.h
 * @brief The re-chunk's traces: phase timings, slice-region split, RSS probes.
 *
 * @details Three channels, read once at the start of each re-chunk:
 *          - CHUNK_LOG=rechunk=debug: the per-phase wall time of rechunk_to()
 *            (RechunkPhaseTrace), the split of the zero refill's slice regions
 *            (SliceProbe, summed in RechunkSliceTimes) and the rebuild schedule;
 *          - CHUNK_LOG=rechunk=trace: the RSS after each slice;
 *          - CHUNK_LOG=memory=debug: exact RSS at the phase boundaries, the
 *            new geometry's occupancy and the old pool's census. A channel of
 *            its own so the phase times are not measured with these probes
 *            inside them (they cost seconds per run on large arrays).
 *
 *          Nothing here changes what a re-chunk does; with the channels off
 *          each probe is one branch on a bool read at its start.
 *
 * @see array_chunk_rechunk_inline.h, the algorithm these probes bracket
 * @see scripts/rechunk_trace_summary.py, which sums the lines printed here
 */

#include <array>   // std::array (the phase marks)
#include <atomic>  // SliceProbe's team-shared maxima
#include <chrono>  // steady_clock
#include <cstddef> // std::size_t
#include <ratio>   // std::milli

#include "chunk/internal/exact_rss.h"      // exact_rss_kib (rss_mib_now)
#include "chunk/internal/teardown_probe.h" // teardown_probe (the tmp teardown split)
#include "chunk/log.h"                     // CHUNK_LOG, logging::enabled

namespace shrink_array::internal
{
/// @brief CHUNK_LOG=rechunk=debug: the per-phase timings of every re-chunk, its
///        rebuild schedule and the split of the slice regions' wall time.
[[nodiscard]] inline bool rechunk_trace_on() noexcept
{
    return logging::enabled(logging::Channel::rechunk, logging::Level::debug);
}

/// @brief CHUNK_LOG=memory=debug: the MEMORY side of the trace -- exact RSS at
///        the phase boundaries (a page-table walk per read), the new
///        geometry's occupancy, the allocator census.
[[nodiscard]] inline bool rechunk_trace_mem_on() noexcept
{
    return logging::enabled(logging::Channel::memory, logging::Level::debug);
}

/// @brief CHUNK_LOG=rechunk=trace: one RSS line per slice of the zero refill,
///        a level above the timings (the per-slice lines would bury their
///        summary). Shows the peak's SHAPE: a saw-tooth means the drains return
///        memory, a monotonic climb that they return nothing.
[[nodiscard]] inline bool rechunk_slice_trace_on() noexcept
{
    return logging::enabled(logging::Channel::rechunk, logging::Level::trace);
}

/// @brief Resident set size of this process in MiB, or -1 if unreadable.
/// @details EXACT (smaps_rollup, see exact_rss.h): statm lags by gigabytes
///          under many threads. Only the traces read it.
[[nodiscard]] inline long rss_mib_now() noexcept
{
    const long kib = shrink_array::exact_rss_kib();
    return kib >= 0 ? kib / 1024 : -1;
}

/// @brief Milliseconds between two steady_clock instants.
[[nodiscard]] inline double elapsed_ms(std::chrono::steady_clock::time_point from,
                                       std::chrono::steady_clock::time_point to) noexcept
{
    return std::chrono::duration<double, std::milli>(to - from).count();
}

/**
 * @brief Where the zero refill's slices spend their time, summed over the
 *        slices of the re-chunk in progress (ms).
 *
 * @details Filled under rechunk_trace_on() only, by the master between or
 *          outside the slice regions; reset by RechunkPhaseTrace at the start
 *          of each re-chunk and printed by it at the end.
 *
 *          Instants on steady_clock (one system-wide counter), relative to the
 *          master just before the region:
 *            fork      the LAST thread's entry into the region
 *            loop      the last end of a thread's rebuild loop, minus fork
 *            barrier   the master past the closing barrier, minus the last loop end
 *            boundary  the slice-boundary drain + recycling (~ drain + collect)
 *            join      the master out of the region, minus the end of the boundary
 *          work max / avg: one thread's rebuild loop, max and mean over the
 *          team (their gap is the imbalance); setup max: guard + buffer before
 *          the loop.
 */
struct RechunkSliceTimes
{
    double drain_ms     = 0.0; ///< slice boundaries: the team drain of the old-size reuse lists
    double collect_ms   = 0.0; ///< slice boundaries: detaching + recycling the slabs it emptied
    int    count        = 0;   ///< slices of the refill (0: not traced)
    double fork_ms      = 0.0; ///< see the class description
    double loop_ms      = 0.0; ///< see the class description
    double barrier_ms   = 0.0; ///< see the class description
    double boundary_ms  = 0.0; ///< see the class description
    double join_ms      = 0.0; ///< see the class description
    double work_max_ms  = 0.0; ///< see the class description
    double work_avg_ms  = 0.0; ///< see the class description
    double setup_max_ms = 0.0; ///< see the class description
    // The team drain at the boundaries, per thread (drain_probe.h): max and
    // mean over the team of its own-rank passes, summed over the boundaries.
    double drain_sort_max_ms = 0.0; ///< sorting the parked blocks by owner, slowest thread
    double drain_sort_avg_ms = 0.0; ///< same, mean thread
    double drain_free_max_ms = 0.0; ///< freeing the blocks its rank owns, slowest thread
    double drain_free_avg_ms = 0.0; ///< same, mean thread
    double drain_blocks_max  = 0.0; ///< blocks freed by the busiest rank
    double drain_blocks_avg  = 0.0; ///< blocks freed by the mean rank
    // "collect" split: detaching (team) vs recycling into the arena (one thread).
    double collect_detach_ms = 0.0; ///< drain end -> the recycling single (detach pass + barrier)
    double collect_slabs     = 0.0; ///< slabs handed to the arena, all boundaries
};

/// @brief The slice times of the re-chunk in progress (re-chunks are serial).
inline RechunkSliceTimes g_slice_times;

/**
 * @brief The instants of ONE slice region, shared by its team.
 *
 * @details begin() on the master before the region; thread_done() on every
 *          thread after its rebuild loop; @c past_barrier / @c past_boundary
 *          set by the master inside the region; accumulate() on the master
 *          after it. Only under rechunk_trace_on().
 */
struct SliceProbe
{
    using Clock = std::chrono::steady_clock; ///< the probe's clock

    std::atomic<long long> entry_max{0};    ///< ns since @c start: the last thread in
    std::atomic<long long> loop_end_max{0}; ///< ns since @c start: the last loop end
    std::atomic<long long> work_max{0};     ///< ns: the longest rebuild loop
    std::atomic<long long> work_sum{0};     ///< ns: the rebuild loops, summed
    std::atomic<long long> setup_max{0};    ///< ns: the longest guard + buffer setup
    std::atomic<int>       n{0};            ///< threads that reported
    Clock::time_point      start{};         ///< master, before the region
    Clock::time_point      past_barrier{};  ///< master, past the closing barrier
    Clock::time_point      past_boundary{}; ///< master, past the slice boundary

    /// @brief Zero the team's slots and start the slice (master, before the region).
    void begin() noexcept;

    /// @brief One thread's report: region entry, loop start, loop end.
    void thread_done(Clock::time_point entry, Clock::time_point loop_start, Clock::time_point loop_end) noexcept;

    /// @brief Add this slice to @p times (master, out of the region).
    void accumulate(RechunkSliceTimes& times) const noexcept;

  private:
    [[nodiscard]] static long long ns(Clock::time_point from, Clock::time_point to) noexcept
    {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(to - from).count();
    }
    [[nodiscard]] static double to_ms(const std::atomic<long long>& slot) noexcept
    {
        return static_cast<double>(slot.load(std::memory_order_relaxed)) * 1e-6;
    }
    static void raise(std::atomic<long long>& slot, long long value) noexcept
    {
        slot.fetch_max(value, std::memory_order_relaxed);
    }
};

/**
 * @brief The phases of one rechunk_to(): their wall time (rechunk=debug), the
 *        RSS at their boundaries (memory=debug), and the lines that print them.
 *
 * @details One mark per phase boundary, in this order. A mark records the
 *          instant when the timings are on, then the RSS when the memory side
 *          is on and the mark is a boundary the "rss" line reports (each read
 *          walks the page tables, so not at every mark).
 */
class RechunkPhaseTrace
{
  public:
    /// @brief The phase boundaries of rechunk_to(), in order.
    enum class Mark : std::size_t
    {
        entry,        ///< the geometry change is committed
        flushed,      ///< this array's deferred writes drained (RSS only)
        before_tmp,   ///< prologue done, the fresh array about to be built
        refill_start, ///< tmp built and seeded
        refill_end,   ///< tmp filled
        swapped,      ///< storage and identity swapped, geometry restored
        torn_down,    ///< tmp (the old storage) destroyed
        drained,      ///< the old-size reuse list drained
        swept,        ///< the old geometry's empty slabs swept
        count
    };

    /// @brief Read the channels and reset the slice times (start of a re-chunk).
    RechunkPhaseTrace() noexcept
    {
        g_slice_times = {};
        rss_.fill(-1); // -1: not read (the channel is off)
        mark(Mark::entry);
    }

    /// @brief The memory side is on (memory=debug).
    [[nodiscard]] bool memory_on() const noexcept
    {
        return memory_;
    }

    /// @brief Record the boundary @p at.
    void mark(Mark at) noexcept;

    /// @brief Arm the teardown split just before tmp dies (teardown_probe.h).
    void arm_teardown() const noexcept
    {
        if (phases_)
        {
            teardown_probe().start();
        }
    }

    /// @brief tmp is destroyed: mark it, and charge what followed the
    ///        teardown probe's last step to "rest" (the remaining members and
    ///        the bases, or all of it for a variant that sets no mark).
    void tmp_destroyed() noexcept;

    /// @brief Print the phases, the teardown split, the slices and the RSS
    ///        lines of the re-chunk @p old_size -> @p new_size (those whose
    ///        channel is on).
    void log(unsigned int old_size, unsigned int new_size) const;

  private:
    [[nodiscard]] static constexpr std::size_t index(Mark at) noexcept
    {
        return static_cast<std::size_t>(at);
    }

    /// The tmp teardown phase, split (teardown_probe.h; the step names are
    /// teardown_step_names, in this order).
    void log_teardown_split(unsigned int old_size, unsigned int new_size) const;

    static constexpr std::size_t n_marks = static_cast<std::size_t>(Mark::count); ///< marks per re-chunk

    bool phases_ = rechunk_trace_on();                                  ///< rechunk=debug, read once
    bool memory_ = rechunk_trace_mem_on();                              ///< memory=debug, read once
    std::array<std::chrono::steady_clock::time_point, n_marks> time_{}; ///< the instants, by Mark
    std::array<long, n_marks>                                  rss_{};  ///< the RSS (MiB), by Mark; -1: not read
    double teardown_rest_ms_ = 0.0;                                     ///< tmp's teardown after the probe's last step
};
} // namespace shrink_array::internal

#include "chunk/internal/array_chunk_rechunk_trace_inline.h" // the long bodies
