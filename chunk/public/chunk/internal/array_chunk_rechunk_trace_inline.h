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

// Out-of-line bodies of the re-chunk's traces (array_chunk_rechunk_trace.h).

#include "chunk/internal/array_chunk_rechunk_trace.h" // the classes these bodies belong to

namespace shrink_array::internal
{

// ---------------------------------------------------------------------------
// shrink_array::internal::SliceProbe
// ---------------------------------------------------------------------------

inline void SliceProbe::begin() noexcept
{
    entry_max.store(0, std::memory_order_relaxed);
    loop_end_max.store(0, std::memory_order_relaxed);
    work_max.store(0, std::memory_order_relaxed);
    work_sum.store(0, std::memory_order_relaxed);
    setup_max.store(0, std::memory_order_relaxed);
    n.store(0, std::memory_order_relaxed);
    start = Clock::now();
}

inline void
SliceProbe::thread_done(Clock::time_point entry, Clock::time_point loop_start, Clock::time_point loop_end) noexcept
{
    raise(entry_max, ns(start, entry));
    raise(loop_end_max, ns(start, loop_end));
    raise(work_max, ns(loop_start, loop_end));
    raise(setup_max, ns(entry, loop_start));
    work_sum.fetch_add(ns(loop_start, loop_end), std::memory_order_relaxed);
    n.fetch_add(1, std::memory_order_relaxed);
}

inline void SliceProbe::accumulate(RechunkSliceTimes& times) const noexcept
{
    const auto   post  = Clock::now();
    const double e_ms  = to_ms(entry_max);
    const double l_ms  = to_ms(loop_end_max);
    const int    n_thr = n.load(std::memory_order_relaxed) > 0 ? n.load(std::memory_order_relaxed) : 1;
    times.fork_ms += e_ms;
    times.loop_ms += l_ms - e_ms;
    times.barrier_ms += elapsed_ms(start, past_barrier) - l_ms;
    times.boundary_ms += elapsed_ms(past_barrier, past_boundary);
    times.join_ms += elapsed_ms(past_boundary, post);
    times.work_max_ms += to_ms(work_max);
    times.work_avg_ms += to_ms(work_sum) / n_thr;
    times.setup_max_ms += to_ms(setup_max);
}

// ---------------------------------------------------------------------------
// shrink_array::internal::RechunkPhaseTrace
// ---------------------------------------------------------------------------

inline void RechunkPhaseTrace::mark(Mark at) noexcept
{
    if (phases_)
    {
        time_[index(at)] = std::chrono::steady_clock::now();
    }
    if (memory_ && at != Mark::refill_start && at != Mark::drained)
    {
        rss_[index(at)] = rss_mib_now();
    }
}

inline void RechunkPhaseTrace::tmp_destroyed() noexcept
{
    if (phases_)
    {
        const auto now                = std::chrono::steady_clock::now();
        time_[index(Mark::torn_down)] = now;
        teardown_rest_ms_             = elapsed_ms(teardown_probe().stop(), now);
    }
    if (memory_)
    {
        rss_[index(Mark::torn_down)] = rss_mib_now();
    }
}

inline void RechunkPhaseTrace::log(unsigned int old_size, unsigned int new_size) const
{
    if (phases_)
    {
        const auto ms = [&](Mark from, Mark to) { return elapsed_ms(time_[index(from)], time_[index(to)]); };
        CHUNK_LOG(debug, rechunk,
                  "{} -> {} phases (ms): prologue {:.0f} | tmp create {:.0f} | refill {:.0f} | of which slice "
                  "drain {:.0f} | of which slice collect {:.0f} | flush+swap {:.0f} | tmp teardown {:.0f} | "
                  "old-pool drain {:.0f} | slab sweep {:.0f} | total {:.0f}",
                  old_size, new_size, ms(Mark::entry, Mark::before_tmp), ms(Mark::before_tmp, Mark::refill_start),
                  ms(Mark::refill_start, Mark::refill_end), g_slice_times.drain_ms, g_slice_times.collect_ms,
                  ms(Mark::refill_end, Mark::swapped), ms(Mark::swapped, Mark::torn_down),
                  ms(Mark::torn_down, Mark::drained), ms(Mark::drained, Mark::swept), ms(Mark::entry, Mark::swept));
        log_teardown_split(old_size, new_size);
        const RechunkSliceTimes& st = g_slice_times;
        if (st.count > 0)
        {
            CHUNK_LOG(debug, rechunk,
                      "{} -> {} slices {} (ms): fork {:.1f} | loop {:.1f} | work max {:.1f} | work avg {:.1f} | "
                      "setup max {:.1f} | barrier {:.1f} | boundary {:.1f} | join {:.1f}",
                      old_size, new_size, st.count, st.fork_ms, st.loop_ms, st.work_max_ms, st.work_avg_ms,
                      st.setup_max_ms, st.barrier_ms, st.boundary_ms, st.join_ms);
        }
    }
    if (memory_)
    {
        const auto rss = [&](Mark at) { return rss_[index(at)]; };
        CHUNK_LOG(debug, memory,
                  "rss {} -> {} (MB): entry {} | after flush {} | before tmp {} | after refill {} | after swap {} "
                  "| after teardown {} | after sweep {}   (end-entry {:+})",
                  old_size, new_size, rss(Mark::entry), rss(Mark::flushed), rss(Mark::before_tmp),
                  rss(Mark::refill_end), rss(Mark::swapped), rss(Mark::torn_down), rss(Mark::swept),
                  rss(Mark::swept) - rss(Mark::entry));
    }
}

inline void RechunkPhaseTrace::log_teardown_split(unsigned int old_size, unsigned int new_size) const
{
    const auto& td = teardown_probe().ms;
    const auto  at = [&](TeardownStep step) { return td[static_cast<std::size_t>(step)]; };
    CHUNK_LOG(debug, rechunk,
              "{} -> {} teardown split (ms): entry {:.1f} | flush {:.1f} | candidates {:.1f} | terminal {:.1f} | "
              "park {:.1f} | registry {:.1f} | tables {:.1f} | key drain {:.1f} | erase {:.1f} | list unmap "
              "{:.1f} | rest {:.1f}",
              old_size, new_size, at(TeardownStep::entry), at(TeardownStep::flush), at(TeardownStep::candidates),
              at(TeardownStep::terminal), at(TeardownStep::park), at(TeardownStep::registry), at(TeardownStep::tables),
              at(TeardownStep::key_drain), at(TeardownStep::erase), at(TeardownStep::list_unmap), teardown_rest_ms_);
}

} // namespace shrink_array::internal
