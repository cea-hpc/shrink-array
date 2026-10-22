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

#include <algorithm>     // std::max, std::min, std::sort
#include <cstddef>       // size_t
#include <cstdint>       // uint32_t, uint64_t
#include <cstdio>        // std::fflush, stdout
#include <functional>    // std::hash
#include <limits>        // std::numeric_limits
#include <atomic>        // std::atomic, std::memory_order
#include <iostream>      // std::cout
#include <thread>        // std::thread::id, std::this_thread
#include <unordered_map> // std::unordered_map
#include <vector>        // std::vector
#include <iomanip>       // std::setw, std::hex

#include <string>   // std::string
#include <stdio.h>  // fileno (POSIX)
#include <unistd.h> // isatty

#include "chunk/config.h"              // build_config::use_openmp (get_tid)
#include "chunk/internal/enum_name.h" // upper_enum_name, table_matches (event_name)
#include "chunk/PMP_CHUNK.h"           // chunk_pmp_team_rank (get_tid)
#include "chunk/ring_log_debug.h"      // the declarations defined here
#include "chunk/log.h"                 // CHUNK_LOG

#ifdef CHUNK_RING_LOG_ANY

namespace shrink_array
{

namespace
{
constexpr size_t ring_size = 1uz << 24U;

std::vector<RingEntry> ring_buf(ring_size);

std::atomic<uint32_t> ring_head{0};

    #if defined(__aarch64__)

inline uint64_t now_cycles()
{
    uint64_t cnt;
    asm volatile("mrs %0, cntvct_el0" : "=r"(cnt));
    return cnt;
}
inline uint64_t rdtsc_start()
{
    return now_cycles();
}

inline uint64_t rdtsc_end()
{
    return now_cycles();
}

    #else
inline uint64_t now_cycles()
{
    unsigned int lo = 0;
    unsigned int hi = 0;
    __asm__ __volatile__("cpuid\n\t" // Serialize to avoid out-of-order execution
                         "rdtsc\n\t"
                         : "=a"(lo), "=d"(hi)
                         : "a"(0)
                         : "%ebx", "%ecx");
    return ((uint64_t)hi << 32U) | lo;
}

inline uint64_t rdtsc_start()
{
    unsigned int eax = 0;
    unsigned int edx = 0;
    __asm__ __volatile__("cpuid\n\t"
                         "rdtsc\n\t"
                         : "=a"(eax), "=d"(edx)
                         : "a"(0)
                         : "rbx", "rcx");
    return ((uint64_t)edx << 32U) | eax;
}

inline uint64_t rdtsc_end()
{
    unsigned int eax = 0;
    unsigned int edx = 0;
    __asm__ __volatile__("rdtscp\n\t" : "=a"(eax), "=d"(edx) : : "rcx");
    __asm__ __volatile__("cpuid" ::: "rax", "rbx", "rcx", "rdx");
    return ((uint64_t)edx << 32U) | eax;
}

    #endif
} // namespace

void ring_init()
{
    ring_head = 0;
}

void ring_log(uint32_t tid, EventCode ev, uint32_t info)
{
    const uint32_t idx = ring_head.fetch_add(1, std::memory_order_relaxed) & (ring_size - 1);
    ring_buf[idx].ts   = now_cycles();
    ring_buf[idx].tid  = tid;
    ring_buf[idx].ev   = ev;
    ring_buf[idx].info = info;
    ring_buf[idx].addr = 0;
}

// Record an op-lifecycle event with the op's full address.
void ring_log_op(uint32_t tid, EventCode ev, uint32_t num, const void* addr)
{
    const uint32_t idx = ring_head.fetch_add(1, std::memory_order_relaxed) & (ring_size - 1);
    ring_buf[idx].ts   = now_cycles();
    ring_buf[idx].tid  = tid;
    ring_buf[idx].ev   = ev;
    ring_buf[idx].info = num;
    ring_buf[idx].addr = reinterpret_cast<uint64_t>(addr);
}

void ring_log_start(uint32_t tid, EventCode ev, uint32_t info)
{
    const uint32_t idx = ring_head.fetch_add(1, std::memory_order_relaxed) & (ring_size - 1);
    ring_buf[idx].ts   = rdtsc_start();
    ring_buf[idx].tid  = tid;
    ring_buf[idx].ev   = ev;
    ring_buf[idx].info = info;
    ring_buf[idx].addr = 0;
}

void ring_log_end(uint32_t tid, EventCode ev, uint32_t info)
{
    const uint32_t idx = ring_head.fetch_add(1, std::memory_order_relaxed) & (ring_size - 1);
    ring_buf[idx].ts   = rdtsc_end();
    ring_buf[idx].tid  = tid;
    ring_buf[idx].ev   = ev;
    ring_buf[idx].info = info;
    ring_buf[idx].addr = 0;
}
const char* event_color(EventCode ev)
{
    switch (ev)
    {
        case ev_begin_slow_path:
            return "\033[36m"; // cyan
        case ev_end_slow_path:
            return "\033[32m"; // green
        case ev_begin_fast_path:
            return "\033[33m"; // yellow
        case ev_end_fast_path:
            return "\033[90m"; // grey
        case ev_compress_attempt:
            return "\033[35m"; // magenta
        case ev_compress_started:
            return "\033[34m"; // blue
        case ev_compress_aborted:
            return "\033[31m"; // red
        case ev_compress_finished:
            return "\033[92m"; // light green
        case ev_policy_begin_write:
            return "\033[96m"; // light cyan
        case ev_policy_end_write:
            return "\033[94m"; // light blue
        case ev_mark_thread_has_ops:
            return "\033[93m"; // light yellow
        case ev_clear_thread_has_ops:
            return "\033[95m"; // light magenta
        default:
            return "\033[0m"; // reset
    }
}

// The event's name: its enumerator without "ev_", upper-cased ("STORE_VALUE").
// The switch is the table of the builds without reflection (Clang, oneAPI):
// no `default:`, so -Wswitch reports an event added without its case. Where
// the compiler has reflection the name is derived from the enumerator, and the
// static_assert below keeps the switch complete and spelled alike.
constexpr const char* event_name_table(EventCode ev) noexcept
{
    switch (ev)
    {
        case ev_begin_slow_path:
            return "BEGIN_SLOW_PATH";
        case ev_end_slow_path:
            return "END_SLOW_PATH";
        case ev_begin_fast_path:
            return "BEGIN_FAST_PATH";
        case ev_end_fast_path:
            return "END_FAST_PATH";
        case ev_compress_attempt:
            return "COMPRESS_ATTEMPT";
        case ev_compress_started:
            return "COMPRESS_STARTED";
        case ev_compress_aborted:
            return "COMPRESS_ABORTED";
        case ev_compress_finished:
            return "COMPRESS_FINISHED";
        case ev_cas_values_to_cst:
            return "CAS_VALUES_TO_CST";
        case ev_cas_values_from_cst:
            return "CAS_VALUES_FROM_CST";
        case ev_cas_values_fail:
            return "CAS_VALUES_FAIL";
        case ev_flush_begin:
            return "FLUSH_BEGIN";
        case ev_flush_end:
            return "FLUSH_END";
        case ev_lambda_async_assign:
            return "LAMBDA_ASYNC_ASSIGN";
        case ev_lambda_async_arith:
            return "LAMBDA_ASYNC_ARITH";
        case ev_values_load_is_cst:
            return "VALUES_LOAD_IS_CST";
        case ev_store_value:
            return "STORE_VALUE";
        case ev_policy_begin_write:
            return "POLICY_BEGIN_WRITE";
        case ev_policy_end_write:
            return "POLICY_END_WRITE";
        case ev_mark_thread_has_ops:
            return "MARK_THREAD_HAS_OPS";
        case ev_clear_thread_has_ops:
            return "CLEAR_THREAD_HAS_OPS";
        case ev_op_alloc:
            return "OP_ALLOC";
        case ev_op_enqueue:
            return "OP_ENQUEUE";
        case ev_op_free:
            return "OP_FREE";
        case ev_op_flush_one:
            return "OP_FLUSH_ONE";
        case ev_value_alloc:
            return "VALUE_ALLOC";
        case ev_value_free:
            return "VALUE_FREE";
    }
    return "UNKNOWN";
}

#if SHRINK_ARRAY_HAS_ENUM_NAME
static_assert(table_matches<EventCode>(event_name_table, "ev_"),
              "event_name_table() disagrees with the EventCode enumerators: add or fix the case");

const char* event_name(EventCode ev)
{
    template for (constexpr std::meta::info en : std::define_static_array(std::meta::enumerators_of(^^EventCode)))
    {
        if (ev == [:en:])
        {
            constexpr const char* name = upper_enum_name(en, "ev_");
            return name;
        }
    }
    return "UNKNOWN";
}
#else
const char* event_name(EventCode ev)
{
    return event_name_table(ev);
}
#endif

void dump_ring()
{
    const uint32_t head  = ring_head.load(std::memory_order_relaxed);
    const uint32_t start = (head > ring_size) ? (head - ring_size) : 0;

    std::cout << "\033[1m---- ring dump (last " << (head - start) << " entries) ----\033[0m\n";
    for (uint32_t i = start; i < head; ++i)
    {
        const RingEntry& entry = ring_buf[i & (ring_size - 1)];
        const char*      color = event_color(entry.ev);
        const char*      name  = event_name(entry.ev);

        std::cout << "[" << i << "] tid=" << entry.tid << " ev=" << color << name << "\033[0m"
                  << " info=" << entry.info << " addr=0x" << std::hex << entry.addr << std::dec << " ts=" << entry.ts
                  << "\n";
    }
    std::cout << "\033[1m---- end ring ----\033[0m\n";
}

// Dump the events of one op address. Two passes:
//  (1) EXACT address match -- the lifecycle of THIS op (offset in slab).
//  (2) block-masked -- other ops sharing the 64 KiB slab, for context.
// The exact pass isolates the op's own alloc/flush/free ordering, which the
// block mask alone would merge with the slab's other ops.
void dump_ring_for_addr(const void* target)
{
    const uint64_t target_addr = reinterpret_cast<uint64_t>(target);
    const uint64_t bmask       = ~static_cast<uint64_t>(0xffffULL);
    const uint32_t head        = ring_head.load(std::memory_order_relaxed);
    const uint32_t start       = (head > ring_size) ? (head - ring_size) : 0;

    std::cout << "\033[1m---- ring dump: EXACT op addr 0x" << std::hex << target_addr << std::dec << " ----\033[0m\n";
    for (uint32_t i = start; i < head; ++i)
    {
        const RingEntry& entry = ring_buf[i & (ring_size - 1)];
        if (entry.addr != target_addr)
        {
            continue;
        }
        std::cout << "[" << i << "] tid=" << entry.tid << " ev=" << event_name(entry.ev) << " num=" << entry.info
                  << " addr=0x" << std::hex << entry.addr << std::dec << " ts=" << entry.ts << "\n";
    }
    std::cout << "\033[1m---- ring dump: block context 0x" << std::hex << (target_addr & bmask) << std::dec
              << " (other offsets in same slab) ----\033[0m\n";
    for (uint32_t i = start; i < head; ++i)
    {
        const RingEntry& entry = ring_buf[i & (ring_size - 1)];
        if (entry.addr == 0)
        {
            continue;
        }
        if ((entry.addr & bmask) != (target_addr & bmask))
        {
            continue;
        }
        if (entry.addr == target_addr)
        {
            continue; // already shown exact
        }
        std::cout << "[" << i << "] tid=" << entry.tid << " ev=" << event_name(entry.ev) << " num=" << entry.info
                  << " addr=0x" << std::hex << entry.addr << std::dec << " ts=" << entry.ts << "\n";
    }
    std::cout << "\033[1m---- end ring for addr ----\033[0m\n";
    std::cout.flush(); // abort() does NOT flush cout -- force it now
    std::fflush(stdout);
}

uint32_t get_tid()
{
    if constexpr (build_config::use_openmp)
    {
        return static_cast<uint32_t>(chunk_pmp_team_rank());
    }
    else
    {
        const auto hasher = std::hash<std::thread::id>{};
        const auto raw    = hasher(std::this_thread::get_id());
        return static_cast<uint32_t>(raw & 0xFFFFFFFFu);
    }
}

namespace
{
struct Interval
{
    uint64_t start;
    uint64_t end;
    bool     is_fallback = false;
};

// Event types for sweep line; order matters for tie-breaking
enum EventType : std::uint8_t
{
    comp_start = 0,
    fast_start = 1,
    fast_end   = 2,
    comp_end   = 3
};

struct Endpoint
{
    uint64_t  time;
    EventType type;
    uint32_t  tid; // tid owning this interval
    size_t    idx; // index of interval in the vector for that tid
};

bool is_stdout_tty()
{
    // Asked once: a function-local static is initialised thread-safely.
    static const bool cached = isatty(fileno(stdout)) != 0;
    return cached;
}

// Change this to control how often to log in non-TTY mode (percent steps)
constexpr int progress_percent_step = 5;

// Utility: print progress line (overwrites using '\r' when stdout is a TTY)
// In non-TTY mode, emit full lines only when integer percent crosses progress_percent_step or at completion
void print_progress_line(const std::string& icon,
                         const std::string& stage,
                         size_t             done,
                         size_t             total,
                         size_t             spinner_idx)
{
    const char   spinner_chars[] = {'|', '/', '-', '\\'};
    const double pct_double = total != 0 ? (100.0 * static_cast<double>(done) / static_cast<double>(total)) : 100.0;
    const int    pct        = static_cast<int>(pct_double);
    const char   spinner    = spinner_chars[spinner_idx % 4];

    if (is_stdout_tty())
    {
        // Overwrite the current line in interactive terminals
        std::cout << "\r" << icon << " " << stage << " " << std::setw(3) << pct << "% [" << spinner << "]"
                  << std::flush;
    }
    else
    {
        // In non-interactive output (e.g. redirected to file), avoid flooding logs:
        // print only every progress_percent_step percent or at completion
        static int last_logged_pct = -1;
        bool       should_log      = false;

        if (done == total)
        {
            should_log = true;
        }
        else if (progress_percent_step > 0)
        {
            // Log when crossing multiples of progress_percent_step
            const int bucket = (pct / progress_percent_step) * progress_percent_step;
            if (bucket != last_logged_pct)
            {
                last_logged_pct = bucket;
                should_log      = true;
            }
        }
        else
        {
            // fallback: log on percent change
            if (pct != last_logged_pct)
            {
                last_logged_pct = pct;
                should_log      = true;
            }
        }

        if (should_log)
        {
            std::cout << icon << " " << stage << " " << std::setw(3) << pct << "% [" << spinner << "]\n" << std::flush;
        }
    }
}
} // namespace

// Analyze the ring buffer and report overlaps where one thread is in fast-path
// while another thread is performing compression using a sweep-line algorithm.

bool analyze_ring_sweep()
{
    const uint32_t head          = ring_head.load(std::memory_order_relaxed);
    const uint32_t start         = (head > ring_size) ? (head - ring_size) : 0;
    const uint64_t total_entries = (head > start) ? (head - start) : 0;

    // Stage 1: scan ring and build intervals
    std::cout << "🔍 Scanning ring buffer and building intervals...\n";
    std::unordered_map<uint32_t, std::vector<Interval>> fast_intervals;
    std::unordered_map<uint32_t, std::vector<Interval>> compress_intervals;

    struct ThreadState
    {
        bool     in_fast        = false;
        uint64_t fast_start     = 0;
        bool     in_compress    = false;
        uint64_t compress_start = 0;
    };
    std::unordered_map<uint32_t, ThreadState> state;

    uint64_t     last_ts       = 0;
    const size_t progress_step = std::max<size_t>(1, total_entries / 100); // update ~100 times
    size_t       seen          = 0;
    size_t       spinner       = 0;

    for (uint32_t i = start; i < head; ++i)
    {
        const RingEntry& entry = ring_buf[i & (ring_size - 1)];
        last_ts                = std::max(last_ts, entry.ts);

        ThreadState& thr_state = state[entry.tid];

        switch (entry.ev)
        {
            case ev_begin_fast_path:
                if (!thr_state.in_fast)
                {
                    thr_state.in_fast    = true;
                    thr_state.fast_start = entry.ts;
                }
                break;

            case ev_end_fast_path:
                if (thr_state.in_fast)
                {
                    fast_intervals[entry.tid].push_back({thr_state.fast_start, entry.ts});
                    thr_state.in_fast = false;
                }
                break;

            case ev_compress_attempt:
            case ev_compress_started:
                if (!thr_state.in_compress)
                {
                    thr_state.in_compress    = true;
                    thr_state.compress_start = entry.ts;
                }
                break;

            case ev_compress_finished:
            case ev_compress_aborted:
                if (thr_state.in_compress)
                {
                    compress_intervals[entry.tid].push_back({thr_state.compress_start, entry.ts});
                    thr_state.in_compress = false;
                }
                break;

            default:
                break;
        }

        ++seen;
        if ((seen % progress_step) == 0 || seen == total_entries)
        {
            print_progress_line("🔍", "Scanning", seen, total_entries, spinner++);
        }
    }
    // End scanning progress line
    std::cout << "\r✅ Scanning complete.                            \n";
    // Close open intervals conservatively at last_ts (robust, with minimal-duration filter)
    const auto push_fallback_if_needed =
        [&](std::unordered_map<uint32_t, std::vector<Interval>>& map, uint32_t tid, uint64_t start_ts, uint64_t end_ts)
    {
        if (end_ts <= start_ts)
        {
            return; // invalid
        }
        const uint64_t     dur             = end_ts - start_ts;
        constexpr uint64_t min_interval_ns = 1000;
        if (dur < min_interval_ns)
        {
            return; // ignore too-short fallback
        }
        auto& vec = map[tid];
        if (!vec.empty())
        {
            const Interval& last = vec.back();
            // Same start, or already covered: assume already recorded.
            if (last.start == start_ts || end_ts <= last.end || start_ts >= last.end)
            {
                return;
            }
        }
        vec.push_back({start_ts, end_ts, true});
    };

    for (auto& p : state)
    {
        const uint32_t tid       = p.first;
        ThreadState&   thr_state = p.second;

        if (thr_state.in_fast)
        {
            // only push if start < last_ts and duration large enough
            if (thr_state.fast_start < last_ts)
            {
                push_fallback_if_needed(fast_intervals, tid, thr_state.fast_start, last_ts);
            }
            thr_state.in_fast = false;
        }

        if (thr_state.in_compress)
        {
            if (thr_state.compress_start < last_ts)
            {
                push_fallback_if_needed(compress_intervals, tid, thr_state.compress_start, last_ts);
            }
            thr_state.in_compress = false;
        }
    }

    // Stage 2: build endpoints
    std::cout << "📦 Building endpoints for sweep-line...\n";
    std::vector<Endpoint> endpoints;
    endpoints.reserve((fast_intervals.size() + compress_intervals.size()) * 2);

    const auto add_endpoints = [&](const std::unordered_map<uint32_t, std::vector<Interval>>& map, bool is_compress)
    {
        for (const auto& p : map)
        {
            const uint32_t tid = p.first;
            const auto&    vec = p.second;
            for (size_t i = 0; i < vec.size(); ++i)
            {
                const Interval& iv = vec[i];
                if (iv.is_fallback)
                {
                    continue; // ⛔ Ignore fallback intervals
                }
                if (is_compress)
                {
                    endpoints.push_back({iv.start, comp_start, tid, i});
                    endpoints.push_back({iv.end, comp_end, tid, i});
                }
                else
                {
                    endpoints.push_back({iv.start, fast_start, tid, i});
                    endpoints.push_back({iv.end, fast_end, tid, i});
                }
            }
        }
    };

    add_endpoints(compress_intervals, true);
    add_endpoints(fast_intervals, false);

    if (endpoints.empty())
    {
        std::cout << "⚠️ No intervals to analyze.\n";
        return false;
    }

    // Stage 3: sort endpoints
    std::cout << "🔢 Sorting endpoints (" << endpoints.size() << " endpoints)...\n";
    std::sort(endpoints.begin(), endpoints.end(),
              [](const Endpoint& lhs, const Endpoint& rhs)
              {
                  if (lhs.time != rhs.time)
                  {
                      return lhs.time < rhs.time;
                  }
                  return lhs.type < rhs.type;
              });
    std::cout << "✅ Sorting complete.\n";

    // Stage 4: sweep
    std::cout << "⚙️  Sweeping endpoints and detecting overlaps...\n";
    struct ActiveCompress
    {
        uint32_t tid;
        size_t   idx;
        uint64_t start;
        uint64_t end;
    };
    std::vector<ActiveCompress> active_list;
    active_list.reserve(64);
    std::unordered_map<uint64_t, size_t> active_index_by_key;

    const auto make_key = [](uint32_t tid, size_t idx) -> uint64_t { return ((uint64_t)tid << 32U) | (uint64_t)idx; };

    struct Overlap
    {
        uint32_t fast_tid;
        size_t   fast_idx;
        uint32_t comp_tid;
        size_t   comp_idx;
        uint64_t start;
        uint64_t end;
    };

    std::vector<Overlap> overlaps;
    overlaps.reserve(256);

    // Show sweep progress every chunk of endpoints processed
    const size_t sweep_total   = endpoints.size();
    const size_t sweep_step    = std::max<size_t>(1, sweep_total / 100);
    size_t       sweep_seen    = 0;
    size_t       sweep_spinner = 0;

    for (const Endpoint& ep : endpoints)
    {
        switch (ep.type)
        {
            case comp_start:
            {
                if (ep.idx >= compress_intervals[ep.tid].size())
                {
                    CHUNK_LOG(warn, general, "ring log: invalid compress index {} for TID {}", ep.idx, ep.tid);
                    continue;
                }
                const Interval&      iv = compress_intervals[ep.tid][ep.idx];
                const ActiveCompress ac{ep.tid, ep.idx, iv.start, iv.end};
                const uint64_t       key = make_key(ep.tid, ep.idx);
                active_index_by_key[key] = active_list.size();
                active_list.push_back(ac);
                break;
            }

            case comp_end:
            {
                const uint64_t key = make_key(ep.tid, ep.idx);
                const auto     it  = active_index_by_key.find(key);
                if (it != active_index_by_key.end())
                {
                    const size_t pos = it->second;
                    if (pos + 1 != active_list.size())
                    {
                        active_list[pos]               = active_list.back();
                        const uint64_t moved_key       = make_key(active_list[pos].tid, active_list[pos].idx);
                        active_index_by_key[moved_key] = pos;
                    }
                    active_list.pop_back();
                    active_index_by_key.erase(it);
                }
                break;
            }

            case fast_start:
            {
                if (ep.idx >= fast_intervals[ep.tid].size())
                {
                    CHUNK_LOG(warn, general, "ring log: invalid fast index {} for TID {}", ep.idx, ep.tid);
                    continue;
                }
                const Interval& fiv = fast_intervals[ep.tid][ep.idx];
                for (const ActiveCompress& ac : active_list)
                {
                    if (ac.tid == ep.tid)
                    {
                        continue;
                    }
                    const uint64_t ov_start = std::max(fiv.start, ac.start);
                    const uint64_t ov_end   = std::min(fiv.end, ac.end);
                    if (ov_start < ov_end)
                    {
                        overlaps.push_back({ep.tid, ep.idx, ac.tid, ac.idx, ov_start, ov_end});
                    }
                }
                break;
            }

            case fast_end:
                // nothing to do; overlaps already recorded at fast_start
                break;
        }

        ++sweep_seen;
        if ((sweep_seen % sweep_step) == 0 || sweep_seen == sweep_total)
        {
            print_progress_line("⚙️", "Sweeping", sweep_seen, sweep_total, sweep_spinner++);
        }
    }
    std::cout << "\r✅ Sweeping complete.                              \n";

    // Stage 5: report
    if (overlaps.empty())
    {
        std::cout << "✅ No fast-path vs compression overlap detected in the analyzed window.\n";
        return false;
    }
    uint64_t min_start = std::numeric_limits<uint64_t>::max();

    for (const auto& o : overlaps)
    {
        const Interval& fiv = fast_intervals[o.fast_tid][o.fast_idx];
        const Interval& civ = compress_intervals[o.comp_tid][o.comp_idx];

        min_start = std::min(min_start, fiv.start);
        min_start = std::min(min_start, civ.start);
    }

    std::cout << "⚠️ Detected overlaps: " << overlaps.size() << "\n";
    std::cout << "Timestamps relative to min_start = " << min_start << "\n";

    // Long fast-path durations are not a correctness signal (cold-path costs: page faults,
    // NUMA, pool contention). The only invariant violation is a same-thread fast-path AND
    // compression overlap: one thread in two states at once.

    size_t n_same_thread = 0;
    for (const auto& o : overlaps)
    {
        if (o.fast_tid == o.comp_tid)
        {
            ++n_same_thread;
        }
    }

    const bool serious = (n_same_thread > 0);
    if (serious)
    {
        std::cout << "❌ SERIOUS signals detected:\n";
        std::cout << "   - same-thread fast/compress overlaps : " << n_same_thread
                  << "  (state-machine inconsistency)\n";
    }

    return serious;
}

} // namespace shrink_array

#endif
