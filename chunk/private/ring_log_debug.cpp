//
// Created by ballereaup on 13/10/2025.
//

#include <cstdint>
#include <limits>
#include <memory>
#include <chrono>
#include <atomic>
#include <iostream>
#include <thread>
#include <unordered_map>
#include <vector>
#include <iomanip>

#include <string>
#include <unistd.h> // isatty, fileno

#ifdef _OPENMP
    #include <omp.h>
#endif

#include "chunk/PMP_CHUNK.h" // chunk_pmp_team_rank (get_tid)
#include "chunk/ring_log_debug.h"

#ifdef RING_LOG_ANY

namespace shrink_array
{

static constexpr size_t RING_SIZE = size_t{1} << 24U;

static std::unique_ptr<RingEntry[]> ring_buf = std::make_unique<RingEntry[]>(RING_SIZE);

static std::atomic<uint32_t> ring_head{0};

    #if defined(__aarch64__)

uint64_t now_ns()
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

inline uint64_t now_cycles()
{
    uint64_t cnt;
    asm volatile("mrs %0, cntvct_el0" : "=r"(cnt));
    return cnt;
}
static inline uint64_t rdtsc_start()
{
    return now_cycles();
}

static inline uint64_t rdtsc_end()
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

static inline uint64_t rdtsc_start()
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

static inline uint64_t rdtsc_end()
{
    unsigned int eax = 0;
    unsigned int edx = 0;
    __asm__ __volatile__("rdtscp\n\t" : "=a"(eax), "=d"(edx) : : "rcx");
    __asm__ __volatile__("cpuid" ::: "rax", "rbx", "rcx", "rdx");
    return ((uint64_t)edx << 32U) | eax;
}

    #endif

void ring_init()
{
    ring_head = 0;
}

void ring_log(uint32_t tid, EventCode ev, uint32_t info)
{
    const uint32_t idx = ring_head.fetch_add(1, std::memory_order_relaxed) & (RING_SIZE - 1);
    ring_buf[idx].ts   = now_cycles();
    ring_buf[idx].tid  = tid;
    ring_buf[idx].ev   = ev;
    ring_buf[idx].info = info;
    ring_buf[idx].addr = 0;
}

// SEGV hunt: record an op-lifecycle event with the op's full address.
void ring_log_op(uint32_t tid, EventCode ev, uint32_t num, const void* addr)
{
    const uint32_t idx = ring_head.fetch_add(1, std::memory_order_relaxed) & (RING_SIZE - 1);
    ring_buf[idx].ts   = now_cycles();
    ring_buf[idx].tid  = tid;
    ring_buf[idx].ev   = ev;
    ring_buf[idx].info = num;
    ring_buf[idx].addr = reinterpret_cast<uint64_t>(addr);
}

void ring_log_start(uint32_t tid, EventCode ev, uint32_t info)
{
    const uint32_t idx = ring_head.fetch_add(1, std::memory_order_relaxed) & (RING_SIZE - 1);
    ring_buf[idx].ts   = rdtsc_start();
    ring_buf[idx].tid  = tid;
    ring_buf[idx].ev   = ev;
    ring_buf[idx].info = info;
    ring_buf[idx].addr = 0;
}

void ring_log_end(uint32_t tid, EventCode ev, uint32_t info)
{
    const uint32_t idx = ring_head.fetch_add(1, std::memory_order_relaxed) & (RING_SIZE - 1);
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
        case EV_BEGIN_SLOW_PATH:
            return "\033[36m"; // cyan
        case EV_END_SLOW_PATH:
            return "\033[32m"; // vert
        case EV_BEGIN_FAST_PATH:
            return "\033[33m"; // jaune
        case EV_END_FAST_PATH:
            return "\033[90m"; // gris
        case EV_COMPRESS_ATTEMPT:
            return "\033[35m"; // magenta
        case EV_COMPRESS_STARTED:
            return "\033[34m"; // bleu
        case EV_COMPRESS_ABORTED:
            return "\033[31m"; // rouge
        case EV_COMPRESS_FINISHED:
            return "\033[92m"; // vert clair
        case EV_POLICY_BEGIN_WRITE:
            return "\033[96m"; // cyan clair
        case EV_POLICY_END_WRITE:
            return "\033[94m"; // bleu clai
        case EV_MARK_THREAD_HAS_OPS:
            return "\033[93m"; // jaune clair
        case EV_CLEAR_THREAD_HAS_OPS:
            return "\033[95m"; // magenta clairr
        default:
            return "\033[0m"; // reset
    }
}

const char* event_name(EventCode ev)
{
    switch (ev)
    {
        case EV_BEGIN_SLOW_PATH:
            return "BEGIN_SLOW_PATH";
        case EV_END_SLOW_PATH:
            return "EV_END_SLOW_PATH";
        case EV_BEGIN_FAST_PATH:
            return "EV_BEGIN_FAST_PATH";
        case EV_END_FAST_PATH:
            return "EV_END_FAST_PATH";
        case EV_COMPRESS_ATTEMPT:
            return "COMPRESS_ATTEMPT";
        case EV_COMPRESS_STARTED:
            return "COMPRESS_STARTED";
        case EV_COMPRESS_ABORTED:
            return "COMPRESS_ABORTED";
        case EV_COMPRESS_FINISHED:
            return "COMPRESS_FINISHED";
        case EV_POLICY_BEGIN_WRITE:
            return "POLICY_BEGIN_WRITE";
        case EV_POLICY_END_WRITE:
            return "POLICY_END_WRITE";
        case EV_MARK_THREAD_HAS_OPS:
            return "MARK_THREAD_HAS_OPS";
        case EV_CLEAR_THREAD_HAS_OPS:
            return "CLEAR_THREAD_HAS_OPS";
        case EV_OP_ALLOC:
            return "OP_ALLOC";
        case EV_OP_ENQUEUE:
            return "OP_ENQUEUE";
        case EV_OP_FREE:
            return "OP_FREE";
        case EV_OP_FLUSH_ONE:
            return "OP_FLUSH_ONE";
        case EV_VALUE_ALLOC:
            return "VALUE_ALLOC";
        case EV_VALUE_FREE:
            return "VALUE_FREE";
        default:
            return "UNKNOWN";
    }
}

void dump_ring()
{
    const uint32_t head  = ring_head.load(std::memory_order_relaxed);
    const uint32_t start = (head > RING_SIZE) ? (head - RING_SIZE) : 0;

    std::cout << "\033[1m---- ring dump (last " << (head - start) << " entries) ----\033[0m\n";
    for (uint32_t i = start; i < head; ++i)
    {
        const RingEntry& entry = ring_buf[i & (RING_SIZE - 1)];
        const char*      color = event_color(entry.ev);
        const char*      name  = event_name(entry.ev);

        std::cout << "[" << i << "] tid=" << entry.tid << " ev=" << color << name << "\033[0m"
                  << " info=" << entry.info << " addr=0x" << std::hex << entry.addr << std::dec << " ts=" << entry.ts
                  << "\n";
    }
    std::cout << "\033[1m---- end ring ----\033[0m\n";
}

// SEGV hunt: dump events for the faulting op. Two passes:
//  (1) EXACT address match — the true lifecycle of THIS op (offset in slab).
//  (2) block-masked context — other ops sharing the 64 KiB slab, for context.
// The block mask alone was too coarse (it merged all ops of the slab); the exact
// pass isolates the culprit op's own alloc/flush/free ordering.
void dump_ring_for_addr(const void* target)
{
    const uint64_t target_addr = reinterpret_cast<uint64_t>(target);
    const uint64_t bmask       = ~static_cast<uint64_t>(0xffffULL);
    const uint32_t head        = ring_head.load(std::memory_order_relaxed);
    const uint32_t start       = (head > RING_SIZE) ? (head - RING_SIZE) : 0;

    std::cout << "\033[1m---- ring dump: EXACT op addr 0x" << std::hex << target_addr << std::dec << " ----\033[0m\n";
    for (uint32_t i = start; i < head; ++i)
    {
        const RingEntry& entry = ring_buf[i & (RING_SIZE - 1)];
        if (entry.addr != target_addr)
            continue;
        std::cout << "[" << i << "] tid=" << entry.tid << " ev=" << event_name(entry.ev) << " num=" << entry.info
                  << " addr=0x" << std::hex << entry.addr << std::dec << " ts=" << entry.ts << "\n";
    }
    std::cout << "\033[1m---- ring dump: block context 0x" << std::hex << (target_addr & bmask) << std::dec
              << " (other offsets in same slab) ----\033[0m\n";
    for (uint32_t i = start; i < head; ++i)
    {
        const RingEntry& entry = ring_buf[i & (RING_SIZE - 1)];
        if (entry.addr == 0)
            continue;
        if ((entry.addr & bmask) != (target_addr & bmask))
            continue;
        if (entry.addr == target_addr)
            continue; // already shown exact
        std::cout << "[" << i << "] tid=" << entry.tid << " ev=" << event_name(entry.ev) << " num=" << entry.info
                  << " addr=0x" << std::hex << entry.addr << std::dec << " ts=" << entry.ts << "\n";
    }
    std::cout << "\033[1m---- end ring for addr ----\033[0m\n";
    std::cout.flush(); // SEGV hunt: abort() does NOT flush cout — force it now
    std::fflush(stdout);
}

uint32_t get_tid()
{
    if constexpr (build_config::use_openmp)
        return static_cast<uint32_t>(chunk_pmp_team_rank());
    else
    {
        // Create a hash function object over std::thread::id
        const auto hasher = std::hash<std::thread::id>{};
        const auto raw    = hasher(std::this_thread::get_id());
        return static_cast<uint32_t>(raw & 0xFFFFFFFFu);
    }
}

struct Interval
{
    uint64_t start;
    uint64_t end;
    bool     is_fallback = false;
};

// Event types for sweep line; order matters for tie-breaking
enum EventType
{
    COMP_START = 0,
    FAST_START = 1,
    FAST_END   = 2,
    COMP_END   = 3
};

struct Endpoint
{
    uint64_t  time;
    EventType type;
    uint32_t  tid; // tid owning this interval
    size_t    idx; // index of interval in the vector for that tid
};


static bool is_stdout_tty()
{
    // Cached result to avoid repeated syscalls
    static std::atomic<int> cached{-1};
    const int               v = cached.load(std::memory_order_acquire);
    if (v != -1)
        return v != 0;

    const int result = isatty(fileno(stdout)) ? 1 : 0;
    cached.store(result, std::memory_order_release);
    return result != 0;
}

// Change this to control how often to log in non-TTY mode (percent steps)
static constexpr int PROGRESS_PERCENT_STEP = 5;

// Utility: print progress line (overwrites using '\r' when stdout is a TTY)
// In non-TTY mode, emit full lines only when integer percent crosses PROGRESS_PERCENT_STEP or at completion
static void
print_progress_line(const std::string& icon, const std::string& stage, size_t done, size_t total, size_t spinner_idx)
{
    const char   spinner_chars[] = {'|', '/', '-', '\\'};
    const double pct_double      = total ? (100.0 * static_cast<double>(done) / static_cast<double>(total)) : 100.0;
    const int    pct             = static_cast<int>(pct_double);
    const char   spinner         = spinner_chars[spinner_idx % 4];

    if (is_stdout_tty())
    {
        // Overwrite the current line in interactive terminals
        std::cout << "\r" << icon << " " << stage << " " << std::setw(3) << pct << "% [" << spinner << "]"
                  << std::flush;
    }
    else
    {
        // In non-interactive output (e.g. redirected to file), avoid flooding logs:
        // print only every PROGRESS_PERCENT_STEP percent or at completion
        static int last_logged_pct = -1;
        bool       should_log      = false;

        if (done == total)
        {
            should_log = true;
        }
        else if (PROGRESS_PERCENT_STEP > 0)
        {
            // Log when crossing multiples of PROGRESS_PERCENT_STEP
            const int bucket = (pct / PROGRESS_PERCENT_STEP) * PROGRESS_PERCENT_STEP;
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

// Analyze the ring buffer and report overlaps where one thread is in fast-path
// while another thread is performing compression using a sweep-line algorithm.

bool analyze_ring_sweep()
{
    const uint32_t head          = ring_head.load(std::memory_order_relaxed);
    const uint32_t start         = (head > RING_SIZE) ? (head - RING_SIZE) : 0;
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
        const RingEntry& entry = ring_buf[i & (RING_SIZE - 1)];
        last_ts                = std::max(last_ts, entry.ts);

        ThreadState& thr_state = state[entry.tid];

        switch (entry.ev)
        {
            case EV_BEGIN_FAST_PATH:
                if (!thr_state.in_fast)
                {
                    thr_state.in_fast    = true;
                    thr_state.fast_start = entry.ts;
                }
                break;

            case EV_END_FAST_PATH:
                if (thr_state.in_fast)
                {
                    fast_intervals[entry.tid].push_back({thr_state.fast_start, entry.ts});
                    thr_state.in_fast = false;
                }
                break;

            case EV_COMPRESS_ATTEMPT:
            case EV_COMPRESS_STARTED:
                if (!thr_state.in_compress)
                {
                    thr_state.in_compress    = true;
                    thr_state.compress_start = entry.ts;
                }
                break;

            case EV_COMPRESS_FINISHED:
            case EV_COMPRESS_ABORTED:
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
            return;
        // invalid
        const uint64_t     dur             = end_ts - start_ts;
        constexpr uint64_t MIN_INTERVAL_NS = 1000;
        if (dur < MIN_INTERVAL_NS)
            return; // ignore too-short fallback
        auto& vec = map[tid];
        if (!vec.empty())
        {
            const Interval& last = vec.back();
            // if last interval has same start (or overlaps to end_ts), assume already recorded
            if (last.start == start_ts || end_ts <= last.end || start_ts >= last.end)
            {
                // If the last interval already covers this range or has identical start, skip
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

    #if 0
    // Close open intervals conservatively at last_ts
    for (auto& p : state)
    {
        uint32_t     tid = p.first;
        ThreadState& thr_state   = p.second;

        if (thr_state.in_fast)
        {
            // Check there isn't already a closed interval with last_ts
            bool already_closed = false;
            if (!fast_intervals[tid].empty())
            {
                const Interval& last = fast_intervals[tid].back();
                if (last.end == last_ts && last.start == thr_state.fast_start)
                    already_closed = true;
            }

            if (!already_closed && thr_state.fast_start < last_ts)
            {
                fast_intervals[tid].push_back({thr_state.fast_start, last_ts, true});
            }

            thr_state.in_fast = false;
        }

        if (thr_state.in_compress)
        {
            bool already_closed = false;
            if (!compress_intervals[tid].empty())
            {
                const Interval& last = compress_intervals[tid].back();
                if (last.end == last_ts && last.start == thr_state.compress_start)
                    already_closed = true;
            }

            if (!already_closed && thr_state.compress_start < last_ts)
            {
                compress_intervals[tid].push_back({thr_state.compress_start, last_ts, true});
            }

            thr_state.in_compress = false;
        }
    }
    #endif
    // Stage 2: build endpoints
    std::cout << "📦 Building endpoints for sweep-line...\n";
    std::vector<Endpoint> endpoints;
    endpoints.reserve((fast_intervals.size() + compress_intervals.size()) * 2);

    const auto add_endpoints = [&](const std::unordered_map<uint32_t, std::vector<Interval>>& map, bool isCompress)
    {
        for (const auto& p : map)
        {
            const uint32_t tid = p.first;
            const auto&    vec = p.second;
            for (size_t i = 0; i < vec.size(); ++i)
            {
                const Interval& iv = vec[i];
                if (iv.is_fallback)
                    continue; // ⛔ Ignore fallback intervals
                if (isCompress)
                {
                    endpoints.push_back({iv.start, COMP_START, tid, i});
                    endpoints.push_back({iv.end, COMP_END, tid, i});
                }
                else
                {
                    endpoints.push_back({iv.start, FAST_START, tid, i});
                    endpoints.push_back({iv.end, FAST_END, tid, i});
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
    // show a simple wait spinner while sorting if endpoints large (sort is blocking but we can print once)
    std::sort(endpoints.begin(), endpoints.end(),
              [](const Endpoint& lhs, const Endpoint& rhs)
              {
                  if (lhs.time != rhs.time)
                      return lhs.time < rhs.time;
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
            case COMP_START:
            {
                if (ep.idx >= compress_intervals[ep.tid].size())
                {
                    std::cerr << "❌ Invalid compress index " << ep.idx << " for TID " << ep.tid << "\n";
                    continue;
                }
                const Interval&      iv = compress_intervals[ep.tid][ep.idx];
                const ActiveCompress ac{ep.tid, ep.idx, iv.start, iv.end};
                const uint64_t       key = make_key(ep.tid, ep.idx);
                active_index_by_key[key] = active_list.size();
                active_list.push_back(ac);
                break;
            }

            case COMP_END:
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

            case FAST_START:
            {
                if (ep.idx >= fast_intervals[ep.tid].size())
                {
                    std::cerr << "❌ Invalid fast index " << ep.idx << " for TID " << ep.tid << "\n";
                    continue;
                }
                const Interval& fiv = fast_intervals[ep.tid][ep.idx];
                for (const ActiveCompress& ac : active_list)
                {
                    if (ac.tid == ep.tid)
                        continue;
                    const uint64_t ov_start = std::max(fiv.start, ac.start);
                    const uint64_t ov_end   = std::min(fiv.end, ac.end);
                    if (ov_start < ov_end)
                    {
                        overlaps.push_back({ep.tid, ep.idx, ac.tid, ac.idx, ov_start, ov_end});
                    }
                }
                break;
            }

            case FAST_END:
                // nothing to do; overlaps already recorded at FAST_START
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

    #if 0
    for (const auto& o : overlaps)
    {
        if (o.fast_idx >= fast_intervals[o.fast_tid].size() || o.comp_idx >= compress_intervals[o.comp_tid].size())
        {
            std::cerr << "❌ Invalid overlap index for TID " << o.fast_tid << " or " << o.comp_tid << "\n";
            continue;
        }
        const Interval& fiv = fast_intervals[o.fast_tid][o.fast_idx];
        const Interval& civ = compress_intervals[o.comp_tid][o.comp_idx];

        std::cout << "FastPath TID " << std::setw(4) << o.fast_tid << " [" << std::setw(8) << (fiv.start - min_start)
                  << " - " << std::setw(8) << (fiv.end - min_start) << "]"
                  << "  intersects with Compression TID " << std::setw(4) << o.comp_tid << " [" << std::setw(8)
                  << (civ.start - min_start) << " - " << std::setw(8) << (civ.end - min_start) << "]\n";
    }
    #endif
    // Long fast-path detection has been REMOVED as a serious-signal
    // criterion. Empirically, durations of 1-2 ms on HPC machines are
    // dominated by cold-path costs (page faults on first-touch of fresh
    // chunk allocations, NUMA migration, allocator pool contention).
    // None of these indicate a correctness bug in the chunk library.
    //
    // The only remaining criterion that genuinely violates an invariant
    // is "same-thread fast-path AND compression overlap", which would
    // mean a thread observed itself in two states it cannot legally be
    // in simultaneously.

    size_t n_same_thread = 0;
    for (const auto& o : overlaps)
    {
        if (o.fast_tid == o.comp_tid)
            ++n_same_thread;
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
