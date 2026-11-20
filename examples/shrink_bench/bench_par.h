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
 * @file bench_par.h
 * @brief The bench's parallel regions and worksharing loops: one source for
 *        the OpenMP and std backends (docs/threading_backends.md).
 *
 * A region is bench_par::region(body), body(Team& t) run by every thread of a
 * team of the library's backend (pmp::team): OpenMP, a `#pragma omp parallel`
 * whose body calls the lambda; std, a team of the backend's std::jthreads;
 * serial, the calling thread. The TBB bench (BENCH_PAR_TBB, a std build with
 * CHUNK_WITH_TBB): one TBB task per virtual rank, which never waits for
 * another -- the arrays written from TBB's threads, as a TBB program would.
 * There region() also begins and ends the library's phase around the tasks,
 * as OMPT does around an OpenMP region (region_ends_phase).
 *
 * Inside it, the loops are written as a header macro followed by the body:
 *
 *     BENCH_FOR_STATIC_NOWAIT_SCALAR(t, i, first_index, start0)
 *     {
 *         arr[i] = ...;
 *     }
 *
 * - OpenMP: the macro is the pragma the loop had, then
 *   `for (Int i = (b); i < (e); ++i)`. The worksharing the compiler sees is
 *   the same as before: orphaned `omp for` constructs binding to the region.
 * - std / serial: the macro is a loop over the calling thread's share,
 *   computed from t.rank() and t.size() exactly as libgomp shares the same
 *   schedule out. A rank gets the same iterations in every loop of the same
 *   bounds and schedule -- what the nowait reasoning of the phases relies on
 *   (bench_phase_compute.cpp) -- and the same blocks from one region to the
 *   next (the std backend's worker k is always rank k: first touch).
 *
 * The schedules: _STATIC (`schedule(static)`, one block per thread), _BLOCK
 * (`schedule(static, sched_blk)`, a local named sched_blk:
 * bench_sched_for(), bench_prelude.h), _ROUND_ROBIN (`schedule(static, 1)`).
 * Every loop is NOWAIT, every single() too: the regions have no barrier,
 * which is what lets the TBB bench run them as tasks (below).
 *
 * Reductions (_RED): the body adds to BENCH_RED(v), not to v: each thread adds
 * into a private partial, BENCH_RED(v), which goes into v atomically once the
 * thread's share is done -- whole after the next barrier or the end of the
 * region. Every build, OpenMP included: an orphaned `omp for reduction(+ : v)`
 * in the body of the region's lambda, v a capture by reference, is refused by
 * clang ("reduction variable must be shared"), accepted by gcc.
 *
 * The macros name the loop's variable, so the body is the same text in every
 * build; `continue` works, `break` is refused by OpenMP anyway.
 */

#pragma once

#include <algorithm>       // std::min (the block schedule's last block), std::sort, std::adjacent_find (shared%)
#include <atomic>          // std::atomic_ref (the reductions' partials, atomic_add)
#include <cstddef>         // std::size_t (the TBB thread count)
#include <cstdint>         // std::uint64_t (the TBB regions' construct numbers)
#include <cstdlib>         // std::abort (a construct the TBB regions do not have)
#include <cstdio>          // std::fputs (its message)
#include <format>          // std::format (shared%)
#include <memory>          // std::unique_ptr (the TBB regions' construct slots)
#include <mutex>           // std::mutex (critical(), the TBB regions' slots)
#include <unordered_map>   // std::unordered_map (the TBB regions' construct slots)
#include <utility>         // std::forward
#include <chrono>          // std::chrono::steady_clock (BENCH_RANK_TIMING)
#include <source_location> // std::source_location (the timed call sites)
#include <string>          // std::string (a call site's name)
#include <string_view>     // std::string_view (its file name)
#include <thread>          // std::this_thread::yield (BENCH_TBB_LINGER_US)
#include <vector>          // std::vector (a call's rank times, the sites)

#include "bench_int.h"                // Int
#include "chunk/internal/pmp_chunk.h" // shrink_array::pmp::team, pmp::Team, pmp::Options, pmp::backend

#if defined(BENCH_PAR_TBB) && BENCH_PAR_TBB
    #include <oneapi/tbb/blocked_range.h>  // oneapi::tbb::blocked_range (the virtual ranks)
    #include <oneapi/tbb/global_control.h> // oneapi::tbb::global_control (the thread count)
    #include <oneapi/tbb/parallel_for.h>   // oneapi::tbb::parallel_for, static_partitioner
    #include <oneapi/tbb/task_arena.h>     // oneapi::tbb::task_arena, this_task_arena::current_thread_index (shared%)
    #include <oneapi/tbb/task_scheduler_observer.h> // oneapi::tbb::task_scheduler_observer (BENCH_RANK_TIMING, BENCH_TBB_PIN)
    #if defined(__linux__)
        #include <pthread.h> // pthread_setaffinity_np (BENCH_TBB_PIN)
        #include <sched.h>   // sched_getaffinity, cpu_set_t (BENCH_TBB_PIN)
    #endif

    #include "chunk/phases.h"      // finalize_parallel_phase (the end of a region)
    #include "chunk/tbb_adaptor.h" // shrink_array::tbb::phase_body (the begin of each task), use_tbb_for_library_teams
#endif

namespace bench_par
{
/// @brief Team::single(f, nowait): no barrier after it.
inline constexpr auto nowait = shrink_array::pmp::nowait;

namespace detail
{
/// @brief true: region() times finalize_parallel_phase() apart (the TBB
///        regions call it themselves); false: the phase ends elsewhere.
#if defined(BENCH_PAR_TBB) && BENCH_PAR_TBB
inline constexpr bool finalize_apart = true;
#else
inline constexpr bool finalize_apart = false;
#endif

/**
 * @brief Per-rank timing of the regions (BENCH_RANK_TIMING=1), to tell where
 *        the idle threads come from -- TBB's workers (thread_dispatcher::process
 *        in a profile), or OpenMP's / the std backend's, to compare.
 * @details Each outermost region() call is timed from the calling thread
 *          (t0: before arena().execute(), W: when it returns), and each of its
 *          n virtual ranks from inside its task (start s_k, end e_k, both from
 *          t0). Per rank, W = s_k + (e_k - s_k) + (e_max - e_k) + (W - e_max),
 *          so the region's n x W thread-time splits exactly into
 *            - busy:      sum (e_k - s_k), the bench's work and the library's;
 *            - imbalance: sum (e_max - e_k), the ranks done waiting for the slowest;
 *            - launch:    sum s_k, a rank waiting for its task to start (worker
 *                         wake-up, begin_parallel_phase(), a rank queued behind
 *                         another on the same thread);
 *            - tail:      n x (W - e_max), the last rank's end to execute()'s return.
 *          After each region, finalize_parallel_phase(): its wall time x threads
 *          is the finalize share (the library's teams work inside it, on TBB's
 *          threads; how busy they are is not split). Between a finalize and the
 *          next region the calling thread alone runs the bench's serial code: its
 *          time, x (threads - 1), is the serial share of the idle workers. The
 *          three wall times sum to the run's. Totals per call site (file:line of the region()
 *          call), printed on stderr at exit, the costliest idle first.
 *          OpenMP and the std backend (region() is a pmp::team): t0 before the
 *          team, W when it returns, each rank from inside it; the team may be
 *          smaller than asked (its real size is used). Under OMPT the phase
 *          ends INSIDE the region (finalize_parallel_phase() at its end, after
 *          the implicit barrier): it falls in the tail, and the finalize line
 *          is 0. Under the std backend without OMPT the bench ends the phase
 *          between regions: it falls in the serial share.
 *          Two more columns tell late workers apart: "late ms", the mean over
 *          the calls of the latest rank start (max s_k), and, TBB only,
 *          "shared%", the calls in which one TBB thread ran two ranks or more
 *          (this_task_arena::current_thread_index()): a worker came late, its
 *          rank's task was stolen and ran after another one -- the region
 *          then lasts about two ranks. "max ms" is the worst start of all.
 *          TBB, under the table: the mean number of distinct arena threads
 *          per region, and the slot indices most often missing from a region
 *          and most often running two ranks -- the same slot every time is a
 *          worker the arena never gets; a varying one, a worker held elsewhere.
 *          Then, from a task_scheduler_observer on the bench's arena, the TBB
 *          workers in it at a region's start, and how many entered and left
 *          it during the region, during its finalize and between the two.
 *          Off by default: one test of a static bool per region, no clock read.
 */
namespace rank_timing
{
using Clock = std::chrono::steady_clock;

[[nodiscard]] inline bool enabled() noexcept
{
    static const bool on = []
    {
        const char* v = std::getenv("BENCH_RANK_TIMING");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    return on;
}

[[nodiscard]] inline double seconds(Clock::duration d) noexcept
{
    return std::chrono::duration<double>(d).count();
}

/// @brief One region() call site's totals, in seconds (thread-time except wall).
struct Site
{
    std::string name;
    long long   calls     = 0;
    double      wall      = 0; ///< sum of W
    double      busy      = 0;
    double      imbalance = 0;
    double      launch    = 0;
    double      tail      = 0;
    double      max_busy  = 0; ///< sum over calls of the slowest rank's busy time
    double      mean_busy = 0; ///< sum over calls of the mean busy time
    double      late      = 0; ///< sum over calls of the latest rank start
    double      max_late  = 0; ///< the latest rank start of all calls
    long long   shared    = 0; ///< calls in which a thread ran two ranks or more (TBB)
    long long   threads   = 0; ///< sum over calls of the distinct threads that ran ranks (TBB)
};

/// @brief TBB: the workers in the bench's arena now, and how many entered and
///        left it so far (a task_scheduler_observer's counts).
struct ArenaCount
{
    long long in      = 0;
    long long entered = 0;
    long long left    = 0;
};

/// @brief The starts and ends of one call's ranks, from t0, in seconds.
struct Call
{
    /// @param threads TBB: record each rank's thread (shared%).
    explicit Call(int n, bool threads = false)
        : start(static_cast<std::size_t>(n))
        , end(static_cast<std::size_t>(n))
        , thread(threads ? static_cast<std::size_t>(n) : 0, -1)
    {
    }
    std::vector<double> start;
    std::vector<double> end;
    std::vector<int>    thread;            ///< TBB: each rank's thread index in the arena; empty otherwise
    Clock::time_point   t0 = Clock::now(); ///< last: the vectors are allocated before it
};

/**
 * @brief TBB: the library's teams, timed by the bench's team launcher
 *        (detail::timed_team_launch(), BENCH_RANK_TIMING only).
 * @details A team of the std backend on TBB (Pool::try_run_launched()) enqueues
 *          its n - 1 tasks, waits until they have all started (up to
 *          CHUNK_PMP_JOIN_US), then works. Per team: the enqueues' time, and
 *          the wait for its last task -- the start of the last task that came
 *          within the join budget, or the whole budget when one never came.
 *          The finalize's teams (those opened while region() runs
 *          finalize_parallel_phase()) are split from the others. The launcher
 *          is serialised by the pool's team mutex; the tasks report with
 *          atomics, each with its team's number and launch time.
 */
struct TeamStats
{
    // The team being launched (written by the launcher, read by its tasks).
    std::atomic<std::uint64_t> gen{0};
    std::atomic<long long>     last_ns{0}; ///< latest start within the budget, from the launch
    std::atomic<int>           came{0};    ///< its tasks started within the budget
    int                        asked    = 0;
    bool                       finalize = false;
    bool                       serial   = false; ///< opened between two regions (no region, no finalize)
    bool                       open     = false; ///< a team not folded yet

    std::atomic<bool>      in_finalize{false}; ///< region() is in finalize_parallel_phase()
    std::atomic<bool>      in_region{false};   ///< region() is in arena().execute() (timed regions)
    std::atomic<long long> late{0};            ///< tasks started past the budget (all teams)
    std::atomic<long long> gap_teams{0};       ///< serial teams launched since the last region (Registry::enter)

    // Totals, the finalize's teams (folded by the launcher and region()).
    long long teams       = 0;
    long long tasks_asked = 0;
    long long tasks_came  = 0;
    double    enqueue_s   = 0;
    double    wait_s      = 0;
    long long other_teams = 0; ///< teams opened outside a finalize
    long long other_asked = 0;
    // Totals, the teams opened between regions (serial code), as the finalize's.
    struct Serial
    {
        long long teams   = 0;
        long long asked   = 0;
        long long came    = 0;
        double    enqueue = 0;
        double    wait    = 0;

        Serial& operator+=(const Serial& o) noexcept
        {
            teams += o.teams, asked += o.asked, came += o.came, enqueue += o.enqueue, wait += o.wait;
            return *this;
        }
        [[nodiscard]] Serial operator-(const Serial& o) const noexcept
        {
            return {teams - o.teams, asked - o.asked, came - o.came, enqueue - o.enqueue, wait - o.wait};
        }
    };
    Serial serial_totals; ///< every team opened outside a region and a finalize (setup included)

    /// @brief Fold the team being launched into the totals (a no-op if none).
    void fold(long long budget_ns) noexcept
    {
        if (!open)
        {
            return;
        }
        open         = false;
        const int  c = came.load(std::memory_order_acquire);
        const auto w = 1e-9 * static_cast<double>(c < asked ? budget_ns : last_ns.load(std::memory_order_acquire));
        if (!finalize)
        {
            ++other_teams;
            other_asked += asked;
            if (serial)
            {
                ++serial_totals.teams;
                serial_totals.asked += asked;
                serial_totals.came += c;
                serial_totals.wait += w;
            }
            return;
        }
        ++teams;
        tasks_asked += asked;
        tasks_came += c;
        wait_s += w;
    }
};

[[nodiscard]] inline TeamStats& team_stats() noexcept
{
    static TeamStats* const t = new TeamStats(); // NOLINT: immortal, reported from atexit
    return *t;
}

class Registry
{
  public:
    static Registry& get()
    {
        static Registry* const r = [] // NOLINT: immortal, reported from atexit
        {
            auto* p = new Registry();
            std::atexit([] { get().report(); });
            return p;
        }();
        return *r;
    }

    /// @brief The calling thread enters an outermost region: the time since
    ///        the previous one ended was serial.
    void enter(Clock::time_point t0)
    {
        const std::scoped_lock lock(mutex_);
        TeamStats&             ts    = team_stats();
        const long long        teams = ts.gap_teams.exchange(0, std::memory_order_acq_rel);
        // The gap's last team is done (the calling thread came back from it):
        // folded now, so that it counts in this gap.
#if defined(BENCH_PAR_TBB) && BENCH_PAR_TBB
        ts.fold(fast_alloc::pmp::std_backend::join_budget_ns());
#endif
        if (have_last_)
        {
            gap_teams_ += ts.serial_totals - serial_mark_;
            const double gap = seconds(t0 - last_end_);
            serial_ += gap;
            (teams > 0 ? serial_with_teams_ : serial_without_teams_) += gap;
            ++(teams > 0 ? gaps_with_teams_ : gaps_without_teams_);
        }
    }

    /// @brief Account one call (@p wall_end: execute() returned, @p finalize_end:
    ///        finalize_parallel_phase() returned).
    void
    record(const std::source_location& loc, const Call& c, Clock::time_point wall_end, Clock::time_point finalize_end)
    {
        const std::size_t n    = c.start.size();
        const double      w    = seconds(wall_end - c.t0);
        double            emax = 0;
        double            bmax = 0;
        double            bsum = 0;
        double            ssum = 0;
        double            smax = 0;
        for (std::size_t k = 0; k < n; ++k)
        {
            emax = std::max(emax, c.end[k]);
            bmax = std::max(bmax, c.end[k] - c.start[k]);
            bsum += c.end[k] - c.start[k];
            ssum += c.start[k];
            smax = std::max(smax, c.start[k]);
        }
        double isum = 0;
        for (std::size_t k = 0; k < n; ++k)
        {
            isum += emax - c.end[k];
        }

        const std::scoped_lock lock(mutex_);
        bool                   shared   = false;
        long long              distinct = 0;
        if (!c.thread.empty())
        {
            // Ranks per arena slot (current_thread_index(); -1: not recorded).
            std::size_t slots = n;
            for (const int id : c.thread)
            {
                slots = std::max(slots, static_cast<std::size_t>(id + 1));
            }
            std::vector<int> per(slots, 0);
            for (const int id : c.thread)
            {
                if (id >= 0)
                {
                    ++per[static_cast<std::size_t>(id)];
                }
            }
            if (missing_.size() < slots)
            {
                missing_.resize(slots, 0);
                doubled_.resize(slots, 0);
            }
            for (std::size_t i = 0; i < slots; ++i)
            {
                distinct += per[i] > 0 ? 1 : 0;
                shared = shared || per[i] > 1;
                missing_[i] += (i < n && per[i] == 0) ? 1 : 0;
                doubled_[i] += per[i] > 1 ? 1 : 0;
            }
            ++slot_calls_;
        }
        Site& s = site(loc);
        ++s.calls;
        s.wall += w;
        s.busy += bsum;
        s.imbalance += isum;
        s.launch += ssum;
        s.tail += static_cast<double>(n) * std::max(0.0, w - emax);
        s.max_busy += bmax;
        s.mean_busy += n > 0 ? bsum / static_cast<double>(n) : 0.0;
        s.late += smax;
        s.max_late = std::max(s.max_late, smax);
        s.shared += shared ? 1 : 0;
        s.threads += distinct;
        finalize_ += seconds(finalize_end - wall_end);
        serial_mark_ = team_stats().serial_totals;
        last_end_    = finalize_end;
        have_last_   = true;
    }

    /// @brief TBB: the arena's counts at a region's start (@p a), its end (@p b)
    ///        and its finalize's end (@p c).
    void record_arena(const ArenaCount& a, const ArenaCount& b, const ArenaCount& c)
    {
        const std::scoped_lock lock(mutex_);
        if (arena_calls_ > 0)
        {
            arena_left_between_ += a.left - arena_last_.left;
            arena_entered_between_ += a.entered - arena_last_.entered;
        }
        ++arena_calls_;
        arena_in_ += a.in;
        arena_entered_region_ += b.entered - a.entered;
        arena_left_region_ += b.left - a.left;
        arena_entered_finalize_ += c.entered - b.entered;
        arena_left_finalize_ += c.left - b.left;
        arena_last_ = c;
    }

  private:
    Registry() = default;

    Site& site(const std::source_location& loc)
    {
        const std::string_view file = loc.file_name();
        const auto             cut  = file.find_last_of('/');
        std::string            name(cut == std::string_view::npos ? file : file.substr(cut + 1));
        name += ':';
        name += std::to_string(loc.line());
        for (auto& s : sites_)
        {
            if (s->name == name)
            {
                return *s;
            }
        }
        sites_.push_back(std::make_unique<Site>());
        sites_.back()->name = std::move(name);
        return *sites_.back();
    }

    void report()
    {
        const std::scoped_lock lock(mutex_);
        if (sites_.empty())
        {
            return;
        }
        const double threads = CHUNK_PMP_MAX_THREADS;
        Site         tot;
        for (const auto& s : sites_)
        {
            tot.wall += s->wall;
            tot.busy += s->busy;
            tot.imbalance += s->imbalance;
            tot.launch += s->launch;
            tot.tail += s->tail;
        }
        // serial_ runs from a finalize's end to the next region: finalize_ is not in it.
        const double serial_idle  = (threads - 1) * serial_;
        const double finalize_all = threads * finalize_;
        const double all          = tot.busy + tot.imbalance + tot.launch + tot.tail + finalize_all + serial_idle;
        const auto   pct          = [all](double x) { return all > 0 ? 100.0 * x / all : 0.0; };

        std::fprintf(stderr,
                     "[rank-timing] %d threads; wall %.3f s: in regions %.3f s, finalize_parallel_phase %.3f s, "
                     "serial between them %.3f s\n",
                     CHUNK_PMP_MAX_THREADS, tot.wall + finalize_ + serial_, tot.wall, finalize_, serial_);
        if (!finalize_apart)
        {
            std::fputs("[rank-timing] (not TBB: the phase's end is not timed apart -- OMPT's runs at each region's "
                       "end, in tail; the bench's own, between regions, in serial)\n",
                       stderr);
        }
        std::fprintf(stderr,
                     "[rank-timing] thread-time %.1f s: busy %.1f%%, imbalance %.1f%%, launch %.1f%%, "
                     "tail %.1f%%, finalize (not split) %.1f%%, serial (threads-1 idle) %.1f%%\n",
                     all, pct(tot.busy), pct(tot.imbalance), pct(tot.launch), pct(tot.tail), pct(finalize_all),
                     pct(serial_idle));

        std::vector<const Site*> order;
        order.reserve(sites_.size());
        for (const auto& s : sites_)
        {
            order.push_back(s.get());
        }
        std::sort(order.begin(), order.end(), [](const Site* a, const Site* b)
                  { return a->imbalance + a->launch + a->tail > b->imbalance + b->launch + b->tail; });

        std::fprintf(stderr, "[rank-timing] %-28s %8s %10s %9s %8s %8s %8s %8s %8s %8s %8s\n", "site", "calls",
                     "wall s", "max/mean", "busy%", "imbal%", "launch%", "tail%", "late ms", "max ms", "shared%");
        for (const Site* s : order)
        {
            const double      site_all = s->busy + s->imbalance + s->launch + s->tail;
            const auto        sp       = [site_all](double x) { return site_all > 0 ? 100.0 * x / site_all : 0.0; };
            const double      calls    = s->calls > 0 ? static_cast<double>(s->calls) : 1.0;
            const std::string shared   = finalize_apart
                                             ? std::format("{:.1f}", 100.0 * static_cast<double>(s->shared) / calls)
                                             : std::string("-");
            std::fprintf(stderr, "[rank-timing] %-28s %8lld %10.3f %9.2f %8.1f %8.1f %8.1f %8.1f %8.3f %8.3f %8s\n",
                         s->name.c_str(), s->calls, s->wall, s->mean_busy > 0 ? s->max_busy / s->mean_busy : 0.0,
                         sp(s->busy), sp(s->imbalance), sp(s->launch), sp(s->tail), 1e3 * s->late / calls,
                         1e3 * s->max_late, shared.c_str());
        }
        report_slots();
        report_arena();
    }

    /// @brief TBB: the workers' comings and goings in the bench's arena, per region.
    void report_arena() const
    {
        if (arena_calls_ == 0)
        {
            return;
        }
        const auto calls = static_cast<double>(arena_calls_);
        const auto per   = [calls](long long x) { return static_cast<double>(x) / calls; };
        std::fprintf(stderr,
                     "[rank-timing] TBB arena workers, per region: %.2f in the arena at its start (of %d); "
                     "during it %.2f entered, %.2f left; during its finalize %.2f entered, %.2f left; "
                     "from the finalize to the next region %.2f entered, %.2f left\n",
                     per(arena_in_), CHUNK_PMP_MAX_THREADS - 1, per(arena_entered_region_), per(arena_left_region_),
                     per(arena_entered_finalize_), per(arena_left_finalize_), per(arena_entered_between_),
                     per(arena_left_between_));
        const TeamStats& t = team_stats();
        if (t.teams > 0)
        {
            const double ms_enq  = 1e3 * t.enqueue_s / calls;
            const double ms_wait = 1e3 * t.wait_s / calls;
            std::fprintf(stderr,
                         "[rank-timing] finalize, per region: %.3f ms in all, %.2f library teams (%.1f tasks asked "
                         "each, %.1f%% started within the join); enqueueing %.3f ms, waiting for the last task "
                         "%.3f ms, the rest (flush, sweep, team ends) %.3f ms; %lld tasks late in all, %lld other "
                         "teams\n",
                         1e3 * finalize_ / calls, static_cast<double>(t.teams) / calls,
                         static_cast<double>(t.tasks_asked) / static_cast<double>(t.teams),
                         t.tasks_asked > 0
                             ? 100.0 * static_cast<double>(t.tasks_came) / static_cast<double>(t.tasks_asked)
                             : 0.0,
                         ms_enq, ms_wait, 1e3 * finalize_ / calls - ms_enq - ms_wait,
                         t.late.load(std::memory_order_relaxed), t.other_teams);
        }
        const long long gaps = gaps_with_teams_ + gaps_without_teams_;
        if (gaps > 0)
        {
            const TeamStats::Serial& g = gap_teams_;
            const auto mean            = [](double x, long long n) { return n > 0 ? x / static_cast<double>(n) : 0.0; };
            std::fprintf(stderr,
                         "[rank-timing] serial between regions: %lld gaps, %.3f s; %lld with library teams (%.3f s, "
                         "%.3f ms each), %lld without (%.3f s, %.3f ms each); their %lld teams (%.1f tasks asked "
                         "each, %.1f%% started within the join): enqueueing %.3f ms, waiting for the last task "
                         "%.3f ms per team, %.3f s + %.3f s in all\n",
                         gaps, serial_, gaps_with_teams_, serial_with_teams_,
                         1e3 * mean(serial_with_teams_, gaps_with_teams_), gaps_without_teams_, serial_without_teams_,
                         1e3 * mean(serial_without_teams_, gaps_without_teams_), g.teams,
                         mean(static_cast<double>(g.asked), g.teams),
                         100.0 * mean(static_cast<double>(g.came), g.asked), 1e3 * mean(g.enqueue, g.teams),
                         1e3 * mean(g.wait, g.teams), g.enqueue, g.wait);
        }
    }

    /// @brief TBB: the arena slots that miss a region or run two ranks, most often first.
    void report_slots() const
    {
        if (slot_calls_ == 0)
        {
            return;
        }
        long long threads = 0;
        for (const auto& s : sites_)
        {
            threads += s->threads;
        }
        const auto calls = static_cast<double>(slot_calls_);
        std::fprintf(stderr,
                     "[rank-timing] TBB arena slots: %.2f distinct threads per region (of %d ranks), %lld regions\n",
                     static_cast<double>(threads) / calls, CHUNK_PMP_MAX_THREADS, slot_calls_);
        const auto top = [calls](const char* what, const std::vector<long long>& v)
        {
            std::vector<std::size_t> idx;
            for (std::size_t i = 0; i < v.size(); ++i)
            {
                if (v[i] > 0)
                {
                    idx.push_back(i);
                }
            }
            std::sort(idx.begin(), idx.end(), [&v](std::size_t a, std::size_t b) { return v[a] > v[b]; });
            std::fprintf(stderr, "[rank-timing]   %s:", what);
            if (idx.empty())
            {
                std::fputs(" none", stderr);
            }
            for (std::size_t j = 0; j < idx.size() && j < 8; ++j)
            {
                std::fprintf(stderr, " slot %zu %.1f%%,", idx[j], 100.0 * static_cast<double>(v[idx[j]]) / calls);
            }
            std::fprintf(stderr, " (%zu slots in all)\n", idx.size());
        };
        top("missing from a region", missing_);
        top("ran two ranks or more", doubled_);
    }

    std::mutex                         mutex_;
    std::vector<std::unique_ptr<Site>> sites_;
    std::vector<long long>             missing_;        ///< TBB: regions each slot ran no rank of
    std::vector<long long>             doubled_;        ///< TBB: regions each slot ran two ranks or more of
    long long                          slot_calls_ = 0; ///< TBB: regions with the slots recorded
    // TBB: the arena's observer counts, summed over the regions (record_arena()).
    long long  arena_calls_            = 0;
    long long  arena_in_               = 0;
    long long  arena_entered_region_   = 0;
    long long  arena_left_region_      = 0;
    long long  arena_entered_finalize_ = 0;
    long long  arena_left_finalize_    = 0;
    long long  arena_entered_between_  = 0;
    long long  arena_left_between_     = 0;
    ArenaCount arena_last_;
    double     serial_ = 0;
    // TBB: serial_ split by whether the gap launched a library team.
    double            serial_with_teams_    = 0;
    double            serial_without_teams_ = 0;
    long long         gaps_with_teams_      = 0;
    long long         gaps_without_teams_   = 0;
    TeamStats::Serial gap_teams_;   ///< the library teams of the gaps between regions
    TeamStats::Serial serial_mark_; ///< team_stats().serial_totals at the last region's end
    double            finalize_  = 0;
    Clock::time_point last_end_  = {};
    bool              have_last_ = false;
};

/// @brief Depth of region() calls on this thread: only the outermost is timed.
inline thread_local int depth = 0;
} // namespace rank_timing
} // namespace detail

#if defined(BENCH_PAR_TBB) && BENCH_PAR_TBB

    #ifdef _OPENMP
        #error "BENCH_PAR_TBB: the TBB bench is not an OpenMP program (CHUNK_THREADS=std)"
    #endif
static_assert(shrink_array::pmp::backend == shrink_array::pmp::BackendKind::std_threads,
              "the TBB bench needs the std backend: TBB's threads write the arrays (CHUNK_THREADS=std)");

/// @brief false: the regions are TBB tasks, not a team of the library -- a
///        team-wide pass of the library (compress()) cannot run inside one.
inline constexpr bool team_regions = false;

/// @brief true: region() begins and ends the library's phase itself, as OMPT
///        does around an OpenMP region (bench_phases.h's wrappers stand aside).
inline constexpr bool region_ends_phase = true;

namespace detail
{
/// @brief What the tasks of one region share: the state of its constructs
///        (a single's claim, a dynamic loop's next block), found by number.
class RegionCtx
{
  public:
    struct Slot
    {
        std::atomic<long long> next{0};
        bool                   claimed = false; ///< under the mutex
    };

    /// @brief The slot of construct @p seq (made by the first task to get there).
    Slot& slot(std::uint64_t seq)
    {
        const std::scoped_lock lock(mutex_);
        auto&                  s = slots_[seq];
        if (!s)
        {
            s = std::make_unique<Slot>();
        }
        return *s;
    }

    /// @brief Claim construct @p seq's single: true for the first task only.
    bool claim(std::uint64_t seq)
    {
        const std::scoped_lock lock(mutex_);
        auto&                  s = slots_[seq];
        if (!s)
        {
            s = std::make_unique<Slot>();
        }
        return !std::exchange(s->claimed, true);
    }

  private:
    std::mutex                                               mutex_;
    std::unordered_map<std::uint64_t, std::unique_ptr<Slot>> slots_;
};

[[noreturn]] inline void refuse(const char* what) noexcept
{
    std::fputs(what, stderr);
    std::abort();
}
} // namespace detail

/**
 * @brief The region as one of its tasks sees it: a virtual rank in [0, size()).
 * @details A region of the TBB bench is @c size() TBB tasks, one per rank, that
 *          never wait for each other: the bench's regions are barrier-free
 *          (every worksharing loop and single is nowait), so a task that runs
 *          after another on the same thread loses nothing but time -- and no
 *          task blocks one that has not started, which is how a barrier deadlocks
 *          TBB. Hence no barrier() here, and the constructs nowait only:
 *          single(), for_dynamic(), and the static loops, whose shares come
 *          from the virtual rank as pmp::Team's do from the thread's.
 */
class Team
{
  public:
    Team(detail::RegionCtx* ctx, int rank, int size) noexcept : ctx_(ctx), rank_(rank), size_(size) {}

    [[nodiscard]] int rank() const noexcept
    {
        return rank_;
    }
    [[nodiscard]] int size() const noexcept
    {
        return size_;
    }

    /// @brief Run @p f in ONE task of the region (nowait only).
    template <typename F>
    void single(F&& f, shrink_array::pmp::WaitMode w)
    {
        if (w != shrink_array::pmp::WaitMode::nowait)
        {
            detail::refuse("bench_par: a single with a barrier in a TBB region (only nowait)\n");
        }
        if (ctx_->claim(met_++))
        {
            f();
        }
    }

    /// @brief One contiguous block per rank, libgomp's split (`schedule(static)`; nowait only).
    template <typename Index, typename Body>
    void for_static(Index first, Index last, Body&& body, shrink_array::pmp::WaitMode w)
    {
        require_nowait(w, "bench_par: a static loop with a barrier in a TBB region (only nowait)\n");
        const long long n = static_cast<long long>(last) - static_cast<long long>(first);
        if (n <= 0)
        {
            return;
        }
        const long long q = n / size_;
        const long long m = n % size_;
        const long long b = rank_ < m ? rank_ * (q + 1) : (m * (q + 1)) + ((rank_ - m) * q);
        const long long e = b + q + (rank_ < m ? 1 : 0);
        for (long long k = b; k < e; ++k)
        {
            body(static_cast<Index>(first + static_cast<Index>(k)));
        }
    }

    /// @brief for_static(): the same share (no OpenMP in a TBB bench).
    template <typename Index, typename Body>
    void for_static_simd(Index first, Index last, Body&& body, shrink_array::pmp::WaitMode w)
    {
        for_static(first, last, std::forward<Body>(body), w);
    }

    /// @brief Blocks of @p chunk iterations, block k to rank k % size()
    ///        (`schedule(static, chunk)`; nowait only).
    template <typename Index, typename Body>
    void for_static_chunks(Index first, Index last, int chunk, Body&& body, shrink_array::pmp::WaitMode w)
    {
        require_nowait(w, "bench_par: a static loop with a barrier in a TBB region (only nowait)\n");
        const long long n  = static_cast<long long>(last) - static_cast<long long>(first);
        const long long c  = chunk > 0 ? chunk : 1;
        const long long nt = size_;
        for (long long b = rank_ * c; b < n; b += nt * c)
        {
            const long long e = b + c < n ? b + c : n;
            for (long long k = b; k < e; ++k)
            {
                body(static_cast<Index>(first + static_cast<Index>(k)));
            }
        }
    }

    /// @brief for_static_chunks(): the same blocks (no OpenMP in a TBB bench).
    template <typename Index, typename Body>
    void for_static_chunks_simd(Index first, Index last, int chunk, Body&& body, shrink_array::pmp::WaitMode w)
    {
        for_static_chunks(first, last, chunk, std::forward<Body>(body), w);
    }

    /// @brief Blocks of @p chunk iterations, first come first served (nowait only).
    template <typename Index, typename Body>
    void for_dynamic(Index first, Index last, int chunk, Body&& body, shrink_array::pmp::WaitMode w)
    {
        if (w != shrink_array::pmp::WaitMode::nowait)
        {
            detail::refuse("bench_par: a dynamic loop with a barrier in a TBB region (only nowait)\n");
        }
        const long long n    = static_cast<long long>(last) - static_cast<long long>(first);
        const long long c    = chunk > 0 ? chunk : 1;
        auto&           slot = ctx_->slot(met_++);
        for (long long b = slot.next.fetch_add(c, std::memory_order_relaxed); b < n;
             b           = slot.next.fetch_add(c, std::memory_order_relaxed))
        {
            const long long e = b + c < n ? b + c : n;
            for (long long k = b; k < e; ++k)
            {
                body(static_cast<Index>(first + static_cast<Index>(k)));
            }
        }
    }

  private:
    static void require_nowait(shrink_array::pmp::WaitMode w, const char* what) noexcept
    {
        if (w != shrink_array::pmp::WaitMode::nowait)
        {
            detail::refuse(what);
        }
    }

    detail::RegionCtx* ctx_;
    int                rank_;
    int                size_;
    std::uint64_t      met_ = 0; ///< constructs this task has met (the slots' numbers)
};

namespace detail
{
/// @brief The bench's TBB threads: CHUNK_PMP_MAX_THREADS of them (the
///        backend's team size: CHUNK_NUM_THREADS, else OMP_NUM_THREADS, as the
///        scripts set it), the calling thread among them. Immortal.
/**
 * @brief BENCH_TBB_PARALLEL_PHASE=1: the bench's arena in one oneTBB parallel
 *        phase for the whole run (task_arena::start_parallel_phase(), a preview
 *        of oneTBB 2022.x under TBB_PREVIEW_PARALLEL_PHASE, which CMake sets).
 * @details What the phase changes is oneTBB's leave policy only: a worker out
 *          of work may stay in the arena, spinning, for at most 1 ms
 *          (worker_wait_leave_duration in oneTBB's waiters.h, fixed) unless the
 *          arena is in "fast leave", and fast leave is the default on HYBRID
 *          CPUs only. There the phase gives the workers that millisecond back
 *          (Core Ultra 5 235, 8 threads: workers in the arena at a region's
 *          start 4.04 -> 6.99 of 7, computation -6 %). Elsewhere (a 72-thread
 *          AArch64 node) they already have it, and the phase changes nothing;
 *          the wait there is the ranks' imbalance, longer than 1 ms:
 *          BENCH_TBB_LINGER_US. The phase is never ended. One line on stderr
 *          says whether it is on; with a oneTBB without the preview the variable
 *          is ignored, and said so.
 */
inline void start_parallel_phase_if_asked(oneapi::tbb::task_arena& a)
{
    const char* v = std::getenv("BENCH_TBB_PARALLEL_PHASE");
    if (v == nullptr || v[0] == '\0' || v[0] == '0')
    {
        return;
    }
    #if defined(__TBB_PREVIEW_PARALLEL_PHASE) && __TBB_PREVIEW_PARALLEL_PHASE
    a.start_parallel_phase();
    std::fputs(
        "[bench] BENCH_TBB_PARALLEL_PHASE: the bench's arena is in one oneTBB parallel phase for the whole run\n",
        stderr);
    #else
    (void)a;
    std::fputs("[bench] BENCH_TBB_PARALLEL_PHASE ignored: this oneTBB has no task_arena::start_parallel_phase() "
               "(a oneTBB 2022.x preview, TBB_PREVIEW_PARALLEL_PHASE)\n",
               stderr);
    #endif
}

/**
 * @brief BENCH_TBB_PIN=1: each thread of the bench's arena pinned to one CPU,
 *        the one of its arena slot -- the TBB counterpart of OMP_PROC_BIND=close.
 * @details The CPUs are those the process may run on at the arena's creation
 *          (sched_getaffinity: the job's cgroup, taskset), in increasing order;
 *          slot s gets the s-th of them (modulo their count). The static
 *          partitioner hands rank k to slot k, so rank k runs on the same CPU
 *          region after region, as OpenMP's thread k on its place: its data
 *          stays in that core's caches and on its NUMA node (first touch). The
 *          pin is set when a thread joins the arena (on_scheduler_entry, the
 *          main thread included) and loosened when it leaves (on_scheduler_exit:
 *          to the arena's CPUs, the first max_concurrency() of them), so that
 *          neither the main thread's serial code nor a worker gone to another
 *          arena stays tied to a slot's CPU. Not back to every CPU of the
 *          process: on a 288-CPU node (4 sockets) the main thread, out of the
 *          arena between every two regions, then ran the serial code (the
 *          reset sweep's scans, the library teams it opens) wherever the kernel
 *          put it, often a socket away from the data first touched by the
 *          arena's -- OpenMP's master stays on its place. Only for a machine the run has to itself: a pinned thread
 * cannot flee a core another process uses, and since every thread spins at the joins, the whole team waits for it --
 * with a 3-job build beside it (nice 19), 14 threads ran 15-30x slower than unpinned; idle, as
 *          fast or a little faster. Linux only; elsewhere the variable is
 *          ignored, and said so.
 *          One line on stderr says whether it is on.
 */
class ArenaPin final : public oneapi::tbb::task_scheduler_observer
{
  public:
    explicit ArenaPin(oneapi::tbb::task_arena& a, std::vector<int> cpus, const cpu_set_t& out)
        : oneapi::tbb::task_scheduler_observer(a), cpus_(std::move(cpus)), out_(out)
    {
        observe(true);
    }
    ArenaPin(const ArenaPin&) = delete ("an observer registered on the arena by address: neither copied nor moved");
    ArenaPin&
    operator=(const ArenaPin&) = delete ("an observer registered on the arena by address: neither copied nor moved");
    ~ArenaPin() override
    {
        observe(false);
    }

    void on_scheduler_entry(bool /*worker*/) override
    {
    #if defined(__linux__)
        const int slot = oneapi::tbb::this_task_arena::current_thread_index();
        if (slot < 0 || cpus_.empty())
        {
            return;
        }
        const int cpu = cpus_[static_cast<std::size_t>(slot) % cpus_.size()];
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(cpu, &set);
        (void)pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
    #endif
    }

    void on_scheduler_exit(bool /*worker*/) override
    {
    #if defined(__linux__)
        (void)pthread_setaffinity_np(pthread_self(), sizeof(out_), &out_);
    #endif
    }

  private:
    std::vector<int> cpus_;
    cpu_set_t        out_; ///< out of the arena: its slots' CPUs
};

/// @brief BENCH_TBB_PIN: the pinning observer on @p a, when asked (see ArenaPin).
inline void pin_if_asked(oneapi::tbb::task_arena& a)
{
    const char* v = std::getenv("BENCH_TBB_PIN");
    if (v == nullptr || v[0] == '\0' || v[0] == '0')
    {
        return;
    }
    #if defined(__linux__)
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof(set), &set) != 0)
    {
        std::fputs("[bench] BENCH_TBB_PIN ignored: sched_getaffinity failed\n", stderr);
        return;
    }
    std::vector<int> cpus;
    for (int c = 0; c < CPU_SETSIZE; ++c)
    {
        if (CPU_ISSET(c, &set))
        {
            cpus.push_back(c);
        }
    }
    if (cpus.empty())
    {
        return;
    }
    const int first = cpus.front();
    const int count = static_cast<int>(cpus.size());
    if (!a.is_active())
    {
        a.initialize();
    }
    // Out of the arena: the CPUs of its slots, not the process's.
    const int slots = std::min(a.max_concurrency(), count);
    cpu_set_t out;
    CPU_ZERO(&out);
    for (int s = 0; s < slots; ++s)
    {
        CPU_SET(cpus[static_cast<std::size_t>(s)], &out);
    }
    const int last = cpus[static_cast<std::size_t>(slots - 1)];
    (void)new ArenaPin(a, std::move(cpus), out); // NOLINT: immortal, as the arena
    std::fprintf(stderr,
                 "[bench] BENCH_TBB_PIN: arena slot s pinned to the s-th of the process's %d CPUs (first %d), "
                 "%d slots; out of the arena, the first %d of them (%d..%d)\n",
                 count, first, a.max_concurrency(), slots, first, last);
    #else
    (void)a;
    std::fputs("[bench] BENCH_TBB_PIN ignored: Linux only\n", stderr);
    #endif
}

inline oneapi::tbb::task_arena& arena()
{
    static const int                    n = CHUNK_PMP_MAX_THREADS;
    static oneapi::tbb::global_control* gc =
        new oneapi::tbb::global_control(oneapi::tbb::global_control::max_allowed_parallelism, // NOLINT: immortal
                                        static_cast<std::size_t>(n));
    static oneapi::tbb::task_arena* const a = []
    {
        auto* const p = new oneapi::tbb::task_arena(n); // NOLINT: immortal
        start_parallel_phase_if_asked(*p);
        pin_if_asked(*p);
        return p;
    }();
    (void)gc;
    return *a;
}

/// @brief The library's own teams take TBB's threads too, from the start (the
///        arrays' creation opens some before the first region) -- in the
///        bench's arena, which region() runs its tasks and its
///        finalize_parallel_phase() in; a team opened outside a region (the
///        creation, a serial pass) would otherwise go to the calling thread's
///        arena, another one, and the workers would change arena for it.
inline const bool library_teams_on_tbb = (shrink_array::tbb::use_tbb_for_library_teams(arena()), true);

/// @brief BENCH_RANK_TIMING: counts the TBB workers entering and leaving the
///        bench's arena (the main thread is not counted).
class ArenaWatch final : public oneapi::tbb::task_scheduler_observer
{
  public:
    explicit ArenaWatch(oneapi::tbb::task_arena& a) : oneapi::tbb::task_scheduler_observer(a)
    {
        observe(true);
    }
    ArenaWatch(const ArenaWatch&) =
        delete ("an observer registered on the bench's arena by address: neither copied nor moved");
    ArenaWatch& operator=(const ArenaWatch&) =
        delete ("an observer registered on the bench's arena by address: neither copied nor moved");
    ~ArenaWatch() override
    {
        observe(false);
    }

    void on_scheduler_entry(bool worker) override
    {
        if (worker)
        {
            in_.fetch_add(1, std::memory_order_relaxed);
            entered_.fetch_add(1, std::memory_order_relaxed);
        }
    }

    void on_scheduler_exit(bool worker) override
    {
        if (worker)
        {
            in_.fetch_sub(1, std::memory_order_relaxed);
            left_.fetch_add(1, std::memory_order_relaxed);
        }
    }

    [[nodiscard]] rank_timing::ArenaCount now() const noexcept
    {
        return {in_.load(std::memory_order_relaxed), entered_.load(std::memory_order_relaxed),
                left_.load(std::memory_order_relaxed)};
    }

  private:
    std::atomic<long long> in_{0};
    std::atomic<long long> entered_{0};
    std::atomic<long long> left_{0};
};

/// @brief The bench arena's observer, BENCH_RANK_TIMING only (nullptr otherwise). Immortal.
inline ArenaWatch* arena_watch()
{
    static ArenaWatch* const w = []() -> ArenaWatch*
    {
        if (!rank_timing::enabled())
        {
            return nullptr;
        }
        oneapi::tbb::task_arena& a = arena();
        if (!a.is_active())
        {
            a.initialize();
        }
        return new ArenaWatch(a); // NOLINT: immortal, as the arena
    }();
    return w;
}

/**
 * @brief BENCH_TBB_FINALIZE_INSIDE=1: region() runs finalize_parallel_phase()
 *        inside its tasks, once every rank is done, and the other tasks serve
 *        the library's teams (TeamServe) instead of TBB tasks being enqueued.
 * @details On TBB a library team enqueues its n - 1 tasks one by one and waits
 *          for the last to start (Pool::try_run_launched()): on a 72-thread
 *          AArch64 node, 0.193 ms of enqueueing and 0.326 ms of waiting per
 *          finalize, for 0.185 ms of work. Inside the region the 71 other
 *          tasks are still running, on their threads: they take the team's
 *          join at once. Only when every rank has started and is done within
 *          the linger budget (BENCH_TBB_LINGER_US, 20 ms when unset); otherwise
 *          the finalize runs after the tasks, as without it. One line on
 *          stderr when it is on.
 */
[[nodiscard]] inline bool tbb_finalize_inside()
{
    static const bool on = []
    {
        const char* const v = std::getenv("BENCH_TBB_FINALIZE_INSIDE");
        const bool        b = v != nullptr && v[0] != '\0' && v[0] != '0';
        if (b)
        {
            std::fputs("[bench] BENCH_TBB_FINALIZE_INSIDE: the finalize runs inside the region's tasks\n", stderr);
        }
        return b;
    }();
    return on;
}

/// @brief One task of a library team: timed from its launch under
///        BENCH_RANK_TIMING (rank_timing::TeamStats), then @p task(@p arg).
inline void run_team_task(void (*task)(std::uint64_t),
                          std::uint64_t                  arg,
                          std::uint64_t                  gen,
                          rank_timing::Clock::time_point t0,
                          long long                      budget_ns,
                          bool                           timed)
{
    namespace rt = rank_timing;
    if (timed)
    {
        rt::TeamStats&  ts = rt::team_stats();
        const long long d  = std::chrono::duration_cast<std::chrono::nanoseconds>(rt::Clock::now() - t0).count();
        if (d > budget_ns)
        {
            ts.late.fetch_add(1, std::memory_order_relaxed);
        }
        else if (ts.gen.load(std::memory_order_acquire) == gen)
        {
            long long cur = ts.last_ns.load(std::memory_order_relaxed);
            while (cur < d && !ts.last_ns.compare_exchange_weak(cur, d, std::memory_order_acq_rel))
            {
            }
            ts.came.fetch_add(1, std::memory_order_acq_rel);
        }
    }
    task(arg);
}

/**
 * @brief BENCH_TBB_FINALIZE_INSIDE: the mailbox through which the region's
 *        tasks serve the library's teams during a finalize run inside it.
 * @details open() by the task that runs the finalize, close() after it. The
 *          launcher posts each team's task in it with as many tickets as tasks
 *          serve (every task of the region but the finalize's: the finalize
 *          runs inside only when none returned to TBB), and enqueues the rest
 *          as usual -- a region narrower than the team; a serving task takes a
 *          ticket under the mutex (one per team and task: a few microseconds
 *          for 71) and runs the team's task. A ticket never taken leaves the
 *          team to start without it, after the join budget. The post's number
 *          (seq) tells the servers a new team without the mutex.
 */
struct TeamServe
{
    std::mutex                 m;
    std::atomic<std::uint64_t> seq{0};
    std::atomic<bool>          active{false};
    int                        servers = 0; ///< the region's tasks that serve (all but the finalize's)

    // The post, under m.
    std::uint64_t post                       = 0;
    int           tickets                    = 0;
    void (*task)(std::uint64_t)              = nullptr;
    std::uint64_t                  arg       = 0;
    std::uint64_t                  gen       = 0;
    rank_timing::Clock::time_point t0        = {};
    long long                      budget_ns = 0;
    bool                           timed     = false;

    void open(int region_servers)
    {
        const std::scoped_lock lock(m);
        tickets = 0;
        servers = region_servers;
        active.store(true, std::memory_order_release);
    }

    void close()
    {
        active.store(false, std::memory_order_release);
        const std::scoped_lock lock(m);
        tickets = 0;
        servers = 0;
    }

    /// @brief Serve the teams until @p done (the finalize returned).
    void serve_until(const std::atomic<bool>& done, std::uint64_t seen)
    {
        while (!done.load(std::memory_order_acquire))
        {
            const std::uint64_t q = seq.load(std::memory_order_acquire);
            if (q == seen)
            {
                std::this_thread::yield();
                continue;
            }
            seen                              = q;
            void (*t)(std::uint64_t)          = nullptr;
            std::uint64_t                  a  = 0;
            std::uint64_t                  g  = 0;
            rank_timing::Clock::time_point c0 = {};
            long long                      b  = 0;
            bool                           tm = false;
            {
                const std::scoped_lock lock(m);
                if (post != q || tickets <= 0)
                {
                    continue;
                }
                --tickets;
                t  = task;
                a  = arg;
                g  = gen;
                c0 = t0;
                b  = budget_ns;
                tm = timed;
            }
            run_team_task(t, a, g, c0, b, tm);
        }
    }
};

[[nodiscard]] inline TeamServe& team_serve() noexcept
{
    static TeamServe* const s = new TeamServe(); // NOLINT: immortal
    return *s;
}

/**
 * @brief The library's team launcher (pmp::set_team_launcher) under
 *        BENCH_RANK_TIMING or BENCH_TBB_FINALIZE_INSIDE: as the TBB adaptor's
 *        -- @p count tasks enqueued in the bench's arena -- but each task timed
 *        from the launch (rank_timing::TeamStats) under BENCH_RANK_TIMING, and
 *        handed to the region's serving tasks (TeamServe) during a finalize
 *        run inside the region.
 */
inline void bench_team_launch(int count, void (*task)(std::uint64_t), std::uint64_t arg)
{
    namespace rt              = rank_timing;
    const bool      timed     = rt::enabled();
    const long long budget_ns = fast_alloc::pmp::std_backend::join_budget_ns();
    std::uint64_t   gen       = 0;
    bool            finalize  = false;
    bool            serial    = false;
    if (timed)
    {
        rt::TeamStats& t = rt::team_stats();
        t.fold(budget_ns);
        gen = t.gen.fetch_add(1, std::memory_order_acq_rel) + 1;
        t.last_ns.store(0, std::memory_order_relaxed);
        t.came.store(0, std::memory_order_relaxed);
        t.asked  = count;
        finalize = t.finalize = t.in_finalize.load(std::memory_order_acquire);
        serial = t.serial = !finalize && !t.in_region.load(std::memory_order_acquire);
        t.open            = true;
        if (serial)
        {
            t.gap_teams.fetch_add(1, std::memory_order_relaxed);
        }
    }
    const auto t0     = rt::Clock::now();
    int        served = 0;
    TeamServe& sv     = team_serve();
    if (sv.active.load(std::memory_order_acquire))
    {
        const std::scoped_lock lock(sv.m);
        if (sv.servers > 0)
        {
            served       = std::min(count, sv.servers);
            sv.tickets   = served;
            sv.task      = task;
            sv.arg       = arg;
            sv.gen       = gen;
            sv.t0        = t0;
            sv.budget_ns = budget_ns;
            sv.timed     = timed;
            sv.post      = sv.seq.load(std::memory_order_relaxed) + 1;
            sv.seq.store(sv.post, std::memory_order_release);
        }
    }
    for (int k = served; k < count; ++k)
    {
        arena().enqueue([task, arg, gen, t0, budget_ns, timed]
                        { run_team_task(task, arg, gen, t0, budget_ns, timed); });
    }
    if (finalize)
    {
        rt::team_stats().enqueue_s += rt::seconds(rt::Clock::now() - t0);
    }
    else if (serial)
    {
        rt::team_stats().serial_totals.enqueue += rt::seconds(rt::Clock::now() - t0);
    }
}

/// @brief The library's teams through bench_team_launch() under
///        BENCH_RANK_TIMING or BENCH_TBB_FINALIZE_INSIDE, after
///        use_tbb_for_library_teams(arena()) (library_teams_on_tbb).
inline const bool library_teams_bench = library_teams_on_tbb && (rank_timing::enabled() || tbb_finalize_inside()) &&
                                        fast_alloc::pmp::set_team_launcher(&bench_team_launch);

/**
 * @brief BENCH_TBB_LINGER_US=<us>: how long a region's task, its rank done,
 *        waits for the other ranks before returning to TBB (0, the default:
 *        it returns at once).
 * @details A TBB worker out of work stays in the arena 1 ms at most (oneTBB's
 *          worker_wait_leave_duration, fixed), then leaves it; a rank that
 *          ends more than 1 ms before the slowest one loses its worker, which
 *          the finalize's team then wakes back, and which is sometimes still
 *          missing at the next region's start (rank_timing's shared%). Waiting
 *          in the task keeps the worker until every rank is done: it then
 *          leaves the task as the region ends, and has its 1 ms for the
 *          finalize's team. A task waits only once every rank has started
 *          (a task spinning in place steals nothing: a rank still without a
 *          thread needs the others back in TBB), and never longer than the
 *          budget, so a task that never starts cannot hold the others. The
 *          wait spins (yield) on cores that would idle; it is counted in the
 *          imbalance, the rank's busy time ending with its body. Outermost
 *          regions only. One line on stderr when it is on.
 */
inline rank_timing::Clock::duration tbb_linger()
{
    static const rank_timing::Clock::duration d = []
    {
        const char* const v  = std::getenv("BENCH_TBB_LINGER_US");
        const long        us = v != nullptr ? std::strtol(v, nullptr, 10) : 0;
        if (us <= 0)
        {
            return rank_timing::Clock::duration::zero();
        }
        std::fprintf(stderr,
                     "[bench] BENCH_TBB_LINGER_US=%ld: a region's task waits up to %ld us for the other ranks\n", us,
                     us);
        return std::chrono::duration_cast<rank_timing::Clock::duration>(std::chrono::microseconds(us));
    }();
    return d;
}

} // namespace detail

/**
 * @brief Run @p body(Team&) once per virtual rank of [0, @p num_threads), as TBB
 *        tasks, as one phase of the library.
 * @details What OMPT does for an OpenMP region, placed here since TBB has no
 *          region hook (a task_scheduler_observer sees a thread join the arena,
 *          not a parallel_for end): begin_parallel_phase() before every task
 *          (tbb_adaptor.h's phase_body()), finalize_parallel_phase() once the
 *          arena call has returned -- every task done. Every region ends a phase,
 *          as every top-level OpenMP region does under OMPT; the bench's own
 *          begin_parallel_phase_g() / finalize_*_g() are no-ops in this build
 *          (region_ends_phase). @p loc names the call site for BENCH_RANK_TIMING
 *          (detail::rank_timing).
 */
template <typename Body>
void region(int num_threads, Body&& body, const std::source_location loc = std::source_location::current())
{
    namespace rt                    = detail::rank_timing;
    const int                 n     = num_threads > 0 ? num_threads : CHUNK_PMP_MAX_THREADS;
    const bool                timed = rt::enabled() && rt::depth == 0;
    std::unique_ptr<rt::Call> call;
    if (timed)
    {
        call = std::make_unique<rt::Call>(n, true);
        rt::Registry::get().enter(call->t0);
    }
    detail::ArenaWatch* const watch    = timed ? detail::arena_watch() : nullptr;
    const rt::ArenaCount      at_start = watch != nullptr ? watch->now() : rt::ArenaCount{};
    rt::Call* const           cp       = call.get();
    detail::RegionCtx         ctx;
    // BENCH_TBB_LINGER_US (detail::tbb_linger()): the ranks started and done.
    // BENCH_TBB_FINALIZE_INSIDE (detail::tbb_finalize_inside()): the finalize
    // run by the first task to see every rank done (leader), the others
    // serving its library teams until it returns (fin_done); tasks_begun /
    // decided / staying: whether every task is still there to serve.
    const bool                inside = rt::depth == 0 && detail::tbb_finalize_inside();
    const rt::Clock::duration linger =
        rt::depth != 0 ? rt::Clock::duration::zero()
        : inside && detail::tbb_linger() == rt::Clock::duration::zero()
            ? std::chrono::duration_cast<rt::Clock::duration>(std::chrono::milliseconds(20))
            : detail::tbb_linger();
    std::atomic<int>      started{0};
    std::atomic<int>      done{0};
    std::atomic<int>      tasks_begun{0}; ///< tasks that entered `tasks` (a task may run several ranks)
    std::atomic<int>      decided{0};     ///< tasks past the "every rank done?" test
    std::atomic<int>      staying{0};     ///< those that saw every rank done: they serve or lead
    std::atomic<bool>     leader{false};
    std::atomic<bool>     fin_done{false};
    bool                  fin_inside = false; // written by the leader, read after the tasks
    const std::uint64_t   serve_seq  = inside ? detail::team_serve().seq.load(std::memory_order_acquire) : 0;
    rt::Clock::time_point wall_end;
    rt::Clock::time_point fin_end;
    rt::ArenaCount        at_end;
    const auto            finalize = [&]
    {
        wall_end = timed ? rt::Clock::now() : rt::Clock::time_point{};
        at_end   = watch != nullptr ? watch->now() : rt::ArenaCount{};
        if (timed)
        {
            rt::team_stats().in_finalize.store(true, std::memory_order_release);
        }
        shrink_array::finalize_parallel_phase();
        if (timed)
        {
            rt::team_stats().in_finalize.store(false, std::memory_order_release);
            rt::team_stats().fold(fast_alloc::pmp::std_backend::join_budget_ns());
            fin_end = rt::Clock::now();
        }
    };
    const auto tasks = [&](const oneapi::tbb::blocked_range<int>& r)
    {
        tasks_begun.fetch_add(1, std::memory_order_seq_cst);
        for (int k = r.begin(); k != r.end(); ++k)
        {
            started.fetch_add(1, std::memory_order_relaxed);
            if (cp != nullptr)
            {
                const auto s = rt::Clock::now();
                Team       t(&ctx, k, n);
                body(t);
                const auto e                            = rt::Clock::now();
                cp->start[static_cast<std::size_t>(k)]  = rt::seconds(s - cp->t0);
                cp->end[static_cast<std::size_t>(k)]    = rt::seconds(e - cp->t0);
                cp->thread[static_cast<std::size_t>(k)] = oneapi::tbb::this_task_arena::current_thread_index();
            }
            else
            {
                Team t(&ctx, k, n);
                body(t);
            }
            done.fetch_add(1, std::memory_order_seq_cst);
        }
        if (linger > rt::Clock::duration::zero())
        {
            const auto until = rt::Clock::now() + linger;
            while (done.load(std::memory_order_acquire) < n && started.load(std::memory_order_relaxed) == n &&
                   rt::Clock::now() < until)
            {
                std::this_thread::yield();
            }
        }
        if (!inside)
        {
            return;
        }
        // Every task takes the "every rank done?" test once, and counts itself
        // past it (decided) AFTER counting itself staying when it saw them all
        // done. The leader waits until every task that began is past the test
        // -- a few instructions each, every rank being done -- and runs the
        // finalize inside only when all of them stay: a task that returned to
        // TBB may have left the arena, and a team waiting for it would wait its
        // whole join budget. (It was one counter, incremented before the test
        // and taken back after it: the 72 tasks leave their linger at the same
        // instant, so the leader nearly always saw another task between the two
        // and ran the finalize after the tasks.) Every task began once every
        // rank is done: tasks_begun is final by then.
        const bool all_done = done.load(std::memory_order_seq_cst) == n;
        if (all_done)
        {
            staying.fetch_add(1, std::memory_order_seq_cst);
        }
        decided.fetch_add(1, std::memory_order_seq_cst);
        if (!all_done)
        {
            return; // a rank still running, or not started: the finalize after the tasks
        }
        bool expected = false;
        if (leader.compare_exchange_strong(expected, true, std::memory_order_seq_cst))
        {
            const int begun = tasks_begun.load(std::memory_order_seq_cst);
            while (decided.load(std::memory_order_seq_cst) < begun)
            {
                std::this_thread::yield();
            }
            if (staying.load(std::memory_order_seq_cst) != begun)
            {
                fin_done.store(true, std::memory_order_release); // the servers return
                return;                                          // the finalize after the tasks
            }
            detail::TeamServe& sv = detail::team_serve();
            sv.open(begun - 1); // the other tasks, not the other ranks: a task may have run several
            finalize();
            sv.close();
            fin_inside = true;
            fin_done.store(true, std::memory_order_release);
        }
        else
        {
            detail::team_serve().serve_until(fin_done, serve_seq);
        }
    };
    ++rt::depth;
    if (timed)
    {
        rt::team_stats().in_region.store(true, std::memory_order_release);
    }
    // The finalize inside the arena, right after the tasks: the workers that
    // finished their rank still spin in it and join the library's team at
    // once, rather than leaving and being woken back for it. Under
    // BENCH_TBB_FINALIZE_INSIDE, in the tasks themselves when they could.
    detail::arena().execute(
        [&]
        {
            oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<int>(0, n, 1), shrink_array::tbb::phase_body(tasks),
                                      oneapi::tbb::static_partitioner{});
            if (!fin_inside)
            {
                finalize();
            }
        });
    --rt::depth;
    if (timed)
    {
        rt::team_stats().in_region.store(false, std::memory_order_release);
        rt::Registry::get().record(loc, *call, wall_end, fin_inside ? fin_end : rt::Clock::now());
        if (watch != nullptr)
        {
            rt::Registry::get().record_arena(at_start, at_end, watch->now());
        }
    }
}

/// @brief region() with CHUNK_PMP_MAX_THREADS virtual ranks.
template <typename Body>
void region(Body&& body, const std::source_location loc = std::source_location::current())
{
    region(0, std::forward<Body>(body), loc);
}
#else

using Team = shrink_array::pmp::Team;

/// @brief true: a region is a team of the library's backend, inside which the
///        library's team-wide passes (compress()) share the work.
inline constexpr bool team_regions = true;

/// @brief false: the phase is begun and ended by OMPT or by the bench's own
///        calls (bench_phases.h), not by region().
inline constexpr bool region_ends_phase = false;

/// @brief Run @p body(Team&) on every thread of a new team of @p num_threads
///        threads (0: the backend's default size). With BENCH_RANK_TIMING=1 the
///        outermost one is timed per rank (detail::rank_timing); @p loc names
///        the call site.
template <typename Body>
void region(int num_threads, Body&& body, const std::source_location loc = std::source_location::current())
{
    namespace rt   = detail::rank_timing;
    const auto opt = shrink_array::pmp::Options{.num_threads = num_threads};
    if (!rt::enabled() || rt::depth != 0) [[likely]]
    {
        shrink_array::pmp::team(opt, std::forward<Body>(body));
        return;
    }
    rt::Call call(num_threads > 0 ? num_threads : CHUNK_PMP_MAX_THREADS);
    rt::Registry::get().enter(call.t0);
    int size = 0; // the team's real size, written by its rank 0
    ++rt::depth;
    shrink_array::pmp::team(opt,
                            [&](Team& team)
                            {
                                const auto s = rt::Clock::now();
                                body(team);
                                const auto        e    = rt::Clock::now();
                                const std::size_t rank = static_cast<std::size_t>(team.rank());
                                if (rank < call.start.size())
                                {
                                    call.start[rank] = rt::seconds(s - call.t0);
                                    call.end[rank]   = rt::seconds(e - call.t0);
                                }
                                if (rank == 0)
                                {
                                    size = team.size();
                                }
                            });
    --rt::depth;
    const auto wall_end = rt::Clock::now();
    const auto n        = static_cast<std::size_t>(size > 0 ? size : 1);
    call.start.resize(std::min(n, call.start.size()));
    call.end.resize(call.start.size());
    rt::Registry::get().record(loc, call, wall_end, wall_end);
}

/// @brief region() with the backend's default team size.
template <typename Body>
void region(Body&& body, const std::source_location loc = std::source_location::current())
{
    region(0, std::forward<Body>(body), loc);
}

#endif

/// @brief The calling thread's share [lo, hi) of [@p first, @p last) under
///        `schedule(static)`: libgomp's split, the first (last - first) % size ranks
///        one iteration more.
[[nodiscard]] inline Int static_lo(const Team& team, Int first, Int last) noexcept
{
    const Int count = last - first;
    if (count <= 0)
    {
        return first;
    }
    const Int nt   = team.size();
    const Int rank = team.rank();
    const Int quot = count / nt;
    const Int rem  = count % nt;
    return first + (rank < rem ? rank * (quot + 1) : (rem * (quot + 1)) + ((rank - rem) * quot));
}

/// @brief One past the end of static_lo()'s share.
[[nodiscard]] inline Int static_hi(const Team& team, Int first, Int last) noexcept
{
    const Int count = last - first;
    if (count <= 0)
    {
        return first;
    }
    const Int nt   = team.size();
    const Int rank = team.rank();
    return static_lo(team, first, last) + (count / nt) + (rank < count % nt ? 1 : 0);
}

/// @brief @p target += @p delta, atomically, whatever the backend: std::atomic_ref is atomic
///        against OpenMP's threads as against the std backend's (it compiles
///        to the same instruction as `omp atomic`).
template <typename T>
inline void atomic_add(T& target, T delta) noexcept
{
    std::atomic_ref<T>(target).fetch_add(delta, std::memory_order_relaxed);
}

/// @brief A block size for Team::for_static_chunks (an int): the bench's
///        sched_blk (bench_sched_for(), an Int, a power of two or a slice).
[[nodiscard]] inline int block(Int sched_blk) noexcept
{
    return static_cast<int>(sched_blk);
}

// The bodies given to the helpers below, and the bench's loop bodies, are
// lambdas marked __attribute__((always_inline)) (after the parameters; the
// [[gnu::always_inline]] spelling there applies to the type and is ignored):
// gcc -O3 otherwise leaves a big body of a big region out of line, a call per
// iteration where the BENCH_FOR_* macros had the body in the loop.

/**
 * @brief A reduction inside a region: @p body(i, part) adds into the calling
 *        thread's partial for its share of [@p first, @p last) under `schedule(static)`,
 *        then the partial goes into @p sum atomically. No barrier: @p sum is whole
 *        after the next barrier or the end of the region.
 * @details Not a `reduction(+ : sum)` clause (an orphaned `omp for reduction` on
 *          a lambda's capture is refused by clang), not pmp::team_sum (a region
 *          of its own, where the bench's phases are one barrier-free region).
 */
template <typename T, typename Body>
void for_static_sum(Team& team, Int first, Int last, T& sum, Body&& body)
{
    T part{};
    team.for_static<Int>(
        first, last, [&](const Int i) __attribute__((always_inline)) { body(i, part); }, shrink_array::pmp::nowait);
    atomic_add(sum, part);
}

/// @brief for_static_sum() over round-robin blocks of @p chunk iterations
///        (`schedule(static, chunk)`).
template <typename T, typename Body>
void for_static_chunks_sum(Team& team, Int first, Int last, int chunk, T& sum, Body&& body)
{
    T part{};
    team.for_static_chunks<Int>(
        first, last, chunk, [&](const Int i) __attribute__((always_inline)) { body(i, part); },
        shrink_array::pmp::nowait);
    atomic_add(sum, part);
}

/// @brief The lock of critical(): one for the whole bench, as an unnamed
///        `omp critical`.
[[nodiscard]] inline std::mutex& critical_mutex() noexcept
{
    static std::mutex mtx;
    return mtx;
}

/// @brief Run @p func under the bench's one lock (an unnamed `omp critical`).
template <typename F>
void critical(F&& func)
{
    const std::scoped_lock lock(critical_mutex());
    func();
}

/// @brief A thread's partial of a reduction (every build, see the file
///        comment): added into the shared variable when the loop's share is done.
template <typename T>
class Partial
{
  public:
    explicit Partial(T& shared) noexcept : shared_(shared) {}
    Partial(const Partial&) = delete ("a thread's partial: added into the shared variable once, when it is destroyed");
    Partial& operator=(const Partial&) =
        delete ("a thread's partial: added into the shared variable once, when it is destroyed");
    Partial(Partial&&) = delete ("a thread's partial: added into the shared variable once, when it is destroyed");
    Partial&
    operator=(Partial&&) = delete ("a thread's partial: added into the shared variable once, when it is destroyed");
    ~Partial()
    {
        atomic_add(shared_, value);
    }

    /// @brief True once: the loop runs its share, then the partial goes in.
    [[nodiscard]] bool once() noexcept
    {
        return !done_ && (done_ = true);
    }

    T value{}; ///< the thread's partial

  private:
    T&   shared_;
    bool done_ = false;
};
} // namespace bench_par

// clang-format off
#ifdef _OPENMP

    #define BENCH_FOR_STATIC_NOWAIT(t, i, b, e) BENCH_OMP_FOR_STATIC_NOWAIT for (Int i = (b); i < (e); ++i)
    #define BENCH_FOR_STATIC_NOWAIT_SCALAR(t, i, b, e) BENCH_OMP_FOR_STATIC_NOWAIT_SCALAR for (Int i = (b); i < (e); ++i)
    // sched_blk copied into a local first: clang 19 and 20 crash on a chunk size that is a
    // capture of the region's lambda (segfault in EmitOMPWorksharingLoop).
    #define BENCH_FOR_BLOCK_NOWAIT_SCALAR(t, i, b, e) \
        if (const Int bench_blk_ = sched_blk; true) _Pragma("omp for schedule(static, bench_blk_) nowait") for (Int i = (b); i < (e); ++i)
    #define BENCH_FOR_ROUND_ROBIN_NOWAIT(t, i, b, e) _Pragma("omp for schedule(static, 1) nowait") for (Int i = (b); i < (e); ++i)
    // A per-thread partial, not a reduction clause: see the file comment.
    #define BENCH_FOR_STATIC_NOWAIT_RED_SCALAR(t, i, b, e, v) \
        for (::bench_par::Partial<decltype(v)> v##_part_(v); v##_part_.once();) BENCH_FOR_STATIC_NOWAIT_SCALAR(t, i, b, e)
    #define BENCH_FOR_BLOCK_NOWAIT_RED_SCALAR(t, i, b, e, v) \
        for (::bench_par::Partial<decltype(v)> v##_part_(v); v##_part_.once();) BENCH_FOR_BLOCK_NOWAIT_SCALAR(t, i, b, e)
    #define BENCH_RED(v) v##_part_.value

#else

    // The calling thread's share, as libgomp deals the same schedule out.
    #define BENCH_FOR_STATIC_NOWAIT(t, i, b, e) \
        for (Int i = ::bench_par::static_lo((t), (b), (e)), i##_end_ = ::bench_par::static_hi((t), (b), (e)); i < i##_end_; ++i)
    #define BENCH_FOR_STATIC_NOWAIT_SCALAR(t, i, b, e) BENCH_FOR_STATIC_NOWAIT(t, i, b, e)
    // schedule(static, sched_blk): block k of sched_blk iterations from b goes to rank k % size.
    #define BENCH_FOR_BLOCK_NOWAIT_SCALAR(t, i, b, e)                                                               \
        for (Int i##_blk_ = (b) + (static_cast<Int>((t).rank()) * sched_blk); i##_blk_ < (e);                     \
             i##_blk_ += static_cast<Int>((t).size()) * sched_blk)                                                \
            for (Int i = i##_blk_, i##_end_ = std::min<Int>(i##_blk_ + sched_blk, (e)); i < i##_end_; ++i)
    #define BENCH_FOR_ROUND_ROBIN_NOWAIT(t, i, b, e) \
        for (Int i = (b) + static_cast<Int>((t).rank()); i < (e); i += static_cast<Int>((t).size()))
    #define BENCH_FOR_STATIC_NOWAIT_RED_SCALAR(t, i, b, e, v) \
        for (::bench_par::Partial<decltype(v)> v##_part_(v); v##_part_.once();) BENCH_FOR_STATIC_NOWAIT(t, i, b, e)
    #define BENCH_FOR_BLOCK_NOWAIT_RED_SCALAR(t, i, b, e, v) \
        for (::bench_par::Partial<decltype(v)> v##_part_(v); v##_part_.once();) BENCH_FOR_BLOCK_NOWAIT_SCALAR(t, i, b, e)
    #define BENCH_RED(v) v##_part_.value

#endif
// clang-format on
