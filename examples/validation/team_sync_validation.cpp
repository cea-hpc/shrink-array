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

// ============================================================================
// The synchronisation of pmp::Team (fast_allocators/internal/pmp_loops.h),
// alone: no array, no variant -- the backend of the build (OpenMP, std or
// serial). Under the std backend: its SpinBarrier and lock-free single, with
// the spinning waits (default) and without (CHUNK_PMP_SPIN_US=0, the second
// ctest).
// ============================================================================
//
// Teams of 2, 3, 8 threads and the backend's default size:
//
//   1. barrier: in each of many rounds, every thread counts itself into the
//      round's counter, then meets the barrier; past it, every thread reads
//      the full count (no thread leaves a barrier before all have arrived,
//      and their writes are visible);
//   2. single, wait form: exactly one thread runs each, and after it every
//      thread sees its write;
//   3. single, nowait form, many in a row with no barrier between: each runs
//      exactly once (a fast thread may be many singles ahead of a slow one);
//   4. singles mixed with nowait dynamic loops (the constructs share one
//      count): each single once, each iteration of each loop once;
//   5. many teams in a row: each runs every rank once.
//
// `--bench [threads]`: the cost of a barrier, of a single (wait) and of a
// team open + close, at that team size (default: the backend's).
//
// Exit status 0 when every check passes.

#include <atomic>  // std::atomic (the counters)
#include <chrono>  // std::chrono::steady_clock (--bench)
#include <print>   // std::println
#include <ratio>   // std::nano (--bench)
#include <cstdlib> // std::atoi
#include <cstring> // std::strcmp
#include <vector>  // std::vector

#include "fast_allocators/internal/pmp_loops.h"   // fast_alloc::pmp::team, Team, Options, WaitMode
#include "fast_allocators/internal/pmp_runtime.h" // fast_alloc::pmp::max_threads

namespace pmp = fast_alloc::pmp;

namespace
{

int g_fails  = 0;
int g_checks = 0;

void check(bool ok, const char* what, int nt)
{
    ++g_checks;
    if (!ok)
    {
        ++g_fails;
        std::println("FAIL {}, team {}", what, nt);
    }
}

void check_barrier(int nt)
{
    constexpr int                 rounds = 2000;
    std::vector<std::atomic<int>> count(rounds);
    std::atomic<int>              bad{0};
    int                           size = 0;
    pmp::team(pmp::Options{.num_threads = nt},
              [&](pmp::Team& team)
              {
                  team.single([&] { size = team.size(); });
                  for (int r = 0; r < rounds; ++r)
                  {
                      count[static_cast<std::size_t>(r)].fetch_add(1, std::memory_order_relaxed);
                      team.barrier();
                      if (count[static_cast<std::size_t>(r)].load(std::memory_order_relaxed) != team.size())
                      {
                          bad.fetch_add(1, std::memory_order_relaxed);
                      }
                  }
              });
    check(size >= 1 && bad.load() == 0, "1: past a barrier, every thread has arrived", nt);
}

void check_single_wait(int nt)
{
    constexpr int                 rounds = 2000;
    std::vector<std::atomic<int>> ran(rounds);
    std::vector<int>              value(rounds, 0);
    std::atomic<int>              bad{0};
    pmp::team(pmp::Options{.num_threads = nt},
              [&](pmp::Team& team)
              {
                  for (int r = 0; r < rounds; ++r)
                  {
                      const auto ur = static_cast<std::size_t>(r);
                      team.single(
                          [&]
                          {
                              ran[ur].fetch_add(1, std::memory_order_relaxed);
                              value[ur] = r + 1; // plain write: the single's barrier publishes it
                          });
                      if (value[ur] != r + 1)
                      {
                          bad.fetch_add(1, std::memory_order_relaxed);
                      }
                  }
              });
    bool once = true;
    for (int r = 0; r < rounds; ++r)
    {
        once = once && ran[static_cast<std::size_t>(r)].load() == 1;
    }
    check(once, "2: a single (wait) runs exactly once", nt);
    check(bad.load() == 0, "2: past a single (wait), every thread sees its write", nt);
}

void check_single_nowait(int nt)
{
    constexpr int                 rounds = 20000;
    std::vector<std::atomic<int>> ran(rounds);
    pmp::team(pmp::Options{.num_threads = nt},
              [&](pmp::Team& team)
              {
                  for (int r = 0; r < rounds; ++r)
                  {
                      team.single([&] { ran[static_cast<std::size_t>(r)].fetch_add(1, std::memory_order_relaxed); },
                                  pmp::nowait);
                  }
                  team.barrier();
              });
    bool once = true;
    for (int r = 0; r < rounds; ++r)
    {
        once = once && ran[static_cast<std::size_t>(r)].load() == 1;
    }
    check(once, "3: nowait singles in a row, each exactly once", nt);
}

void check_mixed(int nt)
{
    constexpr int                 rounds = 300;
    constexpr int                 n      = 257;
    std::vector<std::atomic<int>> ran(rounds);
    std::vector<std::atomic<int>> hits(static_cast<std::size_t>(rounds) * n);
    pmp::team(pmp::Options{.num_threads = nt},
              [&](pmp::Team& team)
              {
                  for (int r = 0; r < rounds; ++r)
                  {
                      team.single([&] { ran[static_cast<std::size_t>(r)].fetch_add(1, std::memory_order_relaxed); },
                                  pmp::nowait);
                      team.for_dynamic(
                          0, n, 7,
                          [&](int i)
                          {
                              hits[(static_cast<std::size_t>(r) * n) + static_cast<std::size_t>(i)].fetch_add(
                                  1, std::memory_order_relaxed);
                          },
                          pmp::nowait);
                  }
                  team.barrier();
              });
    bool once = true;
    for (int r = 0; r < rounds; ++r)
    {
        once = once && ran[static_cast<std::size_t>(r)].load() == 1;
    }
    bool every = true;
    for (const auto& h : hits)
    {
        every = every && h.load() == 1;
    }
    check(once, "4: singles mixed with dynamic loops, each exactly once", nt);
    check(every, "4: dynamic loops mixed with singles, each iteration once", nt);
}

void check_teams(int nt)
{
    constexpr int    teams = 3000;
    std::atomic<int> ranks{0};
    int              size = 0;
    for (int k = 0; k < teams; ++k)
    {
        pmp::team(pmp::Options{.num_threads = nt},
                  [&](pmp::Team& team)
                  {
                      team.masked([&] { size = team.size(); });
                      ranks.fetch_add(1, std::memory_order_relaxed);
                  });
    }
    check(ranks.load() == teams * size, "5: teams in a row, every rank once each", nt);
}

double ns_per(std::chrono::steady_clock::time_point t0, long long n)
{
    return std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count() /
           static_cast<double>(n);
}

int bench(int nt)
{
    constexpr int reps           = 20000;
    constexpr int n_teams        = 2000;
    constexpr int dyn_reps       = 5000;
    double        bar_ns         = 0;
    double        sgl_ns         = 0;
    double        dyn_wait_ns    = 0;
    double        dyn_nowait_ns  = 0;
    double        dyn_rebuild_ns = 0;
    int           size           = 0;
    pmp::team(pmp::Options{.num_threads = nt},
              [&](pmp::Team& team)
              {
                  team.barrier(); // every thread in
                  auto t0 = std::chrono::steady_clock::now();
                  for (int r = 0; r < reps; ++r)
                  {
                      team.barrier();
                  }
                  team.masked([&] { bar_ns = ns_per(t0, reps); });
                  team.barrier();
                  t0 = std::chrono::steady_clock::now();
                  for (int r = 0; r < reps; ++r)
                  {
                      team.single([] {});
                  }
                  team.masked(
                      [&]
                      {
                          sgl_ns = ns_per(t0, reps);
                          size   = team.size();
                      });
                  // A dynamic loop's entry and exit: an empty body, one
                  // iteration per thread, with its barrier...
                  const int n = team.size();
                  team.barrier();
                  t0 = std::chrono::steady_clock::now();
                  for (int r = 0; r < dyn_reps; ++r)
                  {
                      team.for_dynamic(0, n, 1, [](int) {});
                  }
                  team.masked([&] { dyn_wait_ns = ns_per(t0, dyn_reps); });
                  // ... without it, many in a row (every thread at the mutex at once) ...
                  team.barrier();
                  t0 = std::chrono::steady_clock::now();
                  for (int r = 0; r < dyn_reps; ++r)
                  {
                      team.for_dynamic(0, n, 1, [](int) {}, pmp::nowait);
                  }
                  team.barrier();
                  team.masked([&] { dyn_nowait_ns = ns_per(t0, dyn_reps); });
                  // ... and the rebuild loop's shape: 16 x size iterations, chunk 16.
                  team.barrier();
                  t0 = std::chrono::steady_clock::now();
                  for (int r = 0; r < dyn_reps; ++r)
                  {
                      team.for_dynamic(0, 16 * n, 16, [](int) {});
                  }
                  team.masked([&] { dyn_rebuild_ns = ns_per(t0, dyn_reps); });
              });
    const auto t0 = std::chrono::steady_clock::now();
    for (int k = 0; k < n_teams; ++k)
    {
        pmp::team(pmp::Options{.num_threads = nt}, [](pmp::Team&) {});
    }
    const double team_ns = ns_per(t0, n_teams);
    std::println("[team-sync] {} threads: barrier {:.2f} us | single (wait) {:.2f} us | team open+close {:.2f} us",
                 size, bar_ns * 1e-3, sgl_ns * 1e-3, team_ns * 1e-3);
    std::println("[team-sync] {} threads: dynamic (wait) {:.2f} us | dynamic (nowait) {:.2f} us | "
                 "dynamic 16 x size, chunk 16 {:.2f} us",
                 size, dyn_wait_ns * 1e-3, dyn_nowait_ns * 1e-3, dyn_rebuild_ns * 1e-3);
    return 0;
}

} // namespace

int main(int argc, char** argv)
{
    if (argc > 1 && std::strcmp(argv[1], "--bench") == 0)
    {
        return bench(argc > 2 ? std::atoi(argv[2]) : 0);
    }
    for (const int nt : {2, 3, 8, 0})
    {
        check_barrier(nt);
        check_single_wait(nt);
        check_single_nowait(nt);
        check_mixed(nt);
        check_teams(nt);
    }
    std::println("team_sync_validation: {} checks, {} failures (default team {})", g_checks, g_fails,
                 pmp::max_threads());
    return g_fails == 0 ? 0 : 1;
}
