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

// ============================================================================
// The rank registry of the threading backend (fast_allocators/internal/
// pmp_runtime.h): a dense rank for a thread no team numbers. No array, no
// variant, no library: the header alone, driven from std::threads.
// ============================================================================
//
// What the per-rank structures of the library will need from it (the std
// backend, docs/threading_backends.md step 7):
//   - dense: the lowest free rank first, so ranks stay below the number of
//     threads alive at once;
//   - unique: two live threads never share a rank, however they race;
//   - bounded: past the capacity a thread gets -1 (the library reports it),
//     and a later call succeeds once a rank is free;
//   - reused: a thread gives its rank back when it exits, so a program that
//     keeps starting threads never runs out.
//
// Exit 0 = every check holds.

#include <atomic>  // std::atomic
#include <barrier> // std::barrier
#include <print>   // std::println
#include <thread>  // std::thread
#include <vector>  // std::vector

#include "fast_allocators/internal/pmp_runtime.h" // RankRegistry, registered_rank, thread_rank_registry

namespace pmp = fast_alloc::pmp;

namespace
{
int g_failures = 0;

void expect(bool ok, const char* what)
{
    if (!ok)
    {
        std::println("FAIL: {}", what);
        ++g_failures;
    }
}

/// A registry on its own: lowest free first, -1 when full, release and reuse.
void check_sequential()
{
    pmp::RankRegistry r(4);
    expect(r.capacity() == 4, "capacity 4");
    expect(r.acquire() == 0 && r.acquire() == 1 && r.acquire() == 2 && r.acquire() == 3, "ranks 0..3 in order");
    expect(r.acquire() == -1, "full: -1");
    expect(r.in_use() == 4 && r.high_water() == 4, "in_use / high_water at 4");
    r.release(1);
    r.release(3);
    expect(r.acquire() == 1, "the lowest free rank first (1)");
    expect(r.acquire() == 3, "then 3");
    r.release(0);
    r.release(1);
    r.release(2);
    r.release(3);
    expect(r.in_use() == 0 && r.high_water() == 4, "all given back, high water kept");
    pmp::RankRegistry one(0);
    expect(one.capacity() == 1 && one.acquire() == 0 && one.acquire() == -1, "capacity floored at 1");
}

/// Threads racing for the ranks of one registry: all distinct, all below the capacity.
void check_concurrent_acquire()
{
    constexpr int            n = 16;
    pmp::RankRegistry        r(n);
    std::vector<int>         got(n, -2);
    std::barrier             start(n);
    std::vector<std::thread> ts;
    for (int k = 0; k < n; ++k)
    {
        ts.emplace_back(
            [&, k]
            {
                start.arrive_and_wait();
                got[k] = r.acquire();
            });
    }
    for (auto& t : ts)
    {
        t.join();
    }
    std::vector<int> seen(n, 0);
    bool             ok = true;
    for (const int g : got)
    {
        ok = ok && g >= 0 && g < n && seen[g]++ == 0;
    }
    expect(ok, "concurrent acquire: n distinct ranks in [0, n)");
    expect(r.acquire() == -1, "concurrent acquire: then full");
}

/// The process-wide registry through registered_rank(): stable per thread,
/// bounded, given back at thread exit, densely reused.
void check_registered_rank()
{
    constexpr int cap = 4;
    expect(pmp::configure_thread_rank_registry(cap), "configure before first use");
    auto& reg = pmp::thread_rank_registry();
    expect(reg.capacity() == cap, "the configured capacity");
    expect(!pmp::configure_thread_rank_registry(8), "configure after first use refused");

    // cap threads alive at once: distinct ranks, the same at every call; one
    // more thread meanwhile gets -1, then a rank once one is free.
    std::vector<int>         got(cap, -2);
    std::barrier             held(cap + 1);
    std::barrier             done(cap + 1);
    std::vector<std::thread> ts;
    for (int k = 0; k < cap; ++k)
    {
        ts.emplace_back(
            [&, k]
            {
                const int r = pmp::registered_rank();
                got[k]      = (pmp::registered_rank() == r) ? r : -3;
                held.arrive_and_wait();
                done.arrive_and_wait();
            });
    }
    held.arrive_and_wait();
    int         extra = -2;
    std::thread late([&] { extra = pmp::registered_rank(); });
    late.join();
    expect(extra == -1, "a thread past the capacity gets -1");
    std::vector<int> seen(cap, 0);
    bool             ok = true;
    for (const int g : got)
    {
        ok = ok && g >= 0 && g < cap && seen[g]++ == 0;
    }
    expect(ok, "live threads: distinct stable ranks in [0, cap)");
    expect(reg.in_use() == cap, "cap ranks in use");
    done.arrive_and_wait();
    for (auto& t : ts)
    {
        t.join();
    }
    expect(reg.in_use() == 0, "every rank given back at thread exit");

    // A thread started after the others exited: rank 0 again (dense).
    int         again = -2;
    std::thread next([&] { again = pmp::registered_rank(); });
    next.join();
    expect(again == 0, "reused: a new thread gets rank 0");

    // Waves of short-lived threads, many more than the capacity over the run,
    // at most cap at once: each spins until a rank is free, and no two hold
    // the same rank at the same time.
    constexpr int                 waves = 50;
    constexpr int                 per   = 8;
    std::vector<std::atomic<int>> owner(cap);
    for (auto& o : owner)
    {
        o.store(0);
    }
    std::atomic<int> clash{0};
    std::atomic<int> out_of_range{0};
    for (int w = 0; w < waves; ++w)
    {
        std::vector<std::thread> wave;
        for (int k = 0; k < per; ++k)
        {
            wave.emplace_back(
                [&]
                {
                    int r = pmp::registered_rank();
                    while (r < 0)
                    {
                        std::this_thread::yield();
                        r = pmp::registered_rank();
                    }
                    if (r >= cap)
                    {
                        ++out_of_range;
                        return;
                    }
                    if (owner[r].exchange(1) != 0)
                    {
                        ++clash;
                    }
                    std::this_thread::yield();
                    owner[r].store(0);
                    pmp::release_registered_rank();
                });
        }
        for (auto& t : wave)
        {
            t.join();
        }
    }
    expect(clash.load() == 0, "waves: no rank held by two threads at once");
    expect(out_of_range.load() == 0, "waves: every rank below the capacity");
    expect(reg.in_use() == 0, "waves: all given back");

    // The calling thread too, and release_registered_rank() before exit.
    const int mine = pmp::registered_rank();
    expect(mine == 0 && reg.in_use() == 1, "main thread: rank 0");
    pmp::release_registered_rank();
    expect(reg.in_use() == 0, "main thread: given back early");
}

/// The queries of the serial backend are those of one thread.
void check_queries()
{
    if constexpr (!pmp::threaded)
    {
        expect(pmp::max_threads() == 1 && pmp::team_rank() == 0 && pmp::team_size() == 1 && !pmp::in_parallel() &&
                   pmp::level() == 0,
               "serial backend: one thread");
    }
    else
    {
        expect(pmp::level() == 0 && !pmp::in_parallel() && pmp::team_rank() == 0, "outside any region");
    }
    const auto s = pmp::get_runtime_schedule();
    pmp::set_runtime_schedule(s);
    const auto s2 = pmp::get_runtime_schedule();
    expect(s.kind == s2.kind && s.chunk == s2.chunk, "runtime schedule round trip");
}
} // namespace

int main()
{
    check_sequential();
    check_concurrent_acquire();
    check_registered_rank();
    check_queries();
    std::println("rank_registry_validation: backend {}, {}",
                 pmp::backend == pmp::BackendKind::openmp ? "openmp" : "serial",
                 g_failures == 0 ? "all checks hold" : "FAILED");
    return g_failures == 0 ? 0 : 1;
}
