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
// Arrays driven from the application's own std::threads (pool backend,
// docs/threading_backends.md, step 7c). Variant-agnostic, like the other
// harnesses here.
// ============================================================================
//
// Under the pool backend no runtime numbers the application's threads: each
// one takes a rank from the registry (pmp::registered_rank()) at its first
// use of the library, and gives it back when it exits. The phase contract is
// the serial one: begin_parallel_phase() on every thread that writes, before
// its first write, then finalize_parallel_phase() ONCE, from one thread,
// after the writers stopped. Between the two, the application phase is open:
// in_parallel() is true everywhere and the library opens no team of the pool
// (its workers' ranks are the application threads'); after it, the flush and
// the compression pass run on teams of the pool.
//
// WHAT IT CHECKS, per element type (double, int, bool)
//
//   - fresh threads: every phase starts T new std::threads, which write their
//     block (one phase) or every T-th element (the next: the threads share
//     every chunk), a pattern of runs of the first constant, of the second
//     and of varied values; the main thread finalizes, and every other phase
//     runs compress() at the serial point. The ranks of the threads of a
//     phase are distinct and below the capacity, and the registry gives them
//     back when the threads exit (the writers of a phase meet on a latch
//     before writing, so they are all alive at once);
//   - long-lived threads: T threads started once, meeting on a std::barrier;
//     between two phases the main thread finalizes while they wait, so the
//     same threads begin a new phase each time (the phase guard re-arms
//     them);
//   - the main thread among the writers: it writes its share too, between
//     begin_parallel_phase() and finalize_parallel_phase();
//   - in_parallel() is true during the phase, on the writers and on the main
//     thread, and false after finalize_parallel_phase();
//   - work in the phase: every thread compresses its own array in the middle
//     of the phase (a pass on the calling thread: no team of the pool opens
//     during an application phase), rewrites part of it, and writes its share
//     of an array all the threads share, one write in 7 sent down the
//     deferred path (test_hooks::force_defer_every, check builds) -- the
//     buffers freed by those passes, the deferred queues of the registered
//     ranks, drained by finalize_parallel_phase().
//
//   After every phase the array is compared, element by element, with a
//   std::vector that received the same writes.
//
// Pool builds only (CMake registers it there): an OpenMP build numbers only
// OpenMP's threads, std::threads would all be rank 0; a serial build has one
// thread. Exit 0 = every element matches.

#include <algorithm>   // std::sort, std::adjacent_find
#include <atomic>      // std::atomic
#include <barrier>     // std::barrier (the long-lived threads)
#include <latch>       // std::latch (the writers of a phase alive together)
#include <thread>      // std::thread
#include <type_traits> // std::conditional_t
#include <vector>      // std::vector (the reference)

#include "chunk/chunk_traits.h"        // forward decls + default ind_type = int
#include "chunk/array_chunk.h"         // ArrayChunk
#include "chunk/array_chunk2.h"        // ArrayChunk2
#include "chunk/array_chunk2A.h"       // ArrayChunk2A
#include "chunk/param_chunk.h"         // ParamChunk::init_indexes
#include "chunk/chunk_std.h"           // release_chunk_pools, destroy_block_chunk
#include "chunk/config.h"              // chunk_config::use_chunk*
#include "chunk/PMP_CHUNK.h"           // pmp::*, CHUNK_PMP_*, get_thr_rank_fast, chunk_thread_capacity, the registry
#include "chunk/manage_ompt.h"         // begin_parallel_phase / finalize_parallel_phase
#include "chunk/internal/env_parse.h"  // arg_number
#include "chunk/internal/print.h"      // shrink_array::println
#include "chunk/internal/test_hooks.h" // force_defer_every (the in-phase case)

#include "validation_harness.h" // run() is called by validation_driver.cpp

namespace validation::std_thread_validation
{

using namespace shrink_array;

template <typename T>
using Array = std::conditional_t<chunk_config::use_chunk2a,
                                 ArrayChunk2A<T>,
                                 std::conditional_t<chunk_config::use_chunk2, ArrayChunk2<T>, ArrayChunk<T>>>;

constexpr const char* variant_name()
{
    if constexpr (chunk_config::use_chunk2a)
    {
        return "chunk2a";
    }
    else if constexpr (chunk_config::use_chunk2)
    {
        return "chunk2";
    }
    else
    {
        return "chunk";
    }
}

template <typename Arr, typename Val>
void maybe_set_second_value(Arr& a, const Val& v)
{
    if constexpr (requires(Arr x) { x.set_second_value(v); })
    {
        a.set_second_value(v);
    }
}

/// Runs of kRun elements: a whole number of chunks at every geometry built.
constexpr long kRun = 4096;

template <typename T>
struct Scenario
{
    T second;               ///< the second constant
    T (*varied)(long, int); ///< a varied value of element i at phase p
};

/// The value phase @p p writes into element @p i.
template <typename T>
T value_at(const Scenario<T>& sc, long i, int p)
{
    switch ((i / kRun + p) % 3)
    {
        case 0:
            return T(0);
        case 1:
            return sc.second;
        default:
            return sc.varied(i, p);
    }
}

/// Which elements thread @p t of @p nt writes in phase @p p: its block (even
/// phases) or every nt-th element (odd: every chunk shared by the threads).
template <typename F>
void for_share(long N, int t, int nt, int p, F&& f)
{
    if (p % 2 == 0)
    {
        const long b = N * t / nt;
        const long e = N * (t + 1) / nt;
        for (long i = b; i < e; ++i)
        {
            f(i);
        }
    }
    else
    {
        for (long i = t; i < N; i += nt)
        {
            f(i);
        }
    }
}

struct Failures
{
    std::atomic<int> n{0};
    void             fail()
    {
        n.fetch_add(1, std::memory_order_relaxed);
    }
};

/// Compare @p arr with @p ref; report the first mismatch.
template <typename T>
bool compare(const char* label, int p, const Array<T>& arr, const std::vector<T>& ref)
{
    long bad   = 0;
    long first = -1;
    for (long i = 0; i < static_cast<long>(ref.size()); ++i)
    {
        if (arr[i] != ref[static_cast<std::size_t>(i)])
        {
            if (first < 0)
            {
                first = i;
            }
            ++bad;
        }
    }
    if (bad != 0)
    {
        println("[{}/{}] FAIL after phase {}: {}/{} mismatches, first at i={} got={:g} expected={:g}", variant_name(),
                label, p, bad, ref.size(), first, static_cast<double>(arr[first]),
                static_cast<double>(ref[static_cast<std::size_t>(first)]));
    }
    return bad == 0;
}

/// What every writer checks on entering a phase: in_parallel() is true, its
/// rank is valid, and stable.
void check_writer(Failures& f, std::atomic<int>* rank_seen, int t)
{
    if (!CHUNK_PMP_IN_PARALLEL_REGION || !pmp::in_application_phase())
    {
        println("FAIL: a writer thread is not in the application phase after begin_parallel_phase()");
        f.fail();
    }
    const int r = get_thr_rank_fast();
    if (r < 0 || r >= chunk_thread_capacity() || r != pmp::thread_rank())
    {
        println("FAIL: writer rank {} (thread_rank {}, capacity {})", r, pmp::thread_rank(), chunk_thread_capacity());
        f.fail();
    }
    rank_seen[t].store(r, std::memory_order_relaxed);
}

/// The ranks of one phase's writers are distinct.
bool distinct_ranks(const std::atomic<int>* rank_seen, int nt)
{
    std::vector<int> r(static_cast<std::size_t>(nt));
    for (int t = 0; t < nt; ++t)
    {
        r[static_cast<std::size_t>(t)] = rank_seen[t].load(std::memory_order_relaxed);
    }
    std::sort(r.begin(), r.end());
    if (std::adjacent_find(r.begin(), r.end()) != r.end())
    {
        println("FAIL: two writers of the same phase share a rank");
        return false;
    }
    return true;
}

/// The main thread, after finalize_parallel_phase(): no phase, no parallel.
bool check_closed()
{
    if (CHUNK_PMP_IN_PARALLEL_REGION || pmp::in_application_phase())
    {
        println("FAIL: in_parallel() still true after finalize_parallel_phase()");
        return false;
    }
    return true;
}

/// New threads every phase; the main thread writes a share too.
template <typename T>
int fresh_threads(const char* label, long N, int nt, int phases, const Scenario<T>& sc)
{
    Array<T>       arr(N);
    std::vector<T> ref(static_cast<std::size_t>(N), T(0));
    maybe_set_second_value(arr, sc.second);

    Failures                      f;
    std::vector<std::atomic<int>> rank_seen(static_cast<std::size_t>(nt) + 1);
    int                           fails = 0;
    const int                     nw    = nt + 1; // the writers: nt threads and the main thread
    for (int p = 0; p < phases; ++p)
    {
        // All the writers alive at once before anyone writes: their ranks must
        // then be distinct (a thread that exits gives its rank back, and one
        // started after it may take it -- legitimately -- in the same phase).
        std::latch               together(nw);
        std::vector<std::thread> threads;
        threads.reserve(static_cast<std::size_t>(nt));
        for (int t = 0; t < nt; ++t)
        {
            threads.emplace_back(
                [&, t]
                {
                    begin_parallel_phase();
                    check_writer(f, rank_seen.data(), t);
                    together.arrive_and_wait();
                    for_share(N, t, nw, p, [&](long i) { arr[i] = value_at(sc, i, p); });
                });
        }
        begin_parallel_phase();
        check_writer(f, rank_seen.data(), nt);
        together.arrive_and_wait();
        for_share(N, nt, nw, p, [&](long i) { arr[i] = value_at(sc, i, p); });
        for (auto& th : threads)
        {
            th.join();
        }
        finalize_parallel_phase();
        if (!check_closed() || !distinct_ranks(rank_seen.data(), nw))
        {
            ++fails;
        }
        for (long i = 0; i < N; ++i)
        {
            ref[static_cast<std::size_t>(i)] = value_at(sc, i, p);
        }
        if (p % 2 == 1)
        {
            arr.compress(); // a serial point: the pass runs on a team of the pool
        }
        if (!compare(label, p, arr, ref))
        {
            ++fails;
            break;
        }
    }
    // The threads have exited: only the main thread holds a rank now.
    if (::fast_alloc::pmp::thread_rank_registry().in_use() != 1)
    {
        println("FAIL: {} ranks still taken after the threads exited (expected 1, the main thread's)",
                ::fast_alloc::pmp::thread_rank_registry().in_use());
        ++fails;
    }
    fails += f.n.load();
    println("[{}/{}] fresh threads x{} + main, {} phases: {}", variant_name(), label, nt, phases,
            fails == 0 ? "OK" : "FAIL");
    arr.clear();
    return fails == 0 ? 0 : 1;
}

/// The same threads for every phase, meeting on a barrier.
template <typename T>
int long_lived_threads(const char* label, long N, int nt, int phases, const Scenario<T>& sc)
{
    Array<T>       arr(N);
    std::vector<T> ref(static_cast<std::size_t>(N), T(0));
    maybe_set_second_value(arr, sc.second);

    Failures                      f;
    std::vector<std::atomic<int>> rank_seen(static_cast<std::size_t>(nt));
    std::barrier<>                sync(nt + 1);
    int                           fails = 0;

    std::vector<std::thread> threads;
    threads.reserve(static_cast<std::size_t>(nt));
    for (int t = 0; t < nt; ++t)
    {
        threads.emplace_back(
            [&, t]
            {
                for (int p = 0; p < phases; ++p)
                {
                    sync.arrive_and_wait(); // the phase starts
                    begin_parallel_phase();
                    check_writer(f, rank_seen.data(), t);
                    for_share(N, t, nt, p, [&](long i) { arr[i] = value_at(sc, i, p); });
                    sync.arrive_and_wait(); // the writes are done
                    sync.arrive_and_wait(); // the main thread finalized and checked
                }
            });
    }
    for (int p = 0; p < phases; ++p)
    {
        sync.arrive_and_wait();
        sync.arrive_and_wait();
        finalize_parallel_phase();
        if (!check_closed() || !distinct_ranks(rank_seen.data(), nt))
        {
            ++fails;
        }
        for (long i = 0; i < N; ++i)
        {
            ref[static_cast<std::size_t>(i)] = value_at(sc, i, p);
        }
        if (p % 2 == 0)
        {
            arr.compress();
        }
        const bool ok = compare(label, p, arr, ref);
        if (!ok)
        {
            ++fails;
        }
        sync.arrive_and_wait();
    }
    for (auto& th : threads)
    {
        th.join();
    }
    fails += f.n.load();
    println("[{}/{}] long-lived threads x{}, {} phases: {}", variant_name(), label, nt, phases,
            fails == 0 ? "OK" : "FAIL");
    arr.clear();
    return fails == 0 ? 0 : 1;
}

/// Compression, rewrites and deferred writes inside the phase.
template <typename T>
int in_phase_work(const char* label, long N, int nt, int phases, const Scenario<T>& sc)
{
    const long                  n_own = N / 4; // each thread's own array
    std::vector<Array<T>>       own;
    std::vector<std::vector<T>> own_ref(static_cast<std::size_t>(nt),
                                        std::vector<T>(static_cast<std::size_t>(n_own), T(0)));
    own.reserve(static_cast<std::size_t>(nt));
    for (int t = 0; t < nt; ++t)
    {
        own.emplace_back(n_own);
        maybe_set_second_value(own.back(), sc.second);
    }
    Array<T>       shared(N);
    std::vector<T> ref(static_cast<std::size_t>(N), T(0));
    maybe_set_second_value(shared, sc.second);

    if constexpr (test_hooks::force_defer_compiled)
    {
        test_hooks::force_defer_every = 7U;
    }
    const unsigned long deferred0 = test_hooks::forced_deferrals.load();

    Failures f;
    int      fails = 0;
    for (int p = 0; p < phases; ++p)
    {
        std::latch               together(nt);
        std::vector<std::thread> threads;
        threads.reserve(static_cast<std::size_t>(nt));
        for (int t = 0; t < nt; ++t)
        {
            threads.emplace_back(
                [&, t]
                {
                    begin_parallel_phase();
                    together.arrive_and_wait();
                    auto& a = own[static_cast<std::size_t>(t)];
                    for (long i = 0; i < n_own; ++i)
                    {
                        a[i] = value_at(sc, i, p);
                    }
                    for_share(N, t, nt, p, [&](long i) { shared[i] = value_at(sc, i, p); });
                    a.compress(); // in the phase: on this thread, no team
                    if (CHUNK_PMP_NUM_THREADS != 1)
                    {
                        println("FAIL: a pass in the application phase ran on a team of {}", CHUNK_PMP_NUM_THREADS);
                        f.fail();
                    }
                    for (long i = 0; i < n_own / 3; ++i)
                    {
                        a[i] = value_at(sc, i + kRun, p + 1); // re-materialise part of what the pass folded
                    }
                    // Every 5th element of the SAME share, rewritten (the thread's own
                    // elements: two threads never write one element in a phase).
                    for_share(N, t, nt, p,
                              [&](long i)
                              {
                                  if (i % 5 == 0)
                                  {
                                      shared[i] = value_at(sc, i, p + 1);
                                  }
                              });
                });
        }
        for (auto& th : threads)
        {
            th.join();
        }
        finalize_parallel_phase();
        if (!check_closed())
        {
            ++fails;
        }
        // The reference: the second share overwrites every 5th element.
        for (long i = 0; i < N; ++i)
        {
            ref[static_cast<std::size_t>(i)] = (i % 5 == 0) ? value_at(sc, i, p + 1) : value_at(sc, i, p);
        }
        bool ok = compare(label, p, shared, ref);
        for (int t = 0; t < nt && ok; ++t)
        {
            auto& r = own_ref[static_cast<std::size_t>(t)];
            for (long i = 0; i < n_own; ++i)
            {
                r[static_cast<std::size_t>(i)] = i < n_own / 3 ? value_at(sc, i + kRun, p + 1) : value_at(sc, i, p);
            }
            ok = compare(label, p, own[static_cast<std::size_t>(t)], r);
        }
        if (!ok)
        {
            ++fails;
            break;
        }
        if (p % 2 == 0)
        {
            shared.compress(); // and a pass at the serial point, on the pool
        }
    }
    if constexpr (test_hooks::force_defer_compiled)
    {
        test_hooks::force_defer_every = 0U;
    }
    const unsigned long deferred = test_hooks::forced_deferrals.load() - deferred0;
    fails += f.n.load();
    println("[{}/{}] in-phase passes and deferrals x{}, {} phases, {} writes deferred: {}", variant_name(), label, nt,
            phases, deferred, fails == 0 ? "OK" : "FAIL");
    for (auto& a : own)
    {
        a.clear();
    }
    shared.clear();
    return fails == 0 ? 0 : 1;
}

int run(int argc, char** argv)
{
    const long N      = argc > 1 ? shrink_array::arg_number<long>(argv[1], 1L << 18, "N") : (1L << 18);
    const int  phases = argc > 2 ? shrink_array::arg_number<int>(argv[2], 6, "phases") : 6;

    ParamChunk::init_indexes(true);

    // Writers: the backend's team size (OMP_NUM_THREADS of the test), so the
    // thread ranks reach as far as the pool's workers do -- and below the
    // capacity, with the main thread.
    const int nt = std::max(1, std::min(CHUNK_PMP_MAX_THREADS, chunk_thread_capacity() - 1));

    println("=== std_thread_validation [variant = {}]  N={}  threads={}  phases={}  capacity={} ===", variant_name(), N,
            nt, phases, chunk_thread_capacity());

    if constexpr (pmp::backend != pmp::backend_kind::pool)
    {
        println("not a pool build: skipped");
        return 77;
    }

    // The main thread's rank, before any writer: a registered one, outside any team.
    if (get_thr_rank_fast() != pmp::thread_rank() || CHUNK_PMP_IN_PARALLEL_REGION)
    {
        println("FAIL: the main thread outside any team: rank {} / thread_rank {}, in_parallel {}", get_thr_rank_fast(),
                pmp::thread_rank(), CHUNK_PMP_IN_PARALLEL_REGION);
        return 1;
    }

    int fails = 0;

    const Scenario<double> sd{3.5, [](long i, int p) { return 1.0 + static_cast<double>(i) + (0.25 * p); }};
    const Scenario<int>    si{7, [](long i, int p) { return 100 + static_cast<int>(i % 100000) + p; }};
    const Scenario<bool>   sb{true, [](long i, int p) { return ((i + p) % 3) == 0; }};

    fails += fresh_threads<double>("double", N, nt, phases, sd);
    fails += fresh_threads<int>("int", N, nt, phases, si);
    fails += fresh_threads<bool>("bool", N, nt, phases, sb);
    fails += long_lived_threads<double>("double", N, nt, phases, sd);
    fails += long_lived_threads<int>("int", N, nt, phases, si);
    fails += long_lived_threads<bool>("bool", N, nt, phases, sb);
    fails += in_phase_work<double>("double", N, nt, phases, sd);
    fails += in_phase_work<int>("int", N, nt, phases, si);
    fails += in_phase_work<bool>("bool", N, nt, phases, sb);

    release_chunk_pools();
    destroy_block_chunk();

    println("\n[{}] {}", variant_name(), fails == 0 ? "=== STD::THREAD OK ===" : "=== STD::THREAD FAILURES ===");
    return fails == 0 ? 0 : 1;
}

} // namespace validation::std_thread_validation
