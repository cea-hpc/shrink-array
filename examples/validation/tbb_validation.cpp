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
// Arrays driven from a TBB application through shrink_array::tbb::parallel_for
// (chunk/tbb_adaptor.h, docs/threading_backends.md, step 8). Variant-agnostic,
// like the other harnesses here.
// ============================================================================
//
// The adaptor runs oneapi::tbb::parallel_for with begin_parallel_phase() before
// every piece of the range and finalize_parallel_phase() once after the tasks.
// Under the std backend each TBB thread takes a rank of the registry; under
// the serial backend the adaptor runs the body on the calling thread.
//
// WHAT IT CHECKS, per element type (double, int, bool), in a task_arena of T
// threads (T: the backend's team size, OMP_NUM_THREADS of the test)
//
//   - the range form with each partitioner: auto (the default), simple with a
//     small grain (many pieces per thread: begin_parallel_phase() again and
//     again on one thread), static (the same piece on the same thread from
//     one call to the next); the pieces write runs of the first constant, of
//     the second and varied values, every other phase compress() at the
//     serial point;
//   - the index form, step 1 and step 3 (the elements the step skips keep the
//     previous phase's values);
//   - in every piece: in_parallel() and the application phase are on, the
//     rank is the thread's (get_thr_rank_fast() == pmp::thread_rank()), below
//     the capacity, the same for a thread all phase long and distinct between
//     threads; after the call: no phase, no parallel;
//   - work in the phase: one index per array (grain 1), each compresses its
//     array in the middle of its piece (a pass on the calling thread: no team
//     of the std backend in the phase), rewrites part of it and writes its share of
//     a shared array, one write in 7 deferred (test_hooks::force_defer_every,
//     check builds) -- drained by the adaptor's finalize_parallel_phase();
//   - a nested oneapi::tbb::parallel_for inside the body, with phase_body();
//   - a body that throws: the exception reaches the caller, the phase is
//     closed, and the next call runs normally; a step of 0 is refused
//     (std::invalid_argument, as TBB) before any phase opens;
//   - a narrower task_arena (2 threads) for one call.
//
//   After every phase the array is compared, element by element, with a
//   std::vector that received the same writes.
//
// Std and serial builds with CHUNK_WITH_TBB=ON (CMake registers it there):
// an OpenMP build refuses the adaptor. Exit 0 = every element matches.

#include <algorithm>   // std::max, std::min
#include <atomic>      // std::atomic
#include <map>         // std::map (thread -> rank)
#include <mutex>       // std::mutex, std::scoped_lock
#include <stdexcept>   // std::runtime_error (the throwing body), std::invalid_argument (step 0)
#include <thread>      // std::this_thread::get_id
#include <type_traits> // std::conditional_t
#include <vector>      // std::vector (the reference)

#include <oneapi/tbb/blocked_range.h>  // oneapi::tbb::blocked_range
#include <oneapi/tbb/global_control.h> // oneapi::tbb::global_control (max_allowed_parallelism)
#include <oneapi/tbb/parallel_for.h>   // oneapi::tbb::parallel_for (nested), the partitioners
#include <oneapi/tbb/task_arena.h>     // oneapi::tbb::task_arena

#include "chunk/chunk_traits.h"        // forward decls + default IntType = int
#include "chunk/zero_array.h"         // zero::Array
#include "chunk/zero_one_value_array.h"        // zero_one_value::Array
#include "chunk/zero_any_array.h"       // zero_any::Array
#include "chunk/param_chunk.h"         // ParamChunk::init_indexes
#include "chunk/chunk_std.h"           // release_chunk_pools, destroy_block_chunk
#include "chunk/config.h"              // chunk_config::use_chunk*
#include "chunk/PMP_CHUNK.h"           // pmp::*, CHUNK_PMP_* (THREAD_CAPACITY: the registry), get_thr_rank_fast
#include "chunk/tbb_adaptor.h"         // shrink_array::tbb::parallel_for, phase_body
#include "chunk/internal/env_parse.h"  // arg_number
#include "chunk/internal/print.h"      // shrink_array::println
#include "chunk/internal/test_hooks.h" // force_defer_every (the in-phase case)

#include "validation_harness.h" // run() is called by validation_driver.cpp

namespace validation::tbb_validation
{

using namespace shrink_array;
using oneapi::tbb::blocked_range;

template <typename T>
using Array =
    std::conditional_t<chunk_config::use_chunk2a,
                       zero_any::Array<T>,
                       std::conditional_t<chunk_config::use_chunk2, zero_one_value::Array<T>, zero::Array<T>>>;

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

constexpr bool std_build = pmp::backend == pmp::BackendKind::std_threads;

template <typename Arr, typename Val>
void maybe_set_second_value(Arr& a, const Val& v)
{
    if constexpr (requires(Arr x) { x.set_second_value(v); })
    {
        a.set_second_value(v);
    }
}

/// Runs of run_len elements: a whole number of chunks at every geometry built.
constexpr long run_len = 4096;

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
    switch ((i / run_len + p) % 3)
    {
        case 0:
            return T(0);
        case 1:
            return sc.second;
        default:
            return sc.varied(i, p);
    }
}

/// Compare @p arr with @p ref; report the first mismatch.
template <typename T>
bool compare(const char* what, const char* label, int p, const Array<T>& arr, const std::vector<T>& ref)
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
        println("[{}/{}] {}: FAIL after phase {}: {}/{} mismatches, first at i={} got={:g} expected={:g}",
                variant_name(), label, what, p, bad, ref.size(), first, static_cast<double>(arr[first]),
                static_cast<double>(ref[static_cast<std::size_t>(first)]));
    }
    return bad == 0;
}

/**
 * What the pieces of one phase saw: every thread's rank (the same all phase,
 * distinct between threads), and the failures of the in-piece checks.
 */
class PhaseRecord
{
  public:
    /// In a piece, after the adaptor's begin_parallel_phase().
    void piece()
    {
        if constexpr (std_build)
        {
            if (!CHUNK_PMP_IN_PARALLEL_REGION || !pmp::in_application_phase())
            {
                fail("a piece runs outside the application phase");
            }
        }
        const int r = get_thr_rank_fast();
        if (r < 0 || r >= CHUNK_PMP_THREAD_CAPACITY || r != pmp::thread_rank())
        {
            println("FAIL: piece rank {} (thread_rank {}, capacity {})", r, pmp::thread_rank(),
                    CHUNK_PMP_THREAD_CAPACITY);
            fails_.fetch_add(1, std::memory_order_relaxed);
        }
        const std::scoped_lock lock(mutex_);
        const auto [it, inserted] = rank_of_.try_emplace(std::this_thread::get_id(), r);
        if (!inserted && it->second != r)
        {
            println("FAIL: one thread with ranks {} and {} in the same phase", it->second, r);
            fails_.fetch_add(1, std::memory_order_relaxed);
        }
    }

    /// After the adaptor returned: no phase, distinct ranks; the failures.
    [[nodiscard]] int close()
    {
        if (CHUNK_PMP_IN_PARALLEL_REGION || pmp::in_application_phase())
        {
            fail("in_parallel() still true after the adaptor returned");
        }
        std::vector<int> ranks;
        for (const auto& [tid, r] : rank_of_)
        {
            ranks.push_back(r);
        }
        std::sort(ranks.begin(), ranks.end());
        if (std::adjacent_find(ranks.begin(), ranks.end()) != ranks.end())
        {
            fail("two threads of the same phase share a rank");
        }
        threads_ = static_cast<int>(rank_of_.size());
        rank_of_.clear();
        return fails_.exchange(0);
    }

    /// Threads that ran a piece of the last closed phase.
    [[nodiscard]] int threads() const noexcept
    {
        return threads_;
    }

    void fail(const char* msg)
    {
        println("FAIL: {}", msg);
        fails_.fetch_add(1, std::memory_order_relaxed);
    }

  private:
    std::mutex                     mutex_;
    std::map<std::thread::id, int> rank_of_;
    std::atomic<int>               fails_{0};
    int                            threads_ = 0;
};

enum class Part
{
    automatic,
    simple,
    fixed, ///< static_partitioner
};

constexpr const char* part_name(Part k)
{
    switch (k)
    {
        case Part::automatic:
            return "auto";
        case Part::simple:
            return "simple";
        case Part::fixed:
            return "static";
    }
    return "?";
}

/// The range form, with partitioner @p k.
template <typename T>
int range_form(const char* label, long N, int phases, Part k, const Scenario<T>& sc)
{
    Array<T>       arr(N);
    std::vector<T> ref(static_cast<std::size_t>(N), T(0));
    maybe_set_second_value(arr, sc.second);
    PhaseRecord rec;
    int         fails       = 0;
    int         max_threads = 0;
    for (int p = 0; p < phases; ++p)
    {
        auto body = [&](const blocked_range<long>& r)
        {
            rec.piece();
            for (long i = r.begin(); i != r.end(); ++i)
            {
                arr[i] = value_at(sc, i, p);
            }
        };
        switch (k)
        {
            case Part::automatic:
                shrink_array::tbb::parallel_for(blocked_range<long>(0, N), body);
                break;
            case Part::simple:
                shrink_array::tbb::parallel_for(blocked_range<long>(0, N, run_len / 4), body,
                                                oneapi::tbb::simple_partitioner{});
                break;
            case Part::fixed:
                shrink_array::tbb::parallel_for(blocked_range<long>(0, N), body, oneapi::tbb::static_partitioner{});
                break;
        }
        fails += rec.close();
        max_threads = std::max(max_threads, rec.threads());
        for (long i = 0; i < N; ++i)
        {
            ref[static_cast<std::size_t>(i)] = value_at(sc, i, p);
        }
        if (p % 2 == 1)
        {
            arr.compress(); // a serial point: the pass runs on a team of the std backend
        }
        if (!compare("range", label, p, arr, ref))
        {
            ++fails;
            break;
        }
    }
    println("[{}/{}] range form, {} partitioner, {} phases, up to {} threads: {}", variant_name(), label, part_name(k),
            phases, max_threads, fails == 0 ? "OK" : "FAIL");
    arr.clear();
    return fails == 0 ? 0 : 1;
}

/// The index form, step @p step: the skipped elements keep their values.
template <typename T>
int index_form(const char* label, long N, int phases, long step, const Scenario<T>& sc)
{
    Array<T>       arr(N);
    std::vector<T> ref(static_cast<std::size_t>(N), T(0));
    maybe_set_second_value(arr, sc.second);
    PhaseRecord rec;
    int         fails = 0;
    for (int p = 0; p < phases; ++p)
    {
        // The piece check once per index here: the index form has no piece hook.
        auto f = [&](long i)
        {
            if (i % run_len == 0)
            {
                rec.piece();
            }
            arr[i] = value_at(sc, i, p);
        };
        const long first = p % step; // a different subset each phase when step > 1
        if (step == 1)
        {
            shrink_array::tbb::parallel_for(first, N, f);
        }
        else
        {
            shrink_array::tbb::parallel_for(first, N, step, f, oneapi::tbb::static_partitioner{});
        }
        fails += rec.close();
        for (long i = first; i < N; i += step)
        {
            ref[static_cast<std::size_t>(i)] = value_at(sc, i, p);
        }
        if (p % 2 == 0)
        {
            arr.compress();
        }
        if (!compare("index", label, p, arr, ref))
        {
            ++fails;
            break;
        }
    }
    println("[{}/{}] index form, step {}, {} phases: {}", variant_name(), label, step, phases,
            fails == 0 ? "OK" : "FAIL");
    arr.clear();
    return fails == 0 ? 0 : 1;
}

/// Compression, rewrites and deferred writes inside the phase: one index per array.
template <typename T>
int in_phase_work(const char* label, long N, int nt, int phases, const Scenario<T>& sc)
{
    const int                   na    = std::max(2, nt);
    const long                  n_own = N / 4;
    std::vector<Array<T>>       own;
    std::vector<std::vector<T>> own_ref(static_cast<std::size_t>(na),
                                        std::vector<T>(static_cast<std::size_t>(n_own), T(0)));
    own.reserve(static_cast<std::size_t>(na));
    for (int t = 0; t < na; ++t)
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

    PhaseRecord rec;
    int         fails = 0;
    for (int p = 0; p < phases; ++p)
    {
        shrink_array::tbb::parallel_for(
            blocked_range<int>(0, na, 1),
            [&](const blocked_range<int>& r)
            {
                rec.piece();
                for (int t = r.begin(); t != r.end(); ++t)
                {
                    auto& a = own[static_cast<std::size_t>(t)];
                    for (long i = 0; i < n_own; ++i)
                    {
                        a[i] = value_at(sc, i, p);
                    }
                    const long b = N * t / na;
                    const long e = N * (t + 1) / na;
                    for (long i = b; i < e; ++i)
                    {
                        shared[i] = value_at(sc, i, p);
                    }
                    a.compress(); // in the phase: on this thread, no team
                    if (CHUNK_PMP_NUM_THREADS != 1)
                    {
                        rec.fail("a pass in the application phase ran on a team");
                    }
                    for (long i = 0; i < n_own / 3; ++i)
                    {
                        a[i] = value_at(sc, i + run_len, p + 1);
                    }
                    for (long i = b; i < e; i += 5)
                    {
                        shared[i] = value_at(sc, i, p + 1);
                    }
                }
            },
            oneapi::tbb::simple_partitioner{});
        fails += rec.close();
        // The reference: each share, then every 5th element of it from its start.
        for (int t = 0; t < na; ++t)
        {
            const long b = N * t / na;
            const long e = N * (t + 1) / na;
            for (long i = b; i < e; ++i)
            {
                ref[static_cast<std::size_t>(i)] = ((i - b) % 5 == 0) ? value_at(sc, i, p + 1) : value_at(sc, i, p);
            }
        }
        bool ok = compare("in-phase shared", label, p, shared, ref);
        for (int t = 0; t < na && ok; ++t)
        {
            auto& r = own_ref[static_cast<std::size_t>(t)];
            for (long i = 0; i < n_own; ++i)
            {
                r[static_cast<std::size_t>(i)] = i < n_own / 3 ? value_at(sc, i + run_len, p + 1) : value_at(sc, i, p);
            }
            ok = compare("in-phase own", label, p, own[static_cast<std::size_t>(t)], r);
        }
        if (!ok)
        {
            ++fails;
            break;
        }
        if (p % 2 == 0)
        {
            shared.compress();
        }
    }
    if constexpr (test_hooks::force_defer_compiled)
    {
        test_hooks::force_defer_every = 0U;
    }
    const unsigned long deferred = test_hooks::forced_deferrals.load() - deferred0;
    println("[{}/{}] in-phase passes and deferrals x{}, {} phases, {} writes deferred: {}", variant_name(), label, na,
            phases, deferred, fails == 0 ? "OK" : "FAIL");
    for (auto& a : own)
    {
        a.clear();
    }
    shared.clear();
    return fails == 0 ? 0 : 1;
}

/// A oneapi::tbb::parallel_for nested in the adaptor's body, with phase_body().
template <typename T>
int nested(const char* label, long N, int phases, const Scenario<T>& sc)
{
    Array<T>       arr(N);
    std::vector<T> ref(static_cast<std::size_t>(N), T(0));
    maybe_set_second_value(arr, sc.second);
    PhaseRecord rec;
    int         fails  = 0;
    const long  blocks = 4;
    for (int p = 0; p < phases; ++p)
    {
        auto inner = [&](const blocked_range<long>& r)
        {
            rec.piece();
            for (long i = r.begin(); i != r.end(); ++i)
            {
                arr[i] = value_at(sc, i, p);
            }
        };
        shrink_array::tbb::parallel_for(blocked_range<long>(0, blocks, 1),
                                        [&](const blocked_range<long>& outer)
                                        {
                                            for (long b = outer.begin(); b != outer.end(); ++b)
                                            {
                                                oneapi::tbb::parallel_for(
                                                    blocked_range<long>(N * b / blocks, N * (b + 1) / blocks),
                                                    shrink_array::tbb::phase_body(inner));
                                            }
                                        });
        fails += rec.close();
        for (long i = 0; i < N; ++i)
        {
            ref[static_cast<std::size_t>(i)] = value_at(sc, i, p);
        }
        if (p % 2 == 1)
        {
            arr.compress();
        }
        if (!compare("nested", label, p, arr, ref))
        {
            ++fails;
            break;
        }
    }
    println("[{}/{}] nested parallel_for, {} phases: {}", variant_name(), label, phases, fails == 0 ? "OK" : "FAIL");
    arr.clear();
    return fails == 0 ? 0 : 1;
}

/// A body that throws: the exception reaches the caller, the phase is closed.
template <typename T>
int throwing_body(const char* label, long N, const Scenario<T>& sc)
{
    Array<T>       arr(N);
    std::vector<T> ref(static_cast<std::size_t>(N), T(0));
    maybe_set_second_value(arr, sc.second);
    PhaseRecord rec;
    int         fails  = 0;
    bool        caught = false;
    try
    {
        shrink_array::tbb::parallel_for(blocked_range<long>(0, N, run_len),
                                        [&](const blocked_range<long>& r)
                                        {
                                            for (long i = r.begin(); i != r.end(); ++i)
                                            {
                                                arr[i] = value_at(sc, i, 0);
                                            }
                                            if (r.begin() <= N / 2 && N / 2 < r.end())
                                            {
                                                throw std::runtime_error("thrown by one piece");
                                            }
                                        });
    }
    catch (const std::runtime_error&)
    {
        caught = true;
    }
    if (!caught)
    {
        rec.fail("the body's exception did not reach the caller");
    }
    fails += rec.close();
    // A step that is not positive: refused before any task, as TBB does.
    bool refused = false;
    try
    {
        shrink_array::tbb::parallel_for(0L, N, 0L, [&](long i) { arr[i] = T(0); });
    }
    catch (const std::invalid_argument&)
    {
        refused = true;
    }
    if (!refused)
    {
        rec.fail("a step of 0 was not refused");
    }
    fails += rec.close();
    // The next phase runs normally and leaves every element known.
    shrink_array::tbb::parallel_for(blocked_range<long>(0, N),
                                    [&](const blocked_range<long>& r)
                                    {
                                        rec.piece();
                                        for (long i = r.begin(); i != r.end(); ++i)
                                        {
                                            arr[i] = value_at(sc, i, 1);
                                        }
                                    });
    fails += rec.close();
    for (long i = 0; i < N; ++i)
    {
        ref[static_cast<std::size_t>(i)] = value_at(sc, i, 1);
    }
    arr.compress();
    if (!compare("throw", label, 1, arr, ref))
    {
        ++fails;
    }
    println("[{}/{}] throwing body: {}", variant_name(), label, fails == 0 ? "OK" : "FAIL");
    arr.clear();
    return fails == 0 ? 0 : 1;
}

int run(int argc, char** argv)
{
    const long N      = argc > 1 ? shrink_array::arg_number<long>(argv[1], 1L << 18, "N") : (1L << 18);
    const int  phases = argc > 2 ? shrink_array::arg_number<int>(argv[2], 6, "phases") : 6;

    ParamChunk::init_indexes(true);

    // TBB's threads: the backend's team size (OMP_NUM_THREADS of the test),
    // the calling thread among them -- within the capacity of the registry.
    const int nt = std_build ? std::max(1, std::min(CHUNK_PMP_MAX_THREADS, CHUNK_PMP_THREAD_CAPACITY)) : 1;

    println("=== tbb_validation [variant = {}, backend = {}]  N={}  threads={}  phases={}  capacity={} ===",
            variant_name(), std_build ? "std" : "serial", N, nt, phases, CHUNK_PMP_THREAD_CAPACITY);

    const oneapi::tbb::global_control limit(oneapi::tbb::global_control::max_allowed_parallelism,
                                            static_cast<std::size_t>(nt));
    oneapi::tbb::task_arena           arena(nt);

    int fails = 0;

    const Scenario<double> sd{3.5, [](long i, int p) { return 1.0 + static_cast<double>(i) + (0.25 * p); }};
    const Scenario<int>    si{7, [](long i, int p) { return 100 + static_cast<int>(i % 100000) + p; }};
    const Scenario<bool>   sb{true, [](long i, int p) { return ((i + p) % 3) == 0; }};

    arena.execute(
        [&]
        {
            for (const Part k : {Part::automatic, Part::simple, Part::fixed})
            {
                fails += range_form<double>("double", N, phases, k, sd);
                fails += range_form<int>("int", N, phases, k, si);
                fails += range_form<bool>("bool", N, phases, k, sb);
            }
            for (const long step : {1L, 3L})
            {
                fails += index_form<double>("double", N, phases, step, sd);
                fails += index_form<int>("int", N, phases, step, si);
                fails += index_form<bool>("bool", N, phases, step, sb);
            }
            fails += in_phase_work<double>("double", N, nt, phases, sd);
            fails += in_phase_work<int>("int", N, nt, phases, si);
            fails += in_phase_work<bool>("bool", N, nt, phases, sb);
            fails += nested<double>("double", N, phases, sd);
            fails += nested<int>("int", N, phases, si);
            fails += throwing_body<double>("double", N, sd);
        });

    // A narrower arena for one call: its threads are TBB's same workers.
    oneapi::tbb::task_arena narrow(std::min(nt, 2));
    narrow.execute([&] { fails += range_form<double>("double, arena of 2", N, phases, Part::fixed, sd); });

    if constexpr (std_build)
    {
        const auto& reg = ::fast_alloc::pmp::thread_rank_registry();
        if (reg.high_water() > reg.capacity())
        {
            println("FAIL: registry high water {} past its capacity {}", reg.high_water(), reg.capacity());
            ++fails;
        }
    }

    release_chunk_pools();
    destroy_block_chunk();

    println("\n[{}] {}", variant_name(), fails == 0 ? "=== TBB OK ===" : "=== TBB FAILURES ===");
    return fails == 0 ? 0 : 1;
}

} // namespace validation::tbb_validation
