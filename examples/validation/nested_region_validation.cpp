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
// Nested-region harness. Variant-agnostic: chunk / chunk2 / chunk2a by the
// compile-time macro (CHUNK_VARIANT_V1 / _V2 / _V2A), like the other harnesses.
// ============================================================================
//
// Guards ONE property: a thread keeps its own per-rank structures across a
// nested parallel region.
//
// The library indexes its per-thread structures (candidate lists, deferred-op
// queues) with a rank cached in thread-local storage (__fast_omp_rank): the
// thread's rank in the active team (omp_get_ancestor_thread_num when nested),
// cleared at the end of each implicit task. Caching omp_get_thread_num() --
// the rank in the INNERMOST team -- would bring a thread of the outer team
// back from a nested region, even a serialised one, with rank 0, sharing rank
// 0's structures unprotected. Two ACTIVE levels are an invariant failure.
//
// And a second one: the end of a nested region is not the end of the phase.
// With OMPT the runtime reports every region's end, the nested ones too; the
// library's phase end (the serial flush of every rank's deferred writes, the
// compression of the candidates) must run at the OUTERMOST one only, or it
// drains the queues of the outer team's threads while they are still writing.
// Without OMPT the same holds for a parallel_phase() opened inside a phase --
// a component's, a solver's: its finalize_parallel_phase() is the enclosing
// phase's business.
//
// What it does:
//   A. max-active-levels 1: an outer team of 4 writes the arrays; every thread
//      opens a nested region (serialised by that setting), writes in it, and
//      goes on writing in the outer region after it. Checked against a
//      std::vector; check builds also check each cached rank against the
//      thread's rank in the active team.
//   C. A with deferred writes queued when the nested regions end: one write in
//      7 forced down the deferred path (test_hooks.h, check builds). Before the
//      fix: writes lost ("deferred ops still queued"), a crash, or wrong values.
//   D. C with the nested regions opened by parallel_phase(2, ...), as a
//      component that knows nothing of the enclosing phase would.
//   B. re-runs itself with `--nested-active`: max-active-levels 2 and a
//      2-thread inner team writing the arrays, which must stop with the
//      "nested ACTIVE" invariant failure.
// In a serial build there is no nesting: A, C and D run single-threaded, B is
// skipped.
//
// The outer team is num_threads(4) whatever OMP_NUM_THREADS says: where the job
// sees fewer than 4 cores, run it with CHUNK_THREAD_CAPACITY=4 (the ctest
// entries do), or the per-thread capacity refuses rank 3 -- rightly.
// ============================================================================

#include <cstdio>      // std::fopen, std::fclose, std::fread, std::fflush, FILE, stdout, std::remove
#include <cstdlib>     // std::system
#include <string>      // std::string, std::to_string
#include <type_traits> // std::conditional_t
#include <vector>      // std::vector
#include <unistd.h>    // getpid
#include <sys/wait.h>  // WIFEXITED, WEXITSTATUS

#include "chunk/chunk_traits.h"        // forward decls + default IndT = int
#include "chunk/array_chunk.h"         // zero::Array
#include "chunk/array_chunk2.h"        // zero_one_value::Array
#include "chunk/array_chunk2A.h"       // zero_any::Array
#include "chunk/param_chunk.h"         // ParamChunk::init_indexes
#include "chunk/config.h"              // chunk_config::use_chunk2*, build_config::use_openmp / has_not_ompt
#include "chunk/PMP_CHUNK.h"           // chunk_pmp_set_max_active_levels
#include "chunk/manage_ompt.h"         // begin_parallel_phase / finalize_parallel_phase
#include "chunk/internal/print.h"      // shrink_array::println
#include "chunk/internal/test_hooks.h" // force_defer_every, forced_deferrals (C, D)

#include "chunk/chunk_std.h"    // release_chunk_pools, destroy_block_chunk
#include "validation_harness.h" // self_command; run() is called by validation_driver.cpp

namespace validation::nested_region_validation
{

using namespace shrink_array;

static void begin_phase()
{
    if constexpr (build_config::has_not_ompt)
    {
        shrink_array::begin_parallel_phase();
    }
}
static void end_phase()
{
    if constexpr (build_config::has_not_ompt)
    {
        shrink_array::finalize_parallel_phase();
    }
}

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

/// The value element i holds after round `round` (zero runs of 2048 so the
/// chunks alternate constant and materialised).
static int expected_before_inner(long i, int round)
{
    return ((i / 2048) + round) % 3 == 0 ? 0 : static_cast<int>(((i * 5) + round) % 89) + 1;
}

/// The nested region's writes: one thread of the inner team adds 3 to every
/// 7th element of block @p b. A function of its own: an orphaned `omp single`,
/// so the compiler does not see it lexically inside the outer `omp for`.
static void inner_writes(Array<int>& a, long b)
{
#pragma omp single
    for (long i = b * 1024; i < (b + 1) * 1024; ++i)
    {
        if (i % 7 == 0)
        {
            a[i] += 3;
        }
    }
}

/// How the nested regions are opened.
enum class Inner
{
    pragma,        ///< `#pragma omp parallel num_threads(2)`, begin_phase() in it (A, B, C)
    parallel_phase ///< shrink_array::parallel_phase(2, ...) (D)
};

/// Scenario A (and B, C, D's body): each thread of the outer team writes its
/// share, opens a nested region that writes again, then writes after it.
static long run_nested(Array<int>& a, std::vector<int>& ref, int rounds, Inner inner)
{
    const long n = static_cast<long>(ref.size());
    for (int round = 0; round < rounds; ++round)
    {
#pragma omp parallel num_threads(4)
        {
            begin_phase();
#pragma omp for schedule(static)
            for (long i = 0; i < n; ++i)
            {
                a[i] = expected_before_inner(i, round);
            }

            // A nested region per thread, over this thread's own slice (the
            // static schedule gives the same slice as above).
#pragma omp for schedule(static)
            for (long b = 0; b < n / 1024; ++b)
            {
                const auto inner_body = [&] { inner_writes(a, b); };
                if (inner == Inner::parallel_phase)
                {
                    shrink_array::parallel_phase(2, inner_body);
                }
                else
                {
#pragma omp parallel num_threads(2)
                    {
                        begin_phase();
                        inner_body();
                    }
                }
                // Back in the outer region: writes go to THIS thread's rank.
                for (long i = b * 1024; i < (b + 1) * 1024; ++i)
                {
                    a[i] *= 2;
                    if (i % 11 == 0)
                    {
                        a[i] = 0;
                    }
                }
            }
        }
        end_phase();
        for (long i = 0; i < n; ++i)
        {
            ref[i] = expected_before_inner(i, round);
            if (i % 7 == 0)
            {
                ref[i] += 3;
            }
            ref[i] *= 2;
            if (i % 11 == 0)
            {
                ref[i] = 0;
            }
        }
    }

    long fails = 0;
    for (long i = 0; i < n; ++i)
    {
        const int v = a[i];
        if (v != ref[i])
        {
            if (fails < 5)
            {
                println("  MISMATCH i={}  {} (ref {})", i, v, ref[i]);
            }
            ++fails;
        }
    }
    return fails;
}

/// Re-runs this executable with --nested-active and checks that it stopped on
/// the nested-active invariant.
[[maybe_unused]] static int check_nested_active(const char* self) // OpenMP builds only
{
    const std::string log    = "/tmp/nested_region_active_" + std::to_string(getpid()) + ".log";
    const std::string cmd    = validation::self_command(self) + " --nested-active > \"" + log + "\" 2>&1";
    const int         status = std::system(cmd.c_str());
    std::string       out;
    if (FILE* f = std::fopen(log.c_str(), "r"))
    {
        char buf[4096];
        for (std::size_t n = 0; (n = std::fread(buf, 1, sizeof buf, f)) > 0;)
        {
            out.append(buf, n);
        }
        std::fclose(f);
    }
    std::remove(log.c_str());

    const bool stopped = status != -1 && (!WIFEXITED(status) || WEXITSTATUS(status) != 0);
    const bool message = out.find("nested ACTIVE") != std::string::npos;
    println("[nested-active] two active levels: {}, {}", stopped ? "stopped" : "RAN ON",
            message ? "invariant message printed" : "NO invariant message");
    if (!stopped || !message)
    {
        println("---- child output ----\n{}----------------------", out.c_str());
        return 1;
    }
    return 0;
}

static int run_body(int argc, char** argv)
{
    const bool nested_active = argc > 1 && std::string(argv[1]) == "--nested-active";

    chunk_pmp_set_max_active_levels(nested_active ? 2 : 1);
    ParamChunk::init_indexes(true);

    const long       n = 1L << 16;
    Array<int>       a(n);
    std::vector<int> ref(n, 0);

    println("=== nested_region_validation [variant = {}]  outer team 4, nested regions {} ===", variant_name(),
            nested_active ? "ACTIVE (--nested-active)" : "serialised");
    std::fflush(stdout);

    long fails = run_nested(a, ref, 3, Inner::pragma);

    if (nested_active)
    {
        println("[nested-active] two active levels ran without an invariant failure");
        return 1;
    }

    // C and D: deferred writes queued when the nested regions end.
    if constexpr (test_hooks::force_defer_compiled)
    {
        const unsigned saved          = test_hooks::force_defer_every;
        test_hooks::force_defer_every = 7;
        for (const Inner inner : {Inner::pragma, Inner::parallel_phase})
        {
            const unsigned long before = test_hooks::forced_deferrals.load();
            Array<int>          c(n);
            std::vector<int>    ref_c(n, 0);
            const long          f      = run_nested(c, ref_c, 3, inner);
            const unsigned long forced = test_hooks::forced_deferrals.load() - before;
            println("[{}] deferred writes queued at the nested ends: {} forced, {} wrong",
                    inner == Inner::pragma ? "C omp parallel" : "D parallel_phase", forced, f);
            fails += f;
            if (forced == 0)
            {
                println("  the forced-deferral hook did not fire: the scenario tested nothing");
                ++fails;
            }
        }
        test_hooks::force_defer_every = saved;
    }
    else
    {
        println("[C, D] skipped: no forced-deferral hook in this build (check builds have it)");
    }

    if constexpr (build_config::use_openmp)
    {
        fails += check_nested_active(argv[0]);
    }

    if (fails)
    {
        println("=== NESTED REGION FAILED ({}) ===", fails);
        return 1;
    }
    println("=== NESTED REGION OK ===");
    return 0;
}

int run(int argc, char** argv)
{
    // The arrays live in run_body() and are destroyed when it returns, which
    // parks their buffers in the reuse lists; release_chunk_pools() frees them.
    // Without it they stay parked until exit, and LeakSanitizer reports every
    // one in the std-alloc builds (each buffer is a real malloc there).
    const int rc = run_body(argc, argv);
    release_chunk_pools();
    destroy_block_chunk();
    return rc;
}

} // namespace validation::nested_region_validation
