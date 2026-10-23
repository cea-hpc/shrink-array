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
// Concurrent-teams harness. Variant-agnostic: chunk / chunk2 / chunk2a by the
// compile-time macro (CHUNK_VARIANT_V1 / _V2 / _V2A), like the other harnesses.
// ============================================================================
//
// Guards ONE property: two top-level OpenMP teams writing arrays at once stop
// the program with a message, instead of corrupting each other.
//
// The per-thread structures are indexed by the rank in the team, and the end
// of a phase drains every rank's queue. Two teams opened by two host threads
// share ranks 0..n-1: one team's phase end replays the other's writes in the
// middle of its region, and their writers clear each other's flags. Release
// builds gave wrong values, silently (docs/getting_started.md, section 5).
// phase_guard::teams refuses the second team: under OMPT from the
// parallel_begin callback, without OMPT from begin_parallel_phase().
//
// What it does:
//   A. two host threads, one after the other, each opening its own team on its
//      own array: allowed (one team at a time), checked against std::vectors.
//   B. re-runs itself with `--concurrent`: two host threads whose teams are
//      held open together (a latch inside the regions), each writing its own
//      array; must stop with the "top-level" invariant failure.
// In a serial build there are no teams: A runs single-threaded, B is skipped.
// ============================================================================

#include <cstdio>      // std::fopen, std::fclose, std::fread, std::fflush, FILE, stdout, std::remove
#include <cstdlib>     // std::system
#include <latch>       // std::latch (B: the two teams held open together)
#include <string>      // std::string, std::to_string
#include <thread>      // std::thread (the two host threads)
#include <type_traits> // std::conditional_t
#include <vector>      // std::vector
#include <unistd.h>    // getpid
#include <sys/wait.h>  // WIFEXITED, WEXITSTATUS

#include "chunk/chunk_traits.h"   // forward decls + default IntType = int
#include "chunk/array_chunk.h"    // zero::Array
#include "chunk/array_chunk2.h"   // zero_one_value::Array
#include "chunk/array_chunk2A.h"  // zero_any::Array
#include "chunk/param_chunk.h"    // ParamChunk::init_indexes
#include "chunk/config.h"         // chunk_config::use_chunk2*, build_config::use_openmp / has_not_ompt
#include "chunk/PMP_CHUNK.h"      // chunk_pmp_team_rank
#include "chunk/manage_ompt.h"    // begin_parallel_phase / finalize_parallel_phase
#include "chunk/internal/print.h" // shrink_array::println

#include "chunk/chunk_std.h"    // release_chunk_pools, destroy_block_chunk
#include "validation_harness.h" // self_command; run() is called by validation_driver.cpp

namespace validation::concurrent_teams_validation
{

using namespace shrink_array;

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

/// The value element i holds after round `round` of host thread `who`.
static int expected(long i, int round, int who)
{
    return ((i / 2048) + round + who) % 3 == 0 ? 0 : static_cast<int>(((i * 5) + round + who) % 89) + 1;
}

/// One host thread's work: `rounds` phases of a team of 2 on its own array.
/// @p hold, when set, is counted down by rank 0 of every region and waited on:
/// with two host threads the two teams are then open together.
static void phases(Array<int>& a, int who, int rounds, std::latch* hold)
{
    const long N = a.size();
    for (int round = 0; round < rounds; ++round)
    {
#pragma omp parallel num_threads(2)
        {
            if constexpr (build_config::has_not_ompt)
            {
                begin_parallel_phase();
            }
            if (hold != nullptr && chunk_pmp_team_rank() == 0 && round == 0)
            {
                hold->arrive_and_wait();
            }
#pragma omp for schedule(static)
            for (long i = 0; i < N; ++i)
            {
                a[i] = expected(i, round, who);
            }
        }
        if constexpr (build_config::has_not_ompt)
        {
            finalize_parallel_phase();
        }
    }
}

static long check(const Array<int>& a, int rounds, int who)
{
    long fails = 0;
    for (long i = 0; i < a.size(); ++i)
    {
        if (a[i] != expected(i, rounds - 1, who))
        {
            if (fails < 5)
            {
                println("  MISMATCH host thread {} i={}  {} (ref {})", who, i, int(a[i]), expected(i, rounds - 1, who));
            }
            ++fails;
        }
    }
    return fails;
}

/// Re-runs this executable with --concurrent and checks that it stopped on
/// the concurrent-teams invariant.
[[maybe_unused]] static int check_concurrent(const char* self) // OpenMP builds only
{
    const std::string log    = "/tmp/concurrent_teams_" + std::to_string(getpid()) + ".log";
    const std::string cmd    = validation::self_command(self) + " --concurrent > \"" + log + "\" 2>&1";
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
    const bool message = out.find("top-level") != std::string::npos;
    println("[concurrent] two top-level teams at once: {}, {}", stopped ? "stopped" : "RAN ON",
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
    const bool concurrent = argc > 1 && std::string(argv[1]) == "--concurrent";

    ParamChunk::init_indexes(true);

    const long N      = 1L << 16;
    const int  rounds = 3;
    Array<int> a(N);
    Array<int> b(N);

    println("=== concurrent_teams_validation [variant = {}]  two host threads, teams of 2, {} ===", variant_name(),
            concurrent ? "AT ONCE (--concurrent)" : "one after the other");
    std::fflush(stdout);

    if (concurrent)
    {
        std::latch  hold(2);
        std::thread ta([&] { phases(a, 0, rounds, &hold); });
        std::thread tb([&] { phases(b, 1, rounds, &hold); });
        ta.join();
        tb.join();
        println("[concurrent] two top-level teams ran at once without an invariant failure");
        return 1;
    }

    // A: one after the other.
    {
        std::thread ta([&] { phases(a, 0, rounds, nullptr); });
        ta.join();
        std::thread tb([&] { phases(b, 1, rounds, nullptr); });
        tb.join();
    }
    long fails = check(a, rounds, 0) + check(b, rounds, 1);
    println("[sequential] two host threads, one after the other: {} wrong", fails);

    if constexpr (build_config::use_openmp)
    {
        fails += check_concurrent(argv[0]);
    }

    if (fails)
    {
        println("=== CONCURRENT TEAMS FAILED ({}) ===", fails);
        return 1;
    }
    println("=== CONCURRENT TEAMS OK ===");
    return 0;
}

int run(int argc, char** argv)
{
    // As nested_region_validation: release the reuse pools the arrays parked,
    // or LeakSanitizer reports them in the std-alloc builds.
    const int rc = run_body(argc, argv);
    release_chunk_pools();
    destroy_block_chunk();
    return rc;
}

} // namespace validation::concurrent_teams_validation
