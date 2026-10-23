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
// Phase-protocol harness. Variant-agnostic: chunk / chunk2 / chunk2a by the
// compile-time macro (CHUNK_VARIANT_V1 / _V2 / _V2A), like the other harnesses.
// ============================================================================
//
// Guards the guard of chunk/internal/phase_guard.h. Without OMPT the program
// brackets its regions -- begin_parallel_phase() by every thread,
// finalize_parallel_phase() after the region -- and a forgotten call springs
// only when the team size changes between regions: a freed deferred op
// replayed, or values silently wrong. Every case runs in a child (this
// executable again, --case k), with the team size changing from region to
// region and one write in 7 forced down the deferred path (test_hooks.h):
//
//   - the correct uses run to the end and read back what a std::vector does:
//     the brackets by hand, parallel_phase() (a reduction in its body),
//     the only_flush / after_flush pair, a begin in a nested region, a begin
//     outside any region;
//   - a forgotten finalize stops at the next begin, in every build (release
//     included: the check is per region, not per write);
//   - check builds: a forgotten begin stops at the first deferred write, and
//     release_chunk_pools() inside a region stops.
//
// OpenMP without OMPT only (the protocol is manual there); 77 = skipped.
// ============================================================================

#include <cstdio>      // std::fopen, std::fclose, std::fread, FILE, std::remove
#include <cstdlib>     // std::system, std::atoi
#include <cstring>     // std::strcmp
#include <string>      // std::string
#include <type_traits> // std::conditional_t
#include <utility>     // std::as_const
#include <vector>      // std::vector, the reference

#include <sys/wait.h> // WIFEXITED, WEXITSTATUS
#include <unistd.h>   // getpid

#include "chunk/zero_array.h"           // zero::Array
#include "chunk/zero_one_value_array.h" // zero_one_value::Array
#include "chunk/zero_any_array.h"       // zero_any::Array
#include "chunk/zero_chunk.h"           // release_chunk_pools
#include "chunk/config.h"              // chunk_config::use_chunk*, build_config::has_not_ompt, any_check
#include "chunk/internal/test_hooks.h" // test_hooks::force_defer_every
#include "chunk/manage_ompt.h"         // begin_parallel_phase, finalize_parallel_phase, parallel_phase
#include "chunk/param_chunk.h"         // ParamChunk::init_indexes
#include "chunk/internal/print.h"      // shrink_array::println

#include "validation_harness.h" // self_command; run() is called by validation_driver.cpp

namespace validation::phase_protocol_validation
{

using namespace shrink_array;

namespace
{

static_assert(chunk_config::use_chunk || chunk_config::use_chunk2 || chunk_config::use_chunk2a,
              "phase_protocol_validation must be built with CHUNK_VARIANT_V1, _V2 or _V2A");

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

constexpr int n_items   = 1 << 18;
constexpr int n_regions = 6;

/// Region r's value of element i: a third of the chunks constant (they
/// compress between regions), the rest varied.
int value(int r, int i)
{
    return ((i >> 8) % 3 == 0) ? r : (i % 5) + r;
}

/// Region r's team: 2 to 8 threads, a different size from one region to the
/// next -- what makes a forgotten call spring.
int team(int r)
{
    return 2 + (r * 5) % 7;
}

/// Region r's schedule chunk: the owner of an element changes too.
int sched(int r)
{
    return 1 << (4 + r % 5);
}

/// How a case brackets its regions.
enum class Bracket
{
    by_hand,      ///< begin by every thread, finalize after
    helper,       ///< parallel_phase()
    split,        ///< begin, only_flush inside, after_flush after
    nested_begin, ///< by_hand, plus a begin in a serialised nested region
    begin_serial, ///< by_hand, plus a begin outside any region first
    no_finalize,  ///< begin, never finalize
    no_begin,     ///< finalize, never begin
};

/// The helper's reduction. At namespace scope, not a local of regions():
/// the body lambda would capture it, and clang refuses a reduction on a
/// captured variable ("reduction variable must be shared"; GCC accepts it).
long helper_sum = 0;

/// The regions of a case; the elements that do not read back as a
/// std::vector does.
long regions(Bracket how)
{
    Array<int>       a(n_items);
    std::vector<int> ref(n_items, 0);
    helper_sum = 0;
    for (int r = 0; r < n_regions; ++r)
    {
        auto body = [&]
        {
            // A local of the lambda, not a capture: clang crashes generating an
            // orphaned omp for whose schedule chunk names a captured variable
            // (CGOpenMPInlinedRegionInfo::getContextValue; GCC accepts it).
            const int cs = sched(r);
#pragma omp for schedule(static, cs) reduction(+ : helper_sum)
            for (int i = 0; i < n_items; ++i)
            {
                const int v = value(r, i);
                a[i]        = v;
                ref[i]      = v;
                helper_sum += v;
            }
        };
        if (how == Bracket::helper)
        {
            parallel_phase(team(r), body);
            continue;
        }
        if (how == Bracket::begin_serial)
        {
            begin_parallel_phase(); // outside any region: nothing to bracket, ignored
        }
#pragma omp parallel num_threads(team(r))
        {
            if (how != Bracket::no_begin)
            {
                begin_parallel_phase();
            }
            if (how == Bracket::nested_begin)
            {
#pragma omp parallel num_threads(2)
                {
                    begin_parallel_phase(); // serialised: the same thread, the same phase
                }
            }
            body();
            if (how == Bracket::split)
            {
                finalize_parallel_phase_only_flush();
            }
        }
        if (how == Bracket::split)
        {
            finalize_parallel_phase_after_flush();
        }
        else if (how != Bracket::no_finalize)
        {
            finalize_parallel_phase();
        }
    }
    long        bad = 0;
    const auto& ca  = std::as_const(a);
    for (int i = 0; i < n_items; ++i)
    {
        bad += (ca[i] != ref[i]) ? 1 : 0;
    }
    if (how == Bracket::helper)
    {
        long want = 0;
        for (int r = 0; r < n_regions; ++r)
        {
            for (int i = 0; i < n_items; ++i)
            {
                want += value(r, i);
            }
        }
        if (helper_sum != want)
        {
            println("  reduction: {}, expected {}", helper_sum, want);
            ++bad;
        }
    }
    return bad;
}

/// One case: how its child brackets the regions, and what must happen.
struct Case
{
    const char* name;
    Bracket     how;
    bool        teardown_in_region; ///< instead of the regions: release_chunk_pools() in one
    bool        check_build_only;   ///< the check exists in check builds only
    const char* must_print;         ///< nullptr: must run to the end, values right
};

constexpr const char* msg_twice     = "begin_parallel_phase() twice in the same phase";
constexpr const char* msg_deferred  = "deferred write in a parallel region by thread";
constexpr const char* msg_in_region = "release_chunk_pools() called inside a parallel region";

const Case case_table[] = {
    {"brackets by hand", Bracket::by_hand, false, false, nullptr},
    {"parallel_phase()", Bracket::helper, false, false, nullptr},
    {"only_flush / after_flush pair", Bracket::split, false, false, nullptr},
    {"begin in a nested region", Bracket::nested_begin, false, false, nullptr},
    {"begin outside any region", Bracket::begin_serial, false, false, nullptr},
    {"finalize forgotten", Bracket::no_finalize, false, false, msg_twice},
    {"begin forgotten (deferred write)", Bracket::no_begin, false, true, msg_deferred},
    {"release_chunk_pools() in a region", Bracket::by_hand, true, true, msg_in_region},
};
constexpr int nb_cases = static_cast<int>(sizeof case_table / sizeof case_table[0]);

/// The child: run case @p k; 0 when every element read back right.
int child(int k)
{
    const Case& c = case_table[k];
    if (c.teardown_in_region)
    {
#pragma omp parallel
        release_chunk_pools();
        return 0;
    }
    const long bad = regions(c.how);
    if (bad != 0)
    {
        println("  {} wrong element(s)", bad);
    }
    return bad == 0 ? 0 : 1;
}

/// The parent: run case @p k in a child, check what happened.
int check_case(const char* self, int k)
{
    const Case&       c   = case_table[k];
    const std::string log = "/tmp/phase_protocol_" + std::to_string(getpid()) + "_" + std::to_string(k) + ".log";
    const std::string cmd = validation::self_command(self) + " --case " + std::to_string(k) + " > \"" + log + "\" 2>&1";
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

    const bool ran_ok   = status != -1 && WIFEXITED(status) && WEXITSTATUS(status) == 0;
    const bool must_run = c.must_print == nullptr;
    const bool pass     = must_run ? (ran_ok && out.find("[PHASE PROTOCOL]") == std::string::npos)
                                   : (!ran_ok && out.find(c.must_print) != std::string::npos);
    println("[{}/phase-protocol] {:<36} {}{}", variant_name(), c.name,
            must_run ? (ran_ok ? "ran" : "STOPPED") : (ran_ok ? "RAN ON" : "stopped"), pass ? "" : "   <-- FAIL");
    if (!pass)
    {
        println("---- child output ----\n{}----------------------", out);
        return 1;
    }
    return 0;
}

} // namespace

int run(int argc, char** argv)
{
    if constexpr (!build_config::use_openmp || !build_config::has_not_ompt || !test_hooks::force_defer_compiled)
    {
        println("phase_protocol_validation: skipped (OpenMP without OMPT, with the test hooks, only)");
        return 77;
    }
    ParamChunk::init_indexes(true);
    if (argc > 2 && std::strcmp(argv[1], "--case") == 0)
    {
        test_hooks::force_defer_every = 7; // the deferred path, every 7th write of every thread
        return child(std::atoi(argv[2]));
    }

    println("=== phase_protocol_validation [variant = {}]  check build: {} ===", variant_name(),
            build_config::any_check ? "yes" : "no");
    int fails = 0;
    for (int k = 0; k < nb_cases; ++k)
    {
        if (case_table[k].check_build_only && !build_config::any_check)
        {
            println("[{}/phase-protocol] {:<36} skipped (check builds only)", variant_name(), case_table[k].name);
            continue;
        }
        fails += check_case(argv[0], k);
    }
    println("{}", fails ? "phase_protocol_validation: FAILED" : "phase_protocol_validation: OK");
    return fails ? 1 : 0;
}

} // namespace validation::phase_protocol_validation
