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
 * @file cst2_registry_validation.cpp
 * @brief Chunk2A's second-value registry (GlobalVecHolder) is freed by the
 *        last array of EVERY generation, not only the first.
 *
 * @details The registry of a type is shared by every zero_any::Array<T> and
 *          counted: each array takes a reference, and the last one out frees
 *          the second-value arrays, the segments and the region's pages
 *          (clear_if_last). It then reinstalls an empty segment -- through
 *          init(), which also took a reference. With no array alive the count
 *          went 0 -> 1, the next generation's last one out found 1 left, and
 *          nothing was ever freed again before exit.
 *
 *          Three scenarios, on zero_any::Array<double>, each value written over
 *          whole chunks so that it compresses to a registered second value:
 *
 *            A  three generations one after the other, a different value
 *               each: while alive, 1 user and >= 1 entry; once destroyed,
 *               0 users and 0 entries -- every time (the second failed);
 *            B  overlap: an outer array alive across an inner one's life.
 *               The inner one's teardown must free NOTHING (2 -> 1 user, the
 *               outer value still registered and still read back); the
 *               outer one's then frees everything;
 *            C  after all that, one more array: the reinstalled registry
 *               works (values compress, read back, and are freed again).
 *
 *          Every value is also read back after each phase.
 *
 *          The phase brackets as mark_path_validation: begin/finalize when
 *          there is no OMPT tool, and in a serial build the writers' pass
 *          (make_block_compression) the harness runs itself.
 *
 *          Usage: cst2_registry_validation [chunks]
 */

#include <cstdio>     // std::snprintf, std::remove, std::getline
#include <cstdlib>    // std::system
#include <fstream>    // the --id-limit child's log
#include <string>     // std::string, std::to_string, std::getline
#include <sys/wait.h> // WIFEXITED, WEXITSTATUS
#include <unistd.h>   // getpid

#include "chunk/chunk_traits.h"       // forward decls (Chunk2A) + default IntType = int
#include "chunk/array_chunk2A.h"      // zero_any::Array, Chunk2A complete (its registry types)
#include "chunk/param_chunk.h"        // ParamChunk::init_indexes
#include "chunk/config.h"             // chunk_config::use_chunk2a, build_config::has_not_ompt
#include "chunk/PMP_CHUNK.h"          // CHUNK_PMP_MAX_THREADS
#include "chunk/manage_ompt.h"        // begin_parallel_phase / finalize_parallel_phase
#include "chunk/internal/env_parse.h" // arg_number (checked, not atoi)
#include "chunk/internal/print.h"     // shrink_array::println

#include "validation_harness.h" // self_command; run() is called by validation_driver.cpp

namespace validation::cst2_registry_validation
{

using namespace shrink_array;

static_assert(chunk_config::use_chunk2a, "cst2_registry_validation is Chunk2A's registry: link chunk_v2a_<size>");

using Arr      = zero_any::Array<double>;
using Registry = Chunk2A<double>::Holder::VecHolder;

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
    // Serial chunk2a has no between-region pass: run the writers' pass.
    if constexpr (!build_config::use_openmp)
    {
        Arr::make_block_compression();
    }
}

/// @brief Write @p v everywhere in @p a (whole chunks: it compresses to a
///        second value), then the between-region pass.
static void fill(Arr& a, long n, double v)
{
    const int ni = static_cast<int>(n);
#pragma omp parallel
    {
        begin_phase();
#pragma omp for schedule(static)
        for (int i = 0; i < ni; ++i)
        {
            a[i] = v;
        }
    }
    end_phase();
}

/// @brief Elements of @p a that do not read @p v back.
static long mismatches(const Arr& a, long n, double v)
{
    long bad = 0;
    for (long i = 0; i < n; ++i)
    {
        bad += (a[static_cast<int>(i)] != v) ? 1 : 0;
    }
    return bad;
}

static int fails = 0;

/// @brief Check the registry's users and entries at a quiescent point.
///        @p min_entries: at least that many; -1 for exactly 0.
static void expect(const char* where, int users, long min_entries)
{
    const int  u    = Registry::users();
    const long e    = static_cast<long>(Registry::count_entries());
    const bool ok_u = (u == users);
    const bool ok_e = (min_entries < 0) ? (e == 0) : (e >= min_entries);
    println("  {:<44} users {} (expect {})  entries {} (expect {}{}){}", where, u, users, e,
            min_entries < 0 ? "" : ">= ", min_entries < 0 ? 0L : min_entries, (ok_u && ok_e) ? "" : "  <-- WRONG");
    if (!ok_u || !ok_e)
    {
        ++fails;
    }
}

static void expect_values(const char* where, const Arr& a, long n, double v)
{
    const long bad = mismatches(a, n, v);
    if (bad)
    {
        println("  {:<44} {} element(s) do not read {:g} back  <-- WRONG", where, bad, v);
        ++fails;
    }
}

/// @brief --id-limit (child, run under CHUNK_CST_VAL2_ID_LIMIT=16): 40 chunks,
///        each at its own constant -- 40 distinct second values. Must stop
///        at the 17th registration; returning means it ran on.
static int run_id_limit_child()
{
    const long s  = static_cast<long>(Arr(0).chunk_size());
    const int  ni = static_cast<int>(40 * s);
    Arr        a(ni);
#pragma omp parallel
    {
        begin_phase();
#pragma omp for schedule(static)
        for (int i = 0; i < ni; ++i)
        {
            a[i] = 100. + static_cast<double>(i / s);
        }
    }
    end_phase();
    println("[id-limit] 40 distinct second values registered under a limit of 16: RAN ON");
    return 0;
}

/// @brief Re-run this executable with --id-limit under a lowered identifier
///        limit: it must stop, and say why -- alloc_chunk() must not write
///        value_by_id[] out of bounds.
static void check_id_limit(const char* self)
{
    const std::string log = "/tmp/cst2_id_limit_" + std::to_string(getpid()) + ".log";
    const std::string cmd = std::string("CHUNK_CST_VAL2_ID_LIMIT=16 ") + validation::self_command(self) +
                            " --id-limit > \"" + log + "\" 2>&1";
    const int  status  = std::system(cmd.c_str());
    const bool stopped = status != -1 && (!WIFEXITED(status) || WEXITSTATUS(status) != 0);
    bool       said    = false;
    {
        std::ifstream in(log);
        std::string   line;
        while (std::getline(in, line))
        {
            said = said || line.find("identifiers exhausted") != std::string::npos;
        }
    }
    std::remove(log.c_str());
    println("D  identifier limit (16, 40 distinct values)       {}, {}{}", stopped ? "stopped" : "RAN ON",
            said ? "with the message" : "no message", (stopped && said) ? "" : "  <-- WRONG");
    if (!stopped || !said)
    {
        ++fails;
    }
}

int run(int argc, char** argv)
{
    if (argc > 1 && std::string(argv[1]) == "--id-limit")
    {
        ParamChunk::init_indexes(true);
        return run_id_limit_child();
    }

    ParamChunk::init_indexes(true);

    const long nchunks = argc > 1 ? shrink_array::arg_number<long>(argv[1], 16, "chunks") : 16;
    const long s       = static_cast<long>(Arr(0).chunk_size()); // empty: no chunk, so no registry reference
    const long n       = nchunks * s;

    println("=== cst2_registry_validation  chunks={}  chunk_size={}  threads={} ===", nchunks, s,
            CHUNK_PMP_MAX_THREADS);
    expect("start", 0, -1);

    // A: three generations, one after the other.
    println("A  successive generations");
    for (int g = 0; g < 3; ++g)
    {
        const double v = 7. + g;
        {
            Arr a(static_cast<int>(n));
            fill(a, n, v);
            expect_values("gen alive: values", a, n, v);
            char where[64];
            std::snprintf(where, sizeof where, "gen %d alive (value %g)", g + 1, v);
            expect(where, 1, 1);
        }
        char where[64];
        std::snprintf(where, sizeof where, "gen %d destroyed", g + 1);
        expect(where, 0, -1);
    }

    // B: an outer array alive across an inner one's life.
    println("B  overlapping lifetimes");
    {
        Arr outer(static_cast<int>(n));
        fill(outer, n, 21.);
        expect("outer alive", 1, 1);
        {
            Arr inner(static_cast<int>(n));
            fill(inner, n, 22.);
            expect("outer + inner alive", 2, 2);
            expect_values("inner: values", inner, n, 22.);
        }
        expect("inner destroyed (outer alive)", 1, 1);
        expect_values("outer after inner's teardown: values", outer, n, 21.);
        fill(outer, n, 23.); // the registry is still usable
        expect_values("outer rewritten: values", outer, n, 23.);
    }
    expect("outer destroyed", 0, -1);

    // C: the reinstalled registry works.
    println("C  after the extinctions");
    {
        Arr a(static_cast<int>(n));
        fill(a, n, 31.);
        expect_values("fresh array: values", a, n, 31.);
        expect("fresh array alive", 1, 1);
    }
    expect("fresh array destroyed", 0, -1);

    // D: the identifier limit stops the run (a child process).
    check_id_limit(argv[0]);

    println("=== {} ===", fails ? "CST2_REGISTRY FAILED" : "CST2_REGISTRY OK");
    return fails ? 1 : 0;
}

} // namespace validation::cst2_registry_validation
