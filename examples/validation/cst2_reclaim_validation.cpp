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
 * @file cst2_reclaim_validation.cpp
 * @brief ArrayChunk2A::reclaim_second_values() frees exactly the second values
 *        nothing refers to any more, and the registry reuses their identifiers.
 *
 * @details Chunk2A's registry of second values (GlobalVecHolder) keeps every
 *          value it has seen until the last array of the type is destroyed. A
 *          workload whose constants drift registers a new one at each
 *          compression, and the identifiers run out -- which stops the run.
 *          The reclaim marks what is still referenced and frees the rest.
 *          Its failure mode is a WRONG VALUE: an identifier freed while a
 *          chunk still holds it goes to another value, and the chunk reads
 *          that one. So every phase reads every element back.
 *
 *          On ArrayChunk2A<double>, every chunk written whole at its own
 *          constant (base + chunk index), so that each compresses to a
 *          registered second value:
 *
 *            A  rounds at new constants: each reclaim frees exactly the
 *               previous round's values, the registry holds one round, and
 *               the identifiers handed out stop growing after round two;
 *            B  half the chunks rewritten: only their old values go;
 *            C  a chunk materialised again (one element changed) keeps its
 *               second value through a reclaim, and folds back to it;
 *            D  a copy keeps its values when the original is destroyed and
 *               the registry reclaimed;
 *            E  (OpenMP) inside an active parallel region the call is
 *               refused (a team of one runs it: nothing else is running);
 *            F  under a lowered identifier limit (a child process), rounds
 *               that register four times the limit run to the end when each
 *               round reclaims.
 *
 *          Usage: cst2_reclaim_validation [chunks]
 */

#include <cstdio>     // std::remove
#include <cstdlib>    // std::system
#include <format>     // std::format
#include <fstream>    // the --rounds child's log
#include <print>      // std::println
#include <string>     // std::string, std::to_string, std::getline
#include <sys/wait.h> // WIFEXITED, WEXITSTATUS
#include <unistd.h>   // getpid

#include "chunk/chunk_traits.h"       // forward decls (Chunk2A) + default ind_type = int
#include "chunk/array_chunk2A.h"      // ArrayChunk2A, Chunk2A complete (its registry types)
#include "chunk/param_chunk.h"        // ParamChunk::init_indexes
#include "chunk/config.h"             // chunk_config::use_chunk2a, build_config::has_not_ompt
#include "chunk/PMP_CHUNK.h"          // CHUNK_PMP_MAX_THREADS, CHUNK_PMP_NUM_THREADS
#include "chunk/manage_ompt.h"        // begin_parallel_phase / finalize_parallel_phase
#include "chunk/internal/env_parse.h" // arg_number (checked, not atoi)

#include "validation_harness.h" // self_command; run() is called by validation_driver.cpp

namespace validation::cst2_reclaim_validation
{

using namespace shrink_array;

static_assert(chunk_config::use_chunk2a, "cst2_reclaim_validation is Chunk2A's registry: link chunk_v2a_<size>");

using Arr      = ArrayChunk2A<double>;
using Registry = Chunk2A<double>::tab_cst_val2_holder::global_vec_holder;

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

/// @brief The constant chunk @p j holds after a fill at @p base.
static double cst(double base, long j)
{
    return base + static_cast<double>(j);
}

/// @brief Write chunk j of @p a whole at cst(base, j) -- the chunks for which
///        @p only_even is false or j is even -- then the between-region pass.
static void fill(Arr& a, long nch, long S, double base, bool only_even = false)
{
    const int ni = static_cast<int>(nch * S);
#pragma omp parallel
    {
        begin_phase();
#pragma omp for schedule(static)
        for (int i = 0; i < ni; ++i)
        {
            const long j = i / S;
            if (!only_even || j % 2 == 0)
            {
                a[i] = cst(base, j);
            }
        }
    }
    end_phase();
}

static int fails = 0;

/// @brief Every element of @p a against the constant of its chunk:
///        cst(even_base, j) for an even chunk, cst(odd_base, j) for an odd one.
static void expect_values(const char* where, const Arr& a, long nch, long S, double even_base, double odd_base)
{
    long bad = 0;
    for (long i = 0; i < nch * S; ++i)
    {
        const long j = i / S;
        bad += (a[static_cast<int>(i)] != cst(j % 2 == 0 ? even_base : odd_base, j)) ? 1 : 0;
    }
    if (bad != 0)
    {
        std::println("  {:<48} {} element(s) read back wrong  <-- WRONG", where, bad);
        ++fails;
    }
}

/// @brief One count against its expected value, printed either way.
static void expect_eq(const char* where, long got, long want)
{
    std::println("  {:<48} {} (expect {}){}", where, got, want, got == want ? "" : "  <-- WRONG");
    if (got != want)
    {
        ++fails;
    }
}

/// @brief One count against a bound, printed either way.
static void expect_le(const char* where, long got, long bound)
{
    std::println("  {:<48} {} (expect <= {}){}", where, got, bound, got <= bound ? "" : "  <-- WRONG");
    if (got > bound)
    {
        ++fails;
    }
}

static long entries()
{
    return static_cast<long>(Registry::count_entries());
}
static long handed_out()
{
    return static_cast<long>(Registry::ids_handed_out());
}
static long reclaim()
{
    return static_cast<long>(Arr::reclaim_second_values());
}

/// @brief --rounds (child, run under a lowered CHUNK_CST_VAL2_ID_LIMIT):
///        @p rounds rounds of @p nch new constants, a reclaim after each.
///        Returning means it ran to the end.
static int run_rounds_child(long nch, long rounds)
{
    const long S = static_cast<long>(Arr(0).chunk_size());
    Arr        a(static_cast<int>(nch * S));
    for (long r = 0; r < rounds; ++r)
    {
        fill(a, nch, S, 1.0e6 + 1000. * static_cast<double>(r));
        (void)reclaim();
    }
    expect_values("child: last round", a, nch, S, 1.0e6 + 1000. * static_cast<double>(rounds - 1),
                  1.0e6 + 1000. * static_cast<double>(rounds - 1));
    std::println("[rounds] {} distinct second values, reclaimed each round: RAN TO THE END", nch * rounds);
    return fails ? 1 : 0;
}

/// @brief F: re-run this executable with --rounds under a limit of 3 x chunks
///        identifiers, for 12 rounds (four times the limit in distinct
///        values): it must reach the end.
static void check_rounds_under_limit(const char* self, long nch)
{
    const long        limit = 3 * nch;
    const std::string log   = "/tmp/cst2_reclaim_" + std::to_string(getpid()) + ".log";
    const std::string cmd = "CHUNK_CST_VAL2_ID_LIMIT=" + std::to_string(limit) + " " + validation::self_command(self) +
                            " --rounds " + std::to_string(nch) + " 12 > \"" + log + "\" 2>&1";
    const int         status = std::system(cmd.c_str());
    const bool        ok     = status != -1 && WIFEXITED(status) && WEXITSTATUS(status) == 0;
    bool              said   = false;
    {
        std::ifstream in(log);
        std::string   line;
        while (std::getline(in, line))
        {
            said = said || line.find("RAN TO THE END") != std::string::npos;
        }
    }
    std::remove(log.c_str());
    std::println("F  limit {} identifiers, {} distinct values        {}{}", limit, 12 * nch,
                 (ok && said) ? "ran to the end" : "STOPPED", (ok && said) ? "" : "  <-- WRONG");
    if (!ok || !said)
    {
        ++fails;
    }
}

int run(int argc, char** argv)
{
    if (argc > 3 && std::string(argv[1]) == "--rounds")
    {
        ParamChunk::init_indexes(true);
        return run_rounds_child(shrink_array::arg_number<long>(argv[2], 16, "chunks"),
                                shrink_array::arg_number<long>(argv[3], 12, "rounds"));
    }

    ParamChunk::init_indexes(true);

    const long nch = argc > 1 ? shrink_array::arg_number<long>(argv[1], 16, "chunks") : 16;
    const long S   = static_cast<long>(Arr(0).chunk_size()); // empty: no chunk, so no registry reference

    std::println("=== cst2_reclaim_validation  chunks={}  chunk_size={}  threads={} ===", nch, S,
                 CHUNK_PMP_MAX_THREADS);
    {
        Arr a(static_cast<int>(nch * S));

        // A: rounds at new constants.
        std::println("A  rounds at new constants");
        for (int r = 0; r < 4; ++r)
        {
            const double base = 1000. * (r + 1);
            fill(a, nch, S, base);
            std::string where;
            where = std::format("round {}: entries before the reclaim", r);
            expect_eq(where.c_str(), entries(), r == 0 ? nch : 2 * nch);
            where = std::format("round {}: freed", r);
            expect_eq(where.c_str(), reclaim(), r == 0 ? 0 : nch);
            where = std::format("round {}: entries after", r);
            expect_eq(where.c_str(), entries(), nch);
            where = std::format("round {}: identifiers handed out", r);
            expect_le(where.c_str(), handed_out(), 2 * nch);
            where = std::format("round {}: values", r);
            expect_values(where.c_str(), a, nch, S, base, base);
        }

        // B: half the chunks rewritten.
        std::println("B  even chunks rewritten");
        fill(a, nch, S, 5000.);
        (void)reclaim();
        fill(a, nch, S, 6000., true);
        expect_eq("freed", reclaim(), (nch + 1) / 2);
        expect_eq("entries after", entries(), nch);
        expect_values("values", a, nch, S, 6000., 5000.);

        // C: a chunk materialised again keeps its second value.
        std::println("C  materialised chunk");
        fill(a, nch, S, 7000.);
        (void)reclaim();
#pragma omp parallel
        {
            begin_phase();
#pragma omp single
            a[1] = 1.5; // chunk 0 holds a buffer again
        }
        end_phase();
        (void)reclaim();
        if (a[1] != 1.5)
        {
            std::println("  {:<48} reads {:g}  <-- WRONG", "the changed element", static_cast<double>(a[1]));
            ++fails;
        }
#pragma omp parallel
        {
            begin_phase();
#pragma omp single
            a[1] = cst(7000., 0); // back to the chunk's constant: folds again
        }
        end_phase();
        (void)reclaim();
        expect_values("values", a, nch, S, 7000., 7000.);

        // D: a copy outlives the original.
        std::println("D  copy");
        fill(a, nch, S, 8000.);
        (void)reclaim();
        {
            Arr b(a);
            fill(a, nch, S, 9000.);
            expect_eq("freed (the copy still refers to them)", reclaim(), 0);
            expect_values("copy: values", b, nch, S, 8000., 8000.);
            expect_values("original: values", a, nch, S, 9000., 9000.);
        }
        expect_eq("freed (copy destroyed)", reclaim(), nch);
        expect_values("original after the copy: values", a, nch, S, 9000., 9000.);

        // E: refused inside a parallel region.
        if constexpr (build_config::use_openmp)
        {
            std::println("E  inside a parallel region");
            fill(a, nch, S, 10000.);
            long in_region = -1;
            int  team      = 0;
#pragma omp parallel
            {
#pragma omp masked
                {
                    team      = CHUNK_PMP_NUM_THREADS;
                    in_region = reclaim();
                }
            }
            // A team of one is an inactive region: no other thread, the call
            // is a sequential point and runs.
            const bool active = team > 1;
            expect_eq(active ? "freed inside the (active) region" : "freed inside the (inactive) region", in_region,
                      active ? 0 : nch);
            expect_eq("freed after it", reclaim(), active ? nch : 0);
            expect_values("values", a, nch, S, 10000., 10000.);
        }
    }
    expect_eq("all arrays destroyed: entries", entries(), 0);

    // F: under a lowered identifier limit, reclaiming rounds run to the end.
    check_rounds_under_limit(argv[0], nch);

    std::println("=== {} ===", fails ? "CST2_RECLAIM FAILED" : "CST2_RECLAIM OK");
    return fails ? 1 : 0;
}

} // namespace validation::cst2_reclaim_validation
