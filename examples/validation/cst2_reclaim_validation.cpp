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
 * @brief zero_any::Array::reclaim_second_values() frees exactly the second values
 *        nothing refers to any more, and the registry reuses their identifiers.
 *
 * @details zero_any's registry of second values (GlobalVecHolder) keeps every
 *          value it has seen until the last array of the type is destroyed. A
 *          workload whose constants drift registers a new one at each
 *          compression, and the identifiers run out -- which stops the run.
 *          The reclaim marks what is still referenced and frees the rest.
 *          Its failure mode is a WRONG VALUE: an identifier freed while a
 *          chunk still holds it goes to another value, and the chunk reads
 *          that one. So every phase reads every element back.
 *
 *          On zero_any::Array<double>, every chunk written whole at its own
 *          constant (base + chunk index), so that each compresses to a
 *          registered second value:
 *
 *            A  rounds at new constants: each reclaim frees exactly the
 *               previous round's values, the registry holds one round, and
 *               the identifiers handed out stop growing after round two;
 *               second_value_stats() counts the reclaims and what they
 *               freed, and its identifiers in use are the registered values;
 *            B  half the chunks rewritten: only their old values go;
 *            C  a chunk materialised again (one element changed) keeps its
 *               second value through a reclaim, and folds back to it;
 *            D  a copy keeps its values when the original is destroyed and
 *               the registry reclaimed;
 *            E  (OpenMP) inside an active parallel region the call is
 *               refused (a team of one runs it: nothing else is running),
 *               and second_value_stats() counts the refusal;
 *            F  under a lowered identifier limit (a child process), rounds
 *               that register four times the limit run to the end when each
 *               round reclaims;
 *            G  (CHUNK_DYNAMIC) a re-chunk to 2S and back: the reclaims around
 *               it keep exactly what the chunks at the new geometry hold, and
 *               one during a re-chunk rebuild is refused and counted.
 *
 *          Usage: cst2_reclaim_validation [chunks]
 */

#include <cstdio>     // std::snprintf, std::remove, std::getline
#include <cstdlib>    // std::system
#include <format>     // std::format
#include <fstream>    // the --rounds child's log
#include <string>     // std::string, std::to_string, std::getline
#include <sys/wait.h> // WIFEXITED, WEXITSTATUS
#include <unistd.h>   // getpid

#include "chunk/chunk_traits.h"               // forward decls (zero_any::Chunk) + default IntType = int
#include "chunk/zero_any_array.h"             // zero_any::Array, zero_any::Chunk complete (its registry types)
#include "chunk/param_chunk.h"                // ParamChunk::init_indexes
#include "chunk/config.h"                     // chunk_config::use_zero_any, build_config::has_not_ompt
#include "chunk/PMP_CHUNK.h"                  // CHUNK_PMP_MAX_THREADS, CHUNK_PMP_NUM_THREADS
#include "chunk/manage_ompt.h"                // begin_parallel_phase / finalize_parallel_phase
#include "chunk/internal/env_parse.h"         // arg_number (checked, not atoi)
#include "chunk/internal/print.h"             // shrink_array::println
#include "chunk/internal/chunk_size_policy.h" // active_chunk_size, set_active_chunk_size (G)

#include "validation_harness.h" // self_command; run() is called by validation_driver.cpp

namespace validation::cst2_reclaim_validation
{

using namespace shrink_array;

static_assert(chunk_config::use_zero_any, "cst2_reclaim_validation is zero_any's registry: link chunk_zero_any_<size>");

using Arr      = zero_any::Array<double>;
using Registry = zero_any::Chunk<double>::Holder::VecHolder;

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
    // Serial zero_any has no between-region pass: run the writers' pass.
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
static void fill(Arr& a, long nch, long s, double base, bool only_even = false)
{
    const int ni = static_cast<int>(nch * s);
#pragma omp parallel
    {
        begin_phase();
#pragma omp for schedule(static)
        for (int i = 0; i < ni; ++i)
        {
            const long j = i / s;
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
static void expect_values(const char* where, const Arr& a, long nch, long s, double even_base, double odd_base)
{
    long bad = 0;
    for (long i = 0; i < nch * s; ++i)
    {
        const long j = i / s;
        bad += (a[static_cast<int>(i)] != cst(j % 2 == 0 ? even_base : odd_base, j)) ? 1 : 0;
    }
    if (bad != 0)
    {
        println("  {:<48} {} element(s) read back wrong  <-- WRONG", where, bad);
        ++fails;
    }
}

/// @brief One count against its expected value, printed either way.
static void expect_eq(const char* where, long got, long want)
{
    println("  {:<48} {} (expect {}){}", where, got, want, got == want ? "" : "  <-- WRONG");
    if (got != want)
    {
        ++fails;
    }
}

/// @brief One count against a bound, printed either way.
static void expect_le(const char* where, long got, long bound)
{
    println("  {:<48} {} (expect <= {}){}", where, got, bound, got <= bound ? "" : "  <-- WRONG");
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
/// @brief Identifiers in use (second_value_stats()): at a sequential point,
///        exactly the registered values.
static long in_use()
{
    return static_cast<long>(Arr::second_value_stats().in_use());
}

/// @brief --rounds (child, run under a lowered CHUNK_CST_VAL2_ID_LIMIT):
///        @p rounds rounds of @p nch new constants, a reclaim after each.
///        Returning means it ran to the end.
static int run_rounds_child(long nch, long rounds)
{
    const long s = static_cast<long>(Arr(0).chunk_size());
    Arr        a(static_cast<int>(nch * s));
    for (long r = 0; r < rounds; ++r)
    {
        fill(a, nch, s, 1.0e6 + 1000. * static_cast<double>(r));
        (void)reclaim();
    }
    // second_value_stats() reports the lowered bound (read when the registry
    // was first used).
    expect_eq("child: capacity (CHUNK_CST_VAL2_ID_LIMIT)", static_cast<long>(Arr::second_value_stats().capacity),
              3 * nch);
    expect_values("child: last round", a, nch, s, 1.0e6 + 1000. * static_cast<double>(rounds - 1),
                  1.0e6 + 1000. * static_cast<double>(rounds - 1));
    println("[rounds] {} distinct second values, reclaimed each round: RAN TO THE END", nch * rounds);
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
    println("F  limit {} identifiers, {} distinct values        {}{}", limit, 12 * nch,
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
    const long s   = static_cast<long>(Arr(0).chunk_size()); // empty: no chunk, so no registry reference

    println("=== cst2_reclaim_validation  chunks={}  chunk_size={}  threads={} ===", nch, s, CHUNK_PMP_MAX_THREADS);
    {
        Arr a(static_cast<int>(nch * s));

        // A: rounds at new constants.
        println("A  rounds at new constants");
        const SecondValueStats before_a = Arr::second_value_stats();
        long                   freed_a  = 0;
        for (int r = 0; r < 4; ++r)
        {
            const double base = 1000. * (r + 1);
            fill(a, nch, s, base);
            std::string where;
            where = std::format("round {}: entries before the reclaim", r);
            expect_eq(where.c_str(), entries(), r == 0 ? nch : 2 * nch);
            where            = std::format("round {}: freed", r);
            const long freed = reclaim();
            freed_a += freed;
            expect_eq(where.c_str(), freed, r == 0 ? 0 : nch);
            where = std::format("round {}: entries after", r);
            expect_eq(where.c_str(), entries(), nch);
            where = std::format("round {}: identifiers in use (stats)", r);
            expect_eq(where.c_str(), in_use(), entries());
            where = std::format("round {}: identifiers handed out", r);
            expect_le(where.c_str(), handed_out(), 2 * nch);
            where = std::format("round {}: values", r);
            expect_values(where.c_str(), a, nch, s, base, base);
        }
        const SecondValueStats after_a = Arr::second_value_stats();
        expect_eq("stats: reclaims counted", static_cast<long>(after_a.reclaims - before_a.reclaims), 4);
        expect_eq("stats: freed counted", static_cast<long>(after_a.freed - before_a.freed), freed_a);
        expect_eq("stats: none refused", static_cast<long>(after_a.refused() - before_a.refused()), 0);
        expect_eq("stats: free + in use = handed out", static_cast<long>(after_a.free + after_a.in_use()),
                  handed_out());

        // B: half the chunks rewritten.
        println("B  even chunks rewritten");
        fill(a, nch, s, 5000.);
        (void)reclaim();
        fill(a, nch, s, 6000., true);
        expect_eq("freed", reclaim(), (nch + 1) / 2);
        expect_eq("entries after", entries(), nch);
        expect_values("values", a, nch, s, 6000., 5000.);

        // C: a chunk materialised again keeps its second value.
        println("C  materialised chunk");
        fill(a, nch, s, 7000.);
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
            println("  {:<48} reads {:g}  <-- WRONG", "the changed element", static_cast<double>(a[1]));
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
        expect_values("values", a, nch, s, 7000., 7000.);

        // D: a copy outlives the original.
        println("D  copy");
        fill(a, nch, s, 8000.);
        (void)reclaim();
        {
            Arr b(a);
            fill(a, nch, s, 9000.);
            expect_eq("freed (the copy still refers to them)", reclaim(), 0);
            expect_values("copy: values", b, nch, s, 8000., 8000.);
            expect_values("original: values", a, nch, s, 9000., 9000.);
        }
        expect_eq("freed (copy destroyed)", reclaim(), nch);
        expect_values("original after the copy: values", a, nch, s, 9000., 9000.);

        // E: refused inside a parallel region.
        if constexpr (build_config::use_openmp)
        {
            println("E  inside a parallel region");
            fill(a, nch, s, 10000.);
            const std::size_t refused_before = Arr::second_value_stats().refused_parallel;
            long              in_region      = -1;
            long              why            = -1; // the ReclaimRefusal, -1 for none
            int               team           = 0;
#pragma omp parallel
            {
#pragma omp masked
                {
                    team             = CHUNK_PMP_NUM_THREADS;
                    const auto tried = Arr::try_reclaim_second_values();
                    in_region        = static_cast<long>(tried.value_or(0));
                    why              = tried ? -1 : static_cast<long>(tried.error());
                }
            }
            // A team of one is an inactive region: no other thread, the call
            // is a sequential point and runs.
            const bool active = team > 1;
            expect_eq(active ? "freed inside the (active) region" : "freed inside the (inactive) region", in_region,
                      active ? 0 : nch);
            expect_eq("try_reclaim_second_values: the refusal says why", why,
                      active ? static_cast<long>(ReclaimRefusal::parallel_region) : -1);
            // The refusal is counted: 0 freed, but not "nothing to free".
            expect_eq("stats: refused in a parallel region",
                      static_cast<long>(Arr::second_value_stats().refused_parallel - refused_before), active ? 1 : 0);
            expect_eq("freed after it", reclaim(), active ? nch : 0);
            expect_values("values", a, nch, s, 10000., 10000.);
        }

        // G: reclaim across a re-chunk (CHUNK_DYNAMIC). The re-chunk rebuilds
        // every chunk and its second-value slot at the new geometry; a reclaim
        // after it must keep what the new chunks hold and nothing else.
        if constexpr (build_config::chunk_dynamic)
        {
            println("G  re-chunk");
            const unsigned ambient = active_chunk_size();
            const long     s2      = 2 * s;
            const long     nch2    = nch / 2; // the same elements, half as many chunks
            fill(a, nch, s, 11000.);
            (void)reclaim();
            a.rechunk_to(static_cast<unsigned int>(s2));
            expect_eq("chunk size after rechunk_to(2S)", static_cast<long>(a.chunk_size()), s2);
            // Each new chunk spans two old constants: materialised, so the
            // reclaim may free them all; the values must not change.
            (void)reclaim();
            expect_values("values after the re-chunk and a reclaim", a, nch, s, 11000., 11000.);
            expect_eq("identifiers in use = entries", in_use(), entries());
            // Constants at the new geometry: one per new chunk survives.
            fill(a, nch2, s2, 12000.);
            (void)reclaim();
            expect_eq("entries after a refill at 2S and a reclaim", entries(), nch2);
            expect_eq("identifiers in use = entries", in_use(), entries());
            expect_values("values at 2S", a, nch2, s2, 12000., 12000.);
            // Back to S, then constants again.
            a.rechunk_to(static_cast<unsigned int>(s));
            fill(a, nch, s, 13000.);
            (void)reclaim();
            expect_eq("entries back at S", entries(), nch);
            expect_values("values back at S", a, nch, s, 13000., 13000.);
            // A reclaim while a re-chunk rebuild runs is refused, and counted.
            const std::size_t refused_before = Arr::second_value_stats().refused_rechunk;
            g_in_rechunk_rebuild.store(true, std::memory_order_release);
            const long during = reclaim();
            const auto tried  = Arr::try_reclaim_second_values();
            g_in_rechunk_rebuild.store(false, std::memory_order_release);
            expect_eq("freed during a re-chunk rebuild", during, 0);
            expect_eq("try_reclaim_second_values during a re-chunk: the refusal says why",
                      tried ? -1L : static_cast<long>(tried.error()),
                      static_cast<long>(ReclaimRefusal::rechunk_in_progress));
            expect_eq("stats: refused during a re-chunk",
                      static_cast<long>(Arr::second_value_stats().refused_rechunk - refused_before), 2);
            set_active_chunk_size(ambient);
            set_thread_chunk_size(ambient);
        }
    }
    expect_eq("all arrays destroyed: entries", entries(), 0);

    // F: under a lowered identifier limit, reclaiming rounds run to the end.
    check_rounds_under_limit(argv[0], nch);

    println("=== {} ===", fails ? "CST2_RECLAIM FAILED" : "CST2_RECLAIM OK");
    return fails ? 1 : 0;
}

} // namespace validation::cst2_reclaim_validation
