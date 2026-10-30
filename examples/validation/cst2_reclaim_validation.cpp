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

/// @brief The constant chunk @p num_chunk holds after a fill at @p base.
static double cst(double base, long num_chunk)
{
    return base + static_cast<double>(num_chunk);
}

/// @brief Write chunk j of @p arr whole at cst(base, j) -- the chunks for which
///        @p only_even is false or j is even -- then the between-region pass.
static void fill(Arr& arr, long nch, long chunk_sz, double base, bool only_even = false)
{
    const int ni = static_cast<int>(nch * chunk_sz);
#pragma omp parallel
    {
        begin_phase();
#pragma omp for schedule(static)
        for (int i = 0; i < ni; ++i)
        {
            const long num_chunk = i / chunk_sz;
            if (!only_even || num_chunk % 2 == 0)
            {
                arr[i] = cst(base, num_chunk);
            }
        }
    }
    end_phase();
}

static int fails = 0;

/// @brief Every element of @p arr against the constant of its chunk:
///        cst(even_base, j) for an even chunk, cst(odd_base, j) for an odd one.
static void expect_values(const char* where, const Arr& arr, long nch, long chunk_sz, double even_base, double odd_base)
{
    long bad = 0;
    for (long i = 0; i < nch * chunk_sz; ++i)
    {
        const long num_chunk = i / chunk_sz;
        bad += (arr[static_cast<int>(i)] != cst(num_chunk % 2 == 0 ? even_base : odd_base, num_chunk)) ? 1 : 0;
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
    const long chunk_sz = static_cast<long>(Arr(0).chunk_size());
    Arr        arr(static_cast<int>(nch * chunk_sz));
    for (long r = 0; r < rounds; ++r)
    {
        fill(arr, nch, chunk_sz, 1.0e6 + 1000. * static_cast<double>(r));
        (void)reclaim();
    }
    // second_value_stats() reports the lowered bound (read when the registry
    // was first used).
    expect_eq("child: capacity (CHUNK_CST_VAL2_ID_LIMIT)", static_cast<long>(Arr::second_value_stats().capacity),
              3 * nch);
    expect_values("child: last round", arr, nch, chunk_sz, 1.0e6 + 1000. * static_cast<double>(rounds - 1),
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

    const long nch      = argc > 1 ? shrink_array::arg_number<long>(argv[1], 16, "chunks") : 16;
    const long chunk_sz = static_cast<long>(Arr(0).chunk_size()); // empty: no chunk, so no registry reference

    println("=== cst2_reclaim_validation  chunks={}  chunk_size={}  threads={} ===", nch, chunk_sz,
            CHUNK_PMP_MAX_THREADS);
    {
        Arr arr(static_cast<int>(nch * chunk_sz));

        // A: rounds at new constants.
        println("A  rounds at new constants");
        const SecondValueStats before_a = Arr::second_value_stats();
        long                   freed_a  = 0;
        for (int r = 0; r < 4; ++r)
        {
            const double base = 1000. * (r + 1);
            fill(arr, nch, chunk_sz, base);
            char where[64];
            std::snprintf(where, sizeof where, "round %d: entries before the reclaim", r);
            expect_eq(where, entries(), r == 0 ? nch : 2 * nch);
            std::snprintf(where, sizeof where, "round %d: freed", r);
            const long freed = reclaim();
            freed_a += freed;
            expect_eq(where, freed, r == 0 ? 0 : nch);
            std::snprintf(where, sizeof where, "round %d: entries after", r);
            expect_eq(where, entries(), nch);
            std::snprintf(where, sizeof where, "round %d: identifiers in use (stats)", r);
            expect_eq(where, in_use(), entries());
            std::snprintf(where, sizeof where, "round %d: identifiers handed out", r);
            expect_le(where, handed_out(), 2 * nch);
            std::snprintf(where, sizeof where, "round %d: values", r);
            expect_values(where, arr, nch, chunk_sz, base, base);
        }
        const SecondValueStats after_a = Arr::second_value_stats();
        expect_eq("stats: reclaims counted", static_cast<long>(after_a.reclaims - before_a.reclaims), 4);
        expect_eq("stats: freed counted", static_cast<long>(after_a.freed - before_a.freed), freed_a);
        expect_eq("stats: none refused", static_cast<long>(after_a.refused() - before_a.refused()), 0);
        expect_eq("stats: free + in use = handed out", static_cast<long>(after_a.free + after_a.in_use()),
                  handed_out());

        // B: half the chunks rewritten.
        println("B  even chunks rewritten");
        fill(arr, nch, chunk_sz, 5000.);
        (void)reclaim();
        fill(arr, nch, chunk_sz, 6000., true);
        expect_eq("freed", reclaim(), (nch + 1) / 2);
        expect_eq("entries after", entries(), nch);
        expect_values("values", arr, nch, chunk_sz, 6000., 5000.);

        // C: a chunk materialised again keeps its second value.
        println("C  materialised chunk");
        fill(arr, nch, chunk_sz, 7000.);
        (void)reclaim();
#pragma omp parallel
        {
            begin_phase();
#pragma omp single
            arr[1] = 1.5; // chunk 0 holds a buffer again
        }
        end_phase();
        (void)reclaim();
        if (arr[1] != 1.5)
        {
            println("  {:<48} reads {:g}  <-- WRONG", "the changed element", static_cast<double>(arr[1]));
            ++fails;
        }
#pragma omp parallel
        {
            begin_phase();
#pragma omp single
            arr[1] = cst(7000., 0); // back to the chunk's constant: folds again
        }
        end_phase();
        (void)reclaim();
        expect_values("values", arr, nch, chunk_sz, 7000., 7000.);

        // D: a copy outlives the original.
        println("D  copy");
        fill(arr, nch, chunk_sz, 8000.);
        (void)reclaim();
        {
            Arr copy(arr);
            fill(arr, nch, chunk_sz, 9000.);
            expect_eq("freed (the copy still refers to them)", reclaim(), 0);
            expect_values("copy: values", copy, nch, chunk_sz, 8000., 8000.);
            expect_values("original: values", arr, nch, chunk_sz, 9000., 9000.);
        }
        expect_eq("freed (copy destroyed)", reclaim(), nch);
        expect_values("original after the copy: values", arr, nch, chunk_sz, 9000., 9000.);

        // E: refused inside a parallel region.
        if constexpr (build_config::use_openmp)
        {
            println("E  inside a parallel region");
            fill(arr, nch, chunk_sz, 10000.);
            const std::size_t refused_before = Arr::second_value_stats().refused_parallel;
            long              in_region      = -1;
            int               team           = 0;
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
            // The refusal is counted: 0 freed, but not "nothing to free".
            expect_eq("stats: refused in a parallel region",
                      static_cast<long>(Arr::second_value_stats().refused_parallel - refused_before), active ? 1 : 0);
            expect_eq("freed after it", reclaim(), active ? nch : 0);
            expect_values("values", arr, nch, chunk_sz, 10000., 10000.);
        }

        // G: reclaim across a re-chunk (CHUNK_DYNAMIC). The re-chunk rebuilds
        // every chunk and its second-value slot at the new geometry; a reclaim
        // after it must keep what the new chunks hold and nothing else.
        if constexpr (build_config::chunk_dynamic)
        {
            println("G  re-chunk");
            const unsigned ambient = active_chunk_size();
            const long     s2      = 2 * chunk_sz;
            const long     nch2    = nch / 2; // the same elements, half as many chunks
            fill(arr, nch, chunk_sz, 11000.);
            (void)reclaim();
            arr.rechunk_to(static_cast<unsigned int>(s2));
            expect_eq("chunk size after rechunk_to(2S)", static_cast<long>(arr.chunk_size()), s2);
            // Each new chunk spans two old constants: materialised, so the
            // reclaim may free them all; the values must not change.
            (void)reclaim();
            expect_values("values after the re-chunk and a reclaim", arr, nch, chunk_sz, 11000., 11000.);
            expect_eq("identifiers in use = entries", in_use(), entries());
            // Constants at the new geometry: one per new chunk survives.
            fill(arr, nch2, s2, 12000.);
            (void)reclaim();
            expect_eq("entries after a refill at 2S and a reclaim", entries(), nch2);
            expect_eq("identifiers in use = entries", in_use(), entries());
            expect_values("values at 2S", arr, nch2, s2, 12000., 12000.);
            // Back to S, then constants again.
            arr.rechunk_to(static_cast<unsigned int>(chunk_sz));
            fill(arr, nch, chunk_sz, 13000.);
            (void)reclaim();
            expect_eq("entries back at S", entries(), nch);
            expect_values("values back at S", arr, nch, chunk_sz, 13000., 13000.);
            // A reclaim while a re-chunk rebuild runs is refused, and counted.
            const std::size_t refused_before = Arr::second_value_stats().refused_rechunk;
            g_in_rechunk_rebuild.store(true, std::memory_order_release);
            const long during = reclaim();
            g_in_rechunk_rebuild.store(false, std::memory_order_release);
            expect_eq("freed during a re-chunk rebuild", during, 0);
            expect_eq("stats: refused during a re-chunk",
                      static_cast<long>(Arr::second_value_stats().refused_rechunk - refused_before), 1);
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
