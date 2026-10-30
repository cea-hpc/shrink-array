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
// Compression round-trip harness. Variant-agnostic: tests zero / zero_one_value /
// zero_any depending on the compile-time macro (CHUNK_VARIANT_ZERO*), like
// the other harnesses here.
// ============================================================================
//
// Guards ONE property, at whatever thread count OMP_NUM_THREADS gives: after a
// full compression pass and a partial re-materialisation, every element of an
// array reads back exactly what was last written into it, and the library
// tears down cleanly afterwards.
//
// WHAT IT EXERCISES
//
//   Per element type (double, int, bool), an array is materialised everywhere,
//   then overwritten with a run pattern -- runs of the first constant (0), runs
//   of a second constant, runs of varied values -- so that a compression pass
//   has both kinds of constant block to collapse and materialised blocks to
//   leave alone. compress() is then driven from inside a parallel region, one
//   array per thread, the way the bench does it. A quarter of each array is
//   rewritten afterwards, which re-materialises blocks the pass had just
//   compressed. Every element is then compared against the expected value.
//
//   A second double array never declares a second value (zero_one_value's
//   set_second_value): its second constant then coincides with the first, the
//   corner every two-hypothesis scan has to handle (a run of 0 must be filed
//   as first-constant ONLY, never cleared twice).
//
//   The teardown at the end is the library's own (reuse-list release, allocator
//   destruction), so a buffer freed twice or a chunk left pointing at a parked
//   buffer surfaces here rather than in the next program.
//
// THE PHASE PROTOCOL
//
//   Without OMPT (every GNU build) the library does not see region boundaries:
//   a write can be DEFERRED to the writing thread's queue (when the chunk's
//   candidate list is flagged dangerous), and it is the application that must
//   drain those queues between regions -- begin_parallel_phase() by every
//   thread inside the region, finalize_parallel_phase() outside it. Skip that
//   and a deferred write from one region lands after a direct write of the
//   next one: this harness lost `false` writes on zero_any/bool at the dyn
//   geometry and 8 threads (the only combination where the lists get flagged),
//   at exactly the indices the previous region had written `true`. The bench
//   has the same brackets (its *_g() wrappers). Under OMPT the callbacks do it
//   and the calls are no-ops, as they are in a serial build.
//
// WHAT IT DOES NOT COVER
//
//   No re-chunk (rechunk_validation) and no geometry sweep
//   (dyn_geometry_validation). One geometry, the one the executable was built
//   with.
//
// Compile OMP or non-OMP, static or dyn, check. Exit 0 = every element
// round-trips and the teardown returned.

#include <type_traits> // std::conditional_t

#include "chunk/chunk_traits.h"         // forward decls + default IntType = int
#include "chunk/zero_array.h"           // zero::Array
#include "chunk/zero_one_value_array.h" // zero_one_value::Array
#include "chunk/zero_any_array.h"       // zero_any::Array
#include "chunk/param_chunk.h"          // ParamChunk::init_indexes
#include "chunk/zero_chunk.h"           // release_chunk_pools, destroy_block_chunk
#include "chunk/config.h"               // chunk_config::use_zero*
#include "chunk/PMP_CHUNK.h"            // CHUNK_PMP_MAX_THREADS
#include "chunk/manage_ompt.h"          // begin_parallel_phase / finalize_parallel_phase
#include "chunk/internal/env_parse.h"   // env_number / arg_number (checked, not atoi)
#include "chunk/internal/print.h"       // shrink_array::println

// Defined by every executable of the library (the fast allocator reads it).
#include "validation_harness.h" // self_command; run() is called by validation_driver.cpp

namespace validation::compress_roundtrip
{

using namespace shrink_array;

// The phase brackets, no-ops under OMPT (the callbacks do it) and serially.
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
    std::conditional_t<chunk_config::use_zero_any,
                       zero_any::Array<T>,
                       std::conditional_t<chunk_config::use_zero_one_value, zero_one_value::Array<T>, zero::Array<T>>>;

constexpr const char* variant_name()
{
    if constexpr (chunk_config::use_zero_any)
    {
        return "zero_any";
    }
    else if constexpr (chunk_config::use_zero_one_value)
    {
        return "zero_one_value";
    }
    else
    {
        return "zero";
    }
}

// set_second_value only exists on zero_one_value / zero_any; no-op elsewhere.
template <typename Arr, typename Val>
void maybe_set_second_value(Arr& arr, const Val& value)
{
    if constexpr (requires(Arr probe) { probe.set_second_value(value); })
    {
        arr.set_second_value(value);
    }
}

// Runs of run_len elements: 0 -> first constant, 1 -> second constant, 2 -> varied.
// run_len is a multiple of every geometry the harness is built at, so a run is a
// whole number of chunks and the pass sees pure constant blocks.
constexpr long run_len = 4096;

template <typename T>
struct Scenario
{
    T second; // the second constant written in run 1
    T refill; // the value the first quarter is rewritten with after the pass
    T (*varied)(long index);
};

/**
 * @brief Materialise, pattern, compress, refill a quarter, verify.
 * @return 0 if every element reads back as expected, 1 otherwise.
 */
template <typename T>
static int round_trip(const char* label, long count, const Scenario<T>& sc, bool declare_second)
{
    Array<T> arr(count);
    if (declare_second)
    {
        maybe_set_second_value(arr, sc.second);
    }

    // Materialise everything first, so the pattern below overwrites live blocks
    // rather than filling fresh ones.
#pragma omp parallel
    {
        begin_phase();
#pragma omp for
        for (long i = 0; i < count; ++i)
        {
            arr[i] = sc.varied(i);
        }
    }
    end_phase();

#pragma omp parallel
    {
        begin_phase();
#pragma omp for
        for (long i = 0; i < count; ++i)
        {
            const long run = (i / run_len) % 3;
            if (run == 0)
            {
                arr[i] = T(0);
            }
            else if (run == 1)
            {
                arr[i] = sc.second;
            }
        }
    }
    end_phase();

    // The pass, driven the way the bench drives it: from inside a region.
#pragma omp parallel
    {
        begin_phase();
#pragma omp single
        arr.compress();
    }
    end_phase();

    // Re-materialise: the first quarter gets a value that is neither constant.
#pragma omp parallel
    {
        begin_phase();
#pragma omp for
        for (long i = 0; i < count / 4; ++i)
        {
            arr[i] = sc.refill;
        }
    }
    end_phase();

    long bad   = 0;
    long first = -1;
    T    got_first{};
    T    exp_first{};
    for (long i = 0; i < count; ++i)
    {
        const long run      = (i / run_len) % 3;
        T          expected = sc.varied(i);
        if (run == 0)
        {
            expected = T(0);
        }
        else if (run == 1)
        {
            expected = sc.second;
        }
        if (i < count / 4)
        {
            expected = sc.refill;
        }

        const T got = arr[i];
        if (got != expected)
        {
            if (first < 0)
            {
                first     = i;
                got_first = got;
                exp_first = expected;
            }
            ++bad;
        }
    }

    const long csz = static_cast<long>(arr.chunk_size());
    if (bad)
    {
        println("[{}/{}] FAIL: {}/{} mismatches, first at i={} (chunk #{}, offset {}, run {}) "
                "got={:g} expected={:g}",
                variant_name(), label, bad, count, first, first / csz, first % csz, (first / run_len) % 3,
                (double)got_first, (double)exp_first);
    }
    else
    {
        println("[{}/{}] OK: {} elements round-trip through compress() at chunk_size={}", variant_name(), label, count,
                csz);
    }

    arr.clear();
    return bad ? 1 : 0;
}

int run(int argc, char** argv)
{
    const long count = argc > 1 ? shrink_array::arg_number<long>(argv[1], 1L << 20, "N") : (1L << 20);

    ParamChunk::init_indexes(true);

    println("=== compress_roundtrip [variant = {}]  N={}  threads={} ===", variant_name(), count,
            CHUNK_PMP_MAX_THREADS);

    int fails = 0;

    const Scenario<double> sd{3.5, 9.0, [](long i) { return 1.0 + static_cast<double>(i); }};
    const Scenario<int>    si{7, 11, [](long i) { return static_cast<int>(i); }};
    const Scenario<bool>   sb{true, true, [](long i) { return (i % 3) == 0; }};

    fails += round_trip<double>("double", count, sd, /*declare_second=*/true);
    fails += round_trip<double>("double/no-second-value", count, sd, /*declare_second=*/false);
    fails += round_trip<int>("int", count, si, /*declare_second=*/true);
    fails += round_trip<bool>("bool", count, sb, /*declare_second=*/true);

    // The library's own teardown: releases the reuse lists (one region, the
    // frees shared out over the team) and the allocators. A double free or a
    // chunk still pointing at a parked buffer aborts here.
    // No #ifdef: release_chunk_pools() does whichever half this build needs --
    // the per-rank lists under OpenMP, the single allocator otherwise. See
    // ChunkPoolParallelInterface in zero_chunk.h.
    release_chunk_pools();
    destroy_block_chunk();

    println("\n[{}] {}", variant_name(),
            fails == 0 ? "=== ROUND-TRIP OK, TEARDOWN OK ===" : "=== ROUND-TRIP FAILURES ===");
    return fails;
}

} // namespace validation::compress_roundtrip
