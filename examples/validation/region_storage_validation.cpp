/*
 * Copyright 2024-2026 CEA (www.cea.fr)
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
 * @file region_storage_validation.cpp
 * @brief An array created, grown, shrunk and grown again INSIDE a parallel
 *        region -- one per thread, and one under `omp single` -- reads back
 *        exactly what std::vector would.
 *
 * @details The storage paths (create, resize, resize with a value, the
 *          written-chunk count) used to fall back on an orphaned `omp for` when
 *          called inside a region. That is right only when the whole team runs
 *          it for the same array; called by ONE thread of the team -- under
 *          `omp single` or `omp masked`, the way a program grows an array in the
 *          middle of a parallel section -- the loop's implicit barrier is one
 *          the rest of the team never reaches (a hang; with `nowait`, 1/N of the
 *          range silently done).
 *
 *          Only one thread at a time for those: a resize is a sequential point
 *          of the library (it drains the deferred ops and runs a compression
 *          pass over lists shared by every array of the element type), so
 *          resizing several arrays from several threads at once is outside its
 *          contract. The other threads wait, at single's barrier or at the
 *          region's end.
 *
 *          The last case is the whole team at once: every thread builds, fills,
 *          reads back and destroys work arrays of its own, while the team also
 *          writes a shared array. Construction and destruction take the
 *          cold-path lock (internal/cold_path_lock.h); the element accesses
 *          take none. Before the lock, two constructors grew the per-array
 *          tables at once (heap corruption, a crash in most runs).
 *
 *          Each case checks every element. A hang shows as the ctest timeout.
 */

#include <type_traits> // std::conditional_t
#include <utility>     // std::as_const

#include "chunk/internal/chunk_traits.h" // forward decls + default IntType = int
#include "chunk/zero_array.h"            // zero::Array
#include "chunk/zero_one_value_array.h"  // zero_one_value::Array
#include "chunk/zero_any_array.h"        // zero_any::Array
#include "chunk/param_chunk.h"           // ParamChunk::init_indexes
#include "chunk/config.h"                // chunk_config::use_zero*, build_config::has_not_ompt
#include "chunk/manage_ompt.h"           // begin_parallel_phase / finalize_parallel_phase
#include "chunk/internal/PMP_CHUNK.h"    // CHUNK_PMP_MAX_THREADS, CHUNK_PMP_THREAD_RANK
#include "chunk/print.h"                 // shrink_array::println

#include "validation_harness.h" // run() is called by validation_driver.cpp

namespace validation::region_storage_validation
{

using namespace shrink_array;

namespace
{

using Arr = std::conditional_t<
    chunk_config::use_zero_any,
    zero_any::Array<double>,
    std::conditional_t<chunk_config::use_zero_one_value, zero_one_value::Array<double>, zero::Array<double>>>;

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

/// @brief Elements of [@p first, @p last) that do not read @p want.
long mismatches(const Arr& arr, int first, int last, double want)
{
    long bad = 0;
    for (int i = first; i < last; ++i)
    {
        if (static_cast<double>(std::as_const(arr)[i]) != want)
        {
            ++bad;
        }
    }
    return bad;
}

/// @brief An array to grow and the one whose written chunks are counted.
struct Arrays
{
    explicit Arrays(int count) : a(count), w(count) {}
    Arr a;
    Arr w;
};

/**
 * @brief The storage life of an array created at @p count, on the calling thread:
 *        grown with and without a value, shrunk after writes, grown again (the
 *        regrown range must read 0); and the written-chunk count of @p arr.w.
 * @return the number of elements (or counts) that came out wrong
 */
long life_of_one_array(Arrays& arr, int count)
{
    long bad   = 0;
    Arr& arr_a = arr.a;

    bad += mismatches(arr_a, 0, count, 0.0); // as created (rebuild_storage)

    arr_a.resize(2 * count); // growth: new chunks constructed and initialised
    bad += mismatches(arr_a, 0, 2 * count, 0.0);

    arr_a.resize(3 * count, 2.5); // growth with a value: [2n, 3n) at 2.5
    bad += mismatches(arr_a, 0, 2 * count, 0.0);
    bad += mismatches(arr_a, 2 * count, 3 * count, 2.5);

    for (int i = 0; i < 3 * count; ++i) // every element non-zero
    {
        arr_a[i] = 1.0 + i;
    }
    arr_a.resize(count / 3 + 1); // shrink inside a chunk
    arr_a.resize(2 * count);     // regrown range: zeros, not the old values
    bad += mismatches(arr_a, count / 3 + 1, 2 * count, 0.0);

    // One element written in each of the first m chunks: m chunks written.
    Arr&      arr_w = arr.w;
    const int csz   = static_cast<int>(arr_w.chunk_size());
    const int half  = (count / csz) / 2;
    for (int c = 0; c < half; ++c)
    {
        arr_w[c * csz] = 7.0;
    }
    if (arr_w.get_nb_chunk_written() != half)
    {
        ++bad;
    }
    return bad;
}

/// @brief What element @p index holds after round @p round of thread @p rank:
///        by chunk of 256, zeros, one constant, or values that all differ -- so
///        that the passes compress and the writes materialise.
double work_value(int rank, int round, int index)
{
    const int kind = (index / 256 + round) % 3;
    if (kind == 0)
    {
        return 0.0;
    }
    return kind == 1 ? 7.0 + rank : 1.0 + index + rank + round;
}

/// @brief One thread's work arrays, @p rounds times: one built, written twice
///        and read back, a second one built from it while the first is alive,
///        both destroyed.
/// @param rank a number of this thread's own, to tell its values from the others'
long work_arrays(int count, int rank, int rounds)
{
    long bad = 0;
    for (int round = 0; round < rounds; ++round)
    {
        const int len = count / 16 + 37 * ((rank + round) % 5);
        Arr       mine(len);
        for (int pass = 0; pass < 2; ++pass)
        {
            for (int i = 0; i < len; ++i)
            {
                mine[i] = work_value(rank, round + pass, i);
            }
        }
        Arr other(len / 2);
        for (int i = 0; i < len / 2; ++i)
        {
            other[i] = static_cast<double>(std::as_const(mine)[i]);
        }
        for (int i = 0; i < len; ++i)
        {
            bad += static_cast<double>(std::as_const(mine)[i]) != work_value(rank, round + 1, i);
        }
        for (int i = 0; i < len / 2; ++i)
        {
            bad += static_cast<double>(std::as_const(other)[i]) != work_value(rank, round + 1, i);
        }
    }
    return bad;
}

int report(const char* what, int count, long bad)
{
    println("[{}/region-storage] {:<22} n={:7} : {}", variant_name(), what, count, bad ? "FAIL" : "ok");
    if (bad)
    {
        println("  {} wrong element(s) or count(s)", bad);
    }
    return bad ? 1 : 0;
}

} // namespace

int run(int /*argc*/, char** /*argv*/)
{
    ParamChunk::init_indexes(true);
    println("=== region_storage_validation [variant = {}]  threads={} ===", variant_name(), CHUNK_PMP_MAX_THREADS);

    int fails = 0;
    // Below and above the 500-chunk threshold of the init loops, at any
    // geometry this suite builds (64 to 1024 elements per chunk).
    for (const int count : {20000, 700000})
    {
        {
            Arrays arr(count);
            fails += report("serial", count, life_of_one_array(arr, count));
        }

        long bad_single = 0;
#pragma omp parallel
        {
#pragma omp single
            {
                Arrays arr(count); // one thread of the team creates and uses it
                bad_single = life_of_one_array(arr, count);
            }
        }
        fails += report("omp single", count, bad_single);

        long bad_masked = 0;
#pragma omp parallel
        {
#pragma omp masked
            {
                Arrays arr(count); // the primary thread; the others go to the region's end
                bad_masked = life_of_one_array(arr, count);
            }
        }
        fails += report("omp masked", count, bad_masked);

        // The whole team: work arrays of each thread's own, built and destroyed
        // at the same time, between two team loops on a shared array.
        long bad_each = 0;
        {
            constexpr int rounds = 6;
            Arr           shared(count);
#pragma omp parallel reduction(+ : bad_each)
            {
                if constexpr (build_config::has_not_ompt)
                {
                    begin_parallel_phase();
                }
                const int rank = CHUNK_PMP_THREAD_RANK;
#pragma omp for schedule(static)
                for (int i = 0; i < count; ++i)
                {
                    shared[i] = work_value(0, 0, i);
                }
                bad_each += work_arrays(count, rank, rounds);
#pragma omp for schedule(static)
                for (int i = 0; i < count; ++i)
                {
                    shared[i] = work_value(0, 1, i);
                }
            }
            if constexpr (build_config::has_not_ompt)
            {
                finalize_parallel_phase();
            }
            for (int i = 0; i < count; ++i)
            {
                bad_each += static_cast<double>(std::as_const(shared)[i]) != work_value(0, 1, i);
            }
        }
        fails += report("omp parallel (each)", count, bad_each);
    }

    println("{}", fails ? "region_storage_validation: FAILED" : "region_storage_validation: OK");
    return fails ? 1 : 0;
}

} // namespace validation::region_storage_validation
