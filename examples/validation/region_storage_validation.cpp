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
 *          Only one thread at a time: a resize is a sequential point of the
 *          library (it drains the deferred ops and runs a compression pass over
 *          lists shared by every array of the element type), so resizing
 *          several arrays from several threads at once is outside its contract,
 *          as is creating them so (create_common() takes no lock). The other
 *          threads wait, at single's barrier or at the region's end.
 *
 *          Each case checks every element. A hang shows as the ctest timeout.
 */

#include <type_traits> // std::conditional_t
#include <utility>     // std::as_const

#include "chunk/chunk_traits.h"   // forward decls + default ind_type = int
#include "chunk/array_chunk.h"    // ArrayChunk
#include "chunk/array_chunk2.h"   // ArrayChunk2
#include "chunk/array_chunk2A.h"  // ArrayChunk2A
#include "chunk/param_chunk.h"    // ParamChunk::init_indexes
#include "chunk/config.h"         // chunk_config::use_chunk*
#include "chunk/PMP_CHUNK.h"      // CHUNK_PMP_MAX_THREADS
#include "chunk/internal/print.h" // shrink_array::println

#include "validation_harness.h" // run() is called by validation_driver.cpp

namespace validation::region_storage_validation
{

using namespace shrink_array;

namespace
{

using Arr = std::conditional_t<chunk_config::use_chunk2a,
                               ArrayChunk2A<double>,
                               std::conditional_t<chunk_config::use_chunk2, ArrayChunk2<double>, ArrayChunk<double>>>;

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

/// @brief Elements of [b, e) that do not read @p want.
long mismatches(const Arr& a, int b, int e, double want)
{
    long bad = 0;
    for (int i = b; i < e; ++i)
    {
        if (static_cast<double>(std::as_const(a)[i]) != want)
        {
            ++bad;
        }
    }
    return bad;
}

/// @brief An array to grow and the one whose written chunks are counted.
struct Arrays
{
    explicit Arrays(int n) : a(n), w(n) {}
    Arr a;
    Arr w;
};

/**
 * @brief The storage life of an array created at @p n, on the calling thread:
 *        grown with and without a value, shrunk after writes, grown again (the
 *        regrown range must read 0); and the written-chunk count of @p arr.w.
 * @return the number of elements (or counts) that came out wrong
 */
long life_of_one_array(Arrays& arr, int n)
{
    long bad = 0;
    Arr& a   = arr.a;

    bad += mismatches(a, 0, n, 0.0); // as created (rebuild_storage)

    a.resize(2 * n); // growth: new chunks constructed and initialised
    bad += mismatches(a, 0, 2 * n, 0.0);

    a.resize(3 * n, 2.5); // growth with a value: [2n, 3n) at 2.5
    bad += mismatches(a, 0, 2 * n, 0.0);
    bad += mismatches(a, 2 * n, 3 * n, 2.5);

    for (int i = 0; i < 3 * n; ++i) // every element non-zero
    {
        a[i] = 1.0 + i;
    }
    a.resize(n / 3 + 1); // shrink inside a chunk
    a.resize(2 * n);     // regrown range: zeros, not the old values
    bad += mismatches(a, n / 3 + 1, 2 * n, 0.0);

    // One element written in each of the first m chunks: m chunks written.
    Arr&      w   = arr.w;
    const int csz = static_cast<int>(w.chunk_size());
    const int m   = (n / csz) / 2;
    for (int c = 0; c < m; ++c)
    {
        w[c * csz] = 7.0;
    }
    if (w.get_nb_chunk_written() != m)
    {
        ++bad;
    }
    return bad;
}

int report(const char* what, int n, long bad)
{
    println("[{}/region-storage] {:<22} n={:7} : {}", variant_name(), what, n, bad ? "FAIL" : "ok");
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
    for (const int n : {20000, 700000})
    {
        {
            Arrays arr(n);
            fails += report("serial", n, life_of_one_array(arr, n));
        }

        long bad_single = 0;
#pragma omp parallel
        {
#pragma omp single
            {
                Arrays arr(n); // one thread of the team creates and uses it
                bad_single = life_of_one_array(arr, n);
            }
        }
        fails += report("omp single", n, bad_single);

        long bad_masked = 0;
#pragma omp parallel
        {
#pragma omp masked
            {
                Arrays arr(n); // the primary thread; the others go to the region's end
                bad_masked = life_of_one_array(arr, n);
            }
        }
        fails += report("omp masked", n, bad_masked);
    }

    println("{}", fails ? "region_storage_validation: FAILED" : "region_storage_validation: OK");
    return fails ? 1 : 0;
}

} // namespace validation::region_storage_validation
