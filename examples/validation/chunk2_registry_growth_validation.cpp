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
// Chunk2 second-value registry growth harness. Chunk2 only (ArrayChunk2<T>),
// default static size; serial and OpenMP builds.
// ============================================================================
//
// Guards ONE property: declaring a new second value never moves the constant
// arrays chunks already point at.
//
// THE DEFECT
//
//   Every distinct value given to ArrayChunk2::set_second_value() gets a
//   constant array in Chunk2<T>::tab_m_ptr_cst_val2, and a chunk compressed to
//   it stores that array's address as its `values`. The table was a
//   std::vector<std::array<T, cst_buffer_size>> grown by resize(): past its
//   capacity the vector MOVED every array and freed the old storage. Every
//   chunk of every other array compressed to its second value then pointed at
//   freed memory -- 15 360 wrong reads out of 16 384 in this harness (0, or
//   garbage like 4.65e-310), a heap-use-after-free under ASan. shrink_bench
//   never sees it: its arrays all declare their second value at start-up,
//   before anything compresses.
//
// WHAT IT DOES
//
//   1. `a` declares 1.0, is written 1.0 everywhere and compressed: every chunk
//      at its second value (premise, checked).
//   2. kOthers arrays are created, each declaring a NEW second value: the
//      registry grows past several vector capacities. Each one is written to
//      its value and compressed too.
//   3. Every element of every array is read back: its own value, exactly.
//   4. `a` is rewritten (materialised then constant again) and compressed once
//      more: the pass must still recognise the registry's array -- the chunks
//      are compressed again, and still read 1.0.
//
// EXIT STATUS: 0 when every read matches and the premises hold, 1 otherwise.

#include <memory> // std::unique_ptr, std::make_unique
#include <vector> // std::vector

#include "chunk/chunk_traits.h"   // forward decls + default ind_type = int
#include "chunk/array_chunk2.h"   // ArrayChunk2
#include "chunk/param_chunk.h"    // ParamChunk::init_indexes
#include "chunk/chunk_std.h"      // release_chunk_pools, destroy_block_chunk
#include "chunk/config.h"         // build_config::has_not_ompt
#include "chunk/PMP_CHUNK.h"      // CHUNK_PMP_MAX_THREADS
#include "chunk/manage_ompt.h"    // begin_parallel_phase / finalize_parallel_phase
#include "chunk/internal/print.h" // shrink_array::println

#include "validation_harness.h" // self_command; run() is called by validation_driver.cpp

namespace validation::chunk2_registry_growth_validation
{

using namespace shrink_array;

namespace
{

using Arr = ArrayChunk2<double>;
using Ind = Arr::ind_type;

constexpr int kChunks = 64; ///< chunks of `a`
constexpr int kOthers = 40; ///< arrays declaring a new second value each: several vector growths

void begin_phase()
{
    if constexpr (build_config::has_not_ompt)
    {
        shrink_array::begin_parallel_phase();
    }
}

void end_phase()
{
    if constexpr (build_config::has_not_ompt)
    {
        shrink_array::finalize_parallel_phase();
    }
}

/// @brief Write @p v everywhere from a region, then compress from a region.
void fill_and_compress(Arr& arr, double v)
{
    const long n = static_cast<long>(arr.size());
#pragma omp parallel
    {
        begin_phase();
#pragma omp for schedule(static)
        for (long i = 0; i < n; ++i)
        {
            arr[static_cast<Ind>(i)] = v;
        }
    }
    end_phase();
#pragma omp parallel
    {
        begin_phase();
#pragma omp single
        arr.compress();
    }
    end_phase();
}

/// @brief Chunks of @p arr still materialised.
int materialised(const Arr& arr)
{
    const long S = static_cast<long>(arr.chunk_size());
    const long n = static_cast<long>(arr.size());
    int        m = 0;
    for (long j = 0; j * S < n; ++j)
    {
        m += arr.written(static_cast<Ind>(j * S)) ? 1 : 0;
    }
    return m;
}

/// @brief Every element of @p arr equals @p v; prints the first few misses.
long check_all(const char* what, const Arr& arr, double v)
{
    long       bad = 0;
    const long n   = static_cast<long>(arr.size());
    for (long i = 0; i < n; ++i)
    {
        const double got = arr[static_cast<Ind>(i)];
        if (got != v)
        {
            if (bad < 4)
            {
                shrink_array::println("[chunk2/registry] FAIL {}[{}] = {:g}, expected {:g}", what, i, got, v);
            }
            ++bad;
        }
    }
    return bad;
}

} // namespace

int run(int /*argc*/, char** /*argv*/)
{
    ParamChunk::init_indexes(true);

    int fails = 0;
    {
        Arr        a;
        const long S = static_cast<long>(a.chunk_size());
        a.resize(static_cast<Ind>(kChunks * S));
        a.set_second_value(1.0);
        shrink_array::println("=== chunk2_registry_growth_validation  chunk_size={}  threads={}  others={} ===", S,
                              CHUNK_PMP_MAX_THREADS, kOthers);

        // 1. `a` compressed to its second value.
        fill_and_compress(a, 1.0);
        if (const int m = materialised(a); m != 0)
        {
            shrink_array::println(
                "[chunk2/registry] FAIL premise: {} of {} chunks of `a` still materialised after the pass", m, kChunks);
            ++fails;
        }

        // 2. New second values, one per array: the registry grows.
        std::vector<std::unique_ptr<Arr>> others;
        for (int k = 0; k < kOthers; ++k)
        {
            others.push_back(std::make_unique<Arr>(static_cast<Ind>(4 * S)));
            others.back()->set_second_value(2.0 + k);
            fill_and_compress(*others.back(), 2.0 + k);
        }

        // 3. Everything reads back.
        long bad = check_all("a", a, 1.0);
        for (int k = 0; k < kOthers; ++k)
        {
            bad += check_all("other", *others[static_cast<std::size_t>(k)], 2.0 + k);
        }
        if (bad)
        {
            shrink_array::println(
                "[chunk2/registry] FAIL {} wrong read(s) after the registry grew -- chunks pointing at "
                "a constant array that moved",
                bad);
            ++fails;
        }
        else
        {
            shrink_array::println("[chunk2/registry] OK: {} arrays, {} second values, every element reads back",
                                  kOthers + 1, kOthers + 1);
        }

        // 4. `a` materialised and compressed again: still recognised.
        a[0] = 7.0;
        a[0] = 1.0;
        fill_and_compress(a, 1.0);
        const int  m2   = materialised(a);
        const long bad2 = check_all("a (recompressed)", a, 1.0);
        if (m2 != 0 || bad2 != 0)
        {
            shrink_array::println(
                "[chunk2/registry] FAIL after recompression: {} chunk(s) materialised, {} wrong read(s)", m2, bad2);
            ++fails;
        }
        else
        {
            shrink_array::println("[chunk2/registry] OK: `a` recompressed to its second value after the growth");
        }
    }

    release_chunk_pools();
    destroy_block_chunk();

    shrink_array::println("\n[chunk2] {}", fails ? "=== REGISTRY GROWTH FAILED ===" : "=== REGISTRY GROWTH OK ===");
    return fails ? 1 : 0;
}

} // namespace validation::chunk2_registry_growth_validation
