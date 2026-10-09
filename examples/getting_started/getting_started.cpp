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

// The program of docs/getting_started.md: one compressed array, written in
// parallel, read back, folded, and the library shut down. Exit status 0 when
// every value reads back as written.

#include <print>   // std::println
#include <utility> // std::as_const

#include "chunk/array_chunk2A.h" // shrink_array::ArrayChunk2A
#include "chunk/chunk_std.h"     // release_chunk_pools, destroy_block_chunk
#include "chunk/config.h"        // sa::build_config::has_not_ompt
#include "chunk/manage_ompt.h"   // begin_parallel_phase, finalize_parallel_phase
#include "chunk/param_chunk.h"   // ParamChunk::init_indexes

namespace sa = shrink_array;

// Without an OMPT-enabled OpenMP runtime (GCC's libgomp), the program marks
// its parallel phases itself: begin_parallel_phase() by every thread at the
// top of a region, finalize_parallel_phase() once the region is over. With
// OMPT (Clang, oneAPI) the runtime's callbacks do it and these compile to
// nothing; in a serial build too.
static void begin_phase()
{
    if constexpr (sa::build_config::has_not_ompt)
    {
        sa::begin_parallel_phase();
    }
}
static void end_phase()
{
    if constexpr (sa::build_config::has_not_ompt)
    {
        sa::finalize_parallel_phase();
    }
}

// What element i holds after the write loop below.
static double expected(int i, int n)
{
    if (i < n / 8)
    {
        return 1.0 + static_cast<double>(i % 7); // varied values: these chunks keep a buffer
    }
    if (i < n / 2)
    {
        return 3.5; // one constant: these chunks compress
    }
    return 0.0; // never written: compressed from the start
}

int main()
{
    sa::ParamChunk::init_indexes(true); // once, before the first array

    int bad = 0;
    {
        const int                n = 1 << 24; // 16 M doubles: 128 MiB as a std::vector
        sa::ArrayChunk2A<double> a(n);        // every element 0.0, no buffer allocated yet

        // Write, in parallel. a[i] is a write proxy: it decides whether the
        // chunk must be materialised, and queues it for compression when it
        // becomes uniform.
#pragma omp parallel
        {
            begin_phase();
#pragma omp for
            for (int i = 0; i < n / 2; ++i)
            {
                a[i] = expected(i, n);
            }
        }
        end_phase();

        // Read, through a const reference: operator[] then returns a double.
        const auto& ca = std::as_const(a);
#pragma omp parallel for reduction(+ : bad)
        for (int i = 0; i < n; ++i)
        {
            if (ca[i] != expected(i, n))
            {
                ++bad;
            }
        }

        a.shrink_to_fit(); // fold every chunk that has become constant (sequential point)
        std::println("n = {} elements, {} wrong, chunks holding a buffer: {:.1f} %", n, bad, a.stats());
    } // the array releases its chunks here

    sa::release_chunk_pools(); // give the reuse pools back
    sa::destroy_block_chunk();
    return bad == 0 ? 0 : 1;
}
