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

// getting_started.cpp with `import shrink_array;` in place of the #includes
// (chunk/modules/shrink_array.cppm, SHRINK_ARRAY_MODULES=ON). The same program:
// one compressed array, written in parallel, read back, folded, and the
// library shut down. Exit status 0 when every value reads back as written.

#include <utility> // std::as_const

import shrink_array; // zero_any::Array, parallel_phase, ParamChunk, println, release_chunk_pools, ...

namespace sa = shrink_array;
using shrink_array::println;

// What element i holds after the write loop below.
static double expected(int index, int count)
{
    if (index < count / 8)
    {
        return 1.0 + static_cast<double>(index % 7); // varied values: these chunks keep a buffer
    }
    if (index < count / 2)
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
        const int                   count = 1 << 24; // 16 M doubles: 128 MiB as a std::vector
        sa::zero_any::Array<double> arr(count);      // every element 0.0, no buffer allocated yet

        // Write, in parallel. a[i] is a write proxy: it decides whether the
        // chunk must be materialised, and queues it for compression when it
        // becomes uniform. parallel_phase() opens the region and, without an
        // OMPT-enabled runtime (GCC's libgomp), marks the phase for the
        // library: begin_parallel_phase() by every thread, then
        // finalize_parallel_phase() once the region is over. With OMPT
        // (Clang, oneAPI) the runtime's callbacks do that.
        sa::parallel_phase(
            [&]
            {
#pragma omp for
                for (int i = 0; i < count / 2; ++i)
                {
                    arr[i] = expected(i, count);
                }
            });

        // Read, through a const reference: operator[] then returns a double.
        const auto& ca = std::as_const(arr);
#pragma omp parallel for reduction(+ : bad)
        for (int i = 0; i < count; ++i)
        {
            if (ca[i] != expected(i, count))
            {
                ++bad;
            }
        }

        arr.shrink_to_fit(); // fold every chunk that has become constant (sequential point)
        println("n = {} elements, {} wrong, chunks holding a buffer: {:.1f} %", count, bad, arr.stats());
    } // the array releases its chunks here

    sa::release_chunk_pools(); // give the reuse pools back
    sa::destroy_block_chunk();
    return bad == 0 ? 0 : 1;
}
