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

// The two boolean containers of a zero_one_value build, side by side
// (docs/variants.md, section 2, "Booleans: two containers"):
//
//   - zero_one_value::Array<bool>: one byte per element in a materialised chunk;
//   - ArrayChunkBits<>: one bit per element, 8 per byte -- the layout of
//     std::vector<bool> -- whose write is an atomic update of the 64-bit word.
//
// Both get the same mask, written from every thread with schedule(static, 1):
// element i by thread i % threads, so the neighbouring bits of one 64-bit word
// are written by different threads at the same time -- a data race on a
// std::vector<bool>, safe here. Every value is then read back, the constant
// chunks are folded, and the program prints what each array keeps.
// Exit status 0 when every value reads back as written.

#include <cstdio>    // std::fprintf (the catch)
#include <exception> // std::exception
#include <utility>   // std::as_const

#include "chunk/zero_one_value_array.h" // shrink_array::zero_one_value::Array (Array<bool>)
#include "chunk/array_chunk_bits.h"     // shrink_array::ArrayChunkBits
#include "chunk/manage_ompt.h"          // parallel_phase
#include "chunk/param_chunk.h"          // ParamChunk::init_indexes, release_chunk_pools, destroy_block_chunk
#include "chunk/internal/print.h"       // shrink_array::println

namespace sa = shrink_array;
using shrink_array::println;

// The mask: a quarter of mixed values (these chunks keep their data), a quarter
// of true (these compress to the second value, `true`), and a half never written
// (false, compressed from the start).
static bool expected(int index, int count)
{
    if (index < count / 4)
    {
        return index % 3 == 0;
    }
    return index < count / 2;
}

// Elements that do not read back as written, counted in parallel.
template <typename Arr>
static int count_wrong(const Arr& arr, int count)
{
    int bad = 0;
#pragma omp parallel for reduction(+ : bad)
    for (int i = 0; i < count; ++i)
    {
        if (static_cast<bool>(arr[i]) != expected(i, count))
        {
            ++bad;
        }
    }
    return bad;
}

// What an array keeps: its materialised chunks, and the data they hold (the
// per-chunk metadata, the same for both, aside).
template <typename Arr>
static void report(const char* name, const Arr& arr, double bytes_per_element)
{
    const int    written = arr.get_nb_chunk_written();
    const double data_kb = written * static_cast<double>(arr.chunk_size()) * bytes_per_element / 1024.0;
    println("  {:<30} chunks of {:>5}: {:>6} of {:>6} hold data ({:5.1f} %), {:9.1f} KiB of data (metadata aside)",
            name, arr.chunk_size(), written, arr.chunk_count(), arr.stats(), data_kb);
}

int main()
try
{
    sa::ParamChunk::init_indexes(true); // once, before the first array

    int bad = 0;
    {
        const int count = 1 << 22; // 4 M booleans: 4 MiB as bytes, 512 KiB as a std::vector<bool>

        sa::zero_one_value::Array<bool> flags(count); // a byte per element, all false, nothing allocated
        sa::ArrayChunkBits<>            mask(count);  // a bit per element, all false, nothing allocated

        // Write both, neighbouring elements from different threads. No
        // set_second_value(): for booleans the second value is `true`.
        sa::parallel_phase(
            [&]
            {
#pragma omp for schedule(static, 1)
                for (int i = 0; i < count / 2; ++i)
                {
                    flags[i] = expected(i, count);
                    mask[i]  = expected(i, count);
                }
            });

        // Read back through const references: Array<bool> gives a bool,
        // ArrayChunkBits a Bit, which converts to bool.
        bad += count_wrong(std::as_const(flags), count);
        bad += count_wrong(std::as_const(mask), count);

        flags.shrink_to_fit(); // fold the chunks that have become constant (sequential point)
        mask.shrink_to_fit();

        println("n = {} booleans, {} read back wrong", count, bad);
        report("zero_one_value::Array<bool>", flags, 1.0);
        report("ArrayChunkBits<>", mask, 1.0 / 8.0);
        println("  {:<30} {:9.1f} KiB, and a race on neighbouring writes", "std::vector<bool>", count / 8.0 / 1024.0);
    } // the arrays release their chunks here

    sa::release_chunk_pools(); // give the reuse pools back
    sa::destroy_block_chunk();
    return bad == 0 ? 0 : 1;
}
catch (const std::exception& e)
{
    // A failed allocation or std::format: said, not std::terminate().
    std::fprintf(stderr, "bool_arrays: %s\n", e.what());
    return 2;
}
