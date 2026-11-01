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
 * @file chunk_exchange_validation.cpp
 * @brief export_chunk() / import_chunk() (chunk/chunk_exchange.h): an array
 *        rebuilt from its exported chunk states is the same array.
 *
 * @details A zero_any source of eight chunks and a partial ninth: one never
 *          written, one written with the first value, three uniform at second
 *          values (two of them the same one), the others materialised. After
 *          shrink_to_fit() every chunk is exported; the kinds must be the
 *          expected ones. A fresh array of the same size imports them, then:
 *          the same elements, the same states (a materialised chunk's buffer
 *          element for element), the same number of chunks holding a buffer.
 *          The imported array is then written as any other: a write into an
 *          imported second-value chunk materialises it from that value, writing
 *          the value back and shrink_to_fit() compress it again. Both arrays
 *          are destroyed at the end (the shared second-value slots counted
 *          right, or the second destruction fails).
 *
 *          The other variants: not wired yet (the prototype covers zero_any).
 */

#include <cstddef> // std::size_t
#include <utility> // std::as_const
#include <vector>  // std::vector (the exported states)

#include "chunk/chunk_traits.h"         // forward decls + default IntType = int
#include "chunk/zero_array.h"           // zero::Array
#include "chunk/zero_one_value_array.h" // zero_one_value::Array
#include "chunk/zero_any_array.h"       // zero_any::Array
#include "chunk/chunk_exchange.h"       // export_chunk, import_chunk, ChunkState
#include "chunk/param_chunk.h"          // ParamChunk::init_indexes
#include "chunk/manage_ompt.h"          // parallel_phase
#include "chunk/config.h"               // chunk_config::use_zero_any
#include "chunk/PMP_CHUNK.h"            // pmp::current_team
#include "chunk/internal/print.h"       // shrink_array::println

#include "validation_harness.h" // run() is called by validation_driver.cpp

namespace validation::chunk_exchange_validation
{

using namespace shrink_array;

namespace
{

constexpr int n_full_chunks = 8; ///< then a partial chunk of half a chunk

/// @brief The source's element @p i (@p chunk_size elements per chunk).
double source_value(long i, long chunk_size)
{
    switch (i / chunk_size)
    {
        case 0:
        case 4:
            return 0.0; // never written (0), written with the first value (4)
        case 1:
        case 5:
            return 5.0; // one second value, two chunks
        case 2:
            return 7.0; // another
        case 6:
            return (i % chunk_size) < chunk_size / 2 ? 3.0 : static_cast<double>(i);
        default:
            return static_cast<double>(i) + 0.5; // 3, 7 and the partial one: varied
    }
}

/// @brief The kind chunk @p chunk must export as.
ChunkKind expected_kind(int chunk)
{
    switch (chunk)
    {
        case 0:
        case 4:
            return ChunkKind::first_value;
        case 1:
        case 2:
        case 5:
            return ChunkKind::second_value;
        default:
            return ChunkKind::materialised;
    }
}

const char* kind_name(ChunkKind kind)
{
    switch (kind)
    {
        case ChunkKind::first_value:
            return "first_value";
        case ChunkKind::second_value:
            return "second_value";
        case ChunkKind::materialised:
            return "materialised";
    }
    return "?";
}

/// @brief Every state of @p arr.
template <typename Arr>
std::vector<ChunkState<double>> export_all(const Arr& arr)
{
    std::vector<ChunkState<double>> states;
    for (int c = 0; c < arr.chunk_count(); ++c)
    {
        states.push_back(export_chunk(arr, static_cast<std::size_t>(c)));
    }
    return states;
}

/// @brief The differences between two exports (kinds, values, buffers).
template <typename T>
long compare_states(const std::vector<ChunkState<T>>& lhs,
                    const std::vector<ChunkState<T>>& rhs,
                    long                              chunk_size,
                    long                              size)
{
    long bad = lhs.size() != rhs.size() ? 1 : 0;
    for (std::size_t c = 0; c < lhs.size() && c < rhs.size(); ++c)
    {
        const auto& a = lhs[c];
        const auto& b = rhs[c];
        if (a.kind != b.kind || (a.kind == ChunkKind::second_value && a.value != b.value))
        {
            println("  FAIL chunk {}: {} {} vs {} {}", c, kind_name(a.kind), a.value, kind_name(b.kind), b.value);
            ++bad;
            continue;
        }
        if (a.kind == ChunkKind::materialised)
        {
            const long begin = static_cast<long>(c) * chunk_size;
            const long n     = begin + chunk_size < size ? chunk_size : size - begin;
            for (long k = 0; k < n; ++k)
            {
                if (a.data[k] != b.data[k])
                {
                    println("  FAIL chunk {} element {}: {} vs {}", c, k, a.data[k], b.data[k]);
                    ++bad;
                    break;
                }
            }
        }
    }
    return bad;
}

/// @brief The elements of @p arr that differ from source_value().
template <typename Arr>
long wrong_elements(const Arr& arr, long chunk_size)
{
    long bad = 0;
    for (long i = 0; i < static_cast<long>(arr.size()); ++i)
    {
        bad += std::as_const(arr)[static_cast<int>(i)] != source_value(i, chunk_size) ? 1 : 0;
    }
    return bad;
}

template <typename Arr>
long round_trip()
{
    long       fails = 0;
    Arr        src(0);
    const long chunk_size = static_cast<long>(src.chunk_size());
    const long size       = (n_full_chunks * chunk_size) + (chunk_size / 2);
    src.resize(static_cast<int>(size));
    parallel_phase(
        [&]
        {
            pmp::current_team().for_static(0, static_cast<int>(size),
                                           [&](int i)
                                           {
                                               if (i / chunk_size != 0) // chunk 0 is never written
                                               {
                                                   src[i] = source_value(i, chunk_size);
                                               }
                                           });
        });
    src.shrink_to_fit();

    const auto states = export_all(src);
    for (int c = 0; c < static_cast<int>(states.size()); ++c)
    {
        if (states[static_cast<std::size_t>(c)].kind != expected_kind(c))
        {
            println("  FAIL source chunk {} exports as {}, expected {}", c,
                    kind_name(states[static_cast<std::size_t>(c)].kind), kind_name(expected_kind(c)));
            ++fails;
        }
    }

    {
        Arr dst(static_cast<int>(size));
        for (std::size_t c = 0; c < states.size(); ++c)
        {
            import_chunk(dst, c, states[c]);
        }
        const long wrong   = wrong_elements(dst, chunk_size);
        const long differ  = compare_states(states, export_all(dst), chunk_size, size);
        const int  written = dst.get_nb_chunk_written();
        println("[zero_any/exchange] imported: {} wrong element(s), {} state difference(s), {} vs {} chunks with a "
                "buffer  {}",
                wrong, differ, written, src.get_nb_chunk_written(),
                wrong == 0 && differ == 0 && written == src.get_nb_chunk_written() ? "ok" : "FAIL");
        fails += wrong + differ + (written != src.get_nb_chunk_written() ? 1 : 0);

        // Written as any other array: one element of an imported second-value
        // chunk, then that value back, then a pass.
        const int at = static_cast<int>(chunk_size) + 3; // chunk 1, at 5.0
        dst[at]      = 1.0;
        long bad     = std::as_const(dst)[at] != 1.0 ? 1 : 0;
        bad += std::as_const(dst)[at + 1] != 5.0 ? 1 : 0;
        dst[at] = 5.0;
        dst.shrink_to_fit();
        bad += wrong_elements(dst, chunk_size);
        bad += export_chunk(dst, 1).kind != ChunkKind::second_value ? 1 : 0;
        println("[zero_any/exchange] written after the import: {} problem(s)  {}", bad, bad == 0 ? "ok" : "FAIL");
        fails += bad;
    } // dst destroyed first, src below: the shared second-value slots
    return fails;
}

} // namespace

int run(int /*argc*/, char** /*argv*/)
{
    ParamChunk::init_indexes(true);
    long fails = 0;
    if constexpr (chunk_config::use_zero_any)
    {
        fails += round_trip<zero_any::Array<double>>();
    }
    else
    {
        println("[chunk_exchange] not wired for this variant yet (the prototype covers zero_any)");
    }
    println("{}", fails ? "chunk_exchange_validation: FAILED" : "chunk_exchange_validation: OK");
    return fails ? 1 : 0;
}

} // namespace validation::chunk_exchange_validation
