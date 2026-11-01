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
 * @details A source of eight chunks and a partial ninth, every state the
 *          variant has: one chunk never written and one written with the first
 *          value, three uniform at 5 (twice) and 7, two holding the same motif
 *          at offsets 100 and 200, the others varied. What they compress to
 *          depends on the variant (expected_kind()): zero keeps only the first
 *          value, zero_one_value its declared second value (5), zero_any any
 *          value, pattern any value and the motif (installed directly, as the
 *          registry's admission rule would take two passes). After
 *          shrink_to_fit() every chunk is exported, and the kinds must be the
 *          expected ones. A fresh array of the same size imports them, then:
 *          the same elements, the same states (a buffer or a motif element for
 *          element), the same number of chunks holding a buffer. The imported
 *          array is then written as any other: a write into an imported
 *          uniform chunk materialises it from its value, the value written back
 *          and shrink_to_fit() put it back in its state. Both arrays are
 *          destroyed at the end (a second value or motif shared by both counted
 *          right, or the second destruction fails).
 *
 *          CHUNK_EXCHANGE_PATTERN (its own executable, on the zero_any
 *          library): pattern::Array<int>. Otherwise the variant's Array<double>.
 */

#include <cstddef>     // std::size_t
#include <type_traits> // std::conditional_t
#include <utility>     // std::as_const
#include <vector>      // std::vector (the exported states)

#include "chunk/chunk_traits.h"         // forward decls + default IntType = int
#include "chunk/zero_array.h"           // zero::Array
#include "chunk/zero_one_value_array.h" // zero_one_value::Array
#include "chunk/zero_any_array.h"       // zero_any::Array
#if defined(CHUNK_EXCHANGE_PATTERN)
    #include "chunk/pattern_array.h" // pattern::Array
#endif
#include "chunk/chunk_exchange.h" // export_chunk, import_chunk, ChunkState
#include "chunk/param_chunk.h"    // ParamChunk::init_indexes
#include "chunk/manage_ompt.h"    // parallel_phase
#include "chunk/config.h"         // chunk_config::use_zero*
#include "chunk/PMP_CHUNK.h"      // pmp::current_team
#include "chunk/internal/print.h" // shrink_array::println

#include "validation_harness.h" // run() is called by validation_driver.cpp

namespace validation::chunk_exchange_validation
{

using namespace shrink_array;

namespace
{

#if defined(CHUNK_EXCHANGE_PATTERN)
using Arr                    = pattern::Array<int>;
constexpr bool pattern_build = true;
#else
using Arr = std::conditional_t<
    chunk_config::use_zero_any,
    zero_any::Array<double>,
    std::conditional_t<chunk_config::use_zero_one_value, zero_one_value::Array<double>, zero::Array<double>>>;
constexpr bool pattern_build = false;
#endif
using T = Arr::value_type;

constexpr const char* variant_name()
{
    if constexpr (pattern_build)
    {
        return "pattern";
    }
    else if constexpr (chunk_config::use_zero_any)
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

constexpr int n_full_chunks = 8; ///< then a partial chunk of half a chunk

/// @brief Element @p k of the motif of chunks 3 and 7.
T motif(long k)
{
    return static_cast<T>((k * 3) + (k % 7));
}

/// @brief The source's element @p i (@p chunk_size elements per chunk).
T source_value(long i, long chunk_size)
{
    const long k = i % chunk_size;
    switch (i / chunk_size)
    {
        case 0:
        case 4:
            return T{}; // never written (0), written with the first value (4)
        case 1:
        case 5:
            return static_cast<T>(5); // zero_one_value's declared second value
        case 2:
            return static_cast<T>(7);
        case 3:
            return static_cast<T>(motif(k) + 100);
        case 7:
            return static_cast<T>(motif(k) + 200);
        case 6:
            return k < chunk_size / 2 ? static_cast<T>(3) : static_cast<T>(i);
        default:
            return static_cast<T>(i + 1); // the partial chunk: varied
    }
}

/// @brief The kind chunk @p chunk must export as, in this variant.
ChunkKind expected_kind(int chunk)
{
    switch (chunk)
    {
        case 0:
        case 4:
            return ChunkKind::first_value;
        case 1:
        case 5:
            return chunk_config::use_zero_any || chunk_config::use_zero_one_value ? ChunkKind::second_value
                                                                                  : ChunkKind::materialised;
        case 2:
            return chunk_config::use_zero_any ? ChunkKind::second_value : ChunkKind::materialised;
        case 3:
        case 7:
            return pattern_build ? ChunkKind::motif : ChunkKind::materialised;
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
        case ChunkKind::motif:
            return "motif";
        case ChunkKind::materialised:
            return "materialised";
    }
    return "?";
}

/// @brief zero_one_value: declare the second value (5) on @p arr.
template <typename A>
void prepare(A& arr)
{
    if constexpr (chunk_config::use_zero_one_value && !pattern_build)
    {
        arr.set_second_value(static_cast<T>(5));
    }
}

/// @brief Every state of @p arr.
std::vector<ChunkState<T>> export_all(const Arr& arr)
{
    std::vector<ChunkState<T>> states;
    for (int c = 0; c < arr.chunk_count(); ++c)
    {
        states.push_back(export_chunk(arr, static_cast<std::size_t>(c)));
    }
    return states;
}

/// @brief The differences between two exports (kinds, values, elements).
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
        if (a.kind != b.kind || (a.kind != ChunkKind::materialised && a.value != b.value))
        {
            println("  FAIL chunk {}: {} {} vs {} {}", c, kind_name(a.kind), a.value, kind_name(b.kind), b.value);
            ++bad;
            continue;
        }
        if (a.kind == ChunkKind::materialised || a.kind == ChunkKind::motif)
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
long wrong_elements(const Arr& arr, long chunk_size)
{
    long bad = 0;
    for (long i = 0; i < static_cast<long>(arr.size()); ++i)
    {
        bad += std::as_const(arr)[static_cast<int>(i)] != source_value(i, chunk_size) ? 1 : 0;
    }
    return bad;
}

/// @brief Is chunk @p chunk written through a[i] (pattern's motif chunks are installed)?
bool written_by_elements(long chunk)
{
    return chunk != 0 && !(pattern_build && (chunk == 3 || chunk == 7));
}

template <typename A>
long round_trip()
{
    long fails = 0;
    A    src(0);
    prepare(src);
    const long chunk_size = static_cast<long>(src.chunk_size());
    const long size       = (n_full_chunks * chunk_size) + (chunk_size / 2);
    src.resize(static_cast<int>(size));
    if constexpr (pattern_build)
    {
        std::vector<T> elems(static_cast<std::size_t>(chunk_size));
        for (const int c : {3, 7})
        {
            for (long k = 0; k < chunk_size; ++k)
            {
                elems[static_cast<std::size_t>(k)] = source_value((c * chunk_size) + k, chunk_size);
            }
            (void)src.install_motif_for_test(c, elems.data());
        }
    }
    parallel_phase(
        [&]
        {
            pmp::current_team().for_static(0, static_cast<int>(size),
                                           [&](int i)
                                           {
                                               if (written_by_elements(i / chunk_size))
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
        A dst(static_cast<int>(size));
        prepare(dst);
        for (std::size_t c = 0; c < states.size(); ++c)
        {
            import_chunk(dst, c, states[c]);
        }
        const long wrong   = wrong_elements(dst, chunk_size);
        const long differ  = compare_states(states, export_all(dst), chunk_size, size);
        const int  written = dst.get_nb_chunk_written();
        println("[{}/exchange] imported: {} wrong element(s), {} state difference(s), {} vs {} chunks with a "
                "buffer  {}",
                variant_name(), wrong, differ, written, src.get_nb_chunk_written(),
                wrong == 0 && differ == 0 && written == src.get_nb_chunk_written() ? "ok" : "FAIL");
        fails += wrong + differ + (written != src.get_nb_chunk_written() ? 1 : 0);

        // Written as any other array: one element of the imported chunk 1 (at
        // 5), then that value back, then a pass.
        const int at = static_cast<int>(chunk_size) + 3;
        dst[at]      = static_cast<T>(1);
        long bad     = std::as_const(dst)[at] != static_cast<T>(1) ? 1 : 0;
        bad += std::as_const(dst)[at + 1] != static_cast<T>(5) ? 1 : 0;
        dst[at] = static_cast<T>(5);
        dst.shrink_to_fit();
        bad += wrong_elements(dst, chunk_size);
        bad += export_chunk(dst, 1).kind != expected_kind(1) ? 1 : 0;
        if constexpr (pattern_build)
        {
            // A write into an imported motif chunk: materialised from its motif.
            const int in_motif = static_cast<int>(3 * chunk_size) + 2;
            dst[in_motif]      = static_cast<T>(-1);
            bad += std::as_const(dst)[in_motif] != static_cast<T>(-1) ? 1 : 0;
            bad += std::as_const(dst)[in_motif + 1] != source_value(in_motif + 1, chunk_size) ? 1 : 0;
            dst[in_motif] = source_value(in_motif, chunk_size);
            bad += wrong_elements(dst, chunk_size);
        }
        println("[{}/exchange] written after the import: {} problem(s)  {}", variant_name(), bad,
                bad == 0 ? "ok" : "FAIL");
        fails += bad;
    } // dst destroyed first, src below: the shared second values and motif
    return fails;
}

} // namespace

int run(int /*argc*/, char** /*argv*/)
{
    ParamChunk::init_indexes(true);
    const long fails = round_trip<Arr>();
    println("{}", fails ? "chunk_exchange_validation: FAILED" : "chunk_exchange_validation: OK");
    return fails ? 1 : 0;
}

} // namespace validation::chunk_exchange_validation
