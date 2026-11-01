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
 * @brief export_chunk() / import_chunk() and export_array() / import_array()
 *        (chunk/chunk_exchange.h): an array rebuilt from its exported states is
 *        the same array.
 *
 * @details A source of eight chunks and a partial ninth, every state the
 *          variant has: one chunk never written and one written with the first
 *          value, three uniform (at a, a and b), two holding the same motif at
 *          offsets 100 and 200, the others varied. What they compress to
 *          depends on the variant and the type (expected_kind()): zero keeps
 *          only the first value, zero_one_value its declared second value (a;
 *          bool: true), zero_any any value, pattern any value and the motif
 *          (installed directly: the registry's admission rule would take two
 *          passes). After shrink_to_fit() every chunk is exported, and the kinds
 *          must be the expected ones.
 *
 *            1. chunk by chunk into a fresh array: the same elements, the same
 *               states (a buffer or a motif element for element), the same
 *               number of chunks holding a buffer; import_chunk() refuses a
 *               written chunk; the imported array is then written as any other
 *               (a write into an imported uniform chunk materialises it from its
 *               value, the value written back and shrink_to_fit() put it back).
 *            2. import_array() into an array already written with other values,
 *               and into the source itself: the same again.
 *
 *          Every array is destroyed at the end (a second value or motif shared
 *          by several counted right, or a later destruction fails).
 *
 *          The variant's Array<double> and Array<bool>; CHUNK_EXCHANGE_PATTERN
 *          (its own executable, on the zero_any library): pattern::Array<int>.
 */

#include <cstddef>     // std::size_t
#include <type_traits> // std::conditional_t, std::is_same_v
#include <utility>     // std::as_const
#include <vector>      // std::vector (the exported states)

#include "chunk/chunk_traits.h"         // forward decls + default IntType = int
#include "chunk/zero_array.h"           // zero::Array
#include "chunk/zero_one_value_array.h" // zero_one_value::Array
#include "chunk/zero_any_array.h"       // zero_any::Array
#if defined(CHUNK_EXCHANGE_PATTERN)
    #include "chunk/pattern_array.h" // pattern::Array
#endif
#include "chunk/chunk_exchange.h" // export_chunk, import_chunk, export_array, import_array
#include "chunk/param_chunk.h"    // ParamChunk::init_indexes
#include "chunk/manage_ompt.h"    // parallel_phase
#include "chunk/config.h"         // chunk_config::use_zero*
#include "chunk/PMP_CHUNK.h"      // pmp::current_team
#include "chunk/type_name.h"      // type_name<T>()
#include "chunk/internal/print.h" // shrink_array::println

#include "validation_harness.h" // run() is called by validation_driver.cpp

namespace validation::chunk_exchange_validation
{

using namespace shrink_array;

namespace
{

#if defined(CHUNK_EXCHANGE_PATTERN)
constexpr bool pattern_build = true;
#else
constexpr bool pattern_build = false;
template <typename T>
using VariantArray =
    std::conditional_t<chunk_config::use_zero_any,
                       zero_any::Array<T>,
                       std::conditional_t<chunk_config::use_zero_one_value, zero_one_value::Array<T>, zero::Array<T>>>;
#endif

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

template <typename T>
constexpr bool is_bool = std::is_same_v<T, bool>;

/// @brief The uniform values: a (chunks 1 and 5, zero_one_value's second
///        value), b (chunk 2), and the one a write pokes in.
template <typename T>
T value_a()
{
    return is_bool<T> ? static_cast<T>(true) : static_cast<T>(5);
}
template <typename T>
T value_b()
{
    return is_bool<T> ? static_cast<T>(true) : static_cast<T>(7);
}
template <typename T>
T value_poke()
{
    return is_bool<T> ? static_cast<T>(false) : static_cast<T>(1);
}

/// @brief The source's element @p i (@p chunk_size elements per chunk).
template <typename T>
T source_value(long i, long chunk_size)
{
    const long k = i % chunk_size;
    if constexpr (is_bool<T>)
    {
        switch (i / chunk_size)
        {
            case 0:
            case 4:
                return false;
            case 1:
            case 2:
            case 5:
                return true;
            case 6:
                return k < chunk_size / 2 || k % 2 == 0;
            default:
                return i % 3 == 0;
        }
    }
    else
    {
        const auto motif = static_cast<T>((k * 3) + (k % 7));
        switch (i / chunk_size)
        {
            case 0:
            case 4:
                return T{}; // never written (0), written with the first value (4)
            case 1:
            case 5:
                return value_a<T>();
            case 2:
                return value_b<T>();
            case 3:
                return static_cast<T>(motif + 100);
            case 7:
                return static_cast<T>(motif + 200);
            case 6:
                return k < chunk_size / 2 ? static_cast<T>(3) : static_cast<T>(i);
            default:
                return static_cast<T>(i + 1); // the partial chunk: varied
        }
    }
}

/// @brief The kind chunk @p chunk must export as, in this variant, for T.
template <typename T>
ChunkKind expected_kind(int chunk)
{
    const bool second_a = chunk_config::use_zero_any || chunk_config::use_zero_one_value;
    const bool second_b = chunk_config::use_zero_any || (chunk_config::use_zero_one_value && is_bool<T>);
    switch (chunk)
    {
        case 0:
        case 4:
            return ChunkKind::first_value;
        case 1:
        case 5:
            return second_a ? ChunkKind::second_value : ChunkKind::materialised;
        case 2:
            return second_b ? ChunkKind::second_value : ChunkKind::materialised;
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

/// @brief zero_one_value (not bool): declare the second value a on @p arr.
template <typename A>
void prepare(A& arr)
{
    using T = A::value_type;
    if constexpr (chunk_config::use_zero_one_value && !pattern_build && !is_bool<T>)
    {
        arr.set_second_value(value_a<T>());
    }
}

/// @brief The differences between two exports (kinds, values, elements).
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
        if (a.kind != b.kind || (a.kind != ChunkKind::materialised && a.value != b.value))
        {
            println("  FAIL chunk {}: {} {} vs {} {}", c, kind_name(a.kind), a.value, kind_name(b.kind), b.value);
            ++bad;
            continue;
        }
        if ((a.kind == ChunkKind::materialised || a.kind == ChunkKind::motif) && a.data != nullptr)
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
template <typename A>
long wrong_elements(const A& arr, long chunk_size)
{
    using T  = A::value_type;
    long bad = 0;
    for (long i = 0; i < static_cast<long>(arr.size()); ++i)
    {
        bad += std::as_const(arr)[static_cast<int>(i)] != source_value<T>(i, chunk_size) ? 1 : 0;
    }
    return bad;
}

/// @brief Is chunk @p chunk written through a[i] (pattern's motif chunks are installed)?
bool written_by_elements(long chunk)
{
    return chunk != 0 && !(pattern_build && (chunk == 3 || chunk == 7));
}

/// @brief @p arr against @p expected: the elements, the states, the buffer count.
template <typename A>
long check_copy(const A&                                               arr,
                const std::vector<ChunkState<typename A::value_type>>& expected,
                int                                                    written,
                long                                                   chunk_size,
                const char*                                            what)
{
    const long size   = static_cast<long>(arr.size());
    const long wrong  = wrong_elements(arr, chunk_size);
    const long differ = compare_states(expected, export_array(arr).chunks, chunk_size, size);
    const int  got    = arr.get_nb_chunk_written();
    const bool ok     = wrong == 0 && differ == 0 && got == written;
    println("[{}/{}] {}: {} wrong element(s), {} state difference(s), {} vs {} chunks with a buffer  {}",
            variant_name(), type_name<typename A::value_type>(), what, wrong, differ, got, written, ok ? "ok" : "FAIL");
    return wrong + differ + (got != written ? 1 : 0);
}

template <typename A>
long round_trip()
{
    using T    = A::value_type;
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
                elems[static_cast<std::size_t>(k)] = source_value<T>((c * chunk_size) + k, chunk_size);
            }
            (void)src.install_motif(c, elems.data());
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
                                                   src[i] = source_value<T>(i, chunk_size);
                                               }
                                           });
        });
    src.shrink_to_fit();

    const auto states  = export_array(src).chunks;
    const int  written = src.get_nb_chunk_written();
    for (int c = 0; c < static_cast<int>(states.size()); ++c)
    {
        if (states[static_cast<std::size_t>(c)].kind != expected_kind<T>(c))
        {
            println("  FAIL source chunk {} exports as {}, expected {}", c,
                    kind_name(states[static_cast<std::size_t>(c)].kind), kind_name(expected_kind<T>(c)));
            ++fails;
        }
    }

    // 1. Chunk by chunk into a fresh array.
    {
        A dst(static_cast<int>(size));
        prepare(dst);
        for (std::size_t c = 0; c < states.size(); ++c)
        {
            if (!import_chunk(dst, c, states[c]))
            {
                println("  FAIL import_chunk refused fresh chunk {}", c);
                ++fails;
            }
        }
        fails += check_copy(dst, states, written, chunk_size, "imported chunk by chunk");

        // A written chunk is refused (chunk 6: materialised in every variant).
        long bad = import_chunk(dst, 6, states[0]) ? 1 : 0;

        // Written as any other array: one element of the imported chunk 1,
        // then that value back, then a pass.
        const int at = static_cast<int>(chunk_size) + 3;
        dst[at]      = value_poke<T>();
        bad += std::as_const(dst)[at] != value_poke<T>() ? 1 : 0;
        bad += std::as_const(dst)[at + 1] != value_a<T>() ? 1 : 0;
        dst[at] = value_a<T>();
        dst.shrink_to_fit();
        bad += wrong_elements(dst, chunk_size);
        bad += export_chunk(dst, 1).kind != expected_kind<T>(1) ? 1 : 0;
        if constexpr (pattern_build)
        {
            // A write into an imported motif chunk: materialised from its motif.
            const int in_motif = static_cast<int>(3 * chunk_size) + 2;
            dst[in_motif]      = static_cast<T>(-1);
            bad += std::as_const(dst)[in_motif] != static_cast<T>(-1) ? 1 : 0;
            bad += std::as_const(dst)[in_motif + 1] != source_value<T>(in_motif + 1, chunk_size) ? 1 : 0;
            dst[in_motif] = source_value<T>(in_motif, chunk_size);
            bad += wrong_elements(dst, chunk_size);
        }
        println("[{}/{}] refused on a written chunk, then written after the import: {} problem(s)  {}", variant_name(),
                type_name<T>(), bad, bad == 0 ? "ok" : "FAIL");
        fails += bad;
    }

    // 2. import_array() over an array already written, and over the source itself.
    {
        A other(static_cast<int>(size) / 3);
        for (long i = 0; i < static_cast<long>(other.size()); ++i)
        {
            other[static_cast<int>(i)] = value_poke<T>();
        }
        const bool done = import_array(other, export_array(src));
        fails += done ? 0 : 1;
        fails += check_copy(other, states, written, chunk_size, "import_array over a written array");

        // states' buffers were src's: released by the import. Kinds and
        // values only; the elements are checked one by one.
        std::vector<ChunkState<T>> kinds = states;
        for (auto& state : kinds)
        {
            state.data = nullptr;
        }
        const bool self = import_array(src, export_array(src));
        fails += self ? 0 : 1;
        fails += check_copy(src, kinds, written, chunk_size, "import_array over itself");
    }
    return fails;
}

} // namespace

int run(int /*argc*/, char** /*argv*/)
{
    ParamChunk::init_indexes(true);
    long fails = 0;
#if defined(CHUNK_EXCHANGE_PATTERN)
    fails += round_trip<pattern::Array<int>>();
#else
    fails += round_trip<VariantArray<double>>();
    fails += round_trip<VariantArray<bool>>();
#endif
    println("{}", fails ? "chunk_exchange_validation: FAILED" : "chunk_exchange_validation: OK");
    return fails ? 1 : 0;
}

} // namespace validation::chunk_exchange_validation
