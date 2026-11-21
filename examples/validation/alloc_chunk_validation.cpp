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
// alloc_chunk() harness. Every variant, default static size and dyn.
// ============================================================================
//
// alloc_chunk(i) materialises the chunk holding element i ahead of the loop
// that fills it. Its point: once a chunk has its buffer, the loop can go
// through operator()(i), the raw reference (a T& into the buffer, no write
// proxy, no materialisation test: the cost of a read), instead of operator[].
// That is only right if alloc_chunk() leaves the chunk
//
//   - materialised (written(i)), whatever constant it was compressed to: a
//     chunk left on its shared constant block would take the raw stores into
//     that block, read by every chunk of the type at that constant;
//   - holding what it held: T{} for a fresh chunk, the second value for a
//     chunk compressed to it, its elements for a chunk already materialised;
//   - with a buffer of the ARRAY's chunk size, not of the size the calling
//     thread used last (dyn).
//
// CASES (each for double, int and bool)
//
//   window   a fresh array, a window of chunks preallocated in a region (the
//            rank overload in the threaded builds), filled through
//            operator()(i) in a second region: the window's chunks and no
//            other are materialised, they read T{} before the fill, and the
//            whole array reads back as a std::vector does;
//   again    alloc_chunk() on a chunk already materialised changes nothing;
//   first    chunks written, zeroed and folded back (shrink_to_fit()), then
//            preallocated again: T{} everywhere, raw stores stay in their
//            chunk;
//   second   the same from the second value (zero_or_value: the declared
//            one, bool's `true`; zero_any: any constant): the preallocated
//            chunks read the second value, and the raw stores into one of
//            them leave the others, compressed or not, as they were;
//   geometry (dyn) an array at twice the size the thread used last: the
//            preallocated chunks hold chunk_size() elements each.
//
// What is NOT asserted: mixing raw stores and operator[] on one chunk. The raw
// reference bypasses the library (docs/api.md, section 3): a serial build
// counts the constants of a chunk from the proxy writes alone.
//
// EXIT STATUS: 0 when every check holds, 1 otherwise.

#include <cstddef>     // std::size_t
#include <type_traits> // std::conditional_t, std::is_same_v
#include <utility>     // std::as_const
#include <vector>      // std::vector

#include "chunk/internal/chunk_traits.h"      // forward decls + default IntType = int
#include "chunk/zero_array.h"                 // zero::Array
#include "chunk/zero_or_value_array.h"        // zero_or_value::Array
#include "chunk/zero_any_array.h"             // zero_any::Array
#include "chunk/runtime.h"                    // Runtime::init
#include "chunk/phases.h"                     // parallel_phase
#include "chunk/config.h"                     // chunk_config::use_zero*, build_config
#include "chunk/internal/chunk_size_policy.h" // const_buffer_size
#include "chunk/internal/pmp_chunk.h"         // pmp::current_team, get_thr_rank_fast
#include "chunk/print.h"                      // shrink_array::println

#include "validation_harness.h" // run() is called by validation_driver.cpp
#include "validation_variant.h" // Array, variant_name

namespace validation::alloc_chunk_validation
{

using namespace shrink_array;

namespace
{

template <typename T>
using Arr =
    std::conditional_t<chunk_config::use_zero_any,
                       zero_any::Array<T>,
                       std::conditional_t<chunk_config::use_zero_or_value, zero_or_value::Array<T>, zero::Array<T>>>;

template <typename T>
constexpr const char* type_name()
{
    if constexpr (std::is_same_v<T, bool>)
    {
        return "bool";
    }
    else if constexpr (std::is_same_v<T, int>)
    {
        return "int";
    }
    else
    {
        return "double";
    }
}

/// A value that is never T{} nor second_value<T>(): what the fills store.
/// bool has two values only: `true` two times out of three.
template <typename T>
T ramp(int index)
{
    if constexpr (std::is_same_v<T, bool>)
    {
        return (index % 3) != 0;
    }
    else
    {
        return static_cast<T>((index % 251) + 10);
    }
}

/// The constant of the `second` case.
template <typename T>
constexpr T second_value()
{
    if constexpr (std::is_same_v<T, bool>)
    {
        return true;
    }
    else
    {
        return static_cast<T>(7);
    }
}

/// zero_or_value compresses to the second value its array declares, before
/// the writes. bool's is `true` by construction; zero has none; zero_any
/// takes any.
template <typename T>
void declare_second_value(Arr<T>& arr, T value)
{
    if constexpr (!std::is_same_v<T, bool> && requires { arr.set_second_value(value); })
    {
        arr.set_second_value(value);
    }
}

/// alloc_chunk() of chunks [@p first, @p last) in a region, each thread its
/// share, from its own pool in the threaded builds.
template <typename T>
void prealloc_in_region(Arr<T>& arr, int first, int last)
{
    const int chunk_size = static_cast<int>(arr.chunk_size());
    parallel_phase(
        [&]
        {
            pmp::current_team().for_static(first, last,
                                           [&](int chunk)
                                           {
                                               if constexpr (build_config::threaded)
                                               {
                                                   arr.alloc_chunk(chunk * chunk_size, get_thr_rank_fast());
                                               }
                                               else
                                               {
                                                   arr.alloc_chunk(chunk * chunk_size);
                                               }
                                           });
        });
}

/// Chunks [@p first, @p last) of @p arr written through operator[] in a
/// region: @p value everywhere, or the ramp when @p use_ramp.
template <typename T>
void proxy_fill(Arr<T>& arr, int first, int last, T value, bool use_ramp)
{
    const int chunk_size = static_cast<int>(arr.chunk_size());
    parallel_phase(
        [&]
        {
            pmp::current_team().for_static(first, last,
                                           [&](int chunk)
                                           {
                                               for (int offset = 0; offset < chunk_size; ++offset)
                                               {
                                                   const int index = (chunk * chunk_size) + offset;
                                                   arr[index]      = use_ramp ? ramp<T>(index) : value;
                                               }
                                           });
        });
}

/// The elements of @p arr that do not read as @p ref, through both const reads.
template <typename T>
int mismatches(const Arr<T>& arr, const std::vector<T>& ref)
{
    int bad = 0;
    for (int index = 0; index < static_cast<int>(ref.size()); ++index)
    {
        const T want = ref[static_cast<std::size_t>(index)];
        if (static_cast<T>(arr[index]) != want || arr(index) != want)
        {
            ++bad;
        }
    }
    return bad;
}

/// The chunks of [@p first, @p last) materialised, and those outside it.
template <typename T>
void count_written(const Arr<T>& arr, int n_chunks, int first, int last, int& inside, int& outside)
{
    const int chunk_size = static_cast<int>(arr.chunk_size());
    inside               = 0;
    outside              = 0;
    for (int chunk = 0; chunk < n_chunks; ++chunk)
    {
        if (arr.written(chunk * chunk_size))
        {
            ++((chunk >= first && chunk < last) ? inside : outside);
        }
    }
}

int report(const char* type, const char* what, bool ok, int a = 0, int b = 0, int c = 0)
{
    println("[{}/alloc-chunk/{}] {:<9} {}  ({}, {}, {})", variant_name(), type, what, ok ? "ok" : "FAIL", a, b, c);
    return ok ? 0 : 1;
}

// ── window + again ──────────────────────────────────────────────────────────
template <typename T>
int case_window()
{
    constexpr int n_chunks = 16;
    constexpr int first    = 4;
    constexpr int last     = 8;
    Arr<T>        arr(0);
    const int     chunk_size = static_cast<int>(arr.chunk_size());
    const int     count      = n_chunks * chunk_size;
    arr.resize(count);
    std::vector<T> ref(static_cast<std::size_t>(count), T{});

    prealloc_in_region<T>(arr, first, last);
    int inside  = 0;
    int outside = 0;
    count_written<T>(arr, n_chunks, first, last, inside, outside);
    // A preallocated chunk reads the first constant, through the raw reference too.
    int not_zero = mismatches<T>(arr, ref);
    for (int index = first * chunk_size; index < last * chunk_size; ++index)
    {
        if (arr(index) != T{})
        {
            ++not_zero;
        }
    }
    int fails = report(type_name<T>(), "window", inside == last - first && outside == 0 && not_zero == 0, inside,
                       outside, not_zero);
    if (inside != last - first)
    {
        return fails; // a raw store into a chunk with no buffer of its own: not attempted
    }

    // The loop alloc_chunk() is for: raw stores, in a region, nothing to materialise.
    parallel_phase(
        [&]
        {
            pmp::current_team().for_static(first, last,
                                           [&](int chunk)
                                           {
                                               for (int offset = 0; offset < chunk_size; ++offset)
                                               {
                                                   const int index = (chunk * chunk_size) + offset;
                                                   arr(index)      = ramp<T>(index);
                                               }
                                           });
        });
    for (int index = first * chunk_size; index < last * chunk_size; ++index)
    {
        ref[static_cast<std::size_t>(index)] = ramp<T>(index);
    }
    count_written<T>(arr, n_chunks, first, last, inside, outside);
    int bad = mismatches<T>(arr, ref);
    fails +=
        report(type_name<T>(), "raw fill", inside == last - first && outside == 0 && bad == 0, inside, outside, bad);

    // On a chunk that has its buffer: nothing moves, in a region or outside.
    prealloc_in_region<T>(arr, first, last);
    arr.alloc_chunk(first * chunk_size);
    count_written<T>(arr, n_chunks, first, last, inside, outside);
    bad = mismatches<T>(arr, ref);
    fails += report(type_name<T>(), "again", inside == last - first && outside == 0 && bad == 0, inside, outside, bad);
    return fails;
}

// ── first / second: from a chunk compressed to @p constant ──────────────────
template <typename T>
int case_from_constant(const char* what, T constant)
{
    constexpr int n_chunks = 8;
    constexpr int first    = 2; // chunks 2..5 are folded to the constant,
    constexpr int last     = 6; // 3 and 4 preallocated, 3 filled raw
    Arr<T>        arr(0);
    const int     chunk_size = static_cast<int>(arr.chunk_size());
    const int     count      = n_chunks * chunk_size;
    arr.resize(count);
    declare_second_value<T>(arr, second_value<T>());
    std::vector<T> ref(static_cast<std::size_t>(count), T{});

    proxy_fill<T>(arr, first, last, T{}, true);
    proxy_fill<T>(arr, first, last, constant, false);
    arr.shrink_to_fit();
    for (int index = first * chunk_size; index < last * chunk_size; ++index)
    {
        ref[static_cast<std::size_t>(index)] = constant;
    }
    int inside  = 0;
    int outside = 0;
    count_written<T>(arr, n_chunks, first, last, inside, outside);
    int bad = mismatches<T>(arr, ref);
    if (inside != 0 || outside != 0 || bad != 0)
    {
        // Not this harness's subject, but the case below would prove nothing.
        return report(type_name<T>(), what, false, inside, outside, bad);
    }

    arr.alloc_chunk(3 * chunk_size);
    arr.alloc_chunk((4 * chunk_size) + (chunk_size / 2));
    count_written<T>(arr, n_chunks, 3, 5, inside, outside);
    bad         = mismatches<T>(arr, ref);
    int raw_bad = 0;
    if (inside == 2)
    {
        for (int index = 3 * chunk_size; index < 5 * chunk_size; ++index)
        {
            if (arr(index) != constant)
            {
                ++raw_bad;
            }
        }
    }
    int fails =
        report(type_name<T>(), what, inside == 2 && outside == 0 && bad == 0 && raw_bad == 0, inside, outside, bad);
    if (inside != 2)
    {
        return fails; // chunk 3 may still be on the shared block: no raw store into it
    }

    // Raw stores into chunk 3: chunks 2 and 5 (compressed, on the shared block
    // of the constant) and chunk 4 (preallocated) keep the constant.
    for (int offset = 0; offset < chunk_size; ++offset)
    {
        const int index                      = (3 * chunk_size) + offset;
        arr(index)                           = ramp<T>(index);
        ref[static_cast<std::size_t>(index)] = ramp<T>(index);
    }
    count_written<T>(arr, n_chunks, 3, 5, inside, outside);
    bad = mismatches<T>(arr, ref);
    // A fresh array of the type reads its first constant, and one at the same
    // constant reads it: the shared blocks took none of the stores.
    Arr<T> other(0);
    other.resize(2 * chunk_size);
    declare_second_value<T>(other, second_value<T>());
    proxy_fill<T>(other, 0, 1, constant, false);
    other.shrink_to_fit();
    int other_bad = 0;
    for (int index = 0; index < 2 * chunk_size; ++index)
    {
        if (std::as_const(other)(index) != (index < chunk_size ? constant : T{}))
        {
            ++other_bad;
        }
    }
    fails += report(type_name<T>(), "raw store", inside == 2 && outside == 0 && bad == 0 && other_bad == 0, inside, bad,
                    other_bad);
    return fails;
}

// ── geometry (dyn) ──────────────────────────────────────────────────────────
/// An array at twice the chunk size the thread used last: every preallocated
/// chunk must take chunk_size() raw stores without reaching its neighbours.
template <typename T>
int case_geometry()
{
    if constexpr (build_config::chunk_dynamic)
    {
        constexpr int n_chunks = 32;
        Arr<T>        small(0);
        const auto    small_size = static_cast<unsigned int>(small.chunk_size());
        if (2U * small_size > const_buffer_size)
        {
            println("[{}/alloc-chunk/{}] geometry  skipped: {} is the widest chunk size", variant_name(),
                    type_name<T>(), small_size);
            return 0;
        }
        Arr<T> wide(0);
        wide.set_chunk_size(2U * small_size);
        const int chunk_size = static_cast<int>(wide.chunk_size());
        const int count      = n_chunks * chunk_size;
        wide.resize(count);
        small.resize(4 * static_cast<int>(small_size));
        std::vector<T> ref(static_cast<std::size_t>(count), T{});

        // Half of the chunks from a region (its threads never served `wide`),
        // half from this thread, which `small` used last.
        prealloc_in_region<T>(wide, 0, n_chunks / 2);
        small[0] = ramp<T>(1);
        for (int chunk = n_chunks / 2; chunk < n_chunks; ++chunk)
        {
            wide.alloc_chunk(chunk * chunk_size);
        }
        int inside  = 0;
        int outside = 0;
        count_written<T>(wide, n_chunks, 0, n_chunks, inside, outside);
        if (inside != n_chunks)
        {
            return report(type_name<T>(), "geometry", false, inside, chunk_size, 0);
        }
        for (int index = 0; index < count; ++index)
        {
            wide(index)                          = ramp<T>(index);
            ref[static_cast<std::size_t>(index)] = ramp<T>(index);
        }
        const int bad = mismatches<T>(wide, ref);
        return report(type_name<T>(), "geometry", chunk_size == static_cast<int>(2U * small_size) && bad == 0, inside,
                      chunk_size, bad);
    }
    else
    {
        return 0;
    }
}

template <typename T>
int all_cases()
{
    int fails = case_window<T>();
    fails += case_from_constant<T>("first", T{});
    if constexpr (!chunk_config::use_zero)
    {
        fails += case_from_constant<T>("second", second_value<T>());
    }
    fails += case_geometry<T>();
    return fails;
}

} // namespace

int run(int /*argc*/, char** /*argv*/)
{
    Runtime::init(true);
    int fails = 0;
    fails += all_cases<double>();
    fails += all_cases<int>();
    fails += all_cases<bool>();
    println("{}", fails ? "alloc_chunk_validation: FAILED" : "alloc_chunk_validation: OK");
    return fails ? 1 : 0;
}

} // namespace validation::alloc_chunk_validation
