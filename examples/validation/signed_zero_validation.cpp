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
// Signed zero and NaN payload harness. Variant-agnostic: zero / zero_one_value /
// zero_any by the compile-time macro (CHUNK_VARIANT_ZERO*), like the others.
// ============================================================================
//
// Guards ONE property: an element reads back the exact bits written, for the
// values operator== cannot tell apart. -0.0 == +0.0, so a decision taken with
// == ("this value is already there", "this run is the constant") skipped the
// write of a -0.0 onto a +0.0 and compressed a run of -0.0 to the constant
// +0.0: 1/x went from -inf to +inf, std::signbit and std::atan2 changed. The
// library decides with same_value() (internal/same_value.h): the bits for
// float and double, part by part for std::complex, ValueIdentity<T> for an
// aggregate.
//
// THE CASES (each on double, float, std::complex<double>)
//
//   fresh-element     one -0.0 into a chunk still compressed to +0.0
//   fresh-chunk       a whole chunk of -0.0, written by the team, compressed
//   rewrite-chunk     a materialised chunk rewritten with -0.0, compressed
//   onto-plus-zero    one -0.0 onto a materialised +0.0
//   mixed-run         a chunk of alternating +0.0 / -0.0, compressed
//   fill-range        fill_range and fill_range_mt of -0.0, then compressed
//   second-value      -0.0 declared as the second value (zero_one_value), runs of
//                     it and of +0.0 side by side, compressed
//
//   Plus: a run of one NaN payload compresses and reads back bit for bit, and
//   an aggregate with a ValueIdentity specialisation keeps a -0.0 member.
//
//   Every case compares each element's bytes with a std::vector reference
//   (memcmp: the property is the bits, not ==).
//
// Compile OMP or non-OMP, static or dyn, check. Exit 0 = every bit round-trips.

#include <cmath>       // std::signbit
#include <complex>     // std::complex
#include <cstdint>     // std::uint64_t
#include <cstdlib>     // std::strtol
#include <cstring>     // std::memcmp
#include <limits>      // std::numeric_limits
#include <type_traits> // std::conditional_t
#include <utility>     // std::as_const
#include <vector>      // std::vector

#include "chunk/internal/chunk_traits.h" // forward decls + default IntType = int
#include "chunk/zero_array.h"            // zero::Array
#include "chunk/zero_one_value_array.h"  // zero_one_value::Array
#include "chunk/zero_any_array.h"        // zero_any::Array
#include "chunk/array_chunk_range_ops.h" // fill_range, fill_range_mt
#include "chunk/param_chunk.h"           // ParamChunk::init_indexes
#include "chunk/internal/zero_chunk.h"   // release_chunk_pools, destroy_block_chunk
#include "chunk/internal/same_value.h"   // ValueIdentity, same_value
#include "chunk/config.h"                // chunk_config::use_zero_one_value*, build_config::has_not_ompt
#include "chunk/internal/PMP_CHUNK.h"    // CHUNK_PMP_MAX_THREADS
#include "chunk/manage_ompt.h"           // begin_parallel_phase / finalize_parallel_phase
#include "chunk/print.h"                 // shrink_array::println

#include "validation_harness.h" // self_command; run() is called by validation_driver.cpp

namespace validation::signed_zero_validation
{
/// Three doubles: the aggregate case. The defaulted == says {-0.0} == {+0.0};
/// the ValueIdentity specialisation below makes the library tell them apart.
struct Vec3
{
    auto operator<=>(const Vec3&) const = default;

    double x, y, z;
};
} // namespace validation::signed_zero_validation

template <>
struct shrink_array::ValueIdentity<validation::signed_zero_validation::Vec3>
{
    static bool same(const validation::signed_zero_validation::Vec3& a,
                     const validation::signed_zero_validation::Vec3& b) noexcept
    {
        return same_value(a.x, b.x) && same_value(a.y, b.y) && same_value(a.z, b.z);
    }
};

namespace validation::signed_zero_validation
{

using namespace shrink_array;

static void begin_phase()
{
    if constexpr (build_config::has_not_ompt)
    {
        shrink_array::begin_parallel_phase();
    }
}
static void end_phase()
{
    if constexpr (build_config::has_not_ompt)
    {
        shrink_array::finalize_parallel_phase();
    }
}

template <typename T>
using Array =
    std::conditional_t<chunk_config::use_zero_any,
                       zero_any::Array<T>,
                       std::conditional_t<chunk_config::use_zero_one_value, zero_one_value::Array<T>, zero::Array<T>>>;

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

/// The values of a case for one element type: the negative zero, the
/// positive one, and a non-zero value to materialise chunks with.
template <typename T>
struct Values;
template <>
struct Values<double>
{
    static double neg()
    {
        return -0.0;
    }
    static double pos()
    {
        return 0.0;
    }
    static double other(long i)
    {
        return 1.0 + static_cast<double>(i % 97);
    }
};
template <>
struct Values<float>
{
    static float neg()
    {
        return -0.0f;
    }
    static float pos()
    {
        return 0.0f;
    }
    static float other(long i)
    {
        return 1.0f + static_cast<float>(i % 97);
    }
};
template <>
struct Values<std::complex<double>>
{
    static std::complex<double> neg()
    {
        return {-0.0, -0.0};
    }
    static std::complex<double> pos()
    {
        return {0.0, 0.0};
    }
    static std::complex<double> other(long i)
    {
        return {1.0 + static_cast<double>(i % 97), 0.5};
    }
};
template <>
struct Values<Vec3>
{
    static Vec3 neg()
    {
        return {-0.0, 0.0, -0.0};
    }
    static Vec3 pos()
    {
        return {0.0, 0.0, 0.0};
    }
    static Vec3 other(long i)
    {
        return {1.0 + static_cast<double>(i % 97), 2.0, 3.0};
    }
};

template <typename T>
static bool same_bits(const T& a, const T& b)
{
    return std::memcmp(&a, &b, sizeof(T)) == 0;
}

/// Compress from the whole team (the pass every harness uses).
template <typename Arr>
static void compress_all(Arr& arr)
{
#pragma omp parallel
    {
        begin_phase();
        arr.compress();
    }
    end_phase();
}

/// Every element's bits against the reference; prints the first mismatch.
template <typename T>
static int verify(const char* type_name, const char* label, const Array<T>& arr, const std::vector<T>& ref)
{
    long       bad   = 0;
    long       first = -1;
    const long n     = static_cast<long>(ref.size());
    for (long i = 0; i < n; ++i)
    {
        const T got = std::as_const(arr)[i];
        if (!same_bits(got, ref[static_cast<std::size_t>(i)]))
        {
            if (first < 0)
            {
                first = i;
            }
            ++bad;
        }
    }
    if (bad != 0)
    {
        const long csz = static_cast<long>(arr.chunk_size());
        println("[{}/{}/{}] FAIL: {}/{} elements lost their bits, first at i={} (chunk #{}, offset {})", variant_name(),
                type_name, label, bad, n, first, first / csz, first % csz);
        return 1;
    }
    println("[{}/{}/{}] OK", variant_name(), type_name, label);
    return 0;
}

template <typename T>
static int run_type(const char* type_name)
{
    using V   = Values<T>;
    int fails = 0;

    // Four chunks: enough for each case to own whole chunks of the geometry.
    Array<T>   probe(1);
    const long csz = static_cast<long>(probe.chunk_size());
    probe.clear();
    const long n = 4 * csz;

    // fresh-element: one -0.0 into a chunk still compressed to +0.0.
    {
        Array<T>       arr(n);
        std::vector<T> ref(static_cast<std::size_t>(n), V::pos());
        arr[csz + 3]                           = V::neg();
        ref[static_cast<std::size_t>(csz + 3)] = V::neg();
        fails += verify(type_name, "fresh-element", arr, ref);
        compress_all(arr);
        fails += verify(type_name, "fresh-element+compress", arr, ref);
        arr.clear();
    }

    // fresh-chunk: a whole chunk of -0.0 written by the team, then compressed.
    {
        Array<T>       arr(n);
        std::vector<T> ref(static_cast<std::size_t>(n), V::pos());
#pragma omp parallel
        {
            begin_phase();
#pragma omp for
            for (long i = csz; i < 2 * csz; ++i)
            {
                arr[i] = V::neg();
            }
        }
        end_phase();
        for (long i = csz; i < 2 * csz; ++i)
        {
            ref[static_cast<std::size_t>(i)] = V::neg();
        }
        compress_all(arr);
        fails += verify(type_name, "fresh-chunk", arr, ref);
        arr.clear();
    }

    // rewrite-chunk: materialised with non-zero values, rewritten with -0.0,
    // compressed.
    {
        Array<T>       arr(n);
        std::vector<T> ref(static_cast<std::size_t>(n));
        for (long i = 0; i < n; ++i)
        {
            arr[i]                           = V::other(i);
            ref[static_cast<std::size_t>(i)] = V::other(i);
        }
        for (long i = 2 * csz; i < 3 * csz; ++i)
        {
            arr[i]                           = V::neg();
            ref[static_cast<std::size_t>(i)] = V::neg();
        }
        compress_all(arr);
        fails += verify(type_name, "rewrite-chunk", arr, ref);
        arr.clear();
    }

    // onto-plus-zero: one -0.0 onto a +0.0 of a materialised chunk.
    {
        Array<T>       arr(n);
        std::vector<T> ref(static_cast<std::size_t>(n), V::pos());
        for (long i = 0; i < csz; ++i)
        {
            const T v                        = (i % 2 == 0) ? V::pos() : V::other(i);
            arr[i]                           = v;
            ref[static_cast<std::size_t>(i)] = v;
        }
        arr[4] = V::neg();
        ref[4] = V::neg();
        fails += verify(type_name, "onto-plus-zero", arr, ref);
        compress_all(arr);
        fails += verify(type_name, "onto-plus-zero+compress", arr, ref);
        arr.clear();
    }

    // mixed-run: alternating +0.0 / -0.0 over a whole chunk, compressed.
    {
        Array<T>       arr(n);
        std::vector<T> ref(static_cast<std::size_t>(n), V::pos());
        for (long i = csz; i < 2 * csz; ++i)
        {
            const T v                        = (i % 2 == 0) ? V::pos() : V::neg();
            arr[i]                           = v;
            ref[static_cast<std::size_t>(i)] = v;
        }
        compress_all(arr);
        fails += verify(type_name, "mixed-run", arr, ref);
        arr.clear();
    }

    // fill-range: fill_range over a chunk and a half (a whole chunk and a
    // partial one), fill_range_mt from the team over the next chunks.
    {
        Array<T>       arr(n);
        std::vector<T> ref(static_cast<std::size_t>(n), V::pos());
        arr[(2 * csz) + 1]                           = V::other(1); // a materialised chunk under the partial fill
        ref[static_cast<std::size_t>((2 * csz) + 1)] = V::other(1);
        fill_range(arr, 0, static_cast<std::size_t>(csz + (csz / 2)), V::neg());
        for (long i = 0; i < csz + (csz / 2); ++i)
        {
            ref[static_cast<std::size_t>(i)] = V::neg();
        }
#pragma omp parallel
        {
            begin_phase();
#pragma omp for schedule(static, 1)
            for (long c = 2; c < 4; ++c)
            {
                fill_range_mt(arr, static_cast<std::size_t>((c * csz) + 2), static_cast<std::size_t>((c + 1) * csz),
                              V::neg());
            }
        }
        end_phase();
        for (long i = (2 * csz) + 2; i < 3 * csz; ++i)
        {
            ref[static_cast<std::size_t>(i)] = V::neg();
        }
        for (long i = (3 * csz) + 2; i < 4 * csz; ++i)
        {
            ref[static_cast<std::size_t>(i)] = V::neg();
        }
        fails += verify(type_name, "fill-range", arr, ref);
        compress_all(arr);
        fails += verify(type_name, "fill-range+compress", arr, ref);
        arr.clear();
    }

    // second-value: -0.0 as the second constant (zero_one_value), next to runs of +0.0.
    if constexpr (requires(Array<T> a) { a.set_second_value(V::neg()); })
    {
        Array<T>       arr(n);
        std::vector<T> ref(static_cast<std::size_t>(n), V::pos());
        arr.set_second_value(V::neg());
        for (long i = 0; i < n; ++i)
        {
            const T v                        = ((i / csz) % 2 == 1) ? V::neg() : V::other(i);
            arr[i]                           = v;
            ref[static_cast<std::size_t>(i)] = v;
        }
        for (long i = 2 * csz; i < 3 * csz; ++i)
        {
            arr[i]                           = V::pos();
            ref[static_cast<std::size_t>(i)] = V::pos();
        }
        compress_all(arr);
        fails += verify(type_name, "second-value", arr, ref);
        arr[csz + 5]                                 = V::pos(); // a +0.0 into the -0.0 constant chunk
        ref[static_cast<std::size_t>(csz + 5)]       = V::pos();
        arr[(2 * csz) + 5]                           = V::neg(); // a -0.0 into the +0.0 one
        ref[static_cast<std::size_t>((2 * csz) + 5)] = V::neg();
        fails += verify(type_name, "second-value+writes", arr, ref);
        arr.clear();
    }

    return fails;
}

/// A run of one NaN payload compresses (same bits: the same value) and reads
/// back bit for bit; another payload next to it keeps its own.
static int run_nan()
{
    Array<double> probe(1);
    const long    csz = static_cast<long>(probe.chunk_size());
    probe.clear();
    const long n = 3 * csz;

    const double qnan  = std::numeric_limits<double>::quiet_NaN();
    const double other = -qnan; // same payload, sign bit set: a different value

    Array<double>       arr(n);
    std::vector<double> ref(static_cast<std::size_t>(n), 0.0);
    for (long i = csz; i < 2 * csz; ++i)
    {
        arr[i]                           = qnan;
        ref[static_cast<std::size_t>(i)] = qnan;
    }
    arr[(2 * csz) + 7]                           = other;
    ref[static_cast<std::size_t>((2 * csz) + 7)] = other;
    compress_all(arr);
    const int fails = verify("double", "nan-payload", arr, ref);
    arr.clear();
    return fails;
}

int run(int argc, char** argv)
{
    (void)argc;
    (void)argv;
    ParamChunk::init_indexes(true);

    println("=== signed_zero_validation [variant = {}]  threads={} ===", variant_name(), CHUNK_PMP_MAX_THREADS);

    // The predicate itself, before any array.
    int fails = 0;
    if (same_value(-0.0, 0.0) || same_value(-0.0f, 0.0f) || !same_value(-0.0, -0.0) ||
        same_value(std::complex<double>{0.0, -0.0}, std::complex<double>{0.0, 0.0}) ||
        !same_value(std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN()) ||
        same_value(-0.0L, 0.0L) || !same_value(1.5L, 1.5L) || same_value(Vec3{0.0, -0.0, 0.0}, Vec3{})) [[unlikely]]
    {
        println("[{}/same_value] FAIL: the predicate does not tell the bits apart", variant_name());
        ++fails;
    }

    fails += run_type<double>("double");
    fails += run_type<float>("float");
    fails += run_type<std::complex<double>>("complex<double>");
    fails += run_type<Vec3>("Vec3");
    fails += run_nan();

    release_chunk_pools();
    destroy_block_chunk();

    println("\n[{}] {}", variant_name(), fails == 0 ? "=== SIGNED ZERO OK ===" : "=== SIGNED ZERO FAILURES ===");
    return fails;
}

} // namespace validation::signed_zero_validation
