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
// Open element types harness. Variant-agnostic: zero / zero_one_value / zero_any by
// the compile-time macro (CHUNK_VARIANT_ZERO*), like the other harnesses.
// ============================================================================
//
// Guards ONE property: an array of a type the library never heard of -- a
// struct of three doubles, a struct with padding, an enum class, a std::array,
// a 4-byte pair -- behaves like an array of double: every element reads back
// what was written, runs of the first constant (T{}) and of a second value
// compress, a partial rewrite re-materialises, the library tears down.
//
// THE CONTRACT THIS PINS DOWN
//
//   An element type is anything std::vector holds that is also trivially
//   copyable and equality-comparable, with T{} as its first constant: the
//   chunk buffers are raw memory, copied, replaced by a shared constant and
//   compared block-wise, so a type carrying state (std::string, anything with
//   a destructor that matters) is out. The comparison is the type's own
//   operator== (member-wise for a defaulted one), never memcmp: padding bytes
//   are not part of the value. Arithmetic proxies (+=, |=, ++ ...) exist only
//   for the types that have the operators; a struct gets assignment, reads
//   and comparison, which is what this harness uses.
//
// WHAT IT EXERCISES
//
//   Per type: materialise everything with a varied value, overwrite with runs
//   of T{} / second / varied (whole chunks, so the pass sees pure constant
//   blocks), compress() from the whole team, rewrite a quarter, compare every
//   element against a std::vector reference. Phase brackets as in
//   compress_roundtrip.
//
//   No type is declared up front: ParamChunk::init_indexes(bool) alone, and
//   each array registers its element type at its construction -- the first
//   Point3 array after a parallel region has run, a Quad array after a Color
//   one was torn down, and (dyn builds) a type whose first array comes after
//   a second geometry was declared, so the registration replays it.
//
// WHAT IT DOES NOT COVER
//
//   Arithmetic on the element (a struct has none), re-chunking of these
//   arrays, geometry sweeps.
//
// Compile OMP or non-OMP, static or dyn, check. Exit 0 = every element of
// every type round-trips and the teardown returned.

#include <array>       // std::array
#include <cstdint>     // std::int16_t
#include <cstdlib>     // std::strtol
#include <type_traits> // std::conditional_t
#include <utility>     // std::as_const
#include <vector>      // std::vector

#include "chunk/chunk_traits.h"               // forward decls + default IntType = int
#include "chunk/zero_array.h"                 // zero::Array
#include "chunk/zero_one_value_array.h"       // zero_one_value::Array
#include "chunk/zero_any_array.h"             // zero_any::Array
#include "chunk/param_chunk.h"                // ParamChunk::init_indexes, register_chunk_size
#include "chunk/zero_chunk.h"                 // release_chunk_pools, destroy_block_chunk
#include "chunk/internal/chunk_size_policy.h" // active_chunk_size, set_active_chunk_size, set_thread_chunk_size
#include "chunk/config.h" // chunk_config::use_zero_one_value*, build_config::chunk_dynamic / has_not_ompt
#include "chunk/PMP_CHUNK.h"                  // CHUNK_PMP_MAX_THREADS
#include "chunk/manage_ompt.h"                // begin_parallel_phase / finalize_parallel_phase
#include "chunk/internal/print.h"             // shrink_array::println

#include "validation_harness.h" // self_command; run() is called by validation_driver.cpp

namespace validation::open_types_validation
{

using namespace shrink_array;

// ── The types ───────────────────────────────────────────────────────────────

/// Three doubles, no padding: the mesh-point case. The defaulted <=> is what
/// zero_one_value asks of a floating-point struct (it orders second values, see
/// value_less.h); zero and zero_any need == alone.
struct Point3
{
    auto operator<=>(const Point3&) const = default;

    double x, y, z;
};

/// A tag byte then a double: seven bytes of padding inside the value, which
/// a defaulted operator== ignores and a memcmp would not.
struct Padded
{
    auto operator<=>(const Padded&) const = default;

    char   tag;
    double v;
};

/// An enum class: T{} is `none`, comparisons are built in, no arithmetic.
enum class Color : int
{
    none = 0,
    red  = 1,
    blue = 2,
    cyan = 3
};

/// A fixed array: 16 bytes, aggregate, operator== from the standard library.
using Quad = std::array<float, 4>;

/// Two shorts: a 4-byte element, smaller than a double. No operator<: every
/// bit is value (no padding, no floating point), zero_one_value orders it by bytes.
struct Pair16
{
    bool operator==(const Pair16&) const = default;

    std::int16_t a, b;
};

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

template <typename Arr, typename Val>
void maybe_set_second_value(Arr& a, const Val& v)
{
    if constexpr (requires(Arr x) { x.set_second_value(v); })
    {
        a.set_second_value(v);
    }
}

// A multiple of every geometry the harness is built at: a run is a whole
// number of chunks.
constexpr long run_len = 4096;

template <typename T>
struct Scenario
{
    T second;
    T refill;
    T (*varied)(long i);
};

/**
 * @brief Materialise, pattern, compress, refill a quarter, verify.
 * @return 0 if every element reads back as expected, 1 otherwise.
 */
template <typename T>
static int round_trip(const char* label, long n, const Scenario<T>& sc)
{
    Array<T> arr(n);
    maybe_set_second_value(arr, sc.second);

    std::vector<T> ref(n);

#pragma omp parallel
    {
        begin_phase();
#pragma omp for
        for (long i = 0; i < n; ++i)
        {
            arr[i] = sc.varied(i);
        }
    }
    end_phase();
    for (long i = 0; i < n; ++i)
    {
        ref[i] = sc.varied(i);
    }

#pragma omp parallel
    {
        begin_phase();
#pragma omp for
        for (long i = 0; i < n; ++i)
        {
            const long run = (i / run_len) % 3;
            if (run == 0)
            {
                arr[i] = T{};
            }
            else if (run == 1)
            {
                arr[i] = sc.second;
            }
        }
    }
    end_phase();
    for (long i = 0; i < n; ++i)
    {
        const long run = (i / run_len) % 3;
        if (run == 0)
        {
            ref[i] = T{};
        }
        else if (run == 1)
        {
            ref[i] = sc.second;
        }
    }

#pragma omp parallel
    {
        begin_phase();
        arr.compress();
    }
    end_phase();

#pragma omp parallel
    {
        begin_phase();
#pragma omp for
        for (long i = 0; i < n / 4; ++i)
        {
            arr[i] = sc.refill;
        }
    }
    end_phase();
    for (long i = 0; i < n / 4; ++i)
    {
        ref[i] = sc.refill;
    }

    long bad   = 0;
    long first = -1;
    for (long i = 0; i < n; ++i)
    {
        const T got = arr[i];
        if (!(got == ref[i]))
        {
            if (first < 0)
            {
                first = i;
            }
            ++bad;
        }
    }

    const long csz = static_cast<long>(arr.chunk_size());
    if (bad)
    {
        println("[{}/{}] FAIL: {}/{} mismatches, first at i={} (chunk #{}, offset {}, run {})", variant_name(), label,
                bad, n, first, first / csz, first % csz, (first / run_len) % 3);
    }
    else
    {
        println("[{}/{}] OK: {} elements of {} bytes round-trip at chunk_size={}", variant_name(), label, n, sizeof(T),
                csz);
    }

    arr.clear();
    return bad ? 1 : 0;
}

int run(int argc, char** argv)
{
    const long n = argc > 1 ? std::strtol(argv[1], nullptr, 10) : (1L << 19);

    // No type list: each array registers its own element type at its
    // construction (AllTypeChunk::ensure_manager). The process-wide setup alone.
    ParamChunk::init_indexes(true);

    println("=== open_types_validation [variant = {}]  N={}  threads={} ===", variant_name(), n, CHUNK_PMP_MAX_THREADS);

    int fails = 0;

    // A parallel region before any array exists: the runtime's team, and the
    // library's per-rank state that does not depend on a type, are set up
    // before the first registration.
    long warm = 0;
#pragma omp parallel reduction(+ : warm)
    {
        begin_phase();
        warm += 1;
    }
    end_phase();
    println("team of {} before the first array", warm);

    const Scenario<Point3> sp{{1.0, 2.0, 3.0},
                              {-1.0, 0.5, 9.0},
                              [](long i) { return Point3{static_cast<double>(i), static_cast<double>(i % 7), 0.25}; }};
    const Scenario<Padded> sd{{'s', 2.5},
                              {'r', -4.0},
                              [](long i)
                              { return Padded{static_cast<char>('a' + (i % 26)), static_cast<double>(i) * 0.5}; }};
    const Scenario<Color>  sc{Color::red, Color::cyan, [](long i) { return i % 3 == 0 ? Color::blue : Color::cyan; }};
    const Scenario<Quad>   sq{{1.f, 2.f, 3.f, 4.f},
                              {9.f, 8.f, 7.f, 6.f},
                              [](long i) { return Quad{static_cast<float>(i), 0.5f, static_cast<float>(i % 5), 1.f}; }};
    const Scenario<Pair16> s16{
        {7, 7},
        {-3, 3},
        [](long i) { return Pair16{static_cast<std::int16_t>(i % 1000), static_cast<std::int16_t>(i % 13)}; }};

    fails += round_trip<Point3>("Point3", n, sp);
    fails += round_trip<Padded>("Padded", n, sd);
    fails += round_trip<Color>("Color", n, sc);

    // dyn builds: a second geometry declared now, before Quad and Pair16 have
    // ever been seen -- their registration must replay it (register_manager),
    // or an array of theirs at that size would draw from an unprimed pool.
    const unsigned int base_size = active_chunk_size();
    if constexpr (build_config::chunk_dynamic)
    {
        ParamChunk::register_chunk_size(2 * base_size);
    }

    fails += round_trip<Quad>("Quad", n, sq);
    fails += round_trip<Pair16>("Pair16", n, s16);

    // The vector-like surface on a struct element: emplace_back from the
    // members, reserve/capacity, the (first, last) and {...} constructors, a
    // deep copy. Sequential, as those are.
    {
        int        bad   = 0;
        const auto check = [&](bool ok, const char* what)
        {
            if (!ok)
            {
                ++bad;
                println("[{}/vector-like on structs] FAIL: {}", variant_name(), what);
            }
        };
        Array<Point3> pts;
        pts.reserve(10000);
        check(pts.empty() && pts.capacity() >= 10000, "reserve(10000) on a fresh (empty) array");
        for (int i = 0; i < 10000; ++i)
        {
            pts.emplace_back(static_cast<double>(i), 0.5, static_cast<double>(i % 3));
        }
        bool ok = pts.size() == 10000;
        for (int i = 0; ok && i < 10000; ++i)
        {
            ok = std::as_const(pts)[i] == Point3{static_cast<double>(i), 0.5, static_cast<double>(i % 3)};
        }
        check(ok, "emplace_back from the members, read back");
        Array<Point3> cp(pts);
        cp[3] = Point3{-1., -1., -1.};
        check(std::as_const(pts)[3] == Point3{3., 0.5, 0.} && std::as_const(cp)[3] == Point3{-1., -1., -1.},
              "deep copy: a write in the copy leaves the source alone");
        Array<Pair16> cells{Pair16{1, 5}, Pair16{2, 6}, Pair16{}};
        check(cells.size() == 3 && std::as_const(cells)[1] == Pair16{2, 6} && std::as_const(cells)[2] == Pair16{},
              "initializer-list constructor");
        std::vector<Pair16> src(500, Pair16{7, 1});
        Array<Pair16>       rng(src.begin(), src.end());
        check(rng.size() == 500 && std::as_const(rng)[499] == Pair16{7, 1}, "(first, last) constructor");
        if (!bad)
        {
            println("[{}/vector-like on structs] OK: reserve, emplace_back, copy, {{...}}, (first, last)",
                    variant_name());
        }
        fails += bad ? 1 : 0;
        pts.clear();
        cp.clear();
        cells.clear();
        rng.clear();
    }

    if constexpr (build_config::chunk_dynamic)
    {
        // And an array of a late type AT that late geometry.
        set_active_chunk_size(2 * base_size);
        set_thread_chunk_size(2 * base_size);
        fails += round_trip<Pair16>("Pair16@2xsize", n, s16);
        set_active_chunk_size(base_size);
        set_thread_chunk_size(base_size);
    }

    release_chunk_pools();
    destroy_block_chunk();

    println("\n[{}] {}", variant_name(),
            fails == 0 ? "=== OPEN TYPES OK, TEARDOWN OK ===" : "=== OPEN TYPES FAILURES ===");
    return fails;
}

} // namespace validation::open_types_validation
