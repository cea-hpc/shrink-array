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
// A compressed chunk switched to another constant by a range op. Variant-
// agnostic: chunk / chunk2 / chunk2a follow the compile-time macro
// (CHUNK_VARIANT_V1 / _V2 / _V2A), like the other harnesses here.
// ============================================================================
//
// On Chunk2A, fill_range over a whole chunk, and copy_range over a whole chunk
// whose source span is uniform, set the chunk's constant by its state alone
// (Chunk2A::fill_constant_st): a compressed chunk switches its second-value
// slot, type and values pointer in place, with no buffer. Guards:
//
//   1. every element reads back after each switch: first constant -> second
//      constant -> another second constant -> first constant -> second
//      constant, by fill_range and by copy_range;
//   2. no chunk holds a buffer after any of those switches (Chunk2A);
//   3. a write through operator[] into a switched chunk materialises it from
//      ITS constant: one element changes, the others still read the constant
//      (the chunk's second-value slot was set, not only its pointer);
//   4. reclaim_second_values() afterwards frees the values no chunk points at
//      any more, keeps the others, and every element still reads back
//      (Chunk2A; the old constants of a switched chunk record no reference).
//
// The same switches from every thread at once, in a phase, by fill_range_mt
// and copy_range_mt (each thread on its own whole chunks): a compressed Chunk2A
// chunk covered whole by a uniform span switches in place there too, within
// the write session; a span that does not cover the chunk materialises it, and
// a uniform span over a materialised chunk leaves it to the end-of-phase pass.
//
// The other variants take their own paths (materialise, queue): properties 1
// and 3 are checked there.
//
// EXIT STATUS: 0 when every check holds, 1 otherwise. Any thread count.

#include <cstddef>     // std::size_t
#include <type_traits> // std::conditional_t, std::is_same_v
#include <utility>     // std::as_const
#include <vector>      // std::vector (the copy sources)

#include "chunk/chunk_traits.h"          // forward decls + default ind_type = int
#include "chunk/array_chunk.h"           // ArrayChunk
#include "chunk/array_chunk2.h"          // ArrayChunk2
#include "chunk/array_chunk2A.h"         // ArrayChunk2A
#include "chunk/array_chunk_range_ops.h" // fill_range, copy_range, fill_range_mt, copy_range_mt
#include "chunk/param_chunk.h"           // ParamChunk::init_indexes
#include "chunk/chunk_std.h"             // release_chunk_pools, destroy_block_chunk
#include "chunk/config.h"                // chunk_config::use_chunk*
#include "chunk/manage_ompt.h"           // begin_parallel_phase / finalize_parallel_phase
#include "chunk/internal/print.h"        // shrink_array::println

#include "validation_harness.h" // self_command; run() is called by validation_driver.cpp

namespace validation::constant_switch_validation
{

using namespace shrink_array;

template <typename T>
using Array = std::conditional_t<chunk_config::use_chunk2a,
                                 ArrayChunk2A<T>,
                                 std::conditional_t<chunk_config::use_chunk2, ArrayChunk2<T>, ArrayChunk<T>>>;

constexpr const char* variant_name()
{
    if constexpr (chunk_config::use_chunk2a)
    {
        return "chunk2a";
    }
    else if constexpr (chunk_config::use_chunk2)
    {
        return "chunk2";
    }
    else
    {
        return "chunk";
    }
}

constexpr int n_chunks = 64;

static int fails = 0;

/// @brief One region writing [0, n) through operator[], then the pass.
template <typename Fn>
static void phase(Array<double>& arr, long n, Fn&& body)
{
#pragma omp parallel
    {
        if constexpr (build_config::has_not_ompt)
        {
            begin_parallel_phase();
        }
#pragma omp for schedule(static)
        for (long i = 0; i < n; ++i)
        {
            body(i);
        }
    }
    if constexpr (build_config::has_not_ompt)
    {
        finalize_parallel_phase();
    }
    if constexpr (chunk_config::use_chunk2a && !build_config::use_openmp)
    {
        arr.make_block_compression(); // a serial chunk2a has no between-region pass
    }
}

/// @brief Chunks still holding a buffer.
static int materialised(const Array<double>& arr);

/// @brief phase() over chunks [0, n_chunks), returning the chunks materialised
///        INSIDE the region, once every thread's writes are done -- before the
///        end-of-phase pass compresses what the writes left.
template <typename Fn>
static int phase_count(Array<double>& arr, Fn&& body)
{
    int in_region = 0;
#pragma omp parallel
    {
        if constexpr (build_config::has_not_ompt)
        {
            begin_parallel_phase();
        }
#pragma omp for schedule(static)
        for (long c = 0; c < n_chunks; ++c)
        {
            body(c);
        }
#pragma omp single
        in_region = materialised(arr);
    }
    if constexpr (build_config::has_not_ompt)
    {
        finalize_parallel_phase();
    }
    if constexpr (chunk_config::use_chunk2a && !build_config::use_openmp)
    {
        arr.make_block_compression();
    }
    return in_region;
}

/// @brief Chunk2A in an OpenMP build: @p got chunks were materialised inside
///        the region, where a switch in place leaves none.
static void check_in_region(const char* step, int got)
{
    if constexpr (chunk_config::use_chunk2a && build_config::use_openmp)
    {
        if (got != 0)
        {
            println("[{}] FAIL  {}: {} chunk(s) materialised inside the region, expected 0", variant_name(), step, got);
            ++fails;
        }
    }
}

/// @brief Chunks still holding a buffer.
static int materialised(const Array<double>& arr)
{
    const long S = static_cast<long>(arr.chunk_size());
    int        n = 0;
    for (int c = 0; c < n_chunks; ++c)
    {
        n += arr.written(static_cast<int>(c * S));
    }
    return n;
}

/// @brief Every element reads @p expect(i), and (Chunk2A) @p mat_expected
///        chunks hold a buffer.
template <typename Expect>
static void check(const char* step, const Array<double>& arr, Expect&& expect, int mat_expected)
{
    const long N   = n_chunks * static_cast<long>(arr.chunk_size());
    long       bad = 0;
    for (long i = 0; i < N; ++i)
    {
        bad += (std::as_const(arr)[static_cast<int>(i)] != expect(i));
    }
    const int  mat    = materialised(arr);
    const bool mat_ok = !chunk_config::use_chunk2a || mat == mat_expected;
    if (bad == 0 && mat_ok)
    {
        println("[{}] ok    {:<34} {:3} materialised", variant_name(), step, mat);
        return;
    }
    ++fails;
    if (bad != 0)
    {
        println("[{}] FAIL  {}: {} element(s) do not read back", variant_name(), step, bad);
    }
    if (!mat_ok)
    {
        println("[{}] FAIL  {}: {} chunk(s) materialised, expected {}", variant_name(), step, mat, mat_expected);
    }
}

/// @brief Property 4, Chunk2A only: after reclaim_second_values() exactly the
///        n_chunks constants the chunks hold are in use, and every element
///        still reads @p expect(i). A template, so that the members Chunk2A
///        alone declares are named in a discarded branch only.
template <typename Arr, typename Expect>
static void check_reclaim(const Arr& a, Expect&& expect)
{
    if constexpr (std::is_same_v<Arr, ArrayChunk2A<double>>)
    {
        (void)Arr::reclaim_second_values();
        const long in_use = static_cast<long>(Arr::second_value_stats().in_use());
        if (in_use != n_chunks)
        {
            println("[{}] FAIL  reclaim: {} second values in use, expected {}", variant_name(), in_use, n_chunks);
            ++fails;
        }
        else
        {
            println("[{}] ok    reclaim: {} second values in use", variant_name(), in_use);
        }
        check("after the reclaim", a, expect, 0);
    }
    else
    {
        static_assert(!chunk_config::use_chunk2a, "check_reclaim: a Chunk2A build tests an ArrayChunk2A");
    }
}

int run(int /*argc*/, char** /*argv*/)
{
    ParamChunk::init_indexes(true);

    const long S  = static_cast<long>(Array<double>(1).chunk_size());
    const long N  = n_chunks * S;
    const auto sz = [](long x) { return static_cast<std::size_t>(x); };

    println("=== constant_switch_validation [variant = {}]  chunk_size={}  chunks={} ===", variant_name(), S, n_chunks);

    // Chunk c at base + c, one value per chunk, by fill_range.
    const auto fill_per_chunk = [&](Array<double>& a, double base)
    {
        for (long c = 0; c < n_chunks; ++c)
        {
            fill_range(a, sz(c * S), sz((c + 1) * S), base + static_cast<double>(c));
        }
    };
    const auto per_chunk = [S](double base) { return [S, base](long i) { return base + static_cast<double>(i / S); }; };
    const auto at        = [](double v) { return [v](long) { return v; }; };

    // ── By fill_range ──────────────────────────────────────────────────────
    {
        Array<double> a(static_cast<int>(N));
        fill_per_chunk(a, 10.0);
        check("fill: first -> 10 + c", a, per_chunk(10.0), 0);
        fill_per_chunk(a, 20.0);
        check("fill: 10 + c -> 20 + c (in place)", a, per_chunk(20.0), 0);
        fill_range(a, 0, sz(N), 0.0);
        check("fill: 20 + c -> first", a, at(0.0), 0);
        fill_per_chunk(a, 30.0);
        check("fill: first -> 30 + c", a, per_chunk(30.0), 0);
        fill_range(a, 0, sz(N), 30.0); // chunk 0 already at 30: unchanged
        check("fill: 30 + c -> 30 everywhere", a, at(30.0), 0);
        fill_per_chunk(a, 30.0);
        check("fill: 30 -> 30 + c", a, per_chunk(30.0), 0);

        // 3: one element per chunk written through the proxy.
        phase(a, n_chunks, [&](long c) { a[static_cast<int>(c * S + 5)] = 99.0; });
        check(
            "write one element per chunk", a,
            [S](long i) { return (i % S == 5) ? 99.0 : 30.0 + static_cast<double>(i / S); }, n_chunks);
        fill_per_chunk(a, 30.0);
        check("fill: back to 30 + c", a, per_chunk(30.0), 0);

        // 4: 10 + c and 20 + c (but 30 .. 39, shared with 30 + c) are no longer used.
        check_reclaim(a, per_chunk(30.0));
        a.clear();
    }

    // ── By copy_range ──────────────────────────────────────────────────────
    {
        std::vector<double> src(sz(N));
        const auto          source = [&](auto f)
        {
            for (long i = 0; i < N; ++i)
            {
                src[sz(i)] = f(i);
            }
        };

        Array<double> a(static_cast<int>(N));
        source(per_chunk(40.0));
        copy_range(a, 0, sz(N), src.begin());
        check("copy: first -> 40 + c", a, per_chunk(40.0), 0);
        source(per_chunk(50.0));
        copy_range(a, 0, sz(N), src.begin());
        check("copy: 40 + c -> 50 + c (in place)", a, per_chunk(50.0), 0);
        source(at(0.0));
        copy_range(a, 0, sz(N), src.begin());
        check("copy: 50 + c -> first", a, at(0.0), 0);
        source(per_chunk(60.0));
        copy_range(a, 0, sz(N), src.begin());
        check("copy: first -> 60 + c", a, per_chunk(60.0), 0);

        // A mixed span over a compressed chunk: copied, materialised.
        const auto mixed = [S](long i) { return (i % 2) ? 2.5 : 70.0 + static_cast<double>(i / S); };
        source(mixed);
        copy_range(a, 0, sz(N), src.begin());
        check("copy: 60 + c -> mixed", a, mixed, n_chunks);
        // A uniform span over a materialised chunk: released.
        source(per_chunk(80.0));
        copy_range(a, 0, sz(N), src.begin());
        check("copy: mixed -> 80 + c", a, per_chunk(80.0), 0);

        phase(a, n_chunks, [&](long c) { a[static_cast<int>(c * S + S - 1)] = -1.0; });
        check(
            "write one element per chunk", a,
            [S](long i) { return (i % S == S - 1) ? -1.0 : 80.0 + static_cast<double>(i / S); }, n_chunks);
        a.clear();
    }

    // ── From every thread: fill_range_mt, copy_range_mt ────────────────────
    {
        Array<double>       a(static_cast<int>(N));
        std::vector<double> src(sz(N));
        // Chunk c, whole, by the thread the loop gives it; returns the chunks
        // materialised inside the region.
        const auto fill_mt = [&](double base)
        {
            return phase_count(a, [&](long c)
                               { fill_range_mt(a, sz(c * S), sz((c + 1) * S), base + static_cast<double>(c)); });
        };

        check_in_region("fill_mt: first -> 110 + c", fill_mt(110.0));
        check("fill_mt: first -> 110 + c", a, per_chunk(110.0), 0);
        check_in_region("fill_mt: 110 + c -> 120 + c", fill_mt(120.0));
        check("fill_mt: 110 + c -> 120 + c (in place)", a, per_chunk(120.0), 0);
        for (long i = 0; i < N; ++i)
        {
            src[sz(i)] = 130.0 + static_cast<double>(i / S);
        }
        check_in_region(
            "copy_mt: 120 + c -> 130 + c",
            phase_count(a, [&](long c) { copy_range_mt(a, sz(c * S), sz((c + 1) * S), src.begin() + c * S); }));
        check("copy_mt: 120 + c -> 130 + c (in place)", a, per_chunk(130.0), 0);
        check_in_region("fill_mt: 130 + c -> first",
                        phase_count(a, [&](long c) { fill_range_mt(a, sz(c * S), sz((c + 1) * S), 0.0); }));
        check("fill_mt: 130 + c -> first (in place)", a, at(0.0), 0);

        // A span short of its chunk: materialised (the element left is 0).
        phase(a, n_chunks, [&](long c) { fill_range_mt(a, sz(c * S + 1), sz((c + 1) * S), 7.0); });
        check("fill_mt: first -> 0 then 7 (partial)", a, [S](long i) { return (i % S == 0) ? 0.0 : 7.0; }, n_chunks);
        // A uniform span over a materialised chunk: written, compressed by the
        // end-of-phase pass.
        (void)fill_mt(140.0);
        check("fill_mt: materialised -> 140 + c", a, per_chunk(140.0), 0);
        check_reclaim(a, per_chunk(140.0));

        // 3: a write into a switched chunk materialises it from its constant.
        check_in_region("fill_mt: 140 + c -> 150 + c", fill_mt(150.0));
        phase(a, n_chunks, [&](long c) { a[static_cast<int>(c * S + 3)] = -2.0; });
        check(
            "write one element per chunk", a,
            [S](long i) { return (i % S == 3) ? -2.0 : 150.0 + static_cast<double>(i / S); }, n_chunks);
        a.clear();
    }

    release_chunk_pools();
    destroy_block_chunk();

    println("\n[{}] {}", variant_name(),
            fails == 0 ? "=== CONSTANT SWITCH OK, TEARDOWN OK ===" : "=== CONSTANT SWITCH FAILURES ===");
    return fails ? 1 : 0;
}

} // namespace validation::constant_switch_validation
