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
// fill_range (serial) over whole chunks. Variant-agnostic, like the other
// harnesses here; the compression checks are zero_any::Chunk's.
// ============================================================================
//
// fill_range compresses a fully covered zero_any::Chunk chunk at once, whatever the
// value (zero_any::Chunk::compress, as the rechunk fold): no chunk of the range is
// left materialised, the buffer of one is back in the pool before the next is
// armed. v1 does the same for a fill of first_value, zero_one_value::Chunk for a fill of
// either of its constants. Guards, on every variant: the values read back. On
// zero_any::Chunk: no covered chunk stays materialised, from each starting state; on
// v1 and zero_one_value::Chunk, after a fill of one of their constants.
//
//   fresh     all chunks at first_value, filled with a plain value;
//   second    all compressed at V (a second value), filled with W;
//   varied    all materialised with varied values, filled with W;
//   first     all compressed at V, filled with first_value;
//   partial   a range starting and ending inside chunks: the edge chunks
//             stay materialised (not covered), the inner ones do not;
//   distinct  one value per chunk (each chunk its own second value), then
//             first_value everywhere: the second-value registry takes and gives
//             back every value (the teardown checks the registry is empty).
//   second-value  an array whose second value is 1.0: materialised chunks
//             filled with 1.0, then chunks at first_value filled with 1.0 --
//             compressed at once on zero_one_value::Chunk and zero_any::Chunk, not on v1 (1.0 is no
//             constant of v1);
//   constant  a fresh array filled with first_value, whole and in part, then
//             fill_range_mt from every thread: nothing to write, no chunk
//             materialised, on every variant.
//
// Exit 0 = every check holds. Any thread count (fill_range is serial: called
// outside any region).

#include <cstddef>     // std::size_t
#include <type_traits> // std::conditional_t
#include <utility>     // std::as_const

#include "chunk/chunk_traits.h"          // forward decls + default ind_type = int
#include "chunk/zero_array.h"           // zero::Array
#include "chunk/zero_one_value_array.h"          // zero_one_value::Array
#include "chunk/zero_any_array.h"         // zero_any::Array
#include "chunk/array_chunk_range_ops.h" // fill_range, fill_range_mt
#include "chunk/param_chunk.h"           // ParamChunk::init_indexes
#include "chunk/zero_chunk.h"             // release_chunk_pools, destroy_block_chunk
#include "chunk/config.h"                // chunk_config::use_zero_one_value*
#include "chunk/manage_ompt.h"           // begin_parallel_phase / finalize_parallel_phase
#include "chunk/internal/print.h"        // shrink_array::println

#include "validation_harness.h" // self_command; run() is called by validation_driver.cpp

namespace validation::fill_range_constant_validation
{

using namespace shrink_array;

namespace
{
template <typename T>
using Array =
    std::conditional_t<chunk_config::use_zero_any,
                       zero_any::Array<T>,
                       std::conditional_t<chunk_config::use_zero_one_value, zero_one_value::Array<T>, zero::Array<T>>>;

constexpr const char* variant_name()
{
    if constexpr (chunk_config::use_zero_any)
    {
        return "chunk2a";
    }
    else if constexpr (chunk_config::use_zero_one_value)
    {
        return "chunk2";
    }
    else
    {
        return "chunk";
    }
}

constexpr long n_chunks = 40;

// One region writing what body(i) says, then the between-region pass.
template <typename Fn>
void region(long n, Fn&& body)
{
#pragma omp parallel
    {
        if constexpr (build_config::has_not_ompt)
        {
            shrink_array::begin_parallel_phase();
        }
#pragma omp for schedule(static)
        for (long i = 0; i < n; ++i)
        {
            body(i);
        }
    }
    if constexpr (build_config::has_not_ompt)
    {
        shrink_array::finalize_parallel_phase();
    }
}

// Chunks [c0, c1) still holding a buffer.
long materialised(const Array<double>& a, long c0, long c1)
{
    const long S = static_cast<long>(a.chunk_size());
    long       n = 0;
    for (long c = c0; c < c1; ++c)
    {
        n += a.written(static_cast<int>(c * S)) ? 1 : 0;
    }
    return n;
}

int fails = 0;

// zero_one_value::Chunk and zero_any::Chunk: register @p v as the array's second value (v1 has none).
template <typename Arr>
void maybe_set_second_value(Arr& a, double v)
{
    if constexpr (requires { a.set_second_value(v); })
    {
        a.set_second_value(v);
    }
}

void expect_values(const char* scenario, const Array<double>& a, auto want)
{
    long bad = 0;
    for (long i = 0; i < static_cast<long>(a.size()); ++i)
    {
        const double got = std::as_const(a)[static_cast<int>(i)];
        if (got != want(i))
        {
            if (bad < 5)
            {
                println("[{}/{}] FAIL [{}] = {:g}, expected {:g}", variant_name(), scenario, i, got, want(i));
            }
            ++bad;
        }
    }
    fails += bad ? 1 : 0;
}

// The covered chunks [c0, c1) are all compressed: on zero_any::Chunk whatever the
// value, on v1 and zero_one_value::Chunk after a fill of a constant of theirs
// (@p constant_fill: first_value, or zero_one_value::Chunk's second value).
void expect_compressed(const char* scenario, const Array<double>& a, long c0, long c1, bool constant_fill = false)
{
    if (chunk_config::use_zero_any || constant_fill)
    {
        const long m = materialised(a, c0, c1);
        if (m != 0)
        {
            println("[{}/{}] FAIL {} covered chunk(s) of {} left materialised", variant_name(), scenario, m, c1 - c0);
            ++fails;
        }
    }
}
} // namespace

int run(int /*argc*/, char** /*argv*/)
{
    ParamChunk::init_indexes(true);
    const long S = static_cast<long>(Array<double>(1).chunk_size());
    const long N = n_chunks * S;
    println("=== fill_range_constant_validation [variant = {}]  chunk_size={}  chunks={} ===", variant_name(), S,
            n_chunks);

    // fresh: first_value everywhere, filled with a plain value.
    {
        Array<double> a(static_cast<int>(N));
        fill_range(a, 0, static_cast<std::size_t>(N), 2.5);
        expect_values("fresh", a, [](long) { return 2.5; });
        expect_compressed("fresh", a, 0, n_chunks);
    }

    // second: compressed at V, filled with W; first: compressed at V, filled with 0.
    for (const double w : {7.0, 0.0})
    {
        const char*   name = w == 0.0 ? "first" : "second";
        Array<double> a(static_cast<int>(N));
        region(N, [&](long i) { a[static_cast<int>(i)] = 1.0; });
        if constexpr (chunk_config::use_zero_any)
        {
            a.make_block_compression();
        }
        fill_range(a, 0, static_cast<std::size_t>(N), w);
        expect_values(name, a, [w](long) { return w; });
        expect_compressed(name, a, 0, n_chunks, w == 0.0);
    }

    // varied: materialised with varied values, filled with W.
    {
        Array<double> a(static_cast<int>(N));
        region(N, [&](long i) { a[static_cast<int>(i)] = 0.5 * static_cast<double>(i % 7) + 1.0; });
        fill_range(a, 0, static_cast<std::size_t>(N), 9.0);
        expect_values("varied", a, [](long) { return 9.0; });
        expect_compressed("varied", a, 0, n_chunks);
    }

    // partial: [S/2, N - S/2): the two edge chunks are not covered.
    {
        Array<double> a(static_cast<int>(N));
        region(N, [&](long i) { a[static_cast<int>(i)] = 0.5 * static_cast<double>(i % 7) + 1.0; });
        const long lo = S / 2;
        const long hi = N - S / 2;
        fill_range(a, static_cast<std::size_t>(lo), static_cast<std::size_t>(hi), 4.0);
        expect_values("partial", a,
                      [&](long i) { return (i >= lo && i < hi) ? 4.0 : 0.5 * static_cast<double>(i % 7) + 1.0; });
        expect_compressed("partial", a, 1, n_chunks - 1);
        if (materialised(a, 0, 1) + materialised(a, n_chunks - 1, n_chunks) != 2)
        {
            println("[{}/partial] FAIL an edge chunk (not covered) was compressed", variant_name());
            ++fails;
        }
    }

    // distinct: one value per chunk, then first_value everywhere.
    {
        Array<double> a(static_cast<int>(N));
        for (long c = 0; c < n_chunks; ++c)
        {
            fill_range(a, static_cast<std::size_t>(c * S), static_cast<std::size_t>((c + 1) * S),
                       100.0 + static_cast<double>(c));
        }
        expect_values("distinct", a, [S](long i) { return 100.0 + static_cast<double>(i / S); });
        expect_compressed("distinct", a, 0, n_chunks);
        fill_range(a, 0, static_cast<std::size_t>(N), 0.0);
        expect_values("distinct", a, [](long) { return 0.0; });
        expect_compressed("distinct", a, 0, n_chunks, true);
    }

    // second-value: zero_one_value::Chunk's second constant (zero_any::Chunk: any value; v1: none).
    {
        constexpr bool v1 = !chunk_config::use_zero_any && !chunk_config::use_zero_one_value;
        Array<double>  a(static_cast<int>(N));
        maybe_set_second_value(a, 1.0);
        region(N, [&](long i) { a[static_cast<int>(i)] = 0.5 * static_cast<double>(i % 7) + 1.0; });
        fill_range(a, 0, static_cast<std::size_t>(N), 1.0);
        expect_values("second-value/materialised", a, [](long) { return 1.0; });
        expect_compressed("second-value/materialised", a, 0, n_chunks, !v1);
        fill_range(a, 0, static_cast<std::size_t>(N), 0.0); // back to first_value, compressed
        expect_compressed("second-value/to-first", a, 0, n_chunks, true);
        fill_range(a, 0, static_cast<std::size_t>(N), 1.0); // from first_value: armed, then compressed
        expect_values("second-value/from-first", a, [](long) { return 1.0; });
        expect_compressed("second-value/from-first", a, 0, n_chunks, !v1);
    }

    // constant: first_value over chunks already at first_value -- serially,
    // whole and in part, then from every thread in a phase.
    {
        Array<double> a(static_cast<int>(N));
        fill_range(a, 0, static_cast<std::size_t>(N), 0.0);
        fill_range(a, static_cast<std::size_t>(S / 2), static_cast<std::size_t>(N - S / 2), 0.0);
#pragma omp parallel
        {
            if constexpr (build_config::has_not_ompt)
            {
                shrink_array::begin_parallel_phase();
            }
#pragma omp for schedule(static)
            for (long c = 0; c < n_chunks; ++c)
            {
                fill_range_mt(a, static_cast<std::size_t>(c * S + S / 4), static_cast<std::size_t>((c + 1) * S), 0.0);
            }
        }
        if constexpr (build_config::has_not_ompt)
        {
            shrink_array::finalize_parallel_phase();
        }
        expect_values("constant", a, [](long) { return 0.0; });
        const long m = materialised(a, 0, n_chunks);
        if (m != 0)
        {
            println("[{}/constant] FAIL {} chunk(s) materialised by a fill of the constant they hold", variant_name(),
                    m);
            ++fails;
        }
    }

    release_chunk_pools();
    destroy_block_chunk();
    println("\n[{}] {}", variant_name(), fails == 0 ? "=== FILL RANGE OK ===" : "=== FILL RANGE FAILURES ===");
    return fails ? 1 : 0;
}

} // namespace validation::fill_range_constant_validation
