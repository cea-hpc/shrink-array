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

// ============================================================================
// Range ops and the per-chunk counters. Variant-agnostic: tests chunk / chunk2
// / chunk2a depending on the compile-time macro (CHUNK_VARIANT_V1 / _V2 / _V2A),
// like the other harnesses here.
// ============================================================================
//
// In counter mode (use_compression_with_counter: no fast block compression --
// the serial builds) v1 and Chunk2 decide a chunk's compression on counters the
// write proxy maintains incrementally: how many elements equal the first value
// (v1, Chunk2) and the array's second value (Chunk2). fill_range / copy_range
// store most of a chunk straight into its buffer, past those counters, so they
// must recount afterwards (internal::resync_counters). Chunk2 was once left out:
// its stale counts reached chunk_size on a MIXED chunk and the pass folded it
// to a constant, losing everything else in it (rechunk_validation_chunk2_dyn,
// 34819 elements of 1e6).
//
// Each case: a whole chunk filled with a plain value through the raw path,
// then ONE element set to a constant through the proxy, then a compression
// pass. Every element but that one must still read the plain value. With a
// stale count the chunk comes back all constant.
//   first  -- fill_range, then one first value;
//   second -- fill_range, then one second value (Chunk2 / 2A);
//   copy   -- copy_range from a plain source, then one first value.
// In an OpenMP build the counters do not exist and the cases simply pass.
//
// Exit 0 = every check holds. Any thread count.

#include <cstddef>     // std::size_t
#include <cstdio>      // std::printf
#include <type_traits> // std::conditional_t
#include <utility>     // std::as_const
#include <vector>      // std::vector

#include "chunk/chunk_traits.h"          // forward decls + default ind_type = int
#include "chunk/array_chunk.h"           // ArrayChunk
#include "chunk/array_chunk2.h"          // ArrayChunk2
#include "chunk/array_chunk2A.h"         // ArrayChunk2A
#include "chunk/array_chunk_range_ops.h" // fill_range, copy_range
#include "chunk/param_chunk.h"           // ParamChunk::init_indexes
#include "chunk/chunk_std.h"             // release_chunk_pools, destroy_block_chunk
#include "chunk/config.h"                // chunk_config::use_chunk2*, use_compression_with_counter
#include "chunk/manage_ompt.h"           // begin_parallel_phase / finalize_parallel_phase

#include "validation_harness.h" // self_command; run() is called by validation_driver.cpp

namespace validation::range_counters_validation
{

using namespace shrink_array;

namespace
{

static_assert(chunk_config::use_chunk || chunk_config::use_chunk2 || chunk_config::use_chunk2a,
              "range_counters_validation must be built with CHUNK_VARIANT_V1, _V2 or _V2A");

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

void begin_phase()
{
    if constexpr (build_config::has_not_ompt)
    {
        shrink_array::begin_parallel_phase();
    }
}

void end_phase()
{
    if constexpr (build_config::has_not_ompt)
    {
        shrink_array::finalize_parallel_phase();
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

template <typename Arr>
void compress_in_region(Arr& a)
{
#pragma omp parallel
    {
        begin_phase();
#pragma omp single
        a.compress();
    }
    end_phase();
}

constexpr double kPlain  = 1.0; ///< neither constant
constexpr double kSecond = 2.0; ///< the arrays' second value

enum class Case
{
    first,
    second,
    copy
};

const char* case_name(Case c)
{
    switch (c)
    {
        case Case::first:
            return "first";
        case Case::second:
            return "second";
        default:
            return "copy";
    }
}

/// One chunk (chunk #1, so that chunk #0 stays at the first constant around
/// it) through the raw path, one proxied constant, a compression pass.
int run_case(Case c)
{
    const bool second_applies = chunk_config::use_chunk2 || chunk_config::use_chunk2a;
    if (c == Case::second && !second_applies)
    {
        return 0;
    }

    Array<double> a;
    const long    S = static_cast<long>(a.chunk_size());
    const long    N = 3 * S;
    a.resize(static_cast<int>(N));
    maybe_set_second_value(a, kSecond);

    const auto lo = static_cast<std::size_t>(S);
    const auto hi = static_cast<std::size_t>(2 * S);
    if (c == Case::copy)
    {
        const std::vector<double> src(static_cast<std::size_t>(S), kPlain);
        shrink_array::copy_range(a, lo, hi, src.begin());
    }
    else
    {
        shrink_array::fill_range(a, lo, hi, kPlain);
    }

    // The single proxied write of a constant: with a stale count, the one that
    // makes it reach chunk_size.
    const double constant  = (c == Case::second) ? kSecond : 0.0;
    a[static_cast<int>(S)] = constant;
    compress_in_region(a);

    int bad = 0;
    for (long i = 0; i < N; ++i)
    {
        const double want = (i < S || i >= 2 * S) ? 0.0 : (i == S ? constant : kPlain);
        const double got  = std::as_const(a)[static_cast<int>(i)];
        if (got != want)
        {
            if (bad < 5)
            {
                std::printf("[%s/%s] FAIL [%ld] = %g, expected %g\n", variant_name(), case_name(c), i, got, want);
            }
            ++bad;
        }
    }
    if (!bad)
    {
        std::printf("[%s/%s] OK: raw span of %ld plain values + one %s value, compressed, read back\n", variant_name(),
                    case_name(c), S, c == Case::second ? "second" : "first");
    }
    return bad ? 1 : 0;
}

/// `second * 0` through the proxy (Chunk2): the element leaves the second value
/// AND enters the first. Both counters must move; the second-value count once
/// did not (Proxy2::apply_compound_operation), so it stayed one too high and the
/// LAST of the following second-value writes reached chunk_size with the zero
/// still in the chunk: folded to cst2, the zero read back as the second value
/// (shrink_bench_chunk2_dyn, one element of a chunk in a check run).
int run_mul_zero_case()
{
    if constexpr (!chunk_config::use_chunk2)
    {
        return 0;
    }
    Array<double> a;
    const long    S = static_cast<long>(a.chunk_size());
    const long    N = 3 * S;
    a.resize(static_cast<int>(N));
    maybe_set_second_value(a, kSecond);

    shrink_array::fill_range(a, static_cast<std::size_t>(S), static_cast<std::size_t>(2 * S), kPlain);
    a[static_cast<int>(S)] = kSecond;
    a[static_cast<int>(S)] *= 0.0;
    for (long i = S + 1; i < 2 * S; ++i)
    {
        a[static_cast<int>(i)] = kSecond;
    }
    compress_in_region(a);

    int bad = 0;
    for (long i = 0; i < N; ++i)
    {
        const double want = (i < S || i >= 2 * S || i == S) ? 0.0 : kSecond;
        const double got  = std::as_const(a)[static_cast<int>(i)];
        if (got != want)
        {
            if (bad < 5)
            {
                std::printf("[%s/mul_zero] FAIL [%ld] = %g, expected %g\n", variant_name(), i, got, want);
            }
            ++bad;
        }
    }
    if (!bad)
    {
        std::printf("[%s/mul_zero] OK: second * 0, then the rest of the chunk to the second value\n", variant_name());
    }
    return bad ? 1 : 0;
}

} // namespace

int run(int /*argc*/, char** /*argv*/)
{
    ParamChunk::init_indexes(true);

    std::printf("=== range_counters_validation [variant = %s]  counters=%s ===\n", variant_name(),
                chunk_config::use_compression_with_counter ? "on" : "off (nothing to check)");

    int fails = 0;
    fails += run_case(Case::first);
    fails += run_case(Case::second);
    fails += run_case(Case::copy);
    fails += run_mul_zero_case();

    release_chunk_pools();
    destroy_block_chunk();

    std::printf("\n[%s] %s\n", variant_name(), fails ? "=== RANGE COUNTERS FAILED ===" : "=== RANGE COUNTERS OK ===");
    return fails ? 1 : 0;
}

} // namespace validation::range_counters_validation
