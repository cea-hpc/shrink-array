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
// Then the compound operators on a compressed chunk (run_compound_case).
// In an OpenMP build the counters do not exist and the cases simply pass.
//
// Exit 0 = every check holds. Any thread count.

#include <cstddef>     // std::size_t
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
#include "chunk/internal/print.h"        // shrink_array::println

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

constexpr double plain_value  = 1.0; ///< neither constant
constexpr double second_value = 2.0; ///< the arrays' second value

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
    const long    s = static_cast<long>(a.chunk_size());
    const long    n = 3 * s;
    a.resize(static_cast<int>(n));
    maybe_set_second_value(a, second_value);

    const auto lo = static_cast<std::size_t>(s);
    const auto hi = static_cast<std::size_t>(2 * s);
    if (c == Case::copy)
    {
        const std::vector<double> src(static_cast<std::size_t>(s), plain_value);
        shrink_array::copy_range(a, lo, hi, src.begin());
    }
    else
    {
        shrink_array::fill_range(a, lo, hi, plain_value);
    }

    // The single proxied write of a constant: with a stale count, the one that
    // makes it reach chunk_size.
    const double constant  = (c == Case::second) ? second_value : 0.0;
    a[static_cast<int>(s)] = constant;
    compress_in_region(a);

    int bad = 0;
    for (long i = 0; i < n; ++i)
    {
        const double want = (i < s || i >= 2 * s) ? 0.0 : (i == s ? constant : plain_value);
        const double got  = std::as_const(a)[static_cast<int>(i)];
        if (got != want)
        {
            if (bad < 5)
            {
                println("[{}/{}] FAIL [{}] = {:g}, expected {:g}", variant_name(), case_name(c), i, got, want);
            }
            ++bad;
        }
    }
    if (!bad)
    {
        println("[{}/{}] OK: raw span of {} plain values + one {} value, compressed, read back", variant_name(),
                case_name(c), s, c == Case::second ? "second" : "first");
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
    const long    s = static_cast<long>(a.chunk_size());
    const long    n = 3 * s;
    a.resize(static_cast<int>(n));
    maybe_set_second_value(a, second_value);

    shrink_array::fill_range(a, static_cast<std::size_t>(s), static_cast<std::size_t>(2 * s), plain_value);
    a[static_cast<int>(s)] = second_value;
    a[static_cast<int>(s)] *= 0.0;
    for (long i = s + 1; i < 2 * s; ++i)
    {
        a[static_cast<int>(i)] = second_value;
    }
    compress_in_region(a);

    int bad = 0;
    for (long i = 0; i < n; ++i)
    {
        const double want = (i < s || i >= 2 * s || i == s) ? 0.0 : second_value;
        const double got  = std::as_const(a)[static_cast<int>(i)];
        if (got != want)
        {
            if (bad < 5)
            {
                println("[{}/mul_zero] FAIL [{}] = {:g}, expected {:g}", variant_name(), i, got, want);
            }
            ++bad;
        }
    }
    if (!bad)
    {
        println("[{}/mul_zero] OK: second * 0, then the rest of the chunk to the second value", variant_name());
    }
    return bad ? 1 : 0;
}

// ── Compound operators on a compressed chunk (v1, counter mode) ─────────────
// v1 arms a compressed chunk at nb_val_eq_zero = chunk_size - 1, betting that
// the element written turns non-zero. An operator that leaves it at 0 (0 &= x,
// 0 <<= k, 0 >>= k, 0 += 0, 0 -= 0, 0 |= 0, 0 ^= 0) lost the bet with nothing
// to correct it: one zero short for good, the chunk could never be compressed
// again. *=, /= and %= were never exposed: they skip a zero element.
// Each case: an all-zero array, the operator on one element of chunk #1, which
// must end compressed; then that element set to 9 and back to 0, which must
// compress it again (the count is exact); every element reads 0.
enum class Op
{
    and_,
    shl,
    shr,
    add0,
    sub0,
    or0,
    xor0
};

const char* op_name(Op o)
{
    switch (o)
    {
        case Op::and_:
            return "&= 5";
        case Op::shl:
            return "<<= 3";
        case Op::shr:
            return ">>= 1";
        case Op::add0:
            return "+= 0";
        case Op::sub0:
            return "-= 0";
        case Op::or0:
            return "|= 0";
        default:
            return "^= 0";
    }
}

int run_compound_case(Op o)
{
    using I = long long;
    Array<I>   a;
    const long s = static_cast<long>(a.chunk_size());
    a.resize(static_cast<int>(3 * s));
    maybe_set_second_value(a, I{7});
    const int k = static_cast<int>(s) + 3;
    switch (o)
    {
        case Op::and_:
            a[k] &= I{5};
            break;
        case Op::shl:
            a[k] <<= 3;
            break;
        case Op::shr:
            a[k] >>= 1;
            break;
        case Op::add0:
            a[k] += I{0};
            break;
        case Op::sub0:
            a[k] -= I{0};
            break;
        case Op::or0:
            a[k] |= I{0};
            break;
        case Op::xor0:
            a[k] ^= I{0};
            break;
    }
    int        bad    = 0;
    const bool check  = chunk_config::use_chunk && chunk_config::use_compression_with_counter;
    const bool after1 = std::as_const(a).written(k);
    a[k]              = I{9};
    a[k]              = I{0};
    const bool after2 = std::as_const(a).written(k);
    if (check && (after1 || after2))
    {
        println("[{}/0 {}] FAIL: chunk #1 still materialised {}", variant_name(), op_name(o),
                after1 ? "after the operator (a zero result)" : "after 9 then 0 (the zero count drifted)");
        ++bad;
    }
    for (long i = 0; i < 3 * s; ++i)
    {
        if (std::as_const(a)[static_cast<int>(i)] != I{0})
        {
            if (bad < 5)
            {
                println("[{}/0 {}] FAIL [{}] = {}, expected 0", variant_name(), op_name(o), i,
                        static_cast<long long>(std::as_const(a)[static_cast<int>(i)]));
            }
            ++bad;
        }
    }
    if (!bad)
    {
        println("[{}/0 {}] OK{}", variant_name(), op_name(o),
                check ? ": chunk compressed after the operator and after 9 then 0" : " (values only)");
    }
    return bad ? 1 : 0;
}

} // namespace

int run(int /*argc*/, char** /*argv*/)
{
    ParamChunk::init_indexes(true);

    println("=== range_counters_validation [variant = {}]  counters={} ===", variant_name(),
            chunk_config::use_compression_with_counter ? "on" : "off (nothing to check)");

    int fails = 0;
    fails += run_case(Case::first);
    fails += run_case(Case::second);
    fails += run_case(Case::copy);
    fails += run_mul_zero_case();
    for (const Op o : {Op::and_, Op::shl, Op::shr, Op::add0, Op::sub0, Op::or0, Op::xor0})
    {
        fails += run_compound_case(o);
    }

    release_chunk_pools();
    destroy_block_chunk();

    println("\n[{}] {}", variant_name(), fails ? "=== RANGE COUNTERS FAILED ===" : "=== RANGE COUNTERS OK ===");
    return fails ? 1 : 0;
}

} // namespace validation::range_counters_validation
