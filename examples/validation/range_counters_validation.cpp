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
// Range ops and the per-chunk counters. Variant-agnostic: tests zero / zero_one_value
// / zero_any depending on the compile-time macro (CHUNK_VARIANT_ZERO*),
// like the other harnesses here.
// ============================================================================
//
// In counter mode (use_compression_with_counter: no fast block compression --
// the serial builds) zero and zero_one_value decide a chunk's compression on counters the
// write proxy maintains incrementally: how many elements equal the first value
// (zero, zero_one_value) and the array's second value (zero_one_value). fill_range / copy_range
// store most of a chunk straight into its buffer, past those counters, so they
// must recount afterwards (internal::resync_counters). zero_one_value was once left out:
// its stale counts reached chunk_size on a MIXED chunk and the pass folded it
// to a constant, losing everything else in it (rechunk_validation_zero_one_value_dyn,
// 34819 elements of 1e6).
//
// Each case: a whole chunk filled with a plain value through the raw path,
// then ONE element set to a constant through the proxy, then a compression
// pass. Every element but that one must still read the plain value. With a
// stale count the chunk comes back all constant.
//   first  -- fill_range, then one first value;
//   second -- fill_range, then one second value (zero_one_value / zero_any);
//   copy   -- copy_range from a plain source, then one first value.
// Then fill_range over chunks already constant (run_fill_constant_case).
// Then the compound operators on a compressed chunk (run_compound_case).
// In an OpenMP build the counters do not exist and the cases simply pass.
//
// Exit 0 = every check holds. Any thread count.

#include <algorithm>   // std::fill (run_fill_constant_case's reference)
#include <cstddef>     // std::size_t
#include <type_traits> // std::conditional_t
#include <utility>     // std::as_const
#include <vector>      // std::vector

#include "chunk/chunk_traits.h"          // forward decls + default IntType = int
#include "chunk/zero_array.h"            // zero::Array
#include "chunk/zero_one_value_array.h"  // zero_one_value::Array
#include "chunk/zero_any_array.h"        // zero_any::Array
#include "chunk/array_chunk_range_ops.h" // fill_range, copy_range
#include "chunk/param_chunk.h"           // ParamChunk::init_indexes
#include "chunk/zero_chunk.h"            // release_chunk_pools, destroy_block_chunk
#include "chunk/config.h"                // chunk_config::use_zero_one_value*, use_compression_with_counter
#include "chunk/manage_ompt.h"           // begin_parallel_phase / finalize_parallel_phase
#include "chunk/internal/print.h"        // shrink_array::println

#include "validation_harness.h" // self_command; run() is called by validation_driver.cpp

namespace validation::range_counters_validation
{

using namespace shrink_array;

namespace
{

static_assert(chunk_config::use_zero || chunk_config::use_zero_one_value || chunk_config::use_zero_any,
              "range_counters_validation must be built with CHUNK_VARIANT_ZERO, _ZERO_ONE_VALUE or _ZERO_ANY");

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
void maybe_set_second_value(Arr& arr, const Val& value)
{
    if constexpr (requires(Arr probe) { probe.set_second_value(value); })
    {
        arr.set_second_value(value);
    }
}

template <typename Arr>
void compress_in_region(Arr& arr)
{
#pragma omp parallel
    {
        begin_phase();
#pragma omp single
        arr.compress();
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

const char* case_name(Case test_case)
{
    switch (test_case)
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
int run_case(Case test_case)
{
    const bool second_applies = chunk_config::use_zero_one_value || chunk_config::use_zero_any;
    if (test_case == Case::second && !second_applies)
    {
        return 0;
    }

    Array<double> arr;
    const long    chunk_sz = static_cast<long>(arr.chunk_size());
    const long    count    = 3 * chunk_sz;
    arr.resize(static_cast<int>(count));
    maybe_set_second_value(arr, second_value);

    const auto lo = static_cast<std::size_t>(chunk_sz);
    const auto hi = static_cast<std::size_t>(2 * chunk_sz);
    if (test_case == Case::copy)
    {
        const std::vector<double> src(static_cast<std::size_t>(chunk_sz), plain_value);
        shrink_array::copy_range(arr, lo, hi, src.begin());
    }
    else
    {
        shrink_array::fill_range(arr, lo, hi, plain_value);
    }

    // The single proxied write of a constant: with a stale count, the one that
    // makes it reach chunk_size.
    const double constant           = (test_case == Case::second) ? second_value : 0.0;
    arr[static_cast<int>(chunk_sz)] = constant;
    compress_in_region(arr);

    int bad = 0;
    for (long i = 0; i < count; ++i)
    {
        const double want = (i < chunk_sz || i >= 2 * chunk_sz) ? 0.0 : (i == chunk_sz ? constant : plain_value);
        const double got  = std::as_const(arr)[static_cast<int>(i)];
        if (got != want)
        {
            if (bad < 5)
            {
                println("[{}/{}] FAIL [{}] = {:g}, expected {:g}", variant_name(), case_name(test_case), i, got, want);
            }
            ++bad;
        }
    }
    if (!bad)
    {
        println("[{}/{}] OK: raw span of {} plain values + one {} value, compressed, read back", variant_name(),
                case_name(test_case), chunk_sz, test_case == Case::second ? "second" : "first");
    }
    return bad ? 1 : 0;
}

/// fill_range over chunks already constant: the value a chunk already holds
/// (whole chunk, part of one, across two), and the values it does not -- the
/// first value onto a second-constant chunk, the second onto a first-constant
/// one, and on zero_any a second value onto a chunk at ANOTHER second value.
/// Every element against a reference after each fill, then after a pass; a
/// fill of the chunk's own constant must leave it constant.
int run_fill_constant_case()
{
    const bool second_applies = chunk_config::use_zero_one_value || chunk_config::use_zero_any;

    Array<double> arr;
    const long    csz   = static_cast<long>(arr.chunk_size());
    const long    count = 6 * csz;
    arr.resize(static_cast<int>(count));
    maybe_set_second_value(arr, second_value);
    std::vector<double> ref(static_cast<std::size_t>(count), 0.0);

    const auto fill = [&](long lo, long hi, double value)
    {
        shrink_array::fill_range(arr, static_cast<std::size_t>(lo), static_cast<std::size_t>(hi), value);
        std::fill(ref.begin() + lo, ref.begin() + hi, value);
    };
    const auto constant = [&](long c)
    {
        return !shrink_array::internal::materialised(arr, arr.chunk_ref(static_cast<std::size_t>(c)),
                                                     shrink_array::internal::second_value_index(arr));
    };
    int        bad   = 0;
    const auto check = [&](const char* step)
    {
        long wrong = 0;
        for (long i = 0; i < count; ++i)
        {
            const double got = std::as_const(arr)[static_cast<int>(i)];
            if (got != ref[static_cast<std::size_t>(i)])
            {
                if (wrong < 5)
                {
                    println("[{}/fill-constant] FAIL {}: [{}] = {:g}, expected {:g}", variant_name(), step, i, got,
                            ref[static_cast<std::size_t>(i)]);
                }
                ++wrong;
            }
        }
        bad += wrong != 0 ? 1 : 0;
    };
    const auto expect_constant = [&](long c, const char* step)
    {
        if (!constant(c))
        {
            println("[{}/fill-constant] FAIL {}: chunk {} materialised", variant_name(), step, c);
            ++bad;
        }
    };

    // Chunks 0, 4, 5 at the first constant; 1 at the second value (2.0);
    // 2 at 3.0 (zero_any: a second constant of its own; plain elsewhere);
    // 3 mixed.
    fill(1 * csz, 2 * csz, second_applies ? second_value : plain_value);
    fill(2 * csz, 3 * csz, 3.0);
    fill(3 * csz, 4 * csz, plain_value);
    arr[static_cast<int>(3 * csz)]         = 0.0;
    ref[static_cast<std::size_t>(3 * csz)] = 0.0;
    compress_in_region(arr);
    // Serial zero_any has no between-region pass: the writers' pass folds the
    // uniform chunks (as pattern_validation's end_phase).
    if constexpr (chunk_config::use_zero_any && !build_config::use_openmp)
    {
        Array<double>::make_block_compression();
    }
    check("setup");
    expect_constant(0, "setup");
    if (second_applies)
    {
        expect_constant(1, "setup");
    }
    if (chunk_config::use_zero_any)
    {
        expect_constant(2, "setup");
    }

    // The chunk's own constant: nothing to write, the chunk stays constant.
    fill(0, csz, 0.0);
    check("first value, whole cst1 chunk");
    expect_constant(0, "first value, whole cst1 chunk");
    fill(4 * csz + 3, 5 * csz - 5, 0.0);
    check("first value, part of a cst1 chunk");
    expect_constant(4, "first value, part of a cst1 chunk");
    fill(5 * csz - 7, 6 * csz, 0.0);
    check("first value, across two cst1 chunks");
    expect_constant(4, "first value, across two cst1 chunks");
    expect_constant(5, "first value, across two cst1 chunks");
    if (second_applies)
    {
        fill(csz + 5, 2 * csz - 3, second_value);
        check("second value, part of its cst2 chunk");
        expect_constant(1, "second value, part of its cst2 chunk");
        fill(csz, 2 * csz, second_value);
        check("second value, its whole cst2 chunk");
        expect_constant(1, "second value, its whole cst2 chunk");
    }

    // A value the chunk does not hold: written.
    if (chunk_config::use_zero_any)
    {
        fill(2 * csz + 2, 3 * csz - 9, second_value); // onto the chunk at 3.0
        check("second value onto another cst2 chunk");
    }
    fill(csz + 1, csz + 11, 0.0);
    check("first value onto a cst2 chunk");
    fill(5 * csz + 4, 5 * csz + 20, second_value);
    check("second value onto a cst1 chunk");
    fill(3 * csz + 1, 3 * csz + csz / 2, 0.0);
    check("first value onto a mixed chunk");

    compress_in_region(arr);
    check("after the pass");

    if (bad == 0)
    {
        println("[{}/fill-constant] OK: fill_range over constant chunks, {} elements read back", variant_name(), count);
    }
    return bad != 0 ? 1 : 0;
}

/// `second * 0` through the proxy (zero_one_value): the element leaves the second value
/// AND enters the first. Both counters must move; the second-value count once
/// did not (zero_one_value::Proxy::apply_compound_operation), so it stayed one too high and the
/// LAST of the following second-value writes reached chunk_size with the zero
/// still in the chunk: folded to cst2, the zero read back as the second value
/// (shrink_bench_zero_one_value_dyn, one element of a chunk in a check run).
int run_mul_zero_case()
{
    if constexpr (!chunk_config::use_zero_one_value)
    {
        return 0;
    }
    Array<double> arr;
    const long    chunk_sz = static_cast<long>(arr.chunk_size());
    const long    count    = 3 * chunk_sz;
    arr.resize(static_cast<int>(count));
    maybe_set_second_value(arr, second_value);

    shrink_array::fill_range(arr, static_cast<std::size_t>(chunk_sz), static_cast<std::size_t>(2 * chunk_sz),
                             plain_value);
    arr[static_cast<int>(chunk_sz)] = second_value;
    arr[static_cast<int>(chunk_sz)] *= 0.0;
    for (long i = chunk_sz + 1; i < 2 * chunk_sz; ++i)
    {
        arr[static_cast<int>(i)] = second_value;
    }
    compress_in_region(arr);

    int bad = 0;
    for (long i = 0; i < count; ++i)
    {
        const double want = (i < chunk_sz || i >= 2 * chunk_sz || i == chunk_sz) ? 0.0 : second_value;
        const double got  = std::as_const(arr)[static_cast<int>(i)];
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

// ── Compound operators on a compressed chunk (zero, counter mode) ───────────
// zero arms a compressed chunk at nb_val_eq_zero = chunk_size - 1, betting that
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

const char* op_name(Op op)
{
    switch (op)
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

int run_compound_case(Op op)
{
    using I = long long;
    Array<I>   arr;
    const long chunk_sz = static_cast<long>(arr.chunk_size());
    arr.resize(static_cast<int>(3 * chunk_sz));
    maybe_set_second_value(arr, I{7});
    const int span = static_cast<int>(chunk_sz) + 3;
    switch (op)
    {
        case Op::and_:
            arr[span] &= I{5};
            break;
        case Op::shl:
            arr[span] <<= 3;
            break;
        case Op::shr:
            arr[span] >>= 1;
            break;
        case Op::add0:
            arr[span] += I{0};
            break;
        case Op::sub0:
            arr[span] -= I{0};
            break;
        case Op::or0:
            arr[span] |= I{0};
            break;
        case Op::xor0:
            arr[span] ^= I{0};
            break;
    }
    int        bad    = 0;
    const bool check  = chunk_config::use_zero && chunk_config::use_compression_with_counter;
    const bool after1 = std::as_const(arr).written(span);
    arr[span]         = I{9};
    arr[span]         = I{0};
    const bool after2 = std::as_const(arr).written(span);
    if (check && (after1 || after2))
    {
        println("[{}/0 {}] FAIL: chunk #1 still materialised {}", variant_name(), op_name(op),
                after1 ? "after the operator (a zero result)" : "after 9 then 0 (the zero count drifted)");
        ++bad;
    }
    for (long i = 0; i < 3 * chunk_sz; ++i)
    {
        if (std::as_const(arr)[static_cast<int>(i)] != I{0})
        {
            if (bad < 5)
            {
                println("[{}/0 {}] FAIL [{}] = {}, expected 0", variant_name(), op_name(op), i,
                        static_cast<long long>(std::as_const(arr)[static_cast<int>(i)]));
            }
            ++bad;
        }
    }
    if (!bad)
    {
        println("[{}/0 {}] OK{}", variant_name(), op_name(op),
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
    fails += run_fill_constant_case();
    fails += run_mul_zero_case();
    for (const Op op : {Op::and_, Op::shl, Op::shr, Op::add0, Op::sub0, Op::or0, Op::xor0})
    {
        fails += run_compound_case(op);
    }

    release_chunk_pools();
    destroy_block_chunk();

    println("\n[{}] {}", variant_name(), fails ? "=== RANGE COUNTERS FAILED ===" : "=== RANGE COUNTERS OK ===");
    return fails ? 1 : 0;
}

} // namespace validation::range_counters_validation
