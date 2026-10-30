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
// swap() / move-assignment harness. Variant-agnostic: tests zero / zero_one_value /
// zero_any depending on the compile-time macro (CHUNK_VARIANT_ZERO*), like
// the other harnesses here.
// ============================================================================
//
// Both operations exchange chunk lists in O(1), and the chunks are described
// by per-array state kept elsewhere: the proxies' tables indexed by num_tab
// (constant types, zero_any's per-chunk second value, compression candidates)
// and the registry mapping the array's address to num_tab (zero_one_value: to its
// second-value tuple). Moving the chunks alone would make each array read the
// other's metadata -- a zero_any chunk folded to its second value coming back
// as the OTHER array's second value -- and leave the source with nb_chunk over
// an empty list.
//
// Two arrays with different second values, filled by whole chunks (first
// constant, second value, data), compressed, then:
//   swap  -- each reads the other's values, before and after a new compression
//            pass and a rewrite;
//   move  -- the target reads the source's values; the source is empty and
//            takes a resize, writes and a compression pass;
//   move-construct -- the same through the move constructor (not the copy
//            constructor, element by element);
//   vector   -- a std::vector of arrays reallocating: the moves are noexcept,
//            so the arrays are moved (their chunk lists kept), not copied.
//
// Exit 0 = every check holds. Any thread count (the tests run it at 1 and 8).

#include <type_traits> // std::conditional_t, std::is_nothrow_move_constructible_v
#include <utility>     // std::as_const, std::move
#include <vector>      // std::vector

#include "chunk/chunk_traits.h"         // forward decls + default IntType = int
#include "chunk/zero_array.h"           // zero::Array
#include "chunk/zero_one_value_array.h" // zero_one_value::Array
#include "chunk/zero_any_array.h"       // zero_any::Array
#include "chunk/param_chunk.h"          // ParamChunk::init_indexes
#include "chunk/zero_chunk.h"           // release_chunk_pools, destroy_block_chunk
#include "chunk/config.h"               // chunk_config::use_zero*, build_config::has_not_ompt
#include "chunk/PMP_CHUNK.h"            // CHUNK_PMP_MAX_THREADS
#include "chunk/manage_ompt.h"          // begin_parallel_phase / finalize_parallel_phase
#include "chunk/internal/print.h"       // shrink_array::println

#include "validation_harness.h" // self_command; run() is called by validation_driver.cpp

namespace validation::swap_move_validation
{

using namespace shrink_array;

namespace
{

static_assert(chunk_config::use_zero || chunk_config::use_zero_one_value || chunk_config::use_zero_any,
              "swap_move_validation must be built with CHUNK_VARIANT_ZERO, _ZERO_ONE_VALUE or _ZERO_ANY");

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

/// A pattern by whole chunks: first constant, second value @p sv, data.
struct Pattern
{
    [[nodiscard]] double at(long index) const
    {
        switch ((index / s) % 3)
        {
            case 0:
                return 0.0;
            case 1:
                return sv;
            default:
                return base + static_cast<double>(index % 101);
        }
    }

    double sv;   ///< this array's second value
    double base; ///< offset of its data chunks
    long   s;    ///< chunk size
};

template <typename Arr>
void fill(Arr& arr, const Pattern& pattern, long count)
{
#pragma omp parallel
    {
        begin_phase();
#pragma omp for schedule(static)
        for (long i = 0; i < count; ++i)
        {
            arr[static_cast<int>(i)] = pattern.at(i);
        }
    }
    end_phase();
}

template <typename Arr>
int verify(const char* step, const char* name, const Arr& arr, const Pattern& pattern, long count)
{
    int bad = 0;
    if (static_cast<long>(arr.size()) != count)
    {
        println("[{}/{}] FAIL {}: size() = {}, expected {}", variant_name(), step, name, static_cast<long>(arr.size()),
                count);
        return 1;
    }
    for (long i = 0; i < count; ++i)
    {
        const double got  = arr[static_cast<int>(i)];
        const double want = pattern.at(i);
        if (got != want)
        {
            if (bad < 5)
            {
                println("[{}/{}] FAIL {}[{}] = {:g}, expected {:g} (chunk {})", variant_name(), step, name, i, got,
                        want, i / pattern.s);
            }
            ++bad;
        }
    }
    return bad;
}

int check_swap()
{
    const char*   step = "swap";
    Array<double> first_arr;
    Array<double> second_arr;
    const long    chunk_sz = static_cast<long>(first_arr.chunk_size());
    // Different sizes: the per-array tables (indexed by num_tab) have one row
    // per chunk, so chunks exchanged without their identity would index rows
    // of the wrong length.
    const long count = (30 * chunk_sz) + 11;
    const long nb    = (47 * chunk_sz) + 5;
    first_arr.resize(static_cast<int>(count));
    second_arr.resize(static_cast<int>(nb));
    const Pattern pa{2.0, 10.0, chunk_sz};
    const Pattern pb{5.0, 500.0, chunk_sz};
    maybe_set_second_value(first_arr, pa.sv);
    maybe_set_second_value(second_arr, pb.sv);
    fill(first_arr, pa, count);
    fill(second_arr, pb, nb);
    compress_in_region(first_arr);
    compress_in_region(second_arr);

    int bad = verify(step, "a (before)", std::as_const(first_arr), pa, count) +
              verify(step, "b (before)", std::as_const(second_arr), pb, nb);

    first_arr.swap(second_arr);
    bad += verify(step, "a (swapped)", std::as_const(first_arr), pb, nb) +
           verify(step, "b (swapped)", std::as_const(second_arr), pa, count);

    // The exchanged metadata must keep working: a pass, a rewrite, a pass.
    compress_in_region(first_arr);
    compress_in_region(second_arr);
    bad += verify(step, "a (compressed)", std::as_const(first_arr), pb, nb) +
           verify(step, "b (compressed)", std::as_const(second_arr), pa, count);
    const Pattern pb2{pb.sv, 900.0, chunk_sz};
    const Pattern pa2{pa.sv, 90.0, chunk_sz};
    fill(first_arr, pb2, nb);
    fill(second_arr, pa2, count);
    compress_in_region(first_arr);
    compress_in_region(second_arr);
    bad += verify(step, "a (rewritten)", std::as_const(first_arr), pb2, nb) +
           verify(step, "b (rewritten)", std::as_const(second_arr), pa2, count);

    if (!bad)
    {
        println("[{}/{}] OK: N={}, second values {:g}/{:g} exchanged with the chunks, compressed and rewritten",
                variant_name(), step, count, pa.sv, pb.sv);
    }
    return bad ? 1 : 0;
}

int check_move()
{
    const char*   step = "move";
    Array<double> src;
    Array<double> dst;
    const long    chunk_sz    = static_cast<long>(src.chunk_size());
    const long    count       = (25 * chunk_sz) + 3;
    const long    other_count = (12 * chunk_sz) + 5;
    src.resize(static_cast<int>(count));
    dst.resize(static_cast<int>(other_count));
    const Pattern ps{3.0, 30.0, chunk_sz};
    const Pattern pd{7.0, 700.0, chunk_sz};
    maybe_set_second_value(src, ps.sv);
    maybe_set_second_value(dst, pd.sv);
    fill(src, ps, count);
    fill(dst, pd, other_count);
    compress_in_region(src);
    compress_in_region(dst);

    dst = std::move(src);

    int bad = verify(step, "dst (moved into)", std::as_const(dst), ps, count);
    // Reading the moved-from array is the point: the move-assignment's
    // contract leaves it empty and valid, and that is what is checked here.
    // NOLINTNEXTLINE(bugprone-use-after-move)
    if (!src.empty() || src.chunk_count() != 0)
    {
        println("[{}/{}] FAIL source after move: size() = {}, chunk_count() = {}, expected 0 and 0", variant_name(),
                step, static_cast<long>(src.size()), src.chunk_count());
        ++bad;
    }
    compress_in_region(dst);
    bad += verify(step, "dst (compressed)", std::as_const(dst), ps, count);

    // The moved-from array is empty, not broken.
    const Pattern pr{pd.sv, 800.0, chunk_sz};
    maybe_set_second_value(src, pr.sv);
    src.resize(static_cast<int>(other_count));
    fill(src, pr, other_count);
    compress_in_region(src);
    bad += verify(step, "src (reused)", std::as_const(src), pr, other_count);
    bad += verify(step, "dst (untouched)", std::as_const(dst), ps, count);

    if (!bad)
    {
        println("[{}/{}] OK: N={} moved over M={}, the source reused", variant_name(), step, count, other_count);
    }
    return bad ? 1 : 0;
}

int check_move_construct()
{
    const char*   step = "move-construct";
    Array<double> src;
    const long    chunk_sz    = static_cast<long>(src.chunk_size());
    const long    count       = (20 * chunk_sz) + 9;
    const long    other_count = (7 * chunk_sz) + 1;
    src.resize(static_cast<int>(count));
    const Pattern ps{4.0, 40.0, chunk_sz};
    maybe_set_second_value(src, ps.sv);
    fill(src, ps, count);
    compress_in_region(src);

    Array<double> dst(std::move(src));

    int bad = verify(step, "dst (constructed)", std::as_const(dst), ps, count);
    // Reading the moved-from array is the point, as in check_move().
    // NOLINTNEXTLINE(bugprone-use-after-move)
    if (!src.empty() || src.chunk_count() != 0)
    {
        println("[{}/{}] FAIL source after move: size() = {}, chunk_count() = {}, expected 0 and 0", variant_name(),
                step, static_cast<long>(src.size()), src.chunk_count());
        ++bad;
    }
    compress_in_region(dst);
    bad += verify(step, "dst (compressed)", std::as_const(dst), ps, count);

    const Pattern pr{6.0, 600.0, chunk_sz};
    maybe_set_second_value(src, pr.sv);
    src.resize(static_cast<int>(other_count));
    fill(src, pr, other_count);
    compress_in_region(src);
    bad += verify(step, "src (reused)", std::as_const(src), pr, other_count);
    bad += verify(step, "dst (untouched)", std::as_const(dst), ps, count);

    if (!bad)
    {
        println("[{}/{}] OK: N={} moved into a new array, the source reused", variant_name(), step, count);
    }
    return bad ? 1 : 0;
}

static_assert(std::is_nothrow_move_constructible_v<Array<double>> && std::is_nothrow_move_assignable_v<Array<double>>,
              "std::vector copies on reallocation unless the moves are noexcept");

int check_vector_realloc()
{
    const char*                step = "vector";
    std::vector<Array<double>> arrays;
    arrays.reserve(1);
    arrays.emplace_back(0);
    const long    chunk_sz    = static_cast<long>(arrays[0].chunk_size());
    const long    count       = (18 * chunk_sz) + 7;
    const long    other_count = (9 * chunk_sz) + 2;
    const Pattern p0{2.0, 20.0, chunk_sz};
    const Pattern p1{8.0, 80.0, chunk_sz};
    arrays[0].resize(static_cast<int>(count));
    maybe_set_second_value(arrays[0], p0.sv);
    fill(arrays[0], p0, count);
    compress_in_region(arrays[0]);
    const auto* const chunks_before = arrays[0].get_chunk_list().data();

    arrays.emplace_back(static_cast<int>(other_count)); // capacity 1: reallocates, moving arrays[0]

    int bad = 0;
    if (arrays[0].get_chunk_list().data() != chunks_before)
    {
        println("[{}/{}] FAIL the reallocation copied the array instead of moving it", variant_name(), step);
        ++bad;
    }
    maybe_set_second_value(arrays[1], p1.sv);
    fill(arrays[1], p1, other_count);
    compress_in_region(arrays[1]);
    compress_in_region(arrays[0]);
    bad += verify(step, "v[0] (moved)", std::as_const(arrays[0]), p0, count);
    bad += verify(step, "v[1]", std::as_const(arrays[1]), p1, other_count);

    if (!bad)
    {
        println("[{}/{}] OK: N={} moved by the reallocation, chunk list kept", variant_name(), step, count);
    }
    return bad ? 1 : 0;
}

} // namespace

int run(int /*argc*/, char** /*argv*/)
{
    ParamChunk::init_indexes(true);

    println("=== swap_move_validation [variant = {}]  threads={} ===", variant_name(), CHUNK_PMP_MAX_THREADS);

    int fails = 0;
    fails += check_swap();
    fails += check_move();
    fails += check_move_construct();
    fails += check_vector_realloc();

    release_chunk_pools();
    destroy_block_chunk();

    println("\n[{}] {}", variant_name(), fails ? "=== SWAP/MOVE FAILED ===" : "=== SWAP/MOVE OK ===");
    return fails ? 1 : 0;
}

} // namespace validation::swap_move_validation
