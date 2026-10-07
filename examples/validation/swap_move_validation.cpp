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
// swap() / move-assignment harness. Variant-agnostic: tests chunk / chunk2 /
// chunk2a depending on the compile-time macro (CHUNK_VARIANT_V1 / _V2 / _V2A), like
// the other harnesses here.
// ============================================================================
//
// Both operations exchange chunk lists in O(1), and the chunks are described
// by per-array state kept elsewhere: the proxies' tables indexed by num_tab
// (constant types, Chunk2A's per-chunk second value, compression candidates)
// and the registry mapping the array's address to num_tab (Chunk2: to its
// second-value tuple). Moving the chunks alone would make each array read the
// other's metadata -- a Chunk2A chunk folded to its second value coming back
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

#include <cstdio>      // std::printf
#include <type_traits> // std::conditional_t, std::is_nothrow_move_constructible_v
#include <utility>     // std::as_const, std::move
#include <vector>      // std::vector
#include <print>       // std::println

#include "chunk/chunk_traits.h"  // forward decls + default ind_type = int
#include "chunk/array_chunk.h"   // ArrayChunk
#include "chunk/array_chunk2.h"  // ArrayChunk2
#include "chunk/array_chunk2A.h" // ArrayChunk2A
#include "chunk/param_chunk.h"   // ParamChunk::init_indexes
#include "chunk/chunk_std.h"     // release_chunk_pools, destroy_block_chunk
#include "chunk/config.h"        // chunk_config::use_chunk*, build_config::has_not_ompt
#include "chunk/PMP_CHUNK.h"     // CHUNK_PMP_MAX_THREADS
#include "chunk/manage_ompt.h"   // begin_parallel_phase / finalize_parallel_phase

#include "validation_harness.h" // self_command; run() is called by validation_driver.cpp

namespace validation::swap_move_validation
{

using namespace shrink_array;

namespace
{

static_assert(chunk_config::use_chunk || chunk_config::use_chunk2 || chunk_config::use_chunk2a,
              "swap_move_validation must be built with CHUNK_VARIANT_V1, _V2 or _V2A");

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

/// A pattern by whole chunks: first constant, second value @p sv, data.
struct Pattern
{
    [[nodiscard]] double at(long i) const
    {
        switch ((i / S) % 3)
        {
            case 0:
                return 0.0;
            case 1:
                return sv;
            default:
                return base + static_cast<double>(i % 101);
        }
    }

    double sv;   ///< this array's second value
    double base; ///< offset of its data chunks
    long   S;    ///< chunk size
};

template <typename Arr>
void fill(Arr& a, const Pattern& p, long n)
{
#pragma omp parallel
    {
        begin_phase();
#pragma omp for schedule(static)
        for (long i = 0; i < n; ++i)
        {
            a[static_cast<int>(i)] = p.at(i);
        }
    }
    end_phase();
}

template <typename Arr>
int verify(const char* step, const char* name, const Arr& a, const Pattern& p, long n)
{
    int bad = 0;
    if (static_cast<long>(a.size()) != n)
    {
        std::println("[{}/{}] FAIL {}: size() = {}, expected {}", variant_name(), step, name,
                     static_cast<long>(a.size()), n);
        return 1;
    }
    for (long i = 0; i < n; ++i)
    {
        const double got  = a[static_cast<int>(i)];
        const double want = p.at(i);
        if (got != want)
        {
            if (bad < 5)
            {
                std::println("[{}/{}] FAIL {}[{}] = {:g}, expected {:g} (chunk {})", variant_name(), step, name, i, got,
                             want, i / p.S);
            }
            ++bad;
        }
    }
    return bad;
}

int check_swap()
{
    const char*   step = "swap";
    Array<double> a;
    Array<double> b;
    const long    S = static_cast<long>(a.chunk_size());
    // Different sizes: the per-array tables (indexed by num_tab) have one row
    // per chunk, so chunks exchanged without their identity would index rows
    // of the wrong length.
    const long N  = (30 * S) + 11;
    const long NB = (47 * S) + 5;
    a.resize(static_cast<int>(N));
    b.resize(static_cast<int>(NB));
    const Pattern pa{2.0, 10.0, S};
    const Pattern pb{5.0, 500.0, S};
    maybe_set_second_value(a, pa.sv);
    maybe_set_second_value(b, pb.sv);
    fill(a, pa, N);
    fill(b, pb, NB);
    compress_in_region(a);
    compress_in_region(b);

    int bad =
        verify(step, "a (before)", std::as_const(a), pa, N) + verify(step, "b (before)", std::as_const(b), pb, NB);

    a.swap(b);
    bad += verify(step, "a (swapped)", std::as_const(a), pb, NB) + verify(step, "b (swapped)", std::as_const(b), pa, N);

    // The exchanged metadata must keep working: a pass, a rewrite, a pass.
    compress_in_region(a);
    compress_in_region(b);
    bad += verify(step, "a (compressed)", std::as_const(a), pb, NB) +
           verify(step, "b (compressed)", std::as_const(b), pa, N);
    const Pattern pb2{pb.sv, 900.0, S};
    const Pattern pa2{pa.sv, 90.0, S};
    fill(a, pb2, NB);
    fill(b, pa2, N);
    compress_in_region(a);
    compress_in_region(b);
    bad += verify(step, "a (rewritten)", std::as_const(a), pb2, NB) +
           verify(step, "b (rewritten)", std::as_const(b), pa2, N);

    if (!bad)
    {
        std::println("[{}/{}] OK: N={}, second values {:g}/{:g} exchanged with the chunks, compressed and rewritten",
                     variant_name(), step, N, pa.sv, pb.sv);
    }
    return bad ? 1 : 0;
}

int check_move()
{
    const char*   step = "move";
    Array<double> src;
    Array<double> dst;
    const long    S = static_cast<long>(src.chunk_size());
    const long    N = (25 * S) + 3;
    const long    M = (12 * S) + 5;
    src.resize(static_cast<int>(N));
    dst.resize(static_cast<int>(M));
    const Pattern ps{3.0, 30.0, S};
    const Pattern pd{7.0, 700.0, S};
    maybe_set_second_value(src, ps.sv);
    maybe_set_second_value(dst, pd.sv);
    fill(src, ps, N);
    fill(dst, pd, M);
    compress_in_region(src);
    compress_in_region(dst);

    dst = std::move(src);

    int bad = verify(step, "dst (moved into)", std::as_const(dst), ps, N);
    // Reading the moved-from array is the point: the move-assignment's
    // contract leaves it empty and valid, and that is what is checked here.
    // NOLINTNEXTLINE(bugprone-use-after-move)
    if (!src.empty() || src.chunk_count() != 0)
    {
        std::println("[{}/{}] FAIL source after move: size() = {}, chunk_count() = {}, expected 0 and 0",
                     variant_name(), step, static_cast<long>(src.size()), src.chunk_count());
        ++bad;
    }
    compress_in_region(dst);
    bad += verify(step, "dst (compressed)", std::as_const(dst), ps, N);

    // The moved-from array is empty, not broken.
    const Pattern pr{pd.sv, 800.0, S};
    maybe_set_second_value(src, pr.sv);
    src.resize(static_cast<int>(M));
    fill(src, pr, M);
    compress_in_region(src);
    bad += verify(step, "src (reused)", std::as_const(src), pr, M);
    bad += verify(step, "dst (untouched)", std::as_const(dst), ps, N);

    if (!bad)
    {
        std::println("[{}/{}] OK: N={} moved over M={}, the source reused", variant_name(), step, N, M);
    }
    return bad ? 1 : 0;
}

int check_move_construct()
{
    const char*   step = "move-construct";
    Array<double> src;
    const long    S = static_cast<long>(src.chunk_size());
    const long    N = (20 * S) + 9;
    const long    M = (7 * S) + 1;
    src.resize(static_cast<int>(N));
    const Pattern ps{4.0, 40.0, S};
    maybe_set_second_value(src, ps.sv);
    fill(src, ps, N);
    compress_in_region(src);

    Array<double> dst(std::move(src));

    int bad = verify(step, "dst (constructed)", std::as_const(dst), ps, N);
    // Reading the moved-from array is the point, as in check_move().
    // NOLINTNEXTLINE(bugprone-use-after-move)
    if (!src.empty() || src.chunk_count() != 0)
    {
        std::println("[{}/{}] FAIL source after move: size() = {}, chunk_count() = {}, expected 0 and 0",
                     variant_name(), step, static_cast<long>(src.size()), src.chunk_count());
        ++bad;
    }
    compress_in_region(dst);
    bad += verify(step, "dst (compressed)", std::as_const(dst), ps, N);

    const Pattern pr{6.0, 600.0, S};
    maybe_set_second_value(src, pr.sv);
    src.resize(static_cast<int>(M));
    fill(src, pr, M);
    compress_in_region(src);
    bad += verify(step, "src (reused)", std::as_const(src), pr, M);
    bad += verify(step, "dst (untouched)", std::as_const(dst), ps, N);

    if (!bad)
    {
        std::println("[{}/{}] OK: N={} moved into a new array, the source reused", variant_name(), step, N);
    }
    return bad ? 1 : 0;
}

static_assert(std::is_nothrow_move_constructible_v<Array<double>> && std::is_nothrow_move_assignable_v<Array<double>>,
              "std::vector copies on reallocation unless the moves are noexcept");

int check_vector_realloc()
{
    const char*                step = "vector";
    std::vector<Array<double>> v;
    v.reserve(1);
    v.emplace_back(0);
    const long    S = static_cast<long>(v[0].chunk_size());
    const long    N = (18 * S) + 7;
    const long    M = (9 * S) + 2;
    const Pattern p0{2.0, 20.0, S};
    const Pattern p1{8.0, 80.0, S};
    v[0].resize(static_cast<int>(N));
    maybe_set_second_value(v[0], p0.sv);
    fill(v[0], p0, N);
    compress_in_region(v[0]);
    const auto* const chunks_before = v[0].getChunkList().data();

    v.emplace_back(static_cast<int>(M)); // capacity 1: reallocates, moving v[0]

    int bad = 0;
    if (v[0].getChunkList().data() != chunks_before)
    {
        std::println("[{}/{}] FAIL the reallocation copied the array instead of moving it", variant_name(), step);
        ++bad;
    }
    maybe_set_second_value(v[1], p1.sv);
    fill(v[1], p1, M);
    compress_in_region(v[1]);
    compress_in_region(v[0]);
    bad += verify(step, "v[0] (moved)", std::as_const(v[0]), p0, N);
    bad += verify(step, "v[1]", std::as_const(v[1]), p1, M);

    if (!bad)
    {
        std::println("[{}/{}] OK: N={} moved by the reallocation, chunk list kept", variant_name(), step, N);
    }
    return bad ? 1 : 0;
}

} // namespace

int run(int /*argc*/, char** /*argv*/)
{
    ParamChunk::init_indexes(true);

    std::println("=== swap_move_validation [variant = {}]  threads={} ===", variant_name(), CHUNK_PMP_MAX_THREADS);

    int fails = 0;
    fails += check_swap();
    fails += check_move();
    fails += check_move_construct();
    fails += check_vector_realloc();

    release_chunk_pools();
    destroy_block_chunk();

    std::println("\n[{}] {}", variant_name(), fails ? "=== SWAP/MOVE FAILED ===" : "=== SWAP/MOVE OK ===");
    return fails ? 1 : 0;
}

} // namespace validation::swap_move_validation
