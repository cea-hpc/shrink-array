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

// mdspan_tile_ops_validation: md::for_each_tile and md::update_tiles
// (chunk/mdspan_tile_ops.h) against the element path, f[i, j].
//
//   - read: every block holds the tile's elements, t.origin / t.extent are
//     right on padded edges, a uniform tile goes to on_uniform (and to
//     on_block, filled, without it); tiles of one chunk, of a quarter chunk,
//     of two chunks.
//   - update: the result is the element formula; a uniform tile left
//     unchanged stays uniform (not materialised); on_uniform returning false
//     hands a filled block; a tile made uniform reads back uniform.
//   - parallel: update_tiles over disjoint tiles inside parallel_phase, at the
//     team size of OMP_NUM_THREADS.
//   - `misaligned` argument: tiles of a quarter chunk in a parallel region
//     must stop the program (ctest: PASS_REGULAR_EXPRESSION on the message).
//
// Exit 0, 1 with the first mismatches, 77 (ctest SKIP_RETURN_CODE) without
// mdspan. Run by ctest; the variant comes from the CHUNK_VARIANT_* macro.

#include <cstddef>     // std::size_t
#include <string_view> // std::string_view (the argument)
#include <type_traits> // std::conditional_t

#include "chunk/array_chunk.h"     // ArrayChunk
#include "chunk/array_chunk2.h"    // ArrayChunk2
#include "chunk/array_chunk2A.h"   // ArrayChunk2A
#include "chunk/chunk_std.h"       // release_chunk_pools, destroy_block_chunk
#include "chunk/config.h"          // chunk_config::use_chunk2*
#include "chunk/internal/print.h"  // shrink_array::println
#include "chunk/manage_ompt.h"     // parallel_phase
#include "chunk/mdspan_tile_ops.h" // md::for_each_tile, md::update_tiles, SHRINK_ARRAY_HAS_MDSPAN
#include "chunk/param_chunk.h"     // ParamChunk::init_indexes

using namespace shrink_array;

#if SHRINK_ARRAY_HAS_MDSPAN

namespace
{
template <typename T>
using Array = std::conditional_t<chunk_config::use_chunk2a,
                                 ArrayChunk2A<T>,
                                 std::conditional_t<chunk_config::use_chunk2, ArrayChunk2<T>, ArrayChunk<T>>>;

int fails = 0;

void expect(bool ok, const char* what, long a = 0, long b = 0)
{
    if (!ok && ++fails <= 20)
    {
        println("mdspan_tile_ops_validation: FAIL {} ({}, {})", what, a, b);
    }
}

// 0 outside a few blocks, so every variant compresses the untouched tiles
// (T{} is the first constant of all three).
double cell(long i, long j)
{
    if (i >= 20 && i < 45 && j >= 30 && j < 70)
    {
        return static_cast<double>(i * 1000 + j) + 0.25;
    }
    if (i >= 64 && i < 96 && j >= 0 && j < 48)
    {
        return 3.0; // whole tiles of 16 x 16 of one non-zero value
    }
    return 0.0;
}

constexpr long nx = 100; // not multiples of 16: padded tiles on both edges
constexpr long ny = 90;

template <std::size_t TH, std::size_t TW>
void read_tiles()
{
    Array<double> a(static_cast<long>(md::tiled_size<TH, TW>(nx, ny)));
    auto          f = md::tiled_view<TH, TW>(a, nx, ny);
    for (long i = 0; i < nx; ++i)
    {
        for (long j = 0; j < ny; ++j)
        {
            if (cell(i, j) != 0.0)
            {
                f[i, j] = cell(i, j);
            }
        }
    }
    const auto& cf = f;

    // Without on_uniform: every tile as a block, padding read as 0.
    long   tiles = 0;
    double sum   = 0.0;
    md::for_each_tile(cf,
                      [&](const auto& t, auto b)
                      {
                          ++tiles;
                          expect(t.extent[0] == std::min<long>(TH, nx - t.origin[0]) &&
                                     t.extent[1] == std::min<long>(TW, ny - t.origin[1]),
                                 "tile extent", t.origin[0], t.origin[1]);
                          for (long i = 0; i < static_cast<long>(TH); ++i)
                          {
                              for (long j = 0; j < static_cast<long>(TW); ++j)
                              {
                                  const bool   in   = i < t.extent[0] && j < t.extent[1];
                                  const double want = in ? cell(t.origin[0] + i, t.origin[1] + j) : 0.0;
                                  expect(b[i, j] == want, "block value", t.origin[0] + i, t.origin[1] + j);
                                  sum += in ? b[i, j] : 0.0;
                              }
                          }
                      });
    expect(tiles == md::tile_count(f), "every tile visited", tiles, md::tile_count(f));

    // With on_uniform: same sum, uniform tiles in O(1).
    double sum2     = 0.0;
    long   uniforms = 0;
    md::for_each_tile(
        cf,
        [&](const auto& t, auto b)
        {
            for (long i = 0; i < t.extent[0]; ++i)
            {
                for (long j = 0; j < t.extent[1]; ++j)
                {
                    sum2 += b[i, j];
                }
            }
        },
        [&](const auto& t, double v)
        {
            ++uniforms;
            sum2 += v * static_cast<double>(t.extent[0] * t.extent[1]);
        });
    double want = 0.0;
    for (long i = 0; i < nx; ++i)
    {
        for (long j = 0; j < ny; ++j)
        {
            want += cell(i, j);
        }
    }
    expect(sum == want && sum2 == want, "sum", static_cast<long>(sum), static_cast<long>(sum2));
    expect(uniforms > 0 && uniforms < tiles, "some tiles uniform", uniforms, tiles);
}

// f = 2 f + (f != 0): zero stays zero, so the background is never written.
void update_tiles_serial()
{
    Array<double> a(static_cast<long>(md::tiled_size<16, 16>(nx, ny)));
    auto          f = md::tiled_view<16, 16>(a, nx, ny);
    for (long i = 0; i < nx; ++i)
    {
        for (long j = 0; j < ny; ++j)
        {
            if (cell(i, j) != 0.0)
            {
                f[i, j] = cell(i, j);
            }
        }
    }
    // Tiles read as the constant 0 (compressed to T{}, every variant).
    auto count_zero = [&]
    {
        long n = 0;
        md::for_each_tile(
            std::as_const(f), [](const auto&, auto) {}, [&](const auto&, double v) { n += v == 0.0 ? 1 : 0; });
        return n;
    };
    long uniform_before = 0;
    md::for_each_tile(std::as_const(f), [](const auto&, auto) {}, [&](const auto&, double) { ++uniform_before; });
    const long zero_before = count_zero();

    long blocks = 0;
    md::update_tiles(
        f,
        [&](const auto&, auto b)
        {
            ++blocks;
            for (long i = 0; i < 16; ++i)
            {
                for (long j = 0; j < 16; ++j)
                {
                    b[i, j] = b[i, j] == 0.0 ? 0.0 : 2.0 * b[i, j] + 1.0;
                }
            }
        },
        [](const auto&, double& v)
        {
            v = v == 0.0 ? 0.0 : 2.0 * v + 1.0;
            return true;
        });
    for (long i = 0; i < nx; ++i)
    {
        for (long j = 0; j < ny; ++j)
        {
            const double c = cell(i, j);
            const double x = f[i, j];
            expect(x == (c == 0.0 ? 0.0 : 2.0 * c + 1.0), "update value", i, j);
        }
    }
    // The zero tiles were never written: still compressed.
    expect(zero_before > 0 && count_zero() == zero_before, "zero tiles stay compressed", count_zero(), zero_before);
    expect(blocks == md::tile_count(f) - uniform_before, "on_block only for mixed tiles", blocks);

    // on_uniform declining: the block arrives filled with the constant; a
    // gradient on the zero background materialises those tiles.
    md::update_tiles(
        f,
        [](const auto& t, auto b)
        {
            for (long i = 0; i < 16; ++i)
            {
                for (long j = 0; j < 16; ++j)
                {
                    if (t.origin[0] >= 96)
                    {
                        b[i, j] = b[i, j] + static_cast<double>(j);
                    }
                }
            }
        },
        [](const auto& t, double&) { return t.origin[0] < 96; }); // v unchanged: not written
    for (long j = 0; j < ny; ++j)
    {
        const double x = f[97, j];
        const double c = cell(97, j);
        expect(x == (c == 0.0 ? 0.0 : 2.0 * c + 1.0) + static_cast<double>(j % 16), "declined uniform", 97, j);
    }

    // A tile made uniform: written in one go (fill_range_mt), read back as
    // such; Chunk2A compresses it to that value at the next pass.
    md::update_tiles(f, 0, 1,
                     [](const auto&, auto b)
                     {
                         for (long i = 0; i < 16; ++i)
                         {
                             for (long j = 0; j < 16; ++j)
                             {
                                 b[i, j] = 5.0;
                             }
                         }
                     });
    for (long i = 0; i < 16; ++i)
    {
        for (long j = 0; j < 16; ++j)
        {
            const double x = f[i, j];
            expect(x == 5.0, "tile made uniform", i, j);
        }
    }
    if constexpr (chunk_config::use_chunk2a)
    {
        a.make_block_compression();
        long first_uniform = 0;
        md::for_each_tile(
            std::as_const(f), 0, 1, [](const auto&, auto) {},
            [&](const auto&, double v) { first_uniform += v == 5.0 ? 1 : 0; });
        expect(first_uniform == 1, "tile made uniform, compressed", first_uniform);
    }
}

// Disjoint tiles per thread, in a phase; the result equals the serial formula.
void update_tiles_parallel()
{
    Array<double> a(static_cast<long>(md::tiled_size<16, 16>(nx, ny)));
    auto          f = md::tiled_view<16, 16>(a, nx, ny);
    const long    n = md::tile_count(f);
    parallel_phase(
        [&]
        {
    #pragma omp for schedule(dynamic, 1)
            for (long k = 0; k < n; ++k)
            {
                md::update_tiles(f, k, k + 1,
                                 [](const auto& t, auto b)
                                 {
                                     for (long i = 0; i < t.extent[0]; ++i)
                                     {
                                         for (long j = 0; j < t.extent[1]; ++j)
                                         {
                                             b[i, j] = cell(t.origin[0] + i, t.origin[1] + j);
                                         }
                                     }
                                 });
            }
        });
    a.make_block_compression();
    parallel_phase(
        [&]
        {
    #pragma omp for schedule(static, 1)
            for (long k = 0; k < n; ++k)
            {
                md::update_tiles(
                    f, k, k + 1,
                    [](const auto&, auto b)
                    {
                        for (long i = 0; i < 16; ++i)
                        {
                            for (long j = 0; j < 16; ++j)
                            {
                                b[i, j] = -b[i, j];
                            }
                        }
                    },
                    [](const auto&, double& v)
                    {
                        v = -v;
                        return true;
                    });
            }
        });
    for (long i = 0; i < nx; ++i)
    {
        for (long j = 0; j < ny; ++j)
        {
            const double x = f[i, j];
            expect(x == -cell(i, j), "parallel update", i, j);
        }
    }
}

// Tiles of 64 elements on chunks of more: fine serially, refused in a region.
int misaligned()
{
    Array<double> a(static_cast<long>(md::tiled_size<8, 8>(nx, ny)));
    auto          f = md::tiled_view<8, 8>(a, nx, ny);
    if (a.chunk_size() <= 64)
    {
        println("mdspan_tile_ops_validation: chunk size {} <= 64, nothing to refuse", a.chunk_size());
        return 1;
    }
    parallel_phase([&] { md::update_tiles(f, 0, 1, [](const auto&, auto) {}); });
    println("mdspan_tile_ops_validation: a tile sharing a chunk was accepted in a parallel region");
    return 1;
}
} // namespace

int main(int argc, char** argv)
{
    ParamChunk::init_indexes(true);
    if (argc > 1 && std::string_view(argv[1]) == "misaligned")
    {
        return misaligned();
    }
    read_tiles<16, 16>(); // a tile per chunk at 256
    read_tiles<8, 8>();   // four tiles per chunk at 256
    read_tiles<16, 32>(); // two chunks per tile at 256
    update_tiles_serial();
    update_tiles_parallel();
    release_chunk_pools();
    destroy_block_chunk();
    if (fails != 0)
    {
        println("mdspan_tile_ops_validation: {} failure(s)", fails);
        return 1;
    }
    println("mdspan_tile_ops_validation: OK");
    return 0;
}

#else

int main()
{
    shrink_array::println(
        "mdspan_tile_ops_validation: skipped (neither <mdspan> nor kokkos/mdspan on the include path)");
    return 77;
}

#endif
