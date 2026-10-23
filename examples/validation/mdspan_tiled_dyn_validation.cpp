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

// mdspan_tiled_dyn_validation: md::layout_tiled_dyn, the tile given at run time.
//
//   - tile_shape(): the shape of a chunk's worth of elements, 2D and 3D, for
//     the power-of-two and the multiple-of-64 chunk sizes (product exact).
//   - the run-time mapping equals layout_tiled's for the same tile, index by
//     index, padded or not, power of two or not (16 x 24), and so do their
//     sub-views (submdspan); a view of each reads what the other wrote.
//   - tiled_view_dyn(a, ext...) takes a tile of a.chunk_size() elements; the
//     tile operations on it (for_each_tile, update_tiles serially and in a
//     parallel phase, fill_box) give the element formula, and a uniform tile
//     is a compressed chunk (Chunk2A); an odd tile (3 x 5, more than 8 KiB of
//     elements) goes through the same operations.
//   - in a _dyn library: after a re-chunk the view still reads and writes the
//     right elements (nothing moves), serially.
//
// Exit 0, 1 with the first mismatches, 77 (ctest SKIP_RETURN_CODE) without an
// mdspan.

#include <array>       // std::array (tiles, extents, boxes)
#include <cstddef>     // std::size_t
#include <type_traits> // std::conditional_t
#include <utility>     // std::pair (the slices), std::as_const

#include "chunk/zero_array.h"     // zero::Array
#include "chunk/zero_one_value_array.h"    // zero_one_value::Array
#include "chunk/zero_any_array.h"   // zero_any::Array
#include "chunk/chunk_std.h"       // release_chunk_pools, destroy_block_chunk
#include "chunk/config.h"          // chunk_config::use_chunk2*, build_config::chunk_dynamic
#include "chunk/internal/print.h"  // shrink_array::println
#include "chunk/manage_ompt.h"     // parallel_phase
#include "chunk/mdspan_tile_ops.h" // md::layout_tiled_dyn, md::tiled_view_dyn, the tile operations
#include "chunk/param_chunk.h"     // ParamChunk::init_indexes

using namespace shrink_array;

#if SHRINK_ARRAY_HAS_MDSPAN

namespace
{
template <typename T>
using Array =
    std::conditional_t<chunk_config::use_chunk2a,
                       zero_any::Array<T>,
                       std::conditional_t<chunk_config::use_chunk2, zero_one_value::Array<T>, zero::Array<T>>>;

int fails = 0;

void expect(bool ok, const char* what, long a = 0, long b = 0)
{
    if (!ok && ++fails <= 20)
    {
        println("mdspan_tiled_dyn_validation: FAIL {} ({}, {})", what, a, b);
    }
}

double val(long i, long j)
{
    return static_cast<double>(i * 1000 + j) + 0.5;
}

// ── tile_shape ──────────────────────────────────────────────────────────────
void shapes()
{
    struct Case2
    {
        std::size_t n;
        std::size_t a, b;
    };
    for (const Case2 c : {Case2{64, 8, 8}, Case2{128, 8, 16}, Case2{192, 12, 16}, Case2{256, 16, 16},
                          Case2{320, 16, 20}, Case2{384, 16, 24}, Case2{512, 16, 32}, Case2{1024, 32, 32}})
    {
        const auto t = md::tile_shape<2>(c.n);
        expect(t[0] == c.a && t[1] == c.b, "tile_shape 2D", static_cast<long>(c.n), static_cast<long>(t[0]));
    }
    const auto t3 = md::tile_shape<3>(256);
    expect(t3[0] == 4 && t3[1] == 8 && t3[2] == 8, "tile_shape 3D 256", static_cast<long>(t3[0]),
           static_cast<long>(t3[1]));
    for (std::size_t n = 64; n <= 4096; n += 64)
    {
        const auto t = md::tile_shape<3>(n);
        expect(t[0] * t[1] * t[2] == n && t[0] <= t[1] && t[1] <= t[2], "tile_shape 3D exact", static_cast<long>(n));
    }
}

// ── The run-time mapping is layout_tiled's ─────────────────────────────────
template <std::size_t T0, std::size_t T1>
void same_mapping(long nx, long ny)
{
    using I  = int;
    using E  = md::mdns::dextents<I, 2>;
    using MS = md::layout_tiled<T0, T1>::template mapping<E>;
    using MD = md::layout_tiled_dyn<2>::mapping<E>;
    const MS ms(E(nx, ny));
    const MD md(E(nx, ny), {static_cast<I>(T0), static_cast<I>(T1)});
    expect(ms.required_span_size() == md.required_span_size(), "span", nx, ny);
    expect(ms.is_exhaustive() == md.is_exhaustive(), "exhaustive", nx, ny);
    for (long i = 0; i < nx; ++i)
    {
        for (long j = 0; j < ny; ++j)
        {
            expect(ms(i, j) == md(i, j), "index", i, j);
        }
    }

    // Writes through one view, read through the other.
    Array<double> a(static_cast<long>(md::tiled_size<T0, T1>(nx, ny)));
    auto          fs = md::tiled_view<T0, T1>(a, nx, ny);
    auto          fd = md::tiled_view_dyn(a, std::array<long, 2>{T0, T1}, nx, ny);
    for (long i = 0; i < nx; ++i)
    {
        for (long j = 0; j < ny; ++j)
        {
            fd[i, j] = val(i, j);
        }
    }
    for (long i = 0; i < nx; ++i)
    {
        for (long j = 0; j < ny; ++j)
        {
            expect(static_cast<double>(fs[i, j]) == val(i, j), "static reads dyn", i, j);
        }
    }

    #if SHRINK_ARRAY_HAS_SUBMDSPAN
    // Their sub-views, aligned on the tiles, ending at the extent.
    const long i0 = static_cast<long>(T0), j0 = static_cast<long>(2 * T1);
    auto       ss = md::submdspan(fs, std::pair{i0, nx}, std::pair{j0, ny});
    auto       sd = md::submdspan(fd, std::pair{i0, nx}, std::pair{j0, ny});
    expect(ss.data_handle().base == sd.data_handle().base, "sub base", static_cast<long>(sd.data_handle().base));
    for (long i = 0; i < nx - i0; ++i)
    {
        for (long j = 0; j < ny - j0; ++j)
        {
            expect(ss.mapping()(i, j) == sd.mapping()(i, j), "sub index", i, j);
        }
    }
    #endif
    a.clear();
}

// ── Tile operations on a run-time tile ─────────────────────────────────────
double cell(long i, long j, long t0, long t1)
{
    return (i / t0 + j / t1) % 3 == 0 ? 2.0 : val(i, j); // a third of the tiles uniform
}

template <typename View>
void tile_ops(Array<double>& a, const View& f, const char* name)
{
    const long nx = f.extent(0), ny = f.extent(1);
    const long t0 = f.mapping().tile_extent(0), t1 = f.mapping().tile_extent(1);
    for (long i = 0; i < nx; ++i)
    {
        for (long j = 0; j < ny; ++j)
        {
            f[i, j] = cell(i, j, t0, t1);
        }
    }
    // for_each_tile: the sum and the count of the elements.
    double sum = 0, want = 0;
    long   seen = 0;
    md::for_each_tile(
        f,
        [&](const auto& t, auto b)
        {
            for (long i = 0; i < t.extent[0]; ++i)
            {
                for (long j = 0; j < t.extent[1]; ++j)
                {
                    sum += b[i, j];
                    ++seen;
                }
            }
        },
        [&](const auto& t, double v)
        {
            sum += v * static_cast<double>(t.extent[0] * t.extent[1]);
            seen += static_cast<long>(t.extent[0] * t.extent[1]);
        });
    for (long i = 0; i < nx; ++i)
    {
        for (long j = 0; j < ny; ++j)
        {
            want += cell(i, j, t0, t1);
        }
    }
    expect(seen == nx * ny && sum == want, name, seen);

    // update_tiles serially (+1 everywhere), then in a parallel phase when a
    // tile is whole chunks (-1): back to cell().
    md::update_tiles(
        f,
        [&](const auto&, auto b)
        {
            for (long i = 0; i < t0; ++i)
            {
                for (long j = 0; j < t1; ++j)
                {
                    b[i, j] += 1.0;
                }
            }
        },
        [](const auto&, double& v)
        {
            v += 1.0;
            return true;
        });
    for (long i = 0; i < nx; ++i)
    {
        for (long j = 0; j < ny; ++j)
        {
            expect(static_cast<double>(f[i, j]) == cell(i, j, t0, t1) + 1.0, name, i, j);
        }
    }
    const auto minus_one = [&](const auto&, auto b)
    {
        for (long i = 0; i < t0; ++i)
        {
            for (long j = 0; j < t1; ++j)
            {
                b[i, j] -= 1.0;
            }
        }
    };
    if (f.mapping().tile_elements() % static_cast<long>(a.chunk_size()) == 0)
    {
        const long n = md::tile_count(f);
        parallel_phase(
            [&]
            {
    #pragma omp for schedule(dynamic, 1)
                for (long k = 0; k < n; ++k)
                {
                    md::update_tiles(f, k, k + 1, minus_one);
                }
            });
    }
    else
    {
        md::update_tiles(f, minus_one);
    }
    for (long i = 0; i < nx; ++i)
    {
        for (long j = 0; j < ny; ++j)
        {
            expect(static_cast<double>(f[i, j]) == cell(i, j, t0, t1), name, i, j);
        }
    }

    // fill_box, then the tile at (t0, t1) covered: a compressed chunk (Chunk2A,
    // a tile per chunk).
    md::fill_box(f, std::array<long, 2>{t0 / 2, 0}, std::array<long, 2>{3 * t0, 2 * t1 + 1}, 5.0);
    for (long i = 0; i < nx; ++i)
    {
        for (long j = 0; j < ny; ++j)
        {
            const bool in = i >= t0 / 2 && i < 3 * t0 && j < 2 * t1 + 1;
            expect(static_cast<double>(f[i, j]) == (in ? 5.0 : cell(i, j, t0, t1)), name, i, j);
        }
    }
    if constexpr (chunk_config::use_chunk2a)
    {
        if (f.mapping().tile_elements() == static_cast<long>(a.chunk_size()))
        {
            expect(!a.written(static_cast<int>(f.mapping()(t0, t1))), "covered tile compressed",
                   static_cast<long>(f.mapping()(t0, t1)));
        }
    }
}

void dyn_views()
{
    constexpr long nx = 70, ny = 90;
    {
        // A tile per chunk, whatever the chunk size.
        Array<double> probe(1);
        const auto    tile = md::tile_shape<2, long>(probe.chunk_size());
        Array<double> a(static_cast<long>(md::tiled_size_dyn(tile, std::array<long, 2>{nx, ny})));
        auto          f = md::tiled_view_dyn(a, nx, ny);
        expect(f.mapping().tile_elements() == static_cast<long>(a.chunk_size()), "a tile per chunk",
               static_cast<long>(f.mapping().tile_elements()));
        tile_ops(a, f, "tile ops, a tile per chunk");
        a.clear();
    }
    {
        // An odd tile, 3 x 5: not whole chunks (serial only), every edge padded.
        const std::array<long, 2> tile{3, 5};
        Array<double>             a(static_cast<long>(md::tiled_size_dyn(tile, std::array<long, 2>{nx, ny})));
        tile_ops(a, md::tiled_view_dyn(a, tile, nx, ny), "tile ops, 3 x 5");
        a.clear();
    }
    {
        // A tile of more than 8 KiB of doubles: the buffer goes to the heap.
        const std::array<long, 2> tile{40, 32};
        Array<double>             a(static_cast<long>(md::tiled_size_dyn(tile, std::array<long, 2>{nx, ny})));
        tile_ops(a, md::tiled_view_dyn(a, tile, nx, ny), "tile ops, 40 x 32");
        a.clear();
    }
}

// ── After a re-chunk (a _dyn library) ──────────────────────────────────────
void after_rechunk()
{
    if constexpr (build_config::chunk_dynamic)
    {
        constexpr long nx = 64, ny = 96;
        Array<double>  probe(1);
        const auto     tile = md::tile_shape<2, long>(probe.chunk_size());
        Array<double>  a(static_cast<long>(md::tiled_size_dyn(tile, std::array<long, 2>{nx, ny})));
        auto           f = md::tiled_view_dyn(a, nx, ny);
        for (long i = 0; i < nx; ++i)
        {
            for (long j = 0; j < ny; ++j)
            {
                f[i, j] = cell(i, j, tile[0], tile[1]);
            }
        }
        const unsigned old_size = a.chunk_size();
        a.rechunk_to(old_size * 2);
        expect(a.chunk_size() == old_size * 2, "rechunk", static_cast<long>(a.chunk_size()));
        // The view still reads the same elements; the tile is half a chunk now.
        for (long i = 0; i < nx; ++i)
        {
            for (long j = 0; j < ny; ++j)
            {
                expect(static_cast<double>(f[i, j]) == cell(i, j, tile[0], tile[1]), "after rechunk", i, j);
            }
        }
        md::update_tiles(
            f,
            [&](const auto&, auto b)
            {
                for (long i = 0; i < tile[0]; ++i)
                {
                    for (long j = 0; j < tile[1]; ++j)
                    {
                        b[i, j] *= 2.0;
                    }
                }
            },
            [](const auto&, double& v)
            {
                v *= 2.0;
                return true;
            });
        for (long i = 0; i < nx; ++i)
        {
            for (long j = 0; j < ny; ++j)
            {
                expect(static_cast<double>(f[i, j]) == 2.0 * cell(i, j, tile[0], tile[1]), "update after rechunk", i,
                       j);
            }
        }
        a.clear();
    }
}
} // namespace

int main()
{
    ParamChunk::init_indexes(true);
    shapes();
    same_mapping<16, 16>(64, 64);
    same_mapping<16, 16>(50, 70);
    same_mapping<16, 24>(50, 100);
    same_mapping<8, 8>(17, 33);
    dyn_views();
    after_rechunk();
    release_chunk_pools();
    destroy_block_chunk();
    if (fails != 0)
    {
        println("mdspan_tiled_dyn_validation: {} failure(s)", fails);
        return 1;
    }
    println("mdspan_tiled_dyn_validation: OK");
    return 0;
}

#else

int main()
{
    println("mdspan_tiled_dyn_validation: skipped (neither <mdspan> nor kokkos/mdspan on the include path)");
    return 77;
}

#endif
