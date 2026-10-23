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

// mdspan_submdspan_validation: md::submdspan over the views of chunk/mdspan.h.
//
//   - row-major: a 2D sub-block reads and writes the parent's elements, its
//     data handle's base is the block's first element; a row and a column
//     (index slices: rank 1, the column with a stride of a row); a read-only
//     view of a sub-view; fill_box on the block (runs of a row) and on the
//     column (one element per run) leaves everything else alone.
//   - tiled: sub-views aligned on the tiles, one ending at the parent's
//     extent (its last tiles padded by the parent's padding), are tiled
//     again; f[i, j] of the sub-view is the parent's; tile_count, and
//     for_each_tile / update_tiles / fill_box on the sub-view touch exactly
//     its elements, serially and in a parallel phase (a tile per chunk); a
//     uniform tile of the sub-view is a compressed chunk; a sub-view of a
//     sub-view.
//   - `misaligned` argument: a tiled slice that does not start on a tile
//     stops the program (expect_stop.cmake).
//
// Exit 0, 1 with the first mismatches, 77 (ctest SKIP_RETURN_CODE) when no
// submdspan is available (neither kokkos/mdspan nor a C++26 <mdspan>).

#include <array>       // std::array (the boxes)
#include <cstddef>     // std::size_t
#include <string_view> // std::string_view (the argument)
#include <type_traits> // std::conditional_t, std::is_same_v
#include <utility>     // std::pair (the slices), std::as_const

#include "chunk/zero_array.h"     // zero::Array
#include "chunk/zero_one_value_array.h"    // zero_one_value::Array
#include "chunk/zero_any_array.h"   // zero_any::Array
#include "chunk/zero_chunk.h"       // release_chunk_pools, destroy_block_chunk
#include "chunk/config.h"          // chunk_config::use_zero_one_value*
#include "chunk/internal/print.h"  // shrink_array::println
#include "chunk/manage_ompt.h"     // parallel_phase
#include "chunk/mdspan_tile_ops.h" // md::submdspan, md::for_each_tile, md::update_tiles, md::fill_box
#include "chunk/param_chunk.h"     // ParamChunk::init_indexes

using namespace shrink_array;

#if SHRINK_ARRAY_HAS_MDSPAN && SHRINK_ARRAY_HAS_SUBMDSPAN

namespace
{
template <typename T>
using Array =
    std::conditional_t<chunk_config::use_zero_any,
                       zero_any::Array<T>,
                       std::conditional_t<chunk_config::use_zero_one_value, zero_one_value::Array<T>, zero::Array<T>>>;

int fails = 0;

void expect(bool ok, const char* what, long a = 0, long b = 0)
{
    if (!ok && ++fails <= 20)
    {
        println("mdspan_submdspan_validation: FAIL {} ({}, {})", what, a, b);
    }
}

double val(long i, long j)
{
    return static_cast<double>(i * 1000 + j) + 0.5;
}

using Range = std::pair<long, long>;

// ── Row-major ───────────────────────────────────────────────────────────────
void row_major()
{
    constexpr long nx = 37;
    constexpr long ny = 53;
    Array<double>  a(nx * ny);
    auto           f = md::view(a, nx, ny);
    for (long i = 0; i < nx; ++i)
    {
        for (long j = 0; j < ny; ++j)
        {
            f[i, j] = val(i, j);
        }
    }

    // A block.
    auto s = md::submdspan(f, Range{3, 20}, Range{5, 40});
    expect(s.extent(0) == 17 && s.extent(1) == 35, "block extents", static_cast<long>(s.extent(0)),
           static_cast<long>(s.extent(1)));
    expect(s.data_handle().base == 3 * ny + 5, "block base", static_cast<long>(s.data_handle().base));
    expect(&*s.data_handle() == &a, "block array");
    for (long i = 0; i < 17; ++i)
    {
        for (long j = 0; j < 35; ++j)
        {
            expect(static_cast<double>(s[i, j]) == val(i + 3, j + 5), "block read", i, j);
        }
    }
    s[2, 7] = -1.0;
    expect(std::as_const(a)[static_cast<int>(5 * ny + 12)] == -1.0, "block write");
    s[2, 7] = val(5, 12);

    // A read-only view of the block.
    const auto cs = md::submdspan(md::view(std::as_const(a), nx, ny), Range{3, 20}, Range{5, 40});
    expect(static_cast<double>(cs[16, 34]) == val(19, 39), "const block read");

    // A row and a column: index slices.
    auto row = md::submdspan(f, 7, md::full_extent);
    auto col = md::submdspan(f, md::full_extent, 11);
    expect(row.rank() == 1 && col.rank() == 1, "rank 1");
    expect(col.stride(0) == ny, "column stride", static_cast<long>(col.stride(0)));
    for (long j = 0; j < ny; ++j)
    {
        expect(static_cast<double>(row[j]) == val(7, j), "row read", j);
    }
    for (long i = 0; i < nx; ++i)
    {
        expect(static_cast<double>(col[i]) == val(i, 11), "column read", i);
    }

    // fill_box on the block (a run per row) and on the column (a run per element).
    md::fill_box(s, std::array<long, 2>{1, 2}, std::array<long, 2>{4, 30}, 7.0);
    md::fill_box(col, std::array<long, 1>{10}, std::array<long, 1>{20}, 8.0);
    for (long i = 0; i < nx; ++i)
    {
        for (long j = 0; j < ny; ++j)
        {
            double want = val(i, j);
            if (i >= 4 && i < 7 && j >= 7 && j < 35)
            {
                want = 7.0;
            }
            if (j == 11 && i >= 10 && i < 20)
            {
                want = 8.0;
            }
            expect(static_cast<double>(f[i, j]) == want, "fill_box on sub-views", i, j);
        }
    }
    a.clear();
}

// ── Tiled ───────────────────────────────────────────────────────────────────
constexpr long nx = 100; // 7 tiles of 16, the last one padded
constexpr long ny = 90;  // 6 tiles of 16, the last one padded

double cell(long i, long j)
{
    return (i / 16 + j / 16) % 3 == 0 ? 2.0 : val(i, j); // a third of the tiles uniform
}

void fill_parent(Array<double>& a)
{
    auto f = md::tiled_view<16, 16>(a, nx, ny);
    for (long i = 0; i < nx; ++i)
    {
        for (long j = 0; j < ny; ++j)
        {
            f[i, j] = cell(i, j);
        }
    }
}

template <typename Want>
void check_parent(const Array<double>& a, Want&& want, const char* what)
{
    const auto f = md::tiled_view<16, 16>(a, nx, ny);
    for (long i = 0; i < nx; ++i)
    {
        for (long j = 0; j < ny; ++j)
        {
            expect(static_cast<double>(f[i, j]) == want(i, j), what, i, j);
        }
    }
}

void tiled()
{
    Array<double> a(static_cast<long>(md::tiled_size<16, 16>(nx, ny)));
    fill_parent(a);
    auto f = md::tiled_view<16, 16>(a, nx, ny);

    // [16, 64) x [32, 90): ends on a tile in i, at the extent in j.
    constexpr long i0 = 16, i1 = 64, j0 = 32, j1 = 90;
    auto           s = md::submdspan(f, Range{i0, i1}, Range{j0, j1});
    static_assert(std::is_same_v<typename decltype(s)::layout_type, md::layout_tiled<16, 16>>,
                  "a tiled sub-view is tiled");
    expect(s.extent(0) == i1 - i0 && s.extent(1) == j1 - j0, "tiled sub extents");
    expect(md::tile_count(s) == 3 * 4, "tiled sub tile_count", static_cast<long>(md::tile_count(s)));
    for (long i = 0; i < i1 - i0; ++i)
    {
        for (long j = 0; j < j1 - j0; ++j)
        {
            expect(static_cast<double>(s[i, j]) == cell(i + i0, j + j0), "tiled sub read", i, j);
        }
    }
    const auto inside = [&](long i, long j) { return i >= i0 && i < i1 && j >= j0 && j < j1; };

    // for_each_tile: every element of the sub-view once, nothing else.
    double sum = 0, want = 0;
    long   seen = 0;
    md::for_each_tile(
        s,
        [&](const auto& t, auto b)
        {
            for (long i = 0; i < t.extent[0]; ++i)
            {
                for (long j = 0; j < t.extent[1]; ++j)
                {
                    sum += b[i, j];
                    ++seen;
                    expect(b[i, j] == cell(i0 + t.origin[0] + i, j0 + t.origin[1] + j), "for_each_tile block", i, j);
                }
            }
        },
        [&](const auto& t, double v)
        {
            sum += v * static_cast<double>(t.extent[0] * t.extent[1]);
            seen += static_cast<long>(t.extent[0] * t.extent[1]);
        });
    for (long i = i0; i < i1; ++i)
    {
        for (long j = j0; j < j1; ++j)
        {
            want += cell(i, j);
        }
    }
    expect(seen == (i1 - i0) * (j1 - j0), "for_each_tile count", seen);
    expect(sum == want, "for_each_tile sum");

    // update_tiles serially: +1 inside, the padding cells too (the parent's
    // padding past j = 90): the parent reads back +1 inside only.
    md::update_tiles(
        s,
        [](const auto&, auto b)
        {
            for (long i = 0; i < 16; ++i)
            {
                for (long j = 0; j < 16; ++j)
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
    check_parent(a, [&](long i, long j) { return cell(i, j) + (inside(i, j) ? 1.0 : 0.0); }, "update_tiles serial");

    // update_tiles in a parallel phase, a tile per thread: back to cell().
    const long n = md::tile_count(s);
    parallel_phase(
        [&]
        {
    #pragma omp for schedule(dynamic, 1)
            for (long k = 0; k < n; ++k)
            {
                md::update_tiles(s, k, k + 1,
                                 [](const auto&, auto b)
                                 {
                                     for (long i = 0; i < 16; ++i)
                                     {
                                         for (long j = 0; j < 16; ++j)
                                         {
                                             b[i, j] -= 1.0;
                                         }
                                     }
                                 });
            }
        });
    check_parent(a, [&](long i, long j) { return cell(i, j); }, "update_tiles parallel");

    // fill_box on the sub-view: whole tiles and crossed tiles.
    md::fill_box(s, std::array<long, 2>{8, 0}, std::array<long, 2>{40, 50}, 5.0);
    check_parent(
        a, [&](long i, long j) { return (i >= i0 + 8 && i < i0 + 40 && j >= j0 && j < j0 + 50) ? 5.0 : cell(i, j); },
        "fill_box tiled sub");

    // A whole tile of the sub-view at 5.0 is a compressed chunk (Chunk2A:
    // fill_range sets it without a buffer).
    if constexpr (chunk_config::use_zero_any)
    {
        const long k = static_cast<long>(s.data_handle().base) + static_cast<long>(s.mapping()(16, 16));
        expect(!a.written(static_cast<int>(k)), "covered tile compressed", k);
    }

    // A sub-view of the sub-view: [16, 48) x [16, 58) of s.
    auto ss = md::submdspan(s, Range{16, 48}, Range{16, 58});
    for (long i = 0; i < 32; ++i)
    {
        for (long j = 0; j < 42; ++j)
        {
            const double w = (i + i0 + 16 < i0 + 40 && j + j0 + 16 < j0 + 50) ? 5.0 : cell(i + i0 + 16, j + j0 + 16);
            expect(static_cast<double>(ss[i, j]) == w, "sub-sub read", i, j);
        }
    }
    a.clear();
}

// A slice that does not start on a tile: stops.
int misaligned()
{
    Array<double> a(static_cast<long>(md::tiled_size<16, 16>(nx, ny)));
    auto          f = md::tiled_view<16, 16>(a, nx, ny);
    auto          s = md::submdspan(f, Range{8, 40}, md::full_extent);
    println("mdspan_submdspan_validation: a slice off the tiles was accepted ({} x {})", s.extent(0), s.extent(1));
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
    row_major();
    tiled();
    release_chunk_pools();
    destroy_block_chunk();
    if (fails != 0)
    {
        println("mdspan_submdspan_validation: {} failure(s)", fails);
        return 1;
    }
    println("mdspan_submdspan_validation: OK");
    return 0;
}

#else

int main()
{
    println("mdspan_submdspan_validation: skipped (no submdspan: neither kokkos/mdspan nor a C++26 <mdspan>)");
    return 77;
}

#endif
