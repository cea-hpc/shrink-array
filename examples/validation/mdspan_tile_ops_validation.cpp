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
//   - fill_box: a sequence of boxes (whole tiles, partial, clipped, empty) on
//     tiled views (tiles of one, a quarter, two chunks) and on row-major views
//     of rank 2 and 3, against a std::vector filled element by element;
//     serial, then per tile range / per row slice in a parallel phase.
//   - team (the _mt forms, called by every thread, no omp for):
//     update_tiles_mt / for_each_tile_mt over every tile and over the tiles
//     of a box (each such tile visited once, no other; the tiles of a thread
//     within one of the others' count), fill_box_mt over the boxes above on
//     tiled and row-major views, in a phase and outside any region.
//   - `misaligned` argument: tiles of a quarter chunk in a parallel region
//     must stop the program (ctest: PASS_REGULAR_EXPRESSION on the message).
//
// Exit 0, 1 with the first mismatches, 77 (ctest SKIP_RETURN_CODE) without
// mdspan. Run by ctest; the variant comes from the CHUNK_VARIANT_* macro.

#include <algorithm>   // std::min, std::max (the reference boxes), std::minmax_element
#include <array>       // std::array (the boxes)
#include <cstddef>     // std::size_t
#include <string_view> // std::string_view (the argument)
#include <type_traits> // std::conditional_t
#include <vector>      // std::vector (the reference of fill_box)

#include "chunk/zero_array.h"    // zero::Array
#include "chunk/zero_one_value_array.h"   // zero_one_value::Array
#include "chunk/zero_any_array.h"  // zero_any::Array
#include "chunk/zero_chunk.h"      // release_chunk_pools, destroy_block_chunk
#include "chunk/config.h"         // chunk_config::use_chunk2*
#include "chunk/internal/print.h" // shrink_array::println
#include "chunk/manage_ompt.h"    // parallel_phase
#include "chunk/mdspan_tile_ops.h" // md::for_each_tile(_mt), md::update_tiles(_mt), md::fill_box(_mt), SHRINK_ARRAY_HAS_MDSPAN
#include "chunk/param_chunk.h" // ParamChunk::init_indexes
#include "chunk/PMP_CHUNK.h"   // CHUNK_PMP_THREAD_RANK, CHUNK_PMP_NUM_THREADS

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
            md::as_const(f), [](const auto&, auto) {}, [&](const auto&, double v) { n += v == 0.0 ? 1 : 0; });
        return n;
    };
    long uniform_before = 0;
    md::for_each_tile(md::as_const(f), [](const auto&, auto) {}, [&](const auto&, double) { ++uniform_before; });
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
            md::as_const(f), 0, 1, [](const auto&, auto) {},
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

// The boxes of fill_box, applied in this order, each with its own value.
struct Box
{
    std::array<long, 2> lo;
    std::array<long, 2> hi;
    double              v;
};
const Box boxes[] = {
    {{0, 0}, {nx, ny}, 1.0},     // the whole view
    {{16, 16}, {48, 64}, 2.0},   // whole tiles of 16 x 16
    {{5, 7}, {37, 83}, 3.0},     // partial tiles on every side
    {{90, 80}, {200, 200}, 4.0}, // clipped: the last, padded tiles
    {{10, 10}, {10, 20}, 5.0},   // empty
    {{-5, -5}, {3, 3}, 6.0},     // clipped below
    {{0, 0}, {nx, 16}, 0.0},     // T{}: a column of tiles
    {{32, 0}, {33, ny}, 7.0},    // one row
    {{16, 16}, {48, 64}, 8.0},   // the same tiles again, another value
};

void apply_ref(std::vector<double>& ref, const Box& b)
{
    for (long i = std::max(0L, b.lo[0]); i < std::min(nx, b.hi[0]); ++i)
    {
        for (long j = std::max(0L, b.lo[1]); j < std::min(ny, b.hi[1]); ++j)
        {
            ref[static_cast<std::size_t>(i * ny + j)] = b.v;
        }
    }
}

template <typename View>
void compare(const View& f, const std::vector<double>& ref, const char* what, int step)
{
    for (long i = 0; i < nx; ++i)
    {
        for (long j = 0; j < ny; ++j)
        {
            const double x = f[i, j];
            expect(x == ref[static_cast<std::size_t>(i * ny + j)], what, step, i * ny + j);
        }
    }
}

template <std::size_t TH, std::size_t TW>
void fill_box_tiled(bool parallel)
{
    Array<double>       a(static_cast<long>(md::tiled_size<TH, TW>(nx, ny)));
    auto                f = md::tiled_view<TH, TW>(a, nx, ny);
    std::vector<double> ref(static_cast<std::size_t>(nx * ny), 0.0);
    int                 step = 0;
    for (const Box& b : boxes)
    {
        apply_ref(ref, b);
        if (parallel)
        {
            const long n = md::tile_count(f);
            parallel_phase(
                [&]
                {
    #pragma omp for schedule(static, 1)
                    for (long k = 0; k < n; ++k)
                    {
                        md::fill_box(f, b.lo, b.hi, b.v, k, k + 1);
                    }
                });
        }
        else
        {
            md::fill_box(f, b.lo, b.hi, b.v);
        }
        compare(f, ref, parallel ? "fill_box tiled, parallel" : "fill_box tiled", step++);
    }
}

void fill_box_row_major(bool parallel)
{
    Array<double>       a(nx * ny);
    auto                f = md::view(a, nx, ny);
    std::vector<double> ref(static_cast<std::size_t>(nx * ny), 0.0);
    int                 step = 0;
    for (const Box& b : boxes)
    {
        apply_ref(ref, b);
        if (parallel)
        {
            parallel_phase(
                [&]
                {
    #pragma omp for schedule(static, 3)
                    for (long i = 0; i < nx; ++i) // each thread its rows of the box
                    {
                        md::fill_box(f, std::array<long, 2>{std::max(b.lo[0], i), b.lo[1]},
                                     std::array<long, 2>{std::min(b.hi[0], i + 1), b.hi[1]}, b.v);
                    }
                });
        }
        else
        {
            md::fill_box(f, b.lo, b.hi, b.v);
        }
        compare(f, ref, parallel ? "fill_box row-major, parallel" : "fill_box row-major", step++);
    }

    // Rank 3: a box spanning whole rows of the last two dimensions is one run.
    constexpr long n0 = 6;
    constexpr long n1 = 10;
    constexpr long n2 = 30;
    Array<int>     c(n0 * n1 * n2);
    auto           g = md::view(c, n0, n1, n2);
    md::fill_box(g, std::array<long, 3>{1, 0, 0}, std::array<long, 3>{3, n1, n2}, 4);
    md::fill_box(g, std::array<long, 3>{2, 3, 5}, std::array<long, 3>{5, 7, 29}, 9);
    for (long i = 0; i < n0; ++i)
    {
        for (long j = 0; j < n1; ++j)
        {
            for (long k = 0; k < n2; ++k)
            {
                const int x    = g[i, j, k];
                int       want = (i >= 1 && i < 3) ? 4 : 0;
                if (i >= 2 && i < 5 && j >= 3 && j < 7 && k >= 5 && k < 29)
                {
                    want = 9;
                }
                expect(x == want, "fill_box rank 3", i * n1 * n2 + j * n2 + k, x);
            }
        }
    }
}

// The _mt forms: every thread calls them, they deal the tiles themselves.
void team_tiles()
{
    Array<double> a(static_cast<long>(md::tiled_size<16, 16>(nx, ny)));
    auto          f    = md::tiled_view<16, 16>(a, nx, ny);
    const long    n    = md::tile_count(f);
    const long    tw   = (ny + 15) / 16; // tiles per row of tiles
    const auto    k_of = [tw](const auto& t) { return t.origin[0] / 16 * tw + t.origin[1] / 16; };

    // update_tiles_mt over every tile: each written once.
    std::vector<int> visits(static_cast<std::size_t>(n), 0);
    parallel_phase(
        [&]
        {
            md::update_tiles_mt(f,
                                [&](const auto& t, auto b)
                                {
    #pragma omp atomic
                                    ++visits[static_cast<std::size_t>(k_of(t))];
                                    for (long i = 0; i < t.extent[0]; ++i)
                                    {
                                        for (long j = 0; j < t.extent[1]; ++j)
                                        {
                                            b[i, j] = cell(t.origin[0] + i, t.origin[1] + j);
                                        }
                                    }
                                });
        });
    long bad = 0;
    for (const int v : visits)
    {
        bad += v != 1 ? 1 : 0;
    }
    expect(bad == 0, "update_tiles_mt: a tile not written once", bad);
    for (long i = 0; i < nx; ++i)
    {
        for (long j = 0; j < ny; ++j)
        {
            const double x = f[i, j];
            expect(x == cell(i, j), "update_tiles_mt values", i, j);
        }
    }

    // for_each_tile_mt over every tile: each read once, the blocks right,
    // the tiles of a thread within one of the others' count.
    std::fill(visits.begin(), visits.end(), 0);
    std::vector<long> per_thread;
    long              wrong = 0;
    #pragma omp parallel
    {
    #pragma omp single
        per_thread.assign(static_cast<std::size_t>(CHUNK_PMP_NUM_THREADS), 0);
        const auto me    = static_cast<std::size_t>(CHUNK_PMP_THREAD_RANK);
        const auto check = [&](const auto& t, auto value_at)
        {
    #pragma omp atomic
            ++visits[static_cast<std::size_t>(k_of(t))];
            ++per_thread[me];
            for (long i = 0; i < t.extent[0]; ++i)
            {
                for (long j = 0; j < t.extent[1]; ++j)
                {
                    if (value_at(i, j) != cell(t.origin[0] + i, t.origin[1] + j))
                    {
    #pragma omp atomic
                        ++wrong;
                    }
                }
            }
        };
        md::for_each_tile_mt(
            f, [&](const auto& t, auto b) { check(t, [&](long i, long j) { return b[i, j]; }); },
            [&](const auto& t, double v) { check(t, [v](long, long) { return v; }); });
    }
    bad = 0;
    for (const int v : visits)
    {
        bad += v != 1 ? 1 : 0;
    }
    expect(bad == 0, "for_each_tile_mt: a tile not read once", bad);
    expect(wrong == 0, "for_each_tile_mt values", wrong);
    const auto [lo_n, hi_n] = std::minmax_element(per_thread.begin(), per_thread.end());
    expect(*hi_n - *lo_n <= 1, "for_each_tile_mt: tiles of a thread off by more than one", *lo_n, *hi_n);

    // Over a box: the tiles it touches, once each, no other; the callbacks
    // change only the box's cells.
    const std::array<long, 2> blo{21, 5};
    const std::array<long, 2> bhi{70, 66};
    const auto                touches = [&](long k)
    {
        const long ti = k / tw;
        const long tj = k % tw;
        return ti * 16 < bhi[0] && (ti + 1) * 16 > blo[0] && tj * 16 < bhi[1] && (tj + 1) * 16 > blo[1];
    };
    std::fill(visits.begin(), visits.end(), 0);
    parallel_phase(
        [&]
        {
            md::update_tiles_mt(f, blo, bhi,
                                [&](const auto& t, auto b)
                                {
    #pragma omp atomic
                                    ++visits[static_cast<std::size_t>(k_of(t))];
                                    for (long i = 0; i < t.extent[0]; ++i)
                                    {
                                        for (long j = 0; j < t.extent[1]; ++j)
                                        {
                                            const long gi = t.origin[0] + i;
                                            const long gj = t.origin[1] + j;
                                            if (gi >= blo[0] && gi < bhi[0] && gj >= blo[1] && gj < bhi[1])
                                            {
                                                b[i, j] = -b[i, j] - 1.0;
                                            }
                                        }
                                    }
                                });
        });
    bad = 0;
    for (long k = 0; k < n; ++k)
    {
        bad += visits[static_cast<std::size_t>(k)] != (touches(k) ? 1 : 0) ? 1 : 0;
    }
    expect(bad == 0, "update_tiles_mt box: a tile visited wrongly", bad);
    for (long i = 0; i < nx; ++i)
    {
        for (long j = 0; j < ny; ++j)
        {
            const bool   in   = i >= blo[0] && i < bhi[0] && j >= blo[1] && j < bhi[1];
            const double want = in ? -cell(i, j) - 1.0 : cell(i, j);
            const double x    = f[i, j];
            expect(x == want, "update_tiles_mt box values", i, j);
        }
    }
    std::fill(visits.begin(), visits.end(), 0);
    #pragma omp parallel
    md::for_each_tile_mt(
        f, blo, bhi,
        [&](const auto& t, auto)
        {
    #pragma omp atomic
            ++visits[static_cast<std::size_t>(k_of(t))];
        },
        [&](const auto& t, double)
        {
    #pragma omp atomic
            ++visits[static_cast<std::size_t>(k_of(t))];
        });
    bad = 0;
    for (long k = 0; k < n; ++k)
    {
        bad += visits[static_cast<std::size_t>(k)] != (touches(k) ? 1 : 0) ? 1 : 0;
    }
    expect(bad == 0, "for_each_tile_mt box: a tile visited wrongly", bad);
}

// fill_box_mt over the boxes, in a phase (and once outside any region).
template <std::size_t TH, std::size_t TW>
void fill_box_tiled_mt()
{
    Array<double>       a(static_cast<long>(md::tiled_size<TH, TW>(nx, ny)));
    auto                f = md::tiled_view<TH, TW>(a, nx, ny);
    std::vector<double> ref(static_cast<std::size_t>(nx * ny), 0.0);
    int                 step = 0;
    for (const Box& b : boxes)
    {
        apply_ref(ref, b);
        if (step == 2)
        {
            md::fill_box_mt(f, b.lo, b.hi, b.v); // outside a region: the whole box
        }
        else
        {
            parallel_phase([&] { md::fill_box_mt(f, b.lo, b.hi, b.v); });
        }
        compare(f, ref, "fill_box_mt tiled", step++);
    }
}

void fill_box_row_major_mt()
{
    Array<double>       a(nx * ny);
    auto                f = md::view(a, nx, ny);
    std::vector<double> ref(static_cast<std::size_t>(nx * ny), 0.0);
    int                 step = 0;
    for (const Box& b : boxes)
    {
        apply_ref(ref, b);
        parallel_phase([&] { md::fill_box_mt(f, b.lo, b.hi, b.v); });
        compare(f, ref, "fill_box_mt row-major", step++);
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
    fill_box_tiled<16, 16>(false);
    fill_box_tiled<8, 8>(false);
    fill_box_tiled<16, 32>(false);
    fill_box_tiled<16, 16>(true);
    fill_box_tiled<16, 32>(true);
    fill_box_row_major(false);
    fill_box_row_major(true);
    team_tiles();
    fill_box_tiled_mt<16, 16>();
    fill_box_tiled_mt<16, 32>();
    fill_box_row_major_mt();
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
