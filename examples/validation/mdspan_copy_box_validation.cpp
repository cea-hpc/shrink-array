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

// mdspan_copy_box_validation: md::copy_box_out and md::copy_box_in
// (chunk/mdspan_tile_ops.h) against the element path and a std::vector kept
// in step.
//
//   - out: a box of the view copied into a row-major buffer reads what f[i, j]
//     reads, for boxes aligned on the tiles, unaligned, on the padded edges,
//     one row, one column, the whole view, empty; tiled views (a tile per
//     chunk, layout_tiled_dyn, and 16 x 16, layout_tiled), row-major views,
//     rank 3, and with submdspan a sub-view and a column (last stride != 1);
//   - in: a buffer copied into those boxes -- uniform at first_value,
//     uniform at another value, varied -- leaves the view equal to the
//     reference everywhere (inside the box and out); a box of whole tiles
//     copied at first_value leaves its tiles compressed (every variant), at
//     another value too on zero_any::Chunk;
//   - parallel: both, tile by tile (tiled view) or row slice by row slice
//     (row-major view), from every thread of a phase;
//   - team (copy_box_out_mt / copy_box_in_mt, called by every thread): every
//     box, every pattern, on a tiled view (a tile per chunk), rank 3, a
//     row-major view, and with submdspan a sub-view and a column; called
//     outside a region too (the whole box). The deal itself: every tile of
//     the box to one thread, no thread above W / T plus one tile's cells; the
//     strided shares a partition of the box, their inner bounds on a chunk
//     boundary or a row's end.
//
// Exit 0, 1 with the first mismatches, 77 (ctest SKIP_RETURN_CODE) without
// mdspan. Run by ctest; the variant comes from the CHUNK_VARIANT_* macro.

#include <algorithm>   // std::min, std::max, std::fill, std::max_element
#include <array>       // std::array (the boxes)
#include <cstddef>     // std::size_t
#include <type_traits> // std::conditional_t
#include <utility>     // std::pair (submdspan), std::as_const
#include <vector>      // std::vector (the buffers, the reference, the deals)

#include "chunk/zero_array.h"     // zero::Array
#include "chunk/zero_one_value_array.h"    // zero_one_value::Array
#include "chunk/zero_any_array.h"   // zero_any::Array
#include "chunk/zero_chunk.h"       // release_chunk_pools, destroy_block_chunk
#include "chunk/config.h"          // chunk_config::use_zero_one_value*
#include "chunk/internal/print.h"  // shrink_array::println
#include "chunk/manage_ompt.h"     // parallel_phase
#include "chunk/mdspan_tile_ops.h" // md::copy_box_out(_mt), md::copy_box_in(_mt), SHRINK_ARRAY_HAS_MDSPAN
#include "chunk/PMP_CHUNK.h"       // CHUNK_PMP_THREAD_RANK, CHUNK_PMP_NUM_THREADS
#include "chunk/param_chunk.h"     // ParamChunk::init_indexes

using namespace shrink_array;

#if SHRINK_ARRAY_HAS_MDSPAN

namespace
{
template <typename T>
using Array =
    std::conditional_t<chunk_config::use_zero_any,
                       zero_any::Array<T>,
                       std::conditional_t<chunk_config::use_zero_one_value, zero_one_value::Array<T>, zero::Array<T>>>;

using Box = std::array<long, 2>;

int fails = 0;

void expect(bool ok, const char* what, long a = 0, long b = 0)
{
    if (!ok && ++fails <= 20)
    {
        println("mdspan_copy_box_validation: FAIL {} ({}, {})", what, a, b);
    }
}

constexpr long nx = 100; // not multiples of the tile: padded tiles on both edges
constexpr long ny = 90;

// 0 outside a few blocks (every variant compresses the untouched tiles).
double cell(long i, long j)
{
    if (i >= 20 && i < 45 && j >= 30 && j < 70)
    {
        return static_cast<double>(i * 1000 + j) + 0.25;
    }
    if (i >= 64 && i < 96 && j >= 0 && j < 48)
    {
        return 3.0;
    }
    return 0.0;
}

// The boxes of every case: aligned on 16 x 16 tiles, unaligned, the padded
// corner, one row, one column, the whole view, empty.
const std::vector<std::pair<Box, Box>> box_pairs = {
    {{16, 32}, {48, 64}}, {{5, 7}, {37, 83}},  {{80, 70}, {100, 90}}, {{99, 0}, {100, 90}},
    {{0, 89}, {100, 90}}, {{0, 0}, {100, 90}}, {{3, 3}, {3, 10}},
};

long box_size(const Box& lo, const Box& hi)
{
    return (hi[0] - lo[0]) * (hi[1] - lo[1]);
}

// The field of cell() written in a phase -- the end-of-phase pass then
// compresses its uniform tiles (0 on every variant, 3.0 on zero_any::Chunk) -- and the
// reference.
template <typename View>
void fill_from_cells(const View& f, std::vector<double>& ref)
{
    parallel_phase(
        [&]
        {
    #pragma omp for schedule(static)
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
        });
    for (long i = 0; i < nx; ++i)
    {
        for (long j = 0; j < ny; ++j)
        {
            ref[static_cast<std::size_t>(i * ny + j)] = cell(i, j);
        }
    }
}

template <typename View>
long mismatches_with(const View& f, const std::vector<double>& ref)
{
    long bad = 0;
    for (long i = 0; i < nx; ++i)
    {
        for (long j = 0; j < ny; ++j)
        {
            const double v = f[i, j];
            bad += v != ref[static_cast<std::size_t>(i * ny + j)] ? 1 : 0;
        }
    }
    return bad;
}

// The three source patterns copied in.
double pattern(int p, long i, long j)
{
    switch (p)
    {
        case 0:
            return 0.0; // first_value
        case 1:
            return 5.5; // another value
        default:
            return static_cast<double>(i * 7 + j) + 0.5; // varied
    }
}

// Every box out (against f[i, j]) then in (against the reference), serially.
template <typename View>
void out_and_in(const char* name, const View& f, std::vector<double>& ref)
{
    for (const auto& [lo, hi] : box_pairs)
    {
        const long          n = box_size(lo, hi);
        std::vector<double> buf(static_cast<std::size_t>(n), -7.0);
        md::copy_box_out(f, lo, hi, buf.data());
        long bad = 0;
        for (long i = lo[0]; i < hi[0]; ++i)
        {
            for (long j = lo[1]; j < hi[1]; ++j)
            {
                const double v = f[i, j];
                bad += buf[static_cast<std::size_t>((i - lo[0]) * (hi[1] - lo[1]) + (j - lo[1]))] != v ? 1 : 0;
            }
        }
        expect(bad == 0, name, lo[0] * 1000 + lo[1], bad);
    }
    for (int p = 0; p < 3; ++p)
    {
        for (const auto& [lo, hi] : box_pairs)
        {
            std::vector<double> src(static_cast<std::size_t>(box_size(lo, hi)));
            for (long i = lo[0]; i < hi[0]; ++i)
            {
                for (long j = lo[1]; j < hi[1]; ++j)
                {
                    const double v                                                             = pattern(p, i, j);
                    src[static_cast<std::size_t>((i - lo[0]) * (hi[1] - lo[1]) + (j - lo[1]))] = v;
                    ref[static_cast<std::size_t>(i * ny + j)]                                  = v;
                }
            }
            md::copy_box_in(f, lo, hi, src.data());
            const long bad = mismatches_with(f, ref);
            expect(bad == 0, name, p * 100000 + lo[0] * 1000 + lo[1], bad);
        }
    }
}

// Tiled views: a tile per chunk (layout_tiled_dyn, any chunk size) and 16 x 16.
void tiled_serial()
{
    {
        const long          S = static_cast<long>(Array<double>(1).chunk_size());
        Array<double>       a(static_cast<int>(
            md::tiled_size_dyn(md::tile_shape<2, long>(static_cast<std::size_t>(S)), std::array<long, 2>{nx, ny})));
        auto                f = md::tiled_view_dyn(a, nx, ny);
        std::vector<double> ref(static_cast<std::size_t>(nx * ny));
        fill_from_cells(f, ref);
        out_and_in("tiled (a tile per chunk)", f, ref);
    }
    {
        Array<double>       a(static_cast<int>(md::tiled_size<16, 16>(nx, ny)));
        auto                f = md::tiled_view<16, 16>(a, nx, ny);
        std::vector<double> ref(static_cast<std::size_t>(nx * ny));
        fill_from_cells(f, ref);
        out_and_in("tiled 16 x 16", f, ref);
    }
}

void row_major_serial()
{
    Array<double>       a(static_cast<int>(nx * ny));
    auto                f = md::view(a, nx, ny);
    std::vector<double> ref(static_cast<std::size_t>(nx * ny));
    fill_from_cells(f, ref);
    out_and_in("row-major", f, ref);
}

// A box of whole tiles copied in uniform: its tiles hold no buffer after --
// at first_value on every variant, at another value on zero_any::Chunk.
void compression()
{
    const long S    = static_cast<long>(Array<double>(1).chunk_size());
    const auto tile = md::tile_shape<2, long>(static_cast<std::size_t>(S));
    for (const double v : {0.0, 5.5})
    {
        Array<double> a(static_cast<int>(md::tiled_size_dyn(tile, std::array<long, 2>{nx, ny})));
        auto          f = md::tiled_view_dyn(a, nx, ny);
        for (long i = 0; i < nx; ++i)
        {
            for (long j = 0; j < ny; ++j)
            {
                f[i, j] = static_cast<double>(i + j) + 1.0; // every tile materialised
            }
        }
        const Box           lo{tile[0], tile[1]};
        const Box           hi{3 * tile[0], 3 * tile[1]}; // 2 x 2 whole tiles
        std::vector<double> src(static_cast<std::size_t>(box_size(lo, hi)), v);
        md::copy_box_in(f, lo, hi, src.data());
        const long per_row = (ny + tile[1] - 1) / tile[1];
        long       mat     = 0;
        for (long ti = 1; ti < 3; ++ti)
        {
            for (long tj = 1; tj < 3; ++tj)
            {
                mat += a.written(static_cast<int>((ti * per_row + tj) * S)) ? 1 : 0;
            }
        }
        const bool want_compressed = v == 0.0 || chunk_config::use_zero_any;
        expect(!want_compressed || mat == 0, "copy_box_in of a uniform box: whole tiles compressed", mat,
               static_cast<long>(v * 10));
        double got = f[tile[0] + 1, tile[1] + 1];
        expect(got == v, "copy_box_in of a uniform box: value", static_cast<long>(got * 10), static_cast<long>(v * 10));
    }
}

// Rank 3, a tile per chunk.
void rank3()
{
    const long                S    = static_cast<long>(Array<double>(1).chunk_size());
    const auto                tile = md::tile_shape<3, long>(static_cast<std::size_t>(S));
    const long                n0 = 10, n1 = 12, n2 = 20;
    Array<double>             a(static_cast<int>(md::tiled_size_dyn(tile, std::array<long, 3>{n0, n1, n2})));
    auto                      f = md::tiled_view_dyn(a, tile, n0, n1, n2);
    std::vector<double>       ref(static_cast<std::size_t>(n0 * n1 * n2), 0.0);
    const std::array<long, 3> lo{1, 2, 3};
    const std::array<long, 3> hi{9, 11, 19};
    std::vector<double>       src(static_cast<std::size_t>(8 * 9 * 16));
    std::size_t               k = 0;
    for (long i = lo[0]; i < hi[0]; ++i)
    {
        for (long j = lo[1]; j < hi[1]; ++j)
        {
            for (long l = lo[2]; l < hi[2]; ++l, ++k)
            {
                src[k]                                               = static_cast<double>(i * 10000 + j * 100 + l);
                ref[static_cast<std::size_t>((i * n1 + j) * n2 + l)] = src[k];
            }
        }
    }
    md::copy_box_in(f, lo, hi, src.data());
    long bad = 0;
    for (long i = 0; i < n0; ++i)
    {
        for (long j = 0; j < n1; ++j)
        {
            for (long l = 0; l < n2; ++l)
            {
                const double v = f[i, j, l];
                bad += v != ref[static_cast<std::size_t>((i * n1 + j) * n2 + l)] ? 1 : 0;
            }
        }
    }
    expect(bad == 0, "rank 3 copy_box_in", bad);
    std::vector<double> out(src.size(), -1.0);
    md::copy_box_out(f, lo, hi, out.data());
    expect(out == src, "rank 3 copy_box_out");
}

// A sub-view and a column (last stride != 1), with submdspan.
void sub_views()
{
    #if SHRINK_ARRAY_HAS_SUBMDSPAN
    Array<double>       a(static_cast<int>(nx * ny));
    auto                f = md::view(a, nx, ny);
    std::vector<double> ref(static_cast<std::size_t>(nx * ny));
    fill_from_cells(f, ref);

    auto                sub = md::submdspan(f, std::pair{10, 60}, std::pair{5, 80});
    const Box           lo{2, 3};
    const Box           hi{40, 70};
    std::vector<double> buf(static_cast<std::size_t>(box_size(lo, hi)));
    md::copy_box_out(sub, lo, hi, buf.data());
    long bad = 0;
    for (long i = lo[0]; i < hi[0]; ++i)
    {
        for (long j = lo[1]; j < hi[1]; ++j)
        {
            const double v = sub[i, j];
            bad += buf[static_cast<std::size_t>((i - lo[0]) * (hi[1] - lo[1]) + (j - lo[1]))] != v ? 1 : 0;
        }
    }
    expect(bad == 0, "submdspan copy_box_out", bad);

    auto                      col = md::submdspan(f, md::full_extent, 7); // stride ny
    const std::array<long, 1> clo{3};
    const std::array<long, 1> chi{97};
    std::vector<double>       cbuf(94);
    for (std::size_t k = 0; k < cbuf.size(); ++k)
    {
        cbuf[k]                                                            = 100.0 + static_cast<double>(k);
        ref[static_cast<std::size_t>((3 + static_cast<long>(k)) * ny + 7)] = cbuf[k];
    }
    md::copy_box_in(col, clo, chi, cbuf.data());
    expect(mismatches_with(f, ref) == 0, "column copy_box_in", mismatches_with(f, ref));
    std::vector<double> cout(94, -1.0);
    md::copy_box_out(col, clo, chi, cout.data());
    expect(cout == cbuf, "column copy_box_out");
    #endif
}

// From every thread of a phase: tile by tile (tiled), row slices (row-major).
void parallel()
{
    const long S    = static_cast<long>(Array<double>(1).chunk_size());
    const auto tile = md::tile_shape<2, long>(static_cast<std::size_t>(S));
    const Box  lo{5, 7};
    const Box  hi{97, 88};
    const long w = hi[1] - lo[1];
    const long n = box_size(lo, hi);

    std::vector<double> src(static_cast<std::size_t>(n));
    for (long i = lo[0]; i < hi[0]; ++i)
    {
        for (long j = lo[1]; j < hi[1]; ++j)
        {
            // uniform where whole tiles fit, varied elsewhere
            src[static_cast<std::size_t>((i - lo[0]) * w + (j - lo[1]))] =
                (i >= 32 && i < 64) ? 4.0 : static_cast<double>(i * 3 + j) + 0.5;
        }
    }
    {
        Array<double>       a(static_cast<int>(md::tiled_size_dyn(tile, std::array<long, 2>{nx, ny})));
        auto                f = md::tiled_view_dyn(a, nx, ny);
        std::vector<double> ref(static_cast<std::size_t>(nx * ny));
        fill_from_cells(f, ref);
        const auto nt = static_cast<long>(md::tile_count(f));
        parallel_phase(
            [&]
            {
    #pragma omp for schedule(static)
                for (long k = 0; k < nt; ++k)
                {
                    md::copy_box_in(f, lo, hi, src.data(), static_cast<int>(k), static_cast<int>(k + 1));
                }
            });
        for (long i = lo[0]; i < hi[0]; ++i)
        {
            for (long j = lo[1]; j < hi[1]; ++j)
            {
                ref[static_cast<std::size_t>(i * ny + j)] =
                    src[static_cast<std::size_t>((i - lo[0]) * w + (j - lo[1]))];
            }
        }
        expect(mismatches_with(f, ref) == 0, "parallel tiled copy_box_in", mismatches_with(f, ref));
        std::vector<double> out(static_cast<std::size_t>(n), -1.0);
        const auto          cf = md::as_const(f);
    #pragma omp parallel for schedule(static)
        for (long k = 0; k < nt; ++k)
        {
            md::copy_box_out(cf, lo, hi, out.data(), static_cast<int>(k), static_cast<int>(k + 1));
        }
        expect(out == src, "parallel tiled copy_box_out");
    }
    {
        Array<double>       a(static_cast<int>(nx * ny));
        auto                f = md::view(a, nx, ny);
        std::vector<double> ref(static_cast<std::size_t>(nx * ny));
        fill_from_cells(f, ref);
        parallel_phase(
            [&]
            {
                const long nth = CHUNK_PMP_NUM_THREADS;
                const long me  = CHUNK_PMP_THREAD_RANK;
                const long r0  = lo[0] + (hi[0] - lo[0]) * me / nth;
                const long r1  = lo[0] + (hi[0] - lo[0]) * (me + 1) / nth;
                if (r0 < r1)
                {
                    md::copy_box_in(f, Box{r0, lo[1]}, Box{r1, hi[1]}, src.data() + (r0 - lo[0]) * w);
                }
            });
        for (long i = lo[0]; i < hi[0]; ++i)
        {
            for (long j = lo[1]; j < hi[1]; ++j)
            {
                ref[static_cast<std::size_t>(i * ny + j)] =
                    src[static_cast<std::size_t>((i - lo[0]) * w + (j - lo[1]))];
            }
        }
        expect(mismatches_with(f, ref) == 0, "parallel row-major copy_box_in", mismatches_with(f, ref));
        std::vector<double> out(static_cast<std::size_t>(n), -1.0);
        const auto          cf = md::view(std::as_const(a), nx, ny);
    #pragma omp parallel
        {
            const long nth = CHUNK_PMP_NUM_THREADS;
            const long me  = CHUNK_PMP_THREAD_RANK;
            const long r0  = lo[0] + (hi[0] - lo[0]) * me / nth;
            const long r1  = lo[0] + (hi[0] - lo[0]) * (me + 1) / nth;
            if (r0 < r1)
            {
                md::copy_box_out(cf, Box{r0, lo[1]}, Box{r1, hi[1]}, out.data() + (r0 - lo[0]) * w);
            }
        }
        expect(out == src, "parallel row-major copy_box_out");
    }
}

// Every box out then in, the _mt forms from every thread of a region (in: in
// a phase), and once outside any region.
template <typename View>
void out_and_in_mt(const char* name, const View& f, std::vector<double>& ref)
{
    for (const auto& box : box_pairs)
    {
        const Box&          lo = box.first; // not a structured binding: Clang cannot capture one in a region
        const Box&          hi = box.second;
        const long          n  = box_size(lo, hi);
        std::vector<double> buf(static_cast<std::size_t>(n), -7.0);
    #pragma omp parallel
        md::copy_box_out_mt(f, lo, hi, buf.data());
        std::vector<double> alone(static_cast<std::size_t>(n), -9.0);
        md::copy_box_out_mt(f, lo, hi, alone.data()); // outside a region: the whole box
        long bad = 0;
        for (long i = lo[0]; i < hi[0]; ++i)
        {
            for (long j = lo[1]; j < hi[1]; ++j)
            {
                const double      v = f[i, j];
                const std::size_t o = static_cast<std::size_t>((i - lo[0]) * (hi[1] - lo[1]) + (j - lo[1]));
                bad += (buf[o] != v ? 1 : 0) + (alone[o] != v ? 1 : 0);
            }
        }
        expect(bad == 0, name, lo[0] * 1000 + lo[1], bad);
    }
    for (int p = 0; p < 3; ++p)
    {
        for (const auto& [lo, hi] : box_pairs)
        {
            std::vector<double> src(static_cast<std::size_t>(box_size(lo, hi)));
            for (long i = lo[0]; i < hi[0]; ++i)
            {
                for (long j = lo[1]; j < hi[1]; ++j)
                {
                    const double v                                                             = pattern(p, i, j) + 1.0;
                    src[static_cast<std::size_t>((i - lo[0]) * (hi[1] - lo[1]) + (j - lo[1]))] = v;
                    ref[static_cast<std::size_t>(i * ny + j)]                                  = v;
                }
            }
            parallel_phase([&] { md::copy_box_in_mt(f, lo, hi, src.data()); });
            const long bad = mismatches_with(f, ref);
            expect(bad == 0, name, p * 100000 + lo[0] * 1000 + lo[1], bad);
        }
    }
}

// The deals: tiles (each tile of the box to one thread, loads bounded) and
// strided shares (a partition, inner bounds on chunk boundaries or row ends).
void deals()
{
    const long S    = static_cast<long>(Array<double>(1).chunk_size());
    const auto tile = md::tile_shape<2, long>(static_cast<std::size_t>(S));
    // A field of 12 x 9 tiles (padded on both edges), boxes off-centre.
    const long    NX = 12 * tile[0] + 3;
    const long    NY = 9 * tile[1] + 5;
    Array<double> a(static_cast<int>(md::tiled_size_dyn(tile, std::array<long, 2>{NX, NY})));
    auto          f                              = md::tiled_view_dyn(a, NX, NY);
    using Index                                  = decltype(f)::index_type;
    const std::vector<std::pair<Box, Box>> boxes = {
        {{0, 0}, {NX, NY}},
        {{tile[0] / 2, 1}, {NX / 3, NY - 2}},
        {{NX / 2, NY / 2}, {NX, NY}},
        {{3, 0}, {4, NY}},
    };
    const auto nt = static_cast<std::size_t>(md::tile_count(f));
    for (const auto& [blo, bhi] : boxes)
    {
        std::array<Index, 2> lo{static_cast<Index>(blo[0]), static_cast<Index>(blo[1])};
        std::array<Index, 2> hi{static_cast<Index>(bhi[0]), static_cast<Index>(bhi[1])};
        std::vector<int>     owner(nt, -1);
        std::vector<long>    cells;
        long                 heaviest = 0;
        int                  team     = 1;
        long                 twice    = 0;
    #pragma omp parallel
        {
            const int me = CHUNK_PMP_THREAD_RANK;
    #pragma omp single
            {
                team = CHUNK_PMP_NUM_THREADS;
                cells.assign(static_cast<std::size_t>(team), 0);
            }
            md::detail::for_each_dealt_tile(f, lo, hi,
                                            [&](Index k)
                                            {
                                                const auto t = md::detail::tile_at(f, md::detail::tiles_per_dim(f), k);
                                                long       w = 1;
                                                for (std::size_t r = 0; r < 2; ++r)
                                                {
                                                    w *= std::min<long>(hi[r], t.origin[r] + t.extent[r]) -
                                                         std::max<long>(lo[r], t.origin[r]);
                                                }
    #pragma omp critical
                                                {
                                                    twice += owner[static_cast<std::size_t>(k)] >= 0 ? 1 : 0;
                                                    owner[static_cast<std::size_t>(k)] = me;
                                                    cells[static_cast<std::size_t>(me)] += w;
                                                    heaviest = std::max(heaviest, w);
                                                }
                                            });
        }
        expect(twice == 0, "deal: a tile to two threads", twice);
        // every tile the box touches has an owner
        long missing = 0;
        md::detail::for_each_tile_of_box(f, lo, hi, Index{0}, static_cast<Index>(nt),
                                         [&](Index k) { missing += owner[static_cast<std::size_t>(k)] < 0 ? 1 : 0; });
        expect(missing == 0, "deal: a tile of the box to no thread", missing);
        const long W   = box_size(blo, bhi);
        const long top = *std::max_element(cells.begin(), cells.end());
        expect(top <= W / team + 1 + heaviest, "deal: a thread above W / T plus a tile", top, W / team);
    }

    // strided shares, computed for every rank of a would-be team of T
    Array<double> r(static_cast<int>(NX * NY));
    auto          v = md::view(r, NX, NY);
    using RIndex    = decltype(v)::index_type;
    for (const auto& [blo, bhi] : boxes)
    {
        std::array<RIndex, 2> lo{static_cast<RIndex>(blo[0]), static_cast<RIndex>(blo[1])};
        std::array<RIndex, 2> hi{static_cast<RIndex>(bhi[0]), static_cast<RIndex>(bhi[1])};
        const auto            W         = static_cast<std::size_t>(box_size(blo, bhi));
        std::size_t           prev_end  = 0;
        long                  gaps      = 0;
        long                  unaligned = 0;
        int                   team      = 1;
    #pragma omp parallel
        {
    #pragma omp single
            team = CHUNK_PMP_NUM_THREADS;
        }
        std::vector<std::pair<std::size_t, std::size_t>> share(static_cast<std::size_t>(team));
    #pragma omp parallel
        share[static_cast<std::size_t>(CHUNK_PMP_THREAD_RANK)] = md::detail::team_share(v)(lo, hi, true);
        const auto w                                           = static_cast<std::size_t>(hi[1] - lo[1]);
        for (const auto& [c0, c1] : share)
        {
            gaps += (c0 != prev_end || c1 < c0) ? 1 : 0;
            prev_end = c1;
            if (c0 != 0 && c0 != W && c0 % w != 0)
            {
                const auto idx = std::array<RIndex, 2>{static_cast<RIndex>(lo[0] + static_cast<RIndex>(c0 / w)),
                                                       static_cast<RIndex>(lo[1] + static_cast<RIndex>(c0 % w))};
                unaligned += md::detail::strided_at(v, idx) % static_cast<std::size_t>(S) != 0 ? 1 : 0;
            }
        }
        gaps += prev_end != W ? 1 : 0;
        expect(gaps == 0, "strided shares: not a partition of the box", gaps);
        expect(unaligned == 0, "strided shares: an inner bound inside a chunk and a row", unaligned);
    }
}

void team()
{
    deals();
    const long S    = static_cast<long>(Array<double>(1).chunk_size());
    const auto tile = md::tile_shape<2, long>(static_cast<std::size_t>(S));
    {
        Array<double>       a(static_cast<int>(md::tiled_size_dyn(tile, std::array<long, 2>{nx, ny})));
        auto                f = md::tiled_view_dyn(a, nx, ny);
        std::vector<double> ref(static_cast<std::size_t>(nx * ny));
        fill_from_cells(f, ref);
        out_and_in_mt("team tiled", f, ref);
    }
    {
        Array<double>       a(static_cast<int>(nx * ny));
        auto                f = md::view(a, nx, ny);
        std::vector<double> ref(static_cast<std::size_t>(nx * ny));
        fill_from_cells(f, ref);
        out_and_in_mt("team row-major", f, ref);
    }
    {
        // rank 3, a tile per chunk
        const auto                tile3 = md::tile_shape<3, long>(static_cast<std::size_t>(S));
        const long                n0 = 10, n1 = 12, n2 = 20;
        Array<double>             a(static_cast<int>(md::tiled_size_dyn(tile3, std::array<long, 3>{n0, n1, n2})));
        auto                      f = md::tiled_view_dyn(a, tile3, n0, n1, n2);
        const std::array<long, 3> lo{1, 2, 3};
        const std::array<long, 3> hi{9, 11, 19};
        std::vector<double>       src(static_cast<std::size_t>(8 * 9 * 16));
        for (std::size_t k = 0; k < src.size(); ++k)
        {
            src[k] = static_cast<double>(k % 37) + 0.5;
        }
        parallel_phase([&] { md::copy_box_in_mt(f, lo, hi, src.data()); });
        std::vector<double> out(src.size(), -1.0);
    #pragma omp parallel
        md::copy_box_out_mt(f, lo, hi, out.data());
        expect(out == src, "team rank 3");
    }
    #if SHRINK_ARRAY_HAS_SUBMDSPAN
    {
        Array<double>       a(static_cast<int>(nx * ny));
        auto                f = md::view(a, nx, ny);
        std::vector<double> ref(static_cast<std::size_t>(nx * ny));
        fill_from_cells(f, ref);
        auto                sub = md::submdspan(f, std::pair{10, 60}, std::pair{5, 80});
        const Box           lo{2, 3};
        const Box           hi{40, 70};
        std::vector<double> src(static_cast<std::size_t>(box_size(lo, hi)));
        for (long i = lo[0]; i < hi[0]; ++i)
        {
            for (long j = lo[1]; j < hi[1]; ++j)
            {
                const double v = static_cast<double>(i * 11 + j) + 0.75;
                src[static_cast<std::size_t>((i - lo[0]) * (hi[1] - lo[1]) + (j - lo[1]))] = v;
                ref[static_cast<std::size_t>((i + 10) * ny + j + 5)]                       = v;
            }
        }
        parallel_phase([&] { md::copy_box_in_mt(sub, lo, hi, src.data()); });
        expect(mismatches_with(f, ref) == 0, "team submdspan copy_box_in_mt", mismatches_with(f, ref));
        std::vector<double> out(src.size(), -1.0);
        #pragma omp parallel
        md::copy_box_out_mt(sub, lo, hi, out.data());
        expect(out == src, "team submdspan copy_box_out_mt");

        auto                      col = md::submdspan(f, md::full_extent, 7); // stride ny
        const std::array<long, 1> clo{3};
        const std::array<long, 1> chi{97};
        std::vector<double>       cbuf(94);
        for (std::size_t k = 0; k < cbuf.size(); ++k)
        {
            cbuf[k]                                                            = 200.0 + static_cast<double>(k);
            ref[static_cast<std::size_t>((3 + static_cast<long>(k)) * ny + 7)] = cbuf[k];
        }
        parallel_phase([&] { md::copy_box_in_mt(col, clo, chi, cbuf.data()); });
        expect(mismatches_with(f, ref) == 0, "team column copy_box_in_mt", mismatches_with(f, ref));
        std::vector<double> cout(94, -1.0);
        #pragma omp parallel
        md::copy_box_out_mt(col, clo, chi, cout.data());
        expect(cout == cbuf, "team column copy_box_out_mt");
    }
    #endif
}
} // namespace

int main()
{
    ParamChunk::init_indexes(true);
    tiled_serial();
    row_major_serial();
    compression();
    rank3();
    sub_views();
    parallel();
    team();
    release_chunk_pools();
    destroy_block_chunk();
    if (fails != 0)
    {
        println("mdspan_copy_box_validation: {} failure(s)", fails);
        return 1;
    }
    println("mdspan_copy_box_validation: OK ({})", md::mdspan_source);
    return 0;
}

#else

int main()
{
    println("mdspan_copy_box_validation: skipped (neither <mdspan> nor kokkos/mdspan on the include path)");
    return 77;
}

#endif
