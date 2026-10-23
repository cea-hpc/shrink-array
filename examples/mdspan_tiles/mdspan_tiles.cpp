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

/**
 * @file mdspan_tiles.cpp
 * @brief One 2D field, two mdspan layouts over a chunk2A array of chunk size
 *        256: row-major (layout_right) and tiles of 16 x 16 (layout_tiled).
 *        Prints the chunks left materialised after the compression pass in
 *        each, and checks both read back the field.
 *
 * @details The field is a mesh of materials: a uniform background, rectangles
 *          and disks of one value each, and a noisy region (no compression
 *          there in either layout). In row-major a chunk is 256 consecutive
 *          cells of a row: it is uniform only where a material is at least
 *          256 cells wide. In tiles a chunk is a 16 x 16 square: uniform as
 *          soon as the square lies inside one material, whatever its width.
 *          Writes go through the view, in a parallel phase over the rows.
 *
 *          Then two kernels on the tiled field, element by element (f[i, j])
 *          against tile by tile (md::for_each_tile / md::update_tiles,
 *          chunk/mdspan_tile_ops.h), in parallel: the sum of the field, and
 *          f = 0.5 f + 1 on every cell. A uniform tile costs one call; the
 *          update writes it back as one constant. Prints the times and the
 *          chunks materialised, and checks both give the same field. Last,
 *          64 rectangles painted over the field, element by element against
 *          md::fill_box (each thread on its own slice of the tiles); and the
 *          price of the multi-index alone: a sum through a[k], f[i, j]
 *          row-major and f[i, j] in tiles. Between the two, a box of half the
 *          field each way copied out to a buffer and back in, element by
 *          element against md::copy_box_out / md::copy_box_in; then a band
 *          of the field, the tile range overloads in an omp for against
 *          md::copy_box_out_mt / md::copy_box_in_mt (the band's tiles dealt
 *          by weight), the band filled (fill_box_mt) and a sum over every
 *          tile (for_each_tile_mt); the rectangles are painted a third time
 *          with fill_box_mt, a barrier between two. Then one diffusion step,
 *          a 5-point stencil, element by element against
 *          md::stencil_tiles_mt, without and with its uniform shortcut.
 *
 *          Usage: mdspan_tiles [n]   (an n x n field, 2048 by default)
 *          Exit 0, 1 when a layout does not read back the field, 77 when
 *          mdspan is not available.
 */

#include <algorithm> // std::min
#include <array>     // std::array (the rectangles)
#include <chrono>    // std::chrono::steady_clock (the kernel times)
#include <cmath>     // std::sin, std::abs
#include <cstdlib>   // std::atol
#include <utility>   // std::as_const
#include <vector>    // std::vector (the box buffers)

#include "chunk/zero_any_array.h"  // zero_any::Array
#include "chunk/chunk_std.h"      // release_chunk_pools, destroy_block_chunk
#include "chunk/internal/print.h" // shrink_array::println
#include "chunk/manage_ompt.h"    // parallel_phase
#include "chunk/mdspan.h"         // shrink_array::md, SHRINK_ARRAY_HAS_MDSPAN
#include "chunk/PMP_CHUNK.h"      // CHUNK_PMP_THREAD_RANK, CHUNK_PMP_NUM_THREADS
#include "chunk/mdspan_tile_ops.h" // md::for_each_tile(_mt), md::update_tiles, md::fill_box(_mt), md::copy_box_*(_mt), md::stencil_tiles_mt
#include "chunk/param_chunk.h" // ParamChunk::init_indexes

namespace sa = shrink_array;

#if SHRINK_ARRAY_HAS_MDSPAN

namespace
{
// The material at (i, j) of an n x n field.
double cell(long i, long j, long n)
{
    const double x = static_cast<double>(i) / static_cast<double>(n);
    const double y = static_cast<double>(j) / static_cast<double>(n);
    if (x > 0.70 && y > 0.70) // noisy corner: never uniform
    {
        return std::sin(static_cast<double>(i * 7919 + j * 104729));
    }
    for (int d = 0; d < 6; ++d) // disks of 0.04 to 0.09 n of radius
    {
        const double cx = 0.15 + 0.11 * d;
        const double cy = 0.20 + 0.07 * ((d * 3) % 5);
        const double r  = 0.04 + 0.01 * d;
        if ((x - cx) * (x - cx) + (y - cy) * (y - cy) < r * r)
        {
            return 10.0 + d;
        }
    }
    for (int b = 0; b < 8; ++b) // narrow bands, 0.02 to 0.09 n wide
    {
        const double y0 = 0.05 + 0.08 * b;
        if (x > 0.80 - 0.05 * b && x < 0.95 && y > y0 && y < y0 + 0.02 + 0.01 * b)
        {
            return 20.0 + b;
        }
    }
    return 1.0; // background
}

struct Count
{
    long chunks;
    long materialised;
};

Count count(const sa::zero_any::Array<double>& a)
{
    Count c{static_cast<long>(a.nb_chunk_), 0};
    for (long k = 0; k < c.chunks; ++k)
    {
        const auto& ch = a.chunk_list_[k];
        if (!ch.is_cst_val1() && !ch.is_cst_val2())
        {
            ++c.materialised;
        }
    }
    return c;
}

// Fill the field through view f (any layout), compress, check, count.
template <typename View>
Count run(const char* name, sa::zero_any::Array<double>& a, View f, long n, int& bad)
{
    sa::parallel_phase(
        [&]
        {
    #pragma omp for schedule(static)
            for (long i = 0; i < n; ++i)
            {
                for (long j = 0; j < n; ++j)
                {
                    f[i, j] = cell(i, j, n);
                }
            }
        });
    sa::zero_any::Array<double>::make_block_compression();

    // Read back through a read-only view (values, no proxy), converted from f.
    const auto cf    = sa::md::as_const(f);
    long       wrong = 0;
    for (long i = 0; i < n; ++i)
    {
        for (long j = 0; j < n; ++j)
        {
            wrong += cf[i, j] != cell(i, j, n) ? 1 : 0;
        }
    }
    if (wrong != 0)
    {
        sa::println("{}: {} cells read back wrong", name, wrong);
        ++bad;
    }
    const Count c = count(a);
    sa::println("{:<26} {:>9} chunks  {:>9} materialised ({:5.1f} %)  ~{:7.1f} MiB of values", name, c.chunks,
                c.materialised, 100.0 * static_cast<double>(c.materialised) / static_cast<double>(c.chunks),
                static_cast<double>(c.materialised) * 256.0 * sizeof(double) / (1024.0 * 1024.0));
    return c;
}
double seconds_since(std::chrono::steady_clock::time_point t0)
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

// A tiled field, filled and compressed.
template <std::size_t TH, std::size_t TW>
void fill_tiled(sa::zero_any::Array<double>& t, long n)
{
    auto f = sa::md::tiled_view<TH, TW>(t, n, n);
    sa::parallel_phase(
        [&]
        {
    #pragma omp for schedule(static)
            for (long i = 0; i < n; ++i)
            {
                for (long j = 0; j < n; ++j)
                {
                    f[i, j] = cell(i, j, n);
                }
            }
        });
    sa::zero_any::Array<double>::make_block_compression();
}

// The sum and the update, element by element and tile by tile.
void kernels(long n, int& bad)
{
    constexpr int               reps = 5;
    const long                  size = static_cast<long>(sa::md::tiled_size<16, 16>(n, n));
    sa::zero_any::Array<double> e(size);
    sa::zero_any::Array<double> t(size);
    fill_tiled<16, 16>(e, n);
    fill_tiled<16, 16>(t, n);
    auto       fe     = sa::md::tiled_view<16, 16>(e, n, n);
    auto       ft     = sa::md::tiled_view<16, 16>(t, n, n);
    const auto ce     = sa::md::as_const(fe);
    const auto ct     = sa::md::as_const(ft);
    const long ntiles = sa::md::tile_count(ft);

    // The sum, best of reps.
    double s_elem = 0;
    double s_tile = 0;
    double t_elem = 1e30;
    double t_tile = 1e30;
    for (int r = 0; r < reps; ++r)
    {
        auto   t0 = std::chrono::steady_clock::now();
        double s  = 0;
    #pragma omp parallel for schedule(static) reduction(+ : s)
        for (long i = 0; i < n; ++i)
        {
            for (long j = 0; j < n; ++j)
            {
                s += ce[i, j];
            }
        }
        t_elem = std::min(t_elem, seconds_since(t0));
        s_elem = s;

        t0 = std::chrono::steady_clock::now();
        s  = 0;
    #pragma omp parallel for schedule(static) reduction(+ : s)
        for (long k = 0; k < ntiles; ++k)
        {
            sa::md::for_each_tile(
                ct, k, k + 1,
                [&](const auto& tl, auto b)
                {
                    for (long i = 0; i < tl.extent[0]; ++i)
                    {
                        for (long j = 0; j < tl.extent[1]; ++j)
                        {
                            s += b[i, j];
                        }
                    }
                },
                [&](const auto& tl, double v) { s += v * static_cast<double>(tl.extent[0] * tl.extent[1]); });
        }
        t_tile = std::min(t_tile, seconds_since(t0));
        s_tile = s;
    }
    // Summation orders differ: equal to a relative 1e-12.
    if (std::abs(s_elem - s_tile) > 1e-12 * std::abs(s_elem))
    {
        sa::println("sum: element {} against tile {}", s_elem, s_tile);
        ++bad;
    }
    sa::println("sum            element {:8.2f} ms   tile {:8.2f} ms   (x {:.1f})", 1e3 * t_elem, 1e3 * t_tile,
                t_elem / t_tile);

    // The update, once each; then the compression pass.
    auto t0 = std::chrono::steady_clock::now();
    sa::parallel_phase(
        [&]
        {
    #pragma omp for schedule(static)
            for (long i = 0; i < n; ++i)
            {
                for (long j = 0; j < n; ++j)
                {
                    fe[i, j] = 0.5 * ce[i, j] + 1.0;
                }
            }
        });
    const double u_elem = seconds_since(t0);
    const long   m_elem = count(e).materialised;
    t0                  = std::chrono::steady_clock::now();
    sa::parallel_phase(
        [&]
        {
    #pragma omp for schedule(static)
            for (long k = 0; k < ntiles; ++k)
            {
                sa::md::update_tiles(
                    ft, k, k + 1,
                    [](const auto&, auto b)
                    {
                        for (long i = 0; i < 16; ++i)
                        {
                            for (long j = 0; j < 16; ++j)
                            {
                                b[i, j] = 0.5 * b[i, j] + 1.0;
                            }
                        }
                    },
                    [](const auto&, double& v)
                    {
                        v = 0.5 * v + 1.0;
                        return true;
                    });
            }
        });
    const double u_tile = seconds_since(t0);
    const long   m_tile = count(t).materialised;
    t0                  = std::chrono::steady_clock::now();
    sa::zero_any::Array<double>::make_block_compression();
    const double c_both = seconds_since(t0);
    long         wrong  = 0;
    for (long i = 0; i < n; ++i)
    {
        for (long j = 0; j < n; ++j)
        {
            wrong += ce[i, j] != ct[i, j] ? 1 : 0;
        }
    }
    if (wrong != 0)
    {
        sa::println("update: {} cells differ between element and tile", wrong);
        ++bad;
    }
    sa::println("update         element {:8.2f} ms   tile {:8.2f} ms   (x {:.1f})", 1e3 * u_elem, 1e3 * u_tile,
                u_elem / u_tile);
    // The end of a phase already runs a compression pass: the explicit one
    // finds what it left.
    sa::println("  materialised at the end of the phase: element {}, tile {}; after one more pass: {} and {} "
                "({:.2f} ms)",
                m_elem, m_tile, count(e).materialised, count(t).materialised, 1e3 * c_both);
}

// 64 rectangles of one value each, painted in order (the later win).
void paint(long n, int& bad)
{
    struct Rect
    {
        std::array<long, 2> lo;
        std::array<long, 2> hi;
        double              v;
    };
    std::array<Rect, 64> rects{};
    unsigned long        seed = 12345;
    auto                 next = [&seed](long m)
    {
        seed = seed * 6364136223846793005UL + 1442695040888963407UL;
        return static_cast<long>((seed >> 33) % static_cast<unsigned long>(m));
    };
    for (std::size_t r = 0; r < rects.size(); ++r)
    {
        const long h = 8 + next(n / 4);
        const long w = 8 + next(n / 4);
        const long i = next(n - h);
        const long j = next(n - w);
        rects[r]     = Rect{{i, j}, {i + h, j + w}, 30.0 + static_cast<double>(r)};
    }

    const long                  size = static_cast<long>(sa::md::tiled_size<16, 16>(n, n));
    sa::zero_any::Array<double> e(size);
    sa::zero_any::Array<double> t(size);
    fill_tiled<16, 16>(e, n);
    fill_tiled<16, 16>(t, n);
    auto fe = sa::md::tiled_view<16, 16>(e, n, n);
    auto ft = sa::md::tiled_view<16, 16>(t, n, n);

    auto t0 = std::chrono::steady_clock::now();
    sa::parallel_phase(
        [&]
        {
    #pragma omp for schedule(static)
            for (long i = 0; i < n; ++i)
            {
                for (const Rect& r : rects)
                {
                    if (i >= r.lo[0] && i < r.hi[0])
                    {
                        for (long j = r.lo[1]; j < r.hi[1]; ++j)
                        {
                            fe[i, j] = r.v;
                        }
                    }
                }
            }
        });
    const double p_elem = seconds_since(t0);

    const long ntiles = sa::md::tile_count(ft);
    t0                = std::chrono::steady_clock::now();
    sa::parallel_phase(
        [&]
        {
            // Each thread one contiguous slice of the tiles, every rectangle in order.
            const long nth   = CHUNK_PMP_NUM_THREADS;
            const long me    = CHUNK_PMP_THREAD_RANK;
            const long first = ntiles * me / nth;
            const long last  = ntiles * (me + 1) / nth;
            for (const Rect& r : rects)
            {
                sa::md::fill_box(ft, r.lo, r.hi, r.v, first, last);
            }
        });
    const double p_tile = seconds_since(t0);

    // fill_box_mt: each rectangle's tiles dealt to the team, a barrier between
    // two rectangles (they overlap: the next one waits for this one).
    sa::zero_any::Array<double> m(size);
    fill_tiled<16, 16>(m, n);
    auto fm = sa::md::tiled_view<16, 16>(m, n, n);
    t0      = std::chrono::steady_clock::now();
    sa::parallel_phase(
        [&]
        {
            for (const Rect& r : rects)
            {
                sa::md::fill_box_mt(fm, r.lo, r.hi, r.v);
    #pragma omp barrier
            }
        });
    const double p_mt = seconds_since(t0);

    const auto ce    = sa::md::as_const(fe);
    const auto ct    = sa::md::as_const(ft);
    long       wrong = 0;
    for (long i = 0; i < n; ++i)
    {
        for (long j = 0; j < n; ++j)
        {
            wrong += ce[i, j] != ct[i, j] ? 1 : 0;
            wrong += sa::md::as_const(fm)[i, j] != ct[i, j] ? 1 : 0;
        }
    }
    if (wrong != 0)
    {
        sa::println("paint: {} cells differ between element, fill_box and fill_box_mt", wrong);
        ++bad;
    }
    const long m_elem = count(e).materialised;
    const long m_tile = count(t).materialised;
    sa::zero_any::Array<double>::make_block_compression();
    sa::println("paint 64 rect. element {:8.2f} ms   tile {:8.2f} ms   (x {:.1f})   fill_box_mt {:8.2f} ms",
                1e3 * p_elem, 1e3 * p_tile, p_elem / p_tile, 1e3 * p_mt);
    sa::println("  materialised at the end of the phase: element {}, tile {}; after one more pass: {} and {}", m_elem,
                m_tile, count(e).materialised, count(t).materialised);
}

// A box of half the field each way copied out to a contiguous buffer and back
// in, element by element (f[i, j], rows shared by the team) against
// md::copy_box_out / md::copy_box_in (tiles shared by the team), on the tiled
// field and on the row-major one (each thread its slice of rows).
void box_copy(long n, int& bad)
{
    const std::array<long, 2> lo{n / 4, n / 4};
    const std::array<long, 2> hi{3 * n / 4, 3 * n / 4};
    const long                w  = hi[1] - lo[1];
    const long                ne = (hi[0] - lo[0]) * w;
    std::vector<double>       e_out(static_cast<std::size_t>(ne));
    std::vector<double>       t_out(static_cast<std::size_t>(ne), -1.0);

    const long                  size = static_cast<long>(sa::md::tiled_size<16, 16>(n, n));
    sa::zero_any::Array<double> t(size);
    fill_tiled<16, 16>(t, n);
    auto       ft = sa::md::tiled_view<16, 16>(t, n, n);
    const auto ct = sa::md::as_const(ft);
    const long nt = sa::md::tile_count(ft);

    // Out.
    auto        t0 = std::chrono::steady_clock::now();
    #pragma omp parallel for schedule(static)
    for (long i = lo[0]; i < hi[0]; ++i)
    {
        for (long j = lo[1]; j < hi[1]; ++j)
        {
            e_out[static_cast<std::size_t>((i - lo[0]) * w + (j - lo[1]))] = ct[i, j];
        }
    }
    const double o_elem = seconds_since(t0);
    t0                  = std::chrono::steady_clock::now();
    #pragma omp parallel for schedule(static)
    for (long k = 0; k < nt; ++k)
    {
        sa::md::copy_box_out(ct, lo, hi, t_out.data(), static_cast<int>(k), static_cast<int>(k + 1));
    }
    const double o_tile = seconds_since(t0);
    if (t_out != e_out)
    {
        sa::println("box copy: copy_box_out differs from the element loop");
        ++bad;
    }

    // In: the buffer shifted by one, written back, element by element into one
    // field and by copy_box_in into another.
    std::vector<double> src(e_out);
    for (double& v : src)
    {
        v += 1.0;
    }
    sa::zero_any::Array<double> e(size);
    fill_tiled<16, 16>(e, n);
    auto fe = sa::md::tiled_view<16, 16>(e, n, n);
    t0      = std::chrono::steady_clock::now();
    sa::parallel_phase(
        [&]
        {
    #pragma omp for schedule(static)
            for (long i = lo[0]; i < hi[0]; ++i)
            {
                for (long j = lo[1]; j < hi[1]; ++j)
                {
                    fe[i, j] = src[static_cast<std::size_t>((i - lo[0]) * w + (j - lo[1]))];
                }
            }
        });
    const double i_elem = seconds_since(t0);
    t0                  = std::chrono::steady_clock::now();
    sa::parallel_phase(
        [&]
        {
    #pragma omp for schedule(static)
            for (long k = 0; k < nt; ++k)
            {
                sa::md::copy_box_in(ft, lo, hi, src.data(), static_cast<int>(k), static_cast<int>(k + 1));
            }
        });
    const double i_tile = seconds_since(t0);
    const auto   ce     = sa::md::as_const(fe);
    long         wrong  = 0;
    for (long i = 0; i < n; ++i)
    {
        for (long j = 0; j < n; ++j)
        {
            wrong += ce[i, j] != ct[i, j] ? 1 : 0;
        }
    }
    if (wrong != 0)
    {
        sa::println("box copy: {} cells differ between the element loop and copy_box_in", wrong);
        ++bad;
    }
    sa::println("box out, tiles element {:8.2f} ms   copy_box_out {:8.2f} ms   (x {:.1f})", 1e3 * o_elem, 1e3 * o_tile,
                o_elem / o_tile);
    sa::println("box in,  tiles element {:8.2f} ms   copy_box_in  {:8.2f} ms   (x {:.1f})", 1e3 * i_elem, 1e3 * i_tile,
                i_elem / i_tile);

    // Row-major: each thread its slice of the box's rows.
    sa::zero_any::Array<double> r(n * n);
    auto                        fr = sa::md::view(r, n, n);
    sa::parallel_phase(
        [&]
        {
    #pragma omp for schedule(static)
            for (long i = 0; i < n; ++i)
            {
                for (long j = 0; j < n; ++j)
                {
                    fr[i, j] = cell(i, j, n);
                }
            }
        });
    sa::zero_any::Array<double>::make_block_compression();
    const auto cr = sa::md::view(std::as_const(r), n, n);
    t0            = std::chrono::steady_clock::now();
    #pragma omp parallel
    {
        const long nth = CHUNK_PMP_NUM_THREADS;
        const long me  = CHUNK_PMP_THREAD_RANK;
        const long r0  = lo[0] + (hi[0] - lo[0]) * me / nth;
        const long r1  = lo[0] + (hi[0] - lo[0]) * (me + 1) / nth;
        if (r0 < r1)
        {
            sa::md::copy_box_out(cr, std::array<long, 2>{r0, lo[1]}, std::array<long, 2>{r1, hi[1]},
                                 t_out.data() + (r0 - lo[0]) * w);
        }
    }
    const double o_rows = seconds_since(t0);
    if (t_out != e_out)
    {
        sa::println("box copy: row-major copy_box_out differs from the tiled one");
        ++bad;
    }
    t0 = std::chrono::steady_clock::now();
    sa::parallel_phase(
        [&]
        {
            const long nth = CHUNK_PMP_NUM_THREADS;
            const long me  = CHUNK_PMP_THREAD_RANK;
            const long r0  = lo[0] + (hi[0] - lo[0]) * me / nth;
            const long r1  = lo[0] + (hi[0] - lo[0]) * (me + 1) / nth;
            if (r0 < r1)
            {
                sa::md::copy_box_in(fr, std::array<long, 2>{r0, lo[1]}, std::array<long, 2>{r1, hi[1]},
                                    src.data() + (r0 - lo[0]) * w);
            }
        });
    const double i_rows  = seconds_since(t0);
    long         wrong_r = 0;
    for (long i = lo[0]; i < hi[0]; ++i)
    {
        for (long j = lo[1]; j < hi[1]; ++j)
        {
            wrong_r += cr[i, j] != src[static_cast<std::size_t>((i - lo[0]) * w + (j - lo[1]))] ? 1 : 0;
        }
    }
    if (wrong_r != 0)
    {
        sa::println("box copy: row-major copy_box_in, {} cells wrong", wrong_r);
        ++bad;
    }
    sa::println("box out, row-major copy_box_out {:8.2f} ms; box in, row-major copy_box_in {:8.2f} ms", 1e3 * o_rows,
                1e3 * i_rows);
}

// A band of the field, rows [n / 16, 5 n / 16), every column: the omp for over
// [0, tile_count) of the range overloads gives its tiles to the threads that
// hold those rows of tiles (thread 0 alone at 2 threads), copy_box_*_mt deals
// the band's tiles by weight. Best of 5.
void box_copy_team(long n, int& bad)
{
    const std::array<long, 2> lo{n / 16, 0};
    const std::array<long, 2> hi{5 * n / 16, n};
    const long                ne = (hi[0] - lo[0]) * (hi[1] - lo[1]);
    std::vector<double>       range_out(static_cast<std::size_t>(ne), -1.0);
    std::vector<double>       team_out(static_cast<std::size_t>(ne), -2.0);

    const long                  size = static_cast<long>(sa::md::tiled_size<16, 16>(n, n));
    sa::zero_any::Array<double> t(size);
    fill_tiled<16, 16>(t, n);
    auto       ft = sa::md::tiled_view<16, 16>(t, n, n);
    const auto ct = sa::md::as_const(ft);
    const long nt = sa::md::tile_count(ft);

    const auto best = [](auto&& body)
    {
        double b = 1e30;
        for (int rep = 0; rep < 5; ++rep)
        {
            const auto t0 = std::chrono::steady_clock::now();
            body();
            b = std::min(b, seconds_since(t0));
        }
        return b;
    };
    const double o_range = best(
        [&]
        {
    #pragma omp  parallel for schedule(static)
            for (long k = 0; k < nt; ++k)
            {
                sa::md::copy_box_out(ct, lo, hi, range_out.data(), static_cast<int>(k), static_cast<int>(k + 1));
            }
        });
    const double o_team = best(
        [&]
        {
    #pragma omp parallel
            sa::md::copy_box_out_mt(ct, lo, hi, team_out.data());
        });
    if (team_out != range_out)
    {
        sa::println("box copy team: copy_box_out_mt differs from copy_box_out");
        ++bad;
    }
    std::vector<double> src(range_out);
    for (double& v : src)
    {
        v += 1.0;
    }
    const double i_range = best(
        [&]
        {
            sa::parallel_phase(
                [&]
                {
    #pragma omp for schedule(static)
                    for (long k = 0; k < nt; ++k)
                    {
                        sa::md::copy_box_in(ft, lo, hi, src.data(), static_cast<int>(k), static_cast<int>(k + 1));
                    }
                });
        });
    const double i_team = best([&] { sa::parallel_phase([&] { sa::md::copy_box_in_mt(ft, lo, hi, src.data()); }); });
    sa::md::copy_box_out(ct, lo, hi, team_out.data());
    if (team_out != src)
    {
        sa::println("box copy team: copy_box_in_mt did not write the band");
        ++bad;
    }
    sa::println("band out, omp for over the tiles {:8.2f} ms   copy_box_out_mt {:8.2f} ms   (x {:.1f})", 1e3 * o_range,
                1e3 * o_team, o_range / o_team);
    sa::println("band in,  omp for over the tiles {:8.2f} ms   copy_box_in_mt  {:8.2f} ms   (x {:.1f})", 1e3 * i_range,
                1e3 * i_team, i_range / i_team);

    // The band filled (a new value each time, so every tile changes), and a
    // sum over every tile of the field.
    int          rep_v   = 0;
    const double f_range = best(
        [&]
        {
            const double v = 100.0 + rep_v++;
            sa::parallel_phase(
                [&]
                {
    #pragma omp for schedule(static)
                    for (long k = 0; k < nt; ++k)
                    {
                        sa::md::fill_box(ft, lo, hi, v, static_cast<int>(k), static_cast<int>(k + 1));
                    }
                });
        });
    const double f_team = best(
        [&]
        {
            const double v = 100.0 + rep_v++;
            sa::parallel_phase([&] { sa::md::fill_box_mt(ft, lo, hi, v); });
        });
    if (ct[lo[0], lo[1]] != 100.0 + (rep_v - 1) || ct[hi[0] - 1, hi[1] - 1] != 100.0 + (rep_v - 1))
    {
        sa::println("box copy team: fill_box_mt did not fill the band");
        ++bad;
    }
    double     sum_range = 0;
    double     sum_team  = 0;
    const auto add_block = [](double& acc, const auto& t, auto b)
    {
        for (long i = 0; i < t.extent[0]; ++i)
        {
            for (long j = 0; j < t.extent[1]; ++j)
            {
                acc += b[i, j];
            }
        }
    };
    const auto add_uniform = [](double& acc, const auto& t, double v)
    { acc += v * static_cast<double>(t.extent[0] * t.extent[1]); };
    const double s_range = best(
        [&]
        {
            double acc = 0;
    #pragma omp  parallel for schedule(static) reduction(+ : acc)
            for (long k = 0; k < nt; ++k)
            {
                sa::md::for_each_tile(
                    ct, static_cast<int>(k), static_cast<int>(k + 1), [&](const auto& t, auto b)
                    { add_block(acc, t, b); }, [&](const auto& t, double v) { add_uniform(acc, t, v); });
            }
            sum_range = acc;
        });
    const double s_team = best(
        [&]
        {
            double acc = 0;
    #pragma omp parallel reduction(+ : acc)
            sa::md::for_each_tile_mt(
                ct, [&](const auto& t, auto b) { add_block(acc, t, b); },
                [&](const auto& t, double v) { add_uniform(acc, t, v); });
            sum_team = acc;
        });
    if (sum_range != sum_team && std::abs(sum_range - sum_team) > 1e-9 * std::abs(sum_range))
    {
        sa::println("box copy team: for_each_tile_mt sums {} against {}", sum_team, sum_range);
        ++bad;
    }
    sa::println("band fill, omp for over the tiles {:8.2f} ms   fill_box_mt      {:8.2f} ms   (x {:.1f})",
                1e3 * f_range, 1e3 * f_team, f_range / f_team);
    sa::println("sum, omp for, one tile per call   {:8.2f} ms   for_each_tile_mt {:8.2f} ms   (x {:.1f})",
                1e3 * s_range, 1e3 * s_team, s_range / s_team);
}
} // namespace

// One explicit diffusion step, a 5-point stencil with a clamped border, from
// the field into another array: element by element (f[i, j] and its four
// neighbours, rows shared by the team) against md::stencil_tiles_mt (a tile
// and its halo of 1 read in one go), without and with the uniform shortcut
// (a tile whose neighbourhood is one value stays that value, in O(1)). Best
// of 3.
void stencil(long n, int& bad)
{
    const long                  size = static_cast<long>(sa::md::tiled_size<16, 16>(n, n));
    sa::zero_any::Array<double> src(size);
    fill_tiled<16, 16>(src, n);
    const auto       cs = sa::md::as_const(sa::md::tiled_view<16, 16>(src, n, n));
    constexpr double nu = 0.2;

    const auto best = [](auto&& body)
    {
        double b = 1e30;
        for (int rep = 0; rep < 3; ++rep)
        {
            const auto t0 = std::chrono::steady_clock::now();
            body();
            b = std::min(b, seconds_since(t0));
        }
        return b;
    };

    sa::zero_any::Array<double> e(size);
    auto                        fe     = sa::md::tiled_view<16, 16>(e, n, n);
    const double                t_elem = best(
        [&]
        {
            sa::parallel_phase(
                [&]
                {
    #pragma omp for schedule(static)
                    for (long i = 0; i < n; ++i)
                    {
                        const long im = i > 0 ? i - 1 : 0;
                        const long ip = i + 1 < n ? i + 1 : n - 1;
                        for (long j = 0; j < n; ++j)
                        {
                            const long   jm = j > 0 ? j - 1 : 0;
                            const long   jp = j + 1 < n ? j + 1 : n - 1;
                            const double c  = cs[i, j];
                            fe[i, j]        = c + nu * (cs[im, j] + cs[ip, j] + cs[i, jm] + cs[i, jp] - 4.0 * c);
                        }
                    }
                });
        });

    const auto kernel = [](const auto& t, const auto& in, auto out)
    {
        for (long i = 0; i < t.extent[0]; ++i)
        {
            for (long j = 0; j < t.extent[1]; ++j)
            {
                const double c = in[i, j];
                out[i, j]      = c + nu * (in[i - 1, j] + in[i + 1, j] + in[i, j - 1] + in[i, j + 1] - 4.0 * c);
            }
        }
    };
    sa::zero_any::Array<double> b(size);
    auto                        fb     = sa::md::tiled_view<16, 16>(b, n, n);
    const double                t_tile = best(
        [&] { sa::parallel_phase([&] { sa::md::stencil_tiles_mt(cs, fb, 1L, sa::md::clamp_boundary{}, kernel); }); });
    sa::zero_any::Array<double> u(size);
    auto                        fu     = sa::md::tiled_view<16, 16>(u, n, n);
    const double                t_unif = best(
        [&]
        {
            sa::parallel_phase(
                [&]
                {
                    sa::md::stencil_tiles_mt(cs, fu, 1L, sa::md::clamp_boundary{}, kernel,
                                             [](const auto&, const double& v, double& out)
                                             {
                                                 out = v;
                                                 return true;
                                             });
                });
        });

    long wrong = 0;
    for (long i = 0; i < n; ++i)
    {
        for (long j = 0; j < n; ++j)
        {
            const double x = sa::md::as_const(fe)[i, j];
            wrong +=
                (std::abs(sa::md::as_const(fb)[i, j] - x) > 1e-12 || std::abs(sa::md::as_const(fu)[i, j] - x) > 1e-12)
                    ? 1
                    : 0;
        }
    }
    if (wrong != 0)
    {
        sa::println("stencil: {} cells differ between the element loop and stencil_tiles_mt", wrong);
        ++bad;
    }
    sa::println(
        "stencil 5 pts, element {:8.2f} ms   stencil_tiles_mt {:8.2f} ms (x {:.1f})   with on_uniform {:8.2f} ms "
        "(x {:.1f})",
        1e3 * t_elem, 1e3 * t_tile, t_elem / t_tile, 1e3 * t_unif, t_elem / t_unif);
    sa::println("  materialised: element {}, stencil_tiles_mt {}, with on_uniform {}", count(e).materialised,
                count(b).materialised, count(u).materialised);
}

// The price of the multi-index: one sum of the same field read as a[k], as
// f[i, j] row-major and as f[i, j] in tiles, element by element (the const
// operator[] every time), best of 5. The tiled mapping divides by constant
// tile extents: shifts and masks, no division.
void index_cost(long n, int& bad)
{
    sa::zero_any::Array<double> a(n * n);
    sa::zero_any::Array<double> t(static_cast<long>(sa::md::tiled_size<16, 16>(n, n)));
    fill_tiled<16, 16>(t, n);
    sa::parallel_phase(
        [&]
        {
    #pragma omp for schedule(static)
            for (long i = 0; i < n; ++i)
            {
                for (long j = 0; j < n; ++j)
                {
                    a[i * n + j] = cell(i, j, n);
                }
            }
        });
    sa::zero_any::Array<double>::make_block_compression();
    const auto& ca = a;
    const auto  fr = sa::md::view(ca, n, n);
    const auto  ft = sa::md::tiled_view<16, 16>(std::as_const(t), n, n);

    double best[3] = {1e30, 1e30, 1e30};
    double sums[3] = {0, 0, 0};
    for (int r = 0; r < 5; ++r)
    {
        for (int c = 0; c < 3; ++c)
        {
            const auto t0 = std::chrono::steady_clock::now();
            double     s  = 0;
            if (c == 0)
            {
    #pragma omp parallel for schedule(static) reduction(+ : s)
                for (long k = 0; k < n * n; ++k)
                {
                    s += ca[k];
                }
            }
            else if (c == 1)
            {
    #pragma omp parallel for schedule(static) reduction(+ : s)
                for (long i = 0; i < n; ++i)
                {
                    for (long j = 0; j < n; ++j)
                    {
                        s += fr[i, j];
                    }
                }
            }
            else
            {
    #pragma omp parallel for schedule(static) reduction(+ : s)
                for (long i = 0; i < n; ++i)
                {
                    for (long j = 0; j < n; ++j)
                    {
                        s += ft[i, j];
                    }
                }
            }
            best[c] = std::min(best[c], seconds_since(t0));
            sums[c] = s;
        }
    }
    if (std::abs(sums[1] - sums[0]) > 1e-12 * std::abs(sums[0]) ||
        std::abs(sums[2] - sums[0]) > 1e-12 * std::abs(sums[0]))
    {
        sa::println("index cost: sums differ ({}, {}, {})", sums[0], sums[1], sums[2]);
        ++bad;
    }
    const double nn = static_cast<double>(n) * static_cast<double>(n);
    sa::println("read a[k] {:.2f} ns/elem   f[i, j] row-major {:.2f}   f[i, j] tiles {:.2f}", 1e9 * best[0] / nn,
                1e9 * best[1] / nn, 1e9 * best[2] / nn);
}

int main(int argc, char** argv)
{
    sa::ParamChunk::init_indexes(true);
    const long n   = argc > 1 ? std::atol(argv[1]) : 2048;
    int        bad = 0;
    sa::println("mdspan_tiles: {}", sa::md::mdspan_source);
    sa::println("mdspan_tiles: {} x {} doubles ({:.1f} MiB as a std::vector), chunk size 256", n, n,
                static_cast<double>(n) * static_cast<double>(n) * sizeof(double) / (1024.0 * 1024.0));
    {
        sa::zero_any::Array<double> r(n * n);
        run("row-major (layout_right)", r, sa::md::view(r, n, n), n, bad);
    }
    {
        sa::zero_any::Array<double> t(static_cast<long>(sa::md::tiled_size<16, 16>(n, n)));
        run("tiles 16 x 16", t, sa::md::tiled_view<16, 16>(t, n, n), n, bad);
    }
    kernels(n, bad);
    paint(n, bad);
    box_copy(n, bad);
    box_copy_team(n, bad);
    stencil(n, bad);
    index_cost(n, bad);
    sa::release_chunk_pools();
    sa::destroy_block_chunk();
    return bad == 0 ? 0 : 1;
}

#else

int main()
{
    sa::println("mdspan_tiles: skipped (neither <mdspan> nor kokkos/mdspan on the include path)");
    return 77;
}

#endif
