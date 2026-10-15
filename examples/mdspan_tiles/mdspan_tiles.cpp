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
 *          row-major and f[i, j] in tiles.
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

#include "chunk/array_chunk2A.h"   // ArrayChunk2A
#include "chunk/chunk_std.h"       // release_chunk_pools, destroy_block_chunk
#include "chunk/internal/print.h"  // shrink_array::println
#include "chunk/manage_ompt.h"     // parallel_phase
#include "chunk/mdspan.h"          // shrink_array::md, SHRINK_ARRAY_HAS_MDSPAN
#include "chunk/PMP_CHUNK.h"       // CHUNK_PMP_THREAD_RANK, CHUNK_PMP_NUM_THREADS
#include "chunk/mdspan_tile_ops.h" // md::for_each_tile, md::update_tiles, md::fill_box
#include "chunk/param_chunk.h"     // ParamChunk::init_indexes

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

Count count(const sa::ArrayChunk2A<double>& a)
{
    Count c{static_cast<long>(a.nb_chunk), 0};
    for (long k = 0; k < c.chunks; ++k)
    {
        const auto& ch = a.chunk_list[k];
        if (!ch.is_cst_val1() && !ch.is_cst_val2())
        {
            ++c.materialised;
        }
    }
    return c;
}

// Fill the field through view f (any layout), compress, check, count.
template <typename View>
Count run(const char* name, sa::ArrayChunk2A<double>& a, View f, long n, int& bad)
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
    sa::ArrayChunk2A<double>::make_block_compression();

    // Read back through a read-only view (values, no proxy), converted from f.
    const sa::md::view_type<const sa::ArrayChunk2A<double>, typename View::layout_type> cf    = f;
    long                                                                                wrong = 0;
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
void fill_tiled(sa::ArrayChunk2A<double>& t, long n)
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
    sa::ArrayChunk2A<double>::make_block_compression();
}

// The sum and the update, element by element and tile by tile.
void kernels(long n, int& bad)
{
    constexpr int            reps = 5;
    const long               size = static_cast<long>(sa::md::tiled_size<16, 16>(n, n));
    sa::ArrayChunk2A<double> e(size);
    sa::ArrayChunk2A<double> t(size);
    fill_tiled<16, 16>(e, n);
    fill_tiled<16, 16>(t, n);
    auto       fe     = sa::md::tiled_view<16, 16>(e, n, n);
    auto       ft     = sa::md::tiled_view<16, 16>(t, n, n);
    const auto ce     = sa::md::view_type<const sa::ArrayChunk2A<double>, sa::md::layout_tiled<16, 16>>(fe);
    const auto ct     = sa::md::view_type<const sa::ArrayChunk2A<double>, sa::md::layout_tiled<16, 16>>(ft);
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
    sa::ArrayChunk2A<double>::make_block_compression();
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

    const long               size = static_cast<long>(sa::md::tiled_size<16, 16>(n, n));
    sa::ArrayChunk2A<double> e(size);
    sa::ArrayChunk2A<double> t(size);
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

    const auto ce    = sa::md::view_type<const sa::ArrayChunk2A<double>, sa::md::layout_tiled<16, 16>>(fe);
    const auto ct    = sa::md::view_type<const sa::ArrayChunk2A<double>, sa::md::layout_tiled<16, 16>>(ft);
    long       wrong = 0;
    for (long i = 0; i < n; ++i)
    {
        for (long j = 0; j < n; ++j)
        {
            wrong += ce[i, j] != ct[i, j] ? 1 : 0;
        }
    }
    if (wrong != 0)
    {
        sa::println("paint: {} cells differ between element and fill_box", wrong);
        ++bad;
    }
    const long m_elem = count(e).materialised;
    const long m_tile = count(t).materialised;
    sa::ArrayChunk2A<double>::make_block_compression();
    sa::println("paint 64 rect. element {:8.2f} ms   tile {:8.2f} ms   (x {:.1f})", 1e3 * p_elem, 1e3 * p_tile,
                p_elem / p_tile);
    sa::println("  materialised at the end of the phase: element {}, tile {}; after one more pass: {} and {}", m_elem,
                m_tile, count(e).materialised, count(t).materialised);
}
} // namespace

// The price of the multi-index: one sum of the same field read as a[k], as
// f[i, j] row-major and as f[i, j] in tiles, element by element (the const
// operator[] every time), best of 5. The tiled mapping divides by constant
// tile extents: shifts and masks, no division.
void index_cost(long n, int& bad)
{
    sa::ArrayChunk2A<double> a(n * n);
    sa::ArrayChunk2A<double> t(static_cast<long>(sa::md::tiled_size<16, 16>(n, n)));
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
    sa::ArrayChunk2A<double>::make_block_compression();
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
    sa::println("mdspan_tiles: {} x {} doubles ({:.1f} MiB as a std::vector), chunk size 256", n, n,
                static_cast<double>(n) * static_cast<double>(n) * sizeof(double) / (1024.0 * 1024.0));
    {
        sa::ArrayChunk2A<double> r(n * n);
        run("row-major (layout_right)", r, sa::md::view(r, n, n), n, bad);
    }
    {
        sa::ArrayChunk2A<double> t(static_cast<long>(sa::md::tiled_size<16, 16>(n, n)));
        run("tiles 16 x 16", t, sa::md::tiled_view<16, 16>(t, n, n), n, bad);
    }
    kernels(n, bad);
    paint(n, bad);
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
