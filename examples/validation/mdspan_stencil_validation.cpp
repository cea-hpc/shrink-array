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

// mdspan_stencil_validation: md::stencil_tiles and md::stencil_tiles_mt
// (chunk/mdspan_tile_ops.h) against the same stencil computed through f[i, j]
// on a std::vector reference.
//
//   - a 9-point stencil (halo 1: the diagonal neighbours too) and a 13-point
//     one (halo 2), with a constant border (0, and another value) and a
//     clamped one; tiled views of 16 x 16 and of a tile per chunk
//     (layout_tiled_dyn), the extents not multiples of the tile (padded
//     edges), a halo as wide as a tile; rank 3;
//   - on_uniform: the same result with and without it, and the uniform tiles
//     of the result left compressed (every variant at first_value, Chunk2A at
//     any value); the source left untouched;
//   - serial, per tile in an omp for (the range overload) and
//     stencil_tiles_mt from every thread of a phase; and called outside any
//     region;
//   - `alias` argument: the source and the destination one array must stop
//     the program (ctest: expect_stop.cmake).
//
// Exit 0, 1 with the first mismatches, 77 (ctest SKIP_RETURN_CODE) without
// mdspan. Run by ctest; the variant comes from the CHUNK_VARIANT_* macro.

#include <algorithm>   // std::clamp, std::min, std::max
#include <array>       // std::array (the extents)
#include <cstddef>     // std::size_t
#include <string_view> // std::string_view (the argument)
#include <type_traits> // std::conditional_t
#include <vector>      // std::vector (the references)

#include "chunk/array_chunk.h"     // zero::Array
#include "chunk/array_chunk2.h"    // zero_one_value::Array
#include "chunk/array_chunk2A.h"   // zero_any::Array
#include "chunk/chunk_std.h"       // release_chunk_pools, destroy_block_chunk
#include "chunk/config.h"          // chunk_config::use_chunk2*
#include "chunk/internal/print.h"  // shrink_array::println
#include "chunk/manage_ompt.h"     // parallel_phase
#include "chunk/mdspan_tile_ops.h" // md::stencil_tiles(_mt), SHRINK_ARRAY_HAS_MDSPAN
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
        println("mdspan_stencil_validation: FAIL {} ({}, {})", what, a, b);
    }
}

constexpr long nx = 100; // not multiples of the tile: padded tiles on both edges
constexpr long ny = 90;

// Varied in a block, 3.0 on whole tiles, 0 elsewhere: most tiles uniform.
double cell(long i, long j)
{
    if (i >= 20 && i < 45 && j >= 30 && j < 70)
    {
        return static_cast<double>((i * 7 + j * 3) % 11) + 0.5;
    }
    if (i >= 64 && i < 96 && j >= 0 && j < 48)
    {
        return 3.0;
    }
    return 0.0;
}

// The stencil: halo 1, the 9 points; halo 2 adds the 4 points at distance 2.
template <typename In>
double apply(const In& in, long i, long j, long h)
{
    double s = in(i, j) + 0.1 * (in(i - 1, j) + in(i + 1, j) + in(i, j - 1) + in(i, j + 1) - 4.0 * in(i, j)) +
               0.01 * (in(i - 1, j - 1) + in(i + 1, j + 1) - in(i - 1, j + 1) - in(i + 1, j - 1));
    if (h >= 2)
    {
        s += 0.001 * (in(i - 2, j) + in(i + 2, j) + in(i, j - 2) + in(i, j + 2) - 4.0 * in(i, j));
    }
    return s;
}

enum class Border
{
    zero,
    value,
    clamp,
};
constexpr double border_value = 2.5;

// The reference: the stencil through f[i, j] on a vector, the border applied.
std::vector<double> reference(const std::vector<double>& field, long h, Border b)
{
    std::vector<double> out(field.size());
    const auto          at = [&](long i, long j)
    {
        if (b == Border::clamp)
        {
            i = std::clamp(i, 0L, nx - 1);
            j = std::clamp(j, 0L, ny - 1);
        }
        else if (i < 0 || i >= nx || j < 0 || j >= ny)
        {
            return b == Border::zero ? 0.0 : border_value;
        }
        return field[static_cast<std::size_t>(i * ny + j)];
    };
    for (long i = 0; i < nx; ++i)
    {
        for (long j = 0; j < ny; ++j)
        {
            out[static_cast<std::size_t>(i * ny + j)] = apply(at, i, j, h);
        }
    }
    return out;
}

template <typename View>
void write_field(const View& f)
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
}

// The kernel and its uniform shortcut (a uniform neighbourhood stays as is).
auto kernel_for(long h)
{
    return [h](const auto& t, const auto& in, auto out)
    {
        const auto at = [&in](long i, long j) { return in[i, j]; };
        for (long i = 0; i < t.extent[0]; ++i)
        {
            for (long j = 0; j < t.extent[1]; ++j)
            {
                out[i, j] = apply(at, i, j, h);
            }
        }
    };
}

const auto keep_uniform = [](const auto&, const double& v, double& out)
{
    out = v;
    return true;
};

enum class How
{
    serial,
    range,
    team,
    team_outside,
};

template <typename Src, typename Dst, typename B, typename... U>
void run_stencil(How w, const Src& src, const Dst& dst, long h, const B& bnd, U&&... uniform)
{
    const auto kernel = kernel_for(h);
    switch (w)
    {
        case How::serial:
            md::stencil_tiles(src, dst, h, bnd, kernel, uniform...);
            break;
        case How::range:
        {
            const auto nt = md::tile_count(src);
            parallel_phase(
                [&]
                {
    #pragma omp for schedule(static, 1)
                    for (long k = 0; k < nt; ++k)
                    {
                        md::stencil_tiles(src, dst, h, bnd, static_cast<typename Src::index_type>(k),
                                          static_cast<typename Src::index_type>(k + 1), kernel, uniform...);
                    }
                });
            break;
        }
        case How::team:
            parallel_phase([&] { md::stencil_tiles_mt(src, dst, h, bnd, kernel, uniform...); });
            break;
        case How::team_outside:
            md::stencil_tiles_mt(src, dst, h, bnd, kernel, uniform...);
            break;
    }
}

// One case: a tiled view made by @p make over two arrays of @p size.
template <typename Make>
void case_2d(const char* name, long size, Make&& make, long h, Border b, How w, bool with_uniform)
{
    Array<double>       a(static_cast<int>(size));
    Array<double>       d(static_cast<int>(size));
    auto                f = make(a);
    auto                g = make(d);
    std::vector<double> field(static_cast<std::size_t>(nx * ny));
    write_field(f);
    for (long i = 0; i < nx; ++i)
    {
        for (long j = 0; j < ny; ++j)
        {
            field[static_cast<std::size_t>(i * ny + j)] = cell(i, j);
        }
    }
    const auto want = reference(field, h, b);
    const auto src  = md::as_const(f);
    if (b == Border::clamp)
    {
        with_uniform ? run_stencil(w, src, g, h, md::clamp_boundary{}, keep_uniform)
                     : run_stencil(w, src, g, h, md::clamp_boundary{});
    }
    else
    {
        const md::constant_boundary<double> bnd{b == Border::zero ? 0.0 : border_value};
        with_uniform ? run_stencil(w, src, g, h, bnd, keep_uniform) : run_stencil(w, src, g, h, bnd);
    }
    long bad     = 0;
    long src_bad = 0;
    for (long i = 0; i < nx; ++i)
    {
        for (long j = 0; j < ny; ++j)
        {
            const double x = md::as_const(g)[i, j];
            const double y = want[static_cast<std::size_t>(i * ny + j)];
            if (!(x == y || (x - y < 1e-12 && y - x < 1e-12)))
            {
                if (bad == 0)
                {
                    println("  {}: [{}, {}] = {} expected {}", name, i, j, x, y);
                }
                ++bad;
            }
            src_bad += md::as_const(f)[i, j] != cell(i, j) ? 1 : 0;
        }
    }
    expect(bad == 0, name, bad, static_cast<long>(w) * 100 + static_cast<long>(b) * 10 + h);
    expect(src_bad == 0, "the source changed", src_bad);
    // The tiles of the result whose cells are one value hold no buffer: at 0
    // on every variant, at any value on Chunk2A (checked when a tile is one
    // chunk).
    using Index   = typename decltype(g)::index_type;
    const long S  = static_cast<long>(d.chunk_size());
    const long TE = static_cast<long>(g.mapping().tile_elements());
    if (TE == S)
    {
        const auto dims = md::detail::tiles_per_dim(g);
        long       kept = 0;
        for (long k = 0; k < static_cast<long>(md::tile_count(g)); ++k)
        {
            const auto   t       = md::detail::tile_at(g, dims, static_cast<Index>(k));
            const double v0      = want[static_cast<std::size_t>(t.origin[0] * ny + t.origin[1])];
            bool         uniform = true;
            for (long i = 0; i < t.extent[0]; ++i)
            {
                for (long j = 0; j < t.extent[1]; ++j)
                {
                    uniform = uniform && want[static_cast<std::size_t>((t.origin[0] + i) * ny + t.origin[1] + j)] == v0;
                }
            }
            if (uniform && (v0 == 0.0 || chunk_config::use_chunk2a))
            {
                kept += d.written(static_cast<int>(k * S)) ? 1 : 0;
            }
        }
        expect(kept == 0, "a uniform tile of the result holds a buffer", kept, static_cast<long>(w));
    }
}

void tiled(How w)
{
    const long S = static_cast<long>(Array<double>(1).chunk_size());
    for (const long h : {1L, 2L})
    {
        for (const Border b : {Border::zero, Border::value, Border::clamp})
        {
            for (const bool u : {false, true})
            {
                case_2d(
                    "tiled 16 x 16", static_cast<long>(md::tiled_size<16, 16>(nx, ny)),
                    [](auto& arr) { return md::tiled_view<16, 16>(arr, nx, ny); }, h, b, w, u);
                const auto tile = md::tile_shape<2, long>(static_cast<std::size_t>(S));
                case_2d(
                    "tiled, a tile per chunk", static_cast<long>(md::tiled_size_dyn(tile, std::array<long, 2>{nx, ny})),
                    [](auto& arr) { return md::tiled_view_dyn(arr, nx, ny); }, h, b, w, u);
            }
        }
    }
    // A halo as wide as a tile (8 x 8 tiles, halo 2 and 8 ... 8 only serially).
    if (w == How::serial)
    {
        case_2d(
            "tiled 8 x 8, halo 2", static_cast<long>(md::tiled_size<8, 8>(nx, ny)),
            [](auto& arr) { return md::tiled_view<8, 8>(arr, nx, ny); }, 2, Border::clamp, w, true);
    }
}

// Rank 3: a 7-point stencil, clamped, against the element path.
void rank3()
{
    const long    n0 = 20, n1 = 18, n2 = 21;
    Array<double> a(static_cast<int>(md::tiled_size<4, 8, 8>(n0, n1, n2)));
    Array<double> d(static_cast<int>(md::tiled_size<4, 8, 8>(n0, n1, n2)));
    auto          f = md::tiled_view<4, 8, 8>(a, n0, n1, n2);
    auto          g = md::tiled_view<4, 8, 8>(d, n0, n1, n2);
    const auto    v = [](long i, long j, long k)
    { return (i > 5 && j < 9) ? static_cast<double>((i + 2 * j + 3 * k) % 5) : 1.0; };
    for (long i = 0; i < n0; ++i)
    {
        for (long j = 0; j < n1; ++j)
        {
            for (long k = 0; k < n2; ++k)
            {
                f[i, j, k] = v(i, j, k);
            }
        }
    }
    const auto at = [&](long i, long j, long k)
    { return v(std::clamp(i, 0L, n0 - 1), std::clamp(j, 0L, n1 - 1), std::clamp(k, 0L, n2 - 1)); };
    parallel_phase(
        [&]
        {
            md::stencil_tiles_mt(
                md::as_const(f), g, 1L, md::clamp_boundary{},
                [](const auto& t, const auto& in, auto out)
                {
                    for (long i = 0; i < t.extent[0]; ++i)
                    {
                        for (long j = 0; j < t.extent[1]; ++j)
                        {
                            for (long k = 0; k < t.extent[2]; ++k)
                            {
                                out[i, j, k] = in[i - 1, j, k] + in[i + 1, j, k] + in[i, j - 1, k] + in[i, j + 1, k] +
                                               in[i, j, k - 1] + in[i, j, k + 1] - 6 * in[i, j, k];
                            }
                        }
                    }
                },
                [](const auto&, const double&, double& out)
                {
                    out = 0.0;
                    return true;
                });
        });
    long bad = 0;
    for (long i = 0; i < n0; ++i)
    {
        for (long j = 0; j < n1; ++j)
        {
            for (long k = 0; k < n2; ++k)
            {
                const double want = at(i - 1, j, k) + at(i + 1, j, k) + at(i, j - 1, k) + at(i, j + 1, k) +
                                    at(i, j, k - 1) + at(i, j, k + 1) - 6 * at(i, j, k);
                const double x    = md::as_const(g)[i, j, k];
                bad += (x - want > 1e-12 || want - x > 1e-12) ? 1 : 0;
            }
        }
    }
    expect(bad == 0, "rank 3", bad);
}

// The source and the destination one array: refused.
int alias()
{
    Array<double> a(static_cast<int>(md::tiled_size<16, 16>(nx, ny)));
    auto          f = md::tiled_view<16, 16>(a, nx, ny);
    md::stencil_tiles(md::as_const(f), f, 1L, md::clamp_boundary{}, [](const auto&, const auto&, auto) {});
    println("mdspan_stencil_validation: a stencil from an array into itself was accepted");
    return 1;
}
} // namespace

int main(int argc, char** argv)
{
    ParamChunk::init_indexes(true);
    if (argc > 1 && std::string_view(argv[1]) == "alias")
    {
        return alias();
    }
    tiled(How::serial);
    tiled(How::range);
    tiled(How::team);
    tiled(How::team_outside);
    rank3();
    release_chunk_pools();
    destroy_block_chunk();
    if (fails != 0)
    {
        println("mdspan_stencil_validation: {} failure(s)", fails);
        return 1;
    }
    println("mdspan_stencil_validation: OK ({})", md::mdspan_source);
    return 0;
}

#else

int main(int argc, char** argv)
{
    (void)argc;
    (void)argv;
    println("mdspan_stencil_validation: skipped (neither <mdspan> nor kokkos/mdspan on the include path)");
    return 77;
}

#endif
