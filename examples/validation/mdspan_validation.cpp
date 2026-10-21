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

// mdspan_validation: the mdspan view of chunk/mdspan.h over an array.
//
//   - layout_right: f[i, j] = v lands in a[i * ny + j]; a[k] = v reads back
//     through the view; compound operators and element-to-element copies go
//     through the proxy; a 3D view; a read-only view (const array, the
//     conversion mdspan<T> -> mdspan<const T>, md::as_const of a row-major,
//     a tiled and a sub-view; md::tiled_view_type).
//   - layout_tiled: the mapping is a bijection onto [0, required_span_size()),
//     tile k covers elements [k * tile, (k + 1) * tile), with and without
//     padding; writes through the tiled view read back through it; a uniform
//     tile is a uniform chunk, a row-major uniform block of the same shape is not.
//
// Exit 0, 1 with the first mismatches, 77 (ctest SKIP_RETURN_CODE) when neither
// <mdspan> nor kokkos/mdspan is available. Run by ctest; the variant comes from
// the CHUNK_VARIANT_* macro of the target.

#include <cstddef>     // std::size_t
#include <type_traits> // std::conditional_t, std::is_same_v, std::remove_cvref_t
#include <utility>     // std::as_const, std::pair (the sub-view)
#include <vector>      // std::vector (the bijection check)

#include "chunk/array_chunk.h"    // ArrayChunk
#include "chunk/array_chunk2.h"   // ArrayChunk2
#include "chunk/array_chunk2A.h"  // ArrayChunk2A
#include "chunk/chunk_std.h"      // release_chunk_pools, destroy_block_chunk
#include "chunk/config.h"         // chunk_config::use_chunk2*
#include "chunk/internal/print.h" // shrink_array::println
#include "chunk/mdspan.h"         // shrink_array::md, SHRINK_ARRAY_HAS_MDSPAN
#include "chunk/param_chunk.h"    // ParamChunk::init_indexes

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
        println("mdspan_validation: FAIL {} ({}, {})", what, a, b);
    }
}

double val(long i, long j)
{
    return static_cast<double>(i * 1000 + j) + 0.5;
}

void row_major()
{
    constexpr long nx = 37;
    constexpr long ny = 53; // not a multiple of any chunk size: rows straddle chunks
    Array<double>  a(nx * ny);
    auto           f = md::view(a, nx, ny);
    expect(f.extent(0) == nx && f.extent(1) == ny, "extents");
    expect(f.size() == static_cast<std::size_t>(nx * ny), "size");

    for (long i = 0; i < nx; ++i)
    {
        for (long j = 0; j < ny; ++j)
        {
            f[i, j] = val(i, j);
        }
    }
    for (long i = 0; i < nx; ++i)
    {
        for (long j = 0; j < ny; ++j)
        {
            const double x = a[i * ny + j];
            expect(x == val(i, j), "view write, linear read", i, j);
        }
    }

    // The other way round, and the compound operators through the view.
    for (long k = 0; k < nx * ny; ++k)
    {
        a[k] = static_cast<double>(k);
    }
    for (long i = 0; i < nx; ++i)
    {
        for (long j = 0; j < ny; ++j)
        {
            const double x = f[i, j];
            expect(x == static_cast<double>(i * ny + j), "linear write, view read", i, j);
            f[i, j] += 1.0;
            f[i, j] *= 2.0;
        }
    }
    for (long k = 0; k < nx * ny; ++k)
    {
        const double x = a[k];
        expect(x == 2.0 * static_cast<double>(k + 1), "compound through the view", k);
    }

    // Element to element: transpose a square corner through two views of one array.
    Array<double> t(nx * nx);
    auto          g = md::view(t, nx, nx);
    for (long i = 0; i < nx; ++i)
    {
        for (long j = 0; j < nx; ++j)
        {
            g[j, i] = f[i, j];
        }
    }
    for (long i = 0; i < nx; ++i)
    {
        for (long j = 0; j < nx; ++j)
        {
            const double x = t[j * nx + i];
            const double y = a[i * ny + j];
            expect(x == y, "transpose", i, j);
        }
    }

    // Read-only: a const array, and a writable view converted.
    const Array<double>& ca = a;
    auto                 cf = md::view(ca, nx, ny);
    static_assert(std::is_same_v<typename decltype(cf)::reference, double>, "const view reads values");
    md::view_type<const Array<double>> cg = f; // mdspan<T> -> mdspan<const T>
    for (long i = 0; i < nx; ++i)
    {
        for (long j = 0; j < ny; ++j)
        {
            expect(cf[i, j] == cg[i, j] && cf[i, j] == 2.0 * static_cast<double>(i * ny + j + 1), "const view", i, j);
        }
    }

    // md::as_const: the same view over const Array; std::as_const leaves the
    // element type writable (an mdspan is const on the surface only).
    const auto ch = md::as_const(f);
    static_assert(std::is_same_v<std::remove_const_t<decltype(ch)>, md::view_type<const Array<double>>>,
                  "md::as_const gives view_type<const Array>");
    static_assert(std::is_same_v<typename std::remove_cvref_t<decltype(std::as_const(f))>::element_type, double>,
                  "std::as_const keeps the element writable");
    static_assert(std::is_same_v<std::remove_const_t<decltype(md::as_const(ch))>, std::remove_const_t<decltype(ch)>>,
                  "a read-only view is returned as it is");
    Array<double> ta(static_cast<long>(md::tiled_size<16, 16>(nx, ny)));
    auto          tf = md::tiled_view<16, 16>(ta, nx, ny);
    static_assert(std::is_same_v<decltype(tf), md::tiled_view_type<Array<double>, 16, 16>>, "tiled_view_type");
    const auto tc = md::as_const(tf);
    static_assert(std::is_same_v<std::remove_const_t<decltype(tc)>, md::tiled_view_type<const Array<double>, 16, 16>>,
                  "md::as_const of a tiled view");
    for (long i = 0; i < nx; ++i)
    {
        for (long j = 0; j < ny; ++j)
        {
            tf[i, j] = static_cast<double>(i * 1000 + j);
        }
    }
    for (long i = 0; i < nx; ++i)
    {
        for (long j = 0; j < ny; ++j)
        {
            const double x = ch[i, j];
            const double y = f[i, j];
            expect(x == y, "md::as_const of a row-major view", i, j);
            expect(tc[i, j] == static_cast<double>(i * 1000 + j), "md::as_const of a tiled view", i, j);
        }
    }
    #if SHRINK_ARRAY_HAS_SUBMDSPAN
    const auto sub = md::as_const(md::submdspan(f, std::pair{1, nx - 1}, std::pair{2, ny - 3}));
    static_assert(std::is_const_v<typename decltype(sub)::element_type>, "md::as_const of a sub-view");
    for (long i = 0; i + 2 < nx; ++i)
    {
        for (long j = 0; j + 5 < ny; ++j)
        {
            const double y = f[i + 1, j + 2];
            expect(sub[i, j] == y, "md::as_const of a sub-view", i, j);
        }
    }
    #endif
}

void rank3()
{
    constexpr long n0 = 5;
    constexpr long n1 = 7;
    constexpr long n2 = 11;
    Array<int>     a(n0 * n1 * n2);
    auto           f = md::view(a, n0, n1, n2);
    for (long i = 0; i < n0; ++i)
    {
        for (long j = 0; j < n1; ++j)
        {
            for (long k = 0; k < n2; ++k)
            {
                f[i, j, k] = static_cast<int>(i * 10000 + j * 100 + k);
            }
        }
    }
    for (long l = 0; l < n0 * n1 * n2; ++l)
    {
        const int  x = a[l];
        const long i = l / (n1 * n2);
        const long j = (l / n2) % n1;
        const long k = l % n2;
        expect(x == static_cast<int>(i * 10000 + j * 100 + k), "rank 3", l, x);
    }
}

// The mapping alone: a bijection, tile by tile, padding counted.
template <std::size_t TH, std::size_t TW>
void tiled_mapping(long nx, long ny)
{
    using M = md::layout_tiled<TH, TW>::template mapping<md::mdns::dextents<long, 2>>;
    const M           m(md::mdns::dextents<long, 2>(nx, ny));
    const long        span = m.required_span_size();
    const long        th   = static_cast<long>(TH);
    const long        tw   = static_cast<long>(TW);
    const long        want = ((nx + th - 1) / th) * th * ((ny + tw - 1) / tw) * tw;
    std::vector<char> seen(static_cast<std::size_t>(span), 0);
    expect(span == want, "required_span_size", span, want);
    expect(static_cast<long>(md::tiled_size<TH, TW>(nx, ny)) == span, "tiled_size", nx, ny);
    expect(m.is_exhaustive() == (nx % th == 0 && ny % tw == 0), "is_exhaustive", nx, ny);
    const long ntw = (ny + tw - 1) / tw;
    for (long i = 0; i < nx; ++i)
    {
        for (long j = 0; j < ny; ++j)
        {
            const long o = m(i, j);
            expect(o >= 0 && o < span, "offset in range", i, j);
            if (o >= 0 && o < span)
            {
                expect(seen[static_cast<std::size_t>(o)] == 0, "offset unique", i, j);
                seen[static_cast<std::size_t>(o)] = 1;
            }
            // The tile of (i, j) is the (o / tile)-th one.
            expect(o / (th * tw) == (i / th) * ntw + j / tw, "tile of the element", i, j);
        }
    }
}

void tiled_view()
{
    constexpr long nx = 50; // 4 tile rows, the last one padded
    constexpr long ny = 70; // 5 tile columns, the last one padded
    Array<double>  a(static_cast<long>(md::tiled_size<16, 16>(nx, ny)));
    auto           f = md::tiled_view<16, 16>(a, nx, ny);
    for (long i = 0; i < nx; ++i)
    {
        for (long j = 0; j < ny; ++j)
        {
            f[i, j] = val(i, j);
        }
    }
    for (long i = 0; i < nx; ++i)
    {
        for (long j = 0; j < ny; ++j)
        {
            const double x = f[i, j];
            const double y = a[f.mapping()(i, j)];
            expect(x == val(i, j) && y == x, "tiled write, read back", i, j);
        }
    }
}

// What the layout is for: a uniform 16 x 16 block. Tiled, it fills one tile,
// i.e. elements [k * 256, (k + 1) * 256), a whole chunk at size 256. Row-major
// (ny = 1024), it is 16 pieces of 16 elements in 16 different chunks, none of
// them uniform. Checked on the elements, not on the chunk states: the tile
// boundaries are the chunk boundaries when chunk_size() == 256.
void tiled_block()
{
    constexpr long nx = 64;
    constexpr long ny = 1024;
    Array<double>  r(nx * ny);
    Array<double>  t(static_cast<long>(md::tiled_size<16, 16>(nx, ny)));
    auto           fr = md::view(r, nx, ny);
    auto           ft = md::tiled_view<16, 16>(t, nx, ny);
    for (long i = 0; i < nx; ++i)
    {
        for (long j = 0; j < ny; ++j)
        {
            const bool   in = i >= 16 && i < 32 && j >= 32 && j < 48; // block (1, 2)
            const double v  = in ? 7.0 : val(i, j);
            fr[i, j]        = v;
            ft[i, j]        = v;
        }
    }
    const long tile = 1 * (ny / 16) + 2; // the block's tile
    for (long k = tile * 256; k < (tile + 1) * 256; ++k)
    {
        const double x = t[k];
        expect(x == 7.0, "tile = block", k);
    }
    long uniform_runs = 0; // row-major: 256-aligned runs of 7.0
    for (long c = 0; c < nx * ny / 256; ++c)
    {
        bool all = true;
        for (long k = c * 256; k < (c + 1) * 256 && all; ++k)
        {
            all = r[k] == 7.0;
        }
        uniform_runs += all ? 1 : 0;
    }
    expect(uniform_runs == 0, "row-major block is not a run", uniform_runs);
}
} // namespace

int main()
{
    ParamChunk::init_indexes(true);
    row_major();
    rank3();
    tiled_mapping<16, 16>(64, 64);
    tiled_mapping<16, 16>(50, 70);
    tiled_mapping<8, 32>(17, 100);
    tiled_mapping<1, 64>(3, 65);
    tiled_view();
    tiled_block();
    release_chunk_pools();
    destroy_block_chunk();
    if (fails != 0)
    {
        println("mdspan_validation: {} failure(s)", fails);
        return 1;
    }
    println("mdspan_validation: OK ({})", md::mdspan_source);
    return 0;
}

#else

int main()
{
    println("mdspan_validation: skipped (neither <mdspan> nor kokkos/mdspan on the include path)");
    return 77;
}

#endif
