#!/usr/bin/env python3
# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
"""
mdspan_stencil.py -- md::stencil_tiles / md::stencil_tiles_mt: a stencil tile
by tile, the kernel reading a tile and a halo of h cells gathered in one go
from its neighbours, with a constant or clamped border.

Run from the root of the shrink-array tree:  python3 mdspan_stencil.py
After mdspan_copy_box_mt.py and mdspan_tiles_mt.py.
Idempotent; writes nothing unless every hunk applies.

  chunk/public/chunk/mdspan_tile_ops.h        stencil_tiles (every tile, a tile range), stencil_tiles_mt,
                                              halo_block, constant_boundary, clamp_boundary
  examples/validation/mdspan_stencil_validation.cpp   new harness (created)
  examples/validation/CMakeLists.txt          mdspan_stencil_validation_<variant>, mdspan_stencil_alias
  examples/mdspan_tiles/mdspan_tiles.cpp      one diffusion step, element loop against stencil_tiles_mt
  docs/mdspan.md, docs/mdspan_fr.md           the section, the files table
  CHANGELOG.md                                Unreleased Added
"""
import os
import re
import sys

ROOT = os.getcwd()

HUNKS = [
    ('CHANGELOG.md',
     r'''  `omp for`; over the whole view the same time as the `omp for`. The serial
  forms now run on the same code (`detail::read_tiles`, `write_tiles`,
  `fill_tiles`, `fill_strided`), behaviour unchanged.

### Changed

''',
     r'''  `omp for`; over the whole view the same time as the `omp for`. The serial
  forms now run on the same code (`detail::read_tiles`, `write_tiles`,
  `fill_tiles`, `fill_strided`), behaviour unchanged.
- `md::stencil_tiles(src, dst, h, boundary, kernel, on_uniform)`, its tile
  range form and `md::stencil_tiles_mt` (`chunk/mdspan_tile_ops.h`,
  experimental): a stencil tile by tile, the kernel reading the tile and a
  halo of `h` cells (`md::halo_block`, `in[i - 1, j]`) gathered in one go
  from the neighbour tiles, the border `md::constant_boundary<T>{v}` or
  `md::clamp_boundary{}`, the result tile of another array written back in
  one go (as one constant when uniform); `on_uniform` computes a tile whose
  neighbourhood is one value in O(1). Any rank. One diffusion step on a
  2048 x 2048 field: x 7 to 10 against the element loop, x 11 to 23 with
  `on_uniform`. Harness `mdspan_stencil_validation` (every variant, 1 and 8
  threads) and `mdspan_stencil_alias` (a stencil in place refused).

### Changed

'''),
    ('chunk/public/chunk/mdspan_tile_ops.h',
     r''' *        to / from contiguous row-major memory (an MPI halo, an HDF5
 *        hyperslab, a library that wants a T*), tile by tile or row by row;
 *        copy_box_out_mt() / copy_box_in_mt() share the box among the
 *        threads of a region by themselves.
 *
 * @details f[i, j] pays an operator[] per element. With layout_tiled, a tile is
 *          a run of consecutive elements -- one chunk when the tile has
''',
     r''' *        to / from contiguous row-major memory (an MPI halo, an HDF5
 *        hyperslab, a library that wants a T*), tile by tile or row by row;
 *        copy_box_out_mt() / copy_box_in_mt() share the box among the
 *        threads of a region by themselves; stencil_tiles() runs a stencil
 *        tile by tile, the kernel reading a tile and its halo.
 *
 * @details f[i, j] pays an operator[] per element. With layout_tiled, a tile is
 *          a run of consecutive elements -- one chunk when the tile has
'''),
    ('chunk/public/chunk/mdspan_tile_ops.h',
     r'''    #include <algorithm> // std::copy, std::copy_n, std::fill, std::fill_n, std::equal, std::all_of, std::min, std::max, std::clamp
    #include <array>       // std::array (the tile buffer, the tile coordinates)
    #include <concepts>    // std::invocable, std::same_as, std::integral
    #include <cstddef>     // std::size_t
    #include <limits>      // std::numeric_limits (the box corners)
    #include <span>        // std::span (the blocks of for_each_span)
    #include <tuple>       // std::apply (a multi-index into a mapping)
    #include <type_traits> // std::remove_cvref_t, std::is_const_v
    #include <utility>     // std::as_const, std::forward, std::pair
    #include <vector>      // std::vector (a run-time tile larger than the inline buffer)

    #include "chunk/PMP_CHUNK.h"                // CHUNK_PMP_IN_PARALLEL_REGION
    #include "chunk/array_chunk_range_ops.h"    // for_each_span, fill_range(_mt), copy_range(_mt)
''',
     r'''    #include <algorithm> // std::copy, std::copy_n, std::fill, std::fill_n, std::equal, std::all_of, std::min, std::max, std::clamp
    #include <array>       // std::array (the tile buffer, the tile coordinates)
    #include <concepts>    // std::invocable, std::same_as, std::integral
    #include <cstddef>     // std::size_t, std::ptrdiff_t
    #include <limits>      // std::numeric_limits (the box corners)
    #include <span>        // std::span (the blocks of for_each_span)
    #include <tuple>       // std::apply (a multi-index into a mapping)
    #include <type_traits> // std::remove_cvref_t, std::is_const_v
    #include <utility>     // std::as_const, std::forward, std::pair
    #include <vector> // std::vector (a run-time tile larger than the inline buffer, a stencil's halo and tile cache)

    #include "chunk/PMP_CHUNK.h"                // CHUNK_PMP_IN_PARALLEL_REGION
    #include "chunk/array_chunk_range_ops.h"    // for_each_span, fill_range(_mt), copy_range(_mt)
'''),
    ('chunk/public/chunk/mdspan_tile_ops.h',
     r'''                        on_uniform);
}

namespace detail
{
/// @brief fill_box() of a tiled view on the tiles @p walk(lo, hi, fn) gives,
''',
     r'''                        on_uniform);
}

// ─────────────────────────────────────────────────────────────────────────────
// Stencils: a tile with its halo, stencil_tiles()
// ─────────────────────────────────────────────────────────────────────────────

/// @brief Halo cells outside the view take @p value (a Dirichlet border).
template <typename T>
struct constant_boundary
{
    T value{};
};

/// @brief Halo cells outside the view take the value of the nearest cell of
///        the view (a zero-gradient, Neumann border).
struct clamp_boundary
{
};

/**
 * @brief What a stencil kernel reads: a tile and @p h cells around it, read
 *        only. in[i, j] for i in [-h, te0 + h), j in [-h, te1 + h): (0, 0) is
 *        the tile's first cell, te the tile's extents (padding included);
 *        cells outside the view hold the boundary's value.
 */
template <typename T, std::size_t Rank, typename Index>
class halo_block
{
  public:
    halo_block(const T* origin, const std::array<std::ptrdiff_t, Rank>& strides, Index h) noexcept
        : origin_(origin), strides_(strides), h_(h)
    {
    }

    template <std::integral... I>
        requires(sizeof...(I) == Rank)
    const T& operator[](I... i) const noexcept
    {
        std::ptrdiff_t o = 0;
        std::size_t    r = 0;
        ((o += static_cast<std::ptrdiff_t>(i) * strides_[r++]), ...);
        return origin_[o];
    }

    /// @brief The halo's width, cells on each side.
    Index halo() const noexcept
    {
        return h_;
    }

  private:
    const T*                         origin_;
    std::array<std::ptrdiff_t, Rank> strides_;
    Index                            h_;
};

namespace detail
{
template <typename B, typename T>
concept boundary_of = std::same_as<B, clamp_boundary> || std::same_as<B, constant_boundary<T>>;

/// @brief The tiles a stencil reads around the one it computes: gathered once
///        each (lazily: a uniform tile is its constant), kept while the walk
///        goes along a row of tiles.
template <typename View, typename T>
class tile_cache
{
    using Index = typename View::index_type;

  public:
    struct entry
    {
        explicit entry(const View& f) : buf(f) {}
        Index                k       = std::numeric_limits<Index>::max();
        bool                 uniform = false;
        T                    cst{};
        tile_buffer<View, T> buf;
    };

    tile_cache(const View& f, std::size_t n) : f_(f), dims_(tiles_per_dim(f))
    {
        e_.reserve(n);
        for (std::size_t i = 0; i < n; ++i)
        {
            e_.emplace_back(f);
        }
    }

    /// @brief The tile of coordinates @p c (in tiles), read.
    entry& fetch(const std::array<Index, View::rank()>& c)
    {
        Index k = 0;
        for (std::size_t r = 0; r < View::rank(); ++r)
        {
            k = k * dims_[r] + c[r];
        }
        for (auto& x : e_)
        {
            if (x.k == k)
            {
                return x;
            }
        }
        entry& x     = e_[next_];
        next_        = (next_ + 1) % e_.size();
        const auto t = tile_at(f_, dims_, k);
        x.uniform    = gather(*f_.data_handle(), tile_storage(f_, t), x.buf, x.cst, /*lazy=*/true);
        x.k          = k;
        return x;
    }

    const std::array<Index, View::rank()>& dims() const noexcept
    {
        return dims_;
    }

  private:
    const View&                     f_;
    std::array<Index, View::rank()> dims_;
    std::vector<entry>              e_;
    std::size_t                     next_ = 0;
};

/// @brief 3^rank: the tiles around a tile, itself included.
template <std::size_t Rank>
constexpr std::size_t neighbourhood()
{
    std::size_t n = 1;
    for (std::size_t r = 0; r < Rank; ++r)
    {
        n *= 3;
    }
    return n;
}

/// @brief stencil_tiles() on the tiles @p walk(fn) gives.
template <typename Src, typename Dst, typename Boundary, typename Walk, typename OnBlock, typename OnUniform>
void stencil_on(const Src&                       src,
                const Dst&                       dst,
                typename Src::index_type         h,
                [[maybe_unused]] const Boundary& bnd,
                const char*                      who,
                Walk&&                           walk,
                OnBlock&                         on_block,
                OnUniform&                       on_uniform)
{
    using T             = value_of<Src>;
    using Index         = typename Src::index_type;
    constexpr auto rank = Src::rank();
    static_assert(rank == Dst::rank(), "md::stencil_tiles: the source and the destination have one rank");
    static_assert(std::same_as<T, value_of<Dst>>, "md::stencil_tiles: the source and the destination hold one type");
    static_assert(boundary_of<Boundary, T>,
                  "md::stencil_tiles: the boundary is md::clamp_boundary{} or md::constant_boundary<T>{v}");
    constexpr bool has_uniform = !std::same_as<std::remove_cvref_t<OnUniform>, no_uniform>;
    constexpr bool clamp       = std::same_as<Boundary, clamp_boundary>;

    // The two views: one shape, one tiling, two arrays.
    for (std::size_t r = 0; r < rank; ++r)
    {
        const auto rr = static_cast<typename Src::rank_type>(r);
        if (src.extent(r) != dst.extent(r) || src.mapping().tile_extent(rr) != dst.mapping().tile_extent(rr) || h < 0 ||
            h > src.mapping().tile_extent(rr)) [[unlikely]]
        {
            CHUNK_INVARIANT_FAILED("{}: the source and the destination must have the same extents and tiles, and "
                                   "the halo ({}) at most a tile's extent; dimension {}: extents {} and {}, tiles "
                                   "{} and {}",
                                   who, static_cast<long long>(h), r, static_cast<long long>(src.extent(r)),
                                   static_cast<long long>(dst.extent(r)),
                                   static_cast<long long>(src.mapping().tile_extent(rr)),
                                   static_cast<long long>(dst.mapping().tile_extent(rr)));
        }
    }
    if (static_cast<const void*>(&*src.data_handle()) == static_cast<const void*>(&*dst.data_handle())) [[unlikely]]
    {
        CHUNK_INVARIANT_FAILED("{}: the source and the destination are one array: a stencil reads cells its other "
                               "tiles write",
                               who);
    }
    const auto& sa = *src.data_handle();
    auto&       da = *dst.data_handle();
    check_tile_on_chunks(sa, src.mapping().tile_elements(), who);
    check_tile_on_chunks(da, dst.mapping().tile_elements(), who);

    std::array<Index, rank>          te{}; // tile extents
    std::array<Index, rank>          he{}; // halo buffer extents
    std::array<std::ptrdiff_t, rank> hs{}; // its strides
    std::array<std::size_t, rank>    ts{}; // a tile's strides
    for (std::size_t r = 0; r < rank; ++r)
    {
        te[r] = src.mapping().tile_extent(static_cast<typename Src::rank_type>(r));
        he[r] = te[r] + 2 * h;
    }
    {
        std::ptrdiff_t s = 1;
        std::size_t    u = 1;
        for (std::size_t r = rank; r-- > 0;)
        {
            hs[r] = s;
            ts[r] = u;
            s *= static_cast<std::ptrdiff_t>(he[r]);
            u *= static_cast<std::size_t>(te[r]);
        }
    }
    std::size_t hn = 1;
    for (std::size_t r = 0; r < rank; ++r)
    {
        hn *= static_cast<std::size_t>(he[r]);
    }
    std::vector<T> halo(hn);
    std::ptrdiff_t centre = 0;
    for (std::size_t r = 0; r < rank; ++r)
    {
        centre += static_cast<std::ptrdiff_t>(h) * hs[r];
    }

    tile_cache<Src, T>          cache(src, 2 * neighbourhood<rank>());
    const auto                  dims = cache.dims();
    const std::size_t           N    = static_cast<std::size_t>(src.mapping().tile_elements());
    tile_buffer<Dst, T>         outb(dst);
    range_writer<array_of<Dst>> out{da};
    T                           boundary_value{}; // the halo outside the view, constant_boundary's
    if constexpr (!clamp)
    {
        boundary_value = bnd.value;
    }

    walk(
        [&](Index k)
        {
            const auto              t = tile_at(src, dims, k);
            std::array<Index, rank> tc{};
            for (std::size_t r = 0; r < rank; ++r)
            {
                tc[r] = t.origin[r] / te[r];
            }
            auto&      me    = cache.fetch(tc);
            const bool me_u  = me.uniform;
            const T    me_v  = me.cst;
            bool       leave = false; // the halo leaves the view
            for (std::size_t r = 0; r < rank; ++r)
            {
                leave = leave || t.origin[r] < h || t.origin[r] + te[r] + h > src.extent(r);
            }

            // The tile, its neighbours and the border all one value: O(1).
            if constexpr (has_uniform)
            {
                bool uniform = me_u && (clamp || !leave || boundary_value == me_v);
                for (std::size_t d = 0; uniform && h > 0 && d < neighbourhood<rank>(); ++d)
                {
                    std::array<Index, rank> c    = tc;
                    std::size_t             code = d;
                    bool                    in   = true;
                    for (std::size_t r = rank; r-- > 0;)
                    {
                        const auto off = static_cast<long long>(code % 3) - 1;
                        code /= 3;
                        const auto x = static_cast<long long>(tc[r]) + off;
                        in           = in && x >= 0 && x < static_cast<long long>(dims[r]);
                        c[r]         = static_cast<Index>(x < 0 ? 0 : x);
                    }
                    if (in)
                    {
                        const auto& nb = cache.fetch(c);
                        uniform        = nb.uniform && nb.cst == me_v;
                    }
                }
                if (uniform)
                {
                    T v = me_v;
                    if (on_uniform(t, std::as_const(me_v), v))
                    {
                        const std::size_t s = tile_storage(dst, t);
                        out.fill(s, s + N, v);
                        return;
                    }
                }
            }

            // The halo buffer, row by row (every dimension but the last).
            const Index             w     = he[rank - 1];
            const auto              ext_l = static_cast<long long>(src.extent(rank - 1));
            const long long         g0_l  = static_cast<long long>(t.origin[rank - 1]) - h; // global of column 0
            const long long         qa    = std::max<long long>(0, -g0_l);
            const long long         qb    = std::min<long long>(w, ext_l - g0_l);
            std::array<Index, rank> p{}; // halo-local row (last coordinate 0)
            while (true)
            {
                T*                      row     = halo.data();
                bool                    outside = false;
                std::array<Index, rank> gc{}; // the row's cell in the view, clamped
                for (std::size_t r = 0; r + 1 < rank; ++r)
                {
                    row += static_cast<std::ptrdiff_t>(p[r]) * hs[r];
                    const long long g = static_cast<long long>(t.origin[r]) - h + p[r];
                    const long long e = static_cast<long long>(src.extent(r));
                    outside           = outside || g < 0 || g >= e;
                    gc[r]             = static_cast<Index>(std::clamp<long long>(g, 0, e - 1));
                }
                if ((!clamp && outside) || qa >= qb)
                {
                    std::fill_n(row, w, boundary_value);
                }
                else
                {
                    long long q = qa;
                    while (q < qb)
                    {
                        const long long         g = g0_l + q;
                        std::array<Index, rank> c{}; // tile coordinates
                        std::size_t             off = 0;
                        for (std::size_t r = 0; r + 1 < rank; ++r)
                        {
                            c[r] = gc[r] / te[r];
                            off += static_cast<std::size_t>(gc[r] % te[r]) * ts[r];
                        }
                        c[rank - 1]      = static_cast<Index>(g / te[rank - 1]);
                        const auto local = static_cast<Index>(g % te[rank - 1]);
                        const auto n     = std::min<long long>(qb - q, te[rank - 1] - local);
                        auto&      nb    = cache.fetch(c);
                        if (nb.uniform)
                        {
                            std::fill_n(row + q, n, nb.cst);
                        }
                        else
                        {
                            std::copy_n(nb.buf.data() + off + local, n, row + q);
                        }
                        q += n;
                    }
                    std::fill_n(row, qa, clamp ? row[qa] : boundary_value);
                    std::fill(row + qb, row + w, clamp ? row[qb - 1] : boundary_value);
                }
                // next row
                std::size_t r = rank - 1;
                while (r-- > 0)
                {
                    if (++p[r] < he[r])
                    {
                        break;
                    }
                    p[r] = 0;
                }
                if (r == static_cast<std::size_t>(-1))
                {
                    break;
                }
            }

            // The kernel, on a copy of the tile (its padding kept), then the
            // tile written back: as one constant when its cells are one value.
            {
                auto& c = cache.fetch(tc);
                if (c.uniform)
                {
                    std::fill(outb.begin(), outb.end(), c.cst);
                }
                else
                {
                    std::copy_n(c.buf.data(), N, outb.data());
                }
            }
            on_block(t, halo_block<T, rank, Index>(halo.data() + centre, hs, h), make_block<Dst, T>(dst, outb.data()));
            const T                 v0      = outb[0];
            bool                    uniform = true;
            std::array<Index, rank> lo{};
            for_each_row(lo, t.extent,
                         [&](const std::array<Index, rank>& i)
                         {
                             if (!uniform)
                             {
                                 return;
                             }
                             std::size_t o = 0;
                             for (std::size_t r = 0; r < rank; ++r)
                             {
                                 o += static_cast<std::size_t>(i[r]) * ts[r];
                             }
                             const T* x = outb.data() + o;
                             uniform    = std::all_of(x, x + t.extent[rank - 1], [&v0](const T& y) { return y == v0; });
                         });
            const std::size_t s = tile_storage(dst, t);
            if (uniform)
            {
                out.fill(s, s + N, v0);
            }
            else
            {
                out.copy(s, s + N, outb.data());
            }
        });
}
} // namespace detail

/**
 * @brief Apply a stencil tile by tile: tiles [@p first, @p last) of @p dst
 *        computed from @p src read with a halo of @p h cells.
 *
 * @details For each tile, the tile of @p src and the @p h cells around it --
 *          read from its neighbour tiles, each gathered once along a row of
 *          tiles (a uniform one costs its constant) -- go to the kernel as a
 *          halo_block; cells outside the view hold the boundary's value
 *          (constant_boundary<T>{v}, or clamp_boundary{}: the nearest cell of
 *          the view). The kernel writes the tile of @p dst in a buffer that
 *          starts as a copy of the source tile; it is written back in one go,
 *          as one constant when its cells inside the view are one value.
 *
 * @param on_block   on_block(const tile& t, halo_block in, block out): in[i, j]
 *                   for i in [-h, te + h), out an mdspan<T> over the tile
 *                   (padding included: loop to t.extent)
 * @param on_uniform on_uniform(const tile& t, const T& v, T& out) -> bool, for
 *                   a tile whose cells, halo and border are all v, in O(1):
 *                   set out and return true (the tile of @p dst becomes out),
 *                   or return false to get on_block. Omitted: always on_block.
 * @pre   @p src and @p dst have the same extents and tiles, and are two
 *        arrays (checked: a mismatch stops the program); 0 <= h <= a tile's
 *        extent in every dimension (checked). Inside a parallel region, in a
 *        phase, a tile whole chunks, and no thread writes @p src meanwhile.
 * @warning Experimental until 1.0 (see the file).
 */
template <tiled_view_of_array Src,
          tiled_view_of_array Dst,
          typename Boundary,
          typename OnBlock,
          typename OnUniform = detail::no_uniform>
    requires(!std::is_const_v<typename Dst::element_type>)
void stencil_tiles(const Src&               src,
                   const Dst&               dst,
                   typename Src::index_type h,
                   const Boundary&          bnd,
                   typename Src::index_type first,
                   typename Src::index_type last,
                   OnBlock&&                on_block,
                   OnUniform&&              on_uniform = {})
{
    detail::stencil_on(src, dst, h, bnd, "md::stencil_tiles", detail::tile_range(src, first, last), on_block,
                       on_uniform);
}

/// @brief stencil_tiles() over every tile.
template <tiled_view_of_array Src,
          tiled_view_of_array Dst,
          typename Boundary,
          typename OnBlock,
          typename OnUniform = detail::no_uniform>
    requires(!std::is_const_v<typename Dst::element_type>)
void stencil_tiles(const Src&               src,
                   const Dst&               dst,
                   typename Src::index_type h,
                   const Boundary&          bnd,
                   OnBlock&&                on_block,
                   OnUniform&&              on_uniform = {})
{
    detail::stencil_on(src, dst, h, bnd, "md::stencil_tiles", detail::tile_range(src, 0, tile_count(src)), on_block,
                       on_uniform);
}

/**
 * @brief stencil_tiles() called by every thread of a parallel region, in a
 *        phase: the tiles dealt to the team as for_each_tile_mt() deals them
 *        (thread t of T on tiles [n t / T, n (t + 1) / T), consecutive: the
 *        neighbour tiles each thread gathers are mostly its own), no
 *        `omp for`. Outside a region, every tile. No barrier: @p dst is
 *        complete at the region's next barrier, which a next step that reads
 *        it needs.
 * @warning stencil_tiles()'s; every thread of the team calls it.
 */
template <tiled_view_of_array Src,
          tiled_view_of_array Dst,
          typename Boundary,
          typename OnBlock,
          typename OnUniform = detail::no_uniform>
    requires(!std::is_const_v<typename Dst::element_type>)
void stencil_tiles_mt(const Src&               src,
                      const Dst&               dst,
                      typename Src::index_type h,
                      const Boundary&          bnd,
                      OnBlock&&                on_block,
                      OnUniform&&              on_uniform = {})
{
    detail::stencil_on(src, dst, h, bnd, "md::stencil_tiles_mt", detail::dealt_tiles(src), on_block, on_uniform);
}

namespace detail
{
/// @brief fill_box() of a tiled view on the tiles @p walk(lo, hi, fn) gives,
'''),
    ('docs/mdspan.md',
     r'''same field summed element by element, 1 thread, reads at 2.8 ns per element through `a[k]`,
3.3 ns through `f[i, j]` row-major and 3.5 ns through `f[i, j]` in tiles (the tiled mapping
divides by its constant tile extents in unsigned arithmetic: shifts and masks).

### A box in and out of contiguous memory: `md::copy_box_out`, `md::copy_box_in`

''',
     r'''same field summed element by element, 1 thread, reads at 2.8 ns per element through `a[k]`,
3.3 ns through `f[i, j]` row-major and 3.5 ns through `f[i, j]` in tiles (the tiled mapping
divides by its constant tile extents in unsigned arithmetic: shifts and masks).

### Stencils: `md::stencil_tiles`

A stencil reads a cell's neighbours, which near a tile's edge lie in another tile. `md::stencil_tiles`
hands the kernel the tile and `h` cells around it, read in one go from the neighbour tiles (each
gathered once along a row of tiles, a uniform one costing its constant), and writes the result
tile of another array back in one go.

```cpp
auto src = sa::md::tiled_view<16, 16>(a, nx, ny);   // read
auto dst = sa::md::tiled_view<16, 16>(b, nx, ny);   // written: another array
sa::parallel_phase([&] {
    sa::md::stencil_tiles_mt(std::as_const(src), dst, 1, sa::md::clamp_boundary{},
        [](const auto& t, const auto& in, auto out) {    // in[i, j], i in [-1, 17)
            for (long i = 0; i < t.extent[0]; ++i)
                for (long j = 0; j < t.extent[1]; ++j)
                    out[i, j] = in[i, j] + 0.2 * (in[i - 1, j] + in[i + 1, j] + in[i, j - 1]
                                                  + in[i, j + 1] - 4.0 * in[i, j]);
        },
        [](const auto&, const double& v, double& out) { out = v; return true; });  // optional
});
```

| call | does |
|---|---|
| `md::stencil_tiles(src, dst, h, boundary, on_block, on_uniform)` | for each tile: `on_block(t, in, out)`, `in` a `md::halo_block` (`in[i, j]` for `i` in `[-h, te + h)`, `(0, 0)` the tile's first cell), `out` an `mdspan<T>` over the tile of `dst`, starting as a copy of the source tile. The tile is written back as one constant when its cells inside the view are one value, else copied |
| `on_uniform(t, v, out) -> bool` | optional: for a tile whose cells, halo and border are all `v`, in O(1) -- set `out` and return true, the tile of `dst` becomes `out` (no halo read, no kernel); false gets `on_block` |
| `md::constant_boundary<T>{v}`, `md::clamp_boundary{}` | the halo cells outside the view: `v` (Dirichlet), or the nearest cell of the view (zero gradient) |
| `md::stencil_tiles(src, dst, h, boundary, first, last, ...)` | tiles `[first, last)`, for an `omp for` or a slice per thread |
| `md::stencil_tiles_mt(src, dst, h, boundary, ...)` | called by every thread of a region, in a phase: the tiles dealt as `for_each_tile_mt` deals them (consecutive: a thread's neighbour tiles are mostly its own), no `omp for`, no barrier -- the next step, which reads `dst`, needs the region's next barrier |

`src` and `dst` have the same extents and tiles and are two arrays, and `0 <= h <=` a tile's extent
in every dimension: checked, a mismatch stops the program (a stencil in place would read cells its
other tiles have already written). Any rank: `in[i, j, k]` in 3D. Inside a region the rules of the
tile operations hold: a tile whole chunks, `dst` written in a phase, `src` written by no one.

`examples/mdspan_tiles`, one explicit diffusion step (5 points, clamped border) on the 2048 x 2048
field (85 % of its tiles uniform) into another array, against the element loop (`f[i, j]` and its
four neighbours, the border by `min` / `max`), Release, GCC 14, 2-core container, several runs:

| threads | element by element | `stencil_tiles_mt` | with `on_uniform` |
|---|---|---|---|
| 1 | 119 to 181 ms | 15.8 to 18.2 ms (x 7.4 to 10.4) | 7.4 to 7.9 ms (x 15.9 to 22.9) |
| 2 | 63 to 92 ms | 9.7 to 12.6 ms (x 6.5 to 7.3) | 5.3 to 8.5 ms (x 10.6 to 12.0) |

All three leave the same field, with the same 2561 materialised chunks.

### A box in and out of contiguous memory: `md::copy_box_out`, `md::copy_box_in`

'''),
    ('docs/mdspan.md',
     r'''|---|---|
| `chunk/public/chunk/mdspan.h` | `md::mdspan_source`, `md::accessor`, `md::handle`, `md::layout_tiled` (and its `submdspan_mapping`), `md::layout_tiled_dyn`, `md::view`, `md::tiled_view`, `md::tiled_view_dyn`, `md::tiled_size`, `md::tiled_size_dyn`, `md::tile_shape`, `md::submdspan` |
| `examples/validation/mdspan_validation.cpp` | writes through the view and linear read-back, compound operators, rank 3, read-only views, the tiled mapping (bijection, padding), a uniform tile is a uniform run; every variant (one ctest each, `mdspan_validation_<variant>`) |
| `chunk/public/chunk/mdspan_tile_ops.h` | `md::for_each_tile(_mt)`, `md::update_tiles(_mt)`, `md::fill_box(_mt)`, `md::copy_box_out(_mt)`, `md::copy_box_in(_mt)`, `md::tile_count`, `md::tile` |
| `examples/validation/mdspan_tile_ops_validation.cpp` | blocks, tile extents on padded edges, uniform tiles, the update against the element formula, untouched tiles stay compressed, the parallel update; the `_mt` forms (each tile of the view or of a box visited once and no other, a thread's tiles within one of the others' count, `fill_box_mt` on tiled and row-major views, in a phase and outside any region); every variant at 1 and 8 threads, and `mdspan_tile_ops_misaligned` (a tile smaller than a chunk refused in a parallel region) |
| `examples/validation/mdspan_copy_box_validation.cpp` | `copy_box_out` / `copy_box_in` against `f[i, j]` and a reference: boxes aligned, unaligned, on the padded edges, one row, one column, the whole view, empty; tiled (a tile per chunk, 16 x 16), row-major, rank 3, a `submdspan` and a column; whole tiles copied in uniform left compressed; tile by tile and row slice by row slice in a parallel phase; the `_mt` forms on every box and view, and their deal (each tile of the box to one thread, no thread above W / T plus a tile, strided shares a partition with their bounds on chunk boundaries); every variant at 1 and 8 threads |
| `examples/validation/mdspan_submdspan_validation.cpp` | sub-views of a row-major view (block, row, column, read-only, `fill_box`) and of a tiled view (aligned, ending at the extent, a sub-view of a sub-view; the tile operations on it, serially and in a parallel phase); every variant at 1 and 8 threads, and `mdspan_submdspan_misaligned` (a tiled slice off the tiles stops) |
| `examples/validation/mdspan_tiled_dyn_validation.cpp` | `tile_shape` (2D, 3D, every multiple of 64 to 4096), `layout_tiled_dyn` against `layout_tiled` index by index and through their sub-views, `tiled_view_dyn`'s tile per chunk, the tile operations on a tile per chunk, a 3 x 5 tile and one beyond the inline buffer; every variant at the default size and at dyn (a re-chunk under the view), 1 and 8 threads |
| `examples/mdspan_tiles/mdspan_tiles.cpp` | the comparisons of sections 2 and 3, and the cost of the multi-index; `mdspan_tiles 4096` for a larger field |
''',
     r'''|---|---|
| `chunk/public/chunk/mdspan.h` | `md::mdspan_source`, `md::accessor`, `md::handle`, `md::layout_tiled` (and its `submdspan_mapping`), `md::layout_tiled_dyn`, `md::view`, `md::tiled_view`, `md::tiled_view_dyn`, `md::tiled_size`, `md::tiled_size_dyn`, `md::tile_shape`, `md::submdspan` |
| `examples/validation/mdspan_validation.cpp` | writes through the view and linear read-back, compound operators, rank 3, read-only views, the tiled mapping (bijection, padding), a uniform tile is a uniform run; every variant (one ctest each, `mdspan_validation_<variant>`) |
| `chunk/public/chunk/mdspan_tile_ops.h` | `md::for_each_tile(_mt)`, `md::update_tiles(_mt)`, `md::fill_box(_mt)`, `md::copy_box_out(_mt)`, `md::copy_box_in(_mt)`, `md::stencil_tiles(_mt)`, `md::halo_block`, `md::constant_boundary`, `md::clamp_boundary`, `md::tile_count`, `md::tile` |
| `examples/validation/mdspan_tile_ops_validation.cpp` | blocks, tile extents on padded edges, uniform tiles, the update against the element formula, untouched tiles stay compressed, the parallel update; the `_mt` forms (each tile of the view or of a box visited once and no other, a thread's tiles within one of the others' count, `fill_box_mt` on tiled and row-major views, in a phase and outside any region); every variant at 1 and 8 threads, and `mdspan_tile_ops_misaligned` (a tile smaller than a chunk refused in a parallel region) |
| `examples/validation/mdspan_copy_box_validation.cpp` | `copy_box_out` / `copy_box_in` against `f[i, j]` and a reference: boxes aligned, unaligned, on the padded edges, one row, one column, the whole view, empty; tiled (a tile per chunk, 16 x 16), row-major, rank 3, a `submdspan` and a column; whole tiles copied in uniform left compressed; tile by tile and row slice by row slice in a parallel phase; the `_mt` forms on every box and view, and their deal (each tile of the box to one thread, no thread above W / T plus a tile, strided shares a partition with their bounds on chunk boundaries); every variant at 1 and 8 threads |
| `examples/validation/mdspan_stencil_validation.cpp` | `stencil_tiles` / `stencil_tiles_mt` against the stencil through `f[i, j]`: 9 points (halo 1) and 13 (halo 2), constant and clamped borders, tiles of 16 x 16 and of a chunk (padded edges), a halo as wide as a tile, rank 3; the same result with and without `on_uniform`, the uniform tiles of the result holding no buffer, the source untouched; serial, per tile in an `omp for`, from every thread of a phase and outside any region; every variant at 1 and 8 threads, and `mdspan_stencil_alias` (a stencil from an array into itself refused) |
| `examples/validation/mdspan_submdspan_validation.cpp` | sub-views of a row-major view (block, row, column, read-only, `fill_box`) and of a tiled view (aligned, ending at the extent, a sub-view of a sub-view; the tile operations on it, serially and in a parallel phase); every variant at 1 and 8 threads, and `mdspan_submdspan_misaligned` (a tiled slice off the tiles stops) |
| `examples/validation/mdspan_tiled_dyn_validation.cpp` | `tile_shape` (2D, 3D, every multiple of 64 to 4096), `layout_tiled_dyn` against `layout_tiled` index by index and through their sub-views, `tiled_view_dyn`'s tile per chunk, the tile operations on a tile per chunk, a 3 x 5 tile and one beyond the inline buffer; every variant at the default size and at dyn (a re-chunk under the view), 1 and 8 threads |
| `examples/mdspan_tiles/mdspan_tiles.cpp` | the comparisons of sections 2 and 3, and the cost of the multi-index; `mdspan_tiles 4096` for a larger field |
'''),
    ('docs/mdspan_fr.md',
     r'''le même champ sommé élément par élément, 1 thread, se lit à 2,8 ns par élément par `a[k]`,
3,3 ns par `f[i, j]` ligne par ligne et 3,5 ns par `f[i, j]` en tuiles (le mapping en tuiles
divise par ses étendues de tuile constantes en arithmétique non signée : décalages et masques).

### Un rectangle vers et depuis une mémoire contiguë : `md::copy_box_out`, `md::copy_box_in`

''',
     r'''le même champ sommé élément par élément, 1 thread, se lit à 2,8 ns par élément par `a[k]`,
3,3 ns par `f[i, j]` ligne par ligne et 3,5 ns par `f[i, j]` en tuiles (le mapping en tuiles
divise par ses étendues de tuile constantes en arithmétique non signée : décalages et masques).

### Stencils : `md::stencil_tiles`

Un stencil lit les voisins d'une cellule, qui près du bord d'une tuile sont dans une autre tuile.
`md::stencil_tiles` donne au noyau la tuile et `h` cellules autour, lues d'un bloc dans les tuiles
voisines (chacune lue une fois le long d'une rangée de tuiles, une uniforme coûtant sa constante),
et réécrit d'un bloc la tuile résultat d'un autre tableau.

```cpp
auto src = sa::md::tiled_view<16, 16>(a, nx, ny);   // lu
auto dst = sa::md::tiled_view<16, 16>(b, nx, ny);   // écrit : un autre tableau
sa::parallel_phase([&] {
    sa::md::stencil_tiles_mt(std::as_const(src), dst, 1, sa::md::clamp_boundary{},
        [](const auto& t, const auto& in, auto out) {    // in[i, j], i dans [-1, 17)
            for (long i = 0; i < t.extent[0]; ++i)
                for (long j = 0; j < t.extent[1]; ++j)
                    out[i, j] = in[i, j] + 0.2 * (in[i - 1, j] + in[i + 1, j] + in[i, j - 1]
                                                  + in[i, j + 1] - 4.0 * in[i, j]);
        },
        [](const auto&, const double& v, double& out) { out = v; return true; });  // facultatif
});
```

| appel | effet |
|---|---|
| `md::stencil_tiles(src, dst, h, boundary, on_block, on_uniform)` | pour chaque tuile : `on_block(t, in, out)`, `in` un `md::halo_block` (`in[i, j]` pour `i` dans `[-h, te + h)`, `(0, 0)` la première cellule de la tuile), `out` un `mdspan<T>` sur la tuile de `dst`, initialisé à une copie de la tuile source. La tuile est réécrite comme une constante quand ses cellules dans la vue ont une seule valeur, sinon copiée |
| `on_uniform(t, v, out) -> bool` | facultatif : pour une tuile dont les cellules, le halo et le bord valent tous `v`, en O(1) -- fixer `out` et renvoyer true, la tuile de `dst` devient `out` (ni lecture du halo ni noyau) ; false donne `on_block` |
| `md::constant_boundary<T>{v}`, `md::clamp_boundary{}` | les cellules du halo hors de la vue : `v` (Dirichlet), ou la cellule de la vue la plus proche (gradient nul) |
| `md::stencil_tiles(src, dst, h, boundary, first, last, ...)` | tuiles `[first, last)`, pour un `omp for` ou une tranche par thread |
| `md::stencil_tiles_mt(src, dst, h, boundary, ...)` | appelée par tous les threads d'une région, dans une phase : les tuiles réparties comme par `for_each_tile_mt` (consécutives : les tuiles voisines d'un thread sont surtout les siennes), sans `omp for`, sans barrière -- le pas suivant, qui lit `dst`, demande la barrière suivante de la région |

`src` et `dst` ont les mêmes étendues et les mêmes tuiles et sont deux tableaux, et `0 <= h <=`
l'étendue d'une tuile dans chaque dimension : vérifié, un écart arrête le programme (un stencil en
place lirait des cellules que ses autres tuiles ont déjà écrites). Tout rang : `in[i, j, k]` en 3D.
Dans une région, les règles des opérations sur tuiles s'appliquent : une tuile faite de chunks
entiers, `dst` écrit dans une phase, `src` écrit par personne.

`examples/mdspan_tiles`, un pas de diffusion explicite (5 points, bord par recopie) sur le champ de
2048 x 2048 (85 % de ses tuiles uniformes) vers un autre tableau, contre la boucle élément par
élément (`f[i, j]` et ses quatre voisins, le bord par `min` / `max`), Release, GCC 14, conteneur à
2 cœurs, plusieurs passages :

| threads | élément par élément | `stencil_tiles_mt` | avec `on_uniform` |
|---|---|---|---|
| 1 | 119 à 181 ms | 15,8 à 18,2 ms (x 7,4 à 10,4) | 7,4 à 7,9 ms (x 15,9 à 22,9) |
| 2 | 63 à 92 ms | 9,7 à 12,6 ms (x 6,5 à 7,3) | 5,3 à 8,5 ms (x 10,6 à 12,0) |

Les trois laissent le même champ, avec les mêmes 2561 chunks matérialisés.

### Un rectangle vers et depuis une mémoire contiguë : `md::copy_box_out`, `md::copy_box_in`

'''),
    ('docs/mdspan_fr.md',
     r'''|---|---|
| `chunk/public/chunk/mdspan.h` | `md::mdspan_source`, `md::accessor`, `md::handle`, `md::layout_tiled` (et son `submdspan_mapping`), `md::layout_tiled_dyn`, `md::view`, `md::tiled_view`, `md::tiled_view_dyn`, `md::tiled_size`, `md::tiled_size_dyn`, `md::tile_shape`, `md::submdspan` |
| `examples/validation/mdspan_validation.cpp` | écritures par la vue et relecture linéaire, opérateurs composés, rang 3, vues en lecture seule, le mapping en tuiles (bijection, bourrage), une tuile uniforme est une suite uniforme ; toutes les variantes (un ctest chacune, `mdspan_validation_<variant>`) |
| `chunk/public/chunk/mdspan_tile_ops.h` | `md::for_each_tile(_mt)`, `md::update_tiles(_mt)`, `md::fill_box(_mt)`, `md::copy_box_out(_mt)`, `md::copy_box_in(_mt)`, `md::tile_count`, `md::tile` |
| `examples/validation/mdspan_tile_ops_validation.cpp` | blocs, étendues des tuiles sur les bords bourrés, tuiles uniformes, la mise à jour contre la formule élément par élément, les tuiles non touchées restent compressées, la mise à jour en parallèle ; les formes `_mt` (chaque tuile de la vue ou d'un rectangle visitée une fois et aucune autre, les tuiles d'un thread à une près de celles des autres, `fill_box_mt` sur vues en tuiles et ligne par ligne, dans une phase et hors région) ; toutes les variantes à 1 et 8 threads, et `mdspan_tile_ops_misaligned` (une tuile plus petite qu'un chunk refusée dans une région parallèle) |
| `examples/validation/mdspan_copy_box_validation.cpp` | `copy_box_out` / `copy_box_in` contre `f[i, j]` et une référence : rectangles alignés, non alignés, sur les bords bourrés, une ligne, une colonne, toute la vue, vide ; en tuiles (une tuile par chunk, 16 x 16), ligne par ligne, rang 3, une `submdspan` et une colonne ; des tuiles entières copiées uniformes restent compressées ; tuile par tuile et tranche de lignes par tranche dans une phase parallèle ; les formes `_mt` sur chaque rectangle et chaque vue, et leur partage (chaque tuile du rectangle à un seul thread, aucun thread au-delà de W / T plus une tuile, parts des vues à pas formant une partition, bornes sur des frontières de chunk) ; toutes les variantes à 1 et 8 threads |
| `examples/validation/mdspan_submdspan_validation.cpp` | sous-vues d'une vue ligne par ligne (bloc, ligne, colonne, lecture seule, `fill_box`) et d'une vue en tuiles (alignée, finissant à l'étendue, sous-vue d'une sous-vue ; les opérations sur tuiles dessus, en série et en phase parallèle) ; toutes les variantes à 1 et 8 threads, et `mdspan_submdspan_misaligned` (une tranche en tuiles hors des tuiles arrête le programme) |
| `examples/validation/mdspan_tiled_dyn_validation.cpp` | `tile_shape` (2D, 3D, tous les multiples de 64 jusqu'à 4096), `layout_tiled_dyn` contre `layout_tiled` indice par indice et par leurs sous-vues, la tuile par chunk de `tiled_view_dyn`, les opérations sur tuiles sur une tuile par chunk, une tuile 3 x 5 et une au-delà du tampon local ; toutes les variantes à la taille par défaut et en dyn (un rechunk sous la vue), 1 et 8 threads |
| `examples/mdspan_tiles/mdspan_tiles.cpp` | les comparaisons des sections 2 et 3, et le coût du multi-indice ; `mdspan_tiles 4096` pour un champ plus grand |
''',
     r'''|---|---|
| `chunk/public/chunk/mdspan.h` | `md::mdspan_source`, `md::accessor`, `md::handle`, `md::layout_tiled` (et son `submdspan_mapping`), `md::layout_tiled_dyn`, `md::view`, `md::tiled_view`, `md::tiled_view_dyn`, `md::tiled_size`, `md::tiled_size_dyn`, `md::tile_shape`, `md::submdspan` |
| `examples/validation/mdspan_validation.cpp` | écritures par la vue et relecture linéaire, opérateurs composés, rang 3, vues en lecture seule, le mapping en tuiles (bijection, bourrage), une tuile uniforme est une suite uniforme ; toutes les variantes (un ctest chacune, `mdspan_validation_<variant>`) |
| `chunk/public/chunk/mdspan_tile_ops.h` | `md::for_each_tile(_mt)`, `md::update_tiles(_mt)`, `md::fill_box(_mt)`, `md::copy_box_out(_mt)`, `md::copy_box_in(_mt)`, `md::stencil_tiles(_mt)`, `md::halo_block`, `md::constant_boundary`, `md::clamp_boundary`, `md::tile_count`, `md::tile` |
| `examples/validation/mdspan_tile_ops_validation.cpp` | blocs, étendues des tuiles sur les bords bourrés, tuiles uniformes, la mise à jour contre la formule élément par élément, les tuiles non touchées restent compressées, la mise à jour en parallèle ; les formes `_mt` (chaque tuile de la vue ou d'un rectangle visitée une fois et aucune autre, les tuiles d'un thread à une près de celles des autres, `fill_box_mt` sur vues en tuiles et ligne par ligne, dans une phase et hors région) ; toutes les variantes à 1 et 8 threads, et `mdspan_tile_ops_misaligned` (une tuile plus petite qu'un chunk refusée dans une région parallèle) |
| `examples/validation/mdspan_copy_box_validation.cpp` | `copy_box_out` / `copy_box_in` contre `f[i, j]` et une référence : rectangles alignés, non alignés, sur les bords bourrés, une ligne, une colonne, toute la vue, vide ; en tuiles (une tuile par chunk, 16 x 16), ligne par ligne, rang 3, une `submdspan` et une colonne ; des tuiles entières copiées uniformes restent compressées ; tuile par tuile et tranche de lignes par tranche dans une phase parallèle ; les formes `_mt` sur chaque rectangle et chaque vue, et leur partage (chaque tuile du rectangle à un seul thread, aucun thread au-delà de W / T plus une tuile, parts des vues à pas formant une partition, bornes sur des frontières de chunk) ; toutes les variantes à 1 et 8 threads |
| `examples/validation/mdspan_stencil_validation.cpp` | `stencil_tiles` / `stencil_tiles_mt` contre le stencil par `f[i, j]` : 9 points (halo 1) et 13 (halo 2), bords constants et par recopie, tuiles de 16 x 16 et d'un chunk (bords bourrés), un halo aussi large qu'une tuile, rang 3 ; le même résultat avec et sans `on_uniform`, les tuiles uniformes du résultat sans buffer, la source intacte ; en série, par tuile dans un `omp for`, depuis tous les threads d'une phase et hors région ; toutes les variantes à 1 et 8 threads, et `mdspan_stencil_alias` (un stencil d'un tableau vers lui-même refusé) |
| `examples/validation/mdspan_submdspan_validation.cpp` | sous-vues d'une vue ligne par ligne (bloc, ligne, colonne, lecture seule, `fill_box`) et d'une vue en tuiles (alignée, finissant à l'étendue, sous-vue d'une sous-vue ; les opérations sur tuiles dessus, en série et en phase parallèle) ; toutes les variantes à 1 et 8 threads, et `mdspan_submdspan_misaligned` (une tranche en tuiles hors des tuiles arrête le programme) |
| `examples/validation/mdspan_tiled_dyn_validation.cpp` | `tile_shape` (2D, 3D, tous les multiples de 64 jusqu'à 4096), `layout_tiled_dyn` contre `layout_tiled` indice par indice et par leurs sous-vues, la tuile par chunk de `tiled_view_dyn`, les opérations sur tuiles sur une tuile par chunk, une tuile 3 x 5 et une au-delà du tampon local ; toutes les variantes à la taille par défaut et en dyn (un rechunk sous la vue), 1 et 8 threads |
| `examples/mdspan_tiles/mdspan_tiles.cpp` | les comparaisons des sections 2 et 3, et le coût du multi-indice ; `mdspan_tiles 4096` pour un champ plus grand |
'''),
    ('examples/mdspan_tiles/mdspan_tiles.cpp',
     r''' *          md::copy_box_out_mt / md::copy_box_in_mt (the band's tiles dealt
 *          by weight), the band filled (fill_box_mt) and a sum over every
 *          tile (for_each_tile_mt); the rectangles are painted a third time
 *          with fill_box_mt, a barrier between two.
 *
 *          Usage: mdspan_tiles [n]   (an n x n field, 2048 by default)
 *          Exit 0, 1 when a layout does not read back the field, 77 when
''',
     r''' *          md::copy_box_out_mt / md::copy_box_in_mt (the band's tiles dealt
 *          by weight), the band filled (fill_box_mt) and a sum over every
 *          tile (for_each_tile_mt); the rectangles are painted a third time
 *          with fill_box_mt, a barrier between two. Then one diffusion step,
 *          a 5-point stencil, element by element against
 *          md::stencil_tiles_mt, without and with its uniform shortcut.
 *
 *          Usage: mdspan_tiles [n]   (an n x n field, 2048 by default)
 *          Exit 0, 1 when a layout does not read back the field, 77 when
'''),
    ('examples/mdspan_tiles/mdspan_tiles.cpp',
     r'''#include "chunk/manage_ompt.h"    // parallel_phase
#include "chunk/mdspan.h"         // shrink_array::md, SHRINK_ARRAY_HAS_MDSPAN
#include "chunk/PMP_CHUNK.h"      // CHUNK_PMP_THREAD_RANK, CHUNK_PMP_NUM_THREADS
#include "chunk/mdspan_tile_ops.h" // md::for_each_tile(_mt), md::update_tiles, md::fill_box(_mt), md::copy_box_out(_mt), md::copy_box_in(_mt)
#include "chunk/param_chunk.h" // ParamChunk::init_indexes

namespace sa = shrink_array;
''',
     r'''#include "chunk/manage_ompt.h"    // parallel_phase
#include "chunk/mdspan.h"         // shrink_array::md, SHRINK_ARRAY_HAS_MDSPAN
#include "chunk/PMP_CHUNK.h"      // CHUNK_PMP_THREAD_RANK, CHUNK_PMP_NUM_THREADS
#include "chunk/mdspan_tile_ops.h" // md::for_each_tile(_mt), md::update_tiles, md::fill_box(_mt), md::copy_box_*(_mt), md::stencil_tiles_mt
#include "chunk/param_chunk.h" // ParamChunk::init_indexes

namespace sa = shrink_array;
'''),
    ('examples/mdspan_tiles/mdspan_tiles.cpp',
     r'''}
} // namespace

// The price of the multi-index: one sum of the same field read as a[k], as
// f[i, j] row-major and as f[i, j] in tiles, element by element (the const
// operator[] every time), best of 5. The tiled mapping divides by constant
''',
     r'''}
} // namespace

// One explicit diffusion step, a 5-point stencil with a clamped border, from
// the field into another array: element by element (f[i, j] and its four
// neighbours, rows shared by the team) against md::stencil_tiles_mt (a tile
// and its halo of 1 read in one go), without and with the uniform shortcut
// (a tile whose neighbourhood is one value stays that value, in O(1)). Best
// of 3.
void stencil(long n, int& bad)
{
    const long               size = static_cast<long>(sa::md::tiled_size<16, 16>(n, n));
    sa::ArrayChunk2A<double> src(size);
    fill_tiled<16, 16>(src, n);
    const auto cs = sa::md::view_type<const sa::ArrayChunk2A<double>, sa::md::layout_tiled<16, 16>>(
        sa::md::tiled_view<16, 16>(src, n, n));
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

    sa::ArrayChunk2A<double> e(size);
    auto                     fe     = sa::md::tiled_view<16, 16>(e, n, n);
    const double             t_elem = best(
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
    sa::ArrayChunk2A<double> b(size);
    auto                     fb     = sa::md::tiled_view<16, 16>(b, n, n);
    const double             t_tile = best(
        [&] { sa::parallel_phase([&] { sa::md::stencil_tiles_mt(cs, fb, 1L, sa::md::clamp_boundary{}, kernel); }); });
    sa::ArrayChunk2A<double> u(size);
    auto                     fu     = sa::md::tiled_view<16, 16>(u, n, n);
    const double             t_unif = best(
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
            const double x = std::as_const(fe)[i, j];
            wrong += (std::abs(std::as_const(fb)[i, j] - x) > 1e-12 || std::abs(std::as_const(fu)[i, j] - x) > 1e-12)
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
'''),
    ('examples/mdspan_tiles/mdspan_tiles.cpp',
     r'''    paint(n, bad);
    box_copy(n, bad);
    box_copy_team(n, bad);
    index_cost(n, bad);
    sa::release_chunk_pools();
    sa::destroy_block_chunk();
''',
     r'''    paint(n, bad);
    box_copy(n, bad);
    box_copy_team(n, bad);
    stencil(n, bad);
    index_cost(n, bad);
    sa::release_chunk_pools();
    sa::destroy_block_chunk();
'''),
    ('examples/validation/CMakeLists.txt',
     r'''  endif()
endforeach()

# shrink-build-stamp: every executable of this directory prints the build stamp
# and is built by `check` (cmake/shrink_build_stamp.cmake).
shrink_stamp_directory_targets()
''',
     r'''  endif()
endforeach()

# --- mdspan_stencil_validation_<variant>: md::stencil_tiles / md::stencil_tiles_mt
# (chunk/mdspan_tile_ops.h) against the stencil through f[i, j], serially, per
# tile in an omp for and from every thread of a phase, at 1 and 8 threads.
# mdspan_stencil_alias: a stencil from an array into itself must stop the
# program (expect_stop.cmake).
foreach(_vi RANGE ${_rt_nvar_last})
  list(GET _rt_variant_names   ${_vi} _vname)
  list(GET _rt_variant_libbase ${_vi} _libbase)
  list(GET _rt_variant_macro   ${_vi} _vmacro)
  if (TARGET ${_libbase}_${CHUNK_DEFAULT_SIZE})
    set(_tgt mdspan_stencil_validation_${_vname})
    add_executable(${_tgt} mdspan_stencil_validation.cpp)
    target_link_libraries(${_tgt} PRIVATE ${_libbase}_${CHUNK_DEFAULT_SIZE})
    validation_variant_defs(${_tgt} ${_vmacro} ${CHUNK_DEFAULT_SIZE})
    validation_setup_target(${_tgt})
    validation_add_test(${_tgt} ${_tgt} "" "1;8")
    foreach(_nthr 1 8)
      set_property(TEST ${_tgt}_${_nthr}thr PROPERTY SKIP_RETURN_CODE 77)
    endforeach()
    if (_vname STREQUAL "zero_any")
      add_test(NAME mdspan_stencil_alias
               COMMAND ${CMAKE_COMMAND} -DEXE=$<TARGET_FILE:${_tgt}> -DARGS=alias
                       "-DEXPECT=the source and the destination are one array" -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_stop.cmake)
      set_tests_properties(mdspan_stencil_alias PROPERTIES LABELS validation)
    endif()
  endif()
endforeach()

# shrink-build-stamp: every executable of this directory prints the build stamp
# and is built by `check` (cmake/shrink_build_stamp.cmake).
shrink_stamp_directory_targets()
'''),
]

NEW_FILES = {
    'examples/validation/mdspan_stencil_validation.cpp': r'''/*
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
//     of the result left compressed (every variant at first_value, zero_any::Chunk at
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
#include <utility>     // std::as_const
#include <vector>      // std::vector (the references)

#include "chunk/zero_array.h"     // ArrayChunk
#include "chunk/zero_one_value_array.h"    // ArrayChunk2
#include "chunk/zero_any_array.h"   // ArrayChunk2A
#include "chunk/zero_chunk.h"       // release_chunk_pools, destroy_block_chunk
#include "chunk/config.h"          // chunk_config::use_zero_one_value*
#include "chunk/internal/print.h"  // shrink_array::println
#include "chunk/manage_ompt.h"     // parallel_phase
#include "chunk/mdspan_tile_ops.h" // md::stencil_tiles(_mt), SHRINK_ARRAY_HAS_MDSPAN
#include "chunk/param_chunk.h"     // ParamChunk::init_indexes

using namespace shrink_array;

#if SHRINK_ARRAY_HAS_MDSPAN

namespace
{
template <typename T>
using Array = std::conditional_t<chunk_config::use_zero_any,
                                 ArrayChunk2A<T>,
                                 std::conditional_t<chunk_config::use_zero_one_value, ArrayChunk2<T>, ArrayChunk<T>>>;

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

enum class border
{
    zero,
    value,
    clamp,
};
constexpr double border_value = 2.5;

// The reference: the stencil through f[i, j] on a vector, the border applied.
std::vector<double> reference(const std::vector<double>& field, long h, border b)
{
    std::vector<double> out(field.size());
    const auto          at = [&](long i, long j)
    {
        if (b == border::clamp)
        {
            i = std::clamp(i, 0L, nx - 1);
            j = std::clamp(j, 0L, ny - 1);
        }
        else if (i < 0 || i >= nx || j < 0 || j >= ny)
        {
            return b == border::zero ? 0.0 : border_value;
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

enum class how
{
    serial,
    range,
    team,
    team_outside,
};

template <typename Src, typename Dst, typename B, typename... U>
void run_stencil(how w, const Src& src, const Dst& dst, long h, const B& bnd, U&&... uniform)
{
    const auto kernel = kernel_for(h);
    switch (w)
    {
        case how::serial:
            md::stencil_tiles(src, dst, h, bnd, kernel, uniform...);
            break;
        case how::range:
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
        case how::team:
            parallel_phase([&] { md::stencil_tiles_mt(src, dst, h, bnd, kernel, uniform...); });
            break;
        case how::team_outside:
            md::stencil_tiles_mt(src, dst, h, bnd, kernel, uniform...);
            break;
    }
}

// One case: a tiled view made by @p make over two arrays of @p size.
template <typename Make>
void case_2d(const char* name, long size, Make&& make, long h, border b, how w, bool with_uniform)
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
    const auto  want = reference(field, h, b);
    const auto& cf   = std::as_const(f);
    const auto  src  = md::view_type<const Array<double>, typename decltype(f)::layout_type>(f);
    (void)cf;
    if (b == border::clamp)
    {
        with_uniform ? run_stencil(w, src, g, h, md::clamp_boundary{}, keep_uniform)
                     : run_stencil(w, src, g, h, md::clamp_boundary{});
    }
    else
    {
        const md::constant_boundary<double> bnd{b == border::zero ? 0.0 : border_value};
        with_uniform ? run_stencil(w, src, g, h, bnd, keep_uniform) : run_stencil(w, src, g, h, bnd);
    }
    long bad     = 0;
    long src_bad = 0;
    for (long i = 0; i < nx; ++i)
    {
        for (long j = 0; j < ny; ++j)
        {
            const double x = std::as_const(g)[i, j];
            const double y = want[static_cast<std::size_t>(i * ny + j)];
            if (!(x == y || (x - y < 1e-12 && y - x < 1e-12)))
            {
                if (bad == 0)
                {
                    println("  {}: [{}, {}] = {} expected {}", name, i, j, x, y);
                }
                ++bad;
            }
            src_bad += std::as_const(f)[i, j] != cell(i, j) ? 1 : 0;
        }
    }
    expect(bad == 0, name, bad, static_cast<long>(w) * 100 + static_cast<long>(b) * 10 + h);
    expect(src_bad == 0, "the source changed", src_bad);
    // The tiles of the result whose cells are one value hold no buffer: at 0
    // on every variant, at any value on zero_any::Chunk (checked when a tile is one
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
            if (uniform && (v0 == 0.0 || chunk_config::use_zero_any))
            {
                kept += d.written(static_cast<int>(k * S)) ? 1 : 0;
            }
        }
        expect(kept == 0, "a uniform tile of the result holds a buffer", kept, static_cast<long>(w));
    }
}

void tiled(how w)
{
    const long S = static_cast<long>(Array<double>(1).chunk_size());
    for (const long h : {1L, 2L})
    {
        for (const border b : {border::zero, border::value, border::clamp})
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
    if (w == how::serial)
    {
        case_2d(
            "tiled 8 x 8, halo 2", static_cast<long>(md::tiled_size<8, 8>(nx, ny)),
            [](auto& arr) { return md::tiled_view<8, 8>(arr, nx, ny); }, 2, border::clamp, w, true);
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
                std::as_const(f), g, 1L, md::clamp_boundary{},
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
                const double x = std::as_const(g)[i, j, k];
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
    md::stencil_tiles(std::as_const(f), f, 1L, md::clamp_boundary{}, [](const auto&, const auto&, auto) {});
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
    tiled(how::serial);
    tiled(how::range);
    tiled(how::team);
    tiled(how::team_outside);
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
''',
}

def tokens(text):
    # comment line stars are layout, not text
    return re.sub(r"(?m)^[ \t]*\*(?=[ \t]|$)", " ", text).split()


def fuzzy(text):
    sep = r"\s+(?:\*\s+)*"  # no nested quantifier: linear on a miss
    return re.compile(sep.join(re.escape(t) for t in tokens(text)))


def core(old, new):
    """new without the leading and trailing layout (spaces, comment stars)
    old shares with it: the fuzzy match starts and ends on a token."""
    # the tokens of old, as tokens() sees them: a lone star at a line start is layout
    toks = [m for m in re.finditer(r"\S+", old)
            if not (m.group() == "*" and old[old.rfind("\n", 0, m.start()) + 1:m.start()].strip() == "")]
    lead = toks[0].start()
    trail = len(old) - toks[-1].end()
    if new[:lead] == old[:lead]:
        new = new[lead:]
    if trail and new[-trail:] == old[-trail:]:
        new = new[:-trail]
    return new


SENTINEL = "stencil_tiles"  # in every file once changed, in none before

_hdr = os.path.join(ROOT, "chunk/public/chunk/mdspan_tile_ops.h")
if os.path.exists(_hdr):
    with open(_hdr, encoding="utf-8") as _f:
        if "fill_box_mt" not in _f.read():
            print("ERROR: md::fill_box_mt is not in this tree: run mdspan_copy_box_mt.py, then mdspan_tiles_mt.py first",
                  file=sys.stderr)
            print("nothing written", file=sys.stderr)
            sys.exit(1)

errors = []
texts = {}
orig = {}
for p, old, new in HUNKS:
    if p not in texts:
        if not os.path.exists(os.path.join(ROOT, p)):
            if f"{p}: missing" not in errors:
                errors.append(f"{p}: missing")
            continue
        with open(os.path.join(ROOT, p), encoding="utf-8") as f:
            texts[p] = orig[p] = f.read()
    s = texts[p]
    if SENTINEL in orig[p]:
        continue  # this file already has the change
    if s.count(old) == 1:
        texts[p] = s.replace(old, new)
        continue
    # Not found as is: the same text with other spaces, line breaks or
    # comment reflow (another clang-format) -- matched token by token.
    found = list(fuzzy(old).finditer(s))
    if len(found) != 1:
        errors.append(f"{p}: anchor found {len(found)} times: {old.strip().splitlines()[0][:70]!r}")
        continue
    m = found[0]
    texts[p] = s[:m.start()] + core(old, new) + s[m.end():]

for p, content in NEW_FILES.items():
    full = os.path.join(ROOT, p)
    if os.path.exists(full):
        with open(full, encoding="utf-8") as f:
            if f.read() != content:
                errors.append(f"{p}: exists with other content")
    elif not os.path.isdir(os.path.dirname(full)):
        errors.append(f"{p}: no directory {os.path.dirname(p)}")
    texts[p] = content

if errors:
    for e in errors:
        print("ERROR:", e, file=sys.stderr)
    print("nothing written", file=sys.stderr)
    sys.exit(1)
written = 0
for p, s in texts.items():
    full = os.path.join(ROOT, p)
    if os.path.exists(full):
        with open(full, encoding="utf-8") as f:
            if f.read() == s:
                continue
    with open(os.path.join(ROOT, p), "w", encoding="utf-8") as f:
        f.write(s)
    print("written:", p)
    written += 1
if not written:
    print("already applied: nothing to do")
