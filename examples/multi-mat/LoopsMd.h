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

#pragma once

/**
 * @file LoopsMd.h
 * @brief The loop policies of multi-mat-md, the 2D version of the example:
 *        how a material's arrays are viewed (an mdspan) and how a box of
 *        cells is written, zeroed and read through those views.
 *
 * In 1D a material occupies an interval, walked through an indirection
 * list. In 2D it occupies a box [i0, i0 + h) x [j0, j0 + w) of an nx x ny
 * mesh, and every array is seen through a 2D mdspan: the loop body of
 * ComputeLoopsMd.h is written once against `f[i, j]` and four operations
 * (write_box, zero_box, reduce_box, view), and MultiMatAlgo hands it the
 * policy of the material's storage:
 *
 * | policy                | storage                       | layout                 | box loops                 |
 * |-----------------------|-------------------------------|------------------------|---------------------------|
 * | LoopsMdVector         | std::vector                   | row-major (std mdspan) | cells, `omp simd`         |
 * | LoopsMdTiles          | chunk arrays (v1, Chunk2)     | md::layout_tiled       | tiles (mdspan_tile_ops.h) |
 * | LoopsMdTilesElements  | chunk arrays                  | md::layout_tiled       | cells, `f[i, j]`          |
 * | LoopsMdRows           | chunk arrays                  | row-major (md::view)   | cells, `f[i, j]`          |
 *
 * A tile has the chunk's element count (md::tile_shape: 16 x 16 for 256),
 * so tile k is chunk k: the interior of a box is whole uniform chunks
 * whatever its width, where row-major only gives uniform chunks on rows
 * of a box at least a chunk wide. The tile policy then reads and writes a
 * tile in one go, and a uniform tile in O(1).
 *
 * Every operation is called by every thread of the region and shares its
 * work without a barrier (`omp for nowait`), as the loops of Loops.h.
 */

#include "chunk/mdspan.h" // shrink_array::md, SHRINK_ARRAY_HAS_MDSPAN

#if SHRINK_ARRAY_HAS_MDSPAN

    #include <algorithm>   // std::min, std::max, std::copy_n, std::fill
    #include <array>       // std::array (box corners, tile buffers)
    #include <cstddef>     // std::size_t
    #include <optional>    // std::optional (the uniform-tile hint)
    #include <type_traits> // std::is_const_v, std::remove_reference_t

    #include "chunk/internal/chunk_size_config.h" // shrink_array::static_chunk_size
    #include "chunk/mdspan_tile_ops.h"            // md::update_tiles, md::for_each_tile, md::fill_box

    #include "config.h"     // PRAGMA_FORCEINLINE
    #include "Containers.h" // Vect, VectChunk, VectChunk2

namespace multi_mat::md
{

namespace smd  = shrink_array::md;
namespace mdns = shrink_array::md::mdns;

/// @brief The cells [i0, i0 + h) x [j0, j0 + w) a material occupies at one
///        iteration.
struct Box
{
    int i0 = 0; ///< First row
    int j0 = 0; ///< First column
    int h  = 0; ///< Rows
    int w  = 0; ///< Columns

    [[nodiscard]] int i1() const noexcept
    {
        return i0 + h;
    }
    [[nodiscard]] int j1() const noexcept
    {
        return j0 + w;
    }
    [[nodiscard]] std::array<int, 2> lo() const noexcept
    {
        return {i0, j0};
    }
    [[nodiscard]] std::array<int, 2> hi() const noexcept
    {
        return {i1(), j1()};
    }
};

/// @brief The tile of the chunk arrays: exactly one chunk of elements, as
///        square as the chunk size allows (16 x 16 for 256, 32 x 32 for 1024).
inline constexpr auto tile_shape = smd::tile_shape<2>(shrink_array::static_chunk_size);

/// @brief The tiled layout of the chunk materials.
using layout_tiles = smd::layout_tiled<tile_shape[0], tile_shape[1]>;

/// @brief Elements a chunk array needs for an @p nx x @p ny tiled view (the
///        last tiles padded).
[[nodiscard]] inline std::size_t tiled_storage(int nx, int ny) noexcept
{
    return smd::tiled_size<tile_shape[0], tile_shape[1]>(nx, ny);
}

// The array of the library behind a material's container: the views and the
// range operations take the library's own types, not the example's wrappers.

/// @brief The v1 array of a field.
template <typename T>
shrink_array::ArrayChunk<T>& array_of(VectChunk<T>& v) noexcept
{
    return v;
}
/// @brief The v1 array of a field, read-only.
template <typename T>
const shrink_array::ArrayChunk<T>& array_of(const VectChunk<T>& v) noexcept
{
    return v;
}
/// @brief The Chunk2 array of a presence fraction.
template <typename T>
shrink_array::ArrayChunk2<T>& array_of(VectChunk2<T>& v) noexcept
{
    return v;
}
/// @brief The Chunk2 array of a presence fraction, read-only.
template <typename T>
const shrink_array::ArrayChunk2<T>& array_of(const VectChunk2<T>& v) noexcept
{
    return v;
}

/// @brief No uniform-tile hint: the tile policy always takes the block path.
struct no_hint
{
};

/**
 * @brief One array to write on a box: `f[i, j] = fn(i, j)` for every cell
 *        (i, j) of the box.
 * @tparam Hint `hint(origin, extent) -> std::optional<T>`: the value fn takes
 *         on the whole of a tile (`origin`, `extent` inside the view) when
 *         that tile lies inside the box and fn is constant on it; nullopt
 *         otherwise. Only the tile policy asks: it then writes the tile as
 *         one constant, or nothing when it already holds it.
 */
template <typename View, typename Fn, typename Hint = no_hint>
struct BoxWrite
{
    View f;
    Fn   fn;
    Hint hint;
};

/// @brief A BoxWrite without hint.
template <typename View, typename Fn>
BoxWrite<View, Fn> write(const View& f, Fn fn)
{
    return {f, fn, {}};
}

/// @brief A BoxWrite with a uniform-tile hint.
template <typename View, typename Fn, typename Hint>
BoxWrite<View, Fn, Hint> write(const View& f, Fn fn, Hint hint)
{
    return {f, fn, hint};
}

/**
 * @brief The cell loops over a box, through `f[i, j]`: rows shared by the
 *        team, the columns of a row in order -- under `omp simd` when
 *        @p Simd (the std::vector path; on a chunk array an element access
 *        is a proxy, not a lane).
 */
template <bool Simd>
struct CellLoops
{
    /// @brief Every write of @p ws on the cells of @p b.
    template <typename... Ws>
    void write_box(const Box& b, const Ws&... ws) const
    {
    #pragma omp for schedule(static) nowait
        for (int i = b.i0; i < b.i1(); ++i)
        {
            if constexpr (Simd)
            {
    #pragma omp simd
                for (int j = b.j0; j < b.j1(); ++j)
                {
                    ((ws.f[i, j] = ws.fn(i, j)), ...);
                }
            }
            else
            {
                for (int j = b.j0; j < b.j1(); ++j)
                {
                    ((ws.f[i, j] = ws.fn(i, j)), ...);
                }
            }
        }
    }

    /// @brief The cells of @p b set to 0 in every view of @p fs.
    template <typename... Views>
    void zero_box(const Box& b, const Views&... fs) const
    {
        write_box(b, write(fs, [](int, int) { return 0.; })...);
    }

    /**
     * @brief `red += fn(fr[i, j], t0[i, j], t1[i, j], t2[i, j])` over the
     *        cells of @p b; @p red is the calling thread's accumulator.
     */
    template <typename Fn, typename VF, typename V0, typename V1, typename V2>
    void
    reduce_box(const Box& b, double& red, const Fn& fn, const VF& fr, const V0& t0, const V1& t1, const V2& t2) const
    {
    #pragma omp for schedule(static) nowait
        for (int i = b.i0; i < b.i1(); ++i)
        {
            double s = 0;
            if constexpr (Simd)
            {
    #pragma omp simd reduction(+ : s)
                for (int j = b.j0; j < b.j1(); ++j)
                {
                    s += fn(fr[i, j], t0[i, j], t1[i, j], t2[i, j]);
                }
            }
            else
            {
                for (int j = b.j0; j < b.j1(); ++j)
                {
                    s += fn(fr[i, j], t0[i, j], t1[i, j], t2[i, j]);
                }
            }
            red += s;
        }
    }
};

/// @brief The std::vector materials: a row-major std mdspan over the
///        vector's data, cell loops under `omp simd`.
struct LoopsMdVector : CellLoops<true>
{
    static constexpr const char* name = "std::vector, row-major mdspan, cell loops (omp simd)";

    /// @brief The nx x ny view of @p v (read-only when @p v is const).
    template <typename Vec>
    auto view(Vec& v, int nx, int ny) const
    {
        return mdns::mdspan(v.data(), mdns::dextents<int, 2>(nx, ny));
    }
};

/// @brief The chunk materials, row-major (md::view): f[i, j] is a[i * ny + j],
///        a chunk is a piece of a row.
struct LoopsMdRows : CellLoops<false>
{
    static constexpr const char* name = "chunk arrays, row-major md::view, cell loops f[i, j]";

    template <typename Vec>
    auto view(Vec& v, int nx, int ny) const
    {
        return smd::view(array_of(v), nx, ny);
    }
};

/// @brief The chunk materials in tiles (md::tiled_view), walked cell by cell:
///        the layout of LoopsMdTiles with the access of LoopsMdRows.
struct LoopsMdTilesElements : CellLoops<false>
{
    static constexpr const char* name = "chunk arrays, tiled md::tiled_view, cell loops f[i, j]";

    template <typename Vec>
    auto view(Vec& v, int nx, int ny) const
    {
        return smd::tiled_view<tile_shape[0], tile_shape[1]>(array_of(v), nx, ny);
    }
};

/**
 * @brief The chunk materials in tiles, walked tile by tile
 *        (chunk/mdspan_tile_ops.h).
 * @details The tiles a box touches are, row of tiles by row of tiles, a run
 *          of consecutive tile numbers: each thread takes whole rows of
 *          tiles (`omp for` over them), so no two threads share a tile, nor
 *          a chunk -- what the tile operations require in a parallel region.
 *          - write_box: md::update_tiles; a tile is read once, the cells of
 *            the box changed in the buffer, written back in one go, as one
 *            constant when it came out uniform. With a hint, a tile inside
 *            the box that is already the hinted constant is not touched.
 *          - zero_box: md::fill_box: a tile the box covers becomes 0 in one
 *            range op, no buffer.
 *          - reduce_box: the four tiles read once each (a block copy, or the
 *            constant of a compressed chunk), the box cells summed on plain
 *            doubles; four uniform tiles cost one evaluation.
 */
struct LoopsMdTiles
{
    static constexpr const char* name = "chunk arrays, tiled md::tiled_view, tile loops (update_tiles, fill_box)";

    static constexpr int         te0        = static_cast<int>(tile_shape[0]);
    static constexpr int         te1        = static_cast<int>(tile_shape[1]);
    static constexpr std::size_t tile_elems = tile_shape[0] * tile_shape[1];

    template <typename Vec>
    auto view(Vec& v, int nx, int ny) const
    {
        return smd::tiled_view<tile_shape[0], tile_shape[1]>(array_of(v), nx, ny);
    }

    /// @brief The tiles box @p b touches: rows of tiles [r0, r1), columns
    ///        [c0, c1); `per_row` tiles in a row of the view.
    struct TileRange
    {
        int r0;
        int r1;
        int c0;
        int c1;
        int per_row;

        /// @brief Tile numbers [first, last) of row @p r of the range.
        [[nodiscard]] int first(int r) const noexcept
        {
            return r * per_row + c0;
        }
        [[nodiscard]] int last(int r) const noexcept
        {
            return r * per_row + c1;
        }
    };

    template <typename View>
    static TileRange tiles_of(const Box& b, const View& f) noexcept
    {
        const int ny = static_cast<int>(f.extent(1));
        return {b.i0 / te0, (b.i1() - 1) / te0 + 1, b.j0 / te1, (b.j1() - 1) / te1 + 1, (ny + te1 - 1) / te1};
    }

    template <typename... Ws>
    void write_box(const Box& b, const Ws&... ws) const
    {
        if (b.h <= 0 || b.w <= 0)
        {
            return;
        }
        const TileRange tr = tiles_of(b, first_view(ws...));
    #pragma omp for schedule(static) nowait
        for (int r = tr.r0; r < tr.r1; ++r)
        {
            (write_tiles(b, ws, tr.first(r), tr.last(r)), ...);
        }
    }

    template <typename... Views>
    void zero_box(const Box& b, const Views&... fs) const
    {
        if (b.h <= 0 || b.w <= 0)
        {
            return;
        }
        const TileRange tr = tiles_of(b, first_view_of(fs...));
    #pragma omp for schedule(static) nowait
        for (int r = tr.r0; r < tr.r1; ++r)
        {
            (smd::fill_box(fs, b.lo(), b.hi(), 0., tr.first(r), tr.last(r)), ...);
        }
    }

    template <typename Fn, typename VF, typename V0, typename V1, typename V2>
    void
    reduce_box(const Box& b, double& red, const Fn& fn, const VF& fr, const V0& t0, const V1& t1, const V2& t2) const
    {
        if (b.h <= 0 || b.w <= 0)
        {
            return;
        }
        const TileRange tr = tiles_of(b, fr);
    #pragma omp for schedule(static) nowait
        for (int r = tr.r0; r < tr.r1; ++r)
        {
            std::array<Tile, 4> tl;
            for (int k = tr.first(r); k < tr.last(r); ++k)
            {
                load(fr, k, tl[0]);
                load(t0, k, tl[1]);
                load(t1, k, tl[2]);
                load(t2, k, tl[3]);

                // The cells of the box in tile k, in tile coordinates.
                const int oi = r * te0;
                const int oj = (k - r * tr.per_row) * te1;
                const int ia = std::max(b.i0, oi) - oi;
                const int ib = std::min(b.i1(), oi + te0) - oi;
                const int ja = std::max(b.j0, oj) - oj;
                const int jb = std::min(b.j1(), oj + te1) - oj;

                if (tl[0].uniform && tl[1].uniform && tl[2].uniform && tl[3].uniform)
                {
                    red += static_cast<double>((ib - ia) * (jb - ja)) * fn(tl[0].cst, tl[1].cst, tl[2].cst, tl[3].cst);
                    continue;
                }
                for (Tile& x : tl)
                {
                    spread(x);
                }
                // Plain doubles (a uniform tile was spread by load()): a loop
                // the compiler vectorises, as on the std::vector path.
                const double* pf = tl[0].values.data();
                const double* p0 = tl[1].values.data();
                const double* p1 = tl[2].values.data();
                const double* p2 = tl[3].values.data();
                double        s  = 0;
                for (int i = ia; i < ib; ++i)
                {
                    const int row = i * te1;
    #pragma omp simd reduction(+ : s)
                    for (int j = row + ja; j < row + jb; ++j)
                    {
                        s += fn(pf[j], p0[j], p1[j], p2[j]);
                    }
                }
                red += s;
            }
        }
    }

  private:
    /// @brief One tile as read: its values, row-major over the tile, padding
    ///        included; for a uniform tile, its constant too.
    struct Tile
    {
        bool                           uniform = false;
        double                         cst     = 0;
        std::array<double, tile_elems> values{};
    };

    /// @brief Tile @p k of view @p f into @p t. A uniform tile costs no
    ///        block copy: its constant, spread over the values only when the
    ///        other tiles of the cell are not all uniform (reduce_box()).
    template <typename View>
    static void load(const View& f, int k, Tile& t)
    {
        t.uniform = false;
        smd::for_each_tile(
            f, k, k + 1, [&t](const auto&, auto blk) { std::copy_n(blk.data_handle(), tile_elems, t.values.data()); },
            [&t](const auto&, double v)
            {
                t.uniform = true;
                t.cst     = v;
            });
    }

    /// @brief The values of a uniform tile, for the cell loop.
    static void spread(Tile& t)
    {
        if (t.uniform)
        {
            std::fill(t.values.begin(), t.values.end(), t.cst);
        }
    }

    /// @brief Tiles [first, last) of @p w's view: the cells of the box set
    ///        to w.fn, tile by tile.
    template <typename View, typename Fn, typename Hint>
    static void write_tiles(const Box& b, const BoxWrite<View, Fn, Hint>& w, int first, int last)
    {
        const auto on_block = [&b, &w](const auto& t, auto blk)
        {
            const int oi = static_cast<int>(t.origin[0]);
            const int oj = static_cast<int>(t.origin[1]);
            const int ia = std::max(b.i0, oi);
            const int ib = std::min(b.i1(), oi + static_cast<int>(t.extent[0]));
            const int ja = std::max(b.j0, oj);
            const int jb = std::min(b.j1(), oj + static_cast<int>(t.extent[1]));
            for (int i = ia; i < ib; ++i)
            {
                for (int j = ja; j < jb; ++j)
                {
                    blk[i - oi, j - oj] = w.fn(i, j);
                }
            }
        };
        if constexpr (std::is_same_v<Hint, no_hint>)
        {
            smd::update_tiles(w.f, first, last, on_block);
        }
        else
        {
            smd::update_tiles(w.f, first, last, on_block,
                              [&w](const auto& t, double& v)
                              {
                                  const std::optional<double> u = w.hint(t.origin, t.extent);
                                  if (u)
                                  {
                                      v = *u; // unchanged: update_tiles writes nothing
                                  }
                                  return u.has_value();
                              });
        }
    }

    template <typename W, typename... Ws>
    static const auto& first_view(const W& w, const Ws&...) noexcept
    {
        return w.f;
    }
    template <typename V, typename... Vs>
    static const V& first_view_of(const V& f, const Vs&...) noexcept
    {
        return f;
    }
};

} // namespace multi_mat::md

#endif // SHRINK_ARRAY_HAS_MDSPAN
