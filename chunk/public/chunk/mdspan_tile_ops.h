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
 * @file mdspan_tile_ops.h
 * @brief Tile by tile over a tiled mdspan view (chunk/mdspan.h): for_each_tile()
 *        reads, update_tiles() reads and writes back, a uniform tile in O(1);
 *        fill_box() sets a box of a view (tiled or row-major) to one value by
 *        runs instead of elements; copy_box_out() / copy_box_in() copy a box
 *        to / from contiguous row-major memory (an MPI halo, an HDF5
 *        hyperslab, a library that wants a T*), tile by tile or row by row;
 *        copy_box_out_mt() / copy_box_in_mt() share the box among the
 *        threads of a region by themselves; stencil_tiles() runs a stencil
 *        tile by tile, the kernel reading a tile and its halo.
 *
 * @details f[i, j] pays an operator[] per element. With layout_tiled, a tile is
 *          a run of consecutive elements -- one chunk when the tile has
 *          chunk_size() elements -- so it can be read in one go
 *          (for_each_span(): a block copy, or the constant of a compressed
 *          chunk) and written back in one go (copy_range_mt(), or
 *          fill_range_mt() when the result is uniform).
 *
 *          @code
 *          auto f = md::tiled_view<16, 16>(a, nx, ny);
 *          double s = 0;
 *          md::for_each_tile(f,
 *              [&](const auto& t, auto b) {          // a materialised tile
 *                  for (long i = 0; i < t.extent[0]; ++i)
 *                      for (long j = 0; j < t.extent[1]; ++j) s += b[i, j];
 *              },
 *              [&](const auto& t, double v) {        // a uniform tile
 *                  s += v * double(t.extent[0] * t.extent[1]);
 *              });
 *          @endcode
 *
 *          The block is an mdspan of static extents Tile... over a copy of the
 *          tile (layout_right): plain doubles, no proxy, vectorisable loops.
 *          It covers the whole tile, padding included; t.extent is the part
 *          inside the view's extents.
 *
 * @warning Experimental until 1.0, as chunk/mdspan.h: names and signatures may
 *          change in a MINOR release (CHANGELOG.md, versioning policy).
 *
 * @warning Concurrency. Callable inside a parallel region (in a phase, as any
 *          write), each thread on its own tiles (`#pragma omp for` over
 *          [0, tile_count(f)), then the range overloads). While a tile is read,
 *          no other thread may write its chunks: that is what the iterator and
 *          for_each_span() require. Hence, inside a parallel region, a tile
 *          must be made of whole chunks (tile elements a multiple of
 *          chunk_size()), or two threads would share a chunk: checked, a
 *          mismatch stops the program. Outside a parallel region any chunk
 *          size works -- a tile may then straddle chunks.
 */

#include "chunk/mdspan.h" // shrink_array::md, SHRINK_ARRAY_HAS_MDSPAN

#if SHRINK_ARRAY_HAS_MDSPAN

    #include <algorithm> // std::copy, std::copy_n, std::fill, std::fill_n, std::equal, std::all_of, std::min, std::max, std::clamp
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
    #include "chunk/internal/invariant_check.h" // CHUNK_INVARIANT_FAILED

namespace shrink_array::md
{

/**
 * @brief One tile of a tiled view, as passed to the callbacks.
 * @tparam Index the view's index type
 * @tparam Rank  the view's rank
 */
template <typename Index, std::size_t Rank>
struct tile
{
    Index                   number; ///< In [0, tile_count()): the tile's rank in storage order.
    std::array<Index, Rank> origin; ///< Multi-index of the tile's first element in the view.
    std::array<Index, Rank> extent; ///< Elements of the tile inside the view's extents, per dimension.
};

/// @brief A tiled view of this library: layout_tiled or layout_tiled_dyn, and md::accessor.
template <typename View>
concept tiled_view_of_array = requires(const typename View::mapping_type& m) {
    m.tile_elements(); // layout_tiled, layout_tiled_dyn
    m.tile_extent(0);
    requires std::same_as<typename View::accessor_type, accessor<typename View::accessor_type::array_type>> ||
                 std::same_as<typename View::accessor_type, accessor<const typename View::accessor_type::array_type>>;
};

namespace detail
{
template <typename View>
using ArrayOf = typename View::accessor_type::array_type;

template <typename View>
using ValueOf = typename ArrayOf<View>::value_type;

/// @brief The block type a callback receives, row-major over the tile:
///        static extents Tile... (layout_tiled), dextents (layout_tiled_dyn).
template <typename View, typename E, std::size_t... Tile>
auto block_type_of(layout_tiled<Tile...>*) -> mdns::mdspan<E, mdns::extents<typename View::index_type, Tile...>>;

template <typename View, typename E, std::size_t Rank>
auto block_type_of(layout_tiled_dyn<Rank>*) -> mdns::mdspan<E, mdns::dextents<typename View::index_type, Rank>>;

template <typename View, typename E>
using block_t = decltype(block_type_of<View, E>(static_cast<typename View::layout_type*>(nullptr)));

/// @brief The block of view @p f over the tile buffer @p p.
template <typename View, typename E>
block_t<View, E> make_block(const View& f, E* p)
{
    if constexpr (requires { View::mapping_type::tile_elems; })
    {
        return block_t<View, E>(p); // static extents
    }
    else
    {
        return block_t<View, E>(p, f.mapping().tile());
    }
}

/**
 * @brief A tile's worth of elements: a std::array for a tile known at compile
 *        time; at run time (layout_tiled_dyn), inline up to 8 KiB, a vector
 *        beyond -- update_tiles(f, k, k + 1) per tile in a parallel loop
 *        then allocates nothing for the usual chunk sizes.
 */
template <typename View, typename T>
class TileBuffer
{
    static constexpr bool        is_static = requires { View::mapping_type::tile_elems; };
    static constexpr std::size_t inline_n =
        is_static ? 1 : (sizeof(T) >= 8192 ? std::size_t{1} : 8192 / sizeof(T)); // dyn: inline capacity

  public:
    explicit TileBuffer(const View& f)
    {
        if constexpr (!is_static)
        {
            n_ = static_cast<std::size_t>(f.mapping().tile_elements());
            if (n_ > inline_n)
            {
                heap_.resize(n_);
            }
        }
    }
    T* data() noexcept
    {
        if constexpr (is_static)
        {
            return fixed_.data();
        }
        else
        {
            return n_ > inline_n ? heap_.data() : small_.data();
        }
    }
    std::size_t size() const noexcept
    {
        if constexpr (is_static)
        {
            return fixed_.size();
        }
        else
        {
            return n_;
        }
    }
    T& operator[](std::size_t i) noexcept
    {
        return data()[i];
    }
    T* begin() noexcept
    {
        return data();
    }
    T* end() noexcept
    {
        return data() + size();
    }

  private:
    struct None
    {
    };
    static constexpr std::size_t fixed_n = []
    {
        if constexpr (is_static)
        {
            return static_cast<std::size_t>(View::mapping_type::tile_elems);
        }
        else
        {
            return std::size_t{1};
        }
    }();
    // Left uninitialised, as a std::array<T, N> local was: gather() writes
    // every element before anything reads one.
    [[no_unique_address]] std::conditional_t<is_static, std::array<T, fixed_n>, None>  fixed_;
    [[no_unique_address]] std::conditional_t<is_static, None, std::array<T, inline_n>> small_;
    [[no_unique_address]] std::conditional_t<is_static, None, std::vector<T>>          heap_;
    std::size_t                                                                        n_ = 0;
};

/// @brief Tiles per dimension of view @p f.
template <typename View>
std::array<typename View::index_type, View::rank()> tiles_per_dim(const View& f)
{
    std::array<typename View::index_type, View::rank()> n{};
    for (std::size_t r = 0; r < View::rank(); ++r)
    {
        const auto t = f.mapping().tile_extent(static_cast<typename View::rank_type>(r));
        n[r]         = (f.extent(r) + t - 1) / t;
    }
    return n;
}

/// @brief The tile of storage rank @p k: its origin and its extent inside the view.
template <typename View>
tile<typename View::index_type, View::rank()>
tile_at(const View& f, const std::array<typename View::index_type, View::rank()>& per_dim, typename View::index_type k)
{
    tile<typename View::index_type, View::rank()> t{k, {}, {}};
    auto                                          rest = k;
    for (std::size_t r = View::rank(); r-- > 0;)
    {
        const auto te = f.mapping().tile_extent(static_cast<typename View::rank_type>(r));
        const auto c  = rest % per_dim[r];
        rest /= per_dim[r];
        t.origin[r] = c * te;
        t.extent[r] = std::min<typename View::index_type>(te, f.extent(r) - t.origin[r]);
    }
    return t;
}

/// @brief Index in the array of tile @p t's first element: the view's base
///        plus the mapping of the tile's origin -- k * tile_elems for a whole
///        view, the parent's grid for a sub-view (md::submdspan).
template <typename View, typename Tile>
std::size_t tile_storage(const View& f, const Tile& t)
{
    return static_cast<std::size_t>(f.data_handle().base) +
           static_cast<std::size_t>(std::apply([&f](auto... x) { return f.mapping()(x...); }, t.origin));
}

/// @brief Inside a parallel region, a tile must be whole chunks (see the file's warning).
template <typename Array, typename Index>
void check_tile_on_chunks(const Array& a, Index tile_elems, const char* who)
{
    if (CHUNK_PMP_IN_PARALLEL_REGION)
    {
        const auto cs = static_cast<Index>(a.chunk_size());
        if (cs == 0 || tile_elems % cs != 0) [[unlikely]]
        {
            CHUNK_INVARIANT_FAILED("{}: in a parallel region a tile must be whole chunks, but a tile has {} "
                                   "elements and the array's chunk size is {}: two threads would share a chunk",
                                   who, static_cast<long long>(tile_elems), static_cast<long long>(cs));
        }
    }
}

/**
 * @brief Read storage elements [lo, lo + buf.size()) of @p a into @p buf.
 * @return true when they are all one constant @p cst; @p buf is then left
 *         unfilled if @p lazy, so a uniform tile costs O(1).
 */
template <typename Array, typename Buf, typename T>
bool gather(const Array& a, std::size_t lo, Buf& buf, T& cst, bool lazy)
{
    const std::size_t N       = buf.size();
    std::size_t       k       = 0;
    bool              uniform = true;
    bool              pending = false; // a constant run at the start, not yet written to buf
    struct Visitor
    {
        Buf&         buf;
        std::size_t& k;
        bool&        uniform;
        bool&        pending;
        T&           cst;
        bool         lazy;

        void flush_pending() const
        {
            if (pending)
            {
                std::fill(buf.begin(), buf.begin() + static_cast<std::ptrdiff_t>(k), cst);
                pending = false;
            }
        }
        void operator()(std::span<const T> s) const
        {
            flush_pending();
            std::copy(s.begin(), s.end(), buf.begin() + static_cast<std::ptrdiff_t>(k));
            k += s.size();
            uniform = false;
        }
        void operator()(const T& v, std::size_t n) const
        {
            if (k == 0)
            {
                cst     = v;
                pending = lazy;
            }
            else if (uniform && !(v == cst))
            {
                uniform = false;
            }
            if (!pending || !uniform)
            {
                flush_pending();
                std::fill(buf.begin() + static_cast<std::ptrdiff_t>(k),
                          buf.begin() + static_cast<std::ptrdiff_t>(k + n), v);
            }
            k += n;
        }
    };
    for_each_span(a, lo, lo + N, Visitor{buf, k, uniform, pending, cst, lazy});
    if (k != N) [[unlikely]]
    {
        CHUNK_INVARIANT_FAILED("md tile ops: the array has {} elements, the tiled view needs at least {} "
                               "(size the array with md::tiled_size)",
                               static_cast<long long>(a.size()), static_cast<long long>(lo + N));
    }
    return uniform;
}

struct NoUniform
{
};

/**
 * @brief The writes of update_tiles() and fill_box(): the serial range ops
 *        outside a parallel region (the array is quiescent; deferred writes
 *        drained once, by the first write), the _mt ones inside, which are safe
 *        against the other writers and the compressor.
 */
template <typename Array>
struct RangeWriter
{
    Array&     a;
    const bool in_region = CHUNK_PMP_IN_PARALLEL_REGION;
    bool       drained   = false;

    template <typename T>
    void fill(std::size_t lo, std::size_t hi, const T& v)
    {
        if (in_region)
        {
            fill_range_mt(a, lo, hi, v);
        }
        else
        {
            fill_range(a, lo, hi, v, !drained);
            drained = true;
        }
    }

    template <typename T>
    void copy(std::size_t lo, std::size_t hi, const T* src)
    {
        if (in_region)
        {
            copy_range_mt(a, lo, hi, src);
        }
        else
        {
            copy_range(a, lo, hi, src, !drained);
            drained = true;
        }
    }
};

/// @brief Consecutive storage runs merged into one fill: one range op per run.
template <typename Array, typename T>
struct RunFiller
{
    RangeWriter<Array>& out;
    const T&            v;
    std::size_t         lo = 0;
    std::size_t         hi = 0;

    void add(std::size_t a, std::size_t b)
    {
        if (a == hi && hi != lo)
        {
            hi = b;
            return;
        }
        flush();
        lo = a;
        hi = b;
    }
    void flush()
    {
        if (hi > lo)
        {
            out.fill(lo, hi, v);
        }
        lo = hi = 0;
    }
};

/// @brief A box corner in the view's index type, clamped to its range first
///        (a corner beyond the extents is clipped by clip_box afterwards).
template <typename View, typename I>
std::array<typename View::index_type, View::rank()> clipped_box(const std::array<I, View::rank()>& c)
{
    using Index = typename View::index_type;
    std::array<Index, View::rank()> out{};
    for (std::size_t r = 0; r < View::rank(); ++r)
    {
        const auto x = static_cast<long long>(c[r]);
        out[r] =
            static_cast<Index>(std::clamp<long long>(x, 0, static_cast<long long>(std::numeric_limits<Index>::max())));
    }
    return out;
}

/// @brief The box [lo, hi) clipped to the extents of @p f; false when empty.
template <typename View, typename Box>
bool clip_box(const View& f, Box& lo, Box& hi)
{
    for (std::size_t r = 0; r < View::rank(); ++r)
    {
        lo[r] = std::max<typename View::index_type>(lo[r], 0);
        hi[r] = std::min<typename View::index_type>(hi[r], f.extent(r));
        if (lo[r] >= hi[r])
        {
            return false;
        }
    }
    return true;
}
} // namespace detail

// ─────────────────────────────────────────────────────────────────────────────
// Boxes, rows and the shares of a team: the helpers of the box operations
// ─────────────────────────────────────────────────────────────────────────────

namespace detail
{
/// @brief The box [@p blo, @p bhi) of @p f in its index type, checked to lie
///        inside the view's extents (a clipped box would change the buffer's
///        shape: refused, not clipped). @return false when the box is empty.
template <typename View, typename I>
bool checked_box(const View&                                          f,
                 const std::array<I, View::rank()>&                   blo,
                 const std::array<I, View::rank()>&                   bhi,
                 std::array<typename View::index_type, View::rank()>& lo,
                 std::array<typename View::index_type, View::rank()>& hi,
                 const char*                                          who)
{
    using Index    = typename View::index_type;
    bool non_empty = true;
    for (std::size_t r = 0; r < View::rank(); ++r)
    {
        const auto l = static_cast<long long>(blo[r]);
        const auto h = static_cast<long long>(bhi[r]);
        const auto e = static_cast<long long>(f.extent(r));
        if (l < 0 || l > h || h > e) [[unlikely]]
        {
            CHUNK_INVARIANT_FAILED("{}: the box [{}, {}) of dimension {} does not lie inside the view's extent {}", who,
                                   l, h, r, e);
        }
        lo[r]     = static_cast<Index>(l);
        hi[r]     = static_cast<Index>(h);
        non_empty = non_empty && l < h;
    }
    return non_empty;
}

/// @brief Row-major strides of a buffer of extents @p ext.
template <typename Index, std::size_t R>
std::array<std::size_t, R> row_major_strides(const std::array<Index, R>& ext)
{
    std::array<std::size_t, R> s{};
    std::size_t                acc = 1;
    for (std::size_t r = R; r-- > 0;)
    {
        s[r] = acc;
        acc *= static_cast<std::size_t>(ext[r]);
    }
    return s;
}

/// @brief Offset of multi-index @p idx in a row-major buffer whose element 0
///        is @p origin, strides @p s.
template <typename Index, std::size_t R>
std::size_t
offset_in(const std::array<Index, R>& idx, const std::array<Index, R>& origin, const std::array<std::size_t, R>& s)
{
    std::size_t o = 0;
    for (std::size_t r = 0; r < R; ++r)
    {
        o += static_cast<std::size_t>(idx[r] - origin[r]) * s[r];
    }
    return o;
}

/// @brief @p fn(idx) for the first cell of every row of [@p lo, @p hi): idx
///        walks every dimension but the last, idx[R - 1] == lo[R - 1].
template <typename Index, std::size_t R, typename Fn>
void for_each_row(const std::array<Index, R>& lo, const std::array<Index, R>& hi, Fn&& fn)
{
    std::array<Index, R> i = lo;
    while (true)
    {
        fn(std::as_const(i));
        std::size_t r = R - 1;
        while (r-- > 0)
        {
            if (++i[r] < hi[r])
            {
                break;
            }
            i[r] = lo[r];
        }
        if (r == static_cast<std::size_t>(-1))
        {
            break;
        }
    }
}

/// @brief @p fn(k) for every tile k in [@p first, @p last) that box [@p lo,
///        @p hi) of tiled view @p f touches, in storage order.
template <typename View, typename Fn>
void for_each_tile_of_box(const View&                                                f,
                          const std::array<typename View::index_type, View::rank()>& lo,
                          const std::array<typename View::index_type, View::rank()>& hi,
                          typename View::index_type                                  first,
                          typename View::index_type                                  last,
                          Fn&&                                                       fn)
{
    using Index         = typename View::index_type;
    constexpr auto rank = View::rank();
    const auto     dims = tiles_per_dim(f);
    Index          nt   = 1;
    for (const auto d : dims)
    {
        nt *= d;
    }
    last = std::min(last, nt);
    if (first >= last)
    {
        return;
    }
    std::array<Index, rank> tlo{};
    std::array<Index, rank> thi{};
    for (std::size_t r = 0; r < rank; ++r)
    {
        const auto te = f.mapping().tile_extent(static_cast<typename View::rank_type>(r));
        tlo[r]        = lo[r] / te;
        thi[r]        = (hi[r] - 1) / te + 1;
    }
    // Tiles [first, last) are rows of tiles in storage order: only those rows
    // of the box are walked (as fill_box).
    Index rest = 1;
    for (std::size_t r = 1; r < rank; ++r)
    {
        rest *= dims[r];
    }
    tlo[0] = std::max(tlo[0], first / rest);
    thi[0] = std::min(thi[0], (last - 1) / rest + 1);
    if (tlo[0] >= thi[0])
    {
        return;
    }
    for_each_row(tlo, thi,
                 [&](const std::array<Index, rank>& c0)
                 {
                     std::array<Index, rank> c = c0;
                     for (c[rank - 1] = tlo[rank - 1]; c[rank - 1] < thi[rank - 1]; ++c[rank - 1])
                     {
                         Index k = 0;
                         for (std::size_t r = 0; r < rank; ++r)
                         {
                             k = k * dims[r] + c[r];
                         }
                         if (k >= first && k < last)
                         {
                             fn(k);
                         }
                     }
                 });
}

/// @brief The part of tile @p t inside box [@p lo, @p hi): [ilo, ihi).
template <typename Tile, typename Index, std::size_t R>
void tile_box(const Tile&                 t,
              const std::array<Index, R>& lo,
              const std::array<Index, R>& hi,
              std::array<Index, R>&       ilo,
              std::array<Index, R>&       ihi)
{
    for (std::size_t r = 0; r < R; ++r)
    {
        ilo[r] = std::max(lo[r], t.origin[r]);
        ihi[r] = std::min(hi[r], static_cast<Index>(t.origin[r] + t.extent[r]));
    }
}

/// @brief The extents of a tile of @p f, and their row-major strides.
template <typename View>
std::array<std::size_t, View::rank()> tile_strides(const View& f)
{
    std::array<typename View::index_type, View::rank()> te{};
    for (std::size_t r = 0; r < View::rank(); ++r)
    {
        te[r] = f.mapping().tile_extent(static_cast<typename View::rank_type>(r));
    }
    return row_major_strides(te);
}

/// @brief Cell [W t / T]: where thread @p t of @p T starts in a box of @p W
///        cells (the same bound on every thread, no communication).
inline std::size_t share_bound(std::size_t W, std::size_t t, std::size_t T)
{
    return t >= T ? W : W / T * t + W % T * t / T;
}

/// @brief What a tile weighs in a deal: its cells inside the box (a copy,
///        a fill: the work is the box's cells), or one (a callback on the
///        whole tile: the work is the tile).
enum class DealWeight
{
    cells,
    tiles,
};

/// @brief The tiles of box [@p lo, @p hi) dealt to the calling thread of its
///        team: in storage order, a tile goes to the thread whose share of
///        the box's weight, [W t / T, W (t + 1) / T), holds the tile's first
///        unit (a tile weighing its cells inside the box, or one: @p weight).
///        Every thread computes the same deal, with no communication; a tile
///        goes to one thread only. Outside a region: every tile.
template <typename View, typename Fn>
void for_each_dealt_tile(const View&                                                f,
                         const std::array<typename View::index_type, View::rank()>& lo,
                         const std::array<typename View::index_type, View::rank()>& hi,
                         Fn&&                                                       fn,
                         DealWeight                                                 weight = DealWeight::cells)
{
    using Index         = typename View::index_type;
    constexpr auto rank = View::rank();
    const auto     T    = static_cast<std::size_t>(CHUNK_PMP_NUM_THREADS);
    const auto     me   = static_cast<std::size_t>(T > 1 ? chunk_pmp_team_rank() : 0);
    std::size_t    W    = 1;
    for (std::size_t r = 0; r < rank; ++r)
    {
        if (hi[r] <= lo[r])
        {
            return;
        }
        if (weight == DealWeight::cells)
        {
            W *= static_cast<std::size_t>(hi[r] - lo[r]);
        }
        else
        {
            const auto te = f.mapping().tile_extent(static_cast<typename View::rank_type>(r));
            W *= static_cast<std::size_t>((hi[r] - 1) / te - lo[r] / te + 1);
        }
    }
    const std::size_t b0   = share_bound(W, me, T);
    const std::size_t b1   = share_bound(W, me + 1, T);
    const auto        dims = tiles_per_dim(f);
    std::size_t       acc  = 0;
    for_each_tile_of_box(f, lo, hi, Index{0}, std::numeric_limits<Index>::max(),
                         [&](Index k)
                         {
                             if (acc >= b1)
                             {
                                 return;
                             }
                             if (weight == DealWeight::tiles)
                             {
                                 if (acc++ >= b0)
                                 {
                                     fn(k);
                                 }
                                 return;
                             }
                             const auto              t = tile_at(f, dims, k);
                             std::array<Index, rank> ilo{};
                             std::array<Index, rank> ihi{};
                             tile_box(t, lo, hi, ilo, ihi);
                             std::size_t w = 1;
                             for (std::size_t r = 0; r < rank; ++r)
                             {
                                 w *= static_cast<std::size_t>(ihi[r] - ilo[r]);
                             }
                             const std::size_t first_cell = acc;
                             acc += w;
                             if (first_cell >= b0)
                             {
                                 fn(k);
                             }
                         });
}

/// @brief @p fn(idx, m) for the cells [@p c0, @p c1) of box [@p lo, @p hi),
///        counted in row-major order: one call per piece of row, idx its
///        first cell, m its length.
template <typename Index, std::size_t R, typename Fn>
void for_each_row_piece(const std::array<Index, R>& lo,
                        const std::array<Index, R>& hi,
                        std::size_t                 c0,
                        std::size_t                 c1,
                        Fn&&                        fn)
{
    const auto n = static_cast<std::size_t>(hi[R - 1] - lo[R - 1]);
    while (c0 < c1)
    {
        std::size_t          q = c0 / n;
        const std::size_t    p = c0 % n;
        std::array<Index, R> idx{};
        for (std::size_t r = R - 1; r-- > 0;)
        {
            const auto e = static_cast<std::size_t>(hi[r] - lo[r]);
            idx[r]       = static_cast<Index>(lo[r] + static_cast<Index>(q % e));
            q /= e;
        }
        idx[R - 1]          = static_cast<Index>(lo[R - 1] + static_cast<Index>(p));
        const std::size_t m = std::min(n - p, c1 - c0);
        fn(std::as_const(idx), m);
        c0 += m;
    }
}

/// @brief The calling thread's cells [c0, c1) of box [@p lo, @p hi) of a
///        strided view: an equal share of the cells, its bounds moved up to
///        the next chunk boundary of the array when the rows are contiguous
///        (@p chunk > 0), so that no chunk is written by two threads within a
///        row. @p at maps a multi-index to its array index.
template <typename Index, std::size_t R, typename At>
std::pair<std::size_t, std::size_t>
strided_share(const std::array<Index, R>& lo, const std::array<Index, R>& hi, std::size_t chunk, const At& at)
{
    std::size_t W = 1;
    for (std::size_t r = 0; r < R; ++r)
    {
        W *= static_cast<std::size_t>(hi[r] - lo[r]);
    }
    const auto T     = static_cast<std::size_t>(CHUNK_PMP_NUM_THREADS);
    const auto me    = static_cast<std::size_t>(T > 1 ? chunk_pmp_team_rank() : 0);
    const auto n     = static_cast<std::size_t>(hi[R - 1] - lo[R - 1]);
    const auto bound = [&](std::size_t t) -> std::size_t
    {
        if (t == 0 || t >= T || W == 0)
        {
            return t == 0 ? 0 : W;
        }
        const std::size_t c = share_bound(W, t, T);
        if (chunk == 0)
        {
            return c;
        }
        std::size_t c_aligned = c;
        for_each_row_piece(lo, hi, c, c + 1,
                           [&](const std::array<Index, R>& idx, std::size_t)
                           {
                               const std::size_t s  = at(idx);
                               const std::size_t up = (s + chunk - 1) / chunk * chunk;
                               const std::size_t p  = static_cast<std::size_t>(idx[R - 1] - lo[R - 1]);
                               c_aligned            = c - p + std::min(n, p + (up - s));
                           });
        return c_aligned;
    };
    return {bound(me), bound(me + 1)};
}

/// @brief A strided view on a chunked array, readable (and writable for
///        copy_box_in()).
template <typename View>
concept strided_view_of_array =
    View::mapping_type::is_always_strided() &&
    (std::same_as<typename View::accessor_type, accessor<typename View::accessor_type::array_type>> ||
     std::same_as<typename View::accessor_type, accessor<const typename View::accessor_type::array_type>>);

template <typename View>
concept writable_strided_view_of_array =
    View::mapping_type::is_always_strided() && !std::is_const_v<typename View::element_type> &&
    std::same_as<typename View::accessor_type, accessor<typename View::accessor_type::array_type>>;

/// @brief The array index of multi-index @p i of strided view @p f.
template <typename View>
std::size_t strided_at(const View& f, const std::array<typename View::index_type, View::rank()>& i)
{
    return static_cast<std::size_t>(f.data_handle().base) +
           static_cast<std::size_t>(std::apply([&f](auto... x) { return f.mapping()(x...); }, i));
}

/// @brief The whole box: [0, W).
struct WholeBox
{
    template <typename Index, std::size_t R>
    std::pair<std::size_t, std::size_t>
    operator()(const std::array<Index, R>& lo, const std::array<Index, R>& hi, bool) const
    {
        std::size_t W = 1;
        for (std::size_t r = 0; r < R; ++r)
        {
            W *= static_cast<std::size_t>(hi[r] - lo[r]);
        }
        return {0, W};
    }
};

/// @brief The calling thread's share of the box (strided_share()), its
///        bounds on chunk boundaries when the rows are contiguous.
template <typename View>
auto team_share(const View& f)
{
    return [&f](const auto& lo, const auto& hi, bool rows)
    {
        const std::size_t chunk = rows ? static_cast<std::size_t>(f.data_handle()->chunk_size()) : 0;
        return strided_share(lo, hi, chunk, [&f](const auto& i) { return strided_at(f, i); });
    };
}
} // namespace detail

/// @brief Number of tiles of tiled view @p f (padding tiles included): the range of the range overloads.
template <tiled_view_of_array View>
typename View::index_type tile_count(const View& f)
{
    typename View::index_type n = 1;
    for (const auto d : detail::tiles_per_dim(f))
    {
        n *= d;
    }
    return n;
}

namespace detail
{
/// @brief A walk over tiles [@p first, @p last) of @p f (clamped to its tiles).
template <typename View>
auto tile_range(const View& f, typename View::index_type first, typename View::index_type last)
{
    return [&f, first, last](auto&& fn)
    {
        const auto end = std::min(last, tile_count(f));
        for (auto k = first; k < end; ++k)
        {
            fn(k);
        }
    };
}

/// @brief A walk over the tiles box [@p lo, @p hi) of @p f touches dealt to
///        the calling thread (for_each_dealt_tile()); the box clipped to the
///        view, the whole view when not given.
template <typename View>
auto dealt_tiles(const View&                                                f,
                 const std::array<typename View::index_type, View::rank()>& lo,
                 const std::array<typename View::index_type, View::rank()>& hi,
                 DealWeight                                                 weight)
{
    return [&f, lo, hi, weight](auto&& fn) { for_each_dealt_tile(f, lo, hi, fn, weight); };
}

template <typename View>
auto dealt_tiles(const View& f)
{
    std::array<typename View::index_type, View::rank()> lo{};
    std::array<typename View::index_type, View::rank()> hi{};
    for (std::size_t r = 0; r < View::rank(); ++r)
    {
        hi[r] = f.extent(r);
    }
    return dealt_tiles(f, lo, hi, DealWeight::tiles);
}

/// @brief for_each_tile() on the tiles @p walk(fn) gives (fn(k) per tile).
template <typename View, typename Walk, typename OnBlock, typename OnUniform>
void read_tiles(const View& f, const char* who, Walk&& walk, OnBlock& on_block, OnUniform& on_uniform)
{
    using T          = ValueOf<View>;
    using Index      = typename View::index_type;
    const auto& a    = *f.data_handle();
    const auto  dims = tiles_per_dim(f);
    check_tile_on_chunks(a, f.mapping().tile_elements(), who);
    constexpr bool has_uniform = !std::same_as<std::remove_cvref_t<OnUniform>, NoUniform>;

    TileBuffer<View, T> buf(f);
    T                   cst{};
    walk(
        [&](Index k)
        {
            const auto t       = tile_at(f, dims, k);
            const bool uniform = gather(a, tile_storage(f, t), buf, cst, has_uniform);
            if constexpr (has_uniform)
            {
                if (uniform)
                {
                    on_uniform(t, std::as_const(cst));
                    return;
                }
            }
            on_block(t, make_block<View, const T>(f, buf.data()));
        });
}

/// @brief update_tiles() on the tiles @p walk(fn) gives (fn(k) per tile).
template <typename View, typename Walk, typename OnBlock, typename OnUniform>
void write_tiles(const View& f, const char* who, Walk&& walk, OnBlock& on_block, OnUniform& on_uniform)
{
    using T                = detail::ValueOf<View>;
    using Index            = typename View::index_type;
    const std::size_t N    = static_cast<std::size_t>(f.mapping().tile_elements());
    auto&             a    = *f.data_handle();
    const auto        dims = detail::tiles_per_dim(f);
    detail::check_tile_on_chunks(a, f.mapping().tile_elements(), who);
    constexpr bool has_uniform = !std::same_as<std::remove_cvref_t<OnUniform>, detail::NoUniform>;

    detail::TileBuffer<View, T>                buf(f);
    detail::TileBuffer<View, T>                before(f); // a materialised tile as read: written back only if changed
    T                                          cst{};
    detail::RangeWriter<detail::ArrayOf<View>> out{a};
    walk(
        [&](Index k)
        {
            const auto        t       = detail::tile_at(f, dims, k);
            const std::size_t lo      = detail::tile_storage(f, t);
            const bool        uniform = detail::gather(a, lo, buf, cst, has_uniform);
            if constexpr (has_uniform)
            {
                if (uniform)
                {
                    T v = cst;
                    if (on_uniform(t, v))
                    {
                        if (!(v == cst))
                        {
                            out.fill(lo, lo + N, v);
                        }
                        return;
                    }
                    std::fill(buf.begin(), buf.end(), cst);
                }
            }
            if (!uniform)
            {
                std::copy(buf.begin(), buf.end(), before.begin());
            }
            on_block(t, detail::make_block<View, T>(f, buf.data()));

            const T    first_v = buf[0];
            const bool now_uniform =
                std::all_of(buf.begin() + 1, buf.end(), [&first_v](const T& x) { return x == first_v; });
            if (now_uniform)
            {
                if (!(uniform && first_v == cst))
                {
                    out.fill(lo, lo + N, first_v);
                }
            }
            else if (uniform || !std::equal(buf.begin(), buf.end(), before.begin()))
            {
                out.copy(lo, lo + N, buf.data());
            }
        });
}

} // namespace detail

/**
 * @brief Read tiles [@p first, @p last) of tiled view @p f, in storage order:
 *        @p on_block(tile, block) for a tile that holds several values,
 *        @p on_uniform(tile, value) for a tile that is one constant.
 * @param on_block   called as on_block(const tile&, block) where block is an
 *                   mdspan<const T, extents<Tile...>> over a copy of the tile
 * @param on_uniform called as on_uniform(const tile&, const T& v) for a
 *                   uniform tile, in O(1). Omitted: on_block gets the tile
 *                   filled with v.
 * @warning See the file: no other thread writes these tiles meanwhile; in a
 *          parallel region a tile must be whole chunks.
 */
template <tiled_view_of_array View, typename OnBlock, typename OnUniform = detail::NoUniform>
void for_each_tile(const View&               f,
                   typename View::index_type first,
                   typename View::index_type last,
                   OnBlock&&                 on_block,
                   OnUniform&&               on_uniform = {})
{
    detail::read_tiles(f, "md::for_each_tile", detail::tile_range(f, first, last), on_block, on_uniform);
}

/// @brief for_each_tile() over every tile of @p f.
template <tiled_view_of_array View, typename OnBlock, typename OnUniform = detail::NoUniform>
void for_each_tile(const View& f, OnBlock&& on_block, OnUniform&& on_uniform = {})
{
    for_each_tile(f, typename View::index_type{0}, tile_count(f), std::forward<OnBlock>(on_block),
                  std::forward<OnUniform>(on_uniform));
}

/**
 * @brief Update tiles [@p first, @p last) of writable tiled view @p f: each
 *        tile is read into a buffer, changed by the callbacks, and written back
 *        in one go -- only when it changed, as one constant when it is uniform.
 *
 * @details A uniform tile left unchanged is not written: a kernel that leaves
 *          the background alone does not materialise it. A changed tile is
 *          written in one go: fill_range() when it comes out uniform, else
 *          copy_range() -- the serial range ops outside a parallel region, the
 *          _mt ones inside (one write session per chunk, the element path's
 *          guarantees). Outside a region, on zero_any::Chunk, a whole chunk that comes
 *          out uniform takes its constant with no buffer (fill_range(), or
 *          copy_range() for a uniform chunk of a mixed tile). Otherwise a compressed
 *          chunk that changes is materialised for the write and queued: the end
 *          of the phase, or the next compression pass, compresses it again if
 *          it is uniform.
 *
 * @param on_block   called as on_block(const tile&, block) where block is an
 *                   mdspan<T, extents<Tile...>> over the tile's buffer, to
 *                   change in place (padding cells included)
 * @param on_uniform called as on_uniform(const tile&, T& v) -> bool for a
 *                   uniform tile, in O(1): change v and return true (the tile
 *                   becomes v everywhere), or return false and on_block gets
 *                   the tile filled with v. Omitted: always on_block.
 * @warning for_each_tile()'s. Writes are writes: inside a parallel region,
 *          within a phase (parallel_phase()).
 */
template <tiled_view_of_array View, typename OnBlock, typename OnUniform = detail::NoUniform>
    requires(!std::is_const_v<typename View::element_type>)
void update_tiles(const View&               f,
                  typename View::index_type first,
                  typename View::index_type last,
                  OnBlock&&                 on_block,
                  OnUniform&&               on_uniform = {})
{
    detail::write_tiles(f, "md::update_tiles", detail::tile_range(f, first, last), on_block, on_uniform);
}

/// @brief update_tiles() over every tile of @p f.
template <tiled_view_of_array View, typename OnBlock, typename OnUniform = detail::NoUniform>
    requires(!std::is_const_v<typename View::element_type>)
void update_tiles(const View& f, OnBlock&& on_block, OnUniform&& on_uniform = {})
{
    update_tiles(f, typename View::index_type{0}, tile_count(f), std::forward<OnBlock>(on_block),
                 std::forward<OnUniform>(on_uniform));
}

/**
 * @brief for_each_tile() called by every thread of a parallel region: the
 *        tiles of @p f are dealt to the team, no `omp for`.
 * @details Thread t of T reads the tiles [n t / T, n (t + 1) / T) of the n
 *          tiles, in storage order: every thread computes the same deal, with
 *          no communication, and a tile is read by one thread. Outside a
 *          region (or in a team of one), every tile. No barrier: what the
 *          callbacks gather is complete once every thread has returned.
 * @warning for_each_tile()'s; every thread of the team calls it.
 */
template <tiled_view_of_array View, typename OnBlock, typename OnUniform = detail::NoUniform>
void for_each_tile_mt(const View& f, OnBlock&& on_block, OnUniform&& on_uniform = {})
{
    detail::read_tiles(f, "md::for_each_tile_mt", detail::dealt_tiles(f), on_block, on_uniform);
}

/**
 * @brief for_each_tile_mt() on the tiles box [@p lo, @p hi) touches (clipped
 *        to the view): those tiles dealt to the team, one unit each, in
 *        storage order. The callbacks get whole tiles, as for_each_tile()'s.
 */
template <tiled_view_of_array View, std::integral I, typename OnBlock, typename OnUniform = detail::NoUniform>
void for_each_tile_mt(const View&                        f,
                      const std::array<I, View::rank()>& box_lo,
                      const std::array<I, View::rank()>& box_hi,
                      OnBlock&&                          on_block,
                      OnUniform&&                        on_uniform = {})
{
    auto lo = detail::clipped_box<View>(box_lo);
    auto hi = detail::clipped_box<View>(box_hi);
    if (!detail::clip_box(f, lo, hi))
    {
        return;
    }
    detail::read_tiles(f, "md::for_each_tile_mt", detail::dealt_tiles(f, lo, hi, detail::DealWeight::tiles), on_block,
                       on_uniform);
}

/**
 * @brief update_tiles() called by every thread of a parallel region, in a
 *        phase: the tiles of @p f dealt to the team as for_each_tile_mt()
 *        deals them, no `omp for`. Each tile is written by one thread.
 *        Outside a region, every tile serially. No barrier: two calls over
 *        the whole view deal the tiles alike (a tile stays on its thread),
 *        but after a call on a box, or before one, an `omp barrier` is
 *        needed -- the deals differ.
 * @warning update_tiles()'s; every thread of the team calls it.
 */
template <tiled_view_of_array View, typename OnBlock, typename OnUniform = detail::NoUniform>
    requires(!std::is_const_v<typename View::element_type>)
void update_tiles_mt(const View& f, OnBlock&& on_block, OnUniform&& on_uniform = {})
{
    detail::write_tiles(f, "md::update_tiles_mt", detail::dealt_tiles(f), on_block, on_uniform);
}

/**
 * @brief update_tiles_mt() on the tiles box [@p lo, @p hi) touches (clipped
 *        to the view), dealt to the team one unit each. The callbacks get
 *        whole tiles: cells outside the box are theirs to leave alone.
 */
template <tiled_view_of_array View, std::integral I, typename OnBlock, typename OnUniform = detail::NoUniform>
    requires(!std::is_const_v<typename View::element_type>)
void update_tiles_mt(const View&                        f,
                     const std::array<I, View::rank()>& box_lo,
                     const std::array<I, View::rank()>& box_hi,
                     OnBlock&&                          on_block,
                     OnUniform&&                        on_uniform = {})
{
    auto lo = detail::clipped_box<View>(box_lo);
    auto hi = detail::clipped_box<View>(box_hi);
    if (!detail::clip_box(f, lo, hi))
    {
        return;
    }
    detail::write_tiles(f, "md::update_tiles_mt", detail::dealt_tiles(f, lo, hi, detail::DealWeight::tiles), on_block,
                        on_uniform);
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
class TileCache
{
    using Index = typename View::index_type;

  public:
    struct Entry
    {
        explicit Entry(const View& f) : buf(f) {}
        Index               k       = std::numeric_limits<Index>::max();
        bool                uniform = false;
        T                   cst{};
        TileBuffer<View, T> buf;
    };

    TileCache(const View& f, std::size_t n) : f_(f), dims_(tiles_per_dim(f))
    {
        e_.reserve(n);
        for (std::size_t i = 0; i < n; ++i)
        {
            e_.emplace_back(f);
        }
    }

    /// @brief The tile of coordinates @p c (in tiles), read.
    Entry& fetch(const std::array<Index, View::rank()>& c)
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
        Entry& x     = e_[next_];
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
    std::vector<Entry>              e_;
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
    using T             = ValueOf<Src>;
    using Index         = typename Src::index_type;
    constexpr auto rank = Src::rank();
    static_assert(rank == Dst::rank(), "md::stencil_tiles: the source and the destination have one rank");
    static_assert(std::same_as<T, ValueOf<Dst>>, "md::stencil_tiles: the source and the destination hold one type");
    static_assert(boundary_of<Boundary, T>,
                  "md::stencil_tiles: the boundary is md::clamp_boundary{} or md::constant_boundary<T>{v}");
    constexpr bool has_uniform = !std::same_as<std::remove_cvref_t<OnUniform>, NoUniform>;
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

    TileCache<Src, T>         cache(src, 2 * neighbourhood<rank>());
    const auto                dims = cache.dims();
    const std::size_t         N    = static_cast<std::size_t>(src.mapping().tile_elements());
    TileBuffer<Dst, T>        outb(dst);
    RangeWriter<ArrayOf<Dst>> out{da};
    T                         boundary_value{}; // the halo outside the view, constant_boundary's
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
          typename OnUniform = detail::NoUniform>
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
          typename OnUniform = detail::NoUniform>
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
          typename OnUniform = detail::NoUniform>
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
///        in storage order: consecutive covered tiles merged into one range op.
template <typename View, typename I, typename Walk>
void fill_tiles(const View&                        f,
                const std::array<I, View::rank()>& box_lo,
                const std::array<I, View::rank()>& box_hi,
                const ValueOf<View>&               v,
                Walk&&                             walk)
{
    using Index         = typename View::index_type;
    constexpr auto rank = View::rank();
    const auto     N    = static_cast<std::size_t>(f.mapping().tile_elements());
    auto           lo   = clipped_box<View>(box_lo);
    auto           hi   = clipped_box<View>(box_hi);
    if (!clip_box(f, lo, hi))
    {
        return;
    }
    auto&                                   a    = *f.data_handle();
    const auto                              dims = tiles_per_dim(f);
    RangeWriter<ArrayOf<View>>              out{a};
    RunFiller<ArrayOf<View>, ValueOf<View>> runs{out, v};
    walk(lo, hi,
         [&](Index k)
         {
             const auto t     = tile_at(f, dims, k);
             bool       whole = true;
             for (std::size_t r = 0; r < rank; ++r)
             {
                 whole = whole && lo[r] <= t.origin[r] && hi[r] >= t.origin[r] + t.extent[r];
             }
             if (whole)
             {
                 const std::size_t s = tile_storage(f, t);
                 runs.add(s, s + N);
                 return;
             }
             runs.flush();
             update_tiles(
                 f, k, k + 1,
                 [&](const auto& tt, auto b)
                 {
                     std::array<Index, rank> i{};
                     for (std::size_t r = 0; r < rank; ++r)
                     {
                         i[r] = std::max(lo[r], tt.origin[r]) - tt.origin[r];
                     }
                     const std::array<Index, rank> i0 = i;
                     while (true) // every cell of tile x box
                     {
                         b[i]          = v;
                         std::size_t r = rank;
                         while (r-- > 0)
                         {
                             if (++i[r] < std::min(hi[r], tt.origin[r] + tt.extent[r]) - tt.origin[r])
                             {
                                 break;
                             }
                             i[r] = i0[r];
                         }
                         if (r == static_cast<std::size_t>(-1))
                         {
                             break;
                         }
                     }
                 },
                 [&](const auto&, ValueOf<View>& u) { return u == v; });
         });
    runs.flush();
}
} // namespace detail

/**
 * @brief Set the box [@p lo, @p hi) of tiled view @p f to @p v, restricted to
 *        tiles [@p first, @p last): a tile the box covers is filled whole, with
 *        its padding (one range op for consecutive covered tiles); a tile it
 *        crosses is updated as update_tiles() does (left alone when it is
 *        already uniform at @p v).
 * @details The box is clipped to the view's extents. The tile range is for a
 *          `#pragma omp for` over [0, tile_count(f)), each thread on its own
 *          tiles, in a phase; update_tiles()'s rules then hold (a tile made of
 *          whole chunks).
 * @param lo first multi-index of the box, inclusive
 * @param hi last multi-index of the box, exclusive
 * @warning A fill is a write. Outside a parallel region on zero_any::Chunk, a covered
 *          chunk takes the value as its constant, with no buffer (fill_range()).
 *          Inside a region, a compressed chunk that changes is materialised for
 *          the write and compressed again by the end of the phase or the next
 *          pass (see update_tiles()).
 */
template <tiled_view_of_array View, std::integral I>
    requires(!std::is_const_v<typename View::element_type>)
void fill_box(const View&                        f,
              const std::array<I, View::rank()>& box_lo,
              const std::array<I, View::rank()>& box_hi,
              const detail::ValueOf<View>&       v,
              typename View::index_type          first,
              typename View::index_type          last)
{
    detail::fill_tiles(f, box_lo, box_hi, v, [&](const auto& lo, const auto& hi, auto&& fn)
                       { detail::for_each_tile_of_box(f, lo, hi, first, last, fn); });
}

/// @brief fill_box() over every tile of @p f.
template <tiled_view_of_array View, std::integral I>
    requires(!std::is_const_v<typename View::element_type>)
void fill_box(const View&                        f,
              const std::array<I, View::rank()>& lo,
              const std::array<I, View::rank()>& hi,
              const detail::ValueOf<View>&       v)
{
    fill_box(f, lo, hi, v, typename View::index_type{0}, tile_count(f));
}

/**
 * @brief fill_box() called by every thread of a parallel region, in a phase:
 *        the tiles of the box dealt to the team by their cells inside it
 *        (as copy_box_in_mt() deals them), no `omp for`. Each tile is
 *        written by one thread; outside a region, the whole box. No barrier. Two calls on
 *        overlapping boxes need an `omp barrier` between them: the next
 *        deal may give a shared tile to another thread.
 * @warning fill_box()'s; every thread of the team calls it with the same box
 *          and value.
 */
template <tiled_view_of_array View, std::integral I>
    requires(!std::is_const_v<typename View::element_type>)
void fill_box_mt(const View&                        f,
                 const std::array<I, View::rank()>& lo,
                 const std::array<I, View::rank()>& hi,
                 const detail::ValueOf<View>&       v)
{
    detail::fill_tiles(f, lo, hi, v, [&](const auto& blo, const auto& bhi, auto&& fn)
                       { detail::for_each_dealt_tile(f, blo, bhi, fn, detail::DealWeight::cells); });
}

namespace detail
{
/// @brief fill_box() of a strided view on the cells [c0, c1) of the box
///        (row-major order) @p share(lo, hi, rows) gives: one run per piece
///        of row when the rows are contiguous, merged when consecutive in the
///        array; one per element otherwise.
template <typename View, typename I, typename Share>
void fill_strided(const View&                        f,
                  const std::array<I, View::rank()>& box_lo,
                  const std::array<I, View::rank()>& box_hi,
                  const ValueOf<View>&               v,
                  Share&&                            share)
{
    using Index         = typename View::index_type;
    constexpr auto rank = View::rank();
    auto           lo   = clipped_box<View>(box_lo);
    auto           hi   = clipped_box<View>(box_hi);
    if (!clip_box(f, lo, hi))
    {
        return;
    }
    RangeWriter<ArrayOf<View>>              out{*f.data_handle()};
    RunFiller<ArrayOf<View>, ValueOf<View>> runs{out, v};
    const bool                              rows = f.stride(rank - 1) == 1;
    const auto [c0, c1]                          = share(lo, hi, rows);
    for_each_row_piece(lo, hi, c0, c1,
                       [&](const std::array<Index, rank>& idx, std::size_t m)
                       {
                           if (rows)
                           {
                               const std::size_t a = strided_at(f, idx);
                               runs.add(a, a + m);
                               return;
                           }
                           std::array<Index, rank> i = idx;
                           for (std::size_t j = 0; j < m; ++j, ++i[rank - 1])
                           {
                               const std::size_t a = strided_at(f, i);
                               runs.add(a, a + 1);
                           }
                       });
    runs.flush();
}
} // namespace detail

/**
 * @brief Set the box [@p lo, @p hi) of a strided view @p f (md::view(), or a
 *        submdspan of one: layout_right, layout_stride) to @p v: one range op
 *        per contiguous run -- a row
 *        of the box, or several rows when the box spans whole rows. With a
 *        last stride other than 1 (a sub-view taking every other column, a
 *        layout_left view) every element is a run of its own.
 * @details The box is clipped to the view's extents. Callable in a phase,
 *          each thread on its own box (its own rows): the runs of two threads
 *          may share a chunk, which fill_range_mt() allows.
 */
template <detail::writable_strided_view_of_array View, std::integral I>
void fill_box(const View&                        f,
              const std::array<I, View::rank()>& lo,
              const std::array<I, View::rank()>& hi,
              const detail::ValueOf<View>&       v)
{
    detail::fill_strided(f, lo, hi, v, detail::WholeBox{});
}

/**
 * @brief fill_box() of a strided view called by every thread of a parallel
 *        region, in a phase: the box's cells, in row-major order, split in T
 *        consecutive shares, each bound moved up to the next chunk boundary
 *        when the rows are contiguous (as copy_box_in_mt() splits them), no
 *        `omp for`. Outside a region, the whole box. No barrier. Two calls on
 *        overlapping boxes need an `omp barrier` between them: the next
 *        deal may give a shared tile to another thread.
 * @warning fill_box()'s; every thread of the team calls it with the same box
 *          and value.
 */
template <detail::writable_strided_view_of_array View, std::integral I>
void fill_box_mt(const View&                        f,
                 const std::array<I, View::rank()>& lo,
                 const std::array<I, View::rank()>& hi,
                 const detail::ValueOf<View>&       v)
{
    detail::fill_strided(f, lo, hi, v, detail::team_share(f));
}


// ─────────────────────────────────────────────────────────────────────────────
// A box in and out of contiguous memory: copy_box_out(), copy_box_in()
// ─────────────────────────────────────────────────────────────────────────────

namespace detail
{
/// @brief copy_box_out() on the tiles @p walk(lo, hi, fn) gives.
template <typename View, typename I, typename Walk>
void copy_tiles_out(const View&                        f,
                    const std::array<I, View::rank()>& box_lo,
                    const std::array<I, View::rank()>& box_hi,
                    ValueOf<View>*                     dst,
                    const char*                        who,
                    Walk&&                             walk)
{
    using T                      = detail::ValueOf<View>;
    using Index                  = typename View::index_type;
    constexpr auto          rank = View::rank();
    std::array<Index, rank> lo{};
    std::array<Index, rank> hi{};
    if (!detail::checked_box(f, box_lo, box_hi, lo, hi, who))
    {
        return;
    }
    const auto& a = *f.data_handle();
    detail::check_tile_on_chunks(a, f.mapping().tile_elements(), who);
    std::array<Index, rank> ext{};
    for (std::size_t r = 0; r < rank; ++r)
    {
        ext[r] = hi[r] - lo[r];
    }
    const auto                  bs   = detail::row_major_strides(ext);
    const auto                  ts   = detail::tile_strides(f);
    const auto                  dims = detail::tiles_per_dim(f);
    detail::TileBuffer<View, T> buf(f);
    T                           cst{};
    walk(lo, hi,
         [&](Index k)
         {
             const auto              t = detail::tile_at(f, dims, k);
             std::array<Index, rank> ilo{};
             std::array<Index, rank> ihi{};
             detail::tile_box(t, lo, hi, ilo, ihi);
             const bool uniform = detail::gather(a, detail::tile_storage(f, t), buf, cst, /*lazy=*/true);
             const auto n       = static_cast<std::size_t>(ihi[rank - 1] - ilo[rank - 1]);
             detail::for_each_row(ilo, ihi,
                                  [&](const std::array<Index, rank>& idx)
                                  {
                                      T* d = dst + detail::offset_in(idx, lo, bs);
                                      if (uniform)
                                      {
                                          std::fill_n(d, n, cst);
                                      }
                                      else
                                      {
                                          std::copy_n(buf.data() + detail::offset_in(idx, t.origin, ts), n, d);
                                      }
                                  });
         });
}

/// @brief copy_box_in() on the tiles @p walk(lo, hi, fn) gives.
template <typename View, typename I, typename Walk>
void copy_tiles_in(const View&                        f,
                   const std::array<I, View::rank()>& box_lo,
                   const std::array<I, View::rank()>& box_hi,
                   const ValueOf<View>*               src,
                   const char*                        who,
                   Walk&&                             walk)
{
    using T                      = detail::ValueOf<View>;
    using Index                  = typename View::index_type;
    constexpr auto          rank = View::rank();
    std::array<Index, rank> lo{};
    std::array<Index, rank> hi{};
    if (!detail::checked_box(f, box_lo, box_hi, lo, hi, who))
    {
        return;
    }
    auto& a = *f.data_handle();
    detail::check_tile_on_chunks(a, f.mapping().tile_elements(), who);
    std::array<Index, rank> ext{};
    for (std::size_t r = 0; r < rank; ++r)
    {
        ext[r] = hi[r] - lo[r];
    }
    const auto                                 bs   = detail::row_major_strides(ext);
    const auto                                 ts   = detail::tile_strides(f);
    const auto                                 dims = detail::tiles_per_dim(f);
    const std::size_t                          N    = static_cast<std::size_t>(f.mapping().tile_elements());
    detail::TileBuffer<View, T>                buf(f);
    detail::RangeWriter<detail::ArrayOf<View>> out{a};
    walk(lo, hi,
         [&](Index k)
         {
             const auto              t = detail::tile_at(f, dims, k);
             std::array<Index, rank> ilo{};
             std::array<Index, rank> ihi{};
             detail::tile_box(t, lo, hi, ilo, ihi);
             const auto n     = static_cast<std::size_t>(ihi[rank - 1] - ilo[rank - 1]);
             bool       whole = true;
             bool       pad   = false;
             for (std::size_t r = 0; r < rank; ++r)
             {
                 whole = whole && ilo[r] == t.origin[r] && ihi[r] == t.origin[r] + t.extent[r];
                 pad   = pad || t.extent[r] != f.mapping().tile_extent(static_cast<typename View::rank_type>(r));
             }
             const auto copy_rows = [&](T* tile_data)
             {
                 detail::for_each_row(ilo, ihi,
                                      [&](const std::array<Index, rank>& idx)
                                      {
                                          std::copy_n(src + detail::offset_in(idx, lo, bs), n,
                                                      tile_data + detail::offset_in(idx, t.origin, ts));
                                      });
             };
             if (!whole)
             {
                 update_tiles(f, k, k + 1, [&](const auto&, auto b) { copy_rows(b.data_handle()); });
                 return;
             }
             if (pad)
             {
                 std::fill(buf.begin(), buf.end(), src[detail::offset_in(ilo, lo, bs)]);
             }
             copy_rows(buf.data());
             const std::size_t s  = detail::tile_storage(f, t);
             const T           v0 = buf[0];
             if (std::all_of(buf.begin() + 1, buf.end(), [&v0](const T& x) { return x == v0; }))
             {
                 out.fill(s, s + N, v0);
             }
             else
             {
                 out.copy(s, s + N, buf.data());
             }
         });
}
} // namespace detail

/**
 * @brief Copy box [@p lo, @p hi) of tiled view @p f into @p dst, restricted
 *        to tiles [@p first, @p last): @p dst is a row-major buffer of the
 *        box's extents (hi - lo), element (i, j, ...) of the box at
 *        ((i - lo[0]) * (hi[1] - lo[1]) + (j - lo[1])) ...
 *
 * @details Tile by tile: a tile is read in one go (a block copy), a uniform
 *          one costs a fill of its part of the box. The tile range is for a
 *          `#pragma omp for` over [0, tile_count(f)): each thread writes the
 *          rows of @p dst its tiles hold, disjoint from the others'.
 *          copy_box_out_mt() shares the box's tiles by itself, by weight.
 * @pre   The box lies inside the view's extents (checked: a mismatch stops
 *        the program). @p dst holds at least the box's element count. Read
 *        rules of for_each_tile(): in a parallel region a tile is whole
 *        chunks, and no thread writes these tiles meanwhile.
 * @warning Experimental until 1.0 (see the file).
 */
template <tiled_view_of_array View, std::integral I>
void copy_box_out(const View&                        f,
                  const std::array<I, View::rank()>& box_lo,
                  const std::array<I, View::rank()>& box_hi,
                  detail::ValueOf<View>*             dst,
                  typename View::index_type          first,
                  typename View::index_type          last)
{
    detail::copy_tiles_out(f, box_lo, box_hi, dst, "md::copy_box_out", [&](const auto& lo, const auto& hi, auto&& fn)
                           { detail::for_each_tile_of_box(f, lo, hi, first, last, fn); });
}

/// @brief copy_box_out() over every tile of @p f.
template <tiled_view_of_array View, std::integral I>
void copy_box_out(const View&                        f,
                  const std::array<I, View::rank()>& lo,
                  const std::array<I, View::rank()>& hi,
                  detail::ValueOf<View>*             dst)
{
    copy_box_out(f, lo, hi, dst, typename View::index_type{0}, tile_count(f));
}

/**
 * @brief copy_box_out() called by every thread of a parallel region: the
 *        tiles of the box are dealt to the team by weight, no `omp for`.
 * @details The tiles the box touches, in storage order, each weighing its
 *          cells inside the box, are split in T consecutive shares of about
 *          W / T cells (W the box's cells): every thread computes the same
 *          deal, with no communication, and copies its tiles. Outside a
 *          region (or in a team of one), the whole box. No barrier: @p dst is
 *          complete once every thread has returned (the region's next
 *          barrier, or its end).
 * @pre   As copy_box_out(); every thread of the team calls it with the same
 *        box and buffer.
 * @warning Experimental until 1.0 (see the file).
 */
template <tiled_view_of_array View, std::integral I>
void copy_box_out_mt(const View&                        f,
                     const std::array<I, View::rank()>& box_lo,
                     const std::array<I, View::rank()>& box_hi,
                     detail::ValueOf<View>*             dst)
{
    detail::copy_tiles_out(f, box_lo, box_hi, dst, "md::copy_box_out_mt", [&](const auto& lo, const auto& hi, auto&& fn)
                           { detail::for_each_dealt_tile(f, lo, hi, fn); });
}

/**
 * @brief Copy row-major buffer @p src, of the extents of box [@p lo, @p hi),
 *        into that box of tiled view @p f, restricted to tiles [@p first,
 *        @p last).
 *
 * @details A tile the box covers is written in one go, from the buffer, with
 *          no read of the tile: fill_range() when its values are one value
 *          (a chunk already at it is left alone, a compressed one switches,
 *          a materialised one is released where the variant can), else
 *          copy_range() -- the _mt forms inside a parallel region. The cells
 *          of its padding take the value of its first cell, so a uniform part
 *          of the box gives a uniform tile. A tile the box crosses is updated
 *          as update_tiles() does (read, its cells of the box changed, written
 *          back only if changed). The tile range is for a `#pragma omp for`
 *          over [0, tile_count(f)), each thread on its own tiles, in a phase;
 *          copy_box_in_mt() shares the box's tiles by itself.
 * @pre   The box lies inside the view's extents (checked). @p src holds the
 *        box's element count and does not alias the array. Write rules of
 *        update_tiles(): in a parallel region, within a phase, a tile whole
 *        chunks.
 * @warning Experimental until 1.0 (see the file).
 */
template <tiled_view_of_array View, std::integral I>
    requires(!std::is_const_v<typename View::element_type>)
void copy_box_in(const View&                        f,
                 const std::array<I, View::rank()>& box_lo,
                 const std::array<I, View::rank()>& box_hi,
                 const detail::ValueOf<View>*       src,
                 typename View::index_type          first,
                 typename View::index_type          last)
{
    detail::copy_tiles_in(f, box_lo, box_hi, src, "md::copy_box_in", [&](const auto& lo, const auto& hi, auto&& fn)
                          { detail::for_each_tile_of_box(f, lo, hi, first, last, fn); });
}

/// @brief copy_box_in() over every tile of @p f.
template <tiled_view_of_array View, std::integral I>
    requires(!std::is_const_v<typename View::element_type>)
void copy_box_in(const View&                        f,
                 const std::array<I, View::rank()>& lo,
                 const std::array<I, View::rank()>& hi,
                 const detail::ValueOf<View>*       src)
{
    copy_box_in(f, lo, hi, src, typename View::index_type{0}, tile_count(f));
}

/**
 * @brief copy_box_in() called by every thread of a parallel region, in a
 *        phase: the tiles of the box are dealt to the team by weight, as
 *        copy_box_out_mt() deals them, no `omp for`.
 * @details Each tile is written by one thread; outside a region (or in a
 *          team of one), the whole box serially. No barrier. Two calls on
 *          overlapping boxes need an `omp barrier` between them: the next
 *          deal may give a shared tile to another thread.
 * @pre   As copy_box_in(); every thread of the team calls it with the same
 *        box and buffer.
 * @warning Experimental until 1.0 (see the file).
 */
template <tiled_view_of_array View, std::integral I>
    requires(!std::is_const_v<typename View::element_type>)
void copy_box_in_mt(const View&                        f,
                    const std::array<I, View::rank()>& box_lo,
                    const std::array<I, View::rank()>& box_hi,
                    const detail::ValueOf<View>*       src)
{
    detail::copy_tiles_in(f, box_lo, box_hi, src, "md::copy_box_in_mt", [&](const auto& lo, const auto& hi, auto&& fn)
                          { detail::for_each_dealt_tile(f, lo, hi, fn); });
}

namespace detail
{
/// @brief copy_box_out() of a strided view, on the cells [c0, c1) of the box
///        (row-major order) @p share(lo, hi) gives.
template <typename View, typename I, typename Share>
void copy_strided_out(const View&                        f,
                      const std::array<I, View::rank()>& box_lo,
                      const std::array<I, View::rank()>& box_hi,
                      ValueOf<View>*                     dst,
                      const char*                        who,
                      Share&&                            share)
{
    using T                      = ValueOf<View>;
    using Index                  = typename View::index_type;
    constexpr auto          rank = View::rank();
    std::array<Index, rank> lo{};
    std::array<Index, rank> hi{};
    if (!checked_box(f, box_lo, box_hi, lo, hi, who))
    {
        return;
    }
    const auto&             a = *f.data_handle();
    std::array<Index, rank> ext{};
    for (std::size_t r = 0; r < rank; ++r)
    {
        ext[r] = hi[r] - lo[r];
    }
    const auto bs       = row_major_strides(ext);
    const bool rows     = f.stride(rank - 1) == 1;
    const auto [c0, c1] = share(lo, hi, rows);
    for_each_row_piece(lo, hi, c0, c1,
                       [&](const std::array<Index, rank>& idx, std::size_t n)
                       {
                           T* d = dst + offset_in(idx, lo, bs);
                           if (rows)
                           {
                               std::size_t k = 0;
                               struct Visitor
                               {
                                   T*           d;
                                   std::size_t& k;
                                   void         operator()(std::span<const T> s) const
                                   {
                                       std::copy(s.begin(), s.end(), d + k);
                                       k += s.size();
                                   }
                                   void operator()(const T& v, std::size_t m) const
                                   {
                                       std::fill_n(d + k, m, v);
                                       k += m;
                                   }
                               };
                               const std::size_t s0 = strided_at(f, idx);
                               for_each_span(a, s0, s0 + n, Visitor{d, k});
                           }
                           else
                           {
                               std::array<Index, rank> i = idx;
                               for (std::size_t j = 0; j < n; ++j, ++i[rank - 1])
                               {
                                   d[j] = static_cast<T>(
                                       std::as_const(a)[static_cast<index_of<ArrayOf<View>>>(strided_at(f, i))]);
                               }
                           }
                       });
}

/// @brief copy_box_in() of a strided view, on the cells [c0, c1) of the box
///        @p share(lo, hi) gives.
template <typename View, typename I, typename Share>
void copy_strided_in(const View&                        f,
                     const std::array<I, View::rank()>& box_lo,
                     const std::array<I, View::rank()>& box_hi,
                     const ValueOf<View>*               src,
                     const char*                        who,
                     Share&&                            share)
{
    using Index                  = typename View::index_type;
    constexpr auto          rank = View::rank();
    std::array<Index, rank> lo{};
    std::array<Index, rank> hi{};
    if (!checked_box(f, box_lo, box_hi, lo, hi, who))
    {
        return;
    }
    auto&                   a = *f.data_handle();
    std::array<Index, rank> ext{};
    for (std::size_t r = 0; r < rank; ++r)
    {
        ext[r] = hi[r] - lo[r];
    }
    const auto bs       = row_major_strides(ext);
    const bool rows     = f.stride(rank - 1) == 1;
    const auto [c0, c1] = share(lo, hi, rows);
    RangeWriter<ArrayOf<View>> out{a};
    for_each_row_piece(lo, hi, c0, c1,
                       [&](const std::array<Index, rank>& idx, std::size_t n)
                       {
                           const auto* s = src + offset_in(idx, lo, bs);
                           if (rows)
                           {
                               const std::size_t s0 = strided_at(f, idx);
                               out.copy(s0, s0 + n, s);
                           }
                           else
                           {
                               std::array<Index, rank> i = idx;
                               for (std::size_t j = 0; j < n; ++j, ++i[rank - 1])
                               {
                                   a[static_cast<index_of<ArrayOf<View>>>(strided_at(f, i))] = s[j];
                               }
                           }
                       });
}
} // namespace detail

/**
 * @brief Copy box [@p lo, @p hi) of a strided view @p f (md::view(), or a
 *        submdspan of one) into row-major buffer @p dst of the box's extents.
 * @details One read per row of the box when the last stride is 1
 *          (for_each_span: a block copy per materialised chunk, a fill per
 *          compressed one), element by element otherwise. Callable in a
 *          parallel region, each thread on its own box.
 * @pre   The box lies inside the view's extents (checked).
 * @warning Experimental until 1.0 (see the file).
 */
template <detail::strided_view_of_array View, std::integral I>
void copy_box_out(const View&                        f,
                  const std::array<I, View::rank()>& lo,
                  const std::array<I, View::rank()>& hi,
                  detail::ValueOf<View>*             dst)
{
    detail::copy_strided_out(f, lo, hi, dst, "md::copy_box_out", detail::WholeBox{});
}

/**
 * @brief Copy row-major buffer @p src, of the extents of box [@p lo, @p hi),
 *        into that box of a writable strided view @p f.
 * @details One range op per row of the box when the last stride is 1
 *          (copy_range(), copy_range_mt() inside a parallel region: a row
 *          that covers a chunk with one value makes it that constant where
 *          the variant can), element by element otherwise. Callable in a
 *          phase, each thread on its own box; rows of two threads may share
 *          a chunk, which copy_range_mt() allows.
 * @pre   The box lies inside the view's extents (checked). @p src does not
 *        alias the array.
 * @warning Experimental until 1.0 (see the file).
 */
template <detail::writable_strided_view_of_array View, std::integral I>
void copy_box_in(const View&                        f,
                 const std::array<I, View::rank()>& lo,
                 const std::array<I, View::rank()>& hi,
                 const detail::ValueOf<View>*       src)
{
    detail::copy_strided_in(f, lo, hi, src, "md::copy_box_in", detail::WholeBox{});
}

/**
 * @brief copy_box_out() of a strided view called by every thread of a
 *        parallel region: the box's cells, in row-major order, split in T
 *        consecutive shares, no `omp for`.
 * @details Thread t takes the cells [W t / T, W (t + 1) / T) of the box (W
 *          its cells), each bound moved up to the next chunk boundary of the
 *          array when the rows are contiguous (last stride 1): two threads
 *          then meet on a chunk only between rows. Every thread computes the
 *          same split, with no communication. Outside a region (or in a team
 *          of one), the whole box. No barrier: @p dst is complete once every
 *          thread has returned.
 * @pre   As copy_box_out(); every thread of the team calls it with the same
 *        box and buffer.
 * @warning Experimental until 1.0 (see the file).
 */
template <detail::strided_view_of_array View, std::integral I>
void copy_box_out_mt(const View&                        f,
                     const std::array<I, View::rank()>& lo,
                     const std::array<I, View::rank()>& hi,
                     detail::ValueOf<View>*             dst)
{
    detail::copy_strided_out(f, lo, hi, dst, "md::copy_box_out_mt", detail::team_share(f));
}

/**
 * @brief copy_box_in() of a strided view called by every thread of a
 *        parallel region, in a phase: the box split as copy_box_out_mt()
 *        splits it, a share's bounds on chunk boundaries, so that a chunk
 *        inside a row is written by one thread. No barrier. Two calls on
 *        overlapping boxes need an `omp barrier` between them: the next
 *        deal may give a shared tile to another thread.
 * @pre   As copy_box_in(); every thread of the team calls it with the same
 *        box and buffer.
 * @warning Experimental until 1.0 (see the file).
 */
template <detail::writable_strided_view_of_array View, std::integral I>
void copy_box_in_mt(const View&                        f,
                    const std::array<I, View::rank()>& lo,
                    const std::array<I, View::rank()>& hi,
                    const detail::ValueOf<View>*       src)
{
    detail::copy_strided_in(f, lo, hi, src, "md::copy_box_in_mt", detail::team_share(f));
}

} // namespace shrink_array::md

#endif // SHRINK_ARRAY_HAS_MDSPAN
