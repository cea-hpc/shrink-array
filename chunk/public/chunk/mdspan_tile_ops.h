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
 *        runs instead of elements.
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

    #include <algorithm>   // std::copy, std::fill, std::equal, std::all_of, std::min, std::max, std::clamp
    #include <array>       // std::array (the tile buffer, the tile coordinates)
    #include <concepts>    // std::invocable, std::same_as, std::integral
    #include <cstddef>     // std::size_t
    #include <limits>      // std::numeric_limits (the box corners)
    #include <span>        // std::span (the blocks of for_each_span)
    #include <tuple>       // std::apply (a multi-index into a mapping)
    #include <type_traits> // std::remove_cvref_t, std::is_const_v
    #include <utility>     // std::as_const, std::forward
    #include <vector>      // std::vector (a run-time tile larger than the inline buffer)

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
using array_of = typename View::accessor_type::array_type;

template <typename View>
using value_of = typename array_of<View>::value_type;

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
class tile_buffer
{
    static constexpr bool        is_static = requires { View::mapping_type::tile_elems; };
    static constexpr std::size_t inline_n =
        is_static ? 1 : (sizeof(T) >= 8192 ? std::size_t{1} : 8192 / sizeof(T)); // dyn: inline capacity

  public:
    explicit tile_buffer(const View& f)
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
    struct none
    {
    };
    static constexpr std::size_t fixed_n = [] {
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
    [[no_unique_address]] std::conditional_t<is_static, std::array<T, fixed_n>, none> fixed_;
    [[no_unique_address]] std::conditional_t<is_static, none, std::array<T, inline_n>> small_;
    [[no_unique_address]] std::conditional_t<is_static, none, std::vector<T>>          heap_;
    std::size_t                                                                       n_ = 0;
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
    bool        uniform = true;
    bool        pending = false; // a constant run at the start, not yet written to buf
    struct Visitor
    {
        Buf&              buf;
        std::size_t&      k;
        bool&             uniform;
        bool&             pending;
        T&                cst;
        bool              lazy;

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

struct no_uniform
{
};

/**
 * @brief The writes of update_tiles() and fill_box(): the serial range ops
 *        outside a parallel region (the array is quiescent; deferred writes
 *        drained once, by the first write), the _mt ones inside, which are safe
 *        against the other writers and the compressor.
 */
template <typename Array>
struct range_writer
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
struct run_filler
{
    range_writer<Array>& out;
    const T&             v;
    std::size_t          lo = 0;
    std::size_t          hi = 0;

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
template <tiled_view_of_array View, typename OnBlock, typename OnUniform = detail::no_uniform>
void for_each_tile(const View&               f,
                   typename View::index_type first,
                   typename View::index_type last,
                   OnBlock&&                 on_block,
                   OnUniform&&               on_uniform = {})
{
    using T          = detail::value_of<View>;
    const auto& a    = *f.data_handle();
    const auto  dims = detail::tiles_per_dim(f);
    detail::check_tile_on_chunks(a, f.mapping().tile_elements(), "md::for_each_tile");
    constexpr bool has_uniform = !std::same_as<std::remove_cvref_t<OnUniform>, detail::no_uniform>;

    detail::tile_buffer<View, T> buf(f);
    T                            cst{};
    last = std::min(last, tile_count(f));
    for (auto k = first; k < last; ++k)
    {
        const auto t       = detail::tile_at(f, dims, k);
        const bool uniform = detail::gather(a, detail::tile_storage(f, t), buf, cst, has_uniform);
        if constexpr (has_uniform)
        {
            if (uniform)
            {
                on_uniform(t, std::as_const(cst));
                continue;
            }
        }
        on_block(t, detail::make_block<View, const T>(f, buf.data()));
    }
}

/// @brief for_each_tile() over every tile of @p f.
template <tiled_view_of_array View, typename OnBlock, typename OnUniform = detail::no_uniform>
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
 *          guarantees). Outside a region, on Chunk2A, a whole chunk that comes
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
template <tiled_view_of_array View, typename OnBlock, typename OnUniform = detail::no_uniform>
    requires(!std::is_const_v<typename View::element_type>)
void update_tiles(const View&               f,
                  typename View::index_type first,
                  typename View::index_type last,
                  OnBlock&&                 on_block,
                  OnUniform&&               on_uniform = {})
{
    using T           = detail::value_of<View>;
    const std::size_t N    = static_cast<std::size_t>(f.mapping().tile_elements());
    auto&             a    = *f.data_handle();
    const auto        dims = detail::tiles_per_dim(f);
    detail::check_tile_on_chunks(a, f.mapping().tile_elements(), "md::update_tiles");
    constexpr bool has_uniform = !std::same_as<std::remove_cvref_t<OnUniform>, detail::no_uniform>;

    detail::tile_buffer<View, T>                 buf(f);
    detail::tile_buffer<View, T>                 before(f); // a materialised tile as read: written back only if changed
    T                                            cst{};
    detail::range_writer<detail::array_of<View>> out{a};
    last = std::min(last, tile_count(f));
    for (auto k = first; k < last; ++k)
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
                    continue;
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
    }
}

/// @brief update_tiles() over every tile of @p f.
template <tiled_view_of_array View, typename OnBlock, typename OnUniform = detail::no_uniform>
    requires(!std::is_const_v<typename View::element_type>)
void update_tiles(const View& f, OnBlock&& on_block, OnUniform&& on_uniform = {})
{
    update_tiles(f, typename View::index_type{0}, tile_count(f), std::forward<OnBlock>(on_block),
                 std::forward<OnUniform>(on_uniform));
}

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
 * @warning A fill is a write. Outside a parallel region on Chunk2A, a covered
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
              const detail::value_of<View>&      v,
              typename View::index_type          first,
              typename View::index_type          last)
{
    using Index       = typename View::index_type;
    constexpr auto R  = View::rank();
    const auto     N  = static_cast<std::size_t>(f.mapping().tile_elements());
    auto           lo = detail::clipped_box<View>(box_lo);
    auto           hi = detail::clipped_box<View>(box_hi);
    if (!detail::clip_box(f, lo, hi))
    {
        return;
    }
    auto&       a    = *f.data_handle();
    const auto  dims = detail::tiles_per_dim(f);
    const Index nt   = tile_count(f);
    last             = std::min(last, nt);
    if (first >= last)
    {
        return;
    }
    detail::range_writer<detail::array_of<View>>                       out{a};
    detail::run_filler<detail::array_of<View>, detail::value_of<View>> runs{out, v};

    // The tiles the box touches, per dimension: [tlo[r], thi[r]).
    std::array<Index, R> tlo{};
    std::array<Index, R> thi{};
    for (std::size_t r = 0; r < R; ++r)
    {
        const auto te = f.mapping().tile_extent(static_cast<typename View::rank_type>(r));
        tlo[r]        = lo[r] / te;
        thi[r]        = (hi[r] - 1) / te + 1;
    }
    // Tiles [first, last) are rows of tiles in storage order: only those rows
    // of the box are walked (a thread given a slice of the tiles skips the rest).
    Index rest = 1;
    for (std::size_t r = 1; r < R; ++r)
    {
        rest *= dims[r];
    }
    tlo[0] = std::max(tlo[0], first / rest);
    thi[0] = std::min(thi[0], (last - 1) / rest + 1);
    if (tlo[0] >= thi[0])
    {
        return;
    }
    std::array<Index, R> c = tlo; // tile coordinates, row-major over the touched ones
    while (true)
    {
        Index k = 0;
        for (std::size_t r = 0; r < R; ++r)
        {
            k = k * dims[r] + c[r];
        }
        if (k >= first && k < last)
        {
            const auto t     = detail::tile_at(f, dims, k);
            bool       whole = true;
            for (std::size_t r = 0; r < R; ++r)
            {
                whole = whole && lo[r] <= t.origin[r] && hi[r] >= t.origin[r] + t.extent[r];
            }
            if (whole)
            {
                const std::size_t s = detail::tile_storage(f, t);
                runs.add(s, s + N);
            }
            else
            {
                runs.flush();
                update_tiles(
                    f, k, k + 1,
                    [&](const auto& tt, auto b)
                    {
                        std::array<Index, R> i{};
                        for (std::size_t r = 0; r < R; ++r)
                        {
                            i[r] = std::max(lo[r], tt.origin[r]) - tt.origin[r];
                        }
                        const std::array<Index, R> i0 = i;
                        while (true) // every cell of tile x box
                        {
                            b[i]          = v;
                            std::size_t r = R;
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
                    [&](const auto&, detail::value_of<View>& u) { return u == v; });
            }
        }
        std::size_t r = R;
        while (r-- > 0)
        {
            if (++c[r] < thi[r])
            {
                break;
            }
            c[r] = tlo[r];
        }
        if (r == static_cast<std::size_t>(-1))
        {
            break;
        }
    }
    runs.flush();
}

/// @brief fill_box() over every tile of @p f.
template <tiled_view_of_array View, std::integral I>
    requires(!std::is_const_v<typename View::element_type>)
void fill_box(const View&                        f,
              const std::array<I, View::rank()>& lo,
              const std::array<I, View::rank()>& hi,
              const detail::value_of<View>&      v)
{
    fill_box(f, lo, hi, v, typename View::index_type{0}, tile_count(f));
}

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
template <typename View, std::integral I>
    requires(View::mapping_type::is_always_strided() && !std::is_const_v<typename View::element_type> &&
             std::same_as<typename View::accessor_type, accessor<typename View::accessor_type::array_type>>)
void fill_box(const View&                        f,
              const std::array<I, View::rank()>& box_lo,
              const std::array<I, View::rank()>& box_hi,
              const detail::value_of<View>&      v)
{
    using Index       = typename View::index_type;
    constexpr auto R  = View::rank();
    auto           lo = detail::clipped_box<View>(box_lo);
    auto           hi = detail::clipped_box<View>(box_hi);
    if (!detail::clip_box(f, lo, hi))
    {
        return;
    }
    detail::range_writer<detail::array_of<View>>                       out{*f.data_handle()};
    detail::run_filler<detail::array_of<View>, detail::value_of<View>> runs{out, v};
    const auto base = static_cast<std::size_t>(f.data_handle().base);
    // Contiguous along the last dimension: one run per row of the box. Else
    // one run per element (the last dimension walked too).
    const bool           rows = f.stride(R - 1) == 1;
    const std::size_t    walk = rows ? R - 1 : R; // dimensions walked index by index
    std::array<Index, R> i    = lo;
    while (true)
    {
        const auto a =
            base + static_cast<std::size_t>(std::apply([&f](auto... x) { return f.mapping()(x...); }, i));
        runs.add(a, a + (rows ? static_cast<std::size_t>(hi[R - 1] - lo[R - 1]) : std::size_t{1}));
        {
            std::size_t r = walk;
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
    runs.flush();
}

} // namespace shrink_array::md

#endif // SHRINK_ARRAY_HAS_MDSPAN
