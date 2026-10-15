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
 *        reads, update_tiles() reads and writes back, a uniform tile in O(1).
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

    #include <algorithm>   // std::copy, std::fill, std::equal, std::all_of
    #include <array>       // std::array (the tile buffer, the tile coordinates)
    #include <concepts>    // std::invocable, std::same_as
    #include <cstddef>     // std::size_t
    #include <span>        // std::span (the blocks of for_each_span)
    #include <type_traits> // std::remove_cvref_t, std::is_const_v
    #include <utility>     // std::as_const, std::forward

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

/// @brief A tiled view of this library: layout_tiled and md::accessor.
template <typename View>
concept tiled_view_of_array = requires {
    View::mapping_type::tile_elems; // layout_tiled
    requires std::same_as<typename View::accessor_type, accessor<typename View::accessor_type::array_type>> ||
                 std::same_as<typename View::accessor_type, accessor<const typename View::accessor_type::array_type>>;
};

namespace detail
{
template <typename View>
using array_of = typename View::accessor_type::array_type;

template <typename View>
using value_of = typename array_of<View>::value_type;

/// @brief The block type a callback receives: static extents Tile..., row-major.
template <typename View, typename E, std::size_t... Tile>
auto block_type_of(layout_tiled<Tile...>*) -> mdns::mdspan<E, mdns::extents<typename View::index_type, Tile...>>;

template <typename View, typename E>
using block_t = decltype(block_type_of<View, E>(static_cast<typename View::layout_type*>(nullptr)));

/// @brief Tiles per dimension of view @p f.
template <typename View>
std::array<typename View::index_type, View::rank()> tiles_per_dim(const View& f)
{
    using M = typename View::mapping_type;
    std::array<typename View::index_type, View::rank()> n{};
    for (std::size_t r = 0; r < View::rank(); ++r)
    {
        n[r] = (f.extent(r) + M::tile[r] - 1) / M::tile[r];
    }
    return n;
}

/// @brief The tile of storage rank @p k: its origin and its extent inside the view.
template <typename View>
tile<typename View::index_type, View::rank()>
tile_at(const View& f, const std::array<typename View::index_type, View::rank()>& per_dim, typename View::index_type k)
{
    using M = typename View::mapping_type;
    tile<typename View::index_type, View::rank()> t{k, {}, {}};
    auto                                          rest = k;
    for (std::size_t r = View::rank(); r-- > 0;)
    {
        const auto c = rest % per_dim[r];
        rest /= per_dim[r];
        t.origin[r] = c * M::tile[r];
        t.extent[r] = std::min<typename View::index_type>(M::tile[r], f.extent(r) - t.origin[r]);
    }
    return t;
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
 * @brief Read storage elements [lo, lo + N) of @p a into @p buf.
 * @return true when they are all one constant @p cst; @p buf is then left
 *         unfilled if @p lazy, so a uniform tile costs O(1).
 */
template <typename Array, typename T, std::size_t N>
bool gather(const Array& a, std::size_t lo, std::array<T, N>& buf, T& cst, bool lazy)
{
    std::size_t k       = 0;
    bool        uniform = true;
    bool        pending = false; // a constant run at the start, not yet written to buf
    struct Visitor
    {
        std::array<T, N>& buf;
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
} // namespace detail

/// @brief Number of tiles of tiled view @p f (padding tiles included): the range of the range overloads.
template <tiled_view_of_array View>
typename View::index_type tile_count(const View& f)
{
    return static_cast<typename View::index_type>(f.mapping().required_span_size() / View::mapping_type::tile_elems);
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
    using T             = detail::value_of<View>;
    using M             = typename View::mapping_type;
    constexpr auto N    = static_cast<std::size_t>(M::tile_elems);
    const auto&    a    = *f.data_handle();
    const auto     dims = detail::tiles_per_dim(f);
    detail::check_tile_on_chunks(a, M::tile_elems, "md::for_each_tile");
    constexpr bool has_uniform = !std::same_as<std::remove_cvref_t<OnUniform>, detail::no_uniform>;

    std::array<T, N> buf;
    T                cst{};
    last = std::min(last, tile_count(f));
    for (auto k = first; k < last; ++k)
    {
        const auto t       = detail::tile_at(f, dims, k);
        const bool uniform = detail::gather(a, static_cast<std::size_t>(k) * N, buf, cst, has_uniform);
        if constexpr (has_uniform)
        {
            if (uniform)
            {
                on_uniform(t, std::as_const(cst));
                continue;
            }
        }
        on_block(t, detail::block_t<View, const T>(buf.data()));
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
 *          copy_range(). Outside a parallel region these are the serial range
 *          ops, which set a chunk constant without materialising it (any value
 *          for Chunk2A, T{} for all). Inside, fill_range_mt() / copy_range_mt(),
 *          one write session per chunk with the element path's guarantees; they
 *          never release a chunk: a tile made uniform stays materialised,
 *          queued for the next compression pass.
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
    using T             = detail::value_of<View>;
    using M             = typename View::mapping_type;
    constexpr auto N    = static_cast<std::size_t>(M::tile_elems);
    auto&          a    = *f.data_handle();
    const auto     dims = detail::tiles_per_dim(f);
    detail::check_tile_on_chunks(a, M::tile_elems, "md::update_tiles");
    constexpr bool has_uniform = !std::same_as<std::remove_cvref_t<OnUniform>, detail::no_uniform>;

    std::array<T, N> buf;
    std::array<T, N> before; // a materialised tile as read: written back only if changed
    T                cst{};
    // Outside a parallel region the serial range ops (quiescent array): a
    // uniform result sets the chunk constant without materialising it. Inside,
    // the _mt ones, safe against the other writers and the compressor.
    const bool in_region = CHUNK_PMP_IN_PARALLEL_REGION;
    bool       drained   = false; // serial: deferred writes flushed once, by the first write
    auto       fill      = [&](std::size_t lo, const T& v)
    {
        if (in_region)
        {
            fill_range_mt(a, lo, lo + N, v);
        }
        else
        {
            fill_range(a, lo, lo + N, v, !drained);
            drained = true;
        }
    };
    auto copy = [&](std::size_t lo)
    {
        if (in_region)
        {
            copy_range_mt(a, lo, lo + N, buf.data());
        }
        else
        {
            copy_range(a, lo, lo + N, buf.data(), !drained);
            drained = true;
        }
    };
    last = std::min(last, tile_count(f));
    for (auto k = first; k < last; ++k)
    {
        const auto        t       = detail::tile_at(f, dims, k);
        const std::size_t lo      = static_cast<std::size_t>(k) * N;
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
                        fill(lo, v);
                    }
                    continue;
                }
                std::fill(buf.begin(), buf.end(), cst);
            }
        }
        if (!uniform)
        {
            before = buf;
        }
        on_block(t, detail::block_t<View, T>(buf.data()));

        const T    first_v = buf[0];
        const bool now_uniform =
            std::all_of(buf.begin() + 1, buf.end(), [&first_v](const T& x) { return x == first_v; });
        if (now_uniform)
        {
            if (!(uniform && first_v == cst))
            {
                fill(lo, first_v);
            }
        }
        else if (uniform || !std::equal(buf.begin(), buf.end(), before.begin()))
        {
            copy(lo);
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

} // namespace shrink_array::md

#endif // SHRINK_ARRAY_HAS_MDSPAN
