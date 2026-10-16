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
 * @file mdspan.h
 * @brief A multidimensional view (mdspan) over a compressed array.
 *
 * @details The array stays the owner: it holds the elements, compresses and
 *          re-chunks them, and keeps its whole interface. The view adds
 *          multi-index access, `f[i, j]`, through an mdspan accessor whose
 *          reference is the array's own (the write proxy, an ElementRef in
 *          check builds): every write goes through the compression engine,
 *          as `a[k]` does.
 *
 *          @code
 *          ArrayChunk2A<double> a(nx * ny);
 *          auto f = shrink_array::md::view(a, nx, ny);   // row-major
 *          f[i, j] = 1.0;                                // same as a[i * ny + j] = 1.0
 *
 *          // Tiles of 16 x 16 = 256 elements: one tile per chunk of a _256
 *          // array, so a uniform 2D block is a uniform chunk.
 *          ArrayChunk2A<double> b(shrink_array::md::tiled_size<16, 16>(nx, ny));
 *          auto g = shrink_array::md::tiled_view<16, 16>(b, nx, ny);
 *          @endcode
 *
 *          The mdspan comes from <mdspan> (C++23, __cpp_lib_mdspan: libstdc++
 *          15, libc++ 17), else from the reference implementation
 *          (kokkos/mdspan, <mdspan/mdspan.hpp>) when it is on the include
 *          path. With neither, this header defines nothing but
 *          SHRINK_ARRAY_HAS_MDSPAN = 0.
 *
 * @warning The view does not own the array: it must not outlive it (as a
 *          std::span over a std::vector). After a resize, make a new view:
 *          its extents no longer match. A re-chunk keeps it correct (it goes
 *          through operator[]), but a tiled layout is only aligned on the
 *          chunks while the chunk size equals the tile size.
 *
 *          Sub-views: md::submdspan(f, slices...) (std::submdspan or
 *          Kokkos::submdspan). A row-major view takes any slices (the
 *          standard layouts' own submdspan_mapping); a tiled view takes
 *          rank-preserving slices aligned on its tiles -- each range starts
 *          on a tile and ends on a tile or at the view's extent -- and its
 *          sub-view is tiled again, its tiles still chunks
 *          (layout_tiled::mapping's submdspan_mapping).
 *
 * @warning Not a pointer: data_handle() is an md::handle -- the array's
 *          address and the index of the view's first element -- not a T*;
 *          *data_handle() is the array. Code that wants a contiguous T*
 *          still has to copy. The rules of the array apply
 *          unchanged: OpenMP phases bracketed, no proxy kept across a resize
 *          or a re-chunk, and in check builds `auto x = f[i, j];` does not
 *          compile (take a value: `double x = f[i, j];`).
 */

#if __has_include(<mdspan>)
    #include <mdspan> // std::mdspan, std::extents, std::layout_right (C++23)
#endif

#if defined(__cpp_lib_mdspan) && __cpp_lib_mdspan >= 202207L
    #define SHRINK_ARRAY_HAS_MDSPAN 1
    #define SHRINK_ARRAY_MDSPAN_NS std
#elif __has_include(<mdspan/mdspan.hpp>)
    #include <mdspan/mdspan.hpp> // Kokkos::mdspan, the reference implementation
    #define SHRINK_ARRAY_HAS_MDSPAN 1
    #define SHRINK_ARRAY_MDSPAN_NS MDSPAN_IMPL_STANDARD_NAMESPACE
    #define SHRINK_ARRAY_HAS_SUBMDSPAN 1 // Kokkos::submdspan, in every version that has <mdspan/mdspan.hpp>
#else
    #define SHRINK_ARRAY_HAS_MDSPAN 0
#endif

// std::submdspan is C++26 (__cpp_lib_submdspan); the reference
// implementation has it (above).
#if !defined(SHRINK_ARRAY_HAS_SUBMDSPAN)
    #if SHRINK_ARRAY_HAS_MDSPAN && defined(__cpp_lib_submdspan)
        #define SHRINK_ARRAY_HAS_SUBMDSPAN 1
    #else
        #define SHRINK_ARRAY_HAS_SUBMDSPAN 0
    #endif
#endif

#if SHRINK_ARRAY_HAS_MDSPAN

    #include <array>       // std::array (the tile extents, the tile strides)
    #include <cstddef>     // std::size_t
    #include <type_traits> // std::remove_const_t, std::is_const_v, std::conditional_t, std::make_unsigned_t, std::is_convertible_v
    #include <utility>     // std::declval, std::index_sequence

    #include "chunk/internal/invariant_check.h" // CHUNK_INVARIANT_FAILED (a tiled submdspan off the tiles)

namespace shrink_array::md
{

namespace mdns = SHRINK_ARRAY_MDSPAN_NS;

/// @brief The index type of @p Array: the type of a.size() (as array_chunk_range_ops.h).
template <typename Array>
using index_of = decltype(std::declval<const std::remove_const_t<Array>&>().size());

/**
 * @brief The data handle of a view: the array, and the index in it of the
 *        view's element at offset 0 (non-zero for a sub-view).
 * @details *h is the array, as when the handle was the array's address.
 */
template <typename Array>
struct handle
{
    using index_type = index_of<Array>;

    Array*     array = nullptr; ///< The array the view reads and writes.
    index_type base  = 0;       ///< Index in the array of the view's offset 0.

    constexpr handle() noexcept = default;
    constexpr handle(Array* a, index_type b = 0) noexcept // NOLINT(google-explicit-constructor): from &a
        : array(a), base(b)
    {
    }
    /// @brief A read-only handle from a writable one (mdspan<const T> from mdspan<T>).
    template <typename Other>
        requires(std::is_const_v<Array> && std::is_same_v<Other, std::remove_const_t<Array>>)
    constexpr handle(const handle<Other>& h) noexcept // NOLINT(google-explicit-constructor)
        : array(h.array), base(h.base)
    {
    }

    constexpr Array& operator*() const noexcept
    {
        return *array;
    }
    constexpr Array* operator->() const noexcept
    {
        return array;
    }
};

/**
 * @brief mdspan accessor over an array of this library (ArrayChunk, ArrayChunk2,
 *        ArrayChunk2A, or any type with value_type, reference and operator[]).
 * @tparam Array The array type; `const Array` gives a read-only view whose
 *         reference is the value (T), like the array's const operator[].
 */
template <typename Array>
struct accessor
{
    using array_type   = std::remove_const_t<Array>;
    using element_type = std::
        conditional_t<std::is_const_v<Array>, const typename array_type::value_type, typename array_type::value_type>;
    using data_handle_type = handle<Array>;
    using reference        = std::
        conditional_t<std::is_const_v<Array>, typename array_type::const_reference, typename array_type::reference>;
    using offset_policy = accessor;

    constexpr accessor() noexcept = default;

    /// @brief A read-only accessor from a writable one (mdspan<const T> from mdspan<T>).
    template <typename Other>
        requires(std::is_const_v<Array> && std::is_same_v<std::remove_const_t<Other>, array_type>)
    constexpr accessor(accessor<Other>) noexcept // NOLINT(google-explicit-constructor)
    {
    }

    /// @brief Element @p i of the view: the array's operator[] at base + i,
    ///        the proxy or the value.
    reference access(data_handle_type h, std::size_t i) const noexcept
    {
        return (*h.array)[static_cast<index_of<Array>>(static_cast<std::size_t>(h.base) + i)];
    }

    /// @brief The handle of a sub-view starting @p i elements further (submdspan).
    data_handle_type offset(data_handle_type h, std::size_t i) const noexcept
    {
        return {h.array, static_cast<index_of<Array>>(static_cast<std::size_t>(h.base) + i)};
    }
};

namespace detail
{
/**
 * @brief One slice of a tiled submdspan, as [first, first + size): the
 *        slice forms of the reference implementation before and after its
 *        canonical slices (full_extent; a pair or tuple {first, last}; a
 *        strided_slice or extent_slice {offset, extent, stride}; a range
 *        slice {first, last, stride}), stride 1. An index (which drops the
 *        dimension) does not compile: a tiled sub-view keeps every dimension.
 */
template <typename Index, typename Slice>
constexpr void tiled_slice(const Slice& s, Index extent, Index& first, Index& size, Index& stride)
{
    stride = 1;
    if constexpr (std::is_convertible_v<Slice, mdns::full_extent_t>)
    {
        first = 0;
        size  = extent;
    }
    else if constexpr (std::is_convertible_v<Slice, Index>)
    {
        static_assert(!std::is_convertible_v<Slice, Index>,
                      "md::layout_tiled: a submdspan of a tiled view keeps every dimension -- "
                      "take a range {first, last}, not an index");
    }
    else if constexpr (requires { s.offset; s.extent; s.stride; })
    {
        first  = static_cast<Index>(s.offset);
        size   = static_cast<Index>(s.extent);
        stride = static_cast<Index>(s.stride);
    }
    else if constexpr (requires { s.first; s.last; s.stride; })
    {
        first  = static_cast<Index>(s.first);
        size   = static_cast<Index>(s.last) - first;
        stride = static_cast<Index>(s.stride);
    }
    else
    {
        const auto& [lo, hi] = s; // pair-like {first, last}
        first                = static_cast<Index>(lo);
        size                 = static_cast<Index>(hi) - first;
    }
}
} // namespace detail

/**
 * @brief mdspan layout in tiles: the extents cut into tiles of Tile... elements
 *        per dimension, the tiles stored one after the other (row-major), each
 *        tile row-major inside.
 * @details With a tile of as many elements as a chunk (16 x 16 for 256,
 *          32 x 32 for 1024, 8 x 8 x 4 for 256 in 3D), tile k IS chunk k: a
 *          uniform block of the field makes a uniform chunk, which compresses.
 *          In layout_right a chunk is a piece of a row, and a uniform block
 *          only gives uniform chunks where its rows are longer than a chunk.
 *          The extents need not be multiples of the tile: the last tiles are
 *          padded, required_span_size() counts the padding (tiled_size()).
 * @tparam Tile Tile extent in each dimension (rank = sizeof...(Tile)).
 */
template <std::size_t... Tile>
struct layout_tiled
{
    template <typename Extents>
    class mapping
    {
      public:
        using extents_type = Extents;
        using index_type   = typename Extents::index_type;
        using size_type    = typename Extents::size_type;
        using rank_type    = typename Extents::rank_type;
        using layout_type  = layout_tiled;

        static_assert(Extents::rank() == sizeof...(Tile), "layout_tiled: one tile extent per dimension");
        static_assert(((Tile > 0) && ...), "layout_tiled: tile extents must be positive");

        static constexpr std::array<index_type, sizeof...(Tile)> tile{static_cast<index_type>(Tile)...};
        static constexpr index_type                              tile_elems = static_cast<index_type>((Tile * ... * 1));

        constexpr mapping() noexcept = default;
        constexpr explicit mapping(const extents_type& e) noexcept : ext_(e)
        {
            for (rank_type r = 0; r < Extents::rank(); ++r)
            {
                ntiles_[r] = (ext_.extent(r) + tile[r] - 1) / tile[r];
            }
            // Row-major over the tiles.
            index_type s = 1;
            for (rank_type r = Extents::rank(); r-- > 0;)
            {
                tstride_[r] = s;
                s *= ntiles_[r];
            }
        }

        /// @brief A sub-view's mapping: extents @p e laid in a tile grid of
        ///        strides @p tile_strides (its parent's; see submdspan_mapping).
        constexpr mapping(const extents_type& e, const std::array<index_type, sizeof...(Tile)>& tile_strides) noexcept
            : mapping(e)
        {
            tstride_ = tile_strides;
        }

        /// @brief Tiles between two tiles one apart in dimension r, in storage
        ///        order: the view's own grid, or its parent's for a sub-view.
        constexpr const std::array<index_type, sizeof...(Tile)>& tile_strides() const noexcept
        {
            return tstride_;
        }

        constexpr const extents_type& extents() const noexcept
        {
            return ext_;
        }

        /// @brief All the tiles, padding included: the array size the view needs
        ///        (from its offset 0: for a sub-view, up to its last tile).
        constexpr index_type required_span_size() const noexcept
        {
            if constexpr (Extents::rank() == 0)
            {
                return 1;
            }
            index_type last = 0; // the last tile, in storage order
            for (rank_type r = 0; r < Extents::rank(); ++r)
            {
                if (ntiles_[r] == 0)
                {
                    return 0;
                }
                last += (ntiles_[r] - 1) * tstride_[r];
            }
            return (last + 1) * tile_elems;
        }

        template <typename... Idx>
            requires(sizeof...(Idx) == Extents::rank())
        constexpr index_type operator()(Idx... idx) const noexcept
        {
            // Unsigned: the indices are non-negative (mdspan's precondition), and
            // a division of an unsigned by a power of two is a shift -- the signed
            // one needs a correction for negative values: about a tenth of a
            // tiled f[i, j] read (mdspan_tiles, index cost).
            using U = std::make_unsigned_t<index_type>;
            const std::array<U, sizeof...(Idx)> i{static_cast<U>(idx)...};
            U                                   which = 0; // the tile, row-major over the tiles
            U                                   in    = 0; // the element in its tile, row-major
            for (rank_type r = 0; r < Extents::rank(); ++r)
            {
                const auto t = static_cast<U>(tile[r]);
                which += (i[r] / t) * static_cast<U>(tstride_[r]);
                in = in * t + i[r] % t;
            }
            return static_cast<index_type>(which * static_cast<U>(tile_elems) + in);
        }

        static constexpr bool is_always_unique() noexcept
        {
            return true;
        }
        static constexpr bool is_always_exhaustive() noexcept
        {
            return false; // padded when an extent is not a multiple of its tile
        }
        static constexpr bool is_always_strided() noexcept
        {
            return false;
        }
        static constexpr bool is_unique() noexcept
        {
            return true;
        }
        constexpr bool is_exhaustive() const noexcept
        {
            index_type s = 1; // a sub-view's tiles are spread over its parent's grid
            for (rank_type r = Extents::rank(); r-- > 0;)
            {
                if (ext_.extent(r) % tile[r] != 0 || tstride_[r] != s)
                {
                    return false;
                }
                s *= ntiles_[r];
            }
            return true;
        }
        static constexpr bool is_strided() noexcept
        {
            return false;
        }

        template <typename OtherExtents>
        friend constexpr bool operator==(const mapping& a, const mapping<OtherExtents>& b) noexcept
        {
            return a.extents() == b.extents() && a.tile_strides() == b.tile_strides();
        }

    #if SHRINK_ARRAY_HAS_SUBMDSPAN
        /**
         * @brief submdspan of a tiled view: a tiled view again, over the same
         *        tiles (each still one chunk), in its parent's tile grid.
         * @details Every slice keeps its dimension, with stride 1, and is
         *          aligned on the tiles: it starts on a tile and ends on a
         *          tile or at the view's extent. Then a tile of the sub-view
         *          is a tile of the parent, and its padding (past the
         *          sub-view's extent) is the parent's padding: the tile
         *          operations, which read and write whole tiles, touch
         *          nothing outside the sub-view. A slice off the tiles stops
         *          the program.
         */
        template <typename... Slices>
        friend constexpr auto submdspan_mapping(const mapping& m, Slices... slices)
        {
            static_assert(sizeof...(Slices) == Extents::rank(), "md::layout_tiled: one slice per dimension");
            constexpr std::size_t                   R = Extents::rank();
            std::array<index_type, R>               first{};
            std::array<index_type, R>               size{};
            std::array<index_type, R>               stride{};
            [&]<std::size_t... K>(std::index_sequence<K...>)
            {
                (detail::tiled_slice<index_type>(slices, m.ext_.extent(K), first[K], size[K], stride[K]), ...);
            }(std::index_sequence_for<Slices...>{});

            for (std::size_t r = 0; r < R; ++r)
            {
                const index_type end = first[r] + size[r];
                const bool       ok  = first[r] >= 0 && size[r] >= 0 && end <= m.ext_.extent(r) &&
                                (stride[r] == 1 || size[r] <= 1) && first[r] % tile[r] == 0 &&
                                (end % tile[r] == 0 || end == m.ext_.extent(r));
                if (!ok) [[unlikely]]
                {
                    CHUNK_INVARIANT_FAILED("md::layout_tiled: submdspan slice {} = [{}, {}) (stride {}) of extent {}: "
                                           "a tiled sub-view takes ranges of stride 1 that start on a tile of {} "
                                           "and end on one or at the extent",
                                           r, static_cast<long long>(first[r]), static_cast<long long>(end),
                                           static_cast<long long>(stride[r]), static_cast<long long>(m.ext_.extent(r)),
                                           static_cast<long long>(tile[r]));
                }
            }

            using sub_extents = mdns::dextents<index_type, R>;
            const mapping<sub_extents> sub(sub_extents(size), m.tstride_);
            index_type which = 0; // the sub-view's first tile, in storage order
            for (std::size_t r = 0; r < R; ++r)
            {
                which += (first[r] / tile[r]) * m.tstride_[r];
            }
            return mdns::submdspan_mapping_result<mapping<sub_extents>>{
                sub, static_cast<std::size_t>(which) * static_cast<std::size_t>(tile_elems)};
        }
    #endif

      private:
        extents_type                            ext_{};
        std::array<index_type, sizeof...(Tile)> ntiles_{};
        std::array<index_type, sizeof...(Tile)> tstride_{}; ///< see tile_strides()
    };
};

/// @brief The view type of view() and tiled_view().
template <typename Array, typename Layout = mdns::layout_right, std::size_t Rank = 2>
using view_type = mdns::
    mdspan<typename accessor<Array>::element_type, mdns::dextents<index_of<Array>, Rank>, Layout, accessor<Array>>;

/**
 * @brief Row-major view of @p a with extents @p ext... (their product must not
 *        exceed a.size()).
 * @return an mdspan whose f[i, j, ...] is a[i * e1 * ... + j * ... + ...].
 */
template <typename Array, typename... Ext>
auto view(Array& a, Ext... ext)
{
    using A  = std::remove_reference_t<Array>;
    using VT = view_type<A, mdns::layout_right, sizeof...(Ext)>;
    return VT(handle<A>(&a), typename VT::mapping_type(typename VT::extents_type(ext...)), accessor<A>{});
}

/// @brief Elements a tiled view of extents @p ext... needs (padding included).
template <std::size_t... Tile, typename... Ext>
constexpr std::size_t tiled_size(Ext... ext) noexcept
{
    static_assert(sizeof...(Tile) == sizeof...(Ext), "tiled_size: one tile extent per dimension");
    return ((((static_cast<std::size_t>(ext) + Tile - 1) / Tile) * Tile) * ... * std::size_t{1});
}

/**
 * @brief Tiled view of @p a (layout_tiled<Tile...>) with extents @p ext...;
 *        @p a must hold at least tiled_size<Tile...>(ext...) elements.
 */
template <std::size_t... Tile, typename Array, typename... Ext>
auto tiled_view(Array& a, Ext... ext)
{
    using A  = std::remove_reference_t<Array>;
    using VT = view_type<A, layout_tiled<Tile...>, sizeof...(Ext)>;
    return VT(handle<A>(&a), typename VT::mapping_type(typename VT::extents_type(ext...)), accessor<A>{});
}

    #if SHRINK_ARRAY_HAS_SUBMDSPAN
/// @brief submdspan and full_extent of the mdspan in use (std or Kokkos), so
///        that md::submdspan(f, std::pair{0, 64}, md::full_extent) reads the
///        same whichever it is.
using mdns::full_extent;
using mdns::submdspan;
    #endif

} // namespace shrink_array::md

#endif // SHRINK_ARRAY_HAS_MDSPAN
