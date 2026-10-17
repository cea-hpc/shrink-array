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
 *
 *          // The tile at run time (a _dyn library): a.chunk_size() elements.
 *          auto h = shrink_array::md::tiled_view_dyn(c, nx, ny);
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
    #include <utility>     // std::declval, std::index_sequence, std::pair

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

    #if SHRINK_ARRAY_HAS_SUBMDSPAN
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
    #endif

namespace detail
{
/**
 * @brief What layout_tiled and layout_tiled_dyn share: the extents, the tile
 *        grid and its strides, the index, the span, and the sub-view of a
 *        submdspan. @p Derived gives the tile: tile_extent(r) and
 *        tile_elements() (constants for layout_tiled, members for
 *        layout_tiled_dyn).
 */
template <typename Derived, typename Extents>
class tiled_mapping_base
{
  public:
    using extents_type = Extents;
    using index_type   = typename Extents::index_type;
    using size_type    = typename Extents::size_type;
    using rank_type    = typename Extents::rank_type;
    using grid_type    = std::array<index_type, Extents::rank()>;

    constexpr const extents_type& extents() const noexcept
    {
        return ext_;
    }

    /// @brief Tiles between two tiles one apart in dimension r, in storage
    ///        order: the view's own grid, or its parent's for a sub-view.
    constexpr const grid_type& tile_strides() const noexcept
    {
        return tstride_;
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
        return (last + 1) * self().tile_elements();
    }

    template <typename... Idx>
        requires(sizeof...(Idx) == Extents::rank())
    constexpr index_type operator()(Idx... idx) const noexcept
    {
        // Unsigned: the indices are non-negative (mdspan's precondition), and
        // a division of an unsigned by a power of two is a shift -- the signed
        // one needs a correction for negative values: about a tenth of a
        // tiled f[i, j] read (mdspan_tiles, index cost). A tile extent known
        // only at run time (layout_tiled_dyn) is a real division.
        using U = std::make_unsigned_t<index_type>;
        const std::array<U, sizeof...(Idx)> i{static_cast<U>(idx)...};
        U                                   which = 0; // the tile, in the grid's storage order
        U                                   in    = 0; // the element in its tile, row-major
        for (rank_type r = 0; r < Extents::rank(); ++r)
        {
            const auto t = static_cast<U>(self().tile_extent(r));
            which += (i[r] / t) * static_cast<U>(tstride_[r]);
            in = in * t + i[r] % t;
        }
        return static_cast<index_type>(which * static_cast<U>(self().tile_elements()) + in);
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
            if (ext_.extent(r) % self().tile_extent(r) != 0 || tstride_[r] != s)
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

  protected:
    constexpr tiled_mapping_base() noexcept = default;

    /// @brief Extents @p e cut in the tiles of the derived mapping (set
    ///        before this runs), row-major over the tiles.
    constexpr void init(const extents_type& e) noexcept
    {
        ext_ = e;
        for (rank_type r = 0; r < Extents::rank(); ++r)
        {
            ntiles_[r] = (ext_.extent(r) + self().tile_extent(r) - 1) / self().tile_extent(r);
        }
        index_type s = 1;
        for (rank_type r = Extents::rank(); r-- > 0;)
        {
            tstride_[r] = s;
            s *= ntiles_[r];
        }
    }
    constexpr void set_tile_strides(const grid_type& g) noexcept
    {
        tstride_ = g;
    }

    #if SHRINK_ARRAY_HAS_SUBMDSPAN
    /// @brief A submdspan's sub-view: its extents and its offset. Every slice
    ///        keeps its dimension, with stride 1, and is aligned on the tiles
    ///        (it starts on a tile and ends on one or at the extent); a slice
    ///        off the tiles stops the program. @p who names the layout.
    template <typename... Slices>
    constexpr std::pair<grid_type, std::size_t> sub_view(const char* who, Slices... slices) const
    {
        static_assert(sizeof...(Slices) == Extents::rank(), "md::layout_tiled: one slice per dimension");
        constexpr std::size_t R = Extents::rank();
        grid_type             first{};
        grid_type             size{};
        grid_type             stride{};
        [&]<std::size_t... K>(std::index_sequence<K...>)
        {
            (detail::tiled_slice<index_type>(slices, ext_.extent(K), first[K], size[K], stride[K]), ...);
        }(std::index_sequence_for<Slices...>{});

        index_type which = 0; // the sub-view's first tile, in storage order
        for (std::size_t r = 0; r < R; ++r)
        {
            const index_type t   = self().tile_extent(r);
            const index_type end = first[r] + size[r];
            const bool ok = first[r] >= 0 && size[r] >= 0 && end <= ext_.extent(r) && (stride[r] == 1 || size[r] <= 1) &&
                            first[r] % t == 0 && (end % t == 0 || end == ext_.extent(r));
            if (!ok) [[unlikely]]
            {
                CHUNK_INVARIANT_FAILED("{}: submdspan slice {} = [{}, {}) (stride {}) of extent {}: a tiled sub-view "
                                       "takes ranges of stride 1 that start on a tile of {} and end on one or at the "
                                       "extent",
                                       who, r, static_cast<long long>(first[r]), static_cast<long long>(end),
                                       static_cast<long long>(stride[r]), static_cast<long long>(ext_.extent(r)),
                                       static_cast<long long>(t));
            }
            which += (first[r] / t) * tstride_[r];
        }
        return {size, static_cast<std::size_t>(which) * static_cast<std::size_t>(self().tile_elements())};
    }
    #endif

  private:
    constexpr const Derived& self() const noexcept
    {
        return static_cast<const Derived&>(*this);
    }

    extents_type ext_{};
    grid_type    ntiles_{};
    grid_type    tstride_{}; ///< see tile_strides()
};
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
 *          The tile known at compile time, the index divides by constants
 *          (shifts and masks for powers of two); layout_tiled_dyn takes it
 *          at run time.
 * @tparam Tile Tile extent in each dimension (rank = sizeof...(Tile)).
 */
template <std::size_t... Tile>
struct layout_tiled
{
    template <typename Extents>
    class mapping : public detail::tiled_mapping_base<mapping<Extents>, Extents>
    {
        using Base = detail::tiled_mapping_base<mapping<Extents>, Extents>;

      public:
        using typename Base::extents_type;
        using typename Base::grid_type;
        using typename Base::index_type;
        using typename Base::rank_type;
        using typename Base::size_type;
        using layout_type = layout_tiled;

        static_assert(Extents::rank() == sizeof...(Tile), "layout_tiled: one tile extent per dimension");
        static_assert(((Tile > 0) && ...), "layout_tiled: tile extents must be positive");

        static constexpr std::array<index_type, sizeof...(Tile)> tile{static_cast<index_type>(Tile)...};
        static constexpr index_type                              tile_elems = static_cast<index_type>((Tile * ... * 1));

        constexpr mapping() noexcept = default;
        constexpr explicit mapping(const extents_type& e) noexcept
        {
            this->init(e);
        }
        /// @brief A sub-view's mapping: extents @p e laid in a tile grid of
        ///        strides @p tile_strides (its parent's; see submdspan_mapping).
        constexpr mapping(const extents_type& e, const grid_type& tile_strides) noexcept
        {
            this->init(e);
            this->set_tile_strides(tile_strides);
        }

        static constexpr index_type tile_extent(rank_type r) noexcept
        {
            return tile[r];
        }
        static constexpr index_type tile_elements() noexcept
        {
            return tile_elems;
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
            using sub_extents        = mdns::dextents<index_type, Extents::rank()>;
            const auto [size, offset] = m.sub_view("md::layout_tiled", slices...);
            return mdns::submdspan_mapping_result<mapping<sub_extents>>{
                mapping<sub_extents>(sub_extents(size), m.tile_strides()), offset};
        }
    #endif
    };
};

/**
 * @brief layout_tiled with the tile given at run time: a tile per chunk of a
 *        `_dyn` library, whose chunk size is known at run time only
 *        (tiled_view_dyn(), tile_shape()).
 * @details Same storage as layout_tiled<T0, T1, ...> for the same tile: the
 *          two read the same array alike. The index divides by run-time
 *          values (a real division, where layout_tiled's constants give
 *          shifts).
 * @tparam Rank The rank.
 */
template <std::size_t Rank>
struct layout_tiled_dyn
{
    template <typename Extents>
    class mapping : public detail::tiled_mapping_base<mapping<Extents>, Extents>
    {
        using Base = detail::tiled_mapping_base<mapping<Extents>, Extents>;

      public:
        using typename Base::extents_type;
        using typename Base::grid_type;
        using typename Base::index_type;
        using typename Base::rank_type;
        using typename Base::size_type;
        using layout_type = layout_tiled_dyn;

        static_assert(Extents::rank() == Rank, "layout_tiled_dyn: the extents' rank is the layout's");

        /// @brief A one-element tile: only so that the mapping is default-constructible.
        constexpr mapping() noexcept
        {
            tile_.fill(1);
        }
        /// @brief Extents @p e cut in tiles @p tile (every extent > 0).
        constexpr mapping(const extents_type& e, const grid_type& tile) : tile_(tile)
        {
            tile_elems_ = 1;
            for (rank_type r = 0; r < Rank; ++r)
            {
                if (tile_[r] <= 0) [[unlikely]]
                {
                    CHUNK_INVARIANT_FAILED("md::layout_tiled_dyn: tile extent {} of dimension {} is not positive",
                                           static_cast<long long>(tile_[r]), r);
                }
                tile_elems_ *= tile_[r];
            }
            this->init(e);
        }
        /// @brief A sub-view's mapping: as above, in a tile grid of strides
        ///        @p tile_strides (its parent's; see submdspan_mapping).
        constexpr mapping(const extents_type& e, const grid_type& tile, const grid_type& tile_strides)
            : mapping(e, tile)
        {
            this->set_tile_strides(tile_strides);
        }

        constexpr index_type tile_extent(rank_type r) const noexcept
        {
            return tile_[r];
        }
        constexpr index_type tile_elements() const noexcept
        {
            return tile_elems_;
        }
        /// @brief The tile, per dimension.
        constexpr const grid_type& tile() const noexcept
        {
            return tile_;
        }

        template <typename OtherExtents>
        friend constexpr bool operator==(const mapping& a, const mapping<OtherExtents>& b) noexcept
        {
            return a.extents() == b.extents() && a.tile() == b.tile() && a.tile_strides() == b.tile_strides();
        }

    #if SHRINK_ARRAY_HAS_SUBMDSPAN
        /// @brief submdspan of a tiled view: see layout_tiled's.
        template <typename... Slices>
        friend constexpr auto submdspan_mapping(const mapping& m, Slices... slices)
        {
            using sub_extents        = mdns::dextents<index_type, Rank>;
            const auto [size, offset] = m.sub_view("md::layout_tiled_dyn", slices...);
            return mdns::submdspan_mapping_result<mapping<sub_extents>>{
                mapping<sub_extents>(sub_extents(size), m.tile(), m.tile_strides()), offset};
        }
    #endif

      private:
        grid_type  tile_{};
        index_type tile_elems_ = 1;
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

/**
 * @brief A tile of @p elems elements in @p Rank dimensions, as square as the
 *        divisors of @p elems allow, the longer extents last (a tile row is
 *        contiguous in the tile): 256 -> 16 x 16, 512 -> 16 x 32, 384 -> 16 x 24,
 *        64 -> 8 x 8, 256 in 3D -> 4 x 8 x 8.
 * @details Dimension r takes the largest divisor of what is left that does
 *          not exceed its fair share (the (Rank - r)-th root), so the
 *          product is exactly @p elems: tile k of a tiled view is chunk k.
 *          A prime @p elems has no better shape than 1 x elems.
 */
template <std::size_t Rank, typename Index = std::size_t>
constexpr std::array<Index, Rank> tile_shape(std::size_t elems) noexcept
{
    static_assert(Rank > 0, "tile_shape: rank 0 has no tile");
    std::array<Index, Rank> t{};
    std::size_t             left = elems == 0 ? 1 : elems;
    for (std::size_t r = 0; r + 1 < Rank; ++r)
    {
        // The (Rank - r)-th root of what is left, rounded down.
        const std::size_t k    = Rank - r;
        std::size_t       root = 1;
        while (true)
        {
            std::size_t p = 1;
            for (std::size_t q = 0; q < k; ++q)
            {
                p *= root + 1;
            }
            if (p > left)
            {
                break;
            }
            ++root;
        }
        std::size_t d = root;
        while (left % d != 0)
        {
            --d;
        }
        t[r] = static_cast<Index>(d);
        left /= d;
    }
    t[Rank - 1] = static_cast<Index>(left);
    // The longer extents last (the greedy pass can leave 9 x 8 x 16 for 1152).
    for (std::size_t i = 1; i < Rank; ++i)
    {
        for (std::size_t j = i; j > 0 && t[j - 1] > t[j]; --j)
        {
            const Index x = t[j - 1];
            t[j - 1]      = t[j];
            t[j]          = x;
        }
    }
    return t;
}

/// @brief Elements a tiled view of extents @p ext needs with tile @p tile
///        (padding included): tiled_size() with the tile at run time.
template <std::size_t Rank, typename Index>
constexpr std::size_t tiled_size_dyn(const std::array<Index, Rank>& tile, const std::array<Index, Rank>& ext) noexcept
{
    std::size_t n = 1;
    for (std::size_t r = 0; r < Rank; ++r)
    {
        const auto t = static_cast<std::size_t>(tile[r]);
        n *= ((static_cast<std::size_t>(ext[r]) + t - 1) / t) * t;
    }
    return n;
}

/// @brief The view type of tiled_view_dyn().
template <typename Array, std::size_t Rank>
using tiled_dyn_view_type = view_type<Array, layout_tiled_dyn<Rank>, Rank>;

/**
 * @brief Tiled view of @p a with tile @p tile and extents @p ext...
 *        (layout_tiled_dyn); @p a must hold at least
 *        tiled_size_dyn(tile, {ext...}) elements.
 */
template <typename Array, typename I, std::size_t Rank, typename... Ext>
    requires(sizeof...(Ext) == Rank)
auto tiled_view_dyn(Array& a, const std::array<I, Rank>& tile, Ext... ext)
{
    using A   = std::remove_reference_t<Array>;
    using VT  = tiled_dyn_view_type<A, Rank>;
    using Idx = typename VT::index_type;
    typename VT::mapping_type::grid_type t{};
    for (std::size_t r = 0; r < Rank; ++r)
    {
        t[r] = static_cast<Idx>(tile[r]);
    }
    return VT(handle<A>(&a), typename VT::mapping_type(typename VT::extents_type(ext...), t), accessor<A>{});
}

/**
 * @brief Tiled view of @p a with a tile of a.chunk_size() elements
 *        (tile_shape()): each tile is one chunk, whatever the chunk size the
 *        array runs at. @p a must hold the tiled size: size it with
 *        tiled_size_dyn(tile_shape<Rank>(chunk_size), {ext...}).
 * @warning The tile is the chunk size when the view is made. After a
 *          re-chunk the view still reads and writes the right elements
 *          (nothing moves), but a tile is no longer a chunk: make the view
 *          again only if the data is laid out for the new tile (a copy).
 */
template <typename Array, typename... Ext>
auto tiled_view_dyn(Array& a, Ext... ext)
{
    using Idx = index_of<std::remove_reference_t<Array>>;
    return tiled_view_dyn(a, tile_shape<sizeof...(Ext), Idx>(static_cast<std::size_t>(a.chunk_size())), ext...);
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
