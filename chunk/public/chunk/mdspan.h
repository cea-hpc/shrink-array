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
 * @warning Not a pointer: data_handle() is the array's address, not a T*,
 *          and the accessor has no offset() (no submdspan). Code that wants a
 *          contiguous T* still has to copy. The rules of the array apply
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
#else
    #define SHRINK_ARRAY_HAS_MDSPAN 0
#endif

#if SHRINK_ARRAY_HAS_MDSPAN

    #include <array>       // std::array (the tile extents)
    #include <cstddef>     // std::size_t
    #include <type_traits> // std::remove_const_t, std::is_const_v, std::conditional_t, std::make_unsigned_t
    #include <utility>     // std::declval

namespace shrink_array::md
{

namespace mdns = SHRINK_ARRAY_MDSPAN_NS;

/// @brief The index type of @p Array: the type of a.size() (as array_chunk_range_ops.h).
template <typename Array>
using index_of = decltype(std::declval<const std::remove_const_t<Array>&>().size());

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
    using data_handle_type = Array*;
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

    /// @brief Element @p i of the array: its operator[], the proxy or the value.
    reference access(data_handle_type a, std::size_t i) const noexcept
    {
        return (*a)[static_cast<index_of<Array>>(i)];
    }
};

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
        }

        constexpr const extents_type& extents() const noexcept
        {
            return ext_;
        }

        /// @brief All the tiles, padding included: the array size the view needs.
        constexpr index_type required_span_size() const noexcept
        {
            index_type n = tile_elems;
            for (rank_type r = 0; r < Extents::rank(); ++r)
            {
                n *= ntiles_[r];
            }
            return Extents::rank() == 0 ? 1 : n;
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
                which        = which * static_cast<U>(ntiles_[r]) + i[r] / t;
                in           = in * t + i[r] % t;
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
            for (rank_type r = 0; r < Extents::rank(); ++r)
            {
                if (ext_.extent(r) % tile[r] != 0)
                {
                    return false;
                }
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
            return a.extents() == b.extents();
        }

      private:
        extents_type                            ext_{};
        std::array<index_type, sizeof...(Tile)> ntiles_{};
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
    return VT(&a, typename VT::mapping_type(typename VT::extents_type(ext...)), accessor<A>{});
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
    return VT(&a, typename VT::mapping_type(typename VT::extents_type(ext...)), accessor<A>{});
}

} // namespace shrink_array::md

#endif // SHRINK_ARRAY_HAS_MDSPAN
