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
#include "chunk/mdspan.h" // the class these bodies belong to

namespace shrink_array
{

// ---------------------------------------------------------------------------
// shrink_array::TiledMappingBase<Derived, Extents>
// ---------------------------------------------------------------------------

template <typename Derived, typename Extents>
constexpr md::detail::TiledMappingBase<Derived, Extents>::index_type
md::detail::TiledMappingBase<Derived, Extents>::required_span_size(this const Derived& self) noexcept
{
    if constexpr (Extents::rank() == 0)
    {
        return 1;
    }
    index_type last = 0; // the last tile, in storage order
    for (rank_type r = 0; r < Extents::rank(); ++r)
    {
        if (self.ntiles_[r] == 0)
        {
            return 0;
        }
        last += (self.ntiles_[r] - 1) * self.tstride_[r];
    }
    return (last + 1) * self.tile_elements();
}

template <typename Derived, typename Extents>
template <typename... Idx>
    requires(sizeof...(Idx) == Extents::rank())
constexpr md::detail::TiledMappingBase<Derived, Extents>::index_type
md::detail::TiledMappingBase<Derived, Extents>::operator()(this const Derived& self, Idx... idx) noexcept
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
        const auto t = static_cast<U>(self.tile_extent(r));
        which += (i[r] / t) * static_cast<U>(self.tstride_[r]);
        in = in * t + i[r] % t;
    }
    return static_cast<index_type>(which * static_cast<U>(self.tile_elements()) + in);
}

template <typename Derived, typename Extents>
constexpr bool md::detail::TiledMappingBase<Derived, Extents>::is_exhaustive(this const Derived& self) noexcept
{
    index_type s = 1; // a sub-view's tiles are spread over its parent's grid
    for (rank_type r = Extents::rank(); r-- > 0;)
    {
        if (self.ext_.extent(r) % self.tile_extent(r) != 0 || self.tstride_[r] != s)
        {
            return false;
        }
        s *= self.ntiles_[r];
    }
    return true;
}

template <typename Derived, typename Extents>
constexpr void md::detail::TiledMappingBase<Derived, Extents>::init(this Derived& self, const extents_type& e) noexcept
{
    self.ext_ = e;
    for (rank_type r = 0; r < Extents::rank(); ++r)
    {
        self.ntiles_[r] = (self.ext_.extent(r) + self.tile_extent(r) - 1) / self.tile_extent(r);
    }
    index_type s = 1;
    for (rank_type r = Extents::rank(); r-- > 0;)
    {
        self.tstride_[r] = s;
        s *= self.ntiles_[r];
    }
}

template <typename Derived, typename Extents>
template <typename... Slices>
constexpr std::pair<typename md::detail::TiledMappingBase<Derived, Extents>::grid_type, std::size_t>
md::detail::TiledMappingBase<Derived, Extents>::sub_view(this const Derived& self, const char* who, Slices... slices)
{
    static_assert(sizeof...(Slices) == Extents::rank(), "md::layout_tiled: one slice per dimension");
    constexpr std::size_t rank = Extents::rank();
    grid_type             first{};
    grid_type             size{};
    grid_type             stride{};
    [&]<std::size_t... K>(std::index_sequence<K...>)
    {
        (detail::tiled_slice<index_type>(slices, self.ext_.extent(K), first[K], size[K], stride[K]), ...);
    }(std::index_sequence_for<Slices...>{});

    index_type which = 0; // the sub-view's first tile, in storage order
    for (std::size_t r = 0; r < rank; ++r)
    {
        const index_type t   = self.tile_extent(r);
        const index_type end = first[r] + size[r];
        const bool       ok  = first[r] >= 0 && size[r] >= 0 && end <= self.ext_.extent(r) &&
                               (stride[r] == 1 || size[r] <= 1) && first[r] % t == 0 &&
                               (end % t == 0 || end == self.ext_.extent(r));
        if (!ok) [[unlikely]]
        {
            CHUNK_INVARIANT_FAILED("{}: submdspan slice {} = [{}, {}) (stride {}) of extent {}: a tiled sub-view "
                                   "takes ranges of stride 1 that start on a tile of {} and end on one or at the "
                                   "extent",
                                   who, r, static_cast<long long>(first[r]), static_cast<long long>(end),
                                   static_cast<long long>(stride[r]), static_cast<long long>(self.ext_.extent(r)),
                                   static_cast<long long>(t));
        }
        which += (first[r] / t) * self.tstride_[r];
    }
    return {size, static_cast<std::size_t>(which) * static_cast<std::size_t>(self.tile_elements())};
}

// ---------------------------------------------------------------------------
// shrink_array::layout_tiled_dyn<Rank>::mapping<Extents>
// ---------------------------------------------------------------------------

template <std::size_t Rank>
template <typename Extents>
constexpr md::layout_tiled_dyn<Rank>::mapping<Extents>::mapping(const extents_type& e, const grid_type& tile)
    : tile_(tile)
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

} // namespace shrink_array
