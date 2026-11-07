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
#include "chunk/mdspan_tile_ops.h" // the class these bodies belong to

namespace shrink_array
{

// ---------------------------------------------------------------------------
// shrink_array::TileBuffer<View, T>
// ---------------------------------------------------------------------------

template <typename View, typename T>
md::detail::TileBuffer<View, T>::TileBuffer(const View& f)
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

template <typename View, typename T>
T* md::detail::TileBuffer<View, T>::data() noexcept
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

template <typename View, typename T>
std::size_t md::detail::TileBuffer<View, T>::size() const noexcept
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

// ---------------------------------------------------------------------------
// shrink_array::RangeWriter<Array>
// ---------------------------------------------------------------------------

template <typename Array>
template <typename T>
void md::detail::RangeWriter<Array>::fill(std::size_t lo, std::size_t hi, const T& v)
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

template <typename Array>
template <typename T>
void md::detail::RangeWriter<Array>::copy(std::size_t lo, std::size_t hi, const T* src)
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

// ---------------------------------------------------------------------------
// shrink_array::RunFiller<Array, T>
// ---------------------------------------------------------------------------

template <typename Array, typename T>
void md::detail::RunFiller<Array, T>::add(std::size_t a, std::size_t b)
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

// ---------------------------------------------------------------------------
// shrink_array::WholeBox
// ---------------------------------------------------------------------------

template <typename Index, std::size_t R>
std::pair<std::size_t, std::size_t>
md::detail::WholeBox::operator()(const std::array<Index, R>& lo, const std::array<Index, R>& hi, bool) const
{
    std::size_t W = 1;
    for (std::size_t r = 0; r < R; ++r)
    {
        W *= static_cast<std::size_t>(hi[r] - lo[r]);
    }
    return {0, W};
}

// ---------------------------------------------------------------------------
// shrink_array::TileCache<View, T>
// ---------------------------------------------------------------------------

template <typename View, typename T>
md::detail::TileCache<View, T>::Entry& md::detail::TileCache<View, T>::fetch(const std::array<Index, View::rank()>& c)
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

} // namespace shrink_array
