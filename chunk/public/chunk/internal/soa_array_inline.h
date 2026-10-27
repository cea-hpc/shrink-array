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
#include "chunk/soa_array.h" // the class these bodies belong to

namespace shrink_array
{

// ---------------------------------------------------------------------------
// shrink_array::SoaArray<T, Column>
// ---------------------------------------------------------------------------

template <typename T, template <typename> class Column>
template <std::size_t N>
consteval std::size_t SoaArray<T, Column>::index_of(const std::array<std::meta::info, N>& path)
{
    for (std::size_t k = 0; k < n_columns_; ++k)
    {
        bool same = leaves_[k].count == N;
        for (std::size_t d = 0; same && d < N; ++d)
        {
            const soa_detail::Step s = steps_[leaves_[k].first + d];
            same                     = s.index < 0 && s.member == path[d];
        }
        if (same)
        {
            return k;
        }
    }
    return n_columns_;
}

template <typename T, template <typename> class Column>
template <std::size_t K, std::size_t D, typename V>
constexpr decltype(auto) SoaArray<T, Column>::field(V& v) noexcept
{
    if constexpr (D == leaves_[K].count)
    {
        return (v);
    }
    else
    {
        constexpr soa_detail::Step s = steps_[leaves_[K].first + D];
        if constexpr (s.index < 0)
        {
            return field<K, D + 1>(v.[:s.member:]);
        }
        else
        {
            return field<K, D + 1>(v[s.index]);
        }
    }
}

template <typename T, template <typename> class Column>
[[nodiscard]] T SoaArray<T, Column>::get(size_type i) const
{
    T v{};
    template for (constexpr std::size_t k : std::views::iota(0uz, n_columns_))
    {
        field<k>(v) = std::as_const(std::get<k>(columns_))[i];
    }
    return v;
}

} // namespace shrink_array
