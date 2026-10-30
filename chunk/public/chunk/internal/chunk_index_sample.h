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

#include <array>       // std::array
#include <cstddef>     // std::size_t
#include <format>      // std::format_to_n
#include <string_view> // std::string_view
#include <vector>      // std::vector

namespace shrink_array::internal
{
/**
 * @brief "j@e j@e ... (+N more)": the first @p max chunk indices of @p idx,
 *        each with the index of its first element (j * @p chunk_size),
 *        written into @p buf -- the sample a compression-residual message
 *        shows.
 * @return The text, a view of @p buf (cut at its size).
 */
template <std::size_t N, typename I>
[[nodiscard]] std::string_view
chunk_index_sample(std::array<char, N>& buf, const std::vector<int>& idx, I chunk_size, std::size_t max = 10) noexcept
{
    std::size_t       count = 0;
    const std::size_t shown = idx.size() < max ? idx.size() : max;
    for (std::size_t k = 0; k < shown && count < N; ++k)
    {
        const auto num_chunk = idx[k];
        count += static_cast<std::size_t>(std::format_to_n(buf.data() + count, N - count, "{}{}@{}", k == 0 ? "" : " ",
                                                           num_chunk, static_cast<I>(num_chunk) * chunk_size)
                                              .size);
    }
    if (idx.size() > shown && count < N)
    {
        count += static_cast<std::size_t>(
            std::format_to_n(buf.data() + count, N - count, " (+{} more)", idx.size() - shown).size);
    }
    return {buf.data(), count < N ? count : N};
}
} // namespace shrink_array::internal
