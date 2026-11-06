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
#include <string_view> // std::string_view
#include <vector>      // std::vector

#include "fast_allocators/internal/append_text.h" // append_text, append_int

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
    // Without std::format: this runs in noexcept code (append_text.h).
    using fast_alloc::append_int;
    using fast_alloc::append_text;
    char* const       first = buf.data();
    char* const       end   = first + N;
    char*             out   = first;
    const std::size_t shown = idx.size() < max ? idx.size() : max;
    for (std::size_t k = 0; k < shown && out < end; ++k)
    {
        const auto num_chunk = idx[k];
        out                  = append_text(out, end, k == 0 ? "" : " ");
        out                  = append_int(out, end, num_chunk);
        out                  = append_text(out, end, "@");
        out                  = append_int(out, end, static_cast<I>(num_chunk) * chunk_size);
    }
    if (idx.size() > shown)
    {
        out = append_text(out, end, " (+");
        out = append_int(out, end, idx.size() - shown);
        out = append_text(out, end, " more)");
    }
    return {first, static_cast<std::size_t>(out - first)};
}
} // namespace shrink_array::internal
