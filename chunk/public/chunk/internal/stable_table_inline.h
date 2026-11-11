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
#include "chunk/internal/stable_table.h" // the class these bodies belong to

namespace shrink_array
{

// ---------------------------------------------------------------------------
// shrink_array::StableTable<Row>
// ---------------------------------------------------------------------------

template <typename Row>
[[nodiscard]] Row& internal::StableTable<Row>::operator[](std::size_t index) noexcept
{
    if (index < first_size) [[likely]]
    {
        return segments_[0].load(std::memory_order_acquire)[index];
    }
    const std::size_t biased  = index + first_size;
    const std::size_t segment = segment_of(biased);
    return segments_[segment].load(std::memory_order_acquire)[biased - (first_size << segment)];
}

template <typename Row>
[[nodiscard]] const Row& internal::StableTable<Row>::operator[](std::size_t index) const noexcept
{
    if (index < first_size) [[likely]]
    {
        return segments_[0].load(std::memory_order_acquire)[index];
    }
    const std::size_t biased  = index + first_size;
    const std::size_t segment = segment_of(biased);
    return segments_[segment].load(std::memory_order_acquire)[biased - (first_size << segment)];
}

template <typename Row>
void internal::StableTable<Row>::resize(std::size_t count)
{
    if (count <= size_.load(std::memory_order_relaxed))
    {
        return;
    }
    const std::size_t last = segment_of(count - 1 + first_size);
    for (std::size_t segment = 0; segment <= last; ++segment)
    {
        if (segments_[segment].load(std::memory_order_relaxed) == nullptr)
        {
            // NOLINTNEXTLINE(cppcoreguidelines-owning-memory): owned by the table, which is never destroyed
            segments_[segment].store(new Row[first_size << segment](), std::memory_order_release);
        }
    }
    size_.store(count, std::memory_order_release);
}

} // namespace shrink_array
