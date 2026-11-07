/*
 * Copyright 2024-2026 CEA (www.cea.fr)
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
#include "chunk/internal/pointer_map_linear.h" // the class these bodies belong to

namespace shrink_array
{

// ---------------------------------------------------------------------------
// shrink_array::PointerMapLinear<Payload, Storage>
// ---------------------------------------------------------------------------

template <typename Payload, typename Storage>
void PointerMapLinear<Payload, Storage>::erase(Key key)
{
    for (auto it = vec_.begin(); it != vec_.end(); ++it)
    {
        if (it->first == key)
        {
            vec_.erase(it);
            return;
        }
    }
}

template <typename Payload, typename Storage>
bool PointerMapLinear<Payload, Storage>::swap_entries(Key key_a, Key key_b) noexcept
{
    Payload* pa = nullptr;
    Payload* pb = nullptr;
    for (auto& item : vec_)
    {
        if (item.first == key_a)
        {
            pa = &item.second;
        }
        else if (item.first == key_b)
        {
            pb = &item.second;
        }
    }
    if (pa == nullptr || pb == nullptr)
    {
        return false;
    }
    std::swap(*pa, *pb);
    return true;
}

template <typename Payload, typename Storage>
[[nodiscard]] const Payload* PointerMapLinear<Payload, Storage>::find_payload(Key key) const noexcept
{
    for (const auto& item : vec_)
    {
        if (item.first == key)
        {
            return &item.second;
        }
    }
    return nullptr;
}

} // namespace shrink_array
