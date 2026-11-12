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
 * @file flat_address_set_inline.h
 * @brief The member bodies of fast_alloc::detail::FlatAddressSet.
 */

#include <bit> // std::countr_zero (the table's log2)

#include "fast_allocators/internal/flat_address_set.h" // FlatAddressSet: the class whose bodies these are

namespace fast_alloc::detail
{

inline std::size_t FlatAddressSet::find_slot(std::uintptr_t key) const noexcept
{
    std::size_t i = home(key);
    while (slots_[i] != 0 && slots_[i] != key)
    {
        i = (i + 1) & mask_;
    }
    return i;
}

inline bool FlatAddressSet::insert(std::uintptr_t key)
{
    // At most half full: short probe chains, and a free slot always exists.
    if ((size_ + 1) * 2 > slots_.size())
    {
        grow();
    }
    const std::size_t i = find_slot(key);
    if (slots_[i] == key)
    {
        return false;
    }
    slots_[i] = key;
    ++size_;
    return true;
}

inline bool FlatAddressSet::contains(std::uintptr_t key) const noexcept
{
    if (size_ == 0)
    {
        return false;
    }
    return slots_[find_slot(key)] == key;
}

inline std::size_t FlatAddressSet::erase(std::uintptr_t key) noexcept
{
    if (size_ == 0)
    {
        return 0;
    }
    std::size_t i = find_slot(key);
    if (slots_[i] != key)
    {
        return 0;
    }
    // Backward shift: pull later members of the chain into the hole when their
    // home is not in the cyclic range (i, j] -- a lookup from their home must
    // still reach them without crossing an empty slot.
    for (std::size_t j = i;;)
    {
        j = (j + 1) & mask_;
        if (slots_[j] == 0)
        {
            break;
        }
        const std::size_t h        = home(slots_[j]);
        const bool        in_range = i <= j ? (i < h && h <= j) : (i < h || h <= j);
        if (!in_range)
        {
            slots_[i] = slots_[j];
            i         = j;
        }
    }
    slots_[i] = 0;
    --size_;
    return 1;
}

inline void FlatAddressSet::grow()
{
    const std::size_t           new_size = slots_.empty() ? std::size_t{16} : slots_.size() * 2;
    std::vector<std::uintptr_t> old(new_size, 0);
    old.swap(slots_); // throws before any change if the allocation fails
    mask_  = new_size - 1;
    shift_ = 64U - static_cast<unsigned>(std::countr_zero(new_size));
    for (const std::uintptr_t key : old)
    {
        if (key != 0)
        {
            slots_[find_slot(key)] = key;
        }
    }
}

} // namespace fast_alloc::detail
