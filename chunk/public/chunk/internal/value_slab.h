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
 * @file value_slab.h
 * @brief The slab size of the value allocators: one rule for every allocator
 *        of an element type.
 */

#include <unistd.h>  // sysconf(_SC_PAGESIZE)
#include <algorithm> // std::max
#include <cstddef>   // std::size_t

namespace shrink_array
{

/**
 * @brief Slab request, in bytes, of the value allocators of element type @p T
 *        (TCSlab rounds it up to a power of two).
 * @param block_size ParamChunk's block size: the bytes of a slab of 8-byte
 *        elements.
 * @param max_element_bytes The largest buffer these allocators hand out:
 *        storage_units(cst_buffer_size) * sizeof(T).
 * @details @p block_size scaled to the element width (a 1-byte type takes an
 *          eighth: the same element count per slab), then floored
 *          - at one page: an emptied slab goes back to the system by whole
 *            pages (EmptySlabSweep), a slab smaller than a page never does;
 *          - at the largest buffer plus the slab's back-pointer: one always fits.
 *          Depends on @p T and @p block_size only, never on the chunk size:
 *          every allocator of one element type must cut its slabs at the same
 *          size, or slab_of() after a re-chunk reads a back-pointer from the
 *          middle of a slab (TCSlab::slab_bytes_for).
 */
template <typename T>
[[nodiscard]] inline std::size_t value_slab_request(int block_size, std::size_t max_element_bytes) noexcept
{
    constexpr int     per_word = std::max(static_cast<int>(8 / sizeof(T)), 1);
    static const long page     = ::sysconf(_SC_PAGESIZE);

    auto bytes = static_cast<std::size_t>(block_size / per_word);
    bytes      = std::max(bytes, page > 0 ? static_cast<std::size_t>(page) : std::size_t{4096});
    return std::max(bytes, max_element_bytes + sizeof(void*));
}

} // namespace shrink_array
