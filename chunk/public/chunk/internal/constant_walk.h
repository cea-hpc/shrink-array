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
 * @file constant_walk.h
 * @brief The uniformity test of a chunk's buffer -- "do elements [from, count) all hold
 *        this value?" -- as a free function of the core layer (host_device.h): the CPU's
 *        compression scans and the GPU's compress() pass share it.
 */

#include "chunk/internal/host_device.h" // CHUNK_HD
#include "chunk/internal/same_value.h"  // same_value

namespace shrink_array::internal
{

/// @brief Elements per branch-free block of the constant walk: an exit point between
///        blocks, a block long enough for the vector loop to matter (a quarter of a
///        256-element chunk).
inline constexpr unsigned int constant_walk_block = 64;

/// @brief Where walk_constant() stopped, and whether the run held.
struct ConstantWalk
{
    bool         holds = true; ///< elements [from, count) all equal the value
    unsigned int end   = 0;    ///< the walk's end: count, or the end of the block that disproved it
};

/**
 * @brief Whether elements [@p from, @p count) of @p vals all hold @p value (same_value()):
 *        branch-free blocks of constant_walk_block, an exit between them.
 * @param vals  a buffer of at least @p count elements (not a packed Bit buffer).
 * @param value the value of the run.
 * @param from  the first element to compare.
 * @param count one past the last.
 */
template <typename T>
[[nodiscard]] CHUNK_HD ConstantWalk
walk_constant(const T* vals, const T& value, unsigned int from, unsigned int count) noexcept
{
    unsigned int i = from;
    while (i < count)
    {
        const unsigned int block_end = count - i < constant_walk_block ? count : i + constant_walk_block;
        unsigned int       bad       = 0;
        for (unsigned int j = i; j < block_end; ++j)
        {
            // raw-read: the caller owns the buffer (a pass that claimed the chunk, or a sequential caller)
            bad |= static_cast<unsigned int>(!same_value(vals[j], value));
        }
        i = block_end;
        if (bad != 0)
        {
            return ConstantWalk{false, i};
        }
    }
    return ConstantWalk{true, count};
}

} // namespace shrink_array::internal
