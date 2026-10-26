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
 * @file pattern_chunk.h
 * @brief pattern::Chunk, the chunk of pattern::Array (docs/pattern_design.md).
 *
 * @details A member of the zero_any family (chunk_traits.h): its body is
 *          zero_any::ChunkImpl's, so it compresses to zero or to a second value
 *          exactly as zero_any::Chunk does. It adds the motif state: the values
 *          pointer names a shared motif (MotifRegion), and element j reads
 *          motif[j] + the chunk's offset. The offset lives in the chunk's
 *          second-value slot, free since a motif chunk has no second value: no
 *          byte more per chunk, one cache line more per read of a motif chunk
 *          (the slot row is another array). The shared code reads and
 *          materialises a motif chunk through its has_motif_state_v branches.
 *          Integers only.
 */

#include <type_traits> // std::is_integral_v, std::is_same_v, std::make_unsigned_t

#include "chunk/chunk_traits.h"                 // ChunkTraits<pattern::Chunk<T>>, HasMotifState
#include "chunk/internal/motif_region.h"        // pattern::MotifRegion
#include "chunk/internal/relaxed_element.h"     // load_element_relaxed (the offset)
#include "chunk/internal/tab_cst_val2_holder.h" // SecondValueTraits (the slot type)
#include "chunk/zero_any_chunk.h"               // zero_any::ChunkImpl

namespace shrink_array
{

namespace pattern
{

/**
 * @brief Chunk of pattern::Array: zero_any::ChunkImpl with itself as the chunk
 *        type, plus the motif state.
 * @tparam T Element type: an integer type other than bool.
 */
template <typename T>
class Chunk : public zero_any::ChunkImpl<T, Chunk<T>>
{
    static_assert(std::is_integral_v<T> && !std::is_same_v<T, bool>,
                  "pattern::Chunk holds integers only (docs/pattern_design.md, section 1)");
    static_assert(!SecondValueTraits<T>::indexed,
                  "pattern::Chunk keeps a motif chunk's offset in its second-value slot, which must hold a T: "
                  "build without CHUNK_SECOND_VALUE_INDEXED, or use an element type of at most 4 bytes");

  public:
    /// @brief The unsigned type of the same width: the motif arithmetic wraps on it.
    using UnsignedT = std::make_unsigned_t<T>;

    /// @brief Is @p ptr a motif (a values pointer inside MotifRegion<T>)?
    [[nodiscard]] static bool is_motif_ptr(const T* ptr) noexcept
    {
        return MotifRegion<T>::contains(ptr);
    }

    /// @brief @p lhs + @p rhs, wrapping (on UnsignedT: never undefined behaviour).
    [[nodiscard]] static T wrapping_add(T lhs, T rhs) noexcept
    {
        return static_cast<T>(static_cast<UnsignedT>(static_cast<UnsignedT>(lhs) + static_cast<UnsignedT>(rhs)));
    }

    /// @brief @p lhs - @p rhs, wrapping (on UnsignedT).
    [[nodiscard]] static T wrapping_sub(T lhs, T rhs) noexcept
    {
        return static_cast<T>(static_cast<UnsignedT>(static_cast<UnsignedT>(lhs) - static_cast<UnsignedT>(rhs)));
    }

    /**
     * @brief Element @p j of a chunk whose values pointer is @p motif.
     * @param motif       The chunk's values pointer, loaded by the caller (is_motif_ptr()).
     * @param j           Offset in the chunk.
     * @param offset_slot The chunk's second-value slot, which holds its offset.
     * @return motif[j] + offset, wrapping.
     */
    [[nodiscard]] static T motif_value(const T* motif, int j, const T* offset_slot) noexcept
    {
        return wrapping_add(motif[j], internal::load_element_relaxed(offset_slot));
    }
};

} // namespace pattern

} // namespace shrink_array
