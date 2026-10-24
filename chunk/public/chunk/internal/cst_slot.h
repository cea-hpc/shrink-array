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
 * @file cst_slot.h
 * @brief Which of its two constants a zero_any chunk is compressed to.
 */

#include <cstdint> // std::uint8_t

namespace shrink_array
{

/**
 * @brief Which of a zero_any chunk's two constants it is compressed to: the
 *        one byte per chunk of zero_any::Proxy<T>::tab_type_cst_val, read when
 *        the chunk's values are pointed back at its constant.
 * @details Meaningful only while the chunk is constant. `first` is also the
 *          value a fresh row is filled with.
 */
enum class CstSlot : std::uint8_t
{
    first  = 0, ///< the first constant (cst_val1)
    second = 1, ///< the chunk's second constant (cst_val2)
};

/// @brief The slot of constant @p num_compress_value: 0 the first, any other the second.
[[nodiscard]] constexpr CstSlot cst_slot_of(int num_compress_value) noexcept
{
    return num_compress_value == 0 ? CstSlot::first : CstSlot::second;
}

} // namespace shrink_array
