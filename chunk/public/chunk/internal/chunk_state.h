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

/**
 * @file chunk_state.h
 * @brief The types of chunk_exchange.h -- ChunkKind, ChunkState, ArrayState -- with no
 *        dependency on the arrays, and ArrayImage, a state that owns its elements.
 *
 * @details chunk_exchange.h includes this header: its interface is unchanged. The device
 *          arrays (chunk/device/array.h) read and write these types too, in a CUDA or HIP
 *          translation unit that cannot include the arrays' headers (nvcc rejects them:
 *          docs/gpu_design.md, section 14). A program converts between the two in two
 *          translation units: export_array() / import_array() in a C++ one,
 *          device::Array::assign() / image() in the device one, the state passed between.
 */

#include <cstddef>     // std::size_t
#include <cstdint>     // std::uint8_t
#include <memory>      // std::unique_ptr (ArrayImage's elements)
#include <type_traits> // std::is_same_v, std::is_integral_v, std::make_unsigned_t
#include <vector>      // std::vector (ArrayState's chunks)

namespace shrink_array
{

/// @brief What a chunk holds.
enum class ChunkKind : std::uint8_t
{
    first_value,  ///< Compressed to the first value (ChunkState::value): no buffer.
    second_value, ///< Compressed to ChunkState::value: no buffer.
    motif,        ///< pattern: element k is data[k] + value (wrapping), data a shared motif.
    materialised, ///< A buffer of its own: ChunkState::data.
};

/// @brief One chunk, as export_chunk() reads it and import_chunk() writes it.
template <typename T>
struct ChunkState
{
    ChunkKind kind  = ChunkKind::first_value; ///< What the chunk holds.
    T         value = T{};                    ///< The constant, or a motif chunk's offset.
    /// materialised: the chunk's elements; motif: the motif (see chunk_exchange.h). Valid
    /// until the array is written again; nullptr otherwise.
    const T* data = nullptr;
};

/// @brief A whole array, as export_array() reads it and import_array() writes it.
template <typename T>
struct ArrayState
{
    std::size_t                size             = 0;     ///< Element count.
    unsigned int               chunk_size       = 0;     ///< S, the elements per chunk.
    bool                       has_second_value = false; ///< zero_or_value (not bool): a second value declared.
    T                          second_value     = T{};   ///< That second value.
    std::vector<ChunkState<T>> chunks;                   ///< One per chunk, in order.
};

/// @brief An ArrayState whose materialised chunks point into @p elements, which it owns
///        (device::Array::image()): moving it keeps the pointers valid.
template <typename T>
struct ArrayImage
{
    std::unique_ptr<T[]> elements; ///< chunk_size elements per materialised chunk
    ArrayState<T>        state;
};

namespace detail
{

/// @brief A motif chunk's element: @p motif + @p offset, wrapping on an
///        integer (pattern::Chunk::wrapping_add's arithmetic).
template <typename T>
[[nodiscard]] T motif_element(T motif, T offset) noexcept
{
    if constexpr (std::is_integral_v<T> && !std::is_same_v<T, bool>)
    {
        using U = std::make_unsigned_t<T>;
        return static_cast<T>(static_cast<U>(static_cast<U>(motif) + static_cast<U>(offset)));
    }
    else
    {
        return static_cast<T>(motif + offset);
    }
}

} // namespace detail

} // namespace shrink_array
