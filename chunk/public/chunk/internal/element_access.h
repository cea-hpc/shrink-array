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
 * @file element_access.h
 * @brief Element i of a chunk buffer, whatever its layout: `values[i]` for
 *        every element type, a bit of a 64-bit word for Bit (bit.h).
 *
 * @details The chunk2 code reads and writes elements through these rather
 *          than indexing the buffer, so that Chunk2<Bit> -- whose buffer is
 *          words of bits -- runs the same code. For every other T they are
 *          `values[i]`, inlined: no cost, no change.
 *
 *          Writes of Bit are an atomic read-modify-write of the word in an
 *          OpenMP build (bits::set_bit_atomic): two threads writing
 *          neighbouring elements write the same word. A serial build has no
 *          such neighbour and uses the plain one.
 */

#include <cstddef> // std::size_t

#include "chunk/bit.h"                // Bit, is_packed_bit_v
#include "chunk/config.h"             // build_config::use_openmp
#include "chunk/internal/bit_words.h" // bits::word_t, bits::get_bit, bits::set_bit_atomic

namespace shrink_array
{

namespace internal
{
/// @brief A packed buffer seen as its words.
template <typename T>
[[nodiscard]] inline const bits::WordT* as_words(const T* v) noexcept
{
    return reinterpret_cast<const bits::WordT*>(v);
}
/// @copydoc as_words(const T*)
template <typename T>
[[nodiscard]] inline bits::WordT* as_words(T* v) noexcept
{
    return reinterpret_cast<bits::WordT*>(v);
}
} // namespace internal

/// @brief Read element @p i of the buffer @p v.
template <typename T>
[[nodiscard]] inline T elem_get(const T* v, std::size_t i) noexcept
{
    if constexpr (is_packed_bit_v<T>)
    {
        if constexpr (build_config::use_openmp)
        {
            return T{bits::get_bit_relaxed(internal::as_words(v), i)};
        }
        else
        {
            return T{bits::get_bit(internal::as_words(v), i)};
        }
    }
    else
    {
        return v[i];
    }
}

/// @brief Write @p x at element @p i of the buffer @p v.
template <typename T>
inline void elem_set(T* v, std::size_t i, T x) noexcept
{
    if constexpr (is_packed_bit_v<T>)
    {
        if constexpr (build_config::use_openmp)
        {
            bits::set_bit_atomic(internal::as_words(v), i, static_cast<bool>(x));
        }
        else
        {
            bits::set_bit(internal::as_words(v), i, static_cast<bool>(x));
        }
    }
    else
    {
        v[i] = x;
    }
}

/**
 * @brief Apply @p op (a callable taking T&) to element @p i in place.
 * @details The element itself for every T; for Bit, a copy that is written
 *          back -- a bit has no address. Either way the update is one
 *          element's load-modify-store, not an atomic operation on it: two
 *          threads updating the SAME element race as on a plain T.
 */
template <typename T, typename Op>
inline void elem_update(T* v, std::size_t i, Op&& op)
{
    if constexpr (is_packed_bit_v<T>)
    {
        T tmp = elem_get(v, i);
        op(tmp);
        elem_set(v, i, tmp);
    }
    else
    {
        op(v[i]);
    }
}

} // namespace shrink_array
