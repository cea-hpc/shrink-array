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
 * @file bit.h
 * @brief The element type of the bit-packed container: a bool stored as one
 *        bit, 8 per byte.
 *
 * @details ArrayChunkBits is zero_one_value::Array<Bit>: the whole zero_one_value machinery with its own
 *          managers, pools and registries. zero_one_value::Array<bool> keeps one byte per bool and a
 *          plain store per write; the packed one holds 8x less per materialised chunk and
 *          pays an atomic read-modify-write of a 64-bit word per write.
 *
 *          Bit is NOT addressable storage: a chunk of Bit holds bits::bytes_for(n) bytes of
 *          64-bit words (bit_words.h), never Bit objects; every access goes through
 *          element_access.h. A Bit only travels by value.
 *
 *          Bit converts both ways with bool. It declares no operator== of its own: the
 *          built-in one on bool keeps `bit == true` unambiguous.
 */

#include <type_traits> // std::is_same_v, std::is_trivially_copyable_v

namespace shrink_array
{

/// @brief A bool stored as one bit (the element type of ArrayChunkBits).
class Bit
{
  public:
    constexpr Bit() noexcept = default;
    /// @brief Implicit conversion from bool.
    constexpr Bit(bool value) noexcept : v_(value) {} // NOLINT(google-explicit-constructor): a Bit IS a bool
    /// @brief Implicit conversion to bool: the stored value.
    constexpr operator bool() const noexcept // NOLINT(google-explicit-constructor)
    {
        return v_;
    }

  private:
    bool v_ = false;
};

static_assert(sizeof(Bit) == 1 && std::is_trivially_copyable_v<Bit>);

/// @brief T is the packed element (storage is words of bits, not T objects).
template <typename T>
inline constexpr bool is_packed_bit_v = std::is_same_v<T, Bit>;

/// @brief Dynamic geometry: an array of packed bits is born with chunks this
///        many times the process-wide size, capped at cst_buffer_size.
/// @details At the other arrays' size a chunk of bits holds S/8 bytes for ~19 bytes of
///          metadata (at S=64, 2.5 times a std::vector<bool>); eight times the size gives it
///          the bytes of a bool chunk. The re-chunk decider will not correct it: its deadband
///          is absolute (8 MB), which an array of bits rarely weighs. Static geometry: one
///          size for every array.
inline constexpr unsigned int packed_bit_birth_factor = 8;

/// @brief T takes zero_one_value's bool path: a single second value, `true`.
template <typename T>
inline constexpr bool is_bool_like_v = std::is_same_v<T, bool> || is_packed_bit_v<T>;

} // namespace shrink_array
