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
 * @details ArrayChunkBits is ArrayChunk2<Bit>: the whole chunk2 machinery
 *          (write protocol, compression, re-chunk, advisor) with its own
 *          managers, pools and registries, since Chunk2<Bit> is a different
 *          instantiation from Chunk2<bool>. ArrayChunk2<bool> is untouched:
 *          one byte per bool, a plain store per write. The user picks one or
 *          the other by type -- the packed one holds 8x less per
 *          materialised chunk, and pays an atomic read-modify-write of a
 *          64-bit word per write.
 *
 *          Bit is NOT addressable storage. A chunk of Bit holds
 *          bits::bytes_for(n) bytes of 64-bit words (bit_words.h), never an
 *          array of Bit objects: every access goes through element_access.h.
 *          The Bit value itself only travels by value (proxy reads, the
 *          constants, the value registry).
 *
 *          Bit converts both ways with bool, so `bool b = a[i]`, `a[i] = true`
 *          and `a[i] == b` read as they do on a bool container. It declares no
 *          operator== of its own: comparisons go through the built-in one on
 *          bool, which keeps `bit == true` unambiguous.
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
    constexpr Bit(bool b) noexcept : v(b) {} // NOLINT(google-explicit-constructor): a Bit IS a bool
    /// @brief Implicit conversion to bool: the stored value.
    constexpr operator bool() const noexcept // NOLINT(google-explicit-constructor)
    {
        return v;
    }

  private:
    bool v = false;
};

static_assert(sizeof(Bit) == 1 && std::is_trivially_copyable_v<Bit>);

/// @brief T is the packed element (storage is words of bits, not T objects).
template <typename T>
inline constexpr bool is_packed_bit_v = std::is_same_v<T, Bit>;

/// @brief Dynamic geometry: an array of packed bits is born with chunks this
///        many times the process-wide size, capped at cst_buffer_size.
/// @details At the size of the other arrays, a chunk of bits holds S/8 bytes
///          for a 16-byte header plus ~3 bytes of tables: at S=64, 8 bytes of
///          data for 19 of metadata, 2.5 times a std::vector<bool>. Eight
///          times the size gives a chunk the bytes of a bool chunk. The
///          re-chunk decider would not correct it: its deadband is absolute
///          (8 MB), and an array of bits rarely weighs that much -- shrink_bench
///          tab_mask, 3.1 M elements born at 64: 1.0 MB, 0.3 MB at 256, and
///          refused at every tick. Static geometry is one size for every array.
inline constexpr unsigned int packed_bit_birth_factor = 8;

/// @brief T takes chunk2's bool path: a single second value, `true`.
template <typename T>
inline constexpr bool is_bool_like_v = std::is_same_v<T, bool> || is_packed_bit_v<T>;

} // namespace shrink_array
