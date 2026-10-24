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
 * @file raw_memory.h
 * @brief An address as an integer, and a scalar kept in raw bytes: the two
 *        uses of reinterpret_cast an allocator cannot avoid, spelled once.
 *
 * - address_bits(): masks, alignment tests, hashing and sorting by address.
 *   std::bit_cast of the pointer -- the same integer as reinterpret_cast,
 *   defined for any pointer. Go back to a pointer by stepping from a pointer
 *   you have (`base - (address_bits(p) & mask)`), not from the integer: the
 *   pointer keeps its provenance.
 * - load_from_bytes() / store_to_bytes(): a trivially copyable value (a slab's
 *   back-pointer, a free-list link, a size header) stored at an address no
 *   object of that type was created at. std::memcpy is defined there whatever
 *   lived in the bytes before; `*reinterpret_cast<T*>(p)` is not. Compilers
 *   emit the same single load or store.
 *
 * Standard library only: chunk's std-alloc build installs it (chunk/CMakeLists.txt).
 */

#include <bit>         // std::bit_cast
#include <cstdint>     // std::uintptr_t
#include <cstring>     // std::memcpy
#include <type_traits> // std::is_trivially_copyable_v

namespace fast_alloc
{

/// @brief The address of @p p as an integer (see the file comment).
[[nodiscard]] inline std::uintptr_t address_bits(const void* p) noexcept
{
    return std::bit_cast<std::uintptr_t>(p);
}

/// @brief The T stored in the bytes at @p p (see the file comment).
template <typename T>
    requires std::is_trivially_copyable_v<T>
[[nodiscard]] inline T load_from_bytes(const void* p) noexcept
{
    T value;
    std::memcpy(&value, p, sizeof(T));
    return value;
}

/// @brief Store @p value in the bytes at @p p (see the file comment).
template <typename T>
    requires std::is_trivially_copyable_v<T>
inline void store_to_bytes(void* p, const T& value) noexcept
{
    std::memcpy(p, &value, sizeof(T));
}

} // namespace fast_alloc
