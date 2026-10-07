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

#include <concepts>    // std::equality_comparable
#include <type_traits> // std::is_trivially_copyable_v, std::is_default_constructible_v

namespace shrink_array
{

/**
 * @brief What an element type of ArrayChunk / ArrayChunk2 / ArrayChunk2A must be.
 *
 * The contract, and why each clause:
 *  - **trivially copyable**: a chunk is raw memory -- its buffer is copied,
 *    handed to a pool, replaced by a shared constant buffer and taken back,
 *    without ever running a constructor or a destructor of T. Anything that
 *    owns a resource (std::string, a type with a destructor that matters) is
 *    out; that is the one thing a std::vector holds that these arrays do not.
 *  - **equality comparable**: compression is "every element == the constant",
 *    scanned block-wise with the type's own operator== (member-wise for a
 *    defaulted one, so padding bytes are not part of the value; never a
 *    memcmp).
 *  - **default constructible**: T{} is the first constant, the value a fresh
 *    array reads everywhere and the one a run compresses to.
 *
 * That admits every arithmetic type, bool, std::complex, enums, std::array of
 * those, and any aggregate of them with a defaulted operator== -- a struct
 * {double x, y, z} is an element type. The arithmetic and bitwise operators of
 * the proxies (+=, |=, ++ ...) are constrained separately, per operator, on
 * what the type supports; a struct gets assignment, reads and comparison.
 * chunk2 keys its second-value tables by value and asks a little more of a
 * floating-point aggregate (an operator<): see ValueLess in value_less.h.
 *
 * Checked at the array declarations (chunk_traits.h) and asserted where the
 * chunk is defined, so a type that does not qualify is refused at the first
 * use, with this text in the error.
 */
template <typename T>
concept ArrayChunkElement =
    std::is_trivially_copyable_v<T> && std::equality_comparable<T> && std::is_default_constructible_v<T>;

} // namespace shrink_array
