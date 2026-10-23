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
 * @brief What an element type of zero::Array / zero_one_value::Array / zero_any::Array must be.
 *
 * The contract, and why each clause:
 *  - **trivially copyable**: a chunk is raw memory -- its buffer is copied,
 *    pooled, replaced by a shared constant buffer and taken back without ever
 *    running a constructor or destructor of T. Resource owners (std::string)
 *    are out.
 *  - **equality comparable**: compression is "every element == the constant",
 *    scanned with the type's own operator== (padding is not part of the value;
 *    never a memcmp).
 *  - **default constructible**: T{} is the first constant, the value a fresh
 *    array reads everywhere and the one a run compresses to.
 *
 * That admits arithmetic types, bool, std::complex, enums, std::array of those,
 * and aggregates of them with a defaulted operator==. The proxies' arithmetic
 * and bitwise operators are constrained separately, per operator. zero_one_value also
 * needs an ordering (operator< for a floating-point aggregate): see ValueLess
 * in value_less.h.
 *
 * Checked at the array declarations (chunk_traits.h) and where the chunk is
 * defined, so a type that does not qualify is refused at first use.
 */
template <typename T>
concept ArrayChunkElement =
    std::is_trivially_copyable_v<T> && std::equality_comparable<T> && std::is_default_constructible_v<T>;

} // namespace shrink_array
