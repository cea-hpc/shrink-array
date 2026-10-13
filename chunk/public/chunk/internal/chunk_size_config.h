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

// Compile-time chunk geometry: constexpr, so `i >> static_chunk_shift` and
// `i & static_chunk_mask` fold into immediate shifts/masks on the hot path.
//
// Override at build time:  cmake -DCHUNK_SIZE=128  (a power of two or a multiple of 64).

#include <bit> // std::has_single_bit

#include "chunk/internal/func_p2.h" // constexpr log2int

namespace shrink_array
{
#ifndef CHUNK_SIZE
    #define CHUNK_SIZE 64
#endif

inline constexpr unsigned int static_chunk_size = static_cast<unsigned int>(CHUNK_SIZE);
// log2(CHUNK_SIZE): the index shift of a power-of-two size, meaningless otherwise.
inline constexpr unsigned int static_chunk_shift = static_cast<unsigned int>(log2int(CHUNK_SIZE));
inline constexpr unsigned int static_chunk_mask  = static_cast<unsigned int>(CHUNK_SIZE) - 1u;

static_assert(CHUNK_SIZE >= 1, "CHUNK_SIZE must be positive");
static_assert(std::has_single_bit(static_chunk_size) || CHUNK_SIZE % 64 == 0,
              "CHUNK_SIZE must be a power of two or a multiple of 64");

} // namespace shrink_array
