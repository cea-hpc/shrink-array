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
 * @file array_chunk_bits.h
 * @brief ArrayChunkBits: a chunked, compressed array of bools stored one bit
 *        each, 8 per byte.
 *
 * @details A container of its own, not a specialisation of a bool array:
 *          zero_one_value::Array<bool> stays what it is (one byte per bool), and the
 *          choice is made by type, array by array --
 *            - zero_one_value::Array<bool>: a plain byte store per write;
 *            - ArrayChunkBits:    8x less memory per materialised chunk, an
 *                                 atomic read-modify-write of a 64-bit word
 *                                 per write (OpenMP builds).
 *          Constant chunks (all false, all true) cost no data in either.
 *
 *          It is the chunk2 machinery over the element type Bit (bit.h):
 *          write protocol, compression, re-chunk and advisor are chunk2's, and
 *          Chunk2<Bit> has its own managers, pools and registries. The
 *          element is read as a Bit, which converts to and from bool:
 *          @code
 *            ArrayChunkBits<> a(n);
 *            a[i] = true;
 *            const bool b = a[i];
 *          @endcode
 *
 *          Unlike std::vector<bool>, two threads may write two different
 *          elements of one word at the same time: the write is an atomic
 *          read-modify-write of the word, so neither is lost. What is not
 *          there, as for any proxy container here: an element's address
 *          (operator()(i) returning a reference does not compile).
 */

#include "chunk/bit.h"          // Bit
#include "chunk/array_chunk2.h" // zero_one_value::Array

namespace shrink_array
{

/// @brief The bit-packed bool container (see the file comment).
template <typename IntType = int>
using ArrayChunkBits = zero_one_value::Array<Bit, IntType>;

} // namespace shrink_array
