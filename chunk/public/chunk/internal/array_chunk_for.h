
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

#include "chunk/internal/concept_array_chunk_element.h"

namespace shrink_array
{

// Forward declarations of chunk types
template <typename T>
class Chunk;
template <typename T>
class Chunk2;
template <typename T>
class Chunk2A;

// Forward declarations of array chunk types
template <ArrayChunkElement, typename ind_type>
class ArrayChunk;
template <ArrayChunkElement, typename ind_type>
class ArrayChunk2;
template <ArrayChunkElement, typename ind_type>
class ArrayChunk2A;

template <typename ChunkType>
struct ArrayChunkFor; // primary template

template <typename T>
struct ArrayChunkFor<Chunk<T>>
{
    using type = ArrayChunk<T, int>;
};

template <typename T>
struct ArrayChunkFor<Chunk2<T>>
{
    using type = ArrayChunk2<T, int>;
};

template <typename T>
struct ArrayChunkFor<Chunk2A<T>>
{
    using type = ArrayChunk2A<T, int>;
};

template <typename ChunkType>
using ArrayChunkFor_t = ArrayChunkFor<ChunkType>::type;

} // namespace shrink_array