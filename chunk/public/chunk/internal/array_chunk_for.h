
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

namespace shrink_array
{

// Forward declarations of chunk types
template <typename T>
class Chunk;
template <typename T>
class Chunk2;
template <typename T>
class Chunk2A;

template <typename ChunkType>
class ArrayChunkCore;

/**
 * @brief What a proxy of @p ChunkType may cast its type-erased ptr_on_array to.
 * @details The index-independent core of the array, never the array itself: a
 *          chunk type has one array type per index type (`ArrayChunk2A<T, int>`,
 *          `ArrayChunk2A<T, long>`), and ptr_on_array is the address of the core
 *          (ArrayChunkCore::registry_key()) for all of them.
 */
template <typename ChunkType>
class ArrayChunkFor
{
  public:
    using type = ArrayChunkCore<ChunkType>;
};

template <typename ChunkType>
using ArrayChunkFor_t = ArrayChunkFor<ChunkType>::type;

} // namespace shrink_array