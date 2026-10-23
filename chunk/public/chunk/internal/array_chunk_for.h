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

namespace zero
{

// Forward declarations of chunk types
template <typename T>
class Chunk;

} // namespace zero

namespace zero_one_value
{

template <typename T>
class Chunk;

} // namespace zero_one_value

namespace zero_any
{

template <typename T>
class Chunk;

} // namespace zero_any

template <typename ChunkT>
class ArrayChunkCore;

/**
 * @brief What a proxy of @p ChunkT may cast its type-erased ptr_on_array to.
 * @details The index-independent core of the array, never the array itself: a
 *          chunk type has one array type per index type (`zero_any::Array<T, int>`,
 *          `zero_any::Array<T, long>`), and ptr_on_array is the address of the core
 *          (ArrayChunkCore::registry_key()) for all of them.
 */
template <typename ChunkT>
struct ArrayChunkFor
{
    using type = ArrayChunkCore<ChunkT>;
};

template <typename ChunkT>
using ArrayChunkForT = ArrayChunkFor<ChunkT>::type;

} // namespace shrink_array