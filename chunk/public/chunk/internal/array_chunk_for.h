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
 * @brief The array type a proxy of @p ChunkT knows its array as.
 * @details The index-independent core of the array, never the array itself: a
 *          chunk type has one array type per index type (`zero_any::Array<T, int>`,
 *          `zero_any::Array<T, long>`), and they all share this core.
 */
template <typename ChunkT>
struct ArrayChunkFor
{
    using type = ArrayChunkCore<ChunkT>;
};

/**
 * @brief The identity of an array of @p ChunkT: the proxies' ptr_on_array_,
 *        the deferred contexts' ptr_on_array, every share_info_array key.
 * @details The address of the array's core (ArrayChunkCore::registry_key()),
 *          typed: a key reads the array back with no cast, and the derived
 *          array's `this` -- a different address in check builds
 *          (RegistryKeyCanary) -- or any other pointer does not convert to it.
 */
template <typename ChunkT>
using ArrayKey = const ArrayChunkCore<ChunkT>*;

template <typename ChunkT>
using ArrayChunkForT = ArrayChunkFor<ChunkT>::type;

} // namespace shrink_array