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

#include <type_traits> // std::is_same_v

namespace shrink_array
{

namespace zero
{

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

/// @brief ChunkT<T> is Chunk<T>.
template <template <typename> typename ChunkT, typename T>
concept IsChunk = std::is_same_v<ChunkT<T>, zero::Chunk<T>>;

/// @brief ChunkT<T> is zero_any::Chunk<T>.
template <template <typename> typename ChunkT, typename T>
concept IsChunk2A = std::is_same_v<ChunkT<T>, zero_any::Chunk<T>>;

/// @brief ChunkT<T> is zero_one_value::Chunk<T>.
template <template <typename> typename ChunkT, typename T>
concept IsChunk2 = std::is_same_v<ChunkT<T>, zero_one_value::Chunk<T>>;

/// @brief ChunkT<T> is zero_one_value::Chunk<T> or zero_any::Chunk<T>.
template <template <typename> typename ChunkT, typename T>
concept IsChunk2Or2A = IsChunk2<ChunkT, T> || IsChunk2A<ChunkT, T>;

/// @brief The chunk template of a chunk type (specialised below).
template <typename>
struct ChunkTemplateOf;

/// @brief For ChunkTemplate\<T\>, type\<U\> names ChunkTemplate\<U\>.
template <template <typename> typename ChunkTemplate, typename T>
struct ChunkTemplateOf<ChunkTemplate<T>>
{
    template <typename U>
    using type = ChunkTemplate<U>;
};

} // namespace shrink_array