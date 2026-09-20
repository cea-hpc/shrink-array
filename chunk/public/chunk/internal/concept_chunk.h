
#pragma once

#include <type_traits> // std::is_same_v

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


namespace shrink_array
{

template <typename T>
class Chunk;

template <typename T>
class Chunk2;

template <typename T>
class Chunk2A;

template <template <typename> typename CHUNK_T, typename T>
concept IsChunk = std::is_same_v<CHUNK_T<T>, Chunk<T>>;

template <template <typename> typename CHUNK_T, typename T>
concept IsChunk2A = std::is_same_v<CHUNK_T<T>, Chunk2A<T>>;

template <template <typename> typename CHUNK_T, typename T>
concept IsChunk2 = std::is_same_v<CHUNK_T<T>, Chunk2<T>>;

template <template <typename> typename CHUNK_T, typename T>
concept IsChunkOrChunk2 = IsChunk<CHUNK_T, T> || IsChunk2<CHUNK_T, T>;

template <template <typename> typename CHUNK_T, typename T>
concept IsChunk2Or2A = IsChunk2<CHUNK_T, T> || IsChunk2A<CHUNK_T, T>;

template <typename>
struct chunk_template_of;

template <template <typename> typename CH, typename T>
struct chunk_template_of<CH<T>>
{
    template <typename U>
    using type = CH<U>;
};

} // namespace shrink_array