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

#include "chunk/internal/concept_array_chunk_element.h" // ArrayChunkElement

namespace shrink_array
{

// ============================================================================
// 1. PRIMARY FORWARD DECLARATIONS (With default arguments)
// ============================================================================

// The three array classes, one namespace each: what the variant compresses.
//   zero             zeros only
//   zero_one_value   zero and one value declared for the whole array
//   zero_any         zero and, chunk by chunk, any value
namespace zero
{
template <ArrayChunkElement T, typename IntType = int>
class Array;
} // namespace zero
namespace zero_one_value
{
template <ArrayChunkElement T, typename IntType = int>
class Array;
} // namespace zero_one_value
namespace zero_any
{
template <ArrayChunkElement T, typename IntType = int>
class Array;
} // namespace zero_any

// The former names, kept until 1.0.
template <ArrayChunkElement T, typename IntType = int>
using ArrayChunk [[deprecated("use zero::Array")]] = zero::Array<T, IntType>;
template <ArrayChunkElement T, typename IntType = int>
using ArrayChunk2 [[deprecated("use zero_one_value::Array")]] = zero_one_value::Array<T, IntType>;
template <ArrayChunkElement T, typename IntType = int>
using ArrayChunk2A [[deprecated("use zero_any::Array")]] = zero_any::Array<T, IntType>;

template <typename T>
class Chunk;
template <typename T>
class Chunk2;
template <typename T>
class Chunk2A;

template <typename T>
class Proxy;
template <typename T>
class Proxy2;
template <typename T>
class Proxy2A;

template <template <typename> class ChunkT, typename T>
class ManagerChunk;

// ============================================================================
// 2. CHUNK TO PROXY / MANAGER MAPPING (ChunkTraits)
// ============================================================================
//
// No chunk -> array mapping: a chunk type has one array type per index type.
// What is common to all of them is ArrayChunkCore<ChunkType>
// (array_chunk_core.h), which is what ArrayChunkFor names.

template <typename C>
struct ChunkTraits;

/// @brief Chunk<T> maps to Proxy<T> and ManagerChunk<Chunk, T>.
template <typename T>
struct ChunkTraits<Chunk<T>>
{
    using ProxyType   = Proxy<T>;               ///< Write proxy returned by the chunk's non-const operator[].
    using ManagerType = ManagerChunk<Chunk, T>; ///< Per-type manager owning the compression machinery.
};

/// @brief Chunk2<T> maps to Proxy2<T> and ManagerChunk<Chunk2, T>.
template <typename T>
struct ChunkTraits<Chunk2<T>>
{
    using ProxyType   = Proxy2<T>;               ///< Write proxy returned by the chunk's non-const operator[].
    using ManagerType = ManagerChunk<Chunk2, T>; ///< Per-type manager owning the compression machinery.
};

/// @brief Chunk2A<T> maps to Proxy2A<T> and ManagerChunk<Chunk2A, T>.
template <typename T>
struct ChunkTraits<Chunk2A<T>>
{
    using ProxyType   = Proxy2A<T>;               ///< Write proxy returned by the chunk's non-const operator[].
    using ManagerType = ManagerChunk<Chunk2A, T>; ///< Per-type manager owning the compression machinery.
};

// ============================================================================
// 3. ARRAY TO CHUNK MAPPING (ArrayChunkTraits)
// ============================================================================

template <typename Derived>
struct ArrayChunkTraits;

/// @brief zero::Array<T, IntType> maps to Chunk<T>, Proxy<T>, ManagerChunk<Chunk, T>, value T and index IntType.
template <ArrayChunkElement T, typename IntType>
struct ArrayChunkTraits<zero::Array<T, IntType>>
{
    using ChunkType   = Chunk<T>;               ///< Chunk type the array holds.
    using ProxyType   = Proxy<T>;               ///< Write proxy returned by the non-const accessors.
    using ManagerType = ManagerChunk<Chunk, T>; ///< Per-type manager owning the compression machinery.
    using value_type  = T;                      ///< Stored element type.
    using IndexType   = IntType;                   ///< Signed global index type.
};

/// @brief zero_any::Array<T, IntType> maps to Chunk2A<T>, Proxy2A<T>, ManagerChunk<Chunk2A, T>, value T and index
/// IntType.
template <ArrayChunkElement T, typename IntType>
struct ArrayChunkTraits<zero_any::Array<T, IntType>>
{
    using ChunkType   = Chunk2A<T>;               ///< Chunk type the array holds.
    using ProxyType   = Proxy2A<T>;               ///< Write proxy returned by the non-const accessors.
    using ManagerType = ManagerChunk<Chunk2A, T>; ///< Per-type manager owning the compression machinery.
    using value_type  = T;                        ///< Stored element type.
    using IndexType   = IntType;                     ///< Signed global index type.
};

/// @brief zero_one_value::Array<T, IntType> maps to Chunk2<T>, Proxy2<T>, ManagerChunk<Chunk2, T>, value T and index IntType.
template <ArrayChunkElement T, typename IntType>
struct ArrayChunkTraits<zero_one_value::Array<T, IntType>>
{
    using ChunkType   = Chunk2<T>;               ///< Chunk type the array holds.
    using ProxyType   = Proxy2<T>;               ///< Write proxy returned by the non-const accessors.
    using ManagerType = ManagerChunk<Chunk2, T>; ///< Per-type manager owning the compression machinery.
    using value_type  = T;                       ///< Stored element type.
    using IndexType   = IntType;                    ///< Signed global index type.
};

} // namespace shrink_array