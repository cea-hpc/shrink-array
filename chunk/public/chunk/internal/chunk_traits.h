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

#include <type_traits> // std::false_type, std::true_type (ZeroAnyFamily)

#include "chunk/internal/concept_array_chunk_element.h" // ArrayChunkElement

namespace shrink_array
{

// ============================================================================
// 1. PRIMARY FORWARD DECLARATIONS (With default arguments)
// ============================================================================

// The array classes, one namespace each: what the variant compresses.
//   zero             zeros only
//   zero_one_value   zero and one value declared for the whole array
//   zero_any         zero and, chunk by chunk, any value
//   pattern          integers: zero, any value, and (being written) a shared motif
//                    plus an offset per chunk (docs/algo_pattern.md)
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
namespace pattern
{
template <ArrayChunkElement T, typename IntType = int>
class Array;
} // namespace pattern

namespace zero
{

template <typename T>
class Chunk;
template <typename T>
class Proxy;

} // namespace zero

namespace zero_one_value
{

template <typename T>
class Chunk;
template <typename T>
class Proxy;

} // namespace zero_one_value

namespace zero_any
{

template <typename T>
class Chunk;
template <typename T>
class Proxy;

} // namespace zero_any

namespace pattern
{

template <typename T>
class Chunk;
template <typename T>
class Proxy;

} // namespace pattern

template <template <typename> class ChunkT, typename T>
class TypeState;

// ============================================================================
// 2. CHUNK TO PROXY / MANAGER MAPPING (ChunkTraits)
// ============================================================================
//
// No chunk -> array mapping: a chunk type has one array type per index type.
// What is common to all of them is ArrayChunkCore<ChunkType>
// (array_chunk_core.h), which is what ArrayChunkFor names.

/**
 * @brief What a chunk type maps to, for one variant.
 * @tparam ChunkT the variant's chunk template
 * @tparam ProxyT the variant's proxy template
 * @tparam T stored element type
 */
template <template <typename> class ChunkT, template <typename> class ProxyT, typename T>
struct VariantChunkTraits
{
    using ProxyType   = ProxyT<T>;            ///< Write proxy returned by the non-const accessors.
    using ManagerType = TypeState<ChunkT, T>; ///< Per-type manager owning the compression machinery.
};

template <typename C>
struct ChunkTraits;

template <typename T>
struct ChunkTraits<zero::Chunk<T>> : VariantChunkTraits<zero::Chunk, zero::Proxy, T>
{
};

template <typename T>
struct ChunkTraits<zero_one_value::Chunk<T>> : VariantChunkTraits<zero_one_value::Chunk, zero_one_value::Proxy, T>
{
};

template <typename T>
struct ChunkTraits<zero_any::Chunk<T>> : VariantChunkTraits<zero_any::Chunk, zero_any::Proxy, T>
{
};

template <typename T>
struct ChunkTraits<pattern::Chunk<T>> : VariantChunkTraits<pattern::Chunk, pattern::Proxy, T>
{
};

// ============================================================================
// 3. ARRAY TO CHUNK MAPPING (ArrayChunkTraits)
// ============================================================================

/**
 * @brief What an array type maps to, for one variant.
 * @tparam ChunkT the variant's chunk template
 * @tparam ProxyT the variant's proxy template
 * @tparam T stored element type
 * @tparam IntType signed global index type
 */
template <template <typename> class ChunkT, template <typename> class ProxyT, typename T, typename IntType>
struct VariantArrayTraits : VariantChunkTraits<ChunkT, ProxyT, T>
{
    using ChunkType  = ChunkT<T>; ///< Chunk type the array holds.
    using value_type = T;         ///< Stored element type.
    using IndexType  = IntType;   ///< Signed global index type.
};

template <typename Derived>
struct ArrayChunkTraits;

template <ArrayChunkElement T, typename IntType>
struct ArrayChunkTraits<zero::Array<T, IntType>> : VariantArrayTraits<zero::Chunk, zero::Proxy, T, IntType>
{
};

template <ArrayChunkElement T, typename IntType>
struct ArrayChunkTraits<zero_any::Array<T, IntType>> : VariantArrayTraits<zero_any::Chunk, zero_any::Proxy, T, IntType>
{
};

template <ArrayChunkElement T, typename IntType>
struct ArrayChunkTraits<zero_one_value::Array<T, IntType>>
    : VariantArrayTraits<zero_one_value::Chunk, zero_one_value::Proxy, T, IntType>
{
};

template <ArrayChunkElement T, typename IntType>
struct ArrayChunkTraits<pattern::Array<T, IntType>> : VariantArrayTraits<pattern::Chunk, pattern::Proxy, T, IntType>
{
};

// ============================================================================
// 4. THE ZERO_ANY FAMILY
// ============================================================================
//
// zero_any compresses a chunk to zero or to a value of its own. The variants
// built on its code (pattern::Array, docs/algo_pattern.md) share its
// compression lists, its write engine, its relaxed element stores and its
// serial compression path, and differ in what a compressed chunk's values
// pointer names: zero_any's, a second value; pattern's, a motif to which the
// chunk adds its offset. The shared code tests the FAMILY with this trait,
// never the exact type -- a new member would otherwise fall into another
// variant's branch with no build error.

/// @brief Is @p ChunkT a chunk built on zero_any's machinery? Specialised for
///        each member of the family.
template <typename ChunkT>
struct ZeroAnyFamily : std::false_type
{
};

/// @brief zero_any::Chunk is the family's first member.
template <typename T>
struct ZeroAnyFamily<zero_any::Chunk<T>> : std::true_type
{
};

/// @brief pattern::Chunk, zero_any's chunk with shared motifs (docs/algo_pattern.md).
template <typename T>
struct ZeroAnyFamily<pattern::Chunk<T>> : std::true_type
{
};

/// @brief ZeroAnyFamily<ChunkT>::value.
template <typename ChunkT>
inline constexpr bool is_zero_any_family_v = ZeroAnyFamily<ChunkT>::value;

// ============================================================================
// 5. THE MOTIF STATE
// ============================================================================
//
// A member of the family whose chunk can also equal a shared motif plus an
// offset of its own (pattern::Chunk, docs/algo_pattern.md): its values
// pointer then names the motif, and the offset sits in the chunk's
// second-value slot. The shared code reads and materialises such a chunk in
// branches compiled for these members only: zero_any's code is unchanged.

/// @brief Has @p ChunkT the motif state? Specialised for each such member.
template <typename ChunkT>
struct HasMotifState : std::false_type
{
};

/// @brief pattern::Chunk: the motif state's first member.
template <typename T>
struct HasMotifState<pattern::Chunk<T>> : std::true_type
{
};

/// @brief HasMotifState<ChunkT>::value.
template <typename ChunkT>
inline constexpr bool has_motif_state_v = HasMotifState<ChunkT>::value;

} // namespace shrink_array