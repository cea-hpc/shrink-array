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

#include <vector> // std::vector
#include <map>    // std::map
#include <tuple>  // std::tuple
#include <memory> // std::shared_ptr

#include "chunk/internal/value_less.h"        // ValueLess
#include "chunk/internal/chunk_size_policy.h" // ChunkElems (VectorTuple4ChunkInt2)

namespace shrink_array
{
/// @brief A list of queued compression candidates: (chunk, num_chunk,
///        num_tab, the chunk's element count).
/// @details The 4th field is the chunk's OWN element count (ChunkElems),
///          captured at enqueue by push_to_compress_list. Consumers (compress(),
///          fill_list_to_compress_impl(), the drain paths) scan with it, never
///          with alloc_chunk_count() -- the consuming thread's ambient size,
///          which after a rechunk overflows a buffer of another geometry.
///          Producers and consumers must share this single alias.
template <template <typename> class ChunkTemplate, typename T>
using VectorTuple4ChunkInt2 = std::vector<std::tuple<ChunkTemplate<T>*, int, int, ChunkElems>>;

/// @brief Map from a second value to the shared list of queued chunks
///        (4-element tuples) holding it.
/// @details Tuples as VectorTuple4ChunkInt2. Ordered by ValueLess (value_less.h).
template <template <typename> class ChunkTemplate, typename T>
using MapTypeChunk2P = std::map<T, std::shared_ptr<VectorTuple4ChunkInt2<ChunkTemplate, T>>, ValueLess<T>>;

/// @brief Chunk2's candidate lists, one per second value.
/// @details Copied into map_chunk2_to_compress_potentially_thr's shared_ptr
///          (zero_one_value::Array::set_second_value): same tuples as MapTypeChunk2P.
template <template <typename> class ChunkTemplate, typename T>
using TChunk2Comp = std::vector<VectorTuple4ChunkInt2<ChunkTemplate, T>>;

/// @brief Chunk2A's candidate list.
template <template <typename> class ChunkTemplate, typename T>
using TChunk2AComp = VectorTuple4ChunkInt2<ChunkTemplate, T>;

/// @brief Chunk's candidate list (tuples: see VectorTuple4ChunkInt2).
template <template <typename> class ChunkTemplate, typename T>
using TChunkComp = VectorTuple4ChunkInt2<ChunkTemplate, T>;

/**
 * @brief One thread's slot in a per-thread candidate store, on cache lines of
 *        its own.
 *
 * @details Without padding, the containers' 24-byte headers of three threads
 *          share a cache line: every push_back on the write path invalidates
 *          the line the others read on every write
 *          (trigger_compression_if_needed()'s size()), a measurable share of
 *          the write path. 128, not 64, so adjacent-line prefetchers do not
 *          pair two slots (as ManagerChunk::PerThreadFreeList). It inherits the
 *          container, so `store[thr]` is used unchanged.
 */
template <typename Container>
struct alignas(128) PerThreadSlot : public Container
{
    using Container::Container;
};

/// @brief Chunk2A's candidate lists, one slot per thread.
template <template <typename> class ChunkTemplate, typename T>
using TChunk2ACompThr = std::vector<PerThreadSlot<TChunk2AComp<ChunkTemplate, T>>>;

/// @brief Chunk2's candidate lists, one slot per thread.
template <template <typename> class ChunkTemplate, typename T>
using TChunk2CompThr = std::vector<PerThreadSlot<TChunk2Comp<ChunkTemplate, T>>>;

/// @brief Chunk's candidate lists, one slot per thread.
template <template <typename> class ChunkTemplate, typename T>
using TChunkCompThr = std::vector<PerThreadSlot<VectorTuple4ChunkInt2<ChunkTemplate, T>>>;

/// @brief Chunk2's candidate store: one map keyed by second value per thread.
/// @details Only the OpenMP fast-block code reaches it; a serial build keeps
///          it empty.
template <template <typename> class ChunkTemplate, typename T>
using TChunk2CompMap = std::vector<PerThreadSlot<MapTypeChunk2P<ChunkTemplate, T>>>;

} // namespace shrink_array