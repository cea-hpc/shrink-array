#pragma once

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
/// @details The 4th field is the queued chunk's element count, carried as the
///          ChunkElems strong type (chunk_size_policy.h) end to end -- the SAME
///          alias must be used by producers (push_to_compress_list) and
///          consumers (Chunk2A drain paths); this file is its single
///          definition.
template <template <typename> class TCHUNK, typename T>
using VectorTuple4ChunkInt2 = std::vector<std::tuple<TCHUNK<T>*, int, int, ChunkElems>>;

/// @brief Map from a second value to the shared list of queued chunks
///        (4-element tuples) holding it.
/// @details The 4th tuple element is the chunk's OWN geometry, captured at
///          enqueue by push_to_compress_list (as for Chunk v1 and Chunk2A, see
///          TChunkComp / TChunk2AComp below). compress() and
///          fill_list_to_compress_impl() scan with it, not with
///          alloc_chunk_count() (the CONSUMING thread's ambient size), which
///          after a rechunk overflows a buffer allocated at another geometry.
///          Keyed by the second value, ordered by ValueLess (value_less.h).
template <template <typename> class TCHUNK, typename T>
using MapTypeChunk2P = std::map<T, std::shared_ptr<VectorTuple4ChunkInt2<TCHUNK, T>>, ValueLess<T>>;

/// @brief Chunk2's candidate lists, one per second value.
/// @details 4-element tuples: this backing store is copied into the shared_ptr
///          held by map_chunk2_to_compress_potentially_thr (see
///          ArrayChunk2::set_second_value), so both must carry the chunk
///          geometry -- see MapTypeChunk2P above for why.
template <template <typename> class TCHUNK, typename T>
using TChunk2Comp = std::vector<VectorTuple4ChunkInt2<TCHUNK, T>>;

/// @brief Chunk2A's candidate list.
template <template <typename> class TCHUNK, typename T>
using TChunk2AComp = VectorTuple4ChunkInt2<TCHUNK, T>;

/// @brief Chunk's candidate list.
/// @details 4-element: the 4th is the chunk's OWN geometry, captured at enqueue
///          by push_to_compress_list. compress() scans with it, not with
///          alloc_chunk_count() (the CONSUMING thread's ambient size), which
///          overflows a buffer allocated at a different geometry under a live
///          rechunk.
template <template <typename> class TCHUNK, typename T>
using TChunkComp = VectorTuple4ChunkInt2<TCHUNK, T>;

/**
 * @brief One thread's slot in a per-thread candidate store, on cache lines of
 *        its own.
 *
 * @details The stores below are `std::vector<per-thread container>`, so the
 *          containers' HEADERS (begin / end / capacity, 24 bytes) sit side by
 *          side: three threads' vectors share one 64-byte line, and every
 *          push_back by one of them -- mark_for_compression_if_needed, on the
 *          write path -- invalidates the line the two others read on EVERY
 *          write, in trigger_compression_if_needed()'s size(). Measured on
 *          shrink_bench chunk2a_256 (aarch64, 64 threads): that size() was
 *          8.6% of write_with_optional_lock_impl<double>, on a `ldp` of the
 *          header. The same padding ManagerChunk::PerThreadFreeList already
 *          uses; 128 rather than 64 so adjacent-line prefetchers do not pair
 *          two slots either. It IS the container (inherits it), so every
 *          `store[thr]` use is unchanged.
 */
template <typename Container>
struct alignas(128) PerThreadSlot : public Container
{
    using Container::Container;
};

/// @brief Chunk2A's candidate lists, one slot per thread.
template <template <typename> class TCHUNK, typename T>
using TChunk2ACompThr = std::vector<PerThreadSlot<TChunk2AComp<TCHUNK, T>>>;

/// @brief Chunk2's candidate lists, one slot per thread.
template <template <typename> class TCHUNK, typename T>
using TChunk2CompThr = std::vector<PerThreadSlot<TChunk2Comp<TCHUNK, T>>>;

/// @brief Chunk's candidate lists, one slot per thread.
template <template <typename> class TCHUNK, typename T>
using TChunkCompThr = std::vector<PerThreadSlot<VectorTuple4ChunkInt2<TCHUNK, T>>>;

/// @brief Chunk2's candidate store: one map keyed by second value per thread.
/// @details Only the OpenMP fast-block code reaches it; a serial build keeps
///          it empty.
template <template <typename> class TCHUNK, typename T>
using TChunk2CompMap = std::vector<PerThreadSlot<MapTypeChunk2P<TCHUNK, T>>>;

} // namespace shrink_array