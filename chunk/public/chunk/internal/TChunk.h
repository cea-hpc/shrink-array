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

#include <vector>
#include <map>
#include <tuple>
#include <memory>
#include <type_traits> // std::conditional_t (TChunk2CompMap)

#include "chunk/internal/value_less.h"
#include "chunk/config.h"                     // build_config::use_openmp (TChunk2CompMap)
#include "chunk/internal/chunk_size_policy.h" // ChunkElems (VectorTuple4ChunkInt2)

namespace shrink_array
{
template <template <typename> class TCHUNK, typename T>
using VectorTuple3ChunkInt2 = std::vector<std::tuple<TCHUNK<T>*, int, int>>;

template <template <typename> class TCHUNK, typename T>
// Field <3> is the queued chunk's element count, carried as the ChunkElems
// strong type (chunk_size_policy.h) end to end — the SAME alias must be used
// by producers (push_to_compress_list) and consumers (Chunk2A drain paths);
// this file is its single definition.
using VectorTuple4ChunkInt2 = std::vector<std::tuple<TCHUNK<T>*, int, int, ChunkElems>>;

template <template <typename> class TCHUNK, typename T>
using VectorTuple4bChunkCharP = std::vector<std::tuple<TCHUNK<T>*, char*, char*, T*>>;

/// @brief Names the map from a second value to the shared list of queued chunks (4-element tuples) holding it.
template <template <typename> class TCHUNK, typename T>
struct ValueToChunk2PMapHelper
{
    // 4-element tuple (was 3): the 4th is the chunk's OWN geometry, captured at
    // enqueue by push_to_compress_list -- same fix as Chunk v1 and Chunk2A (see
    // TChunkComp / TChunk2AComp below). Chunk2's compress() and
    // fill_list_to_compress_impl() scanned with alloc_chunk_count() (the
    // CONSUMING thread's ambient size); after a rechunk that overflows a buffer
    // allocated at another geometry -- ASan heap-buffer-overflow reading 4 bytes
    // past a 256-byte / 64-int block, from finalize_parallel_phase_after_flush.
    // Keyed by the second value, ordered by ValueLess (value_less.h).
    using type = std::map<T, std::shared_ptr<VectorTuple4ChunkInt2<TCHUNK, T>>, ValueLess<T>>;
};

template <template <typename> class TCHUNK, typename T>
using ValueToChunk2Map_t = ValueToChunk2PMapHelper<TCHUNK, T>::type;

template <template <typename> class TCHUNK, typename T>
using MapTypeChunk2P = ValueToChunk2PMapHelper<TCHUNK, T>::type;

template <template <typename> class TCHUNK, typename T>
// 4-element tuple (was 3): this backing store is copied into the shared_ptr
// held by map_chunk2_to_compress_potentially_thr (see
// ArrayChunk2::set_second_value), so both must carry the chunk geometry --
// see ValueToChunk2PMapHelper above for why.
using TChunk2Comp = std::vector<VectorTuple4ChunkInt2<TCHUNK, T>>;

template <template <typename> class TCHUNK, typename T>
using TChunk2AComp = VectorTuple4ChunkInt2<TCHUNK, T>;

template <template <typename> class TCHUNK, typename T>
// 4-element: the 4th is the chunk's OWN geometry, captured at enqueue by
// push_to_compress_list. Chunk v1's compress() used to scan with
// alloc_chunk_count() (the CONSUMING thread's ambient size), which overflows a
// buffer allocated at a different geometry -- ASan heap-buffer-overflow at
// 512 bytes / 64 doubles under a live rechunk.
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
struct alignas(128) PerThreadSlot : Container
{
    using Container::Container;
};

template <template <typename> class TCHUNK, typename T>
using TChunk2ACompThr = std::vector<PerThreadSlot<TChunk2AComp<TCHUNK, T>>>;

template <template <typename> class TCHUNK, typename T>
using TChunk2CompThr = std::vector<PerThreadSlot<TChunk2Comp<TCHUNK, T>>>;

template <template <typename> class TCHUNK, typename T>
using TChunkCompThr = std::vector<PerThreadSlot<VectorTuple4ChunkInt2<TCHUNK, T>>>;

// Chunk2's candidate store: per-thread maps keyed by second value under
// OpenMP, one plain map serially. Both shapes are well-formed in both modes;
// only the selection depends on the build.
template <template <typename> class TCHUNK, typename T>
using TChunk2CompMap = std::conditional_t<build_config::use_openmp,
                                          std::vector<PerThreadSlot<MapTypeChunk2P<TCHUNK, T>>>,
                                          std::map<T, std::vector<TCHUNK<T>*>, ValueLess<T>>>;

} // namespace shrink_array