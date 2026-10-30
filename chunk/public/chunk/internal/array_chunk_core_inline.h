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
#include "chunk/internal/array_chunk_core.h" // the class these bodies belong to

namespace shrink_array
{

// ---------------------------------------------------------------------------
// shrink_array::ArrayChunkCore<ChunkT>
// ---------------------------------------------------------------------------

template <typename ChunkT>
void ArrayChunkCore<ChunkT>::assign_alloc_key()
{
    if constexpr (build_config::alloc_affinity_min_bytes == 0 || !build_config::chunk_dynamic ||
                  !build_config::use_fast_alloc || !build_config::threaded)
    {
        return;
    }
    if (nb_chunk_ <= 0)
    {
        return;
    }
    // Lazy affinity: only the storage a re-chunk builds takes its own set (an array
    // already keyed keeps its key). A set costs one partly-filled slab per (rank, size),
    // for a benefit -- whole slabs emptied by the NEXT re-chunk -- that only a
    // re-chunked array collects.
    if (alloc_key_ == 0 && !alloc_affinity_detail::g_eager && !alloc_affinity_detail::g_building_rechunk_storage)
    {
        return;
    }
    const std::size_t meta_bytes = static_cast<std::size_t>(nb_chunk_) * sizeof(ChunkType);
    if (meta_bytes < build_config::alloc_affinity_min_bytes)
    {
        return;
    }
    const unsigned int key = (alloc_key_ != 0) ? alloc_key_ : next_alloc_key_.fetch_add(1, std::memory_order_relaxed);
    // Keep the key ONLY if the manager built its allocator set: a key with no registered
    // set makes route_omp_dynamic_alloc fall through to the legacy single allocator,
    // which is neither this size's nor this array's.
    alloc_key_ = ManagerType::register_alloc_key(key, size_policy_.size(), nb_chunk_) ? key : 0u;
}

template <typename ChunkT>
void ArrayChunkCore<ChunkT>::acquire_chunk_list(std::size_t count)
{
    // Only reserves (NoInitAllocator): the caller's init loop (static schedule) is the
    // first touch, page faults and NUMA placement included.
    chunk_list_.resize(count);
}

template <typename ChunkT>
void ArrayChunkCore<ChunkT>::release_chunk_list()
{
    chunk_list_.clear();
}

} // namespace shrink_array
