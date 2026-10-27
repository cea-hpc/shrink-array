/*
 * Copyright 2026 CEA (www.cea.fr)
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

#include <vector>

#include "chunk/pattern_array.h"         // the class these bodies belong to
#include "chunk/internal/motif_region.h" // pattern::MotifRegion (install_motif_for_test)

namespace shrink_array
{

// ---------------------------------------------------------------------------
// shrink_array::Array<T, IntType>
// ---------------------------------------------------------------------------

template <ArrayChunkElement T, typename IntType>
pattern::Array<T, IntType>& pattern::Array<T, IntType>::operator=(Array&& other) noexcept
{
    if (this != &other)
    {
        this->clear();
        this->swap_storage_and_identity(other);
    }
    return *this;
}

template <ArrayChunkElement T, typename IntType>
bool pattern::Array<T, IntType>::install_motif_for_test(int num_chunk, const T* elems)
{
    using ChunkT = typename Array::ChunkType;
    auto& chunk  = this->chunk_list_[static_cast<std::size_t>(num_chunk)];
    if (!chunk.is_cst_val1())
    {
        return false;
    }
    const auto     count = static_cast<std::size_t>(this->chunk_size());
    std::vector<T> motif(count);
    for (std::size_t j = 0; j < count; ++j)
    {
        motif[j] = ChunkT::wrapping_sub(elems[j], elems[0]);
    }
    const T* const entry = MotifRegion<T>::find_or_register(motif.data(), count);
    // The offset first, then the pointer: the order of the compression to come.
    Array::ProxyType::BaseSpec::set(this->num_tab_, num_chunk, elems[0]);
    chunk.values_store(const_cast<T*>(entry)); // never written through: a motif chunk materialises first
    return true;
}

template <ArrayChunkElement T, typename IntType>
std::size_t pattern::Array<T, IntType>::reclaim_second_values()
{
    const std::size_t freed = zero_any::ArrayImpl<Array>::reclaim_second_values();
    if (!CHUNK_PMP_IN_PARALLEL_REGION)
    {
        (void)reclaim_motifs();
    }
    return freed;
}

template <ArrayChunkElement T, typename IntType>
std::size_t pattern::Array<T, IntType>::reclaim_motifs()
{
    using ChunkT = typename Array::ChunkType;
    std::vector<std::uint8_t> referenced(MotifRegion<T>::ids_carved() + 1, 0);
    Array::ProxyType::share_info_array->for_each_key(
        [&referenced](ArrayKey<ChunkT> arr)
        {
            if (arr == nullptr)
            {
                return;
            }
            const int nc = arr->chunk_count();
            for (int j = 0; j < nc; ++j)
            {
                const std::uint32_t id =
                    MotifRegion<T>::id_of(arr->chunk_ref(static_cast<std::size_t>(j)).values_load());
                if (id != 0 && id < referenced.size())
                {
                    referenced[id] = 1;
                }
            }
        });
    return MotifRegion<T>::reclaim_unmarked(referenced);
}

} // namespace shrink_array
