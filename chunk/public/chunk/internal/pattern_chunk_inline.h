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
#include <atomic>  // std::memory_order_release
#include <cstddef> // std::size_t
#include <cstdint> // std::uint64_t
#include <utility> // std::pair

#include "chunk/internal/pattern_chunk.h" // the class these bodies belong to

namespace shrink_array
{

// ---------------------------------------------------------------------------
// shrink_array::Chunk<T>
// ---------------------------------------------------------------------------

template <typename T>
[[nodiscard]] std::pair<const T*, T> pattern::Chunk<T>::find_motif(ChunkElems size)
{
    const auto count = static_cast<std::size_t>(size.count());
    motif_scratch_.resize(count);
    this->read_block(0, static_cast<int>(count), motif_scratch_.data());
    const T offset = motif_scratch_[0];
    for (auto& elem : motif_scratch_)
    {
        elem = wrapping_sub(elem, offset);
    }
    const std::uint64_t h = MotifRegion<T>::hash(motif_scratch_.data(), count);
    return {MotifRegion<T>::find_or_admit(motif_scratch_.data(), count, h, MotifRegion<T>::sighting_tag(this)), offset};
}

template <typename T>
void pattern::Chunk<T>::classify_motif(int chunk_index, int array_id, ChunkElems size)
{
    if (const auto [motif, offset] = find_motif(size); motif != nullptr)
    {
        motif_candidates_.emplace_back(this, motif, offset, chunk_index, array_id, size);
    }
}

template <typename T>
void pattern::Chunk<T>::compress_to_motif(int thr, int chunk_index, int array_id, ChunkElems size)
{
    using ProxyT = ChunkTraits<Chunk>::ProxyType;
    if (const auto [motif, offset] = find_motif(size); motif != nullptr)
    {
        // The geometry pin is the caller's (compress_chunk_to_compress_in_par()).
        const SecondValueRef<T> slot = ProxyT::BaseSpec::get_tab_second_value(array_id, chunk_index);
        block_clear_motif(&ProxyT::tab_type_cst_val[array_id][chunk_index], slot, motif, offset, thr);
    }
}

template <typename T>
void pattern::Chunk<T>::compress_motif_candidates(int thr)
{
    using ProxyT = ChunkTraits<Chunk>::ProxyType;
    for (const auto& [chunk, motif, offset, chunk_index, array_id, size] : motif_candidates_)
    {
        // Per-chunk geometry pin, as make_block_compression() for the constants: the
        // buffer goes back to the pool of ITS size (dynamic builds).
        if constexpr (build_config::chunk_dynamic)
        {
            shrink_array::set_thread_chunk_size(size.count());
        }
        const SecondValueRef<T> slot = ProxyT::BaseSpec::get_tab_second_value(array_id, chunk_index);
        chunk->block_clear_motif(&ProxyT::tab_type_cst_val[array_id][chunk_index], slot, motif, offset, thr);
        chunk->steal_unmark();
    }
    motif_candidates_.clear();
}

template <typename T>
void pattern::Chunk<T>::block_clear_motif(CstSlot*          ptr_type_cst,
                                          SecondValueRef<T> slot,
                                          const T*          motif,
                                          T                 offset,
                                          int               thr_rank)
{
    std::atomic_ref<T>(*slot.slot).store(offset, std::memory_order_release);
    *ptr_type_cst = CstSlot::motif;
    increm_chunk_compress<Chunk>();
    add_chunk_bytes_freed<Chunk>(static_cast<long>(sizeof(T)) * static_cast<long>(shrink_array::alloc_chunk_count()));
    if constexpr (build_config::threaded)
    {
        T* const old = this->try_mark_cst_and_get_old(1, const_cast<T*>(motif)); // NOSONAR(cpp:S859)
        if (old == nullptr) [[unlikely]]
        {
            CHUNK_INVARIANT_FAILED("block_clear_motif: the CAS from the buffer to the motif failed on a claimed "
                                   "chunk ({})",
                                   static_cast<const void*>(this));
        }
        this->free_block_to_pool(thr_rank, old);
    }
    else
    {
        T* const old  = this->values_;
        this->values_ = const_cast<T*>(motif); // NOSONAR(cpp:S859)
        this->free_values(old);
    }
}

} // namespace shrink_array
