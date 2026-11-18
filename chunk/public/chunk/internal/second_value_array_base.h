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

#include <cstddef> // std::size_t

#include "chunk/internal/array_chunk_base.h" // ArrayChunkBase
#include "chunk/config.h"                    // build_config::not_threaded, chunk_config::use_fast_block_compression
#include "chunk/internal/chunk_traits.h"     // is_zero_any_family_v
#include "chunk/internal/in_t.h"             // InT<T>
#include "chunk/internal/pmp_chunk.h"        // CHUNK_PMP_IN_PARALLEL_REGION, pmp::team
#include "chunk/internal/read_reclaim.h"     // read_reclaim::PassScope (perform_compression / compress in a region)

namespace shrink_array
{
/**
 * @brief The part of zero_or_value::Array and zero_any::Array that the two share, on top of
 *        ArrayChunkBase: chunk allocation, compress(), and the common part of
 *        resize().
 */
template <typename Derived>
class SecondValueArrayBase : public ArrayChunkBase<Derived>
{
  private:
    using Base = ArrayChunkBase<Derived>;

  public:
    using Traits      = Base::Traits;      ///< Traits specialisation of the concrete array.
    using ChunkType   = Base::ChunkType;   ///< Chunk type held in chunk_list_.
    using ManagerType = Base::ManagerType; ///< Per-type manager owning the compression machinery.
    using T           = Base::T;           ///< Stored element type.
    using IndexType   = Base::IndexType;   ///< Signed global index type.
    using ProxyType   = Base::ProxyType;   ///< Write proxy returned by the non-const accessors.

    using Base::derived;

    using Base::chunk_count_;
    using Base::chunk_list_;
    using Base::m_size_;


    // ── Identical methods → factorised here ───────────────────────────────

    /**
     * @brief Allocates storage for a specific chunk if it's currently constant.
     * Uses the metadata provided by the derived class.
     */
    void alloc_chunk(IndexType index)
    {
        const int chunk_index = this->chunk_index_of(index);

        // Deliberately NO dirty-marking: the dirty bit is for the DEFERRED path only, which
        // marks at enqueue (see DeferredReadGate, chunk_base.h). Marking here would enable the
        // read gate for preallocated chunks. Exercised by shrink_bench --test_prealloc.
        chunk_list_[chunk_index].alloc_chunk(derived().get_metadata(index));
    }

    /**
     * @brief Thread-aware allocation for a specific chunk.
     */
    void alloc_chunk(IndexType index, int thr_rank)
    {
        const int chunk_index = this->chunk_index_of(index);
        // See alloc_chunk(index) above: deliberately no dirty-marking.
        chunk_list_[chunk_index].alloc_chunk(derived().get_metadata(index), thr_rank);
    }

    // flush_pending_deferred_ops() lives in ParallelSupportCommon / ...Serial
    // (array_chunk_parallel_common.h), a sibling base: reach it through derived().

    /// @brief Compress every compressible chunk now, by every thread of the
    ///        team compress_int() runs on (the caller's region, or one opened
    ///        here when called outside any).
    void compress()
    {
        if (CHUNK_PMP_IN_PARALLEL_REGION)
        {
            // Inside the caller's region: other threads may be reading, so
            // the buffers released wait for a grace period (read_reclaim.h).
            const read_reclaim::PassScope pass_scope;
            derived().compress_int();
        }
        else
        {
            pmp::team(pmp::Options{}, [&](pmp::Team& team) { derived().compress_int(team); });
        }
    }

  protected:
    /// @brief The part of resize() both second-value variants share: compress the
    ///        pending candidates, drain the deferred ops, then grow or shrink
    ///        chunk_list_ (with its dirty bitset and the manager's counters).
    ///        The variant's resize() materialises or seeds the new chunks.
    void resize_common(IndexType newsz)
    {
        // Drain deferred ops on the CURRENT chunks while chunk_count, chunk_list_ and the dirty
        // bitset still have the OLD size (after chunk_count grows, is_chunk_dirty(j) would read
        // past the old bitset).
        //
        // And BEFORE the compression pass below: a replayed write can queue a chunk again
        // (&chunk_list_[j] pushed, flag 1); flushed after the pass, that entry would dangle
        // after chunk_list_.resize() and the chunk could be queued twice.
        // force: see flush_pending_deferred_ops (array_chunk_parallel_common.h).
        derived().flush_pending_deferred_ops(/*force=*/true);

        // Compress the pending list BEFORE chunk_list_ is reallocated: the candidate lists hold
        // &chunk_list_[j] pointers that a reallocation would leave dangling. Needed in non-OMP
        // too, where the sequential compression path also queues chunks. This pass queues
        // nothing, so afterwards every list is empty and every dedup flag 0 -- the
        // precondition of zero_any::Proxy::resize_array_is_int(array_id, ...).
        if constexpr (chunk_config::use_fast_block_compression ||
                      (build_config::not_threaded && is_zero_any_family_v<ChunkType>))
        {
            // Not make_block_compression(): resize() may run in a region, on an
            // array of the calling thread's own -- then this array's candidates
            // only, the thread's list holding other arrays' chunks too.
            if (CHUNK_PMP_IN_PARALLEL_REGION)
            {
                derived().drain_own_candidates();
            }
            else
            {
                derived().drain_candidates();
            }
        }

        const int old_chunk_count = chunk_count_;
        m_size_                   = newsz;
        chunk_count_              = this->chunks_for(newsz);

        if (chunk_count_ < old_chunk_count)
        {
            // Shrink: release the chunks past the new end NOW (same release as clear_core()).
            // Do not return early keeping them: teardown loops stop at chunk_count, so their
            // buffers would leak. The pass above left none of them referenced.
            derived().release_chunk_range(chunk_count_, old_chunk_count);
            this->await_chunk_readers();
            chunk_list_.resize(static_cast<std::size_t>(chunk_count_));
            Base::reallocate_dirty_bitset(old_chunk_count, chunk_count_);

            // Fewer chunks alive: so is the count the compression threshold and reuse budget use.
            ManagerType::inc_total_chunk_create(chunk_count_ - old_chunk_count);
            ManagerType::compute_size_buffer_reuse();
            ManagerType::compute_size_max_number_of_chunk_to_check_compress();
            return;
        }
        if (chunk_count_ == old_chunk_count)
        {
            return;
        }

        this->await_chunk_readers();
        chunk_list_.resize(chunk_count_);
        Base::reallocate_dirty_bitset(old_chunk_count, chunk_count_);

        ManagerType::inc_total_chunk_create(chunk_count_ - old_chunk_count);
        ManagerType::compute_size_buffer_reuse();
        ManagerType::compute_size_max_number_of_chunk_to_check_compress();
        // An array built empty (the default) only becomes worth its own slabs
        // here; idempotent for one that already has a key.
        this->assign_alloc_key();
    }
};

} // namespace shrink_array