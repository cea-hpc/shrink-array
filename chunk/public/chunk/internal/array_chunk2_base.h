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

#include "chunk/internal/array_chunk_base.h" // ArrayChunkBase
#include "chunk/config.h"                    // build_config::not_use_openmp, chunk_config::use_fast_block_compression
#include "chunk/chunk_traits.h"              // ArrayChunk2A forward declaration
#include "chunk/internal/in_t.h"             // in_t<T>
#include "chunk/PMP_CHUNK.h"                 // CHUNK_PMP_IN_PARALLEL_REGION
#include "chunk/internal/read_reclaim.h"     // read_reclaim::PassScope (perform_compression / compress in a region)

namespace shrink_array
{
/**
 * @brief The part of ArrayChunk2 and ArrayChunk2A that the two share, on top of
 *        ArrayChunkBase: chunk allocation, compress(), and the common part of
 *        resize().
 */
template <typename Derived>
class ArrayChunk2Base : public ArrayChunkBase<Derived>
{
  private:
    using Base = ArrayChunkBase<Derived>;

  public:
    using Traits       = Base::Traits;       ///< Traits specialisation of the concrete array.
    using chunk_type   = Base::chunk_type;   ///< Chunk type held in chunk_list.
    using manager_type = Base::manager_type; ///< Per-type manager owning the compression machinery.
    using T            = Base::T;            ///< Stored element type.
    using ind_type     = Base::ind_type;     ///< Signed global index type.
    using proxy_type   = Base::proxy_type;   ///< Write proxy returned by the non-const accessors.

    using Base::derived;

    using Base::chunk_list;
    using Base::m_size;
    using Base::nb_chunk;


    // ── Identical methods → factorised here ───────────────────────────────

    /**
     * @brief Allocates storage for a specific chunk if it's currently constant.
     * Uses the metadata provided by the derived class.
     */
    void alloc_chunk(ind_type i)
    {
        const int num_chunk = this->chunk_index_of(i);

        // Deliberately NO dirty-marking: the dirty bit is for the DEFERRED path only, which
        // marks at enqueue (see DeferredReadGate, chunk_base.h). Marking here would enable the
        // read gate for preallocated chunks. Exercised by shrink_bench --test_prealloc.
        chunk_list[num_chunk].alloc_chunk(derived().get_metadata(i));
    }

    /**
     * @brief Thread-aware allocation for a specific chunk.
     */
    void alloc_chunk(ind_type i, int thr_rank)
    {
        const int num_chunk = this->chunk_index_of(i);
        // See alloc_chunk(i) above: deliberately no dirty-marking.
        chunk_list[num_chunk].alloc_chunk(derived().get_metadata(i), thr_rank);
    }

    // flush_pending_deferred_ops() lives in ParallelSupportCommon / ...Serial
    // (array_chunk_parallel_common.h), a sibling base: reach it through derived().

    /// @brief Compress every compressible chunk now, by every thread of the
    ///        region compress_int() runs in (opened here when called outside one).
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
#pragma omp parallel
            derived().compress_int();
        }
    }

  protected:
    /// @brief The part of resize() both chunk2 variants share: compress the
    ///        pending candidates, drain the deferred ops, then grow or shrink
    ///        chunk_list (with its dirty bitset and the manager's counters).
    ///        The variant's resize() materialises or seeds the new chunks.
    void resize_common(ind_type newsz)
    {
        // Drain deferred ops on the CURRENT chunks while nb_chunk, chunk_list and the dirty
        // bitset still have the OLD size (after nb_chunk grows, is_chunk_dirty(j) would read
        // past the old bitset).
        //
        // And BEFORE the compression pass below: a replayed write can queue a chunk again
        // (&chunk_list[j] pushed, flag 1); flushed after the pass, that entry would dangle
        // after chunk_list.resize() and the chunk could be queued twice.
        // force: see flush_pending_deferred_ops (array_chunk_parallel_common.h).
        derived().flush_pending_deferred_ops(/*force=*/true);

        // Compress the pending list BEFORE chunk_list is reallocated: the candidate lists hold
        // &chunk_list[j] pointers that a reallocation would leave dangling. Needed in non-OMP
        // too, where the sequential compression path also queues chunks. This pass queues
        // nothing, so afterwards every list is empty and every dedup flag 0 -- the
        // precondition of Proxy2A::resize_tab_is_int(num_tab, ...).
        if constexpr (chunk_config::use_fast_block_compression ||
                      (build_config::not_use_openmp && std::is_same_v<Derived, ArrayChunk2A<T, ind_type>>))
        {
            derived().make_block_compression();
        }

        const int old_nb_chunk = nb_chunk;
        m_size                 = newsz;
        nb_chunk               = this->chunks_for(newsz);

        if (nb_chunk < old_nb_chunk)
        {
            // Shrink: release the chunks past the new end NOW (same release as clear_core()).
            // Do not return early keeping them: teardown loops stop at nb_chunk, so their
            // buffers would leak. The pass above left none of them referenced.
            derived().release_chunk_range(nb_chunk, old_nb_chunk);
            chunk_list.resize(static_cast<std::size_t>(nb_chunk));
            Base::reallocate_dirty_bitset(old_nb_chunk, nb_chunk);

            // Fewer chunks alive: so is the count the compression threshold and reuse budget use.
            manager_type::increm_somme_chunk_create(nb_chunk - old_nb_chunk);
            manager_type::compute_size_buffer_reuse();
            manager_type::compute_size_max_number_of_chunk_to_check_compress();
            return;
        }
        if (nb_chunk == old_nb_chunk)
        {
            return;
        }

        chunk_list.resize(nb_chunk);
        Base::reallocate_dirty_bitset(old_nb_chunk, nb_chunk);

        manager_type::increm_somme_chunk_create(nb_chunk - old_nb_chunk);
        manager_type::compute_size_buffer_reuse();
        manager_type::compute_size_max_number_of_chunk_to_check_compress();
        // An array built empty (the default) only becomes worth its own slabs
        // here; idempotent for one that already has a key.
        this->assign_alloc_key();
    }

    /// @brief resize(), then @p val written to the NEW elements only
    ///        ([old size, newsz) -- std::vector's semantics; assign() fills all).
    void resize(ind_type newsz, in_t<T> val)
    {
        const ind_type old_size = m_size;
        derived().resize(newsz); // calls derived resize(sz)

        // A team of its own outside a region (the library's threshold, the
        // phase bracketed), the calling thread inside one.
        this->write_range_by_chunks(old_size, newsz, [&val](ind_type) { return static_cast<T>(val); });
    }
};

} // namespace shrink_array