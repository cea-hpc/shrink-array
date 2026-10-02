
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

#include "chunk/internal/array_chunk_base.h"
#include "chunk/config.h"
#include "chunk/chunk_traits.h"
#include "chunk/internal/in_t.h"
#include "chunk/PMP_CHUNK.h" // CHUNK_PMP_IN_PARALLEL_REGION
#include "chunk/internal/read_reclaim.h" // read_reclaim::PassScope (perform_compression / compress in a region)

namespace shrink_array
{
/**
 * @brief Cache-line aligned atomic word to prevent false sharing.
 * Each word covers 64 chunks.
 */

template <typename Derived>
class ArrayChunk2Base : public ArrayChunkBase<Derived>
{
    using Base = ArrayChunkBase<Derived>;

  public:
    using Traits       = Base::Traits;
    using chunk_type   = Base::chunk_type;
    using manager_type = Base::manager_type;
    using T            = Base::T;
    using ind_type     = Base::ind_type;
    using proxy_type   = Base::proxy_type;

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

        // Deliberately NO dirty-marking here. Historical note: this used to
        // call mark_chunk_dirty(i) with the ELEMENT index — an out-of-bounds
        // bitset write (word_idx = i>>6 vs bitset sized (nb_chunk+63)/64,
        // PaddedDirtyWord is 128 B: up to chunk_size x past the end) that no
        // reader could ever observe (is_chunk_dirty reads at num_chunk>>6).
        // Removing it is observationally identical minus the UB. The dirty
        // bit exists for the DEFERRED path only, and that path marks
        // correctly at enqueue (proxy_base_inline: mark_chunk_dirty(
        // ctx->num_chunk)); the visibility protocol itself lives in
        // DeferredReadGate (chunk_base.h). Marking num_chunk here instead
        // would ENABLE the read gate for freshly-allocated chunks — a
        // behaviour change a preallocation must not smuggle in (and the
        // matching ArrayChunk v1 alloc_chunk never marked either).
        // Exercised by shrink_bench --test_prealloc.
        chunk_list[num_chunk].alloc_chunk(derived().get_metadata(i));
    }

    /**
     * @brief Thread-aware allocation for a specific chunk.
     */
    void alloc_chunk(ind_type i, int thr_rank)
    {
        const int num_chunk = this->chunk_index_of(i);
        // See alloc_chunk(i) above: deliberately no dirty-marking (the old
        // element-index mark was an unobservable OOB write).
        chunk_list[num_chunk].alloc_chunk(derived().get_metadata(i), thr_rank);
    }

    // ── Drain deferred writes still targeting this array's chunks ──────────
    // MUST be called before ANY operation that frees, moves, or reallocates
    // chunk_list (clear, resize growth, move/copy assignment). The deferred
    // maps (CompressionManager::deferred_chunk_per_thread + the per-thread op
    // maps) are global and keyed by chunk pointer, so they outlive this array
    // and survive chunk_list reallocation. Invalidating chunk storage without
    // draining leaves DANGLING chunk references that a later flush_deferred —
    // triggered by compression on ANY array — dereferences (observed: SEGV in
    // values_load() from a stale deferred op after a resize()/reassignment).
    //
    // Defined here in the CRTP base so BOTH ArrayChunk2 and ArrayChunk2A get it
    // (each resolves chunk_type::CompressionManagerType to its own manager).
    // flush_chunk_all_ranks() drains a chunk across all ranks AND erases it
    // from deferred_chunk_per_thread. The dirty bit is reliable for the
    // deferred path (it marks with the CHUNK index via mark_chunk_dirty(
    // ctx->num_chunk); the OOB caveat in alloc_chunk concerns the element-index
    // alloc path only). Serial-context only.
    // flush_pending_deferred_ops() lives in ParallelSupportCommon / ...Serial
    // (array_chunk_parallel_common.h), which the concrete arrays mix in. It used
    // to be duplicated here AND in ParallelSupport for ArrayChunk v1 -- the two
    // bodies were identical. Reach it through derived(): it is a sibling base of
    // this one, so `this->` cannot see it.

  protected:
    /// @brief The part of resize() both chunk2 variants share: compress the
    ///        pending candidates, drain the deferred ops, then grow or shrink
    ///        chunk_list (with its dirty bitset and the manager's counters).
    ///        The variant's resize() materialises or seeds the new chunks.
    void resize_common(ind_type newsz)
    {
        // Compress the pending list BEFORE chunk_list is reallocated below.
        // vec_to_compress_potantially holds &chunk_list[j] pointers; a growth
        // resize reallocates the vector and would leave them dangling, so they
        // must be consumed first. In OMP this runs under use_fast_block_
        // compression; in non-OMP the sequential compression path (driven from
        // the write path) can also have queued chunks, so it must run here too —
        // otherwise the next perform_compression dereferences freed chunks
        // (SIGSEGV in fill_list_to_compress -> values_load).
        if constexpr (chunk_config::use_fast_block_compression ||
                      (build_config::not_use_openmp && std::is_same_v<Derived, ArrayChunk2A<T, ind_type>>))
        {
            derived().make_block_compression();
        }

        // Drain pending deferred ops on the CURRENT chunks BEFORE nb_chunk,
        // chunk_list and the dirty bitset are grown below. Called here — while
        // nb_chunk still holds the OLD count and chunk_list / chunk_dirty_bitset
        // are still the OLD size — so is_chunk_dirty(j) and &chunk_list[j] stay
        // in bounds. (Calling it after nb_chunk grew but before the reallocs
        // read past the old bitset → heap-buffer-overflow.)
        derived().flush_pending_deferred_ops();

        const int old_nb_chunk = nb_chunk;
        m_size                 = newsz;
        nb_chunk               = this->chunks_for(newsz);

        if (nb_chunk <= old_nb_chunk)
            return; // Only handling growth here

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

        const auto lbd = [this, old_size, newsz, val]
        {
#pragma omp for schedule(static) nowait
            for (ind_type i = old_size; i < newsz; ++i)
            {
                derived()[i] = val;
            }
        };

        if (!CHUNK_PMP_IN_PARALLEL_REGION && nb_chunk > 500)
        {
#pragma omp parallel
            lbd();
        }
        else
        {
            lbd();
        }
    }

  public:
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

    // mark_chunk_dirty / is_chunk_dirty are inherited from ArrayChunkBase —
    // the single owner of the dirty-bit verbs (this class used to carry a
    // byte-identical duplicate of mark_chunk_dirty plus a caller-less
    // mark_chunk_safe; both removed, one owner per protocol).

  private:
    /// @brief Is the chunk holding element @p i compressed to the first value?
    [[nodiscard]] bool is_cst_val1(ind_type i) const noexcept
    {
        return chunk_list[this->chunk_index_of(i)].is_cst_val1();
    }
};

} // namespace shrink_array