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

/**
 * @file array_chunk_parallel_common.h
 * @brief What the in-parallel teardown of the three array variants has in common.
 *
 * @details ArrayChunk, ArrayChunk2 and ArrayChunk2A each had their own
 *          clear_in_par(), and the three were the same function: take a share of
 *          the chunk list, collect the chunks that still own a heap buffer, park
 *          those buffers in the calling thread's free-list, put the chunks back
 *          to a constant, then settle the array's size according to
 *          @c opt_chunk_list_size.
 *
 *          Only two steps genuinely differed between them — WHICH chunks to
 *          collect, and HOW to park them — so those two are the callables
 *          clear_in_par_impl() takes. Everything else lives here, once.
 *
 *          That split is not cosmetic. The three copies had already drifted: v2
 *          selects with @c owns_live_buffer() and carries a comment explaining
 *          that @c written() there zeroed the STATIC cst_val2 array and parked
 *          the marker, while v1 and v2A still select with @c written(). Both are
 *          right — v1's Chunk has a single constant, and v2A tracks its second
 *          value per chunk rather than by index, so neither can mistake a marker
 *          for a live buffer — but nothing said so, and nothing made the next
 *          fix of that kind reach all three. Now the shared part cannot drift,
 *          and what is left in each variant is exactly what is specific to it.
 *
 *          The registries used by the ChunkDef::async_reset_list_size branch
 *          are AsyncResetRegistry<chunk_type>: one per chunk type, shared by the
 *          arrays of every index type, and the same one the manager of that
 *          chunk type sizes and clears. (They used to be statics of this class,
 *          i.e. one per Derived -- while the manager only ever sized
 *          ArrayChunk<T, int>'s.)
 *
 * @see array_chunk_parallel_common_inline.h for the body
 * @see array_chunk_parallel.h, array_chunk2_parallel.h, array_chunk2A_parallel.h
 *      for the three mixins that build on this one
 */

#include <type_traits>
#include <vector>

#include "chunk/config.h"
#include "chunk/chunk_traits.h"

namespace shrink_array
{

/**
 * @brief Per-thread lists of the chunk_list / nb_chunk a deferred teardown
 *        (ChunkDef::async_reset_list_size) still has to reset.
 * @details Keyed by the chunk type alone: filled by clear_in_par_impl() for an
 *          array of any index type, sized and cleared by
 *          ManagerChunk<chunk type>.
 */
template <typename ChunkType>
struct AsyncResetRegistry
{
    /** @brief The chunk lists thread @p thr_rank still has to reset. */
    static std::vector<std::vector<ChunkType>*>& get_list_chunk_list(int thr_rank)
    {
        return list_chunk_list[thr_rank];
    }

    /** @brief The whole per-thread table of pending chunk lists. */
    static std::vector<std::vector<std::vector<ChunkType>*>>& get_list_chunk_list()
    {
        return list_chunk_list;
    }

    /** @brief The chunk counters thread @p thr_rank still has to reset. */
    static std::vector<int*>& get_list_nb_chunk(int thr_rank)
    {
        return list_nb_chunk[thr_rank];
    }

    /** @brief The whole per-thread table of pending chunk counters. */
    static std::vector<std::vector<int*>>& get_list_nb_chunk()
    {
        return list_nb_chunk;
    }

    static inline std::vector<std::vector<std::vector<ChunkType>*>>
                                                 list_chunk_list; ///< Backing store of get_list_chunk_list().
    static inline std::vector<std::vector<int*>> list_nb_chunk;   ///< Backing store of get_list_nb_chunk().
};

/**
 * @brief The parallel-teardown machinery shared by the three array variants.
 *
 * @tparam Derived the concrete array type (ArrayChunk, ArrayChunk2 or
 *         ArrayChunk2A). Only reached through the per-variant mixin.
 */
template <typename Derived>
class ParallelSupportCommon
{
  public:
    using Traits       = ArrayChunkTraits<Derived>; ///< Traits specialisation of the concrete array.
    using value_type   = Traits::value_type;        ///< Stored element type.
    using chunk_type   = Traits::chunk_type;        ///< Chunk type held in the array's chunk_list.
    using manager_type = Traits::manager_type;      ///< Per-type manager owning the free-lists.

    /**
     * @brief Grow @p thr_rank's block free-list ahead of parking @p nb buffers.
     * @details Called by clear_in_par_impl() once it knows how many buffers the
     *          teardown is about to park, so the pushes that follow never
     *          reallocate mid-region. One copy for the three variants.
     */
    void move_list_block_free_tmp(int thr_rank, int nb)
    {
        auto& lbf = manager_type::get_list_block_free_thr(thr_rank);
        if (lbf.size() + nb > lbf.capacity())
            lbf.reserve(lbf.size() + 5 * nb);
    }

    /**
     * @name Collected-entry types
     *
     * What the selection loop stores per chunk. Two shapes, because only
     * ArrayChunk2A needs the chunk's index (to reach that chunk's row in the
     * per-tab metadata table) and it must not cost the other two anything: a
     * teardown of a large array collects one entry per written chunk, so
     * widening the entry would widen that vector for every variant.
     *
     * Both are constructible from the same `(chunk_type*, int)` pair, so
     * clear_in_par_impl() appends to either without knowing which it has, and
     * both forward `operator->` to the chunk so the park step reads alike.
     * @{
     */
    /** @brief Chunk only — the entry of the variants that never look at the index. */
    struct ChunkEntry
    {
        chunk_type* chunk;

        ChunkEntry(chunk_type* c, int /*index*/) noexcept : chunk(c) {}

        [[nodiscard]] chunk_type* operator->() const noexcept
        {
            return chunk;
        }
    };

    /** @brief Chunk plus its index in the array — ArrayChunk2A's entry. */
    struct ChunkIndexEntry
    {
        chunk_type* chunk;
        int         index;

        ChunkIndexEntry(chunk_type* c, int j) noexcept : chunk(c), index(j) {}

        [[nodiscard]] chunk_type* operator->() const noexcept
        {
            return chunk;
        }
    };

    // The whole point of ChunkEntry: it must cost exactly a pointer, so the
    // variants that ignore the index keep the footprint they had when each one
    // carried its own chunk_type* vector. Both asserts are load-bearing --
    // together they make std::vector<ChunkEntry> behave exactly like the
    // std::vector<chunk_type*> it replaced: same width, and a growth that is a
    // memmove rather than an element-by-element copy.
    static_assert(sizeof(ChunkEntry) == sizeof(chunk_type*),
                  "ChunkEntry must stay pointer-sized: it exists so that the variants which "
                  "ignore the chunk index do not pay for it in the teardown vector.");
    static_assert(std::is_trivially_copyable_v<ChunkEntry> && std::is_trivially_copyable_v<ChunkIndexEntry>,
                  "Both entry types must stay trivially copyable, so the collection vector "
                  "relocates with memmove on growth.");
    /** @} */

    /**
     * @name Async-reset registries (ChunkDef::async_reset_list_size)
     * @brief This chunk type's AsyncResetRegistry.
     * @{
     */
    static std::vector<std::vector<chunk_type>*>& get_list_chunk_list(int thr_rank)
    {
        return AsyncResetRegistry<chunk_type>::get_list_chunk_list(thr_rank);
    }

    static std::vector<int*>& get_list_nb_chunk(int thr_rank)
    {
        return AsyncResetRegistry<chunk_type>::get_list_nb_chunk(thr_rank);
    }
    /** @} */

    /**
     * @brief Drain the deferred writes still queued on this array's chunks.
     *
     * @details Must run before anything frees or re-points chunk storage
     *          (clear(), resize(), rechunk_to()): a deferred op replayed later
     *          would dereference a chunk this array no longer owns. The dirty
     *          bitset is the coarse pre-filter — only marked chunks are visited.
     *          Serial-context only (no concurrent writer or compressor).
     *
     * @note Guarded on chunk_config::use_fast_block_compression inside: only that
     *       mode queues deferred ops at all.
     */
    /// @param force  When true, bypass the any_pending_ops() / dirty-bit gates
    ///               and flush EVERY chunk. Required before rechunk_to() swaps
    ///               chunk_list away: an op queued on an old-geometry chunk is
    ///               keyed by that chunk's address, so once the old vector is
    ///               destroyed the op survives with a dangling key, gets a
    ///               freshly-allocated chunk of ANOTHER geometry at the recycled
    ///               address, and is replayed out of bounds. The pending-ops
    ///               counter can read 0 while such an op is still queued
    ///               (observed), so the gate is not safe to trust here.
    void flush_pending_deferred_ops(bool force = false);

  protected:
    /**
     * @brief The in-parallel teardown, minus the two steps each variant defines itself.
     *
     * @details Called from inside a parallel region, by every thread of the team
     *          (the `omp for` and the barriers below are worksharing constructs).
     *          In order: take this thread's share of the chunk list and collect
     *          the chunks @p select accepts, reserve room for them in the
     *          thread's block free-list, hand them to @p park_all, then settle
     *          the array's size.
     *
     * @tparam Entry the collected-entry type — ChunkEntry, or ChunkIndexEntry
     *         when the park step needs each chunk's index. Explicit, not
     *         deduced: it decides the width of the collection vector.
     * @tparam SelectFn callable `bool(const chunk_type&)` — true for a chunk that
     *         still owns a heap buffer and must be torn down. This is where the
     *         three variants differ on how "owns a buffer" is tested.
     * @tparam ParkFn callable `void(std::vector<Entry>&, int thr_rank)` —
     *         releases the buffers of the selected chunks and puts those chunks
     *         back to a constant. Invoked once, only when at least one chunk was
     *         selected, and responsible for choosing which free-list receives
     *         the buffers.
     *
     * @param opt_chunk_list_size one of ChunkDef::reset_list_size (barrier, then
     *        reset m_size and nb_chunk), ChunkDef::async_reset_list_size (park
     *        chunk_list / nb_chunk in the registries above for a later reset, no
     *        barrier), or anything else (barrier, then reset nb_chunk).
     *
     * @note The chunk count is read into a local BEFORE the loop: the tail
     *       resets `arr.nb_chunk`, and the loop bound must not follow it.
     * @note The teardown statistic is counted here, on selection, so that all
     *       three variants keep reporting it identically.
     */
    template <typename Entry, typename SelectFn, typename ParkFn>
    void clear_in_par_impl(int opt_chunk_list_size, SelectFn&& select, ParkFn&& park_all);

    /** @brief CRTP down-cast: the concrete array this mixin is a base of. */
    [[nodiscard]] Derived& self() noexcept
    {
        return static_cast<Derived&>(*this);
    }
    /** @copydoc self() */
    [[nodiscard]] const Derived& self() const noexcept
    {
        return static_cast<const Derived&>(*this);
    }
};

/**
 * @brief Serial counterpart: the teardown that has no team to split work across.
 *
 * @details The three serial mixins each carried the same three-line fallback.
 *          It is a plain clear(), guarded on the array having chunks at all.
 *
 * @tparam Derived the concrete array type.
 */
template <typename Derived>
class ParallelSupportCommonSerial
{
  public:
    /** @brief No-op: nothing is deferred without OpenMP. */
    void flush_pending_deferred_ops(bool = false) noexcept {}

    /** @brief Falls back to a plain clear(): there is no team to split the work across. */
    void clear_in_par(int /*opt_chunk_list_size*/)
    {
        Derived& arr = self();

        if (!arr.nb_chunk)
            return;

        arr.clear();
    }

    /** @brief No-op: there is no per-thread block free-list to grow. */
    void move_list_block_free_tmp(int /*thr_rank*/, int /*nb*/) noexcept {}

  protected:
    /** @brief CRTP down-cast: the concrete array this mixin is a base of. */
    [[nodiscard]] Derived& self() noexcept
    {
        return static_cast<Derived&>(*this);
    }
    /** @copydoc self() */
    [[nodiscard]] const Derived& self() const noexcept
    {
        return static_cast<const Derived&>(*this);
    }
};

} // namespace shrink_array
