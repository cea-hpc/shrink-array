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

/**
 * @file array_chunk_parallel_common.h
 * @brief What the three array variants share under OpenMP: the drain of the
 *        deferred writes queued on an array's chunks.
 *
 * @details flush_pending_deferred_ops() must run before anything frees or
 *          re-points chunk storage (clear(), resize(), rechunk_to(), swap());
 *          it is the same for ArrayChunk, ArrayChunk2 and ArrayChunk2A, so it
 *          lives here once, under the per-variant mixins
 *          (array_chunk_parallel.h, array_chunk2_parallel.h,
 *          array_chunk2A_parallel.h). The serial counterpart is a no-op.
 *
 * @see array_chunk_parallel_common_inline.h for the body
 */

#include "chunk/chunk_traits.h" // ArrayChunkTraits

namespace shrink_array
{

/**
 * @brief The OpenMP machinery shared by the three array variants.
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
     * @brief Drain the deferred writes still queued on this array's chunks.
     *
     * @details Must run before anything frees or re-points chunk storage
     *          (clear(), resize(), rechunk_to(), swap()): a deferred op
     *          replayed later would dereference a chunk this array no longer
     *          owns. One drain per rank (flush_deferred), not per chunk: it
     *          also applies the ops queued on the other arrays of the same
     *          (T, chunk type), still in order. Serial-context only (no
     *          concurrent writer or compressor).
     *
     * @param force When true, bypass the any_pending_ops() gate and drain
     *              every rank regardless. Required before rechunk_to() swaps
     *              chunk_list away: an op queued on an old-geometry chunk is
     *              keyed by that chunk's address, so once the old vector is
     *              destroyed the op survives with a dangling key, gets a
     *              freshly-allocated chunk of ANOTHER geometry at the recycled
     *              address, and is replayed out of bounds. The pending-ops
     *              counter can read 0 while such an op is still queued
     *              (observed), so the gate is not safe to trust here.
     *
     * @note Guarded on chunk_config::use_fast_block_compression inside: only
     *       that mode queues deferred ops at all.
     */
    void flush_pending_deferred_ops(bool force = false);
};

/**
 * @brief Serial counterpart: nothing is deferred without OpenMP.
 * @tparam Derived the concrete array type.
 */
template <typename Derived>
struct ParallelSupportCommonSerial
{
    /** @brief No-op: nothing is deferred without OpenMP. */
    void flush_pending_deferred_ops(bool = false) noexcept {}
};

} // namespace shrink_array
