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
 *          it is the same for zero::Array, zero_one_value::Array and zero_any::Array, so it
 *          lives here once. The fast-block pass stays per variant
 *          (zero_array_parallel.h, zero_one_value_array_parallel.h). The serial
 *          counterpart is a no-op.
 *
 * @see array_chunk_parallel_common_inline.h for the body
 */

#include "chunk/internal/chunk_traits.h" // ArrayChunkTraits
#include "chunk/config.h"                // build_config::threaded

namespace shrink_array
{

/// @brief Refuse a public make_block_compression() inside a region (the body:
///        array_chunk_parallel_common_inline.h).
[[nodiscard]] inline bool block_pass_refused_in_region();

/**
 * @brief The OpenMP machinery shared by the three array variants.
 *
 * @tparam Derived the concrete array type (zero::Array, zero_one_value::Array or
 *         zero_any::Array). Only reached through ParallelSupport.
 */
template <typename Derived>
class ParallelSupportCommon
{
  public:
    using Traits      = ArrayChunkTraits<Derived>; ///< Traits specialisation of the concrete array.
    using value_type  = Traits::value_type;        ///< Stored element type.
    using ChunkType   = Traits::ChunkType;         ///< Chunk type held in the array's chunk_list_.
    using ManagerType = Traits::ManagerType;       ///< Per-type manager owning the free-lists.

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
     * @param force Bypass the any_pending_ops() gate and drain every rank. Required at
     *              any sequential point that reallocates or frees chunks (rechunk_to(),
     *              resize(), swap(), clear_core()): an op is keyed by its chunk's address,
     *              and once the old vector is gone it would be replayed, out of bounds,
     *              on whatever chunk is allocated there. The counter can read 0 while such
     *              an op is still queued, so the gate cannot be trusted here.
     *
     * @param own_rank_only Drain the calling thread's rank alone: the teardown, inside a
     *              region, of an array no other rank holds a write of (see
     *              owned_by_calling_rank()).
     *
     * @note Guarded on chunk_config::use_fast_block_compression inside: only
     *       that mode queues deferred ops at all.
     */
    void flush_pending_deferred_ops(bool force = false, bool own_rank_only = false);
};

/**
 * @brief Serial counterpart: nothing is deferred without OpenMP.
 * @tparam Derived the concrete array type.
 */
template <typename Derived>
struct ParallelSupportCommonSerial
{
    /** @brief No-op: nothing is deferred without OpenMP. */
    static void flush_pending_deferred_ops(bool /*force*/ = false, bool /*own_rank_only*/ = false) noexcept {}
};

/**
 * @brief The OpenMP-only members of an array: the deferred-op flush, inert
 *        without threads.
 *
 * @tparam Derived the concrete array type.
 * @tparam Enabled defaulted to build_config::threaded; never pass it
 *         explicitly outside of tests -- the point of the parameter is that the
 *         two specialisations below are selected by the build mode.
 */
template <typename Derived, bool Enabled = build_config::threaded>
struct ParallelSupport;

/// @brief Threaded build: the real surface (flush_pending_deferred_ops() and self()).
template <typename Derived>
struct ParallelSupport<Derived, true> : public ParallelSupportCommon<Derived>
{
};

/// @brief Serial build: the same verb, inert.
template <typename Derived>
struct ParallelSupport<Derived, false> : public ParallelSupportCommonSerial<Derived>
{
};

} // namespace shrink_array
