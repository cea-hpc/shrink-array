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
 * @file zero_one_value_array_parallel.h
 * @brief zero_one_value::Array's two mode-gated surfaces -- the OpenMP-only one (the
 *        deferred-op flush) and the fast-block-only one
 *        (make_block_compression) -- isolated in their own mixins so they
 *        disappear entirely from a build that has neither.
 *
 * @details Same split as ParallelSupport does for zero::Array:
 *          zero_one_value::ParallelSupport<Derived, threaded> carries
 *          flush_pending_deferred_ops() (ParallelSupportCommon, shared by the
 *          three variants), a no-op in the serial specialisation.
 *
 *          make_block_compression() gets its own mixin rather than joining
 *          the first one: its gate is chunk_config::use_fast_block_compression,
 *          which is STRICTER than threaded (it also needs embedded -- i.e. not
 *          manual -- compression).
 *
 * @see zero_one_value_array_parallel_inline.h for the body
 * @see zero_array_parallel.h for zero's version of the same split
 */

#include "chunk/config.h"       // build_config::threaded, chunk_config::use_fast_block_compression
#include "chunk/chunk_traits.h" // ArrayChunkTraits
#include "chunk/internal/array_chunk_parallel_common.h" // ParallelSupportCommon, ParallelSupportCommonSerial

namespace shrink_array
{

namespace zero_one_value
{

/**
 * @brief The OpenMP-only members of zero_one_value::Array, present only when the build has OpenMP.
 *
 * @tparam Derived the concrete array type (only zero_one_value::Array today).
 * @tparam Enabled defaulted to build_config::threaded; never pass it
 *         explicitly outside of tests.
 */
template <typename Derived, bool Enabled = build_config::threaded>
struct ParallelSupport;

// ── OpenMP build: the real parallel surface ──────────────────────────────────
/**
 * @brief OpenMP specialisation: the deferred-op flush.
 * @copydetails zero_one_value::ParallelSupport
 */
template <typename Derived>
struct ParallelSupport<Derived, true> : public ParallelSupportCommon<Derived>
{
    // flush_pending_deferred_ops() and self() come from ParallelSupportCommon.
};

// ── Serial build: nothing to parallelise ─────────────────────────────────────
/**
 * @brief Serial specialisation: the same verb, inert.
 */
template <typename Derived>
struct ParallelSupport<Derived, false> : public ParallelSupportCommonSerial<Derived>
{
};

/**
 * @brief zero_one_value::Array's fast-block compression pass, present only in that mode.
 *
 * @tparam Derived the concrete array type (only zero_one_value::Array today).
 * @tparam Enabled defaulted to chunk_config::use_fast_block_compression; never
 *         pass it explicitly outside of tests.
 *
 * @details Not virtual: the only call site (SecondValueArrayBase::resize_common)
 *          goes through CRTP, and no type-erased pass drives live arrays.
 */
template <typename Derived, bool Enabled = chunk_config::use_fast_block_compression>
class FastBlockSupport;

// ── Fast-block build: the real pass ──────────────────────────────────────────
/**
 * @brief Fast-block specialisation: the deferred-candidate compression pass.
 * @copydetails zero_one_value::FastBlockSupport
 */
template <typename Derived>
class FastBlockSupport<Derived, true>
{
  public:
    using Traits    = ArrayChunkTraits<Derived>; ///< Traits specialisation of the concrete array.
    using ChunkType = Traits::ChunkType;         ///< Chunk type held in the array's chunk_list.

    /**
     * @brief Compress every chunk of this array currently queued as a candidate.
     * @details Drains the calling thread's candidate list, or every thread's
     *          when called from outside a parallel region.
     */
    void make_block_compression();

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

// ── Any other mode: no candidate queue to drain ──────────────────────────────
/**
 * @brief Inert specialisation: nothing queues block-compression candidates.
 *
 * @details Kept (rather than removed) so that SecondValueArrayBase::resize_common,
 *          which guards the call with the same condition, still names a
 *          declared member in every mode.
 */
template <typename Derived>
class FastBlockSupport<Derived, false>
{
  public:
    /** @brief No-op: this mode has no deferred candidate list. */
    static void make_block_compression() noexcept {}
};

} // namespace zero_one_value

} // namespace shrink_array
