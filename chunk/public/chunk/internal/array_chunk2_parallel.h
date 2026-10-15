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
 * @file array_chunk2_parallel.h
 * @brief ArrayChunk2's two mode-gated surfaces -- the OpenMP-only one (the
 *        deferred-op flush) and the fast-block-only one
 *        (make_block_compression) -- isolated in their own mixins so they
 *        disappear entirely from a build that has neither.
 *
 * @details Same split as ParallelSupport does for ArrayChunk (v1):
 *          ParallelSupport2<Derived, threaded> carries
 *          flush_pending_deferred_ops() (ParallelSupportCommon, shared by the
 *          three variants), a no-op in the serial specialisation.
 *
 *          make_block_compression() gets its own mixin rather than joining
 *          the first one: its gate is chunk_config::use_fast_block_compression,
 *          which is STRICTER than threaded (it also needs embedded -- i.e. not
 *          manual -- compression).
 *
 * @see array_chunk2_parallel_inline.h for the body
 * @see array_chunk_parallel.h for v1's version of the same split
 */

#include "chunk/config.h"       // build_config::threaded, chunk_config::use_fast_block_compression
#include "chunk/chunk_traits.h" // ArrayChunkTraits
#include "chunk/internal/array_chunk_parallel_common.h" // ParallelSupportCommon, ParallelSupportCommonSerial

namespace shrink_array
{

/**
 * @brief The OpenMP-only members of ArrayChunk2, present only when the build has OpenMP.
 *
 * @tparam Derived the concrete array type (only ArrayChunk2 today).
 * @tparam Enabled defaulted to build_config::threaded; never pass it
 *         explicitly outside of tests.
 */
template <typename Derived, bool Enabled = build_config::threaded>
struct ParallelSupport2;

// ── OpenMP build: the real parallel surface ──────────────────────────────────
/**
 * @brief OpenMP specialisation: the deferred-op flush.
 * @copydetails ParallelSupport2
 */
template <typename Derived>
struct ParallelSupport2<Derived, true> : public ParallelSupportCommon<Derived>
{
    // flush_pending_deferred_ops() and self() come from ParallelSupportCommon.
};

// ── Serial build: nothing to parallelise ─────────────────────────────────────
/**
 * @brief Serial specialisation: the same verb, inert.
 */
template <typename Derived>
struct ParallelSupport2<Derived, false> : public ParallelSupportCommonSerial<Derived>
{
};

/**
 * @brief ArrayChunk2's fast-block compression pass, present only in that mode.
 *
 * @tparam Derived the concrete array type (only ArrayChunk2 today).
 * @tparam Enabled defaulted to chunk_config::use_fast_block_compression; never
 *         pass it explicitly outside of tests.
 *
 * @details Not virtual: the only call site (ArrayChunk2Base::resize_common)
 *          goes through CRTP, and no type-erased pass drives live arrays.
 */
template <typename Derived, bool Enabled = chunk_config::use_fast_block_compression>
class FastBlockSupport2;

// ── Fast-block build: the real pass ──────────────────────────────────────────
/**
 * @brief Fast-block specialisation: the deferred-candidate compression pass.
 * @copydetails FastBlockSupport2
 */
template <typename Derived>
class FastBlockSupport2<Derived, true>
{
  public:
    using Traits     = ArrayChunkTraits<Derived>; ///< Traits specialisation of the concrete array.
    using chunk_type = Traits::chunk_type;        ///< Chunk type held in the array's chunk_list.

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
 * @details Kept (rather than removed) so that ArrayChunk2Base::resize_common,
 *          which guards the call with the same condition, still names a
 *          declared member in every mode.
 */
template <typename Derived>
class FastBlockSupport2<Derived, false>
{
  public:
    /** @brief No-op: this mode has no deferred candidate list. */
    void make_block_compression() noexcept {}
};

} // namespace shrink_array
