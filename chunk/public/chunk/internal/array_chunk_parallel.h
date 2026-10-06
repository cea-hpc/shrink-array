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
 * @file array_chunk_parallel.h
 * @brief ArrayChunk's (v1) two mode-gated surfaces -- the OpenMP-only one (the
 *        deferred-op flush) and the fast-block-only one
 *        (make_block_compression) -- isolated in their own mixins so they
 *        disappear entirely from a build that has neither.
 *
 * @details ParallelSupport<Derived, use_openmp> carries
 *          flush_pending_deferred_ops(), which drains the deferred writes
 *          queued on this array's chunks (ParallelSupportCommon, shared with the
 *          two other variants); the serial specialisation is an empty class
 *          whose flush is a no-op. FastBlockSupport<Derived,
 *          use_fast_block_compression> carries the fast-block compression pass,
 *          whose gate is stricter than use_openmp.
 *
 *          The mixin also held clear_in_par(), an in-parallel teardown nothing
 *          called, with its per-thread "async reset" registries; both are gone.
 *
 * @see array_chunk_parallel_inline.h for the body
 * @see array_chunk_parallel_common.h for the shared part
 */

#include "chunk/config.h"       // build_config::use_openmp, chunk_config::use_fast_block_compression
#include "chunk/chunk_traits.h" // ArrayChunkTraits
#include "chunk/internal/array_chunk_parallel_common.h" // ParallelSupportCommon, ParallelSupportCommonSerial
#include "chunk/internal/crp_helper.h"                  // crtp_cast_t

namespace shrink_array
{

/**
 * @brief The OpenMP-only members of an array, present only when the build has OpenMP.
 *
 * @tparam Derived the concrete array type (only ArrayChunk today).
 * @tparam Enabled defaulted to build_config::use_openmp; never pass it
 *         explicitly outside of tests — the point of the parameter is that the
 *         two specialisations below are selected by the build mode.
 */
template <typename Derived, bool Enabled = build_config::use_openmp>
struct ParallelSupport;

// ── OpenMP build: the real parallel surface ──────────────────────────────────
/**
 * @brief OpenMP specialisation: the deferred-op flush.
 * @copydetails ParallelSupport
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
 * @brief ArrayChunk's fast-block compression pass, present only in that mode.
 *
 * @tparam Derived the concrete array type (only ArrayChunk today).
 * @tparam Enabled defaulted to chunk_config::use_fast_block_compression; never
 *         pass it explicitly outside of tests.
 *
 * @details v1's copy of FastBlockSupport2 (array_chunk2_parallel.h). The only
 *          call site (ArrayChunk::resize) guards with the same constexpr and
 *          so still names a declared member in every mode.
 */
template <typename Derived, bool Enabled = chunk_config::use_fast_block_compression>
class FastBlockSupport;

// ── Fast-block build: the real pass ──────────────────────────────────────────
/**
 * @brief Fast-block specialisation: the deferred-candidate compression pass.
 * @copydetails FastBlockSupport
 */
template <typename Derived>
class FastBlockSupport<Derived, true>
{
  public:
    using Traits     = ArrayChunkTraits<Derived>; ///< Traits specialisation of the concrete array.
    using chunk_type = Traits::chunk_type;        ///< Chunk type held in the array's chunk_list.

    /**
     * @brief Compress every chunk of this array currently queued as a candidate.
     * @details Drains the calling thread's candidate list, or every thread's
     *          when called from outside a parallel region.
     */
    void make_block_compression(this Derived& self);

};

// ── Any other mode: no candidate queue to drain ──────────────────────────────
/**
 * @brief Inert specialisation: nothing queues block-compression candidates.
 */
template <typename Derived>
class FastBlockSupport<Derived, false>
{
  public:
    /** @brief No-op: this mode has no deferred candidate list. */
    void make_block_compression() noexcept {}
};

} // namespace shrink_array
