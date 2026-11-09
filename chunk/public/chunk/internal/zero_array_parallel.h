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
 * @file zero_array_parallel.h
 * @brief zero::Array's fast-block-only surface (make_block_compression),
 *        isolated in its own mixin so that it disappears from a build without it.
 *
 * @details FastBlockSupport<Derived, use_fast_block_compression> carries the
 *          fast-block compression pass, whose gate is stricter than threaded.
 *          ParallelSupport (array_chunk_parallel_common.h) carries what all the
 *          variants share under threads.
 *
 * @see zero_array_parallel_inline.h for the body
 * @see array_chunk_parallel_common.h for the shared part
 */

#include "chunk/config.h"                               // chunk_config::use_fast_block_compression
#include "chunk/internal/chunk_traits.h"                         // ArrayChunkTraits
#include "chunk/internal/array_chunk_parallel_common.h" // ParallelSupport (zero_array.h uses it through this header)

namespace shrink_array::zero
{

/**
 * @brief zero::Array's fast-block compression pass, present only in that mode.
 *
 * @tparam Derived the concrete array type (only zero::Array today).
 * @tparam Enabled defaulted to chunk_config::use_fast_block_compression; never
 *         pass it explicitly outside of tests.
 *
 * @details zero's copy of zero_one_value::FastBlockSupport (zero_one_value_array_parallel.h). The only
 *          call site (zero::Array::resize) guards with the same constexpr and
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
    using Traits    = ArrayChunkTraits<Derived>; ///< Traits specialisation of the concrete array.
    using ChunkType = Traits::ChunkType;         ///< Chunk type held in the array's chunk_list_.

    /**
     * @brief Compress every chunk of this array currently queued as a candidate.
     * @details Drains every thread's candidate list. A sequential point: outside
     *          any parallel region only. Inside one it is refused (an invariant
     *          failure in check builds, an error in the log otherwise, and no
     *          pass): without a proxy it takes no writer handshake and no reader
     *          grace period. In a region the writers' own passes compress.
     */
    void make_block_compression();

  protected:
    /**
     * @brief The pass behind make_block_compression(), and resize()'s drain.
     * @details Outside a region, every thread's candidate list; inside one (an
     *          array resized by the thread that owns it), the calling thread's
     *          list only, its released buffers held for one reader grace period
     *          (read_reclaim::PassScope). resize() needs it there: the lists hold
     *          &chunk_list_[j] pointers its reallocation would leave dangling.
     */
    void drain_candidates();

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
 */
template <typename Derived>
class FastBlockSupport<Derived, false>
{
  public:
    /** @brief No-op: this mode has no deferred candidate list. */
    static void make_block_compression() noexcept
    {
        // Nothing to fold: the counter compresses a chunk on the write.
    }
};

} // namespace shrink_array::zero
