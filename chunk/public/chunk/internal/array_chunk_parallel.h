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
 * @brief The OpenMP-only surface of ArrayChunk (v1) — in-parallel teardown, the
 *        per-thread async-reset registries and the deferred-op flush — isolated
 *        in its own mixin so it disappears entirely from a serial build.
 *
 * @details Four groups of ArrayChunk members exist only because the build has
 *          OpenMP, and each said so in its own way:
 *          - clear_in_par() — the in-parallel teardown. A serial build has no
 *            team to split the chunk list across, so the whole body already sat
 *            behind `if constexpr (build_config::use_openmp)` with clear() as
 *            the else branch.
 *          - move_list_block_free_tmp() — reserves room in the calling thread's
 *            block free-list before a teardown batch parks buffers there. There
 *            is no per-thread free-list in a serial build, so it is a no-op
 *            there.
 *          - get_list_chunk_list() / get_list_nb_chunk() and their two static
 *            registries — the async_reset_list_size path parks chunk_list and
 *            nb_chunk per thread for a later reset. Present only in the
 *            ParallelSupport<Derived, true> specialisation, so a serial build
 *            never sees them.
 *          - flush_pending_deferred_ops() — drains the deferred writes queued on
 *            this array's chunks. Nothing defers without OpenMP, so the body was
 *            an `if constexpr` that collapsed to nothing.
 *
 *          ParallelSupport<Derived, use_openmp> gathers the four: the OpenMP
 *          specialisation carries them unchanged, the serial one is an empty
 *          class (no member, zero bytes via the empty-base optimisation) whose
 *          verbs are no-ops the optimiser deletes — except clear_in_par(), which
 *          keeps the serial fallback it already had (Derived::clear()).
 *
 *          The mixin is CRTP-generic, but only ArrayChunk uses it: the 2/2A
 *          variants get the same members from ArrayChunk2Base, whose bodies are
 *          the same shape. Their flush_pending_deferred_ops() is in fact a
 *          byte-identical duplicate of the one here.
 *
 * @see array_chunk_parallel_inline.h for the bodies
 * @see array_chunk_dirty_bitset.h for the same split applied to the dirty bitset
 * @see array_chunk_rechunk.h for the same split applied to the re-chunk algorithm
 */

#include <vector>

#include "chunk/config.h"
#include "chunk/chunk_traits.h"
#include "chunk/internal/array_chunk_parallel_common.h"

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
class ParallelSupport;

// ── OpenMP build: the real parallel surface ──────────────────────────────────
/**
 * @brief OpenMP specialisation: the teardown, the registries and the flush.
 * @copydetails ParallelSupport
 */
template <typename Derived>
class ParallelSupport<Derived, true> : public ParallelSupportCommon<Derived>
{
  public:
    using Common       = ParallelSupportCommon<Derived>;
    using Traits       = Common::Traits;       ///< Traits specialisation of the concrete array.
    using value_type   = Common::value_type;   ///< Stored element type.
    using chunk_type   = Common::chunk_type;   ///< Chunk type held in the array's chunk_list.
    using manager_type = Common::manager_type; ///< Per-type manager owning the free-lists.
    using entry_type   = Common::ChunkEntry;   ///< Collected entry: the chunk alone (the index is unused here).

    /**
     * @brief Reserve room in thread @p thr_rank's block free-list for @p nb buffers.
     * @param thr_rank the calling thread's rank
     * @param nb number of buffers about to be parked there
     * @details Growth is amortised (5*nb) because a teardown pass parks its
     *          whole batch one buffer at a time.
     */

    /**
     * @brief In-parallel teardown: compress every written chunk back to constant.
     *
     * @details Called from inside a parallel region. Each thread takes its share
     *          of the chunk list (`omp for ... nowait`), zeroes the buffers it
     *          owns and parks them in its own free-list. What happens to the
     *          array's size/geometry afterwards depends on @p opt_chunk_list_size.
     *
     * @param opt_chunk_list_size one of ChunkDef::reset_list_size (barrier, then
     *        reset m_size and nb_chunk), ChunkDef::async_reset_list_size (park
     *        chunk_list / nb_chunk in the per-thread registries below for a later
     *        reset, no barrier), or anything else (barrier, then reset nb_chunk).
     */
    void clear_in_par(int opt_chunk_list_size);

    // The async-reset registries, the CRTP self() down-cast and the shared body
    // of clear_in_par() all come from ParallelSupportCommon<Derived>. The
    // registries stay per-variant: the base is templated on Derived, so each
    // array type instantiates its own statics, exactly as when every mixin
    // declared its own.
};

// ── Serial build: nothing to parallelise ─────────────────────────────────────
/**
 * @brief Serial specialisation: the same verbs, inert, and no registries at all.
 *
 * @details The two registries are gone rather than left empty — they were
 *          already compiled out in a serial build, and their only reader
 *          (clear_in_par's async_reset_list_size branch) is gone with them.
 *          clear_in_par() falls back to a plain clear().
 */
template <typename Derived>
class ParallelSupport<Derived, false> : public ParallelSupportCommonSerial<Derived>
{
  public:
    // clear_in_par(), move_list_block_free_tmp() and self() come from
    // ParallelSupportCommonSerial<Derived>.
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
 */
template <typename Derived>
class FastBlockSupport<Derived, false>
{
  public:
    /** @brief No-op: this mode has no deferred candidate list. */
    void make_block_compression() noexcept {}
};

} // namespace shrink_array
