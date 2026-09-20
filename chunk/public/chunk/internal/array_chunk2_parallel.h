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
 * @file array_chunk2_parallel.h
 * @brief ArrayChunk2's two mode-gated surfaces — the OpenMP-only one
 *        (in-parallel teardown, per-thread async-reset registries) and the
 *        fast-block-only one (make_block_compression) — isolated in their own
 *        mixins so they disappear entirely from a build that has neither.
 *
 * @details Same split as ParallelSupport does for ArrayChunk (v1), applied to
 *          the members that are specific to the Chunk2 variant:
 *          - clear_in_par() — the in-parallel teardown. A serial build has no
 *            team to split the chunk list across; the body was a preprocessor
 *            conditional with clear() as the other branch, and is now the
 *            ParallelSupport2<Derived, true> specialisation.
 *          - get_list_chunk_list() / get_list_nb_chunk() and their two static
 *            registries — the async_reset_list_size path parks chunk_list and
 *            nb_chunk per thread for a later reset. Present only in the
 *            ParallelSupport2<Derived, true> specialisation, so a serial build
 *            never sees them.
 *
 *          make_block_compression() gets its own mixin rather than joining
 *          the first one: its gate is chunk_config::use_fast_block_compression,
 *          which is STRICTER than use_openmp (it also needs embedded — i.e. not
 *          manual — compression).
 *
 *          The two other OpenMP-only verbs an ArrayChunk2 answers to —
 *          move_list_block_free_tmp() and flush_pending_deferred_ops() — are NOT
 *          here: they live in ArrayChunk2Base, shared with ArrayChunk2A, and
 *          moving them would move them for 2A too. Once 2A gets the same
 *          treatment, those two and the three copies of the registries (v1, 2,
 *          2A) can collapse into a single mixin under ArrayChunkBase.
 *
 * @see array_chunk2_parallel_inline.h for the body
 * @see array_chunk_parallel.h for v1's version of the same split
 */

#include <vector>

#include "chunk/config.h"
#include "chunk/chunk_traits.h"
#include "chunk/internal/array_chunk_parallel_common.h"

namespace shrink_array
{

/**
 * @brief The OpenMP-only members of ArrayChunk2, present only when the build has OpenMP.
 *
 * @tparam Derived the concrete array type (only ArrayChunk2 today).
 * @tparam Enabled defaulted to build_config::use_openmp; never pass it
 *         explicitly outside of tests.
 */
template <typename Derived, bool Enabled = build_config::use_openmp>
class ParallelSupport2;

// ── OpenMP build: the real parallel surface ──────────────────────────────────
/**
 * @brief OpenMP specialisation: the teardown and the registries.
 * @copydetails ParallelSupport2
 */
template <typename Derived>
class ParallelSupport2<Derived, true> : public ParallelSupportCommon<Derived>
{
  public:
    using Common       = ParallelSupportCommon<Derived>;
    using Traits       = Common::Traits;       ///< Traits specialisation of the concrete array.
    using value_type   = Common::value_type;   ///< Stored element type.
    using chunk_type   = Common::chunk_type;   ///< Chunk type held in the array's chunk_list.
    using manager_type = Common::manager_type; ///< Per-type manager owning the free-lists.
    using entry_type   = Common::ChunkEntry;   ///< Collected entry: the chunk alone (the index is unused here).

    /**
     * @brief In-parallel teardown: give every materialised chunk back to the pool.
     *
     * @details Called from inside a parallel region. Each thread takes its share
     *          of the chunk list (`omp for ... nowait`), zeroes the buffers it
     *          owns, parks them in its own free-list and turns the chunk back
     *          into a constant one. What happens to the array's size afterwards
     *          depends on @p opt_chunk_list_size.
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
 * @brief Serial specialisation: the same verb, and no registries at all.
 *
 * @details The registries are gone rather than left empty — they were already
 *          compiled out in a serial build, and their only reader
 *          (clear_in_par's async_reset_list_size branch) is gone with them.
 *          clear_in_par() falls back to a plain clear().
 */
template <typename Derived>
class ParallelSupport2<Derived, false> : public ParallelSupportCommonSerial<Derived>
{
  public:
    // clear_in_par() and self() come from ParallelSupportCommonSerial<Derived>.
};

/**
 * @brief ArrayChunk2's fast-block compression pass, present only in that mode.
 *
 * @tparam Derived the concrete array type (only ArrayChunk2 today).
 * @tparam Enabled defaulted to chunk_config::use_fast_block_compression; never
 *         pass it explicitly outside of tests.
 *
 * @details Used to be a virtual override of ArrayChunk2Int, the type-erased
 *          handle that let a global pass drive every live array through one
 *          virtual call. That pass was never written — the registry of live
 *          arrays was filled and emptied but never walked — so the virtual was
 *          removed along with it, and the only real call site
 *          (ArrayChunk2Base::resize_common) has always gone through CRTP.
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
