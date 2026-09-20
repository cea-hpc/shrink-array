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
 * @file array_chunk2A_parallel.h
 * @brief The OpenMP-only surface of ArrayChunk2A — in-parallel teardown and
 *        the per-thread async-reset registries — isolated in its own mixin so
 *        it disappears entirely from a serial build.
 *
 * @details Same split as ParallelSupport / ParallelSupport2 do for the other two
 *          variants:
 *          - clear_in_par() — the in-parallel teardown. A serial build has no
 *            team to split the chunk list across; the body was already behind
 *            `if constexpr (build_config::use_openmp)` with clear() as the else
 *            branch.
 *          - get_list_chunk_list() / get_list_nb_chunk() and their two static
 *            registries — the async_reset_list_size path parks chunk_list and
 *            nb_chunk per thread for a later reset. Present only in the
 *            ParallelSupport2A<Derived, true> specialisation, so a serial build
 *            never sees them. *
 *          move_list_block_free_tmp() and flush_pending_deferred_ops(), the two
 *          other OpenMP-only verbs an ArrayChunk2A answers to, are NOT here:
 *          they live in ArrayChunk2Base, shared with ArrayChunk2. Now that all
 *          three variants are split, those two and the three copies of the
 *          registries can collapse into one mixin under ArrayChunkBase.
 *
 * @see array_chunk2A_parallel_inline.h for the body
 * @see array_chunk_parallel.h, array_chunk2_parallel.h for the other two variants
 */

#include <vector>
#include <cstdlib> // std::abort

#include "chunk/config.h"
#include "chunk/chunk_traits.h"
#include "chunk/internal/array_chunk_parallel_common.h"

namespace shrink_array
{

/**
 * @brief The OpenMP-only members of ArrayChunk2A, present only when the build has OpenMP.
 *
 * @tparam Derived the concrete array type (only ArrayChunk2A today).
 * @tparam Enabled defaulted to build_config::use_openmp; never pass it
 *         explicitly outside of tests.
 */
template <typename Derived, bool Enabled = build_config::use_openmp>
class ParallelSupport2A;

// ── OpenMP build: the real parallel surface ──────────────────────────────────
/**
 * @brief OpenMP specialisation: the teardown, the registries and the stub.
 * @copydetails ParallelSupport2A
 */
template <typename Derived>
class ParallelSupport2A<Derived, true> : public ParallelSupportCommon<Derived>
{
  public:
    using Common       = ParallelSupportCommon<Derived>;
    using Traits       = Common::Traits;       ///< Traits specialisation of the concrete array.
    using value_type   = Common::value_type;   ///< Stored element type.
    using chunk_type   = Common::chunk_type;   ///< Chunk type held in the array's chunk_list.
    using manager_type = Common::manager_type; ///< Per-type manager owning the free-lists.
    using entry_type =
        Common::ChunkIndexEntry;           ///< Collected entry: the chunk AND its index, needed for the per-tab table.
    using proxy_type = Traits::proxy_type; ///< Proxy type owning the per-tab metadata tables.

    /**
     * @brief In-parallel teardown: give every materialised chunk back to the pool.
     *
     * @details Called from inside a parallel region. Each thread takes its share
     *          of the chunk list (`omp for ... nowait`), zeroes the buffers it
     *          owns, parks them in the free-list of THIS array's geometry and
     *          turns the chunk back into a constant one. What happens to the
     *          array's size afterwards depends on @p opt_chunk_list_size.
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
 * @details The registries and the stub are gone rather than left empty — they
 *          were already compiled out in a serial build. clear_in_par() keeps the
 *          fallback the else branch had: a plain clear().
 */
template <typename Derived>
class ParallelSupport2A<Derived, false> : public ParallelSupportCommonSerial<Derived>
{
  public:
    // clear_in_par() and self() come from ParallelSupportCommonSerial<Derived>.
};

} // namespace shrink_array
