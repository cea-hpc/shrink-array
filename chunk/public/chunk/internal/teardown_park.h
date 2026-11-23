/*
 * Copyright 2026 CEA (www.cea.fr)
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
 * @file teardown_park.h
 * @brief The park step of a threaded teardown (clear_core() of zero::Array,
 *        zero_or_value::Array and zero_any::ArrayImpl): the team parks the
 *        buffers of the array's live chunks into the reuse lists.
 */

#pragma once

#include <memory>      // std::uninitialized_fill_n
#include <type_traits> // std::remove_reference_t
#include <vector>      // std::vector

#include "chunk/internal/pmp_chunk.h"   // pmp::Team, pmp::nowait, get_size_max_array_omp, CHUNK_PMP_THREAD_RANK
#include "chunk/internal/chunk_stats.h" // inc_chunk_teardown_clear, add_chunk_released
#include "chunk/internal/pool_ledger.h" // CHUNK_LEDGER_PARK
#include "chunk/config.h"               // chunk_config::stats

namespace shrink_array::internal
{

/// @brief What park_live_chunks() leaves in a buffer it parks.
enum class ParkedBuffer
{
    as_is,      ///< the chunk's values, untouched
    zero_filled ///< every element T{} (zero::Array)
};

/**
 * @brief Run by every thread of @p team: park the buffers of the live chunks of
 *        a static share of @p chunk_list into this rank's reuse list, then
 *        neutralise those chunks.
 * @details The list is the one of the ARRAY's geometry (@p chunk_size), not the
 *          ambient one, which may be another size mid-rechunk: a size-A buffer
 *          filed under list[B] would be popped and memset at B later, and
 *          overflow. Each buffer goes LIVE -> parked in the ledger (or a later
 *          pop of it is an illegal b_pop). Its chunk is neutralised right after:
 *          still materialised on the parked buffer, a residual compression
 *          candidate would free the SAME buffer again (a double free under
 *          std-alloc); neutralised, every compress path short-circuits.
 *          The share is a for_static(..., nowait): no barrier at the end.
 * @tparam T the element type.
 * @tparam fill what the parked buffer holds (ParkedBuffer).
 * @param team the team of the teardown region.
 * @param chunk_list the array's chunks.
 * @param chunk_count the number of live entries of @p chunk_list.
 * @param chunk_size the array's chunk size (elements per buffer).
 * @param is_live `bool(const Chunk&)`: the chunk owns a buffer to park.
 * @param neutralise `void(Chunk&)`: leave the chunk constant, owning nothing.
 */
template <typename T, ParkedBuffer fill = ParkedBuffer::as_is, typename ChunkList, typename IsLive, typename Neutralise>
void park_live_chunks(pmp::Team&   team,
                      ChunkList&   chunk_list,
                      int          chunk_count,
                      unsigned int chunk_size,
                      IsLive&&     is_live,
                      Neutralise&& neutralise)
{
    using Chunk = std::remove_reference_t<decltype(chunk_list[0])>;

    std::vector<Chunk*> live;
    live.reserve(get_size_max_array_omp(chunk_count));

    team.for_static<int>(
        0, chunk_count,
        [&](int j)
        {
            if (is_live(chunk_list[j]))
            {
                if constexpr (chunk_config::stats)
                {
                    inc_chunk_teardown_clear<Chunk>();
                    // Released below without a compression: out of the live figures.
                    add_chunk_released<Chunk>(1, static_cast<long>(sizeof(T)) * static_cast<long>(chunk_size));
                }
                live.push_back(&chunk_list[j]);
            }
        },
        pmp::nowait);

    if (live.empty())
    {
        return;
    }
    auto& lbf = Chunk::Manager::get_list_block_free_thr_for_size(CHUNK_PMP_THREAD_RANK, chunk_size);
    for (Chunk* chunk : live)
    {
        T* values = chunk->values_;
        if constexpr (fill == ParkedBuffer::zero_filled)
        {
            std::uninitialized_fill_n(values, chunk_size, T{});
        }
        CHUNK_LEDGER_PARK(values, "clear_core.teardown_park");
        lbf.push_back(values);
        neutralise(*chunk);
    }
}

} // namespace shrink_array::internal
