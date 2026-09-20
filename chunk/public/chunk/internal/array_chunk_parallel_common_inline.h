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

// Body of the shared in-parallel teardown. Split from
// array_chunk_parallel_common.h so the declaration stays light enough to be
// included next to the container declarations, while the heavy includes below
// (thread ranks, stats) only reach the three *_parallel_inline.h files.
#include <vector>

#include "chunk/internal/array_chunk_parallel_common.h"
#include "chunk/chunk_def.h"           // ChunkDef::reset_list_size / async_reset_list_size
#include "chunk/PMP_CHUNK.h"           // CHUNK_PMP_THREAD_RANK, get_size_max_tab_omp
#include "chunk/internal/chunk_stats.h"         // increm_chunk_teardown_clear (CHUNK_STATS)
#include "chunk/internal/compression_manager.h" // CompressionManager<> in flush_pending_deferred_ops

namespace shrink_array
{

template <typename Derived>
void ParallelSupportCommon<Derived>::flush_pending_deferred_ops(bool force)
{
    if constexpr (chunk_config::use_fast_block_compression)
    {
        Derived& arr = self();

        // Name CompressionManager<value_type, chunk_type> directly rather than via
        // chunk_type::CompressionManagerType: that alias is public on Chunk2 but
        // private on Chunk2A. Both name the same type; this form avoids the
        // access check on the alias.
        using CM = CompressionManager<value_type, chunk_type>;
        // force: flush every chunk regardless of the pending counter and dirty
        // bits. Used by rechunk_to() before it swaps chunk_list out -- an op
        // left on an old-geometry chunk would otherwise outlive its chunk and be
        // replayed against a recycled address of the wrong size. The counter has
        // been seen to read 0 while ops were still queued, so it cannot gate
        // this path.
        if (force)
        {
            for (int j = 0; j < arr.nb_chunk; ++j)
                static_cast<void>(CM::flush_chunk_all_ranks(&arr.chunk_list[j]));
        }
        else if (CM::any_pending_ops())
        {
            for (int j = 0; j < arr.nb_chunk; ++j)
            {
                if (arr.is_chunk_dirty(j))
                    static_cast<void>(CM::flush_chunk_all_ranks(&arr.chunk_list[j]));
            }
        }
    }
}

template <typename Derived>
template <typename Entry, typename SelectFn, typename ParkFn>
void ParallelSupportCommon<Derived>::clear_in_par_impl(int opt_chunk_list_size, SelectFn&& select, ParkFn&& park_all)
{
    Derived& arr = self();

    if (!arr.nb_chunk)
        return;

    const int nb_chunk_total = arr.nb_chunk; // loop bound; arr.nb_chunk is reset below

    const int nb_chunk0 = get_size_max_tab_omp(nb_chunk_total);

    // reserve(), NOT a sized construction: the loop below appends with
    // emplace_back, so sizing the vector first left nb_chunk0 empty entries in
    // front of the real ones, which the park step then dereferenced.
    std::vector<Entry> ll_chunk_loc;
    ll_chunk_loc.reserve(nb_chunk0);

#pragma omp for schedule(static) nowait
    for (int j = 0; j < nb_chunk_total; ++j)
    {
        if (select(arr.chunk_list[j]))
        {
            if constexpr (chunk_config::stats)
            {
                increm_chunk_teardown_clear<chunk_type>();
            }

            ll_chunk_loc.emplace_back(&arr.chunk_list[j], j);
        }
    }

    const int thr_rank = CHUNK_PMP_THREAD_RANK;

    const int nb = static_cast<int>(ll_chunk_loc.size());

    arr.move_list_block_free_tmp(thr_rank, nb);

    if (nb)
    {
        park_all(ll_chunk_loc, thr_rank);
    }

    if (opt_chunk_list_size == ChunkDef::reset_list_size)
    {
#pragma omp barrier

#pragma omp atomic write
        arr.m_size = 0;

#pragma omp atomic write
        arr.nb_chunk = 0;
    }
    else if (opt_chunk_list_size == ChunkDef::async_reset_list_size)
    {
        auto& lbf0 = get_list_chunk_list(thr_rank);

        lbf0.push_back(&arr.chunk_list);

        auto& lbf1 = get_list_nb_chunk(thr_rank);

        lbf1.push_back(&arr.nb_chunk);
    }
    else
    {
#pragma omp barrier

#pragma omp atomic write
        arr.nb_chunk = 0;
    }
}

} // namespace shrink_array
