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

// Out-of-line bodies of ArrayChunk2's OpenMP-only surface (ParallelSupport2).
// Included at the end of array_chunk2.h, after the class and its own inline
// file, so the CRTP self() down-cast sees a complete Derived and so the heavy
// includes below (manager, stats) stay out of the container declaration itself.
#include <memory> // std::uninitialized_fill_n
#include <vector>

#include "chunk/internal/array_chunk2_parallel.h"
#include "chunk/internal/array_chunk_parallel_common_inline.h" // clear_in_par_impl
#include "chunk/chunk_def.h"                          // ChunkDef::reset_list_size / async_reset_list_size
#include "chunk/PMP_CHUNK.h"                          // CHUNK_PMP_THREAD_RANK / MAX_THREADS, get_size_max_tab_omp
#include "chunk/internal/manager_chunk.h"                      // manager_type::get_list_block_free_thr_for_size
#include "chunk/internal/chunk_stats.h"                        // increm_chunk_teardown_clear (CHUNK_STATS)
#include "chunk/internal/chunk_size_policy.h"                  // set_thread_chunk_size in make_block_compression

namespace shrink_array
{

template <typename Derived>
void ParallelSupport2<Derived, true>::clear_in_par(int opt_chunk_list_size)
{
    Common::template clear_in_par_impl<entry_type>(
        opt_chunk_list_size,
        // owns_live_buffer(): index-agnostic. written(single-index) here also fed
        // the uninitialized_fill_n below, which on a marker-valued chunk
        // (compressed at another index) zeroed the STATIC cst_val2 array AND then
        // parked the marker -> corruption + b_park of UNTRACKED.
        [](const chunk_type& chunk_t) { return chunk_t.owns_live_buffer(); },
        [this](std::vector<entry_type>& entries, int thr_rank)
        {
            Derived& arr = this->self();

            // Park into the reuse list of THIS array's geometry, never the
            // caller's ambient one (same as ArrayChunk::clear_core and
            // ParallelSupport2A::clear_in_par). These buffers are physically
            // sized for size_policy_.size(); the ambient list is keyed by
            // whatever thread_chunk_size() the parking thread last had, which
            // after a rechunk_to() or an OMPT drain need not be this array's.
            const unsigned int   my_sz = arr.size_policy_.size();
            auto&                lbf   = manager_type::get_list_block_free_thr_for_size(thr_rank, my_sz);
            constexpr value_type zero{};

            for (auto& chunk_t : entries)
            {
                value_type* values_loc = chunk_t->values_load();

                std::uninitialized_fill_n(values_loc, arr.size_policy_.size(), zero);
                lbf.push_back(values_loc);

                if constexpr (Derived::is_not_bool)
                {
                    chunk_t->init_values_cst_val1();
                }
                else
                {
                    chunk_t->init_values_cst_val_bool(0);
                }
            }
        });
}

template <typename Derived>
void FastBlockSupport2<Derived, true>::make_block_compression()
{
    // bug in run time openmp llvm

    if constexpr (build_config::chunk_dynamic)
    {
        // The compressibility scan reads the active size; make it THIS array's
        // size regardless of whatever size the calling thread last operated on.
        shrink_array::set_thread_chunk_size(self().chunk_size());
    }

    if (!CHUNK_PMP_IN_PARALLEL_REGION)
    {
        for (int thr = 0; thr < CHUNK_PMP_MAX_THREADS; ++thr)
        {
            chunk_type::make_block_compression(thr, self().get_info_second_value(), nullptr);
        }
    }
    else
    {
        chunk_type::make_block_compression(CHUNK_PMP_THREAD_RANK, self().get_info_second_value(), nullptr);
    }
}

} // namespace shrink_array
