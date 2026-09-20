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

// Out-of-line bodies of ArrayChunk2A's OpenMP-only surface (ParallelSupport2A).
// Included at the end of array_chunk2A.h, after the class and its own inline
// file, so the CRTP self() down-cast sees a complete Derived and so the heavy
// includes below (manager, ledger, stats) stay out of the container declaration
// itself.
#include <memory>  // std::uninitialized_fill_n
#include <utility> // std::pair
#include <vector>

#include "chunk/internal/array_chunk2A_parallel.h"
#include "chunk/internal/array_chunk_parallel_common_inline.h" // clear_in_par_impl
#include "chunk/chunk_def.h"                          // ChunkDef::reset_list_size / async_reset_list_size
#include "chunk/PMP_CHUNK.h"                          // CHUNK_PMP_THREAD_RANK, get_size_max_tab_omp
#include "chunk/internal/manager_chunk.h"                      // manager_type::get_list_block_free_thr_for_size
#include "chunk/internal/pool_ledger.h"                        // CHUNK_LEDGER_PARK
#include "chunk/internal/chunk_stats.h"                        // increm_chunk_teardown_clear (CHUNK_STATS)

namespace shrink_array
{

template <typename Derived>
void ParallelSupport2A<Derived, true>::clear_in_par(int opt_chunk_list_size)
{
    Common::template clear_in_par_impl<entry_type>(
        opt_chunk_list_size,
        // Chunk2A tracks its second value per chunk rather than by a shared
        // index, so written() cannot report a marker as a live buffer here.
        [](const chunk_type& chunk_t) { return chunk_t.written(); },
        [this](std::vector<entry_type>& entries, int thr_rank)
        {
            Derived& arr = this->self();

            // Park into the reuse list of THIS array's geometry, not the ambient
            // t_active (same rationale as clear_core teardown): these buffers are
            // physically sized for size_policy_.size(); filing them under
            // list[t_active] lets a later pop of a different size reuse them and
            // overflow in memset.
            const unsigned int   my_sz = arr.size_policy_.size();
            auto&                lbf   = manager_type::get_list_block_free_thr_for_size(thr_rank, my_sz);
            constexpr value_type zero{};

            for (auto& entry : entries)
            {
                value_type* values_loc = entry->values_load();

                std::uninitialized_fill_n(values_loc, arr.size_policy_.size(), zero);

                // Ledger: LIVE -> PARKED. This buffer is being returned to the reuse
                // free-list (the chunk becomes constant just below). Without tracking
                // the transition, a later allocate_chunk_internal.freelist_pop trips
                // "b_pop of UNTRACKED block". Same fix as clear_core.teardown_park;
                // this terminal-compression path (clear_in_par) had the untracked
                // push, which only surfaced once buffers here were recycled (e.g.
                // all-same in std-alloc).
                CHUNK_LEDGER_PARK(values_loc, "clear_in_par.park");
                lbf.push_back(values_loc);

                char* ptr_type_cst = &proxy_type::tab_type_cst_val[arr.num_tab][entry.index];
                entry->set_cst_val_to_cst_val1(ptr_type_cst);
                entry->init_values_cst_val(ptr_type_cst, nullptr); // cst_val1: the second value is not consulted
            }
        });
}

} // namespace shrink_array
