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

// Out-of-line bodies of zero::Array's FastBlockSupport. Included at the end of zero_array.h, after the class and
// its own inline file, so the explicit object parameter sees a complete Derived and
// so the compression manager stays out of the container declaration itself.
#include <cstddef> // std::size_t

#include "chunk/internal/zero_array_parallel.h"                // FastBlockSupport, the class these bodies belong to
#include "chunk/internal/array_chunk_parallel_common_inline.h" // flush_pending_deferred_ops
#include "chunk/internal/PMP_CHUNK.h"                          // CHUNK_PMP_* (make_block_compression)
#include "chunk/internal/chunk_size_policy.h"                  // set_thread_chunk_size (make_block_compression)
#include "chunk/internal/read_reclaim.h"                       // read_reclaim::PassScope (drain_own_candidates)
#include "chunk/internal/own_candidates.h"                     // with_own_candidates_only (drain_own_candidates)
#include "chunk/config.h"                                      // build_config::chunk_dynamic

namespace shrink_array
{

template <typename Derived>
void zero::FastBlockSupport<Derived, true>::make_block_compression(this Derived& self)
{
    if (!block_pass_refused_in_region())
    {
        self.drain_candidates();
    }
}

template <typename Derived>
void zero::FastBlockSupport<Derived, true>::drain_candidates(this Derived& self)
{
    if constexpr (build_config::chunk_dynamic)
    {
        // The compressibility scan reads the active size; make it THIS array's
        // size regardless of whatever size the calling thread last operated on.
        shrink_array::set_thread_chunk_size(self.chunk_size());
    }

    // Do not open an `omp parallel` region here (it triggers an LLVM OpenMP
    // runtime bug): outside a region the caller drains every rank's list
    // serially, inside one each thread drains its own.
    if (!CHUNK_PMP_IN_PARALLEL_REGION)
    {
        for (int thr = 0; thr < CHUNK_PMP_RANK_BOUND; ++thr)
        {
            ChunkType::make_block_compression(thr, nullptr);
        }
    }
    else
    {
        self.drain_own_candidates();
    }
}

template <typename Derived>
void zero::FastBlockSupport<Derived, true>::drain_own_candidates(this Derived& self)
{
    using ProxyT   = Traits::ProxyType;
    using ManagerT = Traits::ManagerType;

    // An array resized inside a region by its owning thread (resize()). The
    // pass takes no writer handshake: this array's candidates only, the
    // thread's list holding chunks of arrays other threads are writing. Other
    // threads may be reading, so the buffers released wait for a grace period
    // (read_reclaim.h).
    const int rank = CHUNK_PMP_THREAD_RANK;
    if (rank < 0 || static_cast<std::size_t>(rank) >= ManagerT::zero_candidates_thr.size())
    {
        return;
    }
    const int num_tab = ProxyT::share_info_array->get_num_tab(self.registry_key());
    internal::with_own_candidates_only(ManagerT::zero_candidates_thr[static_cast<std::size_t>(rank)], num_tab,
                                       [rank]
                                       {
                                           const read_reclaim::PassScope pass_scope;
                                           ChunkType::make_block_compression(rank, nullptr);
                                       });
}

} // namespace shrink_array
