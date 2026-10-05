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

// Out-of-line bodies of v1's re-chunk hooks (RechunkHooks<Derived, true>).
// Included at the end of array_chunk.h, after the class, so the CRTP self()
// down-cast sees a complete Derived and so the manager include stays out of the
// container declaration itself.
#include <tuple>   // std::get on candidate tuples
#include <utility> // std::swap

#include "chunk/internal/array_chunk_rechunk_hooks.h" // RechunkHooks, the class these bodies belong to
#include "chunk/internal/manager_chunk.h"             // manager_type — candidate store + reuse free-lists
#include "chunk/config.h"                       // build_config::use_openmp, chunk_config::use_fast_block_compression
#include "chunk/PMP_CHUNK.h"                    // get_thr_rank_fast
#include "chunk/internal/chunk_size_policy.h"   // set_thread_chunk_size
#include "chunk/internal/block_pass_precheck.h" // check_no_flush_outside_start

namespace shrink_array
{

template <typename Derived>
void RechunkHooks<Derived, true>::rechunk_drop_candidates()
{
    // COMPRESS them first (Sept '26) -- see RechunkHooks2A: a candidate dropped
    // here stayed materialised through the rebuild and was never re-queued.
    // v1's compress() guards the already-compressed chunk (the zero sentinel).
    if constexpr (chunk_config::use_fast_block_compression)
    {
        // Bypasses CompressionManager::start(): see check_no_flush_outside_start.
        check_no_flush_outside_start<proxy_type>("RechunkHooks::rechunk_drop_candidates");
        const int num_tab = proxy_type::share_info_array.get_num_tab(self().registry_key());
        const int thr     = get_thr_rank_fast();
        shrink_array::set_thread_chunk_size(self().size_policy_.size());
        const auto compress_list = [&](auto& list)
        {
            for (auto& tup : list)
            {
                if (std::get<2>(tup) == num_tab)
                {
                    std::get<0>(tup)->compress(thr, std::get<3>(tup).count());
                }
            }
        };
        if constexpr (build_config::use_openmp)
        {
            for (auto& list : manager_type::chunk_to_compress_potentially_thr)
            {
                compress_list(list);
            }
        }
        else
        {
            compress_list(manager_type::chunk_to_compress_potentially);
        }
        manager_type::remove_from_compress_list_chunk_by_num_tab(num_tab);
    }
}

template <typename Derived>
void RechunkHooks<Derived, true>::rechunk_drain_reuse_list(unsigned int new_size)
{
    manager_type::drain_reuse_list_for_size(new_size);
}

template <typename Derived>
bool RechunkHooks<Derived, true>::rechunk_swap_identity(Derived& tmp)
{
    // v1 sets num_tab in pre_create_hook and reads it in create()/rebuild_storage,
    // so swap the member (kept consistent with the registry swap_keys below).
    using std::swap;
    swap(self().num_tab, tmp.num_tab);
    return proxy_type::share_info_array.swap_keys(self().registry_key(), tmp.registry_key());
}

} // namespace shrink_array
