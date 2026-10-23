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

// Out-of-line bodies of the re-chunk hooks of zero::Array and zero_any::Array
// (RechunkHooks<Derived, true>). Included at the end of zero_array.h and of
// zero_any_array.h, after the class, so the CRTP self() down-cast sees a complete
// Derived and so the manager include stays out of the container declaration itself.
#include <tuple>   // std::get on candidate tuples
#include <utility> // std::swap

#include "chunk/internal/array_chunk_rechunk_hooks.h" // RechunkHooks, the class these bodies belong to
#include "chunk/internal/manager_chunk.h"             // ManagerType — candidate store + reuse free-lists
#include "chunk/config.h"                       // build_config::threaded, chunk_config::use_fast_block_compression
#include "chunk/PMP_CHUNK.h"                    // get_thr_rank_fast
#include "chunk/internal/chunk_size_policy.h"   // set_thread_chunk_size
#include "chunk/internal/block_pass_precheck.h" // check_no_flush_outside_start

namespace shrink_array
{

template <typename Derived>
void RechunkHooks<Derived, true>::rechunk_drop_candidates()
{
    // COMPRESS them first: a candidate dropped here would stay materialised
    // through the rebuild and never be re-queued. compress() is guarded against
    // an already-compressed chunk (zero: the zero sentinel); each entry carries its
    // own element count. See RechunkSupport::rechunk_to for why the drop happens
    // at the owning geometry.
    if constexpr (chunk_config::use_fast_block_compression)
    {
        int num_tab = 0;
        if constexpr (is_chunk2a)
        {
            num_tab = self().num_tab_; // zero_any keys its candidates by the num_tab MEMBER
        }
        else
        {
            // zero only: this pass bypasses CompressionManager::start(), and zero's
            // replay bodies rely on no compressor changing `values` during a flush
            // drain (see check_no_flush_outside_start, zero_proxy_inline.h).
            check_no_flush_outside_start<ProxyType>("RechunkHooks::rechunk_drop_candidates");
            num_tab = ProxyType::share_info_array->get_num_tab(self().registry_key());
        }
        const int thr = get_thr_rank_fast();
        shrink_array::set_thread_chunk_size(self().size_policy_.size());
        const auto compress_list = [&](auto& list)
        {
            for (auto& tup : list)
            {
                if (std::get<2>(tup) == num_tab)
                {
                    if constexpr (is_chunk2a)
                    {
                        std::get<0>(tup)->compress(thr, std::get<1>(tup), num_tab, /*apply=*/true, std::get<3>(tup));
                    }
                    else
                    {
                        std::get<0>(tup)->compress(thr, std::get<3>(tup).count());
                    }
                }
            }
        };
        if constexpr (build_config::threaded)
        {
            for (auto& list : candidate_lists_thr())
            {
                compress_list(list);
            }
        }
        else
        {
            compress_list(candidate_list());
        }
        remove_candidates_of_tab(num_tab);
    }
}

template <typename Derived>
void RechunkHooks<Derived, true>::rechunk_drain_reuse_list(unsigned int new_size)
{
    ManagerType::drain_reuse_list_for_size(new_size);
}

template <typename Derived>
bool RechunkHooks<Derived, true>::rechunk_swap_identity(Derived& tmp)
{
    // num_tab is set in pre_create_hook and indexes the per-array metadata, so
    // swap the member AND make the registry agree (get_num_tab(registry_key())
    // -> the new row).
    using std::swap;
    swap(self().num_tab_, tmp.num_tab_);
    return ProxyType::share_info_array->swap_keys(self().registry_key(), tmp.registry_key());
}

} // namespace shrink_array
