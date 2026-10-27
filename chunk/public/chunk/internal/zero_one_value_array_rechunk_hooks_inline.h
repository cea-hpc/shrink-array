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

// Bodies of zero_one_value::RechunkHooks<Derived, true>. Included at the end of zero_one_value_array.h:
// self() needs a complete Derived, and the manager include stays out of the class.
#include <tuple> // std::get on candidate tuples

#include "chunk/internal/zero_one_value_array_rechunk_hooks.h" // zero_one_value::RechunkHooks, the class these bodies belong to
#include "chunk/internal/manager_chunk.h"                      // ManagerType — candidate store + reuse free-lists
#include "chunk/config.h"                     // build_config::threaded, chunk_config::use_fast_block_compression
#include "chunk/PMP_CHUNK.h"                  // get_thr_rank_fast
#include "chunk/internal/chunk_size_policy.h" // set_thread_chunk_size
#include "chunk/internal/in_t.h"              // InT<value_type>: the captured state

namespace shrink_array
{

template <typename Derived>
void zero_one_value::RechunkHooks<Derived, true>::rechunk_drop_candidates(this Derived& self)
{
    // num_tab from the registry, never the member. The drop happens at the
    // owning geometry (see RechunkSupport::rechunk_to). COMPRESS the candidates
    // first (see RechunkHooks): a dropped one would stay materialised and never
    // be re-queued. bool has no second value and is only dropped.
    if constexpr (chunk_config::use_fast_block_compression)
    {
        const int num_tab = ProxyType::share_info_array->get_num_tab(self.registry_key());
        if constexpr (Derived::is_not_bool && build_config::threaded)
        {
            const auto inf = self.get_info_second_value();
            const int  thr = get_thr_rank_fast();
            shrink_array::set_thread_chunk_size(self.size_policy_.size());
            for (const auto& per_rank : ManagerType::map_chunk2_to_compress_potentially_thr)
            {
                for (const auto& kv : per_rank)
                {
                    if (!kv.second)
                    {
                        continue;
                    }
                    for (auto& tup : *kv.second)
                    {
                        if (std::get<2>(tup) == num_tab)
                        {
                            std::get<0>(tup)->compress(thr, inf, /*apply=*/true, std::get<3>(tup).count());
                        }
                    }
                }
            }
        }
        ManagerType::remove_from_compress_list_chunk2_by_num_tab(num_tab);
    }
}

template <typename Derived>
zero_one_value::RechunkHooks<Derived, true>::value_type
zero_one_value::RechunkHooks<Derived, true>::rechunk_capture_state(this const Derived& self)
{
    // The array-level second value; bool's is re-established by create().
    if constexpr (Derived::is_not_bool)
    {
        return self.get_second_value();
    }
    else
    {
        return value_type{};
    }
}

template <typename Derived>
void zero_one_value::RechunkHooks<Derived, true>::rechunk_seed_fresh(this Derived& self, InT<value_type> state)
{
    // On the FRESH array before refill. Not value_type{}: that is already its
    // state, and re-declaring it would make a degenerate cst_val2 == cst_val1 buffer.
    if constexpr (Derived::is_not_bool)
    {
        if (state != value_type{})
        {
            self.set_second_value(state, false);
        }
    }
    else
    {
        (void)state;
    }
}

template <typename Derived>
bool zero_one_value::RechunkHooks<Derived, true>::rechunk_swap_identity(this Derived& self, Derived& tmp)
{
    // The num_tab, ind_cst_val2 and inf_array MEMBERS are NOT swapped: the
    // identity comes from share_info_array, and the members may be
    // uninitialized (swapping would read indeterminate values). The mirrors are
    // re-read in rechunk_reown_after_swap().
    return ProxyType::share_info_array->swap_entries(self.registry_key(), tmp.registry_key());
}

template <typename Derived>
void zero_one_value::RechunkHooks<Derived, true>::rechunk_reown_after_swap(this Derived&   self,
                                                                           Derived&        tmp,
                                                                           InT<value_type> state)
{
    // Re-own the registration of `state`: map_inf_array[state] points at
    // &tmp.inf_array, which dangles once the stack tmp dies, and the next
    // compression of `state` would dereference it. Copy tmp's (value, ind) and
    // re-point the map at this array's member. Only when the fresh array
    // registered a value (state != value_type{}).
    if constexpr (chunk_config::use_fast_block_compression)
    {
        if constexpr (Derived::is_not_bool)
        {
            if (state != value_type{})
            {
                self.inf_array_    = tmp.inf_array_; // tmp's is valid (set_second_value ran)
                self.ind_cst_val2_ = tmp.ind_cst_val2_;
                ProxyType::share_info_array->insert(state, &self.inf_array_);
            }
        }
    }
    else
    {
        (void)state;
    }

    // Write proxies are armed with the inf_array member's address: re-read both
    // mirrors from the swapped registry (tmp's teardown uses it too). All build
    // modes, bool included.
    self.sync_registry_mirror();
    tmp.sync_registry_mirror();
}

} // namespace shrink_array
