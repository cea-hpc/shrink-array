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

// Out-of-line bodies of ArrayChunk2's re-chunk hooks (RechunkHooks2<Derived,
// true>). Included at the end of array_chunk2.h, after the class, so the CRTP
// self() down-cast sees a complete Derived and so the manager include stays out
// of the container declaration itself.
#include "chunk/internal/array_chunk2_rechunk_hooks.h"
#include "chunk/internal/manager_chunk.h" // manager_type — candidate store + reuse free-lists
#include <tuple>                          // std::get on candidate tuples
#include "chunk/config.h"
#include "chunk/PMP_CHUNK.h"                  // get_thr_rank_fast
#include "chunk/internal/chunk_size_policy.h" // set_thread_chunk_size

namespace shrink_array
{

template <typename Derived>
void RechunkHooks2<Derived, true>::rechunk_drop_candidates()
{
    // Chunk2 resolves num_tab from the registry (keyed by `this`), never the
    // member. See RechunkSupport::rechunk_to for why the drop must happen at the
    // owning geometry.
    //
    // COMPRESS them first (Sept '26) -- see RechunkHooks2A: a candidate dropped
    // here stayed materialised through the rebuild and was never re-queued.
    // Chunk2's compress() takes the array's (second_value, ind, num_tab) tuple;
    // bool has no second value and is left to the drop.
    if constexpr (chunk_config::use_fast_block_compression)
    {
        const int num_tab = proxy_type::share_info_array.get_num_tab(self().registry_key());
        if constexpr (Derived::is_not_bool && build_config::use_openmp)
        {
            const auto inf = self().get_info_second_value();
            const int  thr = get_thr_rank_fast();
            shrink_array::set_thread_chunk_size(self().size_policy_.size());
            for (const auto& per_rank : manager_type::map_chunk2_to_compress_potentially_thr)
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
        manager_type::remove_from_compress_list_chunk2_by_num_tab(num_tab);
    }
}

template <typename Derived>
void RechunkHooks2<Derived, true>::rechunk_drain_reuse_list(unsigned int new_size)
{
    manager_type::drain_reuse_list_for_size(new_size);
}

template <typename Derived>
RechunkHooks2<Derived, true>::value_type RechunkHooks2<Derived, true>::rechunk_capture_state() const
{
    // Preserve the array-level second_value across the rechunk (2A has no such
    // scalar; it is per-chunk there). Bool's second_value is re-established by the
    // fresh array's create(), so capture nothing (value_type{}) for it.
    if constexpr (Derived::is_not_bool)
    {
        return self().get_second_value();
    }
    else
    {
        return value_type{};
    }
}

template <typename Derived>
void RechunkHooks2<Derived, true>::rechunk_seed_fresh(in_t<value_type> state)
{
    // Called on the FRESH array before refill, so its compression decisions use
    // the same second_value. Only propagate a NON-TRIVIAL value: the default
    // (== first_value, value_type{}) is already the fresh array's state, and re-declaring
    // it would create a degenerate cst_val2 == cst_val1 buffer.
    if constexpr (Derived::is_not_bool)
    {
        if (state != value_type{})
        {
            self().set_second_value(state, false);
        }
    }
    else
    {
        (void)state;
    }
}

template <typename Derived>
bool RechunkHooks2<Derived, true>::rechunk_swap_identity(Derived& tmp)
{
    // The num_tab, ind_cst_val2 and inf_array MEMBERS are deliberately NOT
    // swapped: Chunk2 never reads them on the access / compression / teardown
    // paths (num_tab and second_value/ind come from share_info_array, keyed by
    // `this`; the members are only touched by operator= / set_second_value) and
    // they may be uninitialized, so swapping would read indeterminate values
    // (MSan / valgrind). The registry-tuple swap IS the identity swap.
    return proxy_type::share_info_array.swap_entries(self().registry_key(), tmp.registry_key());
}

template <typename Derived>
void RechunkHooks2<Derived, true>::rechunk_reown_after_swap(Derived& tmp, in_t<value_type> state)
{
    // Re-own the fast-block registration the fresh array created for `state`.
    // set_second_value() pointed map_inf_array[state] at &tmp.inf_array; when tmp
    // dies at end of rechunk_to's scope that entry dangles, and the NEXT
    // compression of `state` (find_inf_array -> *inf_ptr) dereferences it ->
    // use-after-return (tmp is a stack array). Copy tmp's valid (value, ind) into
    // *this and re-point the map at *this->inf_array, which lives on. fast-block
    // only (that is where map_inf_array exists); guarded by state != value_type{} (the only
    // case the fresh array registered a value).
    if constexpr (chunk_config::use_fast_block_compression)
    {
        if constexpr (Derived::is_not_bool)
        {
            if (state != value_type{})
            {
                Derived& arr = self();

                arr.inf_array    = tmp.inf_array; // tmp's is valid (set_second_value ran)
                arr.ind_cst_val2 = tmp.ind_cst_val2;
                proxy_type::share_info_array.insert(state, &arr.inf_array);
            }
        }
    }
    else
    {
        (void)state;
    }

    // rechunk_swap_identity() swapped the two REGISTRY payloads and, on
    // purpose, not the members. Write proxies are armed with the address of
    // the inf_array member, so both mirrors must now be re-read from the
    // registry: *this carries tmp's former (value, ind, num_tab), and tmp --
    // whose teardown still goes through the registry -- carries ours. All
    // build modes, bool included (its num_tab is per-array too).
    self().sync_registry_mirror();
    tmp.sync_registry_mirror();
}

} // namespace shrink_array
