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

// Out-of-line bodies of ArrayChunk2A's re-chunk hooks (RechunkHooks2A<Derived,
// true>). Included at the end of array_chunk2A.h, after the class, so the CRTP
// self() down-cast sees a complete Derived and so the manager include stays out
// of the container declaration itself.
#include <utility> // std::swap

#include "chunk/internal/array_chunk2A_rechunk_hooks.h"
#include "chunk/internal/manager_chunk.h" // manager_type — candidate store + reuse free-lists
#include <cstdlib> // std::getenv (CHUNK_RECHUNK_DROP_LEGACY)
#include <tuple>   // std::get on candidate tuples

namespace shrink_array
{

// Defined in each variant's hooks file, guarded: a TU that includes several
// variants must see a single definition.
#ifndef SHRINK_ARRAY_RECHUNK_DROP_LEGACY_DEFINED
#define SHRINK_ARRAY_RECHUNK_DROP_LEGACY_DEFINED
namespace rechunk_detail
{
/// A/B switch: CHUNK_RECHUNK_DROP_LEGACY=1 (read once) makes
/// rechunk_drop_candidates() drop the array's candidates WITHOUT compressing
/// them first -- the behaviour before Sept '26.
inline const bool g_rechunk_drop_legacy = []
{
    const char* e = std::getenv("CHUNK_RECHUNK_DROP_LEGACY");
    return e != nullptr && e[0] != '\0' && e[0] != '0';
}();
} // namespace rechunk_detail
#endif

template <typename Derived>
void RechunkHooks2A<Derived, true>::rechunk_drop_candidates()
{
    // 2A uses the num_tab MEMBER (set in create_common/pre_create_hook) as the
    // key for its candidate store. See RechunkSupport::rechunk_to for why the
    // drop must happen at the owning geometry.
    //
    // COMPRESS them first (Sept '26): a chunk queued by the write that made it
    // uniform, dropped here, stayed materialised through the rebuild (copied,
    // not folded) and was never re-queued. compress() is the variant's own,
    // guarded against an already-compressed chunk; the geometry is the array's
    // current one, and each entry carries its own element count.
    if constexpr (chunk_config::use_fast_block_compression)
    {
        if (!rechunk_detail::g_rechunk_drop_legacy)
        {
            const int num_tab = self().num_tab;
            const int thr     = get_thr_rank_fast();
            shrink_array::set_thread_chunk_size(self().size_policy_.size());
            auto compress_list = [&](auto& list)
            {
                for (auto& tup : list)
                    if (std::get<2>(tup) == num_tab)
                        std::get<0>(tup)->compress(thr, std::get<1>(tup), num_tab, /*apply=*/true, std::get<3>(tup));
            };
            if constexpr (build_config::use_openmp)
            {
                for (auto& list : manager_type::chunk2A_to_compress_potentially_thr)
                    compress_list(list);
            }
            else
            {
                compress_list(manager_type::chunk2A_to_compress_potentially);
            }
        }
        manager_type::remove_from_compress_list_chunk2a_by_num_tab(self().num_tab);
    }
}

template <typename Derived>
void RechunkHooks2A<Derived, true>::rechunk_drain_reuse_list(unsigned int new_size)
{
    manager_type::drain_reuse_list_for_size(new_size);
}

template <typename Derived>
bool RechunkHooks2A<Derived, true>::rechunk_swap_identity(Derived& tmp)
{
    // 2A stores num_tab as a member and indexes its metadata by it, so swap the
    // member AND make the registry agree (get_num_tab(registry_key()) -> S' row).
    using std::swap;
    swap(self().num_tab, tmp.num_tab);
    const bool ok = proxy_type::share_info_array.swap_keys(self().registry_key(), tmp.registry_key());
    // num_tab changed hands: each array now owns the other's metadata rows.
    self().sync_meta_bases();
    tmp.sync_meta_bases();
    return ok;
}

} // namespace shrink_array
