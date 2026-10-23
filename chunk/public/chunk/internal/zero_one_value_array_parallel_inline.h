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

// Out-of-line bodies of zero_one_value::Array's mode-gated surfaces (zero_one_value::ParallelSupport,
// zero_one_value::FastBlockSupport). Included at the end of zero_one_value_array.h, so self() sees a complete
// Derived.

#include <tuple> // std::get on the candidate tuples (sweep of the other second-value keys)

#include "chunk/internal/zero_one_value_array_parallel.h"      // FastBlockSupport2, the class these bodies belong to
#include "chunk/internal/array_chunk_parallel_common_inline.h" // flush_pending_deferred_ops
#include "chunk/PMP_CHUNK.h"                                   // CHUNK_PMP_* (make_block_compression)
#include "chunk/internal/chunk_size_policy.h"                  // set_thread_chunk_size in make_block_compression
#include "chunk/config.h"                                      // build_config::chunk_dynamic

namespace shrink_array
{

template <typename Derived>
void zero_one_value::FastBlockSupport<Derived, true>::make_block_compression(this Derived& self)
{
    if constexpr (build_config::chunk_dynamic)
    {
        // The scan reads the active size: make it THIS array's.
        shrink_array::set_thread_chunk_size(self.chunk_size());
    }

    // bool has its own pass (the (thr, inf_array, proxy) overload static_asserts for it).
    // [&] captures self where the non-bool branch uses it: an explicit capture is unused in
    // the bool instantiation (-Wunused-lambda-capture).
    const auto pass_on_rank = [&](int thr)
    {
        if constexpr (Derived::is_not_bool)
        {
            ChunkType::make_block_compression(thr, self.get_info_second_value(), nullptr);
        }
        else
        {
            ChunkType::make_block_compression_bool(thr, nullptr);
        }
    };

    // Do not open a region here (an `omp parallel` form hits an LLVM OpenMP runtime bug):
    // outside a region every rank's list is drained serially, inside one each thread its own.
    if (!CHUNK_PMP_IN_PARALLEL_REGION)
    {
        for (int thr = 0; thr < CHUNK_PMP_RANK_BOUND; ++thr)
        {
            pass_on_rank(thr);
        }

        // The passes above drain this array's CURRENT second-value key only. Writes queued
        // before a set_second_value() (or a copy-assignment) sit under an EARLIER key; left
        // there, resize() reallocates chunk_list under them (dangling &chunk_list[j]). Sweep
        // every rank and key: compress this array's leftovers against the current value,
        // clear their flags, drop them by num_tab (as rechunk_drop_candidates()).
        // Sequential point: outside a region only.
        if constexpr (Derived::is_not_bool)
        {
            using ProxyT   = typename Traits::ProxyType;
            using ManagerT = typename Traits::ManagerType;

            const int  num_tab  = ProxyT::share_info_array->get_num_tab(self.registry_key());
            const auto inf      = self.get_info_second_value();
            const int  thr_self = get_thr_rank_fast();
            bool       found    = false;
            for (const auto& per_rank : ManagerT::map_chunk2_to_compress_potentially_thr)
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
                            std::get<0>(tup)->compress(thr_self, inf, /*apply=*/true, std::get<3>(tup).count());
                            ProxyT::reset_tab_inside_list_to_compress(num_tab, std::get<1>(tup));
                            found = true;
                        }
                    }
                }
            }
            if (found)
            {
                ManagerT::remove_from_compress_list_chunk2_by_num_tab(num_tab);
            }
        }
    }
    else
    {
        pass_on_rank(CHUNK_PMP_THREAD_RANK);
    }
}

} // namespace shrink_array
