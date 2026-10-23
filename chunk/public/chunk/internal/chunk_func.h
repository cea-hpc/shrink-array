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

#include <vector> // std::vector (the per-rank reuse lists)

#include "chunk/PMP_CHUNK.h" // CHUNK_PMP_RANK_BOUND
#include "chunk/chunk_std.h" // self-contained: needs Chunk<T>::manager_chunk

// Function templates over the per-thread reuse lists, instantiated only by the
// OpenMP manager (ManagerChunkParallelImpl<..., true>); every list access is a
// dependent name, so no preprocessor guard is needed.
//
// Callers instantiate from these definitions. If one sees only the forward
// declaration in manager_chunk_inline.h, include this header there -- do not
// add a hand-maintained explicit instantiation list.

namespace shrink_array
{
/// @brief Whether the per-rank free-block lists of Chunk<T> are unbalanced.
/// @tparam T Element type.
/// @return true when the lists are not all empty and some rank's list size lies outside [0.9, 1.1] times the
///         mean size over CHUNK_PMP_RANK_BOUND ranks.
template <typename T>
bool need_load_balance_list_block_free_thr()
{
    const int max_thr = CHUNK_PMP_RANK_BOUND; // the ranks that may hold a free list

    int lg = 0;

    for (const auto& elm : zero::Chunk<T>::Manager::get_list_block_free_thr())
    {
        lg += elm.list.size();
    }

    if (!lg)
    {
        return false;
    }

    lg /= max_thr;

    for (int thr_rank = 0; thr_rank < max_thr; ++thr_rank)
    {
        const int sz = zero::Chunk<T>::Manager::get_list_block_free_thr_size(thr_rank);
        if (sz < 0.9 * lg || sz > 1.1 * lg)
        {
            return true;
        }
    }

    return false;
}

/// @brief Even out the per-rank free-block lists of Chunk<T>: blocks past the mean size are moved from the
///        longer lists to the shorter ones. Does nothing when the total is below the rank count.
/// @tparam T Element type.
template <typename T>
void load_balance_list_block_free_thr()
{
    int lg = 0;

    const int max_thr = CHUNK_PMP_RANK_BOUND; // the ranks that may hold a free list

    for (int thr_rank = 0; thr_rank < max_thr; ++thr_rank)
    {
        lg += zero::Chunk<T>::Manager::get_list_block_free_thr(thr_rank).size();
    }

    if (lg < max_thr) // also covers lg == 0: max_thr >= 1
    {
        return;
    }

    lg /= max_thr;

    std::vector<T*> ll;

    for (int thr_rank = 0; thr_rank < max_thr; ++thr_rank)
    {
        auto& lbf = zero::Chunk<T>::Manager::get_list_block_free_thr(thr_rank);

        const int sz = lbf.size();

        if (sz > lg)
        {
            const int ll_sz = ll.size();

            ll.resize(ll_sz + sz - lg);

            T** ll_loc = ll.data() + ll_sz - lg;

            for (int i = lg; i < sz; ++i)
            {
                ll_loc[i] = lbf[i];
            }

            lbf.resize(lg);
        }
    }

    for (int thr_rank = 0; thr_rank < max_thr; ++thr_rank)
    {
        auto& lbf = zero::Chunk<T>::Manager::get_list_block_free_thr(thr_rank);

        const int sz = lbf.size();

        if (sz < lg)
        {
            lbf.resize(lbf.size() + lg - sz);

            const int pp = ll.size() - (lg - sz);

            T* const* ll_loc = ll.data() + ll.size() - lg;

            for (int i = sz; i < lg; ++i)
            {
                lbf[i] = ll_loc[i];
            }

            if (pp == -1)
            {
                ll.clear();
            }
            else
            {
                ll.resize(pp);
            }
        }
    }

    if (!ll.empty())
    {
        for (unsigned int i = 0; i < ll.size(); ++i)
        {
            zero::Chunk<T>::Manager::get_list_block_free_thr(0).emplace_back(ll[i]);
        }
    }
}

} // namespace shrink_array
