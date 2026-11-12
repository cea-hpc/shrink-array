/*
 * Copyright 2026 CEA (www.cea.fr)
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

/**
 * @file compress_candidates_inline.h
 * @brief Out-of-line bodies of CompressCandidates, the compression-candidate
 *        lists of one (variant, element type). Included by type_state.h
 *        once TypeState is complete.
 *
 * Almost every body here has the same shape: an `if constexpr` ladder on which
 * chunk variant ChunkT is. That is not incidental — the three variants keep
 * their candidates in genuinely different containers:
 *  - **Chunk** : one list per rank;
 *  - **zero_one_value**: a map per rank, KEYED BY SECOND VALUE, so anything that walks
 *    the candidates has to iterate the array's registered second values first
 *    (and `bool`, whose second value is always `true`, takes a separate branch);
 *  - **zero_any**: one list per rank again, since its second value is per chunk.
 *
 * @warning have_candidates() must mirror compress_pass()'s dispatch EXACTLY.
 *          They are the predicate and the action of the same decision: if the
 *          predicate reports "nothing to compress" for a variant the action
 *          would have handled, the between-phase pass never runs for it and all
 *          of its compression falls on the opportunistic in-loop path.
 */

#include <algorithm> // std::ranges
#include <tuple>     // std::get
#include <utility>   // std::pair

#include "chunk/internal/compress_candidates.h"  // the declarations these bodies belong to
#include "chunk/config.h"                        // build_config, chunk_config
#include "chunk/internal/PMP_CHUNK.h"            // CHUNK_PMP_*
#include "chunk/internal/zero_chunk.h"           // Chunk<T>: the ChunkT<T>:: calls
#include "chunk/internal/zero_one_value_chunk.h" // zero_one_value::Chunk<T>: the ChunkT<T>:: calls
#include "chunk/internal/zero_any_chunk.h"       // zero_any::Chunk<T>: the ChunkT<T>:: calls
#include "chunk/internal/concept_chunk.h"        // IsChunk, IsZeroOneValue, IsZeroAnyFamily

namespace shrink_array
{
/**
 * @brief Compress every compressible chunk of this type, ignoring the candidate lists.
 *
 * The unconditional sweep: it does not ask whether anything was queued, it runs
 * the variant's block compression directly. Used where the caller wants the
 * memory back now rather than at the next phase boundary.
 *
 * @note zero and zero_any take a proxy argument (null here: no write triggered
 *       this), zero_one_value does not — hence the three-way `if constexpr`.
 */
template <template <typename> typename ChunkT, typename T>
void CompressCandidates<ChunkT, T>::force_compress()
{
    // The calling thread's rank, in or out of a region: no region is opened.
    const int thr = CHUNK_PMP_THREAD_RANK;

    if constexpr (chunk_config::use_fast_block_compression && IsChunk<ChunkT, T>)
    {
        ChunkT<T>::make_block_compression(thr, nullptr);
    }

    if constexpr (IsZeroAnyFamily<ChunkT, T>)
    {
        ChunkT<T>::make_block_compression(thr, nullptr);
    }
    else if constexpr (chunk_config::use_fast_block_compression && IsZeroOneValue<ChunkT, T>)
    {
        ChunkT<T>::make_block_compression(thr);
    }
}

/**
 * @brief Sizes this type's compression-candidate containers to the current team.
 *
 * Which containers exist depends on the variant (see the file header): a plain
 * per-rank list for zero and zero_any, plus the per-second-value map for
 * zero_one_value. A list keeps its capacity across the passes' clear(), so nothing
 * pre-reserves it.
 */
template <template <typename> typename ChunkT, typename T>
void CompressCandidates<ChunkT, T>::resize_lists()
{
    const int nthr = CHUNK_PMP_THREAD_CAPACITY; // every rank that may run, as the manager's max_thr

    // --- Standard Chunk case ---
    if constexpr (chunk_config::use_fast_block_compression && IsChunk<ChunkT, T>)
    {
        // Resize the list of chunks potentially needing compression
        zero_candidates_thr.resize(nthr);
    }

    // --- Complex types (zero_one_value/zero_any) multi-threading logic ---
    if constexpr (build_config::threaded)
    {
        // Specific to zero_one_value: Dual-value compression with static maps
        if constexpr (IsZeroOneValue<ChunkT, T>)
        {
            zero_one_value_candidates_map_thr.resize(nthr);
            zero_one_value_candidates_thr.resize(nthr);
        }

        // Specific to zero_any: Per-chunk secondary value compression
        if constexpr (IsZeroAnyFamily<ChunkT, T>)
        {
            zero_any_candidates_thr.resize(nthr);
        }
    }
}

/**
 * @brief Is there anything queued for compression for this type?
 *
 * Cheap gate for the callers that would otherwise open a parallel region for
 * nothing. For zero_one_value it has to ask once per registered second value, since
 * that is what keys the candidate map.
 *
 * @param n_thr   Number of per-rank lists to examine.
 * @return True as soon as one list holds a candidate.
 *
 * @warning This predicate and compress_chunk_to_compress_potentially() must agree
 *          branch for branch — see the @warning in the file header.
 */
template <template <typename> typename ChunkT, typename T>
bool CompressCandidates<ChunkT, T>::have_candidates(int n_thr)
{
    // Mirror compress_chunk_to_compress_potentially's dispatch exactly (zero_one_value is handled
    // whenever use_fast_block_compression is on, GNU included, not only under OMPT): see the
    // file header's @warning.
    if constexpr (chunk_config::use_fast_block_compression)
    {
        if constexpr (IsZeroOneValue<ChunkT, T>)
        {
            if constexpr (is_not_bool)
            {
                const auto& sh_inf           = *zero_one_value::Proxy<T>::share_info_array;
                const auto& v_all_second_val = sh_inf.get_all_second_value();
                return std::ranges::any_of(v_all_second_val, [n_thr](const auto& second_value)
                                           { return ChunkT<T>::have_chunk_to_compress(second_value, n_thr); });
            }
            else
            {
                return ChunkT<T>::have_chunk_to_compress_bool(n_thr);
            }
        }
        else
        {
            return ChunkT<T>::have_chunk_to_compress(n_thr);
        }
    }

    return false;
}

/**
 * @brief Drain this type's compression-candidate backlog.
 *
 * The action whose predicate is have_chunk_to_compress_potentially(). zero_one_value is
 * driven once per registered second value; the other variants have a single
 * entry point.
 *
 * @param apply true: compress the candidates found constant; false: walk the
 *        lists without compressing (the dying array's purge).
 */
template <template <typename> typename ChunkT, typename T>
void CompressCandidates<ChunkT, T>::compress_pass([[maybe_unused]] bool apply)
{
    if constexpr (chunk_config::use_fast_block_compression)
    {
        if constexpr (IsZeroOneValue<ChunkT, T>)
        {
            if constexpr (is_not_bool)
            {
                const auto& sh_inf          = *zero_one_value::Proxy<T>::share_info_array;
                const auto& v_all_inf_array = sh_inf.get_all_inf_array();
                for (const auto& inf_array : v_all_inf_array)
                {
                    ChunkT<T>::compress_chunk_to_compress(*inf_array, apply);
                }
            }
            else
            {
                ChunkT<T>::compress_chunk_to_compress_bool(apply);
            }
        }
        else
        {
            ChunkT<T>::compress_chunk_to_compress(apply);
        }
    }
}

template <template <typename> class ChunkT, typename T>
auto& CompressCandidates<ChunkT, T>::get_zero_any_candidates([[maybe_unused]] int thr) noexcept
{
    if constexpr (build_config::threaded)
    {
        return zero_any_candidates_thr[thr];
    }
    else
    {
        return zero_any_candidates;
    }
}

template <template <typename> class ChunkT, typename T>
[[nodiscard]] bool CompressCandidates<ChunkT, T>::is_zero_any_candidates_empty() noexcept
{
    if constexpr (build_config::threaded)
    {
        return zero_any_candidates_thr.empty();
    }
    else
    {
        return zero_any_candidates.empty();
    }
}

template <template <typename> class ChunkT, typename T>
[[nodiscard]] std::pair<int, int> CompressCandidates<ChunkT, T>::removal_ranks(int nthr, bool own_rank_only) noexcept
{
    if (own_rank_only)
    {
        const int rank = CHUNK_PMP_THREAD_RANK;
        return {rank, rank < nthr ? rank + 1 : rank};
    }
    return {0, nthr};
}

template <template <typename> class ChunkT, typename T>
void CompressCandidates<ChunkT, T>::remove_from_zero_any_candidates_by_ptr_range(
    const void*           lo,
    const void*           hi,
    [[maybe_unused]] bool own_rank_only) noexcept
{
    const auto in_range = [lo, hi](const auto& tup)
    {
        const void* ptr = static_cast<const void*>(std::get<0>(tup));
        return ptr >= lo && ptr < hi;
    };
    if constexpr (build_config::threaded)
    {
        const auto [first, last] = removal_ranks(static_cast<int>(zero_any_candidates_thr.size()), own_rank_only);
        for (int thr = first; thr < last; ++thr)
        {
            std::erase_if(zero_any_candidates_thr[thr], in_range);
        }
    }
    else
    {
        std::erase_if(zero_any_candidates, in_range);
    }
}

template <template <typename> class ChunkT, typename T>
void CompressCandidates<ChunkT, T>::remove_from_zero_any_candidates_by_num_tab(
    int                   num_tab,
    [[maybe_unused]] bool own_rank_only) noexcept
{
    if constexpr (build_config::threaded)
    {
        const auto [first, last] = removal_ranks(static_cast<int>(zero_any_candidates_thr.size()), own_rank_only);
        for (int thr = first; thr < last; ++thr)
        {
            erase_candidates_of_tab(zero_any_candidates_thr[thr], num_tab);
        }
    }
    else
    {
        erase_candidates_of_tab(zero_any_candidates, num_tab);
    }
}

template <template <typename> class ChunkT, typename T>
void CompressCandidates<ChunkT, T>::remove_from_zero_one_value_candidates_by_num_tab(
    [[maybe_unused]] int  num_tab,
    [[maybe_unused]] bool own_rank_only) noexcept
{
    if constexpr (build_config::threaded)
    {
        const auto [first, last] =
            removal_ranks(static_cast<int>(zero_one_value_candidates_map_thr.size()), own_rank_only);
        for (int thr = first; thr < last; ++thr)
        {
            for (const auto& kv : zero_one_value_candidates_map_thr[thr])
            {
                const auto& sp = kv.second;
                if (!sp)
                {
                    continue;
                }
                erase_candidates_of_tab(*sp, num_tab);
            }
        }
    }
}

template <template <typename> class ChunkT, typename T>
void CompressCandidates<ChunkT, T>::remove_from_zero_candidates_by_num_tab(int                   num_tab,
                                                                           [[maybe_unused]] bool own_rank_only) noexcept
{
    if constexpr (build_config::threaded)
    {
        const auto [first, last] = removal_ranks(static_cast<int>(zero_candidates_thr.size()), own_rank_only);
        for (int thr = first; thr < last; ++thr)
        {
            erase_candidates_of_tab(zero_candidates_thr[thr], num_tab);
        }
    }
    else
    {
        erase_candidates_of_tab(zero_candidates, num_tab);
    }
}

} // namespace shrink_array
