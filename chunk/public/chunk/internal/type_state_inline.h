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
e  * @file type_state_inline.h
 * @brief Out-of-line bodies of TypeState, the per-(variant, element type)
 *        state behind every array: what crosses its bases -- the single
 *        compression sweep, the budgets computed from the chunk counters, the
 *        keyed allocator sets.
 *
 * @see reuse_pools_inline.h, compress_candidates_inline.h and
 *      type_state_policies_inline.h for the bodies of its bases.
 * @see all_type_chunk.h for the registry that drives every manager at once.
 */

#include <algorithm>   // std::max
#include <tuple>       // std::tuple
#include <type_traits> // std::conditional_t

#include "chunk/internal/type_state.h"   // the declarations these bodies belong to
#include "chunk/config.h"                   // build_config, chunk_config
#include "chunk/log.h"                      // CHUNK_LOG
#include "chunk/internal/PMP_CHUNK.h"                // CHUNK_PMP_*
#include "chunk/internal/zero_chunk.h"               // Chunk<T>: the ChunkT<T>:: calls
#include "chunk/internal/zero_one_value_chunk.h"     // zero_one_value::Chunk<T>: the ChunkT<T>:: calls
#include "chunk/internal/zero_any_chunk.h"           // zero_any::Chunk<T>: the ChunkT<T>:: calls
#include "chunk/internal/zero_chunk_def.h"           // zero::ChunkDef
#include "chunk/internal/zero_one_value_chunk_def.h" // zero_one_value::ChunkDef
#include "chunk/internal/zero_any_chunk_def.h"       // zero_any::ChunkDef
#include "chunk/internal/concept_chunk.h"   // IsChunk, IsZeroOneValue, IsZeroAnyFamily
#include "chunk/internal/global_stats_chunk_list.h"  // GlobalStatsChunkList
#include "chunk/internal/type_name.h"                // type_name<T>
#include "chunk/internal/singleton.h"                // SingletonChk

namespace shrink_array
{
/// @brief zero_one_value: snapshot the second-value lists the sweep drains.
template <template <typename> typename ChunkT, typename T>
void TypeState<ChunkT, T>::sweep_prepare()
{
    if constexpr (chunk_config::use_fast_block_compression && IsZeroOneValue<ChunkT, T>)
    {
        if constexpr (is_not_bool)
        {
            const auto inf_arrays = zero_one_value::Proxy<T>::share_info_array->get_all_inf_array();
            sweep_inf_arrays_.assign(inf_arrays.begin(), inf_arrays.end());
        }
    }
}

/// @brief Rank @p thr's share of compress_chunk_to_compress_potentially().
template <template <typename> typename ChunkT, typename T>
void TypeState<ChunkT, T>::sweep_rank([[maybe_unused]] int thr, [[maybe_unused]] bool apply)
{
    if constexpr (chunk_config::use_fast_block_compression)
    {
        if constexpr (IsZeroOneValue<ChunkT, T>)
        {
            if constexpr (is_not_bool)
            {
                for (const auto* inf_array : sweep_inf_arrays_)
                {
                    ChunkT<T>::compress_sweep_rank(thr, *inf_array, apply);
                }
            }
            else
            {
                ChunkT<T>::compress_sweep_rank_bool(thr, apply);
            }
        }
        else if constexpr (IsZeroAnyFamily<ChunkT, T>)
        {
            ChunkT<T>::compress_sweep_rank(thr, apply);
        }
        else
        {
            ChunkT<T>::compress_sweep_rank(thr); // Chunk: always compresses
        }
    }
}

/// @brief zero_one_value always swept the ranks past the team; zero_any in one pass;
///        Chunk never.
template <template <typename> typename ChunkT, typename T>
bool TypeState<ChunkT, T>::sweep_mops_up([[maybe_unused]] bool apply) const
{
    if constexpr (chunk_config::use_fast_block_compression)
    {
        if constexpr (IsZeroOneValue<ChunkT, T>)
        {
            return true;
        }
        else if constexpr (IsZeroAnyFamily<ChunkT, T>)
        {
            return ChunkT<T>::compress_sweep_mops_up(apply);
        }
    }
    return false;
}

/**
 * @brief Release this type's allocation blocks.
 * @note Serial builds only. Under OpenMP the per-thread allocators own the
 *       blocks and are torn down through their own path, so the body is empty
 *       there.
 */
template <template <typename> typename ChunkT, typename T>
void TypeState<ChunkT, T>::destroy_block_chunk()
{
    // allocator_data is declared in every build (chunk_base.h) and is a dependent name, so the
    // discarded branch is never instantiated. std-alloc: destroy() is a no-op.
    if constexpr (build_config::not_threaded)
    {
        ChunkT<T>::allocator_data->destroy();
    }
}

template <template <typename> typename ChunkT, typename T>
bool TypeState<ChunkT, T>::register_alloc_key([[maybe_unused]] unsigned int key,
                                                 [[maybe_unused]] unsigned int size,
                                                 [[maybe_unused]] int          nb_chunk)
{
    if constexpr (build_config::chunk_dynamic && build_config::use_fast_alloc && build_config::threaded)
    {
        // Without a slab budget there is nothing to build the keyed allocator
        // set from. Report the failure: the caller must NOT keep the key, or
        // every allocation of that array would route to a set that does not
        // exist and fall back to the legacy single allocator.
        if (key == 0 || registered_block_size_ == 0)
        {
            return false;
        }
        // The per-rank sets are sized for every rank that may ever run (the
        // capacity); the budget is shared by the team that runs now.
        const int capacity = CHUNK_PMP_THREAD_CAPACITY;
        const int max_thr  = CHUNK_PMP_MAX_THREADS;
        ChunkT<T>::register_chunk_size(size, capacity, registered_block_size_, key);
        Pools::register_free_list_key(key, size, capacity);
        // The array's own share per rank, where the global budget uses the
        // process-wide somme_chunk_create.
        Pools::set_size_buffer_reuse(key, static_cast<int>(nb_chunk / (Pools::reuse_share_div() * max_thr)));
        CHUNK_LOG(debug, affinity, "key={} size={} nb_chunk={} budget={} threads={}", key, size, nb_chunk,
                  Pools::get_size_buffer_reuse(key), max_thr);
        return true;
    }
    else
    {
        return false;
    }
}

// compute_size_buffer_reuse (documented in type_state.h): derives ReusePools' reuse
// budget from the number of chunks created so far, so a program that allocates
// more keeps proportionally more buffers parked for reuse.
template <template <typename> typename ChunkT, typename T>
void TypeState<ChunkT, T>::compute_size_buffer_reuse()
{
    // zero_any::ChunkDef's reuse constants serve every variant (zero_any_chunk_def.h).
    Pools::size_buffer_reuse =
        static_cast<int>(somme_chunk_create / (Pools::reuse_share_div() * CHUNK_PMP_MAX_THREADS));
    Pools::size_buffer_reuse = std::max(Pools::size_buffer_reuse, zero_any::ChunkDef::size_min_buffer_reuse);
}

// compute_size_max_number_of_chunk_to_check_compress (documented in
// type_state.h): sets the threshold past which a rank's candidate list is
// flagged dangerous -- the szm that ProxyBase::should_trigger_compression
// compares against.
template <template <typename> typename ChunkT, typename T>
void TypeState<ChunkT, T>::compute_size_max_number_of_chunk_to_check_compress()
{
    auto& global_stats = SingletonChk<GlobalStatsChunkList>::get_ref();

    // Take the tuning constants from THIS manager's own variant: they differ a great deal
    // (see ChunkDef / zero_one_value::ChunkDef / zero_any::ChunkDef). The value is divided by the thread count, so at
    // high thread counts the floor IS the threshold. Another variant's higher floor means a
    // rank's list almost never reaches it: compression almost never runs during execution and
    // every materialisation allocates afresh.
    using VariantDef =
        std::conditional_t<IsZeroAnyFamily<ChunkT, T>, zero_any::ChunkDef,
                           std::conditional_t<IsZeroOneValue<ChunkT, T>, zero_one_value::ChunkDef, zero::ChunkDef>>;

    int threshold = somme_chunk_create / (VariantDef::factor_div_nb_chunk_for_compress * CHUNK_PMP_MAX_THREADS);
    threshold     = std::max(threshold, VariantDef::size_min_list_chunk_to_check_compress);

    Candidates::set_size_max_number_of_chunk_to_check_compress(threshold);
    const size_t element_size = sizeof(std::tuple<ChunkT<T>*, int, int>);

    global_stats.record(type_name<ChunkT<T>>(), somme_chunk_create, threshold, element_size);
}

} // namespace shrink_array
