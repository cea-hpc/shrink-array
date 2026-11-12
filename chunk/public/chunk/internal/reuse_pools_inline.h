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
 * @file reuse_pools_inline.h
 * @brief Out-of-line bodies of ReusePools, the buffer reuse pools of one
 *        (variant, element type). Included by type_state.h once
 *        TypeState is complete.
 */

#include <algorithm>   // std::find, std::max, std::sort, std::unique
#include <atomic>      // std::atomic, std::memory_order
#include <chrono>      // std::chrono::steady_clock (the team drain's probe)
#include <cstddef>     // std::size_t
#include <type_traits> // std::remove_reference_t
#include <utility>     // std::pair
#include <vector>      // std::vector
#include <memory>      // std::make_unique

#include "chunk/internal/reuse_pools.h"          // the declarations these bodies belong to
#include "chunk/config.h"                        // build_config, chunk_config
#include "chunk/log.h"                           // CHUNK_LOG_ALWAYS
#include "fast_allocators/internal/svalloc.h"    // svfree
#include "chunk/internal/PMP_CHUNK.h"            // CHUNK_PMP_*, pmp::
#include "chunk/internal/chunk_size_policy.h"    // alloc_chunk_count, thread_chunk_size
#include "chunk/internal/pool_ledger.h"          // CHUNK_LEDGER_*
#include "chunk/internal/drain_probe.h"          // g_drain_probe (rechunk=debug, the team drain's split)
#include "chunk/internal/freepath_counters.h"    // CHUNK_FREEPATH_*
#include "chunk/internal/zero_chunk.h"           // Chunk<T>: the ChunkT<T>:: calls
#include "chunk/internal/zero_one_value_chunk.h" // zero_one_value::Chunk<T>: the ChunkT<T>:: calls
#include "chunk/internal/zero_any_chunk.h"       // zero_any::Chunk<T>: the ChunkT<T>:: calls
#include "chunk/internal/type_name.h"            // type_name<T>

namespace shrink_array
{
template <template <typename> class ChunkT, typename T>
void ReusePools<ChunkT, T>::register_free_list_size([[maybe_unused]] unsigned int size, [[maybe_unused]] int max_thr)
{
    if constexpr (chunk_config::chunk_dynamic)
    {
        const auto it = list_block_free_thr_by_size_->find(size);
        if (it == list_block_free_thr_by_size_->end())
        {
            (*list_block_free_thr_by_size_)[size].resize(max_thr);
        }
    }
}

template <template <typename> class ChunkT, typename T>
void ReusePools<ChunkT, T>::register_free_list_key(unsigned int key, unsigned int size, int max_thr)
{
    if (key == 0)
    {
        return; // the shared pool is register_free_list_size()'s
    }
    const auto it = list_block_free_thr_keyed_->find(keyed_index(key, size));
    if (it == list_block_free_thr_keyed_->end())
    {
        (*list_block_free_thr_keyed_)[keyed_index(key, size)].resize(max_thr);
    }
}

template <template <typename> class ChunkT, typename T>
void ReusePools<ChunkT, T>::print_reuse_pool_breakdown(const char* label) noexcept
{
    std::size_t legacy_blocks = 0;
    for (const auto& ptf : list_block_free_thr)
    {
        legacy_blocks += ptf.list.size();
    }

    if (legacy_blocks != 0)
    {
        CHUNK_LOG_ALWAYS(pool, "{} {:<22} legacy store        : {:9} block(s), {:8.1f} MB (sized by the ambient {})",
                         label, type_name<T>(), legacy_blocks,
                         (double)(legacy_blocks * sizeof(T) * ChunkT<T>::storage_units(thread_chunk_size())) / 1e6,
                         thread_chunk_size());
    }

    if constexpr (chunk_config::chunk_dynamic)
    {
        for (const auto& kv : *list_block_free_thr_by_size_)
        {
            std::size_t blocks          = 0;
            std::size_t non_empty_ranks = 0;
            for (const auto& ptf : kv.second)
            {
                blocks += ptf.list.size();
                if (!ptf.list.empty())
                {
                    ++non_empty_ranks;
                }
            }
            if (blocks == 0)
            {
                continue;
            }
            CHUNK_LOG_ALWAYS(pool, "{} {:<22} size {:<6}        : {:9} block(s), {:8.1f} MB across {}/{} rank(s)",
                             label, type_name<T>(), kv.first, blocks,
                             (double)(blocks * sizeof(T) * ChunkT<T>::storage_units(kv.first)) / 1e6, non_empty_ranks,
                             kv.second.size());
        }
    }
}

template <template <typename> class ChunkT, typename T>
[[nodiscard]] std::size_t ReusePools<ChunkT, T>::reuse_pool_bytes() noexcept
{
    std::size_t total         = 0;
    std::size_t legacy_blocks = 0;

    for (const auto& ptf : list_block_free_thr)
    {
        legacy_blocks += ptf.list.size();
    }
    total += legacy_blocks * sizeof(T) * ChunkT<T>::storage_units(thread_chunk_size());

    if constexpr (chunk_config::chunk_dynamic)
    {
        for (const auto& kv : *list_block_free_thr_by_size_)
        {
            std::size_t blocks = 0;
            for (const auto& ptf : kv.second)
            {
                blocks += ptf.list.size();
            }
            total += blocks * sizeof(T) * ChunkT<T>::storage_units(kv.first);
        }
    }
    return total;
}

template <template <typename> class ChunkT, typename T>
std::size_t ReusePools<ChunkT, T>::trim_reuse_pools(std::size_t keep_per_thread) noexcept
{
    if constexpr (chunk_config::use_fast_alloc)
    {
        if (CHUNK_PMP_IN_PARALLEL_REGION)
        {
            return 0;
        }
        const std::size_t freed = release_parked_buffers(keep_per_thread);
        (void)sweep_empty_slabs();
        return freed;
    }
    else
    {
        std::size_t freed = 0;

        const auto trim_one = [&freed, keep_per_thread](auto& lst, std::size_t elem_count) noexcept
        {
            if (lst.size() <= keep_per_thread)
            {
                return;
            }
            while (lst.size() > keep_per_thread)
            {
                const T* block = lst.back();
                lst.pop_back();
                CHUNK_LEDGER_DISCARD(block, "trim_reuse_pools");
                fast_alloc::svfree(block);
                freed += ChunkT<T>::storage_units(elem_count) * sizeof(T);
            }
            // Release the pointer array's peak capacity too.
            lst.shrink_to_fit();
        };

        for (auto& ptf : list_block_free_thr)
        {
            trim_one(ptf.list, thread_chunk_size());
        }

        if constexpr (chunk_config::chunk_dynamic)
        {
            for (auto& kv : *list_block_free_thr_by_size_)
            {
                for (auto& ptf : kv.second)
                {
                    trim_one(ptf.list, kv.first);
                }
            }
        }

        return freed;
    }
}

template <template <typename> class ChunkT, typename T>
template <typename F>
void ReusePools<ChunkT, T>::for_each_sized_reuse_set(F&& func)
{
    if constexpr (chunk_config::chunk_dynamic)
    {
        for (auto& kv : *list_block_free_thr_by_size_)
        {
            func(kv.first, kv.second);
        }
    }
    else
    {
        func(alloc_chunk_count(), list_block_free_thr);
    }
    for (auto& kv : *list_block_free_thr_keyed_)
    {
        func(static_cast<unsigned int>(kv.first & 0xffffffffULL), kv.second);
    }
}

template <template <typename> class ChunkT, typename T>
std::size_t ReusePools<ChunkT, T>::release_parked_buffers([[maybe_unused]] std::size_t keep_per_thread) noexcept
{
    std::size_t bytes_freed = 0;
    if constexpr (chunk_config::use_fast_alloc)
    {
        using AllocT                = std::remove_reference_t<decltype(ChunkT<T>::get_chunk_allocator(0))>;
        const unsigned int saved_sz = thread_chunk_size();
        for_each_sized_reuse_set(
            [&](unsigned int size, std::vector<PerThreadFreeList>& per_thr)
            {
                set_thread_chunk_size(size);
                // Any instance of this size: slab_of() needs its geometry only.
                const AllocT&     geometry    = ChunkT<T>::get_chunk_allocator(0, 0u);
                const std::size_t block_bytes = ChunkT<T>::storage_units(size) * sizeof(T);
                for (auto& ptf : per_thr)
                {
                    auto& lst = ptf.list;
                    if (lst.size() <= keep_per_thread)
                    {
                        continue;
                    }
                    std::size_t kept = keep_per_thread;
                    for (std::size_t i = keep_per_thread; i < lst.size(); ++i)
                    {
                        T*   block = lst[i];
                        auto slab  = geometry.slab_of(block);
                        if (slab && slab->owner)
                        {
                            CHUNK_FREEPATH_INC(freed);
                            CHUNK_LEDGER_DISCARD(block, "trim_reuse_pools.owner");
                            static_cast<AllocT*>(slab->owner)->free_elt_pow2(block, slab);
                            bytes_freed += block_bytes;
                        }
                        else
                        {
                            lst[kept++] = block; // no slab claims it: stays parked
                        }
                    }
                    lst.resize(kept);
                    lst.shrink_to_fit();
                }
            });
        set_thread_chunk_size(saved_sz);
    }
    return bytes_freed;
}

template <template <typename> class ChunkT, typename T>
std::size_t ReusePools<ChunkT, T>::sweep_empty_slabs() noexcept
{
    std::size_t collected = 0;
    // OpenMP builds only, like the re-chunk's sweep: the serial
    // TCTypeAllocator has no SlabSweep / collect_empty_slabs.
    if constexpr (chunk_config::use_fast_alloc && build_config::threaded)
    {
        if (CHUNK_PMP_IN_PARALLEL_REGION)
        {
            return 0;
        }
        using AllocT = std::remove_reference_t<decltype(ChunkT<T>::get_chunk_allocator(0))>;
        // (size, key) of every allocator set: the shared pools first.
        std::vector<std::pair<unsigned int, unsigned int>> sets;
        if constexpr (chunk_config::chunk_dynamic)
        {
            for (const auto& kv : *list_block_free_thr_by_size_)
            {
                sets.emplace_back(kv.first, 0u);
            }
        }
        else
        {
            sets.emplace_back(alloc_chunk_count(), 0u);
        }
        for (const auto& kv : *list_block_free_thr_keyed_)
        {
            sets.emplace_back(static_cast<unsigned int>(kv.first & 0xffffffffULL),
                              static_cast<unsigned int>(kv.first >> 32U));
        }
        std::sort(sets.begin(), sets.end());
        sets.erase(std::unique(sets.begin(), sets.end()), sets.end());

        typename AllocT::SlabSweep sweep;
        const unsigned int         saved_sz = thread_chunk_size();
        std::vector<const void*>   tidied; // the instances already tidied (see below)
        for (const auto& [size, key] : sets)
        {
            set_thread_chunk_size(size);
            for (int r = 0; r < CHUNK_PMP_RANK_BOUND; ++r)
            {
                auto& allocator = ChunkT<T>::get_chunk_allocator(r, key);
                // The slabs still in use: their free pages given back. Once per
                // instance: an unregistered (key, size) routes to key 0.
                if (std::find(tidied.begin(), tidied.end(), &allocator) == tidied.end())
                {
                    tidied.push_back(&allocator);
                    (void)allocator.tidy_slabs_in_use();
                }
                // An unregistered (key, size) routes to key 0: the sweep
                // drops the slabs reached twice.
                collected += allocator.collect_empty_slabs(sweep);
            }
        }
        sweep.flush();
        set_thread_chunk_size(saved_sz);
    }
    return collected;
}

template <template <typename> class ChunkT, typename T>
[[nodiscard]] std::size_t ReusePools<ChunkT, T>::reuse_pool_blocks() noexcept
{
    std::size_t blocks = 0;
    for (const auto& ptf : list_block_free_thr)
    {
        blocks += ptf.list.size();
    }
    if constexpr (chunk_config::chunk_dynamic)
    {
        for (const auto& kv : *list_block_free_thr_by_size_)
        {
            for (const auto& ptf : kv.second)
            {
                blocks += ptf.list.size();
            }
        }
    }
    return blocks;
}

template <template <typename> class ChunkT, typename T>
[[nodiscard]] std::size_t ReusePools<ChunkT, T>::reuse_pool_blocks_at([[maybe_unused]] unsigned int size) noexcept
{
    std::size_t blocks = 0;
    if constexpr (chunk_config::chunk_dynamic)
    {
        const auto it = list_block_free_thr_by_size_->find(size);
        if (it != list_block_free_thr_by_size_->end())
        {
            for (const auto& ptf : it->second)
            {
                blocks += ptf.list.size();
            }
        }
    }
    return blocks;
}

template <template <typename> class ChunkT, typename T>
[[nodiscard]] int ReusePools<ChunkT, T>::get_size_buffer_reuse(unsigned int key) noexcept
{
    if (key != 0)
    {
        const auto it = size_buffer_reuse_by_key_->find(key);
        if (it != size_buffer_reuse_by_key_->end())
        {
            return it->second;
        }
    }
    return size_buffer_reuse.load(std::memory_order_relaxed);
}

template <template <typename> class ChunkT, typename T>
[[nodiscard]] std::vector<T*>* ReusePools<ChunkT, T>::find_list_block_free_thr(int thr_rank, unsigned int key) noexcept
{
    const auto         rank = static_cast<size_t>(thr_rank);
    const unsigned int sz   = thread_chunk_size();
    if (key != 0)
    {
        const auto it = list_block_free_thr_keyed_->find(keyed_index(key, sz));
        if (it != list_block_free_thr_keyed_->end() && rank < it->second.size())
        {
            return &it->second[thr_rank].list;
        }
    }
    if constexpr (chunk_config::chunk_dynamic)
    {
        const auto it = list_block_free_thr_by_size_->find(sz);
        if (it != list_block_free_thr_by_size_->end() && rank < it->second.size())
        {
            return &it->second[thr_rank].list;
        }
    }
    if (rank < list_block_free_thr.size())
    {
        return &list_block_free_thr[thr_rank].list;
    }
    return nullptr;
}

template <template <typename> class ChunkT, typename T>
std::vector<T*>& ReusePools<ChunkT, T>::get_list_block_free_thr(int thr_rank, unsigned int key) noexcept
{
    if (key != 0)
    {
        const auto it = list_block_free_thr_keyed_->find(keyed_index(key, thread_chunk_size()));
        if (it != list_block_free_thr_keyed_->end() && static_cast<size_t>(thr_rank) < it->second.size())
        {
            return it->second[thr_rank].list;
        }
    }
    return get_list_block_free_thr(thr_rank);
}

template <template <typename> class ChunkT, typename T>
[[nodiscard]] size_t ReusePools<ChunkT, T>::get_list_block_free_thr_size(int thr_rank) noexcept
{
    // Bounds-checked on purpose: the legacy store may still be unsized here
    // (fast-alloc never parks in it; a geometry born from a re-chunk has no
    // per-size list and falls back to it). Do not index blindly: UB. A
    // missing slot answers 0, which every caller handles.
    if (static_cast<size_t>(thr_rank) >= list_block_free_thr.size())
    {
        if constexpr (chunk_config::chunk_dynamic)
        {
            const auto it = list_block_free_thr_by_size_->find(thread_chunk_size());
            if (it != list_block_free_thr_by_size_->end() && static_cast<size_t>(thr_rank) < it->second.size())
            {
                return it->second[thr_rank].list.size();
            }
        }
        return 0;
    }
    return get_list_block_free_thr(thr_rank).size();
}

template <template <typename> class ChunkT, typename T>
std::vector<T*>& ReusePools<ChunkT, T>::get_list_block_free_thr(int thr_rank) noexcept
{
    if constexpr (chunk_config::chunk_dynamic)
    {
        const unsigned int sz = thread_chunk_size();
        const auto         it = list_block_free_thr_by_size_->find(sz);
        // A rank beyond the per-size vector's size falls back, never indexes.
        if (it != list_block_free_thr_by_size_->end() && static_cast<size_t>(thr_rank) < it->second.size())
        {
            return it->second[thr_rank].list;
        }
        // fallback: legacy single store (single-size / unregistered)
    }
    return list_block_free_thr[thr_rank].list;
}

template <template <typename> class ChunkT, typename T>
std::vector<T*>& ReusePools<ChunkT, T>::get_list_block_free_thr_for_size([[maybe_unused]] int          thr_rank,
                                                                         [[maybe_unused]] unsigned int sz)
{
    if constexpr (chunk_config::chunk_dynamic)
    {
        const auto it = list_block_free_thr_by_size_->find(sz);
        if (it != list_block_free_thr_by_size_->end())
        {
            return it->second[thr_rank].list;
        }
    }

    return list_block_free_thr[thr_rank].list;
}

template <template <typename> class ChunkT, typename T>
void ReusePools<ChunkT, T>::drain_reuse_list_for_size([[maybe_unused]] unsigned int size) noexcept
{
    // Drains ONE element type; other types keep their pools.
    if constexpr (chunk_config::chunk_dynamic)
    {
        // One routine per list set: the shared per-size list, then every
        // keyed list at that size (the array's own lists must empty for its
        // slabs to empty). An unregistered size has nothing parked.
        const auto drain_one = [&](std::vector<PerThreadFreeList>& per_thr)
        {
            const auto n_thr = static_cast<int>(per_thr.size()); // iterate by ITS size, never max_thr

            if constexpr (!chunk_config::use_fast_alloc)
            {
                // std-alloc: parked buffers were svcalloc'd -> svfree each.
                for (int thr = 0; thr < n_thr; ++thr)
                {
                    auto& lst = per_thr[thr].list;
                    for (const T* block : lst)
                    {
                        CHUNK_LEDGER_DISCARD(block, "rechunk_to.drain_reuse");
                        fast_alloc::svfree(block);
                    }
                    lst.clear();
                }
            }
            else
            {
                // fast-alloc: owner-routed drain (free_elt_pow2 is only safe on
                // an owned block). Do not "offer every block to every rank in
                // turn": quadratic (is_free_able_elt() walks the slab chains),
                // and after a zero re-chunk most parked blocks are orphans
                // (clear_st parks in the FREEING thread's list).
                //
                // Every slab names its owning instance and rank in its header
                // (TCAllocator::owner / owner_rank), whatever affinity key it
                // serves, and slab_of() is O(1). So:
                //   1. parallel over q: rank q buckets ITS list by owner into
                //      row q of a q x r matrix (no two ranks write one cell);
                //   2. parallel over r: rank r frees column r into ITS OWN
                //      allocator -- the only mutation of a slab chain, always by
                //      the rank that owns it.
                // Blocks no slab claims (bypass_slabs(), slab torn down) stay
                // parked: an unreachable block is worse than a retained one.
                using AllocT     = std::remove_reference_t<decltype(ChunkT<T>::get_chunk_allocator(0))>;
                AllocT* geometry = nullptr; // any instance of this size: slab_of() needs its geometry only
                {
                    const unsigned int saved_sz = shrink_array::thread_chunk_size();
                    shrink_array::set_thread_chunk_size(size);
                    geometry = &ChunkT<T>::get_chunk_allocator(0, 0u);
                    shrink_array::set_thread_chunk_size(saved_sz);
                }
                if (geometry != nullptr)
                {
                    std::vector<std::vector<std::vector<T*>>> bucket(
                        static_cast<std::size_t>(n_thr), std::vector<std::vector<T*>>(static_cast<std::size_t>(n_thr)));

                    // Passes shared with drain_reuse_list_for_size_team().
                    const auto sort_own_list = [&](int rank) noexcept
                    { drain_sort_rank(per_thr, n_thr, rank, geometry, bucket); };
                    const auto free_own_buckets = [&](int rank) noexcept
                    { drain_free_rank(n_thr, rank, geometry, bucket, /*release_cells=*/true); };

                    if constexpr (chunk_config::not_use_compression_with_ompt)
                    {
                        // Serial call sites. Not under OMPT: opening a region there
                        // would fire compression callbacks that must not run here.
                        pmp::team(pmp::Options{}, [&](pmp::Team& /*team*/) { sort_own_list(CHUNK_PMP_THREAD_RANK); });
                        // Ranks a smaller team did not cover (cheap when empty).
                        for (int q = 0; q < n_thr; ++q)
                        {
                            sort_own_list(q);
                        }
                        pmp::team(pmp::Options{},
                                  [&](pmp::Team& /*team*/) { free_own_buckets(CHUNK_PMP_THREAD_RANK); });
                        for (int r = 0; r < n_thr; ++r)
                        {
                            free_own_buckets(r);
                        }
                    }
                    else
                    {
                        for (int q = 0; q < n_thr; ++q)
                        {
                            sort_own_list(q);
                        }
                        for (int r = 0; r < n_thr; ++r)
                        {
                            free_own_buckets(r);
                        }
                    }
                }
            }
        }; // drain_one
        if (const auto it = list_block_free_thr_by_size_->find(size); it != list_block_free_thr_by_size_->end())
        {
            drain_one(it->second);
        }
        for (auto& kv : *list_block_free_thr_keyed_)
        {
            if (static_cast<unsigned int>(kv.first & 0xffffffffu) == size)
            {
                drain_one(kv.second);
            }
        }
    }
}

template <template <typename> class ChunkT, typename T>
template <typename Alloc>
void ReusePools<ChunkT, T>::drain_sort_rank(std::vector<PerThreadFreeList>& per_thr,
                                            int                             n_thr,
                                            int                             rank,
                                            Alloc*                          geometry,
                                            DrainBuckets&                   bucket,
                                            std::atomic<unsigned char>*     col_dirty) noexcept
{
    if (rank < 0 || rank >= n_thr)
    {
        return;
    }
    auto& lst = per_thr[rank].list;
    if (lst.empty())
    {
        return;
    }
    auto&       row  = bucket[static_cast<std::size_t>(rank)];
    std::size_t kept = 0;
    for (std::size_t i = 0; i < lst.size(); ++i)
    {
        T*          block = lst[i];
        const auto* slab  = geometry->slab_of(block);
        if (slab && slab->owner && slab->owner_rank >= 0 && slab->owner_rank < n_thr)
        {
            row[static_cast<std::size_t>(slab->owner_rank)].push_back(block);
            // Load first: the flag's line is shared by the team, raise it
            // once rather than write it on every block.
            if (col_dirty != nullptr && col_dirty[slab->owner_rank].load(std::memory_order_relaxed) == 0)
            {
                col_dirty[slab->owner_rank].store(1, std::memory_order_relaxed);
            }
        }
        else
        {
            lst[kept++] = block; // no owner claims it: stays parked
        }
    }
    lst.resize(kept);
}

template <template <typename> class ChunkT, typename T>
template <typename Alloc>
void ReusePools<ChunkT, T>::drain_free_rank(int           n_thr,
                                            int           rank,
                                            Alloc*        geometry,
                                            DrainBuckets& bucket,
                                            bool          release_cells) noexcept
{
    if (rank < 0 || rank >= n_thr)
    {
        return;
    }
    // Route through the block's OWNING instance (its slab names it), never
    // get_chunk_allocator(): that keys on the ambient size, and a block freed
    // through another size's allocator corrupts that slab's free-list.
    for (int q = 0; q < n_thr; ++q)
    {
        auto& cell = bucket[static_cast<std::size_t>(q)][static_cast<std::size_t>(rank)];
        if (cell.empty())
        {
            continue;
        }
        for (T* block : cell)
        {
            CHUNK_FREEPATH_INC(freed);
            CHUNK_LEDGER_DISCARD(block, "rechunk_to.drain_reuse.owner");
            auto* slab  = geometry->slab_of(block);
            auto* owner = static_cast<Alloc*>(slab->owner);
            owner->free_elt_pow2(block, slab);
        }
        cell.clear();
        if (release_cells)
        {
            cell.shrink_to_fit();
        }
    }
}

template <template <typename> class ChunkT, typename T>
void ReusePools<ChunkT, T>::drain_reuse_list_for_size_team([[maybe_unused]] pmp::Team&        team,
                                                           [[maybe_unused]] unsigned int      size,
                                                           [[maybe_unused]] TeamDrainScratch& scratch) noexcept
{
    if constexpr (chunk_config::chunk_dynamic && chunk_config::use_fast_alloc && build_config::threaded)
    {
        using AllocT = std::remove_reference_t<decltype(ChunkT<T>::get_chunk_allocator(0))>;
        team.single(
            [&]
            {
                scratch.sets.clear();
                if (const auto it = list_block_free_thr_by_size_->find(size); it != list_block_free_thr_by_size_->end())
                {
                    scratch.sets.push_back(&it->second);
                }
                for (auto& kv : *list_block_free_thr_keyed_)
                {
                    if (static_cast<unsigned int>(kv.first & 0xffffffffu) == size)
                    {
                        scratch.sets.push_back(&kv.second);
                    }
                }
                std::size_t count = 0;
                for (const auto* set : scratch.sets)
                {
                    count = std::max(count, set->size());
                }
                if (scratch.bucket.size() < count)
                {
                    scratch.bucket.assign(count, std::vector<std::vector<T*>>(count));
                }
                if (scratch.n_flags < count)
                {
                    scratch.held    = std::make_unique<std::atomic<unsigned char>[]>(count);
                    scratch.dirty   = std::make_unique<std::atomic<unsigned char>[]>(count);
                    scratch.n_flags = count;
                }
                for (std::size_t r = 0; r < scratch.n_flags; ++r)
                {
                    scratch.held[r].store(0, std::memory_order_relaxed);
                    scratch.dirty[r].store(0, std::memory_order_relaxed);
                }
                const unsigned int saved_sz = shrink_array::thread_chunk_size();
                shrink_array::set_thread_chunk_size(size);
                scratch.geometry = &ChunkT<T>::get_chunk_allocator(0, 0u);
                shrink_array::set_thread_chunk_size(saved_sz);
            }); // implicit barrier: scratch is complete for every thread
        auto* const geometry = static_cast<AllocT*>(scratch.geometry);
        const int   me       = CHUNK_PMP_THREAD_RANK;
        // rechunk=debug: this thread's own-rank passes, barriers excluded
        // (drain_probe.h). Read after the barrier above: the master arms it
        // before entering the drain.
        using ProbeClock = std::chrono::steady_clock;
        const bool probe = internal::g_drain_probe.armed.load(std::memory_order_relaxed);
        const auto ns    = [](ProbeClock::time_point from) noexcept
        { return std::chrono::duration_cast<std::chrono::nanoseconds>(ProbeClock::now() - from).count(); };
        auto* const held  = scratch.held.get();
        auto* const dirty = scratch.dirty.get();
        if (me >= 0 && static_cast<std::size_t>(me) < scratch.n_flags)
        {
            held[me].store(1, std::memory_order_relaxed); // read after the first set's barrier
        }
        for (auto* set : scratch.sets)
        {
            auto&      per_thr = *set;
            const int  n_thr   = static_cast<int>(per_thr.size());
            const auto is_set  = [](const std::atomic<unsigned char>* flags, int rank) noexcept
            { return flags[rank].load(std::memory_order_relaxed) != 0; };
            const auto t_sort = probe ? ProbeClock::now() : ProbeClock::time_point{};
            drain_sort_rank(per_thr, n_thr, me, geometry, scratch.bucket, dirty);
            const long long sort_ns = probe ? ns(t_sort) : 0;
            team.barrier();
            team.single(
                [&]
                {
                    // Ranks no team member holds (a smaller team than the
                    // capacity); cheap for the lists just emptied.
                    for (int q = 0; q < n_thr; ++q)
                    {
                        if (!is_set(held, q))
                        {
                            drain_sort_rank(per_thr, n_thr, q, geometry, scratch.bucket, dirty);
                        }
                    }
                }); // implicit barrier: every row is sorted, every dirty flag raised
            long long free_ns = 0;
            long long blocks  = 0;
            if (me >= 0 && me < n_thr && is_set(dirty, me))
            {
                if (probe)
                {
                    for (int q = 0; q < n_thr; ++q)
                    {
                        blocks += static_cast<long long>(
                            scratch.bucket[static_cast<std::size_t>(q)][static_cast<std::size_t>(me)].size());
                    }
                }
                const auto t_free = probe ? ProbeClock::now() : ProbeClock::time_point{};
                drain_free_rank(n_thr, me, geometry, scratch.bucket, /*release_cells=*/false);
                free_ns = probe ? ns(t_free) : 0;
            }
            if (probe && me >= 0 && me < n_thr)
            {
                internal::g_drain_probe.report(sort_ns, free_ns, blocks); // before the barrier: read after it
            }
            team.barrier();
            team.single(
                [&]
                {
                    for (int r = 0; r < n_thr; ++r)
                    {
                        if (!is_set(held, r) && is_set(dirty, r))
                        {
                            drain_free_rank(n_thr, r, geometry, scratch.bucket, /*release_cells=*/false);
                        }
                    }
                    for (int r = 0; r < n_thr; ++r) // the next set starts clean
                    {
                        dirty[r].store(0, std::memory_order_relaxed);
                    }
                }); // implicit barrier: this set is drained
        }
    }
}

} // namespace shrink_array
