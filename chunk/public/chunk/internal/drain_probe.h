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
 * @file drain_probe.h
 * @brief The per-thread split of the team drain at a re-chunk's slice boundary.
 *
 * @details ReusePools::drain_reuse_list_for_size_team() reports, per thread and
 *          per list set, the time of the passes over its OWN rank -- sorting
 *          its parked blocks by owner, freeing the blocks its rank owns -- and
 *          the number of blocks it freed, barriers excluded. Armed by the
 *          re-chunk's slice boundary under CHUNK_LOG=rechunk=debug only
 *          (Refill::slice_boundary), folded into RechunkSliceTimes there and
 *          printed as the "drain split" line. Kept apart from the trace header
 *          so the pools need no logging include.
 */

#include <atomic> // std::atomic

namespace shrink_array::internal
{
/// @brief One boundary's drain, shared by its team.
struct DrainProbe
{
    std::atomic<bool>      armed{false};  ///< set by the master before the drain, read after its first barrier
    std::atomic<long long> sort_max{0};   ///< ns: the longest own-rank sort pass
    std::atomic<long long> sort_sum{0};   ///< ns: the own-rank sort passes, summed
    std::atomic<long long> free_max{0};   ///< ns: the longest own-rank free pass
    std::atomic<long long> free_sum{0};   ///< ns: the own-rank free passes, summed
    std::atomic<long long> blocks_max{0}; ///< the most blocks one rank freed
    std::atomic<long long> blocks_sum{0}; ///< the blocks freed, summed
    std::atomic<int>       n{0};          ///< reports (threads x list sets)

    /// @brief Zero the slots (master, before the drain).
    void reset() noexcept
    {
        for (auto* slot : {&sort_max, &sort_sum, &free_max, &free_sum, &blocks_max, &blocks_sum})
        {
            slot->store(0, std::memory_order_relaxed);
        }
        n.store(0, std::memory_order_relaxed);
    }

    /// @brief One thread's passes over one list set.
    void report(long long sort_ns, long long free_ns, long long blocks) noexcept
    {
        raise(sort_max, sort_ns);
        raise(free_max, free_ns);
        raise(blocks_max, blocks);
        sort_sum.fetch_add(sort_ns, std::memory_order_relaxed);
        free_sum.fetch_add(free_ns, std::memory_order_relaxed);
        blocks_sum.fetch_add(blocks, std::memory_order_relaxed);
        n.fetch_add(1, std::memory_order_relaxed);
    }

  private:
    static void raise(std::atomic<long long>& slot, long long value) noexcept
    {
        long long cur = slot.load(std::memory_order_relaxed);
        while (value > cur && !slot.compare_exchange_weak(cur, value, std::memory_order_relaxed))
        {
        }
    }
};

/// @brief The drain probe of the re-chunk in progress (re-chunks are serial).
inline DrainProbe g_drain_probe;
} // namespace shrink_array::internal
