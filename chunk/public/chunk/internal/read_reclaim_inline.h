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
#include "chunk/internal/read_reclaim.h" // the class these bodies belong to

namespace shrink_array
{

// ---------------------------------------------------------------------------
// shrink_array::ReadSection
// ---------------------------------------------------------------------------

inline read_reclaim::ReadSection::ReadSection() noexcept
{
    if constexpr (enabled)
    {
        ReaderSlot* slot = t_slot;
        if (slot == nullptr) [[unlikely]]
        {
            slot = claim_slot();
        }
        if (slot != nullptr) [[likely]]
        {
            slot_ = slot;
            seq_  = slot->seq.load(std::memory_order_relaxed) + 1; // odd
            // RELEASE, not relaxed: the reclaimer, having seen this slot odd, waits
            // for it to MOVE, and the value it then reads may be THIS store -- the
            // next read's entry. Release orders the previous read's element load
            // before it; relaxed, by C++20 (no release sequence through a later
            // relaxed store of the same thread), it would order nothing, and only
            // the AArch64 / x86 mappings would keep that load before the free. An
            // stlr instead of an str: +1.9 % on access_cost's worst read case at 1
            // thread on Grace, within its spread, nothing measurable at 64
            // (formal/reader_reclaim.c, MUT_ENTER_RELAXED; docs/formal_models.md F3).
            slot->seq.store(seq_, std::memory_order_release);
            light_fence();
        }
        else
        {
            g_overflow_readers.fetch_add(1, std::memory_order_seq_cst);
            // The announce needs its light side like the slot path: a seq_cst
            // RMW followed by the acquire load of `values` is not a Dekker
            // announce in C11 (store buffering against the reclaimer's acquire
            // load of the counter), though it is on the AArch64 and x86
            // mappings (formal/reader_reclaim.c, MUT_OVERFLOW_NO_FENCE).
            light_fence();
        }
    }
}

inline read_reclaim::ReadSection::~ReadSection()
{
    if constexpr (enabled)
    {
        if (slot_ != nullptr) [[likely]]
        {
            slot_->seq.store(seq_ + 1, std::memory_order_release); // even: the element load is done
        }
        else
        {
            g_overflow_readers.fetch_sub(1, std::memory_order_release);
        }
    }
}

// ---------------------------------------------------------------------------
// shrink_array::PassScope
// ---------------------------------------------------------------------------

inline read_reclaim::PassScope::PassScope() noexcept
{
    if constexpr (enabled)
    {
        if (t_batch == nullptr)
        {
            owner_ = true;
            storage().clear();
            t_batch = &storage();
        }
    }
}

inline read_reclaim::PassScope::~PassScope()
{
    if constexpr (enabled)
    {
        if (!owner_)
        {
            return;
        }
        t_batch     = nullptr;
        auto& batch = storage();
        if (batch.empty())
        {
            return;
        }
        wait_for_readers();
        const unsigned int saved_size = t_active_chunk_size;
        const unsigned int saved_key  = t_alloc_key;
        for (const Deferred& deferred : batch)
        {
            t_active_chunk_size = deferred.chunk_size;
            t_alloc_key         = deferred.alloc_key;
            deferred.release(deferred.chunk, deferred.thr, deferred.block);
        }
        t_active_chunk_size = saved_size;
        t_alloc_key         = saved_key;
        batch.clear();
    }
}

} // namespace shrink_array
