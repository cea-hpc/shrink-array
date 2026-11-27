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
#include <atomic> // std::memory_order_relaxed, std::memory_order_release, std::memory_order_seq_cst

#include "chunk/internal/read_reclaim.h" // the class these bodies belong to

namespace shrink_array
{

// ---------------------------------------------------------------------------
// shrink_array::AnnouncedSection
// ---------------------------------------------------------------------------

inline read_reclaim::AnnouncedSection::AnnouncedSection() noexcept
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

inline read_reclaim::AnnouncedSection::~AnnouncedSection()
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
// shrink_array::ReadSection
// ---------------------------------------------------------------------------

inline read_reclaim::ReadSection::ReadSection() noexcept
{
    if constexpr (optimistic)
    {
        // Acquire: a pass whose bump this read sees has its swaps visible to the
        // `values` load that follows.
        seq_ = g_reclaim_epoch.load(std::memory_order_acquire);
    }
}

/// @details Acquire: the swaps of the pass that bumped the epoch are visible to the
///          read that follows, which therefore loads the new pointer.
[[gnu::noinline, gnu::cold]] inline std::uint64_t read_reclaim::note_retry() noexcept
{
    g_optimistic_retries.fetch_add(1, std::memory_order_relaxed);
    return g_reclaim_epoch.load(std::memory_order_acquire);
}

template <typename V>
[[nodiscard]] CHUNK_FORCE_INLINE bool read_reclaim::ReadSection::retry_after(const V& value) const noexcept
{
    if constexpr (optimistic)
    {
        static_assert(std::is_trivially_copyable_v<V>, "retry_after: the dependency is taken on the value's bytes");
        // Every word of the value: a value wider than a register (std::complex<double>)
        // is two loads, and each must come before the re-check.
        constexpr std::size_t words = (sizeof(V) + sizeof(std::uintptr_t) - 1) / sizeof(std::uintptr_t);
        std::uintptr_t        part[words]{};
        std::memcpy(part, &value, sizeof(V));
        std::uintptr_t raw = 0;
        for (std::size_t w = 0; w < words; ++w)
        {
            raw |= part[w];
        }
        // zero = raw ^ raw, hidden from the compiler: 0 at run time, an address
        // dependency on the element for the CPU (eor on AArch64), so the epoch's
        // load is ordered after the element's.
        std::uintptr_t opaque = raw;
        __asm__("" : "+r"(opaque));
        const std::uintptr_t zero  = opaque ^ raw;
        const auto*          epoch = reinterpret_cast<const std::atomic<std::uint64_t>*>(
            reinterpret_cast<std::uintptr_t>(&g_reclaim_epoch) + zero);
        if (epoch->load(std::memory_order_relaxed) != seq_) [[unlikely]]
        {
            seq_ = note_retry();
            return true;
        }
    }
    return false;
}

[[nodiscard]] CHUNK_FORCE_INLINE bool read_reclaim::ReadSection::retry() const noexcept
{
    if constexpr (optimistic && !fold_block)
    {
        std::atomic_thread_fence(std::memory_order_acquire); // the block's loads before the re-check
        if (g_reclaim_epoch.load(std::memory_order_relaxed) != seq_) [[unlikely]]
        {
            seq_ = note_retry();
            return true;
        }
    }
    else if constexpr (optimistic)
    {
        // As retry_after(), the dependency on the OR of every value loaded (fold()).
        std::uintptr_t opaque = acc_;
        __asm__("" : "+r"(opaque));
        const std::uintptr_t zero  = opaque ^ acc_;
        const auto*          epoch = reinterpret_cast<const std::atomic<std::uint64_t>*>(
            reinterpret_cast<std::uintptr_t>(&g_reclaim_epoch) + zero);
        acc_ = 0;
        if (epoch->load(std::memory_order_relaxed) != seq_) [[unlikely]]
        {
            seq_ = note_retry();
            return true;
        }
    }
    return false;
}

template <typename V>
CHUNK_FORCE_INLINE void read_reclaim::ReadSection::fold(const V& value) const noexcept
{
    if constexpr (optimistic && fold_block)
    {
        static_assert(std::is_trivially_copyable_v<V>, "fold: the dependency is taken on the value's bytes");
        constexpr std::size_t words = (sizeof(V) + sizeof(std::uintptr_t) - 1) / sizeof(std::uintptr_t);
        std::uintptr_t        raw[words]{};
        std::memcpy(raw, &value, sizeof(V));
        for (std::size_t w = 0; w < words; ++w)
        {
            acc_ |= raw[w];
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
