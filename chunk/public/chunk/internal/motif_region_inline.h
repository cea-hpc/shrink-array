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
#include "chunk/internal/motif_region.h" // the class these bodies belong to

namespace shrink_array
{

// ---------------------------------------------------------------------------
// shrink_array::MotifRegion<T>
// ---------------------------------------------------------------------------

template <typename T>
[[nodiscard]] std::uint64_t pattern::MotifRegion<T>::hash(const T* motif, std::size_t count) noexcept
{
    using U         = std::make_unsigned_t<T>;
    std::uint64_t h = 0x9E3779B97F4A7C15ULL ^ count;
    for (std::size_t j = 0; j < count; ++j)
    {
        h ^= static_cast<std::uint64_t>(static_cast<U>(motif[j]));
        h *= 0xBF58476D1CE4E5B9ULL;
        h ^= h >> 31U;
    }
    return h;
}

template <typename T>
[[nodiscard]] const T* pattern::MotifRegion<T>::find(const T* motif, std::size_t count, std::uint64_t h) noexcept
{
    const IndexTable* const table = index_.load(std::memory_order_acquire);
    if (table == nullptr || count != entry_count_.load(std::memory_order_relaxed))
    {
        return nullptr;
    }
    const std::size_t mask = table->cap - 1;
    for (std::size_t probe = 0, i = h & mask; probe < table->cap; ++probe, i = (i + 1) & mask)
    {
        const T* const entry = table->slots[i].entry.load(std::memory_order_acquire);
        if (entry == nullptr)
        {
            return nullptr;
        }
        if (table->slots[i].hash.load(std::memory_order_relaxed) == h &&
            std::memcmp(entry, motif, count * sizeof(T)) == 0)
        {
            return entry;
        }
    }
    return nullptr;
}

template <typename T>
[[nodiscard]] const T*
pattern::MotifRegion<T>::find_or_admit(const T* motif, std::size_t count, std::uint64_t h, std::uint8_t tag)
{
    if (const T* const entry = find(motif, count, h))
    {
        return entry;
    }
    auto&              counter = probation_[static_cast<std::size_t>(h >> (64U - probation_bits))];
    const std::uint8_t seen    = counter.load(std::memory_order_relaxed);
    if (seen == 0 || seen == tag)
    {
        counter.store(tag, std::memory_order_relaxed); // a race only admits one sighting early
        first_seen_.fetch_add(1, std::memory_order_relaxed);
        return nullptr;
    }
    return find_or_register(motif, count, h);
}

template <typename T>
[[nodiscard]] const T* pattern::MotifRegion<T>::find_or_register(const T* motif, std::size_t count, std::uint64_t h)
{
    const std::scoped_lock lock(mutex_);
    if (begin_.load(std::memory_order_relaxed) == nullptr)
    {
        reserve(count);
    }
    else if (count != entry_count_.load(std::memory_order_relaxed))
    {
        CHUNK_INVARIANT_FAILED("pattern::MotifRegion: an entry of {} elements, the region holds {} per entry", count,
                               entry_count_.load(std::memory_order_relaxed));
    }
    if (const T* const entry = find(motif, count, h)) // registered meanwhile
    {
        return entry;
    }
    T* const entry = take_slot();
    std::memcpy(entry, motif, count * sizeof(T));
    live().at(id_of(entry)) = 1;
    index_publish(entry, h); // after the copy: a reader that finds it sees the motif
    entries_.fetch_add(1, std::memory_order_relaxed);
    registered_.fetch_add(1, std::memory_order_relaxed);
    return entry;
}

template <typename T>
[[nodiscard]] std::uint32_t pattern::MotifRegion<T>::id_of(const void* ptr) noexcept
{
    const std::byte* const base = begin_.load(std::memory_order_acquire);
    if (base == nullptr || !contains(ptr))
    {
        return 0;
    }
    const std::size_t bytes = entry_bytes_.load(std::memory_order_relaxed);
    return static_cast<std::uint32_t>((::fast_alloc::address_bits(ptr) - ::fast_alloc::address_bits(base)) / bytes + 1);
}

template <typename T>
std::size_t pattern::MotifRegion<T>::reclaim_unmarked(const std::vector<std::uint8_t>& referenced)
{
    const std::scoped_lock lock(mutex_);
    reclaims_.fetch_add(1, std::memory_order_relaxed);
    auto&             flags = live();
    std::size_t       freed = 0;
    const std::size_t ids   = ids_carved();
    for (std::size_t id = 1; id <= ids; ++id)
    {
        if (flags[id] != 0 && (id >= referenced.size() || referenced[id] == 0))
        {
            flags[id] = 0;
            free_ids().push_back(static_cast<std::uint32_t>(id));
            ++freed;
        }
    }
    index_rebuild();
    for (auto& counter : probation_)
    {
        counter.store(0, std::memory_order_relaxed);
    }
    entries_.fetch_sub(freed, std::memory_order_relaxed);
    freed_.fetch_add(freed, std::memory_order_relaxed);
    return freed;
}

template <typename T>
void pattern::MotifRegion<T>::reserve(std::size_t count)
{
    void* const raw =
        ::mmap(nullptr, region_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (raw == MAP_FAILED)
    {
        CHUNK_INVARIANT_FAILED("pattern::MotifRegion: cannot reserve {} bytes of address space", region_bytes);
    }
    entry_count_.store(count, std::memory_order_relaxed);
    entry_bytes_.store((count * sizeof(T) + entry_align - 1) / entry_align * entry_align, std::memory_order_relaxed);
    live().assign(1, 0); // identifier 0: none
    begin_.store(static_cast<std::byte*>(raw), std::memory_order_release);
}

template <typename T>
T* pattern::MotifRegion<T>::take_slot()
{
    std::byte* const  base  = begin_.load(std::memory_order_relaxed);
    const std::size_t bytes = entry_bytes_.load(std::memory_order_relaxed);
    if (!free_ids().empty())
    {
        const std::uint32_t id = free_ids().back();
        free_ids().pop_back();
        return reinterpret_cast<T*>(base + static_cast<std::size_t>(id - 1) * bytes);
    }
    const std::size_t used = used_.load(std::memory_order_relaxed);
    if (used + bytes > region_bytes)
    {
        CHUNK_INVARIANT_FAILED("pattern::MotifRegion: the region is full ({} entries of {} bytes, "
                               "CHUNK_MOTIF_REGION_BYTES = {})",
                               entries(), bytes, region_bytes);
    }
    used_.store(used + bytes, std::memory_order_relaxed);
    live().push_back(0);
    return reinterpret_cast<T*>(base + used);
}

template <typename T>
void pattern::MotifRegion<T>::index_publish(const T* entry, std::uint64_t h)
{
    IndexTable* table = index_.load(std::memory_order_relaxed);
    if (table == nullptr || 2 * (table->used + 1) > table->cap)
    {
        table = index_grow(table);
    }
    const std::size_t mask = table->cap - 1;
    std::size_t       i    = h & mask;
    while (table->slots[i].entry.load(std::memory_order_relaxed) != nullptr)
    {
        i = (i + 1) & mask;
    }
    table->slots[i].hash.store(h, std::memory_order_relaxed);
    table->slots[i].entry.store(entry, std::memory_order_release);
    ++table->used;
}

template <typename T>
pattern::MotifRegion<T>::IndexTable* pattern::MotifRegion<T>::index_grow(const IndexTable* old)
{
    const std::size_t cap = old == nullptr ? 64 : 2 * old->cap;
    auto              nt  = std::make_unique<IndexTable>(cap);
    if (old != nullptr)
    {
        for (const auto& slot : old->slots)
        {
            if (const T* const entry = slot.entry.load(std::memory_order_relaxed))
            {
                const std::uint64_t h = slot.hash.load(std::memory_order_relaxed);
                std::size_t         i = h & (cap - 1);
                while (nt->slots[i].entry.load(std::memory_order_relaxed) != nullptr)
                {
                    i = (i + 1) & (cap - 1);
                }
                nt->slots[i].hash.store(h, std::memory_order_relaxed);
                nt->slots[i].entry.store(entry, std::memory_order_relaxed);
                ++nt->used;
            }
        }
    }
    IndexTable* const raw = nt.get();
    generations().push_back(std::move(nt));
    index_.store(raw, std::memory_order_release);
    return raw;
}

template <typename T>
void pattern::MotifRegion<T>::index_rebuild()
{
    index_.store(nullptr, std::memory_order_release);
    generations().clear();
    std::byte* const  base  = begin_.load(std::memory_order_relaxed);
    const std::size_t bytes = entry_bytes_.load(std::memory_order_relaxed);
    const std::size_t count = entry_count_.load(std::memory_order_relaxed);
    const auto&       flags = live();
    for (std::size_t id = 1; id < flags.size(); ++id)
    {
        if (flags[id] != 0)
        {
            const T* const entry = reinterpret_cast<const T*>(base + (id - 1) * bytes);
            index_publish(entry, hash(entry, count));
        }
    }
}

} // namespace shrink_array
