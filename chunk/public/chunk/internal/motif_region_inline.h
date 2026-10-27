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
    if (table == nullptr)
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
        // The hash mixes the count in; the entry's own count (its segment's,
        // set before any of its entries was published) confirms it.
        if (table->slots[i].hash.load(std::memory_order_relaxed) == h && entry_count_of(entry) == count &&
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
    auto&               entry = probation_[static_cast<std::size_t>(h >> (64U - probation_bits))];
    const auto          check = static_cast<std::uint16_t>(((h >> (56U - probation_bits)) & 0xFFU) << 8U);
    const std::uint16_t seen  = entry.load(std::memory_order_relaxed);
    if ((seen & 0xFF00U) != check || (seen & 0xFFU) == 0 || (seen & 0xFFU) == tag)
    {
        // A first sighting (or another motif's entry, taken over), or the same
        // sighter again. A race only admits one sighting early.
        entry.store(static_cast<std::uint16_t>(check | tag), std::memory_order_relaxed);
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
        reserve();
    }
    if (const T* const entry = find(motif, count, h)) // registered meanwhile
    {
        return entry;
    }
    T* const entry = take_slot(count);
    if (entry == nullptr)
    {
        full_.fetch_add(1, std::memory_order_relaxed); // no segment left for this size
        return nullptr;
    }
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
    if (!contains(ptr))
    {
        return 0;
    }
    const std::byte* const base = begin_.load(std::memory_order_relaxed);
    const std::size_t      off  = ::fast_alloc::address_bits(ptr) - ::fast_alloc::address_bits(base);
    const Segment&         seg  = segments_[off / segment_bytes];
    const std::size_t      slot = seg.bytes == 0 ? seg.carved : (off % segment_bytes) / seg.bytes;
    return slot < seg.carved ? segment_ids()[off / segment_bytes][slot] : 0;
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
            size_class(entry_count_of(entry_by_id()[id])).free_ids.push_back(static_cast<std::uint32_t>(id));
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
void pattern::MotifRegion<T>::reserve()
{
    void* const raw =
        ::mmap(nullptr, region_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (raw == MAP_FAILED)
    {
        CHUNK_INVARIANT_FAILED("pattern::MotifRegion: cannot reserve {} bytes of address space", region_bytes);
    }
    live().assign(1, 0); // identifier 0: none
    entry_by_id().assign(1, nullptr);
    segment_ids().assign(segment_count, {});
    begin_.store(static_cast<std::byte*>(raw), std::memory_order_release);
}

template <typename T>
pattern::MotifRegion<T>::SizeClass& pattern::MotifRegion<T>::size_class(std::size_t count)
{
    for (auto& cls : size_classes())
    {
        if (cls.count == count)
        {
            return cls;
        }
    }
    size_classes().push_back(SizeClass{count, segment_count, {}});
    sizes_.store(size_classes().size(), std::memory_order_relaxed);
    return size_classes().back();
}

template <typename T>
T* pattern::MotifRegion<T>::take_slot(std::size_t count)
{
    SizeClass& cls = size_class(count);
    if (!cls.free_ids.empty())
    {
        const std::uint32_t id = cls.free_ids.back();
        cls.free_ids.pop_back();
        return const_cast<T*>(entry_by_id()[id]);
    }
    const std::size_t bytes = (count * sizeof(T) + entry_align - 1) / entry_align * entry_align;
    if (bytes > segment_bytes)
    {
        return nullptr; // an entry larger than a segment: CHUNK_MOTIF_SEGMENT_BYTES too small
    }
    if (cls.open == segment_count || (segments_[cls.open].carved + 1) * bytes > segment_bytes)
    {
        if (segments_used_ == segment_count)
        {
            return nullptr;
        }
        cls.open      = segments_used_++;
        Segment& seg  = segments_[cls.open];
        seg.bytes     = bytes;
        // Before any entry of the segment is published (index_publish's release).
        seg.count.store(count, std::memory_order_relaxed);
    }
    Segment&            seg = segments_[cls.open];
    const std::uint32_t id  = static_cast<std::uint32_t>(next_id_.load(std::memory_order_relaxed));
    std::byte* const    at  = begin_.load(std::memory_order_relaxed) + cls.open * segment_bytes + seg.carved * bytes;
    ++seg.carved;
    segment_ids()[cls.open].push_back(id);
    live().push_back(0);
    entry_by_id().push_back(reinterpret_cast<const T*>(at));
    next_id_.store(id + std::size_t{1}, std::memory_order_relaxed);
    return reinterpret_cast<T*>(at);
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
    const auto& flags = live();
    for (std::size_t id = 1; id < flags.size(); ++id)
    {
        if (flags[id] != 0)
        {
            const T* const entry = entry_by_id()[id];
            index_publish(entry, hash(entry, entry_count_of(entry)));
        }
    }
}

} // namespace shrink_array
