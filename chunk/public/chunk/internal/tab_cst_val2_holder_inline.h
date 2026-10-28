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
#include "chunk/internal/tab_cst_val2_holder.h" // the class these bodies belong to

namespace shrink_array
{

// ---------------------------------------------------------------------------
// shrink_array::SecondValueRef<T>
// ---------------------------------------------------------------------------

template <typename T>
[[nodiscard]] T SecondValueRef<T>::get() const noexcept
{
    if constexpr (SecondValueTraits<T>::indexed)
    {
        return TabCstVal2Holder<T>::value_of_id(*slot);
    }
    else
    {
        return *slot;
    }
}

template <typename T>
void SecondValueRef<T>::set(T value) const
{
    if constexpr (SecondValueTraits<T>::indexed)
    {
        // Up to the record: a reclaim must not run in between (LookupGuard).
        const typename GlobalVecHolder<T>::LookupGuard in_flight;
        *slot = TabCstVal2Holder<T>::id_for_value(value);
    }
    else
    {
        *slot = value;
    }
}

template <typename T>
void SecondValueRef<T>::set_from_array(const T* src) const noexcept
{
    // bool has no registry (TabCstVal2Holder<bool>): value_of_ptr would index
    // value_by_id with an address-sized id. Never called for bool today.
    static_assert(!std::is_same_v<T, bool>, "SecondValueRef<bool>::set_from_array: bool has no registry");
    if constexpr (SecondValueTraits<T>::indexed)
    {
        *slot = TabCstVal2Holder<T>::id_of_ptr(src);
    }
    else
    {
        // NOT *src: nothing is stored at a slot, so that would record 0 and mark the
        // chunk constant at the wrong value, silently.
        *slot = GlobalVecHolder<T>::value_of_ptr(src);
    }
}

template <typename T>
void SecondValueRef<T>::set_from_array_release(const T* src) const noexcept
{
    // pattern::Array only, whose element types have a plain slot (static_assert there).
    static_assert(!SecondValueTraits<T>::indexed, "set_from_array_release: an indexed slot holds an identifier");
    std::atomic_ref<SlotT>(*slot).store(GlobalVecHolder<T>::value_of_ptr(src), std::memory_order_release);
}

// ---------------------------------------------------------------------------
// shrink_array::GlobalVecHolder<T>
// ---------------------------------------------------------------------------

template <typename T>
[[nodiscard]] std::size_t GlobalVecHolder<T>::count_entries() noexcept
{
    std::size_t count = 0;
    for (Segment<T>* s = head.load(std::memory_order_acquire); s != nullptr; s = s->next)
    {
        count += static_cast<std::size_t>(s->used);
    }
    return count;
}

template <typename T>
[[nodiscard]] SecondValueStats GlobalVecHolder<T>::stats() noexcept
{
    constexpr auto   rx = std::memory_order_relaxed;
    SecondValueStats stats;
    stats.handed_out       = ids_handed_out();
    stats.free             = free_id_count();
    stats.free             = stats.free < stats.handed_out ? stats.free : stats.handed_out; // two relaxed reads
    stats.capacity         = id_capacity();
    stats.degraded         = index_degraded.load(rx);
    stats.reclaims         = reclaim_runs.load(rx);
    stats.freed            = reclaim_freed.load(rx);
    stats.refused_parallel = refused_parallel.load(rx);
    stats.refused_rechunk  = refused_rechunk.load(rx);
    stats.refused_degraded = refused_degraded.load(rx);
    return stats;
}

// ---------------------------------------------------------------------------
// shrink_array::GlobalVecHolder<T>::LookupGuard
// ---------------------------------------------------------------------------

template <typename T>
GlobalVecHolder<T>::LookupGuard::LookupGuard() noexcept
{
    if constexpr (build_config::any_check)
    {
        // seq_cst on both sides (ReclaimScope): each increments its own
        // counter then reads the other's, so one of the two always sees
        // the other.
        lookups_in_flight.fetch_add(1, std::memory_order_seq_cst);
        if (reclaims_running.load(std::memory_order_seq_cst) != 0) [[unlikely]]
        {
            CHUNK_INVARIANT_FAILED("cst_val2: second-value lookup or registration during a reclaim "
                                   "(reclaim_second_values() needs a sequential point)");
        }
    }
}

// ---------------------------------------------------------------------------
// shrink_array::GlobalVecHolder<T>::ReclaimScope
// ---------------------------------------------------------------------------

template <typename T>
void GlobalVecHolder<T>::ReclaimScope::check_no_lookup(const char* when) noexcept
{
    if (const int in_flight = lookups_in_flight.load(std::memory_order_seq_cst); in_flight != 0) [[unlikely]]
    {
        CHUNK_INVARIANT_FAILED("cst_val2: reclaim {} with {} second-value lookup(s) or registration(s) "
                               "in flight (reclaim_second_values() needs a sequential point)",
                               when, in_flight);
    }
}

// ---------------------------------------------------------------------------
// shrink_array::GlobalVecHolder<T>
// ---------------------------------------------------------------------------

template <typename T>
std::size_t GlobalVecHolder<T>::reclaim_unmarked(const std::vector<std::uint8_t>& referenced)
{
    const ReclaimScope     scope;            // check builds: no lookup in flight (nests in the caller's)
    const std::scoped_lock lk(insert_mutex); // the contract excludes appenders: uncontended

    std::vector<T*> kept;
    std::size_t     freed = 0;
    for (const Segment<T>* s = head.load(std::memory_order_acquire); s != nullptr; s = s->next)
    {
        for (int i = 0; i < s->used; ++i)
        {
            T* const            slot = s->slots[i];
            const std::uint32_t id   = slot_id(slot);
            if (id < referenced.size() && referenced[id] == 0)
            {
                free_ids.push_back(id);
                free_ids_count.fetch_add(1, std::memory_order_relaxed);
                ++freed;
            }
            else
            {
                kept.push_back(slot);
            }
        }
    }
    if (freed == 0)
    {
        return 0;
    }

    // The chain again, packed: occupancy stays contiguous (Segment's note).
    auto*       first = new Segment<T>();
    Segment<T>* last  = first;
    for (T* slot : kept)
    {
        if (last->used == Segment<T>::slots_per_segment)
        {
            last->next = new Segment<T>();
            last       = last->next;
        }
        last->slots[last->used] = slot;
        ++last->used;
    }
    const Segment<T>* old = head.exchange(first, std::memory_order_acq_rel);
    while (old != nullptr)
    {
        const Segment<T>* next = old->next;
        delete old;
        old = next;
    }

    // The index again, from the survivors only: an entry left for a freed
    // slot would answer a lookup of the old value with an identifier that
    // has gone to another one.
    index_clear(); // also drops index_degraded and the cached tail
    for (T* slot : kept)
    {
        index_publish(value_of_ptr(slot), slot);
    }
    tail = last;
    return freed;
}

template <typename T>
void GlobalVecHolder<T>::register_marker(MarkFn mark) noexcept
{
    for (auto& slot : markers)
    {
        MarkFn expected = nullptr;
        if (slot.compare_exchange_strong(expected, mark, std::memory_order_acq_rel) || expected == mark)
        {
            return;
        }
    }
    CHUNK_INVARIANT_FAILED("GlobalVecHolder::register_marker: more than {} members of the zero_any family "
                           "share the second-value registry",
                           max_markers);
}

template <typename T>
void GlobalVecHolder<T>::mark_all(std::vector<std::uint8_t>& referenced)
{
    for (const auto& slot : markers)
    {
        if (const MarkFn mark = slot.load(std::memory_order_acquire); mark != nullptr)
        {
            mark(referenced);
        }
    }
}

template <typename T>
[[nodiscard]] T* GlobalVecHolder<T>::take_free_slot() noexcept
{
    if (free_ids.empty())
    {
        return nullptr;
    }
    const std::uint32_t id = free_ids.back();
    free_ids.pop_back();
    free_ids_count.fetch_sub(1, std::memory_order_relaxed);
    return reinterpret_cast<T*>(region_begin.load(std::memory_order_relaxed) +
                                ((static_cast<std::size_t>(id) - 1u) * region_stride));
}

template <typename T>
std::byte* GlobalVecHolder<T>::region_alloc(std::size_t bytes)
{
    const std::size_t off = region_used.fetch_add(bytes, std::memory_order_relaxed);
    // ONE capacity: what value_by_id can hold (region_usable, lowered by
    // CHUNK_CST_VAL2_ID_LIMIT); alloc_chunk() writes value_by_id[id] for every slot
    // carved here, so a slot past it would be a write past the table. Warn once at half.
    const std::size_t cap_bytes = (bytes != 0 && id_limit * bytes < region_usable) ? id_limit * bytes : region_usable;
    const char*       limiting =
        (max_ids * id_stride) < region_bytes ? "CHUNK_CST_VAL2_MAX_IDS" : "CHUNK_CST_VAL2_REGION_BYTES";
    if (id_limit < max_ids)
    {
        limiting = "CHUNK_CST_VAL2_ID_LIMIT";
    }
    if (bytes != 0 && off + bytes > cap_bytes / 2) [[unlikely]]
    {
        static std::atomic warned{false};
        bool               expected = false;
        if (warned.compare_exchange_strong(expected, true, std::memory_order_relaxed))
        {
            CHUNK_LOG(warn, compress,
                      "second-value identifiers past half used: {} of {} (bound by {}). Raise it if the count keeps "
                      "growing; running out aborts.",
                      off / bytes, cap_bytes / bytes, limiting);
        }
    }
    if (off + bytes > cap_bytes) [[unlikely]]
    {
        CHUNK_INVARIANT_FAILED("[cst_val2] second-value identifiers exhausted: {} of {}, bound by {} "
                               "(CHUNK_CST_VAL2_MAX_IDS={}, CHUNK_CST_VAL2_REGION_BYTES={} MiB). Raise it, or "
                               "reduce the number of DISTINCT second values the workload registers.",
                               off / bytes, cap_bytes / (bytes != 0 ? bytes : 1), limiting, max_ids,
                               region_bytes >> 20U);
    }
    return region_begin.load(std::memory_order_relaxed) + off;
}

template <typename T>
T* GlobalVecHolder<T>::index_lookup(InT<T> value) noexcept
{
    // The table named here is never freed while a reader can hold it (see
    // index_generations). A superseded generation misses newer values: a miss
    // only means "not found here".
    const IndexTable* table = index_tab.load(std::memory_order_acquire);
    if (table == nullptr)
    {
        return nullptr;
    }
    const size_t mask = table->cap - 1;
    const size_t slot = index_hash(value) & mask;
    for (int probe = 0; probe < index_max_probe; ++probe)
    {
        T* entry = table->slots[(slot + probe) & mask].load(std::memory_order_acquire);
        if (entry == nullptr)
        {
            return nullptr; // empty slot => value is not indexed here
        }
        if (value_of_ptr(entry) == value)
        {
            return entry; // validated hit
        }
    }
    return nullptr; // probe budget exhausted => treat as miss
}

template <typename T>
void GlobalVecHolder<T>::index_publish(InT<T> value, T* ptr) noexcept
{
    IndexTable* table = index_tab.load(std::memory_order_relaxed);
    // Grow before the table passes half full: linear probing degrades
    // sharply beyond that.
    if (table == nullptr || (table->used + 1) * 2 > table->cap)
    {
        table = index_grow();
    }
    for (int attempt = 0; attempt < 2; ++attempt)
    {
        const size_t mask = table->cap - 1;
        const size_t slot = index_hash(value) & mask;
        for (int probe = 0; probe < index_max_probe; ++probe)
        {
            const size_t idx = (slot + probe) & mask;
            const T*     cur = table->slots[idx].load(std::memory_order_acquire);
            if (cur != nullptr)
            {
                if (value_of_ptr(cur) == value)
                {
                    return; // already indexed (equivalent pointer)
                }
                continue; // occupied by another value -> probe on
            }
            // Empty slot: publish (single writer under insert_mutex).
            table->slots[idx].store(ptr, std::memory_order_release);
            ++table->used;
            return;
        }
        // Budget exhausted at <= 50% load: pathological clustering. One
        // more doubling before giving up.
        if (attempt == 0)
        {
            table = index_grow();
        }
    }
    // Still exhausted: a registered value may now be unfindable by
    // index_lookup, so the cold path must re-enable its authoritative O(N)
    // segment scan. Sticky flag (never reset until index_clear). Said once:
    // from here on reclaim_second_values() is refused, so the identifiers
    // only grow, and the exhaustion abort alone would not name the cause.
    if (!index_degraded.exchange(true, std::memory_order_relaxed))
    {
        CHUNK_LOG(warn, compress,
                  "second-value index degraded (probe budget exhausted after a regrowth): lookups fall back to the "
                  "O(N) chain scan, and reclaim_second_values() is refused until the last array of this type is "
                  "destroyed -- the identifiers can only grow.");
    }
}

template <typename T>
void GlobalVecHolder<T>::install_head()
{
    static const Sentinel<T> sentinel;

    region_init();

    // Install the initial segment, once. The CAS is the transfer: once it
    // succeeds #head owns the segment (clear_if_last(), reclaim_unmarked()
    // and cleanup_all() free the chain). A plain new, as in
    // append_segment(), rather than a unique_ptr released after the CAS:
    // the dropped release() hid the transfer from static analysis
    // (Coverity CID 661140).
    if (head.load(std::memory_order_relaxed) == nullptr)
    {
        auto*       seg      = new Segment<T>();
        Segment<T>* expected = nullptr;
        if (!head.compare_exchange_strong(expected, seg, std::memory_order_release, std::memory_order_relaxed))
        {
            delete seg; // another thread installed the first segment
        }
    }
}

template <typename T>
void GlobalVecHolder<T>::clear_if_last(int& nb_val2)
{
    const int remaining = ref_count.fetch_sub(1, std::memory_order_acq_rel) - 1;
    if (remaining > 0)
    {
        return; // Other zero_any::Array<T> instances still alive — do not free
    }

    // Last instance: free the segments and count what they held, for ONE
    // summary line. The slots themselves are region memory:
    // region_reset() below gives them back all at once.
    int         nb_segs = 0;
    Segment<T>* seg     = head.exchange(nullptr, std::memory_order_acq_rel);
    while (seg != nullptr)
    {
        ++nb_segs;
        const int used = seg->used;
        for (int i = 0; i < used; ++i)
        {
            if (seg->slots[i] != nullptr)
            {
                ++nb_val2;
            }
        }
        auto owned = std::unique_ptr<Segment<T>>(seg);
        seg        = owned->next;
    }

    if (nb_segs > 1)
    {
        std::cout << "[cst_val2] " << nb_val2 << " distinct second values across " << nb_segs << " segments (capacity "
                  << nb_segs * Segment<T>::slots_per_segment
                  << "), index_degraded=" << index_degraded.load(std::memory_order_relaxed) << "\n";
    }

    // Registry gone: drop the index so no stale pointer survives.
    index_clear();
    region_reset(); // the slots' pages, all at once

    // Reinstall an empty segment for any later pass. NOT init(): that
    // takes a reference, and one taken here, with no array alive, would
    // keep #ref_count above 0 for good.
    install_head();
}

template <typename T>
void GlobalVecHolder<T>::cleanup_all()
{
    const Segment<T>* seg = head.exchange(nullptr, std::memory_order_relaxed);
    while (seg != nullptr)
    {
        // The slots are region memory, freed by region_reset() below.
        const Segment<T>* next = seg->next;
        delete seg;
        seg = next;
    }
    index_clear(); // registry gone: drop stale advisory index entries
    region_reset();
}

template <typename T>
void GlobalVecHolder<T>::region_init()
{
    if (region_begin.load(std::memory_order_acquire) != nullptr)
    {
        return;
    }
    // MM_PROTECT: PROT_NONE for good -- nothing is ever stored in the
    // region, so any dereference of a second-value pointer faults instead
    // of reading a plausible zero. Otherwise RW (never touched either).
    const int prot = build_config::use_mm_protect ? PROT_NONE : (PROT_READ | PROT_WRITE);
    void*     raw  = mmap(nullptr, region_usable, prot, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (raw == MAP_FAILED) [[unlikely]]
    {
        CHUNK_INVARIANT_FAILED("[cst_val2] cannot reserve the second-value region ({} MiB): {}", region_usable >> 20U,
                               std::strerror(errno));
    }
    // Every slot is carved at a multiple of region_stride (id_stride),
    // which is what turns a slot pointer into a small identifier and back
    // (TabCstVal2Holder::id_of_ptr / value_of_id). Nothing is ever stored
    // in the region, so a slot only has to give each identifier a distinct
    // address: one cache line does, and no page is ever backed.
    if (std::byte* expected = nullptr;
        region_begin.compare_exchange_strong(expected, static_cast<std::byte*>(raw), std::memory_order_acq_rel))
    {
        region_end     = static_cast<std::byte*>(raw) + region_usable;
        const auto lim = env_number<std::size_t>("CHUNK_CST_VAL2_ID_LIMIT", max_ids);
        id_limit       = (lim >= 1 && lim < max_ids) ? lim : max_ids; // lowers only
    }
    else
    {
        munmap(raw, region_usable); // lost the race: the winner's region stands
    }
}

template <typename T>
void GlobalVecHolder<T>::region_reset() noexcept
{
    free_ids.clear(); // the identifiers it lists go with the pages
    free_ids_count.store(0, std::memory_order_relaxed);
    std::byte* base = region_begin.load(std::memory_order_relaxed);
    if (base == nullptr)
    {
        return;
    }
    const std::size_t used = region_used.exchange(0, std::memory_order_relaxed);
    if (used == 0)
    {
        return;
    }
    if constexpr (build_config::use_mm_protect)
    {
        mprotect(base, used, PROT_READ | PROT_WRITE); // madvise needs a writable mapping to drop
    }
    madvise(base, used, MADV_DONTNEED);
    if constexpr (build_config::use_mm_protect)
    {
        mprotect(base, used, PROT_NONE);
    }
}

template <typename T>
size_t GlobalVecHolder<T>::index_hash(InT<T> value) noexcept
{
    uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(T) < sizeof(uint64_t) ? sizeof(T) : sizeof(uint64_t));
    bits ^= bits >> 33U;
    bits *= 0xff51afd7ed558ccdULL;
    bits ^= bits >> 33U;
    bits *= 0xc4ceb9fe1a85ec53ULL;
    bits ^= bits >> 33U;
    return static_cast<size_t>(bits);
}

template <typename T>
GlobalVecHolder<T>::IndexTable* GlobalVecHolder<T>::index_grow() noexcept
{
    const IndexTable* old = index_tab.load(std::memory_order_relaxed);
    const std::size_t cap = old == nullptr ? index_cap : old->cap * 2;
    auto              nt  = std::make_unique<IndexTable>(cap);
    if (old != nullptr)
    {
        const std::size_t mask = cap - 1;
        for (std::size_t i = 0; i < old->cap; ++i)
        {
            T* entry = old->slots[i].load(std::memory_order_relaxed);
            if (entry == nullptr)
            {
                continue;
            }
            std::size_t idx = index_hash(value_of_ptr(entry)) & mask;
            while (nt->slots[idx].load(std::memory_order_relaxed) != nullptr)
            {
                idx = (idx + 1) & mask;
            }
            nt->slots[idx].store(entry, std::memory_order_relaxed);
        }
        nt->used = old->used;
    }
    IndexTable* raw = nt.get();
    index_generations.push_back(std::move(nt));
    index_tab.store(raw, std::memory_order_release);
    return raw;
}

template <typename T>
void GlobalVecHolder<T>::index_clear() noexcept
{
    index_degraded.store(false, std::memory_order_relaxed);
    tail = nullptr; // drop cached tail; next cold-path append re-seeds from head
    // Single-threaded by contract, so this is the one point where the
    // superseded generations can go: no reader holds any of them. The
    // live table goes too; the next publish starts again at index_cap.
    index_tab.store(nullptr, std::memory_order_release);
    index_generations.clear();
}

// ---------------------------------------------------------------------------
// shrink_array::TabCstVal2Holder<T>
// ---------------------------------------------------------------------------

template <typename T>
[[nodiscard]] T* TabCstVal2Holder<T>::find_or_set_ind_cst_val2(T second_v, SecondValueRef<T> ptr_second_value)
{
    // Up to the record in the second-value table, which the reclaim's mark
    // scans: a reclaim must not run in between (LookupGuard).
    const typename VecHolder::LookupGuard in_flight;
    T*                                    buf = find_or_register(second_v);
    ptr_second_value.set_from_array(buf);
    return buf;
}

template <typename T>
[[nodiscard]] T* TabCstVal2Holder<T>::find_or_set_ind_cst_val2_release(T second_v, SecondValueRef<T> ptr_second_value)
{
    const typename VecHolder::LookupGuard in_flight; // as find_or_set_ind_cst_val2()
    T*                                    buf = find_or_register(second_v);
    ptr_second_value.set_from_array_release(buf);
    return buf;
}

template <typename T>
[[nodiscard]] T* TabCstVal2Holder<T>::find_cst_val2_ptr(InT<T> second_v)
{
    using Gvh = GlobalVecHolder<T>;
    const typename Gvh::LookupGuard in_flight; // holds an index generation / the chain
    if (T* hit = Gvh::index_lookup(second_v)) [[likely]]
    {
        return hit;
    }
    if (!Gvh::index_degraded.load(std::memory_order_relaxed))
    {
        return nullptr;
    }
    const std::scoped_lock lk(Gvh::insert_mutex);
    for (const Segment<T>* seg = Gvh::head.load(std::memory_order_acquire); seg != nullptr; seg = seg->next)
    {
        for (int i = 0; i < seg->used; ++i)
        {
            if (Gvh::value_of_ptr(seg->slots[i]) == second_v)
            {
                return seg->slots[i];
            }
        }
    }
    return nullptr;
}

template <typename T>
[[nodiscard]] T TabCstVal2Holder<T>::value_of_ptr_checked(const T* ptr) noexcept
{
    if (!is_cst_val2_ptr(ptr))
    {
        return T{};
    }
    const std::uint32_t id = id_of_ptr(ptr);
    if (id == 0u || id > VecHolder::max_ids)
    {
        return T{};
    }
    return VecHolder::value_by_id[id];
}

template <typename T>
T* TabCstVal2Holder<T>::alloc_chunk(T val)
{
    // A reclaimed identifier first; the region only when none is free.
    T* buf = VecHolder::take_free_slot();
    if (buf == nullptr)
    {
        buf = reinterpret_cast<T*>(VecHolder::region_alloc(VecHolder::region_stride));
    }
    // Before the slot can be published: a reader gets the identifier only from a
    // slot written after this returns.
    VecHolder::value_by_id[id_of_ptr(buf)] = val;
    return buf;
}

template <typename T>
[[nodiscard]] T* TabCstVal2Holder<T>::find_or_register(InT<T> second_v)
{
    const typename VecHolder::LookupGuard in_flight; // check builds; nests in the callers'
    // ── Hot path, lock-free ───────────────────────────────────────────
    if (T* hit = VecHolder::index_lookup(second_v)) [[likely]]
    {
        return hit;
    }

    // ── Cold path, under the insert lock ──────────────────────────────
    const std::scoped_lock lk(VecHolder::insert_mutex);

    // Double-check: another thread may have inserted second_v since our miss. The
    // index is single-writer under this lock, so it is authoritative here.
    if (T* hit = VecHolder::index_lookup(second_v)) [[likely]]
    {
        return hit;
    }

    // Only if the index ever overflowed its probe budget (index_degraded) can a
    // registered value be unindexed: then, and only then, scan the chain to avoid a duplicate.
    const bool degraded = VecHolder::index_degraded.load(std::memory_order_relaxed);

    SegmentT* last_seg = VecHolder::tail;
    if (last_seg == nullptr) [[unlikely]]
    {
        // First append after a clear: re-seed the cached tail from head.
        last_seg = VecHolder::head.load(std::memory_order_acquire);
        while (last_seg->next != nullptr)
        {
            last_seg = last_seg->next;
        }
        VecHolder::tail = last_seg;
    }

    if (degraded) [[unlikely]]
    {
        for (const SegmentT* seg = VecHolder::head.load(std::memory_order_acquire); seg != nullptr; seg = seg->next)
        {
            for (int i = 0; i < seg->used; ++i)
            {
                if (VecHolder::value_of_ptr(seg->slots[i]) == second_v)
                {
                    VecHolder::index_publish(second_v, seg->slots[i]);
                    return seg->slots[i];
                }
            }
        }
    }

    // Not present: append. The list is append-only between clears, so the
    // first free slot is last_seg->used (or slot 0 of a fresh segment).
    // create() guarantees head is never nullptr here.
    SegmentT* seg = last_seg;
    if (seg->used == SegmentT::slots_per_segment)
    {
        seg = VecHolder::append_segment(last_seg);
    }

    T* buf                = alloc_chunk(second_v); // value_by_id[] BEFORE publication
    seg->slots[seg->used] = buf;
    ++seg->used;
    VecHolder::index_publish(second_v, buf); // release: readers see the value
    return buf;
}

} // namespace shrink_array
