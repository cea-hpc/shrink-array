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
#include "fast_allocators/slab_arena.h" // the class these bodies belong to

namespace fast_alloc
{

// ---------------------------------------------------------------------------
// fast_alloc::EmptySlabSweep<Slab>
// ---------------------------------------------------------------------------

template <typename Slab>
void EmptySlabSweep<Slab>::add(Slab* slab) noexcept
{
    if (slab == nullptr)
    {
        return;
    }
    try
    {
        items_.push_back(Item{0, 0, slab});
    }
    catch (...)
    {
        // Out of memory for the batch: advise this one on its own rather
        // than leave it resident (the floor is not consulted: it cannot
        // be, the sweep's size is unknown yet).
        detail::advise_one_slab(slab);
    }
}

template <typename Slab>
std::size_t EmptySlabSweep<Slab>::flush() noexcept
{
    if (items_.empty())
    {
        return 0;
    }
    // Whatever path this flush leaves by, it is timed and counted.
    struct SweepTimer
    {
        std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
        SweepTimer()                                = default;
        SweepTimer(const SweepTimer&)               = delete;
        SweepTimer& operator=(const SweepTimer&)    = delete;
        SweepTimer(SweepTimer&&)                    = delete;
        SweepTimer& operator=(SweepTimer&&)         = delete;
        ~SweepTimer()
        {
            const auto dt = std::chrono::steady_clock::now() - start;
            reclaim_sweep_ns().fetch_add(
                static_cast<std::size_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(dt).count()),
                std::memory_order_relaxed);
            reclaim_sweeps().fetch_add(1, std::memory_order_relaxed);
        }
    };
    const SweepTimer timer;

    std::sort(items_.begin(), items_.end(),
              [](const Item& lhs, const Item& rhs) { return std::less<Slab*>{}(lhs.slab, rhs.slab); });
    items_.erase(std::unique(items_.begin(), items_.end(),
                             [](const Item& lhs, const Item& rhs) { return lhs.slab == rhs.slab; }),
                 items_.end());
    // CHUNK_RECLAIM_MIN_SLABS: madvise syscalls against the pages they
    // return, decided by how many slabs this sweep empties at once.
    if (items_.size() < alloc_config::reclaim_min_empty_slabs())
    {
        items_.clear();
        return 0;
    }
    for (Item& it : items_)
    {
        it.len   = detail::reclaimable_bytes(it.slab);
        it.begin = fast_alloc::address_bits(it.slab->data);
    }
    items_.erase(std::remove_if(items_.begin(), items_.end(), [](const Item& it) { return it.len == 0; }),
                 items_.end());
    std::sort(items_.begin(), items_.end(), [](const Item& lhs, const Item& rhs) { return lhs.begin < rhs.begin; });

    std::size_t bytes        = 0;
    std::size_t inline_calls = 0; // runs advised at once because runs_ could not grow
    runs_.clear();
    std::size_t first = 0;
    while (first < items_.size())
    {
        // Extend the run while the next slab starts exactly where this one's
        // advisable bytes end.
        std::size_t last = first;
        while (last + 1 < items_.size() && items_[last].begin + items_[last].len == items_[last + 1].begin)
        {
            ++last;
        }
        const std::uintptr_t begin = items_[first].begin;
        const std::uintptr_t end   = items_[last].begin + items_[last].len;
        // Flag first, for every slab of the run: from here their trailing
        // back-pointers are not to be trusted, and alloc() puts them back.
        for (std::size_t i = first; i <= last; ++i)
        {
            items_[i].slab->backptr_dropped = true;
        }
        try
        {
            runs_.push_back(Run{begin, end - begin});
        }
        catch (...)
        {
            ::madvise(detail::addr_ptr(begin), end - begin, MADV_DONTNEED); // no room: now
            ++inline_calls;
        }
        bytes += end - begin;
        first = last + 1;
    }
    const std::size_t slabs = items_.size();
    reclaim_calls().fetch_add(slabs, std::memory_order_relaxed);
    reclaim_syscalls().fetch_add(runs_.size() + inline_calls, std::memory_order_relaxed);
    reclaim_bytes().fetch_add(bytes, std::memory_order_relaxed);

    advise_runs();

    items_.clear();
    runs_.clear();
    return slabs;
}

template <typename Slab>
void EmptySlabSweep<Slab>::advise_runs() noexcept
{
    const auto n_runs = static_cast<long>(runs_.size());
    const int  nthr   = detail::reclaim_sweep_threads();
    // No nested team (the sweep is meant for a quiescent, serial point),
    // and not for a handful of runs: a team costs more than it saves.
    if (pmp::threaded && nthr > 1 && !pmp::in_parallel() && n_runs >= 4L * nthr)
    {
        const Run* runs = runs_.data();
        pmp::parallel_for(0L, n_runs, pmp::Options{.num_threads = nthr, .sched = pmp::Schedule::dynamic(16)},
                          [runs](long i)
                          { ::madvise(detail::addr_ptr(runs[i].begin), runs[i].len, MADV_DONTNEED); });
        return;
    }
    for (long i = 0; i < n_runs; ++i)
    {
        ::madvise(detail::addr_ptr(runs_[i].begin), runs_[i].len, MADV_DONTNEED);
    }
}

// ---------------------------------------------------------------------------
// fast_alloc::SlabArena
// ---------------------------------------------------------------------------

inline void* SlabArena::allocate(std::size_t size, std::size_t align, int rank)
{
    const std::scoped_lock lk(mtx_);

    // A slab a re-chunk gave back (recycle_batch), this rank's first: its
    // pages are resident, and were first touched by this rank.
    if (void* block = pop_recycled(size, rank); block != nullptr)
    {
        if (live_.contains(fast_alloc::address_bits(block)))
        {
            FAST_ALLOC_FATAL("arena {}: DOUBLE-ALLOC recycled slab {} (size={}): handed out while still live",
                             name_, block, size);
        }
        live_.insert(fast_alloc::address_bits(block));
        bytes_live_ += size;
        bytes_high_ = std::max(bytes_high_, bytes_live_);
        ++alloc_calls_;
        return block;
    }

    // Reuse a same-size slab from this arena's free-list if available.
    const auto it = free_lists_.find(size);
    if (it != free_lists_.end() && !it->second.empty())
    {
        void* block = it->second.back();
        it->second.pop_back();
        // Impossible by construction (a free-list slab was removed from live_).
        if (live_.contains(fast_alloc::address_bits(block)))
        {
            FAST_ALLOC_FATAL("arena {}: DOUBLE-ALLOC slab {} (size={}): handed out while still live -- a slab was "
                             "recycled with live blocks still in it",
                             name_, block, size);
        }
        live_.insert(fast_alloc::address_bits(block));
        bytes_live_ += size;
        bytes_high_ = std::max(bytes_high_, bytes_live_);
        ++alloc_calls_;
        return block;
    }

    // Otherwise bump-allocate, growing the reservation if needed.
    std::uintptr_t aligned = align_up(cursor_, align);
    if (aligned + size > region_end_)
    {
        grow(size + align);
        aligned = align_up(cursor_, align);
    }
    cursor_ = aligned + size;
    live_.insert(aligned);
    bytes_live_ += size;
    bytes_high_ = std::max(bytes_high_, bytes_live_);
    ++alloc_calls_;
    return region_base_ + (aligned - fast_alloc::address_bits(region_base_));
}

inline void SlabArena::deallocate(void* block, std::size_t size) noexcept
{
    if (block == nullptr)
    {
        return;
    }
    const std::scoped_lock lk(mtx_);
    // Not live: warn, do NOT abort (see @note).
    if (live_.erase(fast_alloc::address_bits(block)) == 0)
    {
        CHUNK_LOG(warn, arena, "{}: double-free/unknown slab {} (size={}) -- ignored", name_, block, size);
        return; // don't push an unknown slab onto the free-list
    }
    bytes_live_ -= size;
    ++dealloc_calls_;
    dealloc_bytes_ += size;
    free_lists_[size].push_back(block);
}

inline void SlabArena::recycle_batch(void* const* blocks, std::size_t count, std::size_t size, int rank) noexcept
{
    if (count == 0)
    {
        return;
    }
    const std::scoped_lock lk(mtx_);
    auto&                  lists    = recycled_[size];
    const auto             rank_idx = static_cast<std::size_t>(rank < 0 ? 0 : rank);
    if (lists.by_rank.size() <= rank_idx)
    {
        lists.by_rank.resize(rank_idx + 1);
    }
    auto& mine = lists.by_rank[rank_idx];
    for (std::size_t i = 0; i < count; ++i)
    {
        void* const block = blocks[i];
        if (block == nullptr)
        {
            continue;
        }
        if (live_.erase(fast_alloc::address_bits(block)) == 0)
        {
            CHUNK_LOG(warn, arena, "{}: recycle of an unknown/free slab {} (size={}) -- ignored", name_, block,
                      size);
            continue;
        }
        bytes_live_ -= size;
        ++dealloc_calls_;
        dealloc_bytes_ += size;
        mine.push_back(Recycled{.block = block, .advised = false});
        ++lists.count;
        ++recycled_in_;
    }
}

inline std::size_t SlabArena::advise_recycled(std::size_t keep_bytes) noexcept
{
    const std::scoped_lock lk(mtx_);
    const std::size_t      page = detail::reclaim_page_size();
    if (page == 0)
    {
        return 0;
    }
    struct Span
    {
        std::uintptr_t begin;
        std::size_t    len;
    };
    std::vector<Span> spans;
    std::size_t       slabs = 0;
    // keep_bytes: an even share per rank, of the slabs not advised yet.
    std::size_t n_ranks = 0;
    for (const auto& [size, lists] : recycled_)
    {
        n_ranks = std::max(n_ranks, lists.by_rank.size());
    }
    const std::size_t        per_rank = n_ranks != 0 ? keep_bytes / n_ranks : 0;
    std::vector<std::size_t> kept;
    if (per_rank != 0)
    {
        try
        {
            kept.assign(n_ranks, 0);
        }
        catch (...)
        {
            // No room for the counters: advise them all, as with keep_bytes = 0
            // (the loop below keeps nothing while `kept` is empty).
            kept.clear();
        }
    }
    for (auto& [size, lists] : recycled_)
    {
        const std::size_t adv = (size / page) * page;
        for (std::size_t r = 0; r < lists.by_rank.size(); ++r)
        {
            auto& list = lists.by_rank[r];
            // Newest last: pop_recycled() takes from the back, so a rank
            // keeps from the back.
            for (auto rit = list.rbegin(); rit != list.rend(); ++rit)
            {
                Recycled& entry = *rit;
                if (entry.advised)
                {
                    continue;
                }
                if (!kept.empty() && kept[r] + size <= per_rank)
                {
                    kept[r] += size; // stays resident, for the next allocate()
                    continue;
                }
                const auto begin = fast_alloc::address_bits(entry.block);
                if (adv == 0 || begin % page != 0)
                {
                    entry.advised = true; // never advisable: do not look again
                    continue;
                }
                try
                {
                    spans.push_back(Span{.begin = begin, .len = adv});
                }
                catch (...)
                {
                    ::madvise(entry.block, adv, MADV_DONTNEED); // no room for the batch: now
                    reclaim_syscalls().fetch_add(1, std::memory_order_relaxed);
                    reclaim_bytes().fetch_add(adv, std::memory_order_relaxed);
                }
                entry.advised = true;
                ++slabs;
            }
            if (!kept.empty())
            {
                // Resident slabs behind the advised ones: pop_recycled()
                // hands out the back first, pages already there.
                std::stable_partition(list.begin(), list.end(),
                                      [](const Recycled& entry) { return entry.advised; });
            }
        }
    }
    std::ranges::sort(spans, [](const Span& lhs, const Span& rhs) { return lhs.begin < rhs.begin; });
    // Coalesced in place: spans[0, runs) become the runs.
    std::size_t runs  = 0;
    std::size_t bytes = 0;
    for (std::size_t first = 0; first < spans.size();)
    {
        std::size_t last = first;
        while (last + 1 < spans.size() && spans[last].begin + spans[last].len == spans[last + 1].begin)
        {
            ++last;
        }
        const std::uintptr_t begin = spans[first].begin;
        const std::size_t    len   = spans[last].begin + spans[last].len - begin;
        spans[runs++]              = Span{.begin = begin, .len = len};
        bytes += len;
        first = last + 1;
    }
    // Serial, or from a team of CHUNK_RECLAIM_THREADS as the slab sweep
    // (EmptySlabSweep::advise_runs): still under the lock, which the team
    // does not need, so no allocate() takes a slab mid-advise.
    const auto n_runs = static_cast<long>(runs);
    const int  nthr   = detail::reclaim_sweep_threads();
    if (pmp::threaded && nthr > 1 && !pmp::in_parallel() && n_runs >= 4L * nthr)
    {
        const Span* run = spans.data();
        pmp::parallel_for(0L, n_runs, pmp::Options{.num_threads = nthr, .sched = pmp::Schedule::dynamic(16)},
                          [run](long i) { ::madvise(detail::addr_ptr(run[i].begin), run[i].len, MADV_DONTNEED); });
    }
    else
    {
        for (std::size_t i = 0; i < runs; ++i)
        {
            ::madvise(detail::addr_ptr(spans[i].begin), spans[i].len, MADV_DONTNEED);
        }
    }
    reclaim_calls().fetch_add(slabs, std::memory_order_relaxed);
    reclaim_syscalls().fetch_add(runs, std::memory_order_relaxed);
    reclaim_bytes().fetch_add(bytes, std::memory_order_relaxed);
    recycled_advised_ += slabs;
    return slabs;
}

[[nodiscard]] inline std::size_t SlabArena::resident_bytes() const noexcept
{
    const std::scoped_lock lk(mtx_);
    const long             page = ::sysconf(_SC_PAGESIZE);
    if (page <= 0)
    {
        return 0;
    }
    const auto                 psz = static_cast<std::size_t>(page);
    std::size_t                res = 0;
    std::vector<unsigned char> vec;
    for (const auto& region : regions_)
    {
        const std::size_t len   = region.end - region.begin;
        const std::size_t pages = (len + psz - 1) / psz;
        vec.resize(pages);
        if (::mincore(detail::addr_ptr(region.begin), len, vec.data()) != 0)
        {
            continue;
        }
        for (const unsigned char page_state : vec)
        {
            res += (page_state & 1u) != 0u ? psz : 0;
        }
    }
    return res;
}

[[nodiscard]] inline std::size_t SlabArena::mapped_bytes() const noexcept
{
    const std::scoped_lock lk(mtx_);
    std::size_t            mapped = 0;
    for (const auto& region : regions_)
    {
        mapped += (region.end - region.begin);
    }
    return mapped;
}

inline void SlabArena::report(const char* when) const noexcept
{
    const std::scoped_lock lk(mtx_);
    std::size_t            mapped = 0;
    for (const auto& region : regions_)
    {
        mapped += (region.end - region.begin);
    }
    std::size_t freelist = 0;
    for (const auto& [block_size, blocks] : free_lists_)
    {
        freelist += block_size * blocks.size();
    }
    std::size_t recycled = 0;
    for (const auto& [block_size, lists] : recycled_)
    {
        recycled += block_size * lists.count;
    }
    // CHUNK_ARENA_REPORT asked for it: a report, written whatever CHUNK_LOG says.
    CHUNK_LOG_ALWAYS(arena,
                     "{:<6} {:<16} mapped={:8} KiB  live={:8} KiB  high={:8} KiB  freelist={:8} KiB  regions={}  "
                     "allocs={}  deallocs={}  deallocMiB={}  recycled={} KiB (in={} out={} foreign={} advised={})",
                     name_, when, mapped >> 10U, bytes_live_ >> 10U, bytes_high_ >> 10U, freelist >> 10U,
                     regions_.size(), alloc_calls_, dealloc_calls_, dealloc_bytes_ >> 20U, recycled >> 10U,
                     recycled_in_, recycled_out_, recycled_foreign_, recycled_advised_);
}

inline void* SlabArena::pop_recycled(std::size_t size, int rank) noexcept
{
    const auto it = recycled_.find(size);
    if (it == recycled_.end() || it->second.count == 0)
    {
        return nullptr;
    }
    auto&             lists  = it->second.by_rank;
    const std::size_t nlists = lists.size();
    const bool        known  = rank >= 0 && std::cmp_less(rank, nlists);
    const std::size_t start  = known ? static_cast<std::size_t>(rank) : 0;
    for (std::size_t i = 0; i < nlists; ++i)
    {
        auto& list = lists[(start + i) % nlists];
        if (!list.empty())
        {
            void* const block = list.back().block;
            list.pop_back();
            --it->second.count;
            ++recycled_out_;
            if (i != 0 || !known)
            {
                ++recycled_foreign_;
            }
            return block;
        }
    }
    return nullptr;
}

inline void SlabArena::grow(std::size_t min_bytes)
{
    std::size_t want = reservation_chunk_;
    if (min_bytes > want)
    {
        want = align_up(min_bytes, static_cast<std::size_t>(1) << 21U); // round to 2 MiB
    }

    void* base = ::mmap(nullptr, want, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (base == MAP_FAILED)
    {
        const int err = errno;
        FAST_ALLOC_FATAL("arena {}: mmap of {} bytes failed ({})", name_, want, std::strerror(err));
    }
    // CHUNK_THP: `1` advises huge pages (the slabs are 2 MiB and aligned
    // on it, a huge page each), `never` forbids them, unset leaves the
    // kernel's mode. Advice only: a failure is ignored.
    alloc_config::advise_thp_region(base, want);
    auto region_begin = fast_alloc::address_bits(base);
    regions_.push_back(Region{.begin = region_begin, .end = region_begin + want});
    region_base_ = static_cast<std::byte*>(base);
    cursor_      = region_begin;
    region_end_  = region_begin + want;
}

} // namespace fast_alloc
