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

/**
 * @file slab_arena.h
 * @brief Disjoint address-space arenas, so two slab pools can never collide.
 *
 * @par The hazard this makes impossible
 * Do not let the op pool and the value pool take their slabs from the general
 * allocator: a freed op slab (thread-local allocator destroyed between parallel
 * regions) goes back to glibc, the value pool can reobtain THE SAME ADDRESS while
 * a deferred op still lives there, and value writes overwrite the op's vtable.
 *
 * @par The fix
 * Each pool gets its own arena: one large mmap reservation carved by a bump
 * pointer, never unmapped, so an address of one arena is never handed out by
 * another. Freed slabs are recycled only WITHIN their arena (free-list by size).
 *
 * @par What that costs
 * A slab returned with deallocate() stays resident: the arena keeps it, the OS
 * never gets it back. That is the RSS the free-list holds, and what
 * EmptySlabSweep addresses -- it drops emptied slabs' pages while leaving
 * their addresses valid and the slabs reusable.
 *
 * @note Three arenas, not two: ctx slabs can alias op slabs the same way.
 *       @see ArenaKind
 *
 * @warning The arenas are deliberately leaked. They must outlive every static
 *          and thread_local that might free into them at teardown. @see op_arena
 */

#pragma once

#include <sys/mman.h>    // mmap, madvise
#include <unistd.h>      // sysconf(_SC_PAGESIZE)
#include <algorithm>     // std::max, the high-water mark
#include <cstddef>       // std::size_t
#include <cstdint>       // std::uintptr_t, std::uint8_t
#include <cstdlib>       // std::getenv
#include <functional>    // std::less (the sweep orders slab pointers)
#include <utility>       // std::cmp_less
#include <vector>        // std::vector
#include <mutex>         // std::mutex, std::scoped_lock
#include <atomic>        // std::atomic, std::memory_order_relaxed
#include <charconv>      // std::from_chars (CHUNK_RECLAIM_THREADS)
#include <chrono>        // the sweep timer
#include <cstring>       // std::strlen
#include <unordered_map> // std::unordered_map
#include <unordered_set> // std::unordered_set
#include <system_error>  // std::errc (std::from_chars)

#include "fast_allocators/internal/alloc_config.h" // thp_advise (CHUNK_THP), reclaim_min_empty_slabs
#include "fast_allocators/internal/pmp_alloc.h"    // pmp::parallel_for, pmp::in_parallel, pmp::max_threads (the sweeps)
#include "fast_allocators/internal/fatal.h"        // FAST_ALLOC_FATAL
#include "fast_allocators/internal/log_core.h"     // CHUNK_LOG, CHUNK_LOG_ALWAYS (CHUNK_ARENA_REPORT)

namespace fast_alloc
{

// The reclaim counters live in this lowest-level header so tc_allocator.h and
// report_all_arenas() see one definition (no cross-header ordering issue).

/// @brief How many slabs have been advised away (by an EmptySlabSweep, or
///        one by one by detail::advise_one_slab()). A count of SLABS, not of syscalls: see
///        reclaim_syscalls().
inline std::atomic<std::size_t>& reclaim_calls() noexcept
{
    static std::atomic<std::size_t> counter{0};
    return counter;
}
/// @brief How many bytes that represents.
inline std::atomic<std::size_t>& reclaim_bytes() noexcept
{
    static std::atomic<std::size_t> counter{0};
    return counter;
}
/// @brief How many madvise() system calls those slabs took; smaller than
///        reclaim_calls() when an EmptySlabSweep merges adjacent slabs (each
///        syscall is a TLB shootdown to every CPU of the process).
inline std::atomic<std::size_t>& reclaim_syscalls() noexcept
{
    static std::atomic<std::size_t> counter{0};
    return counter;
}
/// @brief Wall time spent in EmptySlabSweep::flush() (sort, flags, madvise), ns.
inline std::atomic<std::size_t>& reclaim_sweep_ns() noexcept
{
    static std::atomic<std::size_t> counter{0};
    return counter;
}
/// @brief Number of non-empty EmptySlabSweep::flush() calls.
inline std::atomic<std::size_t>& reclaim_sweeps() noexcept
{
    static std::atomic<std::size_t> counter{0};
    return counter;
}

/// @brief Why a slab was NOT advised away, per reason: tells "nothing was
///        drained" from "drained but madvise could not take it".
///@{
inline std::atomic<std::size_t>& reclaim_skip_live() noexcept
{
    static std::atomic<std::size_t> v{0};
    return v;
}
inline std::atomic<std::size_t>& reclaim_skip_unaligned() noexcept
{
    static std::atomic<std::size_t> v{0};
    return v;
}
inline std::atomic<std::size_t>& reclaim_skip_small() noexcept
{
    static std::atomic<std::size_t> v{0};
    return v;
}
inline std::atomic<std::size_t>& reclaim_entered() noexcept
{
    static std::atomic<std::size_t> v{0};
    return v;
}
///@}

/// @brief A global counter alone on its cache lines.
/// @details Do not use a bare `static std::atomic` for a hot counter: the linker
///          packs it next to other globals (e.g. the head of
///          Proxy2<double>::share_info_array, read by every Chunk2 write) and
///          each fetch_add then invalidates that line on every core (false
///          sharing, large at high thread counts). 128, not 64: adjacent-line
///          prefetch, and 128-byte lines.
struct alignas(128) IsolatedCounter
{
    std::atomic<std::size_t> v{0};
};
static_assert(sizeof(IsolatedCounter) == 128, "IsolatedCounter must fill its alignment");

/// @brief What the chunk layer did with a buffer it stopped using: says whether
///        any block reached the allocator (parked blocks are retained by design).
///        Bumped through CHUNK_FREEPATH_INC (chunk/internal/freepath_counters.h).
///@{
inline std::atomic<std::size_t>& freepath_freed() noexcept
{
    static IsolatedCounter counter;
    return counter.v;
}
inline std::atomic<std::size_t>& freepath_park_own() noexcept
{
    static IsolatedCounter counter;
    return counter.v;
}
inline std::atomic<std::size_t>& freepath_park_foreign() noexcept
{
    static IsolatedCounter counter;
    return counter.v;
}
inline std::atomic<std::size_t>& freepath_kept_nomigrate() noexcept
{
    static IsolatedCounter counter;
    return counter.v;
}
///@}
/// @brief Materialisations routed with NO affinity key (the rank's shared pool)
///        and with one. A slab can only empty if it holds one array's blocks: a
///        large keyless count while keys were granted means the key does not
///        reach the allocation, and shared slabs never empty.
///@{
inline std::atomic<std::size_t>& freepath_alloc_key0() noexcept
{
    static IsolatedCounter counter;
    return counter.v;
}
inline std::atomic<std::size_t>& freepath_alloc_keyed() noexcept
{
    static IsolatedCounter counter;
    return counter.v;
}
///@}


namespace detail
{
/// @brief A non-negative integer from the environment, read once by the
///        caller's static; @p fallback when unset, empty or malformed (warned).
inline std::size_t reclaim_env_number(const char* name, std::size_t fallback) noexcept
{
    const char* env = std::getenv(name);
    if (env == nullptr || env[0] == '\0')
    {
        return fallback;
    }
    std::size_t v   = 0;
    const char* end = env + std::strlen(env);
    const auto  res = std::from_chars(env, end, v);
    if (res.ec != std::errc{} || res.ptr != end)
    {
        CHUNK_LOG(warn, env, "{}=\"{}\" is not a valid value: ignored, {} kept", name, env, fallback);
        return fallback;
    }
    return v;
}

/// @brief CHUNK_RECLAIM_THREADS (default 1 = serial): threads issuing the
///        sweep's madvise calls; 0 = the backend's team size (pmp::max_threads(),
///        1 in a serial build).
inline int reclaim_sweep_threads() noexcept
{
    static const int n = []
    {
        const std::size_t v = reclaim_env_number("CHUNK_RECLAIM_THREADS", 1);
        if (v == 0)
        {
            return pmp::max_threads();
        }
        return static_cast<int>(std::min<std::size_t>(v, 4096));
    }();
    return n;
}

/// @brief The system page size, read once.
inline std::size_t reclaim_page_size() noexcept
{
    static const std::size_t page = []
    {
        const long ps = ::sysconf(_SC_PAGESIZE);
        return ps > 0 ? static_cast<std::size_t>(ps) : std::size_t{0};
    }();
    return page;
}

/// @brief The pointer at address @p addr. The arena keeps its addresses as
///        integers (cursor, live set, runs) and only turns them back into
///        pointers for the caller or a syscall (madvise, mincore), here.
inline void* addr_ptr(std::uintptr_t addr) noexcept
{
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    return reinterpret_cast<void*>(addr);
}

/// @brief The checks before a slab is advised away, without the syscall: how
///        many bytes of @p slab may be advised away, or 0 if none. Bumps the
///        reclaim_entered() / reclaim_skip_*() counters.
template <typename Slab>
std::size_t reclaimable_bytes(const Slab* slab) noexcept
{
    reclaim_entered().fetch_add(1, std::memory_order_relaxed);
    if (slab == nullptr || slab->nn_ref != 0)
    {
        if (slab != nullptr)
        {
            reclaim_skip_live().fetch_add(1, std::memory_order_relaxed);
        }
        return 0;
    }
    const void* base = slab->data;
    const auto  len  = static_cast<std::size_t>(slab->block_allocator);
    if (base == nullptr || len == 0)
    {
        return 0;
    }
    const std::size_t up = reclaim_page_size();
    if (up == 0)
    {
        return 0;
    }
    if ((reinterpret_cast<std::uintptr_t>(base) % up) != 0)
    {
        reclaim_skip_unaligned().fetch_add(1, std::memory_order_relaxed);
        return 0;
    }
    const std::size_t adv = (len / up) * up;
    if (adv == 0)
    {
        reclaim_skip_small().fetch_add(1, std::memory_order_relaxed);
        return 0;
    }
    return adv;
}

/**
 * @brief Give ONE emptied slab's pages back, on its own: the fallback of
 *        EmptySlabSweep::add() when the batch cannot grow (bad_alloc).
 * @details A sweep of one slab: same checks, flag-before-madvise and counters;
 *          rationale and preconditions are EmptySlabSweep's. No other caller is
 *          meant to exist (it cannot merge adjacent slabs).
 * @return true if the pages were dropped and @p slab was flagged.
 */
template <typename Slab>
bool advise_one_slab(Slab* slab) noexcept
{
    const std::size_t adv = detail::reclaimable_bytes(slab);
    if (adv == 0)
    {
        return false;
    }
    // Flag first: from here the trailing back-pointer is not to be trusted, and
    // alloc() is what puts it back.
    slab->backptr_dropped = true;
    ::madvise(slab->data, adv, MADV_DONTNEED);
    reclaim_calls().fetch_add(1, std::memory_order_relaxed);
    reclaim_syscalls().fetch_add(1, std::memory_order_relaxed);
    reclaim_bytes().fetch_add(adv, std::memory_order_relaxed);
    return true;
}
} // namespace detail

/**
 * @brief Give emptied slabs' resident pages back, the last page included, with
 *        ONE madvise() per run of adjacent slabs.
 *
 * @tparam Slab Duck-typed on `nn_ref`, `data`, `block_allocator` and
 *         `backptr_dropped`. A template so those members need only be complete
 *         at instantiation.
 *
 * @par Giving a slab's pages back
 *          MADV_DONTNEED returns the pages to the OS but leaves the mapping in
 *          place: the address stays valid, the slab stays on its owner's list,
 *          and the next allocation from it faults fresh zero pages back in.
 *          Safe only for a pool whose free list lives OUTSIDE the slab: the
 *          value family keeps links in a side array, the op/ctx family threads
 *          them through the freed elements and would be destroyed by this.
 *
 *          The last page holds the self back-pointer TCSlab::slab_from() reads
 *          on every free, and it is advised too: kept resident, it is the whole
 *          residue a large re-chunk fails to give back (about a gigabyte at
 *          scale), and single-page slabs could not be advised at all.
 *
 *          The back-pointer can go because nobody can read it: the slab holds
 *          no block (nn_ref == 0 is checked first), so no free can resolve
 *          through it, and the only way one becomes possible again is an
 *          alloc() from this slab -- which restores the back-pointer before it
 *          hands the block out. The slab carries the flag saying so.
 *
 * @par Batching
 *          Each MADV_DONTNEED is a syscall plus a TLB shootdown to every CPU
 *          of the process; one per slab is tens of thousands of calls per
 *          re-chunk sweep. Slabs of one size are carved back to back from the
 *          value arena and aligned on their size, so an abandoned (key, size)
 *          set is largely contiguous; merging turns thousands of calls into a few.
 *
 *          add() only records the slab. flush() drops duplicates (the same slab
 *          reached through two routes, e.g. an unregistered key falling back to
 *          key 0), applies the CHUNK_RECLAIM_MIN_SLABS floor to the WHOLE
 *          sweep, runs detail::reclaimable_bytes() on each slab
 *          (bumping the same counters), sorts by address, merges runs whose
 *          bytes touch exactly, sets backptr_dropped on every slab of a run,
 *          then advises the run. A slab whose advisable length is shorter than
 *          the slab (not a page multiple) ends its run, so nothing outside a
 *          slab is ever advised.
 *
 *          The floor is the sweep's, not each instance's: each rank holds only
 *          ~1/ranks of the abandoned set, and a per-instance floor would drop
 *          whole ranks -- holes in the runs the batching is there to merge.
 *
 * @warning Two preconditions, neither checkable here, held from the first
 *          add() until flush() returns. Every collected slab is empty as a
 *          FACT, not just by what nn_ref says. And no alloc() from any of them
 *          can interleave: backptr_dropped is set before the madvise, so a
 *          concurrent alloc() could restore the back-pointer only to have it
 *          wiped a moment later. A re-chunk's quiescent end-of-move point
 *          qualifies; a free path does not.
 */
template <typename Slab>
class EmptySlabSweep
{
  public:
    EmptySlabSweep()                                 = default;
    EmptySlabSweep(const EmptySlabSweep&)            = delete;
    EmptySlabSweep& operator=(const EmptySlabSweep&) = delete;
    EmptySlabSweep(EmptySlabSweep&&)                 = delete;
    EmptySlabSweep& operator=(EmptySlabSweep&&)      = delete;
    /// @brief Anything collected and not yet flushed is flushed: a collected
    ///        slab is never silently left resident.
    ~EmptySlabSweep()
    {
        flush();
    }

    /// @brief Collect @p slab; flush() decides whether it is advised away (see
    ///        detail::reclaimable_bytes() for the checks).
    void add(Slab* slab) noexcept
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

    /// @brief Advise every collected slab away, one madvise() per run -- or
    ///        none of them if the sweep holds fewer distinct slabs than
    ///        CHUNK_RECLAIM_MIN_SLABS.
    /// @details Serial part: sort, drop duplicates, floor, checks, runs
    ///          (merging adjacent slabs), every backptr_dropped flag and the
    ///          counters. Then the madvise calls, from CHUNK_RECLAIM_THREADS
    ///          threads (serial by default): every flag is set before the first
    ///          madvise, and the syscalls touch disjoint ranges. Timed
    ///          (reclaim_sweep_ns / reclaim_sweeps).
    /// @return Number of slabs advised.
    std::size_t flush() noexcept
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
        } const timer;

        std::sort(items_.begin(), items_.end(),
                  [](const Item& a, const Item& b) { return std::less<Slab*>{}(a.slab, b.slab); });
        items_.erase(
            std::unique(items_.begin(), items_.end(), [](const Item& a, const Item& b) { return a.slab == b.slab; }),
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
            it.begin = reinterpret_cast<std::uintptr_t>(it.slab->data);
        }
        items_.erase(std::remove_if(items_.begin(), items_.end(), [](const Item& it) { return it.len == 0; }),
                     items_.end());
        std::sort(items_.begin(), items_.end(), [](const Item& a, const Item& b) { return a.begin < b.begin; });

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

  private:
    struct Item
    {
        std::uintptr_t begin; ///< slab->data, set by flush()
        std::size_t    len;   ///< advisable bytes (page multiple), set by flush()
        Slab*          slab;  ///< to set backptr_dropped
    };
    struct Run
    {
        std::uintptr_t begin; ///< first byte of the run
        std::size_t    len;   ///< bytes, page multiple
    };

    /// @brief madvise every run of runs_, serially or from a team of the backend.
    void advise_runs() noexcept
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

    std::vector<Item> items_;
    std::vector<Run>  runs_;
};


/**
 * @brief A growable bump arena whose mmap reservations are never unmapped.
 *
 * @details Carves slabs out of large reservations with a bump pointer, and
 *          recycles freed ones through a per-size free list. Nothing goes back
 *          to the OS, on purpose: an address this arena handed out can never be
 *          handed out by another allocator.
 *
 * @note One mutex guards everything: not a hot path (one call per slab).
 *
 * @warning Growing STRANDS the rest of the previous reservation (the cursor
 *          never returns): rare with 256 MiB regions, but many oversized
 *          requests would waste the tail of each region.
 */
class SlabArena
{
  public:
    /**
     * @brief Create an empty arena; the first allocate() maps its first region.
     * @param name Short label, used only in diagnostics.
     * @param reservation_chunk Size of each mmap reservation. A request larger
     *        than this gets its own region, rounded up to 2 MiB.
     */
    explicit SlabArena(const char* name, std::size_t reservation_chunk = (static_cast<std::size_t>(256) << 20U))
        : name_(name), reservation_chunk_(reservation_chunk)
    {
    }

    /**
     * @brief Hand out @p size bytes aligned to @p align.
     * @param size  Slab size in bytes; also the free-list key it is recycled under.
     * @param align Required alignment of the slab.
     * @param rank Rank asking (TCSlab::owner_rank), -1 if none: a slab a
     *        re-chunk recycled (recycle_batch) is taken first from this rank's
     *        list, then from the others', before the free list and the cursor.
     * @return Never null: a failed mmap aborts.
     * @details Prefers an exactly-@p size slab from the free list, falls back to
     *          bumping the cursor, and grows the reservation if that does not
     *          fit.
     * @warning Aborts if the free list hands back a slab still marked live (a
     *          slab recycled with blocks in it). Unconditional and fatal on
     *          purpose: cheap here, and silent recovery would hide the bug.
     */
    void* allocate(std::size_t size, std::size_t align, int rank = -1)
    {
        const std::scoped_lock lk(mtx_);

        // A slab a re-chunk gave back (recycle_batch), this rank's first: its
        // pages are resident, and were first touched by this rank.
        if (void* block = pop_recycled(size, rank); block != nullptr)
        {
            if (live_.contains(reinterpret_cast<std::uintptr_t>(block)))
            {
                FAST_ALLOC_FATAL("arena {}: DOUBLE-ALLOC recycled slab {} (size={}): handed out while still live",
                                 name_, block, size);
            }
            live_.insert(reinterpret_cast<std::uintptr_t>(block));
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
            if (live_.contains(reinterpret_cast<std::uintptr_t>(block)))
            {
                FAST_ALLOC_FATAL("arena {}: DOUBLE-ALLOC slab {} (size={}): handed out while still live -- a slab was "
                                 "recycled with live blocks still in it",
                                 name_, block, size);
            }
            live_.insert(reinterpret_cast<std::uintptr_t>(block));
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
        return detail::addr_ptr(aligned);
    }

    /**
     * @brief Take a slab back into THIS arena's free list. Never to the OS.
     * @param block The slab, as returned by allocate(); null is a no-op.
     * @param size Must be the size it was allocated with -- the free list is
     *        keyed by it, and a wrong key makes the slab unreachable.
     * @note An unknown or already-freed slab warns and is dropped rather than
     *       aborting: at teardown a ~TCTypeCAllocator may free a slab whose
     *       bookkeeping is gone (harmless). A live slab handed out again is
     *       fatal, in allocate().
     */
    void deallocate(void* block, std::size_t size) noexcept
    {
        if (block == nullptr)
        {
            return;
        }
        const std::scoped_lock lk(mtx_);
        // Not live: warn, do NOT abort (see @note).
        if (live_.erase(reinterpret_cast<std::uintptr_t>(block)) == 0)
        {
            CHUNK_LOG(warn, arena, "{}: double-free/unknown slab {} (size={}) -- ignored", name_, block, size);
            return; // don't push an unknown slab onto the free-list
        }
        bytes_live_ -= size;
        ++dealloc_calls_;
        dealloc_bytes_ += size;
        free_lists_[size].push_back(block);
    }

    /**
     * @brief Take back @p n slabs a re-chunk has just emptied, for the next
     *        allocate() of rank @p rank. One lock for the whole batch.
     * @param blocks The slabs, as returned by allocate(); their headers are
     *        gone (TCSlabPool::recycle_empty_slabs). Null entries are skipped.
     * @param n Number of entries in @p blocks.
     * @param size Their allocation size: the key allocate() looks them up by.
     * @param rank The rank that held them (-1 counts as 0): allocate(size, ..,
     *        rank) takes them back first, their pages having been first
     *        touched there.
     * @details Unlike deallocate(), the slabs KEEP their pages: the re-chunk's
     *          next slice allocates about as much at the same slab size, and a
     *          resident slab costs no page fault nor kernel zeroing. What the next
     *          slice does not take is given back by advise_recycled().
     * @note An unknown or already-freed slab warns and is dropped, as in
     *       deallocate().
     */
    void recycle_batch(void* const* blocks, std::size_t n, std::size_t size, int rank) noexcept
    {
        if (n == 0)
        {
            return;
        }
        const std::scoped_lock lk(mtx_);
        auto&                  lists = recycled_[size];
        const auto             r     = static_cast<std::size_t>(rank < 0 ? 0 : rank);
        if (lists.by_rank.size() <= r)
        {
            lists.by_rank.resize(r + 1);
        }
        auto& mine = lists.by_rank[r];
        for (std::size_t i = 0; i < n; ++i)
        {
            void* const block = blocks[i];
            if (block == nullptr)
            {
                continue;
            }
            if (live_.erase(reinterpret_cast<std::uintptr_t>(block)) == 0)
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

    /**
     * @brief Give back the pages of every recycled slab nobody took and that
     *        was not advised yet: one madvise per run of adjacent slabs.
     * @param keep_bytes Leave up to this many bytes of not-yet-advised slabs
     *        resident, split evenly over the ranks (0 = advise them all).
     *        Each rank keeps its newest ones -- the end of its list, which
     *        allocate() takes first -- and its resident slabs are moved
     *        behind its advised ones, so they are handed out first.
     *        CHUNK_RECHUNK_KEEP_RECYCLED_MB sets it for the re-chunk.
     * @return Number of slabs advised.
     * @details The slabs stay on their rank's list, without pages (a later
     *          allocate() still takes them first). Each slab is advised once.
     *          Under the lock, so no allocate() can take a slab while its pages
     *          are dropped. Whole slabs, last page included: nothing of a free
     *          slab is read before its next owner rewrites it
     *          (TCSlab::alloc_block writes the back-pointer).
     * @pre Meant for a quiescent point (the re-chunk calls it between slices,
     *      for the previous boundary's surplus, and after the last one).
     */
    std::size_t advise_recycled(std::size_t keep_bytes = 0) noexcept
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
                // no room for the counters: advise them all, as with keep_bytes = 0
            }
        }
        for (auto& [size, lists] : recycled_)
        {
            const std::size_t adv = (size / page) * page;
            for (std::size_t r = 0; r < lists.by_rank.size(); ++r)
            {
                auto& v = lists.by_rank[r];
                // Newest last: pop_recycled() takes from the back, so a rank
                // keeps from the back.
                for (auto rit = v.rbegin(); rit != v.rend(); ++rit)
                {
                    Recycled& e = *rit;
                    if (e.advised)
                    {
                        continue;
                    }
                    if (!kept.empty() && kept[r] + size <= per_rank)
                    {
                        kept[r] += size; // stays resident, for the next allocate()
                        continue;
                    }
                    const auto b = reinterpret_cast<std::uintptr_t>(e.block);
                    if (adv == 0 || b % page != 0)
                    {
                        e.advised = true; // never advisable: do not look again
                        continue;
                    }
                    try
                    {
                        spans.push_back(Span{.begin = b, .len = adv});
                    }
                    catch (...)
                    {
                        ::madvise(e.block, adv, MADV_DONTNEED); // no room for the batch: now
                        reclaim_syscalls().fetch_add(1, std::memory_order_relaxed);
                        reclaim_bytes().fetch_add(adv, std::memory_order_relaxed);
                    }
                    e.advised = true;
                    ++slabs;
                }
                if (!kept.empty())
                {
                    // Resident slabs behind the advised ones: pop_recycled()
                    // hands out the back first, pages already there.
                    std::stable_partition(v.begin(), v.end(), [](const Recycled& e) { return e.advised; });
                }
            }
        }
        std::ranges::sort(spans, [](const Span& a, const Span& b) { return a.begin < b.begin; });
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

    /**
     * @brief Bytes of this arena's mappings resident RIGHT NOW (mincore()).
     * @details `live` counts slabs handed out, madvise'd ones included, and the
     *          reclaim counters are cumulative: neither is resident memory at a
     *          given moment. One mincore() per mapping, one byte of scratch per
     *          page -- meant for a trace point, not a hot path.
     */
    [[nodiscard]] std::size_t resident_bytes() const noexcept
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
        for (const auto& r : regions_)
        {
            const std::size_t len   = r.end - r.begin;
            const std::size_t pages = (len + psz - 1) / psz;
            vec.resize(pages);
            if (::mincore(detail::addr_ptr(r.begin), len, vec.data()) != 0)
            {
                continue;
            }
            for (const unsigned char v : vec)
            {
                res += (v & 1u) != 0u ? psz : 0;
            }
        }
        return res;
    }

    /// @brief Bytes currently handed out (slabs, madvise'd ones included).
    [[nodiscard]] std::size_t live_bytes() const noexcept
    {
        const std::scoped_lock lk(mtx_);
        return bytes_live_;
    }

    /// @brief Bytes of address space reserved by this arena (never shrinks).
    [[nodiscard]] std::size_t mapped_bytes() const noexcept
    {
        const std::scoped_lock lk(mtx_);
        std::size_t            mapped = 0;
        for (const auto& r : regions_)
        {
            mapped += (r.end - r.begin);
        }
        return mapped;
    }

    /**
     * @brief Print one line of arena accounting on stderr.
     * @param when Free-text label for the moment of the call.
     *
     * @details The four numbers answer different questions:
     *          - `mapped`   virtual reservation, the sum of the mmap regions.
     *                       Never shrinks, and is NOT resident memory -- a 256
     *                       MiB reservation costs nothing until touched.
     *          - `live`     bytes currently handed out.
     *          - `high`     high-water mark of `live` since the run began.
     *          - `freelist` bytes deallocated and recycled internally but still
     *                       RESIDENT. This is the reclaim target.
     */
    void report(const char* when) const noexcept
    {
        const std::scoped_lock lk(mtx_);
        std::size_t            mapped = 0;
        for (const auto& r : regions_)
        {
            mapped += (r.end - r.begin);
        }
        std::size_t freelist = 0;
        for (const auto& kv : free_lists_)
        {
            freelist += kv.first * kv.second.size();
        }
        std::size_t recycled = 0;
        for (const auto& kv : recycled_)
        {
            recycled += kv.first * kv.second.count;
        }
        // CHUNK_ARENA_REPORT asked for it: a report, written whatever CHUNK_LOG says.
        CHUNK_LOG_ALWAYS(arena,
                         "{:<6} {:<16} mapped={:8} KiB  live={:8} KiB  high={:8} KiB  freelist={:8} KiB  regions={}  "
                         "allocs={}  deallocs={}  deallocMiB={}  recycled={} KiB (in={} out={} foreign={} advised={})",
                         name_, when, mapped >> 10U, bytes_live_ >> 10U, bytes_high_ >> 10U, freelist >> 10U,
                         regions_.size(), alloc_calls_, dealloc_calls_, dealloc_bytes_ >> 20U, recycled >> 10U,
                         recycled_in_, recycled_out_, recycled_foreign_, recycled_advised_);
    }


  private:
    /// @brief One mmap reservation, as a half-open address range.
    struct Region
    {
        std::uintptr_t begin; ///< First address of the mapping.
        std::uintptr_t end;   ///< One past its last address.
    };

    /// @brief A slab of @p size from the recycled lists (recycle_batch), or
    ///        null: rank @p rank's list first, then the following ranks' in
    ///        turn (neighbouring ranks, usually the same socket).
    /// @pre The lock is held.
    void* pop_recycled(std::size_t size, int rank) noexcept
    {
        const auto it = recycled_.find(size);
        if (it == recycled_.end() || it->second.count == 0)
        {
            return nullptr;
        }
        auto&             lists = it->second.by_rank;
        const std::size_t n     = lists.size();
        const bool        known = rank >= 0 && std::cmp_less(rank, n);
        const std::size_t start = known ? static_cast<std::size_t>(rank) : 0;
        for (std::size_t i = 0; i < n; ++i)
        {
            auto& v = lists[(start + i) % n];
            if (!v.empty())
            {
                void* const block = v.back().block;
                v.pop_back();
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

    /// @brief Map a fresh reservation and restart the bump cursor in it.
    /// @param min_bytes Lower bound for the new region; larger than
    ///        #reservation_chunk_ rounds up to a 2 MiB multiple.
    /// @warning Abandons the tail of the current region. @see SlabArena
    void grow(std::size_t min_bytes)
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
        auto region_begin = reinterpret_cast<std::uintptr_t>(base);
        regions_.push_back(Region{.begin = region_begin, .end = region_begin + want});
        cursor_     = region_begin;
        region_end_ = region_begin + want;
    }

    /// @brief Round @p v up to a multiple of @p alignment. @p alignment must be a power of two.
    static std::uintptr_t align_up(std::uintptr_t v, std::size_t alignment) noexcept
    {
        return (v + (alignment - 1)) & ~(static_cast<std::uintptr_t>(alignment) - 1);
    }

    const char*         name_;              ///< Diagnostic label only.
    std::size_t         reservation_chunk_; ///< Size of each new mmap region.
    std::uintptr_t      cursor_     = 0;    ///< Bump pointer inside the current region.
    std::uintptr_t      region_end_ = 0;    ///< End of the current region.
    std::vector<Region> regions_;           ///< Every reservation, for the reports.

    /// @brief Recycled slabs, keyed by their exact allocation size.
    std::unordered_map<std::size_t, std::vector<void*>> free_lists_;

    /// @brief Slabs currently handed out. Exists so allocate() can refuse to
    ///        recycle one that is still live -- the corruption check.
    std::unordered_set<std::uintptr_t> live_;

    /// @brief A slab given back by recycle_batch(); advised = its pages were
    ///        dropped since (advise_recycled()).
    struct Recycled
    {
        void* block;
        bool  advised;
    };
    /// @brief One size's recycled slabs, a list per rank that gave them back.
    struct RecycledLists
    {
        std::vector<std::vector<Recycled>> by_rank;
        std::size_t                        count = 0; ///< Over every rank.
    };
    /// @brief Recycled slabs, keyed by their exact allocation size. Taken by
    ///        allocate() before #free_lists_. Never holds a live slab.
    std::unordered_map<std::size_t, RecycledLists> recycled_;
    std::size_t                                    recycled_in_      = 0; ///< Slabs ever given to recycle_batch().
    std::size_t                                    recycled_out_     = 0; ///< ... and taken back by allocate().
    std::size_t                                    recycled_foreign_ = 0; ///< ... by another rank than the giver.
    std::size_t                                    recycled_advised_ = 0; ///< ... advised before being taken.

    std::size_t        bytes_live_    = 0; ///< Bytes currently handed out.
    std::size_t        bytes_high_    = 0; ///< High-water mark of #bytes_live_.
    std::size_t        alloc_calls_   = 0; ///< allocate() calls that handed out a slab.
    std::size_t        dealloc_calls_ = 0; ///< deallocate() calls that reclaimed one.
    std::size_t        dealloc_bytes_ = 0; ///< Bytes ever returned through deallocate().
    mutable std::mutex mtx_;               ///< Guards everything above.
};

/**
 * @brief The arena backing the deferred-operation pool.
 *
 * @warning IMMORTAL on purpose: heap-allocated and never deleted, so it
 *          outlives EVERY static and thread_local, including the
 *          TCTypeCAllocator instances whose destructors call
 *          value_arena().deallocate() at exit. Do not make it a function-local
 *          `static SlabArena`: it would be destroyed at exit and a later
 *          destructor would touch a dead arena.
 */
inline SlabArena& op_arena()
{
    static auto* arena = new SlabArena("ops");
    return *arena;
}

/// @brief The arena backing the deferred-operation CONTEXT pool.
/// @copydetails op_arena
inline SlabArena& ctx_arena()
{
    static auto* arena = new SlabArena("ctx");
    return *arena;
}

/// @brief The arena backing the chunk value slabs.
/// @copydetails op_arena
inline SlabArena& value_arena()
{
    static auto* arena = new SlabArena("values");
    return *arena;
}

/**
 * @brief Which arena an allocator draws its slabs from.
 *
 * @details The three must never share an address: value data or a ctx layout
 *          landing on a live op overwrites its vtable. Hence three reservations.
 *
 * @warning The default is Op. A pool that holds values or contexts has to say
 *          so before its first slab -- @see TCSlabPool::set_arena.
 */
enum class ArenaKind : std::uint8_t
{
    op,   ///< Deferred operations.
    ctx,  ///< Their contexts.
    value ///< Chunk value buffers.
};

/// @brief The arena for @p k.
inline SlabArena& arena_for(ArenaKind k)
{
    switch (k)
    {
        case ArenaKind::ctx:
            return ctx_arena();
        case ArenaKind::value:
            return value_arena();
        case ArenaKind::op:
        default:
            return op_arena();
    }
}

/// @brief Is arena reporting switched on?
/// @details Reads CHUNK_ARENA_REPORT from the environment, ONCE. Setting it
///          after the first call has no effect.
inline bool arena_report_enabled() noexcept
{
    static const bool on = []
    {
        const char* env = std::getenv("CHUNK_ARENA_REPORT");
        return env && env[0] && env[0] != '0';
    }();
    return on;
}

/**
 * @brief Report all three arenas and the reclaim counters, if enabled.
 * @param when Free-text label for the moment of the call.
 * @details Called at each end-of-parallel-region drain point: tells an RSS gap
 *          of recycled-but-resident free-list bytes (EmptySlabSweep can reclaim
 *          them) from one of genuinely live bytes.
 */
inline void report_all_arenas(const char* when) noexcept
{
    if (!arena_report_enabled())
    {
        return;
    }
    op_arena().report(when);
    ctx_arena().report(when);
    value_arena().report(when);
    CHUNK_LOG_ALWAYS(
        arena,
        "reclaim {:<16} madvise_calls={}  slabs={}  advised={} MiB  (emptied-slab calls={}, skipped: still-live={} "
        "unaligned={} too-small={})  sweep={:.1f} ms in {} flushes (threads={})",
        when, reclaim_syscalls().load(std::memory_order_relaxed), reclaim_calls().load(std::memory_order_relaxed),
        reclaim_bytes().load(std::memory_order_relaxed) >> 20U, reclaim_entered().load(std::memory_order_relaxed),
        reclaim_skip_live().load(std::memory_order_relaxed), reclaim_skip_unaligned().load(std::memory_order_relaxed),
        reclaim_skip_small().load(std::memory_order_relaxed),
        static_cast<double>(reclaim_sweep_ns().load(std::memory_order_relaxed)) * 1e-6,
        reclaim_sweeps().load(std::memory_order_relaxed), detail::reclaim_sweep_threads());
    CHUNK_LOG_ALWAYS(arena, "freepath {:<15} freed={}  parked_own={}  parked_foreign={}  kept_no_migrate={}", when,
                     freepath_freed().load(std::memory_order_relaxed),
                     freepath_park_own().load(std::memory_order_relaxed),
                     freepath_park_foreign().load(std::memory_order_relaxed),
                     freepath_kept_nomigrate().load(std::memory_order_relaxed));
    CHUNK_LOG_ALWAYS(arena, "freepath {:<15} materialised: keyed={}  keyless={}", when,
                     freepath_alloc_keyed().load(std::memory_order_relaxed),
                     freepath_alloc_key0().load(std::memory_order_relaxed));
}

} // namespace fast_alloc
