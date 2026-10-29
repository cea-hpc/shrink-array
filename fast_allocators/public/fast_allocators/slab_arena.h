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
#include <bit>           // std::bit_cast (addr_ptr)
#include <cstddef>       // std::size_t, std::byte (region_base_)
#include <cstdint>       // std::uintptr_t, std::uint8_t
#include <cstdlib>       // std::getenv
#include <functional>    // std::less (the sweep orders slab pointers)
#include <utility>       // std::cmp_less
#include <vector>        // std::vector
#include <mutex>         // std::mutex, std::scoped_lock
#include <atomic>        // std::atomic, std::memory_order_relaxed
#include <charconv>      // std::from_chars (CHUNK_RECLAIM_THREADS)
#include <chrono>        // the sweep timer
#include <cstring>       // std::strerror
#include <string_view>   // std::string_view (reclaim_env_number)
#include <unordered_map> // std::unordered_map
#include <unordered_set> // std::unordered_set
#include <system_error>  // std::errc (std::from_chars)

#include "fast_allocators/internal/raw_memory.h"   // address_bits
#include "fast_allocators/internal/alloc_config.h" // reclaim_min_empty_slabs
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
    static std::atomic<std::size_t> count{0};
    return count;
}
inline std::atomic<std::size_t>& reclaim_skip_unaligned() noexcept
{
    static std::atomic<std::size_t> count{0};
    return count;
}
inline std::atomic<std::size_t>& reclaim_skip_small() noexcept
{
    static std::atomic<std::size_t> count{0};
    return count;
}
inline std::atomic<std::size_t>& reclaim_entered() noexcept
{
    static std::atomic<std::size_t> count{0};
    return count;
}
///@}

/// @brief Free pages given back from slabs still in use
///        (TCSlab::advise_free_pages): bytes, madvise()
///        calls, and slabs in use scanned.
///@{
inline std::atomic<std::size_t>& reclaim_partial_bytes() noexcept
{
    static std::atomic<std::size_t> count{0};
    return count;
}
inline std::atomic<std::size_t>& reclaim_partial_syscalls() noexcept
{
    static std::atomic<std::size_t> count{0};
    return count;
}
inline std::atomic<std::size_t>& reclaim_partial_scanned() noexcept
{
    static std::atomic<std::size_t> count{0};
    return count;
}
///@}

/// @brief A global counter alone on its cache lines.
/// @details Do not use a bare `static std::atomic` for a hot counter: the linker
///          packs it next to other globals (e.g. the head of
///          zero_one_value::Proxy<double>::share_info_array, read by every zero_one_value write) and
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
    const std::string_view text{env};
    const char* const      end   = text.data() + text.size();
    std::size_t            value = 0;
    const auto             res   = std::from_chars(text.data(), end, value);
    if (res.ec != std::errc{} || res.ptr != end)
    {
        CHUNK_LOG(warn, env, "{}=\"{}\" is not a valid value: ignored, {} kept", name, env, fallback);
        return fallback;
    }
    return value;
}

/// @brief CHUNK_RECLAIM_THREADS (default 1 = serial): threads issuing the
///        sweep's madvise calls; 0 = the backend's team size (pmp::max_threads(),
///        1 in a serial build).
inline int reclaim_sweep_threads() noexcept
{
    static const int threads = []
    {
        const std::size_t value = reclaim_env_number("CHUNK_RECLAIM_THREADS", 1);
        if (value == 0)
        {
            return pmp::max_threads();
        }
        return static_cast<int>(std::min<std::size_t>(value, 4096));
    }();
    return threads;
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

/// @brief Address @p addr as a syscall argument (madvise, mincore). The arena
///        keeps its addresses as integers (cursor, live set, runs); the kernel
///        takes them as such, nothing dereferences the result. A block handed
///        to a caller is a pointer stepped from its region's base instead
///        (SlabArena::allocate), which keeps its provenance.
inline void* addr_ptr(std::uintptr_t addr) noexcept
{
    return std::bit_cast<void*>(addr);
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
    if ((fast_alloc::address_bits(base) % up) != 0)
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
 *          key 0), runs detail::reclaimable_bytes() on each slab
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
    void add(Slab* slab) noexcept;

    /// @brief Advise every collected slab away, one madvise() per run.
    /// @details Serial part: sort, drop duplicates, checks, runs
    ///          (merging adjacent slabs), every backptr_dropped flag and the
    ///          counters. Then the madvise calls, from CHUNK_RECLAIM_THREADS
    ///          threads (serial by default): every flag is set before the first
    ///          madvise, and the syscalls touch disjoint ranges. Timed
    ///          (reclaim_sweep_ns / reclaim_sweeps).
    /// @return Number of slabs advised.
    std::size_t flush() noexcept;

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
    void advise_runs() noexcept;

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
    void* allocate(std::size_t size, std::size_t align, int rank = -1);

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
    void deallocate(void* block, std::size_t size) noexcept;

    /**
     * @brief Take back @p count slabs a re-chunk has just emptied, for the next
     *        allocate() of rank @p rank. One lock for the whole batch.
     * @param blocks The slabs, as returned by allocate(); their headers are
     *        gone (TCSlabPool::recycle_empty_slabs). Null entries are skipped.
     * @param count Number of entries in @p blocks.
     * @param size Their allocation size: the key allocate() looks them up by.
     * @param rank The rank that held them (-1 counts as 0): allocate(size, ..,
     *        rank) takes them back first, their pages having been first
     *        touched there.
     * @details Unlike deallocate(), the slabs KEEP their pages: the re-chunk's
     *          next slice allocates about as much at the same slab size, and a
     *          resident slab costs no page fault nor kernel zeroing. What the last
     *          slice does not take stays resident too, for the next allocations
     *          (the re-chunk no longer calls advise_recycled()).
     * @note An unknown or already-freed slab warns and is dropped, as in
     *       deallocate().
     */
    void recycle_batch(void* const* blocks, std::size_t count, std::size_t size, int rank) noexcept;

    /**
     * @brief Give back the pages of every recycled slab nobody took and that
     *        was not advised yet: one madvise per run of adjacent slabs.
     * @param keep_bytes Leave up to this many bytes of not-yet-advised slabs
     *        resident, split evenly over the ranks (0 = advise them all).
     *        Each rank keeps its newest ones -- the end of its list, which
     *        allocate() takes first -- and its resident slabs are moved
     *        behind its advised ones, so they are handed out first.
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
    std::size_t advise_recycled(std::size_t keep_bytes = 0) noexcept;

    /**
     * @brief Bytes of this arena's mappings resident RIGHT NOW (mincore()).
     * @details `live` counts slabs handed out, madvise'd ones included, and the
     *          reclaim counters are cumulative: neither is resident memory at a
     *          given moment. One mincore() per mapping, one byte of scratch per
     *          page -- meant for a trace point, not a hot path.
     */
    [[nodiscard]] std::size_t resident_bytes() const noexcept;

    /// @brief Bytes currently handed out (slabs, madvise'd ones included).
    [[nodiscard]] std::size_t live_bytes() const noexcept
    {
        const std::scoped_lock lk(mtx_);
        return bytes_live_;
    }

    /// @brief Bytes of address space reserved by this arena (never shrinks).
    [[nodiscard]] std::size_t mapped_bytes() const noexcept;

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
    void report(const char* when) const noexcept;


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
    void* pop_recycled(std::size_t size, int rank) noexcept;

    /// @brief Map a fresh reservation and restart the bump cursor in it.
    /// @param min_bytes Lower bound for the new region; larger than
    ///        #reservation_chunk_ rounds up to a 2 MiB multiple.
    /// @warning Abandons the tail of the current region. @see SlabArena
    void grow(std::size_t min_bytes);

    /// @brief Round @p value up to a multiple of @p align. @p align must be a power of two.
    static std::uintptr_t align_up(std::uintptr_t value, std::size_t align) noexcept
    {
        return (value + (align - 1)) & ~(static_cast<std::uintptr_t>(align) - 1);
    }

    const char*         name_;                  ///< Diagnostic label only.
    std::size_t         reservation_chunk_;     ///< Size of each new mmap region.
    std::byte*          region_base_ = nullptr; ///< The current region's start: allocate() steps from it.
    std::uintptr_t      cursor_      = 0;       ///< Bump pointer inside the current region.
    std::uintptr_t      region_end_  = 0;       ///< End of the current region.
    std::vector<Region> regions_;               ///< Every reservation, for the reports.

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

/// @brief The arena for @p kind.
inline SlabArena& arena_for(ArenaKind kind)
{
    switch (kind)
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
    CHUNK_LOG_ALWAYS(arena, "partial {:<16} madvise_calls={}  advised={} MiB  (slabs in use scanned={})", when,
                     reclaim_partial_syscalls().load(std::memory_order_relaxed),
                     reclaim_partial_bytes().load(std::memory_order_relaxed) >> 20U,
                     reclaim_partial_scanned().load(std::memory_order_relaxed));
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

#include "fast_allocators/internal/slab_arena_inline.h" // the template bodies
