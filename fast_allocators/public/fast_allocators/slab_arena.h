/**
 * @file slab_arena.h
 * @brief Disjoint address-space arenas, so two slab pools can never collide.
 *
 * @par The bug this exists to make impossible
 * The op pool and the value pool both took their slabs from the general
 * allocator. When an op slab was freed -- thread-local allocator destruction
 * between parallel regions -- its address went back to glibc, and the value
 * pool's next large allocation could reobtain THE SAME ADDRESS while a deferred
 * op still lived there. The value writes then overwrote the op's vtable
 * pointer, which is how it was found: data bytes where a vtable should be, at
 * op+0.
 *
 * @par The fix
 * Each pool gets its own arena: one large mmap reservation carved by a bump
 * pointer, never unmapped. An address handed out by one arena can never be
 * handed out by another, because it was never given back to anyone who could.
 * Freed slabs are still recycled, but only WITHIN their own arena, through a
 * free-list keyed by size.
 *
 * @par What that costs
 * A slab returned with deallocate() stays resident: the arena keeps it, the OS
 * never gets it back. That is the RSS the free-list holds, and what
 * madvise_empty_slab() addresses -- it drops an emptied slab's pages while
 * leaving the address valid and the slab reusable.
 *
 * @note Three arenas, not two: ctx slabs were later found to alias op slabs the
 *       same way. @see ArenaKind
 *
 * @warning The arenas are deliberately leaked. They must outlive every static
 *          and thread_local that might free into them at teardown. @see op_arena
 */

#pragma once

#include <sys/mman.h>
#include <unistd.h>
#include "fast_allocators/internal/alloc_config.h" // thp_advise (CHUNK_THP)
#include <algorithm> // std::max, the high-water mark
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <mutex>
#include <atomic>
#include <unordered_map>
#include <unordered_set>

namespace fast_alloc
{

// The reclaim counters and helper live HERE, in the lowest-level header, so the
// call sites in tc_allocator.h and report_all_arenas() below all see one
// definition -- no cross-header ordering or two-phase-lookup question.

/// @brief How many times madvise_empty_slab() has advised a slab away.
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

/// @brief Why a slab was NOT advised away, counted per reason. An emptied slab
///        that madvise cannot take is the difference between "the drain works"
///        and "the memory comes back"; without these the two look alike from
///        the outside (madvise_calls stays 0 either way).
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
/// @details A bare `static std::atomic` is 8 bytes the linker packs next to
///          whatever comes before and after it. Two of these landed on the
///          line holding the head of Proxy2<double>::share_info_array, which
///          every Chunk2 write reads: each fetch_add then invalidated that
///          line on all the other cores (false sharing; 36% of a 64-thread
///          run attributed to get_info_second_value, a read-only function).
///          128 rather than 64: adjacent-line prefetch, and 128-byte lines.
struct alignas(128) IsolatedCounter
{
    std::atomic<std::size_t> v{0};
};
static_assert(sizeof(IsolatedCounter) == 128, "IsolatedCounter must fill its alignment");

/// @brief What the chunk layer did with a buffer it stopped using. A slab can
///        only empty if blocks actually reach the allocator, so when
///        "emptied-slab calls" is 0 these say whether any free ever happened:
///        parked blocks are retained by design, freed ones are not. Bumped
///        through CHUNK_FREEPATH_INC (chunk/freepath_counters.h).
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
///        and with one. A slab can only empty if it holds a single array's
///        blocks, so a large keyless count while keys were granted means the
///        key does not reach the allocation: several arrays then share slabs
///        and none of them can ever empty, whatever the drain frees.
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


/**
 * @brief Drop an emptied slab's resident pages, keeping it mapped and reusable.
 *
 * @tparam Slab Duck-typed on `nn_ref`, `data` and `BlockAllocator`. A template
 *         purely so those members need only be complete at instantiation.
 * @param slab The slab. Null and non-empty slabs are ignored.
 *
 * @details MADV_DONTNEED returns the pages to the OS but leaves the mapping in
 *          place: the address stays valid, the slab stays on its owner's list,
 *          and the next allocation from it faults fresh zero pages back in.
 *
 * @warning Only ever call this on a slab with `nn_ref == 0`. It re-checks, but
 *          the caller is what makes it true. And it is safe only for a pool
 *          whose free list lives OUTSIDE the slab: the value family keeps links
 *          in a side array, the op/ctx family threads them through the freed
 *          elements and would be destroyed by this.
 *
 * @note The last page is deliberately kept: it holds the trailing back-pointer
 *       that resolves an interior pointer to its slab.
 *
 * @see alloc_config::reclaim_empty_slabs, the build switch on the call site.
 */
template <typename Slab>
void madvise_empty_slab(Slab* slab) noexcept
{
    reclaim_entered().fetch_add(1, std::memory_order_relaxed);
    if (slab != nullptr && slab->nn_ref != 0)
        reclaim_skip_live().fetch_add(1, std::memory_order_relaxed);
    if (slab == nullptr || slab->nn_ref != 0)
        return;
    void*      base = slab->data;
    const auto len  = static_cast<std::size_t>(slab->BlockAllocator);
    if (base == nullptr || len == 0)
        return;
    const long ps = ::sysconf(_SC_PAGESIZE);
    if (ps <= 0)
        return;
    const auto up = static_cast<std::size_t>(ps);
    if ((reinterpret_cast<std::uintptr_t>(base) % up) != 0)
    {
        reclaim_skip_unaligned().fetch_add(1, std::memory_order_relaxed);
        return; // expect page-aligned slab
    }
    std::size_t adv = (len / up) * up;
    // The LAST page holds the trailing self back-pointer that addr_to_block()
    // reads on every free: it must stay resident. A slab of a single page
    // therefore has nothing to give back -- advising it would zero the
    // back-pointer and the next free through that slab would dereference
    // null. (short/char/bool at 64 threads on 64 KiB pages: 60 KiB budget ->
    // one 64 KiB slab -> exactly this case; the old code advised it whole.)
    if (adv < up + up)
    {
        reclaim_skip_small().fetch_add(1, std::memory_order_relaxed);
        return;
    }
    adv -= up;
    ::madvise(base, adv, MADV_DONTNEED);
    reclaim_calls().fetch_add(1, std::memory_order_relaxed);
    reclaim_bytes().fetch_add(adv, std::memory_order_relaxed);
}

/**
 * @brief Give back an emptied slab's pages INCLUDING the last one.
 *
 * @details madvise_empty_slab() keeps the trailing page because it holds the
 *          self back-pointer addr_to_block() reads on every free. That page is
 *          6% of a 1 MiB slab, and at scale it is the whole residue: 17 205
 *          slabs emptied by the four re-chunks of iteration 1000 (64 threads,
 *          10^9 elements, v1) kept 1 075 MB, against 1 032 MB of memory the
 *          re-chunk failed to give back -- the rest of the accounting balances
 *          exactly.
 *
 *          The back-pointer can go because nobody can read it: the slab holds
 *          no block (nn_ref == 0 is checked here), so no free can resolve
 *          through it, and the only way one becomes possible again is an
 *          alloc() from this slab -- which restores the back-pointer before it
 *          hands the block out. The slab carries the flag saying so.
 *
 * @warning Two preconditions, neither checkable here. The caller must know the
 *          slab's emptiness is a FACT, not just what nn_ref says -- the same
 *          rule as reclaim_empty_slabs_now(), which is the only caller. And it
 *          must run where no alloc() from this slab can interleave: the flag is
 *          set before the madvise, so a concurrent alloc could restore the
 *          back-pointer only to have it wiped a moment later. A re-chunk's
 *          end-of-move sweep is such a point; a free path is not.
 *
 * @return true if the pages were dropped and @p slab was flagged.
 */
template <typename Slab>
bool madvise_empty_slab_full(Slab* slab) noexcept
{
    reclaim_entered().fetch_add(1, std::memory_order_relaxed);
    if (slab == nullptr || slab->nn_ref != 0)
    {
        if (slab != nullptr)
            reclaim_skip_live().fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    void*      base = slab->data;
    const auto len  = static_cast<std::size_t>(slab->BlockAllocator);
    if (base == nullptr || len == 0)
        return false;
    const long ps = ::sysconf(_SC_PAGESIZE);
    if (ps <= 0)
        return false;
    const auto up = static_cast<std::size_t>(ps);
    if ((reinterpret_cast<std::uintptr_t>(base) % up) != 0)
    {
        reclaim_skip_unaligned().fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    const std::size_t adv = (len / up) * up;
    if (adv == 0)
    {
        reclaim_skip_small().fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    // Flag first: from here the trailing back-pointer is not to be trusted, and
    // alloc() is what puts it back.
    slab->backptr_dropped = true;
    ::madvise(base, adv, MADV_DONTNEED);
    reclaim_calls().fetch_add(1, std::memory_order_relaxed);
    reclaim_bytes().fetch_add(adv, std::memory_order_relaxed);
    return true;
}


/**
 * @brief A growable bump arena whose mmap reservations are never unmapped.
 *
 * @details Carves slabs out of large reservations with a bump pointer, and
 *          recycles freed ones through a per-size free list. Nothing ever goes
 *          back to the OS, which is the whole point: an address this arena
 *          handed out can never be handed out by another allocator.
 *
 * @note One mutex guards everything. That is affordable because this is not a
 *       hot path -- one call per slab, amortised over every small allocation
 *       carved from it.
 *
 * @warning Growing STRANDS whatever was left of the previous reservation: the
 *          bump cursor jumps to the new region and never returns. With a 256 MiB
 *          default that is a rare event, but a workload of many oversized
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
     * @param fresh If not null, set to whether the slab is fresh from mmap
     *        (never handed out, so still all zeros) or recycled from the free
     *        list (holds its previous occupant's bytes).
     * @return Never null: a failed mmap aborts, matching what posix_memalign
     *         failure used to do here.
     * @details Prefers an exactly-@p size slab from the free list, falls back to
     *          bumping the cursor, and grows the reservation if that does not
     *          fit.
     * @warning Aborts if the free list hands back a slab still marked live --
     *          that would mean a slab was recycled with blocks still in it,
     *          which is the corruption this whole file exists to prevent. The
     *          check is unconditional and fatal on purpose: it is cheap here,
     *          and silent recovery would hide exactly the bug worth finding.
     */
    void* allocate(std::size_t size, std::size_t align, bool* fresh = nullptr)
    {
        const std::scoped_lock lk(mtx_);
        if (fresh != nullptr)
            *fresh = false;

        // Reuse a same-size slab from this arena's free-list if available.
        const auto it = free_lists_.find(size);
        if (it != free_lists_.end() && !it->second.empty())
        {
            void* block = it->second.back();
            it->second.pop_back();
            // SEGV hunt: detect if a slab is handed out while still marked live
            // (would mean it was freed and reused with live blocks in it — the
            // all-zero-op corruption). This should be impossible: a slab on the
            // free-list must have been deallocate()'d (removed from live_).
            if (live_.contains(reinterpret_cast<std::uintptr_t>(block)))
            {
                std::fprintf(stderr,
                             "[ARENA %s] DOUBLE-ALLOC slab %p (size=%zu): handed out while still live "
                             "— a slab was recycled with live blocks still in it\n",
                             name_, block, size);
                std::fflush(stderr);
                std::abort();
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
        if (fresh != nullptr)
            *fresh = true; // below the cursor was never handed out: mmap's zeros
        live_.insert(aligned);
        bytes_live_ += size;
        bytes_high_ = std::max(bytes_high_, bytes_live_);
        ++alloc_calls_;
        return reinterpret_cast<void*>(aligned);
    }

    /**
     * @brief Take a slab back into THIS arena's free list. Never to the OS.
     * @param block The slab, as returned by allocate(); null is a no-op.
     * @param size Must be the size it was allocated with -- the free list is
     *        keyed by it, and a wrong key makes the slab unreachable.
     * @note An unknown or already-freed slab warns and is dropped rather than
     *       aborting: at teardown a ~TCTypeCAllocator may free a slab whose
     *       bookkeeping is already gone, which is harmless. The dangerous
     *       direction -- a live slab handed out again -- is fatal in allocate().
     */
    void deallocate(void* block, std::size_t size) noexcept
    {
        if (block == nullptr)
            return;
        const std::scoped_lock lk(mtx_);
        // SEGV hunt: a slab freed while not live. Warn but do NOT abort — at
        // program teardown a ~TCTypeCAllocator may deallocate() a slab whose
        // bookkeeping is already gone, which is harmless. The DANGEROUS direction
        // (a live slab handed out again) is caught fatally in allocate().
        if (live_.erase(reinterpret_cast<std::uintptr_t>(block)) == 0)
        {
            std::fprintf(stderr, "[ARENA %s] double-free/unknown slab %p (size=%zu) — ignored\n", name_, block, size);
            std::fflush(stderr);
            return; // don't push an unknown slab onto the free-list
        }
        bytes_live_ -= size;
        ++dealloc_calls_;
        dealloc_bytes_ += size;
        free_lists_[size].push_back(block);
    }

  private:
    /// @brief Does @p ptr fall inside one of this arena's reservations?
    /// @note Linear in the region count, and takes the lock. Diagnostics only.
    bool owns(const void* ptr) const noexcept
    {
        const auto             v = reinterpret_cast<std::uintptr_t>(ptr);
        const std::scoped_lock lk(mtx_);
        for (const auto& r : regions_)
            if (v >= r.begin && v < r.end)
                return true;
        return false;
    }

  public:
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
            return 0;
        const auto                 psz = static_cast<std::size_t>(page);
        std::size_t                res = 0;
        std::vector<unsigned char> vec;
        for (const auto& r : regions_)
        {
            const std::size_t len   = r.end - r.begin;
            const std::size_t pages = (len + psz - 1) / psz;
            vec.resize(pages);
            if (::mincore(reinterpret_cast<void*>(r.begin), len, vec.data()) != 0)
                continue;
            for (const unsigned char v : vec)
                res += (v & 1u) ? psz : 0;
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
            mapped += (r.end - r.begin);
        return mapped;
    }

    /**
     * @brief Print one line of arena accounting on stderr.
     * @param when Free-text label for the moment of the call.
     *
     * @details The four numbers answer different questions and are routinely
     *          confused:
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
            mapped += (r.end - r.begin);
        std::size_t freelist = 0;
        for (const auto& kv : free_lists_)
            freelist += kv.first * kv.second.size();
        std::fprintf(stderr,
                     "[ARENA %-6s] %-16s mapped=%8zu KiB  live=%8zu KiB  high=%8zu KiB  "
                     "freelist=%8zu KiB  regions=%zu  allocs=%zu  deallocs=%zu  deallocMiB=%zu\n",
                     name_, when, mapped >> 10U, bytes_live_ >> 10U, bytes_high_ >> 10U, freelist >> 10U,
                     regions_.size(), alloc_calls_, dealloc_calls_, dealloc_bytes_ >> 20U);
        std::fflush(stderr);
    }

  private:
    /// @brief One mmap reservation, as a half-open address range.
    struct Region
    {
        std::uintptr_t begin; ///< First address of the mapping.
        std::uintptr_t end;   ///< One past its last address.
    };

    /// @brief Round @p v up to a multiple of @p alignment. @p alignment must be a power of two.
    static std::uintptr_t align_up(std::uintptr_t v, std::size_t alignment) noexcept
    {
        return (v + (alignment - 1)) & ~(static_cast<std::uintptr_t>(alignment) - 1);
    }

    /// @brief Map a fresh reservation and restart the bump cursor in it.
    /// @param min_bytes Lower bound for the new region; larger than
    ///        #reservation_chunk_ rounds up to a 2 MiB multiple.
    /// @warning Abandons the tail of the current region. @see SlabArena
    void grow(std::size_t min_bytes)
    {
        std::size_t want = reservation_chunk_;
        if (min_bytes > want)
            want = align_up(min_bytes, static_cast<std::size_t>(1) << 21U); // round to 2 MiB

        void* base = ::mmap(nullptr, want, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (base == MAP_FAILED)
        {
            std::fprintf(stderr, "SlabArena(%s): mmap of %zu bytes failed\n", name_, want);
            std::abort();
        }
        // CHUNK_THP (experimental, off by default): the slabs are 2 MiB and
        // aligned on it, a huge page each. Advice only: a failure is ignored.
        if (alloc_config::thp_advise())
            (void)::madvise(base, want, MADV_HUGEPAGE);
        auto region_begin = reinterpret_cast<std::uintptr_t>(base);
        regions_.push_back(Region{.begin = region_begin, .end = region_begin + want});
        cursor_     = region_begin;
        region_end_ = region_begin + want;
    }

    const char*         name_;              ///< Diagnostic label only.
    std::size_t         reservation_chunk_; ///< Size of each new mmap region.
    std::uintptr_t      cursor_     = 0;    ///< Bump pointer inside the current region.
    std::uintptr_t      region_end_ = 0;    ///< End of the current region.
    std::vector<Region> regions_;           ///< Every reservation, for owns().

    /// @brief Recycled slabs, keyed by their exact allocation size.
    std::unordered_map<std::size_t, std::vector<void*>> free_lists_;

    /// @brief Slabs currently handed out. Exists so allocate() can refuse to
    ///        recycle one that is still live -- the corruption check.
    std::unordered_set<std::uintptr_t> live_;

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
 * @warning IMMORTAL, and the leak is the point. It is heap-allocated and never
 *          deleted so it outlives EVERY other static and thread_local --
 *          including the TCTypeCAllocator instances whose destructors call
 *          value_arena().deallocate() at exit. A function-local
 *          `static SlabArena a` would be destroyed at exit too, and a later
 *          destructor touching the dead arena's members segfaults during
 *          teardown; that was observed as a SIGSEGV in
 *          ~unique_ptr<TCTypeCAllocator[]> from __libc_start_main. The memory
 *          has to live for the whole program anyway, so nothing is lost.
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
 * @details The three must never share an address, and each pairing was found
 *          the hard way: op/value had double data overwriting an op's vtable,
 *          op/ctx had ctx layout doing the same. Hence three reservations, not
 *          one shared one.
 *
 * @warning The default is Op. A pool that holds values or contexts has to say
 *          so before its first slab -- @see TCTypeAllocator::set_arena.
 */
enum class ArenaKind : std::uint8_t
{
    Op,   ///< Deferred operations.
    Ctx,  ///< Their contexts.
    Value ///< Chunk value buffers.
};

/// @brief The arena for @p k.
inline SlabArena& arena_for(ArenaKind k)
{
    switch (k)
    {
        case ArenaKind::Ctx:
            return ctx_arena();
        case ArenaKind::Value:
            return value_arena();
        case ArenaKind::Op:
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
 * @details Called at each end-of-parallel-region drain point. What it is for:
 *          telling apart an RSS gap made of recycled-but-resident free-list
 *          bytes -- which madvise_empty_slab() can reclaim -- from one made of
 *          genuinely live bytes, which nothing can.
 */
inline void report_all_arenas(const char* when) noexcept
{
    if (!arena_report_enabled())
        return;
    op_arena().report(when);
    ctx_arena().report(when);
    value_arena().report(when);
    std::fprintf(
        stderr,
        "[RECLAIM] %-16s madvise_calls=%zu  advised=%zu MiB  "
        "(emptied-slab calls=%zu, skipped: still-live=%zu unaligned=%zu too-small=%zu)\n",
        when, reclaim_calls().load(std::memory_order_relaxed), reclaim_bytes().load(std::memory_order_relaxed) >> 20U,
        reclaim_entered().load(std::memory_order_relaxed), reclaim_skip_live().load(std::memory_order_relaxed),
        reclaim_skip_unaligned().load(std::memory_order_relaxed), reclaim_skip_small().load(std::memory_order_relaxed));
    std::fprintf(stderr, "[FREEPATH] %-15s freed=%zu  parked_own=%zu  parked_foreign=%zu  kept_no_migrate=%zu\n", when,
                 freepath_freed().load(std::memory_order_relaxed), freepath_park_own().load(std::memory_order_relaxed),
                 freepath_park_foreign().load(std::memory_order_relaxed),
                 freepath_kept_nomigrate().load(std::memory_order_relaxed));
    std::fprintf(stderr, "[FREEPATH] %-15s materialised: keyed=%zu  keyless=%zu\n", when,
                 freepath_alloc_keyed().load(std::memory_order_relaxed),
                 freepath_alloc_key0().load(std::memory_order_relaxed));
    std::fflush(stderr);
}

} // namespace fast_alloc
