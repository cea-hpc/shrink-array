
#pragma once

/**
 * @file tc_allocator.h
 * @brief Slab allocators: one mmap'd block is carved into fixed-size elements.
 *
 * Four layers, bottom to top:
 *
 *  - ::TCSlab is ONE slab. It owns a block of memory, hands out fixed-size
 *    elements from it, and keeps a free list of the elements handed back. It
 *    is also the node of an intrusive list. ::TCAllocator and ::TCCAllocator
 *    are its two kinds (see below).
 *  - ::sl_insert, ::sl_remove, ::sl_manage_full_empty and ::sl_delete_all are
 *    the list primitives, shared by every family so a fix lands in one place.
 *  - ::TCSlabPool owns a SET of slabs for one element size, split three ways:
 *    `block` (room left), `block_full`, and `block_empty` (a small reserve of
 *    emptied slabs kept for reuse). ::TCTypeAllocator and ::TCTypeCAllocator
 *    are its two pools, one per slab kind, with what is theirs alone.
 *  - ::TCTypeAllocatorEBC is a third family whose element and block sizes are
 *    template parameters rather than run-time values.
 *
 * @par Why two slab kinds
 * ::TCAllocator threads its free list THROUGH the freed elements, writing the
 * next free index into the element's first bytes. ::TCCAllocator does not: it
 * keeps the links in a side array (TCSlab::tab_firstFree) and never writes
 * into a free element, so an emptied slab's pages can be dropped with madvise()
 * without losing anything the allocator needs. That side array is indexed by
 * `unsigned short`, which is what caps a TCCAllocator slab at
 * TCSlab::free_list_end - 1 elements. ::FreeLinks names the choice.
 *
 * @par Configuration
 * The macros are read in alloc_config.h and nowhere else, so the bodies here
 * branch with `if constexpr`. Both arms must compile in every configuration --
 * that is the point. See ::bypass_slabs.
 *
 * @warning A slab's own address is recovered from any interior pointer by
 *          masking (TCSlab::slab_from), which requires the block to be
 *          aligned on its own size and to store a back-pointer in its last
 *          word. Both are established by alloc_block(); neither is checked
 *          afterwards.
 */

#include <vector>      // std::vector
#include <array>       // std::array
#include <bit>         // std::has_single_bit, bit_width, bit_ceil, bit_floor
#include <cerrno>      // errno, after a failed mmap
#include <cstddef>     // std::size_t
#include <cstdio>      // std::fprintf / std::fflush (the [MUNMAP] traces)
#include <cstring>     // std::strerror
#include <iostream>    // std::cerr
#include <cstdlib>     // std::abort, std::getenv (FAST_ALLOC_TRACE)
#include <cstdint>     // uintptr_t
#include <limits>      // std::numeric_limits: TCAllocatorEB's index range
#include <type_traits> // std::conditional_t (free_index_t)
#include <cassert>     // assert (DBG_ASSERT)
#include <sys/mman.h>  // mmap, munmap
#include <unistd.h>    // sysconf (fast_alloc::page_size)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "fast_allocators/internal/alloc_config.h"    // constexpr mirrors of STD_ALLOC / DEBUG / _OPENMP
#include "fast_allocators/slab_arena.h"               // ArenaKind for op/ctx/value slab routing
#include "fast_allocators/internal/next_power_of_2.h" // NextPowerOf2, for slab_of()'s effective slab geometry

/**
 * @brief Must this call hand out plain `new[]` memory instead of slab memory?
 *
 * @details Under STD_ALLOC, for the whole run; never otherwise. A constant: an
 *          allocator's mode cannot change after init() -- a run-time switch
 *          flipped after init() would make ElementSize wrong for the other
 *          mode (see below).
 *
 * @warning The two arms disagree on what ElementSize means: bypass mode counts
 *          ELEMENTS (it feeds `new ElementType[ElementSize]`), slab mode counts
 *          BYTES. init() sets it accordingly.
 */
[[nodiscard]] constexpr bool bypass_slabs() noexcept
{
    return alloc_config::use_std_alloc;
}

namespace fast_alloc
{

/// @brief The system page size, read once (sysconf).
[[nodiscard]] inline long page_size() noexcept
{
    static const long ps = ::sysconf(_SC_PAGESIZE);
    return ps;
}

/// @brief FAST_ALLOC_TRACE is set in the environment: trace the evictions of
///        emptied slabs (the [MUNMAP] lines). Read once.
[[nodiscard]] inline bool trace_evictions() noexcept
{
    static const bool on = std::getenv("FAST_ALLOC_TRACE") != nullptr;
    return on;
}

/**
 * @brief Map @p size anonymous, zero-filled bytes, or abort.
 * @param size Byte count; the caller is expected to have rounded it to pages.
 * @return The mapping. Never null -- a failure prints errno and aborts, because
 *         every caller here is on a path with no way to report failure upward.
 */
inline void* map_anonymous(size_t size)
{
#if defined(BSD)
    void* const tmp = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
#else
    // MAP_ANONYMOUS: not backed by any file; the contents are zero (mmap(2)).
    void* const tmp = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
#endif

    if (tmp == MAP_FAILED)
    {
        const int err = errno;
        std::cerr << "mmap failed: errno = " << err << " (" << std::strerror(err) << ")\n";
        std::cerr << "size = " << size << "\n";
        std::abort();
    }

    return tmp;
}

} // namespace fast_alloc

/// @brief assert() in a DEBUG build; expands to nothing otherwise.
#ifdef DEBUG
    #define DBG_ASSERT(x) assert(x)
#else
    #define DBG_ASSERT(x)
#endif

/**
 * @brief Marks the slow path of a hot function: kept out of line and placed
 *        with the cold code, so the hot path stays small enough to inline and
 *        its instruction cache holds nothing it does not need.
 * @details Slab creation (a `new`, an arena lock) and slab eviction (an
 *          fprintf, a `delete`) are rare next to the bump / free-list step
 *          they surround, but inlined they were the bulk of alloc() and
 *          free_elt_pow2().
 */
#if defined(__GNUC__) || defined(__clang__)
    #define FAST_ALLOC_COLD __attribute__((cold, noinline))
#else
    #define FAST_ALLOC_COLD
#endif


/// @brief Where a slab keeps the links of its free list.
enum class FreeLinks
{
    /// In the freed elements themselves: a freed element's first bytes hold
    /// the index of the next free one. No extra memory, but an element must be
    /// at least an `int` wide and its contents do not survive a free.
    InPayload,
    /// In a side array (TCSlab::tab_firstFree), never in the elements: the
    /// allocator never writes into a free element, so an emptied slab's pages
    /// can be dropped (fast_alloc::EmptySlabSweep) without losing a
    /// link. The array is `unsigned short`, which caps a slab at
    /// TCSlab::free_list_end - 1 elements.
    SideArray
};

/**
 * @brief One slab of fixed-size elements, and a node of a slab list.
 *
 * @details The block is carved linearly while it is fresh (`UsedSize` is a bump
 *          index) and from a free list afterwards; @p Links says where that
 *          free list keeps its links. Freeing the element the bump pointer just
 *          handed out rewinds the bump instead, so an allocate/free pair leaves
 *          the slab exactly as it found it.
 *
 *          The block comes from the arena named by #arena_kind (op, ctx or
 *          value: disjoint address spaces, see slab_arena.h), is aligned on its
 *          own power-of-two size, and stores `this` in its last word: slab_from()
 *          recovers the slab from any interior pointer by masking.
 *
 * @warning NO content guarantee. An element comes back with whatever its
 *          previous owner left in it -- through the free list, a rewound bump
 *          pointer, a slab revived from the reserve, or a slab the arena
 *          recycles. Only a slab fresh from the arena (mmap) is zero. Callers
 *          initialise what they get: ChunkOmpPolicy::zero_fill_fresh() for v1,
 *          the fill functors of allocate_chunk_internal() for v2/v2A; the op
 *          pools construct every op at its own block.
 *
 * @tparam Links Where the free list keeps its links: ::TCAllocator threads it
 *         through the elements, ::TCCAllocator keeps it aside.
 */
template <FreeLinks Links>
class alignas(64) TCSlab
{
  public:
    /// @brief The links live in #tab_firstFree rather than in the elements.
    static constexpr bool side_links = Links == FreeLinks::SideArray;

    /// @brief An element index on the free list.
    using index_type = std::conditional_t<side_links, unsigned short, int>;

    /// @brief Free-list terminator. With side links it is also the exclusive
    ///        upper bound on #BlockSize: the side array indexes elements with it.
    static constexpr index_type free_list_end = static_cast<index_type>(side_links ? 32767 : -1);

    /// @brief Leaves the slab unusable; init() is what makes it a slab.
    TCSlab() = default;

    /// @brief Returns the block to its arena.
    ~TCSlab()
    {
        destroy();
    }

    // Owns its block (and #tab_firstFree): a copy would release them twice.
    TCSlab(const TCSlab&)            = delete;
    TCSlab& operator=(const TCSlab&) = delete;
    TCSlab(TCSlab&&)                 = delete;
    TCSlab& operator=(TCSlab&&)      = delete;

    /// @brief The size this slab was cut at -- slab_bytes_for() of the request.
    [[nodiscard]] uintptr_t slab_bytes() const noexcept
    {
        return BlockAllocator;
    }

    /**
     * @brief Set the geometry and obtain the block.
     * @param ElementSize_ Element size in BYTES.
     * @param BlockAllocator_ Requested block size in bytes; rounded up to a
     *        power of two (slab_bytes_for()). With side links the capacity, not
     *        the size, is capped to fit the index width.
     * @pre #arena_kind is set: the block is taken from that arena here.
     */
    inline void init(int ElementSize_, uintptr_t BlockAllocator_);

    /**
     * @brief Hand out one element.
     * @return An element with unspecified contents (see the class @warning).
     * @pre The slab is not full -- the caller checks is_full().
     */
    [[nodiscard]] inline void* alloc() noexcept;

    /**
     * @brief Take element @p i back. Its bytes are left as they are, but for
     *        the free-list link written over its head when the links are in
     *        the payload.
     * @param i Index from num_block() / num_block_pow2().
     */
    inline void free_block(int i) noexcept;

    /// @brief Make an emptied slab fresh again: bump index back to the start,
    ///        free list empty. For a slab revived from a pool's reserve.
    /// @pre is_empty().
    void reset() noexcept
    {
        UsedSize  = 0;
        firstFree = free_list_end;
    }

    /// @brief Index of the element at @p ptr within this slab (division).
    [[nodiscard]] int num_block(const void* ptr) const noexcept
    {
        return static_cast<int>((static_cast<const char*>(ptr) - data) / ElementSize);
    }

    /// @brief num_block() by shift. Valid only if #ElementSize is a power of two.
    [[nodiscard]] int num_block_pow2(const void* ptr) const noexcept
    {
        return static_cast<int>((static_cast<const char*>(ptr) - data) >> Log2ElementSize);
    }

    /// @brief No element of this slab is out. Only then may it be released.
    [[nodiscard]] bool is_empty() const noexcept
    {
        return nn_ref == 0;
    }

    /// @brief This free brought the slab back from full; it must leave block_full.
    [[nodiscard]] bool was_full() const noexcept
    {
        return nn_ref == BlockSize - 1;
    }

    /// @brief Every element is out; the slab must move to block_full.
    [[nodiscard]] bool is_full() const noexcept
    {
        return nn_ref == BlockSize;
    }

    /// @brief Effective slab size for a requested @p block_bytes: rounded up to
    ///        a power of two. ONE rule, shared by alloc_block(), the pool's
    ///        init() and slab_of(), so a pool resolves a pointer to its slab
    ///        without a slab to ask.
    /// @details Independent of the element size, on purpose. A buffer is freed
    ///          through slab_of() of the allocator the thread's AMBIENT chunk
    ///          size selects (ChunkBase::release_to_owner), which after a
    ///          re-chunk need not be the size that allocated it; that only
    ///          finds the right slab if every allocator of one element type
    ///          cuts its slabs at the same size. So the side-link index width
    ///          caps the CAPACITY (in alloc_block(); the unused tail is never
    ///          touched), never the slab size: a slab size that depends on the
    ///          element size makes slab_of() read a "slab" pointer out of the
    ///          middle of a block (slab_geometry_validation).
    [[nodiscard]] static uintptr_t slab_bytes_for(uintptr_t block_bytes) noexcept
    {
        return NextPowerOf2(block_bytes);
    }

    /// @brief The slab of @p slab_bytes (a power of two) holding @p ptr:
    ///        mask down to the slab base, read the back-pointer in its last word.
    [[nodiscard]] static TCSlab* slab_from(const void* ptr, uintptr_t slab_bytes) noexcept
    {
        const char* base = reinterpret_cast<const char*>(reinterpret_cast<uintptr_t>(ptr) & (~(slab_bytes - 1)));
        return *reinterpret_cast<TCSlab* const*>(base + slab_bytes - sizeof(void*));
    }

// Declaration-level switch, so it stays a directive: serial builds route slab
// headers through their own pool (tc_allocator.cpp, ::TCAllocator only),
// OpenMP builds use the global operators.
#ifndef _OPENMP
    void* operator new(size_t size);  ///< Slab headers come from their own pool.
    void  operator delete(void* ptr); ///< Counterpart of the above.
#endif

    // Which disjoint arena this slab draws from. Op and ctx pools share a slab
    // class but MUST NOT share an address space (a freed ctx slab reobtained by
    // the op pool would overwrite a live op's vtable with ctx layout). Set by
    // the owning pool before init(); the default is the value arena for side
    // links (value slabs only), op otherwise.
    fast_alloc::ArenaKind arena_kind = side_links ? fast_alloc::ArenaKind::Value : fast_alloc::ArenaKind::Op;

    char* data = nullptr; ///< The block itself, aligned on #BlockAllocator.

    uintptr_t BlockAllocator = 0; ///< Block size in bytes; a power of two.

    index_type firstFree{free_list_end}; ///< Head of the free list; #free_list_end if empty.

    int nn_ref{0}; ///< Elements currently out. 0 == releasable, #BlockSize == full.

    int UsedSize{0}; ///< Bump index: how far the linear carve has reached.

    int ElementSize{0}; ///< Element size in BYTES.

    int BlockSize{0}; ///< Capacity of the block, in elements.

    // for the linked list

    TCSlab* prev = nullptr; ///< Previous slab in the owner's block/block_full list.
    TCSlab* next = nullptr; ///< Next slab in the owner's block/block_full list.

    /// @brief The allocator instance whose lists this slab lives in. Written
    ///        once at creation by that instance, never changed while the slab
    ///        exists (a slab moving to another instance would rewrite it).
    ///        Answers "does this element belong to me?" in O(1) -- the list
    ///        walk it replaces was linear in the instance's slabs and, called
    ///        once per block by the reuse-pool drain, quadratic overall.
    const void* owner = nullptr;
    /// @brief Rank (thread) of #owner, or -1 when the owner has no rank. Lets a
    ///        freeing thread decide "mine, free it" vs "foreign, park it"
    ///        without knowing which allocator set the block came from.
    int owner_rank = -1;
    /// @brief Affinity key of #owner (0 = the rank's shared pool, otherwise the
    ///        array whose blocks this slab holds). Tells a freeing thread which
    ///        (rank, key) reuse list a block belongs to, without a lookup.
    unsigned owner_key = 0;
    /// @brief Set when a sweep (fast_alloc::EmptySlabSweep) dropped this slab's pages
    ///        INCLUDING the trailing back-pointer (side links only: the
    ///        in-payload links would go with the pages). alloc() restores it
    ///        before handing out the first block, which is the only moment a
    ///        free through this slab becomes possible again. Plain bool, no
    ///        atomic: an instance's slabs are allocated from by its own rank
    ///        only, and the flag is set at a serial point with no alloc in flight.
    bool backptr_dropped = false;

  private:
    /// @brief No side array: the links are in the payload.
    struct NoSideArray
    {
    };

    /// @brief The free-list link of element @p i.
    [[nodiscard]] index_type& link(int i) noexcept
    {
        if constexpr (side_links)
        {
            return tab_firstFree[static_cast<std::size_t>(i)];
        }
        else
        {
            // store the index in the first bytes of the element
            return *reinterpret_cast<int*>(data + (static_cast<std::ptrdiff_t>(i) * ElementSize));
        }
    }

    /// @brief Free #tab_firstFree and hand the block back to the arena named
    ///        by #arena_kind.
    inline void destroy();

    /**
     * @brief Obtain the block and lay it out.
     * @details Rounds the block size up to a power of two (slab_from() masks
     *          with it), derives #BlockSize from it -- capped to the index width
     *          with side links, which also sizes #tab_firstFree -- takes the
     *          block from the arena and writes `this` into its last word. The
     *          block is not cleared, fresh or recycled: see the class @warning.
     */
    inline void alloc_block();

    int Log2ElementSize = 0; ///< log2(#ElementSize), for num_block_pow2().

    /// @brief Free-list links, one per element, with side links. Kept out of
    ///        the payload, so an emptied slab's pages can be dropped without
    ///        losing them.
    [[no_unique_address]] std::conditional_t<side_links, std::vector<unsigned short>, NoSideArray> tab_firstFree;
};

/// @brief Slab with its free list threaded through the freed elements: the op
///        and ctx pools, and the value pools of serial builds.
using TCAllocator = TCSlab<FreeLinks::InPayload>;

/// @brief Slab with its free list in a side array: the value pools of OpenMP
///        builds, whose emptied slabs reclaim_empty_slabs_now() madvises away.
using TCCAllocator = TCSlab<FreeLinks::SideArray>;

#ifndef _OPENMP
// Defined in tc_allocator.cpp. Only the in-payload slab is ever created in a
// serial build: ::TCTypeCAllocator exists in OpenMP builds only.
template <>
void* TCSlab<FreeLinks::InPayload>::operator new(size_t size);
template <>
void TCSlab<FreeLinks::InPayload>::operator delete(void* ptr);
#endif

/// @brief A whole-list check (sl_check, sl_contains): linear in the list, on
///        every insert and remove -- quadratic over a run, which made Debug
///        builds crawl once a pool held thousands of slabs. Only at the
///        reinforced debug level (FAST_ALLOC_CHECK_LISTS on a DEBUG build, see
///        alloc_config::check_slab_lists); a plain Debug build keeps the cheap
///        DBG_ASSERTs.
#define SL_LIST_CHECK(x)                              \
    do                                                \
    {                                                 \
        if constexpr (alloc_config::check_slab_lists) \
        {                                             \
            assert(x);                                \
        }                                             \
    } while (0)

// ---------------------------------------------------------------------------
// Intrusive doubly-linked slab-list primitives, shared by the three slab
// families (TCTypeAllocator / TCTypeCAllocator / TCTypeAllocatorEBC) so a fix
// lands in one place. Node must expose `Node* next; Node* prev;`.
// ---------------------------------------------------------------------------
/**
 * @brief Is @p node in the list headed by @p list?
 * @note Recursive and linear -- a debug aid, called only from DBG_ASSERT.
 */
template <typename Node>
[[nodiscard]] bool sl_contains(const Node* list, const Node* node) noexcept
{
    return (list != nullptr) && ((list == node) || sl_contains(list->next, node));
}

/**
 * @brief Is the list headed by @p node well formed?
 * @details The head has no predecessor and nothing after it does either.
 *          Catches a node inserted into two lists at once, which is how a slab
 *          ends up in `block` and `block_full` simultaneously.
 * @note Linear -- a debug aid, called only from DBG_ASSERT.
 */
template <typename Node>
[[nodiscard]] bool sl_check(const Node* node) noexcept
{
    if (node == nullptr)
    {
        return true;
    }
    if (node->prev != nullptr)
    {
        return false;
    }
    node = node->next;
    while (node != nullptr)
    {
        if (node->prev == nullptr)
        {
            return false;
        }
        node = node->next;
    }
    return true;
}

/**
 * @brief Unlink @p node from the list headed by @p list.
 * @param list Head pointer, updated in place when @p node was the head.
 * @param node Must be in that list; the debug build asserts it.
 * @post @p node's links are nulled, so it can be inserted elsewhere.
 */
template <typename Node>
void sl_remove(Node*& list, Node* node) noexcept
{
    DBG_ASSERT(node != nullptr);
    DBG_ASSERT(list != nullptr);
    SL_LIST_CHECK(sl_check(list));
    SL_LIST_CHECK(sl_contains(list, node));

    if (node == list)
    {
        list = node->next;
        if (list != nullptr)
        {
            list->prev = nullptr;
        }
    }
    else
    {
        DBG_ASSERT(node->prev != nullptr);
        if (node->next != nullptr)
        {
            node->next->prev = node->prev;
        }
        node->prev->next = node->next;
    }
    node->prev = nullptr;
    node->next = nullptr;
    SL_LIST_CHECK(sl_check(list));
}

/**
 * @brief Push @p node onto the front of the list headed by @p list.
 * @param list Head pointer, updated in place.
 * @param node Must not already be in a list -- sl_remove() first.
 * @details Front insertion is deliberate: the slab most recently touched is the
 *          one the next alloc() looks at.
 */
template <typename Node>
void sl_insert(Node*& list, Node* node) noexcept
{
    DBG_ASSERT(node != nullptr);
    SL_LIST_CHECK(sl_check(list));

    node->prev = nullptr;
    node->next = list;
    if (list != nullptr)
    {
        list->prev = node;
    }
    list = node;

    SL_LIST_CHECK(sl_check(list));
    SL_LIST_CHECK(sl_contains(list, node));
}


/**
 * @brief Re-file a slab after a free, and decide what becomes of it if it emptied.
 *
 * @param block Head of the "room left" list.
 * @param block_full Head of the "no room left" list.
 * @param block_empty The reserve of emptied slabs kept for reuse.
 * @param nb_block_empty How many the reserve currently holds; updated here.
 * @param nb_max_block_empty Reserve capacity. NEGATIVE means "never release a
 *        slab": the reserve is not used at all and @p on_kept is called instead.
 * @param slab The slab that was just freed into.
 * @param on_kept Run on a slab that emptied while @p nb_max_block_empty is
 *        negative, so the slab stays where it is. Value families
 *        MADV_DONTNEED it; op/ctx do nothing.
 * @param on_evict Run on a slab that emptied when the reserve is already full.
 *        Deletes it.
 *
 * @details Two things happen, in order: a slab that was full has room again and
 *          moves back to @p block, and a slab that reached is_empty() leaves the
 *          lists for the reserve, @p on_kept or @p on_evict.
 *
 * @note Shared by every slab family; the two callables are the only divergence.
 *
 * @warning A slab that empties is not thereby reclaimed. With
 *          @p nb_max_block_empty negative -- which is what the chunk layer
 *          passes, to keep a freed value slab's address from being reobtained
 *          by the op pool -- its pages stay resident unless @p on_kept releases
 *          them. See alloc_config::reclaim_empty_slabs.
 */
template <typename Node, class Count, class OnKept, class OnEvict>
void sl_manage_full_empty(Node*&  block,
                          Node*&  block_full,
                          Node**  block_empty,
                          Count&  nb_block_empty,
                          int     nb_max_block_empty,
                          Node*   slab,
                          OnKept  on_kept,
                          OnEvict on_evict) noexcept
{
    if (slab->was_full() && block_full) [[unlikely]]
    {
        sl_remove(block_full, slab);
        sl_insert(block, slab);
    }

    if (slab->is_empty()) [[unlikely]]
    {
        // nb_max_block_empty < 0  ==  never munmap (op/value slab aliasing).
        if (nb_max_block_empty < 0)
        {
            on_kept(slab); // value: MADV_DONTNEED the slab; op/ctx: no-op
            return;
        }
        // Plain comparison: every family declares nb_block_empty as int,
        // matching nb_max_block_empty, so that -1 works.
        if (nb_block_empty < nb_max_block_empty)
        {
            block_empty[nb_block_empty++] = slab;
            sl_remove(block, slab);
        }
        else
        {
            sl_remove(block, slab);
            on_evict(slab); // sync: delete the slab; async: defer to per-thread free list
        }
    }
}

/**
 * @brief Delete every slab of a pool: its two lists and its reserve.
 * @param block Head of the "room left" list.
 * @param block_full Head of the "no room left" list.
 * @param block_empty The reserve of emptied slabs.
 * @param nb_block_empty How many the reserve holds.
 * @warning Leaves the list heads dangling: the caller resets or discards them.
 */
template <typename Node>
void sl_delete_all(Node* block, Node* block_full, Node* const* block_empty, int nb_block_empty) noexcept
{
    for (Node* list : {block, block_full})
    {
        while (list != nullptr)
        {
            Node* next = list->next;
            delete list;
            list = next;
        }
    }

    for (int i = 0; i < nb_block_empty; ++i)
    {
        delete block_empty[i];
    }
}

/**
 * @brief Type-erased handle on a ::TCTypeAllocator, for the memory report.
 * @details Every instance registers itself so a total can be taken across all
 *          element types without the reporting code knowing any of them.
 */
struct VTCTypeAllocator
{
    /// @brief Nothing to set up: the derived allocator enrols itself.
    VTCTypeAllocator() = default;

    /// @brief Virtual, so the registry may hold any element type.
    virtual ~VTCTypeAllocator() = default;

    // Enrolled by address in the registry: neither copied nor moved.
    VTCTypeAllocator(const VTCTypeAllocator&)            = delete;
    VTCTypeAllocator& operator=(const VTCTypeAllocator&) = delete;
    VTCTypeAllocator(VTCTypeAllocator&&)                 = delete;
    VTCTypeAllocator& operator=(VTCTypeAllocator&&)      = delete;

    /**
     * @brief Accumulate this allocator's footprint into the running totals.
     * @param sum_UsedSize Elements ever carved, summed over slabs.
     * @param sum_nn_ref Elements currently out, summed over slabs.
     * @param alloc_sz Bytes held, summed over slabs.
     */
    virtual void addAllocatedSize(size_t& sum_UsedSize, size_t& sum_nn_ref, size_t& alloc_sz) const = 0;
};

/// @brief Enrol an allocator in the registry the memory report walks.
void add_TCTypeAllocator_inst(VTCTypeAllocator*);

/// @brief Withdraw it. Called from the destructor.
void remove_TCTypeAllocator_inst(VTCTypeAllocator*);

/**
 * @brief Owns every slab serving one element size: the machinery common to
 *        ::TCTypeAllocator and ::TCTypeCAllocator.
 *
 * @details Slabs are kept in three places, and which one a slab is in is a
 *          function of its `nn_ref`: #block has room, #block_full has none, and
 *          #block_empty is a small reserve of emptied slabs kept back so the
 *          next allocation does not have to go to the arena. alloc() and the
 *          frees move slabs between them; the moves themselves live in
 *          ::sl_manage_full_empty.
 *
 * @tparam Derived The pool (CRTP). It provides, as `static constexpr`:
 *         `eager_first_slab` (init() creates the first slab), `zero_std_alloc`
 *         (under STD_ALLOC, alloc() zero-fills what it hands out) and
 *         `pool_name` (for the [MUNMAP] trace).
 * @tparam ElemT Element type. What is handed out is an ARRAY of these --
 *         init() takes a count, and alloc() returns a pointer to the first.
 * @tparam Slab ::TCAllocator or ::TCCAllocator.
 *
 * @note One instance per (element size, thread) in the OpenMP value pools.
 *       Nothing here is synchronised: a thread only ever touches its own
 *       instance. Which is what makes freeing delicate: a block can be released
 *       by a thread other than the one that allocated it, and its owning slab
 *       then belongs to another instance. Route it with is_free_able_elt() /
 *       holds_live() before free_elt_pow2(); freeing through a non-owning
 *       instance corrupts the owner's free list.
 *
 * @see bypass_slabs for when none of this machinery runs at all.
 */
template <typename Derived, typename ElemT, typename Slab>
class TCSlabPool
{
  public:
    using ElementType = ElemT; ///< The element type.

    // Owns its slabs, which also name it as their owner (Slab::owner).
    TCSlabPool(const TCSlabPool&)            = delete;
    TCSlabPool& operator=(const TCSlabPool&) = delete;
    TCSlabPool(TCSlabPool&&)                 = delete;
    TCSlabPool& operator=(TCSlabPool&&)      = delete;

    /**
     * @brief Set the geometry. Call once, before any allocation.
     *
     * @param ElementSize_ Number of ElementType per allocation -- a COUNT, not
     *        a byte size. Stored as bytes unless bypassing; see ::bypass_slabs.
     * @param BlockAllocator_ Slab size in bytes, rounded up to a power of two.
     * @param nb_max_block_emptyl Capacity of the emptied-slab reserve.
     *        NEGATIVE means "never release a slab", which also means the reserve
     *        is not allocated at all.
     *
     * @note Whether the first slab is created here is the pool's call
     *       (`Derived::eager_first_slab`); alloc() makes one anyway when #block
     *       is null. ::TCTypeCAllocator creates none: priming every (thread,
     *       geometry) pair eagerly cost one slab per pair that never allocated,
     *       and the value arena never takes a slab back.
     */
    void init(size_t ElementSize_, uintptr_t BlockAllocator_, int nb_max_block_emptyl = 5)
    {
        nb_max_block_empty = nb_max_block_emptyl;

        if (bypass_slabs())
        {
            // new[] takes an element count, so that is what ElementSize holds.
            ElementSize = ElementSize_;
        }
        else
        {
            // A slab is carved in bytes.
            ElementSize = ElementSize_ * sizeof(ElementType);

            BlockAllocator = BlockAllocator_;
            slab_bytes_    = Slab::slab_bytes_for(BlockAllocator_);

            if constexpr (Derived::eager_first_slab)
            {
                // Stamped like every slab alloc_slow() creates: an unstamped
                // slab has no owner, so a free of one of its blocks falls back to
                // the CALLER's allocator -- in a dynamic build, whichever chunk
                // size the thread last served -- which unlinks the slab from
                // lists it was never on. With the owner set, the free goes home
                // whatever the ambient size (ChunkBase::release_to_owner).
                insert(block, new_slab());
            }

            // nb_max_block_empty < 0 means "never
            // munmap" -- nothing is ever stored into block_empty, so don't
            // allocate it (and avoid a vector of -1 elements).
            if (nb_max_block_empty > 0)
            {
                block_empty.assign(static_cast<std::size_t>(nb_max_block_empty), nullptr);
            }
        }

        nb_block_empty = 0;
    }

    /**
     * @brief Hand out room for #ElementSize elements.
     * @return Memory with unspecified contents; the caller initialises it (see
     *         TCSlab's @warning). Zero-filled under STD_ALLOC when the pool asks
     *         for it (`Derived::zero_std_alloc`).
     * @details Draws from the current slab; failing that, revives one from the
     *          reserve; failing that, creates a new slab. A slab that fills up
     *          moves to #block_full, which is why #block can be null again at
     *          any time and why this has to cope with that.
     */
    [[nodiscard]] ElementType* alloc()
    {
        // if constexpr: the memset must not even be instantiated in a slab
        // build (gcc's -Wclass-memaccess on Bit, std::complex).
        if constexpr (bypass_slabs())
        {
            auto* elements = new ElementType[ElementSize]; // ElementSize is a COUNT here
            if constexpr (Derived::zero_std_alloc)
            {
                memset(elements, 0, ElementSize * sizeof(ElementType));
            }
            return elements;
        }

        if (block == nullptr) [[unlikely]]
        {
            return alloc_slow();
        }

        auto* current_block = block;

        auto* ptr = static_cast<ElementType*>(current_block->alloc());

        if (current_block->is_full()) [[unlikely]]
        {
            remove(block, current_block);
            insert(block_full, current_block);
        }

        return ptr;
    }

    /**
     * @brief free_elt() for a power-of-two element size, using a shift.
     * @param ptr Must belong to THIS instance. This is the hot free path, and
     *        it does not check -- see is_free_able_elt().
     * @pre #ElementSize is a power of two when sizeof(ElementType) is, i.e.
     *      the element count given to init() is a power of two; otherwise the
     *      shift picks the wrong index and the free lands on another element
     *      (asserted in debug).
     */
    void free_elt_pow2(const ElementType* ptr) noexcept
    {
        if (bypass_slabs())
        {
            delete[] ptr;
            return;
        }

        free_elt_pow2(ptr, get_allocator(ptr));
    }

    /// @brief free_elt_pow2() for a caller that has already resolved @p slab
    ///        (slab_of()): the slab is not looked up again.
    /// @pre @p slab is @p ptr's slab and belongs to this instance; slab mode.
    void free_elt_pow2(const ElementType* ptr, Slab* slab) noexcept
    {
        // The element size is a power-of-two count of ElementType, so it is a
        // power of two exactly when sizeof(ElementType) is: the shift then,
        // the division otherwise (a 24-byte struct, say) -- folded at compile
        // time, the pow2 path keeps its single instruction.
        // Precondition, not checked in release: the COUNT given to init() is a
        // power of two (chunk sizes are). A TCTypeAllocator<std::byte> sized to
        // a 40-byte op would pick the shift (sizeof(std::byte) == 1) and free
        // the wrong element; the deferred-op pool is such an allocator and
        // never frees through here. The assertion catches a new caller.
        DBG_ASSERT(!std::has_single_bit(sizeof(ElementType)) ||
                   std::has_single_bit(static_cast<unsigned int>(slab->ElementSize)));
        const int i = std::has_single_bit(sizeof(ElementType)) ? slab->num_block_pow2(ptr) : slab->num_block(ptr);

        slab->free_block(i);

        manage_full_empty(slab);
    }

    /**
     * @brief Can THIS allocator free @p ptr?
     * @param ptr Pointer previously returned by some allocator's alloc().
     * @return True if @p ptr's slab is one of ours and holds a live block.
     * @details The routing primitive for cross-thread frees: resolves @p ptr to
     *          its slab through the back-pointer, then asks the slab whether we
     *          own it.
     * @note O(1): the slab records its owner (see TCSlab::owner). A slab parked
     *       in block_empty has nn_ref == 0, so a pointer into one is not a live
     *       element of this instance.
     */
    [[nodiscard]] bool is_free_able_elt(const ElementType* ptr) const noexcept
    {
        if (block == nullptr && block_full == nullptr)
        {
            return false;
        }
        return holds_live(get_allocator(ptr));
    }

    /// @brief is_free_able_elt() for a caller that has already resolved the
    ///        slab (slab_of()): no second resolution.
    [[nodiscard]] bool holds_live(const Slab* slab) const noexcept
    {
        return slab != nullptr && slab->owner == self() && slab->nn_ref > 0;
    }

    /// @brief The slab holding @p ptr, resolved from this instance's slab
    ///        geometry alone -- works even when this instance owns no slab yet.
    ///        Null before init().
    [[nodiscard]] Slab* slab_of(const void* ptr) const noexcept
    {
        // The slabs are cut at the ROUNDED geometry (Slab::slab_bytes_for: a
        // power of two, whatever the element size); #slab_bytes_ is that size,
        // fixed by init() through the same rule -- masking with the raw budget
        // would land beside the slab on any non-pow2 budget (e.g. the 240 KB
        // floor at 64 threads).
        if (slab_bytes_ == 0)
        {
            return nullptr;
        }
        return Slab::slab_from(ptr, slab_bytes_);
    }

    /**
     * @brief Choose which arena this pool's slabs are drawn from.
     * @param k Op, Ctx or Value.
     * @warning Must be set BEFORE the first slab is taken; it is copied into
     *          each slab at creation and never revisited. ::TCTypeAllocator
     *          defaults to Op, so a pool that holds values has to say so or its
     *          slabs will share an address space with the op pool -- which is
     *          the aliasing bug the arenas were split to prevent.
     */
    void set_arena(fast_alloc::ArenaKind k) noexcept
    {
        arena_kind = k;
    }

    /// @brief Rank (thread) this instance serves; stamped into every slab it
    ///        creates (TCSlab::owner_rank). -1 = unranked (serial pools).
    [[nodiscard]] int rank() const noexcept
    {
        return rank_;
    }

    /// @brief Set rank(); before the first slab is created.
    void set_rank(int rank) noexcept
    {
        rank_ = rank;
    }

    /// @brief Affinity key this instance serves (0 = shared pool of the rank);
    ///        stamped into its slabs as owner_key.
    [[nodiscard]] unsigned key() const noexcept
    {
        return key_;
    }

    /// @brief Set key(); before the first slab is created.
    void set_key(unsigned k) noexcept
    {
        key_ = k;
    }

    /**
     * @brief Release every slab and the reserve array, and reset to fresh state.
     * @note Unlike the destructor, leaves the object usable: init() may follow.
     */
    void destroy()
    {
        destroy_int();

        decltype(block_empty)().swap(block_empty); // releases the reserve's storage

        ElementSize = 0;
        block_full = block = nullptr;
        nb_block_empty     = 0;
    }

    Slab* block_full{nullptr}; ///< Slabs with no room left.
    Slab* block{nullptr};      ///< Slabs with room. Null until the first slab.

  protected:
    /// @brief Empty and unusable until init(); its slabs will draw from @p arena.
    explicit TCSlabPool(fast_alloc::ArenaKind arena) noexcept : arena_kind(arena) {}

    /// @brief Frees every slab.
    ~TCSlabPool()
    {
        destroy_int();
    }

  private:
    /// @brief The pool as its slabs name it (TCSlab::owner): the DERIVED
    ///        object. ChunkBase::release_to_owner casts the owner back to the
    ///        pool type, and a base subobject need not share its address --
    ///        ::TCTypeAllocator's sits after the ::VTCTypeAllocator vtable.
    [[nodiscard]] const void* self() const noexcept
    {
        return static_cast<const Derived*>(this);
    }

    /**
     * @brief Delete every slab, leaving the list heads dangling.
     * @warning Internal: it does not reset #block / #block_full. Call destroy()
     *          unless you are about to overwrite them yourself.
     */
    void destroy_int()
    {
        // Nothing to walk when the slabs were never built: under STD_ALLOC there
        // are no slabs at all, each element was an independent new[] that
        // free_elt_pow2() has already deleted.
        if constexpr (alloc_config::use_slabs)
        {
            if (bypass_slabs())
            {
                return;
            }
            sl_delete_all(block, block_full, block_empty.data(), nb_block_empty);
        }
    }

    /// @brief A new slab owned by this instance, from its arena, at its geometry.
    Slab* new_slab()
    {
        auto* slab       = new Slab;
        slab->arena_kind = arena_kind; // route slab to its op/ctx/value arena
        slab->owner      = self();
        slab->owner_rank = rank_;
        slab->owner_key  = key_;
        slab->init(static_cast<int>(ElementSize), BlockAllocator);
        DBG_ASSERT(slab->slab_bytes() == slab_bytes_);
        return slab;
    }

    /// @brief alloc() with no slab at hand: revive one from the reserve or
    ///        create one. Out of line and cold -- see FAST_ALLOC_COLD.
    FAST_ALLOC_COLD ElementType* alloc_slow()
    {
        if (nb_block_empty > 0)
        {
            auto* slab = block_empty[--nb_block_empty];

            insert(block, slab);

            slab->reset();

            return static_cast<ElementType*>(slab->alloc());
        }

        auto* slab = new_slab();

        insert(block, slab);

        return static_cast<ElementType*>(slab->alloc());
    }

    /// @brief Re-file @p slab after a free. @see sl_manage_full_empty
    void manage_full_empty(Slab* slab) noexcept
    {
        sl_manage_full_empty(
            block, block_full, block_empty.data(), nb_block_empty, nb_max_block_empty, slab,
            [](Slab* kept) noexcept
            {
                // NO madvise here, whatever reclaim_empty_slabs says.
                //
                // This lambda runs on the hot path, on every free that brings a
                // slab to nn_ref == 0, and nn_ref is its ONLY evidence that the
                // slab holds nothing live. That evidence is exactly what a
                // mis-accounted block (the same buffer parked or released twice,
                // or two threads touching one slab's counter) falsifies -- and
                // MADV_DONTNEED turns the falsehood into destroyed data: the
                // live block's page is dropped and reads back as zeros. Left
                // mapped, the same accounting error is harmless.
                //
                // Pages are given back only by
                // TCTypeCAllocator::reclaim_empty_slabs_now(), which is called
                // on a set the caller KNOWS abandoned by construction (the
                // (key, size) set a re-chunked array just left, an array's
                // teardown) -- a slab with nn_ref == 0 there is empty because no
                // block of that set can be live any more, not because a counter
                // says so. The emptied-slab case is unchanged otherwise: the
                // slab stays in `block`, still allocatable.
                (void)kept;
            },
            [this](Slab* evicted) noexcept { evict_slab(evicted); });
    }

    /// @brief Release an emptied slab the reserve has no room for. Out of
    ///        line and cold: a `delete` (and, with FAST_ALLOC_TRACE, a trace)
    ///        on every free path.
    FAST_ALLOC_COLD void evict_slab(Slab* slab) noexcept
    {
        if (fast_alloc::trace_evictions())
        {
            std::fprintf(stderr, "[MUNMAP] %s delete slab p=%p nb_max_block_empty=%d\n", Derived::pool_name,
                         static_cast<void*>(slab), nb_max_block_empty);
            std::fflush(stderr);
        }
        delete slab;
    }

    /**
     * @brief The slab owning @p ptr.
     * @pre init() has run -- #slab_bytes_ is the instance's slab geometry, so
     *      no slab of ours is needed to resolve @p ptr.
     */
    [[nodiscard]] Slab* get_allocator(const ElementType* ptr) const noexcept
    {
        return Slab::slab_from(ptr, slab_bytes_);
    }

    /// @copydoc sl_remove
    void remove(Slab*& list, Slab* node) const noexcept
    {
        sl_remove(list, node);
    }

    /// @copydoc sl_insert
    void insert(Slab*& list, Slab* node) const noexcept
    {
        sl_insert(list, node);
    }

    // Which arena the slabs created by this pool draw from, propagated to
    // every slab it news up. See set_arena().
    fast_alloc::ArenaKind arena_kind;

    size_t    ElementSize    = 0;  ///< Bytes per allocation; a COUNT when bypassing.
    uintptr_t BlockAllocator = 0;  ///< Slab size in bytes, passed on to each slab.
    uintptr_t slab_bytes_    = 0;  ///< The ROUNDED slab size every slab of ours is cut at; 0 until init().
    int       rank_          = -1; ///< Rank served; stamped into slabs as owner_rank.
    unsigned  key_           = 0;  ///< Affinity key served; stamped into slabs as owner_key.

    /// @brief Reserve capacity; negative = never release a slab. @see init
    int nb_max_block_empty = 0;

    std::vector<Slab*> block_empty;       ///< The reserve. Empty when the capacity is <= 0.
    int                nb_block_empty{0}; ///< How much of the reserve is occupied.
};

/**
 * @brief Owns every ::TCAllocator slab serving one element size.
 *
 * @details The serial family, and the op and ctx pools of every build:
 *          ::TCSlabPool over slabs whose free list runs through the elements.
 *          Its own parts: the first slab is created by init(), the slabs draw
 *          from the op arena unless set_arena() says otherwise, and every
 *          instance enrols itself in the ::VTCTypeAllocator registry, so the
 *          memory report can total all of them without knowing the type.
 *
 * @tparam ElemT Element type; an allocation is an ARRAY of these.
 *
 * @note ::TCTypeCAllocator is its OpenMP value counterpart (links outside the
 *       payload); ::TCTypeAllocatorEBC is the compile-time geometry variant.
 */
template <typename ElemT>
class TCTypeAllocator final : public VTCTypeAllocator, public TCSlabPool<TCTypeAllocator<ElemT>, ElemT, TCAllocator>
{
  public:
    /// @brief init() creates the first slab. It costs its address space and the
    ///        page of its back-pointer, not the whole slab: alloc_block() does
    ///        not clear it.
    static constexpr bool eager_first_slab = true;
    /// @brief STD_ALLOC hands out new[] as is.
    static constexpr bool zero_std_alloc = false;
    /// @brief For the [MUNMAP] trace.
    static constexpr const char* pool_name = "TCTypeAllocator";

    /// @brief Empty and unusable until init(); enrols in the report registry.
    TCTypeAllocator() : TCSlabPool<TCTypeAllocator<ElemT>, ElemT, TCAllocator>(fast_alloc::ArenaKind::Op)
    {
        add_TCTypeAllocator_inst(this);
    }

    /// @brief Unregisters from the instance list; the base then frees every slab.
    ~TCTypeAllocator() override
    {
        remove_TCTypeAllocator_inst(this);
    }

    // Owns its slabs and is enrolled by address in the report registry.
    TCTypeAllocator(const TCTypeAllocator&)            = delete;
    TCTypeAllocator& operator=(const TCTypeAllocator&) = delete;
    TCTypeAllocator(TCTypeAllocator&&)                 = delete;
    TCTypeAllocator& operator=(TCTypeAllocator&&)      = delete;

  private:
    /**
     * @copydoc VTCTypeAllocator::addAllocatedSize
     * @note @p alloc_sz counts ELEMENTS, not bytes -- it sums BlockSize over the
     *       slabs. The reserve in #block_empty is not counted, so a run that
     *       parks slabs there reports less than it holds.
     */
    void addAllocatedSize(size_t& sum_UsedSize, size_t& sum_nn_ref, size_t& alloc_sz) const noexcept override
    {
        size_t s_blocks   = 0;
        size_t s_UsedSize = 0;
        size_t s_nn_ref   = 0;

        for (const TCAllocator* node = this->block; node != nullptr; node = node->next)
        {
            s_UsedSize += static_cast<size_t>(node->UsedSize) * node->ElementSize;
            s_nn_ref += static_cast<size_t>(node->nn_ref) * node->ElementSize;
            s_blocks += node->BlockSize;
        }

        for (const TCAllocator* node = this->block_full; node != nullptr; node = node->next)
        {
            s_blocks += node->BlockSize;
        }

        sum_UsedSize += s_UsedSize;
        sum_nn_ref += s_nn_ref;
        alloc_sz += s_blocks;
    }
};

// Declaration-level switch, so it stays a directive: this type exists only in
// OpenMP builds. Serial builds use ::TCTypeAllocator throughout.
#ifdef _OPENMP

/**
 * @brief Owns every ::TCCAllocator slab serving one element size.
 *
 * @details The OpenMP value pools: ::TCSlabPool over slabs that keep their free
 *          list outside the payload, so an emptied slab can be madvised away
 *          (reclaim_empty_slabs_now(), its own part). Its slabs draw from the
 *          value arena; init() creates none; it does not enrol in the
 *          ::VTCTypeAllocator report registry; under STD_ALLOC it zero-fills
 *          what it hands out.
 *
 *          alignas(64): it lives in per-rank arrays (one per thread); at 96
 *          bytes one rank's slab list heads (block_full, block) would share a
 *          cache line with the previous rank's reserve (block_empty,
 *          nb_block_empty). Each starts its own line (128 bytes a rank).
 *
 * @tparam ElemT Element type; an allocation is an ARRAY of these.
 */
template <typename ElemT>
class alignas(64) TCTypeCAllocator : public TCSlabPool<TCTypeCAllocator<ElemT>, ElemT, TCCAllocator>
{
  public:
    /// @brief No slab at init(): the first alloc() makes one. See TCSlabPool::init.
    static constexpr bool eager_first_slab = false;
    /// @brief STD_ALLOC zero-fills, as the value pools always did.
    static constexpr bool zero_std_alloc = true;
    /// @brief For the [MUNMAP] trace.
    static constexpr const char* pool_name = "TCCAllocator";
    /// @brief The batch that reclaim_empty_slabs_now() and the re-chunk sweeps
    ///        feed: one madvise per run of adjacent empty slabs.
    using empty_slab_sweep = fast_alloc::EmptySlabSweep<TCCAllocator>;

    /// @brief Empty and unusable until init(); its slabs draw from the value arena.
    TCTypeCAllocator() : TCSlabPool<TCTypeCAllocator<ElemT>, ElemT, TCCAllocator>(fast_alloc::ArenaKind::Value) {}

    /// @brief Give back the pages of every slab of this instance that currently
    ///        holds no block, and report how many were advised.
    /// @details manage_full_empty() only looks at a slab at the instant one of
    ///          its blocks is freed, so a slab that ends up empty without that
    ///          transition being seen stays resident for good (after a re-chunk,
    ///          a whole abandoned (key, size) set of empty slabs, never advised).
    ///          This is the sweep that closes that gap; call it at the points
    ///          where a whole set is known to have been released (the re-chunk
    ///          drain, an array's teardown), not on the hot path.
    std::size_t reclaim_empty_slabs_now() noexcept
    {
        empty_slab_sweep sweep;
        collect_empty_slabs(sweep);
        return sweep.flush();
    }


    /// @brief Hand every slab of this instance that currently holds no block to
    ///        @p sweep, which advises them away at its flush().
    /// @details Lets a caller gather several instances (every rank, several
    ///          keys) into ONE sweep, so runs of adjacent slabs merge across
    ///          instances. Same preconditions as reclaim_empty_slabs_now(),
    ///          held until the sweep is flushed.
    /// @return Number of slabs handed over. The CHUNK_RECLAIM_MIN_SLABS floor
    ///         is applied by the sweep, to all instances together.
    std::size_t collect_empty_slabs([[maybe_unused]] empty_slab_sweep& sweep) noexcept
    {
        if constexpr (alloc_config::use_std_alloc || !alloc_config::reclaim_empty_slabs)
        {
            return 0;
        }
        else
        {
            // No CHUNK_RECLAIM_MIN_SLABS test here: the floor is the sweep's
            // (EmptySlabSweep::flush), so that one rank's share of a set
            // gathered across every rank is not dropped on its own.
            std::size_t collected = 0;
            for (auto* node : {this->block, this->block_full})
            {
                for (auto* p = node; p != nullptr; p = p->next)
                {
                    // Whole slab, trailing page included: this sweep is the one
                    // context that can afford it (see fast_alloc::EmptySlabSweep).
                    // The last page is 6% of a 1 MiB slab and was the entire
                    // unexplained residue of a re-chunk.
                    if (p->nn_ref == 0)
                    {
                        sweep.add(p);
                        ++collected;
                    }
                }
            }
            return collected;
        }
    }
};

#endif


/// @brief Narrowest signed integer that fits in @p bytes bytes (1, 2, else int):
///        a slab's free-list index sized to the element, rather than an `int`
///        per element on a small block.
template <unsigned int bytes>
using free_index_t = std::conditional_t<bytes == 1, signed char, std::conditional_t<bytes == 2, short, int>>;


/// @brief Maps an element type to the allocator that serves it. Specialised elsewhere.
template <typename V>
struct TypeAllocator;

/**
 * @brief One slab whose element and block sizes are template parameters.
 *
 * @details Same job as ::TCAllocator, with the geometry known at compile time:
 *          the capacity, the shift used for indexing and the free-list index
 *          width all become constants, so the division in num_block() and the
 *          per-slab geometry fields go away.
 *
 * @tparam ElementBytes Element size in bytes.
 * @tparam BlockBytes Block size in bytes.
 *
 * @note Used only through ::TCTypeAllocatorEBC, which owns the slab lists.
 */
template <unsigned int ElementBytes, unsigned int BlockBytes>
class TCAllocatorEB
{
  public:
    /// @brief The compile-time geometry: sizes in bytes, capacity in elements.
    enum
    {
        ElementSize     = ElementBytes,
        Log2ElementSize = static_cast<int>(std::bit_width(ElementBytes)) - 1,
        BlockSize       = BlockBytes,
        BlockSizePOT    = static_cast<int>(std::bit_ceil(BlockBytes)), // int: the mask below must sign-extend
        BlockAddrMask   = ~(BlockSizePOT - 1),
        getSizeOf       = sizeof(TCAllocatorEB*),
        UsedBlockSize   = BlockBytes - getSizeOf,
        MaxElements     = UsedBlockSize / ElementBytes
    };

    /// @brief Free-list index type, as narrow as the element allows.
    using T = free_index_t<std::bit_floor(ElementBytes)>;

    // The free list stores an element INDEX of type T inside each freed
    // element: T must fit in an element, and must count every element of the
    // block (a signed char stops at 127, a short at 32767).
    static_assert(sizeof(T) <= static_cast<std::size_t>(ElementSize),
                  "TCAllocatorEB: the free-list index does not fit in an element");
    static_assert(static_cast<long long>(MaxElements) <= static_cast<long long>(std::numeric_limits<T>::max()),
                  "TCAllocatorEB: more elements per block than the free-list index type can count");

    /// @brief Takes its block straight away -- there is no separate init().
    TCAllocatorEB()
    {
        // alloc() inserts the slab right after this constructor returns: a
        // slab without a block (or without its back-pointer) would be handed
        // out and corrupt memory silently. Abort, as the other families do.
        if (alloc_block() != 0)
        {
            std::abort();
        }
    }

    /// @brief Returns the block to whichever allocator alloc_block() used.
    ~TCAllocatorEB()
    {
        if (use_mmap)
        {
            munmap(data, BlockSize);
        }
        else
        {
            free(static_cast<void*>(data));
        }
    }

    // Owns its block: a copy would unmap or free it twice.
    TCAllocatorEB(const TCAllocatorEB&)            = delete;
    TCAllocatorEB& operator=(const TCAllocatorEB&) = delete;
    TCAllocatorEB(TCAllocatorEB&&)                 = delete;
    TCAllocatorEB& operator=(TCAllocatorEB&&)      = delete;

    /**
     * @brief Hand out one element.
     * @return Raw, uninitialised memory.
     * @pre The slab is not full -- the caller checks is_full().
     * @details Free list threaded through the freed elements, as in
     *          ::TCAllocator, but with the index narrowed to #T so a small block
     *          does not spend four bytes per element on it.
     */
    void* alloc() noexcept
    {
        ++nn_ref;
        char* elemPtr = nullptr;

        if (firstFree != -1)
        {
            elemPtr   = data + (firstFree * ElementSize);
            firstFree = *(T*)elemPtr;
        }
        else
        {
            elemPtr = data + (UsedSize * ElementSize);
            ++UsedSize;
        }
        return static_cast<void*>(elemPtr);
    }

    /// @brief Index of the element at @p ptr within this slab (division).
    int num_block(const void* ptr) const noexcept
    {
        return static_cast<int>((static_cast<const char*>(ptr) - data) / ElementSize);
    }

    /// @brief num_block() by shift. Valid only if #ElementSize is a power of two.
    int num_block_pow2(const void* ptr) const noexcept
    {
        return static_cast<int>((static_cast<const char*>(ptr) - data) >> Log2ElementSize);
    }

    /**
     * @brief Take element @p i back.
     * @param i Index from num_block() / num_block_pow2().
     * @details Freeing the element the bump pointer just handed out rewinds the
     *          bump instead of pushing onto the free list.
     */
    void free_block(int i) noexcept
    {
        if (i == (UsedSize - 1))
        {
            *(T*)(data + (i * ElementSize)) = -1;
            --UsedSize;
        }
        else
        {
            *(T*)(data + (i * ElementSize)) = firstFree;
            firstFree                       = i;
        }
        --nn_ref;
    }

    /// @brief No element of this slab is out. Only then may it be deleted.
    [[nodiscard]] bool is_empty() const noexcept
    {
        return nn_ref == 0;
    }

    /// @brief This free brought the slab back from full.
    [[nodiscard]] bool was_full() const noexcept
    {
        return nn_ref == MaxElements - 1;
    }

    /// @brief Every element is out; the slab must move to block_full.
    [[nodiscard]] bool is_full() const noexcept
    {
        return nn_ref == MaxElements;
    }

    /**
     * @brief Recover the owning slab from a pointer to any element inside it.
     * @param ptr Interior pointer, previously returned by alloc().
     * @note Static, unlike the other families' version: the mask and the offset
     *       are both compile-time constants here, so no instance is needed.
     */
    static TCAllocatorEB* addr_to_block(const void* ptr) noexcept
    {
        // this pointer stored at the end of the block
        return *reinterpret_cast<TCAllocatorEB**>(
            reinterpret_cast<char*>(reinterpret_cast<uintptr_t>(ptr) & size_t(BlockAddrMask)) + UsedBlockSize);
    }

    char* data = nullptr; ///< The block itself, aligned on #BlockSizePOT.
    int   firstFree{-1};  ///< Head of the free list, an element index; -1 if empty.

    int nn_ref{0};   ///< Elements currently out. 0 == deletable.
    int UsedSize{0}; ///< Bump index: how far the linear carve has reached.

    // for the linked list
    TCAllocatorEB* prev = nullptr; ///< Previous slab in the owner's block/block_full list.
    TCAllocatorEB* next = nullptr; ///< Next slab in the owner's block/block_full list.

    // A free element holds the index of the next free one, in T -- a type
    // sized from the element. The block must be addressable by it: a
    // compile-time check (it was a runtime DBG_ASSERT, DEBUG builds only).
    static_assert(static_cast<uintptr_t>(BlockBytes) < (uintptr_t{1} << ((sizeof(T) * 8) - 1)),
                  "TCAllocatorEB: block too large for the element's free-list index type");

  private:
    /**
     * @brief Obtain the block and lay it out.
     * @return 0, or 1 when no block could be obtained (posix_memalign failed);
     *         the constructor aborts then.
     * @details The block must be aligned on its own power-of-two size, because
     *          addr_to_block() masks with #BlockAddrMask. The mmap path maps
     *          twice the size and unmaps the unaligned head and the tail; the
     *          posix_memalign path asks for the alignment explicitly.
     *          A failed trimming munmap (ENOMEM once vm.max_map_count is
     *          reached) is not a failure: the aligned block is mapped and
     *          usable, only the surplus stays mapped. It is reported and the
     *          block is laid out as usual.
     */
    int alloc_block()
    {
        const long ps = fast_alloc::page_size();
        if (ps > 0 && (2 * BlockSize) % ps == 0)
        {
            use_mmap = true;

            const uintptr_t block_bytes = BlockSize;

            char* mapping = static_cast<char*>(fast_alloc::map_anonymous(2 * BlockSize));

            data = reinterpret_cast<char*>((reinterpret_cast<uintptr_t>(mapping) + (block_bytes - 1)) & -block_bytes);

            const uintptr_t lng = reinterpret_cast<uintptr_t>(data) - reinterpret_cast<uintptr_t>(mapping);
            if (lng != 0 && munmap(mapping, lng) == -1)
            {
                const int err = errno;
                std::cerr << "munmap (head trim) failed: " << std::strerror(err) << " -- surplus left mapped\n";
            }
            if (munmap(data + BlockSize, (2 * BlockSize) - (lng + BlockSize)) == -1)
            {
                const int err = errno;
                std::cerr << "munmap (tail trim) failed: " << std::strerror(err) << " -- surplus left mapped\n";
            }
        }
        else
        {
            use_mmap    = false;
            void* datal = nullptr;

            const int result = posix_memalign(&datal, BlockSizePOT, BlockSize);

            if (result != 0)
            {
                std::cerr << "posix_memalign failed with error code: " << result << '\n';
                std::cerr << "Error message: " << std::strerror(result) << '\n';
                return 1;
            }
            data = static_cast<char*>(datal);
        }

        *reinterpret_cast<TCAllocatorEB**>(data + UsedBlockSize) = this;

        firstFree = -1;

        return 0;
    }

    bool use_mmap = false; ///< Which of munmap / free the destructor must use.
};

/**
 * @brief Owns every ::TCAllocatorEB slab for one compile-time geometry.
 *
 * @details Third slab family, alongside ::TCTypeAllocator and
 *          ::TCTypeCAllocator, and the same three-way split of slabs between
 *          #block, #block_full and #block_empty. What differs is that the
 *          geometry is a template parameter, so the indexing arithmetic folds
 *          to shifts and the allocator carries no size fields.
 *
 * @tparam ElementSize Element size in bytes.
 * @tparam BlockSize Block size in bytes.
 * @tparam T Unused; kept so the template signature stays stable.
 *
 * @note This is what StaticConstArray's size ladder is built from -- one
 *       instance per small block size (4, 8, 24, ... bytes).
 *
 * @note Like ::TCTypeCAllocator, the constructor takes no slab: the first
 *       alloc() makes one. Seeding all of StaticConstArray's buckets eagerly
 *       reserved ~104 MiB of address space before any of them was needed.
 */
template <unsigned int ElementSize, unsigned int BlockSize, typename T = int>
class TCTypeAllocatorEBC
{
  public:
    /// @brief The slab type this instance owns.
    using TCAllocatorEBT = TCAllocatorEB<ElementSize, BlockSize>;

    /// @brief No slab yet; the first alloc() creates one. No init() to call.
    TCTypeAllocatorEBC() : block_full(nullptr), block(nullptr) {}

    /// @brief Deletes every slab, the reserve included.
    ~TCTypeAllocatorEBC()
    {
        sl_delete_all(block, block_full, block_empty.data(), nb_block_empty);
    }

    // Owns its slabs: a copy would delete them twice.
    TCTypeAllocatorEBC(const TCTypeAllocatorEBC&)            = delete;
    TCTypeAllocatorEBC& operator=(const TCTypeAllocatorEBC&) = delete;
    TCTypeAllocatorEBC(TCTypeAllocatorEBC&&)                 = delete;
    TCTypeAllocatorEBC& operator=(TCTypeAllocatorEBC&&)      = delete;

    /**
     * @brief Hand out one element.
     * @return Raw, uninitialised memory.
     * @details Draws from the current slab; failing that, revives one from the
     *          reserve; failing that, creates a new slab.
     */
    void* alloc()
    {
        if (!block)
        {
            if (nb_block_empty > 0)
            {
                auto* slab = block_empty[--nb_block_empty];

                insert(block, slab);

                slab->UsedSize  = 0;
                slab->firstFree = -1;

                return slab->alloc();
            }

            auto* slab = new TCAllocatorEBT;

            insert(block, slab);

            return slab->alloc();
        }

        auto* current_block = block;

        void* ptr = current_block->alloc();
        if (current_block->is_full())
        {
            remove(block, current_block);
            insert(block_full, current_block);
        }

        return ptr;
    }

    /**
     * @brief Take back a pointer from alloc().
     * @param ptr Resolved to its slab through the back-pointer, so it need not
     *        have come from this instance.
     */
    void free_elt(const void* ptr)
    {
        auto* slab = TCAllocatorEBT::addr_to_block(ptr);

        const int i = slab->num_block(ptr);

        slab->free_block(i);

        manage_full_empty(slab);
    }

    /**
     * @brief free_elt() using a shift instead of a division.
     * @pre The element size is a power of two.
     */
    void free_elt_pow2(const void* ptr) noexcept
    {
        auto* slab = TCAllocatorEBT::addr_to_block(ptr);

        const int i = slab->num_block_pow2(ptr);

        slab->free_block(i);

        manage_full_empty(slab);
    }

    /**
     * @brief Accumulate this allocator's footprint into the running totals.
     * @param sum_UsedSize Bytes ever carved, summed over slabs.
     * @param sum_nn_ref Bytes currently out, summed over slabs.
     * @param alloc_sz Bytes held, summed over slabs.
     * @note Unlike TCTypeAllocator::addAllocatedSize, these are BYTES: the
     *       compile-time ElementSize and BlockSize are folded in here.
     */
    void addAllocatedSize(size_t& sum_UsedSize, size_t& sum_nn_ref, size_t& alloc_sz) const noexcept
    {
        size_t nb_blocks  = 0;
        size_t s_UsedSize = 0;
        size_t s_nn_ref   = 0;

        // const: this walk only totals, it never touches a slab.
        const TCAllocatorEBT* node = block;

        while (node != nullptr)
        {
            s_UsedSize += node->UsedSize;
            s_nn_ref += node->nn_ref;
            nb_blocks++;

            const TCAllocatorEBT* next = node->next;
            node                       = next;
        }

        node = block_full;

        while (node != nullptr)
        {
            nb_blocks++;

            const TCAllocatorEBT* next = node->next;
            node                       = next;
        }

        sum_UsedSize += s_UsedSize * size_t(ElementSize);
        sum_nn_ref += s_nn_ref * size_t(ElementSize);
        alloc_sz += nb_blocks * size_t(BlockSize);
    }

  private:
    /// @brief Reserve capacity. Fixed, unlike the other families' runtime value.
    static constexpr int nb_max_block_empty = 5;

    /**
     * @brief Re-file @p slab after a free, EBC's own way.
     *
     * @details Deliberately NOT ::sl_manage_full_empty. Three differences, kept
     *          so the concurrent cst_val2 path behaves exactly as it did:
     *          a slab that comes back from full returns immediately instead of
     *          also being tested for emptiness; there is no never-release branch
     *          and no MADV_DONTNEED, the reserve being a fixed static array; and
     *          a slab past the reserve is always deleted outright.
     */
    void manage_full_empty(TCAllocatorEBT* slab) noexcept
    {
        if (slab->was_full())
        {
            if (block_full)
            {
                remove(block_full, slab);
                insert(block, slab);
            }
            return;
        }

        if (slab->is_empty())
        {
            if (nb_block_empty < nb_max_block_empty)
            {
                block_empty[nb_block_empty++] = slab;
                remove(block, slab);
            }
            else
            {
                remove(block, slab);
                delete slab;
            }
        }
    }

    /// @copydoc sl_remove
    void remove(TCAllocatorEBT*& list, TCAllocatorEBT* node) const noexcept
    {
        sl_remove(list, node);
    }

    /// @copydoc sl_insert
    void insert(TCAllocatorEBT*& list, TCAllocatorEBT* node) const noexcept
    {
        sl_insert(list, node);
    }

    TCAllocatorEBT* block_full; ///< Slabs with no room left.
    TCAllocatorEBT* block; ///< Slabs with room. Null until the first alloc(), and again whenever the last one fills.

    int                                             nb_block_empty{0}; ///< Reserve occupancy.
    std::array<TCAllocatorEBT*, nb_max_block_empty> block_empty{};     ///< The reserve, inline.
};

/// @brief Default slab size (1 MiB).
/// @details constexpr rather than #define so it carries a type and scope; used
///          as a non-type template argument below and in StaticConstArray,
///          where an unsigned int constant expression is required.
inline constexpr unsigned int SIZE_BLOCK = 1u << 20u; // 1048576

#include "fast_allocators/internal/tc_allocator_inline.h" // the template bodies
