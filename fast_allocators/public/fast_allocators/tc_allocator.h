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

/**
 * @file tc_allocator.h
 * @brief Slab allocators: one mmap'd block is carved into fixed-size elements.
 *
 * Four layers, bottom to top:
 *
 *  - fast_alloc::TCSlab is ONE slab. It owns a block of memory, hands out fixed-size
 *    elements from it, and keeps a free list of the elements handed back. It
 *    is also the node of an intrusive list. fast_alloc::TCAllocator and fast_alloc::TCCAllocator
 *    are its two kinds (see below).
 *  - fast_alloc::sl_insert, fast_alloc::sl_remove, fast_alloc::sl_manage_full_empty and fast_alloc::sl_delete_all are
 *    the list primitives, shared by every family so a fix lands in one place.
 *  - fast_alloc::TCSlabPool owns a SET of slabs for one element size, split three ways:
 *    `block` (room left), `block_full`, and `block_empty` (a small reserve of
 *    emptied slabs kept for reuse). fast_alloc::TCTypeAllocator and fast_alloc::TCTypeCAllocator
 *    are its two pools, one per slab kind, with what is theirs alone.
 *  - fast_alloc::TCTypeAllocatorEBC is a third family whose element and block sizes are
 *    template parameters rather than run-time values.
 *
 * @par Why two slab kinds
 * fast_alloc::TCAllocator threads its free list THROUGH the freed elements, writing the
 * next free index into the element's first bytes. fast_alloc::TCCAllocator does not: it
 * keeps the links in a side array (TCSlab::tab_first_free_) and never writes
 * into a free element, so an emptied slab's pages can be dropped with madvise()
 * without losing anything the allocator needs. That side array is indexed by
 * `unsigned short`, which is what caps a TCCAllocator slab at
 * TCSlab::free_list_end - 1 elements. fast_alloc::FreeLinks names the choice.
 *
 * @par Configuration
 * The options are read in alloc_config.h only (FAST_ALLOC_DBG_ASSERT aside), so
 * the bodies here branch with `if constexpr`: both arms must compile in every
 * configuration. See fast_alloc::bypass_slabs.
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
#include <cstring>     // std::strerror
#include <cstdint>     // uintptr_t
#include <limits>      // std::numeric_limits: TCAllocatorEB's index range
#include <type_traits> // std::conditional_t (FreeIndexT)
#include <cassert>     // assert (FAST_ALLOC_DBG_ASSERT)
#include <sys/mman.h>  // mmap, munmap
#include <unistd.h>    // sysconf (fast_alloc::page_size)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "fast_allocators/internal/alloc_config.h"    // alloc_config::use_std_alloc / is_debug / threaded
#include "fast_allocators/slab_arena.h"               // ArenaKind for op/ctx/value slab routing
#include "fast_allocators/internal/fatal.h"           // FAST_ALLOC_FATAL
#include "fast_allocators/internal/log_core.h"        // CHUNK_LOG, logging::enabled (the eviction trace)
#include "fast_allocators/internal/next_power_of_2.h" // next_power_of_2, for slab_of()'s effective slab geometry

namespace fast_alloc
{

/**
 * @brief Must this call hand out plain `new[]` memory instead of slab memory?
 *
 * @details Under FAST_ALLOC_STD_ALLOC, for the whole run. Must stay a constant:
 *          a run-time switch flipped after init() would make element_size_ wrong
 *          for the other mode (see below).
 *
 * @warning The two arms disagree on what element_size_ means: bypass mode counts
 *          ELEMENTS (it feeds `new ElementType[element_size_]`), slab mode counts
 *          BYTES. init() sets it accordingly.
 */
[[nodiscard]] constexpr bool bypass_slabs() noexcept
{
    return alloc_config::use_std_alloc;
}


/// @brief The system page size, read once (sysconf).
[[nodiscard]] inline long page_size() noexcept
{
    static const long ps = ::sysconf(_SC_PAGESIZE);
    return ps;
}

/// @brief CHUNK_LOG=alloc=trace (formerly FAST_ALLOC_TRACE): trace the
///        evictions of emptied slabs.
[[nodiscard]] inline bool trace_evictions() noexcept
{
    return logging::enabled(logging::Channel::alloc, logging::Level::trace);
}

/**
 * @brief Map @p size anonymous, zero-filled bytes, or abort.
 * @param size Byte count; the caller is expected to have rounded it to pages.
 * @return The mapping. Never null -- a failure prints errno and aborts, because
 *         every caller here is on a path with no way to report failure upward.
 */
inline void* map_anonymous(size_t size)
{
    // MAP_ANONYMOUS: not backed by any file; the contents are zero (mmap(2)).
    // MAP_ANON, its older BSD spelling, where it is the only one.
#ifdef MAP_ANONYMOUS
    void* const tmp = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
#else
    void* const tmp = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
#endif

    if (tmp == MAP_FAILED)
    {
        const int err = errno;
        FAST_ALLOC_FATAL("mmap of {} bytes failed: errno = {} ({})", size, err, std::strerror(err));
    }

    alloc_config::advise_thp_region(tmp, size); // CHUNK_THP, as the slab arenas

    return tmp;
}


/// @brief assert() in a debug build (alloc_config::is_debug); nothing otherwise.
#if FAST_ALLOC_OPT_DEBUG
    #define FAST_ALLOC_DBG_ASSERT(x) assert(x)
#else
    #define FAST_ALLOC_DBG_ASSERT(x)
#endif

/**
 * @brief Marks the slow path of a hot function: kept out of line and placed
 *        with the cold code, so the hot path stays small enough to inline and
 *        its instruction cache holds nothing it does not need.
 * @details Slab creation (a `new`, an arena lock) and slab eviction (an
 *          fprintf, a `delete`) are rare, but inlined they would be the bulk of
 *          alloc() and free_elt_pow2().
 */
#if defined(__GNUC__) || defined(__clang__)
    #define FAST_ALLOC_COLD __attribute__((cold, noinline))
#else
    #define FAST_ALLOC_COLD
#endif


/// @brief Where a slab keeps the links of its free list.
enum class FreeLinks : std::uint8_t
{
    /// In the freed elements themselves: a freed element's first bytes hold
    /// the index of the next free one. No extra memory, but an element must be
    /// at least an `int` wide and its contents do not survive a free.
    in_payload,
    /// In a side array (TCSlab::tab_first_free_), never in the elements: the
    /// allocator never writes into a free element, so an emptied slab's pages
    /// can be dropped (fast_alloc::EmptySlabSweep) without losing a
    /// link. The array is `unsigned short`, which caps a slab at
    /// TCSlab::free_list_end - 1 elements.
    side_array
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
 * @tparam Links Where the free list keeps its links: fast_alloc::TCAllocator threads it
 *         through the elements, fast_alloc::TCCAllocator keeps it aside.
 */
template <FreeLinks Links>
class alignas(64) TCSlab
{
  public:
    /// @brief The links live in #tab_first_free_ rather than in the elements.
    static constexpr bool side_links = Links == FreeLinks::side_array;

    /// @brief An element index on the free list.
    using index_type = std::conditional_t<side_links, unsigned short, int>;

    /// @brief Free-list terminator. With side links it is also the exclusive
    ///        upper bound on #block_size: the side array indexes elements with it.
    static constexpr index_type free_list_end = static_cast<index_type>(side_links ? 32767 : -1);

    /// @brief Leaves the slab unusable; init() is what makes it a slab.
    TCSlab() = default;

    /// @brief Returns the block to its arena.
    ~TCSlab()
    {
        destroy();
    }

    // Owns its block (and #tab_first_free_): a copy would release them twice.
    TCSlab(const TCSlab&)            = delete;
    TCSlab& operator=(const TCSlab&) = delete;
    TCSlab(TCSlab&&)                 = delete;
    TCSlab& operator=(TCSlab&&)      = delete;

    /// @brief The size this slab was cut at -- slab_bytes_for() of the request.
    [[nodiscard]] uintptr_t slab_bytes() const noexcept
    {
        return block_allocator;
    }

    /**
     * @brief Set the geometry and obtain the block.
     * @param elem_size Element size in BYTES.
     * @param block_bytes Requested block size in bytes; rounded up to a
     *        power of two (slab_bytes_for()). With side links the capacity, not
     *        the size, is capped to fit the index width.
     * @pre #arena_kind is set: the block is taken from that arena here.
     */
    inline void init(int elem_size, uintptr_t block_bytes);

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
        used_size  = 0;
        first_free = free_list_end;
    }

    /// @brief Index of the element at @p ptr within this slab: offset / #element_size
    ///        without a division. One 64x64->128 multiplication by #element_magic_
    ///        (Lemire, Kaser & Kurz 2019, exact for every 32-bit dividend and
    ///        divisor >= 2; init() refuses a slab over 4 GiB).
    /// @pre #element_size >= 2 -- a power of two goes through num_block_pow2().
    [[nodiscard]] int num_block(const void* ptr) const noexcept
    {
        const auto off = static_cast<std::uint64_t>(static_cast<const char*>(ptr) - data);
        FAST_ALLOC_DBG_ASSERT(element_size >= 2 && off <= UINT32_MAX);
        FAST_ALLOC_DBG_ASSERT(static_cast<int>((static_cast<unsigned __int128>(element_magic_) * off) >> 64) ==
                              static_cast<int>(off / static_cast<std::uint64_t>(element_size)));
        return static_cast<int>((static_cast<unsigned __int128>(element_magic_) * off) >> 64);
    }

    /// @brief num_block() by shift. Valid only if #element_size is a power of two.
    [[nodiscard]] int num_block_pow2(const void* ptr) const noexcept
    {
        return static_cast<int>((static_cast<const char*>(ptr) - data) >> log2_element_size_);
    }

    /// @brief No element of this slab is out. Only then may it be released.
    [[nodiscard]] bool is_empty() const noexcept
    {
        return nn_ref == 0;
    }

    /// @brief This free brought the slab back from full; it must leave block_full.
    [[nodiscard]] bool was_full() const noexcept
    {
        return nn_ref == block_size - 1;
    }

    /// @brief Every element is out; the slab must move to block_full.
    [[nodiscard]] bool is_full() const noexcept
    {
        return nn_ref == block_size;
    }

    /// @brief Effective slab size for a requested @p block_bytes: rounded up to
    ///        a power of two. ONE rule, shared by alloc_block(), the pool's
    ///        init() and slab_of(), so a pool resolves a pointer to its slab
    ///        without a slab to ask.
    /// @details Must not depend on the element size. A buffer is freed through
    ///          slab_of() of the allocator the thread's AMBIENT chunk size selects
    ///          (ChunkBase::release_to_owner), which after a re-chunk need not be
    ///          the one that allocated it: that finds the right slab only if every
    ///          allocator of one element type cuts its slabs at the same size.
    ///          So the side-link index width caps the CAPACITY (alloc_block()),
    ///          never the slab size; otherwise slab_of() reads a "slab" pointer
    ///          from the middle of a block (slab_geometry_validation).
    [[nodiscard]] static uintptr_t slab_bytes_for(uintptr_t block_bytes) noexcept
    {
        return next_power_of_2(block_bytes);
    }

    /// @brief The slab of @p slab_bytes (a power of two) holding @p ptr:
    ///        mask down to the slab base, read the back-pointer in its last word.
    [[nodiscard]] static TCSlab* slab_from(const void* ptr, uintptr_t slab_bytes) noexcept
    {
        // Step back from ptr itself (not an int-to-pointer cast): keeps its provenance.
        const char* base = static_cast<const char*>(ptr) - (reinterpret_cast<uintptr_t>(ptr) & (slab_bytes - 1));
        return *reinterpret_cast<TCSlab* const*>(base + slab_bytes - sizeof(void*));
    }

// Declaration-level switch, so it stays a directive: serial builds route slab
// headers through their own pool (tc_allocator.cpp, ::TCAllocator only),
// threaded builds (OpenMP, pool) use the global operators.
#if !FAST_ALLOC_PMP_THREADED
    void* operator new(size_t size);  ///< Slab headers come from their own pool.
    void  operator delete(void* ptr); ///< Counterpart of the above.
#endif

    // Which disjoint arena this slab draws from: op and ctx pools share a slab
    // class but MUST NOT share an address space (see slab_arena.h). Set by the
    // owning pool before init(); default Value for side links, Op otherwise.
    fast_alloc::ArenaKind arena_kind = side_links ? fast_alloc::ArenaKind::value : fast_alloc::ArenaKind::op;

    char* data = nullptr; ///< The block itself, aligned on #block_allocator.

    uintptr_t block_allocator = 0; ///< Block size in bytes; a power of two.

    index_type first_free{free_list_end}; ///< Head of the free list; #free_list_end if empty.

    int nn_ref{0}; ///< Elements currently out. 0 == releasable, #block_size == full.

    int used_size{0}; ///< Bump index: how far the linear carve has reached.

    int element_size{0}; ///< Element size in BYTES.

    int block_size{0}; ///< Capacity of the block, in elements.

    // for the linked list

    TCSlab* prev = nullptr; ///< Previous slab in the owner's block/block_full list.
    TCSlab* next = nullptr; ///< Next slab in the owner's block/block_full list.

    /// @brief The allocator instance whose lists this slab lives in. Written
    ///        once at creation by that instance, never changed while the slab
    ///        exists (a slab moving to another instance would rewrite it).
    ///        Answers "does this element belong to me?" in O(1); do not replace
    ///        it by a list walk (quadratic over the reuse-pool drain).
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
    ///        INCLUDING the trailing back-pointer (side links only). alloc()
    ///        restores it before handing out the first block, the only moment a
    ///        free through this slab becomes possible again. Plain bool: an
    ///        instance's slabs are allocated from by its own rank only, and the
    ///        flag is set at a serial point with no alloc in flight.
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
            return tab_first_free_[static_cast<std::size_t>(i)];
        }
        else
        {
            // store the index in the first bytes of the element
            return *reinterpret_cast<int*>(data + (static_cast<std::ptrdiff_t>(i) * element_size));
        }
    }

    /// @brief Free #tab_first_free_ and hand the block back to the arena named
    ///        by #arena_kind.
    inline void destroy();

    /**
     * @brief Obtain the block and lay it out.
     * @details Rounds the block size up to a power of two (slab_from() masks
     *          with it), derives #block_size from it -- capped to the index width
     *          with side links, which also sizes #tab_first_free_ -- takes the
     *          block from the arena and writes `this` into its last word. The
     *          block is not cleared, fresh or recycled: see the class @warning.
     */
    inline void alloc_block();

    int log2_element_size_ = 0; ///< log2(#element_size), for num_block_pow2().

    /// @brief num_block()'s multiplier: floor((2^64 - 1) / #element_size) + 1.
    std::uint64_t element_magic_ = 0;

    /// @brief Free-list links, one per element, with side links. Kept out of
    ///        the payload, so an emptied slab's pages can be dropped without
    ///        losing them.
    [[no_unique_address]] std::conditional_t<side_links, std::vector<unsigned short>, NoSideArray> tab_first_free_;
};

/// @brief Slab with its free list threaded through the freed elements: the op
///        and ctx pools, and the value pools of serial builds.
using TCAllocator = TCSlab<FreeLinks::in_payload>;

/// @brief Slab with its free list in a side array: the value pools of OpenMP
///        builds, whose emptied slabs an EmptySlabSweep can madvise away
///        (TCTypeCAllocator::collect_empty_slabs).
using TCCAllocator = TCSlab<FreeLinks::side_array>;

#if !FAST_ALLOC_PMP_THREADED
// Defined in tc_allocator.cpp. Only the in-payload slab is ever created in a
// serial build: ::TCTypeCAllocator exists in threaded builds only.
template <>
void* TCSlab<FreeLinks::in_payload>::operator new(size_t size);
template <>
void TCSlab<FreeLinks::in_payload>::operator delete(void* ptr);
#endif

/// @brief A whole-list check (sl_check, sl_contains): linear on every insert
///        and remove, so quadratic over a run. Only at the reinforced debug level
///        (FAST_ALLOC_CHECK_LISTS on a DEBUG build, alloc_config::check_slab_lists);
///        a plain Debug build keeps the cheap DBG_ASSERTs.
#define FAST_ALLOC_SL_LIST_CHECK(x)                   \
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
 * @note Recursive and linear -- a debug aid, called only from FAST_ALLOC_DBG_ASSERT.
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
 * @note Linear -- a debug aid, called only from FAST_ALLOC_DBG_ASSERT.
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
    FAST_ALLOC_DBG_ASSERT(node != nullptr);
    FAST_ALLOC_DBG_ASSERT(list != nullptr);
    FAST_ALLOC_SL_LIST_CHECK(sl_check(list));
    FAST_ALLOC_SL_LIST_CHECK(sl_contains(list, node));

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
        FAST_ALLOC_DBG_ASSERT(node->prev != nullptr);
        if (node->next != nullptr)
        {
            node->next->prev = node->prev;
        }
        node->prev->next = node->next;
    }
    node->prev = nullptr;
    node->next = nullptr;
    FAST_ALLOC_SL_LIST_CHECK(sl_check(list));
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
    FAST_ALLOC_DBG_ASSERT(node != nullptr);
    FAST_ALLOC_SL_LIST_CHECK(sl_check(list));

    node->prev = nullptr;
    node->next = list;
    if (list != nullptr)
    {
        list->prev = node;
    }
    list = node;

    FAST_ALLOC_SL_LIST_CHECK(sl_check(list));
    FAST_ALLOC_SL_LIST_CHECK(sl_contains(list, node));
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
 *          @p nb_max_block_empty negative (what the chunk layer passes, so a
 *          freed value slab's address is never reobtained by the op pool) its
 *          pages stay resident unless @p on_kept releases them. See
 *          alloc_config::reclaim_empty_slabs.
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
 * @brief Type-erased handle on a fast_alloc::TCTypeAllocator, for the memory report.
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
     * @param sum_used_size Elements ever carved, summed over slabs.
     * @param sum_nn_ref Elements currently out, summed over slabs.
     * @param alloc_sz Bytes held, summed over slabs.
     */
    virtual void add_allocated_size(size_t& sum_used_size, size_t& sum_nn_ref, size_t& alloc_sz) const = 0;

    /// @brief Former name of add_allocated_size(), kept for the programs that call it (an
    ///        override must use the new name).
    [[deprecated("methods are snake_case: use add_allocated_size")]] void
    addAllocatedSize(size_t& sum_used_size, size_t& sum_nn_ref, size_t& alloc_sz) const
    {
        add_allocated_size(sum_used_size, sum_nn_ref, alloc_sz);
    }
};

/// @brief Enrol an allocator in the registry the memory report walks.
void add_tc_type_allocator_inst(VTCTypeAllocator* /*inst*/);

/// @brief Withdraw it. Called from the destructor.
void remove_tc_type_allocator_inst(VTCTypeAllocator* /*inst*/);

/// @brief Former name of add_tc_type_allocator_inst(), kept for the programs that call it.
[[deprecated("functions are snake_case: use add_tc_type_allocator_inst")]] inline void
add_TCTypeAllocator_inst(VTCTypeAllocator* inst)
{
    add_tc_type_allocator_inst(inst);
}

/// @brief Former name of remove_tc_type_allocator_inst(), kept for the programs that call it.
[[deprecated("functions are snake_case: use remove_tc_type_allocator_inst")]] inline void
remove_TCTypeAllocator_inst(VTCTypeAllocator* inst)
{
    remove_tc_type_allocator_inst(inst);
}

/**
 * @brief Owns every slab serving one element size: the machinery common to
 *        fast_alloc::TCTypeAllocator and fast_alloc::TCTypeCAllocator.
 *
 * @details Slabs are kept in three places, and which one a slab is in is a
 *          function of its `nn_ref`: #block has room, #block_full has none, and
 *          #block_empty is a small reserve of emptied slabs kept back so the
 *          next allocation does not have to go to the arena. alloc() and the
 *          frees move slabs between them; the moves themselves live in
 *          fast_alloc::sl_manage_full_empty.
 *
 * @tparam Derived The pool (CRTP). It provides, as `static constexpr`:
 *         `eager_first_slab` (init() creates the first slab), `zero_std_alloc`
 *         (under FAST_ALLOC_STD_ALLOC, alloc() zero-fills what it hands out) and
 *         `pool_name` (for the [MUNMAP] trace).
 * @tparam ElemT Element type. What is handed out is an ARRAY of these --
 *         init() takes a count, and alloc() returns a pointer to the first.
 * @tparam Slab fast_alloc::TCAllocator or fast_alloc::TCCAllocator.
 *
 * @note One instance per (element size, thread) in the OpenMP value pools.
 *       Nothing here is synchronised: a thread only ever touches its own
 *       instance. A block released by another thread belongs to another
 *       instance's slab: route it with is_free_able_elt() / holds_live() before
 *       free_elt_pow2(); freeing through a non-owning instance corrupts the
 *       owner's free list.
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
     * @param element_size Number of ElementType per allocation -- a COUNT, not
     *        a byte size. Stored as bytes unless bypassing; see fast_alloc::bypass_slabs.
     * @param block_allocator Slab size in bytes, rounded up to a power of two.
     * @param nb_max_block_emptyl Capacity of the emptied-slab reserve.
     *        NEGATIVE means "never release a slab", which also means the reserve
     *        is not allocated at all.
     *
     * @note Whether the first slab is created here is the pool's call
     *       (`Derived::eager_first_slab`); alloc() makes one anyway when #block
     *       is null. fast_alloc::TCTypeCAllocator creates none: eager priming would cost
     *       one slab per (thread, geometry) pair that never allocates, and the
     *       value arena never takes a slab back.
     */
    void init(size_t element_size, uintptr_t block_allocator, int nb_max_block_emptyl = 5)
    {
        nb_max_block_empty_ = nb_max_block_emptyl;

        if (bypass_slabs())
        {
            // new[] takes an element count, so that is what element_size_ holds.
            element_size_ = element_size;
        }
        else
        {
            // A slab is carved in bytes.
            element_size_ = element_size * sizeof(ElementType);
            // new_slab() hands it to TCSlab::init() as an int.
            if (element_size_ > static_cast<size_t>(std::numeric_limits<int>::max())) [[unlikely]]
            {
                FAST_ALLOC_FATAL("TCAllocator: element size {} bytes is past INT_MAX", element_size_);
            }

            block_allocator_ = block_allocator;
            slab_bytes_      = Slab::slab_bytes_for(block_allocator);

            if constexpr (Derived::eager_first_slab)
            {
                // Must be stamped with its owner, as alloc_slow() does: a free of
                // an unstamped slab's block falls back to the CALLER's allocator
                // (in a dynamic build, whatever chunk size the thread last served),
                // which unlinks the slab from lists it was never on
                // (ChunkBase::release_to_owner).
                insert(block, new_slab());
            }

            // nb_max_block_empty < 0 ("never munmap"): block_empty is never used.
            if (nb_max_block_empty_ > 0)
            {
                block_empty_.assign(static_cast<std::size_t>(nb_max_block_empty_), nullptr);
            }
        }

        nb_block_empty_ = 0;
    }

    /**
     * @brief Hand out room for #element_size_ elements.
     * @return Memory with unspecified contents; the caller initialises it (see
     *         TCSlab's @warning). Zero-filled under FAST_ALLOC_STD_ALLOC when the pool asks
     *         for it (`Derived::zero_std_alloc`).
     * @details Draws from the current slab; failing that, revives one from the
     *          reserve; failing that, creates a new slab. A slab that fills up
     *          moves to #block_full, so #block can be null at any time.
     */
    [[nodiscard]] ElementType* alloc()
    {
        // if constexpr: the memset must not even be instantiated in a slab
        // build (gcc's -Wclass-memaccess on Bit, std::complex).
        if constexpr (bypass_slabs())
        {
            auto* elements = new ElementType[element_size_]; // element_size_ is a COUNT here
            if constexpr (Derived::zero_std_alloc)
            {
                std::memset(elements, 0, element_size_ * sizeof(ElementType));
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
     * @pre #element_size_ is a power of two when sizeof(ElementType) is, i.e.
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
        // The shift when this slab's element size is a power of two, the
        // magic multiplication otherwise (e.g. 192 doubles, a 24-byte struct,
        // a 40-byte op).
        // Tested on the slab, not deduced from sizeof(ElementType): the element
        // COUNT need not be a power of two. One well-predicted branch.
        const auto size = static_cast<unsigned int>(slab->element_size);
        const int  i    = std::has_single_bit(size) ? slab->num_block_pow2(ptr) : slab->num_block(ptr);

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
        // Mask with #slab_bytes_ (the ROUNDED size, Slab::slab_bytes_for), never
        // the raw budget: a non-pow2 budget would land beside the slab.
        if (slab_bytes_ == 0)
        {
            return nullptr;
        }
        return Slab::slab_from(ptr, slab_bytes_);
    }

    /**
     * @brief Hand every slab of this instance that holds no block to its
     *        arena's recycled list for this instance's rank
     *        (fast_alloc::SlabArena::recycle_batch): the slab leaves this
     *        instance, its header is deleted, its block KEEPS its pages.
     * @param scratch Reused buffer of block addresses; cleared here.
     * @return Number of slabs handed over.
     * @details At the re-chunk's slice boundary, instead of madvise'ing the
     *          slabs an abandoned (key, size) set just emptied: the next slice
     *          allocates about as many at the same slab size (slab_bytes_for()
     *          ignores the element size) and takes these back resident.
     * @pre Stronger than an EmptySlabSweep's: every slab found empty IS empty
     *      (a set abandoned at a quiescent point), because a recycled slab is
     *      reused by another instance and a block wrongly counted out would be
     *      overwritten. No other thread uses this instance meanwhile: its own
     *      rank does this, or any thread while that rank is idle.
     */
    std::size_t recycle_empty_slabs(std::vector<void*>& scratch) noexcept
    {
        const std::size_t n = detach_empty_slabs(scratch);
        recycle_detached(scratch);
        return n;
    }

    /**
     * @brief The first half of recycle_empty_slabs(): unlink every empty slab
     *        and delete its header, recording its block in @p scratch, WITHOUT
     *        touching the arena (no lock).
     * @param scratch Reused buffer of block addresses; cleared here.
     * @return Number of slabs detached.
     * @details For a team: each thread detaches its own rank's slabs in
     *          parallel, then one thread hands every batch to the arena
     *          (recycle_detached()), rather than all queueing on its mutex. Same
     *          preconditions as recycle_empty_slabs(). The blocks belong to
     *          nobody until recycle_detached() runs: call it before anything allocates.
     */
    std::size_t detach_empty_slabs(std::vector<void*>& scratch) noexcept
    {
        scratch.clear();
        if constexpr (!alloc_config::use_slabs)
        {
            return 0;
        }
        else
        {
            if (bypass_slabs() || slab_bytes_ == 0)
            {
                return 0;
            }
            for (Slab** head : {&block, &block_full})
            {
                for (Slab* p = *head; p != nullptr;)
                {
                    Slab* const next = p->next;
                    if (p->nn_ref == 0)
                    {
                        try
                        {
                            scratch.push_back(p->data);
                        }
                        catch (...)
                        {
                            break; // no room to record it: it stays, as it was
                        }
                        FAST_ALLOC_DBG_ASSERT(p->owner == self());
                        remove(*head, p);
                        p->data = nullptr; // the block goes back in the batch below, not through ~Slab
                        delete p;
                    }
                    p = next;
                }
            }
            return scratch.size();
        }
    }

    /// @brief The second half of recycle_empty_slabs(): hand the blocks
    ///        detach_empty_slabs() recorded to this instance's arena, under
    ///        this instance's rank (fast_alloc::SlabArena::recycle_batch).
    /// @param scratch The blocks, as detach_empty_slabs() left them.
    void recycle_detached(const std::vector<void*>& scratch) const noexcept
    {
        if constexpr (alloc_config::use_slabs)
        {
            fast_alloc::arena_for(arena_kind_).recycle_batch(scratch.data(), scratch.size(), slab_bytes_, rank_);
        }
    }

    /// @brief Give back the pages of the recycled slabs nobody took, in this
    ///        instance's arena (fast_alloc::SlabArena::advise_recycled).
    /// @param keep_bytes Bytes of them to leave resident (SlabArena::advise_recycled).
    /// @return Number of slabs advised.
    std::size_t advise_recycled_slabs(std::size_t keep_bytes = 0) const noexcept
    {
        if constexpr (!alloc_config::use_slabs)
        {
            return 0;
        }
        else
        {
            return fast_alloc::arena_for(arena_kind_).advise_recycled(keep_bytes);
        }
    }

    /**
     * @brief Choose which arena this pool's slabs are drawn from.
     * @param k Op, Ctx or Value.
     * @warning Must be set BEFORE the first slab is taken (copied into each
     *          slab at creation). fast_alloc::TCTypeAllocator defaults to Op: a pool that
     *          holds values must say so, or its slabs share the op pool's address
     *          space (the aliasing slab_arena.h prevents).
     */
    void set_arena(fast_alloc::ArenaKind k) noexcept
    {
        arena_kind_ = k;
    }

    /// @brief Set the rank (thread) this instance serves, stamped into every
    ///        slab it creates (TCSlab::owner_rank); before the first slab.
    ///        -1, the default, means unranked (serial pools).
    void set_rank(int rank) noexcept
    {
        rank_ = rank;
    }

    /// @brief Set the affinity key this instance serves (0, the default, is
    ///        the rank's shared pool), stamped into its slabs as owner_key;
    ///        before the first slab.
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

        decltype(block_empty_)().swap(block_empty_); // releases the reserve's storage

        element_size_ = 0;
        block_full = block = nullptr;
        nb_block_empty_    = 0;
    }

    Slab* block_full{nullptr}; ///< Slabs with no room left.
    Slab* block{nullptr};      ///< Slabs with room. Null until the first slab.

  private:
    friend Derived; // CRTP: only the derived pool constructs this base

    /// @brief Empty and unusable until init(); its slabs will draw from @p arena.
    explicit TCSlabPool(fast_alloc::ArenaKind arena) noexcept : arena_kind_(arena) {}

  protected:
    /// @brief Frees every slab.
    ~TCSlabPool()
    {
        destroy_int();
    }

  private:
    /// @brief The pool as its slabs name it (TCSlab::owner): the DERIVED
    ///        object. ChunkBase::release_to_owner casts the owner back to the
    ///        pool type, and a base subobject need not share its address --
    ///        fast_alloc::TCTypeAllocator's sits after the fast_alloc::VTCTypeAllocator vtable.
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
        // Under FAST_ALLOC_STD_ALLOC there are no slabs: each element was a new[]
        // that free_elt_pow2() already deleted.
        if constexpr (alloc_config::use_slabs)
        {
            if (bypass_slabs())
            {
                return;
            }
            sl_delete_all(block, block_full, block_empty_.data(), nb_block_empty_);
        }
    }

    /// @brief A new slab owned by this instance, from its arena, at its geometry.
    Slab* new_slab()
    {
        auto* slab       = new Slab;
        slab->arena_kind = arena_kind_; // route slab to its op/ctx/value arena
        slab->owner      = self();
        slab->owner_rank = rank_;
        slab->owner_key  = key_;
        slab->init(static_cast<int>(element_size_), block_allocator_);
        FAST_ALLOC_DBG_ASSERT(slab->slab_bytes() == slab_bytes_);
        return slab;
    }

    /// @brief alloc() with no slab at hand: revive one from the reserve or
    ///        create one. Out of line and cold -- see FAST_ALLOC_COLD.
    FAST_ALLOC_COLD ElementType* alloc_slow()
    {
        if (nb_block_empty_ > 0)
        {
            auto* slab = block_empty_[--nb_block_empty_];

            insert(block, slab);

            slab->reset();

            return alloc_and_file(slab);
        }

        auto* slab = new_slab();

        insert(block, slab);

        return alloc_and_file(slab);
    }

    /// @brief alloc() on a slab just put in #block, then file it as the fast
    ///        path does: a slab this allocation filled (one element per slab for
    ///        an element over half the slab) MUST move to #block_full, or the
    ///        next alloc() runs past its end.
    ElementType* alloc_and_file(Slab* slab)
    {
        auto* ptr = static_cast<ElementType*>(slab->alloc());
        if (slab->is_full()) [[unlikely]]
        {
            remove(block, slab);
            insert(block_full, slab);
        }
        return ptr;
    }

    /// @brief Re-file @p slab after a free. @see sl_manage_full_empty
    void manage_full_empty(Slab* slab) noexcept
    {
        sl_manage_full_empty(
            block, block_full, block_empty_.data(), nb_block_empty_, nb_max_block_empty_, slab,
            [](Slab* kept) noexcept
            {
                // NO madvise here, whatever reclaim_empty_slabs says. On this hot
                // path nn_ref == 0 is the ONLY evidence the slab holds nothing
                // live, and a mis-accounted block (a buffer parked or released
                // twice, two threads on one counter) falsifies it: MADV_DONTNEED
                // would turn that into a live page reading back as zeros. Left
                // mapped, the same error is harmless.
                //
                // Pages are given back only through
                // TCTypeCAllocator::collect_empty_slabs() and its sweep, on a set
                // the caller KNOWS abandoned (the (key, size) set a re-chunked
                // array just left). Here the slab stays in `block`, allocatable.
                (void)kept;
            },
            [this](Slab* evicted) noexcept { evict_slab(evicted); });
    }

    /// @brief Release an emptied slab the reserve has no room for. Out of
    ///        line and cold (see FAST_ALLOC_COLD).
    FAST_ALLOC_COLD void evict_slab(Slab* slab) noexcept
    {
        if (fast_alloc::trace_evictions())
        {
            CHUNK_LOG(trace, alloc, "munmap: {} delete slab p={} nb_max_block_empty={}", Derived::pool_name,
                      static_cast<void*>(slab), nb_max_block_empty_);
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

    // Arena of the slabs this pool creates. See set_arena().
    fast_alloc::ArenaKind arena_kind_;

    size_t    element_size_    = 0;  ///< Bytes per allocation; a COUNT when bypassing.
    uintptr_t block_allocator_ = 0;  ///< Slab size in bytes, passed on to each slab.
    uintptr_t slab_bytes_      = 0;  ///< The ROUNDED slab size every slab of ours is cut at; 0 until init().
    int       rank_            = -1; ///< Rank served; stamped into slabs as owner_rank.
    unsigned  key_             = 0;  ///< Affinity key served; stamped into slabs as owner_key.

    /// @brief Reserve capacity; negative = never release a slab. @see init
    int nb_max_block_empty_ = 0;

    std::vector<Slab*> block_empty_;       ///< The reserve. Empty when the capacity is <= 0.
    int                nb_block_empty_{0}; ///< How much of the reserve is occupied.
};

/**
 * @brief Owns every fast_alloc::TCAllocator slab serving one element size.
 *
 * @details The serial family, and the op and ctx pools of every build:
 *          fast_alloc::TCSlabPool over slabs whose free list runs through the elements.
 *          init() creates the first slab, the slabs draw from the op arena
 *          unless set_arena() says otherwise, and every instance enrols in the
 *          fast_alloc::VTCTypeAllocator registry for the memory report.
 *
 * @tparam ElemT Element type; an allocation is an ARRAY of these.
 *
 * @note fast_alloc::TCTypeCAllocator is its OpenMP value counterpart (links outside the
 *       payload); fast_alloc::TCTypeAllocatorEBC is the compile-time geometry variant.
 */
template <typename ElemT>
class TCTypeAllocator final : public VTCTypeAllocator, public TCSlabPool<TCTypeAllocator<ElemT>, ElemT, TCAllocator>
{
  public:
    /// @brief init() creates the first slab. It costs its address space and the
    ///        page of its back-pointer, not the whole slab: alloc_block() does
    ///        not clear it.
    static constexpr bool eager_first_slab = true;
    /// @brief FAST_ALLOC_STD_ALLOC hands out new[] as is.
    static constexpr bool zero_std_alloc = false;
    /// @brief For the [MUNMAP] trace.
    static constexpr const char* pool_name = "TCTypeAllocator";

    /// @brief Empty and unusable until init(); enrols in the report registry.
    TCTypeAllocator() : TCSlabPool<TCTypeAllocator<ElemT>, ElemT, TCAllocator>(fast_alloc::ArenaKind::op)
    {
        add_tc_type_allocator_inst(this);
    }

    /// @brief Unregisters from the instance list; the base then frees every slab.
    ~TCTypeAllocator() override
    {
        remove_tc_type_allocator_inst(this);
    }

    // Owns its slabs and is enrolled by address in the report registry.
    TCTypeAllocator(const TCTypeAllocator&)            = delete;
    TCTypeAllocator& operator=(const TCTypeAllocator&) = delete;
    TCTypeAllocator(TCTypeAllocator&&)                 = delete;
    TCTypeAllocator& operator=(TCTypeAllocator&&)      = delete;

  private:
    /**
     * @copydoc VTCTypeAllocator::add_allocated_size
     * @note @p alloc_sz counts ELEMENTS, not bytes -- it sums block_size over the
     *       slabs. The reserve in #block_empty is not counted, so a run that
     *       parks slabs there reports less than it holds.
     */
    void add_allocated_size(size_t& sum_used_size, size_t& sum_nn_ref, size_t& alloc_sz) const noexcept override
    {
        size_t s_blocks    = 0;
        size_t s_used_size = 0;
        size_t s_nn_ref    = 0;

        for (const TCAllocator* node = this->block; node != nullptr; node = node->next)
        {
            s_used_size += static_cast<size_t>(node->used_size) * node->element_size;
            s_nn_ref += static_cast<size_t>(node->nn_ref) * node->element_size;
            s_blocks += node->block_size;
        }

        for (const TCAllocator* node = this->block_full; node != nullptr; node = node->next)
        {
            s_blocks += node->block_size;
        }

        sum_used_size += s_used_size;
        sum_nn_ref += s_nn_ref;
        alloc_sz += s_blocks;
    }
};

// Declaration-level switch, so it stays a directive: this type exists only in
// threaded builds (OpenMP, pool). Serial builds use ::TCTypeAllocator throughout.
#if FAST_ALLOC_PMP_THREADED

/**
 * @brief Owns every fast_alloc::TCCAllocator slab serving one element size.
 *
 * @details The OpenMP value pools: fast_alloc::TCSlabPool over slabs that keep their free
 *          list outside the payload, so an emptied slab can be madvised away
 *          (collect_empty_slabs(), its own part). Its slabs draw from the
 *          value arena; init() creates none; it does not enrol in the
 *          fast_alloc::VTCTypeAllocator report registry; under FAST_ALLOC_STD_ALLOC it zero-fills
 *          what it hands out.
 *
 *          alignas(64): it lives in per-rank arrays; unaligned, one rank's list
 *          heads (block_full, block) would share a cache line with the previous
 *          rank's reserve (false sharing).
 *
 * @tparam ElemT Element type; an allocation is an ARRAY of these.
 */
template <typename ElemT>
class alignas(64) TCTypeCAllocator : public TCSlabPool<TCTypeCAllocator<ElemT>, ElemT, TCCAllocator>
{
  public:
    /// @brief No slab at init(): the first alloc() makes one. See TCSlabPool::init.
    static constexpr bool eager_first_slab = false;
    /// @brief FAST_ALLOC_STD_ALLOC zero-fills.
    static constexpr bool zero_std_alloc = true;
    /// @brief For the [MUNMAP] trace.
    static constexpr const char* pool_name = "TCCAllocator";
    /// @brief The batch collect_empty_slabs() feeds: one madvise per run of
    ///        adjacent empty slabs.
    using SlabSweep = fast_alloc::EmptySlabSweep<TCCAllocator>;

    /// @brief Empty and unusable until init(); its slabs draw from the value arena.
    TCTypeCAllocator() : TCSlabPool<TCTypeCAllocator<ElemT>, ElemT, TCCAllocator>(fast_alloc::ArenaKind::value) {}

    /// @brief Hand every slab of this instance that currently holds no block to
    ///        @p sweep, which advises them away at its flush().
    /// @details manage_full_empty() never madvises, so an emptied slab stays
    ///          resident (after a re-chunk, a whole abandoned (key, size) set).
    ///          Call this where a whole set is known released (the re-chunk
    ///          drain), never on the hot path. Several instances (every rank,
    ///          several keys) may feed ONE sweep, so adjacent runs merge across
    ///          them. EmptySlabSweep's preconditions hold until its flush.
    /// @param sweep      The batch the empty slabs are handed to.
    /// @param keep_empty Empty slabs this instance keeps resident (the first
    ///        ones found), for its next allocations: a sweep between two
    ///        phases that keeps a few does not give back, then fault in again,
    ///        the slabs the next phase allocates.
    /// @return Number of slabs handed over. The CHUNK_RECLAIM_MIN_SLABS floor
    ///         is applied by the sweep, to all instances together.
    std::size_t collect_empty_slabs([[maybe_unused]] SlabSweep&  sweep,
                                    [[maybe_unused]] std::size_t keep_empty = 0) noexcept
    {
        if constexpr (alloc_config::use_std_alloc || !alloc_config::reclaim_empty_slabs)
        {
            return 0;
        }
        else
        {
            // No CHUNK_RECLAIM_MIN_SLABS test here: the floor is the sweep's
            // (EmptySlabSweep::flush).
            std::size_t collected = 0;
            for (auto* node : {this->block, this->block_full})
            {
                for (auto* p = node; p != nullptr; p = p->next)
                {
                    // Whole slab, trailing page included: see fast_alloc::EmptySlabSweep.
                    if (p->nn_ref == 0)
                    {
                        if (keep_empty > 0)
                        {
                            --keep_empty;
                            continue;
                        }
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


/// @brief Narrowest signed integer that fits in @p Bytes bytes (1, 2, else int):
///        a slab's free-list index sized to the element, rather than an `int`
///        per element on a small block.
template <unsigned int Bytes>
using FreeIndexT = std::conditional_t<Bytes == 1, signed char, std::conditional_t<Bytes == 2, short, int>>;


/**
 * @brief One slab whose element and block sizes are template parameters.
 *
 * @details Same job as fast_alloc::TCAllocator, with the geometry known at compile time:
 *          the capacity, the shift used for indexing and the free-list index
 *          width all become constants, so the division in num_block() and the
 *          per-slab geometry fields go away.
 *
 * @tparam ElementBytes Element size in bytes.
 * @tparam BlockBytes Block size in bytes.
 *
 * @note Used only through fast_alloc::TCTypeAllocatorEBC, which owns the slab lists.
 */
template <unsigned int ElementBytes, unsigned int BlockBytes>
class TCAllocatorEB
{
  public:
    /// @brief The compile-time geometry: sizes in bytes, capacity in elements.
    enum
    {
        element_size      = ElementBytes,
        log2_element_size = std::bit_width(ElementBytes) - 1,
        block_size        = BlockBytes,
        block_size_pot    = static_cast<int>(std::bit_ceil(BlockBytes)), // int: the mask below must sign-extend
        block_addr_mask   = ~(block_size_pot - 1),
        get_size_of       = sizeof(TCAllocatorEB*),
        used_block_size   = BlockBytes - get_size_of,
        max_elements      = used_block_size / ElementBytes
    };

    /// @brief Free-list index type, as narrow as the element allows.
    using T = FreeIndexT<std::bit_floor(ElementBytes)>;

    // The free list stores an element INDEX of type T inside each freed
    // element: T must fit in an element, and must count every element of the
    // block (a signed char stops at 127, a short at 32767).
    static_assert(sizeof(T) <= static_cast<std::size_t>(element_size),
                  "TCAllocatorEB: the free-list index does not fit in an element");
    static_assert(static_cast<long long>(max_elements) <= static_cast<long long>(std::numeric_limits<T>::max()),
                  "TCAllocatorEB: more elements per block than the free-list index type can count");

    /// @brief Takes its block straight away -- there is no separate init().
    TCAllocatorEB()
    {
        // alloc() inserts the slab right after this constructor returns: a
        // slab without a block (or without its back-pointer) would be handed
        // out and corrupt memory silently. Abort, as the other families do.
        if (alloc_block() != 0)
        {
            FAST_ALLOC_FATAL("TCAllocatorEB: its first block could not be allocated");
        }
    }

    /// @brief Returns the block to whichever allocator alloc_block() used.
    ~TCAllocatorEB()
    {
        if (use_mmap_)
        {
            munmap(data, block_size);
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
     *          fast_alloc::TCAllocator, but with the index narrowed to #T so a small block
     *          does not spend four bytes per element on it.
     */
    void* alloc() noexcept
    {
        ++nn_ref;
        char* elem_ptr = nullptr;

        if (first_free != -1)
        {
            elem_ptr   = data + (first_free * element_size);
            first_free = *(T*)elem_ptr;
        }
        else
        {
            elem_ptr = data + (used_size * element_size);
            ++used_size;
        }
        return static_cast<void*>(elem_ptr);
    }

    /// @brief Index of the element at @p ptr within this slab (division).
    int num_block(const void* ptr) const noexcept
    {
        return static_cast<int>((static_cast<const char*>(ptr) - data) / element_size);
    }

    /// @brief num_block() by shift. Valid only if #element_size_ is a power of two.
    int num_block_pow2(const void* ptr) const noexcept
    {
        return static_cast<int>((static_cast<const char*>(ptr) - data) >> log2_element_size);
    }

    /**
     * @brief Take element @p i back.
     * @param i Index from num_block() / num_block_pow2().
     * @details Freeing the element the bump pointer just handed out rewinds the
     *          bump instead of pushing onto the free list.
     */
    void free_block(int i) noexcept
    {
        if (i == (used_size - 1))
        {
            *(T*)(data + (i * element_size)) = -1;
            --used_size;
        }
        else
        {
            *(T*)(data + (i * element_size)) = first_free;
            first_free                       = i;
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
        return nn_ref == max_elements - 1;
    }

    /// @brief Every element is out; the slab must move to block_full.
    [[nodiscard]] bool is_full() const noexcept
    {
        return nn_ref == max_elements;
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
        // Step back from ptr itself (not an int-to-pointer cast): keeps its provenance.
        const char* base =
            static_cast<const char*>(ptr) - (reinterpret_cast<uintptr_t>(ptr) & ~size_t(block_addr_mask));
        return *reinterpret_cast<TCAllocatorEB* const*>(base + used_block_size);
    }

    char* data = nullptr; ///< The block itself, aligned on #block_size_pot.
    int   first_free{-1}; ///< Head of the free list, an element index; -1 if empty.

    int nn_ref{0};    ///< Elements currently out. 0 == deletable.
    int used_size{0}; ///< Bump index: how far the linear carve has reached.

    // for the linked list
    TCAllocatorEB* prev = nullptr; ///< Previous slab in the owner's block/block_full list.
    TCAllocatorEB* next = nullptr; ///< Next slab in the owner's block/block_full list.

    // A free element holds the index of the next free one, in T (sized from the
    // element): the block must be addressable by it.
    static_assert(static_cast<uintptr_t>(BlockBytes) < (uintptr_t{1} << ((sizeof(T) * 8) - 1)),
                  "TCAllocatorEB: block too large for the element's free-list index type");

  private:
    /**
     * @brief Obtain the block and lay it out.
     * @return 0, or 1 when no block could be obtained (posix_memalign failed);
     *         the constructor aborts then.
     * @details The block must be aligned on its own power-of-two size, because
     *          addr_to_block() masks with #block_addr_mask. The mmap path maps
     *          twice the size and unmaps the unaligned head and the tail; the
     *          posix_memalign path asks for the alignment explicitly.
     *          A failed trimming munmap (ENOMEM at vm.max_map_count) is not a
     *          failure: the aligned block is usable, only the surplus stays
     *          mapped. It is reported.
     */
    int alloc_block()
    {
        const long ps = fast_alloc::page_size();
        if (ps > 0 && (2 * block_size) % ps == 0)
        {
            use_mmap_ = true;

            const uintptr_t block_bytes = block_size;

            char* mapping = static_cast<char*>(fast_alloc::map_anonymous(2 * block_size));

            data = mapping + ((0 - reinterpret_cast<uintptr_t>(mapping)) & (block_bytes - 1)); // align up

            const uintptr_t lng = reinterpret_cast<uintptr_t>(data) - reinterpret_cast<uintptr_t>(mapping);
            if (lng != 0 && munmap(mapping, lng) == -1)
            {
                const int err = errno;
                CHUNK_LOG(warn, alloc, "munmap (head trim) failed: {} -- surplus left mapped", std::strerror(err));
            }
            if (munmap(data + block_size, (2 * block_size) - (lng + block_size)) == -1)
            {
                const int err = errno;
                CHUNK_LOG(warn, alloc, "munmap (tail trim) failed: {} -- surplus left mapped", std::strerror(err));
            }
        }
        else
        {
            use_mmap_   = false;
            void* datal = nullptr;

            const int result = posix_memalign(&datal, block_size_pot, block_size);

            if (result != 0)
            {
                CHUNK_LOG(error, alloc, "posix_memalign failed: error {} ({})", result, std::strerror(result));
                return 1;
            }
            data = static_cast<char*>(datal);
        }

        *reinterpret_cast<TCAllocatorEB**>(data + used_block_size) = this;

        first_free = -1;

        return 0;
    }

    bool use_mmap_ = false; ///< Which of munmap / free the destructor must use.
};

/**
 * @brief Owns every fast_alloc::TCAllocatorEB slab for one compile-time geometry.
 *
 * @details Third slab family, same three-way split (#block, #block_full,
 *          #block_empty); the geometry is a template parameter, so indexing
 *          folds to shifts and the allocator carries no size fields.
 *
 * @tparam ElementSize Element size in bytes.
 * @tparam BlockSize Block size in bytes.
 *
 * @note This is what StaticConstArray's size ladder is built from -- one
 *       instance per small block size (4, 8, 24, ... bytes).
 *
 * @note Like fast_alloc::TCTypeCAllocator, the constructor takes no slab: the first
 *       alloc() makes one. Do not seed eagerly: across all of StaticConstArray's
 *       buckets that reserves ~100 MiB of address space up front.
 */
template <unsigned int ElementSize, unsigned int BlockSize>
class TCTypeAllocatorEBC
{
  public:
    /// @brief The slab type this instance owns.
    using TCAllocatorEBT = TCAllocatorEB<ElementSize, BlockSize>;

    /// @brief No slab yet; the first alloc() creates one. No init() to call.
    TCTypeAllocatorEBC() = default;

    /// @brief Deletes every slab, the reserve included.
    ~TCTypeAllocatorEBC()
    {
        sl_delete_all(block_, block_full_, block_empty_.data(), nb_block_empty_);
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
        if (block_ == nullptr)
        {
            if (nb_block_empty_ > 0)
            {
                auto* slab = block_empty_[--nb_block_empty_];

                insert(block_, slab);

                slab->used_size  = 0;
                slab->first_free = -1;

                return slab->alloc();
            }

            auto* slab = new TCAllocatorEBT;

            insert(block_, slab);

            return slab->alloc();
        }

        auto* current_block = block_;

        void* ptr = current_block->alloc();
        if (current_block->is_full())
        {
            remove(block_, current_block);
            insert(block_full_, current_block);
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
     * @param sum_used_size Bytes ever carved, summed over slabs.
     * @param sum_nn_ref Bytes currently out, summed over slabs.
     * @param alloc_sz Bytes held, summed over slabs.
     * @note Unlike TCTypeAllocator::add_allocated_size, these are BYTES: the
     *       compile-time ElementSize and BlockSize are folded in here.
     */
    void add_allocated_size(size_t& sum_used_size, size_t& sum_nn_ref, size_t& alloc_sz) const noexcept
    {
        size_t nb_blocks   = 0;
        size_t s_used_size = 0;
        size_t s_nn_ref    = 0;

        // const: this walk only totals, it never touches a slab.
        const TCAllocatorEBT* node = block_;

        while (node != nullptr)
        {
            s_used_size += node->used_size;
            s_nn_ref += node->nn_ref;
            nb_blocks++;

            const TCAllocatorEBT* next = node->next;
            node                       = next;
        }

        node = block_full_;

        while (node != nullptr)
        {
            nb_blocks++;

            const TCAllocatorEBT* next = node->next;
            node                       = next;
        }

        sum_used_size += s_used_size * size_t(ElementSize);
        sum_nn_ref += s_nn_ref * size_t(ElementSize);
        alloc_sz += nb_blocks * size_t(BlockSize);
    }

    /// @brief Former name of add_allocated_size(), kept for the programs that call it.
    [[deprecated("methods are snake_case: use add_allocated_size")]] void
    addAllocatedSize(size_t& sum_used_size, size_t& sum_nn_ref, size_t& alloc_sz) const noexcept
    {
        add_allocated_size(sum_used_size, sum_nn_ref, alloc_sz);
    }

  private:
    /// @brief Reserve capacity. Fixed, unlike the other families' runtime value.
    static constexpr int nb_max_block_empty = 5;

    /**
     * @brief Re-file @p slab after a free, EBC's own way.
     *
     * @details Deliberately NOT fast_alloc::sl_manage_full_empty. Three differences: a
     *          slab that comes back from full returns immediately instead of
     *          also being tested for emptiness; there is no never-release
     *          branch, the reserve being a fixed inline array; and a slab past
     *          the reserve is always deleted outright.
     */
    void manage_full_empty(TCAllocatorEBT* slab) noexcept
    {
        if (slab->was_full())
        {
            if (block_full_)
            {
                remove(block_full_, slab);
                insert(block_, slab);
            }
            return;
        }

        if (slab->is_empty())
        {
            if (nb_block_empty_ < nb_max_block_empty)
            {
                block_empty_[nb_block_empty_++] = slab;
                remove(block_, slab);
            }
            else
            {
                remove(block_, slab);
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

    TCAllocatorEBT* block_full_{nullptr}; ///< Slabs with no room left.
    /// @brief Slabs with room. Null until the first alloc(), and again whenever
    ///        the last one fills.
    TCAllocatorEBT* block_{nullptr};

    int                                             nb_block_empty_{0}; ///< Reserve occupancy.
    std::array<TCAllocatorEBT*, nb_max_block_empty> block_empty_{};     ///< The reserve, inline.
};

/// @brief Default slab size (1 MiB).
/// @details constexpr rather than a macro so it carries a type and scope; used
///          as a non-type template argument below and in StaticConstArray,
///          where an unsigned int constant expression is required.
inline constexpr unsigned int size_block = 1u << 20u; // 1048576

/// @brief The former name of size_block, kept for programs that use it.
[[deprecated("constants are snake_case: use fast_alloc::size_block")]] inline constexpr unsigned int SIZE_BLOCK =
    size_block;

} // namespace fast_alloc

#include "fast_allocators/internal/tc_allocator_inline.h" // the template bodies
