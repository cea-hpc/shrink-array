
#pragma once

/**
 * @file tc_allocator.h
 * @brief Slab allocators: one mmap'd block is carved into fixed-size elements.
 *
 * Four layers, bottom to top:
 *
 *  - ::TCAllocator and ::TCCAllocator are ONE slab each. They own a block of
 *    memory, hand out fixed-size elements from it, and keep a free list of the
 *    elements handed back. They are also the nodes of an intrusive list.
 *  - ::sl_insert, ::sl_remove and ::sl_manage_full_empty are the list
 *    primitives, shared by every family so a fix lands in one place.
 *  - ::TCTypeAllocator and ::TCTypeCAllocator own a SET of slabs for one
 *    element size, split three ways: `block` (room left), `block_full`, and
 *    `block_empty` (a small reserve of emptied slabs kept for reuse).
 *  - ::TCTypeAllocatorEBC is a third family whose element and block sizes are
 *    template parameters rather than run-time values.
 *
 * @par Why two slab classes
 * ::TCAllocator threads its free list THROUGH the freed elements, writing the
 * next free index into the element's first bytes. ::TCCAllocator does not: it
 * keeps the links in a side array (TCCAllocator::tab_firstFree) and never writes
 * into a free element, so an emptied slab's pages can be dropped with madvise()
 * without losing anything the allocator needs. That
 * side array is indexed by `unsigned short`, which is what caps a TCCAllocator
 * slab at TCCAllocator::first_free_init - 1 elements.
 *
 * @par Configuration
 * The macros are read in alloc_config.h and nowhere else, so the bodies here
 * branch with `if constexpr`. Both arms must compile in every configuration --
 * that is the point. See ::bypass_slabs.
 *
 * @warning A slab's own address is recovered from any interior pointer by
 *          masking (TCAllocator::addr_to_block), which requires the block to be
 *          aligned on its own size and to store a back-pointer in its last
 *          word. Both are established by alloc_block(); neither is checked
 *          afterwards.
 */

#include <vector>
#include <array>
#include <bit>     // std::has_single_bit: the shift or the division in free_elt_pow2
#include <cerrno>  // errno, after a failed mmap
#include <cstddef> // std::size_t
#include <cstdio>  // std::fprintf / std::fflush (the [MUNMAP] traces)
#include <cstring>
#include <iostream>
#include <cstdlib>
#include <cstdint>
#include <limits> // std::numeric_limits: TCAllocatorEB's index range
#include <cassert>
#include <sys/mman.h>
#include <unistd.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "fast_allocators/internal/alloc_config.h"    // constexpr mirrors of STD_ALLOC / DEBUG / _OPENMP
#include "fast_allocators/slab_arena.h"               // FIX: ArenaKind for op/ctx/value slab routing
#include "fast_allocators/internal/next_power_of_2.h" // NextPowerOf2, for slab_of()'s effective slab geometry

/**
 * @brief Must this call hand out plain `new[]` memory instead of slab memory?
 *
 * @details Under STD_ALLOC, for the whole run; never otherwise. A constant: an
 *          allocator's mode cannot change after init(). (A run-time global
 *          was read here until Sept. '26; flipped after init() it made
 *          ElementSize wrong for the other mode -- see below -- and
 *          STD_ALLOC already gives a memory checker individual objects.)
 *
 * @warning The two arms disagree on what ElementSize means: bypass mode counts
 *          ELEMENTS (it feeds `new ElementType[ElementSize]`), slab mode counts
 *          BYTES. init() sets it accordingly.
 */
[[nodiscard]] constexpr bool bypass_slabs() noexcept
{
    return alloc_config::use_std_alloc;
}

/**
 * @brief Map @p size anonymous, zero-filled bytes, or abort.
 * @param size Byte count; the caller is expected to have rounded it to pages.
 * @return The mapping. Never null -- a failure prints errno and aborts, because
 *         every caller here is on a path with no way to report failure upward.
 */
inline void* sctk_mmap(size_t size)
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


/**
 * @brief One slab of raw, fixed-size elements, and a node of a slab list.
 *
 * @details The block is carved linearly while it is fresh (`UsedSize` is a bump
 *          index) and from a free list afterwards. The free list costs no extra
 *          memory: a freed element's first bytes hold the index of the next
 *          free one, with -1 for "none".
 *
 * @warning That makes the element size a lower bound: an element must be at
 *          least `sizeof(int)` wide, and its contents do NOT survive a free.
 *          Neither slab class zero-fills what it hands out: callers
 *          initialise it.
 *
 * @see TCCAllocator for the counterpart that keeps its links out of the payload.
 */
class alignas(64) TCAllocator
{
  public:
    /// @brief Leaves the slab unusable; init() is what makes it a slab.
    TCAllocator() {}

    /// @brief Returns the block to its arena.
    ~TCAllocator()
    {
        destroy();
    }

    // Owns its block: a copy would hand it back to the arena twice.
    TCAllocator(const TCAllocator&)            = delete;
    TCAllocator& operator=(const TCAllocator&) = delete;
    TCAllocator(TCAllocator&&)                 = delete;
    TCAllocator& operator=(TCAllocator&&)      = delete;

    /// @brief The size this slab was cut at -- slab_bytes_for() of the request.
    [[nodiscard]] uintptr_t slab_bytes() const noexcept
    {
        return BlockAllocator;
    }

    /**
     * @brief Set the geometry and obtain the block.
     * @param ElementSize_ Element size in BYTES.
     * @param BlockAllocator_ Requested block size in bytes; rounded up to a
     *        power of two by alloc_block().
     */
    inline void init(int ElementSize_, uintptr_t BlockAllocator_);

    /**
     * @brief Hand out one element.
     * @return Raw, UNINITIALISED memory: either a recycled element (still
     *         holding the previous occupant's bytes, with the free-list link
     *         written over its head) or the next slot of the bump area.
     * @pre The slab is not full -- the caller checks is_full().
     */
    [[nodiscard]] inline void* alloc() noexcept;

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

    /**
     * @brief Take element @p i back.
     * @param i Index from num_block() / num_block_pow2().
     * @details Freeing the element the bump pointer just handed out rewinds the
     *          bump instead of pushing onto the free list, which is what lets an
     *          allocate/free pair leave the slab exactly as it found it.
     */
    inline void free_block(int i) noexcept;

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

    /// @brief Effective slab size for a requested @p block_bytes: alloc_block()
    ///        rounds it up to a power of two. One rule, so the instance can
    ///        resolve a pointer to its slab without a slab to ask.
    [[nodiscard]] static uintptr_t slab_bytes_for(uintptr_t block_bytes) noexcept
    {
        return NextPowerOf2(block_bytes);
    }

    /// @brief The slab of @p slab_bytes (a power of two) holding @p ptr:
    ///        mask down to the slab base, read the back-pointer in its last word.
    [[nodiscard]] static TCAllocator* slab_from(const void* ptr, uintptr_t slab_bytes) noexcept
    {
        const char* base = reinterpret_cast<const char*>(reinterpret_cast<uintptr_t>(ptr) & (~(slab_bytes - 1)));
        return *reinterpret_cast<TCAllocator* const*>(base + slab_bytes - sizeof(void*));
    }

// Declaration-level switch, so it stays a directive: serial builds route slab
// headers through their own pool, OpenMP builds use the global operators.
#ifndef _OPENMP
    void* operator new(size_t size);  ///< Slab headers come from their own pool.
    void  operator delete(void* ptr); ///< Counterpart of the above.
#endif

    // FIX (op/ctx/value slab aliasing): which disjoint arena this slab draws
    // from. Op and ctx pools share the TCAllocator class but MUST NOT share an
    // address space (a freed ctx slab reobtained by the op pool overwrote a live
    // op's vtable with ctx layout). Set by TCTypeAllocator before alloc_block().
    fast_alloc::ArenaKind arena_kind = fast_alloc::ArenaKind::Op;

  private:
    /// @brief Return the block to the arena named by #arena_kind.
    inline void destroy();

    /**
     * @brief Obtain the block and lay it out.
     * @details Rounds the requested size up to a power of two (addr_to_block()
     *          masks with it), derives #BlockSize from it, and writes `this`
     *          into the block's last word.
     */
    inline void alloc_block();

    /**
     * @brief Recover the owning slab from a pointer to any element inside it.
     * @param ptr Interior pointer, previously returned by alloc().
     * @return The slab that owns @p ptr -- whichever allocator it belongs to.
     * @details Masks @p ptr down to the block base, then reads the back-pointer
     *          stored in the block's last word. This is what lets a block be
     *          freed without knowing which allocator handed it out.
     */
    [[nodiscard]] TCAllocator* addr_to_block(const void* ptr) const noexcept
    {
        return slab_from(ptr, BlockAllocator);
    }

    char* data = nullptr; ///< The block itself, aligned on #BlockAllocator.

    uintptr_t BlockAllocator = 0; ///< Block size in bytes; a power of two.

    int Log2ElementSize = 0; ///< log2(#ElementSize), for num_block_pow2().

  public:
    int firstFree{-1}; ///< Head of the free list, an element index; -1 if empty.

    int nn_ref{0}; ///< Elements currently out. 0 == releasable, #BlockSize == full.

    int UsedSize{0}; ///< Bump index: how far the linear carve has reached.

    int ElementSize{0}; ///< Element size in BYTES.

    int BlockSize{0}; ///< Capacity of the block, in elements.

    // for the linked list

    TCAllocator* prev = nullptr; ///< Previous slab in the owner's block/block_full list.
    TCAllocator* next = nullptr; ///< Next slab in the owner's block/block_full list.

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
};

/**
 * @brief One slab of fixed-size elements, and a node of a slab list.
 *
 * @details Same shape as ::TCAllocator, with the free list in the side array
 *          #tab_firstFree instead of inside the elements: the allocator never
 *          writes into a free element, so an emptied slab's pages can be
 *          dropped (madvise_empty_slab_full()) without losing a link.
 *
 * @warning NO content guarantee. An element comes back with whatever its
 *          previous owner left in it -- through the free list, a rewound bump
 *          pointer, a slab revived from the reserve, or a slab the arena
 *          recycles. Only a slab fresh from the arena (mmap) is zero.
 *          Callers initialise what they get: ChunkOmpPolicy::zero_fill_fresh()
 *          for v1, the fill functors of allocate_chunk_internal() for v2/v2A.
 *
 * @note The side array is `unsigned short`, so a slab holds at most
 *       #first_free_init - 1 elements; alloc_block() caps BlockSize there and
 *       leaves the rest of the slab unused (and untouched: see alloc_block()).
 *
 * @note Slabs come from fast_alloc::value_arena(), which is disjoint from the
 *       op and ctx arenas on purpose -- see the note in alloc_block().
 */
class alignas(64) TCCAllocator
{
  public:
    /// @brief Free-list terminator, and the exclusive upper bound on #BlockSize.
    static constexpr short first_free_init = 32767;

    /// @brief Leaves the slab unusable; init() is what makes it a slab.
    TCCAllocator() {}

    /// @brief Returns the block to the value arena.
    ~TCCAllocator()
    {
        destroy();
    }

    // Owns its block and #tab_firstFree: a copy would release both twice.
    TCCAllocator(const TCCAllocator&)            = delete;
    TCCAllocator& operator=(const TCCAllocator&) = delete;
    TCCAllocator(TCCAllocator&&)                 = delete;
    TCCAllocator& operator=(TCCAllocator&&)      = delete;

    /**
     * @brief Set the geometry and obtain the block.
     * @param ElementSize_ Element size in BYTES.
     * @param BlockAllocator_ Requested block size in bytes; rounded up to a
     *        power of two. The capacity, not the size, is capped to fit the
     *        index width (alloc_block()).
     */
    inline void init(int ElementSize_, uintptr_t BlockAllocator_);

    /**
     * @brief Hand out one element.
     * @return An element with unspecified contents (see the class @warning).
     * @pre The slab is not full -- the caller checks is_full().
     */
    inline void* alloc() noexcept;

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
     * @brief Take element @p i back. Its bytes are left as they are.
     * @param i Index from num_block() / num_block_pow2().
     * @details Freeing the element the bump pointer just handed out rewinds the
     *          bump instead of pushing onto the free list.
     */
    inline void free_block(int i) noexcept;

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
    ///        a power of two. ONE rule, shared by alloc_block(), the instance's
    ///        init() and slab_of().
    /// @details Independent of the element size, on purpose. A buffer is freed
    ///          through slab_of() of the allocator the thread's AMBIENT chunk
    ///          size selects (ChunkBase::release_to_owner), which after a
    ///          re-chunk need not be the size that allocated it; that only
    ///          finds the right slab if every allocator of one element type
    ///          cuts its slabs at the same size. This used to shrink a slab
    ///          whose capacity exceeded what the `unsigned short` side array
    ///          indexes -- for small elements (8 and 16 bytes in a 2 MiB budget:
    ///          256 and 512 KiB slabs) the geometry then depended on the element
    ///          size, and slab_of() read a "slab" pointer out of the middle of a
    ///          block (slab_geometry_validation). The capacity is capped in
    ///          alloc_block() instead; the unused tail is never touched.
    /// @param block_bytes   Bytes the slab must hold.
    /// @param element_bytes Unused; kept so that the call sites read alike.
    [[nodiscard]] static uintptr_t slab_bytes_for(uintptr_t block_bytes, [[maybe_unused]] int element_bytes) noexcept
    {
        return NextPowerOf2(block_bytes);
    }

    /// @brief The slab of @p slab_bytes (a power of two) holding @p ptr:
    ///        mask down to the slab base, read the back-pointer in its last word.
    [[nodiscard]] static TCCAllocator* slab_from(const void* ptr, uintptr_t slab_bytes) noexcept
    {
        const char* base = reinterpret_cast<const char*>(reinterpret_cast<uintptr_t>(ptr) & (~(slab_bytes - 1)));
        return *reinterpret_cast<TCCAllocator* const*>(base + slab_bytes - sizeof(void*));
    }

// Declaration-level switch, so it stays a directive. See ::TCAllocator.
#ifndef _OPENMP
    void* operator new(size_t size);  ///< Slab headers come from their own pool.
    void  operator delete(void* ptr); ///< Counterpart of the above.
#endif

    char* data = nullptr; ///< The block itself, aligned on #BlockAllocator.

    unsigned short firstFree{first_free_init}; ///< Head of the free list; #first_free_init if empty.

  private:
    /// @brief Free #tab_firstFree and hand the block back to the value arena.
    inline void destroy();

    /**
     * @brief Obtain the block and lay it out.
     * @details Rounds the block size up to a power of two, clamps #BlockSize so
     *          it fits an `unsigned short` index, allocates #tab_firstFree, then
     *          takes the block from fast_alloc::value_arena(). The block is not
     *          cleared, fresh or recycled: see the class @warning.
     */
    inline void alloc_block();

    /**
     * @brief Recover the owning slab from a pointer to any element inside it.
     * @param ptr Interior pointer, previously returned by alloc().
     * @return The slab that owns @p ptr, whichever allocator handed it out.
     * @see TCAllocator::addr_to_block for the mechanism.
     */
    TCCAllocator* addr_to_block(const void* ptr) const noexcept
    {
        // the block is aligned on BlockAllocator: allows direct access to the block start from any pointer into it

        const char* debutBlock = reinterpret_cast<char*>(reinterpret_cast<uintptr_t>(ptr) & (~(BlockAllocator - 1)));
        // this pointer stored at the end of the block
        return *reinterpret_cast<TCCAllocator**>(const_cast<char*>(debutBlock) + BlockAllocator - sizeof(void*));
    }

    /// @brief Complement of is_full().
    [[nodiscard]] bool is_not_full() const noexcept
    {
        return nn_ref < BlockSize;
    }

    /// @brief Free-list links, one per element. Kept out of the payload, so an
    ///        emptied slab's pages can be dropped without losing them.
    std::vector<unsigned short> tab_firstFree;

  public:
    int       nn_ref{0};          ///< Elements currently out. 0 == releasable.
    int       UsedSize{0};        ///< Bump index: how far the linear carve reached.
    uintptr_t BlockAllocator = 0; ///< Block size in bytes; a power of two.
    int       ElementSize{0};     ///< Element size in BYTES.

  private:
    int Log2ElementSize = 0; ///< log2(#ElementSize), for num_block_pow2().
    int BlockSize{0};        ///< Capacity in elements; < #first_free_init.

  public:
    // for the linked list

    TCCAllocator* prev = nullptr; ///< Previous slab in the owner's block/block_full list.
    TCCAllocator* next = nullptr; ///< Next slab in the owner's block/block_full list.

    /// @brief The allocator instance whose lists this slab lives in. See
    ///        TCAllocator::owner.
    const void* owner = nullptr;
    /// @brief Rank of #owner, or -1. See TCAllocator::owner_rank.
    int owner_rank = -1;
    /// @brief Affinity key of #owner. See TCAllocator::owner_key.
    unsigned owner_key = 0;
    /// @brief Set when madvise_empty_slab_full() dropped this slab's pages
    ///        INCLUDING the trailing back-pointer. alloc() restores it before
    ///        handing out the first block, which is the only moment a free
    ///        through this slab becomes possible again. Plain bool, no atomic:
    ///        an instance's slabs are allocated from by its own rank only, and
    ///        the flag is set at a serial point with no alloc in flight.
    bool backptr_dropped = false;
};

// ---------------------------------------------------------------------------
// Intrusive doubly-linked slab-list primitives, single-sourced. The three slab
// families (TCTypeAllocator / TCTypeCAllocator / TCTypeAllocatorEBC) used to
// carry byte-identical copies of these; they now delegate to these templates so
// a fix lands in one place. Node must expose `Node* next; Node* prev;`.
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
        return true;
    if (node->prev != nullptr)
        return false;
    node = node->next;
    while (node != nullptr)
    {
        if (node->prev == nullptr)
            return false;
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
    DBG_ASSERT(sl_check(list));
    DBG_ASSERT(sl_contains(list, node));

    if (node == list)
    {
        list = node->next;
        if (list != nullptr)
            list->prev = nullptr;
    }
    else
    {
        DBG_ASSERT(node->prev != nullptr);
        if (node->next != nullptr)
            node->next->prev = node->prev;
        node->prev->next = node->next;
    }
    node->prev = nullptr;
    node->next = nullptr;
    DBG_ASSERT(sl_check(list));
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
    DBG_ASSERT(sl_check(list));

    node->prev = nullptr;
    node->next = list;
    if (list != nullptr)
        list->prev = node;
    list = node;

    DBG_ASSERT(sl_check(list));
    DBG_ASSERT(sl_contains(list, node));
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
 * @note This is the single copy for every slab family -- they used to carry
 *       byte-identical versions. The two callables are the only real divergence.
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
        // nb_max_block_empty < 0  ==  never munmap (op/value aliasing fix).
        if (nb_max_block_empty < 0)
        {
            on_kept(slab); // value: MADV_DONTNEED the slab; op/ctx: no-op
            return;
        }
        // Plain comparison: every family now declares nb_block_empty as int,
        // matching nb_max_block_empty. It used to need a static_cast because
        // TCTypeCAllocator declared it unsigned -- the same signedness slip that
        // nb_max_block_empty itself had to be corrected for, so that -1 works.
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
 * @brief Type-erased handle on a ::TCTypeAllocator, for the memory report.
 * @details Every instance registers itself so a total can be taken across all
 *          element types without the reporting code knowing any of them.
 */
class VTCTypeAllocator
{
  public:
    /// @brief Virtual, so the registry may hold any element type.
    virtual ~VTCTypeAllocator() = default;

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
 * @brief Owns every ::TCAllocator slab serving one element size.
 *
 * @details Slabs are kept in three places, and which one a slab is in is a
 *          function of its `nn_ref`: #block has room, #block_full has none, and
 *          #block_empty is a small reserve of emptied slabs kept back so the
 *          next allocation does not have to go to the arena. alloc() and
 *          free_elt() move slabs between them; the moves themselves live in
 *          ::sl_manage_full_empty.
 *
 * @tparam _ElementType Element type. What is handed out is an ARRAY of these --
 *         init() takes a count, and alloc() returns a pointer to the first.
 *
 * @note Every instance enrols itself in the ::VTCTypeAllocator registry, so the
 *       memory report can total all of them without knowing the type.
 *
 * @note This is the serial family. ::TCTypeCAllocator is its OpenMP counterpart
 *       (links outside the payload); ::TCTypeAllocatorEBC is the compile-time
 *       geometry variant.
 *
 * @see bypass_slabs for when none of this machinery runs at all.
 */
template <typename _ElementType>
class TCTypeAllocator final : public VTCTypeAllocator
{
  public:
    using ElementType = _ElementType;

    using TCAllocatorT = TCTypeAllocator<_ElementType>;

    /// @brief Empty and unusable until init(); enrols in the report registry.
    TCTypeAllocator()
    {
        ElementSize = 0;

        add_TCTypeAllocator_inst(this);
    }

    /// @brief Unregisters from the instance list, then frees every slab.
    ~TCTypeAllocator() override
    {
        remove_TCTypeAllocator_inst(this);

        destroy_int();
    }

    // Owns its slabs and is enrolled by address in the report registry.
    TCTypeAllocator(const TCTypeAllocator&)            = delete;
    TCTypeAllocator& operator=(const TCTypeAllocator&) = delete;
    TCTypeAllocator(TCTypeAllocator&&)                 = delete;
    TCTypeAllocator& operator=(TCTypeAllocator&&)      = delete;

    /**
     * @brief Choose which arena this allocator's slabs are drawn from.
     * @param k Op, Ctx or Value.
     * @warning Must be set BEFORE the first slab is taken; it is copied into
     *          each slab at creation and never revisited. The default is Op, so
     *          a pool that holds values has to say so or its slabs will share an
     *          address space with the op pool -- which is the aliasing bug the
     *          arenas were split to prevent.
     */
    void set_arena(fast_alloc::ArenaKind k) noexcept
    {
        arena_kind = k;
    }

    /// @brief The slab holding @p ptr, resolved from this instance's slab
    ///        geometry alone -- works even when this instance owns no slab yet
    ///        (unlike addr_to_block(), which needs a node to call).
    TCAllocator* slab_of(const void* ptr) const noexcept
    {
        // The slabs are cut at the ROUNDED geometry (TCAllocator::init rounds
        // BlockAllocator up to a power of two); #slab_bytes_ is that rounded
        // size, fixed by init() -- masking with the raw budget would land
        // beside the slab on any non-pow2 budget (e.g. the 240 KB floor at 64
        // threads).
        if (slab_bytes_ == 0)
            return nullptr;
        return TCAllocator::slab_from(ptr, slab_bytes_);
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

    /**
     * @brief Hand out room for #ElementSize elements.
     * @return Raw, uninitialised memory.
     * @details Draws from the current slab; failing that, revives one from the
     *          reserve; failing that, creates a new slab. A slab that fills up
     *          moves to #block_full, which is why #block can be null again at
     *          any time and why this has to cope with that.
     */
    [[nodiscard]] ElementType* alloc()
    {
        if (bypass_slabs())
        {
            return new ElementType[ElementSize]; // ElementSize is a COUNT here
        }

        if (block == nullptr) [[unlikely]]
            return alloc_slow();

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
     * @brief Can THIS allocator free @p ptr?
     * @param ptr Pointer previously returned by some allocator's alloc().
     * @return True if @p ptr's slab is one of ours.
     * @details Resolves @p ptr to its slab through the back-pointer, then asks
     *          the slab whether we own it. A block allocated on one thread
     *          and released on another must be routed back with this before
     *          free_elt_pow2(), which is unsafe through a non-owning allocator.
     * @note O(1): the slab records its owner (see TCAllocator::owner).
     */
    [[nodiscard]] bool is_free_able_elt(const ElementType* ptr) const noexcept
    {
        // O(1): the slab names its owner. A slab parked in block_empty has no
        // element out (nn_ref == 0), so a pointer into one is not a live element
        // of this instance -- same answer the list walk gave.
        if (block == nullptr && block_full == nullptr)
            return false;
        return holds_live(get_allocator(ptr));
    }

    /**
     * @brief free_elt() for a power-of-two element size, using a shift.
     * @param ptr Same precondition as free_elt().
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
    void free_elt_pow2(const ElementType* ptr, TCAllocator* slab) noexcept
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
     * @brief Set the geometry. Call once, before any allocation.
     *
     * @param ElementSize_ Number of ElementType per allocation -- a COUNT, not
     *        a byte size. Stored as bytes unless bypassing; see ::bypass_slabs.
     * @param BlockAllocator_ Slab size in bytes, rounded up to a power of two.
     * @param nb_max_block_emptyl Capacity of the emptied-slab reserve.
     *        NEGATIVE means "never release a slab", which also means the reserve
     *        is not allocated at all.
     *
     * @note The first slab IS created here (unless bypassing), although alloc()
     *       would make one anyway (alloc_slow() on a null #block). It costs its
     *       address space and the page of its back-pointer, not the whole slab:
     *       TCAllocator::alloc_block() does not clear it. TCTypeCAllocator::init()
     *       is the one that creates none.
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
            slab_bytes_    = TCAllocator::slab_bytes_for(BlockAllocator_);

            auto* slab       = new TCAllocator;
            slab->arena_kind = arena_kind; // FIX: route slab to op/ctx arena
            // Stamped like every slab alloc_slow() creates. This first one was
            // not: a free of one of its blocks found no owner and fell back to
            // the CALLER's allocator -- in a dynamic build, whichever chunk
            // size the thread last served -- which then unlinked the slab
            // from lists it was never on (sl_remove assertion; silent list
            // corruption without DBG_ASSERT). With the owner set, the free
            // goes home whatever the ambient size (ChunkBase::release_to_owner).
            slab->owner      = this;
            slab->owner_rank = rank_;
            slab->owner_key  = key_;

            slab->init(ElementSize, BlockAllocator_);
            DBG_ASSERT(slab->slab_bytes() == slab_bytes_);

            insert(block, slab);
        }

        // STD_ALLOC has no slabs, so no reserve either.
        if constexpr (alloc_config::use_slabs)
        {
            // FIX (op/value slab aliasing): nb_max_block_empty < 0 means "never
            // munmap" — we never store into block_empty, so don't allocate it (and
            // avoid new TCAllocator*[-1]).
            if (nb_max_block_empty > 0)
            {
                block_empty.assign(static_cast<std::size_t>(nb_max_block_empty), nullptr);
            }
        }

        nb_block_empty = 0;
    }

  private:
    /// @brief Rank (thread) this instance serves; stamped into every slab it
    ///        creates (TCAllocator::owner_rank). -1 = unranked (serial pools).
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
     * @brief Delete every slab, leaving the list heads dangling.
     * @warning Internal: it does not reset #block / #block_full. Call destroy()
     *          unless you are about to overwrite them yourself.
     */
    void destroy_int()
    {
        // Nothing to walk when the slabs were never built: under STD_ALLOC there
        // are no slabs at all, each element was an independent new[] that
        // free_elt() has already deleted.
        if constexpr (alloc_config::use_slabs)
        {
            if (bypass_slabs())
                return;

            TCAllocator* node = block;

            while (node != nullptr)
            {
                TCAllocator* next = node->next;
                delete node;
                node = next;
            }

            node = block_full;

            while (node != nullptr)
            {
                TCAllocator* next = node->next;
                delete node;
                node = next;
            }

            for (int i = 0; i < nb_block_empty; ++i)
            {
                delete block_empty[i];
            }
        }
    }

    /// @brief alloc() with no slab at hand: revive one from the reserve or
    ///        create one. Out of line and cold -- see FAST_ALLOC_COLD.
    FAST_ALLOC_COLD ElementType* alloc_slow()
    {
        if (nb_block_empty)
        {
            auto* slab = block_empty[--nb_block_empty];

            insert(block, slab);

            slab->UsedSize  = 0;
            slab->firstFree = -1;

            return static_cast<ElementType*>(slab->alloc());
        }

        auto* slab       = new TCAllocator;
        slab->arena_kind = arena_kind; // FIX: route slab to op/ctx arena
        slab->owner      = this;
        slab->owner_rank = rank_;
        slab->owner_key  = key_;
        slab->init(ElementSize, BlockAllocator);

        insert(block, slab);

        return static_cast<ElementType*>(slab->alloc());
    }

    /// @brief is_free_able_elt() for a caller that has already resolved the
    ///        slab (slab_of()): no second resolution.
    [[nodiscard]] bool holds_live(const TCAllocator* slab) const noexcept
    {
        return slab != nullptr && slab->owner == this && slab->nn_ref > 0;
    }

    /// @brief Re-file @p slab after a free. @see sl_manage_full_empty
    void manage_full_empty(TCAllocator* slab) noexcept
    {
        sl_manage_full_empty(
            block, block_full, block_empty.data(), nb_block_empty, nb_max_block_empty, slab,
            [](TCAllocator*) noexcept {}, // op/ctx: nothing to reclaim on keep
            [this](TCAllocator* evicted) noexcept { evict_slab(evicted); });
    }

    /// @brief Release an emptied slab the reserve has no room for. Out of
    ///        line and cold: an fprintf and a `delete` on every free path.
    FAST_ALLOC_COLD void evict_slab(TCAllocator* slab) noexcept
    {
        std::fprintf(stderr, "[MUNMAP] TCTypeAllocator delete slab p=%p nb_max_block_empty=%d\n",
                     static_cast<void*>(slab), nb_max_block_empty);
        std::fflush(stderr);
        delete slab;
    }

    /**
     * @brief The slab owning @p ptr.
     * @pre init() has run -- #slab_bytes_ is the instance's slab geometry, so
     *      no slab of ours is needed to resolve @p ptr.
     */
    [[nodiscard]] TCAllocator* get_allocator(const ElementType* ptr) const noexcept
    {
        return TCAllocator::slab_from(ptr, slab_bytes_);
    }

    /**
     * @brief Take back a block obtained from alloc().
     * @param ptr Must come from THIS allocator -- check is_free_able_elt() when
     *        that is not certain. Freeing through a non-owning allocator
     *        corrupts the owner's free list.
     */
    void free_elt(const ElementType* ptr) noexcept
    {
        if (bypass_slabs())
        {
            delete[] ptr;
            return;
        }

        auto* slab = get_allocator(ptr);

        const int i = slab->num_block(ptr);

        slab->free_block(i);

        manage_full_empty(slab);
    }

    /// @copydoc sl_remove
    void remove(TCAllocator*& list, TCAllocator* node) const noexcept
    {
        sl_remove(list, node);
    }

    /// @copydoc sl_insert
    void insert(TCAllocator*& list, TCAllocator* node) const noexcept
    {
        sl_insert(list, node);
    }

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

        const TCAllocator* node = block;

        while (node != nullptr)
        {
            s_UsedSize += static_cast<size_t>(node->UsedSize) * node->ElementSize;
            s_nn_ref += static_cast<size_t>(node->nn_ref) * node->ElementSize;
            s_blocks += node->BlockSize;

            node = node->next;
        }

        node = block_full;

        while (node != nullptr)
        {
            s_blocks += node->BlockSize;

            node = node->next;
        }

        sum_UsedSize += s_UsedSize;
        sum_nn_ref += s_nn_ref;
        alloc_sz += s_blocks;
    }

    // FIX (op/ctx/value slab aliasing): which arena the slabs created by this
    // allocator draw from. Propagated to every TCAllocator it news up. Default Op;
    // the ctx pool sets Ctx via set_arena() so op and ctx never share an address.
    fast_alloc::ArenaKind arena_kind = fast_alloc::ArenaKind::Op;

    size_t    ElementSize;         ///< Bytes per allocation; a COUNT when bypassing.
    uintptr_t BlockAllocator = 0;  ///< Slab size in bytes, passed on to each slab.
    uintptr_t slab_bytes_    = 0;  ///< The ROUNDED slab size every slab of ours is cut at; 0 until init().
    int       rank_          = -1; ///< Rank served; stamped into slabs as owner_rank.
    unsigned  key_           = 0;  ///< Affinity key served; stamped into slabs as owner_key.

    TCAllocator* block_full{nullptr}; ///< Slabs with no room left.
    TCAllocator* block{nullptr};      ///< Slabs with room. Null until the first alloc().

    /// @brief Reserve capacity; negative = never release a slab. @see init
    int nb_max_block_empty = 0;

    std::vector<TCAllocator*> block_empty;       ///< The reserve. Empty when the capacity is <= 0.
    int                       nb_block_empty{0}; ///< How much of the reserve is occupied.
};

// Declaration-level switch, so it stays a directive: this type exists only in
// OpenMP builds. Serial builds use ::TCTypeAllocator throughout.
#ifdef _OPENMP

/**
 * @brief Owns every ::TCCAllocator slab serving one element size.
 *
 * @details The OpenMP counterpart of ::TCTypeAllocator, with the same three-way
 *          split of slabs between #block, #block_full and #block_empty. Two
 *          differences: its slabs keep their free list outside the payload (so
 *          they can be madvised), and it does not enrol in the
 *          ::VTCTypeAllocator report registry. Like it, it gives no guarantee
 *          on the contents of what it hands out.
 *
 * @tparam _ElementType Element type; an allocation is an ARRAY of these.
 *
 * @note One instance per (element size, thread). Nothing here is synchronised:
 *       the design is that a thread only ever touches its own instance.
 *
 * @warning Which is what makes freeing delicate. A block can be released by a
 *          thread other than the one that allocated it, and its owning slab
 *          then belongs to another instance. Route it with is_free_able_elt()
 *          before free_elt_pow2(); freeing through a non-owning instance
 *          corrupts the owner's free list.
 */
template <typename _ElementType>
class TCTypeCAllocator
{
  public:
    using ElementType = _ElementType;

    /// @brief Empty and unusable until init().
    TCTypeCAllocator()
    {
        ElementSize = 0;
    }

    /// @brief Frees every slab.
    ~TCTypeCAllocator()
    {
        destroy_int();
    }

    // Owns its slabs, which also name it as their owner (TCCAllocator::owner).
    TCTypeCAllocator(const TCTypeCAllocator&)            = delete;
    TCTypeCAllocator& operator=(const TCTypeCAllocator&) = delete;
    TCTypeCAllocator(TCTypeCAllocator&&)                 = delete;
    TCTypeCAllocator& operator=(TCTypeCAllocator&&)      = delete;

    /**
     * @brief Hand out room for #ElementSize elements.
     * @return Memory with unspecified contents; the caller initialises it (see
     *         TCCAllocator's @warning).
     * @details Draws from the current slab; failing that, revives one from the
     *          reserve; failing that, creates a new slab.
     */
    [[nodiscard]] ElementType* alloc()
    {
        if constexpr (alloc_config::use_std_alloc)
        {
            auto* elements = new ElementType[ElementSize];
            memset(elements, 0, ElementSize * sizeof(_ElementType));
            return elements;
        }

        if (block == nullptr) [[unlikely]]
            return alloc_slow();

        auto* current_block = block;

        auto* ptr = static_cast<ElementType*>(current_block->alloc());

        if (current_block->is_full()) [[unlikely]]
        {
            remove(block, current_block);
            insert(block_full, current_block);
        }

        return ptr;
    }

    /// @brief is_free_able_elt() for a caller that has already resolved the
    ///        slab (slab_of()): no second resolution.
    [[nodiscard]] bool holds_live(const TCCAllocator* slab) const noexcept
    {
        return slab != nullptr && slab->owner == this && slab->nn_ref > 0;
    }

    /// @brief Give back the pages of every slab of this instance that currently
    ///        holds no block, and report how many were advised.
    /// @details manage_full_empty() only looks at a slab at the instant one of
    ///          its blocks is freed, so a slab that ends up empty without that
    ///          transition being seen stays resident for good. Measured after a
    ///          re-chunk at 64 threads: the abandoned (key, size) set held
    ///          23 362 slabs, 0 live blocks, 23 362 empty -- and not one
    ///          madvise. This is the sweep that closes that gap; call it at the
    ///          points where a whole set is known to have been released (the
    ///          re-chunk drain, an array's teardown), not on the hot path.
    std::size_t reclaim_empty_slabs_now() noexcept
    {
        if constexpr (alloc_config::use_std_alloc || !alloc_config::reclaim_empty_slabs)
            return 0;
        else
        {
            // CHUNK_RECLAIM_MIN_SLABS: a sweep costs one madvise per slab and
            // returns that slab's pages, so what decides whether it is worth
            // doing is how many slabs this set empties at once. Count first,
            // sweep only if the set clears the bar. The count walks the same
            // two lists the sweep would, which is cheap next to the syscalls
            // it may save.
            const std::size_t floor_slabs = alloc_config::reclaim_min_empty_slabs();
            if (floor_slabs > 0)
            {
                std::size_t empty = 0;
                for (const auto* node : {block, block_full})
                    for (const auto* p = node; p != nullptr; p = p->next)
                        if (p->nn_ref == 0)
                            ++empty;
                if (empty < floor_slabs)
                    return 0;
            }
            std::size_t reclaimed = 0;
            for (auto* node : {block, block_full})
                for (auto* p = node; p != nullptr; p = p->next)
                    // Whole slab, trailing page included: this sweep is the one
                    // context that can afford it (see madvise_empty_slab_full).
                    // The last page is 6% of a 1 MiB slab and was the entire
                    // unexplained residue of a re-chunk.
                    if (p->nn_ref == 0 && fast_alloc::madvise_empty_slab_full(p))
                        ++reclaimed;
            return reclaimed;
        }
    }

    /// @brief Set rank(); before the first slab is created.
    void set_rank(int rank) noexcept
    {
        rank_ = rank;
    }

    /// @brief Set key(); before the first slab is created.
    void set_key(unsigned k) noexcept
    {
        key_ = k;
    }
    /// @brief The slab holding @p ptr from this instance's geometry alone; null
    ///        before init(). Unlike get_allocator() it needs no slab of its own.
    TCCAllocator* slab_of(const void* ptr) const noexcept
    {
        // Slab geometry is the ROUNDED one (TCCAllocator::alloc_block: a power
        // of two, whatever the element size -- see slab_bytes_for()); #slab_bytes_
        // is that size, fixed by init() through the same rule.
        if (slab_bytes_ == 0)
            return nullptr;
        return TCCAllocator::slab_from(ptr, slab_bytes_);
    }

    /**
     * @brief free_elt() for a power-of-two element size, using a shift.
     * @param ptr Must belong to THIS instance. This is the hot free path, and
     *        it does not check -- see is_free_able_elt().
     * @pre #ElementSize is a power of two.
     */
    void free_elt_pow2(const ElementType* ptr) noexcept
    {
        if constexpr (alloc_config::use_std_alloc)
        {
            delete[] ptr;
            return;
        }
        else
        {
            free_elt_pow2(ptr, get_allocator(ptr));
        }
    }

    /// @brief free_elt_pow2() for a caller that has already resolved @p slab
    ///        (slab_of()): the slab is not looked up again.
    /// @pre @p slab is @p ptr's slab and belongs to this instance; slab mode.
    void free_elt_pow2(const ElementType* ptr, TCCAllocator* slab) noexcept
    {
        // See TCTypeAllocator::free_elt_pow2: the shift only when
        // sizeof(ElementType) is a power of two, the division otherwise; the
        // element count must be a power of two (checked in debug).
        DBG_ASSERT(!std::has_single_bit(sizeof(ElementType)) ||
                   std::has_single_bit(static_cast<unsigned int>(slab->ElementSize)));
        const int i = std::has_single_bit(sizeof(ElementType)) ? slab->num_block_pow2(ptr) : slab->num_block(ptr);

        slab->free_block(i);

        manage_full_empty(slab);
    }

    /**
     * @brief Set the geometry. Call once, before any allocation.
     *
     * @param ElementSize_ Number of ElementType per allocation -- a COUNT.
     *        Stored as a byte size unless built for STD_ALLOC.
     * @param BlockAllocator_ Slab size in bytes, rounded up to a power of two.
     * @param nb_max_block_emptyl Capacity of the emptied-slab reserve.
     *        NEGATIVE means "never release a slab".
     *
     * @note No slab is created here; the first alloc() makes one. Priming every
     *       (thread, geometry) pair eagerly cost one slab per pair that never
     *       allocated, and the value arena never takes a slab back.
     */
    void init(size_t ElementSize_, uintptr_t BlockAllocator_, int nb_max_block_emptyl = 5)
    {
        nb_max_block_empty = nb_max_block_emptyl;

        if constexpr (alloc_config::use_std_alloc)
        {
            // new[] takes an element count.
            ElementSize = ElementSize_;
        }
        else
        {
            // A slab is carved in bytes.
            ElementSize = ElementSize_ * sizeof(ElementType);

            BlockAllocator = BlockAllocator_;
            slab_bytes_    = TCCAllocator::slab_bytes_for(BlockAllocator_, static_cast<int>(ElementSize));

            // No slab here. alloc() already creates one when `block` is null -- it
            // has to, since a slab that fills up is moved to block_full and leaves
            // `block` empty again -- so building one eagerly only means paying for
            // an allocator that may never be asked for anything.
            //
            // It usually is never asked: init_allocator_thr() and
            // register_chunk_size() prime EVERY thread for EVERY registered
            // geometry, while a serial write loop touches rank 0 alone. Measured on
            // 1e6 doubles with 4 threads, each priming grabbed a 4 MB slab from the
            // value arena, which never gives one back (deallocs = 0).
            //
            // is_free_able_elt() and addAllocatedSize() already guard on `block`,
            // so a not-yet-used allocator reads as what it is: empty.

            // FIX (op/value slab aliasing): nb_max_block_empty < 0 = never munmap;
            // don't allocate block_empty (and avoid new[...-1] / unsigned overflow).
            if (nb_max_block_empty > 0)
            {
                block_empty.assign(static_cast<std::size_t>(nb_max_block_empty), nullptr);
            }
        }

        nb_block_empty = 0;
    }

    TCCAllocator* block_full{nullptr}; ///< Slabs with no room left.
    TCCAllocator* block{nullptr};      ///< Slabs with room. Null until the first alloc().

  private:
    /**
     * @brief Delete every slab, leaving the list heads dangling.
     * @warning Internal: it does not reset #block / #block_full. Call destroy()
     *          unless you are about to overwrite them yourself.
     */
    void destroy_int()
    {
        // STD_ALLOC builds never carve a slab, so there is no list to walk.
        if constexpr (alloc_config::use_slabs)
        {
            auto* node = block;

            while (node != nullptr)
            {
                auto* next = node->next;
                delete node;
                node = next;
            }

            node = block_full;

            while (node != nullptr)
            {
                auto* next = node->next;
                delete node;
                node = next;
            }

            for (int i = 0; i < nb_block_empty; ++i)
            {
                delete block_empty[i];
            }
        }
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

    /// @brief alloc() with no slab at hand: revive one from the reserve or
    ///        create one. Out of line and cold -- see FAST_ALLOC_COLD.
    FAST_ALLOC_COLD ElementType* alloc_slow()
    {
        if (nb_block_empty)
        {
            auto* slab = block_empty[--nb_block_empty];

            insert(block, slab);

            slab->UsedSize  = 0;
            slab->firstFree = TCCAllocator::first_free_init;

            return static_cast<ElementType*>(slab->alloc());
        }

        auto* slab       = new TCCAllocator;
        slab->owner      = this;
        slab->owner_rank = rank_;
        slab->owner_key  = key_;
        slab->init(ElementSize, BlockAllocator);
        DBG_ASSERT(slab->BlockAllocator == slab_bytes_);

        insert(block, slab);

        return static_cast<ElementType*>(slab->alloc());
    }

    /**
     * @brief Can THIS instance free @p ptr?
     * @param ptr Pointer previously returned by some instance's alloc().
     * @return True if @p ptr's slab is one of ours.
     * @details The routing primitive for cross-thread frees: resolve @p ptr to
     *          its slab through the back-pointer, then ask the slab whether we
     *          own it and it holds a live block.
     * @note O(1): the slab records its owner (see TCCAllocator::owner).
     */
    bool is_free_able_elt(const ElementType* ptr) const noexcept
    {
        // O(1) via the slab's owner field -- see TCAllocator::owner. A slab in
        // block_empty has nn_ref == 0 and cannot hold a live element.
        if (block == nullptr && block_full == nullptr)
            return false;
        return holds_live(get_allocator(ptr));
    }

    /// @brief Re-file @p slab after a free. @see sl_manage_full_empty
    void manage_full_empty(TCCAllocator* slab) noexcept
    {
        sl_manage_full_empty(
            block, block_full, block_empty.data(), nb_block_empty, nb_max_block_empty, slab,
            [](TCCAllocator* kept) noexcept
            {
                // NO madvise here, whatever reclaim_empty_slabs says.
                //
                // This lambda runs on the hot path, on every free that brings a
                // slab to nn_ref == 0, and nn_ref is its ONLY evidence that the
                // slab holds nothing live. That evidence is exactly what a
                // mis-accounted block (the same buffer parked or released twice,
                // or two threads touching one slab's counter) falsifies -- and
                // MADV_DONTNEED turns the falsehood into destroyed data: the
                // live block's page is dropped and reads back as zeros.
                // Observed: shrink_bench chunk2a dyn, 32 threads, tab_ulong
                // chunk 35342 materialised, non-constant, its whole buffer read
                // as 0 against the reference file. Before this flag the same
                // accounting error left the slab mapped and intact.
                //
                // Pages are given back only by reclaim_empty_slabs_now(), which
                // is called on a set the caller KNOWS abandoned by construction
                // (the (key, size) set a re-chunked array just left, an array's
                // teardown) -- a slab with nn_ref == 0 there is empty because no
                // block of that set can be live any more, not because a counter
                // says so. The emptied-slab case is unchanged otherwise: the slab
                // stays in `block`, still allocatable.
                (void)kept;
            },
            [this](TCCAllocator* evicted) noexcept { evict_slab(evicted); });
    }

    /// @brief Release an emptied slab the reserve has no room for. Out of
    ///        line and cold: an fprintf and a `delete` on every free path.
    FAST_ALLOC_COLD void evict_slab(TCCAllocator* slab) noexcept
    {
        std::fprintf(stderr, "[MUNMAP] TCCAllocator delete slab p=%p nb_max_block_empty=%d\n", static_cast<void*>(slab),
                     nb_max_block_empty);
        std::fflush(stderr);
        delete slab;
    }

    /**
     * @brief The slab owning @p ptr.
     * @pre init() has run -- #slab_bytes_ is the instance's slab geometry, so
     *      no slab of ours is needed to resolve @p ptr.
     */
    TCCAllocator* get_allocator(const ElementType* ptr) const noexcept
    {
        return TCCAllocator::slab_from(ptr, slab_bytes_);
    }

    /// @brief Rank this instance serves; stamped into its slabs
    ///        (TCCAllocator::owner_rank). -1 = unranked.
    [[nodiscard]] int rank() const noexcept
    {
        return rank_;
    }

    /// @brief Affinity key this instance serves; stamped as owner_key.
    [[nodiscard]] unsigned key() const noexcept
    {
        return key_;
    }

    /**
     * @brief Take back a block obtained from alloc().
     * @param ptr Must come from THIS allocator -- check is_free_able_elt() when
     *        that is not certain. Freeing through a non-owning allocator
     *        corrupts the owner's free list.
     */
    void free_elt(const ElementType* ptr) noexcept
    {
        if (bypass_slabs())
        {
            delete[] ptr;
            return;
        }

        auto* slab = get_allocator(ptr);

        const int i = slab->num_block(ptr);

        slab->free_block(i);

        manage_full_empty(slab);
    }

    /// @copydoc sl_remove
    void remove(TCCAllocator*& list, TCCAllocator* node) noexcept
    {
        sl_remove(list, node);
    }

    /// @copydoc sl_insert
    void insert(TCCAllocator*& list, TCCAllocator* node) noexcept
    {
        sl_insert(list, node);
    }

    size_t    ElementSize;         ///< Bytes per allocation; a COUNT under STD_ALLOC.
    uintptr_t BlockAllocator = 0;  ///< Slab size in bytes, passed on to each slab.
    uintptr_t slab_bytes_    = 0;  ///< The ROUNDED slab size every slab of ours is cut at; 0 until init().
    int       rank_          = -1; ///< Rank served; stamped into slabs as owner_rank.
    unsigned  key_           = 0;  ///< Affinity key served; stamped into slabs as owner_key.

    /// @brief Reserve capacity; negative = never release a slab. @see init
    int nb_max_block_empty = 0; // FIX: was unsigned; int so -1 (never-munmap) works

    std::vector<TCCAllocator*> block_empty;       ///< The reserve. Empty when the capacity is <= 0.
    int                        nb_block_empty{0}; ///< How much of the reserve is occupied.
};

#endif


/**
 * @brief Smallest power of two >= @p i, at compile time.
 * @tparam i Non-negative.
 * @note <tt>NextPOT\<0\>\::value</tt> and <tt>NextPOT\<1\>\::value</tt> are both 1.
 */
template <int i>
class NextPOT
{
  public:
    enum
    {
        value = 2 * NextPOT<(i + 1) / 2>::value
    };
};
template <>
class NextPOT<0>
{
  public:
    enum
    {
        value = 1
    };
};
template <>
class NextPOT<1>
{
  public:
    enum
    {
        value = 1
    };
};

/**
 * @brief Largest power of two <= @p i, at compile time.
 * @tparam i Non-negative.
 * @note <tt>PrevPOT\<0\>\::value</tt> is 1, not 0.
 */
template <int i>
class PrevPOT
{
  public:
    enum
    {
        value = 2 * PrevPOT<i / 2>::value
    };
};
template <>
class PrevPOT<0>
{
  public:
    enum
    {
        value = 1
    };
};
template <>
class PrevPOT<1>
{
  public:
    enum
    {
        value = 1
    };
};

/**
 * @brief floor(log2(@p i)) + 1, at compile time: the bit width of @p i.
 * @tparam i Strictly positive -- there is no specialisation for 0, so it
 *         recurses without end.
 * @note The "+1" is why callers subtract one to get a shift count.
 */
template <int i>
class CstLog2
{
  public:
    enum
    {
        value = 1 + CstLog2<i / 2>::value
    };
};

template <>
class CstLog2<1>
{
  public:
    enum
    {
        value = 1
    };
};

/**
 * @brief Narrowest signed integer that fits in @p sz bytes.
 * @tparam sz Byte width: 1, 2, or anything else for `int`.
 * @details Used to size a slab's free-list index to the slab itself, rather
 *          than spend an `int` per element on a small block.
 */
template <int sz>
class TypeFromSize
{
  public:
    using Tindex = int;
};

template <>
class TypeFromSize<2>
{
  public:
    using Tindex = short;
};

template <>
class TypeFromSize<1>
{
  public:
    using Tindex = signed char;
};


/// @brief Maps an element type to the allocator that serves it. Specialised elsewhere.
template <typename V>
class TypeAllocator;

/// @brief Page size, cached from the environment at startup.
extern int __mem_page_size;

/**
 * @brief One slab whose element and block sizes are template parameters.
 *
 * @details Same job as ::TCAllocator, with the geometry known at compile time:
 *          the capacity, the shift used for indexing and the free-list index
 *          width all become constants, so the division in num_block() and the
 *          per-slab geometry fields go away.
 *
 * @tparam _ElementSize Element size in bytes.
 * @tparam _BlockSize Block size in bytes.
 *
 * @note Used only through ::TCTypeAllocatorEBC, which owns the slab lists.
 */
template <unsigned int _ElementSize, unsigned int _BlockSize>
class TCAllocatorEB
{
  public:
    enum
    {
        ElementSize      = _ElementSize,
        Log2ElementSize  = CstLog2<ElementSize>::value - 1,
        ElementSizePrev2 = PrevPOT<_ElementSize>::value,
        BlockSize        = _BlockSize,
        BlockSizePOT     = NextPOT<BlockSize>::value,
        BlockAddrMask    = ~(BlockSizePOT - 1),
        getSizeOf        = sizeof(TCAllocatorEB*),
        UsedBlockSize    = _BlockSize - getSizeOf,
        MaxElements      = UsedBlockSize / _ElementSize
    };

    using T = TypeFromSize<ElementSizePrev2>::Tindex;

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
    static_assert(static_cast<uintptr_t>(_BlockSize) < (uintptr_t{1} << ((sizeof(T) * 8) - 1)),
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
        if (!__mem_page_size)
        {
            __mem_page_size = getpagesize();
        }

        if (((2 * BlockSize) % __mem_page_size == 0) && __mem_page_size)
        {
            use_mmap = true;

            const uintptr_t block_bytes = BlockSize;

            char* mapping = static_cast<char*>(sctk_mmap(2 * BlockSize));

            data = reinterpret_cast<char*>((reinterpret_cast<uintptr_t>(mapping) + (block_bytes - 1)) & -block_bytes);

            const uintptr_t lng = reinterpret_cast<uintptr_t>(data) - reinterpret_cast<uintptr_t>(mapping);
            if (lng != 0 && munmap(mapping, lng) == -1)
            {
                std::cerr << "munmap (head trim) failed: " << std::strerror(errno) << " -- surplus left mapped\n";
            }
            if (munmap(data + BlockSize, (2 * BlockSize) - (lng + BlockSize)) == -1)
            {
                std::cerr << "munmap (tail trim) failed: " << std::strerror(errno) << " -- surplus left mapped\n";
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

/// @brief Common base for the compile-time-geometry allocators. Carries nothing;
///        it exists so they can be held behind one pointer type.
class VTCAllocatorEB
{
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
 * @warning Unlike the other two families, the constructor takes a slab straight
 *          away rather than waiting for the first alloc().
 */
template <unsigned int ElementSize, unsigned int BlockSize, typename T = int>
class TCTypeAllocatorEBC : public VTCAllocatorEB
{
  public:
    /// @brief The slab type this instance owns.
    using TCAllocatorEBT = TCAllocatorEB<ElementSize, BlockSize>;

    /// @brief Creates the first slab immediately; there is no init() to call.
    TCTypeAllocatorEBC() : block_full(nullptr), block(nullptr)
    {
        auto* slab = new TCAllocatorEBT;

        insert(block, slab);
    }

    /// @brief Deletes every slab, the reserve included.
    ~TCTypeAllocatorEBC()
    {
        auto* node = block;

        while (node != nullptr)
        {
            auto* next = node->next;
            delete node;
            node = next;
        }

        node = block_full;

        while (node != nullptr)
        {
            auto* next = node->next;
            delete node;
            node = next;
        }

        for (int i = 0; i < nb_block_empty; ++i)
        {
            delete block_empty[i];
        }
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
            if (nb_block_empty)
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
    TCAllocatorEBT* block;      ///< Slabs with room. Never null: the ctor seeds one.

    int                                             nb_block_empty{0}; ///< Reserve occupancy.
    std::array<TCAllocatorEBT*, nb_max_block_empty> block_empty{};     ///< The reserve, inline.
};

// SIZE_BLOCK: default slab size (1 MiB). constexpr rather than #define so it
// carries a type and scope; used as a non-type template argument below and
// in StaticConstArray, where an unsigned int constant expression is required.
inline constexpr unsigned int SIZE_BLOCK = 1u << 20u; // 1048576

/**
 * @brief ::TCTypeAllocatorEBC addressed by element TYPE rather than by size.
 *
 * @tparam _ElementType Element type; its `sizeof` becomes the element size.
 * @tparam _BlockSize Block size in bytes.
 *
 * @note Nothing in the tree uses this -- ::TCTypeAllocatorEBC is instantiated
 *       directly, by size, from StaticConstArray. Kept as the typed front end.
 */
template <typename _ElementType, unsigned int _BlockSize = SIZE_BLOCK>
class TCTypeAllocatorEB : public TCTypeAllocatorEBC<sizeof(_ElementType), _BlockSize>
{
  public:
    using ElementType = _ElementType;

    enum
    {
        ElementSize = sizeof(ElementType),
        BlockSize   = _BlockSize
    };

  private:
    /// @brief One element's worth of raw memory, typed.
    static ElementType* alloc()
    {
        return static_cast<ElementType*>(TCTypeAllocatorEBC<ElementSize, BlockSize>::alloc());
    }

    /// @brief Take back a pointer from alloc().
    static void free(ElementType* ptr)
    {
        TCTypeAllocatorEBC<ElementSize, BlockSize>::free_elt(static_cast<void*>(ptr));
    }
};

#include "fast_allocators/internal/tc_allocator_inline.h"
