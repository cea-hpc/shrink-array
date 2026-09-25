
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
 * next free index into the element's first bytes. ::TCCAllocator cannot: it
 * guarantees the memory it hands out is zeroed, so it keeps the links in a side
 * array (TCCAllocator::tab_firstFree) and leaves the payload untouched. That
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

#include <bit>     // std::has_single_bit: the shift or the division in free_elt_pow2
#include <cerrno>  // errno, after a failed mmap
#include <cstddef> // std::size_t
#include <cstdio>  // std::fprintf / std::fflush (the [MUNMAP] traces)
#include <cstring>
#include <iostream>
#include <cstdlib>
#include <cstdint>
#include <cassert>
#include <sys/mman.h>
#include <unistd.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "fast_allocators/internal/alloc_config.h"    // constexpr mirrors of STD_ALLOC / DEBUG / _OPENMP
#include "fast_allocators/slab_arena.h"               // FIX: ArenaKind for op/ctx/value slab routing
#include "fast_allocators/internal/next_power_of_2.h" // NextPowerOf2, for slab_of()'s effective slab geometry

template <typename T>
class Singleton;

extern bool __debug_mem;

/**
 * @brief Must this call hand out plain `new[]` memory instead of slab memory?
 *
 * @details True for the whole run under STD_ALLOC; otherwise it is whatever
 *          `__debug_mem` says, and only where that flag is read at all
 *          (alloc_config::debug_mem_observable). A release serial build folds
 *          this to a compile-time false, so the slab path keeps its
 *          branch-free shape.
 *
 * @warning The two arms disagree on what ElementSize means: bypass mode counts
 *          ELEMENTS (it feeds `new ElementType[ElementSize]`), slab mode counts
 *          BYTES. init() sets it accordingly; do not mix them.
 *
 * @see alloc_config::debug_mem_observable for why `__debug_mem` is currently
 *      unusable.
 */
[[nodiscard]] inline bool bypass_slabs() noexcept
{
    if constexpr (alloc_config::use_std_alloc)
        return true;
    else if constexpr (alloc_config::debug_mem_observable)
        return __debug_mem;
    else
        return false;
}

/**
 * @brief Map @p size anonymous, zero-filled bytes, or abort.
 * @param size Byte count; the caller is expected to have rounded it to pages.
 * @return The mapping. Never null -- a failure prints errno and aborts, because
 *         every caller here is on a path with no way to report failure upward.
 */
inline void* sctk_mmap(size_t size)
{
    void* tmp;
#if defined(BSD)
    tmp = mmap(nullptr, (size), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
#else
    // MAP_ANONYMOUS: not backed by any file; the contents are zero (mmap(2)).
    tmp = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
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
 *          Use ::TCCAllocator when the memory must come back zeroed.
 *
 * @see TCCAllocator for the zero-initialising counterpart.
 */
class alignas(64) TCAllocator
{
  public:
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

    /// @brief Leaves the slab unusable; init() is what makes it a slab.
    TCAllocator() : firstFree(-1), nn_ref(0), UsedSize(0), ElementSize(0), BlockSize(0) {}

    // FIX (op/ctx/value slab aliasing): which disjoint arena this slab draws
    // from. Op and ctx pools share the TCAllocator class but MUST NOT share an
    // address space (a freed ctx slab reobtained by the op pool overwrote a live
    // op's vtable with ctx layout). Set by TCTypeAllocator before alloc_block().
    fast_alloc::ArenaKind arena_kind = fast_alloc::ArenaKind::Op;

    /// @brief Returns the block to its arena.
    ~TCAllocator()
    {
        destroy();
    }

// Declaration-level switch, so it stays a directive: serial builds route slab
// headers through their own pool, OpenMP builds use the global operators.
#ifndef _OPENMP
    void* operator new(size_t s);   ///< Slab headers come from their own pool.
    void  operator delete(void* p); ///< Counterpart of the above.
#endif

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

    // [[nodiscard]] bool is_free() const noexcept
    // { // is there room left?
    //     return (firstFree != -1) || (UsedSize < BlockSize);
    // }

    /// @brief Index of the element at @p p within this slab (division).
    [[nodiscard]] int num_block(const void* p) const noexcept
    {
        return static_cast<int>((static_cast<const char*>(p) - data) / ElementSize);
    }

    /// @brief num_block() by shift. Valid only if #ElementSize is a power of two.
    [[nodiscard]] int num_block_pow2(const void* p) const noexcept
    {
        return static_cast<int>((static_cast<const char*>(p) - data) >> Log2ElementSize);
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

    //    [[nodiscard]] bool is_not_full() const noexcept {
    //        return nn_ref < BlockSize;
    //    }

  private:
    char* data; ///< The block itself, aligned on #BlockAllocator.

    uintptr_t BlockAllocator; ///< Block size in bytes; a power of two.

    int Log2ElementSize; ///< log2(#ElementSize), for num_block_pow2().

  public:
    int firstFree; ///< Head of the free list, an element index; -1 if empty.

    int nn_ref; ///< Elements currently out. 0 == releasable, #BlockSize == full.

    int UsedSize; ///< Bump index: how far the linear carve has reached.

    int ElementSize; ///< Element size in BYTES.

    int BlockSize; ///< Capacity of the block, in elements.

    // for the linked list

    TCAllocator* prev; ///< Previous slab in the owner's block/block_full list.
    TCAllocator* next; ///< Next slab in the owner's block/block_full list.

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
 * @brief One slab of ZEROED fixed-size elements, and a node of a slab list.
 *
 * @details Same shape as ::TCAllocator, one guarantee more: every element it
 *          hands out is all-zero, recycled ones included. That is why the free
 *          list lives in the side array #tab_firstFree instead of inside the
 *          elements -- writing a link into the payload is exactly what would
 *          break the guarantee. alloc_block() zeroes the whole block once, and
 *          nothing ever writes into a free element afterwards.
 *
 * @note The side array is `unsigned short`, so a slab holds at most
 *       #first_free_init - 1 elements; alloc_block() caps BlockSize there and
 *       leaves the rest of the slab unused (and untouched: see alloc_block()).
 *
 * @note Slabs come from fast_alloc::value_arena(), which is disjoint from the
 *       op and ctx arenas on purpose -- see the note in alloc_block().
 */
struct alignas(64) TCCAllocator
{
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

    /// @brief Leaves the slab unusable; init() is what makes it a slab.
    TCCAllocator() : firstFree(first_free_init), nn_ref(0), UsedSize(0), ElementSize(0), BlockSize(0) {}

    /// @brief Returns the block to the value arena.
    ~TCCAllocator()
    {
        destroy();
    }

// Declaration-level switch, so it stays a directive. See ::TCAllocator.
#ifndef _OPENMP
    void* operator new(size_t s);   ///< Slab headers come from their own pool.
    void  operator delete(void* p); ///< Counterpart of the above.
#endif

    /// @brief Free #tab_firstFree and hand the block back to the value arena.
    inline void destroy();

    /**
     * @brief Obtain the block, zero it, and lay it out.
     * @details Rounds the block size up to a power of two, clamps #BlockSize so
     *          it fits an `unsigned short` index, allocates #tab_firstFree, then
     *          takes the block from fast_alloc::value_arena() and memsets it --
     *          a recycled block still holds the previous occupant's bytes, and
     *          this pool promises zeros.
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

    /**
     * @brief Set the geometry and obtain the block.
     * @param ElementSize_ Element size in BYTES.
     * @param BlockAllocator_ Requested block size in bytes; rounded up to a
     *        power of two, then possibly shrunk to fit the index width.
     */
    inline void init(int ElementSize_, uintptr_t BlockAllocator_);

    /**
     * @brief Hand out one element.
     * @return Memory that is all-zero, recycled or not.
     * @pre The slab is not full -- the caller checks is_full().
     */
    inline void* alloc() noexcept;

    // bool is_free() const noexcept
    // { // is there room left?
    //     return (firstFree != first_free_init) || (UsedSize < BlockSize);
    // }

    /// @brief Index of the element at @p p within this slab (division).
    int num_block(const void* p) const noexcept
    {
        return static_cast<int>((static_cast<const char*>(p) - data) / ElementSize);
    }

    /// @brief num_block() by shift. Valid only if #ElementSize is a power of two.
    int num_block_pow2(const void* p) const noexcept
    {
        return static_cast<int>((static_cast<const char*>(p) - data) >> Log2ElementSize);
    }

    /**
     * @brief Take element @p i back, leaving its bytes zeroed.
     * @param i Index from num_block() / num_block_pow2().
     * @details Freeing the element the bump pointer just handed out rewinds the
     *          bump instead of pushing onto the free list.
     */
    inline void free_block(int i) noexcept;

    /// @brief No element of this slab is out. Only then may it be released.
    bool is_empty() const
    {
        return nn_ref == 0;
    }

    /// @brief This free brought the slab back from full; it must leave block_full.
    bool was_full() const
    {
        return nn_ref == BlockSize - 1;
    }

    /// @brief Every element is out; the slab must move to block_full.
    bool is_full() const
    {
        return nn_ref == BlockSize;
    }

    /// @brief Complement of is_full().
    bool is_not_full() const
    {
        return nn_ref < BlockSize;
    }

    char* data; ///< The block itself, aligned on #BlockAllocator, zeroed.

    /// @brief Free-list terminator, and the exclusive upper bound on #BlockSize.
    static constexpr short first_free_init = 32767;

    unsigned short firstFree; ///< Head of the free list; #first_free_init if empty.

    /// @brief Free-list links, one per element. Kept out of the payload so a
    ///        recycled element is still all-zero.
    unsigned short* tab_firstFree;

    int       nn_ref;          ///< Elements currently out. 0 == releasable.
    int       UsedSize;        ///< Bump index: how far the linear carve reached.
    uintptr_t BlockAllocator;  ///< Block size in bytes; a power of two.
    int       ElementSize;     ///< Element size in BYTES.
    int       Log2ElementSize; ///< log2(#ElementSize), for num_block_pow2().
    int       BlockSize;       ///< Capacity in elements; < #first_free_init.

    // for the linked list

    TCCAllocator* prev; ///< Previous slab in the owner's block/block_full list.
    TCCAllocator* next; ///< Next slab in the owner's block/block_full list.

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

// #define VOLATILE(x) ((typeof(x) volatile &)(x))

// ---------------------------------------------------------------------------
// Intrusive doubly-linked slab-list primitives, single-sourced. The three slab
// families (TCTypeAllocator / TCTypeCAllocator / TCTypeAllocatorEBC) used to
// carry byte-identical copies of these; they now delegate to these templates so
// a fix lands in one place. Node must expose `Node* next; Node* prev;`.
// NOTE: is_removable() is intentionally NOT here — it diverges per family
// (listContains vs listContainsNoCheck, by-value vs by-ref).
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
 * @param p The slab that was just freed into.
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
                          Node*   p,
                          OnKept  on_kept,
                          OnEvict on_evict) noexcept
{
    if (p->was_full() && block_full) [[unlikely]]
    {
        sl_remove(block_full, p);
        sl_insert(block, p);
    }

    if (p->is_empty()) [[unlikely]]
    {
        // nb_max_block_empty < 0  ==  never munmap (op/value aliasing fix).
        if (nb_max_block_empty < 0)
        {
            on_kept(p); // value: MADV_DONTNEED the slab; op/ctx: no-op
            return;
        }
        // Plain comparison: every family now declares nb_block_empty as int,
        // matching nb_max_block_empty. It used to need a static_cast because
        // TCTypeCAllocator declared it unsigned -- the same signedness slip that
        // nb_max_block_empty itself had to be corrected for, so that -1 works.
        if (nb_block_empty < nb_max_block_empty)
        {
            block_empty[nb_block_empty++] = p;
            sl_remove(block, p);
        }
        else
        {
            sl_remove(block, p);
            on_evict(p); // sync: delete p; async: defer to per-thread free list
        }
    }
}

/**
 * @brief Does @p ptr's slab belong to the pair of lists @p block / @p block_full?
 *
 * @param block Head of the "room left" list; may be null.
 * @param block_full Head of the "no room left" list; may be null.
 * @param ptr An element pointer, resolved to its slab through the back-pointer
 *        stored at the slab's end.
 * @param is_removable Callable `(const Node* list, const Node* node)` deciding
 *        whether @p node is in @p list. It is a parameter because the two slab
 *        families answer it differently -- one tolerates a null head, the other
 *        does not -- which is the ONLY thing that used to differ between two
 *        otherwise character-identical copies of this function.
 * @return True when the slab is ours, so the block may be freed through this
 *         allocator. False means "not mine", never "not allocated".
 *
 * @note Linear in the number of slabs. This is the routing primitive for a
 *       cross-thread free: offer an orphaned block to each rank in turn and
 *       free it through the one that answers true.
 */
template <typename Node, typename Elem, class IsRemovable>
[[nodiscard]] bool
sl_is_free_able_elt(const Node* block, const Node* block_full, const Elem* ptr, IsRemovable is_removable) noexcept
{
    if (block)
    {
        const auto* p = block->addr_to_block(ptr);

        if (!is_removable(block, p))
        {
            if (block_full)
            {
                if (!is_removable(block_full, p))
                {
                    return false;
                }
            }
            else
            {
                return false;
            }
        }

        return true;
    }
    else if (block_full)
    {
        const auto* p = block_full->addr_to_block(ptr);

        return is_removable(block_full, p);
    }

    return false;
}

/**
 * @brief Type-erased handle on a ::TCTypeAllocator, for the memory report.
 * @details Every instance registers itself so a total can be taken across all
 *          element types without the reporting code knowing any of them.
 */
struct VTCTypeAllocator
{
    /// @brief Virtual, so the registry may hold any element type.
    virtual ~VTCTypeAllocator() = default;

    /**
     * @brief Accumulate this allocator's footprint into the running totals.
     * @param sum_UsedSize Elements ever carved, summed over slabs.
     * @param sum_nn_ref Elements currently out, summed over slabs.
     * @param alloc_sz Bytes held, summed over slabs.
     */
    virtual void addAllocatedSize(size_t& sum_UsedSize, size_t& sum_nn_ref, size_t& alloc_sz) = 0;
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
 *       and hands out zeroed memory; ::TCTypeAllocatorEBC is the compile-time
 *       geometry variant.
 *
 * @see bypass_slabs for when none of this machinery runs at all.
 */
template <typename _ElementType>
struct TCTypeAllocator final : public VTCTypeAllocator
{
    using ElementType = _ElementType;

    using TCAllocatorT = TCTypeAllocator<_ElementType>;

    // FIX (op/ctx/value slab aliasing): which arena the slabs created by this
    // allocator draw from. Propagated to every TCAllocator it news up. Default Op;
    // the ctx pool sets Ctx via set_arena() so op and ctx never share an address.
    fast_alloc::ArenaKind arena_kind = fast_alloc::ArenaKind::Op;
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

    /// @brief Rank (thread) this instance serves; stamped into every slab it
    ///        creates (TCAllocator::owner_rank). -1 = unranked (serial pools).
    int rank() const noexcept
    {
        return rank_;
    }
    /// @brief Set rank(); before the first slab is created.
    void set_rank(int r) noexcept
    {
        rank_ = r;
    }
    /// @brief Affinity key this instance serves (0 = shared pool of the rank);
    ///        stamped into its slabs as owner_key.
    unsigned key() const noexcept
    {
        return key_;
    }
    /// @brief Set key(); before the first slab is created.
    void set_key(unsigned k) noexcept
    {
        key_ = k;
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

    /// @brief Empty and unusable until init(); enrols in the report registry.
    TCTypeAllocator() : block_full(nullptr), block(nullptr), nb_block_empty(0)
    {
        block_empty = nullptr;
        ElementSize = 0;

        add_TCTypeAllocator_inst(this);
    }

    ~TCTypeAllocator() override
    {
        remove_TCTypeAllocator_inst(this);

        destroy_int();

        delete[] block_empty;
    }

    /**
     * @brief Delete every slab, leaving the list heads dangling.
     * @warning Internal: it does not reset #block / #block_full. Call destroy()
     *          unless you are about to overwrite them yourself.
     */
    void destroy_int()
    {
        // Nothing to walk when the slabs were never built: under STD_ALLOC there
        // are no slabs at all, and under __debug_mem each element was an
        // independent new[] that free_elt() has already deleted.
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

    /**
     * @brief Release every slab and the reserve array, and reset to fresh state.
     * @note Unlike the destructor, leaves the object usable: init() may follow.
     */
    void destroy()
    {
        destroy_int();

        delete[] block_empty;
        block_empty = nullptr;

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

    /// @brief alloc() with no slab at hand: revive one from the reserve or
    ///        create one. Out of line and cold -- see FAST_ALLOC_COLD.
    FAST_ALLOC_COLD ElementType* alloc_slow()
    {
        if (nb_block_empty)
        {
            auto* p = block_empty[--nb_block_empty];

            insert(block, p);

            p->UsedSize  = 0;
            p->firstFree = -1;

            return static_cast<ElementType*>(p->alloc());
        }

        auto* p       = new TCAllocator;
        p->arena_kind = arena_kind; // FIX: route slab to op/ctx arena
        p->owner      = this;
        p->owner_rank = rank_;
        p->owner_key  = key_;
        p->init(ElementSize, BlockAllocator);

        insert(block, p);

        return static_cast<ElementType*>(p->alloc());
    }

    /**
     * @brief Can THIS allocator free @p ptr?
     * @param ptr Pointer previously returned by some allocator's alloc().
     * @return True if @p ptr's slab is one of ours.
     * @details Resolves @p ptr to its slab through the back-pointer, then looks
     *          for that slab in our own lists. A block allocated on one thread
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

    /// @brief is_free_able_elt() for a caller that has already resolved the
    ///        slab (slab_of()): no second resolution.
    [[nodiscard]] bool holds_live(const TCAllocator* slab) const noexcept
    {
        return slab != nullptr && slab->owner == this && slab->nn_ref > 0;
    }

    /// @brief Re-file @p p after a free. @see sl_manage_full_empty
    void manage_full_empty(TCAllocator* p) noexcept
    {
        sl_manage_full_empty(
            block, block_full, block_empty, nb_block_empty, nb_max_block_empty, p,
            [](TCAllocator*) noexcept {}, // op/ctx: nothing to reclaim on keep
            [this](TCAllocator* q) noexcept { evict_slab(q); });
    }

    /// @brief Release an emptied slab the reserve has no room for. Out of
    ///        line and cold: an fprintf and a `delete` on every free path.
    FAST_ALLOC_COLD void evict_slab(TCAllocator* q) noexcept
    {
        std::fprintf(stderr, "[MUNMAP] TCTypeAllocator delete slab p=%p nb_max_block_empty=%d\n", static_cast<void*>(q),
                     nb_max_block_empty);
        std::fflush(stderr);
        delete q;
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

        auto* p = get_allocator(ptr);

        const int i = p->num_block(ptr);

        p->free_block(i);

        manage_full_empty(p);
    }

    /**
     * @brief free_elt() for a power-of-two element size, using a shift.
     * @param ptr Same precondition as free_elt().
     * @pre #ElementSize is a power of two; otherwise the index is wrong and the
     *      free lands on another element.
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
     * @note No slab is created here. The first alloc() makes one, which it has
     *       to be able to do anyway.
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

            auto* p       = new TCAllocator;
            p->arena_kind = arena_kind; // FIX: route slab to op/ctx arena
            // Stamped like every slab alloc_slow() creates. This first one was
            // not: a free of one of its blocks found no owner and fell back to
            // the CALLER's allocator -- in a dynamic build, whichever chunk
            // size the thread last served -- which then unlinked the slab
            // from lists it was never on (sl_remove assertion; silent list
            // corruption without DBG_ASSERT). With the owner set, the free
            // goes home whatever the ambient size (ChunkBase::release_to_owner).
            p->owner      = this;
            p->owner_rank = rank_;
            p->owner_key  = key_;

            p->init(ElementSize, BlockAllocator_);
            DBG_ASSERT(p->slab_bytes() == slab_bytes_);

            insert(block, p);
        }

        // Compile-time, not bypass_slabs(): a fast build with __debug_mem set
        // still allocates the reserve, because the flag can only be observed
        // once the allocator is already live. STD_ALLOC never has one.
        if constexpr (alloc_config::use_slabs)
        {
            // FIX (op/value slab aliasing): nb_max_block_empty < 0 means "never
            // munmap" — we never store into block_empty, so don't allocate it (and
            // avoid new TCAllocator*[-1]).
            if (nb_max_block_empty > 0)
            {
                block_empty = new TCAllocator*[nb_max_block_empty];

                for (int i = 0; i < nb_max_block_empty; ++i)
                {
                    block_empty[i] = nullptr;
                }
            }
        }

        nb_block_empty = 0;
    }

    /// @copydoc sl_contains
    [[nodiscard]] bool listContains(const TCAllocator* list, const TCAllocator* node) const noexcept
    {
        return sl_contains(list, node);
    }

    /// @brief May @p node be unlinked from @p list? That is, is it in it?
    [[nodiscard]] bool is_removable(const TCAllocator* list, const TCAllocator* node) const noexcept
    {
        return listContains(list, node);
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
    void addAllocatedSize(size_t& sum_UsedSize, size_t& sum_nn_ref, size_t& alloc_sz) noexcept override
    {
        size_t s_blocks   = 0;
        size_t s_UsedSize = 0;
        size_t s_nn_ref   = 0;

        auto* node = block;

        while (node != nullptr)
        {
            s_UsedSize += static_cast<size_t>(node->UsedSize) * node->ElementSize;
            s_nn_ref += static_cast<size_t>(node->nn_ref) * node->ElementSize;
            s_blocks += node->BlockSize;

            auto* next = node->next;
            node       = next;
        }

        node = block_full;

        while (node != nullptr)
        {
            s_blocks += node->BlockSize;

            auto* next = node->next;
            node       = next;
        }

        sum_UsedSize += s_UsedSize;
        sum_nn_ref += s_nn_ref;
        alloc_sz += s_blocks;
    }

    size_t    ElementSize;         ///< Bytes per allocation; a COUNT when bypassing.
    uintptr_t BlockAllocator = 0;  ///< Slab size in bytes, passed on to each slab.
    uintptr_t slab_bytes_    = 0;  ///< The ROUNDED slab size every slab of ours is cut at; 0 until init().
    int       rank_          = -1; ///< Rank served; stamped into slabs as owner_rank.
    unsigned  key_           = 0;  ///< Affinity key served; stamped into slabs as owner_key.

    TCAllocator* block_full; ///< Slabs with no room left.
    TCAllocator* block;      ///< Slabs with room. Null until the first alloc().

    /// @brief Reserve capacity; negative = never release a slab. @see init
    int nb_max_block_empty;

    TCAllocator** block_empty;    ///< The reserve. Null when the capacity is <= 0.
    int           nb_block_empty; ///< How much of the reserve is occupied.
};

// Declaration-level switch, so it stays a directive: this type exists only in
// OpenMP builds. Serial builds use ::TCTypeAllocator throughout.
#ifdef _OPENMP

/**
 * @brief Owns every ::TCCAllocator slab serving one element size.
 *
 * @details The OpenMP counterpart of ::TCTypeAllocator, with the same three-way
 *          split of slabs between #block, #block_full and #block_empty. Two
 *          differences: what it hands out is ZEROED, and it does not enrol in
 *          the ::VTCTypeAllocator report registry.
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
 *
 * @warning It also does NOT honour `__debug_mem` consistently: free_elt() reads
 *          it, alloc() and free_elt_pow2() do not.
 *          @see alloc_config::debug_mem_observable
 */
template <typename _ElementType>
struct TCTypeCAllocator
{
    using ElementType = _ElementType;

    //   using TCAllocatorT = TCTypeCAllocator<_ElementType>;

    /// @brief Empty and unusable until init().
    TCTypeCAllocator() : block_full(nullptr), block(nullptr), nb_block_empty(0)
    {
        block_empty = nullptr;
        ElementSize = 0;
    }

    ~TCTypeCAllocator()
    {
        destroy_int();

        delete[] block_empty;
    }

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

        delete[] block_empty;
        block_empty = nullptr;

        ElementSize = 0;
        block_full = block = nullptr;
        nb_block_empty     = 0;
    }

    /**
     * @brief Hand out room for #ElementSize elements, all bytes zero.
     * @return Zeroed memory.
     * @details Draws from the current slab; failing that, revives one from the
     *          reserve; failing that, creates a new slab.
     */
    [[nodiscard]] ElementType* alloc()
    {
        // if constexpr, NOT bypass_slabs(): unlike free_elt() below, this path
        // has never honoured __debug_mem. Keeping the asymmetry is deliberate --
        // see alloc_config::debug_mem_observable.
        if constexpr (alloc_config::use_std_alloc)
        {
            auto* p = new ElementType[ElementSize];
            memset(p, 0, ElementSize * sizeof(_ElementType));
            return p;
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

    /// @brief alloc() with no slab at hand: revive one from the reserve or
    ///        create one. Out of line and cold -- see FAST_ALLOC_COLD.
    FAST_ALLOC_COLD ElementType* alloc_slow()
    {
        if (nb_block_empty)
        {
            auto* p = block_empty[--nb_block_empty];

            insert(block, p);

            p->UsedSize  = 0;
            p->firstFree = TCCAllocator::first_free_init;

            return static_cast<ElementType*>(p->alloc());
        }

        auto* p       = new TCCAllocator;
        p->owner      = this;
        p->owner_rank = rank_;
        p->owner_key  = key_;
        p->init(ElementSize, BlockAllocator);
        DBG_ASSERT(p->BlockAllocator == slab_bytes_);

        insert(block, p);

        return static_cast<ElementType*>(p->alloc());
    }

    /**
     * @brief Can THIS instance free @p ptr?
     * @param ptr Pointer previously returned by some instance's alloc().
     * @return True if @p ptr's slab is one of ours.
     * @details The routing primitive for cross-thread frees: resolve @p ptr to
     *          its slab through the back-pointer, then look for that slab in our
     *          own lists. Offer an orphan to each rank in turn and free it
     *          through the one that answers true.
     * @note Linear in the number of slabs.
     */
    bool is_free_able_elt(const ElementType* ptr) const noexcept
    {
        // O(1) via the slab's owner field -- see TCAllocator::owner. A slab in
        // block_empty has nn_ref == 0 and cannot hold a live element.
        if (block == nullptr && block_full == nullptr)
            return false;
        return holds_live(get_allocator(ptr));
    }

    /// @brief is_free_able_elt() for a caller that has already resolved the
    ///        slab (slab_of()): no second resolution.
    [[nodiscard]] bool holds_live(const TCCAllocator* slab) const noexcept
    {
        return slab != nullptr && slab->owner == this && slab->nn_ref > 0;
    }

    /// @brief Re-file @p p after a free. @see sl_manage_full_empty
    void manage_full_empty(TCCAllocator* p) noexcept
    {
        sl_manage_full_empty(
            block, block_full, block_empty, nb_block_empty, nb_max_block_empty, p,
            [](TCCAllocator* q) noexcept
            {
                // NO madvise here, whatever CHUNK_RECLAIM_EMPTY_SLABS says.
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
                (void)q;
            },
            [this](TCCAllocator* q) noexcept { evict_slab(q); });
    }

    /// @brief Release an emptied slab the reserve has no room for. Out of
    ///        line and cold: an fprintf and a `delete` on every free path.
    FAST_ALLOC_COLD void evict_slab(TCCAllocator* q) noexcept
    {
        std::fprintf(stderr, "[MUNMAP] TCCAllocator delete slab p=%p nb_max_block_empty=%d\n", static_cast<void*>(q),
                     nb_max_block_empty);
        std::fflush(stderr);
        delete q;
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
        std::size_t n = 0;
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
                for (auto* node : {block, block_full})
                    for (auto* p = node; p != nullptr; p = p->next)
                        if (p->nn_ref == 0)
                            ++empty;
                if (empty < floor_slabs)
                    return 0;
            }
            for (auto* node : {block, block_full})
                for (auto* p = node; p != nullptr; p = p->next)
                    if (p->nn_ref == 0)
                    {
                        // Whole slab, trailing page included: this sweep is
                        // the one context that can afford it (see
                        // madvise_empty_slab_full). The last page is 6% of a
                        // 1 MiB slab and was the entire unexplained residue of
                        // a re-chunk.
                        if (fast_alloc::madvise_empty_slab_full(p))
                            ++n;
                    }
            return n;
        }
    }
    /// @brief Rank this instance serves; stamped into its slabs
    ///        (TCCAllocator::owner_rank). -1 = unranked.
    int rank() const noexcept
    {
        return rank_;
    }
    /// @brief Set rank(); before the first slab is created.
    void set_rank(int r) noexcept
    {
        rank_ = r;
    }
    /// @brief Affinity key this instance serves; stamped as owner_key.
    unsigned key() const noexcept
    {
        return key_;
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
        // Slab geometry is the ROUNDED one (TCCAllocator::alloc_block: power of
        // two, and re-derived when BlockSize would exceed first_free_init);
        // #slab_bytes_ is that size, fixed by init() through the same rule.
        if (slab_bytes_ == 0)
            return nullptr;
        return TCCAllocator::slab_from(ptr, slab_bytes_);
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

        auto* p = get_allocator(ptr);

        const int i = p->num_block(ptr);

        p->free_block(i);

        manage_full_empty(p);
    }

    /**
     * @brief free_elt() for a power-of-two element size, using a shift.
     * @param ptr Must belong to THIS instance. This is the hot free path, and
     *        it does not check -- see is_free_able_elt().
     * @pre #ElementSize is a power of two.
     */
    void free_elt_pow2(const ElementType* ptr) noexcept
    {
        // if constexpr, NOT bypass_slabs(): the pow2 free has never read
        // __debug_mem, and its caller drain_reuse_list_for_size() relies on it
        // pairing with alloc(), which does not read it either.
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
        // sizeof(ElementType) is a power of two, the division otherwise.
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
                block_empty = new TCCAllocator*[nb_max_block_empty];

                for (int i = 0; i < nb_max_block_empty; ++i)
                {
                    block_empty[i] = nullptr;
                }
            }
        }

        nb_block_empty = 0;
    }

    /// @copydoc sl_contains
    bool listContains(const TCCAllocator* list, const TCCAllocator* node) const noexcept
    {
        return sl_contains(list, node);
    }

    /**
     * @brief listContains() that skips the null test on the first node.
     * @warning @p list must NOT be null. Every caller reaches this through
     *          is_removable(), and is_free_able_elt() is what guarantees it.
     */
    bool listContainsNoCheck(const TCCAllocator* list, const TCCAllocator* node) const noexcept
    {
        return (list == node) || listContains(list->next, node);
    }

    /// @brief May @p node be unlinked from @p list? @p list must not be null.
    bool is_removable(const TCCAllocator* list, const TCCAllocator* node) const noexcept
    {
        return listContainsNoCheck(list, node);
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

    TCCAllocator* block_full; ///< Slabs with no room left.
    TCCAllocator* block;      ///< Slabs with room. Null until the first alloc().

    /// @brief Reserve capacity; negative = never release a slab. @see init
    int nb_max_block_empty; // FIX: was unsigned; int so -1 (never-munmap) works

    TCCAllocator** block_empty;    ///< The reserve. Null when the capacity is <= 0.
    int            nb_block_empty; ///< How much of the reserve is occupied.
};

#endif


/**
 * @brief Smallest power of two >= @p i, at compile time.
 * @tparam i Non-negative.
 * @note `NextPOT<0>::value` and `NextPOT<1>::value` are both 1.
 */
template <int i>
struct NextPOT
{
    enum
    {
        value = 2 * NextPOT<(i + 1) / 2>::value
    };
};
template <>
struct NextPOT<0>
{
    enum
    {
        value = 1
    };
};
template <>
struct NextPOT<1>
{
    enum
    {
        value = 1
    };
};

/**
 * @brief Largest power of two <= @p i, at compile time.
 * @tparam i Non-negative.
 * @note `PrevPOT<0>::value` is 1, not 0.
 */
template <int i>
struct PrevPOT
{
    enum
    {
        value = 2 * PrevPOT<i / 2>::value
    };
};
template <>
struct PrevPOT<0>
{
    enum
    {
        value = 1
    };
};
template <>
struct PrevPOT<1>
{
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
struct CstLog2
{
    enum
    {
        value = 1 + CstLog2<i / 2>::value
    };
};

template <>
struct CstLog2<1>
{
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
struct TypeFromSize
{
    using Tindex = int;
};

template <>
struct TypeFromSize<2>
{
    using Tindex = short;
};

template <>
struct TypeFromSize<1>
{
    using Tindex = signed char;
};


/// @brief Maps an element type to the allocator that serves it. Specialised elsewhere.
template <typename V>
struct TypeAllocator;

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
struct TCAllocatorEB
{
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

    /// @brief Takes its block straight away -- there is no separate init().
    TCAllocatorEB() : firstFree(-1), nn_ref(0), UsedSize(0)
    {
#ifdef DEBUG
        const int  szt            = sizeof(T);
        const auto block_size_max = static_cast<uintptr_t>(uintptr_t{1} << ((szt * 8) - 1));
        DBG_ASSERT(_BlockSize < block_size_max);
#endif
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

            const uintptr_t b = BlockSize;

            char* a = static_cast<char*>(sctk_mmap(2 * BlockSize));

            data = reinterpret_cast<char*>((reinterpret_cast<uintptr_t>(a) + (b - 1)) & -b);

            const uintptr_t lng = reinterpret_cast<uintptr_t>(data) - reinterpret_cast<uintptr_t>(a);
            if (lng != 0 && munmap(a, lng) == -1)
            {
                std::cerr << "munmap (head trim) failed: " << std::strerror(errno) << " -- surplus left mapped\n";
            }
            if (munmap(data + BlockSize, 2 * BlockSize - (lng + BlockSize)) == -1)
            {
                std::cerr << "munmap (tail trim) failed: " << std::strerror(errno) << " -- surplus left mapped\n";
            }
        }
        else
        {
            use_mmap = false;
            void* datal;

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

    /**
     * @brief Recover the owning slab from a pointer to any element inside it.
     * @param ptr Interior pointer, previously returned by alloc().
     * @note Static, unlike the other families' version: the mask and the offset
     *       are both compile-time constants here, so no instance is needed.
     */
    static TCAllocatorEB* addr_to_block(const void* ptr)
    {
        // char * debutBlock = (char*)( (uintptr_t)ptr & size_t(BlockAddrMask) );
        // this pointer stored at the end of the block
        return *reinterpret_cast<TCAllocatorEB**>(
            reinterpret_cast<char*>(reinterpret_cast<uintptr_t>(ptr) & size_t(BlockAddrMask)) + UsedBlockSize);
    }

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
        char* elemPtr;

        if (firstFree != -1)
        {
            elemPtr   = data + firstFree * ElementSize;
            firstFree = *(T*)elemPtr; // firstFree=getIndex(firstFree);
        }
        else
        {
            elemPtr = data + UsedSize * ElementSize;
            ++UsedSize;
        }
        return static_cast<void*>(elemPtr);
    }

    // bool is_free() const noexcept
    // { // is there room left?
    //     return (firstFree != -1) || (UsedSize < MaxElements);
    // }

    /// @brief Index of the element at @p p within this slab (division).
    int num_block(const void* p) const noexcept
    {
        return static_cast<int>((static_cast<const char*>(p) - data) / ElementSize);
    }

    /// @brief num_block() by shift. Valid only if #ElementSize is a power of two.
    int num_block_pow2(const void* p) const noexcept
    {
        return static_cast<int>((static_cast<const char*>(p) - data) >> Log2ElementSize);
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
            *(T*)(data + i * ElementSize) = -1;
            --UsedSize;
        }
        else
        {
            *(T*)(data + i * ElementSize) = firstFree;
            firstFree                     = i;
        }
        --nn_ref;
    }

    /// @brief No element of this slab is out. Only then may it be deleted.
    bool is_empty() const
    {
        return nn_ref == 0;
    }

    /// @brief This free brought the slab back from full.
    bool was_full() const
    {
        return nn_ref == MaxElements - 1;
    }

    /// @brief Every element is out; the slab must move to block_full.
    bool is_full() const
    {
        return nn_ref == MaxElements;
    }

    char* data;      ///< The block itself, aligned on #BlockSizePOT.
    int   firstFree; ///< Head of the free list, an element index; -1 if empty.

    int nn_ref;   ///< Elements currently out. 0 == deletable.
    int UsedSize; ///< Bump index: how far the linear carve has reached.

    // for the linked list

    TCAllocatorEB* prev;
    TCAllocatorEB* next;

    bool use_mmap;
};

/// @brief Common base for the compile-time-geometry allocators. Carries nothing;
///        it exists so they can be held behind one pointer type.
struct VTCAllocatorEB
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
struct TCTypeAllocatorEBC : public VTCAllocatorEB
{
    /// @brief The slab type this instance owns.
    using TCAllocatorEBT = TCAllocatorEB<ElementSize, BlockSize>;

    /// @brief Creates the first slab immediately; there is no init() to call.
    TCTypeAllocatorEBC() : block_full(nullptr), block(nullptr), nb_block_empty(0)
    {
        auto* p = new TCAllocatorEBT;

        insert(block, p);
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
                auto* p = block_empty[--nb_block_empty];

                insert(block, p);

                p->UsedSize  = 0;
                p->firstFree = -1;

                return p->alloc();
            }

            auto* p = new TCAllocatorEBT;

            insert(block, p);

            return p->alloc();
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
     * @brief Re-file @p p after a free, EBC's own way.
     *
     * @details Deliberately NOT ::sl_manage_full_empty. Three differences, kept
     *          so the concurrent cst_val2 path behaves exactly as it did:
     *          a slab that comes back from full returns immediately instead of
     *          also being tested for emptiness; there is no never-release branch
     *          and no MADV_DONTNEED, the reserve being a fixed static array; and
     *          a slab past the reserve is always deleted outright.
     */
    // EBC parking policy — deliberately DIFFERENT from sl_manage_full_empty():
    //   * early-return after was_full migration (no fall-through to is_empty),
    //   * no never-munmap (< 0) branch and no MADV_DONTNEED — EBC has a fixed
    //     static reserve and always deletes past it,
    //   * terminal action is always delete (no deferred free).
    // Kept as its own method (shared by free_elt / free_elt_pow2) rather than
    // routed through the general skeleton, to preserve behaviour exactly on the
    // concurrent cst_val2 path.
    void manage_full_empty(TCAllocatorEBT* p) noexcept
    {
        if (p->was_full())
        {
            if (block_full)
            {
                remove(block_full, p);
                insert(block, p);
            }
            return;
        }

        if (p->is_empty())
        {
            if (nb_block_empty < nb_max_block_empty)
            {
                block_empty[nb_block_empty++] = p;
                remove(block, p);
            }
            else
            {
                remove(block, p);
                delete p;
            }
        }
    }

    /**
     * @brief Take back a pointer from alloc().
     * @param ptr Resolved to its slab through the back-pointer, so it need not
     *        have come from this instance.
     */
    void free_elt(const void* ptr)
    {
        auto* p = TCAllocatorEBT::addr_to_block(ptr);

        const int i = p->num_block(ptr);

        p->free_block(i);

        manage_full_empty(p);
    }

    /**
     * @brief free_elt() using a shift instead of a division.
     * @pre The element size is a power of two.
     */
    void free_elt_pow2(const void* ptr) noexcept
    {
        auto* p = TCAllocatorEBT::addr_to_block(ptr);

        const int i = p->num_block_pow2(ptr);

        p->free_block(i);

        manage_full_empty(p);
    }

    /// @copydoc sl_contains
    bool listContains(TCAllocatorEBT* list, TCAllocatorEBT* node) const noexcept
    {
        return sl_contains(list, node);
    }

    /// @brief May @p node be unlinked from @p list? That is, is it in it?
    bool is_removable(TCAllocatorEBT*& list, TCAllocatorEBT* node) const noexcept
    {
        return listContains(list, node);
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

    /**
     * @brief Accumulate this allocator's footprint into the running totals.
     * @param sum_UsedSize Bytes ever carved, summed over slabs.
     * @param sum_nn_ref Bytes currently out, summed over slabs.
     * @param alloc_sz Bytes held, summed over slabs.
     * @note Unlike TCTypeAllocator::addAllocatedSize, these are BYTES: the
     *       compile-time ElementSize and BlockSize are folded in here.
     */
    void addAllocatedSize(size_t& sum_UsedSize, size_t& sum_nn_ref, size_t& alloc_sz) noexcept
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

    /// @brief Reserve capacity. Fixed, unlike the other families' runtime value.
    static constexpr int nb_max_block_empty = 5;

    TCAllocatorEBT* block_full; ///< Slabs with no room left.
    TCAllocatorEBT* block;      ///< Slabs with room. Never null: the ctor seeds one.

    int             nb_block_empty;                  ///< Reserve occupancy.
    TCAllocatorEBT* block_empty[nb_max_block_empty]; ///< The reserve, inline.
};

// SIZE_BLOCK: default slab size (1 MiB). constexpr rather than #define so it
// carries a type and scope; used as a non-type template argument below and
// in StaticConstArray, where an unsigned int constant expression is required.
inline constexpr unsigned int SIZE_BLOCK = 1u << 20; // 1048576

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
struct TCTypeAllocatorEB : public TCTypeAllocatorEBC<sizeof(_ElementType), _BlockSize>
{
    using ElementType = _ElementType;

    enum
    {
        ElementSize = sizeof(ElementType),
        BlockSize   = _BlockSize
    };

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
