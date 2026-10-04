#pragma once

// Bodies of the slab classes declared in tc_allocator.h, which includes this
// file at its end.

#include <cmath>    // std::log2, the element-size shift
#include <cstddef>  // std::ptrdiff_t, the element offsets
#include <cstdint>  // std::uintptr_t
#include <cstdlib>  // std::abort on an arena failure
#include <iostream> // std::cerr on an arena failure
#include <unistd.h> // getpagesize
#include <vector>   // std::vector

#include "fast_allocators/tc_allocator.h"             // the declarations these bodies belong to
#include "fast_allocators/slab_arena.h"               // FIX: disjoint op/value slab arenas
#include "fast_allocators/internal/next_power_of_2.h" // NextPowerOf2

template <FreeLinks Links>
inline void TCSlab<Links>::alloc_block()
{
    // The rounding rule lives in slab_bytes_for(): the owning pool applies the
    // same one at init(), so it can resolve a pointer to its slab without a
    // slab to ask (TCSlabPool::slab_bytes_).
    BlockAllocator = slab_bytes_for(BlockAllocator);

    BlockSize = static_cast<int>((BlockAllocator - sizeof(void*)) / ElementSize);

    if constexpr (side_links)
    {
        if (BlockSize > free_list_end - 1)
        {
            BlockSize = free_list_end - 1;
        }

        tab_firstFree.resize(static_cast<std::size_t>(BlockSize));
    }

    if (__mem_page_size == 0)
    {
        __mem_page_size = getpagesize();
    }

    // FIX (op/value slab aliasing): the block comes from the slab's own arena,
    // never the glibc. Arena addresses are never returned to the OS and the
    // op, ctx and value arenas are disjoint, so a freed op slab's address can
    // never be reobtained by the value pool (the proven crash path: OP_ALLOC
    // then VALUE_ALLOC at the same address on different tids). Recycled within
    // that arena's own free-list in destroy().
    data = static_cast<char*>(fast_alloc::arena_for(arena_kind).allocate(BlockAllocator, BlockAllocator));

    if (data == nullptr) [[unlikely]]
    {
        std::cerr << "Fatal Error: slab arena (kind " << static_cast<int>(arena_kind) << ") failed to allocate "
                  << BlockAllocator << " bytes\n";
        std::abort();
    }

    // NOTE: no memset here, fresh slab or recycled. The pool makes no promise
    // about element contents (TCSlab's @warning): every consumer initialises
    // what it gets. A fresh arena slab is already zero (MAP_ANONYMOUS), and a
    // recycled one must NOT be blanket-zeroed -- that wiped a block the op pool
    // still considered live (an op read back as all-zero, vptr=0) -- and for
    // values it only ever made the WHOLE slab resident at once: 2 MiB at the
    // first alloc of a rank for a fresh slab, and for a recycled one right
    // after the re-chunk or teardown that gave it back. Left alone, its pages
    // become resident as the elements are handed out.

    *reinterpret_cast<TCSlab**>(data + BlockAllocator - sizeof(void*)) = this;

    firstFree = free_list_end;
}

template <FreeLinks Links>
inline void TCSlab<Links>::destroy()
{
    if constexpr (side_links)
    {
        std::vector<unsigned short>().swap(tab_firstFree); // releases the free-list links
        // (not `= {}`: that is operator=(initializer_list), which keeps the capacity)
    }

    // FIX (op/value slab aliasing): back to the slab's own arena's free-list,
    // never to the glibc: the address stays inside that arena.
    fast_alloc::arena_for(arena_kind).deallocate(static_cast<void*>(data), BlockAllocator);
}

template <FreeLinks Links>
inline void TCSlab<Links>::init(int ElementSize_, uintptr_t BlockAllocator_)
{
    if constexpr (!side_links)
    {
        // The free list is threaded THROUGH the freed elements: free_block()
        // writes the next free index, an int, into a freed element's first
        // bytes. An element must hold that int, at an int-aligned offset --
        // otherwise the write spills into the next element, or is misaligned.
        // ElementSize is a run-time value (a chunk size times sizeof(T)), so
        // this cannot be a static_assert; it runs once per slab and stays on in
        // release.
        if (ElementSize_ < static_cast<int>(sizeof(int)) || ElementSize_ % static_cast<int>(alignof(int)) != 0)
            [[unlikely]]
        {
            std::cerr << "[TCAllocator] element size " << ElementSize_
                      << " bytes cannot hold the free-list index (an int: " << sizeof(int) << " bytes, aligned on "
                      << alignof(int) << ")\n";
            std::abort();
        }
    }

    ElementSize    = ElementSize_;
    BlockAllocator = BlockAllocator_;

    Log2ElementSize = static_cast<int>(std::log2(static_cast<double>(ElementSize)) + 1.e-13);

    alloc_block();
}

template <FreeLinks Links>
inline void* TCSlab<Links>::alloc() noexcept
{
    if constexpr (side_links)
    {
        // A sweep may have given this slab's pages back INCLUDING the trailing
        // back-pointer (madvise_empty_slab_full). Put it back before the block
        // escapes: the pointer we are about to return is exactly what makes a
        // free through this slab possible again, and free_elt() resolves it by
        // reading that word. The store is idempotent -- every writer writes
        // `this` -- and this runs on the slab's own rank, so no atomic is
        // needed.
        if (backptr_dropped) [[unlikely]]
        {
            *reinterpret_cast<TCSlab**>(data + BlockAllocator - sizeof(void*)) = this;
            backptr_dropped                                                    = false;
        }
    }

    ++nn_ref;
    char* elemPtr = nullptr;

    if (firstFree != free_list_end)
    {
        elemPtr   = data + (static_cast<std::ptrdiff_t>(firstFree) * ElementSize);
        firstFree = link(firstFree); // the link is read before the element is handed out
    }
    else
    {
        elemPtr = data + (static_cast<std::ptrdiff_t>(UsedSize) * ElementSize);
        ++UsedSize;
    }

    return elemPtr;
}

template <FreeLinks Links>
inline void TCSlab<Links>::free_block(int i) noexcept
{
    if (i == (UsedSize - 1))
    {
        link(i) = free_list_end;
        --UsedSize;
    }
    else
    {
        link(i)   = firstFree;
        firstFree = static_cast<index_type>(i);
    }

    --nn_ref;
}
