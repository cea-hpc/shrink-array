
#include <cmath>                        // std::log2, the element-size shift
#include <cstddef>                      // std::ptrdiff_t, the element offsets
#include <cstdlib>                      // std::abort on an arena failure
#include <iostream>                     // std::cerr on an arena failure
#include <cstring>                      // std::memset, zeroing a recycled value slab
#include "fast_allocators/slab_arena.h" // FIX: disjoint op/value slab arenas

#include "fast_allocators/internal/next_power_of_2.h"

inline void TCAllocator::alloc_block()
{
    BlockAllocator = NextPowerOf2(BlockAllocator);

    BlockSize = static_cast<int>((BlockAllocator - sizeof(void*)) / ElementSize);

    if (__mem_page_size == 0)
    {
        __mem_page_size = getpagesize();
    }

    // FIX (op/value slab aliasing): allocate the op slab from the dedicated OP
    // arena, never the glibc. Arena addresses are never returned to the OS and are
    // disjoint from the value arena, so a freed op slab's address can never be
    // reobtained by the value pool (the proven crash path: OP_ALLOC then
    // VALUE_ALLOC at the same address on different tids). Recycled within the op
    // arena's own free-list in destroy().
    data = static_cast<char*>(fast_alloc::arena_for(arena_kind).allocate(BlockAllocator, BlockAllocator));

    if (data == nullptr) [[unlikely]]
    {
        std::cerr << "Fatal Error: op_arena failed to allocate " << BlockAllocator << " bytes\n";
        std::abort();
    }

    // NOTE: no slab-wide memset here. A fresh arena slab is already zero
    // (MAP_ANONYMOUS), and a recycled slab must NOT be blanket-zeroed — doing so
    // wipes any block the pool still considers live (seen: an entire op read back
    // as all-zero, vptr=0, after a slab was reused). Each op is fully initialized
    // by construct_at at its own block, so slab-wide zeroing is unnecessary.

    *reinterpret_cast<TCAllocator**>(data + BlockAllocator - sizeof(void*)) = this;

    firstFree = -1;
}

inline void* TCAllocator::alloc() noexcept
{
    ++nn_ref;
    char* elemPtr = nullptr;

    if (firstFree != -1)
    {
        elemPtr   = data + (static_cast<std::ptrdiff_t>(firstFree) * ElementSize);
        firstFree = *reinterpret_cast<int*>(elemPtr); // read the index stored in the first element of the array
    }
    else
    {
        elemPtr = data + (static_cast<std::ptrdiff_t>(UsedSize) * ElementSize);
        ++UsedSize;
    }
    return elemPtr;
}

inline void TCAllocator::destroy()
{
    // FIX (op/value slab aliasing): return the op slab to the OP arena's
    // free-list, NEVER to the glibc. This is what breaks the crash: the address
    // stays inside the op arena and can never be handed to the value pool.
    fast_alloc::arena_for(arena_kind).deallocate(static_cast<void*>(data), BlockAllocator);
}

inline void TCAllocator::init(int ElementSize_, uintptr_t BlockAllocator_)
{
    ElementSize    = ElementSize_;
    BlockAllocator = BlockAllocator_;

    Log2ElementSize = static_cast<int>(std::log2(static_cast<double>(ElementSize)) + 1.e-13);

    alloc_block();
}

inline void TCAllocator::free_block(int i) noexcept
{
    if (i == (UsedSize - 1))
    {
        *reinterpret_cast<int*>(data + (static_cast<std::ptrdiff_t>(i) * ElementSize)) = -1;
        --UsedSize;
    }
    else
    {
        // store the index in the first element of the block
        *reinterpret_cast<int*>(data + (static_cast<std::ptrdiff_t>(i) * ElementSize)) = firstFree;
        firstFree                                                                      = i;
    }
    --nn_ref;
}

inline void TCCAllocator::alloc_block()
{
    // The rounding rule lives in slab_bytes_for(): the owning instance applies
    // the same one at init(), so it can resolve a pointer to its slab without
    // a slab to ask (TCTypeCAllocator::slab_bytes_).
    BlockAllocator = slab_bytes_for(BlockAllocator, ElementSize);

    BlockSize = static_cast<int>((BlockAllocator - sizeof(void*)) / ElementSize);

    if (BlockSize > first_free_init - 1)
        BlockSize = first_free_init - 1;

    tab_firstFree.resize(static_cast<std::size_t>(BlockSize));

    // FIX (op/value slab aliasing): allocate the value slab from the dedicated
    // VALUE arena (disjoint from the op arena, never returned to the OS). Value
    // slabs live in their own address space and can never land on an op slab's
    // address. Recycled within the value arena's own free-list in destroy().
    data = static_cast<char*>(fast_alloc::value_arena().allocate(BlockAllocator, BlockAllocator));

    if (data == nullptr) [[unlikely]]
    {
        std::cerr << "Fatal Error: value_arena failed to allocate " << BlockAllocator << " bytes\n";
        std::abort();
    }

    // Arena slabs are fresh-mmap zeroed on first use, but recycled slabs may hold
    // stale bytes; the pool relies on zero-initialized elements, so clear them --
    // the BlockSize elements only: when BlockSize is capped by the side array,
    // the tail up to the back-pointer is never handed out, and leaving it
    // untouched keeps it out of the resident set (slab_bytes_for()).
    std::memset(data, 0, static_cast<std::size_t>(BlockSize) * static_cast<std::size_t>(ElementSize));

    *reinterpret_cast<TCCAllocator**>(data + BlockAllocator - sizeof(void*)) = this;

    firstFree = first_free_init;
}

inline void TCCAllocator::destroy()
{
    std::vector<unsigned short>().swap(tab_firstFree); // releases the free-list links
    // (not `= {}`: that is operator=(initializer_list), which keeps the capacity)

    // FIX (op/value slab aliasing): return the value slab to the VALUE arena's
    // free-list, never to the glibc.
    fast_alloc::value_arena().deallocate(static_cast<void*>(data), BlockAllocator);
}

inline void TCCAllocator::init(int ElementSize_, uintptr_t BlockAllocator_)
{
    ElementSize    = ElementSize_;
    BlockAllocator = BlockAllocator_;

    Log2ElementSize = static_cast<int>(std::log2(static_cast<double>(ElementSize)) + 1.e-13);

    alloc_block();
}

inline void* TCCAllocator::alloc() noexcept
{
    // A sweep may have given this slab's pages back INCLUDING the trailing
    // back-pointer (madvise_empty_slab_full). Put it back before the block
    // escapes: the pointer we are about to return is exactly what makes a free
    // through this slab possible again, and free_elt() resolves it by reading
    // that word. The store is idempotent -- every writer writes `this` -- and
    // this runs on the slab's own rank, so no atomic is needed.
    if (backptr_dropped) [[unlikely]]
    {
        *reinterpret_cast<TCCAllocator**>(data + BlockAllocator - sizeof(void*)) = this;
        backptr_dropped                                                          = false;
    }

    ++nn_ref;
    char* elemPtr = nullptr;

    if (firstFree != first_free_init)
    {
        elemPtr   = data + (static_cast<std::ptrdiff_t>(firstFree) * ElementSize);
        firstFree = tab_firstFree[firstFree];
    }
    else
    {
        elemPtr = data + (static_cast<std::ptrdiff_t>(UsedSize) * ElementSize);
        ++UsedSize;
    }

    return elemPtr;
}

inline void TCCAllocator::free_block(int i) noexcept
{
    if (i == (UsedSize - 1))
    {
        tab_firstFree[i] = first_free_init;
        --UsedSize;
    }
    else
    {
        tab_firstFree[i] = firstFree;
        firstFree        = i;
    }

    --nn_ref;
}
