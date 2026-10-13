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

// Bodies of the slab classes declared in tc_allocator.h, which includes this
// file at its end.

#include <bit>      // std::bit_width, the element-size shift
#include <cstddef>  // std::ptrdiff_t, the element offsets
#include <cstdint>  // std::uintptr_t
#include <cstdlib>  // std::abort on an arena failure
#include <iostream> // std::cerr on an arena failure
#include <utility>
#include <vector>   // std::vector

#include "fast_allocators/tc_allocator.h" // the declarations these bodies belong to
#include "fast_allocators/slab_arena.h"   // disjoint op/ctx/value slab arenas

namespace fast_alloc
{

template <FreeLinks Links>
inline void TCSlab<Links>::alloc_block()
{
    // The rounding rule lives in slab_bytes_for(): the owning pool applies the
    // same one at init(), so it can resolve a pointer to its slab without a
    // slab to ask (TCSlabPool::slab_bytes_).
    BlockAllocator = slab_bytes_for(BlockAllocator);

    BlockSize = static_cast<int>((BlockAllocator - sizeof(void*)) / ElementSize);

    // Not even one element fits next to the back-pointer: the first one would
    // overwrite it. Nothing to round to -- refuse, with the sizes.
    if (BlockSize < 1) [[unlikely]]
    {
        std::cerr << "[TCAllocator] element of " << ElementSize << " bytes does not fit in a slab of " << BlockAllocator
                  << " bytes (back-pointer included)\n";
        std::abort();
    }

    if constexpr (side_links)
    {
        if (BlockSize > free_list_end - 1)
        {
            BlockSize = free_list_end - 1;
        }

        tab_firstFree.resize(static_cast<std::size_t>(BlockSize));
    }

    // From the slab's own arena, never glibc: arena addresses are never returned to the OS
    // and the op, ctx and value arenas are disjoint, so a freed op slab's address can never
    // be reobtained by the value pool. owner_rank (set before init()): a slab the rank
    // recycled comes back to it first (SlabArena::allocate).
    data = static_cast<char*>(fast_alloc::arena_for(arena_kind).allocate(BlockAllocator, BlockAllocator, owner_rank));

    if (data == nullptr) [[unlikely]]
    {
        std::cerr << "Fatal Error: slab arena (kind " << static_cast<int>(arena_kind) << ") failed to allocate "
                  << BlockAllocator << " bytes\n";
        std::abort();
    }

    // NOTE: no memset, fresh slab or recycled. The pool makes no promise about element
    // contents (TCSlab's @warning). Do NOT blanket-zero a recycled slab: it can wipe a block
    // the op pool still considers live, and it makes the whole slab resident at once instead
    // of page by page as elements are handed out.

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

    // Back to the slab's own arena's free-list, never to glibc: the address
    // stays inside that arena.
    fast_alloc::arena_for(arena_kind).deallocate(static_cast<void*>(data), BlockAllocator);
}

template <FreeLinks Links>
inline void TCSlab<Links>::init(int ElementSize_, uintptr_t BlockAllocator_)
{
    if constexpr (!side_links)
    {
        // The free list is threaded THROUGH freed elements (free_block() writes the next
        // index, an int, into the element's first bytes): an element must hold an aligned int.
        // ElementSize is a run-time value, so this is a release check, once per slab.
        if (std::cmp_less(ElementSize_, sizeof(int)) || ElementSize_ % static_cast<int>(alignof(int)) != 0) [[unlikely]]
        {
            std::cerr << "[TCAllocator] element size " << ElementSize_
                      << " bytes cannot hold the free-list index (an int: " << sizeof(int) << " bytes, aligned on "
                      << alignof(int) << ")\n";
            std::abort();
        }
    }

    ElementSize    = ElementSize_;
    BlockAllocator = BlockAllocator_;

    Log2ElementSize = std::bit_width(static_cast<unsigned int>(ElementSize)) - 1;

    // num_block()'s magic is exact for offsets below 2^32 only. Slabs are a
    // few MiB; a release check, once per slab, like the one above.
    if (slab_bytes_for(BlockAllocator_) > (std::uintptr_t{1} << 32)) [[unlikely]]
    {
        std::cerr << "[TCAllocator] slab of " << slab_bytes_for(BlockAllocator_)
                  << " bytes: num_block() indexes offsets below 4 GiB only\n";
        std::abort();
    }
    // Unused (and 0, the +1 wrapping) for an element size of 1: a power of
    // two, num_block_pow2() takes it.
    ElementMagic = UINT64_MAX / static_cast<std::uint64_t>(ElementSize) + 1u;

    alloc_block();
}

template <FreeLinks Links>
inline void* TCSlab<Links>::alloc() noexcept
{
    if constexpr (side_links)
    {
        // A sweep may have given this slab's pages back INCLUDING the trailing back-pointer
        // (fast_alloc::EmptySlabSweep). Restore it before the block escapes: free_elt()
        // resolves the slab by reading that word. Idempotent store (always `this`), on the
        // slab's own rank: no atomic needed.
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

} // namespace fast_alloc
