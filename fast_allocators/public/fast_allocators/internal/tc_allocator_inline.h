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

#include <bit>     // std::bit_width, the element-size shift
#include <cstddef> // std::ptrdiff_t, the element offsets; std::byte (the blocks)
#include <cstdint> // std::uintptr_t
#include <utility> // std::cmp_less (the element-size check)
#include <vector>  // std::vector

#include "fast_allocators/internal/raw_memory.h" // store_to_bytes
#include "fast_allocators/tc_allocator.h"        // the declarations these bodies belong to
#include "fast_allocators/slab_arena.h"          // disjoint op/ctx/value slab arenas
#include "fast_allocators/internal/fatal.h"      // FAST_ALLOC_FATAL

namespace fast_alloc
{

template <FreeLinks Links>
inline void TCSlab<Links>::alloc_block()
{
    // The rounding rule lives in slab_bytes_for(): the owning pool applies the
    // same one at init(), so it can resolve a pointer to its slab without a
    // slab to ask (TCSlabPool::slab_bytes_).
    block_allocator = slab_bytes_for(block_allocator);

    block_size = static_cast<int>((block_allocator - sizeof(void*)) / element_size);

    // Not even one element fits next to the back-pointer: the first one would
    // overwrite it. Nothing to round to -- refuse, with the sizes.
    if (block_size < 1) [[unlikely]]
    {
        FAST_ALLOC_FATAL("TCAllocator: element of {} bytes does not fit in a slab of {} bytes (back-pointer included)",
                         element_size, block_allocator);
    }

    if constexpr (side_links)
    {
        if (block_size > free_list_end - 1)
        {
            block_size = free_list_end - 1;
        }

        tab_first_free_.resize(static_cast<std::size_t>(block_size));
    }

    // From the slab's own arena, never glibc: arena addresses are never returned to the OS
    // and the op, ctx and value arenas are disjoint, so a freed op slab's address can never
    // be reobtained by the value pool. owner_rank (set before init()): a slab the rank
    // recycled comes back to it first (SlabArena::allocate).
    data = static_cast<std::byte*>(
        fast_alloc::arena_for(arena_kind).allocate(block_allocator, block_allocator, owner_rank));

    if (data == nullptr) [[unlikely]]
    {
        FAST_ALLOC_FATAL("slab arena (kind {}) failed to allocate {} bytes", static_cast<int>(arena_kind),
                         block_allocator);
    }

    // NOTE: no memset, fresh slab or recycled. The pool makes no promise about element
    // contents (TCSlab's @warning). Do NOT blanket-zero a recycled slab: it can wipe a block
    // the op pool still considers live, and it makes the whole slab resident at once instead
    // of page by page as elements are handed out.

    fast_alloc::store_to_bytes(data + block_allocator - sizeof(void*), this);

    first_free = free_list_end;
}

template <FreeLinks Links>
inline void TCSlab<Links>::destroy()
{
    if constexpr (side_links)
    {
        std::vector<unsigned short>().swap(tab_first_free_); // releases the free-list links
        // (not `= {}`: that is operator=(initializer_list), which keeps the capacity)
    }

    // Back to the slab's own arena's free-list, never to glibc: the address
    // stays inside that arena.
    fast_alloc::arena_for(arena_kind).deallocate(static_cast<void*>(data), block_allocator);
}

template <FreeLinks Links>
inline void TCSlab<Links>::init(int elem_size, uintptr_t block_bytes)
{
    if constexpr (!side_links)
    {
        // The free list is threaded THROUGH freed elements (free_block() writes the next
        // index, an int, into the element's first bytes): an element must hold an aligned int.
        // element_size is a run-time value, so this is a release check, once per slab.
        if (std::cmp_less(elem_size, sizeof(int)) || elem_size % static_cast<int>(alignof(int)) != 0) [[unlikely]]
        {
            FAST_ALLOC_FATAL("TCAllocator: element size {} bytes cannot hold the free-list index (an int: {} bytes, "
                             "aligned on {})",
                             elem_size, sizeof(int), alignof(int));
        }
    }

    element_size    = elem_size;
    block_allocator = block_bytes;

    log2_element_size_ = std::bit_width(static_cast<unsigned int>(element_size)) - 1;

    // num_block()'s magic is exact for offsets below 2^32 only. Slabs are a
    // few MiB; a release check, once per slab, like the one above.
    if (slab_bytes_for(block_bytes) > (std::uintptr_t{1} << 32)) [[unlikely]]
    {
        FAST_ALLOC_FATAL("TCAllocator: slab of {} bytes: num_block() indexes offsets below 4 GiB only",
                         slab_bytes_for(block_bytes));
    }
    // Unused (and 0, the +1 wrapping) for an element size of 1: a power of
    // two, num_block_pow2() takes it.
    element_magic_ = UINT64_MAX / static_cast<std::uint64_t>(element_size) + 1u;

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
            fast_alloc::store_to_bytes(data + block_allocator - sizeof(void*), this);
            backptr_dropped = false;
        }
    }

    ++nn_ref;
    std::byte* elem_ptr = nullptr;

    if (first_free != free_list_end)
    {
        elem_ptr   = data + (static_cast<std::ptrdiff_t>(first_free) * element_size);
        first_free = link(first_free); // the link is read before the element is handed out
    }
    else
    {
        elem_ptr = data + (static_cast<std::ptrdiff_t>(used_size) * element_size);
        ++used_size;
    }

    return elem_ptr;
}

template <FreeLinks Links>
inline void TCSlab<Links>::free_block(int elem) noexcept
{
    if (elem == (used_size - 1))
    {
        set_link(elem, free_list_end);
        --used_size;
    }
    else
    {
        set_link(elem, first_free);
        first_free = static_cast<IndexType>(elem);
    }

    --nn_ref;
}

// ---------------------------------------------------------------------------
// fast_alloc::TCSlab<Links>
// ---------------------------------------------------------------------------

template <FreeLinks Links>
[[nodiscard]] TCSlab<Links>::IndexType TCSlab<Links>::link(int elem) const noexcept
{
    if constexpr (side_links)
    {
        return tab_first_free_[static_cast<std::size_t>(elem)];
    }
    else
    {
        // the index sits in the first bytes of the free element
        return fast_alloc::load_from_bytes<IndexType>(data + (static_cast<std::ptrdiff_t>(elem) * element_size));
    }
}

template <FreeLinks Links>
void TCSlab<Links>::set_link(int elem, IndexType next_free) noexcept
{
    if constexpr (side_links)
    {
        tab_first_free_[static_cast<std::size_t>(elem)] = next_free;
    }
    else
    {
        fast_alloc::store_to_bytes(data + (static_cast<std::ptrdiff_t>(elem) * element_size), next_free);
    }
}

// ---------------------------------------------------------------------------
// fast_alloc::TCSlabPool<Derived, ElemT, Slab>
// ---------------------------------------------------------------------------

template <typename Derived, typename ElemT, typename Slab>
void TCSlabPool<Derived, ElemT, Slab>::init(size_t element_size, uintptr_t block_allocator, int nb_max_block_emptyl)
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

template <typename Derived, typename ElemT, typename Slab>
[[nodiscard]] TCSlabPool<Derived, ElemT, Slab>::ElementType* TCSlabPool<Derived, ElemT, Slab>::alloc()
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

template <typename Derived, typename ElemT, typename Slab>
void TCSlabPool<Derived, ElemT, Slab>::free_elt_pow2(const ElementType* ptr) noexcept
{
    if (bypass_slabs())
    {
        delete[] ptr;
        return;
    }

    free_elt_pow2(ptr, get_allocator(ptr));
}

template <typename Derived, typename ElemT, typename Slab>
void TCSlabPool<Derived, ElemT, Slab>::free_elt_pow2(const ElementType* ptr, Slab* slab) noexcept
{
    // The shift when this slab's element size is a power of two, the
    // magic multiplication otherwise (e.g. 192 doubles, a 24-byte struct,
    // a 40-byte op).
    // Tested on the slab, not deduced from sizeof(ElementType): the element
    // COUNT need not be a power of two. One well-predicted branch.
    const auto size = static_cast<unsigned int>(slab->element_size);
    const int  elem = std::has_single_bit(size) ? slab->num_block_pow2(ptr) : slab->num_block(ptr);

    slab->free_block(elem);

    manage_full_empty(slab);
}

template <typename Derived, typename ElemT, typename Slab>
[[nodiscard]] Slab* TCSlabPool<Derived, ElemT, Slab>::slab_of(const void* ptr) const noexcept
{
    // Mask with #slab_bytes_ (the ROUNDED size, Slab::slab_bytes_for), never
    // the raw budget: a non-pow2 budget would land beside the slab.
    if (slab_bytes_ == 0)
    {
        return nullptr;
    }
    return Slab::slab_from(ptr, slab_bytes_);
}

template <typename Derived, typename ElemT, typename Slab>
std::size_t TCSlabPool<Derived, ElemT, Slab>::detach_empty_slabs(std::vector<void*>& scratch) noexcept
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

template <typename Derived, typename ElemT, typename Slab>
std::size_t TCSlabPool<Derived, ElemT, Slab>::advise_recycled_slabs(std::size_t keep_bytes) const noexcept
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

template <typename Derived, typename ElemT, typename Slab>
void TCSlabPool<Derived, ElemT, Slab>::destroy()
{
    destroy_int();

    decltype(block_empty_)().swap(block_empty_); // releases the reserve's storage

    element_size_   = 0;
    block_full      = nullptr;
    block           = nullptr;
    nb_block_empty_ = 0;
}

template <typename Derived, typename ElemT, typename Slab>
void TCSlabPool<Derived, ElemT, Slab>::destroy_int()
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

template <typename Derived, typename ElemT, typename Slab>
Slab* TCSlabPool<Derived, ElemT, Slab>::new_slab()
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

template <typename Derived, typename ElemT, typename Slab>
FAST_ALLOC_COLD TCSlabPool<Derived, ElemT, Slab>::ElementType* TCSlabPool<Derived, ElemT, Slab>::alloc_slow()
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

template <typename Derived, typename ElemT, typename Slab>
TCSlabPool<Derived, ElemT, Slab>::ElementType* TCSlabPool<Derived, ElemT, Slab>::alloc_and_file(Slab* slab)
{
    auto* ptr = static_cast<ElementType*>(slab->alloc());
    if (slab->is_full()) [[unlikely]]
    {
        remove(block, slab);
        insert(block_full, slab);
    }
    return ptr;
}

template <typename Derived, typename ElemT, typename Slab>
void TCSlabPool<Derived, ElemT, Slab>::manage_full_empty(Slab* slab) noexcept
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

template <typename Derived, typename ElemT, typename Slab>
FAST_ALLOC_COLD void TCSlabPool<Derived, ElemT, Slab>::evict_slab(Slab* slab) noexcept
{
    if (fast_alloc::trace_evictions())
    {
        CHUNK_LOG(trace, alloc, "munmap: {} delete slab p={} nb_max_block_empty={}", Derived::pool_name,
                  static_cast<void*>(slab), nb_max_block_empty_);
    }
    delete slab;
}

// ---------------------------------------------------------------------------
// fast_alloc::TCTypeAllocator<ElemT>
// ---------------------------------------------------------------------------

template <typename ElemT>
void TCTypeAllocator<ElemT>::add_allocated_size(size_t& sum_used_size,
                                                size_t& sum_nn_ref,
                                                size_t& alloc_sz) const noexcept
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

// The class exists only in threaded builds (tc_allocator.h): its bodies too.
#if FAST_ALLOC_PMP_THREADED

// ---------------------------------------------------------------------------
// fast_alloc::TCTypeCAllocator<ElemT>
// ---------------------------------------------------------------------------

template <typename ElemT>
std::size_t TCTypeCAllocator<ElemT>::collect_empty_slabs([[maybe_unused]] SlabSweep&  sweep,
                                                         [[maybe_unused]] std::size_t keep_empty) noexcept
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

#endif // FAST_ALLOC_PMP_THREADED

// ---------------------------------------------------------------------------
// fast_alloc::TCAllocatorEB<ElementBytes, BlockBytes>
// ---------------------------------------------------------------------------

template <unsigned int ElementBytes, unsigned int BlockBytes>
TCAllocatorEB<ElementBytes, BlockBytes>::TCAllocatorEB()
{
    // alloc() inserts the slab right after this constructor returns: a
    // slab without a block (or without its back-pointer) would be handed
    // out and corrupt memory silently. Abort, as the other families do.
    if (alloc_block() != 0)
    {
        FAST_ALLOC_FATAL("TCAllocatorEB: its first block could not be allocated");
    }
}

template <unsigned int ElementBytes, unsigned int BlockBytes>
TCAllocatorEB<ElementBytes, BlockBytes>::~TCAllocatorEB()
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

template <unsigned int ElementBytes, unsigned int BlockBytes>
void* TCAllocatorEB<ElementBytes, BlockBytes>::alloc() noexcept
{
    ++nn_ref;
    std::byte* elem_ptr = nullptr;

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

template <unsigned int ElementBytes, unsigned int BlockBytes>
void TCAllocatorEB<ElementBytes, BlockBytes>::free_block(int elem) noexcept
{
    if (elem == (used_size - 1))
    {
        *(T*)(data + (elem * element_size)) = -1;
        --used_size;
    }
    else
    {
        *(T*)(data + (elem * element_size)) = first_free;
        first_free                          = elem;
    }
    --nn_ref;
}

template <unsigned int ElementBytes, unsigned int BlockBytes>
int TCAllocatorEB<ElementBytes, BlockBytes>::alloc_block()
{
    if (const long ps = fast_alloc::page_size(); ps > 0 && (2 * block_size) % ps == 0)
    {
        use_mmap_ = true;

        const uintptr_t block_bytes = block_size;

        std::byte* mapping = static_cast<std::byte*>(fast_alloc::map_anonymous(2 * block_size));

        data = mapping + ((0 - fast_alloc::address_bits(mapping)) & (block_bytes - 1)); // align up

        const auto lng = static_cast<uintptr_t>(data - mapping);
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

        if (const int result = posix_memalign(&datal, block_size_pot, block_size); result != 0)
        {
            CHUNK_LOG(error, alloc, "posix_memalign failed: error {} ({})", result, std::strerror(result));
            return 1;
        }
        data = static_cast<std::byte*>(datal);
    }

    fast_alloc::store_to_bytes(data + used_block_size, this);

    first_free = -1;

    return 0;
}

// ---------------------------------------------------------------------------
// fast_alloc::TCTypeAllocatorEBC<ElementSize, BlockSize>
// ---------------------------------------------------------------------------

template <unsigned int ElementSize, unsigned int BlockSize>
void* TCTypeAllocatorEBC<ElementSize, BlockSize>::alloc()
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

template <unsigned int ElementSize, unsigned int BlockSize>
void TCTypeAllocatorEBC<ElementSize, BlockSize>::add_allocated_size(size_t& sum_used_size,
                                                                    size_t& sum_nn_ref,
                                                                    size_t& alloc_sz) const noexcept
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

template <unsigned int ElementSize, unsigned int BlockSize>
void TCTypeAllocatorEBC<ElementSize, BlockSize>::manage_full_empty(TCAllocatorEBT* slab) noexcept
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

} // namespace fast_alloc
