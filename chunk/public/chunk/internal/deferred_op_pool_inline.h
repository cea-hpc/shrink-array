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
#include "chunk/internal/deferred_op_pool.h" // the class these bodies belong to

namespace shrink_array
{

// ---------------------------------------------------------------------------
// shrink_array::DeferredByteAllocSlot<T>
// ---------------------------------------------------------------------------

template <typename T>
DeferredByteAllocSlot<T>::~DeferredByteAllocSlot()
{
    if constexpr (on_heap)
    {
        if (heap_ != nullptr)
        {
            auto&                  ex = detail::exited_thread_pools();
            const std::scoped_lock lock(ex.mutex);
            ex.list.push_back(heap_);
        }
    }
}

template <typename T>
[[nodiscard]] DeferredByteAllocSlot<T>::Alloc& DeferredByteAllocSlot<T>::alloc()
{
    if constexpr (on_heap)
    {
        if (heap_ == nullptr) [[unlikely]]
        {
            heap_ = new Alloc; // NOLINT(cppcoreguidelines-owning-memory): handed over at thread exit
        }
        return *heap_;
    }
    else
    {
        return inline_;
    }
}

// ---------------------------------------------------------------------------
// shrink_array::DeferredFreeListSlot<T>
// ---------------------------------------------------------------------------

template <typename T>
DeferredFreeListSlot<T>::~DeferredFreeListSlot()
{
    if constexpr (!build_config::use_fast_alloc)
    {
        for (T* block : fl)
        {
            ::operator delete(static_cast<void*>(block));
        }
    }
}

// ---------------------------------------------------------------------------
// shrink_array::FastAllocatorConcrete<T, Arena>
// ---------------------------------------------------------------------------

template <typename T, fast_alloc::ArenaKind Arena>
[[nodiscard]] const void* FastAllocatorConcrete<T, Arena>::slab_addr() const noexcept
{
    if constexpr (build_config::use_fast_alloc)
    {
        return static_cast<const void*>(&allocator());
    }
    else
    {
        return static_cast<const void*>(&free_list());
    }
}

template <typename T, fast_alloc::ArenaKind Arena>
void* FastAllocatorConcrete<T, Arena>::allocate_raw()
{
    if constexpr (build_config::pool_no_reuse)
    {
        // Diagnostic mode (CHUNK_POOL_NO_REUSE): bypass the free-list AND the fast
        // pool, so recycle() is a real ::operator delete and sanitizers see a
        // use-after-recycle that the free-list would hide (stale but mapped data).
        return ::operator new(sizeof(T));
    }
    auto& fl = free_list();
    if (!fl.empty())
    {
        T* block = fl.back();
        fl.pop_back();
        // Check-build guard: a firing here blames the free-list (a corrupt
        // recycled pointer), not the slab.
        if constexpr (build_config::any_check)
        {
            probe_returned_block(static_cast<void*>(block), "freelist");
        }
        return block;
    }

    if constexpr (build_config::use_fast_alloc)
    {
        std::byte* raw = allocator().alloc();
        // Check-build guard: a firing here blames TCTypeAllocator itself
        // (corrupt block/bump state).
        if constexpr (build_config::any_check)
        {
            probe_returned_block(static_cast<void*>(raw), "slab");
        }
        return static_cast<void*>(raw);
    }
    else
    {
        return ::operator new(sizeof(T));
    }
}

template <typename T, fast_alloc::ArenaKind Arena>
void FastAllocatorConcrete<T, Arena>::recycle(void* block)
{
    if constexpr (build_config::pool_no_reuse)
    {
        ::operator delete(block); // CHUNK_POOL_NO_REUSE: real free, visible to sanitizers
        return;
    }
    if constexpr (build_config::use_fast_alloc)
    {
        // Foreign block (cross-thread drain): drop it (see above).
        if (!allocator().is_free_able_elt(static_cast<std::byte*>(block)))
        {
            if constexpr (build_config::any_check)
            {
                // Expected, not an anomaly: recycle=debug reports the
                // first, then on a doubling schedule with the running total
                // (~log2(N) lines per thread); recycle=trace one line per
                // block.
                thread_local std::size_t n_foreign = 0;
                ++n_foreign;
                if (std::has_single_bit(n_foreign))
                {
                    CHUNK_LOG(debug, recycle,
                              "dropped FOREIGN block {} (not owned by this thread's slab) -- cross-thread drain, "
                              "would be use-after-munmap if recycled; {} on this thread so far",
                              block, n_foreign);
                }
                else
                {
                    CHUNK_LOG(trace, recycle,
                              "dropped FOREIGN block {} (not owned by this thread's slab) -- cross-thread drain, "
                              "would be use-after-munmap if recycled; {} on this thread so far",
                              block, n_foreign);
                }
            }
            return;
        }
    }
    free_list().push_back(static_cast<T*>(block));
}

template <typename T, fast_alloc::ArenaKind Arena>
[[nodiscard]] uintptr_t FastAllocatorConcrete<T, Arena>::deferred_pool_slab_granularity() noexcept
{
    static const uintptr_t bytes = []
    {
        constexpr std::size_t def_kb = build_config::deferred_pool_granularity / 1024;
        const auto            kb     = shrink_array::env_number<std::size_t>("CHUNK_DEFERRED_POOL_KB", def_kb);
        return static_cast<uintptr_t>(kb == 0 ? def_kb : kb) * 1024;
    }();
    return bytes;
}

template <typename T, fast_alloc::ArenaKind Arena>
FastAllocatorConcrete<T, Arena>::ByteAlloc& FastAllocatorConcrete<T, Arena>::allocator()
{
    // T-tagged storage against -fipa-icf folding: see DeferredByteAllocSlot.
    thread_local DeferredByteAllocSlot<T> slot;

    if (!slot.initialized)
    {
        // Op and ctx pools MUST draw from disjoint arenas: a freed ctx slab
        // reobtained by the op pool would overwrite a live op's vtable.
        slot.alloc().set_arena(Arena);

        constexpr size_t element_size = sizeof(T);
        // TCAllocator threads its free list through the freed elements (an
        // int in each): TCAllocator::init()'s run-time check, at compile time.
        static_assert(element_size >= sizeof(int) && element_size % alignof(int) == 0,
                      "deferred-op pool: an object must hold TCAllocator's free-list index");

        // Slab size in BYTES (TCSlabPool::init's second parameter; not an
        // object count). Never fewer than 64 objects per slab.
        const uintptr_t slab_request =
            std::max<uintptr_t>(deferred_pool_slab_granularity(), uintptr_t{64} * element_size);

        // nb_max_block_empty = -1: NEVER munmap an empty slab. A freed op
        // slab's address could be remapped by fast_alloc::map_anonymous() for
        // the VALUE pool on another thread and overwrite a still-live op. Op
        // slabs are small and bounded.
        slot.alloc().init(element_size, slab_request, -1);
        slot.initialized = true;
    }
    return slot.alloc();
}

template <typename T, fast_alloc::ArenaKind Arena>
void FastAllocatorConcrete<T, Arena>::probe_returned_block(const void* ptr, const char* path) noexcept
{
    if constexpr (build_config::any_check)
    {
        const auto addr = ::fast_alloc::address_bits(ptr);

        if (ptr == nullptr)
        {
            CHUNK_INVARIANT_FAILED("[ALLOC_PROBE {}] allocate_raw returned NULL for T size={}", path, sizeof(T));
        }

        if ((addr & (alignof(std::max_align_t) - 1)) != 0 && (addr & 0x7U) != 0)
        {
            CHUNK_INVARIANT_FAILED("[ALLOC_PROBE {}] misaligned block p={} (T size={})", path, ptr, sizeof(T));
        }

        // msync(MS_ASYNC) only reads the VMA table: 0 = mapped, -1/ENOMEM = not.
        // The object may straddle two pages: check both ends. Page bases by
        // pointer arithmetic from p, not int -> pointer casts (performance-no-int-to-ptr).
        const long  pg   = ::sysconf(_SC_PAGESIZE);
        const auto  mask = static_cast<std::uintptr_t>(pg - 1);
        const auto* base = static_cast<const char*>(ptr);
        const void* pg0  = base - (addr & mask);
        const void* pg1  = base + (sizeof(T) - 1) - ((addr + sizeof(T) - 1) & mask);

        if (::msync(const_cast<void*>(pg0), 1, MS_ASYNC) != 0 || ::msync(const_cast<void*>(pg1), 1, MS_ASYNC) != 0)
        {
            CHUNK_INVARIANT_FAILED("[ALLOC_PROBE {}] block NOT MAPPED (wild ptr): p={} T size={} slab={}", path, ptr,
                                   sizeof(T), build_config::use_fast_alloc ? &allocator() : nullptr);
        }
    }
}

// ---------------------------------------------------------------------------
// shrink_array::AllocatorRegistry<Tag>
// ---------------------------------------------------------------------------

template <typename Tag>
template <typename T>
FastAllocatorBase* AllocatorRegistry<Tag>::get_or_create()
{
    // Hot (every alloc/release): cached per thread until reset() moves the generation.
    thread_local CachedAllocatorSlot<Tag, T> cached;
    if (cached.alloc != nullptr && cached.generation == generation()) [[likely]]
    {
        return cached.alloc;
    }
    cached.alloc      = find_or_create<T>();
    cached.generation = generation();
    return cached.alloc;
}

template <typename Tag>
template <typename T>
FastAllocatorBase* AllocatorRegistry<Tag>::find_or_create()
{
    auto&        allocators = allocs();
    const size_t sz         = sizeof(T);

    for (auto& alloc : allocators)
    {
        if (alloc->block_size == sz)
        {
            return alloc.get();
        }
    }

    auto               ptr = std::make_unique<FastAllocatorConcrete<T, arena_of_tag<Tag>::value>>();
    FastAllocatorBase* raw = ptr.get();
    allocators.push_back(std::move(ptr));
    return raw;
}

} // namespace shrink_array
