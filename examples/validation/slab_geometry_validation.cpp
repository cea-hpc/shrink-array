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

// ============================================================================
// Slab geometry of the zeroed value allocator (TCTypeCAllocator), across
// element sizes. No array, no variant: the allocator alone.
// ============================================================================
//
// A chunk buffer is freed through ChunkBase::release_to_owner(), which resolves
// the buffer's slab with slab_of() of the allocator the calling thread's
// AMBIENT chunk size selects -- then frees through the slab's stamped owner.
// That is only right if every allocator of one element type cuts its slabs at
// the same size whatever the chunk size: after a re-chunk, a buffer of the old
// size is freed while the ambient size may already be the new one.
//
// TCCAllocator::slab_bytes_for() must not shrink a slab whose capacity exceeds
// what the unsigned-short free list can index: for small elements (a chunk of
// bools is a few bytes) that would make the geometry depend on the element
// size (8- and 16-byte elements in one 2 MiB budget getting 256 KiB and
// 512 KiB slabs), and slab_of() from the other size would mask the pointer at
// the wrong boundary.
//
// Checks, for pairs of element sizes sharing one budget: the slab found for a
// block through the OTHER allocator is the block's own slab, owned by the
// allocator that handed it out. Pointers are only compared, never followed, so
// a wrong geometry fails the check instead of crashing on it.
//
// Exit 0 = every check holds.

#include <cstdint> // std::uintptr_t
#include <cstdio>  // std::printf

#include "fast_allocators/tc_allocator.h" // TCTypeCAllocator, TCCAllocator


namespace
{

int check_pair(int small_bytes, int large_bytes, std::uintptr_t budget)
{
    fast_alloc::TCTypeCAllocator<char> a_small;
    fast_alloc::TCTypeCAllocator<char> a_large;
    a_small.init(static_cast<size_t>(small_bytes), budget, -1);
    a_large.init(static_cast<size_t>(large_bytes), budget, -1);

    int bad = 0;
    for (auto* a : {&a_small, &a_large})
    {
        const auto* const         other = (a == &a_small) ? &a_large : &a_small;
        const char*               p     = a->alloc();
        fast_alloc::TCCAllocator* own   = a->slab_of(p);
        fast_alloc::TCCAllocator* seen  = other->slab_of(p);
        const bool                ok    = own != nullptr && seen == own && own->owner == static_cast<const void*>(a);
        if (!ok)
        {
            std::printf("FAIL budget %zu: %d-byte block, slab %p from its allocator, %p through the %d-byte one\n",
                        static_cast<std::size_t>(budget), a == &a_small ? small_bytes : large_bytes,
                        static_cast<void*>(own), static_cast<void*>(seen), a == &a_small ? large_bytes : small_bytes);
            ++bad;
        }
        a->free_elt_pow2(p);
    }
    if (!bad)
    {
        std::printf("OK: %d- and %d-byte elements in a %zu-byte budget share one slab geometry\n", small_bytes,
                    large_bytes, static_cast<std::size_t>(budget));
    }
    return bad;
}

} // namespace

int main()
{
    std::printf("=== slab_geometry_validation ===\n");
    int fails = 0;
    // 1.875 MiB per rank (15 MiB / 8, one thread) -> 2 MiB slabs: 64 and 128
    // bools as bits (8 and 16 bytes), and as bytes (64 and 128).
    fails += check_pair(8, 16, std::uintptr_t{15} * 1024 * 1024 / 8);
    fails += check_pair(64, 128, std::uintptr_t{15} * 1024 * 1024 / 8);
    // The 240 KiB floor of a large team.
    fails += check_pair(8, 16, std::uintptr_t{240} * 1024);

    std::printf("\n%s\n", fails ? "=== SLAB GEOMETRY FAILED ===" : "=== SLAB GEOMETRY OK ===");
    return fails ? 1 : 0;
}
