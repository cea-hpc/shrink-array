/*
 * Copyright 2026 CEA (www.cea.fr)
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
// BlkAllocator harness: fast_allocators' size-dispatching front end, alone
// (no chunk variant). Nothing in the library calls it -- it is public API for
// code outside shrink-array -- so this is the only thing that runs it.
// ============================================================================
//
//   sizes     alloc_block / free_block and alloc_tab / free_tab at sizes on,
//             between and past the ladder's buckets (1 byte to 1 MiB, the
//             largest bucket being 65536): every block is fully writable and
//             keeps its content until freed;
//   distinct  a few thousand live blocks of mixed sizes at once never overlap
//             (each is filled with its own pattern and checked after all the
//             others were written), then all are freed and taken again;
//   tab       alloc_tab's result is aligned to fast_alloc::alignment;
//   huge      (opt-in, BLK_HUGE=1) 2 GiB - 64, 2 GiB + 64 and 4 GiB + 64
//             through both pairs: the size_t -> int / unsigned truncations
//             (first and last 4 KiB written and read back; up to ~4 GiB
//             resident, one block at a time);
//   parallel  (OpenMP builds) the same from every thread at once -- the
//             ladder's critical sections and the per-thread free batches --
//             then clear_info_free_elt() at the quiescent point after the region.
//
// EXIT STATUS: 0 when every check holds, 1 otherwise.

#include <cstdint> // std::uintptr_t
#include <cstdlib> // std::getenv (BLK_HUGE)
#include <vector>  // std::vector

#include "fast_allocators/internal/static_const_array.h" // BlkAllocator, StaticConstArray
#include "chunk/internal/print.h"                        // shrink_array::println

using shrink_array::println;

namespace
{

constexpr std::size_t test_sizes[] = {
    1,   3,   4,    5,    8,    16,    24,    63,    64,    65,     100,
    512, 513, 1000, 4096, 4097, 32768, 65535, 65536, 65537, 100000, std::size_t{1} << 20};

unsigned char pattern(std::size_t id, std::size_t offset)
{
    return static_cast<unsigned char>((id * 131u + offset * 7u + 1u) & 0xFFu);
}

void fill(void* ptr, std::size_t size, std::size_t id)
{
    auto* bytes = static_cast<unsigned char*>(ptr);
    for (std::size_t k = 0; k < size; ++k)
    {
        bytes[k] = pattern(id, k);
    }
}

bool intact(const void* ptr, std::size_t size, std::size_t id)
{
    const auto* bytes = static_cast<const unsigned char*>(ptr);
    for (std::size_t k = 0; k < size; ++k)
    {
        if (bytes[k] != pattern(id, k))
        {
            return false;
        }
    }
    return true;
}

int fails = 0;

void expect(bool ok, const char* what)
{
    println("  [{}] {}", ok ? " ok " : "FAIL", what);
    if (!ok)
    {
        ++fails;
    }
}

/// @brief One block per size, both front-end pairs.
void check_sizes()
{
    bool ok_blk  = true;
    bool ok_tab  = true;
    bool aligned = true;
    for (std::size_t i = 0; i < std::size(test_sizes); ++i)
    {
        const std::size_t size = test_sizes[i];
        void*             ptr  = fast_alloc::BlkAllocator::alloc_block(size);
        if (ptr == nullptr)
        {
            ok_blk = false;
            continue;
        }
        fill(ptr, size, i);
        ok_blk = ok_blk && intact(ptr, size, i);
        fast_alloc::BlkAllocator::free_block(ptr, size);

        void* tab = fast_alloc::BlkAllocator::alloc_tab(size);
        if (tab == nullptr)
        {
            ok_tab = false;
            continue;
        }
        aligned = aligned && (reinterpret_cast<std::uintptr_t>(tab) % fast_alloc::alignment) == 0;
        fill(tab, size, i + 1000);
        ok_tab = ok_tab && intact(tab, size, i + 1000);
        fast_alloc::BlkAllocator::free_tab(tab);
    }
    expect(ok_blk, "alloc_block/free_block: 1 B to 1 MiB, on, between and past the buckets");
    expect(ok_tab, "alloc_tab/free_tab: same sizes, size read back from the header");
    expect(aligned, "alloc_tab: aligned to fast_alloc::alignment");
}

/// @brief Sizes past INT_MAX and UINT_MAX, where the front end must not
///        narrow before routing (free_block() sending [2 GiB, 4 GiB) to the
///        ladder, alloc_block() serving 4 GiB + 64 from the 64-byte bucket,
///        an int header in alloc_tab()). Opt-in (BLK_HUGE=1): it touches
///        gigabytes. A regression crashes rather than reporting FAIL.
void check_huge()
{
    if (sizeof(std::size_t) < 8 || std::getenv("BLK_HUGE") == nullptr)
    {
        println("  [skip] sizes past 2 GiB / 4 GiB (set BLK_HUGE=1)");
        return;
    }
    constexpr std::size_t gib     = std::size_t{1} << 30;
    constexpr std::size_t window  = 4096; // bytes checked at each end
    const std::size_t     sizes[] = {2 * gib - 64, 2 * gib + 64, 4 * gib + 64};

    const auto ends_ok = [](void* ptr, std::size_t size, std::size_t id)
    {
        auto* bytes = static_cast<unsigned char*>(ptr);
        fill(bytes, window, id);
        fill(bytes + size - window, window, id + 1);
        return intact(bytes, window, id) && intact(bytes + size - window, window, id + 1);
    };

    bool ok_blk = true;
    bool ok_tab = true;
    for (std::size_t i = 0; i < std::size(sizes); ++i)
    {
        const std::size_t size = sizes[i];

        void* ptr = fast_alloc::BlkAllocator::alloc_block(size);
        ok_blk    = ok_blk && ptr != nullptr && ends_ok(ptr, size, 10 * i);
        fast_alloc::BlkAllocator::free_block(ptr, size);

        void* tab = fast_alloc::BlkAllocator::alloc_tab(size);
        ok_tab    = ok_tab && tab != nullptr && ends_ok(tab, size, 10 * i + 5);
        fast_alloc::BlkAllocator::free_tab(tab);
    }
    expect(ok_blk, "alloc_block/free_block: 2 GiB - 64, 2 GiB + 64, 4 GiB + 64");
    expect(ok_tab, "alloc_tab/free_tab: same sizes, size_t header");
}

/// @brief Many live blocks at once: no two overlap; then reuse after free.
void check_distinct()
{
    constexpr std::size_t n_items = 3000;
    for (int round = 0; round < 2; ++round)
    {
        std::vector<void*>       blk(n_items);
        std::vector<void*>       tab(n_items);
        std::vector<std::size_t> sz(n_items);
        for (std::size_t j = 0; j < n_items; ++j)
        {
            sz[j]  = test_sizes[j % 20]; // up to 65537: the ladder and its fallback
            blk[j] = fast_alloc::BlkAllocator::alloc_block(sz[j]);
            tab[j] = fast_alloc::BlkAllocator::alloc_tab(sz[j]);
            fill(blk[j], sz[j], 2 * j);
            fill(tab[j], sz[j], 2 * j + 1);
        }
        bool ok = true;
        for (std::size_t j = 0; j < n_items; ++j)
        {
            ok = ok && intact(blk[j], sz[j], 2 * j) && intact(tab[j], sz[j], 2 * j + 1);
        }
        for (std::size_t j = 0; j < n_items; ++j)
        {
            fast_alloc::BlkAllocator::free_block(blk[j], sz[j]);
            fast_alloc::BlkAllocator::free_tab(tab[j]);
        }
        expect(ok, round == 0 ? "3000 + 3000 live blocks of mixed sizes never overlap"
                              : "same after every block was freed and taken again");
    }
}

/// @brief Every thread at once (OpenMP builds); a serial build runs it on one.
void check_parallel()
{
    long bad = 0;
#pragma omp parallel reduction(+ : bad)
    {
        for (int it = 0; it < 2000; ++it)
        {
            const std::size_t size = test_sizes[static_cast<std::size_t>(it) % 20];
            const std::size_t id   = static_cast<std::size_t>(it) * 97u;
            void*             ptr =
                (it % 2) ? fast_alloc::BlkAllocator::alloc_tab(size) : fast_alloc::BlkAllocator::alloc_block(size);
            fill(ptr, size, id);
            if (!intact(ptr, size, id))
            {
                ++bad;
            }
            if (it % 2)
            {
                fast_alloc::BlkAllocator::free_tab(ptr);
            }
            else
            {
                fast_alloc::BlkAllocator::free_block(ptr, size);
            }
        }
    }
    // Quiescent point: the blocks freed inside the region were batched per
    // thread; hand them back to the ladder (what the chunk library's
    // parallel_end callback does).
    fast_alloc::StaticConstArray::clear_info_free_elt();
    expect(bad == 0, "every thread allocating, writing and freeing at once");
}

/// @brief The ladder reached from inside the CALLER's own unnamed critical
///        section, by every thread. The ladder's criticals are named
///        (fast_alloc_ladder): unnamed, they were the same lock as the
///        caller's, and a critical is not reentrant -- this hung.
void check_inside_caller_critical()
{
    long bad = 0;
#pragma omp parallel reduction(+ : bad)
    {
        for (int it = 0; it < 200; ++it)
        {
            const std::size_t size = test_sizes[static_cast<std::size_t>(it) % 20];
            const std::size_t id   = static_cast<std::size_t>(it) * 131u;
#pragma omp critical
            {
                void* ptr = fast_alloc::BlkAllocator::alloc_block(size);
                fill(ptr, size, id);
                if (!intact(ptr, size, id))
                {
                    ++bad;
                }
                fast_alloc::BlkAllocator::free_block(ptr, size);
            }
        }
    }
    fast_alloc::StaticConstArray::clear_info_free_elt();
    expect(bad == 0, "alloc and free from inside the caller's own unnamed omp critical");
}

} // namespace

int main()
{
    println("=== blk_allocator_validation ===");
    check_sizes();
    check_huge();
    check_distinct();
    check_parallel();
    check_inside_caller_critical();
    check_distinct(); // the ladder is still sound after the batches went back
    println("\n{}", fails ? "=== BLK ALLOCATOR FAILED ===" : "=== BLK ALLOCATOR OK ===");
    return fails ? 1 : 0;
}
