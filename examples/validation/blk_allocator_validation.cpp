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
//   tab       alloc_tab's result is aligned to __ALIGNEMENT__;
//   parallel  (OpenMP builds) the same from every thread at once -- the
//             ladder's critical sections and the per-thread free batches --
//             then clearInfoFreeElt() at the quiescent point after the region.
//
// EXIT STATUS: 0 when every check holds, 1 otherwise.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include <unistd.h>

#include "fast_allocators/internal/static_const_array.h"

int __mem_page_size = 0;

namespace
{

constexpr std::size_t kSizes[] = {1,   3,   4,    5,    8,    16,    24,    63,    64,    65,     100,
                                  512, 513, 1000, 4096, 4097, 32768, 65535, 65536, 65537, 100000, std::size_t{1} << 20};

unsigned char pattern(std::size_t id, std::size_t k)
{
    return static_cast<unsigned char>((id * 131u + k * 7u + 1u) & 0xFFu);
}

void fill(void* p, std::size_t n, std::size_t id)
{
    auto* b = static_cast<unsigned char*>(p);
    for (std::size_t k = 0; k < n; ++k)
    {
        b[k] = pattern(id, k);
    }
}

bool intact(const void* p, std::size_t n, std::size_t id)
{
    const auto* b = static_cast<const unsigned char*>(p);
    for (std::size_t k = 0; k < n; ++k)
    {
        if (b[k] != pattern(id, k))
        {
            return false;
        }
    }
    return true;
}

int fails = 0;

void expect(bool ok, const char* what)
{
    std::printf("  [%s] %s\n", ok ? " ok " : "FAIL", what);
    if (!ok)
    {
        ++fails;
    }
}

/// @brief One block per size, both front-end pairs.
void check_sizes()
{
    bool ok_blk = true, ok_tab = true, aligned = true;
    for (std::size_t i = 0; i < std::size(kSizes); ++i)
    {
        const std::size_t n = kSizes[i];
        void*             p = BlkAllocator::alloc_block(n);
        if (p == nullptr)
        {
            ok_blk = false;
            continue;
        }
        fill(p, n, i);
        ok_blk = ok_blk && intact(p, n, i);
        BlkAllocator::free_block(p, n);

        void* t = BlkAllocator::alloc_tab(n);
        if (t == nullptr)
        {
            ok_tab = false;
            continue;
        }
        aligned = aligned && (reinterpret_cast<std::uintptr_t>(t) % __ALIGNEMENT__) == 0;
        fill(t, n, i + 1000);
        ok_tab = ok_tab && intact(t, n, i + 1000);
        BlkAllocator::free_tab(t);
    }
    expect(ok_blk, "alloc_block/free_block: 1 B to 1 MiB, on, between and past the buckets");
    expect(ok_tab, "alloc_tab/free_tab: same sizes, size read back from the header");
    expect(aligned, "alloc_tab: aligned to __ALIGNEMENT__");
}

/// @brief Many live blocks at once: no two overlap; then reuse after free.
void check_distinct()
{
    constexpr std::size_t kN = 3000;
    for (int round = 0; round < 2; ++round)
    {
        std::vector<void*>       blk(kN);
        std::vector<void*>       tab(kN);
        std::vector<std::size_t> sz(kN);
        for (std::size_t j = 0; j < kN; ++j)
        {
            sz[j]  = kSizes[j % 20]; // up to 65537: the ladder and its fallback
            blk[j] = BlkAllocator::alloc_block(sz[j]);
            tab[j] = BlkAllocator::alloc_tab(sz[j]);
            fill(blk[j], sz[j], 2 * j);
            fill(tab[j], sz[j], 2 * j + 1);
        }
        bool ok = true;
        for (std::size_t j = 0; j < kN; ++j)
        {
            ok = ok && intact(blk[j], sz[j], 2 * j) && intact(tab[j], sz[j], 2 * j + 1);
        }
        for (std::size_t j = 0; j < kN; ++j)
        {
            BlkAllocator::free_block(blk[j], sz[j]);
            BlkAllocator::free_tab(tab[j]);
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
        std::vector<void*> live;
        for (int it = 0; it < 2000; ++it)
        {
            const std::size_t n  = kSizes[static_cast<std::size_t>(it) % 20];
            const std::size_t id = static_cast<std::size_t>(it) * 97u;
            void*             p  = (it % 2) ? BlkAllocator::alloc_tab(n) : BlkAllocator::alloc_block(n);
            fill(p, n, id);
            if (!intact(p, n, id))
            {
                ++bad;
            }
            if (it % 2)
            {
                BlkAllocator::free_tab(p);
            }
            else
            {
                BlkAllocator::free_block(p, n);
            }
        }
    }
    // Quiescent point: the blocks freed inside the region were batched per
    // thread; hand them back to the ladder (what the chunk library's
    // parallel_end callback does).
    StaticConstArray::clearInfoFreeElt();
    expect(bad == 0, "every thread allocating, writing and freeing at once");
}

} // namespace

int main()
{
    __mem_page_size = getpagesize();
    std::printf("=== blk_allocator_validation ===\n");
    check_sizes();
    check_distinct();
    check_parallel();
    check_distinct(); // the ladder is still sound after the batches went back
    std::printf("\n%s\n", fails ? "=== BLK ALLOCATOR FAILED ===" : "=== BLK ALLOCATOR OK ===");
    return fails ? 1 : 0;
}
