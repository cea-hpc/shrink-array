/*
 * Copyright 2024-2025 CEA (www.cea.fr)
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

#include <cassert>
#include "fast_allocators/internal/int_alloc.h"

// Unconditional: the bodies below branch with `if constexpr`, and a non-template
// function compiles BOTH arms whichever way the condition folds. NextPowerOf2 is
// used in the __debug_mem arm, so it has to exist even in a STD_ALLOC build.
#include "fast_allocators/internal/next_power_of_2.h"

#include "fast_allocators/internal/alloc_config.h" // constexpr mirrors of STD_ALLOC / DEBUG
#include "fast_allocators/internal/pmp_alloc.h"

/**
 * @copydoc StaticConstArray::getBlockAllocInt
 * @details The comparison tree is deliberately unbalanced: the small classes,
 *          which are the common ones, are reached in fewer tests than the
 *          large ones.
 */
inline void* StaticConstArray::getBlockAllocInt(unsigned int sz)
{
    if constexpr (alloc_config::use_std_alloc)
    {
        return int_alloc(sz);
    }

    if (sz <= 8)
    {
        if (sz <= 4)
        {
            return _Allocator4.alloc();
        }
        else
        {
            return _Allocator8.alloc();
        }
    }
    else if (sz <= 128)
    {
        if (sz <= 32)
        {
            if (sz <= 16)
            {
                return _Allocator16.alloc();
            }
            else
            {
                return _Allocator32.alloc();
            }
        }
        else
        {
            if (sz <= 64)
            {
                return _Allocator64.alloc();
            }
            else
            {
                return _Allocator128.alloc();
            }
        }
    }
    else
    {
        if (sz <= 1024)
        {
            if (sz <= 256)
            {
                return _Allocator256.alloc();
            }
            else if (sz <= 512)
            {
                return _Allocator512.alloc();
            }
            else
            {
                return _Allocator1024.alloc();
            }
        }
        else
        {
            if (sz <= 2048)
            {
                return _Allocator2048.alloc();
            }
            else if (sz <= 4096)
            {
                return _Allocator4096.alloc();
            }
            else if (sz <= 8192)
            {
                return _Allocator8192.alloc();
            }
            else if (sz <= 16384)
            {
                return _Allocator16384.alloc();
            }
            else if (sz <= 32768)
            {
                return _Allocator32768.alloc();
            }
            else
            {
                return _Allocator65536.alloc();
            }
        }
    }
    //    return 0;
}

/// @copydoc StaticConstArray::getBlockAlloc
inline void* StaticConstArray::getBlockAlloc(unsigned int sz)
{
    if constexpr (alloc_config::use_std_alloc)
    {
        return int_alloc(sz);
    }

    // The same bypass as getBlockAllocAlign, and as freeBlock honours: with the
    // flag set, freeBlock int_free()s whatever it is given, so the block must
    // not come from the ladder's slabs (it used to: free() on a slab pointer).
    if constexpr (alloc_config::debug_mem_observable)
    {
        if (__debug_mem)
        {
            return int_alloc(NextPowerOf2(sz));
        }
    }

    // Tree: unbalanced: favor speed on small blocks
    if (sz <= MAX_CONST_BLOCK_SIZE)
    {
        void* vv;
        if (PMP_IN_PARALLEL_REGION)
        {
#pragma omp critical
            {
                vv = getBlockAllocInt(sz);
            }
        }
        else
        {
            vv = getBlockAllocInt(sz);
        }
        //
        // #if defined(DEBUG) || defined(_OPENMP)
        //         int i = 0;
        //         if (PMP_IN_PARALLEL_REGION)
        //         {
        //             i++;
        //         }
        // #endif

        return vv;
    }

    // #if defined(DEBUG) || defined(_OPENMP)
    //     int i = 0;
    //     if (PMP_IN_PARALLEL_REGION)
    //     {
    //         i++;
    //     }
    // #endif

    return nullptr;
}

/**
 * @copydoc StaticConstArray::getBlockAllocAlignInt
 * @details Which small classes are skipped depends on `__ALIGNEMENT__`: a
 *          bucket narrower than the required alignment cannot satisfy it, so
 *          the ladder starts at the first one that can.
 */
inline void* StaticConstArray::getBlockAllocAlignInt(int sz)
{
    if constexpr (alloc_config::use_std_alloc)
    {
        return int_alloc_align(sz);
    }


    if (sz <= 128)
    {
// Stays a directive: __ALIGNEMENT__ is SIMD_SZ, and mirroring it as constexpr
// would make fast_allocators read a value the chunk layer owns.
#if __ALIGNEMENT__ == 16
        if (sz <= 16)
            return _Allocator16.alloc();
#elif __ALIGNEMENT__ >= 32
        if (sz <= 32)
            return _Allocator32.alloc();
#endif
        if (sz <= 64)
        {
            return _Allocator64.alloc();
        }
        else
        {
            return _Allocator128.alloc();
        }
    }
    else
    {
        if (sz <= 1024)
        {
            if (sz <= 256)
            {
                return _Allocator256.alloc();
            }
            else if (sz <= 512)
            {
                return _Allocator512.alloc();
            }
            else
            {
                return _Allocator1024.alloc();
            }
        }
        else
        {
            if (sz <= 2048)
            {
                return _Allocator2048.alloc();
            }
            else if (sz <= 4096)
            {
                return _Allocator4096.alloc();
            }
            else if (sz <= 8192)
            {
                return _Allocator8192.alloc();
            }
            else if (sz <= 16384)
            {
                return _Allocator16384.alloc();
            }
            else if (sz <= 32768)
            {
                return _Allocator32768.alloc();
            }
            else
            {
                return _Allocator65536.alloc();
            }
        }
    }
    //    return 0;
}

/// @copydoc StaticConstArray::getBlockAllocAlign
inline void* StaticConstArray::getBlockAllocAlign(int sz)
{
    if constexpr (alloc_config::use_std_alloc)
    {
        return int_alloc_align(sz);
    }

    // Rounded up to a power of two so the pointer a memory checker sees has the
    // same footprint the ladder would have handed out.
    if constexpr (alloc_config::debug_mem_observable)
    {
        if (__debug_mem)
        {
            return int_alloc_align(NextPowerOf2(sz));
        }
    }

    // Tree: unbalanced: favor speed on small blocks
    if (sz <= MAX_CONST_BLOCK_SIZE)
    {
        void* vv;
        if (PMP_IN_PARALLEL_REGION)
        {
#pragma omp critical
            {
                vv = getBlockAllocAlignInt(sz);
            }
        }
        else
        {
            vv = getBlockAllocAlignInt(sz);
        }
        //
        // #if defined(DEBUG) || defined(_OPENMP)
        //         int i = 0;
        //         if (PMP_IN_PARALLEL_REGION)
        //         {
        //             i++;
        //         }
        // #endif

        return vv;
    }
    //
    // #if defined(DEBUG) || defined(_OPENMP)
    //     int i = 0;
    //     if (PMP_IN_PARALLEL_REGION)
    //     {
    //         i++;
    //     }
    // #endif

    return nullptr;
}


/**
 * @copydoc StaticConstArray::freeBlockWithInt
 * @details Mirrors getBlockAllocInt()'s tree exactly. The two must agree: a
 *          block released through a different bucket than the one it came from
 *          corrupts that bucket's free list.
 */
inline void StaticConstArray::freeBlockWithInt(int sz, void* ptr)
{
    if constexpr (alloc_config::use_std_alloc)
    {
        int_free(ptr);
        return;
    }

    if (sz <= 8)
    {
        if (sz <= 4)
        {
            _Allocator4.free_elt_pow2(ptr);
            return;
        }
        else
        {
            _Allocator8.free_elt_pow2(ptr);
            return;
        }
    }
    else if (sz <= 128)
    {
        if (sz <= 32)
        {
            if (sz <= 16)
            {
                _Allocator16.free_elt_pow2(ptr);
                return;
            }
            else
            {
                _Allocator32.free_elt_pow2(ptr);
                return;
            }
        }
        else
        {
            if (sz <= 64)
            {
                _Allocator64.free_elt_pow2(ptr);
                return;
            }
            else
            {
                _Allocator128.free_elt_pow2(ptr);
                return;
            }
        }
    }
    else
    {
        if (sz <= 1024)
        {
            if (sz <= 256)
            {
                _Allocator256.free_elt_pow2(ptr);
                return;
            }
            else if (sz <= 512)
            {
                _Allocator512.free_elt_pow2(ptr);
                return;
            }
            else
            {
                _Allocator1024.free_elt_pow2(ptr);
                return;
            }
        }
        else
        {
            if (sz <= 2048)
            {
                _Allocator2048.free_elt_pow2(ptr);
            }
            else if (sz <= 4096)
            {
                _Allocator4096.free_elt_pow2(ptr);
            }
            else if (sz <= 8192)
            {
                _Allocator8192.free_elt_pow2(ptr);
            }
            else if (sz <= 16384)
            {
                _Allocator16384.free_elt_pow2(ptr);
            }
            else if (sz <= 32768)
            {
                _Allocator32768.free_elt_pow2(ptr);
            }
            else
            {
                _Allocator65536.free_elt_pow2(ptr);
            }
        }
    }
}

/**
 * @copydoc StaticConstArray::freeBlock
 * @details Outside a parallel region the block goes straight back to its
 *          bucket. Inside one it is batched instead, and the batch is flushed
 *          under a single critical section when it fills -- which is also the
 *          only moment @p ptr itself is released, since a full batch refuses it.
 */
inline bool StaticConstArray::freeBlock(int sz, void* ptr)
{
    if constexpr (alloc_config::use_std_alloc)
    {
        int_free(ptr);
        return true;
    }

    if constexpr (alloc_config::debug_mem_observable)
    {
        if (__debug_mem)
        {
            int_free(ptr);
            return true;
        }
    }


    if (sz <= MAX_CONST_BLOCK_SIZE)
    {
        // Batch per thread inside a region. Serially PMP_IN_PARALLEL_REGION is
        // a constant false and the batch was never allocated, so the branch
        // is dead; it still type-checks (rank 0, declarations present), which
        // is what lets this be plain code rather than a directive.
        if (alloc_config::use_openmp && PMP_IN_PARALLEL_REGION)
        {
            const int thr_rank = PMP_THREAD_RANK;

            const bool is_full = tab_info_free_elt.push_back(ptr, sz, thr_rank);

            if (is_full)
            {
#pragma omp critical
                {
                    freeBlockWithInt(sz, ptr);

                    clearInfoFreeElt(thr_rank);
                }

                tab_info_free_elt.reset(thr_rank);
            }
        }
        else
        {
            freeBlockWithInt(sz, ptr);
        }

        return true;
    }

    return false;
}

/// @copydoc StaticConstArray::freeBlockWithAlignInt
inline void StaticConstArray::freeBlockWithAlignInt(int sz, void* ptr)
{
    if constexpr (alloc_config::use_std_alloc)
    {
        int_free_align(ptr);
        return;
    }

    if (sz <= 128)
    {
// Stays a directive: __ALIGNEMENT__ is SIMD_SZ, and mirroring it as constexpr
// would make fast_allocators read a value the chunk layer owns.
#if __ALIGNEMENT__ == 16
        if (sz <= 16)
        {
            _Allocator16.free_elt_pow2(ptr);
            return;
        }
#elif __ALIGNEMENT__ >= 32
        if (sz <= 32)
        {
            _Allocator32.free_elt_pow2(ptr);
            return;
        }
#endif

        if (sz <= 64)
        {
            _Allocator64.free_elt_pow2(ptr);
            return;
        }
        else
        {
            _Allocator128.free_elt_pow2(ptr);
            return;
        }
    }
    else
    {
        if (sz <= 1024)
        {
            if (sz <= 256)
            {
                _Allocator256.free_elt_pow2(ptr);
                return;
            }
            else if (sz <= 512)
            {
                _Allocator512.free_elt_pow2(ptr);
                return;
            }
            else
            {
                _Allocator1024.free_elt_pow2(ptr);
                return;
            }
        }
        else
        {
            if (sz <= 2048)
            {
                _Allocator2048.free_elt_pow2(ptr);
            }
            else if (sz <= 4096)
            {
                _Allocator4096.free_elt_pow2(ptr);
            }
            else if (sz <= 8192)
            {
                _Allocator8192.free_elt_pow2(ptr);
            }
            else if (sz <= 16384)
            {
                _Allocator16384.free_elt_pow2(ptr);
            }
            else if (sz <= 32768)
            {
                _Allocator32768.free_elt_pow2(ptr);
            }
            else
            {
                _Allocator65536.free_elt_pow2(ptr);
            }
        }
    }
}

/// @copydoc StaticConstArray::freeBlockAlign
inline bool StaticConstArray::freeBlockAlign(int sz, void* ptr)
{
    if constexpr (alloc_config::use_std_alloc)
    {
        int_free_align(ptr);
        return true;
    }

    if constexpr (alloc_config::debug_mem_observable)
    {
        if (__debug_mem)
        {
            int_free_align(ptr);
            return true;
        }
    }


    if (sz <= MAX_CONST_BLOCK_SIZE)
    {
        // Same shape as freeBlock() above.
        if (alloc_config::use_openmp && PMP_IN_PARALLEL_REGION)
        {
            const int thr_rank = PMP_THREAD_RANK;

            const bool is_full = tab_info_free_elt_align.push_back(ptr, sz, thr_rank);

            if (is_full)
            {
#pragma omp critical
                {
                    freeBlockWithAlignInt(sz, ptr);

                    clearInfoFreeEltAlign(thr_rank);
                }

                tab_info_free_elt_align.reset(thr_rank);
            }
        }
        else
        {
            freeBlockWithAlignInt(sz, ptr);
        }

        return true;
    }

    return false;
}

/// @copydoc StaticConstArray::computeBlockSize
inline int StaticConstArray::computeBlockSize()
{
    if constexpr (alloc_config::use_std_alloc)
    {
        const bool std_alloc_compute_block_size_valid = false;
        assert(std_alloc_compute_block_size_valid);
    }


    return MAX_CONST_BLOCK_SIZE;
}
