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

#pragma once

// Bodies of StaticConstArray, declared in static_const_array.h, which includes
// this file at its end.

#include <cassert>     // assert
#include <cstddef>     // std::size_t
#include <tuple>       // std::get
#include <type_traits> // std::is_void_v
#include <utility>     // std::index_sequence

#include "fast_allocators/internal/static_const_array.h"        // the declarations these bodies belong to
#include "fast_allocators/internal/int_alloc.h"                 // int_alloc, int_free, int_alloc_align, int_free_align
#include "fast_allocators/internal/alloc_config.h"              // constexpr mirrors of STD_ALLOC / DEBUG
#include "fast_allocators/internal/pmp_alloc.h"                 // PMP_IN_PARALLEL_REGION / PMP_THREAD_RANK
#include "fast_allocators/internal/static_const_array_define.h" // MAX_CONST_BLOCK_SIZE
#include "fast_allocators/internal/svalloc.h"                   // __ALIGNEMENT__

template <class F>
decltype(auto) StaticConstArray::with_bucket(std::size_t i, F&& f)
{
    // One comparison per bucket, unrolled; the compiler is free to make it a
    // jump table.
    return [&]<std::size_t... I>(std::index_sequence<I...>) -> decltype(auto)
    {
        using R = decltype(f(std::get<0>(buckets)));
        if constexpr (std::is_void_v<R>)
        {
            (void)((i == I ? (f(std::get<I>(buckets)), true) : false) || ...);
        }
        else
        {
            R result{};
            (void)((i == I ? (result = f(std::get<I>(buckets)), true) : false) || ...);
            return result;
        }
    }(std::make_index_sequence<n_buckets>{});
}

/**
 * @copydoc StaticConstArray::getBlockAllocInt
 * @details The bucket is computed (bucket_index()), not searched for.
 */
inline void* StaticConstArray::getBlockAllocInt(unsigned int sz)
{
    if constexpr (alloc_config::use_std_alloc)
    {
        return int_alloc(sz);
    }

    return with_bucket(bucket_index(sz), [](auto& bucket) { return bucket.alloc(); });
}

inline void* StaticConstArray::getBlockAlloc(unsigned int sz)
{
    if constexpr (alloc_config::use_std_alloc)
    {
        return int_alloc(sz);
    }

    // Tree: unbalanced: favor speed on small blocks
    if (sz <= MAX_CONST_BLOCK_SIZE)
    {
        void* vv = nullptr;
        if (PMP_IN_PARALLEL_REGION)
        {
#pragma omp critical(fast_alloc_ladder)
            {
                vv = getBlockAllocInt(sz);
            }
        }
        else
        {
            vv = getBlockAllocInt(sz);
        }
        return vv;
    }

    return nullptr;
}

/**
 * @copydoc StaticConstArray::getBlockAllocAlignInt
 * @details Which small classes are skipped depends on `__ALIGNEMENT__`: see
 *          aligned_bucket_index().
 */
inline void* StaticConstArray::getBlockAllocAlignInt(int sz)
{
    if constexpr (alloc_config::use_std_alloc)
    {
        return int_alloc_align(sz);
    }

    return with_bucket(aligned_bucket_index(static_cast<unsigned int>(sz)),
                       [](auto& bucket) { return bucket.alloc(); });
}

inline void* StaticConstArray::getBlockAllocAlign(int sz)
{
    if constexpr (alloc_config::use_std_alloc)
    {
        return int_alloc_align(sz);
    }

    // Tree: unbalanced: favor speed on small blocks
    if (sz <= MAX_CONST_BLOCK_SIZE)
    {
        void* vv = nullptr;
        if (PMP_IN_PARALLEL_REGION)
        {
#pragma omp critical(fast_alloc_ladder)
            {
                vv = getBlockAllocAlignInt(sz);
            }
        }
        else
        {
            vv = getBlockAllocAlignInt(sz);
        }
        return vv;
    }
    return nullptr;
}


/**
 * @copydoc StaticConstArray::freeBlockWithInt
 * @details Same bucket_index() as getBlockAllocInt(), which is what keeps the
 *          two in agreement: a block released through a different bucket than
 *          the one it came from corrupts that bucket's free list.
 */
inline void StaticConstArray::freeBlockWithInt(int sz, void* ptr)
{
    if constexpr (alloc_config::use_std_alloc)
    {
        int_free(ptr);
        return;
    }

    with_bucket(bucket_index(static_cast<unsigned int>(sz)), [ptr](auto& bucket) { bucket.free_elt_pow2(ptr); });
}

/**
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


    if (sz <= MAX_CONST_BLOCK_SIZE)
    {
        // Batch per thread inside a region. Serially PMP_IN_PARALLEL_REGION is
        // a constant false and the batch was never allocated, so the branch
        // is dead; it still type-checks (rank 0, declarations present), which
        // is what lets this be plain code rather than a directive.
        if (alloc_config::use_openmp && PMP_IN_PARALLEL_REGION)
        {
            const int thr_rank = PMP_THREAD_RANK;

            // A rank the batches were not sized for (the team grew after
            // they were built, before main): there is no batch to index, so
            // free at once, under the critical section the full path takes.
            if (thr_rank >= tab_info_free_elt.nb_thr) [[unlikely]]
            {
#pragma omp critical(fast_alloc_ladder)
                freeBlockWithInt(sz, ptr);
                return true;
            }

            const bool is_full = tab_info_free_elt.push_back(ptr, sz, thr_rank);

            if (is_full)
            {
#pragma omp critical(fast_alloc_ladder)
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
/// @details Same aligned_bucket_index() as getBlockAllocAlignInt().
inline void StaticConstArray::freeBlockWithAlignInt(int sz, void* ptr)
{
    if constexpr (alloc_config::use_std_alloc)
    {
        int_free_align(ptr);
        return;
    }

    with_bucket(aligned_bucket_index(static_cast<unsigned int>(sz)),
                [ptr](auto& bucket) { bucket.free_elt_pow2(ptr); });
}

inline bool StaticConstArray::freeBlockAlign(int sz, void* ptr)
{
    if constexpr (alloc_config::use_std_alloc)
    {
        int_free_align(ptr);
        return true;
    }


    if (sz <= MAX_CONST_BLOCK_SIZE)
    {
        // Same shape as freeBlock() above.
        if (alloc_config::use_openmp && PMP_IN_PARALLEL_REGION)
        {
            const int thr_rank = PMP_THREAD_RANK;

            // Same bound as freeBlock(): no batch for this rank, free at once.
            if (thr_rank >= tab_info_free_elt_align.nb_thr) [[unlikely]]
            {
#pragma omp critical(fast_alloc_ladder)
                freeBlockWithAlignInt(sz, ptr);
                return true;
            }

            const bool is_full = tab_info_free_elt_align.push_back(ptr, sz, thr_rank);

            if (is_full)
            {
#pragma omp critical(fast_alloc_ladder)
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

inline int StaticConstArray::computeBlockSize()
{
    if constexpr (alloc_config::use_std_alloc)
    {
        const bool std_alloc_compute_block_size_valid = false;
        assert(std_alloc_compute_block_size_valid);
    }


    return MAX_CONST_BLOCK_SIZE;
}
