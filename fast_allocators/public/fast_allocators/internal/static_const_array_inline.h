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

// Bodies of StaticConstArray, declared in static_const_array.h, which includes
// this file at its end.

#include <cstddef>     // std::size_t
#include <mutex>       // std::scoped_lock (ladder_mutex)
#include <tuple>       // std::get (dependent calls, in with_bucket)
#include <type_traits> // std::is_void_v
#include <utility>     // std::index_sequence

#include "fast_allocators/internal/static_const_array.h" // the declarations these bodies belong to
#include "fast_allocators/internal/int_alloc.h"          // int_alloc, int_free
#include "fast_allocators/internal/alloc_config.h"       // alloc_config::use_std_alloc / threaded
#include "fast_allocators/internal/pmp_alloc.h" // FAST_ALLOC_PMP_IN_PARALLEL_REGION / FAST_ALLOC_PMP_THREAD_RANK

namespace fast_alloc
{

template <class F>
decltype(auto) StaticConstArray::with_bucket(std::size_t i, F&& f)
{
    // One comparison per bucket, unrolled; the compiler is free to make it a
    // jump table.
    return [&]<std::size_t... I>(std::index_sequence<I...>) -> decltype(auto)
    {
        using R = decltype(f(std::get<0>(buckets())));
        if constexpr (std::is_void_v<R>)
        {
            (void)((i == I ? (f(std::get<I>(buckets())), true) : false) || ...);
        }
        else
        {
            R result{};
            (void)((i == I ? (result = f(std::get<I>(buckets())), true) : false) || ...);
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
    if (sz <= max_const_block_size)
    {
        void* vv = nullptr;
        if (FAST_ALLOC_PMP_IN_PARALLEL_REGION)
        {
            const std::scoped_lock lock(ladder_mutex());
            vv = getBlockAllocInt(sz);
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
 *          under a single hold of ladder_mutex() when it fills -- which is also the
 *          only moment @p ptr itself is released, since a full batch refuses it.
 */
inline bool StaticConstArray::freeBlock(int sz, void* ptr)
{
    if constexpr (alloc_config::use_std_alloc)
    {
        int_free(ptr);
        return true;
    }


    if (sz <= max_const_block_size)
    {
        // Batch per thread inside a region. Serially FAST_ALLOC_PMP_IN_PARALLEL_REGION is a
        // constant false and the batch was never allocated: the branch is dead but still
        // type-checks, so it can be plain code rather than a directive.
        if (alloc_config::threaded && FAST_ALLOC_PMP_IN_PARALLEL_REGION)
        {
            const int thr_rank = FAST_ALLOC_PMP_THREAD_RANK;

            // A rank the batches were not sized for (the team grew after
            // they were built, before main; pool backend: a registered rank
            // past them, or none, -1, the registry full): there is no batch to
            // index, so free at once, under the lock the full path takes.
            if (thr_rank < 0 || thr_rank >= tab_info_free_elt().nb_thr) [[unlikely]]
            {
                const std::scoped_lock lock(ladder_mutex());
                freeBlockWithInt(sz, ptr);
                return true;
            }

            const bool is_full = tab_info_free_elt().push_back(ptr, sz, thr_rank);

            if (is_full)
            {
                {
                    const std::scoped_lock lock(ladder_mutex());
                    freeBlockWithInt(sz, ptr);

                    clearInfoFreeElt(thr_rank);
                }

                tab_info_free_elt().reset(thr_rank);
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

} // namespace fast_alloc
