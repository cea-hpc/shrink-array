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

template <typename F>
decltype(auto) StaticConstArray::with_bucket(std::size_t bucket, F&& func)
{
    // One comparison per bucket, unrolled; the compiler is free to make it a
    // jump table.
    return [&]<std::size_t... I>(std::index_sequence<I...>) -> decltype(auto)
    {
        using R = decltype(func(std::get<0>(buckets())));
        if constexpr (std::is_void_v<R>)
        {
            (void)((bucket == I ? (func(std::get<I>(buckets())), true) : false) || ...);
        }
        else
        {
            R result{};
            (void)((bucket == I ? (result = func(std::get<I>(buckets())), true) : false) || ...);
            return result;
        }
    }(std::make_index_sequence<n_buckets>{});
}

/**
 * @copydoc StaticConstArray::get_block_alloc_int
 * @details The bucket is computed (bucket_index()), not searched for.
 */
inline void* StaticConstArray::get_block_alloc_int(unsigned int sz)
{
    if constexpr (alloc_config::use_std_alloc)
    {
        return int_alloc(sz);
    }

    return with_bucket(bucket_index(sz), [](auto& bucket) { return bucket.alloc(); });
}

inline void* StaticConstArray::get_block_alloc(unsigned int sz)
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
            vv = get_block_alloc_int(sz);
        }
        else
        {
            vv = get_block_alloc_int(sz);
        }
        return vv;
    }

    return nullptr;
}


/**
 * @copydoc StaticConstArray::free_block_with_int
 * @details Same bucket_index() as get_block_alloc_int(), which is what keeps the
 *          two in agreement: a block released through a different bucket than
 *          the one it came from corrupts that bucket's free list.
 */
inline void StaticConstArray::free_block_with_int(int sz, void* ptr)
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
inline bool StaticConstArray::try_free_block(int sz, void* ptr)
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
            // they were built, before main; std backend: a registered rank
            // past them, or none, -1, the registry full): there is no batch to
            // index, so free at once, under the lock the full path takes.
            if (thr_rank < 0 || thr_rank >= tab_info_free_elt().nb_thr) [[unlikely]]
            {
                const std::scoped_lock lock(ladder_mutex());
                free_block_with_int(sz, ptr);
                return true;
            }

            const bool is_full = tab_info_free_elt().push_back(ptr, sz, thr_rank);

            if (is_full)
            {
                {
                    const std::scoped_lock lock(ladder_mutex());
                    free_block_with_int(sz, ptr);

                    clear_info_free_elt(thr_rank);
                }

                tab_info_free_elt().reset(thr_rank);
            }
        }
        else
        {
            free_block_with_int(sz, ptr);
        }

        return true;
    }

    return false;
}

// ---------------------------------------------------------------------------
// fast_alloc::InfoFreeElt
// ---------------------------------------------------------------------------

inline bool InfoFreeElt::push_back(void* ptr, int sz, int thr_rank)
{
    int& ref_compt = compt[thr_rank];

    if ((ref_compt < max_free_elt && somme_sz_[thr_rank] < 4 * 1024 * 1024))
    {
        tab_ptr[thr_rank][ref_compt] = ptr;
        tab_sz[thr_rank][ref_compt]  = sz;

        somme_sz_[thr_rank] += sz;

        ref_compt++;
        return false;
    }

    return true;
}

// ---------------------------------------------------------------------------
// fast_alloc::BlkAllocator
// ---------------------------------------------------------------------------

inline void* BlkAllocator::alloc_block(size_t size)
{
    if constexpr (alloc_config::use_std_alloc)
    {
        return int_alloc(size);
    }
    else
    {
        // Compared in size_t BEFORE narrowing to unsigned int (4 GiB + 64
        // would become 64).
        void* ptr = size <= max_const_block_size ? get_block_alloc(static_cast<unsigned int>(size)) : nullptr;

        if (ptr == nullptr)
        {
            ptr = int_alloc(size);
        }

        return ptr;
    }
}

inline void BlkAllocator::free_block(void* ptr, size_t size)
{
    if (ptr == nullptr)
    {
        return;
    }

    if constexpr (alloc_config::use_std_alloc)
    {
        int_free(ptr);
    }
    else
    {
        // false: not a ladder size, so it came from int_alloc(). Test in
        // size_t first: static_cast<int> alone turns [2 GiB, 4 GiB) negative.
        if (size <= max_const_block_size && try_free_block(static_cast<int>(size), ptr))
        {
            return;
        }

        int_free(ptr);
    }
}

inline void* BlkAllocator::alloc_tab(size_t size)
{
    // size_t header: an int one would wrap past INT_MAX.
    static_assert(fast_alloc::alignment >= sizeof(std::size_t), "alloc_tab: the header must hold a size_t");
    // size + header must not wrap: near SIZE_MAX it would route to a small
    // ladder block that the caller then writes past.
    if (size > static_cast<std::size_t>(-1) - fast_alloc::alignment) [[unlikely]]
    {
        fast_alloc::out_of_memory("alloc_tab", size, 1, 0);
    }
    const std::size_t total = size + fast_alloc::alignment;

    auto* ptr = static_cast<std::byte*>(
        total <= max_const_block_size ? get_block_alloc(static_cast<unsigned int>(total)) : nullptr);

    if (ptr == nullptr)
    {
        ptr = static_cast<std::byte*>(int_alloc_align(total));
    }

    fast_alloc::store_to_bytes(ptr, size); // the header: the requested size

    return ptr + fast_alloc::alignment;
}

inline void BlkAllocator::free_tab(void* user_ptr)
{
    if (user_ptr == nullptr)
    {
        return;
    }

    std::byte* const m_ptr = static_cast<std::byte*>(user_ptr) - fast_alloc::alignment;

    const std::size_t total = fast_alloc::load_from_bytes<std::size_t>(m_ptr) + fast_alloc::alignment;

    // Same routing as alloc_tab(), decided in size_t.
    if (total > max_const_block_size || !try_free_block(static_cast<int>(total), m_ptr))
    {
        int_free_align(m_ptr);
    }
}

// ---------------------------------------------------------------------------
// fast_alloc::FixedSizeAllocator<ElemT>
// ---------------------------------------------------------------------------

template <typename ElemT>
void* FixedSizeAllocator<ElemT>::alloc_block()
{
    if constexpr (alloc_config::use_std_alloc)
    {
        return int_alloc(sizeof(ElemT));
    }
    else
    {
        void* ptr = get_block_alloc(static_cast<unsigned int>(sizeof(ElemT)));

        if (ptr == nullptr)
        {
            return int_alloc(sizeof(ElemT));
        }

        return ptr;
    }
}

template <typename ElemT>
void FixedSizeAllocator<ElemT>::free_block(void* ptr)
{
    if (ptr == nullptr)
    {
        return;
    }

    if constexpr (alloc_config::use_std_alloc)
    {
        int_free(ptr);
    }
    else
    {
        // try_free_block() returns false for a size the ladder does not serve:
        // the block came from int_alloc().
        if (try_free_block(static_cast<int>(sizeof(ElemT)), ptr))
        {
            return;
        }

        int_free(ptr);
    }
}

} // namespace fast_alloc
