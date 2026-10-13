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

// Out-of-line template method bodies for ArrayChunkBase<Derived>. Included at the
// end of array_chunk_base.h (after the class), so every TU that uses the class
// sees the definitions. Kept separate to keep array_chunk_base.h to declarations.
#include <cstdio>    // std::fprintf (set_chunk_size refusal)
#include <iterator>  // std::forward_iterator, std::distance (assign)
#include <stdexcept> // std::out_of_range (at())

#include "chunk/internal/array_chunk_base.h"      // ArrayChunkBase, the class these bodies belong to
#include "chunk/all_type_chunk.h"                 // AllTypeChunk::ensure_manager, from create
#include "chunk/config.h"                         // build_config::chunk_dynamic, chunk_config::CheckNaN
#include "chunk/internal/chunk_size_policy.h"     // register_runtime_chunk_size / set_thread_chunk_size
#include "chunk/internal/process_verbs.h"         // compress_now_and_trim_pools -- shrink_to_fit()
#include "chunk/PMP_CHUNK.h"                      // CHUNK_PMP_IN_PARALLEL_REGION
#include "chunk/manage_ompt.h"                    // parallel_phase -- write_range_by_chunks
#include "chunk/internal/parallel_storage_init.h" // use_own_team -- write_range_by_chunks

namespace shrink_array
{

template <typename Derived>
void ArrayChunkBase<Derived>::create(ind_type sz)
{
    // The first array of this element type registers its manager; every
    // later one pays a guard load. See AllTypeChunk.
    AllTypeChunk::ensure_manager<manager_type>();

    create_common();

    derived().rebuild_storage(sz);
}

template <typename Derived>
void ArrayChunkBase<Derived>::create_common()
{
    nb_chunk = 0;

    chunk_config::CheckNaN<chunk_config::do_check>::update(true);

    derived().pre_create_hook();

    // In ALL modes, do not gate on use_fast_block_compression: Proxy2A needs
    // its tab_* containers (tab_second_value, ...) even without OpenMP; for
    // Proxy / Proxy2 it is a no-op stub outside OpenMP.
    proxy_type::resize_tab_is_int(nb_tab);
}

template <typename Derived>
bool ArrayChunkBase<Derived>::is_written_at(int j) const noexcept
{
    if constexpr (has_chunk_written_v<Derived>)
    {
        // ArrayChunk2: Chunk2::written() needs the second-value index.
        return derived().chunk_written(j);
    }
    else
    {
        // ArrayChunk and ArrayChunk2A: the chunk knows by itself.
        return chunk_list[j].written();
    }
}

template <typename Derived>
void ArrayChunkBase<Derived>::set_chunk_size([[maybe_unused]] unsigned int new_chunk_size) noexcept
{
    // Same rule as rechunk_to() (this is also its path on an empty array):
    // a power of two or a multiple of 64 (is_valid_chunk_size). Refused,
    // geometry kept.
    if constexpr (build_config::chunk_dynamic)
    {
        if (!is_valid_chunk_size(new_chunk_size))
        {
            std::fprintf(stderr,
                         "[chunk] set_chunk_size(%u) refused: a chunk size must be a power of two or a multiple of "
                         "64\n",
                         new_chunk_size);
            std::fflush(stderr);
            return;
        }
    }
    size_policy_.set_size(new_chunk_size); // no-op under StaticChunkSize
    if constexpr (build_config::chunk_dynamic)
    {
        // Ensure the allocators for this size exist (sequential, idempotent) and
        // make it this thread's active size, so its allocations route there.
        register_runtime_chunk_size(new_chunk_size);
        set_thread_chunk_size(new_chunk_size);
    }
}

template <typename Derived>
int ArrayChunkBase<Derived>::get_nb_chunk_written() const
{
    int        nb_written = 0;
    const auto lbd        = [&](int j) { return static_cast<int>(is_written_at(j)); };

    // Inside a region: the calling thread counts them all (an orphaned
    // `omp for` would give each caller a partial count).
    if (CHUNK_PMP_IN_PARALLEL_REGION)
    {
        for (int j = 0; j < nb_chunk; ++j)
        {
            nb_written += lbd(j);
        }
    }
    else
    {
#pragma omp parallel for reduction(+ : nb_written)
        for (int j = 0; j < nb_chunk; ++j)
        {
            nb_written += lbd(j);
        }
    }
    return nb_written;
}

template <typename Derived>
int ArrayChunkBase<Derived>::get_nb_chunk_written_seq() const
{
    int nb_chunk_written = 0;
    for (int j = 0; j < nb_chunk; ++j)
    {
        nb_chunk_written += static_cast<int>(is_written_at(j));
    }
    return nb_chunk_written;
}

template <typename Derived>
double ArrayChunkBase<Derived>::stats_seq() const
{
    if (nb_chunk == 0)
    {
        return 0.0;
    }
    const int nb_chunk_written = get_nb_chunk_written_seq();
    return 100.0 * static_cast<double>(nb_chunk_written) / nb_chunk;
}

template <typename Derived>
template <typename ValueAt>
void ArrayChunkBase<Derived>::write_range_by_chunks(ind_type first, ind_type last, ValueAt&& value_at)
{
    if (first >= last)
    {
        return;
    }
    // In long long: (c + 1) * chunk_size() may pass ind_type's range near its top.
    const auto      csz     = static_cast<long long>(chunk_size());
    const int       c_first = chunk_index_of(first);
    const int       c_last  = chunk_index_of(last - 1) + 1;
    const auto      span    = [&](int c)
    {
        const long long lo_ll = static_cast<long long>(c) * csz;
        const long long hi_ll = lo_ll + csz;
        const ind_type  lo    = lo_ll > static_cast<long long>(first) ? static_cast<ind_type>(lo_ll) : first;
        const ind_type  hi    = hi_ll < static_cast<long long>(last) ? static_cast<ind_type>(hi_ll) : last;
        for (ind_type i = lo; i < hi; ++i)
        {
            (*this)[i] = value_at(i);
        }
    };

    if constexpr (build_config::use_openmp)
    {
        if (parallel_storage_init::use_own_team(static_cast<long>(c_last) - c_first))
        {
            parallel_phase(
                [&]
                {
#pragma omp for schedule(static)
                    for (int c = c_first; c < c_last; ++c)
                    {
                        span(c);
                    }
                });
            return;
        }
    }
    for (int c = c_first; c < c_last; ++c)
    {
        span(c);
    }
}

template <typename Derived>
void ArrayChunkBase<Derived>::assign(ind_type count, T val)
{
    derived().resize(count);
    write_range_by_chunks(0, count, [&val](ind_type) { return val; });
}

template <typename Derived>
template <std::forward_iterator It>
void ArrayChunkBase<Derived>::assign(It first, It last)
{
    const auto n = static_cast<ind_type>(std::distance(first, last));
    derived().resize(n);
    if constexpr (std::random_access_iterator<It>)
    {
        write_range_by_chunks(0, n, [&first](ind_type i) { return static_cast<T>(first[i]); });
    }
    else
    {
        ind_type i = 0;
        for (; first != last; ++first)
        {
            (*this)[i++] = static_cast<T>(*first);
        }
    }
}

template <typename Derived>
void ArrayChunkBase<Derived>::reserve(ind_type new_capacity)
{
    if (static_cast<size_type>(new_capacity) <= capacity())
    {
        return;
    }
    // resize() builds the storage; reserve() then restores the size.
    const ind_type keep = m_size;
    derived().resize(new_capacity);
    m_size = keep;
}

template <typename Derived>
void ArrayChunkBase<Derived>::copy_from(const Derived& from)
{
    const ind_type from_size = from.m_size;
    derived().resize(from_size);
    // Const reads of @p from (through its gate) from every thread of the team:
    // nothing writes it meanwhile.
    write_range_by_chunks(0, from_size, [&from](ind_type i) { return std::as_const(from)[i]; });
}

template <typename Derived>
void ArrayChunkBase<Derived>::shrink_to_fit()
{
    invalidate_proxies();
    compress_now_and_trim_pools();
}

template <typename Derived>
ArrayChunkBase<Derived>::const_reference ArrayChunkBase<Derived>::at(ind_type i) const
{
    if (i < 0 || static_cast<size_type>(i) >= static_cast<size_type>(m_size)) [[unlikely]]
    {
        throw std::out_of_range("ArrayChunk::at: index out of range");
    }
    return (*this)[i]; // const operator[] -> T (value)
}

template <typename Derived>
ArrayChunkBase<Derived>::reference ArrayChunkBase<Derived>::at(ind_type i)
{
    if (i < 0 || static_cast<size_type>(i) >= static_cast<size_type>(m_size)) [[unlikely]]
    {
        throw std::out_of_range("ArrayChunk::at: index out of range");
    }
    return (*this)[i]; // non-const operator[] -> proxy_type (write entry)
}

template <typename Derived>
ArrayChunkBase<Derived>::const_reference ArrayChunkBase<Derived>::front() const
{
    if (empty()) [[unlikely]]
    {
        throw std::out_of_range("ArrayChunk::front: container is empty");
    }
    return (*this)[0];
}

template <typename Derived>
ArrayChunkBase<Derived>::reference ArrayChunkBase<Derived>::front()
{
    if (empty()) [[unlikely]]
    {
        throw std::out_of_range("ArrayChunk::front: container is empty");
    }
    return (*this)[0];
}

template <typename Derived>
ArrayChunkBase<Derived>::const_reference ArrayChunkBase<Derived>::back() const
{
    if (empty()) [[unlikely]]
    {
        throw std::out_of_range("ArrayChunk::back: container is empty");
    }
    return (*this)[m_size - 1];
}

template <typename Derived>
ArrayChunkBase<Derived>::reference ArrayChunkBase<Derived>::back()
{
    if (empty()) [[unlikely]]
    {
        throw std::out_of_range("ArrayChunk::back: container is empty");
    }
    return (*this)[m_size - 1];
}

} // namespace shrink_array
