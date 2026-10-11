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

#include "chunk/internal/array_chunk_base.h"  // ArrayChunkBase, the class these bodies belong to
#include "chunk/config.h"                     // build_config::chunk_dynamic, chunk_config::CheckNaN
#include "chunk/internal/chunk_size_policy.h" // register_runtime_chunk_size / set_thread_chunk_size
#include "chunk/internal/process_verbs.h"     // compress_now_and_trim_pools -- shrink_to_fit()
#include "chunk/PMP_CHUNK.h"                  // CHUNK_PMP_IN_PARALLEL_REGION

namespace shrink_array
{

template <typename Derived>
void ArrayChunkBase<Derived>::create_common(this Derived& self)
{
    self.nb_chunk = 0;

    chunk_config::CheckNaN<chunk_config::do_check>::update(true);

    self.pre_create_hook();

    // resize_tab_is_int(nb_tab) must be called in ALL modes, not only when
    // use_fast_block_compression is on:
    //   - Proxy<T> / Proxy2<T> : real implementation in OpenMP mode, no-op
    //                            stub in non-OpenMP mode — safe to call.
    //   - Proxy2A<T>           : real implementation in BOTH modes. Proxy2A
    //                            needs its tab_* containers (tab_second_value,
    //                            tab_type_cst_val, ...) populated even without
    //                            OpenMP — there is no counter-mode fallback.
    // Gating this on use_fast_block_compression (false in non-OMP builds)
    // left Proxy2A's tab_second_value empty, so a later tab_second_value[0]
    // aborted with "__n < this->size()" at ArrayChunk2A creation.
    proxy_type::resize_tab_is_int(self.nb_tab);
}

template <typename Derived>
bool ArrayChunkBase<Derived>::is_written_at(this const Derived& self, int j) noexcept
{
    if constexpr (has_chunk_written_v<Derived>)
    {
        // ArrayChunk2: Chunk2::written() needs the second-value index.
        return self.chunk_written(j);
    }
    else
    {
        // ArrayChunk and ArrayChunk2A: the chunk knows by itself.
        return self.chunk_list[j].written();
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
        // Multi-size support: make sure the value allocators for size new_chunk_size exist
        // (sequential, idempotent), and record new_chunk_size as this thread's active
        // size so later allocations on this thread route to the right pool.
        register_runtime_chunk_size(new_chunk_size);
        set_thread_chunk_size(new_chunk_size);
    }
}

template <typename Derived>
int ArrayChunkBase<Derived>::get_nb_chunk_written(this const Derived& self)
{
    int        nb_written = 0;
    const int  nb_chunk   = self.nb_chunk;
    const auto lbd        = [&](int j) { return static_cast<int>(self.is_written_at(j)); };

    // Inside a region: the calling thread counts them all. An orphaned
    // `omp for` summed 1/N of the chunks into a per-thread total (no
    // reduction): every caller got a partial count.
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
int ArrayChunkBase<Derived>::get_nb_chunk_written_seq(this const Derived& self)
{
    int nb_chunk_written = 0;
    for (int j = 0; j < self.nb_chunk; ++j)
    {
        nb_chunk_written += static_cast<int>(self.is_written_at(j));
    }
    return nb_chunk_written;
}

template <typename Derived>
double ArrayChunkBase<Derived>::stats_seq(this const Derived& self)
{
    if (self.nb_chunk == 0)
    {
        return 0.0;
    }
    const int nb_chunk_written = self.get_nb_chunk_written_seq();
    return 100.0 * static_cast<double>(nb_chunk_written) / self.nb_chunk;
}

template <typename Derived>
void ArrayChunkBase<Derived>::assign(this Derived& self, ind_type count, T val)
{
    self.resize(count);
    for (ind_type i = 0; i < count; ++i)
    {
        self[i] = val;
    }
}

template <typename Derived>
template <std::forward_iterator It>
void ArrayChunkBase<Derived>::assign(this Derived& self, It first, It last)
{
    self.resize(static_cast<ind_type>(std::distance(first, last)));
    ind_type i = 0;
    for (; first != last; ++first)
    {
        self[i++] = static_cast<T>(*first);
    }
}

template <typename Derived>
void ArrayChunkBase<Derived>::reserve(this Derived& self, ind_type new_capacity)
{
    if (static_cast<size_type>(new_capacity) <= self.capacity())
    {
        return;
    }
    // resize() builds the storage of new_capacity elements and sets the size to it; the
    // size is what reserve() leaves alone. resize() to new_capacity allocates
    // new_capacity / chunk_size + 1 chunks, so the capacity covers it afterwards.
    const ind_type keep = self.m_size;
    self.resize(new_capacity);
    self.m_size = keep;
}

template <typename Derived>
void ArrayChunkBase<Derived>::copy_from(this Derived& self, const Derived& from)
{
    const ind_type from_size = from.m_size;
    self.resize(from_size);
    for (ind_type i = 0; i < from_size; ++i)
    {
        self[i] = std::as_const(from)[i];
    }
}

template <typename Derived>
void ArrayChunkBase<Derived>::shrink_to_fit()
{
    invalidate_proxies();
    compress_now_and_trim_pools();
}

template <typename Derived>
ArrayChunkBase<Derived>::const_reference ArrayChunkBase<Derived>::at(this const Derived& self, ind_type i)
{
    if (i < 0 || static_cast<size_type>(i) >= static_cast<size_type>(self.m_size)) [[unlikely]]
    {
        throw std::out_of_range("ArrayChunk::at: index out of range");
    }
    return self[i]; // const operator[] -> T (value)
}

template <typename Derived>
ArrayChunkBase<Derived>::reference ArrayChunkBase<Derived>::at(this Derived& self, ind_type i)
{
    if (i < 0 || static_cast<size_type>(i) >= static_cast<size_type>(self.m_size)) [[unlikely]]
    {
        throw std::out_of_range("ArrayChunk::at: index out of range");
    }
    return self[i]; // non-const operator[] -> proxy_type (write entry)
}

template <typename Derived>
ArrayChunkBase<Derived>::const_reference ArrayChunkBase<Derived>::front(this const Derived& self)
{
    if (self.empty()) [[unlikely]]
    {
        throw std::out_of_range("ArrayChunk::front: container is empty");
    }
    return self[0];
}

template <typename Derived>
ArrayChunkBase<Derived>::reference ArrayChunkBase<Derived>::front(this Derived& self)
{
    if (self.empty()) [[unlikely]]
    {
        throw std::out_of_range("ArrayChunk::front: container is empty");
    }
    return self[0];
}

template <typename Derived>
ArrayChunkBase<Derived>::const_reference ArrayChunkBase<Derived>::back(this const Derived& self)
{
    if (self.empty()) [[unlikely]]
    {
        throw std::out_of_range("ArrayChunk::back: container is empty");
    }
    return self[self.m_size - 1];
}

template <typename Derived>
ArrayChunkBase<Derived>::reference ArrayChunkBase<Derived>::back(this Derived& self)
{
    if (self.empty()) [[unlikely]]
    {
        throw std::out_of_range("ArrayChunk::back: container is empty");
    }
    return self[self.m_size - 1];
}

} // namespace shrink_array
