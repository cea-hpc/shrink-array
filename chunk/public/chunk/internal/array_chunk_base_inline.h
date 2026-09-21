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
#include <limits>    // std::numeric_limits (max_size)
#include <stdexcept> // std::out_of_range (at())

#include "chunk/internal/array_chunk_base.h"
#include "chunk/internal/compression_manager.h" // CompressionManager<>::FlushReplayGuard in rechunk_refill
#include "chunk/internal/chunk_stats.h"         // increm_chunk_rechunk* (CHUNK_STATS)

namespace shrink_array
{

template <typename Derived>
void ArrayChunkBase<Derived>::create_common()
{
    nb_chunk = 0;

    chunk_config::CheckNaN<chunk_config::do_check>::update(true);

    derived().pre_create_hook();

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
void ArrayChunkBase<Derived>::set_value(ind_type i, in_t<T> v)
{
    const int num_chunk = i >> size_policy_.decalage();
    const int num       = i & size_policy_.modulo();

    auto& chunk_t = chunk_list[num_chunk];

    // COMPLEX CASES (ArrayChunk2 / ArrayChunk2A):
    // We use the Proxy because it handles decompression,
    // compression lists, and asynchronous operations.
    proxy_type p = chunk_t[num];

    // Arm ptr_on_array / num_chunk in ALL modes (see operator[] comment):
    // set_num_tab()/get_info_second_value() need ptr_on_array set even in
    // non-OpenMP builds, otherwise share_info lookups abort with
    // "key NOT FOUND".
    p.set_this_and_num_chunk(&derived(), num_chunk, num_tab);

    // Chunk2 only: arm the proxy with the address of this array's registry
    // mirror, so the write path copies the (second_value, ind, num_tab)
    // tuple instead of scanning share_info_array for it. Discriminated on
    // the DECLARED proxy type (chunk_traits.h), not on a `requires`: a
    // renamed member must fail to compile, not silently fall back to the
    // registry.
    if constexpr (std::is_same_v<proxy_type, Proxy2<T>>)
    {
        p.arm_inf_array(derived().registry_mirror());
    }

    // Assignment triggers Proxy::operator= which handles everything internally,
    // including calling derived().get_metadata(i) if needed
    p = v;
}

template <typename Derived>
ArrayChunkBase<Derived>::T ArrayChunkBase<Derived>::operator()(ind_type i) const noexcept
{
    const int   num_chunk = i >> size_policy_.decalage();
    const int   num       = i & size_policy_.modulo();
    const auto& chunk_t   = chunk_list[num_chunk];

    sync_chunk(chunk_t, num_chunk, num);

    // 2. Direct read from chunk
    return chunk_t.read_value(num);
}

template <typename Derived>
ArrayChunkBase<Derived>::T& ArrayChunkBase<Derived>::operator()(ind_type i) noexcept
{
    const int num_chunk = i >> size_policy_.decalage();
    const int num       = i & size_policy_.modulo();
    auto&     chunk_t   = chunk_list[num_chunk];

    // 2. Concurrency check
    sync_chunk(chunk_t, num_chunk, num);

    // 3. Return reference to physical storage
    return chunk_t(num);
}

template <typename Derived>
void ArrayChunkBase<Derived>::set_chunk_size([[maybe_unused]] unsigned int n) noexcept
{
    size_policy_.set_size(n); // no-op under StaticChunkSize
    if constexpr (build_config::chunk_dynamic)
    {
        // Multi-size support: make sure the value allocators for size n exist
        // (sequential, idempotent), and record n as this thread's active
        // size so later allocations on this thread route to the right pool.
        register_runtime_chunk_size(n);
        set_thread_chunk_size(n);
    }
}

template <typename Derived>
int ArrayChunkBase<Derived>::get_nb_chunk_written() const
{
    int  n   = 0;
    auto lbd = [&](int j) { return static_cast<int>(is_written_at(j)); };

    if (CHUNK_PMP_IN_PARALLEL_REGION)
    {
#pragma omp for schedule(static) nowait
        for (int j = 0; j < nb_chunk; ++j)
            n += lbd(j);
    }
    else
    {
#pragma omp parallel for reduction(+ : n)
        for (int j = 0; j < nb_chunk; ++j)
            n += lbd(j);
    }
    return n;
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
        return 0.0;
    const int nb_chunk_written = get_nb_chunk_written_seq();
    return 100.0 * static_cast<double>(nb_chunk_written) / nb_chunk;
}

template <typename Derived>
void ArrayChunkBase<Derived>::assign(ind_type count, in_t<T> val)
{
    derived().resize(count);
    for (ind_type i = 0; i < count; ++i)
        (*this)[i] = val;
}

template <typename Derived>
template <std::forward_iterator It>
void ArrayChunkBase<Derived>::assign(It first, It last)
{
    derived().resize(static_cast<ind_type>(std::distance(first, last)));
    ind_type i = 0;
    for (; first != last; ++first)
        (*this)[i++] = static_cast<T>(*first);
}

template <typename Derived>
void ArrayChunkBase<Derived>::reserve(ind_type n)
{
    if (static_cast<size_type>(n) <= capacity())
        return;
    // resize() builds the storage of n elements and sets the size to n; the
    // size is what reserve() leaves alone. resize() to n allocates
    // n / chunk_size + 1 chunks, so the capacity covers n afterwards.
    const ind_type keep = m_size;
    derived().resize(n);
    m_size = keep;
}

template <typename Derived>
void ArrayChunkBase<Derived>::copy_from(const Derived& from)
{
    const ind_type n = from.m_size;
    derived().resize(n);
    for (ind_type i = 0; i < n; ++i)
        (*this)[i] = std::as_const(from)[i];
}

template <typename Derived>
void ArrayChunkBase<Derived>::shrink_to_fit()
{
    compress_now_and_trim_pools();
}

template <typename Derived>
ArrayChunkBase<Derived>::size_type ArrayChunkBase<Derived>::max_size() const noexcept
{
    // Addressable element count is bounded by the (signed) index type, not by
    // size_type: i >> decalage and i & modulo run on ind_type internally.
    return static_cast<size_type>(std::numeric_limits<ind_type>::max());
}

template <typename Derived>
ArrayChunkBase<Derived>::const_reference ArrayChunkBase<Derived>::at(ind_type i) const
{
    if (i < 0 || static_cast<size_type>(i) >= static_cast<size_type>(m_size)) [[unlikely]]
        throw std::out_of_range("ArrayChunk::at: index out of range");
    return (*this)[i]; // const operator[] -> T (value)
}

template <typename Derived>
ArrayChunkBase<Derived>::reference ArrayChunkBase<Derived>::at(ind_type i)
{
    if (i < 0 || static_cast<size_type>(i) >= static_cast<size_type>(m_size)) [[unlikely]]
        throw std::out_of_range("ArrayChunk::at: index out of range");
    return (*this)[i]; // non-const operator[] -> proxy_type (write entry)
}

template <typename Derived>
ArrayChunkBase<Derived>::const_reference ArrayChunkBase<Derived>::front() const
{
    if (empty()) [[unlikely]]
        throw std::out_of_range("ArrayChunk::front: container is empty");
    return (*this)[0];
}

template <typename Derived>
ArrayChunkBase<Derived>::reference ArrayChunkBase<Derived>::front()
{
    if (empty()) [[unlikely]]
        throw std::out_of_range("ArrayChunk::front: container is empty");
    return (*this)[0];
}

template <typename Derived>
ArrayChunkBase<Derived>::const_reference ArrayChunkBase<Derived>::back() const
{
    if (empty()) [[unlikely]]
        throw std::out_of_range("ArrayChunk::back: container is empty");
    return (*this)[m_size - 1];
}

template <typename Derived>
ArrayChunkBase<Derived>::reference ArrayChunkBase<Derived>::back()
{
    if (empty()) [[unlikely]]
        throw std::out_of_range("ArrayChunk::back: container is empty");
    return (*this)[m_size - 1];
}

} // namespace shrink_array
