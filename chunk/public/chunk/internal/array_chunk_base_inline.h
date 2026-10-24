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
#include <iterator>  // std::forward_iterator, std::distance (assign)
#include <stdexcept> // std::out_of_range (at())

#include "chunk/internal/array_chunk_base.h"    // ArrayChunkBase, the class these bodies belong to
#include "chunk/internal/all_type_chunk_decl.h" // AllTypeChunk::ensure_manager, from create (bodies: all_type_chunk.h)
#include "chunk/config.h"                       // build_config::chunk_dynamic, chunk_config::CheckNaN
#include "chunk/internal/chunk_size_policy.h"   // chunk_size_error, register_runtime_chunk_size / set_thread_chunk_size
#include "chunk/internal/process_verbs.h"       // compress_now_and_trim_pools -- shrink_to_fit()
#include "chunk/PMP_CHUNK.h"                    // CHUNK_PMP_IN_PARALLEL_REGION, pmp::parallel_sum, pmp::current_team
#include "chunk/manage_ompt.h"                  // parallel_phase -- write_range_by_chunks
#include "chunk/internal/parallel_storage_init.h" // use_own_team -- write_range_by_chunks
#include "chunk/log.h"                            // CHUNK_LOG (set_chunk_size refusal)

namespace shrink_array
{

template <typename Derived>
void ArrayChunkBase<Derived>::create(this Derived& self, IndType sz)
{
    // The first array of this element type registers its manager; every
    // later one pays a guard load. See AllTypeChunk.
    AllTypeChunk::ensure_manager<ManagerType>();

    self.create_common();

    self.rebuild_storage(sz);
}

template <typename Derived>
void ArrayChunkBase<Derived>::create_common(this Derived& self)
{
    self.nb_chunk_ = 0;

    chunk_config::CheckNaN<chunk_config::do_check>::update(true);

    self.pre_create_hook();

    // In ALL modes, do not gate on use_fast_block_compression: zero_any::Proxy needs
    // its tab_* containers (tab_second_value, ...) even without OpenMP; for
    // Proxy / zero_one_value::Proxy it is a no-op stub outside OpenMP.
    ProxyType::resize_tab_is_int(self.nb_tab);
}

template <typename Derived>
bool ArrayChunkBase<Derived>::is_written_at(this const Derived& self, int num_chunk) noexcept
{
    if constexpr (has_chunk_written_v<Derived>)
    {
        // zero_one_value::Array: zero_one_value::Chunk::written() needs the second-value index.
        return self.chunk_written(num_chunk);
    }
    else
    {
        // zero::Array and zero_any::Array: the chunk knows by itself.
        return self.chunk_list_[num_chunk].written();
    }
}

template <typename Derived>
std::expected<void, ChunkSizeError>
ArrayChunkBase<Derived>::try_set_chunk_size([[maybe_unused]] unsigned int new_chunk_size) noexcept
{
    // Same rule as rechunk_to() (this is also its path on an empty array):
    // check_chunk_size, a power of two or a multiple of 64 and at most
    // cst_buffer_size -- a wider chunk reads the shared constant buffers out
    // of bounds. Refused, geometry kept. A static build has its compiled size only.
    if constexpr (build_config::chunk_dynamic)
    {
        if (const auto ok = check_chunk_size(new_chunk_size); !ok) [[unlikely]]
        {
            return ok;
        }
    }
    else if (new_chunk_size != static_chunk_size)
    {
        return std::unexpected(ChunkSizeError::fixed_geometry);
    }
    size_policy_.set_size(new_chunk_size); // no-op under StaticChunkSize
    if constexpr (build_config::chunk_dynamic)
    {
        // Ensure the allocators for this size exist (sequential, idempotent) and
        // make it this thread's active size, so its allocations route there.
        register_runtime_chunk_size(new_chunk_size);
        set_thread_chunk_size(new_chunk_size);
    }
    return {};
}

template <typename Derived>
void ArrayChunkBase<Derived>::set_chunk_size([[maybe_unused]] unsigned int new_chunk_size) noexcept
{
    // A static build ignores the call, as it always has (no message): the
    // geometry is the compiled one, whatever the caller asks.
    if (const auto set = try_set_chunk_size(new_chunk_size); !set && set.error() != ChunkSizeError::fixed_geometry)
        [[unlikely]]
    {
        warn_chunk_size_refused("set_chunk_size", new_chunk_size, set.error());
    }
}

template <typename Derived>
int ArrayChunkBase<Derived>::get_nb_chunk_written(this const Derived& self)
{
    int        nb_written = 0;
    const int  nb_chunk   = self.nb_chunk_;
    const auto lbd        = [&](int num_chunk) { return static_cast<int>(self.is_written_at(num_chunk)); };

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
        nb_written =
            pmp::parallel_sum<int, 1>(0, nb_chunk, pmp::Options{}, [&](int j, auto& acc) { acc[0] += lbd(j); })[0];
    }
    return nb_written;
}

template <typename Derived>
int ArrayChunkBase<Derived>::get_nb_chunk_written_seq(this const Derived& self)
{
    int nb_chunk_written = 0;
    for (int j = 0; j < self.nb_chunk_; ++j)
    {
        nb_chunk_written += static_cast<int>(self.is_written_at(j));
    }
    return nb_chunk_written;
}

template <typename Derived>
double ArrayChunkBase<Derived>::stats_seq(this const Derived& self)
{
    if (self.nb_chunk_ == 0)
    {
        return 0.0;
    }
    const int nb_chunk_written = self.get_nb_chunk_written_seq();
    return 100.0 * static_cast<double>(nb_chunk_written) / self.nb_chunk_;
}

template <typename Derived>
template <typename ValueAt>
void ArrayChunkBase<Derived>::write_range_by_chunks(this Derived& self, IndType first, IndType last, ValueAt&& value_at)
{
    if (first >= last)
    {
        return;
    }
    // In long long: (c + 1) * chunk_size() may pass IndType's range near its top.
    const auto csz     = static_cast<long long>(self.chunk_size());
    const int  c_first = self.chunk_index_of(first);
    const int  c_last  = self.chunk_index_of(last - 1) + 1;
    const auto span    = [&](int num_chunk)
    {
        const long long lo_ll = static_cast<long long>(num_chunk) * csz;
        const long long hi_ll = lo_ll + csz;
        const IndType   lo    = lo_ll > static_cast<long long>(first) ? static_cast<IndType>(lo_ll) : first;
        const IndType   hi    = hi_ll < static_cast<long long>(last) ? static_cast<IndType>(hi_ll) : last;
        for (IndType i = lo; i < hi; ++i)
        {
            self[i] = value_at(i);
        }
    };

    if constexpr (build_config::threaded)
    {
        if (parallel_storage_init::use_own_team(static_cast<long>(c_last) - c_first))
        {
            // The team parallel_phase opens is the calling thread's current one.
            parallel_phase([&] { pmp::current_team().for_static<int>(c_first, c_last, [&](int c) { span(c); }); });
            return;
        }
    }
    for (int c = c_first; c < c_last; ++c)
    {
        span(c);
    }
}

template <typename Derived>
void ArrayChunkBase<Derived>::assign(this Derived& self, IndType count, T val)
{
    self.resize(count);
    self.write_range_by_chunks(0, count, [&val](IndType) { return val; });
}

template <typename Derived>
template <std::forward_iterator It>
void ArrayChunkBase<Derived>::assign(this Derived& self, It first, It last)
{
    const auto count = static_cast<IndType>(std::distance(first, last));
    self.resize(count);
    if constexpr (std::random_access_iterator<It>)
    {
        self.write_range_by_chunks(0, count, [&first](IndType i) { return static_cast<T>(first[i]); });
    }
    else
    {
        IndType i = 0;
        for (; first != last; ++first)
        {
            self[i++] = static_cast<T>(*first);
        }
    }
}

template <typename Derived>
void ArrayChunkBase<Derived>::reserve(this Derived& self, IndType new_capacity)
{
    if (static_cast<size_type>(new_capacity) <= self.capacity())
    {
        return;
    }
    // resize() builds the storage; reserve() then restores the size.
    const IndType keep = self.m_size_;
    self.resize(new_capacity);
    self.m_size_ = keep;
}

template <typename Derived>
void ArrayChunkBase<Derived>::copy_from(this Derived& self, const Derived& from)
{
    const IndType from_size = from.m_size_;
    self.resize(from_size);
    // Const reads of @p from (through its gate) from every thread of the team:
    // nothing writes it meanwhile.
    self.write_range_by_chunks(0, from_size, [&from](IndType i) { return std::as_const(from)[i]; });
}

template <typename Derived>
void ArrayChunkBase<Derived>::shrink_to_fit()
{
    invalidate_proxies();
    compress_now_and_trim_pools();
}

template <typename Derived>
ArrayChunkBase<Derived>::const_reference ArrayChunkBase<Derived>::at(this const Derived& self, IndType index)
{
    if (index < 0 || static_cast<size_type>(index) >= static_cast<size_type>(self.m_size_)) [[unlikely]]
    {
        throw std::out_of_range("zero::Array::at: index out of range");
    }
    return self[index]; // const operator[] -> T (value)
}

template <typename Derived>
ArrayChunkBase<Derived>::reference ArrayChunkBase<Derived>::at(this Derived& self, IndType index)
{
    if (index < 0 || static_cast<size_type>(index) >= static_cast<size_type>(self.m_size_)) [[unlikely]]
    {
        throw std::out_of_range("zero::Array::at: index out of range");
    }
    return self[index]; // non-const operator[] -> ProxyType (write entry)
}

template <typename Derived>
ArrayChunkBase<Derived>::const_reference ArrayChunkBase<Derived>::front(this const Derived& self)
{
    if (self.empty()) [[unlikely]]
    {
        throw std::out_of_range("zero::Array::front: container is empty");
    }
    return self[0];
}

template <typename Derived>
ArrayChunkBase<Derived>::reference ArrayChunkBase<Derived>::front(this Derived& self)
{
    if (self.empty()) [[unlikely]]
    {
        throw std::out_of_range("zero::Array::front: container is empty");
    }
    return self[0];
}

template <typename Derived>
ArrayChunkBase<Derived>::const_reference ArrayChunkBase<Derived>::back(this const Derived& self)
{
    if (self.empty()) [[unlikely]]
    {
        throw std::out_of_range("zero::Array::back: container is empty");
    }
    return self[self.m_size_ - 1];
}

template <typename Derived>
ArrayChunkBase<Derived>::reference ArrayChunkBase<Derived>::back(this Derived& self)
{
    if (self.empty()) [[unlikely]]
    {
        throw std::out_of_range("zero::Array::back: container is empty");
    }
    return self[self.m_size_ - 1];
}

} // namespace shrink_array
