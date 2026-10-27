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

// ---------------------------------------------------------------------------
// shrink_array::ArrayChunkBase<Derived>::const_iterator
// ---------------------------------------------------------------------------

template <typename Derived>
[[nodiscard]] inline ArrayChunkBase<Derived>::const_iterator::reference
ArrayChunkBase<Derived>::const_iterator::operator*() const noexcept
{
    // chunk_ already sync'd by enter_chunk(): raw read, as chunk_t[num].
    check_no_write_since_entry();
    if constexpr (has_motif_state_v<ChunkType>)
    {
        return chunk_->read_value(num_, owner_->motif_offset_slot(num_chunk_));
    }
    else
    {
        return (*chunk_)[num_];
    }
}

template <typename Derived>
ArrayChunkBase<Derived>::const_iterator& ArrayChunkBase<Derived>::const_iterator::operator++() noexcept
{
    // lin_ is the ONLY thing operator==/!= compare: it must track the
    // cursor, or range-for runs past the logical end.
    ++lin_;
    ++num_;
    if (num_ > modulo_) [[unlikely]] // crossed a chunk boundary
    {
        num_ = 0;
        ++num_chunk_;
        enter_chunk(); // sync + repoint (skipped past the end)
    }
    return *this;
}

template <typename Derived>
ArrayChunkBase<Derived>::const_iterator& ArrayChunkBase<Derived>::const_iterator::operator--() noexcept
{
    --lin_;                     // see operator++: lin_ is what == and != read
    if (num_ == 0) [[unlikely]] // crossed a chunk boundary backwards
    {
        --num_chunk_;
        num_ = modulo_;
        enter_chunk();
    }
    else
    {
        --num_;
    }
    return *this;
}

template <typename Derived>
void ArrayChunkBase<Derived>::const_iterator::check_no_write_since_entry() const noexcept
{
    if constexpr (build_config::any_check && ChunkType::precise)
    {
        if (chunk_->read_gate_should_wait(num_)) [[unlikely]]
        {
            CHUNK_INVARIANT_FAILED("ArrayChunkBase::const_iterator: a write was deferred onto chunk {} of array {} "
                                   "during the iteration (offset {}); the iterator would not see it -- no other "
                                   "thread may write an array that is being iterated",
                                   num_chunk_, owner_, num_);
        }
    }
}

template <typename Derived>
void ArrayChunkBase<Derived>::const_iterator::enter_chunk() noexcept
{
    if (num_chunk_ >= 0 && static_cast<std::size_t>(num_chunk_) < owner_->chunk_list_.size()) [[likely]]
    {
        chunk_ = &owner_->chunk_list_[num_chunk_];
        owner_->sync_chunk(*chunk_, num_chunk_, num_); // once per chunk
    }
    else
    {
        chunk_ = nullptr; // end / rend sentinel: never dereferenced
    }
}

template <typename Derived>
void ArrayChunkBase<Derived>::const_iterator::check_same_owner([[maybe_unused]] const const_iterator& lhs,
                                                               [[maybe_unused]] const const_iterator& rhs) noexcept
{
    if constexpr (build_config::any_check)
    {
        if (lhs.owner_ != rhs.owner_) [[unlikely]]
        {
            CHUNK_INVARIANT_FAILED("ArrayChunkBase::const_iterator: iterators of two different arrays compared "
                                   "or subtracted ({} vs {})",
                                   lhs.owner_, rhs.owner_);
        }
    }
}

// ---------------------------------------------------------------------------
// shrink_array::ArrayChunkBase<Derived>
// ---------------------------------------------------------------------------

template <typename Derived>
void ArrayChunkBase<Derived>::push_back(this Derived& self, InT<T> val)
{
    if (static_cast<size_type>(self.m_size_) < self.capacity()) [[likely]]
    {
        ++self.m_size_;
    }
    else
    {
        self.resize(self.m_size_ + 1);
    }
    self[self.m_size_ - 1] = val;
}

template <typename Derived>
template <typename F>
void ArrayChunkBase<Derived>::for_each_value(IndType first, IndType last, F&& func) const
{
    if constexpr (bulk_fast)
    {
        std::array<T, bulk_block> buf;
        for_each_chunk_span(first, last,
                            [&](const ChunkType& ch, int num_chunk, int lo, int hi)
                            {
                                for (int j = lo; j < hi; j += bulk_block)
                                {
                                    const int len = (hi - j) < bulk_block ? (hi - j) : bulk_block;
                                    read_chunk_block(ch, num_chunk, j, len, buf.data());
                                    check_no_write_during_bulk_read(ch, num_chunk, j);
                                    for (int k = 0; k < len; ++k)
                                    {
                                        func(buf[k]);
                                    }
                                }
                            });
    }
    else
    {
        if (first < 0)
        {
            first = 0;
        }
        if (last > m_size_)
        {
            last = m_size_;
        }
        if (first >= last)
        {
            return;
        }
        // begin() takes `this const Derived&` (deducing this): not callable on the base.
        const Derived&       d      = static_cast<const Derived&>(*this);
        const const_iterator end_it = d.begin() + last;
        for (const_iterator it = d.begin() + first; it != end_it; ++it)
        {
            func(*it);
        }
    }
}

template <typename Derived>
template <typename F>
void ArrayChunkBase<Derived>::for_each_span(IndType first, IndType last, F&& func) const
{
    std::array<T, bulk_block> buf;
    if constexpr (bulk_fast)
    {
        for_each_chunk_span(first, last,
                            [&](const ChunkType& ch, int num_chunk, int lo, int hi)
                            {
                                for (int j = lo; j < hi; j += bulk_block)
                                {
                                    const int len = (hi - j) < bulk_block ? (hi - j) : bulk_block;
                                    if constexpr (std::invocable<F&, const T&, std::size_t>)
                                    {
                                        // A compressed chunk stays so for the rest of the
                                        // call (no writer): the constant covers [j, hi).
                                        T cst{};
                                        if (read_chunk_block_or_constant(ch, num_chunk, j, len, buf.data(), cst))
                                        {
                                            check_no_write_during_bulk_read(ch, num_chunk, j);
                                            func(std::as_const(cst), static_cast<std::size_t>(hi - j));
                                            return;
                                        }
                                    }
                                    else
                                    {
                                        read_chunk_block(ch, num_chunk, j, len, buf.data());
                                    }
                                    check_no_write_during_bulk_read(ch, num_chunk, j);
                                    func(std::span<const T>(buf.data(), static_cast<std::size_t>(len)));
                                }
                            });
    }
    else
    {
        int filled = 0;
        for_each_value(first, last,
                       [&](const T& value)
                       {
                           buf[filled++] = value;
                           if (filled == bulk_block)
                           {
                               func(std::span<const T>(buf.data(), static_cast<std::size_t>(filled)));
                               filled = 0;
                           }
                       });
        if (filled > 0)
        {
            func(std::span<const T>(buf.data(), static_cast<std::size_t>(filled)));
        }
    }
}

template <typename Derived>
template <typename OutIt>
OutIt ArrayChunkBase<Derived>::read_range(IndType first, IndType last, OutIt out) const
{
    if constexpr (bulk_fast && std::is_same_v<OutIt, T*>)
    {
        for_each_chunk_span(first, last,
                            [&](const ChunkType& ch, int num_chunk, int lo, int hi)
                            {
                                for (int j = lo; j < hi; j += bulk_block)
                                {
                                    const int len = (hi - j) < bulk_block ? (hi - j) : bulk_block;
                                    read_chunk_block(ch, num_chunk, j, len, out);
                                    check_no_write_during_bulk_read(ch, num_chunk, j);
                                    out += len;
                                }
                            });
        return out;
    }
    else
    {
        for_each_value(first, last,
                       [&out](const T& value)
                       {
                           *out = value;
                           ++out;
                       });
        return out;
    }
}

template <typename Derived>
void ArrayChunkBase<Derived>::swap(this Derived& self, Derived& other) noexcept
{
    if constexpr (build_config::any_check)
    {
        if (self.size_policy_.size() != other.size_policy_.size())
        {
            CHUNK_INVARIANT_FAILED("zero::Array::swap: mismatched chunk geometry ({} vs {})", self.size_policy_.size(),
                                   other.size_policy_.size());
        }
    }
    self.swap_storage_and_identity(other);
}

template <typename Derived>
CHUNK_FLATTEN CHUNK_FORCE_INLINE ArrayChunkBase<Derived>::T
ArrayChunkBase<Derived>::operator()(this const Derived& self, IndType index) noexcept
{
    const int   num_chunk = self.chunk_index_of(index);
    const int   num       = self.offset_in_chunk(index);
    const auto& chunk_t   = self.chunk_list_[num_chunk];

    self.sync_chunk(chunk_t, num_chunk, num);

    if constexpr (has_motif_state_v<ChunkType>)
    {
        return chunk_t.read_value(num, self.motif_offset_slot(num_chunk));
    }
    else
    {
        return chunk_t.read_value(num);
    }
}

template <typename Derived>
CHUNK_FLATTEN CHUNK_FORCE_INLINE ArrayChunkBase<Derived>::T
ArrayChunkBase<Derived>::operator[](this const Derived& self, IndType index) noexcept
{
    const int num_chunk = self.chunk_index_of(index);
    const int num       = self.offset_in_chunk(index);
    // Lets the compiler drop bounds/sign checks (0 <= num <= modulo by construction).
    // modulo() is read into a local first: clang ignores an assumption that
    // calls a function it cannot prove side-effect free (-Wassume).
    [[maybe_unused]] const int chunk_mod = static_cast<int>(self.size_policy_.modulo());
    CHUNK_ASSUME(num_chunk >= 0);
    CHUNK_ASSUME(num >= 0 && num <= chunk_mod);
    const auto& chunk_t = self.chunk_list_[num_chunk];

    self.sync_chunk(chunk_t, num_chunk, num);

    if constexpr (has_motif_state_v<ChunkType>)
    {
        return chunk_t.read_value(num, self.motif_offset_slot(num_chunk));
    }
    else
    {
        return chunk_t[num];
    }
}

template <typename Derived>
CHUNK_FLATTEN CHUNK_FORCE_INLINE ArrayChunkBase<Derived>::reference
ArrayChunkBase<Derived>::operator[](this Derived& self, IndType index) noexcept
{
    if constexpr (build_config::strict_element_ref)
    {
        return reference(self.element_proxy(index));
    }
    else
    {
        return self.element_proxy(index);
    }
}

template <typename Derived>
CHUNK_FLATTEN CHUNK_FORCE_INLINE ArrayChunkBase<Derived>::ProxyType
ArrayChunkBase<Derived>::element_proxy(this Derived& self, IndType index) noexcept
{
    const int        num_chunk = self.chunk_index_of(index);
    auto&            chunk_t   = self.chunk_list_[num_chunk];
    const int        num       = self.offset_in_chunk(index);
    const ProxyType& val       = chunk_t[num];

    // Arm ptr_on_array / num_chunk in ALL modes, not only under fast block
    // compression: in non-OpenMP builds too, set_num_tab() and
    // get_info_second_value() look ptr_on_array up in share_info_array.
    val.set_this_and_num_chunk(self.registry_key(), num_chunk, self.num_tab_);
    val.arm_storage_generation(&self.storage_generation_); // check builds only

    // zero_one_value only: arm the proxy with this array's registry mirror, so the
    // write path copies (second_value, ind, num_tab) instead of scanning
    // share_info_array. Tested on the DECLARED proxy type, not a `requires`:
    // a renamed member must fail to compile, not fall back silently.
    if constexpr (std::is_same_v<ProxyType, zero_one_value::Proxy<T>>)
    {
        val.arm_inf_array(self.registry_mirror());
    }

    // No sync on the write entry: the read entries pull visibility
    // (ChunkBase::reader_sync), and a same-offset WAW is handled on the
    // direct write path. The const accessors and T& operator() keep sync_chunk.

    chunk_config::CheckNaN<chunk_config::do_check>::maybe(val, static_cast<long>(index));

    return val;
}

template <typename Derived>
void ArrayChunkBase<Derived>::swap_storage_and_identity(this Derived& self, Derived& other)
{
    if (&self == &other)
    {
        return;
    }
    self.flush_pending_deferred_ops(/*force=*/true);
    other.flush_pending_deferred_ops(/*force=*/true);
    self.invalidate_proxies();
    other.invalidate_proxies();

    using std::swap;
    swap(self.chunk_list_, other.chunk_list_);
    swap(self.alloc_key_, other.alloc_key_);
    self.swap_dirty_bitset(other);
    swap(self.m_size_, other.m_size_);
    swap(self.nb_chunk_, other.nb_chunk_);
    swap(self.size_policy_, other.size_policy_);
    self.swap_identity(other);
}

template <typename Derived>
template <typename Fn>
void ArrayChunkBase<Derived>::for_each_chunk_span(IndType first, IndType last, Fn&& fn) const
{
    if (first < 0)
    {
        first = 0;
    }
    if (last > m_size_)
    {
        last = m_size_;
    }
    if (first >= last)
    {
        return;
    }
    const int  c_first = chunk_index_of(first);
    const int  c_last  = chunk_index_of(last - 1);
    const auto full    = static_cast<int>(size_policy_.size());
    for (int c = c_first; c <= c_last; ++c)
    {
        const int        lo = c == c_first ? offset_in_chunk(first) : 0;
        const int        hi = c == c_last ? offset_in_chunk(last - 1) + 1 : full;
        const ChunkType& ch = chunk_list_[static_cast<std::size_t>(c)];
        // sync_chunk takes the derived array (deducing this).
        static_cast<const Derived&>(*this).sync_chunk(ch, c, lo);
        fn(ch, c, lo, hi);
    }
}

template <typename Derived>
void ArrayChunkBase<Derived>::read_chunk_block(const ChunkType& ch, int num_chunk, int first, int count, T* out)
    const noexcept
{
    if constexpr (has_motif_state_v<ChunkType>)
    {
        ch.read_block(first, count, out, motif_offset_slot(num_chunk));
    }
    else
    {
        ch.read_block(first, count, out);
    }
}

template <typename Derived>
bool ArrayChunkBase<Derived>::read_chunk_block_or_constant(const ChunkType& ch,
                                                           int              num_chunk,
                                                           int              first,
                                                           int              count,
                                                           T*               out,
                                                           T&               cst) const noexcept
{
    if constexpr (has_motif_state_v<ChunkType>)
    {
        return ch.read_block_or_constant(first, count, out, cst, motif_offset_slot(num_chunk));
    }
    else
    {
        return ch.read_block_or_constant(first, count, out, cst);
    }
}

template <typename Derived>
void ArrayChunkBase<Derived>::check_no_write_during_bulk_read([[maybe_unused]] const ChunkType& ch,
                                                              [[maybe_unused]] int              num_chunk,
                                                              [[maybe_unused]] int              offset) const noexcept
{
    if constexpr (build_config::any_check && ChunkType::precise)
    {
        if (ch.read_gate_should_wait(offset)) [[unlikely]]
        {
            CHUNK_INVARIANT_FAILED("ArrayChunkBase bulk read: a write was deferred onto chunk {} of array {} "
                                   "during the read (offset {}); it would not be seen -- no other thread may "
                                   "write an array that is being read in bulk",
                                   num_chunk, this, offset);
        }
    }
}

template <typename Derived>
CHUNK_FORCE_INLINE void ArrayChunkBase<Derived>::sync_chunk(this const Derived& self,
                                                            const ChunkType&    chunk_t,
                                                            int                 num_chunk,
                                                            int                 offset) noexcept
{
    if constexpr (build_config::threaded)
    {
        // Coarse pre-filter: the deferred-enqueue path marks this bit with
        // the CHUNK index (proxy_base_inline.h), so a clear bit means no
        // deferred op was ever queued for this chunk: skip the gate. This
        // keeps the hot read path cheap. The bit is sticky until
        // clear_dirty_bitset(); the authoritative gate is the chunk's own
        // pending counter, below.
        if (self.is_chunk_dirty(num_chunk)) [[unlikely]]
        {
            // precise per-chunk gate (DeferredReadGate, chunk_base.h).
            if (chunk_t.read_gate_should_wait(offset)) [[unlikely]]
            {
                chunk_t.wait_for_availability(offset);
            }
        }
    }
}

} // namespace shrink_array
