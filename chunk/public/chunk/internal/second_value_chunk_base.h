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

#include <atomic>          // std::atomic_thread_fence, std::memory_order_release
#include <optional>        // std::optional
#include <source_location> // std::source_location
#include <type_traits>     // std::is_same_v
#include <utility>         // std::forward

#include "chunk/config.h"                       // build_config::any_check, chunk_config::use_fast_block_compression
#include "chunk/PMP_CHUNK.h"                    // get_thr_rank_fast (the constant-buffer write check)
#include "chunk/internal/chunk_base.h"          // ChunkBase
#include "chunk/internal/cst_slot.h"            // CstSlot (clear_impl's type-of-constant slot)
#include "chunk/internal/tab_cst_val2_holder.h" // SecondValueRef, the zero_any second-value slot handle
#include "chunk/type_name.h"                    // type_name<T>

namespace shrink_array
{

template <typename Derived>
struct ProxyTypeTrait;

namespace zero_one_value
{

template <typename T>
class Chunk;
template <typename T>
class Proxy;

} // namespace zero_one_value

namespace zero_any
{

template <typename T>
class Chunk;
template <typename T>
class Proxy;

} // namespace zero_any

namespace pattern
{

template <typename T>
class Chunk;
template <typename T>
class Proxy;

} // namespace pattern

// Specialisation for zero_one_value
template <typename T>
struct ProxyTypeTrait<zero_one_value::Chunk<T>>
{
    using Type = zero_one_value::Proxy<T>;
};

// Specialisation for zero_any
template <typename T>
struct ProxyTypeTrait<zero_any::Chunk<T>>
{
    using Type = zero_any::Proxy<T>;
};

// Specialisation for pattern
template <typename T>
struct ProxyTypeTrait<pattern::Chunk<T>>
{
    using Type = pattern::Proxy<T>;
};

// ============================================================================
// SecondValueChunkBase
// ============================================================================

/// @brief CRTP base of the two-value chunks (zero_one_value and zero_any), on top of ChunkBase.
/// @tparam T       Element type.
/// @tparam Derived The chunk type.
template <typename T, typename Derived>
class SecondValueChunkBase : public ChunkBase<T, Derived>
{
  public:
    using Base = ChunkBase<T, Derived>;
    using Base::first_value;
    using Base::is_bool;
    using Base::is_cst_val1;
    using Base::is_not_bool;
    using Base::is_not_cst_val1;
    using Base::m_ptr_cst_val1;
    using Base::read_value;
    using Base::values_;
    using Base::values_load;
    using Base::values_store;

    SecondValueChunkBase()                                       = default;
    ~SecondValueChunkBase()                                      = default;
    SecondValueChunkBase(const SecondValueChunkBase&)            = default;
    SecondValueChunkBase& operator=(const SecondValueChunkBase&) = default;
    SecondValueChunkBase(SecondValueChunkBase&&)                 = default;
    SecondValueChunkBase& operator=(SecondValueChunkBase&&)      = default;

  protected:
    // ========================================================================
    // zero_one_value specific deallocation with block_clear
    // ========================================================================

    /// @brief Common block clear implementation for zero_one_value/zero_any
    /// @tparam IndexT Type of index parameter (can be std::nullopt_t for zero_any)
    /// @param num_compress_value Which constant value to compress to (0 or 1)
    /// @param index_opt Optional index parameter (for zero_one_value) or std::nullopt (for zero_any)
    /// @param thr_rank Thread rank
    /// @param ptr_type_cst Pointer to constant type indicator (zero_any only)
    /// @param ptr_second_value Pointer to second value (zero_any only)
    /// @param cst2_ptr The shared array to install for a type-2 compression
    ///        (zero_any only; the chunk keeps no copy of it)
    template <typename IndexT>
    void block_clear_common(this Derived&        self,
                            int                  num_compress_value,
                            IndexT               index_opt,
                            [[maybe_unused]] int thr_rank,
                            CstSlot*             ptr_type_cst     = nullptr,
                            SecondValueRef<T>    ptr_second_value = {},
                            T*                   cst2_ptr         = nullptr);

    /**
     * @brief Atomically attempts to install a new buffer or recycles it on failure.
     * @tparam CasFunc Type of the CAS lambda/functor.
     * @param values_loc The newly allocated and initialized buffer.
     * @param thr_rank Rank of the thread performing the operation.
     * @param cas_fn A lambda that returns TRUE if the swap failed (pointer already changed).
     */
    template <typename CasFunc>
    void apply_cas_or_recup(T* values_loc, int thr_rank, CasFunc&& cas_fn) noexcept;

    /// @brief Check-only teardown guard: a chunk about to have its buffer freed
    ///        must not be compressed to either constant (the buffer would be the
    ///        shared cst_val1 / cst_val2 block).
    /// @param num_compress_value Index reported in the failure message.
    /// @param ind_opt zero_one_value's second-value slot index; empty for zero_any and
    ///        bool, whose second value is not table-indexed.
    void check_not_constant_before_clear(this const Derived& self,
                                         int                 num_compress_value,
                                         std::optional<int>  ind_opt) noexcept;

  private:
    static constexpr bool is_zero_any_family_derived = is_zero_any_family_v<Derived>;
    static constexpr bool is_chunk2_derived          = std::is_same_v<Derived, zero_one_value::Chunk<T>>;

    /// @brief Clear chunk and reset to constant value
    /// @tparam IndexT Type of index parameter
    /// @param index_opt Optional index (zero_one_value) or std::nullopt (zero_any)
    /// @param thr_rank Thread rank
    /// @param ptr_type_cst Constant type pointer (zero_any only)
    /// @param ptr_second_value Second value pointer (zero_any only)
    template <typename IndexT>
    void clear_impl(this Derived&     self,
                    IndexT            index_opt,
                    int               thr_rank,
                    CstSlot*          ptr_type_cst     = nullptr,
                    SecondValueRef<T> ptr_second_value = {});

    /**
     * @brief Ensures the provided pointer does not target shared constant memory.
     * @param target_ptr The pointer about to be used for a write operation.
     * @param loc        Call site, reported if the check fails.
     */
    void assert_writable_buffer(this Derived&              self,
                                const T*                   target_ptr,
                                const std::source_location loc = std::source_location::current()) noexcept;
};

} // namespace shrink_array

#include "chunk/internal/second_value_chunk_base_inline.h" // the template bodies
