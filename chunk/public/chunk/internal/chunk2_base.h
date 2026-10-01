
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

#include <atomic> // std::atomic_thread_fence, std::memory_order_release
#include <optional>
#include <source_location> // std::source_location
#include <type_traits>     // std::is_same_v
#include <utility>         // std::forward

#include "chunk/config.h"
#include "chunk/PMP_CHUNK.h" // get_thr_rank_fast (the constant-buffer write check)
#include "chunk/internal/chunk_base.h"
#include "chunk/internal/tab_cst_val2_holder.h" // SecondValueRef, the Chunk2A second-value slot handle
#include "chunk/type_name.h"

namespace shrink_array
{

template <typename Derived>
struct ProxyTypeTrait;

template <typename T>
class Chunk2;

template <typename T>
class Proxy2;

template <typename T>
class Chunk2A;

template <typename T>
class Proxy2A;

// Specialisation for Chunk2
template <typename T>
struct ProxyTypeTrait<Chunk2<T>>
{
    using Type = Proxy2<T>;
};

// Specialisation for Chunk2A
template <typename T>
struct ProxyTypeTrait<Chunk2A<T>>
{
    using Type = Proxy2A<T>;
};

// ============================================================================
// Chunk2Base: Base class for 2-value compression (Chunk2 and Chunk2A)
// Inherits common functionality from ChunkBase
// ============================================================================

template <typename T, typename Derived>
class Chunk2Base : public ChunkBase<T, Derived>
{
  public:
    using Base = ChunkBase<T, Derived>;
    using Base::derived;
    using Base::first_value;
    using Base::is_bool;
    using Base::is_cst_val1;
    using Base::is_not_bool;
    using Base::is_not_cst_val1;
    using Base::m_ptr_cst_val1;
    using Base::read_value;
    using Base::values;
    using Base::values_load;
    using Base::values_store;

    Chunk2Base()  = default;
    ~Chunk2Base() = default;

  protected:
    // ========================================================================
    // Chunk2 specific deallocation with block_clear
    // ========================================================================

    /// @brief Common block clear implementation for Chunk2/Chunk2A
    /// @tparam IndexType Type of index parameter (can be std::nullopt_t for Chunk2A)
    /// @param num_compress_value Which constant value to compress to (0 or 1)
    /// @param index_opt Optional index parameter (for Chunk2) or std::nullopt (for Chunk2A)
    /// @param thr_rank Thread rank
    /// @param ptr_type_cst Pointer to constant type indicator (Chunk2A only)
    /// @param ptr_second_value Pointer to second value (Chunk2A only)
    /// @param cst2_ptr The shared array to install for a type-2 compression
    ///        (Chunk2A only; the chunk keeps no copy of it)
    template <typename IndexType>
    void block_clear_common(int                  num_compress_value,
                            IndexType            index_opt,
                            [[maybe_unused]] int thr_rank,
                            char*                ptr_type_cst     = nullptr,
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
    void apply_cas_or_recup(T* values_loc, int thr_rank, CasFunc&& cas_fn) noexcept
    {
        if constexpr (chunk_config::use_fast_block_compression)
        {
            // Execute the CAS logic specific to the derived chunk type.
            // If it returns true, the installation failed (race condition).
            if (std::forward<CasFunc>(cas_fn)(values_loc))
            {
                // Publish the init before recycling the lost block (ARM).
                std::atomic_thread_fence(std::memory_order_release);
                Base::recup_value(values_loc, thr_rank);
            }
        }
        else
        {
            // Fallback for serial mode or non-fast compression builds.
            Base::recup_value(values_loc, 0);
        }
    }

    /// @brief Check-only guard of the teardown: a chunk about to have its
    ///        buffer freed must not be compressed to either constant (that
    ///        buffer would be the shared cst_val1 / cst_val2 block, not its
    ///        own). Fails the invariant otherwise.
    /// @param num_compress_value Index reported in the failure message.
    /// @param ind_opt Chunk2's second-value slot index; empty for Chunk2A and
    ///        bool, whose second value is not table-indexed.
    void check_not_constant_before_clear(int num_compress_value, std::optional<int> ind_opt) const noexcept
    {
        if constexpr (!build_config::any_check)
        {
            return;
        }

        const void* chunk_ptr  = static_cast<const void*>(this);
        const void* values_ptr = static_cast<const void*>(derived().values_load());
        const int   thr        = get_thr_rank_fast();

        // --- 1. Check primary constant (cst_val1) ---
        if (derived().Base::is_cst_val1())
        {
            invariant_failed(std::source_location::current(),
                             "Critical State Violation: Attempted to free 'cst_val1' inside clear. "
                             "Index: {} | Type: {} | Chunk: {} | Values: {} | Cst1: {} | Thr: {}",
                             num_compress_value, type_name<T>(), chunk_ptr, values_ptr,
                             static_cast<const void*>(Derived::Base::get_cst_val1()), thr);
        }

        // --- 2. Check secondary constant (cst_val2) ---
        bool        is_violating = false;
        const void* cst2_ptr     = nullptr;

        if (ind_opt.has_value())
        {
            // Path for Chunk2 (Table indexed)
            if constexpr (is_chunk2_derived)
            {
                const int ind = *ind_opt;
                if constexpr (is_bool)
                {
                    is_violating = derived().is_cst_val2();
                    cst2_ptr     = static_cast<const void*>(Derived::BaseSpec::get_cst_val2());
                }
                else
                {
                    is_violating = derived().is_cst_val2(ind);
                    cst2_ptr     = static_cast<const void*>(Derived::BaseSpec::get_cst_val2(ind));
                }
            }
        }
        else
        {
            // Path for Chunk2A or Bool (Non-indexed)
            if constexpr (is_bool || is_chunk2A_derived)
            {
                is_violating = derived().is_cst_val2();
                if constexpr (is_chunk2A_derived)
                {
                    // A chunk compressed to a second constant reads from it
                    // through its values pointer; there is no other handle.
                    if (is_violating)
                    {
                        cst2_ptr = static_cast<const void*>(derived().values_load());
                    }
                }
                else
                {
                    cst2_ptr = static_cast<const void*>(Derived::BaseSpec::get_cst_val2());
                }
            }
        }

        if (is_violating)
        {
            invariant_failed(std::source_location::current(),
                             "Critical State Violation: Attempted to free 'cst_val2' inside clear. "
                             "Index: {} | Type: {} | Chunk: {} | Values: {} | Cst2: {} | Thr: {}",
                             num_compress_value, type_name<T>(), chunk_ptr, values_ptr, cst2_ptr, thr);
        }
    }

  private:
    static constexpr bool is_chunk2A_derived = std::is_same_v<Derived, Chunk2A<T>>;
    static constexpr bool is_chunk2_derived  = std::is_same_v<Derived, Chunk2<T>>;

    /// @brief Clear chunk and reset to constant value
    /// @tparam IndexType Type of index parameter
    /// @param index_opt Optional index (Chunk2) or std::nullopt (Chunk2A)
    /// @param thr_rank Thread rank
    /// @param ptr_type_cst Constant type pointer (Chunk2A only)
    /// @param ptr_second_value Second value pointer (Chunk2A only)
    template <typename IndexType>
    void clear_impl(IndexType         index_opt,
                    int               thr_rank,
                    char*             ptr_type_cst     = nullptr,
                    SecondValueRef<T> ptr_second_value = {});

    /**
     * @brief Ensures the provided pointer does not target shared constant memory.
     * @param target_ptr The pointer about to be used for a write operation.
     * @param loc        Call site, reported if the check fails.
     */
    void assert_writable_buffer(const T*                   target_ptr,
                                const std::source_location loc = std::source_location::current()) noexcept
    {
        // Skip entirely if safety checks are disabled
        if constexpr (!build_config::any_check)
        {
            return;
        }

        const auto* c1 = static_cast<const void*>(Chunk2Base::get_cst_val1());

        const void* c2 = nullptr;

        // Chunk2A: the secondary constants live in one region -- the test is
        // a range test on the target, not a pointer to compare with.
        bool target_in_cst2_region = false;
        if constexpr (requires { derived().tab_cst_val2_holder::is_cst_val2_ptr(target_ptr); })
        {
            target_in_cst2_region = derived().tab_cst_val2_holder::is_cst_val2_ptr(target_ptr);
            if (target_in_cst2_region)
            {
                c2 = static_cast<const void*>(target_ptr);
            }
        }
        else if constexpr (requires { derived().get_cst_val2(); })
        {
            // Special case for Chunk2 (boolean or shared table)
            c2 = static_cast<const void*>(derived().get_cst_val2());
        }

        // Violation check: if target matches ANY immutable constant buffer
        if (target_ptr == c1 || (c2 != nullptr && target_ptr == c2)) [[unlikely]]
        {
            // Use the impl version to pass the captured location
            invariant_failed(loc,
                             "Illegal Memory Write: Attempted to write into an immutable shared constant buffer. "
                             "Type: {} | Chunk Addr: {} | Target Ptr: {} | Cst1: {} | Cst2: {} | Thread: {}",
                             type_name<T>(), static_cast<const void*>(this), static_cast<const void*>(target_ptr), c1,
                             c2, get_thr_rank_fast());
        }
    }
};

} // namespace shrink_array

#include "chunk/internal/chunk2_base_inline.h"
