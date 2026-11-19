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

#include <optional>        // std::nullopt_t (the index-less overloads)
#include <atomic>          // std::atomic_thread_fence, std::memory_order_release
#include <utility>         // std::forward
#include <source_location> // std::source_location

#include "chunk/internal/second_value_chunk_base.h" // the declarations these bodies belong to (self-contained, like the other *_inline.h)
#include "chunk/config.h"                         // build_config::any_check, chunk_config::use_compression_with_counter
#include "chunk/internal/pmp_chunk.h"             // get_thr_rank_fast
#include "chunk/internal/second_value_registry.h" // SecondValueRef

namespace shrink_array
{

template <typename T, typename Derived>
template <typename IndexT>
void SecondValueChunkBase<T, Derived>::block_clear_common(int                  const_index,
                                                          IndexT               index_opt,
                                                          [[maybe_unused]] int thr_rank,
                                                          ConstSlot*           ptr_type_const,
                                                          SecondValueRef<T>    ptr_second_value,
                                                          [[maybe_unused]] T*  second_ptr)
{
    if constexpr (is_zero_any_family_derived)
    {
        // 1. Update the metadata BEFORE the CAS.
        if (const_index == 0)
        {
            if constexpr (has_motif_state_v<Derived>)
            {
                ptr_second_value.reset_release(); // a reader of the old motif may load the slot (F5)
            }
            else
            {
                ptr_second_value.reset();
            }
            derived().set_to_first_value(ptr_type_const);
        }
        else
        {
            derived().set_to_second_value(ptr_type_const);
        }

        // 2. The CAS installs the constant pointer AND returns the old heap pointer.
        T* old = derived().try_mark_const_and_get_old(const_index, second_ptr);
        if (!old)
        {
            // CAS failed: another thread has already compressed this chunk.
            // Re-sync 'values' with the metadata written above.
            derived().init_values_const_val(ptr_type_const, ptr_second_value); // Critical re-sync
            return;
        }

        // 3. Free the block ONLY after the CAS succeeded.
        Base::free_block_to_pool(thr_rank, old);
    }
    else
    {
        // Only the thread that wins the atomic transition frees; the others return.
        T* old = derived().try_mark_const_and_get_old(const_index, index_opt);
        if (old)
        {
            Base::free_block_to_pool(thr_rank, old);
        }
        else
        {
            return;
        }
    }

    if constexpr (is_zero_or_value_derived && chunk_config::use_compression_with_counter)
    {
        derived().reset_n_val_eq_second_value();
        derived().reset_n_val_eq_first_value();
    }
}

template <typename T, typename Derived>
template <typename IndexT>
void SecondValueChunkBase<T, Derived>::clear_impl(IndexT            index_opt,
                                                  int               thr_rank,
                                                  ConstSlot*        ptr_type_const,
                                                  SecondValueRef<T> ptr_second_value)
{
    bool is_written = false;
    if constexpr (std::is_same_v<IndexT, std::nullopt_t>)
    {
        is_written = derived().written();
    }
    else
    {
        is_written = derived().written(index_opt);
    }

    if (!is_written)
    {
        return;
    }

    if constexpr (build_config::any_check)
    {
        const auto* current_ptr = static_cast<const void*>(values_load());
        if (is_first_value())
        {
            invariant_failed(
                "Free List Corruption: Attempted to add 'first_value' (zero-constant) to the thread-local free pool. "
                "Type: {} | Chunk Addr: {} | Ptr being freed: {} | First value ptr: {} | Thread Rank: {}",
                type_name<T>(), this, current_ptr, static_cast<const void*>(Base::first_value_ptr()),
                get_thr_rank_fast());
        }

        invariant_failed(
            "Free List Corruption: Attempted to add 'second_value' (secondary constant) to the thread-local free pool. "
            "Type: {} | Chunk Addr: {} | Ptr being freed: {} | Thread Rank: {}",
            type_name<T>(), this, current_ptr, get_thr_rank_fast());
    }

    auto& ll_block_free_thr = derived().get_manager().get_list_block_free_thr(thr_rank);
    ll_block_free_thr.push_back(values_.load());

    // zero_any specific cleanup
    if constexpr (std::is_same_v<IndexT, std::nullopt_t>)
    {
        ptr_second_value.reset();
        derived().set_to_first_value(ptr_type_const);
        derived().init_values_const_val(ptr_type_const, ptr_second_value);
    }
    else
    {
        // zero_or_value specific cleanup
        derived().set_to_first_value();
        derived().init_values_const_val();

        if constexpr (is_zero_or_value_derived && chunk_config::use_compression_with_counter)
        {
            derived().reset_n_val_eq_second_value();
        }
    }
}

// ---------------------------------------------------------------------------
// shrink_array::SecondValueChunkBase<T, Derived>
// ---------------------------------------------------------------------------

template <typename T, typename Derived>
template <typename CasFunc>
void SecondValueChunkBase<T, Derived>::apply_cas_or_reclaim(T* values_loc, int thr_rank, CasFunc&& cas_fn) noexcept
{
    if constexpr (chunk_config::use_fast_block_compression)
    {
        // true: the installation lost the race.
        if (std::forward<CasFunc>(cas_fn)(values_loc))
        {
            // Publish the init before recycling the lost block (ARM).
            std::atomic_thread_fence(std::memory_order_release);
            Base::reclaim_value(values_loc, thr_rank);
        }
    }
    else
    {
        Base::reclaim_value(values_loc, 0);
    }
}

template <typename T, typename Derived>
void SecondValueChunkBase<T, Derived>::check_not_constant_before_clear(int                const_index,
                                                                       std::optional<int> idx_opt) const noexcept
{
    if constexpr (build_config::any_check)
    {
        const void* chunk_ptr  = static_cast<const void*>(this);
        const void* values_ptr = static_cast<const void*>(derived().values_load());
        const int   thr        = get_thr_rank_fast();

        // --- 1. Check primary constant (first_value) ---
        if (derived().Base::is_first_value())
        {
            invariant_failed(std::source_location::current(),
                             "Critical State Violation: Attempted to free 'first_value' inside clear. "
                             "Index: {} | Type: {} | Chunk: {} | Values: {} | First value: {} | Thr: {}",
                             const_index, type_name<T>(), chunk_ptr, values_ptr,
                             static_cast<const void*>(Derived::Base::first_value_ptr()), thr);
        }

        // --- 2. Check secondary constant (second_value) ---
        bool        is_violating = false;
        const void* second_ptr   = nullptr;

        if (idx_opt.has_value())
        {
            // Path for zero_or_value (Table indexed)
            if constexpr (is_zero_or_value_derived)
            {
                const int idx = *idx_opt;
                if constexpr (is_bool)
                {
                    is_violating = derived().is_second_value();
                    second_ptr   = static_cast<const void*>(Derived::BaseSpec::second_value_ptr());
                }
                else
                {
                    is_violating = derived().is_second_value(idx);
                    second_ptr   = static_cast<const void*>(Derived::BaseSpec::second_value_ptr(idx));
                }
            }
        }
        else
        {
            // Path for the zero_any family or Bool (Non-indexed)
            if constexpr (is_bool || is_zero_any_family_derived)
            {
                is_violating = derived().is_second_value();
                if constexpr (is_zero_any_family_derived)
                {
                    // A chunk compressed to a second constant reads from it
                    // through its values pointer; there is no other handle.
                    if (is_violating)
                    {
                        second_ptr = static_cast<const void*>(derived().values_load());
                    }
                }
                else
                {
                    second_ptr = static_cast<const void*>(Derived::BaseSpec::second_value_ptr());
                }
            }
        }

        if (is_violating)
        {
            invariant_failed(std::source_location::current(),
                             "Critical State Violation: Attempted to free 'second_value' inside clear. "
                             "Index: {} | Type: {} | Chunk: {} | Values: {} | Second value: {} | Thr: {}",
                             const_index, type_name<T>(), chunk_ptr, values_ptr, second_ptr, thr);
        }
    }
}

template <typename T, typename Derived>
void SecondValueChunkBase<T, Derived>::assert_writable_buffer(const T*                   target_ptr,
                                                              const std::source_location loc) noexcept
{
    if constexpr (build_config::any_check)
    {
        const auto* c1 = static_cast<const void*>(SecondValueChunkBase::first_value_ptr());

        const void* c2 = nullptr;

        // zero_any: the secondary constants live in one region -- the test is
        // a range test on the target, not a pointer to compare with.
        bool target_in_second_value_region = false;
        if constexpr (requires { derived().second_value_registry::is_second_value_ptr(target_ptr); })
        {
            target_in_second_value_region = derived().second_value_registry::is_second_value_ptr(target_ptr);
            if (target_in_second_value_region)
            {
                c2 = static_cast<const void*>(target_ptr);
            }
        }
        else if constexpr (requires { derived().second_value_ptr(); })
        {
            // zero_or_value bool: one shared second constant.
            c2 = static_cast<const void*>(derived().second_value_ptr());
        }

        if (target_ptr == c1 || (c2 != nullptr && target_ptr == c2)) [[unlikely]]
        {
            invariant_failed(
                loc,
                "Illegal Memory Write: Attempted to write into an immutable shared constant buffer. "
                "Type: {} | Chunk Addr: {} | Target Ptr: {} | First value: {} | Second value: {} | Thread: {}",
                type_name<T>(), this, static_cast<const void*>(target_ptr), c1, c2, get_thr_rank_fast());
        }
    }
}

} // namespace shrink_array
