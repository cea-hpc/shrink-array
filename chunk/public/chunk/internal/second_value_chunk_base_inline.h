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

#include <optional> // std::nullopt_t (the index-less overloads)

#include "chunk/internal/second_value_chunk_base.h" // the declarations these bodies belong to (self-contained, like the other *_inline.h)
#include "chunk/config.h"                       // build_config::any_check, chunk_config::use_compression_with_counter
#include "chunk/PMP_CHUNK.h"                    // get_thr_rank_fast
#include "chunk/internal/tab_cst_val2_holder.h" // SecondValueRef

namespace shrink_array
{

template <typename T, typename Derived>
template <typename IndexT>
void SecondValueChunkBase<T, Derived>::block_clear_common(this Derived&        self,
                                                          int                  num_compress_value,
                                                          IndexT               index_opt,
                                                          [[maybe_unused]] int thr_rank,
                                                          CstSlot*             ptr_type_cst,
                                                          SecondValueRef<T>    ptr_second_value,
                                                          [[maybe_unused]] T*  cst2_ptr)
{
    if constexpr (is_zero_any_family_derived)
    {
        // 1. Update the metadata BEFORE the CAS.
        if (num_compress_value == 0)
        {
            ptr_second_value.reset();
            self.set_cst_val_to_cst_val1(ptr_type_cst);
        }
        else
        {
            self.set_cst_val_to_cst_val2(ptr_type_cst);
        }

        // 2. The CAS installs the constant pointer AND returns the old heap pointer.
        T* old = self.try_mark_cst_and_get_old(num_compress_value, cst2_ptr);
        if (!old)
        {
            // CAS failed: another thread has already compressed this chunk.
            // Re-sync 'values' with the metadata written above.
            self.init_values_cst_val(ptr_type_cst, ptr_second_value); // Critical re-sync
            return;
        }

        // 3. Free the block ONLY after the CAS succeeded.
        self.Base::free_block_to_pool(thr_rank, old);
    }
    else
    {
        // Only the thread that wins the atomic transition frees; the others return.
        T* old = self.try_mark_cst_and_get_old(num_compress_value, index_opt);
        if (old)
        {
            self.Base::free_block_to_pool(thr_rank, old);
        }
        else
        {
            return;
        }
    }

    if constexpr (is_chunk2_derived && chunk_config::use_compression_with_counter)
    {
        self.reset_nb_val_eq_second_value();
        self.reset_nb_val_eq_first_value();
    }
}

template <typename T, typename Derived>
template <typename IndexT>
void SecondValueChunkBase<T, Derived>::clear_impl(this Derived&     self,
                                                  IndexT            index_opt,
                                                  int               thr_rank,
                                                  CstSlot*          ptr_type_cst,
                                                  SecondValueRef<T> ptr_second_value)
{
    bool is_written = false;
    if constexpr (std::is_same_v<IndexT, std::nullopt_t>)
    {
        is_written = self.written();
    }
    else
    {
        is_written = self.written(index_opt);
    }

    if (!is_written)
    {
        return;
    }

    if constexpr (build_config::any_check)
    {
        const auto* current_ptr = static_cast<const void*>(self.values_load());
        if (self.is_cst_val1())
        {
            invariant_failed(
                "Free List Corruption: Attempted to add 'cst_val1' (zero-constant) to the thread-local free pool. "
                "Type: {} | Chunk Addr: {} | Ptr being freed: {} | Cst1 Ptr: {} | Thread Rank: {}",
                type_name<T>(), &self, current_ptr, static_cast<const void*>(Base::get_cst_val1()),
                get_thr_rank_fast());
        }

        invariant_failed(
            "Free List Corruption: Attempted to add 'cst_val2' (secondary constant) to the thread-local free pool. "
            "Type: {} | Chunk Addr: {} | Ptr being freed: {} | Thread Rank: {}",
            type_name<T>(), &self, current_ptr, get_thr_rank_fast());
    }

    auto& ll_block_free_thr = self.get_manager().get_list_block_free_thr(thr_rank);
    ll_block_free_thr.push_back(self.values_.load());

    // zero_any specific cleanup
    if constexpr (std::is_same_v<IndexT, std::nullopt_t>)
    {
        ptr_second_value.reset();
        self.set_cst_val_to_cst_val1(ptr_type_cst);
        self.init_values_cst_val(ptr_type_cst, ptr_second_value);
    }
    else
    {
        // zero_one_value::Chunk specific cleanup
        self.set_cst_val_to_cst_val1();
        self.init_values_cst_val();

        if constexpr (is_chunk2_derived && chunk_config::use_compression_with_counter)
        {
            self.reset_nb_val_eq_second_value();
        }
    }
}

// ---------------------------------------------------------------------------
// shrink_array::SecondValueChunkBase<T, Derived>
// ---------------------------------------------------------------------------

template <typename T, typename Derived>
template <typename CasFunc>
void SecondValueChunkBase<T, Derived>::apply_cas_or_recup(T* values_loc, int thr_rank, CasFunc&& cas_fn) noexcept
{
    if constexpr (chunk_config::use_fast_block_compression)
    {
        // true: the installation lost the race.
        if (std::forward<CasFunc>(cas_fn)(values_loc))
        {
            // Publish the init before recycling the lost block (ARM).
            std::atomic_thread_fence(std::memory_order_release);
            Base::recup_value(values_loc, thr_rank);
        }
    }
    else
    {
        Base::recup_value(values_loc, 0);
    }
}

template <typename T, typename Derived>
void SecondValueChunkBase<T, Derived>::check_not_constant_before_clear(this const Derived& self,
                                                                       int                 num_compress_value,
                                                                       std::optional<int>  ind_opt) noexcept
{
    if constexpr (build_config::any_check)
    {
        const void* chunk_ptr  = static_cast<const void*>(&self);
        const void* values_ptr = static_cast<const void*>(self.values_load());
        const int   thr        = get_thr_rank_fast();

        // --- 1. Check primary constant (cst_val1) ---
        if (self.Base::is_cst_val1())
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
            // Path for zero_one_value (Table indexed)
            if constexpr (is_chunk2_derived)
            {
                const int ind = *ind_opt;
                if constexpr (is_bool)
                {
                    is_violating = self.is_cst_val2();
                    cst2_ptr     = static_cast<const void*>(Derived::BaseSpec::get_cst_val2());
                }
                else
                {
                    is_violating = self.is_cst_val2(ind);
                    cst2_ptr     = static_cast<const void*>(Derived::BaseSpec::get_cst_val2(ind));
                }
            }
        }
        else
        {
            // Path for the zero_any family or Bool (Non-indexed)
            if constexpr (is_bool || is_zero_any_family_derived)
            {
                is_violating = self.is_cst_val2();
                if constexpr (is_zero_any_family_derived)
                {
                    // A chunk compressed to a second constant reads from it
                    // through its values pointer; there is no other handle.
                    if (is_violating)
                    {
                        cst2_ptr = static_cast<const void*>(self.values_load());
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
}

template <typename T, typename Derived>
void SecondValueChunkBase<T, Derived>::assert_writable_buffer(this Derived&              self,
                                                              const T*                   target_ptr,
                                                              const std::source_location loc) noexcept
{
    if constexpr (build_config::any_check)
    {
        const auto* c1 = static_cast<const void*>(SecondValueChunkBase::get_cst_val1());

        const void* c2 = nullptr;

        // zero_any: the secondary constants live in one region -- the test is
        // a range test on the target, not a pointer to compare with.
        bool target_in_cst2_region = false;
        if constexpr (requires { self.tab_cst_val2_holder::is_cst_val2_ptr(target_ptr); })
        {
            target_in_cst2_region = self.tab_cst_val2_holder::is_cst_val2_ptr(target_ptr);
            if (target_in_cst2_region)
            {
                c2 = static_cast<const void*>(target_ptr);
            }
        }
        else if constexpr (requires { self.get_cst_val2(); })
        {
            // zero_one_value::Chunk bool: one shared second constant.
            c2 = static_cast<const void*>(self.get_cst_val2());
        }

        if (target_ptr == c1 || (c2 != nullptr && target_ptr == c2)) [[unlikely]]
        {
            invariant_failed(loc,
                             "Illegal Memory Write: Attempted to write into an immutable shared constant buffer. "
                             "Type: {} | Chunk Addr: {} | Target Ptr: {} | Cst1: {} | Cst2: {} | Thread: {}",
                             type_name<T>(), static_cast<const SecondValueChunkBase*>(std::addressof(self)),
                             static_cast<const void*>(target_ptr), c1, c2, get_thr_rank_fast());
        }
    }
}

} // namespace shrink_array
