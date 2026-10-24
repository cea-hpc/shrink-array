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

} // namespace shrink_array
