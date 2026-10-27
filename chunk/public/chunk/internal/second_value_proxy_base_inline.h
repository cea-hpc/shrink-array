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

#include <utility> // std::forward

#include "chunk/internal/second_value_proxy_base.h" // the declarations these bodies belong to
#include "chunk/internal/in_t.h"                    // InT<T>

namespace shrink_array
{

/**
 * @brief Core unified logic for 2-value compression marking.
 * Handles both standard and 'pre' checks using variadic args for second_value/first_value.
 */

template <typename Derived, typename T, typename ChunkT>
template <typename... Args>
void SecondValueProxyBase<Derived, T, ChunkT>::mark_for_compression_core(InT<T> old_value,
                                                                         InT<T> new_value,
                                                                         InT<T> expected_in_memory,
                                                                         bool   isv,
                                                                         Args&&... compression_args) const
{
    // Check if transition is valid for compression OR if memory changed unexpectedly (Race)
    // isv (Is Valid) guards the entire check for performance.
    if (isv && (derived().should_mark_for_compression(old_value, new_value, std::forward<Args>(compression_args)...) ||
                (Base::load_value() != expected_in_memory))) [[unlikely]]
    {
        add_to_compression_list();
    }
}

// ---------------------------------------------------------------------------
// shrink_array::SecondValueProxyBase<Derived, T, ChunkT>
// ---------------------------------------------------------------------------

template <typename Derived, typename T, typename ChunkT>
void SecondValueProxyBase<Derived, T, ChunkT>::mark_for_compression_if_needed(bool new_value) const
{
    const bool isv = is_cst_val_new_value_is_other_cst_val(new_value, false, true);
    // Base::read_values() (qualified: dependent base), never values[num]:
    // it is the write path's read entry and the only one that resolves a
    // compressed chunk's constant (an unbacked identifier slot); indexing
    // reads a zero page and the decision below is taken against 0.
    mark_for_compression_impl(Base::read_values(), new_value, isv, false, true);
}

template <typename Derived, typename T, typename ChunkT>
void SecondValueProxyBase<Derived, T, ChunkT>::mark_for_compression_before_store(InT<T>   second_val,
                                                                                 InT<T>   new_value,
                                                                                 InT<T>   first_value,
                                                                                 const T* vals) const
{
    // isv false makes the whole test a no-op: skip the read of the old
    // value too, which mark_for_compression_core() would not look at.
    if (!is_cst_val_new_value_is_other_cst_val(new_value, first_value, second_val))
    {
        return;
    }
    const T old = Base::read_values_at(vals);
    if (derived().should_mark_for_compression(old, new_value, second_val, first_value) ||
        (Base::read_values_at(vals) != old)) [[unlikely]]
    {
        add_to_compression_list();
    }
}

template <typename Derived, typename T, typename ChunkT>
template <typename WriteFn>
[[nodiscard]] CHUNK_FORCE_INLINE bool
SecondValueProxyBase<Derived, T, ChunkT>::write_with_optional_lock_with_set_ptr(WriteFn write_fn) noexcept
{
    const auto lbd_info = [this]
    {
        if constexpr (chunk_config::use_fast_block_compression)
        {
            derived().set_num_tab_ptr_is_inside_chunk(); // common initialization required for the fast-path
        }
        else
        {
            // The capture exists for the branch above only. A SERIAL build
            // never instantiates it and clang reports the capture as unused
            // (-Wunused-lambda-capture). Discarding `this` here keeps the
            // capture explicit; do not switch to [&], which hides the intent.
            static_cast<void>(this);
        }
    };

    return Base::write_with_optional_lock_impl([&]() noexcept { write_fn(); }, lbd_info);
}

template <typename Derived, typename T, typename ChunkT>
[[nodiscard]] bool
SecondValueProxyBase<Derived, T, ChunkT>::is_cst_val_new_value_is_other_cst_val(InT<T> new_value,
                                                                                InT<T> first_value,
                                                                                InT<T> second_val) const noexcept
{
    // CRTP dispatch: each variant owns its fast pre-check. zero_one_value keeps the
    // "a constant is written" test below; zero_any widens it (see
    // zero_any::Proxy::pre_check_mark_for_compression) because its
    // should_mark_for_compression() accepts a case this test would reject.
    //
    // INVARIANT: pre_check_mark_for_compression() must never return false
    // for a write that should_mark_for_compression() would accept. `isv` is
    // only a performance filter; if it rejects a qualifying write the chunk
    // is silently never queued for compression and its buffer is never
    // returned to the reuse list.
    return derived().pre_check_mark_for_compression(new_value, first_value, second_val);
}

} // namespace shrink_array