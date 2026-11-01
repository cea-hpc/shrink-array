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

/**
 * @file chunk_exchange.h
 * @brief The state of a chunk, read and written through a stable interface:
 *        export_chunk() / import_chunk().
 *
 * @details What a transfer (host <-> device, a checkpoint on disk) needs of an
 *          array is, chunk by chunk: compressed to which value, or the buffer
 *          of its S elements. Inside the library that answer is encoded in
 *          where the chunk's values pointer points (the shared first-value
 *          block, an identifier inside the second-value region, a motif
 *          region, or a buffer of its own), an encoding that changes with the
 *          implementation. export_chunk() and import_chunk() are the one place
 *          that knows it: code outside the library reads and writes chunks
 *          through ChunkState, and keeps working whatever the encoding becomes.
 *
 *          What ChunkState promises of a materialised chunk's buffer:
 *          chunk_size() elements of T, contiguous, no header, element
 *          `index * chunk_size() + k` at position k; the last chunk of an array
 *          whose size is not a multiple of chunk_size() holds its trailing
 *          elements first.
 *
 *          Prototype: zero_any::Array only, T other than bool (its second
 *          value lives in a flag block, not the registry).
 *
 * @warning Serial contract, as fill_range() / copy_range(): call both from a
 *          quiescent point, after finalize_parallel_phase(), with no deferred
 *          write pending (export_chunk() reads the chunk raw, as the re-chunk
 *          refill does).
 */

#include <cstddef>     // std::size_t
#include <cstdint>     // std::uint8_t
#include <type_traits> // std::is_same_v

#include "chunk/array_chunk_range_ops.h"      // copy_range, assume_drained
#include "chunk/chunk_traits.h"               // is_zero_any_family_v, has_motif_state_v
#include "chunk/internal/chunk_size_policy.h" // set_thread_chunk_size
#include "chunk/internal/cst_slot.h"          // CstSlot
#include "chunk/config.h"                     // build_config::chunk_dynamic

namespace shrink_array
{

/// @brief What a chunk holds.
enum class ChunkKind : std::uint8_t
{
    first_value,  ///< Compressed to the first value (T{} on zero_any): no buffer.
    second_value, ///< Compressed to ChunkState::value: no buffer.
    materialised, ///< A buffer of its own: ChunkState::data.
};

/// @brief One chunk, as export_chunk() reads it and import_chunk() writes it.
template <typename T>
struct ChunkState
{
    ChunkKind kind  = ChunkKind::first_value; ///< What the chunk holds.
    T         value = T{};                    ///< The constant (first_value, second_value).
    /// materialised: the chunk's elements (see the file comment), valid until
    /// the array is written again; nullptr otherwise.
    const T* data = nullptr;
};

/// @brief The arrays export_chunk() / import_chunk() take (the prototype: zero_any).
template <typename Arr>
concept ExchangeableArray =
    is_zero_any_family_v<typename Arr::ChunkType> && !has_motif_state_v<typename Arr::ChunkType> &&
    !std::is_same_v<typename Arr::value_type, bool>;

/**
 * @brief The state of chunk @p index of @p arr.
 * @param arr   the array, at a quiescent point (see the file comment)
 * @param index the chunk, below arr.chunk_count()
 */
template <ExchangeableArray Arr>
[[nodiscard]] ChunkState<typename Arr::value_type> export_chunk(const Arr& arr, std::size_t index)
{
    using T         = Arr::value_type;
    using ChunkType = Arr::ChunkType;

    const auto& chunk = arr.chunk_ref(index);
    const T*    vals  = chunk.values_load(); // one load: the tests below and data read the same pointer
    if (ChunkType::is_constant_ptr(vals))
    {
        if (chunk.is_cst_val2_ptr_of(vals))
        {
            return {ChunkKind::second_value, chunk.second_value_from_ptr(vals), nullptr};
        }
        return {ChunkKind::first_value, static_cast<T>(ChunkType::first_value), nullptr};
    }
    return {ChunkKind::materialised, T{}, vals};
}

/**
 * @brief Give chunk @p index of @p arr the state @p state.
 * @details A second value is installed as a compression would commit it,
 *          without a buffer (the re-chunk refill's install); a materialised
 *          chunk is filled through copy_range(), which arms it.
 * @pre The chunk is still at the first value and was never written: @p arr is
 *      freshly built, at the chunk size the states were exported at.
 * @param arr   the array, at a quiescent point (see the file comment)
 * @param index the chunk, below arr.chunk_count()
 * @param state what the chunk is to hold
 */
template <ExchangeableArray Arr>
void import_chunk(Arr& arr, std::size_t index, const ChunkState<typename Arr::value_type>& state)
{
    using T         = Arr::value_type;
    using ChunkType = Arr::ChunkType;
    using ProxyType = Arr::ProxyType;

    const auto        chunk_sz = static_cast<std::size_t>(arr.chunk_size());
    const auto        size     = static_cast<std::size_t>(arr.size());
    const std::size_t begin    = index * chunk_sz;
    const std::size_t end      = begin + chunk_sz < size ? begin + chunk_sz : size;

    if constexpr (build_config::chunk_dynamic)
    {
        set_thread_chunk_size(static_cast<unsigned int>(chunk_sz));
    }

    switch (state.kind)
    {
        case ChunkKind::first_value:
            return; // the fresh chunk's state
        case ChunkKind::second_value:
        {
            if (state.value == static_cast<T>(ChunkType::first_value))
            {
                return;
            }
            // What a type-2 compression commits (find_or_set on the chunk's
            // second-value slot, type table, values pointer), minus the buffer:
            // array_chunk_rechunk_inline.h, rechunk_refill().
            const int  num_tab = arr.num_tab_;
            const auto ci      = static_cast<int>(index);
            auto&      chunk   = arr.chunk_ref(index);
            const auto psv     = ProxyType::BaseSpec::get_tab_second_value(num_tab, ci);
            CstSlot*   ptc     = &ProxyType::tab_type_cst_val[num_tab][index];
            (void)chunk.find_or_set_ind_cst_val2(state.value, psv);
            ChunkType::set_cst_val_to_cst_val2(ptc);
            chunk.init_values_cst_val(ptc, psv);
            return;
        }
        case ChunkKind::materialised:
            copy_range(arr, begin, end, state.data, assume_drained);
            return;
    }
}

} // namespace shrink_array
