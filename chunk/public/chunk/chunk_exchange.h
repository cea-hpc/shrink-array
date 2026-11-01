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
 *          array is, chunk by chunk: compressed to which value, a motif and its
 *          offset, or the buffer of its S elements. Inside the library that
 *          answer is encoded in where the chunk's values pointer points (the
 *          shared first-value block, zero_one_value's second-value block, an
 *          identifier inside zero_any's second-value region, a motif of
 *          pattern's registry, or a buffer of its own), an encoding that
 *          changes with the implementation. export_chunk() and import_chunk()
 *          are the one place that knows it: code outside the library reads and
 *          writes chunks through ChunkState, and keeps working whatever the
 *          encoding becomes.
 *
 *          What ChunkState promises of a chunk's elements (materialised: data;
 *          motif: data and value): chunk_size() elements of T, contiguous, no
 *          header, element `index * chunk_size() + k` at position k; the last
 *          chunk of an array whose size is not a multiple of chunk_size() holds
 *          its trailing elements first.
 *
 *          Every variant: zero, zero_one_value, zero_any and pattern. Not yet:
 *          bool (zero_any keeps its second value in a flag block, zero_one_value
 *          one shared block) and ArrayChunkBits.
 *
 * @warning Serial contract, as fill_range() / copy_range(): call both from a
 *          quiescent point, after finalize_parallel_phase(), with no deferred
 *          write pending (export_chunk() reads the chunk raw, as the re-chunk
 *          refill does).
 */

#include <cstddef>     // std::size_t
#include <cstdint>     // std::uint8_t
#include <type_traits> // std::is_same_v, std::is_integral_v, std::make_unsigned_t
#include <vector>      // std::vector (a motif chunk's elements, import_chunk)

#include "chunk/array_chunk_range_ops.h"      // copy_range, fill_range, assume_drained, internal::is_v*_v
#include "chunk/bit.h"                        // is_bool_like_v
#include "chunk/chunk_traits.h"               // is_zero_any_family_v, has_motif_state_v
#include "chunk/internal/chunk_size_policy.h" // set_thread_chunk_size
#include "chunk/internal/cst_slot.h"          // CstSlot
#include "chunk/internal/relaxed_element.h"   // load_element_relaxed (a motif chunk's offset)
#include "chunk/config.h"                     // build_config::chunk_dynamic

namespace shrink_array
{

/// @brief What a chunk holds.
enum class ChunkKind : std::uint8_t
{
    first_value,  ///< Compressed to the first value (ChunkState::value): no buffer.
    second_value, ///< Compressed to ChunkState::value: no buffer.
    motif,        ///< pattern: element k is data[k] + value (wrapping), data a shared motif.
    materialised, ///< A buffer of its own: ChunkState::data.
};

/// @brief One chunk, as export_chunk() reads it and import_chunk() writes it.
template <typename T>
struct ChunkState
{
    ChunkKind kind  = ChunkKind::first_value; ///< What the chunk holds.
    T         value = T{};                    ///< The constant, or a motif chunk's offset.
    /// materialised: the chunk's elements; motif: the motif (see the file
    /// comment). Valid until the array is written again; nullptr otherwise.
    const T* data = nullptr;
};

namespace detail
{

/// @brief A motif chunk's element: @p motif + @p offset, wrapping on an
///        integer (pattern::Chunk::wrapping_add's arithmetic).
template <typename T>
[[nodiscard]] T motif_element(T motif, T offset) noexcept
{
    if constexpr (std::is_integral_v<T>)
    {
        using U = std::make_unsigned_t<T>;
        return static_cast<T>(static_cast<U>(static_cast<U>(motif) + static_cast<U>(offset)));
    }
    else
    {
        return static_cast<T>(motif + offset);
    }
}

} // namespace detail

/// @brief The arrays export_chunk() / import_chunk() take.
template <typename Arr>
concept ExchangeableArray = internal::ChunkedArray<Arr> && !is_bool_like_v<typename Arr::value_type>;

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
    const auto  first = ChunkState<T>{ChunkKind::first_value, static_cast<T>(ChunkType::first_value), nullptr};

    if constexpr (is_zero_any_family_v<ChunkType>)
    {
        if constexpr (has_motif_state_v<ChunkType>)
        {
            if (ChunkType::is_motif_ptr(vals))
            {
                const T offset = internal::load_element_relaxed(arr.motif_offset_slot(static_cast<int>(index)));
                return {ChunkKind::motif, offset, vals};
            }
        }
        if (ChunkType::is_constant_ptr(vals))
        {
            if (chunk.is_cst_val2_ptr_of(vals))
            {
                return {ChunkKind::second_value, chunk.second_value_from_ptr(vals), nullptr};
            }
            return first;
        }
    }
    else if constexpr (internal::is_v2_v<Arr>)
    {
        if (chunk.is_cst_val1(vals))
        {
            return first;
        }
        const int ind = arr.get_ind_second_value();
        if (ind >= 0 && chunk.is_cst_val2(ind)) // a second load: quiescent, the same pointer
        {
            return {ChunkKind::second_value, static_cast<T>(arr.get_second_value()), nullptr};
        }
    }
    else if (chunk.is_cst_val1(vals))
    {
        return first; // zero
    }
    return {ChunkKind::materialised, T{}, vals};
}

/**
 * @brief Give chunk @p index of @p arr the state @p state.
 * @details A constant is installed as a compression would commit it, without a
 *          buffer (zero_any: the re-chunk refill's install; zero_one_value:
 *          clear_int_st()'s state, nothing to free); a motif through the
 *          registry (pattern::Array::install_motif_for_test()); a materialised
 *          chunk is filled through copy_range(), which arms it. A state the
 *          array cannot hold as such -- a second value zero_one_value has not
 *          declared, a motif outside pattern -- is filled as elements: the
 *          values are right, the chunk materialised until a pass.
 * @pre The chunk is still at the first value and was never written: @p arr is
 *      freshly built, at the chunk size the states were exported at (and
 *      zero_one_value: with the same second value declared).
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

    if (state.kind == ChunkKind::first_value ||
        (state.kind == ChunkKind::second_value && state.value == static_cast<T>(ChunkType::first_value)))
    {
        return; // the fresh chunk's state
    }
    if (state.kind == ChunkKind::materialised)
    {
        copy_range(arr, begin, end, state.data, assume_drained);
        return;
    }
    if (state.kind == ChunkKind::motif)
    {
        std::vector<T> elems(end - begin);
        for (std::size_t k = 0; k < elems.size(); ++k)
        {
            elems[k] = detail::motif_element(state.data[k], state.value);
        }
        if constexpr (has_motif_state_v<ChunkType>)
        {
            if (elems.size() == chunk_sz && arr.install_motif_for_test(static_cast<int>(index), elems.data()))
            {
                return;
            }
        }
        copy_range(arr, begin, end, elems.data(), assume_drained);
        return;
    }

    // second_value
    auto& chunk = arr.chunk_ref(index);
    if constexpr (is_zero_any_family_v<ChunkType>)
    {
        // What a type-2 compression commits (find_or_set on the chunk's
        // second-value slot, type table, values pointer), minus the buffer:
        // array_chunk_rechunk_inline.h, rechunk_refill().
        const int  num_tab = arr.num_tab_;
        const auto psv     = ProxyType::BaseSpec::get_tab_second_value(num_tab, static_cast<int>(index));
        CstSlot*   ptc     = &ProxyType::tab_type_cst_val[num_tab][index];
        (void)chunk.find_or_set_ind_cst_val2(state.value, psv);
        ChunkType::set_cst_val_to_cst_val2(ptc);
        chunk.init_values_cst_val(ptc, psv);
        return;
    }
    else if constexpr (internal::is_v2_v<Arr>)
    {
        // clear_int_st()'s state (zero_one_value_chunk_inline.h): the values
        // pointer at the second constant, its counter reset; no buffer to free.
        const int ind = arr.get_ind_second_value();
        if (ind >= 0 && state.value == static_cast<T>(arr.get_second_value()))
        {
            chunk.init_values_cst_val(1, ind);
            chunk.reset_nb_val_eq_second_value();
            return;
        }
    }
    fill_range(arr, begin, end, state.value, assume_drained); // not a constant of this array
}

} // namespace shrink_array
