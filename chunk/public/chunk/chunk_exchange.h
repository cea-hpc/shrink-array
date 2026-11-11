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
 *        export_chunk() / import_chunk(), and the whole array at once:
 *        export_array() / import_array().
 *
 * @details What a transfer (host <-> device, a checkpoint on disk) needs of an
 *          array is, chunk by chunk: compressed to which value, a motif and its
 *          offset, or the buffer of its S elements. Inside the library that
 *          answer is encoded in where the chunk's values pointer points (the
 *          shared first-value block, zero_one_value's second-value block, an
 *          identifier inside zero_any's second-value region, a motif of
 *          pattern's registry, or a buffer of its own), an encoding that
 *          changes with the implementation. These functions are the one place
 *          that knows it: code outside the library reads and writes chunks
 *          through ChunkState, and keeps working whatever the encoding becomes.
 *
 *          What ChunkState promises of a chunk's elements (materialised: data;
 *          motif: data and value): chunk_size() elements of T, contiguous, no
 *          header, element `index * chunk_size() + k` at position k; the last
 *          chunk of an array whose size is not a multiple of chunk_size() holds
 *          its trailing elements first.
 *
 *          Every variant -- zero, zero_one_value, zero_any, pattern -- and
 *          every element type but Bit (ArrayChunkBits packs 64 elements per
 *          word: its buffer is not S elements of T). bool: its one second value
 *          is true.
 *
 * @warning Serial contract, as fill_range() / copy_range(): call them from a
 *          quiescent point, after finalize_parallel_phase(), with no deferred
 *          write pending (the export reads the chunks raw, as the re-chunk
 *          refill does).
 */

#include <cstddef>     // std::size_t
#include <cstdint>     // std::uint8_t
#include <memory>      // std::make_unique_for_overwrite (a motif chunk's elements)
#include <type_traits> // std::is_same_v, std::is_integral_v, std::make_unsigned_t
#include <utility>     // std::move (import_array)
#include <vector>      // std::vector (ArrayState's chunks)

#include "chunk/array_chunk_range_ops.h"      // copy_range, fill_range, assume_drained, internal::is_v*_v
#include "chunk/bit.h"                        // is_packed_bit_v
#include "chunk/internal/chunk_traits.h"      // is_zero_any_family_v, has_motif_state_v
#include "chunk/internal/chunk_size_policy.h" // set_thread_chunk_size
#include "chunk/internal/cst_slot.h"          // CstSlot
#include "chunk/internal/relaxed_element.h"   // load_element_relaxed (a motif chunk's offset)
#include "chunk/config.h"                     // build_config::chunk_dynamic
#include "chunk/internal/same_value.h"        // same_value

#include "chunk/internal/module_export.h" // SHRINK_ARRAY_EXPORT

namespace shrink_array
{

/// @brief What a chunk holds.
SHRINK_ARRAY_EXPORT enum class ChunkKind : std::uint8_t
{
    first_value,  ///< Compressed to the first value (ChunkState::value): no buffer.
    second_value, ///< Compressed to ChunkState::value: no buffer.
    motif,        ///< pattern: element k is data[k] + value (wrapping), data a shared motif.
    materialised, ///< A buffer of its own: ChunkState::data.
};

/// @brief One chunk, as export_chunk() reads it and import_chunk() writes it.
SHRINK_ARRAY_EXPORT template <typename T>
struct ChunkState
{
    ChunkKind kind  = ChunkKind::first_value; ///< What the chunk holds.
    T         value = T{};                    ///< The constant, or a motif chunk's offset.
    /// materialised: the chunk's elements; motif: the motif (see the file
    /// comment). Valid until the array is written again; nullptr otherwise.
    const T* data = nullptr;
};

/// @brief A whole array, as export_array() reads it and import_array() writes it.
SHRINK_ARRAY_EXPORT template <typename T>
struct ArrayState
{
    std::size_t                size             = 0;     ///< Element count.
    unsigned int               chunk_size       = 0;     ///< S, the elements per chunk.
    bool                       has_second_value = false; ///< zero_one_value (not bool): a second value declared.
    T                          second_value     = T{};   ///< That second value.
    std::vector<ChunkState<T>> chunks;                   ///< One per chunk, in order.
};

namespace detail
{

/// @brief A motif chunk's element: @p motif + @p offset, wrapping on an
///        integer (pattern::Chunk::wrapping_add's arithmetic).
template <typename T>
[[nodiscard]] T motif_element(T motif, T offset) noexcept
{
    if constexpr (std::is_integral_v<T> && !std::is_same_v<T, bool>)
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

/// @brief The arrays these functions take: every variant, every element type but Bit.
SHRINK_ARRAY_EXPORT template <typename Arr>
concept ExchangeableArray = internal::ChunkedArray<Arr> && !is_packed_bit_v<typename Arr::value_type>;

/**
 * @brief The state of chunk @p index of @p arr.
 * @param arr   the array, at a quiescent point (see the file comment)
 * @param index the chunk, below arr.chunk_count()
 */
SHRINK_ARRAY_EXPORT template <ExchangeableArray Arr>
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
        // A second load in both branches: quiescent, the same pointer.
        if constexpr (std::is_same_v<T, bool>)
        {
            if (chunk.is_cst_val2())
            {
                return {ChunkKind::second_value, true, nullptr};
            }
        }
        else
        {
            const int ind = arr.get_ind_second_value();
            if (ind >= 0 && chunk.is_cst_val2(ind))
            {
                return {ChunkKind::second_value, static_cast<T>(arr.get_second_value()), nullptr};
            }
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
 *          registry (pattern::Array::install_motif()); a materialised chunk is
 *          filled through copy_range(), which arms it. A state the array cannot
 *          hold as such -- a second value zero_one_value has not declared, a
 *          motif outside pattern -- is filled as elements: the values are
 *          right, the chunk materialised until a pass.
 * @pre @p arr at the chunk size the state was exported at (and zero_one_value:
 *      with the same second value declared). To import over written chunks,
 *      import_array(): it starts from a fresh array.
 * @param arr   the array, at a quiescent point (see the file comment)
 * @param index the chunk, below arr.chunk_count()
 * @param state what the chunk is to hold
 * @return false, and nothing done, when the chunk is not at the first value
 *         (written and not compressed back since).
 */
SHRINK_ARRAY_EXPORT template <ExchangeableArray Arr>
[[nodiscard]] bool import_chunk(Arr& arr, std::size_t index, const ChunkState<typename Arr::value_type>& state)
{
    using T         = Arr::value_type;
    using ChunkType = Arr::ChunkType;
    using ProxyType = Arr::ProxyType;

    auto& chunk = arr.chunk_ref(index);
    if (!chunk.is_cst_val1())
    {
        return false;
    }

    const auto        chunk_sz = static_cast<std::size_t>(arr.chunk_size());
    const auto        size     = static_cast<std::size_t>(arr.size());
    const std::size_t begin    = index * chunk_sz;
    const std::size_t end      = begin + chunk_sz < size ? begin + chunk_sz : size;

    if constexpr (build_config::chunk_dynamic)
    {
        set_thread_chunk_size(static_cast<unsigned int>(chunk_sz));
    }

    if (state.kind == ChunkKind::first_value ||
        (state.kind == ChunkKind::second_value && same_value(state.value, static_cast<T>(ChunkType::first_value))))
    {
        return true; // the chunk's state already
    }
    if (state.kind == ChunkKind::materialised)
    {
        copy_range(arr, begin, end, state.data, assume_drained);
        return true;
    }
    if (state.kind == ChunkKind::motif)
    {
        const std::size_t count = end - begin;
        const auto        elems = std::make_unique_for_overwrite<T[]>(count); // not std::vector: bool
        for (std::size_t k = 0; k < count; ++k)
        {
            elems[k] = detail::motif_element(state.data[k], state.value);
        }
        if constexpr (has_motif_state_v<ChunkType>)
        {
            if (count == chunk_sz && arr.install_motif(static_cast<int>(index), elems.get()))
            {
                return true;
            }
        }
        copy_range(arr, begin, end, elems.get(), assume_drained);
        return true;
    }

    // second_value
    if constexpr (is_zero_any_family_v<ChunkType>)
    {
        // What a type-2 compression commits (find_or_set on the chunk's
        // second-value slot, type table, values pointer), minus the buffer:
        // array_chunk_rechunk_refill_inline.h, Refill::install_cst2(). bool registers
        // nothing: its one second value is a shared block of true.
        const int  num_tab = arr.num_tab_;
        const auto psv     = ProxyType::BaseSpec::get_tab_second_value(num_tab, static_cast<int>(index));
        CstSlot*   ptc     = &ProxyType::tab_type_cst_val[num_tab][index];
        if constexpr (!std::is_same_v<T, bool>)
        {
            (void)chunk.find_or_set_ind_cst_val2(state.value, psv);
        }
        ChunkType::set_cst_val_to_cst_val2(ptc);
        chunk.init_values_cst_val(ptc, psv);
        return true;
    }
    else if constexpr (internal::is_v2_v<Arr>)
    {
        // clear_int_st()'s state (zero_one_value_chunk_inline.h): the values
        // pointer at the second constant, its counter reset; no buffer to free.
        if constexpr (std::is_same_v<T, bool>)
        {
            chunk.init_values_cst_val_bool(1);
            chunk.reset_nb_val_eq_second_value();
            return true;
        }
        else
        {
            const int ind = arr.get_ind_second_value();
            if (ind >= 0 && same_value(state.value, static_cast<T>(arr.get_second_value())))
            {
                chunk.init_values_cst_val(1, ind);
                chunk.reset_nb_val_eq_second_value();
                return true;
            }
        }
    }
    fill_range(arr, begin, end, state.value, assume_drained); // not a constant of this array
    return true;
}

/**
 * @brief Every chunk of @p arr, with its size, its chunk size and
 *        zero_one_value's declared second value.
 * @param arr the array, at a quiescent point (see the file comment)
 * @return The state; its chunks' data point into @p arr (see ChunkState).
 */
SHRINK_ARRAY_EXPORT template <ExchangeableArray Arr>
[[nodiscard]] ArrayState<typename Arr::value_type> export_array(const Arr& arr)
{
    using T = Arr::value_type;
    ArrayState<T> state;
    state.size       = static_cast<std::size_t>(arr.size());
    state.chunk_size = arr.chunk_size();
    if constexpr (internal::is_v2_v<Arr> && !std::is_same_v<T, bool>)
    {
        state.has_second_value = arr.get_ind_second_value() >= 0;
        state.second_value     = state.has_second_value ? static_cast<T>(arr.get_second_value()) : T{};
    }
    state.chunks.reserve(static_cast<std::size_t>(arr.chunk_count()));
    for (int c = 0; c < arr.chunk_count(); ++c)
    {
        state.chunks.push_back(export_chunk(arr, static_cast<std::size_t>(c)));
    }
    return state;
}

/**
 * @brief Make @p arr the array @p state describes, whatever it held.
 * @details A fresh array of state.size elements (zero_one_value: with the
 *          second value declared) imports every chunk, then is moved into
 *          @p arr, which releases what it held. @p state may come from @p arr
 *          itself: its buffers are copied before the move. Only the chunks
 *          that hold elements, the first ceil(size / chunk_size), are
 *          imported: an exported array may carry more (reserve(), or the
 *          spare last chunk of a fresh array, which one grown by push_back
 *          does not have), and theirs lie past its size.
 * @param arr   the array to replace, at a quiescent point (see the file comment)
 * @param state an export_array() result (its data still valid)
 * @return false, and @p arr unchanged, when a fresh array has another chunk
 *         size than state.chunk_size (a dynamic build at another geometry), or
 *         when @p state has fewer chunks than its size needs (not an export).
 */
SHRINK_ARRAY_EXPORT template <ExchangeableArray Arr>
[[nodiscard]] bool import_array(Arr& arr, const ArrayState<typename Arr::value_type>& state)
{
    using T = Arr::value_type;
    Arr fresh(static_cast<decltype(arr.size())>(state.size));
    if (fresh.chunk_size() != state.chunk_size || state.chunk_size == 0)
    {
        return false;
    }
    // Not the chunk counts: they differ between two arrays of one size (a
    // reserve(), a fresh array's spare last chunk). The elements decide.
    const std::size_t used = (state.size + state.chunk_size - 1) / state.chunk_size;
    if (state.chunks.size() < used || static_cast<std::size_t>(fresh.chunk_count()) < used)
    {
        return false;
    }
    if constexpr (internal::is_v2_v<Arr> && !std::is_same_v<T, bool>)
    {
        if (state.has_second_value)
        {
            fresh.set_second_value(state.second_value);
        }
    }
    for (std::size_t c = 0; c < used; ++c)
    {
        (void)import_chunk(fresh, c, state.chunks[c]); // a fresh array: every chunk at the first value
    }
    arr = std::move(fresh);
    return true;
}

} // namespace shrink_array
