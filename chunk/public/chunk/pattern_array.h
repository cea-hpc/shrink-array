/*
 * Copyright 2026 CEA (www.cea.fr)
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
 * @file pattern_array.h
 * @brief pattern::Array: a chunked integer array for runs of zeros and motifs
 *        repeated with a shift (docs/algo_pattern.md).
 *
 * @details Being written, in the stages of the design note. Today a member of
 *          the zero_any family with zero_any's behaviour: a chunk compresses to
 *          zero or to any one value, through zero_any's machinery
 *          (zero_any::ArrayImpl, zero_any::ChunkImpl, zero_any::ProxyImpl), with
 *          tables and candidate lists of its own. The motif state comes next.
 *          Integers only; static chunk sizes only (no CHUNK_DYNAMIC build, no
 *          re-chunk): the motifs are per chunk size (section 7).
 *
 *          It shares zero_any's second-value registry of its element type: a
 *          zero_any::Array<T> and a pattern::Array<T> draw their constants from
 *          the same GlobalVecHolder<T>, and reclaim_second_values(), called on
 *          either, keeps what both still name.
 */

#include <initializer_list> // the {a, b, c} constructor
#include <iterator>         // std::forward_iterator, std::distance -- the (first, last) constructor
#include <type_traits>      // std::is_integral_v, std::is_same_v

#include <cstddef> // std::size_t

#include "chunk/chunk_traits.h"           // ArrayChunkTraits<pattern::Array>, ArrayChunkElement
#include "chunk/pattern_chunk.h"          // pattern::Chunk
#include "chunk/internal/pattern_proxy.h" // pattern::Proxy (the only include of it: Array's ProxyType must be complete)
#include "chunk/zero_any_array.h"         // zero_any::ArrayImpl

namespace shrink_array
{

namespace pattern
{

/**
 * @brief Chunked integer array of the zero_any family, to become the array of
 *        shifted motifs (docs/algo_pattern.md). All of it is zero_any::ArrayImpl's;
 *        this class adds the constructors, the destructor and the assignments.
 *
 * @tparam T        stored element type: an integer type other than bool
 * @tparam IntType index type used for sizes and element positions
 */
template <ArrayChunkElement T, typename IntType>
class Array : public zero_any::ArrayImpl<Array<T, IntType>>
{
    static_assert(std::is_integral_v<T> && !std::is_same_v<T, bool>,
                  "pattern::Array holds integers only (docs/algo_pattern.md, section 1)");

  public:
    /**
     * @brief Build an array of @p sz elements.
     * @param sz initial element count; 0 by default, as std::vector
     */
    explicit Array(IntType sz = 0)
    {
        this->init_storage(sz);
    }

    /// @brief Deep copy: from.size() elements read from @p from. Sequential context.
    /// @param from Source array.
    Array(const Array& from) : Array(from.size())
    {
        this->copy_from(from);
    }

    /**
     * @brief Move-construct: an empty array takes @p other's chunks and
     *        identity -- the exchange of swap() -- and @p other is left empty
     *        and valid. O(1) in the elements. Sequential context.
     * @note noexcept, as zero_any::Array's: a failure here terminates.
     */
    Array(Array&& other) noexcept : Array(0)
    {
        this->swap_storage_and_identity(other);
    }

    /// @brief Build an array of std::distance(@p first, @p last) elements and assign them from the range.
    /// @param first Start of the source range.
    /// @param last End of the source range.
    template <std::forward_iterator It>
    Array(It first, It last) : Array(static_cast<IntType>(std::distance(first, last)))
    {
        this->assign(first, last);
    }

    /// @brief Build an array holding the elements of @p il.
    /// @param il Source values.
    explicit Array(std::initializer_list<T> il) : Array(il.begin(), il.end()) {}

    /// @brief Release the storage, then the registry entry (ArrayImpl::destroy_storage()).
    ~Array()
    {
        this->destroy_storage();
    }

    /// @brief Move-assign: this array's own contents are released, then it
    ///        takes @p other's, by the exchange of swap(). Sequential context.
    Array& operator=(Array&& other) noexcept;

    /// @brief Deep copy-assignment (see the copy constructor). Self-assignment
    ///        is a no-op; num_tab is not copied, identity stays with the address.
    Array& operator=(const Array& from)
    {
        if (this != &from)
        {
            this->copy_from(from);
        }
        return *this;
    }

    /**
     * @brief TEST ENTRY, until the compression to motifs exists (stage 2 of the
     *        design note): put chunk @p num_chunk in the motif state for the
     *        elements @p elems, chunk_size() of them.
     * @details The normalised motif (elems[j] - elems[0], wrapping) is
     *          registered in MotifRegion<T> (or found there), the offset
     *          elems[0] goes to the chunk's second-value slot, then the values
     *          pointer to the motif: element j reads motif[j] + offset = elems[j].
     * @pre A sequential point; the chunk at the first constant (not written since
     *      the array was built): no buffer is released, no candidate list holds it.
     * @param num_chunk Chunk index.
     * @param elems     The chunk's elements, chunk_size() of them.
     * @return false, and nothing done, when the chunk is not at the first constant.
     */
    bool install_motif_for_test(int num_chunk, const T* elems);

    /**
     * @brief zero_any's reclaim_second_values() -- the constants no array of the
     *        zero_any family with element type T names any more -- then the motif
     *        registry's: the motifs no chunk of any pattern::Array<T> points at
     *        (docs/algo_pattern.md, section 3). One call, each registry its own
     *        entries.
     * @return How many second values were freed; the motifs freed are in
     *         MotifRegion<T>::stats().
     * @pre The sequential point of zero_any::ArrayImpl::reclaim_second_values(),
     *      over every pattern::Array<T> as well. Refused, for the motifs too, inside
     *      a parallel region.
     */
    static std::size_t reclaim_second_values();

    /**
     * @brief The motif registry's reclaim alone: mark the motif every chunk of every
     *        pattern::Array<T> points at, whatever its index type, and free the others
     *        (MotifRegion::reclaim_unmarked()).
     * @return How many motifs were freed.
     * @pre As reclaim_second_values().
     */
    static std::size_t reclaim_motifs();
};

} // namespace pattern

} // namespace shrink_array

#include "chunk/internal/pattern_array_inline.h" // the template bodies
