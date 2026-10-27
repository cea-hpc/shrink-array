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
 * @file pattern_chunk.h
 * @brief pattern::Chunk, the chunk of pattern::Array (docs/pattern_design.md).
 *
 * @details A member of the zero_any family (chunk_traits.h): its body is
 *          zero_any::ChunkImpl's, so it compresses to zero or to a second value
 *          exactly as zero_any::Chunk does. It adds the motif state: the values
 *          pointer names a shared motif (MotifRegion), and element j reads
 *          motif[j] + the chunk's offset. The offset lives in the chunk's
 *          second-value slot, free since a motif chunk has no second value: no
 *          byte more per chunk, one cache line more per read of a motif chunk
 *          (the slot row is another array). The shared code reads and
 *          materialises a motif chunk through its has_motif_state_v branches.
 *          Integers only.
 */

#include <atomic>      // std::atomic_ref (the offset's release store)
#include <cstddef>     // std::size_t
#include <cstdint>     // std::uint64_t (the motif hash)
#include <tuple>       // std::tuple (the motif candidates)
#include <type_traits> // std::is_integral_v, std::is_same_v, std::make_unsigned_t
#include <vector>      // std::vector (the motif candidates, the scan buffer)

#include "chunk/chunk_traits.h"                 // ChunkTraits<pattern::Chunk<T>>, HasMotifState
#include "chunk/internal/motif_region.h"        // pattern::MotifRegion
#include "chunk/internal/relaxed_element.h"     // load_element_relaxed (the offset)
#include "chunk/internal/tab_cst_val2_holder.h" // SecondValueTraits (the slot type), SecondValueRef
#include "chunk/internal/cst_slot.h"            // CstSlot::motif
#include "chunk/internal/chunk_stats.h"         // increm_chunk_compress, add_chunk_bytes_freed (CHUNK_STATS)
#include "chunk/internal/invariant_check.h"     // CHUNK_INVARIANT_FAILED
#include "chunk/config.h"                       // build_config::threaded
#include "chunk/zero_any_chunk.h"               // zero_any::ChunkImpl

namespace shrink_array
{

namespace pattern
{

/**
 * @brief Chunk of pattern::Array: zero_any::ChunkImpl with itself as the chunk
 *        type, plus the motif state.
 * @tparam T Element type: an integer type other than bool.
 */
template <typename T>
class Chunk : public zero_any::ChunkImpl<T, Chunk<T>>
{
    static_assert(std::is_integral_v<T> && !std::is_same_v<T, bool>,
                  "pattern::Chunk holds integers only (docs/pattern_design.md, section 1)");
    static_assert(!SecondValueTraits<T>::indexed,
                  "pattern::Chunk keeps a motif chunk's offset in its second-value slot, which must hold a T: "
                  "build without CHUNK_SECOND_VALUE_INDEXED, or use an element type of at most 4 bytes");

  public:
    /// @brief The unsigned type of the same width: the motif arithmetic wraps on it.
    using UnsignedT = std::make_unsigned_t<T>;

    /// @brief Is @p ptr a motif (a values pointer inside MotifRegion<T>)?
    [[nodiscard]] static bool is_motif_ptr(const T* ptr) noexcept
    {
        return MotifRegion<T>::contains(ptr);
    }

    /// @brief @p lhs + @p rhs, wrapping (on UnsignedT: never undefined behaviour).
    [[nodiscard]] static T wrapping_add(T lhs, T rhs) noexcept
    {
        return static_cast<T>(static_cast<UnsignedT>(static_cast<UnsignedT>(lhs) + static_cast<UnsignedT>(rhs)));
    }

    /// @brief @p lhs - @p rhs, wrapping (on UnsignedT).
    [[nodiscard]] static T wrapping_sub(T lhs, T rhs) noexcept
    {
        return static_cast<T>(static_cast<UnsignedT>(static_cast<UnsignedT>(lhs) - static_cast<UnsignedT>(rhs)));
    }

    /**
     * @brief Element @p j of a chunk whose values pointer is @p motif and whose
     *        offset is @p offset (motif_offset()).
     * @param motif  The chunk's values pointer, loaded by the caller (is_motif_ptr()).
     * @param j      Offset in the chunk.
     * @param offset The chunk's offset.
     * @return motif[j] + offset, wrapping.
     */
    [[nodiscard]] static T motif_value(const T* motif, int j, T offset) noexcept
    {
        // raw-read: motif_value -- a registered motif is never written while a chunk points at it
        return wrapping_add(motif[j], offset);
    }

    /**
     * @brief The offset of this chunk, whose values pointer the caller loaded as
     *        @p motif: valid only if the pointer has not moved meanwhile.
     * @details The slot is rewritten by a compression pass (to another motif, or to a
     *          constant: its second value) while a reader may still hold the motif
     *          pointer loaded before a writer materialised the chunk; the read section
     *          protects the buffer the pass frees, not the slot it rewrites. So: the
     *          slot loaded with acquire (it pairs with the pass's release store, made
     *          after the pass saw the materialised buffer), then the values pointer
     *          again. Unchanged: the offset is that of the state the two loads
     *          bracket. Moved: the caller starts over (docs/formal_models.md, F5;
     *          formal/motif_offset.c).
     * @param motif       The values pointer the caller loaded (is_motif_ptr()).
     * @param offset_slot The chunk's second-value slot.
     * @param offset      Receives the offset.
     * @return false if the values pointer moved: @p offset is then meaningless.
     */
    [[nodiscard]] bool motif_offset(const T* motif, const T* offset_slot, T& offset) const noexcept
    {
        offset = internal::load_element_acquire(offset_slot);
        return this->values_load() == motif;
    }

    /// @brief A chunk the pass found equal to a registered motif plus an offset:
    ///        the chunk, the motif, the offset, its index, its array's table, its geometry.
    using MotifCandidate = std::tuple<Chunk*, const T*, T, int, int, ChunkElems>;

    /**
     * @brief The pass's third classification, after zero_any's two constants
     *        (fill_list_to_compress()): is this claimed, materialised chunk a
     *        registered motif plus an offset? If so, it is queued for
     *        compress_motif_candidates(), and keeps its claim.
     * @details Reads the chunk in one block, normalises it (element j minus element
     *          0, wrapping: element 0 is the offset), hashes the motif and asks the
     *          registry under its admission rule (MotifRegion::find_or_admit), the
     *          sighter being this chunk: a first sighting only records its tag,
     *          and the chunk stays materialised. An
     *          all-zero motif is a uniform chunk, which zero_any's scan has already
     *          classified: it never gets here.
     * @param num_chunk The chunk's index in its array.
     * @param num_tab   Its array's table.
     * @param size      Its array's geometry.
     */
    void classify_motif(int num_chunk, int num_tab, ChunkElems size);

    /// @brief How many motif candidates this thread's pass holds (fill_list_chunk_to_compress_in_par()).
    [[nodiscard]] static std::size_t motif_candidate_count() noexcept
    {
        return motif_candidates_.size();
    }

    /**
     * @brief Compress the motif candidates of this thread's pass (make_block_compression(),
     *        after the two constants), then release their claims and empty the list.
     * @param thr The pass's rank: whose pool receives the buffers.
     */
    static void compress_motif_candidates(int thr);

    /**
     * @brief Compress this materialised chunk to @p motif plus @p offset.
     * @details The order of docs/pattern_design.md, section 6: the offset to the
     *          chunk's second-value slot first (release), then the values pointer
     *          from the buffer to the motif (the CAS, acq_rel: a reader that sees the
     *          motif sees the offset), then the buffer released after the grace period
     *          (free_block_to_pool(), as zero_any's clear). The chunk is claimed by the
     *          pass: no other pass compresses it and no writer swaps its buffer, so the
     *          CAS cannot fail.
     * @param ptr_type_cst The chunk's type slot: set to CstSlot::motif.
     * @param slot         The chunk's second-value slot: receives the offset.
     * @param motif        The registered motif.
     * @param offset       The chunk's offset (its element 0).
     * @param thr_rank     Rank whose pool receives the buffer.
     */
    void block_clear_motif(CstSlot* ptr_type_cst, SecondValueRef<T> slot, const T* motif, T offset, int thr_rank);

  private:
    static inline thread_local std::vector<MotifCandidate> motif_candidates_; ///< This thread's pass's motif hits.
    static inline thread_local std::vector<T>              motif_scratch_;    ///< The scan's normalised motif.
};

} // namespace pattern

} // namespace shrink_array

#include "chunk/internal/pattern_chunk_inline.h" // the template bodies
