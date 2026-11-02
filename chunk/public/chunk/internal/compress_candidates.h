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
 * @file compress_candidates.h
 * @brief CompressCandidates: the compression-candidate lists of one
 *        (variant, element type), a base of ManagerChunk.
 *
 * Its bodies are in compress_candidates_inline.h, which manager_chunk.h
 * includes once ManagerChunk is complete.
 */

#include <tuple>  // std::get (erase_candidates_of_tab)
#include <vector> // std::erase_if

#include "chunk/bit.h"               // is_bool_like_v: bool and Bit share the bool path
#include "chunk/config.h"            // build_config::threaded
#include "chunk/internal/TChunk.h"   // TChunkComp, TChunk2CompMap, TChunk2AComp
#include "chunk/internal/immortal.h" // Immortal: statics safe for global arrays

namespace shrink_array
{

/**
 * @brief The compression-candidate lists of one (variant, element type),
 *        shared by all its arrays: a base of ManagerChunk, static state only.
 *
 * @details Chunks a write left eligible for compression, queued per rank and
 *          processed in batches by the compression passes. Since the store is
 *          shared across arrays, a destroyed or re-chunked array must remove
 *          its own entries -- see the remove_from_compress_list_* helpers. The
 *          three variants keep them in different containers (see
 *          compress_candidates_inline.h).
 *
 *          The members reached through ManagerChunk only are protected:
 *          ManagerChunk's overrides of ManagerChunkInt forward to the passes
 *          below.
 *
 * @tparam ChunkT chunk variant template (zero, zero_one_value or zero_any)
 * @tparam T      stored element type
 */
template <template <typename> class ChunkT, typename T>
class CompressCandidates
{
  protected:
    static constexpr bool is_bool     = is_bool_like_v<T>; // bool and Bit (bit.h)
    static constexpr bool is_not_bool = !is_bool;

    /// @brief Compress every compressible chunk of this type now, ignoring the
    ///        candidate lists (ManagerChunk::force_chunk_to_compress).
    static void force_compress();
    /// @brief Size the per-rank candidate lists to the team
    ///        (ManagerChunk::resize_chunk_to_compress_potentially).
    static void resize_lists();
    /// @brief Does any of the @p n_thr per-rank candidate lists hold a chunk?
    ///        (ManagerChunk::have_chunk_to_compress_potentially)
    [[nodiscard]] static bool have_candidates(int n_thr);
    /// @brief Drain this type's candidate backlog
    ///        (ManagerChunk::compress_chunk_to_compress_potentially).
    static void compress_pass(bool apply);

    /// @brief Set the candidate-list length that triggers a compression pass.
    /// @details Decides when trigger_compression_if_needed() flags a rank's
    ///          candidate list, hence how often compression runs at all.
    static void set_size_max_number_of_chunk_to_check_compress(int max_chunks)
    {
        size_max_number_of_chunk_to_check_compress = max_chunks;
    }

    /**
     * @brief Drop from @p candidates every queued candidate that belongs to @p num_tab.
     *
     * @details The one place that knows a candidate tuple names its owning array
     *          in field <2>; the remove_from_compress_list_*_by_num_tab helpers
     *          differ only in their traversal.
     *
     * @tparam Vec any sequence of candidate tuples.
     * @param candidates       the sequence to filter, modified in place.
     * @param num_tab tab index whose candidates are removed.
     */
    template <typename Vec>
    static void erase_candidates_of_tab(Vec& candidates, int num_tab) noexcept
    {
        std::erase_if(candidates, [num_tab](const auto& tup) { return std::get<2>(tup) == num_tab; });
    }

    // Serial counterpart of chunk_to_compress_potentially_thr (likewise
    // chunk2a_to_compress_potentially). Declared in EVERY build, empty under
    // OpenMP, for the reason given at list_block_free_thr_by_size_.
    static inline constinit internal::Immortal<TChunkComp<ChunkT, T>> chunk_to_compress_potentially_storage_{};
    static inline TChunkComp<ChunkT, T>& chunk_to_compress_potentially = chunk_to_compress_potentially_storage_.value;

    static inline int size_max_number_of_chunk_to_check_compress;

  public:
    /// @brief Candidate-list length that triggers a compression pass.
    [[nodiscard]] static int get_size_max_number_of_chunk_to_check_compress() noexcept
    {
        return size_max_number_of_chunk_to_check_compress;
    }
    // Serial-only on purpose: under OpenMP it would return the always-empty
    // serial store while the candidates sit per thread, so the constraint makes
    // a call fail to compile there.
    /// @brief The zero_any candidate list (serial builds: a single one).
    static auto& get_chunk2a_to_compress_potentially() noexcept
        requires(!build_config::threaded)
    {
        return chunk2a_to_compress_potentially;
    }
    /// @brief Rank @p thr's zero_any candidate list (the single one in a
    ///        serial build).
    static auto& get_chunk2a_to_compress_potentially([[maybe_unused]] int thr) noexcept;

    /// @brief Is no zero_any chunk queued for compression, on any thread?
    [[nodiscard]] static bool is_chunk2a_to_compress_potentially_empty() noexcept;

    /**
     * @brief Drop every queued zero_any candidate pointing into [@p lo, @p hi).
     *
     * @details Teardown removal by CHUNK-POINTER RANGE: candidates hold
     *          &chunk_list_[j], so filtering on the array's chunk_list_ span
     *          guarantees no pointer into it survives teardown. The by-num_tab
     *          form can MISS entries whose num_tab a rechunk identity swap
     *          shifted; the dangling zero_any::Chunk* would then be double-freed by a
     *          later compression pass.
     *
     * @param lo first address of the array's chunk_list_ span
     * @param hi one past its last address
     * @note In plain fast-block-compression builds (GNU, no OMPT) this cleanup is
     *       the ONLY one: the built-in vec_tup.clear() runs only under OMPT.
     */
    static void remove_from_compress_list_chunk2a_by_ptr_range(const void* lo, const void* hi) noexcept;

    /**
     * @brief Drop every queued zero_any candidate of array @p num_tab, in every
     *        rank's list (zero_any::Array::clear_core and rechunk_to).
     * @param num_tab tab index whose candidates are removed
     * @note Keeps other arrays' candidates. At teardown prefer
     *       remove_from_compress_list_chunk2a_by_ptr_range() (see why there).
     */
    static void remove_from_compress_list_chunk2a_by_num_tab(int num_tab) noexcept;

    /**
     * @brief zero_one_value analogue: drop every queued zero_one_value candidate for @p num_tab.
     *
     * @details Removes the matching tuples from every rank and second-value
     *          bucket, WITHOUT compressing. Called from zero_one_value::Array::clear_core()
     *          so no zero_one_value::Chunk* of a torn-down or re-chunked array dangles there.
     *
     * @param num_tab tab index whose candidates are removed
     * @note OpenMP builds only: a serial build queues no zero_one_value candidate (fast
     *       block compression is OpenMP-only), so there is nothing to drop.
     */
    static void remove_from_compress_list_chunk2_by_num_tab([[maybe_unused]] int num_tab) noexcept;

    /// @brief Drop every queued zero candidate of array @p num_tab
    ///        (zero::Array::clear_core and rechunk_to).
    /// @details Removes this array's candidates without compressing, and keeps
    ///          other arrays' intact.
    static void remove_from_compress_list_chunk_by_num_tab(int num_tab) noexcept;

    /// @brief zero_one_value's candidate lists: per rank, one list per second value.
    static inline constinit internal::Immortal<TChunk2CompMap<ChunkT, T>>
                                             map_chunk2_to_compress_potentially_thr_storage_{};
    static inline TChunk2CompMap<ChunkT, T>& map_chunk2_to_compress_potentially_thr =
        map_chunk2_to_compress_potentially_thr_storage_.value;

    /**
     * @brief Rank @p thr's candidate list for second value @p key, or null
     *        when the key has none (OpenMP builds: the per-rank store).
     * @details A lookup, never operator[]: a miss would insert a null entry, from
     *          the write path, into a map other ranks read (have_chunk_to_compress,
     *          the rank-0 emptiness probe).
     */
    template <typename Key>
    [[nodiscard]] static auto* compress_candidates(int thr, const Key& key) noexcept
    {
        auto&      by_value = map_chunk2_to_compress_potentially_thr[thr];
        const auto it       = by_value.find(key);
        return it != by_value.end() ? it->second.get() : nullptr;
    }
    /// @brief zero_one_value: per rank, one list per second-value index, copied by
    ///        zero_one_value::Array into map_chunk2_to_compress_potentially_thr when it
    ///        registers a second value.
    static inline constinit internal::Immortal<TChunk2CompThr<ChunkT, T>> chunk2_to_compress_potentially_thr_storage_{};
    static inline TChunk2CompThr<ChunkT, T>&                              chunk2_to_compress_potentially_thr =
        chunk2_to_compress_potentially_thr_storage_.value;
    /// @brief zero_any's per-rank candidate lists.
    static inline constinit internal::Immortal<TChunk2ACompThr<ChunkT, T>>
                                              chunk2a_to_compress_potentially_thr_storage_{};
    static inline TChunk2ACompThr<ChunkT, T>& chunk2a_to_compress_potentially_thr =
        chunk2a_to_compress_potentially_thr_storage_.value;
    /// @brief zero's per-rank candidate lists.
    static inline constinit internal::Immortal<TChunkCompThr<ChunkT, T>> chunk_to_compress_potentially_thr_storage_{};
    static inline TChunkCompThr<ChunkT, T>&                              chunk_to_compress_potentially_thr =
        chunk_to_compress_potentially_thr_storage_.value;
    /// @brief zero_any's single candidate list of a serial build.
    static inline constinit internal::Immortal<TChunk2AComp<ChunkT, T>> chunk2a_to_compress_potentially_storage_{};
    static inline TChunk2AComp<ChunkT, T>&                              chunk2a_to_compress_potentially =
        chunk2a_to_compress_potentially_storage_.value;
};

} // namespace shrink_array
