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

#include <cstddef> // std::size_t
#include <vector>  // std::vector

#include "chunk/bit.h"                              // is_bool_like_v: bool and Bit share the bool path
#include "chunk/config.h"                           // build_config::threaded, chunk_config::chunk_dynamic
#include "chunk/internal/PMP_CHUNK.h"                        // CHUNK_PMP_THREAD_CAPACITY (max_thr)
#include "chunk/internal/second_value_info.h"       // SecondValueInfo (InfArrayT)
#include "chunk/internal/type_state_interface.h" // TypeStateInterface
#include "chunk/internal/type_state_policies.h"  // TypeStateDynImpl, TypeStateParallelImpl
#include "chunk/internal/reuse_pools.h"             // ReusePools
#include "chunk/internal/compress_candidates.h"     // CompressCandidates

namespace shrink_array
{
// Forward declaration: only named in contexts that depend on T. The definition
// (chunk/internal/compression_manager.h) comes with type_state_policies_inline.h.
template <typename T, typename ChunkT>
struct CompressionManager;

/**
 * @brief The shared, per-(variant, element-type) state behind every array.
 *
 * @details One instance exists per @c ChunkT / @c T combination, reachable
 *          statically, and AllTypeChunk drives every one of them through
 *          TypeStateInterface. It is made of:
 *
 *          - **ReusePools** (reuse_pools.h): the buffer reuse free-lists, their
 *            budgets, drains and trims;
 *          - **CompressCandidates** (compress_candidates.h): the
 *            compression-candidate lists;
 *          - **TypeStateDynImpl / TypeStateParallelImpl**
 *            (type_state_policies.h): the build-dependent overrides -- the
 *            per-rank allocators, the deferred flush, the teardown of the
 *            reuse lists;
 *          - and, here, what crosses them: the chunk counters both budgets are
 *            computed from, the keyed allocator sets of the arrays with their
 *            own affinity, the single compression sweep, and the overrides of
 *            TypeStateInterface.
 *
 *          The static members of the bases are named through this class as
 *          before (TypeState<ChunkT, T>::get_list_block_free_thr(...)).
 *
 * @tparam ChunkT chunk variant template (zero, zero_one_value or zero_any)
 * @tparam T       stored element type
 */
template <template <typename> class ChunkT, typename T>
class TypeState final : public virtual TypeStateInterface,
                           public TypeStateDynImpl<ChunkT, T>,
                           public TypeStateParallelImpl<ChunkT, T>,
                           public ReusePools<ChunkT, T>,
                           public CompressCandidates<ChunkT, T>
{
    // The implementation policies are BASES and need this class's state, which
    // protected access does not reach: friendship, not a public accessor.
    template <template <typename> class, typename, bool>
    friend struct TypeStateDynImpl;
    template <template <typename> class, typename, bool>
    friend class TypeStateParallelImpl;

  private:
    using Pools      = ReusePools<ChunkT, T>;         ///< the reuse pools, a base
    using Candidates = CompressCandidates<ChunkT, T>; ///< the candidate lists, a base

    static constexpr bool is_bool     = is_bool_like_v<T>; // bool and Bit (bit.h)
    static constexpr bool is_not_bool = !is_bool;

    void destroy_block_chunk() override;
    /// @brief CompressCandidates::force_compress().
    void force_chunk_to_compress() const override
    {
        Candidates::force_compress();
    }
    /// @brief CompressCandidates::have_candidates().
    [[nodiscard]] bool have_chunk_to_compress_potentially(int n_thr) const override
    {
        return Candidates::have_candidates(n_thr);
    }
    /// @brief CompressCandidates::compress_pass().
    void compress_chunk_to_compress_potentially(bool apply) override
    {
        Candidates::compress_pass(apply);
    }
    void               sweep_prepare() override;
    void               sweep_rank(int thr, bool apply) override;
    [[nodiscard]] bool sweep_mops_up(bool apply) const override;

    [[nodiscard]] std::size_t reuse_pool_bytes_int() const override
    {
        return Pools::reuse_pool_bytes();
    }
    std::size_t trim_reuse_pools_int(std::size_t keep) override
    {
        return Pools::trim_reuse_pools(keep);
    }
    void print_reuse_pool_breakdown_int(const char* label) const override
    {
        Pools::print_reuse_pool_breakdown(label);
    }

    const int max_thr_ = CHUNK_PMP_THREAD_CAPACITY;

    /// @brief zero_one_value: the second-value lists sweep_rank() drains, snapshotted
    ///        serially by sweep_prepare().
    std::vector<SecondValueInfo<T>*> sweep_inf_arrays_;

    /// @brief Slab byte budget seen at the last register_chunk_size(); what
    ///        register_alloc_key() hands to the keyed allocator sets.
    static inline int registered_block_size_ = 0;

    static inline long somme_chunk_create{0};

  public:
    /**
     * @brief Captures the thread capacity once, at construction.
     *
     * @note max_thr is the process-wide thread capacity: every per-thread
     *       container has a slot for any rank, even of a team raised later.
     */
    TypeState()           = default;
    ~TypeState() override = default;

    /**
     * @brief Size the static per-rank structures of (ChunkT, T)
     *        (see CompressionManager::ensure_initialised).
     * @details Called by AllTypeChunk::register_manager(), before the manager
     *          is published: the fan-outs and the first array come after it.
     */
    static void init_static_state()
    {
        CompressionManager<T, ChunkT<T>>::ensure_initialised();
        (void)ChunkT<T>::get_manager(); // built now, outside any parallel region of the type
    }

    TypeState(const TypeState&)            = delete;
    TypeState& operator=(const TypeState&) = delete;
    TypeState(TypeState&&)                 = delete;
    TypeState& operator=(TypeState&&)      = delete;

    /// @brief CompressCandidates::resize_lists().
    void resize_chunk_to_compress_potentially() const override
    {
        Candidates::resize_lists();
    }

    // Build-dependent overrides live in the policies:
    //   register_chunk_size()                 -> TypeStateDynImpl
    //   the allocator / free-list surface     -> TypeStateParallelImpl

    /// @brief Recompute how many buffers each thread may keep parked for reuse.
    static void compute_size_buffer_reuse();

    /// @brief Recompute the candidate-list length that triggers a compression pass.
    static void compute_size_max_number_of_chunk_to_check_compress();

    /// @brief Add @p count to the running total of chunks created (statistics).
    static void increm_somme_chunk_create(int count) noexcept
    {
        somme_chunk_create += count;
    }
    /// @brief Subtract @p count from the running total of chunks created (statistics).
    static void decrem_somme_chunk_create(int count) noexcept
    {
        somme_chunk_create -= count;
    }
    /// @brief Chunks created minus chunks destroyed -- what the compression
    ///        threshold is computed from, so it must track the chunks alive.
    [[nodiscard]] static long get_somme_chunk_create() noexcept
    {
        return somme_chunk_create;
    }

    /// @brief Everything an array with its own affinity key needs, in one call
    ///        (sequential, from rebuild_storage once nb_chunk is known): its
    ///        allocator set at (key, size), its per-rank reuse lists, and their
    ///        budget nb_chunk / (factor_div x threads).
    /// @note Dynamic OpenMP fast-alloc builds only; a no-op elsewhere.
    /// @return true when the keyed set, lists and budget exist; false otherwise
    ///         (no slab budget yet, or no dynamic registry) -- the caller must
    ///         then stay on key 0.
    [[nodiscard]] static bool register_alloc_key(unsigned int key, unsigned int size, int nb_chunk);
};

} // namespace shrink_array

// The bodies, once TypeState is complete: they name TypeState<ChunkT, T>::
// and need the chunk classes, whose headers include this one (a cycle the
// order breaks).
#include "chunk/internal/reuse_pools_inline.h"            // ReusePools' bodies
#include "chunk/internal/compress_candidates_inline.h"    // CompressCandidates' bodies
#include "chunk/internal/type_state_policies_inline.h" // the policies' bodies
#include "chunk/internal/type_state_inline.h"          // TypeState's bodies
