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
 * @file type_state_interface.h
 * @brief The type-erased interface of the per-(variant, element type) chunk
 *        managers: TypeStateInterface and its two build-dependent halves.
 *
 * AllTypeChunk drives every manager through it; TypeState
 * (chunk/internal/type_state.h) implements it.
 */

#include <cstddef> // std::size_t

#include "chunk/config.h" // build_config::chunk_dynamic, build_config::threaded

namespace shrink_array
{

/**
 * @brief The CHUNK_DYNAMIC-only half of the manager interface.
 *
 * @details Selected by build_config constants rather than the preprocessor -- the
 *          same policy split the chunk classes use (ChunkAllocPolicy,
 *          ChunkDeferredOpsPolicy, DeferredReadGate).
 *
 * @tparam IsDynamic Whether runtime geometry is compiled in.
 */
template <bool IsDynamic = build_config::chunk_dynamic>
struct TypeStateDynInterface; // primary left undefined: only the 2 specializations exist

/// @brief Dynamic-geometry builds: register_chunk_size().
template <>
struct TypeStateDynInterface<true>
{
    TypeStateDynInterface()          = default;
    virtual ~TypeStateDynInterface() = default;
    TypeStateDynInterface(const TypeStateDynInterface&) =
        delete ("an abstract interface of the managers: a copy through it would slice the manager");
    TypeStateDynInterface& operator=(const TypeStateDynInterface&) =
        delete ("an abstract interface of the managers: a copy through it would slice the manager");
    TypeStateDynInterface(TypeStateDynInterface&&) =
        delete ("an abstract interface of the managers: a copy through it would slice the manager");
    TypeStateDynInterface& operator=(TypeStateDynInterface&&) =
        delete ("an abstract interface of the managers: a copy through it would slice the manager");

    /// @brief Declare a runtime chunk geometry to this type's allocators.
    virtual void register_chunk_size(unsigned int size, int block_size) = 0;
};

/// @brief Static-geometry builds: nothing to declare.
template <>
struct TypeStateDynInterface<false>
{
    TypeStateDynInterface()          = default;
    virtual ~TypeStateDynInterface() = default;
    TypeStateDynInterface(const TypeStateDynInterface&) =
        delete ("an abstract interface of the managers: a copy through it would slice the manager");
    TypeStateDynInterface& operator=(const TypeStateDynInterface&) =
        delete ("an abstract interface of the managers: a copy through it would slice the manager");
    TypeStateDynInterface(TypeStateDynInterface&&) =
        delete ("an abstract interface of the managers: a copy through it would slice the manager");
    TypeStateDynInterface& operator=(TypeStateDynInterface&&) =
        delete ("an abstract interface of the managers: a copy through it would slice the manager");
};

/**
 * @brief The allocator / free-list half of the manager interface.
 *
 * @details The two modes offer DIFFERENT surfaces (OpenMP: per-thread allocators,
 *          reuse lists, deferred ops; serial: one allocator), hence two
 *          specializations rather than one set with no-ops.
 *
 * @tparam Threaded Whether the parallel surface is the one compiled in.
 */
template <bool Threaded = build_config::threaded>
struct TypeStateParallelInterface; // primary left undefined: only the 2 specializations exist

/// @brief OpenMP builds: per-thread allocators, reuse lists and deferred ops.
template <>
struct TypeStateParallelInterface<true>
{
    TypeStateParallelInterface()          = default;
    virtual ~TypeStateParallelInterface() = default;
    TypeStateParallelInterface(const TypeStateParallelInterface&) =
        delete ("an abstract interface of the managers: a copy through it would slice the manager");
    TypeStateParallelInterface& operator=(const TypeStateParallelInterface&) =
        delete ("an abstract interface of the managers: a copy through it would slice the manager");
    TypeStateParallelInterface(TypeStateParallelInterface&&) =
        delete ("an abstract interface of the managers: a copy through it would slice the manager");
    TypeStateParallelInterface& operator=(TypeStateParallelInterface&&) =
        delete ("an abstract interface of the managers: a copy through it would slice the manager");

    // --- per-thread allocator and free-list surface ---
    /// @brief Set every rank's allocator geometry, from the calling thread:
    ///        the allocators are slots indexed by rank, and a slot's slab is
    ///        only taken at that rank's first allocation. Sequential.
    virtual void init_allocators(int block_size) const = 0;
    /// @brief Create this type's array of per-thread chunk allocators
    ///        (fast-alloc builds; a no-op otherwise).
    virtual void alloc_chunk_allocator_thr() = 0;

    /// @brief Size the per-rank free-lists of released buffers to the team.
    virtual void resize_all_list_block_free_thr() = 0;
    /// @brief Apply this type's pending deferred write ops, repeatedly until a
    ///        pass changes nothing (a replay can enqueue another op).
    /// @param in_parallel  True when called from inside a parallel region.
    /// @param have_flushed Set (never cleared) when at least one op was applied.
    virtual void flush_all_deferred(bool in_parallel, bool& have_flushed) = 0;
    /// @brief After the last flush: destroy the per-thread op maps and clear
    ///        every array's dirty bitset, so the next phase starts clean.
    virtual void destroy_after_flush(bool in_parallel) = 0;

    /// @brief Destroy the per-thread free-list containers themselves (teardown).
    virtual void delete_list_block_chunk_free_itself() = 0;

    /// @brief Would load_balance_list_block_free() be worth calling?
    [[nodiscard]] virtual bool need_load_balance_list_block_free() const = 0;
    /// @brief Redistribute released buffers across the per-thread free-lists
    ///        (a thread freeing far more than it allocates would hoard them).
    virtual void load_balance_list_block_free() = 0;

    // --- teardown of the reuse lists, in phases (see type_state_policies_inline.h) ---
    // AllTypeChunk::clear_list_block_free_thr() drives them: prepare() outside
    // the region, then inside ONE region and separated by barriers: screen()
    // per bucket, sort() per bucket, dedup() per address slice, thr() per
    // bucket. Every per-bucket call is independent of the others in its phase,
    // so a thread may take several buckets (team smaller than the count).

    /// @brief Size the teardown buckets. Sequential, before the region.
    /// @return the bucket count the per-bucket phases stride over.
    virtual std::size_t clear_list_block_free_prepare() = 0;

    /// @brief Drain rank @p bucket's reuse lists into its bucket. Out-of-range is a no-op.
    virtual void clear_list_block_free_screen(int bucket) = 0;

    /// @brief Sort bucket @p bucket and drop its own repeats. Out-of-range is a no-op.
    virtual void clear_list_block_free_sort(int bucket) = 0;

    /// @brief Record the cross-bucket repeats within address slice @p slice of @p nslices.
    virtual void clear_list_block_free_dedup(int slice, int nslices) = 0;

    /// @brief Free bucket @p bucket, minus the recorded repeats. Out-of-range is a no-op.
    /// @note One rank per thread when the team allows: a block is cheapest to
    ///       release through the arena it came from.
    virtual void clear_list_block_free_thr(int bucket) = 0;
};

/// @brief Serial builds: one allocator, no per-thread free lists, nothing to flush.
template <>
struct TypeStateParallelInterface<false>
{
    TypeStateParallelInterface()          = default;
    virtual ~TypeStateParallelInterface() = default;
    TypeStateParallelInterface(const TypeStateParallelInterface&) =
        delete ("an abstract interface of the managers: a copy through it would slice the manager");
    TypeStateParallelInterface& operator=(const TypeStateParallelInterface&) =
        delete ("an abstract interface of the managers: a copy through it would slice the manager");
    TypeStateParallelInterface(TypeStateParallelInterface&&) =
        delete ("an abstract interface of the managers: a copy through it would slice the manager");
    TypeStateParallelInterface& operator=(TypeStateParallelInterface&&) =
        delete ("an abstract interface of the managers: a copy through it would slice the manager");

    /// @brief Set the single allocator's geometry.
    virtual void init_allocator(int block_size) const = 0;
    /// @brief Destroy the single allocator (teardown).
    virtual void delete_chunk_allocator() = 0;
};

/**
 * @brief Type-erased interface to a per-(variant, element-type) chunk manager.
 *
 * @details One TypeState class exists per (variant, type); AllTypeChunk, which
 *          acts on all of them (allocator init, compression passes, teardown),
 *          holds TypeStateInterface pointers and dispatches through these virtuals.
 *          The build-dependent methods come from the two interface policies above.
 */
struct TypeStateInterface : public virtual TypeStateDynInterface<>, public virtual TypeStateParallelInterface<>
{
    TypeStateInterface()           = default;
    ~TypeStateInterface() override = default;
    TypeStateInterface(const TypeStateInterface&) =
        delete ("an abstract interface of the managers: a copy through it would slice the manager");
    TypeStateInterface& operator=(const TypeStateInterface&) =
        delete ("an abstract interface of the managers: a copy through it would slice the manager");
    TypeStateInterface(TypeStateInterface&&) =
        delete ("an abstract interface of the managers: a copy through it would slice the manager");
    TypeStateInterface& operator=(TypeStateInterface&&) =
        delete ("an abstract interface of the managers: a copy through it would slice the manager");

    // --- present in every build ---
    /// @brief Release this type's allocation blocks.
    virtual void destroy_block_chunk() = 0;
    /// @brief Size the per-rank candidate lists to the team.
    virtual void resize_chunk_to_compress_potentially() const = 0;
    /// @brief Compress every compressible chunk of this type now, ignoring the
    ///        candidate lists.
    virtual void force_chunk_to_compress() const = 0;
    /// @brief Does any of the @p n_thr per-rank candidate lists hold a chunk?
    [[nodiscard]] virtual bool have_chunk_to_compress_potentially(int n_thr) const = 0;
    /// @brief Drain this type's compression-candidate backlog.
    /// @param apply true: compress the candidates found constant; false: walk
    ///        the lists without compressing (see AllTypeChunk's).
    virtual void compress_chunk_to_compress_potentially(bool apply) = 0;

    /// @name The single sweep region (AllTypeChunk::compress_chunk_to_compress_potentially)
    /// The pass compress_chunk_to_compress_potentially() makes, split so that
    /// every type runs in ONE region: prepare serially, then each thread runs
    /// sweep_rank() for its rank, then the mop-up of the ranks past the team
    /// if sweep_mops_up(). Outside a region only; no-ops without fast block
    /// compression, like the pass itself.
    ///@{
    /// @brief Serial, before the region: snapshot what sweep_rank() walks.
    virtual void sweep_prepare() = 0;
    /// @brief Rank @p thr's share of the pass.
    virtual void sweep_rank(int thr, bool apply) = 0;
    /// @brief Are the ranks past the team swept too, serially, after the region?
    [[nodiscard]] virtual bool sweep_mops_up(bool apply) const = 0;
    ///@}

    /// @brief Bytes parked in this manager's reuse pools (printed with CHUNK_STATS).
    /// @details Every build: the reuse pools are not OpenMP-only.
    [[nodiscard]] virtual std::size_t reuse_pool_bytes_int() const = 0;
    /// @brief Return the pools' excess; see TypeState::trim_reuse_pools.
    virtual std::size_t trim_reuse_pools_int(std::size_t keep_per_thread) = 0;
    /// @brief Print the pools' breakdown by geometry; see
    ///        TypeState::print_reuse_pool_breakdown.
    virtual void print_reuse_pool_breakdown_int(const char* label) const = 0;

    // The build-dependent halves live in the three interface policies above.
};

} // namespace shrink_array
