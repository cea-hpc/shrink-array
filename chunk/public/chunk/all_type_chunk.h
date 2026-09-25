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

//
// Created by ballereaup on 10/12/24.
//
#pragma once

#include <array>
#include <atomic>
#include <cstddef> // std::size_t
#include <memory>
#include <mutex>
#include <span>
#include <vector>

#include "chunk/config.h" // build_config:: mirrors that select the policies below

/**
 * @file all_type_chunk.h
 * @brief Type-erased registry of every per-type chunk manager of the process.
 *
 * @see manager_chunk.h for the ManagerChunkInt interface this class dispatches
 *      through, and manage_ompt.h for the phase transitions that drive it.
 */

namespace shrink_array
{

template <typename T>
class SingletonChk;

class ManagerChunkInt;

/**
 * @brief Process-wide registry of the per-type chunk managers, reached as a singleton.
 *
 * A ManagerChunk is parameterised by both a chunk template (Chunk / Chunk2 /
 * Chunk2A) and an element type, so there is one manager per combination and no
 * single type can name them all. AllTypeChunk owns them behind
 * ManagerChunkInt pointers and offers the "do it for every type" operations the
 * rest of the library needs: allocator setup, deferred-write flush,
 * compression passes, free-list load balancing and teardown.
 *
 * Every public method below is a fan-out: it walks getListTypeChunk() and
 * forwards to the matching ManagerChunkInt virtual, so the semantics of each
 * one are those of the manager method it calls, applied to all types at once.
 *
 * Access is through `SingletonChk<AllTypeChunk>::getRef()`. The registry starts
 * empty and fills itself: the first array of each (chunk template, element
 * type) registers its manager from its create() (ensure_manager), once
 * ParamChunk::init_indexes() has published the block size. A program may also
 * register a list of types up front with ParamChunk::init_indexes<Types>().
 *
 * @note The available surface depends on the build, and the parts that vary
 *       live in the three policies above:
 *       ::AllTypeChunkParallelInterface picks the per-thread allocator and
 *       free-list methods over the single-allocator ones,
 *       ::AllTypeChunkDynInterface adds register_chunk_size().
 */
// ===========================================================================
// Build-dependent halves of the registry's surface.
//
// Which methods AllTypeChunk offers depends on the build, and a method cannot
// be declared conditionally -- so the conditional halves live here, in
// policies selected by a constexpr.
// AllTypeChunk inherits all three; each contributes the members its own
// configuration has.
//
// The bodies stay in all_type_chunk.cpp. A template's member is only
// instantiated when something uses it, and the explicit instantiation at the
// end of that file names exactly one specialisation of each policy -- so the
// other one is never type-checked, which is what lets the OpenMP bodies call
// manager virtuals that a serial build does not declare.
//
// This mirrors what ManagerChunkInt already does with
// ManagerChunkParallelInterface / DynInterface; these forward
// to those.
//
// Each body reaches the registry with an inline static_cast rather than through
// a self() helper or CRTPSupport<Derived>. That is the same rule ChunkAllocPolicy
// and ChunkDeferredOpsPolicy follow, for the same reason, and here it bites
// harder: AllTypeChunk inherits THREE of these policies at once, so a helper
// named in more than one of them makes that name ambiguous on AllTypeChunk --
// measured, in a CHUNK_DYNAMIC build, as four candidates for self().
// ===========================================================================

/**
 * @brief The threading-dependent half of the registry.
 * @tparam Derived AllTypeChunk (CRTP, for getListTypeChunk()).
 * @tparam Enabled build_config::use_openmp.
 */
template <typename Derived, bool Enabled = build_config::use_openmp>
struct AllTypeChunkParallelInterface; // primary left undefined: only the 2 specializations exist

/// @brief OpenMP builds: per-thread allocators, per-thread free lists, and a
///        deferred-write queue to drain.
template <typename Derived>
struct AllTypeChunkParallelInterface<Derived, true>
{
    /// Resizes every manager's per-thread free-block lists to the current thread count.
    void resize_all_list_block_free_thr() const;

    /**
     * @brief Releases the buffers freed by a preceding flush, for every type.
     *
     * @param in_parallel True when called from inside a parallel region (each
     *        thread then handles its own share); false for a joined, serial call.
     */
    void destroy_after_flush(bool in_parallel);

    /**
     * @brief Applies the pending deferred write ops of every type.
     *
     * Clears `__global_need_flush_deferred` for the duration of the pass, then
     * sets it again on exit. When at least one manager actually flushed and the
     * call is not in-region, destroy_after_flush(false) is chained here so the
     * released buffers are reclaimed immediately.
     *
     * @param in_parallel True when called from inside a parallel region: each
     *        thread drains its OWN queue using its in-region rank, and the
     *        buffer reclaim is left to a later destroy_after_flush().
     * @return True when at least one deferred op was applied.
     */
    bool flush_all_deferred(bool in_parallel);

    /**
     * @brief The per-rank structures of ONE manager, at its registration:
     *        free-block lists, chunk lists and counters sized to the team, the
     *        per-rank allocator array, and every rank's allocator geometry --
     *        set from the registering thread, rank by rank (an allocator is a
     *        slot indexed by rank, not a thread-local; its slab comes at the
     *        rank's first allocation). Any thread, no parallel region.
     * @param block_size Number of chunks per allocation block.
     */
    void init_manager(ManagerChunkInt& manager, int block_size) const;

    /// Allocates the first block of each per-thread allocator (no-op in std-alloc builds).
    void alloc_chunk_allocator_thr() const;

    /// Redistributes the free blocks of every type across threads.
    void load_balance_list_block_free() const;

    /// @return True as soon as one type's free lists are unbalanced enough to be worth a pass.
    bool need_load_balance_list_block_free() const;

    /// Empties the per-thread free-block lists of every type (teardown).
    void clear_list_block_free_thr();

    /// Destroys the per-thread free-block list containers themselves (teardown).
    void delete_list_block_chunk_free_itself() const;
};

/// @brief Serial builds: one allocator per type, and nothing deferred.
template <typename Derived>
struct AllTypeChunkParallelInterface<Derived, false>
{
    /// Serial build: nothing is deferred, so there is nothing to reclaim.
    static void destroy_after_flush(bool) {}
    /// Serial build: writes are applied inline, so there is no queue to drain.
    static bool flush_all_deferred(bool)
    {
        return false;
    }

    /**
     * @brief The single allocator of ONE manager, at its registration.
     * @param block_size Number of chunks per allocation block.
     * @note No-op in std-alloc builds.
     */
    void init_manager(ManagerChunkInt& manager, int block_size) const;

    /// Destroys the single chunk allocator of every type (no-op in std-alloc builds).
    void delete_chunk_allocator() const;
};

/**
 * @brief The runtime-geometry half of the registry.
 * @tparam Enabled build_config::chunk_dynamic.
 */
template <typename Derived, bool Enabled = build_config::chunk_dynamic>
struct AllTypeChunkDynInterface; // primary left undefined: only the 2 specializations exist

template <typename Derived>
struct AllTypeChunkDynInterface<Derived, true>
{
    /**
     * @brief Declares a chunk geometry to every type's manager.
     *
     * Present in EVERY dynamic build, fast-alloc and std-alloc alike: fast-alloc
     * also creates the per-thread allocators for that size, but std-alloc still
     * needs the per-size reuse free-list slot. The manager handles the mode
     * split with `if constexpr`.
     *
     * Remembered too, for the managers registered after it: register_manager
     * replays the list. Sequential context.
     *
     * @param size       Chunk size, in elements, being registered.
     * @param block_size Number of chunks per allocation block for that size.
     */
    void register_chunk_size(unsigned int size, int block_size);
};

/// @brief Fixed-geometry builds: no geometry to declare.
template <typename Derived>
struct AllTypeChunkDynInterface<Derived, false>
{
};

class AllTypeChunk : public AllTypeChunkParallelInterface<AllTypeChunk>, public AllTypeChunkDynInterface<AllTypeChunk>
{
    friend class SingletonChk<AllTypeChunk>;
    /// The dynamic-geometry policy records the sizes it declares (registered_sizes_).
    template <typename D, bool E>
    friend struct AllTypeChunkDynInterface;

  private:
    // ── The registry: append-only, published without a lock ────────────────
    //
    // A manager is registered the first time an array of its (chunk template,
    // element type) is constructed (ensure_manager, from the array's create),
    // not from a list of types declared at startup: the element types of a
    // program are whatever it instantiates. Registration takes the mutex,
    // builds the manager and its per-rank structures in full, stores it in
    // the next slot and publishes the count with a release store. The
    // fan-outs below iterate the published prefix -- an acquire load of the
    // count, then the slots -- with no lock: they run from parallel regions
    // (flush, compression passes, load balancing) and may overlap a
    // registration, and never see a manager before it is whole. Nothing is
    // ever removed before the teardown, so the slots are stable.
    //
    // A fixed capacity, so that a slot's address never moves under a reader:
    // 256 is three chunk templates times more element types than any program
    // has had; exceeding it aborts with a message.
    static constexpr int kMaxManagers = 256;

    std::array<std::unique_ptr<ManagerChunkInt>, kMaxManagers> managers_;
    std::atomic<int>                                           nb_managers_{0};
    std::mutex                                                 registry_mutex_;

    /// Set by ParamChunk::init_indexes(): the registry accepts managers only
    /// once the block size (and the stats) exist. An array constructed before
    /// that aborts with a message naming the call.
    bool ready_      = false;
    int  block_size_ = 0;

    /// Runtime geometries declared so far (CHUNK_DYNAMIC): replayed on every
    /// manager registered after them.
    std::vector<unsigned int> registered_sizes_;

  public:
    /**
     * @brief Builds an EMPTY registry; managers arrive with the first array of
     *        each type (ensure_manager), or up front through initialize<Types>().
     *
     * Private-by-convention: instances are created only by
     * `SingletonChk<AllTypeChunk>`, which is a friend.
     */
    AllTypeChunk() = default;

  private:
    /**
     * @brief Build, initialise and publish the manager @p ManagerT
     *        (ManagerChunk<CHUNK_T, T>). Takes the registry mutex; a no-op
     *        when that manager is already registered.
     */
    template <typename ManagerT>
    void register_manager();

    /**
     * @brief Registers one chunk template for every element type of a tuple.
     *
     * @tparam CHUNK_T Chunk template to register.
     * @tparam Types   Type tuple owned by the consumer (e.g. MyChunkTypes).
     */
    template <template <typename> class CHUNK_T, typename Types>
    void fillManagerChunk();

  public:
    /**
     * @brief The manager of @p ManagerT exists after this call -- registered
     *        now if this is the first array of its type, a no-op otherwise.
     *
     * Called by the arrays' create() with their ArrayChunkTraits::manager_type.
     * One function-local static per instantiation: the first call of a type
     * registers under the mutex, every later one costs a guard load; two
     * threads constructing the first arrays of the same type at once are
     * serialised by that static's initialisation. Any context: the per-type
     * work touches nothing another thread can be using yet, and the fan-outs
     * read the registry without a lock (see the members above).
     *
     * @pre ParamChunk::init_indexes() has run.
     */
    template <typename ManagerT>
    static void ensure_manager();

    /**
     * @brief Registers every element type of a tuple up front, for the chunk
     *        templates the build compiles arrays for.
     *
     * Consumer entry point, called by `ParamChunk::init_indexes<Types>()`.
     * Optional since the arrays register their own types: what it buys is
     * having every manager in place before the first parallel region, at the
     * cost of managers for types the program may never use.
     *
     * @tparam Types Tuple of element types the program will store in chunks.
     */
    template <typename Types>
    void initialize();

    /// @brief ParamChunk::init_indexes() has set the block size: managers may
    ///        now be registered. Records the block size they are built with.
    void mark_ready(int block_size)
    {
        const std::lock_guard<std::mutex> lock(registry_mutex_);
        block_size_ = block_size;
        ready_      = true;
    }

    /// The managers' interface, reached through Derived by the policies so
    /// that their bodies stay dependent (each build declares a different half).
    using manager_int_t = ManagerChunkInt;

    /// Non-copyable: the registry owns the managers and is a singleton.
    AllTypeChunk(const AllTypeChunk& ref) = delete;
    virtual ~AllTypeChunk()               = default;

    /// @return The registered managers, in registration order: the prefix
    ///         published so far, read without a lock (see the members above).
    [[nodiscard]] std::span<const std::unique_ptr<ManagerChunkInt>> getListTypeChunk() const noexcept
    {
        const auto count = static_cast<std::size_t>(nb_managers_.load(std::memory_order_acquire));
        return {managers_.data(), count};
    }

    /// Resizes every manager's compression-candidate list to the current thread count.
    void resize_chunk_to_compress_potentially() const;

    /// Compresses every compressible chunk of every type, ignoring the candidate lists.
    void force_chunk_to_compress() const;

    /**
     * @brief Drains the compression-candidate backlog of every type.
     *
     * @param apply true: every candidate found constant is compressed -- what
     *        the between-region passes want; false: the lists are walked,
     *        their flags reset and (chunk2) emptied, and NOTHING is
     *        compressed -- the terminal sweep of a dying array uses it to
     *        purge its chunks from the lists. It used to be called
     *        is_inside_ompt, and compress_int passed false believing it
     *        drained: its "misses" were the candidates it had just un-queued.
     */
    void compress_chunk_to_compress_potentially(bool apply) const;

    /// @return True as soon as one type has at least one pending compression candidate.
    bool have_chunk_to_compress_potentially() const;
    /// Frees the allocation blocks of every type (no-op in std-alloc builds).
    void destroy_block_chunk() const;

    /**
     * @brief Bytes parked across every manager's reuse pools, at the moment of
     *        the call.
     *
     * @details Printed with CHUNK_STATS (see ParamChunk.cpp). The stats counters
     *          measure EVENTS -- allocations, reuses, compressions -- and say
     *          nothing about what is still held when a run ends; this does. On
     *          chunk2a@256 at 64 threads it reported 3.4 GB of a 4.9 GB
     *          footprint, i.e. 70% of the resident memory sitting in the pools.
     *
     * @warning Declared OUTSIDE the _OPENMP block on purpose. It first landed
     *          inside it, and the serial builds then failed to compile
     *          ParamChunk.cpp ("no member named 'reuse_pool_bytes'"): the reuse
     *          pools exist in every build, and so does the stats block that
     *          prints them.
     */
    [[nodiscard]] std::size_t reuse_pool_bytes() const;

    /**
     * @brief Return the reuse pools' excess, every element type.
     *
     * @details Keeps @p keep_per_thread buffers per (geometry, rank) and frees
     *          the rest, then hands the pages back to the kernel. See
     *          ManagerChunk::trim_reuse_pools for why this is needed and when to
     *          call it. No-op under the fast allocator.
     *
     * @param keep_per_thread working margin kept in each list
     * @return bytes returned to the allocator
     */
    std::size_t trim_reuse_pools(std::size_t keep_per_thread);

    /**
     * @brief Print every type's reuse pools, broken down by geometry.
     *
     * @details Diagnosis companion to reuse_pool_bytes(): the total says how
     *          much is parked, this says under which geometry and across how
     *          many thread ranks -- which is what tells a drain that never ran
     *          from one that ran against the wrong key, or from buffers parked
     *          in the geometry-blind legacy store where no drain can reach them.
     *
     * @param label prefix for the printed lines, e.g. "before re-chunk"
     */
    void print_reuse_pool_breakdown(const char* label) const;


    // load_balance_list_block_free(), need_load_balance_list_block_free(),
    // clear_list_block_free_thr() and delete_list_block_chunk_free_itself()
    // moved to
    // AllTypeChunkParallelInterface: they forward to manager virtuals that only
    // ManagerChunkParallelInterface<true> declares.
};

/**
 * @brief Cap the number of glibc malloc arenas this process will create.
 *
 * @details Worth calling FIRST THING in main, before any parallel region.
 *
 *          glibc gives each contending thread its own arena -- up to 8 per core
 *          -- and an arena never lends a freed block to another. A program that
 *          allocates and frees millions of buffers across 64 threads therefore
 *          pays for the high-water mark of EVERY arena, not of the process.
 *          Measured on shrink_bench, chunk2a, 64 threads, --divisor 5: 5232 MB
 *          against 2765 MB once the arenas were capped at one. The same run
 *          reports identical CHUNK_STATS counters and identical pool retention,
 *          so the difference is entirely in how glibc spread the same bytes.
 *
 *          Exporting MALLOC_ARENA_MAX does the same thing, but it has to be set
 *          in the environment: glibc reads it at the first allocation, which
 *          happens during static initialisation, long before main. setenv() from
 *          main is too late. mallopt() is not -- it applies to arenas created
 *          AFTER the call, and arenas are created lazily on thread contention,
 *          i.e. at the first parallel region.
 *
 * @param max_arenas 0 leaves glibc's default. 1 serialises allocation across
 *                   threads, which can cost contention at high thread counts;
 *                   2-4 keeps most of the saving with far less of it. Measure.
 * @return true if the allocator accepted the setting
 */
bool set_max_malloc_arenas(int max_arenas);

/**
 * @brief Print every manager's reuse pools, broken down by geometry.
 *
 * @details Free function so a caller can reach it without pulling AllTypeChunk's
 *          full definition -- array_chunk_rechunk_inline.h cannot include that
 *          header without closing an include cycle through array_chunk2A.h.
 *          Defined in all_type_chunk.cpp.
 *
 * @param label prefix for the printed lines, e.g. "before drain"
 */
void print_all_reuse_pool_breakdown(const char* label);

} // namespace shrink_array

// Template method definitions (kept out of this header to avoid pulling the
// full Chunk/Chunk2/Chunk2A definitions into every includer of AllTypeChunk.h).
#include "chunk/internal/all_type_chunk_inline.h"
