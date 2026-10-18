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

#include <atomic>  // std::atomic
#include <cstddef> // std::size_t
#include <memory>  // std::unique_ptr
#include <mutex>   // std::mutex, std::scoped_lock
#include <span>    // std::span
#include <vector>  // std::vector

#include "chunk/config.h" // build_config:: mirrors that select the policies below

/**
 * @file all_type_chunk_decl.h
 * @brief Type-erased registry of every per-type chunk manager of the process:
 *        the declarations, without the template bodies.
 *
 * @details For a header that must name AllTypeChunk while a variant's array
 *          class is still being defined: array_chunk_base_inline.h, whose
 *          ArrayChunkBase::create() calls ensure_manager(). The bodies
 *          (all_type_chunk_inline.h) reach the managers, then Chunk2A and
 *          array_chunk2A.h -- included from there, inside array_chunk2_base.h,
 *          they closed a cycle and ArrayChunk2Base was not yet declared. They
 *          come with all_type_chunk.h, which each variant's _inline.h
 *          includes after its class.
 *
 * @see manager_chunk.h for the ManagerChunkInt interface this class dispatches
 *      through, and manage_ompt.h for the phase transitions that drive it.
 */

namespace fast_alloc
{
template <typename T>
class Singleton; // fast_allocators/internal/singleton.h; chunk names it SingletonChk
} // namespace fast_alloc

namespace shrink_array
{

struct ManagerChunkInt;

// ===========================================================================
// Build-dependent halves of the registry's surface.
//
// A method cannot be declared conditionally, so the build-dependent halves of
// AllTypeChunk live in policies selected by a constexpr; AllTypeChunk inherits
// both. They mirror (and forward to) ManagerChunkInt's
// ManagerChunkParallelInterface / DynInterface.
//
// The bodies stay in all_type_chunk.cpp, whose explicit instantiation names
// exactly one specialisation of each policy: the other is never type-checked,
// which lets the OpenMP bodies call manager virtuals a serial build does not
// declare.
//
// Each body reaches the registry with an inline static_cast, not a self()
// helper or CRTPSupport<Derived> (same rule as ChunkAllocPolicy /
// ChunkDeferredOpsPolicy): AllTypeChunk inherits both policies, so a helper
// named in both would be ambiguous on AllTypeChunk.
// ===========================================================================

/**
 * @brief The threading-dependent half of the registry.
 * @tparam Derived AllTypeChunk (CRTP, for getListTypeChunk()).
 * @tparam Enabled build_config::threaded.
 */
template <typename Derived, bool Enabled = build_config::threaded>
class AllTypeChunkParallelInterface; // primary left undefined: only the 2 specializations exist

/// @brief OpenMP builds: per-thread allocators, per-thread free lists, and a
///        deferred-write queue to drain.
template <typename Derived>
class AllTypeChunkParallelInterface<Derived, true>
{
  public:
    /**
     * @brief Releases the buffers freed by a preceding flush, for every type.
     *
     * @param in_parallel True when called from inside a parallel region (each
     *        thread then handles its own share); false for a joined, serial call.
     */
    void destroy_after_flush(bool in_parallel) const;

    /**
     * @brief Applies the pending deferred write ops of every type.
     *
     * Clears `g_need_flush_deferred` for the duration of the pass, then
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

    /// Redistributes the free blocks of every type across threads.
    void load_balance_list_block_free() const;

    /// @return True as soon as one type's free lists are unbalanced enough to be worth a pass.
    [[nodiscard]] bool need_load_balance_list_block_free() const;

    /// Empties the per-thread free-block lists of every type (teardown).
    void clear_list_block_free_thr();

    /// Destroys the per-thread free-block list containers themselves (teardown).
    void delete_list_block_chunk_free_itself() const;

  protected:
    /**
     * @brief The per-rank structures of ONE manager, at its registration:
     *        free-block lists, chunk lists and counters sized to the team, the
     *        per-rank allocator array, and every rank's allocator geometry --
     *        set from the registering thread, rank by rank (an allocator is a
     *        slot indexed by rank, not a thread-local; its slab comes at the
     *        rank's first allocation). Any thread, no parallel region.
     * @param manager    The manager being registered.
     * @param block_size Allocation block (slab) size, in bytes.
     */
    void init_manager(ManagerChunkInt& manager, int block_size) const;

  private:
    /// Resizes every manager's per-thread free-block lists to the current thread count.
    void resize_all_list_block_free_thr() const;

    /// Allocates the first block of each per-thread allocator (no-op in std-alloc builds).
    void alloc_chunk_allocator_thr() const;
};

/// @brief Serial builds: one allocator per type, and nothing deferred.
template <typename Derived>
class AllTypeChunkParallelInterface<Derived, false>
{
  public:
    /// Destroys the single chunk allocator of every type (no-op in std-alloc builds).
    void delete_chunk_allocator() const;

    /// Serial build: nothing is deferred, so there is nothing to reclaim.
    static void destroy_after_flush(bool /*in_parallel*/) {}
    /// Serial build: writes are applied inline, so there is no queue to drain.
    static bool flush_all_deferred(bool /*in_parallel*/)
    {
        return false;
    }

  protected:
    /**
     * @brief The single allocator of ONE manager, at its registration.
     * @param manager    The manager being registered.
     * @param block_size Allocation block (slab) size, in bytes.
     * @note No-op in std-alloc builds.
     */
    void init_manager(const ManagerChunkInt& manager, int block_size) const;
};

/**
 * @brief The runtime-geometry half of the registry.
 * @tparam Enabled build_config::chunk_dynamic.
 */
template <typename Derived, bool Enabled = build_config::chunk_dynamic>
struct AllTypeChunkDynInterface; // primary left undefined: only the 2 specializations exist

/// @brief Dynamic-geometry builds: chunk sizes are declared at run time via register_chunk_size().
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
     * @param block_size Allocation block (slab) size, in bytes, for that size.
     */
    void register_chunk_size(unsigned int size, int block_size);
};

/// @brief Fixed-geometry builds: no geometry to declare.
template <typename Derived>
struct AllTypeChunkDynInterface<Derived, false>
{
};

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
 * Access is through <tt>SingletonChk\<AllTypeChunk\>\::getRef()</tt>. The registry starts
 * empty and fills itself: the first array of each (chunk template, element
 * type) registers its manager from its create() (ensure_manager). The first
 * array of the process also runs ParamChunk::init_indexes(true) if the program
 * has not called it (ensure_ready), so an array may be constructed anywhere,
 * a global one included. A program may also register a list of types up front
 * with ParamChunk::init_indexes<Types>().
 *
 * @note The available surface depends on the build, and the parts that vary
 *       live in the two policies above:
 *       AllTypeChunkParallelInterface picks the per-thread allocator and
 *       free-list methods over the single-allocator ones,
 *       AllTypeChunkDynInterface adds register_chunk_size().
 */
class AllTypeChunk : public AllTypeChunkParallelInterface<AllTypeChunk>, public AllTypeChunkDynInterface<AllTypeChunk>
{
  public:
    friend class ::fast_alloc::Singleton<AllTypeChunk>; // SingletonChk<AllTypeChunk>
    /// The dynamic-geometry policy records the sizes it declares (registered_sizes_).
    template <typename D, bool E>
    friend struct AllTypeChunkDynInterface;

    /// The managers' interface, reached through Derived by the policies so
    /// that their bodies stay dependent (each build declares a different half).
    using manager_int_t = ManagerChunkInt;

    /**
     * @brief Builds an EMPTY registry; managers arrive with the first array of
     *        each type (ensure_manager), or up front through initialize<Types>().
     *
     * Private-by-convention: instances are created only by
     * `SingletonChk<AllTypeChunk>`, which is a friend.
     *
     * @note Out of line (all_type_chunk.cpp): it checks, once per process,
     *       that the OMPT tool of an OMPT build is loaded
     *       (check_ompt_tool_loaded, manage_ompt.h).
     */
    AllTypeChunk();

    /// Non-copyable: the registry owns the managers and is a singleton.
    AllTypeChunk(const AllTypeChunk&)            = delete;
    AllTypeChunk& operator=(const AllTypeChunk&) = delete;
    AllTypeChunk(AllTypeChunk&&)                 = delete;
    AllTypeChunk& operator=(AllTypeChunk&&)      = delete;
    virtual ~AllTypeChunk()                      = default;

    /// @brief ParamChunk::init_indexes() has set the block size: managers may
    ///        now be registered. Records the block size they are built with.
    void mark_ready(int block_size)
    {
        const std::scoped_lock lock(registry_mutex_);
        block_size_ = block_size;
        ready_      = true;
    }

    /// @return The registered managers, in registration order: the prefix
    ///         published so far, read without a lock (see the members below).
    [[nodiscard]] std::span<ManagerChunkInt* const> getListTypeChunk() const noexcept
    {
        // The count first: every table published after the slot count-1 was
        // written holds that slot, so the table read next has all of them.
        const auto count = static_cast<std::size_t>(nb_managers_.load(std::memory_order_acquire));
        return {table_.load(std::memory_order_acquire), count};
    }

    /**
     * @brief ParamChunk::init_indexes(true), once, if the program has not run
     *        init_indexes() itself before its first array.
     * @details Called by ensure_manager(), i.e. by the first array of each
     *          type, outside the registry lock. Any context the first array can
     *          be built in: a global array (static initialisation), a parallel
     *          region (std::call_once serialises the threads). An explicit
     *          init_indexes() call still works, before or after.
     */
    static void ensure_ready();

    /// Compresses every compressible chunk of every type, ignoring the candidate lists.
    void force_chunk_to_compress() const;

    /**
     * @brief Drains the compression-candidate backlog of every type.
     *
     * @param apply true: every candidate found constant is compressed -- what
     *        the between-region passes want; false: the lists are walked,
     *        their flags reset and (chunk2) emptied, and NOTHING is
     *        compressed -- the terminal sweep of a dying array uses it to
     *        purge its chunks from the lists. false does NOT drain: a
     *        caller counting "misses" after it sees only what it un-queued.
     */
    void compress_chunk_to_compress_potentially(bool apply) const;

    /// @return True as soon as one type has at least one pending compression candidate.
    [[nodiscard]] bool have_chunk_to_compress_potentially() const;
    /// Frees the allocation blocks of every type (no-op in std-alloc builds).
    void destroy_block_chunk() const;

    /**
     * @brief Bytes parked across every manager's reuse pools, at the moment of
     *        the call.
     *
     * @details Printed with CHUNK_STATS (see param_chunk.cpp). The stats counters
     *          measure EVENTS -- allocations, reuses, compressions -- and say
     *          nothing about what is still held when a run ends; this does (the
     *          pools can hold most of the resident memory).
     *
     * @note Declared in every build, not in the OpenMP policy: the reuse pools
     *       exist in every build, and so does the CHUNK_STATS block of
     *       param_chunk.cpp that prints them.
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
    [[nodiscard]] std::size_t trim_reuse_pools(std::size_t keep_per_thread) const;

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
     * read the registry without a lock (see the members below).
     *
     * Runs ensure_ready() first: init_indexes() is optional.
     */
    template <typename ManagerT>
    static void ensure_manager();

    /**
     * @brief Registers every element type of a tuple up front, for the chunk
     *        templates the build compiles arrays for.
     *
     * Called by `ParamChunk::init_indexes<Types>()`, hence public.
     * Optional since the arrays register their own types: what it buys is
     * having every manager in place before the first parallel region, at the
     * cost of managers for types the program may never use.
     *
     * @tparam Types Tuple of element types the program will store in chunks.
     */
    template <typename Types>
    void initialize();

  private:
    // ── The registry: append-only, published without a lock ────────────────
    //
    // A manager is registered the first time an array of its (chunk template,
    // element type) is constructed (ensure_manager, from the array's create),
    // not from a list of types declared at startup: the element types of a
    // program are whatever it instantiates. Registration takes the mutex,
    // builds the manager and its per-rank structures in full, stores it in
    // the next slot and publishes the count with a release store. The
    // fan-outs above iterate the published prefix -- an acquire load of the
    // count, then the slots -- with no lock: they run from parallel regions
    // (flush, compression passes, load balancing) and may overlap a
    // registration, and never see a manager before it is whole. Nothing is
    // ever removed before the teardown, so the slots are stable.
    //
    // No capacity limit. The slots are a table of manager pointers; when it is
    // full, registration copies it into one twice as large and publishes that.
    // The old tables are kept, not freed -- a reader may still be walking a
    // span of one -- so their memory (64, 128, 256... pointers: less than the
    // last table again) lives as long as the registry.
    static constexpr std::size_t kFirstTableSize = 64;

    /// @brief Store @p manager in the next slot, growing the table if it is
    ///        full, and publish it. Called with registry_mutex_ held.
    void publish_manager(std::unique_ptr<ManagerChunkInt> manager);

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

    /// Resizes every manager's compression-candidate list to the current thread count.
    void resize_chunk_to_compress_potentially() const;

    /// The managers, owned; touched only under registry_mutex_.
    std::vector<std::unique_ptr<ManagerChunkInt>> owned_;
    /// Every slot table published so far, the current one last (see above).
    std::vector<std::unique_ptr<ManagerChunkInt*[]>> tables_;
    /// The current slot table, for the lock-free readers.
    std::atomic<ManagerChunkInt**> table_{nullptr};
    std::size_t                    table_capacity_ = 0;
    std::atomic<int>               nb_managers_{0};
    std::mutex                     registry_mutex_;

    /// Set under registry_mutex_ when the manager type @p ManagerT is
    /// registered: the duplicate check of register_manager(), with no RTTI.
    template <typename ManagerT>
    static inline bool registered_ = false;

    /// Set by ParamChunk::init_indexes(): the registry accepts managers only
    /// once the block size (and the stats) exist. An array constructed before
    /// that aborts with a message naming the call.
    bool ready_      = false;
    int  block_size_ = 0;

    /// Runtime geometries declared so far (CHUNK_DYNAMIC): replayed on every
    /// manager registered after them.
    std::vector<unsigned int> registered_sizes_;

    // The block-free-list verbs (load_balance_list_block_free() & co.) live in
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
 *          allocates and frees millions of buffers across many threads therefore
 *          pays for the high-water mark of EVERY arena, not of the process:
 *          capping the arenas at one roughly halves the RSS of the benchmark,
 *          with the same bytes allocated.
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
