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

#include <algorithm>        // std::min (operator==)
#include <atomic>           // std::atomic (operator==: a difference found)
#include <cstddef>          // std::size_t
#include <cstdint>          // std::uint8_t, std::uint32_t (reclaim_second_values' mark)
#include <iostream>         // for cout, ostream
#include <utility>          // for move
#include <vector>           // for vector, swap
#include <unordered_set>    // for the cst2 GC live set
#include <initializer_list> // the {a, b, c} constructor
#include <iterator>         // std::forward_iterator, std::distance -- the (first, last) constructor
#include <type_traits>      // std::is_same_v (reclaim_second_values: bool has no registry)

#include "chunk/internal/second_value_array_base.h"     // for SecondValueArrayBase
#include "chunk/internal/zero_any_array_parallel.h"     // zero_any::ParallelSupport — OpenMP-only mixin
#include "chunk/internal/array_chunk_rechunk_hooks.h"   // RechunkHooks — CHUNK_DYNAMIC-only mixin
#include "chunk/internal/array_chunk_base.h"            // for ArrayChunkBase
#include "chunk/internal/concept_array_chunk_element.h" // for ArrayChunkElement
#include "chunk/config.h"                               // for threaded, g_in_rechunk_rebuild
#include "chunk/PMP_CHUNK.h"                    // CHUNK_PMP_IN_PARALLEL_REGION (reclaim_second_values), pmp::Team
#include "chunk/internal/tab_cst_val2_holder.h" // for GlobalVecHolder (cst2 GC)
#include "chunk/internal/write_compress_sync_policy.h" // for WriteCompressSyncPolicy
#include "chunk/internal/array_chunk_core.h"           // the core the bases share
#include "chunk/internal/zero_any_proxy.h"             // zero_any::Proxy<T> and its tables
#include "chunk/zero_any_chunk.h"                      // zero_any::Chunk<T>
#include "chunk/log.h"                                 // CHUNK_LOG, CHUNK_LOG_ALWAYS (the second values)

namespace shrink_array
{

template <template <typename> class ChunkT, typename T>
class ManagerChunk;

namespace zero_any
{

/// @brief reclaim_second_values(): identifier @p id is still referenced.
inline void mark_cst2_id(std::vector<std::uint8_t>& referenced, std::uint32_t id) noexcept
{
    if (id != 0 && id < referenced.size())
    {
        referenced[id] = 1;
    }
}

/**
 * @brief The mark of one member of the zero_any family, for
 *        reclaim_second_values(): every second-value identifier its live
 *        arrays still name, set in @p referenced.
 * @details Over every array of the member, whatever its index type (the keys of
 *          ProxyT's registry of arrays):
 *            - a chunk compressed to a second value: its values pointer is the slot;
 *            - a per-chunk second-value slot (ProxyT's table) naming a registered value
 *              (init_values_cst_val() finds the slot from it by value).
 *          The members with the same element type share one registry
 *          (GlobalVecHolder<T>): each registers this function for itself
 *          (GlobalVecHolder::register_marker(), from acquire_cst2_registry()), and
 *          a reclaim on any of them runs all the marks.
 * @tparam T      Element type (not bool: bool has no registry).
 * @tparam ProxyT The member's proxy: its registry of arrays and second-value table.
 * @tparam ChunkT The member's chunk type: the registry's keys.
 * @param referenced One flag per identifier.
 */
template <typename T, typename ProxyT, typename ChunkT>
void mark_second_values(std::vector<std::uint8_t>& referenced)
{
    using Gvh = GlobalVecHolder<T>;

    // Chunks compressed to a second value.
    ProxyT::share_info_array->for_each_key(
        [&referenced](ArrayKey<ChunkT> arr)
        {
            if (arr == nullptr)
            {
                return;
            }
            // The key is the array's core, whatever its index type.
            const int nc = arr->chunk_count();
            for (int j = 0; j < nc; ++j)
            {
                const T* vals = arr->chunk_ref(static_cast<std::size_t>(j)).values_load();
                if (Gvh::region_contains(vals))
                {
                    mark_cst2_id(referenced, Gvh::slot_id(vals));
                }
            }
        });

    // Every slot of every row: a released row is empty, a stale entry only keeps a
    // value one pass longer.
    using Spec              = ProxyT::BaseSpec;
    const std::size_t nrows = Spec::second_value_row_count();
    for (std::size_t r = 0; r < nrows; ++r)
    {
        const auto& row = Spec::get_tab_second_value(static_cast<int>(r));
        for (const auto& slot : row)
        {
            if constexpr (SecondValueTraits<T>::indexed)
            {
                mark_cst2_id(referenced, slot); // the identifier itself; 0 = none
            }
            else if (!(slot == T{}))
            {
                if (const T* hit = Gvh::index_lookup(slot))
                {
                    mark_cst2_id(referenced, Gvh::slot_id(hit));
                }
            }
        }
    }
}

/**
 * @brief The body of zero_any::Array, shared by the zero_any family
 *        (chunk_traits.h): a member's array class derives from
 *        ArrayImpl<itself>.
 * @details Everything that names the element, index, chunk, proxy or manager
 *          type takes it from ArrayChunkTraits<Derived>, so a member gets its
 *          own registry (ProxyType::share_info_array), metadata tables and
 *          candidate lists. The constructors, the destructor and the
 *          assignments stay in the final class -- a constructor is not
 *          inherited, and the copy and move ones name that class -- built on
 *          init_storage() and destroy_storage(). zero_any::Array<T, IntType>
 *          derives from ArrayImpl<Array<T, IntType>> and adds only those.
 * @tparam Derived The member's array class.
 */
template <typename Derived>
class ArrayImpl : public SecondValueArrayBase<Derived>, public ParallelSupport<Derived>, public RechunkHooks<Derived>
{
  public:
    using Base2   = SecondValueArrayBase<Derived>; ///< Second-value common base (dirty bitset, second-value plumbing).
    using Base    = ArrayChunkBase<Derived>;       ///< Base shared by every variant.
    using Hooks2A = RechunkHooks<Derived>;         ///< Re-chunk hooks (dynamic geometry).

    friend Base2; ///< The CRTP base reaches this class's private members.
    friend Base;  ///< For swap_identity().

    // Re-declared (not only inherited): the mixins declare them too, so an outside
    // `Arr::ChunkType` would be ambiguous between base subobjects (lookup precedes access).
    using Traits      = Base2::Traits;      ///< ArrayChunkTraits<Derived>.
    using T           = Base2::T;           ///< Stored element type.
    using IntType     = Base2::IndType;     ///< Index type used for sizes and element positions.
    using ChunkType   = Base2::ChunkType;   ///< The chunk type held in chunk_list (zero_any::Chunk<T>).
    using ManagerType = Base2::ManagerType; ///< The per-type manager (ManagerChunk<zero_any::Chunk, T>).
    using ProxyType   = Base2::ProxyType;   ///< The write proxy of the non-const accessors (zero_any::Proxy<T>).
    using value_type  = T;                  ///< Stored element type.

    using ManagerChunk2A = ManagerType; ///< Same type as ManagerType.

    // The former snake_case spellings, kept until 1.0 (types are PascalCase).
    using chunk_type [[deprecated("types are PascalCase: use ChunkType")]]          = ChunkType;
    using manager_type [[deprecated("types are PascalCase: use ManagerType")]]      = ManagerType;
    using proxy_type [[deprecated("types are PascalCase: use ProxyType")]]          = ProxyType;
    using ind_type [[deprecated("types are PascalCase: use IndType")]]              = IntType;
    using chunk_list_type [[deprecated("types are PascalCase: use ChunkListType")]] = Base::ChunkListType;

    using Base2::chunk_list_;
    using Base2::m_size_;
    using Base2::nb_chunk_;
    using Base2::resize;

    using Base::nb_tab;
    using Base::num_tab_;
    using Base::size_policy_;

    // Mode-specific members of the mixins. Required: ArrayChunkBase re-exports RechunkSupport's
    // hook DEFAULTS under these names, so a hook call through self() would be ambiguous.
    using Hooks2A::rechunk_drain_reuse_list;
    using Hooks2A::rechunk_drop_candidates;
    using Hooks2A::rechunk_swap_identity;

    using Base::assign; // the (first, last) overload, hidden by the one above otherwise

  protected:
    // ── What the final class's constructors and destructor are built on ──────
    //
    // Never copied or moved as such: the final class's copy and move operations
    // build an empty array and then copy the elements or exchange the storage.

    ArrayImpl()  = default;
    ~ArrayImpl() = default;

    ArrayImpl(const ArrayImpl&)            = delete;
    ArrayImpl(ArrayImpl&&)                 = delete;
    ArrayImpl& operator=(const ArrayImpl&) = delete;
    ArrayImpl& operator=(ArrayImpl&&)      = delete;

    /**
     * @brief The constructors' body: build the storage for @p sz elements at
     *        the current active geometry, and claim the array's identity.
     * @param sz initial element count
     */
    void init_storage(IntType sz)
    {
        if constexpr (build_config::threaded)
        {
            WriteCompressSyncPolicy<ProxyType>::ensure_initialized();
        }

        this->create(sz);
    }

    /**
     * @brief The destructor's body: release the storage, then the registry entry.
     * @warning The erase() is required: a later array at this address (common for stack
     *          arrays, e.g. rechunk_to()'s @c tmp) would inherit the stale num_tab
     *          (get_num_tab() returns the FIRST match) and corrupt another tab's metadata.
     */
    void destroy_storage()
    {
        clear();
        ProxyType::share_info_array->erase(this->registry_key());
    }

  public:
    /**
     * @brief Has the chunk holding element @p index been materialised?
     * @param index element index
     * @return false while that chunk is still compressed to a constant
     */
    [[nodiscard]] bool written(IntType index) const noexcept
    {
        return chunk_list_[this->chunk_index_of(index)].written();
    }

    /**
     * @brief Resize the array, preserving the elements that still fit.
     * @param newsz new element count
     * @note Shrinking releases the dropped chunks' buffers (release_chunk_range():
     *       freed serially, parked in the geometry's reuse list under OpenMP);
     *       chunk_list keeps its capacity.
     */
    void resize(IntType newsz);

  private:
    /// @brief Release the buffers of chunks [@p first, @p last) (a shrinking resize()). Same
    ///        release as clear_core(): freed serially; under OpenMP parked in this geometry's
    ///        reuse list, chunk set to cst_val1. Sequential point; no candidate list may hold them.
    void release_chunk_range(int first, int last);

  public:
    /**
     * @brief Resize to @p count elements, all set to @p val.
     * @param count new element count
     * @param val   value every element takes
     *
     * @note Fills [0, count), not only the new elements: Base2::resize(count, val) has
     *       std::vector::resize semantics, so this goes through ArrayChunkBase::assign.
     */
    void assign(IntType count, T val) // T, not InT<T>: see ArrayChunkBase::assign
    {
        Base::assign(count, val);
    }

    /**
     * @brief Element-wise equality.
     * @param other array to compare against
     * @return true if both arrays have the same size and the same values
     * @note Compares decompressed values, so two arrays that are equal but
     *       compressed differently still compare equal. A const ArrayImpl&, not
     *       the final type: C++20's reversed candidate would make `a == b`
     *       ambiguous (one derived-to-base conversion each way). Chunk by chunk, on a
     *       team of its own for a large array outside a region
     *       (parallel_storage_init::for_each_chunk); a difference found stops
     *       the chunks not yet started. Neither array may be written meanwhile.
     */
    bool operator==(const ArrayImpl& other) const noexcept;

    /**
     * @brief Release all storage and report end-of-life statistics.
     * @details clear_core() plus the end-of-life I/O (compression stats and the
     *          size predictor).
     * @see clear_core
     */
    void clear();

    /// @brief Run a compression pass over this array's chunks, shared over
    ///        @p team (the one compress() opens, or the caller's region).
    void compress_int(pmp::Team& team);

    /// @brief compress_int() on the calling thread's current team
    ///        (pmp::current_team(): the caller's region, or a team of one).
    void compress_int()
    {
        pmp::Team team = pmp::current_team();
        compress_int(team);
    }

    /// @brief Read-only access to the underlying chunk list.
    /// @return const reference to the chunks backing this array
    [[nodiscard]] const Base::ChunkListType& get_chunk_list() const noexcept
    {
        return chunk_list_;
    }

    /// @brief Former name of get_chunk_list(), kept for the programs that still spell it.
    [[deprecated("methods are snake_case: use get_chunk_list")]] [[nodiscard]] const Base::ChunkListType&
    getChunkList() const noexcept
    {
        return chunk_list_;
    }

    /**
     * @brief Report how many cst2 buffers a sweep could reclaim.
     * @details Runs gc_collect_live() and compares the live count with
     *          GlobalVecHolder's total; the difference is the reclaimable set.
     *          Diagnostic only -- a report (written whatever CHUNK_LOG says,
     *          channel compress) -- and frees nothing.
     */
    static void gc_mark_report()
    {
        std::unordered_set<const void*> live;
        gc_collect_live(live);
        const std::size_t held = GlobalVecHolder<T>::count_entries();
        CHUNK_LOG_ALWAYS(compress, "cst2-gc: live={} / holder={}  (reclaimable={})", live.size(), held,
                         held > live.size() ? held - live.size() : 0);
    }

    /**
     * @brief Free the second values that no chunk of any array of the
     *        zero_any family with element type T (zero_any::Array<T>,
     *        pattern::Array<T>) refers to any more, so that the registry
     *        reuses their identifiers instead of carving new ones.
     *
     * @details The registry (GlobalVecHolder) forgets nothing on its own: drifting constants
     *          register new values until the identifiers run out (CHUNK_CST_VAL2_MAX_IDS),
     *          which stops the run. This marks what is referenced, over every live array of
     *          every member of the family sharing the registry, whatever its index type
     *          (mark_second_values(), one per member, GlobalVecHolder::mark_all()), and
     *          frees the rest. O(chunks of every such array) + O(registered values).
     *
     * @return How many identifiers were freed; 0 when refused: inside a parallel region,
     *         during a re-chunk rebuild, or with a degraded index (the mark could miss a
     *         value unfindable by value). A refusal is counted by cause in
     *         second_value_stats(), which tells it from a call with nothing to free.
     *
     * @pre   A sequential point over the whole TYPE, not one array: no thread
     *        writes to or compresses ANY zero_any::Array<T> of the program
     *        (whatever its index type, whoever owns it) from the call to its
     *        return, nor holds the result of a second-value lookup it has not
     *        recorded yet. A hard precondition, guarded by contract only (no
     *        epoch, no reader count): docs/api.md, section 3.
     * @warning Out of contract the failure is SILENT: the mark misses a slot a
     *          chunk is about to adopt, frees it, take_free_slot() hands its
     *          identifier to another value, and that chunk reads the other
     *          value. The refusals above see the CALLING thread only: a
     *          std::thread / TBB pool or a team started by another host thread
     *          is not detected -- except under the std backend between a
     *          begin_parallel_phase() and the finalize_parallel_phase() of the
     *          phase contract, where in_parallel() is true on every thread
     *          (the application phase, docs/threading_backends.md step 7c;
     *          std_thread_validation checks the refusal). Check builds fail an
     *          invariant on any lookup or registration overlapping the reclaim
     *          (GlobalVecHolder's LookupGuard / ReclaimScope); release builds
     *          check nothing.
     */
    static std::size_t reclaim_second_values();

    /**
     * @brief Where this type's second-value identifiers stand: in use against
     *        the capacity, the reclaims run and refused, the degradation.
     *
     * @details Running out of identifiers stops the run; sampled at the
     *          application's own checkpoints, in_use() / capacity is the
     *          early warning, and refused() tells a reclaim that freed
     *          nothing because it could not run. Covers every zero_any::Array<T>,
     *          whatever its index type, like reclaim_second_values(). Relaxed
     *          reads, callable at any time, from any thread: exact at a
     *          sequential point. All zero for bool (no registry).
     */
    [[nodiscard]] static SecondValueStats second_value_stats() noexcept;

    /// @brief Compress every chunk currently queued as a candidate.
    static void make_block_compression();

  private:
    /**
     * @brief Claim this array's identity before its storage is built.
     * @details Called by create() through ArrayChunkBase: registers the address and takes
     *          the next free num_tab.
     * @post num_tab is valid and the array is discoverable by address.
     */
    void pre_create_hook()
    {
        auto& sh_inf = *ProxyType::share_info_array;

        sh_inf.insert(this->registry_key(), nb_tab);
        num_tab_ = nb_tab;
        ++nb_tab;
    }

    /**
     * @brief (Re)allocate storage at the current active geometry.
     *
     * @details create()'s allocation WITHOUT identity/registration. Used by create(), and
     *          by rechunk_to() to rebuild a registered, torn-down array at a new geometry
     *          under the same num_tab.
     *
     * @param sz element count to allocate for
     */
    void rebuild_storage(IntType sz);

    /**
     * @brief Second value registered for the chunk holding element @p index.
     * @param index element index
     * @return this chunk's own second value (zero_any registers one per chunk)
     */
    auto get_metadata(IntType index) const noexcept
    {
        return ProxyType::BaseSpec::get_tab_second_value(num_tab_, this->chunk_index_of(index));
    }

    /// @brief Storage teardown shared by clear() and rechunk_to().
    ///
    /// Everything clear() does EXCEPT the end-of-life I/O: drain, terminal compression,
    /// buffer release, metadata reset. rechunk_to() must release storage through this same
    /// (ledger-correct) path. In non-OMP builds it does NOT tear down the shared cst_val2
    /// holder, so it does not affect other arrays.
    void clear_core();

    /**
     * @brief Print this array's compression ratio and verdict.
     * @param os destination stream
     */
    void print_compression_stats(std::ostream& os = std::cout) const;

    /**
     * @brief Take this array's reference on the cst2 registry, once, as soon
     *        as it has chunks.
     * @details The registry is shared by every array of the zero_any family with element
     *          type T and reference-counted (the last release frees everything). Given back
     *          in clear_core() only if taken: an array built empty and grown by resize()
     *          needs one too, and an empty array's teardown must not drop one it never took.
     *          Registers this member's mark with the registry on the way, for
     *          reclaim_second_values().
     */
    void acquire_cst2_registry();

    /// @brief The identity step of swap(): num_tab and the registry entry (the
    ///        per-chunk metadata rows follow the chunks), the cst2 registry
    ///        reference (it goes with the chunks that need it).
    void swap_identity(Derived& other)
    {
        using std::swap;
        swap(this->num_tab_, other.num_tab_);
        swap(holds_cst2_registry_, other.holds_cst2_registry_);
        ProxyType::share_info_array->swap_keys(this->registry_key(), other.registry_key());
    }

    // ── cst2 mark-sweep GC: read-only mark + report ──────────────────────────

    /**
     * @brief Mark phase of the cst2 garbage collection: collect live buffers.
     *
     * @details Scans every live zero_any::Array<T> through the shared registry and inserts the
     *          cst2 buffer pointer of each chunk compressed to its second value. Read-only:
     *          safe at any quiescent point.
     *
     * @param[out] live set the referenced cst2 buffer pointers are inserted into
     * @see gc_mark_report for the reporting counterpart
     */
    static void gc_collect_live(std::unordered_set<const void*>& live);

    /// @brief This array holds one reference on the chunk type's cst2 registry
    ///        (TabCstVal2Holder). See acquire_cst2_registry().
    bool holds_cst2_registry_ = false;
};

/**
 * @brief Chunked array whose chunks may each compress to their OWN second value.
 *
 * @details Same layout as zero_one_value::Array, but where zero_one_value::Array shares a SINGLE second value
 *          across the array, here every chunk registers its own (find-or-set), so a chunk
 *          compresses as soon as all its elements equal any one constant. Costs a per-chunk
 *          second-value table (zero_any::ProxySpec), garbage-collected through GlobalVecHolder.
 *
 *          Each array registers its address in zero_any::Proxy::share_info_array to get a num_tab,
 *          the index into every per-tab metadata table. That registration is the array's
 *          identity: taken in pre_create_hook(), exchanged by rechunk_swap_identity(),
 *          released in the destructor.
 *
 *          All of it is ArrayImpl's, shared by the zero_any family; this class adds the
 *          constructors, the destructor and the assignments.
 *
 * @tparam T        stored element type
 * @tparam IntType index type used for sizes and element positions
 */
template <ArrayChunkElement T, typename IntType>
class Array : public ArrayImpl<Array<T, IntType>>
{
  public:
    /**
     * @brief Build an array of @p sz elements at the current active geometry.
     * @param sz initial element count; 0 by default, as std::vector (a fresh
     *           array is empty, so a push_back on it yields one element)
     */
    explicit Array(IntType sz = 0)
    {
        this->init_storage(sz);
    }

    // ── The copy family: deep, element by element ────────────────────────────
    //
    // Never an alias of the source's buffers: ArrayChunkBase::copy_from reads every element
    // through the source's gate and writes it through this array's proxy. All sequential.
    // zero_any finds each chunk's second value by itself: nothing to carry over.

    /// @brief Deep copy: from.size() elements read from @p from. Sequential context.
    /// @param from Source array.
    Array(const Array& from) : Array(from.size())
    {
        this->copy_from(from);
    }

    /**
     * @brief Move-construct: an empty array at the active geometry takes
     *        @p other's chunks, geometry and identity -- the exchange of
     *        swap() -- and @p other is left empty and valid.
     * @details O(1) in the elements. Must be user-declared: the copy constructor and
     *          move-assignment suppress the implicit one. Sequential context.
     * @note noexcept, so std::vector moves arrays on reallocation. What can fail (the
     *       registration, the deferred-write drain) fails only on memory exhaustion or a
     *       corrupt registry, already fatal: std::terminate.
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
    Array(std::initializer_list<T> il) : Array(il.begin(), il.end()) {}

    /// @brief Release the storage, then the registry entry (ArrayImpl::destroy_storage()).
    ~Array()
    {
        this->destroy_storage();
    }

    /**
     * @brief Move-assign: this array's own contents are released (clear()),
     *        then it takes @p other's -- chunks, geometry and identity, by the
     *        same exchange as swap(); @p other is left empty and valid.
     * @details This array's old chunks are released first (candidates purged); the moved
     *          chunks keep their metadata rows (num_tab). Sequential context, as clear().
     * @note noexcept, as the move constructor: a failure here terminates.
     */
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
};

} // namespace zero_any

} // namespace shrink_array

// Template method bodies, after the class definition.

#include "chunk/internal/zero_any_array_inline.h" // the template bodies

// Bodies of the two mode-specific mixins. After zero_any_array_inline.h, so their
// CRTP self() sees a complete zero_any::Array.
#include "chunk/internal/zero_any_array_parallel_inline.h"   // zero_any::ParallelSupport's bodies
#include "chunk/internal/array_chunk_rechunk_hooks_inline.h" // RechunkHooks' bodies
