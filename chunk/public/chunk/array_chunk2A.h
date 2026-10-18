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

#include "chunk/internal/array_chunk2_base.h"           // for ArrayChunk2Base
#include "chunk/internal/array_chunk2A_parallel.h"      // ParallelSupport2A — OpenMP-only mixin
#include "chunk/internal/array_chunk_rechunk_hooks.h"   // RechunkHooks — CHUNK_DYNAMIC-only mixin
#include "chunk/internal/array_chunk_base.h"            // for ArrayChunkBase
#include "chunk/internal/concept_array_chunk_element.h" // for ArrayChunkElement
#include "chunk/config.h"                               // for threaded, g_in_rechunk_rebuild
#include "chunk/PMP_CHUNK.h"                            // CHUNK_PMP_IN_PARALLEL_REGION (reclaim_second_values), pmp::Team
#include "chunk/internal/tab_cst_val2_holder.h"         // for GlobalVecHolder (cst2 GC)
#include "chunk/internal/write_compress_sync_policy.h"  // for WriteCompressSyncPolicy
#include "chunk/internal/array_chunk_core.h"            // the core the bases share
#include "chunk/internal/proxy2A.h"                     // Proxy2A<T> and its tables
#include "chunk/chunk2A_std.h"                          // Chunk2A<T>
#include "chunk/log.h"                                  // CHUNK_LOG, CHUNK_LOG_ALWAYS (the second values)

namespace shrink_array
{

template <template <typename> class CHUNK_T, typename T>
class ManagerChunk;

/**
 * @brief Chunked array whose chunks may each compress to their OWN second value.
 *
 * @details Same layout as ArrayChunk2, but where ArrayChunk2 shares a SINGLE second value
 *          across the array, here every chunk registers its own (find-or-set), so a chunk
 *          compresses as soon as all its elements equal any one constant. Costs a per-chunk
 *          second-value table (ProxySpec2A), garbage-collected through GlobalVecHolder.
 *
 *          Each array registers its address in Proxy2A::share_info_array to get a num_tab,
 *          the index into every per-tab metadata table. That registration is the array's
 *          identity: taken in pre_create_hook(), exchanged by rechunk_swap_identity(),
 *          released in the destructor.
 *
 * @tparam T        stored element type
 * @tparam ind_type index type used for sizes and element positions
 */
template <ArrayChunkElement T, typename ind_type>
class ArrayChunk2A : public ArrayChunk2Base<ArrayChunk2A<T, ind_type>>,
                     public ParallelSupport2A<ArrayChunk2A<T, ind_type>>,
                     public RechunkHooks<ArrayChunk2A<T, ind_type>>
{
  public:
    using manager_chunk2A = ManagerChunk<Chunk2A, T>; ///< Same type as manager_type.

    using Base2   = ArrayChunk2Base<ArrayChunk2A>; ///< Chunk2/2A common base (dirty bitset, second-value plumbing).
    using Base    = ArrayChunkBase<ArrayChunk2A>;  ///< Base shared by every variant.
    using Hooks2A = RechunkHooks<ArrayChunk2A>;    ///< Re-chunk hooks (dynamic geometry).

    friend Base2;                                     ///< The CRTP base reaches this class's private members.
    friend ArrayChunkBase<ArrayChunk2A>; ///< For swap_identity().

    // Re-declared (not only inherited): the mixins declare them too, so an outside
    // `Arr::chunk_type` would be ambiguous between base subobjects (lookup precedes access).
    using Traits       = Base2::Traits;       ///< ArrayChunkTraits<ArrayChunk2A>.
    using chunk_type   = Base2::chunk_type;   ///< Chunk2A<T>, the chunk type held in chunk_list.
    using manager_type = Base2::manager_type; ///< ManagerChunk<Chunk2A, T>, the per-type manager.
    using proxy_type   = Base2::proxy_type;   ///< Proxy2A<T>, the write proxy of the non-const accessors.
    using value_type   = T;                   ///< Stored element type.

    using Base2::chunk_list;
    using Base2::m_size;
    using Base2::nb_chunk;
    using Base2::resize;

    using Base::nb_tab;
    using Base::num_tab;
    using Base::size_policy_;

    // Mode-specific members of the mixins. Required: ArrayChunkBase re-exports RechunkSupport's
    // hook DEFAULTS under these names, so a hook call through self() would be ambiguous.
    using Hooks2A::rechunk_drain_reuse_list;
    using Hooks2A::rechunk_drop_candidates;
    using Hooks2A::rechunk_swap_identity;

    using Base::assign; // the (first, last) overload, hidden by the one above otherwise

    /**
     * @brief Build an array of @p sz elements at the current active geometry.
     * @param sz initial element count; 0 by default, as std::vector (a fresh
     *           array is empty, so a push_back on it yields one element)
     */
    explicit ArrayChunk2A(ind_type sz = 0)
    {
        if constexpr (build_config::threaded)
        {
            WriteCompressSyncPolicy<Proxy2A<T>>::ensure_initialized();
        }

        this->create(sz);
    }

    // ── The copy family: deep, element by element ────────────────────────────
    //
    // Never an alias of the source's buffers: ArrayChunkBase::copy_from reads every element
    // through the source's gate and writes it through this array's proxy. All sequential.
    // chunk2a finds each chunk's second value by itself: nothing to carry over.

    /// @brief Deep copy: from.size() elements read from @p from. Sequential context.
    /// @param from Source array.
    ArrayChunk2A(const ArrayChunk2A& from) : ArrayChunk2A(from.size())
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
    ArrayChunk2A(ArrayChunk2A&& other) noexcept : ArrayChunk2A(0)
    {
        this->swap_storage_and_identity(other);
    }

    /// @brief Build an array of std::distance(@p first, @p last) elements and assign them from the range.
    /// @param first Start of the source range.
    /// @param last End of the source range.
    template <std::forward_iterator It>
    ArrayChunk2A(It first, It last) : ArrayChunk2A(static_cast<ind_type>(std::distance(first, last)))
    {
        this->assign(first, last);
    }

    /// @brief Build an array holding the elements of @p il.
    /// @param il Source values.
    ArrayChunk2A(std::initializer_list<T> il) : ArrayChunk2A(il.begin(), il.end()) {}

    /**
     * @brief Release the storage, then the registry entry.
     * @warning The erase() is required: a later array at this address (common for stack
     *          arrays, e.g. rechunk_to()'s @c tmp) would inherit the stale num_tab
     *          (get_num_tab() returns the FIRST match) and corrupt another tab's metadata.
     */
    ~ArrayChunk2A()
    {
        clear();
        Proxy2A<T>::share_info_array->erase(this->registry_key());
    }

    /**
     * @brief Move-assign: this array's own contents are released (clear()),
     *        then it takes @p other's -- chunks, geometry and identity, by the
     *        same exchange as swap(); @p other is left empty and valid.
     * @details This array's old chunks are released first (candidates purged); the moved
     *          chunks keep their metadata rows (num_tab). Sequential context, as clear().
     * @note noexcept, as the move constructor: a failure here terminates.
     */
    ArrayChunk2A& operator=(ArrayChunk2A&& other) noexcept
    {
        if (this != &other)
        {
            clear();
            this->swap_storage_and_identity(other);
        }
        return *this;
    }

    /// @brief Deep copy-assignment (see the copy constructor). Self-assignment
    ///        is a no-op; num_tab is not copied, identity stays with the address.
    ArrayChunk2A& operator=(const ArrayChunk2A& from)
    {
        if (this != &from)
        {
            this->copy_from(from);
        }
        return *this;
    }

    /**
     * @brief Has the chunk holding element @p i been materialised?
     * @param i element index
     * @return false while that chunk is still compressed to a constant
     */
    [[nodiscard]] bool written(ind_type i) const noexcept
    {
        return chunk_list[this->chunk_index_of(i)].written();
    }

    /**
     * @brief Resize the array, preserving the elements that still fit.
     * @param newsz new element count
     * @note Shrinking releases the dropped chunks' buffers (release_chunk_range():
     *       freed serially, parked in the geometry's reuse list under OpenMP);
     *       chunk_list keeps its capacity.
     */
    void resize(ind_type newsz);

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
    void assign(ind_type count, T val) // T, not in_t<T>: see ArrayChunkBase::assign
    {
        Base::assign(count, val);
    }

    /**
     * @brief Element-wise equality.
     * @param other array to compare against
     * @return true if both arrays have the same size and the same values
     * @note Compares decompressed values, so two arrays that are equal but
     *       compressed differently still compare equal. Chunk by chunk, on a
     *       team of its own for a large array outside a region
     *       (parallel_storage_init::for_each_chunk); a difference found stops
     *       the chunks not yet started. Neither array may be written meanwhile.
     */
    bool operator==(const ArrayChunk2A& other) const noexcept
    {
        if (m_size != other.m_size)
        {
            return false;
        }
        if (m_size == 0)
        {
            return true;
        }
        const auto        csz = static_cast<long long>(this->chunk_size());
        std::atomic<bool> differ{false};
        parallel_storage_init::for_each_chunk(
            this->chunk_index_of(m_size - 1) + 1,
            [&](int c)
            {
                if (differ.load(std::memory_order_relaxed))
                {
                    return;
                }
                const long long lo = static_cast<long long>(c) * csz;
                const long long hi = std::min(lo + csz, static_cast<long long>(m_size));
                for (long long i = lo; i < hi; ++i)
                {
                    if (Base2::read_value(static_cast<ind_type>(i)) != other.read_value(static_cast<ind_type>(i)))
                    {
                        differ.store(true, std::memory_order_relaxed);
                        return;
                    }
                }
            });
        return !differ.load(std::memory_order_relaxed);
    }

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
    [[nodiscard]] const Base::chunk_list_type& getChunkList() const noexcept
    {
        return chunk_list;
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
     * @brief Free the second values that no chunk of any ArrayChunk2A<T>
     *        refers to any more, so that the registry reuses their
     *        identifiers instead of carving new ones.
     *
     * @details The registry (GlobalVecHolder) forgets nothing on its own: drifting constants
     *          register new values until the identifiers run out (CHUNK_CST_VAL2_MAX_IDS),
     *          which stops the run. This marks what is referenced, over every live array of
     *          the type whatever its index type, and frees the rest:
     *            - a chunk compressed to a second value: its values pointer is the slot;
     *            - a per-chunk second-value slot (Proxy2A's table) naming a registered value
     *              (init_values_cst_val() finds the slot from it by value).
     *          O(chunks of every ArrayChunk2A<T>) + O(registered values).
     *
     * @return How many identifiers were freed; 0 when refused: inside a parallel region,
     *         during a re-chunk rebuild, or with a degraded index (the mark could miss a
     *         value unfindable by value). A refusal is counted by cause in
     *         second_value_stats(), which tells it from a call with nothing to free.
     *
     * @pre   A sequential point over the whole TYPE, not one array: no thread
     *        writes to or compresses ANY ArrayChunk2A<T> of the program
     *        (whatever its index type, whoever owns it) from the call to its
     *        return, nor holds the result of a second-value lookup it has not
     *        recorded yet. A hard precondition, guarded by contract only (no
     *        epoch, no reader count): docs/api.md, section 3.
     * @warning Out of contract the failure is SILENT: the mark misses a slot a
     *          chunk is about to adopt, frees it, take_free_slot() hands its
     *          identifier to another value, and that chunk reads the other
     *          value. The refusals above see the CALLING thread only: a
     *          std::thread / TBB pool or a team started by another host thread
     *          is not detected. Check builds fail an invariant on any lookup or
     *          registration overlapping the reclaim (GlobalVecHolder's
     *          LookupGuard / ReclaimScope); release builds check nothing.
     */
    static std::size_t reclaim_second_values()
    {
        if constexpr (std::is_same_v<T, bool>)
        {
            return 0; // one shared constant array, no registry (TabCstVal2Holder<bool>)
        }
        else
        {
            using gvh = GlobalVecHolder<T>;
            // Each refusal counted by cause (second_value_stats()): it returns
            // 0 as a call with nothing to free does.
            if (CHUNK_PMP_IN_PARALLEL_REGION)
            {
                gvh::refused_parallel.fetch_add(1, std::memory_order_relaxed);
                return 0;
            }
            if (g_in_rechunk_rebuild.load(std::memory_order_acquire))
            {
                gvh::refused_rechunk.fetch_add(1, std::memory_order_relaxed);
                return 0;
            }
            if (gvh::index_degraded.load(std::memory_order_relaxed))
            {
                gvh::refused_degraded.fetch_add(1, std::memory_order_relaxed);
                // Sticky until the last array of the type goes: say it once, the
                // run will otherwise end on the identifier exhaustion with no cause.
                static std::atomic<bool> warned{false};
                if (!warned.exchange(true, std::memory_order_relaxed))
                {
                    CHUNK_LOG(warn, compress,
                              "reclaim_second_values() refused: the second-value index is degraded, a value unfindable "
                              "by value could be freed. Nothing is reclaimed for this type until its last array is "
                              "destroyed.");
                }
                return 0;
            }
            // Check builds: no lookup or registration in flight from here to the
            // end of reclaim_unmarked() -- the mark included, since a slot found
            // before it and recorded after it would be freed (GlobalVecHolder's
            // "Reclaim contract").
            const typename gvh::ReclaimScope scope;
            gvh::reclaim_runs.fetch_add(1, std::memory_order_relaxed);
            const std::size_t handed_out = gvh::ids_handed_out();
            if (handed_out == 0)
            {
                return 0;
            }
            std::vector<std::uint8_t> referenced(handed_out + 1, 0);

            // Chunks compressed to a second value.
            Proxy2A<T>::share_info_array->for_each_key(
                [&referenced](const void* key)
                {
                    if (key == nullptr)
                    {
                        return;
                    }
                    // The key is the array's core, whatever its index type.
                    const auto* arr = static_cast<const ArrayChunkCore<Chunk2A<T>>*>(key);
                    const int   nc  = arr->chunk_count();
                    for (int j = 0; j < nc; ++j)
                    {
                        const T* vals = arr->chunk_ref(static_cast<std::size_t>(j)).values_load();
                        if (gvh::region_contains(vals))
                        {
                            mark_cst2_id(referenced, gvh::slot_id(vals));
                        }
                    }
                });

            // Every slot of every row: a released row is empty, a stale entry only keeps a
            // value one pass longer.
            using spec              = typename Proxy2A<T>::BaseSpec;
            const std::size_t nrows = spec::second_value_row_count();
            for (std::size_t r = 0; r < nrows; ++r)
            {
                const auto& row = spec::get_tab_second_value(static_cast<int>(r));
                for (const auto& slot : row)
                {
                    if constexpr (SecondValueTraits<T>::indexed)
                    {
                        mark_cst2_id(referenced, slot); // the identifier itself; 0 = none
                    }
                    else if (!(slot == T{}))
                    {
                        if (const T* hit = gvh::index_lookup(slot))
                        {
                            mark_cst2_id(referenced, gvh::slot_id(hit));
                        }
                    }
                }
            }
            const std::size_t freed = gvh::reclaim_unmarked(referenced);
            gvh::reclaim_freed.fetch_add(freed, std::memory_order_relaxed);
            return freed;
        }
    }

    /**
     * @brief Where this type's second-value identifiers stand: in use against
     *        the capacity, the reclaims run and refused, the degradation.
     *
     * @details Running out of identifiers stops the run; sampled at the
     *          application's own checkpoints, in_use() / capacity is the
     *          early warning, and refused() tells a reclaim that freed
     *          nothing because it could not run. Covers every ArrayChunk2A<T>,
     *          whatever its index type, like reclaim_second_values(). Relaxed
     *          reads, callable at any time, from any thread: exact at a
     *          sequential point. All zero for bool (no registry).
     */
    [[nodiscard]] static SecondValueStats second_value_stats() noexcept
    {
        if constexpr (std::is_same_v<T, bool>)
        {
            return {};
        }
        else
        {
            return GlobalVecHolder<T>::stats();
        }
    }

    /// @brief Compress every chunk currently queued as a candidate.
    static void make_block_compression();

  private:
    /// @brief reclaim_second_values(): identifier @p id is still referenced.
    static void mark_cst2_id(std::vector<std::uint8_t>& referenced, std::uint32_t id) noexcept
    {
        if (id != 0 && id < referenced.size())
        {
            referenced[id] = 1;
        }
    }

    static constexpr char is_val1 = 0; ///< Tag of tab_type_cst_val: chunk compressed to the first constant

    /**
     * @brief Claim this array's identity before its storage is built.
     * @details Called by create() through ArrayChunkBase: registers the address and takes
     *          the next free num_tab.
     * @post num_tab is valid and the array is discoverable by address.
     */
    void pre_create_hook()
    {
        auto& sh_inf = *Proxy2A<T>::share_info_array;

        sh_inf.insert(this->registry_key(), nb_tab);
        num_tab = nb_tab;
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
    void rebuild_storage(ind_type sz);

    /**
     * @brief Second value registered for the chunk holding element @p i.
     * @param i element index
     * @return this chunk's own second value (2A registers one per chunk)
     */
    auto get_metadata(ind_type i) const noexcept
    {
        return Proxy2A<T>::BaseSpec::get_tab_second_value(num_tab, this->chunk_index_of(i));
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
     * @details The registry is shared by every ArrayChunk2A<T> and reference-counted (the
     *          last release frees everything). Given back in clear_core() only if taken: an
     *          array built empty and grown by resize() needs one too, and an empty array's
     *          teardown must not drop one it never took.
     */
    void acquire_cst2_registry()
    {
        if (!holds_cst2_registry_)
        {
            Chunk2A<T>::tab_cst_val2_holder::create();
            holds_cst2_registry_ = true;
        }
    }

    /// @brief The identity step of swap(): num_tab and the registry entry (the
    ///        per-chunk metadata rows follow the chunks), the cst2 registry
    ///        reference (it goes with the chunks that need it).
    void swap_identity(ArrayChunk2A& other)
    {
        using std::swap;
        swap(this->num_tab, other.num_tab);
        swap(holds_cst2_registry_, other.holds_cst2_registry_);
        Proxy2A<T>::share_info_array->swap_keys(this->registry_key(), other.registry_key());
    }

    // ── cst2 mark-sweep GC: read-only mark + report ──────────────────────────

    /**
     * @brief Mark phase of the cst2 garbage collection: collect live buffers.
     *
     * @details Scans every live ArrayChunk2A<T> through the shared registry and inserts the
     *          cst2 buffer pointer of each chunk compressed to its second value. Read-only:
     *          safe at any quiescent point.
     *
     * @param[out] live set the referenced cst2 buffer pointers are inserted into
     * @see gc_mark_report for the reporting counterpart
     */
    static void gc_collect_live(std::unordered_set<const void*>& live)
    {
        Proxy2A<T>::share_info_array->for_each_key(
            [&](const void* key)
            {
                // The key is the array's core, whatever its index type.
                auto*     arr = static_cast<const ArrayChunkCore<Chunk2A<T>>*>(key);
                const int nc  = arr->chunk_count();
                for (int j = 0; j < nc; ++j)
                {
                    const auto& ch = arr->chunk_ref(static_cast<std::size_t>(j));
                    if (ch.is_cst_val2())
                    {
                        live.insert(static_cast<const void*>(ch.values_load()));
                    }
                }
            });
    }

    /// @brief This array holds one reference on Chunk2A<T>'s cst2 registry
    ///        (TabCstVal2Holder). See acquire_cst2_registry().
    bool holds_cst2_registry_ = false;
};

} // namespace shrink_array

// Template method bodies, after the class definition.

#include "chunk/internal/array_chunk2A_inline.h" // the template bodies

// Bodies of the two mode-specific mixins. After array_chunk2A_inline.h, so their
// CRTP self() sees a complete ArrayChunk2A.
#include "chunk/internal/array_chunk2A_parallel_inline.h"    // ParallelSupport2A's bodies
#include "chunk/internal/array_chunk_rechunk_hooks_inline.h" // RechunkHooks' bodies
