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

#include <cstddef>          // std::size_t
#include <cstdio>           // std::printf
#include <iostream>         // for cout, ostream
#include <utility>          // for move
#include <vector>           // for vector, swap
#include <unordered_set>    // for the cst2 GC live set
#include <initializer_list> // the {a, b, c} constructor
#include <iterator>         // std::forward_iterator, std::distance -- the (first, last) constructor

#include "chunk/internal/array_chunk2_base.h"           // for ArrayChunk2Base
#include "chunk/internal/array_chunk2A_parallel.h"      // ParallelSupport2A — OpenMP-only mixin
#include "chunk/internal/array_chunk2A_rechunk_hooks.h" // RechunkHooks2A — CHUNK_DYNAMIC-only mixin
#include "chunk/internal/array_chunk_base.h"            // for ArrayChunkBase
#include "chunk/internal/concept_array_chunk_element.h" // for ArrayChunkElement
#include "chunk/config.h"                               // for use_openmp
#include "chunk/internal/tab_cst_val2_holder.h"         // for GlobalVecHolder (cst2 GC)
#include "chunk/internal/write_compress_sync_policy.h"  // for WriteCompressSyncPolicy
#include "chunk/internal/array_chunk_core.h"            // the core the bases share
#include "chunk/internal/proxy2A.h"                     // Proxy2A<T> and its tables
#include "chunk/chunk2A_std.h"                          // Chunk2A<T>

namespace shrink_array
{

template <template <typename> class CHUNK_T, typename T>
class ManagerChunk;

/**
 * @brief Chunked array whose chunks may each compress to their OWN second value.
 *
 * @details Same chunked layout as ArrayChunk2, but with a different compression
 *          contract: ArrayChunk2 shares a SINGLE second value across the whole
 *          array, whereas here every chunk registers its own (the find-or-set
 *          path), so a chunk compresses as soon as all its elements are equal to
 *          any one constant. That pays off on data with several distinct
 *          constant runs, and costs an extra per-chunk second-value table --
 *          held by ProxySpec2A and garbage-collected through GlobalVecHolder
 *          (see gc_collect_live()).
 *
 *          Each array registers its address in Proxy2A::share_info_array to get
 *          a num_tab, the index into every per-tab metadata table. That
 *          registration is the array's identity: it is taken in
 *          pre_create_hook(), exchanged by rechunk_swap_identity(), and released
 *          in the destructor.
 *
 * @tparam T        stored element type
 * @tparam ind_type index type used for sizes and element positions
 */
template <ArrayChunkElement T, typename ind_type>
class ArrayChunk2A : public ArrayChunk2Base<ArrayChunk2A<T, ind_type>>,
                     public ParallelSupport2A<ArrayChunk2A<T, ind_type>>,
                     public RechunkHooks2A<ArrayChunk2A<T, ind_type>>
{
  public:
    using manager_chunk2A = ManagerChunk<Chunk2A, T>; ///< Same type as manager_type.

    using Base2   = ArrayChunk2Base<ArrayChunk2A>; ///< Chunk2/2A common base (dirty bitset, second-value plumbing).
    using Base    = ArrayChunkBase<ArrayChunk2A>;  ///< Base shared by every variant.
    using Hooks2A = RechunkHooks2A<ArrayChunk2A>;  ///< Re-chunk hooks (dynamic geometry).

    friend Base2;                                     ///< The CRTP base reaches this class's private members.
    friend ArrayChunkBase<ArrayChunk2A<T, ind_type>>; ///< For swap_identity().

    // Re-declared here (not only inherited): the mixins above declare them too,
    // so without a member of the same name in ArrayChunk2A an outside
    // `Arr::chunk_type` would be ambiguous between base subobjects. Name lookup
    // precedes access control, so hiding them in the mixins would not help.
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

    // Mode-specific members, moved out of this class into the mixins above.
    // The using-declarations are not decoration: ArrayChunkBase re-exports
    // RechunkSupport's hook DEFAULTS under these very names, so without them a
    // hook call through the CRTP self() would see two base subobjects and be
    // ambiguous. Naming them here makes them members of ArrayChunk2A, which
    // hides both bases' versions.
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
        if constexpr (build_config::use_openmp)
        {
            WriteCompressSyncPolicy<Proxy2A<T>>::ensure_initialized();
        }

        create(sz);
    }

    // ── The copy family: deep, element by element ────────────────────────────
    //
    // A copy is a copy, never an alias of the source's buffers: built on
    // ArrayChunkBase::copy_from (sized to the source, every element read
    // through the source's gate and written through this array's proxy);
    // the (first, last) and initializer-list constructors are the same loop
    // over an iterator. All sequential, like resize().
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
     * @details O(1) in the elements. With a copy constructor and a
     *          move-assignment declared, none was generated: `ArrayChunk2A a(std::move(b))`
     *          and every return that is not elided ran the copy constructor,
     *          element by element. Sequential context, as the constructors.
     * @note noexcept, so that std::vector moves arrays when it reallocates
     *       instead of copying them element by element. What can fail here
     *       -- the empty array's registration, the deferred-write drain --
     *       only fails on memory exhaustion or a corrupt registry, which the
     *       library already treats as fatal (alloc_chunk() is noexcept too):
     *       std::terminate.
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
     * @warning The erase() below is not optional. A later array reusing this
     *          address -- very common for stack arrays, e.g. rechunk_to()'s
     *          @c tmp -- would otherwise inherit the stale num_tab, because
     *          get_num_tab() returns the FIRST matching entry; the new array
     *          would then read another tab's metadata and corrupt it.
     */
    ~ArrayChunk2A()
    {
        clear();
        Proxy2A<T>::share_info_array.erase(this->registry_key());
    }

    /**
     * @brief Move-assign: this array's own contents are released (clear()),
     *        then it takes @p other's -- chunks, geometry and identity, by the
     *        same exchange as swap(); @p other is left empty and valid.
     * @details Not just a steal of @p other's chunk list: this array's old
     *          chunks are released first (their candidates purged), the moved
     *          chunks keep their metadata rows (num_tab) with them, and
     *          @p other is left with nb_chunk 0. Sequential context, as clear().
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
     */
    void resize(ind_type newsz);

    /**
     * @brief Resize to @p count elements, all set to @p val.
     * @param count new element count
     * @param val   value every element takes
     *
     * @note Fills [0, count), not only the new elements: Base2::resize(count,
     *       val) has std::vector::resize semantics ([old_size, count) only),
     *       so this goes through ArrayChunkBase::assign -- resize(count), then
     *       fill everything -- as chunk and chunk2 do. Covered by --test_stl.
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
     *       compressed differently still compare equal.
     */
    bool operator==(const ArrayChunk2A& other) const noexcept
    {
        if (m_size != other.m_size)
        {
            return false;
        }
        for (ind_type i = 0; i < m_size; ++i)
        {
            if (Base2::read_value(i) != other.read_value(i))
            {
                return false;
            }
        }
        return true;
    }

    /**
     * @brief Release all storage and report end-of-life statistics.
     * @details clear_core() plus the end-of-life I/O (compression stats and the
     *          size predictor).
     * @see clear_core
     */
    void clear();

    /// @brief Run a compression pass over this array's chunks.
    void compress_int();

    /// @brief Read-only access to the underlying chunk list.
    /// @return const reference to the chunks backing this array
    [[nodiscard]] const std::vector<Chunk2A<T>>& getChunkList() const noexcept
    {
        return chunk_list;
    }

    /**
     * @brief Report how many cst2 buffers a sweep could reclaim.
     * @details Runs gc_collect_live() and compares the live count with
     *          GlobalVecHolder's total; the difference is the reclaimable set.
     *          Diagnostic only -- prints to stderr and frees nothing.
     */
    static void gc_mark_report()
    {
        std::unordered_set<const void*> live;
        gc_collect_live(live);
        const std::size_t held = GlobalVecHolder<T>::count_entries();
        std::fprintf(stderr, "[cst2-gc] live=%zu / holder=%zu  (reclaimable=%zu)\n", live.size(), held,
                     held > live.size() ? held - live.size() : 0);
        std::fflush(stderr);
    }

    /// @brief Compress every chunk currently queued as a candidate.
    static void make_block_compression();

  private:
    static constexpr char is_val1 = 0; ///< Tag of tab_type_cst_val: chunk compressed to the first constant

    /**
     * @brief Claim this array's identity before its storage is built.
     * @details Called by create() through ArrayChunkBase. Registers the array's
     *          address in the shared registry and takes the next free num_tab,
     *          which every per-tab metadata table is then indexed by.
     * @post num_tab is valid and the array is discoverable by address.
     */
    void pre_create_hook()
    {
        auto& sh_inf = Proxy2A<T>::share_info_array;

        sh_inf.insert(this->registry_key(), nb_tab);
        num_tab = nb_tab;
        ++nb_tab;
    }

    /**
     * @brief Register the array and allocate its storage.
     * @param sz element count
     */
    void create(ind_type sz);

    /**
     * @brief (Re)allocate storage at the current active geometry.
     *
     * @details Does the allocation work of create() WITHOUT its identity and
     *          registration side effects. Used by create() right after
     *          create_common(), and by rechunk_to() to rebuild an array that is
     *          already registered and has just been torn down, at a new geometry
     *          but under the same num_tab.
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
    /// Everything clear() does EXCEPT the end-of-life I/O (compression stats and
    /// the size predictor): drain, terminal compression, per-chunk buffer
    /// release, and metadata reset. Factored out so rechunk_to() can release
    /// this array's storage through the SAME proven path (ledger-correct) rather
    /// than reproducing it by hand. In non-OMP builds this does NOT tear down the
    /// shared cst_val2 holder (that lives only on the OMP path), so it is safe to
    /// call on one array without affecting others.
    void clear_core();

    /**
     * @brief Print this array's compression ratio and verdict.
     * @param os destination stream
     */
    void print_compression_stats(std::ostream& os = std::cout) const;

    /**
     * @brief Take this array's reference on the cst2 registry, once, as soon
     *        as it has chunks.
     * @details The registry is shared by every ArrayChunk2A<T> and counted:
     *          create() registers a user, clear() drops one and frees
     *          everything with the last. One reference per array, taken here
     *          as soon as it has chunks and given back in clear_core() only if
     *          taken: an array built empty and grown by resize() needs one too
     *          (its first compression pass looks the registry up), and an empty
     *          array's teardown must not drop one it never took.
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
        Proxy2A<T>::share_info_array.swap_keys(this->registry_key(), other.registry_key());
    }

    // ── cst2 mark-sweep GC — STEP B: read-only mark + report ─────────────────

    /**
     * @brief Mark phase of the cst2 garbage collection: collect live buffers.
     *
     * @details Scans every live ArrayChunk2A<T> through the shared registry and
     *          inserts the cst2 buffer pointer of each chunk that is currently
     *          compressed to its second value. Read-only: it touches no chunk
     *          state, so it is safe at any quiescent point (the advisor tick,
     *          for instance). num_tab and geometry are per-array, so walking
     *          chunk_list without the global size is correct here.
     *
     * @param[out] live set the referenced cst2 buffer pointers are inserted into
     * @see gc_mark_report for the reporting counterpart
     */
    static void gc_collect_live(std::unordered_set<const void*>& live)
    {
        Proxy2A<T>::share_info_array.for_each_key(
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

// Template method bodies. Included here (after the class definition) so that
// every translation unit using ArrayChunk2A<T> sees the definitions and can
// auto-instantiate the exact T it needs.

#include "chunk/internal/array_chunk2A_inline.h" // the template bodies

// Bodies of the two mode-specific mixins. After array_chunk2A_inline.h, so their
// CRTP self() sees a complete ArrayChunk2A.
#include "chunk/internal/array_chunk2A_parallel_inline.h"      // ParallelSupport2A's bodies
#include "chunk/internal/array_chunk2A_rechunk_hooks_inline.h" // RechunkHooks2A's bodies
