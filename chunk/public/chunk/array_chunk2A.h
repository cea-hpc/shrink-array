//
// Created by ballereaup on 23/05/24.
//
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

#include <iostream>                             // for cout, ostream
#include <utility>                              // for move
#include <vector>                               // for vector, swap
#include <unordered_set>                        // for the cst2 GC live set

#include <initializer_list> // the {a, b, c} constructor
#include <iterator>         // std::forward_iterator, std::distance -- the (first, last) constructor
#include "chunk/internal/array_chunk2_base.h"            // for ArrayChunk2Base
#include "chunk/internal/array_chunk2A_parallel.h"       // ParallelSupport2A — OpenMP-only mixin
#include "chunk/internal/array_chunk2A_rechunk_hooks.h"  // RechunkHooks2A — CHUNK_DYNAMIC-only mixin
#include "chunk/internal/array_chunk_base.h"             // for ArrayChunkBase
#include "chunk/internal/concept_array_chunk_element.h"  // for ArrayChunkElement
#include "chunk/config.h"                       // for use_openmp
#include "chunk/internal/tab_cst_val2_holder.h"           // for GlobalVecHolder (cst2 GC)
#include "chunk/internal/proxy2A_meta_bases.h"            // Proxy2AMetaBases (armed metadata rows)
#include "chunk/internal/write_compress_sync_policy.h"   // for WriteCompressSyncPolicy

namespace shrink_array
{

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
    using manager_chunk2A = ManagerChunk<Chunk2A, T>;

    using Base2      = ArrayChunk2Base<ArrayChunk2A>;
    using Base       = ArrayChunkBase<ArrayChunk2A>;
    using Parallel2A = ParallelSupport2A<ArrayChunk2A>;
    using Hooks2A    = RechunkHooks2A<ArrayChunk2A>;

    // Re-declared here (not only inherited): the mixins above declare them too,
    // so without a member of the same name in ArrayChunk2A an outside
    // `Arr::chunk_type` would be ambiguous between base subobjects. Name lookup
    // precedes access control, so hiding them in the mixins would not help.
    using Traits       = Base2::Traits;
    using chunk_type   = Base2::chunk_type;
    using manager_type = Base2::manager_type;
    using proxy_type   = Base2::proxy_type;
    using value_type   = T;

    using Base2::chunk_list;
    using Base2::m_size;
    using Base2::nb_chunk;
    using Base2::resize;

    using Base::nb_tab;
    using Base::size_policy_;
    using Base::num_tab;

    friend Base2;

    /// @brief This array's rows in the per-tab metadata tables, armed into
    ///        every write proxy by operator[] (see Proxy2AMetaBases).
    [[nodiscard]] const Proxy2AMetaBases<T>& meta_bases() const noexcept
    {
        return meta_bases_;
    }

    /// @brief Re-read the rows from the tables. Call after ANY reallocation of
    ///        this array's rows (resize_tab_is_int) or change of num_tab -- a
    ///        missed site is caught by the any_check comparison in
    ///        Proxy2A::meta_from_armed().
    void sync_meta_bases() noexcept
    {
        Proxy2AMetaBases<T> b;
        if (num_tab >= 0 && nb_chunk > 0)
        {
            b.type   = Proxy2A<T>::tab_type_cst_val[num_tab].data();
            b.inside = Proxy2A<T>::get_tab_is_inside_list_to_compress(num_tab).get();
            if constexpr (!std::is_same_v<T, bool>)
                b.sv = Proxy2A<T>::BaseSpec::get_tab_second_value(num_tab).get();
        }
        meta_bases_ = b;
    }

  private:
    Proxy2AMetaBases<T> meta_bases_{}; ///< see meta_bases()

  public:

    // Mode-specific members, moved out of this class into the mixins above.
    // The using-declarations are not decoration: ArrayChunkBase re-exports
    // RechunkSupport's hook DEFAULTS under these very names, so without them a
    // hook call through the CRTP self() would see two base subobjects and be
    // ambiguous. Naming them here makes them members of ArrayChunk2A, which
    // hides both bases' versions.
    using Parallel2A::clear_in_par;

    using Hooks2A::rechunk_drain_reuse_list;
    using Hooks2A::rechunk_drop_candidates;
    using Hooks2A::rechunk_swap_identity;

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
    // Copy-assignment used to alias: chunk_list copied, so two arrays over the
    // same buffers with no contract for it. A copy is a copy now, built on
    // ArrayChunkBase::copy_from (sized to the source, every element read
    // through the source's gate and written through this array's proxy);
    // the (first, last) and initializer-list constructors are the same loop
    // over an iterator. All sequential, like resize().
    // chunk2a finds each chunk's second value by itself: nothing to carry over.
    ArrayChunk2A(const ArrayChunk2A& from) : ArrayChunk2A(from.size())
    {
        this->copy_from(from);
    }

    template <std::forward_iterator It>
    ArrayChunk2A(It first, It last) : ArrayChunk2A(static_cast<ind_type>(std::distance(first, last)))
    {
        this->assign(first, last);
    }

    ArrayChunk2A(std::initializer_list<T> il) : ArrayChunk2A(il.begin(), il.end()) {}

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

        sh_inf.insert(this, nb_tab);
        num_tab = nb_tab;
        ++nb_tab;
    }

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
        Proxy2A<T>::share_info_array.erase(this);
    }

    /**
     * @brief Register the array and allocate its storage.
     * @param sz element count
     */
    void create(ind_type sz);

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
                auto* arr = static_cast<const ArrayChunk2A<T>*>(key);
                const auto& cl = arr->chunk_list;
                const int   nc = arr->nb_chunk;
                for (int j = 0; j < nc; ++j)
                    if (cl[static_cast<std::size_t>(j)].is_cst_val2())
                        live.insert(static_cast<const void*>(
                            cl[static_cast<std::size_t>(j)].values_load()));
            });
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
        std::fprintf(stderr,
                     "[cst2-gc] live=%zu / holder=%zu  (reclaimable=%zu)\n",
                     live.size(), held, held > live.size() ? held - live.size() : 0);
        std::fflush(stderr);
    }

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
     * @brief Move-assign the storage of @p other into this array.
     * @param other array whose chunk storage is taken over
     * @return reference to this array
     * @note num_tab is NOT moved: identity belongs to the address, and the
     *       registry still maps this array's own address to its own tab.
     */
    ArrayChunk2A& operator=(ArrayChunk2A<T, ind_type>&& other) noexcept
    {
        // Drain ops on THIS array's chunks before the move frees this->chunk_list
        // (inherited from ArrayChunk2Base). See its comment for the rationale.
        this->flush_pending_deferred_ops();

        chunk_list = (std::move(other.chunk_list));

        nb_chunk = other.nb_chunk;
        m_size   = other.m_size;

        return *this;
    }

    /**
     * @brief Second value registered for the chunk holding element @p i.
     * @param i element index
     * @return this chunk's own second value (2A registers one per chunk)
     */
    auto get_metadata(ind_type i) const noexcept
    {
        return Proxy2A<T>::BaseSpec::get_tab_second_value(num_tab, i >> this->size_policy_.decalage());
    }

    /**
     * @brief Has the chunk holding element @p i been materialised?
     * @param i element index
     * @return false while that chunk is still compressed to a constant
     */
    [[nodiscard]] bool written(ind_type i) const
    {
        return chunk_list[i >> this->size_policy_.decalage()].written();
    }

    /// @brief Compress every chunk currently queued as a candidate.
    static void make_block_compression();

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
     * @note FIX: deleguait a Base2::resize(count, val), qui a la semantique de
     *       std::vector::resize -- il ne remplit que les elements NOUVEAUX,
     *       [old_size, count). assign doit remplir [0, count). Un
     *       `ArrayChunk2A<double> a(5); a.assign(1000, 2.5);` laissait donc
     *       a[0..4] a zero. Jamais vu parce que le seul test de cette surface,
     *       --test_stl, etait rejete par le filtre d'arguments du bench.
     *       ArrayChunkBase::assign fait resize(count) puis remplit tout ; chunk
     *       et chunk2 l'utilisent deja et passent le test.
     */
    void assign(ind_type count, in_t<T> val)
    {
        Base::assign(count, val);
    }
    using Base::assign; // the (first, last) overload, hidden by the one above otherwise

    /**
     * @brief Exchange the storage of two arrays.
     *
     * @param other array to exchange storage with
     * @note num_tab is deliberately NOT swapped (the disabled line below records
     *       the attempt). The registry maps an ADDRESS to a num_tab, and neither
     *       array changes address here, so swapping the tab index would leave
     *       both arrays disagreeing with the registry about their own identity.
     *       rechunk_to() needs the opposite behaviour and gets it explicitly
     *       from rechunk_swap_identity(), which updates the registry too.
     */
    void swap(ArrayChunk2A& other) noexcept
    {
        std::swap(chunk_list, other.chunk_list);
        std::swap(m_size, other.m_size);
        std::swap(nb_chunk, other.nb_chunk);
    }

    /**
     * @brief Element-wise equality.
     * @param other array to compare against
     * @return true if both arrays have the same size and the same values
     * @note Compares decompressed values, so two arrays that are equal but
     *       compressed differently still compare equal.
     */
    bool operator==(const ArrayChunk2A& other) const
    {
        if (m_size != other.m_size)
            return false;
        for (ind_type i = 0; i < m_size; ++i)
            if (Base2::read_value(i) != other.read_value(i))
                return false;
        return true;
    }

    /**
     * @brief Release all storage and report end-of-life statistics.
     * @details clear_core() plus the end-of-life I/O (compression stats and the
     *          size predictor).
     * @see clear_core
     */
    void clear();

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

    /// @brief Run a compression pass over this array's chunks.
    void compress_int();

    /// @brief Deep copy-assignment (see the copy constructor). Self-assignment
    ///        is a no-op; num_tab is not copied, identity stays with the address.
    ArrayChunk2A& operator=(const ArrayChunk2A& from)
    {
        if (this != &from)
            this->copy_from(from);
        return *this;
    }

    static constexpr char is_val1 = 0; ///< Tag: chunk compressed to the first constant
    static constexpr char is_val2 = 1; ///< Tag: chunk compressed to its own second value

    /// @brief Read-only access to the underlying chunk list.
    /// @return const reference to the chunks backing this array
    [[nodiscard]] const std::vector<Chunk2A<T>>& getChunkList() const
    {
        return chunk_list;
    }
};

} // namespace shrink_array

// Template method bodies. Included here (after the class definition) so that
// every translation unit using ArrayChunk2A<T> sees the definitions and can
// auto-instantiate the exact T it needs.

#include "chunk/internal/array_chunk2A_inline.h"

// Bodies of the two mode-specific mixins. After array_chunk2A_inline.h, so their
// CRTP self() sees a complete ArrayChunk2A (the serial clear_in_par calls clear()).
#include "chunk/internal/array_chunk2A_parallel_inline.h"
#include "chunk/internal/array_chunk2A_rechunk_hooks_inline.h"
