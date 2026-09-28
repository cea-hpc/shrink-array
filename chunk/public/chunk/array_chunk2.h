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

#include <cstdlib>     // for abort
#include <iostream>    // for cout, ostream
#include <tuple>       // for tuple
#include <type_traits> // for is_same_v
#include <utility>     // for get, move
#include <vector>      // for vector

#include <initializer_list>                   // the {a, b, c} constructor
#include <iterator>                           // std::forward_iterator, std::distance -- the (first, last) constructor
#include "chunk/bit.h"                        // is_bool_like_v: bool and Bit share the bool path
#include "chunk/internal/in_t.h"              // in_t<T>: the element snapshots
#include "chunk/internal/array_chunk2_base.h" // for ArrayChunk2Base
#include "chunk/internal/array_chunk2_parallel.h"       // ParallelSupport2 — OpenMP-only mixin
#include "chunk/internal/array_chunk2_rechunk_hooks.h"  // RechunkHooks2 — CHUNK_DYNAMIC-only mixin
#include "chunk/internal/array_chunk_base.h"            // for ArrayChunkBase
#include "chunk/internal/concept_array_chunk_element.h" // for ArrayChunkElement
#include "chunk/config.h"                               // for use_fast_block_compression, use_openmp
#include "chunk/internal/write_compress_sync_policy.h"  // for WriteCompressSyncPolicy
#include "chunk/PMP_CHUNK.h"                            // CHUNK_PMP_THREAD_CAPACITY
#include "chunk/chunk2_std.h"                           // Chunk2<T>
#include "chunk/internal/proxy2.h"                      // Proxy2<T> and its registry

namespace shrink_array
{

template <template <typename> typename CHUNK_T, typename T>
class ManagerChunk;

/**
 * @brief Chunked array compressing on TWO constants: first_value and one second value.
 *
 * @details A chunk collapses to a constant as soon as all its elements equal
 *          either first_value or the array's second value. Unlike ArrayChunk2A,
 *          that second value is SHARED by the whole array rather than chosen per
 *          chunk: cheaper metadata (one entry per array instead of one per
 *          chunk), but it only pays off when the data has a single dominant
 *          non-zero constant.
 *
 *          The second value is not stored in the array. It lives in the shared
 *          registry Proxy2<T>::share_info_array, keyed by the array's ADDRESS,
 *          alongside the num_tab indexing the per-tab metadata. That indirection
 *          is why the destructor must erase both entries, and why rechunk_to()
 *          needs the six hooks below rather than 2A's three.
 *
 * @tparam T        stored element type
 * @tparam ind_type index type used for sizes and element positions
 */

template <ArrayChunkElement T, typename ind_type>
class ArrayChunk2 : public ArrayChunk2Base<ArrayChunk2<T, ind_type>>,
                    public ParallelSupport2<ArrayChunk2<T, ind_type>>,
                    public FastBlockSupport2<ArrayChunk2<T, ind_type>>,
                    public RechunkHooks2<ArrayChunk2<T, ind_type>>
{
  public:
    using Base2      = ArrayChunk2Base<ArrayChunk2>;
    using Base       = ArrayChunkBase<ArrayChunk2>;
    using Parallel2  = ParallelSupport2<ArrayChunk2>;
    using FastBlock2 = FastBlockSupport2<ArrayChunk2>;
    using Hooks2     = RechunkHooks2<ArrayChunk2>;

    // Traits and chunk_type are re-declared here (not only inherited): the
    // mixins above declare them too, so without a member of the same name in
    // ArrayChunk2 an outside `Arr::chunk_type` would be ambiguous between base
    // subobjects. ArrayChunk does the same for the same reason.
    using Traits       = Base2::Traits;
    using chunk_type   = Base2::chunk_type;
    using manager_type = Base2::manager_type;
    using proxy_type   = Base2::proxy_type;
    using value_type   = T;

    using Base2::chunk_list;

    using Base::nb_tab;
    using Base::size_policy_;
    using Base2::m_size;
    using Base2::nb_chunk;
    using Base2::resize; // Import resize(sz, val)

    using inf_array_t = std::tuple<T, int, int>;

    using manager_chunk2 = ManagerChunk<Chunk2, T>;

    friend Base2;

    // RechunkHooks2::rechunk_reown_after_swap copies ind_cst_val2, which is
    // private below. Same reason ArrayChunkBase befriends RechunkSupport.
    template <typename, bool>
    friend class RechunkHooks2;

    // Mode-specific members, moved out of this class into the mixins above.
    // The using-declarations are not decoration: ArrayChunkBase re-exports
    // RechunkSupport's hook DEFAULTS under these very names, so without them a
    // hook call through the CRTP self() would see two base subobjects and be
    // ambiguous. Naming them here makes them members of ArrayChunk2, which hides
    // both bases' versions.
    using FastBlock2::make_block_compression;

    using Hooks2::rechunk_capture_state;
    using Hooks2::rechunk_drain_reuse_list;
    using Hooks2::rechunk_drop_candidates;
    using Hooks2::rechunk_reown_after_swap;
    using Hooks2::rechunk_seed_fresh;
    using Hooks2::rechunk_swap_identity;

    static constexpr bool is_bool     = is_bool_like_v<T>; // bool and Bit (bit.h)
    static constexpr bool is_not_bool = !is_bool;

    /**
     * @brief Build an array of @p sz elements.
     * @param sz initial element count; 0 by default, as std::vector (a fresh
     *           array is empty, so a push_back on it yields one element)
     */
    explicit ArrayChunk2(ind_type sz = 0)
    {
        if constexpr (build_config::use_openmp)
        {
            WriteCompressSyncPolicy<Proxy2<T>>::ensure_initialized();
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
    // chunk2's second value is array-wide: the source's is re-declared here
    // before the elements are written, so that its runs stay what they are.
    ArrayChunk2(const ArrayChunk2& from) : ArrayChunk2(from.size())
    {
        copy_second_value_from(from);
        this->copy_from(from);
    }

    /**
     * @brief Move-construct: an empty array at the active geometry takes
     *        @p other's chunks, geometry and identity -- the exchange of
     *        swap() -- and @p other is left empty and valid.
     * @details O(1) in the elements. With a copy constructor and a
     *          move-assignment declared, none was generated: `ArrayChunk2 a(std::move(b))`
     *          and every return that is not elided ran the copy constructor,
     *          element by element. Sequential context, as the constructors.
     * @note noexcept, so that std::vector moves arrays when it reallocates
     *       instead of copying them element by element. What can fail here
     *       -- the empty array's registration, the deferred-write drain --
     *       only fails on memory exhaustion or a corrupt registry, which the
     *       library already treats as fatal (alloc_chunk() is noexcept too):
     *       std::terminate.
     */
    ArrayChunk2(ArrayChunk2&& other) noexcept : ArrayChunk2(0)
    {
        this->swap_storage_and_identity(other);
    }

    template <std::forward_iterator It>
    ArrayChunk2(It first, It last) : ArrayChunk2(static_cast<ind_type>(std::distance(first, last)))
    {
        this->assign(first, last);
    }

    ArrayChunk2(std::initializer_list<T> il) : ArrayChunk2(il.begin(), il.end()) {}

    /**
     * @brief Release the array's storage, then drop both registry entries.
     *
     * @details Order matters: clear() first, then the two registry erases --
     *          clear() still needs get_num_tab(registry_key()) and its own terminal sweep
     *          still needs a valid inf_array pointer.
     *
     * @warning Neither erase is optional, and each closes a distinct
     *          use-after-free:
     *          - share_info_array.erase(registry_key()) drops the (address -> tuple)
     *            entry. Without it, a later array reusing this address inherits
     *            our stale num_tab and second value through PointerMapChunk2's
     *            first-match lookup. rechunk_to()'s short-lived @c tmp makes
     *            that reuse routine.
     *          - erase_inf_array_by_ptr(&inf_array) drops any map_inf_array
     *            entry still pointing at our inf_array MEMBER.
     *            compress_chunk_to_compress_potentially() iterates
     *            get_all_inf_array() and dereferences those pointers, so leaving
     *            ours behind lets the NEXT array's terminal sweep read a
     *            destroyed object -- ASan reports use-after-return, exposed
     *            deterministically by @c tmp. Fast-block and non-bool only, the
     *            modes where map_inf_array is populated; it removes only entries
     *            we still own, since rechunk_to() re-points its value onto the
     *            survivor first.
     */
    ~ArrayChunk2()
    {
        clear();
        Proxy2<T>::share_info_array.erase(this->registry_key());

        if constexpr (chunk_config::use_fast_block_compression && is_not_bool)
        {
            Proxy2<T>::share_info_array.erase_inf_array_by_ptr(&inf_array);
        }
    }

    /**
     * @brief Register the array and allocate its storage.
     * @param sz element count
     */
    void create(ind_type sz);

    /**
     * @brief (Re)allocate storage at the current active geometry.
     * @details Does create()'s allocation work WITHOUT its registration side
     *          effects (insert_only_num_tab, the nb_tab bump,
     *          resize_tab_is_int(nb_tab)). It resolves num_tab from the
     *          registry, so it is correct on an already-registered array
     *          switching geometry -- which is what rechunk_to() needs.
     * @param sz element count to allocate for
     */
    void rebuild_storage(ind_type sz);

    /**
     * @brief Set the array-wide second constant.
     * @details Registers @p sec_value in the shared cst_val2 registry, reusing
     *          the index of an identical value already registered by another
     *          array. A reused block is re-filled to THIS array's geometry
     *          first: it was only filled up to the registering array's size, so
     *          a chunk of a larger array compressed to it would read stale zeros
     *          past that point.
     * @param sec_value value chunks may now compress to
     * @param message   print the "useless for bool" warning; ignored for every
     *                  other T, and it gates nothing else
     * @warning Registered cst_val2 blocks are never released, even when replaced
     *          by a larger one: a still-live chunk of another array may hold the
     *          old address as its `values`.
     */
    void set_second_value(T sec_value, bool message = true);
    /**
     * @brief The array-wide second constant, read from the shared registry.
     * @return the registered second value; always true for bool
     */
    [[nodiscard]] T get_second_value() const noexcept
    {
        if constexpr (!is_bool_like_v<T>)
        {
            auto* key = const_cast<void*>(this->registry_key());
            return Proxy2<T>::share_info_array.get_second_value(key);
        }
        else
        {
            return true;
        }
    }
    /// @brief CRTP hook: the metadata the base needs for any element.
    /// @return this array's shared second-value tuple (element index unused,
    ///         since the value is array-wide -- unlike ArrayChunk2A)
    auto get_metadata(ind_type /*ind*/) const noexcept
    {
        return this->get_info_second_value();
    }

    /**
     * @brief Move-assign: this array's own contents are released (clear()),
     *        then it takes @p other's -- chunks, geometry and identity, by the
     *        same exchange as swap(); @p other is left empty and valid.
     * @details It used to steal @p other's chunk list and nothing else: this
     *          array's old chunks were destroyed with their buffers still
     *          queued as compression candidates, the moved chunks kept
     *          @p other's metadata rows (num_tab), and @p other kept its
     *          nb_chunk over an empty list. Sequential context, as clear().
     * @note noexcept, as the move constructor: a failure here terminates.
     */
    ArrayChunk2& operator=(ArrayChunk2&& other) noexcept
    {
        if (this != &other)
        {
            clear();
            this->swap_storage_and_identity(other);
        }
        return *this;
    }

    /// @brief Full registry tuple for this array: second value, its index, num_tab.
    [[nodiscard]] inf_array_t get_info_second_value() const noexcept
    {
        auto* key = const_cast<void*>(this->registry_key());
        return Proxy2<T>::share_info_array.get_info_second_value(key);
    }

    /// @brief Index of this array's second value in the shared value table.
    /// @warning Aborts for bool, which registers no second value.
    [[nodiscard]] int get_ind_second_value() const noexcept
    {
        if constexpr (is_not_bool)
        {
            return std::get<1>(get_info_second_value());
        }
        else
        {
            std::abort();
        }
    }

    /**
     * @brief Has the chunk holding element @p i been materialised?
     * @param i element index
     * @return false while that chunk is still compressed to a constant
     */
    [[nodiscard]] bool written(ind_type i) const
    {
        return chunk_written(this->chunk_index_of(i));
    }

    /**
     * @brief Is chunk @p j materialised?
     * @param j CHUNK index (not an element index -- see written(ind_type))
     * @return false while that chunk is compressed to either constant
     * @note Chunk2::written() needs the array's second-value index for
     *       non-bool T, which is why ArrayChunkBase::is_written_at() routes
     *       here rather than calling chunk_list[j].written() itself.
     */
    [[nodiscard]] bool chunk_written(int j) const
    {
        if constexpr (is_not_bool)
        {
            return chunk_list[j].written(get_ind_second_value());
        }
        else
        {
            return chunk_list[j].written();
        }
    }

    /**
     * @brief Resize the array, preserving the elements that still fit.
     * @details Drains the pending compression candidates and deferred ops first
     *          — both hold `&chunk_list[j]` pointers that a reallocation would
     *          dangle — then grows the chunk list and initialises the new
     *          chunks.
     * @param newsz new element count
     * @note Allocation is GROWTH-only: shrinking updates the sizes but releases
     *       no chunk, so the storage never gets smaller.
     */
    void resize(ind_type newsz);

    /**
     * @brief Release all storage and report end-of-life statistics.
     * @details clear_core() plus the pre-drain compression stats.
     * @see clear_core
     */
    void clear();

    /**
     * @brief Storage teardown shared by clear() and rechunk_to().
     * @details Everything clear() does except the pre-drain compression stats,
     *          so rechunk_to() can release this array's storage through the same
     *          proven path instead of reproducing it.
     * @warning Parks buffers by THIS array's geometry (size_policy_.size()), not
     *          the ambient thread size, which may belong to another array
     *          mid-rechunk. It also drops this array's compression candidates by
     *          num_tab, so no Chunk2* of a dead array survives in the shared
     *          candidate store.
     */
    void clear_core();

    /**
     * @brief Print this array's compression ratio and verdict.
     * @details Classifies every chunk by RE-READING its content rather than
     *          trusting `written()`, which only says whether the opportunistic
     *          compression happened to catch the chunk during the run. That
     *          re-scan is what makes the reported count reproducible across
     *          thread counts. Also runs a cross-variant probe counting the
     *          distinct second values present, i.e. what Chunk2A would have
     *          captured that a single-second-value Chunk2 could not.
     * @param os destination stream
     * @note Despite being const and named "print", it also FEEDS the global
     *       end-of-run counters when `chunk_config::stats` is on.
     */
    void print_compression_stats(std::ostream& os = std::cout) const;

    /**
     * @brief Audit pass: report the constant chunks the incremental compression missed.
     * @details Drains the pending candidate list first, so that a chunk which is
     *          constant only because its compression is still QUEUED is not
     *          counted as a miss; then scans for chunks that are materialised
     *          yet fully first_value or second_value. Those are compressed on
     *          the spot (a self-heal), and only the ones that survive that are
     *          reported on stderr as genuine misses.
     * @note Compiled to an immediate return under `PERF`: this is a
     *       check/debug-build diagnostic, not part of the compression engine.
     *       The engine proper is driven from the write path and from the
     *       phase transitions of manage_ompt.h.
     */
    void compress_int();

    /**
     * @brief Public entry point for the audit pass.
     * @see compress_int, which this forwards to — including its `PERF` no-op.
     */
    void compress();

    /// @brief Deep copy-assignment (see the copy constructor). Self-assignment
    ///        is a no-op.
    ArrayChunk2& operator=(const ArrayChunk2& from)
    {
        if (this != &from)
        {
            copy_second_value_from(from);
            this->copy_from(from);
        }
        return *this;
    }

  private:
    /// @brief Declare @p from's second value on this array, when it has one
    ///        (T{} means none: rechunk_seed_fresh's rule) and this array has
    ///        not declared it yet.
    void copy_second_value_from(const ArrayChunk2& from)
    {
        if constexpr (!is_bool_like_v<T>)
        {
            const in_t<T> sv = from.get_second_value();
            if (sv != T{} && get_second_value() != sv)
                set_second_value(sv, false);
        }
    }

  public:
    /// This array's (second value, index, num_tab) tuple: a MIRROR of its entry
    /// in Proxy2<T>::share_info_array, complete in every build mode.
    /// @details The registry stays the source of truth. The mirror is never
    ///          computed, only re-read from it (sync_registry_mirror()), right
    ///          after each of the three places that write the registry:
    ///          create(), set_second_value() and the rechunk identity swap.
    ///          operator[] arms every write proxy with its ADDRESS
    ///          (registry_mirror()), which is what takes the registry scan off
    ///          the write path.
    /// @warning map_inf_array holds a POINTER to this member, so the destructor
    ///          must erase that entry -- see its documentation.
    inf_array_t inf_array;

    /// @brief Address of the registry mirror, for ArrayChunkBase to arm write
    ///        proxies with (ProxySpec2::arm_inf_array).
    [[nodiscard]] const inf_array_t* registry_mirror() const noexcept
    {
        return &inf_array;
    }

    /// @brief Re-read the mirror from the registry. Call after ANY write to
    ///        this array's registry entry -- a missed site is caught by the
    ///        any_check comparison in ProxySpec2::load_inf_array().
    void sync_registry_mirror() noexcept
    {
        inf_array = this->get_info_second_value();
    }

  private:
    /**
     * @brief The (T{}, no_second_value_index) tuple every array without a
     *        second value is drained under.
     * @details The compression pass walks map_inf_array and drains the
     *          candidate list keyed by each tuple's value. An array that has
     *          declared no second value queues its chunks under T{}, so that
     *          key needs a tuple to walk and a list per rank -- both created
     *          by register_no_second_value_key() from create(). One static
     *          tuple per element type, never erased: the destructor only
     *          drops entries pointing at the dying array's own member.
     */
    static inline inf_array_t inf_array_none{T{}, chunk_config::no_second_value_index, 0};

    /// @brief Make the T{} key drainable: its tuple in map_inf_array and a
    ///        candidate list per rank, if this type has none yet. Idempotent;
    ///        fast-block builds only (the serial path has no lists).
    void register_no_second_value_key()
    {
        if constexpr (is_not_bool && chunk_config::use_fast_block_compression)
        {
            auto& sh_inf = Proxy2<T>::share_info_array;
            if (sh_inf.find_inf_array(T{}) == nullptr)
            {
                sh_inf.insert(T{}, &inf_array_none);
                // The between-phase pass asks have_chunk_to_compress() per
                // value of THIS map before draining anything: the T{} key
                // must be in it, at the sentinel index (find_ind_tab_cst2
                // then reports it as unregistered, which is what
                // set_second_value(T{}) must see to build a real block).
                sh_inf.insert_map_ind_cst_val2(T{}, chunk_config::no_second_value_index);
            }
            if constexpr (build_config::use_openmp)
            {
                for (int thr = 0; thr < CHUNK_PMP_THREAD_CAPACITY; ++thr)
                {
                    auto& slot = manager_chunk2::map_chunk2_to_compress_potentially_thr[thr][T{}];
                    if (!slot)
                        slot = std::make_shared<typename std::remove_reference_t<decltype(slot)>::element_type>();
                }
            }
        }
    }

  public:
  private:
    /// Index of the second value in the shared value table;
    /// chunk_config::no_second_value_index until set_second_value().
    int ind_cst_val2 = chunk_config::no_second_value_index;

    friend ArrayChunkBase<ArrayChunk2<T, ind_type>>; // swap_identity()

    /**
     * @brief The identity step of swap(): the registry tuple (second value,
     *        its index, num_tab), which is what Chunk2 reads its identity
     *        from, then the members that mirror it.
     * @details The num_tab member is not swapped, as in rechunk_swap_identity:
     *          Chunk2 never reads it. map_inf_array points each second value at
     *          the inf_array MEMBER of an array that has it; after the exchange
     *          that member holds the other array's tuple, so both values are
     *          re-pointed (a T{} value is the static inf_array_none's).
     */
    void swap_identity(ArrayChunk2& other)
    {
        auto& sh_inf = Proxy2<T>::share_info_array;
        sh_inf.swap_entries(this->registry_key(), other.registry_key());
        std::swap(ind_cst_val2, other.ind_cst_val2);
        sync_registry_mirror();
        other.sync_registry_mirror();
        if constexpr (chunk_config::use_fast_block_compression && is_not_bool)
        {
            for (ArrayChunk2* arr : {this, &other})
            {
                const T value = std::get<0>(arr->inf_array);
                if (value != T{})
                    sh_inf.insert(value, &arr->inf_array);
            }
        }
    }

  public:
    /// @brief Read-only access to the underlying chunk list.
    [[nodiscard]] const std::vector<Chunk2<T>>& getChunkList() const noexcept
    {
        return chunk_list;
    }
};

} // namespace shrink_array

// Template method bodies. Included here (after the class definition) so that
// every translation unit using ArrayChunk2<T> sees the definitions and can
// auto-instantiate the exact T it needs, instead of relying on a fixed list of
// explicit instantiations in array_chunk2.cpp.
#include "chunk/internal/array_chunk2_inline.h"

// Bodies of the two mode-specific mixins. After array_chunk2_inline.h, so their
// CRTP self() sees a complete ArrayChunk2.
#include "chunk/internal/array_chunk2_parallel_inline.h"
#include "chunk/internal/array_chunk2_rechunk_hooks_inline.h"
