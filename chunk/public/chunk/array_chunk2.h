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

#include <iostream>    // for cout, ostream
#include <type_traits> // for is_same_v
#include <vector>      // for vector

#include <initializer_list> // the {a, b, c} constructor
#include <iterator>         // std::forward_iterator, std::distance -- the (first, last) constructor

#include "chunk/bit.h"                                  // is_bool_like_v: bool and Bit share the bool path
#include "chunk/internal/in_t.h"                        // InT<T>: the element snapshots
#include "chunk/internal/second_value_info.h"           // SecondValueInfo (InfArrayT)
#include "chunk/internal/array_chunk2_base.h"           // for ArrayChunk2Base
#include "chunk/internal/array_chunk2_parallel.h"       // ParallelSupport2 — OpenMP-only mixin
#include "chunk/internal/array_chunk2_rechunk_hooks.h"  // RechunkHooks2 — CHUNK_DYNAMIC-only mixin
#include "chunk/internal/array_chunk_base.h"            // for ArrayChunkBase
#include "chunk/internal/concept_array_chunk_element.h" // for ArrayChunkElement
#include "chunk/config.h"                               // for use_fast_block_compression, threaded
#include "chunk/internal/invariant_check.h"             // CHUNK_INVARIANT_FAILED
#include "chunk/internal/write_compress_sync_policy.h"  // for WriteCompressSyncPolicy
#include "chunk/PMP_CHUNK.h"                            // CHUNK_PMP_THREAD_CAPACITY, pmp::Team
#include "chunk/chunk2_std.h"                           // Chunk2<T>
#include "chunk/internal/proxy2.h"                      // Proxy2<T> and its registry

namespace shrink_array
{

template <template <typename> typename ChunkT, typename T>
class ManagerChunk;

namespace zero_one_value
{

/**
 * @brief Chunked array compressing on TWO constants: first_value and one second value.
 *
 * @details Unlike zero_any::Array, the second value is array-wide, not per chunk:
 *          cheaper metadata, useful when one non-zero constant dominates.
 *
 *          The second value lives in the shared registry
 *          Proxy2<T>::share_info_array, keyed by the array's ADDRESS, with the
 *          num_tab: hence the destructor's two erases and rechunk_to()'s six
 *          hooks (2A has three).
 *
 * @tparam T        stored element type
 * @tparam IntType index type used for sizes and element positions
 */
template <ArrayChunkElement T, typename IntType>
class Array : public ArrayChunk2Base<Array<T, IntType>>,
              public ParallelSupport2<Array<T, IntType>>,
              public FastBlockSupport2<Array<T, IntType>>,
              public RechunkHooks2<Array<T, IntType>>
{
  public:
    using Base2      = ArrayChunk2Base<Array>;   ///< Chunk2/2A common base (dirty bitset, second-value plumbing).
    using Base       = ArrayChunkBase<Array>;    ///< Base shared by every variant.
    using FastBlock2 = FastBlockSupport2<Array>; ///< Fast-block compression pass.
    using Hooks2     = RechunkHooks2<Array>;     ///< Re-chunk hooks (dynamic geometry).

    friend Base2;                       ///< The CRTP base reaches this class's private members.
    friend ArrayChunkBase<Array>;       ///< For swap_identity().

    // RechunkHooks2::rechunk_reown_after_swap copies the private ind_cst_val2.
    template <typename, bool>
    friend class RechunkHooks2;

    // Re-declared, not only inherited: the mixins declare them too, and an
    // outside `Arr::ChunkType` would be ambiguous between base subobjects.
    using Traits      = Base2::Traits;      ///< ArrayChunkTraits<zero_one_value::Array>.
    using ChunkType   = Base2::ChunkType;   ///< Chunk2<T>, the chunk type held in chunk_list.
    using ManagerType = Base2::ManagerType; ///< ManagerChunk<Chunk2, T>, the per-type manager.
    using ProxyType   = Base2::ProxyType;   ///< Proxy2<T>, the write proxy of the non-const accessors.
    using value_type  = T;                  ///< Stored element type.

    // The former snake_case spellings, kept until 1.0 (types are PascalCase).
    using chunk_type [[deprecated("types are PascalCase: use ChunkType")]]          = ChunkType;
    using manager_type [[deprecated("types are PascalCase: use ManagerType")]]      = ManagerType;
    using proxy_type [[deprecated("types are PascalCase: use ProxyType")]]          = ProxyType;
    using ind_type [[deprecated("types are PascalCase: use IndType")]]              = IntType;
    using chunk_list_type [[deprecated("types are PascalCase: use ChunkListType")]] = Base::ChunkListType;

    using Base2::chunk_list_;

    using Base::nb_tab;
    using Base::size_policy_;
    using Base2::m_size_;
    using Base2::nb_chunk_;
    using Base2::resize; // resize(sz, val), ArrayChunkBase's

    /// Registry entry of an array: (second value, its index in the shared value table, num_tab).
    using InfArrayT = SecondValueInfo<T>;

    using ManagerChunk2 = ManagerChunk<Chunk2, T>; ///< Same type as ManagerType.

    // Required, not decoration: ArrayChunkBase re-exports RechunkSupport's hook
    // DEFAULTS under these names, so a hook call through self() would be
    // ambiguous. Naming them here hides both bases' versions.
    using FastBlock2::make_block_compression;

    using Hooks2::rechunk_capture_state;
    using Hooks2::rechunk_drain_reuse_list;
    using Hooks2::rechunk_drop_candidates;
    using Hooks2::rechunk_reown_after_swap;
    using Hooks2::rechunk_seed_fresh;
    using Hooks2::rechunk_swap_identity;

    static constexpr bool is_not_bool = !is_bool_like_v<T>; ///< T is neither bool nor Bit: it has a second value.

    /**
     * @brief Build an array of @p sz elements.
     * @param sz initial element count; 0 by default, as std::vector (a fresh
     *           array is empty, so a push_back on it yields one element)
     */
    explicit Array(IntType sz = 0)
    {
        if constexpr (build_config::threaded)
        {
            WriteCompressSyncPolicy<Proxy2<T>>::ensure_initialized();
        }

        create(sz);
    }

    // ── The copy family: deep, element by element ────────────────────────────
    //
    // Never an alias of the source's buffers: ArrayChunkBase::copy_from. The
    // source's second value is re-declared first, so its runs stay compressible.
    // Sequential, like resize().
    /// @brief Deep copy: @p from's second value, then from.size() elements read from @p from. Sequential context.
    /// @param from Source array.
    Array(const Array& from) : Array(from.size())
    {
        copy_second_value_from(from);
        this->copy_from(from);
    }

    /**
     * @brief Move-construct: an empty array at the active geometry takes
     *        @p other's chunks, geometry and identity -- the exchange of
     *        swap() -- and @p other is left empty and valid.
     * @details O(1). Must stay user-declared: with the copy constructor and
     *          move-assignment declared, none is generated and a move would copy.
     *          Sequential context.
     * @note noexcept, so std::vector moves arrays on reallocation. Only memory
     *       exhaustion or a corrupt registry can fail here: std::terminate.
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

    /**
     * @brief Release the array's storage, then drop both registry entries.
     *
     * @details clear() first: it still needs get_num_tab(registry_key()) and a
     *          valid inf_array pointer.
     *
     * @warning Neither erase is optional:
     *          - share_info_array->erase(registry_key()): otherwise a later array
     *            at this address inherits our num_tab and second value
     *            (PointerMapChunk2's first-match lookup); rechunk_to()'s @c tmp
     *            makes address reuse routine.
     *          - erase_inf_array_by_ptr(&inf_array): map_inf_array entries
     *            pointing at our inf_array MEMBER would be dereferenced by the
     *            next array's terminal sweep (use after free). Fast-block,
     *            non-bool only; rechunk_to() re-points its value onto the
     *            survivor first, so only entries we still own go.
     */
    ~Array()
    {
        clear();
        Proxy2<T>::share_info_array->erase(this->registry_key());

        if constexpr (chunk_config::use_fast_block_compression && is_not_bool)
        {
            Proxy2<T>::share_info_array->erase_inf_array_by_ptr(&inf_array_);
        }
    }

    /**
     * @brief Move-assign: this array's own contents are released (clear()),
     *        then it takes @p other's -- chunks, geometry and identity, by the
     *        same exchange as swap(); @p other is left empty and valid.
     * @details The old chunks are released first (candidates purged); the moved
     *          chunks keep their metadata rows (num_tab). Sequential context.
     * @note noexcept, as the move constructor: a failure here terminates.
     */
    Array& operator=(Array&& other) noexcept
    {
        if (this != &other)
        {
            clear();
            this->swap_storage_and_identity(other);
        }
        return *this;
    }

    /// @brief Deep copy-assignment (see the copy constructor). Self-assignment
    ///        is a no-op.
    Array& operator=(const Array& from)
    {
        if (this != &from)
        {
            copy_second_value_from(from);
            this->copy_from(from);
        }
        return *this;
    }

    /**
     * @brief Set the array-wide second constant.
     * @details Registers @p sec_value in the shared cst_val2 registry, reusing an
     *          identical value's index. A reused block is re-filled to THIS
     *          array's geometry first: a larger array would otherwise read stale
     *          zeros past the registering array's size.
     * @param sec_value value chunks may now compress to
     * @param message   print the "useless for bool" warning (bool only)
     * @warning cst_val2 blocks never move and are never released: a live chunk
     *          of another array may hold the address as its `values`. One block
     *          per DISTINCT value ever declared, plus a candidate list per thread.
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
            return Proxy2<T>::share_info_array->get_second_value(this->registry_key());
        }
        else
        {
            return true;
        }
    }

    /// @brief Full registry tuple for this array: second value, its index, num_tab.
    [[nodiscard]] InfArrayT get_info_second_value() const noexcept
    {
        return Proxy2<T>::share_info_array->get_info_second_value(this->registry_key());
    }

    /// @brief Index of this array's second value in the shared value table.
    /// @warning Aborts for bool, which registers no second value.
    [[nodiscard]] int get_ind_second_value() const noexcept
    {
        if constexpr (is_not_bool)
        {
            return get_info_second_value().ind;
        }
        else
        {
            CHUNK_INVARIANT_FAILED(
                "zero_one_value::Array<bool>::get_ind_second_value(): bool registers no second value");
        }
    }

    /**
     * @brief Has the chunk holding element @p i been materialised?
     * @param i element index
     * @return false while that chunk is still compressed to a constant
     */
    [[nodiscard]] bool written(IntType i) const
    {
        return chunk_written(this->chunk_index_of(i));
    }

    /**
     * @brief Is chunk @p j materialised?
     * @param j CHUNK index (not an element index -- see written(IndType))
     * @return false while that chunk is compressed to either constant
     * @note Chunk2::written() needs the array's second-value index: see
     *       has_chunk_written_v (array_chunk_base.h).
     */
    [[nodiscard]] bool chunk_written(int j) const
    {
        if constexpr (is_not_bool)
        {
            return chunk_list_[j].written(get_ind_second_value());
        }
        else
        {
            return chunk_list_[j].written();
        }
    }

    /**
     * @brief Resize the array, preserving the elements that still fit.
     * @details Drains the compression candidates and deferred ops first: both
     *          hold `&chunk_list[j]` pointers a reallocation would dangle.
     * @param newsz new element count
     * @note Shrinking releases the dropped chunks' buffers (release_chunk_range():
     *       freed serially, parked in the geometry's reuse list under OpenMP);
     *       chunk_list keeps its capacity.
     */
    void resize(IntType newsz);

  private:
    /// @brief Release the buffers of chunks [@p first, @p last) -- the chunks
    ///        a shrinking resize() drops. Same release as clear_core(): freed
    ///        in place serially; under OpenMP parked in this array's geometry's
    ///        reuse list and the chunk neutralised to cst_val1. Sequential
    ///        point; the candidate lists must hold none of them.
    void release_chunk_range(int first, int last);

  public:
    /**
     * @brief Release all storage and report end-of-life statistics.
     * @details clear_core() plus the pre-drain compression stats.
     * @see clear_core
     */
    void clear();

    /**
     * @brief Audit pass: report the constant chunks the incremental compression missed.
     * @details Drains the candidate list first (a QUEUED compression is not a
     *          miss), then compresses materialised chunks that are fully one
     *          constant; those that survive are reported on stderr.
     * @note A diagnostic, not the engine (driven by the write path and
     *       manage_ompt.h's phase transitions).
     * @param team the team the scan is shared over (the one compress() opens,
     *             or the caller's region)
     */
    void compress_int(pmp::Team& team);

    /// @brief compress_int() on the calling thread's current team
    ///        (pmp::current_team(): the caller's region, or a team of one).
    void compress_int()
    {
        pmp::Team team = pmp::current_team();
        compress_int(team);
    }

    /**
     * @brief Public entry point for the audit pass.
     * @see compress_int, which this forwards to.
     */
    void compress();

    /// @brief Read-only access to the underlying chunk list.
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

  private:
    static constexpr bool is_bool = is_bool_like_v<T>; // bool and Bit (bit.h)

    /**
     * @brief Register the array and allocate its storage.
     * @param sz element count
     */
    void create(IntType sz);

    /**
     * @brief (Re)allocate storage at the current active geometry.
     * @details create()'s allocation WITHOUT its registration side effects;
     *          num_tab comes from the registry, so it suits an already-registered
     *          array switching geometry (rechunk_to()).
     * @param sz element count to allocate for
     */
    void rebuild_storage(IntType sz);

    /// @brief CRTP hook: the metadata the base needs for any element.
    /// @return this array's shared second-value tuple (element index unused,
    ///         since the value is array-wide -- unlike zero_any::Array)
    auto get_metadata(IntType /*ind*/) const noexcept
    {
        return this->get_info_second_value();
    }

    /**
     * @brief Storage teardown shared by clear() and rechunk_to().
     * @details clear() without the pre-drain compression stats.
     * @warning Parks buffers by THIS array's geometry, not the ambient thread
     *          size (another array's mid-rechunk). Drops this array's candidates
     *          by num_tab, so no Chunk2* of a dead array survives in the store.
     */
    void clear_core();

    /**
     * @brief Print this array's compression ratio and verdict.
     * @details Classifies chunks by RE-READING their content, not `written()`,
     *          so the count is reproducible across thread counts. Also counts
     *          the distinct second values (what Chunk2A would have captured).
     * @param os destination stream
     * @note Also FEEDS the global end-of-run counters when `chunk_config::stats` is on.
     */
    void print_compression_stats(std::ostream& os = std::cout) const;

    /// @brief Declare @p from's second value on this array, when it has one
    ///        (T{} means none: rechunk_seed_fresh's rule) and this array has
    ///        not declared it yet.
    void copy_second_value_from(const Array& from)
    {
        if constexpr (!is_bool_like_v<T>)
        {
            const InT<T> sv = from.get_second_value();
            if (sv != T{} && get_second_value() != sv)
            {
                set_second_value(sv, false);
            }
        }
    }

    /// @brief Address of the registry mirror, for ArrayChunkBase to arm write
    ///        proxies with (ProxySpec2::arm_inf_array).
    [[nodiscard]] const InfArrayT* registry_mirror() const noexcept
    {
        return &inf_array_;
    }

    /// @brief Re-read the mirror from the registry. Call after ANY write to
    ///        this array's registry entry -- a missed site is caught by the
    ///        any_check comparison in ProxySpec2::load_inf_array().
    void sync_registry_mirror() noexcept
    {
        inf_array_ = this->get_info_second_value();
    }

    /// @brief Make the T{} key drainable: its tuple in map_inf_array and a
    ///        candidate list per rank, if this type has none yet. Idempotent;
    ///        fast-block builds only (the serial path has no lists).
    void register_no_second_value_key()
    {
        if constexpr (is_not_bool && chunk_config::use_fast_block_compression)
        {
            auto& sh_inf = *Proxy2<T>::share_info_array;
            if (sh_inf.find_inf_array(T{}) == nullptr)
            {
                sh_inf.insert(T{}, &inf_array_none);
                // The between-phase pass drains only the values of THIS map: T{}
                // must be in it, at the sentinel index (so set_second_value(T{})
                // still sees it unregistered and builds a real block).
                sh_inf.insert_map_ind_cst_val2(T{}, chunk_config::no_second_value_index);
            }
            if constexpr (build_config::threaded)
            {
                for (int thr = 0; thr < CHUNK_PMP_THREAD_CAPACITY; ++thr)
                {
                    auto& slot = ManagerChunk2::map_chunk2_to_compress_potentially_thr[thr][T{}];
                    if (!slot)
                    {
                        slot = std::make_shared<typename std::remove_reference_t<decltype(slot)>::element_type>();
                    }
                }
            }
        }
    }

    /**
     * @brief The identity step of swap(): the registry tuple (second value,
     *        its index, num_tab), which is what Chunk2 reads its identity
     *        from, then the members that mirror it.
     * @details num_tab is not swapped (Chunk2 never reads it). map_inf_array
     *          points each second value at an array's inf_array MEMBER, which
     *          now holds the other tuple: both values are re-pointed.
     */
    void swap_identity(Array& other)
    {
        auto& sh_inf = *Proxy2<T>::share_info_array;
        sh_inf.swap_entries(this->registry_key(), other.registry_key());
        std::swap(ind_cst_val2_, other.ind_cst_val2_);
        sync_registry_mirror();
        other.sync_registry_mirror();
        if constexpr (chunk_config::use_fast_block_compression && is_not_bool)
        {
            for (Array* arr : {this, &other})
            {
                const T value = arr->inf_array_.second_value;
                if (value != T{})
                {
                    sh_inf.insert(value, &arr->inf_array_);
                }
            }
        }
    }

    /// This array's (second value, index, num_tab) tuple: a MIRROR of its entry
    /// in Proxy2<T>::share_info_array, complete in every build mode.
    /// @details The registry is the source of truth; the mirror is only re-read
    ///          from it (sync_registry_mirror()) after each registry write:
    ///          create(), set_second_value(), the identity swaps. Write proxies
    ///          are armed with its address, keeping the registry scan off the
    ///          write path.
    /// @warning map_inf_array holds a POINTER to this member, so the destructor
    ///          must erase that entry -- see its documentation.
    InfArrayT inf_array_;

    /**
     * @brief The (T{}, no_second_value_index) tuple every array without a
     *        second value is drained under.
     * @details The compression pass walks map_inf_array; an array with no
     *          second value queues under T{}, which needs this tuple and a list
     *          per rank (register_no_second_value_key()). Never erased.
     */
    static inline InfArrayT inf_array_none{T{}, chunk_config::no_second_value_index, 0};

    /// Index of the second value in the shared value table;
    /// chunk_config::no_second_value_index until set_second_value().
    int ind_cst_val2_ = chunk_config::no_second_value_index;
};

} // namespace zero_one_value

} // namespace shrink_array

// Template bodies, after the class, so every TU instantiates the T it needs.
#include "chunk/internal/array_chunk2_inline.h" // the template bodies

// The mixins' bodies, after array_chunk2_inline.h: self() needs a complete zero_one_value::Array.
#include "chunk/internal/array_chunk2_parallel_inline.h"      // ParallelSupport2's bodies
#include "chunk/internal/array_chunk2_rechunk_hooks_inline.h" // RechunkHooks2's bodies
