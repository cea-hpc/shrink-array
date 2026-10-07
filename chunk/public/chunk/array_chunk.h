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

#include <iostream> // std::cout: print_compression_stats' default stream
#include <utility>  // std::swap
#include <vector>   // std::vector

#include <initializer_list> // the {a, b, c} constructor
#include <iterator>         // std::forward_iterator, std::distance -- the (first, last) constructor

#include "chunk/chunk_std.h"                     // Chunk<T>
#include "chunk/internal/read_reclaim.h"         // read_reclaim::PassScope (perform_compression / compress in a region)
#include "chunk/internal/array_chunk_base.h"     // ArrayChunkBase
#include "chunk/internal/array_chunk_parallel.h" // ParallelSupport — OpenMP-only mixin
#include "chunk/internal/array_chunk_rechunk_hooks.h"   // RechunkHooks — CHUNK_DYNAMIC-only mixin
#include "chunk/PMP_CHUNK.h"                            // CHUNK_PMP_IN_PARALLEL_REGION
#include "chunk/internal/concept_array_chunk_element.h" // ArrayChunkElement
#include "chunk/config.h"                               // build_config::use_openmp
#include "chunk/internal/write_compress_sync_policy.h"  // WriteCompressSyncPolicy
#include "chunk/internal/in_t.h"                        // in_t<T>: resize's fill value
#include "chunk/internal/proxy.h"                       // Proxy<T>: the element proxy and its registry

namespace shrink_array
{

template <template <typename> typename CHUNK_T, typename T>
class ManagerChunk;

/**
 * @brief Compressed array: chunks of @p T that each compress to the first
 *        value (Chunk, variant v1).
 *
 * @tparam T        stored element type
 * @tparam ind_type index type used for sizes and element positions
 */
template <ArrayChunkElement T, typename ind_type>
class ArrayChunk : public ArrayChunkBase<ArrayChunk<T, ind_type>>,
                   public ParallelSupport<ArrayChunk<T, ind_type>>,
                   public FastBlockSupport<ArrayChunk<T, ind_type>>,
                   public RechunkHooks<ArrayChunk<T, ind_type>>
{
  private:
    friend ArrayChunkBase<ArrayChunk<T, ind_type>>; // swap_identity()

    using Base     = ArrayChunkBase<ArrayChunk>;
    using Parallel = ParallelSupport<ArrayChunk>;
    using Hooks    = RechunkHooks<ArrayChunk>;

    /// @brief Claim this array's identity before its storage is built:
    ///        register its address and take the next free num_tab, which
    ///        every per-tab metadata table is then indexed by. Called by
    ///        create() through ArrayChunkBase.
    void pre_create_hook()
    {
        auto& sh_inf = Proxy<T>::share_info_array;

        sh_inf.insert(this->registry_key(), Base::nb_tab);

        this->num_tab = Base::nb_tab;

        ++nb_tab;
    }

    /// @brief Claim this array's identity and build its storage for @p sz
    ///        elements (the constructors' body; registers the element type's
    ///        manager on the way, see AllTypeChunk::ensure_manager).
    void create(ind_type sz);

    /// @brief Storage (re)allocation at the current active geometry, WITHOUT
    ///        the identity work create_common() does. Used by create() and by
    ///        rechunk_to().
    /// @param sz element count to allocate for
    void rebuild_storage(ind_type sz);

    /// @brief Storage teardown shared by clear() and rechunk_to(): everything
    ///        clear() does except the pre-drain stats. Parks buffers by THIS
    ///        array's geometry and removes this array's compression candidates
    ///        by num_tab.
    void clear_core();

    /// @brief Per-array compression stats + "would Chunk2 have helped?" probe.
    /// Plain Chunk compresses a block only when it is constant to first_value;
    /// this reports the compression ratio (vs a plain std::vector) and scans the
    /// materialised (uncompressed) blocks for ones that are in fact constant to
    /// some OTHER value — those a Chunk2/Chunk2A would have captured. Read at the
    /// start of clear(), before the drain mutates chunk state.
    void print_compression_stats(std::ostream& os = std::cout) const;

    /// @brief The identity step of swap(): num_tab and the registry entry, so
    ///        the per-num_tab tables follow the chunks.
    void swap_identity(ArrayChunk& other)
    {
        using std::swap;
        swap(this->num_tab, other.num_tab);
        Proxy<T>::share_info_array.swap_keys(this->registry_key(), other.registry_key());
    }

  public:
    using Traits       = Base::Traits;       ///< ArrayChunkTraits<ArrayChunk>.
    using chunk_type   = Base::chunk_type;   ///< Chunk<T>, the chunk type held in chunk_list.
    using manager_type = Base::manager_type; ///< ManagerChunk<Chunk, T>, the per-type manager.
    using proxy_type   = Base::proxy_type;   ///< Proxy<T>, the write proxy of the non-const accessors.
    using value_type   = T;                  ///< Stored element type.

    using Base::chunk_list;
    using Base::m_size;
    using Base::nb_chunk;
    using Base::nb_tab;
    using Base::size_policy_;
    using manager_chunk = ManagerChunk<Chunk, T>; ///< Same type as manager_type.

    // Mode-specific members, moved out of this class into the two mixins above.
    // The using-declarations are not decoration: ArrayChunkBase re-exports
    // RechunkSupport's hook DEFAULTS under these very names, so without them a
    // hook call through the CRTP self() would see two base subobjects and be
    // ambiguous. Naming them here makes them members of ArrayChunk, which hides
    // both bases' versions.
    using Parallel::flush_pending_deferred_ops;

    using Hooks::rechunk_drain_reuse_list;
    using Hooks::rechunk_drop_candidates;
    using Hooks::rechunk_swap_identity;

    // make_block_compression() comes from FastBlockSupport (a no-op outside
    // fast-block compression). Re-exported so resize() can call it unqualified.
    using FastBlockSupport<ArrayChunk>::make_block_compression;

    /**
     * @brief Build an array of @p sz elements.
     * @param sz initial element count; 0 by default, as std::vector (a fresh
     *           array is empty, so a push_back on it yields one element)
     */
    explicit ArrayChunk(ind_type sz = 0)
    {
        if constexpr (build_config::use_openmp)
        {
            WriteCompressSyncPolicy<Proxy<T>>::ensure_initialized();
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
    /// @brief Deep copy: an array of from.size() elements, each read from @p from. Sequential context.
    /// @param from Source array.
    ArrayChunk(const ArrayChunk& from) : ArrayChunk(from.size())
    {
        this->copy_from(from);
    }

    /**
     * @brief Move-construct: an empty array at the active geometry takes
     *        @p other's chunks, geometry and identity -- the exchange of
     *        swap() -- and @p other is left empty and valid.
     * @details O(1) in the elements. With a copy constructor and a
     *          move-assignment declared, none was generated: `ArrayChunk a(std::move(b))`
     *          and every return that is not elided ran the copy constructor,
     *          element by element. Sequential context, as the constructors.
     * @note noexcept, so that std::vector moves arrays when it reallocates
     *       instead of copying them element by element. What can fail here
     *       -- the empty array's registration, the deferred-write drain --
     *       only fails on memory exhaustion or a corrupt registry, which the
     *       library already treats as fatal (alloc_chunk() is noexcept too):
     *       std::terminate.
     */
    ArrayChunk(ArrayChunk&& other) noexcept : ArrayChunk(0)
    {
        this->swap_storage_and_identity(other);
    }

    /// @brief Build an array of std::distance(@p first, @p last) elements and assign them from the range.
    /// @param first Start of the source range.
    /// @param last End of the source range.
    template <std::forward_iterator It>
    ArrayChunk(It first, It last) : ArrayChunk(static_cast<ind_type>(std::distance(first, last)))
    {
        this->assign(first, last);
    }

    /// @brief Build an array holding the elements of @p il.
    /// @param il Source values.
    ArrayChunk(std::initializer_list<T> il) : ArrayChunk(il.begin(), il.end()) {}

    /// @brief clear() the array, then erase its entry from Proxy<T>::share_info_array.
    ~ArrayChunk()
    {
        clear();
        // Drop our (this -> num_tab) registry entry so a later array reusing this
        // address does not inherit our stale num_tab via the first-match lookup
        // in PointerMapChunk. Required for rechunk_to's tmp; correct for every
        // array. Must run AFTER clear() (which still needs get_num_tab(registry_key())).
        Proxy<T>::share_info_array.erase(this->registry_key());
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
    ArrayChunk& operator=(ArrayChunk&& other) noexcept
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
    ArrayChunk& operator=(const ArrayChunk<T, ind_type>& from)
    {
        if (this != &from)
        {
            this->copy_from(from);
        }
        return *this;
    }

    /// @brief Has the chunk holding element @p i been materialised?
    [[nodiscard]] bool written(ind_type i) const noexcept
    {
        return chunk_list[this->chunk_index_of(i)].written();
    }

    /// @brief Materialise the chunk holding element @p i if it is still
    ///        compressed; a no-op otherwise.
    void alloc_chunk(ind_type i) noexcept
    {
        chunk_list[this->chunk_index_of(i)].alloc_chunk();
    }

    /// @brief alloc_chunk(), taking the buffer from rank @p thr_rank's pool.
    void alloc_chunk(ind_type i, int thr_rank) noexcept
    {
        chunk_list[this->chunk_index_of(i)].alloc_chunk(thr_rank);
    }

    /// @brief Resize to @p newsz elements; new elements read the first
    ///        constant. A sequential point: the rank's candidates are
    ///        compressed and the deferred ops drained first.
    void resize(ind_type newsz);

    /// @brief resize(), then @p val written to the NEW elements only
    ///        ([old size, newsz) -- std::vector's semantics; assign() fills all).
    void resize(ind_type newsz, in_t<T> val);

    /// @brief Release the storage and report this array's end-of-life stats.
    void clear();

    /// @brief The compression pass proper, run by every thread of the region
    ///        compress() opens (or on the calling thread outside one).
    void compress_int(bool verbose);

    /// @brief Compress every chunk that is constant at the first value now.
    void compress(bool verbose = false)
    {
        const bool in_parr = CHUNK_PMP_IN_PARALLEL_REGION;

        if (in_parr)
        {
            // Inside the caller's region: other threads may be reading, so
            // the buffers released wait for a grace period (read_reclaim.h).
            const read_reclaim::PassScope pass_scope;
            compress_int(verbose);
        }
        else
        {
#pragma omp parallel
            {
                compress_int(verbose);
            }
        }
    }

    /// @brief Read-only access to the underlying chunk list.
    [[nodiscard]] const std::vector<Chunk<T>>& getChunkList() const noexcept
    {
        return chunk_list;
    }
};

} // namespace shrink_array

// Template method bodies. Included here (after the class definition) so that
// every translation unit using ArrayChunk<T> sees the definitions and can
// auto-instantiate the exact T it needs.
#include "chunk/internal/array_chunk_inline.h" // the template bodies

// Bodies of the two mode-specific mixins. After array_chunk_inline.h, so their
// CRTP self() sees a complete ArrayChunk.
#include "chunk/internal/array_chunk_parallel_inline.h"      // ParallelSupport's bodies
#include "chunk/internal/array_chunk_rechunk_hooks_inline.h" // RechunkHooks' bodies
