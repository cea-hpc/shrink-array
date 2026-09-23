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
// Created by ballereaup on 20/11/24.
//

#pragma once

#include <vector>

#include <initializer_list> // the {a, b, c} constructor
#include <iterator>         // std::forward_iterator, std::distance -- the (first, last) constructor
#include "chunk/chunk_std.h"
#include "chunk/internal/array_chunk_base.h"
#include "chunk/internal/array_chunk_parallel.h"      // ParallelSupport — OpenMP-only mixin
#include "chunk/internal/array_chunk_rechunk_hooks.h" // RechunkHooks — CHUNK_DYNAMIC-only mixin
#include "chunk/internal/chunk_stats_printer.h"
#include "chunk/PMP_CHUNK.h"
#include "chunk/internal/concept_array_chunk_element.h"
#include "chunk/config.h"
#include "chunk/internal/write_compress_sync_policy.h"
// !
//  * \brief Compressed "array" class

namespace shrink_array
{

template <template <typename> typename CHUNK_T, typename T>
class ManagerChunk;

template <ArrayChunkElement T, typename ind_type>
class ArrayChunk : public ArrayChunkBase<ArrayChunk<T, ind_type>>,
                   public ParallelSupport<ArrayChunk<T, ind_type>>,
                   public FastBlockSupport<ArrayChunk<T, ind_type>>,
                   public RechunkHooks<ArrayChunk<T, ind_type>>
{
    using Base     = ArrayChunkBase<ArrayChunk>;
    using Parallel = ParallelSupport<ArrayChunk>;
    using Hooks    = RechunkHooks<ArrayChunk>;

  public:
    using Traits       = Base::Traits;
    using chunk_type   = Base::chunk_type;
    using manager_type = Base::manager_type;
    using proxy_type   = Base::proxy_type;
    using value_type   = T;

    using Base::chunk_list;
    using Base::m_size;
    using Base::nb_chunk;
    using Base::nb_tab;
    using Base::size_policy_;
    using manager_chunk = ManagerChunk<Chunk, T>;

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
    // Copy-assignment used to alias: chunk_list copied, so two arrays over the
    // same buffers with no contract for it. A copy is a copy now, built on
    // ArrayChunkBase::copy_from (sized to the source, every element read
    // through the source's gate and written through this array's proxy);
    // the (first, last) and initializer-list constructors are the same loop
    // over an iterator. All sequential, like resize().
    ArrayChunk(const ArrayChunk& from) : ArrayChunk(from.size())
    {
        this->copy_from(from);
    }

    template <std::forward_iterator It>
    ArrayChunk(It first, It last) : ArrayChunk(static_cast<ind_type>(std::distance(first, last)))
    {
        this->assign(first, last);
    }

    ArrayChunk(std::initializer_list<T> il) : ArrayChunk(il.begin(), il.end()) {}

    ~ArrayChunk()
    {
        clear();
        // Drop our (this -> num_tab) registry entry so a later array reusing this
        // address does not inherit our stale num_tab via the first-match lookup
        // in PointerMapChunk. Required for rechunk_to's tmp; correct for every
        // array. Must run AFTER clear() (which still needs get_num_tab(registry_key())).
        Proxy<T>::share_info_array.erase(this->registry_key());
    }
    /// @brief Claim this array's identity before its storage is built:
    ///        register its address and take the next free num_tab, which
    ///        every per-tab metadata table is then indexed by. Called by
    ///        create() through ArrayChunkBase.
    void pre_create_hook()
    {
        auto& sh_inf = Proxy<T>::share_info_array;

        sh_inf.insert(this->registry_key(), Base::nb_tab);

        this->num_tab = Base::nb_tab;

        nb_tab++;
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

    /**
     * @brief Move-assign: this array's own contents are released (clear()),
     *        then it takes @p other's -- chunks, geometry and identity, by the
     *        same exchange as swap(); @p other is left empty and valid.
     * @details It used to steal @p other's chunk list and nothing else: this
     *          array's old chunks were destroyed with their buffers still
     *          queued as compression candidates, the moved chunks kept
     *          @p other's metadata rows (num_tab), and @p other kept its
     *          nb_chunk over an empty list. Sequential context, as clear().
     * @throws std::bad_alloc from clear() -- hence not noexcept.
     */
    ArrayChunk& operator=(ArrayChunk&& other)
    {
        if (this != &other)
        {
            clear();
            this->swap_storage_and_identity(other);
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

    // make_block_compression() comes from FastBlockSupport (a no-op outside
    // fast-block compression). Re-exported so resize() can call it unqualified.
    using FastBlockSupport<ArrayChunk>::make_block_compression;

    /// @brief Resize to @p newsz elements; new elements read the first
    ///        constant. A sequential point: the rank's candidates are
    ///        compressed and the deferred ops drained first.
    void resize(ind_type newsz);

    /// @brief resize(), then @p val written to the NEW elements only
    ///        ([old size, newsz) -- std::vector's semantics; assign() fills all).
    void resize(ind_type newsz, in_t<T> val);

    /// @brief Release the storage and report this array's end-of-life stats.
    void clear();

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

    /// @brief The compression pass proper, run by every thread of the region
    ///        compress() opens (or on the calling thread outside one).
    void compress_int(bool verbose);

    /// @brief Compress every chunk that is constant at the first value now.
    void compress(bool verbose = false)
    {
        const bool in_parr = CHUNK_PMP_IN_PARALLEL_REGION;

        if (in_parr)
        {
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

    /// @brief Deep copy-assignment (see the copy constructor). Self-assignment
    ///        is a no-op.
    ArrayChunk& operator=(const ArrayChunk<T, ind_type>& from)
    {
        if (this != &from)
            this->copy_from(from);
        return *this;
    }

    /// @brief Read-only access to the underlying chunk list.
    [[nodiscard]] const std::vector<Chunk<T>>& getChunkList() const
    {
        return chunk_list;
    }

  private:
    friend ArrayChunkBase<ArrayChunk<T, ind_type>>; // swap_identity()

    /// @brief The identity step of swap(): num_tab and the registry entry, so
    ///        the per-num_tab tables follow the chunks.
    void swap_identity(ArrayChunk& other)
    {
        using std::swap;
        swap(this->num_tab, other.num_tab);
        Proxy<T>::share_info_array.swap_keys(this->registry_key(), other.registry_key());
    }
};

} // namespace shrink_array

// Template method bodies. Included here (after the class definition) so that
// every translation unit using ArrayChunk<T> sees the definitions and can
// auto-instantiate the exact T it needs.
#include "chunk/internal/array_chunk_inline.h"

// Bodies of the two mode-specific mixins. After array_chunk_inline.h, so their
// CRTP self() sees a complete ArrayChunk.
#include "chunk/internal/array_chunk_parallel_inline.h"
#include "chunk/internal/array_chunk_rechunk_hooks_inline.h"
