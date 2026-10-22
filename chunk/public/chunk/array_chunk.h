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
#include "chunk/PMP_CHUNK.h"                            // CHUNK_PMP_IN_PARALLEL_REGION, pmp::team / Team
#include "chunk/internal/concept_array_chunk_element.h" // ArrayChunkElement
#include "chunk/config.h"                               // build_config::threaded
#include "chunk/internal/write_compress_sync_policy.h"  // WriteCompressSyncPolicy
#include "chunk/internal/in_t.h"                        // InT<T>: resize's fill value
#include "chunk/internal/proxy.h"                       // Proxy<T>: the element proxy and its registry

namespace shrink_array
{

template <template <typename> typename ChunkT, typename T>
class ManagerChunk;

/**
 * @brief Compressed array: chunks of @p T that each compress to the first
 *        value (Chunk, variant v1).
 *
 * @tparam T        stored element type
 * @tparam IndType index type used for sizes and element positions
 */
template <ArrayChunkElement T, typename IndType>
class ArrayChunk : public ArrayChunkBase<ArrayChunk<T, IndType>>,
                   public ParallelSupport<ArrayChunk<T, IndType>>,
                   public FastBlockSupport<ArrayChunk<T, IndType>>,
                   public RechunkHooks<ArrayChunk<T, IndType>>
{
  private:
    friend ArrayChunkBase<ArrayChunk>; // swap_identity()

    using Base     = ArrayChunkBase<ArrayChunk>;
    using Parallel = ParallelSupport<ArrayChunk>;
    using Hooks    = RechunkHooks<ArrayChunk>;

    /// @brief Claim this array's identity before its storage is built:
    ///        register its address and take the next free num_tab, which
    ///        every per-tab metadata table is then indexed by. Called by
    ///        create() through ArrayChunkBase.
    void pre_create_hook()
    {
        auto& sh_inf = *Proxy<T>::share_info_array;

        sh_inf.insert(this->registry_key(), Base::nb_tab);

        this->num_tab_ = Base::nb_tab;

        ++nb_tab;
    }

    /// @brief Storage (re)allocation at the current active geometry, WITHOUT
    ///        the identity work create_common() does. Used by create() and by
    ///        rechunk_to().
    /// @param sz element count to allocate for
    void rebuild_storage(IndType sz);

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
        swap(this->num_tab_, other.num_tab_);
        Proxy<T>::share_info_array->swap_keys(this->registry_key(), other.registry_key());
    }

  public:
    using Traits       = Base::Traits;       ///< ArrayChunkTraits<ArrayChunk>.
    using chunk_type   = Base::chunk_type;   ///< Chunk<T>, the chunk type held in chunk_list.
    using manager_type = Base::manager_type; ///< ManagerChunk<Chunk, T>, the per-type manager.
    using proxy_type   = Base::proxy_type;   ///< Proxy<T>, the write proxy of the non-const accessors.
    using value_type   = T;                  ///< Stored element type.

    using Base::chunk_list_;
    using Base::m_size_;
    using Base::nb_chunk_;
    using Base::nb_tab;
    using Base::size_policy_;
    using Manager = ManagerChunk<Chunk, T>; ///< Same type as manager_type.

    // Mode-specific members live in the two mixins above. Do not drop these
    // using-declarations: ArrayChunkBase re-exports RechunkSupport's hook
    // DEFAULTS under the same names, so a hook call through the CRTP self()
    // would be ambiguous. Naming them here hides both bases' versions.
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
    explicit ArrayChunk(IndType sz = 0)
    {
        if constexpr (build_config::threaded)
        {
            WriteCompressSyncPolicy<Proxy<T>>::ensure_initialized();
        }

        this->create(sz);
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
     * @details O(1) in the elements. Must be user-declared: with the copy
     *          constructor and move-assignment declared, none is generated and
     *          a move would run the element-by-element copy. Sequential
     *          context, as the constructors.
     * @note noexcept, so that std::vector moves arrays when it reallocates.
     *       What can fail here (the empty array's registration, the drain)
     *       fails only on memory exhaustion or a corrupt registry, already
     *       fatal in the library: std::terminate.
     */
    ArrayChunk(ArrayChunk&& other) noexcept : ArrayChunk(0)
    {
        this->swap_storage_and_identity(other);
    }

    /// @brief Build an array of std::distance(@p first, @p last) elements and assign them from the range.
    /// @param first Start of the source range.
    /// @param last End of the source range.
    template <std::forward_iterator It>
    ArrayChunk(It first, It last) : ArrayChunk(static_cast<IndType>(std::distance(first, last)))
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
        // Drop our (this -> num_tab) registry entry so a later array at this
        // address does not inherit our stale num_tab via PointerMapChunk's
        // first-match lookup (required for rechunk_to's tmp). Must run AFTER
        // clear(), which still needs get_num_tab(registry_key()).
        Proxy<T>::share_info_array->erase(this->registry_key());
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
    ArrayChunk& operator=(const ArrayChunk<T, IndType>& from)
    {
        if (this != &from)
        {
            this->copy_from(from);
        }
        return *this;
    }

    /// @brief Has the chunk holding element @p i been materialised?
    [[nodiscard]] bool written(IndType i) const noexcept
    {
        return chunk_list_[this->chunk_index_of(i)].written();
    }

    /// @brief Materialise the chunk holding element @p i if it is still
    ///        compressed; a no-op otherwise.
    void alloc_chunk(IndType i) noexcept
    {
        chunk_list_[this->chunk_index_of(i)].alloc_chunk();
    }

    /// @brief alloc_chunk(), taking the buffer from rank @p thr_rank's pool.
    void alloc_chunk(IndType i, int thr_rank) noexcept
    {
        chunk_list_[this->chunk_index_of(i)].alloc_chunk(thr_rank);
    }

    /// @brief Resize to @p newsz elements; new elements read the first
    ///        constant. A sequential point: the rank's candidates are
    ///        compressed and the deferred ops drained first.
    /// @note Shrinking releases the dropped chunks' buffers (release_chunk_range():
    ///       freed serially, parked in the geometry's reuse list under OpenMP);
    ///       chunk_list keeps its capacity.
    void resize(IndType newsz);

  private:
    /// @brief Release the buffers of chunks [@p first, @p last) -- the chunks
    ///        a shrinking resize() drops. Same release as clear_core(): freed
    ///        in place serially; under OpenMP parked in this array's geometry's
    ///        reuse list and the chunk neutralised to cst_val1. Sequential
    ///        point; the candidate lists must hold none of them.
    void release_chunk_range(int first, int last);

  public:
    using Base::resize; // resize(newsz, val), ArrayChunkBase's

    /// @brief Release the storage and report this array's end-of-life stats.
    void clear();

    /// @brief The compression pass proper, run by every thread of @p team
    ///        (the one compress() opens, or the caller's region).
    void compress_int(pmp::Team& team, bool verbose);

    /// @brief compress_int() on the calling thread's current team
    ///        (pmp::current_team(): the caller's region, or a team of one).
    void compress_int(bool verbose)
    {
        pmp::Team team = pmp::current_team();
        compress_int(team, verbose);
    }

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
            pmp::team(pmp::options{}, [&](pmp::Team& team) { compress_int(team, verbose); });
        }
    }

    /// @brief Read-only access to the underlying chunk list.
    [[nodiscard]] const Base::chunk_list_type& get_chunk_list() const noexcept
    {
        return chunk_list_;
    }

    /// @brief Former name of get_chunk_list(), kept for the programs that still spell it.
    [[deprecated("methods are snake_case: use get_chunk_list")]] [[nodiscard]] const Base::chunk_list_type&
    getChunkList() const noexcept
    {
        return chunk_list_;
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
