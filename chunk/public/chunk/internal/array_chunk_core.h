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

/**
 * @file array_chunk_core.h
 * @brief The part of a chunked array that does not depend on its index type.
 *
 * @details zero::Array*<T, IndT> are distinct types for every IndT, but
 *          the machinery around them -- proxies, chunks, managers, the per-T
 *          registries (<tt>Proxy*\<T\>\::share_info_array</tt>) -- is templated on T alone
 *          and reaches the array through a type-erased pointer (the proxy's
 *          ptr_on_array, the registries' keys). Do not cast that pointer back to
 *          `zero::Array*<T, int>`: on a `<T, long>` array the fields sit elsewhere.
 *
 *          ArrayChunkCore<ChunkType> is what that code may see: everything it
 *          reads, laid out identically whatever the index type. The pointer
 *          handed out is the address of THIS subobject (registry_key()), so a
 *          static_cast from the void* back to ArrayChunkCore is well-defined.
 *          The index-typed part (m_size, the element API) stays in ArrayChunkBase.
 *
 *          The per-T counters nb_tab (source of num_tab) and next_alloc_key_ live
 *          here too: they must be shared by `<T, int>` and `<T, long>` arrays, or
 *          two arrays of one process get the same num_tab.
 *
 * @see array_chunk_base.h, which derives from it
 * @see array_chunk_for.h, which maps a chunk type to it for the proxies
 */

#include <atomic>      // std::atomic (next_alloc_key_)
#include <cstddef>     // std::size_t
#include <cstdint>     // std::uint64_t
#include <cstdlib>     // std::getenv (CHUNK_ALLOC_AFFINITY_EAGER)
#include <memory>      // std::construct_at (construct_chunk)
#include <string>      // std::string (debug_name_)
#include <type_traits> // std::conditional_t
#include <utility>     // std::move (set_debug_name)
#include <vector>      // std::vector (chunk_list)

#include "chunk/config.h"                            // build_config::alloc_affinity_min_bytes, build_config::any_check
#include "chunk/chunk_traits.h"                      // ChunkTraits (ProxyType, ManagerType)
#include "chunk/internal/chunk_size_policy.h"        // ActiveSizePolicy
#include "chunk/internal/array_chunk_dirty_bitset.h" // DirtyBitsetSupport
#include "chunk/internal/parallel_storage_init.h"    // NoInitAllocator (chunk_list)

namespace shrink_array
{

namespace alloc_affinity_detail
{
/// Eager (an allocator set per large array at creation) or lazy (only the
/// storage a re-chunk builds gets one, see assign_alloc_key()) -- eager by
/// default for Chunk (v1), lazy for Chunk2 / Chunk2A. CHUNK_ALLOC_AFFINITY_EAGER
/// (read once), when set, decides: =0 lazy, anything else eager.
/// v1's memory is mostly data and its rebuild is sliced: from the shared pool,
/// whose slabs mix several arrays' blocks, no slab empties during the first
/// re-chunk, so eager lowers its peak by about a tenth. Chunk2 / Chunk2A are not
/// sliced and are metadata-heavy: lazy saves them the per-array slabs.
inline const bool g_eager = []() noexcept
{
    const char* e = std::getenv("CHUNK_ALLOC_AFFINITY_EAGER");
    if (e == nullptr || e[0] == '\0')
    {
        return chunk_config::use_chunk; // the variant's default
    }
    return e[0] != '0';
}();

/// True while rechunk_to() constructs its new storage (tmp): the storage that
/// assign_alloc_key() gives an allocator set of its own. Set and reset at a
/// quiescent serial point, around one constructor call.
inline bool g_building_rechunk_storage = false;

/// @brief RAII scope for g_building_rechunk_storage.
struct RechunkStorageScope
{
    /// @brief Set g_building_rechunk_storage.
    RechunkStorageScope() noexcept
    {
        g_building_rechunk_storage = true;
    }
    /// @brief Reset g_building_rechunk_storage to false.
    ~RechunkStorageScope()
    {
        g_building_rechunk_storage = false;
    }
    RechunkStorageScope(const RechunkStorageScope&)            = delete;
    RechunkStorageScope& operator=(const RechunkStorageScope&) = delete;
    RechunkStorageScope(RechunkStorageScope&&)                 = delete;
    RechunkStorageScope& operator=(RechunkStorageScope&&)      = delete;
};

/// CHUNK_RECHUNK_REUSE_LIST (read once, default on; 0 = off): a re-chunk's
/// dying storage hands its chunk_list buffer to the next re-chunk's new
/// storage (ArrayChunkCore::acquire_chunk_list / release_chunk_list) instead of
/// freeing it. Off = a fresh vector per re-chunk.
inline const bool g_reuse_chunk_list = []() noexcept
{
    const char* e = std::getenv("CHUNK_RECHUNK_REUSE_LIST");
    return e == nullptr || e[0] == '\0' || e[0] != '0';
}();

/// True while rechunk_to() destroys its temporary, which by then holds the OLD
/// storage: its clear_core() keeps the chunk_list buffer for the next
/// re-chunk. Set and reset at a quiescent serial point, around one destructor.
inline bool g_releasing_rechunk_storage = false;
} // namespace alloc_affinity_detail

// chunk_list's allocator is NoInitAllocator (parallel_storage_init.h): resize() only
// reserves, and the init loops construct each chunk (ArrayChunkCore::construct_chunk) on
// the thread that initialises it -- first touch and NUMA placement there, not a serial
// value-init sweep on the calling thread. The resize sites and their init loops:
// zero::Array*::create (through acquire_chunk_list) and zero::Array*::resize.

/// @brief The index-type-independent part of a chunked array of ChunkT (see the description at the top of
///        this file).
template <typename ChunkT>
class ArrayChunkCore : public DirtyBitsetSupport<ArrayChunkCore<ChunkT>>
{
  public:
    using ChunkType     = ChunkT;                                             ///< Chunk type held in chunk_list.
    using ChunkListType = std::vector<ChunkType, NoInitAllocator<ChunkType>>; ///< chunk_list's type.
    using ProxyType     = ChunkTraits<ChunkT>::ProxyType;                     ///< Proxy type owning the per-T registry.
    using ManagerType   = ChunkTraits<ChunkT>::ManagerType;                   ///< Per-T manager owning the allocators.

    /// @brief The chunk geometry: elements per chunk.
    [[nodiscard]] unsigned int chunk_size() const noexcept
    {
        return size_policy_.size();
    }
    /// @brief This array's slab-affinity key (0 = shared pool). See alloc_key_.
    [[nodiscard]] unsigned int alloc_key() const noexcept
    {
        return alloc_key_;
    }
    /// @brief Number of live chunks.
    [[nodiscard]] int chunk_count() const noexcept
    {
        return nb_chunk_;
    }

    /**
     * @brief Name this array in the end-of-life stats line, next to its table
     *        number: `[tab_quant #20 double]` instead of `[tab_20 double]`.
     * @details Every re-chunk gives the array a new table number, so the number
     *          alone does not identify the variable. The name stays with the object
     *          (re-chunk and swap() move storage, not it). Empty: the number alone.
     * @param name the name to print
     */
    void set_debug_name(std::string name)
    {
        debug_name_ = std::move(name);
    }
    /// @brief The name set by set_debug_name(), empty if none.
    [[nodiscard]] const std::string& debug_name() const noexcept
    {
        return debug_name_;
    }

    /**
     * @brief Direct access to one chunk, for bulk range operations.
     *
     * @details NOT part of the container interface: it exposes the chunked
     *          representation, for the bulk operations of array_chunk_range_ops.h
     *          (one decision per chunk, not per element). Everything else goes
     *          through operator[].
     *
     * @param i chunk index; the caller guarantees 0 <= i < nb_chunk
     */
    [[nodiscard]] ChunkType& chunk_ref(std::size_t i) noexcept
    {
        return chunk_list_[i];
    }

    /// @brief Const overload of chunk_ref(): the same chunk, read-only.
    [[nodiscard]] const ChunkType& chunk_ref(std::size_t i) const noexcept
    {
        return chunk_list_[i];
    }

    /**
     * @brief Re-chunk this array to @p new_size, whatever its index type.
     * @details For the per-T registry walks (rechunk_all_to), whose keys may be
     *          arrays of different index types: calls the derived rechunk_to()
     *          through the pointer its constructor installed.
     */
    void rechunk_erased(unsigned int new_size)
    {
        rechunk_fn_(*this, new_size);
    }

  protected:
    /**
     * @brief The address under which this array is known outside its own type:
     *        the proxies' ptr_on_array and every <tt>Proxy*\<T\>\::share_info_array</tt> key.
     * @details Always this subobject, never the derived array: the readers cast
     *          it back to ArrayChunkCore, which is only defined from a pointer
     *          that was an ArrayChunkCore* in the first place.
     */
    [[nodiscard]] void* registry_key() noexcept
    {
        return this;
    }
    /// @brief Const overload of registry_key(): the same address.
    [[nodiscard]] const void* registry_key() const noexcept
    {
        return this;
    }

    /// @brief Give this array its own allocator set if it is worth a slab per
    ///        rank; registers allocators, reuse lists and their budget with the
    ///        manager. Call from rebuild_storage() once nb_chunk is known
    ///        (sequential). Idempotent for an array that already has a key.
    void assign_alloc_key()
    {
        if constexpr (build_config::alloc_affinity_min_bytes == 0 || !build_config::chunk_dynamic ||
                      !build_config::use_fast_alloc || !build_config::threaded)
        {
            return;
        }
        if (nb_chunk_ <= 0)
        {
            return;
        }
        // Lazy affinity: only the storage a re-chunk builds takes its own set (an array
        // already keyed keeps its key). A set costs one partly-filled slab per (rank, size),
        // for a benefit -- whole slabs emptied by the NEXT re-chunk -- that only a
        // re-chunked array collects.
        if (alloc_key_ == 0 && !alloc_affinity_detail::g_eager && !alloc_affinity_detail::g_building_rechunk_storage)
        {
            return;
        }
        const std::size_t meta_bytes = static_cast<std::size_t>(nb_chunk_) * sizeof(ChunkType);
        if (meta_bytes < build_config::alloc_affinity_min_bytes)
        {
            return;
        }
        const unsigned int key =
            (alloc_key_ != 0) ? alloc_key_ : next_alloc_key_.fetch_add(1, std::memory_order_relaxed);
        // Keep the key ONLY if the manager built its allocator set: a key with no registered
        // set makes route_omp_dynamic_alloc fall through to the legacy single allocator,
        // which is neither this size's nor this array's.
        alloc_key_ = ManagerType::register_alloc_key(key, size_policy_.size(), nb_chunk_) ? key : 0u;
    }

    /**
     * @brief Size chunk_list to @p n chunks, from the spare a previous
     *        re-chunk left when this is a re-chunk's new storage.
     *
     * @details Every re-chunk builds a new chunk_list and frees the old one; at a
     *          small geometry that is hundreds of MB faulted in, then unmapped (a
     *          TLB shootdown on every CPU). The spare is the previous old storage's
     *          buffer, pages still resident: taken when it holds @p n chunks
     *          without being more than 4x too large. Only for the storage
     *          rechunk_to() builds (quiescent, serial) and only into an empty list,
     *          so no live chunk address ever moves.
     * @param n the chunk count
     */
    void acquire_chunk_list(std::size_t n)
    {
        if (alloc_affinity_detail::g_building_rechunk_storage && alloc_affinity_detail::g_reuse_chunk_list &&
            chunk_list_.capacity() == 0 && spare_chunk_list_.capacity() >= n && spare_chunk_list_.capacity() / 4 <= n)
        {
            chunk_list_.swap(spare_chunk_list_); // spare left empty, without a buffer
        }
        // Only reserves (NoInitAllocator): the caller's init loop (static schedule) is the
        // first touch, page faults and NUMA placement included.
        chunk_list_.resize(n);
    }

    /**
     * @brief Construct chunk @p i of chunk_list in place, value-initialised.
     * @details For the init loops after a growing resize(), so that the chunk
     *          is first written by the thread that initialises it.
     * @param i the chunk index, below chunk_list.size()
     * @return the chunk
     */
    ChunkType& construct_chunk(int i) noexcept
    {
        return *std::construct_at(&chunk_list_[static_cast<std::size_t>(i)]);
    }

    /**
     * @brief Empty chunk_list at the end of a teardown; a re-chunk's old storage
     *        keeps its buffer as the spare for the next re-chunk.
     * @details The larger of the two buffers is kept (more re-chunks fit in
     *          it), the other one freed. At most one spare per chunk type.
     */
    void release_chunk_list()
    {
        chunk_list_.clear();
        if (alloc_affinity_detail::g_releasing_rechunk_storage && alloc_affinity_detail::g_reuse_chunk_list &&
            chunk_list_.capacity() > spare_chunk_list_.capacity())
        {
            chunk_list_.swap(spare_chunk_list_);
            ChunkListType().swap(chunk_list_); // frees the smaller one
        }
    }

    /**
     * @brief Chunk-size policy — the single home for chunk geometry.
     *
     * @details Shared by all containers that derive from ArrayChunkBase
     *          (zero::Array, zero_one_value::Array, zero_any::Array). Geometry use-sites go
     *          through size_policy_.decalage()/modulo()/size().
     */
    [[no_unique_address]] ActiveSizePolicy size_policy_{};

    ChunkListType chunk_list_;  ///< The chunks, in index order.
    int           nb_chunk_{0}; ///< Number of live chunks in `chunk_list`.

    int num_tab_{0}; ///< Identity of this array in the per-type registries.
    /// @brief Number of arrays of this element type created so far (source of
    ///        num_tab), all index types together: num_tab indexes the per-T
    ///        tables of the proxies.
    static inline int nb_tab{0};
    /// @brief Slab-affinity key: 0 = the rank's shared allocator pool; otherwise
    ///        this array's own allocator set (rank x size), so re-chunking or
    ///        destroying it empties whole slabs. Assigned by assign_alloc_key()
    ///        (build_config::alloc_affinity_min_bytes); swapped with the geometry
    ///        at a re-chunk (the old key's slabs die with tmp).
    unsigned int alloc_key_ = 0;
    /// @brief Source of affinity keys, shared by every array of this element type
    ///        whatever its index type (keys of the per-T manager).
    static inline std::atomic<unsigned int> next_alloc_key_{1};

    /// The chunk_list buffer a re-chunk's old storage left (empty, capacity
    /// only), for the next re-chunk's new storage. See acquire_chunk_list().
    static inline ChunkListType spare_chunk_list_{};

    /// @brief The derived array's rechunk_to(), installed by ArrayChunkBase.
    void (*rechunk_fn_)(ArrayChunkCore&, unsigned int) = nullptr;

    /// @brief See set_debug_name().
    std::string debug_name_;
};

/**
 * @brief Check builds only: puts ArrayChunkCore at a nonzero offset in the
 *        array, so the derived address and registry_key() differ.
 * @details A key made from the derived `this` instead of registry_key() would
 *          work by accident while the core sits at offset 0. With this in front
 *          it is a different address, and the lookup fails loudly ("key NOT
 *          FOUND") in the check suite.
 */
class RegistryKeyCanary
{
  private:
    [[maybe_unused]] std::uint64_t registry_key_canary_{0};
};
/// @brief Empty stand-in for RegistryKeyCanary when any_check is off.
struct NoRegistryKeyCanary
{
};
/// @brief ArrayChunkBase's canary base: RegistryKeyCanary in check builds, else the empty stand-in.
using RegistryKeyCanaryBase = std::conditional_t<build_config::any_check, RegistryKeyCanary, NoRegistryKeyCanary>;

} // namespace shrink_array
