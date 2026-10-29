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
 * @details zero::Array*<T, IntType> are distinct types for every IntType, but
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
#include <memory>      // std::construct_at (construct_chunk)
#include <string>      // std::string (debug_name_)
#include <type_traits> // std::conditional_t
#include <utility>     // std::move (set_debug_name)
#include <vector>      // std::vector (chunk_list_)

#include "chunk/config.h"                            // build_config::alloc_affinity_min_bytes, build_config::any_check
#include "chunk/chunk_traits.h"                      // ChunkTraits (ProxyType, ManagerType)
#include "chunk/internal/chunk_size_policy.h"        // ActiveSizePolicy
#include "chunk/internal/array_chunk_dirty_bitset.h" // DirtyBitsetSupport
#include "chunk/internal/array_chunk_for.h"          // ArrayKey (registry_key)
#include "chunk/internal/parallel_storage_init.h"    // NoInitAllocator (chunk_list_)

namespace shrink_array
{

namespace alloc_affinity_detail
{
/// Eager (an allocator set per large array at creation) or lazy (only the
/// storage a re-chunk builds gets one, see assign_alloc_key()) -- eager by
/// default for zero, lazy for zero_one_value / zero_any, fixed at build time.
/// zero's memory is mostly data and its rebuild is sliced: from the shared pool,
/// whose slabs mix several arrays' blocks, no slab empties during the first
/// re-chunk, so eager lowers its peak. zero_one_value / zero_any are not sliced and
/// are metadata-heavy: lazy saves them the per-array slabs. Measured on Milan
/// (shrink_bench int64 dyn, 64 threads, three runs each, October 2026): zero lazy
/// costs +9.8% of time, +14.6% at the peak and +11.8% at the end; zero_any eager
/// +7.4% at the peak; zero_one_value is the same either way.
inline constexpr bool g_eager = chunk_config::use_zero;

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
} // namespace alloc_affinity_detail

// chunk_list_'s allocator is NoInitAllocator (parallel_storage_init.h): resize() only
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
    using ChunkType     = ChunkT;                                             ///< Chunk type held in chunk_list_.
    using ChunkListType = std::vector<ChunkType, NoInitAllocator<ChunkType>>; ///< chunk_list_'s type.
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
     * @param index chunk index; the caller guarantees 0 <= index < nb_chunk
     */
    [[nodiscard]] ChunkType& chunk_ref(std::size_t index) noexcept
    {
        return chunk_list_[index];
    }

    /// @brief Const overload of chunk_ref(): the same chunk, read-only.
    [[nodiscard]] const ChunkType& chunk_ref(std::size_t index) const noexcept
    {
        return chunk_list_[index];
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
     * @brief The identity under which this array is known outside its own type
     *        (ArrayKey): the proxies' ptr_on_array_ and every
     *        <tt>Proxy*\<T\>\::share_info_array</tt> key.
     * @details Always this subobject, never the derived array: the same core
     *          for every index type of the array.
     */
    [[nodiscard]] ArrayKey<ChunkT> registry_key() const noexcept
    {
        return this;
    }

    /// @brief Give this array its own allocator set if it is worth a slab per
    ///        rank; registers allocators, reuse lists and their budget with the
    ///        manager. Call from rebuild_storage() once nb_chunk is known
    ///        (sequential). Idempotent for an array that already has a key.
    void assign_alloc_key();

    /**
     * @brief Size chunk_list_ to @p count chunks (reserve only: see construct_chunk()).
     *
     * @details Always a buffer of its own: a re-chunk's new storage no longer takes
     *          the buffer the previous re-chunk's old storage left. That spare (the
     *          former CHUNK_RECHUNK_REUSE_LIST) outlived the re-chunk, one per chunk
     *          type, and a coarsening -- which needs a SMALLER list -- took a buffer
     *          up to four times too large and kept it resident whole: the old
     *          geometry's chunk_list_ never came back.
     * @param count the chunk count
     */
    void acquire_chunk_list(std::size_t count);

    /**
     * @brief Construct chunk @p index of chunk_list_ in place, value-initialised.
     * @details For the init loops after a growing resize(), so that the chunk
     *          is first written by the thread that initialises it.
     * @param index the chunk index, below chunk_list_.size()
     * @return the chunk
     */
    ChunkType& construct_chunk(int index) noexcept
    {
        return *std::construct_at(&chunk_list_[static_cast<std::size_t>(index)]);
    }

    /**
     * @brief Empty chunk_list_ at the end of a teardown (its chunks destroyed;
     *        the buffer goes with the vector).
     */
    void release_chunk_list();

    /**
     * @brief Chunk-size policy — the single home for chunk geometry.
     *
     * @details Shared by all containers that derive from ArrayChunkBase
     *          (zero::Array, zero_one_value::Array, zero_any::Array). Geometry use-sites go
     *          through size_policy_.decalage()/modulo()/size().
     */
    [[no_unique_address]] ActiveSizePolicy size_policy_{};

    ChunkListType chunk_list_;  ///< The chunks, in index order.
    int           nb_chunk_{0}; ///< Number of live chunks in `chunk_list_`.

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

#include "chunk/internal/array_chunk_core_inline.h" // the template bodies
