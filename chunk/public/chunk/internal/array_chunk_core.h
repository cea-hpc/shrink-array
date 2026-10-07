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
 * @details ArrayChunk<T, ind_type>, ArrayChunk2<T, ind_type> and
 *          ArrayChunk2A<T, ind_type> are distinct types for every ind_type, but
 *          the machinery around them -- proxies, chunks, managers, the per-T
 *          registries (<tt>Proxy*\<T\>\::share_info_array</tt>) -- is templated on T alone
 *          and reaches the array through a type-erased pointer (the proxy's
 *          ptr_on_array, the registries' keys). Casting that pointer back to
 *          `ArrayChunk*<T, int>` was right as long as no other ind_type existed;
 *          on an `<T, long>` array it read nb_chunk, num_tab and alloc_key_ four
 *          bytes off (m_size had grown in front of them).
 *
 *          ArrayChunkCore<chunk_type> is what that code may see: everything it
 *          reads, laid out identically whatever the index type. The pointer
 *          handed out is the address of THIS subobject (registry_key()), so a
 *          static_cast from the void* back to ArrayChunkCore is well-defined.
 *          The index-typed part -- m_size and the element API -- stays in
 *          ArrayChunkBase<Derived>.
 *
 *          The per-T counters live here for the same reason: nb_tab (the source
 *          of num_tab, which indexes per-T tables) and next_alloc_key_ (keys of
 *          the per-T manager) must be shared by `<T, int>` and `<T, long>`
 *          arrays, or two arrays of one process get the same num_tab.
 *
 * @see array_chunk_base.h, which derives from it
 * @see array_chunk_for.h, which maps a chunk type to it for the proxies
 */

#include <atomic>      // std::atomic (next_alloc_key_)
#include <cstddef>     // std::size_t
#include <cstdint>     // std::uint64_t
#include <cstdlib>     // std::getenv (CHUNK_ALLOC_AFFINITY_EAGER)
#include <type_traits> // std::conditional_t
#include <vector>      // std::vector (chunk_list)

#include "chunk/config.h"                            // build_config::alloc_affinity_min_bytes, build_config::any_check
#include "chunk/chunk_traits.h"                      // ChunkTraits (proxy_type, manager_type)
#include "chunk/internal/chunk_size_policy.h"        // ActiveSizePolicy
#include "chunk/internal/array_chunk_dirty_bitset.h" // DirtyBitsetSupport

namespace shrink_array
{

namespace alloc_affinity_detail
{
/// Eager (an allocator set per large array at creation) or lazy (only the
/// storage a re-chunk builds gets one, see assign_alloc_key()) -- eager by
/// default for Chunk (v1), lazy for Chunk2 / Chunk2A. CHUNK_ALLOC_AFFINITY_EAGER
/// (read once), when set, decides: =0 lazy, anything else eager.
/// v1's memory is mostly data and its rebuild is sliced: with its source in
/// the shared pool, whose slabs mix several arrays' blocks, no slab empties
/// during the first re-chunk (+6.0 GB on 39.3 GB, 64-thread chunk_dyn bench;
/// eager peaks 4.4 GB lower for the same end memory and time). Chunk2 /
/// Chunk2A are not sliced and are metadata-heavy: a lazy start saves them the
/// ~780 MiB of per-array slabs and matches the static build's memory.
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
} // namespace alloc_affinity_detail

/// @brief The index-type-independent part of a chunked array of ChunkType (see the description at the top of
///        this file).
template <typename ChunkType>
class ArrayChunkCore : public DirtyBitsetSupport<ArrayChunkCore<ChunkType>>
{
  public:
    using chunk_type   = ChunkType;                            ///< Chunk type held in chunk_list.
    using proxy_type   = ChunkTraits<ChunkType>::proxy_type;   ///< Proxy type owning the per-T registry.
    using manager_type = ChunkTraits<ChunkType>::manager_type; ///< Per-T manager owning the allocators.

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
        return nb_chunk;
    }

    /**
     * @brief Direct access to one chunk, for bulk range operations.
     *
     * @details NOT part of the container interface -- it exposes the chunked
     *          representation, which the element API deliberately hides. It
     *          exists so the bulk operations in array_chunk_range_ops.h can take
     *          ONE decision per chunk instead of one per element, as free
     *          functions rather than members (so the same call site accepts a
     *          std::vector). Everything else should go through operator[].
     *
     * @param i chunk index; the caller guarantees 0 <= i < nb_chunk
     */
    [[nodiscard]] chunk_type& chunk_ref(std::size_t i) noexcept
    {
        return chunk_list[i];
    }

    /// @brief Const overload of chunk_ref(): the same chunk, read-only.
    [[nodiscard]] const chunk_type& chunk_ref(std::size_t i) const noexcept
    {
        return chunk_list[i];
    }

    /**
     * @brief Re-chunk this array to @p new_size, whatever its index type.
     * @details For the per-T walks over the registry (rechunk_all_to), whose
     *          keys may belong to arrays of different index types: they reach
     *          the derived rechunk_to() through the pointer its constructor
     *          installed.
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
                      !build_config::use_fast_alloc || !build_config::use_openmp)
        {
            return;
        }
        if (nb_chunk <= 0)
        {
            return;
        }
        // Lazy affinity: only the storage a re-chunk builds takes its own set
        // (an array already keyed keeps its key). A set costs one partly-filled
        // 1 MiB slab per (rank, size): ~780 MiB over the 64-thread chunk2a bench's
        // 16 arrays, for a benefit -- whole slabs emptied by the NEXT re-chunk --
        // that only an array that gets re-chunked ever collects.
        if (alloc_key_ == 0 && !alloc_affinity_detail::g_eager && !alloc_affinity_detail::g_building_rechunk_storage)
        {
            return;
        }
        const std::size_t meta_bytes = static_cast<std::size_t>(nb_chunk) * sizeof(chunk_type);
        if (meta_bytes < build_config::alloc_affinity_min_bytes)
        {
            return;
        }
        const unsigned int key =
            (alloc_key_ != 0) ? alloc_key_ : next_alloc_key_.fetch_add(1, std::memory_order_relaxed);
        // Keep the key ONLY if the manager really built its allocator set and
        // lists. Otherwise stay on the shared pool: a key with no registered
        // set makes route_omp_dynamic_alloc fall through to the legacy single
        // allocator, which is neither this size's nor this array's.
        alloc_key_ = manager_type::register_alloc_key(key, size_policy_.size(), nb_chunk) ? key : 0u;
    }

    /**
     * @brief Chunk-size policy — the single home for chunk geometry.
     *
     * @details Shared by all containers that derive from ArrayChunkBase
     *          (ArrayChunk, ArrayChunk2, ArrayChunk2A). Geometry use-sites go
     *          through size_policy_.decalage()/modulo()/size().
     */
    [[no_unique_address]] ActiveSizePolicy size_policy_{};

    std::vector<chunk_type> chunk_list;  ///< The chunks, in index order.
    int                     nb_chunk{0}; ///< Number of live chunks in `chunk_list`.

    int num_tab{0}; ///< Identity of this array in the per-type registries.
    /// @brief Number of arrays of this element type created so far (source of
    ///        num_tab), all index types together: num_tab indexes the per-T
    ///        tables of the proxies.
    static inline int nb_tab{0};
    /// @brief Slab-affinity key: 0 = the rank's shared allocator pool; otherwise
    ///        this array's own allocator set (rank x size), so its buffers share
    ///        slabs with no other array and re-chunking or destroying it empties
    ///        whole slabs. Assigned once by assign_alloc_key() when the array is
    ///        large enough (build_config::alloc_affinity_min_bytes); swapped with
    ///        the geometry at a re-chunk (the new storage lives under the new key,
    ///        the old one dies with tmp and its slabs with it).
    unsigned int alloc_key_ = 0;
    /// @brief Source of affinity keys, shared by every array of this element
    ///        type whatever its index type (the keys name allocator sets of the
    ///        per-T manager).
    static inline std::atomic<unsigned int> next_alloc_key_{1};

    /// @brief The derived array's rechunk_to(), installed by ArrayChunkBase.
    void (*rechunk_fn_)(ArrayChunkCore&, unsigned int) = nullptr;
};

/**
 * @brief Check builds only: puts ArrayChunkCore at a nonzero offset in the
 *        array, so the derived address and registry_key() differ.
 * @details A key made from `this` of the derived array instead of
 *          registry_key() works by accident whenever the core happens to sit at
 *          offset 0 -- until a layout change moves it. With this in front, such
 *          a key is a different address, and the registry lookup that follows
 *          fails loudly ("key NOT FOUND") in the check suite.
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
