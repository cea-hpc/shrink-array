#pragma once

/*
 * Copyright 2026 CEA (www.cea.fr)
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

// ─────────────────────────────────────────────────────────────────────────────
// Chunk-size policies
// ─────────────────────────────────────────────────────────────────────────────
//
// A chunk's geometry is three related quantities:
//   * size()     — number of elements per chunk (a power of two),
//   * decalage() — log2(size), used for `flat_index >> decalage()` to get the
//                  chunk number,
//   * modulo()   — size - 1, used for `flat_index & modulo()` to get the offset
//                  inside the chunk.
//
// Two policies provide this interface, so one container implementation can be
// instantiated in two flavours without duplicating its logic:
//
//   * StaticChunkSize  — compile-time geometry, fixed by CHUNK_SIZE /
//     -DCHUNK_SIZE: every accessor is constexpr, so `i >> decalage()` /
//     `i & modulo()` fold into immediate shifts/masks on the hot path and
//     `std::array<T, size()>` remains legal. The default.
//
//   * DynamicChunkSize — run-time geometry, changeable during execution via
//     set_size() (rechunk_to, set_chunk_size). Selected by CHUNK_DYNAMIC.
//
// ActiveSizePolicy (below) picks one; each array holds it as a member
// (ArrayChunkCore::size_policy_, [[no_unique_address]]), so StaticChunkSize
// costs zero bytes and DynamicChunkSize lives once per array, never per chunk.
// The shared constant buffers, which need a compile-time extent, are sized by
// cst_buffer_size: the chunk size in static builds, chunk_dyn_max_size in
// dynamic ones.

#include <cstddef>     // std::size_t (func_p2.h uses size_t unqualified)
#include <type_traits> // std::conditional_t

#include "chunk/config.h"                     // build_config::chunk_dynamic, build_config::any_check
#include "chunk/internal/func_p2.h"           // log2int (constexpr, also usable at run time)
#include "chunk/internal/chunk_size_config.h" // __chunk_size / __decalage_chunk / __modulo_chunk
#include "chunk/internal/invariant_check.h"   // INVARIANT_FAILED (non-power-of-two geometry)

namespace shrink_array
{

// -----------------------------------------------------------------------------
// Runtime "active" chunk size (dynamic builds only). Two layers:
//
//   * g_active_chunk_size : a PROCESS-WIDE seed/default. It is used to prime the
//     allocator pool at init_indexes and as the fallback when a thread has not
//     (yet) declared which size it is operating on.
//
//   * t_active_chunk_size : a PER-THREAD active size. The write/resize paths set
//     it (from the owning array's chunk_size()) right before an allocation may
//     happen, so that — when several arrays with different chunk sizes coexist —
//     each allocation is routed to the allocator for the correct size. Because
//     it is thread-local, worker threads in a parallel region each carry the
//     size of the array they are currently operating on.
//
// The free path needs none of this: freed blocks are address-routed back to
// their owning allocator (each slab block stores a back-pointer to it), so a
// free works regardless of which size the freeing thread currently has active.
inline unsigned int              g_active_chunk_size = __chunk_size;
inline thread_local unsigned int t_active_chunk_size = 0; // 0 = "unset, use global"
// Set around rechunk_to()'s temporary, which the refill needs born at exactly
// the active size: an array of packed bits is otherwise born larger
// (packed_bit_birth_factor, ArrayChunk2::create).
inline thread_local bool t_birth_at_active_size = false;

/**
 * @brief Seed the process-wide runtime geometry.
 *
 * @warning Setting the size is NOT enough: the per-size reuse free-list needs a
 *          slot for @p size, created by register_runtime_chunk_size(). Only
 *          rechunk_to() and ArrayChunkBase::set_chunk_size() ever called it, so
 *          the STARTING geometry -- the one every array is born with -- had no
 *          entry, and get_list_block_free_thr_for_size() fell back to the legacy
 *          store for all of its buffers. drain_reuse_list_for_size(old_size)
 *          never visits that store, so after a re-chunk the old geometry's
 *          buffers stayed resident forever: measured 99482 blocks and a 45 MB
 *          gap (64 then re-chunk to 256 = 169 MB, versus 123 MB for an array
 *          born at 256).
 *
 *          Callers cannot register it themselves without knowing this rule, so
 *          the registration belongs here. It is idempotent and sequential.
 * @note Defined in param_chunk.cpp: the array headers must not include
 *       param_chunk.h (include cycle).
 */
void register_runtime_chunk_size(unsigned int size);

/**
 * @brief Whether @p size can be a chunk geometry: a power of two, not 0.
 * @details The index split is a shift and a mask (ChunkSizePolicy::set_size):
 *          for 48, log2int gives a shift of 5 and the mask is 47 -- two
 *          elements on one slot, and a re-chunk refill that frees a buffer
 *          still in use. The public entry points (rechunk_to, set_chunk_size)
 *          refuse such a size; set_size() and set_active_chunk_size() stop
 *          on it in check builds.
 */
[[nodiscard]] constexpr bool is_valid_chunk_size(unsigned int size) noexcept
{
    return size != 0u && (size & (size - 1u)) == 0u;
}

/// @brief Set the process-wide active chunk size and register it with register_runtime_chunk_size().
/// @param size Chunk size, in elements: a power of two (stops a check build otherwise).
inline void set_active_chunk_size(unsigned int size) noexcept
{
    if constexpr (build_config::any_check)
    {
        if (!is_valid_chunk_size(size)) [[unlikely]]
        {
            INVARIANT_FAILED("set_active_chunk_size({}): a chunk size must be a power of two "
                             "(the index split is a shift and a mask)",
                             size);
        }
    }
    g_active_chunk_size = size;
    register_runtime_chunk_size(size);
}
/// @brief The process-wide active chunk size.
[[nodiscard]] inline unsigned int active_chunk_size() noexcept
{
    return g_active_chunk_size;
}

// Per-thread active size: set by array ops before allocating; read by the
// allocator routing. Falls back to the process-wide seed when unset.
inline void set_thread_chunk_size(unsigned int size) noexcept
{
    t_active_chunk_size = size;
}

// Per-thread active ALLOCATION KEY: the affinity key of the array whose chunk
// is about to be materialised (0 = the rank's shared pool). Set by the proxy
// next to the geometry, read by get_chunk_allocator() / the reuse-list lookup
// in allocate_chunk_internal(). Same implicit-parameter pattern as the size;
// they are two properties of the same array and travel together.
inline thread_local unsigned int t_alloc_key = 0;
/// @brief Set the calling thread's allocation key.
/// @param k Affinity key (0 = the rank's shared pool).
inline void set_thread_alloc_key(unsigned int k) noexcept
{
    t_alloc_key = k;
}
/// @brief The calling thread's allocation key (0 = the rank's shared pool).
[[nodiscard]] inline unsigned int thread_alloc_key() noexcept
{
    return t_alloc_key;
}

/// @brief The calling thread's active chunk size, or the process-wide one when
///        the thread has not set it.
[[nodiscard]] inline unsigned int thread_chunk_size() noexcept
{
    return t_active_chunk_size != 0u ? t_active_chunk_size : g_active_chunk_size;
}

// The element count to allocate for a real chunk's value buffer. In dynamic
// builds this is the runtime active size; in static builds it folds to the
// compile-time __chunk_size (so static codegen is unchanged).
[[nodiscard]] inline unsigned int alloc_chunk_count() noexcept
{
    if constexpr (build_config::chunk_dynamic)
    {
        return thread_chunk_size();
    }
    else
    {
        return __chunk_size;
    }
}

/// @brief RAII: pin the calling thread's active chunk size (dynamic builds
///        only) and restore the previous value on scope exit.
/// @details Used on deferred-replay paths where create_value /
///          allocate_chunk_internal must size a buffer at the op's FROZEN
///          geometry (proxy.chunk_size()), not the flushing thread's ambient --
///          and must not leak that size into the flusher's own in-flight write.
///          No-op in static builds (the geometry is compile-time there).
class DynChunkSizeGuard
{
  public:
    /// @brief Save the thread's active chunk size and, in dynamic builds, set it to @p size.
    /// @param size Chunk size to pin for the scope.
    explicit DynChunkSizeGuard([[maybe_unused]] unsigned int size) noexcept
    {
        if constexpr (build_config::chunk_dynamic)
        {
            t_active_chunk_size = size;
        }
    }
    /// @brief In dynamic builds, restore the saved thread chunk size.
    ~DynChunkSizeGuard() noexcept
    {
        if constexpr (build_config::chunk_dynamic)
        {
            t_active_chunk_size = saved_;
        }
    }
    DynChunkSizeGuard(const DynChunkSizeGuard&)            = delete;
    DynChunkSizeGuard& operator=(const DynChunkSizeGuard&) = delete;
    DynChunkSizeGuard(DynChunkSizeGuard&&)                 = delete;
    DynChunkSizeGuard& operator=(DynChunkSizeGuard&&)      = delete;

  private:
    [[maybe_unused]] unsigned int saved_ = t_active_chunk_size;
};

/// @brief Strong type for a chunk's element count as it travels through the
///        compression pipeline.
/// @details A plain `unsigned int size` parameter sits in the same signatures
///          as `int num_chunk` / `int num_tab`: positional mix-ups and stray
///          defaults compile silently — exactly the failure that let a
///          `= __chunk_size` default under-scan runtime-sized chunks and
///          corrupt data in CHUNK_DYNAMIC builds. The deleted default
///          constructor and explicit conversion make every creation of a count
///          a visible, greppable act; count() is the only way out.
class ChunkElems
{
  public:
    ChunkElems() = delete;
    /// @brief Wrap an element count.
    /// @param count The element count.
    constexpr explicit ChunkElems(unsigned int count) noexcept : n_(count) {}

    /// @brief The element count.
    [[nodiscard]] constexpr unsigned int count() const noexcept
    {
        return n_;
    }

  private:
    unsigned int n_;
};

/// @brief Compile-time geometry (the default): every accessor is constexpr.
struct StaticChunkSize
{
    static constexpr bool is_dynamic = false;

    /// @brief Elements per chunk (the compile-time constant).
    static constexpr unsigned int size() noexcept
    {
        return __chunk_size;
    }
    /// @brief log2(size()): the shift that turns an element index into a chunk index.
    static constexpr unsigned int decalage() noexcept
    {
        return __decalage_chunk;
    }
    /// @brief size() - 1: the mask that turns an element index into an offset in its chunk.
    static constexpr unsigned int modulo() noexcept
    {
        return __modulo_chunk;
    }

    /// @brief No-op in static mode: geometry is fixed at compile time. Present
    ///        so generic code (e.g. ArrayChunkBase::set_chunk_size) compiles
    ///        under both policies.
    static constexpr void set_size(unsigned int) noexcept {}
};

/// @brief Run-time geometry, settable during execution (CHUNK_DYNAMIC builds).
/// @details Starts at the process-wide active size (g_active_chunk_size, by
///          default the compile-time geometry) until set_size() is called.
class DynamicChunkSize
{
  public:
    static constexpr bool is_dynamic = true;

    /// @brief Elements per chunk.
    [[nodiscard]] unsigned int size() const noexcept
    {
        return size_;
    }
    /// @brief log2(size()): the shift that turns an element index into a chunk index.
    [[nodiscard]] unsigned int decalage() const noexcept
    {
        return decalage_;
    }
    /// @brief size() - 1: the mask that turns an element index into an offset in its chunk.
    [[nodiscard]] unsigned int modulo() const noexcept
    {
        return modulo_;
    }

    /// @brief Set a new geometry at run time, and make it the process-wide
    ///        active size. @p size must be a power of two.
    void set_size(unsigned int size) noexcept
    {
        // Last net: rechunk_to() and set_chunk_size() refuse such a size
        // before reaching here (is_valid_chunk_size).
        if constexpr (build_config::any_check)
        {
            if (!is_valid_chunk_size(size)) [[unlikely]]
            {
                INVARIANT_FAILED("ChunkSizePolicy::set_size({}): not a power of two -- shift {} with mask {} "
                                 "would send two elements to one slot",
                                 size, log2int(static_cast<std::size_t>(size)), size - 1u);
            }
        }
        size_               = size;
        decalage_           = static_cast<unsigned int>(log2int(static_cast<std::size_t>(size)));
        modulo_             = size - 1u;
        g_active_chunk_size = size; // keep the allocator-facing runtime size in sync
    }

  private:
    // A freshly constructed array adopts the process-wide active size
    // (g_active_chunk_size), which the program seeds via set_active_chunk_size
    // before priming the allocators. This way arrays built after that seed have
    // the correct geometry immediately, without a separate set_chunk_size call.
    // g_active_chunk_size defaults to __chunk_size, so a dynamic build that never
    // seeds it behaves exactly like the static geometry.
    unsigned int size_     = g_active_chunk_size;
    unsigned int decalage_ = static_cast<unsigned int>(log2int(static_cast<std::size_t>(g_active_chunk_size)));
    unsigned int modulo_   = g_active_chunk_size - 1u;
};


// -----------------------------------------------------------------------------
// Active policy selector. Define CHUNK_DYNAMIC to build the runtime-geometry
// flavour; otherwise the compile-time StaticChunkSize is used (default, zero
// cost). This is the single switch that turns the whole container dynamic.
// -----------------------------------------------------------------------------
using ActiveSizePolicy = std::conditional_t<build_config::chunk_dynamic, DynamicChunkSize, StaticChunkSize>;

// Size of the shared constant-value buffers (m_ptr_cst_val1 / cst_val2). In
// static mode they only ever need __chunk_size elements.
// Dynamic-mode ceiling for the shared constant buffers. A compressed (constant)
// chunk is read as buffer[num] for num in [0, N), so this MUST be >= the largest
// runtime chunk size ever produced by a re-chunk. It is NOT the absolute type
// ceiling (65535): that wastes cst_buffer_size*sizeof(T) per DISTINCT cst2 value
// (e.g. 524 KB/value for double at 65535) and, since those buffers accumulate,
// blows up memory. Cap it at the realistic max instead. Override with
// -DCHUNK_DYN_MAX_SIZE=<power-of-two> if a run re-chunks above the default.
// SAFETY: rechunk_to refuses any new_size > cst_buffer_size (array_chunk_rechunk_inline.h),
// so an out-of-range geometry is rejected up-front, never read out of bounds.
#ifndef CHUNK_DYN_MAX_SIZE
inline constexpr unsigned int chunk_dyn_max_size = 4096u;
#else
inline constexpr unsigned int chunk_dyn_max_size = static_cast<unsigned int>(CHUNK_DYN_MAX_SIZE);
#endif

inline constexpr unsigned int cst_buffer_size = ActiveSizePolicy::is_dynamic ? chunk_dyn_max_size : __chunk_size;

static_assert(chunk_dyn_max_size != 0u && (chunk_dyn_max_size & (chunk_dyn_max_size - 1u)) == 0u,
              "CHUNK_DYN_MAX_SIZE must be a power of two");
static_assert(!ActiveSizePolicy::is_dynamic || chunk_dyn_max_size >= __chunk_size,
              "CHUNK_DYN_MAX_SIZE must be at least the starting chunk size: the shared constant buffers are read "
              "up to the chunk size");

} // namespace shrink_array
