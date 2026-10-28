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

#pragma once

// ─────────────────────────────────────────────────────────────────────────────
// Chunk-size policies
// ─────────────────────────────────────────────────────────────────────────────
//
// A chunk's geometry is three related quantities:
//   * size()     — number of elements per chunk (a power of two or a multiple
//                  of 64, is_valid_chunk_size),
//   * chunk_of(i) / offset_of(i) — i / size() and i % size(), the split of an
//                  element index (a shift and a mask for a power of two),
//   * modulo()   — size - 1, the last offset in a chunk; decalage() is
//                  log2(size), meaningful for a power of two only.
//
//   * StaticChunkSize  — compile-time geometry (-DCHUNK_SIZE), every accessor
//     constexpr. The default.
//   * DynamicChunkSize — run-time geometry, changed by set_size() (rechunk_to,
//     set_chunk_size). Selected by CHUNK_DYNAMIC.
//
// ActiveSizePolicy picks one; each array holds it once
// (ArrayChunkCore::size_policy_, [[no_unique_address]]), never per chunk. The
// shared constant buffers are sized by cst_buffer_size (below).

#include <bit>         // std::has_single_bit
#include <cstddef>     // std::size_t (func_p2.h uses size_t unqualified)
#include <cstdint>     // std::uint64_t (DynamicChunkSize's divider), std::uint8_t (ChunkSizeError)
#include <optional>    // std::optional (chunk_size_error)
#include <type_traits> // std::conditional_t, std::make_unsigned_t

#include "chunk/config.h"                     // build_config::chunk_dynamic, build_config::any_check
#include "chunk/internal/func_p2.h"           // log2int (constexpr, also usable at run time)
#include "chunk/internal/chunk_size_config.h" // static_chunk_size / static_chunk_shift / static_chunk_mask
#include "chunk/internal/invariant_check.h"   // CHUNK_INVARIANT_FAILED (an invalid geometry)
#include "chunk/log.h"                        // CHUNK_LOG (warn_chunk_size_refused)

namespace shrink_array
{

// -----------------------------------------------------------------------------
// Runtime "active" chunk size (dynamic builds only). Two layers:
//
//   * g_active_chunk_size : PROCESS-WIDE seed, used at init_indexes and as the
//     fallback when a thread has declared no size.
//   * t_active_chunk_size : PER-THREAD size, set by the write/resize paths from
//     the owning array right before an allocation may happen, so that each
//     allocation goes to the allocator of the right size.
//
// Frees need neither: a slab block stores a back-pointer to its allocator.
inline unsigned int              g_active_chunk_size = static_chunk_size;
inline thread_local unsigned int t_active_chunk_size = 0; // 0 = "unset, use global"
// Set around rechunk_to()'s temporary, which the refill needs born at exactly
// the active size: an array of packed bits is otherwise born larger
// (packed_bit_birth_factor, zero_one_value::Array::create).
inline thread_local bool t_birth_at_active_size = false;

/**
 * @brief Seed the process-wide runtime geometry.
 *
 * @warning Every size in use, the STARTING one included, must be registered:
 *          an unregistered size's buffers go to the legacy store, which
 *          drain_reuse_list_for_size() never visits, so after a re-chunk they
 *          stay resident for good (a sizeable share of the array's footprint).
 *          Idempotent and sequential.
 * @note Defined in param_chunk.cpp: the array headers must not include
 *       param_chunk.h (include cycle).
 */
void register_runtime_chunk_size(unsigned int size);

/**
 * @brief Whether @p size can be a chunk geometry: a power of two, or a
 *        multiple of 64; not 0.
 * @details A multiple of 64 keeps a buffer whole 64-bit words of packed bits
 *          (and whole cache lines for 8-byte elements). rechunk_to,
 *          set_chunk_size and set_active_chunk_size refuse another size;
 *          set_size() stops on it in check builds.
 */
[[nodiscard]] constexpr bool is_valid_chunk_size(unsigned int size) noexcept
{
    return std::has_single_bit(size) || (size != 0u && size % 64u == 0u);
}

/// @brief Why a chunk size is refused (chunk_size_error()).
enum class ChunkSizeError : std::uint8_t
{
    not_pow2_or_x64, ///< neither a power of two nor a multiple of 64 (0 included): is_valid_chunk_size
    exceeds_buffer,  ///< wider than cst_buffer_size, the shared constant buffers: raise CHUNK_DYN_MAX_SIZE
};

/// @brief The reason @p err, in words.
[[nodiscard]] constexpr const char* to_string(ChunkSizeError err) noexcept
{
    switch (err)
    {
        case ChunkSizeError::not_pow2_or_x64:
            return "a chunk size must be a power of two or a multiple of 64";
        case ChunkSizeError::exceeds_buffer:
            return "exceeds cst_buffer_size (raise CHUNK_DYN_MAX_SIZE)";
    }
    return "?";
}

// chunk_size_error() and set_active_chunk_size() are at the end of the file:
// they check the size against cst_buffer_size, defined below.

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

// Per-thread ALLOCATION KEY: affinity key of the array about to materialise a
// chunk (0 = the rank's shared pool). Set together with the size, read by
// get_chunk_allocator() and allocate_chunk_internal().
inline thread_local unsigned int t_alloc_key = 0;
/// @brief Set the calling thread's allocation key.
/// @param key Affinity key (0 = the rank's shared pool).
inline void set_thread_alloc_key(unsigned int key) noexcept
{
    t_alloc_key = key;
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

// Element count of a chunk's value buffer: the thread's size, or static_chunk_size (static).
[[nodiscard]] inline unsigned int alloc_chunk_count() noexcept
{
    if constexpr (build_config::chunk_dynamic)
    {
        return thread_chunk_size();
    }
    else
    {
        return static_chunk_size;
    }
}

/// @brief RAII: pin the calling thread's active chunk size (dynamic builds
///        only) and restore the previous value on scope exit.
/// @details For deferred replays: allocate at the op's FROZEN geometry, not the
///          flusher's, without leaking it into the flusher's in-flight write.
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
/// @details A plain `unsigned int` next to `int num_chunk` / `int num_tab`
///          lets positional mix-ups and stray defaults compile (a
///          `= static_chunk_size` default under-scans runtime-sized chunks). No
///          default constructor, explicit conversion: every count is visible.
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
        return static_chunk_size;
    }
    /// @brief log2(size()): the shift that turns an element index into a chunk index.
    static constexpr unsigned int decalage() noexcept
    {
        return static_chunk_shift;
    }
    /// @brief size() - 1: the mask that turns an element index into an offset in its chunk.
    static constexpr unsigned int modulo() noexcept
    {
        return static_chunk_mask;
    }

    /// @brief The chunk of element @p index (>= 0): index / size(). Unsigned: a signed
    ///        division would add a fix-up for negative values.
    template <typename I>
    [[nodiscard]] static constexpr I chunk_of(I index) noexcept
    {
        return static_cast<I>(static_cast<std::make_unsigned_t<I>>(index) / size());
    }
    /// @brief Element @p index's (>= 0) offset in its chunk: index % size() (a mask
    ///        for a power of two).
    template <typename I>
    [[nodiscard]] static constexpr I offset_of(I index) noexcept
    {
        return static_cast<I>(static_cast<std::make_unsigned_t<I>>(index) % size());
    }

    /// @brief No-op, so generic code compiles under both policies.
    static constexpr void set_size(unsigned int /*size*/) noexcept {}
};

/// @brief Run-time geometry, settable during execution (CHUNK_DYNAMIC builds).
/// @details Starts at the process-wide active size (g_active_chunk_size, by
///          default the compile-time geometry) until set_size() is called.
class DynamicChunkSize
{
  public:
    static constexpr bool is_dynamic = true;

    /// @brief Starts at the process-wide active size (see size_).
    DynamicChunkSize() noexcept
    {
        set_divider(size_);
    }

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

    /// @brief The chunk of element @p index (>= 0): index / size(), without a division.
    ///        32-bit index: one 64x64->128 multiplication (Lemire, Kaser & Kurz
    ///        2019, exact for every 32-bit dividend). 64-bit: Granlund &
    ///        Montgomery's round-down method (libdivide's branch-free path).
    template <typename I>
    [[nodiscard]] I chunk_of(I index) const noexcept;
    /// @brief Element @p index's (>= 0) offset in its chunk: index % size().
    template <typename I>
    [[nodiscard]] I offset_of(I index) const noexcept;

    /// @brief Set a new geometry at run time, and make it the process-wide
    ///        active size. @p size: a power of two or a multiple of 64.
    void set_size(unsigned int size) noexcept;

  private:
    // A new array adopts the process-wide active size (default static_chunk_size).
    unsigned int  size_        = g_active_chunk_size;
    unsigned int  decalage_    = static_cast<unsigned int>(log2int(static_cast<std::size_t>(g_active_chunk_size)));
    unsigned int  modulo_      = g_active_chunk_size - 1u;
    bool          pow2_        = true; ///< size() is a power of two: chunk_of / offset_of shift and mask
    std::uint64_t m32_         = 0;    ///< chunk_of's multiplier, 32-bit index (set_divider)
    std::uint64_t magic_       = 0;    ///< chunk_of's multiplier, 64-bit index (set_divider)
    unsigned int  magic_shift_ = 0;    ///< chunk_of's final shift, 64-bit index: ceil(log2(size)) - 1

    /// @brief chunk_of's magic numbers for @p size (>= 2). 32-bit index:
    ///        m32 = floor((2^64 - 1) / size) + 1. 64-bit index: with
    ///        l = ceil(log2(size)), magic = floor(2^64 * (2^l - size) / size) + 1
    ///        and a final shift of l - 1.
    void set_divider(unsigned int size) noexcept;
};


// -----------------------------------------------------------------------------
// Active policy selector: CHUNK_DYNAMIC switches the whole container dynamic.
// -----------------------------------------------------------------------------
using ActiveSizePolicy = std::conditional_t<build_config::chunk_dynamic, DynamicChunkSize, StaticChunkSize>;

// Extent of the shared constant buffers (m_ptr_cst_val1 / cst_val2): a constant
// chunk is read as buffer[num], so it MUST be >= every runtime chunk size. Not
// the type ceiling (65535): one buffer per DISTINCT cst2 value accumulates
// (cst_buffer_size * sizeof(T) each). Override with
// -DCHUNK_DYN_MAX_SIZE=<power-of-two>; rechunk_to refuses a larger size
// (array_chunk_rechunk_inline.h).
#ifndef CHUNK_DYN_MAX_SIZE
inline constexpr unsigned int chunk_dyn_max_size = 4096u;
#else
inline constexpr unsigned int chunk_dyn_max_size = static_cast<unsigned int>(CHUNK_DYN_MAX_SIZE);
#endif

inline constexpr unsigned int cst_buffer_size = ActiveSizePolicy::is_dynamic ? chunk_dyn_max_size : static_chunk_size;

static_assert(std::has_single_bit(chunk_dyn_max_size), "CHUNK_DYN_MAX_SIZE must be a power of two");
static_assert(!ActiveSizePolicy::is_dynamic || chunk_dyn_max_size >= static_chunk_size,
              "CHUNK_DYN_MAX_SIZE must be at least the starting chunk size: the shared constant buffers are read "
              "up to the chunk size");

/// @brief Why @p size cannot be a geometry of this build, or nothing if it can:
///        is_valid_chunk_size(), and at most cst_buffer_size -- a chunk wider
///        than that reads the shared constant buffers out of bounds.
/// @details The one rule of every setter that takes a size (rechunk_to(),
///          set_chunk_size(), set_active_chunk_size()) and of
///          ParamChunk::check_chunk_size().
[[nodiscard]] constexpr std::optional<ChunkSizeError> chunk_size_error(unsigned int size) noexcept
{
    if (!is_valid_chunk_size(size)) [[unlikely]]
    {
        return ChunkSizeError::not_pow2_or_x64;
    }
    if (size > cst_buffer_size) [[unlikely]]
    {
        return ChunkSizeError::exceeds_buffer;
    }
    return std::nullopt;
}

/// @brief The warning of a setter that refuses a size: "@p who(@p size)
///        refused: <reason>", channel rechunk, in every build (a caller bug,
///        on a cold path).
inline void warn_chunk_size_refused(const char* who, unsigned int size, ChunkSizeError err) noexcept
{
    if (err == ChunkSizeError::exceeds_buffer)
    {
        CHUNK_LOG(warn, rechunk, "{}({}) refused: exceeds cst_buffer_size={} (raise CHUNK_DYN_MAX_SIZE)", who, size,
                  cst_buffer_size);
        return;
    }
    CHUNK_LOG(warn, rechunk, "{}({}) refused: {}", who, size, to_string(err));
}

/// @brief Set the process-wide active chunk size and register it with register_runtime_chunk_size().
/// @param size Chunk size, in elements: a power of two or a multiple of 64, at
///        most cst_buffer_size (chunk_size_error). Refused otherwise, with a
///        message in EVERY build, as rechunk_to() does: the next
///        DynamicChunkSize would divide by 0 (set_divider), and a chunk wider
///        than cst_buffer_size reads the shared constant buffers out of
///        bounds. Geometry kept.
inline void set_active_chunk_size(unsigned int size) noexcept
{
    if (const auto err = chunk_size_error(size)) [[unlikely]]
    {
        warn_chunk_size_refused("set_active_chunk_size", size, *err);
        return;
    }
    g_active_chunk_size = size;
    register_runtime_chunk_size(size);
}

} // namespace shrink_array

#include "chunk/internal/chunk_size_policy_inline.h" // the template bodies
