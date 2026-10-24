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

#include <cstddef>  // std::size_t
#include <iostream> // std::cout
#include <utility>  // std::cmp_not_equal

#include "chunk/config.h"                     // build_config::threaded, chunk_dynamic
#include "chunk/internal/chunk_size_config.h" // static_chunk_size, static_chunk_shift, static_chunk_mask
#include "chunk/internal/chunk_size_policy.h" // active_chunk_size, chunk_size_error
#include "chunk/internal/func_p2.h"           // is_power_of_two, log2int
#include "chunk/singleton.h"                  // SingletonChk
#include "chunk/all_type_chunk.h"             // AllTypeChunk
#include "chunk/log.h"                        // CHUNK_LOG (the static build's ignored setters)

namespace shrink_array
{

/**
 * @brief Library-wide setup and configuration entry point (all-static, never instantiated).
 *
 * @details This is the single façade an application talks to before using any
 *          chunked array: it publishes the chunk geometry (size / shift / mask),
 *          sizes the allocation blocks, creates the value allocators and the
 *          per-thread reuse structures, and prints the instrumentation report at
 *          the end of the run. A typical program calls at most two of these:
 *          ParamChunk::init_indexes() at startup -- optional, the first array
 *          runs init_indexes(true) when it was not called -- and
 *          ParamChunk::chunk_stats() at exit.
 *
 *          The geometry accessors are compatibility shims: the chunk size is a
 *          compile-time constant (see chunk_size_config.h) so that `i >>
 *          static_chunk_shift` and `i & static_chunk_mask` fold into immediate
 *          shifts/masks on the hot path. The setters change nothing (they warn
 *          on a mismatch); the getters return the constants. Dynamic builds
 *          (CHUNK_DYNAMIC) add a separate, per-array runtime geometry on top;
 *          see ParamChunk::register_chunk_size() and
 *          shrink_array::active_chunk_size().
 *
 * @note Everything here is a sequential-context operation: call it before the
 *       arrays it configures are used in parallel.
 * @see chunk_size_config.h for the compile-time geometry constants
 * @see AllTypeChunk, which holds the per-type managers this class drives
 */
// ===========================================================================
// Build-dependent halves of the façade.
//
// Two of ParamChunk's verbs differ by build in their very NAME, which no `if constexpr` can
// express, so they live in policies selected by a constexpr (as in ManagerChunkInt and
// AllTypeChunk). All static: no CRTP down-cast. The bodies are in param_chunk.cpp, with one
// explicit instantiation of the specialisation this build selects.
// ===========================================================================

/**
 * @brief The thread-model half of the startup: one verb, two bodies.
 * @tparam Enabled build_config::threaded.
 * @tparam AllT    The AllTypeChunk facade. LOAD-BEARING -- see below.
 *
 * @note AllT looks redundant (it is only ever AllTypeChunk) and is not: it keeps the two
 *       bodies TEMPLATES, so only the one this build selects is instantiated. A `template <>`
 *       full specialisation would be compiled eagerly, and the two bodies cannot both compile
 *       (each calls AllTypeChunk verbs only its own AllTypeChunkParallelInterface<Derived, B>
 *       declares). The calls must also stay DEPENDENT on AllT (SingletonChk<AllT>, never
 *       SingletonChk<AllTypeChunk>), or phase-1 lookup rejects the other build's body. Same
 *       reason as the Derived parameter of AllTypeChunkParallelInterface.
 */
template <bool Enabled = build_config::threaded, typename AllT = AllTypeChunk>
struct ParamChunkParallelInterface; // primary left undefined: only the 2 specializations exist

/// @brief OpenMP builds: init_thread_structures() prints the allocator / thread-model banner.
template <typename AllT>
struct ParamChunkParallelInterface<true, AllT>
{
    /**
     * @brief What init_indexes() prints about the thread model (OpenMP build).
     *
     * @details A type is set up when its first array registers its manager
     *          (AllTypeChunk::init_manager, sequential, rank by rank), so
     *          nothing per type is done here: only the allocator banner.
     */
    static void init_thread_structures();
};

/// @brief Serial builds: init_thread_structures() is a no-op.
template <typename AllT>
struct ParamChunkParallelInterface<false, AllT>
{
    /**
     * @brief Serial build: nothing to print, nothing to size.
     * @see ParamChunkParallelInterface<true>::init_thread_structures
     */
    static void init_thread_structures() {}
};

/**
 * @brief The runtime-geometry verb, present only in dynamic builds.
 * @tparam Enabled build_config::chunk_dynamic.
 * @tparam AllT    The AllTypeChunk facade. LOAD-BEARING, as on ParamChunkParallelInterface
 *                 (see the note there).
 *
 * @note The body calls AllTypeChunk::register_chunk_size(), which only
 *       AllTypeChunkDynInterface<Derived, true> declares: keep it a template and the call
 *       dependent on AllT, or the static build fails to compile.
 */
template <bool Enabled = build_config::chunk_dynamic, typename AllT = AllTypeChunk>
struct ParamChunkDynInterface; // primary left undefined: only the 2 specializations exist

/// @brief Dynamic-geometry builds: provides register_chunk_size().
template <typename AllT>
struct ParamChunkDynInterface<true, AllT>
{
    /**
     * @brief Declare a runtime chunk size so arrays of that geometry can allocate.
     *
     * @details Dynamic builds only. Registers @p size with every per-type manager:
     *          in a fast-alloc build this creates and initialises that size's value
     *          allocators, and in every build it gives the size its own slot in the
     *          per-size reuse free-lists.
     *
     * @param size the runtime chunk size to register (a power of two or a multiple of 64)
     *
     * @note Idempotent per size, and to be called in a sequential context before
     *       an array of that size is used in parallel.
     * @warning The free-list slot is not optional even with the standard allocator:
     *          without it every lookup falls back to the single legacy list, and a
     *          buffer of one geometry is popped for a chunk of another (overflow in
     *          init_block's memset).
     * @see shrink_array::register_runtime_chunk_size, the build-agnostic wrapper
     */
    static void register_chunk_size(unsigned int size);
};

/// @brief Fixed-geometry builds: no runtime size to declare.
template <typename AllT>
struct ParamChunkDynInterface<false, AllT>
{
    /**
     * @brief Accepts the call and does nothing (fixed-geometry build).
     *
     * @details Deliberately a silent no-op rather than an absent member, so that
     *          build-agnostic callers (register_runtime_chunk_size()) need no
     *          <tt>\#ifdef CHUNK_DYNAMIC</tt>.
     *
     * @warning So a static build accepts and ignores the call; the compiler will not
     *          say the size cannot be honoured. Guard on build_config::chunk_dynamic
     *          where it matters (dyn_geometry_validation checks this at run time).
     */
    static void register_chunk_size(unsigned int /*size*/) {}
};

/// @brief Library-wide setup and configuration entry point (all-static), inheriting the build-dependent verbs
///        of ParamChunkParallelInterface and ParamChunkDynInterface.
class ParamChunk : public ParamChunkParallelInterface<>, public ParamChunkDynInterface<>
{
  public:
    // The policies read the private #block_size. They are befriended as TEMPLATES: their
    // bodies reach it through a non-dependent expression, access-checked against the
    // template being defined, which a friend instantiation would not cover.
    template <bool, typename>
    friend struct ParamChunkParallelInterface;
    template <bool, typename>
    friend struct ParamChunkDynInterface;

    /// @brief Lower sanity bound for a configured CHUNK_SIZE.
    static constexpr int size_min = 32;
    /**
     * @brief Upper sanity bound for a configured CHUNK_SIZE.
     *
     * @details This is NOT the chunk geometry — use shrink_array::static_chunk_size for
     *          that; it only validates a size coming from an example or a config.
     *          The real hard limit is the per-chunk counter capacity (unsigned
     *          short), enforced by a static_assert in conditional_counter.h, hence
     *          65535.
     */
    static constexpr int size_max = 65535;

    /**
     * @brief Check that a candidate chunk size is valid: a power of two or a
     *        multiple of 64, at most cst_buffer_size -- the rule of the setters
     *        (chunk_size_error).
     *
     * @details The build checks CHUNK_SIZE with a static_assert; this helper is
     *          for a size obtained at run time (command line, config file).
     *          The bounds are size_min / size_max.
     *
     * @param chunk_size the candidate size
     * @return true if it can be a chunk size
     */
    static bool check_chunk_size(int chunk_size) noexcept
    {
        return chunk_size > 0 && !chunk_size_error(static_cast<unsigned int>(chunk_size));
    }

    /// @brief The former power-of-two-only check: chunk sizes may now also be multiples of 64.
    [[deprecated("chunk sizes may be multiples of 64: use check_chunk_size")]] static bool
    check_pow2(int chunk_size) noexcept
    {
        return is_power_of_two(chunk_size);
    }

    /**
     * @brief Print the chunk geometry (size, shift, mask) on stdout.
     *
     * @details In a dynamic build the active runtime geometry may differ from the
     *          compile-time static_chunk_size, so the active size is reported instead —
     *          printing the constant there would be misleading. In a static build
     *          active_chunk_size() == static_chunk_size, and both branches agree.
     */
    static void print_info_chunk()
    {
        if constexpr (build_config::chunk_dynamic)
        {
            const unsigned int cs = active_chunk_size();
            std::cout << "chunk size     : " << cs << " (runtime)\n";
            std::cout << "decalage chunk : " << static_cast<unsigned int>(log2int(static_cast<std::size_t>(cs)))
                      << "\n";
            std::cout << "modulo chunk   : " << (cs - 1u) << "\n";
        }
        else
        {
            std::cout << "chunk size     : " << static_chunk_size << "\n";
            std::cout << "decalage chunk : " << static_chunk_shift << "\n";
            std::cout << "modulo chunk   : " << static_chunk_mask << "\n";
        }
    }

    /**
     * @brief Initialise the library: block size, allocators and per-thread stores.
     *
     * @details Computes the allocation block size from @p big_chunk_block and the
     *          thread count, publishes it via set_block_size(), then sizes the
     *          per-type structures held by the AllTypeChunk singleton: the
     *          compression-candidate lists and, under OpenMP, the per-thread reuse
     *          free-lists and chunk lists. In a fast-alloc build it also creates
     *          and initialises the chunk allocators (one per thread under OpenMP);
     *          with the standard allocator those steps are no-ops. With
     *          -DCHUNK_STATS it resets the instrumentation counters.
     *
     * @param big_chunk_block ask for large allocation blocks (15 MB before the
     *                        per-thread split) instead of the default 2 MB —
     *                        fewer, bigger allocations for memory-heavy runs. The
     *                        result is divided by the thread count and floored at
     *                        a minimum, so it never degenerates on many threads.
     *
     * @note Optional. Called in a sequential context before the first array,
     *       it chooses @p big_chunk_block; otherwise the first array of the
     *       process runs init_indexes(true) itself (AllTypeChunk::ensure_ready),
     *       wherever it is constructed -- a global array included.
     * @warning A chunk must fit in one allocation block; a warning is emitted
     *          otherwise.
     */
    static void init_indexes(bool big_chunk_block);

    /**
     * @brief init_indexes() plus the registration, up front, of a list of
     *        element types.
     *
     * @details The element types need no declaring: the first array of a type
     *          registers its manager (AllTypeChunk::ensure_manager, from the
     *          array's create). This overload registers a list now,
     *          @code
     *          using MyChunkTypes = std::tuple<double, int>;
     *          ParamChunk::init_indexes<MyChunkTypes>(true);
     *          @endcode
     *          which buys having every manager in place before the first
     *          parallel region, at the cost of managers for types the program
     *          may never use. The process-wide setup runs first (the managers
     *          are built with its block size), then the registrations.
     *
     * @tparam Types a std::tuple of the element types to register now
     * @param big_chunk_block forwarded to init_indexes(bool)
     *
     * @note Idempotent: a type already registered, by a list or by an array,
     *       is left alone.
     */
    template <typename Types>
    static void init_indexes(bool big_chunk_block)
    {
        init_indexes(big_chunk_block); // the process-wide setup first: the managers need its block size
        SingletonChk<AllTypeChunk>::get_ref().template initialize<Types>();
    }

    /**
     * @brief Compress every pending candidate chunk now, instead of waiting.
     *
     * @details The fast-block-compression path does not compress on each write: it
     *          queues candidate chunks and sweeps them in batches. This drains that
     *          queue on demand — useful to reach a stable memory footprint at a
     *          known point (a measurement, a checkpoint) rather than at whatever
     *          moment the next sweep happens.
     *
     * @note An empty no-op unless the build has fast block compression
     *       (chunk_config::use_fast_block_compression, OpenMP builds), so call
     *       sites need no guard.
     */
    static void force_compression();

    /**
     * @brief Declare a second compressible value, besides zero.
     *
     * @tparam T the element type concerned
     * @param second_value the additional value to store as a constant
     * @warning Not implemented: deleted, so a use fails at compile time. A
     *          placeholder for the second-constant support that zero_one_value::Array
     *          offers per array (zero_one_value::Array::set_second_value) and zero_any per
     *          chunk.
     */
    template <typename T>
    static void set_second_value(T second_value) = delete ("not implemented: see zero_one_value / zero_any");

    /**
     * @brief Print the instrumentation report accumulated during the run.
     *
     * @details Reduces the per-thread counters of each variant (zero, zero_one_value,
     *          zero_any) and prints one block per variant, preceded by a legend;
     *          variants that saw no activity are skipped. Reports allocations and
     *          reuse, read/write counts with the share of reads served straight
     *          from the constant (what the compression actually saves), compression
     *          attempts and successes, and — in a dynamic build — the re-chunk
     *          counters and the end-of-life chunk-size distribution.
     *
     * @note An empty no-op unless the build defines CHUNK_STATS; the counters
     *       themselves are only maintained in that build.
     */
    static void chunk_stats();

    /// @brief Former name of chunk_stats(), kept for the programs that still spell it.
    [[deprecated("functions are snake_case: use ParamChunk::chunk_stats")]] static void ChunkStats()
    {
        chunk_stats();
    }

  private:
    /**
     * @brief No-op compatibility shim for the former runtime chunk-size setter.
     *
     * @details static_chunk_size is a compile-time constant. Warns when asked for a size
     *          the binary was not built with, rather than ignoring it silently.
     *
     * @param value the chunk size the caller wanted; only compared, never applied
     * @warning To actually change the geometry, rebuild with `-DCHUNK_SIZE=v`
     *          (a power of two or a multiple of 64).
     */
    static void set_chunk_size(int value)
    {
        if (std::cmp_not_equal(value, static_chunk_size))
        {
            CHUNK_LOG(warn, general,
                      "set_chunk_size({}) ignored; built with CHUNK_SIZE={} (rebuild with -DCHUNK_SIZE={})", value,
                      static_chunk_size, value);
        }
    }

    /**
     * @brief Number of elements per chunk, as fixed at build time.
     * @return shrink_array::static_chunk_size (CHUNK_SIZE, default 64)
     * @note In a CHUNK_DYNAMIC build an individual array may have been re-chunked
     *       to another size; this still returns the compile-time value. Use
     *       shrink_array::active_chunk_size() for the geometry currently in effect.
     */
    static constexpr unsigned int get_chunk_size() noexcept
    {
        return static_chunk_size;
    }

    /**
     * @brief No-op compatibility shim: the shift is derived from CHUNK_SIZE.
     *
     * @details Kept so existing call sites keep compiling; the value is
     *          log2(static_chunk_size), computed at compile time.
     *
     * @param value the shift the caller wanted; only compared, never applied
     * @warning Warns when asked for a shift the binary was not built with, as
     *          set_chunk_size() does.
     */
    static void set_decalage_chunk(unsigned int value)
    {
        if (value != static_chunk_shift)
        {
            CHUNK_LOG(warn, general, "set_decalage_chunk({}) ignored; built with CHUNK_SIZE={} (shift {})", value,
                      static_chunk_size, static_chunk_shift);
        }
    }

    /**
     * @brief log2(static_chunk_size): the index shift of a power-of-two size,
     *        meaningless otherwise (chunk = i / static_chunk_size).
     */
    [[deprecated("chunk sizes may be multiples of 64: use i / static_chunk_size")]] static constexpr unsigned int
    get_decalage_chunk() noexcept
    {
        return static_chunk_shift;
    }

    /**
     * @brief No-op compatibility shim: the mask is derived from CHUNK_SIZE.
     *
     * @param value the mask the caller wanted; only compared, never applied
     * @warning Warns rather than ignoring silently. @see set_decalage_chunk
     */
    static void set_modulo_chunk(int value)
    {
        if (std::cmp_not_equal(value, static_chunk_mask))
        {
            CHUNK_LOG(warn, general, "set_modulo_chunk({}) ignored; built with CHUNK_SIZE={} (mask {})", value,
                      static_chunk_size, static_chunk_mask);
        }
    }

    /**
     * @brief Mask extracting the offset of an element inside its chunk.
     * @return static_chunk_size - 1; `offset = i & get_modulo_chunk()`
     */
    static constexpr unsigned int get_modulo_chunk() noexcept
    {
        return static_chunk_mask;
    }

    /**
     * @brief Set the allocation block size, in bytes.
     * @param bz block size; normally computed by init_indexes()
     */
    static void set_block_size(int bz) noexcept
    {
        block_size = bz;
    }

    /// @brief Allocation block size in bytes, as set by init_indexes().
    static int get_block_size() noexcept
    {
        return block_size;
    }

    /// @brief Allocation block size in bytes, shared by all types and threads.
    static int block_size;
};
} // namespace shrink_array