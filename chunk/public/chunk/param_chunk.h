//
// Created by ballereaup on 03/05/24.
//

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

#include <cstddef> // std::size_t
#include <iostream>
#include <tuple>   // std::tuple (the inf_array entries)
#include <utility> // std::cmp_not_equal

#include "chunk/chunk_define.h"
#include "chunk/internal/chunk_size_config.h"
#include "chunk/internal/func_p2.h"
#include "chunk/singleton.h"
#include "chunk/all_type_chunk.h"

namespace shrink_array
{

/**
 * @brief Library-wide setup and configuration entry point (all-static, never instantiated).
 *
 * @details This is the single façade an application talks to before using any
 *          chunked array: it publishes the chunk geometry (size / shift / mask),
 *          sizes the allocation blocks, creates the value allocators and the
 *          per-thread reuse structures, and prints the instrumentation report at
 *          the end of the run. A typical program calls exactly two of these:
 *          @ref init_indexes at startup and @ref ChunkStats at exit.
 *
 *          The geometry accessors are historical. The chunk size used to be three
 *          runtime globals set once through set_chunk_size(); it is now a
 *          compile-time constant (see chunk_size_config.h) so that `i >>
 *          __decalage_chunk` and `i & __modulo_chunk` fold into immediate
 *          shifts/masks on the hot path. The setters are therefore kept only as
 *          compatibility shims — they no longer change anything — while the
 *          getters return the constants. Dynamic builds (CHUNK_DYNAMIC) add a
 *          separate, per-array runtime geometry on top; see @ref register_chunk_size
 *          and shrink_array::active_chunk_size().
 *
 * @note Everything here is a sequential-context operation: call it before the
 *       arrays it configures are used in parallel.
 * @see chunk_size_config.h for the compile-time geometry constants
 * @see AllTypeChunk, which holds the per-type managers this class drives
 */
// ===========================================================================
// Build-dependent halves of the façade.
//
// Two of ParamChunk's verbs differ by build in their very NAME, which no
// `if constexpr` can express -- so they live in policies selected by a
// constexpr, as ManagerChunkInt and AllTypeChunk already do. ParamChunk
// inherits both; everything here is static, so there is no CRTP down-cast to
// arrange, only a name to bring into scope.
//
// The bodies stay in param_chunk.cpp, where one explicit instantiation per
// policy names exactly the specialisation this build selected.
// ===========================================================================

/**
 * @brief The thread-model half of the startup: one verb, two bodies.
 * @tparam Enabled build_config::use_openmp.
 * @tparam AllT    The AllTypeChunk facade. LOAD-BEARING -- see below.
 *
 * @note AllT looks redundant (it is only ever AllTypeChunk) and is not. It is
 *       what keeps the two bodies TEMPLATES, so only the one this build selects
 *       is ever instantiated -- param_chunk.cpp instantiates exactly
 *       ParamChunkParallelInterface<build_config::use_openmp, AllTypeChunk> and
 *       the other body is never compiled. That is what replaced the #ifdef
 *       _OPENMP these bodies used to sit under.
 *
 *       A plain `template <>` full specialisation would NOT do: its members are
 *       ordinary functions, compiled eagerly, and the two bodies cannot both
 *       compile (the OpenMP one calls AllTypeChunk verbs -- the per-rank
 *       resizes, init_allocator_thr -- that only AllTypeChunkParallelInterface
 *       <Derived, true> declares; the serial one calls init_allocator, which
 *       only <Derived, false> declares).
 *
 *       And a merely DUMMY type parameter would not do either: the calls have
 *       to be DEPENDENT on it, or phase-1 lookup resolves them against the
 *       concrete AllTypeChunk at definition time and rejects the body of the
 *       build we are not in. Hence the bodies go through SingletonChk<AllT>,
 *       not SingletonChk<AllTypeChunk>. Same trick, same reason, as the Derived
 *       parameter of AllTypeChunkParallelInterface.
 */
template <bool Enabled = build_config::use_openmp, typename AllT = AllTypeChunk>
struct ParamChunkParallelInterface; // primary left undefined: only the 2 specializations exist

template <typename AllT>
struct ParamChunkParallelInterface<true, AllT>
{
    /**
     * @brief What init_indexes() prints about the thread model (OpenMP build).
     *
     * @details The per-rank structures and allocators of a type used to be
     *          sized and primed here, for every registered type, from inside
     *          a `#pragma omp parallel` region. A type is now set up when its
     *          first array registers its manager (AllTypeChunk::init_manager,
     *          sequential, rank by rank), so nothing per type is left at this
     *          point; the allocator banner remains.
     */
    static void init_thread_structures();
};

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
 * @tparam AllT    The AllTypeChunk facade. LOAD-BEARING, exactly as it is on
 *                 ParamChunkParallelInterface above -- see the note there for
 *                 the full rationale.
 *
 * @note In short: AllT is what keeps the body a TEMPLATE, so a static build
 *       never instantiates it and never compiles it. It has to be so, because
 *       the body calls AllTypeChunk::register_chunk_size(), which only
 *       AllTypeChunkDynInterface<Derived, true> declares -- a `template <>`
 *       full specialisation would be compiled eagerly and fail there. And the
 *       call must stay DEPENDENT on AllT (SingletonChk<AllT>, never
 *       SingletonChk<AllTypeChunk>), or phase-1 lookup resolves it against the
 *       concrete type at definition time and we are back to needing the
 *       #ifdef CHUNK_DYNAMIC this replaced.
 */
template <bool Enabled = build_config::chunk_dynamic, typename AllT = AllTypeChunk>
struct ParamChunkDynInterface; // primary left undefined: only the 2 specializations exist

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
     * @param size the runtime chunk size to register (power of two)
     *
     * @note Idempotent per size, and to be called in a sequential context before
     *       an array of that size is used in parallel.
     * @warning The free-list slot is not optional even with the standard allocator:
     *          without it the by-size map stays empty, every lookup falls back to
     *          the single legacy list, and buffers of different geometries end up
     *          sharing it — a size-A buffer parked there is later popped for a
     *          size-B chunk and overflows in init_block's memset.
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
     * @details Deliberately a silent no-op rather than an absent member. It is
     *          what lets register_runtime_chunk_size() -- and any other
     *          build-agnostic caller -- call ParamChunk::register_chunk_size()
     *          with no `#ifdef CHUNK_DYNAMIC` around it: the call compiles in
     *          every build and costs nothing where there is no runtime geometry
     *          to declare.
     *
     * @warning The trade-off is deliberate and worth knowing: calling
     *          ParamChunk::register_chunk_size() in a static build USED TO BE a
     *          compile error, and is now accepted and ignored. A caller who
     *          expects a size to be honoured will not be told by the compiler
     *          that this build cannot honour it. Guard on
     *          build_config::chunk_dynamic where that distinction matters --
     *          and see dyn_geometry_validation, which catches at run time an
     *          array built at a size no one registered.
     */
    static void register_chunk_size(unsigned int /*size*/) {}
};

class ParamChunk : public ParamChunkParallelInterface<>, public ParamChunkDynInterface<>
{
    // The policies read #block_size, which is private and stays that way: it is
    // an implementation detail of the façade, not part of its surface.
    // Both policies are befriended as TEMPLATES rather than as their
    // instantiations: their bodies reach block_size through a non-dependent
    // expression, which is access-checked against the template being defined --
    // naming only <true, AllTypeChunk> would not cover it.
    template <bool, typename>
    friend struct ParamChunkParallelInterface;
    template <bool, typename>
    friend struct ParamChunkDynInterface;

  public:
    /**
     * @brief No-op compatibility shim for the former runtime chunk-size setter.
     *
     * @details __chunk_size is a compile-time constant now, so the size cannot be
     *          changed at run time. Rather than silently ignoring the call, this
     *          warns when a caller asks for a size the binary was not built with —
     *          otherwise the program would keep running on a geometry it did not
     *          request.
     *
     * @param v the chunk size the caller wanted; only compared, never applied
     * @warning To actually change the geometry, rebuild with `-DCHUNK_SIZE=v`
     *          (must be a power of two).
     */
    static void set_chunk_size(int v)
    {
        if (std::cmp_not_equal(v, __chunk_size))
            std::cerr << "Warning: set_chunk_size(" << v << ") ignored; built with CHUNK_SIZE=" << __chunk_size
                      << " (rebuild with -DCHUNK_SIZE=" << v << ")\n";
    }

    /**
     * @brief Number of elements per chunk, as fixed at build time.
     * @return shrink_array::__chunk_size (CHUNK_SIZE, default 64)
     * @note In a CHUNK_DYNAMIC build an individual array may have been re-chunked
     *       to another size; this still returns the compile-time value. Use
     *       shrink_array::active_chunk_size() for the geometry currently in effect.
     */
    static constexpr unsigned int get_chunk_size() noexcept
    {
        return __chunk_size;
    }

    /**
     * @brief No-op compatibility shim: the shift is derived from CHUNK_SIZE.
     *
     * @details Kept so existing call sites keep compiling; the value is
     *          log2(__chunk_size), computed at compile time.
     *
     * @param v the shift the caller wanted; only compared, never applied
     * @warning Warns when asked for a shift the binary was not built with,
     *          rather than accepting it in silence -- the same contract as
     *          set_chunk_size(). A caller that got no answer here would have
     *          gone on believing it had changed the geometry.
     */
    static void set_decalage_chunk(unsigned int v)
    {
        if (v != __decalage_chunk)
            std::cerr << "Warning: set_decalage_chunk(" << v << ") ignored; built with CHUNK_SIZE=" << __chunk_size
                      << " (shift " << __decalage_chunk << ")\n";
    }

    /**
     * @brief Right-shift turning a global element index into a chunk index.
     * @return log2(__chunk_size); `chunk = i >> get_decalage_chunk()`
     */
    static constexpr unsigned int get_decalage_chunk() noexcept
    {
        return __decalage_chunk;
    }

    /**
     * @brief Check that a candidate chunk size is a power of two.
     *
     * @details The whole index split relies on shift/mask rather than div/mod, so
     *          a non-power-of-two size would silently corrupt the addressing. The
     *          build enforces this with a static_assert; this helper is for
     *          validating a size obtained at run time (command line, config file).
     *
     * @param chunk_size the candidate size
     * @return true if it is a power of two
     */
    static bool check_pow2(int chunk_size) noexcept
    {
        return is_power_of_tow(chunk_size);
    }

    /**
     * @brief No-op compatibility shim: the mask is derived from CHUNK_SIZE.
     *
     * @param v the mask the caller wanted; only compared, never applied
     * @warning Warns rather than ignoring silently. @see set_decalage_chunk
     */
    static void set_modulo_chunk(int v)
    {
        if (std::cmp_not_equal(v, __modulo_chunk))
            std::cerr << "Warning: set_modulo_chunk(" << v << ") ignored; built with CHUNK_SIZE=" << __chunk_size
                      << " (mask " << __modulo_chunk << ")\n";
    }

    /**
     * @brief Mask extracting the offset of an element inside its chunk.
     * @return __chunk_size - 1; `offset = i & get_modulo_chunk()`
     */
    static constexpr unsigned int get_modulo_chunk() noexcept
    {
        return __modulo_chunk;
    }

    /**
     * @brief Print the chunk geometry (size, shift, mask) on stdout.
     *
     * @details In a dynamic build the active runtime geometry may differ from the
     *          compile-time __chunk_size, so the active size is reported instead —
     *          printing the constant there would be misleading. In a static build
     *          active_chunk_size() == __chunk_size, and both branches agree.
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
            std::cout << "chunk size     : " << __chunk_size << "\n";
            std::cout << "decalage chunk : " << __decalage_chunk << "\n";
            std::cout << "modulo chunk   : " << __modulo_chunk << "\n";
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
     * @note Must be called once, in a sequential context, before any chunked array
     *       is used.
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
        SingletonChk<AllTypeChunk>::getRef().template initialize<Types>();
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
     * @note An empty no-op unless the build defines
     *       HAVE_EMBEDDED_FAST_BLOCK_COMPRESSION, so call sites need no guard.
     */
    static void force_compression();


    /**
     * @brief Declare a second compressible value, besides zero.
     *
     * @tparam T the element type concerned
     * @param second_value the additional value to store as a constant
     * @warning Not implemented: deleted, so a use fails at compile time (it
     *          used to be declared only, and failed at link time). A placeholder
     *          for the second-constant support that ArrayChunk2 offers per array
     *          (ArrayChunk2::set_second_value) and chunk2A per chunk.
     */
    template <typename T>
    static void set_second_value(T second_value) = delete;

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

    /**
     * @brief Print the instrumentation report accumulated during the run.
     *
     * @details Reduces the per-thread counters of each variant (chunk, chunk2,
     *          chunk2a) and prints one block per variant, preceded by a legend;
     *          variants that saw no activity are skipped. Reports allocations and
     *          reuse, read/write counts with the share of reads served straight
     *          from the constant (what the compression actually saves), compression
     *          attempts and successes, and — in a dynamic build — the re-chunk
     *          counters and the end-of-life chunk-size distribution.
     *
     * @note An empty no-op unless the build defines CHUNK_STATS; the counters
     *       themselves are only maintained in that build.
     */
    static void ChunkStats();

    /// @brief Lower sanity bound for a configured CHUNK_SIZE.
    static constexpr int size_min = 32;
    /**
     * @brief Upper sanity bound for a configured CHUNK_SIZE.
     *
     * @details This is NOT the chunk geometry — use shrink_array::__chunk_size for
     *          that; it only validates a size coming from an example or a config.
     *          The real hard limit is the per-chunk counter capacity (unsigned
     *          short), enforced by a static_assert in conditional_counter.h, hence
     *          65535.
     *
     * @warning Was 256 (then 128), a stale value predating the constexpr-geometry
     *          refactor, which wrongly rejected CHUNK_SIZE=512.
     */
    static constexpr int size_max = 65535;

  private:
    /// @brief Allocation block size in bytes, shared by all types and threads.
    static int block_size;
};
} // namespace shrink_array