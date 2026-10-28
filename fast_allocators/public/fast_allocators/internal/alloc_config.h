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
 * @file alloc_config.h
 * @brief `constexpr` mirrors of the macros that configure the fast allocators.
 *
 * The macros are read HERE and nowhere else, so the allocator bodies branch with
 * `if constexpr` instead of <tt>\#ifdef</tt>: every branch type-checks in every build.
 *
 * Mirrors chunk/config.h, deliberately separate: fast_allocators must not include chunk/.
 */

#include <charconv>     // std::from_chars (CHUNK_RECLAIM_MIN_SLABS)
#include <cstddef>      // std::size_t
#include <cstdlib>      // std::getenv
#include <cstring>      // std::strlen, std::strcmp (env_switch)
#include <system_error> // std::errc

#include "fast_allocators/build_options.h"        // FAST_ALLOC_OPT_*: the options the library was built with
#include "fast_allocators/internal/log_core.h"    // CHUNK_LOG (the CHUNK_RECLAIM_MIN_SLABS message)
#include "fast_allocators/internal/pmp_runtime.h" // pmp::threaded (checked against FAST_ALLOC_OPT_THREADED)

namespace fast_alloc::alloc_config
{

/// @brief Built against `new[]`/`delete[]` rather than the slab allocator
///        (-DFAST_ALLOC_STD_ALLOC: a header-only use without the library).
inline constexpr bool use_std_alloc =
#ifdef FAST_ALLOC_STD_ALLOC
    true;
#else
    false;
#endif

/// @brief Built with the slab machinery. The complement of #use_std_alloc.
inline constexpr bool use_slabs = !use_std_alloc;

/// @brief Built with OpenMP (build_options.h refuses a file compiled otherwise).
inline constexpr bool use_openmp = FAST_ALLOC_OPT_OPENMP != 0;

/// @brief The library runs threads: every threading backend but serial
///        (CMake CHUNK_THREADS, docs/threading_backends.md). What the per-rank
///        machinery, the atomics and the locks test; #use_openmp stays for
///        what only OpenMP gives.
inline constexpr bool threaded = FAST_ALLOC_OPT_THREADED != 0;
static_assert(threaded == pmp::threaded,
              "fast_allocators: the build's threading flag and the backend pmp_runtime.h selects disagree");

/// @brief Debug build (CMake: CHUNK_DEBUG, on with CMAKE_BUILD_TYPE=Debug).
inline constexpr bool is_debug = FAST_ALLOC_OPT_DEBUG != 0;

/// @brief The reinforced debug level of the slab lists: a debug build with
///        FAST_ALLOC_CHECK_LISTS (CMake option of the same name) walks a whole
///        list on every insert and remove (FAST_ALLOC_SL_LIST_CHECK in tc_allocator.h).
///        Linear per operation: a plain Debug build does not.
inline constexpr bool check_slab_lists = FAST_ALLOC_OPT_DEBUG != 0 && FAST_ALLOC_OPT_CHECK_LISTS != 0;

/**
 * @brief An emptied slab gets MADV_DONTNEED instead of staying resident.
 * @details On by default (CMake `CHUNK_RECLAIM_EMPTY_SLABS=OFF` turns it off), recorded in
 *          the generated build_options.h. Useful only once slabs reach nn_ref == 0 -- see
 *          ManagerChunk::drain_reuse_list_for_size (empties the old geometry's slabs).
 */
inline constexpr bool reclaim_empty_slabs = FAST_ALLOC_OPT_RECLAIM_EMPTY_SLABS != 0;

/**
 * @brief Fewest distinct empty slabs an EmptySlabSweep must hold for its
 *        flush() to advise them, read once from `CHUNK_RECLAIM_MIN_SLABS`. A re-chunk's
 *        sweep spans every rank, so the bar is the whole abandoned set's.
 *
 * @details One madvise per slab: the trade is decided by how many slabs a set empties at
 *          once (a re-chunk returns gigabytes for its syscalls, a teardown of a few slabs
 *          a few megabytes). The threshold keeps the first and drops the second. With
 *          sweeping off, this allocator is a few percent faster than the standard one at
 *          equal memory; sweeping trades that speed for pages.
 *
 *          0 (default when unset) sweeps everything. Read once, at a point that runs a
 *          handful of times per re-chunk. The free path must never madvise: a slab whose
 *          nn_ref briefly reads zero can still hold a live block.
 */
inline std::size_t reclaim_min_empty_slabs() noexcept
{
    static const std::size_t min_slabs = []() -> std::size_t
    {
        // The whole value must be a non-negative integer, else warning and 0 (the rule of
        // chunk/internal/env_parse.h, which this library cannot include).
        const char* env = std::getenv("CHUNK_RECLAIM_MIN_SLABS");
        if (env == nullptr || env[0] == '\0')
        {
            return 0;
        }
        std::size_t value = 0;
        const char* end   = env + std::strlen(env);
        const auto  res   = std::from_chars(env, end, value);
        if (res.ec != std::errc{} || res.ptr != end)
        {
            CHUNK_LOG(warn, env, "CHUNK_RECLAIM_MIN_SLABS=\"{}\" is not a valid value: ignored, 0 kept", env);
            return 0;
        }
        return value;
    }();
    return min_slabs;
}

/**
 * @brief A 0/1 switch from the environment.
 * @return @p fallback when @p name is unset or empty; false for `0`, true for
 *         `1`; anything else is a warning and @p fallback.
 */
inline bool env_switch(const char* name, bool fallback) noexcept
{
    const char* env = std::getenv(name);
    if (env == nullptr || env[0] == '\0')
    {
        return fallback;
    }
    if (std::strcmp(env, "0") == 0)
    {
        return false;
    }
    if (std::strcmp(env, "1") == 0)
    {
        return true;
    }
    CHUNK_LOG(warn, env, "{}=\"{}\" is not a valid value (0 or 1): ignored", name, env);
    return fallback;
}

/**
 * @brief CHUNK_RECLAIM_PARTIAL: does a sweep also give back the free pages of
 *        the value slabs still in use? Read once; on unless `0`.
 * @details A slab goes back whole only once every buffer in it is freed, so a
 *          few survivors keep it resident. A value slab keeps its free-list
 *          links aside (FreeLinks::side_array) and never writes into a free
 *          element: a page covered by free elements only can be dropped, and
 *          comes back zero-filled when an element in it is handed out again --
 *          what malloc_trim() does inside glibc's heaps. See
 *          TCSlab::advise_free_pages().
 */
inline bool reclaim_partial_slabs() noexcept
{
    static const bool on = env_switch("CHUNK_RECLAIM_PARTIAL", true);
    return on;
}

/**
 * @brief CHUNK_SLAB_FULLEST_FIRST: does a sweep re-chain the value slabs with
 *        room fullest first? Read once; on unless `0`.
 * @details The next allocations then fill the dense slabs and leave the sparse
 *          ones to drain, and an empty slab is given back whole. See
 *          TCTypeCAllocator::order_fullest_first().
 */
inline bool slab_fullest_first() noexcept
{
    static const bool on = env_switch("CHUNK_SLAB_FULLEST_FIRST", true);
    return on;
}


/// @brief int_alloc / int_free go through the aligned allocation (-DFAST_ALLOC_ALIGN).
inline constexpr bool alloc_align =
#ifdef FAST_ALLOC_ALIGN
    true;
#else
    false;
#endif

} // namespace fast_alloc::alloc_config
