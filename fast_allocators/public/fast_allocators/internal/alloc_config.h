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

#include "fast_allocators/build_options.h"        // FAST_ALLOC_OPT_*: the options the library was built with
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

/// @brief int_alloc / int_free go through the aligned allocation (-DFAST_ALLOC_ALIGN).
inline constexpr bool alloc_align =
#ifdef FAST_ALLOC_ALIGN
    true;
#else
    false;
#endif

} // namespace fast_alloc::alloc_config
