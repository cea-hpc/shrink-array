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

// Portable thread-count / rank / parallel-region macros for the fast allocator:
// the StaticConstArray ladder (static_const_array.h, its _inline.h, and
// InfoFreeElt's constructor in static_const_array.cpp). Mirrors chunk's
// public/chunk/PMP_CHUNK.h.
//
// The data-parallel loops (fast_alloc::pmp::parallel_for, ...) come with it,
// from pmp_loops.h, which chunk's PMP_CHUNK.h re-exports as shrink_array::pmp.

#include "fast_allocators/internal/pmp_loops.h" // fast_alloc::pmp: parallel_for, parallel_for_local, parallel_sum

#ifdef _OPENMP
    #include <omp.h> // omp_get_max_threads, omp_get_thread_num, omp_in_parallel

    #ifndef FAST_ALLOC_PMP_MAX_THREADS
        #define FAST_ALLOC_PMP_MAX_THREADS omp_get_max_threads()
    #endif

    #ifndef FAST_ALLOC_PMP_THREAD_RANK
        #define FAST_ALLOC_PMP_THREAD_RANK omp_get_thread_num()
    #endif

    #ifndef FAST_ALLOC_PMP_IN_PARALLEL_REGION
        #define FAST_ALLOC_PMP_IN_PARALLEL_REGION omp_in_parallel()
    #endif

#else

    // A serial build has one thread, rank 0 -- values, not empty expansions, so
    // code that names them compiles in both modes (mirrors chunk's PMP_CHUNK.h).
    #ifndef FAST_ALLOC_PMP_MAX_THREADS
        #define FAST_ALLOC_PMP_MAX_THREADS 1
    #endif

    #ifndef FAST_ALLOC_PMP_THREAD_RANK
        #define FAST_ALLOC_PMP_THREAD_RANK 0
    #endif

    #ifndef FAST_ALLOC_PMP_IN_PARALLEL_REGION
        #define FAST_ALLOC_PMP_IN_PARALLEL_REGION false
    #endif

#endif
