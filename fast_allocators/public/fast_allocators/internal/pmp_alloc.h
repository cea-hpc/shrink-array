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
// The macros are the runtime queries of pmp_runtime.h (the backend's).

#include "fast_allocators/internal/pmp_loops.h"   // fast_alloc::pmp: parallel_for, parallel_for_local, parallel_sum
#include "fast_allocators/internal/pmp_runtime.h" // fast_alloc::pmp: max_threads, team_rank, in_parallel

#if FAST_ALLOC_PMP_THREADED

    #ifndef FAST_ALLOC_PMP_MAX_THREADS
        #define FAST_ALLOC_PMP_MAX_THREADS ::fast_alloc::pmp::max_threads()
    #endif

    #ifndef FAST_ALLOC_PMP_THREAD_RANK
        #define FAST_ALLOC_PMP_THREAD_RANK ::fast_alloc::pmp::team_rank()
    #endif

    #ifndef FAST_ALLOC_PMP_IN_PARALLEL_REGION
        #define FAST_ALLOC_PMP_IN_PARALLEL_REGION ::fast_alloc::pmp::in_parallel()
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
