#pragma once

/*
 * Copyright 2024 CEA (www.cea.fr)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */

// Portable thread-count / rank / parallel-region macros for the fast allocator:
// the StaticConstArray ladder (static_const_array.h, its _inline.h, and
// InfoFreeElt's constructor in static_const_array.cpp). Mirrors chunk's
// public/chunk/PMP_CHUNK.h.

#ifdef _OPENMP
    #include <omp.h> // omp_get_max_threads, omp_get_thread_num, omp_in_parallel

    #ifndef PMP_MAX_THREADS
        #define PMP_MAX_THREADS omp_get_max_threads()
    #endif

    #ifndef PMP_THREAD_RANK
        #define PMP_THREAD_RANK omp_get_thread_num()
    #endif

    #ifndef PMP_IN_PARALLEL_REGION
        #define PMP_IN_PARALLEL_REGION omp_in_parallel()
    #endif

#else

    // A serial build has one thread, rank 0 -- values, not empty expansions, so
    // code that names them compiles in both modes (mirrors chunk's PMP_CHUNK.h).
    #ifndef PMP_MAX_THREADS
        #define PMP_MAX_THREADS 1
    #endif

    #ifndef PMP_THREAD_RANK
        #define PMP_THREAD_RANK 0
    #endif

    #ifndef PMP_IN_PARALLEL_REGION
        #define PMP_IN_PARALLEL_REGION false
    #endif

#endif
