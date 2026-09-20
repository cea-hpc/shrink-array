/*
 * Copyright 2024 CEA (www.cea.fr)
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

/*
 * svalloc.h : some macros for memory allocation
 *
 */
#pragma once

#include <assert.h>
#include <stdlib.h>
#include <string.h> // memset / memcpy in the macros below

#ifdef HAVE_ALLOCA
    #include <alloca.h>
#endif
#ifdef __cplusplus
    #include <memory>
#endif

#ifndef SIMD_SZ

    #if defined MIC || defined KNL
        #define SIMD_SZ 64u
    #elif ARM
        #define SIMD_SZ 16u
    #elif ATOS_7
        #define SIMD_SZ 32u
    #else
        #define SIMD_SZ 32u
    #endif

#endif

#define __ALIGNEMENT__ SIMD_SZ

#define svfree_align(var)    \
    do                       \
    {                        \
        assert((long)(var)); \
        free((void*)(var));  \
    } while (0)

#define svalloc_align(var, type, thesz)                                         \
    do                                                                          \
    {                                                                           \
        assert((long)(thesz));                                                  \
        posix_memalign((void**)&(var), __ALIGNEMENT__, (thesz) * sizeof(type)); \
        assert((long)(var));                                                    \
    } while (0)

#define svcalloc_align(var, type, thesz)                                        \
    do                                                                          \
    {                                                                           \
        assert((long)(thesz));                                                  \
        posix_memalign((void**)&(var), __ALIGNEMENT__, (thesz) * sizeof(type)); \
        assert((long)(var));                                                    \
        memset(var, 0, (thesz) * sizeof(type));                                 \
    } while (0)

#define svrealloc_align(var, type, oldthez, thesz)                              \
    do                                                                          \
    {                                                                           \
        assert((long)(thesz));                                                  \
        void* old_ptr = (var);                                                  \
        posix_memalign((void**)&(var), __ALIGNEMENT__, (thesz) * sizeof(type)); \
        assert((long)(var));                                                    \
        if (oldthez)                                                            \
        {                                                                       \
            _Pragma("omp critical (realloc_align)")                             \
            {                                                                   \
                if ((thesz) > (oldthez))                                        \
                    memcpy((var), old_ptr, sizeof(T) * (oldthez));              \
                else                                                            \
                    memcpy((var), old_ptr, sizeof(T) * (thesz));                \
            }                                                                   \
        }                                                                       \
        if (old_ptr)                                                            \
            svfree_align(old_ptr);                                              \
    } while (0)


#define svalloc(var, type, thesz)                     \
    do                                                \
    {                                                 \
        const size_t __size = (thesz);                \
                                                      \
        assert((long)__size);                         \
        (var) = (type*)malloc(__size * sizeof(type)); \
        assert((long)(var));                          \
    } while (0)

#define svcalloc(var, type, thesz)                   \
    do                                               \
    {                                                \
        const size_t __size = (thesz);               \
                                                     \
        assert((long)__size);                        \
        (var) = (type*)calloc(__size, sizeof(type)); \
        assert((long)(var));                         \
    } while (0)

#define svfree(var)                     \
    do                                  \
    {                                   \
        void* __svf_var = (void*)(var); \
                                        \
        assert((long)__svf_var);        \
        free(__svf_var);                \
    } while (0)

#define svtfree(var)                    \
    do                                  \
    {                                   \
        void* __svf_var = (void*)(var); \
                                        \
        if (__svf_var)                  \
            free(__svf_var);            \
    } while (0)

#define svzfree(var, type) \
    do                     \
    {                      \
        svfree(var);       \
        (var) = (type*)0;  \
    } while (0)

#define svalloca(var, type, thesz)                    \
    do                                                \
    {                                                 \
        const size_t __size = (thesz);                \
                                                      \
        assert((long)__size);                         \
        (var) = (type*)alloca(__size * sizeof(type)); \
        assert((long)(var));                          \
    } while (0)

#define svrealloc(var, type, thesz)                           \
    do                                                        \
    {                                                         \
        const size_t __size = (thesz);                        \
                                                              \
        assert((long)__size);                                 \
        (var) = (type*)realloc((var), __size * sizeof(type)); \
        assert((long)(var));                                  \
    } while (0)
