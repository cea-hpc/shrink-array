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

/*
 * posix_memalign and strict aliasing
 * ----------------------------------
 * These macros used to write the result straight into the caller's pointer:
 *
 *     posix_memalign((void**)&(var), __ALIGNEMENT__, n);
 *
 * `var` is a `type*`, so `(void**)&var` makes posix_memalign store a `void*`
 * INTO AN OBJECT OF TYPE `type*`. Those two types are not compatible for
 * aliasing purposes, so under -fstrict-aliasing (on from -O2) the compiler may
 * assume the call did not touch `var` and keep whatever it held before -- i.e.
 * an uninitialised pointer. gcc 16 says so itself:
 *
 *     svalloc.h:69: warning: 'ptr' may be used uninitialized
 *     chunk_base.h:1342: 'ptr' was declared here   ->   T* ptr;
 *
 * That is undefined behaviour, and it matched every observation on the
 * std-alloc dynamic crash: gcc -O2 aborted at teardown with "corrupted
 * double-linked list", -fno-strict-aliasing made it go away, -Og made it go
 * away, clang never reproduced it, and neither ASan nor TSan saw anything --
 * there is no out-of-bounds write and no bad free to see, just a pointer the
 * compiler was entitled not to reload.
 *
 * The fix is to give posix_memalign the only object it may legally write: a
 * real `void*`. Assigning it to the caller's pointer afterwards is an ordinary
 * conversion, and the aliasing rules are satisfied.
 *
 * The return value is checked too -- it was ignored, so an allocation failure
 * left `var` untouched (i.e. uninitialised) rather than null.
 */

#define svfree_align(var)    \
    do                       \
    {                        \
        assert((long)(var)); \
        free((void*)(var));  \
    } while (0)

#define svalloc_align(var, type, thesz)                           \
    do                                                            \
    {                                                             \
        void*  __sv_p = NULL;                                     \
        size_t __sv_n = (size_t)(thesz) * sizeof(type);           \
                                                                  \
        assert((long)(thesz));                                    \
        if (posix_memalign(&__sv_p, __ALIGNEMENT__, __sv_n) != 0) \
            __sv_p = NULL;                                        \
        assert(__sv_p);                                           \
        (var) = (type*)__sv_p;                                    \
    } while (0)

/*
 * NOTE: the zeroing below is a memset, which gcc rightly flags for non-trivial
 * element types (-Wclass-memaccess on std::complex<double>). It is left as is
 * because every T used here is zero-initialisable through an all-bits-zero
 * representation and callers rely on the raw fill; switching to
 * std::uninitialized_value_construct_n would be the clean fix but changes the
 * contract for C callers of this header.
 */
#define svcalloc_align(var, type, thesz)                          \
    do                                                            \
    {                                                             \
        void*  __sv_p = NULL;                                     \
        size_t __sv_n = (size_t)(thesz) * sizeof(type);           \
                                                                  \
        assert((long)(thesz));                                    \
        if (posix_memalign(&__sv_p, __ALIGNEMENT__, __sv_n) != 0) \
            __sv_p = NULL;                                        \
        assert(__sv_p);                                           \
        (var) = (type*)__sv_p;                                    \
        memset(__sv_p, 0, __sv_n);                                \
    } while (0)

#define svrealloc_align(var, type, oldthez, thesz)                 \
    do                                                             \
    {                                                              \
        void*  __sv_p = NULL;                                      \
        size_t __sv_n = (size_t)(thesz) * sizeof(type);            \
                                                                   \
        assert((long)(thesz));                                     \
        void* old_ptr = (void*)(var);                              \
        if (posix_memalign(&__sv_p, __ALIGNEMENT__, __sv_n) != 0)  \
            __sv_p = NULL;                                         \
        assert(__sv_p);                                            \
        (var) = (type*)__sv_p;                                     \
        if (oldthez)                                               \
        {                                                          \
            _Pragma("omp critical (realloc_align)")                \
            {                                                      \
                if ((thesz) > (oldthez))                           \
                    memcpy((var), old_ptr, sizeof(T) * (oldthez)); \
                else                                               \
                    memcpy((var), old_ptr, sizeof(T) * (thesz));   \
            }                                                      \
        }                                                          \
        if (old_ptr)                                               \
            svfree_align(old_ptr);                                 \
    } while (0)


#define svalloc(var, type, thesz)                     \
    do                                                \
    {                                                 \
        size_t __size = (thesz);                      \
                                                      \
        assert((long)__size);                         \
        (var) = (type*)malloc(__size * sizeof(type)); \
        assert((long)(var));                          \
    } while (0)

#define svcalloc(var, type, thesz)                   \
    do                                               \
    {                                                \
        size_t __size = (thesz);                     \
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
        size_t __size = (thesz);                      \
                                                      \
        assert((long)__size);                         \
        (var) = (type*)alloca(__size * sizeof(type)); \
        assert((long)(var));                          \
    } while (0)

#define svrealloc(var, type, thesz)                           \
    do                                                        \
    {                                                         \
        size_t __size = (thesz);                              \
                                                              \
        assert((long)__size);                                 \
        (var) = (type*)realloc((var), __size * sizeof(type)); \
        assert((long)(var));                                  \
    } while (0)
