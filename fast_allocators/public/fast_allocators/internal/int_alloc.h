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

#pragma once

#include <cstring>

#include "fast_allocators/internal/alloc_config.h" // alloc_config::alloc_align
#include "fast_allocators/internal/svalloc.h"

template <typename T>
void int_free(const T* var)
{
    if constexpr (alloc_config::alloc_align)
    {
        svfree_align(var);
    }
    else
    {
        svfree(var);
    }
}

template <typename T>
void int_free_align(const T* var)
{
    svfree_align(var);
}

template <typename T>
void int_alloc(T*& var, size_t size)
{
    if constexpr (alloc_config::alloc_align)
    {
        svalloc_align(var, T, size);
    }
    else
    {
        svalloc(var, T, size);
    }
}

template <typename T>
void int_alloc_align(T*& var, size_t size)
{
    svalloc_align(var, T, size);
}

inline void* int_alloc(size_t size)
{
    char* ptr = nullptr;

    if constexpr (alloc_config::alloc_align)
    {
        svalloc_align(ptr, char, size);
    }
    else
    {
        svalloc(ptr, char, size);
    }

    return (void*)ptr;
}

inline void* int_alloc_align(size_t size)
{
    char* ptr = nullptr;

    svalloc_align(ptr, char, size);

    return (void*)ptr;
}
