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

#include <cstddef> // size_t

#include "fast_allocators/internal/alloc_config.h" // alloc_config::alloc_align
#include "fast_allocators/internal/svalloc.h"      // svalloc, svalloc_align, svfree, svfree_align

namespace fast_alloc
{

/// @brief Release a buffer obtained from int_alloc(size_t), with the matching
///        routine: aligned or not, as alloc_config::alloc_align says.
/// @param var Buffer to release.
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

/// @brief Release a buffer obtained from int_alloc_align(size_t).
/// @param var Buffer to release.
template <typename T>
void int_free_align(const T* var)
{
    svfree_align(var);
}

/// @brief @p size bytes, aligned when alloc_config::alloc_align is set.
/// @param size Number of bytes.
/// @return The buffer; release with int_free().
inline void* int_alloc(size_t size)
{
    if constexpr (alloc_config::alloc_align)
    {
        return svalloc_align<char>(size);
    }
    else
    {
        return svalloc<char>(size);
    }
}

/// @brief @p size bytes, always aligned.
/// @param size Number of bytes.
/// @return The buffer; release with int_free_align().
inline void* int_alloc_align(size_t size)
{
    return svalloc_align<char>(size);
}

} // namespace fast_alloc
