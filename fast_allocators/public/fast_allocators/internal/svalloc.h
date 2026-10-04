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

/**
 * @file svalloc.h
 * @brief Checked malloc / calloc / free, plain and aligned on fast_alloc::alignment.
 *
 * Shared with the chunk library, which includes it from here: a std-alloc
 * build, which has no fast_allocators library, still has fast_allocators'
 * headers on chunk's include path for this one and singleton.h (see
 * chunk/CMakeLists.txt). There used to be a copy in chunk/internal, kept
 * identical by hand.
 *
 * Functions, not the macros they were (`svalloc(var, type, n)` wrote into
 * `var`; `var = svalloc<type>(n)` now). Each allocation asserts a non-zero
 * count and a non-null result; a failure returns null in a release build.
 *
 * posix_memalign writes into a real `void*` here. The macros used to pass it
 * `(void**)&var` with `var` a `type*`: storing a `void*` into an object of type
 * `type*` breaks strict aliasing, and under -O2 gcc was entitled not to reload
 * `var` -- an uninitialised pointer, which matched every observation of the
 * std-alloc dynamic crash ("corrupted double-linked list" at teardown, gone
 * with -fno-strict-aliasing or -Og, never under clang, invisible to ASan/TSan).
 */
#pragma once

#include <cassert> // assert
#include <cstddef> // std::size_t
#include <cstdlib> // posix_memalign, std::malloc, std::calloc, std::free
#include <cstring> // std::memset

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

namespace fast_alloc
{

/// @brief The alignment of the aligned allocations and of the aligned bucket
///        ladder: SIMD_SZ bytes (a build may define SIMD_SZ itself).
inline constexpr std::size_t alignment = SIMD_SZ;

/// @brief @p bytes bytes aligned on #alignment, or null.
[[nodiscard]] inline void* aligned_bytes(std::size_t bytes) noexcept
{
    void* p = nullptr; // the only object posix_memalign may legally write
    if (posix_memalign(&p, alignment, bytes) != 0)
    {
        p = nullptr;
    }
    return p;
}

} // namespace fast_alloc

/// @brief @p n elements of T, uninitialised, aligned on fast_alloc::alignment.
///        Release with svfree_align().
template <typename T>
[[nodiscard]] T* svalloc_align(std::size_t n) noexcept
{
    assert(n != 0);
    void* p = fast_alloc::aligned_bytes(n * sizeof(T));
    assert(p != nullptr);
    return static_cast<T*>(p);
}

/// @brief svalloc_align(), every byte zero. The fill is a raw memset: every T
///        used here is zero-initialisable through all-bits-zero.
template <typename T>
[[nodiscard]] T* svcalloc_align(std::size_t n) noexcept
{
    assert(n != 0);
    void* p = fast_alloc::aligned_bytes(n * sizeof(T));
    assert(p != nullptr);
    if (p != nullptr)
    {
        std::memset(p, 0, n * sizeof(T));
    }
    return static_cast<T*>(p);
}

/// @brief Release a buffer from svalloc_align() / svcalloc_align().
inline void svfree_align(const void* p) noexcept
{
    assert(p != nullptr);
    std::free(const_cast<void*>(p));
}

/// @brief @p n elements of T, uninitialised (malloc). Release with svfree().
template <typename T>
[[nodiscard]] T* svalloc(std::size_t n) noexcept
{
    assert(n != 0);
    void* p = std::malloc(n * sizeof(T));
    assert(p != nullptr);
    return static_cast<T*>(p);
}

/// @brief @p n elements of T, zeroed (calloc). Release with svfree().
template <typename T>
[[nodiscard]] T* svcalloc(std::size_t n) noexcept
{
    assert(n != 0);
    void* p = std::calloc(n, sizeof(T));
    assert(p != nullptr);
    return static_cast<T*>(p);
}

/// @brief Release a buffer from svalloc() / svcalloc().
inline void svfree(const void* p) noexcept
{
    assert(p != nullptr);
    std::free(const_cast<void*>(p));
}
