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

/**
 * @file svalloc.h
 * @brief Checked malloc / calloc / free, plain and aligned on fast_alloc::alignment.
 *
 * Shared with the chunk library, which includes it from here: a std-alloc
 * build, which has no fast_allocators library, still has fast_allocators'
 * headers on chunk's include path for this one and singleton.h (see
 * chunk/CMakeLists.txt).
 *
 * Each allocation asserts a non-zero count (debug builds). An allocation that
 * fails -- or whose byte count would overflow size_t -- prints what was asked
 * and aborts, in EVERY build (the library's policy). Callers never see null,
 * and none of them tests for it.
 *
 * posix_memalign writes into a real `void*` here. Do not pass it
 * `(void**)&var` with `var` a `type*`: that breaks strict aliasing, and under
 * -O2 gcc may then use a stale `var` (heap corruption invisible to ASan/TSan).
 */
#pragma once

#include <cassert> // assert (the non-zero counts)
#include <cerrno>  // errno, ENOMEM: the failure message
#include <cstddef> // std::size_t
#include <cstdint> // SIZE_MAX
#include <cstdlib> // std::malloc, std::calloc, std::free
#include <cstring> // std::memset, std::strerror

#include <stdlib.h> // posix_memalign (POSIX: <cstdlib> need not declare it)

#include "fast_allocators/internal/fatal.h" // FAST_ALLOC_FATAL (out_of_memory)

#ifndef FAST_ALLOC_SIMD_SZ
    #define FAST_ALLOC_SIMD_SZ 32u
#endif

namespace fast_alloc
{

/// @brief The alignment of the aligned allocations and of the aligned bucket
///        ladder: FAST_ALLOC_SIMD_SZ bytes (a build may define FAST_ALLOC_SIMD_SZ itself).
inline constexpr std::size_t alignment = FAST_ALLOC_SIMD_SZ;

/**
 * @brief Report a failed allocation and stop -- the library's policy on
 *        allocation failure, in every build.
 * @param what the allocating function, for the message
 * @param n    elements requested
 * @param elem bytes per element
 * @param err  the error (errno, or posix_memalign's result); 0 when n * elem
 *             overflows size_t
 */
[[noreturn]] [[gnu::cold]] [[gnu::noinline]] inline void
out_of_memory(const char* what, std::size_t n, std::size_t elem, int err) noexcept
{
    if (err != 0)
    {
        FAST_ALLOC_FATAL("{}: allocation of {} x {} bytes failed ({})", what, n, elem, std::strerror(err));
    }
    FAST_ALLOC_FATAL("{}: {} x {} bytes overflows size_t", what, n, elem);
}

/// @brief @p n * @p elem, or out_of_memory() when the product overflows.
[[nodiscard]] inline std::size_t checked_bytes(const char* what, std::size_t n, std::size_t elem) noexcept
{
    if (elem != 0 && n > SIZE_MAX / elem) [[unlikely]]
    {
        out_of_memory(what, n, elem, 0);
    }
    return n * elem;
}

namespace detail
{
/// @brief @p n elements of @p elem bytes aligned on #alignment; never null
///        for a non-zero size (out_of_memory() otherwise).
[[nodiscard]] inline void* aligned_or_die(const char* what, std::size_t n, std::size_t elem) noexcept
{
    const std::size_t bytes = checked_bytes(what, n, elem);
    void*             p     = nullptr; // the only object posix_memalign may legally write
    const int         rc    = posix_memalign(&p, alignment, bytes);
    if (rc != 0 || (p == nullptr && bytes != 0)) [[unlikely]]
    {
        out_of_memory(what, n, elem, rc != 0 ? rc : ENOMEM);
    }
    return p;
}

/// @brief Abort with the message when malloc/calloc returned null for a
///        non-zero size.
inline void check_malloc(const void* p, const char* what, std::size_t n, std::size_t elem) noexcept
{
    if (p == nullptr && n != 0 && elem != 0) [[unlikely]]
    {
        out_of_memory(what, n, elem, errno != 0 ? errno : ENOMEM);
    }
}
} // namespace detail

/// @brief @p n elements of T, uninitialised, aligned on fast_alloc::alignment.
///        Release with svfree_align(). Never null: aborts on failure.
template <typename T>
[[nodiscard]] T* svalloc_align(std::size_t n) noexcept
{
    assert(n != 0);
    return static_cast<T*>(fast_alloc::detail::aligned_or_die("svalloc_align", n, sizeof(T)));
}

/// @brief svalloc_align(), every byte zero. The fill is a raw memset: every T
///        used here is zero-initialisable through all-bits-zero.
template <typename T>
[[nodiscard]] T* svcalloc_align(std::size_t n) noexcept
{
    assert(n != 0);
    void* p = fast_alloc::detail::aligned_or_die("svcalloc_align", n, sizeof(T));
    std::memset(p, 0, n * sizeof(T)); // n * sizeof(T) checked by aligned_or_die
    return static_cast<T*>(p);
}

/// @brief Release a buffer from svalloc_align() / svcalloc_align().
inline void svfree_align(const void* p) noexcept
{
    assert(p != nullptr);
    std::free(const_cast<void*>(p));
}

/// @brief @p n elements of T, uninitialised (malloc). Release with svfree().
///        Never null: aborts on failure.
template <typename T>
[[nodiscard]] T* svalloc(std::size_t n) noexcept
{
    assert(n != 0);
    void* p = std::malloc(fast_alloc::checked_bytes("svalloc", n, sizeof(T)));
    fast_alloc::detail::check_malloc(p, "svalloc", n, sizeof(T));
    return static_cast<T*>(p);
}

/// @brief @p n elements of T, zeroed (calloc). Release with svfree().
///        Never null: aborts on failure.
template <typename T>
[[nodiscard]] T* svcalloc(std::size_t n) noexcept
{
    assert(n != 0);
    (void)fast_alloc::checked_bytes("svcalloc", n, sizeof(T)); // the overflow message, before calloc's null
    void* p = std::calloc(n, sizeof(T));
    fast_alloc::detail::check_malloc(p, "svcalloc", n, sizeof(T));
    return static_cast<T*>(p);
}

/// @brief Release a buffer from svalloc() / svcalloc().
inline void svfree(const void* p) noexcept
{
    assert(p != nullptr);
    std::free(const_cast<void*>(p));
}

} // namespace fast_alloc
