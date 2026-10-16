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
 * @file lifetime.h
 * @brief Start the lifetime of objects in storage the allocators hand out
 *        (std::start_lifetime_as, C++23), where the standard library has it.
 *
 * @details malloc / calloc / aligned_alloc create the objects their caller
 *          uses implicitly (C++20, [intro.object]); a block carved out of a
 *          slab or a mmap'd arena does not -- no object of the element type
 *          exists there until one is created. And a recycled slab element
 *          held a free-list link (an int, a pointer) last: writing that link
 *          ended the life of the elements it overlapped. Accessing the
 *          elements through a plain cast is formally undefined in both cases,
 *          which is what std::start_lifetime_as_array settles: it creates the
 *          elements in place, keeping the bytes, at no cost in code (an empty
 *          asm that tells the optimiser the storage changed).
 *
 *          libc++ (the omp-clang23 preset) and libstdc++ before GCC 16 do not
 *          have it: the plain cast, as before, which every compiler in use
 *          compiles to the same thing.
 *
 *          These start a lifetime: they belong where an object is first
 *          written. A read of an object already living there (a slab's
 *          back-pointer) uses std::launder instead -- starting a lifetime
 *          again would end the one other threads are reading.
 */

#pragma once

#include <cstddef>     // std::size_t
#include <memory>      // std::start_lifetime_as, std::start_lifetime_as_array
#include <type_traits> // std::is_implicit_lifetime_v

namespace fast_alloc
{
/// @brief @p n objects of T in place at @p p (bytes kept), returned as T*.
/// @pre T is an implicit-lifetime type (arithmetic, trivially copyable
///      aggregates, std::complex...): checked where the library can.
template <typename T>
[[nodiscard]] inline T* start_lifetime_as_array(void* p, std::size_t n) noexcept
{
#if defined(__cpp_lib_is_implicit_lifetime)
    static_assert(std::is_implicit_lifetime_v<T>, "only an implicit-lifetime type can start its life in place");
#endif
#if defined(__cpp_lib_start_lifetime_as)
    return std::start_lifetime_as_array<T>(p, n);
#else
    (void)n;
    return static_cast<T*>(p);
#endif
}

/// @brief One object of T in place at @p p (bytes kept), returned as T*.
/// @pre As start_lifetime_as_array().
template <typename T>
[[nodiscard]] inline T* start_lifetime_as(void* p) noexcept
{
#if defined(__cpp_lib_start_lifetime_as)
    return std::start_lifetime_as<T>(p);
#else
    return static_cast<T*>(p);
#endif
}

} // namespace fast_alloc
