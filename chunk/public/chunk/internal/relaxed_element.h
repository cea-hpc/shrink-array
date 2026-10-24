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
/**
 * @file relaxed_element.h
 * @brief Relaxed atomic access to the chunk elements other threads read while
 *        this one writes.
 *
 * @details Writers of DIFFERENT elements of one chunk run concurrently by
 *          design, with nothing between them. Two readers look at elements
 *          they do not own:
 *            - zero_any::Proxy::may_become_uniform(), the marking filter: element 0 or
 *              1, and S-1 or S-2;
 *            - fill_range_mt's may_be_uniform(): 0, S/2 and S-1.
 *          Both only take a hint from the value, but plain accesses there are
 *          still a data race (undefined behaviour). Both sides therefore go
 *          through std::atomic_ref, relaxed: no ordering is wanted, only the
 *          absence of a race. On x86-64 and AArch64 a relaxed load or store of a
 *          lock-free scalar is the plain instruction.
 *
 *          Eligible: types atomic_ref handles without a lock and at their own
 *          alignment -- the scalars. The aggregates (complex, Point3, Cell)
 *          are not: their readers skip the read instead (see
 *          may_become_uniform()).
 */

#include <atomic>      // std::atomic_ref, std::memory_order_relaxed
#include <type_traits> // std::is_same_v, std::is_trivially_copyable_v

#include "chunk/chunk_traits.h" // is_zero_any_family_v

namespace shrink_array::internal
{

/// @brief Whether T is trivially copyable and std::atomic_ref<T> is always lock-free with a required
///        alignment equal to alignof(T).
template <typename T>
consteval bool relaxed_element_access()
{
    // Guarded: atomic_ref<T> is ill-formed for a T that is not trivially copyable.
    if constexpr (!std::is_trivially_copyable_v<T>)
    {
        return false;
    }
    else
    {
        return std::atomic_ref<T>::is_always_lock_free && std::atomic_ref<T>::required_alignment == alignof(T);
    }
}

/// @brief T can be read and written through a lock-free relaxed atomic_ref.
template <typename T>
inline constexpr bool relaxed_element_access_v = relaxed_element_access<T>();

/// @brief The element stores of this chunk type must be relaxed atomics: some
///        reader looks at elements it does not own (the zero_any family, see above).
template <typename ChunkT, typename T>
inline constexpr bool relaxed_element_stores_v = is_zero_any_family_v<ChunkT> && relaxed_element_access_v<T>;

/// @brief Load @p *ptr through a relaxed std::atomic_ref.
/// @param ptr Element to read.
template <typename T>
[[nodiscard]] T load_element_relaxed(const T* ptr) noexcept
{
    static_assert(relaxed_element_access_v<T>);
    // atomic_ref<const T> is C++26; a load never writes through the reference.
    return std::atomic_ref<T>(*const_cast<T*>(ptr)).load(std::memory_order_relaxed);
}

/// @brief Store @p value to @p *ptr through a relaxed std::atomic_ref.
/// @param ptr Element to write.
/// @param value Value to store.
template <typename T>
void store_element_relaxed(T* ptr, T value) noexcept
{
    static_assert(relaxed_element_access_v<T>);
    std::atomic_ref<T>(*ptr).store(value, std::memory_order_relaxed);
}

/**
 * @brief Run an in-place write body (++, --, the bitwise and compound
 *        operators) on element @p index of the buffer @p ptr.
 * @details The body takes the element by reference. Where the element stores
 *          must be relaxed atomics it works on a copy, stored back relaxed; the
 *          read of the copy stays plain -- the element is this writer's own,
 *          the other threads only read it.
 */
template <typename ChunkT, typename T, typename Body>
void modify_element(T* ptr, int index, Body&& body)
{
    if constexpr (relaxed_element_stores_v<ChunkT, T>)
    {
        T elem = ptr[index];
        body(elem);
        store_element_relaxed(ptr + index, elem);
    }
    else
    {
        body(ptr[index]);
    }
}

} // namespace shrink_array::internal
