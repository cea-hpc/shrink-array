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

#include <type_traits> // std::remove_cvref_t, std::is_arithmetic_v

namespace shrink_array
{
/**
 * @brief Concept to detect shrink_array proxy types
 * @tparam T Type to check
 *
 * Proxy types are identified by an 'is_shrink_proxy_tag' typedef.
 */
template <typename T>
concept IsProxy = requires { typename std::remove_cvref_t<T>::is_shrink_proxy_tag; };

/**
 * @brief Concept for scalar arithmetic types
 * @tparam U Type to check
 *
 * Fundamental integer and floating-point types, as opposed to composite ones.
 */
template <typename U>
concept scalar_arithmetic = std::is_arithmetic_v<std::remove_cvref_t<U>>;

/**
 * @brief Concept for composite arithmetic types (non-scalar)
 * @tparam U Type to check
 * @tparam T Base type for conversion/construction
 *
 * Matches types that are neither scalar arithmetic nor proxies, and are
 * constructible from (or convertible from) T: std::complex<T>, matrices,
 * vectors, quaternions, tensors, etc.
 *
 * @note Relies on composite types having a constructor or conversion from
 *       their scalar type T (avoids circular dependencies).
 */
template <typename U, typename T>
concept composite_arithmetic =
    !scalar_arithmetic<U> && !IsProxy<U> &&
    (std::is_constructible_v<std::remove_cvref_t<U>, T> || std::is_convertible_v<T, std::remove_cvref_t<U>>);
} // namespace shrink_array