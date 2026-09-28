
#pragma once
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

#include <type_traits>

namespace shrink_array
{
/**
 * @brief Concept to detect shrink_array proxy types
 * Template parameter T: Type to check
 *
 * Proxy types are identified by the presence of a 'is_shrink_proxy_tag' typedef.
 * This allows distinguishing proxies from regular types in generic contexts.
 */
template <typename T>
concept IsProxy = requires { typename std::remove_cvref_t<T>::is_shrink_proxy_tag; };

/**
 * @brief Concept for scalar arithmetic types
 * Template parameter U: Type to check
 *
 * Matches fundamental arithmetic types: integers and floating-point.
 * These are treated differently from composite types (complex, matrices, etc.)
 */
template <typename U>
concept scalar_arithmetic = std::is_arithmetic_v<std::remove_cvref_t<U>>;

/**
 * @brief Concept for composite arithmetic types (non-scalar)
 * Template parameter U: Type to check
 * Template parameter T: Base type for conversion/construction
 *
 * Matches types that:
 * - Are NOT scalar arithmetic
 * - Are NOT proxies
 * - Can be constructed from or converted from type T
 *
 * This simple approach avoids circular dependency issues while still
 * supporting std::complex<T>, matrices, vectors, quaternions, tensors, etc.
 *
 * @note We rely on the fact that composite types typically have constructors
 *       or conversion operators from their scalar type T
 */
template <typename U, typename T>
concept composite_arithmetic =
    !scalar_arithmetic<U> && !IsProxy<U> &&
    (std::is_constructible_v<std::remove_cvref_t<U>, T> || std::is_convertible_v<T, std::remove_cvref_t<U>>);
} // namespace shrink_array