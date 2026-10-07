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

#include <type_traits> // std::common_type_t

#include "chunk/internal/concept_composite_arithmetic.h" // composite_arithmetic, scalar_arithmetic

namespace shrink_array
{

// ============================================================================
// ProxyOperators: Base CRTP class providing all arithmetic operators
// ============================================================================

/**
 * @brief CRTP base class providing arithmetic operators for proxy types
 * @tparam Derived The derived proxy class (Proxy2A, Proxy2, etc.)
 * @tparam T The underlying value type
 *
 * This class implements all arithmetic operators (+, -, *, /) for proxy types,
 * handling both composite types (complex, matrices, vectors) and scalar types.
 *
 * Key features:
 * - Uses C++20 concepts for cleaner constraints
 * - CRTP pattern for zero-overhead abstraction
 * - Preserves type precision with std::common_type_t for scalars
 * - Supports any composite type (matrices, complex, quaternions, etc.)
 * - Separate overloads for composite and scalar to avoid ambiguity
 *
 * @note All operators are constexpr and noexcept where possible
 */
template <typename Derived, typename T>
class ProxyOperators
{
  public:
    // ========================================================================
    // Multiplication operators
    // ========================================================================

    /**
     * @brief Multiplies a composite type by a Proxy (commutative: composite * proxy)
     * @tparam U Composite arithmetic type (complex, matrix, vector, quaternion, etc.)
     * @param lhs Left operand (composite value)
     * @param rhs Right operand (proxy)
     * @return Result of multiplication in type U
     *
     * @note Works with any type supporting arithmetic operations with T
     */
    template <typename U>
        requires composite_arithmetic<U, T>
    [[nodiscard]] friend constexpr U operator*(U lhs, const Derived& rhs) noexcept(noexcept(lhs * static_cast<T>(rhs)))
    {
        return lhs * static_cast<T>(rhs);
    }

    /**
     * @brief Multiplies a Proxy by a composite type (commutative: proxy * composite)
     * @tparam U Composite arithmetic type (complex, matrix, vector, quaternion, etc.)
     * @param lhs Left operand (proxy)
     * @param rhs Right operand (composite value)
     * @return Result of multiplication in type U
     *
     * @note Works with any type supporting arithmetic operations with T
     */
    template <typename U>
        requires composite_arithmetic<U, T>
    [[nodiscard]] friend constexpr U operator*(const Derived& lhs, U rhs) noexcept(noexcept(static_cast<T>(lhs) * rhs))
    {
        return static_cast<T>(lhs) * rhs;
    }

    /**
     * @brief Multiplies a scalar by a Proxy (commutative: scalar * proxy)
     * @tparam U Scalar arithmetic type (int, float, double, etc.)
     * @param lhs Left operand (scalar)
     * @param rhs Right operand (proxy)
     * @return Result in common type to preserve precision
     */
    template <scalar_arithmetic U>
    [[nodiscard]] friend constexpr std::common_type_t<T, U> operator*(U lhs, const Derived& rhs) noexcept
    {
        return lhs * static_cast<T>(rhs);
    }

    /**
     * @brief Multiplies a Proxy by a scalar (commutative: proxy * scalar)
     * @tparam U Scalar arithmetic type (int, float, double, etc.)
     * @param lhs Left operand (proxy)
     * @param rhs Right operand (scalar)
     * @return Result in common type to preserve precision
     */
    template <scalar_arithmetic U>
    [[nodiscard]] friend constexpr std::common_type_t<T, U> operator*(const Derived& lhs, U rhs) noexcept
    {
        return static_cast<T>(lhs) * rhs;
    }

    // ========================================================================
    // Division operators
    // ========================================================================

    /**
     * @brief Divides a composite type by a Proxy
     * @tparam U Composite arithmetic type (complex, matrix, vector, quaternion, etc.)
     * @param lhs Left operand (composite value)
     * @param rhs Right operand (proxy, divisor)
     * @return Result of division in type U
     *
     * @note Works with any type supporting arithmetic operations with T
     */
    template <typename U>
        requires composite_arithmetic<U, T>
    [[nodiscard]] friend constexpr U operator/(U lhs, const Derived& rhs) noexcept(noexcept(lhs / static_cast<T>(rhs)))
    {
        return lhs / static_cast<T>(rhs);
    }

    /**
     * @brief Divides a Proxy by a composite type
     * @tparam U Composite arithmetic type (complex, matrix, vector, quaternion, etc.)
     * @param lhs Left operand (proxy, dividend)
     * @param rhs Right operand (composite value, divisor)
     * @return Result of division in type U
     *
     * @note Works with any type supporting arithmetic operations with T
     */
    template <typename U>
        requires composite_arithmetic<U, T>
    [[nodiscard]] friend constexpr U operator/(const Derived& lhs, U rhs) noexcept(noexcept(static_cast<T>(lhs) / rhs))
    {
        return static_cast<T>(lhs) / rhs;
    }

    /**
     * @brief Divides a scalar by a Proxy
     * @tparam U Scalar arithmetic type (int, float, double, etc.)
     * @param lhs Left operand (scalar, dividend)
     * @param rhs Right operand (proxy, divisor)
     * @return Result in common type to preserve precision
     */
    template <scalar_arithmetic U>
    [[nodiscard]] friend constexpr std::common_type_t<T, U> operator/(U lhs, const Derived& rhs) noexcept
    {
        return lhs / static_cast<T>(rhs);
    }

    /**
     * @brief Divides a Proxy by a scalar
     * @tparam U Scalar arithmetic type (int, float, double, etc.)
     * @param lhs Left operand (proxy, dividend)
     * @param rhs Right operand (scalar, divisor)
     * @return Result in common type to preserve precision
     */
    template <scalar_arithmetic U>
    [[nodiscard]] friend constexpr std::common_type_t<T, U> operator/(const Derived& lhs, U rhs) noexcept
    {
        return static_cast<T>(lhs) / rhs;
    }

    // ========================================================================
    // Addition operators
    // ========================================================================

    /**
     * @brief Adds a composite type and a Proxy (commutative: composite + proxy)
     * @tparam U Composite arithmetic type (complex, matrix, vector, quaternion, etc.)
     * @param lhs Left operand (composite value)
     * @param rhs Right operand (proxy)
     * @return Result of addition in type U
     *
     * @note Works with any type supporting arithmetic operations with T
     */
    template <typename U>
        requires composite_arithmetic<U, T>
    [[nodiscard]] friend constexpr U operator+(U lhs, const Derived& rhs) noexcept(noexcept(lhs + static_cast<T>(rhs)))
    {
        return lhs + static_cast<T>(rhs);
    }

    /**
     * @brief Adds a Proxy and a composite type (commutative: proxy + composite)
     * @tparam U Composite arithmetic type (complex, matrix, vector, quaternion, etc.)
     * @param lhs Left operand (proxy)
     * @param rhs Right operand (composite value)
     * @return Result of addition in type U
     *
     * @note Works with any type supporting arithmetic operations with T
     */
    template <typename U>
        requires composite_arithmetic<U, T>
    [[nodiscard]] friend constexpr U operator+(const Derived& lhs, U rhs) noexcept(noexcept(static_cast<T>(lhs) + rhs))
    {
        return static_cast<T>(lhs) + rhs;
    }

    /**
     * @brief Adds a scalar and a Proxy (commutative: scalar + proxy)
     * @tparam U Scalar arithmetic type (int, float, double, etc.)
     * @param lhs Left operand (scalar)
     * @param rhs Right operand (proxy)
     * @return Result in common type to preserve precision
     */
    template <scalar_arithmetic U>
    [[nodiscard]] friend constexpr std::common_type_t<T, U> operator+(U lhs, const Derived& rhs) noexcept
    {
        return lhs + static_cast<T>(rhs);
    }

    /**
     * @brief Adds a Proxy and a scalar (commutative: proxy + scalar)
     * @tparam U Scalar arithmetic type (int, float, double, etc.)
     * @param lhs Left operand (proxy)
     * @param rhs Right operand (scalar)
     * @return Result in common type to preserve precision
     */
    template <scalar_arithmetic U>
    [[nodiscard]] friend constexpr std::common_type_t<T, U> operator+(const Derived& lhs, U rhs) noexcept
    {
        return static_cast<T>(lhs) + rhs;
    }

    // ========================================================================
    // Subtraction operators
    // ========================================================================

    /**
     * @brief Subtracts a Proxy from a composite type
     * @tparam U Composite arithmetic type (complex, matrix, vector, quaternion, etc.)
     * @param lhs Left operand (composite value, minuend)
     * @param rhs Right operand (proxy, subtrahend)
     * @return Result of subtraction in type U
     *
     * @note Works with any type supporting arithmetic operations with T
     */
    template <typename U>
        requires composite_arithmetic<U, T>
    [[nodiscard]] friend constexpr U operator-(U lhs, const Derived& rhs) noexcept(noexcept(lhs - static_cast<T>(rhs)))
    {
        return lhs - static_cast<T>(rhs);
    }

    /**
     * @brief Subtracts a composite type from a Proxy
     * @tparam U Composite arithmetic type (complex, matrix, vector, quaternion, etc.)
     * @param lhs Left operand (proxy, minuend)
     * @param rhs Right operand (composite value, subtrahend)
     * @return Result of subtraction in type U
     *
     * @note Works with any type supporting arithmetic operations with T
     */
    template <typename U>
        requires composite_arithmetic<U, T>
    [[nodiscard]] friend constexpr U operator-(const Derived& lhs, U rhs) noexcept(noexcept(static_cast<T>(lhs) - rhs))
    {
        return static_cast<T>(lhs) - rhs;
    }

    /**
     * @brief Subtracts a Proxy from a scalar
     * @tparam U Scalar arithmetic type (int, float, double, etc.)
     * @param lhs Left operand (scalar, minuend)
     * @param rhs Right operand (proxy, subtrahend)
     * @return Result in common type to preserve precision
     */
    template <scalar_arithmetic U>
    [[nodiscard]] friend constexpr std::common_type_t<T, U> operator-(U lhs, const Derived& rhs) noexcept
    {
        return lhs - static_cast<T>(rhs);
    }

    /**
     * @brief Subtracts a scalar from a Proxy
     * @tparam U Scalar arithmetic type (int, float, double, etc.)
     * @param lhs Left operand (proxy, minuend)
     * @param rhs Right operand (scalar, subtrahend)
     * @return Result in common type to preserve precision
     */
    template <scalar_arithmetic U>
    [[nodiscard]] friend constexpr std::common_type_t<T, U> operator-(const Derived& lhs, U rhs) noexcept
    {
        return static_cast<T>(lhs) - rhs;
    }

  protected:
    // Protected constructor ensures this is only used as a base class
    constexpr ProxyOperators() noexcept  = default;
    constexpr ~ProxyOperators() noexcept = default;
};

} // namespace shrink_array