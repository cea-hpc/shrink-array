/*
 * Copyright 2026 CEA (www.cea.fr)
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
 * @file same_value.h
 * @brief The equality every compression and skipped-write decision uses:
 *        "reading back the one gives the other", not operator==.
 *
 * @details operator== on floating point says -0.0 == +0.0 (and NaN != NaN). A
 *          chunk compared that way compresses a run of -0.0 to the constant
 *          +0.0, and a write of -0.0 onto a +0.0 is skipped as a no-op: the sign
 *          is lost, and 1/x, std::signbit, std::copysign, std::atan2 and the
 *          branch cuts of std::complex change result. The library promises that
 *          an element reads back what was written, so every decision of the
 *          form "this value is already there / this run is the constant" goes
 *          through same_value(), never through ==:
 *           - float, double: equal object representations (std::bit_cast to the
 *             unsigned integer of the same size). -0.0 and +0.0 differ; two NaNs
 *             with the same payload are the same value (a run of them compresses
 *             and reads back bit for bit). Same cost as the floating-point
 *             compare, vectorised as an integer one.
 *           - long double (and any other floating-point type): == and the same
 *             sign bit. The x87 format has padding bytes, so no bit_cast; a NaN
 *             is never the same as anything (as before: never compressed).
 *           - std::complex<F>, std::array<E, N>: part by part, element by element.
 *           - anything else: operator==. An aggregate of floating-point members
 *             with a defaulted operator== compares them with ==, so it keeps
 *             losing the sign of a zero member; specialise ValueIdentity<T> for
 *             it (see below) to make it exact.
 *
 *          ValueIdentityLess<T> is the strict weak order that agrees with it,
 *          for the containers keyed by value (value_less.h).
 */

#include <array>       // std::array
#include <cmath>       // std::signbit
#include <complex>     // std::complex
#include <cstddef>     // std::size_t
#include <cstdint>     // std::uint32_t, std::uint64_t
#include <cstring>     // std::memcpy
#include <type_traits> // std::is_floating_point_v, std::conditional_t

#if __has_include(<bit>)
    #include <bit> // std::bit_cast
#endif

namespace shrink_array
{

namespace internal
{
/// @brief The unsigned integer with the object representation of @p v (float, double).
template <typename F>
[[nodiscard]] inline auto float_bits(F v) noexcept
{
    using U = std::conditional_t<sizeof(F) == 4, std::uint32_t, std::uint64_t>;
    static_assert(sizeof(U) == sizeof(F));
#if defined(__cpp_lib_bit_cast)
    return std::bit_cast<U>(v);
#else
    U u;
    std::memcpy(&u, &v, sizeof(U));
    return u;
#endif
}

/// @brief Floating-point types compared through their bits: no padding, 4 or 8 bytes.
template <typename F>
inline constexpr bool bitwise_float_v = std::is_floating_point_v<F> && (sizeof(F) == 4 || sizeof(F) == 8);

template <typename T>
struct is_std_array : std::false_type
{
};
template <typename E, std::size_t N>
struct is_std_array<std::array<E, N>> : std::true_type
{
};

template <typename T>
struct is_std_complex : std::false_type
{
};
template <typename F>
struct is_std_complex<std::complex<F>> : std::true_type
{
};
} // namespace internal

/**
 * @brief Customisation point of same_value() for an element type.
 *
 * The primary template covers the arithmetic types, bool, enums, std::complex
 * and std::array (see the file comment). For an aggregate of floating-point
 * members, specialise it to compare the members with same_value():
 * @code
 * template <> struct shrink_array::ValueIdentity<Point3> {
 *     static bool same(const Point3& a, const Point3& b) noexcept {
 *         return same_value(a.x, b.x) && same_value(a.y, b.y) && same_value(a.z, b.z);
 *     }
 * };
 * @endcode
 * A specialisation must be an equivalence relation that implies operator==
 * (the library still uses == nowhere for its decisions, but a value it calls
 * "the same" is read back in place of the other).
 */
template <typename T>
struct ValueIdentity
{
    /// @brief Whether @p a and @p b are the same stored value (see same_value()).
    [[nodiscard]] static bool same(const T& a, const T& b) noexcept;
};

/**
 * @brief Whether @p a and @p b are the same stored value: reading back one gives
 *        the other. The comparison of every compression and skipped-write decision.
 */
template <typename T>
[[nodiscard]] inline bool same_value(const T& a, const T& b) noexcept
{
    return ValueIdentity<T>::same(a, b);
}

/**
 * @brief same_value() across two types: the packed bool (Bit) against bool, a
 *        value against a constant of another type. @p b is converted to the
 *        type of @p a when same_value() differs from == for it (floating
 *        point, std::complex), compared with == otherwise.
 */
template <typename T, typename U>
    requires(!std::is_same_v<T, U>)
[[nodiscard]] inline bool same_value(const T& a, const U& b) noexcept
{
    if constexpr (std::is_floating_point_v<T> || internal::is_std_complex<T>::value || internal::is_std_array<T>::value)
    {
        return same_value(a, static_cast<T>(b));
    }
    else
    {
        return a == b;
    }
}

template <typename T>
inline bool ValueIdentity<T>::same(const T& a, const T& b) noexcept
{
    if constexpr (internal::bitwise_float_v<T>)
    {
        return internal::float_bits(a) == internal::float_bits(b);
    }
    else if constexpr (std::is_floating_point_v<T>)
    {
        return a == b && std::signbit(a) == std::signbit(b);
    }
    else if constexpr (internal::is_std_complex<T>::value)
    {
        return same_value(a.real(), b.real()) && same_value(a.imag(), b.imag());
    }
    else if constexpr (internal::is_std_array<T>::value)
    {
        for (std::size_t i = 0; i < a.size(); ++i)
        {
            if (!same_value(a[i], b[i]))
            {
                return false;
            }
        }
        return true;
    }
    else
    {
        return a == b;
    }
}

/**
 * @brief A strict weak order whose equivalence is same_value(), for the types
 *        same_value() treats specially (floating point, std::complex,
 *        std::array of those).
 *
 * float/double: the order of the bit patterns (total, NaNs included; not the
 * numeric order -- the containers keyed with it only need a valid order).
 * long double: the numeric order, -0.0 before +0.0. complex and array:
 * lexicographic.
 */
template <typename T>
[[nodiscard]] inline bool identity_less(const T& a, const T& b) noexcept
{
    if constexpr (internal::bitwise_float_v<T>)
    {
        return internal::float_bits(a) < internal::float_bits(b);
    }
    else if constexpr (std::is_floating_point_v<T>)
    {
        return a < b || (a == b && std::signbit(a) && !std::signbit(b));
    }
    else if constexpr (internal::is_std_complex<T>::value)
    {
        if (identity_less(a.real(), b.real()))
        {
            return true;
        }
        if (identity_less(b.real(), a.real()))
        {
            return false;
        }
        return identity_less(a.imag(), b.imag());
    }
    else
    {
        static_assert(internal::is_std_array<T>::value, "identity_less: floating point, std::complex or std::array");
        for (std::size_t i = 0; i < a.size(); ++i)
        {
            if (identity_less(a[i], b[i]))
            {
                return true;
            }
            if (identity_less(b[i], a[i]))
            {
                return false;
            }
        }
        return false;
    }
}

namespace internal
{
/// @brief Whether same_value() differs from == for T, so that ValueLess must order by identity_less().
template <typename T>
struct has_identity_order : std::is_floating_point<T>
{
};
template <typename F>
struct has_identity_order<std::complex<F>> : std::true_type
{
};
template <typename E, std::size_t N>
struct has_identity_order<std::array<E, N>> : has_identity_order<E>
{
};
} // namespace internal

} // namespace shrink_array
