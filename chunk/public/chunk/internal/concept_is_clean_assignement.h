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
 * See the License for the language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <complex>
#include <concepts> // std::convertible_to
#include <type_traits>

#include "chunk/bit.h" // Bit: the packed bool, assigned from bool

namespace shrink_array
{

// --- 1. Helper to detect if a type is a Proxy by looking for value_type ---
// We exclude std::complex from being identified as a Proxy
template <typename T>
concept IsLibraryProxy = requires { typename std::remove_cvref_t<T>::is_shrink_proxy_tag; };

// --- 2. Extract the underlying type (Unwrapper) ---
template <typename T>
struct get_base
{
    using type = std::remove_cvref_t<T>;
};

template <IsLibraryProxy T>
struct get_base<T>
{
    using type = std::remove_cvref_t<T>::value_type;
};

template <typename T>
using base_t = get_base<T>::type;

// --- 3. Simple Complex Detection ---
template <typename T>
struct is_cplx : public std::false_type
{
};
/// @brief std::complex<T> is complex.
template <typename T>
struct is_cplx<std::complex<T>> : public std::true_type
{
};

// --- 4. The Core Concept ---
template <typename T, typename U>
concept IsCleanAssignment =
    // Rule 1: Perfect match (e.g., double = Proxy<double> or complex = complex)
    std::is_same_v<std::remove_cvref_t<T>, base_t<U>> ||

    // Rule 2: Floating point (double/float) receiving any integer
    (std::is_floating_point_v<std::remove_cvref_t<T>> && std::is_integral_v<base_t<U>>) ||

    // Rule 3: Complex receiving its own scalar type (e.g., complex<double> = double/Proxy<double>)
    // This is the rule that was failing for you. We make it very explicit.
    (is_cplx<std::remove_cvref_t<T>>::value &&
     std::is_same_v<typename std::remove_cvref_t<T>::value_type, base_t<U>>) ||

    // Rule 4: Complex receiving any integer (e.g., complex<double> = 0)
    (is_cplx<std::remove_cvref_t<T>>::value && std::is_integral_v<base_t<U>>) ||

    // Rule 5: Small integers from standard int
    ((std::is_same_v<std::remove_cvref_t<T>, short> || std::is_same_v<std::remove_cvref_t<T>, unsigned char> ||
      std::is_same_v<std::remove_cvref_t<T>, char>) &&
     std::is_same_v<base_t<U>, int>) ||

    // Rule 6: the packed bool (ArrayChunkBits) receiving a bool
    (shrink_array::is_packed_bit_v<std::remove_cvref_t<T>> && std::is_same_v<base_t<U>, bool>);

template <typename T, typename U>
concept IsCleanCompoundOp = IsCleanAssignment<T, U>;
/**
 * @brief Defines which shift operations (<<=, >>=) are considered "safe".
 * 1. The array element (T) must be an integral type.
 * 2. The shift amount (U) must be an integral type after unwrapping Proxy.
 */
template <typename T, typename U>
concept IsCleanShiftOp = std::is_integral_v<std::remove_cvref_t<T>> && std::is_integral_v<base_t<U>>;

// --- 5. What the element type itself can do ---
// The compound operators of the proxies exist for the element types that have
// the operation: an aggregate (a struct of doubles, see the README) gets
// assignment, reads and comparison, and `tab[i] += p` on it is refused at the
// call ("no viable overloaded '+='") rather than three templates down, inside
// std::plus or the atomic apply. std::complex has the four arithmetic ones and
// no ++/--, bool has the arithmetic ones (through int) and no ++.
template <typename T>
concept ElementAddable = requires(T lhs, T rhs) {
    { lhs + rhs } -> std::convertible_to<T>;
};
template <typename T>
concept ElementSubtractable = requires(T lhs, T rhs) {
    { lhs - rhs } -> std::convertible_to<T>;
};
template <typename T>
concept ElementMultipliable = requires(T lhs, T rhs) {
    { lhs * rhs } -> std::convertible_to<T>;
};
template <typename T>
concept ElementDivisible = requires(T lhs, T rhs) {
    { lhs / rhs } -> std::convertible_to<T>;
};
template <typename T>
concept ElementModulable = requires(T lhs, T rhs) {
    { lhs % rhs } -> std::convertible_to<T>;
};
template <typename T>
concept ElementIncrementable = requires(T elem) {
    ++elem;
    --elem;
};

} // namespace shrink_array