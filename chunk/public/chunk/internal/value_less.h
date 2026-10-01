//
// Created by ballereaup on 28/05/25.
//
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

#include <concepts>
#include <cstring>     // std::memcmp
#include <type_traits> // std::has_unique_object_representations_v
#include <utility>     // std::unreachable, __cpp_lib_unreachable

#include "chunk/chunk_def.h" // ComplexType
#include "chunk/internal/complex_less.h"
#include "chunk/internal/in_t.h" // in_t<T>: the compared values

namespace shrink_array
{
/**
 * @brief The strict weak order chunk2 keys its second-value tables with.
 *
 * chunk2 (Chunk2 / Proxy2) registers each declared second value in a
 * std::map keyed by the value -- one table of candidate chunks per value --
 * so it needs an ordering the value type itself may not provide. In order of
 * preference:
 *  - std::complex: ComplexLess, real then imaginary (no operator< in std);
 *  - a type with operator<: that one (arithmetic, enum, std::array, a struct
 *    with a defaulted operator<=>);
 *  - a type whose every bit is value (has_unique_object_representations: no
 *    padding, no floating point): the order of its bytes, which is a valid
 *    strict order that agrees with operator==;
 *  - anything else -- a struct of doubles without operator<=> -- is refused
 *    at compile time: its padding or NaN bits would make two equal values two
 *    keys. Declaring `auto operator<=>(const T&) const = default;` is the fix.
 *
 * chunk (v1) and chunk2a do not order values (chunk2a finds each chunk's
 * second value by scanning); the element contract there is == and T{} only.
 */
template <typename T>
struct ValueLess
{
    /// @brief Compare two values with the ordering selected for T (see above).
    /// @return true when @p lhs orders before @p rhs.
    bool operator()(in_t<T> lhs, in_t<T> rhs) const noexcept
    {
        if constexpr (ComplexType<T>)
            return ComplexLess<typename T::value_type>{}(lhs, rhs);
        else if constexpr (std::totally_ordered<T>)
            return lhs < rhs;
        else if constexpr (std::has_unique_object_representations_v<T>)
            return std::memcmp(&lhs, &rhs, sizeof(T)) < 0;
        else
            static_assert(std::totally_ordered<T>,
                          "chunk2 orders second values: this element type needs an operator< "
                          "(a defaulted operator<=> will do), or no padding and no floating point");
        // Unreachable: the static_assert above always fires for a type reaching
        // this branch. Present only for nvc++ (EDG), whose flow analysis does
        // not prune discarded `if constexpr` branches and otherwise warns
        // "missing return statement" [implicit_return_from_non_void_function].
        // std::unreachable() is [[noreturn]], which ends the path for EDG and
        // also tells the optimiser the branch is dead. Keyed on the LIBRARY
        // feature macro, not __cplusplus: -std=c++23 alone does not guarantee
        // the standard library provides it.
#if defined(__cpp_lib_unreachable)
        std::unreachable();
#else
        return false;
#endif
    }
};

} // namespace shrink_array
