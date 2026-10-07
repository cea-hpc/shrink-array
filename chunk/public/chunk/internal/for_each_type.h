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

#include <tuple>   // std::tuple
#include <utility> // std::forward

// Compile-time iteration over a std::tuple<Ts...> of types. Used by consumer
// TUs to drive per-type instantiation from a type list they own (e.g. a
// std::tuple passed to ParamChunk::init_indexes<Types>()).
//
// The library owns no "supported types" list; consumer code is
// responsible for declaring which T it uses.

namespace shrink_array
{

/// @brief Iteration over the types of @p Tuple; only std::tuple is supported.
template <typename Tuple>
struct for_each_type_impl;

/// @brief Specialisation for std::tuple<Ts...>: iterates over Ts.
template <typename... Ts>
struct for_each_type_impl<std::tuple<Ts...>>
{
    /// @brief Call `fn.template operator()<T>()` for each T of the tuple, in order.
    template <typename F>
    static void apply(F&& fn)
    {
        (fn.template operator()<Ts>(), ...);
    }
};

/// @brief Call `fn.template operator()<T>()` for each type T of @p TypeList (a std::tuple), in order.
/// @tparam TypeList std::tuple of the types to iterate over.
/// @param fn Callable with a templated call operator.
template <typename TypeList, typename F>
void for_each_type(F&& fn)
{
    for_each_type_impl<TypeList>::apply(std::forward<F>(fn));
}

} // namespace shrink_array
