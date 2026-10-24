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

#include <meta> // std::meta::template_arguments_of

// Compile-time iteration over a list of types: the template arguments of a
// class template specialization, std::tuple<Ts...> typically. Used by consumer
// TUs to drive per-type instantiation from a type list they own (e.g. a
// std::tuple passed to ParamChunk::init_indexes<Types>()).
//
// The library owns no "supported types" list; consumer code is
// responsible for declaring which T it uses.

namespace shrink_array
{

/// @brief Call `fn.template operator()<T>()` for each type T of @p TypeList, in order.
/// @tparam TypeList A specialization whose template arguments are the types,
///         std::tuple<Ts...> typically (C++26 reflection walks them).
/// @param fn Callable with a templated call operator.
template <typename TypeList, typename F>
void for_each_type(F&& fn)
{
    static_assert(std::meta::has_template_arguments(^^TypeList),
                  "for_each_type: TypeList must be a specialization of types, std::tuple<Ts...> typically");
    template for (constexpr std::meta::info type :
                  std::define_static_array(std::meta::template_arguments_of(^^TypeList)))
    {
        fn.template operator()<typename[:type:]>();
    }
}

} // namespace shrink_array
