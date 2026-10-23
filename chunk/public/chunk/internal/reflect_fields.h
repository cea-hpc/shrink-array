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
 * @file reflect_fields.h
 * @brief The fields of a report struct, by C++26 static reflection: sum two
 *        of them field by field, count them, write them as one JSON object.
 *
 * @details A struct of counters (ChunkStatsThr: 35 of them) is summed, tested
 *          and printed by code that names every field -- a counter added to
 *          the struct and forgotten in one of those lists is silently 0
 *          there. Here the lists come from the struct itself:
 *
 *          - field_count<T>(): the number of non-static data members;
 *          - add_fields(acc, x): acc.f += x.f for every field f;
 *          - append_json(out, obj): `"f": value` for every field, an
 *            arithmetic or bool one, as one JSON object -- what a script
 *            reads without a regular expression per line.
 *
 *          Only where the compiler has reflection (GCC 16, -freflection):
 *          SHRINK_ARRAY_HAS_REFLECT_FIELDS says so. A caller keeps its
 *          hand-written list for the other builds and, in the reflection
 *          builds, static_asserts field_count<T>() against the number of
 *          fields that list names, so a field added later fails the build
 *          until the list has it too.
 */

#if defined(__cpp_impl_reflection)
    #include <meta> // std::meta (reflection)
#endif

#if defined(__cpp_impl_reflection) && defined(__cpp_lib_reflection)
    #define SHRINK_ARRAY_HAS_REFLECT_FIELDS 1
#else
    #define SHRINK_ARRAY_HAS_REFLECT_FIELDS 0
#endif

#if SHRINK_ARRAY_HAS_REFLECT_FIELDS

    #include <cstddef>     // std::size_t
    #include <format>      // std::format_to
    #include <iterator>    // std::back_inserter
    #include <string>      // std::string
    #include <string_view> // std::string_view
    #include <type_traits> // std::is_arithmetic_v, std::is_same_v
    #include <vector>      // std::vector (consteval only: the field lists)

namespace shrink_array
{

namespace reflect_fields_detail
{
/// The non-static data members of @p type, declaration order.
consteval std::vector<std::meta::info> fields_of(std::meta::info type)
{
    return std::meta::nonstatic_data_members_of(type, std::meta::access_context::unchecked());
}
} // namespace reflect_fields_detail

/// @brief The number of non-static data members of T.
template <typename T>
[[nodiscard]] consteval std::size_t field_count()
{
    return reflect_fields_detail::fields_of(^^T).size();
}

/// @brief acc.f += x.f for every field f of T.
template <typename T>
void add_fields(T& acc, const T& x) noexcept
{
    template for (constexpr std::meta::info f : std::define_static_array(reflect_fields_detail::fields_of(^^T)))
    {
        acc.[:f:] += x.[:f:];
    }
}

/**
 * @brief Append @p obj to @p out as one JSON object, `{"f": value, ...}`, in
 *        declaration order, after the members @p prefix (already JSON, with
 *        its trailing comma, e.g. `"variant": "zero_any", `).
 * @details Every field must be arithmetic or bool: a report struct.
 */
template <typename T>
void append_json(std::string& out, const T& obj, std::string_view prefix = {})
{
    out += '{';
    out += prefix;
    bool first = true;
    template for (constexpr std::meta::info f : std::define_static_array(reflect_fields_detail::fields_of(^^T)))
    {
        using F = typename[:std::meta::type_of(f):];
        static_assert(std::is_arithmetic_v<F>, "append_json: a field that is not a number or a bool");
        constexpr std::string_view name = std::meta::identifier_of(f);
        if (!first)
        {
            out += ", ";
        }
        first = false;
        if constexpr (std::is_same_v<F, bool>)
        {
            std::format_to(std::back_inserter(out), "\"{}\": {}", name, obj.[:f:] ? "true" : "false");
        }
        else
        {
            std::format_to(std::back_inserter(out), "\"{}\": {}", name, obj.[:f:]);
        }
    }
    out += '}';
}

} // namespace shrink_array

#endif // SHRINK_ARRAY_HAS_REFLECT_FIELDS
