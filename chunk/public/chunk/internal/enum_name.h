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
 * @file enum_name.h
 * @brief The name of an enumerator, by C++26 static reflection: its
 *        identifier, or without a prefix and upper-cased (`ev_store_value`
 *        with prefix "ev_" gives "STORE_VALUE"). A hand-written table (a
 *        switch from enumerator to string) drifts from its enum; these do not.
 *
 * @details - name_of(value, unknown): the identifier of @p value's
 *            enumerator ("store_value"), @p unknown for a value that names none;
 *          - upper_name_of(value, unknown): the same, upper-cased ("STORE_VALUE");
 *          - upper_enum_name(en, prefix): the name of the enumerator @p en
 *            without @p prefix, upper-cased (consteval);
 *          - names_match<E>(names, prefix, first): true when @p names, an
 *            array of strings, lists E's enumerators in order, as spelled
 *            (lower case, without @p prefix), enumerator k of value
 *            first + k -- for a table that a parser walks (string to enum).
 *
 *          All return strings with static storage, usable at run time.
 */

#include <meta> // std::meta (reflection)

#include <cstddef>     // std::size_t
#include <string>      // std::string (consteval only: the name being built)
#include <string_view> // std::string_view
#include <type_traits> // std::is_enum_v

namespace shrink_array
{

/**
 * @brief The identifier of the enumerator @p en without the leading
 *        @p prefix (kept when absent), upper-cased.
 * @return A null-terminated string with static storage.
 */
consteval const char* upper_enum_name(std::meta::info en, std::string_view prefix)
{
    std::string_view id = std::meta::identifier_of(en);
    if (id.starts_with(prefix))
    {
        id.remove_prefix(prefix.size());
    }
    std::string name(id);
    for (char& c : name)
    {
        if (c >= 'a' && c <= 'z')
        {
            c = static_cast<char>(c - 'a' + 'A');
        }
    }
    return std::define_static_string(name);
}

/**
 * @brief The identifier of the enumerator of E whose value is @p value
 *        ("store_value"), or @p unknown when none has it.
 */
template <typename E>
    requires std::is_enum_v<E>
[[nodiscard]] constexpr const char* name_of(E value, const char* unknown) noexcept
{
    template for (constexpr std::meta::info en : std::define_static_array(std::meta::enumerators_of(^^E)))
    {
        if (value == [:en:])
        {
            constexpr const char* name = std::define_static_string(std::meta::identifier_of(en));
            return name;
        }
    }
    return unknown;
}

/**
 * @brief name_of(), upper-cased ("STORE_VALUE"), or @p unknown when no
 *        enumerator of E has @p value.
 */
template <typename E>
    requires std::is_enum_v<E>
[[nodiscard]] constexpr const char* upper_name_of(E value, const char* unknown) noexcept
{
    template for (constexpr std::meta::info en : std::define_static_array(std::meta::enumerators_of(^^E)))
    {
        if (value == [:en:])
        {
            constexpr const char* name = upper_enum_name(en, "");
            return name;
        }
    }
    return unknown;
}

/**
 * @brief True when @p names lists the enumerators of E, in declaration
 *        order: names[k] is the identifier of enumerator k without the
 *        leading @p prefix, and its value is @p first + k; E has exactly
 *        names.size() enumerators.
 */
template <typename E, typename Names>
consteval bool names_match(const Names& names, std::string_view prefix, long long first = 0)
{
    const auto ens = std::meta::enumerators_of(^^E);
    if (ens.size() != names.size())
    {
        return false;
    }
    for (std::size_t k = 0; k < ens.size(); ++k)
    {
        std::string_view id = std::meta::identifier_of(ens[k]);
        if (id.starts_with(prefix))
        {
            id.remove_prefix(prefix.size());
        }
        if (id != std::string_view(names[k]) ||
            static_cast<long long>(std::meta::extract<E>(ens[k])) != first + static_cast<long long>(k))
        {
            return false;
        }
    }
    return true;
}

} // namespace shrink_array
