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
 *        identifier without a prefix, upper-cased (`ev_store_value` with
 *        prefix "ev_" gives "STORE_VALUE").
 *
 * @details A hand-written name table (a switch from enumerator to string)
 *          drifts from its enum: a case forgotten, a name misspelled. Here:
 *
 *          - upper_enum_name(en, prefix): the name of the enumerator @p en,
 *            a string with static storage, usable at run time;
 *          - table_matches<E>(table, prefix): true when @p table gives, for
 *            every enumerator of E, that same name.
 *
 *          Only where the compiler has reflection (GCC 16, -freflection):
 *          SHRINK_ARRAY_HAS_ENUM_NAME says so. A caller keeps its switch for
 *          the other builds and, in the reflection builds, static_asserts
 *          table_matches() so the switch stays complete and spelled alike.
 */

#if defined(__cpp_impl_reflection)
    #include <meta> // std::meta (reflection)
#endif

#if defined(__cpp_impl_reflection) && defined(__cpp_lib_reflection)
    #define SHRINK_ARRAY_HAS_ENUM_NAME 1
#else
    #define SHRINK_ARRAY_HAS_ENUM_NAME 0
#endif

#if SHRINK_ARRAY_HAS_ENUM_NAME

    #include <string>      // std::string (consteval only: the name being built)
    #include <string_view> // std::string_view

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
 * @brief True when @p table (a constexpr function from E to its name) gives,
 *        for every enumerator of E, upper_enum_name(enumerator, @p prefix).
 */
template <typename E, typename Table>
consteval bool table_matches(Table table, std::string_view prefix)
{
    for (std::meta::info en : std::meta::enumerators_of(^^E))
    {
        if (std::string_view(table(std::meta::extract<E>(en))) != upper_enum_name(en, prefix))
        {
            return false;
        }
    }
    return true;
}

} // namespace shrink_array

#endif // SHRINK_ARRAY_HAS_ENUM_NAME
