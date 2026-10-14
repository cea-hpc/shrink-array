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
 * @file print.h
 * @brief shrink_array::print / println: std::print / std::println in every
 *        build -- the C++23 ones where the library has them, the same
 *        signatures over std::format (C++20) otherwise.
 *
 * The library, the examples and the harnesses print through these rather than
 * through printf: the format is checked at compile time, a type needs no
 * conversion letter, and the develop (C++20) and cpp23 branches write the
 * same lines -- a merge between them does not stop on every message.
 *
 * Where <print> provides them (__cpp_lib_print) they ARE std::print and
 * std::println (using-declarations): no second overload set that an
 * unqualified call could find next to the standard one by ADL.
 */

#include <version> // __cpp_lib_print

#if defined(__cpp_lib_print)
    #include <print> // std::print, std::println

namespace shrink_array
{
using std::print;
using std::println;
} // namespace shrink_array

#else
    #include <cstdio>  // std::FILE, std::fwrite, stdout
    #include <format>  // std::format, std::format_string
    #include <string>  // std::string
    #include <utility> // std::forward

namespace shrink_array
{
/// @brief std::print(stream, fmt, args...): format, then one fwrite.
template <typename... Args>
void print(std::FILE* stream, std::format_string<Args...> fmt, Args&&... args)
{
    const std::string text = std::format(fmt, std::forward<Args>(args)...);
    std::fwrite(text.data(), 1, text.size(), stream);
}

/// @brief std::print(fmt, args...): to stdout.
template <typename... Args>
void print(std::format_string<Args...> fmt, Args&&... args)
{
    print(stdout, fmt, std::forward<Args>(args)...);
}

/// @brief std::println(stream, fmt, args...): the text, then a newline.
template <typename... Args>
void println(std::FILE* stream, std::format_string<Args...> fmt, Args&&... args)
{
    std::string text = std::format(fmt, std::forward<Args>(args)...);
    text.push_back('\n');
    std::fwrite(text.data(), 1, text.size(), stream);
}

/// @brief std::println(fmt, args...): to stdout.
template <typename... Args>
void println(std::format_string<Args...> fmt, Args&&... args)
{
    println(stdout, fmt, std::forward<Args>(args)...);
}

/// @brief std::println(): a newline on stdout.
inline void println()
{
    std::fputc('\n', stdout);
}
} // namespace shrink_array

#endif
