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

// The double guard of a header the module reads (chunk/internal/module_export.h).
#if defined(SHRINK_ARRAY_MODULE_DEPS) ? !defined(SHRINK_ARRAY_INTERNAL_PRINT_H_DEPS) \
                                      : !defined(SHRINK_ARRAY_INTERNAL_PRINT_H)
    #if defined(SHRINK_ARRAY_MODULE_DEPS)
        #define SHRINK_ARRAY_INTERNAL_PRINT_H_DEPS
    #else
        #define SHRINK_ARRAY_INTERNAL_PRINT_H
    #endif

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
 * Called unqualified -- `println("{} chunks", n);` -- from the library and from
 * code under `using namespace shrink_array;`; elsewhere after
 * `using shrink_array::println;`.
 *
 * Where `<print>` provides them (__cpp_lib_print) they ARE std::print and
 * std::println (using-declarations): no second overload set that an
 * unqualified call could find next to the standard one by ADL.
 */

    #include <version> // __cpp_lib_print

    #if defined(__cpp_lib_print)
        #include <print> // std::print, std::println
    #else
        #include <cstdio>  // std::FILE, std::fwrite, stdout
        #include <format>  // std::format, std::format_string
        #include <string>  // std::string
        #include <utility> // std::forward
    #endif

    #include "chunk/internal/module_export.h" // SHRINK_ARRAY_EXPORT (print, println)

    #ifndef SHRINK_ARRAY_MODULE_DEPS // the module's global fragment reads the #includes only

namespace shrink_array
{
        #if defined(__cpp_lib_print)
SHRINK_ARRAY_EXPORT using std::print;
SHRINK_ARRAY_EXPORT using std::println;
        #else
/// @brief std::print(stream, fmt, args...): format, then one fwrite.
SHRINK_ARRAY_EXPORT template <typename... Args>
void print(std::FILE* stream, std::format_string<Args...> fmt, Args&&... args)
{
    const std::string text = std::format(fmt, std::forward<Args>(args)...);
    std::fwrite(text.data(), 1, text.size(), stream);
}

/// @brief std::print(fmt, args...): to stdout.
SHRINK_ARRAY_EXPORT template <typename... Args>
void print(std::format_string<Args...> fmt, Args&&... args)
{
    print(stdout, fmt, std::forward<Args>(args)...);
}

/// @brief std::println(stream, fmt, args...): the text, then a newline.
SHRINK_ARRAY_EXPORT template <typename... Args>
void println(std::FILE* stream, std::format_string<Args...> fmt, Args&&... args)
{
    std::string text = std::format(fmt, std::forward<Args>(args)...);
    text.push_back('\n');
    std::fwrite(text.data(), 1, text.size(), stream);
}

/// @brief std::println(fmt, args...): to stdout.
SHRINK_ARRAY_EXPORT template <typename... Args>
void println(std::format_string<Args...> fmt, Args&&... args)
{
    println(stdout, fmt, std::forward<Args>(args)...);
}

/// @brief std::println(): a newline on stdout.
SHRINK_ARRAY_EXPORT inline void println()
{
    std::fputc('\n', stdout);
}
        #endif
} // namespace shrink_array

    #endif // SHRINK_ARRAY_MODULE_DEPS
#endif     // SHRINK_ARRAY_INTERNAL_PRINT_H
