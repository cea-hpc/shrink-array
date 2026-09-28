//
// Created by ballereaup on 30/04/2026.
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


#include <format>  // std::format, std::format_string, std::formatter
#include <string>  // the byte dump of an unformattable argument
#include <cstdio>  // std::fputs, std::fwrite, std::fputc, std::fflush
#include <cstdlib> // std::abort
#include <utility> // std::forward
#include <type_traits>
#include <source_location>
#include <complex>

#include "chunk/debug.h"
#include "chunk/config.h" // build_config

// --- Specialization must be in the std namespace ---

namespace std
{

/// @brief std::format support for std::complex<T>, printed as "(re+imi)".
template <typename T>
struct formatter<std::complex<T>, char>
{
    /// @brief No format specifier is accepted: parsing stops where it starts.
    constexpr auto parse(std::format_parse_context& ctx)
    {
        return ctx.begin();
    }

    /// @brief Write @p value as "(re+imi)".
    auto format(const std::complex<T>& value, std::format_context& ctx) const
    {
        return std::format_to(ctx.out(), "({}+{}i)", value.real(), value.imag());
    }
};
} // namespace std

namespace shrink_array
{
/// @brief True when std::format has a formatter for V (C++20 spelling of
///        C++23's std::formattable: a disabled formatter has no format()).
template <typename V>
concept invariant_formattable =
    requires(std::formatter<V, char> fmt, const V& v, std::format_context& fmt_ctx) { fmt.format(v, fmt_ctx); };

/**
 * @brief The argument std::format will see for @p arg: itself when its type is
 *        formattable, its bytes in hex otherwise.
 *
 * The invariant messages print element values ("expected={} readback={}").
 * An element type is anything trivially copyable with an operator== -- a
 * struct of doubles, an enum class, a std::array -- and most of those have no
 * std::formatter. Rather than ask every such type for one, or drop the value
 * from the message, the byte dump is what a type with no better spelling
 * gets: enough to tell two values apart in a post-mortem, which is what the
 * message is for.
 */
template <typename A>
    requires invariant_formattable<std::remove_cvref_t<A>>
A&& invariant_fmt_arg(A&& arg)
{
    return std::forward<A>(arg);
}

/// @brief Overload for a type with no formatter: the bytes of @p arg in hex, prefixed with "0x".
/// @param arg Value to dump.
/// @return The hex string, two digits per byte in memory order.
template <typename A>
std::string invariant_fmt_arg(A&& arg)
{
    using V = std::remove_cvref_t<A>;
    std::string out;
    out.reserve(2 + (2 * sizeof(V)));
    out += "0x";
    const auto* bytes = reinterpret_cast<const unsigned char*>(&arg);
    for (std::size_t i = 0; i < sizeof(V); ++i)
    {
        static constexpr char hex[] = "0123456789abcdef";
        out += hex[static_cast<unsigned>(bytes[i]) >> 4U];
        out += hex[static_cast<unsigned>(bytes[i]) & 0xFU];
    }
    return out;
}

/// @brief The value as a string, for the diagnostics written with operator<<:
///        std::format's rendering when the type has one, the bytes otherwise.
template <typename V>
std::string value_repr(const V& v)
{
    if constexpr (invariant_formattable<V>)
        return std::format("{}", v);
    else
        return invariant_fmt_arg(v);
}

/// The type std::format deduces for that argument: the reference for an
/// lvalue, the value for anything else (an rvalue reference would not match).
template <typename A, typename R = decltype(invariant_fmt_arg(std::declval<A>()))>
using invariant_fmt_arg_t = std::conditional_t<std::is_lvalue_reference_v<R>, R, std::remove_reference_t<R>>;

template <typename... Args>
[[noreturn]] void invariant_failed(const std::source_location&                      loc,
                                   std::format_string<invariant_fmt_arg_t<Args>...> fmt,
                                   Args&&... args) noexcept
{
    // 1. Determine prefix based on build mode
    constexpr const char* prefix = build_config::is_debug ? "[DEBUG_INVARIANT_FAILURE] " : "[CRITICAL_FAILURE] ";

    // 2. Format and print the message
    try
    {
        const auto msg = std::format(fmt, invariant_fmt_arg(std::forward<Args>(args))...);
        // 1. Print Header and Location
        std::fprintf(stderr, "%sIn file '%s' at line %u\n", prefix, loc.file_name(), loc.line());
        std::fprintf(stderr, "Inside function: %s\n", loc.function_name());
        std::fputs(prefix, stderr);
        std::fwrite(msg.data(), 1, msg.size(), stderr);
        std::fputc('\n', stderr);
    }
    catch (...)
    {
        // Fallback if formatting itself fails (rare)
        std::fputs("Critical error: Failed to format invariant message.\n", stderr);
    }

    // 3. Print the execution backtrace for easier post-mortem analysis
    std::fputs("\n--- Execution Backtrace ---\n", stderr);
    print_backtrace();
    std::fputs("---------------------------\n", stderr);

    // 4. Final flush and immediate termination
    std::fflush(stderr);
    std::abort();
}

// --- The Frontend Macro ---
// This is the ONLY way to capture source_location automatically
// while keeping std::format_string compile-time checks.
#define INVARIANT_FAILED(fmt, ...) shrink_array::invariant_failed(std::source_location::current(), fmt, ##__VA_ARGS__)

} // namespace shrink_array