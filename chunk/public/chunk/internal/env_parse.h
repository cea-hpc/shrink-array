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
 * @file env_parse.h
 * @brief Numbers from environment variables and command-line arguments,
 *        checked -- in place of std::atoi / std::strtol(s, nullptr, 10).
 *
 * atoi reports nothing: "abc" is 0, "12x" is 12, an overflow is undefined
 * behaviour, and `static_cast<unsigned>(atoi("-1"))` is 4294967295. Here the
 * WHOLE text must be a number of the requested type: integers go through
 * std::from_chars (no locale, no errno, no leading space or '+', a '-' refused
 * for an unsigned type, out of range refused); floating point through strtod
 * with the end of the text and ERANGE checked (from_chars for floating point
 * is not in every standard library the project is built with).
 *
 * In the chunk library, which every build of it has (fast_allocators is not
 * even configured in a CHUNK_STD_ALLOC build); fast_allocators' one reading
 * (CHUNK_RECLAIM_MIN_SLABS, alloc_config.h) applies the same rule in place.
 */

#include <cerrno>       // errno, ERANGE (strtod)
#include <charconv>     // std::from_chars
#include <cstdio>       // std::fprintf
#include <cstdlib>      // std::getenv, std::strtod
#include <cstring>      // std::strlen
#include <expected>     // std::expected, std::unexpected (parse_number)
#include <optional>     // std::optional (env_optional)
#include <system_error> // std::errc
#include <type_traits>

namespace shrink_array
{

/// @brief Why a text is not a number of the requested type.
enum class ParseError
{
    empty,          ///< null or ""
    not_a_number,   ///< no number at the start (a '-' for an unsigned type included)
    trailing_chars, ///< a number followed by something else, e.g. "1o"
    out_of_range    ///< a number too large (or too small) for the type
};

/// @brief The reason, as the warnings print it.
[[nodiscard]] constexpr const char* to_string(ParseError e) noexcept
{
    switch (e)
    {
        case ParseError::empty:
            return "empty";
        case ParseError::not_a_number:
            return "not a number";
        case ParseError::trailing_chars:
            return "trailing characters";
        case ParseError::out_of_range:
            return "out of range";
    }
    return "?";
}

/**
 * @brief Parse ALL of @p text as a T.
 * @return the number when @p text is exactly a number of type T; otherwise why
 *         it is not: empty, not a number (a negative value for an unsigned T
 *         included), trailing characters, or out of T's range.
 */
template <typename T>
[[nodiscard]] std::expected<T, ParseError> parse_number(const char* text) noexcept
{
    static_assert(std::is_arithmetic_v<T> && !std::is_same_v<T, bool>, "parse_number: a number type");
    if (text == nullptr || *text == '\0')
        return std::unexpected(ParseError::empty);
    if constexpr (std::is_integral_v<T>)
    {
        const char* const end = text + std::strlen(text);
        T                 v{};
        const auto [ptr, ec] = std::from_chars(text, end, v);
        if (ec == std::errc::result_out_of_range)
            return std::unexpected(ParseError::out_of_range);
        if (ec != std::errc{})
            return std::unexpected(ParseError::not_a_number);
        if (ptr != end)
            return std::unexpected(ParseError::trailing_chars);
        return v;
    }
    else
    {
        char* end      = nullptr;
        errno          = 0;
        const double v = std::strtod(text, &end);
        if (end == text)
            return std::unexpected(ParseError::not_a_number);
        if (*end != '\0')
            return std::unexpected(ParseError::trailing_chars);
        if (errno == ERANGE)
            return std::unexpected(ParseError::out_of_range);
        return static_cast<T>(v);
    }
}

/// @brief The warning of a variable whose value is not the number expected.
inline void warn_invalid_env(const char* name, const char* value, ParseError why) noexcept
{
    std::fprintf(stderr, "[env] %s=\"%s\" is not a valid value (%s): ignored, the default is kept\n", name, value,
                 to_string(why));
}

/**
 * @brief The number in environment variable @p name, if it holds one.
 * @return nothing when the variable is unset or empty, and when its value is
 *         not a number of type T (with a warning on stderr: a typo such as
 *         CHUNK_RECHUNK_SLICES=1o no longer passes unnoticed).
 */
template <typename T>
[[nodiscard]] std::optional<T> env_optional(const char* name) noexcept
{
    const char* env = std::getenv(name);
    if (env == nullptr || *env == '\0')
        return std::nullopt;
    const auto v = parse_number<T>(env);
    if (!v)
    {
        warn_invalid_env(name, env, v.error());
        return std::nullopt;
    }
    return *v;
}

/**
 * @brief The number in environment variable @p name, or @p fallback.
 * @return @p fallback when the variable is unset or empty, and when its value
 *         is not a number of type T (with a warning, see env_optional).
 */
template <typename T>
[[nodiscard]] T env_number(const char* name, T fallback) noexcept
{
    return env_optional<T>(name).value_or(fallback);
}

/**
 * @brief The number in command-line argument @p text (called @p what in the
 *        warning), or @p fallback when it is not a number of type T.
 */
template <typename T>
[[nodiscard]] inline T arg_number(const char* text, T fallback, const char* what) noexcept
{
    const auto v = parse_number<T>(text);
    if (v)
        return *v;
    std::fprintf(stderr, "[arg] %s \"%s\" is not a valid value (%s): ignored, the default is kept\n", what,
                 text != nullptr ? text : "", to_string(v.error()));
    return fallback;
}

/// @brief A switch: @p name set, non-empty, and not starting with '0' -- the
///        rule the library's `env && env[0] && env[0] != '0'` tests spell out.
[[nodiscard]] inline bool env_flag(const char* name) noexcept
{
    const char* env = std::getenv(name);
    return env != nullptr && env[0] != '\0' && env[0] != '0';
}

} // namespace shrink_array
