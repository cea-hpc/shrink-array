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
 * atoi reports nothing ("abc" is 0, "12x" is 12, overflow is UB). Here the WHOLE
 * text must be a number of the requested type: integers via std::from_chars (no
 * leading space or '+', '-' refused for unsigned, range checked); floating point via
 * strtod with end and ERANGE checked (floating from_chars is not in every standard
 * library the project builds with).
 *
 * Lives in the chunk library, present in every build (fast_allocators is not configured
 * in a CHUNK_STD_ALLOC build); alloc_config.h applies the same rule in place.
 */

#include <cerrno>       // errno, ERANGE (strtod)
#include <charconv>     // std::from_chars
#include <cstdlib>      // std::getenv, std::strtod
#include <cstring>      // std::strlen
#include <system_error> // std::errc
#include <type_traits>  // std::is_arithmetic_v, std::is_integral_v, std::is_same_v

#include "chunk/log.h" // CHUNK_LOG

namespace shrink_array
{

/**
 * @brief Parse ALL of @p text as a T.
 * @return true and @p out set when @p text is exactly a number of type T;
 *         false (and @p out untouched) when it is empty, not a number, has
 *         trailing characters, is out of T's range, or negative for an
 *         unsigned T.
 */
template <typename T>
[[nodiscard]] bool parse_number(const char* text, T& out) noexcept
{
    static_assert(std::is_arithmetic_v<T> && !std::is_same_v<T, bool>, "parse_number: a number type");
    if (text == nullptr || *text == '\0')
    {
        return false;
    }
    if constexpr (std::is_integral_v<T>)
    {
        const char* const end = text + std::strlen(text);
        T                 value{};
        const auto [ptr, ec] = std::from_chars(text, end, value);
        if (ec != std::errc{} || ptr != end)
        {
            return false;
        }
        out = value;
        return true;
    }
    else
    {
        char* end           = nullptr;
        errno               = 0;
        const double parsed = std::strtod(text, &end);
        if (end == text || *end != '\0' || errno == ERANGE)
        {
            return false;
        }
        out = static_cast<T>(parsed);
        return true;
    }
}

/// @brief The warning of a variable whose value is not the number expected.
inline void warn_invalid_env(const char* name, const char* value) noexcept
{
    CHUNK_LOG(warn, env, "{}=\"{}\" is not a valid value: ignored, the default is kept", name, value);
}

/**
 * @brief The number in environment variable @p name, or @p fallback.
 * @return @p fallback when the variable is unset or empty, and when its value
 *         is not a number of type T (with a warning on the log, channel env:
 *         a typo such as CHUNK_REUSE_SHARE_DIV=5o does not pass unnoticed).
 */
template <typename T>
[[nodiscard]] T env_number(const char* name, T fallback) noexcept
{
    const char* env = std::getenv(name);
    if (env == nullptr || *env == '\0')
    {
        return fallback;
    }
    T value{};
    if (parse_number(env, value))
    {
        return value;
    }
    warn_invalid_env(name, env);
    return fallback;
}

/**
 * @brief The number in command-line argument @p text (called @p what in the
 *        warning), or @p fallback when it is not a number of type T.
 */
template <typename T>
[[nodiscard]] inline T arg_number(const char* text, T fallback, const char* what) noexcept
{
    T value{};
    if (parse_number(text, value))
    {
        return value;
    }
    CHUNK_LOG(warn, general, "{} \"{}\" is not a valid value: ignored, the default is kept", what,
              text != nullptr ? text : "");
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
