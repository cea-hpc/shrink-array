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


#include <cstddef>         // std::size_t
#include <format>          // std::format, std::format_string
#include <string>          // std::string (value_repr)
#include <string_view>     // std::string_view
#include <utility>         // std::forward
#include <source_location> // std::source_location

#include "chunk/config.h"                   // build_config
#include "chunk/log.h"                      // logging: fmt_arg, format_message; CHUNK_LOG
#include "fast_allocators/internal/fatal.h" // logging::fatal_report: the report itself

namespace shrink_array
{
// The arguments are those of the log (fast_allocators/internal/log_core.h):
// anything std::format spells, std::complex as "(re+imi)", a struct member by
// member, "{a: 1, inner: {x: 2.5}}" (C++26 reflection), any other type -- a
// union, an enum, a struct holding a union -- as its bytes in hex: the
// element values an invariant message prints.

/// @brief The value as a string, for the diagnostics written with operator<<:
///        std::format's rendering when the type has one, else its members or
///        its bytes, as the log spells it.
template <typename V>
std::string value_repr(const V& value)
{
    return std::format("{}", logging::detail::fmt_arg(value));
}

namespace detail
{
/**
 * @brief Report the failure @p msg at @p loc and abort: the log's fatal
 *        report (fast_allocators/internal/fatal.h: one block on stderr, the
 *        backtrace, the sink; one reporting thread), headed
 *        [DEBUG_INVARIANT_FAILURE] in a debug build, [CRITICAL_FAILURE]
 *        otherwise.
 */
[[noreturn]] inline void invariant_report(const std::source_location& loc, std::string_view msg) noexcept
{
    constexpr std::string_view prefix = build_config::is_debug ? "[DEBUG_INVARIANT_FAILURE] " : "[CRITICAL_FAILURE] ";
    logging::fatal_report(prefix, loc, msg);
}
} // namespace detail

/**
 * @brief Report a broken invariant -- location, message, backtrace -- and abort.
 * @param loc  Where the check failed (CHUNK_INVARIANT_FAILED passes the caller's).
 * @param fmt  std::format string of the message.
 * @param args Its arguments; one with no formatter is printed member by member, or as its bytes.
 *
 * The message is formatted on the stack (logging::line_max bytes, "..." when
 * cut): no allocation, so a failure inside an allocator, or out of memory,
 * still says what it is.
 */
template <typename... Args>
[[noreturn]] void invariant_failed(const std::source_location&                           loc,
                                   std::format_string<logging::detail::FmtArgT<Args>...> fmt,
                                   Args&&... args) noexcept
{
    char              msg[logging::line_max];
    const std::size_t len = logging::detail::format_message(msg, fmt, std::forward<Args>(args)...);
    detail::invariant_report(loc, std::string_view{msg, len});
}

// --- The Frontend Macro ---
// This is the ONLY way to capture source_location automatically
// while keeping std::format_string compile-time checks.
#define CHUNK_INVARIANT_FAILED(fmt, ...) \
    shrink_array::invariant_failed(std::source_location::current(), fmt, ##__VA_ARGS__)

/**
 * @brief A condition that must not happen, but that a release build survives:
 *        an invariant failure in a check build (build_config::any_check:
 *        message, backtrace, abort), an error on channel @p chan of the log
 *        elsewhere -- after which the caller carries on with its fallback.
 *        One message for both; the failure names the channel ("[drain] ...").
 */
#define CHUNK_CHECK_FAILED(chan, fmt, ...)                             \
    do                                                                 \
    {                                                                  \
        if constexpr (::shrink_array::build_config::any_check)         \
        {                                                              \
            CHUNK_INVARIANT_FAILED("[" #chan "] " fmt, ##__VA_ARGS__); \
        }                                                              \
        else                                                           \
        {                                                              \
            CHUNK_LOG(error, chan, fmt, ##__VA_ARGS__);                \
        }                                                              \
    } while (false)

} // namespace shrink_array
