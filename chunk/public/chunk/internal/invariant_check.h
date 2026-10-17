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


#include <atomic>          // std::atomic_flag (one thread reports)
#include <cstddef>         // std::size_t
#include <cstdio>          // std::fputs, std::fwrite, std::fflush
#include <cstdlib>         // std::abort
#include <cstring>         // std::memcpy
#include <format>          // std::format, std::format_to_n, std::format_string
#include <string>          // std::string (value_repr)
#include <string_view>     // std::string_view
#include <utility>         // std::forward
#include <source_location> // std::source_location
#include <unistd.h>        // pause

#include "chunk/debug.h"  // print_backtrace
#include "chunk/config.h" // build_config
#include "chunk/log.h"    // logging: fmt_arg, format_message, emit_fatal; CHUNK_LOG

namespace shrink_array
{
// The arguments are those of the log (fast_allocators/internal/log_core.h):
// anything std::format spells, std::complex as "(re+imi)", any other type as
// its bytes in hex -- the element values an invariant message prints.

/// @brief The value as a string, for the diagnostics written with operator<<:
///        std::format's rendering when the type has one, the bytes otherwise.
template <typename V>
std::string value_repr(const V& v)
{
    return std::format("{}", logging::detail::fmt_arg(v));
}

namespace detail
{
/// @brief Set by the first thread that reports an invariant failure: the
///        others wait for it to abort the process, so the reports of a race
///        seen by several threads at once do not interleave.
inline std::atomic_flag invariant_reporting{};

/**
 * @brief Report the failure @p msg at @p loc -- on stderr, one block, then
 *        the backtrace; then to the log's sink (logging::detail::emit_fatal)
 *        -- and abort.
 */
[[noreturn]] inline void invariant_report(const std::source_location& loc, std::string_view msg) noexcept
{
    constexpr std::string_view prefix = build_config::is_debug ? "[DEBUG_INVARIANT_FAILURE] " : "[CRITICAL_FAILURE] ";

    // A failure while reporting one (in the backtrace, in the sink): stop there.
    thread_local bool reporting = false;
    if (reporting)
    {
        std::fputs("[CRITICAL_FAILURE] invariant failure while reporting one: ", stderr);
        std::fwrite(msg.data(), 1, msg.size(), stderr);
        std::fputs("\n", stderr);
        std::fflush(stderr);
        std::abort();
    }
    reporting = true;

    // Another thread is reporting: it aborts the process.
    if (invariant_reporting.test_and_set(std::memory_order_acq_rel))
    {
        while (true)
        {
            ::pause();
        }
    }

    // 1. Location and message, in one write. A template's function name can
    //    run to kilobytes: the block is cut (with "...") past 8 KiB.
    char        block[8192];
    std::size_t n = 0;
    try
    {
        const auto res =
            std::format_to_n(block, sizeof(block) - 1, "{0}In file '{1}' at line {2}\nInside function: {3}\n{0}{4}",
                             prefix, loc.file_name(), loc.line(), loc.function_name(), msg);
        n = static_cast<std::size_t>(res.size);
        if (n > sizeof(block) - 1)
        {
            n = sizeof(block) - 1;
            std::memcpy(block + n - 3, "...", 3);
        }
    }
    catch (...)
    {
        n = 0;
    }
    block[n++] = '\n';
    std::fwrite(block, 1, n, stderr);

    // 2. The execution backtrace, for the post-mortem.
    std::fputs("\n--- Execution Backtrace ---\n", stderr);
    print_backtrace();
    std::fputs("---------------------------\n", stderr);
    std::fflush(stderr);

    // 3. The program's sink, or CHUNK_LOG_FILE: last, so that a sink that
    //    misbehaves cannot hide what stderr already has.
    logging::detail::emit_fatal(logging::Record{logging::Level::fatal, logging::Channel::general, msg, loc});

    std::abort();
}
} // namespace detail

/**
 * @brief Report a broken invariant -- location, message, backtrace -- and abort.
 * @param loc  Where the check failed (CHUNK_INVARIANT_FAILED passes the caller's).
 * @param fmt  std::format string of the message.
 * @param args Its arguments; one with no formatter is printed as its bytes.
 *
 * The message is formatted on the stack (logging::line_max bytes, "..." when
 * cut): no allocation, so a failure inside an allocator, or out of memory,
 * still says what it is.
 */
template <typename... Args>
[[noreturn]] void invariant_failed(const std::source_location&                             loc,
                                   std::format_string<logging::detail::fmt_arg_t<Args>...> fmt,
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
