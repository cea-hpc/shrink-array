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

/**
 * @file fatal.h
 * @brief The fatal level of the log: report -- location, message, backtrace
 *        -- and abort, whatever CHUNK_LOG says.
 *
 * One report for both libraries: chunk's CHUNK_INVARIANT_FAILED
 * (chunk/internal/invariant_check.h) and fast_allocators' FAST_ALLOC_FATAL
 * end here.
 *
 * - On stderr, one block (location and message in one write), then the
 *   backtrace; then the record to the program's sink (set_sink()) or to
 *   CHUNK_LOG_FILE (log_core.h's emit_fatal), last, so a sink that
 *   misbehaves cannot hide what stderr already has.
 * - One thread reports: the others, failing at the same time (a race seen by
 *   several threads), wait for it to abort the process.
 * - A failure while reporting one (in the backtrace, in the sink) stops at
 *   once.
 * - The message is formatted on the stack (log_core.h's format_message): an
 *   allocator that fails, or a process out of memory, still says what
 *   happened. The backtrace itself allocates (backtrace_symbols, the
 *   demangling): it is printed after the message for that reason.
 */

#include <array>           // std::array (the backtrace frames)
#include <atomic>          // std::atomic_flag (one thread reports)
#include <cstddef>         // std::size_t
#include <cstdio>          // std::fprintf, std::fputs, std::fwrite, std::fflush
#include <cstdlib>         // std::abort, std::free
#include <cstring>         // std::memcpy
#include <format>          // std::format_to_n, std::format_string
#include <source_location> // std::source_location
#include <string_view>     // std::string_view
#include <utility>         // std::forward
#include <cxxabi.h>        // abi::__cxa_demangle
#include <execinfo.h>      // backtrace, backtrace_symbols
#include <unistd.h>        // pause

#include "fast_allocators/internal/log_core.h" // format_message, emit_fatal, fmt_arg_t

namespace fast_alloc::logging
{
/// @brief Print the calling thread's backtrace (up to 100 frames) on stderr,
///        demangling names when possible.
inline void print_backtrace() noexcept
{
    constexpr int                 max_frames = 100;
    std::array<void*, max_frames> buffer{};
    const int                     nptrs   = ::backtrace(buffer.data(), max_frames);
    char**                        symbols = ::backtrace_symbols(buffer.data(), nptrs);
    if (symbols == nullptr)
    {
        return; // backtrace_symbols could not allocate
    }

    for (int i = 0; i < nptrs; ++i)
    {
        const char* mangled_name = nullptr;
        char*       offset_begin = nullptr;
        char*       offset_end   = nullptr;

        for (char* p = symbols[i]; *p != '\0'; ++p)
        {
            if (*p == '(')
            {
                mangled_name = p + 1;
            }
            else if (*p == '+')
            {
                offset_begin = p;
            }
            else if (*p == ')')
            {
                offset_end = p;
                break;
            }
        }

        if (mangled_name != nullptr && offset_begin != nullptr && offset_end != nullptr && mangled_name < offset_begin)
        {
            *offset_begin = '\0';
            *offset_end   = '\0';

            int   status    = -1;
            char* real_name = abi::__cxa_demangle(mangled_name, nullptr, nullptr, &status);

            if (status == 0 && real_name != nullptr)
            {
                std::fprintf(stderr, "%s : %s+%s\n", symbols[i], real_name, offset_begin + 1);
                std::free(real_name);
            }
            else
            {
                std::fprintf(stderr, "%s : %s+%s\n", symbols[i], mangled_name, offset_begin + 1);
            }
        }
        else
        {
            std::fprintf(stderr, "%s\n", symbols[i]);
        }
    }

    std::free(static_cast<void*>(symbols));
}

namespace detail
{
/// @brief Set by the first thread that reports a fatal failure: the others
///        wait for it to abort the process, so the reports of a race seen by
///        several threads at once do not interleave.
inline std::atomic_flag fatal_reporting{};
} // namespace detail

/**
 * @brief Report the failure @p msg at @p loc -- on stderr, one block headed
 *        by @p prefix, then the backtrace; then to the log's sink -- and
 *        abort. See the file comment.
 */
[[noreturn]] inline void
fatal_report(std::string_view prefix, const std::source_location& loc, std::string_view msg) noexcept
{
    // A failure while reporting one (in the backtrace, in the sink): stop there.
    thread_local bool reporting = false;
    if (reporting)
    {
        std::fputs("[CRITICAL_FAILURE] fatal failure while reporting one: ", stderr);
        std::fwrite(msg.data(), 1, msg.size(), stderr);
        std::fputs("\n", stderr);
        std::fflush(stderr);
        std::abort();
    }
    reporting = true;

    // Another thread is reporting: it aborts the process.
    if (detail::fatal_reporting.test_and_set(std::memory_order_acq_rel))
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

    // 3. The program's sink, or CHUNK_LOG_FILE.
    detail::emit_fatal(Record{Level::fatal, Channel::general, msg, loc});

    std::abort();
}

/**
 * @brief Format the message -- on the stack, arguments spelled as the log
 *        spells them (fmt_arg) -- and fatal_report() it.
 */
template <typename... Args>
[[noreturn]] void fatal(std::string_view                             prefix,
                        const std::source_location&                  loc,
                        std::format_string<detail::FmtArgT<Args>...> fmt,
                        Args&&... args) noexcept
{
    char              msg[line_max];
    const std::size_t len = detail::format_message(msg, fmt, std::forward<Args>(args)...);
    fatal_report(prefix, loc, std::string_view{msg, len});
}
} // namespace fast_alloc::logging

/// @brief fast_allocators' fatal failure (an allocation the process cannot
///        go on without, a broken slab): message, backtrace, abort -- whatever
///        CHUNK_LOG says. A std::format string checked at compile time.
#define FAST_ALLOC_FATAL(fmt, ...) \
    ::fast_alloc::logging::fatal("[CRITICAL_FAILURE] ", std::source_location::current(), fmt, ##__VA_ARGS__)
