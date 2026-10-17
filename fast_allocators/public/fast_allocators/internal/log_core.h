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
 * @file log_core.h
 * @brief The one log of the library: a message has a level and a channel,
 *        CHUNK_LOG (the environment variable) says which get through, and
 *        every one that does reaches a single sink.
 *
 * Here, in fast_allocators, because both libraries log and fast_allocators
 * must build without chunk; chunk's headers always see this directory (as
 * they do svalloc.h), and chunk/log.h spells the namespace
 * shrink_array::logging.
 *
 *     CHUNK_LOG(warn, env, "{}=\"{}\" is not a valid value: ignored", name, value);
 *     CHUNK_LOG_FIRST_N(16, error, drain, "thr={} has {} chunks with deferred ops", rank, n);
 *     CHUNK_LOG_POW2(warn, recycle, "dropped FOREIGN block {}", ptr);
 *
 * - Filtering: CHUNK_LOG=warn (the default), CHUNK_LOG=warn,rechunk=debug,
 *   CHUNK_LOG=off. Items apply left to right: a bare level sets every
 *   channel, `channel=level` one. Read once, at the first message;
 *   configure() applies a specification on top of it.
 * - Cost: a message filtered out is one relaxed load and a compare; its
 *   arguments are not evaluated. CHUNK_LOG_MAX_LEVEL (0 fatal .. 5 trace, 5
 *   by default) removes the levels above it from the build altogether.
 * - Output: the message is formatted on the stack (std::format_to_n into
 *   line_max bytes, "..." when cut): no allocation, so a message can come
 *   from an allocator or from the teardown. It goes to the sink of
 *   set_sink(), or by default to stderr (CHUNK_LOG_FILE=path: to that file,
 *   %p replaced by the pid) as one line, written by one fwrite:
 *       [shrink_array][rechunk][W][12345] message
 *   (channel, level letter, kernel thread id), so lines from many threads
 *   do not interleave.
 * - Not here: the fatal level. An invariant failure is CHUNK_INVARIANT_FAILED
 *   (chunk/internal/invariant_check.h), which no setting can turn off.
 */

#include <array>           // std::array (the thresholds, the names)
#include <atomic>          // std::atomic
#include <bit>             // std::has_single_bit (CHUNK_LOG_POW2)
#include <cstddef>         // std::size_t
#include <cstdint>         // std::uint8_t, std::uint64_t
#include <cstdio>          // std::FILE, std::fopen, std::fwrite, std::fflush, stderr
#include <cstdlib>         // std::getenv
#include <cstring>         // std::memcpy
#include <format>          // std::format_to_n, std::format_string
#include <mutex>           // std::mutex, std::once_flag, std::call_once, std::scoped_lock
#include <source_location> // std::source_location
#include <string_view>     // std::string_view
#include <utility>         // std::forward
#include <unistd.h>        // getpid, gettid

#ifndef CHUNK_LOG_MAX_LEVEL
    #define CHUNK_LOG_MAX_LEVEL 5 // trace: every message compiled in, CHUNK_LOG filters at run time
#endif

namespace fast_alloc::logging
{
/// @brief Severity, most severe first. fatal is CHUNK_INVARIANT_FAILED's, never CHUNK_LOG's.
enum class Level : std::uint8_t
{
    fatal,
    error,
    warn,
    info,
    debug,
    trace
};

/// @brief What a message is about: the unit CHUNK_LOG filters on. One list for both libraries.
enum class Channel : std::uint8_t
{
    general,
    env,
    alloc,
    arena,
    pool,
    rechunk,
    drain,
    compress,
    recycle,
    ledger,
    ompt,
    affinity,
    advisor,
    count_
};

inline constexpr std::size_t n_channels = static_cast<std::size_t>(Channel::count_);
inline constexpr std::size_t n_levels   = 6;

/// @brief Longest message, in bytes, before it is cut (with "...").
inline constexpr std::size_t line_max = 1024;

inline constexpr std::array<std::string_view, n_channels> channel_names{
    "general",  "env",     "alloc",  "arena", "pool",     "rechunk", "drain",
    "compress", "recycle", "ledger", "ompt",  "affinity", "advisor",
};
inline constexpr std::array<std::string_view, n_levels> level_names{"fatal", "error", "warn", "info", "debug", "trace"};

/// @brief The name CHUNK_LOG spells channel @p c with.
[[nodiscard]] constexpr std::string_view name(Channel c) noexcept
{
    return channel_names[static_cast<std::size_t>(c)];
}

/// @brief The name CHUNK_LOG spells level @p l with.
[[nodiscard]] constexpr std::string_view name(Level l) noexcept
{
    return level_names[static_cast<std::size_t>(l)];
}

/// @brief True when messages of level @p l are compiled in (CHUNK_LOG_MAX_LEVEL).
[[nodiscard]] constexpr bool compiled(Level l) noexcept
{
    return static_cast<int>(l) <= CHUNK_LOG_MAX_LEVEL;
}

/// @brief One message, as a sink receives it: @p message has no prefix and no newline.
struct Record
{
    Level                level;
    Channel              channel;
    std::string_view     message;
    std::source_location location;
};

/// @brief A replacement for the default sink (see set_sink()). Called under
///        the log's lock, one record at a time; a message it logs itself goes
///        to the default sink.
using Sink = void (*)(const Record& record, void* user) noexcept;

namespace detail
{
/// @brief A CHUNK_LOG specification, parsed: the most verbose level let
///        through per channel, and the items that were not understood.
struct Spec
{
    std::array<Level, n_channels>   max{};
    std::array<std::string_view, 4> bad{}; ///< the first ones; n_bad counts them all
    int                             n_bad = 0;
};

/// @brief Level named @p s ("off" is fatal: nothing but the fatal level, which is never filtered).
[[nodiscard]] constexpr bool parse_level(std::string_view s, Level& out) noexcept
{
    if (s == "off")
    {
        out = Level::fatal;
        return true;
    }
    for (std::size_t i = 1; i < n_levels; ++i)
    {
        if (s == level_names[i])
        {
            out = static_cast<Level>(i);
            return true;
        }
    }
    return false;
}

[[nodiscard]] constexpr std::string_view trim(std::string_view s) noexcept
{
    while (!s.empty() && (s.front() == ' ' || s.front() == '\t'))
    {
        s.remove_prefix(1);
    }
    while (!s.empty() && (s.back() == ' ' || s.back() == '\t'))
    {
        s.remove_suffix(1);
    }
    return s;
}

/**
 * @brief Parse @p spec over @p base: comma-separated items, left to right; a
 *        bare level sets every channel, `channel=level` one. Empty items are
 *        skipped; an item not understood is recorded in Spec::bad and changes
 *        nothing.
 */
[[nodiscard]] constexpr Spec parse_spec(std::string_view spec, const std::array<Level, n_channels>& base) noexcept
{
    Spec out;
    out.max = base;
    while (true)
    {
        const std::size_t      comma = spec.find(',');
        const std::string_view item  = trim(spec.substr(0, comma));
        if (!item.empty())
        {
            const std::size_t eq = item.find('=');
            Level             l{};
            bool              ok = false;
            if (eq == std::string_view::npos)
            {
                ok = parse_level(item, l);
                if (ok)
                {
                    out.max.fill(l);
                }
            }
            else
            {
                const std::string_view chan = trim(item.substr(0, eq));
                for (std::size_t c = 0; c < n_channels && !ok; ++c)
                {
                    if (chan == channel_names[c])
                    {
                        ok = parse_level(trim(item.substr(eq + 1)), l);
                        if (ok)
                        {
                            out.max[c] = l;
                        }
                    }
                }
            }
            if (!ok)
            {
                if (static_cast<std::size_t>(out.n_bad) < out.bad.size())
                {
                    out.bad[static_cast<std::size_t>(out.n_bad)] = item;
                }
                ++out.n_bad;
            }
        }
        if (comma == std::string_view::npos)
        {
            break;
        }
        spec.remove_prefix(comma + 1);
    }
    return out;
}

/// @brief Every channel at warn: what a program logs with CHUNK_LOG unset.
inline constexpr std::array<Level, n_channels> default_max = []
{
    std::array<Level, n_channels> m{};
    m.fill(Level::warn);
    return m;
}();

/// @brief Per channel, 2 + the most verbose level let through; 0 until the
///        configuration is read (zero-initialised: usable during static
///        initialisation, in any order).
inline std::array<std::atomic<std::uint8_t>, n_channels> thresholds{};

inline std::once_flag init_once;
inline std::mutex     sink_mutex;
inline Sink           sink      = nullptr; ///< nullptr: the default sink
inline void*          sink_user = nullptr;
inline std::FILE*     out       = nullptr; ///< the default sink's stream; stderr until CHUNK_LOG_FILE opens

/// @brief The calling thread's kernel id, read once per thread.
[[nodiscard]] inline long thread_id() noexcept
{
    thread_local const long id = static_cast<long>(::gettid());
    return id;
}

inline void store(const std::array<Level, n_channels>& max) noexcept
{
    for (std::size_t c = 0; c < n_channels; ++c)
    {
        thresholds[c].store(static_cast<std::uint8_t>(static_cast<unsigned>(max[c]) + 2U), std::memory_order_relaxed);
    }
}

/// @brief The thresholds as they stand (the default for a channel not configured yet).
[[nodiscard]] inline std::array<Level, n_channels> current() noexcept
{
    std::array<Level, n_channels> max = default_max;
    for (std::size_t c = 0; c < n_channels; ++c)
    {
        const unsigned s = thresholds[c].load(std::memory_order_relaxed);
        if (s >= 2U)
        {
            max[c] = static_cast<Level>(s - 2U);
        }
    }
    return max;
}

/// @brief The default sink: prefix, message, newline, in one fwrite.
inline void default_sink(const Record& r) noexcept
{
    static constexpr std::array<char, n_levels> letter{'F', 'E', 'W', 'I', 'D', 'T'};
    char                                        line[line_max + 64];
    const auto  res = std::format_to_n(line, sizeof(line) - 1, "[shrink_array][{}][{}][{}] {}", name(r.channel),
                                       letter[static_cast<std::size_t>(r.level)], thread_id(), r.message);
    std::size_t n =
        static_cast<std::size_t>(res.size) < sizeof(line) - 1 ? static_cast<std::size_t>(res.size) : sizeof(line) - 1;
    line[n++]          = '\n';
    std::FILE* const f = out != nullptr ? out : stderr;
    std::fwrite(line, 1, n, f);
    std::fflush(f);
}

/// @brief Hand @p r to the sink. A sink that logs reaches the default sink.
inline void emit(const Record& r) noexcept
{
    thread_local bool in_sink = false;
    if (in_sink)
    {
        default_sink(r);
        return;
    }
    try
    {
        const std::scoped_lock lk(sink_mutex);
        if (sink == nullptr)
        {
            default_sink(r);
            return;
        }
        in_sink = true;
        sink(r, sink_user);
        in_sink = false;
    }
    catch (...) // std::mutex::lock: std::system_error
    {
        in_sink = false;
        default_sink(r);
    }
}

/// @brief Format @p fmt into @p buf (line_max bytes, "..." when cut); the length written.
template <typename... Args>
std::size_t format_message(char* buf, std::format_string<Args...> fmt, Args&&... args) noexcept
{
    try
    {
        const auto res = std::format_to_n(buf, line_max, fmt, std::forward<Args>(args)...);
        if (static_cast<std::size_t>(res.size) <= line_max)
        {
            return static_cast<std::size_t>(res.size);
        }
        std::memcpy(buf + line_max - 3, "...", 3);
        return line_max;
    }
    catch (...)
    {
        constexpr std::string_view failed = "(log message could not be formatted)";
        std::memcpy(buf, failed.data(), failed.size());
        return failed.size();
    }
}

/// @brief Write @p message as a log record from the configuration code itself
///        (no ensure_configured(): it runs inside the call_once).
inline void emit_direct(Level                       l,
                        Channel                     c,
                        std::string_view            message,
                        const std::source_location& loc = std::source_location::current()) noexcept
{
    if (static_cast<unsigned>(l) + 1U < thresholds[static_cast<std::size_t>(c)].load(std::memory_order_relaxed))
    {
        emit(Record{l, c, message, loc});
    }
}

/// @brief Warn, on channel env, about the items of @p spec (from @p who) not understood.
inline void report_bad_items(const Spec& spec, std::string_view who) noexcept
{
    const int shown = spec.n_bad < static_cast<int>(spec.bad.size()) ? spec.n_bad : static_cast<int>(spec.bad.size());
    for (int i = 0; i < shown; ++i)
    {
        char       msg[line_max];
        const auto len = format_message(msg,
                                        "{}: item \"{}\" is not understood: ignored (expected a level, or "
                                        "channel=level)",
                                        who, spec.bad[static_cast<std::size_t>(i)]);
        emit_direct(Level::warn, Channel::env, std::string_view{msg, len});
    }
    if (spec.n_bad > shown)
    {
        char       msg[line_max];
        const auto len = format_message(msg, "{}: {} more item(s) not understood", who, spec.n_bad - shown);
        emit_direct(Level::warn, Channel::env, std::string_view{msg, len});
    }
}

/// @brief Read CHUNK_LOG and CHUNK_LOG_FILE: once, at the first message.
inline void read_environment() noexcept
{
    const char* env  = std::getenv("CHUNK_LOG");
    const Spec  spec = parse_spec(env != nullptr ? env : "", default_max);
    store(spec.max);

    const char* path = std::getenv("CHUNK_LOG_FILE");
    if (path != nullptr && *path != '\0')
    {
        char        expanded[4096];
        std::size_t n = 0;
        for (const char* p = path; *p != '\0' && n + 24 < sizeof(expanded); ++p)
        {
            if (p[0] == '%' && p[1] == 'p')
            {
                n += static_cast<std::size_t>(
                    std::format_to_n(expanded + n, sizeof(expanded) - 1 - n, "{}", static_cast<long>(::getpid())).size);
                ++p;
            }
            else
            {
                expanded[n++] = *p;
            }
        }
        expanded[n] = '\0';
        out         = std::fopen(expanded, "a");
        if (out == nullptr)
        {
            char       msg[line_max];
            const auto len = format_message(msg, "CHUNK_LOG_FILE=\"{}\" cannot be opened: the log stays on stderr",
                                            std::string_view{expanded});
            emit_direct(Level::warn, Channel::env, std::string_view{msg, len});
        }
    }

    report_bad_items(spec, "CHUNK_LOG");
}

inline void ensure_configured() noexcept
{
    try
    {
        std::call_once(init_once, read_environment);
    }
    catch (...) // std::call_once: std::system_error
    {
        store(default_max);
    }
}

/// @brief Format and emit one message (CHUNK_LOG's slow path, past enabled()).
template <typename... Args>
void write(Level                       l,
           Channel                     c,
           const std::source_location& loc,
           std::format_string<Args...> fmt,
           Args&&... args) noexcept
{
    char              buf[line_max];
    const std::size_t len = format_message(buf, fmt, std::forward<Args>(args)...);
    emit(Record{l, c, std::string_view{buf, len}, loc});
}
} // namespace detail

/// @brief True when a message of level @p l on channel @p c gets through.
///        One relaxed load and a compare once the configuration is read.
[[nodiscard]] inline bool enabled(Channel c, Level l) noexcept
{
    auto&    t = detail::thresholds[static_cast<std::size_t>(c)];
    unsigned s = t.load(std::memory_order_relaxed);
    if (s == 0U) [[unlikely]]
    {
        detail::ensure_configured();
        s = t.load(std::memory_order_relaxed);
    }
    return static_cast<unsigned>(l) + 1U < s;
}

/**
 * @brief Apply the specification @p spec (CHUNK_LOG's syntax) on top of the
 *        current one -- CHUNK_LOG itself is read first. The items not
 *        understood are reported on channel env.
 * @return The number of items not understood.
 */
inline int configure(std::string_view spec) noexcept
{
    detail::ensure_configured();
    const detail::Spec parsed = detail::parse_spec(spec, detail::current());
    detail::store(parsed.max);
    detail::report_bad_items(parsed, "configure");
    return parsed.n_bad;
}

/**
 * @brief Send every record to @p sink (with @p user) instead of the default
 *        sink; nullptr goes back to the default. For a program with a logger
 *        of its own: install it before the first array, it holds for the rest
 *        of the run.
 */
inline void set_sink(Sink sink, void* user = nullptr) noexcept
{
    try
    {
        const std::scoped_lock lk(detail::sink_mutex);
        detail::sink      = sink;
        detail::sink_user = user;
    }
    catch (...) // std::mutex::lock: std::system_error
    {
    }
}
} // namespace fast_alloc::logging

/// @brief Log a message of level @p lvl (error, warn, info, debug, trace) on
///        channel @p chan: a std::format string, checked at compile time, and
///        its arguments -- evaluated only when the message gets through.
#define CHUNK_LOG(lvl, chan, fmt, ...)                                                                     \
    do                                                                                                     \
    {                                                                                                      \
        static_assert(::fast_alloc::logging::Level::lvl != ::fast_alloc::logging::Level::fatal,            \
                      "CHUNK_LOG: the fatal level is CHUNK_INVARIANT_FAILED");                             \
        if constexpr (::fast_alloc::logging::compiled(::fast_alloc::logging::Level::lvl))                  \
        {                                                                                                  \
            if (::fast_alloc::logging::enabled(::fast_alloc::logging::Channel::chan,                       \
                                               ::fast_alloc::logging::Level::lvl)) [[unlikely]]            \
            {                                                                                              \
                ::fast_alloc::logging::detail::write(::fast_alloc::logging::Level::lvl,                    \
                                                     ::fast_alloc::logging::Channel::chan,                 \
                                                     std::source_location::current(), fmt, ##__VA_ARGS__); \
            }                                                                                              \
        }                                                                                                  \
    } while (false)

/// @brief CHUNK_LOG for the first @p limit times this line's message gets
///        through, then one line saying the rest is suppressed.
#define CHUNK_LOG_FIRST_N(limit, lvl, chan, fmt, ...)                                                              \
    do                                                                                                             \
    {                                                                                                              \
        static_assert(::fast_alloc::logging::Level::lvl != ::fast_alloc::logging::Level::fatal,                    \
                      "CHUNK_LOG_FIRST_N: the fatal level is CHUNK_INVARIANT_FAILED");                             \
        if constexpr (::fast_alloc::logging::compiled(::fast_alloc::logging::Level::lvl))                          \
        {                                                                                                          \
            if (::fast_alloc::logging::enabled(::fast_alloc::logging::Channel::chan,                               \
                                               ::fast_alloc::logging::Level::lvl)) [[unlikely]]                    \
            {                                                                                                      \
                static std::atomic<std::uint64_t> chunk_log_seen_{0};                                              \
                const std::uint64_t chunk_log_k_ = chunk_log_seen_.fetch_add(1, std::memory_order_relaxed);        \
                if (chunk_log_k_ < static_cast<std::uint64_t>(limit))                                              \
                {                                                                                                  \
                    const auto chunk_log_loc_ = std::source_location::current();                                   \
                    ::fast_alloc::logging::detail::write(::fast_alloc::logging::Level::lvl,                        \
                                                         ::fast_alloc::logging::Channel::chan, chunk_log_loc_,     \
                                                         fmt, ##__VA_ARGS__);                                      \
                    if (chunk_log_k_ + 1 == static_cast<std::uint64_t>(limit))                                     \
                    {                                                                                              \
                        ::fast_alloc::logging::detail::write(::fast_alloc::logging::Level::lvl,                    \
                                                             ::fast_alloc::logging::Channel::chan, chunk_log_loc_, \
                                                             "(further messages from {}:{} suppressed)",           \
                                                             chunk_log_loc_.file_name(), chunk_log_loc_.line());   \
                    }                                                                                              \
                }                                                                                                  \
            }                                                                                                      \
        }                                                                                                          \
    } while (false)

/// @brief CHUNK_LOG on the 1st, 2nd, 4th, 8th... time this line's message
///        gets through: a recurring event stays visible at log2(N) lines.
#define CHUNK_LOG_POW2(lvl, chan, fmt, ...)                                                                    \
    do                                                                                                         \
    {                                                                                                          \
        static_assert(::fast_alloc::logging::Level::lvl != ::fast_alloc::logging::Level::fatal,                \
                      "CHUNK_LOG_POW2: the fatal level is CHUNK_INVARIANT_FAILED");                            \
        if constexpr (::fast_alloc::logging::compiled(::fast_alloc::logging::Level::lvl))                      \
        {                                                                                                      \
            if (::fast_alloc::logging::enabled(::fast_alloc::logging::Channel::chan,                           \
                                               ::fast_alloc::logging::Level::lvl)) [[unlikely]]                \
            {                                                                                                  \
                static std::atomic<std::uint64_t> chunk_log_seen_{0};                                          \
                if (std::has_single_bit(chunk_log_seen_.fetch_add(1, std::memory_order_relaxed) + 1))          \
                {                                                                                              \
                    ::fast_alloc::logging::detail::write(::fast_alloc::logging::Level::lvl,                    \
                                                         ::fast_alloc::logging::Channel::chan,                 \
                                                         std::source_location::current(), fmt, ##__VA_ARGS__); \
                }                                                                                              \
            }                                                                                                  \
        }                                                                                                      \
    } while (false)
