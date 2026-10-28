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
 *     CHUNK_LOG_ALWAYS(pool, "{} size {}: {} block(s)", label, size, n); // a report asked for
 *
 * - Filtering: CHUNK_LOG=warn,rechunk=info (the default: a re-chunk the
 *   decider applies says so), CHUNK_LOG=warn,rechunk=debug,
 *   CHUNK_LOG=off. Items apply left to right: a bare level sets every
 *   channel, `channel=level` one. Read once, at the first message;
 *   configure() applies a specification on top of it. The switches of the
 *   former traces raise a channel on top (see legacy_switches).
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
 * - Arguments: anything std::format spells, plus std::complex, "(re+imi)",
 *   a pointer to any object as its address (no static_cast<const void*> at
 *   the call), and any other type as its bytes in hex (fmt_arg): a message
 *   can print an element value of any type. A char pointer stays a string.
 * - Traces and reports: a trace is a flow CHUNK_LOG switches on (CHUNK_LOG);
 *   a report is a summary the program asked for explicitly -- a print_*
 *   function it called, a report switch it set (CHUNK_LEDGER_REPORT,
 *   CHUNK_OMPT_STATS), a trace build option -- and is written whatever
 *   CHUNK_LOG says (CHUNK_LOG_ALWAYS, level info), through the same sink.
 *   A report can run at exit: a sink installed with set_sink() must outlive
 *   it, or be removed (set_sink(nullptr)) before its state goes.
 * - Not here: the fatal level. An invariant failure is CHUNK_INVARIANT_FAILED
 *   (chunk/internal/invariant_check.h), which no setting can turn off.
 */

#include <array>           // std::array (the thresholds, the names)
#include <atomic>          // std::atomic
#include <bit>             // std::has_single_bit (CHUNK_LOG_POW2)
#include <complex>         // std::complex (spelled by fmt_arg)
#include <cstddef>         // std::size_t
#include <cstdint>         // std::uint8_t, std::uint64_t
#include <cstdio>          // std::FILE, std::fopen, std::fwrite, std::fflush, stderr
#include <cstdlib>         // std::getenv
#include <cstring>         // std::memcpy
#include <format>          // std::format_to_n, std::format_string
#include <mutex>           // std::mutex, std::once_flag, std::call_once, std::scoped_lock
#include <span>            // std::span, std::as_bytes (BytesArg)
#include <source_location> // std::source_location
#include <string_view>     // std::string_view
#include <type_traits> // std::remove_cvref_t, std::conditional_t, std::is_lvalue_reference_v, std::is_pointer_v, std::is_object_v
#include <utility>       // std::forward, std::declval
#include <sched.h>       // sched_yield (emit_fatal)
#include <sys/syscall.h> // SYS_gettid
#include <unistd.h>      // getpid, syscall

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
    memory, ///< the probes that cost: exact RSS (a page-table walk), allocator census, occupancy
    count_
};

inline constexpr std::size_t n_channels = static_cast<std::size_t>(Channel::count_);
inline constexpr std::size_t n_levels   = 6;

/// @brief Longest message, in bytes, before it is cut (with "...").
inline constexpr std::size_t line_max = 1024;

inline constexpr std::array<std::string_view, n_channels> channel_names{
    "general",  "env",     "alloc",  "arena", "pool",     "rechunk", "drain",
    "compress", "recycle", "ledger", "ompt",  "affinity", "advisor", "memory",
};
inline constexpr std::array<std::string_view, n_levels> level_names{"fatal", "error", "warn", "info", "debug", "trace"};

/// @brief The name CHUNK_LOG spells channel @p chan with.
[[nodiscard]] constexpr std::string_view name(Channel chan) noexcept
{
    return channel_names[static_cast<std::size_t>(chan)];
}

/// @brief The name CHUNK_LOG spells level @p lvl with.
[[nodiscard]] constexpr std::string_view name(Level lvl) noexcept
{
    return level_names[static_cast<std::size_t>(lvl)];
}

/// @brief True when messages of level @p lvl are compiled in (CHUNK_LOG_MAX_LEVEL).
[[nodiscard]] constexpr bool compiled(Level lvl) noexcept
{
    return static_cast<int>(lvl) <= CHUNK_LOG_MAX_LEVEL;
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
// --- The arguments: what std::format sees for each ---------------------------
// The messages print element values ("expected={} readback={}"). An element
// type is anything trivially copyable with an operator== -- a struct of
// doubles, an enum class, a std::array -- and most of those have no
// std::formatter. Rather than ask every such type for one, or drop the value
// from the message, the byte dump is what a type with no better spelling
// gets: enough to tell two values apart in a post-mortem. std::complex is
// spelled here too, and not by a std::formatter<std::complex<T>>
// specialisation: adding one to namespace std is undefined unless it depends
// on a program-defined type ([namespace.std]), it would leak from this header
// into every user's code, and it would clash with a standard library that
// ships one (C++26, P2197). Both are wrappers formatted in place: no
// allocation.

/// @brief True for std::complex<T>.
template <typename V>
inline constexpr bool is_complex = false;
template <typename T>
inline constexpr bool is_complex<std::complex<T>> = true;

/// @brief True when std::format has a formatter for V (C++20 spelling of
///        C++23's std::Formattable: a disabled formatter has no format()).
template <typename V>
concept Formattable =
    requires(std::formatter<V, char> fmt, const V& value, std::format_context& fmt_ctx) { fmt.format(value, fmt_ctx); };

/// @brief A std::complex, formatted "(re+imi)", "(re-imi)" for a negative imaginary part.
template <typename T>
struct ComplexArg
{
    std::complex<T> z;
};

/// @brief A value of a type with no formatter, formatted as its bytes: "0x" and
///        two hex digits per byte, in memory order.
template <typename V>
struct BytesArg
{
    const V* v;
};

/// @brief True for a pointer to an object std::format does not spell -- any
///        but void* and the character pointers (a string): printed as an address.
template <typename V>
inline constexpr bool is_address =
    std::is_pointer_v<V> && std::is_object_v<std::remove_pointer_t<V>> && !Formattable<V>;

/// @brief The argument std::format sees for @p arg: itself when its type is Formattable.
template <typename A>
    requires(Formattable<std::remove_cvref_t<A>> && !is_complex<std::remove_cvref_t<A>>)
constexpr A&& fmt_arg(A&& arg) noexcept
{
    return std::forward<A>(arg);
}

/// @brief Overload for std::complex.
template <typename A>
    requires is_complex<std::remove_cvref_t<A>>
constexpr auto fmt_arg(A&& value) noexcept
{
    return ComplexArg<typename std::remove_cvref_t<A>::value_type>{value};
}

/// @brief Overload for a pointer to an object: its address, as std::format spells a void*.
template <typename A>
    requires is_address<std::remove_cvref_t<A>>
constexpr const void* fmt_arg(A&& ptr) noexcept
{
    return ptr;
}

/// @brief Overload for a type with no formatter: its bytes.
template <typename A>
    requires(!Formattable<std::remove_cvref_t<A>> && !is_complex<std::remove_cvref_t<A>> &&
             !is_address<std::remove_cvref_t<A>>)
constexpr auto fmt_arg(A&& arg) noexcept
{
    return BytesArg<std::remove_cvref_t<A>>{&arg};
}

/// @brief The type std::format deduces for that argument: the reference for an
///        lvalue, the value for anything else (an rvalue reference would not match).
template <typename A, typename R = decltype(fmt_arg(std::declval<A>()))>
using FmtArgT = std::conditional_t<std::is_lvalue_reference_v<R>, R, std::remove_reference_t<R>>;

/// @brief A CHUNK_LOG specification, parsed: the most verbose level let
///        through per channel, and the items that were not understood.
struct Spec
{
    std::array<Level, n_channels>   max{};
    std::array<std::string_view, 4> bad{}; ///< the first ones; n_bad counts them all
    int                             n_bad = 0;
};

/// @brief Level named @p text ("off" is fatal: nothing but the fatal level, which is never filtered).
[[nodiscard]] constexpr bool parse_level(std::string_view text, Level& out) noexcept
{
    if (text == "off")
    {
        out = Level::fatal;
        return true;
    }
    for (std::size_t i = 1; i < n_levels; ++i)
    {
        if (text == level_names[i])
        {
            out = static_cast<Level>(i);
            return true;
        }
    }
    return false;
}

[[nodiscard]] constexpr std::string_view trim(std::string_view text) noexcept
{
    while (!text.empty() && (text.front() == ' ' || text.front() == '\t'))
    {
        text.remove_prefix(1);
    }
    while (!text.empty() && (text.back() == ' ' || text.back() == '\t'))
    {
        text.remove_suffix(1);
    }
    return text;
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
            Level             lvl{};
            bool              ok = false;
            if (eq == std::string_view::npos)
            {
                ok = parse_level(item, lvl);
                if (ok)
                {
                    out.max.fill(lvl);
                }
            }
            else
            {
                const std::string_view chan = trim(item.substr(0, eq));
                for (std::size_t c = 0; c < n_channels && !ok; ++c)
                {
                    if (chan == channel_names[c])
                    {
                        ok = parse_level(trim(item.substr(eq + 1)), lvl);
                        if (ok)
                        {
                            out.max[c] = lvl;
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

/// @brief What a program logs with CHUNK_LOG unset: every channel at warn,
///        rechunk at info -- the line of each re-chunk the decider applies
///        (one per move, a sequential point), as before the log existed. The
///        channel's other info lines need their switch (CHUNK_RECHUNK_CHURN) or
///        their policy field (RechunkPolicy::force_size). A bare level in
///        CHUNK_LOG sets rechunk too.
inline constexpr std::array<Level, n_channels> default_max = []
{
    std::array<Level, n_channels> levels{};
    levels.fill(Level::warn);
    levels[static_cast<std::size_t>(Channel::rechunk)] = Level::info;
    return levels;
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
    // syscall, not ::gettid(): the glibc wrapper is 2.30+ and RHEL 8 ships 2.28
    thread_local const long id = ::syscall(SYS_gettid);
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
        const unsigned stored = thresholds[c].load(std::memory_order_relaxed);
        if (stored >= 2U)
        {
            max[c] = static_cast<Level>(stored - 2U);
        }
    }
    return max;
}

/// @brief The default sink: prefix, message, newline, in one fwrite.
inline void default_sink(const Record& record) noexcept
{
    static constexpr std::array<char, n_levels> letter{'F', 'E', 'W', 'I', 'D', 'T'};
    char                                        line[line_max + 64];
    const auto  res = std::format_to_n(line, sizeof(line) - 1, "[shrink_array][{}][{}][{}] {}", name(record.channel),
                                       letter[static_cast<std::size_t>(record.level)], thread_id(), record.message);
    std::size_t len =
        static_cast<std::size_t>(res.size) < sizeof(line) - 1 ? static_cast<std::size_t>(res.size) : sizeof(line) - 1;
    line[len++]           = '\n';
    std::FILE* const file = out != nullptr ? out : stderr;
    std::fwrite(line, 1, len, file);
    std::fflush(file);
}

/// @brief True while the calling thread runs the sink of set_sink().
inline thread_local bool in_sink = false;

/// @brief True while the calling thread reads the configuration (inside the
///        call_once): what it warns about can reach a sink that fails, and
///        the failure must not call_once again.
inline thread_local bool configuring = false;

/// @brief Hand @p record to the sink. A sink that logs reaches the default sink.
inline void emit(const Record& record) noexcept
{
    if (in_sink)
    {
        default_sink(record);
        return;
    }
    try
    {
        const std::scoped_lock lk(sink_mutex);
        if (sink == nullptr)
        {
            default_sink(record);
            return;
        }
        in_sink = true;
        sink(record, sink_user);
        in_sink = false;
    }
    catch (...) // std::mutex::lock: std::system_error
    {
        in_sink = false;
        default_sink(record);
    }
}

inline void ensure_configured() noexcept;

/**
 * @brief The record of an invariant failure (CHUNK_INVARIANT_FAILED, which
 *        has already written it on stderr): to the sink of set_sink(), or to
 *        CHUNK_LOG_FILE when the default sink writes there. Never waits for
 *        long: the lock is tried for a while, then given up; a failure inside
 *        the sink skips it.
 */
inline void emit_fatal(const Record& record) noexcept
{
    if (!configuring)
    {
        ensure_configured();
    }
    const bool to_file = out != nullptr && out != stderr;
    if (!in_sink)
    {
        for (int attempt = 0; attempt < 1000; ++attempt)
        {
            if (sink_mutex.try_lock())
            {
                if (sink != nullptr)
                {
                    in_sink = true;
                    sink(record, sink_user);
                    in_sink = false;
                }
                else if (to_file)
                {
                    default_sink(record);
                }
                sink_mutex.unlock();
                return;
            }
            ::sched_yield();
        }
    }
    if (to_file)
    {
        default_sink(record);
    }
}

/// @brief Format @p fmt into @p buf (line_max bytes, "..." when cut); the length written.
template <typename... Args>
std::size_t format_message(char* buf, std::format_string<FmtArgT<Args>...> fmt, Args&&... args) noexcept
{
    try
    {
        const auto res = std::format_to_n(buf, line_max, fmt, fmt_arg(std::forward<Args>(args))...);
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
inline void emit_direct(Level                       lvl,
                        Channel                     chan,
                        std::string_view            message,
                        const std::source_location& loc = std::source_location::current()) noexcept
{
    if (static_cast<unsigned>(lvl) + 1U < thresholds[static_cast<std::size_t>(chan)].load(std::memory_order_relaxed))
    {
        emit(Record{lvl, chan, message, loc});
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

/// @brief An environment switch that raises a channel to a level: a trace
///        of before the log (deprecated: its use is warned about), or a
///        setting whose point is what it logs (CHUNK_RECHUNK_CHURN).
struct LegacySwitch
{
    std::string_view name;
    Channel          channel;
    Level            level;
    bool             deprecated;
};

/// @brief The switches, applied after CHUNK_LOG and raising only: setting
///        one shows what it showed before, whatever CHUNK_LOG says.
///        Set = non-empty and not starting with '0'.
inline constexpr std::array<LegacySwitch, 2> legacy_switches{{
    {"CHUNK_RECHUNK_CHURN", Channel::rechunk, Level::info, false},
    {"FAST_ALLOC_TRACE", Channel::alloc, Level::trace, true},
}};

/// @brief True when switch @p sw is set.
[[nodiscard]] inline bool is_set(const LegacySwitch& sw) noexcept
{
    char name[64];
    std::memcpy(name, sw.name.data(), sw.name.size());
    name[sw.name.size()] = '\0';
    const char* value    = std::getenv(name);
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

/// @brief Raise @p max by the legacy switches set.
inline void raise_by_legacy_switches(std::array<Level, n_channels>& max) noexcept
{
    for (const LegacySwitch& sw : legacy_switches)
    {
        Level& level = max[static_cast<std::size_t>(sw.channel)];
        if (is_set(sw) && level < sw.level)
        {
            level = sw.level;
        }
    }
}

/// @brief Warn, on channel env, about the deprecated switches set.
inline void warn_deprecated_switches() noexcept
{
    for (const LegacySwitch& sw : legacy_switches)
    {
        if (sw.deprecated && is_set(sw))
        {
            char       msg[line_max];
            const auto len =
                format_message(msg, "{} is deprecated: it means CHUNK_LOG={}={}, which it now sets", sw.name,
                               fast_alloc::logging::name(sw.channel), fast_alloc::logging::name(sw.level));
            emit_direct(Level::warn, Channel::env, std::string_view{msg, len});
        }
    }
}

/// @brief Read CHUNK_LOG, the legacy switches and CHUNK_LOG_FILE: once, at the first message.
inline void read_environment() noexcept
{
    configuring      = true;
    const char* env  = std::getenv("CHUNK_LOG");
    Spec        spec = parse_spec(env != nullptr ? env : "", default_max);
    raise_by_legacy_switches(spec.max);
    store(spec.max);

    const char* path = std::getenv("CHUNK_LOG_FILE");
    if (path != nullptr && *path != '\0')
    {
        char        expanded[4096];
        std::size_t used = 0;
        for (const char* p = path; *p != '\0' && used + 24 < sizeof(expanded); ++p)
        {
            if (p[0] == '%' && p[1] == 'p')
            {
                used += static_cast<std::size_t>(
                    std::format_to_n(expanded + used, sizeof(expanded) - 1 - used, "{}", static_cast<long>(::getpid()))
                        .size);
                ++p;
            }
            else
            {
                expanded[used++] = *p;
            }
        }
        expanded[used] = '\0';
        out            = std::fopen(expanded, "a");
        if (out == nullptr)
        {
            char       msg[line_max];
            const auto len = format_message(msg, "CHUNK_LOG_FILE=\"{}\" cannot be opened: the log stays on stderr",
                                            std::string_view{expanded});
            emit_direct(Level::warn, Channel::env, std::string_view{msg, len});
        }
    }

    report_bad_items(spec, "CHUNK_LOG");
    warn_deprecated_switches();
    configuring = false;
}

inline void ensure_configured() noexcept // declared above, for emit_fatal
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
void write(Level                                lvl,
           Channel                              chan,
           const std::source_location&          loc,
           std::format_string<FmtArgT<Args>...> fmt,
           Args&&... args) noexcept
{
    char              buf[line_max];
    const std::size_t len = format_message(buf, fmt, std::forward<Args>(args)...);
    emit(Record{lvl, chan, std::string_view{buf, len}, loc});
}
} // namespace detail

/// @brief True when a message of level @p lvl on channel @p chan gets through.
///        One relaxed load and a compare once the configuration is read.
[[nodiscard]] inline bool enabled(Channel chan, Level lvl) noexcept
{
    const auto& threshold = detail::thresholds[static_cast<std::size_t>(chan)];
    unsigned    stored    = threshold.load(std::memory_order_relaxed);
    if (stored == 0U) [[unlikely]]
    {
        detail::ensure_configured();
        stored = threshold.load(std::memory_order_relaxed);
    }
    return static_cast<unsigned>(lvl) + 1U < stored;
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

/// @brief fmt_arg's std::complex: "(re+imi)".
template <typename T>
struct std::formatter<fast_alloc::logging::detail::ComplexArg<T>, char>
{
    constexpr auto parse(std::format_parse_context& ctx)
    {
        return ctx.begin();
    }

    template <typename Ctx>
    auto format(const fast_alloc::logging::detail::ComplexArg<T>& arg, Ctx& ctx) const
    {
        return std::format_to(ctx.out(), "({}{:+}i)", arg.z.real(), arg.z.imag());
    }
};

/// @brief fmt_arg's byte dump: "0x" and two hex digits per byte, in memory order.
template <typename V>
struct std::formatter<fast_alloc::logging::detail::BytesArg<V>, char>
{
    constexpr auto parse(std::format_parse_context& ctx)
    {
        return ctx.begin();
    }

    template <typename Ctx>
    auto format(const fast_alloc::logging::detail::BytesArg<V>& arg, Ctx& ctx) const
    {
        constexpr std::string_view hex = "0123456789abcdef";
        auto                       it  = ctx.out();
        *it++                          = '0';
        *it++                          = 'x';
        const auto bytes               = std::as_bytes(std::span<const V, 1>(arg.v, 1));
        for (std::size_t i = 0; i < sizeof(V); ++i)
        {
            *it++ = hex[static_cast<unsigned>(bytes[i]) >> 4U];
            *it++ = hex[static_cast<unsigned>(bytes[i]) & 0xFU];
        }
        return it;
    }
};

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

/// @brief A message the program asked for explicitly (a report: see the file
///        comment): written whatever CHUNK_LOG says, at level info on channel
///        @p chan, through the sink.
#define CHUNK_LOG_ALWAYS(chan, fmt, ...)                                                                               \
    do                                                                                                                 \
    {                                                                                                                  \
        ::fast_alloc::logging::detail::ensure_configured();                                                            \
        ::fast_alloc::logging::detail::write(::fast_alloc::logging::Level::info, ::fast_alloc::logging::Channel::chan, \
                                             std::source_location::current(), fmt, ##__VA_ARGS__);                     \
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
