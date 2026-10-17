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

// ============================================================================
// The library's log (fast_allocators/internal/log_core.h), alone: no array,
// no variant.
// ============================================================================
//
//   1. the environment: CHUNK_LOG and CHUNK_LOG_FILE, set before the first
//      message, are read at it -- the levels apply, the file receives the
//      lines, and an item not understood is reported there;
//   2. parse_spec: bare levels, channel=level, left to right, spaces, empty
//      items, the items not understood;
//   3. filtering through configure(): what gets through per channel and
//      level, and the arguments of a message filtered out are not evaluated;
//   4. a sink: the records it receives (level, channel, text, line), a sink
//      that logs itself, back to the default sink;
//   5. CHUNK_LOG_FIRST_N and CHUNK_LOG_POW2 count right;
//   6. a message longer than line_max is cut, with "...";
//   7. no allocation on the path of a message (operator new is counted);
//   8. 8 threads writing to the default sink at once: every line in the file
//      whole, none lost.
//
// Exit status 0 when every check passes.

#include <array>       // std::array
#include <atomic>      // std::atomic
#include <cstdio>      // std::remove, std::sscanf
#include <cstdlib>     // std::malloc, std::free, setenv
#include <cstring>     // std::memcpy
#include <filesystem>  // std::filesystem::temp_directory_path
#include <fstream>     // std::ifstream
#include <new>         // std::bad_alloc
#include <string>      // std::string, std::getline
#include <string_view> // std::string_view
#include <thread>      // std::thread
#include <vector>      // std::vector
#include <unistd.h>    // getpid

#include "chunk/log.h"            // shrink_array::logging, CHUNK_LOG
#include "chunk/internal/print.h" // shrink_array::println

using shrink_array::println;

namespace logging = shrink_array::logging;
using logging::Channel;
using logging::Level;

// --- Allocation counter (check 7) -------------------------------------------
namespace
{
std::atomic<long> g_news{0};
thread_local bool g_counting = false;
} // namespace

void* operator new(std::size_t n)
{
    if (g_counting)
    {
        g_news.fetch_add(1, std::memory_order_relaxed);
    }
    if (void* p = std::malloc(n == 0 ? 1 : n))
    {
        return p;
    }
    throw std::bad_alloc{};
}

void operator delete(void* p) noexcept
{
    std::free(p);
}

void operator delete(void* p, std::size_t /*n*/) noexcept
{
    std::free(p);
}

namespace
{

int g_fails = 0;

void check(bool ok, std::string_view what)
{
    if (!ok)
    {
        ++g_fails;
        println("FAIL {}", what);
    }
}

// --- A capturing sink ---------------------------------------------------------
struct Captured
{
    Level       level;
    Channel     channel;
    std::string message;
    unsigned    line;
};

std::vector<Captured> g_captured;

void capture(const logging::Record& r, void* /*user*/) noexcept
{
    g_captured.push_back(Captured{r.level, r.channel, std::string{r.message}, r.location.line()});
}

// Preallocated: what check 7's sink writes into, so the sink allocates nothing.
std::array<char, logging::line_max> g_last{};
std::size_t                         g_last_len = 0;

void capture_no_alloc(const logging::Record& r, void* /*user*/) noexcept
{
    g_last_len = r.message.size();
    std::memcpy(g_last.data(), r.message.data(), r.message.size());
}

void logging_sink(const logging::Record& r, void* /*user*/) noexcept
{
    capture(r, nullptr);
    CHUNK_LOG(warn, general, "from inside the sink"); // must reach the default sink, not deadlock
}

std::vector<std::string> read_lines(const std::string& path)
{
    std::vector<std::string> lines;
    std::ifstream            in(path);
    std::string              line;
    while (std::getline(in, line))
    {
        lines.push_back(line);
    }
    return lines;
}

std::string g_log_path; // CHUNK_LOG_FILE, expanded

// --- 1 ----------------------------------------------------------------------
void check_environment()
{
    check(logging::enabled(Channel::rechunk, Level::debug), "env: CHUNK_LOG rechunk=debug applies");
    check(!logging::enabled(Channel::rechunk, Level::trace), "env: rechunk stops at debug");
    check(logging::enabled(Channel::pool, Level::warn), "env: the bare level warn applies to pool");
    check(!logging::enabled(Channel::pool, Level::info), "env: pool stops at warn");

    CHUNK_LOG(warn, general, "first line through the default sink");
    const auto lines             = read_lines(g_log_path);
    bool       bad_item_reported = false;
    bool       first_line        = false;
    for (const auto& l : lines)
    {
        bad_item_reported |= l.starts_with("[shrink_array][env][W][") && l.find("\"verbose\"") != std::string::npos;
        first_line |=
            l.starts_with("[shrink_array][general][W][") && l.ends_with("] first line through the default sink");
    }
    check(bad_item_reported, "env: CHUNK_LOG item \"verbose\" reported in CHUNK_LOG_FILE");
    check(first_line, "env: a message reaches CHUNK_LOG_FILE with its prefix");
}

// --- 2 ----------------------------------------------------------------------
void check_parse()
{
    using logging::detail::default_max;
    using logging::detail::parse_spec;
    constexpr auto ix = [](Channel c) { return static_cast<std::size_t>(c); };

    auto s = parse_spec("", default_max);
    check(s.n_bad == 0 && s.max == default_max, "parse: empty spec keeps the base");

    s = parse_spec("debug", default_max);
    check(s.max[ix(Channel::env)] == Level::debug && s.max[ix(Channel::advisor)] == Level::debug,
          "parse: a bare level sets every channel");

    s = parse_spec("warn,rechunk=trace", default_max);
    check(s.max[ix(Channel::rechunk)] == Level::trace && s.max[ix(Channel::pool)] == Level::warn,
          "parse: channel=level after the bare level");

    s = parse_spec("rechunk=trace,error", default_max);
    check(s.max[ix(Channel::rechunk)] == Level::error, "parse: left to right, a later bare level wins");

    s = parse_spec(" info , pool = debug ,, ", default_max);
    check(s.n_bad == 0 && s.max[ix(Channel::pool)] == Level::debug && s.max[ix(Channel::drain)] == Level::info,
          "parse: spaces and empty items");

    s = parse_spec("off", default_max);
    check(s.max[ix(Channel::general)] == Level::fatal, "parse: off lets nothing but fatal");

    s = parse_spec("loud,nochan=debug,pool=loud,fatal,pool=info,a,b,c", default_max);
    check(s.n_bad == 7 && s.bad[0] == "loud" && s.bad[1] == "nochan=debug" && s.bad[2] == "pool=loud" &&
              s.bad[3] == "fatal",
          "parse: the items not understood, in order (fatal is not a level CHUNK_LOG sets)");
    check(s.max[ix(Channel::pool)] == Level::info, "parse: a bad item changes nothing, the good ones apply");
}

// --- 3 ----------------------------------------------------------------------
void check_filter()
{
    logging::configure("warn,drain=info");
    check(logging::enabled(Channel::drain, Level::info), "filter: drain=info lets info");
    check(!logging::enabled(Channel::drain, Level::debug), "filter: drain=info stops debug");
    check(logging::enabled(Channel::general, Level::error), "filter: warn lets error");
    check(!logging::enabled(Channel::general, Level::info), "filter: warn stops info");

    int        evaluated = 0;
    const auto touch     = [&] { return ++evaluated; };
    CHUNK_LOG(debug, general, "filtered out {}", touch());
    check(evaluated == 0, "filter: the arguments of a message filtered out are not evaluated");
    logging::configure("off");
    CHUNK_LOG(error, general, "filtered out {}", touch());
    check(evaluated == 0, "filter: off stops error");
    logging::configure("warn");
}

// --- 4 ----------------------------------------------------------------------
void check_sink()
{
    g_captured.clear();
    logging::set_sink(capture);
    logging::configure("warn,ompt=debug");
    const unsigned line = __LINE__ + 1;
    CHUNK_LOG(debug, ompt, "value {} and {}", 42, std::string_view{"text"});
    CHUNK_LOG(debug, pool, "filtered out");
    CHUNK_LOG(error, pool, "kept");
    check(g_captured.size() == 2, "sink: two records of three");
    if (g_captured.size() == 2)
    {
        check(g_captured[0].level == Level::debug && g_captured[0].channel == Channel::ompt,
              "sink: level and channel of the record");
        check(g_captured[0].message == "value 42 and text", "sink: the message, formatted, no prefix");
        check(g_captured[0].line == line, "sink: the source location is the caller's");
        check(g_captured[1].level == Level::error && g_captured[1].message == "kept", "sink: second record");
    }

    g_captured.clear();
    logging::set_sink(logging_sink);
    CHUNK_LOG(warn, general, "to a sink that logs");
    check(g_captured.size() == 1, "sink: a sink that logs gets its record once (its own goes to the default)");

    logging::set_sink(nullptr);
    logging::configure("warn");
}

// --- 5 ----------------------------------------------------------------------
void check_rate_limits()
{
    g_captured.clear();
    logging::set_sink(capture);
    for (int i = 0; i < 10; ++i)
    {
        CHUNK_LOG_FIRST_N(3, warn, general, "first-n {}", i);
    }
    check(g_captured.size() == 4, "FIRST_N(3) over 10: 3 messages and the suppression line");
    if (g_captured.size() == 4)
    {
        check(g_captured[2].message == "first-n 2", "FIRST_N: the third message is the third call's");
        check(g_captured[3].message.starts_with("(further messages from ") &&
                  g_captured[3].message.ends_with(" suppressed)"),
              "FIRST_N: the suppression line");
    }

    g_captured.clear();
    for (int i = 1; i <= 100; ++i)
    {
        CHUNK_LOG_POW2(warn, general, "pow2 {}", i);
    }
    check(g_captured.size() == 7, "POW2 over 100: calls 1, 2, 4, 8, 16, 32, 64");
    if (g_captured.size() == 7)
    {
        check(g_captured[6].message == "pow2 64", "POW2: the 7th message is the 64th call's");
    }
    logging::set_sink(nullptr);
}

// --- 6 ----------------------------------------------------------------------
void check_truncation()
{
    g_captured.clear();
    logging::set_sink(capture);
    const std::string big(3 * logging::line_max, 'x');
    CHUNK_LOG(warn, general, "{}", big);
    check(g_captured.size() == 1 && g_captured[0].message.size() == logging::line_max &&
              g_captured[0].message.ends_with("xx..."),
          "truncation: line_max bytes, ending in ...");
    logging::set_sink(nullptr);
}

// --- 7 ----------------------------------------------------------------------
std::vector<int>* g_kept = nullptr; // stored: the counter's own check below cannot be elided

void check_no_allocation()
{
    g_news.store(0);
    g_counting = true;
    g_kept     = new std::vector<int>(100);
    g_counting = false;
    check(g_news.load() == 2, "no allocation: the counter counts (a vector and its buffer)");
    delete g_kept;

    logging::set_sink(capture_no_alloc);
    const double v = 2.5;
    g_news.store(0);
    g_counting = true;
    CHUNK_LOG(warn, general, "no allocation: {} {:.3f} {} {}", 123456789L, v, std::string_view{"view"},
              static_cast<const void*>(&v));
    CHUNK_LOG(debug, general, "filtered out {}", 1);
    CHUNK_LOG_POW2(warn, general, "pow2 no allocation {}", 1);
    g_counting = false;
    check(g_news.load() == 0, "no allocation on the path of a message");
    check(std::string_view(g_last.data(), g_last_len) == "pow2 no allocation 1", "no allocation: the messages arrived");
    logging::set_sink(nullptr);
}

// --- 8 ----------------------------------------------------------------------
void check_threads()
{
    constexpr int kThreads = 8;
    constexpr int kLines   = 2000;
    logging::configure("warn,general=info");
    {
        std::vector<std::thread> team;
        for (int t = 0; t < kThreads; ++t)
        {
            team.emplace_back(
                [t]
                {
                    for (int i = 0; i < kLines; ++i)
                    {
                        CHUNK_LOG(info, general, "thread {} line {} {}", t, i, std::string_view{"padding-padding"});
                    }
                });
        }
        for (auto& th : team)
        {
            th.join();
        }
    }
    std::array<int, kThreads> seen{};
    int                       broken = 0;
    for (const auto& l : read_lines(g_log_path))
    {
        if (!l.starts_with("[shrink_array][general][I]["))
        {
            continue;
        }
        int        t   = -1;
        int        i   = -1;
        const auto pos = l.find("] thread ");
        if (pos == std::string::npos || std::sscanf(l.c_str() + pos, "] thread %d line %d", &t, &i) != 2 || t < 0 ||
            t >= kThreads || !l.ends_with(" padding-padding"))
        {
            ++broken;
            continue;
        }
        ++seen[static_cast<std::size_t>(t)];
    }
    check(broken == 0, "threads: every line whole");
    bool all = true;
    for (const int s : seen)
    {
        all &= (s == kLines);
    }
    check(all, "threads: every line of every thread in the file");
    logging::configure("warn");
}

} // namespace

int main()
{
    // Before the first message: the configuration is read at it.
    const std::string dir = std::filesystem::temp_directory_path().string();
    g_log_path            = dir + "/shrink_log_validation." + std::to_string(::getpid()) + ".log";
    std::remove(g_log_path.c_str());
    setenv("CHUNK_LOG", "warn, rechunk=debug, verbose", 1);
    setenv("CHUNK_LOG_FILE", (dir + "/shrink_log_validation.%p.log").c_str(), 1);

    check_environment();
    check_parse();
    check_filter();
    check_sink();
    check_rate_limits();
    check_truncation();
    check_no_allocation();
    check_threads();

    std::remove(g_log_path.c_str());
    println("{}", g_fails == 0 ? "PASS" : "FAILED");
    return g_fails == 0 ? 0 : 1;
}
