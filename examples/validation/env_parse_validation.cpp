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
// The checked number parsing of the environment variables and the command
// line (env_parse.h), alone: no chunk, no array, no variant.
// ============================================================================
//
//   1. parse_number on integers: the whole text or a std::expected error
//      saying why -- empty (null or ""), not a number (a leading space, '+',
//      '-' for an unsigned type), trailing characters, out of range;
//   2. parse_number on floating point: the same, ERANGE included;
//   3. env_number / env_optional: unset or empty gives the fallback (nothing)
//      silently; a valid value is taken; an invalid one gives the fallback
//      (nothing) AND one warning on the env channel (a typo such as 6o must
//      not pass unnoticed);
//   4. arg_number: the same rule for a command-line argument, its warning on
//      the general channel;
//   5. env_flag: unset, empty and a leading '0' are false, anything else true.
//
// The nightly's coverage report had every refusal of this header at 0 %: the
// programs only ever met valid values. Exit status 0 when every check passes.

#include <cstdlib>     // setenv, unsetenv
#include <string>      // std::string
#include <string_view> // std::string_view

#include "chunk/internal/env_parse.h" // parse_number, ParseError, env_number, env_optional, arg_number, env_flag
#include "chunk/print.h"              // shrink_array::println
#include "chunk/log.h"                // logging::set_sink, Record, Level, Channel

using shrink_array::println;

namespace logging = shrink_array::logging;

namespace
{

int g_fails = 0;

void check(bool ok, const char* what)
{
    if (!ok)
    {
        ++g_fails;
        println("FAIL {}", what);
    }
}

// The warnings the checks provoke, counted per channel instead of printed.
int g_env_warnings     = 0;
int g_general_warnings = 0;

void count_warnings(const logging::Record& record, void* /*user*/) noexcept
{
    if (record.level == logging::Level::warn)
    {
        if (record.channel == logging::Channel::env)
        {
            ++g_env_warnings;
        }
        else if (record.channel == logging::Channel::general)
        {
            ++g_general_warnings;
        }
    }
}

// --- 1 ----------------------------------------------------------------------
// The refusal and its reason (cpp23: parse_number returns std::expected).
template <typename T>
bool refused(const char* text, shrink_array::ParseError why)
{
    const auto v = shrink_array::parse_number<T>(text);
    return !v && v.error() == why;
}

void check_integers()
{
    using shrink_array::ParseError;
    check(shrink_array::parse_number<int>("42") == 42, "int: 42");
    check(shrink_array::parse_number<int>("-5") == -5, "int: -5");

    check(refused<int>("", ParseError::empty), "int: empty refused (empty)");
    check(refused<int>(nullptr, ParseError::empty), "int: null refused (empty)");
    check(refused<int>("12x", ParseError::trailing_chars), "int: 12x refused (trailing characters)");
    check(refused<int>(" 1", ParseError::not_a_number), "int: a leading space refused (not a number)");
    check(refused<int>("+1", ParseError::not_a_number), "int: a leading '+' refused (not a number)");
    check(refused<int>("99999999999", ParseError::out_of_range), "int: out of range refused (out of range)");
    check(refused<int>("abc", ParseError::not_a_number), "int: abc refused (not a number)");
    check(refused<unsigned>("-1", ParseError::not_a_number), "unsigned: '-' refused (not a number)");
    check(shrink_array::parse_number<long long>("99999999999") == 99999999999LL, "long long: 99999999999");

    check(std::string_view(shrink_array::to_string(ParseError::trailing_chars)) == "trailing characters",
          "to_string names the reason");
}

// --- 2 ----------------------------------------------------------------------
void check_floating()
{
    using shrink_array::ParseError;
    check(shrink_array::parse_number<double>("1.5") == 1.5, "double: 1.5");
    check(shrink_array::parse_number<double>("-2e3") == -2000.0, "double: -2e3");

    check(refused<double>("", ParseError::empty), "double: empty refused (empty)");
    check(refused<double>("abc", ParseError::not_a_number), "double: abc refused (not a number)");
    check(refused<double>("1.0x", ParseError::trailing_chars), "double: 1.0x refused (trailing characters)");
    check(refused<double>("1e400", ParseError::out_of_range), "double: 1e400 refused (out of range)");
}

// --- 3 ----------------------------------------------------------------------
void check_env_number()
{
    const char* const name = "CHUNK_ENV_PARSE_VALIDATION";

    ::unsetenv(name);
    g_env_warnings = 0;
    check(shrink_array::env_number(name, 64) == 64 && g_env_warnings == 0, "env: unset, the fallback, no warning");

    ::setenv(name, "", 1);
    check(shrink_array::env_number(name, 64) == 64 && g_env_warnings == 0, "env: empty, the fallback, no warning");

    ::setenv(name, "128", 1);
    check(shrink_array::env_number(name, 64) == 128 && g_env_warnings == 0, "env: 128 taken");

    ::setenv(name, "6o", 1);
    check(shrink_array::env_number(name, 64) == 64, "env: 6o, the fallback");
    check(g_env_warnings == 1, "env: 6o, one warning on the env channel");

    ::setenv(name, "0.25", 1);
    check(shrink_array::env_number(name, 1.0) == 0.25, "env: 0.25 as a double");

    // env_optional: the value, or nothing (unset, empty, invalid with its warning).
    ::setenv(name, "7", 1);
    check(shrink_array::env_optional<int>(name) == 7, "env_optional: 7");
    g_env_warnings = 0;
    ::setenv(name, "7x", 1);
    check(!shrink_array::env_optional<int>(name).has_value() && g_env_warnings == 1,
          "env_optional: 7x, nothing and one warning");
    ::unsetenv(name);
    check(!shrink_array::env_optional<int>(name).has_value(), "env_optional: unset, nothing");
    ::unsetenv(name);
}

// --- 4 ----------------------------------------------------------------------
void check_arg_number()
{
    g_general_warnings = 0;
    check(shrink_array::arg_number("3", 1, "--iter") == 3 && g_general_warnings == 0, "arg: 3 taken");
    check(shrink_array::arg_number("x3", 1, "--iter") == 1, "arg: x3, the fallback");
    check(g_general_warnings == 1, "arg: x3, one warning on the general channel");
}

// --- 5 ----------------------------------------------------------------------
void check_env_flag()
{
    const char* const name = "CHUNK_ENV_PARSE_VALIDATION_FLAG";
    ::unsetenv(name);
    check(!shrink_array::env_flag(name), "flag: unset is false");
    for (const char* off : {"", "0", "00"})
    {
        ::setenv(name, off, 1);
        check(!shrink_array::env_flag(name), (std::string("flag: \"") + off + "\" is false").c_str());
    }
    for (const char* on : {"1", "yes", "2"})
    {
        ::setenv(name, on, 1);
        check(shrink_array::env_flag(name), (std::string("flag: \"") + on + "\" is true").c_str());
    }
    ::unsetenv(name);
}

} // namespace

int main()
{
    // The default level shows warnings; a CHUNK_LOG of the caller's environment
    // could hide them, and the counts below would be wrong.
    ::setenv("CHUNK_LOG", "warn", 1);
    logging::set_sink(count_warnings);

    check_integers();
    check_floating();
    check_env_number();
    check_arg_number();
    check_env_flag();

    logging::set_sink(nullptr);
    if (g_fails != 0)
    {
        println("env_parse_validation: {} check(s) FAILED", g_fails);
        return 1;
    }
    println("env_parse_validation: OK");
    return 0;
}
