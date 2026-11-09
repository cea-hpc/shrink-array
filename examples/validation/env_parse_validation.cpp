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
//   1. parse_number on integers: the whole text or nothing -- empty, null,
//      trailing characters, a leading space or '+', out of range, '-' for an
//      unsigned type are refused, and the output is left untouched;
//   2. parse_number on floating point: the same, ERANGE included;
//   3. env_number: unset or empty gives the fallback silently; a valid value
//      is taken; an invalid one gives the fallback AND one warning on the
//      env channel (a typo such as 6o must not pass unnoticed);
//   4. arg_number: the same rule for a command-line argument, its warning on
//      the general channel;
//   5. env_flag: unset, empty and a leading '0' are false, anything else true.
//
// The nightly's coverage report had every refusal of this header at 0 %: the
// programs only ever met valid values. Exit status 0 when every check passes.

#include <cstdlib> // setenv, unsetenv
#include <string>  // std::string

#include "chunk/internal/env_parse.h" // parse_number, env_number, arg_number, env_flag
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
void check_integers()
{
    int value = 7;
    check(shrink_array::parse_number("42", value) && value == 42, "int: 42");
    check(shrink_array::parse_number("-5", value) && value == -5, "int: -5");

    value = 7;
    check(!shrink_array::parse_number("", value) && value == 7, "int: empty refused, untouched");
    check(!shrink_array::parse_number(static_cast<const char*>(nullptr), value) && value == 7,
          "int: null refused, untouched");
    check(!shrink_array::parse_number("12x", value) && value == 7, "int: trailing characters refused");
    check(!shrink_array::parse_number(" 1", value) && value == 7, "int: a leading space refused");
    check(!shrink_array::parse_number("+1", value) && value == 7, "int: a leading '+' refused");
    check(!shrink_array::parse_number("99999999999", value) && value == 7, "int: out of range refused");
    check(!shrink_array::parse_number("abc", value) && value == 7, "int: not a number refused");

    unsigned u = 3;
    check(!shrink_array::parse_number("-1", u) && u == 3, "unsigned: '-' refused");
    long long big = 0;
    check(shrink_array::parse_number("99999999999", big) && big == 99999999999LL, "long long: 99999999999");
}

// --- 2 ----------------------------------------------------------------------
void check_floating()
{
    double value = 2.0;
    check(shrink_array::parse_number("1.5", value) && value == 1.5, "double: 1.5");
    check(shrink_array::parse_number("-2e3", value) && value == -2000.0, "double: -2e3");

    value = 2.0;
    check(!shrink_array::parse_number("", value) && value == 2.0, "double: empty refused");
    check(!shrink_array::parse_number("abc", value) && value == 2.0, "double: not a number refused");
    check(!shrink_array::parse_number("1.0x", value) && value == 2.0, "double: trailing characters refused");
    check(!shrink_array::parse_number("1e400", value) && value == 2.0, "double: out of range (ERANGE) refused");
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
