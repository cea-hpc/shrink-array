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

/// @file bench_args.cpp
/// @brief Command-line parsing. See bench_args.h.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

#include "bench_args.h"
#include "bench_config.h" // set_bench_mode, BenchMode
#include "bench_utils.h"  // valid_args, suggest_closest, print_help

namespace
{
/// @brief The whole of @p text as a base-10 int, or exit(1) naming @p opt.
/// @details std::stoi alone reads the leading digits and drops the rest
///          ("1.9999999" gave 1) and throws on garbage or overflow.
int parse_int_arg(const char* opt, const char* text)
{
    try
    {
        std::size_t used  = 0;
        const int   value = std::stoi(text, &used, 10);
        if (used == std::string(text).size())
            return value;
    }
    catch (const std::exception&)
    {
        // Nothing to do here on purpose: an unparsable value falls through to
        // the same message as a partially parsed one, just below. Catching is
        // what turns stoi's throw into that message.
    }
    std::cerr << "error: " << opt << " expects an integer (got \"" << text << "\")\n";
    std::exit(1);
}

/// @brief The whole of @p text as a finite real > 0, or exit(1) naming @p opt.
double parse_positive_real_arg(const char* opt, const char* text)
{
    char*        end   = nullptr;
    const double value = std::strtod(text, &end);
    if (end != text && *end == '\0' && std::isfinite(value) && value > 0.0)
        return value;
    std::cerr << "error: " << opt << " expects a real > 0 (got \"" << text << "\")\n";
    std::exit(1);
}
} // namespace

BenchArgs parse_bench_args(int argc, char* argv[])
{
    BenchArgs args;

    // The locals the moved body works on, under their original names: the body
    // stays VERBATIM, and we copy into `args` on the way out.
    bool no_io_v       = false;
    bool compress_v    = false;
    bool write_trace   = false;
    bool read_trace    = false;
    bool test_stl      = false; // --test_stl: run the STL-interface self-checks
    bool no_struct     = false; // --no_struct: no Point3 / Cell arrays (bench_args.h)

    // Nouveaux facteurs
    double multiply_factor = 1.0;
    int divisor_factor  = 1;

    // Default iter_max (kept at 2000 to preserve historical behaviour);
    // can be overridden at runtime via --iter_max <N>.
    int iter_max_arg = 2000;

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];

        if (std::ranges::find(valid_args, arg) != valid_args.end())
        {
            if (arg == "--no_io")
                no_io_v = true;
            else if (arg == "--compress")
                compress_v = true;
            else if (arg == "--write_trace")
                write_trace = true;
            else if (arg == "--test_stl")
                test_stl = true;
            else if (arg == "--no_struct")
                no_struct = true;
            else if (arg == "--mode")
            {
                // check / perf: what used to be two separate binaries, built
                // with -DCHECK or -DPERF, is one binary and this flag.
                if (i + 1 < argc)
                {
                    const std::string m = argv[++i];
                    if (m == "perf")
                        set_bench_mode(BenchMode::perf);
                    else if (m == "check")
                        set_bench_mode(BenchMode::check);
                    else
                    {
                        std::cout << "Unknown --mode '" << m << "' (expected check or perf)\n";
                        // exit, not return: this function returns a BenchArgs,
                        // and the other error paths of this loop already leave
                        // this way. Exit code 1 is preserved.
                        std::exit(1);
                    }
                }
            }
            else if (arg == "--read_trace")
                read_trace = true;
            else if (arg == "--multiply")
            {
                if (i + 1 < argc)
                    multiply_factor = parse_positive_real_arg("--multiply", argv[++i]);
            }
            else if (arg == "--divisor")
            {
                if (i + 1 < argc)
                    divisor_factor = parse_int_arg("--divisor", argv[++i]);
            }
            else if (arg == "--iter_max")
            {
                if (i + 1 < argc)
                {
                    iter_max_arg = parse_int_arg("--iter_max", argv[++i]);
                    if (iter_max_arg <= 0)
                    {
                        std::cerr << "--iter_max must be a positive integer (got " << iter_max_arg << ")\n";
                        exit(1);
                    }
                }
            }
            else if (arg == "--help")
            {
                print_help();
                exit(0);
            }
            std::cout << "Recognized argument: " << arg << "\n";
        }
        else
        {
            const std::string suggestion = suggest_closest(arg);
            std::cout << "Unknown argument: " << arg << "\nDid you mean '" << suggestion << "'?\n";
            exit(0);
        }
    }

    args.no_io_v         = no_io_v;
    args.compress_v      = compress_v;
    args.write_trace     = write_trace;
    args.read_trace      = read_trace;
    args.test_stl        = test_stl;
    args.no_struct       = no_struct;
    args.multiply_factor = multiply_factor;
    args.divisor_factor  = divisor_factor;
    args.iter_max_arg    = iter_max_arg;
    return args;
}
