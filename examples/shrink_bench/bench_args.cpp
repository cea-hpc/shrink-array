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

#include <algorithm> // std::ranges::find
#include <cmath>     // std::isfinite
#include <cstdlib>   // std::exit, std::strtod
#include <exception> // std::exception (the stoi guard below)
#include <string>    // std::string, std::stoi
#include <print>     // std::println

#include "bench_args.h"   // the declarations defined here
#include "bench_config.h" // set_bench_mode, BenchMode
#include "bench_utils.h"  // valid_args, suggest_closest, print_help

namespace
{
/// @brief The whole of @p text as a base-10 Int, or exit(1) naming @p opt.
/// @details std::stoi alone reads the leading digits and drops the rest
///          ("1.9999999" gave 1) and throws on garbage or overflow.
Int parse_int_arg(const char* opt, const char* text)
{
    try
    {
        std::size_t used  = 0;
        const Int   value = std::stoi(text, &used, 10);
        if (used == std::string(text).size())
        {
            return value;
        }
    }
    catch (const std::exception&)
    {
        // Nothing to do here on purpose: an unparsable value falls through to
        // the same message as a partially parsed one, just below. Catching is
        // what turns stoi's throw into that message.
    }
    std::println(stderr, "error: {} expects an integer (got \"{}\")", opt, text);
    std::exit(1);
}

/// @brief The whole of @p text as a finite real > 0, or exit(1) naming @p opt.
double parse_positive_real_arg(const char* opt, const char* text)
{
    char*        end   = nullptr;
    const double value = std::strtod(text, &end);
    if (end != text && *end == '\0' && std::isfinite(value) && value > 0.0)
    {
        return value;
    }
    std::println(stderr, "error: {} expects a real > 0 (got \"{}\")", opt, text);
    std::exit(1);
}
} // namespace

BenchArgs parse_bench_args(int argc, char* argv[])
{
    BenchArgs args;

    // The locals the moved body works on, under their original names: the body
    // stays VERBATIM, and we copy into `args` on the way out.
    bool   no_io_v       = false;
    bool   compress_v    = false;
    bool   write_trace   = false;
    bool   read_trace    = false;
    bool   test_stl      = false;      // --test_stl: run the STL-interface self-checks
    bool   test_prealloc = false;      // --test_prealloc: alloc_chunk before a partial loop
    bool   no_struct     = false;      // --no_struct: no Point3 / Cell arrays (bench_args.h)
    bool   no_mask       = false;      // --no_mask: no tab_mask (bench_args.h)
    double mask_scale    = 1.0 / 32.0; // --mask_scale: tab_mask's size / sz (bench_args.h)

    // Nouveaux facteurs
    double multiply_factor = 1.0;
    Int    divisor_factor  = 1;

    // Default iter_max (kept at 2000 to preserve historical behaviour);
    // can be overridden at runtime via --iter_max <N>.
    Int iter_max_arg = 2000;

    for (Int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];

        // The value after a value-taking flag, or exit(1) naming the flag. A
        // flag given last used to be skipped silently and the run went on with
        // the default ("--mode" alone ran the default mode and exited 0).
        const auto next_value = [&]() -> const char*
        {
            if (i + 1 >= argc)
            {
                std::println(stderr, "error: {} expects a value", arg);
                std::exit(1);
            }
            return argv[++i];
        };

        if (std::ranges::find(valid_args, arg) != valid_args.end())
        {
            if (arg == "--no_io")
            {
                no_io_v = true;
            }
            else if (arg == "--compress")
            {
                compress_v = true;
            }
            else if (arg == "--write_trace")
            {
                write_trace = true;
            }
            else if (arg == "--test_stl")
            {
                test_stl = true;
            }
            else if (arg == "--test_prealloc")
            {
                test_prealloc = true;
            }
            else if (arg == "--no_struct")
            {
                no_struct = true;
            }
            else if (arg == "--no_mask")
            {
                no_mask = true;
            }
            else if (arg == "--mask_scale")
            {
                mask_scale = parse_positive_real_arg("--mask_scale", next_value());
            }
            else if (arg == "--mode")
            {
                // check / perf: what used to be two separate binaries, built
                // with -DCHECK or -DPERF, is one binary and this flag.
                const std::string m = next_value();
                if (m == "perf")
                {
                    set_bench_mode(BenchMode::perf);
                }
                else if (m == "check")
                {
                    set_bench_mode(BenchMode::check);
                }
                else
                {
                    std::println(stderr, "error: unknown --mode '{}' (expected check or perf)", m);
                    // exit, not return: this function returns a BenchArgs,
                    // and the other error paths of this loop already leave
                    // this way.
                    std::exit(1);
                }
            }
            else if (arg == "--read_trace")
            {
                read_trace = true;
            }
            else if (arg == "--multiply")
            {
                multiply_factor = parse_positive_real_arg("--multiply", next_value());
            }
            else if (arg == "--divisor")
            {
                divisor_factor = parse_int_arg("--divisor", next_value());
            }
            else if (arg == "--iter_max")
            {
                iter_max_arg = parse_int_arg("--iter_max", next_value());
                if (iter_max_arg <= 0)
                {
                    std::println(stderr, "error: --iter_max must be a positive integer (got {})", iter_max_arg);
                    std::exit(1);
                }
            }
            else if (arg == "--help")
            {
                print_help();
                std::exit(0);
            }
            // No "Recognized argument" echo: one stdout line per flag was noise
            // in the output the scripts parse, and the command line says it.
        }
        else
        {
            // exit(1), not 0: a typo in a CI command line must fail the job,
            // not skip the run and report success.
            const std::string suggestion = suggest_closest(arg);
            std::println(stderr, "error: unknown argument: {}\nDid you mean '{}'? (--help lists them)", arg,
                         suggestion);
            std::exit(1);
        }
    }

    args.no_io_v         = no_io_v;
    args.compress_v      = compress_v;
    args.write_trace     = write_trace;
    args.read_trace      = read_trace;
    args.test_stl        = test_stl;
    args.test_prealloc   = test_prealloc;
    args.no_struct       = no_struct;
    args.no_mask         = no_mask;
    args.mask_scale      = mask_scale;
    args.multiply_factor = multiply_factor;
    args.divisor_factor  = divisor_factor;
    args.iter_max_arg    = iter_max_arg;
    return args;
}
