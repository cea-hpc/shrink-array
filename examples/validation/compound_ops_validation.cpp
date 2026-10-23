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

/**
 * @file compound_ops_validation.cpp
 * @brief `/=` and `%=` on the elements, against a std::vector, and their
 *        divisor check.
 *
 * @details
 *   - `*=`, `/=`, `%=` read back what a std::vector holds, results landing on
 *     zero included (`4 % 2`, `x * 0`): the chunks that become all zero fold
 *     as on any write. Then `shrink_to_fit()`, read again.
 *   - A floating `x /= 0.0` is IEEE (inf), never refused.
 *   - Chunk2: `set_second_value(T{})` on a fresh array declares nothing (T{}
 *     is already its first constant) and the array reads right.
 *   - Check builds only (build_config::any_check): an integer `/= 0` or
 *     `%= 0` -- undefined behaviour in C++ -- stops the program with an
 *     invariant failure. Each in a child, which re-runs this executable.
 */

#include <cstdio>      // std::fopen, std::fclose, std::fread, FILE, std::remove
#include <cstdlib>     // std::system, std::atoi
#include <limits>      // std::numeric_limits
#include <string>      // std::string
#include <sys/wait.h>  // WIFEXITED, WEXITSTATUS
#include <type_traits> // std::conditional_t
#include <unistd.h>    // getpid
#include <utility>     // std::as_const
#include <vector>      // the reference

#include "chunk/chunk_traits.h"   // forward decls + default IntType = int
#include "chunk/zero_array.h"    // zero::Array
#include "chunk/zero_one_value_array.h"   // zero_one_value::Array
#include "chunk/zero_any_array.h"  // zero_any::Array
#include "chunk/param_chunk.h"    // ParamChunk::init_indexes
#include "chunk/config.h"         // chunk_config::use_chunk*, build_config::any_check
#include "chunk/internal/print.h" // shrink_array::println

#include "validation_harness.h" // self_command; run() is called by validation_driver.cpp

namespace validation::compound_ops_validation
{

using namespace shrink_array;

namespace
{

template <typename T>
using Arr = std::conditional_t<chunk_config::use_chunk2a,
                               zero_any::Array<T>,
                               std::conditional_t<chunk_config::use_chunk2, zero_one_value::Array<T>, zero::Array<T>>>;

constexpr const char* variant_name()
{
    if constexpr (chunk_config::use_chunk2a)
    {
        return "chunk2a";
    }
    else if constexpr (chunk_config::use_chunk2)
    {
        return "chunk2";
    }
    else
    {
        return "chunk";
    }
}

template <typename T>
long mismatches(const Arr<T>& a, const std::vector<T>& ref)
{
    long bad = 0;
    for (std::size_t i = 0; i < ref.size(); ++i)
    {
        bad += static_cast<T>(std::as_const(a)[static_cast<int>(i)]) != ref[i];
    }
    return bad;
}

int report(const char* what, long bad)
{
    println("[{}/compound-ops] {:<34} {}", variant_name(), what, bad ? "FAIL" : "ok");
    if (bad)
    {
        println("  {} wrong element(s)", bad);
    }
    return bad ? 1 : 0;
}

/// `*=`, `/=`, `%=` against a std::vector, results on zero included.
int integer_ops()
{
    const int        n = 50000;
    Arr<int>         a(n);
    std::vector<int> ref(static_cast<std::size_t>(n), 0);
    for (int i = 0; i < n; ++i)
    {
        const int v                      = (i / 1000) % 3 == 0 ? 0 : 4 + (i % 9);
        a[i]                             = v;
        ref[static_cast<std::size_t>(i)] = v;
    }
    for (int i = 0; i < n; ++i)
    {
        const auto k = static_cast<std::size_t>(i);
        if (i % 2 == 0)
        {
            a[i] %= 2; // 4 % 2, 6 % 2 ...: zero
            ref[k] %= 2;
        }
        else if (i % 3 == 0)
        {
            a[i] *= 0;
            ref[k] *= 0;
        }
        else
        {
            a[i] /= 3;
            ref[k] /= 3;
        }
    }
    long bad = mismatches(a, ref);
    a.shrink_to_fit();
    bad += mismatches(a, ref);
    return report("int *= /= %= vs std::vector", bad);
}

/// A floating division by zero is IEEE: +inf, -inf, NaN -- never refused.
int float_div_by_zero()
{
    Arr<double> a(4096);
    a[1] = 2.0;
    a[2] = -3.0;
    a[1] /= 0.0;
    a[2] /= 0.0;
    const double p  = std::as_const(a)[1];
    const double m  = std::as_const(a)[2];
    const bool   ok = p == std::numeric_limits<double>::infinity() && m == -std::numeric_limits<double>::infinity();
    return report("double /= 0.0 gives +-inf", ok ? 0 : 1);
}

/// Chunk2: set_second_value(T{}) on a fresh array declares nothing. A
/// template, so that the Chunk2-only calls are discarded on the other variants.
template <typename A = Arr<double>>
int second_value_zero()
{
    if constexpr (chunk_config::use_chunk2 && !chunk_config::use_chunk2a)
    {
        const int           n = 20000;
        A                   a(n);
        std::vector<double> ref(static_cast<std::size_t>(n), 0.0);
        a.set_second_value(0.0, /*message=*/false);
        long bad = a.get_second_value() != 0.0;
        for (int i = 0; i < n; i += 3)
        {
            a[i]                             = 1.5;
            ref[static_cast<std::size_t>(i)] = 1.5;
        }
        a.shrink_to_fit();
        bad += mismatches<double>(a, ref);
        return report("set_second_value(0) is a no-op", bad);
    }
    return 0;
}

/// Child: the operation that must stop a check build.
int run_child(int k)
{
    Arr<int> a(1024);
    a[5]              = 7;
    volatile int zero = 0; // not a constant the compiler could see
    if (k == 0)
    {
        a[5] /= static_cast<int>(zero);
    }
    else
    {
        a[5] %= static_cast<int>(zero);
    }
    println("value {}", static_cast<int>(std::as_const(a)[5]));
    return 0;
}

int check_stops(const char* self, int k)
{
    const char*       what = k == 0 ? "int /= 0 stops (check build)" : "int %= 0 stops (check build)";
    const std::string log  = "/tmp/compound_ops_" + std::to_string(getpid()) + "_" + std::to_string(k) + ".log";
    const std::string cmd =
        validation::self_command(self) + " --child " + std::to_string(k) + " > \"" + log + "\" 2>&1";
    const int   status = std::system(cmd.c_str());
    std::string out;
    if (FILE* f = std::fopen(log.c_str(), "r"))
    {
        char buf[4096];
        for (std::size_t m = 0; (m = std::fread(buf, 1, sizeof buf, f)) > 0;)
        {
            out.append(buf, m);
        }
        std::fclose(f);
    }
    std::remove(log.c_str());
    const bool ran_ok  = status != -1 && WIFEXITED(status) && WEXITSTATUS(status) == 0;
    const bool message = out.find("by zero") != std::string::npos;
    const bool pass    = !ran_ok && message;
    if (!pass)
    {
        println("---- child output ----\n{}----------------------", out.c_str());
    }
    return report(what, pass ? 0 : 1);
}

} // namespace

int run(int argc, char** argv)
{
    if (argc > 2 && std::string(argv[1]) == "--child")
    {
        ParamChunk::init_indexes(true);
        return run_child(std::atoi(argv[2]));
    }
    ParamChunk::init_indexes(true);
    println("=== compound_ops_validation [variant = {}] ===", variant_name());

    int fails = integer_ops() + float_div_by_zero() + second_value_zero();
    if constexpr (build_config::any_check)
    {
        fails += check_stops(argv[0], 0) + check_stops(argv[0], 1);
    }
    else
    {
        println("[{}/compound-ops] not a check build: the divisor check is compiled out", variant_name());
    }
    println("{}", fails ? "compound_ops_validation: FAILED" : "compound_ops_validation: OK");
    return fails ? 1 : 0;
}

} // namespace validation::compound_ops_validation
