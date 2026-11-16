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
// NaN-check harness (CHUNK_DO_CHECK builds; a no-op that passes elsewhere).
// Variant-agnostic: zero / zero_one_value / zero_any by the compile-time macro.
// ============================================================================
//
// Guards two properties of the NaN check on written values:
//   1. every write path is checked, the compound operations included:
//      `a[i] += NAN` and `a[i] /= 0.0` on a zero used to go unreported (only
//      the plain assignment was checked);
//   2. the report names the element's index in the ARRAY, not its position in
//      its chunk.
// Each case runs in a child (`--case <name>`), which writes a NaN to element
// k = 3 * chunk_size + 5 of a double array and must stop with
// "element <k> of the array is NaN". The parent also writes ordinary values,
// infinities included, which must not be reported.
// ============================================================================

#include <cstdio>      // std::fopen, std::fread, std::fclose, std::remove
#include <cstdlib>     // std::system
#include <format>      // std::format
#include <limits>      // std::numeric_limits
#include <string>      // std::string
#include <type_traits> // std::conditional_t
#include <sys/wait.h>  // WIFEXITED, WEXITSTATUS

#include "chunk/config.h"               // chunk_config::use_zero_any / use_zero_one_value, build_config::do_check
#include "chunk/param_chunk.h"          // ParamChunk::init_indexes
#include "chunk/print.h"                // shrink_array::println
#include "chunk/zero_any_array.h"       // zero_any::Array
#include "chunk/zero_array.h"           // zero::Array
#include "chunk/zero_one_value_array.h" // zero_one_value::Array
#include "validation_harness.h"         // self_command, temp_path; run() is called by validation_driver.cpp

namespace validation::nan_check_validation
{

using namespace shrink_array;

namespace
{

template <typename T>
using Array =
    std::conditional_t<chunk_config::use_zero_any,
                       zero_any::Array<T>,
                       std::conditional_t<chunk_config::use_zero_one_value, zero_one_value::Array<T>, zero::Array<T>>>;

constexpr const char* variant_name()
{
    if constexpr (chunk_config::use_zero_any)
    {
        return "zero_any";
    }
    else if constexpr (chunk_config::use_zero_one_value)
    {
        return "zero_one_value";
    }
    else
    {
        return "zero";
    }
}

constexpr long n_chunks = 8;

/// The element every case writes its NaN to: in the fourth chunk, not at its
/// start, so that the index in the array and the position in the chunk differ.
long target_index(long chunk_sz)
{
    return (3 * chunk_sz) + 5;
}

/// The child: writes a NaN to the target element the way @p name says. Returns
/// only if the check did not stop it.
int run_case(const std::string& name)
{
    const double  nan      = std::numeric_limits<double>::quiet_NaN();
    const long    chunk_sz = static_cast<long>(Array<double>(1).chunk_size());
    Array<double> arr(n_chunks * chunk_sz);
    const long    k = target_index(chunk_sz);
    arr[k]          = 1.5; // a materialised chunk: the write path, not the constant one

    if (name == "assign")
    {
        arr[k] = nan;
    }
    else if (name == "add")
    {
        arr[k] += nan;
    }
    else if (name == "sub")
    {
        const double inf = std::numeric_limits<double>::infinity();
        arr[k]           = inf;
        arr[k] -= inf; // inf - inf
    }
    else if (name == "mul")
    {
        arr[k] *= nan;
    }
    else if (name == "div0")
    {
        arr[k] = 0.0;
        arr[k] /= 0.0; // 0 / 0
    }
    else
    {
        println("unknown case {}", name);
        return 2;
    }
    println("[{}] the NaN written by '{}' was not reported", variant_name(), name);
    return 1;
}

/// Runs case @p name in a child and checks that it stopped on the NaN report
/// naming element @p k of the array.
int check_case(const char* self, const std::string& name, long k)
{
    const std::string log    = validation::temp_path("nan_check_" + name + ".log");
    const std::string cmd    = validation::self_command(self) + " --case " + name + " > \"" + log + "\" 2>&1";
    const int         status = std::system(cmd.c_str());
    std::string       out;
    if (FILE* file = std::fopen(log.c_str(), "r"))
    {
        char buf[4096];
        for (std::size_t n = 0; (n = std::fread(buf, 1, sizeof buf, file)) > 0;)
        {
            out.append(buf, n);
        }
        std::fclose(file);
    }
    std::remove(log.c_str());

    const bool        stopped  = status != -1 && (!WIFEXITED(status) || WEXITSTATUS(status) != 0);
    const std::string expected = std::format("element {} of the array is NaN", k);
    const bool        message  = out.find(expected) != std::string::npos;
    println("[{}] {:6}: {}, {}", variant_name(), name, stopped ? "stopped" : "RAN ON",
            message ? "reports the index in the array" : "NO report of '" + expected + "'");
    if (!stopped || !message)
    {
        println("---- child output ----\n{}----------------------", out.c_str());
        return 1;
    }
    return 0;
}

/// Ordinary values, infinities and zeros included, through every operator:
/// none may be reported (the parent would stop).
void write_ordinary(long chunk_sz)
{
    Array<double> arr(n_chunks * chunk_sz);
    const double  inf = std::numeric_limits<double>::infinity();
    for (long i = 0; i < arr.size(); ++i)
    {
        arr[i] = static_cast<double>(i % 7);
        arr[i] += 0.5;
        arr[i] *= 2.0;
        arr[i] -= 1.0;
        arr[i] /= 3.0;
    }
    arr[1] = inf;
    arr[2] = -inf;
    arr[3] = 1.0;
    arr[3] /= 0.0; // +inf, not NaN
}

} // namespace

int run(int argc, char** argv)
{
    ParamChunk::init_indexes(true);
    if (argc > 2 && std::string(argv[1]) == "--case")
    {
        return run_case(argv[2]);
    }

    const long chunk_sz = static_cast<long>(Array<double>(1).chunk_size());
    println("=== nan_check_validation [variant = {}]  chunk size {} ===", variant_name(), chunk_sz);
    if constexpr (!build_config::do_check)
    {
        println("[{}] skipped: the NaN check is compiled in with CHUNK_DO_CHECK only", variant_name());
        return 0;
    }

    write_ordinary(chunk_sz);
    println("[{}] ordinary values and infinities: not reported", variant_name());

    const long k     = target_index(chunk_sz);
    int        fails = 0;
    for (const char* name : {"assign", "add", "sub", "mul", "div0"})
    {
        fails += check_case(argv[0], name, k);
    }
    println("nan_check_validation [{}]: {}", variant_name(), fails ? "FAILED" : "OK");
    return fails ? 1 : 0;
}

} // namespace validation::nan_check_validation
