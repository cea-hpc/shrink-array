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
 *   - `+=`, `-=`, `++`, `--`, `|=` on a zero element, then plain writes in
 *     the same chunk that bring every OTHER element back to zero: the chunk
 *     keeps its buffer and the element its value (a serial zero_or_value
 *     without a declared second value counted the element as still zero,
 *     and folded the chunk). And back: the element returns to zero through
 *     the operator, the chunk folds at shrink_to_fit().
 *   - A floating `x /= 0.0` is IEEE (inf), never refused.
 *   - zero_or_value: `set_second_value(T{})` on a fresh array declares nothing (T{}
 *     is already its first constant) and the array reads right.
 *   - Check builds only (build_config::any_check): an integer `/= 0` or
 *     `%= 0` -- undefined behaviour in C++ -- stops the program with an
 *     invariant failure. Each in a child, which re-runs this executable.
 */

#include <cstdlib>     // std::atoi
#include <limits>      // std::numeric_limits
#include <string>      // std::string
#include <type_traits> // std::conditional_t
#include <utility>     // std::as_const
#include <vector>      // the reference

#include "chunk/internal/chunk_traits.h" // forward decls + default IntType = int
#include "chunk/zero_array.h"            // zero::Array
#include "chunk/zero_or_value_array.h"   // zero_or_value::Array
#include "chunk/zero_any_array.h"        // zero_any::Array
#include "chunk/runtime.h"               // Runtime::init
#include "chunk/config.h"                // chunk_config::use_zero*, build_config::any_check
#include "chunk/print.h"                 // shrink_array::println

#include "validation_harness.h" // run_self; run() is called by validation_driver.cpp
#include "validation_variant.h" // Array, variant_name

namespace validation::compound_ops_validation
{

using namespace shrink_array;

namespace
{

template <typename T>
using Arr =
    std::conditional_t<chunk_config::use_zero_any,
                       zero_any::Array<T>,
                       std::conditional_t<chunk_config::use_zero_or_value, zero_or_value::Array<T>, zero::Array<T>>>;

template <typename T>
long mismatches(const Arr<T>& arr, const std::vector<T>& ref)
{
    long bad = 0;
    for (std::size_t i = 0; i < ref.size(); ++i)
    {
        bad += static_cast<T>(std::as_const(arr)[static_cast<int>(i)]) != ref[i];
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
    const int   count = 50000;
    Arr<int>    arr(count);
    std::vector ref(static_cast<std::size_t>(count), 0);
    for (int i = 0; i < count; ++i)
    {
        const int value                  = (i / 1000) % 3 == 0 ? 0 : 4 + (i % 9);
        arr[i]                           = value;
        ref[static_cast<std::size_t>(i)] = value;
    }
    for (int i = 0; i < count; ++i)
    {
        const auto pos = static_cast<std::size_t>(i);
        if (i % 2 == 0)
        {
            arr[i] %= 2; // 4 % 2, 6 % 2 ...: zero
            ref[pos] %= 2;
        }
        else if (i % 3 == 0)
        {
            arr[i] *= 0;
            ref[pos] *= 0;
        }
        else
        {
            arr[i] /= 3;
            ref[pos] /= 3;
        }
    }
    long bad = mismatches(arr, ref);
    arr.shrink_to_fit();
    bad += mismatches(arr, ref);
    return report("int *= /= %= vs std::vector", bad);
}

/// One chunk per operator: a zero element leaves zero through it, another
/// element of the chunk goes away from zero and back by plain writes -- the
/// transition that folds a chunk whose every element is counted zero.
int leave_zero_by_operator()
{
    constexpr int n_ops = 5;
    Arr<int>      arr(0);
    const auto    chunk_size = static_cast<int>(arr.chunk_size());
    const int     count      = n_ops * chunk_size;
    arr.resize(count);
    std::vector ref(static_cast<std::size_t>(count), 0);
    const auto  apply = [&](int op, int index, bool back)
    {
        int& want = ref[static_cast<std::size_t>(index)];
        switch (op)
        {
            case 0:
                back ? (arr[index] -= 5, want -= 5) : (arr[index] += 5, want += 5);
                break;
            case 1:
                back ? (arr[index] += 3, want += 3) : (arr[index] -= 3, want -= 3);
                break;
            case 2:
                back ? (--arr[index], --want) : (++arr[index], ++want);
                break;
            case 3:
                back ? (++arr[index], ++want) : (--arr[index], --want);
                break;
            default:
                back ? (arr[index] &= 0, want &= 0) : (arr[index] |= 6, want |= 6);
                break;
        }
    };
    for (int op = 0; op < n_ops; ++op)
    {
        const int first = op * chunk_size;
        apply(op, first, false);
        arr[first + 1] = 7;
        arr[first + 1] = 0; // every other element is zero again
    }
    long bad = mismatches(arr, ref);
    bad += arr.written_chunk_count() != n_ops;
    arr.shrink_to_fit();
    bad += mismatches(arr, ref);
    bad += arr.written_chunk_count() != n_ops;
    const int rc = report("+= -= ++ -- |= leave zero", bad);

    // And back to zero through the operator: the chunks fold.
    for (int op = 0; op < n_ops; ++op)
    {
        apply(op, op * chunk_size, true);
    }
    bad = mismatches(arr, ref);
    arr.shrink_to_fit();
    bad += mismatches(arr, ref);
    bad += arr.written_chunk_count() != 0;
    return rc + report("-= += -- ++ &= return to zero", bad);
}

/// A floating division by zero is IEEE: +inf, -inf, NaN -- never refused.
int float_div_by_zero()
{
    Arr<double> arr(4096);
    arr[1] = 2.0;
    arr[2] = -3.0;
    arr[1] /= 0.0;
    arr[2] /= 0.0;
    const double pos_inf = std::as_const(arr)[1];
    const double neg_inf = std::as_const(arr)[2];
    const bool   ok =
        pos_inf == std::numeric_limits<double>::infinity() && neg_inf == -std::numeric_limits<double>::infinity();
    return report("double /= 0.0 gives +-inf", ok ? 0 : 1);
}

/// zero_or_value: set_second_value(T{}) on a fresh array declares nothing. A
/// template, so that the zero_or_value-only calls are discarded on the other variants.
template <typename A = Arr<double>>
int second_value_zero()
{
    if constexpr (chunk_config::use_zero_or_value && !chunk_config::use_zero_any)
    {
        const int   count = 20000;
        A           arr(count);
        std::vector ref(static_cast<std::size_t>(count), 0.0);
        arr.set_second_value(0.0, /*message=*/false);
        long bad = arr.second_value() != 0.0;
        for (int i = 0; i < count; i += 3)
        {
            arr[i]                           = 1.5;
            ref[static_cast<std::size_t>(i)] = 1.5;
        }
        arr.shrink_to_fit();
        bad += mismatches<double>(arr, ref);
        return report("set_second_value(0) is a no-op", bad);
    }
    return 0;
}

/// Child: the operation that must stop a check build.
int run_child(int which)
{
    Arr<int> arr(1024);
    arr[5]              = 7;
    volatile int hidden = 0;      // not a constant the compiler could see
    const int    zero   = hidden; // read once: the operand is a plain int, not a volatile one
    if (which == 0)
    {
        arr[5] /= zero;
    }
    else
    {
        arr[5] %= zero;
    }
    println("value {}", static_cast<int>(std::as_const(arr)[5]));
    return 0;
}

int check_stops(const char* self, int which)
{
    const char* what = which == 0 ? "int /= 0 stops (check build)" : "int %= 0 stops (check build)";
    // The child: this harness again, no shell (validation_process.h).
    const validation::ProcessResult child   = validation::run_self(self, {"--child", std::to_string(which)});
    const std::string&              out     = child.output;
    const bool                      ran_ok  = child.ok();
    const bool                      message = out.find("by zero") != std::string::npos;
    const bool                      pass    = !ran_ok && message;
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
        Runtime::init(true);
        return run_child(std::atoi(argv[2]));
    }
    Runtime::init(true);
    println("=== compound_ops_validation [variant = {}] ===", variant_name());

    int fails = integer_ops() + leave_zero_by_operator() + float_div_by_zero() + second_value_zero();
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
