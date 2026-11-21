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
// Mixed-operand compound assignment harness. Every variant, default static size.
// ============================================================================
//
// `a[i] op= u` with an operand that is not clean for the element (a double
// into an int array: concept_is_clean_assignment.h) compiles with an HPC
// warning and converts the operand to T first, by design
// (operators_validation_cases.h, clean_operand): `int x *= 1.5` multiplies by
// 1. That rule must hold for every variant, every compound operator and
// whichever path the write takes: the direct one, or the deferred one
// (enqueue_async_arithmetic), whose context stores the operand.
//
// Each case runs twice against `int x = start; x op= static_cast<int>(u);`:
// with no forced deferral, then with the forced-deferral hook on every write
// (test_hooks.h). Only the operators the variant's proxy has are exercised.
// Compiled with -Wno-deprecated-declarations (CMakeLists.txt): the HPC
// warnings these operands raise are the point, not a mistake.
//
// EXIT STATUS: 0 when both runs follow that rule, 1 otherwise; the
// deferred run is skipped (noted) in builds without the hook.

#include <type_traits> // std::declval
#include <utility>     // std::as_const, std::forward

#include "chunk/runtime.h"             // Runtime::init
#include "chunk/internal/zero_chunk.h" // release_chunk_pools, Runtime::shutdown
#include "chunk/config.h"              // build_config::use_openmp
#include "chunk/print.h"               // shrink_array::println
#include "chunk/internal/test_hooks.h" // force_defer_every, force_defer_compiled

#include "validation_harness.h" // run() is called by validation_driver.cpp
#include "validation_variant.h" // Array, variant_name, begin_phase / end_phase

namespace validation::mixed_operand_compound_validation
{

using namespace shrink_array;

namespace
{

constexpr long count = 4096; ///< elements per case

enum class Op
{
    mul,
    div,
    add,
    sub,
    mod
};

/// @brief The operand of each case: a double, but a long for %= (an integer
///        that is not clean for an int either: Rule 5 admits int into small
///        integers only).
template <Op op>
constexpr auto operand_of()
{
    if constexpr (op == Op::mul)
    {
        return 1.5;
    }
    else if constexpr (op == Op::div)
    {
        return 2.5;
    }
    else if constexpr (op == Op::add || op == Op::sub)
    {
        return 0.5;
    }
    else
    {
        return 4294967299L; // 2^32 + 3: 3 once converted to int
    }
}

/// @brief `lhs op= rhs`, whatever lhs is (an element of the array, an int).
template <Op op, typename L, typename R>
void apply(L&& lhs, const R& rhs)
{
    if constexpr (op == Op::mul)
    {
        std::forward<L>(lhs) *= rhs;
    }
    else if constexpr (op == Op::div)
    {
        std::forward<L>(lhs) /= rhs;
    }
    else if constexpr (op == Op::add)
    {
        std::forward<L>(lhs) += rhs;
    }
    else if constexpr (op == Op::sub)
    {
        std::forward<L>(lhs) -= rhs;
    }
    else
    {
        std::forward<L>(lhs) %= rhs;
    }
}

/// @brief The variant's proxy has this `op=` with this operand.
template <Op op>
constexpr bool has_op()
{
    using A          = Array<int>;
    constexpr auto u = operand_of<op>();
    if constexpr (op == Op::mul)
    {
        return requires { std::declval<A&>()[0] *= u; };
    }
    else if constexpr (op == Op::div)
    {
        return requires { std::declval<A&>()[0] /= u; };
    }
    else if constexpr (op == Op::add)
    {
        return requires { std::declval<A&>()[0] += u; };
    }
    else if constexpr (op == Op::sub)
    {
        return requires { std::declval<A&>()[0] -= u; };
    }
    else
    {
        return requires { std::declval<A&>()[0] %= u; };
    }
}

/// @brief One case: every element set to @p start then `element op= operand_of<op>()`,
///        in a region; compared with `int x = start; x op= int(operand)`.
/// @return 1 if any element differs from the reference, 0 otherwise.
template <Op op>
int run_case(const char* label, const char* mode, int start)
{
    using Index      = Array<int>::IndexType;
    constexpr auto u = operand_of<op>();
    Array<int>     arr(static_cast<Index>(count));
#pragma omp parallel
    {
        begin_phase();
#pragma omp for schedule(static)
        for (long i = 0; i < count; ++i)
        {
            const auto k = static_cast<Index>(i);
            arr[k]       = start;
            apply<op>(arr[k], u);
        }
    }
    end_phase();
    // The library's rule for an operand that is not clean: converted to T first.
    int want = start;
    apply<op>(want, static_cast<int>(u));
    long bad  = 0;
    int  got0 = 0;
    for (long i = 0; i < count; ++i)
    {
        const int got = std::as_const(arr)[static_cast<Index>(i)];
        got0          = (i == 0) ? got : got0;
        bad += (got != want) ? 1 : 0;
    }
    if (bad != 0)
    {
        println("[{}/{}] FAIL {}: {} of {} element(s) differ from the operand converted to int first (got {}, want {})",
                variant_name(), mode, label, bad, count, got0, want);
        return 1;
    }
    println("[{}/{}] ok {}: {}", variant_name(), mode, label, want);
    return 0;
}

/// @brief Every case the variant's proxy offers.
long run_cases(const char* mode)
{
    long fails = 0;
    if constexpr (has_op<Op::mul>())
    {
        fails += run_case<Op::mul>("2 *= 1.5", mode, 2);
    }
    if constexpr (has_op<Op::div>())
    {
        fails += run_case<Op::div>("7 /= 2.5", mode, 7);
    }
    if constexpr (has_op<Op::add>())
    {
        fails += run_case<Op::add>("-1 += 0.5", mode, -1);
    }
    if constexpr (has_op<Op::sub>())
    {
        fails += run_case<Op::sub>("1 -= 0.5", mode, 1);
    }
    if constexpr (has_op<Op::mod>())
    {
        fails += run_case<Op::mod>("10 %= 2^32 + 3", mode, 10);
    }
    return fails;
}

} // namespace

int run(int /*argc*/, char** /*argv*/)
{
    Runtime::init(true);
    println("=== mixed_operand_compound_validation [variant = {}] ===", variant_name());

    long fails = 0;
    {
        const unsigned every          = test_hooks::force_defer_every;
        test_hooks::force_defer_every = 0U;
        fails += run_cases("direct");
        if constexpr (build_config::use_openmp && test_hooks::force_defer_compiled)
        {
            test_hooks::force_defer_every = 1U; // every write takes the deferred exit
            fails += run_cases("deferred");
        }
        else
        {
            println("[{}] note: no forced-deferral hook in this build, deferred run skipped", variant_name());
        }
        test_hooks::force_defer_every = every;
    }

    release_chunk_pools();
    Runtime::shutdown();
    println("\n[{}] {}", variant_name(),
            fails ? "=== MIXED OPERAND COMPOUND FAILED ===" : "=== MIXED OPERAND COMPOUND OK ===");
    return fails ? 1 : 0;
}

} // namespace validation::mixed_operand_compound_validation
