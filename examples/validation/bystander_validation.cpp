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
// Bystander harness. Every variant, default static size.
// ============================================================================
//
// The arrays of a type share state: the constant blocks compressed chunks
// point at (zero_or_value: one per declared second value; zero_any: the
// registry's), the per-rank candidate lists, the pools. An operation on one
// array must leave the others alone. The other harnesses check the array they
// operate on; this one checks two bystanders that the operation never names:
//
//   c  every element the constant v1 (zero_or_value: its declared second
//      value), compressed -- its chunks point at the shared block of v1;
//   d  a ramp, materialised.
//
// `a` (the constant v1 too, compressed onto the same block as `c`) and `b`
// (the constant v2, or a ramp for zero) go through one operation -- copy-
// assign, move-assign, swap, clear, shrink_to_fit -- and then every live
// element of `a` and of `b` is rewritten (sequentially, then in a region).
// Checked: `c` and `d` read what they held, `c` has as many compressed chunks
// as before, and `a` and `b` read what the operation and the writes say.
//
// EXIT STATUS: 0 when every check holds, 1 otherwise.

#include <string>      // std::string
#include <type_traits> // std::conditional_t
#include <utility>     // std::as_const, std::move

#include "chunk/zero_array.h"          // zero::Array
#include "chunk/zero_or_value_array.h" // zero_or_value::Array
#include "chunk/zero_any_array.h"      // zero_any::Array
#include "chunk/runtime.h"             // Runtime::init
#include "chunk/internal/zero_chunk.h" // release_chunk_pools, Runtime::shutdown
#include "chunk/config.h"              // chunk_config::use_zero_any, use_zero_or_value
#include "chunk/print.h"               // shrink_array::println

#include "validation_harness.h" // run() is called by validation_driver.cpp
#include "validation_variant.h" // Array, variant_name, begin_phase / end_phase

namespace validation::bystander_validation
{

using namespace shrink_array;

namespace
{

using Arr = std::conditional_t<
    chunk_config::use_zero_any,
    zero_any::Array<double>,
    std::conditional_t<chunk_config::use_zero_or_value, zero_or_value::Array<double>, zero::Array<double>>>;
using Index = Arr::IndexType;

constexpr bool has_second_value = chunk_config::use_zero_any || chunk_config::use_zero_or_value;

constexpr double v1 = has_second_value ? 1.0 : 0.0; ///< `c`'s and `a`'s constant
constexpr double v2 = 2.0;                          ///< `b`'s constant (zero: `b` is a ramp)
constexpr double wa = 9.0;                          ///< what `a` is rewritten to
constexpr double wb = 13.0;                         ///< what `b` is rewritten to

constexpr long n_chunks = 16;

double ramp(long i)
{
    return 100.0 + static_cast<double>(i % 7);
}

/// @brief What `b` holds before the operation.
double b_init(long i)
{
    return has_second_value ? v2 : ramp(i);
}

template <typename A> // a template: the discarded branch is not instantiated for zero / zero_any
void declare(A& arr, [[maybe_unused]] double value)
{
    if constexpr (chunk_config::use_zero_or_value)
    {
        arr.set_second_value(value);
    }
}

long compressed(const Arr& arr, long chunk_sz)
{
    long count = 0;
    for (long c = 0; c < arr.size() / chunk_sz; ++c)
    {
        count += arr.written(static_cast<Index>(c * chunk_sz)) ? 0 : 1;
    }
    return count;
}

template <typename F>
long mismatches(const Arr& arr, long count, F&& expected)
{
    if (static_cast<long>(arr.size()) != count)
    {
        return count == 0 ? static_cast<long>(arr.size()) : count;
    }
    long bad = 0;
    for (long i = 0; i < count; ++i)
    {
        bad += (arr[static_cast<Index>(i)] != expected(i)) ? 1 : 0;
    }
    return bad;
}

/// @brief Every element of @p arr set to @p value: half sequentially, half in a region.
void rewrite(Arr& arr, double value)
{
    const auto count = static_cast<long>(arr.size());
    for (long i = 0; i < count / 2; ++i)
    {
        arr[static_cast<Index>(i)] = value;
    }
#pragma omp parallel
    {
        begin_phase();
#pragma omp for schedule(static)
        for (long i = count / 2; i < count; ++i)
        {
            arr[static_cast<Index>(i)] = value;
        }
    }
    end_phase();
}

enum class Op
{
    copy_assign,
    move_assign,
    swap,
    clear,
    shrink_to_fit,
};

const char* op_name(Op op)
{
    using enum Op;
    switch (op)
    {
        case copy_assign:
            return "a = b";
        case move_assign:
            return "a = std::move(b)";
        case swap:
            return "a.swap(b)";
        case clear:
            return "a.clear()";
        case shrink_to_fit:
            return "a.shrink_to_fit()";
    }
    return "?";
}

int run_case(Op op)
{
    const std::string tag = std::string("[bystander/") + op_name(op) + "]";

    Arr        c;
    Arr        d;
    Arr        a;
    Arr        b;
    const auto chunk_sz = static_cast<long>(c.chunk_size());
    const long count    = n_chunks * chunk_sz;
    for (Arr* arr : {&c, &d, &a, &b})
    {
        arr->resize(static_cast<Index>(count));
    }
    declare(c, v1);
    declare(a, v1);
    declare(b, v2);
    for (long i = 0; i < count; ++i)
    {
        c[static_cast<Index>(i)] = v1;
        d[static_cast<Index>(i)] = ramp(i);
        a[static_cast<Index>(i)] = v1;
        b[static_cast<Index>(i)] = b_init(i);
    }
    c.shrink_to_fit();

    const long c_before = compressed(std::as_const(c), chunk_sz);
    if (c_before != n_chunks)
    {
        println("{} FAIL premise: c has {}/{} chunks compressed", tag, c_before, n_chunks);
        return 1;
    }

    // The operation, then the writes; a_exp / b_exp: what `a` and `b` must read.
    long   a_count  = count;
    long   b_count  = count;
    double a_after  = wa;
    bool   b_is_old = false; ///< `b` holds `a`'s old content (swap)
    using enum Op;
    switch (op)
    {
        case copy_assign:
            a = b;
            break;
        case move_assign:
            a       = std::move(b);
            b_count = static_cast<long>(b.size()); // valid, unspecified: whatever it reads now
            break;
        case swap:
            a.swap(b);
            b_is_old = true;
            break;
        case clear:
            a.clear();
            a_count = 0;
            break;
        case shrink_to_fit:
            a.shrink_to_fit();
            break;
    }
    if (a_count != 0)
    {
        rewrite(a, wa);
    }
    // After the move `b` is not rewritten (its content is unspecified).
    const bool b_rewritten = op != move_assign && op != swap;
    if (b_rewritten)
    {
        rewrite(b, wb);
    }

    const Arr& ca    = a;
    const Arr& cb    = b;
    const long bad_c = mismatches(std::as_const(c), count, [](long) { return v1; });
    const long bad_d = mismatches(std::as_const(d), count, [](long i) { return ramp(i); });
    const long bad_a =
        a_count == 0 ? static_cast<long>(ca.size()) : mismatches(ca, a_count, [&](long) { return a_after; });
    long bad_b = 0;
    if (b_rewritten)
    {
        bad_b = mismatches(cb, b_count, [](long) { return wb; });
    }
    else if (b_is_old)
    {
        bad_b = mismatches(cb, b_count, [](long) { return v1; });
    }
    const long c_after = compressed(std::as_const(c), chunk_sz);

    println("{} c: {} bad, {}/{} compressed (was {}); d: {} bad; a: {} bad; b: {} bad", tag, bad_c, c_after, n_chunks,
            c_before, bad_d, bad_a, bad_b);
    const int fails = (bad_c != 0 ? 1 : 0) + (bad_d != 0 ? 1 : 0) + (bad_a != 0 ? 1 : 0) + (bad_b != 0 ? 1 : 0) +
                      (c_after < c_before ? 1 : 0);
    if (fails != 0)
    {
        println("{} FAIL", tag);
    }
    return fails;
}

// ── queued candidates of a bystander, then a temporary dies ─────────────────
//
// `e` is rewritten to the constant v1 (first to 5, then to v1, so that every
// chunk was materialised and becomes constant): its chunks are queued for the
// next phase end. Before that phase end a temporary array of the type is
// destroyed. Its teardown must not empty `e`'s queue: once the phase ends,
// `e` has as many compressed chunks as in the control run (no temporary).

enum class Queue
{
    sequential, ///< `e` rewritten at a sequential point, between phases
    in_region,  ///< `e` rewritten in a region; the temporary dies after it, before the phase end
};

const char* queue_name(Queue q)
{
    return q == Queue::sequential ? "sequential writes" : "region writes";
}

long queued_run(Queue q, bool with_temporary)
{
    Arr        e;
    const auto chunk_sz = static_cast<long>(e.chunk_size());
    const long count    = n_chunks * chunk_sz;
    e.resize(static_cast<Index>(count));
    declare(e, v1);
    for (long i = 0; i < count; ++i)
    {
        e[static_cast<Index>(i)] = 5.0;
    }
    e.shrink_to_fit(); // whatever the fill queued is drained: a clean start
    if (q == Queue::sequential)
    {
        for (long i = 0; i < count; ++i)
        {
            e[static_cast<Index>(i)] = v1;
        }
        if (with_temporary)
        {
            Arr tmp;
            tmp.resize(static_cast<Index>(chunk_sz));
            tmp[0] = 4.0;
        }
#pragma omp parallel
        {
            begin_phase();
        }
        end_phase();
    }
    else
    {
#pragma omp parallel
        {
            begin_phase();
#pragma omp for schedule(static)
            for (long i = 0; i < count; ++i)
            {
                e[static_cast<Index>(i)] = v1;
            }
        }
        if (with_temporary)
        {
            Arr tmp;
            tmp.resize(static_cast<Index>(chunk_sz));
            tmp[0] = 4.0;
        }
        end_phase();
    }
    const long bad = mismatches(std::as_const(e), count, [](long) { return v1; });
    return bad != 0 ? -bad : compressed(std::as_const(e), chunk_sz);
}

int run_queued(Queue q)
{
    const std::string tag     = std::string("[bystander/queued, ") + queue_name(q) + "]";
    const long        control = queued_run(q, false);
    const long        tested  = queued_run(q, true);
    println("{} e compressed: {}/{} in the control run, {}/{} with a temporary destroyed before the phase end", tag,
            control, n_chunks, tested, n_chunks);
    if (control < 0 || tested < 0)
    {
        println("{} FAIL: e reads elements other than {:g}", tag, v1);
        return 1;
    }
    if (tested < control)
    {
        println("{} FAIL: the temporary's teardown dropped {} queued chunk(s) of e", tag, control - tested);
        return 1;
    }
    return 0;
}

} // namespace

int run(int /*argc*/, char** /*argv*/)
{
    using enum Op;
    Runtime::init(true);
    println("=== bystander_validation ({}) ===", variant_name());
    int fails = 0;
    for (Op op : {copy_assign, move_assign, swap, clear, shrink_to_fit})
    {
        fails += run_case(op);
    }
    fails += run_queued(Queue::sequential);
    fails += run_queued(Queue::in_region);
    release_chunk_pools();
    Runtime::shutdown();
    println("\n[{}] {}", variant_name(), fails ? "=== BYSTANDER FAILED ===" : "=== BYSTANDER OK ===");
    return fails ? 1 : 0;
}

} // namespace validation::bystander_validation
