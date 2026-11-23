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
// Differential harness: random sequences of container operations applied to
// an Array<T> and to a std::vector<T>, the two compared after every one.
// Variant-agnostic (zero / zero_or_value / zero_any, static or dyn).
// ============================================================================
//
// The other harnesses each check a scenario someone chose. This one draws the
// operations: the std::vector gives the expected content for any sequence, so
// nothing has to be foreseen -- an assign() after a swap() after pop_back()s
// on a compressed array is drawn sooner or later.
//
// One run (a seed) holds two arrays, a and b, and their references. A step
// draws one operation among:
//   element writes   a[i] = v, runs of one value over whole chunks, the
//                    compound operations (+= -= *= ++ -- and a[i]++),
//                    a[i] = a[j], a[i] = b[j], front() / back() / at();
//   size             resize, reserve, push_back, emplace_back, pop_back,
//                    clear, assign(n, v), assign(first, last);
//   whole array      swap, copy- and move-construction and assignment;
//   compression      compress, shrink_to_fit, reclaim_second_values (zero_any);
//   ranges           fill_range, copy_range, read_range, for_each_value,
//                    fill_range_mt and copy_range_mt from two threads;
//   raw access       alloc_chunk(i) then operator()(i);
//   exchange         export_array / import_array into a third array;
//   parallel         a phase writing every element, or two elements in three;
//   geometry (dyn)   rechunk_to a size drawn from 64 to 2048.
// The values are drawn so that chunks become uniform often: T{} and the
// second value two times in three.
//
// After each step both arrays are read back in full, by operator[] and by the
// iterators. On the first difference the harness prints the seed, the element
// type and the last operations, and stops: `differential_validation <seed> 1`
// replays that run.
//
// What it does not see: a race that leaves the values right, a leak. Run it
// in a check build and under ASan for those; the TSan builds and the GenMC
// models remain the reference for the concurrent paths.
//
// USAGE: differential_validation [first_seed [seeds [steps]]]
//        (defaults: 1, 40, 120). Element types double, int and bool.
// EXIT STATUS: 0 when no run differs, 1 otherwise. Any thread count (the
// tests run it at 1 and 8).

#include <cstddef>     // std::size_t
#include <cstdlib>     // std::strtoul, std::atoi
#include <format>      // std::format
#include <random>      // std::mt19937_64
#include <string>      // std::string
#include <type_traits> // std::is_same_v
#include <utility>     // std::move, std::as_const
#include <vector>      // std::vector (the reference)

#include "chunk/internal/chunk_traits.h" // forward decls + default IntType = int
#include "chunk/runtime.h"               // Runtime::init
#include "chunk/internal/zero_chunk.h"   // release_chunk_pools, Runtime::shutdown
#include "chunk/config.h"                // build_config::chunk_dynamic
#include "chunk/internal/pmp_chunk.h"    // CHUNK_PMP_MAX_THREADS
#include "chunk/range_ops.h"             // fill_range, copy_range, read_range, for_each_value, *_mt
#include "chunk/chunk_exchange.h"        // export_array, import_array
#include "chunk/print.h"                 // shrink_array::println

#include "validation_harness.h" // run() is called by validation_driver.cpp
#include "validation_variant.h" // Array, variant_name, begin_phase / end_phase, maybe_set_second_value

namespace validation::differential_validation
{
using namespace shrink_array;

namespace
{

/// @brief The name of the element type, for the report.
template <typename T>
constexpr const char* type_name()
{
    if constexpr (std::is_same_v<T, double>)
    {
        return "double";
    }
    else if constexpr (std::is_same_v<T, int>)
    {
        return "int";
    }
    else
    {
        return "bool";
    }
}

/// @brief One run: the two arrays, their references, the generator, the trace.
template <typename T>
struct Run
{
    unsigned long            seed;
    std::mt19937_64          rng;
    T                        second; ///< the second value the arrays declare (when with_second)
    bool                     with_second;
    Array<T>                 a;
    Array<T>                 b;
    std::vector<T>           ra; ///< what a must hold
    std::vector<T>           rb; ///< what b must hold
    std::vector<std::string> trace;
};

/// @brief A number in [0, bound).
template <typename T>
int draw(Run<T>& run, int bound)
{
    return static_cast<int>(run.rng() % static_cast<unsigned long>(bound));
}

/// @brief A value: T{} or the second value two times in three, so that chunks
///        become uniform; another constant, or any value, otherwise.
template <typename T>
T draw_value(Run<T>& run)
{
    switch (draw(run, 6))
    {
        case 0:
        case 1:
            return T{};
        case 2:
        case 3:
            return run.second;
        case 4:
            return static_cast<T>(3);
        default:
            return static_cast<T>(draw(run, 1000) - 500);
    }
}

/// @brief A size: within the first chunk one time in four, a few chunks otherwise.
template <typename T>
int draw_size(Run<T>& run)
{
    return draw(run, 4) == 0 ? draw(run, 40) : 256 * draw(run, 6) + draw(run, 300);
}

/// @brief [first, last) inside [0, count).
template <typename T>
void draw_range(Run<T>& run, int count, int& first, int& last)
{
    first = draw(run, count);
    last  = first + draw(run, count - first + 1);
}

/// @brief Reports a difference: where, the seed and the last operations.
template <typename T>
void report(const Run<T>& run, const std::string& what)
{
    println("[{}/{}] MISMATCH: {} (seed {}, {} operations)", variant_name(), type_name<T>(), what, run.seed,
            run.trace.size());
    const std::size_t shown = 20;
    for (std::size_t k = run.trace.size() > shown ? run.trace.size() - shown : 0; k < run.trace.size(); ++k)
    {
        println("    {}", run.trace[k]);
    }
}

/// @brief True when @p arr holds what @p ref holds: the size, every element by
///        operator[], every element by the iterators.
template <typename T>
bool same(const Run<T>& run, const Array<T>& arr, const std::vector<T>& ref, const char* which)
{
    if (static_cast<std::size_t>(arr.size()) != ref.size())
    {
        report(run, std::format("{}: size {} instead of {}", which, arr.size(), ref.size()));
        return false;
    }
    for (std::size_t i = 0; i < ref.size(); ++i)
    {
        const T got = arr[static_cast<int>(i)];
        if (got != ref[i])
        {
            report(run, std::format("{}[{}] reads {} instead of {}", which, i, got, T{ref[i]}));
            return false;
        }
    }
    std::size_t i = 0;
    for (auto it = arr.begin(); it != arr.end(); ++it, ++i)
    {
        if (*it != ref[i])
        {
            report(run, std::format("{}: the iterator reads {} at {} instead of {}", which, T{*it}, i, T{ref[i]}));
            return false;
        }
    }
    return true;
}

// ---- element writes --------------------------------------------------------

template <typename T>
void op_writes(Run<T>& run)
{
    const int count = static_cast<int>(run.ra.size());
    for (int k = draw(run, 50); count != 0 && k >= 0; --k)
    {
        const int i = draw(run, count);
        const T   v = draw_value(run);
        run.a[i]    = v;
        run.ra[i]   = v;
    }
    run.trace.emplace_back("a[i] = v, some elements");
}

template <typename T>
void op_run_of_one_value(Run<T>& run)
{
    const int count = static_cast<int>(run.ra.size());
    if (count == 0)
    {
        return;
    }
    int first = 0;
    int last  = 0;
    draw_range(run, count, first, last);
    const T v = draw_value(run);
    for (int i = first; i < last; ++i)
    {
        run.a[i]  = v;
        run.ra[i] = v;
    }
    run.trace.push_back(std::format("a[i] = {} for i in [{}, {})", v, first, last));
}

template <typename T>
bool op_compound(Run<T>& run)
{
    if constexpr (!std::is_same_v<T, bool>)
    {
        const int count = static_cast<int>(run.ra.size());
        for (int k = draw(run, 30); count != 0 && k >= 0; --k)
        {
            const int i = draw(run, count);
            const T   v = static_cast<T>(draw(run, 5) - 2);
            switch (draw(run, 6))
            {
                case 0:
                    run.a[i] += v;
                    run.ra[i] += v;
                    break;
                case 1:
                    run.a[i] -= v;
                    run.ra[i] -= v;
                    break;
                case 2:
                    run.a[i] *= v;
                    run.ra[i] *= v;
                    break;
                case 3:
                    ++run.a[i];
                    ++run.ra[i];
                    break;
                case 4:
                    --run.a[i];
                    --run.ra[i];
                    break;
                default:
                {
                    const T old = run.a[i]++;
                    if (old != run.ra[i])
                    {
                        report(run, std::format("a[{}]++ returns {} instead of {}", i, old, run.ra[i]));
                        return false;
                    }
                    run.ra[i]++;
                    break;
                }
            }
        }
        run.trace.emplace_back("compound operations, some elements");
    }
    return true;
}

template <typename T>
void op_copy_elements(Run<T>& run, bool from_b)
{
    const int count  = static_cast<int>(run.ra.size());
    const int source = static_cast<int>(from_b ? run.rb.size() : run.ra.size());
    for (int k = draw(run, 30); count != 0 && source != 0 && k >= 0; --k)
    {
        const int i = draw(run, count);
        const int j = draw(run, source);
        if (from_b)
        {
            run.a[i]  = run.b[j];
            run.ra[i] = run.rb[j];
        }
        else
        {
            run.a[i]  = run.a[j];
            run.ra[i] = run.ra[j];
        }
    }
    run.trace.emplace_back(from_b ? "a[i] = b[j], some elements" : "a[i] = a[j], some elements");
}

template <typename T>
void op_ends_and_at(Run<T>& run)
{
    const int count = static_cast<int>(run.ra.size());
    if (count == 0)
    {
        return;
    }
    T v            = draw_value(run);
    run.a.front()  = v;
    run.ra.front() = v;
    v              = draw_value(run);
    run.a.back()   = v;
    run.ra.back()  = v;
    const int i    = draw(run, count);
    v              = draw_value(run);
    run.a.at(i)    = v;
    run.ra[i]      = v;
    run.trace.emplace_back("front() = , back() = , at(i) =");
}

// ---- size ------------------------------------------------------------------

template <typename T>
void op_resize(Run<T>& run)
{
    const int size = draw_size(run);
    run.trace.push_back(std::format("resize({}) from {}", size, run.ra.size()));
    run.a.resize(size);
    run.ra.resize(static_cast<std::size_t>(size));
}

template <typename T>
void op_push_back(Run<T>& run)
{
    const int count = draw(run, 600) + 1;
    for (int k = 0; k < count; ++k)
    {
        const T v = draw_value(run);
        if (k % 2 == 0)
        {
            run.a.push_back(v);
        }
        else
        {
            run.a.emplace_back(v);
        }
        run.ra.push_back(v);
    }
    run.trace.push_back(std::format("push_back / emplace_back x {}", count));
}

template <typename T>
void op_pop_back(Run<T>& run)
{
    int count = 0;
    for (int k = draw(run, 300); k >= 0 && !run.ra.empty(); --k, ++count)
    {
        run.a.pop_back();
        run.ra.pop_back();
    }
    run.trace.push_back(std::format("pop_back x {}", count));
}

template <typename T>
void op_assign(Run<T>& run)
{
    if (draw(run, 2) == 0)
    {
        const int size = draw_size(run);
        const T   v    = draw_value(run);
        run.a.assign(size, v);
        run.ra.assign(static_cast<std::size_t>(size), v);
        run.trace.push_back(std::format("assign({}, {})", size, v));
    }
    else
    {
        std::vector<T> source(static_cast<std::size_t>(draw(run, 900)));
        for (auto&& x : source)
        {
            x = draw_value(run);
        }
        run.a.assign(source.begin(), source.end());
        run.ra = source;
        run.trace.push_back(std::format("assign(first, last), {} elements", source.size()));
    }
}

// ---- whole array -----------------------------------------------------------

template <typename T>
bool op_whole_array(Run<T>& run)
{
    switch (draw(run, 6))
    {
        case 0:
            // At dyn the two geometries may differ (a re-chunk of a): the
            // geometry goes with the chunks, as in a move.
            run.a.swap(run.b);
            run.ra.swap(run.rb);
            run.trace.emplace_back("a.swap(b)");
            break;
        case 1:
            run.b  = run.a;
            run.rb = run.ra;
            run.trace.emplace_back("b = a");
            break;
        case 2:
            run.a  = run.b;
            run.ra = run.rb;
            run.trace.emplace_back("a = b");
            break;
        case 3:
        {
            run.trace.emplace_back("Array c(a); b = c");
            const Array<T> copy(run.a);
            if (!same(run, copy, run.ra, "the copy of a"))
            {
                return false;
            }
            run.b  = copy;
            run.rb = run.ra;
            break;
        }
        case 4:
        {
            Array<T> copy(run.a);
            run.a = std::move(copy);
            run.trace.emplace_back("a = std::move(copy of a)");
            break;
        }
        default:
        {
            Array<T> moved(std::move(run.a));
            run.a = std::move(moved);
            run.trace.emplace_back("Array c(std::move(a)); a = std::move(c)");
            break;
        }
    }
    return true;
}

template <typename T>
void op_refill_b(Run<T>& run)
{
    const int size = draw(run, 900);
    run.b.resize(size);
    run.rb.resize(static_cast<std::size_t>(size));
    for (int i = 0; i < size; ++i)
    {
        if (draw(run, 3) == 0)
        {
            const T v = draw_value(run);
            run.b[i]  = v;
            run.rb[i] = v;
        }
    }
    run.trace.push_back(std::format("b.resize({}), one element in three written", size));
}

// ---- compression -----------------------------------------------------------

template <typename T>
void op_compress(Run<T>& run)
{
    switch (draw(run, 3))
    {
        case 0:
            run.a.compress();
            run.trace.emplace_back("compress()");
            break;
        case 1:
            run.a.shrink_to_fit();
            run.trace.emplace_back("shrink_to_fit()");
            break;
        default:
            if constexpr (requires { Array<T>::reclaim_second_values(); })
            {
                (void)Array<T>::reclaim_second_values();
                run.trace.emplace_back("reclaim_second_values()");
            }
            break;
    }
}

// ---- ranges ----------------------------------------------------------------

template <typename T>
void op_fill_range(Run<T>& run)
{
    const int count = static_cast<int>(run.ra.size());
    if (count == 0)
    {
        return;
    }
    int first = 0;
    int last  = 0;
    draw_range(run, count, first, last);
    const T v = draw_value(run);
    fill_range(run.a, static_cast<std::size_t>(first), static_cast<std::size_t>(last), v);
    for (int i = first; i < last; ++i)
    {
        run.ra[i] = v;
    }
    run.trace.push_back(std::format("fill_range [{}, {}) = {}", first, last, v));
}

/// @brief [first, last) values to copy: one value throughout, or any.
template <typename T>
std::vector<T> draw_source(Run<T>& run, int count)
{
    std::vector<T> source(static_cast<std::size_t>(count));
    const T        one     = draw_value(run);
    const bool     uniform = draw(run, 2) == 0;
    for (auto&& x : source)
    {
        x = uniform ? one : draw_value(run);
    }
    return source;
}

template <typename T>
void op_copy_range(Run<T>& run)
{
    const int count = static_cast<int>(run.ra.size());
    if (count == 0)
    {
        return;
    }
    int first = 0;
    int last  = 0;
    draw_range(run, count, first, last);
    const std::vector<T> source = draw_source(run, last - first);
    copy_range(run.a, static_cast<std::size_t>(first), static_cast<std::size_t>(last), source.begin());
    for (int i = first; i < last; ++i)
    {
        run.ra[i] = source[static_cast<std::size_t>(i - first)];
    }
    run.trace.push_back(std::format("copy_range [{}, {})", first, last));
}

template <typename T>
bool op_bulk_reads(Run<T>& run)
{
    const int count = static_cast<int>(run.ra.size());
    if (count == 0)
    {
        return true;
    }
    int first = 0;
    int last  = 0;
    draw_range(run, count, first, last);
    run.trace.push_back(std::format("read_range and for_each_value [{}, {})", first, last));
    std::vector<T> out(static_cast<std::size_t>(last - first));
    read_range(std::as_const(run.a), static_cast<std::size_t>(first), static_cast<std::size_t>(last), out.begin());
    std::size_t seen = 0;
    bool        ok   = true;
    for_each_value(std::as_const(run.a), static_cast<std::size_t>(first), static_cast<std::size_t>(last),
                   [&](const T& v)
                   {
                       ok = ok && v == run.ra[static_cast<std::size_t>(first) + seen];
                       ++seen;
                   });
    for (int i = first; ok && i < last; ++i)
    {
        ok = out[static_cast<std::size_t>(i - first)] == run.ra[i];
    }
    if (!ok || seen != out.size())
    {
        report(run, "read_range or for_each_value differs from operator[]");
        return false;
    }
    return true;
}

template <typename T>
void op_ranges_mt(Run<T>& run)
{
    const int count = static_cast<int>(run.ra.size());
    if (count < 2)
    {
        return;
    }
    const auto half = static_cast<std::size_t>(count / 2);
    const auto size = static_cast<std::size_t>(count);
    if (draw(run, 2) == 0)
    {
        const T v = draw_value(run);
        begin_phase();
#pragma omp parallel sections
        {
#pragma omp section
            fill_range_mt(run.a, std::size_t{0}, half, v);
#pragma omp section
            fill_range_mt(run.a, half, size, v);
        }
        end_phase();
        run.ra.assign(size, v);
        run.trace.push_back(std::format("fill_range_mt, two halves = {}", v));
    }
    else
    {
        const std::vector<T> source = draw_source(run, count);
        begin_phase();
#pragma omp parallel sections
        {
#pragma omp section
            copy_range_mt(run.a, std::size_t{0}, half, source.begin());
#pragma omp section
            copy_range_mt(run.a, half, size, source.begin() + static_cast<std::ptrdiff_t>(half));
        }
        end_phase();
        run.ra = source;
        run.trace.emplace_back("copy_range_mt, two halves");
    }
}

// ---- raw access, exchange, parallel, geometry -------------------------------

template <typename T>
void op_alloc_chunk(Run<T>& run)
{
    const int count = static_cast<int>(run.ra.size());
    for (int k = draw(run, 4); count != 0 && k >= 0; --k)
    {
        const int i = draw(run, count);
        const T   v = draw_value(run);
        run.a.alloc_chunk(i);
        run.a(i)  = v;
        run.ra[i] = v;
        run.trace.push_back(std::format("alloc_chunk({0}); a({0}) = {1}", i, v));
    }
}

template <typename T>
bool op_exchange(Run<T>& run)
{
    run.trace.emplace_back("export_array(a), import_array into a third array");
    const auto state = export_array(run.a);
    Array<T>   imported(0);
    if (run.with_second)
    {
        maybe_set_second_value(imported, run.second);
    }
    // false: another chunk size (dyn, after a re-chunk of a), nothing imported.
    return !import_array(imported, state) || same(run, imported, run.ra, "the import of a");
}

/// @brief What the parallel phase writes at @p index: blocks of 300 elements,
///        alternately the second value and mostly T{}.
template <typename T>
T phase_value(const Run<T>& run, int index)
{
    if ((index / 300) % 2 != 0)
    {
        return run.second;
    }
    return index % 5 == 0 ? static_cast<T>(index) : T{};
}

template <typename T>
void op_parallel_phase(Run<T>& run)
{
    const int  count = static_cast<int>(run.ra.size());
    const bool every = draw(run, 2) == 0;
    const T    v     = draw_value(run);
    begin_phase();
#pragma omp parallel for
    for (int i = 0; i < count; ++i)
    {
        if (every)
        {
            run.a[i] = v;
        }
        else if (i % 3 != 1)
        {
            run.a[i] = phase_value(run, i);
        }
    }
    end_phase();
    for (int i = 0; i < count; ++i)
    {
        if (every)
        {
            run.ra[i] = v;
        }
        else if (i % 3 != 1)
        {
            run.ra[i] = phase_value(run, i);
        }
    }
    run.trace.push_back(every ? std::format("parallel phase, every element = {}", v)
                              : std::string("parallel phase, two elements in three"));
}

template <typename T>
void op_rechunk(Run<T>& run)
{
    if constexpr (build_config::chunk_dynamic)
    {
        const int size = 64 << draw(run, 6);
        run.a.rechunk_to(size);
        run.trace.push_back(std::format("rechunk_to({})", size));
    }
}

/// @brief One step: an operation drawn, then both arrays read back.
template <typename T>
bool step(Run<T>& run)
{
    bool ok = true;
    switch (draw(run, 30))
    {
        case 0:
        case 1:
        case 2:
            op_writes(run);
            break;
        case 3:
        case 4:
            op_run_of_one_value(run);
            break;
        case 5:
            ok = op_compound(run);
            break;
        case 6:
            op_copy_elements(run, /*from_b=*/false);
            break;
        case 7:
            op_copy_elements(run, /*from_b=*/true);
            break;
        case 8:
            op_ends_and_at(run);
            break;
        case 9:
            op_resize(run);
            break;
        case 10:
            op_push_back(run);
            break;
        case 11:
            op_pop_back(run);
            break;
        case 12:
            if (draw(run, 4) == 0)
            {
                run.a.clear();
                run.ra.clear();
                run.trace.emplace_back("clear()");
            }
            else
            {
                run.a.reserve(static_cast<int>(run.ra.size()) + draw(run, 2000));
                run.trace.emplace_back("reserve()");
            }
            break;
        case 13:
            op_assign(run);
            break;
        case 14:
        case 15:
        case 16:
            ok = op_whole_array(run);
            break;
        case 17:
            op_refill_b(run);
            break;
        case 18:
        case 19:
            op_compress(run);
            break;
        case 20:
            op_fill_range(run);
            break;
        case 21:
            op_copy_range(run);
            break;
        case 22:
            ok = op_bulk_reads(run);
            break;
        case 23:
            op_ranges_mt(run);
            break;
        case 24:
            op_alloc_chunk(run);
            break;
        case 25:
            ok = op_exchange(run);
            break;
        case 26:
        case 27:
            op_parallel_phase(run);
            break;
        default:
            op_rechunk(run);
            break;
    }
    return ok && same(run, run.a, run.ra, "a") && same(run, run.b, run.rb, "b");
}

/// @brief One run of @p steps operations from @p seed.
/// @return true when the arrays never differed from their references.
template <typename T>
bool run_seed(unsigned long seed, int steps)
{
    std::mt19937_64 rng(seed);
    const bool      with_second = seed % 2 == 0;
    const int       size        = rng() % 4 == 0 ? static_cast<int>(rng() % 40)
                                                 : 256 * static_cast<int>(rng() % 6) + static_cast<int>(rng() % 300);
    Run<T>          run{seed,
                        rng,
                        static_cast<T>(7),
                        with_second,
                        Array<T>(size),
                        Array<T>(0),
                        std::vector<T>(static_cast<std::size_t>(size), T{}),
                        {},
                        {}};
    if (with_second)
    {
        maybe_set_second_value(run.a, run.second);
        maybe_set_second_value(run.b, run.second);
    }
    for (int k = 0; k < steps; ++k)
    {
        if (!step(run))
        {
            return false;
        }
    }
    return true;
}

} // namespace

int run(int argc, char** argv)
{
    const unsigned long first = argc > 1 ? std::strtoul(argv[1], nullptr, 10) : 1;
    const int           seeds = argc > 2 ? std::atoi(argv[2]) : 40;
    const int           steps = argc > 3 ? std::atoi(argv[3]) : 120;

    Runtime::init(true);

    println("=== differential_validation [variant = {}]  threads={}  seeds {}..{}, {} steps ===", variant_name(),
            CHUNK_PMP_MAX_THREADS, first, first + static_cast<unsigned long>(seeds) - 1, steps);

    int fails = 0;
    for (unsigned long seed = first; seed < first + static_cast<unsigned long>(seeds) && fails == 0; ++seed)
    {
        fails += run_seed<double>(seed, steps) ? 0 : 1;
        fails += run_seed<int>(seed, steps) ? 0 : 1;
        fails += run_seed<bool>(seed, steps) ? 0 : 1;
    }

    release_chunk_pools();
    Runtime::shutdown();

    println("\n[{}] {}", variant_name(), fails ? "=== DIFFERENTIAL FAILED ===" : "=== DIFFERENTIAL OK ===");
    return fails ? 1 : 0;
}

} // namespace validation::differential_validation
