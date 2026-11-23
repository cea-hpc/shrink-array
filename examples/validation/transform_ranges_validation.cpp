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
// transform_ranges_mt() harness: dst[i] = op(a[i], b[i], ...) by chunks.
// Variant-agnostic (zero / zero_or_value / zero_any, static or dyn).
// ============================================================================
//
// The sources follow a pattern by whole chunks -- T{}, one constant, data --
// shifted from one source to the next, so that a target chunk meets every
// combination: all sources compressed (one call of the operation, the chunk
// filled), some of them, none. The target must read what a std::vector
// computed element by element reads.
//
//   two_sources   two chunked arrays, the whole range from a team, the slices
//                 cut inside chunks (their ends share a chunk);
//   over_data     the same onto a target that already holds data everywhere;
//   kept_compressed  two sources on one pattern: the chunks where both read
//                 T{} are not materialised in the target (builds whose bulk
//                 reads go by blocks: threaded, no statistics, no checks);
//   mixed         three sources: a chunked array of double, a chunked array
//                 of int, a std::vector<double>;
//   element_path  a bool source (read element by element), and a std::vector
//                 as the target;
//   bounds        an empty range, a range past size(), a range inside one
//                 chunk: nothing written outside it;
//   geometry (dyn) a source re-chunked to another size than the target's.
//
// EXIT STATUS: 0 when every check holds, 1 otherwise. Any thread count (the
// tests run it at 1 and 8).

#include <cstddef> // std::size_t
#include <utility> // std::as_const
#include <vector>  // std::vector (the reference, a source, a target)

#include "chunk/internal/chunk_traits.h" // forward decls + default IntType = int
#include "chunk/runtime.h"               // Runtime::init
#include "chunk/internal/zero_chunk.h"   // release_chunk_pools, Runtime::shutdown
#include "chunk/config.h"                // chunk_config::stats, do_check; build_config::chunk_dynamic
#include "chunk/internal/pmp_chunk.h"    // CHUNK_PMP_MAX_THREADS
#include "chunk/threads.h"               // team_size, thread_rank (the slices of the team)
#include "chunk/range_ops.h"             // transform_ranges_mt
#include "chunk/print.h"                 // shrink_array::println

#include "validation_harness.h" // run() is called by validation_driver.cpp
#include "validation_variant.h" // Array, variant_name, begin_phase / end_phase, maybe_set_second_value

namespace validation::transform_ranges_validation
{
using namespace shrink_array;

namespace
{

constexpr double second = 7.0; ///< the constant of the pattern, and the arrays' second value

/// @brief What source @p shift holds at @p index: by whole chunks of @p chunk
///        elements, T{}, the constant, data; @p shift moves the pattern by
///        one chunk.
double pattern(long index, long chunk, int shift)
{
    switch ((index / chunk + shift) % 3)
    {
        case 0:
            return 0.0;
        case 1:
            return second;
        default:
            return 1.0 + static_cast<double>((index + 31 * shift) % 101);
    }
}

/// @brief A chunked array of @p count elements holding pattern(i, chunk, shift),
///        written from a team and compressed.
template <typename T>
Array<T> make_source(long count, long chunk, int shift)
{
    Array<T> arr(static_cast<int>(count));
    maybe_set_second_value(arr, static_cast<T>(second));
#pragma omp parallel
    {
        begin_phase();
#pragma omp for schedule(static)
        for (long i = 0; i < count; ++i)
        {
            arr[static_cast<int>(i)] = static_cast<T>(pattern(i, chunk, shift));
        }
    }
    end_phase();
    arr.shrink_to_fit();
    return arr;
}

/// @brief Calls @p body(first, last) from every thread of a team, on slices
///        of [0, count) cut 7 elements before a multiple of 100: inside
///        chunks, so that two threads share one at each cut.
template <typename Body>
void for_each_slice(long count, Body&& body)
{
#pragma omp parallel
    {
        begin_phase();
        const long team  = team_size();
        const long rank  = thread_rank();
        const auto cut   = [&](long k) { return k >= team ? count : (k == 0 ? 0 : (count * k / team) / 100 * 100 + 93); };
        const long first = cut(rank) > count ? count : cut(rank);
        const long last  = cut(rank + 1) > count ? count : cut(rank + 1);
        if (first < last)
        {
            body(static_cast<std::size_t>(first), static_cast<std::size_t>(last));
        }
    }
    end_phase();
}

/// @brief Compares @p arr with @p want; reports the first differences.
/// @return the number of elements that differ
template <typename Arr>
int verify(const char* step, const Arr& arr, const std::vector<double>& want)
{
    int bad = 0;
    for (std::size_t i = 0; i < want.size(); ++i)
    {
        const double got = static_cast<double>(arr[static_cast<int>(i)]);
        if (got != want[i])
        {
            if (bad < 5)
            {
                println("[{}/{}] FAIL [{}] = {:g}, expected {:g}", variant_name(), step, i, got, want[i]);
            }
            ++bad;
        }
    }
    if (bad == 0)
    {
        println("[{}/{}] OK: {} elements", variant_name(), step, want.size());
    }
    return bad;
}

int check_two_sources(long count, long chunk)
{
    const Array<double> a  = make_source<double>(count, chunk, 0);
    const Array<double> b  = make_source<double>(count, chunk, 1);
    const auto          op = [](double x, double y) { return x + 2.0 * y; };

    std::vector<double> want(static_cast<std::size_t>(count));
    for (long i = 0; i < count; ++i)
    {
        want[static_cast<std::size_t>(i)] = op(pattern(i, chunk, 0), pattern(i, chunk, 1));
    }

    Array<double> dst(static_cast<int>(count));
    maybe_set_second_value(dst, second);
    for_each_slice(count, [&](std::size_t first, std::size_t last) { transform_ranges_mt(dst, first, last, op, a, b); });
    int bad = verify("two_sources", std::as_const(dst), want);

    // Onto a target that holds data everywhere.
    Array<double> full(static_cast<int>(count));
    maybe_set_second_value(full, second);
    for_each_slice(count, [&](std::size_t first, std::size_t last)
                   {
                       for (std::size_t i = first; i < last; ++i)
                       {
                           full[static_cast<int>(i)] = 1000.0 + static_cast<double>(i);
                       }
                   });
    for_each_slice(count, [&](std::size_t first, std::size_t last) { transform_ranges_mt(full, first, last, op, a, b); });
    bad += verify("over_data", std::as_const(full), want);
    return bad;
}

int check_kept_compressed(long count, long chunk)
{
    // Both sources on the same pattern: one chunk in three reads T{} in both.
    const Array<double> a  = make_source<double>(count, chunk, 0);
    const Array<double> b  = make_source<double>(count, chunk, 0);
    const auto          op = [](double x, double y) { return x * y; };

    std::vector<double> want(static_cast<std::size_t>(count));
    long                both_zero = 0;
    for (long i = 0; i < count; ++i)
    {
        want[static_cast<std::size_t>(i)] = op(pattern(i, chunk, 0), pattern(i, chunk, 0));
    }
    const long chunks = (count + chunk - 1) / chunk;
    for (long c = 0; c < chunks; ++c)
    {
        both_zero += pattern(c * chunk, chunk, 0) == 0.0 ? 1 : 0;
    }

    Array<double> dst(static_cast<int>(count));
    maybe_set_second_value(dst, second);
    for_each_slice(count, [&](std::size_t first, std::size_t last) { transform_ranges_mt(dst, first, last, op, a, b); });
    int bad = verify("kept_compressed", std::as_const(dst), want);

    const long written = static_cast<long>(dst.written_chunk_count());
    if constexpr (!chunk_config::stats && !chunk_config::do_check && build_config::threaded)
    {
        if (written > chunks - both_zero)
        {
            println("[{}/kept_compressed] FAIL: {} chunks hold a buffer, at most {} expected ({} of {} read T{{}} in "
                    "both sources)",
                    variant_name(), written, chunks - both_zero, both_zero, chunks);
            ++bad;
        }
        else
        {
            println("[{}/kept_compressed] OK: {} of {} chunks hold a buffer, {} have both sources at T{{}}",
                    variant_name(), written, chunks, both_zero);
        }
    }
    return bad;
}

int check_mixed(long count, long chunk)
{
    const Array<double> a = make_source<double>(count, chunk, 0);
    const Array<int>    b = make_source<int>(count, chunk, 2);
    std::vector<double> v(static_cast<std::size_t>(count));
    for (long i = 0; i < count; ++i)
    {
        v[static_cast<std::size_t>(i)] = 0.5 * static_cast<double>(i % 9);
    }
    const auto op = [](double x, int y, double z) { return x - static_cast<double>(y) + z; };

    std::vector<double> want(static_cast<std::size_t>(count));
    for (long i = 0; i < count; ++i)
    {
        want[static_cast<std::size_t>(i)] =
            op(pattern(i, chunk, 0), static_cast<int>(pattern(i, chunk, 2)), v[static_cast<std::size_t>(i)]);
    }

    Array<double> dst(static_cast<int>(count));
    maybe_set_second_value(dst, second);
    for_each_slice(count,
                   [&](std::size_t first, std::size_t last) { transform_ranges_mt(dst, first, last, op, a, b, v); });
    return verify("mixed", std::as_const(dst), want);
}

int check_element_path(long count, long chunk)
{
    const Array<double> a = make_source<double>(count, chunk, 0);
    Array<bool>         flags(static_cast<int>(count));
    for (long i = 0; i < count; ++i)
    {
        if ((i / 50) % 2 == 1)
        {
            flags[static_cast<int>(i)] = true;
        }
    }
    const auto op = [](double x, bool keep) { return keep ? x : -1.0; };

    std::vector<double> want(static_cast<std::size_t>(count));
    for (long i = 0; i < count; ++i)
    {
        want[static_cast<std::size_t>(i)] = op(pattern(i, chunk, 0), (i / 50) % 2 == 1);
    }

    Array<double> dst(static_cast<int>(count));
    maybe_set_second_value(dst, second);
    const Array<bool>& const_flags = flags;
    for_each_slice(count, [&](std::size_t first, std::size_t last)
                   { transform_ranges_mt(dst, first, last, op, a, const_flags); });
    int bad = verify("element_path", std::as_const(dst), want);

    // A plain container as the target.
    std::vector<double> plain(static_cast<std::size_t>(count), -5.0);
    for_each_slice(count, [&](std::size_t first, std::size_t last)
                   { transform_ranges_mt(plain, first, last, op, a, const_flags); });
    bad += verify("vector_target", plain, want);
    return bad;
}

int check_bounds(long count, long chunk)
{
    const Array<double> a  = make_source<double>(count, chunk, 0);
    const Array<double> b  = make_source<double>(count, chunk, 1);
    const auto          op = [](double x, double y) { return x + y + 1.0; };

    Array<double>       dst(static_cast<int>(count));
    std::vector<double> want(static_cast<std::size_t>(count), 0.0);
    maybe_set_second_value(dst, second);

    const auto size  = static_cast<std::size_t>(count);
    const auto apply = [&](std::size_t first, std::size_t last)
    {
#pragma omp parallel
        {
            begin_phase();
#pragma omp single
            transform_ranges_mt(dst, first, last, op, a, b);
        }
        end_phase();
        for (std::size_t i = first; i < last && i < size; ++i)
        {
            want[i] = op(pattern(static_cast<long>(i), chunk, 0), pattern(static_cast<long>(i), chunk, 1));
        }
    };
    apply(10, 10);                                                               // empty
    apply(size + 5, size + 50);                                                  // past the end
    apply(static_cast<std::size_t>(chunk) + 3, static_cast<std::size_t>(chunk) + 9); // inside one chunk
    apply(size - 20, size + 1000);                                               // clamped
    apply(static_cast<std::size_t>(2 * chunk), static_cast<std::size_t>(3 * chunk)); // one whole chunk
    return verify("bounds", std::as_const(dst), want);
}

int check_geometry([[maybe_unused]] long count, [[maybe_unused]] long chunk)
{
    int bad = 0;
    if constexpr (build_config::chunk_dynamic)
    {
        Array<double>       a  = make_source<double>(count, chunk, 0);
        const Array<double> b  = make_source<double>(count, chunk, 1);
        const auto          op = [](double x, double y) { return 3.0 * x - y; };
        for (const int other : {static_cast<int>(chunk) / 4, static_cast<int>(chunk) * 2})
        {
            a.rechunk_to(other);
            std::vector<double> want(static_cast<std::size_t>(count));
            for (long i = 0; i < count; ++i)
            {
                want[static_cast<std::size_t>(i)] = op(pattern(i, chunk, 0), pattern(i, chunk, 1));
            }
            Array<double> dst(static_cast<int>(count));
            maybe_set_second_value(dst, second);
            const Array<double>& source = a;
            for_each_slice(count, [&](std::size_t first, std::size_t last)
                           { transform_ranges_mt(dst, first, last, op, source, b); });
            bad += verify("geometry", std::as_const(dst), want);
        }
    }
    return bad;
}

} // namespace

int run(int /*argc*/, char** /*argv*/)
{
    Runtime::init(true);

    long chunk = 0;
    {
        const Array<double> probe(1);
        chunk = static_cast<long>(probe.chunk_size());
    }
    const long count = 40 * chunk + 37; // a last chunk that is not full

    println("=== transform_ranges_validation [variant = {}]  threads={}  N={} chunk={} ===", variant_name(),
            CHUNK_PMP_MAX_THREADS, count, chunk);

    int fails = 0;
    fails += check_two_sources(count, chunk);
    fails += check_kept_compressed(count, chunk);
    fails += check_mixed(count, chunk);
    fails += check_element_path(count, chunk);
    fails += check_bounds(count, chunk);
    fails += check_geometry(count, chunk);

    release_chunk_pools();
    Runtime::shutdown();

    println("\n[{}] {}", variant_name(), fails ? "=== TRANSFORM_RANGES FAILED ===" : "=== TRANSFORM_RANGES OK ===");
    return fails ? 1 : 0;
}

} // namespace validation::transform_ranges_validation
