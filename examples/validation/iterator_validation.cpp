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
// const_iterator harness. Variant-agnostic: zero / zero_one_value / zero_any follow the
// compile-time macro (CHUNK_VARIANT_ZERO*), like the other harnesses here.
// ============================================================================
//
// Guards the random-access contract of the read-only iterator, against a
// std::vector holding the same values:
//
//   concepts  (compile time) std::random_access_iterator and
//             random_access_range hold; the CLASSIC iterator_category is
//             random_access too -- std::lower_bound, std::distance and
//             std::advance dispatch on it, not on the C++20 concept, and a
//             bidirectional tag would leave them O(n) moves; the iterator is
//             NOT writable, so std::sortable is false: no in-place sort, by
//             construction (an element folded into a constant has no address).
//   jumps     it + d, d + it, it - d, it += d, it -= d, it[d], it2 - it1 and
//             the four orderings, from positions on and around chunk
//             boundaries, by deltas of 0, +-1, +-2, +-(S-1), +-S, +-(S+1) and
//             more; the value under every jump, and a ++ / -- right after it
//             (the cursor must be coherent after a seek); end() - 1,
//             std::distance, std::advance.
//   searches  std::lower_bound / upper_bound / equal_range / binary_search and
//             their std::ranges forms, for values present (runs of 3 equal
//             values, a folded run of the first constant, a run of the second
//             value), between them, below and above the range; std::lower_bound
//             on the reverse range; then the same searches from every thread at
//             once inside a parallel region.
//   reverse   rbegin()/rend() read back the reversed content, rbegin()[k].
//   sort      the recipe for the sort the iterator cannot do: a copy into a
//             std::vector, std::sort, assign() back.
//
// The array is materialised with plain values first, then overwritten with a
// sorted pattern and compressed, so that the jumps and the searches land in
// chunks folded to a constant as well as in materialised ones. The harness
// checks that the folding happened -- the premise of the test.
//
// EXIT STATUS: 0 when every check holds, 1 otherwise. Any thread count (the
// tests run it at 1 and 8).

#include <algorithm>        // std::lower_bound, std::sort, std::ranges::equal_range
#include <cstdio>           // std::snprintf
#include <initializer_list> // std::initializer_list (range-for over a braced list)
#include <functional>       // std::greater
#include <iterator>         // std::random_access_iterator, std::iterator_traits, std::distance
#include <ranges>           // std::ranges::random_access_range
#include <span>             // std::span (for_each_span)
#include <type_traits>      // std::conditional_t, std::is_same_v
#include <vector>           // std::vector
#include <format>           // std::format
#include <utility>          // std::forward

#include "chunk/chunk_traits.h"          // forward decls + default IntType = int
#include "chunk/zero_array.h"            // zero::Array
#include "chunk/zero_one_value_array.h"  // zero_one_value::Array
#include "chunk/zero_any_array.h"        // zero_any::Array
#include "chunk/param_chunk.h"           // ParamChunk::init_indexes
#include "chunk/zero_chunk.h"            // release_chunk_pools, destroy_block_chunk
#include "chunk/config.h"                // chunk_config::use_zero*, build_config::has_not_ompt
#include "chunk/PMP_CHUNK.h"             // CHUNK_PMP_MAX_THREADS
#include "chunk/manage_ompt.h"           // begin_parallel_phase / finalize_parallel_phase
#include "chunk/array_chunk_range_ops.h" // for_each_value / for_each_span / read_range (bulk-reads step)
#include "chunk/internal/print.h"        // shrink_array::println

#include "validation_harness.h" // self_command; run() is called by validation_driver.cpp

namespace validation::iterator_validation
{

using namespace shrink_array;

namespace
{

static_assert(chunk_config::use_zero || chunk_config::use_zero_one_value || chunk_config::use_zero_any,
              "iterator_validation must be built with CHUNK_VARIANT_ZERO, _ZERO_ONE_VALUE or _ZERO_ANY");

template <typename T>
using Array =
    std::conditional_t<chunk_config::use_zero_any,
                       zero_any::Array<T>,
                       std::conditional_t<chunk_config::use_zero_one_value, zero_one_value::Array<T>, zero::Array<T>>>;

using Arr = Array<double>;
using It  = Arr::const_iterator;
using Ind = Arr::IndType;

// ── The compile-time half ────────────────────────────────────────────────────
static_assert(std::random_access_iterator<It>, "const_iterator must model std::random_access_iterator");
static_assert(std::is_same_v<std::iterator_traits<It>::iterator_category, std::random_access_iterator_tag>,
              "std::lower_bound / std::distance / std::advance dispatch on the classic category: "
              "bidirectional would leave them O(n) moves");
static_assert(std::ranges::random_access_range<const Arr>, "the array must be a random_access_range");
static_assert(std::random_access_iterator<Arr::const_reverse_iterator>, "reverse iterators follow");
static_assert(!std::indirectly_writable<It, double>, "read-only by design: dereference yields a value");
static_assert(!std::sortable<It>, "no in-place sort, by construction: an element has no address");

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

void begin_phase()
{
    if constexpr (build_config::has_not_ompt)
    {
        shrink_array::begin_parallel_phase();
    }
}

void end_phase()
{
    if constexpr (build_config::has_not_ompt)
    {
        shrink_array::finalize_parallel_phase();
    }
}

template <typename A, typename Val>
void maybe_set_second_value(A& arr, const Val& value)
{
    if constexpr (requires(A probe) { probe.set_second_value(value); })
    {
        arr.set_second_value(value);
    }
}

void compress_in_region(Arr& arr)
{
#pragma omp parallel
    {
        begin_phase();
#pragma omp single
        arr.compress();
    }
    end_phase();
}

// ── The sorted pattern, by whole chunks ──────────────────────────────────────
constexpr double second_value = 1.0e9; ///< the second value (above every data value, whatever S)
constexpr long   c0           = 5;     ///< chunks [0, c0): the first constant (0)
constexpr long   c1           = 12;    ///< chunks [c0, c1): runs of 3 equal values, rising
constexpr long   c2           = 18;    ///< chunks [c1, c2): the second value
constexpr long   n_chunks     = 37;    ///< chunks [c2, n_chunks): strictly rising ...
constexpr long   tail_len     = 13;    ///< ... then a partial chunk of tail_len elements

double sorted_at(long index, long chunk_sz)
{
    const long num_chunk = index / chunk_sz;
    if (num_chunk < c0)
    {
        return 0.0;
    }
    if (num_chunk < c1)
    {
        return 10.0 + static_cast<double>((index - (c0 * chunk_sz)) / 3);
    }
    if (num_chunk < c2)
    {
        return second_value;
    }
    return (2.0 * second_value) + static_cast<double>(index - (c2 * chunk_sz));
}

/// @brief Write @p value(i) at every i in [0, N) from a parallel region, then
///        the between-region pass.
/// @details A serial zero_any has no between-region pass (there is no region):
///          its candidates wait for make_block_compression(), which a writer
///          runs once its list is long enough -- a few dozen chunks never get
///          there, and compress() only audits. The harness runs that pass
///          itself there, as mark_path_validation does.
template <typename Fn>
void fill(Arr& arr, long count, Fn&& value)
{
#pragma omp parallel
    {
        begin_phase();
#pragma omp for schedule(static)
        for (long i = 0; i < count; ++i)
        {
            arr[static_cast<Ind>(i)] = value(i);
        }
    }
    end_phase();
    if constexpr (chunk_config::use_zero_any && !build_config::use_openmp)
    {
        arr.make_block_compression();
    }
}

/// @brief Mismatch reporter: counts every failure, prints the first few.
struct Report
{
    // std::format_string checks every call's format against its arguments at compile time.
    template <class... Args>
    void fail(std::format_string<Args...> fmt, Args&&... args)
    {
        if (bad < 8)
        {
            println("[{}/{}] FAIL {}", variant_name(), step, std::format(fmt, std::forward<Args>(args)...));
        }
        ++bad;
    }
    int done(const char* ok_msg) const
    {
        if (!bad)
        {
            println("[{}/{}] OK: {}", variant_name(), step, ok_msg);
        }
        else
        {
            println("[{}/{}] {} failure(s)", variant_name(), step, bad);
        }
        return bad ? 1 : 0;
    }

    const char* step;
    long        bad = 0;
};

/// @brief Chunks folded to a constant: expected at least the first-constant
///        block, plus the second-value block where the variant has one.
int check_premise(const Arr& arr, long count)
{
    Report     report{"premise"};
    const long chunk_sz = static_cast<long>(arr.chunk_size());
    long       folded   = 0;
    for (long j = 0; j * chunk_sz < count; ++j)
    {
        folded += arr.written(static_cast<Ind>(j * chunk_sz)) ? 0 : 1;
    }
    const long want = (chunk_config::use_zero_one_value || chunk_config::use_zero_any) ? c0 + (c2 - c1) : c0;
    if (folded < want)
    {
        report.fail("{} chunk(s) folded after the pass, expected at least {}: the jumps would not land in "
                    "compressed chunks",
                    folded, want);
    }
    char msg[128];
    std::snprintf(msg, sizeof msg, "%ld of %ld chunks folded (chunk_size=%ld)", folded,
                  (count + chunk_sz - 1) / chunk_sz, chunk_sz);
    return report.done(msg);
}

/// @brief Sequential read, it[i] at every i, the distance.
int check_content(const Arr& arr, const std::vector<double>& ref)
{
    Report     report{"content"};
    const long count = static_cast<long>(ref.size());
    if (!std::equal(arr.begin(), arr.end(), ref.begin(), ref.end()))
    {
        report.fail("the sequential read differs from the reference");
    }
    const It first = arr.begin();
    for (long i = 0; i < count; ++i)
    {
        if (first[i] != ref[static_cast<std::size_t>(i)])
        {
            report.fail("begin()[{}] = {:g}, expected {:g}", i, first[i], ref[static_cast<std::size_t>(i)]);
        }
    }
    if (arr.end() - arr.begin() != count || std::distance(arr.begin(), arr.end()) != count)
    {
        report.fail("end() - begin() = {}, std::distance = {}, expected {}", static_cast<long>(arr.end() - arr.begin()),
                    static_cast<long>(std::distance(arr.begin(), arr.end())), count);
    }
    return report.done("sequential read, it[i] everywhere, end() - begin()");
}

/// @brief Every jump from positions around the boundaries, and a step after it.
int check_jumps(const Arr& arr, const std::vector<double>& ref)
{
    Report     report{"jumps"};
    const long chunk_sz = static_cast<long>(arr.chunk_size());
    const long nref     = static_cast<long>(ref.size());
    const auto at       = [&](long index) { return ref[static_cast<std::size_t>(index)]; };

    const long pos[] = {0,
                        1,
                        chunk_sz - 1,
                        chunk_sz,
                        chunk_sz + 1,
                        (c0 * chunk_sz) - 1,
                        c0 * chunk_sz,
                        (c1 * chunk_sz) - 1,
                        c1 * chunk_sz,
                        (c2 * chunk_sz) - 1,
                        c2 * chunk_sz,
                        nref - tail_len,
                        nref - 1,
                        nref};
    const long dl[]  = {0,
                        1,
                        -1,
                        2,
                        -2,
                        chunk_sz - 1,
                        -(chunk_sz - 1),
                        chunk_sz,
                        -chunk_sz,
                        chunk_sz + 1,
                        -(chunk_sz + 1),
                        (3 * chunk_sz) + 7,
                        -((3 * chunk_sz) + 7),
                        11 * chunk_sz,
                        -(11 * chunk_sz)};

    const It first = arr.begin();
    long     count = 0;
    for (const long position : pos)
    {
        const It it = first + position;
        if (it - first != position)
        {
            report.fail("(begin() + {}) - begin() = {}", position, static_cast<long>(it - first));
        }
        if (position < nref && *it != at(position))
        {
            report.fail("*(begin() + {}) = {:g}, expected {:g}", position, *it, at(position));
        }
        for (const long delta : dl)
        {
            const long target = position + delta;
            if (target < 0 || target > nref)
            {
                continue;
            }
            ++count;
            const It j1 = it + delta;
            const It j2 = delta + it;
            It       j3 = it;
            j3 += delta;
            It j4 = it;
            j4 -= -delta;
            const It j5 = it - (-delta);
            if (j1 - first != target || j2 != j1 || j3 != j1 || j4 != j1 || j5 != j1)
            {
                report.fail("from {} by {}: the five forms of the jump disagree (landed at {})", position, delta,
                            static_cast<long>(j1 - first));
            }
            if (j1 - it != delta)
            {
                report.fail("from {} by {}: (it + d) - it = {}", position, delta, static_cast<long>(j1 - it));
            }
            if ((j1 < it) != (delta < 0) || (j1 > it) != (delta > 0) || (j1 <= it) != (delta <= 0) ||
                (j1 >= it) != (delta >= 0))
            {
                report.fail("from {} by {}: the ordering disagrees with the indices", position, delta);
            }
            if (target < nref && (*j1 != at(target) || it[delta] != at(target)))
            {
                report.fail("from {} by {}: *(it + d) = {:g}, it[d] = {:g}, expected {:g}", position, delta, *j1,
                            it[delta], at(target));
            }
            // The cursor after a seek: a step forward and back from it.
            if (target + 1 < nref)
            {
                It moved = j1;
                ++moved;
                if (*moved != at(target + 1) || moved - first != target + 1)
                {
                    report.fail("from {} by {} then ++: {:g}, expected {:g}", position, delta, *moved, at(target + 1));
                }
            }
            if (target >= 1)
            {
                It moved = j1;
                --moved;
                if (*moved != at(target - 1) || moved - first != target - 1)
                {
                    report.fail("from {} by {} then --: {:g}, expected {:g}", position, delta, *moved, at(target - 1));
                }
            }
        }
    }
    // One before begin: the position --begin() and every backward jump past 0
    // reach (seek(-1): chunk -1, null cursor). Never dereferenced; a ++ from it
    // must land back on begin() and read at(0).
    {
        It m1 = first;
        --m1;
        const It m2 = first - 1;
        It       m3 = first;
        m3 += -1;
        for (const long position : pos)
        {
            if (position < 1 || position > nref)
            {
                continue;
            }
            const It m4 = (first + position) - (position + 1);
            if (m4 != m1 || m4 - first != -1)
            {
                report.fail("(begin() + {}) - {} does not land one before begin", position, position + 1);
            }
        }
        if (m1 - first != -1 || m2 != m1 || m3 != m1 || !(m1 < first) || first - m1 != 1)
        {
            report.fail("one before begin: --begin(), begin() - 1 and begin() += -1 disagree");
        }
        It it = m1;
        ++it;
        if (it != first || *it != at(0))
        {
            report.fail("++ from one before begin reads {:g}, expected {:g}", *it, at(0));
        }
        if (*(m1 + 1) != at(0) || m1[1] != at(0))
        {
            report.fail("one before begin: *(it + 1) / it[1] read {:g} / {:g}, expected {:g}", *(m1 + 1), m1[1], at(0));
        }
    }
    if (*(arr.end() - 1) != ref.back())
    {
        report.fail("*(end() - 1) = {:g}, expected {:g}", *(arr.end() - 1), ref.back());
    }
    It adv = arr.begin();
    std::advance(adv, nref - 1);
    if (*adv != ref.back())
    {
        report.fail("std::advance(begin(), N - 1) reads {:g}, expected {:g}", *adv, ref.back());
    }

    char msg[96];
    std::snprintf(msg, sizeof msg, "%ld jumps read and stepped from, end() - 1, std::advance", count);
    return report.done(msg);
}

/// @brief Values present, between, below and above the range.
std::vector<double> queries(const std::vector<double>& ref)
{
    std::vector<double> queries{
        -1.0, 0.0, 0.5, 10.0, second_value - 0.5, second_value, second_value + 0.5, ref.back(), ref.back() + 1.0};
    const std::size_t step = std::max<std::size_t>(1, ref.size() / 1500);
    for (std::size_t i = 0; i < ref.size(); i += step)
    {
        queries.push_back(ref[i]);
        queries.push_back(ref[i] + 0.5);
    }
    return queries;
}

/// @brief The positions every search returns, on @p arr and on @p ref: 0 when they agree.
int search_mismatch(const Arr& arr, const std::vector<double>& ref, double query, long* got, long* want)
{
    const auto ab = arr.begin();
    const auto ae = arr.end();
    const auto rb = ref.begin();
    const auto re = ref.end();

    const auto er_a  = std::equal_range(ab, ae, query);
    const auto er_r  = std::equal_range(rb, re, query);
    const auto rer_a = std::ranges::equal_range(arr, query);

    const long got_idx[] = {
        static_cast<long>(std::lower_bound(ab, ae, query) - ab),
        static_cast<long>(std::upper_bound(ab, ae, query) - ab),
        static_cast<long>(er_a.first - ab),
        static_cast<long>(er_a.second - ab),
        std::binary_search(ab, ae, query) ? 1L : 0L,
        static_cast<long>(std::ranges::lower_bound(arr, query) - ab),
        static_cast<long>(std::ranges::upper_bound(arr, query) - ab),
        static_cast<long>(rer_a.begin() - ab),
        static_cast<long>(rer_a.end() - ab),
        std::ranges::binary_search(arr, query) ? 1L : 0L,
        static_cast<long>(std::lower_bound(arr.rbegin(), arr.rend(), query, std::greater<>{}) - arr.rbegin())};
    const long want_idx[] = {
        static_cast<long>(std::lower_bound(rb, re, query) - rb),
        static_cast<long>(std::upper_bound(rb, re, query) - rb),
        static_cast<long>(er_r.first - rb),
        static_cast<long>(er_r.second - rb),
        std::binary_search(rb, re, query) ? 1L : 0L,
        static_cast<long>(std::lower_bound(rb, re, query) - rb),
        static_cast<long>(std::upper_bound(rb, re, query) - rb),
        static_cast<long>(er_r.first - rb),
        static_cast<long>(er_r.second - rb),
        std::binary_search(rb, re, query) ? 1L : 0L,
        static_cast<long>(std::lower_bound(ref.rbegin(), ref.rend(), query, std::greater<>{}) - ref.rbegin())};
    static_assert(std::size(got_idx) == std::size(want_idx));
    for (std::size_t k = 0; k < std::size(got_idx); ++k)
    {
        if (got_idx[k] != want_idx[k])
        {
            *got  = got_idx[k];
            *want = want_idx[k];
            return static_cast<int>(k) + 1;
        }
    }
    return 0;
}

constexpr const char* search_names[] = {"",
                                        "std::lower_bound",
                                        "std::upper_bound",
                                        "std::equal_range.first",
                                        "std::equal_range.second",
                                        "std::binary_search",
                                        "std::ranges::lower_bound",
                                        "std::ranges::upper_bound",
                                        "std::ranges::equal_range.begin",
                                        "std::ranges::equal_range.end",
                                        "std::ranges::binary_search",
                                        "std::lower_bound on the reverse range"};

int check_searches(const Arr& arr, const std::vector<double>& ref)
{
    Report                    report{"searches"};
    const std::vector<double> query_set = queries(ref);
    for (const double query : query_set)
    {
        long got  = 0;
        long want = 0;
        if (const int mismatch = search_mismatch(arr, ref, query, &got, &want))
        {
            report.fail("{}({:g}) = {}, the std::vector gives {}", search_names[mismatch], query, got, want);
        }
    }
    char msg[96];
    std::snprintf(msg, sizeof msg, "%zu values x 11 searches agree with the std::vector", query_set.size());
    return report.done(msg);
}

/// @brief The same searches from every thread at once, inside a region.
int check_parallel_searches(const Arr& arr, const std::vector<double>& ref)
{
    Report                    report{"parallel-searches"};
    const std::vector<double> query_set = queries(ref);
    const long                nq        = static_cast<long>(query_set.size());
    long                      bad       = 0;
#pragma omp parallel reduction(+ : bad)
    {
        begin_phase();
#pragma omp for schedule(dynamic, 8)
        for (long k = 0; k < nq; ++k)
        {
            long got  = 0;
            long want = 0;
            if (search_mismatch(arr, ref, query_set[static_cast<std::size_t>(k)], &got, &want))
            {
                ++bad;
            }
        }
    }
    end_phase();
    if (bad)
    {
        report.fail("{} of {} values searched concurrently disagree with the std::vector", bad, nq);
    }
    return report.done("every search agrees, run from every thread at once");
}

int check_reverse(const Arr& arr, const std::vector<double>& ref)
{
    Report     report{"reverse"};
    const long chunk_sz = static_cast<long>(arr.chunk_size());
    const long count    = static_cast<long>(ref.size());
    if (!std::equal(arr.rbegin(), arr.rend(), ref.rbegin(), ref.rend()))
    {
        report.fail("the reverse read differs from the reference");
    }
    if (arr.rend() - arr.rbegin() != count)
    {
        report.fail("rend() - rbegin() = {}, expected {}", static_cast<long>(arr.rend() - arr.rbegin()), count);
    }
    for (const long index : {0L, 1L, tail_len - 1, tail_len, tail_len + 1, tail_len + chunk_sz, count - 1})
    {
        if (index >= 0 && index < count && arr.rbegin()[index] != ref.rbegin()[index])
        {
            report.fail("rbegin()[{}] = {:g}, expected {:g}", index, arr.rbegin()[index], ref.rbegin()[index]);
        }
    }
    return report.done("rbegin()/rend(), rbegin()[k]");
}

/// @brief for_each_span callable: appends the spans, and the runs of a
///        compressed chunk expanded, to @p out; counts the runs.
struct SpanCollector
{
    std::vector<double>& out;
    long*                runs = nullptr;

    void operator()(std::span<const double> span) const
    {
        out.insert(out.end(), span.begin(), span.end());
    }
    void operator()(const double& value, std::size_t count) const
    {
        out.insert(out.end(), count, value);
        if (runs != nullptr)
        {
            ++*runs;
        }
    }
};

/// @brief for_each_value / for_each_span / read_range against the reference:
///        whole range, sub-ranges around every boundary, clamping, a T* and an
///        output-iterator destination, slices read from every thread at once,
///        and the same calls on the std::vector (the generic overloads).
int check_bulk_reads(const Arr& arr, const std::vector<double>& ref)
{
    Report     report{"bulk-reads"};
    const long chunk_sz = static_cast<long>(arr.chunk_size());
    const long nref     = static_cast<long>(ref.size());
    const auto sz       = [](long count) { return static_cast<std::size_t>(count); };

    // Whole range, both containers, through the same call.
    for (int pass = 0; pass < 2; ++pass)
    {
        std::vector<double> got;
        got.reserve(sz(nref));
        const auto push = [&got](double elem) { got.push_back(elem); };
        if (pass == 0)
        {
            shrink_array::for_each_value(arr, push);
        }
        else
        {
            shrink_array::for_each_value(ref, push);
        }
        if (got != ref)
        {
            report.fail("for_each_value({}) over the whole range differs from the reference ({} values)",
                        pass == 0 ? "array" : "vector", got.size());
        }
    }

    // for_each_span, whole range: spans only, then spans and runs. The
    // compressed chunks [0, c0) come as runs where the build reads by block.
    {
        std::vector<double> got;
        shrink_array::for_each_span(arr, [&got](std::span<const double> span)
                                    { got.insert(got.end(), span.begin(), span.end()); });
        if (got != ref)
        {
            report.fail("for_each_span(array) with a span-only callable differs from the reference ({} values)",
                        got.size());
        }
        got.clear();
        long runs = 0;
        shrink_array::for_each_span(arr, SpanCollector{got, &runs});
        if (got != ref)
        {
            report.fail("for_each_span(array) with spans and runs differs from the reference ({} values)", got.size());
        }
        if (!chunk_config::stats && !chunk_config::do_check && runs < c0)
        {
            report.fail("for_each_span(array): {} compressed chunks passed as runs, expected at least {}", runs, c0);
        }
    }

    // Sub-ranges around the boundaries: both destinations, both containers.
    const long pos[] = {0,
                        1,
                        chunk_sz - 1,
                        chunk_sz,
                        chunk_sz + 1,
                        (c0 * chunk_sz) - 1,
                        c0 * chunk_sz,
                        c1 * chunk_sz,
                        (c2 * chunk_sz) - 1,
                        nref - tail_len,
                        nref - 1,
                        nref};
    const long len[] = {0, 1, 2, 63, 64, 65, chunk_sz - 1, chunk_sz, chunk_sz + 1, (3 * chunk_sz) + 7, nref};
    long       count = 0;
    for (const long start : pos)
    {
        for (const long length : len)
        {
            const long first = start;
            const long last  = start + length; // may pass N: clamped
            const long want  = std::max(0L, std::min(last, nref) - first);
            ++count;

            std::vector<double> buf(sz(want) + 1, -1.0);
            double* const       end_a = shrink_array::read_range(arr, sz(first), sz(last), buf.data());
            if (end_a - buf.data() != want || !std::equal(buf.begin(), buf.begin() + want, ref.begin() + first) ||
                buf[sz(want)] != -1.0)
            {
                report.fail("read_range(array, {}, {}, T*): {} written, content {}", first, last,
                            static_cast<long>(end_a - buf.data()),
                            std::equal(buf.begin(), buf.begin() + want, ref.begin() + first) ? "ok" : "WRONG");
            }

            std::vector<double> via_it;
            shrink_array::read_range(arr, sz(first), sz(last), std::back_inserter(via_it));
            std::vector<double> via_vec;
            shrink_array::read_range(ref, sz(first), sz(last), std::back_inserter(via_vec));
            if (via_it != via_vec || static_cast<long>(via_vec.size()) != want)
            {
                report.fail("read_range({}, {}) into a back_inserter: array {} values, vector {}, expected {}", first,
                            last, via_it.size(), via_vec.size(), want);
            }

            double sum_a = 0.0;
            double sum_v = 0.0;
            shrink_array::for_each_value(arr, sz(first), sz(last), [&](double elem) { sum_a += elem; });
            shrink_array::for_each_value(ref, sz(first), sz(last), [&](double elem) { sum_v += elem; });
            if (sum_a != sum_v)
            {
                report.fail("for_each_value({}, {}): array sum {:.17g}, vector sum {:.17g}", first, last, sum_a, sum_v);
            }

            std::vector<double> spans_a;
            std::vector<double> spans_v;
            shrink_array::for_each_span(arr, sz(first), sz(last), SpanCollector{spans_a});
            shrink_array::for_each_span(ref, sz(first), sz(last), SpanCollector{spans_v});
            if (spans_a != spans_v || static_cast<long>(spans_v.size()) != want)
            {
                report.fail("for_each_span({}, {}): array {} values, vector {}, expected {}", first, last,
                            spans_a.size(), spans_v.size(), want);
            }
        }
    }

    // Every thread reads its own slice at once (odd slice length: the slices
    // cut chunks in the middle).
    std::vector<double> par(sz(nref), -1.0);
    const long          slice = (chunk_sz / 2) + 3;
#pragma omp parallel
    {
        begin_phase();
#pragma omp for schedule(dynamic, 1)
        for (long lo = 0; lo < nref; lo += slice)
        {
            shrink_array::read_range(arr, sz(lo), sz(lo + slice), par.data() + lo);
        }
    }
    end_phase();
    if (par != ref)
    {
        report.fail("slices read with read_range from every thread at once differ from the reference");
    }

    char msg[128];
    std::snprintf(msg, sizeof msg, "%ld sub-ranges, both containers, T* and back_inserter, parallel slices", count);
    return report.done(msg);
}

/// @brief What the iterator cannot do in place, done on a copy.
int check_sort_recipe(const std::vector<double>& ref)
{
    Report     report{"sort-recipe"};
    const long count = static_cast<long>(ref.size());
    Arr        desc;
    desc.resize(static_cast<Ind>(count));
    maybe_set_second_value(desc, second_value);
    fill(desc, count, [&](long i) { return ref[static_cast<std::size_t>(count - 1 - i)]; }); // descending
    compress_in_region(desc);

    std::vector<double> tmp(desc.begin(), desc.end());
    std::sort(tmp.begin(), tmp.end());
    desc.assign(tmp.begin(), tmp.end());
    compress_in_region(desc);

    if (static_cast<long>(desc.size()) != count || !std::equal(desc.begin(), desc.end(), ref.begin(), ref.end()))
    {
        report.fail("copy, std::sort, assign() does not give the sorted content back");
    }
    return report.done("std::vector<T> v(a.begin(), a.end()); std::sort; a.assign(v.begin(), v.end())");
}

} // namespace

int run(int /*argc*/, char** /*argv*/)
{
    ParamChunk::init_indexes(true);

    println("=== iterator_validation [variant = {}]  threads={} ===", variant_name(), CHUNK_PMP_MAX_THREADS);

    int fails = 0;
    {
        Arr        arr;
        const long chunk_sz = static_cast<long>(arr.chunk_size());
        const long count    = (n_chunks * chunk_sz) + tail_len;
        arr.resize(static_cast<Ind>(count));
        maybe_set_second_value(arr, second_value);
        // Every chunk materialised first (non-uniform content), then the sorted
        // pattern over it and a pass: the constant blocks are folded chunks.
        fill(arr, count, [](long i) { return (i % 2) ? 2.5 : 3.5; });
        fill(arr, count, [chunk_sz](long i) { return sorted_at(i, chunk_sz); });
        compress_in_region(arr);

        std::vector<double> ref(static_cast<std::size_t>(count));
        for (long i = 0; i < count; ++i)
        {
            ref[static_cast<std::size_t>(i)] = sorted_at(i, chunk_sz);
        }

        const Arr& ca = arr;
        fails += check_premise(ca, count);
        fails += check_content(ca, ref);
        fails += check_jumps(ca, ref);
        fails += check_searches(ca, ref);
        fails += check_parallel_searches(ca, ref);
        fails += check_reverse(ca, ref);
        fails += check_bulk_reads(ca, ref);
        fails += check_sort_recipe(ref);
    }

    release_chunk_pools();
    destroy_block_chunk();

    println("\n[{}] {}", variant_name(), fails ? "=== ITERATOR FAILED ===" : "=== ITERATOR OK ===");
    return fails ? 1 : 0;
}

} // namespace validation::iterator_validation
