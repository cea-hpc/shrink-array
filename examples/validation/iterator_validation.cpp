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
// const_iterator harness. Variant-agnostic: chunk / chunk2 / chunk2a follow the
// compile-time macro (CHUNK_VARIANT_V1 / _V2 / _V2A), like the other harnesses here.
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
#include <string_view>      // std::string_view (Report::done)
#include <type_traits>      // std::conditional_t, std::is_same_v
#include <utility>          // std::forward
#include <vector>           // std::vector
#include <format>           // std::format
#include <utility>          // std::forward

#include "chunk/chunk_traits.h"          // forward decls + default IntType = int
#include "chunk/zero_array.h"           // zero::Array
#include "chunk/zero_one_value_array.h"          // zero_one_value::Array
#include "chunk/zero_any_array.h"         // zero_any::Array
#include "chunk/param_chunk.h"           // ParamChunk::init_indexes
#include "chunk/chunk_std.h"             // release_chunk_pools, destroy_block_chunk
#include "chunk/config.h"                // chunk_config::use_chunk*, build_config::has_not_ompt
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

static_assert(chunk_config::use_chunk || chunk_config::use_chunk2 || chunk_config::use_chunk2a,
              "iterator_validation must be built with CHUNK_VARIANT_V1, _V2 or _V2A");

template <typename T>
using Array =
    std::conditional_t<chunk_config::use_chunk2a,
                       zero_any::Array<T>,
                       std::conditional_t<chunk_config::use_chunk2, zero_one_value::Array<T>, zero::Array<T>>>;

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
void maybe_set_second_value(A& a, const Val& v)
{
    if constexpr (requires(A x) { x.set_second_value(v); })
    {
        a.set_second_value(v);
    }
}

void compress_in_region(Arr& a)
{
#pragma omp parallel
    {
        begin_phase();
#pragma omp single
        a.compress();
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

double sorted_at(long i, long s)
{
    const long c = i / s;
    if (c < c0)
    {
        return 0.0;
    }
    if (c < c1)
    {
        return 10.0 + static_cast<double>((i - (c0 * s)) / 3);
    }
    if (c < c2)
    {
        return second_value;
    }
    return (2.0 * second_value) + static_cast<double>(i - (c2 * s));
}

/// @brief Write @p value(i) at every i in [0, N) from a parallel region, then
///        the between-region pass.
/// @details A serial chunk2a has no between-region pass (there is no region):
///          its candidates wait for make_block_compression(), which a writer
///          runs once its list is long enough -- a few dozen chunks never get
///          there, and compress() only audits. The harness runs that pass
///          itself there, as mark_path_validation does.
template <typename Fn>
void fill(Arr& a, long n, Fn&& value)
{
#pragma omp parallel
    {
        begin_phase();
#pragma omp for schedule(static)
        for (long i = 0; i < n; ++i)
        {
            a[static_cast<Ind>(i)] = value(i);
        }
    }
    end_phase();
    if constexpr (chunk_config::use_chunk2a && !build_config::use_openmp)
    {
        a.make_block_compression();
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
    int done(std::string_view ok_msg) const
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
int check_premise(const Arr& a, long n)
{
    Report     r{"premise"};
    const long s      = static_cast<long>(a.chunk_size());
    long       folded = 0;
    for (long j = 0; j * s < n; ++j)
    {
        folded += a.written(static_cast<Ind>(j * s)) ? 0 : 1;
    }
    const long want = (chunk_config::use_chunk2 || chunk_config::use_chunk2a) ? c0 + (c2 - c1) : c0;
    if (folded < want)
    {
        r.fail("{} chunk(s) folded after the pass, expected at least {}: the jumps would not land in "
               "compressed chunks",
               folded, want);
    }
    return r.done(std::format("{} of {} chunks folded (chunk_size={})", folded, (n + s - 1) / s, s));
}

/// @brief Sequential read, it[i] at every i, the distance.
int check_content(const Arr& a, const std::vector<double>& ref)
{
    Report     r{"content"};
    const long n = static_cast<long>(ref.size());
    if (!std::equal(a.begin(), a.end(), ref.begin(), ref.end()))
    {
        r.fail("the sequential read differs from the reference");
    }
    const It b = a.begin();
    for (long i = 0; i < n; ++i)
    {
        if (b[i] != ref[static_cast<std::size_t>(i)])
        {
            r.fail("begin()[{}] = {:g}, expected {:g}", i, b[i], ref[static_cast<std::size_t>(i)]);
        }
    }
    if (a.end() - a.begin() != n || std::distance(a.begin(), a.end()) != n)
    {
        r.fail("end() - begin() = {}, std::distance = {}, expected {}", static_cast<long>(a.end() - a.begin()),
               static_cast<long>(std::distance(a.begin(), a.end())), n);
    }
    return r.done("sequential read, it[i] everywhere, end() - begin()");
}

/// @brief Every jump from positions around the boundaries, and a step after it.
int check_jumps(const Arr& a, const std::vector<double>& ref)
{
    Report     r{"jumps"};
    const long s  = static_cast<long>(a.chunk_size());
    const long n  = static_cast<long>(ref.size());
    auto       at = [&](long i) { return ref[static_cast<std::size_t>(i)]; };

    const long pos[] = {
        0,     1, s - 1, s, s + 1, (c0 * s) - 1, c0 * s, (c1 * s) - 1, c1 * s, (c2 * s) - 1, c2 * s, n - tail_len,
        n - 1, n};
    const long dl[] = {
        0, 1, -1, 2, -2, s - 1, -(s - 1), s, -s, s + 1, -(s + 1), (3 * s) + 7, -((3 * s) + 7), 11 * s, -(11 * s)};

    const It b     = a.begin();
    long     count = 0;
    for (const long k : pos)
    {
        const It it = b + k;
        if (it - b != k)
        {
            r.fail("(begin() + {}) - begin() = {}", k, static_cast<long>(it - b));
        }
        if (k < n && *it != at(k))
        {
            r.fail("*(begin() + {}) = {:g}, expected {:g}", k, *it, at(k));
        }
        for (const long d : dl)
        {
            const long t = k + d;
            if (t < 0 || t > n)
            {
                continue;
            }
            ++count;
            const It j1 = it + d;
            const It j2 = d + it;
            It       j3 = it;
            j3 += d;
            It j4 = it;
            j4 -= -d;
            const It j5 = it - (-d);
            if (j1 - b != t || j2 != j1 || j3 != j1 || j4 != j1 || j5 != j1)
            {
                r.fail("from {} by {}: the five forms of the jump disagree (landed at {})", k, d,
                       static_cast<long>(j1 - b));
            }
            if (j1 - it != d)
            {
                r.fail("from {} by {}: (it + d) - it = {}", k, d, static_cast<long>(j1 - it));
            }
            if ((j1 < it) != (d < 0) || (j1 > it) != (d > 0) || (j1 <= it) != (d <= 0) || (j1 >= it) != (d >= 0))
            {
                r.fail("from {} by {}: the ordering disagrees with the indices", k, d);
            }
            if (t < n && (*j1 != at(t) || it[d] != at(t)))
            {
                r.fail("from {} by {}: *(it + d) = {:g}, it[d] = {:g}, expected {:g}", k, d, *j1, it[d], at(t));
            }
            // The cursor after a seek: a step forward and back from it.
            if (t + 1 < n)
            {
                It moved = j1;
                ++moved;
                if (*moved != at(t + 1) || moved - b != t + 1)
                {
                    r.fail("from {} by {} then ++: {:g}, expected {:g}", k, d, *moved, at(t + 1));
                }
            }
            if (t >= 1)
            {
                It moved = j1;
                --moved;
                if (*moved != at(t - 1) || moved - b != t - 1)
                {
                    r.fail("from {} by {} then --: {:g}, expected {:g}", k, d, *moved, at(t - 1));
                }
            }
        }
    }
    // One before begin: the position --begin() and every backward jump past 0
    // reach (seek(-1): chunk -1, null cursor). Never dereferenced; a ++ from it
    // must land back on begin() and read at(0).
    {
        It m1 = b;
        --m1;
        const It m2 = b - 1;
        It       m3 = b;
        m3 += -1;
        for (const long k : pos)
        {
            if (k < 1 || k > n)
            {
                continue;
            }
            const It m4 = (b + k) - (k + 1);
            if (m4 != m1 || m4 - b != -1)
            {
                r.fail("(begin() + {}) - {} does not land one before begin", k, k + 1);
            }
        }
        if (m1 - b != -1 || m2 != m1 || m3 != m1 || !(m1 < b) || b - m1 != 1)
        {
            r.fail("one before begin: --begin(), begin() - 1 and begin() += -1 disagree");
        }
        It p = m1;
        ++p;
        if (p != b || *p != at(0))
        {
            r.fail("++ from one before begin reads {:g}, expected {:g}", *p, at(0));
        }
        if (*(m1 + 1) != at(0) || m1[1] != at(0))
        {
            r.fail("one before begin: *(it + 1) / it[1] read {:g} / {:g}, expected {:g}", *(m1 + 1), m1[1], at(0));
        }
    }
    if (*(a.end() - 1) != ref.back())
    {
        r.fail("*(end() - 1) = {:g}, expected {:g}", *(a.end() - 1), ref.back());
    }
    It adv = a.begin();
    std::advance(adv, n - 1);
    if (*adv != ref.back())
    {
        r.fail("std::advance(begin(), N - 1) reads {:g}, expected {:g}", *adv, ref.back());
    }

    return r.done(std::format("{} jumps read and stepped from, end() - 1, std::advance", count));
}

/// @brief Values present, between, below and above the range.
std::vector<double> queries(const std::vector<double>& ref)
{
    std::vector<double> q{
        -1.0, 0.0, 0.5, 10.0, second_value - 0.5, second_value, second_value + 0.5, ref.back(), ref.back() + 1.0};
    const std::size_t step = std::max<std::size_t>(1, ref.size() / 1500);
    for (std::size_t i = 0; i < ref.size(); i += step)
    {
        q.push_back(ref[i]);
        q.push_back(ref[i] + 0.5);
    }
    return q;
}

/// @brief The positions every search returns, on @p a and on @p ref: 0 when they agree.
int search_mismatch(const Arr& a, const std::vector<double>& ref, double q, long* got, long* want)
{
    const auto ab = a.begin();
    const auto ae = a.end();
    const auto rb = ref.begin();
    const auto re = ref.end();

    const auto er_a  = std::equal_range(ab, ae, q);
    const auto er_r  = std::equal_range(rb, re, q);
    const auto rer_a = std::ranges::equal_range(a, q);

    const long g[] = {static_cast<long>(std::lower_bound(ab, ae, q) - ab),
                      static_cast<long>(std::upper_bound(ab, ae, q) - ab),
                      static_cast<long>(er_a.first - ab),
                      static_cast<long>(er_a.second - ab),
                      std::binary_search(ab, ae, q) ? 1L : 0L,
                      static_cast<long>(std::ranges::lower_bound(a, q) - ab),
                      static_cast<long>(std::ranges::upper_bound(a, q) - ab),
                      static_cast<long>(rer_a.begin() - ab),
                      static_cast<long>(rer_a.end() - ab),
                      std::ranges::binary_search(a, q) ? 1L : 0L,
                      static_cast<long>(std::lower_bound(a.rbegin(), a.rend(), q, std::greater<>{}) - a.rbegin())};
    const long w[] = {
        static_cast<long>(std::lower_bound(rb, re, q) - rb),
        static_cast<long>(std::upper_bound(rb, re, q) - rb),
        static_cast<long>(er_r.first - rb),
        static_cast<long>(er_r.second - rb),
        std::binary_search(rb, re, q) ? 1L : 0L,
        static_cast<long>(std::lower_bound(rb, re, q) - rb),
        static_cast<long>(std::upper_bound(rb, re, q) - rb),
        static_cast<long>(er_r.first - rb),
        static_cast<long>(er_r.second - rb),
        std::binary_search(rb, re, q) ? 1L : 0L,
        static_cast<long>(std::lower_bound(ref.rbegin(), ref.rend(), q, std::greater<>{}) - ref.rbegin())};
    static_assert(std::size(g) == std::size(w));
    for (std::size_t k = 0; k < std::size(g); ++k)
    {
        if (g[k] != w[k])
        {
            *got  = g[k];
            *want = w[k];
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

int check_searches(const Arr& a, const std::vector<double>& ref)
{
    Report                    r{"searches"};
    const std::vector<double> q = queries(ref);
    for (const double v : q)
    {
        long got  = 0;
        long want = 0;
        if (const int k = search_mismatch(a, ref, v, &got, &want))
        {
            r.fail("{}({:g}) = {}, the std::vector gives {}", search_names[k], v, got, want);
        }
    }
    return r.done(std::format("{} values x 11 searches agree with the std::vector", q.size()));
}

/// @brief The same searches from every thread at once, inside a region.
int check_parallel_searches(const Arr& a, const std::vector<double>& ref)
{
    Report                    r{"parallel-searches"};
    const std::vector<double> q   = queries(ref);
    const long                nq  = static_cast<long>(q.size());
    long                      bad = 0;
#pragma omp parallel reduction(+ : bad)
    {
        begin_phase();
#pragma omp for schedule(dynamic, 8)
        for (long k = 0; k < nq; ++k)
        {
            long got  = 0;
            long want = 0;
            if (search_mismatch(a, ref, q[static_cast<std::size_t>(k)], &got, &want))
            {
                ++bad;
            }
        }
    }
    end_phase();
    if (bad)
    {
        r.fail("{} of {} values searched concurrently disagree with the std::vector", bad, nq);
    }
    return r.done("every search agrees, run from every thread at once");
}

int check_reverse(const Arr& a, const std::vector<double>& ref)
{
    Report     r{"reverse"};
    const long s = static_cast<long>(a.chunk_size());
    const long n = static_cast<long>(ref.size());
    if (!std::equal(a.rbegin(), a.rend(), ref.rbegin(), ref.rend()))
    {
        r.fail("the reverse read differs from the reference");
    }
    if (a.rend() - a.rbegin() != n)
    {
        r.fail("rend() - rbegin() = {}, expected {}", static_cast<long>(a.rend() - a.rbegin()), n);
    }
    for (const long k : {0L, 1L, tail_len - 1, tail_len, tail_len + 1, tail_len + s, n - 1})
    {
        if (k >= 0 && k < n && a.rbegin()[k] != ref.rbegin()[k])
        {
            r.fail("rbegin()[{}] = {:g}, expected {:g}", k, a.rbegin()[k], ref.rbegin()[k]);
        }
    }
    return r.done("rbegin()/rend(), rbegin()[k]");
}

/// @brief for_each_span callable: appends the spans, and the runs of a
///        compressed chunk expanded, to @p out; counts the runs.
struct SpanCollector
{
    std::vector<double>& out;
    long*                runs = nullptr;

    void operator()(std::span<const double> s) const
    {
        out.insert(out.end(), s.begin(), s.end());
    }
    void operator()(const double& v, std::size_t count) const
    {
        out.insert(out.end(), count, v);
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
int check_bulk_reads(const Arr& a, const std::vector<double>& ref)
{
    Report     r{"bulk-reads"};
    const long s  = static_cast<long>(a.chunk_size());
    const long n  = static_cast<long>(ref.size());
    const auto sz = [](long v) { return static_cast<std::size_t>(v); };

    // Whole range, both containers, through the same call.
    for (int pass = 0; pass < 2; ++pass)
    {
        std::vector<double> got;
        got.reserve(sz(n));
        const auto push = [&got](double x) { got.push_back(x); };
        if (pass == 0)
        {
            shrink_array::for_each_value(a, push);
        }
        else
        {
            shrink_array::for_each_value(ref, push);
        }
        if (got != ref)
        {
            r.fail("for_each_value({}) over the whole range differs from the reference ({} values)",
                   pass == 0 ? "array" : "vector", got.size());
        }
    }

    // for_each_span, whole range: spans only, then spans and runs. The
    // compressed chunks [0, c0) come as runs where the build reads by block.
    {
        std::vector<double> got;
        shrink_array::for_each_span(a, [&got](std::span<const double> span)
                                    { got.insert(got.end(), span.begin(), span.end()); });
        if (got != ref)
        {
            r.fail("for_each_span(array) with a span-only callable differs from the reference ({} values)", got.size());
        }
        got.clear();
        long runs = 0;
        shrink_array::for_each_span(a, SpanCollector{got, &runs});
        if (got != ref)
        {
            r.fail("for_each_span(array) with spans and runs differs from the reference ({} values)", got.size());
        }
        if (!chunk_config::stats && !chunk_config::do_check && runs < c0)
        {
            r.fail("for_each_span(array): {} compressed chunks passed as runs, expected at least {}", runs, c0);
        }
    }

    // Sub-ranges around the boundaries: both destinations, both containers.
    const long pos[] = {0, 1, s - 1, s, s + 1, (c0 * s) - 1, c0 * s, c1 * s, (c2 * s) - 1, n - tail_len, n - 1, n};
    const long len[] = {0, 1, 2, 63, 64, 65, s - 1, s, s + 1, (3 * s) + 7, n};
    long       count = 0;
    for (const long p : pos)
    {
        for (const long l : len)
        {
            const long first = p;
            const long last  = p + l; // may pass N: clamped
            const long want  = std::max(0L, std::min(last, n) - first);
            ++count;

            std::vector<double> buf(sz(want) + 1, -1.0);
            double* const       end_a = shrink_array::read_range(a, sz(first), sz(last), buf.data());
            if (end_a - buf.data() != want || !std::equal(buf.begin(), buf.begin() + want, ref.begin() + first) ||
                buf[sz(want)] != -1.0)
            {
                r.fail("read_range(array, {}, {}, T*): {} written, content {}", first, last,
                       static_cast<long>(end_a - buf.data()),
                       std::equal(buf.begin(), buf.begin() + want, ref.begin() + first) ? "ok" : "WRONG");
            }

            std::vector<double> via_it;
            shrink_array::read_range(a, sz(first), sz(last), std::back_inserter(via_it));
            std::vector<double> via_vec;
            shrink_array::read_range(ref, sz(first), sz(last), std::back_inserter(via_vec));
            if (via_it != via_vec || static_cast<long>(via_vec.size()) != want)
            {
                r.fail("read_range({}, {}) into a back_inserter: array {} values, vector {}, expected {}", first, last,
                       via_it.size(), via_vec.size(), want);
            }

            double sum_a = 0.0;
            double sum_v = 0.0;
            shrink_array::for_each_value(a, sz(first), sz(last), [&](double x) { sum_a += x; });
            shrink_array::for_each_value(ref, sz(first), sz(last), [&](double x) { sum_v += x; });
            if (sum_a != sum_v)
            {
                r.fail("for_each_value({}, {}): array sum {:.17g}, vector sum {:.17g}", first, last, sum_a, sum_v);
            }

            std::vector<double> spans_a;
            std::vector<double> spans_v;
            shrink_array::for_each_span(a, sz(first), sz(last), SpanCollector{spans_a});
            shrink_array::for_each_span(ref, sz(first), sz(last), SpanCollector{spans_v});
            if (spans_a != spans_v || static_cast<long>(spans_v.size()) != want)
            {
                r.fail("for_each_span({}, {}): array {} values, vector {}, expected {}", first, last, spans_a.size(),
                       spans_v.size(), want);
            }
        }
    }

    // Every thread reads its own slice at once (odd slice length: the slices
    // cut chunks in the middle).
    std::vector<double> par(sz(n), -1.0);
    const long          slice = (s / 2) + 3;
#pragma omp parallel
    {
        begin_phase();
#pragma omp for schedule(dynamic, 1)
        for (long lo = 0; lo < n; lo += slice)
        {
            shrink_array::read_range(a, sz(lo), sz(lo + slice), par.data() + lo);
        }
    }
    end_phase();
    if (par != ref)
    {
        r.fail("slices read with read_range from every thread at once differ from the reference");
    }

    return r.done(std::format("{} sub-ranges, both containers, T* and back_inserter, parallel slices", count));
}

/// @brief What the iterator cannot do in place, done on a copy.
int check_sort_recipe(const std::vector<double>& ref)
{
    Report     r{"sort-recipe"};
    const long n = static_cast<long>(ref.size());
    Arr        b;
    b.resize(static_cast<Ind>(n));
    maybe_set_second_value(b, second_value);
    fill(b, n, [&](long i) { return ref[static_cast<std::size_t>(n - 1 - i)]; }); // descending
    compress_in_region(b);

    std::vector<double> tmp(b.begin(), b.end());
    std::sort(tmp.begin(), tmp.end());
    b.assign(tmp.begin(), tmp.end());
    compress_in_region(b);

    if (static_cast<long>(b.size()) != n || !std::equal(b.begin(), b.end(), ref.begin(), ref.end()))
    {
        r.fail("copy, std::sort, assign() does not give the sorted content back");
    }
    return r.done("std::vector<T> v(a.begin(), a.end()); std::sort; a.assign(v.begin(), v.end())");
}

} // namespace

int run(int /*argc*/, char** /*argv*/)
{
    ParamChunk::init_indexes(true);

    println("=== iterator_validation [variant = {}]  threads={} ===", variant_name(), CHUNK_PMP_MAX_THREADS);

    int fails = 0;
    {
        Arr        a;
        const long s = static_cast<long>(a.chunk_size());
        const long n = (n_chunks * s) + tail_len;
        a.resize(static_cast<Ind>(n));
        maybe_set_second_value(a, second_value);
        // Every chunk materialised first (non-uniform content), then the sorted
        // pattern over it and a pass: the constant blocks are folded chunks.
        fill(a, n, [](long i) { return (i % 2) ? 2.5 : 3.5; });
        fill(a, n, [s](long i) { return sorted_at(i, s); });
        compress_in_region(a);

        std::vector<double> ref(static_cast<std::size_t>(n));
        for (long i = 0; i < n; ++i)
        {
            ref[static_cast<std::size_t>(i)] = sorted_at(i, s);
        }

        const Arr& ca = a;
        fails += check_premise(ca, n);
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
