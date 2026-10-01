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
// compile-time macro (CHUNK / CHUNK2 / CHUNK2A), like the other harnesses here.
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

#include <algorithm>
#include <cstdarg> // va_list (Report::fail)
#include <cstdio>
#include <initializer_list>
#include <functional> // std::greater
#include <iterator>
#include <ranges>
#include <type_traits>
#include <utility>
#include <vector>
#include <unistd.h>

#include "chunk/chunk_traits.h"
#include "chunk/array_chunk.h"
#include "chunk/array_chunk2.h"
#include "chunk/array_chunk2A.h"
#include "chunk/param_chunk.h"
#include "chunk/chunk_std.h" // release_chunk_pools, destroy_block_chunk
#include "chunk/config.h"
#include "chunk/PMP_CHUNK.h"
#include "chunk/manage_ompt.h"

int __mem_page_size = 0;

using namespace shrink_array;

namespace
{

static_assert(chunk_config::use_chunk || chunk_config::use_chunk2 || chunk_config::use_chunk2a,
              "iterator_validation must be built with CHUNK, CHUNK2 or CHUNK2A");

template <typename T>
using Array = std::conditional_t<chunk_config::use_chunk2a,
                                 ArrayChunk2A<T>,
                                 std::conditional_t<chunk_config::use_chunk2, ArrayChunk2<T>, ArrayChunk<T>>>;

using Arr = Array<double>;
using It  = Arr::const_iterator;
using Ind = Arr::ind_type;

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
        return "chunk2a";
    else if constexpr (chunk_config::use_chunk2)
        return "chunk2";
    else
        return "chunk";
}

void begin_phase()
{
    if constexpr (build_config::has_not_ompt)
        shrink_array::begin_parallel_phase();
}

void end_phase()
{
    if constexpr (build_config::has_not_ompt)
        shrink_array::finalize_parallel_phase();
}

template <typename A, typename Val>
void maybe_set_second_value(A& a, const Val& v)
{
    if constexpr (requires(A x) { x.set_second_value(v); })
        a.set_second_value(v);
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
constexpr double kSecond = 1.0e9; ///< the second value (above every data value, whatever S)
constexpr long   kC0     = 5;     ///< chunks [0, kC0): the first constant (0)
constexpr long   kC1     = 12;    ///< chunks [kC0, kC1): runs of 3 equal values, rising
constexpr long   kC2     = 18;    ///< chunks [kC1, kC2): the second value
constexpr long   kChunks = 37;    ///< chunks [kC2, kChunks): strictly rising ...
constexpr long   kTail   = 13;    ///< ... then a partial chunk of kTail elements

double sorted_at(long i, long S)
{
    const long c = i / S;
    if (c < kC0)
        return 0.0;
    if (c < kC1)
        return 10.0 + static_cast<double>((i - (kC0 * S)) / 3);
    if (c < kC2)
        return kSecond;
    return (2.0 * kSecond) + static_cast<double>(i - (kC2 * S));
}

/// @brief Write @p value(i) at every i in [0, N) from a parallel region, then
///        the between-region pass.
/// @details A serial chunk2a has no between-region pass (there is no region):
///          its candidates wait for make_block_compression(), which a writer
///          runs once its list is long enough -- a few dozen chunks never get
///          there, and compress() only audits. The harness runs that pass
///          itself there, as mark_path_validation does.
template <typename Fn>
void fill(Arr& a, long N, Fn&& value)
{
#pragma omp parallel
    {
        begin_phase();
#pragma omp for schedule(static)
        for (long i = 0; i < N; ++i)
            a[static_cast<Ind>(i)] = value(i);
    }
    end_phase();
    if constexpr (chunk_config::use_chunk2a && !build_config::use_openmp)
        a.make_block_compression();
}

/// @brief Mismatch reporter: counts every failure, prints the first few.
struct Report
{
    // C varargs on purpose: the format attribute keeps -Wformat checking
    // every call (a forwarded, non-literal format would not be checked).
    [[gnu::format(printf, 2, 3)]] void fail(const char* fmt, ...) // NOLINT(cert-dcl50-cpp)
    {
        if (bad < 8)
        {
            std::printf("[%s/%s] FAIL ", variant_name(), step);
            va_list ap;
            va_start(ap, fmt);
            std::vprintf(fmt, ap);
            va_end(ap);
            std::printf("\n");
        }
        ++bad;
    }
    int done(const char* ok_msg) const
    {
        if (!bad)
            std::printf("[%s/%s] OK: %s\n", variant_name(), step, ok_msg);
        else
            std::printf("[%s/%s] %ld failure(s)\n", variant_name(), step, bad);
        return bad ? 1 : 0;
    }

    const char* step;
    long        bad = 0;
};

/// @brief Chunks folded to a constant: expected at least the first-constant
///        block, plus the second-value block where the variant has one.
int check_premise(const Arr& a, long N)
{
    Report     r{"premise"};
    const long S      = static_cast<long>(a.chunk_size());
    long       folded = 0;
    for (long j = 0; j * S < N; ++j)
        folded += a.written(static_cast<Ind>(j * S)) ? 0 : 1;
    const long want = (chunk_config::use_chunk2 || chunk_config::use_chunk2a) ? kC0 + (kC2 - kC1) : kC0;
    if (folded < want)
        r.fail("%ld chunk(s) folded after the pass, expected at least %ld: the jumps would not land in "
               "compressed chunks",
               folded, want);
    char msg[128];
    std::snprintf(msg, sizeof msg, "%ld of %ld chunks folded (chunk_size=%ld)", folded, (N + S - 1) / S, S);
    return r.done(msg);
}

/// @brief Sequential read, it[i] at every i, the distance.
int check_content(const Arr& a, const std::vector<double>& ref)
{
    Report     r{"content"};
    const long N = static_cast<long>(ref.size());
    if (!std::equal(a.begin(), a.end(), ref.begin(), ref.end()))
        r.fail("the sequential read differs from the reference");
    const It b = a.begin();
    for (long i = 0; i < N; ++i)
        if (b[i] != ref[static_cast<std::size_t>(i)])
            r.fail("begin()[%ld] = %g, expected %g", i, b[i], ref[static_cast<std::size_t>(i)]);
    if (a.end() - a.begin() != N || std::distance(a.begin(), a.end()) != N)
        r.fail("end() - begin() = %ld, std::distance = %ld, expected %ld", static_cast<long>(a.end() - a.begin()),
               static_cast<long>(std::distance(a.begin(), a.end())), N);
    return r.done("sequential read, it[i] everywhere, end() - begin()");
}

/// @brief Every jump from positions around the boundaries, and a step after it.
int check_jumps(const Arr& a, const std::vector<double>& ref)
{
    Report     r{"jumps"};
    const long S = static_cast<long>(a.chunk_size());
    const long N = static_cast<long>(ref.size());
    auto       at = [&](long i) { return ref[static_cast<std::size_t>(i)]; };

    const long pos[] = {0,           1,       S - 1,       S,       S + 1,  (kC0 * S) - 1, kC0 * S,
                        (kC1 * S) - 1, kC1 * S, (kC2 * S) - 1, kC2 * S, N - kTail, N - 1,         N};
    const long dl[]  = {0,      1,          -1,          2, -2, S - 1, -(S - 1), S, -S, S + 1, -(S + 1),
                        (3 * S) + 7, -((3 * S) + 7), 11 * S, -(11 * S)};

    const It b = a.begin();
    long     n = 0;
    for (const long k : pos)
    {
        const It it = b + k;
        if (it - b != k)
            r.fail("(begin() + %ld) - begin() = %ld", k, static_cast<long>(it - b));
        if (k < N && *it != at(k))
            r.fail("*(begin() + %ld) = %g, expected %g", k, *it, at(k));
        for (const long d : dl)
        {
            const long t = k + d;
            if (t < 0 || t > N)
                continue;
            ++n;
            const It j1 = it + d;
            const It j2 = d + it;
            It       j3 = it;
            j3 += d;
            It j4 = it;
            j4 -= -d;
            const It j5 = it - (-d);
            if (j1 - b != t || j2 != j1 || j3 != j1 || j4 != j1 || j5 != j1)
                r.fail("from %ld by %ld: the five forms of the jump disagree (landed at %ld)", k, d,
                       static_cast<long>(j1 - b));
            if (j1 - it != d)
                r.fail("from %ld by %ld: (it + d) - it = %ld", k, d, static_cast<long>(j1 - it));
            if ((j1 < it) != (d < 0) || (j1 > it) != (d > 0) || (j1 <= it) != (d <= 0) || (j1 >= it) != (d >= 0))
                r.fail("from %ld by %ld: the ordering disagrees with the indices", k, d);
            if (t < N && (*j1 != at(t) || it[d] != at(t)))
                r.fail("from %ld by %ld: *(it + d) = %g, it[d] = %g, expected %g", k, d, *j1, it[d], at(t));
            // The cursor after a seek: a step forward and back from it.
            if (t + 1 < N)
            {
                It s = j1;
                ++s;
                if (*s != at(t + 1) || s - b != t + 1)
                    r.fail("from %ld by %ld then ++: %g, expected %g", k, d, *s, at(t + 1));
            }
            if (t >= 1)
            {
                It s = j1;
                --s;
                if (*s != at(t - 1) || s - b != t - 1)
                    r.fail("from %ld by %ld then --: %g, expected %g", k, d, *s, at(t - 1));
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
            if (k < 1 || k > N)
                continue;
            const It m4 = (b + k) - (k + 1);
            if (m4 != m1 || m4 - b != -1)
                r.fail("(begin() + %ld) - %ld does not land one before begin", k, k + 1);
        }
        if (m1 - b != -1 || m2 != m1 || m3 != m1 || !(m1 < b) || b - m1 != 1)
            r.fail("one before begin: --begin(), begin() - 1 and begin() += -1 disagree");
        It p = m1;
        ++p;
        if (p != b || *p != at(0))
            r.fail("++ from one before begin reads %g, expected %g", *p, at(0));
        if (*(m1 + 1) != at(0) || m1[1] != at(0))
            r.fail("one before begin: *(it + 1) / it[1] read %g / %g, expected %g", *(m1 + 1), m1[1], at(0));
    }
    if (*(a.end() - 1) != ref.back())
        r.fail("*(end() - 1) = %g, expected %g", *(a.end() - 1), ref.back());
    It adv = a.begin();
    std::advance(adv, N - 1);
    if (*adv != ref.back())
        r.fail("std::advance(begin(), N - 1) reads %g, expected %g", *adv, ref.back());

    char msg[96];
    std::snprintf(msg, sizeof msg, "%ld jumps read and stepped from, end() - 1, std::advance", n);
    return r.done(msg);
}

/// @brief Values present, between, below and above the range.
std::vector<double> queries(const std::vector<double>& ref)
{
    std::vector<double> q{-1.0, 0.0, 0.5, 10.0, kSecond - 0.5, kSecond, kSecond + 0.5, ref.back(), ref.back() + 1.0};
    const std::size_t   step = std::max<std::size_t>(1, ref.size() / 1500);
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
    const long w[] = {static_cast<long>(std::lower_bound(rb, re, q) - rb),
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
        if (g[k] != w[k])
        {
            *got  = g[k];
            *want = w[k];
            return static_cast<int>(k) + 1;
        }
    return 0;
}

constexpr const char* kSearchName[] = {"",
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
            r.fail("%s(%g) = %ld, the std::vector gives %ld", kSearchName[k], v, got, want);
    }
    char msg[96];
    std::snprintf(msg, sizeof msg, "%zu values x 11 searches agree with the std::vector", q.size());
    return r.done(msg);
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
                ++bad;
        }
    }
    end_phase();
    if (bad)
        r.fail("%ld of %ld values searched concurrently disagree with the std::vector", bad, nq);
    return r.done("every search agrees, run from every thread at once");
}

int check_reverse(const Arr& a, const std::vector<double>& ref)
{
    Report     r{"reverse"};
    const long S = static_cast<long>(a.chunk_size());
    const long N = static_cast<long>(ref.size());
    if (!std::equal(a.rbegin(), a.rend(), ref.rbegin(), ref.rend()))
        r.fail("the reverse read differs from the reference");
    if (a.rend() - a.rbegin() != N)
        r.fail("rend() - rbegin() = %ld, expected %ld", static_cast<long>(a.rend() - a.rbegin()), N);
    for (const long k : {0L, 1L, kTail - 1, kTail, kTail + 1, kTail + S, N - 1})
        if (k >= 0 && k < N && a.rbegin()[k] != ref.rbegin()[k])
            r.fail("rbegin()[%ld] = %g, expected %g", k, a.rbegin()[k], ref.rbegin()[k]);
    return r.done("rbegin()/rend(), rbegin()[k]");
}

/// @brief What the iterator cannot do in place, done on a copy.
int check_sort_recipe(const std::vector<double>& ref)
{
    Report     r{"sort-recipe"};
    const long N = static_cast<long>(ref.size());
    Arr        b;
    b.resize(static_cast<Ind>(N));
    maybe_set_second_value(b, kSecond);
    fill(b, N, [&](long i) { return ref[static_cast<std::size_t>(N - 1 - i)]; }); // descending
    compress_in_region(b);

    std::vector<double> tmp(b.begin(), b.end());
    std::sort(tmp.begin(), tmp.end());
    b.assign(tmp.begin(), tmp.end());
    compress_in_region(b);

    if (static_cast<long>(b.size()) != N || !std::equal(b.begin(), b.end(), ref.begin(), ref.end()))
        r.fail("copy, std::sort, assign() does not give the sorted content back");
    return r.done("std::vector<T> v(a.begin(), a.end()); std::sort; a.assign(v.begin(), v.end())");
}

} // namespace

int main()
{
    __mem_page_size = getpagesize();
    ParamChunk::init_indexes(true);

    std::printf("=== iterator_validation [variant = %s]  threads=%d ===\n", variant_name(), CHUNK_PMP_MAX_THREADS);

    int fails = 0;
    {
        Arr        a;
        const long S = static_cast<long>(a.chunk_size());
        const long N = (kChunks * S) + kTail;
        a.resize(static_cast<Ind>(N));
        maybe_set_second_value(a, kSecond);
        // Every chunk materialised first (non-uniform content), then the sorted
        // pattern over it and a pass: the constant blocks are folded chunks.
        fill(a, N, [](long i) { return (i % 2) ? 2.5 : 3.5; });
        fill(a, N, [S](long i) { return sorted_at(i, S); });
        compress_in_region(a);

        std::vector<double> ref(static_cast<std::size_t>(N));
        for (long i = 0; i < N; ++i)
            ref[static_cast<std::size_t>(i)] = sorted_at(i, S);

        const Arr& ca = a;
        fails += check_premise(ca, N);
        fails += check_content(ca, ref);
        fails += check_jumps(ca, ref);
        fails += check_searches(ca, ref);
        fails += check_parallel_searches(ca, ref);
        fails += check_reverse(ca, ref);
        fails += check_sort_recipe(ref);
    }

    release_chunk_pools();
    destroy_block_chunk();

    std::printf("\n[%s] %s\n", variant_name(), fails ? "=== ITERATOR FAILED ===" : "=== ITERATOR OK ===");
    return fails ? 1 : 0;
}
