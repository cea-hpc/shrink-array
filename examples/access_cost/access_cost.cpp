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

/**
 * @file access_cost.cpp
 * @brief The cost of an element access, against a std::vector, in the SAME
 *        binary: the time per element of six access patterns on a
 *        std::vector<double> and on the linked variant's Array<double>.
 *
 * @details One binary per variant (access_cost_<variant>_<size>); each times
 *          the std::vector and the array in turn, alternating which goes
 *          first from one repetition to the next, so both see the same
 *          machine state (performance_measurement.md: compare times within
 *          one binary and one time window, alternated). The cases, each its
 *          own parallel region with a static `omp for` over the N elements:
 *
 *            read-varied     sum of every element, every chunk materialised
 *                            with varied values (nothing to compress)
 *            read-constant   sum of every element of an array all at 0 (every
 *                            chunk compressed; the vector: all zeros)
 *            read-random     sum of N elements at random indices (a fixed
 *                            permutation-like sequence) of the varied array
 *            write-varied    every element overwritten with another varied
 *                            value (chunks stay materialised)
 *            write-fresh     varied values written into an array all at 0:
 *                            every chunk materialised on its first write
 *                            (buffer taken from the allocator); the vector:
 *                            a zeroed vector written over
 *            rmw             a[i] += 0.5 on the varied array
 *
 *          A write case's time includes the end of its phase (the deferred
 *          writes flushed, the between-region compression pass): what a
 *          program pays for the region. The array's end-of-run statistics
 *          are not timed. Printed per case: the median and minimum time per
 *          element (wall clock / N, in ns) of the vector and of the array,
 *          and the ratio of the medians. --csv prints one line per case and
 *          repetition instead (python/access_cost.py aggregates them).
 *
 *          Every value is a multiple of 0.5 well under 2^40, so the sums are
 *          exact in double whatever the reduction order: each case's sum must
 *          be the same on the vector and on the array, and is checked (exit 1
 *          otherwise). The elements are checked against the vector after the
 *          writes when --verify is given.
 *
 *          Memory: the vector (8 N bytes), the array (about the same at its
 *          largest), the random indices (4 N bytes). The default N = 2^27
 *          takes about 3 GB.
 *
 *          Usage: access_cost [--n N] [--reps R] [--csv] [--verify]
 */

#include <algorithm>   // std::sort, std::min_element
#include <chrono>      // std::chrono::steady_clock
#include <cstdint>     // std::uint64_t
#include <cstdio>      // std::fprintf, stderr
#include <print>       // std::println
#include <string>      // std::string
#include <type_traits> // std::conditional_t
#include <utility>     // std::as_const
#include <vector>      // std::vector

#include "chunk/chunk_traits.h"       // the ind_type = int default of the arrays
#include "chunk/array_chunk.h"        // ArrayChunk
#include "chunk/array_chunk2.h"       // ArrayChunk2
#include "chunk/array_chunk2A.h"      // ArrayChunk2A
#include "chunk/param_chunk.h"        // ParamChunk::init_indexes
#include "chunk/config.h"             // chunk_config::use_chunk*, build_config::use_openmp / has_not_ompt
#include "chunk/PMP_CHUNK.h"          // CHUNK_PMP_MAX_THREADS
#include "chunk/manage_ompt.h"        // begin_parallel_phase, finalize_parallel_phase
#include "chunk/internal/env_parse.h" // arg_number (checked, not atoi)

using namespace shrink_array;

static_assert(chunk_config::use_chunk || chunk_config::use_chunk2 || chunk_config::use_chunk2a,
              "access_cost: link chunk_v1_*, chunk_v2_* or chunk_v2a_*");

template <typename T>
using Array = std::conditional_t<chunk_config::use_chunk2a,
                                 ArrayChunk2A<T>,
                                 std::conditional_t<chunk_config::use_chunk2, ArrayChunk2<T>, ArrayChunk<T>>>;

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

namespace
{
// ── Phase brackets: no-ops under OMPT (the callbacks do it) ─────────────────
void begin_phase()
{
    if constexpr (build_config::has_not_ompt)
    {
        shrink_array::begin_parallel_phase();
    }
}

void end_phase(Array<double>& arr)
{
    if constexpr (build_config::has_not_ompt)
    {
        shrink_array::finalize_parallel_phase();
    }
    // Serial chunk2a has no between-region pass: run it (as variant_compare).
    if constexpr (chunk_config::use_chunk2a && !build_config::use_openmp)
    {
        arr.make_block_compression();
    }
    else
    {
        (void)arr;
    }
}

/// Varied within any chunk of 64 elements or more (period 97): nothing to compress.
double varied(long i)
{
    return static_cast<double>((i % 97) + 1) * 0.5;
}

/// Another varied value, for the overwrite (period 89, never 0).
double varied2(long i)
{
    return static_cast<double>((i % 89) + 1) * 0.5;
}

/// A fixed pseudo-random index sequence (splitmix64), every index in [0, n).
std::vector<int> random_indices(long n)
{
    std::vector<int> idx(static_cast<std::size_t>(n));
    const int        ni = static_cast<int>(n);
#pragma omp parallel for schedule(static)
    for (int k = 0; k < ni; ++k)
    {
        std::uint64_t x = (static_cast<std::uint64_t>(k) + 1) * 0x9E3779B97F4A7C15ULL;
        x ^= x >> 30;
        x *= 0xBF58476D1CE4E5B9ULL;
        x ^= x >> 27;
        x *= 0x94D049BB133111EBULL;
        x ^= x >> 31;
        idx[static_cast<std::size_t>(k)] = static_cast<int>(x % static_cast<std::uint64_t>(n));
    }
    return idx;
}

using clk = std::chrono::steady_clock;

double secs_since(clk::time_point t0)
{
    return std::chrono::duration<double>(clk::now() - t0).count();
}

struct Sample
{
    double seconds;
    double sum;
};

// ── The cases, on a std::vector ──────────────────────────────────────────────
Sample vec_read(const std::vector<double>& v)
{
    const auto t0  = clk::now();
    const int  n   = static_cast<int>(v.size());
    double     sum = 0.;
#pragma omp parallel for schedule(static) reduction(+ : sum)
    for (int i = 0; i < n; ++i)
    {
        sum += v[static_cast<std::size_t>(i)];
    }
    return {secs_since(t0), sum};
}

Sample vec_read_random(const std::vector<double>& v, const std::vector<int>& idx)
{
    const auto t0  = clk::now();
    const int  n   = static_cast<int>(idx.size());
    double     sum = 0.;
#pragma omp parallel for schedule(static) reduction(+ : sum)
    for (int k = 0; k < n; ++k)
    {
        sum += v[static_cast<std::size_t>(idx[static_cast<std::size_t>(k)])];
    }
    return {secs_since(t0), sum};
}

template <typename F>
Sample vec_write(std::vector<double>& v, F value)
{
    const auto t0 = clk::now();
    const int  n  = static_cast<int>(v.size());
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i)
    {
        v[static_cast<std::size_t>(i)] = value(i);
    }
    return {secs_since(t0), 0.};
}

Sample vec_rmw(std::vector<double>& v)
{
    const auto t0 = clk::now();
    const int  n  = static_cast<int>(v.size());
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i)
    {
        v[static_cast<std::size_t>(i)] += 0.5;
    }
    return {secs_since(t0), 0.};
}

// ── The same cases, on the array ─────────────────────────────────────────────
Sample arr_read(Array<double>& a, int n)
{
    const auto  t0  = clk::now();
    double      sum = 0.;
    const auto& ca  = std::as_const(a);
#pragma omp parallel
    {
        begin_phase();
#pragma omp for schedule(static) reduction(+ : sum)
        for (int i = 0; i < n; ++i)
        {
            sum += ca[i];
        }
    }
    end_phase(a);
    return {secs_since(t0), sum};
}

Sample arr_read_random(Array<double>& a, const std::vector<int>& idx)
{
    const auto  t0  = clk::now();
    const int   n   = static_cast<int>(idx.size());
    double      sum = 0.;
    const auto& ca  = std::as_const(a);
#pragma omp parallel
    {
        begin_phase();
#pragma omp for schedule(static) reduction(+ : sum)
        for (int k = 0; k < n; ++k)
        {
            sum += ca[idx[static_cast<std::size_t>(k)]];
        }
    }
    end_phase(a);
    return {secs_since(t0), sum};
}

template <typename F>
Sample arr_write(Array<double>& a, int n, F value)
{
    const auto t0 = clk::now();
#pragma omp parallel
    {
        begin_phase();
#pragma omp for schedule(static)
        for (int i = 0; i < n; ++i)
        {
            a[i] = value(i);
        }
    }
    end_phase(a);
    return {secs_since(t0), 0.};
}

Sample arr_rmw(Array<double>& a, int n)
{
    const auto t0 = clk::now();
#pragma omp parallel
    {
        begin_phase();
#pragma omp for schedule(static)
        for (int i = 0; i < n; ++i)
        {
            a[i] += 0.5;
        }
    }
    end_phase(a);
    return {secs_since(t0), 0.};
}

/// The sum of the elements, read serially (the --verify / sum check of the writes).
double arr_sum(const Array<double>& a, int n)
{
    double sum = 0.;
    for (int i = 0; i < n; ++i)
    {
        sum += a[i];
    }
    return sum;
}

double vec_sum(const std::vector<double>& v)
{
    double sum = 0.;
    for (const double x : v)
    {
        sum += x;
    }
    return sum;
}

long verify_elements(const char* what, const Array<double>& a, const std::vector<double>& v)
{
    long bad = 0;
    for (std::size_t i = 0; i < v.size(); ++i)
    {
        if (a[static_cast<int>(i)] != v[i])
        {
            if (bad < 3)
            {
                std::fprintf(stderr, "access_cost: %s: element %zu is %g, the vector has %g\n", what, i,
                             a[static_cast<int>(i)], v[i]);
            }
            ++bad;
        }
    }
    return bad;
}

double median(std::vector<double> x)
{
    std::sort(x.begin(), x.end());
    const std::size_t m = x.size() / 2;
    return x.size() % 2 ? x[m] : 0.5 * (x[m - 1] + x[m]);
}

double minimum(const std::vector<double>& x)
{
    return *std::min_element(x.begin(), x.end());
}

constexpr const char* kCases[] = {"read-varied", "read-constant", "read-random", "write-varied", "write-fresh", "rmw"};
constexpr int         kNCases  = static_cast<int>(sizeof(kCases) / sizeof(kCases[0]));
} // namespace

int main(int argc, char** argv)
{
    ParamChunk::init_indexes(true);

    long N      = 1L << 27;
    int  reps   = 5;
    bool csv    = false;
    bool verify = false;
    for (int a = 1; a < argc; ++a)
    {
        const std::string o = argv[a];
        if (o == "--n" && a + 1 < argc)
        {
            N = shrink_array::arg_number<long>(argv[++a], N, "--n");
        }
        else if (o == "--reps" && a + 1 < argc)
        {
            reps = shrink_array::arg_number<int>(argv[++a], reps, "--reps");
        }
        else if (o == "--csv")
        {
            csv = true;
        }
        else if (o == "--verify")
        {
            verify = true;
        }
        else
        {
            std::fprintf(stderr, "usage: %s [--n N] [--reps R] [--csv] [--verify]\n", argv[0]);
            return 2;
        }
    }
    if (reps < 1 || N < 1 || N > 0x7fffffffL)
    {
        std::fprintf(stderr, "access_cost: need reps >= 1, 1 <= n < 2^31\n");
        return 2;
    }
    const int n = static_cast<int>(N);

    // Read off an EMPTY array: no chunk, no end-of-run stats line of its own.
    const long S = static_cast<long>(Array<double>(0).chunk_size());
    if (csv)
    {
        std::println("variant,chunk_size,threads,n,rep,case,container,seconds,ns_per_elem");
    }
    else
    {
        std::println("=== access_cost [{}]  n={}  chunk_size={}  threads={}  reps={} ===", variant_name(), N, S,
                     CHUNK_PMP_MAX_THREADS, reps);
    }

    const std::vector<int> idx = random_indices(N);
    std::vector<double>    v(static_cast<std::size_t>(N));
    std::vector<double>    zeros(static_cast<std::size_t>(N), 0.);

    // [case][container 0 = vector, 1 = array] -> seconds per repetition
    std::vector<double> t[kNCases][2];
    long                sum_mismatch = 0;
    long                bad_elements = 0;

    const auto check_sum = [&](int c, double vs, double as)
    {
        if (vs != as)
        {
            std::fprintf(stderr, "access_cost: %s: the array sums to %.17g, the vector to %.17g\n", kCases[c], as, vs);
            ++sum_mismatch;
        }
    };
    const auto note = [&](int c, int container, int rep, double s)
    {
        t[c][container].push_back(s);
        if (csv)
        {
            std::println("{},{},{},{},{},{},{},{:.9f},{:.4f}", variant_name(), S, CHUNK_PMP_MAX_THREADS, N, rep,
                         kCases[c], container ? "array" : "vector", s, s * 1e9 / static_cast<double>(N));
        }
    };

    for (int rep = 0; rep < reps; ++rep)
    {
        const bool vec_first = (rep % 2) == 0; // alternate who goes first

        // The varied array and vector, built outside the timings.
        Array<double> a(n);
        arr_write(a, n, varied);
        vec_write(v, varied);

        // A pair of runs of one case, in the alternating order.
        const auto pair = [&](int c, auto&& run_vec, auto&& run_arr)
        {
            Sample sv{};
            Sample sa{};
            if (vec_first)
            {
                sv = run_vec();
                sa = run_arr();
            }
            else
            {
                sa = run_arr();
                sv = run_vec();
            }
            note(c, 0, rep, sv.seconds);
            note(c, 1, rep, sa.seconds);
            return std::pair<Sample, Sample>{sv, sa};
        };

        // read-varied
        {
            const auto [sv, sa] = pair(0, [&] { return vec_read(v); }, [&] { return arr_read(a, n); });
            check_sum(0, sv.sum, sa.sum);
        }
        // read-constant: an array all at 0, a vector of zeros.
        {
            Array<double> z(n);
            const auto [sv, sa] = pair(1, [&] { return vec_read(zeros); }, [&] { return arr_read(z, n); });
            check_sum(1, sv.sum, sa.sum);
        }
        // read-random
        {
            const auto [sv, sa] =
                pair(2, [&] { return vec_read_random(v, idx); }, [&] { return arr_read_random(a, idx); });
            check_sum(2, sv.sum, sa.sum);
        }
        // write-varied: the materialised array overwritten.
        {
            pair(3, [&] { return vec_write(v, varied2); }, [&] { return arr_write(a, n, varied2); });
            check_sum(3, vec_sum(v), arr_sum(a, n));
            if (verify)
            {
                bad_elements += verify_elements("write-varied", a, v);
            }
        }
        // write-fresh: varied values into an array all at 0 (every chunk
        // materialised by its first write) and into a zeroed vector.
        {
            Array<double>       f(n);
            std::vector<double> fv(static_cast<std::size_t>(N), 0.);
            pair(4, [&] { return vec_write(fv, varied); }, [&] { return arr_write(f, n, varied); });
            check_sum(4, vec_sum(fv), arr_sum(f, n));
            if (verify)
            {
                bad_elements += verify_elements("write-fresh", f, fv);
            }
        }
        // rmw on the (overwritten) varied array and vector.
        {
            pair(5, [&] { return vec_rmw(v); }, [&] { return arr_rmw(a, n); });
            check_sum(5, vec_sum(v), arr_sum(a, n));
            if (verify)
            {
                bad_elements += verify_elements("rmw", a, v);
            }
        }
    }

    if (!csv)
    {
        std::println("{:<14} {:>14} {:>14} {:>14} {:>14} {:>8}", "case", "vector med ns", "vector min ns",
                     "array med ns", "array min ns", "ratio");
        for (int c = 0; c < kNCases; ++c)
        {
            const double per = 1e9 / static_cast<double>(N);
            const double vm  = median(t[c][0]) * per;
            const double am  = median(t[c][1]) * per;
            std::println("{:<14} {:>14.3f} {:>14.3f} {:>14.3f} {:>14.3f} {:>8.2f}", kCases[c], vm,
                         minimum(t[c][0]) * per, am, minimum(t[c][1]) * per, vm > 0. ? am / vm : 0.);
        }
        std::println("(ns per element, wall clock / n; ratio = array median / vector median)");
    }

    if (sum_mismatch || bad_elements)
    {
        std::fprintf(stderr, "access_cost: FAILED: %ld sum mismatch(es), %ld element(s) differ\n", sum_mismatch,
                     bad_elements);
        return 1;
    }
    return 0;
}
