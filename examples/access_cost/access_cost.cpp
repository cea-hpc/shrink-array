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
 *          own parallel region with a static `omp for` over the n_elems elements:
 *
 *            read-varied     sum of every element, every chunk materialised
 *                            with varied values (nothing to compress)
 *            read-constant   sum of every element of an array all at 0 (every
 *                            chunk compressed; the vector: all zeros)
 *            read-random     sum of n_elems elements at random indices (a fixed
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
 *          element (wall clock / n_elems, in ns) of the vector and of the array,
 *          and the ratio of the medians. --csv prints one line per case and
 *          repetition instead (python/access_cost.py aggregates them).
 *
 *          Every value is a multiple of 0.5 well under 2^40, so the sums are
 *          exact in double whatever the reduction order: each case's sum must
 *          be the same on the vector and on the array, and is checked (exit 1
 *          otherwise). The elements are checked against the vector after the
 *          writes when --verify is given.
 *
 *          Memory: the vector (8 n_elems bytes), the array (about the same at its
 *          largest), the random indices (4 n_elems bytes). The default n_elems = 2^27
 *          takes about 3 GB.
 *
 *          Usage: access_cost [--n n_elems] [--reps R] [--csv] [--verify]
 */

#include <algorithm>   // std::sort, std::min_element
#include <chrono>      // std::chrono::steady_clock
#include <cstdint>     // std::uint64_t
#include <cstdio>      // std::fprintf, stderr
#include <string>      // std::string
#include <type_traits> // std::conditional_t
#include <utility>     // std::as_const
#include <vector>      // std::vector

#include "chunk/chunk_traits.h"         // the IntType = int default of the arrays
#include "chunk/zero_array.h"           // zero::Array
#include "chunk/zero_one_value_array.h" // zero_one_value::Array
#include "chunk/zero_any_array.h"       // zero_any::Array
#include "chunk/param_chunk.h"          // ParamChunk::init_indexes
#include "chunk/config.h"               // chunk_config::use_zero*, build_config::use_openmp / has_not_ompt
#include "chunk/PMP_CHUNK.h"            // CHUNK_PMP_MAX_THREADS
#include "chunk/manage_ompt.h"          // begin_parallel_phase, finalize_parallel_phase
#include "chunk/internal/env_parse.h"   // arg_number (checked, not atoi)
#include "chunk/internal/print.h"       // shrink_array::println

using namespace shrink_array;

static_assert(chunk_config::use_zero || chunk_config::use_zero_one_value || chunk_config::use_zero_any,
              "access_cost: link chunk_zero_*, chunk_zero_one_value_* or chunk_zero_any_*");

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
    // Serial zero_any has no between-region pass: run it (as variant_compare).
    if constexpr (chunk_config::use_zero_any && !build_config::use_openmp)
    {
        arr.make_block_compression();
    }
    else
    {
        (void)arr;
    }
}

/// Varied within any chunk of 64 elements or more (period 97): nothing to compress.
double varied(long index)
{
    return static_cast<double>((index % 97) + 1) * 0.5;
}

/// Another varied value, for the overwrite (period 89, never 0).
double varied2(long index)
{
    return static_cast<double>((index % 89) + 1) * 0.5;
}

/// A fixed pseudo-random index sequence (splitmix64), every index in [0, n).
std::vector<int> random_indices(long count)
{
    std::vector<int> idx(static_cast<std::size_t>(count));
    const int        ni = static_cast<int>(count);
#pragma omp parallel for schedule(static)
    for (int k = 0; k < ni; ++k)
    {
        std::uint64_t state = (static_cast<std::uint64_t>(k) + 1) * 0x9E3779B97F4A7C15ULL;
        state ^= state >> 30;
        state *= 0xBF58476D1CE4E5B9ULL;
        state ^= state >> 27;
        state *= 0x94D049BB133111EBULL;
        state ^= state >> 31;
        idx[static_cast<std::size_t>(k)] = static_cast<int>(state % static_cast<std::uint64_t>(count));
    }
    return idx;
}

using Clk = std::chrono::steady_clock;

double secs_since(Clk::time_point t0)
{
    return std::chrono::duration<double>(Clk::now() - t0).count();
}

struct Sample
{
    double seconds;
    double sum;
};

// One case's two runs. A plain aggregate, not std::pair: returning a pair by
// value draws GCC's -Wpsabi note on AArch64 (C++17 passing of a type with an
// empty base, changed in GCC 10.1).
struct SamplePair
{
    Sample vec;
    Sample arr;
};

// ── The cases, on a std::vector ──────────────────────────────────────────────
Sample vec_read(const std::vector<double>& vec)
{
    const auto t0    = Clk::now();
    const int  count = static_cast<int>(vec.size());
    double     sum   = 0.;
#pragma omp parallel for schedule(static) reduction(+ : sum)
    for (int i = 0; i < count; ++i)
    {
        sum += vec[static_cast<std::size_t>(i)];
    }
    return {secs_since(t0), sum};
}

Sample vec_read_random(const std::vector<double>& vec, const std::vector<int>& idx)
{
    const auto t0    = Clk::now();
    const int  count = static_cast<int>(idx.size());
    double     sum   = 0.;
#pragma omp parallel for schedule(static) reduction(+ : sum)
    for (int k = 0; k < count; ++k)
    {
        sum += vec[static_cast<std::size_t>(idx[static_cast<std::size_t>(k)])];
    }
    return {secs_since(t0), sum};
}

template <typename F>
Sample vec_write(std::vector<double>& vec, F value)
{
    const auto t0    = Clk::now();
    const int  count = static_cast<int>(vec.size());
#pragma omp parallel for schedule(static)
    for (int i = 0; i < count; ++i)
    {
        vec[static_cast<std::size_t>(i)] = value(i);
    }
    return {secs_since(t0), 0.};
}

Sample vec_rmw(std::vector<double>& vec)
{
    const auto t0    = Clk::now();
    const int  count = static_cast<int>(vec.size());
#pragma omp parallel for schedule(static)
    for (int i = 0; i < count; ++i)
    {
        vec[static_cast<std::size_t>(i)] += 0.5;
    }
    return {secs_since(t0), 0.};
}

// ── The same cases, on the array ─────────────────────────────────────────────
Sample arr_read(Array<double>& arr, int count)
{
    const auto  t0  = Clk::now();
    double      sum = 0.;
    const auto& ca  = std::as_const(arr);
#pragma omp parallel
    {
        begin_phase();
#pragma omp for schedule(static) reduction(+ : sum)
        for (int i = 0; i < count; ++i)
        {
            sum += ca[i];
        }
    }
    end_phase(arr);
    return {secs_since(t0), sum};
}

Sample arr_read_random(Array<double>& arr, const std::vector<int>& idx)
{
    const auto  t0    = Clk::now();
    const int   count = static_cast<int>(idx.size());
    double      sum   = 0.;
    const auto& ca    = std::as_const(arr);
#pragma omp parallel
    {
        begin_phase();
#pragma omp for schedule(static) reduction(+ : sum)
        for (int k = 0; k < count; ++k)
        {
            sum += ca[idx[static_cast<std::size_t>(k)]];
        }
    }
    end_phase(arr);
    return {secs_since(t0), sum};
}

template <typename F>
Sample arr_write(Array<double>& arr, int count, F value)
{
    const auto t0 = Clk::now();
#pragma omp parallel
    {
        begin_phase();
#pragma omp for schedule(static)
        for (int i = 0; i < count; ++i)
        {
            arr[i] = value(i);
        }
    }
    end_phase(arr);
    return {secs_since(t0), 0.};
}

Sample arr_rmw(Array<double>& arr, int count)
{
    const auto t0 = Clk::now();
#pragma omp parallel
    {
        begin_phase();
#pragma omp for schedule(static)
        for (int i = 0; i < count; ++i)
        {
            arr[i] += 0.5;
        }
    }
    end_phase(arr);
    return {secs_since(t0), 0.};
}

/// The sum of the elements, read serially (the --verify / sum check of the writes).
double arr_sum(const Array<double>& arr, int count)
{
    double sum = 0.;
    for (int i = 0; i < count; ++i)
    {
        sum += arr[i];
    }
    return sum;
}

double vec_sum(const std::vector<double>& vec)
{
    double sum = 0.;
    for (const double value : vec)
    {
        sum += value;
    }
    return sum;
}

long verify_elements(const char* what, const Array<double>& arr, const std::vector<double>& vec)
{
    long bad = 0;
    for (std::size_t i = 0; i < vec.size(); ++i)
    {
        if (arr[static_cast<int>(i)] != vec[i])
        {
            if (bad < 3)
            {
                std::fprintf(stderr, "access_cost: %s: element %zu is %g, the vector has %g\n", what, i,
                             arr[static_cast<int>(i)], vec[i]);
            }
            ++bad;
        }
    }
    return bad;
}

double median(std::vector<double> samples)
{
    std::sort(samples.begin(), samples.end());
    const std::size_t mid = samples.size() / 2;
    return samples.size() % 2 ? samples[mid] : 0.5 * (samples[mid - 1] + samples[mid]);
}

double minimum(const std::vector<double>& samples)
{
    return *std::min_element(samples.begin(), samples.end());
}

constexpr const char* case_names[] = {"read-varied",  "read-constant", "read-random",
                                      "write-varied", "write-fresh",   "rmw"};
constexpr int         n_cases      = static_cast<int>(sizeof(case_names) / sizeof(case_names[0]));
} // namespace

int main(int argc, char** argv)
{
    ParamChunk::init_indexes(true);

    long n_elems = 1L << 27;
    int  reps    = 5;
    bool csv     = false;
    bool verify  = false;
    for (int a = 1; a < argc; ++a)
    {
        const std::string opt = argv[a];
        if (opt == "--n" && a + 1 < argc)
        {
            n_elems = shrink_array::arg_number<long>(argv[++a], n_elems, "--n");
        }
        else if (opt == "--reps" && a + 1 < argc)
        {
            reps = shrink_array::arg_number<int>(argv[++a], reps, "--reps");
        }
        else if (opt == "--csv")
        {
            csv = true;
        }
        else if (opt == "--verify")
        {
            verify = true;
        }
        else
        {
            std::fprintf(stderr, "usage: %s [--n N] [--reps R] [--csv] [--verify]\n", argv[0]);
            return 2;
        }
    }
    if (reps < 1 || n_elems < 1 || n_elems > 0x7fffffffL)
    {
        std::fprintf(stderr, "access_cost: need reps >= 1, 1 <= n < 2^31\n");
        return 2;
    }
    const int count = static_cast<int>(n_elems);

    // Read off an EMPTY array: no chunk, no end-of-run stats line of its own.
    const long chunk_sz = static_cast<long>(Array<double>(0).chunk_size());
    if (csv)
    {
        println("variant,chunk_size,threads,n,rep,case,container,seconds,ns_per_elem");
    }
    else
    {
        println("=== access_cost [{}]  n={}  chunk_size={}  threads={}  reps={} ===", variant_name(), n_elems, chunk_sz,
                CHUNK_PMP_MAX_THREADS, reps);
    }

    const std::vector<int> idx = random_indices(n_elems);
    std::vector<double>    vec(static_cast<std::size_t>(n_elems));
    std::vector<double>    zeros(static_cast<std::size_t>(n_elems), 0.);

    // [case][container 0 = vector, 1 = array] -> seconds per repetition
    std::vector<double> times[n_cases][2];
    long                sum_mismatch = 0;
    long                bad_elements = 0;

    const auto check_sum = [&](int c, double vs, double as)
    {
        if (vs != as)
        {
            std::fprintf(stderr, "access_cost: %s: the array sums to %.17g, the vector to %.17g\n", case_names[c], as,
                         vs);
            ++sum_mismatch;
        }
    };
    const auto note = [&](int c, int container, int rep, double seconds)
    {
        times[c][container].push_back(seconds);
        if (csv)
        {
            println("{},{},{},{},{},{},{},{:.9f},{:.4f}", variant_name(), chunk_sz, CHUNK_PMP_MAX_THREADS, n_elems, rep,
                    case_names[c], container ? "array" : "vector", seconds,
                    seconds * 1e9 / static_cast<double>(n_elems));
        }
    };

    for (int rep = 0; rep < reps; ++rep)
    {
        const bool vec_first = (rep % 2) == 0; // alternate who goes first

        // The varied array and vector, built outside the timings.
        Array<double> arr(count);
        arr_write(arr, count, varied);
        vec_write(vec, varied);

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
            return SamplePair{sv, sa};
        };

        // read-varied
        {
            const auto [sv, sa] = pair(0, [&] { return vec_read(vec); }, [&] { return arr_read(arr, count); });
            check_sum(0, sv.sum, sa.sum);
        }
        // read-constant: an array all at 0, a vector of zeros.
        {
            Array<double> zero_arr(count);
            const auto [sv, sa] = pair(1, [&] { return vec_read(zeros); }, [&] { return arr_read(zero_arr, count); });
            check_sum(1, sv.sum, sa.sum);
        }
        // read-random
        {
            const auto [sv, sa] =
                pair(2, [&] { return vec_read_random(vec, idx); }, [&] { return arr_read_random(arr, idx); });
            check_sum(2, sv.sum, sa.sum);
        }
        // write-varied: the materialised array overwritten.
        {
            pair(3, [&] { return vec_write(vec, varied2); }, [&] { return arr_write(arr, count, varied2); });
            check_sum(3, vec_sum(vec), arr_sum(arr, count));
            if (verify)
            {
                bad_elements += verify_elements("write-varied", arr, vec);
            }
        }
        // write-fresh: varied values into an array all at 0 (every chunk
        // materialised by its first write) and into a zeroed vector.
        {
            Array<double>       filled(count);
            std::vector<double> fv(static_cast<std::size_t>(n_elems), 0.);
            pair(4, [&] { return vec_write(fv, varied); }, [&] { return arr_write(filled, count, varied); });
            check_sum(4, vec_sum(fv), arr_sum(filled, count));
            if (verify)
            {
                bad_elements += verify_elements("write-fresh", filled, fv);
            }
        }
        // rmw on the (overwritten) varied array and vector.
        {
            pair(5, [&] { return vec_rmw(vec); }, [&] { return arr_rmw(arr, count); });
            check_sum(5, vec_sum(vec), arr_sum(arr, count));
            if (verify)
            {
                bad_elements += verify_elements("rmw", arr, vec);
            }
        }
    }

    if (!csv)
    {
        println("{:<14} {:>14} {:>14} {:>14} {:>14} {:>8}", "case", "vector med ns", "vector min ns", "array med ns",
                "array min ns", "ratio");
        for (int c = 0; c < n_cases; ++c)
        {
            const double per = 1e9 / static_cast<double>(n_elems);
            const double vm  = median(times[c][0]) * per;
            const double am  = median(times[c][1]) * per;
            println("{:<14} {:>14.3f} {:>14.3f} {:>14.3f} {:>14.3f} {:>8.2f}", case_names[c], vm,
                    minimum(times[c][0]) * per, am, minimum(times[c][1]) * per, vm > 0. ? am / vm : 0.);
        }
        println("(ns per element, wall clock / n; ratio = array median / vector median)");
    }

    if (sum_mismatch || bad_elements)
    {
        std::fprintf(stderr, "access_cost: FAILED: %ld sum mismatch(es), %ld element(s) differ\n", sum_mismatch,
                     bad_elements);
        return 1;
    }
    return 0;
}
