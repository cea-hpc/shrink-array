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
 *            read-bulk       read-varied through for_each_value() (each thread
 *                            its slice): one gate per chunk and one read
 *                            section per block, instead of both per element
 *            read-second     sum of every element of an array all at 1.5,
 *                            written outside the timing: every chunk
 *                            compressed to a constant other than zero where
 *                            the variant can (zero_any: a second value, read
 *                            back through the registry's value table; zero:
 *                            materialised); the vector: all 1.5
 *
 *          A write case's time includes the end of its phase (the deferred
 *          writes flushed, the between-region compression pass): what a
 *          program pays for the region. The array's end-of-run statistics
 *          are not timed. Printed per case: the median and minimum time per
 *          element (wall clock / n_elems, in ns) of the vector and of the array,
 *          and the ratio of the medians. --csv prints one line per case and
 *          repetition instead (python/access_cost.py aggregates them).
 *
 *          A sample lasts at least --min_time seconds (default 0.05): the first
 *          pass of the vector and of the array gives the number of passes, the
 *          same for both (so that rmw's sums still match), and the sample is the
 *          mean of the passes. At a high thread count a pass lasts a few
 *          milliseconds, the order of a parallel region's fork and barrier: one
 *          pass alone gave spreads of several hundred percent. write-fresh gets
 *          a new array all at 0 and a re-zeroed vector before each pass, outside
 *          the timing.
 *
 *          NUMA placement: the vectors are allocated without initialisation and
 *          first touched by a parallel fill with the timed loops' static
 *          schedule, so that each page sits on the node of the thread that
 *          reads it, as the array's chunks do (allocated by the writing
 *          thread). A vector initialised by the main thread has every page on
 *          one node: across several sockets the vector is held to that node's
 *          bandwidth, and the array then looks faster than the vector.
 *
 *          Every value is a multiple of 0.5 well under 2^40, so the sums are
 *          exact in double whatever the reduction order: each case's sum must
 *          be the same on the vector and on the array, and is checked (exit 1
 *          otherwise). The elements are checked against the vector after the
 *          writes when --verify is given.
 *
 *          Memory: the vector (8 n_elems bytes), the array (about the same at its
 *          largest), the random indices (4 n_elems bytes), and the constant vectors
 *          of read-constant and read-second (8 n_elems bytes each): about 36
 *          bytes per element. The default n_elems is max(2^27, threads x 2^20),
 *          at least 2^20 elements per thread so that a case lasts long enough
 *          to time at a high thread count: 2^27 (about 4.5 GB) up to 128
 *          threads, about 3.0e8 (11 GB) at 288.
 *
 *          Usage: access_cost [--n n_elems] [--reps R] [--min_time S] [--csv] [--verify]
 */

#include <algorithm>    // std::sort, std::min_element, std::max, std::min
#include <chrono>       // std::chrono::steady_clock
#include <cmath>        // std::ceil (passes_for)
#include <cstdint>      // std::uint64_t
#include <charconv>     // std::from_chars (arg_value)
#include <cstdio>       // std::fprintf, stderr
#include <cstring>      // std::strlen
#include <exception>    // std::exception (main's catch)
#include <iterator>     // std::size
#include <memory>       // std::allocator (NoInitAllocator)
#include <new>          // placement new (NoInitAllocator)
#include <optional>     // std::optional (write-fresh's array, one per pass)
#include <string>       // std::string
#include <system_error> // std::errc (arg_value)
#include <type_traits>  // std::conditional_t, std::is_nothrow_default_constructible_v
#include <utility>      // std::as_const, std::forward
#include <vector>       // std::vector

#include "chunk/zero_array.h"          // zero::Array
#include "chunk/zero_or_value_array.h" // zero_or_value::Array
#include "chunk/zero_any_array.h"      // zero_any::Array
#include "chunk/runtime.h"             // Runtime::init
#include "chunk/config.h"              // chunk_config::use_zero*, build_config::use_openmp / needs_explicit_phases
#include "chunk/threads.h"             // shrink_array::max_threads, team_size, thread_rank (read-bulk)
#include "chunk/phases.h"              // begin_parallel_phase, finalize_parallel_phase
#include "chunk/print.h"               // shrink_array::println

using namespace shrink_array;

namespace
{

/// @brief A number of the command line, checked (std::from_chars, not atoi):
///        @p fallback kept, with a warning, when @p text is not a whole number.
template <typename T>
T arg_value(const char* text, T fallback, const char* name)
{
    T           value{};
    const char* end = text + std::strlen(text);
    if (const auto [ptr, ec] = std::from_chars(text, end, value); ec == std::errc{} && ptr == end)
    {
        return value;
    }
    println(stderr, "access_cost: {} '{}' is not a number, {} kept", name, text, fallback);
    return fallback;
}

} // namespace

static_assert(chunk_config::use_zero || chunk_config::use_zero_or_value || chunk_config::use_zero_any,
              "access_cost: link chunk_zero_*, chunk_zero_or_value_* or chunk_zero_any_*");

template <typename T>
using Array =
    std::conditional_t<chunk_config::use_zero_any,
                       zero_any::Array<T>,
                       std::conditional_t<chunk_config::use_zero_or_value, zero_or_value::Array<T>, zero::Array<T>>>;

constexpr const char* variant_name()
{
    if constexpr (chunk_config::use_zero_any)
    {
        return "zero_any";
    }
    else if constexpr (chunk_config::use_zero_or_value)
    {
        return "zero_or_value";
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
    if constexpr (build_config::needs_explicit_phases)
    {
        shrink_array::begin_parallel_phase();
    }
}

void end_phase(Array<double>& arr)
{
    if constexpr (build_config::needs_explicit_phases)
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

/// @brief std::allocator that default-initialises instead of value-initialising:
///        a vector of n doubles leaves its pages untouched, for fill_parallel()
///        to touch first (see the NUMA placement above).
template <typename T>
struct NoInitAllocator : std::allocator<T>
{
    NoInitAllocator() = default;
    template <typename U>
    explicit NoInitAllocator(const NoInitAllocator<U>& /*other*/) noexcept
    {
    }
    template <typename U>
    void construct(U* ptr) noexcept(std::is_nothrow_default_constructible_v<U>)
    {
        ::new (static_cast<void*>(ptr)) U;
    }
    template <typename U, typename... Args>
    void construct(U* ptr, Args&&... args)
    {
        ::new (static_cast<void*>(ptr)) U(std::forward<Args>(args)...);
    }
};

using Vec = std::vector<double, NoInitAllocator<double>>;
using Idx = std::vector<int, NoInitAllocator<int>>;

/// Every element set to @p value, with the timed loops' static schedule: the
/// first touch of a NoInitAllocator vector's pages.
template <typename T>
void fill_parallel(std::vector<T, NoInitAllocator<T>>& vec, T value)
{
    const auto count = static_cast<int>(vec.size());
#pragma omp parallel for schedule(static)
    for (int i = 0; i < count; ++i)
    {
        vec[static_cast<std::size_t>(i)] = value;
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

/// A fixed pseudo-random index sequence (splitmix64), every index in [0, count).
Idx random_indices(long count)
{
    Idx        idx(static_cast<std::size_t>(count)); // untouched: the loop below touches it first
    const auto ni = static_cast<int>(count);
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
Sample vec_read(const Vec& vec)
{
    const auto t0    = Clk::now();
    const auto count = static_cast<int>(vec.size());
    double     sum   = 0.;
#pragma omp parallel for schedule(static) reduction(+ : sum)
    for (int i = 0; i < count; ++i)
    {
        sum += vec[static_cast<std::size_t>(i)];
    }
    return {secs_since(t0), sum};
}

Sample vec_read_random(const Vec& vec, const Idx& idx)
{
    const auto t0    = Clk::now();
    const auto count = static_cast<int>(idx.size());
    double     sum   = 0.;
#pragma omp parallel for schedule(static) reduction(+ : sum)
    for (int k = 0; k < count; ++k)
    {
        sum += vec[static_cast<std::size_t>(idx[static_cast<std::size_t>(k)])];
    }
    return {secs_since(t0), sum};
}

template <typename F>
Sample vec_write(Vec& vec, F value)
{
    const auto t0    = Clk::now();
    const auto count = static_cast<int>(vec.size());
#pragma omp parallel for schedule(static)
    for (int i = 0; i < count; ++i)
    {
        vec[static_cast<std::size_t>(i)] = value(i);
    }
    return {secs_since(t0), 0.};
}

Sample vec_rmw(Vec& vec)
{
    const auto t0    = Clk::now();
    const auto count = static_cast<int>(vec.size());
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

// The bulk read (ArrayChunkBase::for_each_value): each thread its contiguous
// slice, as schedule(static) would give it.
Sample arr_read_bulk(Array<double>& arr, int count)
{
    const auto  t0  = Clk::now();
    double      sum = 0.;
    const auto& ca  = std::as_const(arr);
#pragma omp parallel reduction(+ : sum)
    {
        begin_phase();
        const long nt   = shrink_array::team_size();
        const long rank = shrink_array::thread_rank();
        const auto lo   = static_cast<int>(count * rank / nt);
        const auto hi   = static_cast<int>(count * (rank + 1) / nt);
        double     part = 0.;
        ca.for_each_value(lo, hi, [&part](double v) { part += v; });
        sum += part;
    }
    end_phase(arr);
    return {secs_since(t0), sum};
}

Sample arr_read_random(Array<double>& arr, const Idx& idx)
{
    const auto  t0    = Clk::now();
    const auto  count = static_cast<int>(idx.size());
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

double vec_sum(const Vec& vec)
{
    double sum = 0.;
    for (const double value : vec)
    {
        sum += value;
    }
    return sum;
}

long verify_elements(const char* what, const Array<double>& arr, const Vec& vec)
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

constexpr const char* case_names[] = {"read-varied", "read-constant", "read-random", "write-varied",
                                      "write-fresh", "rmw",           "read-second", "read-bulk"};
constexpr int         n_cases      = static_cast<int>(std::size(case_names));
} // namespace

int main(int argc, char** argv)
try
{
    Runtime::init(true);

    // At least 2^20 elements per thread: with a fixed 2^27, 288 threads had
    // 466k elements each, a case lasted a few milliseconds, and the region
    // and phase overheads made the medians too noisy to compare.
    constexpr long min_elems_per_thread = 1L << 20;
    long           n_elems = std::max(1L << 27, static_cast<long>(shrink_array::max_threads()) * min_elems_per_thread);
    n_elems                = std::min(n_elems, 0x7fffffffL);
    int    reps            = 5;
    double min_time        = 0.05; // seconds per sample, at least (see the header)
    bool   csv             = false;
    bool   verify          = false;
    for (int a = 1; a < argc; ++a)
    {
        const std::string opt = argv[a];
        if (opt == "--n" && a + 1 < argc)
        {
            n_elems = arg_value<long>(argv[++a], n_elems, "--n");
        }
        else if (opt == "--reps" && a + 1 < argc)
        {
            reps = arg_value<int>(argv[++a], reps, "--reps");
        }
        else if (opt == "--min_time" && a + 1 < argc)
        {
            min_time = arg_value<double>(argv[++a], min_time, "--min_time");
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
            std::fprintf(stderr, "usage: %s [--n N] [--reps R] [--min_time S] [--csv] [--verify]\n", argv[0]);
            return 2;
        }
    }
    if (reps < 1 || n_elems < 1 || n_elems > 0x7fffffffL || !(min_time >= 0.))
    {
        std::fprintf(stderr, "access_cost: need reps >= 1, 1 <= n < 2^31, min_time >= 0\n");
        return 2;
    }
    const auto count = static_cast<int>(n_elems);

    // Read off an EMPTY array: no chunk, no end-of-run stats line of its own.
    const auto chunk_sz = static_cast<long>(Array<double>(0).chunk_size());
    if (csv)
    {
        println("variant,chunk_size,threads,n,rep,case,container,seconds,ns_per_elem,passes");
    }
    else
    {
        println("=== access_cost [{}]  n={}  chunk_size={}  threads={}  reps={}  min_time={}s ===", variant_name(),
                n_elems, chunk_sz, shrink_array::max_threads(), reps, min_time);
    }

    // Allocated untouched, first touched in parallel (the NUMA placement above):
    // every vector here, before any timing.
    const Idx idx = random_indices(n_elems);
    Vec       vec(static_cast<std::size_t>(n_elems));
    Vec       zeros(static_cast<std::size_t>(n_elems));
    Vec       seconds(static_cast<std::size_t>(n_elems));
    Vec       fresh_vec(static_cast<std::size_t>(n_elems)); // write-fresh's, re-zeroed before each pass
    fill_parallel(zeros, 0.);
    fill_parallel(seconds, 1.5);
    vec_write(vec, varied);

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
    const auto note = [&](int c, int container, int rep, double seconds, int passes)
    {
        times[c][container].push_back(seconds);
        if (csv)
        {
            println("{},{},{},{},{},{},{},{:.9f},{:.4f},{}", variant_name(), chunk_sz, shrink_array::max_threads(),
                    n_elems, rep, case_names[c], container ? "array" : "vector", seconds,
                    seconds * 1e9 / static_cast<double>(n_elems), passes);
        }
    };
    // The passes of a sample: enough for the shorter first pass to reach
    // min_time, capped.
    const auto passes_for = [min_time](double first_pass)
    {
        constexpr int max_passes = 1000;
        if (first_pass >= min_time)
        {
            return 1;
        }
        if (first_pass <= 0.)
        {
            return max_passes;
        }
        return static_cast<int>(std::min(static_cast<double>(max_passes), std::ceil(min_time / first_pass)));
    };

    for (int rep = 0; rep < reps; ++rep)
    {
        const bool vec_first = (rep % 2) == 0; // alternate who goes first

        // The varied array and vector, built outside the timings.
        Array<double> arr(count);
        arr_write(arr, count, varied);
        vec_write(vec, varied);

        // One sample of a case for each container: the same number of passes
        // for both (passes_for() of the shorter first pass), each pass in the
        // alternating order and after its untimed preparation; the time is the
        // mean of the passes, the sum the last pass's.
        const auto pair_prepared = [&](int c, auto&& prep_vec, auto&& run_vec, auto&& prep_arr, auto&& run_arr)
        {
            Sample     sv{};
            Sample     sa{};
            double     tv     = 0.;
            double     ta     = 0.;
            const auto go_vec = [&]
            {
                prep_vec();
                sv = run_vec();
                tv += sv.seconds;
            };
            const auto go_arr = [&]
            {
                prep_arr();
                sa = run_arr();
                ta += sa.seconds;
            };
            const auto one_pass = [&]
            {
                if (vec_first)
                {
                    go_vec();
                    go_arr();
                }
                else
                {
                    go_arr();
                    go_vec();
                }
            };
            one_pass();
            const int passes = passes_for(std::min(tv, ta));
            for (int p = 1; p < passes; ++p)
            {
                one_pass();
            }
            sv.seconds = tv / passes;
            sa.seconds = ta / passes;
            note(c, 0, rep, sv.seconds, passes);
            note(c, 1, rep, sa.seconds, passes);
            return SamplePair{sv, sa};
        };
        const auto nothing = [] {};
        const auto pair    = [&](int c, auto&& run_vec, auto&& run_arr)
        { return pair_prepared(c, nothing, run_vec, nothing, run_arr); };

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
        // materialised by its first write) and into a zeroed vector, both
        // fresh again before each pass.
        {
            std::optional<Array<double>> filled;
            pair_prepared(
                4, [&] { fill_parallel(fresh_vec, 0.); }, [&] { return vec_write(fresh_vec, varied); },
                [&]
                {
                    filled.reset(); // the previous pass's array freed first
                    filled.emplace(count);
                },
                [&] { return arr_write(*filled, count, varied); });
            check_sum(4, vec_sum(fresh_vec), arr_sum(*filled, count));
            if (verify)
            {
                bad_elements += verify_elements("write-fresh", *filled, fresh_vec);
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
        // read-bulk: the varied array again, through for_each_value().
        {
            const auto [sv, sa] = pair(7, [&] { return vec_read(vec); }, [&] { return arr_read_bulk(arr, count); });
            check_sum(7, sv.sum, sa.sum);
        }
        // read-second: an array all at 1.5 (its end of phase compresses it),
        // a vector of 1.5. The sum is exact: n x 1.5 < 2^53.
        {
            Array<double> second_arr(count);
            arr_write(second_arr, count, [](long) { return 1.5; });
            const auto [sv, sa] =
                pair(6, [&] { return vec_read(seconds); }, [&] { return arr_read(second_arr, count); });
            check_sum(6, sv.sum, sa.sum);
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
catch (const std::exception& e)
{
    // A failed allocation, std::format or a bad argument: said, not std::terminate().
    std::fprintf(stderr, "access_cost: %s\n", e.what());
    return 2;
}
