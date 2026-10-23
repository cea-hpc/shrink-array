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
 * @file fill_range_mt_validation.cpp
 * @brief fill_range_mt(), copy_range_mt() and transform_range_mt() called
 *        from every thread at once, interleaved with element writes on the
 *        SAME chunks, against a std::vector reference.
 *
 * @details Each round, inside one parallel region and with no barrier between
 *          the two loops (nowait), the index space is cut into blocks of 64:
 *          blocks of one kind go to the bulk writer of the round -- by turns
 *          fill_range_mt(), copy_range_mt() from a std::vector, and
 *          transform_range_mt() from another chunked array (its const
 *          iterator, read outside the write sessions) -- the others to element
 *          writes `arr[i] = ...`. A 64-block is smaller than a chunk, so a
 *          chunk is shared by several threads and by both kinds of writer --
 *          the case the per-chunk write session must survive. Values rotate
 *          between the first constant (0), a repeated constant (9) and
 *          per-round values, so chunks materialise, return to a constant and
 *          get compressed (write-path compressions, and an explicit pass every
 *          few rounds) while the fills run. Every round ends with a full
 *          comparison against the reference.
 *
 *          Usage: fill_range_mt_validation [N] [rounds]
 *                 fill_range_mt_validation N perf   -- throughput, element loop
 *                                                      vs fill_range_mt and vs
 *                                                      copy_range_mt
 */

#include <chrono>      // std::chrono::steady_clock, std::chrono::duration
#include <cstring>     // std::strcmp
#include <type_traits> // std::conditional_t
#include <vector>      // std::vector

#include "chunk/chunk_traits.h"          // forward decls + default IntType = int
#include "chunk/zero_array.h"           // zero::Array
#include "chunk/zero_one_value_array.h"          // zero_one_value::Array
#include "chunk/zero_any_array.h"         // zero_any::Array
#include "chunk/array_chunk_range_ops.h" // fill_range_mt
#include "chunk/param_chunk.h"           // ParamChunk::init_indexes
#include "chunk/config.h"                // chunk_config::use_zero_one_value*, build_config::has_not_ompt
#include "chunk/PMP_CHUNK.h"             // CHUNK_PMP_MAX_THREADS
#include "chunk/manage_ompt.h"           // begin_parallel_phase / finalize_parallel_phase
#include "chunk/internal/env_parse.h"    // env_number / arg_number (checked, not atoi)
#include "chunk/internal/print.h"        // shrink_array::println

#include "chunk/zero_chunk.h"    // release_chunk_pools, destroy_block_chunk
#include "validation_harness.h" // self_command; run() is called by validation_driver.cpp

namespace validation::fill_range_mt_validation
{

using namespace shrink_array;

static void begin_phase()
{
    if constexpr (build_config::has_not_ompt)
    {
        shrink_array::begin_parallel_phase();
    }
}
static void end_phase()
{
    if constexpr (build_config::has_not_ompt)
    {
        shrink_array::finalize_parallel_phase();
    }
}

template <typename T>
using Array =
    std::conditional_t<chunk_config::use_zero_any,
                       zero_any::Array<T>,
                       std::conditional_t<chunk_config::use_zero_one_value, zero_one_value::Array<T>, zero::Array<T>>>;

constexpr const char* variant_name()
{
    if constexpr (chunk_config::use_zero_any)
    {
        return "chunk2a";
    }
    else if constexpr (chunk_config::use_zero_one_value)
    {
        return "chunk2";
    }
    else
    {
        return "chunk";
    }
}

constexpr long blk = 64;

// Value written by the element writers of round r at index i: mostly a
// constant, so element-written chunks also come and go.
static double elem_value(long i, int r)
{
    const long m = ((i / 700) + r) % 4;
    return m == 0 ? 0.0 : m == 1 ? 9.0 : 1.0 + static_cast<double>((i + r) % 1000);
}

// Source value of the copy / transform rounds at index i: runs of the first
// constant, of a repeated constant and of varying values, moving with r.
static double src_value(long i, int r)
{
    const long m = ((i / 500) + r) % 3;
    return m == 0 ? 0.0 : m == 1 ? 9.0 : 0.5 + static_cast<double>((i * 7 + r) % 991);
}

// The operation of the transform rounds: keeps 0 at 0, so first-constant runs
// stay compressible.
static double transform_op(double x)
{
    return 2.0 * x;
}

/// The bulk writer of round r: fill, copy, transform -- three rounds each.
enum class Mode
{
    fill,
    copy,
    transform
};
static Mode mode_of(int r)
{
    const int m = (r / 3) % 3;
    return m == 0 ? Mode::fill : m == 1 ? Mode::copy : Mode::transform;
}
static const char* mode_name(Mode m)
{
    return m == Mode::fill ? "fill_range_mt" : m == Mode::copy ? "copy_range_mt" : "transform_range_mt";
}

static int run_validation(long n, int rounds)
{
    Array<double>       arr(n);
    std::vector<double> ref(static_cast<std::size_t>(n), 0.0);
    std::vector<double> src(static_cast<std::size_t>(n), 0.0); // copy rounds
    Array<double>       src_arr(n);                            // transform rounds: a chunked source
    const long          nblk = (n + blk - 1) / blk;
    long                bad  = 0;

    for (int r = 0; r < rounds && bad == 0; ++r)
    {
        const Mode mode = mode_of(r);
        // Fill value of this round: first constant, repeated constant, or new.
        const double fill_v = (r % 3 == 0) ? 0.0 : (r % 3 == 1) ? 9.0 : 100.0 + r;
        // The sources, written serially before the round (no writer of
        // src_arr during it: its iterator's contract).
        if (mode == Mode::copy)
        {
            for (long i = 0; i < n; ++i)
            {
                src[static_cast<std::size_t>(i)] = src_value(i, r);
            }
        }
        else if (mode == Mode::transform)
        {
            for (long i = 0; i < n; ++i)
            {
                src_arr[i] = src_value(i, r);
            }
        }
        const Array<double>& csrc = src_arr;
        // Which blocks are filled this round: a pattern that moves with r, so a
        // block alternates between the two kinds of writer over the rounds.
        const auto is_fill = [r](long b) { return (((b / 3) + r) % 2) == 0; };

#pragma omp parallel
        {
            begin_phase();
#pragma omp for schedule(static, 1) nowait
            for (long b = 0; b < nblk; ++b)
            {
                if (!is_fill(b))
                {
                    continue;
                }
                const long lo  = b * blk;
                const long hi  = lo + blk < n ? lo + blk : n;
                const auto ulo = static_cast<std::size_t>(lo);
                const auto uhi = static_cast<std::size_t>(hi);
                if (mode == Mode::fill)
                {
                    fill_range_mt(arr, ulo, uhi, fill_v);
                }
                else if (mode == Mode::copy)
                {
                    copy_range_mt(arr, ulo, uhi, src.begin() + lo);
                }
                else
                {
                    transform_range_mt(arr, ulo, uhi, csrc.begin() + lo, transform_op);
                }
            }
#pragma omp for schedule(static, 1) nowait
            for (long b = 0; b < nblk; ++b)
            {
                if (is_fill(b))
                {
                    continue;
                }
                const long lo = b * blk;
                const long hi = lo + blk < n ? lo + blk : n;
                for (long i = lo; i < hi; ++i)
                {
                    arr[i] = elem_value(i, r);
                }
            }
        }
        end_phase();

        for (long b = 0; b < nblk; ++b)
        {
            const long lo = b * blk;
            const long hi = lo + blk < n ? lo + blk : n;
            for (long i = lo; i < hi; ++i)
            {
                const double bulk                = mode == Mode::fill   ? fill_v
                                                   : mode == Mode::copy ? src_value(i, r)
                                                                        : transform_op(src_value(i, r));
                ref[static_cast<std::size_t>(i)] = is_fill(b) ? bulk : elem_value(i, r);
            }
        }

        if (r % 5 == 4) // an explicit compression pass between rounds
        {
#pragma omp parallel
            {
                begin_phase();
#pragma omp single
                arr.compress();
            }
            end_phase();
        }

        for (long i = 0; i < n; ++i)
        {
            const double got = arr[i];
            if (got != ref[static_cast<std::size_t>(i)])
            {
                if (bad < 5)
                {
                    println("[{}] FAIL round {} ({}): arr[{}] = {:g}, expected {:g}", variant_name(), r,
                            mode_name(mode), i, got, ref[static_cast<std::size_t>(i)]);
                }
                ++bad;
            }
        }
    }
    arr.clear();
    src_arr.clear();
    if (bad)
    {
        println("[{}] {} mismatch(es)", variant_name(), bad);
    }
    else
    {
        println("[{}] OK: {} rounds, N={}, fill / copy / transform and element writes interleaved on shared "
                "chunks",
                variant_name(), rounds, n);
    }
    return bad ? 1 : 0;
}

// Perf blocks: half the smallest chunk this bench builds (64), with values
// alternating between neighbours, so no chunk is ever uniform -- a uniform
// Chunk2A chunk compresses to its second value, and the timings would then
// measure compression churn instead of writes.
constexpr long perf_blk = 32;

static void run_perf(long n)
{
    Array<double>       arr(n);
    const long          nblk = (n + perf_blk - 1) / perf_blk;
    const auto          now  = [] { return std::chrono::steady_clock::now(); };
    double              t_el = 0;
    double              t_mt = 0;
    double              t_ce = 0; // copy: element loop
    double              t_cm = 0; // copy: copy_range_mt
    std::vector<double> src(static_cast<std::size_t>(n));
    for (int rep = 0; rep < 6; ++rep)
    {
        const double v  = 1.0 + rep;
        const auto   vb = [v](long b) { return v + (0.25 * static_cast<double>(b & 1)); }; // neighbours differ
        const auto   t0 = now();
#pragma omp parallel
        {
            begin_phase();
#pragma omp for schedule(static, 1)
            for (long b = 0; b < nblk; ++b)
            {
                const long   lo = b * perf_blk;
                const long   hi = lo + perf_blk < n ? lo + perf_blk : n;
                const double x  = vb(b);
                for (long i = lo; i < hi; ++i)
                {
                    arr[i] = x;
                }
            }
        }
        end_phase();
        const auto t1 = now();
#pragma omp parallel
        {
            begin_phase();
#pragma omp for schedule(static, 1)
            for (long b = 0; b < nblk; ++b)
            {
                const long lo = b * perf_blk;
                const long hi = lo + perf_blk < n ? lo + perf_blk : n;
                fill_range_mt(arr, static_cast<std::size_t>(lo), static_cast<std::size_t>(hi), vb(b) + 0.5);
            }
        }
        end_phase();
        const auto t2 = now();
        // Copy: neighbouring elements differ, so no chunk is ever uniform.
        for (long i = 0; i < n; ++i)
        {
            src[static_cast<std::size_t>(i)] = v + (0.125 * static_cast<double>(i % 7));
        }
        const auto t3 = now();
#pragma omp parallel
        {
            begin_phase();
#pragma omp for schedule(static, 1)
            for (long b = 0; b < nblk; ++b)
            {
                const long lo = b * perf_blk;
                const long hi = lo + perf_blk < n ? lo + perf_blk : n;
                for (long i = lo; i < hi; ++i)
                {
                    arr[i] = src[static_cast<std::size_t>(i)];
                }
            }
        }
        end_phase();
        const auto t4 = now();
#pragma omp parallel
        {
            begin_phase();
#pragma omp for schedule(static, 1)
            for (long b = 0; b < nblk; ++b)
            {
                const long lo = b * perf_blk;
                const long hi = lo + perf_blk < n ? lo + perf_blk : n;
                copy_range_mt(arr, static_cast<std::size_t>(lo), static_cast<std::size_t>(hi), src.begin() + lo);
            }
        }
        end_phase();
        const auto t5 = now();
        if (rep > 0) // the first round materialises everything: not measured
        {
            t_el += std::chrono::duration<double>(t1 - t0).count();
            t_mt += std::chrono::duration<double>(t2 - t1).count();
            t_ce += std::chrono::duration<double>(t4 - t3).count();
            t_cm += std::chrono::duration<double>(t5 - t4).count();
        }
    }
    println("[{}] perf N={} threads={}: element loop {:.3f} s, fill_range_mt {:.3f} s (x{:.2f})", variant_name(), n,
            CHUNK_PMP_MAX_THREADS, t_el, t_mt, t_mt > 0 ? t_el / t_mt : 0.);
    println("[{}] perf N={} threads={}: copy element loop {:.3f} s, copy_range_mt {:.3f} s (x{:.2f})", variant_name(),
            n, CHUNK_PMP_MAX_THREADS, t_ce, t_cm, t_cm > 0 ? t_ce / t_cm : 0.);
    arr.clear();
}

static int run_body(int argc, char** argv)
{
    const long n = argc > 1 ? shrink_array::arg_number<long>(argv[1], (1L << 20) + 777, "N") : (1L << 20) + 777;
    ParamChunk::init_indexes(true);

    if (argc > 2 && std::strcmp(argv[2], "perf") == 0)
    {
        run_perf(n);
        return 0;
    }
    const int rounds = argc > 2 ? shrink_array::arg_number<int>(argv[2], 30, "rounds") : 30;
    println("=== fill_range_mt_validation [variant = {}]  N={}  threads={} ===", variant_name(), n,
            CHUNK_PMP_MAX_THREADS);
    const int fails = run_validation(n, rounds);
    println("[{}] === {} ===", variant_name(), fails ? "FILL_RANGE_MT FAILED" : "FILL_RANGE_MT OK");
    return fails;
}

int run(int argc, char** argv)
{
    // The arrays live in run_body() and are destroyed when it returns, which
    // parks their buffers in the reuse lists; release_chunk_pools() frees them.
    // Without it they stay parked until exit, and LeakSanitizer reports every
    // one in the std-alloc builds (each buffer is a real malloc there).
    const int rc = run_body(argc, argv);
    release_chunk_pools();
    destroy_block_chunk();
    return rc;
}

} // namespace validation::fill_range_mt_validation
