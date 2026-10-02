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
 * @brief fill_range_mt() called from every thread at once, interleaved with
 *        element writes on the SAME chunks, against a std::vector reference.
 *
 * @details Each round, inside one parallel region and with no barrier between
 *          the two loops (nowait), the index space is cut into blocks of 64:
 *          blocks of one kind go to fill_range_mt(), the others to element
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
 *                                                      vs fill_range_mt
 */

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>
#include <unistd.h>
#include <vector>

#include "chunk/chunk_traits.h"
#include "chunk/array_chunk.h"
#include "chunk/array_chunk2.h"
#include "chunk/array_chunk2A.h"
#include "chunk/array_chunk_range_ops.h"
#include "chunk/param_chunk.h"
#include "chunk/config.h"
#include "chunk/PMP_CHUNK.h"
#include "chunk/manage_ompt.h"
#include "chunk/internal/env_parse.h" // env_number / arg_number (checked, not atoi)

#include "validation_harness.h" // __mem_page_size, self_command; run() is called by validation_driver.cpp

namespace validation::fill_range_mt_validation
{

using namespace shrink_array;

static void begin_phase()
{
    if constexpr (build_config::has_not_ompt)
        shrink_array::begin_parallel_phase();
}
static void end_phase()
{
    if constexpr (build_config::has_not_ompt)
        shrink_array::finalize_parallel_phase();
}

template <typename T>
using Array = std::conditional_t<chunk_config::use_chunk2a,
                                 ArrayChunk2A<T>,
                                 std::conditional_t<chunk_config::use_chunk2, ArrayChunk2<T>, ArrayChunk<T>>>;

constexpr const char* variant_name()
{
    if constexpr (chunk_config::use_chunk2a)
        return "chunk2a";
    else if constexpr (chunk_config::use_chunk2)
        return "chunk2";
    else
        return "chunk";
}

constexpr long kBlk = 64;

// Value written by the element writers of round r at index i: mostly a
// constant, so element-written chunks also come and go.
static double elem_value(long i, int r)
{
    const long m = ((i / 700) + r) % 4;
    return m == 0 ? 0.0 : m == 1 ? 9.0 : 1.0 + static_cast<double>((i + r) % 1000);
}

static int run_validation(long N, int rounds)
{
    Array<double>       arr(N);
    std::vector<double> ref(static_cast<std::size_t>(N), 0.0);
    const long          nblk = (N + kBlk - 1) / kBlk;
    long                bad  = 0;

    for (int r = 0; r < rounds && bad == 0; ++r)
    {
        // Fill value of this round: first constant, repeated constant, or new.
        const double fill_v = (r % 3 == 0) ? 0.0 : (r % 3 == 1) ? 9.0 : 100.0 + r;
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
                    continue;
                const long lo = b * kBlk;
                const long hi = lo + kBlk < N ? lo + kBlk : N;
                fill_range_mt(arr, static_cast<std::size_t>(lo), static_cast<std::size_t>(hi), fill_v);
            }
#pragma omp for schedule(static, 1) nowait
            for (long b = 0; b < nblk; ++b)
            {
                if (is_fill(b))
                    continue;
                const long lo = b * kBlk;
                const long hi = lo + kBlk < N ? lo + kBlk : N;
                for (long i = lo; i < hi; ++i)
                    arr[i] = elem_value(i, r);
            }
        }
        end_phase();

        for (long b = 0; b < nblk; ++b)
        {
            const long lo = b * kBlk;
            const long hi = lo + kBlk < N ? lo + kBlk : N;
            for (long i = lo; i < hi; ++i)
                ref[static_cast<std::size_t>(i)] = is_fill(b) ? fill_v : elem_value(i, r);
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

        for (long i = 0; i < N; ++i)
        {
            const double got = arr[i];
            if (got != ref[static_cast<std::size_t>(i)])
            {
                if (bad < 5)
                    std::printf("[%s] FAIL round %d: arr[%ld] = %g, expected %g\n", variant_name(), r, i, got,
                                ref[static_cast<std::size_t>(i)]);
                ++bad;
            }
        }
    }
    arr.clear();
    if (bad)
        std::printf("[%s] %ld mismatch(es)\n", variant_name(), bad);
    else
        std::printf("[%s] OK: %d rounds, N=%ld, fills and element writes interleaved on shared chunks\n",
                    variant_name(), rounds, N);
    return bad ? 1 : 0;
}

// Perf blocks: half the smallest chunk this bench builds (64), with values
// alternating between neighbours, so no chunk is ever uniform -- a uniform
// Chunk2A chunk compresses to its second value, and the timings would then
// measure compression churn instead of writes.
constexpr long kPerfBlk = 32;

static void run_perf(long N)
{
    Array<double> arr(N);
    const long    nblk = (N + kPerfBlk - 1) / kPerfBlk;
    const auto    now  = [] { return std::chrono::steady_clock::now(); };
    double        t_el = 0;
    double        t_mt = 0;
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
                const long   lo = b * kPerfBlk;
                const long   hi = lo + kPerfBlk < N ? lo + kPerfBlk : N;
                const double x  = vb(b);
                for (long i = lo; i < hi; ++i)
                    arr[i] = x;
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
                const long lo = b * kPerfBlk;
                const long hi = lo + kPerfBlk < N ? lo + kPerfBlk : N;
                fill_range_mt(arr, static_cast<std::size_t>(lo), static_cast<std::size_t>(hi), vb(b) + 0.5);
            }
        }
        end_phase();
        const auto t2 = now();
        if (rep > 0) // the first round materialises everything: not measured
        {
            t_el += std::chrono::duration<double>(t1 - t0).count();
            t_mt += std::chrono::duration<double>(t2 - t1).count();
        }
    }
    std::printf("[%s] perf N=%ld threads=%d: element loop %.3f s, fill_range_mt %.3f s (x%.2f)\n", variant_name(), N,
                CHUNK_PMP_MAX_THREADS, t_el, t_mt, t_mt > 0 ? t_el / t_mt : 0.);
    arr.clear();
}

int run(int argc, char** argv)
{
    __mem_page_size = getpagesize();
    const long N    = argc > 1 ? shrink_array::arg_number<long>(argv[1], (1L << 20) + 777, "N") : (1L << 20) + 777;
    ParamChunk::init_indexes(true);

    if (argc > 2 && std::strcmp(argv[2], "perf") == 0)
    {
        run_perf(N);
        return 0;
    }
    const int rounds = argc > 2 ? shrink_array::arg_number<int>(argv[2], 30, "rounds") : 30;
    std::printf("=== fill_range_mt_validation [variant = %s]  N=%ld  threads=%d ===\n", variant_name(), N,
                CHUNK_PMP_MAX_THREADS);
    const int fails = run_validation(N, rounds);
    std::printf("[%s] === %s ===\n", variant_name(), fails ? "FILL_RANGE_MT FAILED" : "FILL_RANGE_MT OK");
    return fails;
}

} // namespace validation::fill_range_mt_validation
