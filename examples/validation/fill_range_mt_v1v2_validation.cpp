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
 * @file fill_range_mt_v1v2_validation.cpp
 * @brief fill_range_mt() in a build that holds TWO variants (chunk_v1v2:
 *        CHUNK=1;CHUNK2=1), on an ArrayChunk and an ArrayChunk2 side by side.
 *
 * @details The range ops once chose their per-variant behaviour from the build
 *          flags (chunk_config::use_chunk / use_chunk2) instead of the array's
 *          type. In a build with both flags set, that sent an ArrayChunk2 down
 *          v1's paths -- a fill_range_mt() of its SECOND value was never queued
 *          for compression, its chunks stayed materialised -- and an ArrayChunk
 *          down Chunk2's, which did not even compile. This harness is that
 *          build: compiling it is the first half of the test, the second is
 *          what each fill leaves behind.
 *
 *          Every chunk is first materialised with non-uniform content, then
 *          refilled from every thread at once by fill_range_mt(), in two
 *          half-chunk spans that different threads take -- so each call covers
 *          only part of a materialised chunk and the raw session path decides
 *          the queueing by value. After the phase (flush + compression pass):
 *
 *            array         fill value              chunks left materialised
 *            ArrayChunk    first value (0)         none
 *            ArrayChunk    anything else           all  (v1 has one constant)
 *            ArrayChunk2   first value (0)         none
 *            ArrayChunk2   its second value (9)    none  <- the regression
 *            ArrayChunk2   an undeclared value     all
 *
 *          and every element reads back the value written. The counts are
 *          asserted in OpenMP builds only: a serial build has no fill session
 *          (fill_range_mt is the element loop there), so only the values are.
 *
 *          The library variant is built without the OMPT tool (NO_OMPT_TOOL),
 *          as multi-mat uses it: this program defines the switch the library
 *          reads, and makes the phase calls itself.
 *
 *          Usage: fill_range_mt_v1v2_validation [chunks]
 */

#include <cstdio>
#include <unistd.h>

#include "chunk/array_chunk.h"
#include "chunk/array_chunk2.h"
#include "chunk/array_chunk_range_ops.h"
#include "chunk/param_chunk.h"
#include "chunk/config.h"
#include "chunk/PMP_CHUNK.h"
#include "chunk/manage_ompt.h"
#include "chunk/internal/env_parse.h" // arg_number (checked, not atoi)

int __mem_page_size = 0;

/// The suspension counter read by chunk_config::OMPTcc: non-zero, no OMPT
/// callback exists in this binary (chunk_v1v2 is built without the tool).
OmptSuspendCounter g_ompt_suspended{1};

using namespace shrink_array;

static_assert(chunk_config::use_chunk && chunk_config::use_chunk2,
              "fill_range_mt_v1v2_validation is the two-variant build: link chunk_v1v2_<size>");

constexpr double kSecond = 9.0; ///< the ArrayChunk2's declared second value
constexpr double kOther  = 5.0; ///< a value neither array can compress to

/// @brief Non-uniform content: two values alternating, so no chunk compresses.
static double plain(long i)
{
    return (i % 2) ? 2.5 : 3.5;
}

/// @brief Chunks of @p arr still holding a buffer.
template <typename Arr>
static long materialised(const Arr& arr, long nchunks)
{
    const long S = static_cast<long>(arr.chunk_size());
    long       n = 0;
    for (long j = 0; j < nchunks; ++j)
        n += arr.written(static_cast<typename Arr::ind_type>(j * S)) ? 1 : 0;
    return n;
}

/// @brief One parallel phase running @p body over [0, n), then the flush and
///        the compression pass -- the manual phase calls of manage_ompt.h.
template <typename Fn>
static void phase(long n, Fn&& body)
{
#pragma omp parallel
    {
        shrink_array::begin_parallel_phase();
#pragma omp for schedule(static, 1)
        for (long k = 0; k < n; ++k)
            body(k);
        shrink_array::finalize_parallel_phase_only_flush();
    }
    shrink_array::finalize_parallel_phase_after_flush();
}

/// @brief Materialise every chunk of @p arr, refill it with @p v by half-chunk
///        fill_range_mt() spans, and check the values and the chunks left
///        materialised against @p expect_materialised (-1: not asserted).
template <typename Arr>
static int run_case(const char* name, Arr& arr, long nchunks, double v, long expect_materialised)
{
    const long S    = static_cast<long>(arr.chunk_size());
    const long N    = nchunks * S;
    const long half = S / 2;

    phase(N, [&](long i) { arr[static_cast<typename Arr::ind_type>(i)] = plain(i); });
    const long before = materialised(arr, nchunks);

    // Two spans per chunk, adjacent iterations: with schedule(static, 1) they
    // go to different threads, so each call covers half a chunk that another
    // thread is filling at the same time.
    phase(2 * nchunks, [&](long k) {
        const long lo = (k / 2) * S + (k % 2) * half;
        const long hi = (k % 2) ? (k / 2 + 1) * S : lo + half;
        fill_range_mt(arr, static_cast<std::size_t>(lo), static_cast<std::size_t>(hi), v);
    });

    long bad = 0;
    for (long i = 0; i < N; ++i)
    {
        const double got = arr[static_cast<typename Arr::ind_type>(i)];
        if (got != v)
        {
            if (bad < 3)
                std::printf("  [%s] arr[%ld] = %g, expected %g\n", name, i, got, v);
            ++bad;
        }
    }
    const long after = materialised(arr, nchunks);

    const bool counts_ok = expect_materialised < 0 || after == expect_materialised;
    std::printf("%-44s materialised %4ld -> %4ld / %ld%s%s\n", name, before, after, nchunks,
                expect_materialised < 0 ? "  (not asserted)" : "",
                counts_ok ? "" : "  <-- WRONG");
    if (before != nchunks)
        std::printf("  [%s] setup: %ld chunk(s) not materialised by the plain content\n", name, nchunks - before);
    return (bad != 0 || !counts_ok || before != nchunks) ? 1 : 0;
}

int main(int argc, char** argv)
{
    __mem_page_size = getpagesize();
    const long nchunks = argc > 1 ? shrink_array::arg_number<long>(argv[1], 64, "chunks") : 64;
    ParamChunk::init_indexes(true);

    constexpr bool omp = build_config::use_openmp;
    std::printf("=== fill_range_mt_v1v2_validation  chunks=%ld  threads=%d  (%s) ===\n", nchunks,
                CHUNK_PMP_MAX_THREADS, omp ? "OpenMP: counts asserted" : "serial: values only");

    // One array per case: every case starts from fresh chunks.
    int fails = 0;
    {
        ArrayChunk<double> a(0);
        a.resize(static_cast<ArrayChunk<double>::ind_type>(nchunks * a.chunk_size()));
        fails += run_case("ArrayChunk,  fill first value (0)", a, nchunks, 0.0, omp ? 0 : -1);
        a.clear();
    }
    {
        ArrayChunk<double> a(0);
        a.resize(static_cast<ArrayChunk<double>::ind_type>(nchunks * a.chunk_size()));
        fails += run_case("ArrayChunk,  fill other value (5)", a, nchunks, kOther, omp ? nchunks : -1);
        a.clear();
    }
    {
        ArrayChunk2<double> a(0);
        a.resize(static_cast<ArrayChunk2<double>::ind_type>(nchunks * a.chunk_size()));
        a.set_second_value(kSecond, false);
        fails += run_case("ArrayChunk2, fill first value (0)", a, nchunks, 0.0, omp ? 0 : -1);
        a.clear();
    }
    {
        ArrayChunk2<double> a(0);
        a.resize(static_cast<ArrayChunk2<double>::ind_type>(nchunks * a.chunk_size()));
        a.set_second_value(kSecond, false);
        fails += run_case("ArrayChunk2, fill second value (9)", a, nchunks, kSecond, omp ? 0 : -1);
        a.clear();
    }
    {
        ArrayChunk2<double> a(0);
        a.resize(static_cast<ArrayChunk2<double>::ind_type>(nchunks * a.chunk_size()));
        a.set_second_value(kSecond, false);
        fails += run_case("ArrayChunk2, fill undeclared value (5)", a, nchunks, kOther, omp ? nchunks : -1);
        a.clear();
    }

    std::printf("=== %s ===\n", fails ? "FILL_RANGE_MT_V1V2 FAILED" : "FILL_RANGE_MT_V1V2 OK");
    return fails ? 1 : 0;
}
