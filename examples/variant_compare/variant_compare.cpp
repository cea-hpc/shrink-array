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
 * @file variant_compare.cpp
 * @brief Chunk2 against Chunk2A on the SAME workload, where both can compress
 *        the same chunks (--constants 1) or where Chunk2A can compress more
 *        (--constants K > 1).
 *
 * @details shrink_bench writes several distinct constants per array (d0, d1,
 *          d2...): Chunk2A, which takes a second value per chunk, compresses
 *          ~8.5 % more chunks there than Chunk2, which has one per array, and
 *          then carries 2-2.5x less resident memory. That hides the question
 *          of the per-operation cost of each variant. This program asks it on
 *          its own:
 *
 *            --constants 1   every constant chunk is at 0 or at the array's
 *                            declared second value (9): both variants
 *                            compress the same chunks, the times compare the
 *                            write / read / RMW paths alone;
 *            --constants K   constant chunks take 0 or one of K values
 *                            (9, 11, 13, ...), only the first declared:
 *                            shrink_bench's situation, Chunk2A's advantage.
 *
 *          One Array<double> of N elements (N rounded up to whole chunks),
 *          all at 0 at the start. Every iteration gives each chunk a role
 *          from a hash of (chunk, iteration), so chunks keep changing state
 *          (materialise, compress, rewrite) from one iteration to the next:
 *
 *            roles 0-1  constant 0 (the first value)       20 %
 *            roles 2-4  constant, second value(s)          30 %
 *            role  5    non-uniform content                10 %
 *            roles 6-9  left alone this iteration          40 %
 *
 *          and four passes, each its own parallel region (so each ends with
 *          the between-region compression pass, which is timed with it):
 *
 *            const   the constant chunks, element by element (operator[])
 *            plain   the non-uniform chunks
 *            read    every element, summed (const operator[])
 *            rmw     the non-uniform chunks, a[i] += 1 (stay non-uniform)
 *
 *          Every value is a multiple of 0.5 well under 2^52, so the sums are
 *          exact in double whatever the reduction order: the checksum is the
 *          same for every variant and thread count, and must be. --verify
 *          also keeps a std::vector mirror and compares every element at the
 *          end (twice the memory: small N only).
 *
 *          Printed: time per pass (summed over the iterations) and in total,
 *          the chunks compressed at the end, the exact resident size
 *          (smaps_rollup: VmRSS trails reality at 64 threads) at the end
 *          and its peak over the iteration ends, the checksum.
 *          Compare two runs at the same N, iterations, K and thread count;
 *          at --constants 1 the compressed fraction must agree (to 0.1 point)
 *          for the times to mean anything.
 *
 *          Usage: variant_compare [--n N] [--iters I] [--constants K] [--verify]
 */

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <type_traits>
#include <utility> // std::as_const
#include <vector>
#include <unistd.h>

#include "chunk/chunk_traits.h"
#include "chunk/array_chunk.h"
#include "chunk/array_chunk2.h"
#include "chunk/array_chunk2A.h"
#include "chunk/param_chunk.h"
#include "chunk/config.h"
#include "chunk/PMP_CHUNK.h"
#include "chunk/manage_ompt.h"
#include "chunk/internal/env_parse.h" // arg_number (checked, not atoi)
#include "chunk/internal/exact_rss.h" // exact_rss_kib (smaps_rollup)

int __mem_page_size = 0;

using namespace shrink_array;

static_assert(chunk_config::use_chunk2or2a, "variant_compare compares chunk2 and chunk2a: link chunk_v2_* or chunk_v2a_*");

template <typename T>
using Array = std::conditional_t<chunk_config::use_chunk2a, ArrayChunk2A<T>, ArrayChunk2<T>>;

constexpr const char* variant_name()
{
    return chunk_config::use_chunk2a ? "chunk2a" : "chunk2";
}

/// The first value is 0; the second values, the first of them declared.
constexpr double kSecond[] = {9., 11., 13., 17., 19., 23., 29., 31.};
constexpr int    kMaxConstants = static_cast<int>(sizeof(kSecond) / sizeof(kSecond[0]));

// ── Phase brackets: no-ops under OMPT (the callbacks do it) ─────────────────
static void begin_phase()
{
    if constexpr (build_config::has_not_ompt)
        shrink_array::begin_parallel_phase();
}
static void end_phase(Array<double>& arr)
{
    if constexpr (build_config::has_not_ompt)
        shrink_array::finalize_parallel_phase();
    // Serial chunk2a has no between-region pass: run it (mark_path_validation).
    if constexpr (chunk_config::use_chunk2a && !build_config::use_openmp)
        arr.make_block_compression();
    else
        (void)arr;
}

/// Chunk2 declares its one second value; Chunk2A finds one per chunk (no
/// such member). A template, so the discarded call is not checked.
template <typename Arr>
static void maybe_declare_second_value(Arr& a, double v)
{
    if constexpr (requires { a.set_second_value(v, false); })
        a.set_second_value(v, false);
}

// ── The workload ─────────────────────────────────────────────────────────────
struct Role
{
    int    kind;  ///< 0: constant, 1: non-uniform, 2: untouched
    double value; ///< the constant, for kind 0
};

static Role role(long chunk, int iter, int nconst)
{
    std::uint64_t h = (static_cast<std::uint64_t>(chunk) + 1) * 0x9E3779B97F4A7C15ULL;
    h ^= static_cast<std::uint64_t>(iter + 1) * 0xBF58476D1CE4E5B9ULL;
    h ^= h >> 31;
    h *= 0x94D049BB133111EBULL;
    h ^= h >> 29;
    const int r = static_cast<int>(h % 10);
    if (r <= 1)
        return {0, 0.};
    if (r <= 4)
        return {0, kSecond[(h >> 8) % static_cast<std::uint64_t>(nconst)]};
    if (r == 5)
        return {1, 0.};
    return {2, 0.};
}

/// Non-uniform within any chunk of 64 elements or more (period 13).
static double plain(long i, int iter)
{
    return static_cast<double>((i * 7 + iter) % 13) * 0.5;
}

int main(int argc, char** argv)
{
    __mem_page_size = getpagesize();
    ParamChunk::init_indexes(true);

    long N      = 1L << 28;
    int  iters  = 20;
    int  nconst = 1;
    bool verify = false;
    for (int a = 1; a < argc; ++a)
    {
        const std::string o = argv[a];
        if (o == "--n" && a + 1 < argc)
            N = shrink_array::arg_number<long>(argv[++a], N, "--n");
        else if (o == "--iters" && a + 1 < argc)
            iters = shrink_array::arg_number<int>(argv[++a], iters, "--iters");
        else if (o == "--constants" && a + 1 < argc)
            nconst = shrink_array::arg_number<int>(argv[++a], nconst, "--constants");
        else if (o == "--verify")
            verify = true;
        else
        {
            std::fprintf(stderr, "usage: %s [--n N] [--iters I] [--constants K (1..%d)] [--verify]\n", argv[0],
                         kMaxConstants);
            return 2;
        }
    }
    if (nconst < 1 || nconst > kMaxConstants || iters < 1 || N < 1 || N > 0x7fffffffL)
    {
        std::fprintf(stderr, "variant_compare: need 1 <= constants <= %d, iters >= 1, 1 <= n < 2^31\n", kMaxConstants);
        return 2;
    }

    // Read off an EMPTY array: no chunk, no end-of-run stats line of its own.
    const long S = static_cast<long>(Array<double>(0).chunk_size());
    N            = (N + S - 1) / S * S;
    const long nchunks = N / S;

    std::printf("=== variant_compare [%s]  n=%ld  chunk_size=%ld  chunks=%ld  iters=%d  constants=%d  threads=%d%s ===\n",
                variant_name(), N, S, nchunks, iters, nconst, CHUNK_PMP_MAX_THREADS, verify ? "  (verify)" : "");

    Array<double> a(static_cast<int>(N));
    maybe_declare_second_value(a, kSecond[0]);
    std::vector<double> ref;
    if (verify)
        ref.assign(static_cast<std::size_t>(N), 0.);

    using clk = std::chrono::steady_clock;
    const auto secs = [](clk::time_point t0) { return std::chrono::duration<double>(clk::now() - t0).count(); };

    double t_const  = 0.;
    double t_plain  = 0.;
    double t_read   = 0.;
    double t_rmw    = 0.;
    double checksum = 0.;
    const int n = static_cast<int>(N);

    // Exact RSS at each iteration end, outside the timed passes (a read
    // walks the page tables). A peak inside a pass is not seen.
    long rss_peak_kib = 0;
    for (int it = 0; it < iters; ++it)
    {
        // const: the constant chunks
        auto t0 = clk::now();
#pragma omp parallel
        {
            begin_phase();
#pragma omp for schedule(static)
            for (int i = 0; i < n; ++i)
            {
                const Role r = role(i / S, it, nconst);
                if (r.kind == 0)
                    a[i] = r.value;
            }
        }
        end_phase(a);
        t_const += secs(t0);

        // plain: the non-uniform chunks
        t0 = clk::now();
#pragma omp parallel
        {
            begin_phase();
#pragma omp for schedule(static)
            for (int i = 0; i < n; ++i)
                if (role(i / S, it, nconst).kind == 1)
                    a[i] = plain(i, it);
        }
        end_phase(a);
        t_plain += secs(t0);

        // read: everything
        t0 = clk::now();
        double    sum = 0.;
        const auto& ca = std::as_const(a);
#pragma omp parallel
        {
            begin_phase();
#pragma omp for schedule(static) reduction(+ : sum)
            for (int i = 0; i < n; ++i)
                sum += ca[i];
        }
        end_phase(a);
        t_read += secs(t0);
        checksum += sum;

        // rmw: the non-uniform chunks again
        t0 = clk::now();
#pragma omp parallel
        {
            begin_phase();
#pragma omp for schedule(static)
            for (int i = 0; i < n; ++i)
                if (role(i / S, it, nconst).kind == 1)
                    a[i] += 1.;
        }
        end_phase(a);
        t_rmw += secs(t0);

        const long rss_it = shrink_array::exact_rss_kib();
        rss_peak_kib      = rss_it > rss_peak_kib ? rss_it : rss_peak_kib;

        if (verify)
            for (long i = 0; i < N; ++i)
            {
                const Role r = role(i / S, it, nconst);
                if (r.kind == 0)
                    ref[static_cast<std::size_t>(i)] = r.value;
                else if (r.kind == 1)
                    ref[static_cast<std::size_t>(i)] = plain(i, it) + 1.;
            }
    }

    long compressed = 0;
    for (long j = 0; j < nchunks; ++j)
        compressed += a.written(static_cast<int>(j * S)) ? 0 : 1;

    const double total = t_const + t_plain + t_read + t_rmw;
    std::printf("time (s)    const %8.3f | plain %8.3f | read %8.3f | rmw %8.3f | total %8.3f\n", t_const, t_plain,
                t_read, t_rmw, total);
    std::printf("compressed  %ld / %ld chunks (%.2f %%)\n", compressed, nchunks,
                100. * static_cast<double>(compressed) / static_cast<double>(nchunks));
    std::printf("memory      RSS %ld MiB at the end | peak %ld MiB over the iteration ends (exact)\n",
                shrink_array::exact_rss_kib() >> 10, rss_peak_kib >> 10);
    std::printf("checksum    %.1f\n", checksum);

    int bad = 0;
    if (verify)
    {
        long nbad = 0;
        for (long i = 0; i < N; ++i)
            if (std::as_const(a)[static_cast<int>(i)] != ref[static_cast<std::size_t>(i)])
            {
                if (nbad < 3)
                    std::printf("  MISMATCH a[%ld] = %g, expected %g\n", i,
                                static_cast<double>(std::as_const(a)[static_cast<int>(i)]),
                                ref[static_cast<std::size_t>(i)]);
                ++nbad;
            }
        std::printf("verify      %s (%ld mismatch(es))\n", nbad ? "FAILED" : "OK", nbad);
        bad = nbad ? 1 : 0;
    }
    a.clear();
    return bad;
}
