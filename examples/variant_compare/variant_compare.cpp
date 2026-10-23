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
 *          One Array<double> of n_elems elements (n_elems rounded up to whole chunks),
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
 *          end (twice the memory: small n_elems only).
 *
 *          Printed: time per pass (summed over the iterations) and in total,
 *          the chunks compressed at the end, the exact resident size
 *          (smaps_rollup: VmRSS trails reality at 64 threads) at the end
 *          and its peak over the iteration ends, the checksum.
 *          Compare two runs at the same n_elems, iterations, K and thread count;
 *          at --constants 1 the compressed fraction must agree (to 0.1 point)
 *          for the times to mean anything.
 *
 *          Usage: variant_compare [--n n_elems] [--iters I] [--constants K] [--verify]
 */

#include <chrono>      // std::chrono::steady_clock
#include <cstdint>     // std::uint64_t
#include <cstdio>      // std::fprintf, stderr
#include <string>      // std::string
#include <type_traits> // std::conditional_t
#include <utility>     // std::as_const
#include <vector>      // std::vector

#include "chunk/chunk_traits.h"       // the IntType = int default of zero_one_value::Array<T> / zero_any::Array<T>
#include "chunk/zero_one_value_array.h"       // zero_one_value::Array
#include "chunk/zero_any_array.h"      // zero_any::Array
#include "chunk/param_chunk.h"        // ParamChunk::init_indexes
#include "chunk/config.h"             // chunk_config::use_chunk2a, build_config::use_openmp
#include "chunk/PMP_CHUNK.h"          // CHUNK_PMP_MAX_THREADS
#include "chunk/manage_ompt.h"        // shrink_array::begin_parallel_phase, finalize_parallel_phase
#include "chunk/internal/env_parse.h" // arg_number (checked, not atoi)
#include "chunk/internal/exact_rss.h" // exact_rss_kib (smaps_rollup)
#include "chunk/internal/print.h"     // shrink_array::println


using namespace shrink_array;

static_assert(chunk_config::use_chunk2or2a,
              "variant_compare compares chunk2 and chunk2a: link chunk_v2_* or chunk_v2a_*");

template <typename T>
using Array = std::conditional_t<chunk_config::use_chunk2a, zero_any::Array<T>, zero_one_value::Array<T>>;

constexpr const char* variant_name()
{
    return chunk_config::use_chunk2a ? "chunk2a" : "chunk2";
}

/// The first value is 0; the second values, the first of them declared.
constexpr double second_value[] = {9., 11., 13., 17., 19., 23., 29., 31.};
constexpr int    max_constants  = static_cast<int>(sizeof(second_value) / sizeof(second_value[0]));

// ── Phase brackets: no-ops under OMPT (the callbacks do it) ─────────────────
static void begin_phase()
{
    if constexpr (build_config::has_not_ompt)
    {
        shrink_array::begin_parallel_phase();
    }
}
static void end_phase(Array<double>& arr)
{
    if constexpr (build_config::has_not_ompt)
    {
        shrink_array::finalize_parallel_phase();
    }
    // Serial chunk2a has no between-region pass: run it (mark_path_validation).
    if constexpr (chunk_config::use_chunk2a && !build_config::use_openmp)
    {
        arr.make_block_compression();
    }
    else
    {
        (void)arr;
    }
}

/// Chunk2 declares its one second value; Chunk2A finds one per chunk (no
/// such member). A template, so the discarded call is not checked.
template <typename Arr>
static void maybe_declare_second_value(Arr& a, double v)
{
    if constexpr (requires { a.set_second_value(v, false); })
    {
        a.set_second_value(v, false);
    }
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
    {
        return {0, 0.};
    }
    if (r <= 4)
    {
        return {0, second_value[(h >> 8) % static_cast<std::uint64_t>(nconst)]};
    }
    if (r == 5)
    {
        return {1, 0.};
    }
    return {2, 0.};
}

/// Non-uniform within any chunk of 64 elements or more (period 13).
static double plain(long i, int iter)
{
    return static_cast<double>((i * 7 + iter) % 13) * 0.5;
}

int main(int argc, char** argv)
{
    ParamChunk::init_indexes(true);

    long n_elems = 1L << 28;
    int  iters   = 20;
    int  nconst  = 1;
    bool verify  = false;
    for (int a = 1; a < argc; ++a)
    {
        const std::string o = argv[a];
        if (o == "--n" && a + 1 < argc)
        {
            n_elems = shrink_array::arg_number<long>(argv[++a], n_elems, "--n");
        }
        else if (o == "--iters" && a + 1 < argc)
        {
            iters = shrink_array::arg_number<int>(argv[++a], iters, "--iters");
        }
        else if (o == "--constants" && a + 1 < argc)
        {
            nconst = shrink_array::arg_number<int>(argv[++a], nconst, "--constants");
        }
        else if (o == "--verify")
        {
            verify = true;
        }
        else
        {
            std::fprintf(stderr, "usage: %s [--n N] [--iters I] [--constants K (1..%d)] [--verify]\n", argv[0],
                         max_constants);
            return 2;
        }
    }
    if (nconst < 1 || nconst > max_constants || iters < 1 || n_elems < 1 || n_elems > 0x7fffffffL)
    {
        std::fprintf(stderr, "variant_compare: need 1 <= constants <= %d, iters >= 1, 1 <= n < 2^31\n", max_constants);
        return 2;
    }

    // Read off an EMPTY array: no chunk, no end-of-run stats line of its own.
    const long s       = static_cast<long>(Array<double>(0).chunk_size());
    n_elems            = (n_elems + s - 1) / s * s;
    const long nchunks = n_elems / s;

    println("=== variant_compare [{}]  n={}  chunk_size={}  chunks={}  iters={}  constants={}  threads={}{} ===",
            variant_name(), n_elems, s, nchunks, iters, nconst, CHUNK_PMP_MAX_THREADS, verify ? "  (verify)" : "");

    Array<double> a(static_cast<int>(n_elems));
    maybe_declare_second_value(a, second_value[0]);
    std::vector<double> ref;
    if (verify)
    {
        ref.assign(static_cast<std::size_t>(n_elems), 0.);
    }

    using Clk       = std::chrono::steady_clock;
    const auto secs = [](Clk::time_point t0) { return std::chrono::duration<double>(Clk::now() - t0).count(); };

    double    t_const  = 0.;
    double    t_plain  = 0.;
    double    t_read   = 0.;
    double    t_rmw    = 0.;
    double    checksum = 0.;
    const int n        = static_cast<int>(n_elems);

    // Exact RSS at each iteration end, outside the timed passes (a read
    // walks the page tables). A peak inside a pass is not seen.
    long rss_peak_kib = 0;
    for (int it = 0; it < iters; ++it)
    {
        // const: the constant chunks
        auto t0 = Clk::now();
#pragma omp parallel
        {
            begin_phase();
#pragma omp for schedule(static)
            for (int i = 0; i < n; ++i)
            {
                const Role r = role(i / s, it, nconst);
                if (r.kind == 0)
                {
                    a[i] = r.value;
                }
            }
        }
        end_phase(a);
        t_const += secs(t0);

        // plain: the non-uniform chunks
        t0 = Clk::now();
#pragma omp parallel
        {
            begin_phase();
#pragma omp for schedule(static)
            for (int i = 0; i < n; ++i)
            {
                if (role(i / s, it, nconst).kind == 1)
                {
                    a[i] = plain(i, it);
                }
            }
        }
        end_phase(a);
        t_plain += secs(t0);

        // read: everything
        t0              = Clk::now();
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
        t_read += secs(t0);
        checksum += sum;

        // rmw: the non-uniform chunks again
        t0 = Clk::now();
#pragma omp parallel
        {
            begin_phase();
#pragma omp for schedule(static)
            for (int i = 0; i < n; ++i)
            {
                if (role(i / s, it, nconst).kind == 1)
                {
                    a[i] += 1.;
                }
            }
        }
        end_phase(a);
        t_rmw += secs(t0);

        const long rss_it = shrink_array::exact_rss_kib();
        rss_peak_kib      = rss_it > rss_peak_kib ? rss_it : rss_peak_kib;

        if (verify)
        {
            for (long i = 0; i < n_elems; ++i)
            {
                const Role r = role(i / s, it, nconst);
                if (r.kind == 0)
                {
                    ref[static_cast<std::size_t>(i)] = r.value;
                }
                else if (r.kind == 1)
                {
                    ref[static_cast<std::size_t>(i)] = plain(i, it) + 1.;
                }
            }
        }
    }

    long compressed = 0;
    for (long j = 0; j < nchunks; ++j)
    {
        compressed += a.written(static_cast<int>(j * s)) ? 0 : 1;
    }

    const double total = t_const + t_plain + t_read + t_rmw;
    println("time (s)    const {:8.3f} | plain {:8.3f} | read {:8.3f} | rmw {:8.3f} | total {:8.3f}", t_const, t_plain,
            t_read, t_rmw, total);
    println("compressed  {} / {} chunks ({:.2f} %)", compressed, nchunks,
            100. * static_cast<double>(compressed) / static_cast<double>(nchunks));
    println("memory      RSS {} MiB at the end | peak {} MiB over the iteration ends (exact)",
            shrink_array::exact_rss_kib() >> 10, rss_peak_kib >> 10);
    println("checksum    {:.1f}", checksum);

    int bad = 0;
    if (verify)
    {
        long nbad = 0;
        for (long i = 0; i < n_elems; ++i)
        {
            if (std::as_const(a)[static_cast<int>(i)] != ref[static_cast<std::size_t>(i)])
            {
                if (nbad < 3)
                {
                    println("  MISMATCH a[{}] = {:g}, expected {:g}", i,
                            static_cast<double>(std::as_const(a)[static_cast<int>(i)]),
                            ref[static_cast<std::size_t>(i)]);
                }
                ++nbad;
            }
        }
        println("verify      {} ({} mismatch(es))", nbad ? "FAILED" : "OK", nbad);
        bad = nbad ? 1 : 0;
    }
    a.clear();
    return bad;
}
