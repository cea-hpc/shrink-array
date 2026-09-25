/*
 * Copyright 2024-2026 CEA (www.cea.fr)
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
// ArrayChunkBits (bools packed 8 per byte) against std::vector<bool>.
// ============================================================================
//
//   1. round trip: materialise, write runs of all-false / all-true / mixed
//      chunks, compress, rewrite a quarter, compare every element with a
//      std::vector<bool> that received the same writes;
//   2. the pass compresses: after it, only the mixed runs are still
//      materialised (a third of the chunks);
//   3. neighbouring elements written by different threads (schedule(static,1):
//      element i by thread i % nthr, so every word is written by all of them
//      at once) -- none lost. OpenMP builds; serial builds run it on one
//      thread;
//   4. the element surface: `bool b = a[i]`, `a[i] = !b`, `a[i] == b`;
//   5. |= &= ^= with a bool operand, element-wise against std::vector<bool>,
//      in parallel over neighbouring elements (schedule(static,1)).
//
// Chunk2 builds only (ArrayChunkBits is ArrayChunk2<Bit>). Exit 0 = pass.

#include <cstdio>
#include <unistd.h>
#include <vector>

#include "chunk/chunk_traits.h"
#include "chunk/array_chunk_bits.h"
#include "chunk/param_chunk.h"
#include "chunk/chunk_std.h" // release_chunk_pools, destroy_block_chunk
#include "chunk/config.h"
#include "chunk/PMP_CHUNK.h"
#include "chunk/manage_ompt.h" // begin_parallel_phase / finalize_parallel_phase
#include "chunk/internal/env_parse.h"

int __mem_page_size = 0;

using namespace shrink_array;

namespace
{

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

// A multiple of every chunk size the harness is built at: a run is whole chunks.
constexpr long kRun = 4096;

bool varied(long i)
{
    return (i % 3) == 0 || (i % 7) == 2;
}

long compare(const ArrayChunkBits<>& a, const std::vector<bool>& ref, const char* step)
{
    long bad = 0, first = -1;
    for (long i = 0; i < static_cast<long>(ref.size()); ++i)
    {
        const bool got = a[i];
        if (got != ref[static_cast<std::size_t>(i)])
        {
            if (first < 0)
                first = i;
            ++bad;
        }
    }
    if (bad)
        std::printf("[bits] FAIL %s: %ld mismatches, first at %ld (chunk %ld)\n", step, bad, first,
                    first / static_cast<long>(a.chunk_size()));
    return bad;
}

int round_trip(long N)
{
    ArrayChunkBits<>  a(N);
    std::vector<bool> ref(static_cast<std::size_t>(N), false);
    int               fails = 0;

#pragma omp parallel
    {
        begin_phase();
#pragma omp for
        for (long i = 0; i < N; ++i)
            a[i] = varied(i);
    }
    end_phase();
    for (long i = 0; i < N; ++i)
        ref[static_cast<std::size_t>(i)] = varied(i);
    fails += compare(a, ref, "materialise") != 0;

    // Runs: 0 -> all false, 1 -> all true, 2 -> left mixed.
#pragma omp parallel
    {
        begin_phase();
#pragma omp for
        for (long i = 0; i < N; ++i)
        {
            const long run = (i / kRun) % 3;
            if (run == 0)
                a[i] = false;
            else if (run == 1)
                a[i] = true;
        }
    }
    end_phase();
    for (long i = 0; i < N; ++i)
    {
        const long run = (i / kRun) % 3;
        if (run != 2)
            ref[static_cast<std::size_t>(i)] = (run == 1);
    }
    fails += compare(a, ref, "constant runs") != 0;

#pragma omp parallel
    {
        begin_phase();
#pragma omp single
        a.compress();
    }
    end_phase();
    fails += compare(a, ref, "after compress") != 0;

    // Only the mixed runs (run % 3 == 2) stay materialised; runs are whole
    // chunks, so the share of written chunks is that of the mixed runs.
    const double written = a.stats_seq();
    const long   n_runs  = N / kRun;
    const double expect  = 100.0 * static_cast<double>(n_runs / 3) / static_cast<double>(n_runs);
    if (written > expect + 1.0 || written < expect - 1.0)
    {
        std::printf("[bits] FAIL compression: %.1f%% of the chunks written, expected ~%.1f%%\n", written, expect);
        ++fails;
    }

    // Rewrite the first quarter (re-materialises constant chunks).
#pragma omp parallel
    {
        begin_phase();
#pragma omp for
        for (long i = 0; i < N / 4; ++i)
            a[i] = (i % 5) == 1;
    }
    end_phase();
    for (long i = 0; i < N / 4; ++i)
        ref[static_cast<std::size_t>(i)] = (i % 5) == 1;
    fails += compare(a, ref, "after refill") != 0;

    if (fails == 0)
        std::printf("[bits] OK round trip: %ld elements, chunk_size=%u, %.1f%% of chunks written after the pass\n",
                    N, a.chunk_size(), written);
    a.clear();
    return fails;
}

int neighbours(long N)
{
    ArrayChunkBits<> a(N);
    int              fails = 0;
    // Two passes, set then clear half: each word written by every thread in both.
#pragma omp parallel
    {
        begin_phase();
#pragma omp for schedule(static, 1)
        for (long i = 0; i < N; ++i)
            a[i] = true;
    }
    end_phase();
#pragma omp parallel
    {
        begin_phase();
#pragma omp for schedule(static, 1)
        for (long i = 0; i < N; ++i)
            if ((i / 2) % 2 == 1)
                a[i] = false;
    }
    end_phase();

    std::vector<bool> ref(static_cast<std::size_t>(N));
    for (long i = 0; i < N; ++i)
        ref[static_cast<std::size_t>(i)] = (i / 2) % 2 == 0;
    fails += compare(a, ref, "neighbouring writes") != 0;
    if (fails == 0)
        std::printf("[bits] OK neighbouring elements written by %d threads: none lost\n", CHUNK_PMP_MAX_THREADS);
    a.clear();
    return fails;
}

int surface()
{
    ArrayChunkBits<> a(1000);
    int              fails = 0;
    a[3]                   = true;
    const bool b3          = a[3];
    const bool b4          = a[4];
    a[4]                   = !b4;
    fails += !(b3 && !b4 && a[4] == true && a[3] == b3 && a[5] == false);
    if (fails)
        std::printf("[bits] FAIL element surface\n");
    a.clear();
    return fails;
}

int logical_ops(long N)
{
    ArrayChunkBits<>  a(N);
    std::vector<bool> ref(static_cast<std::size_t>(N), false);
    const auto        opnd = [](long i, int k) { return ((i * 7 + k * 3) % 5) < 2; };
    int               fails = 0;

    // Start from a pattern with constant runs, so |= / &= also meet compressed chunks.
#pragma omp parallel
    {
        begin_phase();
#pragma omp for
        for (long i = 0; i < N; ++i)
            a[i] = ((i / kRun) % 3 == 1) || ((i / kRun) % 3 == 2 && i % 2 == 0);
    }
    end_phase();
    for (long i = 0; i < N; ++i)
        ref[static_cast<std::size_t>(i)] = ((i / kRun) % 3 == 1) || ((i / kRun) % 3 == 2 && i % 2 == 0);

    for (int k = 0; k < 3; ++k)
    {
#pragma omp parallel
        {
            begin_phase();
#pragma omp for schedule(static, 1)
            for (long i = 0; i < N; ++i)
            {
                const bool b = opnd(i, k);
                if (k == 0)
                    a[i] |= b;
                else if (k == 1)
                    a[i] &= b;
                else
                    a[i] ^= b;
            }
        }
        end_phase();
        for (long i = 0; i < N; ++i)
        {
            const auto j = static_cast<std::size_t>(i);
            const bool b = opnd(i, k);
            ref[j]       = k == 0 ? (ref[j] || b) : k == 1 ? (ref[j] && b) : (ref[j] != b);
        }
        static const char* const names[] = {"|=", "&=", "^="};
        fails += compare(a, ref, names[k]) != 0;
    }

    // A proxy of the same container as the operand.
    ArrayChunkBits<> m(8);
    m[0] = true;
    a[0] = false;
    a[0] |= m[0];
    a[1] = true;
    a[1] &= m[1];
    fails += !(a[0] == true && a[1] == false);

    if (fails == 0)
        std::printf("[bits] OK |= &= ^= against std::vector<bool>, neighbouring elements in parallel\n");
    m.clear();
    a.clear();
    return fails;
}

} // namespace

int main(int argc, char** argv)
{
    __mem_page_size = getpagesize();
    const long N    = argc > 1 ? shrink_array::arg_number<long>(argv[1], 1L << 20, "N") : (1L << 20);
    ParamChunk::init_indexes(true);

    std::printf("=== bits_container_validation  N=%ld  threads=%d ===\n", N, CHUNK_PMP_MAX_THREADS);
    int fails = 0;
    fails += surface();
    fails += round_trip(N);
    fails += neighbours(N);
    fails += logical_ops(N);

    release_chunk_pools();
    destroy_block_chunk();

    std::printf("[bits] %s\n", fails == 0 ? "=== ALL CHECKS PASSED ===" : "=== FAILURES ===");
    return fails;
}
