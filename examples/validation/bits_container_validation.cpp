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
//      in parallel over neighbouring elements (schedule(static,1));
//   6. the chunk-size advisor: the exact row (materialised count, memory at
//      S/8 bytes per materialised chunk) and the finer row S/2, whose
//      sub-blocks it classifies word by word -- chunks half all-true, half
//      mixed must read as one compressible half each. Also an
//      ArrayChunk2<bool> (it did not compile: no second-value index for bool);
//   7. re-chunk (dyn builds): down and up, contents against the reference;
//   8. fill_range / copy_range (bits stored 64 per word): random spans --
//      whole chunks, partial edges, across chunk boundaries -- over constant
//      and mixed chunks, against std::vector<bool>; then a pass compresses
//      the chunks a fill left constant;
//   9. fill_range_mt from every thread of a region, on spans of 1..97
//      elements handed out round-robin (neighbouring spans of one word belong
//      to different threads), twice over, against std::vector<bool>; then
//      copy_range_mt and transform_range_mt the same way, from a byte source
//      (they keep the span's values in a per-thread buffer: it did not
//      compile for bits, std::vector<bool> having no data());
//  10. birth size (dyn builds): born at packed_bit_birth_factor times the
//      active size (capped at cst_buffer_size), which it leaves unchanged --
//      an array of another type created next is born at it.
//
// Chunk2 builds only (ArrayChunkBits is ArrayChunk2<Bit>). Exit 0 = pass.

#include <algorithm> // std::min
#include <cstdint>   // std::uint64_t, std::uint32_t, std::uint8_t (the copy source)
#include <cstdio>    // std::snprintf
#include <random>    // std::mt19937_64
#include <utility>   // std::pair
#include <vector>    // std::vector

#include "chunk/chunk_traits.h"          // forward decls + default ind_type = int (ArrayChunk2<bool>)
#include "chunk/array_chunk_bits.h"      // ArrayChunkBits, Bit (and ArrayChunk2 with them)
#include "chunk/chunk_size_advisor.h"    // ChunkSizeAdvisor
#include "chunk/array_chunk_range_ops.h" // fill_range, copy_range, and their _mt forms
#include "chunk/param_chunk.h"           // ParamChunk::init_indexes
#include "chunk/chunk_std.h"             // release_chunk_pools, destroy_block_chunk
#include "chunk/config.h"                // build_config::chunk_dynamic, build_config::has_not_ompt
#include "chunk/PMP_CHUNK.h"             // CHUNK_PMP_MAX_THREADS
#include "chunk/manage_ompt.h"           // begin_parallel_phase / finalize_parallel_phase
#include "chunk/internal/env_parse.h"    // arg_number (checked, not atoi)
#include "chunk/internal/print.h"        // shrink_array::println

#include "validation_harness.h" // self_command; run() is called by validation_driver.cpp

namespace validation::bits_container_validation
{

using namespace shrink_array;

namespace
{

void begin_phase()
{
    if constexpr (build_config::has_not_ompt)
    {
        shrink_array::begin_parallel_phase();
    }
}
void end_phase()
{
    if constexpr (build_config::has_not_ompt)
    {
        shrink_array::finalize_parallel_phase();
    }
}

// A multiple of every chunk size the harness is built at: a run is whole chunks
// (4096 * 3: every power of two up to 4096, and 192, 384, 768...).
constexpr long run_len = 12288;

bool varied(long i)
{
    return (i % 3) == 0 || (i % 7) == 2;
}

long compare(const ArrayChunkBits<>& a, const std::vector<bool>& ref, const char* step)
{
    long bad   = 0;
    long first = -1;
    for (long i = 0; i < static_cast<long>(ref.size()); ++i)
    {
        const bool got = a[i];
        if (got != ref[static_cast<std::size_t>(i)])
        {
            if (first < 0)
            {
                first = i;
            }
            ++bad;
        }
    }
    if (bad)
    {
        println("[bits] FAIL {}: {} mismatches, first at {} (chunk {})", step, bad, first,
                first / static_cast<long>(a.chunk_size()));
    }
    return bad;
}

int round_trip(long n)
{
    ArrayChunkBits<>  a(n);
    std::vector<bool> ref(static_cast<std::size_t>(n), false);
    int               fails = 0;

#pragma omp parallel
    {
        begin_phase();
#pragma omp for
        for (long i = 0; i < n; ++i)
        {
            a[i] = varied(i);
        }
    }
    end_phase();
    for (long i = 0; i < n; ++i)
    {
        ref[static_cast<std::size_t>(i)] = varied(i);
    }
    fails += compare(a, ref, "materialise") != 0;

    // Runs: 0 -> all false, 1 -> all true, 2 -> left mixed.
#pragma omp parallel
    {
        begin_phase();
#pragma omp for
        for (long i = 0; i < n; ++i)
        {
            const long run = (i / run_len) % 3;
            if (run == 0)
            {
                a[i] = false;
            }
            else if (run == 1)
            {
                a[i] = true;
            }
        }
    }
    end_phase();
    for (long i = 0; i < n; ++i)
    {
        const long run = (i / run_len) % 3;
        if (run != 2)
        {
            ref[static_cast<std::size_t>(i)] = (run == 1);
        }
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
    const long   n_runs  = n / run_len;
    const double expect  = 100.0 * static_cast<double>(n_runs / 3) / static_cast<double>(n_runs);
    if (written > expect + 1.0 || written < expect - 1.0)
    {
        println("[bits] FAIL compression: {:.1f}% of the chunks written, expected ~{:.1f}%", written, expect);
        ++fails;
    }

    // Rewrite the first quarter (re-materialises constant chunks).
#pragma omp parallel
    {
        begin_phase();
#pragma omp for
        for (long i = 0; i < n / 4; ++i)
        {
            a[i] = (i % 5) == 1;
        }
    }
    end_phase();
    for (long i = 0; i < n / 4; ++i)
    {
        ref[static_cast<std::size_t>(i)] = (i % 5) == 1;
    }
    fails += compare(a, ref, "after refill") != 0;

    if (fails == 0)
    {
        println("[bits] OK round trip: {} elements, chunk_size={}, {:.1f}% of chunks written after the pass", n,
                a.chunk_size(), written);
    }
    a.clear();
    return fails;
}

int neighbours(long n)
{
    ArrayChunkBits<> a(n);
    int              fails = 0;
    // Two passes, set then clear half: each word written by every thread in both.
#pragma omp parallel
    {
        begin_phase();
#pragma omp for schedule(static, 1)
        for (long i = 0; i < n; ++i)
        {
            a[i] = true;
        }
    }
    end_phase();
#pragma omp parallel
    {
        begin_phase();
#pragma omp for schedule(static, 1)
        for (long i = 0; i < n; ++i)
        {
            if ((i / 2) % 2 == 1)
            {
                a[i] = false;
            }
        }
    }
    end_phase();

    std::vector<bool> ref(static_cast<std::size_t>(n));
    for (long i = 0; i < n; ++i)
    {
        ref[static_cast<std::size_t>(i)] = (i / 2) % 2 == 0;
    }
    fails += compare(a, ref, "neighbouring writes") != 0;
    if (fails == 0)
    {
        println("[bits] OK neighbouring elements written by {} threads: none lost", CHUNK_PMP_MAX_THREADS);
    }
    a.clear();
    return fails;
}

int birth_size()
{
    if constexpr (!build_config::chunk_dynamic)
    {
        return 0;
    }
    const unsigned int active = active_chunk_size();
    const unsigned int expect = std::min(active * packed_bit_birth_factor, cst_buffer_size);
    ArrayChunkBits<>   a(1L << 16);
    ArrayChunk2<bool>  b(1L << 16);
    int                fails = 0;
    if (a.chunk_size() != expect || active_chunk_size() != active || b.chunk_size() != active)
    {
        println("[bits] FAIL birth size: bits {} (expected {}), then active {} and bool {} (expected {})",
                a.chunk_size(), expect, active_chunk_size(), b.chunk_size(), active);
        ++fails;
    }
    else
    {
        println("[bits] OK birth size: bits at {}, the active size stays {}", a.chunk_size(), active);
    }
    a.clear();
    b.clear();
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
    {
        println("[bits] FAIL element surface");
    }
    a.clear();
    return fails;
}

int logical_ops(long n)
{
    ArrayChunkBits<>  a(n);
    std::vector<bool> ref(static_cast<std::size_t>(n), false);
    const auto        opnd  = [](long i, int k) { return (((i * 7) + (k * 3)) % 5) < 2; };
    int               fails = 0;

    // Start from a pattern with constant runs, so |= / &= also meet compressed chunks.
#pragma omp parallel
    {
        begin_phase();
#pragma omp for
        for (long i = 0; i < n; ++i)
        {
            a[i] = ((i / run_len) % 3 == 1) || ((i / run_len) % 3 == 2 && i % 2 == 0);
        }
    }
    end_phase();
    for (long i = 0; i < n; ++i)
    {
        ref[static_cast<std::size_t>(i)] = ((i / run_len) % 3 == 1) || ((i / run_len) % 3 == 2 && i % 2 == 0);
    }

    for (int k = 0; k < 3; ++k)
    {
#pragma omp parallel
        {
            begin_phase();
#pragma omp for schedule(static, 1)
            for (long i = 0; i < n; ++i)
            {
                const bool b = opnd(i, k);
                if (k == 0)
                {
                    a[i] |= b;
                }
                else if (k == 1)
                {
                    a[i] &= b;
                }
                else
                {
                    a[i] ^= b;
                }
            }
        }
        end_phase();
        for (long i = 0; i < n; ++i)
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
    {
        println("[bits] OK |= &= ^= against std::vector<bool>, neighbouring elements in parallel");
    }
    m.clear();
    a.clear();
    return fails;
}

int advisor_and_rechunk(long n)
{
    ArrayChunkBits<> a(n);
    int              fails = 0;
    if constexpr (build_config::chunk_dynamic)
    {
        a.rechunk_to(256); // room for a finer row (the advisor stops at 64)
    }
    const long s = static_cast<long>(a.chunk_size());

    // Runs of whole chunks: all false, all true, and chunks whose first half
    // is all true and second half mixed.
    const auto pattern = [s](long i)
    {
        const long run = (i / run_len) % 3;
        if (run != 2)
        {
            return run == 1;
        }
        return (i % s) < s / 2 ? true : varied(i);
    };
#pragma omp parallel
    {
        begin_phase();
#pragma omp for
        for (long i = 0; i < n; ++i)
        {
            a[i] = pattern(i);
        }
    }
    end_phase();
#pragma omp parallel
    {
        begin_phase();
#pragma omp single
        a.compress();
    }
    end_phase();
    std::vector<bool> ref(static_cast<std::size_t>(n));
    for (long i = 0; i < n; ++i)
    {
        ref[static_cast<std::size_t>(i)] = pattern(i);
    }
    fails += compare(a, ref, "advisor setup") != 0;

    const long mat_expected = (n / run_len) / 3 * (run_len / s);
    // The advisor does not evaluate below Options::min_size (64): at S=64 the
    // S/2 row exists only if asked for, so ask -- the finer path is what this
    // check is about. No change for S >= 128.
    typename ChunkSizeAdvisor<Bit>::Options opt;
    if (static_cast<long>(opt.min_size) > s / 2)
    {
        opt.min_size = static_cast<std::uint32_t>(s / 2);
    }
    const auto rep = ChunkSizeAdvisor<Bit>::analyze(a, opt);
    if (rep.cur < 0)
    {
        println("[bits] FAIL advisor: no row for the current size {}", s);
        return fails + 1;
    }
    const auto&  cur        = rep.table[static_cast<std::size_t>(rep.cur)];
    const double data_bytes = cur.mem_bytes - (static_cast<double>(cur.n_chunks) * rep.meta_bytes);
    if (cur.mat != mat_expected || data_bytes != static_cast<double>(mat_expected) * static_cast<double>(s) / 8.)
    {
        println("[bits] FAIL advisor exact row: mat {} (expected {}), data {:.0f} B (expected {:.0f})",
                static_cast<long long>(cur.mat), mat_expected, data_bytes,
                static_cast<double>(mat_expected) * static_cast<double>(s) / 8.);
        ++fails;
    }
    // At S/2 each such chunk is one compressible half (all true) and one mixed.
    bool seen_half = false;
    for (const auto& row : rep.table)
    {
        if (static_cast<long>(row.size) == s / 2)
        {
            seen_half      = true;
            const double r = static_cast<double>(row.mat) / static_cast<double>(mat_expected);
            if (r < 0.98 || r > 1.02)
            {
                println("[bits] FAIL advisor finer row S={}: mat {}, expected ~{}", s / 2,
                        static_cast<long long>(row.mat), mat_expected);
                ++fails;
            }
        }
    }
    // S / 2 is a candidate only when it is a chunk size (192 / 2 = 96 is not).
    if (!seen_half && is_valid_chunk_size(static_cast<unsigned int>(s / 2)))
    {
        println("[bits] FAIL advisor: no finer row at S={}", s / 2);
        ++fails;
    }

    // bool (one byte per element): the advisor analyses it too.
    {
        ArrayChunk2<bool> b(n);
        for (long i = 0; i < n; ++i)
        {
            b[i] = pattern(i);
        }
        const auto rb = ChunkSizeAdvisor<bool>::analyze(b);
        fails += rb.cur < 0;
        b.clear();
    }

    if constexpr (build_config::chunk_dynamic)
    {
        for (const unsigned to : {static_cast<unsigned>(s / 2), static_cast<unsigned>(s * 2), static_cast<unsigned>(s)})
        {
            a.rechunk_to(to);
            if (a.chunk_size() != to)
            {
                println("[bits] FAIL rechunk: geometry {}, expected {}", a.chunk_size(), to);
                ++fails;
            }
            char step[48];
            std::snprintf(step, sizeof step, "after rechunk_to(%u)", to);
            fails += compare(a, ref, step) != 0;
        }
    }

    if (fails == 0)
    {
        println("[bits] OK advisor (exact row, S/8 bytes, finer row by word){}",
                build_config::chunk_dynamic ? ", rechunk down and up" : "");
    }
    a.clear();
    return fails;
}

int bulk_ranges(long n)
{
    ArrayChunkBits<>  a(n);
    std::vector<bool> ref(static_cast<std::size_t>(n), false);
    const long        s     = static_cast<long>(a.chunk_size());
    int               fails = 0;

    // A start with constant and mixed chunks.
    for (long i = 0; i < n; ++i)
    {
        const bool v                     = ((i / run_len) % 3 == 1) || ((i / run_len) % 3 == 2 && varied(i));
        a[i]                             = v;
        ref[static_cast<std::size_t>(i)] = v;
    }

    std::mt19937_64   rng(4242);
    std::vector<bool> src(static_cast<std::size_t>(n));
    for (long i = 0; i < n; ++i)
    {
        src[static_cast<std::size_t>(i)] = ((i * 2654435761L) >> 7) % 3 == 0;
    }

    for (int round = 0; round < 400; ++round)
    {
        long first = static_cast<long>(rng() % static_cast<std::uint64_t>(n));
        long len   = static_cast<long>(rng() % static_cast<std::uint64_t>(4 * s)) + 1;
        if (round % 5 == 0) // whole chunks
        {
            first = first / s * s;
            len   = s * (1 + static_cast<long>(rng() % 3));
        }
        const long last = std::min(n, first + len);
        if (round % 2 == 0)
        {
            const bool v = (rng() & 1) != 0;
            fill_range(a, static_cast<std::size_t>(first), static_cast<std::size_t>(last), v);
            for (long i = first; i < last; ++i)
            {
                ref[static_cast<std::size_t>(i)] = v;
            }
        }
        else
        {
            copy_range(a, static_cast<std::size_t>(first), static_cast<std::size_t>(last), src.begin() + first);
            for (long i = first; i < last; ++i)
            {
                ref[static_cast<std::size_t>(i)] = src[static_cast<std::size_t>(i)];
            }
        }
    }
    fails += compare(a, ref, "fill_range / copy_range") != 0;

    // Whole chunks filled with each constant must compress in the next pass.
    const long c0 = (n / s) / 4;
    const long c1 = (n / s) / 2;
    fill_range(a, static_cast<std::size_t>(c0 * s), static_cast<std::size_t>((c0 + 3) * s), false);
    fill_range(a, static_cast<std::size_t>(c1 * s), static_cast<std::size_t>((c1 + 3) * s), true);
    for (long i = c0 * s; i < (c0 + 3) * s; ++i)
    {
        ref[static_cast<std::size_t>(i)] = false;
    }
    for (long i = c1 * s; i < (c1 + 3) * s; ++i)
    {
        ref[static_cast<std::size_t>(i)] = true;
    }
#pragma omp parallel
    {
        begin_phase();
#pragma omp single
        a.compress();
    }
    end_phase();
    fails += compare(a, ref, "after the pass") != 0;
    for (long c = 0; c < 3; ++c)
    {
        if (a.get_chunk_list()[static_cast<std::size_t>(c0 + c)].written() ||
            a.get_chunk_list()[static_cast<std::size_t>(c1 + c)].written())
        {
            println("[bits] FAIL fill_range: whole chunk {} / {} filled with a constant still materialised", c0 + c,
                    c1 + c);
            ++fails;
            break;
        }
    }

    if (fails == 0)
    {
        println("[bits] OK fill_range / copy_range (64 per word), constant chunks compressed after the pass");
    }
    a.clear();
    return fails;
}

int parallel_fill(long n)
{
    ArrayChunkBits<>  a(n);
    std::vector<bool> ref(static_cast<std::size_t>(n), false);
    int               fails = 0;

    std::vector<std::pair<long, long>> spans;
    for (long lo = 0, len = 1; lo < n; lo += len, len = (len % 97) + 1)
    {
        spans.emplace_back(lo, std::min(n, lo + len));
    }
    const auto nspans = static_cast<long>(spans.size());

    for (int pass = 0; pass < 2; ++pass)
    {
#pragma omp parallel
        {
            begin_phase();
#pragma omp for schedule(static, 1)
            for (long s = 0; s < nspans; ++s)
            {
                const bool v = pass == 0 ? (s % 3 != 0) : (s % 2 == 0);
                fill_range_mt(a, static_cast<std::size_t>(spans[static_cast<std::size_t>(s)].first),
                              static_cast<std::size_t>(spans[static_cast<std::size_t>(s)].second), v);
            }
        }
        end_phase();
        for (long s = 0; s < nspans; ++s)
        {
            const bool v = pass == 0 ? (s % 3 != 0) : (s % 2 == 0);
            for (long i = spans[static_cast<std::size_t>(s)].first; i < spans[static_cast<std::size_t>(s)].second; ++i)
            {
                ref[static_cast<std::size_t>(i)] = v;
            }
        }
        fails += compare(a, ref, pass == 0 ? "fill_range_mt pass 1" : "fill_range_mt pass 2") != 0;
    }
    if (fails == 0)
    {
        println("[bits] OK fill_range_mt from {} threads, neighbouring spans in one word", CHUNK_PMP_MAX_THREADS);
    }
    a.clear();
    return fails;
}

int parallel_copy_transform(long n)
{
    ArrayChunkBits<>  a(n);
    std::vector<bool> ref(static_cast<std::size_t>(n), false);
    int               fails = 0;

    // Runs of ones and zeros, and a mixed stretch: spans copied whole uniform
    // and spans copied mixed.
    std::vector<std::uint8_t> src(static_cast<std::size_t>(n));
    for (long i = 0; i < n; ++i)
    {
        src[static_cast<std::size_t>(i)] =
            (i / 300) % 3 == 2 ? static_cast<std::uint8_t>(i % 5 == 0) : static_cast<std::uint8_t>((i / 300) % 3 == 0);
    }

    std::vector<std::pair<long, long>> spans;
    for (long lo = 0, len = 1; lo < n; lo += len, len = (len % 97) + 1)
    {
        spans.emplace_back(lo, std::min(n, lo + len));
    }
    const auto nspans = static_cast<long>(spans.size());

    for (int pass = 0; pass < 2; ++pass) // 0: copy_range_mt, 1: transform_range_mt (negation)
    {
#pragma omp parallel
        {
            begin_phase();
#pragma omp for schedule(static, 1)
            for (long s = 0; s < nspans; ++s)
            {
                const auto lo = static_cast<std::size_t>(spans[static_cast<std::size_t>(s)].first);
                const auto hi = static_cast<std::size_t>(spans[static_cast<std::size_t>(s)].second);
                if (pass == 0)
                {
                    copy_range_mt(a, lo, hi, src.begin() + static_cast<std::ptrdiff_t>(lo));
                }
                else
                {
                    transform_range_mt(a, lo, hi, src.begin() + static_cast<std::ptrdiff_t>(lo),
                                       [](std::uint8_t x) { return x == 0; });
                }
            }
        }
        end_phase();
        for (long i = 0; i < n; ++i)
        {
            const bool v                     = src[static_cast<std::size_t>(i)] != 0;
            ref[static_cast<std::size_t>(i)] = pass == 0 ? v : !v;
        }
        fails += compare(a, ref, pass == 0 ? "copy_range_mt" : "transform_range_mt") != 0;
    }
    if (fails == 0)
    {
        println("[bits] OK copy_range_mt / transform_range_mt from {} threads, neighbouring spans in one word",
                CHUNK_PMP_MAX_THREADS);
    }
    a.clear();
    return fails;
}

} // namespace

int run(int argc, char** argv)
{
    // 85 runs: whole runs and whole chunks at every size the harness is built at.
    constexpr long default_n = 85 * run_len;
    const long     n         = argc > 1 ? shrink_array::arg_number<long>(argv[1], default_n, "N") : default_n;
    ParamChunk::init_indexes(true);

    println("=== bits_container_validation  N={}  threads={} ===", n, CHUNK_PMP_MAX_THREADS);
    int fails = 0;
    fails += birth_size();
    fails += surface();
    fails += round_trip(n);
    fails += neighbours(n);
    fails += logical_ops(n);
    fails += advisor_and_rechunk(n);
    fails += bulk_ranges(n);
    fails += parallel_fill(n);
    fails += parallel_copy_transform(n);

    release_chunk_pools();
    destroy_block_chunk();

    println("[bits] {}", fails == 0 ? "=== ALL CHECKS PASSED ===" : "=== FAILURES ===");
    return fails;
}

} // namespace validation::bits_container_validation
