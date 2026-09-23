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
// 64-bit index harness. Variant-agnostic: tests chunk / chunk2 / chunk2a
// depending on the compile-time macro (CHUNK / CHUNK2 / CHUNK2A), like the
// other harnesses here.
// ============================================================================
//
// The arrays take their index type as a template parameter (ind_type, `int`
// by default). Guards the two things a `long` index needs:
//
//   past-int-max  one ArrayChunk*<char, long> of more than INT_MAX elements,
//                 written on both sides of 2^31 (sequentially, then from a
//                 parallel region), read back, zeroed and compressed. Almost
//                 every chunk stays at the first constant, so the array costs
//                 its chunk metadata (a few hundred MB), not the 2 GB of its
//                 elements -- the reason this can run anywhere.
//
//   mixed         `<double, int>` and `<double, long>` arrays alive in one
//                 process, written in parallel, compressed, read back. The
//                 per-T machinery (proxies, registries, per-num_tab tables,
//                 allocator keys) is shared by the two index types; it used to
//                 cast every array to `<T, int>` (so a `long` array was read
//                 four bytes off) and to number the arrays of each index type
//                 separately (so two arrays shared one num_tab). Under
//                 CHUNK_DYNAMIC, rechunk_all_to() is then run over the mixed
//                 registry: every array, of either index type, must come out
//                 at the new geometry with its values.
//
// Exit 0 = every check holds. Any thread count (the tests run it at 1 and 8).

#include <climits>
#include <cstdio>
#include <cstdlib>
#include <type_traits>
#include <utility>
#include <unistd.h>
#include <vector>

#include "chunk/chunk_traits.h"
#include "chunk/array_chunk.h"
#include "chunk/array_chunk2.h"
#include "chunk/array_chunk2A.h"
#include "chunk/param_chunk.h"
#include "chunk/config.h"
#include "chunk/PMP_CHUNK.h"
#include "chunk/manage_ompt.h"

int __mem_page_size = 0;

using namespace shrink_array;

namespace
{

static_assert(chunk_config::use_chunk || chunk_config::use_chunk2 || chunk_config::use_chunk2a,
              "index64_validation must be built with CHUNK, CHUNK2 or CHUNK2A");

template <typename T, typename I>
using Array = std::conditional_t<chunk_config::use_chunk2a, ArrayChunk2A<T, I>,
              std::conditional_t<chunk_config::use_chunk2, ArrayChunk2<T, I>, ArrayChunk<T, I>>>;

constexpr const char* variant_name()
{
    if constexpr (chunk_config::use_chunk2a)
        return "chunk2a";
    else if constexpr (chunk_config::use_chunk2)
        return "chunk2";
    else
        return "chunk";
}

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

/// Chunk2 compresses to the second value only once the array declared one.
template <typename Arr, typename Val>
void maybe_set_second_value(Arr& a, const Val& v)
{
    if constexpr (requires(Arr x) { x.set_second_value(v); })
        a.set_second_value(v);
}

/// The compression pass, driven the way the bench drives it: from a region.
template <typename Arr>
void compress_in_region(Arr& a)
{
#pragma omp parallel
    {
        begin_phase();
#pragma omp single
        a.compress();
    }
    end_phase();
}

// ---------------------------------------------------------------------------
// past-int-max
// ---------------------------------------------------------------------------
int check_past_int_max()
{
    const char* label = "past-int-max";
    using Big         = Array<char, long>;

    // Three chunks' worth past INT_MAX at the largest geometry (4096), and an
    // odd tail so the last chunk is partial.
    const long N = static_cast<long>(INT_MAX) + 3L * 4096 + 5;
    Big        big(N);
    maybe_set_second_value(big, char(9));
    const long S = static_cast<long>(big.chunk_size());

    int bad = 0;
    if (big.size() != N)
    {
        std::printf("[%s/%s] FAIL: size() = %ld, expected %ld\n", variant_name(), label, static_cast<long>(big.size()),
                    N);
        ++bad;
    }
    if (big.capacity() < static_cast<std::size_t>(N))
    {
        std::printf("[%s/%s] FAIL: capacity() = %zu < %ld\n", variant_name(), label, big.capacity(), N);
        ++bad;
    }

    // Both sides of 2^31, and the chunk boundaries around it.
    const long I = INT_MAX;
    const std::vector<long> pos{0,       1,         S - 1,     S,         I - S,     I - 1,     I,         I + 1,
                                I + 2,   I + S - 1, I + S,     I + 2 * S, N - S - 1, N - 2,     N - 1};
    auto value_at = [](std::size_t k) { return static_cast<char>(1 + (k % 7)); };

    // (1) sequential writes
    for (std::size_t k = 0; k < pos.size(); ++k)
        big[pos[k]] = value_at(k);

    auto verify = [&](const char* step, auto expected) -> int
    {
        int         nbad = 0;
        const auto& cbig = std::as_const(big);
        for (std::size_t k = 0; k < pos.size(); ++k)
        {
            const char want = expected(k);
            const char got  = cbig[pos[k]];
            const char gat  = cbig.at(pos[k]);
            if (got != want || gat != want)
            {
                if (nbad < 5)
                    std::printf("[%s/%s] FAIL %s: [%ld] = %d, at() = %d, expected %d\n", variant_name(), label, step,
                                pos[k], got, gat, want);
                ++nbad;
            }
            // The neighbours nobody wrote are still the first constant.
            for (const long q : {pos[k] - 2, pos[k] + 2})
            {
                if (q < 0 || q >= N)
                    continue;
                bool written = false;
                for (const long p : pos)
                    written = written || p == q;
                if (!written && cbig[q] != char(0))
                {
                    if (nbad < 5)
                        std::printf("[%s/%s] FAIL %s: untouched [%ld] = %d\n", variant_name(), label, step, q,
                                    cbig[q]);
                    ++nbad;
                }
            }
        }
        return nbad;
    };
    bad += verify("sequential", value_at);

    // (2) the same positions rewritten from a parallel region, one per
    //     iteration: each thread writes on both sides of 2^31
    const long npos = static_cast<long>(pos.size());
#pragma omp parallel
    {
        begin_phase();
#pragma omp for schedule(static, 1)
        for (long k = 0; k < npos; ++k)
            big[pos[static_cast<std::size_t>(k)]] = value_at(static_cast<std::size_t>(k) + 3);
    }
    end_phase();
    bad += verify("parallel", [&](std::size_t k) { return value_at(k + 3); });

    // (3) back to the first constant, then a compression pass
    for (const long p : pos)
        big[p] = char(0);
    compress_in_region(big);
    bad += verify("zeroed+compressed", [](std::size_t) { return char(0); });

    if (!bad)
        std::printf("[%s/%s] OK: N=%ld (INT_MAX + %ld), %zu positions across 2^31, chunk_size=%ld\n", variant_name(),
                    label, N, N - I, pos.size(), S);
    return bad ? 1 : 0;
}

// ---------------------------------------------------------------------------
// mixed
// ---------------------------------------------------------------------------

/// Whole chunks of one value, so that the compression pass has constant
/// chunks to fold (to the first value, and to the second where one exists).
double pattern(int which, long i, long S)
{
    const long c = i / S;
    switch (which)
    {
    case 0:
        return (c % 3 == 0) ? 0.0 : (c % 3 == 1) ? 1.0 : 1.0 + static_cast<double>(i % 97);
    case 1:
        return (c % 4 == 0) ? 0.0 : (c % 4 == 1) ? 100.0 : 100.0 + static_cast<double>(i % 89);
    default:
        return (c % 2 == 0) ? 7.0 : static_cast<double>(i % 5);
    }
}

template <typename Arr>
int verify_mixed(const char* label, const char* step, const Arr& a, int which, long M, long S)
{
    int nbad = 0;
    for (long i = 0; i < M; ++i)
    {
        const double want = pattern(which, i, S);
        const double got  = a[static_cast<typename Arr::ind_type>(i)];
        if (got != want)
        {
            if (nbad < 5)
                std::printf("[%s/%s] FAIL %s: array %d [%ld] = %g, expected %g\n", variant_name(), label, step, which, i,
                            got, want);
            ++nbad;
        }
    }
    return nbad;
}

int check_mixed()
{
    const char* label = "mixed";

    // Built alternately, so the per-T numbering is shared: <int>, <long>, <int>.
    Array<double, int>  a;
    Array<double, long> b;
    Array<double, int>  c;
    const long          S = static_cast<long>(a.chunk_size());
    const long          M = 50 * S + 7;
    a.resize(static_cast<int>(M));
    b.resize(M);
    c.resize(static_cast<int>(M));
    maybe_set_second_value(a, 1.0);
    maybe_set_second_value(b, 100.0);
    maybe_set_second_value(c, 7.0);

    auto fill = [&](long round)
    {
#pragma omp parallel
        {
            begin_phase();
#pragma omp for schedule(static, 1)
            for (long i = 0; i < M; ++i)
            {
                a[static_cast<int>(i)] = pattern(0, i, S) + static_cast<double>(round);
                b[i]                   = pattern(1, i, S) + static_cast<double>(round);
                c[static_cast<int>(i)] = pattern(2, i, S) + static_cast<double>(round);
            }
        }
        end_phase();
    };
    // A round of other values first, so the real one overwrites materialised
    // chunks and leaves some of them constant again: compression candidates.
    fill(3);
    fill(0);

    int bad = 0;
    bad += verify_mixed(label, "written", std::as_const(a), 0, M, S);
    bad += verify_mixed(label, "written", std::as_const(b), 1, M, S);
    bad += verify_mixed(label, "written", std::as_const(c), 2, M, S);

    compress_in_region(a);
    compress_in_region(b);
    compress_in_region(c);
    bad += verify_mixed(label, "compressed", std::as_const(a), 0, M, S);
    bad += verify_mixed(label, "compressed", std::as_const(b), 1, M, S);
    bad += verify_mixed(label, "compressed", std::as_const(c), 2, M, S);

    if constexpr (build_config::chunk_dynamic)
    {
        // One walk of the per-T registry, which holds both index types.
        const unsigned int tgt = (S >= 1024) ? static_cast<unsigned int>(S / 2) : static_cast<unsigned int>(S * 2);
        Array<double, int>::rechunk_all_to(tgt);
        // rechunk_all_to() leaves the active geometry as it found it: the
        // caller moves it, once, before the next writes.
        set_active_chunk_size(tgt);
        set_thread_chunk_size(tgt);
        ParamChunk::register_chunk_size(tgt);

        for (const unsigned int got : {a.chunk_size(), b.chunk_size(), c.chunk_size()})
            if (got != tgt)
            {
                std::printf("[%s/%s] FAIL rechunk_all_to: chunk_size %u, expected %u\n", variant_name(), label, got,
                            tgt);
                ++bad;
            }
        bad += verify_mixed(label, "rechunked", std::as_const(a), 0, M, S);
        bad += verify_mixed(label, "rechunked", std::as_const(b), 1, M, S);
        bad += verify_mixed(label, "rechunked", std::as_const(c), 2, M, S);

        // And the arrays still take writes and compress at the new geometry.
        fill(0);
        compress_in_region(b);
        bad += verify_mixed(label, "rechunked+rewritten", std::as_const(a), 0, M, S);
        bad += verify_mixed(label, "rechunked+rewritten", std::as_const(b), 1, M, S);
        bad += verify_mixed(label, "rechunked+rewritten", std::as_const(c), 2, M, S);
    }

    if (!bad)
        std::printf("[%s/%s] OK: <double,int> x2 and <double,long> x1, M=%ld, written in parallel, compressed%s\n",
                    variant_name(), label, M, build_config::chunk_dynamic ? ", rechunked together" : "");
    return bad ? 1 : 0;
}

} // namespace

int main()
{
    __mem_page_size = getpagesize();
    ParamChunk::init_indexes(true);

    std::printf("=== index64_validation [variant = %s]  threads=%d ===\n", variant_name(), CHUNK_PMP_MAX_THREADS);

    int fails = 0;
    fails += check_mixed();
    fails += check_past_int_max();

    release_chunk_pools();
    destroy_block_chunk();

    std::printf("\n[%s] %s\n", variant_name(), fails ? "=== INDEX64 FAILED ===" : "=== INDEX64 OK ===");
    return fails ? 1 : 0;
}
