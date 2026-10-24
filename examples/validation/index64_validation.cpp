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
// 64-bit index harness. Variant-agnostic: tests zero / zero_one_value / zero_any
// depending on the compile-time macro (CHUNK_VARIANT_ZERO*), like the
// other harnesses here.
// ============================================================================
//
// The arrays take their index type as a template parameter (IntType, `int`
// by default). Guards the two things a `long` index needs:
//
//   past-int-max  one zero::Array*<char, long> of more than INT_MAX elements,
//                 written on both sides of 2^31 (sequentially, then from a
//                 parallel region), read back, zeroed and compressed. Almost
//                 every chunk stays at the first constant, so the array costs
//                 its chunk metadata (a few hundred MB), not the 2 GB of its
//                 elements -- the reason this can run anywhere.
//
//   mixed         `<double, int>` and `<double, long>` arrays alive in one
//                 process, written in parallel, compressed, read back. The
//                 per-T machinery (proxies, registries, per-num_tab tables,
//                 allocator keys) is shared by the two index types; it must
//                 neither cast every array to `<T, int>` (a `long` array read
//                 four bytes off) nor number the arrays of each index type
//                 separately (two arrays sharing one num_tab). Under
//                 CHUNK_DYNAMIC, rechunk_all_to() is then run over the mixed
//                 registry: every array, of either index type, must come out
//                 at the new geometry with its values.
//
// --size N (CHUNK_DYNAMIC builds): `mixed` starts at geometry N and re-chunks
// to 2N, where `past-int-max` then runs. With a size that is not a power of
// two (the ctest _np2 runs 384, so 384 -> 768), chunk_of / offset_of take the
// magic-number division rather than the shift, on both sides of 2^31 too.
//
// Exit 0 = every check holds. Any thread count (the tests run it at 1 and 8).

#include <climits>     // INT_MAX
#include <string_view> // --size
#include <type_traits> // std::conditional_t
#include <utility>     // std::as_const
#include <vector>      // std::vector

#include "chunk/chunk_traits.h"               // forward decls + default IntType = int
#include "chunk/zero_array.h"                 // zero::Array
#include "chunk/zero_one_value_array.h"       // zero_one_value::Array
#include "chunk/zero_any_array.h"             // zero_any::Array
#include "chunk/param_chunk.h"                // ParamChunk::register_chunk_size, init_indexes
#include "chunk/zero_chunk.h"                 // release_chunk_pools, destroy_block_chunk
#include "chunk/config.h"                     // chunk_config::use_zero*, build_config::chunk_dynamic / has_not_ompt
#include "chunk/PMP_CHUNK.h"                  // CHUNK_PMP_MAX_THREADS
#include "chunk/manage_ompt.h"                // begin_parallel_phase / finalize_parallel_phase
#include "chunk/internal/chunk_size_policy.h" // set_active_chunk_size, set_thread_chunk_size
#include "chunk/internal/print.h"             // shrink_array::println
#include "chunk/internal/env_parse.h"         // arg_number (checked, not atoi)

#include "validation_harness.h" // self_command; run() is called by validation_driver.cpp

namespace validation::index64_validation
{

using namespace shrink_array;

namespace
{

static_assert(chunk_config::use_zero || chunk_config::use_zero_one_value || chunk_config::use_zero_any,
              "index64_validation must be built with CHUNK_VARIANT_ZERO, _ZERO_ONE_VALUE or _ZERO_ANY");

template <typename T, typename I>
using Array = std::conditional_t<
    chunk_config::use_zero_any,
    zero_any::Array<T, I>,
    std::conditional_t<chunk_config::use_zero_one_value, zero_one_value::Array<T, I>, zero::Array<T, I>>>;

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

/// zero_one_value compresses to the second value only once the array declared one.
template <typename Arr, typename Val>
void maybe_set_second_value(Arr& arr, const Val& value)
{
    if constexpr (requires(Arr probe) { probe.set_second_value(value); })
    {
        arr.set_second_value(value);
    }
}

/// The compression pass, driven the way the bench drives it: from a region.
template <typename Arr>
void compress_in_region(Arr& arr)
{
#pragma omp parallel
    {
        begin_phase();
#pragma omp single
        arr.compress();
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
    const long count = static_cast<long>(INT_MAX) + (3L * 4096) + 5;
    Big        big(count);
    maybe_set_second_value(big, char(9));
    const long chunk_sz = static_cast<long>(big.chunk_size());

    int bad = 0;
    if (big.size() != count)
    {
        println("[{}/{}] FAIL: size() = {}, expected {}", variant_name(), label, static_cast<long>(big.size()), count);
        ++bad;
    }
    if (big.capacity() < static_cast<std::size_t>(count))
    {
        println("[{}/{}] FAIL: capacity() = {} < {}", variant_name(), label, big.capacity(), count);
        ++bad;
    }

    // Both sides of 2^31, and the chunk boundaries around it.
    const long              huge = INT_MAX;
    const std::vector<long> pos{0,
                                1,
                                chunk_sz - 1,
                                chunk_sz,
                                huge - chunk_sz,
                                huge - 1,
                                huge,
                                huge + 1,
                                huge + 2,
                                huge + chunk_sz - 1,
                                huge + chunk_sz,
                                huge + (2 * chunk_sz),
                                count - chunk_sz - 1,
                                count - 2,
                                count - 1};
    const auto              value_at = [](std::size_t k) { return static_cast<char>(1 + (k % 7)); };

    // (1) sequential writes
    for (std::size_t k = 0; k < pos.size(); ++k)
    {
        big[pos[k]] = value_at(k);
    }

    const auto verify = [&](const char* step, auto expected) -> int
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
                {
                    println("[{}/{}] FAIL {}: [{}] = {}, at() = {}, expected {}", variant_name(), label, step, pos[k],
                            static_cast<int>(got), static_cast<int>(gat), static_cast<int>(want));
                }
                ++nbad;
            }
            // The neighbours nobody wrote are still the first constant.
            for (const long probe_pos : {pos[k] - 2, pos[k] + 2})
            {
                if (probe_pos < 0 || probe_pos >= count)
                {
                    continue;
                }
                bool written = false;
                for (const long position : pos)
                {
                    written = written || position == probe_pos;
                }
                if (!written && cbig[probe_pos] != char(0))
                {
                    if (nbad < 5)
                    {
                        println("[{}/{}] FAIL {}: untouched [{}] = {}", variant_name(), label, step, probe_pos,
                                static_cast<int>(cbig[probe_pos]));
                    }
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
        {
            big[pos[static_cast<std::size_t>(k)]] = value_at(static_cast<std::size_t>(k) + 3);
        }
    }
    end_phase();
    bad += verify("parallel", [&](std::size_t k) { return value_at(k + 3); });

    // (3) back to the first constant, then a compression pass
    for (const long position : pos)
    {
        big[position] = char(0);
    }
    compress_in_region(big);
    bad += verify("zeroed+compressed", [](std::size_t) { return char(0); });

    if (!bad)
    {
        println("[{}/{}] OK: N={} (INT_MAX + {}), {} positions across 2^31, chunk_size={}", variant_name(), label,
                count, count - huge, pos.size(), chunk_sz);
    }
    return bad ? 1 : 0;
}

// ---------------------------------------------------------------------------
// mixed
// ---------------------------------------------------------------------------

/// Whole chunks of one value, so that the compression pass has constant
/// chunks to fold (to the first value, and to the second where one exists).
double pattern(int which, long index, long chunk_sz)
{
    const long num_chunk = index / chunk_sz;
    switch (which)
    {
        case 0:
            return (num_chunk % 3 == 0) ? 0.0 : (num_chunk % 3 == 1) ? 1.0 : 1.0 + static_cast<double>(index % 97);
        case 1:
            return (num_chunk % 4 == 0) ? 0.0 : (num_chunk % 4 == 1) ? 100.0 : 100.0 + static_cast<double>(index % 89);
        default:
            return (num_chunk % 2 == 0) ? 7.0 : static_cast<double>(index % 5);
    }
}

template <typename Arr>
int verify_mixed(const char* label, const char* step, const Arr& arr, int which, long count, long chunk_sz)
{
    int nbad = 0;
    for (long i = 0; i < count; ++i)
    {
        const double want = pattern(which, i, chunk_sz);
        const double got  = arr[static_cast<Arr::IndType>(i)];
        if (got != want)
        {
            if (nbad < 5)
            {
                println("[{}/{}] FAIL {}: array {} [{}] = {:g}, expected {:g}", variant_name(), label, step, which, i,
                        got, want);
            }
            ++nbad;
        }
    }
    return nbad;
}

int check_mixed()
{
    const char* label = "mixed";

    // Built alternately, so the per-T numbering is shared: <int>, <long>, <int>.
    Array<double, int>  int_arr;
    Array<double, long> long_arr;
    Array<double, int>  int_arr2;
    const long          chunk_sz = static_cast<long>(int_arr.chunk_size());
    const long          count    = (50 * chunk_sz) + 7;
    int_arr.resize(static_cast<int>(count));
    long_arr.resize(count);
    int_arr2.resize(static_cast<int>(count));
    maybe_set_second_value(int_arr, 1.0);
    maybe_set_second_value(long_arr, 100.0);
    maybe_set_second_value(int_arr2, 7.0);

    const auto fill = [&](long round)
    {
#pragma omp parallel
        {
            begin_phase();
#pragma omp for schedule(static, 1)
            for (long i = 0; i < count; ++i)
            {
                int_arr[static_cast<int>(i)]  = pattern(0, i, chunk_sz) + static_cast<double>(round);
                long_arr[i]                   = pattern(1, i, chunk_sz) + static_cast<double>(round);
                int_arr2[static_cast<int>(i)] = pattern(2, i, chunk_sz) + static_cast<double>(round);
            }
        }
        end_phase();
    };
    // A round of other values first, so the real one overwrites materialised
    // chunks and leaves some of them constant again: compression candidates.
    fill(3);
    fill(0);

    int bad = 0;
    bad += verify_mixed(label, "written", std::as_const(int_arr), 0, count, chunk_sz);
    bad += verify_mixed(label, "written", std::as_const(long_arr), 1, count, chunk_sz);
    bad += verify_mixed(label, "written", std::as_const(int_arr2), 2, count, chunk_sz);

    compress_in_region(int_arr);
    compress_in_region(long_arr);
    compress_in_region(int_arr2);
    bad += verify_mixed(label, "compressed", std::as_const(int_arr), 0, count, chunk_sz);
    bad += verify_mixed(label, "compressed", std::as_const(long_arr), 1, count, chunk_sz);
    bad += verify_mixed(label, "compressed", std::as_const(int_arr2), 2, count, chunk_sz);

    if constexpr (build_config::chunk_dynamic)
    {
        // One walk of the per-T registry, which holds both index types.
        const unsigned int tgt =
            (chunk_sz >= 1024) ? static_cast<unsigned int>(chunk_sz / 2) : static_cast<unsigned int>(chunk_sz * 2);
        Array<double, int>::rechunk_all_to(tgt);
        // rechunk_all_to() leaves the active geometry as it found it: the
        // caller moves it, once, before the next writes.
        set_active_chunk_size(tgt);
        set_thread_chunk_size(tgt);
        ParamChunk::register_chunk_size(tgt);

        for (const unsigned int got : {int_arr.chunk_size(), long_arr.chunk_size(), int_arr2.chunk_size()})
        {
            if (got != tgt)
            {
                println("[{}/{}] FAIL rechunk_all_to: chunk_size {}, expected {}", variant_name(), label, got, tgt);
                ++bad;
            }
        }
        bad += verify_mixed(label, "rechunked", std::as_const(int_arr), 0, count, chunk_sz);
        bad += verify_mixed(label, "rechunked", std::as_const(long_arr), 1, count, chunk_sz);
        bad += verify_mixed(label, "rechunked", std::as_const(int_arr2), 2, count, chunk_sz);

        // And the arrays still take writes and compress at the new geometry.
        fill(0);
        compress_in_region(long_arr);
        bad += verify_mixed(label, "rechunked+rewritten", std::as_const(int_arr), 0, count, chunk_sz);
        bad += verify_mixed(label, "rechunked+rewritten", std::as_const(long_arr), 1, count, chunk_sz);
        bad += verify_mixed(label, "rechunked+rewritten", std::as_const(int_arr2), 2, count, chunk_sz);
    }

    if (!bad)
    {
        println("[{}/{}] OK: <double,int> x2 and <double,long> x1, M={}, written in parallel, compressed{}",
                variant_name(), label, count, build_config::chunk_dynamic ? ", rechunked together" : "");
    }
    return bad ? 1 : 0;
}

} // namespace

int run(int argc, char** argv)
{
    ParamChunk::init_indexes(true);

    // --size N (CHUNK_DYNAMIC): start at geometry N instead of the build's.
    // `mixed` re-chunks to 2N and leaves it active for `past-int-max`: from a
    // size that is not a power of two (384), both run on chunk_of /
    // offset_of's magic-number division (384, then 768 past 2^31).
    if (argc > 2 && std::string_view(argv[1]) == "--size")
    {
        const auto size = shrink_array::arg_number<unsigned int>(argv[2], 0u, "size");
        if (!build_config::chunk_dynamic || !is_valid_chunk_size(size) || size > cst_buffer_size / 2)
        {
            println("index64_validation: --size {}: a dynamic build and a valid size up to {}", size,
                    cst_buffer_size / 2);
            return 1;
        }
        set_active_chunk_size(size);
        set_thread_chunk_size(size);
        ParamChunk::register_chunk_size(size);
    }

    println("=== index64_validation [variant = {}]  threads={}  chunk_size={} ===", variant_name(),
            CHUNK_PMP_MAX_THREADS, active_chunk_size());

    int fails = 0;
    fails += check_mixed();
    fails += check_past_int_max();

    release_chunk_pools();
    destroy_block_chunk();

    println("\n[{}] {}", variant_name(), fails ? "=== INDEX64 FAILED ===" : "=== INDEX64 OK ===");
    return fails ? 1 : 0;
}

} // namespace validation::index64_validation
