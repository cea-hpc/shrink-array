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

// ============================================================================
// SEQUENTIAL re-chunk validation harness (path C, non-OMP) -- v3.
// Variant-agnostic: tests chunk / chunk2 / chunk2a depending on the
// compile-time macro (CHUNK_VARIANT_V1 / _V2 / _V2A), like shrink_bench.cpp.
// ============================================================================
//
// Proves CONTENT INVARIANCE and, on failure, LOCATES the cause:
//   - first checks that the SNAPSHOT (read at the old geometry) is correct;
//   - traces the geometry before/after;
//   - shows the first mismatches along with the corresponding chunk boundary.
//
// The variant is selected by the macro passed by CMake (one executable per
// variant, linked against the matching dyn lib: chunk_v2_dyn / chunk_v2a_dyn /
// chunk_v1_dyn). No std::vector fallback: re-chunk is specific to the chunked
// variants, hence the static_assert.
//
// Chunk2 CONTRACT: the compression-candidate list is indexed by
// second_value (map_chunk2_to_compress_potentially_thr[thr][second_value]),
// populated ONLY by set_second_value. A second_value must therefore be
// declared BEFORE writing, otherwise get_vec_tuple_to_compress_potentially_thr
// asserts (vec == nullptr). This is what the bench does via
// maybe_set_second_value; we reproduce it here for every case. 2A / v1
// don't need this (no-op).
//
// Compile NON-OMP (or OMP), dyn, check. Exit 0 = every invariant holds.
//
// Each case runs twice by default (--mode both): SERIALLY, as it always did,
// then with its writes and checks in parallel regions (--mode parallel) --
// the candidate lists per rank, the deferred path and the ambient chunk size
// of worker threads around a re-chunk, which the serial pass never reaches.
// rechunk_to() itself is always called outside any region.
//
//   rechunk_validation [N [target [rounds]]] [--case NAME] [--mode serial|parallel|both] [--list]
//
// --case runs one case: CMake registers one test per case, so `ctest -j`
// spreads them over the cores (the cases share global re-chunk state -- the
// ambient chunk size, the registries -- so they run in separate processes,
// never side by side in one).

#include <cstdio>      // std::fprintf, stderr
#include <cstring>     // std::strcmp
#include <functional>  // std::function
#include <limits>      // std::numeric_limits (index-split)
#include <string>      // std::string
#include <vector>      // std::vector
#include <cmath>       // std::floor, std::sin
#include <type_traits> // std::conditional_t

#include "chunk/chunk_traits.h"               // forward decls + default IndT = int
#include "chunk/array_chunk.h"                // zero::Array
#include "chunk/array_chunk2.h"               // zero_one_value::Array
#include "chunk/array_chunk2A.h"              // zero_any::Array
#include "chunk/param_chunk.h"                // ParamChunk::register_chunk_size, init_indexes
#include "chunk/internal/chunk_size_policy.h" // set_active_chunk_size, set_thread_chunk_size
#include "chunk/config.h"                     // chunk_config::use_chunk*
#include "chunk/array_chunk_range_ops.h"      // fill_range / copy_range
#include "chunk/manage_ompt.h"                // finalize_parallel_phase (compression backlog drain)
#include "chunk/internal/env_parse.h"         // env_number / arg_number (checked, not atoi)
#include "chunk/internal/print.h"             // shrink_array::println

using namespace shrink_array;

#include "chunk/chunk_std.h"    // release_chunk_pools, destroy_block_chunk
#include "validation_harness.h" // self_command; run() is called by validation_driver.cpp

namespace validation::rechunk_validation
{

namespace
{

// ── Compression backlog drain ────────────────────────────────────────────────
// A write only QUEUES its chunk as a compression candidate; the queue is
// drained at phase boundaries. When OMPT is compiled in, its callbacks do it;
// otherwise the application must, which is what shrink_bench.cpp does around
// each of its parallel phases.
//
// This file opens no parallel region, so under OpenMP it must drain the
// backlog itself: otherwise every all-zero chunk stays materialised, and
// compress_int() -- which in embedded-compression mode only DIAGNOSES, it does
// not compress -- reports them round after round ("Should not have compress
// (0) values"); values stay correct, but no compression is exercised past the
// first round. No-op in serial builds
// (finalize_parallel_phase is `if constexpr (use_openmp)`) and skipped under
// OMPT, which already drains.
void drain_compression_backlog()
{
    if constexpr (build_config::has_not_ompt)
    {
        shrink_array::finalize_parallel_phase();
    }
}

// ── Serial or parallel passes ────────────────────────────────────────────────
bool g_parallel = false; ///< --mode: the pass being run

/// @brief f(i) for i in [0, N): a plain loop, or a work-shared parallel
///        region (static schedule) followed by the phase-end drain.
template <typename F>
void for_each_index(long n, F&& f)
{
    if (!g_parallel)
    {
        for (long i = 0; i < n; ++i)
        {
            f(i);
        }
        return;
    }
#pragma omp parallel
    {
        if constexpr (build_config::has_not_ompt)
        {
            shrink_array::begin_parallel_phase();
        }
#pragma omp for schedule(static)
        for (long i = 0; i < n; ++i)
        {
            f(i);
        }
    }
    drain_compression_backlog();
}

/// @brief How many i in [0, N) satisfy bad(i) -- the checks, in the same
///        serial or parallel pass. Details are then printed by a serial scan.
template <typename F>
long count_if_index(long n, F&& bad)
{
    long count = 0;
    if (!g_parallel)
    {
        for (long i = 0; i < n; ++i)
        {
            count += bad(i) ? 1 : 0;
        }
        return count;
    }
#pragma omp parallel reduction(+ : count)
    {
        if constexpr (build_config::has_not_ompt)
        {
            shrink_array::begin_parallel_phase();
        }
#pragma omp for schedule(static)
        for (long i = 0; i < n; ++i)
        {
            count += bad(i) ? 1 : 0;
        }
    }
    drain_compression_backlog();
    return count;
}

/// @brief The case label with the pass it runs in.
std::string pass_label(const char* label)
{
    return std::string(label) + (g_parallel ? "@par" : "@ser");
}

} // namespace

// --- Active variant selection (same shape as shrink_bench.cpp) -------
static_assert(chunk_config::use_chunk || chunk_config::use_chunk2 || chunk_config::use_chunk2a,
              "rechunk_validation must be built with CHUNK_VARIANT_V1, _V2 or _V2A "
              "(re-chunk is chunk-only; no std::vector fallback here)");

template <typename T>
using Array =
    std::conditional_t<chunk_config::use_chunk2a,
                       zero_any::Array<T>,
                       std::conditional_t<chunk_config::use_chunk2, zero_one_value::Array<T>, zero::Array<T>>>;

constexpr const char* variant_name()
{
    if constexpr (chunk_config::use_chunk2a)
    {
        return "chunk2a";
    }
    else if constexpr (chunk_config::use_chunk2)
    {
        return "chunk2";
    }
    else
    {
        return "chunk";
    }
}

// set_second_value only exists on chunk2 / chunk2a: no-op elsewhere. REQUIRED
// on chunk2 before any write (see the contract at the top of the file); also
// exercises the val2 path (second_value preservation + tab_m_ptr_cst_val2
// queue on growth, for chunk2).
template <typename Arr, typename Val>
void maybe_set_second_value(Arr& a, const Val& v)
{
    if constexpr (requires(Arr x) { x.set_second_value(v); })
    {
        a.set_second_value(v);
    }
}

// sv_value: the value declared as second_value. Choosing a value present in
// the data makes val2 compression actually kick in (a stronger test); any
// value is enough to satisfy the Chunk2 contract (populate the map).
template <typename T>
int check_invariance(const char* label,
                     long        n,
                     unsigned    new_size,
                     T (*gen)(long),
                     T   sv_value    = T(1),
                     int post_rounds = 0)
{
    const std::string lab_s = pass_label(label);
    label                   = lab_s.c_str();

    Array<T> arr(n);
    maybe_set_second_value(arr, sv_value); // no-op on chunk v1; required on chunk2

    fprintf(stderr, "[%s/%s] arr built: chunk_size=%u (g_active should be 512)\n", variant_name(), label,
            arr.chunk_size());

    std::vector<T> ref(n);
    for_each_index(n,
                   [&](long i)
                   {
                       const T v = gen(i);
                       arr[i]    = v;
                       ref[i]    = v;
                   });

    const unsigned old_size = arr.chunk_size();

    // (a) pre-rechunk DIAGNOSTIC: is the content read at the old geometry correct?
    const long pre_bad   = count_if_index(n, [&](long i) { return arr[i] != ref[i]; });
    long       pre_first = -1;
    for (long i = 0; pre_bad && pre_first < 0 && i < n; ++i)
    {
        if (arr[i] != ref[i])
        {
            pre_first = i;
        }
    }
    if (pre_bad)
    {
        println("[{}/{}] PRE-FAIL: {} mismatches BEFORE rechunk (setup/read bug, not rechunk); "
                "first at i={} (chunk #{}, offset {}) got={:g} expected={:g}",
                variant_name(), label, pre_bad, pre_first, pre_first / (long)old_size, pre_first % (long)old_size,
                (double)arr[pre_first], (double)ref[pre_first]);
        return 1;
    }

    arr.rechunk_to(new_size);
    const unsigned got_size = arr.chunk_size();

    if (got_size != new_size)
    {
        println("[{}/{}] FAIL: geometry {} -> {} (expected {})", variant_name(), label, old_size, got_size, new_size);
        return 1;
    }
    if ((long)arr.size() != n)
    {
        println("[{}/{}] FAIL: size {} -> {}", variant_name(), label, n, (long)arr.size());
        return 1;
    }
    const long mismatches = count_if_index(n, [&](long i) { return arr[i] != ref[i]; });
    long       first      = -1;
    for (long i = 0, shown = 0; mismatches && shown < 5 && i < n; ++i)
    {
        if (arr[i] != ref[i])
        {
            if (first < 0)
            {
                first = i;
            }
            println("[{}/{}] MISMATCH i={} got={:.17g} exp={:.17g} (new-chunk #{}, offset {})", variant_name(), label,
                    i, (double)arr[i], (double)ref[i], i / (long)new_size, i % (long)new_size);
            ++shown;
        }
    }
    if (mismatches)
    {
        println("[{}/{}] FAIL: {}/{} mismatches after rechunk {}->{} (first at i={})", variant_name(), label,
                mismatches, n, old_size, new_size, first);
        return 1;
    }
    // ── POST-RECHUNK PHASE: rewrite + compress in rounds, mimicking the bench.
    //    The perf-bench crash appears HERE (writes/compression touching the
    //    freshly re-chunked geometry), not in rechunk_to itself. Each round
    //    overwrites with changing data, forces a compression pass, and re-checks
    //    invariance against the ground-truth ref.
    for (int round = 0; round < post_rounds; ++round)
    {
        for_each_index(n,
                       [&](long i)
                       {
                           const T v = gen(i + ((long)round * 7919));
                           arr[i]    = v;
                           ref[i]    = v;
                       });
        // Hand the queued candidates to the compression pass before asking for
        // the result -- see drain_compression_backlog() above.
        drain_compression_backlog();
        // compress_int signature differs across variants: v1 (CHUNK) takes a
        // verbose bool; CHUNK2 / CHUNK2A take none.
        if constexpr (requires { arr.compress_int(false); })
        {
            arr.compress_int(false);
        }
        else
        {
            arr.compress_int();
        }
        const long bad    = count_if_index(n, [&](long i) { return arr[i] != ref[i]; });
        long       firstb = -1;
        for (long i = 0; bad && firstb < 0 && i < n; ++i)
        {
            if (arr[i] != ref[i])
            {
                firstb = i;
            }
        }
        if (bad)
        {
            println("[{}/{}] POST-FAIL round {}: {}/{} mismatches after compress (first i={}, new-chunk #{} off {})",
                    variant_name(), label, round, bad, n, firstb, firstb / (long)new_size, firstb % (long)new_size);
            return 1;
        }
    }
    println("[{}/{}] OK: {} elements invariant, geometry {}->{} (second_value={:.17g}){}", variant_name(), label, n,
            old_size, new_size, (double)sv_value, post_rounds ? " [+post-rechunk compress rounds]" : "");
    return 0;
}

// ── COEXISTENCE CASE ─────────────────────────────────────────────────────────
// Every check above exercises ONE array at a time, so all live arrays always
// share a single geometry -- and they all pass. The perf bench does not: the
// advisor re-chunks the arrays it tracks and leaves the others at the original
// size, so arrays of DIFFERENT geometries are alive and written to in the same
// loop. That is what crashes (ASan: heap-buffer-overflow in store_value, a
// 64-element buffer written at index 76), on chunk AND on chunk2/chunk2a alike.
//
// This reproduces that state in the small: two arrays, one re-chunked and one
// left alone, written alternately. Passing means geometry coexistence is safe;
// failing pins the defect down to ~30 lines instead of a 1200-iteration bench.
template <typename T>
int check_coexistence(const char* label, long n, unsigned moved_size, int rounds)
{
    using Arr = Array<T>;

    const std::string lab_s = pass_label(label);
    label                   = lab_s.c_str();

    // `fixed` first: it takes the lower registry index (num_tab), so a re-chunk
    // of `moved` whose teardown released the wrong row of the per-type flag
    // table would pull it from under `fixed` -- the writes below then fault.
    Arr fixed(static_cast<int>(n)); // stays at the ambient geometry
    Arr moved(static_cast<int>(n)); // will be re-chunked

    for_each_index(n,
                   [&](long i)
                   {
                       moved[i] = static_cast<T>((i % 7) ? 0 : i);
                       fixed[i] = static_cast<T>((i % 5) ? 0 : i);
                   });

    moved.rechunk_to(moved_size);

    // Alternate writes between the two geometries, the way the bench's loop does.
    for (int r = 0; r < rounds; ++r)
    {
        for_each_index(n,
                       [&](long i)
                       {
                           moved[i] = static_cast<T>((i + r) % 11 ? 0 : (i + r));
                           fixed[i] = static_cast<T>((i + r) % 13 ? 0 : (i + r));
                       });
    }

    // Verify both arrays still read back what was last written.
    const int  r = rounds - 1;
    const long nbad =
        count_if_index(n,
                       [&](long i)
                       {
                           return !(static_cast<T>(moved[i]) == static_cast<T>((i + r) % 11 ? 0 : (i + r))) ||
                                  !(static_cast<T>(fixed[i]) == static_cast<T>((i + r) % 13 ? 0 : (i + r)));
                       });
    int bad = 0;
    for (long i = 0; nbad && i < n && bad < 5; ++i)
    {
        const T em = static_cast<T>((i + r) % 11 ? 0 : (i + r));
        const T ef = static_cast<T>((i + r) % 13 ? 0 : (i + r));
        if (!(static_cast<T>(moved[i]) == em) || !(static_cast<T>(fixed[i]) == ef))
        {
            println("[{}/{}] MISMATCH at {}: moved={:g} exp {:g} | fixed={:g} exp {:g}", variant_name(), label, i,
                    (double)static_cast<T>(moved[i]), (double)em, (double)static_cast<T>(fixed[i]), (double)ef);
            ++bad;
        }
    }
    if (!bad)
    {
        println("[{}/{}] OK: {} elements, geometries {} and ambient coexisting", variant_name(), label, n, moved_size);
    }
    return bad ? 1 : 0;
}

// ── RANGE OPS ────────────────────────────────────────────────────────────────
// fill_range / copy_range take one decision per CHUNK where operator[] takes one
// per element. Correctness is checked against the element-by-element path on the
// SAME data, so any divergence -- a stale zero counter, a chunk wrongly left
// constant, a boundary off by one -- shows up as a mismatch rather than as a
// slow drift noticed three compression passes later.
template <typename T>
int check_range_ops(const char* label, long n)
{
    using Arr = Array<T>;

    const std::string lab_s = pass_label(label);
    label                   = lab_s.c_str();

    Arr via_range(static_cast<int>(n));
    Arr via_elem(static_cast<int>(n));

    const auto expected = [](long i) -> T
    {
        // Long constant runs (so chunks really do compress) with scattered
        // non-constant values, which is what makes the two paths diverge if the
        // bookkeeping is wrong.
        if ((i / 100) % 3 == 0)
        {
            return static_cast<T>(0);
        }
        if ((i % 37) == 0)
        {
            return static_cast<T>(i % 1000);
        }
        return static_cast<T>(0);
    };

    // 1) whole-array zero fill: the case where a correct implementation must
    //    allocate NOTHING (every chunk set constant outright).
    shrink_array::fill_range(via_range, 0, static_cast<std::size_t>(n), static_cast<T>(0));
    for (long i = 0; i < n; ++i)
    {
        via_elem[i] = static_cast<T>(0);
    }

    // 2) scattered writes, both paths identical
    for (long i = 0; i < n; ++i)
    {
        const T e = expected(i);
        if (!(e == static_cast<T>(0)))
        {
            via_range[i] = e;
            via_elem[i]  = e;
        }
    }

    // 3) partial fills straddling chunk boundaries, deliberately unaligned
    const unsigned s = via_range.chunk_size();
    for (const long off : {1L, (long)s - 1, (long)s + 1, (long)(2 * s) + 3})
    {
        const auto a0 = static_cast<std::size_t>(off);
        const auto b0 = static_cast<std::size_t>(off) + (5 * static_cast<std::size_t>(s)) + 7;
        if (b0 >= static_cast<std::size_t>(n))
        {
            continue;
        }
        shrink_array::fill_range(via_range, a0, b0, static_cast<T>(7));
        for (std::size_t i = a0; i < b0; ++i)
        {
            via_elem[static_cast<long>(i)] = static_cast<T>(7);
        }
    }

    // 4) copy_range from a plain vector -- the generic/overloaded pair in use
    std::vector<T> src(static_cast<std::size_t>(n));
    for (long i = 0; i < n; ++i)
    {
        src[static_cast<std::size_t>(i)] = static_cast<T>((i % 11) ? 0 : i);
    }
    const std::size_t ca = 3;
    const std::size_t cb = static_cast<std::size_t>(n) - 3;
    shrink_array::copy_range(via_range, ca, cb, src.begin() + 0);
    for (std::size_t i = ca; i < cb; ++i)
    {
        via_elem[static_cast<long>(i)] = src[i - ca];
    }

    const long nbad =
        count_if_index(n, [&](long i) { return !(static_cast<T>(via_range[i]) == static_cast<T>(via_elem[i])); });
    int bad = 0;
    for (long i = 0; nbad && i < n && bad < 5; ++i)
    {
        const T r = static_cast<T>(via_range[i]);
        const T e = static_cast<T>(via_elem[i]);
        if (!(r == e))
        {
            println("[{}/{}] MISMATCH at {}: range={:g} element={:g}", variant_name(), label, i, (double)r, (double)e);
            ++bad;
        }
    }
    if (!bad)
    {
        println("[{}/{}] OK: {} elements, range ops match the element path", variant_name(), label, n);
    }
    return bad ? 1 : 0;
}

/// @brief A size that is neither a power of two nor a multiple of 64 (48, 96,
///        3, S + 1) is refused -- by rechunk_to() on a filled array and on an
///        empty one, and by set_chunk_size() -- and so is, on the empty array,
///        a valid size wider than cst_buffer_size (it would read the shared
///        constant buffers out of bounds); geometry, content and the ambient
///        size stay as they were. A legal re-chunk must still work afterwards.
int check_non_pow2_refused(const char* label)
{
    const std::string lab_s = pass_label(label);
    label                   = lab_s.c_str();

    const long          n = 100000;
    Array<double>       arr(n);
    std::vector<double> ref(n);
    maybe_set_second_value(arr, 1.0);
    for_each_index(n,
                   [&](long i)
                   {
                       const double v = double((i / 700) % 3);
                       arr[i]         = v;
                       ref[i]         = v;
                   });
    const unsigned old_size = arr.chunk_size();
    const unsigned ambient  = active_chunk_size();

    int bad = 0;
    for (const unsigned s : {48u, 96u, 3u, old_size + 1u})
    {
        arr.rechunk_to(s);
        if (arr.chunk_size() != old_size)
        {
            println("[{}/{}] FAIL: rechunk_to({}) applied: geometry {} -> {}", variant_name(), label, s, old_size,
                    arr.chunk_size());
            ++bad;
        }
    }
    const long mism = count_if_index(n, [&](long i) { return arr[i] != ref[i]; });
    if (mism)
    {
        println("[{}/{}] FAIL: {} element(s) changed by the refused re-chunks", variant_name(), label, mism);
        ++bad;
    }

    Array<double>  empty(0);
    const unsigned e0 = empty.chunk_size();
    empty.rechunk_to(48u);
    empty.set_chunk_size(96u);
    empty.rechunk_to(2u * cst_buffer_size);
    empty.set_chunk_size(2u * cst_buffer_size);
    if (empty.chunk_size() != e0)
    {
        println("[{}/{}] FAIL: empty array: geometry {} -> {}", variant_name(), label, e0, empty.chunk_size());
        ++bad;
    }
    if (active_chunk_size() != ambient)
    {
        println("[{}/{}] FAIL: ambient size {} -> {}", variant_name(), label, ambient, active_chunk_size());
        ++bad;
    }

    // The refusals left the array usable: a legal re-chunk, content intact.
    arr.rechunk_to(old_size * 2u);
    const long after = count_if_index(n, [&](long i) { return arr[i] != ref[i]; });
    if (arr.chunk_size() != old_size * 2u || after)
    {
        println("[{}/{}] FAIL: legal rechunk_to({}) afterwards: geometry {}, {} mismatch(es)", variant_name(), label,
                old_size * 2u, arr.chunk_size(), after);
        ++bad;
    }
    // Back to the case's start geometry for the next pass.
    set_active_chunk_size(ambient);
    set_thread_chunk_size(ambient);

    if (!bad)
    {
        println("[{}/{}] OK: 48/96/3/{} refused (filled and empty arrays), {} on the empty one, content and "
                "geometry kept, {} -> {} still works",
                variant_name(), label, old_size + 1u, 2u * cst_buffer_size, old_size, old_size * 2u);
    }
    return bad ? 1 : 0;
}

/// @brief set_active_chunk_size() refuses, in every build, a size that is not
///        a chunk geometry (0: the next DynamicChunkSize divided by 0; 100)
///        or wider than cst_buffer_size: the active size is kept, and an
///        array born afterwards has it.
int check_active_size_refused(const char* label)
{
    const std::string lab_s = pass_label(label);
    label                   = lab_s.c_str();
    const unsigned ambient  = active_chunk_size();

    int bad = 0;
    for (const unsigned s : {0u, 100u, 2u * cst_buffer_size})
    {
        set_active_chunk_size(s);
        if (active_chunk_size() != ambient)
        {
            println("[{}/{}] FAIL: set_active_chunk_size({}) accepted: active size {}", variant_name(), label, s,
                    active_chunk_size());
            ++bad;
            set_active_chunk_size(ambient);
        }
    }
    Array<double> arr(1000);
    if (arr.chunk_size() != ambient)
    {
        println("[{}/{}] FAIL: array born at {}, not {}", variant_name(), label, arr.chunk_size(), ambient);
        ++bad;
    }

    if (!bad)
    {
        println("[{}/{}] OK: set_active_chunk_size(0 / 100 / {}) refused, {} kept", variant_name(), label,
                2u * cst_buffer_size, ambient);
    }
    return bad ? 1 : 0;
}

/// @brief The index split of the run-time geometry (DynamicChunkSize::chunk_of
///        / offset_of: a shift and a mask for a power of two, a magic-number
///        multiplication otherwise) equals i / S and i % S, for every valid
///        size up to 4096 and 32- and 64-bit indexes. A power of two counted
///        as one by mistake (the flag was set from is_valid_chunk_size once it
///        admitted multiples of 64) sent 1000 to chunk 3 at 384.
int check_index_split(const char* label)
{
    const std::string lab_s = pass_label(label);
    label                   = lab_s.c_str();
    const unsigned ambient  = active_chunk_size();

    long bad   = 0;
    long count = 0;
    for (unsigned d = 32; d <= 4096; d += (d < 64 ? 32 : 64))
    {
        if (!is_valid_chunk_size(d))
        {
            continue;
        }
        DynamicChunkSize p;
        p.set_size(d);
        const auto one = [&](auto i)
        {
            using I = decltype(i);
            ++count;
            if (p.chunk_of(i) != static_cast<I>(i / static_cast<I>(d)) ||
                p.offset_of(i) != static_cast<I>(i % static_cast<I>(d)))
            {
                if (bad++ < 5)
                {
                    println("[{}/{}] FAIL: size {} index {}: chunk {} offset {}", variant_name(), label, d,
                            static_cast<long long>(i), static_cast<long long>(p.chunk_of(i)),
                            static_cast<long long>(p.offset_of(i)));
                }
            }
        };
        for (int i = 0; i <= 20000; ++i)
        {
            one(i);
        }
        for (long k = 1; k < 2000; ++k)
        {
            one(static_cast<int>((k * 1103515245L) & 0x7fffffffL));
            one(static_cast<long>(k) * 4611686018427387L);
        }
        one(std::numeric_limits<int>::max());
        one(std::numeric_limits<long>::max());
    }
    set_active_chunk_size(ambient); // set_size() moved the process-wide size
    set_thread_chunk_size(ambient);
    if (!bad)
    {
        println("[{}/{}] OK: chunk_of / offset_of == / and % for every valid size up to 4096 ({} indexes)",
                variant_name(), label, count);
    }
    return bad ? 1 : 0;
}

/// @brief Re-chunk through sizes that are multiples of 64 but do not divide
///        each other (256 -> 384 -> 192 -> 576 -> ...): the geometry follows
///        and the content survives every step. On v1 with OpenMP this drives
///        the refill's early release with old chunks shared by two new ones.
int check_multiple_of_64_chain(const char* label)
{
    const std::string lab_s = pass_label(label);
    label                   = lab_s.c_str();

    const long          n = 200000;
    Array<double>       arr(n);
    std::vector<double> ref(n);
    maybe_set_second_value(arr, 1.0);
    // Runs of 1000 (not a multiple of any size here): constant chunks and
    // chunks straddling two values at every geometry.
    for_each_index(n,
                   [&](long i)
                   {
                       const double v = (i / 1000) % 4 == 3 ? double(i % 7) : double((i / 1000) % 3);
                       arr[i]         = v;
                       ref[i]         = v;
                   });
    const unsigned ambient = active_chunk_size();

    int bad = 0;
    for (const unsigned s : {384u, 192u, 576u, 320u, 1024u, 448u, 512u, 256u})
    {
        arr.rechunk_to(s);
        const long mism = count_if_index(n, [&](long i) { return arr[i] != ref[i]; });
        if (arr.chunk_size() != s || mism)
        {
            println("[{}/{}] FAIL: rechunk_to({}): geometry {}, {} mismatch(es)", variant_name(), label, s,
                    arr.chunk_size(), mism);
            ++bad;
        }
    }
    // Writes and reads at the last geometry, across chunk boundaries.
    for (long i = 255; i < n; i += 4093)
    {
        arr[i] = ref[static_cast<std::size_t>(i)] = double(i % 11);
    }
    const long after = count_if_index(n, [&](long i) { return arr[i] != ref[i]; });
    if (after)
    {
        println("[{}/{}] FAIL: {} mismatch(es) after writes at {}", variant_name(), label, after, arr.chunk_size());
        ++bad;
    }
    set_active_chunk_size(ambient);
    set_thread_chunk_size(ambient);

    if (!bad)
    {
        println("[{}/{}] OK: 256 -> 384 -> 192 -> 576 -> 320 -> 1024 -> 448 -> 512 -> 256, content kept",
                variant_name(), label);
    }
    return bad ? 1 : 0;
}

namespace
{
/// @brief One case: its name (--case), the geometry it starts from, its body.
struct Case
{
    const char*          name;
    unsigned             start_size;
    std::function<int()> run;
};

/// @brief Declare @p size as the ambient geometry, to the arrays AND to the
///        allocators. set_active_chunk_size() only moves the ARRAY-side
///        geometry: without the registration, arrays report chunk_size() ==
///        size but draw their buffers from whatever size init_indexes()
///        primed, and every read past that width returns another chunk's data.
void set_start_geometry(unsigned size)
{
    set_active_chunk_size(size);
    set_thread_chunk_size(size);
    ParamChunk::register_chunk_size(size);
}
} // namespace

static int run_body(int argc, char** argv)
{
    // Positional arguments (the scale repro: N and the 64->target geometry are
    // configurable, to hunt a size-dependent rechunk crash under ASan without
    // the perf bench's weight):  [N (1e6)] [target size (1024)] [rounds (200)]
    // Flags: --case NAME, --mode serial|parallel|both (both), --list.
    std::vector<const char*> pos;
    const char*              only = nullptr;
    const char*              mode = "both";
    bool                     list = false;
    for (int a = 1; a < argc; ++a)
    {
        if (!std::strcmp(argv[a], "--case") && a + 1 < argc)
        {
            only = argv[++a];
        }
        else if (!std::strcmp(argv[a], "--mode") && a + 1 < argc)
        {
            mode = argv[++a];
        }
        else if (!std::strcmp(argv[a], "--list"))
        {
            list = true;
        }
        else
        {
            pos.push_back(argv[a]);
        }
    }
    const long     n_scale = !pos.empty() ? shrink_array::arg_number<long>(pos[0], 1000000L, "N") : 1000000L;
    const unsigned tgt     = pos.size() > 1 ? shrink_array::arg_number<unsigned>(pos[1], 1024u, "target size") : 1024u;
    const int      rounds  = pos.size() > 2 ? shrink_array::arg_number<int>(pos[2], 200, "rounds") : 200;
    const bool     run_ser = std::strcmp(mode, "parallel") != 0;
    const bool     run_par = std::strcmp(mode, "serial") != 0;
    if (!run_ser && !run_par)
    {
        println("unknown --mode '{}' (serial, parallel or both)", mode);
        return 2;
    }

    const long n = 1000000;
    // Each case declares a second_value present in its data (value -> val2
    // compression actually kicks in). On chunk2 this is REQUIRED; on 2A / v1
    // it is a no-op. The first three start at 64 (the scale case grows it to
    // the target, as the perf bench does), the others at 512.
    const std::vector<Case> cases = {
        // Coexistence first: it is the cheapest and the one the bench actually hits.
        {"coexistence", 64, [] { return check_coexistence<double>("coexistence-64-and-256", 200000L, 256u, 3); }},
        {"range-ops", 64, [] { return check_range_ops<double>("range-ops", 200000L); }},
        {"scale-64-to-tgt", 64,
         [&]
         {
             return check_invariance<double>(
                 "scale-64-to-tgt", n_scale, tgt, [](long i) -> double { return double((i / 5000) % 3); },
                 /*sv_value=*/2.0, /*post_rounds=*/rounds);
         }},
        {"mostly-constant", 512,
         [&]
         {
             return check_invariance<double>(
                 "mostly-constant", n, 2048, [](long i) -> double { return double((i / 5000) % 3); },
                 /*sv_value=*/2.0);
         }},
        {"sin-quantized", 512,
         [&]
         {
             return check_invariance<double>(
                 "sin-quantized", n, 4096,
                 [](long i) -> double { return std::floor(std::sin(double(i) * 0.001) * 4.0); }, /*sv_value=*/1.0);
         }},
        {"shrink-to-smaller", 512,
         [&]
         {
             return check_invariance<double>(
                 "shrink-to-smaller", n, 256, [](long i) -> double { return double((i / 1000) % 2); },
                 /*sv_value=*/1.0);
         }},
        // 4096, not 8192: rechunk_to() refuses anything above cst_buffer_size
        // (CHUNK_DYN_MAX_SIZE, 4096 by default) because the shared cst_val1/cst_val2
        // buffers are sized to that ceiling and a wider chunk would read them out of
        // bounds. Asking for 8192 makes this case fail ("geometry 512 -> 512",
        // with rechunk_to's refusal on stderr). Raise the target here only
        // together with -DCHUNK_DYN_MAX_SIZE.
        {"all-same", 512,
         [&]
         {
             return check_invariance<double>(
                 "all-same", n, 4096, [](long) -> double { return 42.0; },
                 /*sv_value=*/42.0);
         }},
        {"all-different", 512,
         []
         {
             return check_invariance<double>(
                 "all-different", 100000, 1024, [](long i) -> double { return double(i); }, /*sv_value=*/1.0);
         }},
        // Growth 512 -> 2048 with val2: targets the tab_m_ptr_cst_val2 queue
        // (chunk2) + the second_value round-trip.
        {"val2-grow", 512,
         [&]
         {
             return check_invariance<double>(
                 "val2-grow", n, 2048, [](long i) -> double { return double((i / 5000) % 3); }, /*sv_value=*/2.0);
         }},
        // A size that is neither a power of two nor a multiple of 64: refused, nothing changed.
        {"invalid-size-refused", 512, [] { return check_non_pow2_refused("invalid-size-refused"); }},
        // set_active_chunk_size: 0, 100 and a size over cst_buffer_size refused in every build.
        {"active-size-refused", 512, [] { return check_active_size_refused("active-size-refused"); }},
        {"multiple-of-64-chain", 256, [] { return check_multiple_of_64_chain("multiple-of-64-chain"); }},
        {"index-split", 256, [] { return check_index_split("index-split"); }},
        // Shrink with val2 (queue never read past S', must pass).
        {"val2-shrink", 512,
         [&]
         {
             return check_invariance<double>(
                 "val2-shrink", n, 256, [](long i) -> double { return double((i / 1000) % 2); },
                 /*sv_value=*/1.0);
         }},
        // The v1 refill's SLICED path (fast-alloc OpenMP builds): it slices
        // only from 8192 new chunks on (n_new / 4096 slices, up to 16), which
        // no case above reaches. Every slice boundary drains the old geometry's
        // parked blocks and recycles the slabs they empty into the next slice's
        // allocations (CHUNK_RECHUNK_RECYCLE): stale buffers that must come
        // back zero-filled where the data is zero, and exact elsewhere. Mostly
        // distinct values (no constant chunk to fold) with zero spans; grown
        // (4 slices) and shrunk (8 slices), then rewritten and recompressed.
        {"sliced-grow", 64,
         []
         {
             return check_invariance<double>(
                 "sliced-grow-64-to-256", 4L << 20, 256,
                 [](long i) -> double { return (i / 3000) % 4 == 0 ? 0.0 : double(i % 977) + 0.5; },
                 /*sv_value=*/0.5, /*post_rounds=*/2);
         }},
        {"sliced-shrink", 512,
         []
         {
             return check_invariance<double>(
                 "sliced-shrink-512-to-64", 2L << 20, 64,
                 [](long i) -> double { return (i / 5000) % 3 == 0 ? 0.0 : double(i % 613) + 0.25; },
                 /*sv_value=*/0.25, /*post_rounds=*/2);
         }},
    };

    if (list)
    {
        for (const Case& c : cases)
        {
            println("{}", c.name);
        }
        return 0;
    }
    if (only)
    {
        bool known = false;
        for (const Case& c : cases)
        {
            known = known || !std::strcmp(c.name, only);
        }
        if (!known)
        {
            println("unknown --case '{}' (--list gives the names)", only);
            return 2;
        }
    }

    ParamChunk::init_indexes(true);

    println("=== rechunk_validation [variant = {}]  N_scale={}  64->{}  case={}  mode={} ===", variant_name(), n_scale,
            tgt, only ? only : "all", mode);

    int      fails    = 0;
    unsigned geometry = 0;
    for (const Case& c : cases)
    {
        if (only && std::strcmp(c.name, only) != 0)
        {
            continue;
        }
        if (c.start_size != geometry)
        {
            set_start_geometry(c.start_size);
            geometry = c.start_size;
        }
        if (run_ser)
        {
            g_parallel = false;
            fails += c.run();
        }
        if (run_par)
        {
            g_parallel = true;
            fails += c.run();
        }
    }

    println("\n[{}] {}", variant_name(),
            fails == 0 ? "=== ALL INVARIANCE CHECKS PASSED ===" : "=== INVARIANCE FAILURES ===");
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

} // namespace validation::rechunk_validation
