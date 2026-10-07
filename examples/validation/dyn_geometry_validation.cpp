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
// Dynamic-geometry validation harness. Variant-agnostic: tests chunk / chunk2 /
// chunk2a depending on the compile-time macro (CHUNK / CHUNK2 / CHUNK2A), like
// rechunk_validation.cpp and shrink_bench.cpp.
// ============================================================================
//
// Guards ONE property, at every runtime chunk size: an array must read back
// exactly what was written into it. No re-chunk, no compression pass driven by
// hand, no second phase — write, read, compare. That deliberate narrowness is
// the point: when this fails, the cause is in the geometry/allocation pairing
// and nowhere else.
//
// WHAT IT CATCHES
//
//   A chunk's geometry and its BUFFER come from two different places:
//     - set_active_chunk_size(n) moves the array-side geometry, so
//       arr.chunk_size() reports n and operator[] splits indices at n;
//     - the buffer comes from the per-size allocator pool, which only knows the
//       sizes declared to it through ParamChunk::register_chunk_size().
//   Declare one without the other and the two disagree silently: chunks claim n
//   elements while holding a buffer sized for a different geometry, so every
//   read past that width returns a neighbouring chunk's data. Nothing asserts,
//   nothing crashes — the values are simply wrong.
//
//   The failure looks like a re-chunk bug when the only harness exercising
//   several geometries is rechunk_validation; it is about neither re-chunk
//   nor any one variant.
//
// WHAT IT DOES NOT COVER
//
//   Only the supported usage is asserted (register, then use). The unregistered
//   path is left untested on purpose: pinning down today's corruption pattern
//   would turn a bug into a specification.
//
// Compile OMP or non-OMP, dyn, check. Exit 0 = every size round-trips.

#include <cstdio>      // printf
#include <type_traits> // std::conditional_t
#include <vector>      // std::vector
#include <print>       // std::println

#include "chunk/chunk_traits.h"               // forward decls + default ind_type = int
#include "chunk/array_chunk.h"                // ArrayChunk
#include "chunk/array_chunk2.h"               // ArrayChunk2
#include "chunk/array_chunk2A.h"              // ArrayChunk2A
#include "chunk/param_chunk.h"                // ParamChunk::register_chunk_size, init_indexes
#include "chunk/internal/chunk_size_policy.h" // cst_buffer_size, set_active_chunk_size, set_thread_chunk_size
#include "chunk/config.h"                     // chunk_config::use_chunk*
#include "chunk/internal/env_parse.h"         // env_number / arg_number (checked, not atoi)

using namespace shrink_array;

#include "chunk/chunk_std.h"    // release_chunk_pools, destroy_block_chunk
#include "validation_harness.h" // self_command; run() is called by validation_driver.cpp

namespace validation::dyn_geometry_validation
{

// --- Active variant selection (same shape as rechunk_validation.cpp) ---------
static_assert(chunk_config::use_chunk || chunk_config::use_chunk2 || chunk_config::use_chunk2a,
              "dyn_geometry_validation must be built with CHUNK, CHUNK2 or CHUNK2A");

static_assert(build_config::chunk_dynamic,
              "dyn_geometry_validation is meaningless without CHUNK_DYNAMIC: with the "
              "static geometry there is only ever one chunk size to get right");

template <typename T>
using Array = std::conditional_t<chunk_config::use_chunk2a,
                                 ArrayChunk2A<T>,
                                 std::conditional_t<chunk_config::use_chunk2, ArrayChunk2<T>, ArrayChunk<T>>>;

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

// set_second_value only exists on chunk2 / chunk2a; no-op elsewhere. Required on
// chunk2 before any write (its candidate list is keyed by second_value), so it
// is done here for the same reason rechunk_validation does it.
template <typename Arr, typename Val>
void maybe_set_second_value(Arr& a, const Val& v)
{
    if constexpr (requires(Arr x) { x.set_second_value(v); })
    {
        a.set_second_value(v);
    }
}

// The pattern is piecewise-constant with long runs: it compresses well (so the
// constant/materialised transitions are exercised) while making a wrong read
// obvious — a mis-sized buffer returns the value of a DIFFERENT run, not a
// neighbouring one.
static double pattern(long i)
{
    return double((i / 5000) % 3);
}

/**
 * @brief Write a whole array at chunk size @p csz, read it back, compare.
 * @return 0 if every element round-trips, 1 otherwise (details on stdout).
 */
static int round_trip_at(unsigned int csz, long N)
{
    // BOTH halves are needed, and this is the property under test: the array
    // side (set_active_chunk_size) and the allocator side (register_chunk_size).
    set_active_chunk_size(csz);
    set_thread_chunk_size(csz);
    ParamChunk::register_chunk_size(csz);

    Array<double> arr(N);
    maybe_set_second_value(arr, 2.0);

    if (arr.chunk_size() != csz)
    {
        std::println("[{}/size={}] FAIL: array built at chunk_size={}", variant_name(), csz, arr.chunk_size());
        return 1;
    }

    std::vector<double> ref(N);
    for (long i = 0; i < N; ++i)
    {
        const double v = pattern(i);
        arr[i]         = v;
        ref[i]         = v;
    }

    long bad   = 0;
    long first = -1;
    for (long i = 0; i < N; ++i)
    {
        if (arr[i] != ref[i])
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
        // The offset of the first mismatch is the useful number: it is the width
        // the buffer actually had, which names the geometry the allocator used.
        std::println("[{}/size={}] FAIL: {}/{} mismatches, first at i={} "
                     "(chunk #{}, offset {}) got={:g} expected={:g}",
                     variant_name(), csz, bad, N, first, first / (long)csz, first % (long)csz, (double)arr[first],
                     ref[first]);
        return 1;
    }

    std::println("[{}/size={}] OK: {} elements round-trip", variant_name(), csz, N);
    return 0;
}

static int run_body(int argc, char** argv)
{
    const long N = argc > 1 ? shrink_array::arg_number<long>(argv[1], 200000L, "N") : 200000L;

    ParamChunk::init_indexes(true);

    std::println("=== dyn_geometry_validation [variant = {}]  N={} ===", variant_name(), N);

    int fails = 0;

    // Up to cst_buffer_size: past that ceiling the shared constant buffers are
    // too narrow and the library refuses the geometry by design, so a larger
    // size here would test the refusal, not the round-trip.
    for (unsigned int csz = 64; csz <= shrink_array::cst_buffer_size; csz <<= 1)
    {
        fails += round_trip_at(csz, N);
    }

    std::println("\n[{}] {}", variant_name(),
                 fails == 0 ? "=== ALL GEOMETRIES ROUND-TRIP ===" : "=== GEOMETRY FAILURES ===");
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

} // namespace validation::dyn_geometry_validation
