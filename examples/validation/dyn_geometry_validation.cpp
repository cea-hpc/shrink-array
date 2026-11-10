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
// Dynamic-geometry validation harness. Variant-agnostic: tests zero / zero_one_value /
// zero_any depending on the compile-time macro (CHUNK_VARIANT_ZERO*), like
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

#include <string_view> // std::string_view (to_string)
#include <type_traits> // std::conditional_t
#include <vector>      // std::vector

#include "chunk/internal/chunk_traits.h" // forward decls + default IntType = int
#include "chunk/zero_array.h"            // zero::Array
#include "chunk/zero_one_value_array.h"  // zero_one_value::Array
#include "chunk/zero_any_array.h"        // zero_any::Array
#include "chunk/param_chunk.h"           // ParamChunk::register_chunk_size, init_indexes
#include "chunk/internal/chunk_size_policy.h" // cst_buffer_size, set_active_chunk_size, set_thread_chunk_size, chunk_size_error
#include "chunk/config.h"                     // chunk_config::use_zero*
#include "chunk/internal/env_parse.h"         // env_number / arg_number (checked, not atoi)
#include "chunk/print.h"                      // shrink_array::println
#include "chunk/log.h"                        // logging::set_sink (the refusals' warnings)

using namespace shrink_array;

#include "chunk/internal/zero_chunk.h" // release_chunk_pools, destroy_block_chunk
#include "validation_harness.h"        // self_command; run() is called by validation_driver.cpp

namespace validation::dyn_geometry_validation
{

// --- Active variant selection (same shape as rechunk_validation.cpp) ---------
static_assert(chunk_config::use_zero || chunk_config::use_zero_one_value || chunk_config::use_zero_any,
              "dyn_geometry_validation must be built with CHUNK_VARIANT_ZERO, _ZERO_ONE_VALUE or _ZERO_ANY");

static_assert(build_config::chunk_dynamic,
              "dyn_geometry_validation is meaningless without CHUNK_DYNAMIC: with the "
              "static geometry there is only ever one chunk size to get right");

template <typename T>
using Array =
    std::conditional_t<chunk_config::use_zero_any,
                       zero_any::Array<T>,
                       std::conditional_t<chunk_config::use_zero_one_value, zero_one_value::Array<T>, zero::Array<T>>>;

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

// set_second_value only exists on zero_one_value / zero_any; no-op elsewhere. Required on
// zero_one_value before any write (its candidate list is keyed by second_value), so it
// is done here for the same reason rechunk_validation does it.
template <typename Arr, typename Val>
void maybe_set_second_value(Arr& arr, const Val& value)
{
    if constexpr (requires(Arr probe) { probe.set_second_value(value); })
    {
        arr.set_second_value(value);
    }
}

// The pattern is piecewise-constant with long runs: it compresses well (so the
// constant/materialised transitions are exercised) while making a wrong read
// obvious — a mis-sized buffer returns the value of a DIFFERENT run, not a
// neighbouring one.
static double pattern(long index)
{
    return double((index / 5000) % 3);
}

/**
 * @brief Write a whole array at chunk size @p csz, read it back, compare.
 * @return 0 if every element round-trips, 1 otherwise (details on stdout).
 */
static int round_trip_at(unsigned int csz, long count)
{
    // BOTH halves are needed, and this is the property under test: the array
    // side (set_active_chunk_size) and the allocator side (register_chunk_size).
    set_active_chunk_size(csz);
    set_thread_chunk_size(csz);
    ParamChunk::register_chunk_size(csz);

    Array<double> arr(count);
    maybe_set_second_value(arr, 2.0);

    if (arr.chunk_size() != csz)
    {
        println("[{}/size={}] FAIL: array built at chunk_size={}", variant_name(), csz, arr.chunk_size());
        return 1;
    }

    std::vector<double> ref(count);
    for (long i = 0; i < count; ++i)
    {
        const double value = pattern(i);
        arr[i]             = value;
        ref[i]             = value;
    }

    long bad   = 0;
    long first = -1;
    for (long i = 0; i < count; ++i)
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
        println("[{}/size={}] FAIL: {}/{} mismatches, first at i={} "
                "(chunk #{}, offset {}) got={:g} expected={:g}",
                variant_name(), csz, bad, count, first, first / (long)csz, first % (long)csz, (double)arr[first],
                ref[first]);
        return 1;
    }

    println("[{}/size={}] OK: {} elements round-trip", variant_name(), csz, count);
    return 0;
}

// The refusals the round-trips stop short of: a size wider than
// cst_buffer_size, or not a power of two nor a multiple of 64. Each setter
// refuses it with one warning on the rechunk channel and keeps the geometry
// (set_active_chunk_size, an array's rechunk_to); chunk_size_error says why,
// to_string in words. The coverage report had these refusals at 0 %.
static int g_rechunk_warnings = 0;

static void count_rechunk_warnings(const logging::Record& record, void* /*user*/) noexcept
{
    if (record.level == logging::Level::warn && record.channel == logging::Channel::rechunk)
    {
        ++g_rechunk_warnings;
    }
}

static int refused_sizes()
{
    int        fails = 0;
    const auto fail  = [&fails](const char* what)
    {
        ++fails;
        println("  FAIL refused sizes: {}", what);
    };
    const unsigned int too_wide = 2 * cst_buffer_size;

    if (chunk_size_error(too_wide) != ChunkSizeError::exceeds_buffer)
    {
        fail("chunk_size_error(2 x cst_buffer_size) is not exceeds_buffer");
    }
    if (chunk_size_error(100) != ChunkSizeError::not_pow2_or_x64 ||
        chunk_size_error(0) != ChunkSizeError::not_pow2_or_x64)
    {
        fail("chunk_size_error(100) / (0) is not not_pow2_or_x64");
    }
    if (chunk_size_error(cst_buffer_size).has_value())
    {
        fail("chunk_size_error(cst_buffer_size) refuses the widest size allowed");
    }
    if (std::string_view(to_string(ChunkSizeError::exceeds_buffer)).find("CHUNK_DYN_MAX_SIZE") ==
        std::string_view::npos)
    {
        fail("to_string(exceeds_buffer) does not name CHUNK_DYN_MAX_SIZE");
    }

    logging::set_sink(count_rechunk_warnings);
    const unsigned int before = active_chunk_size();
    g_rechunk_warnings        = 0;
    set_active_chunk_size(too_wide);
    if (active_chunk_size() != before || g_rechunk_warnings != 1)
    {
        fail("set_active_chunk_size(2 x cst_buffer_size): the geometry moved, or not one warning");
    }

    set_active_chunk_size(64);
    set_thread_chunk_size(64);
    ParamChunk::register_chunk_size(64);
    {
        Array<double> arr(10000);
        arr[17]            = 1.5;
        g_rechunk_warnings = 0;
        arr.rechunk_to(too_wide);
        if (arr.chunk_size() != 64 || g_rechunk_warnings != 1 || arr[17] != 1.5)
        {
            fail("rechunk_to(2 x cst_buffer_size): the array changed, or not one warning");
        }
        g_rechunk_warnings = 0;
        arr.rechunk_to(100);
        if (arr.chunk_size() != 64 || g_rechunk_warnings != 1)
        {
            fail("rechunk_to(100): the array changed, or not one warning");
        }
    }
    logging::set_sink(nullptr);

    println("[{}] refused sizes: {}", variant_name(), fails == 0 ? "OK" : "FAILED");
    return fails;
}

static int run_body(int argc, char** argv)
{
    const long count = argc > 1 ? shrink_array::arg_number<long>(argv[1], 200000L, "N") : 200000L;

    ParamChunk::init_indexes(true);

    println("=== dyn_geometry_validation [variant = {}]  N={} ===", variant_name(), count);

    int fails = 0;

    // Up to cst_buffer_size: past that ceiling the shared constant buffers are
    // too narrow and the library refuses the geometry by design: refused_sizes()
    // tests that refusal, apart.
    for (unsigned int csz = 64; csz <= shrink_array::cst_buffer_size; csz <<= 1)
    {
        fails += round_trip_at(csz, count);
    }
    fails += refused_sizes();

    println("\n[{}] {}", variant_name(),
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
