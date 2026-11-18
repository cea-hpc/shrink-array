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

/// @file bench_teardown.cpp
/// @brief Array release, final report and library shutdown. See bench_phases.h.

#include <iostream> // std::cout

#include "bench_config.h"              // check_compression, and chunk_config:: through bench_prelude.h
#include "bench_phases.h"              // bench_teardown (defined here), BenchArrays
#include "bench_stl_checks.h"          // run_stl_self_checks
#include "chunk/internal/zero_chunk.h" // clear_all_list_block_free_thr, delete_chunk_allocator
#include "chunk/runtime.h"             // Runtime::chunk_stats
#include "chunk/print.h"               // shrink_array::println (the array_conn note)

// As in shrink_bench.cpp, where this body comes from.
using namespace shrink_array;

void bench_teardown(BenchArrays& bench_arr, bool test_stl)
{
    // Short local names for the arrays (see bench_phase_fill.cpp).
    auto& arr1 = bench_arr.arr1;
    auto& arr2 = bench_arr.arr2;
    auto& arr3 = bench_arr.arr3;
    auto& arr4 = bench_arr.arr4;
    auto& arr5 = bench_arr.arr5;

    auto& array_int       = bench_arr.array_int;
    auto& array_bool      = bench_arr.array_bool;
    auto& array_bool_bits = bench_arr.array_bool_bits;
    auto& array_complex   = bench_arr.array_complex;
    auto& array_float     = bench_arr.array_float;
    auto& array_short     = bench_arr.array_short;
    auto& array_char      = bench_arr.array_char;
    auto& array_long      = bench_arr.array_long;
    auto& array_ulong     = bench_arr.array_ulong;
    auto& array_quant     = bench_arr.array_quant;
    auto& array_point     = bench_arr.array_point;
    auto& array_cell      = bench_arr.array_cell;
    auto& array_mask      = bench_arr.array_mask;
    auto& array_conn      = bench_arr.array_conn;

    [[maybe_unused]] auto& bench_arrays = bench_arr;

    // The self-checks run just before the release, as they did in main().
    run_stl_self_checks(test_stl);

    arr1.clear();
    arr2.clear();
    arr3.clear();
    arr4.clear();
    arr5.clear();

    array_int.clear();
    array_bool.clear();
    array_bool_bits.clear();

    array_complex.clear();

    array_short.clear();
    array_float.clear();
    array_char.clear();

    array_long.clear();
    array_ulong.clear();
    array_quant.clear();
    array_point.clear();
    array_cell.clear();
    array_mask.clear();
    // Zero builds: array_conn is a zero::Array (bench_array.h), and a connectivity
    // has no all-zero chunk -- its census line, printed by clear(), finds
    // nothing compressed. Said before it, so the line is not read as a failure.
    if constexpr (chunk_config::use_zero)
    {
        if (bench_arr.with_conn)
        {
            shrink_array::println(
                "[array_conn] zero build: a connectivity has no all-zero chunk, nothing to compress here; "
                "pattern::Array compresses its motifs in the zero_or_value / zero_any builds");
        }
    }
    array_conn.clear();

    // Chunk instrumentation counters (CHUNK_STATS). Printed HERE, after the
    // array_*.clear() above: clear() runs clear_core(), which is what feeds
    // teardown_clear -- reporting before them always showed 0. Independent of the
    // bench's own `stats` mode (that one is application-level and check-only);
    // these are library counters we want in perf runs too. chunk_stats() is
    // declared unconditionally and compiles to an empty body without CHUNK_STATS,
    // so a plain if constexpr does the job -- no #ifdef needed.
    if constexpr (chunk_config::stats)
    {
        shrink_array::Runtime::chunk_stats();
    }

    // Entries get_info_second_value() scans per call, per element type. It is a
    // library counter like the ones above, so it prints in perf runs too -- the
    // mode whose profile put 34% of a 64-thread zero_or_value run in that function,
    // against 0.37% under libgomp.
    //
    // Printed AFTER the array_*.clear() calls on purpose: `vec` holds one entry
    // per live array, so a count taken mid-run cannot tell five live arrays
    // from five leaked entries. Here every array is gone, and anything left is
    // an entry that outlived the array that put it there -- erase_inf_array_by_ptr()
    // acts on map_inf_array_, and nothing erases from `vec`.
    //
    // A handful: the scan is short and the cost is the CALL RATE, which caching
    // the per-array-invariant result would fix. Hundreds: the list grew.
    // zero_or_value only -- zero_any carries its second value in the pointer and has no
    // such map.
    if constexpr (chunk_config::use_zero_or_value)
    {
        std::cout << "\n[share_info] entries still mapped after clear():"
                  << "  double " << shrink_array::zero_or_value::Proxy<double>::array_registry->share_info_entries()
                  << "  Int " << shrink_array::zero_or_value::Proxy<int>::array_registry->share_info_entries()
                  << "  float " << shrink_array::zero_or_value::Proxy<float>::array_registry->share_info_entries()
                  << "  short " << shrink_array::zero_or_value::Proxy<short>::array_registry->share_info_entries()
                  << "\n";
    }

    // bench_phase_compress_all() is declared in bench_phases.h and defined in
    // every variant (a no-op in the std::vector build), so this one needs no
    // directive -- unlike the block just below.
    if constexpr (chunk_config::is_shrink_enabled && check_compression)
    {
        std::cout << "\nCheck compression  : " << "\n";
        bench_phase_compress_all(bench_arrays);
    }

    // No #ifdef: both names below are declared unguarded (type_state_registry.h
    // and zero_chunk.h) and defined in EVERY variant library -- the symbols are
    // in libchunk_vector.a too -- so the discarded branch both type-checks and
    // links in the std::vector build.
    if constexpr (chunk_config::is_shrink_enabled)
    {
        // No #ifdef _OPENMP: since ChunkPoolParallelInterface
        // (zero_chunk.h) both verbs exist in every build, each a no-op in the
        // model where it has nothing to do -- so this pair is EXACTLY what ran
        // before on either side. Deliberately not release_chunk_pools(): that
        // also calls delete_list_block_chunk_free_itself(), which this example
        // has never called (pool_accounting_validation and compress_roundtrip_validation do).
        // Spelling the pair out keeps the behaviour identical rather than
        // quietly adding a teardown step.
        clear_all_list_block_free_thr();
        delete_chunk_allocator();
    }

    // g_need_flush_deferred is deliberately NOT cleared here any more.
    // Its only readers are on_parallel_end / on_barrier_end (manage_ompt.cpp),
    // and this store was the last statement of the last function main() calls:
    // measured under gdb on an OMPT build (clang, -DHAVE_OMPT, both callbacks
    // present in the binary), neither finalize_parallel_phase nor
    // finalize_barrier_phase is reached between here and normal exit. Restore
    // the store if teardown ever grows a parallel region or a barrier AFTER
    // delete_chunk_allocator() -- the flag is what would keep the callbacks off
    // the freed allocator.
}
