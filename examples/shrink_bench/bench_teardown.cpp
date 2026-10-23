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

#include "bench_config.h"      // check_compression, and chunk_config:: through bench_prelude.h
#include "bench_phases.h"      // bench_teardown (defined here), BenchArrays
#include "bench_stl_checks.h"  // run_stl_self_checks
#include "chunk/zero_chunk.h"  // clear_all_list_block_free_thr, delete_chunk_allocator
#include "chunk/param_chunk.h" // ParamChunk::chunk_stats

// As in shrink_bench.cpp, where this body comes from.
using namespace shrink_array;

void bench_teardown(BenchArrays& bench_arr, bool test_stl)
{
    // Aliases so the body below is VERBATIM (see bench_phase_fill.cpp).
    auto& tab1 = bench_arr.tab1;
    auto& tab2 = bench_arr.tab2;
    auto& tab3 = bench_arr.tab3;
    auto& tab4 = bench_arr.tab4;
    auto& tab5 = bench_arr.tab5;

    auto& tab_int       = bench_arr.tab_int;
    auto& tab_bool      = bench_arr.tab_bool;
    auto& tab_bool_bits = bench_arr.tab_bool_bits;
    auto& tab_complex   = bench_arr.tab_complex;
    auto& tab_float     = bench_arr.tab_float;
    auto& tab_short     = bench_arr.tab_short;
    auto& tab_char      = bench_arr.tab_char;
    auto& tab_long      = bench_arr.tab_long;
    auto& tab_ulong     = bench_arr.tab_ulong;
    auto& tab_quant     = bench_arr.tab_quant;
    auto& tab_point     = bench_arr.tab_point;
    auto& tab_cell      = bench_arr.tab_cell;
    auto& tab_mask      = bench_arr.tab_mask;

    [[maybe_unused]] auto& bench_arrays = bench_arr;

    // The self-checks run just before the release, as they did in main().
    run_stl_self_checks(test_stl);

    tab1.clear();
    tab2.clear();
    tab3.clear();
    tab4.clear();
    tab5.clear();

    tab_int.clear();
    tab_bool.clear();
    tab_bool_bits.clear();

    tab_complex.clear();

    tab_short.clear();
    tab_float.clear();
    tab_char.clear();

    tab_long.clear();
    tab_ulong.clear();
    tab_quant.clear();
    tab_point.clear();
    tab_cell.clear();
    tab_mask.clear();

    // Chunk instrumentation counters (CHUNK_STATS). Printed HERE, after the
    // tab_*.clear() above: clear() runs clear_core(), which is what feeds
    // teardown_clear -- reporting before them always showed 0. Independent of the
    // bench's own `stats` mode (that one is application-level and check-only);
    // these are library counters we want in perf runs too. chunk_stats() is
    // declared unconditionally and compiles to an empty body without CHUNK_STATS,
    // so a plain if constexpr does the job -- no #ifdef needed.
    if constexpr (chunk_config::stats)
    {
        shrink_array::ParamChunk::chunk_stats();
    }

    // Entries get_info_second_value() scans per call, per element type. It is a
    // library counter like the ones above, so it prints in perf runs too -- the
    // mode whose profile put 34% of a 64-thread zero_one_value run in that function,
    // against 0.37% under libgomp.
    //
    // Printed AFTER the tab_*.clear() calls on purpose: `vec` holds one entry
    // per live array, so a count taken mid-run cannot tell five live arrays
    // from five leaked entries. Here every array is gone, and anything left is
    // an entry that outlived the array that put it there -- erase_inf_array_by_ptr()
    // acts on map_inf_array, and nothing erases from `vec`.
    //
    // A handful: the scan is short and the cost is the CALL RATE, which caching
    // the per-array-invariant result would fix. Hundreds: the list grew.
    // zero_one_value only -- zero_any carries its second value in the pointer and has no
    // such map.
    if constexpr (chunk_config::use_zero_one_value)
    {
        std::cout << "\n[share_info] entries still mapped after clear():"
                  << "  double " << shrink_array::zero_one_value::Proxy<double>::share_info_array->share_info_entries()
                  << "  Int " << shrink_array::zero_one_value::Proxy<int>::share_info_array->share_info_entries()
                  << "  float " << shrink_array::zero_one_value::Proxy<float>::share_info_array->share_info_entries()
                  << "  short " << shrink_array::zero_one_value::Proxy<short>::share_info_array->share_info_entries()
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

    // No #ifdef: both names below are declared unguarded (all_type_chunk.h
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
        // has never called (pool_accounting and compress_roundtrip do).
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
