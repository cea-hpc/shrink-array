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

/// @file bench_phase_compress.cpp
/// @brief The compression dispatch. See bench_phases.h.

#include <cstdio> // stdout, std::fflush
#include <print>  // std::println

#include "bench_phases.h" // bench_phase_compress_all (defined here), BenchArrays

namespace
{
/// @brief tab.compress() in a shrink build, nothing in the std::vector baseline.
///
/// compress() exists on the ArrayChunk* alone, and `if constexpr` in a plain
/// function still type-checks its discarded branch -- but the body of a
/// TEMPLATE is genuinely not instantiated, so routing the thirteen calls through
/// one abbreviated function template lets the guard below be `if constexpr`.
void compress_one(auto& tab)
{
    if constexpr (chunk_config::is_shrink_enabled)
    {
        tab.compress();
    }
}

/// @brief One line naming the array about to be compressed, flushed (it was
///        std::endl): one statement, so it still fits under `omp single`.
///        Unused in the std::vector build (nothing is compressed there).
[[maybe_unused]] void announce(const char* name)
{
    std::println("{}", name);
    std::fflush(stdout);
}
} // namespace

void bench_phase_compress_all(BenchArrays& bench_arr)
{
    // The OMPT bracketing lived in a lambda of main() whose entire body it was,
    // around this call. It belongs to the compression: a caller cannot forget
    // what it does not have to write.
    //
    // Inline in main(), this block sat under #ifdef NEED_COMPRESSION /
    // SHRINK_ARRAY depending on the call site. The outer guard is kept -- as an
    // `if constexpr` now -- so the baseline build still does strictly nothing
    // here rather than running the parallel region and printing thirteen names.
    if constexpr (chunk_config::is_shrink_enabled)
    {
        chunk_config::OMPTcc<chunk_config::has_ompt>::update(false);

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

        // Compress the queued candidates first, from outside the region: each
        // compress() below runs in an `omp single`, where compress_int cannot
        // drain (it would nest a parallel region), so a candidate still queued
        // by the serial writes of the iteration (fill_range / copy_range) was
        // reported as a miss its self-heal had to compress.
        if constexpr (build_config::use_openmp && !chunk_config::have_aggressive_compression)
        {
            auto& ens = shrink_array::SingletonChk<shrink_array::AllTypeChunk>::getRef();
            if (ens.have_chunk_to_compress_potentially())
            {
                ens.compress_chunk_to_compress_potentially(true);
            }
        }

#pragma omp parallel
        {
#pragma omp single nowait
            announce("  tab1");
            compress_one(tab1);
#pragma omp single nowait
            announce("  tab2");
            compress_one(tab2);
#pragma omp single nowait
            announce("  tab3");
            compress_one(tab3);
#pragma omp single nowait
            announce("  tab4");
            compress_one(tab4);
#pragma omp single nowait
            announce("  tab5");
            compress_one(tab5);
#pragma omp single nowait
            announce("  tab_int");
            compress_one(tab_int);
#pragma omp single nowait
            announce("  tab_bool");
            compress_one(tab_bool);
#pragma omp single nowait
            announce("  tab_bool_bits");
            compress_one(tab_bool_bits);
#pragma omp single nowait
            announce(" tab_complex");
            compress_one(tab_complex);
#pragma omp single nowait
            announce(" tab_float");
            compress_one(tab_float);
#pragma omp single nowait
            announce(" tab_char");
            compress_one(tab_char);
#pragma omp single nowait
            announce(" tab_short");
            compress_one(tab_short);
#pragma omp single nowait
            announce(" tab_long");
            compress_one(tab_long);
#pragma omp single nowait
            announce(" tab_ulong");
            compress_one(tab_ulong);
#pragma omp single nowait
            announce(" tab_quant");
            compress_one(tab_quant);
            if (bench_arr.with_struct) // --no_struct
            {
#pragma omp single nowait
                announce(" tab_point");
                compress_one(tab_point);
#pragma omp single nowait
                announce(" tab_cell");
                compress_one(tab_cell);
            }
            if (bench_arr.with_mask) // --no_mask
            {
#pragma omp single nowait
                announce(" tab_mask");
                compress_one(tab_mask);
            }
        }

        chunk_config::OMPTcc<chunk_config::has_ompt>::update(true);
    }
    else
    {
        (void)bench_arr; // std::vector baseline build: nothing to compress
    }
}
