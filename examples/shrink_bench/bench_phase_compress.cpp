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

#include <iostream> // std::cout

#include "bench_phases.h" // bench_phase_compress_all (defined here), BenchArrays

namespace
{
/// @brief tab.compress() in a shrink build, nothing in the std::vector baseline.
///
/// compress() exists on the zero::Array* alone, and `if constexpr` in a plain
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
} // namespace

void bench_phase_compress_all(BenchArrays& bench_arr)
{
    // The OMPT bracketing belongs to the compression: a caller cannot forget
    // what it does not have to write. The outer guard keeps the baseline build
    // doing strictly nothing here rather than running the parallel region and
    // printing thirteen names.
    if constexpr (chunk_config::is_shrink_enabled)
    {
        chunk_config::OMPTcc<chunk_config::has_ompt>::update(false);

        // Short local names for the arrays (see bench_phase_fill.cpp).
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
        auto& tab_conn      = bench_arr.tab_conn;

        // Compress the queued candidates first, from outside the region: each
        // compress() below runs in an `omp single`, where compress_int cannot
        // drain (it would nest a parallel region), so a candidate still queued
        // by the serial writes of the iteration (fill_range / copy_range) was
        // reported as a miss its self-heal had to compress.
        if constexpr (build_config::threaded)
        {
            if (!chunk_config::aggressive_compression()) // run-time switch: chunk/config.h
            {
                auto& ens = shrink_array::SingletonChk<shrink_array::AllTypeChunk>::get_ref();
                if (ens.have_chunk_to_compress_potentially())
                {
                    ens.compress_chunk_to_compress_potentially(true);
                }
            }
        }

        // Every array's pass, each announced by say(name). Team regions (OpenMP,
        // std): every thread of the region calls compress(), a team-wide pass
        // that shares the chunks out, and one of them prints. TBB tasks are no
        // team of the library: each array is compressed once, at this serial
        // point, where its pass runs on the library's own team.
        const auto all = [&](auto&& say)
        {
            say("  tab1");
            compress_one(tab1);
            say("  tab2");
            compress_one(tab2);
            say("  tab3");
            compress_one(tab3);
            say("  tab4");
            compress_one(tab4);
            say("  tab5");
            compress_one(tab5);
            say("  tab_int");
            compress_one(tab_int);
            say("  tab_bool");
            compress_one(tab_bool);
            say("  tab_bool_bits");
            compress_one(tab_bool_bits);
            say(" tab_complex");
            compress_one(tab_complex);
            say(" tab_float");
            compress_one(tab_float);
            say(" tab_char");
            compress_one(tab_char);
            say(" tab_short");
            compress_one(tab_short);
            say(" tab_long");
            compress_one(tab_long);
            say(" tab_ulong");
            compress_one(tab_ulong);
            say(" tab_quant");
            compress_one(tab_quant);
            if (bench_arr.with_struct) // --no_struct
            {
                say(" tab_point");
                compress_one(tab_point);
                say(" tab_cell");
                compress_one(tab_cell);
            }
            if (bench_arr.with_mask) // --no_mask
            {
                say(" tab_mask");
                compress_one(tab_mask);
            }
            if (bench_arr.with_conn) // --no_conn
            {
                say(" tab_conn");
                compress_one(tab_conn);
            }
        };
        if constexpr (bench_par::team_regions)
        {
            bench_par::region(
                [&]([[maybe_unused]] bench_par::Team& team)
                {
                    all([&](const char* name)
                        { team.single([&] { std::cout << name << std::endl; }, bench_par::nowait); });
                });
        }
        else
        {
            all([](const char* name) { std::cout << name << std::endl; });
        }

        chunk_config::OMPTcc<chunk_config::has_ompt>::update(true);
    }
    else
    {
        (void)bench_arr; // std::vector baseline build: nothing to compress
    }
}
