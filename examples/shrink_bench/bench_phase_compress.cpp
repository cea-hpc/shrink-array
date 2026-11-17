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
/// @brief arr.compress() in a shrink build, nothing in the std::vector baseline.
///
/// compress() exists on the zero::Array* alone, and `if constexpr` in a plain
/// function still type-checks its discarded branch -- but the body of a
/// TEMPLATE is genuinely not instantiated, so routing the thirteen calls through
/// one abbreviated function template lets the guard below be `if constexpr`.
void compress_one(auto& arr)
{
    if constexpr (chunk_config::is_shrink_enabled)
    {
        arr.compress();
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
    // The OMPT bracketing belongs to the compression: a caller cannot forget
    // what it does not have to write. The outer guard keeps the baseline build
    // doing strictly nothing here rather than running the parallel region and
    // printing thirteen names.
    if constexpr (chunk_config::is_shrink_enabled)
    {
        chunk_config::OMPTcc<chunk_config::has_ompt>::update(false);

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

        // Compress the queued candidates first, from outside the region: each
        // compress() below runs in an `omp single`, where compress_int cannot
        // drain (it would nest a parallel region), so a candidate still queued
        // by the serial writes of the iteration (fill_range / copy_range) was
        // reported as a miss its self-heal had to compress.
        if constexpr (build_config::threaded)
        {
            if (!chunk_config::aggressive_compression()) // run-time switch: chunk/config.h
            {
                const auto& ens = shrink_array::SingletonChk<shrink_array::TypeStateRegistry>::get_ref();
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
            say("  arr1");
            compress_one(arr1);
            say("  arr2");
            compress_one(arr2);
            say("  arr3");
            compress_one(arr3);
            say("  arr4");
            compress_one(arr4);
            say("  arr5");
            compress_one(arr5);
            say("  array_int");
            compress_one(array_int);
            say("  array_bool");
            compress_one(array_bool);
            say("  array_bool_bits");
            compress_one(array_bool_bits);
            say(" array_complex");
            compress_one(array_complex);
            say(" array_float");
            compress_one(array_float);
            say(" array_char");
            compress_one(array_char);
            say(" array_short");
            compress_one(array_short);
            say(" array_long");
            compress_one(array_long);
            say(" array_ulong");
            compress_one(array_ulong);
            say(" array_quant");
            compress_one(array_quant);
            if (bench_arr.with_struct) // --no_struct
            {
                say(" array_point");
                compress_one(array_point);
                say(" array_cell");
                compress_one(array_cell);
            }
            if (bench_arr.with_mask) // --no_mask
            {
                say(" array_mask");
                compress_one(array_mask);
            }
            if (bench_arr.with_conn) // --no_conn
            {
                say(" array_conn");
                compress_one(array_conn);
            }
        };
        if constexpr (bench_par::team_regions)
        {
            bench_par::region(
                [&]([[maybe_unused]] bench_par::Team& team)
                { all([&](const char* name) { team.single([&] { announce(name); }, bench_par::nowait); }); });
        }
        else
        {
            all([](const char* name) { announce(name); });
        }

        chunk_config::OMPTcc<chunk_config::has_ompt>::update(true);
    }
    else
    {
        (void)bench_arr; // std::vector baseline build: nothing to compress
    }
}
