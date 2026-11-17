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

/// @file bench_checks.cpp
/// @brief The reference-file verification pass. See bench_phases.h.

#include <cstdint>  // std::uint64_t
#include <iostream> // std::cout
#include <string>   // std::string, std::to_string

#include "bench_config.h" // the CONSTEXPR macro, empty since the check/perf merge
#include "bench_phases.h" // bench_check_all (defined here), BenchIoFlags
#include "bench_io.h"     // write_binary_file, read_check_binary_file
#include "bench_utils.h"  // calculate_checksum

void bench_check_all(BenchArrays& bench_arr, Int ind, const BenchIoFlags& io)
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

    // The I/O flags, as locals the two lambdas capture.
    const char*                 no_io         = io.no_io;
    const char*                 no_compress   = io.no_compress;
    const Int                   io_res        = io.io_res;
    [[maybe_unused]] const bool read_trace    = io.read_trace;
    [[maybe_unused]] const bool write_trace   = io.write_trace;
    const bool                  checksum_only = io.checksum_only;

    auto lbd_check = [&](Int kk, const std::string& array_name, const auto& arr, std::uint64_t checksum,
                         bool special_writing = false)
    {
        if (no_io)
        {
            return;
        }
        const std::string filename = "data_" + array_name + "_" + std::to_string(kk) + ".bin";

        if (io_res == 1)
        {
            write_binary_file(filename, arr, checksum, special_writing, no_compress, checksum_only);
        }
        // io_res == 2 is unreachable without a shrink build (get_io_mode returns
        // 1, never 2), so this is dead code there -- but it now COMPILES there:
        // read_check_binary_file is a template whose chunk-only parts are
        // `if constexpr`-guarded on a dependent object, so they are not
        // instantiated for a std::vector. That is what retired the directive.
        else if (io_res == 2)
        {
            read_check_binary_file(kk, array_name, filename, arr, checksum, no_compress, read_trace, write_trace);
        }
    };

    const auto lbd_check_all = [&]
    {
        if (no_io)
        {
            return;
        }

        if CONSTEXPR (io_res)
        {
            std::uint64_t arr1_ch            = 0;
            std::uint64_t arr2_ch            = 0;
            std::uint64_t arr3_ch            = 0;
            std::uint64_t arr4_ch            = 0;
            std::uint64_t arr5_ch            = 0;
            std::uint64_t array_int_ch       = 0;
            std::uint64_t array_bool_ch      = 0;
            std::uint64_t array_bool_bits_ch = 0;
            std::uint64_t array_complex_ch   = 0;
            std::uint64_t array_float_ch     = 0;
            std::uint64_t array_char_ch      = 0;
            std::uint64_t array_short_ch     = 0;
            std::uint64_t array_long_ch      = 0;
            std::uint64_t array_ulong_ch     = 0;
            std::uint64_t array_quant_ch     = 0;
            std::uint64_t array_point_ch     = 0;
            std::uint64_t array_cell_ch      = 0;
            std::uint64_t array_mask_ch      = 0;
            std::uint64_t array_conn_ch      = 0;

            VectorChecksum vec1_ch(arr1);
            VectorChecksum vec2_ch(arr2);
            VectorChecksum vec3_ch(arr3);
            VectorChecksum vec4_ch(arr4);
            VectorChecksum vec5_ch(arr5);

            VectorChecksum vec_int_ch(array_int);
            VectorChecksum vec_bool_ch(array_bool);
            VectorChecksum vec_bool_bits_ch(array_bool_bits);
            VectorChecksum vec_complex_ch(array_complex);

            VectorChecksum vec_float_ch(array_float);
            VectorChecksum vec_char_ch(array_char);
            VectorChecksum vec_short_ch(array_short);
            VectorChecksum vec_long_ch(array_long);
            VectorChecksum vec_ulong_ch(array_ulong);
            VectorChecksum vec_quant_ch(array_quant);
            VectorChecksum vec_point_ch(array_point);
            VectorChecksum vec_cell_ch(array_cell);
            VectorChecksum vec_mask_ch(array_mask);
            VectorChecksum vec_conn_ch(array_conn);

            // One calculate_checksum_0 call per array, not a variadic fold over
            // them: a fold helper taking its VectorChecksum arguments by value
            // would sum into copies and lose the per-thread sums.

            std::cout << "\n";

            bench_par::region(
                [&]([[maybe_unused]] bench_par::Team& team)
                {
                    calculate_checksum_0(team, vec1_ch.vec, vec1_ch.checksum_thr);
                    calculate_checksum_0(team, vec2_ch.vec, vec2_ch.checksum_thr);
                    calculate_checksum_0(team, vec3_ch.vec, vec3_ch.checksum_thr);
                    calculate_checksum_0(team, vec4_ch.vec, vec4_ch.checksum_thr);
                    calculate_checksum_0(team, vec5_ch.vec, vec5_ch.checksum_thr);

                    calculate_checksum_0(team, vec_int_ch.vec, vec_int_ch.checksum_thr);
                    calculate_checksum_0(team, vec_bool_ch.vec, vec_bool_ch.checksum_thr);
                    calculate_checksum_0(team, vec_bool_bits_ch.vec, vec_bool_bits_ch.checksum_thr);
                    calculate_checksum_0(team, vec_complex_ch.vec, vec_complex_ch.checksum_thr);

                    calculate_checksum_0(team, vec_short_ch.vec, vec_short_ch.checksum_thr);
                    calculate_checksum_0(team, vec_float_ch.vec, vec_float_ch.checksum_thr);
                    calculate_checksum_0(team, vec_char_ch.vec, vec_char_ch.checksum_thr);
                    calculate_checksum_0(team, vec_long_ch.vec, vec_long_ch.checksum_thr);
                    calculate_checksum_0(team, vec_ulong_ch.vec, vec_ulong_ch.checksum_thr);
                    calculate_checksum_0(team, vec_quant_ch.vec, vec_quant_ch.checksum_thr);
                    calculate_checksum_0(team, vec_point_ch.vec, vec_point_ch.checksum_thr);
                    calculate_checksum_0(team, vec_cell_ch.vec, vec_cell_ch.checksum_thr);
                    calculate_checksum_0(team, vec_mask_ch.vec, vec_mask_ch.checksum_thr);
                });

            for (Int thr = 0; thr < CHUNK_PMP_MAX_THREADS; ++thr)
            {
                arr1_ch += vec1_ch.checksum_thr[thr];
                arr2_ch += vec2_ch.checksum_thr[thr];
                arr3_ch += vec3_ch.checksum_thr[thr];
                arr4_ch += vec4_ch.checksum_thr[thr];
                arr5_ch += vec5_ch.checksum_thr[thr];

                array_int_ch += vec_int_ch.checksum_thr[thr];
                array_bool_ch += vec_bool_ch.checksum_thr[thr];
                array_bool_bits_ch += vec_bool_bits_ch.checksum_thr[thr];
                array_complex_ch += vec_complex_ch.checksum_thr[thr];

                array_float_ch += vec_float_ch.checksum_thr[thr];
                array_short_ch += vec_short_ch.checksum_thr[thr];
                array_char_ch += vec_char_ch.checksum_thr[thr];
                array_long_ch += vec_long_ch.checksum_thr[thr];
                array_ulong_ch += vec_ulong_ch.checksum_thr[thr];
                array_quant_ch += vec_quant_ch.checksum_thr[thr];
                array_point_ch += vec_point_ch.checksum_thr[thr];
                array_cell_ch += vec_cell_ch.checksum_thr[thr];
                array_mask_ch += vec_mask_ch.checksum_thr[thr];
                array_conn_ch += vec_conn_ch.checksum_thr[thr];
            }

            arr1_ch = ~arr1_ch;
            arr2_ch = ~arr2_ch;
            arr3_ch = ~arr3_ch;
            arr4_ch = ~arr4_ch;
            arr5_ch = ~arr5_ch;

            array_int_ch       = ~array_int_ch;
            array_bool_ch      = ~array_bool_ch;
            array_bool_bits_ch = ~array_bool_bits_ch;
            array_complex_ch   = ~array_complex_ch;

            array_short_ch = ~array_short_ch;
            array_float_ch = ~array_float_ch;
            array_char_ch  = ~array_char_ch;
            array_long_ch  = ~array_long_ch;
            array_ulong_ch = ~array_ulong_ch;
            array_quant_ch = ~array_quant_ch;
            array_point_ch = ~array_point_ch;
            array_cell_ch  = ~array_cell_ch;
            array_mask_ch  = ~array_mask_ch;
            array_conn_ch  = ~array_conn_ch;

            std::cout << "\n";

            bench_par::region(
                [&]([[maybe_unused]] bench_par::Team& team)
                {
                    team.single([&] { lbd_check(ind, "arr1_0", arr1, arr1_ch); }, bench_par::nowait);
                    team.single([&] { lbd_check(ind, "arr2_0", arr2, arr2_ch); }, bench_par::nowait);
                    team.single([&] { lbd_check(ind, "arr3_0", arr3, arr3_ch); }, bench_par::nowait);
                    team.single([&] { lbd_check(ind, "arr4_0", arr4, arr4_ch); }, bench_par::nowait);
                    team.single([&] { lbd_check(ind, "arr5_0", arr5, arr5_ch); }, bench_par::nowait);

                    team.single([&] { lbd_check(ind, "array_int", array_int, array_int_ch); }, bench_par::nowait);
                    team.single(
                        [&]
                        {
                            lbd_check(ind, "array_bool", array_bool, array_bool_ch,
                                      true /* special writing for vector */);
                            // array_bool_bits holds what array_bool holds: read back
                            // against array_bool's reference file, never written
                            // (io_res == 1 would write the same file twice). In
                            // this section, after array_bool's check: a checksum
                            // miss decompresses the file, which two of these must
                            // not do at once.
                            if (!no_io && io_res == 2)
                            {
                                read_check_binary_file(
                                    ind, "array_bool_bits", "data_array_bool_" + std::to_string(ind) + ".bin",
                                    array_bool_bits, array_bool_bits_ch, no_compress, read_trace, write_trace);
                            }
                        },
                        bench_par::nowait);
                    team.single([&] { lbd_check(ind, "array_complex", array_complex, array_complex_ch); },
                                bench_par::nowait);
                    team.single([&] { lbd_check(ind, "array_short", array_short, array_short_ch); }, bench_par::nowait);
                    team.single([&] { lbd_check(ind, "array_float", array_float, array_float_ch); }, bench_par::nowait);
                    team.single([&] { lbd_check(ind, "array_char", array_char, array_char_ch); }, bench_par::nowait);
                    team.single([&] { lbd_check(ind, "array_long", array_long, array_long_ch); }, bench_par::nowait);
                    team.single([&] { lbd_check(ind, "array_ulong", array_ulong, array_ulong_ch); }, bench_par::nowait);
                    team.single([&] { lbd_check(ind, "array_quant", array_quant, array_quant_ch); }, bench_par::nowait);
                    team.single(
                        [&]
                        {
                            if (bench_arr.with_struct) // --no_struct: no reference file either
                            {
                                lbd_check(ind, "array_point", array_point, array_point_ch);
                            }
                        },
                        bench_par::nowait);
                    team.single(
                        [&]
                        {
                            if (bench_arr.with_struct)
                            {
                                lbd_check(ind, "array_cell", array_cell, array_cell_ch);
                            }
                        },
                        bench_par::nowait);
                    team.single(
                        [&]
                        {
                            if (bench_arr.with_mask) // --no_mask: no reference file either
                            {
                                lbd_check(ind, "array_mask", array_mask, array_mask_ch, true /* one byte per bool */);
                            }
                        },
                        bench_par::nowait);
                    team.single(
                        [&]
                        {
                            if (bench_arr.with_conn) // --no_conn: no reference file either
                            {
                                lbd_check(ind, "array_conn", array_conn, array_conn_ch);
                            }
                        },
                        bench_par::nowait);
                });
        }
    };

    lbd_check_all();
}
