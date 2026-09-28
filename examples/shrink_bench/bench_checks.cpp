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

#include <cstdint> // std::uint64_t
#include <iostream>
#include <string>

#include "bench_config.h" // the CONSTEXPR macro, empty since the check/perf merge
#include "bench_phases.h"
#include "bench_io.h"    // write_binary_file, read_check_binary_file
#include "bench_utils.h" // calculate_checksum

void bench_check_all(BenchArrays& bench_arr, Int ind, const BenchIoFlags& io)
{
    // Aliases so the body below is VERBATIM (see bench_phase_fill.cpp).
    auto& tab1 = bench_arr.tab1;
    auto& tab2 = bench_arr.tab2;
    auto& tab3 = bench_arr.tab3;
    auto& tab4 = bench_arr.tab4;
    auto& tab5 = bench_arr.tab5;

    auto& tab_int     = bench_arr.tab_int;
    auto& tab_bool    = bench_arr.tab_bool;
    auto& tab_complex = bench_arr.tab_complex;
    auto& tab_float   = bench_arr.tab_float;
    auto& tab_short   = bench_arr.tab_short;
    auto& tab_char    = bench_arr.tab_char;
    auto& tab_long    = bench_arr.tab_long;
    auto& tab_ulong   = bench_arr.tab_ulong;
    auto& tab_quant   = bench_arr.tab_quant;
    auto& tab_point   = bench_arr.tab_point;
    auto& tab_cell    = bench_arr.tab_cell;
    auto& tab_mask    = bench_arr.tab_mask;

    // The flags the two lambdas captured from main(), under their original
    // names so the bodies below are VERBATIM.
    const char*                 no_io       = io.no_io;
    const char*                 no_compress = io.no_compress;
    const Int                   io_res      = io.io_res;
    [[maybe_unused]] const bool read_trace  = io.read_trace;
    [[maybe_unused]] const bool write_trace = io.write_trace;

    auto lbd_check =
        [&](Int kk, const std::string& tab_name, const auto& tab, std::uint64_t checksum, bool special_writing = false)
    {
        if (no_io)
        {
            return;
        }
        const std::string filename = "data_" + tab_name + "_" + std::to_string(kk) + ".bin";

        if (io_res == 1)
        {
            write_binary_file(filename, tab, checksum, special_writing, no_compress);
        }
        // io_res == 2 is unreachable without a shrink build (get_io_mode returns
        // 1, never 2), so this is dead code there -- but it now COMPILES there:
        // read_check_binary_file is a template whose chunk-only parts are
        // `if constexpr`-guarded on a dependent object, so they are not
        // instantiated for a std::vector. That is what retired the directive.
        else if (io_res == 2)
        {
            read_check_binary_file(kk, tab_name, filename, tab, checksum, no_compress, read_trace, write_trace);
        }
    };

    const auto lbd_check_all = [&](Int ind)
    {
        if (no_io)
        {
            return;
        }

        if CONSTEXPR (io_res)
        {
            std::uint64_t tab1_ch = 0, tab2_ch = 0, tab3_ch = 0, tab4_ch = 0, tab5_ch = 0, tab_int_ch = 0,
                          tab_bool_ch = 0, tab_complex_ch = 0, tab_float_ch = 0, tab_char_ch = 0, tab_short_ch = 0,
                          tab_long_ch = 0, tab_ulong_ch = 0, tab_quant_ch = 0, tab_point_ch = 0, tab_cell_ch = 0,
                          tab_mask_ch = 0;

            VectorChecksum vec1_ch(tab1);
            VectorChecksum vec2_ch(tab2);
            VectorChecksum vec3_ch(tab3);
            VectorChecksum vec4_ch(tab4);
            VectorChecksum vec5_ch(tab5);

            VectorChecksum vec_int_ch(tab_int);
            VectorChecksum vec_bool_ch(tab_bool);
            VectorChecksum vec_complex_ch(tab_complex);

            VectorChecksum vec_float_ch(tab_float);
            VectorChecksum vec_char_ch(tab_char);
            VectorChecksum vec_short_ch(tab_short);
            VectorChecksum vec_long_ch(tab_long);
            VectorChecksum vec_ulong_ch(tab_ulong);
            VectorChecksum vec_quant_ch(tab_quant);
            VectorChecksum vec_point_ch(tab_point);
            VectorChecksum vec_cell_ch(tab_cell);
            VectorChecksum vec_mask_ch(tab_mask);

            // One calculate_checksum_0 call per array, not a variadic fold over
            // them: the fold helper (removed) took its VectorChecksum arguments by
            // value, so the per-thread sums went into copies and were lost.

            std::cout << "\n";

#pragma omp parallel
            {
                calculate_checksum_0(vec1_ch.vec, vec1_ch.checksum_thr);
                calculate_checksum_0(vec2_ch.vec, vec2_ch.checksum_thr);
                calculate_checksum_0(vec3_ch.vec, vec3_ch.checksum_thr);
                calculate_checksum_0(vec4_ch.vec, vec4_ch.checksum_thr);
                calculate_checksum_0(vec5_ch.vec, vec5_ch.checksum_thr);

                calculate_checksum_0(vec_int_ch.vec, vec_int_ch.checksum_thr);
                calculate_checksum_0(vec_bool_ch.vec, vec_bool_ch.checksum_thr);
                calculate_checksum_0(vec_complex_ch.vec, vec_complex_ch.checksum_thr);

                calculate_checksum_0(vec_short_ch.vec, vec_short_ch.checksum_thr);
                calculate_checksum_0(vec_float_ch.vec, vec_float_ch.checksum_thr);
                calculate_checksum_0(vec_char_ch.vec, vec_char_ch.checksum_thr);
                calculate_checksum_0(vec_long_ch.vec, vec_long_ch.checksum_thr);
                calculate_checksum_0(vec_ulong_ch.vec, vec_ulong_ch.checksum_thr);
                calculate_checksum_0(vec_quant_ch.vec, vec_quant_ch.checksum_thr);
                calculate_checksum_0(vec_point_ch.vec, vec_point_ch.checksum_thr);
                calculate_checksum_0(vec_cell_ch.vec, vec_cell_ch.checksum_thr);
                calculate_checksum_0(vec_mask_ch.vec, vec_mask_ch.checksum_thr);
            }

            for (Int thr = 0; thr < CHUNK_PMP_MAX_THREADS; ++thr)
            {
                tab1_ch += vec1_ch.checksum_thr[thr];
                tab2_ch += vec2_ch.checksum_thr[thr];
                tab3_ch += vec3_ch.checksum_thr[thr];
                tab4_ch += vec4_ch.checksum_thr[thr];
                tab5_ch += vec5_ch.checksum_thr[thr];

                tab_int_ch += vec_int_ch.checksum_thr[thr];
                tab_bool_ch += vec_bool_ch.checksum_thr[thr];
                tab_complex_ch += vec_complex_ch.checksum_thr[thr];

                tab_float_ch += vec_float_ch.checksum_thr[thr];
                tab_short_ch += vec_short_ch.checksum_thr[thr];
                tab_char_ch += vec_char_ch.checksum_thr[thr];
                tab_long_ch += vec_long_ch.checksum_thr[thr];
                tab_ulong_ch += vec_ulong_ch.checksum_thr[thr];
                tab_quant_ch += vec_quant_ch.checksum_thr[thr];
                tab_point_ch += vec_point_ch.checksum_thr[thr];
                tab_cell_ch += vec_cell_ch.checksum_thr[thr];
                tab_mask_ch += vec_mask_ch.checksum_thr[thr];
            }

            tab1_ch = ~tab1_ch;
            tab2_ch = ~tab2_ch;
            tab3_ch = ~tab3_ch;
            tab4_ch = ~tab4_ch;
            tab5_ch = ~tab5_ch;

            tab_int_ch     = ~tab_int_ch;
            tab_bool_ch    = ~tab_bool_ch;
            tab_complex_ch = ~tab_complex_ch;

            tab_short_ch = ~tab_short_ch;
            tab_float_ch = ~tab_float_ch;
            tab_char_ch  = ~tab_char_ch;
            tab_long_ch  = ~tab_long_ch;
            tab_ulong_ch = ~tab_ulong_ch;
            tab_quant_ch = ~tab_quant_ch;
            tab_point_ch = ~tab_point_ch;
            tab_cell_ch  = ~tab_cell_ch;
            tab_mask_ch  = ~tab_mask_ch;

            std::cout << "\n";

#pragma omp parallel
            {
#pragma omp sections nowait
                {
#pragma omp section
                    {
                        lbd_check(ind, "tab1_0", tab1, tab1_ch);
                    }
#pragma omp section
                    {
                        lbd_check(ind, "tab2_0", tab2, tab2_ch);
                    }
#pragma omp section
                    {
                        lbd_check(ind, "tab3_0", tab3, tab3_ch);
                    }
#pragma omp section
                    {
                        lbd_check(ind, "tab4_0", tab4, tab4_ch);
                    }
#pragma omp section
                    {
                        lbd_check(ind, "tab5_0", tab5, tab5_ch);
                    }

#pragma omp section
                    {
                        lbd_check(ind, "tab_int", tab_int, tab_int_ch);
                    }
#pragma omp section
                    {
                        lbd_check(ind, "tab_bool", tab_bool, tab_bool_ch, true /* special writing for vector */);
                    }
#pragma omp section
                    {
                        lbd_check(ind, "tab_complex", tab_complex, tab_complex_ch);
                    }
#pragma omp section
                    {
                        lbd_check(ind, "tab_short", tab_short, tab_short_ch);
                    }
#pragma omp section
                    {
                        lbd_check(ind, "tab_float", tab_float, tab_float_ch);
                    }
#pragma omp section
                    {
                        lbd_check(ind, "tab_char", tab_char, tab_char_ch);
                    }
#pragma omp section
                    {
                        lbd_check(ind, "tab_long", tab_long, tab_long_ch);
                    }
#pragma omp section
                    {
                        lbd_check(ind, "tab_ulong", tab_ulong, tab_ulong_ch);
                    }
#pragma omp section
                    {
                        lbd_check(ind, "tab_quant", tab_quant, tab_quant_ch);
                    }
#pragma omp section
                    {
                        if (bench_arr.with_struct) // --no_struct: no reference file either
                            lbd_check(ind, "tab_point", tab_point, tab_point_ch);
                    }
#pragma omp section
                    {
                        if (bench_arr.with_struct)
                            lbd_check(ind, "tab_cell", tab_cell, tab_cell_ch);
                    }
#pragma omp section
                    {
                        if (bench_arr.with_mask) // --no_mask: no reference file either
                            lbd_check(ind, "tab_mask", tab_mask, tab_mask_ch, true /* one byte per bool */);
                    }
                }
            }
        }
    };

    lbd_check_all(ind);
}
