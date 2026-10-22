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

/// @file bench_stats.cpp
/// @brief The value census and filling-rate report. See bench_phases.h.

#include <algorithm>   // std::max (name_padding)
#include <complex>     // operator== on ComplexDouble
#include <cstddef>     // std::size_t
#include <iomanip>     // std::setw, std::setprecision
#include <iostream>    // std::cout
#include <sstream>     // std::ostringstream
#include <string>      // std::string
#include <type_traits> // std::is_same_v

#include "bench_config.h"  // the CONSTEXPR macro
#include "bench_prelude.h" // Proxy2<T>::share_info_array (share_info line)
#include "bench_phases.h"  // bench_stats, bench_print_checksums (defined here)
#include "bench_utils.h"   // calculate_checksum

void bench_stats(BenchArrays& bench_arr, bool stats)
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

    auto lbd_fill_values = [&](const std::string& name, const auto& tab, auto sec_val)
    {
        using value_type           = std::remove_reference_t<std::remove_const_t<decltype(tab)>>::value_type;
        constexpr bool is_not_bool = !shrink_array::is_bool_like_v<value_type>; // bool, Bit
        constexpr bool is_complex  = std::is_same_v<value_type, ComplexDouble>;
        constexpr bool is_char     = std::is_same_v<value_type, unsigned char>;

        const Int ssz = tab.size();

        // `omp parallel for schedule(static) reduction(+ : ...)` with OpenMP.
        const auto census = shrink_array::pmp::parallel_sum<Int, 3>(Int{0}, ssz, shrink_array::pmp::options{},
                                                                    [&](Int k, auto& acc)
                                                                    {
                                                                        auto& nbz = acc[0];
                                                                        auto& nbv = acc[1];
                                                                        auto& nb  = acc[2];
                                                                        if constexpr (is_complex)
                                                                        {
                                                                            if (tab[k] == zero_complex)
                                                                            {
                                                                                nbz++;
                                                                            }
                                                                            else if (tab[k] == sec_val)
                                                                            {
                                                                                nbv++;
                                                                            }
                                                                            else
                                                                            {
                                                                                nb++;
                                                                            }
                                                                        }
                                                                        else if constexpr (is_not_bool)
                                                                        {
                                                                            // T{}, not 0: the struct element types
                                                                            // compare with their own null element and
                                                                            // have no conversion from Int.
                                                                            if (tab[k] == value_type{})
                                                                            {
                                                                                nbz++;
                                                                            }
                                                                            else if (tab[k] == sec_val)
                                                                            {
                                                                                nbv++;
                                                                            }
                                                                            else
                                                                            {
                                                                                nb++;
                                                                            }
                                                                        }
                                                                        else
                                                                        {
                                                                            if (tab[k] == 0)
                                                                            {
                                                                                nbz++;
                                                                            }
                                                                            else
                                                                            {
                                                                                nbv++;
                                                                            }
                                                                        }
                                                                    });
        const Int  nbz    = census[0];
        const Int  nbv    = census[1];
        const Int  nb     = census[2];

        // Over the array's own size: tab_float and tab_char hold 2 * sz
        // elements, tab_short sz / 2, tab_mask mask_scale * sz.
        const double pc  = double(nb) / double(ssz);
        const double pcz = double(nbz) / double(ssz);
        const double pcv = double(nbv) / double(ssz);

        std::cout << std::fixed << std::setprecision(2);

        // int, not Int: a display width handed to std::setw(int), never an
        // array index -- Int widens to 64 bits under BENCH_INT64 and would
        // narrow back at setw().
        constexpr int name_width = 20;
        // Signed: a name longer than the width pads by 0, not by (size_t)-1 / 2.
        const int name_padding = std::max<int>(0, (name_width - static_cast<int>(name.length())) / 2);
        std::cout << std::setw(name_padding) << "" << "\nstatistics " << name << "\n";

        constexpr Int label_width = 22;
        constexpr Int value_width = 10;

        // The colon is now output as a separate token to ensure alignment.
        if constexpr (is_not_bool)
        {
            std::cout << std::setw(label_width) << std::left << "Numbers of values" << ":" << std::setw(value_width)
                      << std::right << nb << std::setw(value_width) << std::right << (pc * 100) << " %" << "\n";
        }

        // First output line: label "Numbers of 0"
        const std::string label0 = "Numbers of 0";
        std::cout << std::setw(label_width) << std::left << label0
                  << ":" // Ensures the colon is always in the same position
                  << std::setw(value_width) << std::right << nbz << std::setw(value_width) << std::right << (pcz * 100)
                  << " %" << "\n";

        // Second output line: build the label dynamically with a suffix.
        std::ostringstream oss;
        oss << "Numbers of ";

        if constexpr (is_char)
        {
            oss << static_cast<int>(sec_val);
        }
        else
        {
            oss << sec_val;
        }

        const std::string label = oss.str();
        std::cout << std::setw(label_width) << std::left << label << ":" // Ensures consistent colon alignment
                  << std::setw(value_width) << std::right << nbv << std::setw(value_width) << std::right << (pcv * 100)
                  << " %" << "\n";
    };

    if constexpr (chunk_config::is_shrink_enabled)
    {
        std::cout << "\n";
    }

    // tab_quant has one second value per level it lands on (up to
    // bench_double_bands), not one: its census is zeros and the rest.
    auto lbd_fill_quant = [](const std::string& name, const auto& tab)
    {
        const Int ssz = tab.size();

        // `omp parallel for schedule(static) reduction(+ : nbz)` with OpenMP.
        const Int nbz = shrink_array::pmp::parallel_sum<Int, 1>(Int{0}, ssz, shrink_array::pmp::options{},
                                                                [&](Int k, auto& acc)
                                                                {
                                                                    if (tab[k] == 0.0)
                                                                    {
                                                                        acc[0]++;
                                                                    }
                                                                })[0];
        const Int nb  = ssz - nbz;

        constexpr int name_width   = 20;
        const int     name_padding = std::max<int>(0, (name_width - static_cast<int>(name.length())) / 2);
        std::cout << std::fixed << std::setprecision(2);
        std::cout << std::setw(name_padding) << "" << "\nstatistics " << name << "\n";

        constexpr int label_width = 22;
        constexpr int value_width = 10;
        std::cout << std::setw(label_width) << std::left << "Numbers of 0" << ":" << std::setw(value_width)
                  << std::right << nbz << std::setw(value_width) << std::right << (100.0 * double(nbz) / double(ssz))
                  << " %" << "\n";
        std::cout << std::setw(label_width) << std::left << "Numbers of levels != 0" << ":" << std::setw(value_width)
                  << std::right << nb << std::setw(value_width) << std::right << (100.0 * double(nb) / double(ssz))
                  << " %" << "\n";
    };

    // One lambda instead of two. The body of a generic lambda is a TEMPLATE, so
    // here `if constexpr` really discards: tab.stats(), which std::vector has
    // no member for, is never instantiated in the baseline build. That is the
    // difference with a plain function, where the dead branch would still be
    // type-checked -- and is why most of the other SHRINK_ARRAY guards in this
    // directory have to stay directives.
    auto lbd_fill_chunk = []([[maybe_unused]] const std::string& name, [[maybe_unused]] const auto& tab)
    {
        if constexpr (chunk_config::is_shrink_enabled)
        {
            const double sz0 = tab.stats();
            std::cout << name << " chunk filling rate : " << std::fixed << std::setprecision(3) << sz0 << "%"
                      << std::endl;
        }
    };

    const auto lbd_stats = [&]
    {
        if CONSTEXPR (stats)
        {
            lbd_fill_values("tab1", tab1, second_value_d);
            lbd_fill_values("tab2", tab2, second_value_d);
            lbd_fill_values("tab3", tab3, second_value_d);
            lbd_fill_values("tab4", tab4, second_value_d1);
            lbd_fill_values("tab5", tab5, second_value_d2);

            lbd_fill_values("tab_int", tab_int, second_value_i);
            lbd_fill_values("tab_bool", tab_bool, second_value_b);
            lbd_fill_values("tab_bool_bits", tab_bool_bits, second_value_b);
            lbd_fill_values("tab_complex", tab_complex, second_value_c);

            lbd_fill_values("tab_short", tab_short, second_value_s0);
            lbd_fill_values("tab_float", tab_float, second_value_f0);
            lbd_fill_values("tab_char", tab_char, second_value_c0);
            lbd_fill_values("tab_long", tab_long, second_value_l);
            lbd_fill_values("tab_ulong", tab_ulong, second_value_ul);
            lbd_fill_quant("tab_quant", tab_quant);
            if (bench_arr.with_mask) // --no_mask
            {
                lbd_fill_values("tab_mask", tab_mask, second_value_b);
            }
            if (bench_arr.with_struct) // --no_struct
            {
                lbd_fill_values("tab_point", tab_point, second_value_p);
                lbd_fill_values("tab_cell", tab_cell, second_value_cell);
            }

            if (chunk_config::is_shrink_enabled)
            {
                std::cout << "\n";

                lbd_fill_chunk("tab1", tab1);
                lbd_fill_chunk("tab2", tab2);
                lbd_fill_chunk("tab3", tab3);
                lbd_fill_chunk("tab4", tab4);
                lbd_fill_chunk("tab5", tab5);

                lbd_fill_chunk("tab_int", tab_int);
                lbd_fill_chunk("tab_bool", tab_bool);
                lbd_fill_chunk("tab_bool_bits", tab_bool_bits);

                lbd_fill_chunk("tab_complex", tab_complex);

                lbd_fill_chunk("tab_float", tab_float);
                lbd_fill_chunk("tab_short", tab_short);
                lbd_fill_chunk("tab_char", tab_char);
                lbd_fill_chunk("tab_long", tab_long);
                lbd_fill_chunk("tab_ulong", tab_ulong);
                lbd_fill_chunk("tab_quant", tab_quant);
                if (bench_arr.with_mask)
                {
                    lbd_fill_chunk("tab_mask", tab_mask);
                }
                if (bench_arr.with_struct)
                {
                    lbd_fill_chunk("tab_point", tab_point);
                    lbd_fill_chunk("tab_cell", tab_cell);
                }
            }
        }
    };

    lbd_stats();
}

namespace
{

template <typename... Tabs>
void print_checksums(const char* const* names, const Tabs&... tabs)
{
    std::cout << std::hex;
    std::size_t i = 0;
    ((std::cout << "\n" << names[i++] << " check_sum = " << calculate_checksum(tabs)), ...);
    std::cout << std::dec << "\n";
}

} // namespace

void bench_print_checksums(BenchArrays& bench_arr, bool check_sum)
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

    const auto lbd_check_sum = [&]
    {
        if CONSTEXPR (check_sum)
        {
            static const char* const n[] = {"tab1",      "tab2",     "tab3",          "tab4",        "tab5",
                                            "tab_int",   "tab_bool", "tab_bool_bits", "tab_complex", "tab_float",
                                            "tab_short", "tab_char", "tab_long",      "tab_ulong",   "tab_quant",
                                            "tab_point", "tab_cell"};
            // tab_bool_bits' checksum equals tab_bool's: a Bit is one byte
            // holding 0 or 1, as a bool.
            print_checksums(n, tab1, tab2, tab3, tab4, tab5, tab_int, tab_bool, tab_bool_bits, tab_complex, tab_float,
                            tab_short, tab_char, tab_long, tab_ulong, tab_quant);
            if (bench_arr.with_struct) // --no_struct
            {
                print_checksums(n + 15, tab_point, tab_cell);
            }
        }
    };

    lbd_check_sum();
}
