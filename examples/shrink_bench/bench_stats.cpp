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

#include <algorithm> // std::max (name_padding)
#include <complex>
#include <iomanip>
#include <iostream>
#include <string>
#include <type_traits>

#include "bench_config.h"
#include "bench_prelude.h" // Proxy2<T>::share_info_array (share_info line) // the CONSTEXPR macro
#include "bench_phases.h"
#include "bench_utils.h" // calculate_checksum

void bench_stats(BenchArrays& bench_arr, bool stats)
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
    auto& tab_point   = bench_arr.tab_point;
    auto& tab_cell    = bench_arr.tab_cell;

    const Int sz = bench_arr.sz;

    auto lbd_fill_values = [&](const std::string& name, const auto& tab, auto sec_val)
    {
        using value_type           = std::remove_reference_t<std::remove_const_t<decltype(tab)>>::value_type;
        constexpr bool is_not_bool = !std::is_same_v<value_type, bool>;
        constexpr bool is_complex  = std::is_same_v<value_type, complex_double>;
        constexpr bool is_char     = std::is_same_v<value_type, unsigned char>;

        Int nbz = 0;
        Int nbv = 0;
        Int nb  = 0;

        const Int ssz = tab.size();

#pragma omp parallel for reduction(+ : nbz, nbv, nb)
        for (Int k = 0; k < ssz; ++k)
        {
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
                // T{}, not 0: the struct element types compare with their own
                // null element and have no conversion from Int.
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
        }

        const double pc  = double(nb) / double(sz);
        const double pcz = double(nbz) / double(sz);
        const double pcv = double(nbv) / double(sz);

        std::cout << std::fixed << std::setprecision(2);

        constexpr Int name_width = 20;
        // Signed: a name longer than the width pads by 0, not by (size_t)-1 / 2.
        const Int name_padding = std::max<Int>(0, (name_width - static_cast<Int>(name.length())) / 2);
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
            oss << static_cast<int>(sec_val);
        else
            oss << sec_val;

        const std::string label = oss.str();
        std::cout << std::setw(label_width) << std::left << label << ":" // Ensures consistent colon alignment
                  << std::setw(value_width) << std::right << nbv << std::setw(value_width) << std::right << (pcv * 100)
                  << " %" << "\n";
    };

    if constexpr (chunk_config::is_shrink_enabled)
        std::cout << "\n";

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

    auto lbd_stats = [&]()
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
            lbd_fill_values("tab_complex", tab_complex, second_value_c);

            lbd_fill_values("tab_short", tab_short, second_value_s0);
            lbd_fill_values("tab_float", tab_float, second_value_f0);
            lbd_fill_values("tab_char", tab_char, second_value_c0);
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

                lbd_fill_chunk("tab_complex", tab_complex);

                lbd_fill_chunk("tab_float", tab_float);
                lbd_fill_chunk("tab_short", tab_short);
                lbd_fill_chunk("tab_char", tab_char);
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

    auto lbd_check_sum = [&]()
    {
        if CONSTEXPR (check_sum)
        {
            static const char* const n[] = {
                "tab1",      "tab2",      "tab3",     "tab4",     "tab5",      "tab_int",   "tab_bool",  "tab_complex",
                "tab_float", "tab_short", "tab_char", "tab_long", "tab_ulong", "tab_quant", "tab_point", "tab_cell"};
            print_checksums(n, tab1, tab2, tab3, tab4, tab5, tab_int, tab_bool, tab_complex, tab_float, tab_short,
                            tab_char, tab_long, tab_ulong, tab_quant);
            if (bench_arr.with_struct) // --no_struct
                print_checksums(n + 14, tab_point, tab_cell);
        }
    };

    lbd_check_sum();
}
