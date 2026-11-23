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
#include <array>       // std::array (the checksums' names)
#include <complex>     // operator== on ComplexDouble
#include <cstddef>     // std::size_t
#include <format>      // std::format (the census lines, the filling rates)
#include <iostream>    // std::cout
#include <sstream>     // std::ostringstream
#include <string>      // std::string
#include <type_traits> // std::is_same_v

#include "bench_config.h"  // the CONSTEXPR macro
#include "bench_prelude.h" // zero_or_value::Proxy<T>::array_registry (share_info line)
#include "bench_phases.h"  // bench_stats, bench_print_checksums (defined here)
#include "bench_utils.h"   // calculate_checksum

void bench_stats(BenchArrays& bench_arr, bool stats)
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

    auto lbd_fill_values = [&](const std::string& name, const auto& arr, auto sec_val)
    {
        using value_type           = std::remove_reference_t<std::remove_const_t<decltype(arr)>>::value_type;
        constexpr bool is_not_bool = !shrink_array::is_bool_like_v<value_type>; // bool, Bit
        constexpr bool is_complex  = std::is_same_v<value_type, ComplexDouble>;
        constexpr bool is_char     = std::is_same_v<value_type, unsigned char>;

        const Int ssz = arr.size();

        // `omp parallel for schedule(static) reduction(+ : ...)` with OpenMP.
        const auto census   = shrink_array::pmp::parallel_sum<Int, 3>(Int{0}, ssz, shrink_array::pmp::Options{},
                                                                      [&](Int k, auto& acc)
                                                                      {
                                                                        auto& n_zero   = acc[0];
                                                                        auto& n_values = acc[1];
                                                                        auto& n        = acc[2];
                                                                        if constexpr (is_complex)
                                                                        {
                                                                            if (arr[k] == zero_complex)
                                                                            {
                                                                                n_zero++;
                                                                            }
                                                                            else if (arr[k] == sec_val)
                                                                            {
                                                                                n_values++;
                                                                            }
                                                                            else
                                                                            {
                                                                                n++;
                                                                            }
                                                                        }
                                                                        else if constexpr (is_not_bool)
                                                                        {
                                                                            // T{}, not 0: the struct element types
                                                                            // compare with their own null element and
                                                                            // have no conversion from Int.
                                                                            if (arr[k] == value_type{})
                                                                            {
                                                                                n_zero++;
                                                                            }
                                                                            else if (arr[k] == sec_val)
                                                                            {
                                                                                n_values++;
                                                                            }
                                                                            else
                                                                            {
                                                                                n++;
                                                                            }
                                                                        }
                                                                        else
                                                                        {
                                                                            if (arr[k] == 0)
                                                                            {
                                                                                n_zero++;
                                                                            }
                                                                            else
                                                                            {
                                                                                n_values++;
                                                                            }
                                                                        }
                                                                      });
        const Int  n_zero   = census[0];
        const Int  n_values = census[1];
        const Int  n        = census[2];

        // Over the array's own size: array_float and array_char hold 2 * sz
        // elements, array_short sz / 2, array_mask mask_scale * sz, array_conn whole rows.
        const double pc  = double(n) / double(ssz);
        const double pcz = double(n_zero) / double(ssz);
        const double pcv = double(n_values) / double(ssz);

        // std::format, not std::fixed / std::setw: std::cout's flags stay as
        // they were for what the bench prints next.
        constexpr int name_width = 20;
        // Signed: a name longer than the width pads by 0, not by (size_t)-1 / 2.
        const int name_padding = std::max<int>(0, (name_width - static_cast<int>(name.length())) / 2);
        std::cout << std::string(static_cast<std::size_t>(name_padding), ' ') << "\nstatistics " << name << "\n";

        // One census line: the label padded to 22, the colon, the count and the
        // percentage right-aligned on 10.
        const auto line = [](const std::string& label, Int count, double percent)
        { std::cout << std::format("{:<22}:{:>10}{:>10.2f} %\n", label, count, percent); };

        if constexpr (is_not_bool)
        {
            line("Numbers of values", n, pc * 100);
        }
        line("Numbers of 0", n_zero, pcz * 100);

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

        line(oss.str(), n_values, pcv * 100);
    };

    if constexpr (chunk_config::is_shrink_enabled)
    {
        std::cout << "\n";
    }

    // array_quant has one second value per level it lands on (up to
    // bench_double_bands), not one: its census is zeros and the rest.
    auto lbd_fill_quant = [](const std::string& name, const auto& arr)
    {
        const Int ssz = arr.size();

        // `omp parallel for schedule(static) reduction(+ : n_zero)` with OpenMP.
        const Int n_zero = shrink_array::pmp::parallel_sum<Int, 1>(Int{0}, ssz, shrink_array::pmp::Options{},
                                                                   [&](Int k, auto& acc)
                                                                   {
                                                                       if (arr[k] == 0.0)
                                                                       {
                                                                           acc[0]++;
                                                                       }
                                                                   })[0];
        const Int n      = ssz - n_zero;

        constexpr int name_width   = 20;
        const int     name_padding = std::max<int>(0, (name_width - static_cast<int>(name.length())) / 2);
        std::cout << std::string(static_cast<std::size_t>(name_padding), ' ') << "\nstatistics " << name << "\n";
        std::cout << std::format("{:<22}:{:>10}{:>10.2f} %\n", "Numbers of 0", n_zero,
                                 100.0 * double(n_zero) / double(ssz));
        std::cout << std::format("{:<22}:{:>10}{:>10.2f} %\n", "Numbers of levels != 0", n,
                                 100.0 * double(n) / double(ssz));
    };

    // One lambda instead of two. The body of a generic lambda is a TEMPLATE, so
    // here `if constexpr` really discards: arr.stats(), which std::vector has
    // no member for, is never instantiated in the baseline build. That is the
    // difference with a plain function, where the dead branch would still be
    // type-checked -- and is why most of the other SHRINK_ARRAY guards in this
    // directory have to stay directives.
    auto lbd_fill_chunk = []([[maybe_unused]] const std::string& name, [[maybe_unused]] const auto& arr)
    {
        if constexpr (chunk_config::is_shrink_enabled)
        {
            const double sz0 = arr.stats();
            std::cout << name << " chunk filling rate : " << std::format("{:.3f}", sz0) << "%" << std::endl;
        }
    };

    const auto lbd_stats = [&]
    {
        if CONSTEXPR (stats)
        {
            lbd_fill_values("arr1", arr1, second_value_d);
            lbd_fill_values("arr2", arr2, second_value_d);
            lbd_fill_values("arr3", arr3, second_value_d);
            lbd_fill_values("arr4", arr4, second_value_d1);
            lbd_fill_values("arr5", arr5, second_value_d2);

            lbd_fill_values("array_int", array_int, second_value_i);
            lbd_fill_values("array_bool", array_bool, second_value_b);
            lbd_fill_values("array_bool_bits", array_bool_bits, second_value_b);
            lbd_fill_values("array_complex", array_complex, second_value_c);

            lbd_fill_values("array_short", array_short, second_value_s0);
            lbd_fill_values("array_float", array_float, second_value_f0);
            lbd_fill_values("array_char", array_char, second_value_c0);
            lbd_fill_values("array_long", array_long, second_value_l);
            lbd_fill_values("array_ulong", array_ulong, second_value_ul);
            lbd_fill_quant("array_quant", array_quant);
            if (bench_arr.with_mask) // --no_mask
            {
                lbd_fill_values("array_mask", array_mask, second_value_b);
            }
            if (bench_arr.with_conn) // --no_conn
            {
                lbd_fill_values("array_conn", array_conn, second_value_i);
            }
            if (bench_arr.with_struct) // --no_struct
            {
                lbd_fill_values("array_point", array_point, second_value_p);
                lbd_fill_values("array_cell", array_cell, second_value_cell);
            }

            if (chunk_config::is_shrink_enabled)
            {
                std::cout << "\n";

                lbd_fill_chunk("arr1", arr1);
                lbd_fill_chunk("arr2", arr2);
                lbd_fill_chunk("arr3", arr3);
                lbd_fill_chunk("arr4", arr4);
                lbd_fill_chunk("arr5", arr5);

                lbd_fill_chunk("array_int", array_int);
                lbd_fill_chunk("array_bool", array_bool);
                lbd_fill_chunk("array_bool_bits", array_bool_bits);

                lbd_fill_chunk("array_complex", array_complex);

                lbd_fill_chunk("array_float", array_float);
                lbd_fill_chunk("array_short", array_short);
                lbd_fill_chunk("array_char", array_char);
                lbd_fill_chunk("array_long", array_long);
                lbd_fill_chunk("array_ulong", array_ulong);
                lbd_fill_chunk("array_quant", array_quant);
                if (bench_arr.with_mask)
                {
                    lbd_fill_chunk("array_mask", array_mask);
                }
                if (bench_arr.with_conn)
                {
                    lbd_fill_chunk("array_conn", array_conn);
                }
                if (bench_arr.with_struct)
                {
                    lbd_fill_chunk("array_point", array_point);
                    lbd_fill_chunk("array_cell", array_cell);
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
    std::size_t index = 0;
    ((std::cout << "\n" << names[index++] << " check_sum = " << calculate_checksum(tabs)), ...);
    std::cout << std::dec << "\n";
}

} // namespace

void bench_print_checksums(BenchArrays& bench_arr, bool check_sum)
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
    auto& array_conn      = bench_arr.array_conn;

    const auto lbd_check_sum = [&]
    {
        if CONSTEXPR (check_sum)
        {
            static constexpr std::array<const char*, 17> names{
                "arr1",          "arr2",        "arr3",        "arr4",
                "arr5",          "array_int",   "array_bool",  "array_bool_bits",
                "array_complex", "array_float", "array_short", "array_char",
                "array_long",    "array_ulong", "array_quant", "array_point",
                "array_cell"};
            // array_bool_bits' checksum equals array_bool's: a Bit is one byte
            // holding 0 or 1, as a bool.
            print_checksums(names.data(), arr1, arr2, arr3, arr4, arr5, array_int, array_bool, array_bool_bits,
                            array_complex, array_float, array_short, array_char, array_long, array_ulong, array_quant);
            if (bench_arr.with_struct) // --no_struct
            {
                print_checksums(names.data() + 15, array_point, array_cell);
            }
            if (bench_arr.with_conn) // --no_conn
            {
                static constexpr std::array<const char*, 1> conn_name{"array_conn"};
                print_checksums(conn_name.data(), array_conn);
            }
        }
    };

    lbd_check_sum();
}
