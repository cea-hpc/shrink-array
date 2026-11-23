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
#include <cstdint>     // std::uint8_t (ChecksumGroup)
#include <meta>        // std::meta (the arrays of BenchArrays, print_checksums)
#include <string>      // std::string
#include <string_view> // std::string_view (the arrays' names)
#include <type_traits> // std::is_same_v
#include <vector>      // std::vector (consteval only: the array members)
#include <cstdio>      // std::fflush
#include <print>       // std::println, std::print

#include "bench_config.h"  // the CONSTEXPR macro
#include "bench_prelude.h" // zero_or_value::Proxy<T>::array_registry (share_info line)
#include "bench_phases.h"  // bench_stats, bench_print_checksums (defined here)
#include "bench_utils.h"   // calculate_checksum, bench_text

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

        // int, not Int: a display width handed to the format string, never an
        // array index -- Int widens to 64 bits under BENCH_INT64.
        constexpr int name_width = 20;
        // Signed: a name longer than the width pads by 0, not by (size_t)-1 / 2.
        const int name_padding = std::max<int>(0, (name_width - static_cast<int>(name.length())) / 2);
        std::println("{:{}}\nstatistics {}", "", name_padding, name);

        constexpr Int label_width = 22;
        constexpr Int value_width = 10;

        // The colon is now output as a separate token to ensure alignment.
        if constexpr (is_not_bool)
        {
            std::println("{:<{}}:{:>{}}{:>{}.2f} %", "Numbers of values", label_width, n, value_width, pc * 100,
                         value_width);
        }

        // First output line: label "Numbers of 0"
        const std::string label0 = "Numbers of 0";
        // The colon right after the padded label: always in the same position.
        std::println("{:<{}}:{:>{}}{:>{}.2f} %", label0, label_width, n_zero, value_width, pcz * 100, value_width);

        // Second output line: the label names the second value.
        const std::string label = "Numbers of " + bench_text(sec_val);
        std::println("{:<{}}:{:>{}}{:>{}.2f} %", label, label_width, n_values, value_width, pcv * 100, value_width);
    };

    if constexpr (chunk_config::is_shrink_enabled)
    {
        std::println("");
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
        // std::println like the other censuses: std::cout << std::fixed left
        // the stream state on for every later number the bench printed.
        std::println("{:{}}\nstatistics {}", "", name_padding, name);

        constexpr int label_width = 22;
        constexpr int value_width = 10;
        std::println("{:<{}}:{:>{}}{:>{}.2f} %", "Numbers of 0", label_width, n_zero, value_width,
                     100.0 * double(n_zero) / double(ssz), value_width);
        std::println("{:<{}}:{:>{}}{:>{}.2f} %", "Numbers of levels != 0", label_width, n, value_width,
                     100.0 * double(n) / double(ssz), value_width);
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
            std::println("{} chunk filling rate : {:.3f}%", name, sz0);
            std::fflush(stdout);
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
                std::println("");

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

/// The array members of BenchArrays (its references), in declaration order.
consteval std::vector<std::meta::info> bench_array_members()
{
    std::vector<std::meta::info> members;
    for (std::meta::info member :
         std::meta::nonstatic_data_members_of(^^BenchArrays, std::meta::access_context::unchecked()))
    {
        if (std::meta::is_reference_type(std::meta::type_of(member)))
        {
            members.push_back(member);
        }
    }
    return members;
}

/// The checksum groups, printed in this order, each under its own switch.
enum class ChecksumGroup : std::uint8_t
{
    plain,   ///< every array but the two below
    structs, ///< array_point, array_cell: the aggregate elements (--no_struct leaves them out)
    conn,    ///< array_conn: the mesh connectivity (--no_conn leaves it out)
};

/// The group of the array member named @p name.
constexpr ChecksumGroup checksum_group(std::string_view name)
{
    if (name == "array_point" || name == "array_cell")
    {
        return ChecksumGroup::structs;
    }
    return name == "array_conn" ? ChecksumGroup::conn : ChecksumGroup::plain;
}

/// `name check_sum = <hex>` for every array member of @p bench_arr in
/// @p group, under its identifier; then a newline. array_mask is not summed.
void print_checksums(const BenchArrays& bench_arr, ChecksumGroup group)
{
    template for (constexpr std::meta::info member : std::define_static_array(bench_array_members()))
    {
        constexpr std::string_view name = std::define_static_string(std::meta::identifier_of(member));
        if constexpr (name != "array_mask")
        {
            if (checksum_group(name) == group)
            {
                std::print("\n{} check_sum = {:x}", name, calculate_checksum(bench_arr.[:member:]));
            }
        }
    }
    std::println("");
}

} // namespace

void bench_print_checksums(BenchArrays& bench_arr, bool check_sum)
{
    if CONSTEXPR (check_sum)
    {
        // Every array of BenchArrays under its member's name, in declaration
        // order (reflection: an array added there is summed here); array_point
        // and array_cell after the others, only with the structs, then array_conn,
        // only with it.
        // array_bool_bits' checksum equals array_bool's: a Bit is one byte
        // holding 0 or 1, as a bool.
        print_checksums(bench_arr, ChecksumGroup::plain);
        if (bench_arr.with_struct) // --no_struct
        {
            print_checksums(bench_arr, ChecksumGroup::structs);
        }
        if (bench_arr.with_conn) // --no_conn
        {
            print_checksums(bench_arr, ChecksumGroup::conn);
        }
    }
}
