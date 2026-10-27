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
#include "bench_prelude.h" // zero_one_value::Proxy<T>::share_info_array (share_info line)
#include "bench_phases.h"  // bench_stats, bench_print_checksums (defined here)
#include "bench_utils.h"   // calculate_checksum, bench_text

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
    auto& tab_conn      = bench_arr.tab_conn;

    auto lbd_fill_values = [&](const std::string& name, const auto& tab, auto sec_val)
    {
        using value_type           = std::remove_reference_t<std::remove_const_t<decltype(tab)>>::value_type;
        constexpr bool is_not_bool = !shrink_array::is_bool_like_v<value_type>; // bool, Bit
        constexpr bool is_complex  = std::is_same_v<value_type, ComplexDouble>;

        const Int ssz = tab.size();

        // `omp parallel for schedule(static) reduction(+ : ...)` with OpenMP.
        const auto census = shrink_array::pmp::parallel_sum<Int, 3>(Int{0}, ssz, shrink_array::pmp::Options{},
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
        // elements, tab_short sz / 2, tab_mask mask_scale * sz, tab_conn whole rows.
        const double pc  = double(nb) / double(ssz);
        const double pcz = double(nbz) / double(ssz);
        const double pcv = double(nbv) / double(ssz);

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
            std::println("{:<{}}:{:>{}}{:>{}.2f} %", "Numbers of values", label_width, nb, value_width, pc * 100,
                         value_width);
        }

        // First output line: label "Numbers of 0"
        const std::string label0 = "Numbers of 0";
        // The colon right after the padded label: always in the same position.
        std::println("{:<{}}:{:>{}}{:>{}.2f} %", label0, label_width, nbz, value_width, pcz * 100, value_width);

        // Second output line: the label names the second value.
        const std::string label = "Numbers of " + bench_text(sec_val);
        std::println("{:<{}}:{:>{}}{:>{}.2f} %", label, label_width, nbv, value_width, pcv * 100, value_width);
    };

    if constexpr (chunk_config::is_shrink_enabled)
    {
        std::println("");
    }

    // tab_quant has one second value per level it lands on (up to
    // bench_double_bands), not one: its census is zeros and the rest.
    auto lbd_fill_quant = [](const std::string& name, const auto& tab)
    {
        const Int ssz = tab.size();

        // `omp parallel for schedule(static) reduction(+ : nbz)` with OpenMP.
        const Int nbz = shrink_array::pmp::parallel_sum<Int, 1>(Int{0}, ssz, shrink_array::pmp::Options{},
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
        // std::println like the other censuses: std::cout << std::fixed left
        // the stream state on for every later number the bench printed.
        std::println("{:{}}\nstatistics {}", "", name_padding, name);

        constexpr int label_width = 22;
        constexpr int value_width = 10;
        std::println("{:<{}}:{:>{}}{:>{}.2f} %", "Numbers of 0", label_width, nbz, value_width,
                     100.0 * double(nbz) / double(ssz), value_width);
        std::println("{:<{}}:{:>{}}{:>{}.2f} %", "Numbers of levels != 0", label_width, nb, value_width,
                     100.0 * double(nb) / double(ssz), value_width);
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
            std::println("{} chunk filling rate : {:.3f}%", name, sz0);
            std::fflush(stdout);
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
            if (bench_arr.with_conn) // --no_conn
            {
                lbd_fill_values("tab_conn", tab_conn, second_value_i);
            }
            if (bench_arr.with_struct) // --no_struct
            {
                lbd_fill_values("tab_point", tab_point, second_value_p);
                lbd_fill_values("tab_cell", tab_cell, second_value_cell);
            }

            if (chunk_config::is_shrink_enabled)
            {
                std::println("");

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
                if (bench_arr.with_conn)
                {
                    lbd_fill_chunk("tab_conn", tab_conn);
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
    structs, ///< tab_point, tab_cell: the aggregate elements (--no_struct leaves them out)
    conn,    ///< tab_conn: the mesh connectivity (--no_conn leaves it out)
};

/// The group of the array member named @p name.
constexpr ChecksumGroup checksum_group(std::string_view name)
{
    if (name == "tab_point" || name == "tab_cell")
    {
        return ChecksumGroup::structs;
    }
    return name == "tab_conn" ? ChecksumGroup::conn : ChecksumGroup::plain;
}

/// `name check_sum = <hex>` for every array member of @p bench_arr in
/// @p group, under its identifier; then a newline. tab_mask is not summed.
void print_checksums(const BenchArrays& bench_arr, ChecksumGroup group)
{
    template for (constexpr std::meta::info member : std::define_static_array(bench_array_members()))
    {
        constexpr std::string_view name = std::define_static_string(std::meta::identifier_of(member));
        if constexpr (name != "tab_mask")
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
        // order (reflection: an array added there is summed here); tab_point
        // and tab_cell after the others, only with the structs, then tab_conn,
        // only with it.
        // tab_bool_bits' checksum equals tab_bool's: a Bit is one byte
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
