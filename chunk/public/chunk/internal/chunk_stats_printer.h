/*
 * Copyright 2026 CEA (www.cea.fr)
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

#pragma once

#include <iomanip>  // std::setw, std::setprecision, std::fixed
#include <iostream> // std::ostream
#include <sstream>  // std::stringstream
#include <string>   // std::string, std::to_string
#include <stdio.h>  // fileno: POSIX, not declared by <cstdio>
#include <unistd.h> // isatty

namespace shrink_array
{

/// @brief Formatted per-array compression stats, shared by zero / zero_one_value /
///        zero_any.
///
/// @param os            stream the report is written to.
/// @param num_tab       table number of the array (its registry index).
/// @param array_name    the array's set_debug_name(), printed before the
///                      number when not empty.
/// @param type_name     element type, as printed.
/// @param n_compressed  chunks stored in a compressed (constant) form.
/// @param nb_chunk      total chunks in the array.
/// @param n_t1          type-2 chunks (a distinct second constant). 0 for
///                      zero (no second value), which suppresses the
///                      type-2 column.
/// @param chunk_name    "zero", "zero_one_value" or "zero_any" — used in the verdict.
/// @param n_cross       cross-variant probe: how many *materialised* (un-
///                      compressed) blocks were in fact constant to a value the
///                      current variant cannot capture but a richer one would
///                      (e.g. for zero: blocks zero_one_value would have compressed).
///                      -1 means "not measured" and hides the line.
/// @param cross_name    the richer variant the probe refers to ("zero_one_value"
///                      / "zero_any"). Only used when n_cross >= 0.
/// @param bool_elements the elements are bool: their second constant can only
///                      be `true`, one for the type, so zero_one_value stores
///                      every type-2 chunk zero_any does -- zero_any is never the
///                      verdict for them.
inline void print_formatted_chunk_stats(std::ostream&      os,
                                        int                num_tab,
                                        const std::string& array_name,
                                        const std::string& type_name,
                                        int                n_compressed,
                                        int                nb_chunk,
                                        int                n_t1,
                                        const std::string& chunk_name,         // "zero", "zero_one_value", "zero_any"
                                        int                n_cross       = -1, // cross-variant probe count (-1 = none)
                                        const std::string& cross_name    = "", // richer variant the probe refers to
                                        bool               bool_elements = false) // bool: one second value, true
{
    if (nb_chunk == 0)
    {
        return;
    }

    const double pct_total = 100.0 * n_compressed / nb_chunk;
    const double pct_t1    = 100.0 * n_t1 / nb_chunk;
    const double pct_cross = (n_cross >= 0) ? 100.0 * n_cross / nb_chunk : 0.0;

    // Automatic colour-support detection (disabled when output is redirected to a file).
    const bool use_color = isatty(fileno(stdout)) != 0;

    const std::string reset  = use_color ? "\033[0m" : "";
    const std::string bold   = use_color ? "\033[1m" : "";
    const std::string grey   = use_color ? "\033[90m" : "";
    const std::string red    = use_color ? "\033[31m" : "";
    const std::string yellow = use_color ? "\033[33m" : "";
    const std::string green  = use_color ? "\033[32m" : "";
    const std::string cyan   = use_color ? "\033[36m" : "";

    // ── Verdict 1: is this variant worth it vs a plain std::vector? ───────────
    // A std::vector stores every element; the chunk store only pays off when a
    // material fraction of chunks collapse to a constant. Same threshold logic
    // for all three variants.
    std::string verdict;
    if (n_compressed == 0)
    {
        verdict = red + "no compression — plain std::vector would be lighter" + reset;
    }
    else if (pct_total < 25.0)
    {
        verdict = yellow + "compression marginal vs std::vector — plain storage may be lighter" + reset;
    }
    else if (chunk_name != "zero" && (n_t1 == 0 || pct_t1 < 10.0))
    {
        // zero_one_value/zero_any carrying no, or almost no, type-2: the second value
        // never paid.
        verdict = grey + chunk_name + " overkill — plain zero would suffice" + reset;
    }
    else if (chunk_name == "zero_any" && bool_elements)
    {
        // Type-2 chunks of bools are all-true chunks: zero_one_value's single
        // second value (true) stores every one of them. The per-chunk second
        // value of zero_any has nothing to add.
        verdict =
            grey + std::string("zero_any no better than zero_one_value for bool (one second value: true)") + reset;
    }
    else if (chunk_name != "zero" && pct_t1 < 40.0)
    {
        verdict = cyan + chunk_name + " useful" + reset;
    }
    else if (chunk_name != "zero")
    {
        verdict = green + bold + chunk_name + " highly effective" + reset;
    }
    else
    {
        // zero: no type-2 notion; verdict is purely vs std::vector.
        verdict = (pct_total >= 60.0) ? green + bold + "zero highly effective vs std::vector" + reset
                                      : cyan + "zero useful vs std::vector" + reset;
    }

    // Write everything into a local stringstream first, to guarantee thread-safety.
    std::stringstream ss;
    const auto        sflags = ss.flags();
    const auto        sprec  = ss.precision();

    // The table number changes at every re-chunk; the name, when the caller
    // gave one, is what says which array this is.
    const std::string tab_label = array_name.empty()
                                      ? "[tab_" + std::to_string(num_tab) + " " + type_name + "]"
                                      : "[" + array_name + " #" + std::to_string(num_tab) + " " + type_name + "]";

    // Step 1: the left-aligned prefix.
    ss << std::left << std::setw(40) << tab_label << reset;

    // Step 2: overall compression statistics.
    ss << std::right << std::setw(6) << n_compressed << "/" << std::setw(6) << nb_chunk << " chunks compressed ("
       << std::fixed << std::setprecision(1) << std::setw(5) << pct_total << "%)";

    // Step 3: aligned type-2 statistics.
    if (n_t1 > 0)
    {
        ss << ", " << bold << std::setw(6) << n_t1 << reset << " type-2 (" << std::setw(5) << pct_t1 << "%)";
    }
    else
    {
        ss << std::setw(23) << " ";
    }

    // Step 4: final verdict.
    ss << "  →  " << verdict;

    // Step 5: cross-variant probe ("a richer variant would have helped").
    // Shown only when measured (n_cross >= 0) and non-trivial.
    if (n_cross >= 0 && !cross_name.empty())
    {
        if (n_cross == 0)
        {
            ss << grey << "   [" << cross_name << " would not have helped]" << reset;
        }
        else
        {
            ss << yellow << "   [+" << n_cross << " blocks (" << std::fixed << std::setprecision(1) << pct_cross
               << "%) " << cross_name << " would have compressed]" << reset;
        }
    }

    ss << '\n';

    // Restore the local stream's state.
    ss.flags(sflags);
    ss.precision(sprec);

    // Write it all at once (atomic with respect to other threads).
    os << ss.str();
}

} // namespace shrink_array
