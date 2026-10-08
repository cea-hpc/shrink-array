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

#include <string>   // std::string, std::to_string
#include <cstdio>   // stdout, fileno (colour detection)
#include <format>   // std::format / std::format_to
#include <iterator> // std::back_inserter
#include <ostream>  // std::ostream
#include <stdio.h>  // fileno: POSIX, not declared by <cstdio>
#include <unistd.h> // isatty

namespace shrink_array
{

/// @brief Formatted per-array compression stats, shared by Chunk / Chunk2 /
///        Chunk2A.
///
/// @param os            stream the report is written to.
/// @param num_tab       table number of the array (its registry index).
/// @param array_name    the array's set_debug_name(), printed before the
///                      number when not empty.
/// @param type_name     element type, as printed.
/// @param n_compressed  chunks stored in a compressed (constant) form.
/// @param nb_chunk      total chunks in the array.
/// @param n_t1          type-2 chunks (a distinct second constant). 0 for
///                      plain Chunk (no second value), which suppresses the
///                      type-2 column.
/// @param chunk_name    "Chunk", "Chunk2" or "Chunk2A" — used in the verdict.
/// @param n_cross       cross-variant probe: how many *materialised* (un-
///                      compressed) blocks were in fact constant to a value the
///                      current variant cannot capture but a richer one would
///                      (e.g. for Chunk: blocks a Chunk2 would have compressed).
///                      -1 means "not measured" and hides the line.
/// @param cross_name    the richer variant the probe refers to ("Chunk2" /
///                      "Chunk2A"). Only used when n_cross >= 0.
/// @param bool_elements the elements are bool: their second constant can only
///                      be `true`, one for the type, so Chunk2 stores every
///                      type-2 chunk Chunk2A does -- a Chunk2A is never the
///                      verdict for them.
inline void print_formatted_chunk_stats(std::ostream&      os,
                                        int                num_tab,
                                        const std::string& array_name,
                                        const std::string& type_name,
                                        int                n_compressed,
                                        int                nb_chunk,
                                        int                n_t1,
                                        const std::string& chunk_name,         // e.g. "Chunk", "Chunk2" or "Chunk2A"
                                        int                n_cross       = -1, // cross-variant probe count (-1 = none)
                                        const std::string& cross_name    = "", // richer variant the probe refers to
                                        bool               bool_elements = false)            // bool: one second value, true
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

    const std::string RESET  = use_color ? "\033[0m" : "";
    const std::string BOLD   = use_color ? "\033[1m" : "";
    const std::string GREY   = use_color ? "\033[90m" : "";
    const std::string RED    = use_color ? "\033[31m" : "";
    const std::string YELLOW = use_color ? "\033[33m" : "";
    const std::string GREEN  = use_color ? "\033[32m" : "";
    const std::string CYAN   = use_color ? "\033[36m" : "";

    // ── Verdict 1: is this variant worth it vs a plain std::vector? ───────────
    // A std::vector stores every element; the chunk store only pays off when a
    // material fraction of chunks collapse to a constant. Same threshold logic
    // for all three variants.
    std::string verdict;
    if (n_compressed == 0)
    {
        verdict = RED + "no compression — plain std::vector would be lighter" + RESET;
    }
    else if (pct_total < 25.0)
    {
        verdict = YELLOW + "compression marginal vs std::vector — plain storage may be lighter" + RESET;
    }
    else if (chunk_name != "Chunk" && (n_t1 == 0 || pct_t1 < 10.0))
    {
        // Chunk2/Chunk2A carrying no, or almost no, type-2: the second value
        // never paid.
        verdict = GREY + chunk_name + " overkill — plain Chunk would suffice" + RESET;
    }
    else if (chunk_name == "Chunk2A" && bool_elements)
    {
        // Type-2 chunks of bools are all-true chunks: Chunk2's single second
        // value (true) stores every one of them. The per-chunk second value of
        // Chunk2A has nothing to add.
        verdict = GREY + std::string("Chunk2A no better than Chunk2 for bool (one second value: true)") + RESET;
    }
    else if (chunk_name != "Chunk" && pct_t1 < 40.0)
    {
        verdict = CYAN + chunk_name + " useful" + RESET;
    }
    else if (chunk_name != "Chunk")
    {
        verdict = GREEN + BOLD + chunk_name + " highly effective" + RESET;
    }
    else
    {
        // Plain Chunk: no type-2 notion; verdict is purely vs std::vector.
        verdict = (pct_total >= 60.0) ? GREEN + BOLD + "Chunk highly effective vs std::vector" + RESET
                                      : CYAN + "Chunk useful vs std::vector" + RESET;
    }

    // The whole line is built first and written at once, so that lines from
    // several threads do not interleave.
    // The table number changes at every re-chunk; the name, when the caller
    // gave one, is what says which array this is.
    const std::string tab_label = array_name.empty()
                                      ? "[tab_" + std::to_string(num_tab) + " " + type_name + "]"
                                      : "[" + array_name + " #" + std::to_string(num_tab) + " " + type_name + "]";

    // Left-aligned prefix, then the overall compression statistics.
    std::string line = std::format("{:<40}{}{:>6}/{:>6} chunks compressed ({:>5.1f}%)", tab_label, RESET, n_compressed,
                                   nb_chunk, pct_total);

    // Aligned type-2 statistics, or blanks of the same width.
    if (n_t1 > 0)
    {
        std::format_to(std::back_inserter(line), ", {}{:>6}{} type-2 ({:>5.1f}%)", BOLD, n_t1, RESET, pct_t1);
    }
    else
    {
        line.append(23, ' ');
    }

    // Final verdict.
    std::format_to(std::back_inserter(line), "  →  {}", verdict);

    // Cross-variant probe ("a richer variant would have helped"). Shown only
    // when measured (n_cross >= 0) and non-trivial.
    if (n_cross >= 0 && !cross_name.empty())
    {
        if (n_cross == 0)
        {
            std::format_to(std::back_inserter(line), "{}   [{} would not have helped]{}", GREY, cross_name, RESET);
        }
        else
        {
            std::format_to(std::back_inserter(line), "{}   [+{} blocks ({:.1f}%) {} would have compressed]{}", YELLOW,
                           n_cross, pct_cross, cross_name, RESET);
        }
    }

    line += '\n';
    os << line;
}

} // namespace shrink_array
