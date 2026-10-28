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

#pragma once

#include <algorithm> // std::max
#include <cstddef>   // std::size_t
#include <map>       // std::map
#include <string>    // std::string
#include <format>    // std::format (format_memory)
#include <print>     // std::println (print)

#include "chunk/term_color.h" // tc (TermColor)
#include "chunk/PMP_CHUNK.h"  // CHUNK_PMP_* macros — used HERE, not via includers

namespace shrink_array
{

/// @brief Per element type figures of the compression lists, keyed by type name, and their memory report.
class GlobalStatsChunkList
{
  public:
    /// @brief Figures recorded for one element type.
    struct TypeStats
    {
        long   allocated_chunks;  ///< Total number of data chunks managed
        int    threshold_per_thr; ///< Compression list threshold per thread (entries)
        size_t element_size;      ///< Size in bytes of a single entry in the op-list (tuple)
    };

    /// @brief Record one element type's figures under @p name.
    void record(const std::string& name, long somme, int max_chunks, size_t elem_size)
    {
        all_stats_[name] = {.allocated_chunks = somme, .threshold_per_thr = max_chunks, .element_size = elem_size};
    }

    /**
     * @brief Prints a detailed report of global memory usage for management lists.
     */
    void print() const
    {
        if (all_stats_.empty())
        {
            return;
        }

        // The type column fits the longest name (zero_one_value::Chunk<std::complex<double> > is 57).
        std::size_t longest_name = 0;
        for (const auto& entry : all_stats_)
        {
            longest_name = std::max(longest_name, entry.first.size());
        }
        const int     col_type    = std::max(50, static_cast<int>(longest_name) + 2);
        constexpr int col_data    = 18;
        constexpr int col_list    = 20;
        constexpr int col_mem     = 18;
        const int     total_width = col_type + col_data + col_list + col_mem;

        long double total_metadata_mem = 0;
        const int   nb_threads         = CHUNK_PMP_MAX_THREADS;

        const std::string rule_eq(total_width, '=');
        const std::string rule_dash(total_width, '-');

        std::println("\n{}{}{}", tc.cyan, rule_eq, tc.reset);
        std::println("{}{}{:{}}GLOBAL COMPRESSION LISTS MONITORING{}", tc.bold, tc.cyan, "", (total_width - 35) / 2,
                     tc.reset);
        std::println("{} Configuration: {}{}{}{} active threads{}", tc.white, tc.bold, nb_threads, tc.reset, tc.white,
                     tc.reset);
        std::println("{}{}{}", tc.cyan, rule_eq, tc.reset);

        std::println("{}{:<{}}{:>{}}{:>{}}{:>{}}{}", tc.bold, "Chunk Type Instance", col_type, "Allocated Chunks",
                     col_data, "Op-List Capacity", col_list, "List Memory", col_mem, tc.reset);

        std::println("{}", rule_dash);

        for (const auto& [name, stats] : all_stats_)
        {
            const long        total_list_capacity = static_cast<long>(nb_threads) * stats.threshold_per_thr;
            const long double type_list_bytes     = static_cast<long double>(total_list_capacity) * stats.element_size;
            total_metadata_mem += type_list_bytes;

            std::println("{:<{}}{}{:>{}}{}{}{:>{}}{}{}{:>{}}{}", name, col_type, tc.green, stats.allocated_chunks,
                         col_data, tc.reset, tc.yellow, total_list_capacity, col_list, tc.reset, tc.red,
                         format_memory(type_list_bytes), col_mem, tc.reset);
        }

        std::println("{}", rule_dash);
        std::println("{}AGGREGATED METADATA OVERHEAD:{}", tc.bold, tc.reset);
        std::println("{:<40}: {}{}{}{}", " Total footprint for tracking lists", tc.bold, tc.green,
                     format_memory(total_metadata_mem), tc.reset);
        std::println("{}{}{}\n", tc.cyan, rule_eq, tc.reset);
    }

  private:
    /// @brief @p bytes as a human-readable string (Ko / Mo / Go, powers of 1024).
    static std::string format_memory(long double bytes)
    {
        const long double kib = bytes / 1024.0L;
        const long double mib = kib / 1024.0L;
        const long double gib = mib / 1024.0L;

        if (gib >= 1.0L)
        {
            return std::format("{:.2f} Go", gib);
        }
        if (mib >= 1.0L)
        {
            return std::format("{:.2f} Mo", mib);
        }
        if (kib >= 1.0L)
        {
            return std::format("{:.2f} Ko", kib);
        }
        return std::format("{} bytes", static_cast<long long>(bytes));
    }

    std::map<std::string, TypeStats> all_stats_;
};

} // namespace shrink_array