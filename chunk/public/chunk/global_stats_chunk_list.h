//
// Created by ballereaup on 22/04/2026.
//
#pragma once
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

#include <iomanip>
#include <iostream>
#include <map>
#include <sstream> // std::ostringstream
#include <string>

#include "chunk/term_color.h"
#include "chunk/PMP_CHUNK.h" // CHUNK_PMP_* macros — used HERE, not via includers

namespace shrink_array
{

struct GlobalStatsChunkList
{
    // const std::string RESET  = "\033[0m";
    // const std::string BOLD   = "\033[1m";
    // const std::string RED    = "\033[31m";
    // const std::string GREEN  = "\033[32m";
    // const std::string YELLOW = "\033[33m";
    // const std::string CYAN   = "\033[36m";
    // const std::string WHITE  = "\033[37m";

    struct TypeStats
    {
        long   allocated_chunks;  // Total number of data chunks managed
        int    threshold_per_thr; // Compression list threshold per thread
        size_t element_size;      // Size of a single entry in the op-list (tuple)
    };

    std::map<std::string, TypeStats> all_stats;

    /// @brief Record one element type's figures under @p name.
    void record(const std::string& name, long somme, int max_chunks, size_t elem_size)
    {
        all_stats[name] = {somme, max_chunks, elem_size};
    }

    /// @brief @p bytes as a human-readable string (KiB / MiB / GiB).
    static std::string format_memory(long double bytes)
    {
        const long double kib = bytes / 1024.0L;
        const long double mib = kib / 1024.0L;
        const long double gib = mib / 1024.0L;

        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2);

        if (gib >= 1.0L)
            oss << gib << " Go";
        else if (mib >= 1.0L)
            oss << mib << " Mo";
        else if (kib >= 1.0L)
            oss << kib << " Ko";
        else
            oss << (long long)bytes << " bytes";

        return oss.str();
    }

    /**
     * @brief Prints a detailed report of global memory usage for management lists.
     */

    void print() const
    {
        if (all_stats.empty())
            return;

        constexpr int col_type    = 50;
        constexpr int col_data    = 18;
        constexpr int col_list    = 20;
        constexpr int col_mem     = 18;
        constexpr int total_width = col_type + col_data + col_list + col_mem;

        long double total_metadata_mem = 0;
        const int   nb_threads         = CHUNK_PMP_MAX_THREADS;

        // Header Cyan
        std::cout << "\n" << tc.CYAN << std::string(total_width, '=') << tc.RESET << "\n";
        std::cout << tc.BOLD << tc.CYAN << std::string((total_width - 35) / 2, ' ')
                  << "GLOBAL COMPRESSION LISTS MONITORING" << tc.RESET << "\n";
        std::cout << tc.WHITE << " Configuration: " << tc.BOLD << nb_threads << tc.RESET << tc.WHITE
                  << " active threads" << tc.RESET << "\n";
        std::cout << tc.CYAN << std::string(total_width, '=') << tc.RESET << "\n";

        // Titles Bold White
        std::cout << tc.BOLD << std::left << std::setw(col_type) << "Chunk Type Instance" << std::right
                  << std::setw(col_data) << "Allocated Chunks" << std::right << std::setw(col_list)
                  << "Op-List Capacity" << std::right << std::setw(col_mem) << "List Memory" << tc.RESET << "\n";

        std::cout << std::string(total_width, '-') << "\n";

        for (auto const& [name, stats] : all_stats)
        {
            const long        total_list_capacity = static_cast<long>(nb_threads) * stats.threshold_per_thr;
            const long double type_list_bytes     = static_cast<long double>(total_list_capacity) * stats.element_size;
            total_metadata_mem += type_list_bytes;

            std::cout << std::left << std::setw(col_type) << name << tc.GREEN << std::right << std::setw(col_data)
                      << stats.allocated_chunks << tc.RESET << tc.YELLOW << std::right << std::setw(col_list)
                      << total_list_capacity << tc.RESET << tc.RED << std::right << std::setw(col_mem)
                      << format_memory(type_list_bytes) << tc.RESET << "\n";
        }

        std::cout << std::string(total_width, '-') << "\n";
        std::cout << tc.BOLD << "AGGREGATED METADATA OVERHEAD:" << tc.RESET << "\n";
        std::cout << std::left << std::setw(40) << " Total footprint for tracking lists"
                  << ": " << tc.BOLD << tc.GREEN << format_memory(total_metadata_mem) << tc.RESET << "\n";
        std::cout << tc.CYAN << std::string(total_width, '=') << tc.RESET << "\n" << std::endl;
    }
};

} // namespace shrink_array