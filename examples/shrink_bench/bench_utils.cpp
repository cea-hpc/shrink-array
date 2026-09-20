/*
 * Copyright 2024-2026 CEA (www.cea.fr)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <sys/resource.h>
#include <unistd.h>
#include <malloc.h>
#include <cstdio>
#include <vector>

#include "bench_utils.h"

#include "chunk/all_type_chunk.h"           // reuse_pool_bytes, the retention split of memory_consumption
#ifdef FAST_ALLOC
#include "fast_allocators/slab_arena.h" // report_all_arenas, its slab-side counterpart
#endif
// -----------------------------------------------------------------------------
// File-scope globals (declared in bench_utils.h).
// -----------------------------------------------------------------------------
int  __mem_page_size = 0;
bool __debug_flag    = false;

// FIX: "--test_stl" was missing here while main() has always had a branch to
// parse it. The filter below (ranges::find over valid_args) runs FIRST, so the
// flag was rejected as unknown and that branch could never fire: the
// STL-interface and read-idempotence self-checks had NEVER run. Worse, the
// binary exited 0 after printing "Unknown argument", so bench_and_test.py's STL
// phase went green without executing anything.
const std::vector<std::string> valid_args = {"--no_io",    "--no_compress", "--write_trace", "--read_trace",
                                             "--multiply", "--divisor",     "--iter_max",    "--mode",
                                             "--test_stl", "--no_struct",   "--help"};

// -----------------------------------------------------------------------------
// Seed persistence.
// -----------------------------------------------------------------------------
void save_seed_to_file(unsigned seed, const std::string& filename)
{
    std::cout << "\nSave seed in file\n";

    std::ofstream file(filename);
    if (file.is_open())
    {
        file << seed;
        file.close();
    }
    else
    {
        std::cerr << "Unable to open file for saving seed." << std::endl;
    }
}

unsigned load_seed_to_file(const std::string& filename)
{
    unsigned seed = 0;
    if (std::ifstream file(filename); file.is_open())
    {
        file >> seed;
        file.close();
    }
    else
    {
        std::cerr << "Unable to open file for loading seed. --> create new one" << std::endl;
    }
    std::cout << "\nLoad seed in file\n";

    return seed;
}

void initializeRNG(std::mt19937& rng, const std::string& seed_file_name)
{
    unsigned seed = load_seed_to_file(seed_file_name);
    if (seed == 0)
    {
        // If loading failed, generate a new seed
        std::random_device rd;
        seed = rd();
        save_seed_to_file(seed, seed_file_name);
    }
    rng.seed(seed);
}

// -----------------------------------------------------------------------------
// Memory report.
// -----------------------------------------------------------------------------
double convert_memory(double memoryValue)
{
    constexpr double BYTE_TO_KB = 1.0 / 1024;
    constexpr double KB_TO_MB   = 1.0 / 1024;

    const double kilobytes = memoryValue * BYTE_TO_KB;
    const double megabytes = kilobytes * KB_TO_MB;

    if (std::ofstream fichier("memory_mo.txt"); fichier.is_open())
    {
        fichier << megabytes;
        fichier.close();
    }
    else
    {
        std::cerr << "Error : can't open the file.\n";
    }

    return megabytes;
}

void memory_consumption(const std::string& msg)
{
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) == 0)
    {
        const long rss = usage.ru_maxrss; // PEAK resident set size in KB (high-water mark)

        // Also report the CURRENT resident size. ru_maxrss is a high-water mark
        // that never comes back down, so on its own it cannot distinguish
        // "this configuration uses more memory" from "this configuration had a
        // transient peak". re-chunking has exactly such a peak -- rechunk_to
        // builds the new array while the old one is still alive -- so every
        // comparison made on ru_maxrss alone overstates its steady-state cost.
        // Return what glibc is holding in its arenas before reading RSS.
        // free() hands blocks back to malloc's free lists, not to the kernel, so
        // a correctly drained reuse pool still shows up as resident. Without
        // this the figure conflates "the library retains buffers" with "the
        // allocator has not given the pages back", and the two need different
        // fixes.
        malloc_trim(0);

        // /proc/self/statm: "size resident shared text lib data dt", in pages.
        // Read through a stream so a short or malformed line fails the
        // extraction instead of leaving a half-converted value behind.
        long cur_kb = 0;
        if (std::ifstream statm("/proc/self/statm"); statm.is_open())
        {
            long total_pages = 0;
            long res_pages   = 0;
            if (statm >> total_pages >> res_pages)
                cur_kb = res_pages * (sysconf(_SC_PAGESIZE) / 1024);
        }

        std::cout << "Memory consumption " << msg << " : " << convert_memory(static_cast<double>(rss) * 1024.0)
                  << " Mo (peak)";
        if (cur_kb > 0)
            std::cout << ", " << convert_memory(static_cast<double>(cur_kb) * 1024.0) << " Mo (current)";
        // Where the resident memory is, at this moment, as far as the library
        // knows: the buffers parked in the per-rank reuse pools (blocks a
        // compression released and nothing took back yet), and the buffers
        // still materialised (allocated or reused, not yet released) -- the
        // latter from the CHUNK_STATS byte counters, so only in a stats build.
        // The remainder against "current" is the chunk headers and per-chunk
        // tables, the slabs' unused capacity, and the bench's own arrays.
        if constexpr (chunk_config::is_shrink_enabled)
        {
            const std::size_t parked = shrink_array::SingletonChk<shrink_array::AllTypeChunk>::getRef().reuse_pool_bytes();
            std::cout << ", parked " << convert_memory(static_cast<double>(parked)) << " Mo";
            // Preprocessor guard, not `if constexpr`: ChunkStatsThr and
            // reduce_chunk_stats() only exist in a CHUNK_STATS build, and a
            // discarded branch of a non-template function still looks its
            // names up.
#ifdef CHUNK_STATS
            {
                long live_bytes  = 0;
                long live_chunks = 0;
                for (int v = 0; v < shrink_array::CHUNK_STATS_N_VARIANT; ++v)
                {
                    const shrink_array::ChunkStatsThr t = shrink_array::reduce_chunk_stats(v);
                    live_bytes += t.bytes_alloc + t.bytes_reused - t.bytes_freed;
                    live_chunks += t.create - t.compress;
                }
                std::cout << ", materialised " << convert_memory(static_cast<double>(live_bytes)) << " Mo ("
                          << live_chunks << " chunks)";
            }
#endif
        }
        std::cout << " \n";
        // The slab arenas' own view (stderr): mapped / live / free-listed slab
        // bytes. Against "materialised + parked" above, `live` says how much of
        // the slabs handed out is neither: blocks freed back to their slab, and
        // slab tails never carved. Fast-alloc builds only.
      //  if constexpr (build_config::use_fast_alloc)
        #ifdef FAST_ALLOC
            fast_alloc::report_all_arenas(msg.c_str());
        #endif
    }
    else
    {
        std::cerr << "Unable to retrieve information from memory." << std::endl;
    }
}

// -----------------------------------------------------------------------------
// Correlation shapes used to generate synthetic values.
// -----------------------------------------------------------------------------
int linear_correlation(int i, int sz, int sz1)
{
    return static_cast<int>((static_cast<double>(sz1) / sz) * i);
}

int exponential_correlation(int i, int sz, int sz1, double p)
{
    return static_cast<int>(sz1 * pow(static_cast<double>(i) / sz, p));
}

int interpolation(int i, int sz, int sz1)
{
    return static_cast<int>(sz1 * sqrt(static_cast<double>(i) / sz));
}

// -----------------------------------------------------------------------------
// CLI helpers.
// -----------------------------------------------------------------------------
int levenshtein(const std::string& a, const std::string& b)
{
    const size_t len_a = a.size();
    const size_t len_b = b.size();
    std::vector  dp(len_a + 1, std::vector<int>(len_b + 1));

    for (size_t i = 0; i <= len_a; ++i)
        for (size_t j = 0; j <= len_b; ++j)
        {
            if (i == 0)
                dp[i][j] = static_cast<int>(j);
            else if (j == 0)
                dp[i][j] = static_cast<int>(i);
            else
            {
                const int subst = (a[i - 1] != b[j - 1]) ? 1 : 0;
                dp[i][j]        = std::min({dp[i - 1][j - 1] + subst, dp[i - 1][j] + 1, dp[i][j - 1] + 1});
            }
        }
    return dp[len_a][len_b];
}

void print_help()
{
    std::cout << "Available options:\n";
    for (const auto& arg : valid_args)
    {
        std::cout << "  " << arg << "\n";
    }
}

std::string suggest_closest(const std::string& input)
{
    std::string closest;
    int         min_distance = std::numeric_limits<int>::max();

    for (const auto& arg : valid_args)
    {
        const int dist = levenshtein(input, arg);
        if (dist < min_distance)
        {
            min_distance = dist;
            closest      = arg;
        }
    }
    return closest;
}
