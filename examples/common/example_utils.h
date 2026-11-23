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

/**
 * @file example_utils.h
 * @brief What the example programs (simple_array, vectorisation_demo,
 *        multi_mat) and shrink_bench share, none of it about the library: the
 *        memory figure in MiB and the memory_mo.txt the scripts collect, and
 *        the random seed kept in a file so that a run can be replayed.
 */

#pragma once

#include <cstdio>  // stderr
#include <fstream> // std::ifstream, std::ofstream
#include <print>   // std::println
#include <random>  // std::mt19937, std::random_device
#include <string>  // std::string

namespace example_utils
{

/// @brief Bytes to MiB.
/// @param bytes a byte count
/// @return the same amount in MiB
inline double bytes_to_mib(double bytes)
{
    constexpr double bytes_per_mib = 1024.0 * 1024.0;
    return bytes / bytes_per_mib;
}

/// @brief Write @p mib to memory_mo.txt in the working directory, for the
///        scripts that collect a run's memory (python/bench_and_test.py...).
/// @param mib the figure, in MiB
inline void write_memory_mo(double mib)
{
    if (std::ofstream out_file("memory_mo.txt"); out_file.is_open())
    {
        out_file << mib;
    }
    else
    {
        std::println(stderr, "Error: cannot open memory_mo.txt.");
    }
}

/// @brief bytes_to_mib(), and the figure written to memory_mo.txt.
/// @param bytes a byte count
/// @return the same amount in MiB
inline double report_memory_mib(double bytes)
{
    const double mib = bytes_to_mib(bytes);
    write_memory_mo(mib);
    return mib;
}

/// @brief Write @p seed to the file @p filename.
inline void save_seed_to_file(unsigned seed, const std::string& filename)
{
    std::println("\nSave seed in file");
    if (std::ofstream file(filename); file.is_open())
    {
        file << seed;
    }
    else
    {
        std::println(stderr, "Unable to open file for saving seed.");
    }
}

/// @brief The seed saved in the file @p filename; 0 when there is none.
inline unsigned load_seed_from_file(const std::string& filename)
{
    unsigned seed = 0;
    if (std::ifstream file(filename); file.is_open())
    {
        file >> seed;
    }
    else
    {
        std::println(stderr, "Unable to open file for loading seed. --> create new one");
    }
    std::println("\nLoad seed in file");
    return seed;
}

/// @brief Seed @p rng with the seed of the file @p seed_file_name, or with a
///        new one from std::random_device, then saved there, when the file has
///        none: the next run replays this one.
inline void initialize_rng(std::mt19937& rng, const std::string& seed_file_name)
{
    unsigned seed = load_seed_from_file(seed_file_name);
    if (seed == 0)
    {
        std::random_device rd;
        seed = rd();
        save_seed_to_file(seed, seed_file_name);
    }
    rng.seed(seed);
}

} // namespace example_utils
