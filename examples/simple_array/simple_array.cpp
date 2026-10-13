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

#include <bit>      // std::bit_cast (calculate_checksum)
#include <cassert>  // assert
#include <chrono>   // std::chrono::steady_clock
#include <cmath>    // std::cos, std::sin, std::sqrt
#include <cstdint>  // std::int64_t, uint64_t
#include <cstdlib>  // std::abort
#include <iostream> // std::cout, std::cerr
#include <random>   // std::mt19937, std::uniform_int_distribution
#include <array>    // std::array
#include <fstream>  // std::ifstream, std::ofstream
#include <string>   // std::string, std::to_string

#include <iomanip>        // std::setprecision, std::fixed
#include <type_traits>    // std::conditional_t (Array)
#include <vector>         // std::vector
#include <sys/resource.h> // getrusage

// Both executables have the chunk headers: simple_array_vector links
// chunk_vector, simple_array_chunk chunk_v1.
#include "chunk/param_chunk.h"                // ParamChunk
#include "chunk/array_chunk.h"                // ArrayChunk
#include "chunk/PMP_CHUNK.h"                  // CHUNK_PMP_MAX_THREADS
#include "chunk/chunk_def.h"                  // ChunkDef
#include "chunk/chunk_std.h"                  // destroy_block_chunk
#include "chunk/config.h"                     // chunk_config::use_chunk*, stats
#include "chunk/internal/chunk_size_config.h" // static_chunk_size


/// @brief splitmix64's finaliser: a bijection on 64 bits that spreads every
///        input bit over the whole output.
constexpr uint64_t checksum_mix(uint64_t x) noexcept
{
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30U)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27U)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31U);
}

/// @brief Checksum of the non-zero elements of @p vec, the same for ANY
///        number of threads.
/// @details A wrapping sum of mix(bits ^ mix(index)): a sum does not depend on
///          the order the reduction combines it in, and the index makes an
///          element at the wrong position change the result. The previous
///          step (rotate, then add) depended on the order, and ran without
///          `omp for`: every thread did the whole loop, and the printed value
///          was Nthreads x the serial one.
template <typename V>
uint64_t calculate_checksum(const V& vec)
{
    uint64_t checksum = 0;

    using T       = V::value_type;
    const auto nn = static_cast<std::int64_t>(vec.size());

    constexpr T zero = 0;

#pragma omp parallel for reduction(+ : checksum)
    for (std::int64_t k = 0; k < nn; ++k)
    {
        const T element = vec[k];

        if (element != zero)
        {
            // The bits of the value: bit_cast, not a uint64_t* on a double
            // (undefined behaviour under strict aliasing).
            checksum += checksum_mix(std::bit_cast<uint64_t>(element) ^ checksum_mix(static_cast<uint64_t>(k)));
        }
    }
    return checksum;
}

double convert_memory(double memoryValue)
{
    constexpr double BYTE_TO_KB = 1.0 / 1024;
    constexpr double KB_TO_MB   = 1.0 / 1024;

    const double kilobytes = memoryValue * BYTE_TO_KB;
    const double megabytes = kilobytes * KB_TO_MB;

    std::ofstream fichier("memory_mo.txt");

    if (fichier.is_open())
    {
        fichier << megabytes;
        fichier.close();
    }
    else
    {
        std::cerr << "Error: cannot open the file.\n";
    }

    return megabytes;
}

// Save seed to file
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
        std::cerr << "Unable to open file for saving seed." << '\n';
    }
}

// Load seed from file
unsigned load_seed_to_file(const std::string& filename)
{
    unsigned      seed = 0;
    std::ifstream file(filename);
    if (file.is_open())
    {
        file >> seed;
        file.close();
    }
    else
    {
        std::cerr << "Unable to open file for loading seed. --> create new one" << '\n';
    }
    std::cout << "\nLoad seed in file\n";

    return seed;
}

// Initialize random number generator with saved or new seed
void initializeRNG(std::mt19937& rng, const std::string& seed_file_name)
{
    // Try to load seed from file
    unsigned seed = load_seed_to_file(seed_file_name);
    if (seed == 0)
    {
        // If loading failed, generate a new seed
        std::random_device rd;
        seed = rd();
        // Save the new seed to file
        save_seed_to_file(seed, seed_file_name);
    }
    // Initialize RNG with the seed
    rng.seed(seed);
}

using namespace shrink_array;

/// simple_array_chunk (CHUNK_VARIANT_V1, or _V2) runs on ArrayChunk, simple_array_vector
/// on std::vector.
constexpr bool use_chunk_array = chunk_config::use_chunk || chunk_config::use_chunk2;
using Array                    = std::conditional_t<use_chunk_array, ArrayChunk<double>, std::vector<double>>;

/// @brief Calls @p f on @p arrays in simple_array_chunk only. @p f is a generic
///        lambda using ArrayChunk's members (compress, stats): its body is then
///        instantiated with ArrayChunk alone. The same call in a discarded branch
///        of main(), which is not a template, would be compiled for std::vector.
template <typename F, typename... A>
void if_chunk(F&& f, A&... arrays)
{
    if constexpr (use_chunk_array)
    {
        f(arrays...);
    }
}

int main()
{
    if constexpr (use_chunk_array)
    {
        constexpr int chunk_size = static_cast<int>(shrink_array::static_chunk_size);

        assert(chunk_size <= ChunkDef::size_max);


        if constexpr (chunk_size == 0)
        {
            std::cerr << "The chunk size can't be null\n";
        }

        if (!ParamChunk::check_chunk_size(chunk_size))
        {
            std::cerr << "The chunk size must be a power of two or a multiple of 64";
        }

        if constexpr (chunk_size < ParamChunk::size_min || chunk_size > ParamChunk::size_max)
        {
            std::cerr << "The chunk size must must be superior or equal to " << ParamChunk::size_min
                      << " and inferior or equal to " << ParamChunk::size_max << "\n";
            std::abort();
        }

        ParamChunk::init_indexes(true);
    }

    constexpr int sz = 1000000000;
    //  constexpr int sz = 100000;
    constexpr int iter_max = 10000;

    constexpr int freq_check_sum = iter_max / 5;

    constexpr int freq_compress = iter_max / 10; // chunk arrays only
    // constexpr int freq_raz = iter_max / 500;
    constexpr int freq_raz = iter_max / 50;
    constexpr int interval = 20000;

    const int max_thr = CHUNK_PMP_MAX_THREADS;

    std::cout << "Number of threads : " << max_thr << '\n';

    auto start = std::chrono::steady_clock::now();

    std::cout << "Initialisation ..." << '\n';
    std::cout << "Creation of arrays ..." << '\n';

    // constexpr int interval = 2000;
    Array tab1(sz);
    Array tab2(sz);
    Array tab3(sz);

    auto                          end             = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    start                                         = end;
    std::cout << "\nElapsed time creation : " << elapsed_seconds.count() << "s\n";

#pragma omp          parallel for default(shared)
    for (int i = 0; i < sz; ++i)
    {
        tab1[i] = tab2[i] = tab3[i] = 0;
    }

    end             = std::chrono::steady_clock::now();
    elapsed_seconds = end - start;
    start           = end;
    std::cout << "\nElapsed time Initialisation to zero: " << elapsed_seconds.count() << "s\n";

    const int min = 1;
    const int max = sz;

    // Initialize the random number generator
    const std::string str            = std::to_string(iter_max);
    const std::string seed_file_name = "seed" + str + ".txt"; // File to store the seed
    std::mt19937      rng;                                    // Random number generator
    initializeRNG(rng, seed_file_name);

    // Create a uniform distribution for interval sizes
    std::uniform_int_distribution sizeDistribution(1, interval);

    // Create a uniform distribution for the starting points of the intervals
    std::uniform_int_distribution startDistribution(min, max);

    std::cout << "Iteration ..." << '\n';

    std::array<int, iter_max> tab_size{};
    std::array<int, iter_max> tab_start{};
    std::array<int, iter_max> tab_nba1{};
    std::array<int, iter_max> tab_nba2{};

    for (int k = 0; k < iter_max; ++k)
    {
        const int size   = sizeDistribution(rng);
        const int start0 = startDistribution(rng);
        const int end0   = start0 + size;
        tab_size[k]      = size;
        tab_start[k]     = start0;

        std::uniform_int_distribution<int> uni1(start0, end0);
        tab_nba1[k] = uni1(rng);
        tab_nba2[k] = uni1(rng);
    }

    end             = std::chrono::steady_clock::now();
    elapsed_seconds = end - start;
    start           = end;
    std::cout << "\nElapsed time distribution : " << elapsed_seconds.count() << "s\n";

    double v = 0;

    for (int k = 0; k < iter_max; ++k)
    {
        const int size   = tab_size[k];
        const int start0 = tab_start[k];

        int end0 = start0 + size;

        if (end0 > sz)
        {
            end0 = sz;
        }

        if (k != 0 && k % freq_raz == 0)
        {
            std::cout << "Interval "
                      << ": [" << start0 << ", " << end0 << "]"
                      << " size : " << size << "\n";

            int i = start0;

            for (; i < sz; ++i)
            {
                if (tab1[i] != 0)
                {
                    break;
                }
            }

            // Bounded: i can already be sz here (start0 == sz is drawn by
            // startDistribution(1, sz), or no non-zero was found), and a
            // non-zero run can extend to sz - 1.
            while (i < sz && tab1[i] != 0)
            {
                tab1[i] = tab2[i] = tab3[i] = 0;
                i++;
            }
        }
        else
        {
#pragma omp parallel reduction(+ : v)
            {
                // To simulate comportement of a code where we read more than we write
#pragma omp for schedule(static) nowait
                for (int i = start0; i < end0; ++i)
                {
                    tab1[i] = std::sqrt(static_cast<double>(i));
                    if (i > tab_nba1[k])
                    {
                        tab2[i] = std::cos(double(2 * i));
                    }
                    if (i > tab_nba2[k])
                    {
                        tab3[i] = std::sin(tab1[i] + tab2[i]);
                    }
                }

                for (int p = 0; p < 5; ++p)
                {
                    const auto val = static_cast<double>(p);

#pragma omp for schedule(static) nowait
                    for (int i = start0; i < end0; ++i)
                    {
                        v += std::cos(tab1[i]) + std::sin(tab2[i]) + std::atan(tab3[i]) + std::cos(val);
                    }
                }
            }
        }

        std::cout << "k : " << k << " v = " << v << '\n';

        if (k % freq_check_sum == 0)
        {
            std::cout << "\ntab1 check_sum = " << calculate_checksum(tab1) << "\n";
            std::cout << "tab2 check_sum = " << calculate_checksum(tab2) << "\n";
            std::cout << "tab3 check_sum = " << calculate_checksum(tab3) << '\n';
        }

        if (k % freq_compress == 0)
        {
            if_chunk(
                [](auto&... tabs)
                {
                    std::cout << "Compression\n";
                    (tabs.compress(), ...);
                },
                tab1, tab2, tab3);
        }
    }

    std::cout << "\nv = " << v << '\n';

    end             = std::chrono::steady_clock::now();
    elapsed_seconds = end - start;
    std::cout << "\nElapsed time computation : " << elapsed_seconds.count() << "s\n";

    std::cout << tab2[2] << '\n';

    if_chunk(
        [](const auto&... tabs)
        {
            for (const double sz0 : {tabs.stats()...})
            {
                std::cout << "Chunk filling rate : " << std::fixed << std::setprecision(2) << sz0 << "%" << '\n';
            }

            if constexpr (chunk_config::stats)
            {
                ParamChunk::ChunkStats();
            }
        },
        tab1, tab2, tab3);

    struct rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) == 0)
    {
        const long rss = usage.ru_maxrss; // Resident Set Size in kilobytes
        std::cout << "\nMemory consumption : " << convert_memory(static_cast<double>(rss * 1024)) << " Mo \n\n";
    }
    else
    {
        std::cerr << "Unable to retrieve information from memory." << '\n';
    }

    tab1.clear();
    tab2.clear();
    tab3.clear();

    if constexpr (use_chunk_array)
    {
        destroy_block_chunk();
    }
    return 0;
}
