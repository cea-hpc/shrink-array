/*
 * Copyright 2024-2025 CEA (www.cea.fr)
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

#include <bit> // std::bit_cast (calculate_checksum)
#include <chrono>
#include <cstdio>
#include <print>
#include <random>
#include <array>
#include <fstream>
#include <complex>

#include <type_traits> // std::conditional_t (Array)
#include <vector>
#include <sys/resource.h>
#include <unistd.h>

// Both executables have the chunk headers: simple_array_vector links
// chunk_vector, simple_array_chunk chunk_v1.
#include "chunk/param_chunk.h"
#include "chunk/array_chunk.h"
#include "chunk/PMP_CHUNK.h" // CHUNK_PMP_MAX_THREADS

int __mem_page_size = 0;

template <typename V>
uint64_t calculate_checksum(const V& vec)
{
    uint64_t checksum = 0;

    using T      = V::value_type;
    const int nn = vec.size();

    constexpr T zero = 0;

#pragma omp parallel reduction(+ : checksum)
    for (int k = 0; k < nn; ++k)
    {
        const T element = vec[k];

        if (element != zero)
        {
            // The bits of the value: bit_cast, not a uint64_t* on a double
            // (undefined behaviour under strict aliasing).
            checksum = (checksum >> 1) + ((checksum & 1) << 63);
            checksum += std::bit_cast<uint64_t>(element);
        }
    }
    return checksum;
}

double convert_memory(double memoryValue)
{
    constexpr double BYTE_TO_KB = 1.0 / 1024;
    constexpr double KB_TO_MB   = 1.0 / 1024;
    // const double MB_TO_GB   = 1.0 / 1024;

    const double kilobytes = memoryValue * BYTE_TO_KB;
    const double megabytes = kilobytes * KB_TO_MB;
    //  double gigabytes = megabytes * MB_TO_GB;

    std::ofstream fichier("memory_mo.txt");

    if (fichier.is_open())
    {
        // Écrire un scalaire dans le fichier

        fichier << megabytes;

        // Fermer le fichier
        fichier.close();
    }
    else
    {
        std::println(stderr, "Erreur : Impossible d'ouvrir le fichier.");
    }

    return megabytes;
}

// Save seed to file
void save_seed_to_file(unsigned seed, const std::string& filename)
{
    std::println("\nSave seed in file");

    std::ofstream file(filename);
    if (file.is_open())
    {
        file << seed;
        file.close();
    }
    else
    {
        std::println(stderr, "Unable to open file for saving seed.");
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
        std::println(stderr, "Unable to open file for loading seed. --> create new one");
    }
    std::println("\nLoad seed in file");

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

/// simple_array_chunk (CHUNK, or CHUNK2) runs on ArrayChunk, simple_array_vector
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
        f(arrays...);
}

int main()
{
    if constexpr (use_chunk_array)
    {
        __mem_page_size          = getpagesize();
        constexpr int chunk_size = static_cast<int>(shrink_array::__chunk_size);

        assert(chunk_size <= ChunkDef::size_max);


        if constexpr (chunk_size == 0)
        {
            std::println(stderr, "The chunk size can't be null");
        }

        if (const bool is_pow2 = ParamChunk::check_pow2(chunk_size); !is_pow2)
        {
            std::print(stderr, "The chunk size must be power of two");
        }

        if constexpr (chunk_size < ParamChunk::size_min || chunk_size > ParamChunk::size_max)
        {
            std::println(stderr, "The chunk size must must be superior or equal to {} and inferior or equal to {}",
                         ParamChunk::size_min, ParamChunk::size_max);
            std::abort();
        }

        ParamChunk::init_indexes(true);
    }

    constexpr int sz = 1000000000;
    //  constexpr int sz = 100000;
    // constexpr int iter_max = 10000;
    constexpr int iter_max = 10000;

    constexpr int freq_check_sum = iter_max / 5;

    constexpr int freq_compress = iter_max / 10; // chunk arrays only
    // constexpr int freq_raz = iter_max / 500;
    constexpr int freq_raz = iter_max / 50;
    constexpr int interval = 20000;

    const int max_thr = CHUNK_PMP_MAX_THREADS;

    std::println("Number of threads : {}", max_thr);

    auto start = std::chrono::steady_clock::now();

    std::println("Initialisation ...");
    std::println("Creation of arrays ...");

    // constexpr int interval = 2000;
    Array tab1(sz), tab2(sz), tab3(sz);

    auto                          end             = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    start                                         = end;
    std::println("\nElapsed time creation : {:g}s", elapsed_seconds.count());

#pragma omp          parallel for default(shared)
    for (int i = 0; i < sz; ++i)
    {
        tab1[i] = tab2[i] = tab3[i] = 0;
    }

    end             = std::chrono::steady_clock::now();
    elapsed_seconds = end - start;
    start           = end;
    std::println("\nElapsed time Initialisation to zero: {:g}s", elapsed_seconds.count());

    int min = 1;
    int max = sz;

    // Initialize the random number generator
    // std::random_device rd;
    //  std::mt19937       rng(rd());
    const std::string str            = std::to_string(iter_max);
    const std::string seed_file_name = "seed" + str + ".txt"; // File to store the seed
    std::mt19937      rng;                                    // Random number generator
    initializeRNG(rng, seed_file_name);

    // Create a uniform distribution for interval sizes
    std::uniform_int_distribution sizeDistribution(1, interval);

    // Create a uniform distribution for the starting points of the intervals
    std::uniform_int_distribution startDistribution(min, max);

    std::println("Iteration ...");

    std::array<int, iter_max> tab_size, tab_start, tab_nba1, tab_nba2;

    // #pragma omp parallel for
    for (int k = 0; k < iter_max; ++k)
    {
        const int size   = sizeDistribution(rng);
        const int start0 = startDistribution(rng);
        int       end0   = start0 + size;
        tab_size[k]      = size;
        tab_start[k]     = start0;

        std::uniform_int_distribution<int> uni1(start0, end0);
        tab_nba1[k] = uni1(rng);
        tab_nba2[k] = uni1(rng);
    }

    end             = std::chrono::steady_clock::now();
    elapsed_seconds = end - start;
    start           = end;
    std::println("\nElapsed time distribution : {:g}s", elapsed_seconds.count());

    double v = 0;

    // #pragma omp parallel reduction(+ : v)
    for (int k = 0; k < iter_max; ++k)
    {
        //  std::cout << "iter : " << k << "\n";
        const int size   = tab_size[k];
        int       start0 = tab_start[k];

        int end0 = start0 + size;

        if (end0 > sz)
            end0 = sz;

        if (k != 0 && k % freq_raz == 0)
        {
            // #pragma omp single
            //   {
            std::println("Interval : [{}, {}] size : {}", start0, end0, size);

            int i;

            for (i = start0; i < sz; ++i)
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
            //  }
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
                        tab2[i] = std::cos(double(2 * i));
                    if (i > tab_nba2[k])
                        tab3[i] = std::sin(tab1[i] + tab2[i]);
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

        std::println("k : {} v = {:g}", k, v);

        if (k % freq_check_sum == 0)
        {
            std::println("\ntab1 check_sum = {}", calculate_checksum(tab1));
            std::println("tab2 check_sum = {}", calculate_checksum(tab2));
            std::println("tab3 check_sum = {}", calculate_checksum(tab3));
        }

        if (k % freq_compress == 0)
            if_chunk(
                [](auto&... tabs)
                {
                    std::println("Compression");
                    (tabs.compress(), ...);
                },
                tab1, tab2, tab3);
    }

    std::println("\nv = {:g}", v);

    end             = std::chrono::steady_clock::now();
    elapsed_seconds = end - start;
    std::println("\nElapsed time computation : {:g}s", elapsed_seconds.count());

    std::println("{:g}", static_cast<double>(tab2[2]));

    if_chunk(
        [](const auto&... tabs)
        {
            for (const double sz0 : {tabs.stats()...})
                std::println("Chunk filling rate : {:.2f}%", sz0);

            if constexpr (chunk_config::stats)
                ParamChunk::ChunkStats();
        },
        tab1, tab2, tab3);

    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0)
    {
        long rss = usage.ru_maxrss; // Résident Set Size en kilo-octets
        // {:.2f}: what the chunk builds printed (the filling-rate lines above
        // left std::fixed on std::cout); the vector build printed %g.
        std::println("\nMemory consumption : {:.2f} Mo \n", convert_memory(static_cast<double>(rss * 1024)));
    }
    else
    {
        std::println(stderr, "Unable to retrieve information from memory.");
    }

    tab1.clear();
    tab2.clear();
    tab3.clear();

    if constexpr (use_chunk_array)
        destroy_block_chunk();
    return 0;
}
