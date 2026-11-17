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

// The C headers outside the two forms, for their macros and global names: an
// import brings neither.
#include <cstdint>        // uint64_t (unqualified)
#include <cstdio>         // stderr
#include <sys/resource.h> // getrusage, RUSAGE_SELF

// With CHUNK_MODULES=ON (SHRINK_ARRAY_IMPORT) both programs import the module
// over their library: simple_array_chunk over chunk_zero, simple_array_vector
// over chunk_vector (the std::vector mode). Otherwise both have the chunk
// headers, and link those libraries.
#ifdef SHRINK_ARRAY_IMPORT
import std;          // std::bit_cast, std::chrono, std::mt19937, std::ofstream, std::conditional_t, std::println, ...
import shrink_array; // zero::Array, Runtime, max_threads, chunk_config
#else
    #include <algorithm>   // std::min
    #include <bit>         // std::bit_cast (calculate_checksum)
    #include <chrono>      // std::chrono::steady_clock
    #include <cmath>       // std::cos, std::sin, std::sqrt
    #include <cstdlib>     // std::abort
    #include <random>      // std::mt19937, std::uniform_int_distribution
    #include <array>       // std::array
    #include <fstream>     // std::ifstream, std::ofstream
    #include <string>      // std::string, std::to_string
    #include <type_traits> // std::conditional_t (Array)
    #include <vector>      // std::vector
    #include <print>       // std::println, std::print

    #include "chunk/runtime.h"    // Runtime, Runtime::shutdown
    #include "chunk/zero_array.h" // zero::Array
    #include "chunk/threads.h"    // shrink_array::max_threads
    #include "chunk/config.h"     // chunk_config::use_zero*, stats
#endif


/// @brief splitmix64's finaliser: a bijection on 64 bits that spreads every
///        input bit over the whole output.
constexpr uint64_t checksum_mix(uint64_t value) noexcept
{
    value += 0x9E3779B97F4A7C15ULL;
    value = (value ^ (value >> 30U)) * 0xBF58476D1CE4E5B9ULL;
    value = (value ^ (value >> 27U)) * 0x94D049BB133111EBULL;
    return value ^ (value >> 31U);
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

double convert_memory(double memory_value)
{
    constexpr double byte_to_kb = 1.0 / 1024;
    constexpr double kb_to_mb   = 1.0 / 1024;

    const double kilobytes = memory_value * byte_to_kb;
    const double megabytes = kilobytes * kb_to_mb;

    if (std::ofstream out_file("memory_mo.txt"); out_file.is_open())
    {
        out_file << megabytes;
        out_file.close();
    }
    else
    {
        std::println(stderr, "Error: cannot open the file.");
    }

    return megabytes;
}

// Save seed to file
void save_seed_to_file(unsigned seed, const std::string& filename)
{
    std::println("\nSave seed in file");

    if (std::ofstream file(filename); file.is_open())
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
    unsigned seed = 0;
    if (std::ifstream file(filename); file.is_open())
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
void initialize_rng(std::mt19937& rng, const std::string& seed_file_name)
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

/// simple_array_chunk (CHUNK_VARIANT_ZERO, or _ZERO_ONE_VALUE) runs on zero::Array, simple_array_vector
/// on std::vector.
constexpr bool use_chunk_array = chunk_config::use_zero || chunk_config::use_zero_one_value;
using Array                    = std::conditional_t<use_chunk_array, zero::Array<double>, std::vector<double>>;

/// @brief Calls @p func on @p arrays in simple_array_chunk only. @p func is a generic
///        lambda using zero::Array's members (compress, stats): its body is then
///        instantiated with zero::Array alone. The same call in a discarded branch
///        of main(), which is not a template, would be compiled for std::vector.
template <typename F, typename... A>
void if_chunk(F&& func, A&... arrays)
{
    if constexpr (use_chunk_array)
    {
        func(arrays...);
    }
}

int main()
{
    if constexpr (use_chunk_array)
    {
        // Once, before the first array. The library checks its compiled chunk
        // size itself.
        Runtime::init(true);
    }

    // 3 arrays of sz doubles: 2.4 GB for std::vector. 1e9 (24 GB at once) made
    // the ctest fail with bad_alloc on a 30 GB workstation with anything else running.
    constexpr int sz       = 100000000;
    constexpr int iter_max = 10000;

    constexpr int freq_check_sum = iter_max / 5;

    constexpr int freq_compress = iter_max / 10; // chunk arrays only
    constexpr int freq_reset    = iter_max / 50;
    constexpr int interval      = 20000;

    const int max_thr = shrink_array::max_threads();

    std::println("Number of threads : {}", max_thr);

    auto start = std::chrono::steady_clock::now();

    std::println("Initialisation ...");
    std::println("Creation of arrays ...");

    Array arr1(sz);
    Array arr2(sz);
    Array arr3(sz);

    auto                          end             = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    start                                         = end;
    std::println("\nElapsed time creation : {:g}s", elapsed_seconds.count());

#pragma omp          parallel for default(shared)
    for (int i = 0; i < sz; ++i)
    {
        arr1[i] = arr2[i] = arr3[i] = 0;
    }

    end             = std::chrono::steady_clock::now();
    elapsed_seconds = end - start;
    start           = end;
    std::println("\nElapsed time Initialisation to zero: {:g}s", elapsed_seconds.count());

    const int min = 1;
    const int max = sz;

    // Initialize the random number generator
    const std::string str            = std::to_string(iter_max);
    const std::string seed_file_name = "seed" + str + ".txt"; // File to store the seed
    std::mt19937      rng;                                    // Random number generator
    initialize_rng(rng, seed_file_name);

    // Create a uniform distribution for interval sizes
    std::uniform_int_distribution size_distribution(1, interval);

    // Create a uniform distribution for the starting points of the intervals
    std::uniform_int_distribution start_distribution(min, max);

    std::println("Iteration ...");

    std::array<int, iter_max> array_size{};
    std::array<int, iter_max> array_start{};
    std::array<int, iter_max> array_nba1{};
    std::array<int, iter_max> array_nba2{};

    for (int k = 0; k < iter_max; ++k)
    {
        const int size   = size_distribution(rng);
        const int start0 = start_distribution(rng);
        const int end0   = start0 + size;
        array_size[k]    = size;
        array_start[k]   = start0;

        std::uniform_int_distribution uni1(start0, end0);
        array_nba1[k] = uni1(rng);
        array_nba2[k] = uni1(rng);
    }

    end             = std::chrono::steady_clock::now();
    elapsed_seconds = end - start;
    start           = end;
    std::println("\nElapsed time distribution : {:g}s", elapsed_seconds.count());

    double value = 0;

    for (int k = 0; k < iter_max; ++k)
    {
        const int size   = array_size[k];
        const int start0 = array_start[k];

        int end0 = start0 + size;

        end0 = std::min(end0, sz);

        if (k != 0 && k % freq_reset == 0)
        {
            std::println("Interval : [{}, {}] size : {}", start0, end0, size);

            int i = start0;

            for (; i < sz; ++i)
            {
                if (arr1[i] != 0)
                {
                    break;
                }
            }

            // Bounded: i can already be sz here (start0 == sz is drawn by
            // start_distribution(1, sz), or no non-zero was found), and a
            // non-zero run can extend to sz - 1.
            while (i < sz && arr1[i] != 0)
            {
                arr1[i] = arr2[i] = arr3[i] = 0;
                i++;
            }
        }
        else
        {
#pragma omp parallel reduction(+ : value)
            {
                // To simulate comportement of a code where we read more than we write
#pragma omp for schedule(static) nowait
                for (int i = start0; i < end0; ++i)
                {
                    arr1[i] = std::sqrt(static_cast<double>(i));
                    if (i > array_nba1[k])
                    {
                        arr2[i] = std::cos(double(2 * i));
                    }
                    if (i > array_nba2[k])
                    {
                        arr3[i] = std::sin(arr1[i] + arr2[i]);
                    }
                }

                for (int p = 0; p < 5; ++p)
                {
                    const auto val = static_cast<double>(p);

#pragma omp for schedule(static) nowait
                    for (int i = start0; i < end0; ++i)
                    {
                        value += std::cos(arr1[i]) + std::sin(arr2[i]) + std::atan(arr3[i]) + std::cos(val);
                    }
                }
            }
        }

        std::println("k : {} v = {:g}", k, value);

        if (k % freq_check_sum == 0)
        {
            std::println("\ntab1 check_sum = {}", calculate_checksum(arr1));
            std::println("arr2 check_sum = {}", calculate_checksum(arr2));
            std::println("arr3 check_sum = {}", calculate_checksum(arr3));
        }

        if (k % freq_compress == 0)
        {
            if_chunk(
                [](auto&... tabs)
                {
                    std::println("Compression");
                    (tabs.compress(), ...);
                },
                arr1, arr2, arr3);
        }
    }

    std::println("\nv = {:g}", value);

    end             = std::chrono::steady_clock::now();
    elapsed_seconds = end - start;
    std::println("\nElapsed time computation : {:g}s", elapsed_seconds.count());

    std::println("{:g}", static_cast<double>(arr2[2]));

    if_chunk(
        [](const auto&... tabs)
        {
            for (const double sz0 : {tabs.stats()...})
            {
                std::println("Chunk filling rate : {:.2f}%", sz0);
            }

            if constexpr (chunk_config::stats)
            {
                Runtime::chunk_stats();
            }
        },
        arr1, arr2, arr3);

    if (struct rusage usage{}; getrusage(RUSAGE_SELF, &usage) == 0)
    {
        const long rss = usage.ru_maxrss; // Resident Set Size in kilobytes
        // {:.2f}: what the chunk builds printed (the filling-rate lines above
        // left std::fixed on std::cout); the vector build printed %g.
        std::println("\nMemory consumption : {:.2f} Mo \n", convert_memory(static_cast<double>(rss * 1024)));
    }
    else
    {
        std::println(stderr, "Unable to retrieve information from memory.");
    }

    arr1.clear();
    arr2.clear();
    arr3.clear();

    if constexpr (use_chunk_array)
    {
        Runtime::shutdown();
    }
    return 0;
}
