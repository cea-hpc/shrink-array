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

#include <cassert>   // assert
#include <chrono>    // std::chrono::steady_clock
#include <cmath>     // std::cos, std::sin, std::sqrt
#include <cstdlib>   // std::abort, std::srand
#include <ctime>     // std::time
#include <iostream>  // std::cout, std::cerr
#include <random>    // std::mt19937, std::uniform_int_distribution, std::default_random_engine
#include <array>     // std::array
#include <fstream>   // std::ifstream, std::ofstream
#include <algorithm> // For std::shuffle
#include <string>    // std::string, std::to_string

#include <sys/resource.h> // getrusage

#include <type_traits> // std::is_same_v (the simd clauses)

#include <vector>  // std::vector
#include <iomanip> // std::setprecision, std::fixed


#include "chunk/param_chunk.h"                // ParamChunk
#include "chunk/array_chunk.h"                // ArrayChunk
#include "chunk/PMP_CHUNK.h"                  // CHUNK_PMP_MAX_THREADS
#include "chunk/chunk_def.h"                  // ChunkDef
#include "chunk/chunk_std.h"                  // destroy_block_chunk
#include "chunk/config.h"                     // chunk_config::stats
#include "chunk/internal/chunk_size_config.h" // __chunk_size


using namespace shrink_array;


double convert_memory(double memoryValue)
{
    const double BYTE_TO_KB = 1.0 / 1024;
    const double KB_TO_MB   = 1.0 / 1024;

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

constexpr int iter_max = 10000;

template <typename V>
void init(int                              start0,
          int                              end0,
          int                              k,
          const std::array<int, iter_max>& tab_nba1,
          const std::array<int, iter_max>& tab_nba2,
          V&                               tab1,
          V&                               tab2,
          V&                               tab3)
{
#pragma omp for schedule(static) nowait
    for (int i = start0; i < end0; ++i)
    {
        tab1[i] = std::sqrt(double(i));
        if (i > tab_nba1[k])
        {
            tab2[i] = std::cos(double(2 * i));
        }
        if (i > tab_nba2[k])
        {
            tab3[i] = std::sin(tab1[i] + tab2[i]);
        }
    }
}

template <typename V>
void calculate(double                  val,
               int                     start0,
               int                     end0,
               const std::vector<int>& index,
               std::vector<double>&    tab_vec,
               const V&                tab1,
               const V&                tab2,
               const V&                tab3)
{
    // simd for a std::vector only: the chunk proxies do not vectorise, and asking
    // for it anyway draws clang's -Wpass-failed (multi-mat shows a way around).
#pragma omp for simd if (simd : std::is_same_v<V, std::vector<double>>) nowait
    for (int i = start0; i < end0; ++i)
    {
        const int k = index[i - start0];
        tab_vec[k] += tab1[k] + (tab2[k] * tab3[i]) - tab2[i] + val;
        //  tab_vec[k] += tab1[k] + tab2[k] * std::atan(tab3[k]) + std::cos(val);
        //     tab_vec[k] += std::cos(tab1[k]) + std::sin(tab2[k]) + std::atan(tab3[k]) + std::cos(val);
    }
}

template <typename V>
void calculate1(double                     val,
                int                        start0,
                int                        end0,
                const std::vector<int>&    index,
                const std::vector<double>& tab_vec,
                V&                         tab1,
                const V&                   tab2,
                const V&                   tab3)
{
    // simd for a std::vector only: the chunk proxies do not vectorise, and asking
    // for it anyway draws clang's -Wpass-failed (multi-mat shows a way around).
#pragma omp for simd if (simd : std::is_same_v<V, std::vector<double>>) nowait
    for (int i = start0; i < end0; ++i)
    {
        const int k = index[i - start0];
        //  assert(k >= start0);
        //  assert(k <= end0);

        tab1[k] += tab_vec[k] + (tab2[k] * tab3[i]) - tab2[i] + val;
    }
}

int main()
{
    const int chunk_size = static_cast<int>(shrink_array::__chunk_size);

    assert(chunk_size <= ChunkDef::size_max);


    if constexpr (chunk_size == 0)
    {
        std::cerr << "The chunk size can't be null\n";
    }

    if (!ParamChunk::check_chunk_size(chunk_size))
    {
        std::cerr << "The chunk size must be a power of two or a multiple of 64";
    }

    if (chunk_size < ParamChunk::size_min || chunk_size > ParamChunk::size_max)
    {
        std::cerr << "The chunk size must must be superior or equal to " << ParamChunk::size_min
                  << " and inferior or equal to " << ParamChunk::size_max << "\n";
        std::abort();
    }

    ParamChunk::init_indexes(true);

    constexpr int sz = 100000000;

    constexpr int freq_compress = iter_max / 10;

    constexpr int freq_raz = iter_max / 50;
    constexpr int interval = 20000;

    const int max_thr = CHUNK_PMP_MAX_THREADS;

    std::cout << "Number of threads : " << max_thr << '\n';

    auto start = std::chrono::steady_clock::now();

    std::cout << "Initialisation ..." << '\n';
    std::cout << "Creation of arrays ..." << '\n';

    ArrayChunk<double> tab_chunk1(sz);
    ArrayChunk<double> tab_chunk2(sz);
    ArrayChunk<double> tab_chunk3(sz);

    std::vector<double> tab1(sz);
    std::vector<double> tab2(sz);
    std::vector<double> tab3(sz);
    std::vector<double> tab_vec(sz);

    std::vector<int> index;

    auto                          end             = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    start                                         = end;
    std::cout << "\nElapsed time creation : " << elapsed_seconds.count() << "s\n";

#pragma omp parallel for default(shared)
    for (int i = 0; i < sz; ++i)
    {
        tab1[i] = tab2[i] = tab3[i] = 0;
        tab_vec[i]                  = 0;
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

    const int progress_interval = iter_max / 10;

    for (int k = 0; k < iter_max; ++k)
    {
        const int size   = tab_size[k];
        const int start0 = tab_start[k];

        int end0 = start0 + size;

        if (end0 > sz)
        {
            end0 = sz;
        }

        const int nsz = end0 - start0;

        // Initialize the random number generator with the current time
        std::srand(static_cast<unsigned int>(std::time(nullptr)));

        // The indirection: the cells of the interval in order, then shuffled
        // by windows below.
        index.resize(nsz);

        for (int p = 0; p < nsz; ++p)
        {
            index[p] = p + start0;
        }

        for (int p = 0; p < 10; ++p)
        {
            const int sz = (std::rand() % (1 + (nsz / 100))) + 1;
            // A window [start, start + sz) strictly inside [1, nsz). Drawn
            // directly: the former rejection loop (draw start in [1, nsz],
            // retry while start + sz >= nsz) never terminates once nsz is
            // small -- the interval sizes go down to a few elements, and
            // one such iteration hung the run for good.
            if (nsz - sz - 1 < 1)
            {
                break;
            }
            const int start = 1 + (std::rand() % (nsz - sz - 1));

            // Shuffle the vector to ensure the array is not ordered
            std::shuffle(index.begin() + start, index.begin() + start + sz,
                         std::default_random_engine(std::time(nullptr)));
        }

        if (k % freq_raz == 0)
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

            // Bounded: the run of non-zeros can reach the end of the arrays,
            // and the unbounded form read tab1[sz] (the libstdc++ assertion
            // of the hardened Debug build caught it, intermittently -- the
            // layout depends on the shuffle's time seed).
            while (i < sz && tab1[i] != 0)
            {
                tab1[i] = tab2[i] = tab3[i] = 0;
                tab_chunk1[i] = tab_chunk2[i] = tab_chunk3[i] = 0;
                i++;
            }
        }
        else
        {
#pragma omp parallel
            {
                // To simulate behavior of a code where we read more than we write

                init(start0, end0, k, tab_nba1, tab_nba2, tab1, tab2, tab3);
                init(start0, end0, k, tab_nba1, tab_nba2, tab_chunk1, tab_chunk2, tab_chunk3);

                for (int p = 0; p < 5; ++p)
                {
                    const auto val1 = static_cast<double>(p);
                    calculate(val1, start0, end0, index, tab_vec, tab1, tab2, tab3);
                    const auto val2 = static_cast<double>(2 * p);
                    calculate(val2, start0, end0, index, tab_vec, tab_chunk1, tab_chunk2, tab_chunk3);

                    calculate1(val1, start0, end0, index, tab_vec, tab1, tab2, tab3);
                    calculate1(val2, start0, end0, index, tab_vec, tab_chunk1, tab_chunk2, tab_chunk3);
                }
            }
        }
        const double progress = (static_cast<double>(k) / iter_max) * 100;

        // Print the progress every 10%
        if (k % progress_interval == 0)
        {
            std::cout << "Progression : " << std::fixed << std::setprecision(2) << progress << "%" << '\n';
        }

        if (k % freq_compress == 0)
        {
            std::cout << "Compression\n";

            tab_chunk1.compress();
            tab_chunk2.compress();
            tab_chunk3.compress();
        }
    }

    end             = std::chrono::steady_clock::now();
    elapsed_seconds = end - start;
    std::cout << "\nElapsed time computation : " << elapsed_seconds.count() << "s\n";

    std::cout << tab2[2] << '\n';

    const auto lbd = [](const ArrayChunk<double>& tab)
    {
        const double sz0 = tab.stats();
        std::cout << "Chunk filling rate : " << std::fixed << std::setprecision(2) << sz0 << "%" << '\n';
    };

    lbd(tab_chunk1);
    lbd(tab_chunk2);
    lbd(tab_chunk3);

    if constexpr (chunk_config::stats)
    {
        ParamChunk::ChunkStats();
    }

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

    destroy_block_chunk();

    return 0;
}