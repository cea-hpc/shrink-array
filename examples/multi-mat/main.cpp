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

/**
 * @file main.cpp
 * @brief multi-mat: a mock multi-material simulation on the chunk library.
 *
 * Every material carries three fields and a presence fraction over the whole
 * mesh, but occupies only a random interval of it at each iteration: the
 * fields are written there, the gadget computation reads them back, the
 * rest of the mesh stays zero. Half the materials (by default) keep their
 * arrays in chunk arrays -- v1 for the fields, Chunk2 for the fraction --
 * where the untouched chunks cost nothing; the other half in std::vectors,
 * for the comparison. The run prints the chunk filling rates, the result of
 * the computation, the elapsed time and the peak memory.
 *
 * Two executables from this file (config.h): `multi-mat`, which compresses
 * its fields by explicit calls and makes the library's phase calls itself,
 * and `multi-mat-embedded`, which calls nothing and lets the OMPT tool of
 * manage_ompt.cpp drive the library.
 */

#include <iomanip>
#include <array>
#include <random>
#include <vector>
#include <functional>
#include <fstream>
#include <tuple>
#include <type_traits>
#include <chrono>

#include <unistd.h>
#include <sys/resource.h>

#include "chunk/param_chunk.h"
#include "chunk/array_chunk.h"
#include "chunk/array_chunk2.h"

#include "config.h"
#include "MultiMatAlgo.h"
#include "ComputeLoops.h"
#include "Containers.h"
#include "Dispatch.h"

using namespace shrink_array;

/// @brief Bytes to MiB, and the figure written to memory_mo.txt for the
///        scripts that collect it.
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
        std::cerr << "Error: unable to open the file.\n";
    }

    return megabytes;
}

/// @brief Required by fast_allocators (one definition per executable).
int __mem_page_size = 0;

using std::vector;

/// @brief The storage of every material, read by the containers.
Dispatch dispatch;

void Simu::createTabs()
{
    for (auto& t : tabs)
    {
        t.create(nbmat, mesh_size);
    }
    frac.create(nbmat, mesh_size);
}

int Simu::getNbmat() const
{
    return nbmat;
}

bool Simu::isVerbose() const
{
    return verbose;
}

int Simu::getOptionChunk() const
{
    return option_chunk;
}

int Simu::getMeshSize() const
{
    return mesh_size;
}

const vector<bool>& Simu::getDis() const
{
    return dis;
}

int main()
{
    const int max_thr = multi_mat::max_threads();

    std::cout << "Number of threads : " << max_thr << "\n";
    std::cout << "Compression       : "
              << (multi_mat::embedded_compression ? "embedded (OMPT tool of manage_ompt.cpp, no call from here)"
                                                  : "manual (explicit compress() on the field arrays)")
              << "\n";
    std::cout << "Phases            : "
              << (multi_mat::embedded_compression ? "OMPT callbacks (manage_ompt.cpp)" : "phase calls from the program")
              << "\n\n";

    __mem_page_size          = getpagesize();
    constexpr int chunk_size = static_cast<int>(shrink_array::__chunk_size);

    if constexpr (chunk_size == 0)
    {
        std::cerr << "The chunk size can't be null\n";
    }

    if (bool is_pow2 = ParamChunk::check_pow2(chunk_size); !is_pow2)
    {
        std::cerr << "The chunk size must be power of two";
    }

    if constexpr (chunk_size < ParamChunk::size_min || chunk_size > ParamChunk::size_max)
    {
        std::cerr << "The chunk size must must be superior or equal to " << ParamChunk::size_min
                  << " and inferior or equal to " << ParamChunk::size_max << "\n";
        std::abort();
    }

    ParamChunk::init_indexes(true);

    ParamChunk::print_info_chunk();
    Simu simu;

    simu.inputData();

    const int nbmat = simu.getNbmat();

    MultiMatAlgo mm_algo;

    const int mesh_size = simu.getMeshSize();

    simu.computeInterval();

    auto start = std::chrono::steady_clock::now();

    std::cout << "\nCreate array :" << std::endl;

    dispatch.create(nbmat);
    dispatch.set(simu.getDis());
    simu.createTabs();

    auto                          end             = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "\nElapsed time create array : " << elapsed_seconds.count() << "s\n\n";
    std::cout << "Compute distribution for all cycles (sequential)\n\n";

    simu.computeDistribution();

    double value_compute = 0;

    start = std::chrono::steady_clock::now();

    // At least one chunk material was seen: the library's counters are
    // worth printing at exit (stats build).
    bool have_chunk = false;

    // The filling rates: of the first chunk material only (only_first_chunk,
    // from inside the run) or of every material with its real rate (at the
    // end).
    std::function<void(bool)> lbd_stats_chunk = [&](bool only_first_chunk)
    {
        mm_algo.loop(0, nbmat,
                     [&](int& mat, const auto& get_mat_vec, const auto&)
                     {
                         const auto& tab0_ind = get_mat_vec(simu.getTabs(0));

                         std::cout << "Material : " << mat << "\n";

                         if (dispatch.isChunk(mat))
                         {
                             have_chunk = true;

                             using CV = std::remove_reference_t<decltype(tab0_ind)>;
                             using V  = std::remove_const_t<CV>;

                             using vec_chunk_real = VectChunk<double>;

                             if constexpr (std::is_same_v<V, vec_chunk_real>)
                             {
                                 const double sz0 = only_first_chunk ? tab0_ind.stats_seq() : tab0_ind.stats();

                                 std::cout << "  Chunk filling rate : " << std::fixed << std::setprecision(2) << sz0
                                           << "%" << std::endl;

                                 // The presence fraction, a Chunk2 array: only its
                                 // interface chunks should be materialised, the
                                 // interior ones are compressed to 1 and the empty
                                 // ones to 0.
                                 const auto&  frac_ind = get_mat_vec(simu.getFrac());
                                 const double szf      = only_first_chunk ? frac_ind.stats_seq() : frac_ind.stats();

                                 std::cout << "  Fraction chunk filling rate (Chunk2) : " << std::fixed
                                           << std::setprecision(2) << szf << "%" << std::endl;
                             }

                             if (only_first_chunk)
                                 mat = nbmat;
                         }

                         if (!only_first_chunk)
                         {
                             int s = 0;
#pragma omp parallel for reduction(+ : s)
                             for (int k = 0; k < mesh_size; ++k)
                             {
                                 if (tab0_ind[k] != 0)
                                     s++;
                             }
                             const double sz = static_cast<double>(s) / static_cast<double>(mesh_size) * 100.;

                             std::cout << "  Real filling rate : " << std::fixed << std::setprecision(2) << sz << "%"
                                       << std::endl;
                         }
                     });
    };

    std::cout << "\nCompute loops for all cycles (parallel)\n";

#pragma omp parallel reduction(+ : value_compute)
    {
        simu.computeLoops(mm_algo, lbd_stats_chunk, value_compute);
    }
    multi_mat::end_parallel_phase_after_region(); // manual executable only

    if (simu.isVerbose())
        std::cout << "\nEnd iteration\n\n";

    lbd_stats_chunk(false);

    end             = std::chrono::steady_clock::now();
    elapsed_seconds = end - start;

    std::cout << "\nResult = " << value_compute << std::endl;

    std::cout << "\n\nElapsed time  : " << elapsed_seconds.count() << "s\n\n";

    rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0)
    {
        long rss = usage.ru_maxrss; // Resident Set Size in kilobytes
        std::cout << "Memory consumption : " << convert_memory(rss * 1024) << " Mo\n\n";
    }
    else
    {
        std::cerr << "Unable to retrieve information from memory." << std::endl;
    }

    if constexpr (multi_mat::chunk_stats)
    {
        if (have_chunk)
        {
            ParamChunk::ChunkStats();
        }
    }
}
