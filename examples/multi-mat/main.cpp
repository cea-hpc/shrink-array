
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
#include <array>
#include <random>
#include <vector>
#include <functional>
#include <fstream>

#ifdef __clang__
    #include <tuple>
#endif
#include <type_traits>
#include <chrono>

#include <unistd.h>
#include <sys/resource.h>

#include "chunk/param_chunk.h"
#include "chunk/array_chunk.h"


#include "MultiMatAlgo.h"
#include "ComputeLoops.h"
#include "Containers.h"
#include "Dispatch.h"

using namespace shrink_array;

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
        // Write a scalar to the file

        fichier << megabytes;

        // Close the file
        fichier.close();
    }
    else
    {
        std::cerr << "Error: unable to open the file.\n";
    }

    return megabytes;
}

int __mem_page_size = 0;

using std::vector;

Dispatch dispatch;

void Simu::createTabs()
{
    for (auto& t : tabs)
    {
        t.create(nbmat, mesh_size);
    }
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
#ifdef _OPENMP
    const int max_thr = omp_get_max_threads();
#else
    const int max_thr = 1;
#endif

    std::cout << "Number of threads : " << max_thr << "\n\n";

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

#ifdef CHUNK_STATS
    bool have_chunk = false;
#endif

    std::function<void(bool)> lbd_stats_chunk = [&](bool only_first_chunk)
    {
        mm_algo.loop(0, nbmat,
                     [&](int& mat, const auto& get_mat_vec, const auto&)
                     {
                         const auto& tab0_ind = get_mat_vec(simu.getTabs(0));

                         std::cout << "Material : " << mat << "\n";

                         if (dispatch.isChunk(mat))
                         {
#ifdef CHUNK_STATS
                             have_chunk = true;
#endif

                             using CV = std::remove_reference_t<decltype(tab0_ind)>;
                             using V  = std::remove_const_t<CV>;

                             using vec_chunk_real = VectChunk<double>;

                             if constexpr (std::is_same_v<V, vec_chunk_real>)
                             {
                                 const double sz0 = only_first_chunk ? tab0_ind.stats_seq() : tab0_ind.stats();

                                 std::cout << "  Chunk filling rate : " << std::fixed << std::setprecision(2) << sz0
                                           << "%" << std::endl;
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
#ifdef CHUNK_STATS
    if (have_chunk)
    {
        ParamChunk::ChunkStats();
    }
#endif
}
