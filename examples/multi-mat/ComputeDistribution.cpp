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
 * @file ComputeDistribution.cpp
 * @brief Simu::computeDistribution(): the interval of every material at
 *        every iteration, drawn up front.
 */

#include <cassert> // assert
#include <random>  // std::mt19937, std::uniform_int_distribution
#include <print>   // std::println

#include "Containers.h" // dispatch
#include "Simu.h"       // Simu::computeDistribution

void Simu::computeDistribution()
{
    const int min = 1;
    const int max = mesh_size;

    std::random_device rd_chunk;
    std::mt19937       rng_chunk(rd_chunk());
    // Create a uniform distribution for interval sizes
    std::uniform_int_distribution size_distribution_chunk(1, interval_chunk);
    // Create a uniform distribution for the starting points of the intervals
    std::uniform_int_distribution start_distribution_chunk(min, max - interval_chunk);

    std::random_device            rd_vector;
    std::mt19937                  rng_vector(rd_vector());
    std::uniform_int_distribution size_distribution_vector(1, interval_vector);
    std::uniform_int_distribution start_distribution_vector(min, max - interval_vector);

    ll.resize(nbmat);

    tab_size.resize(iter_max, nbmat);
    tab_start.resize(iter_max, nbmat);

    for (int it = 0; it < iter_max; ++it)
    {
        //  we store the distribution in tables this allows the parallelization of the next loops

        for (int mat = 0; mat < nbmat; ++mat)
        {
            int size  = 0;
            int begin = 0;
            if (dispatch.isChunk(mat))
            {
                size  = size_distribution_chunk(rng_chunk);
                begin = start_distribution_chunk(rng_chunk);
                assert(size <= interval_chunk);
            }
            else
            {
                size  = size_distribution_vector(rng_vector);
                begin = start_distribution_vector(rng_vector);
                assert(size <= interval_vector);
            }

            tab_size[it, mat]  = size;
            tab_start[it, mat] = begin;

            if (isVerbose())
            {
                const int end = begin + size;

                std::println("Iteration : {} : distribution material #{} start : {} end : {} size : {}", it, mat, begin,
                             end, size);
            }
        }
    }
}