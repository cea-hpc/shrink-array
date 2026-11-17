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

#pragma once

/**
 * @file ComputeLoops.h
 * @brief The iterations of the simulation: Simu::compute_loops().
 *
 * Runs on every thread of one parallel region, for the whole run. Each
 * iteration: the indirection of every material is rebuilt from the interval
 * drawn for it (`omp for` over materials), the fields and the presence
 * fraction are written on that interval (or zeroed, every freq_rtz
 * iterations), a barrier, then the gadget computation reads them back five
 * times and reduces. In the manual executable the fields are also
 * compressed every freq_compress iterations; the embedded one leaves that
 * to the library (config.h).
 */

#include <array>       // Simu::arrays (std::array)
#include <vector>      // Simu::ll, array_size, array_start (std::vector)
#include <cmath>       // std::cos, std::sin, std::tan
#include <functional>  // std::function
#include <type_traits> // std::is_same_v, std::remove_reference_t
#include <print>       // std::println

#include "config.h"       // multi_mat::begin_parallel_phase, barrier_phase, embedded_compression
#include "Containers.h"   // MatZeroArray, dispatch
#include "Dispatch.h"     // Dispatch::is_chunk
#include "MultiMatAlgo.h" // MultiMatAlgo
#include "Simu.h"         // Simu: compute_loops() is defined here

template <typename LoopsVec, typename LoopsChunk>
void Simu::compute_loops(MultiMatAlgo<LoopsVec, LoopsChunk>& mm_algo,
                         const std::function<void(bool)>&    lbd_stats,
                         double&                             value)
{
    multi_mat::begin_parallel_phase(); // manual executable only, see config.h

    if (verbose)
    {
#pragma omp single nowait
        std::println("\nInitialisation... ");
    }

    mm_algo.loop(0, n_materials,
                 [this](int, const auto& get_mat_vec, const auto& loops)
                 {
                     // Recovery 3 arrays by material could be "chunk" or vector

                     auto& arr0_idx = get_mat_vec(tabs[0]);
                     auto& arr1_idx = get_mat_vec(tabs[1]);
                     auto& arr2_idx = get_mat_vec(tabs[2]);
                     auto& frac_idx = get_mat_vec(frac);

                     loops.omp_for_nowait_simd(0, mesh_size,
                                               [&](int nn) __attribute__((always_inline))
                                               {
                                                   arr0_idx[nn] = 0;
                                                   arr1_idx[nn] = 0;
                                                   arr2_idx[nn] = 0;
                                                   frac_idx[nn] = 0;
                                               });
                 });

#pragma omp barrier

    const int freq_stat_fr = iter_max / 10;

    for (int it = 0; it < iter_max; ++it)
    {
        if (verbose)
        {
#pragma omp single nowait
            std::println("Iteration : {}", it);
        }
#pragma omp for // Parallelization on this loop : it is not easy to parallelize the inside loop
        for (int mat = 0; mat < n_materials; ++mat)
        {
            auto& ll_idx = ll[mat];

            const int size  = array_size[it, mat];
            const int start = array_start[it, mat];

            const int end = start + size;

            ll_idx.resize(size);

            // We fill the indirection vector

            for (int k = start; k < end; k++)
            {
                const int pos = k - start;
                // We avoid overflow: it can happen
                if (k >= mesh_size)
                {
                    break;
                }
                ll_idx[pos] = k;
            }
        }

        if (verbose && it % freq_stat_fr == 0)
        {
#pragma omp single
            lbd_stats(true);
        }

        mm_algo.loop(0, n_materials,
                     [&](int mat, const auto& get_mat_vec, const auto& loops)
                     {
                         const auto& ll_idx = ll[mat];

                         // Recovery 3 arrays by material could be "chunk" or vector

                         auto& arr0_idx = get_mat_vec(tabs[0]);
                         auto& arr1_idx = get_mat_vec(tabs[1]);
                         auto& arr2_idx = get_mat_vec(tabs[2]);
                         auto& frac_idx = get_mat_vec(frac);

                         // The material's interval this iteration: ll_idx[p] == start + p,
                         // so nn - start is the position of a cell in the interval, what
                         // the presence fraction depends on.
                         const int start = array_start[it, mat];
                         const int size  = array_size[it, mat];

                         if (it % freq_rtz == 0)
                         {
                             if (verbose && mat == 0)
                             {
#pragma omp single nowait
                                 std::println("Reset to zero some elements ");
                             }

                             // Vectorize loop when array*_idx is vector
                             // Scalar loop  when array*_idx is "chunk"

                             loops.omp_simd_foreach_nowait(ll_idx,
                                                           [&](int nn)
                                                           {
                                                               // Initialization of arrays
                                                               arr0_idx[nn] = 0;
                                                               arr1_idx[nn] = 0;
                                                               arr2_idx[nn] = 0;
                                                               frac_idx[nn] = 0;
                                                           });
                         }
                         else
                         {
                             // Vectorize loop when array*_idx is vector
                             // Scalar loop  when array*_idx is "chunk"

                             loops.omp_simd_foreach_nowait(
                                 ll_idx,
                                 [&](int nn)
                                 {
                                     // Set the arrays to some values

                                     const double value = static_cast<double>(nn) + static_cast<double>(it);
                                     arr0_idx[nn]       = value;
                                     arr1_idx[nn]       = static_cast<double>(2 * nn) + static_cast<double>(it);
                                     arr2_idx[nn]       = static_cast<double>(3 * nn) + static_cast<double>(it);
                                     // 1 inside the material, a ramp at its two interfaces:
                                     // on a chunk array the interior writes hit chunks already
                                     // compressed to 1 and return without materialising them.
                                     frac_idx[nn] = presence_fraction(nn - start, size);
                                 });
                         }
                     });

#pragma omp barrier
        multi_mat::barrier_phase(); // the reads below see the writes above (manual executable only)

        // This first lambda allows you to have a unique coding for a double implementation :
        // the first with classic direct addressing, the second with compressed addressing

        mm_algo.loop(0, n_materials,
                     [&](int mat, const auto& get_mat_vec, const auto& loops)
                     {
                         const auto& ll_idx = ll[mat];

                         const auto& arr0_idx = get_mat_vec(tabs[0]);
                         const auto& arr1_idx = get_mat_vec(tabs[1]);
                         const auto& arr2_idx = get_mat_vec(tabs[2]);
                         const auto& frac_idx = get_mat_vec(frac);

                         // This second lambda allows you to vectorize (1)(2) the loop on
                         // the arrays in direct addressing and no to do it when the arrays
                         // are of compressed types (1) when the compiler gets there .. this was the
                         // case with the classic Intel compiler (2) With IntelOneAPI / aocc / clang++
                         // this doesn't seem to vectorize

                         for (int p = 0; p < 5; ++p) // so that the calculation loop lasts longer -->
                         // We read arrays more often than we write them
                         // this is what happens in real codes
                         // --> Better OpenMP speed up
                         {
                             const auto val = static_cast<double>(p);

                             loops.omp_foreach_simd_red_sum(
                                 ll_idx, value,
                                 [&](int nn, double& vl)
                                 {
                                     // Gadget computation, weighted by the presence fraction
                                     // (a read through the zero_one_value array on a chunk material)
                                     const double vv = frac_idx[nn] * (std::cos(arr0_idx[nn]) + std::sin(arr1_idx[nn]) +
                                                                       std::tan(arr2_idx[nn])) +
                                                       std::cos(val);
                                     vl += vv;
                                 });
                         }
                     });

        // Manual executable only: the field arrays are compressed by an
        // explicit pass. In the embedded one the write path queued the chunks
        // that became constant and the library compresses them itself -- in
        // the loop when a thread's list fills, and at the end of the region
        // through its OMPT tool -- so there is nothing to call
        // (config.h).
        const bool manual_pass = !multi_mat::embedded_compression && option_chunk != 2 && it % freq_compress == 0;
        if (manual_pass)
        {
            if (verbose)
            {
#pragma omp single nowait
                std::println("Compression ");
            }

            mm_algo.loop(0, n_materials,
                         [&](int m, const auto& get_mat_vec, const auto&)
                         {
                             if (dispatch.is_chunk(m))
                             {
                                 auto& arr0_idx = get_mat_vec(tabs[0]);
                                 using V        = std::remove_reference_t<decltype(arr0_idx)>;

                                 using VecChunkReal = MatZeroArray<double>;

                                 // Compression only exists for 'chunk' arrays: `if constexpr` keeps the call out
                                 // of the std::vector instantiation.

                                 if constexpr (std::is_same_v<V, VecChunkReal>)
                                 {
                                     auto& arr1_idx = get_mat_vec(tabs[1]);
                                     auto& arr2_idx = get_mat_vec(tabs[2]);

                                     //  If we find "chunk_size" zero contiguous in the arrays then we compress it into
                                     //  a single value
                                     arr0_idx.compress(verbose);
                                     arr1_idx.compress(verbose);
                                     arr2_idx.compress(verbose);
                                 }
                             }
                         });
        }
    }

    multi_mat::end_parallel_phase_in_region(); // manual executable only
}
