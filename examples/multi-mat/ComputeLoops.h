/*
 * Copyright 2024 CEA (www.cea.fr)
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
 * @brief The iterations of the simulation: Simu::computeLoops().
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

#include <print>
#include <array>
#include <vector>
#include <cmath>
#include <functional>
#include <type_traits>

#include "config.h"
#include "Containers.h"
#include "Dispatch.h"
#include "MultiMatAlgo.h"
#include "Simu.h"

template <typename LOOPS, typename LOOPSCHUNK>
void Simu::computeLoops(MultiMatAlgo<LOOPS, LOOPSCHUNK>& mm_algo,
                        const std::function<void(bool)>& lbd_stats,
                        double&                          value)
{
    multi_mat::begin_parallel_phase(); // manual executable only, see config.h

    if (verbose)
    {
#pragma omp single nowait
        std::println("\nInitialisation... ");
    }

    mm_algo.loop(0, nbmat,
                 [this](int, const auto& get_mat_vec, const auto& loops)
                 {
                     // Recovery 3 arrays by material could be "chunk" or vector

                     auto& tab0_ind = get_mat_vec(tabs[0]);
                     auto& tab1_ind = get_mat_vec(tabs[1]);
                     auto& tab2_ind = get_mat_vec(tabs[2]);
                     auto& frac_ind = get_mat_vec(frac);

                     loops.omp_for_nowait_simd(0, mesh_size,
                                               [&](int nn) __attribute__((always_inline))
                                               {
                                                   tab0_ind[nn] = 0;
                                                   tab1_ind[nn] = 0;
                                                   tab2_ind[nn] = 0;
                                                   frac_ind[nn] = 0;
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
        for (int mat = 0; mat < nbmat; ++mat)
        {
            auto& ll_ind = ll[mat];

            const int size  = tab_size[it, mat];
            const int start = tab_start[it, mat];

            const int end = start + size;

            ll_ind.resize(size);

            // We fill the indirection vector

            for (int k = start; k < end; k++)
            {
                const int p = k - start;
                // We avoid overflow: it can happen
                if (k >= mesh_size)
                    break;
                ll_ind[p] = k;
            }
        }

        if (verbose && it % freq_stat_fr == 0)
        {
#pragma omp single
            lbd_stats(true);
        }

        mm_algo.loop(0, nbmat,
                     [&](int mat, const auto& get_mat_vec, const auto& loops)
                     {
                         const auto& ll_ind = ll[mat];

                         // Recovery 3 arrays by material could be "chunk" or vector

                         auto& tab0_ind = get_mat_vec(tabs[0]);
                         auto& tab1_ind = get_mat_vec(tabs[1]);
                         auto& tab2_ind = get_mat_vec(tabs[2]);
                         auto& frac_ind = get_mat_vec(frac);

                         // The material's interval this iteration: ll_ind[p] == start + p,
                         // so nn - start is the position of a cell in the interval, what
                         // the presence fraction depends on.
                         const int start = tab_start[it, mat];
                         const int size  = tab_size[it, mat];

                         if (it % freq_rtz == 0)
                         {
                             if (verbose && mat == 0)
                             {
#pragma omp single nowait
                                 std::println("Reset to zero some elements ");
                             }

                             // Vectorize loop when tab*_ind is vector
                             // Scalar loop  when tab*_ind is "chunk"

                             loops.omp_simd_foreach_nowait(ll_ind,
                                                           [&](int nn)
                                                           {
                                                               // Initialization of arrays
                                                               tab0_ind[nn] = 0;
                                                               tab1_ind[nn] = 0;
                                                               tab2_ind[nn] = 0;
                                                               frac_ind[nn] = 0;
                                                           });
                         }
                         else
                         {
                             // Vectorize loop when tab*_ind is vector
                             // Scalar loop  when tab*_ind is "chunk"

                             loops.omp_simd_foreach_nowait(
                                 ll_ind,
                                 [&](int nn)
                                 {
                                     // Set the arrays to some values

                                     const double v = static_cast<double>(nn) + static_cast<double>(it);
                                     tab0_ind[nn]   = v;
                                     tab1_ind[nn]   = static_cast<double>(2 * nn) + static_cast<double>(it);
                                     tab2_ind[nn]   = static_cast<double>(3 * nn) + static_cast<double>(it);
                                     // 1 inside the material, a ramp at its two interfaces:
                                     // on a chunk array the interior writes hit chunks already
                                     // compressed to 1 and return without materialising them.
                                     frac_ind[nn] = presenceFraction(nn - start, size);
                                 });
                         }
                     });

#pragma omp barrier
        multi_mat::barrier_phase(); // the reads below see the writes above (manual executable only)

        // This first lambda allows you to have a unique coding for a double implementation :
        // the first with classic direct addressing, the second with compressed addressing

        mm_algo.loop(0, nbmat,
                     [&](int mat, const auto& get_mat_vec, const auto& loops)
                     {
                         const auto& ll_ind = ll[mat];

                         const auto& tab0_ind = get_mat_vec(tabs[0]);
                         const auto& tab1_ind = get_mat_vec(tabs[1]);
                         const auto& tab2_ind = get_mat_vec(tabs[2]);
                         const auto& frac_ind = get_mat_vec(frac);

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
                                 ll_ind, value,
                                 [&](int nn, double& vl)
                                 {
                                     // Gadget computation, weighted by the presence fraction
                                     // (a read through the Chunk2 array on a chunk material)
                                     const double vv = frac_ind[nn] * (std::cos(tab0_ind[nn]) + std::sin(tab1_ind[nn]) +
                                                                       std::tan(tab2_ind[nn])) +
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

            mm_algo.loop(0, nbmat,
                         [&](int m, const auto& get_mat_vec, const auto&)
                         {
                             if (dispatch.isChunk(m))
                             {
                                 auto& tab0_ind = get_mat_vec(tabs[0]);
                                 using V        = std::remove_reference_t<decltype(tab0_ind)>;

                                 using vec_chunk_real = VectChunk<double>;

                                 // Compression only exists for 'chunk' arrays: `if constexpr` keeps the call out
                                 // of the std::vector instantiation.

                                 if constexpr (std::is_same_v<V, vec_chunk_real>)
                                 {
                                     auto& tab1_ind = get_mat_vec(tabs[1]);
                                     auto& tab2_ind = get_mat_vec(tabs[2]);

                                     //  If we find "chunk_size" zero contiguous in the arrays then we compress it into
                                     //  a single value
                                     tab0_ind.compress(verbose);
                                     tab1_ind.compress(verbose);
                                     tab2_ind.compress(verbose);
                                 }
                             }
                         });
        }
    }

    multi_mat::end_parallel_phase_in_region(); // manual executable only
}
