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

#pragma once

/**
 * @file ComputeLoopsMd.h
 * @brief The iterations of multi-mat-md (SimuMd::computeLoops()) and their
 *        serial replay (SimuMd::verify()).
 *
 * The iteration of multi-mat, in 2D: the fields and the presence fraction
 * of every material are written on its box (or zeroed, every freq_rtz
 * iterations), a barrier, the gadget computation reads them back five
 * times, a barrier; in the manual executable the fields are compressed
 * every freq_compress iterations. No indirection list: the box is the
 * loop. The body is written once against the views and the four
 * operations of the policy (LoopsMd.h), which MultiMatAlgo picks per
 * material.
 */

#include "SimuMd.h" // SimuMd, SHRINK_ARRAY_HAS_MDSPAN

#if SHRINK_ARRAY_HAS_MDSPAN

    #include <algorithm>   // std::fill, std::max
    #include <chrono>      // std::chrono::steady_clock (the phase times)
    #include <cmath>       // std::abs
    #include <type_traits> // std::is_same_v, std::remove_cvref_t
    #include <utility>     // std::as_const
    #include <vector>      // std::vector (the replay)
    #include <print>       // std::println

    #include "config.h"     // multi_mat::begin_parallel_phase, barrier_phase, embedded_compression
    #include "Containers.h" // VectChunk, dispatch

namespace multi_mat::md
{

template <typename BoxOf, typename RowParts>
void SimuMd::dealParts(const BoxOf& box_of, const RowParts& row_parts, std::vector<Share>& mine) const
{
    const long nth = multi_mat::team_size();
    const long me  = multi_mat::thread_rank();

    long total = 0;
    for (int mat = 0; mat < nbmat; ++mat)
    {
        const Box b = box_of(mat);
        total += static_cast<long>(b.h) * b.w;
    }
    const long lo = total * me / nth;       // this thread's cells: [lo, hi)
    const long hi = total * (me + 1) / nth; // in the order (material, row, column)

    mine.clear();
    long acc = 0; // cells before the current unit
    for (int mat = 0; mat < nbmat && acc < hi; ++mat)
    {
        const Box  b    = box_of(mat);
        const auto cols = tile_cols_of(b);
        if (row_parts(mat))
        {
            // Units of one row of the box: this thread's run of rows.
            Part run{-1, -1, cols[0], cols[1]};
            for (int i = b.i0; i < b.i1() && acc < hi; ++i)
            {
                if (acc >= lo) // the row starts in this thread's share
                {
                    run.row_first = run.row_first < 0 ? i : run.row_first;
                    run.row_last  = i + 1;
                }
                acc += b.w;
            }
            if (run.row_first >= 0)
            {
                mine.push_back({mat, run});
            }
            continue;
        }
        // Units of one tile: this thread's run of tiles in each band.
        const auto bands = bands_of(b);
        for (int r = bands[0]; r < bands[1] && acc < hi; ++r)
        {
            Part run{r * band_rows, (r + 1) * band_rows, -1, -1};
            for (int c = cols[0]; c < cols[1] && acc < hi; ++c)
            {
                const Box cell = cells_of(b, Part{run.row_first, run.row_last, c, c + 1});
                if (acc >= lo) // the tile starts in this thread's share
                {
                    run.c_first = run.c_first < 0 ? c : run.c_first;
                    run.c_last  = c + 1;
                }
                acc += static_cast<long>(cell.h) * cell.w;
            }
            if (run.c_first >= 0)
            {
                mine.push_back({mat, run});
            }
        }
    }
}

template <typename LOOPS, typename LOOPSCHUNK>
void SimuMd::computeLoops(const MultiMatAlgo<LOOPS, LOOPSCHUNK>& mm_algo, double& value)
{
    multi_mat::begin_parallel_phase(); // manual executable only, see config.h

    const Box  mesh{0, 0, nx, ny};
    const auto t_start = std::chrono::steady_clock::now();

    // This thread's parts, rebuilt for every phase whose boxes change: the
    // team splits the cells of all the materials at once, tile by tile, not
    // material by material (a box of a few rows of tiles would leave most of
    // the team idle).
    std::vector<Share> mine;
    // The unit of a material's parts: its policy's (LoopsMd.h).
    const auto row_parts = [](int mat) { return dispatch.isChunk(mat) ? LOOPSCHUNK::row_parts : LOOPS::row_parts; };
    mine.reserve(static_cast<std::size_t>(nbmat));
    const int me = multi_mat::thread_rank();

    // Every array zero over the whole mesh, as multi-mat's initialisation.
    dealParts([&mesh](int) { return mesh; }, row_parts, mine);
    for (const Share& sh : mine)
    {
        mm_algo.loop(sh.mat, sh.mat + 1,
                     [&](int, const auto& get_mat_vec, const auto& loops)
                     {
                         loops.zero_box(mesh, sh.part, loops.view(get_mat_vec(tabs[0]), nx, ny),
                                        loops.view(get_mat_vec(tabs[1]), nx, ny),
                                        loops.view(get_mat_vec(tabs[2]), nx, ny),
                                        loops.view(get_mat_vec(frac), nx, ny));
                     });
    }

    #pragma omp barrier
    multi_mat::barrier_phase();
    if (multi_mat::thread_rank() == 0)
    {
        time_init = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
    }

    const auto bx       = boxes();
    const int  freq_msg = std::max(iter_max / 10, 1);

    for (int it = 0; it < iter_max; ++it)
    {
        if (verbose || it % freq_msg == 0)
        {
    #pragma omp single nowait
            std::println("Iteration : {}{}", it, it % freq_rtz == 0 ? " (boxes reset to zero)" : "");
        }

        // The phase times, measured by thread 0 (barriers included).
        const bool timer = multi_mat::thread_rank() == 0;
        auto       t_ph  = std::chrono::steady_clock::now();
        const auto lap   = [&](double& acc)
        {
            if (timer)
            {
                const auto now = std::chrono::steady_clock::now();
                acc += std::chrono::duration<double>(now - t_ph).count();
                t_ph = now;
            }
        };

        // This thread's parts of this iteration's boxes: the writes and the
        // reads below take the same.
        dealParts([&bx, it](int mat) { return bx[it, mat]; }, row_parts, mine);
        for (const Share& sh : mine)
        {
            const Box c = cells_of(bx[it, sh.mat], sh.part);
            work[static_cast<std::size_t>(me)].cells += static_cast<long>(c.h) * c.w;
        }

        // Writes: the box of every material.
        for (const Share& sh : mine)
        {
            mm_algo.loop(sh.mat, sh.mat + 1,
                         [&](int mat, const auto& get_mat_vec, const auto& loops)
                         {
                             const Box b = bx[it, mat];

                             const auto f0 = loops.view(get_mat_vec(tabs[0]), nx, ny);
                             const auto f1 = loops.view(get_mat_vec(tabs[1]), nx, ny);
                             const auto f2 = loops.view(get_mat_vec(tabs[2]), nx, ny);
                             const auto fr = loops.view(get_mat_vec(frac), nx, ny);

                             if (it % freq_rtz == 0)
                             {
                                 loops.zero_box(b, sh.part, f0, f1, f2, fr);
                             }
                             else
                             {
                                 const double t = static_cast<double>(it);
                                 loops.write_box(
                                     b, sh.part, write(f0, [this, t](int i, int j) { return cellNumber(i, j) + t; }),
                                     write(f1, [this, t](int i, int j) { return 2 * cellNumber(i, j) + t; }),
                                     write(f2, [this, t](int i, int j) { return 3 * cellNumber(i, j) + t; }),
                                     // 1 inside the box, a ramp along its sides: on a chunk
                                     // material in tiles, a tile of the interior already at 1
                                     // is left alone (the hint), compressed. [&], not [this, &b]:
                                     // clang warns on `this` in a generic lambda the cell
                                     // policies never instantiate.
                                     write(
                                         fr, [this, &b](int i, int j) { return presenceFraction(b, i, j); },
                                         [&](const auto& o, const auto& e) { return fractionOnTile(b, o, e); }));
                             }
                         });
        }

    #pragma omp barrier
        multi_mat::barrier_phase(); // the reads below see the writes above (manual executable only)
        lap(time_write);

        // Reads: the gadget computation, five times, on read-only views.
        for (const Share& sh : mine)
        {
            mm_algo.loop(sh.mat, sh.mat + 1,
                         [&](int mat, const auto& get_mat_vec, const auto& loops)
                         {
                             const Box  b  = bx[it, mat];
                             const auto c0 = loops.view(std::as_const(get_mat_vec(tabs[0])), nx, ny);
                             const auto c1 = loops.view(std::as_const(get_mat_vec(tabs[1])), nx, ny);
                             const auto c2 = loops.view(std::as_const(get_mat_vec(tabs[2])), nx, ny);
                             const auto cf = loops.view(std::as_const(get_mat_vec(frac)), nx, ny);

                             for (int p = 0; p < 5; ++p)
                             {
                                 const double val = static_cast<double>(p);
                                 loops.reduce_box(
                                     b, sh.part, value, [val](double fr, double a, double bb, double c)
                                     { return gadget(fr, a, bb, c, val); }, cf, c0, c1, c2);
                             }
                         });
        }

        // The next writes (or the compression pass) after every read.
    #pragma omp barrier
        lap(time_read);

        // Manual executable only: the field arrays compressed by an explicit
        // pass, as multi-mat does (config.h).
        const bool manual_pass = !multi_mat::embedded_compression && option_chunk != 2 && it % freq_compress == 0;
        if (manual_pass)
        {
            mm_algo.loop(0, nbmat,
                         [&](int m, const auto& get_mat_vec, const auto&)
                         {
                             if (dispatch.isChunk(m))
                             {
                                 auto& t0 = get_mat_vec(tabs[0]);
                                 using V  = std::remove_cvref_t<decltype(t0)>;
                                 if constexpr (std::is_same_v<V, VectChunk<double>>)
                                 {
                                     t0.compress(false);
                                     get_mat_vec(tabs[1]).compress(false);
                                     get_mat_vec(tabs[2]).compress(false);
                                 }
                             }
                         });
    #pragma omp barrier
            lap(time_compress);
        }
    }

    multi_mat::end_parallel_phase_in_region(); // manual executable only
}

template <typename LOOPS, typename LOOPSCHUNK>
long SimuMd::verify(const MultiMatAlgo<LOOPS, LOOPSCHUNK>& mm_algo, double result) const
{
    const std::size_t n  = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny);
    const auto        bx = boxes();

    long   wrong   = 0;
    double ref     = 0;
    double ref_abs = 0;

    std::vector<double> r0(n);
    std::vector<double> r1(n);
    std::vector<double> r2(n);
    std::vector<double> rf(n);

    for (int mat = 0; mat < nbmat; ++mat)
    {
        std::fill(r0.begin(), r0.end(), 0.);
        std::fill(r1.begin(), r1.end(), 0.);
        std::fill(r2.begin(), r2.end(), 0.);
        std::fill(rf.begin(), rf.end(), 0.);

        for (int it = 0; it < iter_max; ++it)
        {
            const Box    b = bx[it, mat];
            const double t = static_cast<double>(it);
            for (int i = b.i0; i < b.i1(); ++i)
            {
                for (int j = b.j0; j < b.j1(); ++j)
                {
                    const std::size_t c = static_cast<std::size_t>(i) * static_cast<std::size_t>(ny) + j;
                    const bool        z = it % freq_rtz == 0;
                    r0[c]               = z ? 0. : cellNumber(i, j) + t;
                    r1[c]               = z ? 0. : 2 * cellNumber(i, j) + t;
                    r2[c]               = z ? 0. : 3 * cellNumber(i, j) + t;
                    rf[c]               = z ? 0. : presenceFraction(b, i, j);
                }
            }
            for (int p = 0; p < 5; ++p)
            {
                for (int i = b.i0; i < b.i1(); ++i)
                {
                    for (int j = b.j0; j < b.j1(); ++j)
                    {
                        const std::size_t c = static_cast<std::size_t>(i) * static_cast<std::size_t>(ny) + j;
                        const double      g = gadget(rf[c], r0[c], r1[c], r2[c], static_cast<double>(p));
                        ref += g;
                        ref_abs += std::abs(g);
                    }
                }
            }
        }

        // The arrays of the run, through the read-only views of their policy.
        mm_algo.loop(
            mat, mat + 1,
            [&](int m, const auto& get_mat_vec, const auto& loops)
            {
                const auto c0 = loops.view(std::as_const(get_mat_vec(tabs[0])), nx, ny);
                const auto c1 = loops.view(std::as_const(get_mat_vec(tabs[1])), nx, ny);
                const auto c2 = loops.view(std::as_const(get_mat_vec(tabs[2])), nx, ny);
                const auto cf = loops.view(std::as_const(get_mat_vec(frac)), nx, ny);
                long       w  = 0;
                for (int i = 0; i < nx; ++i)
                {
                    for (int j = 0; j < ny; ++j)
                    {
                        const std::size_t c  = static_cast<std::size_t>(i) * static_cast<std::size_t>(ny) + j;
                        const double      v0 = c0[i, j];
                        const double      v1 = c1[i, j];
                        const double      v2 = c2[i, j];
                        const double      vf = cf[i, j];
                        const bool ok = v0 == r0[c] && v1 == r1[c] && v2 == r2[c] && std::abs(vf - rf[c]) <= 1e-15;
                        if (!ok)
                        {
                            if (w < 5)
                            {
                                std::println("verify: material {} cell ({}, {}): got {} {} {} {}, expected {} {} {} {}",
                                             m, i, j, v0, v1, v2, vf, r0[c], r1[c], r2[c], rf[c]);
                            }
                            ++w;
                        }
                    }
                }
                if (w != 0)
                {
                    std::println("verify: material {} ({}): {} cells wrong", m,
                                 dispatch.isChunk(m) ? "chunk" : "vector", w);
                }
                wrong += w;
            });
    }

    const double tol = 1e-9 * ref_abs;
    std::println("verify: result {:.10e}, serial replay {:.10e} (|diff| {:.3e}, tolerance {:.3e})", result, ref,
                 std::abs(result - ref), tol);
    if (!(std::abs(result - ref) <= tol))
    {
        ++wrong;
    }
    std::println("verify: {}", wrong == 0 ? "OK" : "FAILED");
    return wrong;
}

} // namespace multi_mat::md

#endif // SHRINK_ARRAY_HAS_MDSPAN
