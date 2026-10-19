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
 * @file SimuMd.h
 * @brief The simulation of multi-mat-md: multi-mat on a 2D mesh, every
 *        material in a box per iteration, every array seen through an
 *        mdspan (LoopsMd.h).
 */

#include "LoopsMd.h" // multi_mat::md::Box, the policies, SHRINK_ARRAY_HAS_MDSPAN

#if SHRINK_ARRAY_HAS_MDSPAN

    #include <algorithm> // std::min
    #include <array>     // std::array
    #include <cmath>     // std::cos, std::sin, std::tan
    #include <cstddef>   // std::size_t
    #include <optional>  // std::optional
    #include <vector>    // std::vector

    #include "Containers.h"   // MultiArray, VectChunk2
    #include "MultiMatAlgo.h" // MultiMatAlgo

namespace multi_mat::md
{

/// @brief How the chunk materials are laid out and walked (--chunk-mode).
enum class ChunkMode
{
    tiles,    ///< md::layout_tiled, tile loops (LoopsMdTiles): the default
    elements, ///< md::layout_tiled, cell loops (LoopsMdTilesElements)
    rows,     ///< row-major md::view, cell loops (LoopsMdRows)
};

/**
 * @brief The 2D mock: an nx x ny mesh, every material in a random box at
 *        each iteration, its three fields and its presence fraction written
 *        there and read back by the gadget computation of multi-mat.
 * @details Same steps as Simu (main_md.cpp runs them): parameters (command
 *          line), storage of every material, the boxes of every iteration
 *          drawn up front, the iterations in one parallel region, and
 *          optionally a serial replay that checks the arrays and the result
 *          (verify()).
 */
struct SimuMd
{
    // ── Parameters (command line, main_md.cpp) ───────────────────────────

    int nx       = 3200; ///< Rows of the mesh (3200 x 3200: about the 10 M cells of multi-mat)
    int ny       = 3200; ///< Columns of the mesh
    int nbmat    = 50;   ///< Materials
    int iter_max = 100;  ///< Iterations

    int    option_chunk = 3;  ///< 1 all chunk, 2 all vector, 3 random 50/50, 4 random at chunk_rate
    int    chunk_rate   = 90; ///< Option 4: percentage of chunk materials
    double pc_chunk     = .5; ///< Largest box of a chunk material, % of the mesh's area
    double pc_vector    = 3;  ///< Largest box of a vector material, % of the mesh's area

    int freq_rtz      = 10; ///< Every this many iterations the box is zeroed instead of written
    int freq_compress = 10; ///< Manual executable: every this many iterations the fields are compressed

    int interface_width = 4; ///< Cells of the presence-fraction ramp at each side of a box

    ChunkMode chunk_mode = ChunkMode::tiles;

    unsigned seed          = 0;     ///< Seed of the draws; 0: std::random_device
    bool     verbose       = false; ///< Per-iteration prints
    bool     check         = false; ///< Replay serially and compare (verify())
    double   max_imbalance = 0; ///< Fail when the most loaded thread wrote more than this times the mean (0: no check)

    // ── State ────────────────────────────────────────────────────────────

    std::vector<bool> dis; ///< Storage of every material: true = chunk

    std::array<MultiArray<double>, 3>      tabs; ///< The three fields of every material
    MultiArray<double, VectChunk2<double>> frac; ///< The presence fraction of every material

    double time_init     = 0; ///< Seconds in the initialisation (every array zeroed), barrier included (thread 0)
    double time_write    = 0; ///< Seconds in the writes, barrier included (thread 0)
    double time_read     = 0; ///< Seconds in the reads, barrier included (thread 0)
    double time_compress = 0; ///< Seconds in the manual compression passes (thread 0)

    /// @brief The box of every material at every iteration, [iteration, material]:
    ///        a vector seen through a 2D mdspan.
    std::vector<Box> box_store;

    [[nodiscard]] auto boxes() const
    {
        return mdns::mdspan(box_store.data(), mdns::dextents<int, 2>(iter_max, nbmat));
    }

    // ── Steps (main_md.cpp, ComputeLoopsMd.h) ────────────────────────────

    /// @brief The parameters from the command line; false on a bad argument.
    bool parse(int argc, char** argv);

    /// @brief The storage of every material (option_chunk).
    void drawStorage();

    /// @brief The three fields and the presence fraction of every material,
    ///        sized for the layout of the chunk mode.
    void createTabs();

    /// @brief Cells a thread wrote (resets included), one per thread, padded:
    ///        the balance of the deal (dealParts()).
    struct alignas(64) WorkSlot
    {
        long cells = 0;
    };
    std::vector<WorkSlot> work;

    /// @brief The box of every material at every iteration.
    void computeDistribution();

    /// @brief Elements every array holds: the tiled size in a tiled chunk
    ///        mode (the last tiles padded), nx * ny row-major.
    [[nodiscard]] std::size_t storageSize() const;

    /// @brief A part of material mat's box: a thread's share of it.
    struct Share
    {
        int  mat;
        Part part;
    };

    /**
     * @brief This thread's parts of the boxes `box_of(mat)`: the team splits
     *        the cells of every material's box at once, by weight.
     * @details A material's unit is a tile, or a row of its box when
     *          `row_parts(mat)` (LoopsMd.h). In the order (material, row,
     *          column), thread t of T takes the units whose first cell falls
     *          in [W t / T, W (t + 1) / T), W the cells of all the boxes: one
     *          Share per run of consecutive units (tiles of one band, or rows).
     *          Every thread computes the same split, with no communication;
     *          a unit goes to one thread only.
     */
    template <typename BoxOf, typename RowParts>
    void dealParts(const BoxOf& box_of, const RowParts& row_parts, std::vector<Share>& mine) const;

    /**
     * @brief The iterations. Called by every thread of a parallel region.
     * @param value This thread's share of the gadget result.
     */
    template <typename LOOPS, typename LOOPSCHUNK>
    void computeLoops(const MultiMatAlgo<LOOPS, LOOPSCHUNK>& mm_algo, double& value);

    /**
     * @brief Replay the run serially on plain vectors, compare every array
     *        cell by cell and the result to a relative 1e-9 of the sum of
     *        the absolute terms. Outside a parallel region.
     * @return The number of mismatches (0: the run is right).
     */
    template <typename LOOPS, typename LOOPSCHUNK>
    long verify(const MultiMatAlgo<LOOPS, LOOPSCHUNK>& mm_algo, double result) const;

    // ── The values ───────────────────────────────────────────────────────

    /// @brief The cell number of (i, j): what multi-mat's fields are made of.
    [[nodiscard]] double cellNumber(int i, int j) const noexcept
    {
        return static_cast<double>(i * ny + j);
    }

    /// @brief The presence fraction of cell (i, j) of box @p b: a ramp over
    ///        the interface_width cells along each side, 1 inside.
    [[nodiscard]] double presenceFraction(const Box& b, int i, int j) const noexcept
    {
        const int d = std::min(std::min(i - b.i0, b.i1() - 1 - i), std::min(j - b.j0, b.j1() - 1 - j));
        return d >= interface_width ? 1. : static_cast<double>(d + 1) / static_cast<double>(interface_width + 1);
    }

    /// @brief 1 when the cells [origin, origin + extent) all lie in the
    ///        interior of @p b (presenceFraction() is 1 on all of them).
    template <typename Arr>
    [[nodiscard]] std::optional<double>
    fractionOnTile(const Box& b, const Arr& origin, const Arr& extent) const noexcept
    {
        const int  oi     = static_cast<int>(origin[0]);
        const int  oj     = static_cast<int>(origin[1]);
        const int  ei     = static_cast<int>(extent[0]);
        const int  ej     = static_cast<int>(extent[1]);
        const bool inside = oi >= b.i0 + interface_width && oi + ei <= b.i1() - interface_width &&
                            oj >= b.j0 + interface_width && oj + ej <= b.j1() - interface_width;
        if (inside)
        {
            return 1.;
        }
        return std::nullopt;
    }

    /// @brief The gadget term of a cell (multi-mat's), weighted by the
    ///        presence fraction.
    [[nodiscard]] static double gadget(double fr, double a, double b, double c, double val) noexcept
    {
        return fr * (std::cos(a) + std::sin(b) + std::tan(c)) + std::cos(val);
    }
};

} // namespace multi_mat::md

#endif // SHRINK_ARRAY_HAS_MDSPAN
