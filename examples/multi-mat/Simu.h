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
 * @file Simu.h
 * @brief The simulation: its parameters, its arrays, and the steps main()
 *        runs in order.
 */

#include <vector>     // std::vector
#include <array>      // std::array
#include <functional> // std::function
#include <algorithm>  // std::min, std::max
#include <cstddef>    // std::size_t
#include <version>    // __cpp_lib_mdspan
#if defined(__cpp_lib_mdspan)
    #include <mdspan> // std::mdspan, std::dextents
#endif

#include "Containers.h" // MultiArray, VectChunk2

template <typename LoopsVec, typename LoopsChunk>
class MultiMatAlgo;

/**
 * @brief A mock multi-material simulation: every material occupies, at
 *        each iteration, a random interval of the mesh; its fields are
 *        written there and read back by a gadget computation.
 * @details Steps, in the order main() runs them: input_data() (parameters
 *          and the storage of every material), compute_interval(),
 *          create_tabs(), compute_distribution() (the intervals of every
 *          iteration, drawn up front), compute_loops() (the iterations, in
 *          one parallel region).
 */
struct Simu
{
    static inline int iter_max_default = 100; ///< Iterations, unless asked
    static inline int nbmat_default    = 50;  ///< Materials, unless asked

    using VecInt2D = std::vector<std::vector<int>>;

    Simu() = default;

    /// @brief Number of cells (mesh_size).
    int get_mesh_size() const;
    /// @brief Storage of every material (dis): true = chunk.
    const std::vector<bool>& get_dis() const;

    /// @brief The three fields of every material (tabs).
    std::array<MultiArray<double>, 3>& get_tabs()
    {
        return tabs;
    }

    /// @brief The three fields of every material (tabs), read-only.
    const std::array<MultiArray<double>, 3>& get_tabs() const
    {
        return tabs;
    }

    /// @brief Field @p ind of every material.
    const MultiArray<double>& get_tabs(int ind) const
    {
        return tabs[ind];
    }

    /// @brief The presence fraction of every material.
    const MultiArray<double, VectChunk2<double>>& get_frac() const
    {
        return frac;
    }

    /// @brief The presence fraction at position @p p of an interval of
    ///        @p size cells: a ramp over the first and last interface_width
    ///        cells, 1 in between.
    [[nodiscard]] double presence_fraction(int p, int size) const
    {
        const int d = std::min(p, size - 1 - p); // distance to the nearest end
        return d >= interface_width ? 1. : static_cast<double>(d + 1) / static_cast<double>(interface_width + 1);
    }

    /// @brief Storage of the materials (option_chunk): 1 all chunk, 2 all vector, 3 random 50/50,
    ///        4 random at chunk_rate.
    int get_option_chunk() const;

    /// @brief Whether per-iteration prints are on (verbose).
    bool is_verbose() const;

    /// @brief Number of materials (nbmat).
    int get_nbmat() const;

    /// @brief The parameters -- asked on stdin under
    ///        multi_mat::interactive_input, the defaults above otherwise --
    ///        and the storage of every material (InputData.cpp).
    void input_data();

    /// @brief The three fields and the presence fraction of every material.
    void create_tabs();

    /// @brief The largest intervals, from pc_chunk / pc_vector.
    void compute_interval();

    /// @brief The interval of every material at every iteration, drawn now
    ///        so the iterations need no sequential step.
    void compute_distribution();

    /**
     * @brief The iterations. Called by every thread of a parallel region.
     * @param mm_algo   The per-material dispatch.
     * @param lbd_stats Prints the filling rates; called on one thread now
     *                  and then when verbose.
     * @param value     This thread's share of the gadget result.
     */
    template <typename LoopsVec, typename LoopsChunk>
    void compute_loops(MultiMatAlgo<LoopsVec, LoopsChunk>& mm_algo,
                       const std::function<void(bool)>&    lbd_stats,
                       double&                             value);

    int nbmat = nbmat_default; ///< Materials

    int mesh_size = 10000000; ///< Cells

    bool verbose = true; ///< Per-iteration prints

    int iter_max = iter_max_default; ///< Iterations

    int chunk_rate = 90; ///< Option 4: percentage of chunk materials

    double pc_chunk  = .5; ///< Average interval of a chunk material, % of the mesh
    double pc_vector = 3;  ///< Average interval of a vector material, % of the mesh

    int option_chunk =
        3; ///< Storage of the materials: 1 all chunk, 2 all vector, 3 random 50/50, 4 random at chunk_rate
    int freq_rtz_default = std::max(iter_max_default / 10, 5);
    int freq_rtz         = freq_rtz_default; ///< Every this many iterations the fields are zeroed instead of written

    int freq_compress = 0; ///< Manual mode: every this many iterations the fields are compressed (input_data())

    bool all_chunk  = false; ///< Every material is a chunk one
    bool all_vector = false; ///< Every material is a vector one

    // Largest intervals, set by compute_interval() for the kinds in use. The
    // other one stays 1: compute_distribution() builds the distributions of
    // both kinds, and [1, 1] is a valid range where an unset value was not.
    int interval_chunk  = 1; ///< Largest interval of a chunk material, cells
    int interval_vector = 1; ///< Largest interval of a vector material, cells

    // 3 arrays multi - mat
    std::array<MultiArray<double>, 3> tabs; ///< The three fields of every material

    /// @brief The presence fraction of each material, in [0, 1]: 1 inside
    ///        the material's interval, a ramp of `interface_width` cells at
    ///        each end, 0 elsewhere. A zero_one_value array for the chunk materials,
    ///        second value 1.
    MultiArray<double, VectChunk2<double>> frac;

    /// @brief Cells of the ramp at each end of an interval: well under a
    ///        chunk, so the interior of an interval is chunks of pure 1 and
    ///        compresses.
    int interface_width = 16;

    std::vector<bool> dis; ///< Storage of every material: true = chunk

    /// @brief An [iteration][material] table: one contiguous buffer, read and
    ///        written as t[it, mat] through a 2D mdspan over it. Without
    ///        <mdspan> (libstdc++ before GCC 16), the same row-major index
    ///        by hand.
    class IterMatTable
    {
      public:
#if defined(__cpp_lib_mdspan)
        void resize(int n_iter, int n_mat)
        {
            extents_ = Extents(static_cast<std::size_t>(n_iter), static_cast<std::size_t>(n_mat));
            data_.assign(extents_.extent(0) * extents_.extent(1), 0);
        }

        /// int& on a mutable table, const int& on a const one.
        auto& operator[](this auto& self, int it, int mat)
        {
            return std::mdspan(self.data_.data(), self.extents_)[it, mat];
        }

      private:
        using Extents = std::dextents<std::size_t, 2>;
        std::vector<int> data_;
        Extents          extents_{0, 0};
#else
        void resize(int n_iter, int n_mat)
        {
            n_mat_ = static_cast<std::size_t>(n_mat);
            data_.assign(static_cast<std::size_t>(n_iter) * n_mat_, 0);
        }

        /// int& on a mutable table, const int& on a const one.
        auto& operator[](this auto& self, int it, int mat)
        {
            return self.data_[(static_cast<std::size_t>(it) * self.n_mat_) + static_cast<std::size_t>(mat)];
        }

      private:
        std::vector<int> data_;
        std::size_t      n_mat_ = 0;
#endif
    };

    IterMatTable tab_size;  ///< [iteration, material] size of the interval
    IterMatTable tab_start; ///< [iteration, material] first cell of the interval

    VecInt2D ll; ///< [material] the cells of its current interval (indirection)
};
