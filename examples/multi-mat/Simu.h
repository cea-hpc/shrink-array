
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

#include <vector>
#include <array>
#include <functional>
#include <algorithm>

#include "Containers.h"

template <typename LOOPS, typename LOOPSCHUNK>
class MultiMatAlgo;

//#define INPUT_DATA

class Simu
{
  public:
    static inline int iter_max_default = 100;
    static inline int nbmat_default    = 50;

    int               nbmat            = nbmat_default;

    int mesh_size = 10000000;

    bool verbose = true;

    int iter_max = iter_max_default;

    int chunk_rate = 90;

    double pc_chunk  = .5;
    double pc_vector = 3;

    int option_chunk     = 3;
    int freq_rtz_default = std::max(iter_max_default / 10, 5);
    int freq_rtz;

    int freq_compress;

    bool all_chunk  = false;
    bool all_vector = false;

    int interval_chunk;
    int interval_vector;

    // 3 arrays multi - mat
    std::array<MultiArray<double>, 3> tabs;

    // The presence fraction of each material, in [0, 1]: 1 inside the
    // material's interval, a ramp of `interface_width` cells at each end, 0
    // elsewhere. A Chunk2 array for the chunk materials, second value 1.
    MultiArray<double, VectChunk2<double>> frac;

    // Cells of the ramp at each end of an interval: well under a chunk, so
    // the interior of an interval is chunks of pure 1 and compresses.
    int interface_width = 16;

    std::vector<bool> dis;

    using VecInt2D = std::vector<std::vector<int>>;

    VecInt2D tab_size;
    VecInt2D tab_start;

    VecInt2D ll;

  public:
    Simu() = default;

    int                      getMeshSize() const;
    const std::vector<bool>& getDis() const;

    std::array<MultiArray<double>, 3>& getTabs()
    {
        return tabs;
    }

    const std::array<MultiArray<double>, 3>& getTabs() const
    {
        return tabs;
    }

    const MultiArray<double>& getTabs(int ind) const
    {
        return tabs[ind];
    }

    const MultiArray<double, VectChunk2<double>>& getFrac() const
    {
        return frac;
    }

    /// @brief The presence fraction at position @p p of an interval of
    ///        @p size cells: a ramp over the first and last interface_width
    ///        cells, 1 in between.
    [[nodiscard]] double presenceFraction(int p, int size) const
    {
        const int d = std::min(p, size - 1 - p); // distance to the nearest end
        return d >= interface_width ? 1. : static_cast<double>(d + 1) / static_cast<double>(interface_width + 1);
    }

    int getOptionChunk() const;

    bool isVerbose() const;

    int getNbmat() const;

    void inputData();

    void createTabs();

    void computeInterval();

    void computeDistribution();

    template <typename LOOPS, typename LOOPSCHUNK>
    void computeLoops(MultiMatAlgo<LOOPS, LOOPSCHUNK>& mm_algo, const std::function<void(bool)>& lbd_stats, double& value);
};