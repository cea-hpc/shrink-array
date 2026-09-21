
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

#include "Loops.h"
#include "Containers.h"

template <typename LOOPS = LoopsSimd, typename LOOPSCHUNK = Loops>
class MultiMatAlgo
{
    // The concrete array behind material `mat` of a MultiArray: the one the
    // container names (vec_dir / vec_chunk), so a MultiArray of presence
    // fractions hands out its VectChunk2 the same way a field hands out its
    // VectChunk.
    template <typename MA>
    static auto& getMatVecDir(const MA& gdr, int mat)
    {
        using Vec = typename MA::vec_dir;

        return *(static_cast<Vec*>(gdr[mat]));
    }

    template <typename MA>
    static auto& getMatVecChunk(const MA& gdr, int mat)
    {
        using Vec = typename MA::vec_chunk;

        return *(static_cast<Vec*>(gdr[mat]));
    }

    template <typename LAM>
    void single_direct_int(int& mat, const LAM& lbd_loop_body)
    {
        const LOOPS loops;

        auto lbd_get_mat_vec = [mat](const auto& gdr) -> auto&
        {
            return getMatVecDir(gdr, mat);
        };

        lbd_loop_body(mat, lbd_get_mat_vec, loops);
    }

    template <typename LAM>
    void single_chunk_int(int& mat, const LAM& lbd_loop_body)
    {
        const LOOPSCHUNK loops;

        auto lbd_get_mat_vec = [mat](const auto& gdr) -> auto&
        {
            return getMatVecChunk(gdr, mat);
        };

        lbd_loop_body(mat, lbd_get_mat_vec, loops);
    }

    template <typename LAM>
    void single_int(int& mat, const LAM& lbd_loop_body)
    {
        const bool chunk = dispatch.isChunk(mat);

        if (chunk)
        {
            single_chunk_int(mat, lbd_loop_body);
        }
        else
        {
            single_direct_int(mat, lbd_loop_body);
        }
    }

  public:
    template <typename LAM>
    void loop(int matdeb, int matfin, const LAM& lbd_loop_body)
    {
        for (int mat = matdeb; mat < matfin; ++mat)
        {
            single_int(mat, lbd_loop_body);
        }
    }
};