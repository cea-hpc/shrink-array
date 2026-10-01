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
 * @file MultiMatAlgo.h
 * @brief One loop body, two storages: the dispatch of a material to its
 *        concrete arrays and its loop policy.
 */

#include "Loops.h"
#include "Containers.h"

/**
 * @brief Runs a loop body once per material, handing it the material's
 *        concrete arrays and the loop policy of its storage.
 * @details The body receives `(int& mat, get_mat_vec, loops)`: `get_mat_vec`
 *          turns a MultiArray into the array of `mat` (a std::vector or a
 *          chunk array), `loops` is the policy to run its loops with. One
 *          body, written once, is therefore instantiated twice -- with the
 *          vectorised policy on the std::vector path, the scalar one on the
 *          chunk path.
 * @tparam LOOPS      Policy of the std::vector materials.
 * @tparam LOOPSCHUNK Policy of the chunk materials.
 */
template <typename LOOPS = LoopsSimd, typename LOOPSCHUNK = Loops>
class MultiMatAlgo
{
  public:
    /**
     * @brief The body on every material of [matdeb, matfin).
     * @note The body may set `mat` to end the loop early (the stats print
     *       does, after the first chunk material).
     */
    template <typename LAM>
    void loop(int matdeb, int matfin, const LAM& lbd_loop_body) const
    {
        for (int mat = matdeb; mat < matfin; ++mat)
        {
            single_int(mat, lbd_loop_body);
        }
    }

  private:
    /// @brief The body on a std::vector material.
    template <typename LAM>
    void single_direct_int(int& mat, const LAM& lbd_loop_body) const
    {
        const LOOPS loops;

        const auto lbd_get_mat_vec = [mat](const auto& gdr) -> auto& { return getMatVecDir(gdr, mat); };

        lbd_loop_body(mat, lbd_get_mat_vec, loops);
    }

    /// @brief The body on a chunk material.
    template <typename LAM>
    void single_chunk_int(int& mat, const LAM& lbd_loop_body) const
    {
        const LOOPSCHUNK loops;

        const auto lbd_get_mat_vec = [mat](const auto& gdr) -> auto& { return getMatVecChunk(gdr, mat); };

        lbd_loop_body(mat, lbd_get_mat_vec, loops);
    }

    /// @brief The body on material @p mat, whichever its storage.
    template <typename LAM>
    void single_int(int& mat, const LAM& lbd_loop_body) const
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

    // The concrete array behind material `mat` of a MultiArray: the one the
    // container names (vec_dir / vec_chunk), so a MultiArray of presence
    // fractions hands out its VectChunk2 the same way a field hands out its
    // VectChunk.

    /// @brief The std::vector of material @p mat in @p gdr.
    template <typename MA>
    static auto& getMatVecDir(const MA& gdr, int mat)
    {
        using Vec = MA::vec_dir;

        return *(static_cast<Vec*>(gdr[mat]));
    }

    /// @brief The chunk array of material @p mat in @p gdr.
    template <typename MA>
    static auto& getMatVecChunk(const MA& gdr, int mat)
    {
        using Vec = MA::vec_chunk;

        return *(static_cast<Vec*>(gdr[mat]));
    }
};
