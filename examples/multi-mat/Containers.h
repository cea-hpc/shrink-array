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

#include <vector>

#include "chunk/array_chunk.h"
#include "chunk/array_chunk2.h"
#include "Dispatch.h"

extern Dispatch dispatch;

template <typename T>
class VectCom
{
  private:
    int ind;

  public:
    using value_type = T;

    VectCom()  = default;
    ~VectCom() = default;

    void setInd(int i)
    {
        ind = i;
    }
};

/// @brief A field of a "chunk" material: compressed where it is zero.
template <typename T>
class VectChunk : public VectCom<T>, public shrink_array::ArrayChunk<T>
{
  public:
    void create(int mesh_size)
    {
        this->resize(mesh_size);
    }
};

/// @brief The presence fraction of a "chunk" material, in [0, 1]: 0 where
///        the material is absent, 1 where the cell is pure, in between at the
///        interfaces. Chunk2 compresses the chunks that are all 0 (the first
///        constant) AND the ones that are all 1 (the array's second value),
///        so only the interface chunks are ever materialised.
template <typename T>
class VectChunk2 : public VectCom<T>, public shrink_array::ArrayChunk2<T>
{
  public:
    /// @brief The fraction of a pure cell, the second constant of the array.
    static constexpr T full = 1;

    void create(int mesh_size)
    {
        this->resize(mesh_size);
        this->set_second_value(full, false);
    }
};

template <typename T>
class Vect : public VectCom<T>, public std::vector<T>
{
  public:
    void create(int mesh_size)
    {
        this->resize(mesh_size);
    }
};

/**
 * @brief One array per material, a std::vector or a chunk array as the
 *        dispatch says.
 * @tparam VecChunk The chunk array of the chunk materials: VectChunk<T> for a
 *                  field, VectChunk2<T> for the presence fraction.
 */
template <typename T, typename VecChunk = VectChunk<T>>
class MultiArray : public std::vector<VectCom<T>*>
{
  public:

    using value_type = T;

    using BaseClass = std::vector<VectCom<T>*>;

    using vec_dir   = Vect<T>;
    using vec_chunk = VecChunk;

    void create(int nbmat, int mesh_size);
};

template <typename T, typename VecChunk>
void MultiArray<T, VecChunk>::create(int nbmat, int mesh_size)
{
    BaseClass::resize(nbmat, nullptr);

    for (int k = 0; k < nbmat; ++k)
    {
        if (dispatch.isChunk(k))
        {
            auto* v = new vec_chunk;
            v->setInd(k);
            v->create(mesh_size);
            (*this)[k] = v;
        }
        else
        {
            auto* v = new vec_dir;
            v->setInd(k);
            v->create(mesh_size);
            (*this)[k] = v;
        }
    }
}
