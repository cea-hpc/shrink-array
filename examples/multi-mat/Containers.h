
/*
* Copyright 2024-2025 CEA (www.cea.fr)
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

template <typename T>
class VectChunk : public VectCom<T>, public shrink_array::ArrayChunk<T>
{
  public:

    void create();
};

template <typename T>
class Vect : public VectCom<T>, public std::vector<T>
{
  public:

    void create();
};

template <typename T>
class MultiArray : public std::vector<VectCom<T>*>
{
  public:

    using value_type = T;

    using BaseClass = std::vector<VectCom<T>*>;

    using vec_dir   = Vect<T>;
    using vec_chunk = VectChunk<T>;

    void create(int nbmat, int mesh_size);
};

template <typename T>
void MultiArray<T>::create(int nbmat, int mesh_size)
{
    BaseClass::resize(nbmat, nullptr);

    for (int k = 0; k < nbmat; ++k)
    {
        if (dispatch.isChunk(k))
        {
            auto* v = new vec_chunk;
            v->setInd(k);
            v->resize(mesh_size);
            (*this)[k] = v;
        }
        else
        {
            auto* v = new vec_dir;
            v->setInd(k);
            v->resize(mesh_size);
            (*this)[k] = v;
        }
    }
}
