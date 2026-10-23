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
 * @file Containers.h
 * @brief The per-material arrays: a std::vector or a chunk array behind one
 *        base, and the multi-material array that holds one per material.
 */

#include <vector> // std::vector

#include "chunk/zero_array.h"  // shrink_array::zero::Array
#include "chunk/zero_one_value_array.h" // shrink_array::zero_one_value::Array
#include "Dispatch.h"           // Dispatch (extern dispatch)

extern Dispatch dispatch;

/// @brief What every per-material array has: its material index.
template <typename T>
class VectCom
{
  public:
    using value_type = T;

    /// @brief Virtual: MultiArray deletes its arrays through VectCom<T>*,
    ///        and the derived classes hold a chunk array or a vector whose
    ///        destructor must run.
    virtual ~VectCom() = default;

    /// @brief The material this array belongs to.
    void set_ind(int i)
    {
        ind_ = i;
    }

  private:
    int ind_;
};

/// @brief A field of a "chunk" material: compressed where it is zero.
template <typename T>
struct VectChunk : public VectCom<T>, public shrink_array::zero::Array<T>
{
    /// @brief @p mesh_size elements, all zero -- compressed.
    void create(int mesh_size)
    {
        this->resize(mesh_size);
    }
};

/**
 * @brief The presence fraction of a "chunk" material, in [0, 1]: 0 where
 *        the material is absent, 1 where the cell is pure, in between at the
 *        interfaces.
 * @details Chunk2 compresses the chunks that are all 0 (the first constant)
 *          AND the ones that are all 1 (the array's second value), so only
 *          the interface chunks are ever materialised.
 */
template <typename T>
struct VectChunk2 : public VectCom<T>, public shrink_array::zero_one_value::Array<T>
{
    /// @brief The fraction of a pure cell, the second constant of the array.
    static constexpr T full = 1;

    /// @brief @p mesh_size elements, all zero, and `full` registered as the
    ///        second value.
    void create(int mesh_size)
    {
        this->resize(mesh_size);
        this->set_second_value(full, false);
    }
};

/// @brief An array of a "vector" material: a plain std::vector.
template <typename T>
struct Vect : public VectCom<T>, public std::vector<T>
{
    /// @brief @p mesh_size elements, all zero.
    void create(int mesh_size)
    {
        this->resize(mesh_size);
    }
};

/**
 * @brief One array per material, a std::vector or a chunk array as the
 *        dispatch says.
 * @tparam T        Element type.
 * @tparam VecChunkT The chunk array of the chunk materials: VectChunk<T> for a
 *                  field, VectChunk2<T> for the presence fraction.
 */
template <typename T, typename VecChunkT = VectChunk<T>>
class MultiArray : public std::vector<VectCom<T>*>
{
  public:
    using value_type = T;

    using BaseClass = std::vector<VectCom<T>*>;

    using VecDir   = Vect<T>;
    using VecChunk = VecChunkT;

    MultiArray() = default;

    /// @brief Frees the arrays create() made (through VectCom's virtual
    ///        destructor, so the chunk arrays' teardown runs).
    ~MultiArray()
    {
        release();
    }

    // Owns raw pointers: a copy would delete them twice, and the default
    // move assignment would leak the target's. Neither is needed.
    MultiArray(const MultiArray&)            = delete;
    MultiArray& operator=(const MultiArray&) = delete;
    MultiArray(MultiArray&&)                 = delete;
    MultiArray& operator=(MultiArray&&)      = delete;

    /// @brief One array of @p mesh_size elements per material, of the kind
    ///        the dispatch gives the material. Frees those of a previous
    ///        call first.
    void create(int nbmat, int mesh_size);

  private:
    /// @brief Delete every array and empty the vector.
    void release() noexcept
    {
        for (VectCom<T>*& p : *this)
        {
            delete p;
            p = nullptr;
        }
        BaseClass::clear();
    }
};

template <typename T, typename VecChunk>
void MultiArray<T, VecChunk>::create(int nbmat, int mesh_size)
{
    release(); // a second create() does not leak the first one's arrays
    BaseClass::resize(nbmat, nullptr);

    for (int k = 0; k < nbmat; ++k)
    {
        if (dispatch.is_chunk(k))
        {
            auto* v = new VecChunk;
            v->set_ind(k);
            v->create(mesh_size);
            (*this)[k] = v;
        }
        else
        {
            auto* v = new VecDir;
            v->set_ind(k);
            v->create(mesh_size);
            (*this)[k] = v;
        }
    }
}
