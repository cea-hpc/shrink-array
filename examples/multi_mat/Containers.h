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

#include "chunk/zero_array.h"          // shrink_array::zero::Array
#include "chunk/zero_or_value_array.h" // shrink_array::zero_or_value::Array
#include "Dispatch.h"                  // Dispatch (extern dispatch)

extern Dispatch dispatch;

/// @brief The common base of the per-material arrays.
template <typename T>
class MatArrayBase
{
  public:
    using value_type = T;

    /// @brief Virtual: MultiArray deletes its arrays through MatArrayBase<T>*,
    ///        and the derived classes hold a chunk array or a vector whose
    ///        destructor must run.
    virtual ~MatArrayBase() = default;
};

/// @brief A field of a "chunk" material: compressed where it is zero.
template <typename T>
struct MatZeroArray : public MatArrayBase<T>, public shrink_array::zero::Array<T>
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
 * @details zero_or_value compresses the chunks that are all 0 (the first constant)
 *          AND the ones that are all 1 (the array's second value), so only
 *          the interface chunks are ever materialised.
 */
template <typename T>
struct MatZeroOrValueArray : public MatArrayBase<T>, public shrink_array::zero_or_value::Array<T>
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
struct MatVectorArray : public MatArrayBase<T>, public std::vector<T>
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
 * @tparam VecChunkT The chunk array of the chunk materials: MatZeroArray<T> for a
 *                  field, MatZeroOrValueArray<T> for the presence fraction.
 */
template <typename T, typename VecChunkT = MatZeroArray<T>>
class MultiArray : public std::vector<MatArrayBase<T>*>
{
  public:
    using value_type = T;

    using BaseClass = std::vector<MatArrayBase<T>*>;

    using VecDir   = MatVectorArray<T>;
    using VecChunk = VecChunkT;

    MultiArray() = default;

    /// @brief Frees the arrays create() made (through MatArrayBase's virtual
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
    void create(int n_materials, int mesh_size);

  private:
    /// @brief Delete every array and empty the vector.
    void release() noexcept
    {
        for (MatArrayBase<T>*& vec : *this)
        {
            delete vec;
            vec = nullptr;
        }
        BaseClass::clear();
    }
};

template <typename T, typename VecChunk>
void MultiArray<T, VecChunk>::create(int n_materials, int mesh_size)
{
    release(); // a second create() does not leak the first one's arrays
    BaseClass::resize(n_materials, nullptr);

    for (int k = 0; k < n_materials; ++k)
    {
        if (dispatch.is_chunk(k))
        {
            auto* vec = new VecChunk;
            vec->create(mesh_size);
            (*this)[k] = vec;
        }
        else
        {
            auto* vec = new VecDir;
            vec->create(mesh_size);
            (*this)[k] = vec;
        }
    }
}
