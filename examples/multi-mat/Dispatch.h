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
 * @file Dispatch.h
 * @brief Which materials are stored in chunk arrays, which in std::vectors.
 */

#include <vector>

/// @brief The storage kind of every material, decided once by Simu::inputData().
class Dispatch
{
  public:
    /// @brief Room for @p sz materials, all std::vector until set().
    void create(int sz)
    {
        tab_chunk.resize(sz);
    }

    /// @brief The kinds, one per material.
    void set(const std::vector<bool>& t)
    {
        tab_chunk = t;
    }

    /// @brief Is material @p i kept in chunk arrays?
    [[nodiscard]] bool isChunk(int i) const
    {
        return tab_chunk[i];
    }

    /// @brief Is material @p i kept in std::vectors?
    [[nodiscard]] bool isVector(int i) const
    {
        return !tab_chunk[i];
    }

  private:
    std::vector<bool> tab_chunk; ///< true: chunk arrays; false: std::vector
};
