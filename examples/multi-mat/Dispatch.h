
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

class Dispatch
{
  private:
    std::vector<bool> tab_chunk;

  public:
    void create(int sz)
    {
        tab_chunk.resize(sz);
    }

    void set(const std::vector<bool>& t)
    {
        tab_chunk = t;
    }
    [[nodiscard]] int isChunk(int i) const
    {
        return tab_chunk[i];
    }
    [[nodiscard]] int isVector(int i) const
    {
        return !tab_chunk[i];
    }
};
