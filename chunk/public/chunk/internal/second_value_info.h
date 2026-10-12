/*
 * Copyright 2026 CEA (www.cea.fr)
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

namespace shrink_array
{
/**
 * @brief Chunk2's per-array second-value record (its `inf_array_t`): the
 *        second constant, its slot in the cst2 registry, the array's table
 *        number.
 * @details A named aggregate rather than a tuple so the two ints cannot be
 *          confused; brace-initialised in member order, structured bindings work.
 * @tparam T element type
 */
template <typename T>
struct SecondValueInfo
{
    T   second_value{}; ///< the array's second constant
    int ind     = 0;    ///< its slot in the cst2 registry (chunk_config::no_second_value_index: none)
    int num_tab = 0;    ///< the array's table number
};
} // namespace shrink_array
