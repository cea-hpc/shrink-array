
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

#include <map>

#include "chunk/internal/value_less.h"

// The map from a second value to its index, ordered by ValueLess (see
// value_less.h for what that means per element type).

namespace shrink_array
{
template <typename T>
struct ValueToIntMapHelper
{
    using type = std::map<T, int, ValueLess<T>>;
};

} // namespace shrink_array
