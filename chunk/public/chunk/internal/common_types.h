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

// A sample type list for ParamChunk::init_indexes<Types>(), which registers
// the managers of those element types up front:
//
//     shrink_array::ParamChunk::init_indexes<MyChunkTypes>(true);
//
// Nothing includes it: the arrays register their element types themselves,
// and the examples call init_indexes(bool).

#include <complex> // std::complex
#include <tuple>   // std::tuple (MyChunkTypes)

/// @brief A sample element-type list for ParamChunk::init_indexes<Types>().
using MyChunkTypes =
    std::tuple<int, double, bool, short, unsigned long, long, std::complex<double>, float, unsigned char>;
