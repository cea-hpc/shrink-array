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

// Central type list used by the example / benchmark executables when
// initializing the shrink_array library:
//
//     shrink_array::ParamChunk::init_indexes<MyChunkTypes>(true);
//
// The library itself no longer carries a hard-coded list of "supported" T;
// each consumer is responsible for declaring the types it uses. Grouping the
// list here avoids duplicating (and forgetting to keep in sync) the tuple in
// every example .cpp. Individual examples that only need a subset are
// encouraged to declare their own tuple locally rather than pulling this one.

#include <complex>
#include <tuple>

using MyChunkTypes =
    std::tuple<int, double, bool, short, unsigned long, long, std::complex<double>, float, unsigned char>;
