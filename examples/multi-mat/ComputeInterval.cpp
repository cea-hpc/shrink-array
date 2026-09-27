
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

/**
 * @file ComputeInterval.cpp
 * @brief Simu::computeInterval(): the largest interval of a chunk and of a
 *        vector material, from their average occupation of the mesh.
 */

#include <cmath>
#include <print>

#include "Simu.h"

void Simu::computeInterval()
{
    if (!all_vector)
    {
        interval_chunk = static_cast<int>(std::round(static_cast<double>(mesh_size) * pc_chunk / 100));

        std::println("\nAverage interval chunk : {}", interval_chunk);
    }

    if (!all_chunk)
    {
        interval_vector = static_cast<int>(std::round(static_cast<double>(mesh_size) * pc_vector / 100));

        std::println("\nAverage interval vector : {}", interval_vector);
    }
}
