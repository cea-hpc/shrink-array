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

/**
 * @file device_memory_validation.h
 * @brief What the two halves of device_memory_validation share: the C++ half
 *        (device_memory_validation.cpp, a host array) calls the device half
 *        (device_memory_device.cu, device::Array) through this header, which names neither
 *        (nvcc cannot compile the host arrays' headers: docs/gpu_design.md, section 14).
 */

#include <cstdint> // std::int64_t
#include <vector>  // std::vector

#include "chunk/internal/chunk_state.h" // ArrayImage

namespace device_memory
{

/// @brief One write phase: a(i) = target[i] for every i of the spans [lo[k], hi[k]).
struct Round
{
    std::vector<std::int64_t> lo;        ///< the spans' starts
    std::vector<std::int64_t> hi;        ///< the spans' ends (excluded)
    std::vector<double>       target;    ///< one value per element of the array
    bool                      by_assign; ///< Writer::assign per element, else Writer::write per span
};

/**
 * @brief The rounds on a device array of @p n elements at chunk size @p chunk_size, of the
 *        zero variant if @p zero_variant, else zero_any; after each, finish() and compress().
 * @return image() after each round: the device layout, chunk by chunk.
 */
std::vector<shrink_array::ArrayImage<double>>
device_replay(std::int64_t n, unsigned int chunk_size, bool zero_variant, const std::vector<Round>& rounds);

} // namespace device_memory
