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
 * @file device_convert_validation.h
 * @brief What the two halves of device_convert_validation share: the C++ half
 *        (device_convert_validation.cpp, the host zero_any::Array, export_array /
 *        import_array) calls the device half (device_convert_device.cu, device::Array)
 *        through this header, which names neither. The pattern a program follows: nvcc
 *        cannot compile the host arrays' headers (docs/gpu_design.md, section 14).
 */

#include <cstdint> // std::int64_t
#include <vector>  // std::vector

#include "chunk/internal/chunk_state.h" // ArrayState, ArrayImage

namespace device_convert
{

/// @brief The variant of the device array (device::Variant, which this header does not name).
enum class DeviceVariant
{
    zero_any,
    zero,
    zero_or_value, ///< second value 2.5, declared after the first assign()
};

/// @brief What the device half did with a host state, step by step.
struct DeviceRoundTrip
{
    std::vector<double>              after_assign;              ///< every element after assign(state)
    unsigned long long               materialised_after_assign; ///< chunks with a buffer then
    unsigned long long               folded;                    ///< compress() after the two writes
    unsigned long long               materialised_final;        ///< chunks with a buffer after it
    shrink_array::ArrayImage<double> image;                     ///< image() then
    std::vector<double>              after_reassign;            ///< every element after assign(state) again
    bool                             arena_consistent;          ///< buffers in use == materialised chunks, at the end
};

/**
 * @brief On a device::Array of @p state's size and chunk size, with an arena of 4 buffers
 *        (assign() must grow it): assign(state); a(i) = 2.5 on [lo1, hi1) and a(i) = i on
 *        [lo2, hi2) by parallel_for_chunks; compress(); image(); assign(state) again over the
 *        buffers it holds.
 * @throws std::runtime_error for a chunk size the device half was not built for.
 */
/// @brief @p state into a device array of variant @p variant, [lo1, hi1) written 2.5 and
///        [lo2, hi2) written i, compress(), image(), assign again.
DeviceRoundTrip device_round_trip(const shrink_array::ArrayState<double>& state,
                                  DeviceVariant                           variant,
                                  std::int64_t                            lo1,
                                  std::int64_t                            hi1,
                                  std::int64_t                            lo2,
                                  std::int64_t                            hi2);

} // namespace device_convert
