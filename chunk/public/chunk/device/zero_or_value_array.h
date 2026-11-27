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
 * @file zero_or_value_array.h
 * @brief zero_or_value::DeviceArray, the GPU counterpart of zero_or_value::Array (chunk/zero_or_value_array.h): a
 *        device::Array (chunk/device/array.h) that folds a chunk back to T{} or to the array's second value, set_second_value() between phases (8.2).
 * @details Names only, over the one device implementation: zero_or_value::DeviceArray<T, S> is
 *          device::Array<T, S, device::Variant::zero_or_value>, zero_or_value::DeviceView the view of its read
 *          kernels, zero_or_value::DeviceWriter the access of its write kernels. S defaults to 1024,
 *          the GPU geometry (docs/gpu_design.md section 13). What the host array has and the
 *          device one does not: resize, push_back, iterators, element access from the host
 *          (to_host(), image()). Built by CUDA or HIP; the host arrays' headers are not
 *          included (a transfer goes through chunk/chunk_exchange.h's ArrayState, in a host
 *          translation unit: Array(state), image()). The Kokkos facade of the same variant is
 *          zero_or_value::KokkosArray (chunk/device/kokkos_adaptor.h).
 */

#include "chunk/device/array.h" // device::Array, device::View, device::Writer, device::Variant

namespace shrink_array::zero_or_value
{

/// @brief A zero_or_value array on the GPU: elements of T in chunks of S.
template <typename T, unsigned int S = 1024>
using DeviceArray = device::Array<T, S, device::Variant::zero_or_value>;

/// @brief What a read kernel gets of a zero_or_value::DeviceArray (DeviceArray::view()).
template <typename T, unsigned int S = 1024>
using DeviceView = device::View<T, S, device::Variant::zero_or_value>;

/// @brief What a write kernel gets of a zero_or_value::DeviceArray (DeviceArray::writer()).
template <typename T, unsigned int S = 1024>
using DeviceWriter = device::Writer<T, S, device::Variant::zero_or_value>;

} // namespace shrink_array::zero_or_value
