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

/**
 * @file device_convert_device.cu
 * @brief The device half of device_convert_validation (see device_convert_validation.h):
 *        device::Array::assign() and image() around a write phase and a compress(), in the
 *        zero_any, the zero or the zero_or_value variant.
 */

#include <cstdint>   // std::int64_t
#include <stdexcept> // std::runtime_error
#include <string>    // std::to_string

#include "chunk/device/array.h" // device::Array

#include "device_convert_validation.h" // DeviceRoundTrip, device_round_trip

namespace sd = shrink_array::device;

namespace device_convert
{

namespace
{

struct Constant
{
    double value;

    __device__ double operator()(std::int64_t /*i*/) const noexcept
    {
        return value;
    }
};

struct Index
{
    __device__ double operator()(std::int64_t i) const noexcept
    {
        return static_cast<double>(i);
    }
};

template <unsigned int S, sd::Variant V>
DeviceRoundTrip round_trip_in(const shrink_array::ArrayState<double>& state,
                              std::int64_t                            lo1,
                              std::int64_t                            hi1,
                              std::int64_t                            lo2,
                              std::int64_t                            hi2)
{
    DeviceRoundTrip         r{};
    sd::Array<double, S, V> a(static_cast<std::int64_t>(state.size), 4);
    a.assign(state);
    r.after_assign              = a.to_host();
    r.materialised_after_assign = a.materialised();
    if constexpr (V == sd::Variant::zero_or_value)
    {
        a.set_second_value(2.5); // the ten chunks written 2.5 fold back to it
    }
    a.parallel_for_chunks(lo1, hi1, Constant{2.5});
    a.parallel_for_chunks(lo2, hi2, Index{});
    r.folded             = a.compress();
    r.materialised_final = a.materialised();
    r.image              = a.image();
    a.assign(state);
    r.after_reassign   = a.to_host();
    r.arena_consistent = a.arena().in_use() == a.materialised();
    return r;
}

template <unsigned int S>
DeviceRoundTrip round_trip(const shrink_array::ArrayState<double>& state,
                           DeviceVariant                           variant,
                           std::int64_t                            lo1,
                           std::int64_t                            hi1,
                           std::int64_t                            lo2,
                           std::int64_t                            hi2)
{
    switch (variant)
    {
        case DeviceVariant::zero:
            return round_trip_in<S, sd::Variant::zero>(state, lo1, hi1, lo2, hi2);
        case DeviceVariant::zero_or_value:
            return round_trip_in<S, sd::Variant::zero_or_value>(state, lo1, hi1, lo2, hi2);
        default:
            return round_trip_in<S, sd::Variant::zero_any>(state, lo1, hi1, lo2, hi2);
    }
}

} // namespace

DeviceRoundTrip device_round_trip(const shrink_array::ArrayState<double>& state,
                                  DeviceVariant                           variant,
                                  std::int64_t                            lo1,
                                  std::int64_t                            hi1,
                                  std::int64_t                            lo2,
                                  std::int64_t                            hi2)
{
    switch (state.chunk_size)
    {
        case 64:
            return round_trip<64>(state, variant, lo1, hi1, lo2, hi2);
        case 128:
            return round_trip<128>(state, variant, lo1, hi1, lo2, hi2);
        case 256:
            return round_trip<256>(state, variant, lo1, hi1, lo2, hi2);
        case 512:
            return round_trip<512>(state, variant, lo1, hi1, lo2, hi2);
        case 1024:
            return round_trip<1024>(state, variant, lo1, hi1, lo2, hi2);
        case 2048:
            return round_trip<2048>(state, variant, lo1, hi1, lo2, hi2);
        default:
            throw std::runtime_error("device_round_trip: chunk size " + std::to_string(state.chunk_size) +
                                     " not built");
    }
}

} // namespace device_convert
