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
 * @file device_memory_device.cu
 * @brief The device half of device_memory_validation (see device_memory_validation.h): the
 *        rounds by Writer::write (one block per span) or Writer::assign (one thread per
 *        element), each followed by finish(), compress() and image().
 */

#include <cstddef>   // std::size_t
#include <cstdint>   // std::int64_t
#include <stdexcept> // std::runtime_error
#include <string>    // std::to_string
#include <vector>    // std::vector

#include "chunk/device/array.h"   // device::Array, Writer, Variant
#include "chunk/device/backend.h" // allocate, copy, release

#include "device_memory_validation.h" // Round, device_replay

namespace sd = shrink_array::device;

namespace device_memory
{

namespace
{

constexpr int block = 256;

/// @brief f(i) = target[i], on the device.
struct FromArray
{
    const double* target;

    __device__ double operator()(std::int64_t i) const noexcept
    {
        return target[i];
    }
};

/// @brief Span k: its chunks by Writer::write, one block per span.
template <unsigned int S, sd::Variant V>
__global__ void __launch_bounds__(block) write_spans(sd::Writer<double, S, V> w,
                                                     const std::int64_t*      lo,
                                                     const std::int64_t*      hi,
                                                     std::int64_t             m,
                                                     FromArray                f)
{
    for (std::int64_t k = blockIdx.x; k < m; k += gridDim.x)
    {
        for (std::int64_t c = lo[k] / S; c * S < hi[k]; ++c)
        {
            const std::int64_t clo = c * S > lo[k] ? c * S : lo[k];
            const std::int64_t chi = (c + 1) * S < hi[k] ? (c + 1) * S : hi[k];
            w.template write<block>(c, clo, chi, f);
        }
    }
}

/// @brief Entry k: a(index[k]) = target[index[k]] by Writer::assign, grid-stride.
template <unsigned int S, sd::Variant V>
__global__ void assign_elements(sd::Writer<double, S, V> w, const std::int64_t* index, std::int64_t m, FromArray f)
{
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t k = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x; k < m; k += stride)
    {
        w.assign(index[k], f(index[k]));
    }
}

/// @details A small arena (a quarter of the chunks): the rounds grow it, write failed spans
///          again and replay appends, as a real phase may.
template <unsigned int S, sd::Variant V>
std::vector<shrink_array::ArrayImage<double>> replay(std::int64_t n, const std::vector<Round>& rounds)
{
    const std::int64_t                            n_chunks = (n + S - 1) / S;
    sd::Array<double, S, V>                       a(n, static_cast<unsigned long long>(n_chunks / 4 + 1));
    std::vector<shrink_array::ArrayImage<double>> images;
    double* const                                 d_target = sd::allocate<double>(static_cast<std::size_t>(n));
    const FromArray                               f{d_target};
    for (const Round& r : rounds)
    {
        sd::copy(d_target, r.target.data(), static_cast<std::size_t>(n));
        if (r.by_assign)
        {
            std::vector<std::int64_t> index;
            for (std::size_t k = 0; k < r.lo.size(); ++k)
            {
                for (std::int64_t i = r.lo[k]; i < r.hi[k]; ++i)
                {
                    index.push_back(i);
                }
            }
            if (!index.empty())
            {
                std::int64_t* const d_index = sd::allocate<std::int64_t>(index.size());
                sd::copy(d_index, index.data(), index.size());
                assign_elements<S, V><<<512, 256>>>(a.writer(), d_index, static_cast<std::int64_t>(index.size()), f);
                a.finish();
                sd::release(d_index);
            }
        }
        else if (!r.lo.empty())
        {
            const std::size_t   m    = r.lo.size();
            std::int64_t* const d_lo = sd::allocate<std::int64_t>(m);
            std::int64_t* const d_hi = sd::allocate<std::int64_t>(m);
            sd::copy(d_lo, r.lo.data(), m);
            sd::copy(d_hi, r.hi.data(), m);
            write_spans<S, V><<<1024, block>>>(a.writer(), d_lo, d_hi, static_cast<std::int64_t>(m), f);
            a.template finish<block>(f);
            sd::release(d_lo);
            sd::release(d_hi);
        }
        (void)a.compress();
        images.push_back(a.image());
    }
    sd::release(d_target);
    return images;
}

template <unsigned int S>
std::vector<shrink_array::ArrayImage<double>> replay_in(std::int64_t n, bool zero_variant, const std::vector<Round>& rounds)
{
    return zero_variant ? replay<S, sd::Variant::zero>(n, rounds) : replay<S, sd::Variant::zero_any>(n, rounds);
}

} // namespace

std::vector<shrink_array::ArrayImage<double>>
device_replay(std::int64_t n, unsigned int chunk_size, bool zero_variant, const std::vector<Round>& rounds)
{
    switch (chunk_size)
    {
        case 256:
            return replay_in<256>(n, zero_variant, rounds);
        case 1024:
            return replay_in<1024>(n, zero_variant, rounds);
        default:
            throw std::runtime_error("device_replay: chunk size " + std::to_string(chunk_size) + " not built");
    }
}

} // namespace device_memory
