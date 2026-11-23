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
 * @file device_core_validation.cu
 * @brief The core layer (chunk/internal/host_device.h) on the device: same_value,
 *        identity_less, walk_constant, StaticChunkSize and log2int, computed by a CUDA
 *        kernel and compared with the same calls on the host.
 *
 * @details Built with -DCHUNK_DEVICE_CORE_CHECK=ON (needs a CUDA compiler: nvcc or clang).
 *          The CPU builds never see CUDA; this program checks that the functions marked
 *          CHUNK_HD compile for the device and give the host's answers, signed zeros and
 *          NaN payloads included.
 */

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>

#include "chunk/internal/chunk_size_policy.h" // StaticChunkSize, ChunkElems
#include "chunk/internal/constant_walk.h"     // internal::walk_constant
#include "chunk/internal/pow2.h"              // log2int
#include "chunk/internal/same_value.h"        // same_value, identity_less

namespace sa = shrink_array;

namespace
{

#define DEVICE_CHECK(call)                                                                     \
    do                                                                                         \
    {                                                                                          \
        const cudaError_t e_ = (call);                                                         \
        if (e_ != cudaSuccess)                                                                 \
        {                                                                                      \
            std::printf("%s:%d: %s: %s\n", __FILE__, __LINE__, #call, cudaGetErrorString(e_)); \
            std::exit(2);                                                                      \
        }                                                                                      \
    } while (false)

/// @brief The answers one pair of values gets: same_value both ways, identity_less both ways.
struct PairAnswer
{
    int same;
    int less_ab;
    int less_ba;
};

template <typename T>
__global__ void pairs_kernel(const T* a, const T* b, int n, PairAnswer* out)
{
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n)
    {
        out[i] = PairAnswer{sa::same_value(a[i], b[i]) ? 1 : 0, 0, 0};
        if constexpr (std::is_floating_point_v<T>)
        {
            out[i].less_ab = sa::identity_less(a[i], b[i]) ? 1 : 0;
            out[i].less_ba = sa::identity_less(b[i], a[i]) ? 1 : 0;
        }
    }
}

/// @brief One walk per buffer of @p len elements: (holds, end) from @p from.
template <typename T>
__global__ void walk_kernel(const T* bufs, const T* values, int n_bufs, unsigned len, unsigned from, int* out)
{
    const int k = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (k < n_bufs)
    {
        const sa::internal::ConstantWalk w = sa::internal::walk_constant(bufs + k * len, values[k], from, len);
        out[2 * k]                         = w.holds ? 1 : 0;
        out[2 * k + 1]                     = static_cast<int>(w.end);
    }
}

/// @brief Geometry of element indices on the device: chunk, offset, ChunkElems, log2int.
__global__ void geometry_kernel(const long* index, int n, long* out)
{
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n)
    {
        out[4 * i]     = sa::StaticChunkSize::chunk_of(index[i]);
        out[4 * i + 1] = sa::StaticChunkSize::offset_of(index[i]);
        out[4 * i + 2] = sa::ChunkElems(sa::StaticChunkSize::size()).count();
        out[4 * i + 3] = static_cast<long>(sa::log2int(static_cast<std::size_t>(index[i])));
    }
}

int g_failures = 0;

void expect(bool ok, const char* what)
{
    std::printf("  %-60s %s\n", what, ok ? "ok" : "FAILED");
    g_failures += ok ? 0 : 1;
}

template <typename T>
T* to_device(const std::vector<T>& v)
{
    T* d = nullptr;
    DEVICE_CHECK(cudaMalloc(&d, sizeof(T) * (v.empty() ? 1 : v.size())));
    DEVICE_CHECK(cudaMemcpy(d, v.data(), sizeof(T) * v.size(), cudaMemcpyHostToDevice));
    return d;
}

template <typename T>
std::vector<T> to_host(const T* d, std::size_t n)
{
    std::vector<T> v(n);
    DEVICE_CHECK(cudaMemcpy(v.data(), d, sizeof(T) * n, cudaMemcpyDeviceToHost));
    return v;
}

/// @brief Every pair of a small set of tricky values, on the device and on the host.
template <typename T>
void check_pairs(const char* type_name, const std::vector<T>& values)
{
    std::vector<T> a, b;
    for (const T& x : values)
    {
        for (const T& y : values)
        {
            a.push_back(x);
            b.push_back(y);
        }
    }
    const int   n    = static_cast<int>(a.size());
    T*          da   = to_device(a);
    T*          db   = to_device(b);
    PairAnswer* dout = nullptr;
    DEVICE_CHECK(cudaMalloc(&dout, sizeof(PairAnswer) * n));
    pairs_kernel<<<(n + 127) / 128, 128>>>(da, db, n, dout);
    DEVICE_CHECK(cudaGetLastError());
    const std::vector<PairAnswer> got = to_host(dout, static_cast<std::size_t>(n));
    int                           bad = 0;
    for (int i = 0; i < n; ++i)
    {
        const bool same = sa::same_value(a[i], b[i]);
        bool       ab = false, ba = false;
        if constexpr (std::is_floating_point_v<T>)
        {
            ab = sa::identity_less(a[i], b[i]);
            ba = sa::identity_less(b[i], a[i]);
        }
        bad +=
            (got[i].same != (same ? 1 : 0) || got[i].less_ab != (ab ? 1 : 0) || got[i].less_ba != (ba ? 1 : 0)) ? 1 : 0;
    }
    char what[96];
    std::snprintf(what, sizeof what, "same_value / identity_less, %s: %d pairs, %d differ", type_name, n, bad);
    expect(bad == 0, what);
    (void)cudaFree(da);
    (void)cudaFree(db);
    (void)cudaFree(dout);
}

/// @brief Buffers of @p len elements, each the constant but for one element (or none).
template <typename T>
void check_walks(const char* type_name, T constant, T other)
{
    const unsigned        len = 256;
    std::vector<T>        bufs, values;
    std::vector<unsigned> froms;
    int                   n_bufs = 0;
    for (int spoil = -1; spoil < static_cast<int>(len); spoil += 7)
    {
        for (unsigned j = 0; j < len; ++j)
        {
            bufs.push_back(static_cast<int>(j) == spoil ? other : constant);
        }
        values.push_back(constant);
        ++n_bufs;
    }
    int bad = 0;
    for (unsigned from : {0U, 1U, 63U, 64U, 200U})
    {
        T*   dbufs = to_device(bufs);
        T*   dvals = to_device(values);
        int* dout  = nullptr;
        DEVICE_CHECK(cudaMalloc(&dout, sizeof(int) * 2 * n_bufs));
        walk_kernel<<<(n_bufs + 63) / 64, 64>>>(dbufs, dvals, n_bufs, len, from, dout);
        DEVICE_CHECK(cudaGetLastError());
        const std::vector<int> got = to_host(dout, static_cast<std::size_t>(2 * n_bufs));
        for (int k = 0; k < n_bufs; ++k)
        {
            const sa::internal::ConstantWalk w =
                sa::internal::walk_constant(bufs.data() + k * len, values[k], from, len);
            bad += (got[2 * k] != (w.holds ? 1 : 0) || got[2 * k + 1] != static_cast<int>(w.end)) ? 1 : 0;
        }
        (void)cudaFree(dbufs);
        (void)cudaFree(dvals);
        (void)cudaFree(dout);
    }
    char what[96];
    std::snprintf(what, sizeof what, "walk_constant, %s: %d buffers x 5 starts, %d differ", type_name, n_bufs, bad);
    expect(bad == 0, what);
}

void check_geometry()
{
    std::vector<long> index;
    for (long i = 0; i < 5000; i += 3)
    {
        index.push_back(i);
    }
    index.push_back(std::numeric_limits<int>::max());
    index.push_back(3L << 33);
    const int n    = static_cast<int>(index.size());
    long*     din  = to_device(index);
    long*     dout = nullptr;
    DEVICE_CHECK(cudaMalloc(&dout, sizeof(long) * 4 * n));
    geometry_kernel<<<(n + 127) / 128, 128>>>(din, n, dout);
    DEVICE_CHECK(cudaGetLastError());
    const std::vector<long> got = to_host(dout, static_cast<std::size_t>(4 * n));
    int                     bad = 0;
    for (int i = 0; i < n; ++i)
    {
        bad += (got[4 * i] != sa::StaticChunkSize::chunk_of(index[i]) ||
                got[4 * i + 1] != sa::StaticChunkSize::offset_of(index[i]) ||
                got[4 * i + 2] != static_cast<long>(sa::StaticChunkSize::size()) ||
                got[4 * i + 3] != static_cast<long>(sa::log2int(static_cast<std::size_t>(index[i]))))
                   ? 1
                   : 0;
    }
    char what[96];
    std::snprintf(what, sizeof what, "StaticChunkSize (S = %u), log2int: %d indices, %d differ",
                  sa::StaticChunkSize::size(), n, bad);
    expect(bad == 0, what);
    (void)cudaFree(din);
    (void)cudaFree(dout);
}

double nan_with_payload(std::uint64_t payload)
{
    std::uint64_t bits = 0x7ff8000000000000ULL | payload;
    double        d;
    std::memcpy(&d, &bits, sizeof d);
    return d;
}

} // namespace

int main()
{
    std::printf("device_core_validation: the CHUNK_HD core layer on the device against the host\n");
    const double qnan = std::numeric_limits<double>::quiet_NaN();
    check_pairs<double>("double",
                        {0.0, -0.0, 1.5, -1.5, qnan, nan_with_payload(7), std::numeric_limits<double>::infinity(),
                         -std::numeric_limits<double>::infinity(), std::numeric_limits<double>::denorm_min()});
    check_pairs<float>(
        "float", {0.0F, -0.0F, 2.25F, std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity()});
    check_pairs<int>("int", {0, 1, -1, std::numeric_limits<int>::max(), std::numeric_limits<int>::min()});
    check_pairs<long>("long", {0L, 1L, -1L, 3L << 40});
    check_walks<double>("double, -0.0 in a run of 0.0", 0.0, -0.0);
    check_walks<double>("double, run of NaN", qnan, 1.0);
    check_walks<float>("float", 1.0F, 2.0F);
    check_walks<int>("int", 7, 8);
    check_geometry();
    std::printf("%s: %d failure(s)\n", g_failures == 0 ? "PASSED" : "FAILED", g_failures);
    return g_failures == 0 ? 0 : 1;
}
