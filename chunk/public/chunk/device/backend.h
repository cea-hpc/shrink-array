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
 * @file backend.h
 * @brief The GPU runtime behind chunk/device/: CUDA or HIP, chosen by the compiler, and the
 *        few host calls the device containers make (allocate, free, set, copy, wait), which
 *        throw std::runtime_error on failure.
 *
 * @details Stage 1, experimental (see chunk_desc.h). Only a CUDA or HIP translation unit
 *          includes it: nvcc or clang -x cuda (NVIDIA, warps of 32), hipcc or clang -x hip
 *          (AMD, wavefronts of 64). CHUNK_DEVICE_HIP is defined for the latter.
 */

#if !defined(__CUDACC__) && !defined(__HIP__)
    #error "chunk/device/backend.h: a CUDA or HIP translation unit only"
#endif

#include <cstddef>   // std::size_t
#include <stdexcept> // std::runtime_error
#include <string>    // std::string

#if defined(__HIP__) || defined(__HIP_PLATFORM_AMD__)
    #define CHUNK_DEVICE_HIP 1
    #include <hip/hip_runtime.h> // hipMalloc, hipMemcpy, ...
#else
    #include <cuda_runtime.h> // cudaMalloc, cudaMemcpy, ...
#endif

namespace shrink_array::device
{

#if defined(CHUNK_DEVICE_HIP)
/// Threads per warp (wavefront).
inline constexpr int warp_size = 64;
using Error                    = hipError_t;
#else
/// Threads per warp.
inline constexpr int warp_size = 32;
using Error                    = cudaError_t;
#endif

/// @brief Throws std::runtime_error naming @p what if @p e is not success.
inline void check(Error e, const char* what);

/// @brief @p n objects of @p T in device memory (uninitialised); nullptr for n == 0.
template <typename T>
[[nodiscard]] T* allocate(std::size_t n);

/// @brief Frees what allocate() returned (nullptr: nothing); never throws.
inline void release(void* p) noexcept;

/// @brief Sets @p n objects of @p T at @p p to zero bytes.
template <typename T>
void zero(T* p, std::size_t n);

/// @brief Copies @p n objects of @p T between host and device (either direction).
template <typename T>
void copy(T* dst, const T* src, std::size_t n);

/// @brief Waits for every kernel launched so far; reports a launch or execution error.
inline void synchronize();

// ---------------------------------------------------------------------------

inline void check(Error e, const char* what)
{
#if defined(CHUNK_DEVICE_HIP)
    if (e != hipSuccess)
    {
        throw std::runtime_error(std::string("shrink_array::device: ") + what + ": " + hipGetErrorString(e));
    }
#else
    if (e != cudaSuccess)
    {
        throw std::runtime_error(std::string("shrink_array::device: ") + what + ": " + cudaGetErrorString(e));
    }
#endif
}

template <typename T>
[[nodiscard]] T* allocate(std::size_t n)
{
    void* p = nullptr;
    if (n == 0)
    {
        return nullptr;
    }
#if defined(CHUNK_DEVICE_HIP)
    check(hipMalloc(&p, n * sizeof(T)), "hipMalloc");
#else
    check(cudaMalloc(&p, n * sizeof(T)), "cudaMalloc");
#endif
    return static_cast<T*>(p);
}

inline void release(void* p) noexcept
{
#if defined(CHUNK_DEVICE_HIP)
    (void)hipFree(p);
#else
    (void)cudaFree(p);
#endif
}

template <typename T>
void zero(T* p, std::size_t n)
{
#if defined(CHUNK_DEVICE_HIP)
    check(hipMemset(p, 0, n * sizeof(T)), "hipMemset");
#else
    check(cudaMemset(p, 0, n * sizeof(T)), "cudaMemset");
#endif
}

template <typename T>
void copy(T* dst, const T* src, std::size_t n)
{
#if defined(CHUNK_DEVICE_HIP)
    check(hipMemcpy(dst, src, n * sizeof(T), hipMemcpyDefault), "hipMemcpy");
#else
    check(cudaMemcpy(dst, src, n * sizeof(T), cudaMemcpyDefault), "cudaMemcpy");
#endif
}

inline void synchronize()
{
#if defined(CHUNK_DEVICE_HIP)
    check(hipGetLastError(), "kernel launch");
    check(hipDeviceSynchronize(), "hipDeviceSynchronize");
#else
    check(cudaGetLastError(), "kernel launch");
    check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
#endif
}

} // namespace shrink_array::device
