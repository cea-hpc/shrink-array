// Copyright 2026 CEA (www.cea.fr)
// SPDX-License-Identifier: Apache-2.0

/**
 * @file device_intrinsics.h
 * @brief The dozen device primitives the GPU prototype needs, mapped on CUDA and HIP
 *        (docs/gpu_design.md section 12.1): warp ballot and shuffle, device-scope
 *        atomics, the release / acquire fences around the publication CAS.
 *
 * @details Stage 0 of the GPU plan (docs/gpu_design.md section 13), out of the library's
 *          tree. Everything else in the prototype is written against these names only,
 *          so that the same source builds with nvcc or clang-CUDA (NVIDIA, 32 lanes) and
 *          with hipcc (AMD, 64 lanes).
 */

#pragma once

#include <cstdint>

#if defined(__HIP_PLATFORM_AMD__)
    #include <hip/hip_runtime.h>
#else
    #include <cuda_runtime.h>
#endif

namespace shrink_array::device
{

#if defined(__HIP_PLATFORM_AMD__)
inline constexpr int warp_size = 64;
using LaneMask                 = std::uint64_t;
#else
inline constexpr int warp_size = 32;
using LaneMask                 = std::uint32_t;
#endif

/// @brief This thread's lane in its warp.
__device__ inline int lane_id() noexcept
{
    return static_cast<int>(threadIdx.x) % warp_size;
}

/// @brief The lanes of the warp that execute this call.
__device__ inline LaneMask active_lanes() noexcept
{
#if defined(__HIP_PLATFORM_AMD__)
    return __ballot(1);
#else
    return __activemask();
#endif
}

/// @brief The lanes of @p mask for which @p pred holds. Every lane of @p mask calls it.
__device__ inline LaneMask ballot(LaneMask mask, bool pred) noexcept
{
#if defined(__HIP_PLATFORM_AMD__)
    return __ballot(pred) & mask;
#else
    return __ballot_sync(mask, pred);
#endif
}

/// @brief @p value as lane @p src has it. Every lane of @p mask calls it.
template <typename V>
__device__ inline V shuffle(LaneMask mask, V value, int src) noexcept
{
#if defined(__HIP_PLATFORM_AMD__)
    (void)mask;
    return __shfl(value, src);
#else
    return __shfl_sync(mask, value, src);
#endif
}

/// @brief Pointer shuffle (the 64-bit buffer address of the elected lane).
template <typename P>
__device__ inline P* shuffle_ptr(LaneMask mask, P* ptr, int src) noexcept
{
    const auto bits = reinterpret_cast<unsigned long long>(ptr);
    return reinterpret_cast<P*>(shuffle(mask, bits, src));
}

/// @brief Lowest lane of a non-empty @p mask.
__device__ inline int first_lane(LaneMask mask) noexcept
{
#if defined(__HIP_PLATFORM_AMD__)
    return __ffsll(static_cast<unsigned long long>(mask)) - 1;
#else
    return __ffs(static_cast<int>(mask)) - 1;
#endif
}

/// @brief Number of lanes in @p mask.
__device__ inline int popcount(LaneMask mask) noexcept
{
#if defined(__HIP_PLATFORM_AMD__)
    return __popcll(static_cast<unsigned long long>(mask));
#else
    return __popc(mask);
#endif
}

/// @brief Wait for the lanes of @p mask (their earlier stores visible to each other).
__device__ inline void sync_lanes(LaneMask mask) noexcept
{
#if defined(__HIP_PLATFORM_AMD__)
    (void)mask;
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "wavefront");
#else
    __syncwarp(mask);
#endif
}

/// @brief Device-scope fence: what this thread wrote before is visible device-wide
///        before what it writes after (the release side of the publication CAS).
__device__ inline void fence_device() noexcept
{
    __threadfence();
}

/// @brief Device-scope CAS on a pointer slot: returns the value found (== @p expected on success).
template <typename P>
__device__ inline P* cas_ptr(P** slot, P* expected, P* desired) noexcept
{
    auto* s = reinterpret_cast<unsigned long long*>(slot);
    return reinterpret_cast<P*>(
        atomicCAS(s, reinterpret_cast<unsigned long long>(expected), reinterpret_cast<unsigned long long>(desired)));
}

/// @brief Device-scope exchange of a pointer slot: returns the former value.
template <typename P>
__device__ inline P* exchange_ptr(P** slot, P* desired) noexcept
{
    auto* s = reinterpret_cast<unsigned long long*>(slot);
    return reinterpret_cast<P*>(atomicExch(s, reinterpret_cast<unsigned long long>(desired)));
}

/// @brief Relaxed device-scope load of a pointer slot (no tearing, no caching across the kernel).
template <typename P>
__device__ inline P* load_ptr(P* const* slot) noexcept
{
    return *reinterpret_cast<P* const volatile*>(slot);
}

/// @brief Relaxed device-scope atomic load of a pointer slot, with no compiler barrier: the
///        loads of successive writes can be in flight together (the write path's read of
///        `buf` when ordered by its address dependency, see View::assign).
template <typename P>
__device__ inline P* load_ptr_relaxed(P* const* slot) noexcept
{
#if defined(__HIP_PLATFORM_AMD__)
    return __hip_atomic_load(const_cast<P**>(slot), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    unsigned long long bits;
    asm volatile("ld.relaxed.gpu.global.u64 %0, [%1];" : "=l"(bits) : "l"(slot));
    return reinterpret_cast<P*>(bits);
#else
    return load_ptr(slot);
#endif
}

/// @brief Relaxed load of a signed 64-bit counter (a look before an atomic on it).
__device__ inline long long load_relaxed(const long long* counter) noexcept
{
    return *reinterpret_cast<const volatile long long*>(counter);
}

/// @brief Device-scope acquire load of a pointer slot: a buffer published by the release
///        CAS of 6.2 is seen with its whole pre-fill (the write path's read of `buf`).
template <typename P>
__device__ inline P* load_ptr_acquire(P* const* slot) noexcept
{
#if defined(__HIP_PLATFORM_AMD__)
    return __hip_atomic_load(const_cast<P**>(slot), __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    unsigned long long bits;
    asm volatile("ld.acquire.gpu.global.u64 %0, [%1];" : "=l"(bits) : "l"(slot) : "memory");
    return reinterpret_cast<P*>(bits);
#else
    P* p = load_ptr(slot);
    __threadfence();
    return p;
#endif
}

/// @brief A short pause in a spin loop (the claim wait of materialise()): lets the warp
///        being waited for run instead of hammering its descriptor.
__device__ inline void backoff() noexcept
{
#if defined(__HIP_PLATFORM_AMD__)
    __builtin_amdgcn_s_sleep(2);
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    __nanosleep(64);
#endif
}

/// @brief atomicAdd on a 64-bit counter, device scope; returns the former value.
__device__ inline unsigned long long fetch_add(unsigned long long* counter, unsigned long long v) noexcept
{
    return atomicAdd(counter, v);
}

/// @brief atomicAdd on a signed 64-bit counter (the free stack's top, which may go below 0).
__device__ inline long long fetch_add(long long* counter, long long v) noexcept
{
    return static_cast<long long>(
        atomicAdd(reinterpret_cast<unsigned long long*>(counter), static_cast<unsigned long long>(v)));
}

} // namespace shrink_array::device
