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
 * @file atomics.h
 * @brief The device atomics of chunk/device/: relaxed, device (agent) scope, on the address
 *        of a plain field.
 *
 * @details Relaxed only, on purpose (docs/gpu_design.md, section 13, MI300A): on AMD an
 *          agent-scope acquire invalidates the L2 of an XCD and a release writes it back,
 *          which made every materialisation 15x to 600x a flat write. The device containers
 *          order their phases by kernel boundaries, not by fences; within a kernel their
 *          threads write disjoint memory (docs/gpu_design.md, "relaxed publication").
 */

#include "chunk/device/backend.h" // CUDA or HIP

namespace shrink_array::device
{

/// @brief Adds @p v to @p *counter; returns the former value.
__device__ inline unsigned long long fetch_add(unsigned long long* counter, unsigned long long v) noexcept
{
    return atomicAdd(counter, v);
}

/// @brief Adds @p v (possibly negative) to @p *counter; returns the former value.
__device__ inline long long fetch_add(long long* counter, long long v) noexcept
{
    return static_cast<long long>(
        atomicAdd(reinterpret_cast<unsigned long long*>(counter), static_cast<unsigned long long>(v)));
}

/// @brief Ors @p bits into @p *word; returns the former value.
__device__ inline unsigned fetch_or(unsigned* word, unsigned bits) noexcept
{
#if defined(CHUNK_DEVICE_HIP)
    return __hip_atomic_fetch_or(word, bits, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#else
    return atomicOr(word, bits);
#endif
}

/// @brief Compare-and-swap of a pointer slot; returns the value found (== @p expected on success).
template <typename P>
__device__ inline P* cas_ptr(P** slot, P* expected, P* desired) noexcept
{
    auto* s = reinterpret_cast<unsigned long long*>(slot);
    return reinterpret_cast<P*>(
        atomicCAS(s, reinterpret_cast<unsigned long long>(expected), reinterpret_cast<unsigned long long>(desired)));
}

/// @brief Exchange of a pointer slot; returns the former value.
template <typename P>
__device__ inline P* exchange_ptr(P** slot, P* desired) noexcept
{
    auto* s = reinterpret_cast<unsigned long long*>(slot);
    return reinterpret_cast<P*>(atomicExch(s, reinterpret_cast<unsigned long long>(desired)));
}

/// @brief Atomic load of a pointer slot.
template <typename P>
__device__ inline P* load_ptr(P* const* slot) noexcept
{
#if defined(CHUNK_DEVICE_HIP)
    return __hip_atomic_load(const_cast<P**>(slot), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    unsigned long long bits = 0;
    asm volatile("ld.relaxed.gpu.global.u64 %0, [%1];" : "=l"(bits) : "l"(slot));
    return reinterpret_cast<P*>(bits);
#else
    return *reinterpret_cast<P* const volatile*>(slot);
#endif
}

/// @brief A relaxed atomic store (a flag every writer may set: no read-modify-write).
__device__ inline void store_relaxed(unsigned long long* p, unsigned long long v) noexcept
{
#if defined(CHUNK_DEVICE_HIP)
    __hip_atomic_store(p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    asm volatile("st.relaxed.gpu.global.u64 [%0], %1;" : : "l"(p), "l"(v) : "memory");
#else
    *reinterpret_cast<volatile unsigned long long*>(p) = v;
#endif
}

/// @brief Atomic load of a 32-bit word (a bit of the `written` bitmap).
__device__ inline unsigned load(const unsigned* word) noexcept
{
#if defined(CHUNK_DEVICE_HIP)
    return __hip_atomic_load(const_cast<unsigned*>(word), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#else
    return *reinterpret_cast<const volatile unsigned*>(word);
#endif
}

/// @brief Atomic load of a 64-bit counter (a look before an atomic on it).
template <typename C>
__device__ inline C load(const C* counter) noexcept
{
    static_assert(sizeof(C) == 8, "load: a 64-bit counter");
#if defined(CHUNK_DEVICE_HIP)
    return __hip_atomic_load(const_cast<C*>(counter), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#else
    return *reinterpret_cast<const volatile C*>(counter);
#endif
}

/// @brief A short pause in a polling loop (a claimed descriptor): lets the claim's holder run.
__device__ inline void backoff() noexcept
{
#if defined(CHUNK_DEVICE_HIP)
    __builtin_amdgcn_s_sleep(2);
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    __nanosleep(64);
#endif
}

} // namespace shrink_array::device
