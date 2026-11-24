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
 * @file warp.h
 * @brief The warp primitives of chunk/device/ on CUDA (32 lanes) and HIP (64 lanes): the
 *        lane, the active lanes, ballot, shuffle, the grouping of lanes by key.
 *
 * @details Stage 1, experimental (see chunk_desc.h). Every lane of the mask calls a ballot
 *          (warp-uniform control flow at the call).
 */

#include <cstdint> // std::uint32_t, std::uint64_t

#include "chunk/device/backend.h" // warp_size, CUDA or HIP

namespace shrink_array::device
{

#if defined(CHUNK_DEVICE_HIP)
/// One bit per lane.
using LaneMask = std::uint64_t;
#else
/// One bit per lane.
using LaneMask = std::uint32_t;
#endif

/// Every lane of a warp: the mask of a ballot that every lane reaches, after a loop whose
/// lanes may have left at different iterations (active_lanes() may then see only some).
inline constexpr LaneMask full_warp = ~LaneMask{0};

/// @brief This thread's lane in its warp.
__device__ inline int lane_id() noexcept
{
    return static_cast<int>(threadIdx.x) % warp_size;
}

/// @brief The lanes of the warp that execute this call.
__device__ inline LaneMask active_lanes() noexcept
{
#if defined(CHUNK_DEVICE_HIP)
    return __ballot(1);
#else
    return __activemask();
#endif
}

/// @brief The lanes of @p mask for which @p pred holds.
__device__ inline LaneMask ballot(LaneMask mask, bool pred) noexcept
{
#if defined(CHUNK_DEVICE_HIP)
    return __ballot(pred) & mask;
#else
    return __ballot_sync(mask, pred);
#endif
}

/// @brief @p value as lane @p src has it. Every lane of @p mask calls it.
template <typename V>
__device__ inline V shuffle(LaneMask mask, V value, int src) noexcept
{
#if defined(CHUNK_DEVICE_HIP)
    (void)mask;
    return __shfl(value, src);
#else
    return __shfl_sync(mask, value, src);
#endif
}

/// @brief A pointer as lane @p src has it (the 64-bit address, through an integer).
template <typename P>
__device__ inline P* shuffle_ptr(LaneMask mask, P* ptr, int src) noexcept
{
    const auto bits = reinterpret_cast<unsigned long long>(ptr);
    return reinterpret_cast<P*>(shuffle(mask, bits, src));
}

/// @brief The lowest lane of a non-empty @p mask.
__device__ inline int first_lane(LaneMask mask) noexcept
{
#if defined(CHUNK_DEVICE_HIP)
    return __ffsll(static_cast<unsigned long long>(mask)) - 1;
#else
    return __ffs(static_cast<int>(mask)) - 1;
#endif
}

/**
 * @brief The lanes of @p active grouped by @p key: each lane gets its group (the lanes of
 *        @p active with its key) and the group's leader (its lowest lane). Every lane of
 *        @p active calls it.
 * @details Every lane goes through the loop as many times as the others (`pending` is the same
 *          on every lane), so each ballot and shuffle has the one mask @p active. The
 *          prototype's first version let a lane leave once its own group was found, and the
 *          later shuffles ran with masks that changed under it.
 */
template <typename K>
__device__ inline void group_by_key(K key, LaneMask active, LaneMask& group, int& leader) noexcept
{
    const int lane    = lane_id();
    LaneMask  pending = active;
    group             = 0;
    leader            = lane;
    while (pending != 0)
    {
        const int      l = first_lane(pending);
        const K        k = shuffle(active, key, l);
        const LaneMask g = ballot(active, ((pending >> lane) & 1U) != 0 && key == k);
        if (((g >> lane) & 1U) != 0)
        {
            group  = g;
            leader = l;
        }
        pending &= ~g;
    }
}

/// @brief Waits for the lanes of @p mask (their earlier stores visible to each other).
__device__ inline void sync_lanes(LaneMask mask) noexcept
{
#if defined(CHUNK_DEVICE_HIP)
    (void)mask;
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "wavefront");
#else
    __syncwarp(mask);
#endif
}

} // namespace shrink_array::device
