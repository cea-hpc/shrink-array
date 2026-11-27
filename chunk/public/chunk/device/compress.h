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
 * @file compress.h
 * @brief The compression pass of a device array (docs/gpu_design.md, section 7): a
 *        materialised chunk whose elements are all one value (same_value) takes that value
 *        as its constant and gives its buffer back to the arena (in the zero variant, only
 *        when that value is T{}; in zero_or_value, when it is T{} or the second value).
 *
 * @details Stage 1, experimental (see chunk_desc.h), from the prototype's compress_warp: one
 *          warp per chunk, grid-stride over the chunks, each lane reading S / warp_size
 *          elements, a ballot deciding. Measured there: 10 GB in 2.9 to 4.1 ms on the GH200
 *          and the MI300A when every chunk folds back (S = 1024: 85% and 102% of the stream
 *          read).
 *
 *          A kernel of its own role (docs/gpu_design.md, section 3): no write kernel runs with
 *          it, so it reads the buffers and writes the descriptors with plain accesses, and it
 *          gives back after DeviceArena::normalise() with no take() meanwhile. The fresh
 *          buffers must be settled before it (Array::compress does so).
 *
 *          Only the elements below n are compared: the tail of the last chunk is never read
 *          (it holds the constant of an earlier materialisation, not necessarily the current
 *          one).
 */

#include <cstdint> // std::int64_t

#include "chunk/device/arena.h"        // ArenaRef
#include "chunk/device/atomics.h"      // fetch_add
#include "chunk/device/chunk_desc.h"   // ChunkDesc, Variant, folds_to, zero_value
#include "chunk/device/warp.h"         // ballot, full_warp, lane_id
#include "chunk/internal/same_value.h" // same_value

namespace shrink_array::device::kernels
{

/// @brief The pass: @p folded receives the number of chunks folded back. Blocks of a
///        multiple of warp_size threads: every lane of a warp takes the same chunks.
///        @p second: zero_or_value's second value (folds_to), ignored by the other variants.
///        @p touched: a byte per chunk, set by the writers since the last pass (Writer::touched);
///        an untouched chunk is skipped -- not uniform at the last pass (it would have folded)
///        and unchanged since -- unless @p all. Read only: the caller clears the bytes after
///        the pass, one memset (a clear per chunk here, a warp barrier and a store each, made
///        the pass 20 % slower where every chunk was touched: device_bench, RTX A1000).
template <typename T, unsigned int S, Variant V>
__global__ void compress_chunks(ChunkDesc<T, V>*    desc,
                                std::int64_t        n,
                                ArenaRef<T, S>      arena,
                                T                   second,
                                unsigned long long* folded,
                                const unsigned char* touched,
                                bool                all)
{
    const std::int64_t n_chunks = (n + S - 1) / S;
    const int          lane     = lane_id();
    const std::int64_t warps    = static_cast<std::int64_t>(gridDim.x) * (blockDim.x / warp_size);
    const std::int64_t first    = (blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x) / warp_size;
    unsigned long long mine     = 0;
    for (std::int64_t c = first; c < n_chunks; c += warps)
    {
        // The byte and the descriptor loaded together: a chunk every writer touched (the
        // common case of a pass after a full write) waits for one load, not two in a row.
        ChunkDesc<T, V>&    d   = desc[c];
        const unsigned char tc  = all ? 1 : touched[c];
        T* const            buf = d.buf;
        if (buf == nullptr || tc == 0)
        {
            continue; // uniform per warp; untouched: not uniform at the last pass, unchanged since
        }
        const std::int64_t valid = (c + 1) * S <= n ? S : n - c * S;
        // zero: the chunk folds only to T{}, so every element is compared with it.
        const T head = V == Variant::zero ? zero_value<T>() : buf[0];
        // zero_or_value: a head that is neither value cannot fold (every lane read it).
        bool uniform = folds_to<V>(head, second);
        for (std::int64_t k = lane; k < valid && uniform; k += warp_size)
        {
            uniform = same_value(buf[k], head);
        }
        if (ballot(full_warp, !uniform) == 0 && lane == 0)
        {
            if constexpr (V != Variant::zero)
            {
                d.constant = head;
            }
            d.buf = nullptr;
            arena.give_back(buf);
            ++mine;
        }
    }
    if (lane == 0 && mine != 0)
    {
        fetch_add(folded, mine);
    }
}

} // namespace shrink_array::device::kernels
