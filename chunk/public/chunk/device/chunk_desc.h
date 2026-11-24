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
 * @file chunk_desc.h
 * @brief The device format of a chunk (docs/gpu_design.md, 4.1): a pointer to its buffer,
 *        or nullptr and the constant every element of the chunk reads.
 *
 * @details Stage 1 of the GPU backend, experimental: chunk/public/chunk/device/ is not part
 *          of the interface scripts/public_api.txt freezes. Everything here is CHUNK_HD,
 *          so the host builds and reads the same descriptors as the device.
 *
 *          A descriptor is trivially copyable and has no atomic in its type: the write
 *          kernels (stage 1, next) reach `buf` through atomic_ref-style operations on its
 *          address, and the read kernels load it plainly -- one role per kernel
 *          (docs/gpu_design.md, section 3), so nothing writes `buf` while a kernel reads.
 */

#include <cstdint> // std::uintptr_t

#include "chunk/internal/host_device.h" // CHUNK_HD

namespace shrink_array::device
{

/**
 * @brief One chunk on the device.
 * @tparam T element type, trivially copyable
 */
template <typename T>
struct ChunkDesc
{
    /// nullptr: the chunk is compressed and reads `constant`; busy_mark<T>(): a write kernel
    /// is materialising it (never seen by a read kernel); otherwise a buffer of S elements.
    T* buf;
    /// The value of every element while `buf` is nullptr (the pre-fill of a materialisation).
    T constant;
};

/**
 * @brief The value of ChunkDesc::buf while a write kernel holds the claim of the chunk:
 *        a misaligned address, never a buffer (docs/gpu_design.md, 6.2).
 */
template <typename T>
[[nodiscard]] CHUNK_HD inline T* busy_mark() noexcept
{
    return reinterpret_cast<T*>(std::uintptr_t{1});
}

} // namespace shrink_array::device
