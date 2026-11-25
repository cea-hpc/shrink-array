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
 * @file view.h
 * @brief What a kernel gets of a device array: a trivially copyable view over its chunk
 *        descriptors, passed by value. This first part is the read path
 *        (docs/gpu_design.md, section 5) and the uniformity test of the compression pass
 *        (section 7); the writes come next.
 *
 * @details Stage 1, experimental (see chunk_desc.h). CHUNK_HD throughout: a host program
 *          reads a View over host descriptors exactly as a kernel reads one over device
 *          descriptors, which is how device_core_validation compares them.
 *
 *          A read is `d.buf ? d.buf[i & mask] : d.constant` (T{} for the zero variant): one descriptor load, no
 *          barrier, no atomic. It is only valid in a read kernel (or on the host between
 *          phases): a write kernel may publish `buf` while it runs.
 */

#include <cstdint>     // std::int64_t
#include <type_traits> // std::is_trivially_copyable_v

#include "chunk/device/chunk_desc.h"      // ChunkDesc, Variant, constant_of
#include "chunk/internal/constant_walk.h" // internal::walk_constant
#include "chunk/internal/host_device.h"   // CHUNK_HD
#include "chunk/internal/pow2.h"          // log2int

namespace shrink_array::device
{

/**
 * @brief The descriptors of an array and its element count, at chunk size @p S.
 * @tparam T element type, trivially copyable
 * @tparam S chunk size, a power of two (the GPU geometry is S = 1024, docs/gpu_design.md 13)
 * @tparam V the variant (chunk_desc.h)
 */
template <typename T, unsigned int S, Variant V = Variant::zero_any>
struct View
{
    static_assert(S >= 1 && (S & (S - 1)) == 0, "View: S is a power of two");
    static_assert(std::is_trivially_copyable_v<T>, "View: the elements are copied as bytes between host and device");

    static constexpr unsigned int chunk_size = S;
    static constexpr unsigned int shift      = static_cast<unsigned int>(log2int(S));
    static constexpr std::int64_t mask       = static_cast<std::int64_t>(S) - 1;

    const ChunkDesc<T, V>* desc = nullptr; ///< n_chunks() descriptors
    std::int64_t           n    = 0;       ///< elements

    /// @brief Elements.
    [[nodiscard]] CHUNK_HD std::int64_t size() const noexcept
    {
        return n;
    }

    /// @brief Chunks: the last one may be partial (its buffer still holds S elements).
    [[nodiscard]] CHUNK_HD std::int64_t n_chunks() const noexcept
    {
        return (n + mask) >> shift;
    }

    /// @brief Element @p i (0 <= i < size()), in a read kernel.
    [[nodiscard]] CHUNK_HD T operator()(std::int64_t i) const noexcept
    {
        const ChunkDesc<T, V>& d = desc[i >> shift];
        return d.buf != nullptr ? d.buf[i & mask] : constant_of(d);
    }

    /// @brief Whether chunk @p c holds a buffer.
    [[nodiscard]] CHUNK_HD bool materialised(std::int64_t c) const noexcept
    {
        return desc[c].buf != nullptr;
    }

    /// @brief Whether chunk @p c reads one value only (same_value()), and that value in
    ///        @p value: the test of the compression pass. A materialised chunk is scanned
    ///        over its whole buffer, the tail past size() included (it holds the pre-fill).
    [[nodiscard]] CHUNK_HD bool uniform(std::int64_t c, T& value) const noexcept;
};

template <typename T, unsigned int S, Variant V>
[[nodiscard]] CHUNK_HD bool View<T, S, V>::uniform(std::int64_t c, T& value) const noexcept
{
    const ChunkDesc<T, V>& d = desc[c];
    if (d.buf == nullptr)
    {
        value = constant_of(d);
        return true;
    }
    value = d.buf[0];
    return internal::walk_constant(d.buf, value, 1U, S).holds;
}

} // namespace shrink_array::device
