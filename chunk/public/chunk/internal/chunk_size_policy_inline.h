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
#include "chunk/internal/chunk_size_policy.h" // the class these bodies belong to

namespace shrink_array
{

// ---------------------------------------------------------------------------
// shrink_array::DynamicChunkSize
// ---------------------------------------------------------------------------

template <typename I>
[[nodiscard]] I DynamicChunkSize::chunk_of(I index) const noexcept
{
    // A power of two keeps the shift: the size of an array changes
    // rarely, so this branch is always predicted.
    if (pow2_) [[likely]]
    {
        return static_cast<I>(static_cast<std::make_unsigned_t<I>>(index) >> decalage_);
    }
    if constexpr (sizeof(I) <= 4)
    {
        const auto idx64 = static_cast<std::uint64_t>(static_cast<std::make_unsigned_t<I>>(index));
        return static_cast<I>((static_cast<unsigned __int128>(m32_) * idx64) >> 64);
    }
    else
    {
        const auto idx64    = static_cast<std::uint64_t>(static_cast<std::make_unsigned_t<I>>(index));
        const auto quotient = static_cast<std::uint64_t>((static_cast<unsigned __int128>(magic_) * idx64) >> 64);
        return static_cast<I>((quotient + ((idx64 - quotient) >> 1)) >> magic_shift_);
    }
}

template <typename I>
[[nodiscard]] I DynamicChunkSize::offset_of(I index) const noexcept
{
    if (pow2_) [[likely]]
    {
        return static_cast<I>(static_cast<std::make_unsigned_t<I>>(index) & modulo_);
    }
    if constexpr (sizeof(I) <= 4)
    {
        // Lemire's direct remainder: the fraction left by the quotient's
        // multiplication, scaled by the divisor.
        const std::uint64_t frac = m32_ * static_cast<std::uint64_t>(static_cast<std::make_unsigned_t<I>>(index));
        return static_cast<I>((static_cast<unsigned __int128>(frac) * size_) >> 64);
    }
    else
    {
        return static_cast<I>(index - (chunk_of(index) * static_cast<I>(size_)));
    }
}

inline void DynamicChunkSize::set_size(unsigned int size) noexcept
{
    // Last net: rechunk_to() and set_chunk_size() refuse such a size
    // before reaching here (is_valid_chunk_size).
    if constexpr (build_config::any_check)
    {
        if (!is_valid_chunk_size(size)) [[unlikely]]
        {
            CHUNK_INVARIANT_FAILED("ChunkSizePolicy::set_size({}): a chunk size must be a power of two or a "
                                   "multiple of 64",
                                   size);
        }
    }
    size_     = size;
    decalage_ = static_cast<unsigned int>(log2int(static_cast<std::size_t>(size)));
    modulo_   = size - 1u;
    set_divider(size);
    g_active_chunk_size = size; // keep the allocator-facing runtime size in sync
}

inline void DynamicChunkSize::set_divider(unsigned int size) noexcept
{
    pow2_             = std::has_single_bit(size); // NOT is_valid_chunk_size: a multiple of 64 is valid too
    m32_              = UINT64_MAX / size + 1u;
    unsigned int log2 = 0;
    while ((1ull << log2) < size)
    {
        ++log2;
    }
    magic_ = static_cast<std::uint64_t>((static_cast<unsigned __int128>((1ull << log2) - size) << 64) / size) + 1u;
    magic_shift_ = log2 - 1u;
}

} // namespace shrink_array
