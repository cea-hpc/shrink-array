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

#include <array>       // std::array
#include <cstddef>     // std::size_t
#include <cstdint>     // std::uintptr_t
#include <type_traits> // std::is_arithmetic_v

#include "chunk/internal/host_device.h" // CHUNK_HD
#include "chunk/internal/same_value.h"  // internal::is_std_complex, same_value

namespace shrink_array::device
{

/**
 * @brief The variant of a device array (docs/gpu_design.md, section 8): what a compressed
 *        chunk may read.
 */
enum class Variant : unsigned char
{
    zero,          ///< T{} only (8.1): the descriptor is its pointer, the pass folds only to T{}.
    zero_any,      ///< Any value (8.3): the descriptor carries it.
    zero_or_value, ///< T{} or the array's second value (8.2): the descriptor carries the value,
                   ///< the pass and the whole-chunk write fold only to those two.
};

/**
 * @brief One chunk on the device.
 * @tparam T element type, trivially copyable
 * @tparam V the variant: zero_any and zero_or_value carry the constant, zero has none (it is T{})
 */
template <typename T, Variant V = Variant::zero_any>
struct ChunkDesc
{
    /// nullptr: the chunk is compressed and reads `constant`; busy_mark<T>(): a write kernel
    /// is materialising it; tag_fresh(b): buffer b, published by a chunk-aligned write of
    /// part of the chunk and not settled yet (block_write.h). Neither is seen by a read
    /// kernel. Otherwise a buffer of S elements.
    T* buf;
    /// The value of every element while `buf` is nullptr (the pre-fill of a materialisation).
    T constant;
};

/// @brief The zero variant's descriptor: the pointer alone, a compressed chunk reads T{}
///        (the size of the host zero variant's chunk, half a zero_any one for a double).
template <typename T>
struct ChunkDesc<T, Variant::zero>
{
    T* buf; ///< as ChunkDesc::buf
};

/// Whether T{} is zero bytes and a host constructor to nvcc: std::complex, a std::array of an
/// arithmetic type (zero_value()).
template <typename T>
inline constexpr bool zero_bytes_v = internal::is_std_complex<T>::value;
template <typename E, std::size_t N>
inline constexpr bool zero_bytes_v<std::array<E, N>> = std::is_arithmetic_v<E>;

/**
 * @brief T{}, in host and device code.
 * @details The constructors of std::complex and std::array are constexpr host functions to
 *          nvcc (without --expt-relaxed-constexpr): device code builds those from zero bytes,
 *          which is their value-initialisation when their parts are arithmetic.
 */
template <typename T>
[[nodiscard]] CHUNK_HD inline T zero_value() noexcept
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    if constexpr (zero_bytes_v<T>)
    {
        struct Bytes
        {
            unsigned char b[sizeof(T)];
        };
        return __builtin_bit_cast(T, Bytes{});
    }
    else
    {
        return T{};
    }
#else
    return T{};
#endif
}

/**
 * @brief Whether a chunk of variant @p V may fold to @p value, by the pass (compress.h) or a
 *        write of the whole chunk (Writer::fold_whole): any value in zero_any, T{} in zero,
 *        T{} or @p second (the array's second value, T{} while none is declared) in
 *        zero_or_value.
 */
template <Variant V, typename T>
[[nodiscard]] CHUNK_HD inline bool folds_to(const T& value, const T& second) noexcept
{
    if constexpr (V == Variant::zero_any)
    {
        (void)value;
        (void)second;
        return true;
    }
    else if constexpr (V == Variant::zero)
    {
        (void)second;
        return same_value(value, zero_value<T>());
    }
    else
    {
        return same_value(value, zero_value<T>()) || same_value(value, second);
    }
}

/// @brief What every element of chunk @p d reads while it holds no buffer.
template <typename T, Variant V>
[[nodiscard]] CHUNK_HD inline T constant_of(const ChunkDesc<T, V>& d) noexcept
{
    if constexpr (V == Variant::zero)
    {
        (void)d;
        return zero_value<T>();
    }
    else
    {
        return d.constant;
    }
}

/**
 * @brief The value of ChunkDesc::buf while a write kernel holds the claim of the chunk:
 *        a misaligned address, never a buffer (docs/gpu_design.md, 6.2).
 */
template <typename T>
[[nodiscard]] CHUNK_HD inline T* busy_mark() noexcept
{
    return reinterpret_cast<T*>(std::uintptr_t{1});
}

/// The bit of ChunkDesc::buf that marks a fresh buffer: its elements are valid only where the
/// chunk's `written` bitmap says so, the others take the constant when it is settled. Bit 1:
/// a buffer is aligned on its elements (4 bytes at least) and busy_mark() is bit 0.
inline constexpr std::uintptr_t fresh_tag = 2;

/// @brief Whether @p p is a fresh buffer (tag_fresh()).
template <typename T>
[[nodiscard]] CHUNK_HD inline bool is_fresh(const T* p) noexcept
{
    return (reinterpret_cast<std::uintptr_t>(p) & fresh_tag) != 0;
}

/// @brief Buffer @p p marked fresh.
template <typename T>
[[nodiscard]] CHUNK_HD inline T* tag_fresh(T* p) noexcept
{
    return reinterpret_cast<T*>(reinterpret_cast<std::uintptr_t>(p) | fresh_tag);
}

/// The bit of ChunkDesc::buf that marks a filled buffer (CUDA's Writer::assign): published
/// in this phase with every element written (the constant, then the writes), by a release
/// exchange that its writers acquire; it needs no `written` bitmap, the settle only removes
/// the mark. Bit 2: a buffer is S * sizeof(T) bytes into a 256-byte aligned arena, a
/// multiple of 8.
inline constexpr std::uintptr_t filled_tag = 4;

/// @brief Whether @p p is a filled buffer (tag_filled()).
template <typename T>
[[nodiscard]] CHUNK_HD inline bool is_filled(const T* p) noexcept
{
    return (reinterpret_cast<std::uintptr_t>(p) & filled_tag) != 0;
}

/// @brief Buffer @p p marked filled.
template <typename T>
[[nodiscard]] CHUNK_HD inline T* tag_filled(T* p) noexcept
{
    return reinterpret_cast<T*>(reinterpret_cast<std::uintptr_t>(p) | filled_tag);
}

/// @brief The buffer behind a fresh or filled mark (identity on an unmarked one).
template <typename T>
[[nodiscard]] CHUNK_HD inline T* untag(T* p) noexcept
{
    return reinterpret_cast<T*>(reinterpret_cast<std::uintptr_t>(p) & ~(fresh_tag | filled_tag));
}

} // namespace shrink_array::device
