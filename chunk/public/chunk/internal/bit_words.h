/*
 * Copyright 2024-2026 CEA (www.cea.fr)
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
 * @file bit_words.h
 * @brief Bit-packed storage kernel of ArrayChunkBits: n bools in
 *        ceil(n / 64) 64-bit words, bit k of element i in word i / 64.
 *
 * @details Plain functions over a word pointer, no chunk, no array: the chunk
 *          and the proxy of the packed container are built on them.
 *
 *          Concurrency. Two elements of one word are two bits of the SAME
 *          memory location: a plain read-modify-write of one would erase a
 *          concurrent write of the other. Every write that may run
 *          concurrently with another goes through set_bit_atomic() -- a
 *          fetch_or / fetch_and on a std::atomic_ref of the word -- and every
 *          such read through get_bit_relaxed(). Writing distinct elements from
 *          distinct threads is then race-free, as it is for ArrayChunk2<bool>
 *          (one byte per bool). This is the difference with std::vector<bool>,
 *          where writing two elements of one word from two threads is a race.
 *          The plain versions (get_bit, set_bit, fill_bits) are for a buffer
 *          no other thread can reach: a fresh chunk before publication, a
 *          scan at a quiescent point.
 *
 *          Tail. The bits past n in the last word are kept at 0 by fill_bits()
 *          and never written by set_bit*(); scan_words() masks them anyway, so
 *          a buffer whose tail holds garbage still classifies right.
 */

#include <atomic>
#include <bit>     // std::popcount
#include <cstddef> // std::size_t
#include <cstdint> // std::uint64_t
#include <cstring> // std::memset

namespace shrink_array::bits
{

using word_t = std::uint64_t;

inline constexpr std::size_t word_bits  = 64;
inline constexpr word_t      word_ones  = ~word_t{0};
inline constexpr std::size_t word_bytes = sizeof(word_t);

/// @brief Words holding @p n bits.
[[nodiscard]] constexpr std::size_t words_for(std::size_t n) noexcept
{
    return (n + word_bits - 1) / word_bits;
}

/// @brief Bytes of storage for @p n bits: whole words, at least one.
[[nodiscard]] constexpr std::size_t bytes_for(std::size_t n) noexcept
{
    const std::size_t w = words_for(n);
    return (w > 0 ? w : 1) * word_bytes;
}

/// @brief Mask of bit @p i inside its word.
[[nodiscard]] constexpr word_t bit_mask(std::size_t i) noexcept
{
    return word_t{1} << (i % word_bits);
}

/// @brief Mask of the valid bits of the last word of an @p n-bit buffer
///        (all ones when n is a multiple of 64).
[[nodiscard]] constexpr word_t tail_mask(std::size_t n) noexcept
{
    const std::size_t r = n % word_bits;
    return r == 0 ? word_ones : (word_t{1} << r) - 1;
}

/// @brief Read bit @p i. The buffer must not be written concurrently.
[[nodiscard]] inline bool get_bit(const word_t* w, std::size_t i) noexcept
{
    return (w[i / word_bits] & bit_mask(i)) != 0;
}

/// @brief Read bit @p i while other bits of its word may be written.
[[nodiscard]] inline bool get_bit_relaxed(const word_t* w, std::size_t i) noexcept
{
    // atomic_ref needs a non-const object; the load does not write.
    auto& word = const_cast<word_t&>(w[i / word_bits]);
    return (std::atomic_ref<word_t>(word).load(std::memory_order_relaxed) & bit_mask(i)) != 0;
}

/// @brief Write bit @p i. The buffer must not be reached by another thread.
inline void set_bit(word_t* w, std::size_t i, bool v) noexcept
{
    if (v)
        w[i / word_bits] |= bit_mask(i);
    else
        w[i / word_bits] &= ~bit_mask(i);
}

/**
 * @brief Write bit @p i with an atomic read-modify-write of its word.
 * @details Relaxed: it only has to keep the other 63 bits of the word, which
 *          another thread may be writing. Ordering against the compressor
 *          belongs to the write protocol around it (the Dekker handshake of
 *          the proxy), as for any element store.
 * @return the previous value of the bit.
 */
inline bool set_bit_atomic(word_t* w, std::size_t i, bool v) noexcept
{
    std::atomic_ref<word_t> word(w[i / word_bits]);
    const word_t            m = bit_mask(i);
    const word_t old = v ? word.fetch_or(m, std::memory_order_relaxed) : word.fetch_and(~m, std::memory_order_relaxed);
    return (old & m) != 0;
}

/// @brief Set the @p n bits to @p v and the tail of the last word to 0.
inline void fill_bits(word_t* w, std::size_t n, bool v) noexcept
{
    const std::size_t nw = words_for(n);
    if (nw == 0)
        return;
    std::memset(w, v ? 0xFF : 0x00, nw * word_bytes);
    if (v)
        w[nw - 1] &= tail_mask(n);
}

/// @brief What a scan of a buffer found.
enum class Uniform
{
    all_false, ///< every bit 0 -- the first constant
    all_true,  ///< every bit 1 -- the second constant
    mixed      ///< both values present
};

/**
 * @brief Classify the @p n bits of a buffer, one word at a time.
 * @details A chunk of 256 bools is 4 words: 4 compares where the byte layout
 *          makes 256. The tail of the last word is masked out. @p n = 0 is
 *          all_false, like an empty chunk.
 */
[[nodiscard]] inline Uniform scan_words(const word_t* w, std::size_t n) noexcept
{
    const std::size_t nw = words_for(n);
    if (nw == 0)
        return Uniform::all_false;
    const word_t last = w[nw - 1] & tail_mask(n);
    word_t       any  = last;                    // OR: some 1 bit
    word_t       all  = last | ~tail_mask(n);    // AND: every bit 1 (tail forced to 1)
    for (std::size_t k = 0; k + 1 < nw; ++k)
    {
        any |= w[k];
        all &= w[k];
    }
    if (any == 0)
        return Uniform::all_false;
    if (all == word_ones)
        return Uniform::all_true;
    return Uniform::mixed;
}

/**
 * @brief Classify the @p n bits starting at bit @p first -- any alignment:
 *        whole words where the range covers them, masked words at its ends.
 * @details The sub-block scan of the chunk-size advisor (a finer geometry's
 *          sub-chunk inside a materialised chunk). @p n = 0 is all_false.
 */
[[nodiscard]] inline Uniform scan_range(const word_t* w, std::size_t first, std::size_t n) noexcept
{
    bool              any_one  = false;
    bool              all_ones = true;
    std::size_t       i        = first;
    const std::size_t end      = first + n;
    while (i < end)
    {
        const std::size_t off  = i % word_bits;
        const std::size_t take = (end - i < word_bits - off) ? end - i : word_bits - off;
        const word_t      m    = (take == word_bits ? word_ones : ((word_t{1} << take) - 1)) << off;
        const word_t      x    = w[i / word_bits] & m;
        any_one                = any_one || x != 0;
        all_ones               = all_ones && x == m;
        if (any_one && !all_ones)
            return Uniform::mixed;
        i += take;
    }
    if (!any_one)
        return Uniform::all_false;
    return Uniform::all_true;
}

/// @brief Number of 1 bits among the @p n bits (tail masked).
[[nodiscard]] inline std::size_t count_ones(const word_t* w, std::size_t n) noexcept
{
    const std::size_t nw = words_for(n);
    if (nw == 0)
        return 0;
    std::size_t c = static_cast<std::size_t>(std::popcount(w[nw - 1] & tail_mask(n)));
    for (std::size_t k = 0; k + 1 < nw; ++k)
        c += static_cast<std::size_t>(std::popcount(w[k]));
    return c;
}

} // namespace shrink_array::bits
