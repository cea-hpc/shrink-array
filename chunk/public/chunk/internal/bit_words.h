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
 *          Concurrency. Two elements of one word are bits of the SAME memory
 *          location: a plain read-modify-write of one would erase a concurrent
 *          write of the other. Every write that may run concurrently goes
 *          through set_bit_atomic() (fetch_or / fetch_and on a std::atomic_ref
 *          of the word), every such read through get_bit_relaxed(), so writing
 *          distinct elements from distinct threads is race-free (unlike
 *          std::vector<bool>). The plain versions (get_bit, set_bit,
 *          fill_bits) are for a buffer no other thread can reach: a fresh
 *          chunk before publication, a scan at a quiescent point.
 *
 *          Tail. The bits past n in the last word are kept at 0 by fill_bits()
 *          and never written by set_bit*(); scan_words() masks them anyway, so
 *          a buffer whose tail holds garbage still classifies right.
 */

#include <atomic>  // std::atomic_ref, std::memory_order
#include <bit>     // std::popcount
#include <cstddef> // std::size_t
#include <cstdint> // std::uint64_t
#include <cstring> // std::memset

namespace shrink_array::bits
{

using WordT = std::uint64_t; ///< the storage word

inline constexpr std::size_t word_bits  = 64;            ///< bits per word
inline constexpr WordT       word_ones  = ~WordT{0};     ///< a word of 1 bits
inline constexpr std::size_t word_bytes = sizeof(WordT); ///< bytes per word

/// @brief Words holding @p nbits bits.
[[nodiscard]] constexpr std::size_t words_for(std::size_t nbits) noexcept
{
    return (nbits + word_bits - 1) / word_bits;
}

/// @brief Bytes of storage for @p nbits bits: whole words, at least one.
[[nodiscard]] constexpr std::size_t bytes_for(std::size_t nbits) noexcept
{
    const std::size_t nwords = words_for(nbits);
    return (nwords > 0 ? nwords : 1) * word_bytes;
}

/// @brief Mask of bit @p bit inside its word.
[[nodiscard]] constexpr WordT bit_mask(std::size_t bit) noexcept
{
    return WordT{1} << (bit % word_bits);
}

/// @brief Mask of the valid bits of the last word of an @p nbits-bit buffer
///        (all ones when n is a multiple of 64).
[[nodiscard]] constexpr WordT tail_mask(std::size_t nbits) noexcept
{
    const std::size_t rem = nbits % word_bits;
    return rem == 0 ? word_ones : (WordT{1} << rem) - 1;
}

/// @brief Read bit @p bit. The buffer must not be written concurrently.
[[nodiscard]] inline bool get_bit(const WordT* words, std::size_t bit) noexcept
{
    return (words[bit / word_bits] & bit_mask(bit)) != 0;
}

/// @brief Read bit @p bit while other bits of its word may be written.
[[nodiscard]] inline bool get_bit_relaxed(const WordT* words, std::size_t bit) noexcept
{
    // atomic_ref needs a non-const object; the load does not write.
    auto& word = const_cast<WordT&>(words[bit / word_bits]);
    return (std::atomic_ref<WordT>(word).load(std::memory_order_relaxed) & bit_mask(bit)) != 0;
}

/// @brief Write bit @p bit. The buffer must not be reached by another thread.
inline void set_bit(WordT* words, std::size_t bit, bool value) noexcept
{
    if (value)
    {
        words[bit / word_bits] |= bit_mask(bit);
    }
    else
    {
        words[bit / word_bits] &= ~bit_mask(bit);
    }
}

/**
 * @brief Write bit @p bit with an atomic read-modify-write of its word.
 * @details Relaxed: it only has to keep the other 63 bits of the word, which
 *          another thread may be writing. Ordering against the compressor
 *          belongs to the write protocol around it (the Dekker handshake of
 *          the proxy), as for any element store.
 * @return the previous value of the bit.
 */
inline bool set_bit_atomic(WordT* words, std::size_t bit, bool value) noexcept
{
    const std::atomic_ref<WordT> word(words[bit / word_bits]);
    const WordT                  mask = bit_mask(bit);
    const WordT                  old =
        value ? word.fetch_or(mask, std::memory_order_relaxed) : word.fetch_and(~mask, std::memory_order_relaxed);
    return (old & mask) != 0;
}

/// @brief Set the @p nbits bits to @p value and the tail of the last word to 0.
inline void fill_bits(WordT* words, std::size_t nbits, bool value) noexcept
{
    const std::size_t nw = words_for(nbits);
    if (nw == 0)
    {
        return;
    }
    std::memset(words, value ? 0xFF : 0x00, nw * word_bytes);
    if (value)
    {
        words[nw - 1] &= tail_mask(nbits);
    }
}

/// @brief Mask of bits [@p lo, @p hi) inside one word, 0 <= lo < hi <= 64.
[[nodiscard]] constexpr WordT span_mask(std::size_t lo, std::size_t hi) noexcept
{
    const WordT upto_hi = hi == word_bits ? word_ones : (WordT{1} << hi) - 1;
    return upto_hi & ~((WordT{1} << lo) - 1);
}

/**
 * @brief Set bits [@p lo, @p hi) to @p value: whole words stored, the two edge
 *        words masked. The buffer must not be written concurrently (the
 *        edge words are a plain read-modify-write).
 */
inline void fill_bit_range(WordT* words, std::size_t lo, std::size_t hi, bool value) noexcept
{
    std::size_t i = lo;
    while (i < hi)
    {
        const std::size_t word_idx = i / word_bits;
        const std::size_t off      = i % word_bits;
        const std::size_t end  = (hi - (word_idx * word_bits)) < word_bits ? hi - (word_idx * word_bits) : word_bits;
        const WordT       mask = span_mask(off, end);
        words[word_idx]        = value ? (words[word_idx] | mask) : (words[word_idx] & ~mask);
        i                      = (word_idx * word_bits) + end;
    }
}

/**
 * @brief fill_bit_range() for a buffer other threads write and read at the
 *        same time (fill_range_mt): each word is updated atomically.
 * @details A word the range covers entirely belongs to this range alone (the
 *          callers write disjoint indices): a relaxed atomic store. A word it
 *          covers in part is shared with other writers: fetch_or / fetch_and
 *          under the range's mask, so their bits are kept. Atomic both ways,
 *          so the readers' relaxed word loads (get_bit_relaxed) race with
 *          nothing.
 */
inline void fill_bit_range_atomic(WordT* words, std::size_t lo, std::size_t hi, bool value) noexcept
{
    std::size_t i = lo;
    while (i < hi)
    {
        const std::size_t word_idx = i / word_bits;
        const std::size_t off      = i % word_bits;
        const std::size_t end  = (hi - (word_idx * word_bits)) < word_bits ? hi - (word_idx * word_bits) : word_bits;
        const WordT       mask = span_mask(off, end);
        const std::atomic_ref<WordT> word(words[word_idx]);
        if (mask == word_ones)
        {
            word.store(value ? word_ones : 0, std::memory_order_relaxed);
        }
        else if (value)
        {
            word.fetch_or(mask, std::memory_order_relaxed);
        }
        else
        {
            word.fetch_and(~mask, std::memory_order_relaxed);
        }
        i = (word_idx * word_bits) + end;
    }
}

/**
 * @brief Store bits [@p lo, @p hi) from @p get(i) (a bool for bit i), 64 at a
 *        time: each word is assembled, then stored under its mask. Same
 *        concurrency contract as fill_bit_range().
 */
template <typename Get>
inline void store_bit_range(WordT* words, std::size_t lo, std::size_t hi, Get&& get)
{
    std::size_t i = lo;
    while (i < hi)
    {
        const std::size_t word_idx = i / word_bits;
        const std::size_t off      = i % word_bits;
        const std::size_t end  = (hi - (word_idx * word_bits)) < word_bits ? hi - (word_idx * word_bits) : word_bits;
        WordT             bits = 0;
        for (std::size_t b = off; b < end; ++b)
        {
            bits |= static_cast<WordT>(static_cast<bool>(get((word_idx * word_bits) + b))) << b;
        }
        const WordT mask = span_mask(off, end);
        words[word_idx]  = (words[word_idx] & ~mask) | bits;
        i                = (word_idx * word_bits) + end;
    }
}

/**
 * @brief store_bit_range() for a buffer other threads write and read at the
 *        same time (copy_range_mt, transform_range_mt): each word is updated
 *        atomically, as in fill_bit_range_atomic().
 * @details A word the range covers entirely: a relaxed atomic store of the
 *          assembled word. A word it covers in part: fetch_or of its 1 bits,
 *          then fetch_and clearing its 0 bits -- two RMWs, each keeping the
 *          bits of the other writers, who write other bits.
 */
template <typename Get>
inline void store_bit_range_atomic(WordT* words, std::size_t lo, std::size_t hi, Get&& get)
{
    std::size_t i = lo;
    while (i < hi)
    {
        const std::size_t word_idx = i / word_bits;
        const std::size_t off      = i % word_bits;
        const std::size_t end  = (hi - (word_idx * word_bits)) < word_bits ? hi - (word_idx * word_bits) : word_bits;
        WordT             bits = 0;
        for (std::size_t b = off; b < end; ++b)
        {
            bits |= static_cast<WordT>(static_cast<bool>(get((word_idx * word_bits) + b))) << b;
        }
        const WordT                  mask = span_mask(off, end);
        const std::atomic_ref<WordT> word(words[word_idx]);
        if (mask == word_ones)
        {
            word.store(bits, std::memory_order_relaxed);
        }
        else
        {
            word.fetch_or(bits, std::memory_order_relaxed);
            word.fetch_and(~(mask & ~bits), std::memory_order_relaxed);
        }
        i = (word_idx * word_bits) + end;
    }
}

/// @brief What a scan of a buffer found.
enum class Uniform : std::uint8_t
{
    all_false, ///< every bit 0 -- the first constant
    all_true,  ///< every bit 1 -- the second constant
    mixed      ///< both values present
};

/**
 * @brief Classify the @p nbits bits of a buffer, one word at a time.
 * @details A chunk of 256 bools is 4 words: 4 compares where the byte layout
 *          makes 256. The tail of the last word is masked out. @p nbits = 0 is
 *          all_false, like an empty chunk.
 */
[[nodiscard]] inline Uniform scan_words(const WordT* words, std::size_t nbits) noexcept
{
    const std::size_t nw = words_for(nbits);
    if (nw == 0)
    {
        return Uniform::all_false;
    }
    const WordT last = words[nw - 1] & tail_mask(nbits);
    WordT       any  = last;                     // OR: some 1 bit
    WordT       all  = last | ~tail_mask(nbits); // AND: every bit 1 (tail forced to 1)
    for (std::size_t k = 0; k + 1 < nw; ++k)
    {
        any |= words[k];
        all &= words[k];
    }
    if (any == 0)
    {
        return Uniform::all_false;
    }
    if (all == word_ones)
    {
        return Uniform::all_true;
    }
    return Uniform::mixed;
}

/**
 * @brief Classify the @p nbits bits starting at bit @p first -- any alignment:
 *        whole words where the range covers them, masked words at its ends.
 * @details The sub-block scan of the chunk-size advisor (a finer geometry's
 *          sub-chunk inside a materialised chunk). @p nbits = 0 is all_false.
 */
[[nodiscard]] inline Uniform scan_range(const WordT* words, std::size_t first, std::size_t nbits) noexcept
{
    bool              any_one  = false;
    bool              all_ones = true;
    std::size_t       i        = first;
    const std::size_t end      = first + nbits;
    while (i < end)
    {
        const std::size_t off  = i % word_bits;
        const std::size_t take = (end - i < word_bits - off) ? end - i : word_bits - off;
        const WordT       mask = (take == word_bits ? word_ones : ((WordT{1} << take) - 1)) << off;
        const WordT       bits = words[i / word_bits] & mask;
        any_one                = any_one || bits != 0;
        all_ones               = all_ones && bits == mask;
        if (any_one && !all_ones)
        {
            return Uniform::mixed;
        }
        i += take;
    }
    if (!any_one)
    {
        return Uniform::all_false;
    }
    return Uniform::all_true;
}

/// @brief Number of 1 bits among the @p nbits bits (tail masked).
[[nodiscard]] inline std::size_t count_ones(const WordT* words, std::size_t nbits) noexcept
{
    const std::size_t nw = words_for(nbits);
    if (nw == 0)
    {
        return 0;
    }
    auto ones = static_cast<std::size_t>(std::popcount(words[nw - 1] & tail_mask(nbits)));
    for (std::size_t k = 0; k + 1 < nw; ++k)
    {
        ones += static_cast<std::size_t>(std::popcount(words[k]));
    }
    return ones;
}

} // namespace shrink_array::bits
