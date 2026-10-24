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

// ============================================================================
// The bit-packed storage kernel of ArrayChunkBits (bit_words.h), alone: no
// chunk, no array, no variant.
// ============================================================================
//
//   1. geometry: words_for / bytes_for / tail_mask at the chunk sizes the
//      library uses (32 is ParamChunk::size_min) and at the edges of a word;
//   2. get/set against std::vector<bool>, over a random write sequence;
//   3. scan_words: all-false / all-true / mixed, including a tail full of
//      garbage (it must be masked) and a single differing bit at each end;
//   4. count_ones against the reference;
//   5. concurrency: 8 threads write interleaved bits of the SAME words (bit i
//      belongs to thread i % 8) through set_bit_atomic. A plain `|=` loses
//      bits here; the atomic RMW must lose none.
//
// Exit status 0 when every check passes.

#include <algorithm> // std::min
#include <cstddef>   // std::size_t
#include <random>    // std::mt19937_64, std::uniform_int_distribution
#include <ranges>    // std::views::enumerate
#include <thread>    // std::thread
#include <utility>   // std::pair
#include <vector>    // std::vector

#include "chunk/internal/bit_words.h" // bits::WordT, bits::fill_bits, bits::get_bit, bits::scan_words
#include "chunk/internal/print.h"     // shrink_array::println

using shrink_array::println;

namespace bits = shrink_array::bits;

namespace
{

int g_fails = 0;

void check(bool ok, const char* what, std::size_t nbits)
{
    if (!ok)
    {
        ++g_fails;
        println("FAIL {} (n={})", what, nbits);
    }
}

void geometry()
{
    check(bits::words_for(0) == 0 && bits::bytes_for(0) == 8, "empty: no word, one word of storage", 0);
    for (const std::size_t nbits : {1, 32, 63, 64, 65, 256, 512, 4096})
    {
        check(bits::words_for(nbits) == (nbits + 63) / 64, "words_for", nbits);
        check(bits::bytes_for(nbits) == bits::words_for(nbits) * 8, "bytes_for", nbits);
        const bits::WordT tm = bits::tail_mask(nbits);
        check(nbits % 64 == 0 ? tm == bits::word_ones : tm == (bits::WordT{1} << (nbits % 64)) - 1, "tail_mask", nbits);
    }
}

void against_vector(std::size_t nbits, std::mt19937_64& rng)
{
    std::vector<bits::WordT> words(bits::words_for(nbits) + 1, 0xDEADBEEFDEADBEEFull); // +1: a guard word
    const bits::WordT        guard = words.back();
    bits::fill_bits(words.data(), nbits, false);
    std::vector<bool> ref(nbits, false);

    std::uniform_int_distribution<std::size_t> pick(0, nbits - 1);
    for (std::size_t k = 0; k < 4 * nbits; ++k)
    {
        const std::size_t bit   = pick(rng);
        const bool        value = (rng() & 1) != 0;
        const bool        old =
            (k % 2 == 0) ? bits::set_bit_atomic(words.data(), bit, value) : bits::get_bit(words.data(), bit);
        if (k % 2 != 0)
        {
            bits::set_bit(words.data(), bit, value);
        }
        check(old == ref[bit], "previous value", nbits);
        ref[bit] = value;
    }
    bool same = true;
    for (const auto& [i, r] : std::views::enumerate(ref))
    {
        same = same && bits::get_bit(words.data(), i) == r && bits::get_bit_relaxed(words.data(), i) == r;
    }
    check(same, "get_bit after random writes", nbits);

    std::size_t ones = 0;
    for (const bool r : ref)
    {
        ones += r ? 1 : 0;
    }
    check(bits::count_ones(words.data(), nbits) == ones, "count_ones", nbits);
    check(words.back() == guard, "no write past the last word", nbits);
    check((words[bits::words_for(nbits) - 1] & ~bits::tail_mask(nbits)) == 0, "tail stays 0", nbits);
}

void scans(std::size_t nbits)
{
    std::vector<bits::WordT> words(bits::words_for(nbits));

    bits::fill_bits(words.data(), nbits, false);
    check(bits::scan_words(words.data(), nbits) == bits::Uniform::all_false, "scan all_false", nbits);
    bits::fill_bits(words.data(), nbits, true);
    check(bits::scan_words(words.data(), nbits) == bits::Uniform::all_true, "scan all_true", nbits);
    check(bits::count_ones(words.data(), nbits) == nbits, "count_ones all_true", nbits);

    // Garbage in the tail must not change the answer.
    if (nbits % 64 != 0)
    {
        bits::fill_bits(words.data(), nbits, false);
        words.back() |= ~bits::tail_mask(nbits);
        check(bits::scan_words(words.data(), nbits) == bits::Uniform::all_false, "scan all_false, dirty tail", nbits);
        bits::fill_bits(words.data(), nbits, true);
        words.back() &= bits::tail_mask(nbits); // tail 0 is the kept state; 1 must also pass
        words.back() |= ~bits::tail_mask(nbits);
        check(bits::scan_words(words.data(), nbits) == bits::Uniform::all_true, "scan all_true, dirty tail", nbits);
    }

    // One differing bit, first and last element (nbits >= 2: with one element,
    // flipping it gives the other constant, not a mixed chunk).
    for (const std::size_t i : {0uz, nbits - 1})
    {
        if (nbits < 2)
        {
            break;
        }
        bits::fill_bits(words.data(), nbits, false);
        bits::set_bit(words.data(), i, true);
        check(bits::scan_words(words.data(), nbits) == bits::Uniform::mixed, "scan mixed (one 1)", nbits);
        bits::fill_bits(words.data(), nbits, true);
        bits::set_bit(words.data(), i, false);
        check(bits::scan_words(words.data(), nbits) == bits::Uniform::mixed, "scan mixed (one 0)", nbits);
    }
}

void ranges(std::mt19937_64& rng)
{
    // scan_range over every (first, n) of a 3-word buffer, against a bit loop.
    std::vector<bits::WordT> words(3);
    for (int trial = 0; trial < 40; ++trial)
    {
        for (auto& word : words)
        {
            word = rng();
        }
        if (trial % 4 == 1)
        {
            words.assign(3, 0);
        }
        if (trial % 4 == 2)
        {
            words.assign(3, bits::word_ones);
        }
        if (trial % 4 == 3) // a single 0 in a sea of 1
        {
            words.assign(3, bits::word_ones);
            bits::set_bit(words.data(), static_cast<std::size_t>(rng() % 192), false);
        }
        for (std::size_t first = 0; first < 192; first += 7)
        {
            for (std::size_t n = 0; first + n <= 192; n += 5)
            {
                bool any = false;
                bool all = true;
                for (std::size_t i = first; i < first + n; ++i)
                {
                    any = any || bits::get_bit(words.data(), i);
                    all = all && bits::get_bit(words.data(), i);
                }
                const bits::Uniform expect = !any  ? bits::Uniform::all_false
                                             : all ? bits::Uniform::all_true
                                                   : bits::Uniform::mixed;
                if (bits::scan_range(words.data(), first, n) != expect)
                {
                    check(false, "scan_range", n);
                    return;
                }
            }
        }
    }
}

void concurrent(std::size_t nbits, int rounds)
{
    constexpr int            n_thr = 8;
    std::vector<bits::WordT> words(bits::words_for(nbits));
    bits::fill_bits(words.data(), nbits, false);

    // Each round: every thread sets its bits, then clears the odd ones among
    // them. Neighbouring bits of one word belong to different threads, so
    // every word is written by all 8 at once.
    std::vector<std::thread> pool;
    pool.reserve(n_thr);
    for (int t = 0; t < n_thr; ++t)
    {
        pool.emplace_back(
            [&words, nbits, rounds, t]
            {
                for (int r = 0; r < rounds; ++r)
                {
                    for (std::size_t i = static_cast<std::size_t>(t); i < nbits; i += n_thr)
                    {
                        bits::set_bit_atomic(words.data(), i, true);
                    }
                    for (std::size_t i = static_cast<std::size_t>(t); i < nbits; i += n_thr)
                    {
                        if ((i / n_thr) % 2 == 1)
                        {
                            bits::set_bit_atomic(words.data(), i, false);
                        }
                    }
                }
            });
    }
    for (auto& th : pool)
    {
        th.join();
    }

    bool same = true;
    for (std::size_t i = 0; i < nbits; ++i)
    {
        same = same && bits::get_bit(words.data(), i) == ((i / n_thr) % 2 == 0);
    }
    check(same, "concurrent writes to neighbouring bits: none lost", nbits);
}

void concurrent_ranges(std::size_t nbits, int rounds)
{
    // Spans of 1..37 bits handed out round-robin to 8 threads: every span
    // edge is a word shared with another thread's span. Each thread sets its
    // spans, then clears the odd ones -- fill_bit_range_atomic must lose none.
    constexpr int                                    n_thr = 8;
    std::vector<std::pair<std::size_t, std::size_t>> spans;
    for (std::size_t lo = 0, len = 1; lo < nbits; lo += len, len = (len % 37) + 1)
    {
        spans.emplace_back(lo, std::min(nbits, lo + len));
    }
    std::vector<bits::WordT> words(bits::words_for(nbits));
    bits::fill_bits(words.data(), nbits, false);

    std::vector<std::thread> pool;
    pool.reserve(n_thr);
    for (int t = 0; t < n_thr; ++t)
    {
        pool.emplace_back(
            [&words, &spans, rounds, t]
            {
                for (int r = 0; r < rounds; ++r)
                {
                    for (std::size_t s = static_cast<std::size_t>(t); s < spans.size(); s += n_thr)
                    {
                        bits::fill_bit_range_atomic(words.data(), spans[s].first, spans[s].second, true);
                    }
                    for (std::size_t s = static_cast<std::size_t>(t); s < spans.size(); s += n_thr)
                    {
                        if (s % 2 == 1)
                        {
                            bits::fill_bit_range_atomic(words.data(), spans[s].first, spans[s].second, false);
                        }
                    }
                }
            });
    }
    for (auto& th : pool)
    {
        th.join();
    }

    bool same = true;
    for (std::size_t s = 0; s < spans.size(); ++s)
    {
        for (std::size_t i = spans[s].first; i < spans[s].second; ++i)
        {
            same = same && bits::get_bit(words.data(), i) == (s % 2 == 0);
        }
    }
    check(same, "concurrent fill_bit_range_atomic: no bit lost", nbits);
}

} // namespace

int main()
{
    std::mt19937_64 rng(12345);
    geometry();
    for (const std::size_t nbits : {1, 32, 63, 64, 65, 256, 512, 1000, 4096})
    {
        against_vector(nbits, rng);
        scans(nbits);
    }
    ranges(rng);
    for (const std::size_t nbits : {64, 256, 4096})
    {
        concurrent(nbits, 2000);
    }
    for (const std::size_t nbits : {256, 4096})
    {
        concurrent_ranges(nbits, 500);
    }

    if (g_fails == 0)
    {
        println("bit_words_validation: all checks passed");
    }
    return g_fails == 0 ? 0 : 1;
}
