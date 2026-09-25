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

#include <cstddef>
#include <cstdio>
#include <random>
#include <thread>
#include <vector>

#include "chunk/internal/bit_words.h"

namespace bits = shrink_array::bits;

namespace
{

int g_fails = 0;

void check(bool ok, const char* what, std::size_t n)
{
    if (!ok)
    {
        ++g_fails;
        std::printf("FAIL %s (n=%zu)\n", what, n);
    }
}

void geometry()
{
    check(bits::words_for(0) == 0 && bits::bytes_for(0) == 8, "empty: no word, one word of storage", 0);
    for (std::size_t n : {1, 32, 63, 64, 65, 256, 512, 4096})
    {
        check(bits::words_for(n) == (n + 63) / 64, "words_for", n);
        check(bits::bytes_for(n) == bits::words_for(n) * 8, "bytes_for", n);
        const bits::word_t tm = bits::tail_mask(n);
        check(n % 64 == 0 ? tm == bits::word_ones : tm == (bits::word_t{1} << (n % 64)) - 1, "tail_mask", n);
    }
}

void against_vector(std::size_t n, std::mt19937_64& rng)
{
    std::vector<bits::word_t> w(bits::words_for(n) + 1, 0xDEADBEEFDEADBEEFull); // +1: a guard word
    const bits::word_t        guard = w.back();
    bits::fill_bits(w.data(), n, false);
    std::vector<bool> ref(n, false);

    std::uniform_int_distribution<std::size_t> pick(0, n - 1);
    for (std::size_t k = 0; k < 4 * n; ++k)
    {
        const std::size_t i = pick(rng);
        const bool        v = (rng() & 1) != 0;
        const bool        old = (k % 2 == 0) ? bits::set_bit_atomic(w.data(), i, v) : bits::get_bit(w.data(), i);
        if (k % 2 != 0)
            bits::set_bit(w.data(), i, v);
        check(old == ref[i], "previous value", n);
        ref[i] = v;
    }
    bool same = true;
    for (std::size_t i = 0; i < n; ++i)
        same = same && bits::get_bit(w.data(), i) == ref[i] && bits::get_bit_relaxed(w.data(), i) == ref[i];
    check(same, "get_bit after random writes", n);

    std::size_t ones = 0;
    for (std::size_t i = 0; i < n; ++i)
        ones += ref[i] ? 1 : 0;
    check(bits::count_ones(w.data(), n) == ones, "count_ones", n);
    check(w.back() == guard, "no write past the last word", n);
    check((w[bits::words_for(n) - 1] & ~bits::tail_mask(n)) == 0, "tail stays 0", n);
}

void scans(std::size_t n)
{
    std::vector<bits::word_t> w(bits::words_for(n));

    bits::fill_bits(w.data(), n, false);
    check(bits::scan_words(w.data(), n) == bits::Uniform::all_false, "scan all_false", n);
    bits::fill_bits(w.data(), n, true);
    check(bits::scan_words(w.data(), n) == bits::Uniform::all_true, "scan all_true", n);
    check(bits::count_ones(w.data(), n) == n, "count_ones all_true", n);

    // Garbage in the tail must not change the answer.
    if (n % 64 != 0)
    {
        bits::fill_bits(w.data(), n, false);
        w.back() |= ~bits::tail_mask(n);
        check(bits::scan_words(w.data(), n) == bits::Uniform::all_false, "scan all_false, dirty tail", n);
        bits::fill_bits(w.data(), n, true);
        w.back() &= bits::tail_mask(n); // tail 0 is the kept state; 1 must also pass
        w.back() |= ~bits::tail_mask(n);
        check(bits::scan_words(w.data(), n) == bits::Uniform::all_true, "scan all_true, dirty tail", n);
    }

    // One differing bit, first and last element (n >= 2: with one element,
    // flipping it gives the other constant, not a mixed chunk).
    for (std::size_t i : {std::size_t{0}, n - 1})
    {
        if (n < 2)
            break;
        bits::fill_bits(w.data(), n, false);
        bits::set_bit(w.data(), i, true);
        check(bits::scan_words(w.data(), n) == bits::Uniform::mixed, "scan mixed (one 1)", n);
        bits::fill_bits(w.data(), n, true);
        bits::set_bit(w.data(), i, false);
        check(bits::scan_words(w.data(), n) == bits::Uniform::mixed, "scan mixed (one 0)", n);
    }
}

void ranges(std::mt19937_64& rng)
{
    // scan_range over every (first, n) of a 3-word buffer, against a bit loop.
    std::vector<bits::word_t> w(3);
    for (int trial = 0; trial < 40; ++trial)
    {
        for (auto& x : w)
            x = rng();
        if (trial % 4 == 1)
            w.assign(3, 0);
        if (trial % 4 == 2)
            w.assign(3, bits::word_ones);
        if (trial % 4 == 3) // a single 0 in a sea of 1
        {
            w.assign(3, bits::word_ones);
            bits::set_bit(w.data(), static_cast<std::size_t>(rng() % 192), false);
        }
        for (std::size_t first = 0; first < 192; first += 7)
            for (std::size_t n = 0; first + n <= 192; n += 5)
            {
                bool any = false, all = true;
                for (std::size_t i = first; i < first + n; ++i)
                {
                    any = any || bits::get_bit(w.data(), i);
                    all = all && bits::get_bit(w.data(), i);
                }
                const bits::Uniform expect = !any ? bits::Uniform::all_false
                                             : all ? bits::Uniform::all_true
                                                   : bits::Uniform::mixed;
                if (bits::scan_range(w.data(), first, n) != expect)
                {
                    check(false, "scan_range", n);
                    return;
                }
            }
    }
}

void concurrent(std::size_t n, int rounds)
{
    constexpr int             n_thr = 8;
    std::vector<bits::word_t> w(bits::words_for(n));
    bits::fill_bits(w.data(), n, false);

    // Each round: every thread sets its bits, then clears the odd ones among
    // them. Neighbouring bits of one word belong to different threads, so
    // every word is written by all 8 at once.
    std::vector<std::thread> pool;
    for (int t = 0; t < n_thr; ++t)
        pool.emplace_back(
            [&w, n, rounds, t]
            {
                for (int r = 0; r < rounds; ++r)
                {
                    for (std::size_t i = static_cast<std::size_t>(t); i < n; i += n_thr)
                        bits::set_bit_atomic(w.data(), i, true);
                    for (std::size_t i = static_cast<std::size_t>(t); i < n; i += n_thr)
                        if ((i / n_thr) % 2 == 1)
                            bits::set_bit_atomic(w.data(), i, false);
                }
            });
    for (auto& th : pool)
        th.join();

    bool same = true;
    for (std::size_t i = 0; i < n; ++i)
        same = same && bits::get_bit(w.data(), i) == ((i / n_thr) % 2 == 0);
    check(same, "concurrent writes to neighbouring bits: none lost", n);
}

} // namespace

int main()
{
    std::mt19937_64 rng(12345);
    geometry();
    for (std::size_t n : {1, 32, 63, 64, 65, 256, 512, 1000, 4096})
    {
        against_vector(n, rng);
        scans(n);
    }
    ranges(rng);
    for (std::size_t n : {64, 256, 4096})
        concurrent(n, 2000);

    if (g_fails == 0)
        std::printf("bit_words_validation: all checks passed\n");
    return g_fails == 0 ? 0 : 1;
}
