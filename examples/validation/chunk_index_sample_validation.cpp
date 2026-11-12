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

// ============================================================================
// The chunk sample of the compression-residual messages
// (chunk_index_sample.h), alone: no chunk, no array, no variant.
// ============================================================================
//
//   1. the text: "j@e j@e ...", j a chunk index and e = j * chunk size the
//      index of its first element; nothing for an empty list;
//   2. at most `max` chunks shown, then " (+N more)";
//   3. the stack buffer: a text longer than it is cut at its size, never
//      written past;
//   4. the element index in the array's index type: past INT_MAX with a
//      64-bit one.
//
// Its one caller is a teardown message at compress=info, which the tests
// never enable: the nightly's coverage report had it at 0 %. Exit status 0
// when every check passes.

#include <array>       // std::array
#include <cstddef>     // std::size_t
#include <string_view> // std::string_view
#include <vector>      // std::vector

#include "chunk/internal/chunk_index_sample.h" // internal::chunk_index_sample
#include "chunk/print.h"                       // shrink_array::println

using shrink_array::println;
using shrink_array::internal::chunk_index_sample;

namespace
{

int g_fails = 0;

void check(bool ok, const char* what, std::string_view got)
{
    if (!ok)
    {
        ++g_fails;
        println("FAIL {}: got \"{}\"", what, got);
    }
}

// --- 1 ----------------------------------------------------------------------
void check_text()
{
    std::array<char, 256> buf{};
    std::string_view      got = chunk_index_sample(buf, {}, 256);
    check(got.empty(), "an empty list: nothing", got);

    got = chunk_index_sample(buf, {3, 7}, 256);
    check(got == "3@768 7@1792", "two chunks of 256", got);

    got = chunk_index_sample(buf, {0}, 64);
    check(got == "0@0", "chunk 0", got);
}

// --- 2 ----------------------------------------------------------------------
void check_max()
{
    std::array<char, 256> buf{};
    std::vector<int>      idx;
    for (int j = 0; j < 12; ++j)
    {
        idx.push_back(j);
    }
    std::string_view got = chunk_index_sample(buf, idx, 1);
    check(got == "0@0 1@1 2@2 3@3 4@4 5@5 6@6 7@7 8@8 9@9 (+2 more)", "12 chunks, 10 shown (the default)", got);

    got = chunk_index_sample(buf, idx, 1, 3);
    check(got == "0@0 1@1 2@2 (+9 more)", "12 chunks, 3 shown", got);

    got = chunk_index_sample(buf, {5, 6}, 1, 2);
    check(got == "5@5 6@6", "exactly max chunks: no (+N more)", got);
}

// --- 3 ----------------------------------------------------------------------
void check_buffer()
{
    // A canary past the buffer the sample is given: it must stay intact.
    struct Guarded
    {
        std::array<char, 8> buf{};
        std::array<char, 8> canary{'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C'};
    } g;
    const std::string_view got = chunk_index_sample(g.buf, {123, 456, 789}, 1024);
    check(got.size() == g.buf.size() && got == std::string_view("123@1259", 8), "cut at the buffer's 8 bytes", got);
    bool intact = true;
    for (const char c : g.canary)
    {
        intact = intact && c == 'C';
    }
    check(intact, "nothing written past the buffer", got);

    std::array<char, 16>   mid{};
    const std::string_view cut = chunk_index_sample(mid, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, 1);
    check(cut == "1@1 2@2 3@3 4@4 ", "a list longer than 16 bytes: its first 16", cut);
}

// --- 4 ----------------------------------------------------------------------
void check_wide_index()
{
    std::array<char, 64>   buf{};
    const std::string_view got = chunk_index_sample(buf, {3000000}, static_cast<long>(1024));
    check(got == "3000000@3072000000", "an element index past INT_MAX with a long index type", got);
}

} // namespace

int main()
{
    check_text();
    check_max();
    check_buffer();
    check_wide_index();

    if (g_fails != 0)
    {
        println("chunk_index_sample_validation: {} check(s) FAILED", g_fails);
        return 1;
    }
    println("chunk_index_sample_validation: OK");
    return 0;
}
