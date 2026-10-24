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

// contracts_validation: the library's contracts reach the program's contract
// violation handler. Built with the enforce semantic whatever CHUNK_CONTRACTS
// says (examples/validation/CMakeLists.txt).
//
//   contracts_validation <case>
//     in_bounds : a[0], a[size - 1], read and written, push_back, resize,
//                 reserve, assign, clear, iterators of one array: no
//                 violation (the postconditions hold), exit 0.
//     read      : the const a[size]           -> operator[]'s pre.
//     write     : a[size] = 1.0                -> operator[]'s pre.
//     negative  : a[-1] = 1.0                  -> operator[]'s pre.
//     proxy     : a.element_proxy(size)        -> element_proxy's pre.
//     resize    : a.resize(-1)                 -> resize's pre.
//     reserve   : a.reserve(-1)                -> reserve's pre.
//     iterators : a.begin() == b.begin()       -> the iterator's contract_assert.
//   The handler checks the kind of the violation and its predicate and exits
//   0; a misuse that returns exits 1, an unexpected violation 2.

#include <contracts>   // std::contracts::contract_violation, assertion_kind
#include <cstdlib>     // std::_Exit
#include <string_view> // std::string_view
#include <utility>     // std::as_const

#include "chunk/zero_any_array.h" // zero_any::Array
#include "chunk/param_chunk.h"    // ParamChunk::init_indexes
#include "chunk/internal/print.h" // shrink_array::println

using namespace shrink_array;

namespace
{
// The violation the case provokes: its kind and a word of its predicate.
bool                           g_expect_violation = false;
std::contracts::assertion_kind g_expect_kind      = std::contracts::assertion_kind::pre;
std::string_view               g_expect_word;
} // namespace

// The program's handler: replaces libstdc++exp's default one.
void handle_contract_violation(const std::contracts::contract_violation& violation)
{
    const std::string_view predicate = violation.comment();
    println(stderr, "[contracts] violation at {}:{}: {}", violation.location().file_name(), violation.location().line(),
            predicate);
    if (!g_expect_violation || violation.kind() != g_expect_kind || !predicate.contains(g_expect_word))
    {
        println(stderr, "[contracts] FAIL: unexpected violation");
        std::_Exit(2);
    }
    println("[contracts] OK: the contract caught the misuse");
    std::_Exit(0);
}

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        println(stderr, "usage: contracts_validation in_bounds|read|write|negative|proxy|resize|reserve|iterators");
        return 3;
    }
    const std::string_view what = argv[1];
    ParamChunk::init_indexes(true);

    constexpr int           count = 1000;
    zero_any::Array<double> arr(count);
    if (what == "in_bounds")
    {
        arr[0]           = 1.0;
        arr[count - 1]   = 2.0;
        const double sum = std::as_const(arr)[0] + std::as_const(arr)[count - 1];
        arr.push_back(3.0);
        arr.emplace_back(4.0);
        arr.resize(count / 2);
        arr.resize(count, 5.0);
        arr.reserve(2 * count);
        const bool same = arr.begin() != arr.end() && arr.end() - arr.begin() == count;
        arr.assign(count / 4, 6.0);
        const bool sized = arr.size() == count / 4;
        arr.clear();
        const bool ok = sum == 3.0 && same && sized && arr.empty();
        println("[contracts] {}: {}", what, ok ? "OK" : "FAIL");
        return ok ? 0 : 1;
    }

    g_expect_violation = true;
    g_expect_word      = "index";
    if (what == "read")
    {
        const double value = std::as_const(arr)[count];
        println("[contracts] FAIL: a[{}] read {} past the end", count, value);
    }
    else if (what == "write")
    {
        arr[count] = 1.0;
        println("[contracts] FAIL: a[{}] written past the end", count);
    }
    else if (what == "negative")
    {
        arr[-1] = 1.0;
        println("[contracts] FAIL: a[-1] written");
    }
    else if (what == "proxy")
    {
        auto proxy = arr.element_proxy(count);
        (void)proxy;
        println("[contracts] FAIL: element_proxy({}) returned", count);
    }
    else if (what == "resize")
    {
        g_expect_word = "newsz";
        arr.resize(-1);
        println("[contracts] FAIL: resize(-1) returned");
    }
    else if (what == "reserve")
    {
        g_expect_word = "new_capacity";
        arr.reserve(-1);
        println("[contracts] FAIL: reserve(-1) returned");
    }
    else if (what == "iterators")
    {
        g_expect_kind = std::contracts::assertion_kind::assert;
        g_expect_word = "owner_";
        zero_any::Array<double> other(count);
        const bool              equal = arr.begin() == other.begin();
        println("[contracts] FAIL: iterators of two arrays compared ({})", equal);
    }
    return 1;
}
