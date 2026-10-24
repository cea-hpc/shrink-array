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

// contracts_validation: the preconditions of operator[] and element_proxy
// (CHUNK_PRE, chunk/internal/contracts.h) reach the program's contract
// violation handler. Built only where the compiler has contracts, and with the
// enforce semantic whatever CHUNK_CONTRACTS says (examples/validation/CMakeLists.txt).
//
//   contracts_validation <case>
//     in_bounds : a[0], a[size - 1], read and written: no violation, exit 0.
//     read      : the const a[size]     -> the handler, exit 0.
//     write     : a[size] = 1.0          -> the handler, exit 0.
//     negative  : a[-1] = 1.0            -> the handler, exit 0.
//     proxy     : a.element_proxy(size)  -> the handler, exit 0.
//   The handler checks that the violation is a precondition and exits 0; an
//   out-of-bounds access that returns exits 1, a violation in in_bounds 2.

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
bool g_expect_violation = false;
} // namespace

// The program's handler: replaces libstdc++exp's default one.
void handle_contract_violation(const std::contracts::contract_violation& violation)
{
    const std::string_view predicate = violation.comment();
    println(stderr, "[contracts] violation at {}:{}: {}", violation.location().file_name(), violation.location().line(),
            predicate);
    const bool is_pre = violation.kind() == std::contracts::assertion_kind::pre;
    if (!g_expect_violation || !is_pre || !predicate.contains("index"))
    {
        println(stderr, "[contracts] FAIL: unexpected violation");
        std::_Exit(2);
    }
    println("[contracts] OK: the precondition caught the access");
    std::_Exit(0);
}

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        println(stderr, "usage: contracts_validation in_bounds|read|write|negative|proxy");
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
        println("[contracts] {}: {}", what, sum == 3.0 ? "OK" : "FAIL");
        return sum == 3.0 ? 0 : 1;
    }

    g_expect_violation = true;
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
    return 1;
}
