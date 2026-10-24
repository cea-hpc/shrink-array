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

// element_ref_validation: what operator[] accepts and refuses (internal/element_ref.h).
//
//   - Built without ELEMENT_REF_CASE: an executable. Every use of a[i] that
//     code written for std::vector makes -- assignment, a[i] = b[j], compound
//     operators, ++/--, unary operators, arithmetic with scalars and complex
//     values, comparisons both ways and between two elements, os << a[i],
//     push_back / emplace_back, at() / front() / back() -- compiles, and
//     leaves the values std::vector would. Exit 0, or 1 with the first
//     mismatch.
//   - Built with ELEMENT_REF_CASE=N (1..8): one misuse of a NAMED element proxy
//     (`auto v = a[i];` then v used), which must NOT compile. The ctest builds
//     the object target and passes when the compiler output carries the
//     ElementRef message. N = 9: an element passed to printf, refused by the
//     compiler itself (chunk_hardening's -Werror=conditionally-supported for
//     GCC, an error by default for clang).
//
// Run by ctest; the variant comes from the CHUNK_VARIANT_* macro of the target.

#include <complex>     // std::complex<double>
#include <cstdio>      // std::printf
#include <sstream>     // std::ostringstream (os << a[i])
#include <type_traits> // std::conditional_t, std::is_same_v
#include <utility>     // std::move
#include <vector>      // std::vector (push_back / emplace_back)

#include "chunk/zero_array.h"           // zero::Array
#include "chunk/zero_one_value_array.h" // zero_one_value::Array
#include "chunk/zero_any_array.h"       // zero_any::Array
#include "chunk/config.h"               // chunk_config::use_zero_one_value*, build_config::strict_element_ref
#include "chunk/param_chunk.h"          // ParamChunk::init_indexes
#include "chunk/zero_chunk.h"           // release_chunk_pools, destroy_block_chunk
#include "chunk/internal/print.h"       // shrink_array::println

using namespace shrink_array;

namespace
{
template <typename T>
using Array =
    std::conditional_t<chunk_config::use_zero_any,
                       zero_any::Array<T>,
                       std::conditional_t<chunk_config::use_zero_one_value, zero_one_value::Array<T>, zero::Array<T>>>;

int fails = 0;

template <typename T>
void expect(const char* what, const T& got, const T& want)
{
    if (!(got == want))
    {
        println("[element_ref] FAIL: {}", what);
        ++fails;
    }
}

#ifndef ELEMENT_REF_CASE
int usage()
{
    static_assert(std::is_same_v<Array<double>::reference, ElementRef<Array<double>::ProxyType, double>> ||
                      !build_config::strict_element_ref,
                  "operator[] returns an ElementRef (the bare proxy with -DCHUNK_LOOSE_ELEMENT_REF)");
    constexpr int count = 1000;
    Array<double> arr(count);
    Array<double> other(count);

    arr[1] = 2.0;
    expect("a[1] = 2.0", static_cast<double>(arr[1]), 2.0);
    arr[2] = arr[1]; // proxy-to-proxy, same array
    expect("a[2] = a[1]", static_cast<double>(arr[2]), 2.0);
    other[3] = arr[2]; // proxy-to-proxy, another array
    expect("b[3] = a[2]", static_cast<double>(other[3]), 2.0);
    arr[4] += 1.5;
    arr[4] *= 2.0;
    expect("a[4] += 1.5; *= 2", static_cast<double>(arr[4]), 3.0);
    const double sum = arr[1] + arr[4];
    expect("a[1] + a[4]", sum, 5.0);
    const double neg = -arr[4] * 2.0;
    expect("-a[4] * 2", neg, -6.0);
    ++arr[5];
    arr[5]++;
    --arr[5];
    expect("++ / -- on a[5]", static_cast<double>(arr[5]), 1.0);
    const double result = (arr[6] += 4.0); // a compound result read as a value
    expect("(a[6] += 4) read", result, 4.0);
    expect("a[1] == 2", arr[1] == 2.0, true);
    expect("2 == a[1]", 2.0 == arr[1], true);
    expect("a[1] != a[4]", arr[1] != arr[4], true);
    expect("a[1] == b[3]", arr[1] == other[3], true);

    std::vector<double> vec;
    vec.push_back(arr[1]);
    vec.emplace_back(arr[4]);
    expect("push_back / emplace_back", vec[0] + vec[1], 5.0);

    std::ostringstream os;
    os << arr[1];
    expect("os << a[1]", os.str() == "2", true);

    arr.at(7)   = 7.0;
    arr.front() = 1.0;
    arr.back()  = 9.0;
    expect("at / front / back", static_cast<double>(arr[7]) + arr[0] + arr[count - 1], 17.0);

    auto&& kept     = arr[8]; // a proxy kept on purpose
    std::move(kept) = 8.0;
    expect("std::move(kept) = 8", static_cast<double>(arr[8]), 8.0);
    auto proxy = arr.element_proxy(9); // the proxy itself, in every build
    proxy      = 9.0;
    expect("element_proxy(9) = 9", static_cast<double>(arr[9]), 9.0);

    Array<std::complex<double>> cplx(100);
    cplx[1] = std::complex<double>(1.0, 2.0);
    cplx[2] = -cplx[1];
    cplx[3] = 2.0 * cplx[1];
    cplx[4] = std::complex<double>(0.0, 1.0) * cplx[1];
    cplx[5] = cplx[1];
    expect("complex -c, 2*c, i*c, c = c",
           std::complex<double>(cplx[2]) + std::complex<double>(cplx[3]) + std::complex<double>(cplx[4]) +
               std::complex<double>(cplx[5]),
           std::complex<double>(0.0, 5.0)); // (-1-2i) + (2+4i) + (-2+i) + (1+2i)

    Array<int> ints(100);
    ints[1] = 7;
    ints[2] = ints[1] % 3;
    ints[3] = ~ints[1];
    ints[4] = 1;
    ints[4] <<= 3;
    expect("int %, ~, <<=", ints[2] + ints[3] + ints[4], 1 - 8 + 8);

    Array<bool> bools(100);
    bools[1]      = true;
    bools[2]      = bools[1];
    const bool tb = bools[2];
    expect("bool t[2] = t[1]", tb, true);

    if (fails == 0)
    {
        println("[element_ref] OK: every std::vector-style use of a[i] compiles and holds");
    }
    return fails != 0 ? 1 : 0;
}
#else
// One misuse per build: each must fail to compile with the ElementRef message.
void misuse()
{
    Array<double> arr(10);
    #if ELEMENT_REF_CASE == 1
    auto   elem = arr[0]; // conversion
    double read = elem;
    (void)read;
    #elif ELEMENT_REF_CASE == 2
    auto elem = arr[0]; // assignment
    elem      = 3.0;
    #elif ELEMENT_REF_CASE == 3
    auto elem = arr[0]; // arithmetic
    println("{:g}", elem + 1.0);
    #elif ELEMENT_REF_CASE == 4
    auto   elem = (arr[0] += 1.0); // a compound result kept
    double read = elem;
    (void)read;
    #elif ELEMENT_REF_CASE == 5
    auto elem = arr[0]; // as the right-hand side of an assignment
    arr[1]    = elem;
    #elif ELEMENT_REF_CASE == 6
    auto elem = arr.at(0); // at() too
    ++elem;
    #elif ELEMENT_REF_CASE == 7
    auto elem = arr[0]; // comparison
    if (elem == 1.0)
    {
        println("1");
    }
    #elif ELEMENT_REF_CASE == 8
    auto elem = arr[0]; // a copy
    auto copy = elem;
    (void)copy;
    #elif ELEMENT_REF_CASE == 9
    std::printf("%f\n", arr[0]); // the proxy through `...`, not the value (GCC: -Werror=conditionally-supported)
    #else
        #error "ELEMENT_REF_CASE: 1..9"
    #endif
}
#endif
} // namespace

#ifndef ELEMENT_REF_CASE
int main()
{
    ParamChunk::init_indexes(true);
    const int rc = usage();
    release_chunk_pools();
    destroy_block_chunk();
    return rc;
}
#endif
