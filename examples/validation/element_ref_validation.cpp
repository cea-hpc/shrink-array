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

#include "chunk/array_chunk.h"    // zero::Array
#include "chunk/array_chunk2.h"   // zero_one_value::Array
#include "chunk/array_chunk2A.h"  // zero_any::Array
#include "chunk/config.h"         // chunk_config::use_chunk2*, build_config::strict_element_ref
#include "chunk/param_chunk.h"    // ParamChunk::init_indexes
#include "chunk/chunk_std.h"      // release_chunk_pools, destroy_block_chunk
#include "chunk/internal/print.h" // shrink_array::println

using namespace shrink_array;

namespace
{
template <typename T>
using Array =
    std::conditional_t<chunk_config::use_chunk2a,
                       zero_any::Array<T>,
                       std::conditional_t<chunk_config::use_chunk2, zero_one_value::Array<T>, zero::Array<T>>>;

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
    constexpr int n = 1000;
    Array<double> a(n);
    Array<double> b(n);

    a[1] = 2.0;
    expect("a[1] = 2.0", static_cast<double>(a[1]), 2.0);
    a[2] = a[1]; // proxy-to-proxy, same array
    expect("a[2] = a[1]", static_cast<double>(a[2]), 2.0);
    b[3] = a[2]; // proxy-to-proxy, another array
    expect("b[3] = a[2]", static_cast<double>(b[3]), 2.0);
    a[4] += 1.5;
    a[4] *= 2.0;
    expect("a[4] += 1.5; *= 2", static_cast<double>(a[4]), 3.0);
    const double x = a[1] + a[4];
    expect("a[1] + a[4]", x, 5.0);
    const double y = -a[4] * 2.0;
    expect("-a[4] * 2", y, -6.0);
    ++a[5];
    a[5]++;
    --a[5];
    expect("++ / -- on a[5]", static_cast<double>(a[5]), 1.0);
    const double r = (a[6] += 4.0); // a compound result read as a value
    expect("(a[6] += 4) read", r, 4.0);
    expect("a[1] == 2", a[1] == 2.0, true);
    expect("2 == a[1]", 2.0 == a[1], true);
    expect("a[1] != a[4]", a[1] != a[4], true);
    expect("a[1] == b[3]", a[1] == b[3], true);

    std::vector<double> v;
    v.push_back(a[1]);
    v.emplace_back(a[4]);
    expect("push_back / emplace_back", v[0] + v[1], 5.0);

    std::ostringstream os;
    os << a[1];
    expect("os << a[1]", os.str() == "2", true);

    a.at(7)   = 7.0;
    a.front() = 1.0;
    a.back()  = 9.0;
    expect("at / front / back", static_cast<double>(a[7]) + a[0] + a[n - 1], 17.0);

    auto&& kept     = a[8]; // a proxy kept on purpose
    std::move(kept) = 8.0;
    expect("std::move(kept) = 8", static_cast<double>(a[8]), 8.0);
    auto p = a.element_proxy(9); // the proxy itself, in every build
    p      = 9.0;
    expect("element_proxy(9) = 9", static_cast<double>(a[9]), 9.0);

    Array<std::complex<double>> c(100);
    c[1] = std::complex<double>(1.0, 2.0);
    c[2] = -c[1];
    c[3] = 2.0 * c[1];
    c[4] = std::complex<double>(0.0, 1.0) * c[1];
    c[5] = c[1];
    expect("complex -c, 2*c, i*c, c = c",
           std::complex<double>(c[2]) + std::complex<double>(c[3]) + std::complex<double>(c[4]) +
               std::complex<double>(c[5]),
           std::complex<double>(0.0, 5.0)); // (-1-2i) + (2+4i) + (-2+i) + (1+2i)

    Array<int> k(100);
    k[1] = 7;
    k[2] = k[1] % 3;
    k[3] = ~k[1];
    k[4] = 1;
    k[4] <<= 3;
    expect("int %, ~, <<=", k[2] + k[3] + k[4], 1 - 8 + 8);

    Array<bool> t(100);
    t[1]          = true;
    t[2]          = t[1];
    const bool tb = t[2];
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
    Array<double> a(10);
    #if ELEMENT_REF_CASE == 1
    auto   v = a[0]; // conversion
    double d = v;
    (void)d;
    #elif ELEMENT_REF_CASE == 2
    auto v = a[0]; // assignment
    v      = 3.0;
    #elif ELEMENT_REF_CASE == 3
    auto v = a[0]; // arithmetic
    println("{:g}", v + 1.0);
    #elif ELEMENT_REF_CASE == 4
    auto   v = (a[0] += 1.0); // a compound result kept
    double d = v;
    (void)d;
    #elif ELEMENT_REF_CASE == 5
    auto v = a[0]; // as the right-hand side of an assignment
    a[1]   = v;
    #elif ELEMENT_REF_CASE == 6
    auto v = a.at(0); // at() too
    ++v;
    #elif ELEMENT_REF_CASE == 7
    auto v = a[0]; // comparison
    if (v == 1.0)
    {
        println("1");
    }
    #elif ELEMENT_REF_CASE == 8
    auto v = a[0]; // a copy
    auto w = v;
    (void)w;
    #elif ELEMENT_REF_CASE == 9
    std::printf("%f\n", a[0]); // the proxy through `...`, not the value (GCC: -Werror=conditionally-supported)
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
