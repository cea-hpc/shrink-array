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

// The second-value registry of zero_any::Array (tab_cst_val2_holder.h) on struct
// element types. Every chunk j of an array is filled with one value f(j), then
// the array is folded (shrink_to_fit): each chunk compresses to f(j), and the
// registry should hand out exactly one identifier per DISTINCT value, with its
// index never degraded. Every element must also read back as written.
//
//  point3    {double x, y, z}, x the same in every value: a hash of the first
//            8 bytes only sees x, every value lands on one index slot, the
//            probe budget runs out and the index degrades.
//  mat_rho   {int mat; double rho}: the first 8 bytes are mat and the padding
//            after it -- rho, the member that varies, is not hashed at all.
//  padded    {char c; int i}: ONE value written with two different padding
//            contents. Equal by operator== (padding is not part of the value),
//            so one identifier; a hash of the bytes splits them in two.
//  zero_sign {double x, y, z}: y = -0.0 in one chunk, +0.0 in the other.
//            Equal, so one identifier; a hash of the bytes splits them.
//
// The first two must hold in every build (a hash over all the bytes is
// enough). The last two need the member-wise hash, so C++26 reflection
// (build_config::has_reflection): without it they may hand out 2 identifiers,
// a memory cost, not a wrong value.
//
// Exit status 0 when every check holds.

#include <cstdio>  // std::printf
#include <cstring> // std::memset
#include <utility> // std::as_const

#include "chunk/zero_any_array.h" // shrink_array::zero_any::Array, SecondValueStats
#include "chunk/zero_chunk.h"     // release_chunk_pools, destroy_block_chunk
#include "chunk/config.h"         // build_config::has_reflection
#include "chunk/manage_ompt.h"    // parallel_phase

namespace sa = shrink_array;

struct Point3
{
    double x, y, z;
    bool   operator==(const Point3&) const = default;
};

struct MatRho
{
    int    mat;
    double rho;
    bool   operator==(const MatRho&) const = default;
};

struct Padded
{
    char c;
    int  i;
    bool operator==(const Padded&) const = default;
};

struct ZeroSign
{
    double x, y, z;
    bool   operator==(const ZeroSign&) const = default;
};

// A value of T whose padding bytes are all @p fill, then the members set.
template <typename T, typename Init>
static T with_padding(unsigned char fill, Init init)
{
    T v;
    std::memset(static_cast<void*>(&v), fill, sizeof(T));
    init(v);
    return v;
}

/// Fill chunk j with f(j), fold, and check the registry against @p expected
/// identifiers (or at most @p expected_max when @p exact is false).
template <typename T, typename F>
static int run_case(const char* name, int n_chunks, F f, std::size_t expected, bool exact)
{
    int bad = 0;
    {
        sa::zero_any::Array<T> a(0);
        const int              cs = static_cast<int>(a.chunk_size());
        const int              n  = n_chunks * cs;
        a.resize(n);

        sa::parallel_phase(
            [&]
            {
#pragma omp for
                for (int j = 0; j < n_chunks; ++j)
                {
                    const T v = f(j);
                    for (int k = 0; k < cs; ++k)
                    {
                        a[j * cs + k] = v;
                    }
                }
            });
        a.shrink_to_fit(); // sequential point: every uniform chunk compresses

        const auto& ca = std::as_const(a);
        for (int i = 0; i < n; ++i)
        {
            if (!(ca[i] == f(i / cs)))
            {
                ++bad;
            }
        }

        const sa::SecondValueStats s      = sa::zero_any::Array<T>::second_value_stats();
        const bool                 ids_ok = exact ? s.in_use() == expected : s.in_use() <= expected;
        std::printf("%-10s %5d chunks: %5zu identifier(s) (expected %s%zu), index %s, %d wrong value(s)  %s\n", name,
                    n_chunks, s.in_use(), exact ? "" : "<= ", expected, s.degraded ? "DEGRADED" : "ok", bad,
                    (ids_ok && !s.degraded && bad == 0) ? "OK" : "FAIL");
        if (!ids_ok || s.degraded)
        {
            ++bad;
        }
    } // last array of T: the registry is freed
    return bad;
}

int main()
{
    constexpr bool refl = sa::build_config::has_reflection;
    std::printf("member-wise hash (C++26 reflection): %s\n", refl ? "yes" : "no");

    int       bad = 0;
    const int k   = 2000; // distinct values: far past the index's probe budget (64)

    bad += run_case<Point3>("point3", k, [](int j) { return Point3{1.0, static_cast<double>(j + 1), 2.0}; }, k, true);

    bad += run_case<MatRho>(
        "mat_rho", k, [](int j)
        { return with_padding<MatRho>(0, [j](MatRho& v) { v.mat = 1, v.rho = 1.0 + static_cast<double>(j); }); }, k,
        true);

    // One value, two padding contents: chunks 0, 2, 4... and 1, 3, 5...
    bad += run_case<Padded>(
        "padded", 2,
        [](int j) { return with_padding<Padded>(j % 2 == 0 ? 0xAA : 0x55, [](Padded& v) { v.c = 1, v.i = 7; }); },
        refl ? 1 : 2, refl);

    // One value, the signed zero of y alternating.
    bad += run_case<ZeroSign>(
        "zero_sign", 2, [](int j) { return ZeroSign{1.0, j % 2 == 0 ? -0.0 : 0.0, 3.0}; }, refl ? 1 : 2, refl);

    sa::release_chunk_pools();
    sa::destroy_block_chunk();
    return bad == 0 ? 0 : 1;
}
