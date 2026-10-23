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

// SoaArray<Cell> (chunk/soa_array.h) against a std::vector<Cell>, the same
// writes on both. Cell's members repeat on their own: the material id is
// constant over large regions, the temperature everywhere but in a hot zone,
// the flag mostly false, while the density varies in a quarter of the domain.
// A zero_any::Array<Cell> would compress only where the whole Cell repeats; the
// SoaArray compresses each column on its own runs.
//
// Checks, exit status 0 when all hold:
//  - every member of every element reads back as in the std::vector, after
//    parallel writes (parallel_phase), compound ops (+=), set() and a
//    shrink_to_fit();
//  - each column compresses as its own data allows (mat and flag: almost no
//    buffer; rho: at most the varied quarter);
//  - the generated Ref carries T's member names (a compile-time check);
//  - storage annotations (C++26, P3394): a Particle whose members pick their
//    own columns -- soa::plain (a std::vector), soa::column<zero::Array>,
//    soa::column<zero_any::Array>, the default for the rest, a foreign
//    annotation left alone -- gets those column types (compile time), reads
//    back as a std::vector after parallel writes, compound ops and set(), and
//    its compressed columns compress; two storage annotations on one member
//    are refused (compile time);
//  - the range operations, member by member: fill_range, copy_range (from a
//    std::vector<Particle>: AoS to SoA), their _mt forms from every thread of
//    a phase, read_range back into structs (a compressed chunk as a fill),
//    a range past the end clamped; resize() with and without a value,
//    push_back(); a uniform fill leaves its compressed columns' chunks
//    without a buffer.

#include <algorithm>   // std::fill, std::copy (the reference)
#include <cstdio>      // std::printf
#include <type_traits> // std::is_same_v, std::remove_cvref_t (the column types)
#include <utility>     // std::as_const
#include <vector>      // std::vector, the reference

#include "chunk/chunk_std.h"   // release_chunk_pools, destroy_block_chunk
#include "chunk/manage_ompt.h" // parallel_phase
#include "chunk/array_chunk.h" // zero::Array (an annotated column)
#include "chunk/soa_array.h"   // shrink_array::SoaArray, shrink_array::soa::plain, soa::column

#if !SHRINK_ARRAY_HAS_SOA_ARRAY
    #error "soa_array_validation needs C++26 static reflection (GCC 16, -freflection): CMake builds it only then"
#endif

namespace sa = shrink_array;

struct Cell
{
    double rho;
    double temp;
    int    mat;
    bool   flag;
};

// The Ref of a[i] has Cell's member names, in Cell's order.
static_assert(
    std::meta::identifier_of(std::meta::nonstatic_data_members_of(^^sa::SoaArray<Cell>::Ref,
                                                                  std::meta::access_context::unchecked())[2]) == "mat");

// Members that pick their own storage. x never repeats: no chunks at all.
struct Particle
{
    [[= sa::soa::plain]] double                            x;
    [[= sa::soa::column<sa::zero::Array>]] int             mat;
    double                                                 rho;
    [[ = sa::soa::column<sa::zero_any::Array>, = 7 ]] bool flag; // 7: another library's annotation, ignored
};

template <std::meta::info M>
using ParticleColumn = std::remove_cvref_t<decltype(std::declval<sa::SoaArray<Particle>&>().template column<M>())>;
static_assert(std::is_same_v<ParticleColumn<^^Particle::x>, std::vector<double>>, "soa::plain: a std::vector");
static_assert(std::is_same_v<ParticleColumn<^^Particle::mat>, sa::zero::Array<int>>, "soa::column<zero::Array>");
static_assert(std::is_same_v<ParticleColumn<^^Particle::rho>, sa::zero_any::Array<double>>,
              "no annotation: the default");
static_assert(std::is_same_v<ParticleColumn<^^Particle::flag>, sa::zero_any::Array<bool>>,
              "soa::column<zero_any::Array>");

// Two storage annotations on one member: refused.
struct Twice
{
    [[ = sa::soa::plain, = sa::soa::column<sa::zero::Array> ]] int m;
};
static_assert(!sa::soa_detail::unsupported(^^Twice).empty(), "two storage annotations on one member are refused");
static_assert(sa::soa_detail::unsupported(^^Particle).empty(), "Particle is supported");

static Particle expected_particle(int i, int n)
{
    Particle p{};
    p.x    = 0.5 * static_cast<double>(i) + 1e-3 * static_cast<double>(i % 7); // never repeats
    p.mat  = (i >= n / 2 && i < n / 2 + n / 16) ? 5 : 0;                       // zero::Array compresses its zeros
    p.rho  = (i >= n / 4 && i < n / 4 + n / 32) ? 2.5 : 1.0;
    p.flag = i % (n / 4) == 0;
    return p;
}

static int particles(int n)
{
    int                    bad = 0;
    std::vector<Particle>  v(n);
    sa::SoaArray<Particle> a(n);
    sa::parallel_phase(
        [&]
        {
#pragma omp for
            for (int i = 0; i < n; ++i)
            {
                const Particle p = expected_particle(i, n);
                v[i]             = p;
                if (i % 2 == 0)
                {
                    a[i].x    = p.x; // a double& into the std::vector
                    a[i].mat  = p.mat;
                    a[i].rho  = p.rho;
                    a[i].flag = p.flag;
                }
                else
                {
                    a.set(i, p);
                }
            }
        });
    sa::parallel_phase(
        [&]
        {
#pragma omp for
            for (int i = 0; i < n; i += 3)
            {
                a[i].x += 2.0;
                v[i].x += 2.0;
            }
        });
    a.shrink_to_fit();
    const auto& ca = std::as_const(a);
    for (int i = 0; i < n; ++i)
    {
        const Particle p = ca[i];
        if (!(p.x == v[i].x && p.mat == v[i].mat && p.rho == v[i].rho && p.flag == v[i].flag))
        {
            if (bad++ < 5)
            {
                std::printf("particle %d differs\n", i);
            }
        }
    }
    std::printf("particles: n = %d, %d wrong\n", n, bad);
    a.for_each_column(
        [](std::string_view name, const auto& col)
        {
            if constexpr (requires { col.stats(); })
            {
                std::printf("  column %-4.*s %5.1f %% of chunks hold a buffer\n", static_cast<int>(name.size()),
                            name.data(), col.stats());
            }
            else
            {
                std::printf("  column %-4.*s plain: %zu elements, uncompressed\n", static_cast<int>(name.size()),
                            name.data(), col.size());
            }
        });
    if (a.column<^^Particle::mat>().stats() > 7.0 || a.column<^^Particle::flag>().stats() > 1.0 ||
        a.column<^^Particle::rho>().stats() > 5.0)
    {
        std::printf("particles: an annotated compressed column does not compress\n");
        ++bad;
    }
    return bad;
}

static bool same_particle(const Particle& p, const Particle& q)
{
    return p.x == q.x && p.mat == q.mat && p.rho == q.rho && p.flag == q.flag;
}

static int ranges(int n)
{
    int                    bad = 0;
    std::vector<Particle>  v(static_cast<std::size_t>(n));
    sa::SoaArray<Particle> a(static_cast<std::size_t>(n));
    const auto             check = [&](const char* what)
    {
        std::vector<Particle> out(static_cast<std::size_t>(n), Particle{-1.0, -1, -1.0, true});
        const auto            end = read_range(a, 0, static_cast<std::size_t>(n), out.begin());
        int                   w   = end == out.end() ? 0 : 1;
        for (int i = 0; i < n; ++i)
        {
            w += same_particle(out[static_cast<std::size_t>(i)], v[static_cast<std::size_t>(i)]) &&
                         same_particle(std::as_const(a)[static_cast<std::size_t>(i)], v[static_cast<std::size_t>(i)])
                     ? 0
                     : 1;
        }
        if (w != 0)
        {
            std::printf("ranges: %s, %d wrong\n", what, w);
            ++bad;
        }
    };

    // Serial: a uniform fill over whole chunks, then a copy from structs.
    const Particle u{3.5, 7, 2.0, true};
    sa::fill_range(a, static_cast<std::size_t>(n / 8), static_cast<std::size_t>(n / 4), u);
    std::fill(v.begin() + n / 8, v.begin() + n / 4, u);
    check("fill_range");
    if (a.column<^^Particle::rho>().stats() > 1.0 || a.column<^^Particle::flag>().stats() > 1.0)
    {
        std::printf("ranges: a uniform fill left buffers in a compressed column\n");
        ++bad;
    }
    std::vector<Particle> src(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i)
    {
        src[static_cast<std::size_t>(i)] = expected_particle(i, n);
    }
    sa::copy_range(a, static_cast<std::size_t>(n / 2), static_cast<std::size_t>(n) + 100, src.begin()); // clamped
    std::copy(src.begin(), src.begin() + (n - n / 2), v.begin() + n / 2);
    check("copy_range");

    // _mt: every thread its blocks, in a phase.
    const int block = 4096;
    sa::parallel_phase(
        [&]
        {
#pragma omp for schedule(static)
            for (int b = 0; b < n / block; ++b)
            {
                const auto lo = static_cast<std::size_t>(b) * block;
                if (b % 3 == 0)
                {
                    sa::fill_range_mt(a, lo, lo + block, Particle{1.0 * b, b % 5, 0.5, false});
                }
                else
                {
                    sa::copy_range_mt(a, lo, lo + block, src.begin() + static_cast<std::ptrdiff_t>(lo));
                }
            }
        });
    for (int b = 0; b < n / block; ++b)
    {
        const auto lo = static_cast<std::size_t>(b) * block;
        for (std::size_t i = lo; i < lo + block; ++i)
        {
            v[i] = b % 3 == 0 ? Particle{1.0 * b, b % 5, 0.5, false} : src[i];
        }
    }
    check("fill_range_mt, copy_range_mt");

    // resize, push_back.
    a.resize(static_cast<std::size_t>(n) + 1000, u);
    v.resize(static_cast<std::size_t>(n) + 1000, u);
    a.push_back(src[5]);
    v.push_back(src[5]);
    n = static_cast<int>(v.size());
    if (a.size() != v.size())
    {
        std::printf("ranges: size %zu after resize and push_back, expected %zu\n", a.size(), v.size());
        ++bad;
    }
    check("resize(n, v), push_back");
    a.resize(static_cast<std::size_t>(n / 2));
    v.resize(static_cast<std::size_t>(n / 2));
    n = static_cast<int>(v.size());
    check("resize(n)");
    std::printf("ranges: %d wrong\n", bad);
    return bad;
}

// What element i holds after the parallel write.
static Cell expected(int i, int n)
{
    Cell c{};
    c.rho  = i < n / 4 ? 1.0 + 0.001 * static_cast<double>(i % 1000) : 1.0; // varied in the first quarter
    c.temp = (i >= n / 2 && i < n / 2 + n / 16) ? 900.0 : 300.0;            // a hot zone
    c.mat  = i < n / 3 ? 1 : (i < 2 * (n / 3) ? 2 : 3);                     // three material regions
    c.flag = (i % (n / 8)) == 0;                                            // a few isolated flags
    return c;
}

static bool same(const Cell& a, const Cell& b)
{
    return a.rho == b.rho && a.temp == b.temp && a.mat == b.mat && a.flag == b.flag;
}

int main()
{
    const int n   = 1 << 22;
    int       bad = 0;
    {
        std::vector<Cell>  v(n);
        sa::SoaArray<Cell> a(n);

        // Parallel write: member by member through the generated Ref, and the
        // whole struct through set().
        sa::parallel_phase(
            [&]
            {
#pragma omp for
                for (int i = 0; i < n; ++i)
                {
                    const Cell& c = expected(i, n);
                    v[i]          = c;
                    if (i % 2 == 0)
                    {
                        a[i].rho  = c.rho;
                        a[i].temp = c.temp;
                        a[i].mat  = c.mat;
                        a[i].flag = c.flag;
                    }
                    else
                    {
                        a.set(i, c);
                    }
                }
            });

        // A compound op on one member, in another region.
        sa::parallel_phase(
            [&]
            {
#pragma omp for
                for (int i = n / 2; i < n / 2 + n / 16; ++i)
                {
                    a[i].temp += 100.0;
                    v[i].temp += 100.0;
                }
            });

        a.shrink_to_fit(); // sequential point, every column

        const auto& ca = std::as_const(a);
#pragma omp parallel for reduction(+ : bad)
        for (int i = 0; i < n; ++i)
        {
            if (!same(ca[i], v[i]))
            {
                ++bad;
            }
        }

        // Per column: the share of chunks that hold a buffer.
        std::printf("n = %d elements, %d wrong\n", n, bad);
        a.for_each_column(
            [](std::string_view name, const auto& col)
            {
                std::printf("  column %-5.*s %5.1f %% of chunks hold a buffer\n", static_cast<int>(name.size()),
                            name.data(), col.stats());
            });

        const double mat_buf  = a.column<^^Cell::mat>().stats();
        const double flag_buf = a.column<^^Cell::flag>().stats();
        const double rho_buf  = a.column<^^Cell::rho>().stats();
        if (mat_buf > 1.0 || flag_buf > 1.0 || rho_buf > 26.0)
        {
            std::printf("compression below expectations: mat %.1f %%, flag %.1f %%, rho %.1f %%\n", mat_buf, flag_buf,
                        rho_buf);
            ++bad;
        }
    } // the columns release their chunks here

    bad += particles(1 << 20);
    bad += ranges(1 << 20);

    sa::release_chunk_pools();
    sa::destroy_block_chunk();
    return bad == 0 ? 0 : 1;
}
