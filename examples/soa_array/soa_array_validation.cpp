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
//    without a buffer;
//  - nested values: a Cell3 with struct members (split part by part when
//    they are no ArrayChunkElement, or marked soa::split; one column when
//    they are one), array members, a storage annotation inherited by a
//    struct's parts or set inside it -- the columns, their names and types
//    (compile time), a[i].pos.y, a[i].w[1], a[i].in.q[0] through the
//    generated nested Ref, get/set, copy_range / read_range, against a
//    std::vector; a nested struct with a base class or a private member
//    refused (compile time).

#include <algorithm>   // std::fill, std::copy (the reference)
#include <cstdint>     // std::uint8_t (a colour)
#include <type_traits> // std::is_same_v, std::remove_cvref_t (the column types)
#include <utility>     // std::as_const
#include <vector>      // std::vector, the reference

#include "chunk/zero_chunk.h"     // release_chunk_pools, destroy_block_chunk
#include "chunk/manage_ompt.h"    // parallel_phase
#include "chunk/zero_array.h"     // zero::Array (an annotated column)
#include "chunk/soa_array.h"      // shrink_array::SoaArray, shrink_array::soa::plain, soa::column
#include "chunk/internal/print.h" // shrink_array::println

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
                sa::println("particle {} differs", i);
            }
        }
    }
    sa::println("particles: n = {}, {} wrong", n, bad);
    a.for_each_column(
        [](std::string_view name, const auto& col)
        {
            if constexpr (requires { col.stats(); })
            {
                sa::println("  column {:<4} {:5.1f} % of chunks hold a buffer", name, col.stats());
            }
            else
            {
                sa::println("  column {:<4} plain: {} elements, uncompressed", name, col.size());
            }
        });
    if (a.column<^^Particle::mat>().stats() > 7.0 || a.column<^^Particle::flag>().stats() > 1.0 ||
        a.column<^^Particle::rho>().stats() > 5.0)
    {
        sa::println("particles: an annotated compressed column does not compress");
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
            sa::println("ranges: {}, {} wrong", what, w);
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
        sa::println("ranges: a uniform fill left buffers in a compressed column");
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
        sa::println("ranges: size {} after resize and push_back, expected {}", a.size(), v.size());
        ++bad;
    }
    check("resize(n, v), push_back");
    a.resize(static_cast<std::size_t>(n / 2));
    v.resize(static_cast<std::size_t>(n / 2));
    n = static_cast<int>(v.size());
    check("resize(n)");
    sa::println("ranges: {} wrong", bad);
    return bad;
}

// Nested values.
struct Vec3 // no operator==: no ArrayChunkElement, split part by part
{
    double x, y, z;
};
struct Rgb // an ArrayChunkElement: one column, unless marked soa::split
{
    std::uint8_t r, g, b;
    bool         operator==(const Rgb&) const = default;
};
struct Inner
{
    int                                           id;
    [[= sa::soa::column<sa::zero::Array>]] double q[2];
};
struct Cell3
{
    Vec3                      pos;   // pos.x, pos.y, pos.z
    [[= sa::soa::plain]] Vec3 vel;   // vel.x, vel.y, vel.z: the annotation goes to every part
    double                    w[2];  // w[0], w[1]
    Rgb                       color; // one column of Rgb
    [[= sa::soa::split]] Rgb  tint;  // tint.r, tint.g, tint.b
    Inner                     in;    // in.id, in.q[0], in.q[1] (zero::Array)
    int                       mat;
};
using Soa3 = sa::SoaArray<Cell3>;
static_assert(Soa3::column_count == 16, "Cell3: 16 columns");
static_assert(Soa3::column_name(0) == "pos.x" && Soa3::column_name(4) == "vel.y" && Soa3::column_name(7) == "w[1]" &&
                  Soa3::column_name(8) == "color" && Soa3::column_name(11) == "tint.b" &&
                  Soa3::column_name(13) == "in.q[0]" && Soa3::column_name(15) == "mat",
              "Cell3: the column names");
template <std::meta::info... P>
using Cell3Column = std::remove_cvref_t<decltype(std::declval<Soa3&>().template column<P...>())>;
static_assert(std::is_same_v<Cell3Column<^^Cell3::pos, ^^Vec3::y>, sa::zero_any::Array<double>>, "pos.y");
static_assert(std::is_same_v<Cell3Column<^^Cell3::vel, ^^Vec3::z>, std::vector<double>>, "vel.z: inherited soa::plain");
static_assert(std::is_same_v<Cell3Column<^^Cell3::color>, sa::zero_any::Array<Rgb>>, "color: one column");
static_assert(std::is_same_v<Cell3Column<^^Cell3::tint, ^^Rgb::g>, sa::zero_any::Array<std::uint8_t>>, "tint.g: split");
static_assert(std::is_same_v<Cell3Column<^^Cell3::in, ^^Inner::id>, sa::zero_any::Array<int>>, "in.id");
static_assert(std::is_same_v<std::remove_cvref_t<decltype(std::declval<Soa3&>().template column_at<14>())>,
                             sa::zero::Array<double>>,
              "in.q[1]: the annotation inside the struct");
static_assert(std::meta::identifier_of(
                  std::meta::nonstatic_data_members_of(^^Soa3::Ref, std::meta::access_context::unchecked())[5]) == "in",
              "the Ref carries Cell3's member names");

struct Base
{
    int b;
};
struct Derived : Base
{
    int d;
};
struct WithDerived
{
    Derived v;
};
struct Hidden
{
    int a;

  private:
    int h;
};
struct WithHidden
{
    Hidden v;
    int    k;
};
static_assert(!sa::soa_detail::unsupported(^^WithDerived).empty(), "a nested struct with a base class is refused");
static_assert(!sa::soa_detail::unsupported(^^WithHidden).empty(), "a nested struct with a private member is refused");

static Cell3 expected3(int i, int n)
{
    Cell3        c{};
    const double d = static_cast<double>(i);
    c.pos          = Vec3{d, i < n / 2 ? 0.0 : 1.0, 2.0};
    c.vel          = Vec3{0.1 * d, -0.1 * d, 0.0};
    c.w[0]         = i % 3 == 0 ? 1.0 : 0.0;
    c.w[1]         = 7.0;
    c.color        = Rgb{static_cast<std::uint8_t>(i / (n / 4)), 0, 255};
    c.tint         = Rgb{1, static_cast<std::uint8_t>(i % 2), 3};
    c.in.id        = i < n / 8 ? 9 : 0;
    c.in.q[0]      = 0.0;
    c.in.q[1]      = i % 1000 == 0 ? 4.0 : 0.0;
    c.mat          = 2;
    return c;
}

static bool same3(const Cell3& a, const Cell3& b)
{
    return a.pos.x == b.pos.x && a.pos.y == b.pos.y && a.pos.z == b.pos.z && a.vel.x == b.vel.x && a.vel.y == b.vel.y &&
           a.vel.z == b.vel.z && a.w[0] == b.w[0] && a.w[1] == b.w[1] && a.color == b.color && a.tint == b.tint &&
           a.in.id == b.in.id && a.in.q[0] == b.in.q[0] && a.in.q[1] == b.in.q[1] && a.mat == b.mat;
}

static int nested(int n)
{
    int                bad = 0;
    std::vector<Cell3> v(static_cast<std::size_t>(n));
    Soa3               a(static_cast<std::size_t>(n));
    sa::parallel_phase(
        [&]
        {
#pragma omp for
            for (int i = 0; i < n; ++i)
            {
                const Cell3 c                  = expected3(i, n);
                v[static_cast<std::size_t>(i)] = c;
                const std::size_t k            = static_cast<std::size_t>(i);
                if (i % 2 == 0)
                {
                    a[k].pos.x   = c.pos.x; // through the nested Ref
                    a[k].pos.y   = c.pos.y;
                    a[k].pos.z   = c.pos.z;
                    a[k].vel.x   = c.vel.x;
                    a[k].vel.y   = c.vel.y;
                    a[k].vel.z   = c.vel.z;
                    a[k].w[0]    = c.w[0];
                    a[k].w[1]    = c.w[1];
                    a[k].color   = c.color;
                    a[k].tint.r  = c.tint.r;
                    a[k].tint.g  = c.tint.g;
                    a[k].tint.b  = c.tint.b;
                    a[k].in.id   = c.in.id;
                    a[k].in.q[0] = c.in.q[0];
                    a[k].in.q[1] = c.in.q[1];
                    a[k].mat     = c.mat;
                }
                else
                {
                    a.set(k, c);
                }
            }
        });
    sa::parallel_phase(
        [&]
        {
#pragma omp for
            for (int i = 0; i < n; i += 5)
            {
                a[static_cast<std::size_t>(i)].w[1] += 1.0; // a compound op on an array part
                v[static_cast<std::size_t>(i)].w[1] += 1.0;
            }
        });
    a.shrink_to_fit();
    std::vector<Cell3> out(static_cast<std::size_t>(n));
    read_range(a, 0, static_cast<std::size_t>(n), out.begin());
    int w = 0;
    for (int i = 0; i < n; ++i)
    {
        const std::size_t k = static_cast<std::size_t>(i);
        w += same3(std::as_const(a)[k], v[k]) && same3(out[k], v[k]) ? 0 : 1;
    }
    // the way in from structs, then back
    std::vector<Cell3> src(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i)
    {
        src[static_cast<std::size_t>(i)] = expected3(n - 1 - i, n);
    }
    sa::copy_range(a, 0, static_cast<std::size_t>(n), src.begin());
    read_range(a, 0, static_cast<std::size_t>(n), out.begin());
    for (int i = 0; i < n; ++i)
    {
        w += same3(out[static_cast<std::size_t>(i)], src[static_cast<std::size_t>(i)]) ? 0 : 1;
    }
    sa::println("nested: n = {}, {} columns, {} wrong", n, Soa3::column_count, w);
    a.for_each_column(
        [](std::string_view name, const auto& col)
        {
            if constexpr (requires { col.stats(); })
            {
                sa::println("  column {:<8} {:5.1f} % of chunks hold a buffer", name, col.stats());
            }
            else
            {
                sa::println("  column {:<8} plain", name);
            }
        });
    if (a.column<^^Cell3::mat>().stats() > 1.0 || a.column<^^Cell3::pos, ^^Vec3::z>().stats() > 1.0 ||
        a.column_at<11>().stats() > 1.0)
    {
        sa::println("nested: a constant column holds buffers");
        ++w;
    }
    return w + bad;
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
        sa::println("n = {} elements, {} wrong", n, bad);
        a.for_each_column([](std::string_view name, const auto& col)
                          { sa::println("  column {:<5} {:5.1f} % of chunks hold a buffer", name, col.stats()); });

        const double mat_buf  = a.column<^^Cell::mat>().stats();
        const double flag_buf = a.column<^^Cell::flag>().stats();
        const double rho_buf  = a.column<^^Cell::rho>().stats();
        if (mat_buf > 1.0 || flag_buf > 1.0 || rho_buf > 26.0)
        {
            sa::println("compression below expectations: mat {:.1f} %, flag {:.1f} %, rho {:.1f} %", mat_buf, flag_buf,
                        rho_buf);
            ++bad;
        }
    } // the columns release their chunks here

    bad += particles(1 << 20);
    bad += ranges(1 << 20);
    bad += nested(1 << 18);

    sa::release_chunk_pools();
    sa::destroy_block_chunk();
    return bad == 0 ? 0 : 1;
}
