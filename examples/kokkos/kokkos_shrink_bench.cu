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

/**
 * @file kokkos_shrink_bench.cu
 * @brief shrink_bench's access pattern (examples/shrink_bench) for a Kokkos code: the same
 *        kernels on five Kokkos::View<double*> and on five device::kokkos::Array (the facade,
 *        chunk/device/kokkos_adaptor.h), timed phase by phase, memory and checksums compared.
 *
 * @details docs/gpu_design.md section 13, stage 2 on stage 3's facade. From the CPU bench, the
 *          five double arrays arr1..arr5 and their phases, with the same bands and constants:
 *          - init (once): the bands of bench_phase_fill.cpp -- [0, 256) alternating on i % 3,
 *            [256, sz/2) arr1 = arr2 = 1, [sz/2, 2sz/3) (1, 0, 1, 2, 100), [2sz/3, 3sz/4)
 *            (3.5, 3.5, 3.5, 7, 10.5), an interval of read-modify-writes (their results), an
 *            interval of (1, 2, 2, 2, 100), zeros -- by the chunk-aligned loop
 *            (parallel_for_chunks; a flat parallel_for for the views), then compress();
 *          - every iteration k, a random window [start0, end0) of at most `interval` elements
 *            and two thresholds in it (bench_phase_compute.cpp): the 512 elements before the
 *            window in five sub-bands shifted by k % 16, the 256 after it alternating on i % 3,
 *            the window rewritten (sqrt / cos / sin with the thresholds if k % 4 != 0, constants
 *            otherwise), 20 read passes reduced (cos, sin, atan), 2 read-modify-write passes
 *            (the last tenth of the window written); every freq_reset iterations instead, the
 *            window and 256 past it put back to zero (the reset phase's constant ranges);
 *            compress() every `compress_every` iterations.
 *          A write phase is one or more Kokkos kernels through write() (Writer::assign), then
 *          end_write(); a read phase reads through read(). Not ported: the other element
 *          types, struct / mask / conn, the CPU's teams, stress phases, deferred writes,
 *          rechunking and oracle files.
 *
 *          Usage: kokkos_shrink_bench [log2(sz), default 27] [iterations, default 2000]
 *                                     [interval, default sz / 10^4]
 *                                     [compress every, default 10] [variants: zero_any, zero
 *                                     or both, default both] [arenas from 1/N of the chunks,
 *                                     default 64]
 *          The flat run comes first; each run frees its arrays before the next.
 */

#include <algorithm>   // std::max, std::min
#include <cstddef>     // std::size_t
#include <cstdint>     // std::int64_t
#include <cstdio>      // std::printf, std::setvbuf
#include <cstdlib>     // std::atoi, std::atoll
#include <cstring>     // std::strcmp
#include <exception>   // std::exception
#include <optional>    // std::optional
#include <random>      // std::mt19937_64
#include <type_traits> // std::type_identity
#include <vector>      // std::vector

#include <Kokkos_Core.hpp> // Kokkos::View, parallel_for, parallel_reduce, Timer, sin, cos, ...

#include "chunk/device/kokkos_adaptor.h" // device::kokkos::Array

namespace sd = shrink_array::device;
namespace sk = shrink_array::device::kokkos;

namespace
{

using Int = std::int64_t;

/// The arenas start at 1/g_arena_div of the chunks (sixth argument).
Int g_arena_div = 64;

constexpr double second_value_d  = 1.;
constexpr double second_value_d0 = 3.5;
constexpr double second_value_d1 = 2;
constexpr double second_value_d2 = 100;

/// @brief The five arrays' accesses of one phase, captured by value by the kernels.
template <typename A>
struct Five
{
    A a1, a2, a3, a4, a5;
};

/// @brief The flat baseline: five Kokkos::View<double*>.
struct FlatStore
{
    using W = Kokkos::View<double*>;
    using R = Kokkos::View<double*>;

    Kokkos::View<double*> v[5];

    explicit FlatStore(Int n)
    {
        for (auto& x : v)
        {
            x = Kokkos::View<double*>("flat", n); // zeros
        }
    }
    [[nodiscard]] Five<W> write() const
    {
        return {v[0], v[1], v[2], v[3], v[4]};
    }
    void end_write() const
    {
        Kokkos::fence();
    }
    [[nodiscard]] Five<R> read() const
    {
        return {v[0], v[1], v[2], v[3], v[4]};
    }
    unsigned long long compress() const
    {
        return 0;
    }
    [[nodiscard]] double bytes() const
    {
        return 5.0 * static_cast<double>(v[0].extent(0)) * sizeof(double);
    }
    [[nodiscard]] static double reserved()
    {
        return 0;
    }
    [[nodiscard]] static sd::WriteCounters counters()
    {
        return {};
    }
    [[nodiscard]] static const char* name()
    {
        return "Kokkos::View<double*>";
    }
};

/// @brief Five device::kokkos::Array of one chunk size and variant.
template <unsigned int S, sd::Variant V>
struct ShrinkStore
{
    using W = sk::WriteAccess<double, S, V>;
    using R = sd::View<double, S, V>;

    std::vector<sk::Array<double, S, V>*> a;

    /// @details A small arena to start, 1/64 of the chunks: it grows as the phases need, by a
    ///          quarter at least, and what it reserves counts in the memory reported. An
    ///          append-buffer of 2^20 entries.
    explicit ShrinkStore(Int n)
    {
        const auto chunks = static_cast<unsigned long long>((n + S - 1) / S);
        for (int k = 0; k < 5; ++k)
        {
            a.push_back(new sk::Array<double, S, V>(
                n, chunks / static_cast<unsigned long long>(g_arena_div) + 64, 0.0, 1ULL << 20));
        }
    }
    ~ShrinkStore()
    {
        for (auto* x : a)
        {
            delete x;
        }
    }
    ShrinkStore(const ShrinkStore&)            = delete;
    ShrinkStore& operator=(const ShrinkStore&) = delete;

    [[nodiscard]] Five<W> write() const
    {
        return {a[0]->write(), a[1]->write(), a[2]->write(), a[3]->write(), a[4]->write()};
    }
    void end_write() const
    {
        sk::end_write(*a[0], *a[1], *a[2], *a[3], *a[4]); // one fence and one wait for the five
    }
    [[nodiscard]] Five<R> read() const
    {
        return {a[0]->read(), a[1]->read(), a[2]->read(), a[3]->read(), a[4]->read()};
    }
    unsigned long long compress() const
    {
        unsigned long long f = 0;
        for (auto* x : a)
        {
            f += x->compress();
        }
        return f;
    }
    [[nodiscard]] double bytes() const
    {
        double b = 0;
        for (auto* x : a)
        {
            b += static_cast<double>(x->bytes());
        }
        return b;
    }
    /// @brief The write path's counters, summed over the five arrays (claim_wait, claim_spin,
    ///        cas_lost are cumulative).
    [[nodiscard]] sd::WriteCounters counters() const
    {
        sd::WriteCounters t{};
        for (auto* x : a)
        {
            const sd::WriteCounters c = x->native().write_counters();
            t.claim_wait += c.claim_wait;
            t.claim_spin += c.claim_spin;
            t.cas_lost += c.cas_lost;
        }
        return t;
    }

    /// @brief Bytes of the arenas' free buffers: reserved, not held by a chunk.
    [[nodiscard]] double reserved() const
    {
        double b = 0;
        for (auto* x : a)
        {
            b += static_cast<double>(x->arena_bytes()) -
                 static_cast<double>(x->materialised()) * static_cast<double>(S * sizeof(double));
        }
        return b;
    }
    [[nodiscard]] static const char* name()
    {
        return V == sd::Variant::zero ? "device::kokkos::Array, zero" : "device::kokkos::Array, zero_any";
    }
};

/// @brief One iteration's window and thresholds, the same for every run.
struct Window
{
    Int start0;
    Int end0;
    Int t1;
    Int t2;
};

std::vector<Window> make_windows(Int sz, Int interval, Int iters)
{
    std::mt19937_64     rng(2026);
    std::vector<Window> w;
    for (Int k = 0; k < iters; ++k)
    {
        const Int size   = 1 + static_cast<Int>(rng() % static_cast<unsigned long long>(interval));
        const Int start0 = 1 + static_cast<Int>(rng() % static_cast<unsigned long long>(sz - size - 1));
        const Int t1     = start0 + static_cast<Int>(rng() % static_cast<unsigned long long>(size));
        const Int t2     = start0 + static_cast<Int>(rng() % static_cast<unsigned long long>(size));
        w.push_back({start0, start0 + size, t1, t2});
    }
    return w;
}

/// @brief Wall-clock seconds of the phases, summed over the iterations.
struct Times
{
    double alloc = 0, fill = 0, init_compress = 0, around = 0, rewrite = 0, reads = 0, rmw = 0, reset = 0, compress = 0;
};

/**
 * @brief The value of array @p a (0 to 4) at @p i after the init bands (bench_phase_fill.cpp):
 *        its band's constant, sin(i) in [0, 256), the read-modify-write interval's results.
 */
KOKKOS_INLINE_FUNCTION double fill_value(int a, Int i, Int sz, Int interval)
{
    const Int b1 = sz / 2, b2 = sz - (sz / 3), b3 = sz - (sz / 4);
    const Int b4 = b3 + interval < sz ? b3 + interval : sz;
    const Int b5 = b4 + interval < sz ? b4 + interval : sz;
    if (i < 256)
    {
        const bool   third = i % 3 == 0;
        const double s     = Kokkos::sin(static_cast<double>(i));
        const double v[5]  = {third ? second_value_d : 0.0, third ? s : 0.0, third ? 0.0 : s, 0.0, 0.0};
        return v[a];
    }
    const double band[6][5] = {
        {second_value_d, second_value_d, 0.0, 0.0, 0.0},
        {second_value_d, 0.0, second_value_d, second_value_d1, second_value_d2},
        {second_value_d0, second_value_d0, second_value_d0, 2 * second_value_d0, 3 * second_value_d0},
        {2.0 * 2.0, 2.0 / 2.0, 1.0 + 1.0, 1.0 - 1.0, second_value_d2 + 10 - 10}, // the read-modify-writes
        {1.0, 2.0, 2.0, second_value_d1, second_value_d2},
        {0.0, 0.0, 0.0, 0.0, 0.0},
    };
    const int b = i < b1 ? 0 : i < b2 ? 1 : i < b3 ? 2 : i < b4 ? 3 : i < b5 ? 4 : 5;
    return band[b][a];
}

/// @brief The init bands, array by array, by each store's loop over the whole array: a flat
///        parallel_for, or the facade's chunk-aligned parallel_for_chunks -- a regular loop's
///        path, whose spans the arena cannot serve are written again after it grows (through
///        Writer::assign the whole array would need an arena of every chunk it materialises,
///        or an append-buffer as large: a phase cannot grow the arena).
void fill(const FlatStore& st, Int sz, Int interval)
{
    for (int a = 0; a < 5; ++a)
    {
        const auto v = st.v[a];
        Kokkos::parallel_for("fill", sz, KOKKOS_LAMBDA(Int i) { v(i) = fill_value(a, i, sz, interval); });
    }
    Kokkos::fence();
}

template <unsigned int S, sd::Variant V>
void fill(const ShrinkStore<S, V>& st, Int sz, Int interval)
{
    for (int a = 0; a < 5; ++a)
    {
        st.a[static_cast<std::size_t>(a)]->parallel_for_chunks(
            0, sz, KOKKOS_LAMBDA(Int i) { return fill_value(a, i, sz, interval); });
    }
}

/// @brief The writes around the window (bench_phase_compute.cpp, its first two loops).
template <typename Store>
void around(const Store& st, Int sz, Int k, const Window& win)
{
    const auto w     = st.write();
    const Int  dec   = k % 16;
    const Int  begin = std::max<Int>(win.start0 - 512, 0);
    const Int  s0    = win.start0;
    const Int  e0    = win.end0;
    const Int  after = std::min(e0 + 256, sz);
    const auto kd    = static_cast<double>(k);
    Kokkos::parallel_for(
        "before window", Kokkos::RangePolicy<>(begin, s0), KOKKOS_LAMBDA(Int i) {
            const Int    offset = i - begin;
            const double x      = static_cast<double>(i) + kd;
            if (offset < 60 + dec)
            {
                w.a1(i) = 0.0;
                w.a2(i) = second_value_d;
                w.a3(i) = Kokkos::tan(x);
                w.a4(i) = 0.0;
                w.a5(i) = Kokkos::sin(x);
            }
            else if (offset < 120 + dec)
            {
                w.a1(i) = second_value_d;
                w.a2(i) = 0.0;
                w.a3(i) = 0.0;
                w.a4(i) = second_value_d1;
                w.a5(i) = 0.0;
            }
            else if (offset < 240 + dec)
            {
                w.a1(i) = 0.0;
                w.a2(i) = Kokkos::sin(x);
                w.a3(i) = second_value_d;
                w.a4(i) = 0.0;
                w.a5(i) = second_value_d2;
            }
            else if (offset < 360 + dec)
            {
                w.a1(i) = Kokkos::cos(x);
                w.a2(i) = 0.0;
                w.a3(i) = 0.0;
                w.a4(i) = 0.0;
                w.a5(i) = 0.0;
            }
            else
            {
                w.a1(i) = 0.0;
                w.a2(i) = second_value_d;
                w.a3(i) = second_value_d;
                w.a4(i) = second_value_d1;
                w.a5(i) = second_value_d2;
            }
        });
    Kokkos::parallel_for(
        "after window", Kokkos::RangePolicy<>(e0, after), KOKKOS_LAMBDA(Int i) {
            const double x = static_cast<double>(i);
            if (i % 3 == 0)
            {
                w.a1(i) = second_value_d;
                w.a2(i) = Kokkos::sin(x + kd);
                w.a3(i) = 0.0;
                w.a4(i) = second_value_d1;
                w.a5(i) = Kokkos::cos(x - kd);
            }
            else
            {
                w.a1(i) = 0.0;
                w.a2(i) = 0.0;
                w.a3(i) = Kokkos::sin(x + kd);
                w.a4(i) = 0.0;
                w.a5(i) = Kokkos::cos(x + 2 * kd);
            }
        });
    st.end_write();
}

/// @brief The window rewritten (heavy math, or constants one iteration in four).
template <typename Store>
void rewrite(const Store& st, Int k, const Window& win)
{
    const auto w  = st.write();
    const Int  t1 = win.t1, t2 = win.t2;
    if (k % 4 != 0)
    {
        Kokkos::parallel_for(
            "rewrite", Kokkos::RangePolicy<>(win.start0, win.end0), KOKKOS_LAMBDA(Int i) {
                const double x = static_cast<double>(i);
                w.a1(i)        = Kokkos::sqrt(x);
                if (i > t1)
                {
                    w.a2(i) = Kokkos::cos(2 * x);
                }
                if (i > t2)
                {
                    const double a1 = w.a1(i);
                    const double a2 = w.a2(i);
                    w.a3(i)         = Kokkos::sin(a1 + a2);
                    w.a4(i) *= static_cast<double>(w.a5(i)) + 1;
                }
            });
    }
    else
    {
        Kokkos::parallel_for(
            "rewrite constants", Kokkos::RangePolicy<>(win.start0, win.end0), KOKKOS_LAMBDA(Int i) {
                w.a2(i) = .5;
                w.a3(i) = 2;
                w.a1(i) = static_cast<double>(w.a2(i)) * static_cast<double>(w.a3(i));
                w.a2(i) = second_value_d;
                w.a3(i) = second_value_d;
                w.a4(i) = second_value_d1;
                w.a5(i) = second_value_d2;
                w.a1(i) *= second_value_d;
            });
    }
    st.end_write();
}

/// @brief 20 read passes over the window, reduced (a read phase: read()).
template <typename Store>
double reads(const Store& st, const Window& win)
{
    const auto r   = st.read();
    double     sum = 0;
    for (int p = 0; p < 20; ++p)
    {
        const double val  = static_cast<double>(p);
        const bool   flag = p % 2 == 0;
        double       part = 0;
        Kokkos::parallel_reduce(
            "reads", Kokkos::RangePolicy<>(win.start0, win.end0),
            KOKKOS_LAMBDA(Int i, double& s) {
                const double t1     = r.a1(i);
                const double t2     = r.a2(i);
                const double t3     = r.a3(i);
                const double t5     = r.a5(i);
                const double common = Kokkos::cos(t1) + Kokkos::sin(t2) + Kokkos::atan(t3) + Kokkos::cos(val);
                const double arm    = flag ? common - (Kokkos::tan(t3) + Kokkos::cos(t5))
                                           : (Kokkos::sin(t1) + Kokkos::atan(t2) + Kokkos::cos(t3) + Kokkos::sin(val)) -
                                                 (Kokkos::sin(t3) + Kokkos::atan(t5));
                s += common + arm;
            },
            part);
        sum += part;
    }
    return sum;
}

/// @brief 2 read-modify-write passes over the window, reduced; the last tenth rewritten.
template <typename Store>
double rmw(const Store& st, const Window& win)
{
    double     sum   = 0;
    const Int  e0    = win.end0;
    const Int  tenth = (win.end0 - win.start0) / 10;
    for (int p = 0; p < 2; ++p)
    {
        const auto   w    = st.write();
        const double val  = static_cast<double>(p);
        double       part = 0;
        Kokkos::parallel_reduce(
            "rmw", Kokkos::RangePolicy<>(win.start0, win.end0),
            KOKKOS_LAMBDA(Int i, double& s) {
                const double a1 = w.a1(i);
                const double a2 = w.a2(i);
                const double a3 = w.a3(i);
                s += Kokkos::cos(a1) + Kokkos::sin(a2) + Kokkos::atan(a3) + Kokkos::cos(val);
                s += Kokkos::sin(static_cast<double>(w.a4(i))) * Kokkos::cos(static_cast<double>(w.a5(i)));
                if (i > e0 - tenth)
                {
                    w.a1(i) = (a2 * a3 * Kokkos::sin(a1)) + Kokkos::cos(val);
                    w.a1(i) += a2 - Kokkos::exp(Kokkos::fmod(a3, 700.0));
                }
            },
            part);
        st.end_write();
        sum += part;
    }
    return sum;
}

/// @brief The window and 256 past it back to zero (the reset phase's constant ranges).
template <typename Store>
void reset(const Store& st, Int sz, const Window& win)
{
    const auto w   = st.write();
    const Int  end = std::min(win.end0 + 256, sz);
    Kokkos::parallel_for(
        "reset", Kokkos::RangePolicy<>(win.start0, end), KOKKOS_LAMBDA(Int i) {
            w.a1(i) = 0.0;
            w.a2(i) = 0.0;
            w.a3(i) = 0.0;
            w.a4(i) = 0.0;
            w.a5(i) = 0.0;
        });
    st.end_write();
}

/// @brief Seconds of @p f, fenced on both sides.
template <typename F>
double timed(F&& f)
{
    Kokkos::fence();
    Kokkos::Timer t;
    f();
    Kokkos::fence();
    return t.seconds();
}

/// @brief The checksum of a whole array set: the sum of every element of the five arrays.
template <typename Store>
double checksum(const Store& st, Int sz)
{
    const auto r   = st.read();
    double     sum = 0;
    Kokkos::parallel_reduce(
        "checksum", sz, KOKKOS_LAMBDA(Int i, double& s) { s += r.a1(i) + r.a2(i) + r.a3(i) + r.a4(i) + r.a5(i); },
        sum);
    return sum;
}

struct Result
{
    Times  t;
    double val_sum;
    double check;
    double gb_after_init;
    double gb_end;
    double reserved_end;
};

template <typename Store>
Result run(Int sz, Int interval, Int iters, Int compress_every, const std::vector<Window>& windows)
{
    std::printf("\n== %s ==\n", Store::name());
    Result               r{};
    std::optional<Store> held;
    // The allocation timed too: the views' (and their zero-initialisation) against the arenas'
    // first buffers, while the growth of the arenas falls in the fill.
    r.t.alloc         = timed([&] { held.emplace(sz); });
    Store& st         = *held;
    r.t.fill          = timed([&] { fill(st, sz, interval); });
    r.t.init_compress = timed([&] { (void)st.compress(); });
    r.gb_after_init   = st.bytes() / 1e9;
    std::printf("  after init: %.3f GB held\n", r.gb_after_init);
    const Int freq_reset = std::max<Int>(iters / 10, 1);
    double    val_sum    = 0;
    for (Int k = 0; k < iters; ++k)
    {
        const Window& win = windows[static_cast<std::size_t>(k)];
        if (k % freq_reset == 0)
        {
            r.t.reset += timed([&] { reset(st, sz, win); });
        }
        else
        {
            r.t.around += timed([&] { around(st, sz, k, win); });
            r.t.rewrite += timed([&] { rewrite(st, k, win); });
            r.t.reads += timed([&] { val_sum += reads(st, win); });
            r.t.rmw += timed([&] { val_sum += rmw(st, win); });
        }
        if ((k + 1) % compress_every == 0)
        {
            r.t.compress += timed([&] { (void)st.compress(); });
        }
    }
    r.val_sum      = val_sum;
    r.check        = checksum(st, sz);
    r.gb_end       = st.bytes() / 1e9;
    r.reserved_end = st.reserved() / 1e9;
    const sd::WriteCounters wc = st.counters();
    std::printf("  write path: %llu claim waits, %llu given up (appended), %llu lost CAS\n", wc.claim_wait,
                wc.claim_spin, wc.cas_lost);
    std::printf("  end: %.3f GB held, %.3f GB of free arena buffers; val_sum %.12g, checksum %.12g\n", r.gb_end,
                r.reserved_end, r.val_sum, r.check);
    return r;
}

void line(const char* what, double t, double flat)
{
    std::printf("  %-34s %10.3f ms %9.2fx flat\n", what, t * 1e3, flat > 0 ? t / flat : 0.0);
}

void report(const Result& s, const Result& f)
{
    line("init: allocation", s.t.alloc, f.t.alloc);
    line("init: fill (chunk-aligned)", s.t.fill, f.t.fill);
    line("init: allocation + fill", s.t.alloc + s.t.fill, f.t.alloc + f.t.fill);
    line("init: compress()", s.t.init_compress, f.t.init_compress);
    line("loop: writes around the window", s.t.around, f.t.around);
    line("loop: window rewritten", s.t.rewrite, f.t.rewrite);
    line("loop: 20 read passes", s.t.reads, f.t.reads);
    line("loop: 2 read-modify-write passes", s.t.rmw, f.t.rmw);
    line("loop: resets", s.t.reset, f.t.reset);
    line("loop: compress()", s.t.compress, f.t.compress);
    const double ls = s.t.around + s.t.rewrite + s.t.reads + s.t.rmw + s.t.reset + s.t.compress;
    const double lf = f.t.around + f.t.rewrite + f.t.reads + f.t.rmw + f.t.reset + f.t.compress;
    line("loop: total", ls, lf);
    std::printf("  memory at the end: %.3f GB against %.3f GB flat (%.1f%%)\n", s.gb_end, f.gb_end,
                100.0 * s.gb_end / f.gb_end);
    const bool same = s.check == f.check && (s.val_sum == f.val_sum || (s.val_sum - f.val_sum) * (s.val_sum - f.val_sum) <=
                                                                              1e-20 * f.val_sum * f.val_sum);
    std::printf("  checksums %s the flat run's\n", same ? "equal to" : "DIFFERENT from");
}

} // namespace

int main(int argc, char** argv)
{
    (void)std::setvbuf(stdout, nullptr, _IOLBF, 0);
    Kokkos::ScopeGuard guard(argc, argv);
    const int         log2sz         = argc > 1 ? std::atoi(argv[1]) : 27;
    const Int         sz             = Int{1} << log2sz;
    const Int         iters          = argc > 2 ? std::atoll(argv[2]) : 2000;
    const Int         interval       = argc > 3 ? std::atoll(argv[3]) : std::max<Int>(sz / 10000, 1000);
    const Int         compress_every = argc > 4 ? std::max<Int>(std::atoll(argv[4]), 1) : 10;
    const char* const variants       = argc > 5 ? argv[5] : "both";
    g_arena_div                      = argc > 6 ? std::max<Int>(std::atoll(argv[6]), 1) : 64;
    const bool        any            = std::strcmp(variants, "zero") != 0;
    const bool        zero           = std::strcmp(variants, "zero_any") != 0;
    std::printf("kokkos_shrink_bench (%s): sz = %lld doubles x 5 arrays, %lld iterations, interval %lld, "
                "compress() every %lld, arenas from 1/%lld of the chunks\n",
                Kokkos::DefaultExecutionSpace::name(), static_cast<long long>(sz), static_cast<long long>(iters),
                static_cast<long long>(interval), static_cast<long long>(compress_every),
                static_cast<long long>(g_arena_div));
    try
    {
        const std::vector<Window> windows = make_windows(sz, interval, iters);
        const Result              flat    = run<FlatStore>(sz, interval, iters, compress_every, windows);
        const auto                one     = [&](auto tag, const char* title)
        {
            using Store    = typename decltype(tag)::type;
            const Result s = run<Store>(sz, interval, iters, compress_every, windows);
            std::printf("\n-- %s against the flat views --\n", title);
            report(s, flat);
        };
        if (any)
        {
            one(std::type_identity<ShrinkStore<1024, sd::Variant::zero_any>>{}, "zero_any, S = 1024");
        }
        if (zero)
        {
            one(std::type_identity<ShrinkStore<1024, sd::Variant::zero>>{}, "zero, S = 1024");
        }
    }
    catch (const std::exception& e)
    {
        std::printf("error: %s\n", e.what());
        return 2;
    }
    return 0;
}
