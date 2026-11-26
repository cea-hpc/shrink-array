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
 * @file host_twin_bench.cpp
 * @brief kokkos_shrink_bench's workload on the host arrays, for the CPU against the GPU: the
 *        same five double arrays, bands, windows (same seed), kernels and phases, by OpenMP
 *        loops on five flat double arrays and on five host arrays of this build's variant.
 *
 * @details examples/kokkos/kokkos_shrink_bench.cu is the reference: every phase here is its
 *          Kokkos kernel written as an OpenMP loop (std:: maths instead of Kokkos::), with the
 *          same timing boundaries and the same report lines, so that the two outputs compare
 *          line by line. What differs, and only that:
 *          - the arrays: flat double arrays (zeroed in parallel: first touch) against
 *            zero::Array / zero_or_value::Array / zero_any::Array<double, std::int64_t> of the
 *            variant and chunk size this target links (one binary per variant and size);
 *          - writes inside parallel_phase() (the library's write phase), reads through
 *            std::as_const in a plain parallel region;
 *          - compress(): every array's audit pass (compress()), in one team, every
 *            `compress every` iterations, as the GPU's; the host arrays also fold as they are
 *            written (their candidates), which the GPU's do not;
 *          - reductions: per thread under schedule(static), then summed in thread order, so a
 *            run's sums are the same bits in the flat run and the shrink one (the GPU's differ
 *            in the last bits: other libm, other order);
 *          - memory: the chunks holding a buffer (written_chunk_count() x chunk_size() x 8),
 *            as the GPU's "held"; the chunk tables and the pools are not counted.
 *
 *          zero_or_value declares the CPU bench's second values (bench_array.h): 1 for arr1 to
 *          arr3, 2 for arr4, 100 for arr5, as kokkos_shrink_bench does.
 *
 *          Usage: host_twin_bench_<variant>_<size> [log2(sz), default 27]
 *                 [iterations, default 2000] [interval, default sz / 10^4]
 *                 [compress every, default 10]
 *          The threads: OMP_NUM_THREADS (OMP_PROC_BIND=close OMP_PLACES=cores on Grace).
 */

#include <algorithm>   // std::max, std::min
#include <array>       // std::array
#include <chrono>      // std::chrono::steady_clock
#include <cmath>       // std::sin, std::cos, std::tan, std::atan, std::sqrt, std::exp, std::fmod
#include <cstdint>     // std::int64_t
#include <cstdio>      // std::printf, std::setvbuf
#include <cstdlib>     // std::atoi, std::atoll
#include <exception>   // std::exception
#include <memory>      // std::unique_ptr, std::make_unique, std::make_unique_for_overwrite
#include <random>      // std::mt19937_64
#include <type_traits> // std::conditional_t
#include <utility>     // std::as_const
#include <vector>      // std::vector

#include <omp.h> // omp_get_max_threads, omp_get_thread_num

#include "chunk/config.h"              // sa::chunk_config::use_zero_any, use_zero_or_value, use_zero
#include "chunk/phases.h"              // parallel_phase
#include "chunk/runtime.h"             // Runtime::init, release_chunk_pools, Runtime::shutdown
#include "chunk/zero_any_array.h"      // zero_any::Array
#include "chunk/zero_array.h"          // zero::Array
#include "chunk/zero_or_value_array.h" // zero_or_value::Array

namespace sa = shrink_array;

namespace
{

using Int = std::int64_t;

constexpr double second_value_d  = 1.;
constexpr double second_value_d0 = 3.5;
constexpr double second_value_d1 = 2;
constexpr double second_value_d2 = 100;

/// This build's host array.
using HostArray = std::conditional_t<sa::chunk_config::use_zero_any,
                                     sa::zero_any::Array<double, Int>,
                                     std::conditional_t<sa::chunk_config::use_zero_or_value,
                                                        sa::zero_or_value::Array<double, Int>,
                                                        sa::zero::Array<double, Int>>>;

[[nodiscard]] const char* variant_name()
{
    return sa::chunk_config::use_zero_any ? "zero_any" : sa::chunk_config::use_zero_or_value ? "zero_or_value" : "zero";
}

/// @brief arr.set_second_value(v) where the variant has one (zero_or_value): a template, so
///        that the other variants' discarded branch is not instantiated.
void set_second(auto& arr, double v)
{
    if constexpr (requires { arr.set_second_value(v, false); })
    {
        arr.set_second_value(v, false);
    }
}

/// @brief Seconds of @p f.
template <typename F>
double timed(F&& f)
{
    const auto t0 = std::chrono::steady_clock::now();
    f();
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

/// @brief One thread's partial sum per slot, summed in thread order (deterministic).
class Partials
{
  public:
    Partials() : p_(static_cast<std::size_t>(omp_get_max_threads()), 0.0) {}
    void set(double v)
    {
        p_[static_cast<std::size_t>(omp_get_thread_num())] = v;
    }
    [[nodiscard]] double sum() const
    {
        double s = 0;
        for (const double v : p_)
        {
            s += v;
        }
        return s;
    }

  private:
    std::vector<double> p_;
};

/// @brief A flat array of doubles, zeroed in parallel (its pages first touched by the threads
///        of the static schedule that later loops use), as Kokkos::View zeroes on the GPU.
class FlatArray
{
  public:
    explicit FlatArray(Int n) : n_(n), p_(std::make_unique_for_overwrite<double[]>(static_cast<std::size_t>(n)))
    {
        double* const p = p_.get();
#pragma omp parallel for schedule(static)
        for (Int i = 0; i < n; ++i)
        {
            p[i] = 0.0;
        }
    }
    [[nodiscard]] double& operator[](Int i) noexcept
    {
        return p_[static_cast<std::size_t>(i)];
    }
    [[nodiscard]] const double& operator[](Int i) const noexcept
    {
        return p_[static_cast<std::size_t>(i)];
    }
    [[nodiscard]] Int size() const noexcept
    {
        return n_;
    }

  private:
    Int                       n_;
    std::unique_ptr<double[]> p_;
};

/// @brief The flat baseline: five flat double arrays.
struct FlatStore
{
    using A = FlatArray;
    std::array<std::unique_ptr<A>, 5> a;

    explicit FlatStore(Int n)
    {
        for (auto& x : a)
        {
            x = std::make_unique<A>(n);
        }
    }
    /// @brief @p body on every thread of a region (its loops are `omp for`).
    template <typename B>
    static void write_phase(B&& body)
    {
#pragma omp parallel
        body();
    }
    template <typename B>
    static void read_phase(B&& body)
    {
#pragma omp parallel
        body();
    }
    static void          compress() {}
    [[nodiscard]] double bytes() const
    {
        return 5.0 * static_cast<double>(a[0]->size()) * sizeof(double);
    }
    [[nodiscard]] static const char* name()
    {
        return "flat double[]";
    }
};

/// @brief Five host arrays of this build's variant.
struct ShrinkStore
{
    using A = HostArray;
    std::array<std::unique_ptr<A>, 5> a;

    explicit ShrinkStore(Int n)
    {
        for (auto& x : a)
        {
            x = std::make_unique<A>(n);
        }
        if constexpr (sa::chunk_config::use_zero_or_value)
        {
            // shrink_bench.cpp's maybe_set_second_value: second_value_d, _d1, _d2.
            constexpr double second[5] = {second_value_d, second_value_d, second_value_d, second_value_d1,
                                          second_value_d2};
            for (int k = 0; k < 5; ++k)
            {
                set_second(*a[static_cast<std::size_t>(k)], second[k]);
            }
        }
    }
    template <typename B>
    static void write_phase(B&& body)
    {
        sa::parallel_phase([&] { body(); });
    }
    template <typename B>
    static void read_phase(B&& body)
    {
#pragma omp parallel
        body();
    }
    /// @brief Every array's audit pass, in one team (bench_phase_compress_all's pattern: every
    ///        thread of the region calls compress(), a team-wide pass that shares the chunks).
    void compress() const
    {
#pragma omp parallel
        for (const auto& x : a)
        {
            x->compress();
        }
    }
    [[nodiscard]] double bytes() const
    {
        double b = 0;
        for (const auto& x : a)
        {
            b += static_cast<double>(x->written_chunk_count()) * static_cast<double>(x->chunk_size()) * sizeof(double);
        }
        return b;
    }
    [[nodiscard]] static const char* name()
    {
        return sa::chunk_config::use_zero_any        ? "zero_any::Array<double>"
               : sa::chunk_config::use_zero_or_value ? "zero_or_value::Array<double>"
                                                     : "zero::Array<double>";
    }
};

/// @brief One iteration's window and thresholds (kokkos_shrink_bench's, same seed).
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

struct Times
{
    double alloc = 0, fill = 0, init_compress = 0, around = 0, rewrite = 0, reads = 0, rmw = 0, reset = 0, compress = 0;
};

/// @brief kokkos_shrink_bench's fill_value: array @p a's value at @p i after the init bands.
double fill_value(int a, Int i, Int sz, Int interval)
{
    const Int b1 = sz / 2, b2 = sz - (sz / 3), b3 = sz - (sz / 4);
    const Int b4 = b3 + interval < sz ? b3 + interval : sz;
    const Int b5 = b4 + interval < sz ? b4 + interval : sz;
    if (i < 256)
    {
        const bool   third = i % 3 == 0;
        const double s     = std::sin(static_cast<double>(i));
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

template <typename Store>
void fill(Store& st, Int sz, Int interval)
{
    for (int k = 0; k < 5; ++k)
    {
        auto& x = *st.a[static_cast<std::size_t>(k)];
        Store::write_phase(
            [&]
            {
#pragma omp for schedule(static)
                for (Int i = 0; i < sz; ++i)
                {
                    x[i] = fill_value(k, i, sz, interval);
                }
            });
    }
}

template <typename Store>
void around(Store& st, Int sz, Int k, const Window& win)
{
    auto&      a1    = *st.a[0];
    auto&      a2    = *st.a[1];
    auto&      a3    = *st.a[2];
    auto&      a4    = *st.a[3];
    auto&      a5    = *st.a[4];
    const Int  dec   = k % 16;
    const Int  begin = std::max<Int>(win.start0 - 512, 0);
    const Int  s0    = win.start0;
    const Int  e0    = win.end0;
    const Int  after = std::min(e0 + 256, sz);
    const auto kd    = static_cast<double>(k);
    Store::write_phase(
        [&]
        {
#pragma omp for schedule(static)
            for (Int i = begin; i < s0; ++i)
            {
                const Int    offset = i - begin;
                const double x      = static_cast<double>(i) + kd;
                if (offset < 60 + dec)
                {
                    a1[i] = 0.0;
                    a2[i] = second_value_d;
                    a3[i] = std::tan(x);
                    a4[i] = 0.0;
                    a5[i] = std::sin(x);
                }
                else if (offset < 120 + dec)
                {
                    a1[i] = second_value_d;
                    a2[i] = 0.0;
                    a3[i] = 0.0;
                    a4[i] = second_value_d1;
                    a5[i] = 0.0;
                }
                else if (offset < 240 + dec)
                {
                    a1[i] = 0.0;
                    a2[i] = std::sin(x);
                    a3[i] = second_value_d;
                    a4[i] = 0.0;
                    a5[i] = second_value_d2;
                }
                else if (offset < 360 + dec)
                {
                    a1[i] = std::cos(x);
                    a2[i] = 0.0;
                    a3[i] = 0.0;
                    a4[i] = 0.0;
                    a5[i] = 0.0;
                }
                else
                {
                    a1[i] = 0.0;
                    a2[i] = second_value_d;
                    a3[i] = second_value_d;
                    a4[i] = second_value_d1;
                    a5[i] = second_value_d2;
                }
            }
        // The GPU's second kernel: a loop of its own after the first one's barrier.
#pragma omp for schedule(static)
            for (Int i = e0; i < after; ++i)
            {
                const double x = static_cast<double>(i);
                if (i % 3 == 0)
                {
                    a1[i] = second_value_d;
                    a2[i] = std::sin(x + kd);
                    a3[i] = 0.0;
                    a4[i] = second_value_d1;
                    a5[i] = std::cos(x - kd);
                }
                else
                {
                    a1[i] = 0.0;
                    a2[i] = 0.0;
                    a3[i] = std::sin(x + kd);
                    a4[i] = 0.0;
                    a5[i] = std::cos(x + (2 * kd));
                }
            }
        });
}

template <typename Store>
void rewrite(Store& st, Int k, const Window& win)
{
    auto&     a1 = *st.a[0];
    auto&     a2 = *st.a[1];
    auto&     a3 = *st.a[2];
    auto&     a4 = *st.a[3];
    auto&     a5 = *st.a[4];
    const Int t1 = win.t1, t2 = win.t2;
    const Int s0 = win.start0, e0 = win.end0;
    if (k % 4 != 0)
    {
        Store::write_phase(
            [&]
            {
#pragma omp for schedule(static)
                for (Int i = s0; i < e0; ++i)
                {
                    const double x = static_cast<double>(i);
                    a1[i]          = std::sqrt(x);
                    if (i > t1)
                    {
                        a2[i] = std::cos(2 * x);
                    }
                    if (i > t2)
                    {
                        const double v1 = static_cast<double>(a1[i]);
                        const double v2 = static_cast<double>(a2[i]);
                        a3[i]           = std::sin(v1 + v2);
                        a4[i]           = static_cast<double>(a4[i]) * (static_cast<double>(a5[i]) + 1);
                    }
                }
            });
    }
    else
    {
        Store::write_phase(
            [&]
            {
#pragma omp for schedule(static)
                for (Int i = s0; i < e0; ++i)
                {
                    a2[i] = .5;
                    a3[i] = 2;
                    a1[i] = static_cast<double>(a2[i]) * static_cast<double>(a3[i]);
                    a2[i] = second_value_d;
                    a3[i] = second_value_d;
                    a4[i] = second_value_d1;
                    a5[i] = second_value_d2;
                    a1[i] = static_cast<double>(a1[i]) * second_value_d;
                }
            });
    }
}

template <typename Store>
double reads(Store& st, const Window& win)
{
    const auto& a1  = std::as_const(*st.a[0]);
    const auto& a2  = std::as_const(*st.a[1]);
    const auto& a3  = std::as_const(*st.a[2]);
    const auto& a5  = std::as_const(*st.a[4]);
    const Int   s0  = win.start0;
    const Int   e0  = win.end0;
    double      sum = 0;
    for (int p = 0; p < 20; ++p)
    {
        const double val  = static_cast<double>(p);
        const bool   flag = p % 2 == 0;
        Partials     part;
        Store::read_phase(
            [&]
            {
                double s = 0;
#pragma omp for schedule(static)
                for (Int i = s0; i < e0; ++i)
                {
                    const double t1     = a1[i];
                    const double t2     = a2[i];
                    const double t3     = a3[i];
                    const double t5     = a5[i];
                    const double common = std::cos(t1) + std::sin(t2) + std::atan(t3) + std::cos(val);
                    const double arm    = flag ? common - (std::tan(t3) + std::cos(t5))
                                               : (std::sin(t1) + std::atan(t2) + std::cos(t3) + std::sin(val)) -
                                                     (std::sin(t3) + std::atan(t5));
                    s += common + arm;
                }
                part.set(s);
            });
        sum += part.sum();
    }
    return sum;
}

template <typename Store>
double rmw(Store& st, const Window& win)
{
    auto&     a1    = *st.a[0];
    auto&     a2    = *st.a[1];
    auto&     a3    = *st.a[2];
    auto&     a4    = *st.a[3];
    auto&     a5    = *st.a[4];
    const Int s0    = win.start0;
    const Int e0    = win.end0;
    const Int tenth = (e0 - s0) / 10;
    double    sum   = 0;
    for (int p = 0; p < 2; ++p)
    {
        const double val = static_cast<double>(p);
        Partials     part;
        Store::write_phase(
            [&]
            {
                double s = 0;
#pragma omp for schedule(static)
                for (Int i = s0; i < e0; ++i)
                {
                    const double v1 = static_cast<double>(a1[i]);
                    const double v2 = static_cast<double>(a2[i]);
                    const double v3 = static_cast<double>(a3[i]);
                    s += std::cos(v1) + std::sin(v2) + std::atan(v3) + std::cos(val);
                    s += std::sin(static_cast<double>(a4[i])) * std::cos(static_cast<double>(a5[i]));
                    if (i > e0 - tenth)
                    {
                        const double w1 = (v2 * v3 * std::sin(v1)) + std::cos(val);
                        a1[i]           = w1 + (v2 - std::exp(std::fmod(v3, 700.0)));
                    }
                }
                part.set(s);
            });
        sum += part.sum();
    }
    return sum;
}

template <typename Store>
void reset(Store& st, Int sz, const Window& win)
{
    const Int s0  = win.start0;
    const Int end = std::min(win.end0 + 256, sz);
    Store::write_phase(
        [&]
        {
            for (auto& x : st.a)
            {
                auto& arr = *x;
#pragma omp for schedule(static) nowait
                for (Int i = s0; i < end; ++i)
                {
                    arr[i] = 0.0;
                }
            }
        });
}

template <typename Store>
double checksum(Store& st, Int sz)
{
    const auto& a1 = std::as_const(*st.a[0]);
    const auto& a2 = std::as_const(*st.a[1]);
    const auto& a3 = std::as_const(*st.a[2]);
    const auto& a4 = std::as_const(*st.a[3]);
    const auto& a5 = std::as_const(*st.a[4]);
    Partials    part;
    Store::read_phase(
        [&]
        {
            double s = 0;
#pragma omp for schedule(static)
            for (Int i = 0; i < sz; ++i)
            {
                s += static_cast<double>(a1[i]) + static_cast<double>(a2[i]) + static_cast<double>(a3[i]) +
                     static_cast<double>(a4[i]) + static_cast<double>(a5[i]);
            }
            part.set(s);
        });
    return part.sum();
}

struct Result
{
    Times  t;
    double val_sum;
    double check;
    double gb_after_init;
    double gb_end;
};

template <typename Store>
Result run(Int sz, Int interval, Int iters, Int compress_every, const std::vector<Window>& windows)
{
    std::printf("\n== %s ==\n", Store::name());
    Result                 r{};
    std::unique_ptr<Store> held;
    r.t.alloc         = timed([&] { held = std::make_unique<Store>(sz); });
    Store& st         = *held;
    r.t.fill          = timed([&] { fill(st, sz, interval); });
    r.t.init_compress = timed([&] { st.compress(); });
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
            r.t.compress += timed([&] { st.compress(); });
        }
    }
    r.val_sum = val_sum;
    r.check   = checksum(st, sz);
    r.gb_end  = st.bytes() / 1e9;
    std::printf("  end: %.3f GB held; val_sum %.12g, checksum %.12g\n", r.gb_end, r.val_sum, r.check);
    return r;
}

void line(const char* what, double s, double f)
{
    std::printf("  %-34s %10.3f ms %9.2fx flat\n", what, s * 1e3, f > 0 ? s / f : 0.0);
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
    const bool same = s.check == f.check && s.val_sum == f.val_sum;
    std::printf("  checksums %s the flat run's\n", same ? "equal to" : "DIFFERENT from");
}

} // namespace

int main(int argc, char** argv)
try
{
    (void)std::setvbuf(stdout, nullptr, _IOLBF, 0);
    const int log2sz         = argc > 1 ? std::atoi(argv[1]) : 27;
    const Int sz             = Int{1} << log2sz;
    const Int iters          = argc > 2 ? std::atoll(argv[2]) : 2000;
    const Int interval       = argc > 3 ? std::atoll(argv[3]) : std::max<Int>(sz / 10000, 1000);
    const Int compress_every = argc > 4 ? std::max<Int>(std::atoll(argv[4]), 1) : 10;
    sa::Runtime::init(true);
    const Int chunk = HostArray(0).chunk_size();
    std::printf(
        "host_twin_bench (%s, %d OpenMP threads): sz = %lld doubles x 5 arrays, %lld iterations, interval %lld, "
        "compress() every %lld, chunk size %lld\n",
        variant_name(), omp_get_max_threads(), static_cast<long long>(sz), static_cast<long long>(iters),
        static_cast<long long>(interval), static_cast<long long>(compress_every), static_cast<long long>(chunk));
    const std::vector<Window> windows = make_windows(sz, interval, iters);
    const Result              flat    = run<FlatStore>(sz, interval, iters, compress_every, windows);
    const Result              s       = run<ShrinkStore>(sz, interval, iters, compress_every, windows);
    std::printf("\n-- %s, S = %lld against the flat arrays --\n", variant_name(), static_cast<long long>(chunk));
    report(s, flat);
    sa::release_chunk_pools();
    sa::Runtime::shutdown();
    return 0;
}
catch (const std::exception& e)
{
    std::printf("error: %s\n", e.what());
    return 2;
}
