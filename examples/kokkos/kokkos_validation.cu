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
 * @file kokkos_validation.cu
 * @brief The Kokkos facade (chunk/device/kokkos_adaptor.h) against a host mirror: every
 *        phase written by Kokkos kernels through the facade only, read back by a Kokkos
 *        kernel, compared element by element.
 *
 * @details Built with -DCHUNK_DEVICE_KOKKOS=ON (Kokkos fetched, compiled for CUDA or HIP as a
 *          CMake language). For S = 256 and 1024, the zero_any and the zero variants, a
 *          partial last chunk: scattered `w(i) = v` in two kernels before one end_write();
 *          `w(i) += 1.5` twice by the owner thread; parallel_for_chunks over a range; a write
 *          of zeros over half the array, then compress() (its folds against the host model);
 *          a parallel_reduce over read(); one phase over two arrays ended by one
 *          end_write(a, b) (one wait for both, device::finish_all).
 */

#include <cstdint> // std::int64_t
#include <cstdio>  // std::printf, std::snprintf
#include <cstring>   // std::memcmp
#include <exception> // std::exception
#include <vector>  // std::vector

#include <Kokkos_Core.hpp> // Kokkos::View, parallel_for, parallel_reduce, deep_copy

#include "chunk/device/kokkos_adaptor.h" // device::kokkos::Array

namespace sd = shrink_array::device;
namespace sk = shrink_array::device::kokkos;

namespace
{

int g_failures = 0;

void expect(bool ok, const char* what)
{
    std::printf("  %-66s %s\n", what, ok ? "ok" : "FAILED");
    g_failures += ok ? 0 : 1;
}

bool same_bits(double a, double b)
{
    return std::memcmp(&a, &b, sizeof a) == 0;
}

/// @brief Element k of a permutation of [0, n): k * p mod n, p a prime not dividing n.
KOKKOS_INLINE_FUNCTION std::int64_t permuted(std::int64_t k, std::int64_t n)
{
    return static_cast<std::int64_t>((static_cast<unsigned long long>(k) * 2654435761ULL) %
                                     static_cast<unsigned long long>(n));
}

/// @brief The value written to element i in round r: repeats of 0.0, -0.0 and 2.5, or varied.
KOKKOS_INLINE_FUNCTION double value_of(std::int64_t i, int r)
{
    switch ((i * 7 + r) % 6)
    {
        case 0:
            return 0.0;
        case 1:
            return -0.0;
        case 2:
            return 2.5;
        default:
            return 0.25 * static_cast<double>(i % 1000) + r;
    }
}

/// @brief Every element against the mirror, read by a Kokkos kernel through read().
template <typename A>
void check(const A& a, const std::vector<double>& mirror, const char* what)
{
    const std::int64_t                n = a.size();
    Kokkos::View<double*>             out("out", n);
    const auto                        r = a.read();
    Kokkos::parallel_for("read back", n, KOKKOS_LAMBDA(std::int64_t i) { out(i) = r(i); });
    const auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out);
    long       wrong = 0;
    for (std::int64_t i = 0; i < n; ++i)
    {
        wrong += same_bits(h(i), mirror[static_cast<std::size_t>(i)]) ? 0 : 1;
    }
    char line[160];
    std::snprintf(line, sizeof line, "%s: %ld wrong", what, wrong);
    expect(wrong == 0, line);
}

template <unsigned int S, sd::Variant V>
void run(std::int64_t n, unsigned long long arena_buffers, const char* label)
{
    constexpr bool zero = V == sd::Variant::zero;
    std::printf("[%s%s] S=%u n=%lld arena=%llu\n", zero ? "zero, " : "", label, S, static_cast<long long>(n),
                arena_buffers);
    // An append-buffer of n entries: the small arenas send most writes there.
    sk::Array<double, S, V> a(n, arena_buffers, 0.0, static_cast<unsigned long long>(n));
    std::vector<double>     mirror(static_cast<std::size_t>(n), 0.0);

    // 1. Scattered writes, a permutation of a third of the elements, in two kernels.
    {
        const std::int64_t m = n / 3;
        const auto         w = a.write();
        Kokkos::parallel_for("scatter 1", m / 2, KOKKOS_LAMBDA(std::int64_t k) {
            const std::int64_t i = permuted(k, n);
            w(i)                 = value_of(i, 1);
        });
        Kokkos::parallel_for("scatter 2", Kokkos::RangePolicy<>(m / 2, m), KOKKOS_LAMBDA(std::int64_t k) {
            const std::int64_t i = permuted(k, n);
            w(i)                 = value_of(i, 1);
        });
        a.end_write();
        for (std::int64_t k = 0; k < m; ++k)
        {
            const std::int64_t i                = permuted(k, n);
            mirror[static_cast<std::size_t>(i)] = value_of(i, 1);
        }
        check(a, mirror, "scattered w(i) = v, two kernels, one end_write()");
    }

    // 2. w(i) += 1.5, twice, by the thread that owns i.
    {
        const auto w = a.write();
        Kokkos::parallel_for("add", n, KOKKOS_LAMBDA(std::int64_t i) {
            w(i) += 1.5;
            w(i) += 1.5;
        });
        a.end_write();
        for (auto& x : mirror)
        {
            x = x + 1.5 + 1.5;
        }
        check(a, mirror, "w(i) += 1.5 twice");
    }

    // 3. The chunk-aligned path over a range.
    {
        const std::int64_t lo = n / 5, hi = n - n / 7;
        a.parallel_for_chunks(lo, hi, KOKKOS_LAMBDA(std::int64_t i) { return value_of(i, 3); });
        for (std::int64_t i = lo; i < hi; ++i)
        {
            mirror[static_cast<std::size_t>(i)] = value_of(i, 3);
        }
        check(a, mirror, "parallel_for_chunks over a range");
    }

    // 4. Zeros over the first half by w(i) = 0.0, then compress(): the chunks of the first
    //    half that hold only zeros fold back (in both variants; 2.5 chunks do not exist here).
    {
        const std::int64_t half = n / 2;
        const auto         w    = a.write();
        Kokkos::parallel_for("zeros", half, KOKKOS_LAMBDA(std::int64_t i) { w(i) = 0.0; });
        a.end_write();
        for (std::int64_t i = 0; i < half; ++i)
        {
            mirror[static_cast<std::size_t>(i)] = 0.0;
        }
        const unsigned long long before = a.materialised();
        unsigned long long       folds  = 0;
        for (std::int64_t c = 0; c * S < half; ++c)
        {
            const std::int64_t e   = (c + 1) * S < n ? (c + 1) * S : n;
            bool               all = true;
            for (std::int64_t i = c * S; i < e && all; ++i)
            {
                all = same_bits(mirror[static_cast<std::size_t>(i)], 0.0);
            }
            folds += all ? 1 : 0;
        }
        const unsigned long long got = a.compress();
        char                     line[160];
        std::snprintf(line, sizeof line, "compress(): %llu folded of %llu materialised, %llu expected", got, before,
                      folds);
        expect(got == folds, line);
        check(a, mirror, "after compress()");
    }

    // 5. A reduction over read().
    {
        const auto r   = a.read();
        double     sum = 0;
        Kokkos::parallel_reduce("sum", n, KOKKOS_LAMBDA(std::int64_t i, double& s) { s += r(i); }, sum);
        double want = 0;
        for (const double x : mirror)
        {
            want += x;
        }
        char line[160];
        std::snprintf(line, sizeof line, "parallel_reduce over read(): %.6g, %.6g expected", sum, want);
        expect(sum == want || (sum - want) * (sum - want) <= 1e-18 * want * want, line);
        std::printf("  %zu bytes held (%llu materialised chunks), %zu bytes of arena\n", a.bytes(), a.materialised(),
                    a.arena_bytes());
    }

    // 6. One phase over a and a second array, ended by one end_write(a, b).
    {
        sk::Array<double, S, V> b(n, arena_buffers, 0.0, static_cast<unsigned long long>(n));
        std::vector<double>     mirror_b(static_cast<std::size_t>(n), 0.0);
        const std::int64_t      m  = n / 4;
        const auto              wa = a.write();
        const auto              wb = b.write();
        Kokkos::parallel_for("two arrays", m, KOKKOS_LAMBDA(std::int64_t k) {
            const std::int64_t i = permuted(k, n);
            wa(i)                = value_of(i, 6);
            wb(n - 1 - i)        = value_of(i, 7);
        });
        sk::end_write(a, b);
        for (std::int64_t k = 0; k < m; ++k)
        {
            const std::int64_t i                            = permuted(k, n);
            mirror[static_cast<std::size_t>(i)]             = value_of(i, 6);
            mirror_b[static_cast<std::size_t>(n - 1 - i)]   = value_of(i, 7);
        }
        check(a, mirror, "end_write(a, b): a");
        check(b, mirror_b, "end_write(a, b): b");
    }
}

} // namespace

int main(int argc, char** argv)
{
    Kokkos::ScopeGuard guard(argc, argv);
    std::printf("kokkos_validation: the Kokkos facade against a host mirror (%s)\n",
                Kokkos::DefaultExecutionSpace::name());
    try
    {
        run<256, sd::Variant::zero_any>((1L << 20) + 77, 1L << 10, "small arena, tail chunk");
        run<1024, sd::Variant::zero_any>((1L << 20) + 77, (1L << 10) + 1, "roomy arena, tail chunk");
        run<256, sd::Variant::zero>((1L << 20) + 77, 1L << 10, "small arena, tail chunk");
        run<1024, sd::Variant::zero>((1L << 20) + 77, (1L << 10) + 1, "roomy arena, tail chunk");
    }
    catch (const std::exception& e)
    {
        std::printf("error: %s\n", e.what());
        return 2;
    }
    std::printf("%s: %d failure(s)\n", g_failures == 0 ? "PASSED" : "FAILED", g_failures);
    return g_failures == 0 ? 0 : 1;
}
