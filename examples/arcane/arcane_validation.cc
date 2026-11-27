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
 * @file arcane_validation.cc
 * @brief The Arcane facade (chunk/device/arcane_adaptor.h) against a host mirror: every phase
 *        written by RUNCOMMAND_LOOP1 commands through the facade only, read back by another
 *        into a NumArray, compared element by element. kokkos_validation's checks.
 *
 * @details Built with -DCHUNK_DEVICE_ARCANE=ON, run with -A,AcceleratorRuntime=cuda (or hip).
 *          For S = 256 and 1024, the zero_any and the zero variants, a partial last chunk:
 *          scattered `w(i) = v` in two commands before one end_write(); `w(i) += 1.5` twice by
 *          the owner thread; parallel_for_chunks over a range; a write of zeros over half the
 *          array, then compress() (its folds against the host model); a sum over read(); one
 *          phase over two arrays ended by one end_write(queue, a, b), then one
 *          compress(queue, a, b). Then the facade under the variants' names
 *          (zero::ArcaneArray, ...): Array(n), a write phase, compress() by each variant's
 *          rule, set_second_value(), written_chunk_count() and stats(); Array(image().state),
 *          assign(state), to_host(). Last, a Sequential queue refused by write().
 */

#include <cstdint>     // std::int64_t
#include <cstdio>      // std::printf, std::snprintf
#include <cstring>     // std::memcmp
#include <stdexcept>   // std::runtime_error
#include <type_traits> // std::is_same_v
#include <vector>      // std::vector

#include <arcane/launcher/ArcaneLauncher.h>          // ArcaneLauncher, StandaloneAcceleratorMng
#include <arcane/utils/Exception.h>                  // arcaneCallFunctionAndCatchException
#include <arcane/utils/NumArray.h>                   // NumArray
#include <arcane/accelerator/core/IAcceleratorMng.h> // IAcceleratorMng::defaultQueue
#include <arcane/accelerator/core/Runner.h>          // Runner (the Sequential queue)
#include <arcane/accelerator/NumArrayViews.h>        // viewOut
#include <arcane/accelerator/RunCommandLoop.h>       // RUNCOMMAND_LOOP1
#include <arcane/accelerator/RunQueue.h>             // makeCommand, makeQueue

#include "chunk/device/arcane_adaptor.h" // device::arcane::Array

namespace sd = shrink_array::device;
namespace sc = shrink_array::device::arcane;

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
ARCCORE_HOST_DEVICE inline std::int64_t permuted(std::int64_t k, std::int64_t n)
{
    return static_cast<std::int64_t>((static_cast<unsigned long long>(k) * 2654435761ULL) %
                                     static_cast<unsigned long long>(n));
}

/// @brief The value written to element i in round r: repeats of 0.0, -0.0 and 2.5, or varied.
ARCCORE_HOST_DEVICE inline double value_of(std::int64_t i, int r)
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

/// @brief f(i) for i in [lo, hi) by one RUNCOMMAND_LOOP1 on @p queue. f: a
///        `[=] ARCCORE_HOST_DEVICE (std::int64_t)` lambda.
template <typename Q, typename F>
void loop(Q queue, std::int64_t lo, std::int64_t hi, F f)
{
    using namespace Arcane::Accelerator;
    auto command = makeCommand(queue);
    command << RUNCOMMAND_LOOP1(iter, hi - lo)
    {
        auto [k] = iter();
        f(lo + static_cast<std::int64_t>(k));
    };
}

/// @brief Every element against the mirror, read by a command through read() into a NumArray.
template <typename Q, typename A>
void check(Q queue, const A& a, const std::vector<double>& mirror, const char* what)
{
    using namespace Arcane::Accelerator;
    const std::int64_t                      n = a.size();
    Arcane::NumArray<double, Arcane::MDDim1> out(n);
    {
        auto       command = makeCommand(queue);
        const auto r       = a.read(queue);
        auto       o       = viewOut(command, out);
        command << RUNCOMMAND_LOOP1(iter, n)
        {
            auto [i] = iter();
            o(i)     = r(i);
        };
    }
    sc::barrier(queue);
    long wrong = 0;
    for (std::int64_t i = 0; i < n; ++i)
    {
        wrong += same_bits(out(static_cast<Arcane::Int32>(i)), mirror[static_cast<std::size_t>(i)]) ? 0 : 1;
    }
    char line[160];
    std::snprintf(line, sizeof line, "%s: %ld wrong", what, wrong);
    expect(wrong == 0, line);
}

/// @brief Chunks of [0, half) that hold only +0.0 in @p mirror: what compress() folds there.
unsigned long long zero_chunks(const std::vector<double>& mirror, std::int64_t S, std::int64_t half)
{
    const auto         n     = static_cast<std::int64_t>(mirror.size());
    unsigned long long folds = 0;
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
    return folds;
}

template <unsigned int S, sd::Variant V, typename Q>
void run(Q queue, std::int64_t n, unsigned long long arena_buffers, const char* label)
{
    constexpr bool zero = V == sd::Variant::zero;
    std::printf("[%s%s] S=%u n=%lld arena=%llu\n", zero ? "zero, " : "", label, S, static_cast<long long>(n),
                arena_buffers);
    // An append-buffer of n entries: the small arenas send most writes there.
    sc::Array<double, S, V> a(n, arena_buffers, 0.0, static_cast<unsigned long long>(n));
    std::vector<double>     mirror(static_cast<std::size_t>(n), 0.0);

    // 1. Scattered writes, a permutation of a third of the elements, in two commands.
    {
        const std::int64_t m = n / 3;
        const auto         w = a.write(queue);
        loop(queue, 0, m / 2, [=] ARCCORE_HOST_DEVICE(std::int64_t k) {
            const std::int64_t i = permuted(k, n);
            w(i)                 = value_of(i, 1);
        });
        loop(queue, m / 2, m, [=] ARCCORE_HOST_DEVICE(std::int64_t k) {
            const std::int64_t i = permuted(k, n);
            w(i)                 = value_of(i, 1);
        });
        a.end_write(queue);
        for (std::int64_t k = 0; k < m; ++k)
        {
            const std::int64_t i                = permuted(k, n);
            mirror[static_cast<std::size_t>(i)] = value_of(i, 1);
        }
        check(queue, a, mirror, "scattered w(i) = v, two commands, one end_write()");
    }

    // 2. w(i) += 1.5, twice, by the thread that owns i.
    {
        const auto w = a.write(queue);
        loop(queue, 0, n, [=] ARCCORE_HOST_DEVICE(std::int64_t i) {
            w(i) += 1.5;
            w(i) += 1.5;
        });
        a.end_write(queue);
        for (auto& x : mirror)
        {
            x = x + 1.5 + 1.5;
        }
        check(queue, a, mirror, "w(i) += 1.5 twice");
    }

    // 3. The chunk-aligned path over a range.
    {
        const std::int64_t lo = n / 5, hi = n - n / 7;
        a.parallel_for_chunks(queue, lo, hi, [=] ARCCORE_HOST_DEVICE(std::int64_t i) { return value_of(i, 3); });
        for (std::int64_t i = lo; i < hi; ++i)
        {
            mirror[static_cast<std::size_t>(i)] = value_of(i, 3);
        }
        check(queue, a, mirror, "parallel_for_chunks over a range");
    }

    // 4. Zeros over the first half by w(i) = 0.0, then compress(): the chunks of the first
    //    half that hold only zeros fold back (in both variants).
    {
        const std::int64_t half = n / 2;
        const auto         w    = a.write(queue);
        loop(queue, 0, half, [=] ARCCORE_HOST_DEVICE(std::int64_t i) { w(i) = 0.0; });
        a.end_write(queue);
        for (std::int64_t i = 0; i < half; ++i)
        {
            mirror[static_cast<std::size_t>(i)] = 0.0;
        }
        const unsigned long long before = a.materialised();
        const unsigned long long folds  = zero_chunks(mirror, S, half);
        const unsigned long long got    = a.compress(queue);
        char                     line[160];
        std::snprintf(line, sizeof line, "compress(): %llu folded of %llu materialised, %llu expected", got, before,
                      folds);
        expect(got == folds, line);
        check(queue, a, mirror, "after compress()");
    }

    // 5. A sum over read(), through a NumArray (the host adds what a command read).
    {
        Arcane::NumArray<double, Arcane::MDDim1> out(n);
        {
            auto       command = makeCommand(queue);
            const auto r       = a.read(queue);
            auto       o       = Arcane::Accelerator::viewOut(command, out);
            command << RUNCOMMAND_LOOP1(iter, n)
            {
                auto [i] = iter();
                o(i)     = r(i) * 2.0;
            };
        }
        sc::barrier(queue);
        double sum = 0, want = 0;
        for (std::int64_t i = 0; i < n; ++i)
        {
            sum += out(static_cast<Arcane::Int32>(i));
            want += 2.0 * mirror[static_cast<std::size_t>(i)];
        }
        char line[160];
        std::snprintf(line, sizeof line, "a sum over read(): %.6g, %.6g expected", sum, want);
        expect(sum == want, line);
        std::printf("  %zu bytes held (%llu materialised chunks), %zu bytes of arena\n", a.bytes(), a.materialised(),
                    a.arena_bytes());
    }

    // 6. One phase over a and a second array, ended by one end_write(queue, a, b), then zeros
    //    over both first halves and one compress(queue, a, b).
    {
        sc::Array<double, S, V> b(n, arena_buffers, 0.0, static_cast<unsigned long long>(n));
        std::vector<double>     mirror_b(static_cast<std::size_t>(n), 0.0);
        const std::int64_t      m  = n / 4;
        const auto              wa = a.write(queue);
        const auto              wb = b.write(queue);
        loop(queue, 0, m, [=] ARCCORE_HOST_DEVICE(std::int64_t k) {
            const std::int64_t i = permuted(k, n);
            wa(i)                = value_of(i, 6);
            wb(n - 1 - i)        = value_of(i, 7);
        });
        sc::end_write(queue, a, b);
        for (std::int64_t k = 0; k < m; ++k)
        {
            const std::int64_t i                          = permuted(k, n);
            mirror[static_cast<std::size_t>(i)]           = value_of(i, 6);
            mirror_b[static_cast<std::size_t>(n - 1 - i)] = value_of(i, 7);
        }
        check(queue, a, mirror, "end_write(queue, a, b): a");
        check(queue, b, mirror_b, "end_write(queue, a, b): b");

        const std::int64_t half = n / 2;
        const auto         za   = a.write(queue); // a new phase: new accesses
        const auto         zb   = b.write(queue);
        loop(queue, 0, half, [=] ARCCORE_HOST_DEVICE(std::int64_t i) {
            za(i) = 0.0;
            zb(i) = 0.0;
        });
        sc::end_write(queue, a, b);
        for (std::int64_t i = 0; i < half; ++i)
        {
            mirror[static_cast<std::size_t>(i)]   = 0.0;
            mirror_b[static_cast<std::size_t>(i)] = 0.0;
        }
        const unsigned long long folds = zero_chunks(mirror, S, half) + zero_chunks(mirror_b, S, half);
        const unsigned long long got   = sc::compress(queue, a, b);
        char                     line[160];
        std::snprintf(line, sizeof line, "compress(queue, a, b): %llu folded, %llu expected", got, folds);
        expect(got == folds, line);
        check(queue, a, mirror, "compress(queue, a, b): a");
        check(queue, b, mirror_b, "compress(queue, a, b): b");
    }
}

/// @brief The facade under a variant's name (zero::ArcaneArray, ...): Array(n) and its
///        default arena, a write phase, compress() by the variant's rule (a value 3 everywhere
///        folds in zero_any, in zero_or_value once 3 is the second value, never in zero),
///        read back; then the conversion.
template <typename A, typename Q>
void run_alias(Q queue, const char* name)
{
    constexpr auto     V = A::variant;
    const std::int64_t n = (1L << 18) + 77;
    std::printf("[%s::ArcaneArray, S=%u] n=%lld\n", name, A::chunk_size(), static_cast<long long>(n));
    A                   a(n);
    std::vector<double> mirror(static_cast<std::size_t>(n), 3.0);
    if constexpr (V == sd::Variant::zero_or_value)
    {
        a.set_second_value(3.0);
        expect(a.second_value() == 3.0, "set_second_value(3.0), second_value()");
    }
    {
        const auto w = a.write(queue);
        loop(queue, 0, n, [=] ARCCORE_HOST_DEVICE(std::int64_t i) { w(i) = 3.0; });
        a.end_write(queue);
    }
    (void)a.compress(queue);
    const auto chunks   = static_cast<unsigned long long>((n + A::chunk_size() - 1) / A::chunk_size());
    const auto expected = V == sd::Variant::zero ? chunks : 0ULL;
    char       line[160];
    std::snprintf(line, sizeof line, "3 everywhere, compress(): %llu/%llu chunks hold a buffer (expected %llu)",
                  a.materialised(), chunks, expected);
    expect(a.materialised() == expected, line);
    expect(a.written_chunk_count() == a.materialised(), "written_chunk_count() == materialised()");
    expect(a.stats() == 100.0 * static_cast<double>(expected) / static_cast<double>(chunks), "stats()");
    check(queue, a, mirror, "read back");

    // The conversion: a ramp on every eighth chunk, compress(), image(), A(image().state),
    // assign(state) onto a third array, to_host() of each against the mirror.
    {
        const std::int64_t S = A::chunk_size();
        const auto         w = a.write(queue);
        loop(queue, 0, n, [=] ARCCORE_HOST_DEVICE(std::int64_t i) {
            if ((i / S) % 8 == 0)
            {
                w(i) = 0.5 * static_cast<double>(i % 97);
            }
        });
        a.end_write(queue);
        for (std::int64_t i = 0; i < n; ++i)
        {
            if ((i / S) % 8 == 0)
            {
                mirror[static_cast<std::size_t>(i)] = 0.5 * static_cast<double>(i % 97);
            }
        }
    }
    (void)a.compress(queue);
    const auto image = a.image(queue);
    A          b(image.state);
    std::snprintf(line, sizeof line, "A(image().state): %llu chunks hold a buffer, as the source (%llu)",
                  b.materialised(), a.materialised());
    expect(b.materialised() == a.materialised(), line);
    check(queue, b, mirror, "A(image().state) read back");
    A c(n);
    c.assign(queue, image.state);
    check(queue, c, mirror, "assign(image().state) read back");
    expect(a.to_host(queue) == mirror && b.to_host(queue) == mirror, "to_host() of the source and the copy");
}

void run_all()
{
    using namespace Arcane;
    using namespace Arcane::Accelerator;
    StandaloneAcceleratorMng launcher(ArcaneLauncher::createStandaloneAcceleratorMng());
    IAcceleratorMng*         acc_mng = launcher.acceleratorMng();
    auto                     queue   = acc_mng->defaultQueue();

    run<256, sd::Variant::zero_any>(queue, (1L << 20) + 77, 1L << 10, "small arena, tail chunk");
    run<1024, sd::Variant::zero_any>(queue, (1L << 20) + 77, (1L << 10) + 1, "roomy arena, tail chunk");
    run<256, sd::Variant::zero>(queue, (1L << 20) + 77, 1L << 10, "small arena, tail chunk");
    run<1024, sd::Variant::zero>(queue, (1L << 20) + 77, (1L << 10) + 1, "roomy arena, tail chunk");
    namespace sa = shrink_array;
    static_assert(std::is_same_v<sa::zero_any::ArcaneArray<double, 256>, sc::Array<double, 256, sd::Variant::zero_any>>,
                  "the per-variant names are the facade");
    run_alias<sa::zero::ArcaneArray<double>>(queue, "zero");
    run_alias<sa::zero_any::ArcaneArray<double>>(queue, "zero_any");
    run_alias<sa::zero_or_value::ArcaneArray<double>>(queue, "zero_or_value");
    run_alias<sa::zero_any::ArcaneArray<double, 256>>(queue, "zero_any");

    // A host queue: write() refuses it before any command touches device memory.
    std::printf("[a Sequential queue]\n");
    {
        Runner                       runner(eExecutionPolicy::Sequential);
        RunQueue                     host_queue = makeQueue(runner);
        sa::zero_any::ArcaneArray<double> a(1 << 12);
        bool                         refused = false;
        try
        {
            (void)a.write(host_queue);
        }
        catch (const std::runtime_error&)
        {
            refused = true;
        }
        expect(refused, "write() on a Sequential queue throws");
    }
}

} // namespace

int main(int argc, char* argv[])
{
    std::printf("arcane_validation: the Arcane facade against a host mirror\n");
    auto func = [&] {
        Arcane::ArcaneLauncher::init(Arcane::CommandLineArguments(&argc, &argv));
        run_all();
    };
    const int caught = Arcane::arcaneCallFunctionAndCatchException(func);
    if (caught != 0)
    {
        std::printf("arcane_validation: an exception was caught\n");
        return caught;
    }
    std::printf("%s: %d failure(s)\n", g_failures == 0 ? "PASSED" : "FAILED", g_failures);
    return g_failures == 0 ? 0 : 1;
}
