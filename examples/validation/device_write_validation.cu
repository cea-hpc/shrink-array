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
 * @file device_write_validation.cu
 * @brief The chunk-aligned write of device::Array (chunk/device/block_write.h, array.h)
 *        against a host mirror: whole-array writes by parallel_for_chunks, random spans by
 *        a kernel that calls Writer::write itself (two kernels per round, the second meeting
 *        the first one's unsettled fresh buffers), writes of the constant only (votes, no
 *        buffer), a small arena (failed spans written again after growth), a tail chunk and
 *        a constant other than zero.
 *
 * @details Built with -DCHUNK_DEVICE_CORE_CHECK=ON, for CUDA or HIP
 *          (-DCHUNK_DEVICE_LANGUAGE). Each round checks every element and the number of
 *          materialised chunks: exactly those that were ever written a value other than
 *          the constant (same_value: -0.0 is not 0.0), there being no compress() yet.
 */

#include <algorithm> // std::count, std::fill, std::min, std::shuffle
#include <cstdint>   // std::int64_t
#include <cstdio>    // std::printf, std::snprintf
#include <cstring>   // std::memcmp
#include <exception> // std::exception
#include <random>    // std::mt19937_64
#include <vector>    // std::vector

#include "chunk/device/array.h"   // device::Array, Writer
#include "chunk/device/backend.h" // allocate, copy, release, synchronize

namespace sd = shrink_array::device;

namespace
{

int g_failures = 0;

void expect(bool ok, const char* what)
{
    std::printf("  %-62s %s\n", what, ok ? "ok" : "FAILED");
    g_failures += ok ? 0 : 1;
}

/// @brief f(i) = target[i]: a function of i, the same in a span's second pass.
struct FromArray
{
    const double* target;

    __device__ double operator()(std::int64_t i) const noexcept
    {
        return target[i];
    }
};

struct Range
{
    std::int64_t lo;
    std::int64_t hi;
};

/// @brief Each block takes ranges and writes their chunks with Writer::write: two ranges
///        may share a chunk, written by two blocks of the kernel.
template <int BLOCK, unsigned int S>
__global__ void __launch_bounds__(BLOCK) write_ranges(sd::Writer<double, S> w, const Range* r, long n, FromArray f)
{
    for (long k = blockIdx.x; k < n; k += gridDim.x)
    {
        for (std::int64_t c = r[k].lo / S; c * S < r[k].hi; ++c)
        {
            const std::int64_t lo = c * S > r[k].lo ? c * S : r[k].lo;
            const std::int64_t hi = (c + 1) * S < r[k].hi ? (c + 1) * S : r[k].hi;
            w.template write<BLOCK>(c, lo, hi, f);
        }
    }
}

bool same_bits(double a, double b)
{
    return std::memcmp(&a, &b, sizeof a) == 0;
}

template <unsigned int S>
void run(unsigned seed, std::int64_t n, unsigned long long arena_buffers, double constant, const char* label)
{
    constexpr int BLOCK = 256;
    std::printf("[%s] S=%u n=%lld arena=%llu constant=%g\n", label, S, static_cast<long long>(n), arena_buffers,
                constant);
    sd::Array<double, S> a(n, arena_buffers, constant);
    const auto           nn = static_cast<std::size_t>(n);
    std::vector<double>  mirror(nn, constant), target(nn);
    std::vector<char>    ever(static_cast<std::size_t>(a.n_chunks()), 0);
    std::mt19937_64      rng(seed);
    const double         palette[] = {0.0, -0.0, 1.5, 2.5};
    double*              d_target  = sd::allocate<double>(nn);
    const FromArray      f{d_target};
    const auto           note = [&](std::int64_t lo, std::int64_t hi)
    {
        for (std::int64_t i = lo; i < hi; ++i)
        {
            mirror[static_cast<std::size_t>(i)] = target[static_cast<std::size_t>(i)];
            if (!same_bits(target[static_cast<std::size_t>(i)], constant))
            {
                ever[static_cast<std::size_t>(i / S)] = 1;
            }
        }
    };

    for (int round = 0; round < 6; ++round)
    {
        for (auto& x : target)
        {
            const auto r = rng() % 8;
            x            = r < 4 ? palette[r] : static_cast<double>(rng() % 1000) / 7.0;
        }
        // Rounds 0, 2, 4: ranges, the first on a compressed array (fresh buffers shared by
        // two blocks, lost CAS); 1 and 5: the constant everywhere (votes only, no buffer);
        // 3: the whole array.
        const bool ranges_round = round % 2 == 0;
        if (round == 1 || round == 5)
        {
            std::fill(target.begin(), target.end(), constant);
        }
        sd::copy(d_target, target.data(), nn);
        if (!ranges_round)
        {
            a.template parallel_for_chunks<BLOCK>(0, n, f);
            note(0, n);
        }
        else
        {
            std::vector<Range> ranges;
            for (std::int64_t lo = 0; lo < n;)
            {
                const std::int64_t hi = std::min<std::int64_t>(n, lo + 1 + static_cast<std::int64_t>(rng() % (3ULL * S)));
                if (rng() % 3 != 0) // a third of the gaps not written
                {
                    ranges.push_back({lo, hi});
                    note(lo, hi);
                }
                lo = hi;
            }
            std::shuffle(ranges.begin(), ranges.end(), rng);
            Range* d_ranges = sd::allocate<Range>(ranges.size());
            sd::copy(d_ranges, ranges.data(), ranges.size());
            // Two kernels, half the ranges each, then one finish(): the second kernel meets
            // the first one's fresh buffers, not settled yet.
            const auto half = static_cast<long>(ranges.size() / 2);
            write_ranges<BLOCK, S><<<1024, BLOCK>>>(a.writer(), d_ranges, half, f);
            write_ranges<BLOCK, S><<<1024, BLOCK>>>(a.writer(), d_ranges + half,
                                                    static_cast<long>(ranges.size()) - half, f);
            a.template finish<BLOCK>(f);
            sd::release(d_ranges);
        }
        const std::vector<double> got   = a.to_host();
        long                      wrong = 0;
        for (std::size_t i = 0; i < nn; ++i)
        {
            wrong += same_bits(got[i], mirror[i]) ? 0 : 1;
        }
        const auto expected = static_cast<unsigned long long>(std::count(ever.begin(), ever.end(), 1));
        const auto mat      = a.materialised();
        char       line[160];
        std::snprintf(line, sizeof line, "round %d (%s): %ld wrong, %llu/%llu chunks materialised", round,
                      ranges_round ? "ranges" : round == 3 ? "whole array" : "the constant", wrong, mat, expected);
        expect(wrong == 0 && mat == expected, line);
    }
    std::printf("  arena %llu buffers in %d arena(s), %llu in use, %llu lost CAS\n", a.arena().buffers(),
                a.arena().arenas(), a.arena().in_use(), a.cas_lost());
    expect(a.arena().in_use() == a.materialised(), "one buffer per materialised chunk, the lost ones given back");
    sd::release(d_target);
}

} // namespace

int main()
{
    std::printf("device_write_validation: device::Array, the chunk-aligned write\n");
    try
    {
        run<256>(11, 1L << 20, 1L << 12, 0.0, "exact arena");
        run<1024>(12, (1L << 20) + 300, (1L << 10) + 1, 0.0, "exact arena, tail chunk");
        run<256>(13, 1L << 18, 16, 0.0, "small arena: failed spans written again");
        run<1024>(14, (1L << 18) + 77, 4, 2.5, "small arena, tail chunk, constant 2.5");
    }
    catch (const std::exception& e)
    {
        std::printf("error: %s\n", e.what());
        return 2;
    }
    std::printf("%s: %d failure(s)\n", g_failures == 0 ? "PASSED" : "FAILED", g_failures);
    return g_failures == 0 ? 0 : 1;
}
