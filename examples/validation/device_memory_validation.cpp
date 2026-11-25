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
 * @file device_memory_validation.cpp
 * @brief Stage 1's criterion (docs/gpu_design.md, section 13): the same final memory as the
 *        CPU variant at equal geometry. The same rounds of writes on a host array and on a
 *        device::Array of the same variant, each followed by shrink_to_fit() on the host and
 *        compress() on the device; after every round the two layouts compared chunk by chunk:
 *        the kind (first value, second value, materialised), the constant bit for bit, the
 *        elements of a materialised chunk, hence the bytes of buffers.
 *
 * @details The C++ half; the device half is device_memory_device.cu (see
 *          device_memory_validation.h). Built twice: device_memory_validation_zero_any
 *          (CHUNK_VARIANT_ZERO_ANY, host zero_any::Array against Variant::zero_any) and
 *          device_memory_validation_zero (CHUNK_VARIANT_ZERO, zero::Array against
 *          Variant::zero). The rounds alternate the device's two write paths (Writer::write
 *          per span, Writer::assign per element) and the values that decide a fold: 0.0 and
 *          -0.0, a constant per chunk (distinct second values), NaN with a payload, the
 *          partial last chunk, and a round of zeros everywhere. One difference is known and
 *          accepted, and only that one (known_tail_fold): the partial last chunk, uniform over
 *          its valid elements, folds on the device and keeps its buffer on the host.
 */

#include <algorithm> // std::min
#include <cstddef>   // std::size_t
#include <cstdint>   // std::int64_t, std::uint64_t
#include <cstdio>    // std::printf, std::snprintf
#include <cstring>   // std::memcpy, std::memcmp
#include <exception> // std::exception
#include <random>    // std::mt19937_64
#include <utility>   // std::move
#include <vector>    // std::vector

#include "chunk/chunk_exchange.h"     // export_array, ChunkKind
#include "chunk/internal/pmp_chunk.h" // pmp::current_team
#include "chunk/phases.h"             // parallel_phase
#include "chunk/runtime.h"            // Runtime::init
#if defined(CHUNK_VARIANT_ZERO)
    #include "chunk/zero_array.h" // zero::Array
#else
    #include "chunk/zero_any_array.h" // zero_any::Array
#endif

#include "device_memory_validation.h" // Round, device_replay

namespace sa = shrink_array;

namespace
{

#if defined(CHUNK_VARIANT_ZERO)
using HostArray                  = sa::zero::Array<double>;
constexpr bool        zero_variant = true;
constexpr const char* variant      = "zero";
#else
using HostArray                  = sa::zero_any::Array<double>;
constexpr bool        zero_variant = false;
constexpr const char* variant      = "zero_any";
#endif

int g_failures = 0;

void expect(bool ok, const char* what)
{
    std::printf("  %-70s %s\n", what, ok ? "ok" : "FAILED");
    g_failures += ok ? 0 : 1;
}

bool same_bits(double a, double b)
{
    return std::memcmp(&a, &b, sizeof a) == 0;
}

double nan_with_payload(std::uint64_t payload)
{
    const std::uint64_t bits = 0x7ff8000000000000ULL | payload;
    double              d    = 0;
    std::memcpy(&d, &bits, sizeof d);
    return d;
}

/// @brief The rounds: spans or elements, by one write path or the other, over @p n elements
///        in chunks of @p s.
std::vector<device_memory::Round> make_rounds(std::int64_t n, std::int64_t s)
{
    std::mt19937_64                   rng(2026);
    const std::int64_t                n_chunks  = (n + s - 1) / s;
    const double                      palette[] = {0.0, -0.0, 1.5, 2.5, nan_with_payload(5)};
    std::vector<device_memory::Round> rounds;
    const auto                        add_span = [](device_memory::Round& r, std::int64_t lo, std::int64_t hi)
    {
        r.lo.push_back(lo);
        r.hi.push_back(hi);
    };
    const auto random_spans = [&](device_memory::Round& r)
    {
        for (std::int64_t lo = 0; lo < n;)
        {
            const std::int64_t hi = std::min<std::int64_t>(n, lo + 1 + static_cast<std::int64_t>(rng() % (3ULL * s)));
            if (rng() % 3 != 0)
            {
                add_span(r, lo, hi);
            }
            lo = hi;
        }
    };
    // 0. Spans by Writer::write, values from the palette or varied.
    {
        device_memory::Round r{{}, {}, std::vector<double>(static_cast<std::size_t>(n)), false};
        for (auto& x : r.target)
        {
            const auto k = rng() % 8;
            x            = k < 5 ? palette[k] : static_cast<double>(rng() % 1000) / 7.0;
        }
        random_spans(r);
        rounds.push_back(std::move(r));
    }
    // 1. Every chunk whole by Writer::assign, one value per chunk: 0.0, -0.0, a value of its
    //    own (distinct second values on the host), NaN with a payload, or varied.
    {
        device_memory::Round r{{}, {}, std::vector<double>(static_cast<std::size_t>(n)), true};
        for (std::int64_t c = 0; c < n_chunks; ++c)
        {
            const std::int64_t hi = std::min(n, (c + 1) * s);
            for (std::int64_t i = c * s; i < hi; ++i)
            {
                double v = 0;
                switch (c % 5)
                {
                    case 0:
                        v = 0.0;
                        break;
                    case 1:
                        v = -0.0;
                        break;
                    case 2:
                        v = 0.5 + static_cast<double>(c);
                        break;
                    case 3:
                        v = nan_with_payload(9);
                        break;
                    default:
                        v = static_cast<double>(i);
                }
                r.target[static_cast<std::size_t>(i)] = v;
            }
        }
        add_span(r, 0, n);
        rounds.push_back(std::move(r));
    }
    // 2. Spans by Writer::write over that layout, a constant per span.
    {
        device_memory::Round r{{}, {}, std::vector<double>(static_cast<std::size_t>(n)), false};
        random_spans(r);
        for (std::size_t k = 0; k < r.lo.size(); ++k)
        {
            const double v = palette[rng() % 5];
            for (std::int64_t i = r.lo[k]; i < r.hi[k]; ++i)
            {
                r.target[static_cast<std::size_t>(i)] = v;
            }
        }
        rounds.push_back(std::move(r));
    }
    // 3. Scattered elements by Writer::assign: one in 64, palette values.
    {
        device_memory::Round r{{}, {}, std::vector<double>(static_cast<std::size_t>(n)), true};
        for (std::int64_t i = static_cast<std::int64_t>(rng() % 64); i < n; i += 1 + static_cast<std::int64_t>(rng() % 127))
        {
            r.target[static_cast<std::size_t>(i)] = palette[rng() % 5];
            add_span(r, i, i + 1);
        }
        rounds.push_back(std::move(r));
    }
    // 4. The partial last chunk alone, then 2.5 over its valid elements (folds on both sides
    //    only if the tail past n is not compared).
    {
        device_memory::Round r{{}, {}, std::vector<double>(static_cast<std::size_t>(n), 2.5), false};
        add_span(r, (n_chunks - 1) * s, n);
        rounds.push_back(std::move(r));
    }
    // 5. Zeros everywhere, by Writer::assign: everything folds to the first value.
    {
        device_memory::Round r{{}, {}, std::vector<double>(static_cast<std::size_t>(n), 0.0), true};
        add_span(r, 0, n);
        rounds.push_back(std::move(r));
    }
    return rounds;
}

/// @brief The round on the host: every element of its spans written, in one phase, then
///        shrink_to_fit().
void host_round(HostArray& a, const device_memory::Round& r)
{
    std::vector<int> index;
    for (std::size_t k = 0; k < r.lo.size(); ++k)
    {
        for (std::int64_t i = r.lo[k]; i < r.hi[k]; ++i)
        {
            index.push_back(static_cast<int>(i));
        }
    }
    sa::parallel_phase(
        [&]
        {
            sa::pmp::current_team().for_static(0, static_cast<int>(index.size()),
                                               [&](int k)
                                               {
                                                   const int i = index[static_cast<std::size_t>(k)];
                                                   a[i]        = r.target[static_cast<std::size_t>(i)];
                                               });
        });
    a.shrink_to_fit();
}

/**
 * @brief Whether chunk @p c is the one known difference: the partial last chunk, uniform over
 *        its valid elements, folded on the device and materialised on the host.
 * @details The host tests a chunk over its chunk_size() elements, the tail past size() included
 *          (a chunk knows neither its rank nor the array's size; ChunkElems carries
 *          chunk_size() through the compression pipeline); the tail holds the constant of the
 *          chunk's materialisation, so a last chunk uniform at another value keeps its buffer
 *          there. The device compares the elements below n only (compress.h). One buffer more on
 *          the host, the device's memory below it: a CPU fix for develop, not done (it touches
 *          the incremental compression queue, see docs/gpu_design.md stage 1).
 */
bool known_tail_fold(const sa::ArrayState<double>& host, const sa::ArrayState<double>& dev, std::size_t c)
{
    const std::size_t s     = host.chunk_size;
    const std::size_t used  = (host.size + s - 1) / s;
    const std::size_t valid = host.size - (c * s);
    const auto&       h     = host.chunks[c];
    const auto&       d     = dev.chunks[c];
    if (c + 1 != used || valid >= s || h.kind != sa::ChunkKind::materialised ||
        (d.kind != sa::ChunkKind::first_value && d.kind != sa::ChunkKind::second_value))
    {
        return false;
    }
    for (std::size_t e = 0; e < valid; ++e)
    {
        if (!same_bits(h.data[e], d.value))
        {
            return false;
        }
    }
    return true;
}

/// @brief The host layout against the device's, chunk by chunk.
void compare(const sa::ArrayState<double>& host, const sa::ArrayState<double>& dev, int round)
{
    const std::size_t s     = host.chunk_size;
    const std::size_t used  = (host.size + s - 1) / s;
    long              kinds = 0, values = 0, elements = 0, mat_host = 0, mat_dev = 0, tail_folds = 0;
    for (std::size_t c = 0; c < used; ++c)
    {
        const auto& h = host.chunks[c];
        const auto& d = dev.chunks[c];
        mat_host += h.kind == sa::ChunkKind::materialised ? 1 : 0;
        mat_dev += d.kind == sa::ChunkKind::materialised ? 1 : 0;
        if (h.kind != d.kind)
        {
            if (known_tail_fold(host, dev, c))
            {
                ++tail_folds;
            }
            else
            {
                ++kinds;
            }
            continue;
        }
        if (h.kind != sa::ChunkKind::materialised)
        {
            values += same_bits(h.value, d.value) ? 0 : 1;
            continue;
        }
        const std::size_t valid = host.size - (c * s) < s ? host.size - (c * s) : s;
        for (std::size_t e = 0; e < valid; ++e)
        {
            elements += same_bits(h.data[e], d.data[e]) ? 0 : 1;
        }
    }
    char line[240];
    std::snprintf(line, sizeof line,
                  "round %d: %ld/%ld materialised (%ld KiB of buffers on the host), %ld kinds, %ld constants, %ld "
                  "elements differ%s",
                  round, mat_host, mat_dev, mat_host * static_cast<long>(s * sizeof(double)) / 1024, kinds, values,
                  elements, tail_folds != 0 ? "; the last chunk folded on the device only (known)" : "");
    expect(host.size == dev.size && host.chunk_size == dev.chunk_size && kinds == 0 && values == 0 && elements == 0 &&
               mat_host == mat_dev + tail_folds,
           line);
}

void run()
{
    HostArray          a(0);
    const std::int64_t s = a.chunk_size();
    const std::int64_t n = (200 * s) + 37;
    a.resize(static_cast<int>(n));
    std::printf("[%s::Array<double> against device::Array, Variant::%s] S=%lld n=%lld\n", variant, variant,
                static_cast<long long>(s), static_cast<long long>(n));
    const std::vector<device_memory::Round>             rounds = make_rounds(n, s);
    const std::vector<sa::ArrayImage<double>>           dev    = device_memory::device_replay(n, static_cast<unsigned int>(s), zero_variant, rounds);
    for (std::size_t r = 0; r < rounds.size(); ++r)
    {
        host_round(a, rounds[r]);
        compare(sa::export_array(a), dev[r].state, static_cast<int>(r));
    }
}

} // namespace

int main()
{
    sa::Runtime::init(true);
    std::printf("device_memory_validation: the same final memory on the host and on the device\n");
    try
    {
        run();
    }
    catch (const std::exception& e)
    {
        std::printf("error: %s\n", e.what());
        return 2;
    }
    std::printf("%s: %d failure(s)\n", g_failures == 0 ? "PASSED" : "FAILED", g_failures);
    return g_failures == 0 ? 0 : 1;
}
