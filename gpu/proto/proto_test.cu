// Copyright 2026 CEA (www.cea.fr)
// SPDX-License-Identifier: Apache-2.0

/**
 * @file proto_test.cu
 * @brief Correctness of the stage-0 prototype against a host mirror (std::vector):
 *        random writes from many warps at once (lost CAS), chunks folded back by
 *        compress(), arena exhaustion (append-buffer, growth, replay), append overflow.
 *
 * @details Each round writes a set of distinct indices (two writes of one element in a
 *          kernel are a race of the user program, docs/gpu_design.md 6.4), in a shuffled
 *          order so that the lanes needing one chunk sit in many warps. After finalize()
 *          every element is compared bit for bit with the mirror, and the number of
 *          materialised chunks with the mirror's non-uniform chunks.
 *
 *          Usage: proto_test [seed]. Exit status 0 when every case passes.
 */

#include <algorithm>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>

#include "zero_any_device.h"

namespace sa = shrink_array::device;

namespace
{

template <typename T, int S>
__global__ void write_ops(sa::View<T, S> v, const long* index, const T* value, long n_ops)
{
    for (long k = blockIdx.x * static_cast<long>(blockDim.x) + threadIdx.x; k < n_ops;
         k += static_cast<long>(gridDim.x) * blockDim.x)
    {
        v.assign(index[k], value[k]);
    }
}

/// @brief The values of a chunk-aligned write: element i gets v[i].
struct FromArray
{
    const double* v;

    __device__ double operator()(long i) const noexcept
    {
        return v[i];
    }
};

/// @brief A range [lo, hi) of a chunk-aligned round.
struct Span
{
    long lo;
    long hi;
};

/// @brief Several ranges in one kernel, each by one block over its chunks (block_write): two
///        ranges may share a chunk, so two blocks may race for its publication.
template <int BLOCK, int S>
__global__ void write_spans(sa::View<double, S> v, const Span* spans, long n_spans, FromArray f)
{
    for (long r = blockIdx.x; r < n_spans; r += gridDim.x)
    {
        const Span g = spans[r];
        for (long c = g.lo / S; c * S < g.hi; ++c)
        {
            const long lo = c * S > g.lo ? c * S : g.lo;
            const long hi = (c + 1) * S < g.hi ? (c + 1) * S : g.hi;
            v.template block_write<BLOCK>(c, lo, hi, f);
        }
    }
}

/// @brief One round of writes: copies the ops to the device, one thread per op.
template <typename T, int S>
void run_ops(sa::DeviceArray<T, S>& a, const std::vector<long>& index, const std::vector<T>& value)
{
    long* d_index = nullptr;
    T*    d_value = nullptr;
    SA_GPU(gpuMalloc(&d_index, sizeof(long) * index.size()));
    SA_GPU(gpuMalloc(&d_value, sizeof(T) * value.size()));
    SA_GPU(gpuMemcpy(d_index, index.data(), sizeof(long) * index.size(), gpuMemcpyHostToDevice));
    SA_GPU(gpuMemcpy(d_value, value.data(), sizeof(T) * value.size(), gpuMemcpyHostToDevice));
    const long n_ops  = static_cast<long>(index.size());
    const auto blocks = static_cast<unsigned>(std::clamp<long>((n_ops + 255) / 256, 1, 4096));
    write_ops<<<blocks, 256>>>(a.view(), d_index, d_value, n_ops);
    SA_GPU(gpuGetLastError());
    SA_GPU(gpuDeviceSynchronize());
    (void)gpuFree(d_index);
    (void)gpuFree(d_value);
}

/// @brief Chunks of the mirror that are not uniform (what must stay materialised).
template <typename T, int S>
unsigned long long non_uniform_chunks(const std::vector<T>& mirror)
{
    unsigned long long count = 0;
    const long         n     = static_cast<long>(mirror.size());
    for (long c = 0; c * S < n; ++c)
    {
        const long end = std::min<long>((c + 1) * S, n);
        for (long i = c * S + 1; i < end; ++i)
        {
            if (!sa::same_bits(mirror[static_cast<std::size_t>(i)], mirror[static_cast<std::size_t>(c * S)]))
            {
                ++count;
                break;
            }
        }
    }
    return count;
}

int g_failures = 0;

void expect(bool ok, const char* what)
{
    std::printf("  %-64s %s\n", what, ok ? "ok" : "FAILED");
    g_failures += ok ? 0 : 1;
}

/// @brief The array equals the mirror, element by element, and holds exactly its non-uniform chunks.
template <typename T, int S>
void check_state(const sa::DeviceArray<T, S>& a, const std::vector<T>& mirror, const char* what)
{
    const std::vector<T> got   = a.to_host();
    long                 wrong = 0;
    for (std::size_t i = 0; i < mirror.size(); ++i)
    {
        wrong += sa::same_bits(got[i], mirror[i]) ? 0 : 1;
    }
    const sa::Stats st       = a.stats();
    const auto      expected = non_uniform_chunks<T, S>(mirror);
    char            line[160];
    std::snprintf(line, sizeof line, "%s: %ld wrong, %llu/%llu chunks materialised", what, wrong, st.materialised,
                  expected);
    expect(wrong == 0 && st.materialised == expected, line);
}

/// @brief Random rounds; values from a small set (0, -0.0, 1.5, 2.5) so that chunks fold back.
template <int S>
void random_rounds(unsigned           seed,
                   long               n,
                   unsigned long long arena_chunks,
                   unsigned long long append_capacity,
                   const char*        label,
                   sa::Coordination   coord     = sa::Coordination::race,
                   sa::WriteLoad      load      = sa::WriteLoad::acquire,
                   bool               flip_pass = false)
{
    const char* load_name[] = {"acquire", "relaxed", "plain"};
    std::printf("[%s, %s, %s load%s] S=%d n=%ld arena=%llu chunks append=%llu\n", label,
                coord == sa::Coordination::claim ? "claim" : "race", load_name[static_cast<int>(load)],
                flip_pass ? ", other compress()" : "", S, n, arena_chunks, append_capacity);
    sa::DeviceArray<double, S> a(n, arena_chunks, append_capacity);
    a.set_coordination(coord);
    a.set_write_load(load);
    if (flip_pass)
    {
        a.set_compress_per_warp(S > 256);
    }
    std::vector<double> mirror(static_cast<std::size_t>(n), 0.0);
    std::mt19937_64     rng(seed);
    const double        palette[] = {0.0, -0.0, 1.5, 2.5};

    for (int round = 0; round < 6; ++round)
    {
        // Distinct indices: a random subset, plus whole chunks set to one value (folded back).
        std::vector<long> index(static_cast<std::size_t>(n));
        std::iota(index.begin(), index.end(), 0L);
        std::shuffle(index.begin(), index.end(), rng);
        index.resize(static_cast<std::size_t>(n / 3));
        std::vector<double> value(index.size());
        for (std::size_t k = 0; k < index.size(); ++k)
        {
            const auto r = rng() % 8;
            value[k]     = r < 4 ? palette[r] : static_cast<double>(rng() % 1000) / 7.0;
        }
        // Rounds 3 and 5: every element of every chunk with an even number gets 2.5
        // instead, so those chunks must fold back to the constant 2.5.
        if (round == 3 || round == 5)
        {
            index.resize(0);
            value.resize(0);
            for (long c = 0; c * S < n; c += 2)
            {
                for (long i = c * S; i < std::min<long>((c + 1) * S, n); ++i)
                {
                    index.push_back(i);
                    value.push_back(2.5);
                }
            }
            std::vector<std::size_t> perm(index.size());
            std::iota(perm.begin(), perm.end(), std::size_t{0});
            std::shuffle(perm.begin(), perm.end(), rng);
            std::vector<long>   i2(index.size());
            std::vector<double> v2(index.size());
            for (std::size_t k = 0; k < perm.size(); ++k)
            {
                i2[k] = index[perm[k]];
                v2[k] = value[perm[k]];
            }
            index.swap(i2);
            value.swap(v2);
        }
        for (std::size_t k = 0; k < index.size(); ++k)
        {
            mirror[static_cast<std::size_t>(index[k])] = value[k];
        }
        run_ops(a, index, value);
        a.finalize();
        char what[64];
        std::snprintf(what, sizeof what, "round %d", round);
        check_state(a, mirror, what);
    }
    // Everything back to 0: no chunk may stay materialised, every buffer back in the stack.
    std::vector<long> all(static_cast<std::size_t>(n));
    std::iota(all.begin(), all.end(), 0L);
    std::shuffle(all.begin(), all.end(), rng);
    std::vector<double> zeros(all.size(), 0.0);
    std::fill(mirror.begin(), mirror.end(), 0.0);
    run_ops(a, all, zeros);
    a.finalize();
    check_state(a, mirror, "all back to 0");
    const sa::Stats st = a.stats();
    std::printf("  stats: arena %llu buffers, %llu lost CAS, %llu arena failures, %llu replays, %llu claim waits, "
                "%llu given up\n",
                st.arena_buffers, st.cas_lost, st.arena_fail, st.replays, st.claim_wait, st.claim_spin);
    if (coord == sa::Coordination::claim)
    {
        expect(st.cas_lost == 0, "claim: no pre-fill lost");
    }
}

/// @brief Chunk-aligned rounds (block_write): random non-overlapping spans in one kernel,
///        whole-array writes by parallel_for_chunks, spans of the constant (no buffer).
template <int S>
void chunk_rounds(unsigned           seed,
                  long               n,
                  unsigned long long arena_chunks,
                  unsigned long long append_capacity,
                  const char*        label,
                  bool               relaxed_publish = false)
{
    constexpr int BLOCK = 256;
    std::printf("[%s, chunk-aligned%s] S=%d n=%ld arena=%llu chunks append=%llu\n", label,
                relaxed_publish ? ", relaxed publication" : "", S, n, arena_chunks, append_capacity);
    sa::DeviceArray<double, S> a(n, arena_chunks, append_capacity);
    a.set_relaxed_publish(relaxed_publish);
    std::vector<double>        mirror(static_cast<std::size_t>(n), 0.0);
    std::vector<double>        target(mirror.size());
    std::mt19937_64            rng(seed);
    const double               palette[] = {0.0, -0.0, 1.5, 2.5};
    double*                    d_target  = nullptr;
    SA_GPU(gpuMalloc(&d_target, sizeof(double) * mirror.size()));
    const FromArray f{d_target};

    for (int round = 0; round < 6; ++round)
    {
        for (auto& x : target)
        {
            const auto r = rng() % 8;
            x            = r < 4 ? palette[r] : static_cast<double>(rng() % 1000) / 7.0;
        }
        if (round == 2)
        {
            std::fill(target.begin(), target.end(), 0.0); // the constant everywhere: votes only
        }
        if (round == 4)
        {
            // Some chunks one value (fold back), the others varying.
            for (long c = 0; c * S < n; c += 2)
            {
                std::fill(target.begin() + c * S, target.begin() + std::min<long>((c + 1) * S, n), 2.5);
            }
        }
        SA_GPU(gpuMemcpy(d_target, target.data(), sizeof(double) * target.size(), gpuMemcpyHostToDevice));
        if (round % 2 == 0)
        {
            a.template parallel_for_chunks<BLOCK>(0, n, f);
            mirror = target;
        }
        else
        {
            std::vector<Span> spans;
            for (long lo = 0; lo < n;)
            {
                const long len = 1 + static_cast<long>(rng() % (3UL * S));
                const long hi  = std::min(n, lo + len);
                if (rng() % 3 != 0) // a third of the gaps not written
                {
                    spans.push_back({lo, hi});
                    std::copy(target.begin() + lo, target.begin() + hi, mirror.begin() + lo);
                }
                lo = hi;
            }
            std::shuffle(spans.begin(), spans.end(), rng);
            Span* d_spans = nullptr;
            SA_GPU(gpuMalloc(&d_spans, sizeof(Span) * spans.size()));
            SA_GPU(gpuMemcpy(d_spans, spans.data(), sizeof(Span) * spans.size(), gpuMemcpyHostToDevice));
            // Two kernels, half the spans each: with the relaxed publication the second one
            // meets the first one's fresh buffers, not settled yet.
            const auto half = static_cast<long>(spans.size() / 2);
            write_spans<BLOCK, S><<<1024, BLOCK>>>(a.view(), d_spans, half, f);
            write_spans<BLOCK, S><<<1024, BLOCK>>>(a.view(), d_spans + half, static_cast<long>(spans.size()) - half, f);
            SA_GPU(gpuGetLastError());
            SA_GPU(gpuDeviceSynchronize());
            (void)gpuFree(d_spans);
        }
        a.finalize();
        char what[64];
        std::snprintf(what, sizeof what, "round %d (%s)", round, round % 2 == 0 ? "whole array" : "spans");
        check_state(a, mirror, what);
    }
    (void)gpuFree(d_target);
    const sa::Stats st = a.stats();
    std::printf("  stats: arena %llu buffers, %llu lost CAS, %llu arena failures, %llu replays\n", st.arena_buffers,
                st.cas_lost, st.arena_fail, st.replays);
}

/// @brief An append-buffer too small for what a phase parks: finalize() must report it.
void append_overflow()
{
    std::printf("[append overflow] S=256\n");
    constexpr int              S = 256;
    const long                 n = 64L * S;
    sa::DeviceArray<double, S> a(n, 1, 8); // one buffer, 8 parked writes at most
    std::vector<long>          index(static_cast<std::size_t>(n));
    std::iota(index.begin(), index.end(), 0L);
    std::vector<double> value(index.size(), 3.25);
    for (std::size_t k = 0; k < value.size(); ++k)
    {
        value[k] = static_cast<double>(k); // not uniform: every chunk must be materialised
    }
    run_ops(a, index, value);
    bool thrown = false;
    try
    {
        a.finalize();
    }
    catch (const std::runtime_error& e)
    {
        thrown = true;
        std::printf("  finalize: %s\n", e.what());
    }
    expect(thrown, "finalize() reports the lost writes");
}

} // namespace

int main(int argc, char** argv)
{
    const unsigned seed = argc > 1 ? static_cast<unsigned>(std::strtoul(argv[1], nullptr, 10)) : 12345U;
    std::printf("proto_test, seed %u\n", seed);
    // Each case on its own: an exception (lost writes, a device error) fails that case and
    // the others still run.
    const auto run = [](const auto& f)
    {
        try
        {
            f();
        }
        catch (const std::exception& e)
        {
            std::printf("  error: %s\n", e.what());
            g_failures += 1;
        }
    };
    // Arena large enough for no exhaustion on NVIDIA; then small arenas: growth and replays.
    // Roomy: twice the chunks, room for the buffers of lost CAS (parked until compress()),
    // and an append-buffer of n entries: on the MI300A a race first write loses ~12 CAS per
    // chunk at S = 1024 and exhausts even this arena, the replay must absorb it.
    constexpr unsigned long long roomy_append = 1ULL << 20;
    run([&] { random_rounds<256>(seed, 1L << 20, 2L << 12, roomy_append, "roomy arena"); });
    run([&] { random_rounds<1024>(seed + 1, 1L << 20, 2L << 10, roomy_append, "roomy arena"); });
    run([&] { random_rounds<256>(seed + 2, (1L << 18) + 77, 16, 1L << 18, "small arena, tail chunk"); });
    run([&] { random_rounds<1024>(seed + 3, 1L << 18, 4, 1L << 18, "small arena"); });
    run([&] { random_rounds<256>(seed + 4, 1L << 20, 1L << 12, 1L << 16, "exact arena", sa::Coordination::claim); });
    run([&] { random_rounds<1024>(seed + 5, 1L << 20, 1L << 10, 1L << 16, "exact arena", sa::Coordination::claim); });
    run([&]
        { random_rounds<256>(seed + 6, (1L << 18) + 77, 16, 1L << 18, "small arena, tail chunk", sa::Coordination::claim); });
    run([&]
        {
            random_rounds<256>(seed + 10, 1L << 20, 1L << 12, 1L << 16, "exact arena", sa::Coordination::claim,
                               sa::WriteLoad::plain);
        });
    run([&]
        {
            random_rounds<1024>(seed + 11, (1L << 18) + 77, 16, 1L << 18, "small arena, tail chunk",
                                sa::Coordination::claim, sa::WriteLoad::relaxed, true);
        });
    run([&]
        {
            random_rounds<256>(seed + 12, 1L << 20, 2L << 12, roomy_append, "roomy arena", sa::Coordination::race,
                               sa::WriteLoad::plain, true);
        });
    run([&] { chunk_rounds<256>(seed + 7, 1L << 20, 1L << 12, 1L << 16, "exact arena"); });
    run([&] { chunk_rounds<1024>(seed + 8, (1L << 20) + 300, (1L << 10) + 1, 1L << 16, "exact arena, tail chunk"); });
    run([&] { chunk_rounds<256>(seed + 9, 1L << 18, 16, 1L << 18, "small arena"); });
    run([&] { chunk_rounds<256>(seed + 13, 1L << 20, 1L << 12, 1L << 16, "exact arena", true); });
    run([&] { chunk_rounds<1024>(seed + 14, (1L << 20) + 300, (1L << 10) + 1, 1L << 16, "exact arena, tail chunk", true); });
    run([&] { chunk_rounds<256>(seed + 15, 1L << 18, 16, 1L << 18, "small arena", true); });
    run([&] { append_overflow(); });
    std::printf("%s: %d failure(s)\n", g_failures == 0 ? "PASSED" : "FAILED", g_failures);
    return g_failures == 0 ? 0 : 1;
}
