// Copyright 2026 CEA (www.cea.fr)
// SPDX-License-Identifier: Apache-2.0

/**
 * @file proto_bench.cu
 * @brief The stage-0 measurements of docs/gpu_design.md section 13, against a flat array:
 *        streaming read, writes into materialised chunks, random range rewrites (as
 *        shrink_bench does) followed by compress(), and compress() alone.
 *
 * @details Criteria of stage 0: read <= 1.5x the flat array; write into a materialised
 *          chunk <= 2x; compress() of 10 GB materialised <= 10 ms on an H100-class GPU
 *          (about 1 TB/s, i.e. ~30% of a 3.35 TB/s HBM3 peak). The last one is reported as a
 *          fraction of this device's measured stream bandwidth, which carries over from a
 *          laptop GPU to a GH200 or an MI300A; the times themselves do not.
 *
 *          Usage: proto_bench [log2(elements), default 27] [reps, default 10]
 */

#include <algorithm>
#include <cstdio>
#include <random>
#include <vector>

#include "zero_any_device.h"

namespace sa = shrink_array::device;

namespace
{

#if defined(__HIP_PLATFORM_AMD__)
using GpuEvent = hipEvent_t;
    #define gpuEventCreate hipEventCreate
    #define gpuEventRecord hipEventRecord
    #define gpuEventSynchronize hipEventSynchronize
    #define gpuEventElapsedTime hipEventElapsedTime
    #define gpuEventDestroy hipEventDestroy
#else
using GpuEvent = cudaEvent_t;
    #define gpuEventCreate cudaEventCreate
    #define gpuEventRecord cudaEventRecord
    #define gpuEventSynchronize cudaEventSynchronize
    #define gpuEventElapsedTime cudaEventElapsedTime
    #define gpuEventDestroy cudaEventDestroy
#endif

/// @brief The flat baseline, with the same interface as the view.
struct Flat
{
    double* p;
    long    n;

    __device__ double operator()(long i) const noexcept
    {
        return p[i];
    }
    __device__ void assign(long i, double v) const noexcept
    {
        p[i] = v;
    }
};

template <typename A>
__global__ void read_sum(A a, long n, double* out)
{
    double s = 0;
    for (long i = blockIdx.x * static_cast<long>(blockDim.x) + threadIdx.x; i < n;
         i += static_cast<long>(gridDim.x) * blockDim.x)
    {
        s += a(i);
    }
    for (int d = sa::warp_size / 2; d > 0; d /= 2)
    {
        s += sa::shuffle(sa::active_lanes(), s, (sa::lane_id() + d) % sa::warp_size);
    }
    if (sa::lane_id() == 0)
    {
        atomicAdd(out, s);
    }
}

template <typename A>
__global__ void write_ramp(A a, long n, double base)
{
    for (long i = blockIdx.x * static_cast<long>(blockDim.x) + threadIdx.x; i < n;
         i += static_cast<long>(gridDim.x) * blockDim.x)
    {
        a.assign(i, base + 0.5 * static_cast<double>(i & 1023));
    }
}

/// @brief write_ramp with four descriptors loaded ahead per thread: tests whether the
///        write path is bound by one descriptor load in flight per warp (the store into a
///        buffer may alias the descriptors, so a compiler cannot hoist the next load above it).
template <int S>
__global__ void write_ramp_ahead(sa::View<double, S> a, long n, double base)
{
    constexpr int U      = 4;
    const long    stride = static_cast<long>(gridDim.x) * blockDim.x;
    for (long i = blockIdx.x * static_cast<long>(blockDim.x) + threadIdx.x; i < n; i += U * stride)
    {
        double* buf[U];
#pragma unroll
        for (int k = 0; k < U; ++k)
        {
            const long j = i + k * stride;
            buf[k]       = j < n ? sa::load_ptr_relaxed(&a.desc[j >> sa::View<double, S>::shift].buf) : nullptr;
        }
#pragma unroll
        for (int k = 0; k < U; ++k)
        {
            const long   j = i + k * stride;
            const double v = base + 0.5 * static_cast<double>(j & 1023);
            if (j < n)
            {
                if (buf[k] != nullptr && buf[k] != sa::busy_mark<double>())
                {
                    buf[k][j & sa::View<double, S>::mask] = v;
                }
                else
                {
                    a.assign(j, v);
                }
            }
        }
    }
}

template <typename A>
__global__ void write_value(A a, long n, double v)
{
    for (long i = blockIdx.x * static_cast<long>(blockDim.x) + threadIdx.x; i < n;
         i += static_cast<long>(gridDim.x) * blockDim.x)
    {
        a.assign(i, v);
    }
}

/// @brief A range rewrite as shrink_bench's: kind 0 = zeros, 1 = one constant, 2 = varying values.
struct Range
{
    long start;
    long length;
    int  kind;
};

template <typename A>
__global__ void write_ranges(A a, const Range* ranges, long n_ranges)
{
    for (long r = blockIdx.x; r < n_ranges; r += gridDim.x)
    {
        const Range g = ranges[r];
        for (long k = threadIdx.x; k < g.length; k += blockDim.x)
        {
            const double v = g.kind == 0 ? 0.0 : g.kind == 1 ? 3.0 : 1.0 + static_cast<double>((g.start + k) & 255);
            a.assign(g.start + k, v);
        }
    }
}

/// @brief write_ramp's values, for the chunk-aligned loop (DeviceArray::parallel_for_chunks).
struct Ramp
{
    double base;

    __device__ double operator()(long i) const noexcept
    {
        return base + 0.5 * static_cast<double>(i & 1023);
    }
};

/// @brief write_ranges' values for one range.
struct RangeValue
{
    long start;
    int  kind;

    __device__ double operator()(long i) const noexcept
    {
        return kind == 0 ? 0.0 : kind == 1 ? 3.0 : 1.0 + static_cast<double>(i & 255);
    }
};

/// @brief write_ranges by the chunk-aligned path: each block takes a range and goes over
///        its chunks with block_write (two ranges may share a chunk).
template <int BLOCK, int S>
__global__ void __launch_bounds__(BLOCK) write_ranges_chunked(sa::View<double, S> a, const Range* ranges, long n_ranges)
{
    for (long r = blockIdx.x; r < n_ranges; r += gridDim.x)
    {
        const Range      g   = ranges[r];
        const long       end = g.start + g.length;
        const RangeValue f{g.start, g.kind};
        for (long c = g.start / S; c * S < end; ++c)
        {
            const long lo = c * S > g.start ? c * S : g.start;
            const long hi = (c + 1) * S < end ? (c + 1) * S : end;
            a.template block_write<BLOCK>(c, lo, hi, f);
        }
    }
}

/// @brief Median time in ms of @p reps runs of @p launch (one warm-up first).
template <typename F>
float time_ms(int reps, F&& launch)
{
    GpuEvent e0, e1;
    SA_GPU(gpuEventCreate(&e0));
    SA_GPU(gpuEventCreate(&e1));
    launch();
    SA_GPU(gpuDeviceSynchronize());
    std::vector<float> t;
    for (int r = 0; r < reps; ++r)
    {
        SA_GPU(gpuEventRecord(e0));
        launch();
        SA_GPU(gpuEventRecord(e1));
        SA_GPU(gpuEventSynchronize(e1));
        float ms = 0;
        SA_GPU(gpuEventElapsedTime(&ms, e0, e1));
        t.push_back(ms);
    }
    (void)gpuEventDestroy(e0);
    (void)gpuEventDestroy(e1);
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}

/// @brief Time in ms of one run of @p launch, with no warm-up: for the operations whose
///        first run changes the state (compress(), a first write that materialises).
template <typename F>
float time_once(F&& launch)
{
    GpuEvent e0, e1;
    SA_GPU(gpuEventCreate(&e0));
    SA_GPU(gpuEventCreate(&e1));
    SA_GPU(gpuDeviceSynchronize());
    SA_GPU(gpuEventRecord(e0));
    launch();
    SA_GPU(gpuEventRecord(e1));
    SA_GPU(gpuEventSynchronize(e1));
    float ms = 0;
    SA_GPU(gpuEventElapsedTime(&ms, e0, e1));
    (void)gpuEventDestroy(e0);
    (void)gpuEventDestroy(e1);
    return ms;
}

/// @brief Median of a vector of times.
float median(std::vector<float> t)
{
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}

constexpr unsigned grid  = 2048;
constexpr unsigned block = 256;

void line(const char* what, float ms, float ref_ms, const char* criterion)
{
    std::printf("  %-44s %9.3f ms  %6.2fx flat  %s\n", what, ms, ms / ref_ms, criterion);
}

template <int S>
void bench(long n, int reps)
{
    const double gib = static_cast<double>(n) * sizeof(double) / (1024.0 * 1024 * 1024);
    std::printf("\n== S=%d, %ld doubles (%.2f GiB) ==\n", S, n, gib);
    const long n_chunks = (n + S - 1) / S;

    double* flat_p = nullptr;
    double* d_sum  = nullptr;
    SA_GPU(gpuMalloc(&flat_p, sizeof(double) * n));
    SA_GPU(gpuMalloc(&d_sum, sizeof(double)));
    const Flat flat{flat_p, n};

    // Room for every chunk plus a quarter for the buffers of lost CAS: a grid-stride write
    // puts S/32 warps on each chunk at once, and their losers wait for compress().
    sa::DeviceArray<double, S> a(n, static_cast<unsigned long long>(n_chunks + n_chunks / 4), 1UL << 20);

    // Stream bandwidth of this device: the flat copy-free reference (read + write).
    write_ramp<<<grid, block>>>(flat, n, 1.0);
    const float  flat_write = time_ms(reps, [&] { write_ramp<<<grid, block>>>(flat, n, 1.0); });
    const float  flat_read  = time_ms(reps, [&] { read_sum<<<grid, block>>>(flat, n, d_sum); });
    const double read_gbs   = static_cast<double>(n) * sizeof(double) / (flat_read * 1e6);
    std::printf("  flat: read %.1f GB/s, write %.1f GB/s\n", read_gbs,
                static_cast<double>(n) * sizeof(double) / (flat_write * 1e6));

    // 0. The first write over the whole array (every chunk compressed before): it
    //    materialises every chunk. Grid-stride assign() puts S/32 warps on each chunk at
    //    once (race: they all pre-fill; claim: one pre-fills, the others wait); the
    //    chunk-aligned loop gives each chunk to one block.
    const char* mode_name[] = {"race", "claim", "chunk-aligned"};
    for (int mode = 0; mode < 3; ++mode)
    {
        a.set_coordination(mode == 1 ? sa::Coordination::claim : sa::Coordination::race);
        std::vector<float> t;
        sa::Stats          before{}, after{};
        for (int r = 0; r < reps; ++r)
        {
            write_value<<<grid, block>>>(a.view(), n, 0.0);
            a.finalize();
            before = a.stats();
            t.push_back(time_once(
                [&]
                {
                    if (mode == 2)
                    {
                        a.parallel_for_chunks(0, n, Ramp{1.0});
                    }
                    else
                    {
                        write_ramp<<<grid, block>>>(a.view(), n, 1.0);
                    }
                }));
            after = a.stats();
        }
        char what[64];
        std::snprintf(what, sizeof what, "first write, %s", mode_name[mode]);
        line(what, median(t), flat_write, "");
        std::printf("  %-44s lost CAS %.2f, claim waits %.2f per chunk; %llu given up, %llu arena failures\n", "",
                    static_cast<double>(after.cas_lost - before.cas_lost) / static_cast<double>(n_chunks),
                    static_cast<double>(after.claim_wait - before.claim_wait) / static_cast<double>(n_chunks),
                    after.claim_spin - before.claim_spin, after.arena_fail - before.arena_fail);
        write_value<<<grid, block>>>(a.view(), n, 0.0);
        a.finalize();
    }
    a.set_coordination(sa::Coordination::race);

    // 1. Read, every chunk compressed (descriptors only), then every chunk materialised.
    const float read_compressed = time_ms(reps, [&] { read_sum<<<grid, block>>>(a.view(), n, d_sum); });
    write_ramp<<<grid, block>>>(a.view(), n, 1.0);
    a.finalize();
    const float read_materialised = time_ms(reps, [&] { read_sum<<<grid, block>>>(a.view(), n, d_sum); });
    line("read, every chunk materialised", read_materialised, flat_read, "(criterion <= 1.5x)");
    line("read, every chunk compressed", read_compressed, flat_read, "");

    // 2. Write into materialised chunks (the hot path: acquire load + store).
    const char* load_name[] = {"acquire", "relaxed", "plain"};
    for (int w = 0; w < 3; ++w)
    {
        a.set_write_load(static_cast<sa::WriteLoad>(w));
        const float write_materialised = time_ms(reps, [&] { write_ramp<<<grid, block>>>(a.view(), n, 2.0); });
        char        what[64];
        std::snprintf(what, sizeof what, "write into materialised chunks, %s load", load_name[w]);
        line(what, write_materialised, flat_write, "(criterion <= 2x)");
    }
    a.set_write_load(sa::WriteLoad::acquire);
    const float write_ahead = time_ms(reps, [&] { write_ramp_ahead<S><<<grid, block>>>(a.view(), n, 2.0); });
    line("write into materialised chunks, 4 loads ahead", write_ahead, flat_write, "");
    const float write_materialised_chunked = time_ms(reps, [&] { a.parallel_for_chunks(0, n, Ramp{2.0}); });
    line("write into materialised chunks, chunk-aligned", write_materialised_chunked, flat_write, "");

    // 3. Random range rewrites, as shrink_bench's phases: lengths up to 4 chunks, a third of
    //    the ranges zeros, a third one constant, a third varying; then compress().
    std::mt19937_64    rng(7);
    const long         n_ranges = n / (2L * S);
    std::vector<Range> ranges(static_cast<std::size_t>(n_ranges));
    for (auto& r : ranges)
    {
        r.length = 1 + static_cast<long>(rng() % (4UL * S));
        r.start  = static_cast<long>(rng() % static_cast<unsigned long>(n - r.length));
        r.kind   = static_cast<int>(rng() % 3);
    }
    // Ranges must not overlap within one kernel (a race of the user program): keep a
    // non-overlapping subset, in a random order.
    std::sort(ranges.begin(), ranges.end(), [](const Range& x, const Range& y) { return x.start < y.start; });
    std::vector<Range> kept;
    long               end = 0;
    for (const auto& r : ranges)
    {
        if (r.start >= end)
        {
            kept.push_back(r);
            end = r.start + r.length;
        }
    }
    std::shuffle(kept.begin(), kept.end(), rng);
    Range* d_ranges = nullptr;
    SA_GPU(gpuMalloc(&d_ranges, sizeof(Range) * kept.size()));
    SA_GPU(gpuMemcpy(d_ranges, kept.data(), sizeof(Range) * kept.size(), gpuMemcpyHostToDevice));
    const auto  n_kept   = static_cast<long>(kept.size());
    const float flat_rng = time_ms(reps, [&] { write_ranges<<<grid, block>>>(flat, d_ranges, n_kept); });
    // The array starts half materialised (every other chunk at a varying ramp) for each rep.
    for (int mode = 0; mode < 3; ++mode)
    {
        a.set_coordination(mode == 1 ? sa::Coordination::claim : sa::Coordination::race);
        std::vector<float> tw, tc;
        for (int r = 0; r < reps; ++r)
        {
            write_value<<<grid, block>>>(a.view(), n, 0.0);
            a.finalize();
            write_ramp<<<grid, block>>>(a.view(), n / 2, 5.0);
            a.finalize();
            tw.push_back(time_once(
                [&]
                {
                    if (mode == 2)
                    {
                        write_ranges_chunked<block, S><<<grid, block>>>(a.view(), d_ranges, n_kept);
                    }
                    else
                    {
                        write_ranges<<<grid, block>>>(a.view(), d_ranges, n_kept);
                    }
                }));
            tc.push_back(time_once([&] { a.finalize(); }));
        }
        const float     rng_write    = median(tw);
        const float     rng_compress = median(tc);
        const sa::Stats st           = a.stats();
        std::printf("  random ranges, %s: %ld non-overlapping ranges; after compress %llu/%ld chunks "
                    "materialised; so far %llu lost CAS, %llu claim waits, %llu given up, %llu replays\n",
                    mode_name[mode], n_kept, st.materialised, st.n_chunks, st.cas_lost, st.claim_wait, st.claim_spin,
                    st.replays);
        line("  write kernel", rng_write, flat_rng, "");
        line("  finalize() (compress)", rng_compress, flat_rng, "");
        line("  write + finalize()", rng_write + rng_compress, flat_rng, "");
    }
    a.set_coordination(sa::Coordination::race);

    // 4. compress() alone: every chunk materialised and uniform (all fold back), then every
    //    chunk materialised and not uniform (a pure scan, nothing folds back).
    const double bytes  = static_cast<double>(n) * sizeof(double);
    const auto   report = [&](const char* what, float ms)
    {
        const double gbs = bytes / (ms * 1e6);
        std::printf("  %-44s %9.3f ms  %7.1f GB/s = %4.0f%% of stream read; 10 GB at 3.35 TB/s: %5.1f ms "
                    "(criterion <= 10 ms)\n",
                    what, ms, gbs, 100.0 * gbs / read_gbs, 10e9 / (gbs / read_gbs * 3.35e12) * 1e3);
    };
    for (int per_warp = 0; per_warp < 2; ++per_warp)
    {
        a.set_compress_per_warp(per_warp != 0);
        std::vector<float> t;
        for (int r = 0; r < reps; ++r)
        {
            write_ramp<<<grid, block>>>(a.view(), n, 1.0); // materialise everything
            a.finalize();
            write_value<<<grid, block>>>(a.view(), n, 4.0); // uniform again, still materialised
            SA_GPU(gpuDeviceSynchronize());
            t.push_back(time_once([&] { a.compress(); }));
        }
        write_ramp<<<grid, block>>>(a.view(), n, 1.0);
        a.finalize();
        const float scan = time_ms(reps, [&] { a.compress(); });
        char        what[64];
        std::snprintf(what, sizeof what, "compress() by %s, every chunk folds back", per_warp ? "warp" : "block");
        report(what, median(t));
        std::snprintf(what, sizeof what, "compress() by %s, nothing folds (scan)", per_warp ? "warp" : "block");
        report(what, scan);
    }

    (void)gpuFree(d_ranges);
    (void)gpuFree(flat_p);
    (void)gpuFree(d_sum);
}

} // namespace

int main(int argc, char** argv)
{
    const int  log2n = argc > 1 ? std::atoi(argv[1]) : 27;
    const int  reps  = argc > 2 ? std::atoi(argv[2]) : 10;
    const long n     = 1L << log2n;
#if defined(__HIP_PLATFORM_AMD__)
    hipDeviceProp_t p{};
    SA_GPU(hipGetDeviceProperties(&p, 0));
#else
    cudaDeviceProp p{};
    SA_GPU(cudaGetDeviceProperties(&p, 0));
#endif
    std::printf("proto_bench on %s (%d SMs, %.1f GB), %d reps, median times\n", p.name, p.multiProcessorCount,
                static_cast<double>(p.totalGlobalMem) / 1e9, reps);
    try
    {
        bench<256>(n, reps);
        bench<1024>(n, reps);
    }
    catch (const std::exception& e)
    {
        std::printf("error: %s\n", e.what());
        return 2;
    }
    return 0;
}
