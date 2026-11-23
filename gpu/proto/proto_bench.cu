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
    //    materialises every chunk, S/32 warps reaching each chunk at once.
    {
        std::vector<float> t;
        unsigned long long lost_before = 0, lost = 0;
        for (int r = 0; r < reps; ++r)
        {
            write_value<<<grid, block>>>(a.view(), n, 0.0);
            a.finalize();
            lost_before = a.stats().cas_lost;
            t.push_back(time_once([&] { write_ramp<<<grid, block>>>(a.view(), n, 1.0); }));
            lost = a.stats().cas_lost - lost_before;
        }
        line("first write (materialises every chunk)", median(t), flat_write, "");
        std::printf("  %-44s %llu lost CAS for %ld chunks (%.1f per chunk)\n", "", lost, n_chunks,
                    static_cast<double>(lost) / static_cast<double>(n_chunks));
        write_value<<<grid, block>>>(a.view(), n, 0.0);
        a.finalize();
    }

    // 1. Read, every chunk compressed (descriptors only), then every chunk materialised.
    const float read_compressed = time_ms(reps, [&] { read_sum<<<grid, block>>>(a.view(), n, d_sum); });
    write_ramp<<<grid, block>>>(a.view(), n, 1.0);
    a.finalize();
    const float read_materialised = time_ms(reps, [&] { read_sum<<<grid, block>>>(a.view(), n, d_sum); });
    line("read, every chunk materialised", read_materialised, flat_read, "(criterion <= 1.5x)");
    line("read, every chunk compressed", read_compressed, flat_read, "");

    // 2. Write into materialised chunks (the hot path: acquire load + store).
    const float write_materialised = time_ms(reps, [&] { write_ramp<<<grid, block>>>(a.view(), n, 2.0); });
    line("write into materialised chunks", write_materialised, flat_write, "(criterion <= 2x)");

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
    float rng_write = 0, rng_compress = 0;
    {
        std::vector<float> tw, tc;
        for (int r = 0; r < reps; ++r)
        {
            write_value<<<grid, block>>>(a.view(), n, 0.0);
            a.finalize();
            write_ramp<<<grid, block>>>(a.view(), n / 2, 5.0);
            a.finalize();
            tw.push_back(time_once([&] { write_ranges<<<grid, block>>>(a.view(), d_ranges, n_kept); }));
            tc.push_back(time_once([&] { a.finalize(); }));
        }
        std::sort(tw.begin(), tw.end());
        std::sort(tc.begin(), tc.end());
        rng_write    = tw[tw.size() / 2];
        rng_compress = tc[tc.size() / 2];
    }
    const sa::Stats st = a.stats();
    std::printf("  %ld non-overlapping ranges; after compress %llu/%ld chunks materialised, %llu lost CAS so far\n",
                n_kept, st.materialised, st.n_chunks, st.cas_lost);
    line("random ranges: write kernel", rng_write, flat_rng, "");
    line("random ranges: finalize() (compress)", rng_compress, flat_rng, "");
    line("random ranges: write + finalize()", rng_write + rng_compress, flat_rng, "");

    // 4. compress() alone: every chunk materialised and uniform (all fold back), then every
    //    chunk materialised and not uniform (a pure scan, nothing folds back).
    const double bytes = static_cast<double>(n) * sizeof(double);
    float        fold  = 0;
    {
        std::vector<float> t;
        for (int r = 0; r < reps; ++r)
        {
            write_ramp<<<grid, block>>>(a.view(), n, 1.0); // materialise everything
            a.finalize();
            write_value<<<grid, block>>>(a.view(), n, 4.0); // uniform again, still materialised
            SA_GPU(gpuDeviceSynchronize());
            t.push_back(time_once([&] { a.compress(); }));
        }
        std::sort(t.begin(), t.end());
        fold = t[t.size() / 2];
    }
    write_ramp<<<grid, block>>>(a.view(), n, 1.0);
    a.finalize();
    const float scan   = time_ms(reps, [&] { a.compress(); });
    const auto  report = [&](const char* what, float ms)
    {
        const double gbs = bytes / (ms * 1e6);
        std::printf("  %-44s %9.3f ms  %7.1f GB/s = %4.0f%% of stream read; 10 GB at 3.35 TB/s: %5.1f ms "
                    "(criterion <= 10 ms)\n",
                    what, ms, gbs, 100.0 * gbs / read_gbs, 10e9 / (gbs / read_gbs * 3.35e12) * 1e3);
    };
    report("compress(), every chunk folds back", fold);
    report("compress(), nothing folds back (scan)", scan);

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
