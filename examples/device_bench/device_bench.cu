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
 * @file device_bench.cu
 * @brief device::Array (chunk/device/) against a flat device array: the measurements of the
 *        stage-0 prototype (gpu/proto/proto_bench.cu, docs/gpu_design.md section 13) on the
 *        library's code, to check that the port lost nothing.
 *
 * @details For S = 256 and 1024: read (every chunk materialised, every chunk compressed);
 *          the chunk-aligned write (first write over a compressed array, write into
 *          materialised chunks), with finish() inside the timing; random range rewrites by
 *          a kernel calling Writer::write, then compress(); compress() alone, every chunk
 *          folding back and none; then the host <-> device conversion (assign() of a fully
 *          materialised state, image()) on 1 GiB at most.
 *
 *          The stage-0 criteria: read <= 1.5x the flat array, write into materialised chunks
 *          <= 2x, compress() of 10 GB <= 10 ms at 3.35 TB/s (reported as a fraction of this
 *          device's measured stream read, which carries over between GPUs). Times are
 *          wall-clock around calls that wait for the device, medians of the repetitions.
 *
 *          Usage: device_bench [log2(elements), default 27] [repetitions, default 10]
 *                              [log2(elements) of the conversion, default min(that, 27)]
 *          The conversion holds two host copies of its array (the source and the image):
 *          a job with little host memory takes a smaller one (MI300A: killed at 1 GiB).
 */

#include <algorithm> // std::sort, std::shuffle
#include <cstdint>   // std::int64_t
#include <cstdio>    // std::printf
#include <cstdlib>   // std::atoi
#include <ctime>     // std::timespec_get (not <chrono>: see now_ms)
#include <exception> // std::exception
#include <random>    // std::mt19937_64
#include <vector>    // std::vector

#include "chunk/device/array.h"   // device::Array, View, Writer
#include "chunk/device/backend.h" // allocate, copy, release, synchronize

namespace sd = shrink_array::device;

namespace
{

constexpr unsigned int grid      = 65536;
constexpr unsigned int read_grid = 4096; ///< one partial sum per thread: 8 MiB, not 128
constexpr unsigned int block     = 256;

struct Ramp
{
    double base;

    __device__ double operator()(std::int64_t i) const noexcept
    {
        return base + static_cast<double>(i);
    }
};

struct Constant
{
    double value;

    __device__ double operator()(std::int64_t /*i*/) const noexcept
    {
        return value;
    }
};

struct Range
{
    std::int64_t lo;
    std::int64_t hi;
};

__global__ void flat_write(double* a, std::int64_t n, double base)
{
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x; i < n; i += stride)
    {
        a[i] = base + static_cast<double>(i);
    }
}

__global__ void flat_read(const double* a, std::int64_t n, double* out)
{
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    const std::int64_t tid    = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    double             s      = 0;
    for (std::int64_t i = tid; i < n; i += stride)
    {
        s += a[i];
    }
    out[tid] = s;
}

template <unsigned int S>
__global__ void view_read(sd::View<double, S> v, double* out)
{
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    const std::int64_t tid    = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    double             s      = 0;
    for (std::int64_t i = tid; i < v.size(); i += stride)
    {
        s += v(i);
    }
    out[tid] = s;
}

__global__ void flat_ranges(double* a, const Range* r, std::int64_t n_ranges)
{
    for (std::int64_t k = blockIdx.x; k < n_ranges; k += gridDim.x)
    {
        for (std::int64_t i = r[k].lo + threadIdx.x; i < r[k].hi; i += blockDim.x)
        {
            a[i] = 3.0 + static_cast<double>(i);
        }
    }
}

/// @brief Each block takes ranges and writes their chunks with Writer::write.
template <unsigned int S>
__global__ void __launch_bounds__(block) view_ranges(sd::Writer<double, S> w, const Range* r, std::int64_t n_ranges)
{
    const Ramp f{3.0};
    for (std::int64_t k = blockIdx.x; k < n_ranges; k += gridDim.x)
    {
        for (std::int64_t c = r[k].lo / S; c * S < r[k].hi; ++c)
        {
            const std::int64_t lo = c * S > r[k].lo ? c * S : r[k].lo;
            const std::int64_t hi = (c + 1) * S < r[k].hi ? (c + 1) * S : r[k].hi;
            w.template write<static_cast<int>(block)>(c, lo, hi, f);
        }
    }
}

/// @brief A monotonic-enough wall clock in milliseconds. Not <chrono>: in C++20 it includes
///        <format>, whose GCC 14.1 headers (bits/unicode.h, `++this`) ROCm's clang rejects once
///        it is pointed at that GCC (--gcc-install-dir, MI300A).
double now_ms()
{
    std::timespec ts{};
    (void)std::timespec_get(&ts, TIME_UTC);
    return (static_cast<double>(ts.tv_sec) * 1e3) + (static_cast<double>(ts.tv_nsec) * 1e-6);
}

/// @brief Milliseconds taken by @p f, which waits for the device.
template <typename F>
double time_once(F&& f)
{
    sd::synchronize();
    const double t0 = now_ms();
    f();
    sd::synchronize();
    return now_ms() - t0;
}

double median(std::vector<double> t)
{
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}

/// @brief Median of @p reps runs of @p f, each after @p setup (not timed).
template <typename Setup, typename F>
double time_median(int reps, Setup&& setup, F&& f)
{
    std::vector<double> t;
    for (int r = 0; r < reps; ++r)
    {
        setup();
        t.push_back(time_once(f));
    }
    return median(t);
}

void line(const char* what, double ms, double flat_ms, const char* note)
{
    std::printf("  %-50s %9.3f ms  %7.2fx flat  %s\n", what, ms, ms / flat_ms, note);
}

template <unsigned int S>
void bench(std::int64_t n, int reps)
{
    const std::int64_t n_chunks = (n + S - 1) / S;
    const double       gib      = static_cast<double>(n) * sizeof(double) / (1 << 30);
    std::printf("\n== S=%u, %lld doubles (%.2f GiB) ==\n", S, static_cast<long long>(n), gib);
    const auto  nothing = [] {};
    double*     flat    = sd::allocate<double>(static_cast<std::size_t>(n));
    double*     sums    = sd::allocate<double>(static_cast<std::size_t>(read_grid) * block);
    const double flat_w = time_median(reps, nothing, [&] { flat_write<<<grid, block>>>(flat, n, 1.0); });
    const double flat_r = time_median(reps, nothing, [&] { flat_read<<<read_grid, block>>>(flat, n, sums); });
    const double bytes  = static_cast<double>(n) * sizeof(double);
    const double read_gbs = bytes / (flat_r * 1e6);
    std::printf("  flat: read %.1f GB/s, write %.1f GB/s\n", read_gbs, bytes / (flat_w * 1e6));

    sd::Array<double, S> a(n, static_cast<unsigned long long>(n_chunks + n_chunks / 8));
    const auto           fold_all = [&]
    {
        a.parallel_for_chunks(0, n, Constant{0.0});
        (void)a.compress();
    };

    // 1. The chunk-aligned write: first write (every chunk compressed before), then into
    //    materialised chunks.
    line("first write, chunk-aligned (+ finish)", time_median(reps, fold_all, [&] { a.parallel_for_chunks(0, n, Ramp{1.0}); }),
         flat_w, "");
    line("write into materialised chunks, chunk-aligned", time_median(reps, nothing, [&] { a.parallel_for_chunks(0, n, Ramp{2.0}); }),
         flat_w, "(criterion <= 2x)");

    // 2. Read, every chunk materialised (as left above), then every chunk compressed.
    line("read, every chunk materialised",
         time_median(reps, nothing, [&] { view_read<S><<<read_grid, block>>>(a.view(), sums); }), flat_r,
         "(criterion <= 1.5x)");
    fold_all();
    line("read, every chunk compressed", time_median(reps, nothing, [&] { view_read<S><<<read_grid, block>>>(a.view(), sums); }),
         flat_r, "");

    // 3. Random range rewrites (lengths up to 4 chunks, non-overlapping, shuffled) over an
    //    array half materialised; then compress().
    std::mt19937_64    rng(12345);
    std::vector<Range> ranges;
    for (std::int64_t lo = 0; lo < n;)
    {
        const std::int64_t hi = std::min<std::int64_t>(n, lo + 1 + static_cast<std::int64_t>(rng() % (4ULL * S)));
        if (rng() % 3 == 0)
        {
            ranges.push_back({lo, hi});
        }
        lo = hi + static_cast<std::int64_t>(rng() % (2ULL * S));
    }
    std::shuffle(ranges.begin(), ranges.end(), rng);
    const auto n_ranges = static_cast<std::int64_t>(ranges.size());
    Range*     d_ranges = sd::allocate<Range>(ranges.size());
    sd::copy(d_ranges, ranges.data(), ranges.size());
    const double flat_rng = time_median(reps, nothing, [&] { flat_ranges<<<grid, block>>>(flat, d_ranges, n_ranges); });
    const auto   half     = [&]
    {
        fold_all();
        a.parallel_for_chunks(0, n / 2, Ramp{5.0});
    };
    std::vector<double> tw, tc;
    for (int r = 0; r < reps; ++r)
    {
        half();
        tw.push_back(time_once(
            [&]
            {
                view_ranges<S><<<grid, block>>>(a.writer(), d_ranges, n_ranges);
                a.finish(Ramp{3.0});
            }));
        tc.push_back(time_once([&] { (void)a.compress(); }));
    }
    std::printf("  random ranges: %lld ranges, %llu/%lld chunks materialised after compress()\n",
                static_cast<long long>(n_ranges), a.materialised(), static_cast<long long>(n_chunks));
    line("  write kernel + finish()", median(tw), flat_rng, "");
    line("  compress()", median(tc), flat_rng, "");
    line("  write + finish() + compress()", median(tw) + median(tc), flat_rng, "");
    sd::release(d_ranges);

    // 4. compress() alone: every chunk materialised and uniform (all fold), then every
    //    chunk materialised and varied (a scan; it stops at the first difference).
    const auto report = [&](const char* what, double ms)
    {
        const double gbs = bytes / (ms * 1e6);
        std::printf("  %-50s %9.3f ms  %7.1f GB/s = %4.0f%% of stream read; 10 GB at 3.35 TB/s: %5.1f ms "
                    "(criterion <= 10 ms)\n",
                    what, ms, gbs, 100.0 * gbs / read_gbs, 10e9 / (gbs / read_gbs * 3.35e12) * 1e3);
    };
    report("compress(), every chunk folds back",
           time_median(
               reps,
               [&]
               {
                   a.parallel_for_chunks(0, n, Ramp{1.0});
                   a.parallel_for_chunks(0, n, Constant{4.0});
               },
               [&] { (void)a.compress(); }));
    report("compress(), nothing folds (scan)",
           time_median(reps, [&] { a.parallel_for_chunks(0, n, Ramp{1.0}); }, [&] { (void)a.compress(); }));

    sd::release(flat);
    sd::release(sums);
}

/// @brief The host <-> device conversion of a fully materialised array of @p n elements, on
///        an array of its own once bench() has freed its memory: on the MI300A the host and
///        the device share the HBM, and the 8 GiB run (flat array, arena, host copy, image)
///        was killed for lack of memory.
template <unsigned int S>
void bench_conversion(std::int64_t n, int reps)
{
    const std::int64_t   n_chunks = (n + S - 1) / S;
    const double         bytes    = static_cast<double>(n) * sizeof(double);
    const auto           nothing  = [] {};
    sd::Array<double, S> a(n, static_cast<unsigned long long>(n_chunks));
    std::printf("\n== conversion, S=%u, %lld doubles (%.2f GiB), every chunk materialised ==\n", S,
                static_cast<long long>(n), bytes / (1 << 30));
    std::vector<double> host(static_cast<std::size_t>(n));
    for (std::size_t i = 0; i < host.size(); ++i)
    {
        host[i] = static_cast<double>(i);
    }
    shrink_array::ArrayState<double> state;
    state.size       = host.size();
    state.chunk_size = S;
    state.chunks.resize(static_cast<std::size_t>(n_chunks));
    for (std::size_t c = 0; c < state.chunks.size(); ++c)
    {
        state.chunks[c] = {shrink_array::ChunkKind::materialised, 0.0, host.data() + c * S};
    }
    // A last partial chunk would read past host's end in assign(): give it its own buffer.
    std::vector<double> last(S, 0.0);
    if (n % S != 0)
    {
        std::copy(host.end() - (n % S), host.end(), last.begin());
        state.chunks.back().data = last.data();
    }
    const double t_assign = time_median(reps, nothing, [&] { a.assign(state); });
    const double t_image  = time_median(reps, nothing, [&] { (void)a.image(); });
    std::printf("  %-50s %9.3f ms  %7.1f GB/s\n", "assign() (host -> device)", t_assign, bytes / (t_assign * 1e6));
    std::printf("  %-50s %9.3f ms  %7.1f GB/s\n", "image() (device -> host)", t_image, bytes / (t_image * 1e6));
}

} // namespace

int main(int argc, char** argv)
{
    const int          log2n = argc > 1 ? std::atoi(argv[1]) : 27;
    const int          reps  = argc > 2 ? std::atoi(argv[2]) : 10;
    const std::int64_t n     = std::int64_t{1} << log2n;
    std::printf("device_bench: device::Array against a flat array, %d repetitions, medians\n", reps);
    try
    {
        bench<256>(n, reps);
        bench<1024>(n, reps);
        const int          log2c  = argc > 3 ? std::atoi(argv[3]) : (log2n < 27 ? log2n : 27);
        const std::int64_t n_conv = std::int64_t{1} << log2c;
        bench_conversion<256>(n_conv, reps);
        bench_conversion<1024>(n_conv, reps);
    }
    catch (const std::exception& e)
    {
        std::printf("error: %s\n", e.what());
        return 2;
    }
    return 0;
}
