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
 * @file device_api_validation.cu
 * @brief The per-variant names of the GPU arrays (chunk/device/zero_array.h,
 *        zero_any_array.h, zero_or_value_array.h) and the host arrays' API they share:
 *        DeviceArray(n) with its default arena, size(), chunk_size(), stats(),
 *        written_chunk_count(), compress(), and DeviceArray(state) from an image() of
 *        another; DeviceView in a read kernel, DeviceWriter in a write kernel.
 * @details For zero, zero_any and zero_or_value, double and int, S = 256 and 1024 (the
 *          default): a fresh array holds no buffer; a ramp over the first half materialises
 *          its chunks; the constant written over it folds them back at compress(); a value 3
 *          over the whole array (the chunk-aligned loop, then a DeviceWriter kernel) folds
 *          back in zero_any, in zero_or_value once 3 is its second value, never in zero; the
 *          image of the array
 *          built again reads the same and holds the same buffers. Each step compares every
 *          element with a host mirror, and the chunks holding a buffer with the count the
 *          variant's rule gives. Built with -DCHUNK_DEVICE_CORE_CHECK=ON, CUDA or HIP.
 */

#include <algorithm> // std::fill
#include <cstdint>   // std::int64_t
#include <cstdio>    // std::printf, std::snprintf
#include <exception> // std::exception
#include <vector>    // std::vector

#include "chunk/device/backend.h"             // allocate, copy, release, synchronize
#include "chunk/device/zero_any_array.h"      // zero_any::DeviceArray, DeviceView, DeviceWriter
#include "chunk/device/zero_array.h"          // zero::DeviceArray, DeviceView, DeviceWriter
#include "chunk/device/zero_or_value_array.h" // zero_or_value::DeviceArray, DeviceView, DeviceWriter

namespace sa = shrink_array;
namespace sd = shrink_array::device;

namespace
{

int g_failures = 0;

void expect(bool ok, const char* what)
{
    std::printf("  %-70s %s\n", what, ok ? "ok" : "FAILED");
    g_failures += ok ? 0 : 1;
}

/// @brief f(i) = 1 + i % 7, never 0 nor 3.
template <typename T>
struct Ramp
{
    __host__ __device__ T operator()(std::int64_t i) const noexcept
    {
        return static_cast<T>(1 + i % 7);
    }
};

/// @brief f(i) = c.
template <typename T>
struct Const
{
    T c;

    __host__ __device__ T operator()(std::int64_t) const noexcept
    {
        return c;
    }
};

/// @brief A read kernel through the per-variant view: out[i] = v(i).
template <typename ViewT, typename T>
__global__ void read_all(ViewT v, T* out)
{
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x; i < v.size(); i += stride)
    {
        out[i] = v(i);
    }
}

/// @brief A write kernel through the per-variant writer: w(i) = c on every odd element of [lo, hi).
template <typename WriterT, typename T>
__global__ void write_odd(WriterT w, std::int64_t lo, std::int64_t hi, T c)
{
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t i = lo + 1 + 2 * (blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x); i < hi;
         i += 2 * stride)
    {
        w.assign(i, c);
    }
}

/// @brief A write kernel through the per-variant writer: w(i) = c on every element.
template <typename WriterT, typename T>
__global__ void write_all(WriterT w, std::int64_t n, T c)
{
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x; i < n; i += stride)
    {
        w.assign(i, c);
    }
}

template <typename Arr, typename T>
long wrong(const Arr& a, const std::vector<T>& mirror)
{
    const std::vector<T> got = a.to_host();
    long                 bad = got.size() == mirror.size() ? 0 : 1;
    for (std::size_t i = 0; i < got.size() && i < mirror.size(); ++i)
    {
        bad += got[i] == mirror[i] ? 0 : 1;
    }
    return bad;
}

/// @brief One variant's names: Arr = <variant>::DeviceArray<T, S>, ViewT and WriterT its
///        DeviceView and DeviceWriter.
template <typename Arr, typename ViewT, typename WriterT>
void run(const char* name)
{
    using T              = typename Arr::value_type;
    constexpr auto     S = Arr::chunk_size();
    constexpr auto     V = Arr::variant;
    const std::int64_t n = 37 * static_cast<std::int64_t>(S) + 5; // a tail chunk
    const std::int64_t h = n / 2;                                 // not a multiple of S
    std::printf("[%s, %s, S=%u] n=%lld\n", name, sizeof(T) == 8 ? "double" : "int", S, static_cast<long long>(n));
    char line[160];

    Arr               a(n);
    std::vector<T>    mirror(static_cast<std::size_t>(n), T{});
    const std::int64_t chunks = (n + S - 1) / S;
    expect(a.size() == n && a.n_chunks() == chunks && Arr::chunk_size() == S, "DeviceArray(n): size, n_chunks, chunk_size");
    expect(a.written_chunk_count() == 0 && a.stats() == 0.0 && Arr::default_arena(n) >= 1,
           "a fresh array holds no buffer (written_chunk_count, stats)");

    // A fresh DeviceArray(n), every element written by assign(): the default arena holds every
    // chunk, so no write goes to the append-buffer and none is lost (a buffer per 64 chunks
    // lost 254029 of 262221 writes of a zero array on the GH200).
    {
        Arr            c(n);
        std::vector<T> all(static_cast<std::size_t>(n), T{3});
        write_all<WriterT, T><<<256, 256>>>(c.writer(), n, T{3});
        bool threw = false;
        try
        {
            c.finish();
        }
        catch (const std::exception&)
        {
            threw = true;
        }
        std::snprintf(line, sizeof line, "DeviceArray(n), assign() of every element: %s, %llu appended",
                      threw ? "writes lost" : "nothing lost", c.appended());
        expect(!threw && wrong(c, all) == 0 && c.appended() == 0, line);
    }

    // compress() right after a write kernel, no finish(): the fresh buffers it published are
    // settled first (from the fresh list), so a chunk written on its odd elements only reads
    // the constant on the even ones and does not fold.
    {
        Arr            c(n);
        std::vector<T> odd(static_cast<std::size_t>(n), T{});
        for (std::int64_t i = 1; i < n; i += 2)
        {
            odd[static_cast<std::size_t>(i)] = T{3};
        }
        write_odd<WriterT, T><<<256, 256>>>(c.writer(), 0, n, T{3});
        const auto folded = c.compress();
        std::snprintf(line, sizeof line, "compress() with no finish() after the kernel: %llu folded, %llu hold a buffer",
                      folded, c.written_chunk_count());
        expect(wrong(c, odd) == 0 && folded == 0 && c.written_chunk_count() == static_cast<unsigned long long>(chunks),
               line);
    }

    // compress_all() over two arrays: x uniform at 3 (folds in zero_any only: zero folds to
    // T{} alone, zero_or_value has no second value here), y at 3 on its odd elements (folds in
    // none); the count is the two arrays' and both read as before.
    {
        Arr            x(n);
        Arr            y(n);
        std::vector<T> all3(static_cast<std::size_t>(n), T{3});
        std::vector<T> odd3(static_cast<std::size_t>(n), T{});
        for (std::int64_t i = 1; i < n; i += 2)
        {
            odd3[static_cast<std::size_t>(i)] = T{3};
        }
        write_all<WriterT, T><<<256, 256>>>(x.writer(), n, T{3});
        x.finish();
        write_odd<WriterT, T><<<256, 256>>>(y.writer(), 0, n, T{3});
        y.finish();
        const auto folded = sd::compress_all(x, y);
        const auto want   = V == sd::Variant::zero_any ? static_cast<unsigned long long>(chunks) : 0ULL;
        std::snprintf(line, sizeof line, "compress_all(x, y): %llu folded (expected %llu), x %llu and y %llu hold a buffer",
                      folded, want, x.written_chunk_count(), y.written_chunk_count());
        expect(folded == want && wrong(x, all3) == 0 && wrong(y, odd3) == 0 &&
                   y.written_chunk_count() == static_cast<unsigned long long>(chunks),
               line);
    }

    // zero_or_value: chunks uniform at 3 that a pass could not fold (no second value), left
    // untouched, fold at the pass after set_second_value(3): compress() looks at the chunks
    // written since the last pass only, but at every one after a new second value.
    if constexpr (V == sd::Variant::zero_or_value)
    {
        Arr            z(n);
        std::vector<T> all3(static_cast<std::size_t>(n), T{3});
        write_all<WriterT, T><<<256, 256>>>(z.writer(), n, T{3});
        z.finish();
        const auto before = z.compress(); // no second value: nothing folds, the marks cleared
        z.set_second_value(T{3});
        const auto after = z.compress();
        std::snprintf(line, sizeof line, "set_second_value(3) on untouched chunks: %llu then %llu folded, %llu left",
                      before, after, z.written_chunk_count());
        expect(before == 0 && after == static_cast<unsigned long long>(chunks) && z.written_chunk_count() == 0 &&
                   wrong(z, all3) == 0,
               line);
    }

    // A ramp over the first half: its chunks materialise.
    a.parallel_for_chunks(0, h, Ramp<T>{});
    for (std::int64_t i = 0; i < h; ++i)
    {
        mirror[static_cast<std::size_t>(i)] = Ramp<T>{}(i);
    }
    const auto ramp_chunks = static_cast<unsigned long long>((h + S - 1) / S);
    std::snprintf(line, sizeof line, "ramp over the first half: %llu/%llu chunks hold a buffer, %.1f%%",
                  a.written_chunk_count(), ramp_chunks, a.stats());
    expect(wrong(a, mirror) == 0 && a.written_chunk_count() == ramp_chunks &&
               a.stats() > 100.0 * static_cast<double>(ramp_chunks - 1) / static_cast<double>(chunks),
           line);

    // The constant over it, by the chunk-aligned loop: compress() folds every chunk back,
    // the one the half ends in too (its elements past the half were never written).
    a.parallel_for_chunks(0, h, Const<T>{T{}});
    std::fill(mirror.begin(), mirror.begin() + h, T{});
    const auto folded = a.compress();
    std::snprintf(line, sizeof line, "the constant back over it, compress(): %llu folded, %llu hold a buffer",
                  folded, a.written_chunk_count());
    expect(wrong(a, mirror) == 0 && folded == ramp_chunks && a.written_chunk_count() == 0, line);

    // 3 over the whole array, by the chunk-aligned loop and then by a write kernel through
    // the writer (the odd elements again): zero_any folds to it; zero_or_value once it is the
    // second value; zero never.
    if constexpr (V == sd::Variant::zero_or_value)
    {
        a.set_second_value(T{3});
    }
    a.parallel_for_chunks(0, n, Const<T>{T{3}});
    write_odd<WriterT, T><<<256, 256>>>(a.writer(), 0, n, T{3});
    a.finish();
    std::fill(mirror.begin(), mirror.end(), T{3});
    (void)a.compress();
    const auto expected = V == sd::Variant::zero ? static_cast<unsigned long long>(chunks) : 0ULL;
    std::snprintf(line, sizeof line, "3 everywhere (loop, then DeviceWriter), compress(): %llu/%lld hold a buffer",
                  a.written_chunk_count(), static_cast<long long>(chunks));
    expect(wrong(a, mirror) == 0 && a.written_chunk_count() == expected, line);

    // A ramp again on a third, then the image built again: the same elements, the same buffers.
    a.parallel_for_chunks(n / 3, 2 * n / 3, Ramp<T>{});
    for (std::int64_t i = n / 3; i < 2 * n / 3; ++i)
    {
        mirror[static_cast<std::size_t>(i)] = Ramp<T>{}(i);
    }
    const auto image = a.image();
    Arr        b(image.state);
    std::snprintf(line, sizeof line, "DeviceArray(image().state): %llu chunks hold a buffer, as the source (%llu)",
                  b.written_chunk_count(), a.written_chunk_count());
    expect(wrong(b, mirror) == 0 && b.written_chunk_count() == a.written_chunk_count(), line);

    // A read kernel through the view.
    T* d_out = sd::allocate<T>(static_cast<std::size_t>(n));
    read_all<ViewT, T><<<256, 256>>>(b.view(), d_out);
    sd::synchronize();
    std::vector<T> out(static_cast<std::size_t>(n));
    sd::copy(out.data(), d_out, out.size());
    sd::release(d_out);
    long bad = 0;
    for (std::size_t i = 0; i < out.size(); ++i)
    {
        bad += out[i] == mirror[i] ? 0 : 1;
    }
    expect(bad == 0, "a read kernel through DeviceView");
}

template <typename T, unsigned int S>
void run_all()
{
    run<sa::zero::DeviceArray<T, S>, sa::zero::DeviceView<T, S>, sa::zero::DeviceWriter<T, S>>("zero");
    run<sa::zero_any::DeviceArray<T, S>, sa::zero_any::DeviceView<T, S>, sa::zero_any::DeviceWriter<T, S>>("zero_any");
    run<sa::zero_or_value::DeviceArray<T, S>, sa::zero_or_value::DeviceView<T, S>,
        sa::zero_or_value::DeviceWriter<T, S>>("zero_or_value");
}

} // namespace

int main()
{
    std::printf("device_api_validation: zero / zero_any / zero_or_value::DeviceArray\n");
    try
    {
        static_assert(sa::zero_any::DeviceArray<double>::chunk_size() == 1024, "S defaults to 1024");
        run_all<double, 256>();
        run_all<double, 1024>();
        run_all<int, 256>();
        run_all<int, 1024>();
    }
    catch (const std::exception& e)
    {
        std::printf("error: %s\n", e.what());
        return 2;
    }
    std::printf("%s: %d failure(s)\n", g_failures == 0 ? "PASSED" : "FAILED", g_failures);
    return g_failures == 0 ? 0 : 1;
}
