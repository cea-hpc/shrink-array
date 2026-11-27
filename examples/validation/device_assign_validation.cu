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
 * @file device_assign_validation.cu
 * @brief Writer::assign and Writer::load (chunk/device/block_write.h, docs/gpu_assign.md)
 *        against a host mirror: scattered writes from many threads, grid-stride first
 *        writes, scattered writes into materialised and compressed chunks mixed (the lanes
 *        of a warp on both paths of assign()), `+=` by the thread that owns the element, one
 *        thread writing an element
 *        twice and reading it back in between, two `assign()` on divergent paths of one warp,
 *        two kernels before one finish(), assign() after a chunk-aligned write, and small
 *        arenas, where the append-buffer and its replay carry the writes.
 *
 * @details Built with -DCHUNK_DEVICE_CORE_CHECK=ON, for CUDA or HIP
 *          (-DCHUNK_DEVICE_LANGUAGE). Each round checks every element, the materialised
 *          chunks against a host model (a chunk materialises when a value other than its
 *          constant is written into it; compress() after some rounds folds the uniform ones),
 *          and that the arena holds exactly one buffer per materialised chunk after finish()
 *          (the spare and lost buffers given back); an arena of one buffer per chunk must not
 *          grow (the claim comes before the buffer: no buffer taken for nothing). A thread that does not read back what it
 *          wrote counts as a failure too.
 */

#include <algorithm> // std::all_of, std::count, std::fill, std::min, std::shuffle
#include <cstdint>   // std::int64_t
#include <cstdio>    // std::printf, std::snprintf
#include <cstring>   // std::memcmp
#include <exception> // std::exception
#include <numeric>   // std::iota
#include <random>    // std::mt19937_64
#include <vector>    // std::vector

#include "chunk/device/array.h"   // device::Array, Writer
#include "chunk/device/backend.h" // allocate, copy, release, synchronize, zero

namespace sd = shrink_array::device;

namespace
{

int g_failures = 0;

void expect(bool ok, const char* what)
{
    std::printf("  %-64s %s\n", what, ok ? "ok" : "FAILED");
    g_failures += ok ? 0 : 1;
}

__host__ __device__ bool same_bits(double a, double b)
{
    return __builtin_bit_cast(unsigned long long, a) == __builtin_bit_cast(unsigned long long, b);
}

/// @brief Entry k: a(index[k]) = value[k], grid-stride.
template <unsigned int S, sd::Variant V>
__global__ void scatter(sd::Writer<double, S, V> w, const std::int64_t* index, const double* value, std::int64_t m)
{
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t k = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x; k < m; k += stride)
    {
        w.assign(index[k], value[k]);
    }
}

/// @brief Entry k: a(i) = v1, read back, a(i) = v2 (the same thread twice), the even lanes and
///        the odd ones on two divergent paths.
template <unsigned int S, sd::Variant V>
__global__ void twice(sd::Writer<double, S, V> w, const std::int64_t* index, const double* v1, const double* v2,
                      std::int64_t m, unsigned long long* bad)
{
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t k = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x; k < m; k += stride)
    {
        const std::int64_t i = index[k];
        if ((threadIdx.x & 1U) == 0)
        {
            w.assign(i, v1[k]);
            if (!same_bits(w.load(i), v1[k]))
            {
                sd::fetch_add(bad, 1ULL);
            }
            w.assign(i, v2[k]);
        }
        else
        {
            w.assign(i, v1[k]);
            if (!same_bits(w.load(i), v1[k]))
            {
                sd::fetch_add(bad, 1ULL);
            }
            w.assign(i, v2[k]);
        }
    }
}

/// @brief a(i) = 3.25 + i % 7, then a(i) = c, the chunk's constant, by the same thread: in a
///        compressed chunk of an exhausted arena the first write is appended, and the second
///        must win (kokkos_shrink_bench on the MI300A: `a2 = .5; a2 = 1.0` on chunks of 1.0).
template <unsigned int S, sd::Variant V>
__global__ void then_constant(sd::Writer<double, S, V> w, std::int64_t n, double c)
{
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x; i < n; i += stride)
    {
        w.assign(i, 3.25 + static_cast<double>(i % 7));
        w.assign(i, c);
    }
}

/// @brief a(i) += 1.5 for every i, then += 0.25: the owner thread reads its own writes.
template <unsigned int S, sd::Variant V>
__global__ void add_twice(sd::Writer<double, S, V> w, std::int64_t n)
{
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x; i < n; i += stride)
    {
        w.assign(i, w.load(i) + 1.5);
        w.assign(i, w.load(i) + 0.25);
    }
}

/// @brief f(i) = 0: the chunk-aligned write that makes every chunk foldable.
struct Zero
{
    __device__ double operator()(std::int64_t /*i*/) const noexcept
    {
        return 0.0;
    }
};

/// @brief f(i) = 7 + i: a chunk-aligned write before the assign() rounds.
struct Ramp
{
    __device__ double operator()(std::int64_t i) const noexcept
    {
        return 7.0 + static_cast<double>(i);
    }
};

/// @brief The host side of one array: the mirror and the model of its chunks.
template <unsigned int S, sd::Variant V>
struct Model
{
    std::vector<double> mirror;
    std::vector<char>   mat;
    std::vector<double> cst;

    Model(std::int64_t n, double constant)
        : mirror(static_cast<std::size_t>(n), constant)
        , mat(static_cast<std::size_t>((n + S - 1) / S), 0)
        , cst(mat.size(), constant)
    {
    }

    /// @brief The chunk-aligned write of @p v over the whole array: zero_any, a compressed
    ///        chunk takes v as its constant, no buffer (Writer::fold_whole).
    void write_all_aligned(double v)
    {
        for (std::size_t c = 0; c < mat.size(); ++c)
        {
            const std::size_t lo = c * S;
            const std::size_t hi = std::min<std::size_t>(lo + S, mirror.size());
            if (V == sd::Variant::zero_any && mat[c] == 0)
            {
                std::fill(mirror.begin() + static_cast<long>(lo), mirror.begin() + static_cast<long>(hi), v);
                cst[c] = v;
                continue;
            }
            for (std::size_t i = lo; i < hi; ++i)
            {
                write(static_cast<std::int64_t>(i), v);
            }
        }
    }

    void write(std::int64_t i, double v)
    {
        const auto c                        = static_cast<std::size_t>(i / S);
        mirror[static_cast<std::size_t>(i)] = v;
        if (!same_bits(v, cst[c]))
        {
            mat[c] = 1;
        }
    }

    /// @brief compress(): the folds it must do (zero: to +0.0 only).
    unsigned long long fold()
    {
        unsigned long long folds = 0;
        for (std::size_t c = 0; c < mat.size(); ++c)
        {
            const std::size_t lo = c * S;
            const std::size_t hi = std::min<std::size_t>(lo + S, mirror.size());
            const double head = V == sd::Variant::zero ? 0.0 : mirror[lo];
            if (mat[c] != 0 && std::all_of(mirror.begin() + static_cast<long>(lo), mirror.begin() + static_cast<long>(hi),
                                           [&](double x) { return same_bits(x, head); }))
            {
                mat[c] = 0;
                cst[c] = head;
                ++folds;
            }
        }
        return folds;
    }
};

template <unsigned int S, sd::Variant V>
void check(sd::Array<double, S, V>& a, const Model<S, V>& model, const char* what)
{
    const std::vector<double> got   = a.to_host();
    long                      wrong = 0;
    for (std::size_t i = 0; i < got.size(); ++i)
    {
        wrong += same_bits(got[i], model.mirror[i]) ? 0 : 1;
    }
    const auto expected = static_cast<unsigned long long>(std::count(model.mat.begin(), model.mat.end(), 1));
    const auto mat      = a.materialised();
    const auto in_use   = a.arena().in_use();
    char       line[200];
    std::snprintf(line, sizeof line, "%s: %ld wrong, %llu/%llu materialised, %llu in use", what, wrong, mat, expected,
                  in_use);
    expect(wrong == 0 && mat == expected && in_use == mat, line);
}

template <unsigned int S, sd::Variant V = sd::Variant::zero_any>
void run(unsigned seed, std::int64_t n, unsigned long long arena_buffers, double constant, const char* label)
{
    std::printf("[%s%s] S=%u n=%lld arena=%llu constant=%g\n", V == sd::Variant::zero ? "zero, " : "", label, S,
                static_cast<long long>(n), arena_buffers, constant);
    // An append-buffer of 2n entries: the small arenas send most writes there, step 0 two per
    // element (a value, then the constant after it).
    sd::Array<double, S, V> a(n, arena_buffers, constant, 2 * static_cast<unsigned long long>(n));
    Model<S, V>             model(n, constant);
    std::mt19937_64      rng(seed);
    const double         palette[] = {0.0, -0.0, 1.5, 2.5};
    const auto           nn        = static_cast<std::size_t>(n);
    const auto           pick      = [&]
    {
        const auto r = rng() % 8;
        return r < 4 ? palette[r] : static_cast<double>(rng() % 1000) / 7.0;
    };
    std::int64_t*       d_index = sd::allocate<std::int64_t>(nn);
    double*             d_v1    = sd::allocate<double>(nn);
    double*             d_v2    = sd::allocate<double>(nn);
    unsigned long long* d_bad   = sd::allocate<unsigned long long>(1);

    // 0. On the array as made (every chunk compressed to the constant): a value, then the
    //    constant again, by each thread, every element. The small arenas append the first
    //    writes; the constant must be what every element reads after finish().
    {
        then_constant<S, V><<<512, 256>>>(a.writer(), n, constant);
        a.finish();
        for (std::int64_t i = 0; i < n; ++i)
        {
            model.write(i, 3.25 + static_cast<double>(i % 7));
            model.write(i, constant);
        }
        // Every element reads the constant: the chunks that got a buffer fold back.
        (void)a.compress();
        std::fill(model.mat.begin(), model.mat.end(), 0);
        check(a, model, "a value then the constant, by one thread, then compress()");
    }

    // 1. Scattered writes: a third of the elements, shuffled, values with repeats of the
    //    constant; in two kernels before one finish().
    {
        std::vector<std::int64_t> index(nn);
        std::iota(index.begin(), index.end(), std::int64_t{0});
        std::shuffle(index.begin(), index.end(), rng);
        index.resize(nn / 3);
        std::vector<double> value(index.size());
        for (std::size_t k = 0; k < index.size(); ++k)
        {
            value[k] = pick();
            model.write(index[k], value[k]);
        }
        sd::copy(d_index, index.data(), index.size());
        sd::copy(d_v1, value.data(), value.size());
        const auto half = static_cast<std::int64_t>(index.size() / 2);
        scatter<S, V><<<512, 256>>>(a.writer(), d_index, d_v1, half);
        scatter<S, V><<<512, 256>>>(a.writer(), d_index + half, d_v1 + half, static_cast<std::int64_t>(index.size()) - half);
        a.finish();
        check(a, model, "scattered writes, two kernels, one finish()");
    }

    // 1b. Scattered writes into an array of materialised and compressed chunks mixed: unaligned
    //     runs of consecutive indices (a constant per run, so that compress() folds some
    //     chunks), then a quarter of the elements shuffled. The lanes of a warp then take both
    //     paths of assign(): those on materialised chunks write and go on to their next
    //     grid-stride iteration while the others are in the claim. With the claim grouped per
    //     warp (__activemask, ballot, shuffles) this kernel hung on every run on NVIDIA
    //     (RTX A1000, n = 2^20, S = 1024): two subsets of a warp, with masks naming each other,
    //     waited for each other in a shuffle and a __syncwarp (Writer::materialise_grouped).
    {
        std::vector<std::int64_t> index;
        std::vector<double>       value;
        std::int64_t              at = static_cast<std::int64_t>(rng() % S);
        while (at < n)
        {
            const std::int64_t len = 1 + static_cast<std::int64_t>(rng() % (3 * S));
            const std::int64_t hi  = std::min(at + len, n);
            if (rng() % 3 != 0)
            {
                const double c = pick();
                for (std::int64_t i = at; i < hi; ++i)
                {
                    index.push_back(i);
                    value.push_back(c);
                }
            }
            at = hi + static_cast<std::int64_t>(rng() % (S / 2));
        }
        for (std::size_t k = 0; k < index.size(); ++k)
        {
            model.write(index[k], value[k]);
        }
        sd::copy(d_index, index.data(), index.size());
        sd::copy(d_v1, value.data(), value.size());
        scatter<S, V><<<512, 256>>>(a.writer(), d_index, d_v1, static_cast<std::int64_t>(index.size()));
        a.finish();
        (void)a.compress();
        (void)model.fold();
        check(a, model, "unaligned runs by assign(), then compress()");

        std::vector<std::int64_t> shuffled(nn);
        std::iota(shuffled.begin(), shuffled.end(), std::int64_t{0});
        std::shuffle(shuffled.begin(), shuffled.end(), rng);
        shuffled.resize(nn / 4);
        std::vector<double> v(shuffled.size());
        for (std::size_t k = 0; k < shuffled.size(); ++k)
        {
            v[k] = pick();
            model.write(shuffled[k], v[k]);
        }
        sd::copy(d_index, shuffled.data(), shuffled.size());
        sd::copy(d_v1, v.data(), v.size());
        scatter<S, V><<<512, 256>>>(a.writer(), d_index, d_v1, static_cast<std::int64_t>(shuffled.size()));
        a.finish();
        check(a, model, "scattered writes into materialised and compressed chunks mixed");
    }

    // 2. The same thread twice on its element, reading back in between, on two divergent
    //    paths of each warp; every element of a random half.
    {
        std::vector<std::int64_t> index(nn);
        std::iota(index.begin(), index.end(), std::int64_t{0});
        std::shuffle(index.begin(), index.end(), rng);
        index.resize(nn / 2);
        std::vector<double> v1(index.size()), v2(index.size());
        for (std::size_t k = 0; k < index.size(); ++k)
        {
            v1[k] = pick();
            v2[k] = pick();
            model.write(index[k], v1[k]);
            model.write(index[k], v2[k]);
        }
        sd::copy(d_index, index.data(), index.size());
        sd::copy(d_v1, v1.data(), v1.size());
        sd::copy(d_v2, v2.data(), v2.size());
        sd::zero(d_bad, 1);
        twice<S, V><<<512, 256>>>(a.writer(), d_index, d_v1, d_v2, static_cast<std::int64_t>(index.size()), d_bad);
        a.finish();
        unsigned long long bad = 0;
        sd::copy(&bad, d_bad, 1);
        char line[160];
        std::snprintf(line, sizeof line, "the same thread twice: %llu reads that missed its own write", bad);
        expect(bad == 0, line);
        check(a, model, "the same thread twice, divergent paths");
    }

    // 3. compress(), then a grid-stride first write of the whole array (every chunk compressed
    //    first: the constant everywhere, folded).
    {
        a.parallel_for_chunks(0, n, Zero{});
        model.write_all_aligned(0.0);
        const unsigned long long expected = model.fold();
        const unsigned long long folds    = a.compress();
        char                     line[160];
        std::snprintf(line, sizeof line, "compress(): %llu/%llu folded", folds, expected);
        expect(folds == expected, line);
        std::vector<std::int64_t> index(nn);
        std::vector<double>       value(nn);
        for (std::size_t i = 0; i < nn; ++i)
        {
            index[i] = static_cast<std::int64_t>(i);
            value[i] = 1.0 + static_cast<double>(i % 977);
            model.write(index[i], value[i]);
        }
        sd::copy(d_index, index.data(), nn);
        sd::copy(d_v1, value.data(), nn);
        scatter<S, V><<<512, 256>>>(a.writer(), d_index, d_v1, n);
        a.finish();
        check(a, model, "grid-stride first write over compressed chunks");
    }

    // 4. A chunk-aligned write on a range, then += twice by the owner thread everywhere.
    {
        a.parallel_for_chunks(n / 4, n / 2, Ramp{});
        for (std::int64_t i = n / 4; i < n / 2; ++i)
        {
            model.write(i, 7.0 + static_cast<double>(i));
        }
        (void)model.fold();
        (void)a.compress();
        add_twice<S, V><<<512, 256>>>(a.writer(), n);
        a.finish();
        for (std::int64_t i = 0; i < n; ++i)
        {
            const double v = model.mirror[static_cast<std::size_t>(i)];
            model.write(i, v + 1.5);
            model.write(i, v + 1.5 + 0.25);
        }
        check(a, model, "chunk-aligned write, compress(), then += twice");
    }
    const sd::WriteCounters wc = a.write_counters();
    std::printf("  arena %llu buffers in %d arena(s); %llu claim waits, %llu given up, %llu lost CAS\n",
                a.arena().buffers(), a.arena().arenas(), wc.claim_wait, wc.claim_spin, wc.cas_lost);
    if (arena_buffers >= static_cast<unsigned long long>(a.n_chunks()))
    {
        // One buffer per chunk at most: the claim comes before the buffer (docs/gpu_assign.md 3.3).
        expect(a.arena().arenas() == 1, "a roomy arena did not grow");
    }
    sd::release(d_index);
    sd::release(d_v1);
    sd::release(d_v2);
    sd::release(d_bad);
}

} // namespace

int main()
{
    std::printf("device_assign_validation: Writer::assign / load against a host mirror\n");
    try
    {
        run<256>(21, 1L << 20, 1L << 12, 0.0, "roomy arena");
        run<1024>(22, (1L << 20) + 300, (1L << 10) + 1, 0.0, "roomy arena, tail chunk");
        // Small: once the arena has run out, load() searches the append-buffer.
        run<256>(23, 1L << 16, 8, 0.0, "small arena: appends and their replay");
        run<1024>(24, (1L << 16) + 77, 2, 2.5, "small arena, tail chunk, constant 2.5");
        // The zero variant: -0.0 among the values written materialises, folds to +0.0 only.
        constexpr auto zero = sd::Variant::zero;
        run<256, zero>(31, 1L << 20, 1L << 12, 0.0, "roomy arena");
        run<1024, zero>(32, (1L << 20) + 300, (1L << 10) + 1, 0.0, "roomy arena, tail chunk");
        run<256, zero>(33, 1L << 16, 8, 0.0, "small arena: appends and their replay");
        run<1024, zero>(34, (1L << 16) + 77, 2, 0.0, "small arena, tail chunk");
    }
    catch (const std::exception& e)
    {
        std::printf("error: %s\n", e.what());
        return 2;
    }
    std::printf("%s: %d failure(s)\n", g_failures == 0 ? "PASSED" : "FAILED", g_failures);
    return g_failures == 0 ? 0 : 1;
}
