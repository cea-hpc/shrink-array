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
 * @file access_cost_pattern.cpp
 * @brief The cost of an element access on pattern::Array, against a std::vector
 *        and a zero_any::Array in the SAME binary, and the memory of a mesh
 *        connectivity in the three (docs/algo_pattern.md, section 9).
 * @details One binary, access_cost_pattern_<size>, linked with the zero_any
 *          library (pattern::Array is a member of the zero_any family). The three
 *          containers hold `long` elements; each case times them in turn, the
 *          order rotated from one repetition to the next (performance_measurement.md:
 *          compare within one binary and one time window). The data:
 *            varied        a hash of the index: every chunk materialised in both
 *                          arrays (no constant, no motif);
 *            connectivity  the 8 node numbers of each cell of a structured
 *                          hexahedral mesh, cell by cell, x fastest: consecutive
 *                          cells are the same 8 numbers plus one, so a chunk of
 *                          whole cells is one motif plus an offset (pattern::Array
 *                          compresses it; zero_any::Array cannot).
 *          The cases, each its own parallel region with a static `omp for`:
 *            read-varied       sum of every element of the varied data: what the
 *                              motif range test costs a materialised read;
 *            read-conn         sum of every element of the connectivity: the
 *                              motif read (offset slot and pointer re-check, F5);
 *            read-conn-random  sum of n elements of the connectivity at random
 *                              indices;
 *            write-conn        the connectivity written into an empty container,
 *                              the end of the phase included (pattern::Array's
 *                              pass classifies and compresses the chunks).
 *          Printed: per case and container, the median and minimum time per
 *          element (ns) and the ratio of the medians to the vector's; then the
 *          memory of the connectivity in each container: the vector's bytes, and
 *          for the arrays the materialised and compressed chunks, the motifs,
 *          and an estimate (buffers + chunk objects + second-value slots +
 *          motifs). --csv prints one line per case, container and repetition.
 *          Sums are integers: each case's must be the same in the three (exit 1
 *          otherwise); --verify compares every element after the writes.
 *          Usage: access_cost_pattern [--n n_elems] [--reps R] [--csv] [--verify]
 *          (n_elems is rounded down to whole cells of a mesh 256 cells wide).
 */

#include <algorithm> // std::sort, std::min_element
#include <chrono>    // std::chrono::steady_clock
#include <cstdint>   // std::uint64_t
#include <cstdio>    // std::fprintf, stderr
#include <string>    // std::string
#include <utility>   // std::as_const
#include <vector>    // std::vector

#include "chunk/pattern_chunk.h"      // pattern::Chunk
#include "chunk/chunk_traits.h"       // the IntType = int default of the arrays
#include "chunk/zero_any_array.h"     // zero_any::Array
#include "chunk/pattern_array.h"      // pattern::Array, pattern::MotifRegion
#include "chunk/param_chunk.h"        // ParamChunk::init_indexes
#include "chunk/config.h"             // chunk_config::use_zero_any, build_config::use_openmp / has_not_ompt
#include "chunk/PMP_CHUNK.h"          // CHUNK_PMP_MAX_THREADS
#include "chunk/manage_ompt.h"        // begin_parallel_phase, finalize_parallel_phase
#include "chunk/internal/env_parse.h" // arg_number (checked, not atoi)
#include "chunk/internal/print.h"     // shrink_array::println

using namespace shrink_array;

static_assert(chunk_config::use_zero_any, "access_cost_pattern: link chunk_zero_any_*");

using T    = long;
using ZArr = zero_any::Array<T>;
using PArr = pattern::Array<T>;

namespace
{
// ── Phase brackets: no-ops under OMPT (the callbacks do it) ─────────────────
void begin_phase()
{
    if constexpr (build_config::has_not_ompt)
    {
        shrink_array::begin_parallel_phase();
    }
}

template <typename Arr>
void end_phase(Arr& arr)
{
    if constexpr (build_config::has_not_ompt)
    {
        shrink_array::finalize_parallel_phase();
    }
    // Serial builds have no between-region pass: run it (as access_cost).
    if constexpr (!build_config::use_openmp)
    {
        arr.make_block_compression();
    }
    else
    {
        (void)arr;
    }
}

/// A hash of the index (splitmix64), below 2^31: nothing to compress.
T varied(long index)
{
    std::uint64_t state = (static_cast<std::uint64_t>(index) + 1) * 0x9E3779B97F4A7C15ULL;
    state ^= state >> 30;
    state *= 0xBF58476D1CE4E5B9ULL;
    state ^= state >> 27;
    return static_cast<T>(state >> 33U);
}

/// The structured hexahedral mesh: nx x ny x nz cells, nodes numbered x fastest.
struct Mesh
{
    long nx = 256; ///< Cells along x: a multiple of 32, so a chunk of 256 elements (32 cells) never spans two rows.
    long ny = 1;
    long nz = 1;

    [[nodiscard]] long cells() const
    {
        return nx * ny * nz;
    }
    /// Element @p index of the connectivity: node @p index % 8 of cell @p index / 8.
    [[nodiscard]] T operator()(long index) const
    {
        const long cell = index / 8;
        const long k    = index % 8;
        const long i    = cell % nx;
        const long j    = (cell / nx) % ny;
        const long l    = cell / (nx * ny);
        const long di   = k & 1;
        const long dj   = (k >> 1) & 1;
        const long dl   = (k >> 2) & 1;
        return (i + di) + (nx + 1) * ((j + dj) + (ny + 1) * (l + dl));
    }
};

/// A fixed pseudo-random index sequence (splitmix64), every index in [0, n).
std::vector<int> random_indices(long count)
{
    std::vector<int> idx(static_cast<std::size_t>(count));
    const int        ni = static_cast<int>(count);
#pragma omp parallel for schedule(static)
    for (int k = 0; k < ni; ++k)
    {
        std::uint64_t state = (static_cast<std::uint64_t>(k) + 1) * 0xD1B54A32D192ED03ULL;
        state ^= state >> 31;
        state *= 0x94D049BB133111EBULL;
        state ^= state >> 29;
        idx[static_cast<std::size_t>(k)] = static_cast<int>(state % static_cast<std::uint64_t>(count));
    }
    return idx;
}

using Clk = std::chrono::steady_clock;

double secs_since(Clk::time_point t0)
{
    return std::chrono::duration<double>(Clk::now() - t0).count();
}

struct Sample
{
    double    seconds = 0.;
    long long sum     = 0;
};

// ── The cases, on a std::vector ──────────────────────────────────────────────
Sample vec_read(const std::vector<T>& vec)
{
    const auto t0    = Clk::now();
    const int  count = static_cast<int>(vec.size());
    long long  sum   = 0;
#pragma omp parallel for schedule(static) reduction(+ : sum)
    for (int i = 0; i < count; ++i)
    {
        sum += vec[static_cast<std::size_t>(i)];
    }
    return {secs_since(t0), sum};
}

Sample vec_read_random(const std::vector<T>& vec, const std::vector<int>& idx)
{
    const auto t0    = Clk::now();
    const int  count = static_cast<int>(idx.size());
    long long  sum   = 0;
#pragma omp parallel for schedule(static) reduction(+ : sum)
    for (int k = 0; k < count; ++k)
    {
        sum += vec[static_cast<std::size_t>(idx[static_cast<std::size_t>(k)])];
    }
    return {secs_since(t0), sum};
}

template <typename F>
Sample vec_write(std::vector<T>& vec, F value)
{
    const auto t0    = Clk::now();
    const int  count = static_cast<int>(vec.size());
#pragma omp parallel for schedule(static)
    for (int i = 0; i < count; ++i)
    {
        vec[static_cast<std::size_t>(i)] = value(i);
    }
    return {secs_since(t0), 0};
}

// ── The same cases, on an array ──────────────────────────────────────────────
template <typename Arr>
Sample arr_read(Arr& arr, int count)
{
    const auto  t0  = Clk::now();
    long long   sum = 0;
    const auto& ca  = std::as_const(arr);
#pragma omp parallel
    {
        begin_phase();
#pragma omp for schedule(static) reduction(+ : sum)
        for (int i = 0; i < count; ++i)
        {
            sum += ca[i];
        }
    }
    end_phase(arr);
    return {secs_since(t0), sum};
}

template <typename Arr>
Sample arr_read_random(Arr& arr, const std::vector<int>& idx)
{
    const auto  t0    = Clk::now();
    const int   count = static_cast<int>(idx.size());
    long long   sum   = 0;
    const auto& ca    = std::as_const(arr);
#pragma omp parallel
    {
        begin_phase();
#pragma omp for schedule(static) reduction(+ : sum)
        for (int k = 0; k < count; ++k)
        {
            sum += ca[idx[static_cast<std::size_t>(k)]];
        }
    }
    end_phase(arr);
    return {secs_since(t0), sum};
}

template <typename Arr, typename F>
Sample arr_write(Arr& arr, int count, F value)
{
    const auto t0 = Clk::now();
#pragma omp parallel
    {
        begin_phase();
#pragma omp for schedule(static)
        for (int i = 0; i < count; ++i)
        {
            arr[i] = value(i);
        }
    }
    end_phase(arr);
    return {secs_since(t0), 0};
}

template <typename Arr>
long long arr_sum(const Arr& arr, int count)
{
    long long sum = 0;
    for (int i = 0; i < count; ++i)
    {
        sum += arr[i];
    }
    return sum;
}

long long vec_sum(const std::vector<T>& vec)
{
    long long sum = 0;
    for (const T value : vec)
    {
        sum += value;
    }
    return sum;
}

template <typename Arr>
long verify_elements(const char* what, const char* name, const Arr& arr, const std::vector<T>& vec)
{
    long bad = 0;
    for (std::size_t i = 0; i < vec.size(); ++i)
    {
        if (arr[static_cast<int>(i)] != vec[i])
        {
            if (bad < 3)
            {
                std::fprintf(stderr, "access_cost_pattern: %s, %s: element %zu is %ld, the vector has %ld\n", what,
                             name, i, static_cast<long>(arr[static_cast<int>(i)]), static_cast<long>(vec[i]));
            }
            ++bad;
        }
    }
    return bad;
}

/// Chunks of @p arr: materialised, compressed to a constant, compressed to a motif,
/// and the distinct motifs they point at.
struct Census
{
    long mat    = 0;
    long cst    = 0;
    long motif  = 0;
    long motifs = 0;
};

template <typename Arr>
Census census(const Arr& arr, long n_chunks)
{
    Census                c;
    std::vector<const T*> seen;
    for (long k = 0; k < n_chunks; ++k)
    {
        const auto& ch = arr.get_chunk_list()[static_cast<std::size_t>(k)];
        if constexpr (has_motif_state_v<typename Arr::ChunkType>)
        {
            if (Arr::ChunkType::is_motif_ptr(ch.values_load()))
            {
                ++c.motif;
                seen.push_back(ch.values_load());
                continue;
            }
        }
        if (ch.written())
        {
            ++c.mat;
        }
        else
        {
            ++c.cst;
        }
    }
    std::sort(seen.begin(), seen.end());
    c.motifs = static_cast<long>(std::unique(seen.begin(), seen.end()) - seen.begin());
    return c;
}

double median(std::vector<double> samples)
{
    std::sort(samples.begin(), samples.end());
    const std::size_t mid = samples.size() / 2;
    return samples.size() % 2 ? samples[mid] : 0.5 * (samples[mid - 1] + samples[mid]);
}

double minimum(const std::vector<double>& samples)
{
    return *std::min_element(samples.begin(), samples.end());
}

constexpr const char* case_names[]      = {"read-varied", "read-conn", "read-conn-random", "write-conn"};
constexpr int         n_cases           = static_cast<int>(sizeof(case_names) / sizeof(case_names[0]));
constexpr const char* container_names[] = {"vector", "zero_any", "pattern"};
constexpr int         n_containers      = 3;
} // namespace

int main(int argc, char** argv)
{
    ParamChunk::init_indexes(true);

    long n_elems = 1L << 26;
    int  reps    = 5;
    bool csv     = false;
    bool verify  = false;
    for (int a = 1; a < argc; ++a)
    {
        const std::string opt = argv[a];
        if (opt == "--n" && a + 1 < argc)
        {
            n_elems = shrink_array::arg_number<long>(argv[++a], n_elems, "--n");
        }
        else if (opt == "--reps" && a + 1 < argc)
        {
            reps = shrink_array::arg_number<int>(argv[++a], reps, "--reps");
        }
        else if (opt == "--csv")
        {
            csv = true;
        }
        else if (opt == "--verify")
        {
            verify = true;
        }
        else
        {
            std::fprintf(stderr, "usage: %s [--n N] [--reps R] [--csv] [--verify]\n", argv[0]);
            return 2;
        }
    }
    // Whole cells, whole rows of 256 cells: n = 8 * 256 * rows.
    Mesh mesh;
    mesh.ny = n_elems / (8 * mesh.nx);
    if (reps < 1 || mesh.ny < 1 || n_elems > 0x7fffffffL)
    {
        std::fprintf(stderr, "access_cost_pattern: need reps >= 1, 2048 <= n < 2^31\n");
        return 2;
    }
    n_elems         = 8 * mesh.cells();
    const int count = static_cast<int>(n_elems);

    const long chunk_sz = static_cast<long>(PArr(0).chunk_size());
    const long n_chunks = (n_elems + chunk_sz - 1) / chunk_sz;
    if (csv)
    {
        println("chunk_size,threads,n,rep,case,container,seconds,ns_per_elem");
    }
    else
    {
        println("=== access_cost_pattern  n={} ({} x {} cells, 8 nodes)  chunk_size={}  threads={}  reps={} ===",
                n_elems, mesh.nx, mesh.ny, chunk_sz, CHUNK_PMP_MAX_THREADS, reps);
    }

    const std::vector<int> idx = random_indices(n_elems);
    std::vector<T>         vvar(static_cast<std::size_t>(n_elems));
    std::vector<T>         vconn(static_cast<std::size_t>(n_elems));
    vec_write(vvar, varied);
    vec_write(vconn, mesh);

    std::vector<double> times[n_cases][n_containers];
    long                sum_mismatch = 0;
    long                bad_elements = 0;
    Census              zc;
    Census              pc;

    const auto check_sums = [&](int c, const long long (&sums)[n_containers])
    {
        for (int k = 1; k < n_containers; ++k)
        {
            if (sums[k] != sums[0])
            {
                std::fprintf(stderr, "access_cost_pattern: %s: %s sums to %lld, the vector to %lld\n", case_names[c],
                             container_names[k], sums[k], sums[0]);
                ++sum_mismatch;
            }
        }
    };
    const auto note = [&](int c, int k, int rep, double seconds)
    {
        times[c][k].push_back(seconds);
        if (csv)
        {
            println("{},{},{},{},{},{},{:.9f},{:.4f}", chunk_sz, CHUNK_PMP_MAX_THREADS, n_elems, rep, case_names[c],
                    container_names[k], seconds, seconds * 1e9 / static_cast<double>(n_elems));
        }
    };

    // The arrays the reads use, built once, outside the timings: built again with
    // the same values, the varied chunks of a new array would be second sightings
    // of the old one's -- motifs admitted for data no program repeats.
    ZArr zvar(count);
    PArr pvar(count);
    ZArr zconn(count);
    PArr pconn(count);
    arr_write(zvar, count, varied);
    arr_write(pvar, count, varied);
    arr_write(zconn, count, mesh);
    arr_write(pconn, count, mesh);
    zc = census(zconn, n_chunks);
    pc = census(pconn, n_chunks);

    for (int rep = 0; rep < reps; ++rep)
    {
        // The three runs of one case, the order rotated with the repetition.
        const auto run3 = [&](int c, auto&& run_vec, auto&& run_z, auto&& run_p)
        {
            Sample s[n_containers];
            for (int step = 0; step < n_containers; ++step)
            {
                const int k = (rep + step) % n_containers;
                s[k]        = k == 0 ? run_vec() : (k == 1 ? run_z() : run_p());
            }
            for (int k = 0; k < n_containers; ++k)
            {
                note(c, k, rep, s[k].seconds);
            }
            const long long sums[n_containers] = {s[0].sum, s[1].sum, s[2].sum};
            return std::vector<long long>(sums, sums + n_containers);
        };
        const auto check = [&](int c, const std::vector<long long>& v)
        {
            const long long sums[n_containers] = {v[0], v[1], v[2]};
            check_sums(c, sums);
        };

        check(0, run3(
                     0, [&] { return vec_read(vvar); }, [&] { return arr_read(zvar, count); },
                     [&] { return arr_read(pvar, count); }));
        check(1, run3(
                     1, [&] { return vec_read(vconn); }, [&] { return arr_read(zconn, count); },
                     [&] { return arr_read(pconn, count); }));
        check(2, run3(
                     2, [&] { return vec_read_random(vconn, idx); }, [&] { return arr_read_random(zconn, idx); },
                     [&] { return arr_read_random(pconn, idx); }));
        {
            std::vector<T> fv(static_cast<std::size_t>(n_elems), 0);
            ZArr           zf(count);
            PArr           pf(count);
            run3(
                3, [&] { return vec_write(fv, mesh); }, [&] { return arr_write(zf, count, mesh); },
                [&] { return arr_write(pf, count, mesh); });
            const long long sums[n_containers] = {vec_sum(fv), arr_sum(zf, count), arr_sum(pf, count)};
            check_sums(3, sums);
            if (verify)
            {
                bad_elements += verify_elements("write-conn", "zero_any", zf, fv);
                bad_elements += verify_elements("write-conn", "pattern", pf, fv);
            }
        }
        if (verify && rep == 0) // the read arrays, unchanged by the reads
        {
            bad_elements += verify_elements("varied", "pattern", pvar, vvar);
            bad_elements += verify_elements("connectivity", "pattern", pconn, vconn);
        }
    }

    if (!csv)
    {
        println("{:<17} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>8} {:>8}", "case", "vector med", "vector min",
                "zero_any med", "zero_any min", "pattern med", "pattern min", "z/vec", "p/vec");
        const double per = 1e9 / static_cast<double>(n_elems);
        for (int c = 0; c < n_cases; ++c)
        {
            const double vm = median(times[c][0]) * per;
            const double zm = median(times[c][1]) * per;
            const double pm = median(times[c][2]) * per;
            println("{:<17} {:>10.3f} {:>10.3f} {:>12.3f} {:>12.3f} {:>11.3f} {:>11.3f} {:>8.2f} {:>8.2f}",
                    case_names[c], vm, minimum(times[c][0]) * per, zm, minimum(times[c][1]) * per, pm,
                    minimum(times[c][2]) * per, vm > 0. ? zm / vm : 0., vm > 0. ? pm / vm : 0.);
        }
        println("(ns per element, wall clock / n; z/vec, p/vec = median / vector median)");

        // The connectivity's memory. Per chunk: the chunk object and its second-value slot;
        // per materialised chunk its buffer; per motif its entry (shared).
        const double buf    = static_cast<double>(chunk_sz) * sizeof(T);
        const double per_ch = static_cast<double>(sizeof(PArr::ChunkType)) + sizeof(T);
        const double vbytes = static_cast<double>(n_elems) * sizeof(T);
        const double zbytes = static_cast<double>(n_chunks) * per_ch + static_cast<double>(zc.mat) * buf;
        const auto   motifs = static_cast<double>(pc.motifs);
        const double pbytes = static_cast<double>(n_chunks) * per_ch + static_cast<double>(pc.mat) * buf + motifs * buf;
        println("connectivity memory ({} chunks of {}):", n_chunks, chunk_sz);
        println("  {:<9} {:>14.0f} bytes", "vector", vbytes);
        println("  {:<9} {:>14.0f} bytes  ({:.1f} %)  materialised {}, constant {}", "zero_any", zbytes,
                100. * zbytes / vbytes, zc.mat, zc.cst);
        println("  {:<9} {:>14.0f} bytes  ({:.1f} %)  materialised {}, constant {}, motif {}, distinct motifs {}",
                "pattern", pbytes, 100. * pbytes / vbytes, pc.mat, pc.cst, pc.motif, pc.motifs);
    }

    if (sum_mismatch || bad_elements)
    {
        std::fprintf(stderr, "access_cost_pattern: FAILED: %ld sum mismatch(es), %ld element(s) differ\n", sum_mismatch,
                     bad_elements);
        return 1;
    }
    return 0;
}
