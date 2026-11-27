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
 * @file kokkos_getting_started.cu
 * @brief A compressed array in a Kokkos code, from creation to reads: the GPU counterpart of
 *        examples/getting_started (docs/api.md section 11, docs/variants.md section 5).
 *
 * @details One zero_any array of 2^26 doubles (512 MiB as a Kokkos::View), every element 0.0
 *          at first and no buffer allocated:
 *          1. a regular loop over the first half, by the library's chunk-aligned loop
 *             (parallel_for_chunks): varied values on the first eighth, 3.5 on the next three
 *             eighths -- a chunk written with one value takes it as its constant, no buffer;
 *          2. any Kokkos kernels, through write() and end_write(): 3.5 written element by
 *             element over the fifth eighth (its chunks get a buffer), one element in 1000 of
 *             the last quarter set to 2.0, and 1.0 added to every element of the first eighth;
 *          3. compress(), between phases: the chunks that became uniform -- the fifth
 *             eighth, all 3.5 -- fold back;
 *          4. any Kokkos kernel reading through read(): every element checked.
 *          The array then holds a buffer for the chunks of varied data only.
 *
 *          Built with -DCHUNK_DEVICE_KOKKOS=ON (Kokkos fetched and built for CUDA or HIP), linked
 *          to the CMake target shrink_array::device as a program of yours would be. Exit status
 *          0 when every element reads as expected.
 */

#include <cstdint> // std::int64_t
#include <cstdio>  // std::printf

#include <Kokkos_Core.hpp> // Kokkos::ScopeGuard, parallel_for, parallel_reduce

#include "chunk/device/kokkos_adaptor.h" // zero_any::KokkosArray

namespace sa = shrink_array;

namespace
{

constexpr std::int64_t n = std::int64_t{1} << 26;

/// @brief What element i holds after the phases below.
KOKKOS_INLINE_FUNCTION double expected(std::int64_t i)
{
    if (i < n / 8)
    {
        return 1.0 + static_cast<double>(i % 7) + 1.0; // step 1, then += 1.0 in step 2
    }
    if (i < 5 * n / 8)
    {
        return 3.5; // step 1 (one constant, no buffer), then step 2 element by element
    }
    if (i >= 3 * n / 4 && (i - 3 * n / 4) % 1000 == 0)
    {
        return 2.0; // step 2
    }
    return 0.0; // never written
}

} // namespace

int main(int argc, char** argv)
{
    Kokkos::ScopeGuard guard(argc, argv);
    long               wrong = 0;
    {
        // n elements, every one 0.0, no buffer: a first arena of one buffer per chunk is
        // reserved (no phase can run out of it); KokkosArray(n, arena_buffers) starts smaller.
        sa::zero_any::KokkosArray<double> a(n);

        // 1. A regular loop: f(i) for i in [0, n / 2), one block per chunk. A phase by itself.
        a.parallel_for_chunks(0, n / 2, KOKKOS_LAMBDA(std::int64_t i) {
            return i < n / 8 ? 1.0 + static_cast<double>(i % 7) : 3.5;
        });

        // 2. Any Kokkos kernels between write() and end_write(): w(i) = v, w(i) += v, ...
        //    Two writes of one element by two threads in a phase are a race, as on a View.
        {
            const auto w = a.write();
            Kokkos::parallel_for("sparse", n / 4 / 1000 + 1, KOKKOS_LAMBDA(std::int64_t k) {
                const std::int64_t i = 3 * n / 4 + 1000 * k;
                if (i < n)
                {
                    w(i) = 2.0;
                }
            });
            Kokkos::parallel_for("uniform", n / 8, KOKKOS_LAMBDA(std::int64_t k) { w(n / 2 + k) = 3.5; });
            Kokkos::parallel_for("add", n / 8, KOKKOS_LAMBDA(std::int64_t i) { w(i) += 1.0; });
            a.end_write();
        }

        // 3. Between phases: the chunks that became uniform give their buffer back (the fifth
        //    eighth: n / 8 / 1024 chunks at S = 1024).
        const unsigned long long folded = a.compress();

        // 4. A read phase: read() in any kernel, until the next write().
        const auto r = a.read();
        Kokkos::parallel_reduce(
            "check", n, KOKKOS_LAMBDA(std::int64_t i, long& bad) { bad += r(i) == expected(i) ? 0 : 1; }, wrong);

        const double flat_gb = static_cast<double>(n) * sizeof(double) / 1e9;
        std::printf("kokkos_getting_started (%s): n = %lld doubles, %ld wrong\n", Kokkos::DefaultExecutionSpace::name(),
                    static_cast<long long>(n), wrong);
        std::printf("  %llu chunks folded by compress(); %llu of %lld chunks hold a buffer\n", folded,
                    a.materialised(), static_cast<long long>((n + a.chunk_size() - 1) / a.chunk_size()));
        std::printf("  %.3f GB held against %.3f GB as a Kokkos::View (%.1f %%)\n", static_cast<double>(a.bytes()) / 1e9,
                    flat_gb, 100.0 * static_cast<double>(a.bytes()) / 1e9 / flat_gb);
    } // the array gives its arenas back here, before Kokkos finalises
    return wrong == 0 ? 0 : 1;
}
