/*
 * Copyright 2024-2026 CEA (www.cea.fr)
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

#pragma once
/**
 * @file parallel_storage_init.h
 * @brief The two parallel pieces of an array's storage (re)build:
 *        the first touch of its per-chunk vectors, and the per-chunk
 *        initialisation loop.
 *
 * @details rebuild_storage() of the three variants sizes a handful of
 *          per-chunk vectors (chunk_list, and for Chunk2A the second-value
 *          and type-of-constant rows), then runs one init per chunk. The init
 *          loop was parallel; the sizing was not, and it was most of the time:
 *          measured on ArrayChunk2A<double> with 4.2 M chunks, 70 ms serial
 *          (chunk_list 40, the Proxy2A rows 30) against an 84 ms init loop
 *          that a team divides. Most of those 70 ms are page faults: a fresh
 *          vector of tens of MB is faulted in page by page by the one thread
 *          that value-initialises it -- which also puts every page on that
 *          thread's NUMA node.
 *
 *          reserve_first_touched() takes the faults on the team: it reserves
 *          the vector and touches one byte per page of the new storage in a
 *          static-schedule loop, so the serial resize that follows only
 *          stores, and the pages land where the static per-chunk loops will
 *          use them.
 *
 *          for_each_chunk() is the init loop of the three variants, in one
 *          place: a team of its own outside a region for a large array, the
 *          calling thread otherwise -- never an orphaned `omp for`, which is
 *          wrong for an array built inside a region (an array created by each
 *          thread of a team gets 1/N of its chunks initialised; one created
 *          under `omp single` blocks on a barrier the others never reach).
 *          Same rule as chunk_array_new::init_chunks.
 */

#include <cstddef> // std::size_t

#include "chunk/PMP_CHUNK.h" // CHUNK_PMP_MAX_THREADS / CHUNK_PMP_IN_PARALLEL_REGION
#include "chunk/config.h"    // build_config::use_openmp

namespace shrink_array::parallel_storage_init
{

/// @brief Below this many bytes a vector is faulted in by the calling thread:
///        a team costs more than the faults it would share.
inline constexpr std::size_t kFirstTouchMinBytes = std::size_t{2} << 20U;

/// @brief Page stride of the first touch (the smallest page; a huge page
///        is touched several times, harmlessly).
inline constexpr std::size_t kTouchStride = 4096;

/// @brief May this call open a team of its own? Outside a parallel region,
///        with at least two threads available.
inline bool can_open_team() noexcept
{
    if constexpr (build_config::use_openmp)
    {
        return CHUNK_PMP_MAX_THREADS >= 2 && !CHUNK_PMP_IN_PARALLEL_REGION;
    }
    return false;
}

/**
 * @brief Reserve @p v for @p n elements and fault its storage in from a team.
 *
 * @details Only for a vector about to be filled from empty (size 0, too small
 *          a capacity): a vector that already holds elements or a buffer large
 *          enough -- the spare chunk_list a re-chunk reuses -- is left alone,
 *          its pages are resident. The touch writes one zero byte per page of
 *          raw storage that no element occupies yet; the resize that follows
 *          constructs the elements over it.
 * @param v the vector
 * @param n the element count it is about to be resized to
 */
template <typename Vec>
void reserve_first_touched(Vec& v, std::size_t n)
{
    if constexpr (build_config::use_openmp)
    {
        using value_type        = typename Vec::value_type;
        const std::size_t bytes = n * sizeof(value_type);
        if (!v.empty() || v.capacity() >= n || bytes < kFirstTouchMinBytes || !can_open_team())
        {
            return;
        }
        v.reserve(n);
        auto*      base   = reinterpret_cast<volatile unsigned char*>(v.data());
        const long npages = static_cast<long>((bytes + kTouchStride - 1) / kTouchStride);
#pragma omp parallel for schedule(static)
        for (long p = 0; p < npages; ++p)
        {
            base[static_cast<std::size_t>(p) * kTouchStride] = 0;
        }
    }
    else
    {
        (void)v;
        (void)n;
    }
}

/**
 * @brief Run @p init(i) for every chunk index i in [0, @p n): on a team of its
 *        own when outside a region and @p n exceeds @p min_parallel, on the
 *        calling thread otherwise.
 * @param n            chunk count
 * @param min_parallel chunk count up to which the loop stays serial
 * @param init         callable `void(int)`; iterations must be independent
 */
template <typename Init>
void for_each_chunk(int n, int min_parallel, Init&& init)
{
    if constexpr (build_config::use_openmp)
    {
        if (n > min_parallel && can_open_team())
        {
#pragma omp parallel for schedule(static)
            for (int i = 0; i < n; ++i)
            {
                init(i);
            }
            return;
        }
    }
    for (int i = 0; i < n; ++i)
    {
        init(i);
    }
}

} // namespace shrink_array::parallel_storage_init
