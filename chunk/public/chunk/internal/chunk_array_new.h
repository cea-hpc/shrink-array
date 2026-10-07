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
 * @file chunk_array_new.h
 * @brief The array-new of the three chunk variants, in one place.
 *
 * @details Chunk, Chunk2 and Chunk2A each offer `operator new[]`, and each one
 *          does the same thing: allocate the raw memory, then point every
 *          chunk of the array at its first constant, in parallel when the
 *          array is large enough. Only the per-chunk initialiser differs
 *          (`init_values_cst_val()` on v1, `init_after_constructor()` on
 *          chunk2 / chunk2A), so that is what the helper below takes, and the
 *          three operators are one line each.
 *
 * @note These operators are for a PLACEMENT-new array. `new C[n]` runs the
 *       default constructors right after the operator, which overwrite what it
 *       wrote; nothing in the library array-news a chunk (the arrays hold a
 *       std::vector<chunk_type>, initialised by their own loops).
 *
 * @see examples/validation/chunk_array_new_validation.cpp -- the harness that
 *      covers every size regime of all three variants, inside and outside a
 *      parallel region.
 */

#include <cstddef> // std::size_t

#include "chunk/PMP_CHUNK.h" // CHUNK_PMP_MAX_THREADS / CHUNK_PMP_IN_PARALLEL_REGION
#include "chunk/config.h"    // build_config::use_openmp

namespace shrink_array::chunk_array_new
{

/// @brief Chunk count under which the initialisation stays on the calling
///        thread -- the threshold the array paths use (ArrayChunk::create,
///        ArrayChunk2::rebuild_storage, ...).
inline constexpr std::size_t kParallelInitMinChunks = 500;

/**
 * @brief Run @p init on each of the @p count chunks at @p v, in parallel past
 *        kParallelInitMinChunks.
 *
 * @details The initialiser is one store per chunk, so this is a memory-bound
 *          sweep: worth a team on a large array, and only OUTSIDE a parallel
 *          region, where this function may open its own. Inside one it stays
 *          on the calling thread, deliberately:
 *
 *          - an orphaned `omp for` would require EVERY thread of the team to
 *            call this, which no allocation does. Called from one thread (an
 *            `omp single`, a `master`, a branch), the caller blocks on the
 *            loop's barrier and the others on theirs: measured, it hangs at
 *            4 threads;
 *          - a nested `omp parallel` is worse: every default runtime
 *            serialises it (max_active_levels 1), so it runs on ONE thread
 *            anyway, and with nesting enabled it would open N x N/2 threads
 *            from an allocation.
 *
 * @tparam C    chunk type
 * @tparam Init callable `void(C&)`
 * @param v    first chunk
 * @param count    how many
 * @param init what to run on each
 */
template <typename C, typename Init>
void init_chunks(C* v, std::size_t count, Init init) noexcept
{
    if constexpr (build_config::use_openmp)
    {
        if (count >= kParallelInitMinChunks && CHUNK_PMP_MAX_THREADS >= 2 && !CHUNK_PMP_IN_PARALLEL_REGION)
        {
#pragma omp parallel for simd
            for (std::size_t k = 0; k < count; ++k)
            {
                init(v[k]);
            }
            return;
        }
    }

#pragma omp simd
    for (std::size_t k = 0; k < count; ++k)
    {
        init(v[k]);
    }
}

} // namespace shrink_array::chunk_array_new
