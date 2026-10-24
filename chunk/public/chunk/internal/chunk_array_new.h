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
 * @details Each variant's `operator new[]` allocates, then points every chunk
 *          at its first constant (in parallel when large enough); only the
 *          per-chunk initialiser differs (`init_values_cst_val()` on zero,
 *          `init_after_constructor()` on zero_one_value / zero_any), which the helper takes.
 *
 * @note For a PLACEMENT-new array only: `new C[n]` runs the default
 *       constructors afterwards, overwriting it. The library never array-news
 *       a chunk.
 *
 * @see examples/validation/chunk_array_new_validation.cpp -- covers every size
 *      regime of the three variants, inside and outside a parallel region.
 */

#include <cstddef> // std::size_t

#include "chunk/internal/parallel_storage_init.h" // for_each_chunk, parallel_init_min_chunks

namespace shrink_array::chunk_array_new
{

/**
 * @brief Run @p init on each of the @p count chunks at @p chunks, on a team of its
 *        own from parallel_storage_init::parallel_init_min_chunks chunks on,
 *        outside a region (parallel_storage_init::for_each_chunk()).
 *
 * @details A memory-bound sweep (one store per chunk). Inside a parallel region
 *          it stays on the calling thread, deliberately:
 *
 *          - an orphaned `omp for` requires EVERY thread of the team to call
 *            this, which no allocation does: called from one thread (`single`,
 *            `master`, a branch), it hangs on the loop's barrier;
 *          - a nested `omp parallel` is serialised by default runtimes
 *            (max_active_levels 1), and with nesting enabled would open N x N/2
 *            threads from an allocation.
 *
 * @tparam C    chunk type
 * @tparam Init callable `void(C&)`
 * @param chunks    first chunk
 * @param count    how many
 * @param init what to run on each
 */
template <typename C, typename Init>
void init_chunks(C* chunks, std::size_t count, Init init) noexcept
{
    parallel_storage_init::for_each_chunk(static_cast<int>(count), [chunks, &init](int k) { init(chunks[k]); });
}

} // namespace shrink_array::chunk_array_new
