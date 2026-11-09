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

#pragma once

/**
 * @file threads.h
 * @brief The threads the library runs on, whatever its backend: OpenMP, its
 *        own std::thread teams (CHUNK_THREADS=std) or none (serial).
 *
 * A program sizes its per-thread data and numbers its threads with these, not
 * with omp_get_max_threads() / omp_get_thread_num(), so that it builds and
 * runs the same with every backend:
 *
 *     std::vector<double> partial(shrink_array::pmp::max_threads());
 *     shrink_array::parallel_phase([&] {
 *         partial[shrink_array::pmp::thread_rank()] += ...;
 *     });
 *
 * | function        | OpenMP                 | std backend                         | serial |
 * |-----------------|------------------------|-------------------------------------|--------|
 * | max_threads()   | omp_get_max_threads()  | the team size of the next region    | 1      |
 * | team_size()     | omp_get_num_threads()  | the size of the innermost team      | 1      |
 * | thread_rank()   | omp_get_thread_num()   | the rank in the innermost team, or  | 0      |
 * |                 |                        | the thread's registered rank        |        |
 */

#include "fast_allocators/internal/pmp_runtime.h" // fast_alloc::pmp::max_threads, team_size, thread_rank

namespace shrink_array::pmp
{

/// @brief Team size of the next parallel region opened without an explicit
///        thread count: what to size per-thread data with.
using ::fast_alloc::pmp::max_threads;

/// @brief Size of the calling thread's innermost team (1 outside a region).
using ::fast_alloc::pmp::team_size;

/// @brief The calling thread's rank, in [0, max_threads()) inside a region
///        the library's backend opened.
using ::fast_alloc::pmp::thread_rank;

} // namespace shrink_array::pmp
