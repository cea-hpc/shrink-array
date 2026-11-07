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

// The double guard of a header the module reads (chunk/internal/module_export.h).
#if defined(SHRINK_ARRAY_MODULE_DEPS) ? !defined(SHRINK_ARRAY_MANAGE_OMPT_H_DEPS) : !defined(SHRINK_ARRAY_MANAGE_OMPT_H)
    #if defined(SHRINK_ARRAY_MODULE_DEPS)
        #define SHRINK_ARRAY_MANAGE_OMPT_H_DEPS
    #else
        #define SHRINK_ARRAY_MANAGE_OMPT_H
    #endif

/**
 * @file manage_ompt.h
 * @brief parallel_phase(): a parallel region with the phase protocol in place,
 *        and, through chunk/internal/phase_hooks.h, the phase hooks it calls
 *        (begin_parallel_phase(), finalize_parallel_phase(), ...) and their
 *        contract (docs/threading_backends.md, section 2).
 *
 * No header of the library includes this one (they include phase_hooks.h):
 * the module reads it in its purview and exports parallel_phase() by its own
 * declarations (chunk/internal/module_export.h).
 */

    #include <utility> // std::forward

    #include "chunk/internal/phase_hooks.h"   // phase_detail::parallel_phase_impl, the phase hooks
    #include "chunk/internal/module_export.h" // SHRINK_ARRAY_EXPORT (parallel_phase)

    #ifndef SHRINK_ARRAY_MODULE_DEPS // the module's global fragment reads the #includes only

namespace shrink_array
{

/**
 * @brief A parallel region with the phase protocol already in place:
 *        begin_parallel_phase() by every thread before @p body,
 *        finalize_parallel_phase() once the region is over.
 * @details A team of the backend (pmp::team). With OpenMP, the same as
 *          @code
 *          #pragma omp parallel num_threads(num_threads)
 *          {
 *              begin_parallel_phase();
 *              body();
 *          }
 *          finalize_parallel_phase();
 *          @endcode
 *          with both calls left out under OMPT, where the runtime makes them.
 *          The worksharing goes inside @p body: pmp::current_team().for_static()
 *          and its kin, which every backend splits. An orphaned `#pragma omp for`
 *          splits only under OpenMP: the std backend runs @p body on each rank
 *          of its team with the pragma gone, so every rank runs the whole loop.
 *          A serial build runs @p body once.
 * @param num_threads team size (the num_threads clause); 0: the runtime's.
 * @param body        what every thread of the team runs.
 */
SHRINK_ARRAY_EXPORT template <typename Body>
void parallel_phase(int num_threads, Body&& body)
{
    phase_detail::parallel_phase_impl(num_threads, std::forward<Body>(body));
}

/// @brief parallel_phase() with the runtime's team size.
SHRINK_ARRAY_EXPORT template <typename Body>
void parallel_phase(Body&& body)
{
    phase_detail::parallel_phase_impl(0, std::forward<Body>(body));
}

} // namespace shrink_array

    #endif // SHRINK_ARRAY_MODULE_DEPS
#endif     // SHRINK_ARRAY_MANAGE_OMPT_H
