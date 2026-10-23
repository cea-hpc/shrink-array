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

// The lock of the library's diagnostic reports: the residual and self-heal
// dumps of compress_int() (zero::Array, zero_one_value::Array, zero_any::Array), several
// lines each, that must not interleave when several threads print at once.
//
// It replaces the unnamed `#pragma omp critical` those reports used (threads
// branch, step 2): the same exclusion, with no OpenMP runtime involved, and no
// longer shared with the application's own unnamed critical sections.
//
// Constant-initialised and never destroyed (Immortal, immortal.h): a report
// can be printed from a global array's constructor or destructor.

#include <mutex> // std::mutex

#include "chunk/internal/immortal.h" // Immortal

namespace shrink_array::internal
{

/// Storage of diag_print_mutex(); use the function.
inline constinit Immortal<std::mutex> diag_print_mutex_storage{};

/// @brief The mutex every multi-line diagnostic report holds while it prints.
/// @warning Not reentrant: nothing that runs under it may print a report.
[[nodiscard]] inline std::mutex& diag_print_mutex() noexcept
{
    return diag_print_mutex_storage.value;
}

} // namespace shrink_array::internal
