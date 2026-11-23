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

// ============================================================================
// The variant a validation harness is built for, and the helpers that depend
// on it: the array type, its name, the phase brackets, the second value.
// ============================================================================
//
// One executable per variant (validation_<variant>_<geometry>): the variant is
// chunk_config::use_zero_any / use_zero_or_value (zero otherwise). A harness
// that needs a different version of one of these -- pattern's variant name, an
// end_phase() that also runs the serial pass -- defines its own in its
// namespace, which hides this one.
// ============================================================================

#pragma once

#include <type_traits> // std::conditional_t

#include "chunk/internal/chunk_traits.h" // forward decls + default IntType = int
#include "chunk/config.h"                // chunk_config::use_zero*, build_config::needs_explicit_phases
#include "chunk/phases.h"                // begin_parallel_phase / finalize_parallel_phase
#include "chunk/zero_any_array.h"        // zero_any::Array
#include "chunk/zero_array.h"            // zero::Array
#include "chunk/zero_or_value_array.h"   // zero_or_value::Array

namespace validation
{

/// @brief The array of the variant this executable is built for.
/// @tparam T the element type.
/// @tparam IntType the index type, if not the library's default (int).
template <typename T, typename... IntType>
using Array = std::conditional_t<shrink_array::chunk_config::use_zero_any,
                                 shrink_array::zero_any::Array<T, IntType...>,
                                 std::conditional_t<shrink_array::chunk_config::use_zero_or_value,
                                                    shrink_array::zero_or_value::Array<T, IntType...>,
                                                    shrink_array::zero::Array<T, IntType...>>>;

/// @brief The name of the variant this executable is built for, for the
///        harnesses' report lines.
constexpr const char* variant_name()
{
    if constexpr (shrink_array::chunk_config::use_zero_any)
    {
        return "zero_any";
    }
    else if constexpr (shrink_array::chunk_config::use_zero_or_value)
    {
        return "zero_or_value";
    }
    else
    {
        return "zero";
    }
}

/// @brief Opens a parallel phase where the program has to (no OMPT tool);
///        a no-op under OMPT (the callbacks do it) and serially.
inline void begin_phase()
{
    if constexpr (shrink_array::build_config::needs_explicit_phases)
    {
        shrink_array::begin_parallel_phase();
    }
}

/// @brief Closes the phase begin_phase() opened (the between-region pass);
///        a no-op under OMPT and serially.
inline void end_phase()
{
    if constexpr (shrink_array::build_config::needs_explicit_phases)
    {
        shrink_array::finalize_parallel_phase();
    }
}

/// @brief arr.set_second_value(value) where the variant has one
///        (zero_or_value, zero_any); a no-op on zero.
/// @details zero_or_value compresses to the second value only once the array
///          declared one, and requires it before any write: its candidate
///          list is keyed by the second value.
template <typename Arr, typename Val>
void maybe_set_second_value(Arr& arr, const Val& value)
{
    if constexpr (requires(Arr probe) { probe.set_second_value(value); })
    {
        arr.set_second_value(value);
    }
}

} // namespace validation
