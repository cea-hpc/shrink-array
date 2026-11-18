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

#include "chunk/build_options.h" // CHUNK_OPT_DO_CHECK

#if CHUNK_OPT_DO_CHECK

    #include <cmath>   // std::isnan
    #include <complex> // std::complex

    #include "chunk/internal/invariant_check.h" // CHUNK_INVARIANT_FAILED

namespace shrink_array
{

/// @brief Report element @p i as NaN (location, backtrace) and abort.
/// @param i        Index of the element.
/// @param in_array true: @p i is the index in the array; false: only the
///                 position in its chunk is known (a chunk read, a proxy not
///                 armed by an array).
inline void global_check_nan(long i, bool in_array) noexcept
{
    if (in_array)
    {
        CHUNK_INVARIANT_FAILED("element {} of the array is NaN (CHUNK_DO_CHECK)", i);
    }
    CHUNK_INVARIANT_FAILED("the element at position {} of its chunk is NaN (CHUNK_DO_CHECK; "
                           "the index in the array is not known here)",
                           i);
}

/// @brief Abort when @p val is NaN.
/// @param val      Value to check.
/// @param i        Index of the element, reported on failure.
/// @param in_array See global_check_nan(long, bool).
inline void global_check_nan(const double& val, long i, bool in_array) noexcept
{
    if (std::isnan(val))
    {
        global_check_nan(i, in_array);
    }
}

/// @brief Abort when @p val is NaN.
/// @param val      Value to check.
/// @param i        Index of the element, reported on failure.
/// @param in_array See global_check_nan(long, bool).
inline void global_check_nan(const float& val, long i, bool in_array) noexcept
{
    if (std::isnan(val))
    {
        global_check_nan(i, in_array);
    }
}

/// @brief Abort when both the real and imaginary parts of @p val are NaN.
/// @param val      Value to check.
/// @param i        Index of the element, reported on failure.
/// @param in_array See global_check_nan(long, bool).
inline void global_check_nan(const std::complex<double>& val, long i, bool in_array) noexcept
{
    if (std::isnan(val.real()) && std::isnan(val.imag()))
    {
        global_check_nan(i, in_array);
    }
}

/// @brief Other element types: no check.
template <typename T>
void global_check_nan([[maybe_unused]] const T& val, [[maybe_unused]] long i, [[maybe_unused]] bool in_array) noexcept
{
}

} // namespace shrink_array

#endif
