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

#ifdef DO_CHECK

    #include <cmath>    // std::isnan
    #include <complex>  // std::complex
    #include <cstdlib>  // std::abort
    #include <iostream> // std::cerr

/// @brief Report element @p i as NaN on std::cerr and abort.
/// @param i Index of the element.
static void global_check_nan(long i) noexcept
{
    std::cerr << " ** FATAL: Element : " << i << " of the array is nan\n\n";
    std::abort();
}

/// @brief Abort when @p val is NaN.
/// @param val Value to check.
/// @param i Index of the element, reported on failure.
static void global_check_nan(const double& val, long i) noexcept
{
    if (std::isnan(val))
    {
        global_check_nan(i);
    }
}

/// @brief Abort when @p val is NaN.
/// @param val Value to check.
/// @param i Index of the element, reported on failure.
static void global_check_nan(const float& val, long i) noexcept
{
    if (std::isnan(val))
    {
        global_check_nan(i);
    }
}

/// @brief Abort when both the real and imaginary parts of @p val are NaN.
/// @param val Value to check.
/// @param i Index of the element, reported on failure.
static void global_check_nan(const std::complex<double>& val, long i) noexcept
{
    if (std::isnan(val.real()) && std::isnan(val.imag()))
    {
        global_check_nan(i);
    }
}

/// @brief Other element types: no check.
template <typename T>
void global_check_nan([[maybe_unused]] const T& val, [[maybe_unused]] long i) noexcept
{
}

#endif
