//
// Created by ballereaup on 28/05/25.
//
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

#include <complex>

/// @brief Lexicographic ordering of std::complex<T>: real part first, then imaginary part.
template <typename T>
class ComplexLess
{
  public:
    /// @brief Compare two complex numbers.
    /// @return true when @p lhs orders before @p rhs.
    bool operator()(const std::complex<T>& lhs, const std::complex<T>& rhs) const noexcept
    {
        if (lhs.real() < rhs.real())
            return true;
        if (lhs.real() > rhs.real())
            return false;
        return lhs.imag() < rhs.imag();
    }
};