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

    #include <cmath> // std::isnan
    #include <complex>
    #include <cstdlib>  // std::abort
    #include <iostream> // std::cerr

static void global_check_nan(long i)
{
    std::cerr << " ** FATAL: Element : " << i << " of the array is nan\n\n";
    std::abort();
}

static void global_check_nan(const double& val, long i)
{
    if (std::isnan(val))
    {
        global_check_nan(i);
    }
}

static void global_check_nan(const float& val, long i)
{
    if (std::isnan(val))
    {
        global_check_nan(i);
    }
}

static void global_check_nan(const std::complex<double>& val, long i)
{
    if (std::isnan(val.real()) && std::isnan(val.imag()))
    {
        global_check_nan(i);
    }
}

template <typename T>
void global_check_nan(const T& val, long n)
{
}

#endif
// extern void global_check_nan(long n, const char* name);