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

#include <array>      // std::array
#include <execinfo.h> // backtrace, backtrace_symbols
#include <cxxabi.h>   // abi::__cxa_demangle
#include <cstdlib>    // std::free (demangled name)
#include <iostream>   // std::cerr

namespace shrink_array
{

/// @brief Print the calling thread's backtrace (up to 100 frames) to std::cerr, demangling names when possible.
inline void print_backtrace()
{
    constexpr int                 max_frames = 100;
    std::array<void*, max_frames> buffer{};
    const int                     nptrs   = backtrace(buffer.data(), max_frames);
    char**                        symbols = backtrace_symbols(buffer.data(), nptrs);
    if (symbols == nullptr)
    {
        return; // backtrace_symbols could not allocate
    }

    for (int i = 0; i < nptrs; ++i)
    {
        const char* mangled_name = nullptr;
        char*       offset_begin = nullptr;
        char*       offset_end   = nullptr;

        for (char* p = symbols[i]; *p != '\0'; ++p)
        {
            if (*p == '(')
            {
                mangled_name = p + 1;
            }
            else if (*p == '+')
            {
                offset_begin = p;
            }
            else if (*p == ')')
            {
                offset_end = p;
                break;
            }
        }

        if (mangled_name && offset_begin && offset_end && mangled_name < offset_begin)
        {
            *offset_begin = '\0';
            *offset_end   = '\0';

            int   status    = -1;
            char* real_name = abi::__cxa_demangle(mangled_name, nullptr, nullptr, &status);

            if (status == 0 && real_name)
            {
                std::cerr << symbols[i] << " : " << real_name << "+" << (offset_begin + 1) << '\n';
                std::free(real_name);
            }
            else
            {
                std::cerr << symbols[i] << " : " << mangled_name << "+" << (offset_begin + 1) << '\n';
            }
        }
        else
        {
            std::cerr << symbols[i] << '\n';
        }
    }

    std::free(static_cast<void*>(symbols));
}

} // namespace shrink_array

