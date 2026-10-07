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

#include <cstdio> // stderr
#include <print>  // std::println

// CHUNK_HAVE_STACKTRACE: set by chunk/CMakeLists.txt when the standard library
// has std::stacktrace (and linked with the archive that holds it).
#ifdef CHUNK_HAVE_STACKTRACE
    #include <stacktrace> // std::stacktrace
#else
    #include <array>      // std::array
    #include <cstdlib>    // std::free (demangled name)
    #include <cxxabi.h>   // abi::__cxa_demangle
    #include <execinfo.h> // backtrace, backtrace_symbols
#endif

/**
 * @brief Print the calling thread's backtrace (up to 100 frames) to stderr.
 * @details With std::stacktrace: demangled names, and file:line when the
 *          binary has debug info. No frame is skipped: this function is
 *          inline, and when it is inlined the first frame is its caller's.
 *          Otherwise <execinfo.h>: module+offset, names demangled when the
 *          symbol is exported (-rdynamic).
 */
inline void print_backtrace()
{
    constexpr int max_frames = 100;
#ifdef CHUNK_HAVE_STACKTRACE
    std::println(stderr, "{}", std::stacktrace::current(0, max_frames));
#else
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
                std::println(stderr, "{} : {}+{}", symbols[i], real_name, offset_begin + 1);
                std::free(real_name);
            }
            else
            {
                std::println(stderr, "{} : {}+{}", symbols[i], mangled_name, offset_begin + 1);
            }
        }
        else
        {
            std::println(stderr, "{}", symbols[i]);
        }
    }

    std::free(static_cast<void*>(symbols));
#endif
}
