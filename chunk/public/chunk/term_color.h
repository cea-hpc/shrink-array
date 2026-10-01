//
// Created by ballereaup on 30/04/2026.
//
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

#include <unistd.h> // isatty / STDOUT_FILENO

/// @brief ANSI colour codes, or empty strings when stdout is not a terminal.
///
/// const char*, not std::string: the members hold string literals either way, and
/// eight std::string assignments in a constructor that runs at STATIC INIT time
/// can throw std::bad_alloc where nothing can catch it -- std::terminate, before
/// main (bugprone-throwing-static-initialization). Literals cannot throw and cost
/// no allocation.
struct TermColor
{
    const char* RESET;
    const char* BOLD;
    const char* RED;
    const char* GREEN;
    const char* YELLOW;
    const char* CYAN;
    const char* WHITE;
    const char* MAGENTA;
};

/// @brief The one shared instance.
///
/// `inline`, not `static`: at namespace scope in a header every translation unit
/// got its own copy, each calling isatty() and, before the change above,
/// allocating eight strings. One object now, initialised once.
inline const TermColor tc = []() noexcept
{
    const bool i = isatty(STDOUT_FILENO) != 0;
    return TermColor{
        /*RESET  */ i ? "\033[0m" : "",
        /*BOLD   */ i ? "\033[1m" : "",
        /*RED    */ i ? "\033[31m" : "",
        /*GREEN  */ i ? "\033[32m" : "",
        /*YELLOW */ i ? "\033[33m" : "",
        /*CYAN   */ i ? "\033[36m" : "",
        /*WHITE  */ i ? "\033[37m" : "",
        /*MAGENTA*/ i ? "\033[35m" : "",
    };
}();