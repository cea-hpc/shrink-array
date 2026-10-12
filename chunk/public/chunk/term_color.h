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

namespace shrink_array
{

/// @brief ANSI colour codes, or empty strings when stdout is not a terminal.
///
/// const char*, not std::string: the instance is built at static-init time, where an
/// allocation that throws terminates before main (bugprone-throwing-static-initialization).
struct TermColor
{
    const char* RESET;   ///< Reset all attributes (SGR 0).
    const char* BOLD;    ///< Bold (SGR 1).
    const char* RED;     ///< Red foreground.
    const char* GREEN;   ///< Green foreground.
    const char* YELLOW;  ///< Yellow foreground.
    const char* CYAN;    ///< Cyan foreground.
    const char* WHITE;   ///< White foreground.
    const char* MAGENTA; ///< Magenta foreground.
};

/// @brief The one shared instance.
///
/// `inline`, not `static`: one object for the program, not one per translation unit.
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

} // namespace shrink_array
