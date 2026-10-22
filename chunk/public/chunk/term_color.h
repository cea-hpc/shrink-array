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
    const char* reset;   ///< Reset all attributes (SGR 0).
    const char* bold;    ///< Bold (SGR 1).
    const char* red;     ///< Red foreground.
    const char* green;   ///< Green foreground.
    const char* yellow;  ///< Yellow foreground.
    const char* cyan;    ///< Cyan foreground.
    const char* white;   ///< White foreground.
    const char* magenta; ///< Magenta foreground.
};

/// @brief The one shared instance.
///
/// `inline`, not `static`: one object for the program, not one per translation unit.
inline const TermColor tc = []() noexcept
{
    const bool i = isatty(STDOUT_FILENO) != 0;
    return TermColor{
        .reset   = i ? "\033[0m" : "",
        .bold    = i ? "\033[1m" : "",
        .red     = i ? "\033[31m" : "",
        .green   = i ? "\033[32m" : "",
        .yellow  = i ? "\033[33m" : "",
        .cyan    = i ? "\033[36m" : "",
        .white   = i ? "\033[37m" : "",
        .magenta = i ? "\033[35m" : "",
    };
}();

} // namespace shrink_array
