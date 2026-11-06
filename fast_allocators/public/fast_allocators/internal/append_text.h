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

/**
 * @file append_text.h
 * @brief Text and integers appended to a fixed char buffer, noexcept.
 *
 * @details For the noexcept paths that only put integers in a string (a
 *          /proc path, a pid in a file name, a list of chunk indices): there
 *          std::format_to_n cannot fail -- its format string is checked at
 *          compile time and an integer's formatter does not throw -- but its
 *          runtime path holds a `throw format_error` a static analyser
 *          cannot prove unreachable (Coverity: UNCAUGHT_EXCEPT in a noexcept
 *          function). std::to_chars is noexcept by definition, allocates
 *          nothing and touches nothing but the buffer.
 *
 *          Both functions write within [out, end) and return one past the
 *          last character written; what does not fit is cut, as
 *          std::format_to_n cuts. The caller places the terminating NUL.
 */

#include <algorithm>    // std::min
#include <charconv>     // std::to_chars
#include <concepts>     // std::integral
#include <cstddef>      // std::size_t
#include <cstring>      // std::memcpy
#include <limits>       // std::numeric_limits (append_int's digits)
#include <string_view>  // std::string_view
#include <system_error> // std::errc

namespace fast_alloc
{

/// @brief Copy @p text at @p out, cut at @p end.
/// @return One past the last character written.
[[nodiscard]] inline char* append_text(char* out, char* end, std::string_view text) noexcept
{
    const auto n = std::min(text.size(), static_cast<std::size_t>(end - out));
    std::memcpy(out, text.data(), n);
    return out + n;
}

/// @brief Write @p value in decimal at @p out, cut at @p end.
/// @return One past the last character written.
template <std::integral Int>
[[nodiscard]] char* append_int(char* out, char* end, Int value) noexcept
{
    char       digits[std::numeric_limits<Int>::digits10 + 3]; // sign + one digit digits10 does not count
    const auto res = std::to_chars(digits, digits + sizeof(digits), value);
    return append_text(out, end, std::string_view{digits, res.ec == std::errc{} ? res.ptr : digits});
}

} // namespace fast_alloc
