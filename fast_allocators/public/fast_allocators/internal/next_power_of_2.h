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

#include <bit> // std::bit_ceil (C++20)

namespace fast_alloc
{

/**
 * @brief Smallest power of two >= @p value; 1 for 0 and 1.
 * @details std::bit_ceil, which is defined for 0 and 1 (both give 1). The
 *          slab-size rule, TCSlab::slab_bytes_for(), rounds with it.
 */
[[nodiscard]] inline unsigned long NextPowerOf2(unsigned long value) noexcept
{
    return std::bit_ceil(value);
}

} // namespace fast_alloc

