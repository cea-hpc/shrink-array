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

#include <cstddef> // size_t

constexpr bool is_power_of_tow(int value) noexcept
{
    return ((value & (value - 1)) == 0) && (value != 0);
}

constexpr size_t log2int(size_t value) noexcept
{
    return (value < 2) ? 0 : 1 + log2int(value / 2);
}
