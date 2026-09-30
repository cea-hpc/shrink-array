//
// Created by ballereaup on 28/04/2026.
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

#include <climits>
#include <variant>
#include <type_traits>

#include "chunk/internal/chunk_size_config.h" // __chunk_size (compile-time)

namespace shrink_array
{

// Backing type for the per-chunk counters (nb_val_eq_zero, nb_val_eq_*_value).
// These count cells within a chunk, so they must be able to represent values
// up to __chunk_size (a full chunk of zeros has nb_val_eq_zero == __chunk_size,
// tested by is_full_zero()). This was a raw `unsigned char`, which overflows
// for __chunk_size > 255: at CHUNK_SIZE=512 the count wrapped, is_full_zero()
// could never become true, cells were left unmaterialised and read back as
// garbage (subnormal doubles) — a sequential (single-thread, non-OpenMP)
// correctness bug that appeared past 255. `unsigned short` (up to 65535) covers
// every realistic chunk size while staying compact (2 bytes).
using counter_value_t = unsigned short;

static_assert(__chunk_size <= USHRT_MAX,
              "CHUNK_SIZE exceeds the per-chunk counter capacity (unsigned short). "
              "Widen counter_value_t in conditional_counter.h if you really need "
              "chunks larger than 65535 elements.");

/// @brief A per-chunk cell counter that is a real counter_value_t when
///        @p Enabled and an empty type otherwise, so the disabled variants
///        cost no storage. @p Tag lets two disabled counters of one class be
///        distinct empty TYPES, so both can sit at offset 0 under
///        [[no_unique_address]] (two members of the same empty type may not
///        share an address) and the class stays empty.
template <bool Enabled, int Tag = 0>
struct ConditionalCounter
{
    /// @name The operators do nothing when disabled; the conversion reads 0.
    /// @{
    void operator++() noexcept
    {
        if constexpr (Enabled)
            ++value;
    }
    void operator--() noexcept
    {
        if constexpr (Enabled)
            --value;
    }
    void operator=(counter_value_t v) noexcept
    {
        if constexpr (Enabled)
            value = v;
    }

    operator counter_value_t() const noexcept
    {
        if constexpr (Enabled)
            return value;
        else
            return 0;
    }

  private:
    [[no_unique_address]] std::conditional_t<Enabled, counter_value_t, std::monostate> value{};

    /// @}
};

} // namespace shrink_array
