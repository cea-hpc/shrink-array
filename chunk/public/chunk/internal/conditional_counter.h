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

#include <climits>     // USHRT_MAX
#include <variant>     // std::monostate
#include <type_traits> // std::conditional_t

#include "chunk/internal/chunk_size_config.h" // static_chunk_size (compile-time)

namespace shrink_array
{

/// @brief Backing type for the per-chunk counters (nb_val_eq_zero_,
///        nb_val_eq_*_value).
/// @details These count cells within a chunk, so they must be able to represent values
///          up to static_chunk_size (a full chunk of zeros has nb_val_eq_zero_ == static_chunk_size,
///          tested by is_full_zero()). Do not narrow it to `unsigned char`: past 255 the
///          count wraps, is_full_zero() never holds and cells read back as garbage, even
///          single-threaded. `unsigned short` covers every realistic chunk size in 2 bytes.
using CounterValueT = unsigned short;

static_assert(static_chunk_size <= USHRT_MAX,
              "CHUNK_SIZE exceeds the per-chunk counter capacity (unsigned short). "
              "Widen counter_value_t in conditional_counter.h if you really need "
              "chunks larger than 65535 elements.");

/// @brief A per-chunk cell counter that is a real CounterValueT when
///        @p Enabled and an empty type otherwise, so the disabled variants
///        cost no storage. @p Tag lets two disabled counters of one class be
///        distinct empty TYPES, so both can sit at offset 0 under
///        [[no_unique_address]] (two members of the same empty type may not
///        share an address) and the class stays empty.
template <bool Enabled, int Tag = 0>
class ConditionalCounter
{
  public:
    /// @name The operators do nothing when disabled; the conversion reads 0.
    /// @{
    void operator++() noexcept
    {
        if constexpr (Enabled)
        {
            ++value_;
        }
    }
    void operator--() noexcept
    {
        if constexpr (Enabled)
        {
            --value_;
        }
    }
    void operator=(CounterValueT value) noexcept
    {
        if constexpr (Enabled)
        {
            value_ = value;
        }
    }

    operator CounterValueT() const noexcept
    {
        if constexpr (Enabled)
        {
            return value_;
        }
        else
        {
            return 0;
        }
    }
    /// @}

  private:
    [[no_unique_address]] std::conditional_t<Enabled, CounterValueT, std::monostate> value_{};
};

} // namespace shrink_array
