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

#include <complex>     // std::complex
#include <type_traits> // std::is_same_v

namespace shrink_array
{

/// @brief True when T is an instance of std::complex.
template <typename T>
concept ComplexType = requires { typename T::value_type; } && std::is_same_v<T, std::complex<typename T::value_type>>;

/// @brief Compile-time settings of Chunk: CHUNK_SIZE bound, compression-list and buffer-reuse thresholds.
struct ChunkDef
{
    /// Upper bound on CHUNK_SIZE, a sanity bound for config validation, NOT the
    /// geometry (shrink_array::static_chunk_size). The hard limit is the per-chunk
    /// counter capacity (unsigned short, static_assert in conditional_counter.h).
    constexpr static int size_max = 65535;

    constexpr static int size_min_list_chunk_to_check_compress = 100; ///< Floor of the per-thread list threshold
    constexpr static int factor_div_nb_chunk_for_compress      = 100; ///< Threshold = chunks / (this * max threads)
    constexpr static int size_min_buffer_reuse                 = 100; ///< Floor of a per-size reuse budget
};

} // namespace shrink_array
