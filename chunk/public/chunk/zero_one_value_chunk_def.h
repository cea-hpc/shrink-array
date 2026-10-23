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

#include "chunk/config.h" // chunk_config::tuning_level
namespace shrink_array
{

namespace zero_one_value
{

/**
 * @brief Compile-time settings of zero_one_value: the compression-list thresholds,
 *        adjusted by chunk_config::tuning_level.
 */
struct ChunkDef
{
    // --- Compression List Settings ---
    // Level 1: Trigger compression every 2 modified chunks (Extreme stress)
    // Level 3: zero_any's floor. At high thread counts the floor IS the threshold: a higher
    // one keeps threads x floor materialised candidates alive until a rank compresses, and
    // their slabs stay resident (a few hundred MiB more RSS).
    /// Floor of the per-thread "to compress" list threshold (entries), by chunk_config::tuning_level.
    constexpr static int size_min_list_chunk_to_check_compress = []
    {
        switch (chunk_config::tuning_level)
        {
            case 1:
                return 2;
            case 3:
                return 500;
            default:
                return 100;
        }
    }();

    // Level 1: Divisor 2000 -> Threshold will be very low even for huge arrays
    /// List threshold = chunks created / (this * max threads), floored by size_min_list_chunk_to_check_compress.
    constexpr static int factor_div_nb_chunk_for_compress = []
    {
        switch (chunk_config::tuning_level)
        {
            case 1:
                return 2000;
            case 3:
                return 50; // zero_any's
            default:
                return 100;
        }
    }();
};

} // namespace zero_one_value

} // namespace shrink_array
