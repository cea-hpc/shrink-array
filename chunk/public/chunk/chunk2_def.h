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

#include "chunk/config.h"
/**
 * @class Chunk2Def
 * @brief Configuration constants for buffer sizing and compression thresholds.
 * Thresholds are adjusted based on the chunk_config::tuning_level.
 */

class Chunk2Def
{
  public:
    // --- Compression List Settings ---
    // Level 1: Trigger compression every 2 modified chunks (Extreme stress)
    // Level 3: Chunk2A's floor (500, was 2000). At 64 threads the floor IS the
    // threshold, and 2000 kept up to 64 x 2000 materialised candidates alive
    // until a rank compressed -- 256 MiB of 2 KiB buffers, whose slabs stay
    // resident (variant_compare, --constants 1: 584 vs 455 MiB for Chunk2A).
    /// Floor of the per-thread "to compress" list threshold (entries), by chunk_config::tuning_level.
    constexpr static int size_min_list_chunk_to_check_compress = (chunk_config::tuning_level == 1)   ? 2
                                                                 : (chunk_config::tuning_level == 3) ? 500
                                                                                                     : 100;

    // Level 1: Divisor 2000 -> Threshold will be very low even for huge arrays
    /// List threshold = chunks created / (this * max threads), floored by size_min_list_chunk_to_check_compress.
    constexpr static int factor_div_nb_chunk_for_compress = (chunk_config::tuning_level == 1) ? 2000
                                                            : (chunk_config::tuning_level == 3)
                                                                ? 50 // Chunk2A's (was 20)
                                                                : 100;

  private:
    // Upper bound on CHUNK_SIZE. This is NOT the chunk geometry (use
    // shrink_array::__chunk_size for that); it is only a sanity bound for the
    // example/config validation. The real hard limit is the per-chunk counter
    // capacity (unsigned short) enforced by static_assert in
    // conditional_counter.h, so 65535 is the true maximum. (Was 256/128, a
    // stale value that predates the constexpr-geometry refactor and wrongly
    // rejected CHUNK_SIZE=512.)
    constexpr static int size_max = 65535;

    // --- Buffer Reuse Settings (Free lists) ---
    // Level 1: Keep only 4 blocks in cache to force real alloc/free cycles
    constexpr static int size_min_buffer_reuse = (chunk_config::tuning_level == 1)   ? 4
                                                 : (chunk_config::tuning_level == 3) ? 10000
                                                                                     : 500;

    // Level 1: Minimal recycling factor
    constexpr static int factor_div_nb_chunk_for_buffer_reuse = (chunk_config::tuning_level == 1)   ? 2000
                                                                : (chunk_config::tuning_level == 3) ? 20
                                                                                                    : 100;
};