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

namespace shrink_array
{

namespace zero_any
{

/// @brief Compile-time settings of zero_any: compression-list and buffer-reuse thresholds.
class ChunkDef
{
  private:
    // --- Compression List Settings ---
    // The compression pass (make_block_compression -> start()) runs only once the
    // per-thread "to compress" list grows past szm/2, szm derived from the two
    // constants below. A higher threshold amortizes start()'s atomic CAS and
    // cross-thread flush scan over many writes.
    //
    // Driven by its own knob, CHUNK2A_COMPRESS_LEVEL, not CHUNK_TUNING_LEVEL: the
    // global one also moves zero_one_value's thresholds and buffer reuse, which would make
    // A/B comparisons of zero_any meaningless. -DCHUNK2A_COMPRESS_LEVEL=<0|1|2|3>:
    //   0 (default): factor 50 / min 500.
    //   1 : stress -- compress very often (tiny threshold).
    //   2 : intermediate between 0 and 3.
    //   3 : compress rarely. Slower overall: writes then take the costly
    //       enqueue_deferred_write path on still-uncompressed blocks.
#ifndef CHUNK2A_COMPRESS_LEVEL
    #define CHUNK2A_COMPRESS_LEVEL 0
#endif

    constexpr static int compress_level = CHUNK2A_COMPRESS_LEVEL;

  public:
    /// Floor of the per-thread "to compress" list threshold (entries), by CHUNK2A_COMPRESS_LEVEL.
    constexpr static int size_min_list_chunk_to_check_compress = []
    {
        switch (compress_level)
        {
            case 1:
                return 2;
            case 2:
                return 1000;
            case 3:
                return 2000;
            default:
                return 500;
        }
    }();

    /// List threshold = chunks created / (this * max threads), floored by size_min_list_chunk_to_check_compress.
    constexpr static int factor_div_nb_chunk_for_compress = []
    {
        switch (compress_level)
        {
            case 1:
                return 2000;
            case 2:
                return 35;
            case 3:
                return 20;
            default:
                return 50;
        }
    }();

    /// Floor of the global (unsized) free-list reuse budget, in buffers per thread.
    constexpr static int size_min_buffer_reuse = 500;

    /// Reuse budget per thread = chunks / (this * max threads); used for every variant, global and per size.
    constexpr static int factor_div_somme_chunk_buffer_resuse = 50;
};

} // namespace zero_any

} // namespace shrink_array
