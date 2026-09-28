/*
 * Copyright 2024-2025 CEA (www.cea.fr)
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

/// @brief Compile-time settings of Chunk2A: CHUNK_SIZE bound, compression-list and buffer-reuse thresholds.
struct Chunk2ADef
{
    // Upper bound on CHUNK_SIZE. This is NOT the chunk geometry (use
    // shrink_array::__chunk_size for that); it is only a sanity bound for the
    // example/config validation. The real hard limit is the per-chunk counter
    // capacity (unsigned short) enforced by static_assert in
    // conditional_counter.h, so 65535 is the true maximum. (Was 256/128, a
    // stale value that predates the constexpr-geometry refactor and wrongly
    // rejected CHUNK_SIZE=512.)
    constexpr static int size_max = 65535;
    // --- Compression List Settings ---
    // The compression pass (make_block_compression -> start()) runs only once
    // the per-thread "to compress" list grows past szm/2, where szm derives
    // from these two constants. Raising the threshold amortizes start()'s
    // atomic CAS + cross-thread flush scan over many writes instead of firing
    // on nearly every write (the __aarch64_cas8_rel hotspot on short/ulong
    // arrays at high thread counts).
    //
    // This is driven by a DEDICATED knob, CHUNK2A_COMPRESS_LEVEL, NOT the global
    // SHRINK_LEVEL: the global one also drives chunk2's thresholds and buffer
    // reuse, so tuning chunk2a through it would move several unrelated
    // variables at once and make A/B comparisons meaningless.
    // Override at build time with -DCHUNK2A_COMPRESS_LEVEL=<0|1|2|3>:
    //   0 (default): original chunk2a values (factor 50 / min 500).
    //   1 : extreme stress — compress very often (tiny threshold).
    //   2 : intermediate — between original and level 3 (sweet-spot sweep).
    //   3 : throughput — compress rarely (large threshold). Removed the CAS but
    //       doubled the time in testing (writes then hit the costly
    //       enqueue_deferred_write path on still-uncompressed blocks).
#ifndef CHUNK2A_COMPRESS_LEVEL
    #define CHUNK2A_COMPRESS_LEVEL 0
#endif
    constexpr static int compress_level = CHUNK2A_COMPRESS_LEVEL;

    constexpr static int size_min_list_chunk_to_check_compress = (compress_level == 1)   ? 2
                                                                 : (compress_level == 2) ? 1000
                                                                 : (compress_level == 3) ? 2000
                                                                                         : 500;

    constexpr static int factor_div_nb_chunk_for_compress = (compress_level == 1)   ? 2000
                                                            : (compress_level == 2) ? 35
                                                            : (compress_level == 3) ? 20
                                                                                    : 50;

    constexpr static int nb_max_compress_value = 256;

    constexpr static int size_min_buffer_reuse                = 500;
    constexpr static int factor_div_nb_chunk_for_buffer_reuse = 100;

    constexpr static int factor_div_somme_chunk_buffer_resuse = 50;
};