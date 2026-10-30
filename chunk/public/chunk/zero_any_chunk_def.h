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

namespace shrink_array::zero_any
{

/// @brief Compile-time settings of zero_any: compression-list and buffer-reuse thresholds.
class ChunkDef
{
  public:
    // --- Compression List Settings ---
    // The compression pass (make_block_compression -> start()) runs only once the
    // per-thread "to compress" list grows past szm/2, szm derived from the two
    // constants below. A higher threshold amortizes start()'s atomic CAS and
    // cross-thread flush scan over many writes.
    //
    // A single set of thresholds: the levels 1 to 3 of the former
    // CHUNK2A_COMPRESS_LEVEL macro (stress, intermediate, rare) were never
    // built or tested.

    /// Floor of the per-thread "to compress" list threshold (entries).
    constexpr static int size_min_list_chunk_to_check_compress = 500;

    /// List threshold = chunks created / (this * max threads), floored by size_min_list_chunk_to_check_compress.
    constexpr static int factor_div_nb_chunk_for_compress = 50;

    /// Floor of the global (unsized) free-list reuse budget, in buffers per thread.
    constexpr static int size_min_buffer_reuse = 500;

    /// Reuse budget per thread = chunks / (this * max threads); used for every variant, global and per size.
    constexpr static int factor_div_somme_chunk_buffer_reuse = 50;

    /// Former, misspelt name of factor_div_somme_chunk_buffer_reuse, until 1.0.
    [[deprecated(
        "use factor_div_somme_chunk_buffer_reuse")]] constexpr static int factor_div_somme_chunk_buffer_resuse =
        factor_div_somme_chunk_buffer_reuse;
};

} // namespace shrink_array::zero_any
