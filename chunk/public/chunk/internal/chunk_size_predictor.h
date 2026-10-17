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

// ============================================================================
// Chunk-size predictor (observation phase — log only, no re-chunk)
// ============================================================================
//
// For a given array, estimate the memory it WOULD occupy at chunk sizes larger
// than the current one, to see whether a bigger chunk would compress better
// (fewer materialised buffers, fewer metadata records).
//
// Nothing is re-chunked. The owning array collects a per-chunk summary and
// calls report_size_prediction with it, which logs current vs optimal.
//
// Method (sizes >= current only, cheap):
//   1. The array summarises each chunk as {is_constant, value}. A chunk is
//      constant if compressed (!written()) or if all its elements are equal.
//   2. For k in {1,2,4,8,...} (S' = k * S), a simulated block of k adjacent
//      chunks is constant iff all k chunks are constant AND share one value
//      (chunk_list is in logical order: chunk j covers [j*S,(j+1)*S)). The last
//      block may be partial.
//   3. Predicted memory = (non-constant blocks) * S' * sizeof(T)   [buffers]
//                       + (blocks) * meta_per_block.               [records]
//      Constant blocks cost only metadata (values live in the shared cst
//      buffer). The buffer term dominates and is what varies with S'.
//
// Sizes < current are out of scope here (need intra-chunk re-reading, usually
// lose on metadata). Gated by the caller on any_check and on the log
// (advisor=info) -- never in production.

#include <cstddef>  // std::size_t
#include <optional> // std::optional
#include <utility>  // std::pair
#include <vector>   // std::vector

#include "chunk/log.h" // CHUNK_LOG

namespace shrink_array
{

/// @brief Predicted figures at chunk size S_prime: memory, constant blocks and total blocks.
struct PredictorPoint
{
    unsigned    S_prime;
    std::size_t mem_bytes;
    int         n_const_blocks;
    int         n_blocks;
};

// Pure aggregation — unit-testable without any array.
template <typename T>
[[nodiscard]] PredictorPoint
predict_at_multiple(const std::vector<std::pair<bool, T>>& summary, unsigned S, unsigned k, std::size_t meta_per_block)
{
    const unsigned Sp       = S * k;
    const int      nchunks  = static_cast<int>(summary.size());
    int            n_blocks = 0;
    int            n_const  = 0;

    for (int base = 0; base < nchunks; base += static_cast<int>(k))
    {
        ++n_blocks;
        bool             all_const = true;
        std::optional<T> ref;
        for (unsigned j = 0; j < k; ++j)
        {
            const int idx = base + static_cast<int>(j);
            if (idx >= nchunks)
            {
                break; // partial trailing block
            }
            if (!summary[idx].first)
            {
                all_const = false;
                break;
            }
            if (!ref)
            {
                ref = summary[idx].second;
            }
            else if (*ref != summary[idx].second)
            {
                all_const = false;
                break;
            }
        }
        if (all_const)
        {
            ++n_const;
        }
    }

    const std::size_t mem = (static_cast<std::size_t>(n_blocks - n_const) * Sp * sizeof(T)) +
                            (static_cast<std::size_t>(n_blocks) * meta_per_block);
    return {.S_prime = Sp, .mem_bytes = mem, .n_const_blocks = n_const, .n_blocks = n_blocks};
}

// Predict memory for S' in {S,2S,4S,...} from a summary the array collected,
// and log current vs predicted-optimal. Log only — no re-chunk.
template <typename T>
void report_size_prediction(const std::vector<std::pair<bool, T>>& summary,
                            unsigned                               S,
                            int                                    num_tab,
                            const char*                            variant,
                            std::size_t                            meta_per_block,
                            unsigned                               max_multiple = 16)
{
    const int nb_chunk = static_cast<int>(summary.size());
    if (nb_chunk <= 0)
    {
        return;
    }

    PredictorPoint best{.S_prime = S, .mem_bytes = ~std::size_t{0}, .n_const_blocks = 0, .n_blocks = 0};
    PredictorPoint current{};
    bool           have_current = false;

    for (unsigned k = 1; k <= max_multiple; k *= 2)
    {
        if (std::cmp_greater(k, nb_chunk) && k != 1)
        {
            break;
        }
        const PredictorPoint point = predict_at_multiple<T>(summary, S, k, meta_per_block);
        if (k == 1)
        {
            current      = point;
            have_current = true;
        }
        if (point.mem_bytes < best.mem_bytes)
        {
            best = point;
        }
    }

    if (!have_current)
    {
        return;
    }

    if (best.S_prime != current.S_prime && best.mem_bytes < current.mem_bytes)
    {
        const double gain_pct =
            100.0 * static_cast<double>(current.mem_bytes - best.mem_bytes) / static_cast<double>(current.mem_bytes);
        if (gain_pct >= 10.0)
        {
            CHUNK_LOG(info, advisor,
                      "size prediction {} tab={}: current S={} -> {} B ; optimal S'={} -> {} B (mem -{:.1f}%, {}/{} "
                      "blocks constant)",
                      variant, num_tab, S, current.mem_bytes, best.S_prime, best.mem_bytes, gain_pct,
                      best.n_const_blocks, best.n_blocks);
        }
    }
}

} // namespace shrink_array
