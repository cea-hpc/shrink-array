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

// Body of flush_pending_deferred_ops(). Split from
// array_chunk_parallel_common.h so the declaration stays light enough to be
// included next to the container declarations, while the compression manager
// only reaches the three *_parallel_inline.h files.
#include "chunk/internal/array_chunk_parallel_common.h" // ParallelSupportCommon, the class these bodies belong to
#include "chunk/internal/compression_manager.h"         // CompressionManager<> in flush_pending_deferred_ops
#include "chunk/config.h"                   // build_config::any_check, chunk_config::use_fast_block_compression
#include "chunk/PMP_CHUNK.h"                // CHUNK_PMP_RANK_BOUND, CHUNK_PMP_IN_PARALLEL_REGION
#include "chunk/internal/invariant_check.h" // CHUNK_INVARIANT_FAILED (check builds)

namespace shrink_array
{

template <typename Derived>
void ParallelSupportCommon<Derived>::flush_pending_deferred_ops(bool force)
{
    if constexpr (chunk_config::use_fast_block_compression)
    {
        // Name CompressionManager<value_type, chunk_type> directly rather than via
        // chunk_type::CompressionManagerType: that alias is public on Chunk2 but
        // private on Chunk2A. Both name the same type; this form avoids the
        // access check on the alias.
        using CM = CompressionManager<value_type, chunk_type>;
        // force: drain every rank regardless of the pending counter. Used by
        // rechunk_to() before it swaps chunk_list out -- an op left on an
        // old-geometry chunk would otherwise outlive its chunk and be replayed
        // against a recycled address of the wrong size. The counter has been
        // seen to read 0 while ops were still queued, so it cannot gate this
        // path.
        //
        // ONE drain per rank, not one per chunk. flush_chunk_all_ranks() pays
        // its fixed cost -- announce_flush() (counter + seq_cst fence), the
        // hard-compressor wait, then every rank's shard lock and a find() --
        // BEFORE knowing whether the chunk holds anything: per chunk of the
        // array, that was a million announces and ~64 M lock round trips on a
        // million-chunk array at 64 threads, for chunks almost all empty (and
        // the dirty bits are sticky). flush_deferred(r) announces once and
        // walks rank r's chunk set, which holds exactly the chunks with ops
        // queued. It also drains the other arrays of this (T, chunk type):
        // their ops are applied earlier, still in order.
        if (force || CM::any_pending_ops())
        {
            const int max_thr = CHUNK_PMP_RANK_BOUND;
            for (int r = 0; r < max_thr; ++r)
            {
                static_cast<void>(CM::flush_deferred(r));
            }

            // Check builds: after a forced drain at a quiescent point (no
            // region, so no writer enqueues meanwhile), every rank's counter
            // must read 0 -- anything else is counter drift.
            if constexpr (build_config::any_check)
            {
                if (force && !CHUNK_PMP_IN_PARALLEL_REGION)
                {
                    for (int r = 0; r < max_thr; ++r)
                    {
                        if (const auto left = CM::pending_ops_rank(r); left != 0) [[unlikely]]
                        {
                            CHUNK_INVARIANT_FAILED("flush_pending_deferred_ops(force): rank {} still counts {} pending "
                                                   "op(s) after its drain, outside any region (counter drift)",
                                                   r, left);
                        }
                    }
                }
            }
        }
    }
}

} // namespace shrink_array
