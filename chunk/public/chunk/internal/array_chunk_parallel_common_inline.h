#pragma once

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

// Body of flush_pending_deferred_ops(). Split from
// array_chunk_parallel_common.h so the declaration stays light enough to be
// included next to the container declarations, while the compression manager
// only reaches the three *_parallel_inline.h files.
#include "chunk/internal/array_chunk_parallel_common.h"
#include "chunk/internal/compression_manager.h" // CompressionManager<> in flush_pending_deferred_ops

namespace shrink_array
{

template <typename Derived>
void ParallelSupportCommon<Derived>::flush_pending_deferred_ops(bool force)
{
    if constexpr (chunk_config::use_fast_block_compression)
    {
        Derived& arr = self();

        // Name CompressionManager<value_type, chunk_type> directly rather than via
        // chunk_type::CompressionManagerType: that alias is public on Chunk2 but
        // private on Chunk2A. Both name the same type; this form avoids the
        // access check on the alias.
        using CM = CompressionManager<value_type, chunk_type>;
        // force: flush every chunk regardless of the pending counter and dirty
        // bits. Used by rechunk_to() before it swaps chunk_list out -- an op
        // left on an old-geometry chunk would otherwise outlive its chunk and be
        // replayed against a recycled address of the wrong size. The counter has
        // been seen to read 0 while ops were still queued, so it cannot gate
        // this path.
        if (force)
        {
            for (int j = 0; j < arr.nb_chunk; ++j)
                static_cast<void>(CM::flush_chunk_all_ranks(&arr.chunk_list[j]));
        }
        else if (CM::any_pending_ops())
        {
            for (int j = 0; j < arr.nb_chunk; ++j)
            {
                if (arr.is_chunk_dirty(j))
                    static_cast<void>(CM::flush_chunk_all_ranks(&arr.chunk_list[j]));
            }
        }
    }
}

} // namespace shrink_array
