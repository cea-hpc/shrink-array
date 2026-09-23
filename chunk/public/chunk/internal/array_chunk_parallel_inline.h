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

// Out-of-line bodies of ArrayChunk's mode-gated surfaces (ParallelSupport,
// FastBlockSupport). Included at the end of array_chunk.h, after the class and
// its own inline file, so the CRTP self() down-cast sees a complete Derived and
// so the compression manager stays out of the container declaration itself.
#include "chunk/internal/array_chunk_parallel.h"
#include "chunk/internal/array_chunk_parallel_common_inline.h" // flush_pending_deferred_ops
#include "chunk/PMP_CHUNK.h"                                   // CHUNK_PMP_* (make_block_compression)
#include "chunk/internal/chunk_size_policy.h"                  // set_thread_chunk_size (make_block_compression)

namespace shrink_array
{

template <typename Derived>
void FastBlockSupport<Derived, true>::make_block_compression()
{
    // bug in run time openmp llvm

    if constexpr (build_config::chunk_dynamic)
    {
        // The compressibility scan reads the active size; make it THIS array's
        // size regardless of whatever size the calling thread last operated on.
        shrink_array::set_thread_chunk_size(self().chunk_size());
    }

    if (!CHUNK_PMP_IN_PARALLEL_REGION)
    {
        for (int thr = 0; thr < CHUNK_PMP_MAX_THREADS; ++thr)
        {
            chunk_type::make_block_compression(thr, nullptr);
        }
    }
    else
    {
        chunk_type::make_block_compression(CHUNK_PMP_THREAD_RANK, nullptr);
    }
}

} // namespace shrink_array
