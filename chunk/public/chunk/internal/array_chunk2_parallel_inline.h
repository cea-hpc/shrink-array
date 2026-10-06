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

// Out-of-line bodies of ArrayChunk2's mode-gated surfaces (ParallelSupport2,
// FastBlockSupport2). Included at the end of array_chunk2.h, after the class and
// its own inline file, so the CRTP self() down-cast sees a complete Derived and
// so the compression manager stays out of the container declaration itself.

#include "chunk/internal/array_chunk2_parallel.h"              // FastBlockSupport2, the class these bodies belong to
#include "chunk/internal/array_chunk_parallel_common_inline.h" // flush_pending_deferred_ops
#include "chunk/PMP_CHUNK.h"                                   // CHUNK_PMP_* (make_block_compression)
#include "chunk/internal/chunk_size_policy.h"                  // set_thread_chunk_size in make_block_compression
#include "chunk/config.h"                                      // build_config::chunk_dynamic

namespace shrink_array
{

template <typename Derived>
void FastBlockSupport2<Derived, true>::make_block_compression()
{
    // bug in run time openmp llvm

    if constexpr (build_config::chunk_dynamic)
    {
        // The compressibility scan reads the active size; make it THIS array's
        // size regardless of whatever size the calling thread last operated on.
        shrink_array::set_thread_chunk_size(self().chunk_size());
    }

    // bool has its own pass and no second value: the (thr, inf_array, proxy)
    // overload static_asserts for it, so ArrayChunk2<bool>::resize() did not
    // compile with fast-block compression.
    // [&], not [this]: `this` is used by the non-bool branch only, and an
    // explicit capture unused in the bool instantiation is a clang warning
    // (-Wunused-lambda-capture). [&] captures it where it is used.
    const auto pass_on_rank = [&](int thr)
    {
        if constexpr (Derived::is_not_bool)
        {
            chunk_type::make_block_compression(thr, self().get_info_second_value(), nullptr);
        }
        else
        {
            chunk_type::make_block_compression_bool(thr, nullptr);
        }
    };

    if (!CHUNK_PMP_IN_PARALLEL_REGION)
    {
        for (int thr = 0; thr < CHUNK_PMP_RANK_BOUND; ++thr)
        {
            pass_on_rank(thr);
        }
    }
    else
    {
        pass_on_rank(CHUNK_PMP_THREAD_RANK);
    }
}

} // namespace shrink_array
