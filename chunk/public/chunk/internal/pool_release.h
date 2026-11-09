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

/**
 * @file pool_release.h
 * @brief The two shutdown calls of a program: release_chunk_pools() and
 *        destroy_block_chunk(), defined in chunk/private/chunk_std.cpp.
 *
 * @details Declarations only, with no include but SHRINK_ARRAY_EXPORT's: the
 *          shrink_array module exports them (chunk/modules/shrink_array.cppm),
 *          chunk/param_chunk.h (the public header a program includes for its
 *          start and its end) and chunk/internal/zero_chunk.h (the library's
 *          own callers) take them from here.
 */

#include "chunk/internal/module_export.h" // SHRINK_ARRAY_EXPORT

namespace shrink_array
{

/// @brief When fast allocation is enabled, call destroy_block_chunk() on the AllTypeChunk singleton; otherwise
///        do nothing. Nothing either while an array is alive (see release_chunk_pools()).
SHRINK_ARRAY_EXPORT extern void destroy_block_chunk();

/**
 * @brief Let go of the chunk reuse pools, whichever thread model this build is.
 *
 * @details OpenMP releases the per-rank lists (clear_all_list_block_free_thr()
 *          then delete_list_block_chunk_free_itself()); a serial build deletes
 *          its single allocator (delete_chunk_allocator()).
 * @note Pair it with destroy_block_chunk(). Both are optional: the process can
 *       simply exit.
 * @note While an array is alive (typically a global one, destroyed after main)
 *       this only empties the free lists, and destroy_block_chunk(),
 *       delete_list_block_chunk_free_itself() and delete_chunk_allocator() do
 *       nothing: the array parks its buffers there when it dies.
 * @warning Sequential point: call it once, after the last parallel region. From
 *          inside a region (OpenMP's, the std backend's team or application
 *          phase) a check build aborts; any other build says so on stderr and
 *          does nothing. Every thread of the team would otherwise empty the
 *          same pools at once. The same holds for destroy_block_chunk(),
 *          delete_chunk_allocator(), clear_all_list_block_free_thr(),
 *          delete_list_block_chunk_free_itself() and an array's shrink_to_fit().
 */
SHRINK_ARRAY_EXPORT extern void release_chunk_pools();

} // namespace shrink_array
