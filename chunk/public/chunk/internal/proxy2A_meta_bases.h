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

/**
 * @file proxy2A_meta_bases.h
 * @brief Proxy2AMetaBases: one ArrayChunk2A's rows in the per-tab metadata
 *        tables. Its own header: both proxy2A.h and array_chunk2A.h need it.
 */

#pragma once

#include <atomic>

#include "chunk/internal/tab_cst_val2_holder.h" // SecondValueRef

namespace shrink_array
{

/**
 * @brief Base addresses of one ArrayChunk2A's rows in the three per-tab
 *        metadata tables (type-of-constant, "queued" flag, second value).
 * @details Kept by the array (ArrayChunk2A::meta_bases), re-read from the
 *          tables after every reallocation of its rows, and copied into each
 *          write proxy by operator[] so a chunk's metadata address is
 *          base + num_chunk. type == nullptr means "not armed".
 */
template <typename T>
struct Proxy2AMetaBases
{
    char*                               type   = nullptr;
    std::atomic<char>*                  inside = nullptr;
    typename SecondValueRef<T>::slot_t* sv     = nullptr; ///< unused for bool
};

} // namespace shrink_array
