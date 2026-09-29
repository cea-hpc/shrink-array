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

#include <atomic>  // std::atomic (__global_need_flush_deferred)

// The extern declaration of the global below (manager_chunk_inline.h, through
// the facade).
#include "chunk/internal/manager_chunk.h"

namespace shrink_array
{

std::atomic<bool> __global_need_flush_deferred{true};

// Template method bodies now live in chunk/manager_chunk_inline.h (included
// transitively via chunk/manager_chunk.h). This TU keeps only the non-template
// global state; the explicit instantiations are gone (Stage 5b): every
// ManagerChunk<CHUNK_T, T> auto-instantiates at its use site in the consumer TU
// (initialize<Types>()), linker-dedup'd.


} // namespace shrink_array
