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

#include <atomic> // std::atomic (g_need_flush_deferred)

#include "chunk/internal/manager_chunk.h" // extern g_need_flush_deferred (manager_chunk_inline.h)

namespace shrink_array
{

std::atomic<bool> g_need_flush_deferred{true};

// This TU holds only the non-template global state. Method bodies are in
// chunk/internal/manager_chunk_inline.h; there is no explicit instantiation: each
// ManagerChunk<CHUNK_T, T> is instantiated in the consumer TU (initialize<Types>()).


} // namespace shrink_array
