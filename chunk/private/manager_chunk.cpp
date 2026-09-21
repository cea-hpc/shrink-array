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

// #include "chunk/PMP_CHUNK.h"
// #include <cstdio> // FIX: fprintf for the drain safety-bound warning
// #include "chunk/internal/invariant_check.h"
// #include "chunk/chunk_std.h"
// #include "chunk/chunk2_std.h"
#include "chunk/chunk2A_std.h"
// #include "chunk/array_chunk.h"
// #include "chunk/array_chunk2.h"
// #include "chunk/internal/concept_chunk.h"
// #include "chunk/global_stats_chunk_list.h"
// #include "chunk/internal/chunk_func.h"

//#include "chunk/type_name.h"
#include <cstdlib> // std::getenv (CHUNK_BARRIER_FLUSH_LEGACY)

#include "chunk/internal/manager_chunk.h"
//#include "chunk/singleton.h"
//#include "chunk/global_stats_chunk_list.h"

namespace shrink_array
{

std::atomic<bool> __global_need_flush_deferred{true};

// CHUNK_BARRIER_FLUSH_LEGACY=1: the in-region flush toggles the global flag and
// flushes every type unconditionally, as before (A/B, bisection). Read once,
// during static initialisation, before any thread exists; never written after.
const bool g_barrier_flush_legacy = []
{
    const char* e = std::getenv("CHUNK_BARRIER_FLUSH_LEGACY");
    return e != nullptr && e[0] != '\0' && e[0] != '0';
}();

// Template method bodies now live in chunk/manager_chunk_inline.h (included
// transitively via chunk/manager_chunk.h). This TU keeps only the non-template
// global state and the explicit instantiations (Stage 2: instantiations are
// now removed (Stage 5b): every ManagerChunk<CHUNK_T, T> auto-instantiates at
// its use site in the consumer TU (initialize<Types>()), linker-dedup'd.


} // namespace shrink_array
