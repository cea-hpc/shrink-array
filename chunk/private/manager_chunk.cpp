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
#include <cstdlib> // std::getenv (CHUNK_BARRIER_FLUSH_LEGACY)

// The extern declarations of the three globals below (manager_chunk_inline.h,
// through the facade): without them the two `const bool` would get internal
// linkage here and the other TUs would not link.
#include "chunk/internal/manager_chunk.h"

namespace shrink_array
{

std::atomic<bool> __global_need_flush_deferred{true};

// CHUNK_BARRIER_FLUSH_LEGACY=1: the in-region flush toggles the global flag and
// flushes every type unconditionally, as before (A/B, bisection). Read once,
// during static initialisation, before any thread exists; never written after.
const bool g_barrier_flush_legacy = []
{
    const char* env = std::getenv("CHUNK_BARRIER_FLUSH_LEGACY");
    return env != nullptr && env[0] != '\0' && env[0] != '0';
}();

// CHUNK_SERIAL_DRAIN_LEGACY=1: the serial flush opens its drain regions even
// when no op is pending, as before (A/B, bisection). Same reading rules.
const bool g_serial_drain_legacy = []
{
    const char* env = std::getenv("CHUNK_SERIAL_DRAIN_LEGACY");
    return env != nullptr && env[0] != '\0' && env[0] != '0';
}();

// Template method bodies now live in chunk/manager_chunk_inline.h (included
// transitively via chunk/manager_chunk.h). This TU keeps only the non-template
// global state; the explicit instantiations are gone (Stage 5b): every
// ManagerChunk<CHUNK_T, T> auto-instantiates at its use site in the consumer TU
// (initialize<Types>()), linker-dedup'd.


} // namespace shrink_array
