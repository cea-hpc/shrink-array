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

// ============================================================================
// CHECK-BUILD INSTRUMENTATION — deferred ctx/op liveness sentinels (grep tag:
// DEFERRED_LIFECYCLE). Born of the deferred use-after-recycle hunt; kept as a
// permanent guard because the checking costs nothing in release (only the
// sentinel word stays in the layout, see arithmetic_context.h).
//
// Detects a deferred ctx / op being read after it has been destroyed and
// recycled by the per-thread pool (deferred_op_pool.h). The failure mode:
// a deferred write_body() lambda captures `ctx`, but by the time it runs the
// ctx has been free_ctx()'d and reused, so ctx->chunk references arbitrary
// memory and chunk.values_load() returns data-shaped garbage.
//
// Mechanism: a magic word stamped at construction, verified at the *top* of
// write_body() before ctx->chunk is touched, and poisoned at destruction.
// An `executed` flag additionally catches a double-flush (same op run twice).
//
// All checks compile away when build_config::any_check is false.
// ============================================================================

#include <cstdint> // std::uint64_t

namespace shrink_array::deferred_lifecycle
{

// Distinct, easily-recognisable sentinels (avoid all-zero / all-one so a
// zeroed or -1-filled block does not accidentally look alive).
inline constexpr std::uint64_t ctx_alive_magic = 0xC7A11'0000'A11EULL; // "CTX ALIVE"
inline constexpr std::uint64_t ctx_dead_magic  = 0xDEAD'C7A1'DEAD'C7A1ULL;
inline constexpr std::uint64_t op_alive_magic  = 0x0FA11'0000'A11EULL; // "OP  ALIVE"
inline constexpr std::uint64_t op_dead_magic   = 0xDEAD'0FA1'DEAD'0FA1ULL;

} // namespace shrink_array::deferred_lifecycle
