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

// ============================================================================
// CHECK-BUILD INSTRUMENTATION — deferred ctx/op liveness sentinels (grep tag:
// DEFERRED_LIFECYCLE). Costs nothing in release (only the sentinel word stays in
// the layout, see arithmetic_context.h).
//
// Detects a deferred ctx / op read after it was destroyed and recycled by the
// per-thread pool (deferred_op_pool.h): a write_body() lambda whose captured `ctx`
// was free_ctx()'d and reused reads data-shaped garbage through ctx->chunk.
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
inline constexpr std::uint64_t ctx_alive_magic = 0xC7A11'0000'A11EULL;     ///< a live ctx ("CTX ALIVE")
inline constexpr std::uint64_t ctx_dead_magic  = 0xDEAD'C7A1'DEAD'C7A1ULL; ///< a destroyed ctx
inline constexpr std::uint64_t op_alive_magic  = 0x0FA11'0000'A11EULL;     ///< a live op ("OP  ALIVE")
inline constexpr std::uint64_t op_dead_magic   = 0xDEAD'0FA1'DEAD'0FA1ULL; ///< a destroyed op

} // namespace shrink_array::deferred_lifecycle
