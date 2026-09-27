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

/**
 * @file array_chunk2A_parallel.h
 * @brief The OpenMP-only surface of ArrayChunk2A -- the deferred-op flush --
 *        isolated in its own mixin so it disappears entirely from a serial
 *        build.
 *
 * @details Same split as ParallelSupport / ParallelSupport2 do for the other
 *          two variants: ParallelSupport2A<Derived, use_openmp> carries
 *          flush_pending_deferred_ops() (ParallelSupportCommon, shared by the
 *          three variants), a no-op in the serial specialisation.
 *
 *          The mixin also held clear_in_par(), an in-parallel teardown nothing
 *          called, with its per-thread "async reset" registries; both are gone.
 *
 * @see array_chunk_parallel.h, array_chunk2_parallel.h for the other two variants
 */

#include "chunk/config.h"
#include "chunk/internal/array_chunk_parallel_common.h"

namespace shrink_array
{

/**
 * @brief The OpenMP-only members of ArrayChunk2A, present only when the build has OpenMP.
 *
 * @tparam Derived the concrete array type (only ArrayChunk2A today).
 * @tparam Enabled defaulted to build_config::use_openmp; never pass it
 *         explicitly outside of tests.
 */
template <typename Derived, bool Enabled = build_config::use_openmp>
class ParallelSupport2A;

// ── OpenMP build: the real parallel surface ──────────────────────────────────
/**
 * @brief OpenMP specialisation: the deferred-op flush.
 * @copydetails ParallelSupport2A
 */
template <typename Derived>
class ParallelSupport2A<Derived, true> : public ParallelSupportCommon<Derived>
{
    // flush_pending_deferred_ops() and self() come from ParallelSupportCommon.
};

// ── Serial build: nothing to parallelise ─────────────────────────────────────
/**
 * @brief Serial specialisation: the same verb, inert.
 */
template <typename Derived>
class ParallelSupport2A<Derived, false> : public ParallelSupportCommonSerial<Derived>
{
};

} // namespace shrink_array
