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
// CHECK-BUILD INSTRUMENTATION — values-pointer sanity (grep tag:
// CHUNK_CHECK_VALUES_PTR). Free in release; turns a wild pointer into a named site
// instead of a SEGV.
//
// Checks pointers installed into chunk.values at the *producer* side (values_store /
// CAS / free-list pop / cst-getter), so the report names whoever fabricated the bad
// pointer, not the eventual reader in the deferred write_body.
//
// A valid values pointer is one of:
//   - nullptr (transient, tolerated by is_written checks)
//   - a cst_val1 / cst_val2(ind) constant block (aligned to alignof(T))
//   - a chunk buffer from allocate_chunk_internal (always ≥ 8-byte aligned,
//     and at an address well above the low pages)
//
// Anything below page 1 or misaligned for T is a debris pointer
// (torn read of a size/capacity int, wild free-list, etc.).
//
// All checks compile to nothing when build_config::any_check is false.
// ============================================================================

#include <cstdint>         // std::uintptr_t
#include <source_location> // std::source_location

#include "chunk/config.h"                   // build_config::any_check
#include "chunk/internal/invariant_check.h" // invariant_failed
#include "chunk/PMP_CHUNK.h"                // get_thr_rank_fast, CHUNK_PMP_*, and <omp.h> with them

namespace shrink_array::bad_ptr
{

/**
 * @brief Validate that @p ptr is either null or a plausible values pointer.
 *
 * Compiles to nothing when build_config::any_check is false.
 * The source_location captured by CHUNK_INVARIANT_FAILED points at the caller,
 * because @p loc is defaulted to std::source_location::current() at the
 * call site, not inside this function.
 *
 * @param ptr        Pointer to validate (may be null).
 * @param site       Short static string identifying the producer site.
 * @param chunk_ptr  Optional pointer to the owning chunk, for cross-referencing.
 * @param loc        Auto-captured at the call site; do not pass explicitly.
 */
template <typename T>
[[gnu::always_inline]] inline void check(const T*             ptr,
                                         const char*          site,
                                         const void*          chunk_ptr = nullptr,
                                         std::source_location loc       = std::source_location::current()) noexcept
{
    if constexpr (build_config::any_check)
    {
        const auto addr = reinterpret_cast<std::uintptr_t>(ptr);

        // nullptr is tolerated — is_written checks handle it explicitly.
        if (ptr == nullptr)
        {
            return;
        }

        // Valid: address above the low pages AND aligned to alignof(T). Do not hard-code 8:
        // the cst_val1/cst_val2 sentinels are static std::array<T> whose .data() is only
        // alignof(T)-aligned; the linker usually over-aligns statics, but not always
        // (e.g. under sanitizer redzones).
        constexpr std::uintptr_t align_mask = alignof(T) - 1;
        if (addr >= 0x1000 && (addr & align_mask) == 0) [[likely]]
        {
            return;
        }

        // Deliberately the RAW omp rank as well as get_thr_rank_fast(): printing both shows
        // a divergence between OpenMP's numbering and the library's cached one. -1 means
        // "outside any region".
        const int omp_tid   = CHUNK_PMP_IN_PARALLEL_REGION ? CHUNK_PMP_THREAD_RANK : -1;
        const int fast_rank = get_thr_rank_fast();

        ::shrink_array::invariant_failed(loc, "[BAD_PTR] site='{}' ptr={} chunk={} omp_tid={} fast_rank={}", site,
                                         static_cast<const void*>(ptr), chunk_ptr, omp_tid, fast_rank);
    }
}

} // namespace shrink_array::bad_ptr

// ─── Call-site macros ────────────────────────────────────────────────────────
// Kept as macros for grep-ability: `grep -rl CHUNK_CHECK_VALUES_PTR chunk/`
// lists every instrumentation site.

#define CHUNK_CHECK_VALUES_PTR(p, site) ::shrink_array::bad_ptr::check((p), (site))
#define CHUNK_CHECK_VALUES_PTR_C(p, site, chunk_ptr) ::shrink_array::bad_ptr::check((p), (site), (chunk_ptr))
