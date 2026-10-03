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
 * @file array_chunk2A_rechunk_hooks.h
 * @brief ArrayChunk2A's re-chunk hooks — a CHUNK_DYNAMIC-only concern, isolated
 *        in its own mixin so they vanish from a static-geometry build.
 *
 * @details Same split as RechunkHooks / RechunkHooks2 do for the other two
 *          variants. 2A overrides only three of the six: its second values are
 *          per-chunk and rebuilt by the write path, so capture/seed/reown keep
 *          RechunkSupport's no-op defaults. All three only ever run under
 *          RechunkSupport<Derived, true>::rechunk_to(), which does not exist in
 *          a static-geometry build.
 *
 * @see array_chunk_rechunk.h for the algorithm that calls these
 * @see array_chunk2A_rechunk_hooks_inline.h for the bodies
 */

#include "chunk/config.h"              // build_config::chunk_dynamic
#include "chunk/chunk_traits.h"        // ArrayChunkTraits
#include "chunk/internal/crp_helper.h" // crtp_cast_t

namespace shrink_array
{

/**
 * @brief ArrayChunk2A's re-chunk hooks, present only when the geometry is a runtime value.
 *
 * @tparam Derived the concrete array type (only ArrayChunk2A today).
 * @tparam Dynamic defaulted to build_config::chunk_dynamic; never pass it
 *         explicitly outside of tests.
 */
template <typename Derived, bool Dynamic = build_config::chunk_dynamic>
class RechunkHooks2A;

// ── CHUNK_DYNAMIC build: the real hooks ──────────────────────────────────────
/**
 * @brief Dynamic specialisation: the three hooks 2A overrides.
 * @copydetails RechunkHooks2A
 */
template <typename Derived>
class RechunkHooks2A<Derived, true>
{
  public:
    using Traits       = ArrayChunkTraits<Derived>; ///< Traits specialisation of the concrete array.
    using proxy_type   = Traits::proxy_type;        ///< Proxy type owning the per-type array registry.
    using manager_type = Traits::manager_type;      ///< Per-type manager owning the candidate store.

    /// @brief Drop this array's queued compression candidates before teardown.
    void rechunk_drop_candidates();

    /**
     * @brief Release this array's parked buffers held at the old geometry.
     * @param new_size geometry the array is moving to
     */
    void rechunk_drain_reuse_list(unsigned int new_size);

    /**
     * @brief Exchange identity with the scratch array built at the new geometry.
     * @param tmp freshly built array whose storage this array takes over
     * @return whatever the registry swap reports
     * @note Unlike swap(), this DOES move num_tab, updating the registry so each
     *       address keeps pointing at its own metadata.
     */
    bool rechunk_swap_identity(Derived& tmp);

  protected:
    /** @brief CRTP down-cast: the concrete array this mixin is a base of,
     *         const on a const object (one member for both: deducing this). */
    template <typename Self>
    [[nodiscard]] auto& self(this Self&& obj) noexcept
    {
        return static_cast<crtp_cast_t<Self, Derived>&>(obj);
    }
};

// ── Static-geometry build: nothing to hook ───────────────────────────────────
/**
 * @brief Static specialisation: the same three signatures, inert.
 *
 * @details Never called — RechunkSupport<Derived, false>::rechunk_to() does
 *          nothing — but kept so ArrayChunk2A can pull the names in with the
 *          same using-declarations in both modes.
 */
template <typename Derived>
class RechunkHooks2A<Derived, false>
{
  public:
    /** @brief No-op: the geometry is fixed at compile time. */
    void rechunk_drop_candidates() noexcept {}
    /** @brief No-op: the geometry is fixed at compile time. */
    void rechunk_drain_reuse_list(unsigned int /*new_size*/) noexcept {}
    /** @brief No-op: there is no temporary array to swap identities with. */
    bool rechunk_swap_identity(Derived& /*tmp*/) noexcept
    {
        return true;
    }
};

} // namespace shrink_array
