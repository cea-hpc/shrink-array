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
 * @file array_chunk2_rechunk_hooks.h
 * @brief ArrayChunk2's re-chunk hooks — a CHUNK_DYNAMIC-only concern, isolated
 *        in its own mixin so they vanish from a static-geometry build.
 *
 * @details Same split as RechunkHooks does for ArrayChunk (v1), except that
 *          Chunk2 overrides all SIX hooks where v1 needs three: it resolves
 *          num_tab from the registry rather than from a member, carries the
 *          array-level second value across the swap, and re-owns its
 *          map_inf_array entry afterwards. All six only ever run under
 *          RechunkSupport<Derived, true>::rechunk_to(), which does not exist in
 *          a static-geometry build.
 *
 *          RechunkHooks2<Derived, chunk_dynamic> splits them the same way as the
 *          algorithm they serve: the dynamic specialisation carries the real
 *          bodies (unchanged), the static one keeps the signatures as no-ops so
 *          ArrayChunk2's using-declarations stay valid in both modes.
 *
 * @see array_chunk_rechunk.h for the algorithm that calls these
 * @see array_chunk_rechunk_hooks.h for v1's three
 * @see array_chunk2_rechunk_hooks_inline.h for the bodies
 */

#include "chunk/config.h"        // build_config::chunk_dynamic
#include "chunk/chunk_traits.h"  // ArrayChunkTraits
#include "chunk/internal/in_t.h" // in_t<T>: the captured state

namespace shrink_array
{

/**
 * @brief ArrayChunk2's re-chunk hooks, present only when the geometry is a runtime value.
 *
 * @tparam Derived the concrete array type (only ArrayChunk2 today).
 * @tparam Dynamic defaulted to build_config::chunk_dynamic; never pass it
 *         explicitly outside of tests.
 */
template <typename Derived, bool Dynamic = build_config::chunk_dynamic>
class RechunkHooks2;

// ── CHUNK_DYNAMIC build: the real hooks ──────────────────────────────────────
/**
 * @brief Dynamic specialisation: the six hooks Chunk2 overrides.
 * @copydetails RechunkHooks2
 */
template <typename Derived>
class RechunkHooks2<Derived, true>
{
  public:
    using Traits       = ArrayChunkTraits<Derived>; ///< Traits specialisation of the concrete array.
    using value_type   = Traits::value_type;        ///< Stored element type.
    using proxy_type   = Traits::proxy_type;        ///< Proxy type owning the per-type array registry.
    using manager_type = Traits::manager_type;      ///< Per-type manager owning the candidate store.

    /// @brief Drop this array's queued compression candidates before teardown.
    void rechunk_drop_candidates();

    /**
     * @brief Release this array's parked buffers held at the old geometry.
     * @param new_size geometry the array is moving to
     */
    void rechunk_drain_reuse_list(unsigned int new_size);

    /// @brief Capture the array-level state that must survive the rechunk.
    /// @return this array's second value
    value_type rechunk_capture_state() const;

    /**
     * @brief Re-declare the captured state on the freshly built array.
     * @param state value returned by rechunk_capture_state()
     */
    void rechunk_seed_fresh(in_t<value_type> state);

    /**
     * @brief Exchange identity with the scratch array built at the new geometry.
     * @param tmp freshly built array whose storage this array takes over
     * @return true if the registry entries were exchanged
     */
    bool rechunk_swap_identity(Derived& tmp);

    /**
     * @brief Re-own the map_inf_array entry after the identity swap.
     * @param tmp   scratch array the storage came from
     * @param state value returned by rechunk_capture_state()
     * @details Re-points the shared entry at the SURVIVOR's inf_array member, so
     *          that tmp's destructor does not erase an entry the survivor still
     *          needs -- see ArrayChunk2's destructor, second warning.
     */
    void rechunk_reown_after_swap(Derived& tmp, in_t<value_type> state);

  protected:
    /** @brief CRTP down-cast: the concrete array this mixin is a base of. */
    [[nodiscard]] Derived& self() noexcept
    {
        return static_cast<Derived&>(*this);
    }
    /** @copydoc self() */
    [[nodiscard]] const Derived& self() const noexcept
    {
        return static_cast<const Derived&>(*this);
    }
};

// ── Static-geometry build: nothing to hook ───────────────────────────────────
/**
 * @brief Static specialisation: the same six signatures, inert.
 *
 * @details Never called — RechunkSupport<Derived, false>::rechunk_to() does
 *          nothing — but kept so ArrayChunk2 can pull the names in with the same
 *          using-declarations in both modes.
 */
template <typename Derived>
class RechunkHooks2<Derived, false>
{
  public:
    using Traits     = ArrayChunkTraits<Derived>; ///< Traits specialisation of the concrete array.
    using value_type = Traits::value_type;        ///< Stored element type.

    /** @brief No-op: the geometry is fixed at compile time. */
    void rechunk_drop_candidates() noexcept {}
    /** @brief No-op: the geometry is fixed at compile time. */
    void rechunk_drain_reuse_list(unsigned int /*new_size*/) noexcept {}
    /** @brief No state to carry: there is no re-chunk to carry it across. */
    value_type rechunk_capture_state() const
    {
        return value_type{};
    }
    /** @brief No-op: there is no fresh array to seed. */
    void rechunk_seed_fresh(in_t<value_type> /*state*/) noexcept {}
    /** @brief No-op: there is no temporary array to swap identities with. */
    bool rechunk_swap_identity(Derived& /*tmp*/) noexcept
    {
        return true;
    }
    /** @brief No-op: no identity was swapped, so nothing to re-own. */
    void rechunk_reown_after_swap(Derived& /*tmp*/, in_t<value_type> /*state*/) noexcept {}
};

} // namespace shrink_array
