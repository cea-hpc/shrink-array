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

/**
 * @file zero_one_value_array_rechunk_hooks.h
 * @brief zero_one_value::Array's re-chunk hooks — a CHUNK_DYNAMIC-only concern, isolated
 *        in its own mixin so they vanish from a static-geometry build.
 *
 * @details Like RechunkHooks for zero, but zero_one_value overrides all SIX hooks: it
 *          resolves num_tab from the registry, carries the array-level second
 *          value across the swap, and re-owns its map_inf_array entry. They run
 *          only under RechunkSupport<Derived, true>::rechunk_to(); the static
 *          specialisation keeps the signatures as no-ops so zero_one_value::Array's
 *          using-declarations stay valid in both modes.
 *
 * @see array_chunk_rechunk.h for the algorithm that calls these
 * @see array_chunk_rechunk_hooks.h for zero's three
 * @see zero_one_value_array_rechunk_hooks_inline.h for the bodies
 */

#include "chunk/config.h"        // build_config::chunk_dynamic
#include "chunk/chunk_traits.h"  // ArrayChunkTraits
#include "chunk/internal/in_t.h" // InT<T>: the captured state

namespace shrink_array
{

namespace zero_one_value
{

/**
 * @brief zero_one_value::Array's re-chunk hooks, present only when the geometry is a runtime value.
 *
 * @tparam Derived the concrete array type (only zero_one_value::Array today).
 * @tparam Dynamic defaulted to build_config::chunk_dynamic; never pass it
 *         explicitly outside of tests.
 */
template <typename Derived, bool Dynamic = build_config::chunk_dynamic>
class RechunkHooks;

// ── CHUNK_DYNAMIC build: the real hooks ──────────────────────────────────────
/**
 * @brief Dynamic specialisation: the six hooks zero_one_value overrides.
 * @copydetails zero_one_value::RechunkHooks
 */
template <typename Derived>
class RechunkHooks<Derived, true>
{
  public:
    using Traits      = ArrayChunkTraits<Derived>; ///< Traits specialisation of the concrete array.
    using value_type  = Traits::value_type;        ///< Stored element type.
    using ProxyType   = Traits::ProxyType;         ///< Proxy type owning the per-type array registry.
    using ManagerType = Traits::ManagerType;       ///< Per-type manager owning the candidate store.

    /// @brief Drop this array's queued compression candidates before teardown.
    void rechunk_drop_candidates(this Derived& self);

    /**
     * @brief Drain the reuse free-list of the target geometry.
     * @param new_size the chunk size the array is moving to
     * @details The buffers parked there were sized for an older geometry.
     */
    void rechunk_drain_reuse_list(this const Derived& /*self*/, unsigned int new_size)
    {
        ManagerType::drain_reuse_list_for_size(new_size);
    }

    /// @brief Capture the array-level state that must survive the rechunk.
    /// @return this array's second value
    [[nodiscard]] value_type rechunk_capture_state(this const Derived& self);

    /**
     * @brief Re-declare the captured state on the freshly built array.
     * @param state value returned by rechunk_capture_state()
     */
    void rechunk_seed_fresh(this Derived& self, InT<value_type> state);

    /**
     * @brief Exchange identity with the scratch array built at the new geometry.
     * @param tmp freshly built array whose storage this array takes over
     * @return true if the registry entries were exchanged
     */
    bool rechunk_swap_identity(this Derived& self, Derived& tmp);

    /**
     * @brief Re-own the map_inf_array entry after the identity swap.
     * @param tmp   scratch array the storage came from
     * @param state value returned by rechunk_capture_state()
     * @details Re-points the shared entry at the SURVIVOR's inf_array member, so
     *          that tmp's destructor does not erase an entry the survivor still
     *          needs -- see zero_one_value::Array's destructor, second warning.
     */
    void rechunk_reown_after_swap(this Derived& self, Derived& tmp, InT<value_type> state);
};

// ── Static-geometry build: nothing to hook ───────────────────────────────────
/**
 * @brief Static specialisation: the same six signatures, inert.
 *
 * @details Never called; kept for zero_one_value::Array's using-declarations.
 */
template <typename Derived>
class RechunkHooks<Derived, false>
{
  public:
    using Traits     = ArrayChunkTraits<Derived>; ///< Traits specialisation of the concrete array.
    using value_type = Traits::value_type;        ///< Stored element type.

    /** @brief No-op: the geometry is fixed at compile time. */
    void rechunk_drop_candidates() const noexcept {}
    /** @brief No-op: the geometry is fixed at compile time. */
    void rechunk_drain_reuse_list(unsigned int /*new_size*/) const noexcept {}
    /** @brief No state to carry: there is no re-chunk to carry it across. */
    [[nodiscard]] value_type rechunk_capture_state() const
    {
        return value_type{};
    }
    /** @brief No-op: there is no fresh array to seed. */
    void rechunk_seed_fresh(InT<value_type> /*state*/) const noexcept {}
    /** @brief No-op: there is no temporary array to swap identities with. */
    bool rechunk_swap_identity(Derived& /*tmp*/) const noexcept
    {
        return true;
    }
    /** @brief No-op: no identity was swapped, so nothing to re-own. */
    void rechunk_reown_after_swap(Derived& /*tmp*/, InT<value_type> /*state*/) const noexcept {}
};

} // namespace zero_one_value

} // namespace shrink_array
