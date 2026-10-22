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
 * @file array_chunk_rechunk_hooks.h
 * @brief The re-chunk hooks of ArrayChunk (v1) and ArrayChunk2A — a
 *        CHUNK_DYNAMIC-only concern, isolated in its own mixin so they vanish
 *        from a static-geometry build.
 *
 * @details RechunkSupport (array_chunk_rechunk.h) carries the shared re-chunk
 *          algorithm and calls back into the concrete array through the
 *          rechunk_* hooks; v1 and 2A provide three of them and inherit the
 *          base defaults for the other three. Those three only ever run under
 *          RechunkSupport<Derived, true>::rechunk_to(), which does not exist in
 *          a static-geometry build.
 *
 *          v1 and 2A share them: both keep num_tab as a member, in step with
 *          the registry, and queue their compression candidates in the
 *          manager's per-rank lists. They differ only in which lists, the
 *          compress() call of a candidate, where num_tab is read, and v1's
 *          flush tripwire (see rechunk_drop_candidates()).
 *
 *          RechunkHooks<Derived, chunk_dynamic> splits them the same way as the
 *          algorithm they serve: the dynamic specialisation carries the real
 *          bodies, the static one keeps the signatures as no-ops so the
 *          arrays' using-declarations stay valid in both modes.
 *
 * @note ArrayChunk2 keeps its own hooks (RechunkHooks2): a per-value candidate
 *       map, a second value to carry over, and an identity that must NOT be
 *       swapped.
 *
 * @see array_chunk_rechunk.h for the algorithm that calls these
 * @see array_chunk_rechunk_hooks_inline.h for the bodies
 */

#include <type_traits> // std::is_same_v (v1 or 2A)

#include "chunk/config.h"       // build_config::chunk_dynamic
#include "chunk/chunk_traits.h" // ArrayChunkTraits

namespace shrink_array
{

/**
 * @brief The re-chunk hooks of ArrayChunk and ArrayChunk2A, present only when
 *        the geometry is a runtime value.
 *
 * @tparam Derived the concrete array type (ArrayChunk or ArrayChunk2A).
 * @tparam Dynamic defaulted to build_config::chunk_dynamic; never pass it
 *         explicitly outside of tests.
 */
template <typename Derived, bool Dynamic = build_config::chunk_dynamic>
class RechunkHooks;

// ── CHUNK_DYNAMIC build: the real hooks ──────────────────────────────────────
/**
 * @brief Dynamic specialisation: the three hooks v1 and 2A actually override.
 * @copydetails RechunkHooks
 */
template <typename Derived>
class RechunkHooks<Derived, true>
{
  public:
    using Traits       = ArrayChunkTraits<Derived>; ///< Traits specialisation of the concrete array.
    using value_type   = Traits::value_type;        ///< Stored element type.
    using proxy_type   = Traits::proxy_type;        ///< Proxy type owning the per-type array registry.
    using manager_type = Traits::manager_type;      ///< Per-type manager owning the candidate store.

    /// @brief ArrayChunk2A (otherwise ArrayChunk, v1): selects the candidate lists and the compress() call.
    static constexpr bool is_chunk2a = std::is_same_v<typename Traits::chunk_type, Chunk2A<value_type>>;

    /**
     * @brief Compress, then drop, this array's queued compression candidates, by num_tab.
     * @details Must happen at the OWNING geometry: a candidate scanned later at
     *          another chunk size would walk past the end of its buffer.
     */
    void rechunk_drop_candidates();

    /**
     * @brief Drain the reuse free-list of the target geometry.
     * @param new_size the chunk size the array is moving to
     * @details The buffers parked there were sized for an older geometry.
     */
    void rechunk_drain_reuse_list(unsigned int new_size);

    /**
     * @brief Swap registry identity with the fresh array built at the new geometry.
     * @param tmp the temporary array rechunk_to() swaps storage with
     * @return whatever the registry swap reports
     * @details num_tab is set by pre_create_hook() and indexes the per-array
     *          metadata, so the member is swapped as well as the registry
     *          entry, keeping the two consistent (get_num_tab(registry_key())
     *          then names this array's new row).
     */
    bool rechunk_swap_identity(Derived& tmp);

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

  private:
    /// @brief The manager's per-rank candidate lists of this variant (OpenMP builds).
    [[nodiscard]] static auto& candidate_lists_thr() noexcept
    {
        if constexpr (is_chunk2a)
        {
            return manager_type::chunk2a_to_compress_potentially_thr;
        }
        else
        {
            return manager_type::chunk_to_compress_potentially_thr;
        }
    }
    /// @brief The manager's single candidate list of this variant (serial builds).
    [[nodiscard]] static auto& candidate_list() noexcept
    {
        if constexpr (is_chunk2a)
        {
            return manager_type::chunk2a_to_compress_potentially;
        }
        else
        {
            return manager_type::chunk_to_compress_potentially;
        }
    }
    /// @brief Remove every candidate of the array @p num_tab from this variant's lists.
    static void remove_candidates_of_tab(int num_tab) noexcept
    {
        if constexpr (is_chunk2a)
        {
            manager_type::remove_from_compress_list_chunk2a_by_num_tab(num_tab);
        }
        else
        {
            manager_type::remove_from_compress_list_chunk_by_num_tab(num_tab);
        }
    }
};

// ── Static-geometry build: nothing to hook ───────────────────────────────────
/**
 * @brief Static specialisation: the same three signatures, inert.
 *
 * @details Never called — RechunkSupport<Derived, false>::rechunk_to() does
 *          nothing — but kept so the arrays can pull the names in with the same
 *          using-declarations in both modes.
 */
template <typename Derived>
class RechunkHooks<Derived, false>
{
  public:
    /** @brief No-op: the geometry is fixed at compile time. */
    static void rechunk_drop_candidates() noexcept {}
    /** @brief No-op: the geometry is fixed at compile time. */
    static void rechunk_drain_reuse_list(unsigned int /*new_size*/) noexcept {}
    /** @brief No-op: there is no temporary array to swap identities with. */
    static bool rechunk_swap_identity(Derived& /*tmp*/) noexcept
    {
        return true;
    }
};

} // namespace shrink_array
