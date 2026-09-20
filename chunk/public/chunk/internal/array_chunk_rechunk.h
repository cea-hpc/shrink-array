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
 * @file array_chunk_rechunk.h
 * @brief Runtime geometry change (re-chunk) — a CHUNK_DYNAMIC-only concern,
 *        isolated in its own mixin so it vanishes from a static-geometry build.
 *
 * @details Re-chunking means moving an array from a chunk size S to another size
 *          S'. That is only meaningful when the geometry is a runtime value
 *          (DynamicChunkSize / -DCHUNK_DYNAMIC): with the compile-time geometry
 *          the size is baked into every shift and mask, so there is nothing to
 *          change. The algorithm nevertheless lived in ArrayChunkBase, opening
 *          with an `if constexpr (!chunk_dynamic) return;` — some 200 lines of
 *          machinery every static build had to read, and that every reader of
 *          the container had to skip past.
 *
 *          RechunkSupport<Derived, chunk_dynamic> splits it in two: the dynamic
 *          specialisation carries the real algorithm (unchanged), the static one
 *          keeps the same signatures as no-ops. Both keep the @c rechunk_* hooks
 *          with their base defaults, because the derived arrays override them
 *          unconditionally.
 *
 * @see array_chunk_rechunk_inline.h for the two large bodies
 * @see chunk_size_policy.h for the geometry policies this switches between
 */

#include <vector> // std::vector

#include "chunk/config.h"
#include "chunk/chunk_traits.h"

namespace shrink_array
{

/**
 * @brief Re-chunking support, present only when the geometry is a runtime value.
 *
 * @tparam Derived the concrete array type (ArrayChunk, ArrayChunk2, ArrayChunk2A).
 * @tparam Dynamic defaulted to build_config::chunk_dynamic; never pass it
 *         explicitly outside of tests.
 */
template <typename Derived, bool Dynamic = build_config::chunk_dynamic>
class RechunkSupport;

// ── CHUNK_DYNAMIC build: the real algorithm ──────────────────────────────────
/**
 * @brief Dynamic specialisation: the shared re-chunk algorithm and its hooks.
 */
template <typename Derived>
class RechunkSupport<Derived, true>
{
  public:
    using Traits     = ArrayChunkTraits<Derived>;
    using T          = Traits::value_type;
    using ind_type   = Traits::index_type;
    using chunk_type = Traits::chunk_type;
    using proxy_type = Traits::proxy_type;

    /**
     * @brief Re-chunk EVERY live array of this variant to @p new_size (quiescent point only).
     *
     * @details Mirrors DirtyBitsetSupport::clear_all_dirty_bitsets: iterate the
     *          per-type registry and act on each typed array. Intended for a
     *          GLOBAL adaptive re-chunk where all arrays must share one geometry
     *          (e.g. a bench that writes several arrays interleaved under a
     *          single active_chunk_size): the allocation path reads the
     *          process-wide t_active, so mixed per-array geometries would misfit
     *          a materialising write.
     *
     * @param new_size the geometry every array should end up with
     * @warning Each rechunk_to() restores the OLD active geometry on exit, so the
     *          CALLER must set the active/thread size to @p new_size AFTER this
     *          returns (once, for all types) before the next write phase.
     * @see rechunk_to
     */
    static void rechunk_all_to(unsigned int new_size)
    {
        proxy_type::share_info_array.for_each_key(
            [new_size](const void* key)
            {
                if (key)
                {
                    auto* arr = static_cast<Derived*>(const_cast<void*>(key));
                    arr->rechunk_to(new_size);
                }
            });
    }

    // ---- Re-chunk to a new geometry S' (SEQUENTIAL / quiescent point) --------
    /**
     * @brief Re-chunk this array to a new geometry S' (sequential, quiescent point).
     *
     * @details Single implementation shared by ArrayChunk / ArrayChunk2 /
     *          ArrayChunk2A via CRTP; the per-variant differences live in the
     *          rechunk_* hooks the derived classes provide. The delicate
     *          invariants are documented ONCE, in the body.
     *
     *          Approach: build a FRESH array at S', refill it through the normal
     *          write path, then SWAP storage + identity with *this. The library
     *          assumes geometry is immutable post-construction (frozen size,
     *          cached proxy pointers, deferred-compression state...); a fresh S'
     *          array never violates that. After the swap *this carries S' storage
     *          under tmp's num_tab, and tmp carries the old identity+geometry,
     *          which it destroys coherently on scope exit.
     *
     *          Hooks (see each ArrayChunk*):
     *          - rechunk_drop_candidates() — remove this array's queued
     *            compression candidates by num_tab
     *          - rechunk_drain_reuse_list(new_size) — drain the S' reuse free-list
     *          - rechunk_capture_state() -> T — variant state to carry across
     *            (e.g. Chunk2's array-level second_value)
     *          - rechunk_seed_fresh(state) — apply that state to the fresh array
     *          - rechunk_swap_identity(tmp) -> bool — swap num_tab member (if any)
     *            + registry entry (swap_keys/entries)
     *          - rechunk_reown_after_swap(tmp,state) — post-swap fixup (Chunk2
     *            re-owns its map_inf_array entry so it doesn't dangle when tmp dies)
     *
     * @param new_size the new chunk size (power of two)
     * @warning Restores the previously active geometry on exit; see
     *          rechunk_all_to for what the caller must do afterwards.
     */
    void rechunk_to(unsigned int new_size);

    /**
     * @brief Refill @p tmp from *this across a geometry change.
     *
     * @details Bounded tiled element transcription — stream in ~1M-element tiles,
     *          toggling the ambient geometry per tile (read *this at OLD, write
     *          tmp at NEW). Peak extra RAM is tile*sizeof(T), not n*sizeof(T).
     *          Read side is parallel pure-read (operator[] is const); bool stays
     *          serial (std::vector<bool> bit-packing). Runs at a quiescent serial
     *          point, so the t_active toggles are safe.
     *
     * @param tmp the fresh array at the NEW geometry, to be filled
     * @param old_size chunk size of *this (the read side)
     * @param new_size chunk size of @p tmp (the write side)
     *
     * @note A chunk-level rebuild that SKIPS constant regions is much faster, but
     *       skipping leaves those tmp chunks fresh — never routed through the
     *       write path that ARMS the lazy per-chunk pointers (ptr_second_value /
     *       compression bookkeeping). Unarmed chunks double-free in the later
     *       compression/clear at scale (seen on the 2A perf run; rechunk_validation
     *       is too small to trip it). The fast path therefore needs DIRECT chunk
     *       construction that arms the chunk (Level-2), not a plain skip. Kept as
     *       the safe element transcription for now.
     */
    void rechunk_refill(Derived& tmp, int old_size, int new_size);

    /**
     * @name Re-chunk hooks (base defaults)
     *
     * Base defaults for the rechunk_* hooks: no-ops that suit a variant with no
     * deferred candidates / no second_value / no registry. Each ArrayChunk*
     * overrides the ones it needs (they are hidden, not virtual — resolved via
     * CRTP self()/derived()). Kept here so a variant only defines what differs.
     * @{
     */
    /** @brief Forget this array's compression candidates before the swap (default: none to drop). */
    void rechunk_drop_candidates() {}
    /** @brief Release this array's parked buffers held at the old geometry (default: none parked). */
    void rechunk_drain_reuse_list(unsigned int /*new_size*/) {}
    /** @brief Capture the variant state to carry across a re-chunk (default: none). */
    T rechunk_capture_state() const
    {
        return T{};
    }
    /** @brief Apply the captured state to the fresh array (default: no-op). */
    void rechunk_seed_fresh(const T& /*state*/) {}
    /** @brief Post-swap fixup so the dying temporary does not take state with it (default: no-op). */
    void rechunk_reown_after_swap(Derived& /*tmp*/, const T& /*state*/) {}
    /** @brief Swap registry identity with the temporary (default: nothing to swap, succeeds). */
    bool rechunk_swap_identity(Derived& /*tmp*/)
    {
        return true;
    }
    /** @} */

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

// ── Static-geometry build: nothing to re-chunk ───────────────────────────────
/**
 * @brief Static specialisation: the same surface, inert.
 *
 * @details With a compile-time chunk size there is no geometry to move to, so
 *          rechunk_to()/rechunk_all_to()/rechunk_refill() do nothing. They are
 *          kept (rather than removed) because call sites — the re-chunk decider,
 *          the manager, the benches — are written once for both modes. The hooks
 *          keep their defaults so the derived overrides stay valid declarations
 *          in a static build too.
 */
template <typename Derived>
class RechunkSupport<Derived, false>
{
  public:
    using Traits     = ArrayChunkTraits<Derived>;
    using T          = Traits::value_type;
    using chunk_type = Traits::chunk_type;

    /** @brief No-op: the geometry is fixed at compile time. */
    static void rechunk_all_to(unsigned int /*new_size*/) noexcept {}
    /** @brief No-op: the geometry is fixed at compile time. */
    void rechunk_to(unsigned int /*new_size*/) noexcept {}
    /** @brief No-op: never reached, since rechunk_to() does nothing. */
    void rechunk_refill(Derived& /*tmp*/, unsigned int /*old_size*/, unsigned int /*new_size*/) noexcept {}

    /** @name Re-chunk hooks (base defaults) — same no-ops as the dynamic specialisation's. @{ */
    /** @brief Nothing to drop. */
    void rechunk_drop_candidates() {}
    /** @brief Nothing pooled by geometry. */
    void rechunk_drain_reuse_list(unsigned int /*new_size*/) {}
    /** @brief No state to carry. */
    T    rechunk_capture_state() const
    {
        return T{};
    }
    /** @brief No state to apply. */
    void rechunk_seed_fresh(const T& /*state*/) {}
    /** @brief Nothing to re-own. */
    void rechunk_reown_after_swap(Derived& /*tmp*/, const T& /*state*/) {}
    /** @brief Nothing to swap; succeeds. */
    bool rechunk_swap_identity(Derived& /*tmp*/)
    {
        return true;
    }
    /** @} */

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

} // namespace shrink_array
