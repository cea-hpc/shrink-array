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

#include <cstdlib>     // std::abort
#include <functional>  // std::divides, std::multiplies
#include <iostream>    // std::cerr
#include <type_traits> // std::is_integral_v, std::is_same_v
#include <utility>     // std::forward

#include "chunk/bit.h"    // is_bool_like_v: bool and Bit share the bool path
#include "chunk/config.h" // chunk_config::use_fast_block_compression, build_config::do_check
#include "chunk/internal/concept_is_clean_assignement.h" // base_t, ElementMultipliable, ElementModulable
#include "chunk/internal/def_force_inline.h"             // CHUNK_FORCE_INLINE
#include "chunk/internal/in_t.h"                         // in_t<T>
#include "chunk/internal/proxy_base.h"                   // ProxyBase

namespace shrink_array
{

// ============================================================================
// Proxy2Base: logic shared by Proxy2 and Proxy2A (second_value, etc.)
// ============================================================================

/**
 * @brief Base class for proxies with a second_value (CRTP)
 * @tparam Derived The derived proxy class (Proxy2<T> or Proxy2A<T>)
 * @tparam T The underlying type
 * @tparam ChunkType The chunk type (Chunk2<T> or Chunk2A<T>)
 *
 * Contains the logic specific to chunks with two constant values.
 */

template <typename Derived, typename T, typename ChunkType>
class Proxy2Base : public ProxyBase<Derived, T, ChunkType>
{
  public:
    using Base = ProxyBase<Derived, T, ChunkType>;
    friend Base;

    using Base::add_to_compression_list;
    using Base::chunk;
    using Base::derived;
    using Base::is_cst_val1;
    using Base::is_not_cst_val1;
    using Base::num;
    using Base::num_chunk;
    using Base::num_tab;
    using Base::values;

    using typename Base::atomic_ptr_ref_t;
    using values_type_ref = atomic_ptr_ref_t;

    // Both members exist on ProxyBase in every mode (the serial specialisation
    // declares them too); only their USE is mode-specific.
    using Base::ptr_is_inside_chunk_list_to_compress;
    using Base::tab_is_inside_chunk_list_to_compress;

    // ========================================================================
    // CRTP: Access to derived class
    // ========================================================================

    /**
     * @brief Whether the owning chunk is currently compressed to its second
     *        constant (delegates to the derived class, which knows the
     *        index/ind for non-bool types).
     * @return true if the chunk is compressed to the second constant.
     */
    [[nodiscard]] bool is_cst_val2() const noexcept
    {
        return derived().is_cst_val2();
    }

    /**
     * @brief Negation of is_cst_val().
     * @return true if the chunk is materialised (compressed to neither constant).
     */
    [[nodiscard]] bool is_not_cst_val() const noexcept
    {
        return !is_cst_val();
    }

    /**
     * @brief Multiply this element by `v` in place.
     * @tparam U Type of the operand.
     * @param v Multiplier.
     * @return This proxy. Converting it to T reads the resulting value,
     *         synchronised (operator T() -> read_pure()); discarding it,
     *         as `a[i] += x;` does, keeps the operation fully deferrable.
     *         Bool: not implemented -- prints an error and leaves the
     *         element unchanged.*/
    template <typename U>
        requires ElementMultipliable<T>
    Derived operator*=(const U& v) noexcept
    {
        if constexpr (is_bool)
        {
            std::cerr << "operator not implemented for bool\n";
            return derived();
        }
        else
        {
            // Early-out: multiplying by 1 changes nothing, so there is nothing to
            // write. Nothing to read either: the proxy is returned, and reading
            // it goes through operator T() -> read_pure() and its read-side gate.
            // base_t<U> extracts the arithmetic type (e.g., double) even if U is a Proxy.
            if (v == static_cast<base_t<U>>(1))
            {
                return derived();
            }

            // 2. Call the common math engine
            // We use the transparent functor std::multiplies<>
            derived().template apply_compound_operation<U>(std::multiplies<>{}, v, true);
            return derived();
        }
    }

    /**
     * @brief Divide this element by `v` in place.
     * @tparam U Type of the operand.
     * @param v Divisor. Debug builds abort on v == 0.
     * @return This proxy. Converting it to T reads the resulting value,
     *         synchronised (operator T() -> read_pure()); discarding it,
     *         as `a[i] += x;` does, keeps the operation fully deferrable.
     *         Bool: not implemented -- prints an error and leaves the
     *         element unchanged.*/
    template <typename U>
        requires ElementDivisible<T>
    Derived operator/=(const U& v) noexcept
    {
        if constexpr (is_bool)
        {
            std::cerr << "operator not implemented for bool\n";
            return derived();
        }
        else
        {
            if constexpr (build_config::do_check)
            {
                if (v == 0)
                {
                    std::abort();
                }
            }

            // Early-out: dividing by 1 changes nothing -- nothing to write, and
            // nothing to read: reading the returned proxy goes through
            // operator T() -> read_pure() (see operator*=).
            if (v == static_cast<base_t<U>>(1))
            {
                return derived();
            }
            // 2. Execution via the common engine
            derived().template apply_compound_operation<U>(std::divides<>{}, v, true);
            return derived();
        }
    }

    /**
     * @brief Compute this element modulo `v` in place.
     * @tparam U Type of the operand.
     * @param v Modulus.
     * @return This proxy. Converting it to T reads the resulting value,
     *         synchronised (operator T() -> read_pure()); discarding it,
     *         as `a[i] += x;` does, keeps the operation fully deferrable.
     *         Bool: not implemented -- prints an error and leaves the
     *         element unchanged.*/
    template <typename U>
        requires ElementModulable<T> || is_bool_like_v<T>
    Derived operator%=(const U& v) noexcept
    {
        if constexpr (std::is_integral_v<T> && is_not_bool)
        {
            // 2. Execution via the common engine. true, as for *= and /=: a
            // non-zero slot can land on zero (4 % 2), and with false that zero
            // was never counted -- the chunk then never reached "full first
            // value" and stayed materialised.
            derived().template apply_compound_operation<U>(typename Base::modulo{}, v, true);
            return derived();
        }
        else
        {
            std::cerr << "operator not implemented (requires integral non-bool type)\n";
            return derived();
        }
    }

  protected:
    /**
     * @brief Construct a proxy for element `num_` of `chunk_`.
     * @param values_ Reference to the chunk's values pointer.
     * @param num_ Element offset within the chunk.
     * @param chunk_ Owning chunk.
     */
    explicit Proxy2Base(values_type_ref values_, int num_, ChunkType& chunk_) noexcept : Base(values_, num_, chunk_) {}

    /**
     * @brief Negation of is_cst_val2().
     * @return true if the chunk is NOT compressed to the second constant.
     */
    [[nodiscard]] bool is_not_cst_val2() const noexcept
    {
        return !is_cst_val2();
    }

    /**
     * @brief Whether the chunk is compressed to EITHER constant (first or second).
     * @return true if the chunk is compressed.
     */
    [[nodiscard]] bool is_cst_val() const noexcept
    {
        return is_cst_val1() || is_cst_val2();
    }

    /**
     * @brief Pre-check shared by every variant: the write is of a constant.
     * @details A write is worth the marking path (one read of the old value,
     *          then add_to_compression_list) when the value written is one of
     *          the chunk's two constants -- whatever the chunk's state.
     *          should_mark_for_compression() then keeps the writes that change
     *          the element (old != new), so a chunk is queued at most once per
     *          pass, on its first write of a constant over something else.
     *
     *          Not only a chunk ALREADY at one constant being written the
     *          other: a chunk materialised by a plain value (2.5 over cst_val1)
     *          and then written back to a constant element by element must be
     *          queued too, or it stays materialised for the run. The cost is
     *          paid only by writes of a constant: an old-value read and a
     *          compare, and, on the first such write per pass, the flag's CAS.
     *          The candidate list then also holds chunks that will not be
     *          constant at the pass, scanned at chunk_size each.
     * @param new_value The new value to check.
     * @param first_value The first constant value to compare against.
     * @param second_val The second constant value to compare against.
     * @return true if new_value is first_value or second_val.
     */
    [[nodiscard]] static bool
    pre_check_mark_for_compression_strict(in_t<T> new_value, in_t<T> first_value, in_t<T> second_val) noexcept
    {
        return new_value == first_value || new_value == second_val;
    }

    // --- Boolean overloads ---
    /**
     * @brief Marks block for compression based on a boolean new value
     * @param new_value The new boolean value
     * @note Uses false and true as constant values for comparison
     */
    void mark_for_compression_if_needed(bool new_value) const
    {
        const bool isv = is_cst_val_new_value_is_other_cst_val(new_value, false, true);
        // Base::read_values(), not values[num]: inherited from a dependent
        // base, so it must be qualified. This is the write path's read
        // entry and the only one that resolves a compressed chunk's
        // constant (an unbacked identifier slot). Indexing here read a
        // zero page and the decision below was taken against 0.
        mark_for_compression_impl(Base::read_values(), new_value, isv, false, true);
    }

    /**
     * @brief Marks block for compression with custom constant values
     * @param second_val The second constant value
     * @param new_value The new value to be set
     * @param first_value The first constant value
     */
    void mark_for_compression_if_needed(in_t<T> second_val, in_t<T> new_value, in_t<T> first_value) const
    {
        const bool isv = is_cst_val_new_value_is_other_cst_val(new_value, first_value, second_val);
        mark_for_compression_impl(Base::read_values(), new_value, isv, second_val, first_value);
    }

    /**
     * @brief Marking for a caller that stores AFTER marking (Proxy2A's
     *        assignment: mark, then perform_core_assignment).
     * @details Same test as mark_for_compression_if_needed() above, except the
     *          memory check of mark_for_compression_core() expects the OLD
     *          value. The helper above expects the new one -- right for a
     *          caller that has already stored (memory then differs only if a
     *          race changed it), but before the store memory still holds the
     *          old value, so its check would fire on every write that changes
     *          the value and queue the chunk whatever should_mark said.
     */
    void mark_for_compression_before_store(in_t<T> second_val, in_t<T> new_value, in_t<T> first_value) const
    {
        const bool isv = is_cst_val_new_value_is_other_cst_val(new_value, first_value, second_val);
        const T    old = Base::read_values();
        mark_for_compression_core(old, new_value, old, isv, second_val, first_value);
    }

    /**
     * @brief Marks block for compression with an explicit old boolean value
     *        (provided to avoid a race), rather than reading it back from
     *        values[num].
     * @param old_value The previous boolean value.
     * @param new_value The new boolean value.
     */
    void mark_for_compression_if_needed(bool old_value, bool new_value) const
    {
        const bool isv = is_cst_val_new_value_is_other_cst_val(new_value, false, true);
        mark_for_compression_impl(old_value, new_value, isv, true, false);
    }

    /**
     * @brief Marks block for compression with explicit old and new values
     * @param second_val The second constant value
     * @param old_value The previous value (provided explicitly to avoid race)
     * @param new_value The new value to be set
     * @param first_value The first constant value
     */
    void
    mark_for_compression_if_needed(in_t<T> second_val, in_t<T> old_value, in_t<T> new_value, in_t<T> first_value) const
    {
        const bool isv = is_cst_val_new_value_is_other_cst_val(new_value, first_value, second_val);
        mark_for_compression_impl(old_value, new_value, isv, second_val, first_value);
    }

    /**
     * @brief Write operation with metadata initialization.
     * @tparam WriteFn Callable type for write operation.
     * @param write_fn Function to execute after metadata setup.
     * @details Combines fast/slow path locking with automatic metadata
     *          initialization (set_num_tab_ptr_is_inside_chunk(), needed by
     *          the fast-block-compression path). Useful for operations that
     *          need tab/chunk information before writing.
     */
    template <typename WriteFn>
    [[nodiscard]] CHUNK_FORCE_INLINE bool write_with_optional_lock_with_set_ptr(WriteFn write_fn) noexcept
    {
        const auto lbd_info = [this]
        {
            if constexpr (chunk_config::use_fast_block_compression)
            {
                derived().set_num_tab_ptr_is_inside_chunk(); // common initialization required for the fast-path
            }
            else
            {
                // The capture exists for the branch above and for nothing
                // else. use_fast_block_compression IS use_openmp, so a SERIAL
                // build never instantiates that branch and clang reports the
                // capture as unused (-Wunused-lambda-capture; 13 reports for
                // this one lambda on a chunk2 bench TU, and the same on
                // chunk2a). Keeping `this` explicit and discarding it here
                // says why it is there -- switching to [&] would silence the
                // warning by hiding the intent.
                static_cast<void>(this);
            }
        };

        // wrap the call in a lambda that calls write_fn(v)
        return Base::write_with_optional_lock_impl([&]() noexcept { write_fn(); }, lbd_info);
    }

    /**
     * @brief Static entry point for async callers that only have a proxy
     *        reference in hand (not `this`). Delegates to the instance
     *        method of the same name.
     * @param proxy Proxy whose compression marking to trigger.
     * @param old_value Value before the write.
     * @param new_value Value after the write.
     */
    static void
    mark_for_compression_if_needed_with_value_async(const Derived& proxy, in_t<T> old_value, in_t<T> new_value)
    {
        proxy.mark_for_compression_if_needed_with_value(old_value, new_value);
    }

  private:
    static constexpr bool is_bool     = is_bool_like_v<T>; // bool and Bit (bit.h)
    static constexpr bool is_not_bool = !is_bool;

    /**
     * @brief Whether the chunk is currently at ONE of the two constants and
     *        new_value is precisely the OTHER one -- i.e. a direct
     *        constant-to-constant transition (cst1 -> second_val, or
     *        cst2 -> first_value).
     * @note Keep the terms conjunctive. `(A && is_cst_val2()) || is_not_cst_val2()`
     *       reduces to `A || !is_cst_val2()`, and since a chunk is at most one
     *       of cst_val1 / cst_val2, one of the two `!B` terms is always true:
     *       the pre-check (`isv` of mark_for_compression_core) would then
     *       filter nothing.
     * @param new_value The new value to check.
     * @param first_value The first constant value to compare against.
     * @param second_val The second constant value to compare against.
     * @return true if the chunk is at cst_val1 and new_value == second_val,
     *         or at cst_val2 and new_value == first_value.
     */
    [[nodiscard]] bool
    is_cst_val_new_value_is_other_cst_val(in_t<T> new_value, in_t<T> first_value, in_t<T> second_val) const noexcept
    {
        // CRTP dispatch: each variant owns its fast pre-check. Chunk2 keeps the
        // "a constant is written" test below; Chunk2A widens it (see
        // Proxy2A::pre_check_mark_for_compression) because its
        // should_mark_for_compression() accepts a case this test would reject.
        //
        // INVARIANT: pre_check_mark_for_compression() must never return false
        // for a write that should_mark_for_compression() would accept. `isv` is
        // only a performance filter; if it rejects a qualifying write the chunk
        // is silently never queued for compression and its buffer is never
        // returned to the reuse list.
        return derived().pre_check_mark_for_compression(new_value, first_value, second_val);
    }

    /**
     * @brief Default pre-check hook (Chunk2 behaviour).
     * @details Variants that widen should_mark_for_compression() MUST shadow
     *          this method and widen the pre-check to match -- otherwise the
     *          extra predicate clauses are unreachable.
     * @param new_value The new value to check.
     * @param first_value The first constant value to compare against.
     * @param second_val The second constant value to compare against.
     * @return true if the write is worth examining further.
     */
    [[nodiscard]] bool
    pre_check_mark_for_compression(in_t<T> new_value, in_t<T> first_value, in_t<T> second_val) const noexcept
    {
        return pre_check_mark_for_compression_strict(new_value, first_value, second_val);
    }

    /**
     * @brief Core unified logic for 2-value compression marking.
     * @details When `isv` is true, checks whether this write's transition
     *          qualifies for compression (derived().should_mark_for_compression())
     *          or whether the in-memory value no longer matches
     *          `expected_in_memory` (a race with a concurrent writer); either
     *          way, adds this element to the compression candidate list.
     *          `isv` itself is a fast pre-check meant to let obviously
     *          irrelevant writes skip that work entirely (see
     *          is_cst_val_new_value_is_other_cst_val()).
     * @tparam Args Extra args forwarded to should_mark_for_compression()
     *              (the second/first constant values).
     * @param old_value Value before the write.
     * @param new_value Value after the write.
     * @param expected_in_memory The value this write expects to still be in
     *                           memory.
     * @param isv Fast pre-check result; when false, this call is a no-op.
     * @param compression_args Extra args forwarded to should_mark_for_compression().
     */
    template <typename... Args>
    void mark_for_compression_core(in_t<T> old_value,
                                   in_t<T> new_value,
                                   in_t<T> expected_in_memory,
                                   bool    isv,
                                   Args&&... compression_args) const;

    /**
     * @brief Post-write wrapper around mark_for_compression_core(), passing
     *        new_value as the expected-in-memory value.
     * @tparam Args Extra args forwarded to mark_for_compression_core().
     * @param old_value Value before the write.
     * @param new_value Value after the write.
     * @param isv Fast pre-check result (see mark_for_compression_core()).
     * @param args Extra args forwarded to mark_for_compression_core().
     */
    template <typename... Args>
    void mark_for_compression_impl(in_t<T> old_value, in_t<T> new_value, bool isv, Args&&... args) const
    {
        mark_for_compression_core(old_value, new_value, new_value, isv, std::forward<Args>(args)...);
    }
};

} // namespace shrink_array

#include "chunk/internal/proxy2_base_inline.h" // the template bodies