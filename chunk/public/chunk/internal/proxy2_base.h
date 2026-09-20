
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

#include <cstdlib>     // std::abort
#include <functional>  // std::divides, std::multiplies
#include <iostream>    // std::cerr
#include <type_traits> // std::is_integral_v, std::is_same_v
#include <utility>     // std::forward

#include "chunk/internal/in_t.h"
#include "chunk/internal/proxy_base.h"

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

    static constexpr bool is_bool     = std::is_same_v<T, bool>;
    static constexpr bool is_not_bool = !is_bool;

    // Both members exist on ProxyBase in every mode (the serial specialisation
    // declares them too); only their USE is mode-specific.
    using Base::ptr_is_inside_chunk_list_to_compress;
    using Base::tab_is_inside_chunk_list_to_compress;

  protected:
    /**
     * @brief Construct a proxy for element `num_` of `chunk_`.
     * @param values_ Reference to the chunk's values pointer.
     * @param num_ Element offset within the chunk.
     * @param chunk_ Owning chunk.
     */
    explicit Proxy2Base(values_type_ref values_, int num_, ChunkType& chunk_) noexcept : Base(values_, num_, chunk_) {}

  public:
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
     * @brief Negation of is_cst_val().
     * @return true if the chunk is materialised (compressed to neither constant).
     */
    [[nodiscard]] bool is_not_cst_val() const noexcept
    {
        return !is_cst_val();
    }

    /**
     * @brief Whether the chunk is currently at ONE of the two constants and
     *        new_value is precisely the OTHER one -- i.e. a direct
     *        constant-to-constant transition (cst1 -> second_val, or
     *        cst2 -> first_value).
     * @note BUG FIX: this used to be
     *       `((new_value == first_value && is_cst_val2()) || is_not_cst_val2()) ||
     *        ((new_value == second_val && is_cst_val1()) || is_not_cst_val1())`.
     *       Since is_not_cst_val2() == !is_cst_val2() and is_not_cst_val1() ==
     *       !is_cst_val1(), each `(A && B) || !B` term reduces to `A || !B`,
     *       and a chunk can only ever be non-cst_val1 XOR non-cst_val2 XOR
     *       materialised -- so at least one of the two `!B` terms was always
     *       true, making the whole expression a tautology (always true, in
     *       all three reachable chunk states, verified exhaustively). The
     *       caller (mark_for_compression_if_needed) passes this as `isv`, a
     *       fast pre-check meant to guard the more expensive
     *       should_mark_for_compression() / load_value() check "for
     *       performance" (see mark_for_compression_core) -- with the old
     *       expression, that guard never filtered anything.
     * @param new_value The new value to check.
     * @param first_value The first constant value to compare against.
     * @param second_val The second constant value to compare against.
     * @return true if the chunk is at cst_val1 and new_value == second_val,
     *         or at cst_val2 and new_value == first_value.
     */
    bool
    is_cst_val_new_value_is_other_cst_val(in_t<T> new_value, in_t<T> first_value, in_t<T> second_val) const noexcept
    {
        // CRTP dispatch: each variant owns its fast pre-check. Chunk2 keeps the
        // strict constant-to-constant test below; Chunk2A widens it (see
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
     * @brief Strict constant-to-constant pre-check, shared by every variant.
     * @details Accepts only a write that moves the chunk directly from one
     *          constant to the other. This is the corrected (non-tautological)
     *          form; it is sufficient for Chunk2, whose
     *          should_mark_for_compression() only ever fires on such a
     *          transition.
     * @param new_value The new value to check.
     * @param first_value The first constant value to compare against.
     * @param second_val The second constant value to compare against.
     * @return true if the chunk is at cst_val1 and new_value == second_val,
     *         or at cst_val2 and new_value == first_value.
     */
    [[nodiscard]] bool
    pre_check_mark_for_compression_strict(in_t<T> new_value, in_t<T> first_value, in_t<T> second_val) const noexcept
    {
        return (is_cst_val1() && new_value == second_val) || (is_cst_val2() && new_value == first_value);
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
                                   bool     isv,
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
        // constant under CHUNK_CST_VAL2_NO_ARRAYS. Indexing here read a
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
    bool write_with_optional_lock_with_set_ptr(WriteFn write_fn) noexcept
    {
        auto lbd_info = [this]()
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
     * @brief Multiply this element by `v` in place.
     * @tparam U Type of the operand.
     * @param v Multiplier.
     * @return The resulting value. Bool: prints an error and returns v
     *         unchanged (not implemented for bool).
     */
    template <typename U>
        requires ElementMultipliable<T>
    T operator*=(const U& v) noexcept
    {
        if constexpr (is_bool)
        {
            std::cerr << "operator not implemented for bool\n";
            return static_cast<T>(v);
        }
        else
        {
            // Optimization: multiplying by 1 changes nothing
            // base_t<U> extracts the arithmetic type (e.g., double) even if U is a Proxy.
            // Pure-read early-out: route through read_pure() so it honours the
            // read-side gate under CHUNK_SYNC_ON_READ (byte-identical otherwise).
            if (v == static_cast<base_t<U>>(1))
                return Base::read_pure();

            // 2. Call the common math engine
            // We use the transparent functor std::multiplies<>
            return derived().template apply_compound_operation<U>(std::multiplies<>{}, v, true);
        }
    }

    /**
     * @brief Divide this element by `v` in place.
     * @tparam U Type of the operand.
     * @param v Divisor. Debug builds abort on v == 0.
     * @return The resulting value. Bool: prints an error and returns v
     *         unchanged (not implemented for bool).
     */
    template <typename U>
        requires ElementDivisible<T>
    T operator/=(const U& v) noexcept
    {
        if constexpr (is_bool)
        {
            std::cerr << "operator not implemented for bool\n";
            return static_cast<T>(v);
        }
        else
        {
            if constexpr (build_config::do_check)
            {
                if (v == 0)
                    std::abort();
            }

            // Pure-read early-out: route through read_pure() so it honours the
            // read-side gate under CHUNK_SYNC_ON_READ (byte-identical otherwise).
            if (v == static_cast<base_t<U>>(1))
                return Base::read_pure();
            // 2. Execution via the common engine
            return derived().template apply_compound_operation<U>(std::divides<>{}, v, true);
        }
    }

    /**
     * @brief Compute this element modulo `v` in place.
     * @tparam U Type of the operand.
     * @param v Modulus.
     * @return The resulting value. Non-integral or bool: prints an error
     *         and returns v unchanged (not implemented).
     */
    template <typename U>
        requires ElementModulable<T> || std::is_same_v<T, bool>
    T operator%=(const U& v) noexcept
    {
        if constexpr (std::is_integral_v<T> && is_not_bool)
        {
            // 2. Execution via the common engine
            return derived().template apply_compound_operation<U>(Base::modulo(), v, false);
        }
        else
        {
            std::cerr << "operator not implemented (requires integral non-bool type)\n";
            return static_cast<T>(v);
        }
    }
};

} // namespace shrink_array

#include "chunk/internal/proxy2_base_inline.h"