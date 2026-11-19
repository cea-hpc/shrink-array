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

#include <functional>  // std::divides, std::multiplies
#include <type_traits> // std::is_integral_v, std::is_same_v
#include <utility>     // std::forward

#include "chunk/bit.h"    // is_bool_like_v: bool and Bit share the bool path
#include "chunk/config.h" // chunk_config::use_fast_block_compression, build_config::do_check
#include "chunk/internal/concept_is_clean_assignment.h" // BaseT, ElementMultipliable, ElementModulable
#include "chunk/internal/def_force_inline.h"            // CHUNK_FORCE_INLINE
#include "chunk/internal/in_t.h"                        // InT<T>
#include "chunk/internal/proxy_base.h"                  // ProxyBase
#include "chunk/log.h"                                  // CHUNK_LOG (the operators bool has not)
#include "chunk/internal/same_value.h"                  // same_value

namespace shrink_array
{

// ============================================================================
// SecondValueProxyBase: logic shared by zero_or_value::Proxy and zero_any::Proxy (second_value, etc.)
// ============================================================================

/**
 * @brief Base class for proxies with a second_value (CRTP)
 * @tparam Derived The derived proxy class (zero_or_value::Proxy<T> or zero_any::Proxy<T>)
 * @tparam T The underlying type
 * @tparam ChunkT The chunk type (zero_or_value::Chunk<T> or zero_any::Chunk<T>)
 *
 * Contains the logic specific to chunks with two constant values.
 */

template <typename Derived, typename T, typename ChunkT>
class SecondValueProxyBase : public ProxyBase<Derived, T, ChunkT>
{
  public:
    using Base = ProxyBase<Derived, T, ChunkT>;
    friend Base;

    using Base::array_id_;
    using Base::chunk_;
    using Base::chunk_index_;
    using Base::elem_offset_;
    using Base::is_first_value;
    using Base::is_not_first_value;
    using Base::queue_as_candidate;
    using Base::values_;

    using typename Base::AtomicPtrRefT;
    using ValuesTypeRef = AtomicPtrRefT;

    // Both members exist on ProxyBase in every mode (the serial specialisation
    // declares them too); only their USE is mode-specific.
    using Base::candidate_flag_;
    using Base::candidate_flags;

    // ========================================================================
    // Constant-state queries. is_second_value() itself is the derived proxy's:
    // only it knows the index/idx for non-bool types. These reach it through
    // self; the delegating is_second_value() this class used to declare for them
    // is gone.
    // ========================================================================

    /**
     * @brief Negation of is_const_val().
     * @return true if the chunk is materialised (compressed to neither constant).
     */
    [[nodiscard]] bool is_not_const_val(this const Derived& self) noexcept
    {
        return !self.is_const_val();
    }

    /**
     * @brief Multiply this element by @p operand in place.
     * @tparam U Type of the operand.
     * @param operand Multiplier.
     * @return This proxy. Converting it to T reads the resulting value,
     *         synchronised (operator T() -> read_pure()); discarding it,
     *         as `a[i] += x;` does, keeps the operation fully deferrable.*/
    template <typename U>
        requires ElementMultipliable<T>
    Derived operator*=(this auto&& self, const U& operand) noexcept
    {
        // Early-out: multiplying by 1 writes nothing. Nothing to read either:
        // reading the returned proxy goes through operator T() -> read_pure()
        // and its read-side gate. BaseT<U> is the arithmetic type even if U
        // is a Proxy.
        if (operand == static_cast<BaseT<U>>(1))
        {
            return self;
        }

        self.template apply_compound_operation<U>(std::multiplies<>{}, operand);
        return self;
    }

    /**
     * @brief Divide this element by @p operand in place.
     * @tparam U Type of the operand.
     * @param operand Divisor. Integer element: check builds stop on operand == 0
     *        (internal::check_integer_divisor); a floating one gives inf/NaN.
     * @return This proxy. Converting it to T reads the resulting value,
     *         synchronised (operator T() -> read_pure()); discarding it,
     *         as `a[i] += x;` does, keeps the operation fully deferrable.*/
    template <typename U>
        requires ElementDivisible<T>
    Derived operator/=(this auto&& self, const U& operand) noexcept
    {
        internal::check_integer_divisor<T>(operand, "/=");

        // Early-out: dividing by 1 changes nothing -- nothing to write, and
        // nothing to read: reading the returned proxy goes through
        // operator T() -> read_pure() (see operator*=).
        if (operand == static_cast<BaseT<U>>(1))
        {
            return self;
        }
        self.template apply_compound_operation<U>(std::divides<>{}, operand);
        return self;
    }

    /**
     * @brief Compute this element modulo @p operand in place.
     * @tparam U Type of the operand.
     * @param operand Modulus. Check builds stop on operand == 0
     *        (internal::check_integer_divisor).
     * @return This proxy. Converting it to T reads the resulting value,
     *         synchronised (operator T() -> read_pure()); discarding it,
     *         as `a[i] += x;` does, keeps the operation fully deferrable.*/
    template <typename U>
        requires ElementModulable<T>
    Derived operator%=(this auto&& self, const U& operand) noexcept
    {
        if constexpr (std::is_integral_v<T>)
        {
            internal::check_integer_divisor<T>(operand, "%=");
            // A non-zero slot can land on zero (4 % 2): apply_compound_operation
            // counts a result equal to the first value, so the chunk can reach
            // "full first value" and fold.
            self.template apply_compound_operation<U>(typename Base::modulo{}, operand);
            return self;
        }
        else
        {
            CHUNK_LOG(error, general,
                      "operator not implemented (requires an integral type): the element is "
                      "left unchanged");
            return self;
        }
    }

  protected:
    /**
     * @brief Construct a proxy for element `elem_offset_` of `chunk_`.
     * @param values Reference to the chunk's values pointer.
     * @param elem_offset Element offset within the chunk.
     * @param chunk Owning chunk.
     */
    explicit SecondValueProxyBase(ValuesTypeRef values, int elem_offset, ChunkT& chunk) noexcept
        : Base(values, elem_offset, chunk)
    {
    }

    /**
     * @brief Negation of is_second_value().
     * @return true if the chunk is NOT compressed to the second constant.
     */
    [[nodiscard]] bool is_not_second_value(this const Derived& self) noexcept
    {
        return !self.is_second_value();
    }

    /**
     * @brief Whether the chunk is compressed to EITHER constant (first or second).
     * @return true if the chunk is compressed.
     */
    [[nodiscard]] bool is_const_val(this const Derived& self) noexcept
    {
        return self.is_first_value() || self.is_second_value();
    }

    /**
     * @brief Pre-check shared by every variant: the write is of a constant.
     * @details A write is worth the marking path (one read of the old value,
     *          then queue_as_candidate) when the value written is one of
     *          the chunk's two constants -- whatever the chunk's state.
     *          should_mark_for_compression() then keeps the writes that change
     *          the element (old != new), so a chunk is queued at most once per
     *          pass, on its first write of a constant over something else.
     *
     *          Not only a chunk ALREADY at one constant being written the
     *          other: a chunk materialised by a plain value (2.5 over first_value)
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
    pre_check_mark_for_compression_strict(InT<T> new_value, InT<T> first_value, InT<T> second_val) noexcept
    {
        return same_value(new_value, first_value) || same_value(new_value, second_val);
    }

    // --- Boolean overloads ---
    /**
     * @brief Marks block for compression based on a boolean new value
     * @param new_value The new boolean value
     * @note Uses false and true as constant values for comparison
     */
    void mark_for_compression_if_needed(this const Derived& self, bool new_value);

    /**
     * @brief Marks block for compression with custom constant values
     * @param second_val The second constant value
     * @param new_value The new value to be set
     * @param first_value The first constant value
     */
    void
    mark_for_compression_if_needed(this const Derived& self, InT<T> second_val, InT<T> new_value, InT<T> first_value)
    {
        const bool isv = self.passes_mark_pre_check(new_value, first_value, second_val);
        self.mark_for_compression_impl(self.Base::read_values(), new_value, isv, second_val, first_value);
    }

    /**
     * @brief Marking for a caller that stores AFTER marking (zero_any::Proxy's
     *        assignment: mark, then perform_core_assignment).
     * @details Same test as mark_for_compression_if_needed() above, except the
     *          memory check of mark_for_compression_core() expects the OLD
     *          value. The helper above expects the new one -- right for a
     *          caller that has already stored (memory then differs only if a
     *          race changed it), but before the store memory still holds the
     *          old value, so its check would fire on every write that changes
     *          the value and queue the chunk whatever should_mark said.
     * @param second_val  The second constant value
     * @param new_value   The value about to be stored
     * @param first_value The first constant value
     * @param vals The chunk's values pointer, loaded by the caller inside the
     *        write session: the old value and the race check read through it
     *        instead of reloading it (see zero_any::Proxy::handle_no_bool_assignment).
     */
    void mark_for_compression_before_store(this const Derived& self,
                                           InT<T>              second_val,
                                           InT<T>              new_value,
                                           InT<T>              first_value,
                                           const T*            vals);

    /**
     * @brief Marks block for compression with an explicit old boolean value
     *        (provided to avoid a race), rather than reading it back from
     *        values[elem_offset].
     * @param old_value The previous boolean value.
     * @param new_value The new boolean value.
     */
    void mark_for_compression_if_needed(this const Derived& self, bool old_value, bool new_value)
    {
        const bool isv = self.passes_mark_pre_check(new_value, false, true);
        self.mark_for_compression_impl(old_value, new_value, isv, true, false);
    }

    /**
     * @brief Marks block for compression with explicit old and new values
     * @param second_val The second constant value
     * @param old_value The previous value (provided explicitly to avoid race)
     * @param new_value The new value to be set
     * @param first_value The first constant value
     */
    void mark_for_compression_if_needed(this const Derived& self,
                                        InT<T>              second_val,
                                        InT<T>              old_value,
                                        InT<T>              new_value,
                                        InT<T>              first_value)
    {
        const bool isv = self.passes_mark_pre_check(new_value, first_value, second_val);
        self.mark_for_compression_impl(old_value, new_value, isv, second_val, first_value);
    }

    /**
     * @brief Write operation with metadata initialization.
     * @tparam WriteFn Callable type for write operation.
     * @param write_fn Function to execute after metadata setup.
     * @details Combines fast/slow path locking with automatic metadata
     *          initialization (set_array_id_ptr_is_inside_chunk(), needed by
     *          the fast-block-compression path). Useful for operations that
     *          need array/chunk information before writing.
     */
    template <typename WriteFn>
    [[nodiscard]] CHUNK_FORCE_INLINE bool write_with_optional_lock_with_set_ptr(this Derived& self,
                                                                                WriteFn       write_fn) noexcept;

    /**
     * @brief Static entry point for async callers that only have a proxy
     *        reference in hand (not `this`). Delegates to the instance
     *        method of the same name.
     * @param proxy Proxy whose compression marking to trigger.
     * @param old_value Value before the write.
     * @param new_value Value after the write.
     */
    static void
    mark_for_compression_if_needed_with_value_async(const Derived& proxy, InT<T> old_value, InT<T> new_value)
    {
        proxy.mark_for_compression_if_needed_with_value(old_value, new_value);
    }

  private:
    static constexpr bool is_bool     = is_bool_like_v<T>; // bool and Bit (bit.h)
    static constexpr bool is_not_bool = !is_bool;

    /**
     * @brief Does this write pass the variant's marking pre-check?
     * @details The `isv` filter of every marking entry point: the CRTP call of
     *          Derived::pre_check_mark_for_compression(), which this class
     *          defaults to pre_check_mark_for_compression_strict() -- a constant
     *          is written, WHATEVER the chunk's state (zero_or_value adds that
     *          the rank has a candidate list for the value; zero_any and
     *          pattern widen it). It does not look at the chunk's state: a
     *          chunk materialised by a plain value and written back to a
     *          constant element by element must be queued too.
     * @note Invariant: it never rejects a write that
     *       should_mark_for_compression() would accept. It only spares the
     *       other writes the old-value read; a qualifying write it rejected
     *       would leave its chunk materialised for the run.
     * @param new_value The new value to check.
     * @param first_value The first constant value to compare against.
     * @param second_val The second constant value to compare against.
     * @return true if the write is worth the marking path.
     */
    [[nodiscard]] bool
    passes_mark_pre_check(this const Derived& self, InT<T> new_value, InT<T> first_value, InT<T> second_val) noexcept;

    /**
     * @brief Default pre-check hook (zero_or_value behaviour).
     * @details Variants that widen should_mark_for_compression() MUST shadow
     *          this method and widen the pre-check to match -- otherwise the
     *          extra predicate clauses are unreachable.
     * @param new_value The new value to check.
     * @param first_value The first constant value to compare against.
     * @param second_val The second constant value to compare against.
     * @return true if the write is worth examining further.
     */
    [[nodiscard]] bool
    pre_check_mark_for_compression(InT<T> new_value, InT<T> first_value, InT<T> second_val) const noexcept
    {
        return pre_check_mark_for_compression_strict(new_value, first_value, second_val);
    }

    /**
     * @brief Core unified logic for 2-value compression marking.
     * @details When `isv` is true, checks whether this write's transition
     *          qualifies for compression (should_mark_for_compression())
     *          or whether the in-memory value no longer matches
     *          `expected_in_memory` (a race with a concurrent writer); either
     *          way, adds this element to the compression candidate list.
     *          `isv` itself is a fast pre-check meant to let obviously
     *          irrelevant writes skip that work entirely (see
     *          passes_mark_pre_check()).
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
    void mark_for_compression_core(this const Derived& self,
                                   InT<T>              old_value,
                                   InT<T>              new_value,
                                   InT<T>              expected_in_memory,
                                   bool                isv,
                                   Args&&... compression_args);

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
    void
    mark_for_compression_impl(this const Derived& self, InT<T> old_value, InT<T> new_value, bool isv, Args&&... args)
    {
        self.mark_for_compression_core(old_value, new_value, new_value, isv, std::forward<Args>(args)...);
    }
};

} // namespace shrink_array

#include "chunk/internal/second_value_proxy_base_inline.h" // the template bodies