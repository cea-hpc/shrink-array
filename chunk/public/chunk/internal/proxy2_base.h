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
#include <type_traits> // std::is_integral_v, std::is_same_v
#include <utility>     // std::forward

#include "chunk/bit.h"    // is_bool_like_v: bool and Bit share the bool path
#include "chunk/config.h" // chunk_config::use_fast_block_compression, build_config::do_check
#include "chunk/internal/concept_is_clean_assignement.h" // base_t, ElementMultipliable, ElementModulable
#include "chunk/internal/def_force_inline.h"             // CHUNK_FORCE_INLINE
#include "chunk/internal/in_t.h"                         // in_t<T>
#include "chunk/internal/proxy_base.h"                   // ProxyBase
#include "chunk/log.h"                                   // CHUNK_LOG (the operators bool has not)

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
    // Constant-state queries. is_cst_val2() itself is the derived proxy's:
    // only it knows the index/ind for non-bool types. These reach it through
    // self; the delegating is_cst_val2() this class used to declare for them
    // is gone.
    // ========================================================================

    /**
     * @brief Negation of is_cst_val().
     * @return true if the chunk is materialised (compressed to neither constant).
     */
    [[nodiscard]] bool is_not_cst_val(this const Derived& self) noexcept
    {
        return !self.is_cst_val();
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
    Derived operator*=(this auto&& self, const U& v) noexcept
    {
        if constexpr (is_bool)
        {
            CHUNK_LOG(error, general, "operator not implemented for bool: the element is left unchanged");
            return self;
        }
        else
        {
            // Early-out: multiplying by 1 writes nothing. Nothing to read either:
            // reading the returned proxy goes through operator T() -> read_pure()
            // and its read-side gate. base_t<U> is the arithmetic type even if U
            // is a Proxy.
            if (v == static_cast<base_t<U>>(1))
            {
                return self;
            }

            self.template apply_compound_operation<U>(std::multiplies<>{}, v);
            return self;
        }
    }

    /**
     * @brief Divide this element by `v` in place.
     * @tparam U Type of the operand.
     * @param v Divisor. Integer element: check builds stop on v == 0
     *        (internal::check_integer_divisor); a floating one gives inf/NaN.
     * @return This proxy. Converting it to T reads the resulting value,
     *         synchronised (operator T() -> read_pure()); discarding it,
     *         as `a[i] += x;` does, keeps the operation fully deferrable.
     *         Bool: not implemented -- prints an error and leaves the
     *         element unchanged.*/
    template <typename U>
        requires ElementDivisible<T>
    Derived operator/=(this auto&& self, const U& v) noexcept
    {
        if constexpr (is_bool)
        {
            CHUNK_LOG(error, general, "operator not implemented for bool: the element is left unchanged");
            return self;
        }
        else
        {
            internal::check_integer_divisor<T>(v, "/=");

            // Early-out: dividing by 1 changes nothing -- nothing to write, and
            // nothing to read: reading the returned proxy goes through
            // operator T() -> read_pure() (see operator*=).
            if (v == static_cast<base_t<U>>(1))
            {
                return self;
            }
            self.template apply_compound_operation<U>(std::divides<>{}, v);
            return self;
        }
    }

    /**
     * @brief Compute this element modulo `v` in place.
     * @tparam U Type of the operand.
     * @param v Modulus. Check builds stop on v == 0
     *        (internal::check_integer_divisor).
     * @return This proxy. Converting it to T reads the resulting value,
     *         synchronised (operator T() -> read_pure()); discarding it,
     *         as `a[i] += x;` does, keeps the operation fully deferrable.
     *         Bool: not implemented -- prints an error and leaves the
     *         element unchanged.*/
    template <typename U>
        requires ElementModulable<T> || is_bool_like_v<T>
    Derived operator%=(this auto&& self, const U& v) noexcept
    {
        if constexpr (std::is_integral_v<T> && is_not_bool)
        {
            internal::check_integer_divisor<T>(v, "%=");
            // A non-zero slot can land on zero (4 % 2): apply_compound_operation
            // counts a result equal to the first value, so the chunk can reach
            // "full first value" and fold.
            self.template apply_compound_operation<U>(typename Base::modulo{}, v);
            return self;
        }
        else
        {
            CHUNK_LOG(error, general,
                      "operator not implemented (requires an integral non-bool type): the element is "
                      "left unchanged");
            return self;
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
    [[nodiscard]] bool is_not_cst_val2(this const Derived& self) noexcept
    {
        return !self.is_cst_val2();
    }

    /**
     * @brief Whether the chunk is compressed to EITHER constant (first or second).
     * @return true if the chunk is compressed.
     */
    [[nodiscard]] bool is_cst_val(this const Derived& self) noexcept
    {
        return self.is_cst_val1() || self.is_cst_val2();
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
    void mark_for_compression_if_needed(this const Derived& self, bool new_value)
    {
        const bool isv = self.is_cst_val_new_value_is_other_cst_val(new_value, false, true);
        // Base::read_values() (qualified: dependent base), never values[num]:
        // it is the write path's read entry and the only one that resolves a
        // compressed chunk's constant (an unbacked identifier slot); indexing
        // reads a zero page and the decision below is taken against 0.
        self.mark_for_compression_impl(self.Base::read_values(), new_value, isv, false, true);
    }

    /**
     * @brief Marks block for compression with custom constant values
     * @param second_val The second constant value
     * @param new_value The new value to be set
     * @param first_value The first constant value
     */
    void
    mark_for_compression_if_needed(this const Derived& self, in_t<T> second_val, in_t<T> new_value, in_t<T> first_value)
    {
        const bool isv = self.is_cst_val_new_value_is_other_cst_val(new_value, first_value, second_val);
        self.mark_for_compression_impl(self.Base::read_values(), new_value, isv, second_val, first_value);
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
     * @param second_val  The second constant value
     * @param new_value   The value about to be stored
     * @param first_value The first constant value
     * @param vals The chunk's values pointer, loaded by the caller inside the
     *        write session: the old value and the race check read through it
     *        instead of reloading it (see Proxy2A::handle_no_bool_assignment).
     */
    void mark_for_compression_before_store(this const Derived& self,
                                           in_t<T>             second_val,
                                           in_t<T>             new_value,
                                           in_t<T>             first_value,
                                           const T*            vals)
    {
        // isv false makes the whole test a no-op: skip the read of the old
        // value too, which mark_for_compression_core() would not look at.
        if (!self.is_cst_val_new_value_is_other_cst_val(new_value, first_value, second_val))
        {
            return;
        }
        const T old = self.Base::read_values_at(vals);
        if (self.should_mark_for_compression(old, new_value, second_val, first_value) ||
            (self.Base::read_values_at(vals) != old)) [[unlikely]]
        {
            self.add_to_compression_list();
        }
    }

    /**
     * @brief Marks block for compression with an explicit old boolean value
     *        (provided to avoid a race), rather than reading it back from
     *        values[num].
     * @param old_value The previous boolean value.
     * @param new_value The new boolean value.
     */
    void mark_for_compression_if_needed(this const Derived& self, bool old_value, bool new_value)
    {
        const bool isv = self.is_cst_val_new_value_is_other_cst_val(new_value, false, true);
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
                                        in_t<T>             second_val,
                                        in_t<T>             old_value,
                                        in_t<T>             new_value,
                                        in_t<T>             first_value)
    {
        const bool isv = self.is_cst_val_new_value_is_other_cst_val(new_value, first_value, second_val);
        self.mark_for_compression_impl(old_value, new_value, isv, second_val, first_value);
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
    [[nodiscard]] CHUNK_FORCE_INLINE bool write_with_optional_lock_with_set_ptr(this Derived& self,
                                                                                WriteFn       write_fn) noexcept
    {
        const auto lbd_info = [&self]
        {
            if constexpr (chunk_config::use_fast_block_compression)
            {
                self.set_num_tab_ptr_is_inside_chunk(); // common initialization required for the fast-path
            }
            else
            {
                // The capture exists for the branch above only. A SERIAL build
                // never instantiates it and clang reports the capture as unused
                // (-Wunused-lambda-capture). Discarding `self` here keeps the
                // capture explicit; do not switch to [&], which hides the intent.
                static_cast<void>(self);
            }
        };

        return self.write_with_optional_lock_impl([&]() noexcept { write_fn(); }, lbd_info);
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
    [[nodiscard]] bool is_cst_val_new_value_is_other_cst_val(this const Derived& self,
                                                             in_t<T>             new_value,
                                                             in_t<T>             first_value,
                                                             in_t<T>             second_val) noexcept
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
        return self.pre_check_mark_for_compression(new_value, first_value, second_val);
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
     *          qualifies for compression (should_mark_for_compression())
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
    void mark_for_compression_core(this const Derived& self,
                                   in_t<T>             old_value,
                                   in_t<T>             new_value,
                                   in_t<T>             expected_in_memory,
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
    mark_for_compression_impl(this const Derived& self, in_t<T> old_value, in_t<T> new_value, bool isv, Args&&... args)
    {
        self.mark_for_compression_core(old_value, new_value, new_value, isv, std::forward<Args>(args)...);
    }
};

} // namespace shrink_array

#include "chunk/internal/proxy2_base_inline.h" // the template bodies