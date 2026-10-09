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

#include <concepts>   // std::assignable_from
#include <functional> // std::plus, std::minus, std::multiplies, std::divides
#include <ostream>    // std::ostream
#include <utility>    // std::pair
#include <vector>     // std::vector

#include "chunk/internal/pointer_map_linear.h" // linear scan shared with PointerMapChunk2
#include "chunk/config.h"                      // chunk_config::use_fast_block_compression, build_config::chunk_dynamic
#include "chunk/internal/array_chunk_for.h"    // ArrayChunkFor_t
#include "chunk/internal/chunk_size_policy.h"  // set_thread_chunk_size, set_thread_alloc_key
#include "chunk/internal/concept_is_clean_assignement.h" // IsCleanAssignment, ElementAddable, ElementModulable
#include "chunk/internal/in_t.h"                         // in_t<T>
#include "chunk/internal/invariant_check.h"              // CHUNK_INVARIANT_FAILED
#include "chunk/internal/conditional_counter.h"          // counter_value_t — resync_nb_val_eq_zero
#include "chunk/internal/def_force_inline.h"             // CHUNK_FORCE_INLINE
#include "chunk/internal/proxy_base.h"                   // ProxyBase
#include "chunk/internal/read_reclaim.h"                 // read_reclaim::ReadSection
#include "chunk/internal/chunk_stats.h" // increm_chunk_read<> (a dependent call the include tools cannot attribute)
#include "chunk/internal/immortal.h"    // Immortal / LazyImmortal: statics safe for global arrays

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace shrink_array
{

template <template <typename> class CHUNK_T, typename T>
class ManagerChunk;

/**
 * @brief Registry mapping array-element keys (pointers) to their table
 *        number. Unlike PointerMapChunk2, it has no value -> inf_array
 *        lookup: v1 compression does not group chunks by value.
 * @tparam T Element type this registry is keyed on.
 */
template <typename T>
class PointerMapChunk : public PointerMapLinear<int, std::vector<std::pair<const void*, int>>>
{
  public:
    /**
     * @brief Register @p key with table number @p num_tab.
     * @param key Array address used as the key.
     * @param num_tab Table number associated with the key.
     */
    void insert(void* key, int num_tab)
    {
        vec.emplace_back(key, num_tab);
    }

    /**
     * @brief Look up the table number registered for a key.
     * @param key Key to look up.
     * @return The registered table number.
     * @note A key that is not registered is an invariant failure (CHUNK_INVARIANT_FAILED: message, then abort).
     */
    [[nodiscard]] int get_num_tab(const void* key) const noexcept
    {
        for (const auto& [k, v] : vec)
        {
            if (k == key)
            {
                return v;
            }
        }

        CHUNK_INVARIANT_FAILED("[PointerMapChunk get_num_tab] key NOT FOUND: this={} size={} key={}",
                               static_cast<const void*>(this), vec.size(), static_cast<const void*>(key));
    }

    /**
     * @brief Swap the table numbers registered for two existing keys.
     * @details The name the v1 / v2A re-chunk hooks call; v2 calls
     *          swap_entries(). Same operation.
     * @param key_a First key (an array address).
     * @param key_b Second key.
     * @return false if either key is absent (nothing is changed).
     */
    bool swap_keys(const void* key_a, const void* key_b) noexcept
    {
        return this->swap_entries(key_a, key_b);
    }

    // erase(), swap_entries(), exists(), size() and for_each_key() come from
    // PointerMapLinear: they were the same traversals as in PointerMapChunk2.

  private:
    /// What a key maps to: its table number.
    using inf_array_t = int;

    // The storage (a std::vector) and the traversal operations come from
    // PointerMapLinear. The container is a template parameter there:
    // PointerMapChunk2's is a std::deque, for a reason documented over there.
    using Linear = PointerMapLinear<inf_array_t, std::vector<std::pair<const void*, inf_array_t>>>;
    /// Brings the base's storage back into this class's scope (dependent base).
    using Linear::vec;
};

/**
 * @brief Policy to handle Fast Block Compression specific logic for Proxies.
 */
template <typename Derived, typename T, bool Enabled>
class ProxyFastCompPolicy : public ProxyBase<Derived, T, Chunk<T>>
{
  public:
    using ChunkType     = Chunk<T>;
    using manager_chunk = ManagerChunk<Chunk, T>;
    using Base          = ProxyBase<Derived, T, ChunkType>;

    using CompressionManagerType = CompressionManager<T, ChunkType>;

    using values_type_ref = Base::atomic_ptr_ref_t;

    // --- fast-block-compression surface: this policy's Enabled is that axis
    // (chunk_config::use_fast_block_compression).
    using Base::num_tab;
    using Base::ptr_on_array;
    using Base::read_values;
    using Base::read_values_async;
    using Base::read_values_at;
    using Base::tab_is_inside_chunk_list_to_compress;

    using ProxyBase<Derived, T, Chunk<T>>::ProxyBase;

    /**
     * @brief Access to thread `thr`'s compression-candidate list.
     * @param thr Thread rank whose candidate list to fetch.
     * @return Reference to thread thr's candidate list.
     */
    auto& get_vec_tuple_to_compress_potentially_thr(int thr) const noexcept
    {
        return manager_chunk::chunk_to_compress_potentially_thr[thr];
    }

    /**
     * @brief Check whether this thread's compression-candidate list has
     *        grown past the "dangerous" size threshold, and if so, flag it
     *        (CompressionManagerType::mark_list_dangerous_for_thread()) so
     *        other machinery reacts (e.g. prioritising a drain).
     * @param thr_rank Thread whose candidate list to check.
     */
    void trigger_compression_if_needed(int thr_rank) noexcept;

    /**
     * @brief Decide whether NOW is a good time to run a compression pass
     *        for this thread, and if so, trigger it.
     * @details Checks the candidate-list size against the trigger
     *          threshold; if crossed and no deferred queue is already
     *          active for this thread, activates the slow write-compress
     *          path (when running multi-threaded) and calls
     *          Chunk<T>::make_block_compression() directly.
     */
    void perform_compression(this const Derived& self) noexcept;

    /**
     * @brief Async-context counterpart of execute_arithmetic_compound_operation().
     * @tparam OP Binary operator (e.g. std::plus<>, std::minus<>).
     * @tparam LBD Callable used only on the "chunk was not written" fast path.
     * @tparam U Type of the operand.
     * @param ctx Async write context.
     * @param op The operator to apply.
     * @param v Operand to combine with the current value.
     * @param lbd Called (with `op`) when the chunk was not yet written.
     */
    template <typename OP, typename LBD, typename U>
    void execute_arithmetic_compound_operation_async(auto* ctx, OP op, [[maybe_unused]] const U& v, LBD lbd) const;

    /**
     * @brief Resolve and cache this proxy's table number from ptr_on_array.
     */
    void set_num_tab() noexcept
    {
        // Armed by the array's operator[] (see Proxy2A::set_num_tab): the
        // registry lookup only for a proxy armed otherwise.
        if (num_tab >= 0) [[likely]]
        {
            return;
        }
        num_tab = Derived::get_num_tab(ptr_on_array);
    }

    /**
     * @brief Post-write entry point for callers that already have both the
     *        old and new value in hand.
     * @param old_value Value before the write.
     * @param new_value Value after the write.
     */
    void mark_for_compression_if_needed_with_value(this const Derived& self, in_t<T> old_value, in_t<T> new_value)
    {
        // Unconditional: mark_for_compression_impl carries the race check, which
        // must run on every write.
        self.mark_for_compression_impl(old_value, new_value);
    }

    /**
     * @brief mark_for_compression_if_needed_with_value() for a write body
     *        that holds the chunk's values pointer: the race check reads
     *        through @p vals instead of reloading it.
     * @param vals Loaded inside the write session (or the replay), after the
     *        materialisation if there was one -- no compressor can change it
     *        until the session ends, and no writer either once it is not the
     *        constant (see Proxy::assign_impl).
     * @param old_value Value before the write.
     * @param new_value Value after the write.
     */
    void mark_for_compression_if_needed_with_value(this const Derived& self,
                                                   in_t<T>             old_value,
                                                   in_t<T>             new_value,
                                                   const T*            vals)
    {
        self.mark_for_compression_core(old_value, new_value, new_value, vals);
    }

    /**
     * @brief Async-context version of the table-number lookup: resolves
     *        which array a given chunk pointer belongs to.
     * @param ptr Chunk pointer to look up.
     * @return The table number @p ptr belongs to.
     */
    [[nodiscard]] static int get_num_tab_async(auto* ctx, [[maybe_unused]] const void* ptr) noexcept
    {
        // The context's num_tab, captured at enqueue: the one the replay proxy
        // is armed with (make_proxy_with_info), without a registry scan per
        // replayed op. Check builds compare it with the registry while the
        // array is still registered.
        if constexpr (build_config::any_check)
        {
            if (Derived::share_info_array->exists(ptr) && Derived::share_info_array->get_num_tab(ptr) != ctx->num_tab)
            {
                CHUNK_INVARIANT_FAILED("[Proxy get_num_tab_async] ctx num_tab={} but the registry says {} for {}",
                                       ctx->num_tab, Derived::share_info_array->get_num_tab(ptr), ptr);
            }
        }
        return ctx->num_tab;
    }

    /**
     * @brief Async-context counterpart of execute_inplace_operation().
     * @tparam LAMBDA Callable performing the actual in-place modification.
     * @param ctx Async write context.
     * @param operation The operation to run.
     */
    template <typename LAMBDA>
    static void execute_inplace_operation_async(auto* ctx, LAMBDA operation);

    /**
     * @brief Async-context counterpart of execute_simple_operation():
     *        materialise the chunk referenced by `ctx` if it is currently
     *        compressed, then run `operation` on the reloaded values pointer.
     * @tparam LAMBDA Callable performing the actual write.
     * @param ctx Async write context.
     * @param operation The write to perform.
     */
    template <typename LAMBDA>
    static void execute_simple_operation_async(auto* ctx, LAMBDA operation);

    /**
     * @brief Clear one chunk's "queued for compression" flag, so a later
     *        write can queue it again.
     * @param num_tab Which table the chunk belongs to.
     * @param num_chunk The chunk's index within that table.
     *
     * @warning Deliberately NOT removed in favour of the identically named
     *          ProxyBase one, which this hides. They differ where it matters:
     *          the base stores with memory_order_release, this one with a plain
     *          assignment, i.e. seq_cst. Dropping this override would weaken v1's
     *          ordering, which is a decision about the flag's visibility, not a
     *          deduplication.
     */
    static void reset_tab_inside_list_to_compress(int num_tab, int num_chunk) noexcept
    {
        tab_is_inside_chunk_list_to_compress[num_tab][num_chunk] = 0;
    }

  protected:
    /**
     * @brief Register this chunk's table number and compression-list flag
     *        pointer, so later compression bookkeeping can find its way
     *        back to this chunk.
     */
    void set_info_on_chunk(this Derived& self) noexcept
    {
        // Qualified: this policy's no-arg overload (above); Proxy's `using
        // Base::set_num_tab` would hide it from an unqualified self.set_num_tab().
        self.ProxyFastCompPolicy::set_num_tab();
        self.Base::set_ptr_is_inside_chunk_list_to_compress();
    }

  private:
    constexpr static T zero{};

    /**
     * @brief Core compression-marking logic. Must run on EVERY write, not
     *        just zero-transitions: should_mark_for_compression() is tested
     *        internally, but this function ALSO carries the data-race check,
     *        which needs to run regardless of whether this particular write
     *        is itself a zero-transition.
     * @details Adds this element to the compression candidate list if it is
     *          a zero-transition OR if a race was detected (the in-memory
     *          value no longer matches `expected_in_memory`).
     * @param old_value Value before the write.
     * @param new_value Value after the write.
     * @param expected_in_memory The value this write expects to still be in
     *                           memory (mismatch means a concurrent writer
     *                           raced this one).
     */
    void mark_for_compression_core(this const Derived& self,
                                   in_t<T>             old_value,
                                   in_t<T>             new_value,
                                   in_t<T>             expected_in_memory);

    /// @brief mark_for_compression_core() reading memory through @p vals, a
    ///        values pointer the caller loaded inside the write session.
    void mark_for_compression_core(this const Derived& self,
                                   in_t<T>             old_value,
                                   in_t<T>             new_value,
                                   in_t<T>             expected_in_memory,
                                   const T*            vals);

    // Wrapper for post-write check: we expect the 'new_value' to be in memory

    /**
     * @brief Post-write wrapper around mark_for_compression_core(), passing
     *        new_value as both the write result and the expected-in-memory
     *        value.
     * @param old_value Value before the write.
     * @param new_value Value after the write.
     */
    void mark_for_compression_impl(this const Derived& self, in_t<T> old_value, in_t<T> new_value)
    {
        // Unconditional, in every build, on EVERY write -- not only under
        // CHUNK_STATS, nor only on zero-transitions: mark_for_compression_core
        // carries the data-race check (read_values() != expected_in_memory) and
        // already tests should_mark_for_compression itself.
        self.mark_for_compression_core(old_value, new_value, new_value);
    }

    /**
     * @brief Static entry point for async callers that only have a proxy
     *        reference in hand (not `this`). The marking logic itself needs
     *        no async-specific context, so this simply delegates to the
     *        instance method.
     * @param proxy Proxy whose compression marking to trigger.
     * @param old_value Value before the write.
     * @param new_value Value after the write.
     */
    static void mark_for_compression_if_needed_with_value_async(const Derived& proxy, T old_value, T new_value)
    {
        proxy.mark_for_compression_if_needed_with_value(old_value, new_value);
    }

    /**
     * @brief Whether a write transition is a compression candidate (a
     *        zero-transition: was non-zero, now zero).
     * @param old_value Value before the write.
     * @param new_value Value after the write.
     * @return true if old_value != zero and new_value == zero.
     */
    [[nodiscard]] static bool should_mark_for_compression(in_t<T> old_value, in_t<T> new_value) noexcept
    {
        return old_value != zero && new_value == zero;
    }
};

/**
 * @brief Specialization for Disabled Fast Compression (Zero Overhead).
 */
template <typename Derived, typename T>
class ProxyFastCompPolicy<Derived, T, false> : public ProxyBase<Derived, T, Chunk<T>>
{
  public:
    using ProxyBase<Derived, T, Chunk<T>>::ProxyBase;
    using ProxyBase<Derived, T, Chunk<T>>::chunk;

  protected:
    /// @brief Nothing to record on the chunk without fast block compression.
    static void set_info_on_chunk() {}

    /**
     * @brief Counter-mode: called when the per-chunk zero-counter reaches
     *        the chunk size (every element is now zero) -- compresses the
     *        chunk.
     * @details Counter mode is exactly the non-fast-block-compression mode
     *          (use_compression_with_counter == !use_fast_block_compression),
     *          so it lives on this specialisation; the call sites in Proxy
     *          sit under `if constexpr (use_compression_with_counter)`.
     */
    void manage_full_zero() const
    {
        chunk.clear_st();
    }
};

#ifdef __clang__
    #define CHUNK_INHERIT_CTOR_TYPENAME typename
#else
    #define CHUNK_INHERIT_CTOR_TYPENAME
#endif

/*!
 * \brief Lazy-write proxy returned by Chunk's non-const operator[].
 *
 * Lets `chunk[i] = value` avoid an unnecessary allocation when `value` is
 * the chunk's compressible constant: rather than eagerly materialising the
 * chunk on every write, the proxy defers to operator=, which can detect a
 * write of the compressible value and route it into the compression-marking
 * path instead. Debug builds additionally check that assigning that value
 * to an already-compressed chunk never triggers a wasted allocation.
 */

template <typename T>
class Proxy : public ProxyFastCompPolicy<Proxy<T>, T, chunk_config::use_fast_block_compression>
{
  public:
    /// @brief Stream output of the element's value (defined below).
    template <typename U>
    friend std::ostream& operator<<(std::ostream& os, const Proxy<U>& proxy);

    using ChunkType = Chunk<T>;
    using Base      = ProxyBase<Proxy, T, Chunk<T>>;

    using CompressionManagerType = CompressionManager<T, ChunkType>;
    using FastCompPolicy         = ProxyFastCompPolicy<Proxy, T, chunk_config::use_fast_block_compression>;

    friend Base;
    friend FastCompPolicy;

    using FastCompPolicy::set_info_on_chunk;

    using Base::add_to_compression_list;
    using Base::set_num_tab;

    using Base::ptr_is_inside_chunk_list_to_compress; ///< Per-chunk compression flag pointer
    using Base::tab_is_inside_chunk_list_to_compress; ///< Global compression tracking array

    /// @brief Functor wrapping operator% for apply_compound_operation().
    struct modulo
    {
        /// @brief Apply operator% to the two operands.
        /// @return lhs % rhs.
        static constexpr T operator()(in_t<T> lhs, in_t<T> rhs) noexcept
        {
            return lhs % rhs;
        }
    };

  private:
    using manager_chunk = ManagerChunk<Chunk, T>;

    using Base::read_values;
    using Base::values;

    using Base::num;
    /// Reference to the associated chunk.
    ///
    using Base::chunk; ///< owning chunk

    using Base::num_tab; ///< tab index for this array

    using Base::chunk_size_frozen_; ///< non-zero on a proxy rebuilt from a replay context
    using Base::num_chunk;          ///< chunk index
    using Base::ptr_on_array;       ///< pointer to array (for tab lookup)

    using Base::arithmetic_assign;

    using values_type_ref = Base::atomic_ptr_ref_t;

    constexpr static T zero{};

    static constexpr bool is_bool     = std::is_same_v<T, bool>;
    static constexpr bool is_not_bool = !is_bool;

    // reset_tab_inside_list_to_compress(), set_num_tab() and the
    // mark_for_compression_* family live on ProxyFastCompPolicy<..., true>;
    // manage_full_zero() on the <..., false> (counter) specialisation.

    /**
     * @brief Whether the owning chunk is currently compressed to its
     *        constant.
     * @return true if the chunk is compressed.
     */
    [[nodiscard]] bool is_cst_val() const noexcept
    {
        return chunk.is_cst_val1();
    }

    /**
     * @brief Materialise the owning chunk (allocate + init), pinning the
     *        active chunk size for this array first in dynamic builds.
     */
    void create_value() const noexcept
    {
        if constexpr (build_config::chunk_dynamic)
        {
            // Route this (possibly worker-thread) allocation to the allocator
            // for THIS array's chunk size before materializing.
            shrink_array::set_thread_chunk_size(this->chunk_size());
            shrink_array::set_thread_alloc_key(this->alloc_key());
        }
        chunk.create_value();
    }

    /**
     * @brief Materialise the owning chunk only if `need_alloc` is true (the
     *        chunk is currently compressed).
     * @param need_alloc Whether materialisation is actually needed.
     */
    void create_value(bool need_alloc) const noexcept
    {
        if (need_alloc)
        {
            create_value();
        }
    }

    /**
     * @brief Run `write_fn` under the write-synchronisation protocol,
     *        registering this chunk's compression-tracking info first
     *        (fast-block-compression builds only).
     * @tparam WriteFn Callable performing the actual write.
     * @param write_fn The write to perform.
     * @return Whatever Base::write_with_optional_lock_impl() returns.
     */
    template <typename WriteFn>
    CHUNK_FORCE_INLINE bool write_with_optional_lock(WriteFn write_fn) noexcept
    {
        const auto lbd_info = [this]
        {
            if constexpr (chunk_config::use_fast_block_compression)
            {
                set_info_on_chunk();
            }
        };

        // wrap the call in a lambda that calls write_fn(v)
        return Base::write_with_optional_lock_impl([&]() noexcept { write_fn(); }, lbd_info);
    }
    /**
     * @brief Perform the actual write: install `v` at this proxy's index,
     *        materialising the chunk first if it is currently compressed.
     * @param vals The chunk's values pointer, loaded by the caller inside the
     *        write session; it decides whether to materialise.
     * @param v Value to write.
     * @return The pointer the store went through: @p vals, or the one loaded
     *         again after the materialisation.
     */
    T* perform_core_assignment(T* vals, in_t<T> v) const noexcept;

    /**
     * @brief Shared implementation behind operator+=/operator-=: materialise
     *        the chunk if needed, then apply `op` in a common high-precision
     *        type before truncating back to T.
     * @tparam OP Binary operator (e.g. std::plus<>, std::minus<>).
     * @tparam LBD Callable used only on the "chunk was not written" fast path.
     * @tparam U Type of the operand being combined with the current value.
     * @param op The operator to apply.
     * @param v Operand to combine with the current value.
     * @param lbd Called (with `op`) when the chunk was not yet written, right
     *            after materialising it.
     */
    template <typename OP, typename LBD, typename U>
    void execute_arithmetic_compound_operation(OP op, [[maybe_unused]] const U& v, LBD lbd);

    /**
     * @brief Run an arbitrary in-place operation on this element,
     *        materialising the chunk first if it is currently compressed,
     *        and maintaining the counter-mode zero-tracking bookkeeping
     *        around it.
     * @tparam LAMBDA Callable performing the actual in-place modification.
     * @param operation The operation to run.
     */
    template <typename LAMBDA>
    void execute_inplace_operation(LAMBDA operation);

    /**
     * @brief Early-out check for operator=(): is the chunk already
     *        compressed AND is the value being assigned the compressible
     *        constant? If so, the write is a genuine no-op.
     * @tparam U Type of the value being assigned.
     * @param v Value being assigned.
     * @return true if v == zero and the chunk is already compressed.
     * @note Never while the chunk has a deferred op pending (pending_any()):
     *       that op, older than this write in program order, would replay
     *       AFTER it -- the engine's WAW guard retires it first, and this
     *       early-out sits before the engine. Constant-false serially.
     */
    template <typename U>
    [[nodiscard]] bool is_cst_val_and_v_equal_zero(const U& v) const noexcept
    {
        return v == zero && is_cst_val() && !chunk.pending_any();
    }

    /**
     * @brief Core implementation behind operator=(): writes `v` at this
     *        proxy's index, with an early-out when the chunk is already
     *        compressed to the value being assigned, and (in counter-mode
     *        builds) maintains the per-chunk zero-element counter around
     *        the write.
     * @tparam U Type of the value being assigned.
     * @param v Value to write.
     * @return The value that was written.
     */
    template <typename U>
    // in_t<U>, not const U&: this is the write engine, the one frame the
    // operators do NOT inline into the caller, so its parameter is what
    // crosses a call -- a char by value stays in a register, a struct goes by
    // reference. Non-deducible, hence the explicit assign_impl<U>(v) below.
    T assign_impl(in_t<U> v) noexcept;

    /**
     * @brief Run an arbitrary write operation on this element, materialising
     *        the chunk first if it is currently compressed, and maintaining
     *        the counter-mode zero-tracking bookkeeping around it (non-OMP
     *        path; see execute_simple_operation_async() on
     *        ProxyFastCompPolicy for the async counterpart).
     * @tparam LAMBDA Callable performing the actual write.
     * @param operation The write to perform.
     */
    template <typename LAMBDA>
    void execute_simple_operation(LAMBDA operation);

    /**
     * @brief Shared implementation behind operator/=, operator*=, and
     *        operator%=.
     * @details First pulls any foreign deferred write via
     *          chunk.reader_sync(). In counter-mode builds, the operation
     *          is skipped entirely when the current value is already zero
     *          (multiplying, dividing, or taking the modulo of zero always
     *          yields zero, so there is nothing to compute or compress);
     *          otherwise it runs through
     *          execute_arithmetic_compound_operation() and updates the
     *          zero-element counter afterwards. In non-counter
     *          (deferred/OMP) builds, it instead runs
     *          Base::apply_atomic_op() directly, unless the value is zero
     *          AND no deferred queue is active for this thread (in which
     *          case the result is still zero and there is nothing to do).
     * @tparam U Type of the operand.
     * @tparam BinaryOp Binary operator to apply (divides/multiplies/modulo).
     * @param v Operand to combine with the current value.
     * @param op The operator to apply.
     */
    template <typename U, typename BinaryOp>
    void apply_compound_operation(in_t<U> v, BinaryOp op); // in_t: see assign_impl

  public:
    using CHUNK_INHERIT_CTOR_TYPENAME FastCompPolicy::ProxyFastCompPolicy;

    /**
     * @brief Construct a proxy for element `num_` of `chunk_`.
     * @param values_ Reference to the chunk's values pointer.
     * @param num_ Element offset within the chunk.
     * @param chunk_ Owning chunk.
     */
    explicit Proxy(values_type_ref values_, int num_, Chunk<T>& chunk_) noexcept : FastCompPolicy(values_, num_, chunk_)
    {
    }

    Proxy(const Proxy& other)     = default;
    Proxy(Proxy&& other) noexcept = default;

    /** @brief Clean path: exact match or safe conversion (small operand, by value). */
    template <typename U>
        requires IsCleanAssignment<T, U>
    T operator=(const U& v) noexcept
    {
        return assign_impl<U>(v);
    }

    /** @brief Trap path: risky conversion (e.g. float = double). */
    template <typename U>
        requires(!IsCleanAssignment<T, U>) && std::assignable_from<T&, U>
    [[deprecated("HPC WARNING: Type mismatch in assignment. Use 'f' suffix for floats.")]]
    T operator=(const U& v) noexcept
    {
        return assign_impl<T>(static_cast<T>(v));
    }

    /**
     * @brief Proxy-to-proxy assignment: read `from`'s current value through
     *        operator T() (gated read) and write it into this proxy's slot
     *        through assign_impl() (the full write engine).
     * @param from Proxy to read the value from.
     * @return *this.
     */
    const Proxy& operator=(const Proxy& from) noexcept;

    /**
     * @brief Negation of is_cst_val().
     * @return true if the chunk is materialised (not compressed).
     */
    [[nodiscard]] bool is_not_cst_val() const noexcept
    {
        return chunk.is_not_cst_val1();
    }

    /**
     * @brief Re-state the owning chunk's zero count after a BULK fill that did
     *        not go through this write path.
     *
     * @details nb_val_eq_zero is maintained one write at a time by
     *          perform_core_assignment(), and manage_full_zero() fires on the
     *          exact equality `nb_val_eq_zero == chunk_size`. Anything that
     *          stores into the chunk's buffer directly therefore leaves the
     *          counter describing a buffer that no longer exists, and the drift
     *          is silent until some later single-element write happens to land
     *          the count on chunk_size -- at which point a chunk that is not all
     *          zero is compressed to cst_val1 and reads back as the constant.
     *
     *          The counter lives in a private base of Chunk<T> whose only friend
     *          is this proxy, which is why the repair verb belongs here rather
     *          than at the bulk writer's call site.
     *
     *          The only such writer today is
     *          RechunkSupport::rechunk_refill()'s raw-fill; a no-op when
     *          counters are compiled out (set_nb_val_eq_zero() collapses).
     *
     * @param n_zeros number of zero cells over the chunk's FULL chunk_size()
     *        window -- the same window is_full_zero() is asked about, so the
     *        untouched tail of a partially-filled last chunk counts too.
     */
    void resync_nb_val_eq_zero(counter_value_t n_zeros) const noexcept
    {
        chunk.set_nb_val_eq_zero(n_zeros);
    }

    /**
     * @brief Add `v` to this element in place.
     * @tparam U Type of the operand.
     * @param v Operand to add.
     * @return This proxy. Converting it to T reads the resulting value,
     *         synchronised (operator T() -> read_pure()); discarding it,
     *         as `a[i] += x;` does, keeps the operation fully deferrable.*/
    template <typename U>
        requires ElementAddable<T>
    Proxy operator+=(const U& v) noexcept
    {
        this->template arithmetic_assign<U>(v, std::plus<>{});
        return *this;
    }

    /**
     * @brief Subtract `v` from this element in place.
     * @tparam U Type of the operand.
     * @param v Operand to subtract.
     * @return This proxy. Converting it to T reads the resulting value,
     *         synchronised (operator T() -> read_pure()); discarding it,
     *         as `a[i] += x;` does, keeps the operation fully deferrable.*/
    template <typename U>
        requires ElementSubtractable<T>
    Proxy operator-=(const U& v) noexcept
    {
        this->template arithmetic_assign<U>(v, std::minus<>{});
        return *this;
    }

    /**
     * @brief Divide this element by `v` in place.
     * @tparam U Type of the operand.
     * @param v Divisor. Integer element: check builds stop on v == 0
     *        (internal::check_integer_divisor); a floating one gives inf/NaN.
     * @return This proxy. Converting it to T reads the resulting value,
     *         synchronised (operator T() -> read_pure()); discarding it,
     *         as `a[i] += x;` does, keeps the operation fully deferrable.*/
    template <typename U>
        requires ElementDivisible<T>
    Proxy operator/=(const U& v) noexcept
    {
        internal::check_integer_divisor<T>(v, "/=");
        apply_compound_operation<U>(v, std::divides<>());
        return *this;
    }

    /**
     * @brief Multiply this element by `v` in place.
     * @tparam U Type of the operand.
     * @param v Multiplier.
     * @return This proxy. Converting it to T reads the resulting value,
     *         synchronised (operator T() -> read_pure()); discarding it,
     *         as `a[i] += x;` does, keeps the operation fully deferrable.*/
    template <typename U>
        requires ElementMultipliable<T>
    Proxy operator*=(const U& v) noexcept
    {
        apply_compound_operation<U>(v, std::multiplies<>());
        return *this;
    }

    /**
     * @brief Compute this element modulo `v` in place. Integral T only; a
     *        no-op passthrough (returns v unchanged) for non-integral T.
     * @tparam U Type of the operand.
     * @param v Modulus. Check builds stop on v == 0
     *        (internal::check_integer_divisor).
     * @return This proxy. Converting it to T reads the resulting value,
     *         synchronised (operator T() -> read_pure()); discarding it,
     *         as `a[i] += x;` does, keeps the operation fully deferrable.*/
    template <typename U>
        requires ElementModulable<T> || std::is_same_v<T, bool>
    Proxy operator%=(const U& v) noexcept
    {
        if constexpr (std::is_integral_v<T>)
        {
            internal::check_integer_divisor<T>(v, "%=");
            apply_compound_operation<U>(v, typename Base::modulo{});
            return *this;
        }
        else
        {
            return *this;
        }
    }

    /**
     * @brief Implicit read conversion: returns the element's current value.
     *
     * @details This is one of the read entries the non-const
     *          ArrayChunkBase::operator[] delegates visibility to: the write
     *          entry does not flush, so a read must pull any deferred write
     *          queued for this offset before looking at the buffer. A plain
     *          values[num] could hand back the value a SUPERSEDED round left
     *          there while the current round's write is still queued -- stale
     *          reads on every `arr[i]` taken on a non-const array. Proxy2 /
     *          Proxy2A inherit ProxyBase::operator T(), which is read_pure().
     *
     *          The gate is behind the SAME pre-filter as
     *          ArrayChunkBase::sync_chunk(): the per-chunk dirty bit is set by
     *          the deferred-enqueue path, so a clear bit means no deferred op
     *          was ever queued for this chunk and the fast read stands. Only
     *          chunks that actually took a deferred write pay read_pure()'s
     *          reader_sync. The bitset folds to a constant false when it is
     *          compiled out (serial builds), taking the whole branch with it.
     */
    operator T() const noexcept
    {
        this->check_not_stale();
        if constexpr (chunk_config::stats)
        {
            increm_chunk_read<ChunkType>();
        }

        if constexpr (build_config::use_openmp)
        {
            // chunk_size_frozen_ != 0 marks a proxy rebuilt from a replay
            // context, whose ptr_on_array may already be dangling (see
            // ProxyBase::chunk_size_frozen_): never dereference the array there.
            // Such a proxy is only ever used from inside a flush, where the
            // ops are being applied rather than waited on, so go straight to
            // the gated read.
            if (chunk_size_frozen_ == 0U && ptr_on_array != nullptr) [[likely]]
            {
                using ArrayType = ArrayChunkFor_t<ChunkType>;
                if (!static_cast<const ArrayType*>(ptr_on_array)->is_chunk_dirty(num_chunk)) [[likely]]
                {
                    // A pure read, like element_at: inside a ReadSection, or a
                    // pass in the region may release the buffer between the
                    // pointer load and the element load (read_reclaim.h).
                    const read_reclaim::ReadSection section;
                    // raw-read: inside a read_reclaim::ReadSection
                    return values[num];
                }
            }
            return Base::read_pure();
        }

        // raw-read: serial build (no OpenMP, no concurrent pass)
        return values[num];
    }

    // No address, and no operator T&() either: with one, `tab[nc] = val` was
    // routed through it and created blocks for nothing. The price: foo(a[i])
    // does not compile when foo takes a `T&`.
    T*       operator&()       = delete;
    const T* operator&() const = delete;

    /**
     * @brief Resolve which array (table) a given pointer belongs to.
     * @param ptr Pointer to look up.
     * @return The table number @p ptr belongs to.
     */
    [[nodiscard]] static int get_num_tab(const void* ptr) noexcept
    {
        return share_info_array->get_num_tab(ptr);
    }

    /**
     * @brief This proxy's table number: the armed one (operator[] arms it),
     *        the registry lookup otherwise.
     * @param array_ptr This proxy's array (ptr_on_array).
     * @return The table number of @p array_ptr.
     */
    [[nodiscard]] int get_num_tab_cached(const void* array_ptr) const noexcept
    {
        return this->num_tab >= 0 ? this->num_tab : get_num_tab(array_ptr);
    }

    /// Registry of every v1 array: array address -> table number.
    static inline constinit internal::LazyImmortal<PointerMapChunk<T>> share_info_array;
};

/// @brief Write the element's value, read through proxy.read_pure(), to @p os.
/// @return @p os.
template <typename T>
std::ostream& operator<<(std::ostream& os, const Proxy<T>& proxy)
{
    // Through the proxy's read entry, like operator T(): the raw values[num]
    // snapshot is stale while a deferred op pends on this offset (forced-
    // defer builds print 0 right after the write).
    os << proxy.read_pure();
    return os;
}

} // namespace shrink_array

#include "chunk/internal/proxy_inline.h" // the template bodies