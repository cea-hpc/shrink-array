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

#include <atomic>   // std::atomic, std::memory_order_release
#include <concepts> // std::assignable_from
#include <cstddef>  // std::size_t
#include <memory>   // std::addressof (operator=: operator& is deleted)
#include <ostream>  // std::ostream
#include <vector>   // std::vector

#include "chunk/bit.h"                         // is_bool_like_v: bool and Bit share the bool path
#include "chunk/internal/in_t.h"               // InT<T>
#include "chunk/config.h"                      // chunk_config::use_fast_block_compression, build_config::chunk_dynamic
#include "chunk/internal/invariant_check.h"    // CHUNK_INVARIANT_FAILED
#include "chunk/PMP_CHUNK.h"                   // get_thr_rank_fast
#include "chunk/internal/pointer_map_chunk2.h" // PointerMapChunk2
#include "chunk/internal/concept_is_clean_assignement.h" // IsCleanAssignment
#include "chunk/internal/chunk_size_policy.h"            // set_thread_chunk_size
#include "chunk/internal/TChunk.h"                       // VectorTuple4ChunkInt2

#include "chunk/internal/chunk2_candidate_list_check.h" // chunk2_missing_candidate_list
#include "chunk/internal/proxy_base.h"                  // ProxyBase
#include "chunk/internal/proxy2_base.h"                 // Proxy2Base
#include "chunk/internal/chunk_stats.h" // increm_chunk_write<> in the bodies (a dependent call the include tools cannot attribute)
#include "chunk/internal/immortal.h"          // Immortal / LazyImmortal: statics safe for global arrays
#include "chunk/internal/second_value_info.h" // SecondValueInfo (InfArrayT)

// compression_manager.h (CompressionManagerType) comes unconditionally through proxy_base.h.

namespace shrink_array
{

template <template <typename> class ChunkT, typename T>
class ManagerChunk;

template <typename T>
class Chunk2;

/**
 * @brief Proxy2 class for managing operations on a Chunk2<T>.
 *
 * This class encapsulates read/write operations, compression logic,
 * and constant value handling within a chunk. It is optimized for
 * multithreaded environments and supports optional OpenMP integration.
 *
 * @tparam T The data type managed by the chunk.
 */

template <typename T, typename Derived>
// ind is set by load_inf_array() before
// any use; left unset here on purpose, the proxy is built on every element access.
// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
class ProxySpec2 : public Proxy2Base<Derived, T, Chunk2<T>>
{
  public:
    using ChunkType     = Chunk2<T>;
    using Base          = Proxy2Base<Derived, T, Chunk2<T>>;
    using ValuesTypeRef = Proxy2Base<Derived, T, ChunkType>::ValuesTypeRef;
    using InfArrayT     = SecondValueInfo<T>;

    using Base::Base;

    /**
     * @brief Initialize second_value and index without full metadata load
     * @details Lighter-weight alternative to set_info_on_chunk() when tab/chunk info
     *          is not immediately needed.
     * @post second_value and ind are valid; num_tab unchanged
     */

    void set_info_second_value_ind(this Derived& self);

    /**
     * @brief Like set_info_second_value_ind(), for an explicit array key instead of
     *        ptr_on_array (e.g. rechunk's tmp).
     * @param key Array pointer to resolve metadata for.
     * @post second_value and ind are valid; num_tab unchanged.
     */
    void set_info_second_value_ind_from_key(const void* key);

    /**
     * @brief Load all compression metadata from global map
     * @details Retrieves (second_value, ind, num_tab) record from share_info_array
     *          using ptr_on_array as key. Also initializes compression flag pointer.
     * @pre ptr_on_array must point to valid array
     * @post second_value, ind, num_tab, and ptr_is_inside_chunk_list_to_compress are valid
     * @note For bool specialization, this is a no-op as metadata is fixed
     */

    void set_info_on_chunk(this Derived& self);

    /**
     * @brief The cached second_value field, as set by set_info_on_chunk() /
     *        set_info_second_value_ind().
     * @return The second constant value.
     */
    [[nodiscard]] T get_second_value() const noexcept
    {
        return second_value_;
    }

    /**
     * @brief The cached (second_value, ind, num_tab) metadata tuple.
     * @return Reference to inf_array.
     */
    const InfArrayT& get_inf_array() const noexcept
    {
        return inf_array_;
    }

    /// @brief Arm this proxy with the owning array's registry mirror
    ///        (zero_one_value::Array::registry_mirror()).
    /// @details Called by ArrayChunkBase::operator[]. An unarmed proxy (deferred-replay
    ///          reconstruction, *_from_key path) resolves the tuple through share_info_array.
    /// @note Without it every write scans the registry, a large share of the write cost.
    void arm_inf_array(const InfArrayT* src) const noexcept
    {
        inf_array_src_ = src;
    }

  protected:
    /**
     * @brief The second_value in inf_array (set by set_info_on_chunk()), for callers that did
     *        the full metadata load.
     * @return The second constant value.
     */
    [[nodiscard]] T make_second_value() const noexcept
    {
        return inf_array_.second_value;
    }

    /**
     * @brief This array's num_tab, for the per-write refresh (Proxy2::set_num_tab()):
     *        from the armed mirror when there is one, from inf_array otherwise.
     * @details Not read back from inf_array when armed: set_info_second_value_ind()
     *          has just copied the 16-byte record there with one q-register store,
     *          and a narrower load of it cannot be forwarded on AArch64 (Neoverse)
     *          -- the write then waits for the store to reach the cache. Same
     *          hazard as in set_info_on_chunk().
     */
    [[nodiscard]] int armed_num_tab(this const Derived& self)
    {
        if (const InfArrayT* src = self.armed_inf_array()) [[likely]]
        {
            return src->num_tab;
        }
        return self.unarmed_num_tab();
    }

    /// @brief armed_num_tab()'s unarmed path, out of line: inlined, the compiler
    ///        turns the branch into a select that loads inf_array.num_tab on
    ///        every write anyway (and waits for the store again).
    [[gnu::noinline, gnu::cold]] int unarmed_num_tab() const noexcept
    {
        return inf_array_.num_tab;
    }

    /**
     * @brief Most common non-zero value in the chunk
     * @details Used for second-level compression. When a chunk contains mostly
     *          this value (and first_value), it can be compressed to save memory.
     */
    T second_value_;

    /**
     * @brief Index into the chunk's second-value table
     * @details Multiple chunks can share second-value storage; this index
     *          identifies which slot this chunk uses.
     */

    int ind_;

    /**
     * @brief Cached compression metadata tuple
     * @details Contains (second_value, ind, num_tab) to avoid repeated lookups.
     *          Populated by load_inf_array().
     */
    InfArrayT inf_array_; ///< Compression metadata associated with this proxy.

  private:
    /// @brief The (second_value, ind, num_tab) record: from the armed mirror when
    ///        there is one, from the registry otherwise. Under any_check the
    ///        two are compared (model: Proxy2A::set_num_tab).
    [[nodiscard]] InfArrayT load_inf_array(this const Derived& self);
    /// @brief The armed mirror (checked under any_check), or nullptr. The
    ///        per-write refreshes read the fields through it, not through
    ///        load_inf_array()'s copy: see set_info_on_chunk().
    [[nodiscard]] const InfArrayT* armed_inf_array(this const Derived& self);

    /// Owning array's registry mirror; nullptr until armed.
    mutable const InfArrayT* inf_array_src_ = nullptr;
};

// bool and Bit (bit.h): the second constant is `true`, only num_tab is per array.
template <typename T, typename Derived>
    requires is_bool_like_v<T>
class ProxySpec2<T, Derived> : public Proxy2Base<Derived, T, Chunk2<T>>
{
  public:
    using Base      = Proxy2Base<Derived, T, Chunk2<T>>;
    using InfArrayT = SecondValueInfo<T>;

    using ChunkType = Chunk2<T>;

    using ValuesTypeRef = Proxy2Base<Derived, T, ChunkType>::ValuesTypeRef;

    using Base::Base;

    /// @brief Arm this proxy with the owning array's registry mirror (see the non-bool
    ///        ProxySpec2::arm_inf_array()).
    void arm_inf_array(const InfArrayT* src) const noexcept
    {
        inf_array_src_ = src;
    }

    /// @brief Bool specialization of set_info_on_chunk(). See
    ///        set_info_second_value_ind() below for why this is NOT a no-op.
    void set_info_on_chunk(this Derived& self);

    // Must NOT be a no-op: bool's second_value (true) and ind (0) are fixed, but num_tab is
    // per array. With a second bool array (rechunk's tmp) a default num_tab 0 indexes the wrong
    // compression flag array (heap overflow) and queues candidates rechunk_drop_candidates(tmp)
    // cannot remove. Only num_tab is consumed.
    /**
     * @brief Resolve this proxy's real per-array num_tab from the global
     *        metadata map (second_value/ind stay their constexpr values).
     * @post inf_array's num_tab is valid for this proxy's own array.
     */
    void set_info_second_value_ind(this Derived& self)
    {
        static_cast<ProxySpec2&>(self).inf_array_ = self.load_inf_array();
    }

    /**
     * @brief Like set_info_second_value_ind(), but resolves num_tab for an
     *        explicitly given array pointer rather than this proxy's own
     *        ptr_on_array.
     * @param key Array pointer to resolve num_tab for.
     */
    void set_info_second_value_ind_from_key(this Derived& self, const void* key)
    {
        static_cast<ProxySpec2&>(self).inf_array_ = self.get_info_second_value(key);
    }

  protected:
    // Named as the generic ProxySpec2 member, for the using-declarations both share.
    static constexpr bool second_value_ = true;

    /**
     * @brief Bool-only: the second constant is always `true`.
     * @return true.
     */
    constexpr bool make_second_value() noexcept
    {
        return true;
    }

    /**
     * @brief This array's num_tab, for the per-write refresh (Proxy2::set_num_tab()):
     *        from the armed mirror when there is one, from inf_array otherwise.
     * @details Not read back from inf_array when armed: set_info_second_value_ind()
     *          has just copied the 16-byte record there with one q-register store,
     *          and a narrower load of it cannot be forwarded on AArch64 (Neoverse)
     *          -- the write then waits for the store to reach the cache. Same
     *          hazard as in set_info_on_chunk().
     */
    [[nodiscard]] int armed_num_tab(this const Derived& self)
    {
        if (const InfArrayT* src = self.armed_inf_array()) [[likely]]
        {
            return src->num_tab;
        }
        return self.unarmed_num_tab();
    }

    /// @brief armed_num_tab()'s unarmed path, out of line: inlined, the compiler
    ///        turns the branch into a select that loads inf_array.num_tab on
    ///        every write anyway (and waits for the store again).
    [[gnu::noinline, gnu::cold]] int unarmed_num_tab() const noexcept
    {
        return inf_array_.num_tab;
    }

    /**
     * @brief Cached compression metadata tuple
     * @details Contains (second_value, ind, num_tab) to avoid repeated lookups.
     *          Populated by load_inf_array().
     */
    InfArrayT inf_array_; ///< Compression metadata associated with this proxy.

    /// @brief The (second_value, ind, num_tab) record: from the armed mirror when
    ///        there is one, from the registry otherwise. Under any_check the
    ///        two are compared (model: Proxy2A::set_num_tab).
    [[nodiscard]] InfArrayT load_inf_array(this const Derived& self);
    /// @brief The armed mirror (checked under any_check), or nullptr. The
    ///        per-write refreshes read the fields through it, not through
    ///        load_inf_array()'s copy: see set_info_on_chunk().
    [[nodiscard]] const InfArrayT* armed_inf_array(this const Derived& self);

    /// Owning array's registry mirror; nullptr until armed.
    mutable const InfArrayT* inf_array_src_ = nullptr;
};

/**
 * @brief Policy to handle Fast Block Compression specific logic for Proxy2.
 * Encapsulates per-thread list management and trigger logic.
 */
template <typename Derived, typename T, typename ChunkT, bool Enabled>
class Proxy2FastCompPolicy : public ProxySpec2<T, Derived>
{
  public:
    using Base2 = Proxy2Base<Derived, T, Chunk2<T>>;
    using Base  = ProxyBase<Derived, T, Chunk2<T>>;

    using Base::chunk_;
    using Base::num_;
    using Base::read_values;
    using Base::read_values_async;
    using Base::tab_is_inside_chunk_list_to_compress;
    using Base::values_;
    using ProxySpec2<T, Derived>::inf_array_;

    // second_value is in a dependent base: import it for unqualified use (two-phase lookup).
    using ProxySpec2<T, Derived>::second_value_;

    using ManagerChunk2 = ManagerChunk<Chunk2, T>;

    using CompressionManagerType = CompressionManager<T, ChunkT>;

    using ValuesTypeRef = Proxy2Base<Derived, T, ChunkT>::ValuesTypeRef;

    using ProxySpec2<T, Derived>::ProxySpec2;

    /**
     * @brief Check if compression should be triggered for current thread
     * @details Examines compression list size and activates slow-path if threshold exceeded.
     *          Activates slow-path mode to prevent concurrent fast-path access during compression.
     * @param thr_rank Calling thread's rank (whose list is examined).
     * @post May activate slow-path mode if compression is needed
     * @note Thread safety: Sets atomic slow-path flag visible to all threads
     */
    void trigger_compression_if_needed(this Derived& self, int thr_rank) noexcept;

    /**
     * @brief A candidate was just queued on the calling thread's list @p vec
     *        (ProxyBase's add paths): note it when the list is now past the
     *        threshold, for trigger_compression_if_needed() to raise the
     *        rank's flag after this write, as its own test did.
     * @param vec The candidate list just pushed to.
     */
    static void note_candidate_pushed(const auto& vec) noexcept
    {
        if (static_cast<int>(vec.size()) > ManagerChunk2::get_size_max_number_of_chunk_to_check_compress())
        {
            list_over_threshold_ = true;
        }
    }

    /**
     * @brief Performs actual block compression for the current thread's candidates.
     */
    void perform_compression(this Derived& self) noexcept;

    /**
     * @brief Async-context counterpart of execute_inplace_operation():
     *        materialise the chunk referenced by `ctx` if it is at either
     *        constant, run `operation` on the reloaded values pointer, then
     *        mark for compression based on the old/new values.
     * @tparam Lambda Callable performing the actual in-place modification.
     * @param ctx Async write context.
     * @param operation The operation to run.
     */
    template <typename Lambda>
    void execute_inplace_operation_async(auto* ctx, Lambda operation) const;

  private:
    /// @brief The calling thread pushed a candidate that took its list past
    ///        the threshold, not yet turned into the rank's dangerous flag
    ///        (note_candidate_pushed / trigger_compression_if_needed). A
    ///        thread's lists only grow by its own pushes.
    static inline thread_local bool list_over_threshold_ = false;

  public:
    /**
     * @brief Async-context counterpart of execute_simple_operation():
     *        materialise the chunk referenced by `ctx` if it is at either
     *        constant, run `operation` on the reloaded values pointer, then
     *        mark for compression based on the old/new values.
     * @tparam Lambda Callable performing the actual write.
     * @param ctx Async write context.
     * @param operation The write to perform.
     */
    template <typename Lambda>
    void execute_simple_operation_async(auto* ctx, Lambda operation);

    /**
     * @brief Async-context counterpart of Proxy2::execute_arithmetic_compound_operation().
     * @details Computes the result in high precision first; if the chunk is
     *          currently at the second constant and the result would leave
     *          it (will_be_not_second_value), materialises it and runs
     *          `operation`. Otherwise, if not at the second constant, stores
     *          the result directly and marks for compression. If the chunk
     *          is at the second constant AND the result would stay there,
     *          this is a no-op (the chunk remains correctly compressed).
     * @tparam Op Binary operator (e.g. std::multiplies<>, std::divides<>).
     * @tparam Lambda Callable used only on the materialise-and-apply path.
     * @tparam U Type of the operand.
     * @param ctx Async write context.
     * @param op The operator to apply.
     * @param operand Operand to combine with the current value.
     * @param operation Called (with `op`) when the chunk had to be materialised.
     */
    template <typename Op, typename Lambda, typename U>
    void execute_arithmetic_compound_operation_async(auto* ctx, Op op, const U& operand, Lambda operation) const;

    /**
     * @brief This proxy's second constant value. Bool: always true.
     * @return The second constant value.
     */
    [[nodiscard]] auto get_second_value() const noexcept
    {
        if constexpr (is_not_bool)
        {
            return second_value_;
        }
        else
        {
            return true;
        }
    }
    /**
     * @brief Get the compression candidate list for a specific thread
     * @param thr Thread rank [0, MAX_THREADS)
     * @return Reference to vector of (chunk*, chunk_idx, tab_idx) tuples
     * @details Uses second_value as key to index into per-thread compression maps.
     *          This allows grouping chunks with the same second_value for batch compression.
     */

    VectorTuple4ChunkInt2<Chunk2, T>& get_vec_tuple_to_compress_potentially_thr(int thr) const noexcept
    {
        const auto second_value = get_second_value();
        // find(), not operator[]: operator[] INSERTS a null entry for a missing
        // key, which have_chunk_to_compress() would dereference later.
        auto&      candidates = ManagerChunk2::map_chunk2_to_compress_potentially_thr[thr];
        const auto it         = candidates.find(second_value);
        if (it == candidates.end() || it->second == nullptr) [[unlikely]]
        {
            chunk2_missing_candidate_list(thr);
        }
        return *it->second;
    }

    /**
     * @brief Async-context counterpart of get_vec_tuple_to_compress_potentially_thr().
     * @param ctx Async write context (supplies the second_value key).
     * @param thr Thread rank [0, MAX_THREADS).
     * @return Reference to vector of (chunk*, chunk_idx, tab_idx) tuples.
     */
    VectorTuple4ChunkInt2<Chunk2, T>& get_vec_tuple_to_compress_potentially_async_thr(auto* ctx, int thr) const
    {
        const auto second_value = get_second_value(ctx);
        // find(), not operator[]: operator[] INSERTS a null entry for a missing
        // key, which have_chunk_to_compress() would dereference later.
        auto&      candidates = ManagerChunk2::map_chunk2_to_compress_potentially_thr[thr];
        const auto it         = candidates.find(second_value);
        if (it == candidates.end() || it->second == nullptr) [[unlikely]]
        {
            chunk2_missing_candidate_list(thr);
        }
        return *it->second;
    }

    /**
     * @brief Mark for compression with explicit old value
     * @param old_value Previous value at this location
     * @param new_value New value being written
     * @details Provides better accuracy than reading current value due to potential
     *          race conditions in multi-threaded environments.
     */

    void mark_for_compression_if_needed_with_value(this const Derived& self, InT<T> old_value, InT<T> new_value)
    {
        self.Base2::mark_for_compression_if_needed(self.second_value_, old_value, new_value, Chunk2<T>::first_value);
    }

    /**
     * @brief Real (non-stub) async-context array pointer lookup: the ctx
     *        already carries ptr_on_array directly, so this is a plain cast.
     * @param ctx Async write context.
     */
    static const T* get_ptr_on_array_async(auto* ctx) noexcept
    {
        return static_cast<const T*>(ctx->ptr_on_array);
    }

    // ------------------------------------------------------------------
    // Compression-tracking surface: needs the per-tab tracking arrays, which exist only in
    // this mode. Same shape as ProxyBaseFastCompPolicy in proxy_base.h.
    // ------------------------------------------------------------------

    /**
     * @brief No-op, called at array teardown. The slot's buffer is released when
     *        resize_tab_is_int() reassigns this num_tab; a slot never reused keeps it until exit.
     * @param num_tab Tab index being torn down.
     */
    static void clear_is_int([[maybe_unused]] int num_tab)
    {
        // Intentionally empty, but must exist: zero_one_value::Array::clear_core() calls it in its
        // `if constexpr (use_fast_block_compression)` branch.
        //  tab_is_inside_chunk_list_to_compress[num_tab].clear();
    }

    /**
     * @brief Get the compression tracking array for a tab
     * @param num_tab Tab index
     * @return Reference to the atomic char array
     * @pre num_tab must be < tab_is_inside_chunk_list_to_compress.size()
     */

    static ChunkFlagRowT& get_tab_is_inside_list_to_compress(int num_tab)
    {
        return tab_is_inside_chunk_list_to_compress[num_tab];
    }

    /**
     * @brief Reset compression flag for a specific chunk
     * @param num_tab Tab index
     * @param num_chunk Chunk index
     * @post Chunk marked as not in compression list (can be re-added)
     * @note Thread safety: Memory order release ensures visibility to all threads
     */

    static void reset_tab_is_inside_list_to_compress(int num_tab, int num_chunk) noexcept
    {
        tab_is_inside_chunk_list_to_compress[num_tab][num_chunk].store(0, std::memory_order_release);
    }

    /**
     * @brief Async-context counterpart of get_second_value(): the second
     *        constant for the chunk referenced by `ctx`. Bool: always true.
     * @param ctx Async write context.
     * @return The second constant value.
     */
    [[nodiscard]] static auto get_second_value(auto* ctx) noexcept
    {
        if constexpr (is_not_bool)
        {
            return ctx->second_value;
        }
        else
        {
            return true;
        }
    }

    /**
     * @brief Checks if a value change should trigger compression consideration
     * @param old_value Previous value
     * @param new_value New value
     * @param second_value Current second value for comparison
     * @param first_value First constant value for comparison
     * @return true if the chunk should be marked for compression
     */
    [[nodiscard]] static bool
    should_mark_for_compression(InT<T> old_value, InT<T> new_value, InT<T> second_value, InT<T> first_value) noexcept
    {
        return CompressionManagerType::should_mark_for_compression(old_value, new_value, second_value, first_value);
    }

  protected:
    /**
     * @brief Shared implementation of operator*=, /=, %=: materialise the chunk if needed,
     *        apply `op` in a common high-precision type, truncate back to T.
     * @details Deferred-compression form: the result is compared with the second constant
     *          BEFORE the store, then the write is marked for compression. The serial
     *          specialisation keeps per-chunk counters instead.
     * @tparam Op Binary operator to apply.
     * @tparam Lambda Callable `(op)` used on the materialise-and-apply path.
     * @tparam U Type of the operand.
     */
    template <typename Op, typename Lambda, typename U>
    void execute_arithmetic_compound_operation(this Derived& self, Op op, const U& operand, Lambda operation);

    /**
     * @brief Mark chunk for compression if new value creates compression opportunity
     * @param new_value Value being written
     * @details Checks if write creates a state where chunk becomes compressible:
     *          - All values become first_value
     *          - All values become second_value
     *          Atomically adds chunk to compression list if needed.
     * @note Thread safety: Uses compare_exchange for atomic list insertion
     */
    void mark_for_compression_if_needed(this const Derived& self, InT<T> new_value)
    {
        self.Base2::mark_for_compression_if_needed(self.second_value_, new_value, Chunk2<T>::first_value);
    }

  private:
    static constexpr bool is_bool     = is_bool_like_v<T>; // bool and Bit (bit.h)
    static constexpr bool is_not_bool = !is_bool;
};

/**
 * @brief Specialization for Disabled Fast Compression (Zero Overhead).
 */
template <typename Derived, typename T, typename ChunkT>
class Proxy2FastCompPolicy<Derived, T, ChunkT, false> : public ProxySpec2<T, Derived>
{
  public:
    using ValuesTypeRef = Proxy2Base<Derived, T, ChunkT>::ValuesTypeRef;
    using Base          = ProxyBase<Derived, T, Chunk2<T>>;
    using Base::chunk_;
    using ProxySpec2<T, Derived>::second_value_;
    using ProxySpec2<T, Derived>::inf_array_;

    using ProxySpec2<T, Derived>::ProxySpec2;

    // ------------------------------------------------------------------
    // Counter mode is the exact complement of fast block compression
    // (use_compression_with_counter == !use_fast_block_compression, config.h), so this
    // specialisation IS the counter-based mode; the handlers below are its compression trigger.
    // Callers (proxy2_inline.h) reach them via `this->` under
    // `if constexpr (chunk_config::use_compression_with_counter)`.
    // ------------------------------------------------------------------
    // ========================================================================
    // Full compression handlers (counter-based variant)
    // ========================================================================

    /**
     * @brief Handle chunk that became fully first_value
     * @details Frees allocated storage and sets chunk to compressed first_value state.
     *          Resets value counter to zero.
     * @post Chunk compressed; memory freed; values points to const storage
     */
    void manage_full_first_value() const
    {
        // Pin the ambient chunk size before the free chain (clear_st -> ... ->
        // get_chunk_allocator(thr_rank).free_elt_pow2()): get_chunk_allocator() routes by the
        // ambient thread_chunk_size(), not by the buffer. A stale ambient (left by another
        // array's write) selects the wrong pool, which then removes a node it does not own.
        // Fast-alloc only: std-alloc's svfree() does not route by ambient.
        if constexpr (build_config::chunk_dynamic)
        {
            shrink_array::set_thread_chunk_size(this->chunk_size());
        }
        if constexpr (is_not_bool)
        {
            chunk_.clear_st(0, this->ind_);
        }
        else
        {
            chunk_.clear_st(0, 0); // bool: ind ignored by bool-aware clear_int_st
        }
        chunk_.values_ = static_cast<T*>(chunk_.get_cst_val1());
    }

    /**
     * @brief Handle chunk that became fully second_value
     * @details Frees allocated storage and sets chunk to compressed second_value state.
     *          Resets value counter to zero.
     * @post Chunk compressed; memory freed; values points to second_value storage
     */
    void manage_full_second_value() const
    {
        // Same rule as manage_full_first_value(): see that function's comment.
        if constexpr (build_config::chunk_dynamic)
        {
            shrink_array::set_thread_chunk_size(this->chunk_size());
        }
        if constexpr (is_not_bool)
        {
            chunk_.clear_st(1, this->ind_);
        }
        else
        {
            chunk_.clear_st(1, 0); // bool: ind ignored by bool-aware clear_int_st
        }
    }

  protected:
    /**
     * @brief Serial form of execute_arithmetic_compound_operation(): the
     *        counter path -- a chunk leaving its second constant gets its
     *        second-value counter set, and `operation` writes through the
     *        values pointer it is handed.
     * @tparam Lambda Callable `(op, T* values)`.
     */
    template <typename Op, typename Lambda, typename U>
    void execute_arithmetic_compound_operation(this Derived& self, Op op, const U& operand, Lambda operation);

  private:
    // Used by the counter-based handlers above (the Enabled=true specialisation has its own).
    static constexpr bool is_bool     = is_bool_like_v<T>; // bool and Bit (bit.h)
    static constexpr bool is_not_bool = !is_bool;
};

/// @brief Proxy to one element of a Chunk2<T> chunk: reads, assignments and compound operators on it.
/// @tparam T Element type.
template <typename T>
// no move assignment on purpose -- assigning a proxy
// WRITES the element (see its operator=); a defaulted move would rebind it instead.
// NOLINTNEXTLINE(cppcoreguidelines-special-member-functions)
class Proxy2 : public Proxy2FastCompPolicy<Proxy2<T>, T, Chunk2<T>, chunk_config::use_fast_block_compression>
{
  public:
    /// @brief Stream output of the element's value (defined below).
    template <typename U>
    friend std::ostream& operator<<(std::ostream& os, const Proxy2<U>& proxy);

    using Base2          = Proxy2Base<Proxy2, T, Chunk2<T>>;
    using Base           = ProxyBase<Proxy2, T, Chunk2<T>>;
    using ChunkType      = Chunk2<T>;
    using FastCompPolicy = Proxy2FastCompPolicy<Proxy2, T, Chunk2<T>, chunk_config::use_fast_block_compression>;

    using CompressionManagerType = CompressionManager<T, ChunkType>;

    friend Base;
    friend Base2;

    using Base::set_num_tab;

    using BaseSpec = ProxySpec2<T, Proxy2>;
    using BaseSpec::second_value_;

  private:
    /// Alias for the chunk manager associated with type T.
    using ManagerChunk2 = ManagerChunk<Chunk2, T>;

    using Base::chunk_;        ///< Reference to the associated chunk.
    using Base::num_;          ///< Index of the current element within the chunk.
    using Base::num_chunk_;    ///< Index of the chunk within the source array.
    using Base::num_tab_;      ///< Index used for compression table lookup.
    using Base::ptr_on_array_; ///< pointer to array (for tab lookup)
    using Base::read_values;
    using Base::values_; ///< Pointer to chunk's data array

    using BaseSpec::inf_array_;
    using InfArrayT = BaseSpec::InfArrayT;

    using ValuesTypeRef = Base::AtomicPtrRefT; // C++20: no typename in an alias

    // Both members exist in both ProxyBaseFastCompPolicy specialisations: no guard needed.
    // Do not add `using Base::read_values_async;`: it exists only in the Enabled=true
    // specialisation (imported in Proxy2FastCompPolicy<..., true>), so it breaks builds
    // without fast block compression.
    using Base::ptr_is_inside_chunk_list_to_compress_; ///< Per-chunk compression flag pointer
    using Base::tab_is_inside_chunk_list_to_compress;  ///< Global compression tracking array

    // Declared per Proxy2FastCompPolicy specialisation: the deferred and counter forms are
    // different algorithms (and pass `operation` different arguments).
    using FastCompPolicy::execute_arithmetic_compound_operation;

    static constexpr bool is_bool     = is_bool_like_v<T>; // bool and Bit (bit.h)
    static constexpr bool is_not_bool = !is_bool;

    /**
     * @brief Chunk2's marking pre-check: a write of a constant, on an array that
     *        has a candidate list for its second value.
     * @details Shadows Proxy2Base::pre_check_mark_for_compression(), as Proxy2A does.
     *          Candidate lists are keyed by second value and created only by
     *          zero_one_value::Array::set_second_value(); an array without one is not walked by the
     *          compression pass either. Reject such writes HERE, before
     *          add_to_compression_list() CASes the chunk's "queued" flag: giving up after the
     *          CAS would leave the chunk flagged for good.
     * @param new_value The new value to check.
     * @param first_value The first constant value to compare against.
     * @param second_val The second constant value to compare against.
     * @return true if the write is worth examining further.
     */
    [[nodiscard]] bool
    pre_check_mark_for_compression(InT<T> new_value, InT<T> first_value, InT<T> second_val) const noexcept
    {
        return this->pre_check_mark_for_compression_strict(new_value, first_value, second_val) &&
               has_candidate_list(second_val);
    }

    // ========================================================================
    // Chunk allocation and initialization
    // ========================================================================

    /**
     * @brief Allocate chunk storage if currently compressed
     * @details Decompresses the chunk by allocating full array storage.
     *          Chooses initialization based on current compression state:
     *          - cst_val1: Fill with first_value (typically 0)
     *          - cst_val2: Fill with second_value
     * @post Chunk has allocated values array; no longer compressed
     * @note Thread safety: Uses atomic compare-exchange in multi-threaded builds
     */
    void create_value() const noexcept
    {
        create_value(is_cst_val2());
    }

    /**
     * @brief create_value() with the compression state already known.
     * @param is_cst2 Whether the chunk is at its second constant (else at the
     *        first), from the caller's load of `values`. A stale state is
     *        harmless: the chunk can only have been materialised since, and
     *        the CAS of chunk.create_value() then fails.
     */
    void create_value(bool is_cst2) const noexcept
    {
        if constexpr (build_config::chunk_dynamic)
        {
            // Route this (possibly worker-thread) allocation to the pool of THIS array's chunk
            // size: set the per-thread active size right before materialising.
            shrink_array::set_thread_chunk_size(this->chunk_size());
        }
        if constexpr (is_not_bool)
        {
            chunk_.create_value(is_cst2, inf_array_);
        }
        else
        {
            chunk_.create_value_bool(is_cst2);
        }
    }

    /**
     * @brief Core assignment logic without compression tracking
     * @param v Value to assign
     * @details Handles:
     *          - Short-circuit if value matches compression state
     *          - Decompression if needed
     *          - Actual value write
     * @note Thread safety: Caller must ensure appropriate locking
     */
    void perform_core_assignment(InT<T> v) const noexcept;

    /// @brief Materialise the chunk if it is compressed (cst_val1 / cst_val2), so `values`
    ///        points at writable heap storage.
    ///
    /// Twin of Proxy2A::ensure_writable_storage: decompresses the second-value form, then
    /// the first-value form, seeding the nb_val_eq_* counters to "full". is_cst_val1() is
    /// re-tested AFTER the cst_val2 step on purpose: the states are mutually exclusive (single
    /// storage pointer), so at most one create_value runs.
    void ensure_writable_storage();

    /**
     * @brief Run an arbitrary write operation on this element, materialising
     *        the chunk first if it is currently compressed, then marking
     *        for compression based on the old/new values.
     * @tparam Lambda Callable performing the actual write.
     * @param operation The write to perform.
     */
    template <typename Lambda>
    void execute_simple_operation(Lambda operation);

    /**
     * @brief Run an arbitrary in-place operation on this element,
     *        materialising the chunk first if it is currently compressed.
     * @tparam Lambda Callable performing the actual in-place modification.
     * @param operation The operation to run.
     */
    template <typename Lambda>
    void execute_inplace_operation(Lambda operation);

    // ========================================================================
    // ASSIGNMENT OPERATOR
    // ========================================================================

    /**
     * @brief Core implementation behind operator=(): writes `v` at this
     *        proxy's index.
     * @details
     * Synchronous fast-path flow:
     * 1. Fast-path: If v == first_value and chunk is cst_val1, no-op.
     * 2. Load metadata (set_info_second_value_ind()) if needed.
     * 3. Fast-path: if v == second_value and chunk is already at/equal to
     *    it, no-op.
     * 4. Decompress chunk if compressed, write value, mark for compression.
     * There is also a slower asynchronous path (deferred/lock-contended
     * case), which pools a write context and defers the same core logic.
     * @tparam U Type of the value being assigned.
     * @param v Value to assign.
     * @return The assigned value.
     * @note Complexity: O(1) amortized; may trigger O(n) decompression.
     * @note Thread safety: Thread-safe via fast-path counters or slow-path lock.
     */
    template <typename U>
    // InT<U>, not const U&: this frame is not inlined into the caller, so small types pass
    // in registers. Non-deducible, hence the explicit assign_impl<U>(v).
    [[nodiscard]] T assign_impl(InT<U> v) noexcept;

    // ========================================================================
    // COMPOUND ARITHMETIC OPERATORS
    // ========================================================================

    /**
     * @brief Read-modify-write behind the compound operators (`+=`, `-=`...):
     *        `values[num] = op(values[num], v)`, with the second-value
     *        bookkeeping of a write (a slot leaving or reaching the second
     *        value updates the chunk's counter, a full chunk folds).
     * @tparam U Type of the operand.
     * @tparam Op Binary operator applied to (current, v).
     * @param v Operand.
     * @param op The operator.
     * @note A result equal to the first value is counted as one, and a chunk
     *       it fills folds: `*=`, `/=` and `%=` (the callers) can land on it
     *       from a non-zero slot.
     * @note Materialises any foreign deferred write to this offset first
     *       (reader_sync) so the RMW cannot clobber it.
     */
    template <typename U, typename Op>
    void apply_compound_operation(Op op, InT<U> v) noexcept; // InT: see assign_impl

  public:
    /**
     * @brief Construct a proxy for element `num_` of `chunk_`.
     * @param values Reference to the chunk's values pointer.
     * @param num Element offset within the chunk.
     * @param chunk Owning chunk.
     */
    explicit Proxy2(ValuesTypeRef values, int num, Chunk2<T>& chunk) noexcept
        : Proxy2FastCompPolicy<Proxy2, T, Chunk2<T>, chunk_config::use_fast_block_compression>(values, num, chunk)
    {
    }

    // -------------------------------------------------------------------------
    // PUBLIC OPERATORS (The Filter)
    // -------------------------------------------------------------------------

    /**
     * @brief Standard assignment for identical types (e.g., float = float).
     * No warning triggered.
     */
    template <typename U>
        requires IsCleanAssignment<T, U>
    T operator=(const U& v) noexcept
    {
        return assign_impl<U>(v);
    }

    /**
     * @brief Trap assignment for different types (e.g., float = double).
     * Triggers a compiler warning to prevent missing 'f' suffix for float constants.
     */
    template <typename U>
        requires(!IsCleanAssignment<T, U>) && std::assignable_from<T&, U>
    [[deprecated("HPC WARNING: Type mismatch in assignment. Possible precision loss. Use 'f' suffix for floats.")]]
    T operator=(const U& v) noexcept
    {
        // The cast is explicit here to clear the final target value
        return assign_impl<T>(static_cast<T>(v));
    }

    /**
     * @brief Proxy-to-proxy assignment, `a[i] = b[j]`: read @p from through operator T()
     *        (the gated read), then write through assign_impl() (Proxy::operator='s meaning).
     * @param from Proxy to read the value from.
     * @return *this.
     */
    const Proxy2& operator=(const Proxy2& from) noexcept
    {
        if (std::addressof(from) != this) // operator& is deleted
        {
            const InT<T> v = static_cast<T>(from); // complete before the write starts
            static_cast<void>(assign_impl<T>(v));  // the value written: not needed here
        }
        return *this;
    }
    Proxy2(const Proxy2& other) = default;

    T*       operator&()       = delete;
    const T* operator&() const = delete;

    /**
     * @brief Pointer to the owning array (used as the compression-metadata
     *        map key).
     */
    [[nodiscard]] const T* get_ptr_on_array() const noexcept
    {
        return static_cast<const T*>(ptr_on_array_);
    }

    // ========================================================================
    // Chunk state queries
    // ========================================================================

    /**
     * @brief Check if chunk is compressed to the second constant at index `ind`.
     * @param ind Index of the second constant to check.
     * @return true if the chunk is compressed to the second constant at ind.
     * @note Non-bool only: a compile error on a bool chunk (bool has no
     *       indexed second constant -- use the no-arg overload below).
     */
    [[nodiscard]] bool is_cst_val2(int ind) const noexcept
    {
        if constexpr (is_not_bool)
        {
            return chunk_.is_cst_val2(ind);
        }
        else
        {
            static_assert(false, "is_cst_val2(int ind) is for non-bool elements: use is_cst_val2()");
        }
    }

    /**
     * @brief Check if chunk is compressed to the second constant. Works for
     *        both bool and non-bool (unlike the indexed overload above):
     *        non-bool resolves the index from BaseSpec::ind, bool has a
     *        single shared second constant.
     * @return true if the chunk is compressed to the second constant.
     */
    [[nodiscard]] bool is_cst_val2() const noexcept
    {
        if constexpr (is_not_bool)
        {
            return chunk_.is_cst_val2(BaseSpec::ind_);
        }
        else
        {
            return chunk_.is_cst_val2();
        }
    }

    /**
     * @brief is_cst_val2() judged from an already-loaded values pointer.
     * @details Saves the reload -- `values == ...` on the atomic is a seq_cst
     *          load (ldar on aarch64) -- and keeps the caller's tests on one
     *          snapshot.
     * @param vals The pointer the caller loaded.
     * @return true if @p vals is the chunk's second constant buffer.
     */
    [[nodiscard]] bool is_cst_val2_at(const T* vals) const noexcept
    {
        if constexpr (is_not_bool)
        {
            return vals == chunk_.get_cst_val2(BaseSpec::ind_);
        }
        else
        {
            return vals == chunk_.get_cst_val2();
        }
    }

    // ========================================================================
    // Metadata initialization
    // ========================================================================

    /**
     * @brief Extract and cache tab number from compression metadata
     * @pre inf_array must be populated via get_info_second_value()
     * @post num_tab contains valid tab index
     */
    void set_num_tab() noexcept
    {
        num_tab_ = BaseSpec::armed_num_tab(); // not inf_array.num_tab: see armed_num_tab()
    }

    /**
     * @brief This proxy's cached table number.
     * @return The table number.
     */
    [[nodiscard]] int get_num_tab() const noexcept
    {
        return Base::get_num_tab();
    }

    /**
     * @brief Initialize both tab number and compression flag pointer
     * @details Convenience method combining set_num_tab() and
     *          set_ptr_is_inside_chunk_list_to_compress()
     * @post Both num_tab and ptr_is_inside_chunk_list_to_compress are valid
     *
     * @note No build-mode guard: the sole caller
     *       (Proxy2Base::write_with_optional_lock_with_set_ptr) is under
     *       `if constexpr (chunk_config::use_fast_block_compression)`, so the empty
     *       tab_is_inside_chunk_list_to_compress of other modes is never indexed.
     */
    void set_num_tab_ptr_is_inside_chunk() noexcept
    {
        set_num_tab();

        Base::set_ptr_is_inside_chunk_list_to_compress();
    }

    /**
     * @brief Does the calling rank have a candidate list for @p second_val?
     * @details Reads the rank's own map only; set_second_value() fills the maps
     *          outside the parallel phases. Always true in builds without the
     *          per-value lists (no OpenMP / no fast block compression).
     */
    [[nodiscard]] static bool has_candidate_list(InT<T> second_val) noexcept
    {
        if constexpr (build_config::threaded && chunk_config::use_fast_block_compression)
        {
            const auto& per_thr = ManagerChunk2::map_chunk2_to_compress_potentially_thr;
            const int   thr     = get_thr_rank_fast();
            if (thr < 0 || static_cast<std::size_t>(thr) >= per_thr.size()) [[unlikely]]
            {
                return false;
            }
            const auto& candidates = per_thr[thr];
            const auto  it         = candidates.find(second_val);
            return it != candidates.end() && it->second != nullptr;
        }
        else
        {
            (void)second_val;
            return true;
        }
    }

    /**
     * @brief Look up the (second_value, ind, num_tab) metadata tuple
     *        registered for array pointer @p ptr.
     * @param ptr Array pointer to look up.
     * @return The registered metadata tuple.
     */
    [[nodiscard]] static InfArrayT get_info_second_value(const void* ptr) noexcept
    {
        return share_info_array->get_info_second_value(ptr);
    }

    /**
     * @brief Global map linking chunk values to their associated metadata.
     *
     * Used to retrieve compression-related information for a given value.
     */
    static inline constinit internal::LazyImmortal<PointerMapChunk2<T>> share_info_array;
};

/// @brief Write the element's value, read through proxy.read_pure(), to @p os.
/// @return @p os.
template <typename T>
std::ostream& operator<<(std::ostream& os, const Proxy2<T>& proxy)
{
    // Through the proxy's read entry, like operator T(): the raw values[num] snapshot is stale
    // while a deferred op pends on this offset.
    os << proxy.read_pure();
    return os;
}

} // namespace shrink_array

#include "chunk/internal/proxy2_inline.h" // the template bodies
