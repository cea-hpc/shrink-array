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

#include "chunk/internal/in_t.h"
#include "chunk/internal/bad_ptr_check.h" // CHUNK_CHECK_VALUES_PTR (check builds)
#include <cstdlib> // std::getenv (CHUNK_PROXY2_NO_ACQ_FENCE)
#include "chunk/internal/proxy2.h"        // the declarations these bodies belong to

namespace shrink_array
{

/// The acquire fences that ended set_info_second_value_ind() and the two
/// set_info_on_chunk() are OFF by default: they followed plain (non-atomic)
/// reads of the share_info_array registry and paired with no release. Their
/// `dmb ishld` was the hottest instruction of assign_impl on aarch64; without
/// them, 64 threads chunk2_256 --mode perf: 11.48 s -> 10.68 s (-7.0%).
/// Validated: forced-deferral build at 8/16/32 threads, 10 x --mode check at
/// 64 threads, every chunk2 validation harness at 1/8/16/64 threads.
/// CHUNK_PROXY2_ACQ_FENCE=1 (read once, at load) restores them.
/// TODO: once the wider validation campaign is done, remove this switch and
///       the three guarded fences.
inline const bool g_proxy2_acq_fence = []
{
    const char* e = std::getenv("CHUNK_PROXY2_ACQ_FENCE");
    return e != nullptr && e[0] != '\0' && e[0] != '0';
}();

/// A/B switch: CHUNK_PROXY2_ARM_LEGACY=1 (read once, at load) makes every
/// proxy resolve its tuple through share_info_array again, armed or not.
inline const bool g_proxy2_arm_legacy = []
{
    const char* e = std::getenv("CHUNK_PROXY2_ARM_LEGACY");
    return e != nullptr && e[0] != '\0' && e[0] != '0';
}();

template <typename T, typename Derived>
typename ProxySpec2<T, Derived>::inf_array_t ProxySpec2<T, Derived>::load_inf_array() const
{
    if (inf_array_src_ != nullptr && !g_proxy2_arm_legacy) [[likely]]
    {
        if constexpr (build_config::any_check)
        {
            // ind and num_tab only: ind maps one-to-one to the value, and a
            // value compare would raise a false alarm on a NaN second value.
            const inf_array_t reg = Derived::get_info_second_value(get_ptr_on_array());
            if (std::get<1>(reg) != std::get<1>(*inf_array_src_) || std::get<2>(reg) != std::get<2>(*inf_array_src_))
                INVARIANT_FAILED("ProxySpec2::load_inf_array: armed (ind {}, num_tab {}) differs from the registry's "
                                 "(ind {}, num_tab {}) for array {} -- a registry write without sync_registry_mirror()?",
                                 std::get<1>(*inf_array_src_), std::get<2>(*inf_array_src_), std::get<1>(reg),
                                 std::get<2>(reg), static_cast<const void*>(get_ptr_on_array()));
        }
        return *inf_array_src_;
    }
    return Derived::get_info_second_value(get_ptr_on_array());
}

template <typename Derived>
typename ProxySpec2<bool, Derived>::inf_array_t ProxySpec2<bool, Derived>::load_inf_array() const
{
    if (inf_array_src_ != nullptr && !g_proxy2_arm_legacy) [[likely]]
    {
        if constexpr (build_config::any_check)
        {
            // ind and num_tab only: ind maps one-to-one to the value, and a
            // value compare would raise a false alarm on a NaN second value.
            const inf_array_t reg = Derived::get_info_second_value(get_ptr_on_array());
            if (std::get<1>(reg) != std::get<1>(*inf_array_src_) || std::get<2>(reg) != std::get<2>(*inf_array_src_))
                INVARIANT_FAILED("ProxySpec2::load_inf_array: armed (ind {}, num_tab {}) differs from the registry's "
                                 "(ind {}, num_tab {}) for array {} -- a registry write without sync_registry_mirror()?",
                                 std::get<1>(*inf_array_src_), std::get<2>(*inf_array_src_), std::get<1>(reg),
                                 std::get<2>(reg), static_cast<const void*>(get_ptr_on_array()));
        }
        return *inf_array_src_;
    }
    return Derived::get_info_second_value(get_ptr_on_array());
}

template <typename T, typename Derived>
void ProxySpec2<T, Derived>::set_info_on_chunk()
{
    inf_array    = load_inf_array();
    second_value = std::get<0>(inf_array);
    ind          = std::get<1>(inf_array);
    derived().set_num_tab();

    if constexpr (chunk_config::use_fast_block_compression)
    {
        derived().set_ptr_is_inside_chunk_list_to_compress();
    }

    if (g_proxy2_acq_fence) [[unlikely]] // transition switch, see its declaration
        std::atomic_thread_fence(std::memory_order_acquire);
}

template <typename T, typename Derived>
void ProxySpec2<T, Derived>::set_info_second_value_ind()
{
    inf_array    = load_inf_array();
    second_value = std::get<0>(inf_array);
    ind          = std::get<1>(inf_array);

    if (g_proxy2_acq_fence) [[unlikely]] // transition switch, see its declaration
        std::atomic_thread_fence(std::memory_order_acquire);
}

template <typename T, typename Derived>
void ProxySpec2<T, Derived>::set_info_second_value_ind_from_key(const void* key)
{
    inf_array    = Derived::get_info_second_value(key);
    second_value = std::get<0>(inf_array);
    ind          = std::get<1>(inf_array);
}

template <typename T, typename Derived>
void ProxySpec2<T, Derived>::set_second_value()
{
    inf_array    = load_inf_array();
    second_value = std::get<0>(inf_array);
}

template <typename Derived>
void ProxySpec2<bool, Derived>::set_info_on_chunk()
{
    inf_array = load_inf_array();

    derived().set_num_tab();

    if constexpr (chunk_config::use_fast_block_compression)
    {
        derived().set_ptr_is_inside_chunk_list_to_compress();
    }

    if (g_proxy2_acq_fence) [[unlikely]] // transition switch, see its declaration
        std::atomic_thread_fence(std::memory_order_acquire);
}

// ── Methods only present when fast block compression is active ────────────
// Members of the Enabled=true policy only: Proxy2 instantiates
// Proxy2FastCompPolicy<..., chunk_config::use_fast_block_compression>, and the
// <..., false> specialisation declares none of them, so nothing here is ever
// looked at by a serial build. No preprocessor guard needed.

template <typename Derived, typename T, typename ChunkType, bool Enabled>
void Proxy2FastCompPolicy<Derived, T, ChunkType, Enabled>::trigger_compression_if_needed(int thr_rank) noexcept
{
    if constexpr (is_not_bool)
    {
        derived().second_value = derived().make_second_value();
    }

    auto* vec_tuple_p = manager_chunk2::map_chunk2_to_compress_potentially_thr[thr_rank][derived().second_value].get();

    if (!vec_tuple_p)
        return;

    auto& vec_tuple = *vec_tuple_p;

    const int sz  = static_cast<int>(vec_tuple.size());
    const int szm = manager_chunk2::get_size_max_number_of_chunk_to_check_compress();

    if (Base::should_trigger_compression(sz, szm))
    {
        CompressionManagerType::mark_list_dangerous_for_thread(thr_rank);

        if constexpr (chunk_config::verbose)
            std::cerr << "⚠️ List became dangerous: size=" << sz << "\n";
    }
}

template <typename Derived, typename T, typename ChunkType, bool Enabled>
void Proxy2FastCompPolicy<Derived, T, ChunkType, Enabled>::perform_compression() noexcept
{
    const int szm          = manager_chunk2::get_size_max_number_of_chunk_to_check_compress();
    const int thr_rank     = get_thr_rank_fast();
    const int sz_threshold = szm / 2;

    const auto& ml = manager_chunk2::map_chunk2_to_compress_potentially_thr[thr_rank];

    if constexpr (is_not_bool)
    {
        for (auto& elm : ml)
        {
            // in_t: the map key, by reference when T is large (a struct), by
            // value when it is a scalar. The map is not touched in the loop.
            const in_t<T> second_val  = elm.first;
            auto*   vec_tuple_p = manager_chunk2::map_chunk2_to_compress_potentially_thr[thr_rank][second_val].get();
            if (vec_tuple_p)
            {
                const int sz_before = static_cast<int>(vec_tuple_p->size());

                if (sz_before > sz_threshold)
                {
                    static_cast<void>(CompressionManagerType::flush_deferred(thr_rank));

                    auto* inf_ptr = Derived::share_info_array.find_inf_array(second_val);
                    Chunk2<T>::make_block_compression(thr_rank, *inf_ptr, self());

                    derived().set_info_on_chunk();
                }
            }
        }
    }
    else
    {
        auto* vec_tuple_p = manager_chunk2::map_chunk2_to_compress_potentially_thr[thr_rank][true].get();
        if (vec_tuple_p)
        {
            const int sz_before = static_cast<int>(vec_tuple_p->size());

            if (sz_before > sz_threshold)
            {
                static_cast<void>(CompressionManagerType::flush_deferred(thr_rank));

                Chunk2<T>::make_block_compression_bool(thr_rank, self());

                derived().set_info_on_chunk();
            }
        }
    }
    // Looked at, whether or not any bucket had work: see the 2A counterpart
    // and mark_list_dangerous_for_thread.
    CompressionManagerType::mark_list_safe_for_thread(thr_rank);
}

template <typename T>
void Proxy2<T>::perform_core_assignment(in_t<T> v) const noexcept
{
    bool need_alloc;

    if constexpr (chunk_config::use_compression_with_counter)
    {
        if (v == static_cast<T>(Chunk2<T>::first_value))
        {
            if (Base::is_not_cst_val1() && Base2::is_not_cst_val2())
            {
                if (values[num] != Chunk2<T>::first_value)
                    chunk.inc_nb_val_eq_first_value();

                if (chunk.is_full_first_value(this->chunk_size()))
                {
                    this->manage_full_first_value();
                    return;
                }
            }
        }
        else if (Base::is_not_cst_val1() && values[num] == Chunk2<T>::first_value)
        {
            chunk.dec_nb_val_eq_first_value();
        }
    }

    if (v == second_value)
    {
        if (Base::is_cst_val1())
        {
            create_value();

            if constexpr (chunk_config::use_compression_with_counter)
            {
                chunk.inc_nb_val_eq_second_value();
                chunk.set_nb_val_eq_first_value(this->chunk_size() - 1);
            }

            values[num] = second_value;
            return;
        }

        if constexpr (chunk_config::use_compression_with_counter)
        {
            chunk.inc_nb_val_eq_second_value();

            if (chunk.is_full_second_value(this->chunk_size()))
            {
                this->manage_full_second_value();
                return;
            }
        }
    }
    else if constexpr (chunk_config::use_compression_with_counter)
    {
        if (Base2::is_not_cst_val2())
        {
            // Read once, before the store below: in_t, no copy of a struct.
            const in_t<T> vv = values[num];
            if (vv == second_value)
                chunk.dec_nb_val_eq_second_value();
        }
    }

    if constexpr (chunk_config::stats)
    {
        increm_chunk_write<ChunkType>(v);
    }

    need_alloc = Base2::is_cst_val();

    if (need_alloc)
    {
        [[maybe_unused]] bool was_cst_val2 = false;
        [[maybe_unused]] bool was_cst_val1 = false;

        if constexpr (chunk_config::use_compression_with_counter)
        {
            was_cst_val2 = is_cst_val2();
            was_cst_val1 = Base::is_cst_val1();
        }

        create_value();

        if constexpr (chunk_config::use_compression_with_counter)
        {
            if (was_cst_val2)
            {
                chunk.set_nb_val_eq_second_value(this->chunk_size() - 1);
                // The write that leaves cst_val2 skipped the first-value block
                // at the top (it only counts on a materialised chunk), and
                // create_value's crossed reset just zeroed nb_first_value: a
                // first_value written here is the one this chunk holds, count
                // it. Without it a chunk that went all-second then all-first
                // stopped at size - 1 and never compressed back (serial builds:
                // the counters are the only path; chunk2<bool> showed it, the
                // other types were reclaimed by compress_int's self-heal).
                if (v == static_cast<T>(Chunk2<T>::first_value))
                    chunk.inc_nb_val_eq_first_value();
            }
            else if (v == second_value)
            {
                chunk.inc_nb_val_eq_second_value();
                if (was_cst_val1)
                    chunk.set_nb_val_eq_first_value(this->chunk_size() - 1);
            }
            else if (was_cst_val1)
            {
                chunk.set_nb_val_eq_first_value(this->chunk_size() - 1);
            }
            else if (v == Chunk2<T>::first_value)
            {
                chunk.inc_nb_val_eq_first_value();
            }
        }
    }

    Base::store_value(v);

    if constexpr (build_config::do_check)
    {
        global_check_write_nan(v, num);
    }
}

template <typename T>
template <typename U>
T Proxy2<T>::assign_impl(in_t<U> v) noexcept
{
    // Only one read of values_loc
    const T* const values_loc = chunk.values_load();

    // Fast exit: already first_value and assigning first_value
    if (Base::is_cst_val1_and_v_equal_first_value(values_loc, v))
        return v;

    // Initialize second_value / ind for this chunk/array
    BaseSpec::set_info_second_value_ind();

    // If assigning the second compression value and we are already in that state,
    // or the current stored value is already second_value, nothing to do --
    // unless the chunk has a deferred op pending: an older op at this offset
    // would replay over the value we are leaving in place (see
    // is_cst_val1_and_v_equal_first_value). The stored-value test is what made
    // this early-out the frequent one: a bool re-set to true over a pending
    // `false` lost the true in deferred_path_validation.
    if (!chunk.pending_any()) [[likely]]
    {
        if constexpr (is_not_bool)
        {
            if (v == second_value &&
                (values_loc == chunk.get_cst_val2(BaseSpec::ind) || values_loc[num] == second_value))
                return v;
        }
        else
        {
            if (v && (values_loc == chunk.get_cst_val2() || values_loc[num]))
                return true;
        }
    }

    // Synchronous core path (fast path)
    // By reference: v is this call's parameter, alive for the synchronous
    // call (see Proxy2A::assign_impl); the deferred path copies it.
    auto lbd_equal_core = [this, &v]()
    {
        if constexpr (chunk_config::use_compression_without_counter)
        {
            this->mark_for_compression_if_needed(v);
        }
        perform_core_assignment(v);
    };

    [[maybe_unused]] const bool need_async = Base2::write_with_optional_lock_with_set_ptr(lbd_equal_core);

    if constexpr (chunk_config::use_compression_without_counter)
    {
        // Asynchronous slow path: simple but safe
        if (need_async)
        {
            using CtxType = ArithmeticAffectationContext<T, std::remove_cvref_t<U>, ChunkType, Proxy2<T>>;

            // Capture only location + operand
            auto* ctx = pool_new_ctx<CtxType>(*this, v, ptr_on_array);

            auto lbd_equal_asynch = [ctx]() // the operand lives in the context: no second copy in the closure
            {
                auto&     chunk = ctx->chunk;
                const int num   = ctx->num;

                const Proxy2 proxy = Base::template make_proxy_with_info<Proxy2>(ctx);

                // Deferred replay: pin the frozen per-op geometry so create_value
                // below materialises at proxy.chunk_size(), not the flushing thread's
                // ambient thread_chunk_size(). Restored on scope exit. Dynamic only.
                [[maybe_unused]] const DynChunkSizeGuard sz_guard{proxy.chunk_size()};

                const bool is_cst1 = proxy.is_cst_val1();
                const bool is_cst2 = proxy.is_cst_val2();

                if (is_cst1 || is_cst2)
                {
                    if constexpr (is_not_bool)
                    {
                        auto inf = Proxy2<T>::get_info_second_value(ctx->ptr_on_array);
                        chunk.create_value(is_cst2, inf);
                    }
                    else
                    {
                        chunk.create_value_bool(is_cst2);
                    }
                }

                if constexpr (build_config::any_check)
                {
                    const bool is_cst1_loc = proxy.is_cst_val1();
                    if (is_cst1_loc)
                        INVARIANT_FAILED("[deferred write_body] chunk still cst_val1 after create_value "
                                         "(chunk={}, num={})",
                                         static_cast<const void*>(&chunk), num);

                    const bool is_cst2_loc = proxy.is_cst_val2();
                    if (is_cst2_loc)
                        INVARIANT_FAILED("[deferred write_body] chunk still cst_val2 after create_value "
                                         "(chunk={}, num={})",
                                         static_cast<const void*>(&chunk), num);
                }

                T* values = chunk.values_load();
                // Check-build guard: read-side sanity check to correlate with producer-side
                // instrumentation (values_store / CAS / freelist / get_cst_val2).
                // Producer traces should fire first; if only this fires, a store site
                // is missing instrumentation.
                CHUNK_CHECK_VALUES_PTR_C(values, "deferred.write_body.values_load", &chunk);
                values[num] = static_cast<T>(ctx->operand);
            };

            Base::enqueue_deferred_write(ctx, std::move(lbd_equal_asynch));
        }
    }

    return static_cast<T>(v);
}

/**
 * @brief Handles simple compound operations (+, -, bit operations)
 * @tparam Lambda Callable that performs the operation
 * @param operation Lambda taking T* and performing operation on values_[num_]
 *
 * @details
 * Pattern for operations like +=, -=, &=, |=, ^=, <<=, >>=, ++, --
 * 1. Allocate storage if compressed
 * 2. Read old value
 * 3. Apply operation
 * 4. Track for compression
 */
template <typename T>
void Proxy2<T>::ensure_writable_storage()
{
    // Route this materialisation to THIS array's chunk-size pool -- the sync
    // twin of the fix in Proxy2AFastCompPolicy::ensure_writable_storage_async.
    // Without it, chunk.create_value() below allocates at whatever the STALE
    // per-thread thread_chunk_size happens to be (left behind by an unrelated
    // array's create_value()/rechunk_to(), neither of which restores it --
    // only Proxy2<T>::create_value() self-corrects it, and this function
    // calls chunk.create_value() directly, bypassing that wrapper), while
    // num/store_value use this array's OWN size_policy_ -- heap-buffer-
    // overflow the moment num exceeds the stale (smaller) allocation.
    if constexpr (build_config::chunk_dynamic)
    {
        shrink_array::set_thread_chunk_size(this->chunk_size());
        shrink_array::set_thread_alloc_key(this->alloc_key());
    }

    // Verbatim extraction of the materialisation sequence previously inlined at
    // the top of execute_simple_operation (see header note). Any behaviour
    // change here is a regression by definition.
    if constexpr (chunk_config::use_compression_with_counter && requires { this->set_info_second_value_ind(); })
    {
        this->set_info_second_value_ind();
    }

    bool is_val2;
    if constexpr (is_not_bool)
        is_val2 = is_cst_val2(BaseSpec::ind);
    else
        is_val2 = is_cst_val2();

    if (is_val2)
    {
        chunk.create_value(true, inf_array);
        if constexpr (chunk_config::use_compression_with_counter)
            chunk.set_nb_val_eq_second_value(this->chunk_size());
    }

    const bool is_val1 = Base::is_cst_val1();

    if (is_val1)
    {
        chunk.create_value(false, inf_array);
        if constexpr (chunk_config::use_compression_with_counter)
            chunk.set_nb_val_eq_first_value(this->chunk_size());
    }
}

template <typename T>
template <typename LAMBDA>
void Proxy2<T>::execute_simple_operation(LAMBDA operation)
{
    ensure_writable_storage();

    // Capture state AFTER create_value: before it, `values` still points at the
    // shared cst buffer, so v_old / was_* would read the wrong storage and the
    // counter transitions below would desync (driving the chunk to a spurious
    // "full second value" and re-compressing it — the tab_int 10-vs-1 bug).
    const in_t<T> v_old = read_values();

    [[maybe_unused]] bool was_second_value = false;
    [[maybe_unused]] bool was_first_value  = false;

    if constexpr (chunk_config::use_compression_with_counter)
    {
        was_second_value = (v_old == second_value);
        was_first_value  = (v_old == static_cast<T>(Chunk2<T>::first_value));
    }

    operation();

    if constexpr (chunk_config::use_compression_without_counter)
    {
        this->mark_for_compression_if_needed_with_value(v_old, read_values());
    }

    if constexpr (chunk_config::use_compression_with_counter)
    {
        // Transitions OUT: position no longer holds the constant it held before.
        if (was_second_value && values[num] != second_value)
        {
            chunk.dec_nb_val_eq_second_value();
        }
        else if (was_first_value && values[num] != static_cast<T>(Chunk2<T>::first_value))
        {
            chunk.dec_nb_val_eq_first_value();
        }

        // Transitions IN: position now holds a constant it did not hold before.
        // Guarded by !was_* to avoid double counting.
        if (!was_second_value && values[num] == second_value)
        {
            chunk.inc_nb_val_eq_second_value();
            if (chunk.is_full_second_value(this->chunk_size()))
                this->manage_full_second_value();
        }
        else if (!was_first_value && values[num] == static_cast<T>(Chunk2<T>::first_value))
        {
            chunk.inc_nb_val_eq_first_value();
            if (chunk.is_full_first_value(this->chunk_size()))
                this->manage_full_first_value();
        }
    }
}

// ── execute_arithmetic_compound_operation ─────────────────────────────────
// Two entirely different implementations depending on compression mode, one
// per Proxy2FastCompPolicy specialisation.

template <typename Derived, typename T, typename ChunkType, bool Enabled>
template <typename LAMBDA>
void Proxy2FastCompPolicy<Derived, T, ChunkType, Enabled>::execute_simple_operation_async(auto* ctx, LAMBDA operation)
{
    auto& chunk = ctx->chunk;

    const Proxy2 proxy = Base::template make_proxy_with_info<Derived>(ctx);

    const bool is_val2 = proxy.is_cst_val2();
    const bool is_val1 = proxy.is_cst_val1();

    if constexpr (is_not_bool)
    {
        if (is_val2)
            chunk.create_value(true, proxy.get_inf_array());
        else if (is_val1)
            chunk.create_value(false, proxy.get_inf_array());
    }

    const in_t<T> v_old  = read_values_async(ctx);
    T*      values = chunk.values_load();
    operation(values);

    const in_t<T> v_new = read_values_async(ctx);
    proxy.mark_for_compression_if_needed_with_value(v_old, v_new);
}

/**
 * @brief Handles arithmetic compound operations with operator (*, /, %),
 *        deferred-compression form (Enabled=true policy).
 * @tparam Op  Operation type (e.g., std::multiplies)
 * @tparam Lambda Callable `(op)` that performs the operation
 * @param op Operation to apply
 * @param operand Right-hand operand
 * @param operation Called when the chunk had to be materialised
 */
template <typename Derived, typename T, typename ChunkType, bool Enabled>
template <typename OP, typename LAMBDA, typename U>
void Proxy2FastCompPolicy<Derived, T, ChunkType, Enabled>::execute_arithmetic_compound_operation(OP     op,
                                                                                                 const U& operand,
                                                                                                 LAMBDA operation)
{
    // Same fix as ensure_writable_storage(): this function may call
    // chunk.create_value() below (via the will_be_not_second_value branch)
    // without going through Proxy2<T>::create_value()'s self-correcting
    // wrapper -- pin the ambient size to THIS array's own geometry first.
    if constexpr (build_config::chunk_dynamic)
    {
        shrink_array::set_thread_chunk_size(this->chunk_size());
        shrink_array::set_thread_alloc_key(this->alloc_key());
    }

    using CalcType = std::common_type_t<T, U>;

    const in_t<T> v_old = read_values();

    // Perform math in high precision
    const CalcType v_new_precise = op(static_cast<CalcType>(v_old), static_cast<CalcType>(operand));

    if constexpr (is_not_bool)
        second_value = derived().make_second_value();

    // Compare the PRECISE result with the second constant value
    const bool will_be_not_second_value =
        (v_new_precise != static_cast<CalcType>(second_value)) && derived().is_cst_val2();

    if (will_be_not_second_value)
    {
        chunk.create_value(true, inf_array);
        operation(op);
    }
    else if (Base2::is_not_cst_val2())
    {
        values[num] = static_cast<T>(v_new_precise);
        this->mark_for_compression_if_needed_with_value(v_old, static_cast<T>(v_new_precise));
    }
}

template <typename Derived, typename T, typename ChunkType, bool Enabled>
template <typename OP, typename LAMBDA, typename U>
void Proxy2FastCompPolicy<Derived, T, ChunkType, Enabled>::execute_arithmetic_compound_operation_async(
    auto*    ctx,
    OP       op,
    const U& operand,
    LAMBDA   operation) const
{
    using CalcType     = std::common_type_t<T, U>;
    const Proxy2 proxy = Base::template make_proxy_with_info<Derived>(ctx);
    const in_t<T> v_old = read_values_async(ctx);

    const CalcType v_new_precise = op(static_cast<CalcType>(v_old), static_cast<CalcType>(operand));

    const bool will_be_not_second_value =
        (v_new_precise != static_cast<CalcType>(proxy.get_second_value())) && proxy.is_cst_val2();

    if (will_be_not_second_value)
    {
        ctx->chunk.create_value(true, proxy.get_inf_array());
        operation(op);
    }
    else if (!proxy.is_cst_val2())
    {
        T* values_ptr        = ctx->chunk.values_load();
        values_ptr[ctx->num] = static_cast<T>(v_new_precise);
        proxy.mark_for_compression_if_needed_with_value(v_old, static_cast<T>(v_new_precise));
    }
}

/**
 * @brief Serial (counter) form of execute_arithmetic_compound_operation()
 *        (Enabled=false policy).
 * @tparam Lambda Callable `(op, T* values)`.
 */
template <typename Derived, typename T, typename ChunkType>
template <typename OP, typename LAMBDA, typename U>
void Proxy2FastCompPolicy<Derived, T, ChunkType, false>::execute_arithmetic_compound_operation(OP     op,
                                                                                               const U& operand,
                                                                                               LAMBDA operation)
{
    // Same fix as ensure_writable_storage(): see that function's comment.
    if constexpr (build_config::chunk_dynamic)
    {
        shrink_array::set_thread_chunk_size(this->chunk_size());
        shrink_array::set_thread_alloc_key(this->alloc_key());
    }

    using CalcType = std::common_type_t<T, base_t<U>>;

    const bool was_not_written = !chunk.written(this->ind);

    if (was_not_written)
    {
        const CalcType intermediate =
            op(static_cast<CalcType>(Base::values[Base::num]), static_cast<CalcType>(operand));
        // Built where it is used (was a default-constructed `T res` at function
        // scope, then assigned: two writes of a struct for one value).
        const in_t<T> res = static_cast<T>(intermediate);

        const bool will_be_not_second_value = (res != second_value) && derived().is_cst_val2();

        if (will_be_not_second_value)
        {
            chunk.create_value(true, inf_array);
            if constexpr (chunk_config::use_compression_with_counter)
                chunk.set_nb_val_eq_second_value(this->chunk_size() - 1);
            operation(op, Base::values);
            return;
        }
    }

    operation(op, Base::values);
}

template <typename T>
template <typename LAMBDA>
void Proxy2<T>::execute_inplace_operation(LAMBDA operation)
{
    // Same fix as ensure_writable_storage(): see that function's comment.
    if constexpr (build_config::chunk_dynamic)
    {
        shrink_array::set_thread_chunk_size(this->chunk_size());
        shrink_array::set_thread_alloc_key(this->alloc_key());
    }

    if constexpr (chunk_config::use_compression_with_counter && requires { this->set_info_second_value_ind(); })
    {
        this->set_info_second_value_ind();
    }

    const bool is_val2 = Base2::is_cst_val2();
    const bool is_val1 = Base::is_cst_val1();

    [[maybe_unused]] T v_old{};
    if constexpr (chunk_config::use_compression_without_counter)
        v_old = read_values();

    if constexpr (is_not_bool)
    {
        if (is_val2 || is_val1)
        {
            Base::chunk.create_value(is_val2, inf_array);
            if constexpr (chunk_config::use_compression_with_counter)
            {
                if (is_val2)
                    chunk.set_nb_val_eq_second_value(this->chunk_size());
                else if (is_val1)
                    chunk.set_nb_val_eq_first_value(this->chunk_size());
            }
        }
    }

    // Capture position-level state AFTER any create_value: before it, `values`
    // still points at the shared cst_val1/cst_val2 buffer, so values[num] would
    // read the wrong storage and corrupt the counter transitions below (the
    // chunk could be wrongly driven to "full second value" and re-compressed).
    [[maybe_unused]] bool was_second_value = false;
    [[maybe_unused]] bool was_first_value  = false;

    if constexpr (chunk_config::use_compression_with_counter)
    {
        was_second_value = (values[num] == second_value);
        was_first_value  = (values[num] == static_cast<T>(Chunk2<T>::first_value));
    }

    operation(Base::values);

    if constexpr (chunk_config::use_compression_with_counter)
    {
        // Transitions OUT: the position no longer holds the constant it held before.
        if (was_second_value && values[num] != second_value)
        {
            chunk.dec_nb_val_eq_second_value();
        }
        else if (was_first_value && values[num] != static_cast<T>(Chunk2<T>::first_value))
        {
            chunk.dec_nb_val_eq_first_value();
        }

        // Transitions IN: the position now holds a constant it did not hold before.
        // Test the actual stored value, NOT is_cst_val2(ind) (which checks the
        // storage pointer and is false right after create_value decompressed the
        // chunk, so it would miss the increment and desync the counter).
        if (!was_second_value && values[num] == second_value)
        {
            chunk.inc_nb_val_eq_second_value();
            if (chunk.is_full_second_value(this->chunk_size()))
                this->manage_full_second_value();
        }
        else if (!was_first_value && values[num] == static_cast<T>(Chunk2<T>::first_value))
        {
            chunk.inc_nb_val_eq_first_value();
            if (chunk.is_full_first_value(this->chunk_size()))
                this->manage_full_first_value();
        }
    }

    if constexpr (chunk_config::use_compression_without_counter)
    {
        this->mark_for_compression_if_needed_with_value(v_old, read_values());
    }
}

// ── execute_inplace_operation_async ───────────────────────────────────────
// Only present when fast block compression is active (the Enabled=true policy).

template <typename Derived, typename T, typename ChunkType, bool Enabled>
template <typename LAMBDA>
void Proxy2FastCompPolicy<Derived, T, ChunkType, Enabled>::execute_inplace_operation_async(auto*  ctx,
                                                                                           LAMBDA operation) const
{
    auto&     chunk = ctx->chunk;
    const int num   = ctx->num;

    const in_t<T> v_old = read_values_async(ctx);
    const Proxy2 proxy = Base::template make_proxy_with_info<Derived>(ctx);

    const bool is_val1 = chunk.is_cst_val1();
    const bool is_val2 = proxy.is_cst_val2();

    if (is_val1 || is_val2)
    {
        if constexpr (is_not_bool)
        {
            auto inf = Proxy2<T>::get_info_second_value(ctx->ptr_on_array);
            chunk.create_value(is_val2, inf);
        }
        else
        {
            chunk.create_value_bool(is_val2, get_thr_rank_fast());
        }
    }

    T* values = chunk.values_load();
    operation(values);

    // The value just written, read for the counters: nothing writes it
    // afterwards, so in_t (a reference for a struct) rather than a copy.
    const in_t<T> v_new = values[num];
    Base2::mark_for_compression_if_needed_with_value_async(proxy, v_old, v_new);
}

template <typename T>
template <typename U, typename Op>
T Proxy2<T>::apply_compound_operation([[maybe_unused]] Op   op,
                                      in_t<U>               v,
                                      [[maybe_unused]] bool check_first_value_result) noexcept
{
    // Cross-rank RMW materialisation (CHUNK_SYNC_ON_READ): pull any foreign
    // deferred write to this offset to applied state before reading v_old, so the
    // RMW cannot clobber it. Precise gate, outside begin_write. See
    // ProxyBase::arithmetic_assign for the full rationale.
    chunk.reader_sync(num);

    if constexpr (chunk_config::use_compression_with_counter)
    {
        BaseSpec::set_info_second_value_ind(); // Initialize second_value / ind — needed by both branches below

        const in_t<T>    v_old_snapshot = values[num];
        const bool is_first_value = (v_old_snapshot == static_cast<T>(Chunk2<T>::first_value));

        if (!is_first_value) [[likely]]
        {
            if constexpr (build_config::do_check)
            {
                global_check_write_nan(v, num);
            }

            const bool was_second_value  = (v_old_snapshot == second_value);
            const bool was_second_value0 = was_second_value && Base2::is_not_cst_val2();

            execute_arithmetic_compound_operation(
                op, v,
                // v by reference (this call's parameter, alive for the call);
                // `current` in_t: the right-hand side is evaluated in full
                // before the store, so a reference into the slot is read once,
                // before it is written.
                [this, &v](auto op_func, T* val_ptr)
                {
                    using LocalCalcType   = std::common_type_t<T, base_t<U>>;
                    const in_t<T> current = val_ptr[num];
                    val_ptr[num] =
                        static_cast<T>(op_func(static_cast<LocalCalcType>(current), static_cast<LocalCalcType>(v)));
                });

            // Read after the operation, compared before each branch's action
            // (a manage_full_* may fold the chunk, and nothing reads v_new
            // after it): in_t, no copy of a struct.
            const in_t<T> v_new = values[num];
            if (check_first_value_result && Base::is_not_cst_val1() && v_new == Chunk2<T>::first_value)
            {
                chunk.inc_nb_val_eq_first_value();
                if (chunk.is_full_first_value(this->chunk_size()))
                    this->manage_full_first_value();
            }
            else if (was_second_value0 && v_new != second_value)
            {
                chunk.dec_nb_val_eq_second_value();
            }
            else if (!was_second_value && v_new == second_value)
            {
                chunk.inc_nb_val_eq_second_value();
                if (chunk.is_full_second_value(this->chunk_size()))
                    this->manage_full_second_value();
            }
        }
    }

    if constexpr (chunk_config::use_compression_without_counter)
    {
        const in_t<T> v_old = read_values();
        if (v_old != Chunk2<T>::first_value)
            Base::apply_atomic_op(op, v, v_old);
    }

    return static_cast<T>(v);
}

} // namespace shrink_array