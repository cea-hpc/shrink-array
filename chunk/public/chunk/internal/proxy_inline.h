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

#include <atomic>      // std::atomic_thread_fence, std::memory_order_seq_cst
#include <iostream>    // std::cerr (the mismatch diagnostic)
#include <type_traits> // std::common_type_t
#include <utility>     // std::move

#include "chunk/internal/in_t.h"
#include "chunk/internal/proxy.h" // the declarations these bodies belong to

namespace shrink_array
{

template <typename T>
const Proxy<T>& Proxy<T>::operator=(const Proxy<T>& from) noexcept
{
    if (&from != this)
    {
        const in_t<T> v = from.values[from.num];

        const bool need_alloc = is_cst_val();

        if (v == zero && need_alloc)
        {
            return *this;
        }

        if constexpr (chunk_config::stats)
        {
            increm_chunk_write<ChunkType>(v);
        }

        create_value(need_alloc);

        if constexpr (build_config::do_check)
        {
            global_check_write_nan(v, num);
        }
        values[num] = v;
    }

    return *this;
}

template <typename T>
void Proxy<T>::perform_core_assignment(in_t<T> v) const noexcept
{
    if constexpr (chunk_config::use_compression_with_counter)
    {
        if (v == static_cast<T>(Chunk<T>::first_value))
        {
            if (is_not_cst_val())
            {
                if (values[num] != Chunk<T>::first_value)
                {
                    chunk.inc_nb_val_eq_zero();
                }

                if (chunk.is_full_zero(this->chunk_size()))
                {
                    this->manage_full_zero();
                    return;
                }
            }
        }
        else if (is_not_cst_val() && values[num] == Chunk<T>::first_value)
        {
            chunk.dec_nb_val_eq_zero();
        }
    }

    if constexpr (chunk_config::stats)
    {
        increm_chunk_write<ChunkType>(v);
    }

    const bool need_alloc = is_cst_val();

    if (need_alloc)
    {
        // Zero-overhead state helper: members exist only if Enabled is true
        struct
        {
            [[maybe_unused]] bool was_cst;

            // Captured before allocation to determine correct counter initialization
            void capture(const Proxy* p)
            {
                if constexpr (chunk_config::use_compression_with_counter)
                    was_cst = p->is_cst_val();
            }

            // Applied after allocation to sync counters with the new heap block
            void apply(const Proxy* p, T val)
            {
                if constexpr (chunk_config::use_compression_with_counter)
                {
                    if (was_cst)
                        p->chunk.set_nb_val_eq_zero(p->chunk_size() - 1);
                    else if (val == Chunk<T>::first_value)
                        p->chunk.inc_nb_val_eq_zero();
                }
            }
        } state;

        state.capture(this); // Becomes a No-Op if counters are disabled

        create_value(); // Perform heap allocation

        state.apply(this, v); // Becomes a No-Op if counters are disabled
    }

    Base::store_value(v);

    if constexpr (build_config::do_check)
    {
        global_check_write_nan(v, num);
    }
}

/**
 * @brief Core unified logic for compression marking.
 * Handles both standard and 'pre' checks by comparing memory with an expected state.
 */
template <typename Derived, typename T, bool Enabled>
void ProxyFastCompPolicy<Derived, T, Enabled>::mark_for_compression_core(in_t<T> old_value,
                                                                         in_t<T> new_value,
                                                                         in_t<T> expected_in_memory) const
{
    // Check if transition qualifies for compression OR if a data race occurred (value changed)
    if (should_mark_for_compression(old_value, new_value) || (read_values() != expected_in_memory)) [[unlikely]]
    {
        Base::add_to_compression_list();
        return;
    }
}

template <typename T>
template <typename OP, typename LBD, typename U>
void Proxy<T>::execute_arithmetic_compound_operation(OP op, [[maybe_unused]] const U& v, LBD lbd)
{
    // Determine the high-precision calculation type (STL parity)
    using CalcType = std::common_type_t<T, base_t<U>>;

    if constexpr (chunk_config::use_fast_block_compression)
    {
        set_info_on_chunk();
        FastCompPolicy::perform_compression();
    }

    const bool was_not_written = !chunk.written();

    if (was_not_written)
    {
        // Decompress the chunk (allocate memory)
        create_value();
        // Call the lambda with two arguments: the operator and the pointer to values
        lbd(op);
    }
    else
    {
        const in_t<T> v_old = values[num];

        // 1. Perform math in high precision using CalcType
        const CalcType v_new_precise = op(static_cast<CalcType>(v_old), static_cast<CalcType>(v));

        // 2. Truncate back to storage type T
        const in_t<T> v_new = static_cast<T>(v_new_precise);

        values[num] = v_new;

        if constexpr (chunk_config::use_fast_block_compression)
        {
            // Trigger compression marking logic if the result is back to zero
            this->mark_for_compression_if_needed_with_value(v_old, v_new);
        }
    }
}

template <typename Derived, typename T, bool Enabled>
template <typename OP, typename LAMBDA, typename U>
void ProxyFastCompPolicy<Derived, T, Enabled>::execute_arithmetic_compound_operation_async(auto*              ctx,
                                                                                           OP                 op,
                                                                                           [[maybe_unused]] const U& v,
                                                                                           LAMBDA             lbd) const
{
    // Determine the high-precision calculation type for STL parity
    using CalcType = std::common_type_t<T, base_t<U>>;

    const bool was_not_written = !ctx->chunk.written();

    if (was_not_written)
    {
        // Decompress: the chunk was currently in a constant state
        ctx->chunk.create_value();

        // Execute the write logic using the newly allocated memory pointer
        lbd(op);
    }
    else
    {
        // Rebuild a temporary proxy to access maintenance metadata
        const Proxy proxy = Base::template make_proxy_with_info<Derived>(ctx);

        auto&     chunk_loc = ctx->chunk;
        const int num       = ctx->num;

        // Reload the current values pointer (with acquire semantics)
        T* values_ptr = chunk_loc.values_load();

        const in_t<T> old_value = values_ptr[num];

        // 1. Perform math in high precision to avoid rounding drift
        const CalcType res = op(static_cast<CalcType>(old_value), static_cast<CalcType>(v));

        // 2. Truncate result back to the array's storage type
        const in_t<T> final_val = static_cast<T>(res);

        // 3. Store the value in physical memory
        values_ptr[num] = final_val;

        // 4. Trigger the compression check logic (e.g. if the result is zero)
        mark_for_compression_if_needed_with_value_async(proxy, old_value, final_val);
    }
}

template <typename T>
template <typename U>
T Proxy<T>::assign_impl(in_t<U> v) noexcept
{
    if (is_cst_val_and_v_equal_zero(v))
    {
        return v;
    }

    const bool need_alloc = is_cst_val();

    if constexpr (chunk_config::use_compression_with_counter)
    {
        if (v == zero)
        {
            if (need_alloc)
            {
                return zero;
            }
            else
            {
                if (values[num] != zero)
                {
                    chunk.inc_nb_val_eq_zero();
                }

                if (chunk.is_full_zero(this->chunk_size()))
                {
                    this->manage_full_zero();
                    return zero;
                }
            }
        }
        else if (!is_cst_val() && values[num] == zero)
        {
            chunk.dec_nb_val_eq_zero();
        }
    }
    else
    {
        // Same early-out as is_cst_val_and_v_equal_zero() above, same rule:
        // not over a pending deferred op (deferred_path_validation, chunk #7
        // offset 0: the chunk's first write deferred, then `= 0` skipped here,
        // then the replay).
        if (v == zero && need_alloc && !chunk.pending_any())
        {
            return zero;
        }
    }

    // By reference: v is this call's parameter, alive for the synchronous
    // call (see Proxy2A::assign_impl); the deferred path copies it.
    [[maybe_unused]] auto lbd_equal_core = [this, &v]()
    {
        // Capture the value that is there BEFORE the write, reading it THROUGH the
        // chunk. Two distinct traps sit here:
        //  - it must be read before perform_core_assignment(), which materialises
        //    the chunk: a freshly materialised buffer is all-zero, so reading
        //    afterwards always yields 0;
        //  - it must NOT go through the proxy's `values`, which is a reference to
        //    the chunk pointer captured when the proxy was built. On a compressed
        //    chunk that reference still designates the shared constant buffer, so
        //    values[num] returned the constant (zero) whatever the real content.
        // Either way old_value came back as 0, the non-zero -> zero transition was
        // never seen, and no chunk was ever queued for compression under OpenMP
        // (CHUNK_STATS: 70M calls from this path, 0 marks, compress flat at 0).
        // read_value() re-loads the atomic pointer at call time, so it sees the
        // buffer that is actually live now.
        const in_t<T> old_value = chunk.read_value(num);

        perform_core_assignment(v);

        if constexpr (chunk_config::use_fast_block_compression)
        {
            this->mark_for_compression_if_needed_with_value(old_value, v);
        }
    };

    if constexpr (chunk_config::use_fast_block_compression)
    {
        const bool have_to_fill_aynch_op = write_with_optional_lock(lbd_equal_core);
        if (have_to_fill_aynch_op)
        {
            using CtxType = ArithmeticAffectationProxyContext<T, std::remove_cvref_t<U>>;

            auto* ctx = pool_new_ctx<CtxType>(*this, v, ptr_on_array);

            auto lbd_equal_asynch = [ctx] // the operand lives in the context: no second copy in the closure
            {
                auto&     chunk = ctx->chunk;
                const int num   = ctx->num;

                const Proxy proxy = Base::template make_proxy_with_info<Proxy>(ctx);

                const bool is_cst = proxy.is_cst_val();

                // Capture the previous value BEFORE materialising. create_value()
                // installs a fresh all-zero buffer, so reading values[num] after it
                // always returned 0 and the non-zero -> zero transition below could
                // never fire: no chunk was ever queued for compression on this path
                // (CHUNK_STATS: 81M entries in mark_for_compression_core, 0 marks,
                // compress stuck at 0 for the whole run). On a still-compressed
                // chunk values[] holds the constant, which is the correct old value.
                const in_t<T> v_old = chunk.read_value(num);

                if (is_cst)
                {
                    chunk.create_value();
                }

                T* values = chunk.values_load();

                values[num] = static_cast<T>(ctx->operand);

                proxy.mark_for_compression_if_needed_with_value(v_old, values[num]);
            };

            Base::enqueue_deferred_write(ctx, std::move(lbd_equal_asynch));
        }
    }
    else
    {
        // Serial / non-OpenMP path. assign_impl's counter block above ALREADY
        // maintained nb_val_eq_zero for the already-armed transitions. We must
        // therefore NOT go through lbd_equal_core -> perform_core_assignment,
        // whose own counter block (and manage_full_zero) would run the counting
        // a SECOND time -> nb_val_eq_zero drifts and manage_full_zero fires on a
        // chunk that is not actually all-zero (checksum failure). In the OpenMP
        // fast path this never happened because use_compression_with_counter is
        // false there (block compression replaces the counter), so both counter
        // blocks are compiled out. Here we apply the write directly (arm + store,
        // like the OpenMP async lambda), doing ONLY the arm-time counter init
        // that assign_impl's block does not do for a non-zero write.
        const bool need_alloc = is_cst_val();
        if (need_alloc)
        {
            create_value();
            if constexpr (chunk_config::use_compression_with_counter)
            {
                chunk.set_nb_val_eq_zero(this->chunk_size() - 1);
            }
        }
        Base::store_value(v);
    }

    if constexpr (chunk_config::stats)
    {
        increm_chunk_write<ChunkType>(v);
    }

    return v;
}

template <typename T>
template <typename LAMBDA>
void Proxy<T>::execute_simple_operation(LAMBDA operation)
{
    // Snapshot BEFORE any arming: needed to maintain nb_val_eq_zero across the
    // write for an ALREADY-ARMED chunk (the arming case is covered by the
    // set_nb_val_eq_zero below). Without this, a sync += that turns an element
    // from 0 into non-zero never decremented the zero-counter, so it drifted up
    // to __chunk_size and manage_full_zero() wrongly re-compressed a chunk that
    // was not actually all-zero (non-OMP-only: the OpenMP += path is async and
    // maintains the counter separately). Mirrors execute_inplace_operation.
    [[maybe_unused]] const bool was_first_value = values[num] == static_cast<T>(Chunk<T>::first_value);

    const bool is_val1 = Base::is_cst_val1();

    if (is_val1)
    {
        create_value();

        if constexpr (chunk_config::use_compression_with_counter)
        {
            chunk.set_nb_val_eq_zero(this->chunk_size() - 1);
        }
    }

    operation();

    if constexpr (chunk_config::use_compression_with_counter)
    {
        // Already-armed transitions (arming handled above).
        if (was_first_value && !is_val1 && values[num] != static_cast<T>(Chunk<T>::first_value))
        {
            chunk.dec_nb_val_eq_zero();
        }
        if (values[num] == static_cast<T>(Chunk<T>::first_value) && !was_first_value)
        {
            chunk.inc_nb_val_eq_zero();
            if (chunk.is_full_zero(this->chunk_size()))
            {
                this->manage_full_zero();
            }
        }
    }
}

template <typename Derived, typename T, bool Enabled>
template <typename LAMBDA>
void ProxyFastCompPolicy<Derived, T, Enabled>::execute_simple_operation_async(auto* ctx, LAMBDA operation)
{
    const bool is_val1 = ctx->chunk.is_cst_val1();

    if (is_val1)
    {
        ctx->chunk.create_value();
    }

    // Reload the real values pointer from the chunk and pass it to the lambda
    T* values = ctx->chunk.values_load();
    operation(values);
}

template <typename T>
template <typename LAMBDA>
void Proxy<T>::execute_inplace_operation(LAMBDA operation)
{
    [[maybe_unused]] const bool was_first_value = values[num] == static_cast<T>(Chunk<T>::first_value);

    const bool         is_val = is_cst_val();
    [[maybe_unused]] T v_old  = T{};

    if constexpr (chunk_config::use_fast_block_compression)
    {
        v_old = read_values();
    }

    if (is_val)
    {
        // Through Proxy::create_value(), NOT chunk.create_value() directly: the
        // wrapper pins the ambient geometry to THIS array's chunk_size() before
        // materialising. The direct call let ChunkOmpPolicy::create_value pop
        // the reuse list of whichever array wrote last -- with two geometries
        // alive at once, a 32-element buffer for a 64-element chunk (ASan: read
        // 248 bytes past a 256-byte region, from operator++). The v1 twin of
        // ff6954c9 ("pin ambient before Proxy2's direct create_value() calls").
        create_value();

        if constexpr (chunk_config::use_compression_with_counter)
        {
            chunk.set_nb_val_eq_zero(this->chunk_size() - 1);
        }
    }

    operation(Base::values);

    if constexpr (chunk_config::use_compression_with_counter)
    {
        if (was_first_value && !is_val && values[num] != Chunk<T>::first_value)
        {
            chunk.dec_nb_val_eq_zero();
        }
        if (values[num] == static_cast<T>(Chunk<T>::first_value) && !was_first_value)
        {
            chunk.inc_nb_val_eq_zero();
            if (chunk.is_full_zero(this->chunk_size()))
            {
                this->manage_full_zero();
            }
        }
    }

    if constexpr (chunk_config::use_fast_block_compression)
    {
        this->mark_for_compression_if_needed_with_value(v_old, read_values());
    }
}

template <typename Derived, typename T, bool Enabled>
template <typename LAMBDA>
void ProxyFastCompPolicy<Derived, T, Enabled>::execute_inplace_operation_async(auto* ctx, LAMBDA operation)
{
    auto&     chunk_loc = ctx->chunk;
    const int num       = ctx->num;

    const in_t<T> v_old = Base::read_values_async(ctx);

    // Rebuild a fully initialized Proxy2 (restores second_value + ind)
    const Proxy proxy  = Base::template make_proxy_with_info<Derived>(ctx);
    const bool  is_val = chunk_loc.is_cst_val1();

    if (is_val)
    {
        // Same as above: materialise through the rebuilt proxy so the ambient
        // geometry is pinned to the op's own (frozen) chunk_size() first.
        proxy.create_value();
    }

    // Reload the actual values pointer after potential allocation
    T* values = chunk_loc.values_load();

    // Apply the user-provided operation on the real values
    operation(values);

    // Re-read the updated value: nothing writes it afterwards, in_t (a
    // reference for a struct) rather than a copy.
    const in_t<T> v_new = values[num];

    mark_for_compression_if_needed_with_value_async(proxy, v_old, v_new);
}

template <typename T>
template <typename U, typename BinaryOp>
T Proxy<T>::apply_compound_operation(in_t<U> v, BinaryOp op)
{
    // Cross-rank RMW materialisation: pull any foreign deferred write first.
    chunk.reader_sync(num);

    if constexpr (is_not_bool)
    {
        if constexpr (chunk_config::use_compression_with_counter)
        {
            if (values[num] != zero)
            {
                if constexpr (build_config::do_check)
                {
                    global_check_write_nan(v, num);
                }

                execute_arithmetic_compound_operation(
                    op, static_cast<T>(v),
                    [this, &v](auto op_func) // v: this call's parameter, alive for the call
                    {
                        using LocalCalcType = std::common_type_t<T, base_t<U>>;

                        const in_t<T> result = static_cast<T>(
                            op_func(static_cast<LocalCalcType>(Base::load_value()), static_cast<LocalCalcType>(v)));

                        Base::store_value(result);
                    });

                if ((v == zero || values[num] == zero) && !is_cst_val())
                {
                    chunk.inc_nb_val_eq_zero();
                    if (chunk.is_full_zero(this->chunk_size()))
                    {
                        this->manage_full_zero();
                    }
                }
            }
        }
        else
        {
            const in_t<T> v_old = values[num];
            // we can't test values[num] if deferred mode is activate : because values[num] could be affect after
            if (v_old != zero || CompressionManagerType::have_deferred_queue())
            {
                Base::apply_atomic_op(op, static_cast<T>(v), v_old);

                // Do NOT read values[num] back here: apply_atomic_op may have
                // DEFERRED the op, and a foreign thread draining this chunk
                // (reader_sync -> flush_chunk_all_ranks) can be replaying it
                // right now -- TSan: write in
                // execute_arithmetic_compound_operation_async vs this read
                // (shrink_bench_chunk_check_256, tab4[i] *= ...). Report the
                // result the op yields from the value read above, which is what
                // the synchronous path stores; Proxy2 / Proxy2A likewise never
                // re-read after apply_atomic_op.
                using CalcType = std::common_type_t<T, base_t<U>>;
                return static_cast<T>(op(static_cast<CalcType>(v_old), static_cast<CalcType>(v)));
            }
            return v_old; // zero, and the op was skipped: still zero
        }
    }
    else
    {
        std::cerr << "operator not implemented for bool\n";
    }

    return values[num];
}

template <typename Derived, typename T, bool Enabled>
void ProxyFastCompPolicy<Derived, T, Enabled>::trigger_compression_if_needed(int thr_rank) noexcept
{
    auto& vec_tuple = manager_chunk::chunk_to_compress_potentially_thr[thr_rank];

    const int sz = static_cast<int>(vec_tuple.size());

    const int szm = manager_chunk::get_size_max_number_of_chunk_to_check_compress();

    // ✅ Track state transitions

    if (Base::should_trigger_compression(sz, szm))
    {
        CompressionManagerType::mark_list_dangerous_for_thread(thr_rank);

        if constexpr (chunk_config::verbose)
            std::cerr << "⚠️ List became dangerous: size=" << sz << "\n";
    }
}
template <typename Derived, typename T, bool Enabled>
void ProxyFastCompPolicy<Derived, T, Enabled>::perform_compression() const noexcept
{
    const int thr_rank = get_thr_rank_fast();

    const int szm = manager_chunk::get_size_max_number_of_chunk_to_check_compress();

    auto& vec_tuple = manager_chunk::chunk_to_compress_potentially_thr[thr_rank];

    const int sz_before = static_cast<int>(vec_tuple.size());

    if (Base::should_trigger_compression(sz_before, szm))
    {
        const bool have_queue = CompressionManagerType::have_deferred_queue();

        if (have_queue)
            return;

        std::atomic_thread_fence(std::memory_order_seq_cst);

        Chunk<T>::make_block_compression(thr_rank, this->self());
    }
    // Looked at: clear the flag the trigger raised (it only ever came down
    // through finish(), once per completed compression, against one raise per
    // write above the threshold). See mark_list_dangerous_for_thread.
    CompressionManagerType::mark_list_safe_for_thread(thr_rank);
}

} // namespace shrink_array