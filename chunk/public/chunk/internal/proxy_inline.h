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
#include <cstddef>     // std::size_t
#include <type_traits> // std::common_type_t
#include <memory>      // std::addressof (operator=: operator& is deleted)
#include <utility>     // std::move

#include "chunk/internal/in_t.h" // InT<T>
#include "chunk/internal/proxy.h" // the declarations these bodies belong to
#include "chunk/config.h"    // chunk_config::use_compression_with_counter, chunk_config::use_fast_block_compression
#include "chunk/PMP_CHUNK.h" // get_thr_rank_fast
#include "chunk/internal/arithmetic_context.h"           // ArithmeticAffectationProxyContext
#include "chunk/internal/concept_is_clean_assignement.h" // BaseT
#include "chunk/internal/read_reclaim.h"    // read_reclaim::PassScope (perform_compression / compress in a region)
#include "chunk/internal/invariant_check.h" // CHUNK_INVARIANT_FAILED
#include "chunk/log.h"                      // CHUNK_LOG, CHUNK_LOG_ALWAYS

namespace shrink_array
{

template <typename T>
const Proxy<T>& Proxy<T>::operator=(const Proxy<T>& from) noexcept
{
    // std::addressof: Proxy's operator& is deleted (no element address).
    if (std::addressof(from) != this)
    {
        // Read through operator T() (the dirty-bit gated read), complete before
        // the write starts: no reader/writer nesting even in the same chunk.
        const InT<T> v = static_cast<T>(from);

        // Write through the engine (handshake / deferred path, WAW-erase,
        // compression marking, zero counter). Do not store inline: it could
        // land in a CAS-swapped buffer, be overwritten by an older deferred
        // replay, or drift the zero counter.
        assign_impl<T>(v);
    }

    return *this;
}

template <typename T>
T* Proxy<T>::perform_core_assignment(T* vals, InT<T> v) const noexcept
{
    if constexpr (chunk_config::use_compression_with_counter)
    {
        if (v == static_cast<T>(Chunk<T>::first_value))
        {
            if (is_not_cst_val())
            {
                // raw-read: serial counter mode (no fast block compression, no concurrent pass)
                if (values_[num_] != Chunk<T>::first_value)
                {
                    chunk_.inc_nb_val_eq_zero();
                }

                if (chunk_.is_full_zero(this->chunk_size()))
                {
                    this->manage_full_zero();
                    return vals;
                }
            }
        }
        // raw-read: serial counter mode (no fast block compression, no concurrent pass)
        else if (is_not_cst_val() && values_[num_] == Chunk<T>::first_value)
        {
            chunk_.dec_nb_val_eq_zero();
        }
    }

    if constexpr (chunk_config::stats)
    {
        increm_chunk_write<ChunkType>(v);
    }

    const bool need_alloc = this->is_cst_val1(vals);

    if (need_alloc)
    {
        // Counter-mode state across the allocation; empty work otherwise.
        class
        {
          public:
            void capture(const Proxy* proxy) noexcept
            {
                if constexpr (chunk_config::use_compression_with_counter)
                {
                    was_cst = proxy->is_cst_val();
                }
            }

            void apply(const Proxy* proxy, T val)
            {
                if constexpr (chunk_config::use_compression_with_counter)
                {
                    if (was_cst)
                    {
                        proxy->chunk_.set_nb_val_eq_zero(proxy->chunk_size() - 1);
                    }
                    else if (val == Chunk<T>::first_value)
                    {
                        proxy->chunk_.inc_nb_val_eq_zero();
                    }
                }
            }

            [[maybe_unused]] bool was_cst = false;
        } state;

        state.capture(this);

        create_value();

        state.apply(this, v);

        // The one reload: a concurrent writer may have won create_value()'s CAS
        // (ours went back to the pool). Materialised now, so stable until the
        // session ends.
        vals = chunk_.values_load();
    }

    Base::store_value_at(vals, v);

    if constexpr (build_config::do_check)
    {
        global_check_write_nan(v, num_);
    }
    return vals;
}

/**
 * @brief Core unified logic for compression marking.
 * Marks on a qualifying transition, or when memory differs from the expected
 * state (a concurrent write).
 */
template <typename Derived, typename T, bool Enabled>
void ProxyFastCompPolicy<Derived, T, Enabled>::mark_for_compression_core(InT<T> old_value,
                                                                         InT<T> new_value,
                                                                         InT<T> expected_in_memory) const
{
    if (should_mark_for_compression(old_value, new_value) || (read_values() != expected_in_memory)) [[unlikely]]
    {
        Base::add_to_compression_list();
    }
}

template <typename Derived, typename T, bool Enabled>
void ProxyFastCompPolicy<Derived, T, Enabled>::mark_for_compression_core(InT<T>   old_value,
                                                                         InT<T>   new_value,
                                                                         InT<T>   expected_in_memory,
                                                                         const T* vals) const
{
    // Same test; the race check reads the element again, not the pointer.
    if (should_mark_for_compression(old_value, new_value) || (read_values_at(vals) != expected_in_memory)) [[unlikely]]
    {
        Base::add_to_compression_list();
    }
}

template <typename T>
template <typename Op, typename Body, typename U>
void Proxy<T>::execute_arithmetic_compound_operation(Op op, [[maybe_unused]] const U& v, Body lbd)
{
    using CalcType = std::common_type_t<T, BaseT<U>>;

    // No compression pass from here: this body runs inside the rank's writer
    // window, so start()'s "no writer" wait would see our own flag and run to
    // its budget, and perform_compression() would clear the dangerous flag the
    // post-end_write driver acts on. The engine drives the pass after end_write().

    const bool was_not_written = !chunk_.written();

    if (was_not_written)
    {
        create_value();
        lbd(op);
    }
    else
    {
        // const T, not InT<T>: a snapshot read before the store below and
        // used after it (in_t.h, corollary for local variables).
        // raw-read: write session -- the write/compress handshake keeps compressors off this chunk
        const T v_old = values_[num_];

        const CalcType v_new_precise = op(static_cast<CalcType>(v_old), static_cast<CalcType>(v));

        const auto v_new = static_cast<T>(v_new_precise);

        // raw-read: write session -- the write/compress handshake keeps compressors off this chunk
        values_[num_] = v_new;

        if constexpr (chunk_config::use_fast_block_compression)
        {
            this->mark_for_compression_if_needed_with_value(v_old, v_new);
        }
    }
}

// ─── The v1 replay bodies (`*_async`) and the hypothesis they rest on ──────
//
// The `_async` bodies below (execute_arithmetic_compound_operation_async,
// the deferred body of assign_impl, execute_simple_operation_async,
// execute_inplace_operation_async) are called ONLY from a flush drain
// (CompressionManager::flush_deferred() / flush_chunk_all_ranks()). Both
// drains announce the flush and wait out the in-flight hard compressors.
// Invariant: NO COMPRESSOR CAN CHANGE `values` DURING A REPLAY, so these
// bodies load the values pointer once and store through it, with no retry:
//
//   - the block passes (make_block_compression) go through
//     CompressionManager::start(), which refuses while a flush is announced;
//   - Chunk::compress_sweep_rank (phase-end sweep) and
//     RechunkHooks::rechunk_drop_candidates (after the re-chunk's own flush,
//     serial) call compress() directly, bypassing start(): safe only because
//     they run where no drain can. check_no_flush_outside_start()
//     (block_pass_precheck.h) trips in check builds otherwise.
//
// (The re-chunk's fold has no v1 branch: array_chunk_rechunk_inline.h.)
//
// Proxy2A's twins (proxy2A_inline.h) carry a retry loop instead. Do not add a
// compressor bypassing start() where a drain can run, nor call an `_async`
// body outside a drain: the store would land in a released buffer and be lost
// silently. check_replay_store() below catches it in check builds.

namespace internal
{
/// @brief Check builds: the chunk's values pointer is still the one the replay
///        stored through. A difference means the hypothesis above no longer
///        holds and the store is lost.
/// @param chunk     Chunk the replay wrote to.
/// @param stored_to Values pointer the store went through.
/// @param where     Replay body, for the message.
/// @param num       Element offset, for the message.
template <typename ChunkT, typename T>
inline void check_replay_store([[maybe_unused]] ChunkT&     chunk,
                               [[maybe_unused]] const T*    stored_to,
                               [[maybe_unused]] const char* where,
                               [[maybe_unused]] int         num) noexcept
{
    if constexpr (build_config::any_check)
    {
        if (chunk.values_load() != stored_to) [[unlikely]]
        {
            CHUNK_INVARIANT_FAILED("{}: values changed during a deferred replay (chunk={} num={}): a compressor ran "
                                   "outside CompressionManager::start() or the replay ran outside a flush drain; "
                                   "the store is lost",
                                   where, static_cast<const void*>(&chunk), num);
        }
    }
}
} // namespace internal

/// @brief Replay of a compound arithmetic operation (v1).
/// @pre Called from a flush drain only -- see "The v1 replay bodies" above.
template <typename Derived, typename T, bool Enabled>
template <typename Op, typename Lambda, typename U>
void ProxyFastCompPolicy<Derived, T, Enabled>::execute_arithmetic_compound_operation_async(auto* ctx,
                                                                                           Op    op,
                                                                                           [[maybe_unused]] const U& v,
                                                                                           Lambda lbd) const
{
    using CalcType = std::common_type_t<T, BaseT<U>>;

    const bool was_not_written = !ctx->chunk.written();

    if (was_not_written)
    {
        ctx->chunk.create_value();
        [[maybe_unused]] const T* const materialised = ctx->chunk.values_load();

        lbd(op);

        internal::check_replay_store(ctx->chunk, materialised, "execute_arithmetic_compound_operation_async", ctx->num);
    }
    else
    {
        const Proxy proxy = Base::template make_proxy_with_info<Derived>(ctx);

        auto&     chunk_loc = ctx->chunk;
        const int num       = ctx->num;

        T* values_ptr = chunk_loc.values_load();

        const T old_value = values_ptr[num];

        const CalcType res = op(static_cast<CalcType>(old_value), static_cast<CalcType>(v));

        const auto final_val = static_cast<T>(res);

        values_ptr[num] = final_val;
        internal::check_replay_store(chunk_loc, values_ptr, "execute_arithmetic_compound_operation_async", num);

        mark_for_compression_if_needed_with_value_async(proxy, old_value, final_val);
    }
}

template <typename T>
template <typename U>
T Proxy<T>::assign_impl(InT<U> v) noexcept
{
    this->check_not_stale();
    if (is_cst_val_and_v_equal_zero(v))
    {
        return v;
    }

    // Do not hoist `need_alloc = is_cst_val()` here: an acquire load paid by
    // every write for a test only the `v == zero` branch makes.
    if constexpr (chunk_config::use_compression_with_counter)
    {
        if (v == zero)
        {
            if (is_cst_val())
            {
                return zero;
            }

            // raw-read: serial counter mode (no fast block compression, no concurrent pass)
            if (values_[num_] != zero)
            {
                chunk_.inc_nb_val_eq_zero();
            }

            if (chunk_.is_full_zero(this->chunk_size()))
            {
                this->manage_full_zero();
                return zero;
            }
        }
        // raw-read: serial counter mode (no fast block compression, no concurrent pass)
        else if (!is_cst_val() && values_[num_] == zero)
        {
            chunk_.dec_nb_val_eq_zero();
        }
    }
    // Without the counter, the early-out for `= 0` on a compressed chunk is
    // is_cst_val_and_v_equal_zero() above, and only it. It must not apply over a
    // pending deferred op (the `= 0` would be skipped, then the replay lands).

    // By reference: v is this call's parameter, alive for the synchronous
    // call (see Proxy2A::assign_impl); the deferred path copies it.
    [[maybe_unused]] const auto lbd_equal_core = [this, &v]
    {
        // ONE load of the values pointer for the whole body, inside the session
        // (or a flush replay): no compressor can change it until end_write. A
        // concurrent WRITER can, but only constant -> materialised, so a
        // materialised capture is stable and a constant one is reloaded once
        // by perform_core_assignment() after create_value().
        T* vals = chunk_.values_load();

        // The value BEFORE the write. Two traps, both making old_value 0 so the
        // non-zero -> zero transition is never seen and nothing gets queued:
        //  - read it before perform_core_assignment(), which materialises the
        //    chunk (a fresh buffer is all-zero);
        //  - read it through a pointer loaded NOW, not the one captured when the
        //    proxy was built (it may designate the constant buffer since).
        // On a still-constant chunk this yields the constant: the right old
        // value. read_value() under CHUNK_STATS, for its read counters.
        const InT<T> old_value = [&]
        {
            if constexpr (chunk_config::stats)
            {
                return chunk_.read_value(num_);
            }
            else
            {
                return Base::read_values_at(vals);
            }
        }();

        vals = perform_core_assignment(vals, v);

        if constexpr (chunk_config::use_fast_block_compression)
        {
            this->mark_for_compression_if_needed_with_value(old_value, v, vals);
        }
    };

    if constexpr (chunk_config::use_fast_block_compression)
    {
        const bool have_to_fill_aynch_op = write_with_optional_lock(lbd_equal_core);
        if (have_to_fill_aynch_op)
        {
            using CtxType = ArithmeticAffectationProxyContext<T, std::remove_cvref_t<U>>;

            auto* ctx = pool_new_ctx<CtxType>(*this, v, ptr_on_array_);

            auto lbd_equal_asynch = [ctx] // the operand lives in the context: no second copy in the closure
            {
                auto&     chunk = ctx->chunk;
                const int num   = ctx->num;

                const auto proxy = Base::template make_proxy_with_info<Proxy>(ctx);

                // See "The v1 replay bodies" above. One load of the pointer,
                // reloaded once after a materialisation, as in lbd_equal_core.
                T* values = chunk.values_load();

                const bool is_cst = chunk.is_cst_val1(values);

                // Capture the old value BEFORE materialising: create_value()
                // installs an all-zero buffer, and the non-zero -> zero transition
                // would never fire (see lbd_equal_core).
                const InT<T> v_old = [&]
                {
                    if constexpr (chunk_config::stats)
                    {
                        return chunk.read_value(num);
                    }
                    else
                    {
                        return internal::element_of(values, static_cast<std::size_t>(num));
                    }
                }();

                if (is_cst)
                {
                    chunk.create_value();
                    values = chunk.values_load();
                }

                // raw-read: deferred replay, after materialisation -- flushes exclude the stealers
                values[num] = static_cast<T>(ctx->operand);
                internal::check_replay_store(chunk, values, "assign_impl (deferred)", num);

                // raw-read: deferred replay, after materialisation -- flushes exclude the stealers
                proxy.mark_for_compression_if_needed_with_value(v_old, values[num], values);
            };

            Base::enqueue_deferred_write(ctx, std::move(lbd_equal_asynch));
        }
    }
    else
    {
        // Serial path. assign_impl's counter block above ALREADY maintained
        // nb_val_eq_zero. Do NOT go through lbd_equal_core ->
        // perform_core_assignment: its counter block would count a SECOND time,
        // and manage_full_zero would fire on a chunk that is not all-zero.
        // Apply the write directly (arm + store), with only the arm-time
        // counter init.
        const bool need_alloc = is_cst_val();
        if (need_alloc)
        {
            create_value();
            if constexpr (chunk_config::use_compression_with_counter)
            {
                chunk_.set_nb_val_eq_zero(this->chunk_size() - 1);
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
template <typename Lambda>
void Proxy<T>::execute_simple_operation(Lambda operation)
{
    // Snapshot BEFORE any arming, to maintain nb_val_eq_zero across the write
    // for an ALREADY-ARMED chunk (0 -> non-zero must decrement it, else the
    // counter drifts up and manage_full_zero() compresses a non-zero chunk).
    // Mirrors execute_inplace_operation.
    // raw-read: write session -- the write/compress handshake keeps compressors off this chunk
    [[maybe_unused]] const bool was_first_value = values_[num_] == static_cast<T>(Chunk<T>::first_value);

    const bool is_val1 = Base::is_cst_val1();

    if (is_val1)
    {
        create_value();

        if constexpr (chunk_config::use_compression_with_counter)
        {
            chunk_.set_nb_val_eq_zero(this->chunk_size() - 1);
        }
    }

    operation();

    if constexpr (chunk_config::use_compression_with_counter)
    {
        // Armed above at chunk_size - 1, betting the element turns non-zero;
        // 0 += 0 leaves it at the first value and neither transition below
        // fires, so the count would stay one short for good. Settle the bet.
        // raw-read: serial counter mode (no fast block compression, no concurrent pass)
        if (is_val1 && values_[num_] == static_cast<T>(Chunk<T>::first_value))
        {
            chunk_.inc_nb_val_eq_zero();
            if (chunk_.is_full_zero(this->chunk_size()))
            {
                this->manage_full_zero();
            }
            return;
        }
        // Already-armed transitions (arming handled above).
        // raw-read: serial counter mode (no fast block compression, no concurrent pass)
        if (was_first_value && !is_val1 && values_[num_] != static_cast<T>(Chunk<T>::first_value))
        {
            chunk_.dec_nb_val_eq_zero();
        }
        // raw-read: serial counter mode (no fast block compression, no concurrent pass)
        if (values_[num_] == static_cast<T>(Chunk<T>::first_value) && !was_first_value)
        {
            chunk_.inc_nb_val_eq_zero();
            if (chunk_.is_full_zero(this->chunk_size()))
            {
                this->manage_full_zero();
            }
        }
    }
}

/// @brief Replay of a plain store (v1).
/// @pre Called from a flush drain only -- see "The v1 replay bodies" above.
template <typename Derived, typename T, bool Enabled>
template <typename Lambda>
void ProxyFastCompPolicy<Derived, T, Enabled>::execute_simple_operation_async(auto* ctx, Lambda operation)
{
    const bool is_val1 = ctx->chunk.is_cst_val1();

    if (is_val1)
    {
        ctx->chunk.create_value();
    }

    T* values = ctx->chunk.values_load();
    operation(values);
    internal::check_replay_store(ctx->chunk, values, "execute_simple_operation_async", ctx->num);
}

template <typename T>
template <typename Lambda>
void Proxy<T>::execute_inplace_operation(Lambda operation)
{
    // raw-read: write session -- the write/compress handshake keeps compressors off this chunk
    [[maybe_unused]] const bool was_first_value = values_[num_] == static_cast<T>(Chunk<T>::first_value);

    const bool         is_val = is_cst_val();
    [[maybe_unused]] T v_old  = T{};

    if constexpr (chunk_config::use_compression_with_counter)
    {
        // A compressed chunk the operation leaves at the first value (0 &= x,
        // 0 <<= k, 0 |= 0, ...) has nothing to write: do not materialise it. The
        // operation (a pure function of the element) is tried on the constant.
        if (is_val)
        {
            auto probe = static_cast<T>(Chunk<T>::first_value);
            operation(probe);
            if (probe == static_cast<T>(Chunk<T>::first_value))
            {
                return;
            }
        }
    }

    if constexpr (chunk_config::use_fast_block_compression)
    {
        v_old = read_values();
    }

    if (is_val)
    {
        // Through Proxy::create_value(), NOT chunk.create_value() directly: the
        // wrapper pins the ambient geometry to THIS array's chunk_size() first.
        // A direct call pops the reuse list of whichever array wrote last, a
        // buffer of the wrong size when two geometries are alive.
        create_value();

        if constexpr (chunk_config::use_compression_with_counter)
        {
            chunk_.set_nb_val_eq_zero(this->chunk_size() - 1);
        }
    }

    // raw-read: write session -- the write/compress handshake keeps compressors off this chunk
    operation(Base::values_[num_]);

    if constexpr (chunk_config::use_compression_with_counter)
    {
        // Unreachable after the probe above; kept as execute_simple_operation's twin.
        // raw-read: serial counter mode (no fast block compression, no concurrent pass)
        if (is_val && values_[num_] == static_cast<T>(Chunk<T>::first_value))
        {
            chunk_.inc_nb_val_eq_zero();
            if (chunk_.is_full_zero(this->chunk_size()))
            {
                this->manage_full_zero();
            }
            return;
        }
        // raw-read: serial counter mode (no fast block compression, no concurrent pass)
        if (was_first_value && !is_val && values_[num_] != Chunk<T>::first_value)
        {
            chunk_.dec_nb_val_eq_zero();
        }
        // raw-read: serial counter mode (no fast block compression, no concurrent pass)
        if (values_[num_] == static_cast<T>(Chunk<T>::first_value) && !was_first_value)
        {
            chunk_.inc_nb_val_eq_zero();
            if (chunk_.is_full_zero(this->chunk_size()))
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

/// @brief Replay of an in-place operation (++, --; v1).
/// @pre Called from a flush drain only -- see "The v1 replay bodies" above.
template <typename Derived, typename T, bool Enabled>
template <typename Lambda>
void ProxyFastCompPolicy<Derived, T, Enabled>::execute_inplace_operation_async(auto* ctx, Lambda operation)
{
    auto&     chunk_loc = ctx->chunk;
    const int num       = ctx->num;

    const InT<T> v_old = Base::read_values_async(ctx);

    const Proxy proxy  = Base::template make_proxy_with_info<Derived>(ctx);
    const bool  is_val = chunk_loc.is_cst_val1();

    if (is_val)
    {
        // Through the proxy: pins the ambient geometry (see execute_inplace_operation).
        proxy.create_value();
    }

    T* values = chunk_loc.values_load();

    // raw-read: deferred replay, after materialisation -- flushes exclude the stealers
    operation(values[num]);
    internal::check_replay_store(chunk_loc, values, "execute_inplace_operation_async", num);

    // InT: nothing writes it afterwards.
    // raw-read: deferred replay, after materialisation -- flushes exclude the stealers
    const InT<T> v_new = values[num];

    mark_for_compression_if_needed_with_value_async(proxy, v_old, v_new);
}

template <typename T>
template <typename U, typename BinaryOp>
void Proxy<T>::apply_compound_operation(InT<U> v, BinaryOp op)
{
    this->check_not_stale();
    // Cross-rank RMW materialisation: pull any foreign deferred write first.
    chunk_.reader_sync(num_);

    if constexpr (is_not_bool)
    {
        if constexpr (chunk_config::use_compression_with_counter)
        {
            // raw-read: serial counter mode (no fast block compression, no concurrent pass)
            if (values_[num_] != zero)
            {
                if constexpr (build_config::do_check)
                {
                    global_check_write_nan(v, num_);
                }

                execute_arithmetic_compound_operation(
                    op, static_cast<T>(v),
                    [this, &v](auto op_func) // v: this call's parameter, alive for the call
                    {
                        using LocalCalcType = std::common_type_t<T, BaseT<U>>;

                        const InT<T> result = static_cast<T>(
                            op_func(static_cast<LocalCalcType>(Base::load_value()), static_cast<LocalCalcType>(v)));

                        Base::store_value(result);
                    });

                // Test the result, not the operand: `x /= 0.0` is +-inf, and
                // counting it as a zero would recompress a chunk holding infinities.
                // raw-read: serial counter mode (no fast block compression, no concurrent pass)
                if (values_[num_] == zero && !is_cst_val())
                {
                    chunk_.inc_nb_val_eq_zero();
                    if (chunk_.is_full_zero(this->chunk_size()))
                    {
                        this->manage_full_zero();
                    }
                }
            }
        }
        else
        {
            // const T, not InT<T>: apply_atomic_op() stores to values[num]
            // and still reads v_old (in_t.h). Read under a ReadSection (see
            // ProxyBase::read_values_outside_session).
            const T v_old = Base::read_values_outside_session();
            // A zero v_old is not enough to skip: a queued deferred op may change it.
            if (v_old != zero || CompressionManagerType::have_deferred_queue())
            {
                Base::apply_atomic_op(op, static_cast<T>(v), v_old);

                // Do NOT read values[num] back here: apply_atomic_op may have
                // DEFERRED the op, and a foreign thread draining this chunk
                // (reader_sync -> flush_chunk_all_ranks) can be replaying it now.
            }
        }
    }
    else
    {
        CHUNK_LOG(error, general, "operator not implemented for bool: the element is left unchanged");
    }
}

template <typename Derived, typename T, bool Enabled>
void ProxyFastCompPolicy<Derived, T, Enabled>::trigger_compression_if_needed(int thr_rank) noexcept
{
    const auto& vec_tuple = Manager::chunk_to_compress_potentially_thr[thr_rank];

    const int sz = static_cast<int>(vec_tuple.size());

    const int szm = Manager::get_size_max_number_of_chunk_to_check_compress();

    if (Base::should_trigger_compression(sz, szm))
    {
        CompressionManagerType::mark_list_dangerous_for_thread(thr_rank);

        if constexpr (chunk_config::verbose)
        {
            CHUNK_LOG_ALWAYS(compress, "candidate list became dangerous: size={}", sz);
        }
    }
}
template <typename Derived, typename T, bool Enabled>
void ProxyFastCompPolicy<Derived, T, Enabled>::perform_compression() const noexcept
{
    // Runs inside a region, next to pure readers: the buffers this pass
    // releases wait for one grace period (read_reclaim.h).
    const read_reclaim::PassScope pass_scope;
    const int                     thr_rank = get_thr_rank_fast();

    const int szm = Manager::get_size_max_number_of_chunk_to_check_compress();

    const auto& vec_tuple = Manager::chunk_to_compress_potentially_thr[thr_rank];

    const int sz_before = static_cast<int>(vec_tuple.size());

    if (Base::should_trigger_compression(sz_before, szm))
    {
        // Drain this rank's deferred ops first: they must be applied before
        // their chunks are compressed. Do not RETURN when the rank has any
        // before the flag below is cleared: the rank would stay dangerous and
        // every write would take run_slow_path(), which skips the backlog drain.
        static_cast<void>(CompressionManagerType::flush_deferred(thr_rank));

        std::atomic_thread_fence(std::memory_order_seq_cst);

        Chunk<T>::make_block_compression(thr_rank, this->self());
    }
    // Looked at, whether or not there was work: clear the flag. See
    // mark_list_dangerous_for_thread.
    CompressionManagerType::mark_list_safe_for_thread(thr_rank);
}

} // namespace shrink_array