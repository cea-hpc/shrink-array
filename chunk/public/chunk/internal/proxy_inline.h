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
#include <iostream>    // std::cerr (the mismatch diagnostic)
#include <type_traits> // std::common_type_t
#include <utility>     // std::move

#include "chunk/internal/in_t.h"  // in_t<T>
#include "chunk/internal/proxy.h" // the declarations these bodies belong to
#include "chunk/config.h"    // chunk_config::use_compression_with_counter, chunk_config::use_fast_block_compression
#include "chunk/PMP_CHUNK.h" // get_thr_rank_fast
#include "chunk/internal/arithmetic_context.h"           // ArithmeticAffectationProxyContext
#include "chunk/internal/concept_is_clean_assignement.h" // base_t
#include "chunk/internal/read_reclaim.h"    // read_reclaim::PassScope (perform_compression / compress in a region)
#include "chunk/internal/invariant_check.h" // CHUNK_INVARIANT_FAILED

namespace shrink_array
{

template <typename T>
const Proxy<T>& Proxy<T>::operator=(const Proxy<T>& from) noexcept
{
    if (&from != this)
    {
        // Read through operator T(): the dirty-bit gated read (read_pure()
        // when a deferred op may be queued on from's chunk). The read is
        // complete before the write starts, so no reader/writer nesting even
        // when both proxies sit in the same chunk.
        const in_t<T> v = static_cast<T>(from);

        // Write through the engine: Dekker handshake / deferred path,
        // WAW-erase, pending_any()-guarded zero early-out, compression
        // marking and nb_val_eq_zero maintenance. An inline store would
        // bypass all of it (a[i] = b[j] could land in a CAS-swapped buffer,
        // be overwritten by an older deferred replay, or drift the zero
        // counter into a spurious full-zero compression).
        assign_impl<T>(v);
    }

    return *this;
}

template <typename T>
T* Proxy<T>::perform_core_assignment(T* vals, in_t<T> v) const noexcept
{
    if constexpr (chunk_config::use_compression_with_counter)
    {
        if (v == static_cast<T>(Chunk<T>::first_value))
        {
            if (is_not_cst_val())
            {
                // raw-read: serial counter mode (no fast block compression, no concurrent pass)
                if (values[num] != Chunk<T>::first_value)
                {
                    chunk.inc_nb_val_eq_zero();
                }

                if (chunk.is_full_zero(this->chunk_size()))
                {
                    this->manage_full_zero();
                    return vals;
                }
            }
        }
        // raw-read: serial counter mode (no fast block compression, no concurrent pass)
        else if (is_not_cst_val() && values[num] == Chunk<T>::first_value)
        {
            chunk.dec_nb_val_eq_zero();
        }
    }

    if constexpr (chunk_config::stats)
    {
        increm_chunk_write<ChunkType>(v);
    }

    const bool need_alloc = this->is_cst_val1(vals);

    if (need_alloc)
    {
        // Zero-overhead state helper: members exist only if Enabled is true
        class
        {
          public:
            // Captured before allocation to determine correct counter initialization
            void capture(const Proxy* proxy) noexcept
            {
                if constexpr (chunk_config::use_compression_with_counter)
                {
                    was_cst = proxy->is_cst_val();
                }
            }

            // Applied after allocation to sync counters with the new heap block
            void apply(const Proxy* proxy, T val)
            {
                if constexpr (chunk_config::use_compression_with_counter)
                {
                    if (was_cst)
                    {
                        proxy->chunk.set_nb_val_eq_zero(proxy->chunk_size() - 1);
                    }
                    else if (val == Chunk<T>::first_value)
                    {
                        proxy->chunk.inc_nb_val_eq_zero();
                    }
                }
            }

            [[maybe_unused]] bool was_cst = false;
        } state;

        state.capture(this); // Becomes a No-Op if counters are disabled

        create_value(); // Perform heap allocation

        state.apply(this, v); // Becomes a No-Op if counters are disabled

        // The one reload: create_value() does not return the live buffer, and
        // a concurrent writer may have won its CAS (ours then went back to the
        // pool). Materialised now, so stable until the session ends.
        vals = chunk.values_load();
    }

    Base::store_value_at(vals, v);

    if constexpr (build_config::do_check)
    {
        global_check_write_nan(v, num);
    }
    return vals;
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
    }
}

template <typename Derived, typename T, bool Enabled>
void ProxyFastCompPolicy<Derived, T, Enabled>::mark_for_compression_core(in_t<T>  old_value,
                                                                         in_t<T>  new_value,
                                                                         in_t<T>  expected_in_memory,
                                                                         const T* vals) const
{
    // Same test; the race check reads the element again, not the pointer.
    if (should_mark_for_compression(old_value, new_value) || (read_values_at(vals) != expected_in_memory)) [[unlikely]]
    {
        Base::add_to_compression_list();
    }
}

template <typename T>
template <typename OP, typename LBD, typename U>
void Proxy<T>::execute_arithmetic_compound_operation(OP op, [[maybe_unused]] const U& v, LBD lbd)
{
    // Determine the high-precision calculation type (STL parity)
    using CalcType = std::common_type_t<T, base_t<U>>;

    // No compression pass from here. In fast-block builds this body runs
    // inside the rank's writer window (apply_atomic_op -> write_with_optional_
    // lock_info): start()'s "no writer" wait would see this rank's own flag and
    // run to its budget, and perform_compression() would clear the dangerous
    // flag the post-end_write driver acts on. The engine drives the pass after
    // end_write(), and has already armed set_info_on_chunk().

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
        // const T, not in_t<T>: a snapshot read before the store below and
        // used after it (in_t.h, corollary for local variables).
        // raw-read: write session -- the write/compress handshake keeps compressors off this chunk
        const T v_old = values[num];

        // 1. Perform math in high precision using CalcType
        const CalcType v_new_precise = op(static_cast<CalcType>(v_old), static_cast<CalcType>(v));

        // 2. Truncate back to storage type T
        const auto v_new = static_cast<T>(v_new_precise);

        // raw-read: write session -- the write/compress handshake keeps compressors off this chunk
        values[num] = v_new;

        if constexpr (chunk_config::use_fast_block_compression)
        {
            // Trigger compression marking logic if the result is back to zero
            this->mark_for_compression_if_needed_with_value(v_old, v_new);
        }
    }
}

// ─── The v1 replay bodies (`*_async`) and the hypothesis they rest on ──────
//
// The `_async` bodies below (execute_arithmetic_compound_operation_async,
// the deferred body of assign_impl, execute_simple_operation_async,
// execute_inplace_operation_async) are called ONLY from a flush drain:
// CompressionManager::flush_deferred() / flush_chunk_all_ranks(), which
// wait_for_availability() reaches too. Both drains announce the flush
// (Policy::announce_flush()) and then wait out the in-flight hard
// compressors. The invariant these bodies rest on: NO COMPRESSOR CAN CHANGE
// `values` DURING A REPLAY. So they load the values pointer once and store
// through it, with no retry. Two kinds of compressor keep it:
//
//   - the block passes (make_block_compression) go through
//     CompressionManager::start(), which refuses while a flush is announced;
//   - two v1 compressors bypass start() and call compress() directly:
//     Chunk::compress_sweep_rank (the phase-end sweep, its own parallel
//     region) and RechunkHooks::rechunk_drop_candidates (right after the
//     re-chunk's own flush, from serial code). They are safe because they run
//     where no drain can -- outside the writers' parallel region, after the
//     flush -- not because of start(). check_no_flush_outside_start()
//     (block_pass_precheck.h) trips in check builds if one is ever reached
//     while a flush is in progress.
//
// (The re-chunk's fold, a direct compress() for Chunk2 / Chunk2A, has no v1
// branch: array_chunk_rechunk_inline.h.)
//
// Proxy2A's twins (proxy2A_inline.h) carry a pre-/post-store retry loop
// instead; v1 does not need one as long as the invariant holds. It breaks if
// a new compressor bypasses start() from where a drain can run, or an
// `_async` body is called outside a drain. A store would then land in a
// released buffer and the write would be lost in silence -- which is what
// check_replay_store() below turns into an CHUNK_INVARIANT_FAILED in check builds
// (build_config::any_check; nothing in release).

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
template <typename OP, typename LAMBDA, typename U>
void ProxyFastCompPolicy<Derived, T, Enabled>::execute_arithmetic_compound_operation_async(auto* ctx,
                                                                                           OP    op,
                                                                                           [[maybe_unused]] const U& v,
                                                                                           LAMBDA lbd) const
{
    // Determine the high-precision calculation type for STL parity
    using CalcType = std::common_type_t<T, base_t<U>>;

    const bool was_not_written = !ctx->chunk.written();

    if (was_not_written)
    {
        // Decompress: the chunk was currently in a constant state
        ctx->chunk.create_value();
        [[maybe_unused]] const T* const materialised = ctx->chunk.values_load();

        // Execute the write logic using the newly allocated memory pointer
        lbd(op);

        internal::check_replay_store(ctx->chunk, materialised, "execute_arithmetic_compound_operation_async", ctx->num);
    }
    else
    {
        // Rebuild a temporary proxy to access maintenance metadata
        const Proxy proxy = Base::template make_proxy_with_info<Derived>(ctx);

        auto&     chunk_loc = ctx->chunk;
        const int num       = ctx->num;

        // Reload the current values pointer (with acquire semantics)
        T* values_ptr = chunk_loc.values_load();

        const T old_value = values_ptr[num];

        // 1. Perform math in high precision to avoid rounding drift
        const CalcType res = op(static_cast<CalcType>(old_value), static_cast<CalcType>(v));

        // 2. Truncate result back to the array's storage type
        const auto final_val = static_cast<T>(res);

        // 3. Store the value in physical memory
        values_ptr[num] = final_val;
        internal::check_replay_store(chunk_loc, values_ptr, "execute_arithmetic_compound_operation_async", num);

        // 4. Trigger the compression check logic (e.g. if the result is zero)
        mark_for_compression_if_needed_with_value_async(proxy, old_value, final_val);
    }
}

template <typename T>
template <typename U>
T Proxy<T>::assign_impl(in_t<U> v) noexcept
{
    this->check_not_stale();
    if (is_cst_val_and_v_equal_zero(v))
    {
        return v;
    }

    // No `need_alloc = is_cst_val()` hoisted here: it is an acquire load,
    // which the compiler cannot drop, paid by every write for a test only
    // the `v == zero` branch makes.
    if constexpr (chunk_config::use_compression_with_counter)
    {
        if (v == zero)
        {
            if (is_cst_val())
            {
                return zero;
            }

            // raw-read: serial counter mode (no fast block compression, no concurrent pass)
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
        // raw-read: serial counter mode (no fast block compression, no concurrent pass)
        else if (!is_cst_val() && values[num] == zero)
        {
            chunk.dec_nb_val_eq_zero();
        }
    }
    // Without the counter, the early-out for `= 0` on a compressed chunk is
    // is_cst_val_and_v_equal_zero() above, and only it: a second copy of the
    // same test stood here, one more load of `values` for every `= 0`. Same
    // rule for both: not over a pending deferred op (deferred_path_validation,
    // chunk #7 offset 0: the chunk's first write deferred, then `= 0` skipped,
    // then the replay).

    // By reference: v is this call's parameter, alive for the synchronous
    // call (see Proxy2A::assign_impl); the deferred path copies it.
    [[maybe_unused]] const auto lbd_equal_core = [this, &v]
    {
        // ONE load of the values pointer for the whole body, taken here, inside
        // the session (or a flush replay). From here to end_write no compressor
        // can change it (the Dekker handshake, or the flush's exclusion of the
        // stealers); a concurrent WRITER can, but only once and one way --
        // constant -> materialised -- so a materialised capture is stable and a
        // constant one is reloaded once, by perform_core_assignment(), after
        // create_value() -- one acquire instead of five (old value, is_cst_val,
        // store, the race check of the marking).
        T* vals = chunk.values_load();

        // The value there BEFORE the write. Two distinct traps sit here:
        //  - it must be read before perform_core_assignment(), which materialises
        //    the chunk: a freshly materialised buffer is all-zero, so reading
        //    afterwards always yields 0;
        //  - it must be read through a pointer loaded NOW, not one captured when
        //    the proxy was built: on a chunk compressed since, that one still
        //    designates the shared constant buffer, and the read returned the
        //    constant (zero) whatever the real content.
        // Either way old_value came back as 0, the non-zero -> zero transition was
        // never seen, and no chunk was ever queued for compression under OpenMP
        // (CHUNK_STATS: 70M calls from this path, 0 marks, compress flat at 0).
        // On a still-constant chunk the capture yields the constant: the right
        // old value. read_value() under CHUNK_STATS, for its read counters.
        const in_t<T> old_value = [&]
        {
            if constexpr (chunk_config::stats)
            {
                return chunk.read_value(num);
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

            auto* ctx = pool_new_ctx<CtxType>(*this, v, ptr_on_array);

            auto lbd_equal_asynch = [ctx] // the operand lives in the context: no second copy in the closure
            {
                auto&     chunk = ctx->chunk;
                const int num   = ctx->num;

                const auto proxy = Base::template make_proxy_with_info<Proxy>(ctx);

                // Replay body: see "The v1 replay bodies" above. One load of the
                // pointer, reloaded once after a materialisation -- as in
                // lbd_equal_core.
                T* values = chunk.values_load();

                const bool is_cst = chunk.is_cst_val1(values);

                // Capture the previous value BEFORE materialising. create_value()
                // installs a fresh all-zero buffer, so reading values[num] after it
                // always returned 0 and the non-zero -> zero transition below could
                // never fire: no chunk was ever queued for compression on this path
                // (CHUNK_STATS: 81M entries in mark_for_compression_core, 0 marks,
                // compress stuck at 0 for the whole run). On a still-compressed
                // chunk values[] holds the constant, which is the correct old value.
                const in_t<T> v_old = [&]
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
    // raw-read: write session -- the write/compress handshake keeps compressors off this chunk
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
        // Armed above at chunk_size - 1, betting that the element turns
        // non-zero -- but 0 += 0, 0 -= 0 leave it at the first value, and
        // neither transition below fires (was_first_value is true): the count
        // stayed one short for good and the chunk could never be compressed
        // again. Settle the bet: the chunk is all first values, compress it.
        // raw-read: serial counter mode (no fast block compression, no concurrent pass)
        if (is_val1 && values[num] == static_cast<T>(Chunk<T>::first_value))
        {
            chunk.inc_nb_val_eq_zero();
            if (chunk.is_full_zero(this->chunk_size()))
            {
                this->manage_full_zero();
            }
            return;
        }
        // Already-armed transitions (arming handled above).
        // raw-read: serial counter mode (no fast block compression, no concurrent pass)
        if (was_first_value && !is_val1 && values[num] != static_cast<T>(Chunk<T>::first_value))
        {
            chunk.dec_nb_val_eq_zero();
        }
        // raw-read: serial counter mode (no fast block compression, no concurrent pass)
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

/// @brief Replay of a plain store (v1).
/// @pre Called from a flush drain only -- see "The v1 replay bodies" above.
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
    internal::check_replay_store(ctx->chunk, values, "execute_simple_operation_async", ctx->num);
}

template <typename T>
template <typename LAMBDA>
void Proxy<T>::execute_inplace_operation(LAMBDA operation)
{
    // raw-read: write session -- the write/compress handshake keeps compressors off this chunk
    [[maybe_unused]] const bool was_first_value = values[num] == static_cast<T>(Chunk<T>::first_value);

    const bool         is_val = is_cst_val();
    [[maybe_unused]] T v_old  = T{};

    if constexpr (chunk_config::use_compression_with_counter)
    {
        // A compressed chunk the operation leaves at the first value -- 0 &= x,
        // 0 <<= k, 0 >>= k, 0 |= 0, 0 ^= 0 -- has nothing to write. Do not
        // materialise it: the operation (a pure function of the element: the
        // bitwise operators, ++, --) is tried on the constant instead, as
        // apply_compound_operation() skips *= /= %= on a zero element.
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

    // raw-read: write session -- the write/compress handshake keeps compressors off this chunk
    operation(Base::values[num]);

    if constexpr (chunk_config::use_compression_with_counter)
    {
        // Unreachable after the probe above, kept as the simple engine's twin:
        // armed at chunk_size - 1 for a result that is the first value after
        // all, the count would stay one short and the chunk never compress.
        // raw-read: serial counter mode (no fast block compression, no concurrent pass)
        if (is_val && values[num] == static_cast<T>(Chunk<T>::first_value))
        {
            chunk.inc_nb_val_eq_zero();
            if (chunk.is_full_zero(this->chunk_size()))
            {
                this->manage_full_zero();
            }
            return;
        }
        // raw-read: serial counter mode (no fast block compression, no concurrent pass)
        if (was_first_value && !is_val && values[num] != Chunk<T>::first_value)
        {
            chunk.dec_nb_val_eq_zero();
        }
        // raw-read: serial counter mode (no fast block compression, no concurrent pass)
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

/// @brief Replay of an in-place operation (++, --; v1).
/// @pre Called from a flush drain only -- see "The v1 replay bodies" above.
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
    // raw-read: deferred replay, after materialisation -- flushes exclude the stealers
    operation(values[num]);
    internal::check_replay_store(chunk_loc, values, "execute_inplace_operation_async", num);

    // Re-read the updated value: nothing writes it afterwards, in_t (a
    // reference for a struct) rather than a copy.
    // raw-read: deferred replay, after materialisation -- flushes exclude the stealers
    const in_t<T> v_new = values[num];

    mark_for_compression_if_needed_with_value_async(proxy, v_old, v_new);
}

template <typename T>
template <typename U, typename BinaryOp>
void Proxy<T>::apply_compound_operation(in_t<U> v, BinaryOp op)
{
    this->check_not_stale();
    // Cross-rank RMW materialisation: pull any foreign deferred write first.
    chunk.reader_sync(num);

    if constexpr (is_not_bool)
    {
        if constexpr (chunk_config::use_compression_with_counter)
        {
            // raw-read: serial counter mode (no fast block compression, no concurrent pass)
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

                // raw-read: serial counter mode (no fast block compression, no concurrent pass)
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
            // const T, not in_t<T>: apply_atomic_op() stores to values[num]
            // and still reads v_old (in_t.h, corollary for local variables).
            // Outside the write session: under a ReadSection (see
            // ProxyBase::read_values_outside_session).
            const T v_old = Base::read_values_outside_session();
            // we can't test values[num] if deferred mode is activate : because values[num] could be affect after
            if (v_old != zero || CompressionManagerType::have_deferred_queue())
            {
                Base::apply_atomic_op(op, static_cast<T>(v), v_old);

                // Do NOT read values[num] back here: apply_atomic_op may have
                // DEFERRED the op, and a foreign thread draining this chunk
                // (reader_sync -> flush_chunk_all_ranks) can be replaying it
                // right now -- TSan: write in
                // execute_arithmetic_compound_operation_async vs this read
                // (shrink_bench_chunk_256 --mode check, tab4[i] *= ...). Nothing is
                // returned any more, so there is no reason to read it back.
            }
        }
    }
    else
    {
        std::cerr << "operator not implemented for bool\n";
    }
}

template <typename Derived, typename T, bool Enabled>
void ProxyFastCompPolicy<Derived, T, Enabled>::trigger_compression_if_needed(int thr_rank) noexcept
{
    const auto& vec_tuple = manager_chunk::chunk_to_compress_potentially_thr[thr_rank];

    const int sz = static_cast<int>(vec_tuple.size());

    const int szm = manager_chunk::get_size_max_number_of_chunk_to_check_compress();

    // ✅ Track state transitions

    if (Base::should_trigger_compression(sz, szm))
    {
        CompressionManagerType::mark_list_dangerous_for_thread(thr_rank);

        if constexpr (chunk_config::verbose)
        {
            std::cerr << "⚠️ List became dangerous: size=" << sz << "\n";
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

    const int szm = manager_chunk::get_size_max_number_of_chunk_to_check_compress();

    const auto& vec_tuple = manager_chunk::chunk_to_compress_potentially_thr[thr_rank];

    const int sz_before = static_cast<int>(vec_tuple.size());

    if (Base::should_trigger_compression(sz_before, szm))
    {
        // Drain this rank's deferred ops first -- they must be applied before
        // their chunks are compressed -- as Proxy2 / Proxy2A do. Do not RETURN
        // when the rank has any (have_deferred_queue()) before the flag below is
        // cleared: the rank would stay dangerous, every write would take
        // run_slow_path(), which skips the backlog drain, so the ops would never
        // drain -- the slow path feeding itself until the between-phase drain.
        static_cast<void>(CompressionManagerType::flush_deferred(thr_rank));

        std::atomic_thread_fence(std::memory_order_seq_cst);

        Chunk<T>::make_block_compression(thr_rank, this->self());
    }
    // Looked at, whether or not there was work: clear the flag the trigger
    // raised (it only ever came down through finish(), once per completed
    // compression, against one raise per write above the threshold). See
    // mark_list_dangerous_for_thread and the Proxy2A counterpart.
    CompressionManagerType::mark_list_safe_for_thread(thr_rank);
}

} // namespace shrink_array