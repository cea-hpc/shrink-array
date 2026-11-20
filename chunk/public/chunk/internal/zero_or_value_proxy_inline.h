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

#include <cstddef>     // std::size_t
#include <type_traits> // std::remove_cvref_t
#include <memory>      // std::addressof (operator=: operator& is deleted)
#include <tuple>       // std::tuple (the key of the check builds' registry memo)

#include "chunk/internal/in_t.h"                // InT<T>
#include "chunk/internal/bad_ptr_check.h"       // CHUNK_CHECK_VALUES_PTR (check builds)
#include "chunk/internal/cold_path_lock.h"      // verify_once_per_generation (check builds)
#include "chunk/internal/zero_or_value_proxy.h" // the declarations these bodies belong to
#include "chunk/config.h"             // chunk_config::use_compression_with_counter, build_config::chunk_dynamic
#include "chunk/internal/pmp_chunk.h" // get_thr_rank_fast
#include "chunk/internal/arithmetic_context.h"          // ArithmeticAffectationContext
#include "chunk/internal/chunk_size_policy.h"           // DynChunkSizeGuard, set_thread_chunk_size
#include "chunk/internal/concept_is_clean_assignment.h" // BaseT
#include "chunk/internal/invariant_check.h"             // CHUNK_INVARIANT_FAILED
#include "chunk/internal/read_reclaim.h" // read_reclaim::PassScope (perform_compression / compress in a region)
#include "chunk/internal/test_hooks.h"   // TEST KNOB: test_hooks::maybe_read_delay
#include "chunk/log.h"                   // CHUNK_LOG_ALWAYS (verbose)
#include "chunk/internal/same_value.h"   // same_value

namespace shrink_array
{

// No acquire fence ends set_info_second_value_idx() or the two
// set_info_on_chunk(): they would follow plain (non-atomic) reads of the
// array_registry registry and pair with no release. Do not add one: on
// aarch64 it was the hottest instruction of assign_impl (several % of a run).

// The armed mirror, checked against the registry under any_check; nullptr when
// the proxy is not armed. Callers read the fields THROUGH this pointer: see the
// note on set_info_on_chunk() below for why not through a returned copy.
template <typename T, typename Derived>
const zero_or_value::ProxySpec<T, Derived>::ArrayInfo* zero_or_value::ProxySpec<T, Derived>::armed_array_info() const
{
    if (array_info_src_ != nullptr) [[likely]]
    {
        if constexpr (build_config::any_check)
        {
            // idx and array_id only: idx maps one-to-one to the value, and a
            // value compare would raise a false alarm on a NaN second value.
            // Once per (array, idx, array_id) and registry generation, not per
            // write: see verify_once_per_generation().
            const auto ptr      = get_ptr_on_array();
            const int  idx      = array_info_src_->idx;
            const int  array_id = array_info_src_->array_id;
            internal::verify_once_per_generation(
                ptr, std::tuple{ptr, idx, array_id},
                [ptr, idx, array_id]
                {
                    const ArrayInfo reg = Derived::get_info_second_value(ptr);
                    if (reg.idx != idx || reg.array_id != array_id)
                    {
                        CHUNK_INVARIANT_FAILED(
                            "zero_or_value::ProxySpec::load_array_info: armed (index_value {}, array_id {}) "
                            "differs from the registry's (index_value {}, array_id {}) for array {} -- a "
                            "registry write without sync_registry_mirror()?",
                            idx, array_id, reg.idx, reg.array_id, ptr);
                    }
                });
        }
    }
    return array_info_src_;
}

template <typename T, typename Derived>
zero_or_value::ProxySpec<T, Derived>::ArrayInfo zero_or_value::ProxySpec<T, Derived>::load_array_info() const
{
    if (const ArrayInfo* src = armed_array_info()) [[likely]]
    {
        return *src;
    }
    return Derived::get_info_second_value(get_ptr_on_array());
}

template <typename T, typename Derived>
    requires is_bool_like_v<T>
const zero_or_value::ProxySpec<T, Derived>::ArrayInfo* zero_or_value::ProxySpec<T, Derived>::armed_array_info() const
{
    if (array_info_src_ != nullptr) [[likely]]
    {
        if constexpr (build_config::any_check)
        {
            // idx and array_id only: idx maps one-to-one to the value, and a
            // value compare would raise a false alarm on a NaN second value.
            // Once per (array, idx, array_id) and registry generation, not per
            // write: see verify_once_per_generation().
            const auto ptr      = get_ptr_on_array();
            const int  idx      = array_info_src_->idx;
            const int  array_id = array_info_src_->array_id;
            internal::verify_once_per_generation(
                ptr, std::tuple{ptr, idx, array_id},
                [ptr, idx, array_id]
                {
                    const ArrayInfo reg = Derived::get_info_second_value(ptr);
                    if (reg.idx != idx || reg.array_id != array_id)
                    {
                        CHUNK_INVARIANT_FAILED(
                            "zero_or_value::ProxySpec::load_array_info: armed (index_value {}, array_id {}) "
                            "differs from the registry's (index_value {}, array_id {}) for array {} -- a "
                            "registry write without sync_registry_mirror()?",
                            idx, array_id, reg.idx, reg.array_id, ptr);
                    }
                });
        }
    }
    return array_info_src_;
}

template <typename T, typename Derived>
    requires is_bool_like_v<T>
zero_or_value::ProxySpec<T, Derived>::ArrayInfo zero_or_value::ProxySpec<T, Derived>::load_array_info() const
{
    if (const ArrayInfo* src = armed_array_info()) [[likely]]
    {
        return *src;
    }
    return Derived::get_info_second_value(get_ptr_on_array());
}

// Every write runs one of these. They take the fields from the armed mirror --
// memory written long before -- never from a copy just stored. Not
//     array_info = load_array_info(); second_value = array_info.second_value; ...
// which returns the 16-byte record through a stack temporary (a q-register
// store) and reads it straight back with 8- and 4-byte loads: a store the core
// cannot forward to narrower loads, so each write waits for it to reach the
// cache -- a large share of assign_impl's time on AArch64. Unarmed proxies
// (replay, *_from_key) keep the registry path: rare, and a registry lookup
// costs more than the stall anyway.
// Forced inline, for the same kind of stall on the other side: called out of
// line (the engine's lbd_info, every write), it stores into the proxy -- on the
// caller's stack -- with narrow stores that the caller's 16-byte reloads of the
// proxy span (x86, 1 thread: rmw 13.2 -> 6.0 ns, more than read + write before).
template <typename T, typename Derived>
CHUNK_FORCE_INLINE void zero_or_value::ProxySpec<T, Derived>::set_info_on_chunk()
{
    if (const ArrayInfo* src = armed_array_info()) [[likely]]
    {
        array_info_   = *src;
        second_value_ = src->second_value;
        idx_          = src->idx;
        derived().set_array_id(src->array_id); // what set_array_id() would read back from array_info
    }
    else
    {
        array_info_   = Derived::get_info_second_value(get_ptr_on_array());
        second_value_ = array_info_.second_value;
        idx_          = array_info_.idx;
        derived().set_array_id();
    }

    if constexpr (chunk_config::use_fast_block_compression)
    {
        derived().bind_candidate_flag();
    }
}

template <typename T, typename Derived>
void zero_or_value::ProxySpec<T, Derived>::set_info_second_value_idx()
{
    if (const ArrayInfo* src = armed_array_info()) [[likely]]
    {
        array_info_   = *src;
        second_value_ = src->second_value;
        idx_          = src->idx;
        return;
    }
    array_info_   = Derived::get_info_second_value(get_ptr_on_array());
    second_value_ = array_info_.second_value;
    idx_          = array_info_.idx;
}

template <typename T, typename Derived>
void zero_or_value::ProxySpec<T, Derived>::set_info_second_value_idx_from_key(ArrayKey<Chunk<T>> key)
{
    array_info_   = Derived::get_info_second_value(key);
    second_value_ = array_info_.second_value;
    idx_          = array_info_.idx;
}

// Hidden from Doxygen: the zero_or_value::Proxy alias (BaseSpec) and the bool specialization
// make this out-of-line definition ambiguous to it. Documented in zero_or_value_proxy.h.
/// @cond
template <typename T, typename Derived>
void zero_or_value::ProxySpec<T, Derived>::set_second_value()
{
    if (const ArrayInfo* src = armed_array_info()) [[likely]]
    {
        array_info_   = *src;
        second_value_ = src->second_value;
        return;
    }
    array_info_   = Derived::get_info_second_value(get_ptr_on_array());
    second_value_ = array_info_.second_value;
}
/// @endcond

template <typename T, typename Derived>
    requires is_bool_like_v<T>
CHUNK_FORCE_INLINE void zero_or_value::ProxySpec<T, Derived>::set_info_on_chunk()
{
    if (const ArrayInfo* src = armed_array_info()) [[likely]]
    {
        array_info_ = *src;
        derived().set_array_id(src->array_id); // not read back from array_info (see above)
    }
    else
    {
        array_info_ = Derived::get_info_second_value(get_ptr_on_array());
        derived().set_array_id();
    }

    if constexpr (chunk_config::use_fast_block_compression)
    {
        derived().bind_candidate_flag();
    }
}

// ── Methods only present when fast block compression is active ────────────
// Members of the Enabled=true policy only: zero_or_value::Proxy instantiates
// zero_or_value::ProxyFastCompPolicy<..., chunk_config::use_fast_block_compression>, and the
// <..., false> specialisation declares none of them, so nothing here is ever
// looked at by a serial build. No preprocessor guard needed.

template <typename Derived, typename T, typename ChunkT, bool Enabled>
void zero_or_value::ProxyFastCompPolicy<Derived, T, ChunkT, Enabled>::trigger_compression_if_needed(
    int thr_rank) noexcept
{
    if constexpr (is_not_bool)
    {
        derived().second_value_ = derived().make_second_value();
    }

    // The threshold is tested where a list grows -- the push, which only the
    // owning thread makes (note_candidate_pushed) -- not here: that cost a
    // std::map find by second value, a shared_ptr and a size load on EVERY
    // write (~2 ns of a ~14 ns write, x86; a chain of dependent loads). The
    // flag is turned into the rank's dangerous flag here, after the write, so
    // the pass still runs from the rank's next write.
    if (!list_over_threshold_) [[likely]]
    {
        return;
    }
    list_over_threshold_ = false;
    CompressionManagerType::mark_list_dangerous_for_thread(thr_rank);

    if constexpr (chunk_config::verbose)
    {
        CHUNK_LOG_ALWAYS(compress, "candidate list became dangerous (rank {})", thr_rank);
    }
}

template <typename Derived, typename T, typename ChunkT, bool Enabled>
void zero_or_value::ProxyFastCompPolicy<Derived, T, ChunkT, Enabled>::perform_compression() noexcept
{
    // Runs inside a region, next to pure readers: the buffers this pass
    // releases wait for one grace period (read_reclaim.h).
    const read_reclaim::PassScope pass_scope;
    const int                     szm          = Manager::get_size_max_number_of_chunk_to_check_compress();
    const int                     thr_rank     = get_thr_rank_fast();
    const int                     sz_threshold = szm / 2;

    const auto& ml = Manager::zero_or_value_candidates_map_thr[thr_rank];

    if constexpr (is_not_bool)
    {
        for (const auto& elm : ml)
        {
            // InT: the map key, by reference when T is large (a struct), by
            // value when it is a scalar. The map is not touched in the loop.
            const InT<T> second_val  = elm.first;
            const auto*  vec_tuple_p = elm.second.get();
            if (vec_tuple_p)
            {
                const int sz_before = static_cast<int>(vec_tuple_p->size());

                if (sz_before > sz_threshold)
                {
                    static_cast<void>(CompressionManagerType::flush_deferred(thr_rank));

                    // Every value with a candidate list is registered (set_second_value
                    // does both); tested all the same, as compress_sweep_rank does.
                    const auto* inf_ptr = Derived::array_registry->find_array_info(second_val);
                    if (inf_ptr == nullptr) [[unlikely]]
                    {
                        continue;
                    }
                    Chunk<T>::make_block_compression(thr_rank, *inf_ptr, self());

                    derived().set_info_on_chunk();
                }
            }
        }
    }
    else
    {
        const auto* vec_tuple_p = Manager::compress_candidates(thr_rank, true);
        if (vec_tuple_p)
        {
            const auto sz_before = static_cast<int>(vec_tuple_p->size());

            if (sz_before > sz_threshold)
            {
                static_cast<void>(CompressionManagerType::flush_deferred(thr_rank));

                Chunk<T>::make_block_compression_bool(thr_rank, self());

                derived().set_info_on_chunk();
            }
        }
    }
    // Looked at, whether or not any bucket had work: see the zero_any counterpart
    // and mark_list_dangerous_for_thread.
    CompressionManagerType::mark_list_safe_for_thread(thr_rank);
}

template <typename T>
void zero_or_value::Proxy<T>::perform_core_assignment(InT<T> value) const noexcept
{
    if constexpr (chunk_config::use_compression_with_counter)
    {
        if (same_value(value, static_cast<T>(Chunk<T>::first_value)))
        {
            if (Base::is_not_first_value() && SecondValueBase::is_not_second_value())
            {
                if (!same_value(this->raw_elem(), Chunk<T>::first_value))
                {
                    chunk_.inc_n_val_eq_first_value();
                }

                if (chunk_.is_full_first_value(this->chunk_size()))
                {
                    this->manage_full_first_value();
                    return;
                }
            }
        }
        else if (Base::is_not_first_value() && same_value(this->raw_elem(), Chunk<T>::first_value))
        {
            chunk_.dec_n_val_eq_first_value();
        }
    }

    // ONE load of `values` for the state tests and the store, as in
    // zero_any::Proxy::perform_core_assignment(): is_first_value(), is_second_value() (a
    // seq_cst load, ldar) and store_value() each reloaded it. Under the write
    // session the chunk cannot be compressed, only materialised by a
    // concurrent writer -- create_value() then loses its CAS, and the reload
    // after it picks the winner's buffer.
    T*         vals     = chunk_.values_load();
    const bool is_first = Base::is_first_value(vals);

    if (same_value(value, second_value_))
    {
        if (is_first)
        {
            create_value(false);

            if constexpr (chunk_config::use_compression_with_counter)
            {
                chunk_.inc_n_val_eq_second_value();
                chunk_.set_n_val_eq_first_value(this->chunk_size() - 1);
            }

            this->raw_elem_set(second_value_);
            return;
        }

        if constexpr (chunk_config::use_compression_with_counter)
        {
            chunk_.inc_n_val_eq_second_value();

            if (chunk_.is_full_second_value(this->chunk_size()))
            {
                this->manage_full_second_value();
                return;
            }
        }
    }
    else if constexpr (chunk_config::use_compression_with_counter)
    {
        if (SecondValueBase::is_not_second_value())
        {
            // Read once, before the store below: InT, no copy of a struct.
            const InT<T> vv = this->raw_elem();
            if (same_value(vv, second_value_))
            {
                chunk_.dec_n_val_eq_second_value();
            }
        }
    }

    if constexpr (chunk_config::stats)
    {
        inc_chunk_write<ChunkType>(value);
    }

    if (const bool is_second = !is_first && is_second_value_at(vals); is_first || is_second)
    {
        [[maybe_unused]] const bool chunk_was_second_value = is_second;
        [[maybe_unused]] const bool chunk_was_first_value  = is_first;

        create_value(is_second);
        vals = chunk_.values_load();

        if constexpr (chunk_config::use_compression_with_counter)
        {
            if (chunk_was_second_value)
            {
                chunk_.set_n_val_eq_second_value(this->chunk_size() - 1);
                // The write that leaves second_value skipped the first-value block
                // at the top (it only counts on a materialised chunk), and
                // create_value's crossed reset just zeroed n_val_eq_first_value_:
                // count a first_value written here. Otherwise a chunk that went
                // all-second then all-first stops at size - 1 and never
                // compresses back (serial builds: the counters are the only path).
                if (same_value(value, static_cast<T>(Chunk<T>::first_value)))
                {
                    chunk_.inc_n_val_eq_first_value();
                }
            }
            else if (same_value(value, second_value_))
            {
                chunk_.inc_n_val_eq_second_value();
                if (chunk_was_first_value)
                {
                    chunk_.set_n_val_eq_first_value(this->chunk_size() - 1);
                }
            }
            else if (chunk_was_first_value)
            {
                chunk_.set_n_val_eq_first_value(this->chunk_size() - 1);
            }
            else if (same_value(value, Chunk<T>::first_value))
            {
                chunk_.inc_n_val_eq_first_value();
            }
        }
    }

    Base::store_value_at(vals, value);
}

template <typename T>
template <typename U>
T zero_or_value::Proxy<T>::assign_impl(InT<U> value) noexcept
{
    this->check_not_stale();
    // Only one read of values_loc
    const T* const values_loc = chunk_.values_load();

    // Fast exit: already first_value and assigning first_value
    if (Base::is_first_value_equal_to(values_loc, value))
    {
        return value;
    }

    // Initialize second_value / idx for this chunk/array
    BaseSpec::set_info_second_value_idx();

    // If assigning the second compression value and we are already in that state,
    // or the current stored value is already second_value, nothing to do --
    // unless the chunk has a deferred op pending: an older op at this offset
    // would replay over the value we are leaving in place (see
    // is_first_value_equal_to); e.g. a bool re-set to true over a
    // pending `false` would lose the true.
    //
    // The stored-value test reads the buffer OUTSIDE the write session: a pass
    // in the region may release it between a pointer load and the element load,
    // and a second-constant materialisation refill it with second_value, making
    // the early-out skip a real write. So the test reloads the pointer inside a
    // ReadSection (read_reclaim.h), which that pass waits for -- only when
    // reached, so an ordinary assignment pays nothing.
    if (!chunk_.pending_any()) [[likely]]
    {
        if constexpr (is_not_bool)
        {
            if (same_value(static_cast<T>(value), second_value_)) // as stored: an int for a std::complex T
            {
                if (values_loc == chunk_.second_value_ptr(BaseSpec::idx_))
                {
                    return value;
                }
                const read_reclaim::ReadSection section;
                const T* const                  live = chunk_.values_load();
                test_hooks::maybe_read_delay(); // TEST KNOB (test_hooks.h)
                // raw-read: inside a read_reclaim::ReadSection
                if (live == chunk_.second_value_ptr(BaseSpec::idx_) || same_value(live[elem_offset_], second_value_))
                {
                    return value;
                }
            }
        }
        else
        {
            if (value)
            {
                if (values_loc == chunk_.second_value_ptr())
                {
                    return true;
                }
                // elem_get: for Bit, values_loc[elem_offset] would be a whole byte of 8
                // elements, non-zero as soon as a NEIGHBOUR is true.
                const read_reclaim::ReadSection section;
                const T* const                  live = chunk_.values_load();
                // raw-read: inside a read_reclaim::ReadSection
                if (live == chunk_.second_value_ptr() || elem_get(live, static_cast<std::size_t>(elem_offset_)))
                {
                    return true;
                }
            }
        }
    }

    // Synchronous core path (fast path)
    // By reference: value is this call's parameter, alive for the synchronous
    // call (see zero_any::Proxy::assign_impl); the deferred path copies it.
    const auto lbd_equal_core = [this, &value]
    {
        if constexpr (chunk_config::use_compression_without_counter)
        {
            this->mark_for_compression_if_needed(value);
        }
        perform_core_assignment(value);
    };

    [[maybe_unused]] const bool need_async = SecondValueBase::write_with_optional_lock_with_set_ptr(lbd_equal_core);

    if constexpr (chunk_config::use_compression_without_counter)
    {
        // Asynchronous slow path: simple but safe
        if (need_async)
        {
            using CtxType = ArithmeticAffectationContext<T, std::remove_cvref_t<U>, ChunkType, Proxy<T>>;

            // Capture only location + operand
            auto* ctx = pool_new_ctx<CtxType>(*this, value, ptr_on_array_);

            auto lbd_equal_asynch = [ctx] // the operand lives in the context: no second copy in the closure
            {
                auto&     chunk       = ctx->chunk;
                const int elem_offset = ctx->elem_offset;

                const auto proxy = Base::template make_proxy_with_info<Proxy>(ctx);

                // Deferred replay: pin the frozen per-op geometry so create_value
                // below materialises at proxy.chunk_size(), not the flushing thread's
                // ambient thread_chunk_size(). Restored on scope exit. Dynamic only.
                [[maybe_unused]] const DynChunkSizeGuard sz_guard{proxy.chunk_size()};

                const bool is_first = proxy.is_first_value();
                if (const bool is_second = proxy.is_second_value(); is_first || is_second)
                {
                    if constexpr (is_not_bool)
                    {
                        const auto array_info = Proxy<T>::get_info_second_value(ctx->ptr_on_array);
                        chunk.create_value(is_second, array_info);
                    }
                    else
                    {
                        chunk.create_value_bool(is_second);
                    }
                }

                if constexpr (build_config::any_check)
                {
                    if (const bool is_first_loc = proxy.is_first_value(); is_first_loc)
                    {
                        CHUNK_INVARIANT_FAILED("[deferred write_body] chunk still first_value after create_value "
                                               "(chunk={}, elem_offset={})",
                                               &chunk, elem_offset);
                    }

                    const bool is_second_loc = proxy.is_second_value();
                    if (is_second_loc)
                    {
                        CHUNK_INVARIANT_FAILED("[deferred write_body] chunk still second_value after create_value "
                                               "(chunk={}, elem_offset={})",
                                               &chunk, elem_offset);
                    }
                }

                T* values = chunk.values_load();
                // Check-build guard: read-side sanity check to correlate with producer-side
                // instrumentation (values_store / CAS / freelist / second_value_ptr).
                // Producer traces should fire first; if only this fires, a store site
                // is missing instrumentation.
                CHUNK_CHECK_VALUES_PTR_C(values, "deferred.write_body.values_load", &chunk);
                // The element's value before this write -- the chunk's constant if it
                // was just materialised -- for the marking below.
                const InT<T> v_old = Base::read_values_async(ctx);
                elem_set(values, static_cast<std::size_t>(elem_offset), static_cast<T>(ctx->operand));
                Base::check_nan_written_async(ctx, static_cast<T>(ctx->operand));

                // As the compound ops' replay: a replayed write that makes the chunk
                // uniform queues it, or it waits for the next direct write. After the
                // store, unlike the direct path: a flush excludes the passes.
                SecondValueBase::mark_for_compression_if_needed_with_value_async(proxy, v_old,
                                                                                 static_cast<T>(ctx->operand));
            };

            Base::enqueue_deferred_write(ctx, std::move(lbd_equal_asynch));
        }
    }

    return static_cast<T>(value);
}

template <typename T>
void zero_or_value::Proxy<T>::ensure_writable_storage()
{
    // Route this materialisation to THIS array's chunk-size pool -- the sync
    // twin of zero_any::ProxyFastCompPolicy::ensure_writable_storage_async. Required:
    // chunk.create_value() (called directly, bypassing zero_or_value::Proxy<T>::create_value()'s
    // self-correction) allocates at the per-thread thread_chunk_size, which an
    // unrelated array may have left STALE, while elem_offset/store_value use this
    // array's own size_policy_: a heap overflow once elem_offset exceeds a smaller
    // stale allocation.
    if constexpr (build_config::chunk_dynamic)
    {
        shrink_array::set_thread_chunk_size(this->chunk_size());
        shrink_array::set_thread_alloc_key(this->alloc_key());
    }

    // The materialisation sequence at the top of execute_simple_operation (see
    // header note). Any behaviour change here is a regression.
    if constexpr (chunk_config::use_compression_with_counter && requires { this->set_info_second_value_idx(); })
    {
        this->set_info_second_value_idx();
    }

    bool is_second = false;
    if constexpr (is_not_bool)
    {
        is_second = is_second_value(BaseSpec::idx_);
    }
    else
    {
        is_second = is_second_value();
    }

    if (is_second)
    {
        chunk_.create_value(true, array_info_);
        if constexpr (chunk_config::use_compression_with_counter)
        {
            chunk_.set_n_val_eq_second_value(this->chunk_size());
        }
    }

    const bool is_first = Base::is_first_value();

    if (is_first)
    {
        chunk_.create_value(false, array_info_);
        if constexpr (chunk_config::use_compression_with_counter)
        {
            chunk_.set_n_val_eq_first_value(this->chunk_size());
        }
    }
}

template <typename T>
void zero_or_value::Proxy<T>::count_constant_transitions([[maybe_unused]] bool was_second_value,
                                                         [[maybe_unused]] bool was_first_value)
{
    if constexpr (chunk_config::use_compression_with_counter)
    {
        // The value actually stored, NOT is_second_value(idx): that one tests
        // the storage pointer, false right after create_value().
        const bool now_second = same_value(this->raw_elem(), second_value_);
        const bool now_first  = same_value(this->raw_elem(), static_cast<T>(Chunk<T>::first_value));

        // Each counter on its own, never `else if`: an array that declared no
        // second value has T{} for both constants, and perform_core_assignment()
        // then moves both counters for such an element. Updating one alone
        // left the other counting an element that had left zero as still
        // zero: the next plain write that zeroed the rest of the chunk found
        // it "full" and folded it, the element with it.
        if (was_second_value && !now_second)
        {
            chunk_.dec_n_val_eq_second_value();
        }
        if (was_first_value && !now_first)
        {
            chunk_.dec_n_val_eq_first_value();
        }
        if (!was_second_value && now_second)
        {
            chunk_.inc_n_val_eq_second_value();
        }
        if (!was_first_value && now_first)
        {
            chunk_.inc_n_val_eq_first_value();
            // The first constant first, as perform_core_assignment() does:
            // with both constants T{}, the chunk folds to the first one.
            if (chunk_.is_full_first_value(this->chunk_size()))
            {
                this->manage_full_first_value();
                return;
            }
        }
        if (!was_second_value && now_second && chunk_.is_full_second_value(this->chunk_size()))
        {
            this->manage_full_second_value();
        }
    }
}

template <typename T>
template <typename Lambda>
void zero_or_value::Proxy<T>::execute_simple_operation(Lambda operation)
{
    ensure_writable_storage();

    // Capture state AFTER create_value: before it, `values` still points at the
    // shared constant buffer, so v_old / was_* would read the wrong storage and the
    // counter transitions below would desync (driving the chunk to a spurious
    // "full second value" and re-compressing it).
    [[maybe_unused]] bool was_second_value = false;
    [[maybe_unused]] bool was_first_value  = false;

    if constexpr (chunk_config::use_compression_with_counter)
    {
        const InT<T> v_old = read_values();
        was_second_value   = same_value(v_old, second_value_);
        was_first_value    = same_value(v_old, static_cast<T>(Chunk<T>::first_value));
    }

    // Without counters (fast block compression) the body -- arithmetic_assign's,
    // the only caller -- marks from its own (old, new) pair, read in the same
    // session: no second read and second marking here.
    operation();

    if constexpr (chunk_config::use_compression_with_counter)
    {
        count_constant_transitions(was_second_value, was_first_value);
    }
}

// ── execute_arithmetic_compound_operation ─────────────────────────────────
// Two entirely different implementations depending on compression mode, one
// per zero_or_value::ProxyFastCompPolicy specialisation.

template <typename Derived, typename T, typename ChunkT, bool Enabled>
template <typename Lambda>
void zero_or_value::ProxyFastCompPolicy<Derived, T, ChunkT, Enabled>::execute_simple_operation_async(auto*  ctx,
                                                                                                     Lambda operation)
{
    auto& chunk = ctx->chunk;

    const Proxy proxy = Base::template make_proxy_with_info<Derived>(ctx);

    const bool is_second = proxy.is_second_value();
    const bool is_first  = proxy.is_first_value();

    if constexpr (is_not_bool)
    {
        if (is_second)
        {
            chunk.create_value(true, proxy.array_info());
        }
        else if (is_first)
        {
            chunk.create_value(false, proxy.array_info());
        }
    }

    const InT<T> v_old  = read_values_async(ctx);
    T*           values = chunk.values_load();
    operation(values);

    const InT<T> v_new = read_values_async(ctx);
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
template <typename Derived, typename T, typename ChunkT, bool Enabled>
template <typename Op, typename Lambda, typename U>
void zero_or_value::ProxyFastCompPolicy<Derived, T, ChunkT, Enabled>::execute_arithmetic_compound_operation(
    Op       op,
    const U& operand,
    Lambda   operation)
{
    // Same rule as ensure_writable_storage(): this function may call
    // chunk.create_value() below (via the will_be_not_second_value branch)
    // without going through zero_or_value::Proxy<T>::create_value()'s self-correcting
    // wrapper -- pin the ambient size to THIS array's own geometry first.
    if constexpr (build_config::chunk_dynamic)
    {
        shrink_array::set_thread_chunk_size(this->chunk_size());
        shrink_array::set_thread_alloc_key(this->alloc_key());
    }

    using ComputeType = std::common_type_t<T, U>;

    const InT<T> v_old = read_values();

    // Perform math in high precision
    const ComputeType v_new_precise = op(static_cast<ComputeType>(v_old), static_cast<ComputeType>(operand));

    if constexpr (is_not_bool)
    {
        second_value_ = derived().make_second_value();
    }

    // Compare the PRECISE result with the second constant value
    const bool will_be_not_second_value =
        !same_value(v_new_precise, static_cast<ComputeType>(second_value_)) && derived().is_second_value();

    if (will_be_not_second_value)
    {
        chunk_.create_value(true, array_info_);
        operation(op);
    }
    else if (SecondValueBase::is_not_second_value())
    {
        this->raw_elem_set(static_cast<T>(v_new_precise));
        this->mark_for_compression_if_needed_with_value(v_old, static_cast<T>(v_new_precise));
    }
}

template <typename Derived, typename T, typename ChunkT, bool Enabled>
template <typename Op, typename Lambda, typename U>
void zero_or_value::ProxyFastCompPolicy<Derived, T, ChunkT, Enabled>::execute_arithmetic_compound_operation_async(
    auto*    ctx,
    Op       op,
    const U& operand,
    Lambda   operation) const
{
    using ComputeType  = std::common_type_t<T, U>;
    const Proxy  proxy = Base::template make_proxy_with_info<Derived>(ctx);
    const InT<T> v_old = read_values_async(ctx);

    const ComputeType v_new_precise = op(static_cast<ComputeType>(v_old), static_cast<ComputeType>(operand));

    const bool will_be_not_second_value =
        !same_value(v_new_precise, static_cast<ComputeType>(proxy.get_second_value())) && proxy.is_second_value();

    if (will_be_not_second_value)
    {
        ctx->chunk.create_value(true, proxy.array_info());
        operation(op);
    }
    else if (!proxy.is_second_value())
    {
        T* values_ptr = ctx->chunk.values_load();
        elem_set(values_ptr, static_cast<std::size_t>(ctx->elem_offset), static_cast<T>(v_new_precise));
        Base::check_nan_written_async(ctx, static_cast<T>(v_new_precise));
        proxy.mark_for_compression_if_needed_with_value(v_old, static_cast<T>(v_new_precise));
    }
}

/**
 * @brief Serial (counter) form of execute_arithmetic_compound_operation()
 *        (Enabled=false policy).
 * @tparam Lambda Callable `(op, T* values)`.
 */
template <typename Derived, typename T, typename ChunkT>
template <typename Op, typename Lambda, typename U>
void zero_or_value::ProxyFastCompPolicy<Derived, T, ChunkT, false>::execute_arithmetic_compound_operation(
    Op       op,
    const U& operand,
    Lambda   operation)
{
    // Same rule as ensure_writable_storage(): see that function's comment.
    if constexpr (build_config::chunk_dynamic)
    {
        shrink_array::set_thread_chunk_size(this->chunk_size());
        shrink_array::set_thread_alloc_key(this->alloc_key());
    }

    using ComputeType = std::common_type_t<T, BaseT<U>>;

    if (const bool was_not_written = !chunk_.written(this->idx_); was_not_written)
    {
        const ComputeType intermediate =
            op(static_cast<ComputeType>(this->raw_elem()), static_cast<ComputeType>(operand));
        // Built where it is used: one write of a struct for one value.
        const auto res = static_cast<T>(intermediate);

        const bool will_be_not_second_value = !same_value(res, second_value_) && derived().is_second_value();

        if (will_be_not_second_value)
        {
            chunk_.create_value(true, array_info_);
            if constexpr (chunk_config::use_compression_with_counter)
            {
                chunk_.set_n_val_eq_second_value(this->chunk_size() - 1);
            }
            operation(op, Base::values_);
            return;
        }
    }

    operation(op, Base::values_);
}

/**
 * @brief Handles simple compound operations (+, -, bit operations)
 * @tparam Lambda Callable that performs the operation
 * @param operation Callable taking the element (T&) and applying the operation to it
 *
 * @details
 * Pattern for operations like +=, -=, &=, |=, ^=, <<=, >>=, ++, --
 * 1. Allocate storage if compressed
 * 2. Read old value
 * 3. Apply operation
 * 4. Track for compression
 */
template <typename T>
template <typename Lambda>
void zero_or_value::Proxy<T>::execute_inplace_operation(Lambda operation)
{
    // Same rule as ensure_writable_storage(): see that function's comment.
    if constexpr (build_config::chunk_dynamic)
    {
        shrink_array::set_thread_chunk_size(this->chunk_size());
        shrink_array::set_thread_alloc_key(this->alloc_key());
    }

    if constexpr (chunk_config::use_compression_with_counter && requires { this->set_info_second_value_idx(); })
    {
        this->set_info_second_value_idx();
    }

    const bool is_second = SecondValueBase::is_second_value();
    const bool is_first  = Base::is_first_value();

    [[maybe_unused]] T v_old{};
    if constexpr (chunk_config::use_compression_without_counter)
    {
        v_old = read_values();
    }

    // A compressed chunk is materialised before the in-place update: its
    // `values` is the shared first_value/second_value block, which an update through it
    // would rewrite for every chunk of every array at that constant.
    if (is_second || is_first)
    {
        if constexpr (is_not_bool)
        {
            Base::chunk_.create_value(is_second, array_info_);
        }
        else
        {
            Base::chunk_.create_value_bool(is_second);
        }
        if constexpr (chunk_config::use_compression_with_counter)
        {
            if (is_second)
            {
                chunk_.set_n_val_eq_second_value(this->chunk_size());
            }
            else if (is_first)
            {
                chunk_.set_n_val_eq_first_value(this->chunk_size());
            }
        }
    }

    // Capture position-level state AFTER any create_value: before it, `values`
    // still points at the shared first_value/second_value buffer, so values[elem_offset] would
    // read the wrong storage and corrupt the counter transitions below (the
    // chunk could be wrongly driven to "full second value" and re-compressed).
    [[maybe_unused]] bool was_second_value = false;
    [[maybe_unused]] bool was_first_value  = false;

    if constexpr (chunk_config::use_compression_with_counter)
    {
        was_second_value = same_value(this->raw_elem(), second_value_);
        was_first_value  = same_value(this->raw_elem(), static_cast<T>(Chunk<T>::first_value));
    }

    elem_update(static_cast<T*>(Base::values_), static_cast<std::size_t>(elem_offset_), operation);

    if constexpr (chunk_config::use_compression_with_counter)
    {
        count_constant_transitions(was_second_value, was_first_value);
    }

    if constexpr (chunk_config::use_compression_without_counter)
    {
        this->mark_for_compression_if_needed_with_value(v_old, read_values());
    }
}

// ── execute_inplace_operation_async ───────────────────────────────────────
// Only present when fast block compression is active (the Enabled=true policy).

template <typename Derived, typename T, typename ChunkT, bool Enabled>
template <typename Lambda>
void zero_or_value::ProxyFastCompPolicy<Derived, T, ChunkT, Enabled>::execute_inplace_operation_async(
    auto*  ctx,
    Lambda operation) const
{
    auto&     chunk       = ctx->chunk;
    const int elem_offset = ctx->elem_offset;

    const InT<T> v_old = read_values_async(ctx);
    const Proxy  proxy = Base::template make_proxy_with_info<Derived>(ctx);

    const bool is_first = chunk.is_first_value();
    if (const bool is_second = proxy.is_second_value(); is_first || is_second)
    {
        if constexpr (is_not_bool)
        {
            const auto array_info = Proxy<T>::get_info_second_value(ctx->ptr_on_array);
            chunk.create_value(is_second, array_info);
        }
        else
        {
            chunk.create_value_bool(is_second, get_thr_rank_fast());
        }
    }

    T* values = chunk.values_load();
    // raw-read: deferred replay, after materialisation -- flushes exclude the stealers
    elem_update(values, static_cast<std::size_t>(elem_offset), operation);

    // The value just written, read for the counters: nothing writes it
    // afterwards, so InT (a reference for a struct) rather than a copy.
    // raw-read: deferred replay, after materialisation -- flushes exclude the stealers
    const InT<T> v_new = elem_get(values, static_cast<std::size_t>(elem_offset));
    SecondValueBase::mark_for_compression_if_needed_with_value_async(proxy, v_old, v_new);
}

template <typename T>
template <typename U, typename Op>
void zero_or_value::Proxy<T>::apply_compound_operation([[maybe_unused]] Op op, InT<U> operand) noexcept
{
    this->check_not_stale();
    // Cross-rank RMW materialisation: pull any foreign
    // deferred write to this offset to applied state before reading v_old, so the
    // RMW cannot clobber it. precise gate, outside begin_write. See
    // ProxyBase::arithmetic_assign for the full rationale.
    chunk_.reader_sync(elem_offset_);

    if constexpr (chunk_config::use_compression_with_counter)
    {
        BaseSpec::set_info_second_value_idx(); // Initialize second_value / idx — needed by both branches below

        const InT<T> v_old_snapshot     = this->raw_elem();
        const bool   equals_first_value = same_value(v_old_snapshot, static_cast<T>(Chunk<T>::first_value));

        if (!equals_first_value) [[likely]]
        {
            const bool was_second_value  = same_value(v_old_snapshot, second_value_);
            const bool was_second_value0 = was_second_value && SecondValueBase::is_not_second_value();

            execute_arithmetic_compound_operation(
                op, operand,
                // operand by reference (this call's parameter, alive for the call);
                // `current` InT: the right-hand side is evaluated in full
                // before the store, so a reference into the slot is read once,
                // before it is written.
                [this, &operand](auto op_func, T* val_ptr)
                {
                    using LocalComputeType = std::common_type_t<T, BaseT<U>>;
                    const T current        = elem_get(val_ptr, static_cast<std::size_t>(elem_offset_));
                    elem_set(val_ptr, static_cast<std::size_t>(elem_offset_),
                             static_cast<T>(op_func(static_cast<LocalComputeType>(current),
                                                    static_cast<LocalComputeType>(operand))));
                });

            // Read after the operation, compared before each branch's action
            // (a manage_full_* may fold the chunk, and nothing reads v_new
            // after it): InT, no copy of a struct.
            const InT<T> v_new = this->raw_elem();
            Base::check_nan_written(v_new); // the result, not the operand: `x += NAN` and `0.0 / 0.0`
            // Transition OUT of the second value first, independently of the
            // transition IN below (as execute_simple_operation does). Do not
            // chain it to the first-value branch: `second_value * 0` (or
            // `/ inf`) would count the new zero and keep the old second value
            // counted, and a later write would fold the chunk to its second value with one
            // element still different (lost).
            if (was_second_value0 && !same_value(v_new, second_value_))
            {
                chunk_.dec_n_val_eq_second_value();
            }
            if (Base::is_not_first_value() && same_value(v_new, Chunk<T>::first_value))
            {
                chunk_.inc_n_val_eq_first_value();
                if (chunk_.is_full_first_value(this->chunk_size()))
                {
                    this->manage_full_first_value();
                }
            }
            else if (!was_second_value && same_value(v_new, second_value_))
            {
                chunk_.inc_n_val_eq_second_value();
                if (chunk_.is_full_second_value(this->chunk_size()))
                {
                    this->manage_full_second_value();
                }
            }
        }
        else if constexpr (std::is_floating_point_v<T> || internal::is_std_complex<T>::value)
        {
            const auto first = static_cast<T>(Chunk<T>::first_value);
            if (!op_keeps_value(op, first, operand))
            {
                // A first value the operation does not keep (0 / 0, 0 * inf:
                // NaN; 0 * -1: -0.0). Serial: the result is op(first_value,
                // operand), written as an assignment, which keeps the counters.
                static_cast<void>(assign_impl<T>(static_cast<T>(op(first, static_cast<T>(operand)))));
            }
        }
    }

    if constexpr (chunk_config::use_compression_without_counter)
    {
        // Outside the write session: under a ReadSection, or the `0 * x`
        // shortcut may act on a released, zero-filled buffer (see
        // ProxyBase::read_values_outside_session).
        const InT<T> v_old = Base::read_values_outside_session();
        // The first value absorbs (0 * x == 0) when the operation keeps it.
        if (const auto first = static_cast<T>(Chunk<T>::first_value); same_value(v_old, first))
        {
            // Not kept (0 / 0, 0 * inf: NaN; 0 * -1: -0.0, see op_keeps_value):
            // the result is op(first, operand), written as an assignment, which
            // materialises a constant chunk. Not apply_atomic_op(): it takes
            // the chunk for materialised, and would write into the shared
            // constant.
            if constexpr (std::is_floating_point_v<T> || internal::is_std_complex<T>::value)
            {
                if (!op_keeps_value(op, first, operand))
                {
                    static_cast<void>(assign_impl<T>(static_cast<T>(op(first, static_cast<T>(operand)))));
                }
            }
        }
        else
        {
            // An operand the operation cannot take as it is -- an integer for a
            // std::complex, whose operators are templates -- goes in as a T.
            if constexpr (std::is_invocable_r_v<T, Op, T, BaseT<U>>)
            {
                Base::apply_atomic_op(op, operand, v_old);
            }
            else
            {
                Base::apply_atomic_op(op, static_cast<T>(operand), v_old);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// shrink_array::ProxyFastCompPolicy<Derived, T, ChunkT, Enabled>
// ---------------------------------------------------------------------------

template <typename Derived, typename T, typename ChunkT, bool Enabled>
[[nodiscard]] auto zero_or_value::ProxyFastCompPolicy<Derived, T, ChunkT, Enabled>::get_second_value() const noexcept
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

template <typename Derived, typename T, typename ChunkT, bool Enabled>
CandidateList<zero_or_value::Chunk, T>&
zero_or_value::ProxyFastCompPolicy<Derived, T, ChunkT, Enabled>::candidates_of_thread(int thr) const noexcept
{
    const auto second_value = get_second_value();
    // find(), not operator[]: operator[] INSERTS a null entry for a missing
    // key, which has_queued() would dereference later.
    auto&      candidates = Manager::zero_or_value_candidates_map_thr[thr];
    const auto it         = candidates.find(second_value);
    if (it == candidates.end() || it->second == nullptr) [[unlikely]]
    {
        zero_or_value_missing_candidate_list(thr);
    }
    return *it->second;
}

template <typename Derived, typename T, typename ChunkT, bool Enabled>
CandidateList<zero_or_value::Chunk, T>&
zero_or_value::ProxyFastCompPolicy<Derived, T, ChunkT, Enabled>::candidates_of_thread_async(auto* ctx, int thr) const
{
    const auto second_value = get_second_value(ctx);
    // find(), not operator[]: operator[] INSERTS a null entry for a missing
    // key, which has_queued() would dereference later.
    auto&      candidates = Manager::zero_or_value_candidates_map_thr[thr];
    const auto it         = candidates.find(second_value);
    if (it == candidates.end() || it->second == nullptr) [[unlikely]]
    {
        zero_or_value_missing_candidate_list(thr);
    }
    return *it->second;
}

template <typename Derived, typename T, typename ChunkT, bool Enabled>
[[nodiscard]] auto zero_or_value::ProxyFastCompPolicy<Derived, T, ChunkT, Enabled>::get_second_value(auto* ctx) noexcept
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

// ---------------------------------------------------------------------------
// shrink_array::ProxyFastCompPolicy<Derived, T, ChunkT, false>
// ---------------------------------------------------------------------------

template <typename Derived, typename T, typename ChunkT>
void zero_or_value::ProxyFastCompPolicy<Derived, T, ChunkT, false>::manage_full_first_value() const
{
    // Pin the ambient chunk size before the free chain (clear_st -> ... ->
    // get_chunk_allocator(thr_rank).free_elem_pow2()): get_chunk_allocator() routes by the
    // ambient thread_chunk_size(), not by the buffer. A stale ambient (left by another
    // array's write) selects the wrong pool, which then removes a node it does not own.
    // Fast-alloc only: std-alloc's svfree() does not route by ambient.
    if constexpr (build_config::chunk_dynamic)
    {
        shrink_array::set_thread_chunk_size(this->chunk_size());
    }
    if constexpr (is_not_bool)
    {
        chunk_.clear_st(0, this->idx_);
    }
    else
    {
        chunk_.clear_st(0, 0); // bool: idx ignored by bool-aware clear_int_st
    }
    chunk_.values_ = static_cast<T*>(chunk_.first_value_ptr());
}

template <typename Derived, typename T, typename ChunkT>
void zero_or_value::ProxyFastCompPolicy<Derived, T, ChunkT, false>::manage_full_second_value() const
{
    // Same rule as manage_full_first_value(): see that function's comment.
    if constexpr (build_config::chunk_dynamic)
    {
        shrink_array::set_thread_chunk_size(this->chunk_size());
    }
    if constexpr (is_not_bool)
    {
        chunk_.clear_st(1, this->idx_);
    }
    else
    {
        chunk_.clear_st(1, 0); // bool: idx ignored by bool-aware clear_int_st
    }
}

// ---------------------------------------------------------------------------
// shrink_array::Proxy<T>
// ---------------------------------------------------------------------------

template <typename T>
void zero_or_value::Proxy<T>::create_value(bool is_second) const noexcept
{
    if constexpr (build_config::chunk_dynamic)
    {
        // Route this (possibly worker-thread) allocation to the pool of THIS array's chunk
        // size: set the per-thread active size right before materialising.
        shrink_array::set_thread_chunk_size(this->chunk_size());
    }
    if constexpr (is_not_bool)
    {
        chunk_.create_value(is_second, array_info_);
    }
    else
    {
        chunk_.create_value_bool(is_second);
    }
}

template <typename T>
const zero_or_value::Proxy<T>& zero_or_value::Proxy<T>::operator=(const Proxy& from) noexcept
{
    if (std::addressof(from) != this) // operator& is deleted
    {
        const InT<T> value = static_cast<T>(from); // complete before the write starts
        static_cast<void>(assign_impl<T>(value));  // the value written: not needed here
    }
    return *this;
}

template <typename T>
[[nodiscard]] bool zero_or_value::Proxy<T>::is_second_value(int idx) const noexcept
{
    if constexpr (is_not_bool)
    {
        return chunk_.is_second_value(idx);
    }
    else
    {
        CHUNK_INVARIANT_FAILED(
            "zero_or_value::Proxy<bool>::is_second_value(index_value): bool has no indexed second constant "
            "-- use is_second_value()");
    }
}

template <typename T>
[[nodiscard]] bool zero_or_value::Proxy<T>::is_second_value() const noexcept
{
    if constexpr (is_not_bool)
    {
        return chunk_.is_second_value(BaseSpec::idx_);
    }
    else
    {
        return chunk_.is_second_value();
    }
}

template <typename T>
[[nodiscard]] bool zero_or_value::Proxy<T>::is_second_value_at(const T* vals) const noexcept
{
    if constexpr (is_not_bool)
    {
        return vals == chunk_.second_value_ptr(BaseSpec::idx_);
    }
    else
    {
        return vals == chunk_.second_value_ptr();
    }
}

template <typename T>
[[nodiscard]] bool zero_or_value::Proxy<T>::has_candidate_list(InT<T> second_val) noexcept
{
    if constexpr (build_config::threaded && chunk_config::use_fast_block_compression)
    {
        const auto& per_thr = Manager::zero_or_value_candidates_map_thr;
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

} // namespace shrink_array