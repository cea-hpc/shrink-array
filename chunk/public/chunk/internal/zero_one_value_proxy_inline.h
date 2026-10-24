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

#include "chunk/internal/in_t.h"                 // InT<T>
#include "chunk/internal/bad_ptr_check.h"        // CHUNK_CHECK_VALUES_PTR (check builds)
#include "chunk/internal/zero_one_value_proxy.h" // the declarations these bodies belong to
#include "chunk/config.h"    // chunk_config::use_compression_with_counter, build_config::chunk_dynamic
#include "chunk/PMP_CHUNK.h" // get_thr_rank_fast
#include "chunk/internal/arithmetic_context.h"           // ArithmeticAffectationContext
#include "chunk/internal/chunk_size_policy.h"            // DynChunkSizeGuard, set_thread_chunk_size
#include "chunk/internal/concept_is_clean_assignement.h" // BaseT
#include "chunk/internal/invariant_check.h"              // CHUNK_INVARIANT_FAILED
#include "chunk/internal/read_reclaim.h" // read_reclaim::PassScope (perform_compression / compress in a region)
#include "chunk/internal/test_hooks.h"   // TEST KNOB: test_hooks::maybe_read_delay
#include "chunk/log.h"                   // CHUNK_LOG_ALWAYS (verbose)

namespace shrink_array
{

// No acquire fence ends set_info_second_value_ind() or the two
// set_info_on_chunk(): they would follow plain (non-atomic) reads of the
// share_info_array registry and pair with no release. Do not add one: on
// aarch64 it was the hottest instruction of assign_impl (several % of a run).

// The armed mirror, checked against the registry under any_check; nullptr when
// the proxy is not armed. Callers read the fields THROUGH this pointer: see the
// note on set_info_on_chunk() below for why not through a returned copy.
template <typename T, typename Derived>
const zero_one_value::ProxySpec<T, Derived>::InfArrayT*
zero_one_value::ProxySpec<T, Derived>::armed_inf_array(this const Derived& self)
{
    if (self.inf_array_src_ != nullptr) [[likely]]
    {
        if constexpr (build_config::any_check)
        {
            // ind and num_tab only: ind maps one-to-one to the value, and a
            // value compare would raise a false alarm on a NaN second value.
            const InfArrayT reg = Derived::get_info_second_value(self.get_ptr_on_array());
            if (reg.ind != self.inf_array_src_->ind || reg.num_tab != self.inf_array_src_->num_tab)
            {
                CHUNK_INVARIANT_FAILED(
                    "zero_one_value::ProxySpec::load_inf_array: armed (ind {}, num_tab {}) differs from the registry's "
                    "(ind {}, num_tab {}) for array {} -- a registry write without sync_registry_mirror()?",
                    self.inf_array_src_->ind, self.inf_array_src_->num_tab, reg.ind, reg.num_tab,
                    self.get_ptr_on_array());
            }
        }
    }
    return self.inf_array_src_;
}

template <typename T, typename Derived>
zero_one_value::ProxySpec<T, Derived>::InfArrayT
zero_one_value::ProxySpec<T, Derived>::load_inf_array(this const Derived& self)
{
    if (const InfArrayT* src = self.armed_inf_array()) [[likely]]
    {
        return *src;
    }
    return Derived::get_info_second_value(self.get_ptr_on_array());
}

template <typename T, typename Derived>
    requires is_bool_like_v<T>
const zero_one_value::ProxySpec<T, Derived>::InfArrayT*
zero_one_value::ProxySpec<T, Derived>::armed_inf_array(this const Derived& self)
{
    if (self.inf_array_src_ != nullptr) [[likely]]
    {
        if constexpr (build_config::any_check)
        {
            // ind and num_tab only: ind maps one-to-one to the value, and a
            // value compare would raise a false alarm on a NaN second value.
            const InfArrayT reg = Derived::get_info_second_value(self.get_ptr_on_array());
            if (reg.ind != self.inf_array_src_->ind || reg.num_tab != self.inf_array_src_->num_tab)
            {
                CHUNK_INVARIANT_FAILED(
                    "zero_one_value::ProxySpec::load_inf_array: armed (ind {}, num_tab {}) differs from the registry's "
                    "(ind {}, num_tab {}) for array {} -- a registry write without sync_registry_mirror()?",
                    self.inf_array_src_->ind, self.inf_array_src_->num_tab, reg.ind, reg.num_tab,
                    self.get_ptr_on_array());
            }
        }
    }
    return self.inf_array_src_;
}

template <typename T, typename Derived>
    requires is_bool_like_v<T>
zero_one_value::ProxySpec<T, Derived>::InfArrayT
zero_one_value::ProxySpec<T, Derived>::load_inf_array(this const Derived& self)
{
    if (const InfArrayT* src = self.armed_inf_array()) [[likely]]
    {
        return *src;
    }
    return Derived::get_info_second_value(self.get_ptr_on_array());
}

// Every write runs one of these. They take the fields from the armed mirror --
// memory written long before -- never from a copy just stored. Not
//     inf_array = load_inf_array(); second_value = inf_array.second_value; ...
// which returns the 16-byte record through a stack temporary (a q-register
// store) and reads it straight back with 8- and 4-byte loads: a store the core
// cannot forward to narrower loads, so each write waits for it to reach the
// cache -- a large share of assign_impl's time on AArch64. Unarmed proxies
// (replay, *_from_key) keep the registry path: rare, and a registry lookup
// costs more than the stall anyway.
template <typename T, typename Derived>
void zero_one_value::ProxySpec<T, Derived>::set_info_on_chunk(this Derived& self)
{
    if (const InfArrayT* src = self.armed_inf_array()) [[likely]]
    {
        static_cast<ProxySpec&>(self).inf_array_ = *src;
        self.second_value_                       = src->second_value;
        self.ind_                                = src->ind;
        self.set_num_tab(src->num_tab); // what set_num_tab() would read back from inf_array_
    }
    else
    {
        static_cast<ProxySpec&>(self).inf_array_ = Derived::get_info_second_value(self.get_ptr_on_array());
        self.second_value_                       = static_cast<ProxySpec&>(self).inf_array_.second_value;
        self.ind_                                = static_cast<ProxySpec&>(self).inf_array_.ind;
        self.set_num_tab();
    }

    if constexpr (chunk_config::use_fast_block_compression)
    {
        self.set_ptr_is_inside_chunk_list_to_compress();
    }
}

template <typename T, typename Derived>
void zero_one_value::ProxySpec<T, Derived>::set_info_second_value_ind(this Derived& self)
{
    if (const InfArrayT* src = self.armed_inf_array()) [[likely]]
    {
        static_cast<ProxySpec&>(self).inf_array_ = *src;
        self.second_value_                       = src->second_value;
        self.ind_                                = src->ind;
        return;
    }
    static_cast<ProxySpec&>(self).inf_array_ = Derived::get_info_second_value(self.get_ptr_on_array());
    self.second_value_                       = static_cast<ProxySpec&>(self).inf_array_.second_value;
    self.ind_                                = static_cast<ProxySpec&>(self).inf_array_.ind;
}

template <typename T, typename Derived>
void zero_one_value::ProxySpec<T, Derived>::set_info_second_value_ind_from_key(ArrayKey<Chunk<T>> key)
{
    inf_array_    = Derived::get_info_second_value(key);
    second_value_ = inf_array_.second_value;
    ind_          = inf_array_.ind;
}

template <typename T, typename Derived>
    requires is_bool_like_v<T>
void zero_one_value::ProxySpec<T, Derived>::set_info_on_chunk(this Derived& self)
{
    if (const InfArrayT* src = self.armed_inf_array()) [[likely]]
    {
        static_cast<ProxySpec&>(self).inf_array_ = *src;
        self.set_num_tab(src->num_tab); // not read back from inf_array_ (see above)
    }
    else
    {
        static_cast<ProxySpec&>(self).inf_array_ = Derived::get_info_second_value(self.get_ptr_on_array());
        self.set_num_tab();
    }

    if constexpr (chunk_config::use_fast_block_compression)
    {
        self.set_ptr_is_inside_chunk_list_to_compress();
    }
}

// ── Methods only present when fast block compression is active ────────────
// Members of the Enabled=true policy only: zero_one_value::Proxy instantiates
// zero_one_value::ProxyFastCompPolicy<..., chunk_config::use_fast_block_compression>, and the
// <..., false> specialisation declares none of them, so nothing here is ever
// looked at by a serial build. No preprocessor guard needed.

template <typename Derived, typename T, typename ChunkT, bool Enabled>
void zero_one_value::ProxyFastCompPolicy<Derived, T, ChunkT, Enabled>::trigger_compression_if_needed(
    this Derived& self,
    int           thr_rank) noexcept
{
    if constexpr (is_not_bool)
    {
        self.second_value_ = self.make_second_value();
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
void zero_one_value::ProxyFastCompPolicy<Derived, T, ChunkT, Enabled>::perform_compression(this Derived& self) noexcept
{
    // Runs inside a region, next to pure readers: the buffers this pass
    // releases wait for one grace period (read_reclaim.h).
    const read_reclaim::PassScope pass_scope;
    const int                     szm          = ManagerChunk2::get_size_max_number_of_chunk_to_check_compress();
    const int                     thr_rank     = get_thr_rank_fast();
    const int                     sz_threshold = szm / 2;

    const auto& ml = ManagerChunk2::map_chunk2_to_compress_potentially_thr[thr_rank];

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

                    const auto* inf_ptr = Derived::share_info_array->find_inf_array(second_val);
                    zero_one_value::Chunk<T>::make_block_compression(thr_rank, *inf_ptr, std::addressof(self));

                    self.set_info_on_chunk();
                }
            }
        }
    }
    else
    {
        const auto* vec_tuple_p = ManagerChunk2::compress_candidates(thr_rank, true);
        if (vec_tuple_p)
        {
            const int sz_before = static_cast<int>(vec_tuple_p->size());

            if (sz_before > sz_threshold)
            {
                static_cast<void>(CompressionManagerType::flush_deferred(thr_rank));

                zero_one_value::Chunk<T>::make_block_compression_bool(thr_rank, std::addressof(self));

                self.set_info_on_chunk();
            }
        }
    }
    // Looked at, whether or not any bucket had work: see the zero_any counterpart
    // and mark_list_dangerous_for_thread.
    CompressionManagerType::mark_list_safe_for_thread(thr_rank);
}

template <typename T>
void zero_one_value::Proxy<T>::perform_core_assignment(InT<T> v) const noexcept
{
    if constexpr (chunk_config::use_compression_with_counter)
    {
        if (v == static_cast<T>(zero_one_value::Chunk<T>::first_value))
        {
            if (Base::is_not_cst_val1() && Base2::is_not_cst_val2())
            {
                if (this->raw_elem() != zero_one_value::Chunk<T>::first_value)
                {
                    chunk_.inc_nb_val_eq_first_value();
                }

                if (chunk_.is_full_first_value(this->chunk_size()))
                {
                    this->manage_full_first_value();
                    return;
                }
            }
        }
        else if (Base::is_not_cst_val1() && this->raw_elem() == zero_one_value::Chunk<T>::first_value)
        {
            chunk_.dec_nb_val_eq_first_value();
        }
    }

    // ONE load of `values` for the state tests and the store, as in
    // zero_any::Proxy::perform_core_assignment(): is_cst_val1(), is_cst_val2() (a
    // seq_cst load, ldar) and store_value() each reloaded it. Under the write
    // session the chunk cannot be compressed, only materialised by a
    // concurrent writer -- create_value() then loses its CAS, and the reload
    // after it picks the winner's buffer.
    T*         vals    = chunk_.values_load();
    const bool is_cst1 = Base::is_cst_val1(vals);

    if (v == second_value_)
    {
        if (is_cst1)
        {
            create_value(false);

            if constexpr (chunk_config::use_compression_with_counter)
            {
                chunk_.inc_nb_val_eq_second_value();
                chunk_.set_nb_val_eq_first_value(this->chunk_size() - 1);
            }

            this->raw_elem_set(second_value_);
            return;
        }

        if constexpr (chunk_config::use_compression_with_counter)
        {
            chunk_.inc_nb_val_eq_second_value();

            if (chunk_.is_full_second_value(this->chunk_size()))
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
            // Read once, before the store below: InT, no copy of a struct.
            const InT<T> vv = this->raw_elem();
            if (vv == second_value_)
            {
                chunk_.dec_nb_val_eq_second_value();
            }
        }
    }

    if constexpr (chunk_config::stats)
    {
        increm_chunk_write<ChunkType>(v);
    }

    const bool is_cst2 = !is_cst1 && is_cst_val2_at(vals);

    if (is_cst1 || is_cst2)
    {
        [[maybe_unused]] const bool was_cst_val2 = is_cst2;
        [[maybe_unused]] const bool was_cst_val1 = is_cst1;

        create_value(is_cst2);
        vals = chunk_.values_load();

        if constexpr (chunk_config::use_compression_with_counter)
        {
            if (was_cst_val2)
            {
                chunk_.set_nb_val_eq_second_value(this->chunk_size() - 1);
                // The write that leaves cst_val2 skipped the first-value block
                // at the top (it only counts on a materialised chunk), and
                // create_value's crossed reset just zeroed nb_val_eq_first_value:
                // count a first_value written here. Otherwise a chunk that went
                // all-second then all-first stops at size - 1 and never
                // compresses back (serial builds: the counters are the only path).
                if (v == static_cast<T>(zero_one_value::Chunk<T>::first_value))
                {
                    chunk_.inc_nb_val_eq_first_value();
                }
            }
            else if (v == second_value_)
            {
                chunk_.inc_nb_val_eq_second_value();
                if (was_cst_val1)
                {
                    chunk_.set_nb_val_eq_first_value(this->chunk_size() - 1);
                }
            }
            else if (was_cst_val1)
            {
                chunk_.set_nb_val_eq_first_value(this->chunk_size() - 1);
            }
            else if (v == zero_one_value::Chunk<T>::first_value)
            {
                chunk_.inc_nb_val_eq_first_value();
            }
        }
    }

    Base::store_value_at(vals, v);

    if constexpr (build_config::do_check)
    {
        global_check_write_nan(v, num_);
    }
}

template <typename T>
template <typename U>
T zero_one_value::Proxy<T>::assign_impl(InT<U> v) noexcept
{
    this->check_not_stale();
    // Only one read of values_loc
    const T* const values_loc = chunk_.values_load();

    // Fast exit: already first_value and assigning first_value
    if (Base::is_cst_val1_and_v_equal_first_value(values_loc, v))
    {
        return v;
    }

    // Initialize second_value / ind for this chunk/array
    BaseSpec::set_info_second_value_ind();

    // If assigning the second compression value and we are already in that state,
    // or the current stored value is already second_value, nothing to do --
    // unless the chunk has a deferred op pending: an older op at this offset
    // would replay over the value we are leaving in place (see
    // is_cst_val1_and_v_equal_first_value); e.g. a bool re-set to true over a
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
            if (v == second_value_)
            {
                if (values_loc == chunk_.get_cst_val2(BaseSpec::ind_))
                {
                    return v;
                }
                const read_reclaim::ReadSection section;
                const T* const                  live = chunk_.values_load();
                test_hooks::maybe_read_delay(); // TEST KNOB (test_hooks.h)
                // raw-read: inside a read_reclaim::ReadSection
                if (live == chunk_.get_cst_val2(BaseSpec::ind_) || live[num_] == second_value_)
                {
                    return v;
                }
            }
        }
        else
        {
            if (v)
            {
                if (values_loc == chunk_.get_cst_val2())
                {
                    return true;
                }
                // elem_get: for Bit, values_loc[num] would be a whole byte of 8
                // elements, non-zero as soon as a NEIGHBOUR is true.
                const read_reclaim::ReadSection section;
                const T* const                  live = chunk_.values_load();
                // raw-read: inside a read_reclaim::ReadSection
                if (live == chunk_.get_cst_val2() || elem_get(live, static_cast<std::size_t>(num_)))
                {
                    return true;
                }
            }
        }
    }

    // Synchronous core path (fast path)
    // By reference: v is this call's parameter, alive for the synchronous
    // call (see zero_any::Proxy::assign_impl); the deferred path copies it.
    const auto lbd_equal_core = [this, &v]
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
            using CtxType =
                ArithmeticAffectationContext<T, std::remove_cvref_t<U>, ChunkType, zero_one_value::Proxy<T>>;

            // Capture only location + operand
            auto* ctx = pool_new_ctx<CtxType>(*this, v, ptr_on_array_);

            auto lbd_equal_asynch = [ctx] // the operand lives in the context: no second copy in the closure
            {
                auto&     chunk = ctx->chunk;
                const int num   = ctx->num;

                const auto proxy = Base::template make_proxy_with_info<Proxy>(ctx);

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
                        const auto inf = zero_one_value::Proxy<T>::get_info_second_value(ctx->ptr_on_array);
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
                    {
                        CHUNK_INVARIANT_FAILED("[deferred write_body] chunk still cst_val1 after create_value "
                                               "(chunk={}, num={})",
                                               &chunk, num);
                    }

                    const bool is_cst2_loc = proxy.is_cst_val2();
                    if (is_cst2_loc)
                    {
                        CHUNK_INVARIANT_FAILED("[deferred write_body] chunk still cst_val2 after create_value "
                                               "(chunk={}, num={})",
                                               &chunk, num);
                    }
                }

                T* values = chunk.values_load();
                // Check-build guard: read-side sanity check to correlate with producer-side
                // instrumentation (values_store / CAS / freelist / get_cst_val2).
                // Producer traces should fire first; if only this fires, a store site
                // is missing instrumentation.
                CHUNK_CHECK_VALUES_PTR_C(values, "deferred.write_body.values_load", &chunk);
                elem_set(values, static_cast<std::size_t>(num), static_cast<T>(ctx->operand));
            };

            Base::enqueue_deferred_write(ctx, std::move(lbd_equal_asynch));
        }
    }

    return static_cast<T>(v);
}

template <typename T>
void zero_one_value::Proxy<T>::ensure_writable_storage()
{
    // Route this materialisation to THIS array's chunk-size pool -- the sync
    // twin of zero_any::ProxyFastCompPolicy::ensure_writable_storage_async. Required:
    // chunk.create_value() (called directly, bypassing zero_one_value::Proxy<T>::create_value()'s
    // self-correction) allocates at the per-thread thread_chunk_size, which an
    // unrelated array may have left STALE, while num/store_value use this
    // array's own size_policy_: a heap overflow once num exceeds a smaller
    // stale allocation.
    if constexpr (build_config::chunk_dynamic)
    {
        shrink_array::set_thread_chunk_size(this->chunk_size());
        shrink_array::set_thread_alloc_key(this->alloc_key());
    }

    // The materialisation sequence at the top of execute_simple_operation (see
    // header note). Any behaviour change here is a regression.
    if constexpr (chunk_config::use_compression_with_counter && requires { this->set_info_second_value_ind(); })
    {
        this->set_info_second_value_ind();
    }

    bool is_val2 = false;
    if constexpr (is_not_bool)
    {
        is_val2 = is_cst_val2(BaseSpec::ind_);
    }
    else
    {
        is_val2 = is_cst_val2();
    }

    if (is_val2)
    {
        chunk_.create_value(true, inf_array_);
        if constexpr (chunk_config::use_compression_with_counter)
        {
            chunk_.set_nb_val_eq_second_value(this->chunk_size());
        }
    }

    const bool is_val1 = Base::is_cst_val1();

    if (is_val1)
    {
        chunk_.create_value(false, inf_array_);
        if constexpr (chunk_config::use_compression_with_counter)
        {
            chunk_.set_nb_val_eq_first_value(this->chunk_size());
        }
    }
}

template <typename T>
template <typename Lambda>
void zero_one_value::Proxy<T>::execute_simple_operation(Lambda operation)
{
    ensure_writable_storage();

    // Capture state AFTER create_value: before it, `values` still points at the
    // shared cst buffer, so v_old / was_* would read the wrong storage and the
    // counter transitions below would desync (driving the chunk to a spurious
    // "full second value" and re-compressing it).
    [[maybe_unused]] bool was_second_value = false;
    [[maybe_unused]] bool was_first_value  = false;

    if constexpr (chunk_config::use_compression_with_counter)
    {
        const InT<T> v_old = read_values();
        was_second_value   = (v_old == second_value_);
        was_first_value    = (v_old == static_cast<T>(zero_one_value::Chunk<T>::first_value));
    }

    // Without counters (fast block compression) the body -- arithmetic_assign's,
    // the only caller -- marks from its own (old, new) pair, read in the same
    // session: no second read and second marking here.
    operation();

    if constexpr (chunk_config::use_compression_with_counter)
    {
        // Transitions OUT: position no longer holds the constant it held before.
        if (was_second_value && this->raw_elem() != second_value_)
        {
            chunk_.dec_nb_val_eq_second_value();
        }
        else if (was_first_value && this->raw_elem() != static_cast<T>(zero_one_value::Chunk<T>::first_value))
        {
            chunk_.dec_nb_val_eq_first_value();
        }

        // Transitions IN: position now holds a constant it did not hold before.
        // Guarded by !was_* to avoid double counting.
        if (!was_second_value && this->raw_elem() == second_value_)
        {
            chunk_.inc_nb_val_eq_second_value();
            if (chunk_.is_full_second_value(this->chunk_size()))
            {
                this->manage_full_second_value();
            }
        }
        else if (!was_first_value && this->raw_elem() == static_cast<T>(zero_one_value::Chunk<T>::first_value))
        {
            chunk_.inc_nb_val_eq_first_value();
            if (chunk_.is_full_first_value(this->chunk_size()))
            {
                this->manage_full_first_value();
            }
        }
    }
}

// ── execute_arithmetic_compound_operation ─────────────────────────────────
// Two entirely different implementations depending on compression mode, one
// per zero_one_value::ProxyFastCompPolicy specialisation.

template <typename Derived, typename T, typename ChunkT, bool Enabled>
template <typename Lambda>
void zero_one_value::ProxyFastCompPolicy<Derived, T, ChunkT, Enabled>::execute_simple_operation_async(auto*  ctx,
                                                                                                      Lambda operation)
{
    auto& chunk = ctx->chunk;

    const zero_one_value::Proxy proxy = Base::template make_proxy_with_info<Derived>(ctx);

    const bool is_val2 = proxy.is_cst_val2();
    const bool is_val1 = proxy.is_cst_val1();

    if constexpr (is_not_bool)
    {
        if (is_val2)
        {
            chunk.create_value(true, proxy.get_inf_array());
        }
        else if (is_val1)
        {
            chunk.create_value(false, proxy.get_inf_array());
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
void zero_one_value::ProxyFastCompPolicy<Derived, T, ChunkT, Enabled>::execute_arithmetic_compound_operation(
    this Derived& self,
    Op            op,
    const U&      operand,
    Lambda        operation)
{
    // Same rule as ensure_writable_storage(): this function may call
    // chunk.create_value() below (via the will_be_not_second_value branch)
    // without going through zero_one_value::Proxy<T>::create_value()'s self-correcting
    // wrapper -- pin the ambient size to THIS array's own geometry first.
    if constexpr (build_config::chunk_dynamic)
    {
        shrink_array::set_thread_chunk_size(self.chunk_size());
        shrink_array::set_thread_alloc_key(self.alloc_key());
    }

    using CalcType = std::common_type_t<T, U>;

    const InT<T> v_old = self.Base::read_values();

    // Perform math in high precision
    const CalcType v_new_precise = op(static_cast<CalcType>(v_old), static_cast<CalcType>(operand));

    if constexpr (is_not_bool)
    {
        self.second_value_ = self.make_second_value();
    }

    // Compare the PRECISE result with the second constant value
    const bool will_be_not_second_value =
        (v_new_precise != static_cast<CalcType>(self.second_value_)) && self.is_cst_val2();

    if (will_be_not_second_value)
    {
        self.Base::chunk_.create_value(true, self.zero_one_value::ProxySpec<T, Derived>::inf_array_);
        operation(op);
    }
    else if (self.Base2::is_not_cst_val2())
    {
        self.raw_elem_set(static_cast<T>(v_new_precise));
        self.mark_for_compression_if_needed_with_value(v_old, static_cast<T>(v_new_precise));
    }
}

template <typename Derived, typename T, typename ChunkT, bool Enabled>
template <typename Op, typename Lambda, typename U>
void zero_one_value::ProxyFastCompPolicy<Derived, T, ChunkT, Enabled>::execute_arithmetic_compound_operation_async(
    auto*    ctx,
    Op       op,
    const U& operand,
    Lambda   operation) const
{
    using CalcType                    = std::common_type_t<T, U>;
    const zero_one_value::Proxy proxy = Base::template make_proxy_with_info<Derived>(ctx);
    const InT<T>                v_old = read_values_async(ctx);

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
        T* values_ptr = ctx->chunk.values_load();
        elem_set(values_ptr, static_cast<std::size_t>(ctx->num), static_cast<T>(v_new_precise));
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
void zero_one_value::ProxyFastCompPolicy<Derived, T, ChunkT, false>::execute_arithmetic_compound_operation(
    this Derived& self,
    Op            op,
    const U&      operand,
    Lambda        operation)
{
    // Same rule as ensure_writable_storage(): see that function's comment.
    if constexpr (build_config::chunk_dynamic)
    {
        shrink_array::set_thread_chunk_size(self.chunk_size());
        shrink_array::set_thread_alloc_key(self.alloc_key());
    }

    using CalcType = std::common_type_t<T, BaseT<U>>;

    const bool was_not_written = !self.Base::chunk_.written(self.ind_);

    if (was_not_written)
    {
        const CalcType intermediate = op(static_cast<CalcType>(self.raw_elem()), static_cast<CalcType>(operand));
        // Built where it is used: one write of a struct for one value.
        const auto res = static_cast<T>(intermediate);

        const bool will_be_not_second_value = (res != self.second_value_) && self.is_cst_val2();

        if (will_be_not_second_value)
        {
            self.Base::chunk_.create_value(true, self.zero_one_value::ProxySpec<T, Derived>::inf_array_);
            if constexpr (chunk_config::use_compression_with_counter)
            {
                self.Base::chunk_.set_nb_val_eq_second_value(self.chunk_size() - 1);
            }
            operation(op, self.Base::values_);
            return;
        }
    }

    operation(op, self.Base::values_);
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
void zero_one_value::Proxy<T>::execute_inplace_operation(Lambda operation)
{
    // Same rule as ensure_writable_storage(): see that function's comment.
    if constexpr (build_config::chunk_dynamic)
    {
        shrink_array::set_thread_chunk_size(this->chunk_size());
        shrink_array::set_thread_alloc_key(this->alloc_key());
    }

    if constexpr (chunk_config::use_compression_with_counter && requires { this->set_info_second_value_ind(); })
    {
        this->set_info_second_value_ind();
    }

    const bool is_val2 = is_cst_val2();
    const bool is_val1 = Base::is_cst_val1();

    [[maybe_unused]] T v_old{};
    if constexpr (chunk_config::use_compression_without_counter)
    {
        v_old = read_values();
    }

    if constexpr (is_not_bool)
    {
        if (is_val2 || is_val1)
        {
            Base::chunk_.create_value(is_val2, inf_array_);
            if constexpr (chunk_config::use_compression_with_counter)
            {
                if (is_val2)
                {
                    chunk_.set_nb_val_eq_second_value(this->chunk_size());
                }
                else if (is_val1)
                {
                    chunk_.set_nb_val_eq_first_value(this->chunk_size());
                }
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
        was_second_value = (this->raw_elem() == second_value_);
        was_first_value  = (this->raw_elem() == static_cast<T>(zero_one_value::Chunk<T>::first_value));
    }

    elem_update(static_cast<T*>(Base::values_), static_cast<std::size_t>(num_), operation);

    if constexpr (chunk_config::use_compression_with_counter)
    {
        // Transitions OUT: the position no longer holds the constant it held before.
        if (was_second_value && this->raw_elem() != second_value_)
        {
            chunk_.dec_nb_val_eq_second_value();
        }
        else if (was_first_value && this->raw_elem() != static_cast<T>(zero_one_value::Chunk<T>::first_value))
        {
            chunk_.dec_nb_val_eq_first_value();
        }

        // Transitions IN: the position now holds a constant it did not hold before.
        // Test the actual stored value, NOT is_cst_val2(ind) (which checks the
        // storage pointer and is false right after create_value decompressed the
        // chunk, so it would miss the increment and desync the counter).
        if (!was_second_value && this->raw_elem() == second_value_)
        {
            chunk_.inc_nb_val_eq_second_value();
            if (chunk_.is_full_second_value(this->chunk_size()))
            {
                this->manage_full_second_value();
            }
        }
        else if (!was_first_value && this->raw_elem() == static_cast<T>(zero_one_value::Chunk<T>::first_value))
        {
            chunk_.inc_nb_val_eq_first_value();
            if (chunk_.is_full_first_value(this->chunk_size()))
            {
                this->manage_full_first_value();
            }
        }
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
void zero_one_value::ProxyFastCompPolicy<Derived, T, ChunkT, Enabled>::execute_inplace_operation_async(
    auto*  ctx,
    Lambda operation) const
{
    auto&     chunk = ctx->chunk;
    const int num   = ctx->num;

    const InT<T>                v_old = read_values_async(ctx);
    const zero_one_value::Proxy proxy = Base::template make_proxy_with_info<Derived>(ctx);

    const bool is_val1 = chunk.is_cst_val1();
    const bool is_val2 = proxy.is_cst_val2();

    if (is_val1 || is_val2)
    {
        if constexpr (is_not_bool)
        {
            const auto inf = zero_one_value::Proxy<T>::get_info_second_value(ctx->ptr_on_array);
            chunk.create_value(is_val2, inf);
        }
        else
        {
            chunk.create_value_bool(is_val2, get_thr_rank_fast());
        }
    }

    T* values = chunk.values_load();
    // raw-read: deferred replay, after materialisation -- flushes exclude the stealers
    elem_update(values, static_cast<std::size_t>(num), operation);

    // The value just written, read for the counters: nothing writes it
    // afterwards, so InT (a reference for a struct) rather than a copy.
    // raw-read: deferred replay, after materialisation -- flushes exclude the stealers
    const InT<T> v_new = elem_get(values, static_cast<std::size_t>(num));
    Base2::mark_for_compression_if_needed_with_value_async(proxy, v_old, v_new);
}

template <typename T>
template <typename U, typename Op>
void zero_one_value::Proxy<T>::apply_compound_operation([[maybe_unused]] Op op, InT<U> v) noexcept
{
    this->check_not_stale();
    // Cross-rank RMW materialisation: pull any foreign
    // deferred write to this offset to applied state before reading v_old, so the
    // RMW cannot clobber it. precise gate, outside begin_write. See
    // ProxyBase::arithmetic_assign for the full rationale.
    chunk_.reader_sync(num_);

    if constexpr (chunk_config::use_compression_with_counter)
    {
        BaseSpec::set_info_second_value_ind(); // Initialize second_value / ind — needed by both branches below

        const InT<T> v_old_snapshot = this->raw_elem();
        const bool   is_first_value = (v_old_snapshot == static_cast<T>(zero_one_value::Chunk<T>::first_value));

        if (!is_first_value) [[likely]]
        {
            if constexpr (build_config::do_check)
            {
                global_check_write_nan(v, num_);
            }

            const bool was_second_value  = (v_old_snapshot == second_value_);
            const bool was_second_value0 = was_second_value && Base2::is_not_cst_val2();

            execute_arithmetic_compound_operation(
                op, v,
                // v by reference (this call's parameter, alive for the call);
                // `current` InT: the right-hand side is evaluated in full
                // before the store, so a reference into the slot is read once,
                // before it is written.
                [this, &v](auto op_func, T* val_ptr)
                {
                    using LocalCalcType = std::common_type_t<T, BaseT<U>>;
                    const T current     = elem_get(val_ptr, static_cast<std::size_t>(num_));
                    elem_set(
                        val_ptr, static_cast<std::size_t>(num_),
                        static_cast<T>(op_func(static_cast<LocalCalcType>(current), static_cast<LocalCalcType>(v))));
                });

            // Read after the operation, compared before each branch's action
            // (a manage_full_* may fold the chunk, and nothing reads v_new
            // after it): InT, no copy of a struct.
            const InT<T> v_new = this->raw_elem();
            // Transition OUT of the second value first, independently of the
            // transition IN below (as execute_simple_operation does). Do not
            // chain it to the first-value branch: `second_value * 0` (or
            // `/ inf`) would count the new zero and keep the old second value
            // counted, and a later write would fold the chunk to cst2 with one
            // element still different (lost).
            if (was_second_value0 && v_new != second_value_)
            {
                chunk_.dec_nb_val_eq_second_value();
            }
            if (Base::is_not_cst_val1() && v_new == zero_one_value::Chunk<T>::first_value)
            {
                chunk_.inc_nb_val_eq_first_value();
                if (chunk_.is_full_first_value(this->chunk_size()))
                {
                    this->manage_full_first_value();
                }
            }
            else if (!was_second_value && v_new == second_value_)
            {
                chunk_.inc_nb_val_eq_second_value();
                if (chunk_.is_full_second_value(this->chunk_size()))
                {
                    this->manage_full_second_value();
                }
            }
        }
    }

    if constexpr (chunk_config::use_compression_without_counter)
    {
        // Outside the write session: under a ReadSection, or the `0 * x`
        // shortcut may act on a released, zero-filled buffer (see
        // ProxyBase::read_values_outside_session).
        const InT<T> v_old = Base::read_values_outside_session();
        if (v_old != zero_one_value::Chunk<T>::first_value)
        {
            Base::apply_atomic_op(op, v, v_old);
        }
    }
}

} // namespace shrink_array