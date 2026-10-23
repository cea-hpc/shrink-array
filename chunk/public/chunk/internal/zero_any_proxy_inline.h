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

#include <atomic>          // std::atomic<char> (the "queued" flags)
#include <source_location> // std::source_location
#include <thread>          // std::this_thread::yield (replay_retry)
#include <type_traits>     // std::remove_cvref_t
#include <utility>         // std::move, std::cmp_greater_equal
#include <vector>          // the per-chunk "queued" flags

#include "chunk/internal/zero_any_proxy.h"            // the declarations these bodies belong to
#include "chunk/internal/in_t.h"               // InT<T>
#include "chunk/zero_any_array.h"               // zero_any::Array (dependent calls the include tools cannot attribute)
#include "chunk/config.h"                      // build_config::any_check, build_config::chunk_dynamic
#include "chunk/PMP_CHUNK.h"                   // get_thr_rank_fast
#include "chunk/internal/arithmetic_context.h" // ArithmeticAffectationContext
#include "chunk/internal/array_chunk_for.h"    // ArrayChunkForT
#include "chunk/internal/concept_is_clean_assignement.h" // BaseT
#include "chunk/internal/deferred_op_pool.h"             // pool_new_ctx
#include "chunk/internal/bad_ptr_check.h"                // CHUNK_CHECK_VALUES_PTR (check builds)
#include "chunk/internal/invariant_check.h"              // CHUNK_INVARIANT_FAILED
#include "chunk/internal/chunk_stats.h"                  // increm_chunk_* (CHUNK_STATS)
#include "chunk/internal/relaxed_element.h"              // store_element_relaxed (the deferred replay)
#include "chunk/internal/chunk_size_policy.h"            // DynChunkSizeGuard, set_thread_chunk_size
#include "chunk/internal/read_reclaim.h" // read_reclaim::PassScope (perform_compression / compress in a region)
#include "chunk/internal/parallel_storage_init.h" // make_filled, resize_filled, for_each_chunk
#include "chunk/log.h"                            // CHUNK_LOG, CHUNK_LOG_ALWAYS

/**
 * @file zero_any_proxy_inline.h
 * @brief Out-of-line definitions of Proxy2A and Proxy2AFastCompPolicy.
 *
 * Chunk2A stores TWO constants: a global `first_value` and a per-chunk
 * `second_value`. A chunk is cst_val1, cst_val2, or materialised on the heap;
 * this file moves between those states without losing a concurrent write:
 * metadata arming (set_info_on_chunk() and siblings), compression driving
 * (trigger_compression_if_needed(), perform_compression()), materialisation
 * (ensure_writable_storage()), and the write bodies, each with an `_async`
 * twin replayed from a deferred op.
 *
 * The `_async` protocol: a replay runs long after its write was issued, so it
 * rebuilds a fresh proxy from the context, checks that `chunk.values` is a heap
 * buffer BEFORE storing and re-reads it AFTER -- a change means a compressor
 * swapped the buffer and the store was lost, so the whole body retries
 * (internal::replay_retry).
 *
 * Under dynamic geometry every materialisation first pins the per-thread chunk
 * size to the chunk being written: allocating at the flushing thread's ambient
 * size overflows the buffer.
 *
 * @see zero_any_proxy.h for the declarations, proxy_base_inline.h for the write engine
 *      these bodies are handed to, zero_any_array.h for the chunk itself.
 */

namespace shrink_array
{

namespace internal
{
/// Deferred-replay retries past which a check build fails and a release
/// build warns (once per process).
inline constexpr int replay_retry_cap = 1000;
/// Retries past which each further one yields the core first.
inline constexpr int replay_retry_spin = 64;

/**
 * @brief One more turn of a deferred-replay retry loop (the `_async` bodies
 *        below and lbd_equal_asynch).
 *
 * @details A retry means a compressor finished in between: progress was made,
 *          so the loop is NOT bounded in release (an abort would turn a
 *          contention spike into a lost run). Past replay_retry_spin the thread
 *          yields; at replay_retry_cap a check build fails and a release build
 *          warns once per process and keeps going.
 *
 * @param retry_count the loop's counter, starting at 0
 * @param where       the loop and path, for the message
 * @param chunk       chunk being written, for the message
 * @param num         element offset, for the message
 * @param loc         the caller (defaulted: the retry site, not this function)
 */
inline void replay_retry(int&                        retry_count,
                         const char*                 where,
                         const void*                 chunk,
                         int                         num,
                         const std::source_location& loc = std::source_location::current()) noexcept
{
    if (retry_count <= replay_retry_cap)
    {
        ++retry_count;
    }
    if (retry_count > replay_retry_cap)
    {
        if constexpr (build_config::any_check)
        {
            invariant_failed(loc, "{}: retry > {} on chunk={} num={} -- pathological contention?", where,
                             replay_retry_cap, chunk, num);
        }
        else
        {
            static std::atomic<bool> warned{false};
            if (!warned.exchange(true, std::memory_order_relaxed))
            {
                CHUNK_LOG(warn, drain,
                          "{}: deferred replay retried more than {} times on chunk={} num={} (still retrying; reported "
                          "once per process)",
                          where, replay_retry_cap, static_cast<const void*>(chunk), num);
            }
        }
    }
    if (retry_count > replay_retry_spin)
    {
        std::this_thread::yield();
    }
}
} // namespace internal

// Metadata arming. Order matters -- num_tab must be resolved before any pointer
// is computed, since every table is indexed by it. The assignment bodies
// (handle_*_assignment) arm only what the write engine has not.

// Full arming: resolves num_tab, then every cached pointer.
template <typename T>
void zero_any::Proxy<T>::set_info_on_chunk() noexcept
{
    set_num_tab();

    Base::set_ptr_is_inside_chunk_list_to_compress();

    ptr_type_cst_ = &tab_type_cst_val[num_tab_][num_chunk_];
    if constexpr (is_not_bool)
    {
        BaseSpec::ptr_second_value_ = BaseSpec::get_tab_second_value(num_tab_, num_chunk_);
    }
}

/**
 * @brief Arms ptr_second_value alone, from an already-valid num_tab.
 *
 * @pre num_tab and num_chunk are already resolved.
 * @note No-op for bool, whose specialisation keeps no per-chunk second value.
 */
template <typename T>
void zero_any::Proxy<T>::set_ptr_second_value() noexcept
{
    if constexpr (is_not_bool)
    {
        BaseSpec::ptr_second_value_ = BaseSpec::get_tab_second_value(num_tab_, num_chunk_);
    }
}

// Re-reads the slot through ptr_second_value; prefer the explicit-old-value
// variant below wherever the caller already knows it. bool carries no per-chunk
// second value, so it routes to the plain ProxyBase form.
template <typename T>
void zero_any::Proxy<T>::mark_for_compression_if_needed(InT<T> new_value, [[maybe_unused]] const T* vals) const
{
    if constexpr (is_not_bool)
    {
        // Called BEFORE the store (handle_no_bool_assignment): the memory check
        // must expect the old value -- see mark_for_compression_before_store.
        Base2::mark_for_compression_before_store(BaseSpec::ptr_second_value_.get(), new_value, zero_any::Chunk<T>::first_value,
                                                 vals);
    }
    else
    {
        Base2::mark_for_compression_if_needed(new_value);
    }
}

// Explicit-old-value variant: preferred, as it does not re-read the slot (which
// could race with another writer). bool routes to Proxy2Base's (old, new) form;
// this arm is not instantiated today, so a wrong qualifier would not be caught.
template <typename T>
void zero_any::Proxy<T>::mark_for_compression_if_needed_with_value(InT<T> old_value, InT<T> new_value) const
{
    if constexpr (is_not_bool)
    {
        Base2::mark_for_compression_if_needed(BaseSpec::ptr_second_value_.get(), old_value, new_value,
                                              zero_any::Chunk<T>::first_value);
    }
    else
    {
        Base2::mark_for_compression_if_needed(old_value, new_value);
    }
}

// Grow the three per-tab tables in lockstep. They are indexed by the same
// num_tab, so they must always share the same outer length.
template <typename T>
void zero_any::Proxy<T>::resize_tab_is_int(int nb_tab)
{
    BaseSpec::resize(nb_tab);
    tab_type_cst_val.resize(nb_tab);
    Base::tab_is_inside_chunk_list_to_compress.resize(nb_tab);
}

// Grow one tab's per-chunk tables. BaseSpec::resize() and tab_type_cst_val keep
// existing entries, but tab_is_inside_chunk_list_to_compress is REALLOCATED:
// every "queued for compression" flag goes back to 0. Safe only if no chunk of
// this tab is queued -- a queued chunk with a cleared flag could be queued twice
// (pointing into the chunk_list just reallocated).
// ArrayChunk2Base::resize_common() guarantees it by flushing the deferred ops
// (whose replay can queue chunks) BEFORE the compression pass that empties every
// list. Do not swap those two steps. Asserted below (flag 1 <=> entry in a list).
template <typename T>
void zero_any::Proxy<T>::resize_tab_is_int(int num_tab, int nb_chunk_old, int nb_chunk)
{
    BaseSpec::resize(num_tab, nb_chunk_old, nb_chunk);
    // Check builds: no chunk of this tab may still be queued (see above).
    if constexpr (build_config::any_check)
    {
        const auto& row = Base::tab_is_inside_chunk_list_to_compress[static_cast<std::size_t>(num_tab)];
        for (std::size_t j = 0; j < row.size(); ++j)
        {
            if (row[j].load(std::memory_order_relaxed) != 0) [[unlikely]]
            {
                CHUNK_INVARIANT_FAILED("resize_tab_is_int: tab {} chunk {} still queued for compression while its "
                                       "flag row is reallocated ({} -> {} chunks)",
                                       num_tab, j, nb_chunk_old, nb_chunk);
            }
        }
    }
    // Every flag 0; the new type-of-constant entries 0. Both constructed by a
    // team for a large array (parallel_storage_init).
    Base::tab_is_inside_chunk_list_to_compress[static_cast<std::size_t>(num_tab)] =
        parallel_storage_init::make_filled<std::atomic<char>>(static_cast<std::size_t>(nb_chunk), char{0});
    parallel_storage_init::resize_filled(tab_type_cst_val[num_tab], static_cast<std::size_t>(nb_chunk), char{0});
}

/**
 * @brief Flags this thread's candidate list as worth a compression pass.
 *
 * Called after every fast-path write. It does NOT compress: past the threshold
 * it marks the list "dangerous", so the next write takes the slow path into
 * perform_compression(), which clears the mark. OpenMP-only.
 *
 * @param thr_rank Rank whose candidate list is examined.
 */
template <typename T>
void zero_any::Proxy<T>::trigger_compression_if_needed(int thr_rank)
{
    const auto& vec_to_compress_potantially = ManagerChunk2A::get_chunk2a_to_compress_potentially(thr_rank);

    if constexpr (build_config::threaded)
    {
        const int szm = ManagerChunk2A::get_size_max_number_of_chunk_to_check_compress();

        const int sz = static_cast<int>(vec_to_compress_potantially.size());

        if (Base::should_trigger_compression(sz, szm))
        {
            CompressionManagerType::mark_list_dangerous_for_thread(thr_rank);

            if constexpr (chunk_config::verbose)
            {
                CHUNK_LOG_ALWAYS(compress, "candidate list became dangerous: size={}", sz);
            }
        }
    }
}

/**
 * @brief Compresses this thread's candidate chunks, if there are enough of them.
 *
 * The write engine's slow path, and the sequential build's only compression
 * driver. Self-gating: below half the configured maximum it does nothing.
 * Above, it drains the thread's deferred ops first (they must be applied
 * before their chunks are compressed), then runs the block compression.
 */
template <typename T>
void zero_any::Proxy<T>::perform_compression() noexcept
{
    // Runs inside a region, next to pure readers: the buffers this pass
    // releases wait for one grace period (read_reclaim.h).
    const read_reclaim::PassScope pass_scope;
    const int                     szm = ManagerChunk2A::get_size_max_number_of_chunk_to_check_compress();

    const int thr_rank = get_thr_rank_fast();

    const auto& vec_to_compress_potantially = ManagerChunk2A::get_chunk2a_to_compress_potentially(thr_rank);

    const int sz_before = static_cast<int>(vec_to_compress_potantially.size());

    const int sz_threshold = szm / 2;

    if (sz_before > sz_threshold)
    {
        // Serial builds defer nothing, and flush_deferred exists only in the
        // OpenMP policy specialisation.
        if constexpr (build_config::threaded)
        {
            static_cast<void>(CompressionManagerType::flush_deferred(thr_rank));
        }

        // free_values picks its allocator from thread_chunk_size, which an op on
        // a different-size array may have left stale: pin it to THIS chunk, or
        // the block is freed into the wrong-size slab (fast-alloc).
        if constexpr (build_config::chunk_dynamic)
        {
            shrink_array::set_thread_chunk_size(this->chunk_size());
            shrink_array::set_thread_alloc_key(this->alloc_key());
        }
        zero_any::Chunk<T>::make_block_compression(thr_rank, this);
    }
    // Cleared whether or not there was work. Do not move it inside the `if`:
    // a list drained meanwhile would keep the rank on the slow path for every
    // write (see mark_list_dangerous_for_thread). OpenMP only, as the trigger.
    if constexpr (build_config::threaded)
    {
        CompressionManagerType::mark_list_safe_for_thread(thr_rank);
    }
}

// Materialises a constant chunk onto the heap. is_cst_val2() is sampled BEFORE
// set_info_on_chunk(); the arming sits between the sample and create_value().
template <typename T>
void zero_any::Proxy<T>::ensure_writable_storage()
{
    if constexpr (build_config::chunk_dynamic)
    {
        if constexpr (build_config::any_check)
        {
            const unsigned int frozen = this->chunk_size_frozen_;
            const unsigned int live =
                this->ptr_on_array_ ? static_cast<const ArrayChunkForT<ChunkType>*>(this->ptr_on_array_)->chunk_size()
                                    : 0;
            const unsigned int used = this->chunk_size();
            if (used != live || frozen != 0U)
            {
                CHUNK_LOG(debug, rechunk, "ensure_writable_storage: used={} live={} frozen={}", used, live, frozen);
            }
        }
        shrink_array::set_thread_chunk_size(this->chunk_size());
        shrink_array::set_thread_alloc_key(this->alloc_key());
    }
    const bool is_val2 = is_cst_val2();

    // The cached per-chunk pointers are armed here, never earlier. Under fast
    // block compression the write engine already armed num_tab and the
    // queued-flag pointer (write_with_optional_lock_info's lbd_info, as
    // handle_no_bool_assignment() relies on): only the type-of-constant slot
    // and the second value are left -- set_info_on_chunk() redid both lookups.
    if constexpr (chunk_config::use_fast_block_compression)
    {
        ptr_type_cst_ = &tab_type_cst_val[num_tab_][num_chunk_];
        set_ptr_second_value();
    }
    else
    {
        set_info_on_chunk();
    }

    // Allocate from THIS array's chunk-size pool: the per-thread size may be
    // stale (left by an op on another array), and a smaller buffer written at
    // this array's offsets overflows. The in-place path reaches create_value
    // only through here; Proxy2A::create_value (zero_any_proxy.h) does the same on
    // the direct-assignment path.
    if constexpr (build_config::chunk_dynamic)
    {
        shrink_array::set_thread_chunk_size(this->chunk_size());
        shrink_array::set_thread_alloc_key(this->alloc_key());
    }

    if constexpr (is_not_bool)
    {
        if (is_val2)
        {
            chunk_.create_value(true, BaseSpec::ptr_second_value_);
        }
        else if (Base::is_cst_val1())
        {
            chunk_.create_value(false, BaseSpec::ptr_second_value_);
        }
    }
    else
    {
        if (is_val2)
        {
            chunk_.create_value_bool(true);
        }
        else if (Base::is_cst_val1())
        {
            chunk_.create_value_bool(false);
        }
    }
}

/**
 * @brief Materialises a constant chunk from a replay context.
 *
 * Async twin of Proxy2A::ensure_writable_storage(): the state is read from a
 * proxy rebuilt by the caller rather than from `*this`, and no metadata arming
 * happens here — make_proxy_with_info() already did it.
 *
 * @param proxy Proxy rebuilt from the deferred-operation context.
 * @note Reached only from the async materialisers, never from a deferred
 *       replay, so `proxy.chunk_size()` is the live geometry here.
 */
template <typename Derived, typename T, typename ChunkT, bool Enabled>
void zero_any::ProxyFastCompPolicy<Derived, T, ChunkT, Enabled>::ensure_writable_storage_async(const Derived& proxy) const
{
    const bool is_val2 = proxy.is_cst_val2();

    // THIS array's chunk-size pool: see ensure_writable_storage() above.
    if constexpr (build_config::chunk_dynamic)
    {
        shrink_array::set_thread_chunk_size(proxy.chunk_size());
        shrink_array::set_thread_alloc_key(proxy.alloc_key());
    }

    if constexpr (is_not_bool)
    {
        if (is_val2)
        {
            this->chunk_.create_value(true, proxy.ptr_second_value_);
        }
        else if (Base::is_cst_val1())
        {
            this->chunk_.create_value(false, proxy.ptr_second_value_);
        }
    }
    else
    {
        if (is_val2)
        {
            this->chunk.create_value_bool(true);
        }
        else if (Base::is_cst_val1())
        {
            this->chunk.create_value_bool(false);
        }
    }
}

/**
 * @brief Stores a value, materialising the chunk only if it has to.
 *
 * Three outcomes, cheapest first:
 *  -# the second value onto a chunk that already reads it back: no-op;
 *  -# from cst_val1 to the second value: materialise, then store;
 *  -# otherwise materialise if still constant, then store.
 *
 * @param vals The chunk's values pointer, loaded by the caller inside the write
 *        session; every state test reads it instead of reloading `values`.
 * @param v Value to store.
 * @pre The metadata pointers this reads are armed (see the handle_*_assignment
 *      callers, which arm exactly what their branch needs).
 */
template <typename T>
void zero_any::Proxy<T>::perform_core_assignment(T* vals, InT<T> v) const
{
    // The one reload, after a materialisation: a concurrent writer may have won
    // the CAS (ours went back to the pool). Stable until the session ends.
    const auto materialise = [this, &vals]
    {
        if constexpr (is_not_bool)
        {
            create_value(BaseSpec::ptr_second_value_);
        }
        else
        {
            create_value_bool();
        }
        vals = chunk_.values_load();
    };

    const bool is_cst2 = chunk_.is_cst_val2_ptr_of(vals);

    if constexpr (is_not_bool)
    {
        if (v == static_cast<T>(BaseSpec::ptr_second_value_.get()))
        {
            if (is_cst2 || Base::read_values_at(vals) == BaseSpec::ptr_second_value_.get())
            {
                return;
            }
            if (Base::is_cst_val1(vals))
            {
                materialise();

                Base::store_value_at(vals, BaseSpec::ptr_second_value_.get());

                return;
            }
        }
    }
    else if (v)
    {
        if (is_cst2 || Base::read_values_at(vals))
        {
            return;
        }
        if (Base::is_cst_val1(vals))
        {
            materialise();

            Base::store_value_at(vals, true);

            return;
        }
    }

    if constexpr (chunk_config::stats)
    {
        increm_chunk_write<ChunkType>(v);
    }

    if (is_cst2 || Base::is_cst_val1(vals))
    {
        materialise();
    }

    Base::store_value_at(vals, v);

    if constexpr (build_config::do_check)
    {
        global_check_write_nan(v, num_);
    }
}
// ========================================================================
// ASSIGNMENT OPERATOR
// ========================================================================

/**
 * @brief Assignment body for non-bool element types.
 *
 * Arms the metadata the write engine has not, marks for compression, stores.
 * The second value is resolved first, for the cst_val2 early-out.
 *
 * @param v Value to assign.
 */
template <typename T>
void zero_any::Proxy<T>::handle_no_bool_assignment(InT<T> v)
{
    if constexpr (is_not_bool)
    {
        // ONE load of the values pointer, as in Proxy::assign_impl: inside the
        // write session no compressor can change it, and no writer once it is
        // not a constant. A concurrent materialisation is what
        // perform_core_assignment() reloads after.
        T* const vals = chunk_.values_load();

        // With fast block compression the engine already armed num_tab and the
        // queued-flag pointer (write_with_optional_lock_with_set_ptr).
        if constexpr (!chunk_config::use_fast_block_compression)
        {
            set_num_tab();
            Base::set_ptr_is_inside_chunk_list_to_compress();
        }
        set_ptr_second_value();

        // Second value onto a cst_val2 chunk: nothing to do.
        if (chunk_.is_cst_val2_ptr_of(vals) && v == static_cast<T>(BaseSpec::ptr_second_value_.get()))
        {
            return;
        }

        ptr_type_cst_ = &tab_type_cst_val[num_tab_][num_chunk_];

        mark_for_compression_if_needed(v, vals);

        perform_core_assignment(vals, v);
    }
}

/**
 * @brief Assignment body for `T = bool`.
 *
 * As handle_no_bool_assignment(); for bool, cst_val2 means "all true".
 *
 * @param v Value to assign.
 */
template <typename T>
void zero_any::Proxy<T>::handle_bool_assignment(InT<T> v)
{
    if constexpr (is_bool)
    {
        // One load of the values pointer, as in handle_no_bool_assignment().
        T* const vals = chunk_.values_load();

        // Armed by the write engine with the fast block compression: see
        // handle_no_bool_assignment().
        if constexpr (!chunk_config::use_fast_block_compression)
        {
            set_num_tab();
            Base::set_ptr_is_inside_chunk_list_to_compress();
        }

        if (chunk_.is_cst_val2_ptr_of(vals) && v)
        {
            return;
        }

        ptr_type_cst_ = &tab_type_cst_val[num_tab_][num_chunk_];

        mark_for_compression_if_needed(v, vals);

        perform_core_assignment(vals, v);
    }
}

// Entry point of every assignment: early-out on first_value onto cst_val1, the
// body through the write engine, a deferred op when the engine refuses. The
// deferred body does not reuse handle_*_assignment: a replay cannot trust the
// cached metadata, so it works off a fresh values pointer under the retry protocol.
template <typename T>
template <typename U>
T zero_any::Proxy<T>::assign_impl(InT<U> v) noexcept
{
    this->check_not_stale();
    if (Base::is_cst_val1_and_v_equal_first_value(v))
    {
        return v;
    }

    [[maybe_unused]] bool need_async = false;

    if constexpr (is_not_bool)
    {
        // By reference: v outlives the synchronous call; the deferred path copies it.
        const auto assignment_body = [this, &v] { handle_no_bool_assignment(v); };

        need_async = Base2::write_with_optional_lock_with_set_ptr(assignment_body);
    }
    else
    {
        const auto assignment_body_bool = [this, &v] { handle_bool_assignment(v); };

        need_async = Base2::write_with_optional_lock_with_set_ptr(assignment_body_bool);
    }

    if constexpr (build_config::threaded)
    {
        if (need_async)
        {
            // The context stores the operand by value: U itself, or the type
            // behind the large-operand operator='s `const U&`.
            using CtxType = ArithmeticAffectationContext<T, std::remove_cvref_t<U>, ChunkType, zero_any::Proxy<T>>;

            auto* ctx = pool_new_ctx<CtxType>(*this, v, ptr_on_array_);

            auto lbd_equal_asynch = [ctx] // the operand lives in the context: no second copy in the closure
            {
                auto&      chunk = ctx->chunk;
                const int  num   = ctx->num;
                const auto proxy = Base::template make_proxy_with_info<Proxy>(ctx);

                // The flusher's ambient chunk size is unrelated to THIS op's chunk:
                // pin the frozen per-op geometry, restored on exit for the
                // flusher's own in-flight write.
                [[maybe_unused]] const DynChunkSizeGuard sz_guard{proxy.chunk_size()};

                int retry_count = 0;

                while (true)
                {
                    const bool is_cst1 = proxy.is_cst_val1();
                    const bool is_cst2 = proxy.is_cst_val2();

                    if (is_cst1 || is_cst2)
                    {
                        if constexpr (is_not_bool)
                        {
                            chunk.create_value(is_cst2, proxy.ptr_second_value_);
                        }
                        else
                        {
                            chunk.create_value_bool(is_cst2);
                        }
                    }

                    T* values_before = chunk.values_load();

                    // Pre-store guard: ensure values_before is a heap buffer.
                    if (chunk.is_constant_ptr(values_before)) [[unlikely]]
                    {
                        internal::replay_retry(retry_count, "lbd_equal_asynch (pre-store)",
                                               static_cast<const void*>(&chunk), num);
                        continue;
                    }

                    if constexpr (build_config::any_check)
                    {
                        // Catch an out-of-range index BEFORE it writes. Compare against
                        // proxy.chunk_size() (frozen at enqueue), NOT the static
                        // static_chunk_size, which is only the default in a dynamic build.
                        const unsigned int csz_guard = proxy.chunk_size();
                        if (std::cmp_greater_equal(num, csz_guard))
                        {
                            CHUNK_INVARIANT_FAILED("[deferred2A write] num OUT OF RANGE "
                                                   "(num={}, chunk_size={}, chunk={})",
                                                   num, csz_guard, static_cast<const void*>(&chunk));
                        }

                        CHUNK_CHECK_VALUES_PTR_C(values_before, "deferred2A.pre_store", &chunk);

                        // Narrow-race re-check: a chunk (re)compressed since
                        // values_load() goes back through the retry loop instead of
                        // being written blind.
                        if (proxy.is_cst_val1() || proxy.is_cst_val2()) [[unlikely]]
                        {
                            if (++retry_count > internal::replay_retry_cap)
                            {
                                CHUNK_INVARIANT_FAILED("[deferred2A write] chunk keeps reverting to cst before "
                                                       "store (chunk={}, num={}, values_before={}, cur_cst_val1={}, "
                                                       "cur_values={})",
                                                       static_cast<const void*>(&chunk), num,
                                                       static_cast<const void*>(values_before),
                                                       static_cast<const void*>(chunk.get_cst_val1()),
                                                       static_cast<const void*>(chunk.values_load()));
                            }
                            continue;
                        }
                    }

                    // raw-read: deferred replay, after materialisation -- flushes exclude the stealers.
                    // A relaxed atomic store where Chunk2A's element stores must be one
                    // (relaxed_element.h): other writers' scans (may_become_uniform)
                    // read this element meanwhile -- as the compound ops' replay does.
                    if constexpr (internal::relaxed_element_stores_v<zero_any::Chunk<T>, T>)
                    {
                        internal::store_element_relaxed(values_before + num, static_cast<T>(ctx->operand));
                    }
                    else
                    {
                        // raw-read: deferred replay, after materialisation (as above)
                        values_before[num] = static_cast<T>(ctx->operand);
                    }

                    // Post-store confirmation.
                    const T* values_after = chunk.values_load();
                    if (values_after != values_before) [[unlikely]]
                    {
                        internal::replay_retry(retry_count, "lbd_equal_asynch (post-store)",
                                               static_cast<const void*>(&chunk), num);
                        continue;
                    }

                    // Diagnostic checks (any_check only).
                    if constexpr (build_config::any_check)
                    {
                        if (proxy.is_cst_val1())
                        {
                            CHUNK_INVARIANT_FAILED("lbd_equal_asynch: is_cst_val1 TRUE after successful store! "
                                                   "chunk={} num={} values_before={} values_after={}",
                                                   static_cast<const void*>(&chunk), num,
                                                   static_cast<const void*>(values_before),
                                                   static_cast<const void*>(values_after));
                        }
                        if (proxy.is_cst_val2())
                        {
                            CHUNK_INVARIANT_FAILED("lbd_equal_asynch: is_cst_val2 TRUE after successful store! "
                                                   "chunk={} num={} values_before={} values_after={}",
                                                   static_cast<const void*>(&chunk), num,
                                                   static_cast<const void*>(values_before),
                                                   static_cast<const void*>(values_after));
                        }
                        // raw-read: deferred replay, after materialisation -- flushes exclude the stealers
                        const InT<T> readback = values_after[num];
                        const auto   expected = static_cast<T>(ctx->operand);
                        if (readback != expected)
                        {
                            CHUNK_INVARIANT_FAILED("lbd_equal_asynch: readback mismatch after store! "
                                                   "chunk={} num={} expected={} readback={}",
                                                   static_cast<const void*>(&chunk), num, expected, readback);
                        }
                    }

                    break;
                }
            };
            Base::enqueue_deferred_write(ctx, std::move(lbd_equal_asynch));
        }
    }

    return v;
}

/**
 * @brief Runs a write body on a materialised chunk, bracketed by compression marking.
 *
 * Materialise, snapshot the old value, run the body (which does its own
 * store), then re-mark for compression from the before/after pair.
 *
 * @tparam ModifyFn Callable performing the store.
 * @param modify_fn Write body.
 */
template <typename T>
template <typename ModifyFn>
void zero_any::Proxy<T>::execute_simple_operation(ModifyFn modify_fn)
{
    ensure_writable_storage();

    // Under fast block compression the body (arithmetic_assign's, the only
    // caller) marks from its own (old, new) pair, read in the same session:
    // marking here as well read the element twice more and ran the same test
    // a second time.
    if constexpr (chunk_config::use_fast_block_compression)
    {
        modify_fn();
    }
    else
    {
        const InT<T> v_old = read_values();

        modify_fn();

        mark_for_compression_if_needed_with_value(v_old, read_values());
    }
}

/**
 * @brief Replay twin of execute_simple_operation().
 *
 * The file's retry protocol; a failed check restarts the whole body, proxy
 * included, since a compressor may have changed the chunk state, not just its buffer.
 *
 * @tparam Lambda Callable taking the values pointer to write through.
 * @param ctx       Deferred-operation context being replayed.
 * @param operation Write body.
 */
template <typename Derived, typename T, typename ChunkT, bool Enabled>
template <typename Lambda>
void zero_any::ProxyFastCompPolicy<Derived, T, ChunkT, Enabled>::execute_simple_operation_async(this Derived& self,
                                                                                        auto*         ctx,
                                                                                        Lambda        operation)
{
    int retry_count = 0;

    while (true)
    {
        const zero_any::Proxy proxy = Base::template make_proxy_with_info<Derived>(ctx);
        self.ensure_writable_storage_async(proxy);

        const InT<T> v_old = Base::read_values_async(ctx);

        T* values_before = ctx->chunk.values_load();

        if (ctx->chunk.is_constant_ptr(values_before)) [[unlikely]]
        {
            internal::replay_retry(retry_count, "execute_simple_operation_async (pre-store)",
                                   static_cast<const void*>(&ctx->chunk), ctx->num);
            continue;
        }

        operation(values_before);

        const T* values_after = ctx->chunk.values_load();
        if (values_after != values_before) [[unlikely]]
        {
            internal::replay_retry(retry_count, "execute_simple_operation_async (post-store)",
                                   static_cast<const void*>(&ctx->chunk), ctx->num);
            continue;
        }

        Base2::mark_for_compression_if_needed_with_value_async(proxy, v_old, self.read_values());
        break;
    }
}

/**
 * @brief Applies a compound arithmetic operation, materialising only when the result leaves the constant.
 *
 * Computes in `std::common_type_t<T, U>`, truncates to T, then:
 *  - cst_val2 chunk, result != second value: materialise, @p modify_fn stores;
 *  - materialised chunk: store and re-mark for compression;
 *  - cst_val2 chunk, result == second value: nothing (a compressed chunk whose
 *    value does not change must not be materialised).
 *
 * @tparam Op       Binary operation type.
 * @tparam ModifyFn Callable storing the result on the materialised path.
 * @tparam U        Operand type.
 * @param op        Operation to apply.
 * @param v         Operand.
 * @param modify_fn Store body, invoked as `modify_fn(op)`.
 */
template <typename T>
template <typename Op, typename ModifyFn, typename U>
void zero_any::Proxy<T>::execute_arithmetic_compound_operation(Op op, const U& v, ModifyFn modify_fn)
{
    using CalcType = std::common_type_t<T, BaseT<U>>;

    if constexpr (build_config::chunk_dynamic)
    {
        shrink_array::set_thread_chunk_size(this->chunk_size());
        shrink_array::set_thread_alloc_key(this->alloc_key());
    }

    const InT<T> v_old = read_values();

    const CalcType v_new_precise   = op(static_cast<CalcType>(v_old), static_cast<CalcType>(v));
    const auto     v_new_truncated = static_cast<T>(v_new_precise);

    const bool cst_val2 = is_cst_val2();

    if constexpr (is_not_bool)
    {
        const bool will_be_not_second_value = (v_new_truncated != BaseSpec::ptr_second_value_.get()) && cst_val2;

        if (will_be_not_second_value)
        {
            chunk_.create_value(true, BaseSpec::ptr_second_value_);

            modify_fn(op);
        }
        else if (Base2::is_not_cst_val2())
        {
            Base::store_value(v_new_truncated);

            mark_for_compression_if_needed_with_value(v_old, v_new_truncated);
        }
    }
    else
    {
        // bool: cst_val2 means "all true".
        const bool will_be_not_second_value = !v_new_truncated && cst_val2;

        if (will_be_not_second_value)
        {
            chunk_.create_value_bool(true);
            modify_fn(op);
        }
        else if (Base::is_not_cst_val2())
        {
            chunk_.assert_writable_buffer(chunk_.values_load());

            Base::store_value(v_new_truncated);
            mark_for_compression_if_needed_with_value(v_old, v_new_truncated);
        }
    }
}

/**
 * @brief Replay twin of execute_arithmetic_compound_operation().
 *
 * Same three-way branch, each arm under the retry protocol. The no-op arm is
 * re-tested before exiting: the state it was decided on may have changed.
 *
 * @tparam Op  Binary operation type.
 * @tparam Body Callable storing the result on the materialised path.
 * @tparam U   Operand type.
 * @param ctx Deferred-operation context being replayed.
 * @param op  Operation to apply.
 * @param v   Operand.
 * @param lbd Store body, invoked as `lbd(op)`.
 */
template <typename Derived, typename T, typename ChunkT, bool Enabled>
template <typename Op, typename Body, typename U>
void zero_any::ProxyFastCompPolicy<Derived, T, ChunkT, Enabled>::execute_arithmetic_compound_operation_async(auto*    ctx,
                                                                                                     Op       op,
                                                                                                     const U& v,
                                                                                                     Body     lbd) const
{
    using CalcType = std::common_type_t<T, BaseT<U>>;

    int retry_count = 0;

    while (true)
    {
        // Rebuilt each iteration: it caches state that must be fresh.
        const zero_any::Proxy proxy = Base::template make_proxy_with_info<Derived>(ctx);

        // create_value below must allocate at THIS op's frozen geometry, not the
        // stale per-thread size: otherwise the store overflows the buffer (heap
        // corruption detected only at a later free).
        if constexpr (build_config::chunk_dynamic)
        {
            shrink_array::set_thread_chunk_size(proxy.chunk_size());
            shrink_array::set_thread_alloc_key(proxy.alloc_key());
        }

        const InT<T>   v_old           = Base::read_values_async(ctx);
        const CalcType v_new_precise   = op(static_cast<CalcType>(v_old), static_cast<CalcType>(v));
        const auto     v_new_truncated = static_cast<T>(v_new_precise);

        const bool cst_val2 = proxy.is_cst_val2();

        if constexpr (is_not_bool)
        {
            const bool will_be_not_second_value = (v_new_truncated != proxy.ptr_second_value_.get()) && cst_val2;

            if (will_be_not_second_value)
            {
                ctx->chunk.create_value(true, proxy.ptr_second_value_);

                const T* values_before = ctx->chunk.values_load();

                if (ctx->chunk.is_constant_ptr(values_before)) [[unlikely]]
                {
                    internal::replay_retry(retry_count, "execute_arithmetic_compound_operation_async (decompress path)",
                                           static_cast<const void*>(&ctx->chunk), ctx->num);
                    continue;
                }

                lbd(op);

                const T* values_after = ctx->chunk.values_load();
                if (values_after != values_before) [[unlikely]]
                {
                    internal::replay_retry(retry_count,
                                           "execute_arithmetic_compound_operation_async (decompress path post-store)",
                                           static_cast<const void*>(&ctx->chunk), ctx->num);
                    continue;
                }

                break;
            }
            if (proxy.is_not_cst_val2())
            {
                const T* values_before = ctx->chunk.values_load();

                if (ctx->chunk.is_constant_ptr(values_before)) [[unlikely]]
                {
                    internal::replay_retry(retry_count, "execute_arithmetic_compound_operation_async (heap path)",
                                           static_cast<const void*>(&ctx->chunk), ctx->num);
                    continue;
                }

                Base::store_value_async(ctx, v_new_truncated);

                const T* values_after = ctx->chunk.values_load();
                if (values_after != values_before) [[unlikely]]
                {
                    internal::replay_retry(retry_count,
                                           "execute_arithmetic_compound_operation_async (heap path post-store)",
                                           static_cast<const void*>(&ctx->chunk), ctx->num);
                    continue;
                }

                proxy.mark_for_compression_if_needed_with_value(v_old, v_new_truncated);
                break;
            }
            {
                // No-op arm (result == second value on cst_val2): re-test before exiting.
                if (proxy.is_cst_val2() && v_new_truncated == proxy.ptr_second_value_.get())
                {
                    break;
                }
                // State changed between decision and now: retry.
                internal::replay_retry(retry_count, "execute_arithmetic_compound_operation_async (no-op path)",
                                       static_cast<const void*>(&ctx->chunk), ctx->num);
            }
        }
        else
        {
            // bool: the same protocol, second value = true. Unreachable today (no
            // compound operator on a bool proxy); kept complete so a future one
            // does not skip the protocol.
            const bool will_be_not_second_value = !v_new_truncated && cst_val2;

            if (will_be_not_second_value)
            {
                ctx->chunk.create_value_bool(true);

                const T* values_before = ctx->chunk.values_load();
                if (ctx->chunk.is_constant_ptr(values_before)) [[unlikely]]
                {
                    internal::replay_retry(retry_count,
                                           "execute_arithmetic_compound_operation_async (bool, decompress path)",
                                           static_cast<const void*>(&ctx->chunk), ctx->num);
                    continue;
                }

                lbd(op);

                if (ctx->chunk.values_load() != values_before) [[unlikely]]
                {
                    internal::replay_retry(
                        retry_count, "execute_arithmetic_compound_operation_async (bool, decompress path post-store)",
                        static_cast<const void*>(&ctx->chunk), ctx->num);
                    continue;
                }
                break;
            }
            if (proxy.is_not_cst_val2())
            {
                const T* values_before = ctx->chunk.values_load();
                if (ctx->chunk.is_constant_ptr(values_before)) [[unlikely]]
                {
                    internal::replay_retry(retry_count, "execute_arithmetic_compound_operation_async (bool, heap path)",
                                           static_cast<const void*>(&ctx->chunk), ctx->num);
                    continue;
                }

                Base::store_value_async(ctx, v_new_truncated);

                if (ctx->chunk.values_load() != values_before) [[unlikely]]
                {
                    internal::replay_retry(retry_count,
                                           "execute_arithmetic_compound_operation_async (bool, heap path post-store)",
                                           static_cast<const void*>(&ctx->chunk), ctx->num);
                    continue;
                }

                proxy.mark_for_compression_if_needed_with_value(v_old, v_new_truncated);
                break;
            }
            if (proxy.is_cst_val2() && v_new_truncated)
            {
                break;
            }
            // State changed between decision and now: retry.
            internal::replay_retry(retry_count, "execute_arithmetic_compound_operation_async (bool, no-op path)",
                                   static_cast<const void*>(&ctx->chunk), ctx->num);
        }
    }
}
/**
 * @brief Runs an in-place write body (bitwise, ++/--) on a materialised chunk.
 *
 * Like execute_simple_operation(), but the body writes through the values array.
 *
 * @note The old value is snapshotted BEFORE materialising here, AFTER in
 *       execute_simple_operation(). Same value either way, but do not assume
 *       the two wrappers order these steps alike.
 *
 * @tparam ModifyFn Callable taking the values array.
 * @param modify_fn Write body.
 */
template <typename T>
template <typename ModifyFn>
void zero_any::Proxy<T>::execute_inplace_operation(ModifyFn modify_fn)
{
    const InT<T> v_old = read_values();

    ensure_writable_storage();

    internal::modify_element<zero_any::Chunk<T>, T>(values_, num_, modify_fn);

    mark_for_compression_if_needed_with_value(v_old, read_values());
}

// Replay twin of execute_inplace_operation(), under the retry protocol. Check
// builds assert the chunk did not go back to a constant during the store.
template <typename Derived, typename T, typename ChunkT, bool Enabled>
template <typename Lambda>
void zero_any::ProxyFastCompPolicy<Derived, T, ChunkT, Enabled>::execute_inplace_operation_async(this const Derived& self,
                                                                                         auto*               ctx,
                                                                                         Lambda              operation)
{
    int retry_count = 0;

    while (true)
    {
        // Best-effort; re-checked by the post-store test.
        const InT<T> v_old = Base::read_values_async(ctx);

        const zero_any::Proxy proxy = Base::template make_proxy_with_info<Derived>(ctx);

        self.ensure_writable_storage_async(proxy);

        T* values_before = ctx->chunk.values_load();

        if (ctx->chunk.is_constant_ptr(values_before)) [[unlikely]]
        {
            internal::replay_retry(retry_count, "execute_inplace_operation_async (pre-store)",
                                   static_cast<const void*>(&ctx->chunk), ctx->num);
            continue;
        }

        internal::modify_element<ChunkT, T>(values_before, ctx->num, operation);

        const T* values_after = ctx->chunk.values_load();
        if (values_after != values_before) [[unlikely]]
        {
            internal::replay_retry(retry_count, "execute_inplace_operation_async (post-store)",
                                   static_cast<const void*>(&ctx->chunk), ctx->num);
            continue;
        }

        if constexpr (build_config::any_check)
        {
            if (proxy.is_cst_val1())
            {
                CHUNK_INVARIANT_FAILED("execute_inplace_operation_async: is_cst_val1 TRUE after store! "
                                       "chunk={} num={}",
                                       static_cast<const void*>(&ctx->chunk), ctx->num);
            }
            if (proxy.is_cst_val2())
            {
                CHUNK_INVARIANT_FAILED("execute_inplace_operation_async: is_cst_val2 TRUE after store! "
                                       "chunk={} num={}",
                                       static_cast<const void*>(&ctx->chunk), ctx->num);
            }
        }

        Base2::mark_for_compression_if_needed_with_value_async(proxy, v_old, self.read_values());
        break;
    }
}
/**
 * @brief Compound operation entry point, with an absorbing-element short-circuit.
 *
 * Applies any foreign deferred write on this offset first, so the RMW sees a
 * truthful old value, then delegates to ProxyBase::apply_atomic_op(). Skipped
 * when the current value is `first_value`, which absorbs the callers'
 * operations (`0 * v == 0`).
 *
 * @tparam U  Operand type.
 * @tparam Op Binary operation type.
 * @param op Operation to apply.
 * @param v  Operand.
 */
template <typename T>
template <typename U, typename Op>
void zero_any::Proxy<T>::apply_compound_operation(Op op, InT<U> v) noexcept
{
    this->check_not_stale();
    // No set_info_on_chunk() here: nothing before apply_atomic_op() reads the
    // cached pointers, and under fast block compression the engine arms them
    // (write_with_optional_lock_info's lbd_info). Arming here would do it twice.

    // Apply any foreign deferred write to this offset before reading v_old, so
    // the RMW cannot clobber it (see arithmetic_assign). Outside begin_write.
    chunk_.reader_sync(num_);

    // Outside the write session: read under a ReadSection, or a pass may release
    // (and a materialisation zero-fill) the buffer between the pointer and the
    // element loads, and the shortcut below would drop the operation
    // (read_reclaim_validation, mode `compound`).
    const InT<T> v_old = Base::read_values_outside_session();

    // first_value absorbs (0 * v == 0). Racy but safe: a concurrent writer
    // cannot make the "was first_value" branch wrong, provided v_old is a value
    // the element really held (hence the ReadSection).
    if (v_old != zero_any::Chunk<T>::first_value)
    {
        // Serial build: apply_atomic_op() runs the body without the engine's
        // lbd_info, and the body reads ptr_second_value.
        if constexpr (!chunk_config::use_fast_block_compression)
        {
            set_info_on_chunk();
        }
        Base::apply_atomic_op(op, v, v_old);
    }
}

} // namespace shrink_array