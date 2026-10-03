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
#include <cstdio>          // std::fprintf (check-build trace)
#include <iostream>        // std::cerr (verbose)
#include <source_location> // std::source_location
#include <type_traits>     // std::remove_cvref_t
#include <utility>         // std::move, std::cmp_greater_equal
#include <vector>          // the per-chunk "queued" flags

#include "chunk/internal/proxy2A.h"            // the declarations these bodies belong to
#include "chunk/internal/in_t.h"               // in_t<T>
#include "chunk/array_chunk2A.h"               // ArrayChunk2A (dependent calls the include tools cannot attribute)
#include "chunk/config.h"                      // build_config::any_check, build_config::chunk_dynamic
#include "chunk/PMP_CHUNK.h"                   // get_thr_rank_fast
#include "chunk/internal/arithmetic_context.h" // ArithmeticAffectationContext
#include "chunk/internal/array_chunk_for.h"    // ArrayChunkFor_t
#include "chunk/internal/concept_is_clean_assignement.h" // base_t
#include "chunk/internal/deferred_op_pool.h"             // pool_new_ctx
#include "chunk/internal/bad_ptr_check.h"                // CHUNK_CHECK_VALUES_PTR (check builds)
#include "chunk/internal/invariant_check.h"              // INVARIANT_FAILED
#include "chunk/internal/chunk_stats.h"                  // increm_chunk_* (CHUNK_STATS)
#include "chunk/internal/chunk_size_policy.h"            // DynChunkSizeGuard, set_thread_chunk_size
#include "chunk/internal/read_reclaim.h" // read_reclaim::PassScope (perform_compression / compress in a region)

/**
 * @file proxy2A_inline.h
 * @brief Out-of-line definitions of Proxy2A and Proxy2AFastCompPolicy.
 *
 * Proxy2A is the element proxy of Chunk2A, the chunk flavour that stores TWO
 * constant values: a global `first_value` shared by every chunk, and a
 * per-chunk `second_value`. A chunk is therefore in one of three states —
 * cst_val1 (all elements are first_value), cst_val2 (all are this chunk's
 * second value), or materialised on the heap — and the whole file is organised
 * around moving between them without ever losing a concurrent write.
 *
 * What lives here:
 *  - **metadata arming** (set_info_on_chunk() and its three narrower siblings),
 *    which caches the per-chunk pointers a write is about to need;
 *  - **compression driving**: trigger_compression_if_needed() flags the
 *    candidate list as worth a pass, perform_compression() runs it;
 *  - **materialisation**: ensure_writable_storage(), which decompresses a
 *    constant chunk before a write can touch it;
 *  - **the write bodies**, each in a synchronous form and an `_async` twin
 *    replayed later from a deferred op.
 *
 * The `_async` twins share a common shape, and the reason for it: a replay runs
 * long after its write was issued, so nothing may be assumed to still hold.
 * Each one rebuilds a fresh proxy from the context, re-checks that
 * `chunk.values` is a real heap buffer BEFORE storing, and re-reads it AFTER —
 * a change means a compressor swapped the buffer under us and the store was
 * lost, so the whole body retries. Under `build_config::any_check` a retry
 * counter turns a livelock into a diagnosable failure instead of a hang.
 *
 * Under dynamic geometry these paths also pin the per-thread chunk size before
 * every materialisation: the flushing thread's ambient size has nothing to do
 * with the chunk being written, and allocating at the wrong size is a
 * heap-buffer-overflow (see the inline notes, which record the ASan findings).
 *
 * @see proxy2A.h for the declarations, proxy_base_inline.h for the write engine
 *      these bodies are handed to, array_chunk2A.h for the chunk itself.
 */

namespace shrink_array
{

// Metadata arming. These four form a family: each arms a different subset of the
// cached per-chunk pointers, and callers pick the cheapest one that covers what
// they are about to touch. Order matters -- num_tab must be resolved before any
// pointer is computed, since every table is indexed by it.

// Full arming: resolves num_tab, then every cached pointer. The most expensive
// of the four, and the default for a write that has not proved it needs less.
template <typename T>
void Proxy2A<T>::set_info_on_chunk() noexcept
{
    set_num_tab();

    Base::set_ptr_is_inside_chunk_list_to_compress();

    ptr_type_cst = &tab_type_cst_val[num_tab][num_chunk];
    if constexpr (is_not_bool)
    {
        BaseSpec::ptr_second_value = BaseSpec::get_tab_second_value(num_tab, num_chunk);
    }
}

// Everything set_info_on_chunk() does EXCEPT ptr_second_value, and without
// re-resolving num_tab: for callers that already hold a valid num_tab and do not
// need the second value.
template <typename T>
void Proxy2A<T>::set_other_info_on_chunk() noexcept
{
    Base::set_ptr_is_inside_chunk_list_to_compress();

    ptr_type_cst = &tab_type_cst_val[num_tab][num_chunk];
}

/**
 * @brief Arms ptr_second_value alone, from an already-valid num_tab.
 *
 * Cheapest member of the arming family: for callers that hold a valid num_tab
 * and only need to compare against this chunk's second value.
 *
 * @pre num_tab and num_chunk are already resolved.
 * @note No-op for bool, whose specialisation keeps no per-chunk second value.
 */
template <typename T>
void Proxy2A<T>::set_ptr_second_value() noexcept
{
    if constexpr (is_not_bool)
    {
        BaseSpec::ptr_second_value = BaseSpec::get_tab_second_value(num_tab, num_chunk);
    }
}

/**
 * @brief Resolves num_tab, then arms ptr_second_value from it.
 *
 * For callers holding neither. Everything each table is indexed by comes from
 * num_tab, so it must be resolved before any pointer is computed.
 *
 * @note The second-value arming is skipped for bool.
 */
template <typename T>
void Proxy2A<T>::set_ptr_second_value_and_num_tab() noexcept
{
    set_num_tab();
    if constexpr (is_not_bool)
    {
        set_ptr_second_value();
    }
}

// Re-reads the slot through ptr_second_value; prefer the explicit-old-value
// variant below wherever the caller already knows it. bool carries no per-chunk
// second value, so it routes to the plain ProxyBase form.
template <typename T>
void Proxy2A<T>::mark_for_compression_if_needed(in_t<T> new_value) const
{
    if constexpr (is_not_bool)
    {
        // Called BEFORE the store (handle_no_bool_assignment): the memory check
        // must expect the old value -- see mark_for_compression_before_store.
        Base2::mark_for_compression_before_store(BaseSpec::ptr_second_value.get(), new_value, Chunk2A<T>::first_value);
    }
    else
    {
        Base2::mark_for_compression_if_needed(new_value);
    }
}

// Explicit-old-value variant: preferred over mark_for_compression_if_needed()
// because it does not re-read the slot, which could race with another writer.
// bool has no per-chunk second value, so it routes to the plain ProxyBase form.
template <typename T>
void Proxy2A<T>::mark_for_compression_if_needed_with_value(in_t<T> old_value, in_t<T> new_value) const
{
    if constexpr (is_not_bool)
    {
        Base2::mark_for_compression_if_needed(BaseSpec::ptr_second_value.get(), old_value, new_value,
                                              Chunk2A<T>::first_value);
    }
    else
    {
        Base::mark_for_compression_if_needed(old_value, new_value);
    }
}

// Grow the three per-tab tables in lockstep. They are indexed by the same
// num_tab, so they must always share the same outer length.
template <typename T>
void Proxy2A<T>::resize_tab_is_int(int nb_tab)
{
    BaseSpec::resize(nb_tab);
    tab_type_cst_val.resize(nb_tab);
    Base::tab_is_inside_chunk_list_to_compress.resize(nb_tab);
}

// Grow one tab's per-chunk tables. Note the asymmetry between the three:
// BaseSpec::resize() and tab_type_cst_val preserve existing entries and only
// initialise the new tail, whereas tab_is_inside_chunk_list_to_compress is
// REALLOCATED outright, so every chunk's "already queued for compression" flag
// is reset to 0, not just the new ones. That is only safe if no chunk of this
// tab is queued at the time -- a chunk still sitting in a candidate list with a
// cleared flag could be queued a second time.
template <typename T>
void Proxy2A<T>::resize_tab_is_int(int num_tab, int nb_chunk_old, int nb_chunk)
{
    BaseSpec::resize(num_tab, nb_chunk_old, nb_chunk);
    // Value-initialised, every flag 0 (make_unique<T[]> value-initialised too:
    // the tail loop that followed it stored zeros over zeros).
    Base::tab_is_inside_chunk_list_to_compress[static_cast<std::size_t>(num_tab)] =
        std::vector<std::atomic<char>>(static_cast<std::size_t>(nb_chunk));
    tab_type_cst_val[num_tab].resize(nb_chunk, 0);
}

/**
 * @brief Flags this thread's candidate list as worth a compression pass.
 *
 * Called after every successful write on the fast path. It does NOT compress:
 * once the list grows past the configured threshold it only marks it
 * "dangerous", which is what makes the next write take the slow path and call
 * perform_compression(), which clears it (mark_list_safe_for_thread) once it
 * has looked at the list, work or not.
 *
 * OpenMP-only: the sequential build drives perform_compression() directly.
 *
 * @param thr_rank Rank whose candidate list is examined.
 */
template <typename T>
void Proxy2A<T>::trigger_compression_if_needed(int thr_rank)
{
    const auto& vec_to_compress_potantially = manager_chunk2A::get_chunk2A_to_compress_potentially(thr_rank);

    if constexpr (build_config::use_openmp)
    {
        const int szm = manager_chunk2A::get_size_max_number_of_chunk_to_check_compress();

        const int sz = static_cast<int>(vec_to_compress_potantially.size());

        if (Base::should_trigger_compression(sz, szm))
        {
            CompressionManagerType::mark_list_dangerous_for_thread(thr_rank);

            if constexpr (chunk_config::verbose)
            {
                std::cerr << "⚠️ List became dangerous: size=" << sz << "\n";
            }
        }
    }
}

/**
 * @brief Compresses this thread's candidate chunks, if there are enough of them.
 *
 * The slow path of the write engine, and the sequential build's only
 * compression driver. Self-gating: below half the configured maximum the
 * candidate list is left alone, so calling it speculatively stays cheap.
 *
 * Above the threshold it drains the thread's deferred write ops first — they
 * must be applied before their chunks are compressed — then runs the block
 * compression and clears the "dangerous" mark. Both of those are OpenMP-only
 * and guarded as such (see the inline notes for why an unguarded decrement made
 * the counter drift negative).
 */
template <typename T>
void Proxy2A<T>::perform_compression() noexcept
{
    // Runs inside a region, next to pure readers: the buffers this pass
    // releases wait for one grace period (read_reclaim.h).
    const read_reclaim::PassScope pass_scope;
    const int                     szm = manager_chunk2A::get_size_max_number_of_chunk_to_check_compress();

    const int thr_rank = get_thr_rank_fast();

    const auto& vec_to_compress_potantially = manager_chunk2A::get_chunk2A_to_compress_potentially(thr_rank);

    const int sz_before = static_cast<int>(vec_to_compress_potantially.size());

    const int sz_threshold = szm / 2;

    if (sz_before > sz_threshold)
    {
        // flush_deferred drains per-thread deferred write ops. Those are only
        // ever enqueued on the OMP write path; the non-OMP write executes
        // write_body() directly (no deferral), so there is nothing to drain — and
        // flush_deferred is declared only in CompressionManagerOmpPolicy's
        // <..., true> specialisation, so naming it at all in a serial build
        // would not compile. Guard the call.
        if constexpr (build_config::use_openmp)
        {
            static_cast<void>(CompressionManagerType::flush_deferred(thr_rank));
        }

        // Route make_block_compression's free_values (get_chunk_allocator keys on
        // thread_chunk_size) to THIS chunk's allocator. An earlier op on a
        // DIFFERENT-size array may have left the per-thread size stale, so freeing
        // this chunk's block would hit the wrong-size slab -> TCAllocator
        // sl_contains assert (fast-alloc). Harmless under std-alloc (svfree is
        // size-agnostic). Dynamic geometry only.
        if constexpr (build_config::chunk_dynamic)
        {
            shrink_array::set_thread_chunk_size(this->chunk_size());
            shrink_array::set_thread_alloc_key(this->alloc_key());
        }
        Chunk2A<T>::make_block_compression(thr_rank, this);
    }
    // Whether or not there was work: the list has been looked at, and the
    // trigger raises the flag again when it next crosses the threshold. Left
    // inside the `if` above, a list drained meanwhile (the between-phase
    // region) kept the rank on the slow path for every write -- see
    // mark_list_dangerous_for_thread. OMP only: the trigger that raises the
    // flag is compiled out in the serial build.
    if constexpr (build_config::use_openmp)
    {
        CompressionManagerType::mark_list_safe_for_thread(thr_rank);
    }
}

// Materialises a constant chunk onto the heap. Note the ordering constraint:
// is_cst_val2() is sampled BEFORE set_info_on_chunk(), but the cached pointers
// it arms are only valid against the post-allocation state, hence the arming
// sits between the sample and create_value().
template <typename T>
void Proxy2A<T>::ensure_writable_storage()
{
    if constexpr (build_config::chunk_dynamic)
    {
        if constexpr (build_config::any_check)
        {
            const unsigned int frozen = this->chunk_size_frozen_;
            const unsigned int live =
                this->ptr_on_array ? static_cast<const ArrayChunkFor_t<ChunkType>*>(this->ptr_on_array)->chunk_size()
                                   : 0;
            const unsigned int used = this->chunk_size();
            if (used != live || frozen != 0U)
            {
                std::fprintf(stderr, "[EWS] used=%u live=%u frozen=%u\n", used, live, frozen);
            }
        }
        shrink_array::set_thread_chunk_size(this->chunk_size());
        shrink_array::set_thread_alloc_key(this->alloc_key());
    }
    const bool is_val2 = is_cst_val2();

    // Cached per-chunk pointers (ptr_second_value, ptr_type_cst, …) are only
    // valid against the post-allocation chunk state, so arm them here — never
    // before.
    set_info_on_chunk();

    // Route the materialisation to THIS array's chunk-size pool. Without this the
    // buffer is allocated at the STALE per-thread thread_chunk_size (left by a
    // previous op on another array), while operator[]/num use this array's
    // size_policy_ -> a smaller buffer written at a larger intra-chunk index ->
    // heap-buffer-overflow under dynamic geometry (ASan-confirmed on the
    // operator--/in-place path, which reaches create_value only through here).
    // Mirrors Proxy2A::create_value (proxy2A.h), which already sets it on the
    // direct-assignment path. Dynamic-only: fixed geometry has one size.
    if constexpr (build_config::chunk_dynamic)
    {
        shrink_array::set_thread_chunk_size(this->chunk_size());
        shrink_array::set_thread_alloc_key(this->alloc_key());
    }

    if constexpr (is_not_bool)
    {
        if (is_val2)
        {
            chunk.create_value(true, BaseSpec::ptr_second_value);
        }
        else if (Base::is_cst_val1())
        {
            chunk.create_value(false, BaseSpec::ptr_second_value);
        }
    }
    else
    {
        if (is_val2)
        {
            chunk.create_value_bool(true);
        }
        else if (Base::is_cst_val1())
        {
            chunk.create_value_bool(false);
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
template <typename Derived, typename T, typename ChunkType, bool Enabled>
void Proxy2AFastCompPolicy<Derived, T, ChunkType, Enabled>::ensure_writable_storage_async(const Derived& proxy) const
{
    const bool is_val2 = proxy.is_cst_val2();

    // Route the materialisation to THIS array's chunk-size pool -- the async twin
    // of the fix in the sync ensure_writable_storage above. Without it, create_value
    // allocates at the STALE per-thread thread_chunk_size while operator[]/num use
    // size_policy_ -> heap-buffer-overflow after a re-chunk (read_value OOB: i >=
    // alloc_chunk_count). proxy.chunk_size() is the live geometry here (frozen only
    // on the deferred-replay path, which does not reach this async materialiser).
    if constexpr (build_config::chunk_dynamic)
    {
        shrink_array::set_thread_chunk_size(proxy.chunk_size());
        shrink_array::set_thread_alloc_key(proxy.alloc_key());
    }

    if constexpr (is_not_bool)
    {
        if (is_val2)
        {
            this->chunk.create_value(true, proxy.ptr_second_value);
        }
        else if (Base::is_cst_val1())
        {
            this->chunk.create_value(false, proxy.ptr_second_value);
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
 * The store itself, once the metadata is armed. Three outcomes, cheapest first:
 *  -# writing the chunk's second value onto a chunk that already reads back as
 *     that value is a no-op — the compressed state is exactly what was wanted;
 *  -# from cst_val1 to the second value, the chunk is materialised and the
 *     value stored;
 *  -# otherwise the chunk is materialised if still constant, then written.
 *
 * @param v Value to store.
 * @pre The metadata pointers this reads are armed (see the handle_*_assignment
 *      callers, which arm exactly what their branch needs).
 */
template <typename T>
void Proxy2A<T>::perform_core_assignment(in_t<T> v) const
{
    if constexpr (is_not_bool)
    {
        if (v == static_cast<T>(BaseSpec::ptr_second_value.get()))
        {
            if (is_cst_val2() || Base::load_value() == BaseSpec::ptr_second_value.get())
            {
                return;
            }
            if (Base::is_cst_val1())
            {
                create_value(BaseSpec::ptr_second_value);

                Base::store_value(BaseSpec::ptr_second_value.get());

                return;
            }
        }
    }
    else if (v)
    {
        if (is_cst_val2() || Base::load_value())
        {
            return;
        }
        if (Base::is_cst_val1())
        {
            create_value_bool();

            Base::store_value(true);

            return;
        }
    }

    if constexpr (chunk_config::stats)
    {
        increm_chunk_write<ChunkType>(v);
    }

    const bool need_alloc = Base2::is_cst_val();

    if (need_alloc)
    {
        if constexpr (is_not_bool)
        {
            create_value(BaseSpec::ptr_second_value);
        }
        else
        {
            create_value_bool();
        }
    }

    Base::store_value(v);

    if constexpr (build_config::do_check)
    {
        global_check_write_nan(v, num);
    }
}
// ========================================================================
// ASSIGNMENT OPERATOR
// ========================================================================

/**
 * @brief Assignment body for non-bool element types.
 *
 * Arms the per-chunk metadata, marks the chunk for compression when the new
 * value closes a constant run, then stores. The arming is deliberately split:
 * on a cst_val2 chunk only the second value is resolved first, so that a write
 * of that very value can return without paying for the rest.
 *
 * @param v Value to assign.
 */
template <typename T>
void Proxy2A<T>::handle_no_bool_assignment(in_t<T> v)
{
    if constexpr (is_not_bool)
    {
        if (is_cst_val2())
        {
            set_ptr_second_value_and_num_tab();

            if (v == static_cast<T>(BaseSpec::ptr_second_value.get()))
            {
                return;
            }

            set_other_info_on_chunk();
        }
        else
        {
            set_info_on_chunk();
        }

        mark_for_compression_if_needed(v);

        perform_core_assignment(v);
    }
}

/**
 * @brief Assignment body for `T = bool`.
 *
 * Same shape as handle_no_bool_assignment(), with the comparisons specialised:
 * bool keeps no per-chunk second value, so cst_val2 simply means "all true" and
 * the early-out tests `v == true` directly.
 *
 * @param v Value to assign.
 */
template <typename T>
void Proxy2A<T>::handle_bool_assignment(in_t<T> v)
{
    if constexpr (is_bool)
    {
        if (is_cst_val2())
        {
            set_num_tab();

            if (v)
            {
                return;
            }

            set_other_info_on_chunk();
        }
        else
        {
            set_info_on_chunk();
        }

        mark_for_compression_if_needed(v);

        perform_core_assignment(v);
    }
}

// Entry point of every assignment. Early-outs when the chunk is already
// all-first_value and that is what is being written, hands the real body to the
// write engine, and re-issues it as a deferred op when the engine refuses.
// The deferred body is written out in full here rather than reusing
// handle_*_assignment: a replay cannot trust the cached metadata, so it works
// straight off a freshly loaded values pointer under the retry protocol.
template <typename T>
template <typename U>
T Proxy2A<T>::assign_impl(in_t<U> v) noexcept
{
    this->check_not_stale();
    if (Base::is_cst_val1_and_v_equal_first_value(v))
    {
        return v;
    }

    [[maybe_unused]] bool need_async = false;

    if constexpr (is_not_bool)
    {
        // By reference: v is this call's parameter (a value, or the caller's
        // object through the large-operand operator=), alive for the whole
        // synchronous call; the deferred path below copies it into its context.
        const auto assignment_body = [this, &v] { handle_no_bool_assignment(v); };

        need_async = Base2::write_with_optional_lock_with_set_ptr(assignment_body);
    }
    else
    {
        const auto assignment_body_bool = [this, &v] { handle_bool_assignment(v); };

        need_async = Base2::write_with_optional_lock_with_set_ptr(assignment_body_bool);
    }

    // Asynchronous slow path
    if constexpr (build_config::use_openmp)
    {
        if (need_async)
        {
            // The context stores the operand by value: U itself, or the type
            // behind the large-operand operator='s `const U&`.
            using CtxType = ArithmeticAffectationContext<T, std::remove_cvref_t<U>, ChunkType, Proxy2A<T>>;

            // Capture only location + operand
            auto* ctx = pool_new_ctx<CtxType>(*this, v, ptr_on_array);

            auto lbd_equal_asynch = [ctx] // the operand lives in the context: no second copy in the closure
            {
                auto&      chunk = ctx->chunk;
                const int  num   = ctx->num;
                const auto proxy = Base::template make_proxy_with_info<Proxy2A>(ctx);

                // Deferred replay: create_value -> allocate_chunk_internal sizes the
                // buffer from the ambient thread_chunk_size(), but here the flushing
                // thread's ambient is unrelated to THIS op's chunk. Pin the frozen
                // per-op geometry for the write, restore on exit so the flusher's own
                // in-flight write is untouched. Dynamic geometry only.
                [[maybe_unused]] const DynChunkSizeGuard sz_guard{proxy.chunk_size()};

                [[maybe_unused]] int retry_count = 0;

                while (true)
                {
                    const bool is_cst1 = proxy.is_cst_val1();
                    const bool is_cst2 = proxy.is_cst_val2();

                    if (is_cst1 || is_cst2)
                    {
                        if constexpr (is_not_bool)
                        {
                            chunk.create_value(is_cst2, proxy.ptr_second_value);
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
                        if constexpr (build_config::any_check)
                        {
                            if (++retry_count > 1000)
                            {
                                INVARIANT_FAILED("lbd_equal_asynch: pre-store retry loop exceeded 1000 "
                                                 "iterations on chunk={} num={} — pathological contention?",
                                                 static_cast<const void*>(&chunk), num);
                            }
                        }
                        continue;
                    }

                    if constexpr (build_config::any_check)
                    {
                        // Check-build guard: catch a CULPRIT op whose index is out of range
                        // BEFORE it writes out of bounds (same guard as the v2 path).
                        //
                        // fw16 (dyn guard): compare against proxy.chunk_size(), the
                        // per-op geometry frozen at enqueue by fw14, NOT the static
                        // compile-time __chunk_size. In a dynamic build the real
                        // chunk width is runtime (size_policy_.size()); __chunk_size
                        // is only the static default (e.g. 64), so this guard fired
                        // a false OUT OF RANGE whenever the dynamic geometry differed
                        // from that default — every other write here already uses
                        // proxy.chunk_size(). Static builds are unaffected: there
                        // proxy.chunk_size() == __chunk_size.
                        const unsigned int csz_guard = proxy.chunk_size();
                        if (std::cmp_greater_equal(num, csz_guard))
                        {
                            INVARIANT_FAILED("[deferred2A write] num OUT OF RANGE "
                                             "(num={}, chunk_size={}, chunk={})",
                                             num, csz_guard, static_cast<const void*>(&chunk));
                        }

                        // Check-build guard: garbage-pointer check on the write target.
                        CHUNK_CHECK_VALUES_PTR_C(values_before, "deferred2A.pre_store", &chunk);

                        // Check-build guard: narrow-race re-check. The pre-store guard above
                        // looked at values_before; if the chunk was (re)compressed
                        // between values_load() and here, writing would corrupt a
                        // constant array.
                        // Re-checking cst state routes those cases back through the
                        // retry loop (consistent with the post-store retry design)
                        // instead of writing blind; the retry_count cap still turns
                        // a persistent loop into a diagnosable failure.
                        if (proxy.is_cst_val1() || proxy.is_cst_val2()) [[unlikely]]
                        {
                            if (++retry_count > 1000)
                            {
                                INVARIANT_FAILED("[deferred2A write] chunk keeps reverting to cst before "
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

                    // Safe to write.
                    // raw-read: deferred replay, after materialisation -- flushes exclude the stealers
                    values_before[num] = static_cast<T>(ctx->operand);

                    // Post-store confirmation.
                    const T* values_after = chunk.values_load();
                    if (values_after != values_before) [[unlikely]]
                    {
                        if constexpr (build_config::any_check)
                        {
                            if (++retry_count > 1000)
                            {
                                INVARIANT_FAILED("lbd_equal_asynch: post-store retry loop exceeded 1000 "
                                                 "iterations on chunk={} num={} — pathological contention?",
                                                 static_cast<const void*>(&chunk), num);
                            }
                        }
                        continue;
                    }

                    // Diagnostic checks (any_check only).
                    if constexpr (build_config::any_check)
                    {
                        if (proxy.is_cst_val1())
                        {
                            INVARIANT_FAILED("lbd_equal_asynch: is_cst_val1 TRUE after successful store! "
                                             "chunk={} num={} values_before={} values_after={}",
                                             static_cast<const void*>(&chunk), num,
                                             static_cast<const void*>(values_before),
                                             static_cast<const void*>(values_after));
                        }
                        if (proxy.is_cst_val2())
                        {
                            INVARIANT_FAILED("lbd_equal_asynch: is_cst_val2 TRUE after successful store! "
                                             "chunk={} num={} values_before={} values_after={}",
                                             static_cast<const void*>(&chunk), num,
                                             static_cast<const void*>(values_before),
                                             static_cast<const void*>(values_after));
                        }
                        // raw-read: deferred replay, after materialisation -- flushes exclude the stealers
                        const in_t<T> readback = values_after[num];
                        const auto    expected = static_cast<T>(ctx->operand);
                        if (readback != expected)
                        {
                            INVARIANT_FAILED("lbd_equal_asynch: readback mismatch after store! "
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
 * The generic sync wrapper the load/store write paths funnel through:
 * materialise, snapshot the old value, run the body, then re-mark the chunk for
 * compression from the before/after pair. The body does its own store — this
 * only guarantees it runs against heap storage.
 *
 * @tparam ModifyFn Callable performing the store.
 * @param modify_fn Write body.
 */
template <typename T>
template <typename ModifyFn>
void Proxy2A<T>::execute_simple_operation(ModifyFn modify_fn)
{
    ensure_writable_storage();

    const in_t<T> v_old = read_values();

    modify_fn();

    mark_for_compression_if_needed_with_value(v_old, read_values());
}

/**
 * @brief Replay twin of execute_simple_operation().
 *
 * Follows the file's retry protocol: rebuild the proxy, materialise, check that
 * the values pointer is a real heap buffer, run the body on THAT pointer, and
 * confirm afterwards that it did not move. Any failed check restarts the loop —
 * the whole body, proxy included, since a compressor may have changed the chunk
 * state, not just its buffer.
 *
 * @tparam LAMBDA Callable taking the values pointer to write through.
 * @param ctx       Deferred-operation context being replayed.
 * @param operation Write body.
 */
template <typename Derived, typename T, typename ChunkType, bool Enabled>
template <typename LAMBDA>
void Proxy2AFastCompPolicy<Derived, T, ChunkType, Enabled>::execute_simple_operation_async(auto* ctx, LAMBDA operation)
{
    [[maybe_unused]] int retry_count = 0;

    while (true)
    {
        const Proxy2A proxy = Base::template make_proxy_with_info<Derived>(ctx);
        derived().ensure_writable_storage_async(proxy);

        const in_t<T> v_old = Base::read_values_async(ctx);

        // Fresh load
        T* values_before = ctx->chunk.values_load();

        // Pre-store guard
        if (ctx->chunk.is_constant_ptr(values_before)) [[unlikely]]
        {
            if constexpr (build_config::any_check)
            {
                if (++retry_count > 1000)
                {
                    invariant_failed(std::source_location::current(),
                                     "execute_simple_operation_async: pre-store retry > 1000 "
                                     "on chunk={} num={}",
                                     static_cast<const void*>(&ctx->chunk), ctx->num);
                }
            }
            continue;
        }

        operation(values_before);

        // Post-store confirmation
        const T* values_after = ctx->chunk.values_load();
        if (values_after != values_before) [[unlikely]]
        {
            if constexpr (build_config::any_check)
            {
                if (++retry_count > 1000)
                {
                    invariant_failed(std::source_location::current(),
                                     "execute_simple_operation_async: post-store retry > 1000 "
                                     "on chunk={} num={}",
                                     static_cast<const void*>(&ctx->chunk), ctx->num);
                }
            }
            continue;
        }

        Base2::mark_for_compression_if_needed_with_value_async(proxy, v_old, derived().read_values());
        break;
    }
}

/**
 * @brief Applies a compound arithmetic operation, materialising only when the result leaves the constant.
 *
 * Computes in `std::common_type_t<T, U>` and truncates back to T, then branches
 * on what the result means for the chunk's state:
 *  - result differs from the chunk's second value while the chunk IS cst_val2:
 *    the constant no longer holds, so materialise and let @p modify_fn store;
 *  - chunk already materialised: store directly and re-mark for compression;
 *  - result equals the second value on a cst_val2 chunk: nothing to do, the
 *    compressed representation already says it.
 *
 * That third case is why the operation is not simply "read, compute, store":
 * a compressed chunk whose value does not change must not be materialised.
 *
 * @tparam OP       Binary operation type.
 * @tparam ModifyFn Callable storing the result on the materialised path.
 * @tparam U        Operand type.
 * @param op        Operation to apply.
 * @param v         Operand.
 * @param modify_fn Store body, invoked as `modify_fn(op)`.
 */
template <typename T>
template <typename OP, typename ModifyFn, typename U>
void Proxy2A<T>::execute_arithmetic_compound_operation(OP op, const U& v, ModifyFn modify_fn)
{
    // Determine high-precision type (e.g. double if mixing float array and double constant)
    using CalcType = std::common_type_t<T, base_t<U>>;

    if constexpr (build_config::chunk_dynamic)
    {
        shrink_array::set_thread_chunk_size(this->chunk_size());
        shrink_array::set_thread_alloc_key(this->alloc_key());
    }

    // Read current value from memory
    const in_t<T> v_old = read_values();

    // 1. Perform math in high precision to verify if result matches constant values
    const CalcType v_new_precise   = op(static_cast<CalcType>(v_old), static_cast<CalcType>(v));
    const auto     v_new_truncated = static_cast<T>(v_new_precise);

    const bool cst_val2 = is_cst_val2();

    if constexpr (is_not_bool)
    {
        // 2. Check if the result deviates from the second constant value stored in this tab
        const bool will_be_not_second_value = (v_new_truncated != BaseSpec::ptr_second_value.get()) && cst_val2;

        if (will_be_not_second_value)
        {
            // Result is no longer the constant second_value: allocate memory
            chunk.create_value(true, BaseSpec::ptr_second_value);

            // Execute the final write using the provided lambda
            modify_fn(op);
        }
        else if (Base2::is_not_cst_val2())
        {
            // Standard path: write result back to allocated memory
            Base::store_value(v_new_truncated);

            // Check if this write makes the chunk compressible again
            mark_for_compression_if_needed_with_value(v_old, v_new_truncated);
        }
    }
    else
    {
        // 3. Boolean specific logic (cst_val2 is true)
        const bool will_be_not_second_value = !v_new_truncated && cst_val2;

        if (will_be_not_second_value)
        {
            chunk.create_value_bool(true);
            modify_fn(op);
        }
        else if (Base::is_not_cst_val2())
        {
            chunk.assert_writable_buffer(chunk.values_load());

            Base::store_value(v_new_truncated);
            mark_for_compression_if_needed_with_value(v_old, v_new_truncated);
        }
    }
}

/**
 * @brief Replay twin of execute_arithmetic_compound_operation().
 *
 * Same three-way branch on the chunk state, each arm wrapped in the file's
 * retry protocol. The no-op arm is guarded too: "the result already equals the
 * second value" was decided from a state that may have changed since, so it is
 * re-tested before the loop is allowed to exit.
 *
 * @tparam OP  Binary operation type.
 * @tparam LBD Callable storing the result on the materialised path.
 * @tparam U   Operand type.
 * @param ctx Deferred-operation context being replayed.
 * @param op  Operation to apply.
 * @param v   Operand.
 * @param lbd Store body, invoked as `lbd(op)`.
 */
template <typename Derived, typename T, typename ChunkType, bool Enabled>
template <typename OP, typename LBD, typename U>
void Proxy2AFastCompPolicy<Derived, T, ChunkType, Enabled>::execute_arithmetic_compound_operation_async(auto*    ctx,
                                                                                                        OP       op,
                                                                                                        const U& v,
                                                                                                        LBD lbd) const
{
    using CalcType = std::common_type_t<T, base_t<U>>;

    [[maybe_unused]] int retry_count = 0;

    while (true)
    {
        // Rebuild proxy each iteration (it caches state we want fresh)
        const Proxy2A proxy = Base::template make_proxy_with_info<Derived>(ctx);

        // Route create_value below to THIS op's geometry (proxy.chunk_size(),
        // frozen for a replay), not the stale per-thread thread_chunk_size. Mirror
        // of the sync twin execute_arithmetic_compound_operation. Without it,
        // create_value allocates at the stale TLS while num / the store offset use
        // the proxy geometry -> out-of-bounds write -> heap corruption that surfaces
        // at a later free (malloc_consolidate / free(): corrupted unsorted chunks in
        // destroy_after_flush). Dynamic geometry only.
        if constexpr (build_config::chunk_dynamic)
        {
            shrink_array::set_thread_chunk_size(proxy.chunk_size());
            shrink_array::set_thread_alloc_key(proxy.alloc_key());
        }

        const in_t<T>  v_old           = Base::read_values_async(ctx);
        const CalcType v_new_precise   = op(static_cast<CalcType>(v_old), static_cast<CalcType>(v));
        const auto     v_new_truncated = static_cast<T>(v_new_precise);

        const bool cst_val2 = proxy.is_cst_val2();

        if constexpr (is_not_bool)
        {
            const bool will_be_not_second_value = (v_new_truncated != proxy.ptr_second_value.get()) && cst_val2;

            if (will_be_not_second_value)
            {
                // Decompress path: lbd will re-read and store
                ctx->chunk.create_value(true, proxy.ptr_second_value);

                const T* values_before = ctx->chunk.values_load();

                // Pre-store guard inside the decompressed-path branch
                if (ctx->chunk.is_constant_ptr(values_before)) [[unlikely]]
                {
                    if constexpr (build_config::any_check)
                    {
                        if (++retry_count > 1000)
                        {
                            INVARIANT_FAILED("execute_arithmetic_compound_operation_async (decompress path): "
                                             "retry > 1000 on chunk={} num={}",
                                             static_cast<const void*>(&ctx->chunk), ctx->num);
                        }
                    }
                    continue;
                }

                lbd(op);

                // Post-store: did `values` change?
                const T* values_after = ctx->chunk.values_load();
                if (values_after != values_before) [[unlikely]]
                {
                    if constexpr (build_config::any_check)
                    {
                        if (++retry_count > 1000)
                        {
                            INVARIANT_FAILED(
                                "execute_arithmetic_compound_operation_async (decompress path post-store): "
                                "retry > 1000 on chunk={} num={}",
                                static_cast<const void*>(&ctx->chunk), ctx->num);
                        }
                    }
                    continue;
                }

                break;
            }
            if (proxy.is_not_cst_val2())
            {
                const T* values_before = ctx->chunk.values_load();

                // Pre-store guard
                if (ctx->chunk.is_constant_ptr(values_before)) [[unlikely]]
                {
                    if constexpr (build_config::any_check)
                    {
                        if (++retry_count > 1000)
                        {
                            INVARIANT_FAILED("execute_arithmetic_compound_operation_async (heap path): "
                                             "retry > 1000 on chunk={} num={}",
                                             static_cast<const void*>(&ctx->chunk), ctx->num);
                        }
                    }
                    continue;
                }

                Base::store_value_async(ctx, v_new_truncated);

                // Post-store confirmation
                const T* values_after = ctx->chunk.values_load();
                if (values_after != values_before) [[unlikely]]
                {
                    if constexpr (build_config::any_check)
                    {
                        if (++retry_count > 1000)
                        {
                            INVARIANT_FAILED("execute_arithmetic_compound_operation_async (heap path post-store): "
                                             "retry > 1000 on chunk={} num={}",
                                             static_cast<const void*>(&ctx->chunk), ctx->num);
                        }
                    }
                    continue;
                }

                proxy.mark_for_compression_if_needed_with_value(v_old, v_new_truncated);
                break;
            }
            {
                // is_cst_val2 AND v_new_truncated == ptr_second_value.get()
                // → no-op write (the result is already the constant value)
                // BUT we should validate this still holds at break time:
                if (proxy.is_cst_val2() && v_new_truncated == proxy.ptr_second_value.get())
                {
                    // Confirmed: nothing to do.
                    break;
                }
                // State changed between decision and now: retry.
                if constexpr (build_config::any_check)
                {
                    if (++retry_count > 1000)
                    {
                        INVARIANT_FAILED("execute_arithmetic_compound_operation_async (no-op path): "
                                         "retry > 1000 on chunk={} num={}",
                                         static_cast<const void*>(&ctx->chunk), ctx->num);
                    }
                }
            }
        }
        else
        {
            // bool specialization — same pattern, adapted
            const bool will_be_not_second_value = !v_new_truncated && cst_val2;

            if (will_be_not_second_value)
            {
                ctx->chunk.create_value_bool(true);
                lbd(op);
                // Optional: post-store check on bool — left as exercise
                break;
            }
            if (proxy.is_not_cst_val2())
            {
                Base::store_value_async(ctx, v_new_truncated);
                proxy.mark_for_compression_if_needed_with_value(v_old, v_new_truncated);
                break;
            }
            {
                if (proxy.is_cst_val2() && v_new_truncated)
                {
                    break;
                }
            }
        }
    }
}
/**
 * @brief Runs an in-place write body (bitwise, ++/--) on a materialised chunk.
 *
 * Like execute_simple_operation(), except the body is handed the values array
 * and writes through it directly.
 *
 * @note The old value is snapshotted BEFORE materialising here, and AFTER in
 *       execute_simple_operation(). Both read the same value — a constant chunk
 *       still reads back its constant — but the asymmetry is real, so do not
 *       assume the two wrappers order these steps alike.
 *
 * @tparam ModifyFn Callable taking the values array.
 * @param modify_fn Write body.
 */
template <typename T>
template <typename ModifyFn>
void Proxy2A<T>::execute_inplace_operation(ModifyFn modify_fn)
{
    const in_t<T> v_old = read_values();

    ensure_writable_storage();

    internal::modify_element<Chunk2A<T>, T>(values, num, modify_fn);

    mark_for_compression_if_needed_with_value(v_old, read_values());
}

// Replay twin of execute_inplace_operation(): the file's retry protocol around
// a body that writes through the freshly-loaded values pointer. The extra
// any_check assertions after the store catch a chunk that went back to a
// constant state while we were writing it.
template <typename Derived, typename T, typename ChunkType, bool Enabled>
template <typename LAMBDA>
void Proxy2AFastCompPolicy<Derived, T, ChunkType, Enabled>::execute_inplace_operation_async(auto*  ctx,
                                                                                            LAMBDA operation) const
{
    [[maybe_unused]] int retry_count = 0;

    while (true)
    {
        // Load old value (best-effort; will be re-checked via post-store)
        const in_t<T> v_old = Base::read_values_async(ctx);

        // Rebuild proxy each iteration to refresh state
        const Proxy2A proxy = Base::template make_proxy_with_info<Derived>(ctx);

        // Ensure heap storage (decompresses if needed)
        derived().ensure_writable_storage_async(proxy);

        // Fresh load of values pointer for the actual store
        T* values_before = ctx->chunk.values_load();

        // Pre-store guard
        if (ctx->chunk.is_constant_ptr(values_before)) [[unlikely]]
        {
            if constexpr (build_config::any_check)
            {
                if (++retry_count > 1000)
                {
                    INVARIANT_FAILED("execute_inplace_operation_async: pre-store retry > 1000 "
                                     "on chunk={} num={}",
                                     static_cast<const void*>(&ctx->chunk), ctx->num);
                }
            }
            continue;
        }

        // Apply the operation on the freshly-loaded pointer
        internal::modify_element<ChunkType, T>(values_before, ctx->num, operation);

        // Post-store confirmation
        const T* values_after = ctx->chunk.values_load();
        if (values_after != values_before) [[unlikely]]
        {
            if constexpr (build_config::any_check)
            {
                if (++retry_count > 1000)
                {
                    INVARIANT_FAILED("execute_inplace_operation_async: post-store retry > 1000 "
                                     "on chunk={} num={}",
                                     static_cast<const void*>(&ctx->chunk), ctx->num);
                }
            }
            continue;
        }

        // Diagnostic checks (any_check only)
        if constexpr (build_config::any_check)
        {
            if (proxy.is_cst_val1())
            {
                INVARIANT_FAILED("execute_inplace_operation_async: is_cst_val1 TRUE after store! "
                                 "chunk={} num={}",
                                 static_cast<const void*>(&ctx->chunk), ctx->num);
            }
            if (proxy.is_cst_val2())
            {
                INVARIANT_FAILED("execute_inplace_operation_async: is_cst_val2 TRUE after store! "
                                 "chunk={} num={}",
                                 static_cast<const void*>(&ctx->chunk), ctx->num);
            }
        }

        // Success — mark for compression and exit
        Base2::mark_for_compression_if_needed_with_value_async(proxy, v_old, derived().read_values());
        break;
    }
}
/**
 * @brief Compound operation entry point, with an absorbing-element short-circuit.
 *
 * Arms the metadata, materialises any foreign deferred write on this offset so
 * the read-modify-write sees a truthful old value, then delegates to
 * ProxyBase::apply_atomic_op().
 *
 * Skips the operation entirely when the current value is `first_value` — the
 * callers pass operations for which first_value absorbs (`0 * v == 0`). The
 * check is racy by design and safe: a writer changing the value concurrently
 * would only have made this call store what is returned anyway.
 *
 * @tparam U  Operand type.
 * @tparam Op Binary operation type.
 * @param op Operation to apply.
 * @param v  Operand.
 */
template <typename T>
template <typename U, typename Op>
void Proxy2A<T>::apply_compound_operation(Op op, in_t<U> v, bool /*check_first_value_result*/) noexcept
{
    this->check_not_stale();
    // No set_info_on_chunk() here: nothing before apply_atomic_op() reads the
    // cached pointers (reader_sync is the chunk's, read_values resolves the
    // values pointer itself), and under fast block compression the engine
    // arms them anyway (write_with_optional_lock_info's lbd_info). Arming
    // here too recomputed every pointer twice per *=, /=, %=.

    // Cross-rank RMW materialisation (CHUNK_SYNC_ON_READ): see arithmetic_assign.
    // Pull any foreign deferred write to this offset to applied state before we
    // read v_old, so the RMW cannot clobber it. Precise gate, outside begin_write.
    chunk.reader_sync(num);

    // Outside the write session: read under a ReadSection, or a pass in the
    // region may release (and a materialisation zero-fill) the buffer between
    // the pointer load and the element load -- the shortcut below then drops
    // the operation (read_reclaim_validation, mode `compound`).
    const in_t<T> v_old = Base::read_values_outside_session();

    // Fast shortcut: if current value is first_value (usually 0), 0 * v is 0.
    // Safe to short-circuit because a concurrent writer that changes this
    // value cannot make the previous-value-was-first_value branch wrong:
    // the operation would have left it unchanged anyway. That holds for a
    // value the element really held -- hence the ReadSection above.
    if (v_old != Chunk2A<T>::first_value)
    {
        // Serial build: apply_atomic_op() runs the body straight away, without
        // the engine and its lbd_info, and the body reads ptr_second_value.
        if constexpr (!chunk_config::use_fast_block_compression)
        {
            set_info_on_chunk();
        }
        Base::apply_atomic_op(op, v, v_old);
    }
}

} // namespace shrink_array