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

#include <algorithm> // std::fill_n (the create_value fills)
#include <cstdlib>   // std::abort
#include <optional>  // std::nullopt (the clears)

#include "chunk/internal/in_t.h"            // InT<T>: the element snapshots
#include "chunk/internal/chunk_array_new.h" // chunk_array_new::init_chunks (operator new[])
#include "chunk/internal/invariant_check.h" // CHUNK_INVARIANT_FAILED (steal-mark check)

#include "chunk/internal/chunk_size_policy.h" // ChunkElems, set_thread_chunk_size
#include "chunk/PMP_CHUNK.h"                  // CHUNK_PMP_*, get_thr_rank_fast, pmp::team
#include "chunk/config.h"                     // build_config::threaded, chunk_config::use_chunk_steal_mark
#include "chunk/internal/cpu_pause.h"         // CHUNK_CPU_PAUSE (spin in create_value*)
#include "chunk/zero_any_chunk.h" // the declarations these bodies belong to (self-contained, like the other *_inline.h)
#include "chunk/internal/zero_any_proxy.h"      // zero_any::Proxy<T>
#include "chunk/internal/manager_chunk.h"       // ManagerChunk2A:: (dependent calls the include tools cannot attribute)
#include "chunk/internal/block_pass_precheck.h" // block_pass_yields_to_flush
#include "chunk/internal/write_compress_sync_policy.h" // WriteCompressSyncPolicy::is_thread_writing_on (wait_before_materialise)
#include "chunk/chunk_traits.h"                        // ChunkTraits<ChunkT>::ProxyType (wait_before_materialise)
#include "chunk/internal/tab_cst_val2_holder.h" // SecondValueRef
#include "chunk/internal/test_hooks.h"          // TEST KNOB: test_hooks::maybe_pass_delay

/**
 * @file zero_any_chunk_inline.h
 * @brief Out-of-line definitions of zero_any::Chunk: materialisation and compression.
 *
 * A zero_any block is in one of three states; every function here moves it between them
 * or decides that it should move:
 *  - **cst_val1**: `values` points at the shared all-`first_value` block;
 *  - **cst_val2**: `values` points at the identifier slot of THIS chunk's own
 *    second value (tab_cst_val2_holder.h; zero_any's distinguishing feature —
 *    zero_one_value has one second value per array, zero_any one per chunk);
 *  - **materialised**: `values` points at a private heap buffer.
 *
 * The two directions:
 *  - **materialise** (create_value(), create_value_bool()): allocate a buffer, pre-fill it,
 *    then CAS it into `values`. The CAS is the publication: a concurrent reader sees either
 *    the old constant or the finished buffer, never a half-built one;
 *  - **compress** (the clear / block_clear / clear_int_st family): release the
 *    buffer and point `values` back at a shared constant block.
 *
 * Around those sits the candidate pipeline: a write flags a chunk as a
 * candidate, fill_list_to_compress() scans it and sorts it into the
 * "all-first_value" or "all-one-other-value" list, and make_block_compression()
 * applies the clears and clears the flags.
 *
 *  - **The materialise / clear functions exist twice**, on the two specialisations of
 *    zero_any::ChunkOmpPolicy (build_config::threaded); the passes branch with `if constexpr`.
 *    The serial twin often takes a shorter path (no CAS, no rank, no handshake).
 *  - **Geometry is pinned per chunk, never inherited.** A candidate list can hold chunks
 *    of several arrays, so a single pin at the top of a pass is wrong: each entry carries
 *    its array's ChunkElems, set as ambient size right before the clear. Otherwise a buffer
 *    is freed through the wrong size pool.
 *
 * @see zero_any_chunk.h for the declarations, chunk_base.h for the shared
 *      allocate/free machinery, zero_any_proxy_inline.h for the write paths that feed
 *      the candidate lists.
 */

namespace shrink_array
{

// ── Array-new (placement-new arrays; see chunk_array_new.h) ────────────────
// Out of line on purpose, as operator delete[] below: see there.
template <typename T, typename Derived, bool Enabled>
[[gnu::noinline]] void* zero_any::ChunkOmpPolicy<T, Derived, Enabled>::operator new[](std::size_t sz)
{
    auto* v = static_cast<zero_any::Chunk<T>*>(::operator new(sz));
    init_chunks_cst_val(v, sz / sizeof(zero_any::Chunk<T>));
    return static_cast<void*>(v);
}

// Out of line on purpose, as operator new[] above: inlined, either one shows gcc's
// -Wmismatched-new-delete a ::operator new/delete paired with the class's other operator
// (a false positive). As calls, the pair it sees is the class's own new[] / delete[].
template <typename T, typename Derived, bool Enabled>
[[gnu::noinline]] void zero_any::ChunkOmpPolicy<T, Derived, Enabled>::operator delete[](void* ptr)
{
    ::operator delete(ptr);
}

template <typename T, typename Derived, bool Enabled>
void zero_any::ChunkOmpPolicy<T, Derived, Enabled>::init_chunks_cst_val(zero_any::Chunk<T>* v,
                                                                        std::size_t         count) noexcept
{
    chunk_array_new::init_chunks(v, count, [](zero_any::Chunk<T>& chunk) { chunk.init_after_constructor(); });
}

/**
 * @brief Records which of the two constants this chunk is being compressed to.
 *
 * @param num_compress_value 0 for the first constant, anything else for the second.
 * @param ptr_type_cst Slot of the per-chunk type table to write.
 */
template <typename T>
void zero_any::Chunk<T>::set_cst_val_to_cst_val(int num_compress_value, char* ptr_type_cst) noexcept
{
    if (num_compress_value == 0)
    {
        set_cst_val_to_cst_val1(ptr_type_cst);
    }
    else
    {
        set_cst_val_to_cst_val2(ptr_type_cst);
    }
}

/**
 * @brief Scans this chunk and sorts it into one of the two compression lists.
 *
 * One read of the chunk tracks two hypotheses -- "all first_value" and "all one other
 * value" -- and stops once both are disproved. A chunk satisfying both is filed compact1 only.
 *
 * The candidate second value is element 0; a compact2 hit is queued WITH that value, for
 * the caller to register. A chunk already at cst_val2 is not scanned (see the body).
 *
 * @param ll_compt1  List of chunks compressible to first_value.
 * @param ll_compt2  List of chunks compressible to one other value.
 * @param num_chunk  Index of this chunk in its array.
 * @param num_tab    Table number of the owning array.
 * @param size       Geometry of the owning array, carried through to the clear.
 * @param claimed    Whether the caller claimed (steal-marked) the chunk first;
 *                   check builds assert it.
 *
 * @note Chunks already compressed to first_value are skipped outright.
 * @note make_block_compression() treats a `T{}` second value as "skip": unambiguous only
 *       because `first_value` is 0 (an all-zero chunk is classified compact1).
 */
template <typename T>
void zero_any::Chunk<T>::fill_list_to_compress(VecTupleChunkA2pInt2&  ll_compt1,
                                               VecTupleChunkA2pTint2& ll_compt2,
                                               int                    num_chunk,
                                               int                    num_tab,
                                               ChunkElems             size,
                                               [[maybe_unused]] bool  claimed)
{
    if (Base::is_not_cst_val1())
    {
        // The chunk scanned must be claimed: make_block_compression() marks every entry
        // before start(), whose barrier publishes the marks to the writers. Asserted: an
        // unclaimed scan races store_value().
        if constexpr (build_config::any_check && chunk_config::use_chunk_steal_mark)
        {
            // Only where a claim was made: the resize entry point
            // (make_block_compression(thr, nullptr)) runs outside any parallel region and
            // skips the claim loop by design.
            if (claimed && !Base::steal_marked()) [[unlikely]]
            {
                CHUNK_INVARIANT_FAILED("fill_list_to_compress on an unclaimed chunk (num_chunk=%d num_tab=%d)",
                                       num_chunk, num_tab);
            }
        }

        // CHUNK_STATS: one candidate scanned (compress/compress_attempt = scan yield).
        increm_chunk_compress_attempt<zero_any::Chunk<T>>();

        // Already at its second value (compress() ran over it while queued): do not scan.
        // `values` is a second-value slot that reads as all first_value; the chunk would go
        // on ll_compt1 and block_clear() would free the slot as a buffer.
        if (this->is_cst_val2())
        {
            return;
        }

        // The first element is the candidate second constant.
        const InT<T> second_v = Base::read_value_scan(0);
        const auto   scan     = Base::scan_constants(first_value, second_v, size.count());

        if (scan.first)
        {
            ll_compt1.emplace_back(std::make_tuple(this, num_chunk, num_tab, size));
        }
        if (scan.second)
        {
            ll_compt2.emplace_back(std::make_tuple(this, second_v, num_chunk, num_tab, size));
        }
    }
}

/**
 * @brief Runs a full compression pass over one thread's candidate list.
 *
 * Scans every queued candidate into the two lists (fill_list_chunk_to_compress_in_par),
 * applies the clears, resets the per-chunk "already queued" flags and empties the list.
 *
 * A second-list chunk's new second value must be REGISTERED (find_or_set_ind_cst_val2)
 * before the clear, so clear_int_st() commits the right pointer. `T{}` entries are skipped.
 *
 * @param thr     Rank whose candidate list is processed.
 * @param proxy2a Proxy of the write that triggered the pass; may be null. Used ONLY for
 *                the compression handshake and re-queueing its chunk, never for sizing.
 *
 * @note Under OpenMP the body is gated by `compression_mgr.start()`, which can decline
 *       (another thread is compressing): the pass then does nothing.
 */
template <typename T>
void zero_any::Chunk<T>::make_block_compression(int thr, zero_any::Proxy<T>* proxy2a)
{
    // Geometry is pinned PER CHUNK, never once here (see the @file comment): each entry
    // carries its array's ChunkElems (from push_to_compress_list()), pinned right before
    // clear_int_st() reaches ChunkBase::free_values(), which routes by the ambient
    // thread_chunk_size() (see zero_one_value::Proxy::manage_full_first_value() in zero_one_value_proxy.h).
    auto& vec_tuple_to_compress = ManagerChunk2A::get_chunk2a_to_compress_potentially(thr);

    const auto sz = static_cast<int>(vec_tuple_to_compress.size());

    if (sz == 0)
    {
        return;
    }

    // The body runs in both modes; only the compression_mgr start/finish coordination is
    // OpenMP-specific.
    if constexpr (build_config::threaded)
    {
        if (proxy2a)
        {
            // Per-chunk exclusion (ChunkBase::steal_mark_): claim every candidate BEFORE
            // start(), whose heavy barrier publishes the marks to the writers (its window then
            // waits only for writers whose flag points to a marked chunk). Each chunk is
            // released when the pass is done with it, and all again at the end.
            // A flush in progress makes start() decline: ask first, before marking
            // (block_pass_precheck.h).
            if (block_pass_yields_to_flush<zero_any::Proxy<T>, zero_any::Chunk<T>>(CHUNK_PMP_MAX_THREADS))
            {
                return;
            }
            for (int k = 0; k < sz; ++k)
            {
                if (auto* const chunk_t = std::get<0>(vec_tuple_to_compress[k]))
                {
                    chunk_t->steal_mark();
                }
            }
            const bool ret = proxy2a->compression_mgr.template start<zero_any::Proxy<T>>();
            if (!ret)
            {
                for (int k = 0; k < sz; ++k)
                {
                    if (auto* const chunk_t = std::get<0>(vec_tuple_to_compress[k]))
                    {
                        chunk_t->steal_unmark();
                    }
                }
                return;
            }
            // Check builds: window shut, commit the claims (tested by store_value()'s guard).
            if constexpr (build_config::any_check)
            {
                for (int k = 0; k < sz; ++k)
                {
                    if (auto* const chunk_t = std::get<0>(vec_tuple_to_compress[k]))
                    {
                        chunk_t->steal_commit();
                    }
                }
            }
        }
    }

    VecTupleChunkA2pInt2  ll_compt1;
    VecTupleChunkA2pTint2 ll_compt2;

    ll_compt1.reserve((sz / 50) + 1);
    ll_compt2.reserve((sz / 50) + 1);

    // `claimed`: the claim loop runs only with a proxy (without: the resize path).
    fill_list_chunk_to_compress_in_par(thr, ll_compt1, ll_compt2, proxy2a != nullptr);

    for (auto& tup : ll_compt1)
    {
        const int num_chunk = std::get<1>(tup);
        const int num_tab   = std::get<2>(tup);

        auto* chunk_ptr = std::get<0>(tup);

        // Per-chunk geometry pin -- see this function's header comment.
        if constexpr (build_config::chunk_dynamic)
        {
            shrink_array::set_thread_chunk_size(std::get<3>(tup).count());
        }

        char* ptr_type_cst = &zero_any::Proxy<T>::tab_type_cst_val[num_tab][num_chunk];

        // Long form in both modes: thr is 0 with OpenMP off.
        const SecondValueRef<T> ptr_second_value =
            zero_any::Proxy<T>::BaseSpec::get_tab_second_value(num_tab, num_chunk);
        chunk_ptr->clear_int_st(0, ptr_type_cst, ptr_second_value, thr);
        chunk_ptr->steal_unmark();
    }

    for (const auto& tup : ll_compt2)
    {
        auto [chunk_ptr, second_v, num_chunk, num_tab, csize] = tup;

        // Per-chunk geometry pin -- see this function's header comment.
        if constexpr (build_config::chunk_dynamic)
        {
            shrink_array::set_thread_chunk_size(csize.count());
        }

        if constexpr (Base::is_not_bool)
        {
            if (second_v != T{})
            {
                char* ptr_type_cst = &zero_any::Proxy<T>::tab_type_cst_val[num_tab][num_chunk];

                // find_or_set records the second value in the chunk's slot and returns the
                // shared array; clear_int_st installs it as `values` (the CAS).
                const SecondValueRef<T> ptr_second_value =
                    zero_any::Proxy<T>::BaseSpec::get_tab_second_value(num_tab, num_chunk);

                T* new_cst_val2_ptr = chunk_ptr->Holder::find_or_set_ind_cst_val2(second_v, ptr_second_value);

                chunk_ptr->clear_int_st(1, ptr_type_cst, ptr_second_value, thr, new_cst_val2_ptr);
            }
        }
        else
        {
            // bool: no second-value pointer, ordering issue does not apply
            if (second_v)
            {
                chunk_ptr->Holder::find_or_set_ind_cst_val2();
            }

            char* ptr_type_cst = &zero_any::Proxy<T>::tab_type_cst_val[num_tab][num_chunk];

            const SecondValueRef<T> ptr_second_value =
                zero_any::Proxy<T>::BaseSpec::get_tab_second_value(num_tab, num_chunk);
            chunk_ptr->clear_int_st(1, ptr_type_cst, ptr_second_value, thr);
        }
        chunk_ptr->steal_unmark();
    }

    for (int k = 0; k < sz; ++k)
    {
        const auto& tup       = vec_tuple_to_compress[k];
        const int   num_chunk = std::get<1>(tup);
        // A negative num_chunk is a corrupt entry (purged entries are erased): reported in
        // check builds, skipped otherwise.
        if constexpr (build_config::any_check)
        {
            if (num_chunk < 0) [[unlikely]]
            {
                CHUNK_INVARIANT_FAILED("zero_any block pass: candidate {} of {} has num_chunk={}", k, sz, num_chunk);
            }
        }
        // Hand the candidate back: its mark FIRST, then the dedup flag. While the flag is up
        // the chunk is in no other thread's list, so this unmark only drops our own claim.
        // The other order lets another thread queue the chunk, start its pass and mark it,
        // and this unmark would erase that claim: writers would then write into a chunk
        // that pass is scanning and stealing.
        if (auto* const chunk_t = std::get<0>(tup))
        {
            chunk_t->steal_unmark();
        }
        test_hooks::maybe_pass_delay(); // TEST KNOB (test_hooks.h)
        if constexpr (build_config::any_check)
        {
            if (auto* const marked = std::get<0>(tup); marked != nullptr && marked->steal_marked()) [[unlikely]]
            {
                CHUNK_INVARIANT_FAILED("compression pass: candidate {} was marked while this pass still held its "
                                       "dedup flag -- two passes on one chunk",
                                       static_cast<const void*>(marked));
            }
        }
        if (num_chunk >= 0)
        {
            const int num_tab = std::get<2>(tup);

            zero_any::Proxy<T>::Base::reset_tab_inside_list_to_compress(num_tab, num_chunk);
        }
    }

    vec_tuple_to_compress.clear();
    vec_tuple_to_compress.reserve((sz / 10) + 1);

    if (proxy2a)
    {
        if (proxy2a->is_not_cst_val())
        {
            proxy2a->set_info_on_chunk();
            // Only one thread runs this section at a time, but another may be in
            // add_to_compression_list concurrently.
            proxy2a->add_to_compression_list(vec_tuple_to_compress);
        }

        if constexpr (build_config::threaded)
        {
            proxy2a->compression_mgr.template finish<zero_any::Proxy<T>>();
        }
    }
}

/**
 * @brief Scans one chunk and compresses it in place if it is constant.
 *
 * Single-chunk counterpart of make_block_compression(): same scan, the clear done here.
 *
 * @param thr_rank       Calling thread's rank (OMPT path only).
 * @param num_chunk      Index of this chunk in its array.
 * @param num_tab        Table number of the owning array.
 * @param apply          true: scan, and compress the chunk if constant;
 *                       false: nothing at all in an OpenMP build without OMPT (see the
 *                       body), the scan alone elsewhere.
 * @param size           Geometry of the owning array.
 *
 * @note In an OpenMP build without OMPT the between-region pass is
 *       make_block_compression() (compress_chunk_to_compress); this is what
 *       the rechunk fold calls on a chunk of its own, with apply = true.
 */
template <typename T>
void zero_any::Chunk<T>::compress([[maybe_unused]] int  thr_rank,
                                  int                   num_chunk,
                                  int                   num_tab,
                                  [[maybe_unused]] bool apply,
                                  ChunkElems            size)
{
    // OpenMP without OMPT, apply = false (the lists' purge walks): the body would only scan
    // and register a cst_val2 that make_block_compression redoes -- a large cost for no
    // compression, so skip it. apply = true (rechunk fold, chunk no writer can reach) runs.
    if constexpr (build_config::threaded && chunk_config::not_use_compression_with_ompt)
    {
        if (!apply)
        {
            return;
        }
    }

    // Only a MATERIALISED chunk can be compressed: a queued entry can outlive the chunk's
    // materialisation. Do not test cst_val1 alone: a chunk at cst_val2 would be scanned and
    // its shared second-value block cleared a second time.
    if (this->written())
    {
        // CHUNK_STATS: one candidate scanned (compress/compress_attempt = scan yield).
        increm_chunk_compress_attempt<Chunk>();
        bool compact1 = false;
        bool compact2 = false;

        // new_cst_val2_ptr: result of find_or_set, the array clear_int_st
        // installs as the chunk's values pointer.
        T* new_cst_val2_ptr = nullptr;

        if constexpr (Base::is_not_bool)
        {
            // Against values[0], as fill_list_to_compress does, not the registered second
            // value: a chunk uniform at another value is compressed to THAT value.
            const SecondValueRef<T> ptr_second_value =
                zero_any::Proxy<T>::BaseSpec::get_tab_second_value(num_tab, num_chunk);
            const InT<T> second_v = Base::read_value_scan(0);
            const auto   scan     = Base::scan_constants(first_value, second_v, size.count());
            compact1              = scan.first;
            compact2              = scan.second;
            if (compact2)
            {
                new_cst_val2_ptr = Holder::find_or_set_ind_cst_val2(second_v, ptr_second_value);
            }
        }
        else
        {
            const auto scan = Base::scan_constants(static_cast<T>(false), static_cast<T>(true), size.count());
            compact1        = scan.first;
            compact2        = scan.second;
            if (compact2)
            {
                Holder::find_or_set_ind_cst_val2();
            }
        }

        // Both modes go through the long clear_int_st(): the type-2 clear MUST commit
        // new_cst_val2_ptr first, or init_values_cst_val() points the chunk at a never
        // installed second-value slot (nullptr).
        if (apply)
        {
            const SecondValueRef<T> ptr_second_value =
                zero_any::Proxy<T>::BaseSpec::get_tab_second_value(num_tab, num_chunk);
            char* ptr_type_cst = &zero_any::Proxy<T>::tab_type_cst_val[num_tab][num_chunk];
            if (compact1)
            {
                clear_int_st(0, ptr_type_cst, ptr_second_value, thr_rank);
            }
            if (compact2)
            {
                clear_int_st(1, ptr_type_cst, ptr_second_value, thr_rank, new_cst_val2_ptr);
            }
        }
    }
}

/**
 * @brief Sets every element of this chunk to @p v through its state alone.
 *
 * @details A whole-chunk fill needs no buffer: the chunk ends up a constant.
 *          - materialised: register @p v (unless it is the first value) and
 *            run the compression's own clear (clear_int_st), which frees the
 *            buffer and installs the constant -- what compress() does after a
 *            scan, the scan being pointless here;
 *          - compressed: switch the constant in place. The second-value slot
 *            and the type are written first, then the values pointer: the
 *            order clear_int_st keeps. No CAS, no buffer allocated, filled,
 *            scanned and freed again. Model-checked against the compression pass
 *            (formal/switch_mt.c, docs/formal_models.md 6.1): safe because the switch
 *            only touches a chunk that is not a buffer, which the pass only ever
 *            changes by a CAS; load-bearing is the caller's precondition, that nobody
 *            else materialises this chunk (the span covers all of it): the pointer is
 *            stored, not swapped.
 *          The value the chunk held before stays registered; a chunk
 *          records no reference to it (reclaim_second_values() frees what no
 *          chunk points at any more).
 *
 * @param thr_rank  Rank whose pool takes a released buffer.
 * @param num_chunk Index of this chunk in its array.
 * @param num_tab   Table number of the owning array.
 * @param v         The value of every element.
 */
template <typename T>
void zero_any::Chunk<T>::fill_constant_st(int thr_rank, int num_chunk, int num_tab, InT<T> v)
{
    static_assert(Base::is_not_bool, "fill_constant_st: bool chunks are bit-packed, fill them element by element");

    const SecondValueRef<T> ptr_second_value = zero_any::Proxy<T>::BaseSpec::get_tab_second_value(num_tab, num_chunk);
    char*                   ptr_type_cst     = &zero_any::Proxy<T>::tab_type_cst_val[num_tab][num_chunk];
    const bool              to_first         = (v == static_cast<T>(first_value));

    if (this->written())
    {
        // CHUNK_STATS: a compression reached without a scan, its attempt
        // counted with it (as fill_range does on v1).
        increm_chunk_compress_attempt<Chunk>();
        T* cst2_ptr = to_first ? nullptr : Holder::find_or_set_ind_cst_val2(v, ptr_second_value);
        clear_int_st(to_first ? 0 : 1, ptr_type_cst, ptr_second_value, thr_rank, cst2_ptr);
        return;
    }

    if (to_first)
    {
        if (Base::is_cst_val1())
        {
            return;
        }
        ptr_second_value.reset();
        set_cst_val_to_cst_val1(ptr_type_cst);
        Base::values_store(Base::get_cst_val1());
        return;
    }

    T* cst2_ptr = Holder::find_or_set_ind_cst_val2(v, ptr_second_value);
    if (Base::values_load() == cst2_ptr)
    {
        return; // already at v
    }
    set_cst_val_to_cst_val2(ptr_type_cst);
    Base::values_store(cst2_ptr);
}

/**
 * @brief Compresses every thread's candidate chunks, opening a parallel region if needed.
 *
 * Used between parallel phases (by compress_chunk_to_compress(), which disarms the OMPT
 * callbacks so its own region does not re-enter compression). Without OMPT the per-chunk
 * compress() is a no-op and the pass is make_block_compression() over every rank.
 *
 * @param thr            Rank whose candidate chunks are compressed.
 * @param apply          Forwarded to compress().
 */
template <typename T>
void zero_any::Chunk<T>::compress_sweep_rank(int thr, bool apply)
{
    if constexpr (build_config::threaded && chunk_config::not_use_compression_with_ompt)
    {
        // The only between-region compressor for zero_any in this mode. No writer is live
        // between regions, so the pass runs without handshake (nullptr proxy), each rank on
        // its own list (the second-value registry takes its mutex); ranks beyond the team
        // are done serially (compress_sweep_mops_up), as zero_one_value does.
        if (apply)
        {
            make_block_compression(thr, nullptr);
            return;
        }
    }
    compress_chunk_to_compress_in_par(thr, apply);
}

template <typename T>
void zero_any::Chunk<T>::compress_chunk_to_compress(bool apply)
{
    // Inside a region the calling rank runs compress_chunk_to_compress_in_par; outside,
    // compress_sweep_rank, and the ranks beyond the team only when
    // compress_sweep_mops_up (see ChunkBase::sweep_every_rank).
    Base::sweep_every_rank([apply](int thr) { compress_chunk_to_compress_in_par(thr, apply); },
                           [apply](int thr) { compress_sweep_rank(thr, apply); }, compress_sweep_mops_up(apply));
}

/**
 * @brief Is there any chunk waiting to be compressed?
 *
 * Cheap gate for callers that would otherwise open a parallel region for nothing.
 *
 * @param max_thr Number of per-thread lists to examine (OpenMP builds only).
 * @return True as soon as one list is non-empty.
 */
template <typename T>
bool zero_any::Chunk<T>::have_chunk_to_compress([[maybe_unused]] int max_thr)
{
    const bool is_empty = is_vec_tuple_to_compress_potentially_empty_thr();
    if (is_empty)
    {
        return false;
    }

    if constexpr (build_config::threaded)
    {
        for (int thr = 0; thr < max_thr; ++thr)
        {
            const auto& chunk_t      = get_vec_tuple_to_compress_potentially_thr(thr);
            const bool  is_empty_thr = chunk_t.empty();
            if (!is_empty_thr)
            {
                return true;
            }
        }
    }
    else
    {
        if (!is_empty)
        {
            return true;
        }
    }

    return false;
}

/**
 * @brief Compresses one thread's candidate chunks, from inside a parallel region.
 *
 * Walks the rank's list and calls compress() on each entry, using the geometry
 * each entry carries rather than any ambient size.
 *
 * @param thr            Rank whose list is processed.
 * @param apply          Forwarded to compress().
 *
 * @warning The "already queued" flag reset and the list clear MUST stay under the same
 *          condition (`use_compression_with_ompt`): a flag cleared with the entry still listed
 *          lets a later write queue the chunk twice, and the second clear hits a constant.
 */
template <typename T>
void zero_any::Chunk<T>::compress_chunk_to_compress_in_par([[maybe_unused]] int thr, bool apply)
{
    auto& vec_tup = ManagerChunk2A::get_chunk2a_to_compress_potentially(thr);

    if constexpr (build_config::threaded)
    {
        // The clear reaches free_block_to_pool(), which routes by the ambient
        // thread_chunk_size(); the list mixes arrays (e.g. an array and its rechunk tmp), so
        // pin each chunk to its own csize and restore afterwards, as make_block_compression().
        [[maybe_unused]] const unsigned int saved_sz = shrink_array::thread_chunk_size();

        for (const auto& tup : vec_tup)
        {
            auto* const      chunk_t   = std::get<0>(tup);
            const int        num_chunk = std::get<1>(tup);
            const int        num_tab   = std::get<2>(tup);
            const ChunkElems csize     = std::get<3>(tup);

            if (chunk_t)
            {
                if constexpr (build_config::chunk_dynamic)
                {
                    set_thread_chunk_size(csize.count());
                }

                chunk_t->compress(thr, num_chunk, num_tab, apply, csize);
                // Same condition as the list clear below: see the @warning above.
                if constexpr (chunk_config::use_compression_with_ompt)
                {
                    zero_any::Proxy<T>::set_tab_is_inside_chunk_list_to_compress(num_tab, num_chunk, 0);
                }
            }
        }

        if constexpr (build_config::chunk_dynamic)
        {
            set_thread_chunk_size(saved_sz);
        }
    }
    else
    {
        for (auto& tup : vec_tup)
        {
            const int num_chunk = std::get<1>(tup);
            const int num_tab   = std::get<2>(tup);

            // Per-entry geometry, as above: a default size would under-scan dynamic chunks.
            std::get<0>(tup)->compress(0, num_chunk, num_tab, apply, std::get<3>(tup));
            zero_any::Proxy<T>::set_tab_is_inside_chunk_list_to_compress(num_tab, num_chunk, 0);
        }
    }

    if constexpr (chunk_config::use_compression_with_ompt)
    {
        // clear() keeps the capacity for the next pass.
        vec_tup.clear();
    }
}

/**
 * @brief Scans one thread's candidate list into the two compression lists.
 *
 * Forwards each live entry to fill_list_to_compress(), passing the geometry the
 * entry carries. Null chunk pointers are the batch sentinels and are skipped.
 *
 * @param thr       Rank whose candidate list is scanned.
 * @param ll_compt1 Output: chunks compressible to first_value.
 * @param ll_compt2 Output: chunks compressible to one other value.
 * @param claimed   Whether the caller claimed (steal-marked) the candidates
 *                  first; forwarded to fill_list_to_compress().
 */
template <typename T>
void zero_any::Chunk<T>::fill_list_chunk_to_compress_in_par([[maybe_unused]] int   thr,
                                                            VecTupleChunkA2pInt2&  ll_compt1,
                                                            VecTupleChunkA2pTint2& ll_compt2,
                                                            [[maybe_unused]] bool  claimed)
{
    auto& vec_tup = ManagerChunk2A::get_chunk2a_to_compress_potentially(thr);

    const int sz = vec_tup.size();

    for (int k = 0; k < sz; ++k)
    {
        const auto& tup = vec_tup[k];

        auto* const chunk_t = std::get<0>(tup);
        if (chunk_t)
        {
            const int num_chunk = std::get<1>(tup);
            const int num_tab   = std::get<2>(tup);
            // Per-entry geometry: the chunk size of its owning array.
            const ChunkElems csize = std::get<3>(tup);

            const auto n1 = ll_compt1.size();
            const auto n2 = ll_compt2.size();
            chunk_t->fill_list_to_compress(ll_compt1, ll_compt2, num_chunk, num_tab, csize, claimed);
            // Not selected: nothing will be stolen, release it to the writers now.
            if (ll_compt1.size() == n1 && ll_compt2.size() == n2)
            {
                chunk_t->steal_unmark();
            }
        }
    }
}

// ---------------------------------------------------------------------------
// zero_any::ChunkOmpPolicy<T, Derived, true>: materialise and clear, OpenMP mode.
// ---------------------------------------------------------------------------
/**
 * @brief Materialise-side exclusion against the compressor's steal.
 *
 * make_block_compression() CAS-steals a chunk's `values` and free_block_to_pool()s it;
 * materialising that chunk meanwhile races the free on the same recycled block.
 *
 * The compressor only steals a chunk it has MARKED, holding the mark from before start()
 * to after the CAS, so this chunk's own mark is the exact condition (cheaper than the global
 * is_hard_compression_inflight(), which makes every materialisation spin for a whole pass).
 *
 * A fast-path writer -- its flag raised on THIS chunk, check #2 passed -- does not wait: a
 * mark it sees now was set AFTER its check #2 (set before, the handshake made the writer
 * see it and defer), so the pass that owns it is parked in start(), waiting for this very
 * flag, and steals nothing before end_write(). Waiting here was the writer waiting for the
 * compressor waiting for the writer, resolved only by the compressor's yield budget
 * (CHUNK_COMPRESS_MAX_YIELDS): a stall of both for that whole budget, then an abandoned
 * pass (formal/writer_compressor.c, MUT_MATERIALISE_WAIT_UNDER_FLAG). The callers with no
 * flag keep the wait: alloc_chunk() (pre-allocation, usable from a parallel region) and a
 * flush replay, which never overlaps a stealer anyway (flush_deferred waits
 * stealing_inflight_ out and blocks start()). Never a broad "compression active" as the
 * condition: it also counts flush-draining slow paths, and deadlocks from a flusher's write
 * body. (The serial build has no marks and no concurrent pass: the other branch is its
 * no-op.)
 *
 * @param chunk    The chunk about to be materialised.
 * @param thr_rank The calling thread's rank: whose writer flag is consulted.
 */
template <typename T, typename ChunkT>
void wait_before_materialise(const ChunkT& chunk, [[maybe_unused]] int thr_rank) noexcept
{
    if constexpr (chunk_config::use_chunk_steal_mark)
    {
        using Policy = WriteCompressSyncPolicy<typename ChunkTraits<ChunkT>::ProxyType>;
        if (Policy::is_thread_writing_on(thr_rank, static_cast<const void*>(&chunk)))
        {
            return; // our own write session: the pass holding the mark is parked on our flag
        }
        while (chunk.steal_marked())
        {
            CHUNK_CPU_PAUSE();
        }
    }
    else
    {
        while (CompressionManager<T, ChunkT>::is_hard_compression_inflight())
        {
            CHUNK_CPU_PAUSE();
        }
    }
}

/**
 * @brief Materialises this chunk: allocate a buffer, fill it, publish it by CAS.
 *
 * The buffer is filled BEFORE publication, a single CAS against the constant pointer the
 * chunk holds: a reader sees the old constant or the finished buffer. Losing the CAS means
 * another thread materialised first; apply_cas_or_recup() recycles our buffer.
 *
 * @warning Waits first until no compressor may steal THIS chunk
 *          (wait_before_materialise(); not when the caller's own writer flag is
 *          on the chunk). Never on a broad "compression
 *          active": this can run from a flusher's write body while a slow-path
 *          thread waits on the lock that flusher holds -- a deadlock.
 */
template <typename T, typename Derived, bool Enabled>
void zero_any::ChunkOmpPolicy<T, Derived, Enabled>::create_value(this Derived&     self,
                                                         bool              need_set_second_value,
                                                         SecondValueRef<T> ptr_second_value,
                                                         int               thr_rank)
{
    if constexpr (is_not_bool)
    {
        // Exclude a concurrent steal of this chunk: see wait_before_materialise().
        wait_before_materialise<T>(self, thr_rank);

        const auto init_func = [need_set_second_value, ptr_second_value](T* ptr)
        {
            if (!need_set_second_value)
            {
                Base::init_block(ptr);
            }
            else
            {
                // Object-level fill: safe for any T (complex...).
                std::fill_n(ptr, shrink_array::alloc_chunk_count(), ptr_second_value.get());
            }
        };

        // SecondValueChunkBase spelled out, not Base: `self.Base::` looks Base up in Derived first,
        // where zero_any::Chunk's alias is private (clang rejects it; gcc lets it through).
        T* values_loc = self.SecondValueChunkBase<T, Derived>::allocate_chunk_internal(thr_rank, init_func);

        self.SecondValueChunkBase<T, Derived>::apply_cas_or_recup(values_loc, thr_rank,
                                                        [&self, need_set_second_value](T* cst_ptr)
                                                        {
                                                            // One CAS on the specific constant pointer.
                                                            return self.compare_and_swap(cst_ptr,
                                                                                         need_set_second_value);
                                                        });
    }

    else
    {
        static_assert(false, "create_value(..., second value) is for non-bool elements: use create_value_bool()");
    }
}

/**
 * @brief Materialises a `bool` chunk.
 *
 * Same allocate-fill-CAS shape as create_value(); bool has no per-chunk second value, so
 * the flag doubles as the fill value.
 *
 * @note Same wait_before_materialise() as create_value(): bool chunks are stolen alike.
 */
template <typename T, typename Derived, bool Enabled>
void zero_any::ChunkOmpPolicy<T, Derived, Enabled>::create_value_bool(this Derived& self,
                                                              bool          need_set_second_value,
                                                              int           thr_rank)
{
    if constexpr (is_bool)
    {
        // Same exclusion as create_value(): a bool chunk is CAS-stolen like any other (see
        // wait_before_materialise()).
        wait_before_materialise<T>(self, thr_rank);

        const auto init_func = [need_set_second_value](bool* ptr)
        {
            // The flag is the fill value.
            std::fill_n(ptr, shrink_array::alloc_chunk_count(), need_set_second_value);
        };

        T* values_loc = self.SecondValueChunkBase<T, Derived>::allocate_chunk_internal(thr_rank, init_func);

        self.SecondValueChunkBase<T, Derived>::apply_cas_or_recup(
            values_loc, thr_rank, [&self, need_set_second_value](bool* cst_ptr)
            { return self.compare_and_swap(cst_ptr, need_set_second_value); });
    }
    else
    {
        static_assert(false, "create_value_bool() is for bool elements: use create_value(..., second value)");
    }
}

/// @brief create_value() on the calling thread's rank.
template <typename T, typename Derived, bool Enabled>
void zero_any::ChunkOmpPolicy<T, Derived, Enabled>::create_value(this Derived&     self,
                                                         bool              need_set_second_value,
                                                         SecondValueRef<T> ptr_second_value)
{
    self.create_value(need_set_second_value, ptr_second_value, get_thr_rank_fast());
}

/// @brief create_value_bool() on the calling thread's rank.
template <typename T, typename Derived, bool Enabled>
void zero_any::ChunkOmpPolicy<T, Derived, Enabled>::create_value_bool(this Derived& self, bool need_set_second_value)
{
    self.create_value_bool(need_set_second_value, get_thr_rank_fast());
}

// ---------------------------------------------------------------------------
// zero_any::ChunkOmpPolicy<T, Derived, false>: materialise and clear, serial mode.
// ---------------------------------------------------------------------------

/**
 * @brief Materialises this chunk (serial build): allocate, fill, assign.
 *
 * No CAS and no inflight wait — with nothing else running, the buffer is simply
 * assigned to `values`.
 */
template <typename T, typename Derived>
void zero_any::ChunkOmpPolicy<T, Derived, false>::create_value(this Derived&     self,
                                                       bool              need_set_second_value,
                                                       SecondValueRef<T> ptr_second_value)
{
    const auto init_func = [need_set_second_value, ptr_second_value](T* ptr)
    {
        if (!need_set_second_value)
        {
            Base::init_block(ptr);
        }
        else
        {
            for (unsigned int i = 0; i < alloc_chunk_count(); ++i)
            {
                ptr[i] = ptr_second_value.get();
            }
        }
    };

    self.values_ = self.SecondValueChunkBase<T, Derived>::allocate_chunk_internal(0, init_func);
}

/**
 * @brief Materialises a `bool` chunk (serial build).
 * @param need_set_second_value Fill value — true for the second constant, false for the first.
 */
template <typename T, typename Derived>
void zero_any::ChunkOmpPolicy<T, Derived, false>::create_value_bool(this Derived& self, bool need_set_second_value)
{
    // bool's second value is always `true`: this IS create_value() at that value
    // (Base::init_block() is the `false` fill).
    static T k_second_value = static_cast<T>(true); // a slot the handle can point at; never written
    self.create_value(need_set_second_value, SecondValueRef<T>{&k_second_value});
}

/**
 * @brief Compresses this chunk back to a constant, if it is materialised.
 *
 * @param ptr_type_cst     Slot of the per-chunk type table to update.
 * @param ptr_second_value The chunk's second-value slot.
 * @param thr_rank         Rank whose pool receives the released buffer.
 */
template <typename T, typename Derived, bool Enabled>
void zero_any::ChunkOmpPolicy<T, Derived, Enabled>::clear(this Derived&     self,
                                                  char*             ptr_type_cst,
                                                  SecondValueRef<T> ptr_second_value,
                                                  int               thr_rank)
{
    self.SecondValueChunkBase<T, Derived>::clear_impl(std::nullopt, thr_rank, ptr_type_cst, ptr_second_value);
}

/**
 * @brief Compresses this chunk to the FIRST constant and forgets its second value.
 *
 * Serial-build variant used at teardown: releases the buffer, resets the
 * chunk's second value to zero, and points it back at the first constant.
 *
 * @param ptr_type_cst     Slot of the per-chunk type table to update.
 * @param ptr_second_value The chunk's second-value slot, reset to `T{}`.
 */
template <typename T, typename Derived>
void zero_any::ChunkOmpPolicy<T, Derived, false>::clear(this Derived&     self,
                                                char*             ptr_type_cst,
                                                SecondValueRef<T> ptr_second_value)
{
    if (self.written())
    {
        self.SecondValueChunkBase<T, Derived>::free_values(self.values_);

        ptr_second_value.reset();
        Derived::set_cst_val_to_cst_val1(ptr_type_cst);
        self.init_values_cst_val(ptr_type_cst, ptr_second_value);
    }
}

/**
 * @brief Unconditional twin of clear(): compresses without testing `written()`.
 *
 * For callers that have already established the chunk is materialised; the test
 * is kept as a debug-only assertion.
 *
 * @param ptr_type_cst     Slot of the per-chunk type table to update.
 * @param ptr_second_value The chunk's second-value slot, reset to `T{}`.
 */
template <typename T, typename Derived>
void zero_any::ChunkOmpPolicy<T, Derived, false>::clear_st(this Derived&     self,
                                                   char*             ptr_type_cst,
                                                   SecondValueRef<T> ptr_second_value)
{
    self.SecondValueChunkBase<T, Derived>::debug_check_written();

    self.SecondValueChunkBase<T, Derived>::free_values(self.values_);

    ptr_second_value.reset();
    Derived::set_cst_val_to_cst_val1(ptr_type_cst);
    self.init_values_cst_val(ptr_type_cst, ptr_second_value);
}

/**
 * @brief Releases this chunk's buffer and points it back at a constant.
 *
 * The single point every compression path funnels through (both modes, every variant):
 * the "block compressed" statistic is counted here and nowhere else.
 */
template <typename T, typename Derived, bool Enabled>
void zero_any::ChunkOmpPolicy<T, Derived, Enabled>::block_clear(this Derived&        self,
                                                        int                  num_compress_value,
                                                        char*                ptr_type_cst,
                                                        SecondValueRef<T>    ptr_second_value,
                                                        [[maybe_unused]] int thr_rank,
                                                        T*                   cst2_ptr)
{
    // CHUNK_STATS: one block released (see the function comment).
    increm_chunk_compress<Derived>();
    // Bytes released, as zero_one_value counts them (with bytes_alloc/bytes_reused: live buffers).
    add_chunk_bytes_freed<Derived>(static_cast<long>(sizeof(T)) * static_cast<long>(shrink_array::alloc_chunk_count()));
    self.SecondValueChunkBase<T, Derived>::block_clear_common(num_compress_value, 0, thr_rank, ptr_type_cst, ptr_second_value,
                                                    cst2_ptr);
}

/**
 * @brief Compresses this chunk, committing its second-value pointer first.
 *
 * When compressing to the second constant, the registered pointer is resolved BEFORE
 * block_clear(), which installs it; otherwise it would pick up a stale one.
 */
template <typename T, typename Derived, bool Enabled>
void zero_any::ChunkOmpPolicy<T, Derived, Enabled>::clear_int_st(this Derived&     self,
                                                         int               num_compress_value,
                                                         char*             ptr_type_cst,
                                                         SecondValueRef<T> ptr_second_value,
                                                         int               thr_rank,
                                                         T*                new_cst_val2_ptr)
{
    self.SecondValueChunkBase<T, Derived>::check_not_constant_before_clear(num_compress_value, std::nullopt);

    // Type-2 array: the caller's, or the one registered for the second value (bool path).
    if (num_compress_value == 1 && new_cst_val2_ptr == nullptr)
    {
        new_cst_val2_ptr = Derived::Holder::find_cst_val2_ptr(ptr_second_value.get());
    }

    self.block_clear(num_compress_value, ptr_type_cst, ptr_second_value, thr_rank, new_cst_val2_ptr);
}

/// @brief Serial block_clear(): frees the buffer outright instead of parking it
///        in a rank pool, and points the chunk at a constant.
/// @param num_compress_value 0 for the first constant, otherwise the second.
/// @param ptr_type_cst       Slot of the per-chunk type table to update.
/// @param ptr_second_value   The chunk's second-value slot.
/// @param cst2_ptr           Second-value slot of the constant (type 2).
/// @note The rank argument is unused: there is no rank pool serially.
template <typename T, typename Derived>
void zero_any::ChunkOmpPolicy<T, Derived, false>::block_clear(this Derived&     self,
                                                      int               num_compress_value,
                                                      char*             ptr_type_cst,
                                                      SecondValueRef<T> ptr_second_value,
                                                      int /*thr_rank*/,
                                                      T* cst2_ptr)
{
    // As try_mark_cst_and_get_old() on the OpenMP side: a chunk already at a constant has no
    // buffer; freeing that pointer corrupts the allocator (or faults). The type table already
    // describes the constant in place.
    if (self.values_ == Base::get_cst_val1() || Derived::Holder::is_cst_val2_ptr(self.values_)) [[unlikely]]
    {
        return;
    }

    increm_chunk_compress<Derived>();
    self.SecondValueChunkBase<T, Derived>::free_values(self.values_);
    Derived::set_cst_val_to_cst_val(num_compress_value, ptr_type_cst);

    // Serial: no CAS; the caller's array for a type 2, from the tables otherwise.
    if (num_compress_value == 1 && cst2_ptr != nullptr)
    {
        self.values_ = cst2_ptr;
    }
    else
    {
        self.init_values_cst_val(ptr_type_cst, ptr_second_value);
    }
}

/**
 * @brief Serial twin of the five-argument clear_int_st(): same commit-then-block_clear
 *        order. thr_rank is 0 here.
 * @copydetails zero_any::ChunkOmpPolicy::clear_int_st
 */
template <typename T, typename Derived>
void zero_any::ChunkOmpPolicy<T, Derived, false>::clear_int_st(this Derived&     self,
                                                       int               num_compress_value,
                                                       char*             ptr_type_cst,
                                                       SecondValueRef<T> ptr_second_value,
                                                       int               thr_rank,
                                                       T*                new_cst_val2_ptr)
{
    self.SecondValueChunkBase<T, Derived>::check_not_constant_before_clear(num_compress_value, std::nullopt);
    if (num_compress_value == 1 && new_cst_val2_ptr == nullptr)
    {
        new_cst_val2_ptr = Derived::Holder::find_cst_val2_ptr(ptr_second_value.get());
    }
    self.block_clear(num_compress_value, ptr_type_cst, ptr_second_value, thr_rank, new_cst_val2_ptr);
}

// clear_st(num_compress_value, ptr_type_cst, ptr_second_value), which supplies the calling
// rank, is defined in-class on the OpenMP zero_any::ChunkOmpPolicy specialisation.

} // namespace shrink_array