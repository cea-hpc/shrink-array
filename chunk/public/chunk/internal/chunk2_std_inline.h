
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

#include "chunk/PMP_CHUNK.h"
#include "chunk/chunk2_std.h"
#include "chunk/internal/proxy2.h"

/**
 * @file chunk2_std_inline.h
 * @brief Out-of-line definitions of Chunk2's policies: materialisation and compression.
 *
 * Chunk2 is the two-constant chunk whose second value is shared by the WHOLE
 * array — that is what separates it from Chunk2A, where each chunk carries its
 * own. The consequence runs through this entire file: the candidate lists are
 * keyed BY second value (`map_chunk2_to_compress_potentially_thr[thr][value]`),
 * so a compression pass is always "drain the list of one (rank, second value)
 * pair", never "drain everything this rank queued".
 *
 * A chunk is in one of three states — compressed to first_value, compressed to
 * the array's second value, or materialised on a private heap buffer — and the
 * two directions between them are:
 *  - **materialise**: create_value() / create_value_bool() allocate a buffer,
 *    pre-fill it, and (under OpenMP) publish it with a CAS that refuses a
 *    wrong-side transition after a cst_val1/cst_val2 race;
 *  - **compress**: the clear / block_clear / clear_int_st family releases the
 *    buffer and points `values` back at a shared constant.
 *
 * Three things to keep in mind when reading:
 *
 *  - **Nearly everything exists twice over `bool`.** bool has no room for a
 *    distinct second value, so its second constant is `true` with no index, and
 *    it gets its own `_bool` twin of almost every function rather than a
 *    parameter. The two are kept in step by hand.
 *  - **And twice again over OpenMP**, where the serial twin takes a genuinely
 *    different path: no CAS, no rank, and the element counters
 *    (`use_compression_with_counter`) to reset, which the parallel path does not
 *    maintain. The two live on the two specialisations of Chunk2OmpPolicy
 *    (build_config::use_openmp); the queue-driven compression is on
 *    Chunk2FastCompPolicy<..., true>.
 *  - **Geometry comes from the queued entry, never from the ambient size.**
 *    Every candidate tuple carries the chunk count of its owning array
 *    (field `<3>`), because the thread consuming the entry may have a different
 *    ambient size — after a rechunk, scanning at the ambient size overruns a
 *    buffer allocated at another geometry.
 *
 * @see chunk2_std.h for the declarations, chunk_base.h for the shared
 *      allocate/free machinery, proxy2_inline.h for the write paths that queue
 *      the candidates.
 */

namespace shrink_array
{

// ---------------------------------------------------------------------------
// Chunk2FastCompPolicy<T, Derived, true>: the queue-driven compression.
// ---------------------------------------------------------------------------

/**
 * @brief Scans this chunk and files it into one of the two compression lists.
 *
 * Thin wrapper over fill_list_to_compress_impl(): it unpacks the array-wide
 * second value from @p inf_array and supplies the current cst_val2 test.
 *
 * @param ll_compt1  Output list of chunks compressible to first_value.
 * @param ll_compt2  Output list of chunks compressible to the second value.
 * @param inf_array  (second_value, index, ...) of the owning array.
 * @param num_chunk  Index of this chunk in its array.
 * @param num_tab    Table number of the owning array.
 * @param n_elems    This chunk's own element count, captured when it was queued.
 */
template <typename T, typename Derived, bool Enabled>
void Chunk2FastCompPolicy<T, Derived, Enabled>::fill_list_to_compress(vec_tuple_chunk2p_int2& ll_compt1,
                                      vec_tuple_chunk2p_int2& ll_compt2,
                                      const inf_array_t&      inf_array,
                                      int                     num_chunk,
                                      int                     num_tab,
                                      unsigned int            n_elems)
{
    // in_t: a reference into the (const, caller-owned) tuple when T is large.
    const in_t<T> second_v = std::get<0>(inf_array);
    fill_list_to_compress_impl(
        ll_compt1, ll_compt2, num_chunk, num_tab, [&second_v]() -> in_t<T> { return second_v; },
        derived().is_not_cst_val2(std::get<1>(inf_array)), n_elems);
}

/**
 * @brief Bool twin of fill_list_to_compress().
 *
 * The second value is `true` with no index to look up, so the value supplier is
 * a constant and no @c inf_array is needed.
 *
 * @param ll_compt1  Output list of chunks compressible to first_value.
 * @param ll_compt2  Output list of chunks compressible to `true`.
 * @param num_chunk  Index of this chunk in its array.
 * @param num_tab    Table number of the owning array.
 * @param n_elems    This chunk's own element count, captured when it was queued.
 */
template <typename T, typename Derived, bool Enabled>
void Chunk2FastCompPolicy<T, Derived, Enabled>::fill_list_to_compress_bool(vec_tuple_chunk2p_int2& ll_compt1,
                                           vec_tuple_chunk2p_int2& ll_compt2,
                                           int                     num_chunk,
                                           int                     num_tab,
                                           unsigned int            n_elems)
{
    fill_list_to_compress_impl(ll_compt1, ll_compt2, num_chunk, num_tab, []() { return true; }, derived().is_not_cst_val2(),
                               n_elems);
}

/**
 * @brief Shared body of the two fill_list_to_compress entry points.
 *
 * Reads the chunk once, tracking two hypotheses at the same time — "every
 * element is first_value" and "every element is the second value" — and stops
 * as soon as both are disproved. A constant run of first_value satisfies both;
 * it is filed as compact1 only.
 *
 * Chunks that are already compressed (to either constant) are skipped: there is
 * nothing left to release.
 *
 * @tparam SecondValueFunc Nullary callable returning the second value.
 * @param ll_compt1        Output list of chunks compressible to first_value.
 * @param ll_compt2        Output list of chunks compressible to the second value.
 * @param num_chunk        Index of this chunk in its array.
 * @param num_tab          Table number of the owning array.
 * @param get_second_value Supplies the second value; deferred so the bool twin
 *                         can pass a constant and skip the tuple lookup.
 * @param check_cst_val2   True when the chunk is NOT currently compressed to its
 *                         second value — i.e. when the scan is worth doing. The
 *                         name reads as the opposite of what it holds.
 * @param n_elems          This chunk's own element count, captured at enqueue.
 */
template <typename T, typename Derived, bool Enabled>
template <typename SecondValueFunc>
void Chunk2FastCompPolicy<T, Derived, Enabled>::fill_list_to_compress_impl(vec_tuple_chunk2p_int2& ll_compt1,
                                           vec_tuple_chunk2p_int2& ll_compt2,
                                           int                     num_chunk,
                                           int                     num_tab,
                                           SecondValueFunc&&       get_second_value,
                                           bool                    check_cst_val2,
                                           unsigned int            n_elems)
{
    // Checks whether the block points to the one holding the first values,
    // or to the one holding the second values (in which case, no further action is required)
    // The is_not_cst_val2 argument is used to access the array for the second compression value

    if (Base::is_not_cst_val1() && check_cst_val2)
    {
        // CHUNK_STATS: one compression candidate scanned (see chunk2a).
        increm_chunk_compress_attempt<Derived>();

        // n_elems: see Chunk2<T>::compress() -- the chunk's own geometry,
        // captured at enqueue, never the ambient alloc_chunk_count().
        const auto scan = Base::scan_constants(first_value, get_second_value(), n_elems);

        if (scan.first)
        {
            ll_compt1.emplace_back(std::make_tuple(&derived(), num_chunk, num_tab, n_elems));
        }
        if (scan.second)
        {
            ll_compt2.emplace_back(std::make_tuple(&derived(), num_chunk, num_tab, n_elems));
        }
    }
}

/**
 * @brief Scans one chunk and compresses it in place if it is constant.
 *
 * Single-chunk counterpart of the make_block_compression() sweep: the same
 * two-hypothesis scan, but the clear happens here rather than being collected
 * into a list. When the array's second value equals first_value the two
 * hypotheses coincide and a constant run is reported as the first only.
 *
 * @tparam SecondValueFunc Nullary callable returning the array's second value.
 * @tparam ClearFunc       Callable `(int num_compress_value)` applying the clear.
 * @param apply            true: compress a constant chunk; false: scan only.
 * @param n_elems          This chunk's own element count, captured at enqueue.
 * @param get_second_value Supplies the second value (a constant for the bool twin).
 * @param clear_one        Clears to the first (0) or second (1) constant.
 *
 * @warning The clear only runs when @p apply is true. Called with false,
 *          this scans the whole chunk and discards the verdict -- it
 *          compresses nothing (the flag was named apply, and
 *          compress_int's drain passed false for a year believing it
 *          compressed).
 */
template <typename T, typename Derived, bool Enabled>
template <typename SecondValueFunc, typename ClearFunc>
void Chunk2FastCompPolicy<T, Derived, Enabled>::compress_impl(bool apply, unsigned int n_elems, SecondValueFunc get_second_value,
                              ClearFunc&& clear_one)
{
    if (Base::is_not_cst_val1())
    {
        // CHUNK_STATS: this is a SECOND compression-candidate scan, independent of
        // fill_list_to_compress (it scans values[] itself, just below, then calls
        // clear_int_st). Counting only the other one left attempts at 0 while
        // compress was non-zero. Its scan reads values[] directly, so they never
        // inflated `read` and need no read_value_scan routing.
        increm_chunk_compress_attempt<Derived>();

        // n_elems is THIS chunk's own geometry, captured at enqueue (tuple
        // field <3>), not alloc_chunk_count() -- the consuming thread's
        // ambient size, which after a rechunk overflows a buffer allocated
        // at a different geometry. Same fix as Chunk v1 and Chunk2A.
        //
        // Always scanned. The scan used to be skipped when the array's second
        // value equalled first_value, which left BOTH verdicts at their initial
        // `true` and, from an OMPT callback, cleared the chunk twice without
        // having looked at it. scan_constants reports such a run as `first`
        // only.
        const auto scan = Base::scan_constants(first_value, get_second_value(), n_elems);

        if (apply)
        {
            if (scan.first)
            {
                clear_one(0);
            }
            if (scan.second)
            {
                clear_one(1);
            }
        }
    }
}

template <typename T, typename Derived, bool Enabled>
void Chunk2FastCompPolicy<T, Derived, Enabled>::compress([[maybe_unused]] int thr_rank, const inf_array_t& inf_array, bool apply,
                         unsigned int n_elems)
{
    // Only a MATERIALISED chunk can be compressed. A queued entry can outlive
    // the chunk's materialisation: make_block_compression (or a previous
    // drain) compresses it, and the OMPT end-of-region drain then replays
    // the stale entry. compress_impl only refuses cst_val1; a chunk already
    // at cst_val2 passed, its scan read the SHARED second-value block, found
    // it constant, and cleared it a second time -- freeing that shared block
    // in release builds (rechunk_validation on clang/OMPT: values wrong
    // before any re-chunk). fill_list_to_compress_impl always tested both.
    if (!derived().written(std::get<1>(inf_array)))
        return;

    compress_impl(
        apply, n_elems, [&inf_array]() { return std::get<0>(inf_array); },
        [this, thr_rank, &inf_array](int num_compress_value)
        { this->clear_int_st(num_compress_value, std::get<1>(inf_array), thr_rank); });
}

/**
 * @brief Bool twin of compress().
 *
 * @param thr_rank       Rank whose pool receives the released buffer.
 * @param apply          true: compress the chunk if constant; false: scan only.
 * @param n_elems        This chunk's own element count, captured at enqueue.
 *
 * @warning Same as compress(): with @p apply false the scan runs and the
 *          verdict is dropped.
 */
template <typename T, typename Derived, bool Enabled>
void Chunk2FastCompPolicy<T, Derived, Enabled>::compress_bool([[maybe_unused]] int thr_rank, bool apply, unsigned int n_elems)
{
    if constexpr (is_bool)
    {
        if (!derived().written()) // see compress(): a stale entry for a chunk already at a constant
            return;
        compress_impl(apply, n_elems, []() { return true; },
                      [this, thr_rank](int num_compress_value) { this->clear_int_st_bool(num_compress_value, thr_rank); });
    }
}

/**
 * @brief Compresses the candidates of one second value, across every rank.
 *
 * Entry point used between parallel phases. The OMPT callbacks are disarmed for
 * the duration, so nothing re-enters the compression machinery underneath.
 *
 * @param inf_array      (second_value, index, ...) selecting which candidate
 *                       lists to drain.
 * @param apply          Forwarded to compress().
 *
 * @note Like the Chunk2A equivalent: outside a parallel region it opens one and
 *       lets each rank sweep its own list (serial mop-up for ranks beyond the
 *       team size); inside a region it sweeps the calling rank only.
 */
template <typename T, typename Derived, bool Enabled>
void Chunk2FastCompPolicy<T, Derived, Enabled>::compress_chunk_to_compress(const inf_array_t& inf_array, bool apply)
{
    chunk_config::OMPTcc<chunk_config::has_ompt>::update(false);

    if (CHUNK_PMP_IN_PARALLEL_REGION)
    {
        compress_chunk_to_compress_in_par(CHUNK_PMP_THREAD_RANK, inf_array, apply);
    }
    else
    {
        // Same shape as Chunk2A::compress_chunk_to_compress(): one region, each
        // rank sweeps ITS OWN list. Everything the per-rank body touches is
        // rank-local -- map_chunk2_to_compress_potentially_thr[thr_rank], the
        // freed buffer goes to thr_rank's pool (clear_int_st), and the
        // candidate flag reset is an atomic store -- so the ranks are
        // independent. This used to be a serial loop over CHUNK_PMP_MAX_THREADS
        // on the calling thread; after a re-chunk on the full bench (64
        // threads, 15.6M chunks, most of them queued by the refill) that loop
        // was where the "re-chunk iteration" spent its time, one rank at a
        // time while 63 waited.
        int team = 0;
#pragma omp parallel
        {
            const int thr = CHUNK_PMP_THREAD_RANK;
            if (thr == 0) // one writer; `master` is deprecated since OpenMP 5.1
                team = CHUNK_PMP_NUM_THREADS;
            compress_chunk_to_compress_in_par(thr, inf_array, apply);
        }
        // Lists are sized by CHUNK_PMP_MAX_THREADS; a team smaller than that
        // leaves the upper ranks unswept. Serial mop-up, empty lists are cheap.
        for (int thr_rank = team; thr_rank < CHUNK_PMP_MAX_THREADS; ++thr_rank)
            compress_chunk_to_compress_in_par(thr_rank, inf_array, apply);
    }

    chunk_config::OMPTcc<chunk_config::has_ompt>::update(true);
}

/**
 * @brief Bool twin of compress_chunk_to_compress().
 * @param apply          Forwarded to compress_bool().
 */
template <typename T, typename Derived, bool Enabled>
void Chunk2FastCompPolicy<T, Derived, Enabled>::compress_chunk_to_compress_bool(bool apply)
{
    if constexpr (is_bool)
    {
        chunk_config::OMPTcc<chunk_config::has_ompt>::update(false);

        if (CHUNK_PMP_IN_PARALLEL_REGION)
        {
            compress_chunk_to_compress_bool_in_par(CHUNK_PMP_THREAD_RANK, apply);
        }
        else
        {
            // Same as compress_chunk_to_compress(): one region, rank-local sweeps.
            int team = 0;
#pragma omp parallel
            {
                const int thr = CHUNK_PMP_THREAD_RANK;
                if (thr == 0)
                    team = CHUNK_PMP_NUM_THREADS;
                compress_chunk_to_compress_bool_in_par(thr, apply);
            }
            for (int thr_rank = team; thr_rank < CHUNK_PMP_MAX_THREADS; ++thr_rank)
                compress_chunk_to_compress_bool_in_par(thr_rank, apply);
        }
        chunk_config::OMPTcc<chunk_config::has_ompt>::update(true);
    }
}

/**
 * @brief Compresses one rank's candidates for one key, then releases them.
 *
 * Shared body of the two entry points below: they differ only in the key that
 * indexes the candidate map and in which compress overload each chunk gets, so
 * those are the two things the caller supplies. Null chunk pointers are batch
 * sentinels and are skipped; each live entry is compressed at the geometry it
 * carries. The list is emptied afterwards and re-reserved to a tenth of its size.
 *
 * ORDER -- compress FIRST, clear the "already queued" flag SECOND.
 *
 * That flag is the CAS gate of ProxyBase::add_to_compression_list(): a 0 -> 1
 * flip is what admits a chunk to the writing thread's candidate list, and it is
 * the only thing keeping the same chunk out of two lists at once. Clearing it
 * before compress() re-opens the gate while the chunk is still being compressed,
 * so a concurrent write re-queues the SAME chunk, and the drain that later
 * reaches that second entry calls compress() on a chunk that is already
 * constant -- the double-clear Chunk2A carries a @warning about
 * (check_not_constant_before_clear fires: Values == Cst1). Compressing first
 * closes the window: the flag stays 1 for the whole compression, every
 * concurrent write loses the CAS, and re-queueing only becomes possible once
 * the chunk has reached its final state.
 *
 * The two entry points used to disagree on this (the bool twin compressed
 * first, the other reset first). Today the drain always runs at a point with no
 * concurrent writers -- on_parallel_end() puts an `omp barrier` in front of it,
 * and the ArrayChunk2 call sites are gated on !CHUNK_PMP_IN_PARALLEL_REGION --
 * so both orders happen to produce the same result. This is the one that stays
 * correct if that precondition ever slips, and the one Chunk2A settled on after
 * the double-clear was diagnosed there.
 *
 * @tparam CompressFn callable `(chunk_type*, unsigned int n_elems)`.
 * @param thr_rank     Rank whose list is drained.
 * @param key          Second value indexing the per-rank candidate map.
 * @param compress_one Per-chunk compression.
 */
template <typename T, typename Derived, bool Enabled>
template <typename CompressFn>
void Chunk2FastCompPolicy<T, Derived, Enabled>::compress_chunk_to_compress_in_par_impl(int thr_rank, in_t<T> key, CompressFn&& compress_one)
{
    // The store is a per-rank VECTOR of maps, sized by resize_all_list_block_free_thr();
    // until that has run, [thr_rank] is out of bounds -- silent UB in a normal
    // build, an _GLIBCXX_ASSERTIONS abort otherwise. Same guard as the reuse-list
    // lookups (ed84ebf4). It was on the bool path only, although both paths index
    // the same store. Probing rank 0 is enough: a second value is registered for
    // every rank in one loop (ArrayChunk2::add_second_value), so an empty map at
    // rank 0 means no key exists at any rank.
    if (is_vec_tuple_to_compress_potentially_empty_thr())
        return;

    auto& vec_tuple = *(manager_chunk2::map_chunk2_to_compress_potentially_thr[thr_rank][key]);

    // Geometry: this list mixes chunks of every array sharing `key` -- e.g. an
    // array and the tmp rechunk_to() builds from it, at two chunk sizes. The
    // scan below reads n_elems explicitly, but the clear it triggers reaches
    // free_block_to_pool(), which routes by the ambient thread_chunk_size()
    // (reuse list AND allocator). Pin each chunk to its own size, restore
    // afterwards so the OMPT drain leaves the thread as it found it. Without
    // this, the 4096-wide blocks of a rechunked "all-same" array were parked
    // in the @512 list on the master thread and the next 512-array popped
    // them (rechunk_validation PRE-FAIL on clang/OMPT). Same pin as
    // make_block_compression_impl() and Chunk2A<T>::make_block_compression().
    [[maybe_unused]] const unsigned int saved_sz = shrink_array::thread_chunk_size();

    for (auto& tup : vec_tuple)
    {
        auto* chunk_t = std::get<0>(tup);
        if (chunk_t)
        {
            const int num_chunk = std::get<1>(tup);
            const int num_tab   = std::get<2>(tup);

            if constexpr (build_config::chunk_dynamic)
                shrink_array::set_thread_chunk_size(std::get<3>(tup).count());

            compress_one(chunk_t, std::get<3>(tup).count());

            // AFTER the compression, never before -- see the ORDER note above.
            Proxy2<T>::reset_tab_is_inside_list_to_compress(num_tab, num_chunk);
        }
    }

    if constexpr (build_config::chunk_dynamic)
        shrink_array::set_thread_chunk_size(saved_sz);

    const size_t sz = vec_tuple.size();
    if constexpr (chunk_config::use_compression_with_ompt)
        manager_chunk2::old_size_chunk2_to_compress_potentially[thr_rank] = sz;
    vec_tuple.clear();
    vec_tuple.reserve(sz / 10);
}

/**
 * @brief Compresses one rank's candidates for one second value.
 * @param thr_rank       Rank whose list is drained.
 * @param inf_array      (second_value, index, ...) selecting the list.
 * @param apply          Forwarded to compress().
 */
template <typename T, typename Derived, bool Enabled>
void Chunk2FastCompPolicy<T, Derived, Enabled>::compress_chunk_to_compress_in_par(int thr_rank, const inf_array_t& inf_array, bool apply)
{
    compress_chunk_to_compress_in_par_impl(thr_rank, std::get<0>(inf_array),
                                           [thr_rank, &inf_array, apply](auto* chunk_t, unsigned int n_elems)
                                           { chunk_t->compress(thr_rank, inf_array, apply, n_elems); });
}

/**
 * @brief Bool twin of compress_chunk_to_compress_in_par().
 * @param thr_rank       Rank whose list is drained.
 * @param apply          Forwarded to compress_bool().
 */
template <typename T, typename Derived, bool Enabled>
void Chunk2FastCompPolicy<T, Derived, Enabled>::compress_chunk_to_compress_bool_in_par(int thr_rank, bool apply)
{
    if constexpr (is_bool)
    {
        compress_chunk_to_compress_in_par_impl(thr_rank, true,
                                               [thr_rank, apply](auto* chunk_t, unsigned int n_elems)
                                               { chunk_t->compress_bool(thr_rank, apply, n_elems); });
    }
}

/**
 * @brief Scans one rank's candidates for one key into the two compression lists.
 *
 * Shared body of the two entry points below: they differ only in which key
 * indexes the candidate map and which per-chunk scan is run, so those are the
 * two things the caller supplies. Null chunk pointers are batch sentinels and
 * are skipped; each live entry is scanned at the geometry it carries.
 *
 * @tparam ScanFn callable `(chunk_type*, int num_chunk, int num_tab, ChunkElems)`.
 * @param thr_rank Rank whose candidate list is scanned.
 * @param key      Second value indexing the per-rank candidate map.
 * @param scan_one Per-chunk scan.
 */
template <typename T, typename Derived, bool Enabled>
template <typename ScanFn>
void Chunk2FastCompPolicy<T, Derived, Enabled>::fill_list_chunk_to_compress_in_par_impl(int thr_rank, in_t<T> key, ScanFn&& scan_one)
{
    const auto& vec_tuple = *(manager_chunk2::map_chunk2_to_compress_potentially_thr[thr_rank][key]);

    const int nn = vec_tuple.size();

    for (int k = 0; k < nn; ++k)
    {
        const auto& tup     = vec_tuple[k];
        const auto& chunk_t = std::get<0>(tup);
        if (chunk_t)
        {
            const int& num_chunk = std::get<1>(tup);
            const int  num_tab   = std::get<2>(tup);
            // Per-entry geometry (see chunk2_std.h): in dynamic builds each
            // queued chunk carries the chunk size of its owning array.
            scan_one(chunk_t, num_chunk, num_tab, std::get<3>(tup).count());
        }
    }
}

/**
 * @brief Scans one rank's candidates for one second value into the two lists.
 * @param thr_rank  Rank whose candidate list is scanned.
 * @param ll_compt1 Output: chunks compressible to first_value.
 * @param ll_compt2 Output: chunks compressible to the second value.
 * @param inf_array (second_value, index, ...) selecting the list.
 */
template <typename T, typename Derived, bool Enabled>
void Chunk2FastCompPolicy<T, Derived, Enabled>::fill_list_chunk_to_compress_in_par(int                     thr_rank,
                                                   vec_tuple_chunk2p_int2& ll_compt1,
                                                   vec_tuple_chunk2p_int2& ll_compt2,
                                                   const inf_array_t&      inf_array)
{
    fill_list_chunk_to_compress_in_par_impl(
        thr_rank, std::get<0>(inf_array),
        [&ll_compt1, &ll_compt2, &inf_array](auto* chunk_t, int num_chunk, int num_tab, unsigned int n_elems)
        {
            const auto n1 = ll_compt1.size();
            const auto n2 = ll_compt2.size();
            chunk_t->fill_list_to_compress(ll_compt1, ll_compt2, inf_array, num_chunk, num_tab, n_elems);
            // Not selected by the scan: the writers may have it back now.
            if (ll_compt1.size() == n1 && ll_compt2.size() == n2)
                chunk_t->steal_unmark();
        });
}

/**
 * @brief Bool twin of fill_list_chunk_to_compress_in_par().
 * @param thr_rank  Rank whose candidate list is scanned.
 * @param ll_compt1 Output: chunks compressible to first_value.
 * @param ll_compt2 Output: chunks compressible to `true`.
 */
template <typename T, typename Derived, bool Enabled>
void Chunk2FastCompPolicy<T, Derived, Enabled>::fill_list_chunk_to_compress_bool_in_par(int                     thr_rank,
                                                        vec_tuple_chunk2p_int2& ll_compt1,
                                                        vec_tuple_chunk2p_int2& ll_compt2)
{
    fill_list_chunk_to_compress_in_par_impl(
        thr_rank, static_cast<T>(true),
        [&ll_compt1, &ll_compt2](auto* chunk_t, int num_chunk, int num_tab, unsigned int n_elems)
        {
            const auto n1 = ll_compt1.size();
            const auto n2 = ll_compt2.size();
            chunk_t->fill_list_to_compress_bool(ll_compt1, ll_compt2, num_chunk, num_tab, n_elems);
            if (ll_compt1.size() == n1 && ll_compt2.size() == n2)
                chunk_t->steal_unmark();
        });
}

// --- The sweep. Precedent: ChunkFastCompPolicy<T,Derived,Enabled>::
// make_block_compression for plain Chunk<T>.

// make_block_compression_impl (declared in chunk2_std.h). Shape: pin geometry ->
// take the compression handshake -> scan into the two buckets -> clear each ->
// reset every queued flag -> empty the list -> re-queue the triggering chunk.
// The two callables are the only bool/non-bool difference, which is why the
// whole sweep exists once.
// Geometry: the candidate list of one (rank, second_value) pair mixes chunks
// from EVERY array sharing that second value -- e.g. an array and the tmp
// rechunk_to() builds from it, at two different chunk sizes. The clear below
// reaches free_block_to_pool(), which routes by the ambient thread_chunk_size()
// (reuse list AND allocator), so each chunk is pinned to its own n_elems right
// before its clear, exactly as Chunk2A<T>::make_block_compression() does. A
// single pin on the triggering proxy's size (kept, for the handshake) parked
// the other array's buffers under the wrong key: under OMPT, the 4096-wide
// blocks of a rechunked "all-same" array landed in the @512 list and the next
// 512-array read a foreign block (rechunk_validation PRE-FAIL).
template <typename T, typename Derived, bool Enabled>
template <typename FillFunc, typename ClearFunc>
void Chunk2FastCompPolicy<T, Derived, Enabled>::make_block_compression_impl(int         thr_rank,
                                            T           second_value,
                                            Proxy2<T>*  proxy2,
                                            FillFunc&&  fill_func,
                                            ClearFunc&& clear_func)
{
    // FIX (SIGABRT sur un tableau neuf): find, PAS operator[]. La table est un
    // map<T, shared_ptr<...>> par rang ; operator[] sur une cle absente y insere
    // un shared_ptr NUL, que le dereferencement suivant faisait exploser --
    // assertion __shared_ptr_deref, ou SIGSEGV en release.
    //
    // La cle absente n'est pas une anomalie : elle signifie que personne n'a
    // declare cette seconde valeur sur ce rang, donc qu'il n'y a aucun candidat
    // a compresser. C'est exactement le cas `sz == 0` traite juste apres, et on
    // sort pareil. Un ArrayChunk2 construit sans add_second_value prealable --
    // ce que font les auto-tests --test_stl -- tombait dedans des sa
    // construction, via resize_common -> make_block_compression.
    //
    // is_vec_tuple_to_compress_potentially_empty_thr() ne suffisait pas ici : il
    // ne teste que « aucune cle nulle part », pas « cette cle sur ce rang ».
    auto&      map_thr = manager_chunk2::map_chunk2_to_compress_potentially_thr[thr_rank];
    const auto it      = map_thr.find(second_value);

    if (it == map_thr.end() || !it->second)
        return;

    auto& vec_tuple_to_compress = *(it->second);

    const int sz = vec_tuple_to_compress.size();

    if (sz == 0)
        return;

    // Geometry: every free below is pinned to the freed chunk's OWN size (see
    // the header comment); the ambient is saved here and restored on every exit
    // so the sweep -- an OMPT callback on whichever thread -- leaves the
    // caller's thread_chunk_size() untouched. The triggering proxy's size is
    // still applied first for the handshake, as before.
    [[maybe_unused]] const unsigned int saved_sz = shrink_array::thread_chunk_size();

    if (proxy2)
    {
        if constexpr (build_config::chunk_dynamic)
            shrink_array::set_thread_chunk_size(proxy2->chunk_size());

        // Per-chunk exclusion (ChunkBase::steal_mark_): claim every candidate
        // BEFORE start(), whose heavy barrier publishes the marks; released
        // per chunk as the pass goes, and all again at the end.
        for (int k = 0; k < sz; ++k)
        {
            if (auto* const chunk_t = std::get<0>(vec_tuple_to_compress[k]))
                chunk_t->steal_mark();
        }
        const bool ret = proxy2->compression_mgr.template start<Proxy2<T>>();
        if (!ret)
        {
            for (int k = 0; k < sz; ++k)
            {
                if (auto* const chunk_t = std::get<0>(vec_tuple_to_compress[k]))
                    chunk_t->steal_unmark();
            }
            if constexpr (build_config::chunk_dynamic)
                shrink_array::set_thread_chunk_size(saved_sz);
            return;
        }
        // Check builds: the window is shut, commit the claims -- what the
        // store_value() guard tests (ChunkBase::steal_commit).
        if constexpr (build_config::any_check)
        {
            for (int k = 0; k < sz; ++k)
            {
                if (auto* const chunk_t = std::get<0>(vec_tuple_to_compress[k]))
                    chunk_t->steal_commit();
            }
        }
    }

    vec_tuple_chunk2p_int2 ll_compt1;
    vec_tuple_chunk2p_int2 ll_compt2;

    ll_compt1.reserve(sz / 50 + 1);
    ll_compt2.reserve(sz / 50 + 1);

    // Call the fill function (different for bool vs non-bool)
    fill_func(ll_compt1, ll_compt2);

    // Per-chunk geometry pin -- see this function's header comment.
    for (const auto& tup : ll_compt1)
    {
        if constexpr (build_config::chunk_dynamic)
            shrink_array::set_thread_chunk_size(std::get<3>(tup));
        clear_func(std::get<0>(tup), 0);
        std::get<0>(tup)->steal_unmark();
    }

    for (const auto& tup : ll_compt2)
    {
        if constexpr (build_config::chunk_dynamic)
            shrink_array::set_thread_chunk_size(std::get<3>(tup));
        clear_func(std::get<0>(tup), 1);
        std::get<0>(tup)->steal_unmark();
    }

    for (int k = 0; k < sz; ++k)
    {
        const auto& tup       = vec_tuple_to_compress[k];
        const int   num_chunk = std::get<1>(tup);

        const int num_tab = std::get<2>(tup);

        Proxy2<T>::reset_tab_is_inside_list_to_compress(num_tab, num_chunk);
        // The pass is over for every candidate: no mark may outlive it.
        if (auto* const chunk_t = std::get<0>(tup))
            chunk_t->steal_unmark();
    }

    vec_tuple_to_compress.clear();
    vec_tuple_to_compress.reserve(sz / 10 + 1);

    if (proxy2)
    {
        if (proxy2->is_not_cst_val())
        {
            proxy2->add_to_compression_list(vec_tuple_to_compress);
        }

        proxy2->compression_mgr.template finish<Proxy2<T>>();
    }

    if constexpr (build_config::chunk_dynamic)
        shrink_array::set_thread_chunk_size(saved_sz);
}

// make_block_compression (declared in chunk2_std.h). Binds the non-bool scan and
// clear callables and hands them to the shared sweep. Instantiating it for bool
// is a programming error: it aborts in debug builds and does nothing otherwise.
template <typename T, typename Derived, bool Enabled>
void Chunk2FastCompPolicy<T, Derived, Enabled>::make_block_compression(int thr_rank, const inf_array_t& inf_array, Proxy2<T>* proxy2)
{
    if constexpr (is_not_bool)
    {
        const in_t<T> second_value = std::get<0>(inf_array); // the caller's tuple, alive for the call

        auto lbd_clear_func = [thr_rank, &inf_array](auto* chunk_t, int clear_type)
        { chunk_t->clear_int_st(clear_type, std::get<1>(inf_array), thr_rank); };

        make_block_compression_impl(
            thr_rank, second_value, proxy2, [thr_rank, &inf_array](auto& ll_compt1, auto& ll_compt2)
            { Derived::fill_list_chunk_to_compress_in_par(thr_rank, ll_compt1, ll_compt2, inf_array); }, lbd_clear_func);
    }
    else
    {
        if constexpr (build_config::is_debug)
        {
            std::abort();
        }
    }
}

// make_block_compression_bool (declared in chunk2_std.h). Same binding for bool:
// second value is `true`, and the clear goes through clear_int_st_bool, which
// needs no index. Mirror image of the guard above -- non-bool aborts here.
template <typename T, typename Derived, bool Enabled>
void Chunk2FastCompPolicy<T, Derived, Enabled>::make_block_compression_bool(int thr_rank, Proxy2<T>* proxy2)
{
    if constexpr (is_bool)
    {
        auto lbd_clear_func = [thr_rank](auto* chunk_t, int clear_type)
        { chunk_t->clear_int_st_bool(clear_type, thr_rank); };

        make_block_compression_impl(
            thr_rank, true, proxy2, [thr_rank](auto& ll_compt1, auto& ll_compt2)
            { Derived::fill_list_chunk_to_compress_bool_in_par(thr_rank, ll_compt1, ll_compt2); }, lbd_clear_func);
    }
    else
    {
        if constexpr (build_config::is_debug)
        {
            std::abort();
        }
    }
}

// make_block_compression, thread-only overload (declared in chunk2_std.h). One
// sweep per distinct second value this rank has queued -- the per-value keying
// of the candidate map is why a "compress everything for this thread" pass has
// to iterate. Passes nullptr as the proxy, hence no handshake and no geometry
// pin (see the gap noted on make_block_compression_impl).
template <typename T, typename Derived, bool Enabled>
void Chunk2FastCompPolicy<T, Derived, Enabled>::make_block_compression(int thr_rank)
{
    if constexpr (is_not_bool)
    {
        const auto& ml = manager_chunk2::map_chunk2_to_compress_potentially_thr[thr_rank];

        for (auto& elm : ml)
        {
            const in_t<T> second_val = elm.first; // the map key, by reference when T is large
            const auto it         = manager_chunk2::map_chunk2_to_compress_potentially_thr[thr_rank].find(second_val);
            if (it == manager_chunk2::map_chunk2_to_compress_potentially_thr[thr_rank].end())
                continue;
            auto* vec_tuple_p = it->second.get();

            if (vec_tuple_p)
            {
                // The map keeps a slot per second value ever declared; the
                // arrays that declared this one may all be gone (cleared:
                // clear_core drops their candidates by num_tab, not the slot).
                // No array, no info tuple -- and nothing of that value left
                // to fold: skip it. Dereferenced unguarded, this was a SIGSEGV
                // in force_compression() / shrink_to_fit() after the clear of
                // a chunk2 array with a second value.
                const auto* inf = Proxy2<T>::share_info_array.find_inf_array(second_val);
                if (inf == nullptr)
                    continue;
                make_block_compression(thr_rank, *inf, nullptr);
            }
        }
    }
    else
    {
        if (is_vec_tuple_to_compress_potentially_empty_thr())
            return;

        const auto it = manager_chunk2::map_chunk2_to_compress_potentially_thr[thr_rank].find(true);
        if (it == manager_chunk2::map_chunk2_to_compress_potentially_thr[thr_rank].end())
            return;

        auto* vec_tuple_p = it->second.get();

        if (vec_tuple_p)
        {
            make_block_compression_bool(thr_rank, nullptr);
        }
    }
}

/**
 * @brief Pre-sizes each rank's candidate list for one second value.
 *
 * Discards nothing despite the name: it reserves every rank's list to the size
 * that rank reached during the previous phase, so a phase queueing as many
 * candidates as the last one does not reallocate while doing it.
 *
 * @param second_value Which per-value list to reserve.
 * @param max_thr      Number of per-rank lists to reserve.
 * @note OMPT builds only — called from the parallel-begin callback.
 */
template <typename T, typename Derived, bool Enabled>
void Chunk2FastCompPolicy<T, Derived, Enabled>::clean_chunk_to_compress(T second_value, int max_thr)
{
    if (is_vec_tuple_to_compress_potentially_empty_thr())
        return;

    for (int thr_rank = 0; thr_rank < max_thr; ++thr_rank)
    {
        auto& vec_pair = *(manager_chunk2::map_chunk2_to_compress_potentially_thr[thr_rank][second_value]);

        const int sz = manager_chunk2::old_size_chunk2_to_compress_potentially[thr_rank];
        if (sz)
            vec_pair.reserve(sz);
        else
            vec_pair.reserve(100);
    }
}
/**
 * @brief Bool twin of clean_chunk_to_compress().
 *
 * bool's second value is always `true`, so this IS the general reservation at
 * that key. The body used to be duplicated verbatim, with `true` hard-coded in
 * place of the parameter.
 *
 * @param max_thr Number of per-rank lists to reserve.
 */
template <typename T, typename Derived, bool Enabled>
void Chunk2FastCompPolicy<T, Derived, Enabled>::clean_chunk_to_compress_bool(int max_thr)
{
    clean_chunk_to_compress(static_cast<T>(true), max_thr);
}

// ---------------------------------------------------------------------------
// Chunk2OmpPolicy<T, Derived, true>: materialise and clear, OpenMP mode.
// ---------------------------------------------------------------------------

/**
 * @brief Shared body of create_value() and create_value_bool(): allocate, fill, publish.
 *
 * The two differ only in how the fresh block is filled and in which CAS
 * publishes it. The fill is a callable; the CAS is selected by @p IsBoolCas,
 * NOT passed in as a lambda -- see the note, this is load-bearing.
 *
 * @tparam IsBoolCas true to publish with compare_and_swap_bool, false with the
 *                   indexed compare_and_swap.
 * @tparam InitFn    callable `(T*)` pre-filling the freshly allocated block.
 * @param thr_rank             Rank whose allocator provides the buffer.
 * @param init_func            Pre-fill applied before publication.
 * @param ind                  Index of the array's second value; unused when
 *                             @p IsBoolCas (bool has no index).
 * @param need_set_second_value Which constant the block was filled with.
 *
 * @warning The CAS exists ONLY on the fast-block-compression policy: in a
 *          SERIAL build FastCompPolicy is the <false> specialisation, which has
 *          no compare_and_swap at all. So the lambda naming it must stay INSIDE
 *          the `if constexpr (use_fast_block_compression)` branch below -- a
 *          discarded branch is not instantiated. Passing that lambda in from
 *          the call site does NOT work, generic or not: once Chunk2<T> is
 *          instantiated FastCompPolicy is a concrete type, so the qualified
 *          name is looked up eagerly and the serial build fails to compile.
 *          That mistake was made here once; the manual-compression build that
 *          originally caught it is gone, so the serial configurations are what
 *          guard it now.
 */
template <typename T, typename Derived, bool Enabled>
template <bool IsBoolCas, typename InitFn>
void Chunk2OmpPolicy<T, Derived, Enabled>::create_value_impl(int thr_rank, InitFn&& init_func, [[maybe_unused]] int ind,
                                  [[maybe_unused]] bool need_set_second_value) noexcept
{
    T* values_loc = Base::allocate_chunk_internal(thr_rank, std::forward<InitFn>(init_func));

    if constexpr (chunk_config::use_fast_block_compression)
    {
        Base::apply_cas_or_recup(values_loc, thr_rank,
                                 [this, ind, need_set_second_value](T* p)
                                 {
                                     if constexpr (IsBoolCas)
                                     {
                                         (void) ind; // only the non-bool CAS takes the index
                                         return derived().compare_and_swap_bool(p, need_set_second_value);
                                     }
                                     else
                                         // Propagate the observed state to the CAS so it refuses a
                                         // wrong-side transition after a cst_val1 <-> cst_val2 race.
                                         return derived().compare_and_swap(p, ind, need_set_second_value);
                                 });
    }
    else
    {
        // apply_cas_or_recup takes its non-CAS branch in this mode anyway.
        Base::apply_cas_or_recup(values_loc, thr_rank, [](T*) { return false; });
    }
}

/**
 * @brief Materialises this chunk: allocate a buffer, fill it, publish it by CAS.
 *
 * The buffer is filled with first_value or the array's second value BEFORE it
 * is published, and publication is a single CAS. A reader therefore sees the
 * old constant or the finished buffer, never a half-built one. The CAS is told
 * which constant it is replacing, so it refuses a wrong-side transition after a
 * cst_val1/cst_val2 race; losing it means another thread materialised first,
 * and apply_cas_or_recup() recycles our buffer.
 *
 * @param need_set_second_value True to pre-fill with the second value, false for first_value.
 * @param inf_array             (second_value, index, ...) of the owning array.
 * @param thr_rank              Rank whose allocator provides the buffer.
 *
 * @note In a serial build there is no CAS to call — the policy has no such
 *       member — so the publication falls back to the non-CAS branch of
 *       apply_cas_or_recup().
 */
template <typename T, typename Derived, bool Enabled>
void Chunk2OmpPolicy<T, Derived, Enabled>::create_value(bool need_set_second_value, const inf_array_t& inf_array, int thr_rank) noexcept
{
    if constexpr (is_not_bool)
    {
        create_value_impl</*IsBoolCas=*/false>(
            thr_rank,
            [need_set_second_value, &inf_array](T* ptr)
            {
                if (!need_set_second_value)
                    Base::init_block(ptr);
                else
                    BaseSpec::init_block_2nd_value(ptr, inf_array);
            },
            std::get<1>(inf_array), need_set_second_value);
    }
    else
    {
        std::abort();
    }
}

/**
 * @brief Materialises a `bool` chunk.
 *
 * Same allocate-fill-CAS shape as create_value(), with no index to carry.
 *
 * @param need_set_second_value True to pre-fill with `true`, false for first_value.
 * @param thr_rank              Rank whose allocator provides the buffer.
 */
template <typename T, typename Derived, bool Enabled>
void Chunk2OmpPolicy<T, Derived, Enabled>::create_value_bool(bool need_set_second_value, int thr_rank) noexcept
{
    create_value_impl</*IsBoolCas=*/true>(
        thr_rank,
        [need_set_second_value](T* ptr)
        {
            if (!need_set_second_value)
                Base::init_block(ptr);
            else
                BaseSpec::init_block_2nd_value(ptr);
        },
        /*ind=*/0, need_set_second_value);
}

// ---------------------------------------------------------------------------
// Chunk2OmpPolicy<T, Derived, false>: materialise and clear, serial mode.
// ---------------------------------------------------------------------------

/**
 * @brief Materialises this chunk (serial build): allocate, fill, assign.
 *
 * No CAS — with nothing else running, the buffer is simply assigned to
 * `values`. It does one thing the parallel twin does not: reset the counter of
 * the constant the chunk is leaving, since the serial build tracks per-chunk
 * element counts (`use_compression_with_counter`) that the parallel one does
 * not maintain.
 *
 * @tparam Init2ndFn callable `(T*)` filling the block with the second value.
 * @param need_set_second_value True to pre-fill with the second value, false for first_value.
 * @param init_2nd              The second-value fill, applied when the flag is set.
 */
template <typename T, typename Derived>
template <typename Init2ndFn>
void Chunk2OmpPolicy<T, Derived, false>::create_value_serial_impl(bool need_set_second_value, Init2ndFn&& init_2nd) noexcept
{
    auto init_func = [this, need_set_second_value, &init_2nd](T* ptr)
    {
        // The counter reset is CROSSED on purpose: filling with one constant
        // invalidates the count kept for the OTHER one, which is the one that
        // has to go back to zero.
        if (!need_set_second_value)
        {
            Base::init_block(ptr);
            derived().reset_nb_val_eq_second_value();
        }
        else
        {
            init_2nd(ptr);
            derived().reset_nb_val_eq_first_value();
        }
    };

    values = Base::allocate_chunk_internal(0, init_func);
}

template <typename T, typename Derived>
void Chunk2OmpPolicy<T, Derived, false>::create_value(bool need_set_second_value, const inf_array_t& inf_array) noexcept
{
    if constexpr (is_not_bool)
    {
        create_value_serial_impl(need_set_second_value,
                                 [&inf_array](T* ptr) { BaseSpec::init_block_2nd_value(ptr, inf_array); });
    }
    else if constexpr (build_config::is_debug)
    {
        std::cerr << "Chunk2_std_inline::create_value() called" << std::endl;
        std::abort();
    }
}

/**
 * @brief Materialises a `bool` chunk (serial build).
 * @param need_set_second_value True to pre-fill with `true`, false for first_value.
 */
template <typename T, typename Derived>
void Chunk2OmpPolicy<T, Derived, false>::create_value_bool(bool need_set_second_value) noexcept
{
    if constexpr (is_bool)
    {
        create_value_serial_impl(need_set_second_value,
                                 [](T* ptr) { BaseSpec::init_block_2nd_value(ptr); });
    }
    else if constexpr (build_config::is_debug)
    {
        std::cerr << "Chunk2_std_inline::create_value_bool() called" << std::endl;
        std::abort();
    }
}

/**
 * @brief Compresses this chunk back to first_value, if it is materialised.
 * @param ind      Index of the array's second value, needed by the state test.
 * @param thr_rank Rank whose pool receives the released buffer.
 */
template <typename T, typename Derived, bool Enabled>
inline void Chunk2OmpPolicy<T, Derived, Enabled>::clear(int ind, int thr_rank)
{
    Base::clear_impl(ind, thr_rank);
}

/**
 * @brief Compresses this chunk to first_value without testing `written()` (serial build).
 *
 * For callers that have already established the chunk is materialised; the test
 * survives as a debug-only assertion.
 *
 * @param ind Index of the array's second value, used by that assertion.
 */
template <typename T, typename Derived>
inline void Chunk2OmpPolicy<T, Derived, false>::clear_st(int ind)
{
    Base::debug_check_written(ind);

    Base::free_values(values);
    derived().init_values_cst_val1();

    if constexpr (chunk_config::use_compression_with_counter)
    {
        derived().reset_nb_val_eq_value();
    }
}

/**
 * @brief Releases this chunk's buffer and points it back at a constant.
 *
 * The single point every compression path funnels through, in both modes and
 * every chunk variant — which is why the "block compressed" and "bytes freed"
 * statistics are counted here and nowhere else.
 *
 * @param num_compress_value 0 for first_value, otherwise the second value.
 * @param ind                Index of the array's second value.
 * @param thr_rank           Rank whose pool receives the released buffer.
 */
template <typename T, typename Derived, bool Enabled>
inline void Chunk2OmpPolicy<T, Derived, Enabled>::block_clear(int num_compress_value, int ind, [[maybe_unused]] int thr_rank)
{
    // CHUNK_STATS: one materialised block released by a successful compression.
    // Counted here because block_clear is the single point every variant and
    // both OMP/non-OMP modes funnel through (the old counter only fired on the
    // plain-chunk sweep, so chunk2/chunk2a always reported 0).
    increm_chunk_compress<Derived>();
    add_chunk_bytes_freed<Derived>(static_cast<long>(sizeof(T)) * static_cast<long>(shrink_array::alloc_chunk_count()));
    Base::block_clear_common(num_compress_value, ind, thr_rank);
}

/// @copydoc Chunk2OmpPolicy::block_clear
/// Serial build: frees the buffer outright instead of parking it in a rank
/// pool, and resets the per-chunk element counters.
template <typename T, typename Derived>
inline void Chunk2OmpPolicy<T, Derived, false>::block_clear(int num_compress_value, int ind, int)
{
    // CHUNK_STATS: one materialised block released by a successful compression.
    // Counted here because block_clear is the single point every variant and
    // both OMP/non-OMP modes funnel through (the old counter only fired on the
    // plain-chunk sweep, so chunk2/chunk2a always reported 0).
    increm_chunk_compress<Derived>();
    add_chunk_bytes_freed<Derived>(static_cast<long>(sizeof(T)) * static_cast<long>(shrink_array::alloc_chunk_count()));
    Base::free_values(values);

    derived().init_values_cst_val(num_compress_value, ind);
    if constexpr (chunk_config::use_compression_with_counter)
    {
        derived().reset_nb_val_eq_value();
    }
}

/**
 * @brief Compresses this chunk to the named constant, after an invariant check.
 * @param num_compress_value 0 for first_value, otherwise the second value.
 * @param ind                Index of the array's second value.
 * @param thr_rank           Rank whose pool receives the released buffer.
 * @pre The chunk is materialised — compressing an already-constant chunk would
 *      release a shared constant block. Checked by
 *      check_not_constant_before_clear().
 */
template <typename T, typename Derived, bool Enabled>
inline void Chunk2OmpPolicy<T, Derived, Enabled>::clear_int_st(int num_compress_value, int ind, int thr_rank)
{
    Base::check_not_constant_before_clear(num_compress_value, ind);
    block_clear(num_compress_value, ind, thr_rank);
}

/**
 * @brief Bool twin of clear_int_st(): no second-value index to pass.
 * @param num_compress_value 0 for first_value, otherwise `true`.
 * @param thr_rank           Rank whose pool receives the released buffer.
 */
template <typename T, typename Derived, bool Enabled>
inline void Chunk2OmpPolicy<T, Derived, Enabled>::clear_int_st_bool(int num_compress_value, int thr_rank)
{
    if constexpr (is_bool)
    {
        Base::check_not_constant_before_clear(num_compress_value, std::nullopt);
        block_clear(num_compress_value, 0, thr_rank);
    }
}

/**
 * @brief Compresses this chunk to the named constant (serial build).
 *
 * Serial counterpart of the OpenMP clear_int_st(): it does NOT go through
 * block_clear() but frees and re-initialises directly, so it carries its own
 * copies of the statistics and, unlike the parallel path, resets the per-chunk
 * element counter of the constant being adopted.
 *
 * @param num_compress_value 0 for first_value, otherwise the second value.
 * @param ind                Index of the array's second value; unused for bool.
 * @pre The chunk is materialised (debug-only assertion).
 */
template <typename T, typename Derived>
inline void Chunk2OmpPolicy<T, Derived, false>::clear_int_st(int num_compress_value, int ind)
{
    // --- check only on debug ---

    // Bool chunks share a single second value with no index: use the no-arg
    // written check and the bool-specific reinit. Non-bool uses the indexed
    // path. (Matches the counter-policy split in the reference implementation.)
    if constexpr (is_bool)
        Base::debug_check_written();
    else
        Base::debug_check_written(ind);

    // CHUNK_STATS: this is the non-OMP twin of block_clear -- reached from
    // compress_int's sweep via clear_st(num_compress_value, ind), which
    // bypasses block_clear entirely. Without this, compress stayed at 0 in
    // sequential builds even though the sweep was actively compressing chunks.
    // No if constexpr(chunk_config::stats) needed: both functions already
    // resolve to a no-op overload when CHUNK_STATS is off (selected at their
    // definition, in chunk_stats.h) -- matching the OMP
    // twin of block_clear above, which calls them unconditionally too. An
    // earlier version of this guard had its braces missing, silently
    // dropping the second call out of the (redundant, and therefore
    // harmless) if constexpr; removed rather than re-braced, for consistency.
    increm_chunk_compress<Derived>();
    add_chunk_bytes_freed<Derived>(static_cast<long>(sizeof(T)) * static_cast<long>(shrink_array::alloc_chunk_count()));

    Base::free_values(values);

    if (num_compress_value == 0)
    {
        derived().reset_nb_val_eq_first_value();
    }
    else
    {
        derived().reset_nb_val_eq_second_value();
    }

    if constexpr (is_bool)
        derived().init_values_cst_val_bool(num_compress_value);
    else
        derived().init_values_cst_val(num_compress_value, ind);
}

// clear_st(int num_compress_value, int ind) -- the entry point that supplies
// the calling thread's rank (or has none to supply) -- is defined in-class on
// each Chunk2OmpPolicy specialisation.

} // end namespace shrink_array
