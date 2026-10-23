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

#include <cassert>  // assert (the candidate-list lookups)
#include <cstddef>  // std::size_t
#include <optional> // std::nullopt (the clears)

#include "chunk/config.h"                        // build_config::chunk_dynamic, chunk_config::OMPTcc
#include "chunk/PMP_CHUNK.h"                     // CHUNK_PMP_*, pmp::team
#include "chunk/internal/chunk_array_new.h"      // chunk_array_new::init_chunks (operator new[])
#include "chunk/internal/chunk_size_policy.h"    // ChunkElems, set_thread_chunk_size
#include "chunk/internal/in_t.h"                 // InT<T>
#include "chunk/zero_one_value_chunk.h"          // zero_one_value::Chunk, the class these bodies belong to
#include "chunk/internal/zero_one_value_proxy.h" // zero_one_value::Proxy<T>
#include "chunk/internal/block_pass_precheck.h"  // block_pass_yields_to_flush
#include "chunk/internal/dependent_false.h"      // dependent_false_v (wrong-type branches)
#include "chunk/internal/invariant_check.h"      // CHUNK_INVARIANT_FAILED
#include "chunk/internal/test_hooks.h"           // TEST KNOB: test_hooks::maybe_pass_delay

/**
 * @file zero_one_value_chunk_inline.h
 * @brief Out-of-line definitions of zero_one_value's policies: materialisation and compression.
 *
 * zero_one_value is the two-constant chunk whose second value is shared by the WHOLE
 * array — that is what separates it from zero_any, where each chunk carries its
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
 *    (build_config::threaded); the queue-driven compression is on
 *    Chunk2FastCompPolicy<..., true>.
 *  - **Geometry comes from the queued entry, never from the ambient size.**
 *    Every candidate tuple carries the chunk count of its owning array
 *    (field `<3>`), because the thread consuming the entry may have a different
 *    ambient size — after a rechunk, scanning at the ambient size overruns a
 *    buffer allocated at another geometry.
 *
 * @see zero_one_value_chunk.h for the declarations, chunk_base.h for the shared
 *      allocate/free machinery, zero_one_value_proxy_inline.h for the write paths that queue
 *      the candidates.
 */

namespace shrink_array
{

// ── Array-new (placement-new arrays; see chunk_array_new.h) ────────────────
// Out of line on purpose, as operator delete[] below: see there.
template <typename T, typename Derived, bool Enabled>
[[gnu::noinline]] void* zero_one_value::ChunkOmpPolicy<T, Derived, Enabled>::operator new[](std::size_t sz)
{
    auto* v = static_cast<zero_one_value::Chunk<T>*>(::operator new(sz));
    init_chunks_cst_val(v, sz / sizeof(zero_one_value::Chunk<T>));
    return static_cast<void*>(v);
}

// Out of line on purpose, as operator new[] above: inlined, either one shows
// gcc's -Wmismatched-new-delete a ::operator new/delete paired with the class's
// other operator, a false positive. Kept as calls, it sees the class's own
// new[] / delete[] pair.
template <typename T, typename Derived, bool Enabled>
[[gnu::noinline]] void zero_one_value::ChunkOmpPolicy<T, Derived, Enabled>::operator delete[](void* ptr)
{
    ::operator delete(ptr);
}

template <typename T, typename Derived, bool Enabled>
void zero_one_value::ChunkOmpPolicy<T, Derived, Enabled>::init_chunks_cst_val(zero_one_value::Chunk<T>* v,
                                                                              std::size_t               count) noexcept
{
    chunk_array_new::init_chunks(v, count, [](zero_one_value::Chunk<T>& chunk) { chunk.init_after_constructor(); });
}


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
void zero_one_value::ChunkFastCompPolicy<T, Derived, Enabled>::fill_list_to_compress(VecTupleChunk2pInt2& ll_compt1,
                                                                                     VecTupleChunk2pInt2& ll_compt2,
                                                                                     const InfArrayT&     inf_array,
                                                                                     int                  num_chunk,
                                                                                     int                  num_tab,
                                                                                     unsigned int         n_elems)
{
    // InT: a reference into the (const, caller-owned) tuple when T is large.
    const InT<T> second_v = inf_array.second_value;
    fill_list_to_compress_impl(
        ll_compt1, ll_compt2, num_chunk, num_tab, [&second_v]() -> InT<T> { return second_v; },
        derived().is_not_cst_val2(inf_array.ind), n_elems);
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
void zero_one_value::ChunkFastCompPolicy<T, Derived, Enabled>::fill_list_to_compress_bool(
    VecTupleChunk2pInt2& ll_compt1,
    VecTupleChunk2pInt2& ll_compt2,
    int                  num_chunk,
    int                  num_tab,
    unsigned int         n_elems)
{
    fill_list_to_compress_impl(
        ll_compt1, ll_compt2, num_chunk, num_tab, [] { return true; }, derived().is_not_cst_val2(), n_elems);
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
void zero_one_value::ChunkFastCompPolicy<T, Derived, Enabled>::fill_list_to_compress_impl(
    VecTupleChunk2pInt2& ll_compt1,
    VecTupleChunk2pInt2& ll_compt2,
    int                  num_chunk,
    int                  num_tab,
    SecondValueFunc&&    get_second_value,
    bool                 check_cst_val2,
    unsigned int         n_elems)
{
    // Skip a chunk already compressed to either constant.
    if (Base::is_not_cst_val1() && check_cst_val2)
    {
        // CHUNK_STATS: one compression candidate scanned (see zero_any).
        increm_chunk_compress_attempt<Derived>();

        // n_elems: see zero_one_value::Chunk<T>::compress() -- the chunk's own geometry,
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
 *          compresses nothing.
 */
template <typename T, typename Derived, bool Enabled>
template <typename SecondValueFunc, typename ClearFunc>
void zero_one_value::ChunkFastCompPolicy<T, Derived, Enabled>::compress_impl(bool            apply,
                                                                             unsigned int    n_elems,
                                                                             SecondValueFunc get_second_value,
                                                                             ClearFunc&&     clear_one)
{
    if (Base::is_not_cst_val1())
    {
        // CHUNK_STATS: a SECOND candidate scan, independent of
        // fill_list_to_compress, so it counts its own attempt. It reads values[]
        // directly, so it never inflates `read` and needs no read_value_scan.
        increm_chunk_compress_attempt<Derived>();

        // n_elems is THIS chunk's own geometry, captured at enqueue (tuple
        // field <3>), never alloc_chunk_count() -- the consuming thread's
        // ambient size overflows a buffer allocated at another geometry after
        // a rechunk. Same rule as zero and zero_any.
        //
        // Always scan, even when the second value equals first_value: skipping
        // would leave BOTH verdicts `true` and clear the chunk twice.
        // scan_constants reports such a run as `first` only.
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
void zero_one_value::ChunkFastCompPolicy<T, Derived, Enabled>::compress([[maybe_unused]] int thr_rank,
                                                                        const InfArrayT&     inf_array,
                                                                        bool                 apply,
                                                                        unsigned int         n_elems)
{
    // Only a MATERIALISED chunk can be compressed. A queued entry can outlive
    // the chunk's materialisation: make_block_compression (or a previous
    // drain) compresses it, and the OMPT end-of-region drain then replays
    // the stale entry. compress_impl only refuses cst_val1; a chunk already at
    // cst_val2 would pass, scan the SHARED second-value block, find it
    // constant and clear it a second time -- freeing that shared block.
    if (!derived().written(inf_array.ind))
    {
        return;
    }

    compress_impl(
        apply, n_elems, [&inf_array] { return inf_array.second_value; },
        [this, thr_rank, &inf_array](int num_compress_value)
        { this->clear_int_st(num_compress_value, inf_array.ind, thr_rank); });
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
void zero_one_value::ChunkFastCompPolicy<T, Derived, Enabled>::compress_bool([[maybe_unused]] int thr_rank,
                                                                             bool                 apply,
                                                                             unsigned int         n_elems)
{
    if constexpr (is_bool)
    {
        if (!derived().written()) // see compress(): a stale entry for a chunk already at a constant
        {
            return;
        }
        compress_impl(
            apply, n_elems, [] { return true; },
            [this, thr_rank](int num_compress_value) { this->clear_int_st_bool(num_compress_value, thr_rank); });
    }
}

/**
 * @brief Compresses the candidates of one second value, across every rank.
 *
 * Entry point used between parallel phases. The OMPT callbacks are disarmed for
 * the duration, so nothing re-enters the compression machinery underneath.
 *
 * @note Like the zero_any equivalent: outside a parallel region it opens one and
 *       lets each rank sweep its own list (serial mop-up for ranks beyond the
 *       team size); inside a region it sweeps the calling rank only.
 */
template <typename T, typename Derived, bool Enabled>
void zero_one_value::ChunkFastCompPolicy<T, Derived, Enabled>::compress_chunk_to_compress(const InfArrayT& inf_array,
                                                                                          bool             apply)
{
    // Everything the per-rank body touches is rank-local --
    // map_chunk2_to_compress_potentially_thr[thr_rank], the freed buffer goes to
    // thr_rank's pool (clear_int_st), and the candidate flag reset is an atomic
    // store -- so the ranks are independent (a serial loop leaves the team idle
    // during a re-chunk).
    const auto sweep = [&inf_array, apply](int thr) { compress_chunk_to_compress_in_par(thr, inf_array, apply); };
    Base::sweep_every_rank(sweep, sweep);
}

/**
 * @brief Bool twin of compress_chunk_to_compress().
 */
template <typename T, typename Derived, bool Enabled>
void zero_one_value::ChunkFastCompPolicy<T, Derived, Enabled>::compress_chunk_to_compress_bool(bool apply)
{
    if constexpr (is_bool)
    {
        // Same as compress_chunk_to_compress(): rank-local sweeps.
        const auto sweep = [apply](int thr) { compress_chunk_to_compress_bool_in_par(thr, apply); };
        Base::sweep_every_rank(sweep, sweep);
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
 * That flag is the CAS gate of ProxyBase::add_to_compression_list(): its 0 -> 1
 * flip admits a chunk to the writer's candidate list and is the only thing
 * keeping a chunk out of two lists at once. Cleared before compress(), it lets
 * a concurrent write re-queue the SAME chunk mid-compression, and a later drain
 * of that entry double-clears an already-constant chunk (the zero_any @warning;
 * check_not_constant_before_clear fires). Compressing first keeps the flag at 1
 * throughout, so every concurrent write loses the CAS until the chunk reaches
 * its final state.
 *
 * Today the drain has no concurrent writers (on_parallel_end() puts an
 * `omp barrier` in front of it; the zero_one_value::Array call sites are gated on
 * !CHUNK_PMP_IN_PARALLEL_REGION), so either order works; this one stays
 * correct if that precondition slips, as on zero_any.
 *
 * @tparam CompressFn callable `(ChunkType*, unsigned int n_elems)`.
 * @param thr_rank     Rank whose list is drained.
 * @param key          Second value indexing the per-rank candidate map.
 * @param compress_one Per-chunk compression.
 */
template <typename T, typename Derived, bool Enabled>
template <typename CompressFn>
void zero_one_value::ChunkFastCompPolicy<T, Derived, Enabled>::compress_chunk_to_compress_in_par_impl(
    int          thr_rank,
    InT<T>       key,
    CompressFn&& compress_one)
{
    // The store is a per-rank VECTOR of maps, sized by resize_all_list_block_free_thr();
    // until that has run, [thr_rank] is out of bounds (UB). Probing rank 0 is
    // enough: zero_one_value::Array::add_second_value registers a key for every rank in
    // one loop, so an empty map at rank 0 means no key at any rank.
    if (is_vec_tuple_to_compress_potentially_empty_thr())
    {
        return;
    }

    auto* const vec_tuple_p = ManagerChunk2::compress_candidates(thr_rank, key);
    assert(vec_tuple_p != nullptr && "key registered for every rank by add_second_value()");
    if (vec_tuple_p == nullptr) [[unlikely]]
    {
        return;
    }
    auto& vec_tuple = *vec_tuple_p;

    // Geometry: this list mixes chunks of every array sharing `key` (e.g. an
    // array and the tmp rechunk_to() builds from it). The clear reaches
    // free_block_to_pool(), which routes by the ambient thread_chunk_size()
    // (reuse list AND allocator), so pin each chunk to its own size -- else
    // wide blocks are parked in a narrower list -- and restore afterwards so
    // the OMPT drain leaves the thread as it found it. Same pin as
    // make_block_compression_impl() and zero_any::Chunk<T>::make_block_compression().
    [[maybe_unused]] const unsigned int saved_sz = thread_chunk_size();

    for (auto& tup : vec_tuple)
    {
        auto* chunk_t = std::get<0>(tup);
        if (chunk_t)
        {
            const int num_chunk = std::get<1>(tup);
            const int num_tab   = std::get<2>(tup);

            if constexpr (build_config::chunk_dynamic)
            {
                shrink_array::set_thread_chunk_size(std::get<3>(tup).count());
            }

            compress_one(chunk_t, std::get<3>(tup).count());

            // AFTER the compression, never before -- see the ORDER note above.
            zero_one_value::Proxy<T>::reset_tab_is_inside_list_to_compress(num_tab, num_chunk);
        }
    }

    if constexpr (build_config::chunk_dynamic)
    {
        set_thread_chunk_size(saved_sz);
    }

    // clear() keeps the capacity: the list is already reserved for the next
    // pass, at the size this one reached.
    vec_tuple.clear();
}

/**
 * @brief Compresses one rank's candidates for one second value.
 * @param thr_rank       Rank whose list is drained.
 * @param inf_array      (second_value, index, ...) selecting the list.
 * @param apply          Forwarded to compress().
 */
template <typename T, typename Derived, bool Enabled>
void zero_one_value::ChunkFastCompPolicy<T, Derived, Enabled>::compress_chunk_to_compress_in_par(
    int              thr_rank,
    const InfArrayT& inf_array,
    bool             apply)
{
    compress_chunk_to_compress_in_par_impl(thr_rank, inf_array.second_value,
                                           [thr_rank, &inf_array, apply](auto* chunk_t, unsigned int n_elems)
                                           { chunk_t->compress(thr_rank, inf_array, apply, n_elems); });
}

/**
 * @brief Bool twin of compress_chunk_to_compress_in_par().
 * @param thr_rank       Rank whose list is drained.
 * @param apply          Forwarded to compress_bool().
 */
template <typename T, typename Derived, bool Enabled>
void zero_one_value::ChunkFastCompPolicy<T, Derived, Enabled>::compress_chunk_to_compress_bool_in_par(int  thr_rank,
                                                                                                      bool apply)
{
    if constexpr (is_bool)
    {
        compress_chunk_to_compress_in_par_impl(thr_rank, true, [thr_rank, apply](auto* chunk_t, unsigned int n_elems)
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
 * @tparam ScanFn callable `(ChunkType*, int num_chunk, int num_tab, ChunkElems)`.
 * @param thr_rank Rank whose candidate list is scanned.
 * @param key      Second value indexing the per-rank candidate map.
 * @param scan_one Per-chunk scan.
 */
template <typename T, typename Derived, bool Enabled>
template <typename ScanFn>
void zero_one_value::ChunkFastCompPolicy<T, Derived, Enabled>::fill_list_chunk_to_compress_in_par_impl(
    int      thr_rank,
    InT<T>   key,
    ScanFn&& scan_one)
{
    const auto* const vec_tuple_p = ManagerChunk2::compress_candidates(thr_rank, key);
    assert(vec_tuple_p != nullptr && "key registered for every rank by add_second_value()");
    if (vec_tuple_p == nullptr) [[unlikely]]
    {
        return;
    }
    const auto& vec_tuple = *vec_tuple_p;

    const int nn = vec_tuple.size();

    for (int k = 0; k < nn; ++k)
    {
        const auto& tup     = vec_tuple[k];
        const auto& chunk_t = std::get<0>(tup);
        if (chunk_t)
        {
            const int& num_chunk = std::get<1>(tup);
            const int  num_tab   = std::get<2>(tup);
            // Per-entry geometry (see zero_one_value_chunk.h): in dynamic builds each
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
void zero_one_value::ChunkFastCompPolicy<T, Derived, Enabled>::fill_list_chunk_to_compress_in_par(
    int                  thr_rank,
    VecTupleChunk2pInt2& ll_compt1,
    VecTupleChunk2pInt2& ll_compt2,
    const InfArrayT&     inf_array)
{
    fill_list_chunk_to_compress_in_par_impl(
        thr_rank, inf_array.second_value,
        [&ll_compt1, &ll_compt2, &inf_array](auto* chunk_t, int num_chunk, int num_tab, unsigned int n_elems)
        {
            const auto n1 = ll_compt1.size();
            const auto n2 = ll_compt2.size();
            chunk_t->fill_list_to_compress(ll_compt1, ll_compt2, inf_array, num_chunk, num_tab, n_elems);
            // Not selected by the scan: the writers may have it back now.
            if (ll_compt1.size() == n1 && ll_compt2.size() == n2)
            {
                chunk_t->steal_unmark();
            }
        });
}

/**
 * @brief Bool twin of fill_list_chunk_to_compress_in_par().
 * @param thr_rank  Rank whose candidate list is scanned.
 * @param ll_compt1 Output: chunks compressible to first_value.
 * @param ll_compt2 Output: chunks compressible to `true`.
 */
template <typename T, typename Derived, bool Enabled>
void zero_one_value::ChunkFastCompPolicy<T, Derived, Enabled>::fill_list_chunk_to_compress_bool_in_par(
    int                  thr_rank,
    VecTupleChunk2pInt2& ll_compt1,
    VecTupleChunk2pInt2& ll_compt2)
{
    fill_list_chunk_to_compress_in_par_impl(
        thr_rank, static_cast<T>(true),
        [&ll_compt1, &ll_compt2](auto* chunk_t, int num_chunk, int num_tab, unsigned int n_elems)
        {
            const auto n1 = ll_compt1.size();
            const auto n2 = ll_compt2.size();
            chunk_t->fill_list_to_compress_bool(ll_compt1, ll_compt2, num_chunk, num_tab, n_elems);
            if (ll_compt1.size() == n1 && ll_compt2.size() == n2)
            {
                chunk_t->steal_unmark();
            }
        });
}

// --- The sweep. Precedent: ChunkFastCompPolicy<T,Derived,Enabled>::
// make_block_compression for zero::Chunk<T>.

// make_block_compression_impl (declared in zero_one_value_chunk.h). Shape: pin geometry ->
// take the compression handshake -> scan into the two buckets -> clear each ->
// reset every queued flag -> empty the list -> re-queue the triggering chunk.
// The two callables are the only bool/non-bool difference.
// Geometry: as in compress_chunk_to_compress_in_par_impl(), the list mixes
// arrays of different chunk sizes, so each chunk is pinned to its own n_elems
// right before its clear (as zero_any::Chunk<T>::make_block_compression() does). Do not
// rely on the single pin on the triggering proxy's size (kept, for the
// handshake): it would park the other array's buffers under the wrong size.
template <typename T, typename Derived, bool Enabled>
template <typename FillFunc, typename ClearFunc>
void zero_one_value::ChunkFastCompPolicy<T, Derived, Enabled>::make_block_compression_impl(
    int                       thr_rank,
    T                         second_value,
    zero_one_value::Proxy<T>* proxy2,
    FillFunc&&                fill_func,
    ClearFunc&&               clear_func)
{
    // find, NOT operator[]: on a missing key operator[] inserts a NULL
    // shared_ptr that the next dereference crashes on.
    //
    // A missing key is normal (no one declared this second value on this
    // rank: nothing to compress, same exit as `sz == 0`); a zero_one_value::Array built
    // without add_second_value reaches it via resize_common ->
    // make_block_compression. is_vec_tuple_to_compress_potentially_empty_thr()
    // is not enough: it only tests "no key anywhere".
    auto&      map_thr = ManagerChunk2::map_chunk2_to_compress_potentially_thr[thr_rank];
    const auto it      = map_thr.find(second_value);

    if (it == map_thr.end() || !it->second)
    {
        return;
    }

    auto& vec_tuple_to_compress = *it->second;

    const int sz = vec_tuple_to_compress.size();

    if (sz == 0)
    {
        return;
    }

    // Geometry: every free below is pinned to the freed chunk's OWN size (see
    // the header comment); the ambient is saved here and restored on every exit
    // so the sweep -- an OMPT callback on whichever thread -- leaves the
    // caller's thread_chunk_size() untouched. The triggering proxy's size is
    // applied first, for the handshake.
    [[maybe_unused]] const unsigned int saved_sz = thread_chunk_size();

    if (proxy2)
    {
        if constexpr (build_config::chunk_dynamic)
        {
            shrink_array::set_thread_chunk_size(proxy2->chunk_size());
        }

        // A flush in progress makes start() decline: ask first, before
        // marking -- see block_pass_precheck.h.
        if (block_pass_yields_to_flush<zero_one_value::Proxy<T>, zero_one_value::Chunk<T>>(CHUNK_PMP_MAX_THREADS))
        {
            if constexpr (build_config::chunk_dynamic)
            {
                set_thread_chunk_size(saved_sz);
            }
            return;
        }
        // Per-chunk exclusion (ChunkBase::steal_mark_): claim every candidate
        // BEFORE start(), whose heavy barrier publishes the marks; released
        // per chunk as the pass goes, and all again at the end.
        for (int k = 0; k < sz; ++k)
        {
            if (auto* const chunk_t = std::get<0>(vec_tuple_to_compress[k]))
            {
                chunk_t->steal_mark();
            }
        }
        const bool ret = proxy2->compression_mgr.template start<zero_one_value::Proxy<T>>();
        if (!ret)
        {
            for (int k = 0; k < sz; ++k)
            {
                if (auto* const chunk_t = std::get<0>(vec_tuple_to_compress[k]))
                {
                    chunk_t->steal_unmark();
                }
            }
            if constexpr (build_config::chunk_dynamic)
            {
                set_thread_chunk_size(saved_sz);
            }
            return;
        }
        // Check builds: the window is shut, commit the claims -- what the
        // store_value() guard tests (ChunkBase::steal_commit).
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

    VecTupleChunk2pInt2 ll_compt1;
    VecTupleChunk2pInt2 ll_compt2;

    ll_compt1.reserve((sz / 50) + 1);
    ll_compt2.reserve((sz / 50) + 1);

    fill_func(ll_compt1, ll_compt2);

    // Per-chunk geometry pin -- see this function's header comment.
    for (const auto& tup : ll_compt1)
    {
        if constexpr (build_config::chunk_dynamic)
        {
            shrink_array::set_thread_chunk_size(std::get<3>(tup));
        }
        clear_func(std::get<0>(tup), 0);
        std::get<0>(tup)->steal_unmark();
    }

    for (const auto& tup : ll_compt2)
    {
        if constexpr (build_config::chunk_dynamic)
        {
            shrink_array::set_thread_chunk_size(std::get<3>(tup));
        }
        clear_func(std::get<0>(tup), 1);
        std::get<0>(tup)->steal_unmark();
    }

    for (int k = 0; k < sz; ++k)
    {
        const auto& tup       = vec_tuple_to_compress[k];
        const int   num_chunk = std::get<1>(tup);

        const int num_tab = std::get<2>(tup);

        // Hand the candidate back: its mark FIRST, then the dedup flag. While
        // the flag is up the chunk is in no other thread's list, so no other
        // pass can have marked it and this unmark only drops our own claim
        // (idempotent for the ones released above). Do not swap them: once the
        // flag is down another thread can queue the chunk, start its pass and
        // mark it, and this unmark would erase that claim -- writers would then
        // write into a chunk that pass is scanning and stealing.
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
        zero_one_value::Proxy<T>::reset_tab_is_inside_list_to_compress(num_tab, num_chunk);
    }

    vec_tuple_to_compress.clear();
    vec_tuple_to_compress.reserve((sz / 10) + 1);

    if (proxy2)
    {
        // Re-queue the writer's chunk only into the list of ITS array's second
        // value. zero_one_value::Proxy::perform_compression runs the pass of every key of the
        // rank with the same writer: queued under another array's key, the
        // chunk could be compressed to the OTHER array's constant, and its own
        // array, which only recognises its own constant, would then write into
        // that shared block, changing every chunk compressed to it.
        if (proxy2->is_not_cst_val() && proxy2->second_value_ == second_value)
        {
            proxy2->add_to_compression_list(vec_tuple_to_compress);
        }

        proxy2->compression_mgr.template finish<zero_one_value::Proxy<T>>();
    }

    if constexpr (build_config::chunk_dynamic)
    {
        set_thread_chunk_size(saved_sz);
    }
}

// make_block_compression (declared in zero_one_value_chunk.h). Binds the non-bool scan and
// clear callables and hands them to the shared sweep. Instantiating it for bool
// is a compile error (static_assert).
template <typename T, typename Derived, bool Enabled>
void zero_one_value::ChunkFastCompPolicy<T, Derived, Enabled>::make_block_compression(int              thr_rank,
                                                                                      const InfArrayT& inf_array,
                                                                                      zero_one_value::Proxy<T>* proxy2)
{
    if constexpr (is_not_bool)
    {
        const InT<T> second_value = inf_array.second_value; // the caller's record, alive for the call

        const auto lbd_clear_func = [thr_rank, &inf_array](auto* chunk_t, int clear_type)
        { chunk_t->clear_int_st(clear_type, inf_array.ind, thr_rank); };

        make_block_compression_impl(
            thr_rank, second_value, proxy2, [thr_rank, &inf_array](auto& ll_compt1, auto& ll_compt2)
            { Derived::fill_list_chunk_to_compress_in_par(thr_rank, ll_compt1, ll_compt2, inf_array); },
            lbd_clear_func);
    }
    else
    {
        static_assert(dependent_false_v<T>, "make_block_compression(thr, inf_array, proxy) is for non-bool elements: "
                                            "use make_block_compression_bool()");
    }
}

// make_block_compression_bool (declared in zero_one_value_chunk.h). Same binding for bool:
// second value is `true`, and the clear goes through clear_int_st_bool, which
// needs no index. Mirror image of the guard above -- non-bool fails to compile.
template <typename T, typename Derived, bool Enabled>
void zero_one_value::ChunkFastCompPolicy<T, Derived, Enabled>::make_block_compression_bool(
    int                       thr_rank,
    zero_one_value::Proxy<T>* proxy2)
{
    if constexpr (is_bool)
    {
        const auto lbd_clear_func = [thr_rank](auto* chunk_t, int clear_type)
        { chunk_t->clear_int_st_bool(clear_type, thr_rank); };

        make_block_compression_impl(
            thr_rank, true, proxy2, [thr_rank](auto& ll_compt1, auto& ll_compt2)
            { Derived::fill_list_chunk_to_compress_bool_in_par(thr_rank, ll_compt1, ll_compt2); }, lbd_clear_func);
    }
    else
    {
        static_assert(
            dependent_false_v<T>,
            "make_block_compression_bool() is for bool elements: use make_block_compression(thr, inf_array, proxy)");
    }
}

// make_block_compression, thread-only overload (declared in zero_one_value_chunk.h). One
// sweep per distinct second value this rank has queued (the candidate map is
// keyed by value). Passes nullptr as the proxy: no handshake and no proxy-size
// pin (each freed chunk is still pinned to its own size).
template <typename T, typename Derived, bool Enabled>
void zero_one_value::ChunkFastCompPolicy<T, Derived, Enabled>::make_block_compression(int thr_rank)
{
    if constexpr (is_not_bool)
    {
        const auto& ml = ManagerChunk2::map_chunk2_to_compress_potentially_thr[thr_rank];

        for (auto& elm : ml)
        {
            const InT<T> second_val = elm.first; // the map key, by reference when T is large
            const auto   it         = ManagerChunk2::map_chunk2_to_compress_potentially_thr[thr_rank].find(second_val);
            if (it == ManagerChunk2::map_chunk2_to_compress_potentially_thr[thr_rank].end())
            {
                continue;
            }
            const auto* vec_tuple_p = it->second.get();

            if (vec_tuple_p)
            {
                // The map keeps a slot per second value ever declared; the
                // arrays that declared this one may all be gone (clear_core
                // drops their candidates by num_tab, not the slot). No array,
                // no info tuple, nothing to fold: skip it, do not dereference.
                const auto* inf = zero_one_value::Proxy<T>::share_info_array->find_inf_array(second_val);
                if (inf == nullptr)
                {
                    continue;
                }
                make_block_compression(thr_rank, *inf, nullptr);
            }
        }
    }
    else
    {
        if (is_vec_tuple_to_compress_potentially_empty_thr())
        {
            return;
        }

        const auto it = ManagerChunk2::map_chunk2_to_compress_potentially_thr[thr_rank].find(true);
        if (it == ManagerChunk2::map_chunk2_to_compress_potentially_thr[thr_rank].end())
        {
            return;
        }

        const auto* vec_tuple_p = it->second.get();

        if (vec_tuple_p)
        {
            make_block_compression_bool(thr_rank, nullptr);
        }
    }
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
 *          SERIAL build FastCompPolicy is the \<false\> specialisation, which has
 *          no compare_and_swap at all. So the lambda naming it must stay INSIDE
 *          the `if constexpr (use_fast_block_compression)` branch below -- a
 *          discarded branch is not instantiated. Passing that lambda in from
 *          the call site does NOT work, generic or not: once zero_one_value::Chunk<T> is
 *          instantiated FastCompPolicy is a concrete type, so the qualified
 *          name is looked up eagerly and the serial build fails to compile.
 */
template <typename T, typename Derived, bool Enabled>
template <bool IsBoolCas, typename InitFn>
void zero_one_value::ChunkOmpPolicy<T, Derived, Enabled>::create_value_impl(
    int                   thr_rank,
    InitFn&&              init_func,
    [[maybe_unused]] int  ind,
    [[maybe_unused]] bool need_set_second_value) noexcept
{
    T* values_loc = Base::allocate_chunk_internal(thr_rank, std::forward<InitFn>(init_func));

    if constexpr (chunk_config::use_fast_block_compression)
    {
        Base::apply_cas_or_recup(values_loc, thr_rank,
                                 [this, ind, need_set_second_value](T* cst_ptr)
                                 {
                                     if constexpr (IsBoolCas)
                                     {
                                         (void)ind; // only the non-bool CAS takes the index
                                         return derived().compare_and_swap_bool(cst_ptr, need_set_second_value);
                                     }
                                     else
                                     {
                                         // Propagate the observed state to the CAS so it refuses a
                                         // wrong-side transition after a cst_val1 <-> cst_val2 race.
                                         return derived().compare_and_swap(cst_ptr, ind, need_set_second_value);
                                     }
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
 * @note In a serial build there is no CAS to call — the policy has no such
 *       member — so the publication falls back to the non-CAS branch of
 *       apply_cas_or_recup().
 */
template <typename T, typename Derived, bool Enabled>
void zero_one_value::ChunkOmpPolicy<T, Derived, Enabled>::create_value(bool             need_set_second_value,
                                                                       const InfArrayT& inf_array,
                                                                       int              thr_rank) noexcept
{
    if constexpr (is_not_bool)
    {
        create_value_impl</*IsBoolCas=*/false>(
            thr_rank,
            [need_set_second_value, &inf_array](T* ptr)
            {
                if (!need_set_second_value)
                {
                    Base::init_block(ptr);
                }
                else
                {
                    BaseSpec::init_block_2nd_value(ptr, inf_array);
                }
            },
            inf_array.ind, need_set_second_value);
    }
    else
    {
        CHUNK_INVARIANT_FAILED("zero_one_value::Chunk::create_value(need_set_second_value, inf_array, rank) on a bool "
                               "chunk: the bool path has its own");
    }
}

/**
 * @brief Materialises a `bool` chunk.
 *
 * Same allocate-fill-CAS shape as create_value(), with no index to carry.
 */
template <typename T, typename Derived, bool Enabled>
void zero_one_value::ChunkOmpPolicy<T, Derived, Enabled>::create_value_bool(bool need_set_second_value,
                                                                            int  thr_rank) noexcept
{
    create_value_impl</*IsBoolCas=*/true>(
        thr_rank,
        [need_set_second_value](T* ptr)
        {
            if (!need_set_second_value)
            {
                Base::init_block(ptr);
            }
            else
            {
                BaseSpec::init_block_2nd_value(ptr);
            }
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
void zero_one_value::ChunkOmpPolicy<T, Derived, false>::create_value_serial_impl(bool        need_set_second_value,
                                                                                 Init2ndFn&& init_2nd) noexcept
{
    const auto init_func = [this, need_set_second_value, &init_2nd](T* ptr)
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

    values_ = Base::allocate_chunk_internal(0, init_func);
}

template <typename T, typename Derived>
void zero_one_value::ChunkOmpPolicy<T, Derived, false>::create_value(bool             need_set_second_value,
                                                                     const InfArrayT& inf_array) noexcept
{
    if constexpr (is_not_bool)
    {
        create_value_serial_impl(need_set_second_value,
                                 [&inf_array](T* ptr) { BaseSpec::init_block_2nd_value(ptr, inf_array); });
    }
    else
    {
        static_assert(dependent_false_v<T>,
                      "create_value(bool, inf_array) is for non-bool elements: use create_value_bool()");
    }
}

/**
 * @brief Materialises a `bool` chunk (serial build).
 */
template <typename T, typename Derived>
void zero_one_value::ChunkOmpPolicy<T, Derived, false>::create_value_bool(bool need_set_second_value) noexcept
{
    if constexpr (is_bool)
    {
        create_value_serial_impl(need_set_second_value, [](T* ptr) { BaseSpec::init_block_2nd_value(ptr); });
    }
    else
    {
        static_assert(dependent_false_v<T>,
                      "create_value_bool() is for bool elements: use create_value(bool, inf_array)");
    }
}

/**
 * @brief Compresses this chunk back to first_value, if it is materialised.
 */
template <typename T, typename Derived, bool Enabled>
void zero_one_value::ChunkOmpPolicy<T, Derived, Enabled>::clear(int ind, int thr_rank)
{
    Base::clear_impl(ind, thr_rank);
}

/**
 * @brief Compresses this chunk to first_value without testing `written()` (serial build).
 *
 * For callers that have already established the chunk is materialised; the test
 * survives as a debug-only assertion.
 */
template <typename T, typename Derived>
void zero_one_value::ChunkOmpPolicy<T, Derived, false>::clear_st(int ind)
{
    Base::debug_check_written(ind);

    Base::free_values(values_);
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
 */
template <typename T, typename Derived, bool Enabled>
void zero_one_value::ChunkOmpPolicy<T, Derived, Enabled>::block_clear(int                  num_compress_value,
                                                                      int                  ind,
                                                                      [[maybe_unused]] int thr_rank)
{
    // CHUNK_STATS: one materialised block released by a successful compression,
    // counted here: block_clear is the single point every variant and both
    // OMP/non-OMP modes funnel through.
    increm_chunk_compress<Derived>();
    add_chunk_bytes_freed<Derived>(static_cast<long>(sizeof(T)) * static_cast<long>(alloc_chunk_count()));
    Base::block_clear_common(num_compress_value, ind, thr_rank);
}

/// Serial build: frees the buffer outright instead of parking it in a rank
/// pool, and resets the per-chunk element counters. The rank argument is
/// unused (documented with the declaration).
template <typename T, typename Derived>
void zero_one_value::ChunkOmpPolicy<T, Derived, false>::block_clear(int num_compress_value, int ind, int /*thr_rank*/)
{
    // CHUNK_STATS: as the OpenMP block_clear above.
    increm_chunk_compress<Derived>();
    add_chunk_bytes_freed<Derived>(static_cast<long>(sizeof(T)) * static_cast<long>(alloc_chunk_count()));
    Base::free_values(values_);

    derived().init_values_cst_val(num_compress_value, ind);
    if constexpr (chunk_config::use_compression_with_counter)
    {
        derived().reset_nb_val_eq_value();
    }
}

/**
 * @brief Compresses this chunk to the named constant, after an invariant check.
 * @pre The chunk is materialised — compressing an already-constant chunk would
 *      release a shared constant block. Checked by
 *      check_not_constant_before_clear().
 */
template <typename T, typename Derived, bool Enabled>
void zero_one_value::ChunkOmpPolicy<T, Derived, Enabled>::clear_int_st(int num_compress_value, int ind, int thr_rank)
{
    Base::check_not_constant_before_clear(num_compress_value, ind);
    block_clear(num_compress_value, ind, thr_rank);
}

/**
 * @brief Bool twin of clear_int_st(): no second-value index to pass.
 */
template <typename T, typename Derived, bool Enabled>
void zero_one_value::ChunkOmpPolicy<T, Derived, Enabled>::clear_int_st_bool(int num_compress_value, int thr_rank)
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
 * @pre The chunk is materialised (debug-only assertion).
 */
template <typename T, typename Derived>
void zero_one_value::ChunkOmpPolicy<T, Derived, false>::clear_int_st(int num_compress_value, int ind)
{
    // Debug-only check. Bool chunks share a single second value with no index:
    // no-arg written check and the bool-specific reinit; non-bool is indexed.
    if constexpr (is_bool)
    {
        Base::debug_check_written();
    }
    else
    {
        Base::debug_check_written(ind);
    }

    // CHUNK_STATS: counted here, since this path (compress_int's sweep via
    // clear_st(num_compress_value, ind)) bypasses block_clear. No if constexpr
    // needed: both are no-op overloads when CHUNK_STATS is off (chunk_stats.h).
    increm_chunk_compress<Derived>();
    add_chunk_bytes_freed<Derived>(static_cast<long>(sizeof(T)) * static_cast<long>(alloc_chunk_count()));

    Base::free_values(values_);

    if (num_compress_value == 0)
    {
        derived().reset_nb_val_eq_first_value();
    }
    else
    {
        derived().reset_nb_val_eq_second_value();
    }

    if constexpr (is_bool)
    {
        derived().init_values_cst_val_bool(num_compress_value);
    }
    else
    {
        derived().init_values_cst_val(num_compress_value, ind);
    }
}

// clear_st(int num_compress_value, int ind) -- the entry point that supplies
// the calling thread's rank (or has none to supply) -- is defined in-class on
// each Chunk2OmpPolicy specialisation.

} // end namespace shrink_array
