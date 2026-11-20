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

#include <cassert>     // assert (the candidate-list lookups)
#include <cstddef>     // std::size_t
#include <optional>    // std::nullopt (the clears)
#include <type_traits> // std::remove_cvref_t (the re-queued candidates)
#include <atomic>      // std::memory_order_acq_rel, std::memory_order_acquire, std::memory_order_release

#include "chunk/config.h"                       // build_config::chunk_dynamic, chunk_config::OMPTcc
#include "chunk/internal/pmp_chunk.h"           // CHUNK_PMP_*, pmp::team
#include "chunk/internal/chunk_size_policy.h"   // ChunkElems, set_thread_chunk_size
#include "chunk/internal/in_t.h"                // InT<T>
#include "chunk/internal/zero_or_value_chunk.h" // zero_or_value::Chunk, the class these bodies belong to
#include "chunk/internal/zero_or_value_proxy.h" // zero_or_value::Proxy<T>
#include "chunk/internal/block_pass_precheck.h" // block_pass_yields_to_flush
#include "chunk/internal/invariant_check.h"     // CHUNK_INVARIANT_FAILED
#include "chunk/internal/queue_flag.h"          // queue_flag::hand_back / release: the pass's hand-back
#include "chunk/internal/same_value.h"          // same_value

/**
 * @file zero_or_value_chunk_inline.h
 * @brief Out-of-line definitions of zero_or_value's policies: materialisation and compression.
 *
 * zero_or_value is the two-constant chunk whose second value is shared by the WHOLE
 * array — that is what separates it from zero_any, where each chunk carries its
 * own. The consequence runs through this entire file: the candidate lists are
 * keyed BY second value (`zero_or_value_candidates_map_thr[thr][value]`),
 * so a compression pass is always "drain the list of one (rank, second value)
 * pair", never "drain everything this rank queued".
 *
 * A chunk is in one of three states — compressed to first_value, compressed to
 * the array's second value, or materialised on a private heap buffer — and the
 * two directions between them are:
 *  - **materialise**: create_value() / create_value_bool() allocate a buffer,
 *    pre-fill it, and (under OpenMP) publish it with a CAS that refuses a
 *    wrong-side transition after a first_value/second_value race;
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
 *    maintain. The two live on the two specialisations of zero_or_value::ChunkThreadPolicy
 *    (build_config::threaded); the queue-driven compression is on
 *    zero_or_value::ChunkFastCompPolicy<..., true>.
 *  - **Geometry comes from the queued entry, never from the ambient size.**
 *    Every candidate tuple carries the chunk count of its owning array
 *    (field `<3>`), because the thread consuming the entry may have a different
 *    ambient size — after a rechunk, scanning at the ambient size overruns a
 *    buffer allocated at another geometry.
 *
 * @see zero_or_value_chunk.h for the declarations, chunk_base.h for the shared
 *      allocate/free machinery, zero_or_value_proxy_inline.h for the write paths that queue
 *      the candidates.
 */

namespace shrink_array
{

// ── Array-new (placement-new arrays; see chunk_array_new.h) ────────────────
// Out of line on purpose, as operator delete[] below: see there.
template <typename T, typename Derived, bool Enabled>
[[gnu::noinline]] void* zero_or_value::ChunkThreadPolicy<T, Derived, Enabled>::operator new[](std::size_t sz)
{
    auto* chunks = static_cast<zero_or_value::Chunk<T>*>(::operator new(sz));
    init_chunks_const_val(chunks, sz / sizeof(zero_or_value::Chunk<T>));
    return static_cast<void*>(chunks);
}

// Out of line on purpose, as operator new[] above: inlined, either one shows
// gcc's -Wmismatched-new-delete a ::operator new/delete paired with the class's
// other operator, a false positive. Kept as calls, it sees the class's own
// new[] / delete[] pair.
template <typename T, typename Derived, bool Enabled>
[[gnu::noinline]] void zero_or_value::ChunkThreadPolicy<T, Derived, Enabled>::operator delete[](void* ptr)
{
    ::operator delete(ptr);
}


// ---------------------------------------------------------------------------
// zero_or_value::ChunkFastCompPolicy<T, Derived, true>: the queue-driven compression.
// ---------------------------------------------------------------------------

/**
 * @brief Scans this chunk and files it into one of the two compression lists.
 *
 * Thin wrapper over classify_candidate_impl(): it unpacks the array-wide
 * second value from @p array_info and supplies the current second_value test.
 *
 * @param ll_compt1  Output list of chunks compressible to first_value.
 * @param ll_compt2  Output list of chunks compressible to the second value.
 * @param array_info  (second_value, index, ...) of the owning array.
 * @param chunk_index  Index of this chunk in its array.
 * @param array_id    Table number of the owning array.
 * @param n_elems    This chunk's own element count, captured when it was queued.
 */
template <typename T, typename Derived, bool Enabled>
void zero_or_value::ChunkFastCompPolicy<T, Derived, Enabled>::classify_candidate(this Derived&    self,
                                                                                 CandidateTuples& ll_compt1,
                                                                                 CandidateTuples& ll_compt2,
                                                                                 const ArrayInfo& array_info,
                                                                                 int              chunk_index,
                                                                                 int              array_id,
                                                                                 unsigned int     n_elems)
{
    // InT: a reference into the (const, caller-owned) tuple when T is large.
    const InT<T> second_v = array_info.second_value;
    self.classify_candidate_impl(
        ll_compt1, ll_compt2, chunk_index, array_id, [&second_v]() -> InT<T> { return second_v; },
        self.is_not_second_value(array_info.idx), n_elems);
}

/**
 * @brief Shared body of the two classify_candidate entry points.
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
 * @param chunk_index        Index of this chunk in its array.
 * @param array_id          Table number of the owning array.
 * @param get_second_value Supplies the second value; deferred so the bool twin
 *                         can pass a constant and skip the tuple lookup.
 * @param check_second_value   True when the chunk is NOT currently compressed to its
 *                         second value — i.e. when the scan is worth doing. The
 *                         name reads as the opposite of what it holds.
 * @param n_elems          This chunk's own element count, captured at enqueue.
 */
template <typename T, typename Derived, bool Enabled>
template <typename SecondValueFunc>
void zero_or_value::ChunkFastCompPolicy<T, Derived, Enabled>::classify_candidate_impl(
    this Derived&     self,
    CandidateTuples&  ll_compt1,
    CandidateTuples&  ll_compt2,
    int               chunk_index,
    int               array_id,
    SecondValueFunc&& get_second_value,
    bool              check_second_value,
    unsigned int      n_elems)
{
    // Skip a chunk already compressed to either constant.
    if (self.Base::is_not_first_value() && check_second_value)
    {
        // CHUNK_STATS: one compression candidate scanned (see zero_any).
        inc_chunk_compress_attempt<Derived>();

        // n_elems: see zero_or_value::Chunk<T>::compress() -- the chunk's own geometry,
        // captured at enqueue, never the ambient alloc_chunk_count().
        const auto scan = self.Base::scan_constants(first_value, get_second_value(), n_elems);

        if (scan.first)
        {
            ll_compt1.emplace_back(std::make_tuple(&self, chunk_index, array_id, n_elems));
        }
        if (scan.second)
        {
            ll_compt2.emplace_back(std::make_tuple(&self, chunk_index, array_id, n_elems));
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
 * @tparam ClearFunc       Callable `(int const_index)` applying the clear.
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
void zero_or_value::ChunkFastCompPolicy<T, Derived, Enabled>::compress_impl(this Derived&   self,
                                                                            bool            apply,
                                                                            unsigned int    n_elems,
                                                                            SecondValueFunc get_second_value,
                                                                            ClearFunc&&     clear_one)
{
    if (self.Base::is_not_first_value())
    {
        // CHUNK_STATS: a SECOND candidate scan, independent of
        // classify_candidate, so it counts its own attempt. It reads values[]
        // directly, so it never inflates `read` and needs no read_value_scan.
        inc_chunk_compress_attempt<Derived>();

        // n_elems is THIS chunk's own geometry, captured at enqueue (tuple
        // field <3>), never alloc_chunk_count() -- the consuming thread's
        // ambient size overflows a buffer allocated at another geometry after
        // a rechunk. Same rule as zero and zero_any.
        //
        // Always scan, even when the second value equals first_value: skipping
        // would leave BOTH verdicts `true` and clear the chunk twice.
        // scan_constants reports such a run as `first` only.
        const auto scan = self.Base::scan_constants(first_value, get_second_value(), n_elems);

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
void zero_or_value::ChunkFastCompPolicy<T, Derived, Enabled>::compress(this Derived&        self,
                                                                       [[maybe_unused]] int thr_rank,
                                                                       const ArrayInfo&     array_info,
                                                                       bool                 apply,
                                                                       unsigned int         n_elems)
{
    // Only a MATERIALISED chunk can be compressed. A queued entry can outlive
    // the chunk's materialisation: make_block_compression (or a previous
    // drain) compresses it, and the OMPT end-of-region drain then replays
    // the stale entry. compress_impl only refuses first_value; a chunk already at
    // second_value would pass, scan the SHARED second-value block, find it
    // constant and clear it a second time -- freeing that shared block.
    if (!self.written(array_info.idx))
    {
        return;
    }

    self.compress_impl(
        apply, n_elems, [&array_info] { return array_info.second_value; },
        [&self, thr_rank, &array_info](int const_index) { self.clear_int_st(const_index, array_info.idx, thr_rank); });
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
void zero_or_value::ChunkFastCompPolicy<T, Derived, Enabled>::compress_bool(this Derived&        self,
                                                                            [[maybe_unused]] int thr_rank,
                                                                            bool                 apply,
                                                                            unsigned int         n_elems)
{
    if constexpr (is_bool)
    {
        if (!self.written()) // see compress(): a stale entry for a chunk already at a constant
        {
            return;
        }
        self.compress_impl(
            apply, n_elems, [] { return true; },
            [&self, thr_rank](int const_index) { self.clear_int_st_bool(const_index, thr_rank); });
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
void zero_or_value::ChunkFastCompPolicy<T, Derived, Enabled>::compress_queued(const ArrayInfo& array_info, bool apply)
{
    // Everything the per-rank body touches is rank-local --
    // zero_or_value_candidates_map_thr[thr_rank], the freed buffer goes to
    // thr_rank's pool (clear_int_st), and the candidate flag reset is an atomic
    // store -- so the ranks are independent (a serial loop leaves the team idle
    // during a re-chunk).
    const auto sweep = [&array_info, apply](int thr) { compress_rank_queue(thr, array_info, apply); };
    Base::sweep_every_rank(sweep, sweep);
}

/**
 * @brief Bool twin of compress_queued().
 */
template <typename T, typename Derived, bool Enabled>
void zero_or_value::ChunkFastCompPolicy<T, Derived, Enabled>::compress_queued_bool(bool apply)
{
    if constexpr (is_bool)
    {
        // Same as compress_queued(): rank-local sweeps.
        const auto sweep = [apply](int thr) { compress_rank_queue_bool(thr, apply); };
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
 * That flag is the CAS gate of ProxyBase::queue_as_candidate(): its 0 -> 1
 * flip admits a chunk to the writer's candidate list and is the only thing
 * keeping a chunk out of two lists at once. Cleared before compress(), it lets
 * a concurrent write re-queue the SAME chunk mid-compression, and a later drain
 * of that entry double-clears an already-constant chunk (the zero_any @warning;
 * check_not_constant_before_clear fires). Compressing first keeps the flag at 1
 * throughout, so every concurrent write loses the CAS until the chunk reaches
 * its final state.
 *
 * Today the drain has no concurrent writers (on_parallel_end() puts an
 * `omp barrier` in front of it; the zero_or_value::Array call sites are gated on
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
void zero_or_value::ChunkFastCompPolicy<T, Derived, Enabled>::compress_rank_queue_impl(int          thr_rank,
                                                                                       InT<T>       key,
                                                                                       CompressFn&& compress_one)
{
    // The store is a per-rank VECTOR of maps, sized by resize_all_list_block_free_thr();
    // until that has run, [thr_rank] is out of bounds (UB). Probing rank 0 is
    // enough: zero_or_value::Array::set_second_value registers a key for every rank in
    // one loop, so an empty map at rank 0 means no key at any rank.
    if (no_candidates())
    {
        return;
    }

    auto* const vec_tuple_p = Manager::compress_candidates(thr_rank, key);
    assert(vec_tuple_p != nullptr && "key registered for every rank by set_second_value()");
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
            const int chunk_index = std::get<1>(tup);
            const int array_id    = std::get<2>(tup);

            if constexpr (build_config::chunk_dynamic)
            {
                shrink_array::set_thread_chunk_size(std::get<3>(tup).count());
            }

            compress_one(chunk_t, std::get<3>(tup).count());

            // AFTER the compression, never before -- see the ORDER note above.
            Proxy<T>::clear_candidate_flag(array_id, chunk_index);
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
 * @param array_info      (second_value, index, ...) selecting the list.
 * @param apply          Forwarded to compress().
 */
template <typename T, typename Derived, bool Enabled>
void zero_or_value::ChunkFastCompPolicy<T, Derived, Enabled>::compress_rank_queue(int              thr_rank,
                                                                                  const ArrayInfo& array_info,
                                                                                  bool             apply)
{
    compress_rank_queue_impl(thr_rank, array_info.second_value,
                             [thr_rank, &array_info, apply](auto* chunk_t, unsigned int n_elems)
                             { chunk_t->compress(thr_rank, array_info, apply, n_elems); });
}

/**
 * @brief Bool twin of compress_rank_queue().
 * @param thr_rank       Rank whose list is drained.
 * @param apply          Forwarded to compress_bool().
 */
template <typename T, typename Derived, bool Enabled>
void zero_or_value::ChunkFastCompPolicy<T, Derived, Enabled>::compress_rank_queue_bool(int thr_rank, bool apply)
{
    if constexpr (is_bool)
    {
        compress_rank_queue_impl(thr_rank, true, [thr_rank, apply](auto* chunk_t, unsigned int n_elems)
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
 * @tparam ScanFn callable `(ChunkType*, int chunk_index, int array_id, ChunkElems)`.
 * @param thr_rank Rank whose candidate list is scanned.
 * @param key      Second value indexing the per-rank candidate map.
 * @param scan_one Per-chunk scan.
 */
template <typename T, typename Derived, bool Enabled>
template <typename ScanFn>
void zero_or_value::ChunkFastCompPolicy<T, Derived, Enabled>::classify_rank_candidates_impl(int      thr_rank,
                                                                                            InT<T>   key,
                                                                                            ScanFn&& scan_one)
{
    const auto* const vec_tuple_p = Manager::compress_candidates(thr_rank, key);
    assert(vec_tuple_p != nullptr && "key registered for every rank by set_second_value()");
    if (vec_tuple_p == nullptr) [[unlikely]]
    {
        return;
    }
    const auto& vec_tuple = *vec_tuple_p;

    const auto nn = static_cast<int>(vec_tuple.size());

    for (int k = 0; k < nn; ++k)
    {
        const auto& tup     = vec_tuple[k];
        const auto& chunk_t = std::get<0>(tup);
        if (chunk_t)
        {
            const int& chunk_index = std::get<1>(tup);
            const int  array_id    = std::get<2>(tup);
            // Per-entry geometry (see zero_or_value_chunk.h): in dynamic builds each
            // queued chunk carries the chunk size of its owning array.
            scan_one(chunk_t, chunk_index, array_id, std::get<3>(tup).count());
        }
    }
}

/**
 * @brief Scans one rank's candidates for one second value into the two lists.
 * @param thr_rank  Rank whose candidate list is scanned.
 * @param ll_compt1 Output: chunks compressible to first_value.
 * @param ll_compt2 Output: chunks compressible to the second value.
 * @param array_info (second_value, index, ...) selecting the list.
 */
template <typename T, typename Derived, bool Enabled>
void zero_or_value::ChunkFastCompPolicy<T, Derived, Enabled>::classify_rank_candidates(int              thr_rank,
                                                                                       CandidateTuples& ll_compt1,
                                                                                       CandidateTuples& ll_compt2,
                                                                                       const ArrayInfo& array_info)
{
    classify_rank_candidates_impl(
        thr_rank, array_info.second_value,
        [&ll_compt1, &ll_compt2, &array_info](auto* chunk_t, int chunk_index, int array_id, unsigned int n_elems)
        {
            const auto n1 = ll_compt1.size();
            const auto n2 = ll_compt2.size();
            chunk_t->classify_candidate(ll_compt1, ll_compt2, array_info, chunk_index, array_id, n_elems);
            // Scanned, still marked: hand its queue flag back before the mark can go down,
            // here for the chunks the scan drops (queue_flag.h, F6).
            queue_flag::hand_back(Proxy<T>::candidate_flag_row(array_id)[chunk_index]);
            // Not selected by the scan: the writers may have it back now.
            if (ll_compt1.size() == n1 && ll_compt2.size() == n2)
            {
                chunk_t->steal_unmark();
            }
        });
}

/**
 * @brief Bool twin of classify_rank_candidates().
 * @param thr_rank  Rank whose candidate list is scanned.
 * @param ll_compt1 Output: chunks compressible to first_value.
 * @param ll_compt2 Output: chunks compressible to `true`.
 */
template <typename T, typename Derived, bool Enabled>
void zero_or_value::ChunkFastCompPolicy<T, Derived, Enabled>::classify_rank_candidates_bool(int              thr_rank,
                                                                                            CandidateTuples& ll_compt1,
                                                                                            CandidateTuples& ll_compt2)
{
    classify_rank_candidates_impl(
        thr_rank, static_cast<T>(true),
        [&ll_compt1, &ll_compt2](auto* chunk_t, int chunk_index, int array_id, unsigned int n_elems)
        {
            const auto n1 = ll_compt1.size();
            const auto n2 = ll_compt2.size();
            chunk_t->classify_candidate_bool(ll_compt1, ll_compt2, chunk_index, array_id, n_elems);
            // Scanned, still marked: hand its queue flag back first (queue_flag.h, F6).
            queue_flag::hand_back(Proxy<T>::candidate_flag_row(array_id)[chunk_index]);
            if (ll_compt1.size() == n1 && ll_compt2.size() == n2)
            {
                chunk_t->steal_unmark();
            }
        });
}

// --- The sweep. Precedent: ChunkFastCompPolicy<T,Derived,Enabled>::
// make_block_compression for zero::Chunk<T>.

// make_block_compression_impl (declared in zero_or_value_chunk.h). Shape: pin geometry ->
// take the compression handshake -> scan into the two buckets -> clear each ->
// reset every queued flag -> empty the list -> re-queue the triggering chunk.
// The two callables are the only bool/non-bool difference.
// Geometry: as in compress_rank_queue_impl(), the list mixes
// arrays of different chunk sizes, so each chunk is pinned to its own n_elems
// right before its clear (as zero_any::Chunk<T>::make_block_compression() does). Do not
// rely on the single pin on the triggering proxy's size (kept, for the
// handshake): it would park the other array's buffers under the wrong size.
template <typename T, typename Derived, bool Enabled>
template <typename FillFunc, typename ClearFunc>
void zero_or_value::ChunkFastCompPolicy<T, Derived, Enabled>::make_block_compression_impl(
    int                      thr_rank,
    T                        second_value,
    zero_or_value::Proxy<T>* proxy2,
    FillFunc&&               fill_func,
    ClearFunc&&              clear_func)
{
    // find, NOT operator[]: on a missing key operator[] inserts a NULL
    // shared_ptr that the next dereference crashes on.
    //
    // A missing key is normal (no one declared this second value on this
    // rank: nothing to compress, same exit as `sz == 0`); a zero_or_value::Array built
    // without set_second_value reaches it via resize_common ->
    // make_block_compression. no_candidates()
    // is not enough: it only tests "no key anywhere".
    auto&      map_thr = Manager::zero_or_value_candidates_map_thr[thr_rank];
    const auto it      = map_thr.find(second_value);

    if (it == map_thr.end() || !it->second)
    {
        return;
    }

    auto& candidates = *it->second;

    const auto sz = static_cast<int>(candidates.size());

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
        if (block_pass_yields_to_flush<Proxy<T>, Chunk<T>>(CHUNK_PMP_MAX_THREADS))
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
            if (auto* const chunk_t = std::get<0>(candidates[k]))
            {
                chunk_t->steal_mark();
            }
        }
        if (const bool ret = proxy2->compression_mgr.template start<Proxy<T>>(); !ret)
        {
            for (int k = 0; k < sz; ++k)
            {
                if (auto* const chunk_t = std::get<0>(candidates[k]))
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
                if (auto* const chunk_t = std::get<0>(candidates[k]))
                {
                    chunk_t->steal_commit();
                }
            }
        }
    }

    CandidateTuples ll_compt1;
    CandidateTuples ll_compt2;

    ll_compt1.reserve((sz / 50) + 1);
    ll_compt2.reserve((sz / 50) + 1);

    // The scan hands every candidate's queue flag back as it goes, before any mark goes
    // down (classify_rank_candidates*: queue_flag.h, F6); the loop at the end
    // releases them.
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

    queue_flag::release_candidates(
        candidates, sz,
        [](int array_id, int chunk_index) -> auto& { return Proxy<T>::candidate_flag_row(array_id)[chunk_index]; },
        [](int array_id, int chunk_index) { Proxy<T>::clear_candidate_flag(array_id, chunk_index); });

    if (proxy2)
    {
        // Re-queue the writer's chunk only into the list of ITS array's second
        // value. zero_or_value::Proxy::perform_compression runs the pass of every key of the
        // rank with the same writer: queued under another array's key, the
        // chunk could be compressed to the OTHER array's constant, and its own
        // array, which only recognises its own constant, would then write into
        // that shared block, changing every chunk compressed to it.
        if (proxy2->is_not_const_val() && same_value(proxy2->second_value_, second_value))
        {
            proxy2->queue_as_candidate(candidates);
        }

        proxy2->compression_mgr.template finish<Proxy<T>>();
    }

    if constexpr (build_config::chunk_dynamic)
    {
        set_thread_chunk_size(saved_sz);
    }
}

// make_block_compression (declared in zero_or_value_chunk.h). Binds the non-bool scan and
// clear callables and hands them to the shared sweep. Instantiating it for bool
// is a compile error (static_assert).
template <typename T, typename Derived, bool Enabled>
void zero_or_value::ChunkFastCompPolicy<T, Derived, Enabled>::make_block_compression(int              thr_rank,
                                                                                     const ArrayInfo& array_info,
                                                                                     zero_or_value::Proxy<T>* proxy2)
{
    if constexpr (is_not_bool)
    {
        const InT<T> second_value = array_info.second_value; // the caller's record, alive for the call

        const auto lbd_clear_func = [thr_rank, &array_info](auto* chunk_t, int clear_type)
        { chunk_t->clear_int_st(clear_type, array_info.idx, thr_rank); };

        make_block_compression_impl(
            thr_rank, second_value, proxy2, [thr_rank, &array_info](auto& ll_compt1, auto& ll_compt2)
            { Derived::classify_rank_candidates(thr_rank, ll_compt1, ll_compt2, array_info); }, lbd_clear_func);
    }
    else
    {
        static_assert(false, "make_block_compression(thr, array_info, proxy) is for non-bool elements: "
                             "use make_block_compression_bool()");
    }
}

// make_block_compression_bool (declared in zero_or_value_chunk.h). Same binding for bool:
// second value is `true`, and the clear goes through clear_int_st_bool, which
// needs no index. Mirror image of the guard above -- non-bool fails to compile.
template <typename T, typename Derived, bool Enabled>
void zero_or_value::ChunkFastCompPolicy<T, Derived, Enabled>::make_block_compression_bool(
    int                      thr_rank,
    zero_or_value::Proxy<T>* proxy2)
{
    if constexpr (is_bool)
    {
        const auto lbd_clear_func = [thr_rank](auto* chunk_t, int clear_type)
        { chunk_t->clear_int_st_bool(clear_type, thr_rank); };

        make_block_compression_impl(
            thr_rank, true, proxy2, [thr_rank](auto& ll_compt1, auto& ll_compt2)
            { Derived::classify_rank_candidates_bool(thr_rank, ll_compt1, ll_compt2); }, lbd_clear_func);
    }
    else
    {
        static_assert(
            false,
            "make_block_compression_bool() is for bool elements: use make_block_compression(thr, array_info, proxy)");
    }
}

// make_block_compression, thread-only overload (declared in zero_or_value_chunk.h). One
// sweep per distinct second value this rank has queued (the candidate map is
// keyed by value). Passes nullptr as the proxy: no handshake and no proxy-size
// pin (each freed chunk is still pinned to its own size).
template <typename T, typename Derived, bool Enabled>
void zero_or_value::ChunkFastCompPolicy<T, Derived, Enabled>::make_block_compression(int thr_rank)
{
    if constexpr (is_not_bool)
    {
        const auto& ml = Manager::zero_or_value_candidates_map_thr[thr_rank];

        for (auto& elm : ml)
        {
            const InT<T> second_val = elm.first; // the map key, by reference when T is large
            const auto   it         = Manager::zero_or_value_candidates_map_thr[thr_rank].find(second_val);
            if (it == Manager::zero_or_value_candidates_map_thr[thr_rank].end())
            {
                continue;
            }
            const auto* vec_tuple_p = it->second.get();

            if (vec_tuple_p)
            {
                // The map keeps a slot per second value ever declared; the
                // arrays that declared this one may all be gone (clear_core
                // drops their candidates by array_id, not the slot). No array,
                // no info tuple, nothing to fold: skip it, do not dereference.
                const auto* array_info = Proxy<T>::array_registry->find_array_info(second_val);
                if (array_info == nullptr)
                {
                    continue;
                }
                make_block_compression(thr_rank, *array_info, nullptr);
            }
        }
    }
    else
    {
        if (no_candidates())
        {
            return;
        }

        const auto it = Manager::zero_or_value_candidates_map_thr[thr_rank].find(true);
        if (it == Manager::zero_or_value_candidates_map_thr[thr_rank].end())
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
// zero_or_value::ChunkThreadPolicy<T, Derived, true>: materialise and clear, threaded mode.
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
 * @param idx                  Index of the array's second value; unused when
 *                             @p IsBoolCas (bool has no index).
 * @param need_set_second_value Which constant the block was filled with.
 *
 * @warning The CAS exists ONLY on the fast-block-compression policy: in a
 *          SERIAL build FastCompPolicy is the \<false\> specialisation, which has
 *          no compare_and_swap at all. So the lambda naming it must stay INSIDE
 *          the `if constexpr (use_fast_block_compression)` branch below -- a
 *          discarded branch is not instantiated. Passing that lambda in from
 *          the call site does NOT work, generic or not: once zero_or_value::Chunk<T> is
 *          instantiated FastCompPolicy is a concrete type, so the qualified
 *          name is looked up eagerly and the serial build fails to compile.
 */
template <typename T, typename Derived, bool Enabled>
template <bool IsBoolCas, typename InitFn>
void zero_or_value::ChunkThreadPolicy<T, Derived, Enabled>::create_value_impl(
    this Derived&         self,
    int                   thr_rank,
    InitFn&&              init_func,
    [[maybe_unused]] int  idx,
    [[maybe_unused]] bool need_set_second_value) noexcept
{
    T* values_loc = self.Base::allocate_chunk_internal(thr_rank, std::forward<InitFn>(init_func));

    if constexpr (chunk_config::use_fast_block_compression)
    {
        self.Base::apply_cas_or_reclaim(values_loc, thr_rank,
                                        [&self, idx, need_set_second_value](T* const_ptr)
                                        {
                                            if constexpr (IsBoolCas)
                                            {
                                                (void)idx; // only the non-bool CAS takes the index
                                                return self.compare_and_swap_bool(const_ptr, need_set_second_value);
                                            }
                                            else
                                            {
                                                // Propagate the observed state to the CAS so it refuses a
                                                // wrong-side transition after a first_value <-> second_value race.
                                                return self.compare_and_swap(const_ptr, idx, need_set_second_value);
                                            }
                                        });
    }
    else
    {
        // apply_cas_or_reclaim takes its non-CAS branch in this mode anyway.
        self.Base::apply_cas_or_reclaim(values_loc, thr_rank, [](T*) { return false; });
    }
}

/**
 * @brief Materialises this chunk: allocate a buffer, fill it, publish it by CAS.
 *
 * The buffer is filled with first_value or the array's second value BEFORE it
 * is published, and publication is a single CAS. A reader therefore sees the
 * old constant or the finished buffer, never a half-built one. The CAS is told
 * which constant it is replacing, so it refuses a wrong-side transition after a
 * first_value/second_value race; losing it means another thread materialised first,
 * and apply_cas_or_reclaim() recycles our buffer.
 *
 * @note In a serial build there is no CAS to call — the policy has no such
 *       member — so the publication falls back to the non-CAS branch of
 *       apply_cas_or_reclaim().
 */
template <typename T, typename Derived, bool Enabled>
void zero_or_value::ChunkThreadPolicy<T, Derived, Enabled>::create_value(this Derived&    self,
                                                                         bool             need_set_second_value,
                                                                         const ArrayInfo& array_info,
                                                                         int              thr_rank) noexcept
{
    if constexpr (is_not_bool)
    {
        self.template create_value_impl</*IsBoolCas=*/false>(
            thr_rank,
            [need_set_second_value, &array_info](T* ptr)
            {
                if (!need_set_second_value)
                {
                    Base::init_block(ptr);
                }
                else
                {
                    BaseSpec::init_block_2nd_value(ptr, array_info);
                }
            },
            array_info.idx, need_set_second_value);
    }
    else
    {
        static_assert(false, "create_value(bool, array_info, thr) is for non-bool elements: use create_value_bool()");
    }
}

/**
 * @brief Materialises a `bool` chunk.
 *
 * Same allocate-fill-CAS shape as create_value(), with no index to carry.
 */
template <typename T, typename Derived, bool Enabled>
void zero_or_value::ChunkThreadPolicy<T, Derived, Enabled>::create_value_bool(this Derived& self,
                                                                              bool          need_set_second_value,
                                                                              int           thr_rank) noexcept
{
    self.template create_value_impl</*IsBoolCas=*/true>(
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
        /*idx=*/0, need_set_second_value);
}

// ---------------------------------------------------------------------------
// zero_or_value::ChunkThreadPolicy<T, Derived, false>: materialise and clear, serial mode.
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
void zero_or_value::ChunkThreadPolicy<T, Derived, false>::create_value_serial_impl(this Derived& self,
                                                                                   bool          need_set_second_value,
                                                                                   Init2ndFn&&   init_2nd) noexcept
{
    const auto init_func = [&self, need_set_second_value, &init_2nd](T* ptr)
    {
        // The counter reset is CROSSED on purpose: filling with one constant
        // invalidates the count kept for the OTHER one, which is the one that
        // has to go back to zero.
        if (!need_set_second_value)
        {
            Base::init_block(ptr);
            self.reset_n_val_eq_second_value();
        }
        else
        {
            init_2nd(ptr);
            self.reset_n_val_eq_first_value();
        }
    };

    self.values_ = self.Base::allocate_chunk_internal(0, init_func);
}

template <typename T, typename Derived>
void zero_or_value::ChunkThreadPolicy<T, Derived, false>::create_value(this Derived&    self,
                                                                       bool             need_set_second_value,
                                                                       const ArrayInfo& array_info) noexcept
{
    if constexpr (is_not_bool)
    {
        self.create_value_serial_impl(need_set_second_value,
                                      [&array_info](T* ptr) { BaseSpec::init_block_2nd_value(ptr, array_info); });
    }
    else
    {
        static_assert(false, "create_value(bool, array_info) is for non-bool elements: use create_value_bool()");
    }
}

/**
 * @brief Materialises a `bool` chunk (serial build).
 */
template <typename T, typename Derived>
void zero_or_value::ChunkThreadPolicy<T, Derived, false>::create_value_bool(this Derived& self,
                                                                            bool need_set_second_value) noexcept
{
    if constexpr (is_bool)
    {
        self.create_value_serial_impl(need_set_second_value, [](T* ptr) { BaseSpec::init_block_2nd_value(ptr); });
    }
    else
    {
        static_assert(false, "create_value_bool() is for bool elements: use create_value(bool, array_info)");
    }
}

/**
 * @brief Compresses this chunk to first_value without testing `written()` (serial build).
 *
 * For callers that have already established the chunk is materialised; the test
 * survives as a debug-only assertion.
 */
template <typename T, typename Derived>
void zero_or_value::ChunkThreadPolicy<T, Derived, false>::clear_st(this Derived& self, int idx)
{
    self.Base::debug_check_written(idx);

    self.Base::free_values(self.values_);
    self.init_to_first_value();

    if constexpr (chunk_config::use_compression_with_counter)
    {
        self.reset_n_val_eq_value();
    }
}

template <typename T, typename Derived, bool Enabled>
void zero_or_value::ChunkThreadPolicy<T, Derived, Enabled>::release_st(this Derived& self, int idx, int thr_rank)
{
    // CHUNK_STATS: a release, not a compression (block_clear counts those): the
    // bytes still leave the live figure. block_clear_common's CAS cannot lose --
    // the source of a refill is quiescent -- and the buffer goes through
    // free_block_to_pool(), routed by the ambient geometry (the caller's: the
    // old one) and by the slab's owner key, as a compression's would.
    if constexpr (chunk_config::stats)
    {
        add_chunk_released<Derived>(1, static_cast<long>(sizeof(T)) * static_cast<long>(alloc_chunk_count()));
    }
    self.Base::block_clear_common(0, idx, thr_rank);
}

/**
 * @brief Releases this chunk's buffer and points it back at a constant.
 *
 * The single point every compression path funnels through, in both modes and
 * every chunk variant — which is why the "block compressed" and "bytes freed"
 * statistics are counted here and nowhere else.
 */
template <typename T, typename Derived, bool Enabled>
void zero_or_value::ChunkThreadPolicy<T, Derived, Enabled>::block_clear(this Derived&        self,
                                                                        int                  const_index,
                                                                        int                  idx,
                                                                        [[maybe_unused]] int thr_rank)
{
    // CHUNK_STATS: one materialised block released by a successful compression,
    // counted here: block_clear is the single point every variant and both
    // threaded/serial modes funnel through.
    inc_chunk_compress<Derived>();
    add_chunk_bytes_freed<Derived>(static_cast<long>(sizeof(T)) * static_cast<long>(alloc_chunk_count()));
    self.Base::block_clear_common(const_index, idx, thr_rank);
}

/// Serial build: frees the buffer outright instead of parking it in a rank
/// pool, and resets the per-chunk element counters. The rank argument is
/// unused (documented with the declaration).
template <typename T, typename Derived>
void zero_or_value::ChunkThreadPolicy<T, Derived, false>::block_clear(this Derived& self,
                                                                      int           const_index,
                                                                      int           idx,
                                                                      int /*thr_rank*/)
{
    // CHUNK_STATS: as the threaded block_clear above.
    inc_chunk_compress<Derived>();
    add_chunk_bytes_freed<Derived>(static_cast<long>(sizeof(T)) * static_cast<long>(alloc_chunk_count()));
    self.Base::free_values(self.values_);

    self.init_values_const_val(const_index, idx);
    if constexpr (chunk_config::use_compression_with_counter)
    {
        self.reset_n_val_eq_value();
    }
}

/**
 * @brief Bool twin of clear_int_st(): no second-value index to pass.
 */
template <typename T, typename Derived, bool Enabled>
void zero_or_value::ChunkThreadPolicy<T, Derived, Enabled>::clear_int_st_bool(this Derived& self,
                                                                              int           const_index,
                                                                              int           thr_rank)
{
    if constexpr (is_bool)
    {
        self.Base::check_not_constant_before_clear(const_index, std::nullopt);
        self.block_clear(const_index, 0, thr_rank);
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
void zero_or_value::ChunkThreadPolicy<T, Derived, false>::clear_int_st(this Derived& self, int const_index, int idx)
{
    // Debug-only check. Bool chunks share a single second value with no index:
    // no-arg written check and the bool-specific reinit; non-bool is indexed.
    if constexpr (is_bool)
    {
        self.Base::debug_check_written();
    }
    else
    {
        self.Base::debug_check_written(idx);
    }

    // CHUNK_STATS: counted here, since this path (compress_int's sweep via
    // clear_st(const_index, idx)) bypasses block_clear. No if constexpr
    // needed: both are no-op overloads when CHUNK_STATS is off (chunk_stats.h).
    inc_chunk_compress<Derived>();
    add_chunk_bytes_freed<Derived>(static_cast<long>(sizeof(T)) * static_cast<long>(alloc_chunk_count()));

    self.Base::free_values(self.values_);

    if (const_index == 0)
    {
        self.reset_n_val_eq_first_value();
    }
    else
    {
        self.reset_n_val_eq_second_value();
    }

    if constexpr (is_bool)
    {
        self.init_values_const_val_bool(const_index);
    }
    else
    {
        self.init_values_const_val(const_index, idx);
    }
}

// clear_st(int const_index, int idx) -- the entry point that supplies
// the calling thread's rank (or has none to supply) -- is defined in-class on
// each zero_or_value::ChunkThreadPolicy specialisation.

// ---------------------------------------------------------------------------
// shrink_array::ChunkSpec<T>
// ---------------------------------------------------------------------------

template <typename T>
[[nodiscard]] T* zero_or_value::ChunkSpec<T>::second_value_ptr(int idx) noexcept
{
    // An array with no second value (chunk_config::no_second_value_index)
    // has no block: nullptr, which no chunk's `values` ever equals.
    if (idx < 0)
    {
        return nullptr;
    }
    if constexpr (build_config::any_check)
    {
        if (static_cast<size_t>(idx) >= second_value_buffers->size()) [[unlikely]]
        {
            CHUNK_INVARIANT_FAILED("[chunk] suspect second_value_ptr<{}>: idx={} size={}", typeid(T).name(), idx,
                                   second_value_buffers->size());
            return nullptr;
        }
    }

    T* ret = nullptr;
    if constexpr (use_mm_protect)
    {
        ret = (*second_value_buffers)[idx];
    }
    else
    {
        ret = (*second_value_buffers)[idx].data();
    }
    CHUNK_CHECK_VALUES_PTR(ret, "second_value_ptr");
    return ret;
}

template <typename T>
[[nodiscard]] bool zero_or_value::ChunkSpec<T>::is_any_second_value(const T* vals) noexcept
{
    const auto n_second_values = static_cast<int>(second_value_buffers->size());
    for (int i = 0; i < n_second_values; ++i)
    {
        if (vals == second_value_ptr(i))
        {
            return true;
        }
    }
    return false;
}

template <typename T>
    requires is_bool_like_v<T>
void zero_or_value::ChunkSpec<T>::init_block_2nd_value(T* values_loc)
{
    if constexpr (is_packed_bit_v<T>)
    {
        bits::fill_bits(internal::as_words(values_loc), alloc_chunk_count(), true);
    }
    else
    {
        std::fill_n(values_loc, alloc_chunk_count(), true);
    }
}

// ---------------------------------------------------------------------------
// shrink_array::ChunkThreadPolicy<T, Derived, Enabled>
// ---------------------------------------------------------------------------

template <typename T, typename Derived, bool Enabled>
void zero_or_value::ChunkThreadPolicy<T, Derived, Enabled>::create_value(this Derived&    self,
                                                                         bool             need_set_second_value,
                                                                         const ArrayInfo& array_info) noexcept
{
    if constexpr (is_not_bool)
    {
        self.create_value(need_set_second_value, array_info, get_thr_rank_fast());
    }
    else
    {
        static_assert(false, "create_value(bool, array_info) is for non-bool elements: use create_value_bool()");
    }
}

template <typename T, typename Derived, bool Enabled>
void zero_or_value::ChunkThreadPolicy<T, Derived, Enabled>::create_value_bool(this Derived& self,
                                                                              bool need_set_second_value) noexcept
{
    if constexpr (is_bool)
    {
        self.create_value_bool(need_set_second_value, get_thr_rank_fast());
    }
    else
    {
        static_assert(false, "create_value_bool() is for bool elements: use create_value(bool, array_info)");
    }
}

// ---------------------------------------------------------------------------
// shrink_array::ChunkFastCompPolicy<T, Derived, Enabled>
// ---------------------------------------------------------------------------

template <typename T, typename Derived, bool Enabled>
bool zero_or_value::ChunkFastCompPolicy<T, Derived, Enabled>::has_queued(T second_value, int max_thr)
{
    if (no_candidates())
    {
        return false;
    }

    for (int thr_rank = 0; thr_rank < max_thr; ++thr_rank)
    {
        auto&      map_thr = Manager::zero_or_value_candidates_map_thr[thr_rank];
        const auto it      = map_thr.find(second_value);
        if (it != map_thr.end() && !it->second->empty())
        {
            return true;
        }
    }
    return false;
}

template <typename T, typename Derived, bool Enabled>
[[nodiscard]] bool
zero_or_value::ChunkFastCompPolicy<T, Derived, Enabled>::compare_and_swap(this Derived& self,
                                                                          T*            values_loc,
                                                                          int           idx,
                                                                          bool          expected_second_value) noexcept
{
    if constexpr (is_not_bool)
    {
        // One CAS against the precise expected pointer (see zero_any::Chunk::compare_and_swap): two
        // CAS combined with && could succeed on the wrong side after a first_value <-> second_value
        // race, installing a buffer whose init does not match the observed state.
        T* expected = expected_second_value ? BaseSpec::second_value_ptr(idx) : Base::first_value_ptr();
        return self.Base::compare_and_swap_gen(values_loc, expected);
    }
    else
    {
        static_assert(false,
                      "compare_and_swap(values_loc, idx, ...) is for non-bool elements: use compare_and_swap_bool()");
    }
}

template <typename T, typename Derived, bool Enabled>
[[nodiscard]] bool
zero_or_value::ChunkFastCompPolicy<T, Derived, Enabled>::compare_and_swap_bool(this Derived& self,
                                                                               T*            values_loc,
                                                                               bool expected_second_value) noexcept
{
    if constexpr (is_bool)
    {
        T* expected = expected_second_value ? BaseSpec::second_value_ptr() : Base::first_value_ptr();
        return self.Base::compare_and_swap_gen(values_loc, expected);
    }
    else
    {
        static_assert(false,
                      "compare_and_swap_bool() is for bool elements: use compare_and_swap(values_loc, idx, ...)");
    }
}

template <typename T, typename Derived, bool Enabled>
CandidateList<zero_or_value::Chunk, T>&
zero_or_value::ChunkFastCompPolicy<T, Derived, Enabled>::candidates_of_thread(int thr, InT<T> second_value) noexcept
{
    // find(), not operator[]: see zero_or_value::Proxy::candidates_of_thread.
    auto&      lists_by_value = Manager::zero_or_value_candidates_map_thr[thr];
    const auto it             = lists_by_value.find(second_value);
    if (it == lists_by_value.end() || it->second == nullptr) [[unlikely]]
    {
        zero_or_value_missing_candidate_list(thr);
    }
    return *it->second;
}

// ---------------------------------------------------------------------------
// shrink_array::Chunk<T>
// ---------------------------------------------------------------------------

template <typename T>
zero_or_value::Chunk<T>::Chunk(const Chunk& other) : CounterPolicy(other)
{
    if constexpr (build_config::threaded)
    {
        values_.store(other.values_.load());
    }
    else
    {
        values_ = other.values_;
    }
}

template <typename T>
zero_or_value::Chunk<T>& zero_or_value::Chunk<T>::operator=(const Chunk<T>& other)
{
    if (this == &other)
    {
        return *this;
    }

    // See the copy ctor: the counters must follow `values`.
    CounterPolicy::operator=(other);

    if constexpr (build_config::threaded)
    {
        T* loaded_value = other.values.load();
        values_.store(loaded_value);
    }
    else
    {
        values_ = other.values;
    }

    return *this;
}

template <typename T>
void zero_or_value::Chunk<T>::init_after_constructor()
{
    if constexpr (is_not_bool)
    {
        init_to_first_value();
    }
    else
    {
        init_values_const_val_bool(0);
    }
}

template <typename T>
void zero_or_value::Chunk<T>::init_values_const_val(int const_index, int idx)
{
    if (const_index == 0)
    {
        values_ = Base::first_value_ptr();
    }
    else
    {
        if constexpr (is_not_bool)
        {
            values_ = BaseSpec::second_value_ptr(idx);
        }
        else
        {
            static_assert(
                false,
                "Logic Error: init_values_const_val is for non-bool elements: use init_values_const_val_bool instead.");
        }
    }
}

template <typename T>
void zero_or_value::Chunk<T>::init_values_const_val_bool(int const_index)
{
    if constexpr (is_bool)
    {
        values_ = const_index == 0 ? Base::first_value_ptr() : BaseSpec::second_value_buffer_.data();
    }
    else
    {
        static_assert(false, "Logic Error: init_values_const_val_bool called on a non-boolean chunk type.");
    }
}

template <typename T>
[[nodiscard]] bool zero_or_value::Chunk<T>::written() const
    requires is_bool_like_v<T>
{
    return values_ != Base::first_value_ptr() && is_not_second_value();
}

template <typename T>
[[nodiscard]] bool zero_or_value::Chunk<T>::owns_live_buffer() const noexcept
{
    const T* vals = Base::values_load();
    if (vals == Base::first_value_ptr())
    {
        return false;
    }
    if constexpr (is_bool)
    {
        return !BaseSpec::is_second_value(vals);
    }
    else
    {
        return !BaseSpec::is_any_second_value(vals);
    }
}

template <typename T>
[[nodiscard]] bool zero_or_value::Chunk<T>::written(int idx) const
{
    if constexpr (is_not_bool)
    {
        return values_ != Base::first_value_ptr() && is_not_second_value(idx);
    }
    else
    {
        static_assert(false, "Logic Error: written(int idx) called on a boolean chunk. Boolean "
                             "chunks use a single shared second value, use written() instead.");
    }
}

template <typename T>
[[nodiscard]] bool zero_or_value::Chunk<T>::is_second_value(int idx) const noexcept
{
    if constexpr (is_not_bool)
    {
        return values_ == BaseSpec::second_value_ptr(idx);
    }
    else
    {
        static_assert(false, "Logic Error: is_second_value(int idx) called on a boolean chunk. Boolean "
                             "chunks do not use indexed constants.");
    }
}

template <typename T>
[[nodiscard]] bool zero_or_value::Chunk<T>::is_second_value() const noexcept
{
    if constexpr (is_bool)
    {
        return values_ == BaseSpec::second_value_ptr();
    }
    else
    {
        static_assert(
            false,
            "Logic Error: is_second_value() called on a non-boolean chunk. Non-bool types require an index (int idx).");
    }
}

template <typename T>
[[nodiscard]] bool zero_or_value::Chunk<T>::is_not_second_value(int idx) const noexcept
{
    if constexpr (is_not_bool)
    {
        return !is_second_value(idx);
    }
    else
    {
        static_assert(false, "Logic Error: is_not_second_value(int idx) called on a boolean chunk.");
    }
}

template <typename T>
[[nodiscard]] bool zero_or_value::Chunk<T>::is_not_second_value() const noexcept
{
    if constexpr (is_bool)
    {
        return !is_second_value();
    }
    else
    {
        static_assert(false, "Logic Error: is_not_second_value() called on a non-boolean chunk.");
    }
}

template <typename T>
bool zero_or_value::Chunk<T>::alloc_from_second_value(const ArrayInfo& array_info) const noexcept
{
    if constexpr (is_not_bool)
    {
        return !Base::is_first_value() && is_second_value(array_info.idx);
    }
    else
    {
        return !Base::is_first_value() && is_second_value();
    }
}

template <typename T>
void zero_or_value::Chunk<T>::alloc_set_counter([[maybe_unused]] bool from_second) noexcept
{
    if constexpr (chunk_config::use_compression_with_counter)
    {
        const auto full = static_cast<CounterValueT>(thread_chunk_size());
        if (from_second)
        {
            this->set_n_val_eq_second_value(full);
        }
        else
        {
            this->set_n_val_eq_first_value(full);
        }
    }
}

template <typename T>
void zero_or_value::Chunk<T>::alloc_chunk(const ArrayInfo& array_info)
{
    if (is_const_val(array_info.idx))
    {
        // From the constant the chunk holds: filled with the first one, a
        // chunk compressed to the second value read T{} from then on.
        const bool from_second = alloc_from_second_value(array_info);
        // bool has its own materialiser.
        if constexpr (is_not_bool)
        {
            create_value(from_second, array_info);
        }
        else
        {
            create_value_bool(from_second);
        }
        alloc_set_counter(from_second);
    }
}

template <typename T>
void zero_or_value::Chunk<T>::alloc_chunk(const ArrayInfo& array_info, [[maybe_unused]] int thr_rank)
{
    if (is_const_val(array_info.idx))
    {
        if constexpr (chunk_config::stats)
        {
            inc_chunk_write<ChunkType>();
        }

        const bool from_second = alloc_from_second_value(array_info);
        if constexpr (build_config::threaded)
        {
            if constexpr (is_not_bool)
            {
                create_value(from_second, array_info, thr_rank);
            }
            else
            {
                create_value_bool(from_second, thr_rank);
            }
        }
        else
        {
            if constexpr (is_not_bool)
            {
                create_value(from_second, array_info);
            }
            else
            {
                create_value_bool(from_second);
            }
            alloc_set_counter(from_second);
        }
    }
}

template <typename T>
void zero_or_value::Chunk<T>::values_store(T* ptr) noexcept
{
    if constexpr (build_config::threaded)
    {
        values_.store(ptr, std::memory_order_release);
    }
    else
    {
        values_ = ptr;
    }
}

template <typename T>
[[nodiscard]] T* zero_or_value::Chunk<T>::try_mark_const_and_get_old(this auto& self, int const_index, int idx)
{
    T* old = self.Base::values_load();
    if constexpr (is_bool)
    {
        if (old == Base::first_value_ptr() || BaseSpec::is_second_value(old)) [[unlikely]]
        {
            return nullptr;
        }
    }
    else
    {
        if (old == Base::first_value_ptr() || self.BaseSpec::is_second_value(old, idx)) [[unlikely]]
        {
            return nullptr;
        }
        // No second value declared: no block to compress to. The scans never file such a
        // chunk under the second constant, so this is a guard, not a path.
        if (const_index != 0 && idx < 0) [[unlikely]]
        {
            if constexpr (build_config::any_check)
            {
                CHUNK_INVARIANT_FAILED("[chunk] compress to second_value on an array with no second value");
            }
            return nullptr;
        }
    }

    T* expected = old;
    T* new_ptr  = nullptr;

    if (const_index == 0)
    {
        new_ptr = Base::first_value_ptr();
    }
    else
    {
        if constexpr (is_bool)
        {
            new_ptr = BaseSpec::second_value_ptr();
        }
        else
        {
            new_ptr = BaseSpec::second_value_ptr(idx);
        }
    }

    // Only the thread whose CAS replaces 'old' by 'new_ptr' may free/recycle the old block.
    CHUNK_CHECK_VALUES_PTR_C(new_ptr, "try_mark.new_ptr", &self);
    if (!self.values_.compare_exchange_strong(expected, new_ptr, std::memory_order_acq_rel, std::memory_order_acquire))
        [[unlikely]]
    {
        // Another thread changed 'values': free nothing.
        return nullptr;
    }

    // Unique transition to a constant: the caller frees/recycles the old pointer.
    return old;
}

template <typename T>
[[nodiscard]] bool zero_or_value::Chunk<T>::is_const_val() const noexcept
{
    if constexpr (is_bool)
    {
        return Base::is_first_value() || is_second_value();
    }
    else
    {
        static_assert(false, "Logic Error: is_const_val() without index called on a non-boolean chunk.");
    }
}

template <typename T>
[[nodiscard]] bool zero_or_value::Chunk<T>::is_const_val(int idx) const noexcept
{
    if constexpr (is_not_bool)
    {
        return Base::is_first_value() || BaseSpec::is_second_value(values_, idx);
    }
    else
    {
        return Base::is_first_value() || BaseSpec::is_second_value(values_);
    }
}

} // end namespace shrink_array
