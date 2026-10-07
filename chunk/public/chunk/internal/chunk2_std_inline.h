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
#include <cstdlib>  // std::abort
#include <optional> // std::nullopt (the clears)

#include "chunk/config.h"                       // build_config::chunk_dynamic, chunk_config::OMPTcc
#include "chunk/PMP_CHUNK.h"                    // CHUNK_PMP_*
#include "chunk/internal/chunk_array_new.h"     // chunk_array_new::init_chunks (operator new[])
#include "chunk/internal/chunk_size_policy.h"   // ChunkElems, set_thread_chunk_size
#include "chunk/internal/in_t.h"                // in_t<T>
#include "chunk/chunk2_std.h"                   // Chunk2, the class these bodies belong to
#include "chunk/internal/proxy2.h"              // Proxy2<T>
#include "chunk/internal/block_pass_precheck.h" // block_pass_yields_to_flush
#include "chunk/internal/invariant_check.h"     // CHUNK_INVARIANT_FAILED
#include "chunk/internal/test_hooks.h"          // TEST KNOB: test_hooks::maybe_pass_delay

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

// ── Array-new (placement-new arrays; see chunk_array_new.h) ────────────────
template <typename T, typename Derived, bool Enabled>
void* Chunk2OmpPolicy<T, Derived, Enabled>::operator new[](std::size_t sz)
{
    auto* v = static_cast<Chunk2<T>*>(::operator new(sz));
    init_chunks_cst_val(v, sz / sizeof(Chunk2<T>));
    return static_cast<void*>(v);
}

template <typename T, typename Derived, bool Enabled>
void Chunk2OmpPolicy<T, Derived, Enabled>::operator delete[](void* ptr)
{
    ::operator delete(ptr);
}

template <typename T, typename Derived, bool Enabled>
void Chunk2OmpPolicy<T, Derived, Enabled>::init_chunks_cst_val(Chunk2<T>* v, std::size_t count) noexcept
{
    chunk_array_new::init_chunks(v, count, [](Chunk2<T>& chunk) { chunk.init_after_constructor(); });
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
void Chunk2FastCompPolicy<T, Derived, Enabled>::fill_list_to_compress(this Derived&           self,
                                                                      vec_tuple_chunk2p_int2& ll_compt1,
                                                                      vec_tuple_chunk2p_int2& ll_compt2,
                                                                      const inf_array_t&      inf_array,
                                                                      int                     num_chunk,
                                                                      int                     num_tab,
                                                                      unsigned int            n_elems)
{
    // in_t: a reference into the (const, caller-owned) tuple when T is large.
    const in_t<T> second_v = std::get<0>(inf_array);
    self.fill_list_to_compress_impl(
        ll_compt1, ll_compt2, num_chunk, num_tab, [&second_v]() -> in_t<T> { return second_v; },
        self.is_not_cst_val2(std::get<1>(inf_array)), n_elems);
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
void Chunk2FastCompPolicy<T, Derived, Enabled>::fill_list_to_compress_bool(this Derived&           self,
                                                                           vec_tuple_chunk2p_int2& ll_compt1,
                                                                           vec_tuple_chunk2p_int2& ll_compt2,
                                                                           int                     num_chunk,
                                                                           int                     num_tab,
                                                                           unsigned int            n_elems)
{
    self.fill_list_to_compress_impl(
        ll_compt1, ll_compt2, num_chunk, num_tab, [] { return true; }, self.is_not_cst_val2(), n_elems);
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
void Chunk2FastCompPolicy<T, Derived, Enabled>::fill_list_to_compress_impl(this Derived&           self,
                                                                           vec_tuple_chunk2p_int2& ll_compt1,
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

    if (self.Base::is_not_cst_val1() && check_cst_val2)
    {
        // CHUNK_STATS: one compression candidate scanned (see chunk2a).
        increm_chunk_compress_attempt<Derived>();

        // n_elems: see Chunk2<T>::compress() -- the chunk's own geometry,
        // captured at enqueue, never the ambient alloc_chunk_count().
        const auto scan = self.Base::scan_constants(first_value, get_second_value(), n_elems);

        if (scan.first)
        {
            ll_compt1.emplace_back(std::make_tuple(&self, num_chunk, num_tab, n_elems));
        }
        if (scan.second)
        {
            ll_compt2.emplace_back(std::make_tuple(&self, num_chunk, num_tab, n_elems));
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
void Chunk2FastCompPolicy<T, Derived, Enabled>::compress_impl(this Derived&  self,
                                                              bool            apply,
                                                              unsigned int    n_elems,
                                                              SecondValueFunc get_second_value,
                                                              ClearFunc&&     clear_one)
{
    if (self.Base::is_not_cst_val1())
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
        // at a different geometry. Same rule as Chunk v1 and Chunk2A.
        //
        // Always scanned, even when the array's second value equals
        // first_value: skipping the scan would leave BOTH verdicts at their
        // initial `true` and clear the chunk twice without looking at it.
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
void Chunk2FastCompPolicy<T, Derived, Enabled>::compress(this Derived&        self,
                                                         [[maybe_unused]] int thr_rank,
                                                         const inf_array_t&   inf_array,
                                                         bool                 apply,
                                                         unsigned int         n_elems)
{
    // Only a MATERIALISED chunk can be compressed. A queued entry can outlive
    // the chunk's materialisation: make_block_compression (or a previous
    // drain) compresses it, and the OMPT end-of-region drain then replays
    // the stale entry. compress_impl only refuses cst_val1; a chunk already at
    // cst_val2 would pass, scan the SHARED second-value block, find it
    // constant and clear it a second time -- freeing that shared block.
    if (!self.written(std::get<1>(inf_array)))
    {
        return;
    }

    self.compress_impl(
        apply, n_elems, [&inf_array] { return std::get<0>(inf_array); },
        [&self, thr_rank, &inf_array](int num_compress_value)
        { self.clear_int_st(num_compress_value, std::get<1>(inf_array), thr_rank); });
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
void Chunk2FastCompPolicy<T, Derived, Enabled>::compress_bool(this Derived&        self,
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
            [&self, thr_rank](int num_compress_value) { self.clear_int_st_bool(num_compress_value, thr_rank); });
    }
}

/**
 * @brief Compresses the candidates of one second value, across every rank.
 *
 * Entry point used between parallel phases. The OMPT callbacks are disarmed for
 * the duration, so nothing re-enters the compression machinery underneath.
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
        // independent. A serial loop over the ranks made a re-chunk wait on
        // one thread while the rest of the team idled.
        int team = 0;
#pragma omp parallel
        {
            const int thr = CHUNK_PMP_THREAD_RANK;
            if (thr == 0) // one writer; `master` is deprecated since OpenMP 5.1
            {
                team = CHUNK_PMP_NUM_THREADS;
            }
            compress_chunk_to_compress_in_par(thr, inf_array, apply);
        }
        // Lists are sized by the thread capacity; a team smaller than the
        // ranks seen leaves the upper ranks unswept. Serial mop-up up to
        // CHUNK_PMP_RANK_BOUND, empty lists are cheap.
        for (int thr_rank = team; thr_rank < CHUNK_PMP_RANK_BOUND; ++thr_rank)
        {
            compress_chunk_to_compress_in_par(thr_rank, inf_array, apply);
        }
    }

    chunk_config::OMPTcc<chunk_config::has_ompt>::update(true);
}

/**
 * @brief Bool twin of compress_chunk_to_compress().
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
                {
                    team = CHUNK_PMP_NUM_THREADS;
                }
                compress_chunk_to_compress_bool_in_par(thr, apply);
            }
            for (int thr_rank = team; thr_rank < CHUNK_PMP_RANK_BOUND; ++thr_rank)
            {
                compress_chunk_to_compress_bool_in_par(thr_rank, apply);
            }
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
 * Today the drain always runs at a point with no concurrent writers --
 * on_parallel_end() puts an `omp barrier` in front of it, and the ArrayChunk2
 * call sites are gated on !CHUNK_PMP_IN_PARALLEL_REGION -- so either order
 * would give the same result. This is the one that stays correct if that
 * precondition ever slips, as on Chunk2A.
 *
 * @tparam CompressFn callable `(chunk_type*, unsigned int n_elems)`.
 * @param thr_rank     Rank whose list is drained.
 * @param key          Second value indexing the per-rank candidate map.
 * @param compress_one Per-chunk compression.
 */
template <typename T, typename Derived, bool Enabled>
template <typename CompressFn>
void Chunk2FastCompPolicy<T, Derived, Enabled>::compress_chunk_to_compress_in_par_impl(int          thr_rank,
                                                                                       in_t<T>      key,
                                                                                       CompressFn&& compress_one)
{
    // The store is a per-rank VECTOR of maps, sized by resize_all_list_block_free_thr();
    // until that has run, [thr_rank] is out of bounds -- silent UB in a normal
    // build, an _GLIBCXX_ASSERTIONS abort otherwise. Same guard as the reuse-list
    // lookups. Probing rank 0 is enough: a second value is registered for
    // every rank in one loop (ArrayChunk2::add_second_value), so an empty map at
    // rank 0 means no key exists at any rank.
    if (is_vec_tuple_to_compress_potentially_empty_thr())
    {
        return;
    }

    auto* const vec_tuple_p = manager_chunk2::compress_candidates(thr_rank, key);
    assert(vec_tuple_p != nullptr && "key registered for every rank by add_second_value()");
    if (vec_tuple_p == nullptr) [[unlikely]]
    {
        return;
    }
    auto& vec_tuple = *vec_tuple_p;

    // Geometry: this list mixes chunks of every array sharing `key` -- e.g. an
    // array and the tmp rechunk_to() builds from it, at two chunk sizes. The
    // scan below reads n_elems explicitly, but the clear it triggers reaches
    // free_block_to_pool(), which routes by the ambient thread_chunk_size()
    // (reuse list AND allocator). Pin each chunk to its own size, restore
    // afterwards so the OMPT drain leaves the thread as it found it: otherwise
    // a rechunked array's wide blocks are parked in a narrower list, and the
    // next array of that size pops them. Same pin as
    // make_block_compression_impl() and Chunk2A<T>::make_block_compression().
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
            Proxy2<T>::reset_tab_is_inside_list_to_compress(num_tab, num_chunk);
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
void Chunk2FastCompPolicy<T, Derived, Enabled>::compress_chunk_to_compress_in_par(int                thr_rank,
                                                                                  const inf_array_t& inf_array,
                                                                                  bool               apply)
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
 * @tparam ScanFn callable `(chunk_type*, int num_chunk, int num_tab, ChunkElems)`.
 * @param thr_rank Rank whose candidate list is scanned.
 * @param key      Second value indexing the per-rank candidate map.
 * @param scan_one Per-chunk scan.
 */
template <typename T, typename Derived, bool Enabled>
template <typename ScanFn>
void Chunk2FastCompPolicy<T, Derived, Enabled>::fill_list_chunk_to_compress_in_par_impl(int      thr_rank,
                                                                                        in_t<T>  key,
                                                                                        ScanFn&& scan_one)
{
    const auto* const vec_tuple_p = manager_chunk2::compress_candidates(thr_rank, key);
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
void Chunk2FastCompPolicy<T, Derived, Enabled>::fill_list_chunk_to_compress_bool_in_par(
    int                     thr_rank,
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
            {
                chunk_t->steal_unmark();
            }
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
// single pin on the triggering proxy's size (kept, for the handshake) would
// park the other array's buffers under the wrong size.
template <typename T, typename Derived, bool Enabled>
template <typename FillFunc, typename ClearFunc>
void Chunk2FastCompPolicy<T, Derived, Enabled>::make_block_compression_impl(int         thr_rank,
                                                                            T           second_value,
                                                                            Proxy2<T>*  proxy2,
                                                                            FillFunc&&  fill_func,
                                                                            ClearFunc&& clear_func)
{
    // find, NOT operator[]: the table is a map<T, shared_ptr<...>> per rank,
    // and operator[] on a missing key inserts a NULL shared_ptr that the next
    // dereference blows up on.
    //
    // A missing key is not an anomaly: it means nobody declared this second
    // value on this rank, so there is no candidate to compress -- the
    // `sz == 0` case handled just below, with the same exit. An ArrayChunk2
    // built without a prior add_second_value reaches it at construction,
    // through resize_common -> make_block_compression.
    //
    // is_vec_tuple_to_compress_potentially_empty_thr() is not enough here: it
    // only tests "no key anywhere", not "this key on this rank".
    auto&      map_thr = manager_chunk2::map_chunk2_to_compress_potentially_thr[thr_rank];
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
    // still applied first for the handshake, as before.
    [[maybe_unused]] const unsigned int saved_sz = thread_chunk_size();

    if (proxy2)
    {
        if constexpr (build_config::chunk_dynamic)
        {
            shrink_array::set_thread_chunk_size(proxy2->chunk_size());
        }

        // A flush in progress makes start() decline: ask first, before
        // marking -- see block_pass_precheck.h.
        if (block_pass_yields_to_flush<Proxy2<T>, Chunk2<T>>(CHUNK_PMP_MAX_THREADS))
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
        const bool ret = proxy2->compression_mgr.template start<Proxy2<T>>();
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

    vec_tuple_chunk2p_int2 ll_compt1;
    vec_tuple_chunk2p_int2 ll_compt2;

    ll_compt1.reserve((sz / 50) + 1);
    ll_compt2.reserve((sz / 50) + 1);

    // Call the fill function (different for bool vs non-bool)
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
        // (idempotent for the ones released above). The other way round,
        // another thread could queue the chunk once the flag was down, start
        // its pass and mark it -- and this unmark erased that claim: writers
        // then wrote into a chunk that pass was scanning and stealing
        // (read_reclaim_validation, mode `remark`).
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
        Proxy2<T>::reset_tab_is_inside_list_to_compress(num_tab, num_chunk);
    }

    vec_tuple_to_compress.clear();
    vec_tuple_to_compress.reserve((sz / 10) + 1);

    if (proxy2)
    {
        // Re-queue the writer's chunk only into the list of ITS array's second
        // value. Proxy2::perform_compression runs the pass of every key of the
        // rank with the same writer: queued under another array's key, the
        // chunk would be compressed to the OTHER array's constant whenever it
        // was constant at that value -- and its own array, which only
        // recognises its own constant, would then write into that shared
        // array, changing every chunk compressed to it
        // (chunk2_foreign_key_validation).
        if (proxy2->is_not_cst_val() && proxy2->second_value == second_value)
        {
            proxy2->add_to_compression_list(vec_tuple_to_compress);
        }

        proxy2->compression_mgr.template finish<Proxy2<T>>();
    }

    if constexpr (build_config::chunk_dynamic)
    {
        set_thread_chunk_size(saved_sz);
    }
}

// make_block_compression (declared in chunk2_std.h). Binds the non-bool scan and
// clear callables and hands them to the shared sweep. Instantiating it for bool
// is a programming error, and a compile error.
template <typename T, typename Derived, bool Enabled>
void Chunk2FastCompPolicy<T, Derived, Enabled>::make_block_compression(int                thr_rank,
                                                                       const inf_array_t& inf_array,
                                                                       Proxy2<T>*         proxy2)
{
    if constexpr (is_not_bool)
    {
        const in_t<T> second_value = std::get<0>(inf_array); // the caller's tuple, alive for the call

        const auto lbd_clear_func = [thr_rank, &inf_array](auto* chunk_t, int clear_type)
        { chunk_t->clear_int_st(clear_type, std::get<1>(inf_array), thr_rank); };

        make_block_compression_impl(
            thr_rank, second_value, proxy2, [thr_rank, &inf_array](auto& ll_compt1, auto& ll_compt2)
            { Derived::fill_list_chunk_to_compress_in_par(thr_rank, ll_compt1, ll_compt2, inf_array); },
            lbd_clear_func);
    }
    else
    {
        static_assert(false, "make_block_compression(thr, inf_array, proxy) is for non-bool elements: "
                             "use make_block_compression_bool()");
    }
}

// make_block_compression_bool (declared in chunk2_std.h). Same binding for bool:
// second value is `true`, and the clear goes through clear_int_st_bool, which
// needs no index. Mirror image of the guard above -- non-bool fails to compile here.
template <typename T, typename Derived, bool Enabled>
void Chunk2FastCompPolicy<T, Derived, Enabled>::make_block_compression_bool(int thr_rank, Proxy2<T>* proxy2)
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
            false,
            "make_block_compression_bool() is for bool elements: use make_block_compression(thr, inf_array, proxy)");
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
            const auto    it = manager_chunk2::map_chunk2_to_compress_potentially_thr[thr_rank].find(second_val);
            if (it == manager_chunk2::map_chunk2_to_compress_potentially_thr[thr_rank].end())
            {
                continue;
            }
            const auto* vec_tuple_p = it->second.get();

            if (vec_tuple_p)
            {
                // The map keeps a slot per second value ever declared; the
                // arrays that declared this one may all be gone (cleared:
                // clear_core drops their candidates by num_tab, not the slot).
                // No array, no info tuple -- and nothing of that value left
                // to fold: skip it (unguarded, force_compression() /
                // shrink_to_fit() after such a clear dereferences null).
                const auto* inf = Proxy2<T>::share_info_array->find_inf_array(second_val);
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

        const auto it = manager_chunk2::map_chunk2_to_compress_potentially_thr[thr_rank].find(true);
        if (it == manager_chunk2::map_chunk2_to_compress_potentially_thr[thr_rank].end())
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
 *          the call site does NOT work, generic or not: once Chunk2<T> is
 *          instantiated FastCompPolicy is a concrete type, so the qualified
 *          name is looked up eagerly and the serial build fails to compile.
 *          The serial configurations are what catch that mistake.
 */
template <typename T, typename Derived, bool Enabled>
template <bool IsBoolCas, typename InitFn>
void Chunk2OmpPolicy<T, Derived, Enabled>::create_value_impl(this Derived&         self,
                                                             int                   thr_rank,
                                                             InitFn&&              init_func,
                                                             [[maybe_unused]] int  ind,
                                                             [[maybe_unused]] bool need_set_second_value) noexcept
{
    T* values_loc = self.Base::allocate_chunk_internal(thr_rank, std::forward<InitFn>(init_func));

    if constexpr (chunk_config::use_fast_block_compression)
    {
        self.Base::apply_cas_or_recup(values_loc, thr_rank,
                                      [&self, ind, need_set_second_value](T* cst_ptr)
                                      {
                                          if constexpr (IsBoolCas)
                                          {
                                              (void)ind; // only the non-bool CAS takes the index
                                              return self.compare_and_swap_bool(cst_ptr, need_set_second_value);
                                          }
                                          else
                                          {
                                              // Propagate the observed state to the CAS so it refuses a
                                              // wrong-side transition after a cst_val1 <-> cst_val2 race.
                                              return self.compare_and_swap(cst_ptr, ind, need_set_second_value);
                                          }
                                      });
    }
    else
    {
        // apply_cas_or_recup takes its non-CAS branch in this mode anyway.
        self.Base::apply_cas_or_recup(values_loc, thr_rank, [](T*) { return false; });
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
void Chunk2OmpPolicy<T, Derived, Enabled>::create_value(this Derived&      self,
                                                        bool               need_set_second_value,
                                                        const inf_array_t& inf_array,
                                                        int                thr_rank) noexcept
{
    if constexpr (is_not_bool)
    {
        self.template create_value_impl</*IsBoolCas=*/false>(
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
            std::get<1>(inf_array), need_set_second_value);
    }
    else
    {
        static_assert(false, "create_value(bool, inf_array, thr) is for non-bool elements: use create_value_bool()");
    }
}

/**
 * @brief Materialises a `bool` chunk.
 *
 * Same allocate-fill-CAS shape as create_value(), with no index to carry.
 */
template <typename T, typename Derived, bool Enabled>
void Chunk2OmpPolicy<T, Derived, Enabled>::create_value_bool(this Derived& self,
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
void Chunk2OmpPolicy<T, Derived, false>::create_value_serial_impl(this Derived& self,
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
            self.reset_nb_val_eq_second_value();
        }
        else
        {
            init_2nd(ptr);
            self.reset_nb_val_eq_first_value();
        }
    };

    self.values = self.Base::allocate_chunk_internal(0, init_func);
}

template <typename T, typename Derived>
void Chunk2OmpPolicy<T, Derived, false>::create_value(this Derived&      self,
                                                      bool               need_set_second_value,
                                                      const inf_array_t& inf_array) noexcept
{
    if constexpr (is_not_bool)
    {
        self.create_value_serial_impl(need_set_second_value,
                                      [&inf_array](T* ptr) { BaseSpec::init_block_2nd_value(ptr, inf_array); });
    }
    else
    {
        static_assert(false, "create_value(bool, inf_array) is for non-bool elements: use create_value_bool()");
    }
}

/**
 * @brief Materialises a `bool` chunk (serial build).
 */
template <typename T, typename Derived>
void Chunk2OmpPolicy<T, Derived, false>::create_value_bool(this Derived& self, bool need_set_second_value) noexcept
{
    if constexpr (is_bool)
    {
        self.create_value_serial_impl(need_set_second_value, [](T* ptr) { BaseSpec::init_block_2nd_value(ptr); });
    }
    else
    {
        static_assert(false, "create_value_bool() is for bool elements: use create_value(bool, inf_array)");
    }
}

/**
 * @brief Compresses this chunk back to first_value, if it is materialised.
 */
template <typename T, typename Derived, bool Enabled>
void Chunk2OmpPolicy<T, Derived, Enabled>::clear(this Derived& self, int ind, int thr_rank)
{
    self.Base::clear_impl(ind, thr_rank);
}

/**
 * @brief Compresses this chunk to first_value without testing `written()` (serial build).
 *
 * For callers that have already established the chunk is materialised; the test
 * survives as a debug-only assertion.
 */
template <typename T, typename Derived>
void Chunk2OmpPolicy<T, Derived, false>::clear_st(this Derived& self, int ind)
{
    self.Base::debug_check_written(ind);

    self.Base::free_values(self.values);
    self.init_values_cst_val1();

    if constexpr (chunk_config::use_compression_with_counter)
    {
        self.reset_nb_val_eq_value();
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
void Chunk2OmpPolicy<T, Derived, Enabled>::block_clear(this Derived& self, int num_compress_value, int ind, [[maybe_unused]] int thr_rank)
{
    // CHUNK_STATS: one materialised block released by a successful compression.
    // Counted here because block_clear is the single point every variant and
    // both OMP/non-OMP modes funnel through (the old counter only fired on the
    // plain-chunk sweep, so chunk2/chunk2a always reported 0).
    increm_chunk_compress<Derived>();
    add_chunk_bytes_freed<Derived>(static_cast<long>(sizeof(T)) * static_cast<long>(alloc_chunk_count()));
    self.Base::block_clear_common(num_compress_value, ind, thr_rank);
}

/// Serial build: frees the buffer outright instead of parking it in a rank
/// pool, and resets the per-chunk element counters. The rank argument is
/// unused (documented with the declaration).
template <typename T, typename Derived>
void Chunk2OmpPolicy<T, Derived, false>::block_clear(this Derived& self, int num_compress_value, int ind, int)
{
    // CHUNK_STATS: one materialised block released by a successful compression.
    // Counted here because block_clear is the single point every variant and
    // both OMP/non-OMP modes funnel through (the old counter only fired on the
    // plain-chunk sweep, so chunk2/chunk2a always reported 0).
    increm_chunk_compress<Derived>();
    add_chunk_bytes_freed<Derived>(static_cast<long>(sizeof(T)) * static_cast<long>(alloc_chunk_count()));
    self.Base::free_values(self.values);

    self.init_values_cst_val(num_compress_value, ind);
    if constexpr (chunk_config::use_compression_with_counter)
    {
        self.reset_nb_val_eq_value();
    }
}

/**
 * @brief Compresses this chunk to the named constant, after an invariant check.
 * @pre The chunk is materialised — compressing an already-constant chunk would
 *      release a shared constant block. Checked by
 *      check_not_constant_before_clear().
 */
template <typename T, typename Derived, bool Enabled>
void Chunk2OmpPolicy<T, Derived, Enabled>::clear_int_st(this Derived& self, int num_compress_value, int ind, int thr_rank)
{
    self.Base::check_not_constant_before_clear(num_compress_value, ind);
    self.block_clear(num_compress_value, ind, thr_rank);
}

/**
 * @brief Bool twin of clear_int_st(): no second-value index to pass.
 */
template <typename T, typename Derived, bool Enabled>
void Chunk2OmpPolicy<T, Derived, Enabled>::clear_int_st_bool(this Derived& self, int num_compress_value, int thr_rank)
{
    if constexpr (is_bool)
    {
        self.Base::check_not_constant_before_clear(num_compress_value, std::nullopt);
        self.block_clear(num_compress_value, 0, thr_rank);
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
void Chunk2OmpPolicy<T, Derived, false>::clear_int_st(this Derived& self, int num_compress_value, int ind)
{
    // --- check only on debug ---

    // Bool chunks share a single second value with no index: use the no-arg
    // written check and the bool-specific reinit. Non-bool uses the indexed
    // path. (Matches the counter-policy split in the reference implementation.)
    if constexpr (is_bool)
    {
        self.Base::debug_check_written();
    }
    else
    {
        self.Base::debug_check_written(ind);
    }

    // CHUNK_STATS: this is the non-OMP twin of block_clear -- reached from
    // compress_int's sweep via clear_st(num_compress_value, ind), which
    // bypasses block_clear entirely, so it counts here. No if constexpr
    // needed: both functions resolve to a no-op overload when CHUNK_STATS is
    // off (chunk_stats.h), as in the OMP twin of block_clear above.
    increm_chunk_compress<Derived>();
    add_chunk_bytes_freed<Derived>(static_cast<long>(sizeof(T)) * static_cast<long>(alloc_chunk_count()));

    self.Base::free_values(self.values);

    if (num_compress_value == 0)
    {
        self.reset_nb_val_eq_first_value();
    }
    else
    {
        self.reset_nb_val_eq_second_value();
    }

    if constexpr (is_bool)
    {
        self.init_values_cst_val_bool(num_compress_value);
    }
    else
    {
        self.init_values_cst_val(num_compress_value, ind);
    }
}

// clear_st(int num_compress_value, int ind) -- the entry point that supplies
// the calling thread's rank (or has none to supply) -- is defined in-class on
// each Chunk2OmpPolicy specialisation.

} // end namespace shrink_array
