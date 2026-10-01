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

#include <algorithm>       // std::fill_n (init_block)
#include <atomic>          // std::atomic_thread_fence
#include <cstddef>         // std::size_t
#include <iostream>        // std::cerr (debug-build clear guard)
#include <source_location> // std::source_location
#include <tuple>           // the candidate-list entries

#include "chunk/chunk_std.h" // the declarations these bodies belong to (self-contained, like the other *_inline.h)
#include "chunk/config.h"
#include "chunk/internal/invariant_check.h"     // INVARIANT_FAILED
#include "chunk/internal/chunk_array_new.h"     // chunk_array_new::init_chunks (operator new[])
#include "chunk/internal/chunk_size_policy.h"   // alloc_chunk_count
#include "chunk/internal/chunk_stats.h"         // increm_chunk_* / add_chunk_* (CHUNK_STATS)
#include "chunk/internal/manager_chunk.h"       // manager_chunk:: candidate lists
#include "chunk/PMP_CHUNK.h"                    // CHUNK_PMP_*
#include "chunk/internal/pool_ledger.h"         // CHUNK_LEDGER_*
#include "chunk/internal/block_pass_precheck.h" // block_pass_yields_to_flush

namespace shrink_array
{
// No build guard: the ChunkOmpPolicy definitions below belong to the PRIMARY
// template, instantiated only with Enabled=true (i.e. only under OpenMP); the
// serial build selects the empty <..., false> specialisation. Chunk's own
// members here compile in both modes.

template <typename T, typename Derived, bool Enabled>
void* ChunkOmpPolicy<T, Derived, Enabled>::operator new[](std::size_t sz)
{
    auto* v = static_cast<Chunk<T>*>(::operator new(sz));
    init_chunks_cst_val(v, sz / sizeof(Chunk<T>));
    return static_cast<void*>(v);
}

template <typename T, typename Derived, bool Enabled>
void ChunkOmpPolicy<T, Derived, Enabled>::init_chunks_cst_val(Chunk<T>* v, std::size_t count) noexcept
{
    chunk_array_new::init_chunks(v, count, [](Chunk<T>& chunk) { chunk.init_values_cst_val(); });
}

template <typename T, typename Derived, bool Enabled>
void ChunkOmpPolicy<T, Derived, Enabled>::operator delete[](void* ptr)
{
    ::operator delete(ptr);
}

template <typename T, typename Derived, bool Enabled>
void ChunkFastCompPolicy<T, Derived, Enabled>::fill_list_chunk_to_compress_in_par(int                    thr,
                                                                                  vec_tuple_chunkp_int2& ll_compt)
{
    const auto& vec_tup = Derived::manager_chunk::chunk_to_compress_potentially_thr[thr];

    const int sz = vec_tup.size();

    for (int k = 0; k < sz; ++k)
    {
        const auto& tup = vec_tup[k];

        const auto& chunk_t   = std::get<0>(tup);
        const int&  num_chunk = std::get<1>(tup);
        const int   num_tab   = std::get<2>(tup);

        // Pass the chunk's OWN geometry, captured at enqueue -- not this
        // thread's ambient size, which may belong to a different array.
        const auto count = ll_compt.size();
        chunk_t->fill_list_to_compress(ll_compt, num_chunk, num_tab, std::get<3>(tup).count());
        // Not selected by the scan: the writers may have it back now.
        if (ll_compt.size() == count)
        {
            chunk_t->steal_unmark();
        }
    }
}

template <typename T, typename Derived, bool Enabled>
void ChunkFastCompPolicy<T, Derived, Enabled>::fill_list_to_compress(vec_tuple_chunkp_int2& ll_compt,
                                                                     int                    num_chunk,
                                                                     int                    num_tab,
                                                                     unsigned int           n_elems)
{
    if (derived().written())
    {
        // CHUNK_STATS: one compression candidate scanned (counted as scan
        // reads, never as application reads).
        increm_chunk_compress_attempt<Derived>();
        if (derived().is_constant_to(zero, n_elems))
        {
            ll_compt.push_back(std::make_tuple(self(), num_chunk, num_tab, ChunkElems{n_elems}));
        }
    }
}

template <typename T, typename Derived, bool Enabled>
void ChunkOmpPolicy<T, Derived, Enabled>::compress(int thr_rank, unsigned int n_elems)
{
    // Already compressed -> nothing to do. Chunk2A::compress() has always
    // guarded this way; v1 did not, and the consequence is not benign. A chunk
    // can be reached twice: the candidate queued by the write that materialised
    // it stays in the list even after something else compresses the chunk, and
    // the re-chunk rebuild now folds chunks back on the spot. On a cst1 chunk
    // `values` IS the shared zero sentinel, so is_constant_to(0) reads it,
    // answers true, and clear_st() parks the SENTINEL into a reuse list -- from
    // there it is handed out as an ordinary chunk buffer. A check build trips
    // the invariant inside clear_st first ("trying to clear a chunk that is not
    // written"); a release build reports nothing at all.
    if (!derived().written())
    {
        return;
    }

    Base::debug_check_written();

    // CHUNK_STATS: counted here so the scan yield reflects what actually runs.
    if constexpr (chunk_config::stats)
    {
        increm_chunk_compress_attempt<ChunkType>();
    }

    if (Base::is_constant_to(T{}, n_elems))
    {
        this->clear_st(thr_rank, n_elems);
    }
}

template <typename T>
void Chunk<T>::compress_sweep_rank(int thr)
{
    auto& chunk_t = manager_chunk::chunk_to_compress_potentially_thr[thr];
    // Dynamic geometry: pin the ambient chunk size to EACH candidate's own
    // (the tuple's ChunkElems, captured when it was queued), restored after
    // the loop. clear_st() -> free_block_to_pool() parks the buffer in the
    // reuse list of the AMBIENT size, and a thread's ambient size after a
    // re-chunk is that array's new one (the rebuild region sets it and
    // never restores the workers'): compressing a 64-chunk of another
    // array from a rank left at 128 parked its 64-buffer in the 128 list,
    // which handed it out as a 128-buffer later -- a re-chunk right before
    // a zeroing phase gave non-reproducible results two iterations on, at
    // 2-3 threads (shrink_bench chunk_dyn, forced advisor, Sept '26).
    // Chunk2 and Chunk2A pin the same way (compress_chunk_to_compress_in_par_impl,
    // make_block_compression_impl).
    [[maybe_unused]] const unsigned int saved_sz = thread_chunk_size();
    for (auto& tup : chunk_t)
    {
        auto* chunk_p = std::get<0>(tup);
        if constexpr (build_config::chunk_dynamic)
        {
            shrink_array::set_thread_chunk_size(std::get<3>(tup).count());
        }
        chunk_p->compress(thr, std::get<3>(tup).count());

        // Release the dedup gate for this chunk. add_to_compression_list only
        // queues a chunk when its is_inside flag CASes 0->1, so a flag left at 1
        // means the chunk can NEVER be queued again: it was compressed once and
        // then ignored for the rest of the run (measured: 98.7% of marks refused,
        // compress stuck at 0 for chunk v1 under OpenMP). Reset it TOGETHER with
        // the list clear below -- resetting one without the other lets a later
        // write re-append the same chunk and make_block_compression then
        // double-clears it (see the same note in
        // Chunk2A<T>::compress_chunk_to_compress_in_par).
        Proxy<T>::reset_tab_inside_list_to_compress(std::get<2>(tup), std::get<1>(tup));
    }

    if constexpr (build_config::chunk_dynamic)
    {
        shrink_array::set_thread_chunk_size(saved_sz);
    }
    // clear() keeps the capacity: the list is already reserved for the
    // next pass, at the size this one reached.
    chunk_t.clear();
}

template <typename T>
void Chunk<T>::compress_chunk_to_compress(bool /*apply*/)
{
    chunk_config::OMPTcc<chunk_config::has_ompt>::update(false);

#pragma omp parallel
    {
        compress_sweep_rank(CHUNK_PMP_THREAD_RANK);
    }

    chunk_config::OMPTcc<chunk_config::has_ompt>::update(true);
}


template <typename T, typename Derived, bool Enabled>
bool ChunkFastCompPolicy<T, Derived, Enabled>::have_chunk_to_compress(int max_thr)
{
    if (manager_chunk::chunk_to_compress_potentially_thr.empty())
    {
        return false;
    }

    for (int thr = 0; thr < max_thr; ++thr)
    {
        const auto& chunk_t = manager_chunk::chunk_to_compress_potentially_thr[thr];

        const bool is_empty = chunk_t.empty();
        if (!is_empty)
        {
            return true;
        }
    }

    return false;
}

/**
 * @brief Zero-fill an about-to-be-published v1 buffer, mirroring alloc_values(true).
 * @param vals buffer of shrink_array::alloc_chunk_count() elements
 * @details svcalloc_align already returns zeroed memory, so the fill is only
 *          needed for the pooled allocator, whose blocks carry whatever the
 *          previous owner left. Skipping it under the std allocator used to be
 *          alloc_values(false), which mapped to plain malloc and left garbage
 *          (subnormal doubles, say) in cells not written before the next read
 *          -- a wrong-value/checksum failure the fast allocator happened to
 *          hide because its pooled blocks are usually already zero.
 */
template <typename T, typename Derived, bool Enabled>
void ChunkOmpPolicy<T, Derived, Enabled>::zero_fill_fresh(T* vals) noexcept
{
    if constexpr (build_config::use_fast_alloc)
    {
        std::fill_n(vals, alloc_chunk_count(), Base::zero);
    }
}

template <typename T, typename Derived, bool Enabled>
void ChunkOmpPolicy<T, Derived, Enabled>::create_value(int thr_rank) noexcept
{
    // Two sources: the manager's per-thread reuse pool, then a fresh allocation.
    // Both now live in ChunkAllocPolicy::allocate_chunk_internal, shared with
    // v2/v2A -- v1 used to keep its own copy, which is how it missed the
    // bounds-checked size lookup (get_list_block_free_thr() ends in an
    // unchecked operator[] on a store that is not guaranteed to be sized here)
    // until that fix had to be applied twice.
    //
    // A third source used to come first -- a free-list of buffers released BY a
    // compression pass -- but its producer went away with clear_reversible, so
    // the list was permanently empty and the branch always fell through.
    //
    // The two initialisers are the one thing v1 does NOT share: a v1 chunk is
    // compressed only when it is all-zero, so a buffer coming back from the
    // reuse pool is already zero and re-filling it would memset the hot path
    // for nothing. v2/v2A compress to an arbitrary constant and so hand the
    // same initialiser to both branches.
    T* values_loc = Base::allocate_chunk_internal(
        thr_rank, [](T*) noexcept { /* recycled: already zero, see above */ },
        [](T* vals) noexcept { zero_fill_fresh(vals); });

    if (compare_and_swap(values_loc))
    {
        Base::recup_value(values_loc, thr_rank);
    }

    // ← FIX: fence release so that the stores following in the caller
    //   cannot be reordered before the CAS by the ARM hardware

    std::atomic_thread_fence(std::memory_order_release);
}

// Single definition for both modes. Under OpenMP it delegates to the
// rank-aware overload the policy provides; serially it allocates directly,
// since there is no per-thread free-list to consult. `this->` is required on
// the delegating call: create_value(int) lives in the dependent base
// ChunkOmpPolicy, which unqualified lookup does not search.
template <typename T, typename Derived, bool Enabled>
void ChunkOmpPolicy<T, Derived, Enabled>::create_value() noexcept
{
    create_value(get_thr_rank_fast());
}

template <typename T, typename Derived>
void ChunkOmpPolicy<T, Derived, false>::create_value() noexcept
{
    // Same shared path as the OpenMP overload above; serially there is no reuse
    // list, so the first initialiser is never reached (see the serial
    // ChunkAllocPolicy specialisation, which accepts and ignores it).
    values = Base::allocate_chunk_internal(
        0, [](T*) noexcept {},
        [](T* vals) noexcept
        {
            if constexpr (build_config::use_fast_alloc)
            {
                std::fill_n(vals, alloc_chunk_count(), Base::zero);
            }
        });
}

// Out-of-line body of the Enabled=true specialisation's method. No guard is
// needed: the primary template is only ever instantiated with Enabled=true (the
// false case is a separate specialisation with its own no-op), so this is never
// instantiated in the counter-based build.
template <typename T, typename Derived, bool Enabled>
void ChunkFastCompPolicy<T, Derived, Enabled>::make_block_compression(int thr, const Proxy<T>* proxy)
{
    // Before the first parallel region the per-rank lists do not exist yet
    // (they are sized at region begin), and ArrayChunk::resize() calls this
    // from create(): nothing queued, nothing to do. Indexing the empty vector
    // aborted every v1 array creation under a hardened libstdc++ (multi-mat),
    // and read past it elsewhere.
    if (thr < 0 || static_cast<size_t>(thr) >= manager_chunk::chunk_to_compress_potentially_thr.size())
    {
        return;
    }

    auto& vec_tuple_to_compress = manager_chunk::chunk_to_compress_potentially_thr[thr];

    const int sz = vec_tuple_to_compress.size();

    if (sz == 0)
    {
        return;
    }

    const int max_thr = CHUNK_PMP_MAX_THREADS;

    if (proxy && max_thr > 1)
    {
        // A flush in progress makes start() decline: ask first, before
        // marking -- see block_pass_precheck.h (10 million mark/unmark rounds
        // at 128 threads on Grace otherwise).
        if (block_pass_yields_to_flush<Proxy<T>, Chunk<T>>(max_thr))
        {
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
        const bool ret = proxy->compression_mgr.template start<Proxy<T>>();

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

    vec_tuple_chunkp_int2 ll_compt;

    ll_compt.reserve((sz / 10) + 1);

    fill_list_chunk_to_compress_in_par(thr, ll_compt);

    // Geometry: the list of one rank mixes chunks from EVERY array, at their
    // own sizes once one has been re-chunked. clear_int_st() frees the buffer
    // into the reuse list of the AMBIENT size (free_block_to_pool), so each
    // chunk is pinned to its own n_elems right before its clear, as Chunk2's
    // make_block_compression_impl and Chunk2A's do. Pinned once to the
    // triggering array's size (FastBlockSupport::make_block_compression),
    // a 64-chunk of another array cleared from a pass at 128 parked its
    // buffer in the 128 list, which handed it out as a 128-buffer later:
    // non-reproducible results two iterations after a re-chunk, at 2-3
    // threads (shrink_bench chunk_dyn, forced advisor, Sept '26).
    [[maybe_unused]] const unsigned int saved_sz = shrink_array::thread_chunk_size();
    for (const auto& tup : ll_compt)
    {
        auto* chunk_ptr = std::get<0>(tup);

        if constexpr (build_config::chunk_dynamic)
        {
            shrink_array::set_thread_chunk_size(std::get<3>(tup).count());
        }
        chunk_ptr->clear_int_st(thr);
        chunk_ptr->steal_unmark();
    }
    if constexpr (build_config::chunk_dynamic)
    {
        set_thread_chunk_size(saved_sz);
    }

    for (int k = 0; k < sz; ++k)
    {
        const auto& tup = vec_tuple_to_compress[k];

        const int num_chunk = std::get<1>(tup);

        const int num_tab = std::get<2>(tup);
        Proxy<T>::reset_tab_inside_list_to_compress(num_tab, num_chunk);
        // The pass is over for every candidate: no mark may outlive it.
        if (auto* const chunk_t = std::get<0>(tup))
        {
            chunk_t->steal_unmark();
        }
    }

    vec_tuple_to_compress.clear();
    vec_tuple_to_compress.reserve((sz / 10) + 1);

    if (proxy && max_thr > 1)
    {
        if (proxy->is_not_cst_val())
        {
            proxy->add_to_compression_list(vec_tuple_to_compress);
        }

        proxy->compression_mgr.template finish<Proxy<T>>();
    }
}

// CHUNK_STATS: the release counter differs between modes -- pool under OpenMP
// (the buffer goes back to the per-thread pool), free serially (it is released
// outright). NOT counted by the OMP caller: unlike chunk2/chunk2a, whose
// clear_int_st delegates here, the OMP chunk-v1 path calls clear_int_st
// directly and releases the block itself, so the counter lives there; adding
// one here too would double-count if this entry point is ever wired up.
template <typename T>
void Chunk<T>::block_clear([[maybe_unused]] int thr_rank)
{
    if constexpr (chunk_config::stats)
    {
        increm_chunk_compress<Chunk<T>>();
        add_chunk_bytes_freed<Chunk<T>>(static_cast<long>(sizeof(T)) *
                                        static_cast<long>(shrink_array::alloc_chunk_count()));
        if constexpr (build_config::use_openmp)
        {
            increm_chunk_release_pool<Chunk<T>>();
        }
        else
        {
            increm_chunk_release_free<Chunk<T>>();
        }
    }

    if constexpr (build_config::use_openmp)
    {
        Base::free_block_to_pool(thr_rank, values);
    }
    else
    {
        Base::free_values();
        derived().init_values_cst_val();
    }
}

template <typename T, typename Derived, bool Enabled>
void ChunkOmpPolicy<T, Derived, Enabled>::clear_int_st(int thr_rank)
{
    T* old = try_mark_cst_and_get_old();
    if (old)
    {
        if constexpr (chunk_config::stats)
        {
            increm_chunk_compress<Chunk<T>>();
            add_chunk_bytes_freed<Chunk<T>>(static_cast<long>(sizeof(T)) *
                                            static_cast<long>(shrink_array::alloc_chunk_count()));
            increm_chunk_release_pool<Chunk<T>>();
        }
        Base::free_block_to_pool(thr_rank, old);
    }
}


template <typename T, typename Derived, bool Enabled>
void ChunkOmpPolicy<T, Derived, Enabled>::clear_st(int thr_rank, unsigned int n_elems)
{
    // CHUNK_STATS: a compression completing -- the buffer is parked and the chunk
    // returns to the constant. Counted here rather than in init_values_cst_val:
    // that one is also reached from teardown and neutralisation paths, which are
    // not compressions (it reported 1175% success against `attempts`). The
    // re-chunk rebuild releases old chunks through release_st() instead: counted
    // here, every rebuilt chunk was also a "compression" (198% success, and
    // ~29 GB of phantom compress bytes on a 64-thread shrink_bench run).
    if constexpr (chunk_config::stats)
    {
        increm_chunk_compress<Chunk<T>>();
        add_chunk_bytes_freed<Chunk<T>>(static_cast<long>(sizeof(T)) * static_cast<long>(n_elems));
        increm_chunk_release_park<Chunk<T>>();
    }
    release_st(thr_rank, n_elems, /*count_release=*/false);
}

template <typename T, typename Derived, bool Enabled>
void ChunkOmpPolicy<T, Derived, Enabled>::release_st(int thr_rank, unsigned int n_elems, bool count_release)
{
    // CHUNK_STATS: a release that is not a compression (the re-chunk rebuild's
    // old chunks) -- its bytes leave the live figure all the same.
    if constexpr (chunk_config::stats)
    {
        if (count_release)
        {
            add_chunk_released<Chunk<T>>(1, static_cast<long>(sizeof(T)) * static_cast<long>(n_elems));
        }
    }

    if constexpr (build_config::any_check)
    {
        if (!derived().written())
        {
            invariant_failed(std::source_location::current(), "trying to clear a chunk that is not written (chunk={})",
                             static_cast<const void*>(this));
        }
    }

    // Park into the list keyed by the buffer's OWN geometry (n_elems), not the
    // ambient one. get_list_block_free_thr(thr) routes by thread_chunk_size(),
    // i.e. by whichever array happened to write last. Once two geometries are
    // alive at once -- a re-chunked array next to one left at the original size
    // -- that is the wrong list roughly half the time: a 64-element buffer freed
    // by the 64-array while the ambient reads 256 lands in the 256 list, and
    // the next create_value() at 256 pops it for a 256-element chunk. ASan then
    // shows tab2 reading 0 bytes past a 512-byte region still owned by tab5.
    //
    // The memset below already used n_elems for exactly this reason; the push
    // must agree with it. Same defect family as 4f5a6c48 ("pin ambient on the
    // free path, not just alloc"), fixed for Chunk2 and never for v1.
    auto& ll_block_free_thr = manager_chunk::get_list_block_free_thr_for_size(thr_rank, n_elems);

    Base::init_block(values.load(), n_elems);
    CHUNK_LEDGER_PARK(values.load(), "clear_st.park"); // LIVE -> PARKED before it re-enters the pool
    ll_block_free_thr.push_back(values);

    derived().init_values_cst_val();
}

template <typename T, typename Derived>
void ChunkOmpPolicy<T, Derived, false>::clear_st()
{
    // CHUNK_STATS: serial twin of the OpenMP clear_st above -- count the
    // compression, then release.
    if constexpr (chunk_config::stats)
    {
        increm_chunk_compress<Chunk<T>>();
        add_chunk_bytes_freed<Chunk<T>>(static_cast<long>(sizeof(T)) *
                                        static_cast<long>(shrink_array::alloc_chunk_count()));
        increm_chunk_release_free<Chunk<T>>();
    }
    release_st(/*count_release=*/false);
}

template <typename T, typename Derived>
void ChunkOmpPolicy<T, Derived, false>::release_st(bool count_release)
{
    if constexpr (chunk_config::stats)
    {
        if (count_release)
        {
            add_chunk_released<Chunk<T>>(1, static_cast<long>(sizeof(T)) *
                                                static_cast<long>(shrink_array::alloc_chunk_count()));
        }
    }

    if constexpr (build_config::is_debug)
    {
        if (!derived().written())
        {
            std::cerr << "Error: trying to clear a chunk that is not written\n";
            return;
        }
    }

    Base::free_values();

    derived().init_values_cst_val();
}

} // namespace shrink_array