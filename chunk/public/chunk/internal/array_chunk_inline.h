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

#include <cstdlib>  // std::abort
#include <iostream> // std::cerr
#include <vector>   // std::vector

#include "chunk/all_type_chunk.h"                       // ensure_manager, from create
#include "chunk/array_chunk.h"                          // ArrayChunk, the class these bodies belong to
#include "chunk/internal/chunk_stats.h"                 // increm_chunk_* (CHUNK_STATS)
#include "chunk/internal/chunk_stats_printer.h"         // print_formatted_chunk_stats
#include "chunk/internal/manager_chunk.h"               // per-size free lists, candidate filter
#include "chunk/PMP_CHUNK.h"                            // CHUNK_PMP_*
#include "chunk/internal/pool_ledger.h"                 // CHUNK_LEDGER_PARK / DISCARD
#include "chunk/type_name.h"                            // type_name<T>
#include "chunk/config.h"                               // build_config::chunk_dynamic, chunk_config::stats
#include "chunk/internal/chunk_size_policy.h"           // set_thread_chunk_size
#include "chunk/internal/concept_array_chunk_element.h" // ArrayChunkElement
#include "chunk/internal/in_t.h"                        // in_t<T>

namespace shrink_array
{

template <ArrayChunkElement T, typename ind_type>
void ArrayChunk<T, ind_type>::create(ind_type sz)
{
    // The first array of this element type registers its manager; every
    // later one pays a guard load. See AllTypeChunk.
    AllTypeChunk::ensure_manager<ManagerChunk<Chunk, T>>();

    Base::create_common();

    rebuild_storage(sz);
}

template <ArrayChunkElement T, typename ind_type>
void ArrayChunk<T, ind_type>::rebuild_storage(ind_type sz)
{
    if (sz)
    {
        if (this->size_policy_.size())
        {
            nb_chunk = this->chunks_for(sz);
            manager_chunk::increm_somme_chunk_create(nb_chunk);
            manager_chunk::compute_size_buffer_reuse();
            manager_chunk::compute_size_max_number_of_chunk_to_check_compress();

            if constexpr (chunk_config::use_fast_block_compression)
            {
                Proxy<T>::resize_tab_is_int(this->num_tab, nb_chunk);
            }
        }
        else if (sz && !this->size_policy_.size())
        {
            std::cerr << "__chunk_size null !\n";
            std::abort();
        }

        if (nb_chunk)
        {
            chunk_list.resize(nb_chunk);
            this->alloc_dirty_bitset(nb_chunk);
            this->assign_alloc_key(); // own slabs if large enough (see alloc_key_)

            if (!CHUNK_PMP_IN_PARALLEL_REGION && nb_chunk > 500)
            {
#pragma omp parallel for
                for (int i = 0; i < nb_chunk; ++i)
                {
                    chunk_list[i].init_values_cst_val();
                }
            }
            else
            {
#pragma omp for
                for (int i = 0; i < nb_chunk; ++i)
                {
                    chunk_list[i].init_values_cst_val();
                }
            }
        }
    }

    m_size = sz;
}

template <ArrayChunkElement T, typename ind_type>
void ArrayChunk<T, ind_type>::print_compression_stats(std::ostream& os) const
{
    if (nb_chunk == 0)
    {
        return;
    }

    const auto& sh_inf  = Proxy<T>::share_info_array;
    const int   num_tab = sh_inf.get_num_tab(this->registry_key());

    // Plain Chunk classification: a block that is NOT written() is stored in its
    // compressed (constant == first_value) form; written() means it is
    // materialised. Read here, before the clear() drain below mutates state.
    //
    // Content check, on top of the written() classification -- same as
    // Chunk2/Chunk2A. written() only says whether the deferred compression
    // happened to catch a chunk during the run, not whether its CURRENT content
    // is compressible: a materialised block that is all first_value is one the
    // sequential build would have collapsed eagerly, and the fraction the live
    // drain caught before teardown depends on the thread count (measured: 649
    // vs 1 of 4097 chunks at 1 vs 6 threads on identical final data). Counting
    // by content makes still@exit a function of the data alone.
    //
    // Cross-variant probe ("would Chunk2 have helped?"): plain Chunk can only
    // collapse a block whose constant is first_value. A block constant to some
    // OTHER value stays materialised here, yet a Chunk2/Chunk2A would have
    // compressed it as a second constant. The same scan counts those.
    // This is the real "did a richer variant have headroom" measure — it walks
    // the block contents, not a candidate list. O(materialised * chunk_size),
    // once, at teardown -- and every chunk is classified on its own, so the
    // walk is shared out over the team when one is available (the counts are
    // reductions). Same guard as the other teardown loops: never nest inside a
    // caller's region, and not worth a team for a handful of chunks.
    const auto csz = static_cast<unsigned int>(this->size_policy_.size());

    int n_compressed  = 0;
    int n_would_be_t2 = 0;

#pragma omp parallel for reduction(+ : n_compressed, n_would_be_t2) if (!CHUNK_PMP_IN_PARALLEL_REGION && nb_chunk > 500)
    for (int j = 0; j < nb_chunk; ++j)
    {
        const auto& ck = chunk_list[j];
        if (!ck.written())
        {
            ++n_compressed; // collapsed to first_value
            continue;
        }
        // Materialised: is it constant to a single value?
        const auto v0 = ck.constant_value(csz);
        if (!v0)
        {
            continue;
        }
        if (*v0 == Chunk<T>::first_value)
        {
            ++n_compressed; // compressible now; the drain just had not caught it yet
        }
        else
        {
            ++n_would_be_t2; // a second constant: Chunk2/Chunk2A material
        }
    }

    // Plain Chunk has no type-2 (n_t1 = 0). The cross probe reports how many
    // materialised blocks a Chunk2 (or Chunk2A) would have captured.
    if constexpr (chunk_config::stats)
    {
        add_chunk_still_compressed_at_exit<Chunk<T>>(n_compressed);
        // Dyn-only: record this array's CURRENT chunk size in the end-of-life
        // size distribution (no-op on static builds, see chunk_stats.h).
        record_chunk_size_at_exit<Chunk<T>>(static_cast<unsigned int>(this->size_policy_.size()));
    }
    print_formatted_chunk_stats(os, num_tab, type_name<T>(), n_compressed, nb_chunk,
                                /*n_t1=*/0, "Chunk",
                                /*n_cross=*/n_would_be_t2, /*cross_name=*/"Chunk2");
}

template <ArrayChunkElement T, typename ind_type>
void ArrayChunk<T, ind_type>::clear()
{
    this->invalidate_proxies();
    // Per-array compression stats, read BEFORE the drain below mutates state.
    if (this->exit_census_)
    {
        print_compression_stats();
    }

    clear_core();
}

template <ArrayChunkElement T, typename ind_type>
void ArrayChunk<T, ind_type>::clear_core()
{
    // Pin the ambient size to THIS array's own geometry before the sweep
    // below: get_chunk_allocator() routes on thread_chunk_size(), not on the
    // buffer being freed, so a size left by another array's write sent these
    // buffers to the wrong size's allocator (sl_contains assertion in
    // TCTypeAllocator after a re-chunk, fast-alloc only). ArrayChunk2 and
    // ArrayChunk2A had this pin; v1 did not. See ArrayChunk2::clear_core().
    if constexpr (build_config::chunk_dynamic)
    {
        set_thread_chunk_size(static_cast<unsigned int>(this->size_policy_.size()));
    }

    // Drain deferred writes on this array's chunks before chunk_list is freed.
    // force: the pending-op counter has been seen to read 0 while ops were
    // still queued (array_chunk_parallel_common_inline.h); a sequential point
    // that reallocates or frees chunks cannot trust it -- as rechunk, swap and
    // ArrayChunk2A::clear_core() already do.
    this->flush_pending_deferred_ops(/*force=*/true);

    // Remove this array's queued compression candidates by num_tab before its
    // chunks are freed. v1's clear() runs no terminal sweep, but candidates left
    // behind would dangle for the next make_block_compression; and rechunk_to
    // relies on this to keep the shared list free of foreign-geometry chunks.
    if constexpr (chunk_config::use_fast_block_compression)
    {
        const auto& sh_inf   = Proxy<T>::share_info_array;
        const int   num_tab_ = sh_inf.get_num_tab(this->registry_key());
        manager_chunk::remove_from_compress_list_chunk_by_num_tab(num_tab_);
    }

    if (!nb_chunk)
    {
        m_size = 0;
        chunk_list.clear();
        return;
    }

    // Give this array's chunks back to the manager's chunk count, exactly as
    // ArrayChunk2::clear_core() and ArrayChunk2A::clear_core() do. Without it
    // somme_chunk_create only ever grows in v1: every array constructed since
    // program start stays counted after its destruction, so the compression
    // threshold (somme_chunk_create / (factor_div * threads), see
    // compute_size_max_number_of_chunk_to_check_compress) climbs linearly with
    // the program's history instead of tracking the chunks currently alive.
    // Measured on v1 dyn: threshold 390 at start of run, ~42 000 by the end,
    // hence fewer and fewer compressions, hence buffers almost never returned
    // to the reuse lists, hence 66.8 M fresh allocations for 788 k
    // compressions. rechunk_to() goes clear_core() -> rebuild_storage(), so the
    // decrement here keeps that path balanced too.
    manager_chunk::decrem_somme_chunk_create(nb_chunk);
    // And recompute what derives from the total, as rebuild_storage() does
    // after its increment: a re-chunk creates the new storage while the old
    // one still counts, then destroys the old one here -- without this the
    // compression threshold stays sized for the geometry it just left (v1
    // doubles 64 -> 1024: ~3400 candidates where ~900 is right; 1.5x less
    // throughput for the rest of the 64-thread chunk_dyn bench).
    manager_chunk::compute_size_buffer_reuse();
    manager_chunk::compute_size_max_number_of_chunk_to_check_compress();

    if constexpr (build_config::use_openmp)
    {
#pragma omp parallel
        {
            const int nb_chunk0 = get_size_max_tab_omp(nb_chunk);

            std::vector<Chunk<T>*> ll_chunk_loc;
            ll_chunk_loc.reserve(nb_chunk0);

#pragma omp for schedule(static) nowait
            for (int j = 0; j < nb_chunk; ++j)
            {
                if (chunk_list[j].written())
                {
                    if constexpr (chunk_config::stats)
                    {
                        increm_chunk_teardown_clear<Chunk<T>>();
                        // Released below without a compression: out of the live figures.
                        add_chunk_released<Chunk<T>>(1, static_cast<long>(sizeof(T)) *
                                                            static_cast<long>(this->size_policy_.size()));
                    }
                    ll_chunk_loc.push_back(&chunk_list[j]);
                }
            }

            const int nb = ll_chunk_loc.size();

            if (nb > 0)
            {
                const int thr_rank = CHUNK_PMP_THREAD_RANK;

                // Park into the reuse list of THIS array's geometry, not the
                // ambient t_active (which may be another size mid-rechunk).
                // Falls back to the legacy single list in single-size builds.
                const unsigned int my_sz = this->size_policy_.size();
                auto&              lbf   = Chunk<T>::manager_chunk::get_list_block_free_thr_for_size(thr_rank, my_sz);

                // Where these buffers land matters: get_list_block_free_thr_for_size()
                // falls back to the LEGACY list when my_sz has no per-size list,
                // and drain_reuse_list_for_size() only ever drains the by-size
                // map -- so a fallback means the old geometry's blocks are never
                // drained at a re-chunk and pile up until teardown.
                for (int k = 0; k < nb; ++k)
                {
                    T*          values_loc = ll_chunk_loc[k]->values;
                    constexpr T zero{};
                    std::uninitialized_fill_n(values_loc, my_sz, zero);

                    // Ledger LIVE -> PARKED (no-op when the ledger is off).
                    CHUNK_LEDGER_PARK(values_loc, "clear_core.teardown_park");
                    lbf.push_back(values_loc);

                    // Neutralise the chunk NOW that its buffer is parked.
                    // Otherwise the chunk stays materialised -- values still
                    // points at the buffer that now belongs to the reuse list --
                    // so a residual compression candidate, one that slipped the
                    // removal above or the terminal sweep, would
                    // free_block_to_pool the SAME buffer a second time. Under
                    // std-alloc that frees it twice (glibc "corrupted
                    // double-linked list" at teardown, after a re-chunk);
                    // fast-alloc parks instead of svfreeing, which hides it.
                    //
                    // init_values_cst_val() only assigns values = cst_val1: it
                    // frees nothing, and written() then reports false, so every
                    // compress path short-circuits. Same as
                    // ArrayChunk2::clear_core and ArrayChunk2A::clear_core.
                    ll_chunk_loc[k]->init_values_cst_val();
                }
            }
        }
    }
    else
    {
#pragma omp for schedule(static) nowait
        for (int j = 0; j < nb_chunk; ++j)
        {
            if (chunk_list[j].written())
            {
                if constexpr (chunk_config::stats)
                {
                    increm_chunk_teardown_clear<Chunk<T>>();
                }
                // release_st: counted as teardown_clear above, not also as a
                // compression.
                chunk_list[j].release_st();
            }
        }
    }
    m_size = 0;

    chunk_list.clear(); // Mandatory with the new allocator

    nb_chunk = 0;

    // An array with its own slab affinity gives its memory back NOW: every
    // buffer parked above (own rank) or by another rank (foreign, under this
    // array's key) goes home to its allocator set, whose slabs hold nothing
    // else and so empty -- and reclaim_empty_slabs madvises them away.
    // Sequential point (destructor / rechunk_to); the drain opens its own
    // region. Key 0 (shared pool): nothing to give back per array. Same hook
    // as ArrayChunk2 / ArrayChunk2A.
    if (this->alloc_key_ != 0 && !CHUNK_PMP_IN_PARALLEL_REGION)
    {
        ManagerChunk<Chunk, T>::drain_reuse_list_for_size(this->size_policy_.size());
    }

    // The candidate flags (one byte per chunk): released now that the purge
    // at the top has run and the chunks are gone -- a re-chunk's old storage
    // dies under its own num_tab, which nothing reuses.
    Proxy<T>::release_tab_is_int(this->num_tab);
}

template <ArrayChunkElement T, typename ind_type>
void ArrayChunk<T, ind_type>::compress_int(bool verbose)
{
    // Pin the ambient size to THIS array's geometry: the frees below route on
    // thread_chunk_size(), not on the buffer (see clear_core()). Left to
    // whichever array the thread last served, a re-chunked array's buffers
    // went to another size's allocator (sl_remove assertion, fast-alloc only).
    if constexpr (build_config::chunk_dynamic)
    {
        shrink_array::set_thread_chunk_size(static_cast<unsigned int>(this->size_policy_.size()));
    }


    const int nb_chunk0 = get_size_max_tab_omp(nb_chunk, true);

    std::vector<int> ll;
    ll.reserve(nb_chunk0);

#pragma omp for schedule(static) nowait
    for (int j = 0; j < nb_chunk; ++j)
    {
        const auto& chunk = chunk_list[j];

        if (chunk.written())
        {
            // CHUNK_STATS: this sweep is a compression attempt per materialised
            // chunk. It reaches clear_st() directly (see below), bypassing
            // compress(), so without this the successes were counted but not the
            // tries -- attempts showed 1175% success.
            if constexpr (chunk_config::stats)
            {
                increm_chunk_compress_attempt<Chunk<T>>();
            }

            if (chunk.is_constant_to(T{}, static_cast<unsigned int>(this->size_policy_.size())))
            {
                ll.push_back(j);
            }
        }
    }

    const auto nb = static_cast<int>(ll.size());

    // ── What this reports, and what it does NOT assert ───────────────────────
    // In embedded-compression mode compress_int() only SCANS. What it finds is
    // the residual: chunks that are constant at zero right now and are still
    // materialised. That is NOT an invariant violation, which is what the
    // previous wording ("Should not have compress (0) values") claimed.
    // Compression is incremental -- a write only QUEUES its chunk, and the
    // queue is drained by the phase-end pass -- so between two passes a chunk
    // that has just become constant is normally waiting for the next one.
    // ArrayChunk2A's copy of this scan says the same at length; the two now
    // agree.
    //
    // What the number IS good for is its SIZE and its TREND: a handful is the
    // normal in-flight residual, a large or growing count means the incremental
    // path is not keeping up -- which is exactly how the ~500 residual that led
    // to drain_compression_backlog() in rechunk_validation.cpp was found.
    //
    // And it is printed only when asked. The parameter was declared and never
    // read, so callers passing false -- rechunk_validation.cpp does, on
    // purpose -- got the dump anyway.
    if (nb != 0 && verbose)
    {
#pragma omp critical
        {
            std::cerr << "[compress-residual chunk] " << nb
                      << " chunk(s) constant at zero, still materialised "
                         "(awaiting the next compression pass)\n";
            std::cerr << "ArrayChunk type: ArrayChunk<" << type_name<T>() << ", " << type_name<ind_type>() << ">"
                      << std::endl;

            // Cap at 10 ENTRIES. `k > 10` printed twelve, and the ellipsis
            // below it fired on nb > 10, so eleven and twelve entries came out
            // truncated with nothing saying so.
            const int shown = nb < 10 ? nb : 10;
            for (int k = 0; k < shown; ++k)
            {
                const int ind1 = ll[k];
                std::cerr << ind1 << " "
                          << static_cast<ind_type>(ind1) * static_cast<ind_type>(this->size_policy_.size()) << "\n";
            }
            if (nb > shown)
            {
                std::cerr << "... (" << (nb - shown) << " more)\n";
            }
        }
    }
}

// make_block_compression() is FastBlockSupport<ArrayChunk, true>'s
// (array_chunk_parallel_inline.h); the false specialisation is a no-op.
template <ArrayChunkElement T, typename ind_type>
void ArrayChunk<T, ind_type>::release_chunk_range(int first, int last)
{
    if (first >= last)
    {
        return;
    }
    // The release routes by the ambient geometry: pin it to this array's.
    if constexpr (build_config::chunk_dynamic)
    {
        set_thread_chunk_size(static_cast<unsigned int>(this->size_policy_.size()));
    }

    if constexpr (build_config::use_openmp)
    {
        const int          thr_rank = CHUNK_PMP_THREAD_RANK;
        const unsigned int my_sz    = this->size_policy_.size();
        auto&              lbf      = Chunk<T>::manager_chunk::get_list_block_free_thr_for_size(thr_rank, my_sz);
        for (int j = first; j < last; ++j)
        {
            if (chunk_list[j].written())
            {
                if constexpr (chunk_config::stats)
                {
                    increm_chunk_teardown_clear<Chunk<T>>();
                    add_chunk_released<Chunk<T>>(1, static_cast<long>(sizeof(T)) * static_cast<long>(my_sz));
                }
                // As clear_core(): v1's reuse lists hold zeroed buffers.
                T*          values_loc = chunk_list[j].values;
                constexpr T zero{};
                std::uninitialized_fill_n(values_loc, my_sz, zero);
                CHUNK_LEDGER_PARK(values_loc, "resize.shrink_park");
                lbf.push_back(values_loc);
                chunk_list[j].init_values_cst_val();
            }
        }
    }
    else
    {
        for (int j = first; j < last; ++j)
        {
            if (chunk_list[j].written())
            {
                if constexpr (chunk_config::stats)
                {
                    increm_chunk_teardown_clear<Chunk<T>>();
                }
                chunk_list[j].release_st();
            }
        }
    }
}

template <ArrayChunkElement T, typename ind_type>
void ArrayChunk<T, ind_type>::resize(ind_type newsz)
{
    this->invalidate_proxies();

    // Drain the deferred ops on the CURRENT chunks, before chunk_list (a
    // std::vector) and the dirty bitset are reallocated below: a queued op
    // holds a reference to its chunk, and replaying it after the realloc
    // wrote into freed memory. The declaration always said "drained first";
    // the body never did. No-op without fast-block compression.
    //
    // And BEFORE the compression pass, not after -- as in
    // ArrayChunk2Base::resize_common(): a replayed write can queue its chunk
    // again (flag 1, &chunk_list[j] pushed). The pass that follows empties
    // the lists and clears the flags, which is what the realloc and
    // Proxy<T>::resize_tab_is_int() below rely on.
    // force: the pending-op counter has been seen to read 0 while ops were
    // still queued (array_chunk_parallel_common_inline.h); a sequential point
    // that reallocates or frees chunks cannot trust it -- as rechunk, swap and
    // ArrayChunk2A::clear_core() already do.
    this->flush_pending_deferred_ops(/*force=*/true);

    if constexpr (chunk_config::use_fast_block_compression)
    {
        make_block_compression();
    }

    const auto& sh_inf = Proxy<T>::share_info_array;

    const int num_tab = sh_inf.get_num_tab(this->registry_key());

    const ind_type old_size = m_size; // the stale-tail reset at the end

    m_size = newsz;

    const int old_nb_chunk = nb_chunk;

    nb_chunk = this->chunks_for(newsz);

    // Shrink: chunk_list.resize() destroys the chunks past the new end, and
    // ~Chunk() releases nothing -- their buffers leaked. Release them first,
    // as clear_core() does; the pass above emptied the candidate lists.
    if (nb_chunk < old_nb_chunk)
    {
        release_chunk_range(nb_chunk, old_nb_chunk);
    }

    chunk_list.resize(nb_chunk);
    Base::reallocate_dirty_bitset(old_nb_chunk, nb_chunk);

    if constexpr (chunk_config::use_fast_block_compression)
    {
        Proxy<T>::resize_tab_is_int(num_tab, nb_chunk);
    }
    manager_chunk::increm_somme_chunk_create(nb_chunk - old_nb_chunk);
    manager_chunk::compute_size_buffer_reuse();
    manager_chunk::compute_size_max_number_of_chunk_to_check_compress();
    // An array built empty (the default) only becomes worth its own slabs
    // here; idempotent for one that already has a key.
    this->assign_alloc_key();

    if (!CHUNK_PMP_IN_PARALLEL_REGION && nb_chunk > 500)
    {
#pragma omp parallel for
        for (int i = old_nb_chunk; i < nb_chunk; ++i)
        {
            chunk_list[i].init_values_cst_val();
        }
    }
    else
    {
#pragma omp for
        for (int i = old_nb_chunk; i < nb_chunk; ++i)
        {
            chunk_list[i].init_values_cst_val();
        }
    }

    // std::vector semantics for the regrown range [old_size, newsz): it reads
    // T{}. The chunks added above are fresh, but the one holding the OLD end
    // keeps its tail -- after a shrink, the values written there before would
    // come back on the next growth (deferred_path_validation, resize-grow-again).
    // Reset that tail: at most one chunk, through the normal write path, after
    // the reallocations. `omp for` like the init loop above.
    if (newsz > old_size)
    {
        const auto     csz       = static_cast<ind_type>(this->chunk_size());
        // Up to the end of the OLD chunk count: chunks_for() keeps a chunk past
        // the last element, so when old_size fell on a chunk boundary that
        // spare chunk kept its stale values too (pool_accounting, grow back).
        // In long long: old_nb_chunk * csz may pass ind_type's range near its top.
        const long long cover_ll  = static_cast<long long>(old_nb_chunk) * static_cast<long long>(csz);
        const ind_type  old_cover = cover_ll < static_cast<long long>(newsz) ? static_cast<ind_type>(cover_ll) : newsz;
        const ind_type stale_end = newsz < old_cover ? newsz : old_cover;
#pragma omp for
        for (ind_type i = old_size; i < stale_end; ++i)
        {
            (*this)[i] = T{};
        }
    }
}

template <ArrayChunkElement T, typename ind_type>
void ArrayChunk<T, ind_type>::resize(ind_type newsz, in_t<T> val)
{
    const ind_type old_size = m_size;

    resize(newsz);

    const auto lbd_init = [this, old_size, newsz, val]
    {
#pragma omp for schedule(static) nowait
        for (ind_type i = old_size; i < newsz; ++i)
        {
            (*this)[i] = val;
        }
    };

    if (!CHUNK_PMP_IN_PARALLEL_REGION && nb_chunk > 500)
    {
#pragma omp parallel
        {
            lbd_init();
        }
    }
    else
    {
        lbd_init();
    }
}

} // namespace shrink_array
