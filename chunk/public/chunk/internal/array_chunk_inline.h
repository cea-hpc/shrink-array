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

#include <array>  // std::array (the residual's sample)
#include <vector> // std::vector

#include "chunk/all_type_chunk.h"                       // ensure_manager, from create
#include "chunk/array_chunk.h"                          // ArrayChunk, the class these bodies belong to
#include "chunk/internal/chunk_stats.h"                 // increm_chunk_* (CHUNK_STATS)
#include "chunk/internal/chunk_stats_printer.h"         // print_formatted_chunk_stats
#include "chunk/internal/manager_chunk.h"               // per-size free lists, candidate filter
#include "chunk/PMP_CHUNK.h"                            // CHUNK_PMP_*
#include "chunk/internal/pool_ledger.h"                 // CHUNK_LEDGER_PARK / DISCARD
#include "chunk/type_name.h"                            // type_name<T>
#include "chunk/config.h"                               // build_config::chunk_dynamic, chunk_config::stats
#include "chunk/internal/invariant_check.h"             // CHUNK_INVARIANT_FAILED
#include "chunk/internal/chunk_index_sample.h"          // chunk_index_sample (the residual's sample)
#include "chunk/log.h"                                  // CHUNK_LOG_ALWAYS
#include "chunk/internal/chunk_size_policy.h"           // set_thread_chunk_size
#include "chunk/internal/concept_array_chunk_element.h" // ArrayChunkElement
#include "chunk/internal/parallel_storage_init.h"       // for_each_chunk
#include "chunk/internal/in_t.h"                        // in_t<T>

namespace shrink_array
{

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
            CHUNK_INVARIANT_FAILED("create({} elements): the chunk size is 0 (no geometry set)", sz);
        }

        if (nb_chunk)
        {
            this->acquire_chunk_list(static_cast<std::size_t>(nb_chunk));
            this->alloc_dirty_bitset(nb_chunk);
            this->assign_alloc_key(); // own slabs if large enough (see alloc_key_)

            // A team of its own outside a region; the calling thread inside one
            // (an orphaned `omp for` would initialise 1/N of an array created by
            // each thread, and hang one created under `omp single`).
            parallel_storage_init::for_each_chunk(nb_chunk,
                                                  [this](int i) { this->construct_chunk(i).init_values_cst_val(); });
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

    const auto& sh_inf  = *Proxy<T>::share_info_array;
    const int   num_tab = sh_inf.get_num_tab(this->registry_key());

    // Read before the clear() drain mutates state. Not written() = compressed
    // to first_value. A written() chunk is also classified by CONTENT (as in
    // Chunk2/Chunk2A): whether the deferred compression caught it depends on
    // the thread count, so counting by content makes still@exit a function of
    // the data alone.
    //
    // Cross-variant probe: a block constant to another value than first_value
    // stays materialised here but a Chunk2/Chunk2A would have compressed it;
    // the same scan counts those. O(materialised * chunk_size), once, shared
    // over a team (never nested in a caller's region, not for a few chunks).
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

    // Plain Chunk has no type-2 (n_t1 = 0).
    if constexpr (chunk_config::stats)
    {
        add_chunk_still_compressed_at_exit<Chunk<T>>(n_compressed);
        // Dyn-only end-of-life size distribution (see chunk_stats.h).
        record_chunk_size_at_exit<Chunk<T>>(static_cast<unsigned int>(this->size_policy_.size()));
    }
    print_formatted_chunk_stats(os, num_tab, this->debug_name(), type_name<T>(), n_compressed, nb_chunk,
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
    // Pin the ambient size to THIS array's geometry before the sweep:
    // get_chunk_allocator() routes on thread_chunk_size(), not on the buffer, so
    // a size left by another array would send these buffers to the wrong
    // allocator. See ArrayChunk2::clear_core().
    if constexpr (build_config::chunk_dynamic)
    {
        set_thread_chunk_size(static_cast<unsigned int>(this->size_policy_.size()));
    }

    // Drain deferred writes before chunk_list is freed. force: the pending-op
    // counter can read 0 while ops are queued (array_chunk_parallel_common_inline.h),
    // so a point that frees chunks must not trust it.
    this->flush_pending_deferred_ops(/*force=*/true);

    // Remove this array's queued candidates before its chunks are freed: they
    // would dangle for the next make_block_compression, and rechunk_to relies on
    // it to keep foreign-geometry chunks out of the shared list.
    if constexpr (chunk_config::use_fast_block_compression)
    {
        const auto& sh_inf   = *Proxy<T>::share_info_array;
        const int   num_tab_ = sh_inf.get_num_tab(this->registry_key());
        manager_chunk::remove_from_compress_list_chunk_by_num_tab(num_tab_);
    }

    if (!nb_chunk)
    {
        m_size = 0;
        chunk_list.clear();
        return;
    }

    // Give this array's chunks back to the manager's count (as ArrayChunk2 /
    // 2A do): the compression threshold (somme_chunk_create / (factor_div *
    // threads)) must track the chunks alive, not the program's history --
    // otherwise it climbs by orders of magnitude, compressions become rare and
    // buffers stop returning to the reuse lists. Also balances rechunk_to()
    // (clear_core() -> rebuild_storage()).
    manager_chunk::decrem_somme_chunk_create(nb_chunk);
    // And recompute what derives from the total, as rebuild_storage() does: a
    // re-chunk would otherwise keep a threshold sized for the old geometry
    // (several times too large, a large throughput loss).
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
                // ambient one. If my_sz has no per-size list this falls back to
                // the LEGACY list, which drain_reuse_list_for_size() never
                // drains: those blocks then stay until teardown.
                const unsigned int my_sz = this->size_policy_.size();
                auto&              lbf   = Chunk<T>::manager_chunk::get_list_block_free_thr_for_size(thr_rank, my_sz);

                for (int k = 0; k < nb; ++k)
                {
                    T*          values_loc = ll_chunk_loc[k]->values;
                    constexpr T zero{};
                    std::uninitialized_fill_n(values_loc, my_sz, zero);

                    // Ledger LIVE -> PARKED (no-op when the ledger is off).
                    CHUNK_LEDGER_PARK(values_loc, "clear_core.teardown_park");
                    lbf.push_back(values_loc);

                    // Neutralise the chunk NOW that its buffer is parked: still
                    // materialised, a residual candidate would free the SAME
                    // buffer a second time (a double free under std-alloc).
                    // init_values_cst_val() frees nothing; written() then
                    // reports false, so every compress path short-circuits.
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

    this->release_chunk_list(); // Mandatory with the new allocator (a re-chunk keeps the buffer)

    nb_chunk = 0;

    // An array with its own slab affinity gives its memory back NOW: its parked
    // buffers go home to its allocator set, whose slabs then empty (and
    // reclaim_empty_slabs madvises them). Sequential point; the drain opens its
    // own region. Key 0: nothing to give back per array.
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
    // Pin the ambient size to THIS array's geometry (see clear_core()).
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
            // One compression attempt per materialised chunk: this sweep
            // bypasses compress(), which would otherwise count it.
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
    // compress_int() only SCANS. It finds the residual: chunks constant at zero
    // and still materialised. NOT an invariant violation: compression is
    // incremental (a write only QUEUES its chunk), so such a chunk normally
    // waits for the next pass. Its SIZE and TREND matter: a handful is normal,
    // a large or growing count means the incremental path is not keeping up.
    // Written when the caller asks (`verbose`: a report, whatever CHUNK_LOG
    // says), one line per thread (its share of the chunks).
    if (nb != 0 && verbose)
    {
        std::array<char, 256> sample{};
        CHUNK_LOG_ALWAYS(
            compress,
            "residual ArrayChunk<{}, {}>: {} chunk(s) constant at zero, still materialised (awaiting the next "
            "compression pass): {}",
            type_name<T>(), type_name<ind_type>(), nb,
            internal::chunk_index_sample(sample, ll, static_cast<ind_type>(this->size_policy_.size())));
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
        // A large shrink takes a team of its own, as clear_core()'s release
        // does: each thread zeroes and parks into ITS rank's reuse list of this
        // array's geometry. Inside a region, or for a few chunks, the calling
        // thread.
        const unsigned int my_sz = this->size_policy_.size();
        parallel_storage_init::for_each_chunk_in(
            first, last, [my_sz]() -> auto&
            { return Chunk<T>::manager_chunk::get_list_block_free_thr_for_size(CHUNK_PMP_THREAD_RANK, my_sz); },
            [this, my_sz](int j, auto& lbf)
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
            });
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

    // Drain the deferred ops on the CURRENT chunks before chunk_list and the
    // dirty bitset are reallocated: a queued op holds a reference to its chunk.
    // And BEFORE the compression pass (as ArrayChunk2Base::resize_common()): a
    // replayed write can queue its chunk again, and the realloc and
    // resize_tab_is_int() rely on the pass leaving the lists empty.
    // force: see clear_core().
    this->flush_pending_deferred_ops(/*force=*/true);

    if constexpr (chunk_config::use_fast_block_compression)
    {
        make_block_compression();
    }

    const auto& sh_inf = *Proxy<T>::share_info_array;

    const int num_tab = sh_inf.get_num_tab(this->registry_key());

    const ind_type old_size = m_size; // the stale-tail reset at the end

    m_size = newsz;

    const int old_nb_chunk = nb_chunk;

    nb_chunk = this->chunks_for(newsz);

    // Shrink: ~Chunk() releases nothing, so release the dropped chunks' buffers
    // first (the pass above emptied the candidate lists).
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

    // The new chunks [old_nb_chunk, nb_chunk): a team of its own outside a
    // region, the calling thread inside one -- never an orphaned `omp for`
    // (it would construct only 1/N of a thread's own array).
    parallel_storage_init::for_each_chunk(nb_chunk - old_nb_chunk, [this, old_nb_chunk](int k)
                                          { this->construct_chunk(old_nb_chunk + k).init_values_cst_val(); });

    // std::vector semantics: the regrown range [old_size, newsz) reads T{}. The
    // chunk holding the OLD end keeps its tail (values from before a shrink),
    // so reset it through the normal write path. Serial: inside a region an
    // `omp for` would cover 1/N of it.
    if (newsz > old_size)
    {
        const auto csz = static_cast<ind_type>(this->chunk_size());
        // Up to the end of the OLD chunk count: chunks_for() keeps a spare chunk
        // when old_size falls on a boundary, stale too. In long long:
        // old_nb_chunk * csz may pass ind_type's range.
        const long long cover_ll  = static_cast<long long>(old_nb_chunk) * static_cast<long long>(csz);
        const ind_type  old_cover = cover_ll < static_cast<long long>(newsz) ? static_cast<ind_type>(cover_ll) : newsz;
        const ind_type  stale_end = newsz < old_cover ? newsz : old_cover;
        for (ind_type i = old_size; i < stale_end; ++i)
        {
            (*this)[i] = T{};
        }
    }
}

} // namespace shrink_array
