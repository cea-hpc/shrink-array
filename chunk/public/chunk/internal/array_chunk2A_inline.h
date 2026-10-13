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

#include <atomic>      // std::memory_order (the missed-compression audit)
#include <cstddef>     // std::size_t
#include <cstdio>      // std::fprintf (check-build trace)
#include <iomanip>     // std::setw (print_type_name)
#include <iostream>    // std::cout / std::cerr
#include <mutex>       // std::scoped_lock (the residual report)
#include <string>      // the type name in print_type_name
#include <type_traits> // std::remove_reference_t
#include <utility>     // std::pair (the clear() summary)
#include <vector>      // std::vector

#include "chunk/array_chunk2A.h"                        // ArrayChunk2A, the class these bodies belong to
#include "chunk/all_type_chunk.h"                       // AllTypeChunk::ensure_manager
#include "chunk/internal/chunk_size_predictor.h"        // report_size_prediction
#include "chunk/internal/chunk_stats.h"                 // increm_chunk_* (CHUNK_STATS)
#include "chunk/internal/chunk_stats_printer.h"         // print_formatted_chunk_stats
#include "chunk/internal/manager_chunk.h"               // ManagerChunk: per-size free lists, candidate filter
#include "chunk/PMP_CHUNK.h"                            // CHUNK_PMP_*, get_size_max_tab_omp
#include "chunk/singleton.h"                            // SingletonChk<AllTypeChunk>::getRef
#include "chunk/type_name.h"                            // type_name<T>
#include "chunk/internal/invariant_check.h" // CHUNK_INVARIANT_FAILED
#include "chunk/config.h"                               // build_config::chunk_dynamic, chunk_config::stats
#include "chunk/internal/chunk_size_policy.h"           // set_thread_chunk_size
#include "chunk/internal/concept_array_chunk_element.h" // ArrayChunkElement
#include "chunk/internal/parallel_storage_init.h"       // for_each_chunk
#include "chunk/internal/pool_ledger.h"                 // CHUNK_LEDGER_* (check builds)
#include "chunk/internal/diag_mutex.h"                  // diag_print_mutex (the residual report)

namespace shrink_array
{

template <ArrayChunkElement T, typename ind_type>
void ArrayChunk2A<T, ind_type>::create(ind_type sz)
{
    // The first array of this element type registers its manager; every
    // later one pays a guard load. See AllTypeChunk.
    AllTypeChunk::ensure_manager<ManagerChunk<Chunk2A, T>>();

    Base::create_common();

    // Geometry-dependent allocation only, so rechunk_to can call it on an
    // already-registered, just torn-down array WITHOUT re-running create_common
    // (which would re-register it under a fresh num_tab via pre_create_hook).
    rebuild_storage(sz);
}

template <ArrayChunkElement T, typename ind_type>
void ArrayChunk2A<T, ind_type>::rebuild_storage(ind_type sz)
{
    if (sz)
    {
        if (this->size_policy_.size())
        {
            nb_chunk = this->chunks_for(sz);
        }
        else if (sz && !this->size_policy_.size())
        {
            CHUNK_INVARIANT_FAILED("create({} elements): the chunk size is 0 (no geometry set)", sz);
        }

        if (nb_chunk)
        {
            manager_chunk2A::increm_somme_chunk_create(nb_chunk);
            manager_chunk2A::compute_size_buffer_reuse();
            manager_chunk2A::compute_size_max_number_of_chunk_to_check_compress();

            acquire_cst2_registry();
            this->acquire_chunk_list(static_cast<std::size_t>(nb_chunk));

            this->alloc_dirty_bitset(nb_chunk);

            // old=0: full (re)build of the metadata tables. Correct because every
            // entry is reinitialised just below, BEFORE any refill, so no proxy
            // holds a cached ptr_second_value into a table being reallocated.
            Proxy2A<T>::resize_tab_is_int(num_tab, 0, nb_chunk);
            this->assign_alloc_key(); // own slabs if large enough (see alloc_key_)

            // Per-chunk init: every iteration writes slot i of tables sized just
            // above and nothing else, so it parallelises as is (a serial loop is
            // a visible master-only cost on large arrays). Same pattern as
            // ArrayChunk2::rebuild_storage; serial inside a region (the caller
            // need not be the whole team) and below the one threshold,
            // parallel_storage_init::kParallelInitMinChunks.
            const auto init_chunk = [&](int i)
            {
                Proxy2A<T>::Base::set_tab_is_inside_chunk_list_to_compress(num_tab, i, 0);
                Proxy2A<T>::tab_type_cst_val[num_tab][i] = is_val1;
                Proxy2A<T>::BaseSpec::set(num_tab, i, T{});
                this->construct_chunk(i).init_after_constructor();
            };
            parallel_storage_init::for_each_chunk(nb_chunk, init_chunk);
        }
    }

    m_size = sz;
}

/// @brief Print the demangled name of T and @p nb_val2 on std::cout.
/// @param nb_val2 Number of compression values to report.
template <ArrayChunkElement T>
void print_type_name(int nb_val2)
{
    std::cout << std::left << std::setw(30) << ("Type: " + type_name<T>())
              << " → Number of compression values : " << nb_val2 << "\n";
}

template <ArrayChunkElement T, typename ind_type>
void ArrayChunk2A<T, ind_type>::print_compression_stats(std::ostream& os) const
{
    if (nb_chunk == 0)
    {
        return;
    }

    // Do not classify from tab_type_cst_val alone: it is written only at
    // construction and never reset on materialisation, so a MATERIALISED chunk
    // would be counted as compressed. written() tests the LIVE state
    // (values_load() != cst_val1 && is_not_cst_val2()); tab_type_cst_val is
    // trusted only for chunks that are NOT written.
    // A written chunk is then re-tested on its CONTENT (constant_value(), the
    // same scan as the size predictor): written() only says whether the
    // opportunistic deferred compression happened to catch the chunk, which is
    // thread-count dependent. Classifying on content makes still@exit a
    // function of the data alone, matching the sequential build's eager
    // compression.
    //
    // Each chunk is classified on its own, so the scan is shared out over the
    // team (the counts are reductions). Same guard as the other teardown loops:
    // never nest inside a caller's region, no team for a handful of chunks.
    const auto csz_audit = static_cast<unsigned int>(this->size_policy_.size());
    int        n_t0      = 0;
    int        n_t1      = 0;

#pragma omp parallel for reduction(+ : n_t0, n_t1) if (!CHUNK_PMP_IN_PARALLEL_REGION && nb_chunk > 500)
    for (int j = 0; j < nb_chunk; ++j)
    {
        const auto& ck = chunk_list[j];
        if (ck.written())
        {
            const auto v0 = ck.constant_value(csz_audit);
            if (!v0)
            {
                continue;
            }
            if (*v0 == Chunk2A<T>::first_value)
            {
                ++n_t0;
            }
            else
            {
                ++n_t1;
            }
            continue;
        }
        const char ct = Proxy2A<T>::tab_type_cst_val[num_tab][j];
        if (ct == 0)
        {
            ++n_t0;
        }
        else
        {
            ++n_t1;
        }
    }
    const int n_compressed = n_t0 + n_t1;
    if constexpr (chunk_config::stats)
    {
        add_chunk_still_compressed_at_exit<Chunk2A<T>>(n_compressed);
        record_chunk_size_at_exit<Chunk2A<T>>(static_cast<unsigned int>(this->size_policy_.size()));
    }

    // Shared layout/colours/verdict: see chunk_stats_printer.h.
    print_formatted_chunk_stats(os, num_tab, this->debug_name(), type_name<T>(), n_compressed, nb_chunk, n_t1,
                                "Chunk2A",
                                /*n_cross=*/-1, /*cross_name=*/"", /*bool_elements=*/std::is_same_v<T, bool>);
}

template <ArrayChunkElement T, typename ind_type>
void ArrayChunk2A<T, ind_type>::clear()
{
    this->invalidate_proxies();
    // Chunk-size predictor (log only, no re-chunk; any_check only): scans the
    // chunks once, logs the current vs predicted-optimal size. Runs before the
    // drain, while the chunk state is still valid. One summary slot per chunk,
    // written by index, so the scan is shared over the team (same guard as the
    // stats scan) and keeps chunk order.
    if constexpr (build_config::any_check)
    {
        if (nb_chunk > 0)
        {
            const auto                      csz = static_cast<unsigned int>(this->size_policy_.size());
            std::vector<std::pair<bool, T>> summary(static_cast<std::size_t>(nb_chunk));
#pragma omp parallel for if (!CHUNK_PMP_IN_PARALLEL_REGION && nb_chunk > 500)
            for (int j = 0; j < nb_chunk; ++j)
            {
                const auto& ck = chunk_list[j];
                if (!ck.written())
                {
                    summary[static_cast<std::size_t>(j)] = {true, ck.read_value_scan(0)};
                    continue;
                }
                const auto v0                        = ck.constant_value(csz);
                summary[static_cast<std::size_t>(j)] = {v0.has_value(), v0.value_or(T{})};
            }
            const auto& sh_inf   = *Proxy2A<T>::share_info_array;
            const int   num_tab_ = sh_inf.get_num_tab(this->registry_key());
            report_size_prediction<T>(summary, csz, num_tab_, "chunk2a",
                                      sizeof(std::remove_reference_t<decltype(chunk_list[0])>));
        }
    }

    // Everything below (drain, terminal compression, buffer release, metadata
    // reset) is the storage teardown shared with rechunk_to().
    clear_core();
}

template <ArrayChunkElement T, typename ind_type>
void ArrayChunk2A<T, ind_type>::clear_core()
{
    // Pin the ambient size to THIS array's own geometry before the sweep
    // below. Same rule as ArrayChunk2::clear_core() -- see that
    // function's comment.
    if constexpr (build_config::chunk_dynamic)
    {
        set_thread_chunk_size(static_cast<unsigned int>(this->size_policy_.size()));
    }

    // Drain any deferred write still targeting this array's chunks before
    // chunk_list is freed below. force=true (see
    // ParallelSupportCommon::flush_pending_deferred_ops): the non-force path is
    // gated on CM::any_pending_ops() and on the per-chunk dirty bit, and when
    // either misreads NOTHING is flushed -- the audit below then scans a chunk
    // whose writes are still queued, constant_value() (raw values_load()) sees a
    // non-constant buffer, and the chunk is under-counted as still-compressed.
    // 2A carries far more deferred traffic than v1/v2, so it exposes this first.
    this->flush_pending_deferred_ops(/*force=*/true);

    // Per-array compression stats: read after the flush (before it, in-flight
    // async compression moves the written() state, a thread-count-dependent
    // snapshot) and before the terminal compression/release sweep below mutates it.
    if (this->exit_census_)
    {
        print_compression_stats();
    }

    // Remove this array's chunks from the shared compression candidate list
    // before they are freed: in non-OMPT builds nothing else clears it, and a
    // later compression pass would walk the dangling Chunk2A* (use-after-free).
    // By num_tab, so other live arrays' candidates are untouched.
    manager_chunk2A::remove_from_compress_list_chunk2a_by_num_tab(num_tab);

    // Also drop any candidate whose chunk pointer lies in THIS array's
    // chunk_list span: the by-num_tab removal misses a tuple whose num_tab was
    // shifted by a rechunk swap, and a later compress pass would double-free its
    // buffer. Must run before the terminal sweep and before chunk_list.clear().
    if (!chunk_list.empty())
    {
        manager_chunk2A::remove_from_compress_list_chunk2a_by_ptr_range(
            static_cast<const void*>(chunk_list.data()),
            static_cast<const void*>(chunk_list.data() + chunk_list.size()));
    }

    if (!nb_chunk)
    {
        m_size = 0;

        chunk_list.clear();
        Proxy2A<T>::clear_tab_is_int(num_tab);

        return;
    }

    ManagerChunk<Chunk2A, T>::decrem_somme_chunk_create(nb_chunk);
    // Recompute what derives from the total -- see ArrayChunk::clear_core():
    // otherwise a re-chunk leaves the threshold sized for the geometry it just left.
    ManagerChunk<Chunk2A, T>::compute_size_buffer_reuse();
    ManagerChunk<Chunk2A, T>::compute_size_max_number_of_chunk_to_check_compress();

    if constexpr (chunk_config::not_use_compression_with_ompt)
    {
        // Skip this terminal sweep in aggressive mode (compression is left to the
        // in-loop perform_compression -> make_block_compression path): there,
        // compress()'s clear_int_st calls are compiled out, so the sweep would
        // only scan every chunk's values without compressing anything -- a cost
        // that dominates the profile.
        if constexpr (!chunk_config::have_aggressive_compression)
        {
            const auto& ens_type_chunk = SingletonChk<AllTypeChunk>::getRef();

            if (ens_type_chunk.have_chunk_to_compress_potentially())
            {
                ens_type_chunk.compress_chunk_to_compress_potentially(false);
            }
        }

        // ── Missed-compression audit ─────────────────────────────────────────────
        // Check + non-aggressive only. After the terminal pass, walk THIS array's
        // chunks (not the candidate list, so never-enqueued blocks are seen) for
        // every block (a) still materialised (written()), (b) never registered as
        // a candidate (tab_is_inside_chunk_list_to_compress == 0), yet (c)
        // constant. Two different things, told apart by the value:
        //
        //   MISSED  constant at first_value, or at the chunk's own second value
        //           (get_tab_second_value): the pass would have compressed it,
        //           the write path let it become constant without queueing it.
        //           A few is benign (a straggler), a recurring/large count means
        //           the marking path has a hole.
        //   STUCK   constant at ANOTHER value: chunk2a keeps a chunk's second
        //           value until the chunk passes through first_value (the clear
        //           to cst1 resets it), so a chunk compressed to V once and
        //           uniformly W later is not compressible as it stands -- not by
        //           the write path, not by compress(). This count is the
        //           memory that design choice costs on this array -- not a
        //           marking miss.
        //
        // Non-fatal; O(nb_chunk * chunk_size) once at teardown, check builds only.
        if constexpr (build_config::any_check && !chunk_config::have_aggressive_compression)
        {
            const auto  csz           = static_cast<unsigned int>(this->size_policy_.size());
            const auto& inside        = Proxy2A<T>::get_tab_is_inside_list_to_compress(num_tab);
            std::size_t missed_first  = 0;
            std::size_t missed_second = 0;
            std::size_t stuck         = 0;

            for (int j = 0; j < nb_chunk; ++j)
            {
                const auto& ck = chunk_list[j];
                if (!ck.written())
                {
                    continue; // already constant/compressed — nothing owed
                }
                if (inside[j].load(std::memory_order_acquire) != 0)
                {
                    continue; // legitimately queued (a straggler the pass left) — not "missed"
                }

                // Is it actually constant? (same notion fill_list_to_compress uses:
                // every element equal to the block's first element.)
                const auto v = ck.constant_value(csz);
                if (!v)
                {
                    continue;
                }
                if (*v == static_cast<T>(Chunk2A<T>::first_value))
                {
                    ++missed_first;
                }
                else if (*v == Proxy2A<T>::BaseSpec::get_tab_second_value(num_tab, j).get())
                {
                    ++missed_second;
                }
                else
                {
                    ++stuck;
                }
            }

            if (missed_first + missed_second != 0)
            {
                std::fprintf(stderr,
                             "[COMPRESS-AUDIT] chunk2a tab=%d: %zu constant block(s) never queued for "
                             "compression (%zu at first_value, %zu at their second value: missed by the "
                             "marking path). A few is benign; a recurring/large count is not.\n",
                             num_tab, missed_first + missed_second, missed_first, missed_second);
                std::fflush(stderr);
            }
            if (stuck != 0)
            {
                std::fprintf(stderr,
                             "[COMPRESS-AUDIT] chunk2a tab=%d: %zu block(s) constant at a value other than "
                             "their second value: stuck, not a marking miss (chunk2a keeps a chunk's "
                             "second value until it passes through first_value).\n",
                             num_tab, stuck);
                std::fflush(stderr);
            }
        }
    }

    if constexpr (build_config::not_use_openmp)
    {
        for (int j = 0; j < nb_chunk; ++j)
        {
            if (chunk_list[j].written())
            {
                if constexpr (chunk_config::stats)
                {
                    increm_chunk_teardown_clear<Chunk2A<T>>();
                }
                chunk_list[j].clear_st(&Proxy2A<T>::tab_type_cst_val[num_tab][j],
                                       Proxy2A<T>::BaseSpec::get_tab_second_value(num_tab, j));
            }
        }

        // Give back the registry reference this array took
        // (acquire_cst2_registry), as the OpenMP branch does after its barrier;
        // without it the last array out never frees the registry.
        if (holds_cst2_registry_)
        {
            int nb_val2 = 0;
            Chunk2A<T>::tab_cst_val2_holder::clear(nb_val2); // single thread here
            holds_cst2_registry_ = false;
            if (nb_val2 != 0)
            {
                print_type_name<T>(nb_val2);
            }
        }
    }
    else
    {
        int nb_val2 = 0;

#pragma omp parallel reduction(+ : nb_val2)
        {
            const int nb_chunk0 = get_size_max_tab_omp(nb_chunk);

            std::vector<Chunk2A<T>*> ll_chunk_loc;
            ll_chunk_loc.reserve(nb_chunk0);

#pragma omp for schedule(static) nowait
            for (int j = 0; j < nb_chunk; ++j)
            {
                if (chunk_list[j].written_test())
                {
                    if constexpr (chunk_config::stats)
                    {
                        increm_chunk_teardown_clear<Chunk2A<T>>();
                        // Released below without a compression: out of the live figures.
                        add_chunk_released<Chunk2A<T>>(1, static_cast<long>(sizeof(T)) *
                                                              static_cast<long>(this->size_policy_.size()));
                    }
                    ll_chunk_loc.push_back(&chunk_list[j]);
                }
            }

            const int thr_rank = CHUNK_PMP_THREAD_RANK;

            const int nb = ll_chunk_loc.size();

            if (nb > 0)
            {
                // Park into the reuse list of THIS array's geometry, not the ambient
                // t_active (may be another size mid-rechunk): a size-A buffer filed
                // under list[B] would be memset at B by a later pop and overflow.
                const unsigned int my_sz = this->size_policy_.size();
                auto&              lbf   = manager_chunk2A::get_list_block_free_thr_for_size(thr_rank, my_sz);

                for (int k = 0; k < nb; ++k)
                {
                    T* values_loc = ll_chunk_loc[k]->values;
                    // Ledger: LIVE -> PARKED, or a later pop of this block from the
                    // free-list is an ILLEGAL b_pop (state=LIVE, expected PARKED).
                    CHUNK_LEDGER_PARK(values_loc, "clear_core.teardown_park");
                    lbf.push_back(values_loc);

                    // Neutralise the chunk NOW that its buffer is parked: a residual
                    // compression candidate (one that slipped the removals above, or
                    // the terminal sweep) would otherwise free_block_to_pool the SAME
                    // buffer again (double-free). With cst_val1 stored, every compress
                    // path short-circuits (the CAS reads values == cst_val1 and
                    // returns nullptr). Mirrors the non-OMP clear_st path.
                    ll_chunk_loc[k]->values_store(ChunkBase<T, Chunk2A<T>>::get_cst_val1());
                }
            }

// BARRIER REQUIRED: the loop above is `omp for ... nowait`, so other threads may
// still be in it writing values, whose slow path (make_block_compression ->
// find_or_set_ind_cst_val2) SCANS the shared segment list. clear() below tears
// that list down (head.exchange(nullptr) + delete of every Segment); without
// the barrier the teardown races the scan (freed-Segment dereference).
#pragma omp barrier
#pragma omp single
            {
                // Only the reference this array took (acquire_cst2_registry).
                if (holds_cst2_registry_)
                {
                    Chunk2A<T>::tab_cst_val2_holder::clear(nb_val2);
                    holds_cst2_registry_ = false;
                }
            }
        }

        if (nb_val2 != 0)
        {
            print_type_name<T>(nb_val2);
        }
    }

    m_size = 0;

    this->release_chunk_list(); // required by the allocator (a re-chunk keeps the buffer)

    Proxy2A<T>::clear_tab_is_int(num_tab);

    nb_chunk = 0;

    // An array with its own slab affinity gives its memory back NOW: every
    // buffer parked above (own rank) or by another rank (foreign, under this
    // array's key) goes home to its allocator set, whose slabs hold nothing
    // else and so empty -- and reclaim_empty_slabs madvises them away.
    // Sequential point (destructor / rechunk_to); the drain opens its own
    // region. Key 0 (shared pool): nothing to give back per array.
    if (this->alloc_key_ != 0 && !CHUNK_PMP_IN_PARALLEL_REGION)
    {
        ManagerChunk<Chunk2A, T>::drain_reuse_list_for_size(this->size_policy_.size());
    }
}

template <ArrayChunkElement T, typename ind_type>
void ArrayChunk2A<T, ind_type>::compress_int()
{
    // Pin the ambient size to THIS array's geometry: the frees below route on
    // thread_chunk_size(), not on the buffer (see clear_core()); left to the
    // array the thread last served, a re-chunked array's buffers would go to
    // another size's allocator.
    if constexpr (build_config::chunk_dynamic)
    {
        shrink_array::set_thread_chunk_size(static_cast<unsigned int>(this->size_policy_.size()));
    }

    std::vector<int> tab_nb1;
    std::vector<int> tab_nb2;

    const int nb_res = (nb_chunk / (50 * CHUNK_PMP_MAX_THREADS)) + 1;
    tab_nb1.reserve(nb_res);
    tab_nb2.reserve(nb_res);

#pragma omp for schedule(static) nowait
    for (int j = 0; j < nb_chunk; ++j)
    {
        const auto& chunk = chunk_list[j];

        // Through SecondValueRef::get(), not the raw slot: under
        // CHUNK_SECOND_VALUE_INDEXED a slot holds the value's IDENTIFIER, and
        // `const T v = slot` would silently convert that integer into a T.
        const T second_value = Proxy2A<T>::BaseSpec::get_tab_second_value(num_tab, j).get();

        if (chunk.written())
        {
            // second_value == first_value comes back as `first` only.
            const auto scan = chunk.scan_constants(Chunk2A<T>::first_value, second_value,
                                                   static_cast<unsigned int>(this->size_policy_.size()));
            if (scan.first)
            {
                tab_nb1.push_back(j);
            }
            if (scan.second)
            {
                tab_nb2.push_back(j);
            }
        }
    }

    // ── What this reports, and what it does NOT assert ───────────────────────
    // compress_int() only SCANS; it never compresses (unlike ArrayChunk's, whose
    // non-embedded branch does the clear itself). It reports the residual:
    // blocks constant right now and still materialised.
    //
    // Do not treat that residual as an invariant violation: compression is
    // incremental (a write only QUEUES its chunk; make_block_compression / the
    // phase-end pass drains the queue), so a block that just became constant
    // normally waits for the next pass.
    //
    // What matters is its SIZE and TREND: a handful is the normal in-flight
    // residual; a large or growing count means the incremental path is not
    // keeping up (or, in a use_openmp && !OMPT build, that Chunk2A::compress()
    // returns early by design and make_block_compression is the only
    // compressor left -- see its guard in chunk2A_std_inline.h).
    const auto nb1 = static_cast<int>(tab_nb1.size());

    if (nb1 != 0)
    {
        {
            const std::scoped_lock diag_lock(::shrink_array::internal::diag_print_mutex());
            std::cerr << "[compress-residual chunk2a] " << nb1
                      << " block(s) constant at first_value, still materialised "
                         "(awaiting the next compression pass)\n";
            std::cerr << "ArrayChunk2A type: ArrayChunk2A<" << type_name<T>() << ", " << type_name<ind_type>() << ">"
                      << std::endl;

            // Cap the dump at 10 ENTRIES (not at chunk index 10: an array whose
            // residual starts past chunk 10 would print a single line).
            int shown = 0;
            for (const int ind : tab_nb1)
            {
                std::cerr << ind << " " << static_cast<ind_type>(ind) * static_cast<ind_type>(this->size_policy_.size())
                          << "\n";
                if (++shown >= 10)
                {
                    break;
                }
            }
            if (nb1 > shown)
            {
                std::cerr << "\n...\n";
            }
        }
    }

    const auto nb2 = static_cast<int>(tab_nb2.size());

    if (nb2 != 0)
    {
        {
            const std::scoped_lock diag_lock(::shrink_array::internal::diag_print_mutex());
            std::cerr << "[compress-residual chunk2a] " << nb2
                      << " block(s) constant at the second value, still materialised "
                         "(awaiting the next compression pass)\n";
            std::cerr << "ArrayChunk2A type: ArrayChunk2A<" << type_name<T>() << ", " << type_name<ind_type>() << ">"
                      << std::endl;

            int shown = 0;
            for (const int ind : tab_nb2)
            {
                std::cerr << ind << " " << static_cast<ind_type>(ind) * static_cast<ind_type>(this->size_policy_.size())
                          << "\n";
                if (++shown >= 10)
                {
                    break;
                }
            }
            if (nb2 > shown)
            {
                std::cerr << "\n...\n";
            }
        }
    }
}

template <ArrayChunkElement T, typename ind_type>
void ArrayChunk2A<T, ind_type>::make_block_compression()
{
    // Every rank's list, drained serially by the caller (no region opened
    // here, no writer handshake: nullptr proxy).
    for (int thr = 0; thr < CHUNK_PMP_RANK_BOUND; ++thr)
    {
        Chunk2A<T>::make_block_compression(thr, nullptr);
    }
}

template <ArrayChunkElement T, typename ind_type>
void ArrayChunk2A<T, ind_type>::release_chunk_range(int first, int last)
{
    if (first >= last)
    {
        return;
    }
    // The release routes by the ambient geometry: pin it to this array's.
    if constexpr (build_config::chunk_dynamic)
    {
        shrink_array::set_thread_chunk_size(static_cast<unsigned int>(this->size_policy_.size()));
    }

    if constexpr (build_config::not_use_openmp)
    {
        // Before Proxy2A::resize_tab_is_int() truncates the per-chunk type
        // table: clear_st() reads tab_type_cst_val[num_tab][j].
        const int num_tab = Proxy2A<T>::share_info_array->get_num_tab(this->registry_key());
        for (int j = first; j < last; ++j)
        {
            if (chunk_list[j].written())
            {
                if constexpr (chunk_config::stats)
                {
                    increm_chunk_teardown_clear<Chunk2A<T>>();
                }
                chunk_list[j].clear_st(&Proxy2A<T>::tab_type_cst_val[num_tab][j],
                                       Proxy2A<T>::BaseSpec::get_tab_second_value(num_tab, j));
            }
        }
    }
    else
    {
        // A large shrink takes a team of its own, as clear_core()'s release
        // does: each thread parks into ITS rank's reuse list of this array's
        // geometry. Inside a region, or for a few chunks, the calling thread.
        const unsigned int my_sz = this->size_policy_.size();
        parallel_storage_init::for_each_chunk_in(
            first, last,
            [my_sz]() -> auto& {
                return manager_chunk2A::get_list_block_free_thr_for_size(CHUNK_PMP_THREAD_RANK, my_sz);
            },
            [this, my_sz](int j, auto& lbf)
            {
                if (chunk_list[j].written_test())
                {
                    if constexpr (chunk_config::stats)
                    {
                        increm_chunk_teardown_clear<Chunk2A<T>>();
                        add_chunk_released<Chunk2A<T>>(1, static_cast<long>(sizeof(T)) * static_cast<long>(my_sz));
                    }
                    T* values_loc = chunk_list[j].values;
                    CHUNK_LEDGER_PARK(values_loc, "resize.shrink_park");
                    lbf.push_back(values_loc);
                    // Neutralise, as clear_core(): the buffer belongs to the
                    // reuse list now.
                    chunk_list[j].values_store(ChunkBase<T, Chunk2A<T>>::get_cst_val1());
                }
            });
    }
}

template <ArrayChunkElement T, typename ind_type>
void ArrayChunk2A<T, ind_type>::resize(ind_type newsz)
{
    this->invalidate_proxies();
    const auto& sh_inf = *Proxy2A<T>::share_info_array;

    const int      num_tab      = sh_inf.get_num_tab(this->registry_key());
    const int      old_nb_chunk = nb_chunk;
    const ind_type old_size     = this->m_size; // the stale-tail reset at the end

    Base2::resize_common(newsz);
    if (nb_chunk > 0)
    {
        acquire_cst2_registry(); // an array built empty gets its chunks here
    }

    Proxy2A<T>::resize_tab_is_int(num_tab, old_nb_chunk, nb_chunk);

    // The new chunks [old_nb_chunk, nb_chunk) and their table slots: a team of
    // its own outside a region, the calling thread inside one. Never an
    // orphaned `omp for`: inside a region it would construct only 1/N of a
    // thread's own array. Each iteration writes slot i of tables sized above.
    parallel_storage_init::for_each_chunk(nb_chunk - old_nb_chunk,
                                          [this, old_nb_chunk, num_tab](int k)
                                          {
                                              const int i = old_nb_chunk + k;
                                              this->construct_chunk(i).init_after_constructor();
                                              Proxy2A<T>::Base::set_tab_is_inside_chunk_list_to_compress(num_tab, i, 0);
                                              Proxy2A<T>::tab_type_cst_val[num_tab][i] = is_val1;
                                              // T{}: the struct elements have no conversion from 0
                                              Proxy2A<T>::BaseSpec::set(num_tab, i, T{});
                                          });

    // std::vector semantics for the regrown range [old_size, newsz): it reads
    // T{}. The chunks added above are fresh, but the one holding the OLD end
    // keeps its tail (after a shrink, the old values would come back on
    // growth). Reset that tail through the normal write path, after the
    // reallocations. Serial: at most one chunk, and inside a region an
    // `omp for` would cover 1/N of it.
    if (newsz > old_size)
    {
        const auto csz = static_cast<ind_type>(this->chunk_size());
        // Up to the end of the OLD chunk count: chunks_for() keeps a chunk past
        // the last element, so when old_size falls on a chunk boundary that
        // spare chunk holds stale values too.
        // In long long: old_nb_chunk * csz may pass ind_type's range near its top.
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
