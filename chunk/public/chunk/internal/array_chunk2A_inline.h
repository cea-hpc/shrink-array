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
#include <cstdlib>     // std::getenv, std::abort
#include <iomanip>     // std::setw (print_type_name)
#include <iostream>    // std::cout / std::cerr
#include <string>      // the type name in print_type_name
#include <type_traits> // std::remove_reference_t
#include <utility>     // std::pair (the clear() summary)
#include <vector>

#include "chunk/array_chunk2A.h"
#include "chunk/all_type_chunk.h"
#include "chunk/internal/chunk_size_predictor.h"
#include "chunk/internal/chunk_stats.h"         // increm_chunk_* (CHUNK_STATS)
#include "chunk/internal/chunk_stats_printer.h" // print_formatted_chunk_stats
#include "chunk/internal/manager_chunk.h"
#include "chunk/PMP_CHUNK.h" // CHUNK_PMP_*, get_size_max_tab_omp
#include "chunk/singleton.h"
#include "chunk/type_name.h"
#include "chunk/config.h"
#include "chunk/internal/chunk_size_policy.h"
#include "chunk/internal/concept_array_chunk_element.h"
#include "chunk/internal/in_t.h"
#include "chunk/internal/pool_ledger.h" // CHUNK_LEDGER_* (check builds)

namespace shrink_array
{

template <typename T>
class SingletonChk;

template <ArrayChunkElement T, typename ind_type>
void ArrayChunk2A<T, ind_type>::create(ind_type sz)
{
    // The first array of this element type registers its manager; every
    // later one pays a guard load. See AllTypeChunk.
    AllTypeChunk::ensure_manager<ManagerChunk<Chunk2A, T>>();

    Base::create_common();

    // Storage allocation at the current active geometry. Extracted so rechunk_to
    // can reuse it WITHOUT re-running create_common (which would re-register the
    // array under a fresh num_tab via pre_create_hook). create() keeps its
    // identity setup here; rebuild_storage does only the geometry-dependent
    // allocation, so it is correct to call on an already-registered array that
    // has just been torn down (clear_core) and is switching geometry.
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
            std::cerr << "__chunk_size null !\n";
            std::abort();
        }

        if (nb_chunk)
        {
            manager_chunk2A::increm_somme_chunk_create(nb_chunk);
            manager_chunk2A::compute_size_buffer_reuse();
            manager_chunk2A::compute_size_max_number_of_chunk_to_check_compress();

            acquire_cst2_registry();
            chunk_list.resize(nb_chunk);

            this->alloc_dirty_bitset(nb_chunk);

            // old=0: full (re)build of the metadata tables at the new geometry.
            // Correct here because every entry is reinitialised just below, and
            // it runs BEFORE any refill, so no proxy holds a cached
            // ptr_second_value into a table we are about to reallocate.
            Proxy2A<T>::resize_tab_is_int(num_tab, 0, nb_chunk);
            this->assign_alloc_key(); // own slabs if large enough (see alloc_key_)

            // Per-chunk init: every iteration writes slot i of tables sized just
            // above and nothing else, so it parallelises as is. Was serial: ~1 s
            // of master-only time per run of shrink_bench_chunk2a_dyn at 64
            // threads (creation + the new array of each rechunk). Same pattern as
            // ArrayChunk2::rebuild_storage; serial inside a region (the caller
            // need not be the whole team) and for small arrays.
            const auto init_chunk = [&](int i)
            {
                Proxy2A<T>::Base::set_tab_is_inside_chunk_list_to_compress(num_tab, i, 0);
                Proxy2A<T>::tab_type_cst_val[num_tab][i] = is_val1;
                Proxy2A<T>::BaseSpec::set(num_tab, i, T{});
                chunk_list[i].init_after_constructor();
            };
            const bool parallel_chunk_init = !CHUNK_PMP_IN_PARALLEL_REGION && nb_chunk > 16384;
            if (parallel_chunk_init)
            {
#pragma omp parallel for schedule(static)
                for (int i = 0; i < nb_chunk; ++i)
                    init_chunk(i);
            }
            else
            {
                for (int i = 0; i < nb_chunk; ++i)
                    init_chunk(i);
            }
        }
    }

    m_size = sz;
}

/// @brief Print the demangled name of T and @p nb_val2 on std::cout.
/// @param nb_val2 Number of compression values to report.
template <ArrayChunkElement T>
void print_type_name(int nb_val2)
{
    // type_name<T>() already demangles; this used to carry its own copy of
    // the abi::__cxa_demangle dance.
    std::cout << std::left << std::setw(30) << ("Type: " + type_name<T>())
              << " → Number of compression values : " << nb_val2 << "\n";
}

template <ArrayChunkElement T, typename ind_type>
void ArrayChunk2A<T, ind_type>::print_compression_stats(std::ostream& os) const
{
    if (nb_chunk == 0)
        return;

    // Chunk2A-specific classification. FIXED: tab_type_cst_val[num_tab][j] is
    // ONLY ever written at construction (array_chunk2A_inline.h) -- it records
    // which constant a chunk should use WHEN compressed, but materialisation
    // never resets it. Reading it alone therefore classified every chunk as
    // "compressed to 0 or 1" even when it was currently MATERIALISED (stale
    // metadata from construction or an earlier compression), making
    // n_compressed silently equal nb_chunk for every run -- caught the same way
    // as the chunk2 bug: still@exit + teardown_clear failed to partition the
    // known total chunk count across sequential/OpenMP runs.
    // written() tests the LIVE state (values_load() != cst_val1 &&
    // is_not_cst_val2()), same meaning as plain Chunk::written(); only among
    // the chunks that are NOT written do we trust tab_type_cst_val to say
    // which constant they hold.
    // Final content check, added ON TOP of the written() classification below.
    // written() only reflects whether the OPPORTUNISTIC deferred-compression
    // mechanism happened to catch a chunk during the run -- not whether its
    // CURRENT content is actually compressible. With identical final data
    // (verified: long_val_sum matches exactly across sequential/OMP-8/OMP-16
    // for the same seed), the fraction of chunks the live drain caught before
    // teardown is thread-count dependent even though the true content is not
    // -- so still@exit disagreed by ~10% purely from this timing gap, not
    // from any data difference. Re-testing content here (same scan the
    // size-predictor above already performs) makes the classification a
    // function of content alone, matching the sequential build's eager
    // compression -- deterministic and thread-count-independent.
    //
    // Each chunk is classified on its own, so the scan is shared out over the
    // team when one is available (the counts are reductions). Same guard as
    // the other teardown loops: never nest inside a caller's region, and not
    // worth a team for a handful of chunks.
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
                ++n_t0;
            else
                ++n_t1;
            continue;
        }
        const char ct = Proxy2A<T>::tab_type_cst_val[num_tab][j];
        if (ct == 0)
            ++n_t0;
        else
            ++n_t1;
    }
    const int n_compressed = n_t0 + n_t1;
    if constexpr (chunk_config::stats)
    {
        add_chunk_still_compressed_at_exit<Chunk2A<T>>(n_compressed);
        record_chunk_size_at_exit<Chunk2A<T>>(static_cast<unsigned int>(this->size_policy_.size()));
    }

    // Centralized rendering: shared layout/colors/verdict with ArrayChunk2
    // (see chunk_stats_printer.h).
    print_formatted_chunk_stats(os, num_tab, type_name<T>(), n_compressed, nb_chunk, n_t1, "Chunk2A",
                                /*n_cross=*/-1, /*cross_name=*/"", /*bool_elements=*/std::is_same_v<T, bool>);
}

template <ArrayChunkElement T, typename ind_type>
void ArrayChunk2A<T, ind_type>::clear()
{
    // Chunk-size predictor (observation phase, log only — no re-chunk). Scans
    // this array's chunks once, predicts memory at larger sizes, logs the
    // current vs predicted-optimal size. Same pre-drain point as the stats, for
    // the same reason (chunk state still valid). any_check only.
    //
    // One summary slot per chunk, written by index, so the content scan is
    // shared out over the team like the stats scan (same guard: not nested in
    // a caller's region, not for a handful of chunks) and the summary keeps
    // its chunk order.
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
            const auto& sh_inf   = Proxy2A<T>::share_info_array;
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
    // below. Same fix and rationale as ArrayChunk2::clear_core() -- see that
    // function's comment.
    if constexpr (build_config::chunk_dynamic)
        set_thread_chunk_size(static_cast<unsigned int>(this->size_policy_.size()));

    // Drain any deferred write still targeting this array's chunks before
    // chunk_list is freed below (inherited from ArrayChunk2Base).
    // force=true, for the reason array_chunk_parallel_common_inline.h states
    // about rechunk_to(): the non-force path is gated on CM::any_pending_ops(),
    // "the counter has been seen to read 0 while ops were still queued, so it
    // cannot gate this path" -- and on the per-chunk dirty bit, a second live
    // flag. When either misreads, NOTHING is flushed and the audit below scans
    // a chunk whose writes are still queued: constant_value() goes through
    // values_load(), the raw snapshot, sees a non-constant buffer and returns
    // nullopt, and the chunk is not counted as still-compressed.
    //
    // That is the residual seq/OMP drift on 2A, after the ~10% one the comment
    // below already fixed: still@exit always BELOW sequential, never above, by
    // a handful of chunks (aarch64 gnu-stats, seed pinned, long_val_sum equal:
    // 2A@256 -5 at 8 threads; 2A@dyn -18 / -27 / -4 at 8 / 16 / 64), varying
    // non-monotonically with the thread count and absent from the sequential
    // build -- where use_fast_block_compression is false and this whole
    // function is a no-op. v1 and v2 have the same gate and do not show it;
    // 2A is the variant with the deferred traffic to expose it (~73M mark
    // events against plain Chunk's ~3M, see below).
    this->flush_pending_deferred_ops(/*force=*/true);

    // Per-array compression stats. Moved here (after the flush, before the
    // rest of the teardown drain below) because the OLD placement -- before
    // flush_pending_deferred_ops -- read written()/compressed state while a
    // large volume of async compression work could still be in flight for
    // this variant (mark ~73M events vs plain Chunk's ~3M), making the
    // snapshot noisy and thread-count-dependent (measured: sequential vs
    // OMP-16t disagreed on still-compressed-at-exit by ~10%, growing with
    // thread count -- a timing artifact, not data corruption). written() is a
    // live state test (unlike the old tab_type_cst_val read), so it is safe to
    // read here, past the flush but before the terminal compression/release
    // sweep further down still mutates state.
    print_compression_stats();

    // Remove this array's chunks from the shared compression candidate list
    // before they are freed. In non-OMPT builds nothing else clears that list,
    // so a destroyed array's Chunk2A* would dangle there and a later array's
    // compression pass would walk them (heap-use-after-free). Filter by this
    // array's num_tab so other live arrays' candidates are untouched.
    manager_chunk2A::remove_from_compress_list_chunk2a_by_num_tab(num_tab);

    // Belt-and-suspenders (num_tab-agnostic): also drop any candidate whose chunk
    // pointer lies in THIS array's chunk_list span. The by-num_tab removal above
    // can miss a tuple whose num_tab was shifted by a rechunk swap; left behind it
    // dangles once chunk_list is destroyed and a later compress pass double-frees
    // its buffer. Done here (before the terminal sweep below and before
    // chunk_list.clear()) so no dangling Chunk2A* of this array can be compressed.
    if (!chunk_list.empty())
        manager_chunk2A::remove_from_compress_list_chunk2a_by_ptr_range(
            static_cast<const void*>(chunk_list.data()),
            static_cast<const void*>(chunk_list.data() + chunk_list.size()));

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
        // in-loop perform_compression -> make_block_compression path). In the GNU
        // build this sweep is also pure overhead regardless of mode: compress()'s
        // clear_int_st calls are compiled out here, so compress_chunk_to_compress
        // only scans every chunk's values without compressing or clearing the
        // potentially-list — which is why it dominates the profile. Gating it on
        // !have_agressive_compression removes that waste in aggressive builds.
        if constexpr (!chunk_config::have_agressive_compression)
        {
            auto& ens_type_chunk = SingletonChk<AllTypeChunk>::getRef();

            if (ens_type_chunk.have_chunk_to_compress_potentially())
            {
                ens_type_chunk.compress_chunk_to_compress_potentially(false);
            }
        }

        // ── Missed-compression audit (option 2) ──────────────────────────────────
        // debug/check + non-aggressive only. After the terminal compression pass
        // above, walk THIS array's chunks (not the candidate list) and look at
        // every block that is (a) still materialised (written()), (b) never
        // registered as a compression candidate (tab_is_inside_chunk_list_to_
        // compress == 0), yet (c) actually constant (all elements equal). Two
        // very different things hide behind that, told apart by the value:
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
        //           the write path, not by compress(). Probed Sept '26: 100
        //           chunks at 1.0, then all 3.0 -> 100 materialised, then all 0
        //           then all 3.0 -> 0. This count is the memory that design
        //           choice costs on this array; it used to be reported as a
        //           marking miss, which sent the search the wrong way.
        //
        // Scans the chunks directly rather than the candidate list, so it sees
        // blocks that were never enqueued (which the empty-list check
        // structurally cannot). Non-fatal. Runs once, at array teardown -- the
        // O(nb_chunk * chunk_size) cost is check-only and compiled out in
        // production (any_check false).
        if constexpr (build_config::any_check && !chunk_config::have_agressive_compression)
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
                    continue; // already constant/compressed — nothing owed
                if (inside[j].load(std::memory_order_acquire) != 0)
                    continue; // legitimately queued (a straggler the pass left) — not "missed"

                // Is it actually constant? (same notion fill_list_to_compress uses:
                // every element equal to the block's first element.)
                const auto v = ck.constant_value(csz);
                if (!v)
                    continue;
                if (*v == static_cast<T>(Chunk2A<T>::first_value))
                    ++missed_first;
                else if (*v == Proxy2A<T>::BaseSpec::get_tab_second_value(num_tab, j).get())
                    ++missed_second;
                else
                    ++stuck;
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
                    increm_chunk_teardown_clear<Chunk2A<T>>();
                chunk_list[j].clear_st(&Proxy2A<T>::tab_type_cst_val[num_tab][j],
                                       Proxy2A<T>::BaseSpec::get_tab_second_value(num_tab, j));
            }
        }

        // The registry reference this array took (acquire_cst2_registry), as
        // the OpenMP branch gives it back after its barrier. It was missing
        // here: in a serial build the user count only grew, and the last
        // array out never freed the registry (cst2_registry_validation).
        if (holds_cst2_registry_)
        {
            int nb_val2 = 0;
            Chunk2A<T>::tab_cst_val2_holder::clear(nb_val2); // single thread here
            holds_cst2_registry_ = false;
            if (nb_val2)
                print_type_name<T>(nb_val2);
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

            if (nb)
            {
                // Park into the reuse list of THIS array's geometry, not the ambient
                // t_active: these buffers are physically sized for size_policy_.size(),
                // and t_active may already be another size (e.g. mid-rechunk). Routing
                // by ambient size would file a size-A buffer under list[B]; a later
                // size-B pop would then memset it at B and overflow it.
                const unsigned int my_sz = this->size_policy_.size();
                auto&              lbf   = manager_chunk2A::get_list_block_free_thr_for_size(thr_rank, my_sz);

                for (int k = 0; k < nb; ++k)
                {
                    T* values_loc = ll_chunk_loc[k]->values;
                    // Ledger: LIVE -> PARKED. This teardown pushes the block straight
                    // into the reuse free-list; without notifying the ledger the block
                    // stays recorded as LIVE, and a later array that pops it from the
                    // free-list trips an ILLEGAL b_pop (state=LIVE, expected PARKED).
                    CHUNK_LEDGER_PARK(values_loc, "clear_core.teardown_park");
                    lbf.push_back(values_loc);

                    // Neutralise the chunk NOW that its buffer is parked. Without this
                    // the chunk stays materialised (values != cst_val1), so a residual
                    // compression candidate -- one that slipped remove_from_compress_
                    // list_chunk2a_by_num_tab above, or the terminal sweep -- would
                    // try_mark_cst_and_get_old and free_block_to_pool the SAME buffer a
                    // second time (double-free / corrupted double-linked list under
                    // std-alloc, where free_block_to_pool svfrees immediately). Storing
                    // cst_val1 makes written()/is_not_cst_val1() report the block as
                    // already compressed, so every compress path short-circuits (the
                    // CAS reads values == cst_val1 and returns nullptr). Mirrors the
                    // non-OMP clear_st path, which sets cst as it frees.
                    ll_chunk_loc[k]->values_store(ChunkBase<T, Chunk2A<T>>::get_cst_val1());
                }
            }

// BARRIER REQUIRED: the loop above is `#pragma omp for schedule(static) nowait`, so a
// thread can reach this point while other threads are still inside that
// loop writing values — whose slow path (make_block_compression ->
// find_or_set_ind_cst_val2) SCANS the shared segment list. clear()
// below tears that list down (head.exchange(nullptr) + delete of every
// Segment). Without this barrier the single-thread teardown races the
// concurrent scan -> freed-Segment dereference -> SIGSEGV (only at large
// CHUNK_SIZE, where the compression slow path fires often enough to
// overlap). The barrier guarantees every scan has finished before the
// list is destroyed.
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

        if (nb_val2)
        {
            print_type_name<T>(nb_val2);
        }
    }

    m_size = 0;

    chunk_list.clear(); // Mandatory with the new allocator

    Proxy2A<T>::clear_tab_is_int(num_tab);

    nb_chunk = 0;

    // An array with its own slab affinity gives its memory back NOW: every
    // buffer parked above (own rank) or by another rank (foreign, under this
    // array's key) goes home to its allocator set, whose slabs hold nothing
    // else and so empty -- and CHUNK_RECLAIM_EMPTY_SLABS madvises them away.
    // Sequential point (destructor / rechunk_to); the drain opens its own
    // region. Key 0 (shared pool) keeps the old behaviour: nothing to give
    // back per array.
    if (this->alloc_key_ != 0 && !CHUNK_PMP_IN_PARALLEL_REGION)
        ManagerChunk<Chunk2A, T>::drain_reuse_list_for_size(this->size_policy_.size());
}

template <ArrayChunkElement T, typename ind_type>
void ArrayChunk2A<T, ind_type>::compress_int()
{
    // Pin the ambient size to THIS array's geometry: the frees below route on
    // thread_chunk_size(), not on the buffer (see clear_core()). Left to
    // whichever array the thread last served, a re-chunked array's buffers
    // went to another size's allocator (sl_remove assertion, fast-alloc only).
    if constexpr (build_config::chunk_dynamic)
        shrink_array::set_thread_chunk_size(static_cast<unsigned int>(this->size_policy_.size()));

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
        // `const T v = slot` converted that integer into a T in silence.
        const T second_value = Proxy2A<T>::BaseSpec::get_tab_second_value(num_tab, j).get();

        if (chunk.written())
        {
            // second_value == first_value comes back as `first` only, so the
            // two former branches (first-only scan, two-hypothesis scan) are one.
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
    // non-embedded branch does the clear itself). What it finds is the residual:
    // blocks that are constant right now and still materialised.
    //
    // That residual is NOT an invariant violation, which is what the previous
    // wording ("Should not have compress (0) values") claimed. Compression is
    // incremental: a write only QUEUES its chunk, and the queue is drained by
    // make_block_compression / the phase-end pass. Between two passes it is
    // perfectly normal for a block to have just become constant and to be
    // waiting for the next one -- traced on a chunk that goes non-zero at one
    // round and zero at the next: it is simply not in the list until the write
    // that makes it constant, and the following pass does take it.
    //
    // What the number IS good for is its SIZE and its TREND: a handful is the
    // normal in-flight residual, while a large or growing count means the
    // incremental path is not keeping up (or, in a use_openmp && !OMPT build,
    // that Chunk2A::compress() returns early by design and make_block_compression
    // is the only compressor left -- see its guard in chunk2A_std_inline.h).
    const auto nb1 = static_cast<int>(tab_nb1.size());

    if (nb1 != 0)
    {
#pragma omp critical
        {
            std::cerr << "[compress-residual chunk2a] " << nb1
                      << " block(s) constant at first_value, still materialised "
                         "(awaiting the next compression pass)\n";
            std::cerr << "ArrayChunk2A type: ArrayChunk2A<" << type_name<T>() << ", " << type_name<ind_type>() << ">"
                      << std::endl;

            // Cap the dump at 10 ENTRIES. This used to break on `ind > 10`, the
            // chunk INDEX, so any array whose residual starts past chunk 10
            // printed exactly one line -- the listing was unusable on the arrays
            // that most needed it.
            int shown = 0;
            for (const int ind : tab_nb1)
            {
                std::cerr << ind << " " << static_cast<ind_type>(ind) * static_cast<ind_type>(this->size_policy_.size())
                          << "\n";
                if (++shown >= 10)
                    break;
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
#pragma omp critical
        {
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
                    break;
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
    //     if (!CHUNK_PMP_IN_PARALLEL_REGION)
    //     {
    // #pragma omp parallel
    //         Chunk2A<T>::make_block_compression(CHUNK_PMP_THREAD_RANK, nullptr);
    //     }
    //     else
    //     {
    // we can do in // because tab_m_ptr_cst_val2 is not by thread
    for (int thr = 0; thr < CHUNK_PMP_RANK_BOUND; ++thr)
    {
        Chunk2A<T>::make_block_compression(thr, nullptr);
    }
}

template <ArrayChunkElement T, typename ind_type>
void ArrayChunk2A<T, ind_type>::resize(ind_type newsz)
{
    const auto& sh_inf = Proxy2A<T>::share_info_array;

    const int num_tab      = sh_inf.get_num_tab(this->registry_key());
    const int old_nb_chunk = nb_chunk;

    Base2::resize_common(newsz);
    if (nb_chunk > 0)
        acquire_cst2_registry(); // an array built empty gets its chunks here

    Proxy2A<T>::resize_tab_is_int(num_tab, old_nb_chunk, nb_chunk);

    if (!CHUNK_PMP_IN_PARALLEL_REGION && nb_chunk > 500)
    {
#pragma omp parallel for
        for (int i = old_nb_chunk; i < nb_chunk; ++i)
        {
            chunk_list[i].init_after_constructor();
        }
    }
    else
    {
#pragma omp for
        for (int i = old_nb_chunk; i < nb_chunk; ++i)
        {
            chunk_list[i].init_after_constructor();
        }
    }

    //#pragma omp parallel for if (!CHUNK_PMP_IN_PARALLEL_REGION && nb_chunk > 500)
    for (int i = old_nb_chunk; i < nb_chunk; ++i)
    {
        Proxy2A<T>::Base::set_tab_is_inside_chunk_list_to_compress(num_tab, i, 0);
        Proxy2A<T>::tab_type_cst_val[num_tab][i] = is_val1;
        Proxy2A<T>::BaseSpec::set(num_tab, i, T{}); // T{}: the struct elements have no conversion from 0
    }
    //     else
    //     {
    // #pragma omp for
    //         for (int i = old_nb_chunk; i < nb_chunk; ++i)
    //         {
    //             Proxy2A<T>::Base::set_tab_is_inside_chunk_list_to_compress(num_tab, i, 0);
    //             Proxy2A<T>::tab_type_cst_val[num_tab][i] = is_val1;
    //             Proxy2A<T>::BaseSpec::set(num_tab, i, 0);
    //         }
    //     }
}

} // namespace shrink_array
