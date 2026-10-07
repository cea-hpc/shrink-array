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

#include <algorithm> // std::ranges::find (the healed-chunk report), std::min (the birth size of bits)
#include <atomic>    // std::memory_order
#include <cstddef>   // std::size_t
#include <cstdio>    // stderr
#include <cstdlib>   // std::abort
#include <ostream>   // print_compression_stats(std::ostream&)
#include <print>     // std::println
#include <string>    // the healed-chunk report
#include <vector>    // std::vector

#include <sys/mman.h> // mprotect (use_mm_protect: the read-only cst_val2 blocks)
#include <unistd.h>   // sysconf(_SC_PAGESIZE) (same)

#include "chunk/internal/in_t.h"                        // in_t<T>: the element snapshots
#include "chunk/array_chunk2.h"                         // ArrayChunk2, the class these bodies belong to
#include "chunk/type_name.h"                            // type_name<T>
#include "chunk/singleton.h"                            // SingletonChk<AllTypeChunk>::getRef
#include "chunk/internal/proxy2.h"                      // Proxy2<T>
#include "chunk/internal/chunk_stats.h"                 // increm_chunk_teardown_clear, add_chunk_released
#include "chunk/all_type_chunk.h"                       // AllTypeChunk::ensure_manager
#include "chunk/config.h"                               // chunk_config::use_fast_block_compression, chunk_config::stats
#include "chunk/PMP_CHUNK.h"                            // CHUNK_PMP_*
#include "chunk/internal/concept_array_chunk_element.h" // ArrayChunkElement
// Explicit includes for symbols used by clear_core() below: ManagerChunk
// (per-size free-list + candidate-filter helpers), pool_ledger
// (CHUNK_LEDGER_PARK/DISCARD), chunk_size_policy (runtime geometry). All
// pragma-once; reachable transitively today, made explicit here so this TU
// never depends on the include chain staying that way.
#include "chunk/internal/manager_chunk.h"     // ManagerChunk: per-size free lists, candidate filter
#include "chunk/internal/pool_ledger.h"       // CHUNK_LEDGER_PARK
#include "chunk/internal/chunk_size_policy.h" // set_thread_chunk_size, cst_buffer_size

/**
 * @file array_chunk2_inline.h
 * @brief Out-of-line definitions of ArrayChunk2.
 *
 * ArrayChunk2 is the array of Chunk2 blocks: every chunk compresses either to
 * the global `first_value` or to ONE array-wide `second_value` — the difference
 * with Chunk2A, which lets each chunk pick its own. That single shared constant
 * is what shapes most of this file.
 *
 * What lives here:
 *  - **lifecycle**: create() registers the array then delegates the allocation
 *    to rebuild_storage(), which rechunk_to() reuses on an already-registered
 *    array; resize() grows it; clear() / clear_core() tear it down;
 *  - **the shared second constant**: set_second_value(), and the registry of
 *    read-only cst_val2 blocks it maintains across arrays;
 *  - **auditing**: compress_int() / compress(), which look for constant chunks
 *    the incremental compression missed, and print_compression_stats().
 *
 * Two invariants recur across these bodies, both learned from crashes and
 * recorded in the inline comments next to the code that enforces them:
 *
 *  - **Drain before restructuring.** The compression-candidate lists and the
 *    deferred-op maps hold raw `&chunk_list[j]` pointers. Any operation that
 *    reallocates or frees the chunk list must flush them first, or leave
 *    dangling pointers a later pass will walk.
 *  - **Pin the geometry, do not inherit it.** Under dynamic geometry the
 *    allocator routes purely on the ambient per-thread chunk size, which may
 *    have been left by an operation on a DIFFERENT array. Every path that frees
 *    or parks buffers sets it from `size_policy_.size()` first; getting this
 *    wrong frees through the wrong size pool.
 *
 * @see array_chunk2.h for the declarations and each member's contract,
 *      array_chunk2_base.h for the parts shared with ArrayChunk2A.
 */

namespace shrink_array
{

// Identity setup only: claim a num_tab in the shared registry, size the
// per-tab tables to match, then hand the geometry-dependent work to
// rebuild_storage(). Everything here must happen exactly once per array, which
// is precisely what rechunk_to() must NOT redo.
template <ArrayChunkElement T, typename ind_type>
void ArrayChunk2<T, ind_type>::create(ind_type sz)
{
    // The first array of this element type registers its manager; every
    // later one pays a guard load. See AllTypeChunk.
    AllTypeChunk::ensure_manager<ManagerChunk<Chunk2, T>>();

    chunk_config::CheckNaN<chunk_config::do_check>::update(true);

    auto& sh_inf = Proxy2<T>::share_info_array;

    sh_inf.insert_only_num_tab(this->registry_key(), nb_tab);
    ++nb_tab;

    // Registry entry written: refresh the mirror the write proxies are armed
    // with (see ArrayChunk2::inf_array).
    sync_registry_mirror();

    if constexpr (chunk_config::use_fast_block_compression)
    {
        Proxy2<T>::resize_tab_is_int(nb_tab);
        register_no_second_value_key();
    }

    // Packed bits are born with larger chunks (packed_bit_birth_factor), except
    // rechunk_to()'s temporary, whose refill writes at the size it asked for.
    // set_chunk_size() registers the size's allocators, but also makes it the
    // process-wide and this thread's active size: both are put back, or every
    // array created after this one would be born at it.
    if constexpr (build_config::chunk_dynamic && is_packed_bit_v<T>)
    {
        const unsigned int birth =
            std::min(this->size_policy_.size() * packed_bit_birth_factor, shrink_array::cst_buffer_size);
        if (!t_birth_at_active_size && birth != this->size_policy_.size())
        {
            const unsigned int global = g_active_chunk_size;
            const unsigned int thread = t_active_chunk_size;
            this->set_chunk_size(birth);
            g_active_chunk_size = global;
            set_thread_chunk_size(thread);
        }
    }

    // Geometry-dependent allocation is factored into rebuild_storage() so that
    // rechunk_to() can reuse it WITHOUT re-registering the array (create() keeps
    // the identity setup above).
    rebuild_storage(sz);
}

// Chunk count derives from the geometry by a shift, so a null chunk size is a
// hard error rather than a degenerate case. The chunk-init loop is written
// twice on purpose: `omp parallel for` when we are outside a region and the
// array is big enough to be worth opening one, an orphaned `omp for` otherwise
// (which is serial when no region is open, and shares the work when one is).
template <ArrayChunkElement T, typename ind_type>
void ArrayChunk2<T, ind_type>::rebuild_storage(ind_type sz)
{
    // Already registered: num_tab is the registry value, not nb_tab-1 (only
    // valid right after create()'s ++nb_tab).
    const int num_tab = Proxy2<T>::share_info_array.get_num_tab(this->registry_key());

    if (sz)
    {
        if (this->size_policy_.size())
        {
            nb_chunk = this->chunks_for(sz);

            if constexpr (chunk_config::use_fast_block_compression)
            {
                Proxy2<T>::resize_tab_is_int(num_tab, nb_chunk);
            }
            this->assign_alloc_key(); // own slabs if large enough (see alloc_key_)
        }
        else if (sz && !this->size_policy_.size())
        {
            std::println(stderr, "__chunk_size null !");
            std::abort();
        }

        if (nb_chunk)
        {
            manager_chunk2::increm_somme_chunk_create(nb_chunk);
            manager_chunk2::compute_size_buffer_reuse();
            manager_chunk2::compute_size_max_number_of_chunk_to_check_compress();

            // chunk will be init to 0 (pointer on m_ptr_cst_val1)

            chunk_list.resize(nb_chunk);
            this->alloc_dirty_bitset(nb_chunk);

            if (!CHUNK_PMP_IN_PARALLEL_REGION && nb_chunk > 500)
            {
#pragma omp parallel for
                for (int i = 0; i < nb_chunk; ++i)
                {
                    chunk_list[i].init_after_constructor();
                }
            }
            else
            {
#pragma omp for
                for (int i = 0; i < nb_chunk; ++i)
                {
                    chunk_list[i].init_after_constructor();
                }
            }
        }
    }
    else
    {
        nb_chunk = 0;
    }
    // Outside the `if (sz)`: an array built empty (the default) and grown by
    // resize() or push_back() needs bool's second value registered too.
    if constexpr (is_bool)
    {
        set_second_value(true, false);
    }

    m_size = sz;
}

// Two paths, on whether the value is already registered:
//  - NEW value: allocate its read-only cst_val2 block (mmap+mprotect when
//    use_mm_protect, so a stray write to a compressed chunk faults instead of
//    corrupting every chunk sharing it), then size the per-value candidate
//    lists to the new index.
//  - KNOWN value: reuse the index, but re-fill the block to THIS array's
//    geometry first -- see the inline note, this is a real bug that fires on
//    every rechunk to a larger geometry.
// bool takes the `else` branch entirely: its second value is `true`, index 0,
// with nothing to look up.
template <ArrayChunkElement T, typename ind_type>
void ArrayChunk2<T, ind_type>::set_second_value(T sec_value, bool message)
{
    if (message)
    {
        if constexpr (is_bool)
        {
            std::println(stderr, "\nWarning useless method for bool : second_value is true");
        }
    }

    auto& sh_inf = Proxy2<T>::share_info_array;

    if constexpr (is_not_bool)
    {
        int ind = sh_inf.find_ind_tab_cst2(sec_value);

        if (ind == -1)
        {
            ind = Chunk2<T>::tab_m_ptr_cst_val2.size();

            if constexpr (build_config::use_mm_protect)
            {
                const size_t bytes = this->size_policy_.size() * sizeof(T);

                const size_t pagesize   = sysconf(_SC_PAGESIZE);
                size_t       alloc_size = (bytes + pagesize - 1) / pagesize * pagesize;

                void* ptr = mmap(nullptr, alloc_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

                T* arr = static_cast<T*>(ptr);

                for (int i = 0; i < static_cast<int>(this->size_policy_.size()); ++i)
                {
                    arr[i] = sec_value;
                }

                mprotect(ptr, alloc_size, PROT_READ);

                Chunk2<T>::tab_m_ptr_cst_val2.push_back(arr);
                if (Chunk2<T>::tab_m_ptr_cst_val2_mm_elems.size() <= static_cast<std::size_t>(ind))
                {
                    Chunk2<T>::tab_m_ptr_cst_val2_mm_elems.resize(ind + 1, 0);
                }
                Chunk2<T>::tab_m_ptr_cst_val2_mm_elems[ind] = this->size_policy_.size();
            }
            else
            {
                Chunk2<T>::tab_m_ptr_cst_val2.resize(ind + 1);
                for (int i = 0; i < static_cast<int>(this->size_policy_.size()); ++i)
                {
                    Chunk2<T>::tab_m_ptr_cst_val2[ind][i] = sec_value;
                }
            }

            sh_inf.replace(this->registry_key(), sec_value, ind);
            sh_inf.insert_map_ind_cst_val2(sec_value, ind);

            ind_cst_val2 = ind;
            if constexpr (chunk_config::use_fast_block_compression)
            {
                std::get<0>(inf_array) = sec_value;
                std::get<1>(inf_array) = ind;
            }

            if constexpr (build_config::use_openmp)
            {
                const int sz = manager_chunk2::chunk2_to_compress_potentially_thr[0].size();

                if (sz <= ind)
                {
                    for (int thr = 0; thr < CHUNK_PMP_THREAD_CAPACITY; ++thr)
                    {
                        manager_chunk2::chunk2_to_compress_potentially_thr[thr].resize(ind + 1);
                    }
                }

                for (int thr = 0; thr < CHUNK_PMP_THREAD_CAPACITY; ++thr)
                {
                    const auto& vec_ref    = manager_chunk2::chunk2_to_compress_potentially_thr[thr][ind];
                    const auto  shared_vec = std::make_shared<VectorTuple4ChunkInt2<Chunk2, T>>(vec_ref);
                    manager_chunk2::map_chunk2_to_compress_potentially_thr[thr][sec_value] = shared_vec;
                }
            }
        }
        else
        {
            // ind already registered for sec_value -- possibly by a DIFFERENT
            // array, at a DIFFERENT (smaller) geometry: tab_m_ptr_cst_val2[ind]
            // was only filled up to THAT array's size_policy_.size() at
            // registration time. Reused as-is here, any chunk of THIS array
            // compressed to cst_val2 at offsets beyond the original fill would
            // read stale zeros instead of sec_value -- the failure this rechunk
            // hits every time (rechunk_seed_fresh re-declares the SAME
            // second_value on the fresh, larger-geometry array). Re-filling to
            // the CURRENT geometry is idempotent (same value, so it never
            // corrupts a smaller array still reading the same ind) and grow-only
            // (never shrinks/reallocates out from under another array's cst_val2
            // pointer), so it is always safe to redo unconditionally here.
            if constexpr (build_config::use_mm_protect)
            {
                const std::size_t cur_elems =
                    Chunk2<T>::tab_m_ptr_cst_val2_mm_elems.size() > static_cast<std::size_t>(ind)
                        ? Chunk2<T>::tab_m_ptr_cst_val2_mm_elems[ind]
                        : 0;
                const std::size_t need_elems = this->size_policy_.size();

                if (need_elems > cur_elems)
                {
                    const size_t bytes    = need_elems * sizeof(T);
                    const size_t pagesize = sysconf(_SC_PAGESIZE);
                    const size_t alloc_sz = (bytes + pagesize - 1) / pagesize * pagesize;

                    void* ptr = mmap(nullptr, alloc_sz, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
                    T*    arr = static_cast<T*>(ptr);
                    for (std::size_t i = 0; i < need_elems; ++i)
                    {
                        arr[i] = sec_value;
                    }
                    mprotect(ptr, alloc_sz, PROT_READ);

                    // Old buffer (if any) is intentionally NOT unmapped: another
                    // still-live array's chunk may hold its address as `values`
                    // (immortal-arena discipline, same as elsewhere in this file).
                    Chunk2<T>::tab_m_ptr_cst_val2[ind] = arr;
                    if (Chunk2<T>::tab_m_ptr_cst_val2_mm_elems.size() <= static_cast<std::size_t>(ind))
                    {
                        Chunk2<T>::tab_m_ptr_cst_val2_mm_elems.resize(ind + 1, 0);
                    }
                    Chunk2<T>::tab_m_ptr_cst_val2_mm_elems[ind] = need_elems;
                }
            }
            else
            {
                for (int i = 0; i < static_cast<int>(this->size_policy_.size()); ++i)
                {
                    Chunk2<T>::tab_m_ptr_cst_val2[ind][i] = sec_value;
                }
            }

            ind_cst_val2 = ind;

            if constexpr (chunk_config::use_fast_block_compression)
            {
                std::get<0>(inf_array) = sec_value;
                std::get<1>(inf_array) = ind;
            }
        }
        const bool exi = sh_inf.exists(this->registry_key());

        if (exi && sec_value != get_second_value())
        {
            sh_inf.replace_all_other_elm(this->registry_key(), sec_value, ind_cst_val2);
            sh_inf.insert_map_ind_cst_val2(sec_value, ind);
        }
    }
    else
    {
        if constexpr (is_not_bool)
        {
            Chunk2<T>::tab_m_ptr_cst_val2.resize(+1);
            for (int i = 0; i < static_cast<int>(this->size_policy_.size()); ++i)
            {
                Chunk2<T>::tab_m_ptr_cst_val2[0][i] = true;
            }
        }
        if constexpr (chunk_config::use_fast_block_compression)
        {
            std::get<0>(inf_array) = true;
            std::get<1>(inf_array) = 0;
        }
        if constexpr (build_config::use_openmp)
        {
            for (int thr = 0; thr < CHUNK_PMP_THREAD_CAPACITY; ++thr)
            {
                manager_chunk2::chunk2_to_compress_potentially_thr[thr].resize(1);
                const auto& vec_ref    = manager_chunk2::chunk2_to_compress_potentially_thr[thr][0];
                const auto  shared_vec = std::make_shared<VectorTuple4ChunkInt2<Chunk2, T>>(vec_ref);
                manager_chunk2::map_chunk2_to_compress_potentially_thr[thr][sec_value] = shared_vec;
            }
        }

        sh_inf.replace_all_other_elm(this->registry_key(), true, 0);
    }
    if constexpr (chunk_config::use_fast_block_compression)
    {
        sh_inf.insert(sec_value, &inf_array);
    }

    // Every branch above may have rewritten this array's registry entry
    // (replace / replace_all_other_elm): refresh the mirror from it, in ALL
    // build modes. The fast-block writes to std::get<0>/<1>(inf_array) above
    // become a subset of this and are left as they were.
    sync_registry_mirror();
}

// Two independent scans: the classification into n_t0 / n_t1 (compressible to
// the first / second constant), then the "would Chunk2A have helped?" probe.
// Both re-read chunk content instead of trusting the written() flag -- the long
// note below records why that distinction was not academic.
template <ArrayChunkElement T, typename ind_type>
void ArrayChunk2<T, ind_type>::print_compression_stats(std::ostream& os) const
{
    if (nb_chunk == 0)
    {
        return;
    }

    const auto& sh_inf  = Proxy2<T>::share_info_array;
    const int   num_tab = sh_inf.get_num_tab(const_cast<void*>(this->registry_key()));

    // Chunk2-specific classification. FIXED: written(ind) tests
    // `values != cst_val1 && is_not_cst_val2(ind)` -- i.e. it is true for a
    // REALLY MATERIALISED chunk, same meaning as plain Chunk::written(). The
    // previous version of this audit had the comment (and the bucket meaning)
    // backwards: it read !written(ind) as "compressed to first constant" and
    // written(ind) as "carries a specific second value", when in truth
    // !written(ind) covers BOTH compressed states (cst_val1 and cst_val2) and
    // written(ind) is the materialised state. Since n_t0+n_t1 partitioned ALL
    // chunks by construction, n_compressed was silently equal to nb_chunk for
    // every run (a tautology, not an actual compressed count) -- caught by
    // comparing still@exit + teardown_clear against the total chunk count
    // across sequential/OpenMP runs, where it should partition but did not.
    // n_t0 = compressed to the first constant, n_t1 = compressed to the second
    // constant (kept for the existing "type-2" reporting downstream);
    // Final content check, added ON TOP of the written(ind) classification.
    // written(ind) only reflects whether the OPPORTUNISTIC deferred-compression
    // mechanism happened to catch a chunk during the run -- not whether its
    // CURRENT content is actually compressible. Confirmed on Chunk2A (same
    // pattern): re-testing content here made still@exit converge to the exact
    // same value across sequential/OMP-8t/OMP-16t (delta 0), whereas trusting
    // written(ind) alone gave a ~10% thread-count-dependent gap despite
    // identical final data. Re-scan is the same test compress()/the type-2
    // probe below already do.
    //
    // Each chunk is classified on its own, so the scans below are shared out
    // over the team when one is available (the counts are reductions). Same
    // guard as the other teardown loops: never nest inside a caller's region,
    // and not worth a team for a handful of chunks.
    const auto csz_audit = static_cast<unsigned int>(this->size_policy_.size());
    int        n_t0      = 0;
    int        n_t1      = 0;

    if constexpr (is_not_bool)
    {
        const int ind = get_ind_second_value();
#pragma omp parallel for reduction(+ : n_t0, n_t1) if (!CHUNK_PMP_IN_PARALLEL_REGION && nb_chunk > 500)
        for (int j = 0; j < nb_chunk; ++j)
        {
            const auto& ck = chunk_list[j];
            if (ck.written(ind))
            {
                const auto v0 = ck.constant_value(csz_audit);
                if (!v0)
                {
                    continue;
                }
                if (*v0 == Chunk2<T>::first_value)
                {
                    ++n_t0;
                }
                else
                {
                    ++n_t1;
                }
                continue;
            }
            if (ck.is_cst_val1())
            {
                ++n_t0;
            }
            else
            {
                ++n_t1;
            }
        }
    }
    else
    {
        // bool (and Bit, ArrayChunkBits): the second constant is `true`, one
        // for the type -- an all-true chunk is type-2, as a chunk at the
        // second value is for the other types. Counting it as type-1 made the
        // verdict "plain Chunk would suffice" for arrays whose true chunks a
        // plain Chunk cannot compress.
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
                if (*v0 == Chunk2<T>::first_value)
                {
                    ++n_t0;
                }
                else
                {
                    ++n_t1;
                }
                continue;
            }
            if (ck.is_cst_val1())
            {
                ++n_t0;
            }
            else
            {
                ++n_t1;
            }
        }
    }
    const int n_compressed = n_t0 + n_t1;

    // Cross-variant probe ("would Chunk2A have helped?"): Chunk2 shares ONE
    // second value across the array, whereas Chunk2A registers a second value
    // per chunk (find_or_set path). So Chunk2A pays off exactly on the blocks
    // that are constant to a value which is NEITHER first_value NOR the array's
    // registered second value: Chunk2 cannot compress those in any state (they
    // are always materialised here), Chunk2A would have. Counted by content
    // alone, like the classification above -- an earlier version looked only
    // at what the drain had left materialised and took its reference from the
    // first such block, which made the answer depend on the thread count.
    // O(materialised * chunk_size), once, at teardown.
    int n_would_be_2a = -1;
    if constexpr (is_not_bool) // bool has no room for a distinct second value
    {
        const auto    csz          = static_cast<unsigned int>(this->size_policy_.size());
        const int     ind          = get_ind_second_value();
        const in_t<T> second_value = get_second_value();
        n_would_be_2a              = 0;
#pragma omp parallel for reduction(+ : n_would_be_2a) if (!CHUNK_PMP_IN_PARALLEL_REGION && nb_chunk > 500)
        for (int j = 0; j < nb_chunk; ++j)
        {
            const auto& ck = chunk_list[j];
            if (!ck.written(ind))
            {
                continue; // compressed: holds first_value or second_value by definition
            }
            const in_t<T> v = ck.read_value_scan(0);
            if (v == Chunk2<T>::first_value || v == second_value)
            {
                continue; // Chunk2 material, caught or not
            }
            if (ck.is_constant_to(v, csz))
            {
                ++n_would_be_2a;
            }
        }
    }

    // Centralized rendering: shared layout/colors/verdict with ArrayChunk2A
    // (see chunk_stats_printer.h).
    if constexpr (chunk_config::stats)
    {
        add_chunk_still_compressed_at_exit<Chunk2<T>>(n_compressed);
        record_chunk_size_at_exit<Chunk2<T>>(static_cast<unsigned int>(this->size_policy_.size()));
    }
    print_formatted_chunk_stats(os, num_tab, type_name<T>(), n_compressed, nb_chunk, n_t1, "Chunk2",
                                /*n_cross=*/n_would_be_2a, /*cross_name=*/"Chunk2A");
}

// clear() and clear_core() are currently the same teardown; the split exists so
// rechunk_to() can release its temporary array through clear_core() without
// going through the public entry point.
template <ArrayChunkElement T, typename ind_type>
void ArrayChunk2<T, ind_type>::clear()
{
    this->invalidate_proxies();
    clear_core();
}

// Teardown order is load-bearing, and each step below carries the note of what
// broke without it: pin the geometry, drain deferred ops, snapshot the stats,
// drop this array's compression candidates by num_tab, then release the chunks.
// The release itself differs by build -- serial frees in place, OpenMP parks
// each buffer in the per-geometry reuse list and neutralises the chunk to
// cst_val1 so no residual candidate can free it a second time.
template <ArrayChunkElement T, typename ind_type>
void ArrayChunk2<T, ind_type>::clear_core()
{
    // Pin the ambient size to THIS array's own geometry before the
    // clear_st()/free_values() sweep below (and before flush_pending_
    // deferred_ops(), which can itself trigger frees). get_chunk_allocator()
    // routes purely on the ambient thread_chunk_size(), not on the buffer
    // being freed, so if some OTHER array's write left the ambient stale by
    // the time this array's own clear_core() runs -- whether that's a normal
    // destructor call or rechunk_to()'s teardown of tmp -- every clear_st()
    // call below would free through the WRONG size pool, producing
    // sl_contains(list, node) assertion failures / SIGSEGVs (fast-alloc
    // only: std-alloc's svfree() doesn't route by ambient at all). Same rule
    // as Proxy2::manage_full_first_value() and manage_full_second_value() --
    // this covers the array-wide sweep entry point those two don't.
    if constexpr (build_config::chunk_dynamic)
    {
        shrink_array::set_thread_chunk_size(static_cast<unsigned int>(this->size_policy_.size()));
    }

    // Drain any deferred write still targeting this array's chunks before
    // chunk_list is freed below. See ArrayChunk2Base::flush_pending_deferred_ops()
    // for the full rationale (dangling chunk refs in the global deferred maps).
    // Safe here: nb_chunk / chunk_list / the dirty bitset are unchanged at this
    // point, so the dirty-bit scan stays in bounds.
    // force: the pending-op counter has been seen to read 0 while ops were
    // still queued (array_chunk_parallel_common_inline.h); a sequential point
    // that reallocates or frees chunks cannot trust it -- as rechunk, swap and
    // ArrayChunk2A::clear_core() already do.
    this->flush_pending_deferred_ops(/*force=*/true);

    // Per-array compression stats, read here: after the flush, before the
    // rest of the teardown drain below -- as in ArrayChunk2A::clear_core().
    // Read before the flush, async compression work may still be in flight
    // and the snapshot depends on the thread count.
    if (this->exit_census_)
    {
        print_compression_stats();
    }

    if constexpr (chunk_config::use_fast_block_compression)
    {
        const auto& sh_inf = Proxy2<T>::share_info_array;

        const int num_tab = sh_inf.get_num_tab(this->registry_key());

        Proxy2<T>::clear_is_int(num_tab);

        // Remove THIS array's queued compression candidates by num_tab, before
        // its chunks are freed below. Chunk2's candidate store is
        // map_chunk2_to_compress_potentially_thr[thr][second_value] (keyed by
        // value, shared across arrays); in GNU / non-OMPT fast-block builds
        // nothing else clears it, so a destroyed array's Chunk2* would dangle
        // there and a later compression pass would walk it -> use-after-free.
        // Targeted (no compression) — this is also what makes rechunk_to's tmp
        // teardown safe. Filters by num_tab, so other live arrays are untouched.
        manager_chunk2::remove_from_compress_list_chunk2_by_num_tab(num_tab);
    }

    if (!nb_chunk)
    {
        m_size = 0;

        chunk_list.clear();

        return;
    }

    // See ArrayChunk2A::clear(): skip in aggressive mode; in the GNU build this
    // sweep scans without compressing (compress()'s clear paths are compiled
    // out), so it is wasted work that dominates the profile.
    if constexpr (chunk_config::not_use_compression_with_ompt && !chunk_config::have_aggressive_compression)
    {
        const auto& ens_type_chunk = SingletonChk<AllTypeChunk>::getRef();

        if (ens_type_chunk.have_chunk_to_compress_potentially())
        {
            ens_type_chunk.compress_chunk_to_compress_potentially(false);
        }
    }

    ManagerChunk<Chunk2, T>::decrem_somme_chunk_create(nb_chunk);
    // Recompute what derives from the total -- see ArrayChunk::clear_core():
    // otherwise a re-chunk leaves the threshold sized for the geometry it just left.
    ManagerChunk<Chunk2, T>::compute_size_buffer_reuse();
    ManagerChunk<Chunk2, T>::compute_size_max_number_of_chunk_to_check_compress();

    if constexpr (build_config::not_use_openmp)
    {
        if constexpr (is_not_bool)
        {
            const int ind = get_ind_second_value();

            for (int j = 0; j < nb_chunk; ++j)
            {
                // owns_live_buffer(), not written(ind): written() tests the
                // CURRENT second value's index only, so a chunk compressed to
                // the cst2 block of an earlier value passed as written and
                // clear_st() freed a pointer into the cst2 deque. The OpenMP
                // branch below already tests this way.
                if (chunk_list[j].owns_live_buffer())
                {
                    if constexpr (chunk_config::stats)
                    {
                        increm_chunk_teardown_clear<Chunk2<T>>();
                    }
                    chunk_list[j].clear_st(0, ind);
                }
            }
        }
        else
        {
            // Bool chunks share a single second value with no index: use the
            // no-arg written()/clear_st(0, 0) path and never call
            // get_ind_second_value() (which aborts for bool).
            for (int j = 0; j < nb_chunk; ++j)
            {
                if (chunk_list[j].written())
                {
                    if constexpr (chunk_config::stats)
                    {
                        increm_chunk_teardown_clear<Chunk2<T>>();
                    }
                    chunk_list[j].clear_st(0, 0);
                }
            }
        }
    }
    else
    {
#pragma omp parallel
        {
            const int nb_chunk0 = get_size_max_tab_omp(nb_chunk);

            std::vector<Chunk2<T>*> ll_chunk_loc;
            ll_chunk_loc.reserve(nb_chunk0);

            // owns_live_buffer(): index-agnostic, and already the bool test too
            // (bool has one cst_val2 marker). written(ind) with a single global
            // index misclassified chunks compressed at other indices, parking
            // their (never-allocated) cst_val2 markers -> b_park of UNTRACKED /
            // std-alloc double-free of a static marker.
#pragma omp for schedule(static) nowait
            for (int j = 0; j < nb_chunk; ++j)
            {
                if (chunk_list[j].owns_live_buffer())
                {
                    if constexpr (chunk_config::stats)
                    {
                        increm_chunk_teardown_clear<Chunk2<T>>();
                        // Released below without a compression: out of the live figures.
                        add_chunk_released<Chunk2<T>>(1, static_cast<long>(sizeof(T)) *
                                                             static_cast<long>(this->size_policy_.size()));
                    }

                    ll_chunk_loc.push_back(&chunk_list[j]);
                }
            }

            const int thr_rank = CHUNK_PMP_THREAD_RANK;

            const int nb = ll_chunk_loc.size();

            if (nb > 0)
            {
                // Park into the reuse list of THIS array's geometry, not the
                // ambient t_active (which may be another size mid-rechunk).
                // Ambient routing would file a size-A buffer under list[B]; a
                // later size-B pop then memsets it at B and overflows. Falls
                // back to the legacy single list in single-size builds.
                const unsigned int my_sz = this->size_policy_.size();
                auto&              lbf   = manager_chunk2::get_list_block_free_thr_for_size(thr_rank, my_sz);

                for (int k = 0; k < nb; ++k)
                {
                    T* values_loc = ll_chunk_loc[k]->values;
                    // Ledger LIVE -> PARKED: without it the block stays recorded
                    // LIVE and a later pop trips an ILLEGAL b_pop. No-op when the
                    // ledger is off.
                    CHUNK_LEDGER_PARK(values_loc, "clear_core.teardown_park");
                    lbf.push_back(values_loc);

                    // Neutralise the chunk NOW that its buffer is parked. Without
                    // this the chunk stays materialised (values still points at the
                    // parked buffer), so a residual compression candidate -- one that
                    // slipped the removal above or the terminal sweep -- would
                    // try_mark_cst_and_get_old + free_block_to_pool the SAME buffer a
                    // second time. Under std-alloc free_block_to_pool svfrees
                    // immediately, so the block is freed there AND again by
                    // clear_list_block_free_thr -> free(): invalid pointer at teardown
                    // (fast-alloc parks instead of svfreeing, so it never bites --
                    // matching the std-alloc-only crash). Storing cst_val1 makes every
                    // compress path short-circuit. Mirrors ArrayChunk2A::clear_core.
                    if constexpr (is_not_bool)
                    {
                        ll_chunk_loc[k]->init_values_cst_val1();
                    }
                    else
                    {
                        ll_chunk_loc[k]->init_values_cst_val_bool(0);
                    }
                }
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
    // region. Key 0 (shared pool) keeps the old behaviour: nothing to give
    // back per array.
    if (this->alloc_key_ != 0 && !CHUNK_PMP_IN_PARALLEL_REGION)
    {
        ManagerChunk<Chunk2, T>::drain_reuse_list_for_size(this->size_policy_.size());
    }

    // The candidate flags (one byte per chunk): released HERE, after the
    // purge (clear_is_int() above runs before it and must stay a no-op) and
    // after the chunks -- a re-chunk's old storage dies under its own num_tab.
    // From the REGISTRY, like clear_is_int() above: ArrayChunk2 never sets
    // the base member num_tab (it stays 0), so releasing this->num_tab freed
    // row 0 -- another live array's, when the first array of this type was
    // not the one torn down (shrink_bench chunk2_dyn: tab_mask's re-chunk
    // emptied tab_bool_bits' row, and its next write faulted).
    if constexpr (chunk_config::use_fast_block_compression)
    {
        Proxy2<T>::release_tab_is_int(Proxy2<T>::share_info_array.get_num_tab(this->registry_key()));
    }
}

template <ArrayChunkElement T, typename ind_type>
void ArrayChunk2<T, ind_type>::compress_int()
{
    // Diagnostic-only pass: scans for constant-but-materialised chunks the
    // incremental compression missed; it reports, never compresses (the abort is
    // commented out below), so it is pure overhead in perf/release. Skip it there
    // via PERF (defined by the perf targets); the check targets (CHECK) and debug
    // builds keep it. NB: do NOT gate on build_config::any_check here -- that is
    // is_debug||check_optim (DEBUG/CHECK_OPTIM) and does NOT match the _check
    // target's CHECK macro, so it would silently disable the scan in check builds.
    if constexpr (build_config::perf_build)
    {
        return;
    }

    // Pin the ambient size to THIS array's geometry: the frees below route on
    // thread_chunk_size(), not on the buffer (see clear_core()). Left to
    // whichever array the thread last served, a re-chunked array's buffers
    // went to another size's allocator (sl_remove assertion, fast-alloc only).
    if constexpr (build_config::chunk_dynamic)
    {
        shrink_array::set_thread_chunk_size(static_cast<unsigned int>(this->size_policy_.size()));
    }

    // Rigorous miss detection: drain the pending compression list FIRST so a
    // chunk that is constant only because its compression is still QUEUED (not
    // yet applied) is not miscounted as a genuine miss -- after the drain, any
    // residual constant-materialised chunk is a TRUE miss. compress_int runs
    // serially (ArrayChunk2::compress on the main thread; the #pragma omp for
    // below is orphaned = serial), and compress_chunk_to_compress_potentially runs
    // its OWN parallel region, so it must not be nested: guard on
    // !CHUNK_PMP_IN_PARALLEL_REGION (if compress_int ever runs inside a parallel
    // region, skip the drain -- pre-existing behaviour, no nesting).
    // `true`, as finalize_parallel_phase passes it: with false the pass scans
    // each candidate, DROPS the verdict and resets its flag (compress_impl), so
    // the "drain" un-queued the candidates and the self-heal below then
    // reported them as never marked. `drained` qualifies that report: without
    // the drain, a chunk it compresses may only have been queued.
    bool drained = false;
    if constexpr (chunk_config::not_use_compression_with_ompt && !chunk_config::have_aggressive_compression)
    {
        if (!CHUNK_PMP_IN_PARALLEL_REGION)
        {
            const auto& ens_type_chunk = shrink_array::SingletonChk<shrink_array::AllTypeChunk>::getRef();
            if (ens_type_chunk.have_chunk_to_compress_potentially())
            {
                ens_type_chunk.compress_chunk_to_compress_potentially(true);
            }
            drained = true;
        }
    }

    std::vector<int> tab_nb1;
    std::vector<int> tab_nb2;

    const int nb_res = (nb_chunk / (50 * CHUNK_PMP_MAX_THREADS)) + 1;
    tab_nb1.reserve(nb_res);
    tab_nb2.reserve(nb_res);
    T   second_value{};
    int ind = 0;

    if constexpr (is_not_bool)
    {
        second_value = get_second_value();
        ind          = get_ind_second_value();

#pragma omp for schedule(static) nowait
        for (int j = 0; j < nb_chunk; ++j)
        {
            const auto& chunk = chunk_list[j];

            if (chunk.written(ind))
            {
                // A run of first_value when the second value IS first_value
                // (none registered, or 0) comes back as `first` only: filed in
                // both lists, the reclaim below would clear_st the same chunk
                // twice and trip check_not_constant_before_clear.
                const auto scan = chunk.scan_constants(Chunk2<T>::first_value, second_value,
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
    }
    else
    {
#pragma omp for schedule(static) nowait
        for (int j = 0; j < nb_chunk; ++j)
        {
            const auto& chunk = chunk_list[j];

            if (chunk.written())
            {
                const auto scan = chunk.scan_constants(Chunk2<T>::first_value, static_cast<T>(true),
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
    }

    // Reclaim the misses the scan just found: materialised chunks fully
    // first_value / second_value that were never marked (is_inside=0) -- e.g. a
    // chunk that became all-V while V was not the designated second_value, then V
    // became designated later, with no write since to re-trigger the mark. The
    // scan already proved them constant (compact1/compact2), so compress them
    // directly via clear_int_st -- the same call make_block_compression uses --
    // rather than only reporting. Runs in check builds only (compress_int is
    // PERF-gated): a check-build self-heal, zero release cost, release memory stays
    // bounded. compress_int is serial here, so thr_rank is the calling thread.
    {
        if constexpr (is_not_bool)
        {
            // clear_st(num, ind) is the one name both Chunk2OmpPolicy
            // specialisations declare: under OpenMP it supplies the calling
            // thread's rank to clear_int_st(num, ind, thr_rank); serially it
            // routes to the 2-arg clear_int_st(num, ind), which is a different
            // algorithm (eager, no queue, counters to reset). The two are not
            // interchangeable, so call clear_st and let the policy choose.
            for (const int jj : tab_nb1)
            {
                chunk_list[jj].clear_st(0, ind);
            }
            for (const int jj : tab_nb2)
            {
                chunk_list[jj].clear_st(1, ind);
            }
        }
        else if constexpr (build_config::use_openmp)
        {
            // bool compresses via clear_int_st_bool (not clear_int_st); no clear_st
            // wrapper exists. In non-OMP, bool uses the counter path (manage_full_*
            // compresses inline) so there is nothing to reclaim -> empty else.
            for (const int jj : tab_nb1)
            {
                chunk_list[jj].clear_int_st_bool(0, CHUNK_PMP_THREAD_RANK);
            }
            for (const int jj : tab_nb2)
            {
                chunk_list[jj].clear_int_st_bool(1, CHUNK_PMP_THREAD_RANK);
            }
        }
    }

    // Post-reclaim filter: keep only chunks the reclaim did NOT compress (genuine
    // failures). The scan populated tab_nb1/tab_nb2 before clear_st ran, so without
    // this the report would list chunks that are now compressed. A chunk still
    // `written` after clear_st is a real reclaim failure worth reporting; the rest
    // are dropped, so the message fires only on genuine misses. if constexpr picks
    // written(ind) (non-bool) vs written() (bool) -- both have a runtime type guard.
    // What the reclaim DID compress is reported too. After a drain each of those
    // is a chunk the write path left constant and unmarked -- a marking miss
    // the self-heal would otherwise hide. Inside a parallel region (the bench
    // compresses one array per `omp single`) the drain is skipped and a queued
    // compression looks the same, so the report says which case it is.
    {
        const auto scanned1      = tab_nb1.size();
        const auto scanned2      = tab_nb2.size();
        const auto still_written = [&](int jj) -> bool
        {
            if constexpr (is_not_bool)
            {
                return chunk_list[jj].written(ind);
            }
            else
            {
                return chunk_list[jj].written();
            }
        };
        std::vector<int> keep1;
        std::vector<int> keep2;
        keep1.reserve(tab_nb1.size());
        keep2.reserve(tab_nb2.size());
        for (const int jj : tab_nb1)
        {
            if (still_written(jj))
            {
                keep1.push_back(jj);
            }
        }
        for (const int jj : tab_nb2)
        {
            if (still_written(jj))
            {
                keep2.push_back(jj);
            }
        }
        tab_nb1.swap(keep1);
        tab_nb2.swap(keep2);
        // After the swap keep1 / keep2 hold the scanned lists.
        const auto& keep1_scanned = keep1;
        const auto& keep2_scanned = keep2;

        const auto healed1 = scanned1 - tab_nb1.size();
        const auto healed2 = scanned2 - tab_nb2.size();
        if (healed1 + healed2 != 0)
        {
            // The healed chunks are those the filter dropped: the scanned list
            // minus the kept one. A few indices name the chunk for whoever
            // wants to map it back to the writes that left it constant.
            const auto healed_indices = [](const std::vector<int>& scanned, const std::vector<int>& kept)
            {
                std::string out;
                int         shown = 0;
                for (const int j : scanned)
                {
                    if (std::ranges::find(kept, j) != kept.end())
                    {
                        continue;
                    }
                    if (shown++ == 8)
                    {
                        out += " ...";
                        break;
                    }
                    out += ' ';
                    out += std::to_string(j);
                }
                return out;
            };
#pragma omp critical
            {
                std::println(stderr,
                             "[compress_int self-heal] ArrayChunk2<{}, {}>: {} chunk(s) at first_value [{} ], {} at "
                             "second_value [{} ], materialised and not compressed: compressed now{}",
                             type_name<T>(), type_name<ind_type>(), healed1, healed_indices(keep1_scanned, tab_nb1),
                             healed2, healed_indices(keep2_scanned, tab_nb2),
                             drained ? " (pending list drained first: never marked)"
                                     : " (in a parallel region, pending list NOT drained: maybe only queued)");
            }
        }
    }

    const auto nb1 = static_cast<int>(tab_nb1.size());

    if (nb1 != 0)
    {
#pragma omp critical
        {
            std::println(stderr, "Should not have compress (0) values");
            std::println(stderr, "Number of compress block : {}", nb1);
            std::println(stderr, "ArrayChunk2 type: ArrayChunk2<{}, {}>", type_name<T>(), type_name<ind_type>());

            for (int k = 0; k < nb1; ++k)
            {
                const int ind1 = tab_nb1[k];
                std::print(stderr, "{} {}", ind1,
                           static_cast<ind_type>(ind1) * static_cast<ind_type>(this->size_policy_.size()));
                // The candidate-list flag exists in fast-block builds only: a
                // serial build has no such table, indexing it aborted here.
                if constexpr (chunk_config::use_fast_block_compression)
                {
                    std::print(stderr, " is_inside={}",
                               static_cast<int>(shrink_array::Proxy2<T>::get_tab_is_inside_list_to_compress(
                                                    shrink_array::Proxy2<T>::share_info_array.get_num_tab(
                                                        const_cast<void*>(this->registry_key())))[ind1]
                                                    .load(std::memory_order_acquire)));
                }
                std::println(stderr);
                if (k > 10)
                {
                    break;
                }
                if constexpr (chunk_config::use_compression_with_counter)
                {
                    const auto& chunk = chunk_list[ind1];
                    std::println(stderr, "nb first_value : {}", static_cast<unsigned>(chunk.get_nb_first_value()));
                }
            }
            if (nb1 > 10)
            {
                std::print(stderr, "\n...\n");
            }
        }
    }

    const auto nb2 = static_cast<int>(tab_nb2.size());
    if (nb2 != 0)
    {
#pragma omp critical
        {
            if constexpr (is_not_bool)
            {
                std::println(stderr, "Should not have compress of second values ( {} )", value_repr(second_value));
            }
            else
            {
                std::println(stderr, "Should not have compress of second values ( {:d} )", true);
            }
            std::println(stderr, "Number of compress block : {}", nb2);
            std::println(stderr, "ArrayChunk2 type: ArrayChunk2<{}, {}>", type_name<T>(), type_name<ind_type>());

            for (int k = 0; k < nb2; ++k)
            {
                const int ind2 = tab_nb2[k];
                std::print(stderr, "{} {}", ind2,
                           static_cast<ind_type>(ind2) * static_cast<ind_type>(this->size_policy_.size()));
                // The candidate-list flag exists in fast-block builds only: a
                // serial build has no such table, indexing it aborted here.
                if constexpr (chunk_config::use_fast_block_compression)
                {
                    std::print(stderr, " is_inside={}",
                               static_cast<int>(shrink_array::Proxy2<T>::get_tab_is_inside_list_to_compress(
                                                    shrink_array::Proxy2<T>::share_info_array.get_num_tab(
                                                        const_cast<void*>(this->registry_key())))[ind2]
                                                    .load(std::memory_order_acquire)));
                }
                std::println(stderr);
                if (k > 10)
                {
                    break;
                }
                if constexpr (chunk_config::use_compression_with_counter)
                {
                    const auto& chunk = chunk_list[ind2];
                    std::println(stderr, "nb second_value : {}", static_cast<unsigned>(chunk.get_nb_second_value()));
                }
            }
            if (nb2 > 10)
            {
                std::print(stderr, "\n...\n");
            }
        }
    }

    if (nb1 > 0 || nb2 > 0)
    {
        std::println(stderr, "the algorithm can miss certain blocks to compress: normally rare:"
                             " it’s an issue if there are many");
    }
}

// Plain forwarder. The commented-out in-parallel-region test is kept as the
// record of a workaround for an LLVM OpenMP runtime bug.
template <ArrayChunkElement T, typename ind_type>
void ArrayChunk2<T, ind_type>::compress()
{
    //  const bool in_parr = CHUNK_PMP_IN_PARALLEL_REGION;
    // bug in run time openmp llvm

    compress_int();
}

// old_nb_chunk is sampled BEFORE resize_common() updates nb_chunk, so the loop
// below initialises only the chunks that were just added. On a shrink
// resize_common() returns early and nb_chunk <= old_nb_chunk, which leaves that
// loop empty -- the array reports the smaller size but keeps its storage.
template <ArrayChunkElement T, typename ind_type>
void ArrayChunk2<T, ind_type>::release_chunk_range(int first, int last)
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
        for (int j = first; j < last; ++j)
        {
            if (chunk_list[j].owns_live_buffer())
            {
                if constexpr (chunk_config::stats)
                {
                    increm_chunk_teardown_clear<Chunk2<T>>();
                }
                if constexpr (is_not_bool)
                {
                    chunk_list[j].clear_st(0, get_ind_second_value());
                }
                else
                {
                    chunk_list[j].clear_st(0, 0);
                }
            }
        }
    }
    else
    {
        const int          thr_rank = CHUNK_PMP_THREAD_RANK;
        const unsigned int my_sz    = this->size_policy_.size();
        auto&              lbf      = manager_chunk2::get_list_block_free_thr_for_size(thr_rank, my_sz);
        for (int j = first; j < last; ++j)
        {
            if (chunk_list[j].owns_live_buffer())
            {
                if constexpr (chunk_config::stats)
                {
                    increm_chunk_teardown_clear<Chunk2<T>>();
                    add_chunk_released<Chunk2<T>>(1, static_cast<long>(sizeof(T)) * static_cast<long>(my_sz));
                }
                T* values_loc = chunk_list[j].values;
                CHUNK_LEDGER_PARK(values_loc, "resize.shrink_park");
                lbf.push_back(values_loc);
                // Neutralise, as clear_core(): the buffer belongs to the reuse
                // list now.
                if constexpr (is_not_bool)
                {
                    chunk_list[j].init_values_cst_val1();
                }
                else
                {
                    chunk_list[j].init_values_cst_val_bool(0);
                }
            }
        }
    }
}

template <ArrayChunkElement T, typename ind_type>
void ArrayChunk2<T, ind_type>::resize(ind_type newsz)
{
    this->invalidate_proxies();
    const int      old_nb_chunk = nb_chunk;
    const ind_type old_size     = this->m_size; // the stale-tail reset at the end

    Base2::resize_common(newsz);

    if constexpr (chunk_config::use_fast_block_compression)
    {
        const auto& sh_inf = Proxy2<T>::share_info_array;

        const int num_tab = sh_inf.get_num_tab(this->registry_key());
        Proxy2<T>::resize_tab_is_int(num_tab, nb_chunk);
    }

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

    // std::vector semantics for the regrown range [old_size, newsz): it reads
    // T{}. The chunks added above are fresh, but the one holding the OLD end
    // keeps its tail -- after a shrink, the values written there before came
    // back on the next growth (deferred_path_validation, resize-grow-again).
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

} // namespace shrink_array
