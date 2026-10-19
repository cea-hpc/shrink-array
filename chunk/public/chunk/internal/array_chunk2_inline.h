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

#include <algorithm>   // std::ranges::find (the healed-chunk report), std::min (the birth size of bits)
#include <array>       // std::array (the residual's sample)
#include <atomic>      // std::memory_order
#include <cstddef>     // std::size_t
#include <ostream>     // print_compression_stats(std::ostream&)
#include <string>      // the healed-chunk report
#include <string_view> // std::string_view (the residual's kind)
#include <vector>      // std::vector

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
#include "chunk/internal/invariant_check.h"             // CHUNK_INVARIANT_FAILED, value_repr
#include "chunk/internal/chunk_index_sample.h"          // chunk_index_sample (the residual's sample)
#include "chunk/log.h"                                  // CHUNK_LOG, logging::enabled
#include "chunk/PMP_CHUNK.h"                            // CHUNK_PMP_*, pmp::parallel_sum
#include "chunk/internal/concept_array_chunk_element.h" // ArrayChunkElement
#include "chunk/internal/parallel_storage_init.h"       // for_each_chunk
// Explicit includes for symbols used by clear_core() below, so this TU does not
// depend on the transitive include chain.
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
 * Two invariants recur across these bodies, stated next to the code that
 * enforces them:
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
// rebuild_storage(). Everything here happens exactly once per array:
// rechunk_to() must NOT redo it.
template <ArrayChunkElement T, typename ind_type>
void ArrayChunk2<T, ind_type>::create(ind_type sz)
{
    // The first array of this element type registers its manager; every
    // later one pays a guard load. See AllTypeChunk.
    AllTypeChunk::ensure_manager<ManagerChunk<Chunk2, T>>();

    chunk_config::CheckNaN<chunk_config::do_check>::update(true);

    auto& sh_inf = *Proxy2<T>::share_info_array;

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
        // Capped at the largest multiple of the size that the shared constant
        // buffers cover: a multiple of a valid size is valid (768 * 8 = 6144 >
        // 4096 gives 3840, not 4096, which 768 does not divide).
        const unsigned int size = this->size_policy_.size();
        const unsigned int birth =
            std::min(size * packed_bit_birth_factor, shrink_array::cst_buffer_size / size * size);
        if (!t_birth_at_active_size && birth != this->size_policy_.size())
        {
            const unsigned int global = g_active_chunk_size;
            const unsigned int thread = t_active_chunk_size;
            this->set_chunk_size(birth);
            g_active_chunk_size = global;
            set_thread_chunk_size(thread);
        }
    }

    // rechunk_to() reuses rebuild_storage() WITHOUT re-registering the array.
    rebuild_storage(sz);
}

// Chunk count derives from the geometry by a shift, so a null chunk size is a
// hard error rather than a degenerate case. The chunks are initialised by
// parallel_storage_init::for_each_chunk(): a team of its own outside a region
// for a large enough array, the calling thread otherwise.
template <ArrayChunkElement T, typename ind_type>
void ArrayChunk2<T, ind_type>::rebuild_storage(ind_type sz)
{
    // Already registered: num_tab is the registry value, not nb_tab-1 (only
    // valid right after create()'s ++nb_tab).
    const int num_tab = Proxy2<T>::share_info_array->get_num_tab(this->registry_key());

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
            CHUNK_INVARIANT_FAILED("create({} elements): the chunk size is 0 (no geometry set)", sz);
        }

        if (nb_chunk)
        {
            manager_chunk2::increm_somme_chunk_create(nb_chunk);
            manager_chunk2::compute_size_buffer_reuse();
            manager_chunk2::compute_size_max_number_of_chunk_to_check_compress();

            // Chunks start at cst_val1 (the first value).
            this->acquire_chunk_list(static_cast<std::size_t>(nb_chunk));
            this->alloc_dirty_bitset(nb_chunk);

            // A team of its own outside a region; the calling thread inside one
            // (an orphaned `omp for` would initialise 1/N of an array created by
            // each thread, and hang one created under `omp single`).
            parallel_storage_init::for_each_chunk(nb_chunk,
                                                  [this](int i) { this->construct_chunk(i).init_after_constructor(); });
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
//    geometry first (see the inline note).
// bool takes the `else` branch entirely: its second value is `true`, index 0,
// with nothing to look up.
template <ArrayChunkElement T, typename ind_type>
void ArrayChunk2<T, ind_type>::set_second_value(T sec_value, bool message)
{
    if (message)
    {
        if constexpr (is_bool)
        {
            CHUNK_LOG(warn, general, "set_second_value: useless for bool, its second value is true");
        }
    }

    if constexpr (is_not_bool)
    {
        // T{} is the first value; while the second is T{} too (none
        // declared), declaring T{} would register a block and candidate lists
        // for a value every write already folds as the first one.
        if (sec_value == T{} && get_second_value() == T{})
        {
            if (message)
            {
                CHUNK_LOG(warn, general, "set_second_value(T{{}}) ignored: T{{}} is already the first value");
            }
            return;
        }
    }

    auto& sh_inf = *Proxy2<T>::share_info_array;

    if constexpr (is_not_bool)
    {
        int ind = sh_inf.find_ind_tab_cst2(sec_value);

        if (ind == -1)
        {
            ind = Chunk2<T>::tab_m_ptr_cst_val2->size();

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

                Chunk2<T>::tab_m_ptr_cst_val2->push_back(arr);
                if (Chunk2<T>::tab_m_ptr_cst_val2_mm_elems.size() <= static_cast<std::size_t>(ind))
                {
                    Chunk2<T>::tab_m_ptr_cst_val2_mm_elems.resize(ind + 1, 0);
                }
                Chunk2<T>::tab_m_ptr_cst_val2_mm_elems[ind] = this->size_policy_.size();
            }
            else
            {
                Chunk2<T>::tab_m_ptr_cst_val2->resize(ind + 1);
                for (int i = 0; i < static_cast<int>(this->size_policy_.size()); ++i)
                {
                    (*Chunk2<T>::tab_m_ptr_cst_val2)[ind][i] = sec_value;
                }
            }

            sh_inf.replace(this->registry_key(), sec_value, ind);
            sh_inf.insert_map_ind_cst_val2(sec_value, ind);

            ind_cst_val2 = ind;
            if constexpr (chunk_config::use_fast_block_compression)
            {
                inf_array.second_value = sec_value;
                inf_array.ind          = ind;
            }

            if constexpr (build_config::threaded)
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
            // array at a smaller geometry: the block was only filled up to THAT
            // array's size. Reused as-is, a chunk of THIS array at cst_val2 would
            // read zeros past the original fill (every rechunk to a larger
            // geometry re-declares the same second value). Re-filling to the
            // CURRENT geometry is idempotent (same value) and grow-only (never
            // shrinks or reallocates under another array's cst_val2 pointer), so
            // it is always safe to redo here.
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

                    // Do NOT unmap the old buffer: a live array's chunk may hold its
                    // address as `values` (immortal-arena discipline).
                    (*Chunk2<T>::tab_m_ptr_cst_val2)[ind] = arr;
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
                    (*Chunk2<T>::tab_m_ptr_cst_val2)[ind][i] = sec_value;
                }
            }

            ind_cst_val2 = ind;

            if constexpr (chunk_config::use_fast_block_compression)
            {
                inf_array.second_value = sec_value;
                inf_array.ind          = ind;
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
            Chunk2<T>::tab_m_ptr_cst_val2->resize(+1);
            for (int i = 0; i < static_cast<int>(this->size_policy_.size()); ++i)
            {
                (*Chunk2<T>::tab_m_ptr_cst_val2)[0][i] = true;
            }
        }
        if constexpr (chunk_config::use_fast_block_compression)
        {
            inf_array.second_value = true;
            inf_array.ind          = 0;
        }
        if constexpr (build_config::threaded)
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
    // build modes.
    sync_registry_mirror();
}

// Two independent scans: the classification into n_t0 / n_t1 (compressible to
// the first / second constant), then the "would Chunk2A have helped?" probe.
// Both re-read chunk content instead of trusting the written() flag (see below).
template <ArrayChunkElement T, typename ind_type>
void ArrayChunk2<T, ind_type>::print_compression_stats(std::ostream& os) const
{
    if (nb_chunk == 0)
    {
        return;
    }

    const auto& sh_inf  = *Proxy2<T>::share_info_array;
    const int   num_tab = sh_inf.get_num_tab(this->registry_key());

    // Classification. written(ind) is true for a MATERIALISED chunk (`values !=
    // cst_val1 && is_not_cst_val2(ind)`, as Chunk::written()); !written(ind)
    // covers BOTH compressed states. n_t0 = at the first constant, n_t1 = at the
    // second (the "type-2" reporting downstream). Do not read !written(ind) as
    // "first constant" and written(ind) as "second": n_t0 + n_t1 would then
    // count every chunk.
    // A materialised chunk is also re-tested by content: written(ind) only says
    // whether the opportunistic deferred compression happened to catch it, not
    // whether its CURRENT content is compressible. Trusting it alone makes the
    // count depend on the thread count for identical data. Same test as
    // compress() and the type-2 probe below.
    //
    // Each chunk is classified on its own, so the scans below are shared out
    // over the team when one is available (the counts are reductions). Same
    // guard as the other teardown loops: never nest inside a caller's region,
    // and not worth a team for a handful of chunks.
    const auto         csz_audit = static_cast<unsigned int>(this->size_policy_.size());
    int                n_t0      = 0; // compressed (or compressible) to first_value
    int                n_t1      = 0; // ... to the second value
    const pmp::options team_opt{.use_team = !CHUNK_PMP_IN_PARALLEL_REGION && nb_chunk > 500};

    if constexpr (is_not_bool)
    {
        const int  ind    = get_ind_second_value();
        const auto counts = pmp::parallel_sum<int, 2>(0, nb_chunk, team_opt,
                                                      [&](int j, auto& acc)
                                                      {
                                                          const auto& ck = chunk_list[j];
                                                          if (ck.written(ind))
                                                          {
                                                              const auto v0 = ck.constant_value(csz_audit);
                                                              if (!v0)
                                                              {
                                                                  return;
                                                              }
                                                              ++acc[*v0 == Chunk2<T>::first_value ? 0 : 1];
                                                              return;
                                                          }
                                                          ++acc[ck.is_cst_val1() ? 0 : 1];
                                                      });
        n_t0              = counts[0];
        n_t1              = counts[1];
    }
    else
    {
        // bool (and Bit, ArrayChunkBits): the second constant is `true`, one
        // for the type -- an all-true chunk is type-2. Do not count it as
        // type-1: the verdict would say "plain Chunk would suffice" for arrays
        // whose true chunks a plain Chunk cannot compress.
        const auto counts = pmp::parallel_sum<int, 2>(0, nb_chunk, team_opt,
                                                      [&](int j, auto& acc)
                                                      {
                                                          const auto& ck = chunk_list[j];
                                                          if (ck.written())
                                                          {
                                                              const auto v0 = ck.constant_value(csz_audit);
                                                              if (!v0)
                                                              {
                                                                  return;
                                                              }
                                                              ++acc[*v0 == Chunk2<T>::first_value ? 0 : 1];
                                                              return;
                                                          }
                                                          ++acc[ck.is_cst_val1() ? 0 : 1];
                                                      });
        n_t0              = counts[0];
        n_t1              = counts[1];
    }
    const int n_compressed = n_t0 + n_t1;

    // Cross-variant probe ("would Chunk2A have helped?"): Chunk2 shares ONE
    // second value across the array, whereas Chunk2A registers a second value
    // per chunk (find_or_set path). So Chunk2A pays off exactly on the blocks
    // that are constant to a value which is NEITHER first_value NOR the array's
    // registered second value: Chunk2 cannot compress those in any state (they
    // are always materialised here), Chunk2A would have. Counted by content
    // alone, like the classification above, so the answer does not depend on
    // the thread count. O(materialised * chunk_size), once, at teardown.
    int n_would_be_2a = -1;
    if constexpr (is_not_bool) // bool has no room for a distinct second value
    {
        const auto    csz          = static_cast<unsigned int>(this->size_policy_.size());
        const int     ind          = get_ind_second_value();
        const in_t<T> second_value = get_second_value();
        n_would_be_2a =
            pmp::parallel_sum<int, 1>(0, nb_chunk, team_opt,
                                      [&](int j, auto& acc)
                                      {
                                          const auto& ck = chunk_list[j];
                                          if (!ck.written(ind))
                                          {
                                              return; // compressed: holds first_value or second_value by definition
                                          }
                                          const in_t<T> v = ck.read_value_scan(0);
                                          if (v == Chunk2<T>::first_value || v == second_value)
                                          {
                                              return; // Chunk2 material, caught or not
                                          }
                                          if (ck.is_constant_to(v, csz))
                                          {
                                              ++acc[0];
                                          }
                                      })[0];
    }

    // Centralized rendering: shared layout/colors/verdict with ArrayChunk2A
    // (see chunk_stats_printer.h).
    if constexpr (chunk_config::stats)
    {
        add_chunk_still_compressed_at_exit<Chunk2<T>>(n_compressed);
        record_chunk_size_at_exit<Chunk2<T>>(static_cast<unsigned int>(this->size_policy_.size()));
    }
    print_formatted_chunk_stats(os, num_tab, this->debug_name(), type_name<T>(), n_compressed, nb_chunk, n_t1, "Chunk2",
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

// Teardown order is load-bearing (each step below says why): pin the geometry,
// drain deferred ops, snapshot the stats, drop this array's compression
// candidates by num_tab, then release the chunks.
// The release itself differs by build -- serial frees in place, OpenMP parks
// each buffer in the per-geometry reuse list and neutralises the chunk to
// cst_val1 so no residual candidate can free it a second time.
template <ArrayChunkElement T, typename ind_type>
void ArrayChunk2<T, ind_type>::clear_core()
{
    // Pin the ambient size to THIS array's geometry before the clear_st() /
    // free_values() sweep below (and before flush_pending_deferred_ops(), which
    // can itself free). get_chunk_allocator() routes on the ambient
    // thread_chunk_size(), not on the buffer, so an ambient left by ANOTHER
    // array (destructor, or rechunk_to()'s teardown of tmp) would free through
    // the WRONG size pool (fast-alloc only; std-alloc's svfree() does not route).
    // Same rule as Proxy2::manage_full_first_value() / manage_full_second_value(),
    // applied to the array-wide sweep.
    if constexpr (build_config::chunk_dynamic)
    {
        shrink_array::set_thread_chunk_size(static_cast<unsigned int>(this->size_policy_.size()));
    }

    // Drain any deferred write still targeting this array's chunks before
    // chunk_list is freed below. See ArrayChunk2Base::flush_pending_deferred_ops()
    // for the full rationale (dangling chunk refs in the global deferred maps).
    // Safe here: nb_chunk / chunk_list / the dirty bitset are unchanged at this
    // point, so the dirty-bit scan stays in bounds.
    // force: the pending-op counter can read 0 while ops are still queued
    // (array_chunk_parallel_common_inline.h); a sequential point that
    // reallocates or frees chunks must not trust it (as rechunk, swap and
    // ArrayChunk2A::clear_core()).
    this->flush_pending_deferred_ops(/*force=*/true);

    // Per-array compression stats, read here: after the flush (before it, async
    // compression work may be in flight and the snapshot depends on the thread
    // count), before the rest of the teardown -- as in ArrayChunk2A::clear_core().
    if (this->exit_census_)
    {
        print_compression_stats();
    }

    if constexpr (chunk_config::use_fast_block_compression)
    {
        const auto& sh_inf = *Proxy2<T>::share_info_array;

        const int num_tab = sh_inf.get_num_tab(this->registry_key());

        Proxy2<T>::clear_is_int(num_tab);

        // Remove THIS array's queued compression candidates by num_tab, before
        // its chunks are freed below. Chunk2's candidate store is
        // map_chunk2_to_compress_potentially_thr[thr][second_value] (keyed by
        // value, shared across arrays); in GNU / non-OMPT fast-block builds
        // nothing else clears it, so a destroyed array's Chunk2* would dangle
        // there for a later pass to walk. Targeted (no compression, other arrays
        // untouched); it also makes rechunk_to's tmp teardown safe.
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
    // out), so it would be costly wasted work.
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

    if constexpr (build_config::not_threaded)
    {
        if constexpr (is_not_bool)
        {
            const int ind = get_ind_second_value();

            for (int j = 0; j < nb_chunk; ++j)
            {
                // owns_live_buffer(), not written(ind): written() tests the
                // CURRENT second value's index only, so a chunk compressed to
                // the cst2 block of an earlier value would pass as written and
                // clear_st() would free a pointer into the cst2 deque.
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
        pmp::team(pmp::options{},
                  [&](pmp::Team& team)
                  {
                      const int nb_chunk0 = get_size_max_tab_omp(nb_chunk);

                      std::vector<Chunk2<T>*> ll_chunk_loc;
                      ll_chunk_loc.reserve(nb_chunk0);

                      // owns_live_buffer(): index-agnostic, and the bool test too (bool has
                      // one cst_val2 marker). Not written(ind): with a single global index
                      // it misclassifies chunks compressed at other indices and parks
                      // their (never-allocated) cst_val2 markers.
                      team.for_static<int>(
                          0, nb_chunk,
                          [&](int j)
                          {
                              if (chunk_list[j].owns_live_buffer())
                              {
                                  if constexpr (chunk_config::stats)
                                  {
                                      increm_chunk_teardown_clear<Chunk2<T>>();
                                      // Released below without a compression: out of the live figures.
                                      add_chunk_released<Chunk2<T>>(1,
                                                                    static_cast<long>(sizeof(T)) *
                                                                        static_cast<long>(this->size_policy_.size()));
                                  }

                                  ll_chunk_loc.push_back(&chunk_list[j]);
                              }
                          },
                          pmp::nowait);

                      const int thr_rank = CHUNK_PMP_THREAD_RANK;

                      const int nb = ll_chunk_loc.size();

                      if (nb > 0)
                      {
                          // Park into the reuse list of THIS array's geometry, not the
                          // ambient t_active (which may be another size mid-rechunk):
                          // a size-A buffer filed under list[B] is later popped and
                          // memset at B, and overflows. Single-size builds have one list.
                          const unsigned int my_sz = this->size_policy_.size();
                          auto&              lbf   = manager_chunk2::get_list_block_free_thr_for_size(thr_rank, my_sz);

                          for (int k = 0; k < nb; ++k)
                          {
                              T* values_loc = ll_chunk_loc[k]->values;
                              // Ledger LIVE -> PARKED, or a later pop trips an ILLEGAL b_pop.
                              CHUNK_LEDGER_PARK(values_loc, "clear_core.teardown_park");
                              lbf.push_back(values_loc);

                              // Neutralise the chunk NOW that its buffer is parked. Otherwise
                              // it stays materialised on the parked buffer, and a residual
                              // compression candidate (one that slipped the removal above)
                              // would try_mark_cst_and_get_old + free_block_to_pool the SAME
                              // buffer again: under std-alloc that frees it twice (there and
                              // in clear_list_block_free_thr). Storing cst_val1 makes every
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
                  });
    }

    m_size = 0;

    this->release_chunk_list(); // Mandatory with the new allocator (a re-chunk keeps the buffer)

    nb_chunk = 0;

    // An array with its own slab affinity gives its memory back NOW: every
    // buffer parked above (own rank) or by another rank (foreign, under this
    // array's key) goes home to its allocator set, whose slabs hold nothing
    // else and so empty -- and reclaim_empty_slabs madvises them away.
    // Sequential point (destructor / rechunk_to); the drain opens its own
    // region. Key 0 (shared pool): nothing to give back per array.
    if (this->alloc_key_ != 0 && !CHUNK_PMP_IN_PARALLEL_REGION)
    {
        ManagerChunk<Chunk2, T>::drain_reuse_list_for_size(this->size_policy_.size());
    }

    // The candidate flags (one byte per chunk): released HERE, after the
    // purge (clear_is_int() above runs before it and must stay a no-op) and
    // after the chunks -- a re-chunk's old storage dies under its own num_tab.
    // num_tab from the REGISTRY, like clear_is_int() above. Not this->num_tab:
    // ArrayChunk2 never sets that base member (it stays 0), so it would free
    // row 0, another live array's.
    if constexpr (chunk_config::use_fast_block_compression)
    {
        Proxy2<T>::release_tab_is_int(Proxy2<T>::share_info_array->get_num_tab(this->registry_key()));
    }
}

template <ArrayChunkElement T, typename ind_type>
void ArrayChunk2<T, ind_type>::compress_int(pmp::Team& team)
{
    // Diagnostic pass: scans for constant-but-materialised chunks the
    // incremental compression missed, reports them and self-heals them (below).
    // Do NOT gate it on build_config::any_check: that is DEBUG/CHECK_OPTIM, not
    // the _check target's CHECK macro, so it would silently disable the scan there.

    // Pin the ambient size to THIS array's geometry: the frees below route on
    // thread_chunk_size(), not on the buffer (see clear_core()).
    if constexpr (build_config::chunk_dynamic)
    {
        shrink_array::set_thread_chunk_size(static_cast<unsigned int>(this->size_policy_.size()));
    }

    // Drain the pending compression list FIRST, so a chunk constant only because
    // its compression is still QUEUED is not counted as a miss; after the drain,
    // a constant materialised chunk is a TRUE miss. The drain opens its OWN
    // parallel region: skip it inside one (no nesting).
    // Pass `true`, as finalize_parallel_phase does: with false the pass scans
    // each candidate, DROPS the verdict and resets its flag (compress_impl), so
    // the drain would un-queue the candidates and the self-heal below would
    // report them as never marked. `drained` qualifies that report.
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

        team.for_static<int>(
            0, nb_chunk,
            [&](int j)
            {
                const auto& chunk = chunk_list[j];

                if (chunk.written(ind))
                {
                    // A run of first_value when the second value IS first_value
                    // (none registered, or 0) comes back as `first` only: filed in
                    // both lists, the reclaim below would clear_st it twice.
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
            },
            pmp::nowait);
    }
    else
    {
        team.for_static<int>(
            0, nb_chunk,
            [&](int j)
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
            },
            pmp::nowait);
    }

    // Reclaim the misses the scan just found: materialised chunks fully
    // first_value / second_value that were never marked (is_inside=0) -- e.g. a
    // chunk that became all-V before V was designated second_value, with no write
    // since to re-trigger the mark. The scan proved them constant, so compress
    // them directly (the call make_block_compression uses). Check builds only
    // (compress_int is PERF-gated). Serial here: thr_rank is the calling thread.
    //
    // A scanned chunk may be on a compression list -- another rank's, whose
    // pass can be running now, in a parallel region (the drain above skipped):
    // its scan done, its clear to come. A clear here, outside that pass's
    // exclusion, has one side free a shared constant block
    // (check_not_constant_before_clear). So each chunk's dedup flag is taken
    // first, as a queueing write takes it: already up, the chunk is its pass's
    // and is left to it (and out of the reports below); held, no pass can take
    // it, and the chunk is re-checked -- a pass may have compressed it since
    // the scan -- before the clear, then the flag handed back.
    const auto materialised = [&](int jj) -> bool
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
    std::vector<int> left_to_pass;
    {
        [[maybe_unused]] chunk_flag_row_t* flags = nullptr;
        if constexpr (chunk_config::use_fast_block_compression)
        {
            flags = &Proxy2<T>::get_tab_is_inside_list_to_compress(
                Proxy2<T>::share_info_array->get_num_tab(const_cast<void*>(this->registry_key())));
        }
        const auto heal = [&](int jj, auto&& clear)
        {
            if constexpr (chunk_config::use_fast_block_compression)
            {
                auto& flag     = (*flags)[static_cast<std::size_t>(jj)];
                char  expected = 0;
                if (!flag.compare_exchange_strong(expected, 1, std::memory_order_acq_rel, std::memory_order_acquire))
                {
                    left_to_pass.push_back(jj);
                    return;
                }
                if (materialised(jj))
                {
                    clear();
                }
                flag.store(0, std::memory_order_release);
            }
            else
            {
                clear();
            }
        };

        if constexpr (is_not_bool)
        {
            // Call clear_st(num, ind), the one name both Chunk2OmpPolicy
            // specialisations declare, and let the policy choose: under OpenMP
            // clear_int_st(num, ind, thr_rank); serially the 2-arg
            // clear_int_st(num, ind), a different algorithm (eager, no queue,
            // counters to reset). They are not interchangeable.
            for (const int jj : tab_nb1)
            {
                heal(jj, [&] { chunk_list[jj].clear_st(0, ind); });
            }
            for (const int jj : tab_nb2)
            {
                heal(jj, [&] { chunk_list[jj].clear_st(1, ind); });
            }
        }
        else if constexpr (build_config::threaded)
        {
            // bool compresses via clear_int_st_bool (not clear_int_st); no clear_st
            // wrapper exists. In non-OMP, bool uses the counter path (manage_full_*
            // compresses inline) so there is nothing to reclaim -> empty else.
            for (const int jj : tab_nb1)
            {
                heal(jj, [&] { chunk_list[jj].clear_int_st_bool(0, CHUNK_PMP_THREAD_RANK); });
            }
            for (const int jj : tab_nb2)
            {
                heal(jj, [&] { chunk_list[jj].clear_int_st_bool(1, CHUNK_PMP_THREAD_RANK); });
            }
        }
    }

    // Post-reclaim filter: keep only chunks the reclaim did NOT compress (still
    // `written`): those are the genuine failures reported below.
    // What the reclaim DID compress is reported too: after a drain each is a
    // chunk the write path left constant and unmarked -- a marking miss the
    // self-heal would otherwise hide. Inside a parallel region the drain is
    // skipped and a queued compression looks the same, so the report says which.
    {
        // Left to their pass: neither healed here nor a miss.
        const auto to_pass = [&](int jj) { return std::ranges::find(left_to_pass, jj) != left_to_pass.end(); };
        std::erase_if(tab_nb1, to_pass);
        std::erase_if(tab_nb2, to_pass);
        const auto scanned1      = tab_nb1.size();
        const auto scanned2      = tab_nb2.size();
        const auto still_written = materialised;
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
            CHUNK_LOG(info, compress,
                      "self-heal ArrayChunk2<{}, {}>: {} chunk(s) at first_value [{} ], {} at second_value [{} ], "
                      "materialised and not compressed: compressed now{}",
                      type_name<T>(), type_name<ind_type>(), healed1, healed_indices(keep1_scanned, tab_nb1), healed2,
                      healed_indices(keep2_scanned, tab_nb2),
                      drained ? " (pending list drained first: never marked)"
                              : " (in a parallel region, pending list NOT drained: maybe only queued)");
        }
    }

    // compress=info: how many, and a sample; compress=debug: per chunk, the
    // candidate flag (fast-block builds) and the counter (counter builds).
    // One line per kind and thread (its share of the chunks).
    const auto report = [&](const std::vector<int>& tab, std::string_view at, bool second)
    {
        if (tab.empty())
        {
            return;
        }
        std::array<char, 256> sample{};
        CHUNK_LOG(info, compress,
                  "residual ArrayChunk2<{}, {}>: {} chunk(s) constant at {}, still materialised: {} (the algorithm can "
                  "miss a few blocks to compress: an issue only if there are many)",
                  type_name<T>(), type_name<ind_type>(), tab.size(), at,
                  internal::chunk_index_sample(sample, tab, static_cast<ind_type>(this->size_policy_.size())));
        if (!logging::enabled(logging::Channel::compress, logging::Level::debug))
        {
            return;
        }
        const std::size_t shown = tab.size() < 10 ? tab.size() : 10;
        for (std::size_t k = 0; k < shown; ++k)
        {
            const int ind       = tab[k];
            int       is_inside = -1; // the candidate-list flag exists in fast-block builds only
            if constexpr (chunk_config::use_fast_block_compression)
            {
                is_inside = static_cast<int>(
                    shrink_array::Proxy2<T>::get_tab_is_inside_list_to_compress(
                        shrink_array::Proxy2<T>::share_info_array->get_num_tab(this->registry_key()))[ind]
                        .load(std::memory_order_acquire));
            }
            int count = -1; // the counters exist in counter builds only
            if constexpr (chunk_config::use_compression_with_counter)
            {
                const auto& chunk = chunk_list[ind];
                count             = static_cast<int>(second ? chunk.get_nb_second_value() : chunk.get_nb_first_value());
            }
            CHUNK_LOG(debug, compress, "residual ArrayChunk2: chunk {} (element {}): is_inside={} count={}", ind,
                      static_cast<ind_type>(ind) * static_cast<ind_type>(this->size_policy_.size()), is_inside, count);
        }
    };
    report(tab_nb1, "first_value", false);
    if (!tab_nb2.empty())
    {
        if constexpr (is_not_bool)
        {
            const std::string at = "the second value " + value_repr(second_value);
            report(tab_nb2, at, true);
        }
        else
        {
            report(tab_nb2, "the second value true", true);
        }
    }
}

// Plain forwarder. Do not add an in-parallel-region test here: it hit an LLVM
// OpenMP runtime bug.
template <ArrayChunkElement T, typename ind_type>
void ArrayChunk2<T, ind_type>::compress()
{
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

    if constexpr (build_config::not_threaded)
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
        // A large shrink takes a team of its own, as clear_core()'s release
        // does: each thread parks into ITS rank's reuse list of this array's
        // geometry. Inside a region, or for a few chunks, the calling thread.
        const unsigned int my_sz = this->size_policy_.size();
        parallel_storage_init::for_each_chunk_in(
            first, last, [my_sz]() -> auto&
            { return manager_chunk2::get_list_block_free_thr_for_size(CHUNK_PMP_THREAD_RANK, my_sz); },
            [this](int j, auto& lbf)
            {
                if (chunk_list[j].owns_live_buffer())
                {
                    if constexpr (chunk_config::stats)
                    {
                        increm_chunk_teardown_clear<Chunk2<T>>();
                        // The size read here, not captured: stats builds only use it.
                        add_chunk_released<Chunk2<T>>(1, static_cast<long>(sizeof(T)) *
                                                             static_cast<long>(this->size_policy_.size()));
                    }
                    T* values_loc = chunk_list[j].values;
                    CHUNK_LEDGER_PARK(values_loc, "resize.shrink_park");
                    lbf.push_back(values_loc);
                    // Neutralise, as clear_core(): the buffer belongs to the
                    // reuse list now.
                    if constexpr (is_not_bool)
                    {
                        chunk_list[j].init_values_cst_val1();
                    }
                    else
                    {
                        chunk_list[j].init_values_cst_val_bool(0);
                    }
                }
            });
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
        const auto& sh_inf = *Proxy2<T>::share_info_array;

        const int num_tab = sh_inf.get_num_tab(this->registry_key());
        Proxy2<T>::resize_tab_is_int(num_tab, nb_chunk);
    }

    // The new chunks [old_nb_chunk, nb_chunk): a team of its own outside a
    // region, the calling thread inside one -- never an orphaned `omp for`,
    // which inside a region would leave 1/N of a thread's own array unconstructed.
    parallel_storage_init::for_each_chunk(nb_chunk - old_nb_chunk, [this, old_nb_chunk](int k)
                                          { this->construct_chunk(old_nb_chunk + k).init_after_constructor(); });

    // std::vector semantics for the regrown range [old_size, newsz): it reads
    // T{}. The chunks added above are fresh, but the one holding the OLD end
    // keeps its tail: after a shrink, the values written there would come back
    // on the next growth. Reset that tail through the normal write path, after
    // the reallocations. Serial: at most one chunk, and inside a region an
    // `omp for` would cover 1/N of it.
    if (newsz > old_size)
    {
        const auto csz = static_cast<ind_type>(this->chunk_size());
        // Up to the end of the OLD chunk count: chunks_for() keeps a chunk past
        // the last element, so when old_size falls on a chunk boundary that
        // spare chunk keeps stale values too.
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
