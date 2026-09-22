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
#include <algorithm> // std::find, the registered sizes
#include <cstddef>   // std::size_t (the reuse-pool byte counts)
#include <mutex>
#include <atomic> // for atomic_thread_fence, memory_order_seq_cst, atomic
#include <memory> // for unique_ptr
#include <vector> // for vector
#include <malloc.h>

#include "chunk/config.h"
#include "chunk/singleton.h"
#include "chunk/internal/manager_chunk.h"
#include "chunk/all_type_chunk.h"
#include "chunk/PMP_CHUNK.h"        // for CHUNK_PMP_MAX_THREADS, CHUNK_PMP_THREAD_RANK, __fast...
#include "chunk/chunk_define.h"     // for HAVE_EMBEDDED_COMPRESSION_WITH_OMPT
#include "chunk/internal/deferred_op_pool.h" // for reset_all
#include "chunk/internal/phase_guard.h"      // check-build guard of the phase protocol
#include "chunk/manage_ompt.h"      // the declarations of the five phase functions below

// Stays a directive, and cannot be anything else: a std-alloc build does not put
// fast_allocators/public on the include path at all, so the header is not merely
// unused there -- it is unreachable. The two report_all_arenas() calls below are
// guarded the same way for the same reason.
#ifndef CHUNK_STD_ALLOC
    #include "fast_allocators/slab_arena.h" // step-1 arena RSS instrumentation
#endif

namespace shrink_array
{

// Aggressive-compression scheduling is now a constexpr build mode driven by the
// CMake option HAVE_AGRESSIVE_COMPRESSION (-> chunk_config::have_agressive_compression),
// replacing the old raw DEBUG_AGRESSIVE_COMPRESSION macro.
//
//   have_agressive_compression == false (default, "release"):
//       the deferred sweep is triggered explicitly at end-of-parallel-phase
//       boundaries via compress_chunk_to_compress_potentially(true) below.
//   have_agressive_compression == true ("aggressive", historical behaviour):
//       the between-phase sweep is skipped; all compression falls on the
//       opportunistic in-loop path (heavier MT stress, better bug detection).
//
// The two if constexpr guards below select the between-phase sweep only in the
// release mode, i.e. when !have_agressive_compression.

void begin_parallel_phase()
{
    if constexpr (build_config::use_openmp)
    {
        __fast_omp_rank = CHUNK_PMP_THREAD_RANK;
    }
    if constexpr (phase_guard::active)
        phase_guard::arm();
}

void finalize_parallel_phase_only_flush()
{
    if constexpr (build_config::use_openmp && chunk_config::is_shrink_enabled)
    {
// Barrier + flush must both happen inside the same enclosing
// #pragma omp parallel region:
//  - barrier: all threads finish enqueueing deferred ops before any
//    thread starts flushing (required because the surrounding
//    #pragma omp for are typically `nowait`)
//  - flush: each thread drains its OWN queue using its own rank,
//    which is stable inside the current parallel region
#pragma omp barrier

        auto& ens_type_chunk = SingletonChk<AllTypeChunk>::getRef();
        ens_type_chunk.flush_all_deferred(true);

        // Restored: empirically load-bearing (a checksum corruption reappeared
        // when this was removed). Do NOT remove — the enclosing OpenMP barrier
        // does not, in practice, subsume the ordering this fence provides here.
        std::atomic_thread_fence(std::memory_order_seq_cst);
    }
}

void finalize_parallel_phase_after_flush()
{
    if constexpr (phase_guard::active)
        phase_guard::next_phase();
    // (Historical breadcrumb: the is_shrink=0 CMake-variant bug was diagnosed at
    //  this entry point; fixed by the per-variant chunk_v1/v2/v2a targets.)
    if constexpr (build_config::use_openmp && chunk_config::is_shrink_enabled)
    {
        auto& ens_type_chunk = SingletonChk<AllTypeChunk>::getRef();
        ens_type_chunk.destroy_after_flush(false);

        reset_all();
        if constexpr (!chunk_config::have_agressive_compression)
        {
            if (ens_type_chunk.have_chunk_to_compress_potentially())
                ens_type_chunk.compress_chunk_to_compress_potentially(true);
        }
        // Restored: empirically load-bearing (see the checksum regression note in
        // finalize_parallel_phase_only_flush). Do NOT remove.
        std::atomic_thread_fence(std::memory_order_seq_cst);
#ifndef CHUNK_STD_ALLOC
        fast_alloc::report_all_arenas("after_flush");
#endif
    }
}

void finalize_parallel_phase()
{
    if constexpr (phase_guard::active)
        phase_guard::next_phase();
    if constexpr (build_config::use_openmp && chunk_config::is_shrink_enabled)
    {
        auto& ens_type_chunk = SingletonChk<AllTypeChunk>::getRef();
        ens_type_chunk.flush_all_deferred(false);

        reset_all();

        if constexpr (!chunk_config::have_agressive_compression)
        {
            // Drain the deferred-compression backlog here, between parallel
            // phases, where it runs parallel and with no concurrent writers
            // (no start() barrier). Condition was inverted (compressed only
            // when there was NOTHING to compress); compress when there IS
            // something.
            if (ens_type_chunk.have_chunk_to_compress_potentially())
                ens_type_chunk.compress_chunk_to_compress_potentially(true);
        }
        // Restored: empirically load-bearing (see the checksum regression note in
        // finalize_parallel_phase_only_flush). Do NOT remove.
        std::atomic_thread_fence(std::memory_order_seq_cst);
#ifndef CHUNK_STD_ALLOC
        fast_alloc::report_all_arenas("finalize_phase");
#endif
    }
}

bool finalize_barrier_phase()
{
    if constexpr (build_config::use_openmp && chunk_config::is_shrink_enabled)
    {
        // NOT const: flush_all_deferred() mutates the managers it drains.
        auto&      ens_type_chunk = SingletonChk<AllTypeChunk>::getRef();
        const bool flushed        = ens_type_chunk.flush_all_deferred(true);

        // Restored: empirically load-bearing (see the checksum regression note in
        // finalize_parallel_phase_only_flush). Do NOT remove.
        std::atomic_thread_fence(std::memory_order_seq_cst);
        return flushed;
    }
    return false;
}

// pushManagerChunk<CHUNK_T, T> is now defined in chunk/all_type_chunk_inline.h
// (template definition belongs in a header; the .cpp keeps only the non-template
// fillManagerChunk<CHUNK_T>() and the runtime dispatch methods).
// The hard-coded fillManagerChunk<CHUNK_T>() list was removed in Stage 5b;
// registration is now fillManagerChunk<CHUNK_T, Types>() in
// all_type_chunk_inline.h, driven by the consumer's type tuple.

void AllTypeChunk::resize_chunk_to_compress_potentially() const
{
    for (auto& elm : getListTypeChunk())
    {
        elm->resize_chunk_to_compress_potentially();
    }
}

void AllTypeChunk::force_chunk_to_compress() const
{
    for (auto& elm : getListTypeChunk())
    {
        elm->force_chunk_to_compress();
    }
}

void AllTypeChunk::compress_chunk_to_compress_potentially(bool apply) const
{
    for (auto& elm : getListTypeChunk())
    {
        elm->compress_chunk_to_compress_potentially(apply);
    }
}
bool AllTypeChunk::have_chunk_to_compress_potentially() const
{
    const int max_thr = CHUNK_PMP_MAX_THREADS;

    for (const auto& elm : getListTypeChunk())
    {
        if (elm->have_chunk_to_compress_potentially(max_thr))
        {
            return true;
        }
    }
    return false;
}


template <typename Derived>
void AllTypeChunkParallelInterface<Derived, true>::resize_list_nb_chunk_thr() const
{
    for (auto& elm : static_cast<const Derived&>(*this).getListTypeChunk())
    {
        elm->resize_list_nb_chunk_thr();
    }
}

template <typename Derived>
void AllTypeChunkParallelInterface<Derived, true>::destroy_after_flush(bool in_parallel)
{
    for (auto& elm : static_cast<const Derived&>(*this).getListTypeChunk())
    {
        elm->destroy_after_flush(in_parallel);
    }
}

template <typename Derived>
bool AllTypeChunkParallelInterface<Derived, true>::flush_all_deferred(bool in_parallel)
{
    bool have_flushed = false;

    // Only the serial flush owns the global flag (see
    // ManagerChunkParallelImpl::flush_all_deferred).
    const bool toggle_flag = !in_parallel || g_barrier_flush_legacy;

    for (auto& elm : static_cast<const Derived&>(*this).getListTypeChunk())
    {
        elm->flush_all_deferred(in_parallel, have_flushed);
    }

    if (toggle_flag)
        __global_need_flush_deferred.store(false, std::memory_order_relaxed);

    // #pragma omp barrier // dead lock with ompt ?

    if (have_flushed && !in_parallel)
    {
        destroy_after_flush(false);
    }

    //  #pragma omp barrier // dead lock with ompt

    if (toggle_flag)
        __global_need_flush_deferred.store(true, std::memory_order_relaxed);

    return have_flushed;
}

template <typename Derived>
void AllTypeChunkParallelInterface<Derived, true>::load_balance_list_block_free() const
{
    for (auto& elm : static_cast<const Derived&>(*this).getListTypeChunk())
    {
        elm->load_balance_list_block_free();
    }
}

template <typename Derived>
void AllTypeChunkParallelInterface<Derived, true>::resize_all_list_block_free_thr() const
{
    for (auto& elm : static_cast<const Derived&>(*this).getListTypeChunk())
    {
        elm->resize_all_list_block_free_thr();
    }
}

template <typename Derived>
void AllTypeChunkParallelInterface<Derived, true>::resize_list_chunk_list_thr() const
{
    for (auto& elm : static_cast<const Derived&>(*this).getListTypeChunk())
    {
        elm->resize_list_chunk_list_thr();
    }
}

template <typename Derived>
void AllTypeChunkParallelInterface<Derived, true>::init_manager(ManagerChunkInt& m, int block_size) const
{
    // The sequence ParamChunk::init_indexes() used to run over the whole
    // registry, for one manager, at its registration. Everything here is a
    // slot indexed by rank; nothing needs the rank's own thread. Reached
    // through Derived's alias so that the calls stay dependent: the other
    // thread model's interface does not declare them.
    typename Derived::manager_int_t& mm = m;
    mm.resize_all_list_block_free_thr();
    mm.resize_list_chunk_list_thr();
    mm.resize_list_nb_chunk_thr();
    // std-alloc: the two below are no-ops on the manager.
    mm.alloc_chunk_allocator_thr();
    mm.init_allocators(block_size);
}
template <typename Derived>
void AllTypeChunkParallelInterface<Derived, true>::alloc_chunk_allocator_thr() const
{
    // std-alloc: each manager's alloc_chunk_allocator_thr() is a no-op.
    for (auto& elm : static_cast<const Derived&>(*this).getListTypeChunk())
    {
        elm->alloc_chunk_allocator_thr();
    }
}

template <typename Derived>
void AllTypeChunkParallelInterface<Derived, true>::reserve_nb_chunk_int(int sz)
{
    for (auto& elm : static_cast<const Derived&>(*this).getListTypeChunk())
    {
        elm->reserve_nb_chunk_int(sz);
    }
}

std::size_t AllTypeChunk::reuse_pool_bytes() const
{
    std::size_t total = 0;
    for (const auto& elm : getListTypeChunk())
        total += elm->reuse_pool_bytes_int();
    return total;
}

std::size_t AllTypeChunk::trim_reuse_pools(std::size_t keep_per_thread)
{
    std::size_t freed = 0;
    for (const auto& elm : getListTypeChunk())
        freed += elm->trim_reuse_pools_int(keep_per_thread);

#if defined(__GLIBC__)
    // Give the pages back to the KERNEL, not just to malloc.
    //
    // free() returns a block to its arena's free lists and the resident size
    // does not move, so without this the trim only changes which of our two
    // layers holds the memory -- invisible from outside the process.
    if (freed != 0)
        malloc_trim(0);
#endif

    return freed;
}

void AllTypeChunk::print_reuse_pool_breakdown(const char* label) const
{
    for (const auto& elm : getListTypeChunk())
        elm->print_reuse_pool_breakdown_int(label);
}

void print_all_reuse_pool_breakdown(const char* label)
{
    SingletonChk<AllTypeChunk>::getRef().print_reuse_pool_breakdown(label);
}

template <typename Derived>
void AllTypeChunkParallelInterface<Derived, true>::clear_list_block_free_thr()
{
    // ONE region for all nine element types, four phases separated by
    // barriers -- see ManagerChunkParallelImpl::clear_list_block_free_* for
    // what each does and why it is independent per bucket / per slice.
    //
    //   screen  rank r's reuse lists -> bucket r          (per bucket)
    //   sort    sort bucket r, drop its own repeats       (per bucket)
    //   dedup   repeats across buckets, by address slice  (per slice = thread)
    //   free    release bucket r                          (per bucket)
    //
    // The screening used to run sequentially before the region (before that,
    // under an `omp single` inside it): one unordered_set had to see every
    // pointer of every rank before the first free. Sorting per bucket and
    // merging per address slice keeps that guarantee with every thread busy.
    //
    // Per-bucket phases stride the buckets over the team: a team smaller than
    // the bucket count still covers every bucket, and with a full team the
    // free phase releases rank r's blocks from thread r, which is what is
    // wanted -- glibc gives each thread its own arena and a block is cheapest
    // to release through the one it came from.
    //
    // Called at teardown, outside any other parallel region.
    const auto& types = static_cast<const Derived&>(*this).getListTypeChunk();

    std::vector<std::size_t> nb_buckets;
    nb_buckets.reserve(types.size());
    for (const auto& elm : types)
    {
        nb_buckets.push_back(elm->clear_list_block_free_prepare());
    }

#pragma omp parallel
    {
        const int thr_rank = CHUNK_PMP_THREAD_RANK;
        const int nthr     = CHUNK_PMP_NUM_THREADS; // 1 without OpenMP

        auto for_each_bucket = [&](auto&& phase)
        {
            for (std::size_t i = 0; i < types.size(); ++i)
            {
                for (std::size_t b = static_cast<std::size_t>(thr_rank); b < nb_buckets[i];
                     b += static_cast<std::size_t>(nthr))
                {
                    phase(types[i], static_cast<int>(b));
                }
            }
        };

        for_each_bucket([](const auto& elm, int b) { elm->clear_list_block_free_screen(b); });
#pragma omp barrier
        for_each_bucket([](const auto& elm, int b) { elm->clear_list_block_free_sort(b); });
#pragma omp barrier
        for (const auto& elm : types)
        {
            elm->clear_list_block_free_dedup(thr_rank, nthr);
        }
#pragma omp barrier
        for_each_bucket([](const auto& elm, int b) { elm->clear_list_block_free_thr(b); });
    }
}

template <typename Derived>
void AllTypeChunkParallelInterface<Derived, true>::delete_list_block_chunk_free_itself() const
{
    for (auto& elm : static_cast<const Derived&>(*this).getListTypeChunk())
    {
        elm->delete_list_block_chunk_free_itself();
    }
}

template <typename Derived>
bool AllTypeChunkParallelInterface<Derived, true>::need_load_balance_list_block_free() const
{
    for (const auto& elm : static_cast<const Derived&>(*this).getListTypeChunk())
    {
        if (elm->need_load_balance_list_block_free())
            return true;
    }

    return false;
}


template <typename Derived>
void AllTypeChunkParallelInterface<Derived, false>::init_manager(ManagerChunkInt& m, int block_size) const
{
    // std-alloc: the manager's init_allocator() is a no-op. Through Derived's
    // alias so that the call stays dependent (see the OpenMP body).
    typename Derived::manager_int_t& mm = m;
    mm.init_allocator(block_size);
}

template <typename Derived>
void AllTypeChunkParallelInterface<Derived, false>::delete_chunk_allocator() const
{
    // std-alloc: each manager's delete_chunk_allocator() is a no-op.
    for (auto& elm : static_cast<const Derived&>(*this).getListTypeChunk())
    {
        elm->delete_chunk_allocator();
    }
}

// One definition for both threading modes. The two used to be spelled out
// separately, character for character, in the days when the split was made with
// the preprocessor; the interface is now specialised on a bool instead.
template <typename Derived>
void AllTypeChunkDynInterface<Derived, true>::register_chunk_size(unsigned int size, int block_size)
{
    // Remembered first, so that a manager registered from now on gets it
    // (register_manager replays the list); then declared to the managers that
    // exist. Both under the registry mutex: a first array of a new type may be
    // registering on another thread.
    auto& reg = static_cast<Derived&>(*this);
    const std::lock_guard<std::mutex> lock(reg.registry_mutex_);
    if (std::find(reg.registered_sizes_.begin(), reg.registered_sizes_.end(), size) == reg.registered_sizes_.end())
        reg.registered_sizes_.push_back(size);
    for (auto& elm : reg.getListTypeChunk())
    {
        elm->register_chunk_size(size, block_size);
    }
}


void AllTypeChunk::destroy_block_chunk() const
{
    // std-alloc: each manager's destroy_block_chunk() is a no-op (NullValueAllocator).
    for (auto& elm : getListTypeChunk())
    {
        elm->destroy_block_chunk();
    }
}

bool set_max_malloc_arenas(int max_arenas)
{
#if defined(__GLIBC__)
    if (max_arenas <= 0)
        return false;
    return mallopt(M_ARENA_MAX, max_arenas) == 1;
#else
    (void)max_arenas;
    return false; // not glibc: no arena notion to cap
#endif
}

// Instantiate exactly the specialisation this build selected. The other one is
// never named, so its bodies are never instantiated -- which is what lets the
// OpenMP versions call manager virtuals that a serial build does not declare.
template struct AllTypeChunkParallelInterface<AllTypeChunk, build_config::use_openmp>;
template struct AllTypeChunkDynInterface<AllTypeChunk, build_config::chunk_dynamic>;

} // namespace shrink_array
