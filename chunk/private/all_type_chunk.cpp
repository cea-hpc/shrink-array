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

#include <algorithm> // std::ranges::find (the registered sizes), std::copy_n (the registry table),
                     // std::ranges::any_of (the manager fan-outs)
#include <cstddef>   // std::size_t (the reuse-pool byte counts)
#include <mutex>     // std::mutex, std::scoped_lock
#include <atomic>    // for atomic_thread_fence, memory_order_seq_cst, atomic
#include <memory>    // for unique_ptr
#include <optional>  // SerialFlushOwnership
#include <utility>   // std::move
#include <vector>    // for vector
#include <malloc.h>  // malloc_trim, mallopt

#include "chunk/config.h"                  // build_config::threaded / use_openmp, chunk_config
#include "chunk/singleton.h"               // SingletonChk
#include "chunk/internal/manager_chunk.h"  // ManagerChunkInt, ManagerChunkParallelImpl
#include "chunk/all_type_chunk.h"          // the declarations defined here
#include "chunk/internal/churn_counters.h" // churn::note_phase_end
#include "chunk/PMP_CHUNK.h" // CHUNK_PMP_THREAD_RANK / _RANK_BOUND / _NUM_THREADS, chunk_rank_cache, pmp::team, current_team
#include "chunk/internal/deferred_op_pool.h" // for reset_all
#include "chunk/internal/env_parse.h"        // env_number (CHUNK_RECLAIM_PHASE_KEEP)
#include "chunk/internal/phase_guard.h"      // check-build guard of the phase protocol
#include "chunk/manage_ompt.h"               // the declarations of the five phase functions below
#include "chunk/param_chunk.h"               // ParamChunk::init_indexes (ensure_ready)

// Must stay a directive: a std-alloc build does not put fast_allocators/public on the
// include path, so the header is unreachable there. Same for the two report_all_arenas() calls.
#if !CHUNK_OPT_STD_ALLOC
    #include "fast_allocators/slab_arena.h" // step-1 arena RSS instrumentation
#endif

namespace shrink_array
{
/// @brief Empty registry; warns once if the OMPT tool this build relies on is
///        not loaded (manage_ompt.h, check_ompt_tool_loaded).
AllTypeChunk::AllTypeChunk()
{
    // Weak: absent from a NO_OMPT_TOOL library (manage_ompt.h).
    if (&check_ompt_tool_loaded != nullptr)
    {
        check_ompt_tool_loaded();
    }
}

void AllTypeChunk::ensure_ready()
{
    static std::once_flag once;
    std::call_once(once,
                   []
                   {
                       auto& reg   = SingletonChk<AllTypeChunk>::get_ref();
                       bool  ready = false;
                       {
                           const std::scoped_lock lock(reg.registry_mutex_);
                           ready = reg.ready_;
                       }
                       if (!ready)
                       {
                           ParamChunk::init_indexes(true);
                       }
                   });
}

void AllTypeChunk::publish_manager(std::unique_ptr<ManagerChunkInt> manager)
{
    const auto count = static_cast<std::size_t>(nb_managers_.load(std::memory_order_relaxed));
    if (count == table_capacity_)
    {
        // Full (or none yet): a table twice as large, the published slots
        // copied, published before the count that will cover the new slot.
        const std::size_t capacity = table_capacity_ == 0 ? first_table_size : 2 * table_capacity_;
        auto              table    = std::make_unique<ManagerChunkInt*[]>(capacity);
        if (table_capacity_ != 0)
        {
            std::copy_n(tables_.back().get(), count, table.get());
        }
        table_.store(table.get(), std::memory_order_release);
        tables_.push_back(std::move(table));
        table_capacity_ = capacity;
    }
    owned_.push_back(std::move(manager));
    tables_.back()[count] = owned_.back().get();
    nb_managers_.store(static_cast<int>(count + 1), std::memory_order_release); // publishes the slot
    phase_guard::teams::arrays_exist.store(true, std::memory_order_release);    // two top-level teams now refused
}

namespace
{
// Cross-thread exclusion of the SERIAL deferred flush. Deliberately SEPARATE
// from g_need_flush_deferred, which is a re-entrancy guard with a
// sequential per-type toggle: turning THAT flag into a take-or-give-up lock
// would make an inner per-type flush give up whenever an outer level held it,
// silently dropping deferred writes. Value-initialised to clear (C++20).
std::atomic_flag g_serial_flush_owner{};

/// Releases g_serial_flush_owner on every exit path of the serial flush.
struct SerialFlushOwnership
{
    SerialFlushOwnership()                                       = default;
    SerialFlushOwnership(const SerialFlushOwnership&)            = delete;
    SerialFlushOwnership& operator=(const SerialFlushOwnership&) = delete;
    SerialFlushOwnership(SerialFlushOwnership&&)                 = delete;
    SerialFlushOwnership& operator=(SerialFlushOwnership&&)      = delete;

    ~SerialFlushOwnership()
    {
        g_serial_flush_owner.clear(std::memory_order_release);
    }
};
} // namespace

// chunk_config::aggressive_compression() (run time: CHUNK_AGGRESSIVE_COMPRESSION):
//   false (default): the deferred sweep runs at end-of-parallel-phase boundaries,
//       via compress_chunk_to_compress_potentially(true) below;
//   true: no between-phase sweep; all compression falls on the opportunistic
//       in-loop path (heavier MT stress, better bug detection).

void begin_parallel_phase()
{
    if constexpr (build_config::threaded)
    {
        // Pool backend, from a thread in no team of the pool (an application
        // thread): the application phase opens -- in_parallel() true, no team
        // of the pool -- until finalize_parallel_phase(). Nothing otherwise.
        pmp::enter_application_phase();
        chunk_rank_cache() = CHUNK_PMP_THREAD_RANK;
    }
    phase_guard::check_begin(); // finalize forgotten before this region: stops, in every build
    // One top-level team at a time (phase_guard::teams): each thread of a
    // top-level team takes its rank for the phase; a rank already held by
    // another thread is refused.
    if constexpr (build_config::use_openmp && build_config::has_not_ompt)
    {
        if (CHUNK_PMP_LEVEL == 1)
        {
            phase_guard::teams::manual_begin(g_fast_omp_rank);
        }
    }
}

void finalize_parallel_phase_only_flush()
{
    // One compression phase (churn_counters.h). This function runs on EVERY
    // thread of the region: count it once, on rank 0, or each phase counts
    // max_threads times on one shared atomic.
    if (CHUNK_PMP_THREAD_RANK == 0)
    {
        churn::note_phase_end();
    }
    // use_openmp, not threaded: the barrier is the application's team's,
    // which only OpenMP gives the library (the phase contract, manage_ompt.h).
    if constexpr (build_config::use_openmp && chunk_config::is_shrink_enabled)
    {
        // Barrier + flush must both happen inside the same enclosing
        // parallel region (the application's team, hence OpenMP only):
        //  - barrier: all threads finish enqueueing deferred ops before any
        //    thread starts flushing (required because the surrounding
        //    #pragma omp for are typically `nowait`)
        //  - flush: each thread drains its OWN queue using its own rank,
        //    which is stable inside the current parallel region
        pmp::current_team().barrier();

        auto& ens_type_chunk = SingletonChk<AllTypeChunk>::get_ref();
        ens_type_chunk.flush_all_deferred(true);

        // Empirically load-bearing (a checksum corruption appears without it).
        // Do NOT remove -- the enclosing OpenMP barrier does not, in practice,
        // subsume the ordering this fence provides here.
        std::atomic_thread_fence(std::memory_order_seq_cst);
    }
}

void finalize_parallel_phase_after_flush()
{
    phase_guard::teams::manual_end(); // the phase is over: the next one may be any team's
    if constexpr (phase_guard::tracked)
    {
        phase_guard::next_phase();
    }
    // use_openmp: a shape of the phase contract only OpenMP has (manage_ompt.h).
    if constexpr (build_config::use_openmp && chunk_config::is_shrink_enabled)
    {
        const auto& ens_type_chunk = SingletonChk<AllTypeChunk>::get_ref();
        ens_type_chunk.destroy_after_flush(false);

        reset_all();
        if (!chunk_config::aggressive_compression()) // run-time switch: chunk/config.h
        {
            if (ens_type_chunk.have_chunk_to_compress_potentially())
            {
                ens_type_chunk.compress_chunk_to_compress_potentially(true);
            }
        }
        // Empirically load-bearing (see the note in
        // finalize_parallel_phase_only_flush). Do NOT remove.
        std::atomic_thread_fence(std::memory_order_seq_cst);
#if !CHUNK_OPT_STD_ALLOC
        fast_alloc::report_all_arenas("after_flush");
#endif
    }
}

void finalize_parallel_phase()
{
    // Inside an active parallel region -- a parallel_phase() opened within a phase, by
    // a component that knows nothing of it -- the enclosing phase is not over:
    // its other threads may still be writing, and the flush below drains every
    // rank's queue without a lock. The enclosing phase's own end finalizes
    // (manage_ompt.h; nested_region_validation D). A team, not
    // CHUNK_PMP_IN_PARALLEL_REGION: the pool backend's application phase is
    // in_parallel() too, and this is where it closes.
    if (pmp::active_level() > 0)
    {
        return;
    }
    phase_guard::teams::manual_end(); // the phase is over: the next one may be any team's
    // The application's threads stopped writing (the contract): close its
    // phase first, so the flush and the compression below run on teams of
    // the pool (pool backend; nothing otherwise).
    pmp::leave_application_phase();
    if constexpr (phase_guard::tracked)
    {
        phase_guard::next_phase();
    }
    if constexpr (build_config::threaded && chunk_config::is_shrink_enabled)
    {
        auto& ens_type_chunk = SingletonChk<AllTypeChunk>::get_ref();
        ens_type_chunk.flush_all_deferred(false);

        reset_all();
        // Pool backend: the op pools of the application threads that exited
        // in this phase; their ops were just replayed (deferred_op_pool.h).
        retire_exited_thread_pools();

        if (!chunk_config::aggressive_compression()) // run-time switch: chunk/config.h
        {
            // Drain the deferred-compression backlog here, between parallel
            // phases: parallel, with no concurrent writers (no start() barrier).
            if (ens_type_chunk.have_chunk_to_compress_potentially())
            {
                ens_type_chunk.compress_chunk_to_compress_potentially(true);
            }
        }
        // Empirically load-bearing (see the note in
        // finalize_parallel_phase_only_flush). Do NOT remove.
        std::atomic_thread_fence(std::memory_order_seq_cst);
#if !CHUNK_OPT_STD_ALLOC
        fast_alloc::report_all_arenas("finalize_phase");
#endif
    }
}

bool finalize_barrier_phase()
{
    // use_openmp: a shape of the phase contract only OpenMP has (manage_ompt.h).
    if constexpr (build_config::use_openmp && chunk_config::is_shrink_enabled)
    {
        // NOT const: flush_all_deferred() mutates the managers it drains.
        auto&      ens_type_chunk = SingletonChk<AllTypeChunk>::get_ref();
        const bool flushed        = ens_type_chunk.flush_all_deferred(true);

        // Empirically load-bearing (see the note in
        // finalize_parallel_phase_only_flush). Do NOT remove.
        std::atomic_thread_fence(std::memory_order_seq_cst);
        return flushed;
    }
    return false;
}

// pushManagerChunk<ChunkT, T> and fill_manager_chunk<ChunkT, Types>() are
// defined in chunk/internal/all_type_chunk_inline.h (template definitions belong in a
// header); registration is driven by the consumer's type tuple.

void AllTypeChunk::resize_chunk_to_compress_potentially() const
{
    for (const auto& elm : get_list_type_chunk())
    {
        elm->resize_chunk_to_compress_potentially();
    }
}

void AllTypeChunk::force_chunk_to_compress() const
{
    for (const auto& elm : get_list_type_chunk())
    {
        elm->force_chunk_to_compress();
    }
}

void AllTypeChunk::compress_chunk_to_compress_potentially(bool apply) const
{
    // Only the types with a candidate. The callers' gate,
    // have_chunk_to_compress_potentially(), is true as soon as ONE type has
    // one, and each type's pass costs a fork/join even on empty lists.
    const int                     max_thr = CHUNK_PMP_RANK_BOUND;
    std::vector<ManagerChunkInt*> work;
    for (const auto& elm : get_list_type_chunk())
    {
        if (elm->have_chunk_to_compress_potentially(max_thr))
        {
            work.push_back(elm);
        }
    }
    if (work.empty())
    {
        return;
    }

    // Inside a region (or without fast block compression, where the pass is
    // a no-op): each type's own path, which knows what to do there.
    if (!chunk_config::use_fast_block_compression || CHUNK_PMP_IN_PARALLEL_REGION)
    {
        for (auto* elm : work)
        {
            elm->compress_chunk_to_compress_potentially(apply);
        }
        return;
    }

    // ONE region for every type (precedent: clear_list_block_free_thr), not
    // one per type -- and per second value for Chunk2. Each thread runs every
    // type's per-rank body for its own rank; those touch rank-local lists
    // only, so the types need no barrier between them. Then the serial mop-up
    // of the ranks past the team, for the variants whose pass had one.
    for (auto* elm : work)
    {
        elm->sweep_prepare();
    }

    chunk_config::OMPTcc<chunk_config::has_ompt>::update(false);
    int team = 0;
    pmp::team(pmp::Options{},
              [&](pmp::Team& /*team*/)
              {
                  const int thr = CHUNK_PMP_THREAD_RANK;
                  if (thr == 0)
                  {
                      team = CHUNK_PMP_NUM_THREADS;
                  }
                  for (auto* elm : work)
                  {
                      elm->sweep_rank(thr, apply);
                  }
              });
    for (auto* elm : work)
    {
        if (elm->sweep_mops_up(apply))
        {
            for (int thr = team; thr < CHUNK_PMP_RANK_BOUND; ++thr)
            {
                elm->sweep_rank(thr, apply);
            }
        }
    }
    chunk_config::OMPTcc<chunk_config::has_ompt>::update(true);

    // CHUNK_RECLAIM_PHASE_KEEP=K (unset: off): the pass's freed buffers
    // went back to their slabs, which stay resident until a sweep. Give
    // the empty ones back now, keeping K per allocator instance for the
    // next phase's materialisations (fewer madvise / fault-in round
    // trips). Serial and quiescent here: the pass's region has joined.
    static const long phase_keep = env_number<long>("CHUNK_RECLAIM_PHASE_KEEP", -1L);
    if (phase_keep >= 0)
    {
        for (auto* elm : work)
        {
            (void)elm->sweep_empty_slabs_int(static_cast<std::size_t>(phase_keep));
        }
    }
}

bool AllTypeChunk::have_chunk_to_compress_potentially() const
{
    const int max_thr = CHUNK_PMP_RANK_BOUND;

    return std::ranges::any_of(get_list_type_chunk(),
                               [max_thr](const auto* elm) { return elm->have_chunk_to_compress_potentially(max_thr); });
}


template <typename Derived>
void AllTypeChunkParallelInterface<Derived, true>::destroy_after_flush(bool in_parallel) const
{
    for (const auto& elm : static_cast<const Derived&>(*this).get_list_type_chunk())
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
    const bool toggle_flag = !in_parallel;

    // Two threads must not run the SERIAL flush at once: each drains every
    // rank and reclaims buffers, so concurrent passes race on the same ops.
    // The loser returns -- the owner is already flushing everything. acquire
    // pairs with the owner's release on exit: this thread sees that flush's
    // effects. The in-region flush is collaborative by design and not gated.
    if (!in_parallel && g_serial_flush_owner.test_and_set(std::memory_order_acquire))
    {
        return false;
    }
    std::optional<SerialFlushOwnership> serial_ownership;
    if (!in_parallel)
    {
        serial_ownership.emplace();
    }

    for (const auto& elm : static_cast<const Derived&>(*this).get_list_type_chunk())
    {
        elm->flush_all_deferred(in_parallel, have_flushed);
    }

    if (toggle_flag)
    {
        g_need_flush_deferred.store(false, std::memory_order_relaxed);
    }

    // #pragma omp barrier // dead lock with ompt ?

    if (have_flushed && !in_parallel)
    {
        destroy_after_flush(false);
    }

    //  #pragma omp barrier // dead lock with ompt

    if (toggle_flag)
    {
        g_need_flush_deferred.store(true, std::memory_order_release);
    }

    return have_flushed;
}

template <typename Derived>
void AllTypeChunkParallelInterface<Derived, true>::load_balance_list_block_free() const
{
    for (const auto& elm : static_cast<const Derived&>(*this).get_list_type_chunk())
    {
        elm->load_balance_list_block_free();
    }
}

template <typename Derived>
void AllTypeChunkParallelInterface<Derived, true>::resize_all_list_block_free_thr() const
{
    for (const auto& elm : static_cast<const Derived&>(*this).get_list_type_chunk())
    {
        elm->resize_all_list_block_free_thr();
    }
}

template <typename Derived>
void AllTypeChunkParallelInterface<Derived, true>::init_manager(ManagerChunkInt& manager, int block_size) const
{
    // Per-manager setup, at its registration. Everything here is a
    // slot indexed by rank; nothing needs the rank's own thread. Reached
    // through Derived's alias so that the calls stay dependent: the other
    // thread model's interface does not declare them.
    typename Derived::ManagerIntT& mm = manager;
    mm.resize_all_list_block_free_thr();
    // std-alloc: the two below are no-ops on the manager.
    mm.alloc_chunk_allocator_thr();
    mm.init_allocators(block_size);
}

template <typename Derived>
void AllTypeChunkParallelInterface<Derived, true>::alloc_chunk_allocator_thr() const
{
    // std-alloc: each manager's alloc_chunk_allocator_thr() is a no-op.
    for (const auto& elm : static_cast<const Derived&>(*this).get_list_type_chunk())
    {
        elm->alloc_chunk_allocator_thr();
    }
}

std::size_t AllTypeChunk::reuse_pool_bytes() const
{
    std::size_t total = 0;
    for (const auto& elm : get_list_type_chunk())
    {
        total += elm->reuse_pool_bytes_int();
    }
    return total;
}

std::size_t AllTypeChunk::trim_reuse_pools(std::size_t keep_per_thread) const
{
    std::size_t freed = 0;
    for (const auto& elm : get_list_type_chunk())
    {
        freed += elm->trim_reuse_pools_int(keep_per_thread);
    }

#ifdef __GLIBC__
    // Give the pages back to the KERNEL, not just to malloc: free() alone leaves
    // the resident size unchanged, the memory only moves from our pool to malloc's.
    if (freed != 0)
    {
        malloc_trim(0);
    }
#endif

    return freed;
}

void AllTypeChunk::print_reuse_pool_breakdown(const char* label) const
{
    for (const auto& elm : get_list_type_chunk())
    {
        elm->print_reuse_pool_breakdown_int(label);
    }
}

void print_all_reuse_pool_breakdown(const char* label)
{
    SingletonChk<AllTypeChunk>::get_ref().print_reuse_pool_breakdown(label);
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
    // No sequential pre-pass: sorting per bucket and merging per address
    // slice keeps the "every pointer seen before the first free" guarantee
    // with every thread busy.
    //
    // Per-bucket phases stride the buckets over the team: a team smaller than
    // the bucket count still covers every bucket, and with a full team the
    // free phase releases rank r's blocks from thread r, which is what is
    // wanted -- glibc gives each thread its own arena and a block is cheapest
    // to release through the one it came from.
    //
    // Called at teardown, outside any other parallel region.
    const auto& types = static_cast<const Derived&>(*this).get_list_type_chunk();

    std::vector<std::size_t> nb_buckets;
    nb_buckets.reserve(types.size());
    for (const auto& elm : types)
    {
        nb_buckets.push_back(elm->clear_list_block_free_prepare());
    }

    pmp::team(pmp::Options{},
              [&](pmp::Team& team)
              {
                  const int thr_rank = CHUNK_PMP_THREAD_RANK;
                  const int nthr     = CHUNK_PMP_NUM_THREADS; // 1 without OpenMP

                  const auto for_each_bucket = [&](auto&& phase)
                  {
                      for (std::size_t i = 0; i < types.size(); ++i)
                      {
                          for (auto b = static_cast<std::size_t>(thr_rank); b < nb_buckets[i];
                               b += static_cast<std::size_t>(nthr))
                          {
                              phase(types[i], static_cast<int>(b));
                          }
                      }
                  };

                  for_each_bucket([](const auto& elm, int bucket) { elm->clear_list_block_free_screen(bucket); });
                  team.barrier();
                  for_each_bucket([](const auto& elm, int bucket) { elm->clear_list_block_free_sort(bucket); });
                  team.barrier();
                  for (const auto& elm : types)
                  {
                      elm->clear_list_block_free_dedup(thr_rank, nthr);
                  }
                  team.barrier();
                  for_each_bucket([](const auto& elm, int bucket) { elm->clear_list_block_free_thr(bucket); });
              });
}

template <typename Derived>
void AllTypeChunkParallelInterface<Derived, true>::delete_list_block_chunk_free_itself() const
{
    for (const auto& elm : static_cast<const Derived&>(*this).get_list_type_chunk())
    {
        elm->delete_list_block_chunk_free_itself();
    }
}

template <typename Derived>
bool AllTypeChunkParallelInterface<Derived, true>::need_load_balance_list_block_free() const
{
    return std::ranges::any_of(static_cast<const Derived&>(*this).get_list_type_chunk(),
                               [](const auto* elm) { return elm->need_load_balance_list_block_free(); });
}

template <typename Derived>
void AllTypeChunkParallelInterface<Derived, false>::init_manager(const ManagerChunkInt& manager, int block_size) const
{
    // std-alloc: the manager's init_allocator() is a no-op. Through Derived's
    // alias so that the call stays dependent (see the OpenMP body).
    const typename Derived::ManagerIntT& mm = manager;
    mm.init_allocator(block_size);
}

template <typename Derived>
void AllTypeChunkParallelInterface<Derived, false>::delete_chunk_allocator() const
{
    // std-alloc: each manager's delete_chunk_allocator() is a no-op.
    for (const auto& elm : static_cast<const Derived&>(*this).get_list_type_chunk())
    {
        elm->delete_chunk_allocator();
    }
}

// One definition for both threading modes: the interface is specialised on a
// bool.
template <typename Derived>
void AllTypeChunkDynInterface<Derived, true>::register_chunk_size(unsigned int size, int block_size)
{
    // Remembered first, so that a manager registered from now on gets it
    // (register_manager replays the list); then declared to the managers that
    // exist. Both under the registry mutex: a first array of a new type may be
    // registering on another thread.
    auto&                  reg = static_cast<Derived&>(*this);
    const std::scoped_lock lock(reg.registry_mutex_);
    if (std::ranges::find(reg.registered_sizes_, size) == reg.registered_sizes_.end())
    {
        reg.registered_sizes_.push_back(size);
    }
    for (const auto& elm : reg.get_list_type_chunk())
    {
        elm->register_chunk_size(size, block_size);
    }
}


void AllTypeChunk::destroy_block_chunk() const
{
    // std-alloc: each manager's destroy_block_chunk() is a no-op (NullValueAllocator).
    for (const auto& elm : get_list_type_chunk())
    {
        elm->destroy_block_chunk();
    }
}

bool set_max_malloc_arenas(int max_arenas)
{
#ifdef __GLIBC__
    if (max_arenas <= 0)
    {
        return false;
    }
    return mallopt(M_ARENA_MAX, max_arenas) == 1;
#else
    (void)max_arenas;
    return false; // not glibc: no arena notion to cap
#endif
}

// Instantiate exactly the specialisation this build selected. The other one is
// never named, so its bodies are never instantiated -- which is what lets the
// OpenMP versions call manager virtuals that a serial build does not declare.
template class AllTypeChunkParallelInterface<AllTypeChunk, build_config::threaded>;
template struct AllTypeChunkDynInterface<AllTypeChunk, build_config::chunk_dynamic>;
} // namespace shrink_array
