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

// Out-of-line bodies of the re-chunk's refill (array_chunk_rechunk_refill.h).
// Included by it, itself included at the end of array_chunk_base.h through
// array_chunk_rechunk_inline.h, after the class: the arrays are complete here.

#include <algorithm>   // std::min / std::max (the slice count), std::fill_n, std::ranges::fill
#include <atomic>      // fetch_add on the release counters and census totals
#include <chrono>      // the slice boundaries' timers
#include <cstddef>     // std::size_t, std::ptrdiff_t
#include <cstring>     // std::memcpy / std::memcmp (the bulk read)
#include <string_view> // the rebuild schedule's text, in the trace
#include <type_traits> // std::is_same_v, is_trivially_copyable_v, has_unique_object_representations_v
#include <utility>     // std::as_const
#include <vector>      // the refill buffers

#include "chunk/internal/PMP_CHUNK.h"                  // get_thr_rank_fast, pmp::team, chunk_pmp_*_schedule
#include "chunk/array_chunk_range_ops.h"               // copy_range -- the public bulk fill
#include "chunk/bit.h"                                 // is_bool_like_v (the bulk read gate)
#include "chunk/internal/chunk_traits.h"               // is_zero_any_family_v, has_motif_state_v
#include "chunk/config.h"                              // build_config::threaded, chunk_config::use_zero*
#include "chunk/internal/array_chunk_rechunk_refill.h" // Refill, the class these bodies belong to
#include "chunk/internal/array_chunk_rechunk_trace.h"  // rechunk_trace_on, rss_mib_now, g_slice_times
#include "chunk/internal/drain_probe.h"                // g_drain_probe (the boundary drain's split)
#include "chunk/internal/chunk_size_policy.h"          // set_thread_chunk_size, DynChunkSizeGuard
#include "chunk/internal/chunk_stats.h"                // increm_chunk_rechunk_* (CHUNK_STATS)
#include "chunk/internal/compression_manager.h"        // CompressionManager<>::FlushReplayGuard
#include "chunk/internal/cst_slot.h"                   // CstSlot (zero_any's type-of-constant slot)
#include "chunk/internal/invariant_check.h"            // CHUNK_INVARIANT_FAILED (the bulk read check)
#include "chunk/log.h"                                 // CHUNK_LOG (the rebuild schedule, the slices)
#include "chunk/vector_range_ops.h"                    // assume_drained

namespace shrink_array
{

template <typename Derived>
RechunkSupport<Derived, true>::Refill::Refill(Derived&     src,
                                              Derived&     tmp,
                                              int          old_size,
                                              int          new_size,
                                              unsigned int old_key)
    : src_(src)
    , tmp_(tmp)
    , old_size_(old_size)
    , new_size_(new_size)
    , old_key_(old_key)
    , total_(src.m_size_)
    , n_new_(tmp.nb_chunk_)
    , trace_on_(internal::rechunk_trace_on())
    , trace_mem_on_(internal::rechunk_trace_mem_on())
    , slice_trace_(internal::rechunk_slice_trace_on())
    , ind_old2_(internal::rechunk_second_ind<Derived, ChunkType>(src))
    , inf_new2_(internal::rechunk_second_info<Derived, ChunkType>(tmp))
    , aligned_coarsening_(new_size % old_size == 0)
    , n_slices_(slice_count())
{
    if constexpr (releases_sources)
    {
        if (!aligned_coarsening_)
        {
            old_done_ = std::vector<std::atomic<int>>(static_cast<std::size_t>(src.nb_chunk_));
        }
    }
}

template <typename Derived>
int RechunkSupport<Derived, true>::Refill::slice_count() const noexcept
{
    // SLICED rebuild. Inside one region the refill releases each source chunk as
    // soon as it is transcribed (release_sources), but a block freed by a rank
    // that does not own its slab is only parked in that rank's reuse list
    // (walking the owner's slab chains from inside a region is what the
    // owner-routed drain avoids). Unsliced, no page comes back before the
    // drain after the WHOLE rebuild: peak = old + new.
    //
    // Draining at each slice boundary routes that slice's parked blocks home,
    // empties the old slabs of the range just consumed and gives their pages
    // back BEFORE the next slice materialises: peak ~ old + new/slices.
    //
    // 16 slices with the fast allocator and threads, for every array that
    // releases its sources (releases_sources: not ArrayChunkBits, which has
    // nothing to give back between slices -- the extra regions would be pure
    // cost). The std allocator needs no slice: each
    // release is a free() the next allocation reuses.
    if constexpr (releases_sources && build_config::use_fast_alloc)
    {
        // 16: cuts the peak above the pre-re-chunk level about fourfold;
        // more slices gain little. The remainder is the two geometries
        // coexisting at slab granularity, insensitive to slicing. Index
        // order is already nearly slab-local (slab-order rebuild gains
        // nothing). Measured on Milan (zero_dyn, recycled slabs kept, five
        // runs each): at 64 threads 32, 64 and 128 slices are 1.3%, 5.2%
        // and 6.5% slower, 8 is 1.8% faster but +0.5% of memory; at 32
        // threads all are within 0.7%, 64 excepted (+1.7%). Fixed: the
        // former CHUNK_RECHUNK_SLICES is gone.
        // Not worth slicing an array too small to empty a slab per slice.
        return std::min(16, std::max(1, n_new_ / 4096));
    }
    else
    {
        return 1;
    }
}

template <typename Derived>
void RechunkSupport<Derived, true>::Refill::run()
{
    // Geometry is per-array (src.size_=old, tmp.size_=new), so operator[]
    // maps correctly without the global. Only alloc_chunk_count()
    // (= thread_chunk_size(), TLS) needs the size, set per thread in
    // rebuild_one(). Seed the process-wide global ONCE (final value = new);
    // NEVER set the global inside the loop — it would race under parallelism.
    set_active_chunk_size(static_cast<unsigned int>(new_size_));

    // Parallel rebuild: each thread takes disjoint new chunks and drives the
    // write path on its own rank; the schedule (see
    // internal::rebuild_schedule()) balances cst1-skip (cheap) vs materialise
    // (costly). Geometry is per-array so no global race; only per-thread TLS
    // + per-rank allocator/compression state are touched.
    //
    // The OMPT parallel-end finalize is suppressed (g_in_rechunk_rebuild,
    // below): a region's end must NOT flush/compress the half-written array
    // (it corrupts grow regions that span several old chunks). rechunk_to
    // flushes at a quiescent point after the refill. g_in_rechunk_rebuild is
    // held across the WHOLE sliced loop, drains included: a slice boundary is
    // precisely "half rebuilt".
    //
    // The rebuild loop is schedule(runtime): set the chosen schedule on this
    // task (the slice regions inherit it) and restore the caller's after the
    // slices. Both calls are no-ops in a serial build (PMP_CHUNK.h).
    const ChunkPmpSchedule caller_schedule = chunk_pmp_get_schedule();
    if constexpr (build_config::threaded)
    {
        const auto& rs = internal::rebuild_schedule();
        chunk_pmp_set_schedule(rs.schedule);
        if (trace_on_)
        {
            CHUNK_LOG(debug, rechunk, "rebuild schedule: {}", std::string_view{rs.text.data()});
        }
    }
    g_in_rechunk_rebuild.store(true, std::memory_order_release);
    // rechunk=trace: the RSS after each slice (internal::rechunk_slice_trace_on).
    if (slice_trace_)
    {
        rss_start_ = internal::rss_mib_now();
        CHUNK_LOG(trace, rechunk, "slice 0/{} start       rss={} MB", n_slices_, rss_start_);
    }
    if (trace_on_)
    {
        internal::g_slice_times.count = n_slices_;
    }
    for (int sl = 0; sl < n_slices_; ++sl)
    {
        run_slice(sl);
    }
    if (slice_trace_)
    {
        const long rss_mib = internal::rss_mib_now();
        CHUNK_LOG(trace, rechunk, "slice {}/{} end         rss={} MB  ({:+} since start)", n_slices_, n_slices_,
                  rss_mib, rss_mib - rss_start_);
    }
    // The last boundary's recycled slabs the last slice did not take stay
    // on the arena's recycled lists, resident: SlabArena::allocate() takes
    // them first, so the next writes and the next re-chunk reuse their
    // pages instead of faulting fresh ones. Measured on Milan (zero_dyn,
    // 64 threads): 11% faster than advising them away, for +0.1% of
    // memory at the end of the run. (The former CHUNK_RECHUNK_KEEP_RECYCLED
    // and _MB chose between the two.)
    g_in_rechunk_rebuild.store(false, std::memory_order_release);
    chunk_pmp_set_schedule(caller_schedule); // the caller's, untouched
    if (trace_mem_on_)                       // a serial count inside the refill phase
    {
        log_census();
    }
}

template <typename Derived>
void RechunkSupport<Derived, true>::Refill::run_slice(int slice)
{
    const int c0 = static_cast<int>((static_cast<long>(n_new_) * slice) / n_slices_);
    const int c1 = static_cast<int>((static_cast<long>(n_new_) * (slice + 1)) / n_slices_);
    if (trace_on_)
    {
        probe_.begin();
    }
    pmp::team(pmp::Options{},
              [&](pmp::Team& team)
              {
                  const SliceClock::time_point entry = trace_on_ ? SliceClock::now() : SliceClock::time_point{};
                  // Apply the rebuild's arming writes DIRECTLY, never via the
                  // deferred path: deferred arming writes would fill tmp's op map,
                  // and a rehash frees bucket storage an enqueued op still targets
                  // (use-after-free at the flush). Under in_flush_replay each
                  // write goes straight through write_body (materialise + store,
                  // no enqueue). Safe: each thread rebuilds DISJOINT new chunks,
                  // so the bypassed gate guards nothing. Per-thread
                  // (thread_local), so it lives inside the region.
                  const typename CompressionManager<T, ChunkType>::FlushReplayGuard rechunk_replay_guard;
                  std::vector<T>               buf(static_cast<std::size_t>(new_size_));
                  Tally                        tally; // this thread's fold counts for this slice
                  const SliceClock::time_point loop_start = trace_on_ ? SliceClock::now() : SliceClock::time_point{};
                  // schedule(runtime): guided,16 (internal::rebuild_schedule).
                  // nowait + the explicit barrier below: the same closing barrier,
                  // with each thread's loop end observable (rechunk=debug).
                  team.for_runtime<int>(c0, c1, [&](int c) { rebuild_one(c, buf, tally); }, pmp::nowait);
                  if (trace_on_)
                  {
                      probe_.thread_done(entry, loop_start, SliceClock::now());
                  }
                  // One RMW per thread and slice, not one per folded chunk; and
                  // none at all unless the census, their only reader, is on.
                  if (trace_mem_on_)
                  {
                      merge(tally);
                  }
                  team.barrier();
                  team.masked(
                      [&]
                      {
                          if (trace_on_)
                          {
                              probe_.past_barrier = SliceClock::now();
                          }
                      });
                  // n_slices_ > 1 only in these builds (slice_count()).
                  if constexpr (build_config::use_fast_alloc && build_config::threaded)
                  {
                      if (slice + 1 < n_slices_)
                      {
                          slice_boundary(team);
                      }
                  }
                  team.masked(
                      [&]
                      {
                          if (trace_on_)
                          {
                              probe_.past_boundary = SliceClock::now();
                          }
                      });
              });
    if (trace_on_)
    {
        probe_.accumulate(internal::g_slice_times);
    }
    if (slice_trace_ && slice + 1 < n_slices_)
    {
        const long rss_mib = internal::rss_mib_now();
        CHUNK_LOG(trace, rechunk, "slice {}/{} after-drain rss={} MB  ({:+} since start)", slice + 1, n_slices_,
                  rss_mib, rss_mib - rss_start_);
    }
}

template <typename Derived>
void RechunkSupport<Derived, true>::Refill::slice_boundary(pmp::Team& team)
{
    // Slice boundary, INSIDE the region (the loop's barrier has closed the
    // slice): send this slice's parked old-geometry blocks home, then give
    // back the slabs they just emptied. Only blocks already released by
    // release_st are touched -- the source chunks the remaining slices still
    // have to read are untouched. The whole team drains (each rank its own
    // list and its own slabs) and one thread sweeps: master-only boundaries
    // between two regions add up over a re-chunk.
    team.masked(
        [&]
        {
            if (trace_on_)
            {
                slice_t0_ = SliceClock::now();
                internal::g_drain_probe.reset();
                // Read past the drain's first barrier.
                internal::g_drain_probe.armed.store(true, std::memory_order_relaxed);
            }
        });
    ManagerT::drain_reuse_list_for_size_team(team, static_cast<unsigned int>(old_size_), team_drain_);
    // The slabs this drain emptied go to the value arena for the next slice,
    // which allocates about as much at the new size (same slab size): their
    // resident pages are reused, where a madvise gave them to the kernel and
    // the next slice faulted fresh zero pages back. Each rank recycles its own
    // slabs -- the drain's owner discipline -- and the arena hands a rank its
    // own back first (NUMA). The set and the safety argument are the final
    // sweep's (rechunk_to()): the old storage's set, a quiescent point, a slab
    // found empty is empty. The trace books the recycling as "collect".
    // Always on: without it (the former CHUNK_RECHUNK_RECYCLE=0, a madvise
    // sweep per boundary) zero_dyn ran 29% slower on Milan.
    team.single(
        [&]
        {
            if (trace_on_)
            {
                slice_t1_ = SliceClock::now();
                internal::g_slice_times.drain_ms += internal::elapsed_ms(slice_t0_, slice_t1_);
                // Every thread reported before the drain's last barrier.
                auto&        dp  = internal::g_drain_probe;
                auto&        st  = internal::g_slice_times;
                const int    nr  = dp.n.load(std::memory_order_relaxed);
                const double inv = nr > 0 ? 1.0 / nr : 0.0;
                const auto   ld  = [](const std::atomic<long long>& slot)
                { return static_cast<double>(slot.load(std::memory_order_relaxed)); };
                st.drain_sort_max_ms += ld(dp.sort_max) * 1e-6;
                st.drain_sort_avg_ms += ld(dp.sort_sum) * 1e-6 * inv;
                st.drain_free_max_ms += ld(dp.free_max) * 1e-6;
                st.drain_free_avg_ms += ld(dp.free_sum) * 1e-6 * inv;
                st.drain_blocks_max += ld(dp.blocks_max);
                st.drain_blocks_avg += ld(dp.blocks_sum) * inv;
                dp.armed.store(false, std::memory_order_relaxed);
            }
            const auto n_ranks = static_cast<std::size_t>(CHUNK_PMP_RANK_BOUND);
            if (recycle_held_.size() != n_ranks)
            {
                recycle_held_.assign(n_ranks, 0);
                recycle_scratch_.resize(n_ranks);
            }
            std::ranges::fill(recycle_held_, static_cast<unsigned char>(0));
            // The previous boundary's slabs this slice did not take stay
            // recycled, resident (see the end of run()): no madvise between
            // slices.
        }); // implicit barrier: recycle_held_ is clear
    {
        // Each thread unlinks its own rank's empty slabs, in parallel and
        // without the arena's lock; the single below hands them over (do not
        // call recycle_batch() from every thread: the whole team queues on
        // that one mutex).
        const int me = CHUNK_PMP_THREAD_RANK;
        if (me >= 0 && static_cast<std::size_t>(me) < recycle_held_.size())
        {
            recycle_held_[static_cast<std::size_t>(me)] = 1;
            const DynChunkSizeGuard at_old{static_cast<unsigned int>(old_size_)};
            ChunkType::get_chunk_allocator(me, old_key_)
                .detach_empty_slabs(recycle_scratch_[static_cast<std::size_t>(me)]);
        }
    }
    team.barrier();
    team.single(
        [&]
        {
            const DynChunkSizeGuard at_old{static_cast<unsigned int>(old_size_)};
            for (std::size_t r = 0; r < recycle_held_.size(); ++r)
            {
                auto& allocator = ChunkType::get_chunk_allocator(static_cast<int>(r), old_key_);
                if (recycle_held_[r] == 0)
                {
                    // A rank no team member holds (a smaller team than the
                    // rank bound).
                    allocator.detach_empty_slabs(recycle_scratch_[r]);
                }
                // One uncontended lock per rank, each rank's slabs under its
                // own rank (NUMA).
                allocator.recycle_detached(recycle_scratch_[r]);
            }
            if (trace_on_)
            {
                internal::g_slice_times.collect_ms += internal::elapsed_ms(slice_t1_, SliceClock::now());
            }
        }); // implicit barrier: the next slice allocates from the recycled slabs
}

template <typename Derived>
void RechunkSupport<Derived, true>::Refill::rebuild_one(int new_chunk, std::vector<T>& buf, Tally& tally)
{
    // Pure read of src_ (old) + armed write of tmp_ (new), into this
    // thread's own buffer.
    const IndType base = static_cast<IndType>(new_chunk) * static_cast<IndType>(new_size_);
    if (base >= total_)
    {
        return;
    }
    // In long long: base + new_size_ passes IndType's range when the array
    // ends less than a chunk below its top (int index, ~2^31 elements), and the
    // wrapped end then sized the copy below. Same as write_range_by_chunks().
    const long long end_ll = static_cast<long long>(base) + new_size_;
    const IndType   end    = end_ll < static_cast<long long>(total_) ? static_cast<IndType>(end_ll) : total_;
    const Span      span{.base      = base,
                         .end       = end,
                         .old_first = static_cast<int>(base / old_size_),
                         .old_last  = static_cast<int>((end - 1) / old_size_)};

    const Sources sources = scan_sources(span);
    if (sources.all_cst1)
    {
        // Nothing to rebuild: the whole span was still the compressible
        // constant, so the new chunk stays compressed (work avoided).
        increm_chunk_rechunk_skipped<ChunkType>();
        return; // fresh cst1, leave it
    }
    increm_chunk_rechunk_rebuilt<ChunkType>();
    if (install_cst2(new_chunk, span, sources, tally))
    {
        return;
    }

    // read src_ at OLD; TLS is per-thread
    set_thread_chunk_size(static_cast<unsigned int>(old_size_));
    read_span(new_chunk, span, buf);
    // Released while the TLS is still old_size, deliberately (see there).
    release_sources(span);

    // Fill tmp's new chunk through the public bulk API. copy_range() is the
    // fast path (the per-element proxy path is much slower): it takes one
    // decision per chunk, arms with real proxy writes, stores the rest
    // straight into the buffer, and restores the zero counter the bulk store
    // bypasses -- with the same variant split, bool (bit-packed) going per
    // element.
    //
    // assume_drained: rechunk_to() has already flushed this array's deferred
    // ops, and this runs inside the parallel loop, where a per-chunk
    // whole-array drain would be both wrong and pointless.
    set_thread_chunk_size(static_cast<unsigned int>(new_size_));
    copy_range(tmp_, static_cast<std::size_t>(span.base), static_cast<std::size_t>(span.end), buf.begin(),
               assume_drained);

    fold_back(new_chunk, sources, tally);
}

template <typename Derived>
auto RechunkSupport<Derived, true>::Refill::scan_sources(const Span& span) const -> Sources
{
    Sources sources;
    for (int oc = span.old_first; oc <= span.old_last; ++oc)
    {
        const auto& och = src_.chunk_list_[static_cast<std::size_t>(oc)];
        if constexpr (is_zero_any_family_v<ChunkType>)
        {
            if (och.is_cst_val1())
            {
                sources.any_cst1 = true;
            }
        }
        if (och.is_cst_val1())
        {
            continue;
        }
        sources.all_cst1 = false;
        if constexpr (is_zero_any_family_v<ChunkType>)
        {
            if constexpr (has_motif_state_v<ChunkType>)
            {
                // A motif chunk (pattern::Array): is_cst_val2() for the write
                // engine, but no constant -- its elements are motif + offset.
                if (ChunkType::is_motif_ptr(och.values_load()))
                {
                    sources.all_folded = false;
                }
            }
            if (!och.is_cst_val2())
            {
                sources.all_folded = false;
            }
            else if (sources.cst2 == nullptr)
            {
                sources.cst2 = och.values_load();
            }
            else if (och.values_load() != sources.cst2)
            {
                sources.same_cst2 = false;
            }
        }
        else if constexpr (second_index)
        {
            // Same test as zero_any, with the second-value index
            // zero_one_value needs and zero_any carries implicitly.
            if (!och.is_cst_val2(ind_old2_))
            {
                sources.all_folded = false;
            }
        }
        else
        {
            sources.all_folded = false; // zero, or bool: no second value
        }
    }
    return sources;
}

template <typename Derived>
bool RechunkSupport<Derived, true>::Refill::install_cst2([[maybe_unused]] int            new_chunk,
                                                         [[maybe_unused]] const Span&    span,
                                                         [[maybe_unused]] const Sources& sources,
                                                         [[maybe_unused]] Tally&         tally)
{
    // zero_any: all sources are the same second constant -> install it.
    // Filling a buffer and folding it back (fold_back) gives the same state,
    // but the pages those allocations touch stay with the process. This is
    // what a type-2 compression commits (find_or_set on the chunk's
    // second-value slot, type table, values pointer), minus the buffer; tmp's
    // chunk is reachable by no other thread until the swap, so no CAS.
    if constexpr (is_zero_any_family_v<ChunkType> && !std::is_same_v<T, bool>)
    {
        if (sources.all_folded && sources.same_cst2 && !sources.any_cst1 && sources.cst2 != nullptr)
        {
            set_thread_chunk_size(static_cast<unsigned int>(old_size_));
            // auto: SecondValueRef is not declared in every TU that
            // instantiates this file (zero_one_value_array.cpp), and the name
            // would be looked up even in a discarded branch.
            const T    value = static_cast<T>(std::as_const(src_)[span.base]);
            auto&      nch   = tmp_.chunk_list_[static_cast<std::size_t>(new_chunk)];
            const auto psv   = ProxyType::BaseSpec::get_tab_second_value(tmp_.num_tab_, new_chunk);
            CstSlot*   ptc   = &ProxyType::tab_type_cst_val[tmp_.num_tab_][new_chunk];
            (void)nch.find_or_set_ind_cst_val2(value, psv);
            ChunkType::set_cst_val_to_cst_val2(ptc);
            nch.init_values_cst_val(ptc, psv);
            ++tally.folded;
            ++tally.installed;
            return true;
        }
    }
    return false;
}

template <typename Derived>
void RechunkSupport<Derived, true>::Refill::read_span([[maybe_unused]] int new_chunk,
                                                      const Span&          span,
                                                      std::vector<T>&      buf) const
{
    // Bulk read, one values_load() per SOURCE chunk instead of the full read
    // protocol per element (const operator[]: index split, sync_chunk, a
    // ReadSection with its light_fence -- a full seq_cst fence without
    // asymmetric Dekker --, acquire load, zero_any range test; none of it
    // vectorises).
    //
    // Sound because the source is QUIESCENT for the whole refill: rechunk_to()
    // flushed its deferred ops (nothing for sync_chunk to pull) and dropped
    // its compression candidates (rechunk_drop_candidates), and nobody writes
    // src_ meanwhile, so no pass can claim, compress or release one of its
    // chunks -- what the ReadSection guards against. The only release is
    // release_st (release_sources), coordinated by old_done_: a source
    // chunk is freed once every span reading it is done.
    //
    // Per source chunk, the same decision element_at_unguarded() makes per
    // element, on the same single load: a zero_any cst2 pointer names an
    // identifier region that was never written -> fill with the constant;
    // anything else (materialised buffer, cst1 sentinel, zero_one_value's
    // filled cst2 array) is readable as is -> memcpy. bool / Bit keep the
    // per-element path (bit-packed storage).
    if constexpr (std::is_trivially_copyable_v<T> && !is_bool_like_v<T>)
    {
        std::size_t pos = 0;
        for (int oc = span.old_first; oc <= span.old_last; ++oc)
        {
            const auto&   och     = src_.chunk_list_[static_cast<std::size_t>(oc)];
            const IndType oc_base = static_cast<IndType>(oc) * static_cast<IndType>(old_size_);
            const IndType lo      = span.base > oc_base ? span.base : oc_base;
            // In long long: oc_base + old_size_ wraps for the last old chunk of
            // an array within a chunk of IndType's top; hi - lo then went
            // negative, a huge size_t length for the memcpy below.
            const long long oc_end = static_cast<long long>(oc_base) + old_size_;
            const IndType   hi = static_cast<long long>(span.end) < oc_end ? span.end : static_cast<IndType>(oc_end);
            const auto      src_len = static_cast<std::size_t>(hi - lo);
            const auto      off     = static_cast<std::size_t>(lo - oc_base);
            const T* const  vals    = och.values_load();
            if constexpr (is_zero_any_family_v<ChunkType>)
            {
                if (och.is_cst_val2_ptr_of(vals))
                {
                    std::fill_n(buf.begin() + static_cast<std::ptrdiff_t>(pos), src_len,
                                och.second_value_from_ptr(vals));
                    pos += src_len;
                    continue;
                }
            }
            if constexpr (has_motif_state_v<ChunkType>)
            {
                // A motif chunk: motif + its offset, from its second-value slot
                // (quiescent source: the slot is stable, no re-check needed).
                if (ChunkType::is_motif_ptr(vals))
                {
                    const T offset = internal::load_element_relaxed(src_.motif_offset_slot(oc));
                    for (std::size_t k = 0; k < src_len; ++k)
                    {
                        buf[pos + k] = ChunkType::motif_value(vals, static_cast<int>(off + k), offset);
                    }
                    pos += src_len;
                    continue;
                }
            }
            // raw-read: refill bulk read -- quiescent source, see above
            std::memcpy(buf.data() + pos, vals + off, src_len * sizeof(T));
            pos += src_len;
        }
        if constexpr (build_config::any_check)
        {
            check_read(new_chunk, span, pos, buf);
        }
    }
    else
    {
        const IndType len = span.end - span.base;
        for (IndType k = 0; k < len; ++k)
        {
            buf[static_cast<std::size_t>(k)] = static_cast<T>(std::as_const(src_)[span.base + k]);
        }
    }
}

template <typename Derived>
void RechunkSupport<Derived, true>::Refill::check_read([[maybe_unused]] int                   new_chunk,
                                                       [[maybe_unused]] const Span&           span,
                                                       [[maybe_unused]] std::size_t           copied,
                                                       [[maybe_unused]] const std::vector<T>& buf) const
{
    // Sampled (first, last, every 64th): this is a check, not a second copy.
    // Bytewise, so only where bytes ARE the value: scalars (floats included --
    // both paths copy the same bits, NaN and -0.0 alike) and classes without
    // padding.
    if constexpr (!std::is_class_v<T> || std::has_unique_object_representations_v<T>)
    {
        const IndType len = span.end - span.base;
        if (copied != static_cast<std::size_t>(len)) [[unlikely]]
        {
            CHUNK_INVARIANT_FAILED("rechunk refill bulk read: new chunk {} copied {} elements, span is {}", new_chunk,
                                   copied, static_cast<std::size_t>(len));
        }
        const auto check_at = [&](IndType index)
        {
            const T ref = static_cast<T>(std::as_const(src_)[span.base + index]);
            if (std::memcmp(&ref, &buf[static_cast<std::size_t>(index)], sizeof(T)) != 0) [[unlikely]]
            {
                CHUNK_INVARIANT_FAILED("rechunk refill bulk read: new chunk {} element {} (global {}) differs "
                                       "from the per-element read ({} -> {})",
                                       new_chunk, static_cast<long long>(index),
                                       static_cast<long long>(span.base + index), old_size_, new_size_);
            }
        };
        for (IndType j = 0; j < len; j += 64)
        {
            check_at(j);
        }
        check_at(len - 1);
    }
}

template <typename Derived>
void RechunkSupport<Derived, true>::Refill::release_sources([[maybe_unused]] const Span& span)
{
    // Release the source chunks this span has just consumed, NOW rather than
    // at tmp's destruction: tmp at S' and src at S otherwise coexist until the
    // swap, and the peak is close to their sum.
    //
    // A source chunk is done for good once every element of it has been
    // transcribed. When the new size is a multiple of the old one, this span
    // covers every chunk of [old_first, old_last] whole, and releases them.
    // Otherwise (fining, or sizes that do not divide each other: 256 -> 384)
    // an old chunk is read by every new chunk that overlaps it, on whichever
    // threads the schedule gives them: the LAST to finish releases it,
    // counted per old chunk (old_done_) against its number of readers. Never
    // "the last new chunk BY INDEX": a schedule block boundary can fall
    // between c and c+1, and the source is then freed while the thread
    // holding the previous index still reads it.
    //
    // Released while the TLS is still old_size, deliberately:
    // free_block_to_pool() routes by the AMBIENT geometry, so freeing a
    // S-sized buffer under S' would park it in the wrong pool.
    //
    // Each variant's release_st() frees the buffer and points the chunk at the
    // first constant without counting a compression -- exactly what "already
    // transcribed" means here. zero_one_value / zero_any's go through their
    // compression's own block_clear_common(0, ...) (type slot to cst1, second-
    // value slot reset, CAS, free_block_to_pool), which is sound on the source:
    // quiescent, so the CAS cannot lose, and torn down right after the swap,
    // where a cst1 chunk owns nothing. Only materialised chunks are released:
    // a chunk at a second constant points at a shared array, never freed.
    if constexpr (releases_sources)
    {
        const int thr = get_thr_rank_fast();
        for (int oc = span.old_first; oc <= span.old_last; ++oc)
        {
            const IndType oc_base = static_cast<IndType>(oc) * static_cast<IndType>(old_size_);
            // In long long: oc_base + old_size_ passes IndType's range for the
            // last old chunk of an array within a chunk of its top (see rebuild_one).
            const long long oc_end = static_cast<long long>(oc_base) + old_size_;
            if (!aligned_coarsening_)
            {
                // Old chunk oc's readers: the new chunks overlapping its
                // elements [oc_base, min(oc_end, total)). Release it with the
                // last of them to finish, whoever that is.
                const IndType last_elem =
                    (oc_end < static_cast<long long>(total_) ? static_cast<IndType>(oc_end) : total_) - 1;
                const auto readers = static_cast<int>((last_elem / static_cast<IndType>(new_size_)) -
                                                      (oc_base / static_cast<IndType>(new_size_)) + 1);
                if (old_done_[static_cast<std::size_t>(oc)].fetch_add(1, std::memory_order_acq_rel) + 1 != readers)
                {
                    continue;
                }
            }
            else if (oc_end > static_cast<long long>(span.end))
            {
                continue; // still needed by the next new chunk
            }
            auto& och = src_.chunk_list_[static_cast<std::size_t>(oc)];
            // release_st, not clear_st: a release once copied, not a
            // compression (CHUNK_STATS counted every rebuilt chunk as one).
            if constexpr (chunk_config::use_zero)
            {
                if (!och.is_cst_val1())
                {
                    och.release_st(thr, old_size_);
                }
            }
            else if constexpr (is_zero_any_family_v<ChunkType>)
            {
                // written(): neither cst1 nor a cst2 (a pattern motif included).
                if (och.written())
                {
                    och.release_st(&ProxyType::tab_type_cst_val[src_.num_tab_][oc],
                                   ProxyType::BaseSpec::get_tab_second_value(src_.num_tab_, oc), thr);
                }
            }
            else
            {
                // owns_live_buffer(): neither cst1 nor the second constant.
                if (och.owns_live_buffer())
                {
                    och.release_st(ind_old2_, thr);
                }
            }
        }
    }
}

template <typename Derived>
void RechunkSupport<Derived, true>::Refill::fold_back([[maybe_unused]] int            new_chunk,
                                                      [[maybe_unused]] const Sources& sources,
                                                      [[maybe_unused]] Tally&         tally)
{
    // Fold back NOW what was already compressed.
    //
    // copy_range materialises the new chunk whatever its content. When every
    // source chunk was a compressed constant (all_folded), the buffer holds
    // only constants; leaving it for the next end-of-phase sweep keeps the
    // WHOLE array's cst2 share materialised until then (it can double the
    // process RSS).
    //
    // So run the variant's own compress() on the chunk right here, on this
    // thread's rank, while the TLS is still new_size (clear_int_st frees the
    // buffer into the pool routed by the ambient geometry). It is exactly what
    // the sweep would do: scan, then clear_int_st with the variant's state --
    // nothing synthesised. The transient is bounded to one chunk per thread.
    // If the constants differ across the span the scan refuses and the chunk
    // stays materialised. The candidate the arming write queued stays in the
    // list; the sweep finds a cst1/cst2 chunk and skips it (compress() guards
    // on is_not_cst_val1(); the scan on a cst2 chunk re-derives the same
    // state).
    //
    // Wired for zero_any (compress() by num_chunk, num_tab) and zero_one_value
    // (by the array's inf_array record). bool registers no second value
    // (get_ind_second_value aborts) and is bit-packed: never this path.
    if constexpr (is_zero_any_family_v<ChunkType>)
    {
        if (sources.all_folded)
        {
            ++tally.folded;
            auto& nch = tmp_.chunk_list_[static_cast<std::size_t>(new_chunk)];
            nch.compress(get_thr_rank_fast(), new_chunk, tmp_.num_tab_,
                         /*apply=*/true, ChunkElems{static_cast<unsigned>(new_size_)});
            // Every source chunk WAS a constant here, so a refusal means the
            // span merged constants that differ -- the genuine cost of
            // coarsening, the one the forecast has to carry.
            if (!nch.is_cst_val1() && !nch.is_cst_val2())
            {
                ++tally.refused;
            }
        }
    }
    // No zero branch: zero's one constant (zero) spans already took the early
    // return in rebuild_one(), and scanning every other rebuilt chunk for
    // zeros folds almost none for a full buffer scan each; the end-of-phase
    // sweep handles chunks that actually became zero. Chunk<T>::compress()
    // keeps the guard that a second visit to a cst1 chunk must not park the
    // zero sentinel -- load-bearing for the sweep regardless of this loop.
    //
    // zero_one_value: bool has no second value to fold to (second_index is
    // false), so it simply does nothing here. Also gated on
    // use_fast_block_compression: this four-argument compress() lives in
    // zero_one_value::ChunkFastCompPolicy<..., Enabled>, which zero_one_value
    // inherits with Enabled = use_fast_block_compression (= threaded): a
    // SERIAL build has no such overload. There, copy_range() already folded
    // every chunk it left uniform, through the counters
    // (internal::fold_if_full).
    else if constexpr (second_index && chunk_config::use_fast_block_compression)
    {
        if (sources.all_folded)
        {
            ++tally.folded;
            auto& nch = tmp_.chunk_list_[static_cast<std::size_t>(new_chunk)];
            nch.compress(get_thr_rank_fast(), inf_new2_,
                         /*apply=*/true, static_cast<unsigned int>(new_size_));
            if (!nch.is_cst_val1() && !nch.is_cst_val2(inf_new2_.ind))
            {
                ++tally.refused;
            }
        }
    }
}

template <typename Derived>
void RechunkSupport<Derived, true>::Refill::merge(const Tally& tally) noexcept
{
    n_folded_.fetch_add(tally.folded, std::memory_order_relaxed);
    n_refused_.fetch_add(tally.refused, std::memory_order_relaxed);
    n_installed_.fetch_add(tally.installed, std::memory_order_relaxed);
}

template <typename Derived>
void RechunkSupport<Derived, true>::Refill::log_census() const
{
    // What the new geometry actually HOLDS, counted on tmp before the swap:
    // "folded" counts spans the rebuild could fold, this the chunks left with
    // a real buffer. The gap is the diagnosis:
    //  - few folded, many materialised -> the fold is misfiring (wrong
    //    second-value index, or the span's sources were not constants);
    //  - many folded, many materialised -> genuine merging cost: two
    //    neighbours carrying DIFFERENT constants make a chunk that no
    //    single-constant representation can hold, and it must stay
    //    materialised. That cost is real and belongs in the decider's data
    //    term.
    //  - many folded, few materialised -> the residue is not in the chunks;
    //    look at the pools (CHUNK_LOG=arena=info).
    long n_mat = 0;
    for (int c = 0; c < n_new_; ++c)
    {
        const auto& ch         = tmp_.chunk_list_[static_cast<std::size_t>(c)];
        bool        compressed = ch.is_cst_val1();
        if constexpr (is_zero_any_family_v<ChunkType>)
        {
            compressed = compressed || ch.is_cst_val2();
        }
        else if constexpr (second_index)
        {
            compressed = compressed || ch.is_cst_val2(inf_new2_.ind);
        }
        if (!compressed)
        {
            ++n_mat;
        }
    }
    CHUNK_LOG(debug, memory,
              "{} -> {}: {} new chunks, {} folded back at rebuild ({} installed directly), {} still materialised ({} "
              "MB of buffers), {} refused merged-constant spans",
              old_size_, new_size_, n_new_, n_folded_.load(), n_installed_.load(), n_mat,
              (n_mat * static_cast<long>(new_size_) * static_cast<long>(sizeof(T))) >> 20, n_refused_.load());
}

} // namespace shrink_array
