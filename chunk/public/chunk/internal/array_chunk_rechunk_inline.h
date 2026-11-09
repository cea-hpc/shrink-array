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

// Out-of-line bodies of the CHUNK_DYNAMIC re-chunk algorithm
// (RechunkSupport<Derived, true>). Included at the end of array_chunk_base.h,
// after the class, so the explicit object parameter sees a complete Derived and so
// the heavy includes below (compression manager, stats) stay out of the
// container declaration itself. The refill is in array_chunk_rechunk_refill.h,
// the traces in array_chunk_rechunk_trace.h.

#include <cstddef>     // std::size_t
#include <type_traits> // std::remove_reference_t (the sweep's type)
#include <utility>     // std::swap (the array swap)

#include "chunk/internal/in_t.h"                       // InT<T>: the captured state
#include "chunk/internal/array_chunk_base.h"           // ArrayChunkBase: self.set_chunk_size, invalidate_proxies
#include "chunk/internal/array_chunk_core.h"           // alloc_affinity_detail::RechunkStorageScope
#include "chunk/internal/array_chunk_rechunk.h"        // RechunkSupport, the class these bodies belong to
#include "chunk/internal/array_chunk_rechunk_refill.h" // RechunkSupport::Refill (rechunk_refill)
#include "chunk/internal/array_chunk_rechunk_trace.h"  // RechunkPhaseTrace
#include "chunk/internal/chunk_stats.h"                // increm_chunk_rechunk (CHUNK_STATS)
#include "chunk/internal/PMP_CHUNK.h"                  // CHUNK_PMP_RANK_BOUND
#include "chunk/log.h"                                 // CHUNK_LOG (the pool census)
#include "chunk/config.h"                              // build_config::threaded, use_fast_alloc, any_check
#include "chunk/internal/chunk_size_policy.h"          // set_*_chunk_size, DynChunkSizeGuard, BirthAtActiveSize
#include "chunk/internal/invariant_check.h"            // CHUNK_INVARIANT_FAILED (the identity swap)

namespace shrink_array
{

template <typename Derived>
void RechunkSupport<Derived, true>::rechunk_to(this Derived& self, unsigned int new_size)
{
    // A refusal is a caller bug, on a cold path: a message in EVERY build.
    if (const auto done = self.try_rechunk_to(new_size); !done) [[unlikely]]
    {
        warn_chunk_size_refused("rechunk_to", new_size, done.error());
    }
}

template <typename Derived>
std::expected<void, ChunkSizeError> RechunkSupport<Derived, true>::try_rechunk_to(this Derived& self,
                                                                                  unsigned int  new_size)
{
    const unsigned int old_size = self.size_policy_.size();
    // Captured before the swap below hands `self` tmp's key: the allocator set
    // holding the OLD geometry's blocks is (old_key, old_size), and it is the
    // one whose slabs must empty for the re-chunk to give memory back.
    const unsigned int old_key = self.alloc_key();
    if (new_size == old_size)
    {
        return {};
    }

    // A power of two or a multiple of 64 (is_valid_chunk_size; 0 refused),
    // and at most cst_buffer_size: the shared constant buffers (cst_val1 /
    // cst_val2) are that wide (the dynamic-mode ceiling), and a wider chunk
    // would read them out of bounds. Raise CHUNK_DYN_MAX_SIZE if a legitimate
    // geometry needs to be larger.
    if (const auto ok = check_chunk_size(new_size); !ok) [[unlikely]]
    {
        return ok;
    }

    // Serial fast-alloc builds re-chunk too: the per-size allocator pool
    // (allocator_pool_serial_) routes each allocation to the right size, so the
    // old and new geometries coexist during the refill.

    const IndType total = self.m_size_;
    if (total <= 0)
    {
        self.set_chunk_size(new_size); // legal on an empty array (the size is checked above)
        return {};
    }

    // A real geometry change is now committed: every no-op / refusal exit
    // (same size, size 0, invalid or too wide a size, empty array) is above
    // this point, and nothing below aborts. Count it for the run's stats.
    ++shrink_array::g_rechunk_count;

    // rechunk=debug: per-phase wall time (tmp creation, refill, flush+swap,
    // tmp teardown); memory=debug: the RSS at the phase boundaries.
    using Mark = internal::RechunkPhaseTrace::Mark;
    internal::RechunkPhaseTrace trace;

    // Drain THIS array's deferred WRITE ops at the OLD geometry, before the
    // t_active switch.
    self.flush_pending_deferred_ops();
    trace.mark(Mark::flushed);

    // Remove THIS array's pending compression CANDIDATES at the OLD geometry
    // before t_active changes: a perform_compression during tmp's refill
    // (t_active == new_size) must not meet these OLD-size chunks (over-scan on
    // shrink, under-scan on grow). Content-neutral: the refill reads
    // materialised values.
    self.rechunk_drop_candidates();

    // No dense O(n) snapshot: the refill streams old -> new chunk by chunk.

    // Variant state captured BEFORE the geometry switch (zero_one_value's
    // array-level second_value; T{} / ignored for zero & zero_any).
    const InT<T> state = self.rechunk_capture_state();

    // 2) build the fresh array at S'. Set the active geometry around its
    //    construction + refill only, then restore. g_active is saved (only
    //    rechunk_to() touches it); t_active is NOT -- see the restore site.
    const unsigned int saved_g_active = active_chunk_size();

    set_active_chunk_size(new_size);
    set_thread_chunk_size(new_size);
    register_runtime_chunk_size(new_size); // primes S' pools

    // Drain the S' reuse free-list BEFORE anything allocates at S' (a
    // buffer of a different geometry parked here would overflow init).
    self.rechunk_drain_reuse_list(new_size);

    {
        trace.mark(Mark::before_tmp);
        // Born at S', own num_tab, primed pool. The new storage takes its own
        // allocator set (lazy affinity: an array gets one at its first
        // re-chunk, see assign_alloc_key()), and is born at exactly S',
        // whatever the element type; both scopes end once it is built
        // (guaranteed elision: tmp is the lambda's return object).
        Derived tmp = [total]
        {
            const alloc_affinity_detail::RechunkStorageScope affinity_scope;
            const BirthAtActiveSize                          birth_scope;
            return Derived(total);
        }();

        // Seed variant state into tmp BEFORE refill (zero_one_value second_value,
        // so its compression decisions use the same value).
        tmp.rechunk_seed_fresh(state);

        // 3) Refill tmp from *this (OLD -> NEW geometry): rechunk_refill()
        //    rebuilds chunk by chunk and skips the spans still at the first
        //    constant (only materialised chunks are written).
        // CHUNK_STATS: one geometry change, plus the per-chunk work it costs
        // (counted inside rebuild_one below).
        increm_chunk_rechunk<ChunkType>();
        trace.mark(Mark::refill_start);
        self.rechunk_refill(tmp, old_size, new_size, old_key);
        trace.mark(Mark::refill_end);

        tmp.flush_pending_deferred_ops(); // materialise tmp's deferred writes

        // tmp's refill CANDIDATES are deliberately KEPT. Dropped, every rebuilt
        // chunk holding a constant stays materialised until teardown (a
        // zero_one_value/zero_any chunk only re-enters the list on a write that transitions
        // it to a constant) -- a large memory cost per re-chunk.
        //
        // No geometry-mismatch hazard: every candidate tuple carries the
        // chunk's own geometry (field <3>, captured at enqueue) and all three
        // variants scan and compress at that size.
        //
        // What makes keeping them correct across the swap below:
        //  - the tuples point into tmp.chunk_list_'s storage, which the
        //    std::swap hands to *this unchanged (buffer swap, no move);
        //  - they carry tmp's num_tab, and rechunk_swap_identity() makes that
        //    *this's identity (member swap for zero/zero_any, registry swap for
        //    zero_one_value), so the sweep's reset_tab_inside_list_to_compress(num_tab,
        //    num_chunk) lands on the row allocated for tmp at new_size;
        //  - tmp's own destruction removes candidates by ITS post-swap num_tab
        //    (= *this's old identity, already emptied by rechunk_drop_candidates()
        //    above) and, for zero_any, by the OLD chunk_list_'s pointer range --
        //    neither touches these.
        // The next end-of-phase sweep folds the constant ones back.

        // Re-drain THIS array's deferred ops before swapping storage out.
        // FORCED: the ops are keyed by the OLD chunk_list_'s addresses, about to
        // be destroyed with tmp, and any_pending_ops() can read 0 while ops are
        // still queued -- left behind, they would be replayed against a
        // recycled, wrong-size chunk. Flush while the old geometry is in place.
        self.flush_pending_deferred_ops(/*force=*/true);

        // 4) SWAP the common storage. nb_tab is static and NOT swapped.
        //    num_tab + registry (+ any variant members) go through the
        //    identity hook: zero / zero_any swap the num_tab member, zero_one_value
        //    resolves num_tab from the registry and must NOT swap it.
        using std::swap;
        swap(self.chunk_list_, tmp.chunk_list_);
        // The new storage was built under tmp's own affinity key (its slabs);
        // the key travels with the storage. self keeps tmp's key from here on,
        // tmp takes the old one and dies with it: its clear_core() drains that
        // key's lists and the old slabs empty.
        swap(self.alloc_key_, tmp.alloc_key_);
        self.swap_dirty_bitset(tmp);
        swap(self.m_size_, tmp.m_size_);
        swap(self.nb_chunk_, tmp.nb_chunk_);
        swap(self.size_policy_, tmp.size_policy_);
        self.invalidate_proxies();

        const bool ok = self.rechunk_swap_identity(tmp);
        if constexpr (build_config::any_check)
        {
            if (!ok)
            {
                // The storage is swapped already, the registry is not: the
                // per-num_tab tables no longer follow the chunks.
                CHUNK_INVARIANT_FAILED("[RECHUNK] identity swap failed (this or tmp not registered) -- storage "
                                       "swapped, registry not");
            }
        }
        (void)ok;

        // Variant finalisation after the swap (zero_one_value re-owns its
        // map_inf_array_ entry so it does not dangle when tmp dies).
        self.rechunk_reown_after_swap(tmp, state);

        // Restore the active geometry BEFORE tmp is destroyed, so tmp (now
        // OLD geometry) frees its blocks into the OLD-size pool.
        //
        // t_active is restored to old_size, NOT to a value saved at entry:
        // every write path re-sets t_active to ITS OWN array's chunk_size()
        // right before materialising (see zero_one_value::Proxy::ensure_writable_storage()),
        // so a value saved at entry may belong to an unrelated array and would
        // route free_elt_pow2() (via get_chunk_allocator(), keyed on the
        // ambient) to the WRONG size pool. old_size comes from this array's
        // own size_policy_; g_active is touched only by rechunk_to(), so
        // saved_g_active is a trustworthy baseline.
        set_active_chunk_size(saved_g_active);
        set_thread_chunk_size(old_size);

        trace.mark(Mark::swapped);
        // tmp destroyed here: clear_core drops tmp's candidates by num_tab
        // and parks its OLD-geometry blocks by size. Without the end-of-life
        // census: tmp's death is not an exit (ArrayChunkBase::exit_census_).
        tmp.exit_census_ = false;
        // Its chunk_list_ (the OLD geometry's) is freed with it.
        // rechunk=debug: split tmp's teardown into its steps (teardown_probe.h).
        trace.arm_teardown();
    }
    trace.tmp_destroyed();

    // Free the OLD-geometry blocks tmp just parked into the old-size reuse
    // list. Nothing pops them again (the array has moved off old_size), so
    // without this they are pure RETENTION, ~one array's worth of buffers per
    // geometry change. Returns them to the OS under std-alloc (under fast
    // alloc, the slab sweep below does), and a later pop can never grab a
    // wrong-size block.
    self.rechunk_drain_reuse_list(old_size);
    trace.mark(Mark::drained);
    sweep_old_geometry(old_key, old_size);
    trace.mark(Mark::swept);
    if (trace.memory_on())
    {
        log_old_pool_census(old_key, old_size);
    }
    trace.log(old_size, new_size);
    return {};
}

template <typename Derived>
void RechunkSupport<Derived, true>::sweep_old_geometry(unsigned int old_key, unsigned int old_size)
{
    // The array has left (old_key, old_size) for good: once the drain has run
    // its slabs there hold nothing, but manage_full_empty() only inspects a
    // slab at the instant a block is freed, so slabs that became empty
    // unobserved stay resident. Sweep them now -- key 0 too: the shared
    // per-type pool every array below CHUNK_ALLOC_AFFINITY_MIN_BYTES falls
    // back to, where most of that residue sits.
    //
    // Sweeping a SHARED pool means trusting nn_ref, which the hot path may
    // not do: manage_full_empty() fires on every free, concurrently, where a
    // transient zero is possible. This runs at the serial, quiescent end of a
    // re-chunk, with no concurrent free in flight, so an empty slab is empty.
    // Emptiness is the whole claim -- a key-0 slab mixes several arrays'
    // blocks and no ownership is assumed.
    if constexpr (build_config::use_fast_alloc && build_config::threaded)
    {
        const DynChunkSizeGuard at_old{old_size};
        // Every rank, both keys, into ONE sweep: the abandoned set's slabs were
        // carved back to back across ranks, so one madvise per contiguous run
        // instead of one per slab (each one a TLB shootdown on every CPU).
        typename std::remove_reference_t<decltype(ChunkType::get_chunk_allocator(0, 0u))>::SlabSweep sweep;
        for (int r = 0; r < CHUNK_PMP_RANK_BOUND; ++r)
        {
            if (old_key != 0)
            {
                ChunkType::get_chunk_allocator(r, old_key).collect_empty_slabs(sweep);
            }
            ChunkType::get_chunk_allocator(r, 0u).collect_empty_slabs(sweep);
        }
        sweep.flush();
    }
}

template <typename Derived>
void RechunkSupport<Derived, true>::log_old_pool_census(unsigned int old_key, unsigned int old_size)
{
    // Occupancy of the allocator set the array just left: after the drain
    // its slabs should be empty. A non-zero "live" means blocks of that
    // geometry are still out somewhere, so no slab can empty and the
    // re-chunk's memory never comes back. Also reported for key 0 (the
    // shared per-type pool), whose slabs mix several arrays' blocks and
    // cannot empty on one array's re-chunk.
    struct Occupancy
    {
        std::size_t slabs = 0;
        std::size_t live  = 0;
        std::size_t empty = 0;
    };
    // [&], not [old_size]: old_size is unused where the branch below is
    // discarded (serial or std-alloc builds), and clang warns of an explicit
    // capture then (-Wunused-lambda-capture).
    const auto scan = [&]([[maybe_unused]] unsigned int key)
    {
        Occupancy occ;
        if constexpr (build_config::use_fast_alloc && build_config::threaded)
        {
            const DynChunkSizeGuard at_old{old_size};
            for (int r = 0; r < CHUNK_PMP_RANK_BOUND; ++r)
            {
                const auto& allocator = ChunkType::get_chunk_allocator(r, key);
                for (const auto* node : {allocator.block, allocator.block_full})
                {
                    for (; node != nullptr; node = node->next)
                    {
                        ++occ.slabs;
                        occ.live += static_cast<std::size_t>(node->nn_ref);
                        if (node->nn_ref == 0)
                        {
                            ++occ.empty;
                        }
                    }
                }
            }
        }
        return occ;
    };
    const Occupancy   own    = scan(old_key);
    const Occupancy   shared = old_key != 0 ? scan(0u) : Occupancy{};
    const std::size_t blk    = static_cast<std::size_t>(old_size) * sizeof(T);
    CHUNK_LOG(debug, memory,
              "pool S={} key={} after drain: slabs={}  live_blocks={} ({} MB)  empty_slabs={} | key=0 (shared "
              "pool): slabs={}  live_blocks={} ({} MB)  empty_slabs={}",
              old_size, old_key, own.slabs, own.live, (own.live * blk) >> 20U, own.empty, shared.slabs, shared.live,
              (shared.live * blk) >> 20U, shared.empty);
}

template <typename Derived>
void RechunkSupport<Derived, true>::rechunk_refill(this Derived& self,
                                                   Derived&      tmp,
                                                   int           old_size,
                                                   int           new_size,
                                                   unsigned int  old_key)
{
    Refill(self, tmp, old_size, new_size, old_key).run();
}

// ---------------------------------------------------------------------------
// shrink_array::RechunkSupport<Derived, true>
// ---------------------------------------------------------------------------

template <typename Derived>
void RechunkSupport<Derived, true>::rechunk_all_to(unsigned int new_size)
{
    ProxyType::share_info_array->for_each_key(
        [new_size](ArrayKey<ChunkType> key)
        {
            if (key != nullptr)
            {
                // The key is an array's core, and the arrays of this
                // element type may differ in index type: reach each one's
                // own rechunk_to() through the core (the registry hands out
                // const keys; a re-chunk rebuilds the array).
                const_cast<ArrayChunkCore<ChunkType>*>(key)->rechunk_erased(new_size);
            }
        });
}

} // namespace shrink_array
