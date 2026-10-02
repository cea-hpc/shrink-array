#pragma once
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

#include <exception>   // std::exception (deferred enqueue)
#include <memory>      // std::make_unique
#include <type_traits> // std::remove_reference_t

#include "chunk/internal/chunk_base.h"  // the declarations these bodies belong to
#include "chunk/config.h"               // build_config::use_fast_alloc, chunk_config::
#include "chunk/internal/chunk_stats.h" // increm_chunk_read_sync / _cross (CHUNK_STATS)
#include "chunk/internal/pool_ledger.h" // CHUNK_LEDGER_PARK / CHUNK_LEDGER_RELEASE
#include "chunk/PMP_CHUNK.h"            // CHUNK_PMP_IN_PARALLEL_REGION
#include "chunk/internal/deferred_op_pool.h"
#include "chunk/internal/deferred_ops_storage.h"
#include "chunk/internal/freepath_counters.h" // CHUNK_FREEPATH_INC
#include "chunk/internal/invariant_check.h"   // INVARIANT_FAILED
#include "chunk/internal/svalloc.h"
#include "chunk/internal/write_compress_sync_policy.h"

namespace shrink_array
{

template <typename T, typename Derived>
void ChunkBase<T, Derived>::free_block_to_pool([[maybe_unused]] int thr_rank, T* val)
{
    // A pass that may run while other threads read (read_reclaim::PassScope:
    // perform_compression, compress() inside a region) collects the buffer
    // instead: a reader may still hold it between its `values` load and its
    // element load. The pass releases it here, after one grace period.
    if (read_reclaim::defer_release(&ChunkBase::release_deferred_block, this, thr_rank, val))
    {
        return;
    }

    if constexpr (build_config::use_fast_alloc)
    {
        // The block's slab says which array it belongs to (owner_key): the
        // reuse list is that array's, on this rank, and so is the budget. Key
        // 0 -- the shared pool -- is the list and budget everything used before.
        const auto&    mgr               = derived().get_manager();
        auto&          allocator         = derived().get_chunk_allocator(thr_rank);
        auto*          slab              = allocator.slab_of(val); // resolved ONCE; every route below reuses it
        const unsigned key               = slab ? slab->owner_key : 0u;
        auto&          ll_block_free_thr = mgr.get_list_block_free_thr(thr_rank, key);

        if (static_cast<int>(ll_block_free_thr.size()) < mgr.get_size_buffer_reuse(key))
        {
            // Ledger: LIVE -> PARKED (own free-list). Trips on double-park.
            CHUNK_LEDGER_PARK(val, "free_block_to_pool.park_own");
            CHUNK_FREEPATH_INC(park_own);
            ll_block_free_thr.push_back(val);
        }
        else
        {
            if (allocator.holds_live(slab))
            {
                // Ledger: LIVE -> FREE. Trips on double-free.
                CHUNK_LEDGER_RELEASE(val, "free_block_to_pool.free_elt_pow2");
                CHUNK_FREEPATH_INC(freed);
                allocator.free_elt_pow2(val, slab);
            }
            else if (slab && slab->owner && (slab->owner_rank < 0 || !CHUNK_PMP_IN_PARALLEL_REGION))
            {
                // Owned by another instance (another key, or another rank while
                // no region is open): its chains are safe to touch -- free it
                // where it came from instead of parking it for the drain.
                CHUNK_FREEPATH_INC(freed);
                CHUNK_LEDGER_RELEASE(val, "free_block_to_pool.free_to_owner");
                static_cast<std::remove_reference_t<decltype(allocator)>*>(const_cast<void*>(slab->owner))
                    ->free_elt_pow2(val, slab);
            }
            else
            {
                if (CHUNK_PMP_IN_PARALLEL_REGION)
                {
                    // Ledger: LIVE -> PARKED (own free-list, foreign-slab block).
                    CHUNK_LEDGER_PARK(val, "free_block_to_pool.park_own_foreign");
                    ll_block_free_thr.push_back(val);
                }
                else
                {
                    // FIX (cross-thread block migration, chunk pool -- analogous
                    // to the recycle() fix on the ctx/op pool side): do NOT push
                    // a foreign block onto ANOTHER thread's free-list. This
                    // unsynchronized push (a) races with the owning thread's own
                    // pop/push on ITS free-list (std::vector corruption), and
                    // (b) makes a later free_elt_pow2 route the block to the
                    // wrong allocator's manage_full_empty -> remove/insert on
                    // inconsistent linked lists + delete/munmap of a block that
                    // is still alive (the SEGV observed in manage_full_empty).
                    //
                    // Keep the block on the LOCAL free-list instead. It doesn't
                    // belong to the local slab, so it will never be passed to
                    // the local free_elt_pow2 (is_free_able_elt will return
                    // false): it simply sits there as a direct-reuse reserve
                    // (memory overwritten before use) without ever touching
                    // another allocator's linked lists. The owning slab
                    // releases everything at teardown. Bounded leak, not a UAF.
                    CHUNK_FREEPATH_INC(kept_nomigrate);
                    CHUNK_LEDGER_PARK(val, "free_block_to_pool.keep_local_no_migrate");
                    ll_block_free_thr.push_back(val);
                }
            }
        }
    }
    else
    {
        // fw15b (ledger balance): std-alloc path returns the block to the system
        // via svfree without notifying the ledger. Same hole as free_values: the
        // page goes back to malloc, gets re-handed to a later allocate_chunk_-
        // internal, and on_block_alloc sees it still LIVE -> spurious
        // INVARIANT_FAILED. This is the OpenMP recycle path (block_clear ->
        // free_block_to_pool), the actual culprit behind the check+ASan abort.
        // LIVE -> FREE, before the free, while the pointer is still ours.
        CHUNK_LEDGER_RELEASE(val, "free_block_to_pool.svfree");
        svfree(val);
    }
}

// to_be_queued_for_deferred_processing/wait_for_availability/is_not_available
// bodies live in ChunkDeferredOpsPolicy<T, Derived, true> (chunk_base.h):
// every name they use (DeferredOpsStorage,
// WriteCompressSyncPolicy, DeferredOpPtr, CompressionManager<T,Derived>) is
// unconditionally declared elsewhere, and ChunkDeferredOpsPolicy<T,Derived,
// true> is simply never instantiated in a serial build (nothing ever names
// it there), so these definitions are never compiled/linked in that case.
// self() -> static_cast<Derived*>(this): this policy does not inherit
// CRTPSupport<Derived> a second time (see the class comment in chunk_base.h).

template <typename T, typename Derived>
void ChunkDeferredOpsPolicy<T, Derived, true>::to_be_queued_for_deferred_processing(
    int                                                thr_rank,
    int                                                num,
    CompressionManager<T, Derived>::CmpDeferredOpBase* v) const
{
    // Ensure the global array exists.
    ensure_deferred_array_initialized();

    // Cache the singleton pointer once per thread — eliminates the
    // __cxa_guard_acquire check on every call without any race risk.
    // The singleton lifetime exceeds any thread lifetime.
    thread_local auto* storage = &DeferredOpsStorage<per_thread_map>::instance();

    auto* arr = storage->get(); // simple acquire load, always fresh

    auto* chunk_ptr = static_cast<Derived*>(const_cast<ChunkDeferredOpsPolicy*>(this));

    // ── Enqueue vs cross-thread flush: PER-RANK shard lock ────────────────
    // The former Dekker handshake made every enqueue back off on the GLOBAL
    // num_thread_flushing counter: while ANY flush was in progress anywhere,
    // NO writer in the process could enqueue. Under read-heavy phases flushes
    // are quasi-permanent, so writers piled into a naked spin (the 89% x86
    // hotspot, 93s @128T) while pending ops could not drain — a convoy. The
    // actual requirement is pairwise (flusher of rank r <-> writer of rank r):
    // exactly what xflush_lock_rank[thr_rank] provides. Uncontended cost: one
    // CAS + one release store, cheaper than the two seq_cst stores + fence +
    // global load of the Dekker. The lock is owner-recursive because a drain's
    // write_body may re-enter here (re-enqueue on the own rank).
    //
    // begin_write/end_write are KEPT: they are the writer half of the
    // compressor-yield protocol (start() spins on any_thread_writing), which
    // is orthogonal to map safety and unchanged here.
    using Policy = WriteCompressSyncPolicy<typename Derived::ProxyType>;

    CompressionManager<T, Derived>::enqueue_lock(thr_rank);
    Policy::announce_write(thr_rank, chunk_ptr); // flag + seq_cst fence, one Dekker verb

    // The container steps below allocate (the per-chunk map, its entries)
    // and ankerl's maps throw on a bucket overflow or a key found changed.
    // Callers are noexcept (the proxies' write operators), so an escaping
    // exception would be a silent std::terminate: turn it into an invariant
    // failure that says where. Out of memory here is fatal either way.
    try
    {
        per_thread_map& thread_map = arr[thr_rank];

        // Get or create the per-chunk map.
        auto& ops_map_ptr = thread_map[chunk_ptr];

        // Allocate the inner map if this is the first deferred op for this chunk.
        if (!ops_map_ptr)
        {
            ops_map_ptr = std::make_unique<unordered_map_deferred_operations>();
        }

        // ── Same-offset re-enqueue: REPLACE, do not double-count ──────────────
        // A second deferred write to the SAME offset on the SAME rank supersedes
        // the queued one. The old blind `map[num] = op` replaced the entry while
        // pending_publish/inc_pending_ops had been counted for BOTH: the counters
        // drifted up one per replacement and NEVER returned to zero for the chunk
        // (stuck owner word, perpetual spurious drains/gates, any_pending_ops
        // pinned), and the replaced op was never destroy()ed (no-op deleter ->
        // leaked pool slot). Publish/inc only for genuinely NEW entries; for a
        // replacement the count is already right (still one op at this offset)
        // and the reader window is already closed by the existing op's publish.
        const auto it_existing = ops_map_ptr->find(num);
        if (it_existing != ops_map_ptr->end())
        {
            it_existing->second->destroy(); // idempotent teardown of the superseded op
            it_existing->second = DeferredOpPtr(static_cast<DeferredOperationBase*>(v), DeferredOpDeleter{});
        }
        else
        {
            // Per-chunk pending counter: publish BEFORE inserting the op — the read
            // gate is `dirty && pending_any()`; publishing first makes pending>0
            // hold for the op's ENTIRE lifetime (the op itself is published to
            // flushers via the Dekker write flag, not this counter).
            chunk_ptr->pending_publish(thr_rank); // owner-tagged; no-op when counterless
            (*ops_map_ptr)[num] = DeferredOpPtr(static_cast<DeferredOperationBase*>(v), DeferredOpDeleter{});
            // Publish globally BEFORE releasing the writing flag, so a flusher that
            // observes !is_thread_writing(r) and proceeds to scan also observes it.
            CompressionManager<T, Derived>::inc_pending_ops(thr_rank);
        }
    }
    catch (const std::exception& e)
    {
        INVARIANT_FAILED("deferred enqueue failed: {} (rank {}, chunk {}, offset {})", e.what(), thr_rank,
                         static_cast<const void*>(chunk_ptr), num);
    }

    Policy::end_write(thr_rank);
    CompressionManager<T, Derived>::enqueue_unlock(thr_rank);

    // ── P3: bounded backlog drain (CHUNK_SYNC_ON_READ) ────────────────────────
    // Writes no longer flush (P1), so a write-only phase could let this rank's
    // deferred backlog grow without bound (memory + long tail on the eventual
    // read). Once it crosses the threshold, drain OUR OWN rank here — AFTER
    // end_write and enqueue_unlock, so no writer flag or shard lock is held while
    // flushing — and never while replaying a flush (flush_deferred sets
    // in_flush_replay and itself re-enters this function via write_body).
    //
    // Not while a compressor is stealing, though. A write is deferred BECAUSE
    // a compressor is in the way (must_defer_write), and flush_deferred() waits
    // the stealers out before it replays: draining from here parked the writer in sched_yield
    // for the whole compression -- 7% of the perf bench at 8 threads once the
    // deferred path stopped flushing every op on the spot. The threshold is
    // then honoured by the fast path, on the rank's next direct write
    // (write_with_optional_lock_impl), when no compressor is in the way.
    {
        using CM = CompressionManager<T, Derived>;
        if (!CM::in_flush_replay && !CM::is_hard_compression_inflight() &&
            CM::pending_ops_rank(thr_rank) > build_config::deferred_drain_threshold) [[unlikely]]
        {
            (void)CM::flush_deferred(thr_rank);
        }
    }
}

template <typename T, typename Derived>
void ChunkDeferredOpsPolicy<T, Derived, true>::wait_for_availability([[maybe_unused]] int num) const
{
    // HOTSPOT NOTE (x86, Aug '26): this function used to open with a naked
    // 2-instruction spin — `while (is_compression_active()) {}` — no
    // CPU_PAUSE, no bound, and on the WRONG predicate: is_compression_active
    // includes the DRIVER count, which under load is quasi-permanently
    // nonzero (any thread anywhere in the slow path). At 64-128 threads on
    // dual-socket x86, every reader piled into that spin (89% of all cycles;
    // init phase 93s @128T vs 1.4s on single-socket ARM), and the spinners'
    // seq_cst loads slowed the compressors' RMWs on the same line — a
    // feedback loop. Waiting on the driver count is also the July-'26
    // deadlock anti-pattern.
    //
    // The spin was REDUNDANT: flush_chunk_all_ranks() below already
    // establishes the only ordering that matters — begin_flush(self_rank)
    // blocks any future compressor start(), then a well-behaved drain
    // (CPU_PAUSE + yield, HARD count only) waits out in-flight compressors.
    // Driver-only threads steal nothing. So: no pre-wait, go flush.

    // CHUNK_STATS: we are on the SLOW sync path. This is the hotspot the notes
    // above describe (it once burned 89% of cycles on dual-socket x86), and it
    // was invisible in the counters. read_sync / (read + read_cst) is the share
    // of accesses that pay synchronisation at all.
    if constexpr (chunk_config::stats)
    {
        increm_chunk_read_sync<Derived>();
    }

    auto* chunk_ptr = static_cast<Derived*>(const_cast<ChunkDeferredOpsPolicy*>(this));

    // ── Stage 1: drain OUR OWN rank only (counter builds only) ─────────────
    // sync_chunk runs on the WRITE operators too (operator() non-const and
    // the proxy operator[]), so the dominant trigger by far is: a thread
    // re-touching a chunk whose pending op is ITS OWN, enqueued a few
    // elements earlier. Draining only self_rank costs one (recursive,
    // usually uncontended) lock instead of a full walk acquiring EVERY
    // rank's lock — the walk made every written element pay O(threads) lock
    // traffic and collapsed x86 at 64-128 threads even after the enqueue
    // convoy was fixed. This stage only makes sense with the per-chunk
    // counter (the early-out needs pending_any()).
    if constexpr (DeferredReadGate<Derived>::precise)
    {
        static_cast<void>(CompressionManager<T, Derived>::flush_chunk_all_ranks(chunk_ptr, /*self_only=*/true));
        if (!chunk_ptr->pending_any())
        {
            return; // every pending op was ours — no cross-rank walk needed
        }
    }

    // ── Stage 2: cross-thread flush (option 4) ────────────────────────────
    // Foreign ops remain (or no per-chunk counter in this build): flush every
    // pending op for this chunk across ALL ranks under the per-rank locks.
    // This subsumes the old own-rank-only flush_deferred(thr_rank) call,
    // which missed ops enqueued by other ranks — the root cause of the
    // lost-write corruption.
    // CHUNK_STATS: stage 1 did not settle it, so we take the expensive cross-rank
    // walk (every rank's lock). read_sync_cross / read_sync tells how often the
    // cheap self-rank drain was enough.
    if constexpr (chunk_config::stats)
    {
        increm_chunk_read_sync_cross<Derived>();
    }
    static_cast<void>(CompressionManager<T, Derived>::flush_chunk_all_ranks(chunk_ptr));
}

template <typename T, typename Derived>
bool ChunkDeferredOpsPolicy<T, Derived, true>::is_not_available() const
{
    // Cheap global cross-thread filter (option 4): if NO rank has any
    // pending op, this slot is available regardless of who wrote what.
    // A single sharded-sum acquire load — dominant early-out on the hot
    // read path.
    //
    // IMPORTANT: we deliberately do NOT scan other ranks' maps here.
    // Reading arr[r] for r != self_rank races with a concurrent insert on
    // rank r that can rehash ankerl's backing vector, turning our find()
    // into a use-after-free (observed: SIGSEGV in table::end with this=0x54
    // at 64 threads; ASan hides it by perturbing timing). Instead, any
    // pending op anywhere makes us return true, and wait_for_availability()
    // performs the authoritative per-chunk scan under the Dekker flush lock,
    // where all other ranks are quiescent and the scan is race-free.
    //
    // Cost: when any op is pending we may call wait_for_availability even if
    // THIS chunk has none. wait_for_availability then flushes this chunk
    // across all ranks (a no-op if empty) and returns. Correct, slightly
    // more conservative than a precise per-chunk test — but a precise test
    // cannot be done safely without a lock.

    return CompressionManager<T, Derived>::any_pending_ops();
}

} // namespace shrink_array