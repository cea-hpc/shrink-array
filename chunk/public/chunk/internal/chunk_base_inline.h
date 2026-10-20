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

#include <exception>   // std::exception (deferred enqueue)
#include <memory>      // std::make_unique
#include <type_traits> // std::remove_reference_t

#include "chunk/internal/chunk_base.h"                 // the declarations these bodies belong to
#include "chunk/config.h"                              // build_config::use_fast_alloc, chunk_config::
#include "chunk/internal/chunk_stats.h"                // increm_chunk_read_sync / _cross (CHUNK_STATS)
#include "chunk/internal/pool_ledger.h"                // CHUNK_LEDGER_PARK / CHUNK_LEDGER_RELEASE
#include "chunk/PMP_CHUNK.h"                           // CHUNK_PMP_IN_PARALLEL_REGION
#include "chunk/internal/deferred_op_pool.h"           // DeferredOperationBase, DeferredOpDeleter, DeferredOpPtr
#include "chunk/internal/deferred_ops_storage.h"       // DeferredOpsStorage
#include "chunk/internal/freepath_counters.h"          // CHUNK_FREEPATH_INC
#include "chunk/internal/invariant_check.h"            // CHUNK_INVARIANT_FAILED
#include "chunk/internal/read_reclaim.h"               // read_reclaim::defer_release
#include "fast_allocators/internal/svalloc.h"          // svfree
#include "chunk/internal/write_compress_sync_policy.h" // WriteCompressSyncPolicy

namespace shrink_array
{

template <typename T, typename Derived>
void ChunkBase<T, Derived>::free_block_to_pool(this Derived& self, [[maybe_unused]] int thr_rank, T* val)
{
    // A pass that may run while other threads read (read_reclaim::PassScope:
    // perform_compression, compress() inside a region) collects the buffer
    // instead: a reader may still hold it between its `values` load and its
    // element load. The pass releases it here, after one grace period.
    if (read_reclaim::defer_release(&ChunkBase::release_deferred_block, &self, thr_rank, val))
    {
        return;
    }

    if constexpr (build_config::use_fast_alloc)
    {
        // The block's slab says which array it belongs to (owner_key): the reuse list and
        // budget are that array's, on this rank. Key 0 is the shared pool.
        const auto&    mgr               = self.get_manager();
        auto&          allocator         = self.get_chunk_allocator(thr_rank);
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
                using AllocT = std::remove_reference_t<decltype(allocator)>;
                const_cast<AllocT*>(static_cast<const AllocT*>(slab->owner))->free_elt_pow2(val, slab);
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
                    // Do NOT push a foreign block onto ANOTHER thread's free-list: the
                    // unsynchronised push races with the owner's own pop/push (vector
                    // corruption), and a later free_elt_pow2 would route the block to the
                    // wrong allocator's lists (delete / munmap of a live block). Same rule
                    // as recycle() on the op pool side.
                    //
                    // Keep it on the LOCAL free-list: is_free_able_elt is false for it, so
                    // it is never freed locally, only reused directly; the owning slab
                    // releases it at teardown. Bounded leak, not a UAF.
                    CHUNK_FREEPATH_INC(kept_nomigrate);
                    CHUNK_LEDGER_PARK(val, "free_block_to_pool.keep_local_no_migrate");
                    ll_block_free_thr.push_back(val);
                }
            }
        }
    }
    else
    {
        // Ledger LIVE -> FREE, before the svfree while the pointer is still ours: otherwise
        // malloc re-hands the page to a later allocation and on_block_alloc sees it still
        // LIVE (spurious invariant failure). free_values is the other such path.
        CHUNK_LEDGER_RELEASE(val, "free_block_to_pool.fast_alloc::svfree");
        fast_alloc::svfree(val);
    }
}

// Bodies of ChunkDeferredOpsPolicy<T, Derived, true> (chunk_base.h). Every name they use is
// declared in every build, and the policy is never instantiated in a serial build. They cast
// `this` to Derived* directly, not through an explicit object parameter: these members are
// called from sibling policies whose *this is not a Derived (see chunk_base.h).

template <typename T, typename Derived>
void ChunkDeferredOpsPolicy<T, Derived, true>::to_be_queued_for_deferred_processing(
    int                                                thr_rank,
    int                                                num,
    CompressionManager<T, Derived>::CmpDeferredOpBase* v) const
{
    ensure_deferred_array_initialized();

    // Cached per thread: no guard check per call; the singleton outlives every thread.
    thread_local auto* storage = &DeferredOpsStorage<per_thread_map>::instance();

    auto* arr = storage->get(); // simple acquire load, always fresh

    auto* chunk_ptr = static_cast<Derived*>(const_cast<ChunkDeferredOpsPolicy*>(this));

    // ── Enqueue vs cross-thread flush: PER-RANK shard lock ────────────────
    // The requirement is pairwise (flusher of rank r <-> writer of rank r), which is what
    // xflush_lock_rank[thr_rank] provides. Do not replace it by a back-off on a GLOBAL flush
    // counter: under read-heavy phases flushes are quasi-permanent and writers convoy in a
    // spin while pending ops cannot drain (it dominated the profile at high thread counts).
    // Owner-recursive: a drain's write_body may re-enter here on the own rank.
    //
    // announce_write/end_write are the writer half of the compressor-yield protocol (start()
    // spins on any_thread_writing), orthogonal to map safety.
    using Policy = WriteCompressSyncPolicy<typename Derived::ProxyType>;

    CompressionManager<T, Derived>::enqueue_lock(thr_rank);
    Policy::announce_write(thr_rank, chunk_ptr); // flag + seq_cst fence, one Dekker verb

    // The container steps below allocate, and ankerl's maps can throw. Callers are noexcept,
    // so an escaping exception would be a silent std::terminate: report where instead.
    try
    {
        per_thread_map& thread_map = arr[thr_rank];

        auto& ops_map_ptr = thread_map[chunk_ptr];

        if (!ops_map_ptr)
        {
            ops_map_ptr = std::make_unique<unordered_map_deferred_operations>();
        }

        // ── Same-offset re-enqueue: REPLACE, do not double-count ──────────────
        // A second deferred write to the SAME offset on the SAME rank supersedes the queued
        // one. Do not use a blind `map[num] = op`: the counters would count both, drift up and
        // never return to zero (any_pending_ops pinned), and the replaced op would leak.
        // Publish/inc only for NEW entries; for a replacement the count is already right.
        const auto it_existing = ops_map_ptr->find(num);
        if (it_existing != ops_map_ptr->end())
        {
            it_existing->second->destroy(); // idempotent teardown of the superseded op
            it_existing->second = DeferredOpPtr(static_cast<DeferredOperationBase*>(v), DeferredOpDeleter{});
        }
        else
        {
            // Publish the per-chunk counter BEFORE inserting: the read gate is
            // `dirty && pending_any()`, so pending>0 must hold for the op's whole lifetime
            // (flushers see the op via the Dekker write flag, not this counter).
            chunk_ptr->pending_publish(thr_rank); // owner-tagged; no-op when counterless
            (*ops_map_ptr)[num] = DeferredOpPtr(static_cast<DeferredOperationBase*>(v), DeferredOpDeleter{});
            // Publish globally BEFORE releasing the writing flag, so a flusher that
            // observes !is_thread_writing(r) and proceeds to scan also observes it.
            CompressionManager<T, Derived>::inc_pending_ops(thr_rank);
        }
    }
    catch (const std::exception& e)
    {
        CHUNK_INVARIANT_FAILED("deferred enqueue failed: {} (rank {}, chunk {}, offset {})", e.what(), thr_rank,
                               static_cast<const void*>(chunk_ptr), num);
    }

    Policy::end_write(thr_rank);
    CompressionManager<T, Derived>::enqueue_unlock(thr_rank);

    // ── Bounded backlog drain ─────────────────────────────────────────────────
    // Writes do not flush, so a write-only phase could grow this rank's backlog without bound.
    // Past the threshold, drain OUR OWN rank -- after end_write and enqueue_unlock (no flag or
    // lock held), and never while replaying a flush (flush_deferred re-enters here).
    //
    // Not while a compressor is stealing: the write was deferred because of it, and
    // flush_deferred() would park the writer for the whole compression. The fast path
    // (write_with_optional_lock_impl) honours the threshold on the next direct write.
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
    // No pre-wait for compressors here, and never on a predicate that counts DRIVERS: under
    // load it is quasi-permanently non-zero, every reader piles into the spin, and it is the
    // deadlock anti-pattern documented on hard_inflight_. flush_chunk_all_ranks() already
    // provides the only ordering that matters: begin_flush(self_rank) blocks any future
    // compressor start(), then the drain waits out in-flight stealers (driver-only threads
    // steal nothing).

    // CHUNK_STATS: the SLOW sync path. read_sync / (read + read_cst) is the share of accesses
    // that pay synchronisation at all.
    if constexpr (chunk_config::stats)
    {
        increm_chunk_read_sync<Derived>();
    }

    auto* chunk_ptr = static_cast<Derived*>(const_cast<ChunkDeferredOpsPolicy*>(this));

    // ── Stage 1: drain OUR OWN rank only ───────────────────────────────────
    // sync_chunk runs on the write operators too, so the dominant case is a thread re-touching
    // a chunk whose pending op is ITS OWN. Draining self_rank costs one (recursive, usually
    // uncontended) lock; do not go straight to the full walk, which makes every written element
    // pay O(threads) lock traffic. The per-chunk counter (pending_any()) says when it sufficed.
    static_cast<void>(CompressionManager<T, Derived>::flush_chunk_all_ranks(chunk_ptr, /*self_only=*/true));
    if (!chunk_ptr->pending_any())
    {
        return; // every pending op was ours — no cross-rank walk needed
    }

    // ── Stage 2: cross-thread flush ────────────────────────────────────────────
    // Foreign ops remain: flush this chunk across ALL ranks under the
    // per-rank locks; an own-rank-only flush would lose other ranks' writes.
    // CHUNK_STATS: read_sync_cross / read_sync tells how often stage 1 was not enough.
    if constexpr (chunk_config::stats)
    {
        increm_chunk_read_sync_cross<Derived>();
    }
    static_cast<void>(CompressionManager<T, Derived>::flush_chunk_all_ranks(chunk_ptr));
}

} // namespace shrink_array