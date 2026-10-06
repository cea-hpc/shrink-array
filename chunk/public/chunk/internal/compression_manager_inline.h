
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

#include <atomic>      // std::atomic, std::memory_order, std::atomic_thread_fence
#include <cstdint>     // std::uintptr_t, uint32_t, int64_t
#include <exception>   // std::exception (erase failures)
#include <iostream>    // std::cerr (the counter checks)
#include <thread>      // std::this_thread — used HERE, not by our includers
#include <type_traits> // std::remove_reference_t (the swap-drain batch type)

#include "chunk/internal/compression_manager.h"        // the declarations these bodies belong to
#include "chunk/config.h"                              // build_config::any_check, chunk_config::ring_log_debug
#include "chunk/PMP_CHUNK.h"                           // get_thr_rank_fast, CHUNK_PMP_RANK_BOUND
#include "chunk/ring_log_debug.h"                      // ScopedRingLogManualEnd, ring_log_op
#include "chunk/internal/cpu_pause.h"                  // CPU_PAUSE
#include "chunk/internal/deferred_op_pool.h"           // DeferredOperationBase
#include "chunk/internal/invariant_check.h"            // INVARIANT_FAILED
#include "chunk/internal/write_compress_sync_policy.h" // WriteCompressSyncPolicy

namespace shrink_array
{

// No build guard: every definition below belongs to the PRIMARY
// CompressionManagerOmpPolicy template, which is only instantiated with
// Enabled=true -- i.e. only in OpenMP builds. The serial build selects the
// empty <..., false> specialisation and never reaches any of this.

// ── Per-rank flush lock (sharded xflush_lock) ─────────────────────────────
// Spin-acquire the flush lock for rank r. The library's spin discipline
// (CPU_PAUSE, yield after 1000 spins). Guards the per-rank op-map
// container arr[r] against concurrent structural mutation by another flusher.
template <typename T, typename ChunkType, bool Enabled>
void CompressionManagerOmpPolicy<T, ChunkType, Enabled>::rank_flush_acquire(int rank) noexcept
{
    const int my = get_thr_rank_fast();
    auto&     lk = xflush_lock_rank[rank];

    // Re-entrant acquire by the owner (write_body re-enqueue during a drain).
    if (lk.owner.load(std::memory_order_relaxed) == my)
    {
        ++lk.depth; // only the owner touches depth
        return;
    }

    int expected = -1;
    int spin     = 0;
    while (!lk.owner.compare_exchange_weak(expected, my, std::memory_order_acq_rel, std::memory_order_relaxed))
    {
        expected = -1;
        if (++spin > 1000)
        {
            std::this_thread::yield();
            spin = 0;
        }
        else
        {
            CPU_PAUSE();
        }
    }
    lk.depth = 1;
}

template <typename T, typename ChunkType, bool Enabled>
void CompressionManagerOmpPolicy<T, ChunkType, Enabled>::rank_flush_release(int rank) noexcept
{
    auto& lk = xflush_lock_rank[rank];
    if (--lk.depth > 0)
    {
        return; // still nested inside the outer drain
    }
    lk.owner.store(-1, std::memory_order_release);
}

template <typename T, typename ChunkType, bool Enabled>
bool CompressionManagerOmpPolicy<T, ChunkType, Enabled>::flush_deferred(int thr, bool /*wait_for_compression*/) noexcept
{
    if (deferred_chunk_per_thread.empty())
    {
        return false;
    }

    auto& deferred_chunk_loc = deferred_chunk_per_thread[thr];
    // Emptiness is checked under rank_flush_acquire(thr) below, NOT here:
    // flush_chunk_all_ranks erases from deferred_chunk_per_thread[thr] while
    // holding that same rank shard, so an unlocked .empty() read here is a data
    // race on the unordered_dense set (TSan-confirmed).

    auto* arr = ChunkType::get_deferred_ops();
    if (!arr)
    {
        return false;
    }

    auto& thread_map = arr[thr];

    using Policy = WriteCompressSyncPolicy<typename ChunkType::ProxyType>;

    // Serialize against flush_chunk_all_ranks (cross-thread flush): both may
    // drain arr[thr] for the same chunk, so they must not run concurrently.
    // flush_deferred touches ONLY arr[thr], so it holds this rank's shard for
    // the whole loop. flush_chunk_all_ranks acquires the same shard when (and
    // only when) it processes rank thr — giving mutual exclusion on arr[thr]
    // while letting flushes on other ranks proceed in parallel.
    rank_flush_acquire(thr);
    struct RankFlushGuard
    {
        ~RankFlushGuard()
        {
            rank_flush_release(rank);
        }

        int rank;
    } xflush_guard{thr};

    // Now under the rank shard: safe to read emptiness against concurrent erases.
    if (deferred_chunk_loc.empty())
    {
        return false; // RankFlushGuard releases the shard
    }

    // ─── Flush has priority over compression ──────────────────────────
    // ONE announce for the whole call, not one per chunk. begin_flush()
    // blocks any FUTURE compressor's start() (it returns false); we then
    // wait for any INFLIGHT compressor to drain -- they cannot be aborted,
    // but they finish in bounded time (finite chunk list). Once no
    // compressor is stealing, none can start while our flushing count is
    // held, so every chunk below is replayed under one exclusion.
    //
    // Not one announce per chunk: announce_flush() is a membarrier() (an
    // IPI to every CPU running one of our threads), and the bulk-init phases
    // flush tens of thousands of chunks per second per rank -- per-chunk
    // announces turned seconds into minutes at 64 threads. Holding the flag
    // across the loop only defers compressors a little longer, which they
    // already tolerate (start() yields to a flush).
    //
    // No livelock: bounded inflight drain x no new admissions.
    Policy::announce_flush(); // counter + heavy barrier, one Dekker verb
    increm_chunk_gate_flush<ChunkType>();
    struct FlushFlagGuard
    {
        ~FlushFlagGuard()
        {
            Policy::end_flush();
        }
    } flush_flag_guard;

    // Drain any compression that won the race. This loop terminates
    // because (a) inflight is monotonically non-increasing once we
    // hold the flushing flag, and (b) live compressors are finite.
    {
        int spin = 0;
        // HARD compressors only — never the driver count (deadlock
        // invariant, see compression_manager.h).
        while (is_hard_compression_inflight())
        {
            if (++spin > 1000)
            {
                std::this_thread::yield();
                spin = 0;
            }
            else
            {
                CPU_PAUSE();
            }
        }
    }

    // Walk a SNAPSHOT of this rank's chunk set, not the live set. Nothing
    // should insert into it meanwhile: a replayed write runs under
    // FlushReplayGuard (in_flush_replay) and is applied directly, never
    // enqueued -- record_enqueue_deferred() checks it -- and other threads'
    // inserts into rank thr's set take the shard lock held here. The walk no
    // longer depends on it: an insert into the live set cannot invalidate
    // this loop, and a chunk that still holds ops afterwards goes back into
    // the live set (end of the loop), where the next flush finds it.
    std::remove_reference_t<decltype(deferred_chunk_loc)> chunks;
    chunks.swap(deferred_chunk_loc);
    for (ChunkType* chunk : chunks)
    {
        const auto it_chunk = thread_map.find(chunk);
        if (it_chunk == thread_map.end())
        {
            continue;
        }

        const auto& ops_map_ptr = it_chunk->second;
        if (!ops_map_ptr)
        {
            continue;
        }

        auto* op_map = ops_map_ptr.get();

        const uint32_t payload = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(chunk) >> 4U) & 0x00FFFFFFU;
        // ─── Dekker held — execute deferred ops ───────────────────────────

        using FlushLog = ScopedRingLogManualEnd<EV_FLUSH_BEGIN, EV_FLUSH_END>;
        FlushLog log(payload);

        int64_t flushed_here = 0;
        // Drain by ONE swap: move the current entries into a local batch and
        // iterate that, never *op_map itself. A replayed write_body() applies
        // its write directly (in_flush_replay) and should enqueue nothing;
        // should an op land in *op_map all the same, it is left there for the
        // NEXT flush and the chunk stays in the set (below) -- not redrained
        // here (looping while !empty could livelock).
        {
            std::remove_reference_t<decltype(*op_map)> batch;
            batch.swap(*op_map);
            // Replay context: gate bypassed, writes applied directly — safe
            // here (begin_flush + hard drain above), see in_flush_replay.
            const FlushReplayGuard replay_guard;
            for (auto& [num, op_ptr] : batch)
            {
                if constexpr (chunk_config::ring_log_debug || chunk_config::ring_log_op_only)
                {
                    ring_log_op(get_tid(), EV_OP_FLUSH_ONE, static_cast<uint32_t>(num),
                                static_cast<const void*>(op_ptr.get()));
                }

                op_ptr->write_body();
                op_ptr->destroy();
                ++flushed_here;
            }
            // batch destroyed here; any reentrant enqueue now sits in *op_map
            // and will be drained by the NEXT flush (not redrained here).
        }

        // Keep the sharded pending counter in sync (read by any_pending_ops).
        dec_pending_ops(thr, flushed_here);

        // Per-chunk counter: -flushed AFTER draining/applying, release so a
        // reader that acquire-observes 0 also sees the write_body results.
        // (pending_retire is a no-op in counterless builds.)
        if (flushed_here > 0)
        {
            chunk->pending_retire(static_cast<int32_t>(flushed_here));
        }

        log.end();

        // Preserve concurrent re-enqueues, as flush_chunk_all_ranks does.
        // The batch was swapped out; ops enqueued into *op_map after the swap
        // are un-replayed pending writes. clear()+erase would drop them
        // (lost write + counter drift). Erase the entry only when empty;
        // otherwise keep it AND put the chunk back into the live set: the set
        // is the only thing flush_deferred walks, so a chunk left out of it
        // would never be flushed again, and flush_all_deferred's
        // storage.destroy() would discard its ops.
        if (op_map->empty())
        {
            (void)thread_map.erase(it_chunk);
        }
        else
        {
            try // ankerl throws on a bucket overflow; this function is noexcept
            {
                deferred_chunk_loc.insert(chunk);
            }
            catch (const std::exception& e)
            {
                INVARIANT_FAILED("flush_deferred: re-insert failed: {} (rank {}, chunk {})", e.what(), thr,
                                 static_cast<const void*>(chunk));
            }
        }
    }

    // Nothing put back (the usual case): hand the walked buffer back, so the
    // live set keeps its capacity from one flush to the next.
    chunks.clear();
    if (deferred_chunk_loc.empty())
    {
        deferred_chunk_loc.swap(chunks);
    }

    if (thread_map.empty())
    {
        clear_thread_has_ops(thr);
    }

    if constexpr (build_config::any_check)
    {
        // What is left in the set must be exactly the chunks with ops still
        // queued (put back above): anything else is a stale entry.
        for (ChunkType* c : deferred_chunk_loc)
        {
            const auto it = thread_map.find(c);
            if (it == thread_map.end() || !it->second || it->second->empty())
            {
                INVARIANT_FAILED("flush_deferred(thr={}): chunk {} left in the set with no op queued", thr,
                                 static_cast<const void*>(c));
            }
        }
    }

    return true;
}

// ── Cross-thread targeted flush (option 4) ────────────────────────────────
template <typename T, typename ChunkType, bool Enabled>
bool CompressionManagerOmpPolicy<T, ChunkType, Enabled>::flush_chunk_all_ranks(ChunkType* chunk,
                                                                               bool       self_only) noexcept
{
    auto* arr = ChunkType::get_deferred_ops();
    if (!arr)
    {
        return false;
    }

    using Policy = WriteCompressSyncPolicy<typename ChunkType::ProxyType>;

    const int self_rank = get_thr_rank_fast();
    const int max_thr   = CHUNK_PMP_RANK_BOUND;

    const uint32_t payload = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(chunk) >> 4U) & 0x00FFFFFFU;

    // ─── Cross-flush mutual exclusion is now PER RANK ─────────────────────
    // The double-free guard (two flushers draining the same arr[r][chunk] and
    // both destroy()-ing an op) only needs mutual exclusion on the rank whose
    // map is being drained. We therefore take xflush_lock_rank[r] INSIDE the
    // per-rank loop below, not a single global lock here — so concurrent
    // flush_chunk_all_ranks() calls draining DIFFERENT ranks run in parallel.
    // begin_flush()/inflight below are per-thread/global atomics (compressor
    // exclusion), independent of the container lock, so they stay unguarded.

    // ─── Take the flush lock on OUR rank ──────────────────────────────────
    // begin_flush(self_rank) blocks any future compressor start(); we then
    // drain inflight compressors. Once inflight == 0 and we hold the flush
    // flag, no compressor is mutating any chunk.
    Policy::announce_flush(); // counter + seq_cst fence, one Dekker verb
    increm_chunk_gate_flush<ChunkType>();

    {
        int spin = 0;
        // HARD compressors only — never the driver count (deadlock invariant,
        // see compression_manager.h).
        while (is_hard_compression_inflight())
        {
            if (++spin > 1000)
            {
                std::this_thread::yield();
                spin = 0;
            }
            else
            {
                CPU_PAUSE();
            }
        }
    }

    using FlushLog = ScopedRingLogManualEnd<EV_FLUSH_BEGIN, EV_FLUSH_END>;
    FlushLog log(payload);

    bool flushed_any = false;

    // Total ops drained for THIS chunk across all ranks. Feeds pending_retire
    // (a no-op in counterless builds — a plain int64 accumulation is cheaper
    // than gating it).
    int64_t total_flushed_chunk = 0;

    // ─── Scan the selected rank(s)' map for this chunk ────────────────────
    // self_only: just our rank (stage 1 of wait_for_availability) — one lock,
    // no cross-rank walk. Otherwise every rank (stage 2 / legacy behaviour).
    const int r_begin = self_only ? self_rank : 0;
    const int r_end   = self_only ? self_rank + 1 : max_thr;
    for (int r = r_begin; r < r_end; ++r)
    {
        auto& thread_map = arr[r];

        // Hold rank r's shard for the entire time we touch arr[r] (find, drain,
        // erase) and its dangerous-list. RAII so every exit (the two `continue`s
        // and normal fall-through) releases it. This is the mutual exclusion
        // that prevents two flushers from double-draining arr[r][chunk]; other
        // ranks' shards stay free so their flushes run concurrently.
        rank_flush_acquire(r);
        struct RankScopeGuard
        {
            ~RankScopeGuard()
            {
                rank_flush_release(rank);
            }

            int rank;
        } rank_scope_guard{r};

        // No writer wait needed here anymore: the enqueue path now takes
        // rank r's shard lock around its map insert, so holding
        // rank_flush_acquire(r) IS the mutual exclusion with rank r's writer.
        // (The former Dekker — writers backing off on the GLOBAL
        // num_thread_flushing — made every enqueue in the process wait for
        // ANY flush anywhere: a convoy that collapsed x86 at 64-128 threads.)

        const auto it_chunk = thread_map.find(chunk);
        if (it_chunk == thread_map.end())
        {
            continue;
        }

        const auto& ops_map_ptr = it_chunk->second;
        if (!ops_map_ptr)
        {
            continue;
        }

        auto*   op_map       = ops_map_ptr.get();
        int64_t flushed_here = 0;
        // Drain by ONE swap (see flush_deferred): iterate a local batch, leave
        // reentrant enqueues in *op_map for the next flush. Do NOT loop to
        // redrain — that can livelock if each write_body re-enqueues.
        {
            std::remove_reference_t<decltype(*op_map)> batch;
            batch.swap(*op_map);
            // Replay context: gate bypassed, writes applied directly — safe
            // here (begin_flush + hard drain above), see in_flush_replay.
            const FlushReplayGuard replay_guard;
            for (auto& [num, op_ptr] : batch)
            {
                // Check-build guard: same probe as flush_deferred — see comment
                // there. Cross-thread path (flush_chunk_all_ranks).
                if constexpr (build_config::any_check)
                {
                    DeferredOperationBase* raw  = op_ptr.get();
                    const auto             addr = reinterpret_cast<std::uintptr_t>(raw);
                    if (raw == nullptr || addr < 0x1000 || (addr & 0x7U) != 0)
                    {
                        INVARIANT_FAILED("[FLUSH_PROBE xrank] bad op_ptr in op_map: num={} op={} rank={}", num,
                                         static_cast<const void*>(raw), r);
                    }
                    const std::uintptr_t vptr = *reinterpret_cast<const std::uintptr_t*>(raw);
                    if (vptr < 0x1000)
                    {
                        INVARIANT_FAILED("[FLUSH_PROBE xrank] op has null/garbage vptr (recycled under map?): "
                                         "num={} op={} vptr={:#x} rank={}",
                                         num, static_cast<const void*>(raw), vptr, r);
                    }
                }

                // Ring-log trace of the cross-rank flush, same as flush_deferred;
                // compiled only under ring_log_debug / ring_log_op_only.
                if constexpr (chunk_config::ring_log_debug || chunk_config::ring_log_op_only)
                {
                    ring_log_op(get_tid(), EV_OP_FLUSH_ONE, static_cast<uint32_t>(num),
                                static_cast<const void*>(op_ptr.get()));
                }

                op_ptr->write_body();
                op_ptr->destroy();
                ++flushed_here;
            }
            // batch destroyed here; reentrant enqueues are now in *op_map,
            // picked up by the next while pass.
        }

        if (flushed_here > 0)
        {
            dec_pending_ops(r, flushed_here);
            total_flushed_chunk += flushed_here;
            flushed_any = true;
        }

        // ── Preserve concurrent re-enqueues ──────────────────────────────────
        // We drained a SWAPPED-OUT batch; any op enqueued into *op_map AFTER the
        // swap (a writer still active on rank r during this reader-triggered
        // flush) is a genuine pending write that has NOT been replayed. A
        // `op_map->clear(); thread_map.erase()` would destroy exactly those ops
        // without applying them AND without decrementing pending_ops for them
        // (dec only counts the batch) -- a lost published write plus a counter
        // that no longer matches reality. Never clear() the live map: erase the
        // chunk entry ONLY when nothing was re-enqueued; otherwise leave it (and
        // the dangerous-list membership) for the next flush to pick up, exactly
        // as the batch-swap comment promises.
        if (op_map->empty())
        {
            // ankerl's erase throws on a key it finds changed (a corrupted
            // map); this function is noexcept: say where instead of terminate.
            try
            {
                thread_map.erase(it_chunk);
                // Drop the chunk from rank r's dangerous-list set so a later
                // flush_deferred(r) doesn't try to re-find an absent entry.
                deferred_chunk_per_thread[r].erase(chunk);
            }
            catch (const std::exception& e)
            {
                INVARIANT_FAILED("flush_chunk_all_ranks: erase failed: {} (rank {}, chunk {})", e.what(), r,
                                 static_cast<const void*>(chunk));
            }
        }
    }

    Policy::end_flush();
    log.end();

    // Per-chunk counter: subtract everything drained across ranks (mirrors
    // flush_deferred). Independent atomic; each op is drained exactly once (the
    // per-rank shard makes flush paths mutually exclusive on any given rank),
    // so no underflow even without a container lock held here. Release pairs
    // with the reader's acquire load.
    if (total_flushed_chunk > 0)
    {
        chunk->pending_retire(static_cast<int32_t>(total_flushed_chunk));
    }

    return flushed_any;
}

// ── WAW guard: discard a superseded same-offset op on the own rank ───────────
template <typename T, typename ChunkType, bool Enabled>
void CompressionManagerOmpPolicy<T, ChunkType, Enabled>::erase_own_rank_op(int thr, ChunkType* chunk, int num) noexcept
{
    auto* arr = ChunkType::get_deferred_ops();
    if (!arr)
    {
        return;
    }

    // Serialise against a concurrent flush of OUR rank for this chunk (the same
    // shard lock flush_deferred / flush_chunk_all_ranks take). Owner-recursive;
    // uncontended in the common case. Held only for the single-entry erase.
    rank_flush_acquire(thr);
    struct RankFlushGuard
    {
        ~RankFlushGuard()
        {
            rank_flush_release(rank);
        }

        int rank;
    } xflush_guard{thr};

    auto&      thread_map = arr[thr];
    const auto it_chunk   = thread_map.find(chunk);
    if (it_chunk == thread_map.end())
    {
        return;
    }

    const auto& ops_map_ptr = it_chunk->second;
    if (!ops_map_ptr)
    {
        return;
    }
    auto* op_map = ops_map_ptr.get();

    const auto it_op = op_map->find(num);
    if (it_op == op_map->end())
    {
        return; // no older op at this offset — the direct write is unshadowed
    }

    // APPLY-then-erase (not discard): the direct write that follows supersedes
    // the queued value, so discarding LOOKED equivalent — but the write body
    // reads v_old (read_values) to feed the incremental compression counters
    // (mark_for_compression_if_needed_with_value). Discarding left values[num]
    // at its pre-op state, so v_old was OLDER than program order and the
    // counters received a wrong (old->new) transition — a silent corruption
    // vector on counter-trusted compression paths, frequent exactly where
    // pendings survive across parallel regions (no per-barrier drains: gnu /
    // libgomp). Replaying the single op here restores the OFF-mode program
    // order: op value applied, THEN our direct write overwrites it. The body
    // runs under FlushReplayGuard so it writes directly (no re-defer, no lock
    // machinery — this is one op body, not a flush walk).
    {
        const FlushReplayGuard replay_guard;
        it_op->second->write_body();
    }
    it_op->second->destroy();
    try // ankerl's erase throws on a key it finds changed; this function is noexcept
    {
        op_map->erase(it_op);
    }
    catch (const std::exception& e)
    {
        INVARIANT_FAILED("erase_own_rank_op: erase failed: {} (rank {}, chunk {}, offset {})", e.what(), thr,
                         static_cast<const void*>(chunk), num);
    }

    // Exact bookkeeping of the flush erase path.
    dec_pending_ops(thr, 1);
    // release so a reader that acquire-observes the decremented per-chunk count
    // also sees the direct store that follows. No-op in counterless builds.
    chunk->pending_retire(1);

    // If this chunk has no more ops on our rank, drop it from the rank tracking
    // structures exactly as flush_deferred does.
    if (op_map->empty())
    {
        try // ankerl's erase throws on a key it finds changed; this function is noexcept
        {
            thread_map.erase(it_chunk);
            deferred_chunk_per_thread[thr].erase(chunk);
        }
        catch (const std::exception& e)
        {
            INVARIANT_FAILED("erase_own_rank_op: chunk erase failed: {} (rank {}, chunk {})", e.what(), thr,
                             static_cast<const void*>(chunk));
        }
        if (thread_map.empty())
        {
            clear_thread_has_ops(thr);
        }
    }
}

template <typename T, typename ChunkType, bool Enabled>
template <typename ProxyType>
bool CompressionManagerOmpPolicy<T, ChunkType, Enabled>::start()
{
    if constexpr (chunk_config::ring_log_debug)
    {
        ring_log_start(get_tid(), EV_COMPRESS_STARTED);
    }

    add_compression_active();
    increm_chunk_gate_start<ChunkType>();

    const int max_thr = CHUNK_PMP_RANK_BOUND;

    // ─── Yield to flush_deferred ──────────────────────────────────────
    // After incrementing hard_inflight_ and the seq_cst fence, check
    // whether any thread has already claimed a flush. If so, undo our
    // increment and return false — the caller (make_block_compression_impl)
    // already handles this by skipping compression. This implements
    // "defer compression, but not flush_deferred".
    //
    // This first look is only an early-out: the flush drains spin on
    // stealing_inflight_, not on hard_inflight_, so the load that actually
    // pairs with their announce_flush() is the one AFTER the writer wait,
    // once stealing_inflight_ is published (see the end of this function).
    if (WriteCompressSyncPolicy<ProxyType>::any_thread_flushing(max_thr))
    {
        increm_chunk_gate_start_flush_yield<ChunkType>();
        decrement_inflight();
        if constexpr (chunk_config::ring_log_debug)
        {
            ring_log_end(get_tid(), EV_COMPRESS_FINISHED);
        }

        return false;
    }

    // The compressor's announce towards the WRITERS: the seq_cst increment
    // above, then the heavy barrier -- a fence, and under the asymmetric
    // protocol the membarrier() that stands in for the fence the writers no
    // longer execute (WriteCompressSyncPolicy::announce_write() says why one
    // side then always sees the other). AFTER the flush early-out on
    // purpose: that early-out pairs with nothing (see above), and on the
    // bulk-init phases it is taken 96% of the time -- a membarrier() before
    // it was an IPI storm for compressions that never ran.
    dekker::heavy_barrier();

    // Drain any write_body() calls already in progress (existing logic).
    // Bounded wait for a "no writer" window. Under heavy write load (e.g. many
    // threads assigning to a short-typed array, where most chunks are
    // compressible so perform_compression fires constantly) that window may never
    // open, and an unbounded spin here livelocks every compressing thread at ~97%
    // CPU in sched_yield with no progress. Bound the wait: after a yield budget,
    // give up and return false. The caller (make_block_compression) then skips
    // compression WITHOUT touching P — the chunks stay queued and are retried on
    // the next opportunistic pass, and are drained for good by the dedicated
    // compress_chunk_to_compress OMP region between phases (which runs with no
    // concurrent writers and no start() barrier). So deferring here loses no
    // compression; it only avoids spinning when the opportunistic window is shut.
    int spin   = 0;
    int yields = 0;
// Spin budget before deferring the opportunistic compression. Under heavy
// write load the "no writer" window barely opens, so a large budget makes a
// compressing thread spin for a long time (seconds) — the periodic global
// stalls observed at high thread counts. Since deferring loses no
// compression (the dedicated between-phase region drains the chunks), a low
// budget is usually better: give up fast and keep writing. Override with
// -DCHUNK_COMPRESS_MAX_YIELDS=<n> (0 = give up on the first blocked check).
#ifndef CHUNK_COMPRESS_MAX_YIELDS
    #define CHUNK_COMPRESS_MAX_YIELDS 10000
#endif
    constexpr int kMaxYields = CHUNK_COMPRESS_MAX_YIELDS; // tune via -D

    // Bounded wait for a "no writer" window, HOLDING our hard increment:
    // that is deliberate — the write-path gate consults hard_inflight_, so a
    // parked compressor steers new writers to the slow path and the window
    // actually opens (compressor priority). This cannot re-create the parked
    // livelock triangle: since the Parked-vs-Stealing split, create_value and
    // the flush drains wait on stealing_inflight_ only, which we have NOT
    // incremented yet.
    // With the per-chunk marks the window is "no writer on a chunk this
    // pass MARKED" -- the writers of the other chunks are not in the way and
    // are neither waited for nor steered away; without the marks it is "no
    // writer anywhere", as it always was.
    const auto window_shut = [max_thr]
    {
        if constexpr (chunk_config::use_chunk_steal_mark)
        {
            return WriteCompressSyncPolicy<ProxyType>::template any_thread_writing_on_marked<ChunkType>(max_thr);
        }
        else
        {
            return WriteCompressSyncPolicy<ProxyType>::any_thread_writing(max_thr);
        }
    };
    while (window_shut())
    {
        if (++spin > 1000)
        {
            std::this_thread::yield();
            spin = 0;
            if (++yields > kMaxYields)
            {
                // Give up this opportunistic compression: undo the active-count
                // increment (symmetric with add_compression_active() above, as in
                // the any_thread_flushing early-out) and report deferral.
                increm_chunk_gate_start_deferred<ChunkType>();
                decrement_inflight();
                if constexpr (chunk_config::ring_log_debug)
                {
                    ring_log_end(get_tid(), EV_COMPRESS_FINISHED);
                }
                return false;
            }
        }
        else
        {
            CPU_PAUSE();
        }
    }

    // Writer gate established. Publish the transition Parked -> Stealing;
    // finish() undoes it. seq_cst pairs with the waiters' loads in
    // is_hard_compression_inflight().
    stealing_inflight_.fetch_add(1, std::memory_order_seq_cst);
    std::atomic_thread_fence(std::memory_order_seq_cst);

    // Re-check the flushers AFTER publishing stealing_inflight_: this is the
    // Dekker pair the flush drains actually rely on. flush_deferred() and
    // flush_chunk_all_ranks() do announce_flush() (flushing++, fence) and then
    // spin on stealing_inflight_ -- NOT on hard_inflight_ (waiting on a parked
    // compressor is the livelock triangle, see compression_manager.h). So the
    // any_thread_flushing() check at the top of this function, taken while we
    // were only parked, pairs with nothing: a flusher that announced itself
    // between that check and the fetch_add above saw stealing == 0 and is
    // already replaying deferred write bodies. Without this second look we
    // would scan and CAS-steal `values` underneath that replay: the scan sees
    // the pre-replay constant run, the replay's store lands (its post-store
    // pointer check passes, the CAS has not happened yet), then the CAS parks
    // the chunk back on the constant and the replayed write is lost -- seen as
    // a stale second value at one index of a Chunk2A (shrink_bench, tab3,
    // "read : 0 != 1", once in ~2300 runs).
    //
    // Before the Parked/Stealing split start() re-took hard_inflight_ and
    // re-checked the flushers after the writer wait; the split dropped that
    // second check. Flush keeps priority ("defer compression, not
    // flush_deferred"): a flusher never backs off, we do.
    if (WriteCompressSyncPolicy<ProxyType>::any_thread_flushing(max_thr))
    {
        increm_chunk_gate_start_flush_yield<ChunkType>();
        increm_chunk_gate_start_yield_late<ChunkType>();
        stealing_inflight_.fetch_sub(1, std::memory_order_seq_cst);
        decrement_inflight();
        if constexpr (chunk_config::ring_log_debug)
        {
            ring_log_end(get_tid(), EV_COMPRESS_FINISHED);
        }
        return false;
    }

    return true;
}

// Yield budget of the writer-side wait at the gate. With the per-chunk marks
// the wait is a parked compressor's window (microseconds) or the pass
// reaching this chunk; without them it is the whole pass, scan included
// (~300 us at 6 threads on x86, ~1 ms at 64 on AArch64). Past the budget the
// write is deferred, as it would be without the wait, so a bigger budget
// only buys fewer deferrals when a pass is unusually
// long. A yield is ~1 us; the 1000 pauses before it are ~50 us on x86 and
// under 1 us on AArch64 (`yield` is a hint there), so the budget in time is
// mostly the yields.
#ifndef CHUNK_WRITE_WAIT_MAX_YIELDS
    #define CHUNK_WRITE_WAIT_MAX_YIELDS 64
#endif

template <typename T, typename ChunkType, bool Enabled>
void CompressionManagerOmpPolicy<T, ChunkType, Enabled>::wait_out_compressor(const ChunkType& chunk) noexcept
{
    constexpr int kMaxYields = CHUNK_WRITE_WAIT_MAX_YIELDS;

    int                         spin     = 0;
    int                         yields   = 0;
    [[maybe_unused]] const bool mark_hit = chunk.steal_marked();
    // Relaxed, the same predicate as the gate: the decision is re-taken by
    // the caller's gate loads, which the handshake orders (announce_write,
    // then check #2).
    while (must_defer_write(chunk))
    {
        if (++spin > 1000)
        {
            std::this_thread::yield();
            spin = 0;
            if (++yields > kMaxYields)
            {
                break; // budget spent: the caller defers, as it always did
            }
        }
        else
        {
            CPU_PAUSE();
        }
    }
    increm_chunk_gate_wait<ChunkType>(yields, mark_hit);
}

template <typename T, typename ChunkType, bool Enabled>
template <typename ProxyType>
void CompressionManagerOmpPolicy<T, ChunkType, Enabled>::finish([[maybe_unused]] bool log)
{
    // Undo the Parked -> Stealing transition FIRST: waiters on
    // stealing_inflight_ (create_value, flush drains) may proceed as soon as
    // we are no longer allowed to steal; hard follows for the gate/remaining
    // accounting. finish() only runs after a start() that returned true, so
    // the counters are always paired.
    stealing_inflight_.fetch_sub(1, std::memory_order_seq_cst);

    const int remaining = decrement_inflight();

    std::atomic_thread_fence(std::memory_order_seq_cst);

    if (remaining < 0)
    {
        if constexpr (verbose)
        {
            std::cerr << "🐞 Compression counter went negative!\n";
        }
        return;
    }

    const int thr = get_thr_rank_fast();

    if (remaining > 0)
    {
        if constexpr (verbose)
        {
            std::cerr << "⚠️ Compression still active in another thread\n";
        }
        return;
    }

    // Ici: remaining == 0 → dernier thread de compression

    // Mark the list as safe again.
    mark_list_safe_for_thread(thr);

    if constexpr (chunk_config::ring_log_debug)
    {
        ring_log_end(get_tid(), EV_COMPRESS_FINISHED);
    }
}


} // namespace shrink_array