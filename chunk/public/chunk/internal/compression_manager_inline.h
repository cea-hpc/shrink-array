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
#include <thread>      // std::this_thread — used HERE, not by our includers
#include <type_traits> // std::remove_reference_t (the swap-drain batch type)

#include "chunk/internal/compression_manager.h"        // the declarations these bodies belong to
#include "chunk/config.h"                              // build_config::any_check, chunk_config::ring_log_debug
#include "chunk/PMP_CHUNK.h"                           // get_thr_rank_fast, CHUNK_PMP_RANK_BOUND
#include "chunk/ring_log_debug.h"                      // ScopedRingLogManualEnd, ring_log_op
#include "chunk/internal/cpu_pause.h"                  // CHUNK_CPU_PAUSE
#include "chunk/internal/deferred_op_pool.h"           // DeferredOperationBase
#include "chunk/internal/invariant_check.h"            // CHUNK_INVARIANT_FAILED
#include "chunk/internal/write_compress_sync_policy.h" // WriteCompressSyncPolicy

namespace shrink_array
{

// No build guard: everything below belongs to the PRIMARY CompressionManagerOmpPolicy
// template, instantiated only with Enabled=true (OpenMP builds).

// ── Per-rank flush lock (sharded xflush_lock) ─────────────────────────────
// Spin-acquire the flush lock for rank r (CHUNK_CPU_PAUSE, yield after 1000 spins).
// Guards the per-rank op-map arr[r] against structural mutation by another flusher.
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
            CHUNK_CPU_PAUSE();
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
bool CompressionManagerOmpPolicy<T, ChunkType, Enabled>::flush_deferred(int thr) noexcept
{
    if (deferred_chunk_per_thread.empty())
    {
        return false;
    }

    auto& deferred_chunk_loc = deferred_chunk_per_thread[thr];
    // Do not test emptiness here: flush_chunk_all_ranks erases from this set under the
    // rank-thr shard, so an unlocked .empty() is a data race. Checked under the lock below.

    auto* arr = ChunkType::get_deferred_ops();
    if (arr == nullptr)
    {
        return false;
    }

    auto& thread_map = arr[thr];

    using Policy = WriteCompressSyncPolicy<typename ChunkType::ProxyType>;

    // Serialise against flush_chunk_all_ranks: both may drain arr[thr] for the same chunk.
    // We touch ONLY arr[thr] and hold its shard for the whole loop; flush_chunk_all_ranks
    // takes the same shard only while processing rank thr, so other ranks flush in parallel.
    const RankFlushLock xflush_guard(thr);

    if (deferred_chunk_loc.empty())
    {
        return false; // RankFlushLock releases the shard
    }

    // ─── Flush has priority over compression ──────────────────────────
    // The announce blocks any FUTURE compressor's start() (it returns false); we then
    // wait for INFLIGHT compressors, which cannot be aborted but finish in bounded time.
    // Once none is stealing, none can start while our flushing count is held, so every
    // chunk below is replayed under one exclusion. No livelock: bounded drain, no new
    // admissions.
    //
    // ONE announce per call, not per chunk: announce_flush() is a membarrier() (an IPI to
    // every CPU running one of our threads) and bulk-init phases flush many thousands of
    // chunks per second per rank; per-chunk announces slow them by an order of magnitude.
    // Holding the flag across the loop only defers compressors a little longer.
    Policy::announce_flush(); // counter + heavy barrier, one Dekker verb
    increm_chunk_gate_flush<ChunkType>();
    struct FlushFlagGuard
    {
        FlushFlagGuard() = default;
        ~FlushFlagGuard()
        {
            Policy::end_flush();
        }
        FlushFlagGuard(const FlushFlagGuard&)            = delete;
        FlushFlagGuard& operator=(const FlushFlagGuard&) = delete;
        FlushFlagGuard(FlushFlagGuard&&)                 = delete;
        FlushFlagGuard& operator=(FlushFlagGuard&&)      = delete;
    };
    const FlushFlagGuard flush_flag_guard;

    // Drain any compression that won the race. Terminates: inflight cannot grow while we
    // hold the flushing flag, and live compressors are finite.
    {
        int spin = 0;
        // HARD compressors only, never the driver count (deadlock invariant, see
        // compression_manager.h).
        while (is_hard_compression_inflight())
        {
            if (++spin > 1000)
            {
                std::this_thread::yield();
                spin = 0;
            }
            else
            {
                CHUNK_CPU_PAUSE();
            }
        }
    }

    // Walk a SNAPSHOT of this rank's chunk set, not the live set. Nothing should insert
    // meanwhile (a replayed write runs under FlushReplayGuard and is applied directly --
    // record_enqueue_deferred() checks it -- and other threads take the shard lock held
    // here), but the walk does not depend on it: an insert cannot invalidate this loop, and
    // a chunk still holding ops goes back into the live set (end of the loop).
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
        // Drain by ONE swap into a local batch; never iterate *op_map itself. A replayed
        // write_body() should enqueue nothing; an op that lands in *op_map anyway is left
        // for the NEXT flush (the chunk stays in the set, below). Do not loop while
        // !empty: that could livelock.
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
        }

        // Keep the sharded pending counter in sync (read by any_pending_ops).
        dec_pending_ops(thr, flushed_here);

        // Per-chunk counter, AFTER applying, with release: a reader that acquire-observes 0
        // also sees the write_body results. (No-op in counterless builds.)
        if (flushed_here > 0)
        {
            chunk->pending_retire(static_cast<int32_t>(flushed_here));
        }

        log.end();

        // Preserve concurrent re-enqueues (see flush_chunk_all_ranks): erase the entry only
        // when empty; otherwise keep it AND put the chunk back into the live set, the only
        // thing flush_deferred walks -- left out, its ops would never be flushed and
        // flush_all_deferred's storage.destroy() would discard them.
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
                CHUNK_INVARIANT_FAILED("flush_deferred: re-insert failed: {} (rank {}, chunk {})", e.what(), thr,
                                       static_cast<const void*>(chunk));
            }
        }
    }

    // Nothing put back (the usual case): hand the buffer back so the live set keeps its capacity.
    chunks.clear();
    if (deferred_chunk_loc.empty())
    {
        deferred_chunk_loc.swap(chunks);
    }

    // The flag is the CALLING thread's: only a drain of its own rank may lower it. A false
    // "holds none" would let a compound op on a zero skip itself while one of this thread's
    // ops on the element is still queued.
    if (thread_map.empty() && thr == get_thr_rank_fast())
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
                CHUNK_INVARIANT_FAILED("flush_deferred(thr={}): chunk {} left in the set with no op queued", thr,
                                       static_cast<const void*>(c));
            }
        }
    }

    return true;
}

// ── Cross-thread targeted flush ───────────────────────────────────────────
template <typename T, typename ChunkType, bool Enabled>
bool CompressionManagerOmpPolicy<T, ChunkType, Enabled>::flush_chunk_all_ranks(ChunkType* chunk,
                                                                               bool       self_only) noexcept
{
    auto* arr = ChunkType::get_deferred_ops();
    if (arr == nullptr)
    {
        return false;
    }

    using Policy = WriteCompressSyncPolicy<typename ChunkType::ProxyType>;

    const int self_rank = get_thr_rank_fast();
    const int max_thr   = CHUNK_PMP_RANK_BOUND;

    const uint32_t payload = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(chunk) >> 4U) & 0x00FFFFFFU;

    // ─── Cross-flush mutual exclusion is PER RANK ─────────────────────────
    // Preventing two flushers from draining the same arr[r][chunk] (double destroy())
    // needs exclusion only on the rank being drained: xflush_lock_rank[r] is taken INSIDE
    // the loop, so flushes of DIFFERENT ranks run in parallel. The flush announce and the
    // inflight drain (compressor exclusion) are independent atomics, outside that lock.

    // ─── Take the flush flag ──────────────────────────────────────────────
    // Blocks any future compressor start(); once inflight compressors are drained, none
    // is mutating any chunk.
    Policy::announce_flush(); // counter + heavy barrier, one Dekker verb
    increm_chunk_gate_flush<ChunkType>();

    {
        int spin = 0;
        // HARD compressors only (see flush_deferred).
        while (is_hard_compression_inflight())
        {
            if (++spin > 1000)
            {
                std::this_thread::yield();
                spin = 0;
            }
            else
            {
                CHUNK_CPU_PAUSE();
            }
        }
    }

    using FlushLog = ScopedRingLogManualEnd<EV_FLUSH_BEGIN, EV_FLUSH_END>;
    FlushLog log(payload);

    bool flushed_any = false;

    // Ops drained for THIS chunk across all ranks, for pending_retire.
    int64_t total_flushed_chunk = 0;

    // self_only: just our rank (stage 1 of wait_for_availability); otherwise every rank (stage 2).
    const int r_begin = self_only ? self_rank : 0;
    const int r_end   = self_only ? self_rank + 1 : max_thr;
    for (int r = r_begin; r < r_end; ++r)
    {
        auto& thread_map = arr[r];

        // Hold rank r's shard while touching arr[r] (find, drain, erase) and its
        // dangerous-list: no double drain of arr[r][chunk]; other ranks stay free.
        const RankFlushLock rank_scope_guard(r);

        // No writer wait: the enqueue path takes rank r's shard lock around its map
        // insert, so this lock IS the exclusion with rank r's writer. Do not make writers
        // back off on a GLOBAL flushing count instead: every enqueue would wait for any
        // flush anywhere, a convoy that collapses throughput at high thread counts.

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
        // Drain by ONE swap (see flush_deferred); do NOT loop to redrain (livelock).
        {
            std::remove_reference_t<decltype(*op_map)> batch;
            batch.swap(*op_map);
            // Replay context: gate bypassed, writes applied directly — safe
            // here (begin_flush + hard drain above), see in_flush_replay.
            const FlushReplayGuard replay_guard;
            for (auto& [num, op_ptr] : batch)
            {
                // Check-build guard: a null, misaligned or low op pointer, or one
                // whose vptr is null or garbage, is a block recycled while still
                // in the map; say so here rather than crash in write_body().
                if constexpr (build_config::any_check)
                {
                    DeferredOperationBase* raw  = op_ptr.get();
                    const auto             addr = reinterpret_cast<std::uintptr_t>(raw);
                    if (raw == nullptr || addr < 0x1000 || (addr & 0x7U) != 0)
                    {
                        CHUNK_INVARIANT_FAILED("[FLUSH_PROBE xrank] bad op_ptr in op_map: num={} op={} rank={}", num,
                                               static_cast<const void*>(raw), r);
                    }
                    const std::uintptr_t vptr = *reinterpret_cast<const std::uintptr_t*>(raw);
                    if (vptr < 0x1000)
                    {
                        CHUNK_INVARIANT_FAILED("[FLUSH_PROBE xrank] op has null/garbage vptr (recycled under map?): "
                                               "num={} op={} vptr={:#x} rank={}",
                                               num, static_cast<const void*>(raw), vptr, r);
                    }
                }

                if constexpr (chunk_config::ring_log_debug || chunk_config::ring_log_op_only)
                {
                    ring_log_op(get_tid(), EV_OP_FLUSH_ONE, static_cast<uint32_t>(num),
                                static_cast<const void*>(op_ptr.get()));
                }

                op_ptr->write_body();
                op_ptr->destroy();
                ++flushed_here;
            }
        }

        if (flushed_here > 0)
        {
            dec_pending_ops(r, flushed_here);
            total_flushed_chunk += flushed_here;
            flushed_any = true;
        }

        // ── Preserve concurrent re-enqueues ──────────────────────────────────
        // An op enqueued into *op_map AFTER the swap (a writer still active on rank r) is a
        // pending write NOT yet replayed. Never clear() the live map: that would drop it
        // unapplied and leave pending_ops wrong (dec counts only the batch). Erase the
        // chunk entry ONLY when nothing was re-enqueued; otherwise leave it (and the
        // dangerous-list membership) for the next flush.
        if (op_map->empty())
        {
            // ankerl's erase throws on a key it finds changed (a corrupted map); this
            // function is noexcept: say where instead of terminate.
            try
            {
                thread_map.erase(it_chunk);
                // Keep a later flush_deferred(r) from looking for an absent entry.
                deferred_chunk_per_thread[r].erase(chunk);
            }
            catch (const std::exception& e)
            {
                CHUNK_INVARIANT_FAILED("flush_chunk_all_ranks: erase failed: {} (rank {}, chunk {})", e.what(), r,
                                       static_cast<const void*>(chunk));
            }
        }
    }

    Policy::end_flush();
    log.end();

    // Per-chunk counter: subtract everything drained (as flush_deferred). Each op is drained
    // exactly once (per-rank shard exclusion), so no underflow without a lock here. Release
    // pairs with the reader's acquire load.
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
    if (arr == nullptr)
    {
        return;
    }

    // Serialise against a concurrent flush of OUR rank (the flushes' shard lock).
    // Owner-recursive; usually uncontended.
    const RankFlushLock xflush_guard(thr);

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

    // APPLY-then-erase, do not just discard: the direct write that follows supersedes the
    // queued value, but its write body reads v_old to feed the incremental compression
    // counters (mark_for_compression_if_needed_with_value). Discarded, values[num] stays at
    // its pre-op state, v_old is older than program order and the counters get a wrong
    // transition (silent corruption where pendings survive across parallel regions, e.g.
    // libgomp). Replaying restores program order: op applied, THEN our direct write. Under
    // FlushReplayGuard the body writes directly (no re-defer, no lock machinery).
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
        CHUNK_INVARIANT_FAILED("erase_own_rank_op: erase failed: {} (rank {}, chunk {}, offset {})", e.what(), thr,
                               static_cast<const void*>(chunk), num);
    }

    // Same bookkeeping as the flush erase path.
    dec_pending_ops(thr, 1);
    // Release: a reader that acquire-observes the decremented count also sees the direct
    // store that follows. No-op in counterless builds.
    chunk->pending_retire(1);

    // No more ops on our rank: drop the chunk from the rank tracking, as flush_deferred does.
    if (op_map->empty())
    {
        try // ankerl's erase throws on a key it finds changed; this function is noexcept
        {
            thread_map.erase(it_chunk);
            deferred_chunk_per_thread[thr].erase(chunk);
        }
        catch (const std::exception& e)
        {
            CHUNK_INVARIANT_FAILED("erase_own_rank_op: chunk erase failed: {} (rank {}, chunk {})", e.what(), thr,
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
    // A flush already claimed: undo our increment and return false; the caller
    // (make_block_compression_impl) skips compression. "Defer compression, not the flush."
    //
    // Only an early-out: the flush drains spin on stealing_inflight_, not hard_inflight_,
    // so the load that pairs with their announce_flush() is the one AFTER the writer wait
    // (end of this function).
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

    // The compressor's announce towards the WRITERS: the seq_cst increment above, then the
    // heavy barrier -- a fence, and under the asymmetric protocol the membarrier() standing
    // in for the writers' fence (see WriteCompressSyncPolicy::announce_write()). Keep it
    // AFTER the flush early-out: that early-out pairs with nothing, and on bulk-init phases
    // it is taken almost every time -- a membarrier() before it is an IPI storm for
    // compressions that never run.
    dekker::heavy_barrier();

    // Bounded wait for a "no writer" window (drains write_body() calls in progress). Under
    // heavy write load the window may never open, and an unbounded spin livelocks every
    // compressing thread in sched_yield. Past the yield budget, return false: the caller
    // skips compression WITHOUT touching P, the chunks stay queued and are retried on the
    // next pass, and the between-phase compress_chunk_to_compress region (no concurrent
    // writers, no start() barrier) drains them for good. Deferring loses no compression.
    int spin   = 0;
    int yields = 0;
// Spin budget before deferring. A large budget makes a compressing thread spin for a long
// time (up to seconds: periodic global stalls at high thread counts); since deferring loses
// nothing, a low budget is usually better. Override with -DCHUNK_COMPRESS_MAX_YIELDS=<n>
// (0 = give up on the first blocked check).
#ifndef CHUNK_COMPRESS_MAX_YIELDS
    #define CHUNK_COMPRESS_MAX_YIELDS 10000
#endif
    constexpr int kMaxYields = CHUNK_COMPRESS_MAX_YIELDS; // tune via -D

    // Wait HOLDING our hard increment, deliberately: the write-path gate consults
    // hard_inflight_, so a parked compressor steers new writers to the slow path and the
    // window opens (compressor priority). No parked livelock triangle: create_value and the
    // flush drains wait on stealing_inflight_ only, not incremented yet.
    // With the per-chunk marks the window is "no writer on a chunk this pass MARKED" (other
    // chunks' writers are neither waited for nor steered away); without them, "no writer
    // anywhere".
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
                // Give up: undo add_compression_active() (as the flush early-out does).
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
            CHUNK_CPU_PAUSE();
        }
    }

    // Writer gate established. Publish the transition Parked -> Stealing;
    // finish() undoes it. seq_cst pairs with the waiters' loads in
    // is_hard_compression_inflight().
    stealing_inflight_.fetch_add(1, std::memory_order_seq_cst);
    std::atomic_thread_fence(std::memory_order_seq_cst);

    // Re-check the flushers AFTER publishing stealing_inflight_: this is the Dekker pair
    // the flush drains rely on. They announce_flush() (flushing++, fence) then spin on
    // stealing_inflight_ -- NOT hard_inflight_ (waiting on a parked compressor is the
    // livelock triangle, see compression_manager.h). So the early check at the top, taken
    // while only parked, pairs with nothing: a flusher announced between it and the
    // fetch_add above saw stealing == 0 and is already replaying write bodies. Do not drop
    // this second look: we would scan and CAS-steal `values` under that replay (scan sees
    // the pre-replay constant run, the replayed store lands, the CAS parks the chunk back
    // on the constant) and the replayed write would be lost. Flush keeps priority: a
    // flusher never backs off, we do.
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

// Yield budget of the writer-side wait at the gate. With the per-chunk marks the wait is a
// parked compressor's window (microseconds) or the pass reaching this chunk; without them,
// the whole pass (hundreds of microseconds to a millisecond). Past the budget the write is
// deferred, as without the wait, so a bigger budget only buys fewer deferrals on unusually
// long passes. The budget in time is mostly the yields (~1 us each).
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
    // The same predicate as the gate: the decision is re-taken by
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
                break; // budget spent: the caller defers
            }
        }
        else
        {
            CHUNK_CPU_PAUSE();
        }
    }
    increm_chunk_gate_wait<ChunkType>(yields, mark_hit);
}

template <typename T, typename ChunkType, bool Enabled>
template <typename ProxyType>
void CompressionManagerOmpPolicy<T, ChunkType, Enabled>::finish()
{
    // Undo Parked -> Stealing FIRST: waiters on stealing_inflight_ (create_value, flush
    // drains) may proceed as soon as we can no longer steal; hard follows. finish() only
    // runs after a start() that returned true, so the counters are paired.
    stealing_inflight_.fetch_sub(1, std::memory_order_seq_cst);

    const int remaining = decrement_inflight();

    std::atomic_thread_fence(std::memory_order_seq_cst);

    // Another compressor still in its window (> 0), or a transient negative
    // count (see hard_inflight_): only the last one out goes on.
    if (remaining != 0)
    {
        return;
    }

    // The last compressor out: mark this rank's list safe again.
    mark_list_safe_for_thread(get_thr_rank_fast());

    if constexpr (chunk_config::ring_log_debug)
    {
        ring_log_end(get_tid(), EV_COMPRESS_FINISHED);
    }
}


} // namespace shrink_array