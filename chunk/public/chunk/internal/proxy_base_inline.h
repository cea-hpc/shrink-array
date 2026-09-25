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

/**
 * @file proxy_base_inline.h
 * @brief Out-of-line definitions of ProxyBase and ProxyBaseFastCompPolicy.
 *
 * Holds the parts of the proxy write engine that are too large to sit in the
 * class body, and the free helpers they use. Three concerns live here:
 *
 *  - **compression bookkeeping**: add_to_compression_list() and its async /
 *    sentinel variants, which flag a chunk as a compression candidate exactly
 *    once and push it onto the per-thread candidate list;
 *  - **the write engine**: write_with_optional_lock_impl(), the Dekker
 *    handshake against the compressor, its slow path, and the deferred-write
 *    enqueue path it falls back to;
 *  - **the arithmetic / bitwise compound assignments**, each in a sync flavour
 *    and an async one replayed later from a deferred op.
 *
 * The declarations, with their public contract, are in proxy_base.h; the blocks
 * below document what each definition actually does. Most of them exist in two
 * shapes selected by `chunk_config::use_fast_block_compression`: the OpenMP
 * engine with its handshake and deferrals, and a serial path that just runs the
 * write body.
 *
 * @see proxy_base.h, compression_manager.h, write_compress_sync_policy.h
 */

#include "chunk/internal/in_t.h"       // in_t<T>: the element snapshots
#include "chunk/internal/proxy_base.h" // the declarations these bodies belong to (self-contained, like the other *_inline.h)
#include "chunk/internal/chunk_size_policy.h" // ChunkElems — used HERE, not via includers
#include "chunk/internal/phase_guard.h"       // check-build guard of the phase protocol

namespace shrink_array
{

/**
 * @brief Appends one chunk to a compression-candidate list, whatever its tuple shape.
 *
 * The candidate lists are tuples of either 3 or 4 fields depending on the
 * build: the 4-field form also carries the chunk geometry, needed once the
 * geometry is dynamic. This helper picks the right emplace_back() so no call
 * site has to know which shape is in use.
 *
 * @param vec       Destination candidate list.
 * @param chunk     Chunk being queued (may be a null sentinel).
 * @param num_chunk Index of the chunk inside its array.
 * @param num_tab   Table number of the array the chunk belongs to.
 * @param size      Chunk geometry, dropped when the tuple has only 3 fields.
 */
template <typename Vec, typename ChunkPtr>
void push_to_compress_list(Vec& vec, ChunkPtr chunk, int num_chunk, int num_tab, ChunkElems size)
{
    using Tup = Vec::value_type; // C++20: typename not needed in an alias declaration
    if constexpr (std::tuple_size_v<Tup> == 4)
        vec.emplace_back(chunk, num_chunk, num_tab, size);
    else
        vec.emplace_back(chunk, num_chunk, num_tab);
}


// ============================================================================
// Helper function to call get_num_tab with the correct signature
// ============================================================================

/**
 * @brief Resolves the table number of a chunk from an async (replay) context.
 *
 * Proxy families differ in how they find that number: Proxy looks it up from
 * the array pointer, so it takes both the context and the pointer; a proxy that
 * already carries the number needs the context alone.
 *
 * @param ctx   Deferred-operation context being replayed.
 * @param proxy Proxy rebuilt from that context.
 * @param p     Array pointer recorded at enqueue time.
 * @return The table number of the chunk.
 *
 * @note The detection probes the SYNCHRONOUS `get_num_tab(p)`, not the async
 *       overload it then calls: having the pointer-keyed sync lookup is what
 *       identifies the family that also provides the two-argument async one.
 *       Only Proxy (via proxy_inline.h) reaches this helper today, so the
 *       one-argument branch is currently never instantiated — and no
 *       `get_num_tab_async(ctx)` overload exists for it to bind to.
 */
template <typename ProxyType>
constexpr int get_num_tab_async_helper(auto* ctx, const ProxyType& proxy, const void* p)
{
    if constexpr (requires { proxy.get_num_tab(p); })
        return proxy.get_num_tab_async(ctx, p);
    else
        return proxy.get_num_tab_async(ctx);
}

/**
 * @brief Arms the "already queued" flag pointer from a replay context.
 *
 * Async counterpart of set_ptr_is_inside_chunk_list_to_compress(): the
 * (num_tab, num_chunk) pair comes from the recorded context instead of the live
 * array, since the replay may run long after the write was issued.
 *
 * @param ctx Deferred-operation context carrying num_tab and num_chunk.
 */
template <typename Derived, typename T, typename ChunkType, bool Enabled>
void ProxyBaseFastCompPolicy<Derived, T, ChunkType, Enabled>::set_ptr_is_inside_chunk_list_to_compress_async(auto* ctx)
{
    const int tab       = ctx->num_tab;
    const int num_chunk = ctx->num_chunk;

    auto& arr = tab_is_inside_chunk_list_to_compress[tab];

    ptr_is_inside_chunk_list_to_compress = &arr[num_chunk];
}

/**
 * @brief Flags this chunk as a compression candidate on a caller-supplied list.
 *
 * The per-chunk flag is claimed with a CAS, so among all the threads writing
 * the chunk exactly one pushes it: the list can never hold a duplicate. Losing
 * the CAS is the normal outcome while the chunk waits in the list and is simply
 * counted (see the stats comments inline).
 *
 * Serial builds do nothing here — the whole body is OpenMP-only.
 *
 * @param vec_tuple Candidate list to push onto.
 * @pre `ptr_is_inside_chunk_list_to_compress` is armed, i.e. set_info_on_chunk()
 *      ran on this write path (checked under `build_config::any_check`).
 */
template <typename Derived, typename T, typename ChunkType>
void ProxyBase<Derived, T, ChunkType>::add_to_compression_list([[maybe_unused]] auto& vec_tuple) const
{
    if constexpr (build_config::use_openmp)
    {
        if constexpr (build_config::any_check)
        {
            if (ptr_is_inside_chunk_list_to_compress == nullptr) [[unlikely]]
            {
                INVARIANT_FAILED("add_to_compression_list(vec_tuple): chunk={} num={} "
                                 "num_chunk={} num_tab={} "
                                 "ptr_is_inside_chunk_list_to_compress is null "
                                 "(set_info_on_chunk not called on this write path?)",
                                 static_cast<const void*>(&chunk), num, num_chunk, num_tab);
            }
        }
        // Test, then test-and-set. The flag is 1 for 99.7% of the calls on
        // the perf bench (the chunk is already queued, and stays so until the
        // pass resets it), and a compare_exchange takes the line exclusively
        // whether it succeeds or not -- one char per chunk, 64 chunks per
        // line, neighbouring chunks written by different ranks: 5.4 G locked
        // RMWs and their line ping-pong per reference run for nothing. The
        // relaxed probe decides nothing by itself: the CAS below still does.
        char expected = 0;
        if (ptr_is_inside_chunk_list_to_compress->load(std::memory_order_relaxed) == 0 &&
            ptr_is_inside_chunk_list_to_compress->compare_exchange_strong(expected, 1, std::memory_order_acq_rel,
                                                                          std::memory_order_acquire))
        {
            if constexpr (build_config::is_debug)
            {
                if (num_chunk < 0 || num_tab < 0 || num_tab > 100)
                    std::abort();
            }
            // CHUNK_STATS: this write newly flagged the chunk as a candidate.
            if constexpr (chunk_config::stats)
                increm_chunk_mark_ok<ChunkType>();
            push_to_compress_list(vec_tuple, &chunk, num_chunk, num_tab, ChunkElems{this->chunk_size()});
        }
        else if constexpr (chunk_config::stats)
        {
            // CHUNK_STATS: the flag was ALREADY 1, so this write does not re-queue
            // the chunk. That is normal while it waits in the list, but if the flag
            // is never reset after the compression sweep the chunk can never be a
            // candidate again -- mark_refused then grows without bound while
            // mark_ok stays flat, which is exactly the signature to look for.
            increm_chunk_mark_refused<ChunkType>();
        }
    }
    else
    {
        //    add_null_sentinel_to_compression_list(vec_tuple);
    }
}

/**
 * @brief Flags this chunk as a compression candidate on this thread's own list.
 *
 * Same once-only CAS discipline as the vec_tuple overload, with two differences:
 * the destination list is looked up from the calling thread's rank, and a
 * relaxed load short-circuits the CAS when the chunk is already flagged — the
 * common case under sustained compression, where the CAS would cost ~15-20
 * cycles against ~1 for the load.
 *
 * Only compiled under `chunk_config::use_fast_block_compression`; the serial
 * fallback below it uses plain accesses instead of atomics.
 *
 * @pre `ptr_is_inside_chunk_list_to_compress` is armed (checked under
 *      `build_config::any_check`).
 */
template <typename Derived, typename T, typename ChunkType>
void ProxyBase<Derived, T, ChunkType>::add_to_compression_list() const
{
    if constexpr (chunk_config::use_fast_block_compression)
    {
        if constexpr (build_config::any_check)
        {
            if (ptr_is_inside_chunk_list_to_compress == nullptr) [[unlikely]]
            {
                INVARIANT_FAILED("add_to_compression_list (fast-block): chunk={} num={} "
                                 "num_chunk={} num_tab={} "
                                 "ptr_is_inside_chunk_list_to_compress is null "
                                 "(set_info_on_chunk not called on this write path?)",
                                 static_cast<const void*>(&chunk), num, num_chunk, num_tab);
            }
        }
        // CHUNK_STATS: the flag gates queueing. mark_refused counts the writes that
        // find it ALREADY at 1: normal while the chunk waits in the list, but if the
        // flag is never reset after the compression sweep the chunk can never be
        // queued again -- refused then grows without bound while queued stays flat.
        if (ptr_is_inside_chunk_list_to_compress->load(std::memory_order_relaxed) != 0)
        {
            if constexpr (chunk_config::stats)
                increm_chunk_mark_refused<ChunkType>();
        }
        else
        {
            char expected = 0;
            if (ptr_is_inside_chunk_list_to_compress->compare_exchange_strong(expected, 1, std::memory_order_acq_rel,
                                                                              std::memory_order_acquire))
            {
                if constexpr (chunk_config::stats)
                    increm_chunk_mark_ok<ChunkType>();
                const int thr = get_thr_rank_fast();
                auto&     vec = derived().get_vec_tuple_to_compress_potentially_thr(thr);

                if constexpr (build_config::is_debug)
                {
                    if (num_chunk < 0 || num_tab < 0 || num_tab > 100)
                        std::abort();
                }

                push_to_compress_list(vec, &chunk, num_chunk, num_tab, ChunkElems{this->chunk_size()});
            }
        }
    }
    else
    {
        if constexpr (build_config::not_use_openmp)
        {
            if (!(*ptr_is_inside_chunk_list_to_compress))
            {
                *ptr_is_inside_chunk_list_to_compress = 1;
                auto& vec                             = derived().get_vec_tuple_to_compress_potentially();
                push_to_compress_list(vec, &chunk, num_chunk, num_tab, ChunkElems{this->chunk_size()});
            }
        }
    }
}

/**
 * @brief Rebuilds a fully armed proxy from a recorded deferred-write context.
 *
 * A deferred op cannot keep the original proxy alive, so the replay recreates
 * one and restores everything the write body needs: chunk index, table number,
 * second-value info, and the compression flag pointer. Which of those steps
 * apply is decided per proxy type with `requires`, since Proxy, Proxy2 and
 * Proxy2A do not all carry the same state.
 *
 * The frozen geometry is restored too: chunk_size() must not read it back from
 * `ptr_on_array`, which may point at an array destroyed since the enqueue.
 *
 * @tparam ProxyType Proxy type to rebuild.
 * @tparam Ctx       Context type recorded at enqueue time.
 * @param ctx Context to rebuild from.
 * @return The rebuilt proxy, ready for the replayed write body.
 */
template <typename Derived, typename T, typename ChunkType, bool Enabled>
template <typename ProxyType, typename Ctx>
ProxyType ProxyBaseFastCompPolicy<Derived, T, ChunkType, Enabled>::make_proxy_with_info(Ctx* ctx)
{
    auto&     chunk = ctx->chunk;
    const int num   = ctx->num;

    std::atomic<T*>& atomic_values = chunk.values;

    ProxyType proxy(atomic_values, num, chunk);

    proxy.set_num_chunk(ctx->num_chunk);

    // fw14: carry the enqueue-time geometry onto the replay proxy so its
    // chunk_size() does not dereference a possibly-dead ptr_on_array.
    proxy.set_chunk_size_frozen(ctx->chunk_size_frozen, ctx->alloc_key_frozen);

    if constexpr (requires { proxy.BaseSpec::set_info_second_value_ind_from_key(ctx->ptr_on_array); })
    {
        proxy.BaseSpec::set_info_second_value_ind_from_key(ctx->ptr_on_array);
    }

    if constexpr (requires(ProxyType p) { p.set_num_tab(0); })
    {
        proxy.set_num_tab(ctx->num_tab);
    }

    if constexpr (requires { proxy.set_ptr_second_value(); })
    {
        proxy.set_ptr_second_value();
    }

    proxy.set_num_chunk(ctx->num_chunk);

    proxy.Base::set_ptr_is_inside_chunk_list_to_compress_async(ctx);

    return proxy;
}

/**
 * @brief Flags a chunk as a compression candidate from a replayed deferred op.
 *
 * Async counterpart of add_to_compression_list(): everything it needs — chunk,
 * offset, chunk index, array pointer — is read from the context rather than
 * from the proxy's own state, because the write is being applied long after it
 * was issued. Same once-only CAS discipline, same relaxed-load fast path.
 *
 * Chunk2 keeps its candidates in a context-dependent list, hence the
 * `IsChunk2` branch on which per-thread list to push onto.
 *
 * @param self The proxy, deduced (explicit object parameter).
 * @param ctx  Deferred-operation context being replayed.
 */
// Deducing this: `self` replaces derived(). The class still has its Derived
// parameter, so Derived::tab_... and the CRTPSupport base are unchanged; only
// the derived() calls become self.
template <typename Derived, typename T, typename ChunkType, bool Enabled>
void ProxyBaseFastCompPolicy<Derived, T, ChunkType, Enabled>::add_to_compression_list_async(this auto const& self,
                                                                                            auto*            ctx)
{
    auto&     chunk = ctx->chunk;
    const int num   = ctx->num;

    using ProxyType = std::remove_cvref_t<decltype(self)>;

    // Rebuild a fresh proxy from current chunk state
    std::atomic<T*>& atomic_values = chunk.values;
    ProxyType        proxy(atomic_values, num, chunk);

    // Compute num_tab for this array/chunk
    const int num_tab = get_num_tab_async_helper(ctx, proxy, ctx->ptr_on_array);

    // Get the flag for "already in compression list"
    auto& flag = Derived::tab_is_inside_chunk_list_to_compress[num_tab][ctx->num_chunk];

    // Fast path: cheap relaxed load before the expensive CAS.
    // If the chunk is already marked (common case during active compression),
    // we avoid the acq_rel CAS entirely — ~1 cycle vs ~15-20 cycles.
    if (flag.load(std::memory_order_relaxed) != 0)
        return;

    char expected = 0;
    if (flag.compare_exchange_strong(expected, 1, std::memory_order_acq_rel, std::memory_order_acquire))
    {
        const int thr = get_thr_rank_fast();

        auto& vec = [&]() -> auto&
        {
            if constexpr (IsChunk2<chunk_template_of<ChunkType>::template type, T>)
                return self.get_vec_tuple_to_compress_potentially_async_thr(ctx, thr);
            else
                return self.get_vec_tuple_to_compress_potentially_thr(thr);
        }();

        push_to_compress_list(vec, &ctx->chunk, ctx->num_chunk, num_tab, self.chunk_size());
    }
}

/**
 * @brief Runs a write body against the compressor, or reports that it must be deferred.
 *
 * The engine every proxy write funnels through. It tries to take a fast path
 * where the write happens in place, and returns true when it could not: the
 * caller must then re-issue the write as a deferred operation (see
 * enqueue_deferred_write() / enqueue_async_arithmetic()).
 *
 * Under `chunk_config::use_fast_block_compression` it goes, in order:
 *  -# **flush replay** — already inside a drain, apply directly and return;
 *  -# **cross-rank drain and own-replay exclusion** — materialise foreign
 *     pending ops on this offset, and hold our own rank lock so no concurrent
 *     replay of our queued ops can run behind our back;
 *  -# **Dekker handshake** with the compressor (announce, fence, re-check),
 *     then the write body itself, then end_write; after that, either the
 *     compression trigger, or -- if the trigger had flagged this rank's list
 *     as dangerous -- a compression pass, driven here, after the write;
 *  -# **slow path** whenever a compressor is in flight: return true so the
 *     caller defers the write; a dangerous rank drives its compression pass
 *     on the way out, the others just leave.
 *
 * Without fast block compression the body simply runs, plus a Chunk2A-only
 * sequential compression trigger (see the inline note for why it is restricted
 * to Chunk2A and skipped during a flush replay).
 *
 * Each step's correctness argument — the memory-order proof of the handshake,
 * the WAW guard, the regressions that motivated the guards — is kept in the
 * inline comments next to the code it justifies.
 *
 * @tparam WriteCallable Callable performing the actual store.
 * @tparam LBD_INFO      Callable arming the per-chunk bookkeeping beforehand.
 * @param write_body Write to perform.
 * @param lbd_info   Ran first on the fast path; typically set_info_on_chunk().
 * @return True if the write did NOT happen and must be deferred by the caller;
 *         false if it was applied.
 */
template <typename Derived, typename T, typename ChunkType>
template <typename WriteCallable, typename LBD_INFO>
[[nodiscard]] bool
ProxyBase<Derived, T, ChunkType>::write_with_optional_lock_impl(WriteCallable&&           write_body,
                                                                [[maybe_unused]] LBD_INFO lbd_info) noexcept
{
    if constexpr (chunk_config::use_fast_block_compression)
    {
        lbd_info(); // common initialization required for fast-path

        // ─── Flush replay: apply DIRECTLY, never re-defer ────────────────────
        // We are inside a flush drain (begin_flush held, hard compressors
        // drained): no compressor can steal values, so the Dekker handshake
        // and the compression-active gate are both unnecessary AND harmful —
        // consulting the gate here made replayed ops re-enqueue themselves
        // forever under sustained compression activity.
        if (compression_mgr.in_flush_replay) [[unlikely]]
        {
            std::forward<WriteCallable>(write_body)();
            return false;
        }

        // ─── Cross-rank direct-write guard (CHUNK_SYNC_ON_READ) ─────────────
        // A pending op enqueued by ANOTHER rank at this offset would replay
        // AFTER our write and clobber it (parallel regions remap thread->rank,
        // so the same element can legally be written by different ranks in
        // successive regions; only a barrier orders them). Own-rank pending is
        // safe (same-offset own op erased by P2 below; other offsets replay
        // elsewhere) and keeps the bulk-init fast path: one acquire load when
        // nothing is pending, one extra load + TLS read when own ops are. The
        // drain runs HERE, before begin_write, so it is never re-entrant. This
        // also covers the deferred-ENQUEUE path decided below: two same-offset
        // ops on different ranks replay in arbitrary order, so the foreign one
        // must be materialised before we enqueue ours.
        //
        // ─── Own-pending replay exclusion (fw5) ─────────────────────────────
        // The gnu battery bisection (fw4a) proved the residual corruption needs
        // the OFF-mode structural guarantee: no replay may be IN FLIGHT on a
        // chunk while a thread direct-writes it. Foreign ops are drained above;
        // OUR own pending ops (other offsets of this chunk) can still be
        // replayed CONCURRENTLY by another thread's gate/stage-2 walk — and a
        // replay bypasses the Dekker (in_flush_replay), so the two writers are
        // mutually invisible. Every replayer of rank r's ops holds
        // xflush_lock[r]; holding OUR OWN rank lock for the duration of the
        // write therefore excludes any concurrent replay of our ops WITHOUT
        // draining anything: the bulk-init deferral regime keeps its ops
        // queued (owner-recursive lock, uncontended there — the enqueue path
        // below takes the same lock anyway).
        [[maybe_unused]] int own_replay_excl_rank = -1;
        if constexpr (build_config::use_openmp)
        {
            if (chunk.pending_any()) [[unlikely]]
            {
                const int thr_entry = get_thr_rank_fast();
                if (chunk.foreign_pending(thr_entry, num))
                    chunk.reader_sync(num);
                if (chunk.pending_any()) // own ops remain queued: exclude their replay
                {
                    CompressionManagerType::enqueue_lock(thr_entry);
                    own_replay_excl_rank = thr_entry;
                }
            }
        }
        struct OwnReplayExclGuard
        {
            int r;
            ~OwnReplayExclGuard()
            {
                if (r >= 0)
                    CompressionManagerType::enqueue_unlock(r);
            }
        } own_replay_excl_guard{own_replay_excl_rank};

        // ─── Check #1: pure optimisation, not correctness ───────────────────
        // Relaxed probe before touching begin_write's cache line. If a
        // compression is clearly already in flight, skip straight to the slow
        // path without contending the per_thread_writing[thr] cache line
        // that the compressor is busy-waiting on.
        //
        // A stale read here is safe: check #2 below, after announce_write,
        // is the one the handshake's ordering covers.
        //
        // The gate is a parked compressor (its writer window is being
        // opened, microseconds) or a compressor's pass having THIS chunk
        // marked (ChunkBase::steal_mark_); without the marks, any hard
        // compression in flight. It used to be "a slow-path driver
        // is running, or a compressor, or MY list is dangerous": the first
        // sent every writer down the deferred path for the whole of someone
        // else's flush-and-try, the last sent the dangerous rank's own write
        // there -- to be enqueued, then replayed by the flush_deferred() its
        // perform_compression() ran an instant later, under a flush announce
        // that made every other rank's start() yield. That was the circular
        // regime: 390 000 deferred writes and 17 000 flushes per second at 8
        // threads, 98% of the start() calls yielding, nothing compressed
        // until the between-phase region. Now a dangerous rank WRITES, under
        // the same handshake as everyone, and drives its compression pass
        // after end_write (below); only a compressor in flight defers a write
        // -- and even then only after a bounded wait for it: no flag is held
        // here, so waiting shuts no window, and a window is a few hundred
        // microseconds against an enqueue and a replay per write (see
        // wait_out_compressor).
        if (compression_mgr.must_defer_write(chunk)) [[unlikely]]
        {
            compression_mgr.wait_out_compressor(chunk);
        }

        // Test knob (build_config::force_defer_every): one write in N takes
        // the deferred exit as if a compressor held the chunk -- the even
        // hits fail check #1, the odd ones check #2. Compiled out at 0.
        bool force_defer_1 = false;
        bool force_defer_2 = false;
        if constexpr (build_config::force_defer_every > 0)
        {
            thread_local unsigned force_defer_count = 0;
            thread_local bool     force_defer_odd   = false;
            if (++force_defer_count == build_config::force_defer_every) [[unlikely]]
            {
                force_defer_count                                 = 0;
                force_defer_odd                                   = !force_defer_odd;
                (force_defer_odd ? force_defer_2 : force_defer_1) = true;
            }
        }

        if (!force_defer_1 && !compression_mgr.must_defer_write(chunk)) [[likely]]
        {
            const int thr = get_thr_rank_fast();

            // ─── Dekker handshake ───────────────────────────────────────────
            // (1)+(2) announce_write = OUR flag, ordered before the loads
            // below: a plain store and a compiler barrier under the
            // asymmetric handshake (the compressor's membarrier() does the
            // rest), a seq_cst store + fence under the symmetric one. One
            // policy verb, so no call site can forget the ordering; both
            // arguments live with it (write_compress_sync_policy.h).
            WriteCompressSyncPolicy<Derived>::announce_write(thr, &chunk);

            // ─── Invariant: our flag must be observably set ─────────────────
            if constexpr (build_config::any_check)
            {
                using Policy = WriteCompressSyncPolicy<Derived>;
                if (!Policy::is_thread_writing_seq(thr))
                {
                    INVARIANT_FAILED("Dekker: per_thread_writing[{}] is FALSE right after begin_write", thr);
                }
            }

            // (3) check THEIR flag (inflight), relaxed. If this load does not
            //     see inflight>0 then the compressor's scan of
            //     per_thread_writing[thr] sees our flag and spins -- see
            //     announce_write() for why, in either handshake shape.
            if (!force_defer_2 && !compression_mgr.must_defer_write(chunk)) [[likely]]
            {
                // Dekker established. chunk.values cannot be mutated by any
                // compressor until we call end_write. Concurrent writers on the
                // same chunk are handled by the is_cst_val re-checks inside
                // handle_{bool,no_bool}_assignment — no extra guard needed here.

                const ScopedRingLog<EV_BEGIN_FAST_PATH, EV_END_FAST_PATH> fast_log;

                // ─── Pre-write invariant: under Dekker, chunk should not be in cst_val
                //     unless our writer's caller will explicitly handle that. The
                //     chunk-specific path (perform_core_assignment / handle_*_assignment)
                //     does the re-checks. We only assert that the flag still holds.
                if constexpr (build_config::any_check)
                {
                    using Policy = WriteCompressSyncPolicy<Derived>;
                    if (!Policy::is_thread_writing(thr))
                    {
                        INVARIANT_FAILED("Dekker: per_thread_writing[{}] cleared between begin_write and write_body",
                                         thr);
                    }
                }

                // ─── WAW guard (CHUNK_SYNC_ON_READ, P2) ──────────────────────
                // Writes no longer flush (P1), so an OLDER deferred op for THIS
                // offset may still sit on our own rank. Discard it before the
                // direct store below, else a later replay would clobber the fresh
                // value. Only fires when this chunk actually has pendings.
                // pending_any() is constant-false in counterless builds, so the
                // erase call folds away there — same effect as the old #if, but
                // selected by the policy. use_openmp in the constexpr condition
                // keeps the branch un-instantiated in serial builds (the erase
                // body names OpenMP-only machinery).
                if constexpr (build_config::use_openmp)
                {
                    if (chunk.pending_any()) [[unlikely]]
                        CompressionManagerType::erase_own_rank_op(thr, &chunk, num);
                }

                std::forward<WriteCallable>(write_body)();

                // ─── Post-write invariant: flag must still be ours ─────────
                if constexpr (build_config::any_check)
                {
                    using Policy = WriteCompressSyncPolicy<Derived>;
                    if (!Policy::is_thread_writing(thr))
                    {
                        INVARIANT_FAILED(
                            "Dekker: per_thread_writing[{}] cleared during write_body (someone else cleared our flag)",
                            thr);
                    }
                }

                WriteCompressSyncPolicy<Derived>::end_write(thr);

                // ─── Compression pass, AFTER the write ───────────────────────
                // The trigger flags the rank's list once it crosses the
                // threshold; the next write of that rank -- this one -- drives
                // perform_compression() once its own store is done and its
                // writer flag released. The pass starts with the rank's
                // flush_deferred(), which announces nothing when the rank has
                // no deferred op (the usual case now), then start(), whose
                // parked hard_inflight_ is what steers the other writers away
                // for the "no writer" window. One driver per threshold
                // crossing, not one per write.
                if (compression_mgr.is_dangerous_list(thr)) [[unlikely]]
                {
                    compression_mgr.run_slow_path([&] { derived().perform_compression(); });
                }
                else
                {
                    // ─── P3 backlog drain, fast-path side ───────────────────
                    // The ops this rank deferred while a compressor was in
                    // flight are drained once the backlog crosses the
                    // threshold -- from here, a direct write with no
                    // compressor in the way, rather than from the deferred
                    // path, where the drain waited the compressor out (see
                    // to_be_queued_for_deferred_processing). One thread-local
                    // load when the rank holds nothing, the common case.
                    if constexpr (build_config::use_openmp)
                    {
                        if (CompressionManagerType::thread_may_have_ops_own()) [[unlikely]]
                        {
                            if (CompressionManagerType::pending_ops_rank(thr) > build_config::deferred_drain_threshold)
                            {
                                (void)CompressionManagerType::flush_deferred(thr);
                            }
                        }
                    }
                    derived().trigger_compression_if_needed(thr);
                }
                return false;
            }

            const ScopedRingLog<EV_BEGIN_SLOW_PATH, EV_END_SLOW_PATH> slow_log;

            // Check #2 failed: a compressor slipped in between check #1 and
            // begin_write. Release our flag, then defer the write.
            WriteCompressSyncPolicy<Derived>::end_write(thr);

            // ─── Invariant after end_write on slow-path-via-check#2 ────────
            if constexpr (build_config::any_check)
            {
                using Policy = WriteCompressSyncPolicy<Derived>;
                if (Policy::is_thread_writing(thr))
                {
                    INVARIANT_FAILED("Dekker: per_thread_writing[{}] still TRUE after end_write (slow path)", thr);
                }
            }

            drive_compression_if_dangerous(thr);
            return true;
        }

        // Check #1 failed: a compressor is in flight; skip the Dekker handshake
        // entirely. No end_write here — begin_write was never called in this
        // branch, so the flag is already false (writing it would dirty a
        // contended cache line for nothing).
        if constexpr (build_config::any_check)
        {
            using Policy  = WriteCompressSyncPolicy<Derived>;
            const int thr = get_thr_rank_fast();
            if (Policy::is_thread_writing(thr))
            {
                INVARIANT_FAILED(
                    "Dekker: per_thread_writing[{}] unexpectedly TRUE on check#1 fail (begin_write not called)", thr);
            }
        }

        drive_compression_if_dangerous(get_thr_rank_fast());
        return true;
    }
    else
    {
        // Non-OMP path (use_fast_block_compression is false without OpenMP):
        // execute the write directly. In OMP builds the Dekker fast-path above
        // calls trigger_compression_if_needed after each write; that path is
        // compiled out here, so nothing would ever fire block compression in
        // sequential builds. Drive it directly instead: perform_compression()
        // self-checks the threshold (size > szm/2) and no-ops below it, so this
        // stays cheap. Gated on the proxy actually having perform_compression
        // (Chunk2A/Chunk2/Chunk do; simple proxies may not).
        std::forward<WriteCallable>(write_body)();

        // Restricted to Chunk2A: only its compression algorithm is identical in
        // OMP and non-OMP, so driving it sequentially is correct. Chunk/Chunk2
        // have distinct non-OMP paths and must NOT be triggered here.
        if constexpr (build_config::not_use_openmp && std::is_same_v<ChunkType, Chunk2A<T>>)
        {
            // Skip during rechunk_refill / flush replay (in_flush_replay): the refill
            // materialises a chunk here and RAW-fills it right after; compressing it in
            // between frees its freshly-allocated block (block_clear -> free_elt_pow2),
            // leaving the raw-fill to write a freed block -> slab free-list corruption
            // (TCAllocator sl_contains assert) / use-after-free. The OMP write path
            // already early-returns on in_flush_replay (see above); mirror that here.
            if (!compression_mgr.in_flush_replay)
                derived().perform_compression();
        }
        return false;
    }
}

/**
 * @brief Chunk geometry seen by this proxy, in elements.
 *
 * Reads the live geometry from the array on a direct write, but returns the
 * value frozen at enqueue time on a deferred replay, where `ptr_on_array` may
 * already be dangling. Geometry is immutable after construction, so the two
 * sources always agree.
 *
 * @return The number of elements per chunk.
 */
template <typename Derived, typename T, typename ChunkType>
inline unsigned int ProxyBase<Derived, T, ChunkType>::chunk_size() const noexcept
{
    // fw14: on the deferred-replay path chunk_size_frozen_ is armed (non-zero)
    // from the enqueue-time geometry, so we must NOT dereference ptr_on_array —
    // it may point at an already-destroyed ArrayChunk (use-after-free, SIGSEGV
    // under std-alloc + DynamicChunkSize). On the direct write path the array is
    // alive and the value was never frozen (0), so we read it live as before.
    // Geometry is immutable post-construction, so both paths agree.
    if (chunk_size_frozen_ != 0U) [[unlikely]]
        return chunk_size_frozen_;

    using ArrayType = ArrayChunkFor_t<ChunkType>;
    return static_cast<const ArrayType*>(ptr_on_array)->chunk_size();
}

template <typename Derived, typename T, typename ChunkType>
inline unsigned int ProxyBase<Derived, T, ChunkType>::alloc_key() const noexcept
{
    if (chunk_size_frozen_ != 0u) [[unlikely]]
        return alloc_key_frozen_;
    using ArrayType = ArrayChunkFor_t<ChunkType>;
    return static_cast<const ArrayType*>(ptr_on_array)->alloc_key();
}

/**
 * @brief Queues a write for later replay on the calling thread's rank.
 *
 * Marks the chunk dirty on its array, allocates the operation from the pool,
 * pushes it onto the rank's queue and records the enqueue with the compression
 * manager (which is what makes `pending_any()` / `foreign_pending()` see it).
 * The queue is drained by the phase transitions of manage_ompt.h.
 *
 * @tparam CtxType    Context type holding the write location and operands.
 * @tparam LambdaBody Callable replaying the write from that context.
 * @param ctx  Context, already pool-allocated by the caller.
 * @param body Write body to replay.
 */
template <typename Derived, typename T, typename ChunkType>
template <typename CtxType, typename LambdaBody>
void ProxyBase<Derived, T, ChunkType>::enqueue_deferred_write(CtxType* ctx, LambdaBody&& body) const
{
    using ArrayType = ArrayChunkFor_t<ChunkType>;
    auto* arr       = static_cast<ArrayType*>(const_cast<void*>(ptr_on_array));
    arr->mark_chunk_dirty(ctx->num_chunk);
    increm_chunk_gate_deferred<ChunkType>();

    // The moment the protocol matters: this write is about to be parked in a
    // per-thread queue that only the phase end drains. See phase_guard.h.
    if constexpr (phase_guard::active)
    {
        if (CHUNK_PMP_IN_PARALLEL_REGION && !phase_guard::armed()) [[unlikely]]
            INVARIANT_FAILED("[PHASE PROTOCOL] deferred write from a parallel region that did not call "
                             "begin_parallel_phase() (or whose previous phase was never finalized): without "
                             "OMPT every region must be bracketed by begin_parallel_phase() on each thread "
                             "and finalize_parallel_phase() after the region, or this write replays after "
                             "a later region's writes. chunk={} num={} rank={}",
                             static_cast<const void*>(&chunk), ctx->num, get_thr_rank_fast());
    }

    const int thr_rank = get_thr_rank_fast();

    using OpType = CompressionManagerType::template DeferredOperation<CtxType, std::decay_t<LambdaBody>>;
    auto* op_ptr = pool_new_op<OpType>(ctx, std::forward<LambdaBody>(body), thr_rank);

    chunk.to_be_queued_for_deferred_processing(thr_rank, ctx->num, op_ptr);
    CompressionManagerType::record_enqueue_deferred(thr_rank, ctx->chunk);
}

/**
 * @brief Defers a compound arithmetic operation that the fast path refused.
 *
 * Captures the location, the operand and the operation into a pooled context —
 * a lighter one for simple proxies, the full chunk-aware one otherwise — wraps
 * @p async_body in a lambda that rebuilds a proxy with make_proxy_with_info() at
 * replay time, and hands the pair to enqueue_deferred_write().
 *
 * @tparam U         Operand type.
 * @tparam Op        Binary operation type.
 * @tparam AsyncBody Callable `(ctx, proxy)` applying the operation on replay.
 * @param v          Operand, stored converted to T.
 * @param operation  Operation to replay.
 * @param async_body Replay body.
 */
template <typename Derived, typename T, typename ChunkType>
template <typename U, typename Op, typename AsyncBody>
void ProxyBase<Derived, T, ChunkType>::enqueue_async_arithmetic(const U& v, Op operation, AsyncBody&& async_body)
{
    using ProxyType = std::remove_reference_t<decltype(derived())>;

    using CtxType = std::conditional_t<is_simple_proxy<ProxyType>::value, ArithmeticOperationProxyContext<T, Op>,
                                       ArithmeticOperationContext<T, Op, ChunkType, ProxyType>>;

    // Create a minimal context (location + operand + operation)
    auto* ctx = pool_new_ctx<CtxType>(derived(), static_cast<T>(v), operation, ptr_on_array);

    // Async wrapper: reconstruct a fresh ProxyType at execution time
    auto op_async = [ctx, body = std::forward<AsyncBody>(async_body)]() mutable
    {
        using ProxyType = std::remove_cvref_t<decltype(derived())>;

        auto proxy = ProxyBase::template make_proxy_with_info<ProxyType>(ctx);

        body(ctx, proxy);
    };

    enqueue_deferred_write(ctx, std::move(op_async));
}
/**
 * @brief Tests a proxy's second-constant state, false for proxies that have none.
 *
 * Lets the shared invariant checks below name is_cst_val2() unconditionally:
 * only Chunk2 / Chunk2A carry a second constant value, and the plain Chunk
 * proxy simply answers false.
 *
 * @param p Proxy to test.
 * @return True if @p p exposes is_cst_val2() and it holds.
 */
template <typename P>
constexpr bool check_is_cst_val2([[maybe_unused]] const P& p)
{
    if constexpr (requires { p.is_cst_val2(); })
    {
        return p.is_cst_val2();
    }
    else
    {
        return false;
    }
}

/**
 * @brief Applies a compound arithmetic operation, deferring it if the fast path refuses.
 *
 * Computes in `std::common_type_t<T, U>` — the STL promotion rule — then
 * truncates back to T, so `float += double` keeps the double-precision
 * intermediate. The same computation is written twice: once as the synchronous
 * body handed to write_with_optional_lock_info(), once as the replay body
 * handed to enqueue_async_arithmetic() when that returns "deferred".
 *
 * Both bodies re-mark the chunk for compression from the before/after value
 * pair, and, under `build_config::any_check`, assert the invariants that a
 * broken handshake would violate: no cst_val state visible while writing, and a
 * `chunk.values` pointer unchanged across the store (a change means a
 * compressor swapped the buffer and the store was lost).
 *
 * @tparam U  Operand type.
 * @tparam Op Binary operation, invocable as `T(T, U)`.
 * @param operation Operation to apply.
 * @param v         Operand.
 * @param v_old     Value before the operation, used for the compression marking.
 */
template <typename Derived, typename T, typename ChunkType>
template <typename U, typename Op>
    requires std::is_invocable_r_v<T, Op, T, U>
void ProxyBase<Derived, T, ChunkType>::apply_atomic_op(Op operation, const U& v, in_t<T> v_old)
{
    if constexpr (chunk_config::stats)
        increm_chunk_write<ChunkType>(v);

    // Promotion rule identical to STL
    using CalcType = std::common_type_t<T, base_t<U>>;

    auto compute_operation = [this, operation, &v, v_old] // v: the call's parameter, alive for the call
    {
        std::ignore = v_old;

        derived().execute_arithmetic_compound_operation(
            operation, static_cast<T>(v),
            [this, &v, operation](auto) // operation functor is used instead of the passed one to ensure consistency
            {
                const in_t<T> current = load_value();
                // Perform high-precision calculation
                const CalcType res = operation(static_cast<CalcType>(current), static_cast<CalcType>(v));

                // ─── Pre-store check: cst_val should not be visible here under Dekker ───
                if constexpr (build_config::any_check)
                {
                    if (derived().is_cst_val1())
                    {
                        INVARIANT_FAILED(
                            "apply_atomic_op SYNC: chunk={} num={} is_cst_val1 BEFORE store_value (Dekker breach?)",
                            static_cast<const void*>(&chunk), num);
                    }

                    if (check_is_cst_val2(derived()))
                    {
                        INVARIANT_FAILED(
                            "apply_atomic_op SYNC: chunk={} num={} is_cst_val2 BEFORE store_value (Dekker breach?)",
                            static_cast<const void*>(&chunk), num);
                    }
                }

                // Snapshot the values pointer before the store
                T* values_before = chunk.values_load();

                store_value(static_cast<T>(res));

                // ─── Post-store check: values pointer should not have been CASed away ───
                if constexpr (build_config::any_check)
                {
                    T* values_after = chunk.values_load();
                    if (values_after != values_before)
                    {
                        INVARIANT_FAILED("apply_atomic_op SYNC: chunk={} num={} values pointer CHANGED across store: "
                                         "before={} after={} ; the store may have been LOST",
                                         static_cast<const void*>(&chunk), num, static_cast<const void*>(values_before),
                                         static_cast<const void*>(values_after));
                    }
                }
            });

        if constexpr (chunk_config::use_fast_block_compression)
        {
            derived().mark_for_compression_if_needed_with_value(v_old, read_values());
        }
    };

    if constexpr (chunk_config::use_fast_block_compression)
    {
        if (const bool need_async = write_with_optional_lock_info(compute_operation); need_async)
        {
            enqueue_async_arithmetic(
                v, operation,
                [v_old](auto* ctx, auto& proxy)
                {
                    proxy.execute_arithmetic_compound_operation_async(
                        ctx, ctx->operation, ctx->operand,
                        [ctx, &proxy](auto op_func)
                        {
                            // ─── Pre-read invariant: chunk shouldn't be in a cst_val state
                            //     by the time we reach the lambda body (the
                            //     execute_arithmetic_compound_operation_async wrapper has
                            //     called create_value upstream if needed). ───
                            if constexpr (build_config::any_check)
                            {
                                if (proxy.is_cst_val1())
                                {
                                    INVARIANT_FAILED("apply_atomic_op ASYNC lambda: chunk={} num={} is_cst_val1 "
                                                     "before read+store (decompression skipped?)",
                                                     static_cast<const void*>(&ctx->chunk), ctx->num);
                                }
                                if (check_is_cst_val2(proxy))
                                {
                                    INVARIANT_FAILED("apply_atomic_op ASYNC lambda: chunk={} num={} is_cst_val2 "
                                                     "before read+store (decompression skipped?)",
                                                     static_cast<const void*>(&ctx->chunk), ctx->num);
                                }
                            }

                            const in_t<T> current = proxy.read_values_async(ctx);

                            const CalcType res =
                                op_func(static_cast<CalcType>(current), static_cast<CalcType>(ctx->operand));

                            // Snapshot before store
                            T* values_before = ctx->chunk.values_load();

                            proxy.store_value_async(ctx, static_cast<T>(res));

                            // ─── Post-store check: detect lost write due to concurrent compression ───
                            if constexpr (build_config::any_check)
                            {
                                T* values_after = ctx->chunk.values_load();
                                if (values_after != values_before)
                                {
                                    INVARIANT_FAILED("apply_atomic_op ASYNC lambda: chunk={} num={} values pointer "
                                                     "CHANGED across store_value_async: before={} after={} ; "
                                                     "the store may have landed in a stale buffer (LOST)",
                                                     static_cast<const void*>(&ctx->chunk), ctx->num,
                                                     static_cast<const void*>(values_before),
                                                     static_cast<const void*>(values_after));
                                }
                                // Additional sanity: the stored value should be readable back
                                // from the current pointer.
                                const in_t<T> readback = values_after[ctx->num];
                                const auto    expected = static_cast<T>(res);
                                if (readback != expected)
                                {
                                    INVARIANT_FAILED("apply_atomic_op ASYNC lambda: chunk={} num={} readback mismatch: "
                                                     "wrote={} but readback={}",
                                                     static_cast<const void*>(&ctx->chunk), ctx->num, expected,
                                                     readback);
                                }
                            }
                        });

                    proxy.mark_for_compression_if_needed_with_value(v_old, proxy.read_values_async(ctx));
                });
        }
    }
    else
    {
        compute_operation();
    }
}

// arithmetic_assign (declared in proxy_base.h): sync body + deferred replay body,
// as in apply_atomic_op. Returns nothing: the operators return the proxy, and on
// the deferred path the result does not exist yet anyway. The reader_sync below
// is what makes v_old -- the input of the read-modify-write -- trustworthy.
template <typename Derived, typename T, typename ChunkType>
template <typename U, typename Op>
void ProxyBase<Derived, T, ChunkType>::arithmetic_assign(in_t<U> v, Op operation) noexcept
{
    if constexpr (chunk_config::stats)
    {
        increm_chunk_write<ChunkType>(v);
    }

    // ── Cross-rank RMW materialisation (CHUNK_SYNC_ON_READ) ───────────────────
    // A read-modify-write needs a fresh v_old. Under sync_on_read the write path
    // no longer flushes (P1), so a deferred write to THIS offset enqueued on
    // ANOTHER rank may be unmaterialised — load_value() would then read a stale
    // v_old and the RMW would clobber that foreign write (the ~1/41-run
    // "read 1 != 2" lost-write corruption). Materialise cross-rank HERE, at the
    // RMW entry, BEFORE begin_write, so the drain is not re-entrant (that was the
    // earlier SIGSEGV). reader_sync is a precise gate: no pending -> no-op, so a
    // plain-write phase keeps the P1 fast path. Plain assignment reads no v_old
    // and never reaches this engine.
    chunk.reader_sync(num);

    using CalcType = std::common_type_t<T, base_t<U>>;

    auto op_lambda = [this, &v, operation] // v: the call's parameter, alive for the call
    {
        execute_operation(
            [this, &v, operation]
            {
                const in_t<T> current = load_value();
                // 1. Promote storage and operand to higher precision
                // 2. Perform the math in high precision
                const CalcType intermediate = operation(static_cast<CalcType>(current), static_cast<CalcType>(v));
                // 3. Truncate back to storage type T

                const T result = static_cast<T>(intermediate);
                store_value(result);

                if constexpr (chunk_config::use_fast_block_compression)
                {
                    derived().mark_for_compression_if_needed_with_value(current, result);
                }
            });
    };

    if constexpr (chunk_config::use_fast_block_compression)
    {
        if (const bool need_async = write_with_optional_lock_info(op_lambda); need_async)
        {
            enqueue_async_arithmetic(
                v, operation,
                [&](auto* ctx, auto& proxy)
                {
                    proxy.execute_operation_async(
                        ctx,
                        // By reference: execute_operation_async() calls it at once,
                        // while the outer lambda's proxy is alive. Read only by the
                        // checked builds' invariants, hence the explicit discard.
                        [ctx, &proxy](T* values)
                        {
                            static_cast<void>(proxy);
                            const in_t<T>  current = values[ctx->num];
                            const CalcType intermediate =
                                ctx->operation(static_cast<CalcType>(current), static_cast<CalcType>(ctx->operand));
                            const auto final_val = static_cast<T>(intermediate);

                            if constexpr (build_config::any_check)
                            {
                                T* current_chunk_ptr = ctx->chunk.values_load();
                                if (current_chunk_ptr != values)
                                {
                                    invariant_failed(std::source_location::current(),
                                                     "Lambda async ARITH: STALE values pointer! "
                                                     "lambda_received={} chunk.values_load()={} chunk={} num={}",
                                                     static_cast<const void*>(values),
                                                     static_cast<const void*>(current_chunk_ptr),
                                                     static_cast<const void*>(&ctx->chunk), ctx->num);
                                }
                                if (proxy.is_cst_val1())
                                {
                                    invariant_failed(
                                        std::source_location::current(),
                                        "Lambda async ARITH: chunk={} num={} is_cst_val1 just before store!",
                                        static_cast<const void*>(&ctx->chunk), ctx->num);
                                }
                                if constexpr (requires { proxy.is_cst_val2(); })
                                {
                                    if (proxy.is_cst_val2())
                                    {
                                        invariant_failed(
                                            std::source_location::current(),
                                            "Lambda async ARITH: chunk={} num={} is_cst_val2 just before store!",
                                            static_cast<const void*>(&ctx->chunk), ctx->num);
                                    }
                                }
                            }

                            if constexpr (chunk_config::ring_log_debug)
                            {
                                auto&      chunk_ref = ctx->chunk;
                                const auto chunk_id =
                                    static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&chunk_ref) >> 4);
                                const uint32_t info =
                                    (chunk_id & 0x00FFFFFFU) | (static_cast<uint32_t>(ctx->num & 0xFF) << 24);

                                // Race detector: if values still points to cst_val here,
                                // the chunk was re-compressed between create_value and
                                // values_load — about to corrupt cst_val.
                                T* const cst1 = chunk_ref.get_cst_val1();

                                bool is_cst2 = false;
                                if constexpr (requires { chunk_ref.tab_cst_val2_holder::is_cst_val2_ptr(values); })
                                {
                                    is_cst2 = chunk_ref.tab_cst_val2_holder::is_cst_val2_ptr(values);
                                }

                                if (values == cst1 || is_cst2)
                                {
                                    ring_log(get_tid(), EV_VALUES_LOAD_IS_CST, info);
                                    // Skip the corrupting write so cst_val stays clean
                                    // and observability is preserved.
                                    return;
                                }
                                ring_log(get_tid(), EV_LAMBDA_ASYNC_ARITH, info);
                            }

                            if constexpr (detail::relaxed_element_stores_v<ChunkType, T>)
                                detail::store_element_relaxed(values + ctx->num, static_cast<T>(final_val));
                            else
                                values[ctx->num] = final_val;
                        });
                });
        }
    }
    else
    {
        op_lambda();
    }

}
// bitwise_assign (declared in proxy_base.h): the single engine of the bitwise
// compound operators. It operates on the raw values array
// (execute_inplace_operation), so there is no load/store pair to guard -- only
// the promotion to std::common_type_t and the truncation back to T. Returns
// nothing, so the whole operation can be deferred without anyone waiting.
template <typename Derived, typename T, typename ChunkType>
template <typename U, typename Op>
void ProxyBase<Derived, T, ChunkType>::bitwise_assign(in_t<U> v, Op operation) noexcept
    requires std::is_integral_v<T> && std::is_integral_v<U>
{
    if constexpr (chunk_config::stats)
    {
        increm_chunk_write<ChunkType>(v);
    }

    // Cross-rank RMW materialisation (CHUNK_SYNC_ON_READ): the same gate as
    // arithmetic_assign / apply_inplace_operation / Proxy2A::apply_compound_
    // operation. This engine reads val[num] in place; a deferred write to THIS
    // offset still queued on another rank (or the plain `= v` that preceded
    // this op and was refused) has not reached the buffer yet, so the RMW
    // would compute from a stale cell and the two writes would resolve in the
    // wrong order. Precise gate: no pending -> no-op. The bitwise engines were
    // the only RMW paths without it (shrink_bench chunk2a dyn, 32-56 threads:
    // tab_long with += *= -- passes, tab_ulong with |= &= on the same indices
    // in the same loop loses whole per-thread slices).
    chunk.reader_sync(num);
    using CalcType = std::common_type_t<T, U>;

    auto op_lambda = [this, &v, operation] // v: the call's parameter, alive for the call
    {
        derived().execute_inplace_operation(
            [&v, operation](T& e)
            {
                const CalcType intermediate = operation(static_cast<CalcType>(e), static_cast<CalcType>(v));
                e                           = static_cast<T>(intermediate);
            });
    };

    if constexpr (chunk_config::use_fast_block_compression)
    {
        if (const bool need_async = write_with_optional_lock_info(op_lambda); need_async)
        {
            // The replay body outlives this call (enqueue_async_arithmetic
            // stores it): it must not capture anything of this frame. The
            // operation is read from the context, which holds its own copy --
            // as arithmetic_assign does. It used to capture `operation` by
            // reference through [&], dangling at replay; only stateless
            // functors (std::bit_or<> ...) made that harmless.
            enqueue_async_arithmetic(v, operation,
                                     [](auto* ctx, auto& proxy)
                                     {
                                         proxy.execute_inplace_operation_async(
                                             ctx,
                                             [ctx](T& e)
                                             {
                                                 const CalcType intermediate = ctx->operation(
                                                     static_cast<CalcType>(e), static_cast<CalcType>(ctx->operand));
                                                 e = static_cast<T>(intermediate);
                                             });
                                     });
        }
    }
    else
    {
        op_lambda();
    }
}

// apply_inplace_operation (declared in proxy_base.h): ++/--/in-place engine. The
// two bodies are supplied by the caller rather than built here, so this only
// picks the sync one or, when the write is refused, allocates the context and
// enqueues the async one. Context type follows the proxy family: the light
// ArithmeticProxyContext for Proxy<T>, the chunk-aware ArithmeticContext for
// Proxy2<T> / Proxy2A<T>.
template <typename Derived, typename T, typename ChunkType>
template <typename SyncOp, typename AsyncOp>
void ProxyBase<Derived, T, ChunkType>::apply_inplace_operation(SyncOp                   sync_op,
                                                               [[maybe_unused]] AsyncOp async_op) noexcept
{
    // Cross-rank RMW materialisation (CHUNK_SYNC_ON_READ): ++/-- and other
    // in-place ops read val[num] directly; pull any foreign deferred write to
    // this offset to applied state first, outside begin_write. See
    // arithmetic_assign for the rationale. No-op when nothing is pending.
    chunk.reader_sync(num);

    // --- Synchronous fast‑path -------------------------------------------------

    if constexpr (chunk_config::use_fast_block_compression)
    {
        if (const bool need_async = write_with_optional_lock_info(sync_op); need_async)
        {
            using ProxyType = std::remove_reference_t<decltype(derived())>;

            using CtxType = std::conditional_t<is_simple_proxy<ProxyType>::value,
                                               ArithmeticProxyContext<T>,                 // Proxy<T>
                                               ArithmeticContext<T, ChunkType, ProxyType> // Proxy2<T>, Proxy2A<T>
                                               >;
            // Create context
            auto* ctx = pool_new_ctx<CtxType>(derived(), ptr_on_array);

            // Async body: rebuild a fresh proxy at execution time

            auto write_body = [ctx, op = std::forward<AsyncOp>(async_op)]() mutable
            {
                const auto proxy = ProxyBase::template make_proxy_with_info<ProxyType>(ctx);

                op(ctx, proxy);
            };

            enqueue_deferred_write(ctx, std::move(write_body));
        }
    }
    else
    {
        sync_op();
    }
}

} // namespace shrink_array