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

#include <atomic>          // std::atomic, std::memory_order
#include <cstdint>         // uint32_t, uintptr_t (the ring_log payloads)
#include <cstdlib>         // std::size_t
#include <source_location> // std::source_location
#include <tuple>           // std::tuple_size_v, std::ignore
#include <type_traits>     // std::remove_cvref_t, std::common_type_t, std::conditional_t

#include "fast_allocators/internal/raw_memory.h" // address_bits
#include "chunk/internal/in_t.h"                 // InT<T>: the element snapshots
#include "chunk/internal/proxy_base.h" // the declarations these bodies belong to (self-contained, like the other *_inline.h)
#include "chunk/internal/chunk_size_policy.h"  // ChunkElems — used HERE, not via includers
#include "chunk/internal/phase_guard.h"        // check-build guard of the phase protocol
#include "chunk/config.h"                      // build_config::any_check, chunk_config::use_fast_block_compression
#include "chunk/chunk_traits.h"                // is_zero_any_family_v
#include "chunk/PMP_CHUNK.h"                   // get_thr_rank_fast, CHUNK_PMP_IN_PARALLEL_REGION
#include "chunk/ring_log_debug.h"              // ScopedRingLog, ring_log, get_tid
#include "chunk/internal/arithmetic_context.h" // ArithmeticProxyContext, ArithmeticContext
#include "chunk/internal/array_chunk_for.h"    // ArrayChunkForT
#include "chunk/internal/concept_chunk.h"      // IsChunk2, ChunkTemplateOf
#include "chunk/internal/concept_is_clean_assignement.h" // BaseT
#include "chunk/internal/def_force_inline.h"             // CHUNK_FORCE_INLINE
#include "chunk/internal/invariant_check.h"              // CHUNK_INVARIANT_FAILED
#include "chunk/internal/write_compress_sync_policy.h"   // WriteCompressSyncPolicy
#include "chunk/internal/test_hooks.h"                   // TEST KNOB: test_hooks::forced_deferral

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
    {
        vec.emplace_back(chunk, num_chunk, num_tab, size);
    }
    else
    {
        vec.emplace_back(chunk, num_chunk, num_tab);
    }
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
 * @param ptr     Array pointer recorded at enqueue time.
 * @return The table number of the chunk.
 *
 * @note The detection probes the SYNCHRONOUS `get_num_tab(p)`: having the
 *       pointer-keyed sync lookup identifies the family that also provides the
 *       two-argument async one. Only Proxy reaches this helper, so the
 *       one-argument branch is never instantiated (no `get_num_tab_async(ctx)`
 *       overload exists for it).
 */
template <typename ProxyT>
constexpr int get_num_tab_async_helper(auto* ctx, const ProxyT& proxy, auto ptr) noexcept
{
    if constexpr (requires { proxy.get_num_tab(ptr); })
    {
        return proxy.get_num_tab_async(ctx, ptr);
    }
    else
    {
        return proxy.get_num_tab_async(ctx);
    }
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
template <typename Derived, typename T, typename ChunkT, bool Enabled>
void ProxyBaseFastCompPolicy<Derived, T, ChunkT, Enabled>::set_ptr_is_inside_chunk_list_to_compress_async(auto* ctx)
{
    const int tab       = ctx->num_tab;
    const int num_chunk = ctx->num_chunk;

    auto& arr = tab_is_inside_chunk_list_to_compress[static_cast<std::size_t>(
        tab)]; // not const: the flags are written through the pointer

    ptr_is_inside_chunk_list_to_compress_ = &arr[num_chunk];
}

/**
 * @brief Flags this chunk as a compression candidate on a caller-supplied list.
 *
 * The per-chunk flag is claimed with a CAS, so exactly one writer pushes the
 * chunk: the list never holds a duplicate. Losing the CAS is normal while the
 * chunk waits in the list.
 *
 * Serial builds do nothing here — the whole body is OpenMP-only.
 *
 * @param vec_tuple Candidate list to push onto.
 * @pre `ptr_is_inside_chunk_list_to_compress_` is armed, i.e. set_info_on_chunk()
 *      ran on this write path (checked under `build_config::any_check`).
 */
template <typename Derived, typename T, typename ChunkT>
void ProxyBase<Derived, T, ChunkT>::add_to_compression_list([[maybe_unused]] auto& vec_tuple) const
{
    if constexpr (build_config::threaded)
    {
        if constexpr (build_config::any_check)
        {
            if (ptr_is_inside_chunk_list_to_compress_ == nullptr) [[unlikely]]
            {
                CHUNK_INVARIANT_FAILED("add_to_compression_list(vec_tuple): chunk={} num={} "
                                       "num_chunk={} num_tab={} "
                                       "ptr_is_inside_chunk_list_to_compress is null "
                                       "(set_info_on_chunk not called on this write path?)",
                                       &chunk_, num_, num_chunk_, num_tab_);
            }
        }
        // Test, then test-and-set. The flag is almost always 1, and a CAS takes
        // the line exclusively even when it fails -- one char per chunk, 64
        // chunks per line written by different ranks: constant line ping-pong.
        // The relaxed probe decides nothing by itself: the CAS still does.
        char expected = 0;
        if (ptr_is_inside_chunk_list_to_compress_->load(std::memory_order_relaxed) == 0 &&
            ptr_is_inside_chunk_list_to_compress_->compare_exchange_strong(expected, 1, std::memory_order_acq_rel,
                                                                           std::memory_order_acquire))
        {
            if constexpr (build_config::is_debug)
            {
                // num_tab is never reused (every array and re-chunk takes the next
                // one): do not check it against a fixed ceiling. The bounds are the
                // flag table's: its outer size, and this array's row.
                const auto& flags = FastCompPolicy::tab_is_inside_chunk_list_to_compress;
                if (num_chunk_ < 0 || num_tab_ < 0 || static_cast<std::size_t>(num_tab_) >= flags.size() ||
                    static_cast<std::size_t>(num_chunk_) >= flags[static_cast<std::size_t>(num_tab_)].size())
                    [[unlikely]]
                {
                    CHUNK_INVARIANT_FAILED("add_to_compression_list: num_chunk={} num_tab={} outside the flag table "
                                           "({} tables)",
                                           num_chunk_, num_tab_, flags.size());
                }
            }
            if constexpr (chunk_config::stats)
            {
                increm_chunk_mark_ok<ChunkT>();
            }
            push_to_compress_list(vec_tuple, &chunk_, num_chunk_, num_tab_, ChunkElems{this->chunk_size()});
        }
        else if constexpr (chunk_config::stats)
        {
            // CHUNK_STATS: flag ALREADY 1, normal while the chunk waits in the list.
            // mark_refused growing while mark_ok stays flat means the flag is not
            // reset after the compression sweep.
            increm_chunk_mark_refused<ChunkT>();
        }
    }
}

/**
 * @brief Flags this chunk as a compression candidate on this thread's own list.
 *
 * Same once-only CAS discipline as the vec_tuple overload, with two differences:
 * the destination list is looked up from the calling thread's rank, and a
 * relaxed load short-circuits the CAS when the chunk is already flagged (the
 * common case; the load is an order of magnitude cheaper than the CAS).
 *
 * Only compiled under `chunk_config::use_fast_block_compression`; the serial
 * fallback below it uses plain accesses instead of atomics.
 *
 * @pre `ptr_is_inside_chunk_list_to_compress_` is armed (checked under
 *      `build_config::any_check`).
 */
template <typename Derived, typename T, typename ChunkT>
void ProxyBase<Derived, T, ChunkT>::add_to_compression_list(this const Derived& self)
{
    if constexpr (chunk_config::use_fast_block_compression)
    {
        if constexpr (build_config::any_check)
        {
            if (self.ptr_is_inside_chunk_list_to_compress_ == nullptr) [[unlikely]]
            {
                CHUNK_INVARIANT_FAILED("add_to_compression_list (fast-block): chunk={} num={} "
                                       "num_chunk={} num_tab={} "
                                       "ptr_is_inside_chunk_list_to_compress is null "
                                       "(set_info_on_chunk not called on this write path?)",
                                       &self.chunk_, self.num_, self.num_chunk_, self.num_tab_);
            }
        }
        // CHUNK_STATS: see the vec_tuple overload for reading mark_refused / mark_ok.
        if (self.ptr_is_inside_chunk_list_to_compress_->load(std::memory_order_relaxed) != 0)
        {
            if constexpr (chunk_config::stats)
            {
                increm_chunk_mark_refused<ChunkT>();
            }
        }
        else
        {
            char expected = 0;
            if (self.ptr_is_inside_chunk_list_to_compress_->compare_exchange_strong(
                    expected, 1, std::memory_order_acq_rel, std::memory_order_acquire))
            {
                if constexpr (chunk_config::stats)
                {
                    increm_chunk_mark_ok<ChunkT>();
                }
                const int thr = get_thr_rank_fast();
                auto&     vec = self.get_vec_tuple_to_compress_potentially_thr(thr);

                if constexpr (build_config::is_debug)
                {
                    // Bounds of the flag table, not a fixed num_tab ceiling: see above.
                    const auto& flags = FastCompPolicy::tab_is_inside_chunk_list_to_compress;
                    if (self.num_chunk_ < 0 || self.num_tab_ < 0 ||
                        static_cast<std::size_t>(self.num_tab_) >= flags.size() ||
                        static_cast<std::size_t>(self.num_chunk_) >=
                            flags[static_cast<std::size_t>(self.num_tab_)].size()) [[unlikely]]
                    {
                        CHUNK_INVARIANT_FAILED(
                            "add_to_compression_list: num_chunk={} num_tab={} outside the flag table "
                            "({} tables)",
                            self.num_chunk_, self.num_tab_, flags.size());
                    }
                }

                push_to_compress_list(vec, &self.chunk_, self.num_chunk_, self.num_tab_, ChunkElems{self.chunk_size()});
                if constexpr (IsChunk2<ChunkTemplateOf<ChunkT>::template type, T>)
                {
                    Derived::note_candidate_pushed(vec); // the threshold test (zero_one_value::Proxy trigger)
                }
            }
        }
    }
    else
    {
        if constexpr (build_config::not_threaded)
        {
            if (!(*self.ptr_is_inside_chunk_list_to_compress_))
            {
                *self.ptr_is_inside_chunk_list_to_compress_ = 1;
                auto& vec                                   = self.get_vec_tuple_to_compress_potentially();
                push_to_compress_list(vec, &self.chunk_, self.num_chunk_, self.num_tab_, ChunkElems{self.chunk_size()});
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
 * apply is decided per proxy type with `requires`, since zero::Proxy, zero_one_value::Proxy and
 * zero_any::Proxy do not all carry the same state.
 *
 * The frozen geometry is restored too: chunk_size() must not read it back from
 * `ptr_on_array`, which may point at an array destroyed since the enqueue.
 *
 * @tparam ProxyT    Proxy type to rebuild.
 * @tparam Ctx       Context type recorded at enqueue time.
 */
template <typename Derived, typename T, typename ChunkT, bool Enabled>
template <typename ProxyT, typename Ctx>
ProxyT ProxyBaseFastCompPolicy<Derived, T, ChunkT, Enabled>::make_proxy_with_info(Ctx* ctx)
{
    auto&     chunk = ctx->chunk;
    const int num   = ctx->num;

    std::atomic<T*>& atomic_values = chunk.values_;

    ProxyT proxy(atomic_values, num, chunk);

    proxy.set_num_chunk(ctx->num_chunk);

    // chunk_size() must not dereference a possibly-dead ptr_on_array.
    proxy.set_chunk_size_frozen(ctx->chunk_size_frozen, ctx->alloc_key_frozen);

    if constexpr (requires { proxy.BaseSpec::set_info_second_value_ind_from_key(ctx->ptr_on_array); })
    {
        proxy.BaseSpec::set_info_second_value_ind_from_key(ctx->ptr_on_array);
    }

    // A probe fails open: for the zero_any family, whose replayed proxy NEEDS its table
    // number (its second-value slot is found through it), a false probe is a build
    // error, not a proxy left at num_tab -1 (a null second-value slot at replay).
    static_assert(
        !is_zero_any_family_v<ChunkT> || requires(ProxyT probe) { probe.set_num_tab(0); },
        "make_proxy_with_info: a zero_any proxy's set_num_tab(int) must be reachable from here "
        "(it is private: the proxy befriends ::shrink_array::ProxyBaseFastCompPolicy)");
    if constexpr (requires(ProxyT probe) { probe.set_num_tab(0); })
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
 * zero_one_value keeps its candidates in a context-dependent list, hence the
 * `IsChunk2` branch on which per-thread list to push onto.
 *
 * @param self The proxy, deduced (explicit object parameter).
 * @param ctx  Deferred-operation context being replayed.
 */
// Deducing this: `self` replaces the CRTP down-cast. The class keeps its
// Derived parameter; the calls go through the explicit object parameter.
template <typename Derived, typename T, typename ChunkT, bool Enabled>
void ProxyBaseFastCompPolicy<Derived, T, ChunkT, Enabled>::add_to_compression_list_async(this const auto& self,
                                                                                         auto*            ctx)
{
    auto&     chunk = ctx->chunk;
    const int num   = ctx->num;

    using ProxyType = std::remove_cvref_t<decltype(self)>;

    std::atomic<T*>& atomic_values = chunk.values;
    ProxyType        proxy(atomic_values, num, chunk);

    const int num_tab = get_num_tab_async_helper(ctx, proxy, ctx->ptr_on_array);

    auto& flag = Derived::tab_is_inside_chunk_list_to_compress[num_tab][ctx->num_chunk];

    // Relaxed probe before the CAS: the chunk is usually already marked.
    if (flag.load(std::memory_order_relaxed) != 0)
    {
        return;
    }

    char expected = 0;
    if (flag.compare_exchange_strong(expected, 1, std::memory_order_acq_rel, std::memory_order_acquire))
    {
        const int thr = get_thr_rank_fast();

        auto& vec = [&]() -> auto&
        {
            if constexpr (IsChunk2<ChunkTemplateOf<ChunkT>::template type, T>)
            {
                return self.get_vec_tuple_to_compress_potentially_async_thr(ctx, thr);
            }
            else
            {
                return self.get_vec_tuple_to_compress_potentially_thr(thr);
            }
        }();

        push_to_compress_list(vec, &ctx->chunk, ctx->num_chunk, num_tab, self.chunk_size());
        if constexpr (IsChunk2<ChunkTemplateOf<ChunkT>::template type, T>)
        {
            Derived::note_candidate_pushed(vec); // the threshold test (zero_one_value::Proxy trigger)
        }
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
 * Without fast block compression the body simply runs, plus a zero_any-only
 * sequential compression trigger (see the inline note).
 *
 * Each step's correctness argument is kept inline next to the code.
 *
 * @tparam WriteCallable Callable performing the actual store.
 * @tparam LbdInfo      Callable arming the per-chunk bookkeeping beforehand.
 * @param write_body Write to perform.
 * @param lbd_info   Ran first on the fast path; typically set_info_on_chunk().
 * @return True if the write did NOT happen and must be deferred by the caller;
 *         false if it was applied.
 */
template <typename Derived, typename T, typename ChunkT>
template <typename WriteCallable, typename LbdInfo>
[[nodiscard]] CHUNK_FORCE_INLINE bool
ProxyBase<Derived, T, ChunkT>::write_with_optional_lock_impl([[maybe_unused]] this Derived& self,
                                                             WriteCallable&&                write_body,
                                                             [[maybe_unused]] LbdInfo       lbd_info) noexcept
{
    if constexpr (chunk_config::use_fast_block_compression)
    {
        lbd_info(); // common initialization required for fast-path

        // ─── Flush replay: apply DIRECTLY, never re-defer ────────────────────
        // Inside a flush drain (begin_flush held, hard compressors drained) no
        // compressor can steal values. Do not consult the gate here: a replayed
        // op would re-enqueue itself forever under sustained compression.
        if (compression_mgr.in_flush_replay) [[unlikely]]
        {
            std::forward<WriteCallable>(write_body)();
            return false;
        }

        // ─── Cross-rank direct-write guard ──────────────────────────────────
        // A pending op enqueued by ANOTHER rank at this offset would replay
        // AFTER our write and clobber it (regions remap thread->rank, so
        // different ranks may write the same element in successive regions).
        // Own-rank pending is safe (a same-offset own op is erased by the WAW
        // guard below) and keeps the bulk-init fast path cheap. The drain runs
        // HERE, before begin_write, so it is never re-entrant. It also covers
        // the deferred-ENQUEUE path below: same-offset ops on different ranks
        // replay in arbitrary order, so the foreign one is materialised first.
        //
        // ─── Own-pending replay exclusion ───────────────────────────────────
        // No replay may be IN FLIGHT on a chunk while a thread direct-writes it.
        // OUR own pending ops (other offsets) can be replayed concurrently by
        // another thread's gate/stage-2 walk, and a replay bypasses the Dekker
        // (in_flush_replay): the two writers would be mutually invisible. Every
        // replayer of rank r's ops holds xflush_lock_rank[r], so holding OUR
        // rank lock for the write excludes them WITHOUT draining anything
        // (owner-recursive, uncontended; the enqueue path takes it anyway).
        [[maybe_unused]] int own_replay_excl_rank = -1;
        // Read once here, reused by the WAW guard below: see there.
        [[maybe_unused]] bool entry_pending = false;
        if constexpr (build_config::threaded)
        {
            entry_pending = self.chunk_.pending_any();
            if (entry_pending) [[unlikely]]
            {
                const int thr_entry = get_thr_rank_fast();
                if (self.chunk_.foreign_pending(thr_entry, self.num_))
                {
                    self.chunk_.reader_sync(self.num_);
                }
                if (self.chunk_.pending_any()) // own ops remain queued: exclude their replay
                {
                    CompressionManagerType::enqueue_lock(thr_entry);
                    own_replay_excl_rank = thr_entry;
                }
            }
        }
        // An aggregate, brace-initialised below: declaring the copy/move
        // operations would make it a non-aggregate.
        struct OwnReplayExclGuard // NOLINT(cppcoreguidelines-special-member-functions)
        {
            ~OwnReplayExclGuard()
            {
                if (rank >= 0)
                {
                    CompressionManagerType::enqueue_unlock(rank);
                }
            }

            int rank;
        };
        const OwnReplayExclGuard own_replay_excl_guard{own_replay_excl_rank};

        // ─── Check #1: pure optimisation, not correctness ───────────────────
        // A probe of the gate (must_defer_write) before touching begin_write's
        // cache line: if a compression is clearly in flight, wait it out or
        // take the slow path without contending the per_thread_writing[thr]
        // line the compressor is busy-waiting on. Reloaded only after a wait.
        //
        // A stale read here is safe: check #2, after announce_write, is the one
        // the handshake's ordering covers -- and must be a fresh load, never
        // this value reused.
        //
        // The gate is a parked compressor or a pass having THIS chunk marked
        // (ChunkBase::steal_mark_); without the marks, any hard compression in
        // flight. Do NOT gate on "a slow-path driver is running" (every writer
        // would defer for someone else's whole flush-and-try) nor on "MY list
        // is dangerous" (the rank's own perform_compression() would replay the
        // write under a flush announce that makes every other start() yield:
        // a circular regime in which nothing compresses). A dangerous rank
        // WRITES and drives its pass after end_write; only a compressor in
        // flight defers a write, after a bounded wait (no flag is held here,
        // so waiting shuts no window; see wait_out_compressor).
        bool gate_closed = compression_mgr.must_defer_write(self.chunk_);
        if (gate_closed) [[unlikely]]
        {
            compression_mgr.wait_out_compressor(self.chunk_);
            gate_closed = compression_mgr.must_defer_write(self.chunk_);
        }

        // TEST KNOB (test_hooks.h, force_defer_every): one write in N takes the
        // deferred exit, alternately at check #1 and #2. 0 unless a harness set it.
        const int  forced        = test_hooks::forced_deferral();
        const bool force_defer_1 = forced == 1;
        const bool force_defer_2 = forced == 2;

        if (!force_defer_1 && !gate_closed) [[likely]]
        {
            const int thr = get_thr_rank_fast();

            // ─── Dekker handshake ───────────────────────────────────────────
            // (1)+(2) announce_write = OUR flag, ordered before the loads
            // below: plain store + compiler barrier under the asymmetric
            // handshake (the compressor's membarrier() does the rest), seq_cst
            // store + fence under the symmetric one. Both arguments live in
            // write_compress_sync_policy.h.
            WriteCompressSyncPolicy<Derived>::announce_write(thr, &self.chunk_);

            // ─── Invariant: our flag must be observably set ─────────────────
            if constexpr (build_config::any_check)
            {
                using Policy = WriteCompressSyncPolicy<Derived>;
                if (!Policy::is_thread_writing_seq(thr))
                {
                    CHUNK_INVARIANT_FAILED("Dekker: per_thread_writing[{}] is FALSE right after begin_write", thr);
                }
            }

            // (3) check THEIR flag -- the chunk's steal mark (acquire), or
            //     hard_inflight_ without the marks: a FRESH load, after the
            //     announce. If it does not see the gate closed, the
            //     compressor's scan of per_thread_writing[thr] sees our flag
            //     and spins -- see announce_write() for why, in either
            //     handshake shape.
            if (!force_defer_2 && !compression_mgr.must_defer_write(self.chunk_)) [[likely]]
            {
                // Dekker established: no compressor mutates chunk.values until
                // end_write. Concurrent writers on the same chunk are handled by
                // the is_cst_val re-checks in handle_{bool,no_bool}_assignment.

                const ScopedRingLog<EventCode::begin_fast_path, EventCode::end_fast_path> fast_log;

                // ─── Pre-write invariant: our flag still holds
                if constexpr (build_config::any_check)
                {
                    using Policy = WriteCompressSyncPolicy<Derived>;
                    if (!Policy::is_thread_writing(thr))
                    {
                        CHUNK_INVARIANT_FAILED(
                            "Dekker: per_thread_writing[{}] cleared between begin_write and write_body", thr);
                    }
                }

                // ─── WAW guard ───────────────────────────────────────────────
                // Writes do not flush, so an OLDER deferred op for THIS offset
                // may sit on our own rank: discard it before the direct store,
                // else a later replay clobbers the fresh value. threaded keeps
                // the branch un-instantiated in serial builds (OpenMP-only
                // machinery). entry_pending is enough as a first test: only this
                // rank's own thread enqueues on it, so no own op can have
                // appeared since the entry read found the chunk clear.
                if constexpr (build_config::threaded)
                {
                    if (entry_pending && self.chunk_.pending_any()) [[unlikely]]
                    {
                        CompressionManagerType::erase_own_rank_op(thr, &self.chunk_, self.num_);
                    }
                }

                std::forward<WriteCallable>(write_body)();

                // ─── Post-write invariant: flag must still be ours ─────────
                if constexpr (build_config::any_check)
                {
                    using Policy = WriteCompressSyncPolicy<Derived>;
                    if (!Policy::is_thread_writing(thr))
                    {
                        CHUNK_INVARIANT_FAILED(
                            "Dekker: per_thread_writing[{}] cleared during write_body (someone else cleared our flag)",
                            thr);
                    }
                }

                WriteCompressSyncPolicy<Derived>::end_write(thr);

                // ─── Compression pass, AFTER the write ───────────────────────
                // The trigger flags the rank's list once it crosses the
                // threshold; the rank's next write -- this one -- drives
                // perform_compression() after its store and end_write. The pass
                // runs the rank's flush_deferred() (announces nothing when empty),
                // then start(), whose parked hard_inflight_ steers the other
                // writers away. One driver per threshold crossing.
                if (compression_mgr.is_dangerous_list(thr)) [[unlikely]]
                {
                    compression_mgr.run_slow_path([&] { self.perform_compression(); });
                }
                else
                {
                    // ─── Backlog drain, fast-path side ──────────────────────
                    // Ops deferred while a compressor was in flight are drained
                    // once the backlog crosses the threshold -- from here, with
                    // no compressor in the way, rather than from the deferred
                    // path (see to_be_queued_for_deferred_processing). One TLS
                    // load when the rank holds nothing.
                    if constexpr (build_config::threaded)
                    {
                        if (CompressionManagerType::thread_may_have_ops_own()) [[unlikely]]
                        {
                            if (CompressionManagerType::pending_ops_rank(thr) > build_config::deferred_drain_threshold)
                            {
                                (void)CompressionManagerType::flush_deferred(thr);
                            }
                        }
                    }
                    self.trigger_compression_if_needed(thr);
                }
                return false;
            }

            const ScopedRingLog<EventCode::begin_slow_path, EventCode::end_slow_path> slow_log;

            // Check #2 failed: a compressor slipped in between check #1 and
            // begin_write. Release our flag, then defer the write.
            WriteCompressSyncPolicy<Derived>::end_write(thr);

            // ─── Invariant after end_write on slow-path-via-check#2 ────────
            if constexpr (build_config::any_check)
            {
                using Policy = WriteCompressSyncPolicy<Derived>;
                if (Policy::is_thread_writing(thr))
                {
                    CHUNK_INVARIANT_FAILED("Dekker: per_thread_writing[{}] still TRUE after end_write (slow path)",
                                           thr);
                }
            }

            self.drive_compression_if_dangerous(thr);
            return true;
        }

        // Check #1 failed: a compressor is in flight. No end_write: begin_write
        // was never called, and the store would dirty a contended line.
        if constexpr (build_config::any_check)
        {
            using Policy  = WriteCompressSyncPolicy<Derived>;
            const int thr = get_thr_rank_fast();
            if (Policy::is_thread_writing(thr))
            {
                CHUNK_INVARIANT_FAILED(
                    "Dekker: per_thread_writing[{}] unexpectedly TRUE on check#1 fail (begin_write not called)", thr);
            }
        }

        self.drive_compression_if_dangerous(get_thr_rank_fast());
        return true;
    }
    else
    {
        // Non-OMP path: execute the write directly. Nothing else would fire
        // block compression in sequential builds, so drive it here;
        // perform_compression() self-checks the threshold (size > szm/2).
        std::forward<WriteCallable>(write_body)();

        // The zero_any family only: its compression algorithm is identical in
        // OMP and non-OMP. zero/zero_one_value have distinct non-OMP paths and
        // must NOT be triggered here.
        if constexpr (build_config::not_threaded && is_zero_any_family_v<ChunkT>)
        {
            // Skip during rechunk_refill / flush replay (in_flush_replay): the refill
            // materialises a chunk here and RAW-fills it right after; compressing it in
            // between frees the fresh block, and the raw-fill writes a freed block
            // (slab free-list corruption). Mirrors the OMP path's early return.
            if (!compression_mgr.in_flush_replay)
            {
                self.perform_compression();
            }
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
template <typename Derived, typename T, typename ChunkT>
unsigned int ProxyBase<Derived, T, ChunkT>::chunk_size() const noexcept
{
    // Armed (non-zero) on replay: do NOT dereference ptr_on_array there.
    if (chunk_size_frozen_ != 0U) [[unlikely]]
    {
        return chunk_size_frozen_;
    }

    return ptr_on_array_->chunk_size();
}

template <typename Derived, typename T, typename ChunkT>
unsigned int ProxyBase<Derived, T, ChunkT>::alloc_key() const noexcept
{
    if (chunk_size_frozen_ != 0u) [[unlikely]]
    {
        return alloc_key_frozen_;
    }
    return ptr_on_array_->alloc_key();
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
 */
template <typename Derived, typename T, typename ChunkT>
template <typename CtxType, typename LambdaBody>
void ProxyBase<Derived, T, ChunkT>::enqueue_deferred_write(CtxType* ctx, LambdaBody&& body) const
{
    // This const guards the proxy, not the array: enqueueing marks the array's
    // dirty bitset now, the write itself replays at the phase end.
    auto* arr = const_cast<ArrayChunkForT<ChunkT>*>(ptr_on_array_);
    arr->mark_chunk_dirty(ctx->num_chunk);
    increm_chunk_gate_deferred<ChunkT>();

    // The moment the protocol matters: this write is about to be parked in a
    // per-thread queue that only the phase end drains. See phase_guard.h.
    phase_guard::check_write(&chunk_, static_cast<long>(ctx->num));

    const int thr_rank = get_thr_rank_fast();

    using OpType = CompressionManagerType::template DeferredOperation<CtxType, std::decay_t<LambdaBody>>;
    auto* op_ptr = pool_new_op<OpType>(ctx, std::forward<LambdaBody>(body), thr_rank);

    chunk_.to_be_queued_for_deferred_processing(thr_rank, ctx->num, op_ptr);
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
 * @param operand          Operand, stored converted to T.
 * @param operation  Operation to replay.
 * @param async_body Replay body.
 */
template <typename Derived, typename T, typename ChunkT>
template <typename U, typename Op, typename AsyncBody>
void ProxyBase<Derived, T, ChunkT>::enqueue_async_arithmetic(this Derived& self,
                                                             const U&      operand,
                                                             Op            operation,
                                                             AsyncBody&&   async_body)
{
    using ProxyType = Derived;

    using CtxType = std::conditional_t<is_simple_proxy<ProxyType>::value, ArithmeticOperationProxyContext<T, Op>,
                                       ArithmeticOperationContext<T, Op, ChunkT, ProxyType>>;

    auto* ctx = pool_new_ctx<CtxType>(self, static_cast<T>(operand), operation, self.ptr_on_array_);

    auto op_async = [ctx, body = std::forward<AsyncBody>(async_body)]() mutable
    {
        auto proxy = ProxyBase::template make_proxy_with_info<ProxyType>(ctx);

        body(ctx, proxy);
    };

    self.enqueue_deferred_write(ctx, std::move(op_async));
}
/**
 * @brief Tests a proxy's second-constant state, false for proxies that have none.
 *
 * Lets the shared invariant checks below name is_cst_val2() unconditionally:
 * only zero_one_value / zero_any carry a second constant value, and zero
 * proxy simply answers false.
 *
 * @param proxy Proxy to test.
 * @return True if @p proxy exposes is_cst_val2() and it holds.
 */
template <typename P>
constexpr bool check_is_cst_val2([[maybe_unused]] const P& proxy)
{
    if constexpr (requires { proxy.is_cst_val2(); })
    {
        return proxy.is_cst_val2();
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
 */
template <typename Derived, typename T, typename ChunkT>
template <typename U, typename Op>
    requires std::is_invocable_r_v<T, Op, T, U>
void ProxyBase<Derived, T, ChunkT>::apply_atomic_op(this Derived& self, Op operation, const U& operand, InT<T> v_old)
{
    if constexpr (chunk_config::stats)
    {
        increm_chunk_write<ChunkT>(operand);
    }

    using CalcType = std::common_type_t<T, BaseT<U>>;

    const auto compute_operation =
        [&self, operation, &operand, v_old] // operand: the call's parameter, alive for the call
    {
        std::ignore = v_old;

        self.execute_arithmetic_compound_operation(
            operation, static_cast<T>(operand),
            [&self, &operand,
             operation](auto) // operation functor is used instead of the passed one to ensure consistency
            {
                const InT<T>   current = self.load_value();
                const CalcType res     = operation(static_cast<CalcType>(current), static_cast<CalcType>(operand));

                // ─── Pre-store check: cst_val should not be visible here under Dekker ───
                if constexpr (build_config::any_check)
                {
                    if (self.is_cst_val1())
                    {
                        CHUNK_INVARIANT_FAILED(
                            "apply_atomic_op SYNC: chunk={} num={} is_cst_val1 BEFORE store_value (Dekker breach?)",
                            &self.chunk_, self.num_);
                    }

                    if (check_is_cst_val2(self))
                    {
                        CHUNK_INVARIANT_FAILED(
                            "apply_atomic_op SYNC: chunk={} num={} is_cst_val2 BEFORE store_value (Dekker breach?)",
                            &self.chunk_, self.num_);
                    }
                }

                const T* values_before = self.chunk_.values_load();

                self.store_value(static_cast<T>(res));

                // ─── Post-store check: values pointer should not have been CASed away ───
                if constexpr (build_config::any_check)
                {
                    const T* values_after = self.chunk_.values_load();
                    if (values_after != values_before)
                    {
                        CHUNK_INVARIANT_FAILED(
                            "apply_atomic_op SYNC: chunk={} num={} values pointer CHANGED across store: "
                            "before={} after={} ; the store may have been LOST",
                            &self.chunk_, self.num_, static_cast<const void*>(values_before),
                            static_cast<const void*>(values_after));
                    }
                }
            });

        if constexpr (chunk_config::use_fast_block_compression)
        {
            self.mark_for_compression_if_needed_with_value(v_old, self.read_values());
        }
    };

    if constexpr (chunk_config::use_fast_block_compression)
    {
        if (const bool need_async = self.write_with_optional_lock_info(compute_operation); need_async)
        {
            self.enqueue_async_arithmetic(
                operand, operation,
                [v_old](auto* ctx, auto& proxy)
                {
                    proxy.execute_arithmetic_compound_operation_async(
                        ctx, ctx->operation, ctx->operand,
                        [ctx, &proxy](auto op_func)
                        {
                            // ─── Pre-read invariant: not in a cst_val state (the async
                            //     wrapper called create_value upstream). ───
                            if constexpr (build_config::any_check)
                            {
                                if (proxy.is_cst_val1())
                                {
                                    CHUNK_INVARIANT_FAILED("apply_atomic_op ASYNC lambda: chunk={} num={} is_cst_val1 "
                                                           "before read+store (decompression skipped?)",
                                                           &ctx->chunk, ctx->num);
                                }
                                if (check_is_cst_val2(proxy))
                                {
                                    CHUNK_INVARIANT_FAILED("apply_atomic_op ASYNC lambda: chunk={} num={} is_cst_val2 "
                                                           "before read+store (decompression skipped?)",
                                                           &ctx->chunk, ctx->num);
                                }
                            }

                            const InT<T> current = proxy.read_values_async(ctx);

                            const CalcType res =
                                op_func(static_cast<CalcType>(current), static_cast<CalcType>(ctx->operand));

                            const T* values_before = ctx->chunk.values_load();

                            proxy.store_value_async(ctx, static_cast<T>(res));

                            // ─── Post-store check: detect lost write due to concurrent compression ───
                            if constexpr (build_config::any_check)
                            {
                                const T* values_after = ctx->chunk.values_load();
                                if (values_after != values_before)
                                {
                                    CHUNK_INVARIANT_FAILED(
                                        "apply_atomic_op ASYNC lambda: chunk={} num={} values pointer "
                                        "CHANGED across store_value_async: before={} after={} ; "
                                        "the store may have landed in a stale buffer (LOST)",
                                        &ctx->chunk, ctx->num, static_cast<const void*>(values_before),
                                        static_cast<const void*>(values_after));
                                }
                                // raw-read: deferred replay, after materialisation -- flushes exclude the stealers
                                const InT<T> readback = elem_get(values_after, static_cast<std::size_t>(ctx->num));
                                const auto   expected = static_cast<T>(res);
                                if (readback != expected)
                                {
                                    CHUNK_INVARIANT_FAILED(
                                        "apply_atomic_op ASYNC lambda: chunk={} num={} readback mismatch: "
                                        "wrote={} but readback={}",
                                        &ctx->chunk, ctx->num, expected, readback);
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
template <typename Derived, typename T, typename ChunkT>
template <typename U, typename Op>
void ProxyBase<Derived, T, ChunkT>::arithmetic_assign(this Derived& self, InT<U> operand, Op operation) noexcept
{
    self.check_not_stale();
    if constexpr (chunk_config::stats)
    {
        increm_chunk_write<ChunkT>(operand);
    }

    // ── Cross-rank RMW materialisation ──────────────────────────────────────
    // A read-modify-write needs a fresh v_old: a deferred write to THIS offset
    // queued on ANOTHER rank may be unmaterialised, and the RMW would clobber
    // it. Materialise HERE, BEFORE begin_write, so the drain is not re-entrant.
    // reader_sync is a no-op when nothing is pending.
    self.chunk_.reader_sync(self.num_);

    using CalcType = std::common_type_t<T, BaseT<U>>;

    const auto op_lambda = [&self, &operand, operation] // operand: the call's parameter, alive for the call
    {
        self.execute_operation(
            [&self, &operand, operation]
            {
                const InT<T> current = self.load_value();
                // Compute in CalcType, truncate back to T.
                const CalcType intermediate = operation(static_cast<CalcType>(current), static_cast<CalcType>(operand));

                const T result = static_cast<T>(intermediate);
                self.store_value(result);

                if constexpr (chunk_config::use_fast_block_compression)
                {
                    self.mark_for_compression_if_needed_with_value(current, result);
                }
            });
    };

    if constexpr (chunk_config::use_fast_block_compression)
    {
        if (const bool need_async = self.write_with_optional_lock_info(op_lambda); need_async)
        {
            self.enqueue_async_arithmetic(
                operand, operation,
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
                            // raw-read: deferred replay, after materialisation -- flushes exclude the stealers
                            const InT<T>   current = elem_get(values, static_cast<std::size_t>(ctx->num));
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
                                                     static_cast<const void*>(current_chunk_ptr), &ctx->chunk,
                                                     ctx->num);
                                }
                                if (proxy.is_cst_val1())
                                {
                                    invariant_failed(
                                        std::source_location::current(),
                                        "Lambda async ARITH: chunk={} num={} is_cst_val1 just before store!",
                                        &ctx->chunk, ctx->num);
                                }
                                if constexpr (requires { proxy.is_cst_val2(); })
                                {
                                    if (proxy.is_cst_val2())
                                    {
                                        invariant_failed(
                                            std::source_location::current(),
                                            "Lambda async ARITH: chunk={} num={} is_cst_val2 just before store!",
                                            &ctx->chunk, ctx->num);
                                    }
                                }
                            }

                            if constexpr (chunk_config::ring_log_debug)
                            {
                                auto&      chunk_ref = ctx->chunk;
                                const auto chunk_id =
                                    static_cast<uint32_t>(::fast_alloc::address_bits(&chunk_ref) >> 4);
                                const uint32_t info =
                                    (chunk_id & 0x00FFFFFFU) | (static_cast<uint32_t>(ctx->num & 0xFF) << 24);

                                // Race detector: values pointing to cst_val here means a
                                // re-compression since create_value.
                                T* const cst1 = chunk_ref.get_cst_val1();

                                bool is_cst2 = false;
                                if constexpr (requires { chunk_ref.tab_cst_val2_holder::is_cst_val2_ptr(values); })
                                {
                                    is_cst2 = chunk_ref.tab_cst_val2_holder::is_cst_val2_ptr(values);
                                }

                                if (values == cst1 || is_cst2)
                                {
                                    ring_log(get_tid(), EventCode::values_load_is_cst, info);
                                    return; // skip the write: keep cst_val clean
                                }
                                ring_log(get_tid(), EventCode::lambda_async_arith, info);
                            }

                            if constexpr (internal::relaxed_element_stores_v<ChunkT, T>)
                            {
                                internal::store_element_relaxed(values + ctx->num, static_cast<T>(final_val));
                            }
                            else
                            {
                                elem_set(values, static_cast<std::size_t>(ctx->num), static_cast<T>(final_val));
                            }
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
template <typename Derived, typename T, typename ChunkT>
template <typename U, typename Op>
void ProxyBase<Derived, T, ChunkT>::bitwise_assign(this Derived& self, InT<U> operand, Op operation) noexcept
    requires std::is_integral_v<T> && std::is_integral_v<U>
{
    self.check_not_stale();
    if constexpr (chunk_config::stats)
    {
        increm_chunk_write<ChunkT>(operand);
    }

    // Cross-rank RMW materialisation, as in arithmetic_assign: this engine
    // reads val[num] in place, and a deferred write to THIS offset still queued
    // (possibly the refused plain `= operand` before this op) would resolve in the
    // wrong order.
    self.chunk_.reader_sync(self.num_);

    using CalcType = std::common_type_t<T, U>;

    const auto op_lambda = [&self, &operand, operation] // operand: the call's parameter, alive for the call
    {
        self.execute_inplace_operation(
            [&operand, operation](T& elem)
            {
                const CalcType intermediate = operation(static_cast<CalcType>(elem), static_cast<CalcType>(operand));
                elem                        = static_cast<T>(intermediate);
            });
    };

    if constexpr (chunk_config::use_fast_block_compression)
    {
        if (const bool need_async = self.write_with_optional_lock_info(op_lambda); need_async)
        {
            // The replay body outlives this call: capture nothing of this
            // frame (a [&] capture of `operation` would dangle at replay). The
            // operation is read from the context, which holds its own copy.
            self.enqueue_async_arithmetic(operand, operation,
                                          [](auto* ctx, auto& proxy)
                                          {
                                              proxy.execute_inplace_operation_async(
                                                  ctx,
                                                  [ctx](T& elem)
                                                  {
                                                      const CalcType intermediate =
                                                          ctx->operation(static_cast<CalcType>(elem),
                                                                         static_cast<CalcType>(ctx->operand));
                                                      elem = static_cast<T>(intermediate);
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
// zero_one_value::Proxy<T> / zero_any::Proxy<T>.
template <typename Derived, typename T, typename ChunkT>
template <typename SyncOp, typename AsyncOp>
void ProxyBase<Derived, T, ChunkT>::apply_inplace_operation(this Derived&            self,
                                                            SyncOp                   sync_op,
                                                            [[maybe_unused]] AsyncOp async_op) noexcept
{
    self.check_not_stale();
    // Cross-rank RMW materialisation, outside begin_write: see arithmetic_assign.
    self.chunk_.reader_sync(self.num_);

    // --- Synchronous fast‑path -------------------------------------------------

    if constexpr (chunk_config::use_fast_block_compression)
    {
        if (const bool need_async = self.write_with_optional_lock_info(sync_op); need_async)
        {
            using ProxyType = Derived;

            using CtxType = std::conditional_t<is_simple_proxy<ProxyType>::value,
                                               ArithmeticProxyContext<T>,              // Proxy<T>
                                               ArithmeticContext<T, ChunkT, ProxyType> // the second-value proxies
                                               >;
            auto* ctx     = pool_new_ctx<CtxType>(self, self.ptr_on_array_);

            // Rebuild a fresh proxy at replay time.
            auto write_body = [ctx, op = std::forward<AsyncOp>(async_op)]() mutable
            {
                const auto proxy = ProxyBase::template make_proxy_with_info<ProxyType>(ctx);

                op(ctx, proxy);
            };

            self.enqueue_deferred_write(ctx, std::move(write_body));
        }
    }
    else
    {
        sync_op();
    }
}

} // namespace shrink_array