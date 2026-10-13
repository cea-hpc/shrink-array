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

#include <vector>    // std::vector
#include <cstddef>   // std::size_t
#include <cstdint>   // std::int32_t, std::uint64_t, int64_t
#include <exception> // std::exception (record_enqueue_deferred)
#include <utility>   // std::move, std::forward
#include <atomic>    // std::atomic, std::memory_order
#include <memory>    // std::destroy_at
#include <mutex>     // std::call_once, std::once_flag (ensure_initialised)

#include "chunk/vendor/ankerl/unordered_dense.h" // ankerl::unordered_dense::set

#include "chunk/config.h"                            // build_config::any_check, chunk_config::ring_log_debug
#include "chunk/PMP_CHUNK.h"                         // get_thr_rank_fast
#include "chunk/internal/in_t.h"                     // in_t<T>
#include "chunk/ring_log_debug.h"                    // ring_log, get_tid, EV_*_THREAD_HAS_OPS
#include "chunk/internal/invariant_check.h"          // CHUNK_INVARIANT_FAILED
#include "chunk/internal/chunk_size_policy.h"        // t_active_chunk_size — per-op geometry pin in write_body
#include "chunk/internal/deferred_op_pool.h"         // DeferredOperationBase
#include "chunk/internal/deferred_lifecycle_check.h" // DEFERRED_LIFECYCLE sentinels
#include "chunk/internal/chunk_stats.h"              // increm_chunk_gate_* (the handshake, rare side)
#include "chunk/internal/immortal.h"                 // Immortal / LazyImmortal: statics safe for global arrays

namespace shrink_array
{

/**
 * @brief The flush-replay flag and its RAII bracket, shared by both thread models.
 *
 * @details TRUE while THIS thread is replaying deferred ops from a flush drain
 *          (flush_deferred / flush_chunk_all_ranks batch loops, and the
 *          re-chunk refill). In that context execute_operation() applies the
 *          write DIRECTLY instead of consulting the compression-active gate:
 *          both drains run under begin_flush(), which makes start() early-out,
 *          and have already drained the hard compressors, so no compressor can
 *          steal values during the replay.
 *
 *          One base for the two CompressionManagerOmpPolicy specialisations:
 *          the OpenMP policy's own drains set the flag, and the serial build
 *          still consults it from execute_operation(). Keyed by (T, ChunkType)
 *          like the policy, so each manager has its own flag.
 *
 * @warning Without it, a replayed op hitting the gate would re-enqueue itself:
 *          the queue never drains and every written element re-triggers a
 *          flush.
 *
 * @tparam T         Element type
 * @tparam ChunkType Chunk variant the manager coordinates
 */
template <typename T, typename ChunkType>
struct FlushReplayState
{
    /// @brief RAII bracket marking this thread as replaying flushed ops.
    ///        Nesting-safe: saves and restores the flag.
    class FlushReplayGuard
    {
      public:
        /// @brief Save the current flag (member initialiser) and set in_flush_replay.
        FlushReplayGuard() noexcept
        {
            in_flush_replay = true;
        }
        /// @brief Restore the saved flag.
        ~FlushReplayGuard()
        {
            in_flush_replay = saved;
        }

        FlushReplayGuard(const FlushReplayGuard&)            = delete;
        FlushReplayGuard& operator=(const FlushReplayGuard&) = delete;
        FlushReplayGuard(FlushReplayGuard&&)                 = delete;
        FlushReplayGuard& operator=(FlushReplayGuard&&)      = delete;

      private:
        bool saved = in_flush_replay;
    };

    static inline thread_local bool in_flush_replay = false; ///< this thread replays flushed ops
};

/**
 * @brief All of CompressionManager's OpenMP-only machinery.
 *
 * @details The inflight counters, per-rank flush locks, deferred-op queues and
 *          pending counters, dangerous-list tracking and the start()/finish()
 *          session verbs exist only when there are threads to coordinate (same
 *          split as ChunkFastCompPolicy, ChunkDeferredOpsPolicy,
 *          Proxy2FastCompPolicy), so the serial build carries no such state.
 *
 *          The serial specialisation is (nearly) empty: every call site of a
 *          member below sits in a branch discarded by `if constexpr` there.
 *
 * @tparam T         Element type
 * @tparam ChunkType Chunk variant the manager coordinates
 * @tparam Enabled   build_config::use_openmp; false leaves the policy empty
 */
template <typename T, typename ChunkType, bool Enabled = build_config::use_openmp>
class CompressionManagerOmpPolicy : public FlushReplayState<T, ChunkType>
{
  public:
    // Dependent base: the drains in compression_manager_inline.h name the guard
    // unqualified, so bring it into scope here.
    using FlushReplayGuard = FlushReplayState<T, ChunkType>::FlushReplayGuard; // C++20: no typename in an alias

    /**
     * @brief Base of every deferred write this manager can replay.
     * @details Adds the enqueuing rank to DeferredOperationBase, so a drain can
     *          tell own-rank from cross-rank replay.
     */
    class CmpDeferredOpBase : public DeferredOperationBase
    {
      protected:
        int thread_rank = -1; ///< rank that enqueued this op (set by DeferredOperation)

      private:
        /// @brief Replay the deferred write. Implemented by DeferredOperation.
        void write_body() override = 0;
    };

    /**
     * @brief A captured write, replayable later by a flush.
     * @tparam CTX     context type holding the frozen target metadata
     * @tparam WriteFn callable performing the write when replayed
     */
    template <typename CTX, typename WriteFn>
    class DeferredOperation : public CmpDeferredOpBase
    {
      public:
        /**
         * @brief Capture a write for later replay.
         * @param context context holding the frozen target metadata
         * @param fn_     callable performing the write
         * @param rank    enqueuing thread's rank
         */
        DeferredOperation(CTX* context, WriteFn&& fn_, int rank) : ctx(context), fn(std::move(fn_))
        {
            this->thread_rank = rank;
        }

        // Owns ctx until destroy(): a copy would destroy it twice.
        DeferredOperation(const DeferredOperation&)            = delete;
        DeferredOperation& operator=(const DeferredOperation&) = delete;
        DeferredOperation(DeferredOperation&&)                 = delete;
        DeferredOperation& operator=(DeferredOperation&&)      = delete;

        /// @brief Destroy the context object, if any, without freeing its memory, and null ctx.
        ~DeferredOperation() override
        {
            // Destroy context object (but do NOT free memory)
            if (ctx)
            {
                std::destroy_at(ctx);
                ctx = nullptr;
            }
        }

      private:
        /**
         * @brief Replay the captured write.
         * @details Under any_check, first verifies the op is still alive and has
         *          not already run -- the two ways a double flush shows up.
         */
        void write_body() override
        {
            if constexpr (build_config::any_check)
            {
                const int flusher_rank = get_thr_rank_fast();

                // (1) This op must still be alive. A poisoned magic means the op
                //     was already destroy()'d — i.e. we are flushing it twice.
                if (magic != deferred_lifecycle::op_alive_magic)
                {
                    CHUNK_INVARIANT_FAILED("[DEFERRED_LIFECYCLE] op magic corrupt/poisoned "
                                           "(op={}, magic={:#x}, executed={}, enqueue_rank={}, flusher_rank={})",
                                           static_cast<const void*>(this), magic, executed, this->thread_rank,
                                           flusher_rank);
                }

                // (2) This op must not have run before (double-flush guard).
                if (executed)
                {
                    CHUNK_INVARIANT_FAILED("[DEFERRED_LIFECYCLE] op executed twice "
                                           "(op={}, enqueue_rank={}, flusher_rank={})",
                                           static_cast<const void*>(this), this->thread_rank, flusher_rank);
                }

                // (3) The captured ctx must still be alive. A poisoned or absent
                //     ctx_alive_magic means the ctx was destroyed/recycled while
                //     this op still referenced it (the root use-after-recycle).
                if (ctx == nullptr)
                {
                    CHUNK_INVARIANT_FAILED("[DEFERRED_LIFECYCLE] op has null ctx at write_body "
                                           "(op={}, enqueue_rank={}, flusher_rank={})",
                                           static_cast<const void*>(this), this->thread_rank, flusher_rank);
                }

                if (ctx->magic != deferred_lifecycle::ctx_alive_magic)
                {
                    CHUNK_INVARIANT_FAILED("[DEFERRED_LIFECYCLE] ctx not alive at write_body "
                                           "(op={}, ctx={}, ctx_magic={:#x}, enqueue_rank={}, flusher_rank={})",
                                           static_cast<const void*>(this), static_cast<const void*>(ctx), ctx->magic,
                                           this->thread_rank, flusher_rank);
                }

                executed = true;
            }

            // ── Per-op geometry pin ───────────────────────────────────────
            // The replay may run on a thread whose ambient geometry
            // (t_active_chunk_size) now belongs to another array, and
            // create_value() sizes the buffer from it: `values[num]` would run
            // off a buffer of the wrong width. Pin the enqueue-time geometry
            // (ctx->chunk_size_frozen, as the proxy's chunk_size_frozen_) and
            // alloc key, and restore both exactly, including the "unset" 0.
            if constexpr (build_config::chunk_dynamic)
            {
                const unsigned int saved     = t_active_chunk_size;
                const unsigned int saved_key = t_alloc_key;
                if (ctx->chunk_size_frozen != 0U)
                {
                    set_thread_chunk_size(ctx->chunk_size_frozen);
                }
                set_thread_alloc_key(ctx->alloc_key_frozen); // the array's slabs, not the caller's

                fn();

                t_active_chunk_size = saved;
                t_alloc_key         = saved_key;
            }
            else
            {
                fn();
            }
        }

        /**
         * @brief Release the context and the op itself, exactly once.
         * @see destroyed for why the once-only guarantee is required
         */
        void destroy() noexcept override
        {
            // Teardown exactly once, even if two flush paths call destroy() on
            // the same op concurrently: the first caller wins.
            if (destroyed.exchange(true, std::memory_order_acq_rel))
            {
                return;
            }

            if (ctx)
            {
                std::destroy_at(ctx); // poisons ctx->magic via ~BaseArithmeticContext
                free_ctx(ctx);
                ctx = nullptr;
            }

            // DEFERRED_LIFECYCLE: poison before recycling so a later
            // double-flush observes op_dead_magic instead of op_alive_magic.
            if constexpr (build_config::any_check)
            {
                magic = deferred_lifecycle::op_dead_magic;
            }

            std::destroy_at(this);
            free_op(this); // recycle the raw memory
        }

        CTX*    ctx; ///< frozen target metadata, owned until destroy()
        WriteFn fn;  ///< the write to replay

        /// Liveness sentinel, poisoned by destroy(). A wrong value in
        /// write_body() means the op is being flushed twice.
        /// @note Unconditional on purpose (DEFERRED_LIFECYCLE): keeping the
        ///       member makes the op layout identical across every translation
        ///       unit regardless of DEBUG/CHECK_OPTIM. Only the verification in
        ///       write_body() is gated on build_config::any_check.
        std::uint64_t magic = deferred_lifecycle::op_alive_magic;

        /// Set once write_body() has run, so a second replay is detectable.
        bool executed = false;

        /// Guards destroy() so its teardown runs exactly once: the first caller
        /// wins the exchange, the rest return immediately.
        ///
        /// @warning Not cosmetic. The own-rank flush_deferred() and the
        ///          cross-rank flush_chunk_all_ranks() can both hold the SAME op
        ///          and both call destroy(). A non-atomic `if (ctx)` gate lets
        ///          both free the ctx: the block lands on the free list twice
        ///          and is later handed to two owners.
        std::atomic<bool> destroyed{false};
    };

    /// @brief One rank's set of chunks with deferred ops queued.
    using unordered_set_deferred_chunks = ankerl::unordered_dense::set<ChunkType*>;

    /**
     * @brief One rank's pending-deferred-op count, padded to its own line.
     * @details Sharded per rank: a single counter is one hot cache line under
     *          concurrent enqueues.
     */
    struct alignas(128) PaddedPendingCounter
    {
        std::atomic<int64_t> v{0}; ///< the rank's pending ops
    };

    /// @brief Nothing to build: constexpr, so that the compression_mgr of each
    ///        proxy is constant-initialised (chunk/internal/immortal.h). The
    ///        per-rank structures are sized by ensure_initialised().
    constexpr CompressionManagerOmpPolicy() = default;

    /**
     * @brief Size the per-rank tracking structures this policy owns, once.
     * @details Called when the manager of (T, ChunkType) is registered
     *          (AllTypeChunk::register_manager): before the first array of the
     *          type, and before any fan-out can reach these structures. Not by
     *          the constructor of a static: that runs at an unspecified point
     *          of the static initialisation, possibly after a global array's
     *          construction.
     */
    static void ensure_initialised()
    {
        std::call_once(init_once_,
                       []
                       {
                           init_dangerous_lists_tracking();
                           init_deferred_queues();
                       });
    }

    ~CompressionManagerOmpPolicy() = default;

    CompressionManagerOmpPolicy(const CompressionManagerOmpPolicy&)            = delete;
    CompressionManagerOmpPolicy& operator=(const CompressionManagerOmpPolicy&) = delete;

    CompressionManagerOmpPolicy(CompressionManagerOmpPolicy&&)            = delete;
    CompressionManagerOmpPolicy& operator=(CompressionManagerOmpPolicy&&) = delete;

    /**
     * @brief Open a compression window: the compressor's side of the Dekker
     *        handshake against writers and flushers.
     *
     * @details In order: enter hard_inflight_ (the compressor is PARKED, which
     *          steers writers away from its marked chunks); give up at once if a
     *          flush is in progress; dekker::heavy_barrier(); wait, bounded by
     *          CHUNK_COMPRESS_MAX_YIELDS, until no writer flag points to a
     *          marked chunk (any writer at all without the per-chunk marks);
     *          enter stealing_inflight_ (the compressor may now CAS-steal
     *          buffers); look at the flushers again and back off if one
     *          announced itself meanwhile -- the look that pairs with
     *          announce_flush(). Every give-up undoes its counters and returns
     *          false: the candidates stay queued, nothing is lost.
     *          docs/synchronisation_dekker_fr.md §5.3 explains each step.
     *
     * @tparam ProxyType proxy whose WriteCompressSyncPolicy is consulted for the
     *                   Dekker handshake against writers and flushers
     * @return true if the caller may scan and steal its marked chunks, and must
     *         then call finish(); false if it gave up
     *
     * @pre OpenMP build; the caller has marked its candidates (steal_mark())
     */
    template <typename ProxyType>
    bool start();

    /**
     * @brief Close the window opened by a start() that returned true.
     *
     * @details Leaves stealing_inflight_ first (waiters on it may go on), then
     *          hard_inflight_; the last compressor out marks its rank's
     *          candidate list safe again.
     *
     * @tparam ProxyType same proxy type the matching start() was called with
     * @note Handles a negative remaining count explicitly: see hard_inflight_'s
     *       signed-counter contract for why that state is tolerated.
     */
    template <typename ProxyType>
    void finish();

    /**
     * @brief Enqueue-side bracket around the deferred-map insert.
     * @param thr_rank rank whose map is being inserted into
     * @details Takes the same per-rank, owner-recursive shard lock the drains
     *          use: a pairwise exclusion per rank. A global back-off here
     *          convoys every enqueuer.
     * @note Public on purpose, so to_be_queued() can bracket its insert; the raw
     *       acquire/release stay private to the drain paths.
     */
    static void enqueue_lock(int thr_rank) noexcept
    {
        rank_flush_acquire(thr_rank);
    }

    /// @brief Close the bracket opened by enqueue_lock().
    /// @param thr_rank rank passed to the matching enqueue_lock()
    static void enqueue_unlock(int thr_rank) noexcept
    {
        rank_flush_release(thr_rank);
    }

    /// @brief The calling thread's "may hold deferred ops" hint, read by the
    ///        thread itself for its own decision.
    /// @return false only when the calling thread certainly holds no op
    /// @note A plain load: the flag is thread-local and every store to it is
    ///       the owner's own (mark_thread_has_ops from its enqueue, clear_ from
    ///       its own drains), so program order is all the ordering there is.
    ///       Read after every fast-path write (the backlog check).
    [[nodiscard]] static bool thread_may_have_ops_own() noexcept
    {
        return get_thread_ops_flag_tls().v;
    }

    /// @brief Count one more pending op on @p thr_rank.
    /// @param thr_rank enqueuing rank
    /// @note Release ordering publishes the preceding map insert to any reader
    ///       that observes the resulting sum as non-zero.
    static void inc_pending_ops(int thr_rank) noexcept
    {
        pending_ops_shard[thr_rank].v.fetch_add(1, std::memory_order_release);
    }

    /// @brief Pending-op count for a single rank.
    /// @param thr_rank rank to query
    /// @return that rank's count, or 0 before init_deferred_queues()
    /// @note A single acquire load -- cheaper than any_pending_ops(), which sums
    ///       every shard. Used by the backlog threshold check.
    [[nodiscard]] static int64_t pending_ops_rank(int thr_rank) noexcept
    {
        if (pending_ops_shard.empty())
        {
            return 0;
        }
        return pending_ops_shard[thr_rank].v.load(std::memory_order_acquire);
    }

    /// @brief Does any rank hold an unflushed deferred op?
    /// @return true if the summed shards are non-zero, meaning the caller must
    ///         take the rare scan path
    [[nodiscard]] static bool any_pending_ops() noexcept
    {
        if (pending_ops_shard.empty())
        {
            return false;
        }
        const int max_thr = CHUNK_PMP_RANK_BOUND;
        int64_t   total   = 0;
        for (int i = 0; i < max_thr; ++i)
        {
            total += pending_ops_shard[i].v.load(std::memory_order_acquire);
        }
        return total > 0;
    }

    /**
     * @brief Register that @p chunk now has a deferred op queued on @p thr_rank.
     * @param thr_rank enqueuing rank
     * @param chunk    chunk the op targets
     * @post The chunk joins this rank's deferred set and the rank is flagged as
     *       possibly holding ops.
     */
    static void record_enqueue_deferred(int thr_rank, ChunkType& chunk) noexcept
    {
        // Insert UNDER rank thr_rank's shard lock (the one to_be_queued()
        // already holds): flush_chunk_all_ranks() erases from this set under
        // it, and an unlocked insert races that erase on the table's storage.
        //
        // A flush replay applies its writes directly (FlushReplayGuard): an
        // enqueue from inside one is a write path ignoring in_flush_replay.
        if constexpr (build_config::any_check)
        {
            if (FlushReplayState<T, ChunkType>::in_flush_replay) [[unlikely]]
            {
                CHUNK_INVARIANT_FAILED("record_enqueue_deferred: op enqueued during a flush replay (rank {}, chunk {})",
                                       thr_rank, static_cast<const void*>(&chunk));
            }
        }
        enqueue_lock(thr_rank);
        // The insert allocates, and ankerl throws on a bucket overflow; the
        // writers calling this are noexcept: fail with the context instead of
        // a silent std::terminate.
        try
        {
            deferred_chunk_per_thread[thr_rank].insert(&chunk);
        }
        catch (const std::exception& e)
        {
            CHUNK_INVARIANT_FAILED("record_enqueue_deferred: insert failed: {} (rank {}, chunk {})", e.what(), thr_rank,
                                   static_cast<const void*>(&chunk));
        }
        enqueue_unlock(thr_rank);

        mark_thread_has_ops(thr_rank);
    }

    /**
     * @brief Replay and clear every deferred op queued on rank @p thr.
     * @param thr rank whose queue is drained
     * @return true if at least one op was replayed
     */
    [[nodiscard]] static bool flush_deferred(int thr) noexcept;

    /**
     * @brief Cross-thread targeted flush: drain one chunk's pending ops.
     *
     * @details Flushes all pending ops for a single chunk, under the per-rank
     *          flush lock so the scan of other ranks' maps is race-free. Called
     *          by wait_for_availability() when a reader on rank R touches a
     *          chunk that may have ops enqueued by another rank.
     *
     * @param chunk     chunk whose ops are drained
     * @param self_only when true, drain ONLY the caller's rank for this chunk,
     *                  costing O(1) locks; when false, walk every rank
     * @return true if any op was flushed
     * @note self_only exists because the overwhelmingly common case on the write
     *       path is "the pending op is my own, enqueued a few elements ago on
     *       this very chunk" -- hence the two-stage logic in
     *       wait_for_availability().
     */
    static bool flush_chunk_all_ranks(ChunkType* chunk, bool self_only = false) noexcept;

    /**
     * @brief WAW guard: discard a superseded same-offset op on the OWN rank.
     *
     * @details On the DIRECT write path, a deferred op still queued for the same
     *          offset is about to be superseded by the write that immediately
     *          follows; replaying it later would clobber the fresh store. This
     *          drops it WITHOUT replaying, applying the exact per-op bookkeeping
     *          of the flush erase path (dec_pending_ops(), the per-chunk
     *          pending_ops_ decrement and the tracking-structure cleanup).
     *
     * @param thr   caller's rank -- the op must be on this rank
     * @param chunk chunk holding the queued op
     * @param num   offset within the chunk; a no-op when nothing is queued there
     * @note Runs under this rank's owner-recursive shard lock, uncontended.
     * @warning Resolves only same-thread W(defer) -> W(direct). Cross-thread
     *          unsynchronised WAW stays non-deterministic, by design.
     */
    static void erase_own_rank_op(int thr, ChunkType* chunk, int num) noexcept;

    /**
     * @brief Might the calling rank hold deferred ops?
     * @return false only when it certainly holds none
     * @warning Do not replace with an .empty() check on the deferred table: an
     *          unlocked read races its erase. The per-thread flag has no false
     *          negatives and may only lag a cross-thread drain (which defers a
     *          compression, never skips a needed flush), and keeps the hot
     *          write path off the shard lock.
     */
    [[nodiscard]] static bool have_deferred_queue() noexcept
    {
        return thread_may_have_ops();
    }

    /**
     * @brief Hot write-path gate: must this write keep off @p chunk?
     * @param chunk The chunk about to be written.
     * @return with the per-chunk marks, true if a compressor's pass has this
     *         chunk marked; without them, if any hard compression is in
     *         flight, parked or stealing
     * @note With the marks: one acquire load on the chunk's header line --
     *       the line `values` was just loaded from -- and nothing global.
     *       The compressor's window waits only for the writers whose flag
     *       points to a marked chunk (any_thread_writing_on_marked), so the
     *       others need no steering. The handshake needs no stronger order
     *       from this load: the compressor's heavy barrier sits between its
     *       mark stores and its loads of the writer flags, and the writer's
     *       announce_write() precedes this load -- one side always sees the
     *       other (see announce_write()). Never seq_cst: on AArch64 that load
     *       stalls behind the preceding barrier, a large cost on the write path.
     * @note "In flight" means per (variant, T): these statics are per template
     *       instantiation, so a compression on double does not gate writes on
     *       int. The rank's own "dangerous" flag is not a gate: it decides,
     *       AFTER the write, whether this thread drives a compression
     *       (write_with_optional_lock_impl).
     */
    [[nodiscard]] static bool must_defer_write(const ChunkType& chunk) noexcept __attribute__((always_inline))
    {
        if constexpr (chunk_config::use_chunk_steal_mark)
        {
            // Acquire, paired with steal_mark()'s release: a writer that sees
            // the mark clear here is ordered before the claim, and one that
            // sees it set takes the slow path. A relaxed pair would let a
            // writer miss a mark already set.
            return chunk.steal_marked();
        }
        else
        {
            return hard_inflight_.load(std::memory_order_relaxed) > 0;
        }
    }

    /**
     * @brief Run @p fn as the slow-path compression driver.
     * @tparam F callable to run inside the driver bracket
     * @param fn the driver body, typically perform_compression()
     * @details No counter of drivers in flight: nothing would read it (the
     *          write-path gate reads hard_inflight_ or the chunk's mark), and
     *          nothing may wait on one (see hard_inflight_'s deadlock
     *          invariant). NO heavy barrier here either: the exclusion a driver
     *          needs comes from start(), inside fn, with its own announce.
     */
    template <typename F>
    static void run_slow_path(F&& fn)
    {
        increm_chunk_gate_driver<ChunkType>();
        std::forward<F>(fn)();
    }

    /// @brief Is a compressor inside the CAS-stealing window?
    /// @return true while stealing_inflight_ is positive: a compressor past the
    ///         writer gate, which may CAS-steal and free a chunk's `values`
    /// @details The predicate create_value() and the flush drains spin on.
    ///          Never hard_inflight_ (a parked compressor): see the deadlock
    ///          invariant on hard_inflight_ and stealing_inflight_.
    [[nodiscard]] static bool is_hard_compression_inflight() noexcept
    {
        return stealing_inflight_.load(std::memory_order_seq_cst) > 0;
    }

    /**
     * @brief Flag @p thr_rank's candidate list as needing a compression pass:
     *        its next write, once applied, drives perform_compression().
     * @param thr_rank rank whose list becomes dangerous
     *
     * @details A FLAG, not a counter: incremented on every write above the
     *          threshold and decremented once per compression, a counter keeps
     *          the rank "dangerous" long after its list was drained, and sends
     *          every one of its writes through run_slow_path() for a
     *          perform_compression() that finds nothing to do.
     *
     *          Own-rank state: set by the owning thread's trigger, read by its
     *          own write engine, cleared by its own perform_compression() (and
     *          by finish()). The load-then-store avoids dirtying the line on
     *          every write above the threshold.
     */
    static void mark_list_dangerous_for_thread(int thr_rank) noexcept
    {
        auto& v = dangerous_lists_per_thread[thr_rank].value;
        if (v.load(std::memory_order_relaxed) == 0)
        {
            v.store(1, std::memory_order_release);
        }
    }

    /// @brief The list has been looked at: clear the flag. Idempotent; the
    ///        trigger raises it again when the list next crosses the threshold.
    /// @param thr_rank rank whose list becomes safe again
    static void mark_list_safe_for_thread(int thr_rank) noexcept
    {
        dangerous_lists_per_thread[thr_rank].value.store(0, std::memory_order_release);
    }

    /// @brief Is @p thr_rank's candidate list currently flagged dangerous?
    /// @param thr_rank rank to query
    static bool is_dangerous_list(int thr_rank) noexcept
    {
        return dangerous_lists_per_thread[thr_rank].value.load(std::memory_order_relaxed) > 0;
    }

    /**
     * @brief Writer-side bounded wait for the gate to open on @p chunk.
     *
     * Called by the write engine when must_defer_write(chunk) is true, BEFORE
     * announce_write(): the caller holds no writer flag, so a parked
     * compressor's "no writer" window is not held shut by the wait, and the
     * compressor past that window takes no rank lock and drains nothing, so
     * nothing the caller may hold (its own rank's flush lock, see the
     * own-replay exclusion in write_with_optional_lock_impl) is on the
     * compressor's path. Bounded: past the budget the caller defers the
     * write, so every progress argument of the deferred path still holds.
     *
     * What is waited for: a parked compressor (microseconds -- the window
     * opens as soon as the writers in flight finish, and this thread is not
     * one of them) or, with the marks, the compressor's pass reaching this
     * chunk (it unmarks each chunk as it goes, a chunk at the end of a long
     * list waits the longest). Without the marks it is the whole pass, scan
     * included: a few hundred microseconds to a millisecond.
     *
     * Why wait rather than defer: a deferred write costs an enqueue, a replay
     * and the op's memory until the flush (a few % of wall time, more peak RSS).
     *
     * @note Spin discipline of the library: CHUNK_CPU_PAUSE, a yield every 1000,
     *       CHUNK_WRITE_WAIT_MAX_YIELDS yields at most (-D tunable).
     */
    static void wait_out_compressor(const ChunkType& chunk) noexcept;

  private:
    /**
     * @brief Owner-recursive mutual exclusion for one rank's deferred-op map.
     *
     * @details begin_flush() blocks compressors and same-rank writers, but not
     *          two readers on DIFFERENT ranks flushing the same chunk: both
     *          would drain and destroy the same deferred op (double free).
     *
     *          The lock protects the per-rank op-map arr[r] (an ankerl map whose
     *          erase/insert rehashes the whole backing vector) against concurrent
     *          structural mutation, so its granularity is the RANK: flushes of
     *          different ranks' maps run in parallel.
     *
     *          Owner-recursive because a drain runs write_body(), which may
     *          RE-ENQUEUE onto the same rank. @c owner is the holding rank (-1
     *          when free); @c depth is touched only by the owner (plain int).
     *
     * @warning Do not shard per CHUNK: it would forbid erasing from arr[r]
     *          (erase rehashes, racing across shards), so arr[r] would grow to
     *          O(distinct chunks). Per-rank keeps the erase and bounds memory.
     * @note Per rank, not one lock per type: flush_chunk_all_ranks() is hot on
     *       the read path, and a single lock serialises every cross-thread
     *       flush and stops scaling at high thread counts.
     */
    struct alignas(128) PaddedFlushLock
    {
        std::atomic<int> owner{-1}; ///< rank currently holding the lock, -1 if free
        int              depth{0};  ///< nested acquire count, owner-only
    };

    /// @brief One dangerous-list flag, padded to its own cache line.
    /// @note 128 bytes rather than 64, to stay safe on ARM.
    struct alignas(128) PaddedDangerousFlag
    {
        std::atomic<int> value{0}; ///< 0 or 1 -- see mark_list_dangerous_for_thread
    };

    /// @brief "This thread may hold deferred ops" hint.
    /// @note A plain bool: a thread_local, reached only through
    ///       get_thread_ops_flag_tls() by the thread that owns it -- no other
    ///       thread ever reads or writes it, so there is nothing to order.
    struct ThreadOpsFlag
    {
        bool v = false; ///< the thread may hold deferred ops
    };

    /// @brief Spin-acquire rank @p rank's flush lock (recursive for the owner).
    static void rank_flush_acquire(int rank) noexcept;
    /// @brief Release rank @p rank's flush lock, or drop one nesting level.
    static void rank_flush_release(int rank) noexcept;

    /// @brief Scoped hold of one rank's flush lock: rank_flush_acquire() at
    ///        construction, rank_flush_release() on every exit of the scope.
    class RankFlushLock
    {
      public:
        /// @brief Acquire rank @p rank's flush lock.
        explicit RankFlushLock(int rank) noexcept : rank_(rank)
        {
            rank_flush_acquire(rank_);
        }
        /// @brief Release it (or drop one nesting level).
        ~RankFlushLock()
        {
            rank_flush_release(rank_);
        }

        RankFlushLock(const RankFlushLock&)            = delete;
        RankFlushLock& operator=(const RankFlushLock&) = delete;
        RankFlushLock(RankFlushLock&&)                 = delete;
        RankFlushLock& operator=(RankFlushLock&&)      = delete;

      private:
        int rank_;
    };

    /// @brief Decrement hard_inflight_ and return the new value.
    /// @return remaining hard-inflight count (finish() reports on the last one
    ///         out, at 0); may transiently be negative (see hard_inflight_'s
    ///         signed-counter contract)
    static int decrement_inflight() noexcept
    {
        return hard_inflight_.fetch_sub(1, std::memory_order_seq_cst) - 1;
    }

    /// @brief This thread's own "may hold deferred ops" flag.
    /// @return reference to the calling thread's flag
    /// @note Thread-local, so it is reached without indexing by rank.
    static ThreadOpsFlag& get_thread_ops_flag_tls()
    {
        thread_local ThreadOpsFlag tls;
        return tls;
    }

    /// @brief Announce that this thread now holds at least one deferred op.
    /// @param thr_rank caller's rank, for the debug ring log only
    /// @note Only this thread reads it back: the drains of other threads go by
    ///       the per-rank pending counters and the chunks' pending words.
    static void mark_thread_has_ops(int thr_rank) noexcept
    {
        if constexpr (chunk_config::ring_log_debug)
        {
            ring_log(get_tid(), EV_MARK_THREAD_HAS_OPS, static_cast<uint32_t>(thr_rank));
        }
        get_thread_ops_flag_tls().v = true;
    }

    /// @brief Announce that this thread's deferred ops have all been drained.
    /// @param thr_rank caller's rank, for the debug ring log only
    static void clear_thread_has_ops(int thr_rank) noexcept
    {
        if constexpr (chunk_config::ring_log_debug)
        {
            ring_log(get_tid(), EV_CLEAR_THREAD_HAS_OPS, static_cast<uint32_t>(thr_rank));
        }
        get_thread_ops_flag_tls().v = false;
    }

    /// @brief Conservative check: might this thread hold deferred ops?
    /// @note No rank argument: the flag is thread-local, not rank-indexed.
    /// @return false only when the calling thread certainly holds none
    [[nodiscard]] static bool thread_may_have_ops() noexcept
    {
        return get_thread_ops_flag_tls().v;
    }

    /**
     * @brief Allocate the per-rank deferred-op structures.
     * @post The chunk sets, pending-op shards and flush locks are all sized to
     *       the thread capacity (CHUNK_PMP_THREAD_CAPACITY) and zero-initialised.
     */
    static void init_deferred_queues()
    {
        // Built whole, never resized: the atomics are not movable. The member
        // initialisers give the zero counts and the free (-1) locks.
        const auto max_thr        = static_cast<std::size_t>(CHUNK_PMP_THREAD_CAPACITY);
        deferred_chunk_per_thread = std::vector<unordered_set_deferred_chunks>(max_thr);
        pending_ops_shard         = std::vector<PaddedPendingCounter>(max_thr);
        // Per-rank flush lock shards (see xflush_lock_rank declaration).
        xflush_lock_rank = std::vector<PaddedFlushLock>(max_thr);
    }

    /// @brief Count @p count drained ops off @p thr_rank.
    /// @param thr_rank rank whose ops were drained
    /// @param count    number of ops drained
    static void dec_pending_ops(int thr_rank, int64_t count = 1) noexcept
    {
        const int64_t before = pending_ops_shard[thr_rank].v.fetch_sub(count, std::memory_order_release);
        // Never below zero: any_pending_ops() SUMS the ranks, so a rank at -2
        // hides two real ops on another one and the array-level flush is
        // skipped. inc_pending_ops() runs under the same shard lock as the
        // op insert, so a negative count is a double count, not a race.
        if constexpr (build_config::any_check)
        {
            if (before - count < 0) [[unlikely]]
            {
                CHUNK_INVARIANT_FAILED("dec_pending_ops: rank {} count {} -> {} (retired {}): more ops retired than "
                                       "published -- a double count",
                                       thr_rank, before, before - count, count);
            }
        }
    }

    /// @brief Enter the hard compression window (start()'s counterpart).
    /// @see hard_inflight_
    static void add_compression_active() noexcept
    {
        hard_inflight_.fetch_add(1, std::memory_order_seq_cst);
    }

    /**
     * @brief Allocate the per-rank dangerous-list counters.
     * @details One zero-initialised counter per rank of the thread capacity
     *          (CHUNK_PMP_THREAD_CAPACITY).
     */
    static void init_dangerous_lists_tracking()
    {
        // Built whole (atomics are not movable); the member initialiser gives 0.
        dangerous_lists_per_thread =
            std::vector<PaddedDangerousFlag>(static_cast<std::size_t>(CHUNK_PMP_THREAD_CAPACITY));
    }

    // ─── Unified compression state ───────────────────────────────────────

    /// Compressors between start() and finish(), parked ones included. The
    /// write-path gate reads it (must_defer_write, without the per-chunk
    /// marks): a parked compressor steers new writers away, which is what
    /// opens its "no writer" window.
    ///
    /// @warning DEADLOCK INVARIANT. Code that must let a flush-draining slow
    ///          path proceed (create_value(), the flush drain loops) waits on
    ///          stealing_inflight_ ONLY, through is_hard_compression_inflight().
    ///          Never on a count of slow-path drivers: a driver may be merely
    ///          draining flush_deferred(), blocked on an xflush lock the waiter
    ///          holds transitively. Never on this counter: see
    ///          stealing_inflight_.
    ///
    /// @warning SIGNED on purpose -- part of the contract, do not make it
    ///          unsigned. Some give-up interleavings decrement twice (start()'s
    ///          yield-to-flush and bounded-wait paths, plus finish(), which
    ///          handles `remaining < 0` explicitly). A transient -1 keeps the
    ///          gate predicate (load > 0) false; unsigned, it wraps to ~4e9 and
    ///          the gate never opens again.
    ///
    /// @note Every access is seq_cst: required by the Dekker pairing with
    ///       WriteCompressSyncPolicy's flushing/writing flags in start() and in
    ///       the flush drains.
    alignas(128) static inline std::atomic<std::int32_t> hard_inflight_{0};

    /// Compressors PAST the writer gate: Dekker established, allowed to
    /// CAS-steal `values`. A stealer depends on no writer flag, so waiters on
    /// this counter always drain.
    ///
    /// @details Parked vs stealing, two states, two counters. hard_inflight_
    ///          counts every compressor between start() and finish(),
    ///          INCLUDING one parked in start()'s bounded any_thread_writing()
    ///          wait. The write-path gate must see parked compressors -- that
    ///          opens the "no writer" window and gives compressors priority --
    ///          but create_value() and the flush drains must NOT wait on them.
    ///
    /// @warning Keep the two counters; both ways of merging them fail.
    ///          Waiting on hard_inflight_ from under a writer flag livelocks
    ///          (parked compressors wait for the writers, the writers in
    ///          create_value() wait for them, the drains wait on both).
    ///          Releasing hard while parked removes compressor priority:
    ///          compression starves and memory doubles.
    alignas(128) static inline std::atomic<std::int32_t> stealing_inflight_{0};

    /// ensure_initialised() runs the sizing once.
    static inline constinit std::once_flag init_once_{};

    /// One lock per rank; sized at init_deferred_queues() with max_thr entries.
    static inline constinit internal::Immortal<std::vector<PaddedFlushLock>> xflush_lock_rank_storage_{};
    static inline std::vector<PaddedFlushLock>& xflush_lock_rank = xflush_lock_rank_storage_.value;

    /// Per rank: "this rank's candidate list has crossed the threshold and
    /// perform_compression() has not looked at it since". Read by the rank's
    /// own write engine after each write (write_with_optional_lock_impl).
    static inline constinit internal::Immortal<std::vector<PaddedDangerousFlag>> dangerous_lists_per_thread_storage_{};
    static inline std::vector<PaddedDangerousFlag>&                              dangerous_lists_per_thread =
        dangerous_lists_per_thread_storage_.value;

    /// Per rank: the chunks with deferred ops queued on it (what
    /// flush_deferred() walks). Written under the rank's flush lock.
    static inline constinit internal::Immortal<std::vector<unordered_set_deferred_chunks>>
                                                              deferred_chunk_per_thread_storage_{};
    static inline std::vector<unordered_set_deferred_chunks>& deferred_chunk_per_thread =
        deferred_chunk_per_thread_storage_.value;

    /// Per-rank pending-op counts, the mechanism that makes cross-thread ops
    /// visible to a reader running on a different rank than the writer.
    /// @details Incremented at enqueue BEFORE the map insert is published, and
    ///          decremented at flush AFTER the op is drained. A reader sums the
    ///          shards: zero means no chunk anywhere has a pending op, so the
    ///          sync can early-out without scanning.
    static inline constinit internal::Immortal<std::vector<PaddedPendingCounter>> pending_ops_shard_storage_{};
    static inline std::vector<PaddedPendingCounter>& pending_ops_shard = pending_ops_shard_storage_.value;
};

/**
 * @brief Serial specialisation: no threads to coordinate, so nothing to hold.
 * @tparam T         Element type
 * @tparam ChunkType Chunk variant the manager coordinates
 */
template <typename T, typename ChunkType>
class CompressionManagerOmpPolicy<T, ChunkType, false> : public FlushReplayState<T, ChunkType>
{
  public:
    /**
     * @brief Serial builds queue nothing, so no rank can hold deferred ops.
     *
     * @details Mirrors the OpenMP policy's have_deferred_queue(). The write path
     *          calls it unconditionally -- see Proxy::apply_compound_operation,
     *          where the guard reads
     *          `v_old != zero || CompressionManagerType::have_deferred_queue()`.
     *          That call sits in a plain `else`, not a discarded `if constexpr`
     *          branch, so the name must exist in EVERY configuration: clang and
     *          icpx resolve this non-dependent name at phase 1 (gcc does not).
     *
     * @return always false: nothing is ever deferred without OpenMP.
     */
    [[nodiscard]] static constexpr bool have_deferred_queue() noexcept
    {
        return false;
    }

    /// @brief Nothing to size serially.
    static void ensure_initialised() noexcept {}
};

/**
 * @brief Manages compression lifecycle and thread synchronization
 *
 * @details Coordinates compression against concurrent writers and flushes. Most
 *          of that coordination only exists under OpenMP and lives in
 *          CompressionManagerOmpPolicy, which this class inherits; what remains
 *          here is what the serial build also needs: the pure predicate
 *          should_mark_for_compression(), and the flush-replay guard consulted
 *          by execute_operation() in both modes.
 *
 * @tparam T         Element type
 * @tparam ChunkType Chunk variant the manager coordinates
 * @note Thread safety: All methods are thread-safe
 */
template <typename T, typename ChunkType>
struct CompressionManager : public CompressionManagerOmpPolicy<T, ChunkType, build_config::use_openmp>
{
    CompressionManager()  = default;
    ~CompressionManager() = default;

    CompressionManager(const CompressionManager&)            = delete;
    CompressionManager& operator=(const CompressionManager&) = delete;
    CompressionManager(CompressionManager&&)                 = delete;
    CompressionManager& operator=(CompressionManager&&)      = delete;

    /**
     * @brief Checks if a value change should trigger compression consideration
     * @param old_value Previous value
     * @param new_value New value
     * @param second_value Current second value for comparison
     * @param first_value First constant value for comparison
     * @return true if the chunk should be marked for compression
     */
    [[nodiscard]] static bool should_mark_for_compression(in_t<T> old_value,
                                                          in_t<T> new_value,
                                                          in_t<T> second_value,
                                                          in_t<T> first_value) noexcept
    {
        return (old_value != first_value && new_value == first_value) ||
               (old_value != second_value && new_value == second_value);
    }
};

} // namespace shrink_array

#include "chunk/internal/compression_manager_inline.h" // the template bodies
