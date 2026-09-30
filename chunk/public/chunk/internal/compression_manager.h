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

#include <vector>
#include <cstdint>
#include <exception> // std::exception (record_enqueue_deferred)
#include <utility>
#include <atomic>
#include <memory> // std::destroy_at

#include "ankerl/unordered_dense.h"

#include "chunk/config.h"
#include "chunk/PMP_CHUNK.h" // get_thr_rank_fast
#include "chunk/internal/in_t.h"
#include "chunk/ring_log_debug.h"
#include "chunk/internal/invariant_check.h"
#include "chunk/internal/chunk_size_policy.h" // t_active_chunk_size — per-op geometry pin in write_body
#include "chunk/internal/deferred_op_pool.h"
#include "chunk/internal/deferred_lifecycle_check.h" // DEFERRED_LIFECYCLE sentinels
#include "chunk/internal/chunk_stats.h"              // increm_chunk_gate_* (the handshake, rare side)

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
 * @warning Without this, a replayed op hitting a quasi-permanently-true gate
 *          RE-ENQUEUED itself: pending_ops_ never drained, every written
 *          element re-triggered a flush, and the flush machinery recycled ops
 *          forever -- the residual x86 collapse, with TotalView showing
 *          wait_for_availability -> write_body -> to_be_queued ->
 *          inc_pending_ops stacks.
 *
 * @tparam T         Element type
 * @tparam ChunkType Chunk variant the manager coordinates
 */
template <typename T, typename ChunkType>
struct FlushReplayState
{
    static inline thread_local bool in_flush_replay = false;

    /// @brief RAII bracket marking this thread as replaying flushed ops.
    ///        Nesting-safe: saves and restores the flag.
    struct FlushReplayGuard
    {
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
};

/**
 * @brief All of CompressionManager's OpenMP-only machinery.
 *
 * @details Everything the compression lifecycle needs to coordinate against
 *          concurrent writers and flushes -- the three inflight counters, the
 *          per-rank flush locks, the deferred-op queues and their pending
 *          counters, the dangerous-list tracking and the start()/finish()
 *          session verbs -- exists only when there are threads to coordinate.
 *          Splitting it out on the same axis the rest of the codebase uses
 *          (ChunkFastCompPolicy, ChunkDeferredOpsPolicy, Proxy2FastCompPolicy)
 *          keeps the serial build free of state it can never use, and removes
 *          the last preprocessor conditional from this header.
 *
 *          The serial specialisation is empty. Every call site of a member
 *          below sits in a branch discarded by `if constexpr` in serial builds,
 *          so the names are never looked up there.
 *
 * @tparam T         Element type
 * @tparam ChunkType Chunk variant the manager coordinates
 * @tparam Enabled   build_config::use_openmp; false leaves the policy empty
 */
template <typename T, typename ChunkType, bool Enabled = build_config::use_openmp>
struct CompressionManagerOmpPolicy : FlushReplayState<T, ChunkType>
{
    // Dependent base: the drains in compression_manager_inline.h name the guard
    // unqualified, so bring it into scope here.
    using FlushReplayGuard = FlushReplayState<T, ChunkType>::FlushReplayGuard; // C++20: no typename in an alias

  private:
    // ─── Unified compression state ───────────────────────────────────────

    /// Threads between start() and finish(): the exact window in which a
    /// chunk's `values` may be CAS-stolen and freed by the compressor.
    ///
    /// @warning DEADLOCK INVARIANT (July '26). Waiters that must let a
    ///          flush-draining slow path proceed wait on THIS counter ONLY. A
    ///          thread contributing to driver_inflight_ while hard_inflight_ is
    ///          zero is merely draining flush_deferred() and steals nothing;
    ///          waiting on it while it blocks on an xflush lock held
    ///          transitively by the waiter deadlocks. The flush drain loops and
    ///          is_hard_compression_inflight() therefore consult hard_inflight_
    ///          and MUST NEVER consult driver_inflight_.
    ///
    /// @warning SIGNED on purpose -- part of the contract, do not "clean up" to
    ///          unsigned again. Some give-up interleavings decrement twice
    ///          (start()'s yield-to-flush and bounded-wait paths, plus
    ///          finish(), which has an explicit `remaining < 0` handler
    ///          precisely because this is a known, tolerated state). With int, a
    ///          transient -1 keeps the drain predicate (load > 0) false and
    ///          everything proceeds. With uint32_t the same double-decrement
    ///          wrapped to about 4e9, is_hard_compression_inflight() stayed true
    ///          forever, and EVERY flush drain deadlocked in its spin -- all 64
    ///          threads parked in stage-1 flush_chunk_all_ranks(), Aug '26.
    ///
    /// @note Every access is seq_cst: required by the Dekker pairing with
    ///       WriteCompressSyncPolicy's flushing/writing flags in start() and in
    ///       the flush drains.
    alignas(128) static inline std::atomic<std::int32_t> hard_inflight_{0};

    /// Threads inside the slow-path compression driver
    /// (run_slow_path{perform_compression}) -- a superset of the hard window on
    /// the proxy path.
    ///
    /// @note Accounting only since Sept '26: the write-path gate reads
    ///       hard_inflight_ alone (must_defer_write). This counter started
    ///       life as global_slow_path_inflight (Feb '26), when the slow path
    ///       took a lock and every writer had to queue behind it; once the
    ///       slow path became "defer the write", steering the other writers
    ///       for the driver's whole run only manufactured deferred ops -- and
    ///       the flushes, and the yielded start()s, that came with them. It
    ///       feeds the CHUNK_STATS gate driver counter and nothing else.
    /// @warning Never wait on this counter; see hard_inflight_'s deadlock
    ///          invariant.
    /// @warning LAYOUT (x86 2-socket regression, Aug '26). This counter and
    ///          hard_inflight_ were briefly PACKED into one 64-bit word.
    ///          Semantically identical, but it concentrated 4 seq_cst RMW per
    ///          compression -- driver enter/exit plus hard enter/exit -- onto
    ///          the single cache line every write-path gate reads. Cross-socket
    ///          ping-pong collapsed the x86 init phase (93 s at 128 threads vs
    ///          32 s at 64; single-socket ARM was unaffected). They must stay on
    ///          TWO dedicated cache lines.
    alignas(128) static inline std::atomic<std::int32_t> driver_inflight_{0};

    /// Compressors PAST the writer gate: Dekker established, allowed to
    /// CAS-steal `values`. A stealer depends on no writer flag, so waiters on
    /// this counter always drain.
    ///
    /// @details Parked vs stealing, two states, two counters (Aug '26).
    ///          hard_inflight_ counts every compressor between start() and
    ///          finish(), INCLUDING one parked in start()'s bounded
    ///          any_thread_writing() wait. The write-path gate must see parked
    ///          compressors -- that is what opens the "no writer" window and
    ///          gives compressors priority -- but create_value() and the flush
    ///          drains must NOT wait on them.
    ///
    /// @warning Both ways of collapsing this into one counter have already
    ///          failed. Waiting on hard_inflight_ from under a writer flag
    ///          closes a livelock triangle: N parked holders, writers stuck in
    ///          create_value() holding per_thread_writing, flush drains stuck on
    ///          hard > 0. Conversely, releasing hard while parked removed
    ///          compressor priority entirely: compression starved, memory
    ///          doubled (23.8 GB vs 11.9) and the driver spun at 40%.
    alignas(128) static inline std::atomic<std::int32_t> stealing_inflight_{0};


    /**
     * @brief Owner-recursive mutual exclusion for one rank's deferred-op map.
     *
     * @details Cross-thread flush needs exclusion that begin_flush() does not
     *          provide: it blocks compressors and same-rank writers, but two
     *          readers on DIFFERENT ranks flushing the same chunk do not exclude
     *          each other, and would both drain and destroy the same deferred op
     *          -- a double-free.
     *
     *          The lock's only real job is protecting the per-rank op-map
     *          arr[r], an ankerl map whose erase/insert rehashes the whole
     *          backing vector, against concurrent structural mutation. Its
     *          natural granularity is therefore the RANK: flushes touching
     *          different ranks' maps run in parallel, while two flushers
     *          draining the same arr[r][chunk] still serialize.
     *
     *          Owner-recursive because a drain executes write_body(), which may
     *          RE-ENQUEUE onto the same rank and take this lock again. @c owner
     *          holds the holding rank (-1 when free); @c depth counts nested
     *          acquires and is only ever touched by the owner, so a plain int
     *          suffices.
     *
     * @warning Sharding per CHUNK instead of per rank would require never
     *          erasing from arr[r] -- erase rehashes and would race across
     *          shards -- letting arr[r] grow to O(distinct chunks) and blowing
     *          up memory. Per-rank keeps the erase and bounds memory.
     * @note Replaces a single global flag that serialized ALL cross-thread
     *       flushes for this type; since flush_chunk_all_ranks() is hot on the
     *       read path, that was the 128-thread scalability collapse.
     */
    struct alignas(128) PaddedFlushLock
    {
        std::atomic<int> owner{-1}; ///< rank currently holding the lock, -1 if free
        int              depth{0};  ///< nested acquire count, owner-only
    };
    /// One lock per rank; sized at init_deferred_queues() with max_thr entries.
    static inline std::vector<PaddedFlushLock> xflush_lock_rank;

    /// @brief Spin-acquire rank @p rank's flush lock (recursive for the owner).
    static void rank_flush_acquire(int rank) noexcept;
    /// @brief Release rank @p rank's flush lock, or drop one nesting level.
    static void rank_flush_release(int rank) noexcept;

  public:
    /**
     * @brief Enqueue-side bracket around the deferred-map insert.
     * @param thr_rank rank whose map is being inserted into
     * @details Takes the same per-rank, owner-recursive shard lock the drains
     *          use. This pairwise exclusion REPLACES the former global Dekker
     *          back-off on num_thread_flushing, which produced the x86 convoy.
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

  private:
    /// @brief One dangerous-list flag, padded to its own cache line.
    /// @note 128 bytes rather than 64, to stay safe on ARM.
    struct alignas(128) PaddedDangerousFlag
    {
        std::atomic<int> value{0}; ///< 0 or 1 -- see mark_list_dangerous_for_thread
    };

    /// Per rank: "this rank's candidate list has crossed the threshold and
    /// perform_compression() has not looked at it since". Read by the rank's
    /// own write engine after each write (write_with_optional_lock_impl).
    static inline std::vector<PaddedDangerousFlag> dangerous_lists_per_thread;

    /// @brief "This rank may hold deferred ops" hint, padded to its own line.
    struct alignas(128) ThreadOpsFlag
    {
        std::atomic<bool> v{false};
    };

  private:
    static constexpr bool verbose = false;

  public:
    /**
     * @brief Base of every deferred write this manager can replay.
     * @details Adds the enqueuing rank to DeferredOperationBase, so a drain can
     *          tell own-rank from cross-rank replay.
     */
    struct CmpDeferredOpBase : DeferredOperationBase
    {
      protected:
        int thread_rank = -1; ///< rank that enqueued this op (set by DeferredOperation)

      private:
        /// @brief Replay the deferred write. Implemented by DeferredOperation.
        void write_body() override = 0;

      public:
        ~CmpDeferredOpBase() override = default;
    };

    /**
     * @brief A captured write, replayable later by a flush.
     * @tparam CTX     context type holding the frozen target metadata
     * @tparam WriteFn callable performing the write when replayed
     */
    template <typename CTX, typename WriteFn>
    struct DeferredOperation : CmpDeferredOpBase
    {
      private:
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
        /// @warning Not cosmetic. Two flush paths -- own-rank flush_deferred()
        ///          and cross-rank flush_chunk_all_ranks() -- can hold an op_ptr
        ///          to the SAME op and both call destroy(). The previous
        ///          `if (ctx)` gate is not atomic, so both saw ctx != nullptr and
        ///          both free_ctx()'d it: the ctx block landed on the free-list
        ///          twice, allocate_raw() handed the same live ctx to two owners
        ///          (POOL_LEDGER reported ctx_alloc DOUBLE-alloc / SLABS_SAME),
        ///          and their overwrite corrupted ptr_on_array -- surfacing as
        ///          "Key not found" and, before the arena, vtable corruption.
        std::atomic<bool> destroyed{false};

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
                    INVARIANT_FAILED("[DEFERRED_LIFECYCLE] op magic corrupt/poisoned "
                                     "(op={}, magic={:#x}, executed={}, enqueue_rank={}, flusher_rank={})",
                                     static_cast<const void*>(this), magic, executed, this->thread_rank, flusher_rank);

                // (2) This op must not have run before (double-flush guard).
                if (executed)
                    INVARIANT_FAILED("[DEFERRED_LIFECYCLE] op executed twice "
                                     "(op={}, enqueue_rank={}, flusher_rank={})",
                                     static_cast<const void*>(this), this->thread_rank, flusher_rank);

                // (3) The captured ctx must still be alive. A poisoned or absent
                //     ctx_alive_magic means the ctx was destroyed/recycled while
                //     this op still referenced it (the root use-after-recycle).
                if (ctx == nullptr)
                    INVARIANT_FAILED("[DEFERRED_LIFECYCLE] op has null ctx at write_body "
                                     "(op={}, enqueue_rank={}, flusher_rank={})",
                                     static_cast<const void*>(this), this->thread_rank, flusher_rank);

                if (ctx->magic != deferred_lifecycle::ctx_alive_magic)
                    INVARIANT_FAILED("[DEFERRED_LIFECYCLE] ctx not alive at write_body "
                                     "(op={}, ctx={}, ctx_magic={:#x}, enqueue_rank={}, flusher_rank={})",
                                     static_cast<const void*>(this), static_cast<const void*>(ctx), ctx->magic,
                                     this->thread_rank, flusher_rank);

                executed = true;
            }

            // ── Per-op geometry pin ───────────────────────────────────────
            // The replay can run long after the enqueue, on a thread whose
            // ambient geometry meanwhile moved to another array's (any write
            // path sets t_active_chunk_size from the array it is serving).
            // The replayed body allocates through create_value(), and
            // allocate_chunk_internal() takes its element count from
            // alloc_chunk_count() -- the AMBIENT size. Replaying under the wrong ambient therefore
            // hands a chunk a buffer of the wrong width, and the `values[num]`
            // right after runs off it: ASan heap-buffer-overflow, 0 bytes after
            // a 512-byte region (64 doubles) reached at an offset valid for the
            // array's real 128-wide geometry. Silent under the fast allocator,
            // which absorbs it in the slab; a crash under std-alloc.
            //
            // ctx->chunk_size_frozen is that enqueue-time geometry -- the same
            // value make_proxy_with_info() uses to arm the proxy's
            // chunk_size_frozen_, which already protects chunk_size() on this
            // path (fw14). The allocation was never covered. Same per-entry pin
            // as Chunk2A's compression sweep, and restored exactly, including
            // the "unset, use global" 0.
            if constexpr (build_config::chunk_dynamic)
            {
                const unsigned int saved     = t_active_chunk_size;
                const unsigned int saved_key = t_alloc_key;
                if (ctx->chunk_size_frozen != 0U)
                    set_thread_chunk_size(ctx->chunk_size_frozen);
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
            // FIX (ctx double-recycle): run teardown exactly once even if two
            // flush paths call destroy() on the same op concurrently. The first
            // caller wins; the rest return without touching ctx/this again.
            if (destroyed.exchange(true, std::memory_order_acq_rel))
                return;

            // Destroy the context object if present
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

            // Destroy this DeferredOperation object
            std::destroy_at(this);

            // Recycle the raw memory of the operation
            free_op(this);
        }

      public:
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
    };

    using unordered_set_deferred_chunks = ankerl::unordered_dense::set<ChunkType*>;

  private:
    static inline std::vector<unordered_set_deferred_chunks> deferred_chunk_per_thread;

  public:
    /// @brief Allocate the per-rank tracking structures this policy owns.
    CompressionManagerOmpPolicy()
    {
        init_dangerous_lists_tracking();

        init_deferred_queues();
    };

    ~CompressionManagerOmpPolicy() = default;

    CompressionManagerOmpPolicy(const CompressionManagerOmpPolicy&)            = delete;
    CompressionManagerOmpPolicy& operator=(const CompressionManagerOmpPolicy&) = delete;

    CompressionManagerOmpPolicy(CompressionManagerOmpPolicy&&)            = delete;
    CompressionManagerOmpPolicy& operator=(CompressionManagerOmpPolicy&&) = delete;

  private:
    // Returns the number of hard compressors still in flight after this one
    // left (finish() reports on the last one out, remaining == 0).
    /// @brief Decrement hard_inflight_ and return the new value.
    /// @return remaining hard-inflight count; may transiently be negative
    ///         (see hard_inflight_'s signed-counter contract)
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
    /// @note seq_cst: the flag must be visible before a drain decides to skip
    ///       this rank.
    static void mark_thread_has_ops(int thr_rank) noexcept
    {
        if constexpr (chunk_config::ring_log_debug)
        {
            ring_log(get_tid(), EV_MARK_THREAD_HAS_OPS, static_cast<uint32_t>(thr_rank));
        }
        get_thread_ops_flag_tls().v.store(true, std::memory_order_seq_cst);
    }

    /// @brief Announce that this thread's deferred ops have all been drained.
    /// @param thr_rank caller's rank, for the debug ring log only
    static void clear_thread_has_ops(int thr_rank) noexcept
    {
        if constexpr (chunk_config::ring_log_debug)
        {
            ring_log(get_tid(), EV_CLEAR_THREAD_HAS_OPS, static_cast<uint32_t>(thr_rank));
        }
        get_thread_ops_flag_tls().v.store(false, std::memory_order_release);
    }

    /// @brief Conservative check: might this thread hold deferred ops?
    /// @note No rank argument: the flag is thread-local, not rank-indexed.
    /// @return false only when the calling thread certainly holds none
    [[nodiscard]] static bool thread_may_have_ops() noexcept
    {
        return get_thread_ops_flag_tls().v.load(std::memory_order_acquire);
    }

  public:
    /// @brief The same hint, read by its owner for its own decision.
    /// @return false only when the calling thread certainly holds no op
    /// @note Relaxed: the flag is thread-local and every store to it is the
    ///       owner's own (mark_thread_has_ops from its enqueue, clear_ from
    ///       its own drains), so program order is all the ordering there is.
    ///       Read after every fast-path write (the P3 backlog check): the
    ///       acquire form was an `ldaprb` on AArch64, and the loads behind it
    ///       carried most of write_with_optional_lock_impl's samples.
    [[nodiscard]] static bool thread_may_have_ops_own() noexcept
    {
        return get_thread_ops_flag_tls().v.load(std::memory_order_relaxed);
    }

  private:
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

  public:
    /**
     * @brief One rank's pending-deferred-op count, padded to its own line.
     * @details Sharded per rank rather than a single counter, to avoid one hot
     *          cache line under 64-thread enqueue storms.
     */
    struct alignas(128) PaddedPendingCounter
    {
        std::atomic<int64_t> v{0};
    };

  private:
    /// Per-rank pending-op counts, the mechanism that makes cross-thread ops
    /// visible to a reader running on a different rank than the writer.
    /// @details Incremented at enqueue BEFORE the map insert is published, and
    ///          decremented at flush AFTER the op is drained. A reader sums the
    ///          shards: zero means no chunk anywhere has a pending op, so the
    ///          sync can early-out without scanning.
    static inline std::vector<PaddedPendingCounter> pending_ops_shard;

  public:
    /// @brief Count one more pending op on @p thr_rank.
    /// @param thr_rank enqueuing rank
    /// @note Release ordering publishes the preceding map insert to any reader
    ///       that observes the resulting sum as non-zero.
    static void inc_pending_ops(int thr_rank) noexcept
    {
        // release: publishes prior map insert to any acquiring reader that
        // observes the resulting sum > 0.
        pending_ops_shard[thr_rank].v.fetch_add(1, std::memory_order_release);
    }

  private:
    /// @brief Count @p count drained ops off @p thr_rank.
    /// @param thr_rank rank whose ops were drained
    /// @param count    number of ops drained
    static void dec_pending_ops(int thr_rank, int64_t count = 1) noexcept
    {
        pending_ops_shard[thr_rank].v.fetch_sub(count, std::memory_order_release);
    }

  public:
    /// @brief Pending-op count for a single rank.
    /// @param thr_rank rank to query
    /// @return that rank's count, or 0 before init_deferred_queues()
    /// @note A single acquire load -- cheaper than any_pending_ops(), which sums
    ///       every shard. Used by the P3 threshold check.
    [[nodiscard]] static int64_t pending_ops_rank(int thr_rank) noexcept
    {
        if (pending_ops_shard.empty())
            return 0;
        return pending_ops_shard[thr_rank].v.load(std::memory_order_acquire);
    }

    /// @brief Does any rank hold an unflushed deferred op?
    /// @return true if the summed shards are non-zero, meaning the caller must
    ///         take the rare scan path
    [[nodiscard]] static bool any_pending_ops() noexcept
    {
        // acquire sum across shards. A non-zero result means at least one
        // rank has an unflushed op; we then take the (rare) scan path.
        if (pending_ops_shard.empty())
            return false;
        const int max_thr = CHUNK_PMP_RANK_BOUND;
        int64_t   total   = 0;
        for (int i = 0; i < max_thr; ++i)
            total += pending_ops_shard[i].v.load(std::memory_order_acquire);
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
        // Record the chunk pointer UNDER rank thr_rank's shard lock. The
        // cross-rank flush (flush_chunk_all_ranks) erases from this same set
        // while holding rank r's lock (rank_flush_acquire(r) -> ...
        // deferred_chunk_per_thread[r].erase). Without taking the SAME lock
        // here, that erase races this insert on the ankerl table's internal
        // storage (empty()/emplace_back on its vector) -> corrupted table ->
        // a later free of a mangled Chunk2A* ("corrupted double-linked list").
        // TSan-confirmed data race (erase T_flush vs emplace T_owner). The lock
        // is the same owner-recursive shard lock to_be_queued already uses for
        // the op-map insert, so bracketing the chunk-set insert with it closes
        // the last unsynchronised access to deferred_chunk_per_thread.
        // A flush replay applies its writes directly (FlushReplayGuard): an
        // enqueue from inside one means a write path that ignores
        // in_flush_replay. flush_deferred() now survives it (it walks a
        // snapshot and keeps the chunk), but the replay was meant to empty the
        // queue, not refill it.
        if constexpr (build_config::any_check)
        {
            if (FlushReplayState<T, ChunkType>::in_flush_replay) [[unlikely]]
                INVARIANT_FAILED("record_enqueue_deferred: op enqueued during a flush replay (rank {}, chunk {})",
                                 thr_rank, static_cast<const void*>(&chunk));
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
            INVARIANT_FAILED("record_enqueue_deferred: insert failed: {} (rank {}, chunk {})", e.what(), thr_rank,
                             static_cast<const void*>(&chunk));
        }
        enqueue_unlock(thr_rank);

        mark_thread_has_ops(thr_rank);
    }

    /**
     * @brief Replay and clear every deferred op queued on rank @p thr.
     * @param thr rank whose queue is drained
     * @param wait_for_compression currently ignored by the implementation
     * @return true if at least one op was replayed
     */
    [[nodiscard]] static bool flush_deferred(int thr, bool wait_for_compression = true) noexcept;

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
     * @warning Deliberately NOT an .empty() check on the deferred table: that
     *          unlocked read raced its do_erase (TSan-confirmed). The lock-free
     *          per-thread flag used instead has no false negatives -- it never
     *          misses a real deferred op -- and may only lag a cross-thread
     *          drain, which defers a compression rather than skipping a needed
     *          flush. It also keeps the hot write path off the shard lock.
     */
    [[nodiscard]] static bool have_deferred_queue() noexcept
    {
        // Was: !deferred_chunk_per_thread[thr].empty() -- an UNLOCKED read of the
        // ankerl table's internal size. flush_chunk_all_ranks erases from
        // shard[thr] under rank_flush_acquire(thr); this unlocked .empty() raced
        // its do_erase (TSan-confirmed: .empty() read vs do_erase write on the
        // deferred table). Use the lock-free per-thread ops flag instead: set on
        // enqueue, cleared when the rank drains, so it has no false negatives
        // (never misses a real deferred op). It may lag a cross-thread drain
        // (stay true briefly after), which only defers a compression, never skips
        // a needed flush. Both call sites pass the caller's own rank, matching the
        // flag's TLS/self-rank nature -- and this stays off the shard lock on the
        // hot write path.
        return thread_may_have_ops();
    }

    /**
     * @brief Hot write-path gate: must this write keep off @p chunk?
     * @param chunk The chunk about to be written.
     * @return with the per-chunk marks, true if a compressor's pass has this
     *         chunk marked; without them, if any hard compression is in
     *         flight, parked or stealing
     * @note With the marks: one relaxed load on the chunk's header line --
     *       the line `values` was just loaded from -- and nothing global.
     *       The compressor's window waits only for the writers whose flag
     *       points to a marked chunk (any_thread_writing_on_marked), so the
     *       others need no steering. The load carries no ordering: the
     *       compressor's heavy barrier sits between its mark stores and its
     *       loads of the writer flags, and the writer's announce_write()
     *       precedes this load -- one side always sees the other (see
     *       announce_write()). The seq_cst the global load had cost an
     *       `ldar` on AArch64, stalled behind the preceding barrier: ~70% of
     *       write_with_optional_lock_impl in the profile.
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
            // sees it set takes the slow path. The relaxed pair this replaced
            // let a writer miss a mark that had already been set.
            return chunk.steal_marked();
        }
        else
        {
            return hard_inflight_.load(std::memory_order_relaxed) > 0;
        }
    }

  private:
    /// @brief Enter the hard compression window (start()'s counterpart).
    /// @see hard_inflight_
    static void add_compression_active() noexcept
    {
        hard_inflight_.fetch_add(1, std::memory_order_seq_cst);
    }

  public:
    template <typename F>
    /**
     * @brief Run @p fn as the slow-path compression driver.
     * @tparam F callable to run inside the driver bracket
     * @param fn the driver body, typically perform_compression()
     * @details Brackets @p fn with driver_inflight_. The counter lives here, not
     *          in WriteCompressSyncPolicy, because it is compression-lifecycle
     *          state rather than write/flush-handshake state.
     * @note seq_cst supersedes the former acq_rel: this is the cold path, so the
     *       cost is irrelevant.
     */
    static void run_slow_path(F&& fn)
    {
        increm_chunk_gate_driver<ChunkType>();
        driver_inflight_.fetch_add(1, std::memory_order_seq_cst);
        // NO heavy barrier here. driver_inflight_ is accounting; the exclusion
        // a driver needs is established by start(), inside f, with its own
        // announce. (When it still steered the other writers this was also
        // not rare: a dangerous rank sent every write through here, 483 000/s
        // at 4 threads on the perf bench, and a membarrier() per call made
        // the bench 60-100% slower.)
        std::forward<F>(fn)();
        driver_inflight_.fetch_sub(1, std::memory_order_seq_cst);
    }

    // True ONLY while an actual compression is in flight: make_block_compression
    // has called start() (hard_count += 1) and not yet finished — the exact
    // window in which a chunk's `values` may be CAS-stolen and freed by the
    // compressor.
    //
    // Deliberately consults hard_inflight_ and NOT driver_inflight_ (unlike
    // the former is_compression_active(), which counted both and is gone). A
    // slow-path thread that is merely draining
    // flush_deferred contributes to driver_count with hard_count == 0: it is
    // not stealing anything. Waiting on it re-creates the flush-vs-compression
    // priority inversion that start() already resolves (start() yields to any
    // in-progress flush by decrementing the hard count and returning false),
    // and it DEADLOCKS: the flush-draining thread is blocked on xflush_lock
    // while the waiter (running inside a flush's write_body, or holding
    // begin_write) transitively holds what that thread needs. See
    // create_value() and the deadlock invariant above.
    // The predicate create_value and the flush drains spin on: compressors
    // ACTUALLY STEALING (past the writer gate) — never the driver count
    // (July invariant) and never parked compressors (Parked-vs-Stealing
    // split above). The name is kept so the three wait sites are untouched.
    /// @brief Is a compressor inside the CAS-stealing window?
    /// @return true while hard_inflight_ is positive
    /// @warning Consults hard_inflight_ only, never driver_inflight_ -- see the
    ///          deadlock invariant on hard_inflight_.
    [[nodiscard]] static bool is_hard_compression_inflight() noexcept
    {
        return stealing_inflight_.load(std::memory_order_seq_cst) > 0;
    }

  private:
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

  public:
    /**
     * @brief Flag @p thr_rank's candidate list as needing a compression pass:
     *        its next write, once applied, drives perform_compression().
     * @param thr_rank rank whose list becomes dangerous
     *
     * @details A FLAG, not a counter. It was a counter: +1 from
     *          trigger_compression_if_needed() on EVERY write while the list
     *          was above the threshold, -1 once per compression that ran
     *          (perform_compression, finish). Thousands of increments per
     *          decrement, so a rank stayed "dangerous" long after its list had
     *          been drained -- by the between-phase region, mostly -- and every
     *          write of that rank went through run_slow_path() for a
     *          perform_compression() that found nothing to do: 463 000
     *          slow-path writes per second at 64 threads on the perf bench
     *          (the CHUNK_STATS gate driver counter), with the flush and
     *          start() attempts each of them makes on the way.
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
            v.store(1, std::memory_order_release);
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
     * @brief Starts a compression session
     *
     * @details
     * Performs the following operations:
     * 1. Acquires the global compression lock
     * 2. Switches the epoch to invalidate old counters
     * 3. Waits for all threads to exit the fast-path
     * 4. Activates slow-path for new operations
     *
     * Every 100th attempt forces a full wait to guarantee progress.
     *
     * @tparam ProxyType proxy whose WriteCompressSyncPolicy is consulted for the
     *                   Dekker handshake against writers and flushers
     * @return true if compression can start immediately
     * @return false if threads are still active in fast-path
     *
     * @pre OpenMP must be enabled
     * @post Slow-path is activated if returns true
     *
     * @note Thread-safe, but only one thread should call this
     */
    template <typename ProxyType>
    bool start();

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
     * included: ~300 us at 6 threads on x86, ~1 ms at 64 on AArch64.
     *
     * Why wait rather than defer: a deferred write is an enqueue plus a
     * replay (~200 ns each, plus the op's memory until the flush); once
     * compressions actually ran, deferring every write behind them meant
     * 16 M deferred writes per run at 8 threads (3.6% of the writes), +6%
     * wall time and +16% peak RSS.
     *
     * @note Spin discipline of the library: CPU_PAUSE, a yield every 1000,
     *       CHUNK_WRITE_WAIT_MAX_YIELDS yields at most (-D tunable).
     */
    static void wait_out_compressor(const ChunkType& chunk) noexcept;

    /**
     * @brief Finishes a compression session
     *
     * @details
     * Decrements the in-flight counter and deactivates slow-path
     * if no other compressions are active.
     *
     * @tparam ProxyType same proxy type the matching start() was called with
     * @param log Enable logging of the event
     * @note Handles a negative remaining count explicitly: see hard_inflight_'s
     *       signed-counter contract for why that state is tolerated.
     */
    template <typename ProxyType>
    void finish([[maybe_unused]] bool log = true);
};

/**
 * @brief Serial specialisation: no threads to coordinate, so nothing to hold.
 * @tparam T         Element type
 * @tparam ChunkType Chunk variant the manager coordinates
 */
template <typename T, typename ChunkType>
struct CompressionManagerOmpPolicy<T, ChunkType, false> : FlushReplayState<T, ChunkType>
{
    /**
     * @brief Serial builds queue nothing, so no rank can hold deferred ops.
     *
     * @details Mirrors the OpenMP policy's have_deferred_queue(). The write path
     *          calls it unconditionally -- see Proxy::apply_compound_operation,
     *          where the guard reads
     *          `v_old != zero || CompressionManagerType::have_deferred_queue()`.
     *          That call sits in a plain `else`, not a discarded `if constexpr`
     *          branch, so the name must exist in EVERY configuration.
     *
     *          gcc happened to accept its absence here; clang/icpx resolve the
     *          non-dependent name at phase 1 and reject it, which is why only
     *          the oneapi and llvm non-OpenMP flavours failed with
     *          "no member named 'have_deferred_queue'".
     *
     * @return always false: nothing is ever deferred without OpenMP.
     */
    [[nodiscard]] static constexpr bool have_deferred_queue() noexcept
    {
        return false;
    }
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
class CompressionManager : public CompressionManagerOmpPolicy<T, ChunkType, build_config::use_openmp>
{
  public:
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

#include "chunk/internal/compression_manager_inline.h"
