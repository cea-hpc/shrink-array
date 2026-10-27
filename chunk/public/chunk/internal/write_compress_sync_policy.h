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

#include <vector>  // std::vector
#include <atomic>  // std::atomic, std::atomic_thread_fence, std::memory_order
#include <cstdint> // uint32_t (the ring_log payloads)
#include <mutex>   // std::once_flag / std::call_once — NOT just std::mutex; do not remove again

#include "chunk/config.h"            // chunk_config::asymmetric_dekker, chunk_config::ring_log_debug
#include "chunk/PMP_CHUNK.h"         // CHUNK_PMP_THREAD_CAPACITY
#include "chunk/ring_log_debug.h"    // ring_log, get_tid
#include "chunk/internal/immortal.h" // Immortal / LazyImmortal: statics safe for global arrays

#ifdef __linux__
    #include <sys/syscall.h> // SYS_membarrier
    #include <unistd.h>      // syscall
#endif

namespace shrink_array
{

// membarrier(2) commands, spelled out: the values are ABI (Linux 4.14), older
// uapi headers stop before them, and <linux/membarrier.h> declares them as
// enumerators, which a fallback macro could not detect.
inline constexpr int membarrier_cmd_private_expedited          = 1 << 3; // MEMBARRIER_CMD_PRIVATE_EXPEDITED
inline constexpr int membarrier_cmd_register_private_expedited = 1 << 4; // MEMBARRIER_CMD_REGISTER_PRIVATE_EXPEDITED

/**
 * @brief The heavy side of the asymmetric handshake: process-wide, not per
 *        proxy type, because membarrier() registration is per process.
 *
 * @details membarrier(MEMBARRIER_CMD_PRIVATE_EXPEDITED) makes every CPU that
 *          is running a thread of this process execute a full memory barrier
 *          (an IPI, or the barrier a context switch implies) and returns only
 *          once all have. Cost: some tens of microseconds at 64 threads, paid
 *          on the rare side only; chunk_config::asymmetric_dekker explains the
 *          trade, ParamChunk::chunk_stats() counts the calls (gate start / flush /
 *          driver).
 *
 *          The expedited command must be registered once, before use;
 *          register_membarrier() does so from the policy's ensure_initialized(),
 *          i.e. from the first array's constructor of each variant -- before
 *          any parallel write. A failed registration (kernel < 4.14, seccomp,
 *          not Linux) leaves membarrier_ready false and every announce below
 *          keeps its symmetric shape, so the protocol is never weaker than
 *          the symmetric one, only faster when it can be.
 */
namespace dekker
{
/// @brief True once membarrier() is registered and heavy_barrier() may rely on it.
/// Written once, before the parallel regions, by register_membarrier();
/// atomic so the readers need no other ordering (a relaxed load is a plain
/// load on every target).
inline std::atomic<bool> membarrier_ready{false};

/// @brief Register MEMBARRIER_CMD_PRIVATE_EXPEDITED for this process.
/// @return true if heavy_barrier() will use it; idempotent.
inline bool register_membarrier() noexcept
{
    if constexpr (!chunk_config::asymmetric_dekker)
    {
        return false;
    }
    else
    {
#if defined(__linux__) && defined(SYS_membarrier)
        if (membarrier_ready.load(std::memory_order_relaxed))
        {
            return true;
        }
        const long rc = ::syscall(SYS_membarrier, membarrier_cmd_register_private_expedited, 0, 0);
        membarrier_ready.store(rc == 0, std::memory_order_release);
        return rc == 0;
#else
        return false;
#endif
    }
}

/**
 * @brief The rare side's barrier: a seq_cst fence, and -- asymmetric
 *        protocol, membarrier registered -- a barrier on every other thread
 *        of the process too.
 * @details The fence stays in both shapes: it is what the symmetric proof
 *          needs, and under the asymmetric one it orders this thread's own
 *          announce before the syscall, which membarrier(2) requires of the
 *          caller anyway.
 */
inline void heavy_barrier() noexcept
{
    std::atomic_thread_fence(std::memory_order_seq_cst);
    if constexpr (chunk_config::asymmetric_dekker)
    {
#if defined(__linux__) && defined(SYS_membarrier)
        if (membarrier_ready.load(std::memory_order_relaxed)) [[likely]]
        {
            ::syscall(SYS_membarrier, membarrier_cmd_private_expedited, 0, 0);
        }
#endif
    }
}

/// @brief Is the writer side running the light (asymmetric) announce?
[[nodiscard]] inline bool asymmetric_active() noexcept
{
    if constexpr (!chunk_config::asymmetric_dekker)
    {
        return false;
    }
    else
    {
        return membarrier_ready.load(std::memory_order_relaxed);
    }
}

/// @brief What ParamChunk::chunk_stats() reports as the handshake in use.
[[nodiscard]] inline const char* protocol_name() noexcept
{
    if constexpr (!chunk_config::asymmetric_dekker)
    {
        if constexpr (chunk_config::tsan_build)
        {
            return "symmetric (fences; ThreadSanitizer build)";
        }
        else
        {
            return "symmetric (fences; no membarrier on this platform)";
        }
    }
    else
    {
        return membarrier_ready.load(std::memory_order_relaxed)
                   ? "asymmetric (plain store + membarrier on the rare side)"
                   : "symmetric (fences; membarrier registration FAILED, kernel < 4.14 or seccomp)";
    }
}
} // namespace dekker

/**
 * @brief Dekker handshake between element writers and compression/flush passes.
 *
 * @details A compression pass, or a cross-rank flush, must not run while another
 *          thread is inside a write critical section: it would move or free the
 *          buffer that writer is about to touch. There is no lock on the write
 *          path -- it is the hot path -- so the two sides synchronise with a
 *          Dekker-style handshake instead:
 *
 *          - the writer announces itself (announce_write()) and then checks
 *            whether any flush is in progress;
 *          - the flusher announces itself (announce_flush()) and then checks
 *            whether any writer is active (any_thread_writing()).
 *
 *          A Dekker needs a full barrier between each side's announce and its
 *          look at the other side: a plain store plus an acquire load lets both
 *          sides miss each other and proceed together, even on x86. Where that
 *          barrier sits is the protocol's shape (chunk_config::asymmetric_dekker):
 *          symmetric, a seq_cst fence on both sides; asymmetric, nothing but a
 *          compiler barrier on the writer side and a membarrier() on the rare
 *          side, which runs the barrier on every CPU of the process.
 *          announce_write() and announce_flush() exist precisely so that no call
 *          site can announce without its ordering -- see announce_write() for
 *          both proofs, and docs/synchronisation_dekker_fr.md.
 *
 *          This class owns ONLY the write/flush handshake state. The
 *          compressors' own counts (hard_inflight_, stealing_inflight_) live in
 *          CompressionManager.
 *
 *          The whole policy is OpenMP-only; the serial build gets the empty
 *          specialisation below, since a single thread cannot race with itself.
 *
 * @tparam ProxyT    proxy type the policy is instantiated for -- it keys the static
 *                 state, giving one independent handshake per (variant, T)
 * @tparam Enabled build_config::threaded; never pass it explicitly
 */
template <typename ProxyT, bool Enabled = build_config::threaded>
class WriteCompressSyncPolicy
{
  public:
    /// @brief One writer flag, padded to its own cache line: the chunk the
    ///        rank is writing, or nullptr.
    /// @details Per-thread, not a shared counter: a counter written on every
    ///          write is a cache-line storm. The write side pays only for its
    ///          own line; the scan cost goes to the rare reader,
    ///          any_thread_writing().
    ///
    ///          A pointer rather than a bool (same store, same line): the
    ///          compressor's writer window becomes "no writer on a chunk I
    ///          MARKED" (any_thread_writing_on_marked) instead of "no writer
    ///          anywhere", which at high thread counts makes the compressor wait
    ///          very often. "Non-null" keeps the flag's other meaning, an enqueue
    ///          in progress on this rank, for the flushers (is_thread_writing).
    struct alignas(128) WritingFlag
    {
        std::atomic<const void*> v{nullptr}; ///< The chunk being written, or nullptr.
    };

    WriteCompressSyncPolicy() noexcept = default;

    // The state is static: an instance is a handle, neither copied nor moved.
    WriteCompressSyncPolicy(const WriteCompressSyncPolicy&)            = delete;
    WriteCompressSyncPolicy& operator=(const WriteCompressSyncPolicy&) = delete;
    WriteCompressSyncPolicy(WriteCompressSyncPolicy&&)                 = delete;
    WriteCompressSyncPolicy& operator=(WriteCompressSyncPolicy&&)      = delete;

    ~WriteCompressSyncPolicy() = default;

    /**
     * @brief Allocate the per-thread writer flags, once per instantiation.
     * @post per_thread_writing holds CHUNK_PMP_THREAD_CAPACITY flags and
     *       num_thread_flushing is zero.
     * @note Idempotent (std::call_once); callers may invoke it freely.
     */
    static void ensure_initialized();

    /**
     * @brief Writer-side Dekker announce: raise the flag, then order it before
     *        the caller's look at the other side.
     *
     * @details Packaged as one verb so that no call site can announce without
     *          the ordering. Two shapes, chosen by chunk_config::asymmetric_dekker
     *          and dekker::asymmetric_active():
     *
     *          ASYMMETRIC (the fast one): a RELAXED store of our flag and a
     *          compiler barrier (atomic_signal_fence). No CPU fence at all: on
     *          AArch64 this is `str` followed by the caller's plain `ldr`s. The
     *          CPU may make the store visible after the load -- that is exactly
     *          what a Dekker forbids, and here the OTHER side repairs it. The
     *          compressor / flusher announces itself with a seq_cst RMW, then
     *          calls dekker::heavy_barrier(): membarrier() runs a full barrier
     *          on every CPU executing one of our threads, at some instant of
     *          each writer's instruction stream, before returning. For a given
     *          writer, its store S and its load L are ordered S before L in
     *          program order, and that instant falls either
     *            - before both: L executes after the barrier, so it sees the
     *              opponent's announce (visible before the syscall) -> slow path;
     *            - between S and L: S is drained by the barrier, the opponent's
     *              scan after the syscall sees our flag -> it waits;
     *            - after both: L was already executed, so S was too (it precedes
     *              L in program order and the barrier drains the store buffer):
     *              the scan sees our flag -> it waits.
     *          Exactly one side proceeds. The compiler barrier keeps S before L
     *          in program order; without it the compiler could hoist the load.
     *          Same argument as membarrier(2) itself (and Folly's
     *          asymmetricHeavyBarrier).
     *
     *          SYMMETRIC (non-Linux and TSan builds, and the run-time fallback
     *          when membarrier registration fails): seq_cst store + seq_cst fence, the
     *          opponent likewise; the caller's load may then be relaxed. By
     *          [atomics.order]: with A our store, X our fence, Y the opponent's
     *          fence and B the opponent's acquire-load of our flag, if X
     *          precedes Y in the total order S then B sees A (the opponent
     *          waits); otherwise Y precedes X, and by the same rule with the
     *          roles swapped OUR load after X sees the opponent's announce (we
     *          take the slow path). Never both.
     *
     * @param thr   calling thread's rank
     * @param chunk the chunk about to be written (or enqueued on): what the
     *              compressor's window looks for, see WritingFlag
     * @see announce_flush for the mirror image on the flush side, and
     *      CompressionManager::start() for the compressor's
     */
    // always_inline: otherwise gcc -O3 may outline it (the fallback branch
    // tips its heuristic), a call on every element write.
    [[gnu::always_inline]] static void announce_write(int thr, const void* chunk) noexcept;

    /**
     * @brief Lower this thread's writer flag, ending its critical section.
     * @param thr calling thread's rank
     */
    static void end_write(int thr) noexcept
    {
        if constexpr (chunk_config::ring_log_debug)
        {
            ring_log(get_tid(), EventCode::policy_end_write, static_cast<uint32_t>(thr));
        }

        per_thread_writing[thr].v.store(nullptr, std::memory_order_release);
    }

    /**
     * @brief Is any rank inside a write critical section ON A MARKED CHUNK?
     * @tparam ChunkT the chunk type the flags point to (the policy's proxy
     *                writes one chunk type)
     * @param max_thr number of ranks to scan
     * @return true as soon as one rank's flag points to a chunk whose
     *         steal_mark_ is set
     * @note The compressor's writer window with the per-chunk marks
     *       (CompressionManager::start()): the writers of unmarked chunks are
     *       not in its way and are not waited for -- nor steered away. The
     *       pointer read here is a live chunk: a rank clears its flag before
     *       its write returns, and chunks die outside parallel regions only.
     */
    template <typename ChunkT>
    static bool any_thread_writing_on_marked(int max_thr) noexcept;

    /**
     * @brief Is rank @p thr currently inside its write critical section?
     * @param thr rank to test
     * @return true while that rank is writing
     * @note Used by flush_chunk_all_ranks() to wait for one specific rank's map
     *       to go quiescent before scanning it cross-thread.
     */
    static bool is_thread_writing(int thr) noexcept
    {
        return per_thread_writing[thr].v.load(std::memory_order_acquire) != nullptr;
    }

    /**
     * @brief seq_cst flavour of is_thread_writing(), for self-diagnostics only.
     * @param thr rank to test -- in practice the caller's own
     * @return true while that rank is writing
     * @note Exists for the any_check Dekker assertions: a thread verifying its
     *       OWN flag must participate in the same total order S as the announce
     *       it just performed, which the acquire-load form does not guarantee.
     */
    static bool is_thread_writing_seq(int thr) noexcept
    {
        return per_thread_writing[thr].v.load(std::memory_order_seq_cst) != nullptr;
    }

    /**
     * @brief Does rank @p thr hold its writer flag on @p chunk -- is it inside
     *        a write critical section on THAT chunk?
     * @param thr   rank to test -- the caller's own
     * @param chunk the chunk
     * @return true while that rank's flag is @p chunk's address
     * @note For a thread reading its OWN flag (wait_before_materialise()): a
     *       relaxed load returns what this thread stored, in program order;
     *       no other thread writes a rank's flag.
     */
    static bool is_thread_writing_on(int thr, const void* chunk) noexcept
    {
        return per_thread_writing[thr].v.load(std::memory_order_relaxed) == chunk;
    }

    /**
     * @brief Flusher-side Dekker announce: enter the section, then the heavy
     *        barrier.
     * @details The rare side: seq_cst increment, then dekker::heavy_barrier()
     *          -- a seq_cst fence, plus membarrier() under the asymmetric
     *          protocol -- before scanning the writer flags. See
     *          announce_write() for both arguments; this is their other half.
     */
    static void announce_flush() noexcept
    {
        begin_flush();
        dekker::heavy_barrier();
    }

    /**
     * @brief Leave a flush critical section.
     * @details Release ordering publishes "this flush is done". An atomic RMW
     *          rather than a store, so concurrent begin/end from different
     *          threads can never lose an update.
     */
    static void end_flush() noexcept
    {
        num_thread_flushing.fetch_sub(1, std::memory_order_release);
    }

    /**
     * @brief Is ANY thread currently inside a flush critical section?
     * @param max_thr unused -- kept for symmetry with any_thread_writing()
     * @return true while at least one flush is in progress
     * @note O(1) counter load. seq_cst so it joins the Dekker total order with
     *       begin_flush()'s increment; the writer has already done its own
     *       seq_cst store and fence before reaching here.
     */
    static bool any_thread_flushing([[maybe_unused]] int max_thr) noexcept
    {
        return num_thread_flushing.load(std::memory_order_seq_cst) > 0;
    }

    /**
     * @brief Is ANY rank currently inside a write critical section?
     * @param max_thr number of ranks to scan
     * @return true as soon as one raised flag is found
     * @note O(max_thr). Acceptable because only the rare flush/compress side
     *       calls it; the hot write path never does.
     */
    static bool any_thread_writing(int max_thr) noexcept;

  private:
    /**
     * @brief Raise this thread's writer flag.
     * @param thr   calling thread's rank
     * @param chunk the chunk being written, published in the flag
     * @warning Do NOT call directly to announce a write: without the fence that
     *          follows it, the handshake is unsound on weakly-ordered targets.
     *          Use announce_write().
     */
    static void begin_write(int thr, const void* chunk) noexcept
    {
        if constexpr (chunk_config::ring_log_debug)
        {
            ring_log(get_tid(), EventCode::policy_begin_write, static_cast<uint32_t>(thr));
        }

        per_thread_writing[thr].v.store(chunk, std::memory_order_seq_cst);
    }

    /**
     * @brief Enter a flush critical section.
     * @details While any flush is in progress, compressors calling start() back
     *          off and return false, giving the flush exclusive write access.
     *          The increment is seq_cst: it is the flush side's place in the
     *          Dekker total order that start() depends on.
     * @post Must be paired with end_flush().
     * @warning Like begin_write(), this is not an announce: use announce_flush()
     *          when the caller then inspects the writer flags.
     */
    static void begin_flush() noexcept
    {
        // seq_cst: Dekker handshake with the deferred-enqueue path
        // (to_be_queued_for_deferred_processing) and start().
        num_thread_flushing.fetch_add(1, std::memory_order_seq_cst);
    }

    /// @brief One writer flag per thread rank, sized by ensure_initialized().
    static inline constinit internal::Immortal<std::vector<WritingFlag>> per_thread_writing_storage_{};
    static inline std::vector<WritingFlag>& per_thread_writing = per_thread_writing_storage_.value;

    /// @brief Number of threads currently inside a flush critical section
    ///        (begin_flush()/end_flush()).
    /// @details A counter, not a per-thread flag array: the only query is "is
    ///          ANY thread flushing?". The asymmetry with per_thread_writing is
    ///          deliberate: this is written only on the RARE flush path, and it
    ///          makes the HOT reader any_thread_flushing() (every deferred
    ///          enqueue, start()) one O(1) load instead of an O(max_thr) scan,
    ///          which was a significant share of the enqueue cost at high
    ///          thread counts.
    alignas(128) static inline std::atomic<int> num_thread_flushing{0};
};

/**
 * @brief Serial specialisation: empty, since a single thread cannot race itself.
 * @tparam ProxyT proxy type the policy would be instantiated for
 */
template <typename ProxyT>
class WriteCompressSyncPolicy<ProxyT, false>
{
};

} // namespace shrink_array

#include "chunk/internal/write_compress_sync_policy_inline.h" // the template bodies
