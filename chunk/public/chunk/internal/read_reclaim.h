/*
 * Copyright 2026 CEA (www.cea.fr)
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
 * @file read_reclaim.h
 * @brief Pure reads against buffer reclaim: a grace period per compression pass.
 *
 * @par The hazard
 * A pure read (ChunkBase::element_at) loads a chunk's `values` pointer, then dereferences
 * it. A compression pass run INSIDE a region (perform_compression, or compress() from a
 * region) CASes `values` to the constant and releases the old buffer. The write/compress
 * handshake waits for WRITERS only, so without this protocol a reader between its load and
 * its dereference reads a released buffer (use-after-free, or a recycled block holding a
 * value the chunk never held).
 *
 * @par The protocol
 *  - READER (ReadSection, around element_at's load + dereference): a per-thread
 *    sequence number, odd while a read is in flight. Entry: a release store of
 *    the odd value, then the LIGHT side of the asymmetric Dekker (a compiler
 *    barrier when membarrier() is registered, a seq_cst fence otherwise).
 *    Exit: a release store of the next even value. Both releases order the
 *    element load before them: the reclaimer trusts an even value, and, after
 *    waiting for an odd slot to move, the next odd one as well.
 *  - RECLAIMER (PassScope, around a pass run inside a region): every buffer the
 *    pass releases is collected instead of freed. At the end of the pass,
 *    dekker::heavy_barrier() (membarrier: every running thread of the process
 *    executes a full barrier), then each reader slot is read: a reader seen
 *    even either finished before, or loads `values` after the barrier and sees
 *    the constant; a reader seen odd may hold an old pointer, and the
 *    reclaimer waits until its sequence moves. Then the buffers are released
 *    as an unscoped pass releases them -- to the reuse list, at once.
 *
 *    One membarrier per PASS, not per chunk; reuse stays immediate (deferring it to the end
 *    of the region would make every materialisation in a long region allocate afresh). The
 *    wait is only for reads in flight -- nanoseconds, unless the reader was preempted.
 *
 *    Passes run between regions (the OMPT / finalize_parallel_phase drain) are
 *    not scoped: no user read runs then.
 *
 * @note OpenMP builds only; a serial build compiles all of this out.
 * @warning The ordering argument is the one of the write/compress handshake
 *          (write_compress_sync_policy.h). Re-run read_reclaim_validation (with
 *          and without CHUNK_TEST_READ_DELAY_EVERY, under ASan in std-alloc)
 *          whenever any of it is touched.
 */

#include <array>       // std::array (g_slots)
#include <algorithm>   // std::min
#include <atomic>      // std::atomic, std::atomic_thread_fence, std::memory_order
#include <cstdint>     // std::uint64_t
#include <cstdio>      // std::fprintf (EXPERIMENT mode 5 report)
#include <cstdlib>     // std::abort (EXPERIMENT mode 5)
#include <cstring>     // std::memcpy (EXPERIMENT mode 3)
#include <type_traits> // std::is_trivially_copyable_v (EXPERIMENT mode 3)
#include <thread>      // std::this_thread::yield
#include <vector>      // std::vector

#include "chunk/config.h"                              // build_config::threaded
#include "chunk/internal/chunk_size_policy.h"          // t_active_chunk_size / t_alloc_key, restored per released block
#include "chunk/internal/cpu_pause.h"                  // CHUNK_CPU_PAUSE
#include "chunk/internal/def_force_inline.h"           // CHUNK_FORCE_INLINE (EXPERIMENT mode 3)
#include "chunk/internal/write_compress_sync_policy.h" // dekker::heavy_barrier / asymmetric_active

#ifndef CHUNK_READ_RECLAIM_SLOTS
    #define CHUNK_READ_RECLAIM_SLOTS 1024
#endif

// EXPERIMENT (branch exp/optimistic-read): what a pure read costs under each
// read-side protocol. Measurement only -- modes 1 and 2 are NOT safe against a
// pass releasing a buffer under a reader.
//   0  the announce (default): odd release store + light fence ... even release store
//   1  nothing: the upper bound of any change to the read side
//   2  the cost of an optimistic read on its success path: an acquire load of the
//      reclaim epoch before, an acquire fence + a load + a compare after; the
//      retry is not implemented (a mismatch is only counted)
//   3  optimistic, no fence: acquire load of the epoch before; after, a relaxed re-load
//      whose ADDRESS depends on the element value (pass()), ordered after the element
//      load by the Arm address-dependency rule; without pass() (the other call sites)
//      a plain relaxed re-load, same cost, no ordering
//   4  mode 3 with no acquire at all: the entry load of the epoch is relaxed, and
//      element_at's `values` load depends on it by address (dep()); the element load
//      depends on `values`, the re-check on the element (pass()). Other call sites
//      get no ordering from the entry load (measurement only)
//   5  quiescent-state reclamation (QSBR): a read does nothing; a thread announces a
//      quiescent point -- no pointer read from a chunk held -- on every write entry and
//      after the phase-end barrier, and comes online at begin_parallel_phase(). A pass
//      in a region parks the buffers it releases, tagged with a new epoch; they are
//      released by the same thread at its next pass once every other online thread
//      has passed a quiescent point since, and by finalize_parallel_phase() otherwise
//   6  mode 5 plus a read-side report: a ReadSection's entry is a quiescent point too
//      (no pointer from an earlier read is held). It loads the epoch relaxed and
//      compares it with its own slot; only when the epoch moved does it take an
//      acquire fence and store the epoch, released, into its slot. A thread that
//      only reads thus lets the writers' parked buffers go at their next pass
//   7  optimistic WITH retry (mode 3 made correct): a read announces nothing; it loads
//      the epoch (acquire), reads, and re-checks the epoch -- ordered by an address
//      dependency on the value (retry_after) or by an acquire fence (retry, the block
//      reads). If a grace period started in between, it reads again. A pass bumps the
//      epoch after its swaps and before its releases, and releases at once, as mode 0:
//      no memory held longer. Needs the fast allocator: a released value buffer stays
//      mapped (value pools never unmap, MADV_DONTNEED keeps the mapping), so a read of
//      a reused buffer returns garbage the re-check discards, never a fault.
//      The compressor's scans keep an announce (ScanSection), as in modes 5 / 6
#ifndef CHUNK_READ_SECTION_MODE
    #define CHUNK_READ_SECTION_MODE 0
#endif

namespace shrink_array::read_reclaim
{

/// @brief The protocol is compiled in: OpenMP builds only.
inline constexpr bool enabled = build_config::threaded;

/// @brief Reader slots: one per thread that has ever read, a cache line each.
inline constexpr int slot_count = CHUNK_READ_RECLAIM_SLOTS;

/// @brief One reader's sequence number, on a cache line of its own.
struct alignas(64) ReaderSlot
{
    std::atomic<std::uint64_t> seq{0};        ///< Odd while a read is in flight. Mode 5: the last quiescent epoch.
    std::atomic<bool>          online{false}; ///< EXPERIMENT mode 5: in the current phase
    std::atomic<std::uint64_t> scan_seq{0};   ///< EXPERIMENT mode 5: odd while a ScanSection is open
};

/// @brief EXPERIMENT mode 5: wait_for_readers() and ScanSection use scan_seq.
inline constexpr bool scan_announce =
    CHUNK_READ_SECTION_MODE == 5 || CHUNK_READ_SECTION_MODE == 6 || CHUNK_READ_SECTION_MODE == 7;

/// @brief EXPERIMENT mode 7: optimistic reads that retry.
inline constexpr bool optimistic_retry = enabled && CHUNK_READ_SECTION_MODE == 7;
static_assert(!optimistic_retry || build_config::use_fast_alloc,
              "CHUNK_READ_SECTION_MODE=7 needs the fast allocator: a free()d buffer may be unmapped");

/// @brief The sequence wait_for_readers() waits on.
inline std::atomic<std::uint64_t>& announce_seq(ReaderSlot& slot) noexcept
{
    return scan_announce ? slot.scan_seq : slot.seq;
}

inline std::array<ReaderSlot, slot_count> g_slots;     ///< The reader slots, claimed in order.
inline std::atomic<int>                   g_nslots{0}; ///< Slots claimed so far (may exceed slot_count).
/// @brief Readers past slot_count: counted in one shared RMW (seq_cst), slow but
///        correct. A program with more reader threads than slot_count raises
///        CHUNK_READ_RECLAIM_SLOTS.
inline std::atomic<long>        g_overflow_readers{0};
inline thread_local ReaderSlot* t_slot     = nullptr; ///< This thread's slot, once claimed.
inline thread_local bool        t_overflow = false;   ///< This thread found no slot left.

/// @brief EXPERIMENT mode 2: bumped by every grace period, before its releases.
alignas(64) inline std::atomic<std::uint64_t> g_reclaim_epoch{0};
/// @brief EXPERIMENT mode 2: reads that saw the epoch move (would have retried).
inline thread_local long                 t_optimistic_conflicts = 0;
/// @brief EXPERIMENT mode 7: reads retried, all threads (counted on the cold path only).
alignas(64) inline std::atomic<long> g_optimistic_retries{0};
[[gnu::noinline, gnu::cold]] inline void note_optimistic_conflict() noexcept
{
    ++t_optimistic_conflicts;
}

/// @brief EXPERIMENT mode 7: count a retry, return the new epoch (acquire).
std::uint64_t note_retry() noexcept;

/// @brief EXPERIMENT mode 7: the retry count, at exit.
struct RetryReport
{
    RetryReport()                              = default;
    RetryReport(const RetryReport&)            = delete;
    RetryReport& operator=(const RetryReport&) = delete;
    ~RetryReport()
    {
        if constexpr (enabled && CHUNK_READ_SECTION_MODE == 7)
        {
            std::fprintf(stderr, "[optimistic] reads retried %ld\n", g_optimistic_retries.load());
        }
    }
};
inline RetryReport g_retry_report;

/// @brief First read of this thread: claim a slot (or join the overflow).
inline ReaderSlot* claim_slot() noexcept
{
    if (t_overflow)
    {
        return nullptr;
    }
    if (const int slot_idx = g_nslots.fetch_add(1, std::memory_order_acq_rel); slot_idx < slot_count) [[likely]]
    {
        t_slot = &g_slots[slot_idx];
        return t_slot;
    }
    t_overflow = true;
    return nullptr;
}

/// @brief The light side: free with membarrier registered, a fence otherwise.
inline void light_fence() noexcept
{
    if (dekker::asymmetric_active()) [[likely]]
    {
        std::atomic_signal_fence(std::memory_order_seq_cst);
    }
    else
    {
        std::atomic_thread_fence(std::memory_order_seq_cst);
    }
}

/**
 * @brief Brackets one pure read: from before the `values` load to after the
 *        element load.
 */
class ReadSection
{
  public:
    /// @brief Enter: the slot's sequence goes odd, then the light fence.
    ReadSection() noexcept;
    /// @brief Leave: the sequence goes even, released after the element load.
    ~ReadSection();
    // A scope: neither copied nor moved.
    ReadSection(const ReadSection&)            = delete;
    ReadSection& operator=(const ReadSection&) = delete;
    ReadSection(ReadSection&&)                 = delete;
    ReadSection& operator=(ReadSection&&)      = delete;

    /// @brief EXPERIMENT mode 3: the re-check, ordered after the load of @p value by an
    ///        address dependency; returns @p value. Other modes: returns @p value.
    template <typename V>
    [[nodiscard]] V pass(V value) const noexcept;

    /// @brief EXPERIMENT mode 7: true when a grace period began since the entry (or the
    ///        last retry): the caller reads again. The re-check is ordered after the load
    ///        of @p value by an address dependency. Other modes: false.
    template <typename V>
    [[nodiscard]] bool retry_after(const V& value) const noexcept;

    /// @brief EXPERIMENT mode 7: retry_after() for a read of several values (a block):
    ///        the re-check is ordered by an address dependency on the OR of every
    ///        value fold()ed since the last check (no fence). Other modes: false.
    [[nodiscard]] bool retry() const noexcept;

    /// @brief EXPERIMENT mode 7: add @p value's bytes to the block's dependency.
    ///        Other modes: nothing.
    template <typename V>
    void fold(const V& value) const noexcept;

    /// @brief EXPERIMENT mode 4: 0 at run time, an address dependency on the
    ///        entry load of the epoch for the CPU. Other modes: 0.
    [[nodiscard]] std::uintptr_t dep() const noexcept
    {
        if constexpr (CHUNK_READ_SECTION_MODE == 4)
        {
            std::uintptr_t opaque = seq_;
            __asm__ volatile("" : "+r"(opaque));
            return opaque ^ seq_;
        }
        return 0;
    }

  private:
    ReaderSlot*           slot_   = nullptr;
    mutable std::uint64_t seq_    = 0; ///< mode 7: refreshed by a retry
    mutable bool          passed_ = false;
    mutable std::uintptr_t acc_   = 0; ///< EXPERIMENT mode 7: fold()ed bits of a block ///< EXPERIMENT mode 3: pass() did the re-check
};

/**
 * @brief EXPERIMENT mode 5: a read section that announces itself, for the
 *        library's own scans (a compressor's start() reading the chunk a writer
 *        published): what ArrayChunkCore::await_chunk_readers() waits for. In
 *        mode 5 a ReadSection does nothing; in the other modes this IS one.
 */
#if CHUNK_READ_SECTION_MODE == 5 || CHUNK_READ_SECTION_MODE == 6 || CHUNK_READ_SECTION_MODE == 7
class ScanSection
{
  public:
    ScanSection() noexcept
    {
        if constexpr (enabled)
        {
            ReaderSlot* slot = t_slot != nullptr ? t_slot : claim_slot();
            if (slot != nullptr) [[likely]]
            {
                slot_ = slot;
                seq_  = slot->scan_seq.load(std::memory_order_relaxed) + 1; // odd
                slot->scan_seq.store(seq_, std::memory_order_release);
                light_fence();
            }
            else
            {
                g_overflow_readers.fetch_add(1, std::memory_order_seq_cst);
                light_fence();
            }
        }
    }
    ~ScanSection()
    {
        if constexpr (enabled)
        {
            if (slot_ != nullptr) [[likely]]
            {
                slot_->scan_seq.store(seq_ + 1, std::memory_order_release);
            }
            else
            {
                g_overflow_readers.fetch_sub(1, std::memory_order_release);
            }
        }
    }
    ScanSection(const ScanSection&)            = delete;
    ScanSection& operator=(const ScanSection&) = delete;
    ScanSection(ScanSection&&)                 = delete;
    ScanSection& operator=(ScanSection&&)      = delete;

  private:
    ReaderSlot*   slot_ = nullptr;
    std::uint64_t seq_  = 0;
};
#else
using ScanSection = ReadSection;
#endif

/**
 * @brief The grace period: every read in flight when it starts has ended when
 *        it returns. The caller has already swapped the pointers it is about
 *        to release.
 */
inline void wait_for_readers() noexcept
{
    g_reclaim_epoch.fetch_add(1, std::memory_order_seq_cst); // EXPERIMENT mode 2
    dekker::heavy_barrier(); // seq_cst fence + membarrier: orders the CASes before the slot reads
    const int         nslots = std::min(g_nslots.load(std::memory_order_acquire), slot_count);
    const ReaderSlot* mine   = t_slot;
    for (int i = 0; i < nslots; ++i)
    {
        if (&g_slots[i] == mine)
        {
            continue;
        }
        const std::uint64_t seq = announce_seq(g_slots[i]).load(std::memory_order_acquire);
        if ((seq & 1U) == 0U)
        {
            continue;
        }
        for (int spins = 0; announce_seq(g_slots[i]).load(std::memory_order_acquire) == seq;)
        {
            CHUNK_CPU_PAUSE();
            if (++spins == 1000)
            {
                std::this_thread::yield(); // the reader may have been preempted on this CPU
                spins = 0;
            }
        }
    }
    for (int spins = 0; g_overflow_readers.load(std::memory_order_acquire) != 0;)
    {
        CHUNK_CPU_PAUSE();
        if (++spins == 1000)
        {
            std::this_thread::yield();
            spins = 0;
        }
    }
}

/// @brief A buffer a pass released, with what its release needs: the chunk
///        (for the type), the rank, and the per-thread geometry and allocation
///        key in force when the pass let it go (dynamic builds route the free
///        by them).
struct Deferred
{
    void (*release)(void* chunk, int thr, void* block) noexcept; ///< The chunk type's release.
    void*        chunk;                                          ///< The chunk the buffer belonged to.
    void*        block;                                          ///< The buffer.
    int          thr;                                            ///< The releasing rank.
    unsigned int chunk_size;                                     ///< raw t_active_chunk_size (0 = unset)
    unsigned int alloc_key;                                      ///< t_alloc_key at the release
};

/// @brief Non-null while a scoped pass runs on this thread: free_block_to_pool
///        appends here instead of releasing.
inline thread_local std::vector<Deferred>* t_batch = nullptr;

/**
 * @brief Scope of a compression pass that may run while other threads read.
 * @details Collects the buffers the pass releases, then -- one grace period
 *          for the whole pass -- releases them. Nested scopes fold into the
 *          outermost one.
 */
class PassScope
{
  public:
    /// @brief Open the batch, unless an enclosing scope already did.
    PassScope() noexcept;
    /// @brief The outermost scope: wait for the readers, then release the batch.
    ~PassScope();
    // A scope: neither copied nor moved.
    PassScope(const PassScope&)            = delete;
    PassScope& operator=(const PassScope&) = delete;
    PassScope(PassScope&&)                 = delete;
    PassScope& operator=(PassScope&&)      = delete;

  private:
    /// @brief Per-thread, reused: a pass does not allocate its batch afresh.
    static std::vector<Deferred>& storage() noexcept
    {
        thread_local std::vector<Deferred> batch;
        return batch;
    }
    bool owner_ = false;
};

/// @brief Collect @p block instead of releasing it, if a scoped pass is running.
/// @return true if collected (the caller must not release it).
inline bool defer_release(void (*release)(void*, int, void*) noexcept, void* chunk, int thr, void* block)
{
    if constexpr (enabled)
    {
        if (std::vector<Deferred>* batch = t_batch; batch != nullptr) [[unlikely]]
        {
            batch->push_back(Deferred{.release    = release,
                                      .chunk      = chunk,
                                      .block      = block,
                                      .thr        = thr,
                                      .chunk_size = t_active_chunk_size,
                                      .alloc_key  = t_alloc_key});
            return true;
        }
    }
    return false;
}

// ---------------------------------------------------------------------------
// EXPERIMENT mode 5: quiescent-state reclamation
// ---------------------------------------------------------------------------

/// @brief Mode 5 is compiled in.
inline constexpr bool qsbr = enabled && (CHUNK_READ_SECTION_MODE == 5 || CHUNK_READ_SECTION_MODE == 6);

/// @brief A buffer parked by a pass, with the epoch its pass opened.
struct QPending
{
    Deferred      d;
    std::uint64_t epoch;
};

/// @brief Per slot: touched by its own thread in a region, by finalize outside one.
inline std::array<std::vector<QPending>, slot_count> g_qpending;
inline std::atomic<long>                             g_qheld{0};      ///< Buffers parked now.
inline std::atomic<long>                             g_qheld_peak{0}; ///< Most buffers parked at once.
inline std::atomic<long>                             g_qparked{0};    ///< Buffers parked in all.
inline std::atomic<long> g_qdrained{0};     ///< Of those, released by finalize (not by a later pass).
inline std::atomic<int>  g_qonline_peak{0}; ///< Most threads online in one phase.
alignas(64) inline std::atomic<long> g_qread_reports{0}; ///< EXPERIMENT mode 6: read entries that saw the epoch move.

/// @brief The report, at exit.
struct QReport
{
    QReport()                          = default;
    QReport(const QReport&)            = delete;
    QReport& operator=(const QReport&) = delete;
    ~QReport()
    {
        if constexpr (qsbr)
        {
            std::fprintf(stderr,
                         "[qsbr] buffers parked %ld, released at phase end %ld, parked at once (peak) %ld, "
                         "threads online (peak) %d, read-side reports %ld\n",
                         g_qparked.load(), g_qdrained.load(), g_qheld_peak.load(), g_qonline_peak.load(),
                         g_qread_reports.load());
        }
    }
};
inline QReport g_qreport;

/// @brief Release one parked buffer, under the geometry and key of its pass.
inline void q_release(const QPending& pending) noexcept
{
    const unsigned int saved_size = t_active_chunk_size;
    const unsigned int saved_key  = t_alloc_key;
    t_active_chunk_size           = pending.d.chunk_size;
    t_alloc_key                   = pending.d.alloc_key;
    pending.d.release(pending.d.chunk, pending.d.thr, pending.d.block);
    t_active_chunk_size = saved_size;
    t_alloc_key         = saved_key;
}

/// @brief This thread's slot, claimed on first use; aborts past slot_count.
inline ReaderSlot* q_slot() noexcept
{
    ReaderSlot* slot = t_slot != nullptr ? t_slot : claim_slot();
    if (slot == nullptr) [[unlikely]]
    {
        std::fprintf(stderr, "[qsbr] more threads than CHUNK_READ_RECLAIM_SLOTS\n");
        std::abort();
    }
    return slot;
}

/// @brief begin_parallel_phase(), on every thread of the team: online, at the current epoch.
/// @details seq_cst against the reclaimer's epoch increment + online load: either
///          it sees this thread online (and waits for its next quiescent point) or
///          this load sees its epoch (and the thread reads after the pass's swaps).
inline void q_online() noexcept
{
    if constexpr (qsbr)
    {
        ReaderSlot* slot = q_slot();
        slot->online.store(true, std::memory_order_seq_cst);
        slot->seq.store(g_reclaim_epoch.load(std::memory_order_seq_cst), std::memory_order_seq_cst);
    }
}

/// @brief A point where this thread holds no pointer read from a chunk.
/// @details Release: this thread's earlier element loads before the store; the
///          acquire load of the epoch: a pass whose increment it sees has its
///          swaps visible to this thread's later loads.
CHUNK_FORCE_INLINE void quiescent_point() noexcept
{
    if constexpr (qsbr)
    {
        if (ReaderSlot* slot = t_slot; slot != nullptr) [[likely]]
        {
            slot->seq.store(g_reclaim_epoch.load(std::memory_order_acquire), std::memory_order_release);
        }
    }
}

/// @brief EXPERIMENT mode 6, the slow side of a read's entry: the epoch moved to @p epoch.
/// @details The acquire fence pairs with the pass's seq_cst increment that the relaxed
///          load read: the pass's swaps happen before this read's loads. The release
///          store orders this thread's earlier reads (all their sections closed) before
///          the slot the pass's owner acquires. Same pairing as quiescent_point().
[[gnu::noinline]] inline void q_read_report(ReaderSlot* slot, std::uint64_t epoch) noexcept
{
    std::atomic_thread_fence(std::memory_order_acquire);
    slot->seq.store(epoch, std::memory_order_release);
    g_qread_reports.fetch_add(1, std::memory_order_relaxed);
}

/// @brief EXPERIMENT mode 6: a read's entry, a quiescent point reported only when the epoch moved.
CHUNK_FORCE_INLINE void q_read_entry() noexcept
{
    if (ReaderSlot* slot = t_slot; slot != nullptr) [[likely]]
    {
        const std::uint64_t epoch = g_reclaim_epoch.load(std::memory_order_relaxed);
        if (epoch != slot->seq.load(std::memory_order_relaxed)) [[unlikely]]
        {
            q_read_report(slot, epoch);
        }
    }
}

/// @brief Release this thread's parked buffers that every other online thread has passed.
inline void q_try_release_own() noexcept
{
    ReaderSlot* mine = t_slot;
    if (mine == nullptr)
    {
        return;
    }
    auto& parked = g_qpending[static_cast<std::size_t>(mine - g_slots.data())];
    if (parked.empty())
    {
        return;
    }
    std::uint64_t oldest = ~std::uint64_t{0};
    const int     nslots = std::min(g_nslots.load(std::memory_order_acquire), slot_count);
    for (int i = 0; i < nslots; ++i)
    {
        if (&g_slots[i] == mine || !g_slots[i].online.load(std::memory_order_seq_cst))
        {
            continue;
        }
        oldest = std::min(oldest, g_slots[i].seq.load(std::memory_order_acquire));
    }
    std::size_t done = 0;
    while (done < parked.size() && parked[done].epoch <= oldest)
    {
        q_release(parked[done]);
        ++done;
    }
    if (done != 0)
    {
        parked.erase(parked.begin(), parked.begin() + static_cast<std::ptrdiff_t>(done));
        g_qheld.fetch_sub(static_cast<long>(done), std::memory_order_relaxed);
    }
}

/// @brief End of a pass (PassScope): park its batch under a new epoch, then release what may be.
inline void q_park(std::vector<Deferred>& batch) noexcept
{
    ReaderSlot*         mine   = q_slot();
    const std::uint64_t epoch  = g_reclaim_epoch.fetch_add(1, std::memory_order_seq_cst) + 1;
    auto&               parked = g_qpending[static_cast<std::size_t>(mine - g_slots.data())];
    for (const Deferred& deferred : batch)
    {
        parked.push_back(QPending{.d = deferred, .epoch = epoch});
    }
    const long n    = static_cast<long>(batch.size());
    const long held = g_qheld.fetch_add(n, std::memory_order_relaxed) + n;
    g_qparked.fetch_add(n, std::memory_order_relaxed);
    for (long peak = g_qheld_peak.load(std::memory_order_relaxed);
         held > peak && !g_qheld_peak.compare_exchange_weak(peak, held, std::memory_order_relaxed);)
    {
    }
    batch.clear();
    q_try_release_own();
}

/// @brief Outside any region (finalize_parallel_phase): release every parked buffer, every thread offline.
inline void q_drain_all() noexcept
{
    if constexpr (qsbr)
    {
        const int nslots = std::min(g_nslots.load(std::memory_order_acquire), slot_count);
        int       online = 0;
        for (int i = 0; i < nslots; ++i)
        {
            auto& parked = g_qpending[static_cast<std::size_t>(i)];
            for (const QPending& pending : parked)
            {
                q_release(pending);
            }
            g_qdrained.fetch_add(static_cast<long>(parked.size()), std::memory_order_relaxed);
            g_qheld.fetch_sub(static_cast<long>(parked.size()), std::memory_order_relaxed);
            parked.clear();
            online += g_slots[i].online.exchange(false, std::memory_order_seq_cst) ? 1 : 0;
        }
        if (online > g_qonline_peak.load(std::memory_order_relaxed))
        {
            g_qonline_peak.store(online, std::memory_order_relaxed);
        }
    }
}

} // namespace shrink_array::read_reclaim

#include "chunk/internal/read_reclaim_inline.h" // the template bodies
