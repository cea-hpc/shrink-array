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
 * @par Optimistic reads (chunk_config::optimistic_reads)
 * Where the argument below holds, a pure read does not announce itself: no store,
 * no fence per read. It loads the reclaim epoch (acquire), loads `values` and the
 * element, then loads the epoch again; if a grace period began in between, it reads
 * again, taking the new epoch with acquire, so that it loads the swapped pointer.
 * wait_for_readers() bumps the epoch (seq_cst) after the pass's swaps and before its
 * releases; the releases stay immediate, as with the announce: no memory held longer.
 *  - The re-check must come after the element load. It is ordered there by an address
 *    dependency on the value (ReadSection::retry_after(): the epoch's address is
 *    `&epoch + (v ^ v)`, the xor hidden from the compiler), or, for a block, on the OR
 *    of its values (fold(), AArch64) or by an acquire fence (elsewhere: free on x86).
 *    That is the hardware's ordering (AArch64, x86), not C++'s, as for
 *    rcu_dereference(): formal/optimistic_read.c passes GenMC under ARMv8 and TSO and
 *    fails under RC11 / IMM, as expected. Hence x86-64 and AArch64 only.
 *  - A read that loses the race reads a buffer a pass released, maybe already reused:
 *    it must not fault. Value buffers come from the fast allocator's pools, whose slabs
 *    the chunk layer never unmaps (MADV_DONTNEED keeps the mapping): a stale read
 *    returns garbage that the re-check discards. Hence the fast allocator only. No read
 *    uses a value it loaded from a buffer as an address before its re-check.
 *  - The compressor's own scans (ScanSection) keep the announce: an array torn down
 *    inside a region waits for them (ArrayChunkCore::await_chunk_readers()).
 *  - In C++ terms the stale read is a data race: off under ThreadSanitizer.
 * Elsewhere (CHUNK_OPTIMISTIC_READS=OFF, CHUNK_STD_ALLOC, other architectures, TSan)
 * every pure read announces itself as above.
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
#include <cstddef>     // std::size_t (retry_after, fold)
#include <cstdint>     // std::uint64_t, std::uintptr_t
#include <cstring>     // std::memcpy (retry_after, fold)
#include <type_traits> // std::conditional_t, std::is_trivially_copyable_v
#include <thread>      // std::this_thread::yield
#include <vector>      // std::vector

#include "chunk/config.h"                              // build_config::threaded
#include "chunk/internal/chunk_size_policy.h"          // t_active_chunk_size / t_alloc_key, restored per released block
#include "chunk/internal/cpu_pause.h"                  // CHUNK_CPU_PAUSE
#include "chunk/internal/def_force_inline.h"           // CHUNK_FORCE_INLINE (retry_after, retry, fold)
#include "chunk/internal/write_compress_sync_policy.h" // dekker::heavy_barrier / asymmetric_active

#ifndef CHUNK_READ_RECLAIM_SLOTS
    #define CHUNK_READ_RECLAIM_SLOTS 1024
#endif

namespace shrink_array::read_reclaim
{

/// @brief The protocol is compiled in: OpenMP builds only.
inline constexpr bool enabled = build_config::threaded;

/// @brief Pure reads are optimistic (see the file comment): they announce nothing.
inline constexpr bool optimistic = enabled && chunk_config::optimistic_reads;
static_assert(!optimistic || build_config::use_fast_alloc,
              "optimistic reads need the fast allocator: a free()d buffer may be unmapped");

/// @brief An optimistic block read orders its re-check by the OR of its values (AArch64:
///        an acquire fence is a dmb ishld there, read-bulk +4 % on Grace); elsewhere by an
///        acquire fence (x86: loads are not reordered with loads, the fence is free, and
///        the per-value OR cost read-bulk +40 % under gcc).
#if defined(__aarch64__)
inline constexpr bool fold_block = true;
#else
inline constexpr bool fold_block = false;
#endif

/// @brief Reader slots: one per thread that has ever read, a cache line each.
inline constexpr int slot_count = CHUNK_READ_RECLAIM_SLOTS;

/// @brief One reader's sequence number, on a cache line of its own.
struct alignas(64) ReaderSlot
{
    std::atomic<std::uint64_t> seq{0}; ///< Odd while a read is in flight.
};

inline std::array<ReaderSlot, slot_count> g_slots;     ///< The reader slots, claimed in order.
inline std::atomic<int>                   g_nslots{0}; ///< Slots claimed so far (may exceed slot_count).
/// @brief Readers past slot_count: counted in one shared RMW (seq_cst), slow but
///        correct. A program with more reader threads than slot_count raises
///        CHUNK_READ_RECLAIM_SLOTS.
inline std::atomic<long>        g_overflow_readers{0};
inline thread_local ReaderSlot* t_slot     = nullptr; ///< This thread's slot, once claimed.
inline thread_local bool        t_overflow = false;   ///< This thread found no slot left.

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
 * @brief An announced read: from before the `values` load to after the element
 *        load, the thread's slot is odd (the protocol above).
 * @details Every pure read when reads are not optimistic (ReadSection holds one);
 *          the compressor's scans always (ScanSection).
 */
class AnnouncedSection
{
  public:
    /// @brief Enter: the slot's sequence goes odd, then the light fence.
    AnnouncedSection() noexcept;
    /// @brief Leave: the sequence goes even, released after the element load.
    ~AnnouncedSection();
    // A scope: neither copied nor moved.
    AnnouncedSection(const AnnouncedSection&)            = delete;
    AnnouncedSection& operator=(const AnnouncedSection&) = delete;
    AnnouncedSection(AnnouncedSection&&)                 = delete;
    AnnouncedSection& operator=(AnnouncedSection&&)      = delete;

  private:
    ReaderSlot*   slot_ = nullptr;
    std::uint64_t seq_  = 0;
};

/// @brief A scan of the compressor (CompressionManager start()): always announced.
using ScanSection = AnnouncedSection;

/// @brief Bumped by every grace period (wait_for_readers()), after the pass's swaps
///        and before its releases: what an optimistic read re-checks.
alignas(64) inline std::atomic<std::uint64_t> g_reclaim_epoch{0};
/// @brief Optimistic reads that read again, all threads (counted on the cold path).
alignas(64) inline std::atomic<long> g_optimistic_retries{0};

/// @brief The optimistic reads that read again so far, all threads (0 when reads announce).
[[nodiscard]] inline long optimistic_retries() noexcept
{
    return g_optimistic_retries.load(std::memory_order_relaxed);
}

/// @brief The cold side of a retry: count it, return the new epoch (acquire).
std::uint64_t note_retry() noexcept;

/// @brief Not announced: the ReadSection of an optimistic build.
struct NoAnnounce
{
};

/**
 * @brief Brackets one pure read: from before the `values` load to after the
 *        element load(s).
 * @details Optimistic reads: the entry loads the epoch, and the caller loops
 *          until retry_after() (one value) or retry() (a block, its values
 *          fold()ed) says no grace period began meanwhile. Otherwise the
 *          section announces the read (AnnouncedSection) and both return false:
 *          the same loop runs once.
 */
class ReadSection
{
  public:
    /// @brief Enter: the epoch (optimistic), or the announce.
    ReadSection() noexcept;
    // A scope: neither copied nor moved.
    ReadSection(const ReadSection&)            = delete;
    ReadSection& operator=(const ReadSection&) = delete;
    ReadSection(ReadSection&&)                 = delete;
    ReadSection& operator=(ReadSection&&)      = delete;

    /// @brief True when a grace period began since the entry (or the last retry): the
    ///        caller reads again. The re-check is ordered after the load of @p value by
    ///        an address dependency. False when reads announce.
    template <typename V>
    [[nodiscard]] bool retry_after(const V& value) const noexcept;

    /// @brief retry_after() for a read of several values (a block), every one of them
    ///        fold()ed since the last check. False when reads announce.
    [[nodiscard]] bool retry() const noexcept;

    /// @brief Add @p value's bytes to the block's dependency (fold_block). Nothing otherwise.
    template <typename V>
    void fold(const V& value) const noexcept;

  private:
    /// @brief The announce, when reads are not optimistic: held for its constructor and
    ///        destructor only (RAII), hence maybe_unused.
    [[no_unique_address, maybe_unused]] std::conditional_t<optimistic, NoAnnounce, AnnouncedSection> announce_;
    mutable std::uint64_t                                                                            seq_ = 0;
    mutable std::uintptr_t                                                                           acc_ = 0;
};

/**
 * @brief The grace period: every read in flight when it starts has ended when
 *        it returns. The caller has already swapped the pointers it is about
 *        to release.
 */
inline void wait_for_readers() noexcept
{
    if constexpr (optimistic)
    {
        // After the caller's swaps, before its releases: an optimistic read whose
        // element came from a buffer released after this point sees the new epoch.
        g_reclaim_epoch.fetch_add(1, std::memory_order_seq_cst);
    }
    dekker::heavy_barrier(); // seq_cst fence + membarrier: orders the CASes before the slot reads
    const int         nslots = std::min(g_nslots.load(std::memory_order_acquire), slot_count);
    const ReaderSlot* mine   = t_slot;
    for (int i = 0; i < nslots; ++i)
    {
        if (&g_slots[i] == mine)
        {
            continue;
        }
        const std::uint64_t seq = g_slots[i].seq.load(std::memory_order_acquire);
        if ((seq & 1U) == 0U)
        {
            continue;
        }
        for (int spins = 0; g_slots[i].seq.load(std::memory_order_acquire) == seq;)
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

} // namespace shrink_array::read_reclaim

#include "chunk/internal/read_reclaim_inline.h" // the template bodies
