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
 *    sequence number, odd while a read is in flight. Entry: a plain store of
 *    the odd value, then the LIGHT side of the asymmetric Dekker (a compiler
 *    barrier when membarrier() is registered, a seq_cst fence otherwise).
 *    Exit: a release store of the next even value -- it orders the element
 *    load before it, which is what lets the reclaimer trust an even value.
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

#include <array>     // std::array (g_slots)
#include <algorithm> // std::min
#include <atomic>    // std::atomic, std::atomic_thread_fence, std::memory_order
#include <cstdint>   // std::uint64_t
#include <thread>    // std::this_thread::yield
#include <vector>    // std::vector

#include "chunk/config.h"                              // build_config::threaded
#include "chunk/internal/chunk_size_policy.h"          // t_active_chunk_size / t_alloc_key, restored per released block
#include "chunk/internal/cpu_pause.h"                  // CHUNK_CPU_PAUSE
#include "chunk/internal/write_compress_sync_policy.h" // dekker::heavy_barrier / asymmetric_active

#ifndef CHUNK_READ_RECLAIM_SLOTS
    #define CHUNK_READ_RECLAIM_SLOTS 1024
#endif

namespace shrink_array::read_reclaim
{

/// @brief The protocol is compiled in: OpenMP builds only.
inline constexpr bool enabled = build_config::threaded;

/// @brief Reader slots: one per thread that has ever read, a cache line each.
inline constexpr int kSlots = CHUNK_READ_RECLAIM_SLOTS;

/// @brief One reader's sequence number, on a cache line of its own.
struct alignas(64) ReaderSlot
{
    std::atomic<std::uint64_t> seq{0}; ///< Odd while a read is in flight.
};

inline std::array<ReaderSlot, kSlots> g_slots;     ///< The reader slots, claimed in order.
inline std::atomic<int>               g_nslots{0}; ///< Slots claimed so far (may exceed kSlots).
/// @brief Readers past kSlots: counted in one shared RMW (seq_cst), slow but
///        correct. A program with more reader threads than kSlots raises
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
    const int i = g_nslots.fetch_add(1, std::memory_order_acq_rel);
    if (i < kSlots) [[likely]]
    {
        t_slot = &g_slots[i];
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
    ReadSection() noexcept
    {
        if constexpr (enabled)
        {
            ReaderSlot* s = t_slot;
            if (s == nullptr) [[unlikely]]
            {
                s = claim_slot();
            }
            if (s != nullptr) [[likely]]
            {
                slot_ = s;
                seq_  = s->seq.load(std::memory_order_relaxed) + 1; // odd
                s->seq.store(seq_, std::memory_order_relaxed);
                light_fence();
            }
            else
            {
                g_overflow_readers.fetch_add(1, std::memory_order_seq_cst);
            }
        }
    }
    /// @brief Leave: the sequence goes even, released after the element load.
    ~ReadSection()
    {
        if constexpr (enabled)
        {
            if (slot_ != nullptr) [[likely]]
            {
                slot_->seq.store(seq_ + 1, std::memory_order_release); // even: the element load is done
            }
            else
            {
                g_overflow_readers.fetch_sub(1, std::memory_order_release);
            }
        }
    }
    // A scope: neither copied nor moved.
    ReadSection(const ReadSection&)            = delete;
    ReadSection& operator=(const ReadSection&) = delete;
    ReadSection(ReadSection&&)                 = delete;
    ReadSection& operator=(ReadSection&&)      = delete;

  private:
    ReaderSlot*   slot_ = nullptr;
    std::uint64_t seq_  = 0;
};

/**
 * @brief The grace period: every read in flight when it starts has ended when
 *        it returns. The caller has already swapped the pointers it is about
 *        to release.
 */
inline void wait_for_readers() noexcept
{
    dekker::heavy_barrier(); // seq_cst fence + membarrier: orders the CASes before the slot reads
    const int         n    = std::min(g_nslots.load(std::memory_order_acquire), kSlots);
    const ReaderSlot* mine = t_slot;
    for (int i = 0; i < n; ++i)
    {
        if (&g_slots[i] == mine)
        {
            continue;
        }
        const std::uint64_t v = g_slots[i].seq.load(std::memory_order_acquire);
        if ((v & 1U) == 0U)
        {
            continue;
        }
        for (int spins = 0; g_slots[i].seq.load(std::memory_order_acquire) == v;)
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
    PassScope() noexcept
    {
        if constexpr (enabled)
        {
            if (t_batch == nullptr)
            {
                owner_ = true;
                storage().clear();
                t_batch = &storage();
            }
        }
    }
    /// @brief The outermost scope: wait for the readers, then release the batch.
    ~PassScope()
    {
        if constexpr (enabled)
        {
            if (!owner_)
            {
                return;
            }
            t_batch     = nullptr;
            auto& batch = storage();
            if (batch.empty())
            {
                return;
            }
            wait_for_readers();
            const unsigned int saved_size = t_active_chunk_size;
            const unsigned int saved_key  = t_alloc_key;
            for (const Deferred& d : batch)
            {
                t_active_chunk_size = d.chunk_size;
                t_alloc_key         = d.alloc_key;
                d.release(d.chunk, d.thr, d.block);
            }
            t_active_chunk_size = saved_size;
            t_alloc_key         = saved_key;
            batch.clear();
        }
    }
    // A scope: neither copied nor moved.
    PassScope(const PassScope&)            = delete;
    PassScope& operator=(const PassScope&) = delete;
    PassScope(PassScope&&)                 = delete;
    PassScope& operator=(PassScope&&)      = delete;

  private:
    /// @brief Per-thread, reused: a pass does not allocate its batch afresh.
    static std::vector<Deferred>& storage() noexcept
    {
        thread_local std::vector<Deferred> v;
        return v;
    }
    bool owner_ = false;
};

/// @brief Collect @p block instead of releasing it, if a scoped pass is running.
/// @return true if collected (the caller must not release it).
inline bool defer_release(void (*release)(void*, int, void*) noexcept, void* chunk, int thr, void* block)
{
    if constexpr (enabled)
    {
        if (std::vector<Deferred>* b = t_batch; b != nullptr) [[unlikely]]
        {
            b->push_back(Deferred{.release    = release,
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
