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

#include <exception>   // std::exception (deferred enqueue)
#include <memory>      // std::make_unique
#include <type_traits> // std::remove_reference_t

#include "chunk/internal/chunk_base.h"                 // the declarations these bodies belong to
#include "chunk/config.h"                              // build_config::use_fast_alloc, chunk_config::
#include "chunk/internal/chunk_stats.h"                // increm_chunk_read_sync / _cross (CHUNK_STATS)
#include "chunk/internal/pool_ledger.h"                // CHUNK_LEDGER_PARK / CHUNK_LEDGER_RELEASE
#include "chunk/PMP_CHUNK.h"                           // CHUNK_PMP_IN_PARALLEL_REGION
#include "chunk/internal/deferred_op_pool.h"           // DeferredOperationBase, DeferredOpDeleter, DeferredOpPtr
#include "chunk/internal/deferred_ops_storage.h"       // DeferredOpsStorage
#include "chunk/internal/freepath_counters.h"          // CHUNK_FREEPATH_INC
#include "chunk/internal/invariant_check.h"            // CHUNK_INVARIANT_FAILED
#include "chunk/internal/read_reclaim.h"               // read_reclaim::defer_release
#include "chunk/internal/value_slab.h"                 // value_slab_request (the value slabs' size)
#include "fast_allocators/internal/svalloc.h"          // svfree
#include "chunk/internal/write_compress_sync_policy.h" // WriteCompressSyncPolicy

namespace shrink_array
{

template <typename T, typename Derived>
void ChunkBase<T, Derived>::free_block_to_pool(this Derived& self, [[maybe_unused]] int thr_rank, T* val)
{
    // A pass that may run while other threads read (read_reclaim::PassScope:
    // perform_compression, compress() inside a region) collects the buffer
    // instead: a reader may still hold it between its `values` load and its
    // element load. The pass releases it here, after one grace period.
    if (read_reclaim::defer_release(&ChunkBase::release_deferred_block, &self, thr_rank, val))
    {
        return;
    }

    if constexpr (build_config::use_fast_alloc)
    {
        // The block's slab says which array it belongs to (owner_key): the reuse list and
        // budget are that array's, on this rank. Key 0 is the shared pool.
        const auto&    mgr               = self.get_manager();
        auto&          allocator         = self.get_chunk_allocator(thr_rank);
        auto*          slab              = allocator.slab_of(val); // resolved ONCE; every route below reuses it
        const unsigned key               = slab ? slab->owner_key : 0u;
        auto&          ll_block_free_thr = mgr.get_list_block_free_thr(thr_rank, key);

        if (static_cast<int>(ll_block_free_thr.size()) < mgr.get_size_buffer_reuse(key))
        {
            // Ledger: LIVE -> parked (own free-list). Trips on double-park.
            CHUNK_LEDGER_PARK(val, "free_block_to_pool.park_own");
            CHUNK_FREEPATH_INC(park_own);
            ll_block_free_thr.push_back(val);
        }
        else
        {
            if (allocator.holds_live(slab))
            {
                // Ledger: LIVE -> FREE. Trips on double-free.
                CHUNK_LEDGER_RELEASE(val, "free_block_to_pool.free_elt_pow2");
                CHUNK_FREEPATH_INC(freed);
                allocator.free_elt_pow2(val, slab);
            }
            else if (slab && slab->owner && (slab->owner_rank < 0 || !CHUNK_PMP_IN_PARALLEL_REGION))
            {
                // Owned by another instance (another key, or another rank while
                // no region is open): its chains are safe to touch -- free it
                // where it came from instead of parking it for the drain.
                CHUNK_FREEPATH_INC(freed);
                CHUNK_LEDGER_RELEASE(val, "free_block_to_pool.free_to_owner");
                using AllocT = std::remove_reference_t<decltype(allocator)>;
                const_cast<AllocT*>(static_cast<const AllocT*>(slab->owner))->free_elt_pow2(val, slab);
            }
            else
            {
                if (CHUNK_PMP_IN_PARALLEL_REGION)
                {
                    // Ledger: LIVE -> parked (own free-list, foreign-slab block).
                    CHUNK_LEDGER_PARK(val, "free_block_to_pool.park_own_foreign");
                    ll_block_free_thr.push_back(val);
                }
                else
                {
                    // Do NOT push a foreign block onto ANOTHER thread's free-list: the
                    // unsynchronised push races with the owner's own pop/push (vector
                    // corruption), and a later free_elt_pow2 would route the block to the
                    // wrong allocator's lists (delete / munmap of a live block). Same rule
                    // as recycle() on the op pool side.
                    //
                    // Keep it on the LOCAL free-list: is_free_able_elt is false for it, so
                    // it is never freed locally, only reused directly; the owning slab
                    // releases it at teardown. Bounded leak, not a UAF.
                    CHUNK_FREEPATH_INC(kept_nomigrate);
                    CHUNK_LEDGER_PARK(val, "free_block_to_pool.keep_local_no_migrate");
                    ll_block_free_thr.push_back(val);
                }
            }
        }
    }
    else
    {
        // Ledger LIVE -> FREE, before the svfree while the pointer is still ours: otherwise
        // malloc re-hands the page to a later allocation and on_block_alloc sees it still
        // LIVE (spurious invariant failure). free_values is the other such path.
        CHUNK_LEDGER_RELEASE(val, "free_block_to_pool.fast_alloc::svfree");
        fast_alloc::svfree(val);
    }
}

// Bodies of ChunkDeferredOpsPolicy<T, Derived, true> (chunk_base.h). Every name they use is
// declared in every build, and the policy is never instantiated in a serial build. They cast
// `this` to Derived* directly, not through an explicit object parameter: these members are
// called from sibling policies whose *this is not a Derived (see chunk_base.h).

template <typename T, typename Derived>
void ChunkDeferredOpsPolicy<T, Derived, true>::to_be_queued_for_deferred_processing(
    int                                                thr_rank,
    int                                                num,
    CompressionManager<T, Derived>::CmpDeferredOpBase* op) const
{
    ensure_deferred_array_initialized();

    // Cached per thread: no guard check per call; the singleton outlives every thread.
    thread_local auto* storage = &DeferredOpsStorage<PerThreadMap>::instance();

    auto* arr = storage->get(); // simple acquire load, always fresh

    auto* chunk_ptr = static_cast<Derived*>(const_cast<ChunkDeferredOpsPolicy*>(this));

    // ── Enqueue vs cross-thread flush: PER-RANK shard lock ────────────────
    // The requirement is pairwise (flusher of rank r <-> writer of rank r), which is what
    // xflush_lock_rank[thr_rank] provides. Do not replace it by a back-off on a GLOBAL flush
    // counter: under read-heavy phases flushes are quasi-permanent and writers convoy in a
    // spin while pending ops cannot drain (it dominated the profile at high thread counts).
    // Owner-recursive: a drain's write_body may re-enter here on the own rank.
    //
    // announce_write/end_write are the writer half of the compressor-yield protocol (start()
    // spins on any_thread_writing), orthogonal to map safety.
    using Policy = WriteCompressSyncPolicy<typename Derived::ProxyType>;

    CompressionManager<T, Derived>::enqueue_lock(thr_rank);
    Policy::announce_write(thr_rank, chunk_ptr); // flag + seq_cst fence, one Dekker verb

    // The container steps below allocate, and ankerl's maps can throw. Callers are noexcept,
    // so an escaping exception would be a silent std::terminate: report where instead.
    try
    {
        PerThreadMap& thread_map = arr[thr_rank];

        auto& ops_map_ptr = thread_map[chunk_ptr];

        if (!ops_map_ptr)
        {
            ops_map_ptr = std::make_unique<UnorderedMapDeferredOperations>();
        }

        // ── Same-offset re-enqueue: REPLACE, do not double-count ──────────────
        // A second deferred write to the SAME offset on the SAME rank supersedes the queued
        // one. Do not use a blind `map[num] = op`: the counters would count both, drift up and
        // never return to zero (any_pending_ops pinned), and the replaced op would leak.
        // Publish/inc only for NEW entries; for a replacement the count is already right.
        const auto it_existing = ops_map_ptr->find(num);
        if (it_existing != ops_map_ptr->end())
        {
            it_existing->second->destroy(); // idempotent teardown of the superseded op
            it_existing->second = DeferredOpPtr(static_cast<DeferredOperationBase*>(op), DeferredOpDeleter{});
        }
        else
        {
            // Publish the per-chunk counter BEFORE inserting: the read gate is
            // `dirty && pending_any()`, so pending>0 must hold for the op's whole lifetime
            // (flushers see the op via the Dekker write flag, not this counter).
            chunk_ptr->pending_publish(thr_rank); // owner-tagged; no-op when counterless
            (*ops_map_ptr)[num] = DeferredOpPtr(static_cast<DeferredOperationBase*>(op), DeferredOpDeleter{});
            // Publish globally BEFORE releasing the writing flag, so a flusher that
            // observes !is_thread_writing(r) and proceeds to scan also observes it.
            CompressionManager<T, Derived>::inc_pending_ops(thr_rank);
        }
    }
    catch (const std::exception& e)
    {
        CHUNK_INVARIANT_FAILED("deferred enqueue failed: {} (rank {}, chunk {}, offset {})", e.what(), thr_rank,
                               chunk_ptr, num);
    }

    Policy::end_write(thr_rank);
    CompressionManager<T, Derived>::enqueue_unlock(thr_rank);

    // ── Bounded backlog drain ─────────────────────────────────────────────────
    // Writes do not flush, so a write-only phase could grow this rank's backlog without bound.
    // Past the threshold, drain OUR OWN rank -- after end_write and enqueue_unlock (no flag or
    // lock held), and never while replaying a flush (flush_deferred re-enters here).
    //
    // Not while a compressor is stealing: the write was deferred because of it, and
    // flush_deferred() would park the writer for the whole compression. The fast path
    // (write_with_optional_lock_impl) honours the threshold on the next direct write.
    {
        using CM = CompressionManager<T, Derived>;
        if (!CM::in_flush_replay && !CM::is_hard_compression_inflight() &&
            CM::pending_ops_rank(thr_rank) > build_config::deferred_drain_threshold) [[unlikely]]
        {
            (void)CM::flush_deferred(thr_rank);
        }
    }
}

template <typename T, typename Derived>
void ChunkDeferredOpsPolicy<T, Derived, true>::wait_for_availability([[maybe_unused]] int num) const
{
    // No pre-wait for compressors here, and never on a predicate that counts DRIVERS: under
    // load it is quasi-permanently non-zero, every reader piles into the spin, and it is the
    // deadlock anti-pattern documented on hard_inflight_. flush_chunk_all_ranks() already
    // provides the only ordering that matters: begin_flush(self_rank) blocks any future
    // compressor start(), then the drain waits out in-flight stealers (driver-only threads
    // steal nothing).

    // CHUNK_STATS: the SLOW sync path. read_sync / (read + read_cst) is the share of accesses
    // that pay synchronisation at all.
    if constexpr (chunk_config::stats)
    {
        increm_chunk_read_sync<Derived>();
    }

    auto* chunk_ptr = static_cast<Derived*>(const_cast<ChunkDeferredOpsPolicy*>(this));

    // ── Stage 1: drain OUR OWN rank only ───────────────────────────────────
    // sync_chunk runs on the write operators too, so the dominant case is a thread re-touching
    // a chunk whose pending op is ITS OWN. Draining self_rank costs one (recursive, usually
    // uncontended) lock; do not go straight to the full walk, which makes every written element
    // pay O(threads) lock traffic. The per-chunk counter (pending_any()) says when it sufficed.
    static_cast<void>(CompressionManager<T, Derived>::flush_chunk_all_ranks(chunk_ptr, /*self_only=*/true));
    if (!chunk_ptr->pending_any())
    {
        return; // every pending op was ours — no cross-rank walk needed
    }

    // ── Stage 2: cross-thread flush ────────────────────────────────────────────
    // Foreign ops remain: flush this chunk across ALL ranks under the
    // per-rank locks; an own-rank-only flush would lose other ranks' writes.
    // CHUNK_STATS: read_sync_cross / read_sync tells how often stage 1 was not enough.
    if constexpr (chunk_config::stats)
    {
        increm_chunk_read_sync_cross<Derived>();
    }
    static_cast<void>(CompressionManager<T, Derived>::flush_chunk_all_ranks(chunk_ptr));
}

// ---------------------------------------------------------------------------
// shrink_array::DeferredReadGate<Derived, DeferReadGate::precise>
// ---------------------------------------------------------------------------

template <typename Derived>
void DeferredReadGate<Derived, DeferReadGate::precise>::pending_publish(this Derived&        self,
                                                                        [[maybe_unused]] int thr) noexcept
{
    auto& pending = self.pending_ops_;
    {
        const uint32_t me =
            (static_cast<uint32_t>(thr) + 1u < owner_multi) ? static_cast<uint32_t>(thr) + 1u : owner_multi;
        auto cur = static_cast<uint32_t>(pending.load(std::memory_order_acquire));
        for (;;)
        {
            // count = SIGNED 20-bit (two's complement): a transient
            // over-retire goes NEGATIVE (pending_any() false) and
            // self-heals at the next publish. Never clamp it: a clamp
            // leaves the count inflated after an over-retire, so the gates
            // stick and the drains run hyperactive.
            const int32_t  cnt      = sign_extend_count(cur);
            const uint32_t own      = cur >> count_bits;
            const uint32_t nown     = (cnt <= 0 || own == me) ? me : owner_multi;
            const uint32_t ncnt     = static_cast<uint32_t>(cnt + 1) & count_mask;
            const uint32_t nw       = (nown << count_bits) | ncnt;
            auto           expected = static_cast<int32_t>(cur);
            if (pending.compare_exchange_weak(expected, static_cast<int32_t>(nw), std::memory_order_acq_rel,
                                              std::memory_order_acquire))
            {
                return;
            }
            cur = static_cast<uint32_t>(expected);
        }
    }
}

template <typename Derived>
void DeferredReadGate<Derived, DeferReadGate::precise>::pending_retire(this Derived& self, int32_t count) noexcept
{
    auto& pending = self.pending_ops_;
    {
        auto cur = static_cast<uint32_t>(pending.load(std::memory_order_acquire));
        for (;;)
        {
            const int32_t  cnt      = sign_extend_count(cur);
            const int32_t  ncnt     = cnt - count;                              // may go NEGATIVE: two's
            const uint32_t bits     = static_cast<uint32_t>(ncnt) & count_mask; // complement in 20 bits
            const uint32_t nw       = (ncnt <= 0) ? bits                        // owner cleared while <= 0
                                                  : ((cur >> count_bits) << count_bits) | bits;
            auto           expected = static_cast<int32_t>(cur);
            if (pending.compare_exchange_weak(expected, static_cast<int32_t>(nw), std::memory_order_acq_rel,
                                              std::memory_order_acquire))
            {
                // Check builds stop on an over-retire: negative, pending_any()
                // is false, and a publish from -1 leaves 0 with an op queued
                // (what the force flushes work around). publish runs BEFORE
                // the insert, so this is a double count, not a race. Release
                // builds keep the self-healing semantics above.
                if constexpr (build_config::any_check)
                {
                    if (ncnt < 0) [[unlikely]]
                    {
                        CHUNK_INVARIANT_FAILED("pending_retire: chunk {} pending count {} -> {} (retired {}): "
                                               "more ops retired than published -- a double count",
                                               &self, cnt, ncnt, count);
                    }
                }
                return;
            }
            cur = static_cast<uint32_t>(expected);
        }
    }
}

// ---------------------------------------------------------------------------
// shrink_array::ChunkAllocPolicy<T, Derived, true>
// ---------------------------------------------------------------------------

template <typename T, typename Derived>
template <typename InitReused, typename InitFresh>
T* ChunkAllocPolicy<T, Derived, true>::allocate_chunk_internal(int        thr_rank,
                                                               InitReused init_reused,
                                                               InitFresh  init_fresh)
{
    const auto& self       = static_cast<const Derived&>(*this);
    T*          values_loc = nullptr;

    // The per-thread reuse list is not always sized here (fast-alloc builds
    // never park in the legacy store; a geometry born from a re-chunk has no
    // per-size list): find_list_block_free_thr() is bounds-checked and
    // answers nullptr for a slot never sized. Do not index the list directly.
    //
    // The reuse list of THIS array on this rank (thread_alloc_key(), primed by
    // the proxy next to the geometry): a block taken here comes from the
    // array's own slabs, so affinity survives recycling.
    const unsigned int alloc_key = thread_alloc_key();
    CHUNK_FREEPATH_ALLOC(alloc_key);
    // Churn measurement (churn_counters.h): one materialisation of the
    // array this key names -- rank-local, dynamic builds only.
    churn::note_materialisation<T>(thr_rank, alloc_key);
    // One lookup serves both the emptiness test and the pop.
    if (auto* const free_list = self.get_manager().find_list_block_free_thr(thr_rank, alloc_key);
        free_list != nullptr && !free_list->empty())
    {
        auto& ll_block_free_thr = *free_list;
        if constexpr (chunk_config::stats)
        {
            increm_chunk_list<Derived>();
            add_chunk_bytes_reused<Derived>(static_cast<long>(sizeof(T)) *
                                            static_cast<long>(Derived::storage_units(alloc_chunk_count())));
        }
        values_loc = ll_block_free_thr.back();
        CHUNK_CHECK_VALUES_PTR(values_loc, "freelist.pop");
        ll_block_free_thr.pop_back();
        // Ledger: parked -> LIVE, recorded BEFORE the memset so a concurrent
        // free of the same block trips on the state mismatch instead of
        // racing the init undetected.
        CHUNK_LEDGER_POP(values_loc, "allocate_chunk_internal.freelist_pop");
        init_reused(values_loc);
    }
    else
    {
        // Allocate new memory
        if constexpr (chunk_config::stats)
        {
            increm_chunk_alloc<Derived>();
            add_chunk_bytes_alloc<Derived>(static_cast<long>(sizeof(T)) *
                                           static_cast<long>(Derived::storage_units(alloc_chunk_count())));
        }
        if constexpr (build_config::use_fast_alloc)
        {
            values_loc = self.get_chunk_allocator(thr_rank).alloc();
            CHUNK_CHECK_VALUES_PTR(values_loc, "alloc.fast");
            CHUNK_LEDGER_ALLOC(values_loc, "allocate_chunk_internal.fast_alloc");
            init_fresh(values_loc);
        }
        else
        {
            values_loc = fast_alloc::svcalloc_align<T>(Derived::storage_units(alloc_chunk_count()));
            CHUNK_CHECK_VALUES_PTR(values_loc, "alloc.fast_alloc::svcalloc");
            CHUNK_LEDGER_ALLOC(values_loc, "allocate_chunk_internal.fast_alloc::svcalloc");
            init_fresh(values_loc);
        }
    }
    if constexpr (chunk_config::stats)
    {
        increm_chunk_create<Derived>();
    }

    // Debug ring log: the value buffer's address, so dump_ring_for_addr()
    // can tell an op block from a value block at the same address.
    if constexpr (chunk_config::ring_log_debug || chunk_config::ring_log_op_only)
    {
        ring_log_op(get_tid(), EventCode::value_alloc, 0, static_cast<const void*>(values_loc));
    }

    return values_loc;
}

// ---------------------------------------------------------------------------
// shrink_array::ChunkAllocPolicy<T, Derived, false>
// ---------------------------------------------------------------------------

template <typename T, typename Derived>
template <typename InitReused, typename InitFresh>
T* ChunkAllocPolicy<T, Derived, false>::allocate_chunk_internal([[maybe_unused]] int        thr_rank,
                                                                [[maybe_unused]] InitReused init_reused,
                                                                InitFresh                   init_fresh)
{
    const auto& self       = static_cast<const Derived&>(*this);
    T*          values_loc = nullptr;

    if constexpr (chunk_config::stats)
    {
        increm_chunk_create<Derived>();
        increm_chunk_alloc<Derived>();
        add_chunk_bytes_alloc<Derived>(static_cast<long>(sizeof(T)) *
                                       static_cast<long>(Derived::storage_units(alloc_chunk_count())));
    }

    if constexpr (build_config::use_fast_alloc)
    {
        values_loc = self.get_chunk_allocator(0).alloc();
        CHUNK_CHECK_VALUES_PTR(values_loc, "alloc.fast_serial");
        // Ledger: FREE -> LIVE, as in the OpenMP variant.
        CHUNK_LEDGER_ALLOC(values_loc, "allocate_chunk_internal.fast_serial");
        init_fresh(values_loc);
    }
    else
    {
        values_loc = fast_alloc::svcalloc_align<T>(Derived::storage_units(alloc_chunk_count()));
        CHUNK_CHECK_VALUES_PTR(values_loc, "alloc.svcalloc_serial");
        CHUNK_LEDGER_ALLOC(values_loc, "allocate_chunk_internal.svcalloc_serial");
        init_fresh(values_loc);
    }


    return values_loc;
}

// ---------------------------------------------------------------------------
// shrink_array::ChunkDeferredOpsPolicy<T, Derived, true>
// ---------------------------------------------------------------------------

template <typename T, typename Derived>
[[nodiscard]] bool ChunkDeferredOpsPolicy<T, Derived, true>::compare_and_swap_gen(T* values_loc,
                                                                                  T* m_ptr_cst_val) noexcept
{
    auto& self = static_cast<Derived&>(*this);
    CHUNK_CHECK_VALUES_PTR_C(values_loc, "CAS.values_loc", &self);
    CHUNK_CHECK_VALUES_PTR_C(m_ptr_cst_val, "CAS.expected_cst", &self);

    if (self.values_.load(std::memory_order_relaxed) != m_ptr_cst_val) [[unlikely]]
    {
        return true; // failed (already changed)
    }

    return !self.values_.compare_exchange_strong(m_ptr_cst_val, values_loc, std::memory_order_acq_rel,
                                                 std::memory_order_acquire);
}

template <typename T, typename Derived>
void ChunkDeferredOpsPolicy<T, Derived, true>::recup_value(T* values_loc, int thr_rank) const
{
    const auto& self = static_cast<const Derived&>(*this);
    // Ledger: LIVE -> parked. This block was just allocated by create_value
    // and lost the install CAS; parking a block that is not LIVE (e.g. one
    // another thread already owns) trips here.
    CHUNK_LEDGER_PARK(values_loc, "recup_value");
    // Back to the list it was taken from / would have been taken from: the
    // array's, on this rank.
    auto& ll_block_free_thr = self.get_manager().get_list_block_free_thr(thr_rank, thread_alloc_key());
    ll_block_free_thr.push_back(values_loc);
}

// ---------------------------------------------------------------------------
// shrink_array::ChunkDynamicAllocPolicy<T, Derived, true>
// ---------------------------------------------------------------------------

template <typename T, typename Derived>
void ChunkDynamicAllocPolicy<T, Derived, true>::register_chunk_size(unsigned int size,
                                                                    int          max_thr,
                                                                    int          block_size,
                                                                    unsigned int key)
{
    if (allocator_pool_->contains(pool_key(key, size)))
    {
        return; // already registered
    }
    // Built whole (an allocator is not movable); moving the vector into the
    // map moves its buffer, not the allocators.
    std::vector<TCTypeCAllocator<T>> arr(static_cast<std::size_t>(max_thr));
    // One slab rule for every allocator of T, whatever its chunk size (value_slab.h).
    const auto slab_bytes =
        static_cast<uintptr_t>(value_slab_request<T>(block_size, Derived::storage_units(cst_buffer_size) * sizeof(T)));
    for (int thr = 0; thr < max_thr; ++thr)
    {
        auto& allocator = arr[static_cast<std::size_t>(thr)];
        allocator.set_rank(thr); // stamped into every slab as owner_rank
        allocator.set_key(key);  // ... and owner_key
        allocator.init(static_cast<int>(Derived::storage_units(size)), slab_bytes, -1);
    }
    allocator_pool_->emplace(pool_key(key, size), std::move(arr));
}

template <typename T, typename Derived>
void ChunkDynamicAllocPolicy<T, Derived, true>::register_chunk_size_serial(unsigned int size, int block_size)
{
    if (allocator_pool_serial_->contains(size))
    {
        return; // already registered
    }
    auto allocator = std::make_unique<TCTypeAllocator<T>>();
    // Value arena, explicitly: TCAllocator defaults to the Op arena, and a
    // value pool sharing it with the deferred-op pool lets a freed slab of
    // one be reobtained by the other, corrupting both pools' slab lists.
    allocator->set_arena(fast_alloc::ArenaKind::value);
    allocator->init(static_cast<size_t>(Derived::storage_units(size)), static_cast<uintptr_t>(block_size), -1);
    allocator_pool_serial_->emplace(size, std::move(allocator));
}

template <typename T, typename Derived>
auto& ChunkDynamicAllocPolicy<T, Derived, true>::route_omp_dynamic_alloc(int thr_rank, unsigned int key) noexcept
{
    const unsigned int sz = thread_chunk_size();
    if (const auto it = allocator_pool_->find(pool_key(key, sz)); it != allocator_pool_->end())
    {
        return it->second[thr_rank];
    }
    // Unregistered KEY at a registered size: the shared set of that size is
    // the right fallback -- allocator_data below is the legacy single pool,
    // whose geometry has nothing to do with `sz`.
    if (key != 0)
    {
        if (const auto it0 = allocator_pool_->find(pool_key(0u, sz)); it0 != allocator_pool_->end())
        {
            return it0->second[thr_rank];
        }
    }
    // Fallback: not registered (not expected on the alloc path once
    // set_chunk_size ran): the legacy single pool, so the free path and
    // single-size use keep working.
    return (*Derived::allocator_data)[thr_rank];
}

template <typename T, typename Derived>
auto& ChunkDynamicAllocPolicy<T, Derived, true>::route_serial_dynamic_alloc() noexcept
{
    const unsigned int sz = thread_chunk_size();
    if (const auto it = allocator_pool_serial_->find(sz); it != allocator_pool_serial_->end())
    {
        return *it->second; // deref unique_ptr
    }
    // Fallback: unregistered size -> the legacy single allocator, so
    // single-size use and the free path keep working.
    return *Derived::allocator_data;
}

// ---------------------------------------------------------------------------
// shrink_array::ChunkBase<T, Derived>
// ---------------------------------------------------------------------------

template <typename T, typename Derived>
[[nodiscard]] T* ChunkBase<T, Derived>::values_load() const noexcept
{
    if constexpr (build_config::threaded)
    {
        return values_.load(std::memory_order_acquire);
    }
    else
    {
        return values_;
    }
}

template <typename T, typename Derived>
void ChunkBase<T, Derived>::values_store(T* new_values) noexcept
{
    CHUNK_CHECK_VALUES_PTR_C(new_values, "values_store", this);
    if constexpr (build_config::threaded)
    {
        values_.store(new_values, std::memory_order_release);
    }
    else
    {
        values_ = new_values;
    }
}

template <typename T, typename Derived>
[[nodiscard]] ChunkBase<T, Derived>::ConstantScan ChunkBase<T, Derived>::scan_constants(this const Derived& self,
                                                                                        InT<T>              first_v,
                                                                                        InT<T>              second_v,
                                                                                        unsigned int count) noexcept
{
    // Block-wise, branch-free inside a block: an early exit per element
    // keeps the compiler from vectorising the walk, and most scans run to
    // the end anyway (the chunk IS constant). Inside a block the two
    // verdicts are OR reductions of comparisons, which vectorise (AVX2 /
    // NEON); the early exit survives between blocks, so a chunk disproved
    // at its first element still costs one block, not the whole chunk.
    const T* vals = self.values_load();
    if constexpr (is_packed)
    {
        // One compare per 64 elements (bit_words.h).
        const bits::Uniform scan     = bits::scan_words(internal::as_words(vals), count);
        const auto          all_of_v = [scan](bool bit) noexcept
        { return scan == (bit ? bits::Uniform::all_true : bits::Uniform::all_false); };
        if constexpr (chunk_config::stats)
        {
            add_chunk_scan_read<Derived>(static_cast<long>(count));
        }
        ConstantScan packed{all_of_v(first_v), all_of_v(second_v)};
        if (packed.first)
        {
            packed.second = false;
        }
        return packed;
    }
    // Coinciding constants (frequent in zero_any: second_v is values[0], and a
    // sparse chunk often starts at first_value): one compare per element,
    // not two identical ones. Same verdict -- `first` only -- as below.
    if (first_v == second_v)
    {
        return ConstantScan{self.is_constant_to(first_v, count), false};
    }
    ConstantScan scan{true, true};
    unsigned int i = 0;
    while (i < count)
    {
        const unsigned int block_end  = std::min(count, i + scan_block);
        unsigned int       not_first  = 0;
        unsigned int       not_second = 0;
        for (unsigned int j = i; j < block_end; ++j)
        {
            // raw-read: compressor scan of a chunk this pass claimed (steal mark), or a sequential caller
            not_first |= static_cast<unsigned int>(vals[j] != first_v);
            not_second |= static_cast<unsigned int>(vals[j] != second_v);
        }
        i           = block_end;
        scan.first  = scan.first && not_first == 0;
        scan.second = scan.second && not_second == 0;
        if (!scan.first && !scan.second)
        {
            break;
        }
        if (scan.first != scan.second)
        {
            // One hypothesis left (zero_any: from the first block, its second_v
            // being element 0): one compare per element for the rest, not two.
            const ConstantWalk rest = walk_constant(vals, scan.first ? first_v : second_v, i, count);
            i                       = rest.end;
            scan.first              = scan.first && rest.holds;
            scan.second             = scan.second && rest.holds;
            break;
        }
    }
    if constexpr (chunk_config::stats)
    {
        add_chunk_scan_read<Derived>(static_cast<long>(i)); // block granularity
    }
    if (scan.first)
    {
        scan.second = false;
    }
    return scan;
}

template <typename T, typename Derived>
[[nodiscard]] bool
ChunkBase<T, Derived>::is_constant_to(this const Derived& self, InT<T> value, unsigned int count) noexcept
{
    if constexpr (is_packed)
    {
        if constexpr (chunk_config::stats)
        {
            add_chunk_scan_read<Derived>(static_cast<long>(count));
        }
        return bits::scan_words(internal::as_words(self.values_load()), count) ==
               (static_cast<bool>(value) ? bits::Uniform::all_true : bits::Uniform::all_false);
    }
    const ConstantWalk walk = walk_constant(self.values_load(), value, 0, count);
    if constexpr (chunk_config::stats)
    {
        add_chunk_scan_read<Derived>(static_cast<long>(walk.end)); // block granularity
    }
    return walk.holds;
}

template <typename T, typename Derived>
[[nodiscard]] ChunkBase<T, Derived>::ConstantWalk
ChunkBase<T, Derived>::walk_constant(const T* vals, InT<T> value, unsigned int from, unsigned int count) noexcept
{
    // Same shape as scan_constants(): branch-free blocks, exit between them.
    unsigned int i = from;
    while (i < count)
    {
        const unsigned int block_end = std::min(count, i + scan_block);
        unsigned int       bad       = 0;
        for (unsigned int j = i; j < block_end; ++j)
        {
            // raw-read: compressor scan of a chunk this pass claimed (steal mark), or a sequential caller
            bad |= static_cast<unsigned int>(vals[j] != value);
        }
        i = block_end;
        if (bad != 0)
        {
            return ConstantWalk{false, i};
        }
    }
    return ConstantWalk{true, count};
}

template <typename T, typename Derived>
[[nodiscard]] T ChunkBase<T, Derived>::read_value(this const Derived& self, int offset, const T* offset_slot) noexcept
    requires has_motif_state_v<Derived>
{
    if constexpr (chunk_config::stats)
    {
        if (self.is_cst_val1())
        {
            increm_chunk_read_cst<Derived>();
        }
        else
        {
            increm_chunk_read<Derived>();
        }
    }
    return self.element_at(offset, offset_slot);
}

template <typename T, typename Derived>
[[nodiscard]] T ChunkBase<T, Derived>::read_value(this const Derived& self, int offset) noexcept
{
    if constexpr (chunk_config::stats)
    {
        // Split the read by what served it: a materialised chunk vs a
        // compressed one (values points at the constant marker). read_cst /
        // (read + read_cst) is the compression hit rate on reads.
        // is_cst_val1(), not written(): zero_one_value::Chunk::written() (no arg) aborts on
        // non-bool types -- it needs the second-value index a read lacks.
        if (self.is_cst_val1())
        {
            increm_chunk_read_cst<Derived>();
        }
        else
        {
            increm_chunk_read<Derived>();
        }
    }
    return self.element_at(offset);
}

template <typename T, typename Derived>
T ChunkBase<T, Derived>::operator[](this const Derived& self, int offset) noexcept
{
    if constexpr (chunk_config::stats)
    {
        // Same split as read_value().
        if (self.is_cst_val1())
        {
            increm_chunk_read_cst<Derived>();
        }
        else
        {
            increm_chunk_read<Derived>();
        }
    }

    // ONE element_at(): maybe() is a no-op outside check builds, but its
    // argument is still evaluated -- a whole read section (two sequence
    // stores, one of them a release, and an acquire load of `values`) whose
    // ordering the compiler must keep. Every read paid two (perf annotate,
    // Grace: the release store of the second section held the read).
    const T value = self.element_at(offset);
    chunk_config::CheckNaN<chunk_config::do_check>::maybe(value, static_cast<long>(offset));
    return value;
}

template <typename T, typename Derived>
void ChunkBase<T, Derived>::read_block(this const Derived& self, int first, int count, T* out) noexcept
{
    if constexpr (has_motif_state_v<Derived>)
    {
        // A motif chunk needs its offset slot: read_block(first, count, out, slot).
        if (Derived::is_motif_ptr(self.values_load())) [[unlikely]]
        {
            CHUNK_INVARIANT_FAILED("read_block: a motif chunk read without its offset slot");
        }
    }
    const read_reclaim::ReadSection section;
    const T* const                  vals = self.values_load();
    test_hooks::maybe_read_delay(); // TEST KNOB, as in element_at_unguarded
    if constexpr (is_zero_any_family_v<Derived>)
    {
        // Same single load as element_at_unguarded: a chunk compressed to a
        // second constant points at an identifier, not at values.
        if (self.is_cst_val2_ptr_of(vals)) [[unlikely]]
        {
            const T value = self.second_value_from_ptr(vals);
            for (int j = 0; j < count; ++j)
            {
                out[j] = value;
            }
            return;
        }
    }
    for (int j = 0; j < count; ++j)
    {
        // raw-read: read_block -- under its own ReadSection
        out[j] = load_at<false>(vals, first + j);
    }
}

template <typename T, typename Derived>
void ChunkBase<T, Derived>::read_block(this const Derived& self,
                                       int                 first,
                                       int                 count,
                                       T*                  out,
                                       const T*            offset_slot) noexcept
    requires has_motif_state_v<Derived>
{
    const read_reclaim::ReadSection section;
    for (;;)
    {
        const T* const vals = self.values_load();
        test_hooks::maybe_read_delay(); // TEST KNOB, as in element_at_unguarded
        if (self.is_cst_val2_ptr_of(vals)) [[unlikely]]
        {
            const T value = self.second_value_from_ptr(vals);
            for (int j = 0; j < count; ++j)
            {
                out[j] = value;
            }
            return;
        }
        if (Derived::is_motif_ptr(vals)) [[unlikely]]
        {
            T off{};
            if (!self.motif_offset(vals, offset_slot, off)) [[unlikely]]
            {
                continue; // the pointer moved: the slot may hold another state's value (F5)
            }
            for (int j = 0; j < count; ++j)
            {
                out[j] = Derived::motif_value(vals, first + j, off);
            }
            return;
        }
        for (int j = 0; j < count; ++j)
        {
            // raw-read: read_block -- under its own ReadSection
            out[j] = load_at<false>(vals, first + j);
        }
        return;
    }
}

template <typename T, typename Derived>
bool ChunkBase<T, Derived>::read_block_or_constant(this const Derived& self,
                                                   int                 first,
                                                   int                 count,
                                                   T*                  out,
                                                   T&                  cst,
                                                   const T*            offset_slot) noexcept
    requires has_motif_state_v<Derived>
{
    const read_reclaim::ReadSection section;
    for (;;)
    {
        const T* const vals = self.values_load();
        test_hooks::maybe_read_delay(); // TEST KNOB, as in element_at_unguarded
        if (is_cst_val1(vals))
        {
            // raw-read: read_block_or_constant -- the shared first-constant buffer
            cst = load_at<false>(vals, 0);
            return true;
        }
        if (self.is_cst_val2_ptr_of(vals)) [[unlikely]]
        {
            cst = self.second_value_from_ptr(vals);
            return true;
        }
        if (Derived::is_motif_ptr(vals)) [[unlikely]]
        {
            T off{};
            if (!self.motif_offset(vals, offset_slot, off)) [[unlikely]]
            {
                continue; // the pointer moved: the slot may hold another state's value (F5)
            }
            for (int j = 0; j < count; ++j)
            {
                out[j] = Derived::motif_value(vals, first + j, off);
            }
            return false;
        }
        for (int j = 0; j < count; ++j)
        {
            // raw-read: read_block_or_constant -- under its own ReadSection
            out[j] = load_at<false>(vals, first + j);
        }
        return false;
    }
}

template <typename T, typename Derived>
bool ChunkBase<T, Derived>::read_block_or_constant(this const Derived& self,
                                                   int                 first,
                                                   int                 count,
                                                   T*                  out,
                                                   T&                  cst) noexcept
{
    if constexpr (has_motif_state_v<Derived>)
    {
        // A motif chunk needs its offset slot: the overload above.
        if (Derived::is_motif_ptr(self.values_load())) [[unlikely]]
        {
            CHUNK_INVARIANT_FAILED("read_block_or_constant: a motif chunk read without its offset slot");
        }
    }
    const read_reclaim::ReadSection section;
    const T* const                  vals = self.values_load();
    test_hooks::maybe_read_delay(); // TEST KNOB, as in element_at_unguarded
    if (self.is_cst_val1(vals))
    {
        // raw-read: read_block_or_constant -- the shared first-constant buffer
        cst = load_at<false>(vals, 0);
        return true;
    }
    if constexpr (is_zero_any_family_v<Derived>)
    {
        if (self.is_cst_val2_ptr_of(vals)) [[unlikely]]
        {
            cst = self.second_value_from_ptr(vals);
            return true;
        }
    }
    for (int j = 0; j < count; ++j)
    {
        // raw-read: read_block_or_constant -- under its own ReadSection
        out[j] = load_at<false>(vals, first + j);
    }
    return false;
}

template <typename T, typename Derived>
T& ChunkBase<T, Derived>::operator()(this const Derived& self, int offset) noexcept
{
    if constexpr (chunk_config::stats)
    {
        increm_chunk_read<Derived>();
    }
    // Contract: a mutable reference that bypasses the write proxy, valid only
    // on a materialised chunk. On a compressed one it points into a shared
    // constant (zero/zero_one_value) or an unbacked identifier slot (zero_any): a
    // write through it corrupts the constant for every chunk, or faults.
    // Checked builds stop on it.
    if constexpr (build_config::any_check && requires { self.written(); })
    {
        if (!self.written()) [[unlikely]]
        {
            CHUNK_INVARIANT_FAILED("operator()(int) on a chunk that is not materialised");
        }
    }
    static_assert(!is_packed, "operator()(int) returns T&: a Bit has no address (bit.h)");
    // raw-read: hands out an address / reference -- materialised first, the caller keeps it valid
    return self.values_load()[offset];
}

template <typename T, typename Derived>
void ChunkBase<T, Derived>::free_values(this Derived& self, T* ptr, [[maybe_unused]] int thr_rank)
{
    // Debug ring log: value-buffer frees.
    if constexpr (chunk_config::ring_log_debug || chunk_config::ring_log_op_only)
    {
        ring_log_op(get_tid(), EventCode::value_free, 0, static_cast<const void*>(ptr));
    }
    // Ledger: LIVE -> FREE, before the actual free, while the pointer is
    // still ours. Without it, std-alloc's malloc may hand the same address
    // to a later allocation that the ledger still sees as LIVE.
    CHUNK_LEDGER_RELEASE(static_cast<const void*>(ptr), "free_values");
    if constexpr (build_config::use_fast_alloc)
    {
        self.release_to_owner(thr_rank, ptr);
    }
    else
    {
        fast_alloc::svfree(ptr);
    }
}

template <typename T, typename Derived>
auto& ChunkBase<T, Derived>::get_chunk_allocator([[maybe_unused]] int          thr_rank,
                                                 [[maybe_unused]] unsigned int key) noexcept
{
    if constexpr (build_config::use_fast_alloc)
    {
        // Dynamic-vs-static routing lives on ChunkDynamicAllocPolicy.
        if constexpr (build_config::threaded)
        {
            return Derived::route_omp_dynamic_alloc(thr_rank, key);
        }
        else
        {
            return Derived::route_serial_dynamic_alloc();
        }
    }
    else
    {
        CHUNK_INVARIANT_FAILED("get_chunk_allocator() in a build without the fast allocator");
    }
}

template <typename T, typename Derived>
T* ChunkBase<T, Derived>::get_cst_val1() noexcept
{
    if constexpr (use_mm_protect)
    {
        // Mapped at the first call, never unmapped: a global array reaches
        // it before main (the member below is a null pointer then) and
        // after main returns.
        static T* const buffer = map_cst_val1();
        return buffer;
    }
    else
    {
        return m_ptr_cst_val1.data();
    }
}

template <typename T, typename Derived>
void ChunkBase<T, Derived>::init_block(T* values_loc, unsigned int n_elems) noexcept
{
    const std::size_t units = Derived::storage_units(n_elems);
    if constexpr ((std::is_arithmetic_v<T> && std::is_trivially_constructible_v<T>) || is_packed)
    {
        std::memset(static_cast<void*>(values_loc), 0, units * sizeof(T)); // Bit: all words 0, every bit false
    }
    else
    {
        std::fill_n(values_loc, units, T{});
    }
}

template <typename T, typename Derived>
[[nodiscard]] constexpr std::size_t ChunkBase<T, Derived>::storage_units(std::size_t n_elems) noexcept
{
    if constexpr (is_packed)
    {
        return bits::bytes_for(n_elems) / sizeof(T); // whole 64-bit words
    }
    else
    {
        return n_elems;
    }
}

template <typename T, typename Derived>
void ChunkBase<T, Derived>::init_block(T* values_loc) noexcept
{
    const std::size_t units = Derived::storage_units(alloc_chunk_count());
    if constexpr ((std::is_arithmetic_v<T> && std::is_trivially_constructible_v<T>) || is_packed)
    {
        std::memset(static_cast<void*>(values_loc), 0, units * sizeof(T)); // Bit: raw words, not Bit objects
    }
    else
    {
        std::fill_n(values_loc, units, T{});
    }
}

template <typename T, typename Derived>
T* ChunkBase<T, Derived>::map_cst_val1()
{
    const size_t bytes     = cst_buffer_size * sizeof(T);
    const size_t pagesize  = sysconf(_SC_PAGESIZE);
    const size_t allocsize = ((bytes + pagesize - 1) / pagesize) * pagesize;

    void* ptr = mmap(nullptr, allocsize, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

    if (ptr == MAP_FAILED)
    {
        CHUNK_INVARIANT_FAILED("mmap of the read-only first-value buffer ({} bytes) failed: {}", allocsize,
                               std::strerror(errno));
    }

    T* arr = static_cast<T*>(ptr);
    for (size_t i = 0; i < cst_buffer_size; ++i)
    {
        arr[i] = first_value;
    }

    mprotect(ptr, allocsize, PROT_READ);
    return arr;
}

template <typename T, typename Derived>
template <typename InRegion, typename Sweep>
void ChunkBase<T, Derived>::sweep_every_rank(InRegion&& in_region, Sweep&& sweep, bool mop_up)
{
    chunk_config::OMPTcc<chunk_config::has_ompt>::update(false);

    if (CHUNK_PMP_IN_PARALLEL_REGION)
    {
        in_region(CHUNK_PMP_THREAD_RANK);
    }
    else
    {
        int team = 0;
        pmp::team(pmp::Options{},
                  [&](pmp::Team& /*team*/)
                  {
                      const int thr = CHUNK_PMP_THREAD_RANK;
                      if (thr == 0) // one writer; `master` is deprecated since OpenMP 5.1
                      {
                          team = CHUNK_PMP_NUM_THREADS;
                      }
                      sweep(thr);
                  });
        if (mop_up)
        {
            for (int thr = team; thr < CHUNK_PMP_RANK_BOUND; ++thr)
            {
                sweep(thr);
            }
        }
    }

    chunk_config::OMPTcc<chunk_config::has_ompt>::update(true);
}

template <typename T, typename Derived>
void ChunkBase<T, Derived>::debug_check_written(this const Derived& self, int offset) noexcept
{
    if constexpr (build_config::any_check)
    {
        if (!self.is_written_at(offset))
        {
            CHUNK_INVARIANT_FAILED("trying to clear a chunk that is not written (offset {})", offset);
        }
    }
}

template <typename T, typename Derived>
void ChunkBase<T, Derived>::debug_check_written(this const Derived& self) noexcept
{
    if constexpr (build_config::any_check)
    {
        if (!self.written())
        {
            CHUNK_INVARIANT_FAILED("trying to clear a chunk that is not written");
        }
    }
}

template <typename T, typename Derived>
[[nodiscard]] bool ChunkBase<T, Derived>::is_written_at(this const Derived& self, int offset) noexcept
{
    if constexpr (requires { self.written(offset); })
    {
        return self.written(offset);
    }
    else
    {
        return self.written();
    }
}

template <typename T, typename Derived>
template <bool Relaxed>
[[nodiscard]] CHUNK_FORCE_INLINE T ChunkBase<T, Derived>::element_at(this const Derived& self, int offset) noexcept
{
    // A pure read is bracketed by a ReadSection (read_reclaim.h): a
    // compression pass running in the same region waits for it before it
    // releases the buffer this read may have loaded. The Relaxed reads are
    // the compressor's own scans and the marking filter: not bracketed.
    if constexpr (Relaxed)
    {
        return self.template element_at_unguarded<true>(offset);
    }
    else
    {
        const read_reclaim::ReadSection section;
        return self.template element_at_unguarded<false>(offset);
    }
}

template <typename T, typename Derived>
template <bool Relaxed>
[[nodiscard]] T ChunkBase<T, Derived>::element_at(this const Derived& self, int offset, const T* offset_slot) noexcept
    requires has_motif_state_v<Derived>
{
    if constexpr (Relaxed)
    {
        return self.template element_at_unguarded<true>(offset, offset_slot);
    }
    else
    {
        const read_reclaim::ReadSection section;
        return self.template element_at_unguarded<false>(offset, offset_slot);
    }
}

template <typename T, typename Derived>
template <bool Relaxed>
[[nodiscard]] T
ChunkBase<T, Derived>::element_at_unguarded(this const Derived& self, int offset, const T* offset_slot) noexcept
    requires has_motif_state_v<Derived>
{
    for (;;)
    {
        const T* const vals = self.values_load();
        if constexpr (!Relaxed)
        {
            test_hooks::maybe_read_delay(); // TEST KNOB, compiled out by default
        }
        if (self.is_cst_val2_ptr_of(vals)) [[unlikely]]
        {
            return self.second_value_from_ptr(vals);
        }
        if (Derived::is_motif_ptr(vals)) [[unlikely]]
        {
            T off{};
            if (self.motif_offset(vals, offset_slot, off)) [[likely]]
            {
                return Derived::motif_value(vals, offset, off);
            }
            continue; // the pointer moved: the slot may hold another state's value (F5)
        }
        // raw-read: element_at_unguarded -- under element_at's ReadSection, or a scan of a claimed chunk
        return load_at<Relaxed>(vals, offset);
    }
}

template <typename T, typename Derived>
template <bool Relaxed>
[[nodiscard]] T ChunkBase<T, Derived>::element_at_unguarded(this const Derived& self, int offset) noexcept
{
    // The one branch identifiers cost: a compressed-to-second-constant
    // chunk points into a region that was never written, so the
    // constant comes from the value table instead. A pointer range test,
    // very well predicted (most reads go to materialised chunks).
    //
    // ONE load of `values`, tested through the CRTP hooks (chunk_base knows
    // nothing of the registry). Do not test self.is_cst_val2() and then
    // dereference values_load(): a compressor claiming the chunk between the
    // two loads swaps in a region identifier, and the second load indexes an
    // unmapped page. Same rule as read_values().
    //
    // Gated on the family trait (chunk_traits.h), NOT on a `requires` probe
    // for the hooks: a probe fails OPEN (where false, the read returns 0
    // from the unwritten region -- a wrong number, not a build error), and
    // nvhpc evaluates it differently on the still-incomplete CRTP Derived.
    // A specialisation per member of the family cannot differ.
    //
    // The three reads (here, read_block, read_block_or_constant) know one
    // kind of registry entry: a constant, zero_any's second value. A member
    // whose entries differ element by element (pattern::Array: a motif
    // plus the chunk's offset) fails the static_assert below until its own
    // branch is written here. The calls stay those of zero_any, not a hook
    // in front of them: even forced inline, one more function level on
    // this path changed gcc 14's code for zero_any's write path.
    if constexpr (is_zero_any_family_v<Derived>)
    {
        static_assert(
            requires(const Derived& obj, const T* ptr) {
                obj.is_cst_val2_ptr_of(ptr);
                obj.second_value_from_ptr(ptr);
            }, "a chunk of the zero_any family must provide is_cst_val2_ptr_of() and second_value_from_ptr(), "
               "or its own branch of the reads: its compressed chunks point at registry entries, not at "
               "their values");
        const T* const vals = self.values_load();
        if constexpr (!Relaxed)
        {
            test_hooks::maybe_read_delay(); // TEST KNOB, compiled out by default
        }
        if (self.is_cst_val2_ptr_of(vals)) [[unlikely]]
        {
            return self.second_value_from_ptr(vals);
        }
        if constexpr (has_motif_state_v<Derived>)
        {
            // A motif chunk: its elements are motif[j] + an offset kept in its
            // second-value slot, which only the caller can name -- the
            // element_at(offset, slot) overload below. Without it a scan reads
            // the motif alone (the marking filter's hint, never a value); a
            // read is a caller that forgot the slot.
            if (Derived::is_motif_ptr(vals)) [[unlikely]]
            {
                if constexpr (!Relaxed)
                {
                    CHUNK_INVARIANT_FAILED("element_at({}): a motif chunk read without its offset slot", offset);
                }
                // raw-read: element_at_unguarded -- the motif alone, for a relaxed scan
                return load_at<Relaxed>(vals, offset);
            }
        }
        // raw-read: element_at_unguarded -- under element_at's ReadSection, or a scan of a claimed chunk
        return load_at<Relaxed>(vals, offset);
    }
    const T* const vals = self.values_load();
    // TEST KNOB (test_hooks.h): widens the load -> dereference window
    // a concurrent compressor's CAS + free can fall into. Compiled out
    // unless CHUNK_TEST_READ_DELAY_EVERY > 0, and then only on threads
    // that opted in. Not on the Relaxed (scan) reads.
    if constexpr (!Relaxed)
    {
        test_hooks::maybe_read_delay();
    }
    // raw-read: element_at_unguarded -- under element_at's ReadSection, or a scan of a claimed chunk
    return load_at<Relaxed>(vals, offset);
}

template <typename T, typename Derived>
void ChunkBase<T, Derived>::release_to_owner(this Derived& self, int thr_rank, T* ptr)
{
    // Callers such as block_clear()/clear_st() pass the default rank 0 from
    // inside a region: the rank that matters here is the one running this
    // code, both for "is it mine" and for which list a foreign block is
    // parked on (a rank only ever touches its own list inside a region).
    if (CHUNK_PMP_IN_PARALLEL_REGION)
    {
        thr_rank = CHUNK_PMP_THREAD_RANK;
    }
    auto& mine = get_chunk_allocator(thr_rank); // same slab geometry for every key
    auto* slab = mine.slab_of(ptr);
    if (slab == nullptr || slab->owner == nullptr)
    {
        // No slab behind this pointer (bypass_slabs, or an instance never
        // initialised): the legacy route is the only one there is.
        mine.free_elt_pow2(ptr);
        return;
    }
    using AllocT = std::remove_reference_t<decltype(mine)>;
    auto* owner  = const_cast<AllocT*>(static_cast<const AllocT*>(slab->owner));
    if (slab->owner_rank < 0 || slab->owner_rank == thr_rank || !CHUNK_PMP_IN_PARALLEL_REGION)
    {
        CHUNK_FREEPATH_INC(freed);
        owner->free_elt_pow2(ptr, slab);
        return;
    }
    CHUNK_FREEPATH_INC(park_foreign);
    CHUNK_LEDGER_PARK(ptr, "release_to_owner.park_foreign");
    self.get_manager().get_list_block_free_thr(thr_rank, slab->owner_key).push_back(ptr);
}

template <typename T, typename Derived>
template <bool Relaxed>
[[nodiscard]] T ChunkBase<T, Derived>::load_at(const T* vals, int offset) noexcept
{
    CHUNK_ASSUME(offset >= 0); // an offset in the chunk: the address needs no sign extension
    if constexpr (is_packed)
    {
        // raw-read: load_at, called by element_at_unguarded only
        return elem_get(vals, static_cast<std::size_t>(offset)); // relaxed word load under OpenMP
    }
    else if constexpr (Relaxed)
    {
        return internal::load_element_relaxed(vals + offset);
    }
    else
    {
        // raw-read: load_at, called by element_at_unguarded only
        return vals[offset];
    }
}

} // namespace shrink_array