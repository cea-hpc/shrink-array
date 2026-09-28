#pragma once
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
#include "chunk/chunk_traits.h" // Chunk2A fwd decl (element_at variant gate)
#include <vector>
#include <unistd.h>
#include <algorithm> // std::min, the constant scans
#include <map>
#include <cstdint>
#include <type_traits>
#include <atomic>
#include <array>
#include <cstddef>
#include <cstdio> // perror (use_mm_protect: the cst_val1 buffer)
#include <cstdlib>
#include <cstring> // std::memset
#include <iostream>
#include <memory>
#include <optional>
#include <sys/mman.h>

#include "chunk/internal/concept_array_chunk_element.h" // ArrayChunkElement, asserted on ChunkBase
#include "chunk/internal/churn_counters.h"              // churn::note_materialisation
#include "chunk/internal/chunk_size_policy.h"
#include "chunk/internal/in_t.h"            // in_t<T>: the scans' constants, by value or by reference
#include "chunk/internal/element_access.h"  // elem_get: Bit buffers are words of bits
#include "chunk/internal/relaxed_element.h" // element_at<true>: the marking filter's reads
#include "chunk/internal/svalloc.h"
#include "chunk/config.h"
#include "chunk/PMP_CHUNK.h"
#include "chunk/ring_log_debug.h"           // SEGV hunt: EV_VALUE_ALLOC/FREE to prove aliasing
#include "chunk/internal/global_check.h"    // global_check_nan (DO_CHECK: config.h's CheckNaN<true>)
#include "chunk/internal/invariant_check.h" // INVARIANT_FAILED
#include "chunk/internal/deferred_op_pool.h"
#include "chunk/internal/compression_manager.h"
#include "chunk/internal/pool_ledger.h" // CHUNK_LEDGER_* block-lifecycle checks (check builds)
#include "chunk/internal/chunk_stats.h"
#include "chunk/internal/crp_helper.h"
#include "chunk/internal/bad_ptr_check.h" // CHUNK_CHECK_VALUES_PTR (check builds)

#include "chunk/internal/freepath_counters.h" // [FREEPATH] counters, needed in std-alloc builds too

#ifdef FAST_ALLOC
    #include "fast_allocators/tc_allocator.h"
#else
template <typename T>
class TCTypeCAllocator; // forward declaration

template <typename T>
class TCTypeAllocator;

    #include "chunk/internal/svalloc.h"
#endif

#include "flat_hash_map.hpp"
#include "ankerl/unordered_dense.h"

// deferred_ops_storage.h has no internal dependency on _OPENMP (just
// <memory>/<mutex>/<atomic>/PMP_CHUNK.h), so it is always included: nothing
// in it costs anything when DeferredOpsStorage<...> is never instantiated
// (serial builds), and doing so lets ChunkDeferredOpsPolicy below reference
// it unconditionally.
#include "chunk/internal/deferred_ops_storage.h"
namespace shrink_array
{

template <typename Derived>
struct ProxyTypeTrait;

template <typename T>
class Chunk;

template <typename T>
class Proxy;

// Specialisation for Chunk
template <typename T>
struct ProxyTypeTrait<Chunk<T>>
{
    using Type = Proxy<T>;
};

// STAYS a directive, and it is not the redundant twin of the declaration at the
// top of this file: that one is at GLOBAL scope, this one is inside
// `namespace shrink_array` (opened just above), so they name two different
// entities -- ::TCTypeCAllocator and shrink_array::TCTypeCAllocator.
//
// It is load-bearing. The real ::TCTypeCAllocator (fast_allocators/tc_allocator.h)
// is the OpenMP per-rank allocator; in a SERIAL build this incomplete
// declaration deliberately SHADOWS it for the rest of the namespace, so the
// OpenMP-only allocator pool below (allocator_pool_, alloc_chunk_allocator_thr)
// parses against a type that is never completed and never instantiated there.
// Deleting the guard would resolve those uses to the real type in a serial
// build; deleting the declaration would resolve them to the real type too.
// Neither is wanted, and no `if constexpr` can express "declare a different
// type", so this is one of the cases the preprocessor is actually for.
#ifndef _OPENMP
template <typename T>
struct TCTypeCAllocator;
#endif


// ============================================================================
// DeferredReadGate: read-side visibility gate, selected by compile-time mode
// ----------------------------------------------------------------------------
// The read gate (reader_sync + ArrayChunkBase::sync_chunk) as a CRTP policy, in
// the same style as Chunk2CounterPolicy / Proxy2CounterPolicy. `if constexpr`
// alone cannot express this: pending_ops_ / wait_for_availability /
// is_not_available exist in some builds only, so a discarded `if constexpr`
// branch would still phase-1-look-up a missing member and fail. Three template
// specializations — one per config — each name ONLY the members that exist in
// that config, and only the selected specialization is ever instantiated.
//
//   None    (serial, !use_openmp)            : nothing can be pending -> no-op
//   Coarse  (OpenMP + SHRINK_SMALL_CHUNK)     : no counter -> is_not_available()
//   Precise (OpenMP + per-chunk pending_ops_) : precise pending_ops_ load
// ============================================================================
enum class DeferReadGate : std::uint8_t
{
    None,
    Coarse,
    Precise
};

inline constexpr DeferReadGate defer_read_gate_mode = !build_config::use_openmp ? DeferReadGate::None
                                                      : !chunk_config::use_precise_pending_gate
                                                          ? DeferReadGate::Coarse
                                                          : DeferReadGate::Precise;

template <typename Derived, DeferReadGate Mode = defer_read_gate_mode>
struct DeferredReadGate; // primary left undefined: only the 3 specializations exist

// ── Pending-visibility protocol: THE invariants, in one place ────────────────
// This policy is the single owner of every access to the per-chunk pending
// counter. Nothing outside it may touch pending_ops_ directly. The verbs:
//
//   pending_publish()   enqueue side. +1 RELEASE, called BEFORE the map insert
//                       (to_be_queued), so pending_any()>0 spans a pending op's
//                       ENTIRE lifetime — closes the reader stale-read window.
//                       The op itself is published to flushers via the Dekker
//                       write flag (begin_write/end_write), not this counter.
//   pending_retire(n)   drain side. -n RELEASE, called AFTER the ops' write
//                       bodies ran (flush_deferred / flush_chunk_all_ranks /
//                       erase_own_rank_op), so a reader that acquire-observes
//                       the decrement also observes the drained results.
//   pending_any()       gate side. ACQUIRE load > 0; pairs with the publish.
//                       False can only mean "nothing pending on this chunk".
//   read_gate_should_wait(i) / reader_sync(i)
//                       read-side gates built on the above (or on the coarse
//                       is_not_available() when the counter doesn't exist).
//   precise             constexpr: true iff the per-chunk counter exists —
//                       lets algorithmic stages that only make sense with the
//                       counter (e.g. wait_for_availability's own-rank fast
//                       stage) be `if constexpr`-selected.
//
// Storage note (PERF): the counter member itself stays declared in ChunkBase
// (under its historical #if) so the object layout — cache-line placement of
// `values` — is bit-for-bit unchanged. The policy owns the ACCESS, not the
// storage. Every verb is a one-liner around the exact same atomic op and
// memory order as the code it replaced: codegen-identical by construction.
// ─────────────────────────────────────────────────────────────────────────────

// Serial build: the deferred-op machinery does not exist, nothing is ever
// pending, so the gate is a pure no-op (and names no OpenMP-only member).
template <typename Derived>
struct DeferredReadGate<Derived, DeferReadGate::None>
{
    static constexpr bool precise = false;

    /// @brief Read-side gate check: always false (serial build has no
    ///        deferred writes, so nothing can ever be pending).
    /// @param i Element offset being read; unused.
    /// @return Always false.
    [[nodiscard]] bool read_gate_should_wait([[maybe_unused]] int i) const noexcept
    {
        return false;
    }

    /// @brief No-op: nothing to synchronise against in a serial build.
    /// @param i Element offset being read; unused.
    void reader_sync([[maybe_unused]] int i) const noexcept {}

    /// @brief No-op: no per-chunk pending counter exists in a serial build.
    /// @param thr Calling thread's rank; unused.
    void pending_publish([[maybe_unused]] int thr) noexcept {}

    /// @brief No-op: no per-chunk pending counter exists in a serial build.
    /// @param count Number of operations retired; unused.
    void pending_retire([[maybe_unused]] int32_t count) noexcept {}

    /// @brief Always false: nothing can ever be pending in a serial build.
    /// @return Always false.
    [[nodiscard]] bool pending_any() const noexcept
    {
        return false;
    }

    /// @brief Always false: nothing can ever be pending in a serial build.
    /// @param thr Calling thread's rank; unused.
    /// @param i Element offset; unused.
    /// @return Always false.
    [[nodiscard]] bool foreign_pending([[maybe_unused]] int thr, [[maybe_unused]] int i) const noexcept
    {
        return false;
    }
};

// OpenMP + SHRINK_SMALL_CHUNK: no per-chunk counter; the authoritative gate is
// the (coarser) map-scan is_not_available().
template <typename Derived>
struct DeferredReadGate<Derived, DeferReadGate::Coarse>
{
    static constexpr bool precise = false;

    // No counter in this build: publish/retire are no-ops and pending_any() is
    // constant false — call sites that gate counter-only logic on it fold to
    // nothing, exactly as when those sites were #if'd out.
    /// @brief No-op: this build has no per-chunk pending counter.
    /// @param thr Calling thread's rank; unused.
    void pending_publish([[maybe_unused]] int thr) noexcept {}

    /// @brief No-op: this build has no per-chunk pending counter.
    /// @param count Number of operations retired; unused.
    void pending_retire([[maybe_unused]] int32_t count) noexcept {}

    /// @brief Always false: this build has no per-chunk pending counter to
    ///        report on.
    /// @return Always false.
    [[nodiscard]] bool pending_any() const noexcept
    {
        return false;
    }

    // No counter, no owner: be CONSERVATIVE — any pending element makes the
    // direct write drain first. Correct but slow; SMALL_CHUNK + sync_on_read
    // is a documented perf anti-combination.
    /// @brief Conservative direct-write guard predicate.
    /// @param thr Calling thread's rank; unused (no owner tracking in this mode).
    /// @param i Element offset being written; unused -- the check is per-chunk.
    /// @return Derived::is_not_available() -- the coarse, map-scan-based check.
    [[nodiscard]] bool foreign_pending([[maybe_unused]] int thr, [[maybe_unused]] int i) const noexcept
    {
        return static_cast<const Derived&>(*this).is_not_available();
    }

    /// @brief Read-side gate check, using the coarse map-scan since no
    ///        per-chunk counter exists in this build.
    /// @param i Element offset being read; unused -- the check is per-chunk.
    /// @return Derived::is_not_available().
    [[nodiscard]] bool read_gate_should_wait([[maybe_unused]] int i) const noexcept
    {
        return static_cast<const Derived&>(*this).is_not_available();
    }

    /// @brief Drain this chunk if the coarse check says something might be
    ///        pending on it.
    /// @param i Element offset being read.
    void reader_sync(int i) const noexcept
    {
        const auto& self = static_cast<const Derived&>(*this);
        if (self.is_not_available())
            self.wait_for_availability(i);
    }
};

// OpenMP + per-chunk counter: precise gate. A flat acquire load pairs with the
// release fetch_add at enqueue; >0 means this chunk has a pending op.
template <typename Derived>
struct DeferredReadGate<Derived, DeferReadGate::Precise>
{
    static constexpr bool precise = true;

    // ── Pending word, dual mode ─────────────────────────────────────────────
    // OFF (sync_on_read=false): the counter is EXACTLY the historical signed
    // int32 with fetch_add(release)/fetch_sub(release)/load(acquire)>0 — byte-
    // identical to the pre-owner-tagging behaviour. Rationale (hard-won): the
    // owner-encoded unsigned word changed the OVER-RETIRE failure mode. The
    // old signed counter absorbed a transient over-retire by going negative
    // (pending_any()==false, self-healing at the next publish); the unsigned
    // [owner:12|count:20] word wraps into a HUGE count with a clobbered owner
    // — gates poisoned permanently, drain paths hyperactive, and the gnu OFF
    // battery corrupted at ~1/355 (bisection: b601 clean at 7774 runs,
    // 4505ad27 failing at 107/674). OFF never reads `owner`, so it gets the
    // proven semantics back verbatim.
    // ON (sync_on_read=true): the encoded word [owner:12|count:20] is required
    // by foreign_pending(); the CAS verbs now MASK and SATURATE so an
    // over-retire clamps count to 0 (self-healing, like the signed counter)
    // instead of wrapping, and publish saturates instead of overflowing into
    // the owner field.
    static constexpr uint32_t kCountBits  = 20;
    static constexpr uint32_t kCountMask  = (uint32_t{1} << kCountBits) - 1;
    static constexpr uint32_t kOwnerMulti = 0xFFFu;

    /// @brief Sign-extend the low 20 bits of the pending word (the count is
    ///        stored as two's-complement inside those 20 bits).
    /// @param word Raw pending word.
    /// @return The count, sign-extended to a full int32_t.
    [[nodiscard]] static constexpr int32_t sign_extend_count(uint32_t word) noexcept
    {
        return static_cast<int32_t>(word << (32 - kCountBits)) >> (32 - kCountBits);
    }

    /// @brief Publish a newly-enqueued deferred op: increment the per-chunk
    ///        pending count (and, under sync_on_read, record this thread as
    ///        the owner). See the class comment above for the dual OFF/ON
    ///        encoding.
    /// @param thr Calling (publishing) thread's rank.
    void pending_publish([[maybe_unused]] int thr) noexcept
    {
        auto& pending = static_cast<Derived&>(*this).pending_ops_;
        {
            const uint32_t me =
                (static_cast<uint32_t>(thr) + 1u < kOwnerMulti) ? static_cast<uint32_t>(thr) + 1u : kOwnerMulti;
            auto cur = static_cast<uint32_t>(pending.load(std::memory_order_acquire));
            for (;;)
            {
                // count = SIGNED 20-bit (two's complement), reproducing the
                // proven historical fetch_add/fetch_sub semantics exactly: a
                // transient over-retire goes NEGATIVE (pending_any false) and
                // self-heals at the next publish — it is never clamped, so no
                // debt accumulates and the gates never stick (the fw6 clamp
                // left the count permanently inflated after an over-retire:
                // poison-lite, hyperactive drains, the ~1/80 ON class).
                const int32_t  cnt      = sign_extend_count(cur);
                const uint32_t own      = cur >> kCountBits;
                const uint32_t nown     = (cnt <= 0) ? me : (own == me ? own : kOwnerMulti);
                const uint32_t ncnt     = static_cast<uint32_t>(cnt + 1) & kCountMask;
                const uint32_t nw       = (nown << kCountBits) | ncnt;
                auto           expected = static_cast<int32_t>(cur);
                if (pending.compare_exchange_weak(expected, static_cast<int32_t>(nw), std::memory_order_acq_rel,
                                                  std::memory_order_acquire))
                    return;
                cur = static_cast<uint32_t>(expected);
            }
        }
    }
    /// @brief Retire `count` deferred ops that have now been applied: decrement
    ///        the per-chunk pending count (clearing the owner once it
    ///        reaches <= 0).
    /// @param count Number of operations being retired.
    void pending_retire(int32_t count) noexcept
    {
        auto& pending = static_cast<Derived&>(*this).pending_ops_;
        {
            auto cur = static_cast<uint32_t>(pending.load(std::memory_order_acquire));
            for (;;)
            {
                const int32_t  cnt      = sign_extend_count(cur);
                const int32_t  ncnt     = cnt - count;                              // may go NEGATIVE: two's
                const uint32_t bits     = static_cast<uint32_t>(ncnt) & kCountMask; // complement in 20 bits
                const uint32_t nw       = (ncnt <= 0) ? bits                        // owner cleared while <= 0
                                                      : ((cur >> kCountBits) << kCountBits) | bits;
                auto           expected = static_cast<int32_t>(cur);
                if (pending.compare_exchange_weak(expected, static_cast<int32_t>(nw), std::memory_order_acq_rel,
                                                  std::memory_order_acquire))
                    return;
                cur = static_cast<uint32_t>(expected);
            }
        }
    }
    /// @brief Whether anything is currently pending on this chunk (either
    ///        OFF or ON encoding, decoded correctly either way).
    /// @return true if the (sign-extended) pending count is > 0.
    [[nodiscard]] bool pending_any() const noexcept
    {
        const auto raw = static_cast<const Derived&>(*this).pending_ops_.load(std::memory_order_acquire);
        return sign_extend_count(static_cast<uint32_t>(raw)) > 0;
    }

    // Direct-write guard predicate — only meaningful under sync_on_read (the
    // owner field only exists there); OFF returns false (and is never called).
    /// @brief Is a FOREIGN thread's op pending on this chunk right now?
    /// @param thr Calling (writing) thread's rank.
    /// @param i Element offset being written; unused (the check is per-chunk).
    /// @return true iff something is pending and it was not published by `thr`.
    [[nodiscard]] bool foreign_pending([[maybe_unused]] int thr, [[maybe_unused]] int i) const noexcept
    {
        {
            const auto w =
                static_cast<uint32_t>(static_cast<const Derived&>(*this).pending_ops_.load(std::memory_order_acquire));
            return sign_extend_count(w) > 0 && (w >> kCountBits) != static_cast<uint32_t>(thr) + 1u;
        }
    }
    // ── fw13 (gatefix): the read gates MUST decode the word ────────────────
    // These two predicates used a RAW SIGNED compare on the whole 32-bit word
    // (`pending_ops_.load() > 0`). Under sync_on_read the word is encoded
    // [owner:12|count:20]; when a SECOND rank publishes on the same chunk while
    // count>0, pending_publish() sets owner := kOwnerMulti (0xFFF). Bit 11 of
    // the owner field is bit 31 of the word, so the raw int32 goes NEGATIVE
    // (e.g. 0xFFF00002 = two pending ops, two owners) and the raw `> 0` gate
    // silently OPENS: read_pure/reader_sync and sync_chunk/read_gate_should_wait
    // both skip wait_for_availability while ops are genuinely pending — the
    // reader consumes the stale pre-op value. This is exactly the fw8b-MINIMAL
    // residual (read hooks only, still ~1/58-147): the read hook itself was the
    // hole. It also explains the llvm/gnu asymmetry: OMPT per-barrier drains
    // keep pendings short-lived under llvm, so the multi-owner overlap window
    // barely exists there; libgomp leaves pendings alive across whole phases.
    // pending_any() is the single decode-correct predicate (sign-extended
    // 20-bit count under ON, verbatim signed compare under OFF — codegen
    // byte-identical to the old raw load in OFF builds).
    /// @brief Read-side gate check: does a reader need to wait/drain before
    ///        reading element `i`? See the fw13 note above for why this
    ///        must go through pending_any() rather than a raw sign compare.
    /// @return pending_any(). (`i` unused: the check is per-chunk, not per-offset.)
    [[nodiscard]] bool read_gate_should_wait(int /*i*/) const noexcept
    {
        return pending_any();
    }

    /// @brief Drain this chunk across all ranks if anything is pending on it.
    /// @param i Element offset being read.
    void reader_sync(int i) const noexcept
    {
        if (pending_any()) [[unlikely]]
            static_cast<const Derived&>(*this).wait_for_availability(i);
    }
};

// ============================================================================
// ChunkAllocPolicy: OpenMP vs serial chunk allocation, selected by
// build_config::use_openmp. Unlike DeferredReadGate/the deferred-ops machinery/CAS,
// this is a genuine two-implementation split: BOTH modes allocate a chunk,
// initialize it, and keep stats/ledger/leak-tracking in sync -- only the
// OpenMP version additionally tries the per-thread free list first. Same
// style as DeferredReadGate: static_cast<Derived&>(*this) inline rather than
// inheriting CRTPSupport<Derived> a second time, which would make ChunkBase's
// own derived()/self() ambiguous (two paths to the same stateless mixin).
// ----------------------------------------------------------------------------
template <typename T, typename Derived, bool UseOpenMP = build_config::use_openmp>
struct ChunkAllocPolicy; // primary left undefined: only the 2 specializations exist

// OpenMP: thread-local free-list reuse first, fresh allocation otherwise.
template <typename T, typename Derived>
struct ChunkAllocPolicy<T, Derived, true>
{
    /// @brief Allocate a chunk with thread-local caching (OpenMP version).
    /// @tparam InitReused Initialiser applied to a buffer popped from the reuse list
    /// @tparam InitFresh  Initialiser applied to a freshly allocated buffer
    /// @param thr_rank Thread rank/ID
    /// @param init_reused Called on a recycled buffer, which still carries
    ///        whatever its previous owner left in it
    /// @param init_fresh Called on a brand-new buffer, which carries whatever
    ///        the allocator hands out -- zeroed by svcalloc_align, raw from the
    ///        pooled allocator
    /// @return Pointer to the allocated and initialized chunk
    /// @note The two-functor form is what lets v1 share this path. A v1 chunk is
    ///       compressed only when it is all-zero, so a buffer coming back from
    ///       the reuse pool is already zero and re-filling it would memset the
    ///       hot path (~97% of materialisations) for nothing. v2/v2A compress to
    ///       an arbitrary constant and must initialise both branches alike, so
    ///       they call the single-functor overload below, which passes the same
    ///       callable twice.
    template <typename InitReused, typename InitFresh>
    T* allocate_chunk_internal(int thr_rank, InitReused init_reused, InitFresh init_fresh)
    {
        auto& self       = static_cast<Derived&>(*this);
        T*    values_loc = nullptr;

        // The per-thread reuse list is only indexable once
        // resize_all_list_block_free_thr() has sized it to max_thr, and that is NOT
        // guaranteed here: in fast-alloc builds nothing ever parks buffers in
        // the legacy store, so it can still be empty when the first chunk is
        // allocated, and in dynamic builds a geometry born from a re-chunk has
        // no per-size list at all. Indexing it anyway is silent UB in a normal
        // build and aborts under _GLIBCXX_ASSERTIONS with
        // "Assertion '__n < this->size()' failed" -- which is how this first
        // showed up, from here, on nvhpc.
        //
        // Both geometries go through the same call now:
        // get_list_block_free_thr_size() is bounds-checked at the source (see
        // ManagerChunk), so a slot that was never sized answers 0 instead of
        // indexing out of bounds. The earlier version guarded only the static
        // case, on the assumption that in dynamic builds the by-size map always
        // owns the list -- but a geometry born from a re-chunk is never
        // registered, so the lookup falls back to the very legacy store the
        // guard was protecting. Same hole, one branch further along.
        //
        // Try to reuse from thread-local free list.
        // The reuse list of THIS array on this rank (thread_alloc_key(), primed by
        // the proxy next to the geometry): a block taken here comes from the
        // array's own slabs, so affinity survives recycling.
        const unsigned int alloc_key = thread_alloc_key();
        CHUNK_FREEPATH_ALLOC(alloc_key);
        // Churn measurement (churn_counters.h): one materialisation of the
        // array this key names -- rank-local, dynamic builds only.
        churn::note_materialisation<T>(thr_rank, alloc_key);
        if (const auto nb_free = static_cast<int>(self.get_manager().get_list_block_free_thr_size(thr_rank, alloc_key));
            nb_free > 0)
        {
            // Safe now: nb_free > 0 implies the slot for thr_rank exists.
            auto& ll_block_free_thr = self.get_manager().get_list_block_free_thr(thr_rank, alloc_key);
            if constexpr (chunk_config::stats)
            {
                increm_chunk_list<Derived>();
                add_chunk_bytes_reused<Derived>(static_cast<long>(sizeof(T)) * static_cast<long>(Derived::storage_units(alloc_chunk_count())));
            }
            values_loc = ll_block_free_thr.back();
            CHUNK_CHECK_VALUES_PTR(values_loc, "freelist.pop");
            ll_block_free_thr.pop_back();
            // Ledger: PARKED -> LIVE, recorded BEFORE the memset so a concurrent
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
                add_chunk_bytes_alloc<Derived>(static_cast<long>(sizeof(T)) * static_cast<long>(Derived::storage_units(alloc_chunk_count())));
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
                // The svcalloc branch used to sit inside an
                //   if constexpr (std::is_same_v<decltype(init_func(...)), void>)
                // which left values_loc uninitialized -- and the chunk
                // unallocated -- for any initialiser that returned a value.
                // Every caller's initialiser returns void, so the guard was
                // always true; it only stood in the way of ever writing one
                // that did not.
                svcalloc_align(values_loc, T, Derived::storage_units(alloc_chunk_count()));
                CHUNK_CHECK_VALUES_PTR(values_loc, "alloc.svcalloc");
                CHUNK_LEDGER_ALLOC(values_loc, "allocate_chunk_internal.svcalloc");
                init_fresh(values_loc);
            }
        }
        if constexpr (chunk_config::stats)
        {
            increm_chunk_create<Derived>();
        }

        // SEGV hunt: record the actual chunk-value buffer address (fast path,
        // both freelist-reuse and fresh-alloc). If one lands on/near a live op's
        // address, dump_ring_for_addr will show it (op/value pool aliasing).
        if constexpr (chunk_config::ring_log_debug || chunk_config::ring_log_op_only)
            ring_log_op(get_tid(), EV_VALUE_ALLOC, 0, static_cast<const void*>(values_loc));

        return values_loc;
    }

    /// @brief Same initialiser on both branches -- the v2/v2A form.
    /// @tparam InitFunc Initialization function type
    /// @param thr_rank Thread rank/ID
    /// @param init_func Function to initialize the allocated block
    /// @return Pointer to the allocated and initialized chunk
    template <typename InitFunc>
    T* allocate_chunk_internal(int thr_rank, InitFunc init_func)
    {
        return allocate_chunk_internal(thr_rank, init_func, init_func);
    }
};

// Serial: always allocate fresh -- no per-thread free list to contend over,
// so there is nothing to reuse from.
template <typename T, typename Derived>
struct ChunkAllocPolicy<T, Derived, false>
{
    /// @brief Allocate a chunk (non-OpenMP version).
    /// @tparam InitReused Unused here: serially there is no reuse list to pop from
    /// @tparam InitFresh  Initialiser applied to the freshly allocated buffer
    /// @param thr_rank Thread rank (unused in single-threaded mode)
    /// @param init_reused Accepted and ignored, so that a caller can hand the
    ///        same pair of initialisers to either specialisation
    /// @param init_fresh Function to initialize the allocated block
    /// @return Pointer to the allocated and initialized chunk
    template <typename InitReused, typename InitFresh>
    T* allocate_chunk_internal([[maybe_unused]] int        thr_rank,
                               [[maybe_unused]] InitReused init_reused,
                               InitFresh                   init_fresh)
    {
        auto& self       = static_cast<Derived&>(*this);
        T*    values_loc = nullptr;

        if constexpr (chunk_config::stats)
        {
            increm_chunk_create<Derived>();
            increm_chunk_alloc<Derived>();
            add_chunk_bytes_alloc<Derived>(static_cast<long>(sizeof(T)) * static_cast<long>(Derived::storage_units(alloc_chunk_count())));
        }

        if constexpr (build_config::use_fast_alloc)
        {
            values_loc = self.get_chunk_allocator(0).alloc();
            CHUNK_CHECK_VALUES_PTR(values_loc, "alloc.fast_serial");
            // Ledger: FREE -> LIVE. The non-OMP path previously skipped this,
            // leaving every serially-allocated chunk UNTRACKED, so free_values
            // later tripped a spurious b_release-of-UNTRACKED. Matches the OMP
            // variant above.
            CHUNK_LEDGER_ALLOC(values_loc, "allocate_chunk_internal.fast_serial");
            init_fresh(values_loc);
        }
        else
        {
            svcalloc_align(values_loc, T, Derived::storage_units(alloc_chunk_count()));
            CHUNK_CHECK_VALUES_PTR(values_loc, "alloc.svcalloc_serial");
            CHUNK_LEDGER_ALLOC(values_loc, "allocate_chunk_internal.svcalloc_serial");
            init_fresh(values_loc);
        }


        return values_loc;
    }

    /// @brief Same initialiser on both branches -- the v2/v2A form.
    /// @tparam InitFunc Initialization function type
    /// @param thr_rank Thread rank (unused in single-threaded mode)
    /// @param init_func Function to initialize the allocated block
    /// @return Pointer to the allocated and initialized chunk
    template <typename InitFunc>
    T* allocate_chunk_internal(int thr_rank, InitFunc init_func)
    {
        return allocate_chunk_internal(thr_rank, init_func, init_func);
    }
};

// ============================================================================
// ChunkDeferredOpsPolicy: the whole OpenMP-only deferred-write subsystem --
// type aliases, CAS, thread-local free-list plumbing, and the deferred-op
// enqueue/flush/read-gate methods -- combined into ONE policy rather than
// four separate ones, because they are tightly coupled (per_thread_map is
// used unqualified across all of them, and to_be_queued_for_deferred_
// processing/wait_for_availability/is_not_available are declared here with
// their bodies out-of-line in chunk_base_inline.h, exactly as before).
// Every name they use (DeferredOpsStorage, WriteCompressSyncPolicy,
// DeferredOpPtr, CompressionManager<T,Derived>) is unconditionally declared
// elsewhere, and the false specialization below is simply never instantiated
// in a serial build.
// Storage note: `values` (ChunkBase's own atomic pointer) and `pending_ops_`
// stay declared on ChunkBase itself for object-layout stability (same
// principle as DeferredReadGate above); this policy only owns the access,
// reaching back via static_cast<Derived&>(*this) -- matching DeferredReadGate's
// own style, not CRTPSupport<Derived>, to avoid making ChunkBase's
// derived()/self() ambiguous.
// ----------------------------------------------------------------------------
template <typename T, typename Derived, bool UseOpenMP = build_config::use_openmp>
struct ChunkDeferredOpsPolicy; // primary left undefined: only the 2 specializations exist

template <typename T, typename Derived>
struct ChunkDeferredOpsPolicy<T, Derived, true>
{
    using defer_op_ptr = std::unique_ptr<DeferredOperationBase, DeferredOpDeleter>;

    using unordered_map_deferred_operations =
        ankerl::unordered_dense::map<int, std::unique_ptr<DeferredOperationBase, DeferredOpDeleter>>;
    // per-thread map type: maps Derived* (chunk pointer) -> unique_ptr<map<num, op>>
    using per_thread_map = ska::flat_hash_map<Derived*, std::unique_ptr<unordered_map_deferred_operations>>;
    // ---------- Global static storage (one per CompressionManager instantiation) ----------
    // These statics are declared here for convenience; you can also move them into CompressionManager
    // if you prefer centralizing them there. They must be visible to both ChunkBase and CompressionManager.
    using global_map_type = ankerl::unordered_dense::map<int, per_thread_map>;

    // ── Compare and Swap ────────────────────────────────────────────────────
    /// @brief Type-tag helper: yields per_thread_map's type without needing
    ///        an instance. Always returns nullptr; only the pointer's static
    ///        type is ever used by callers.
    /// @return nullptr, typed as per_thread_map*.
    static per_thread_map* per_thread_map_type_id()
    {
        return nullptr;
    }

    /// @brief Atomic CAS of values from expected constant pointer to a new buffer.
    /// @param values_loc  New buffer to install on success.
    /// @param m_ptr_cst_val Expected current value of values. Taken by value on
    ///                      purpose — compare_exchange_strong would otherwise
    ///                      clobber a caller-owned reference on failure.
    /// @return true on failure (caller owns values_loc and must recup it).
    ///         false on success (values now points to values_loc).
    ///
    /// Fast path: a relaxed load of values lets us skip the CAS altogether when
    /// the chunk has obviously moved away from the expected constant state. This
    /// saves the cache line exclusive transition that compare_exchange_strong
    /// would force, which matters on ARM/AArch64 under contention.
    ///
    /// Safety on weak memory models (ARM, POWER):
    ///   - The relaxed load is never dereferenced; it is only compared to a
    ///     statically-known pointer, so stale reads cannot cause UB.
    ///   - A stale "match" (load sees old == expected while reality has moved)
    ///     is caught by the strong CAS that follows; the CAS is authoritative.
    ///   - A stale "mismatch" (load sees a newer value than what the CAS will
    ///     observe) is a benign spurious failure; the caller re-checks state.
    ///   - The acq_rel CAS on success provides the only ordering we rely on.
    [[nodiscard]] bool compare_and_swap_gen(T* values_loc, T* m_ptr_cst_val) noexcept
    {
        auto& self = static_cast<Derived&>(*this);
        CHUNK_CHECK_VALUES_PTR_C(values_loc, "CAS.values_loc", &self);
        CHUNK_CHECK_VALUES_PTR_C(m_ptr_cst_val, "CAS.expected_cst", &self);

        if (self.values.load(std::memory_order_relaxed) != m_ptr_cst_val) [[unlikely]]
        {
            return true; // failed (already changed)
        }

        return !self.values.compare_exchange_strong(m_ptr_cst_val, values_loc, std::memory_order_acq_rel,
                                                    std::memory_order_acquire);
    }

    /// @brief Compare-and-swap against first constant value
    /// @param values_loc New pointer value to set
    /// @return true if swap failed, false if successful
    [[nodiscard]] bool compare_and_swap1(T* values_loc) noexcept
    {
        return compare_and_swap_gen(values_loc, Derived::get_cst_val1());
    }

    /// @brief Reclaim a memory block to the thread-local free list
    /// @param values_loc Pointer to the block to reclaim
    /// @param thr_rank Thread rank/ID
    void recup_value(T* values_loc, int thr_rank)
    {
        auto& self = static_cast<Derived&>(*this);
        // Ledger: LIVE -> PARKED. This block was just allocated by create_value
        // and lost the install CAS; parking a block that is not LIVE (e.g. one
        // another thread already owns) trips here.
        CHUNK_LEDGER_PARK(values_loc, "recup_value");
        // Back to the list it was taken from / would have been taken from: the
        // array's, on this rank.
        auto& ll_block_free_thr = self.get_manager().get_list_block_free_thr(thr_rank, thread_alloc_key());
        ll_block_free_thr.push_back(values_loc);
    }

    // ── Deferred-op enqueue / flush / read gate ────────────────────────────
    /// @brief Ensure that the global deferred array is initialized exactly once.
    /// @details This replaces std::call_once with a CAS-based mechanism that can be reset.
    ///          - The first thread that flips deferred_initializing from false→true performs init.
    ///          - Other threads spin until deferred_initialized becomes true.
    ///          - This mechanism is fully resettable by clearing deferred_initialized.
    void ensure_deferred_array_initialized() const
    {
        if (get_deferred_ops() != nullptr) [[likely]]
            return;
        DeferredOpsStorage<per_thread_map>::instance().allocate();
    }

    /// @brief Current per-thread deferred-ops array, or nullptr if it has not
    ///        been allocated yet (see ensure_deferred_array_initialized()).
    /// @return Pointer to the per-thread map array, or nullptr.
    static per_thread_map* get_deferred_ops() noexcept
    {
        return DeferredOpsStorage<per_thread_map>::instance().get();
    }

    /// @brief Enqueue a deferred write operation for later application to a chunk.
    /// @param thr_rank Calling thread's rank; selects which per-thread map to
    ///                 enqueue on.
    /// @param num Element offset within the chunk the operation targets.
    /// @param v Deferred operation to enqueue; ownership transfers to the queue.
    void to_be_queued_for_deferred_processing(int                                                thr_rank,
                                              int                                                num,
                                              CompressionManager<T, Derived>::CmpDeferredOpBase* v) const;

    /// @brief Block until every deferred operation pending on this chunk has
    ///        been applied (drains this chunk across all thread ranks).
    /// @param num Element offset being read; currently unused -- the drain is
    ///            per-chunk, not per-offset. Kept because the call sites have
    ///            the offset in hand and the three DeferredReadGate modes must
    ///            agree on one signature.
    void wait_for_availability(int num) const;

    /// @brief Cheap, conservative pre-check for wait_for_availability(): true
    ///        means SOME deferred op is pending somewhere (possibly not for
    ///        this chunk); false is a precise guarantee that nothing is
    ///        pending anywhere.
    /// @return false only if it is certain nothing is pending anywhere.
    [[nodiscard]] bool is_not_available() const;
};

// Serial: none of the above exists -- no deferred writes, no CAS
// (nothing concurrent to protect against), no thread-local free list
// (nothing to contend over).
template <typename T, typename Derived>
struct ChunkDeferredOpsPolicy<T, Derived, false>
{
};

// ============================================================================
// ChunkDynamicAllocPolicy: the CHUNK_DYNAMIC-only per-size allocator pools
// (allocator_pool_/allocator_pool_serial_) and their registration methods.
// Static members, so unlike `values`/`pending_ops_` there is no object-layout
// concern moving them off ChunkBase -- a static member costs nothing per
// instance either way. Selected by build_config::chunk_dynamic (see config.h's
// own comment on that constant, which states the same declaration-vs-statement
// distinction this file's other policies are built around).
// get_chunk_allocator() (on ChunkBase) still reads these pools directly, via
// Derived::allocator_pool_/allocator_pool_serial_ -- restructuring that
// routing itself is a separate, later step.
// ----------------------------------------------------------------------------
template <typename T, typename Derived, bool IsDynamic = build_config::chunk_dynamic>
struct ChunkDynamicAllocPolicy; // primary left undefined: only the 2 specializations exist

template <typename T, typename Derived>
struct ChunkDynamicAllocPolicy<T, Derived, true>
{
    // ---------------------------------------------------------------------
    // Multi-size support (dynamic builds): a pool of per-thread allocator
    // arrays, keyed by chunk size. register_chunk_size() (called from the
    // sequential ArrayChunkBase::set_chunk_size, before any parallel alloc)
    // is the ONLY writer; get_chunk_allocator() only ever reads, so parallel
    // regions do lookups on a map that is not being mutated. std::map gives
    // reference/pointer stability across inserts, which matters because a new
    // size may be registered between two arrays' lifetimes while old entries
    // are still in use.
    // ---------------------------------------------------------------------
    /// @brief Registry key: (affinity key, chunk size). Key 0 is the rank's
    ///        shared pool -- what every array used before per-array affinity;
    ///        a non-zero key is one array's own allocator set, so that its
    ///        blocks sit in slabs holding nothing else and a re-chunk or a
    ///        destruction of that array empties whole slabs.
    static constexpr std::uint64_t pool_key(unsigned int key, unsigned int size) noexcept
    {
        return (static_cast<std::uint64_t>(key) << 32U) | size;
    }
    static inline std::map<std::uint64_t, std::vector<TCTypeCAllocator<T>>> allocator_pool_;

    /// @brief Register (create + init) the per-thread allocator array for
    ///        `size`, once.
    /// @details Sequential context only.
    /// @param size Chunk element count this allocator serves.
    /// @param max_thr Number of per-thread allocator slots to create.
    /// @param block_size Per-thread slab byte budget (mirrored from
    ///                   init_allocator_thr).
    /// @param key Affinity key (0 = shared rank pool). See pool_key().
    static void register_chunk_size(unsigned int size, int max_thr, int block_size, unsigned int key = 0)
    {
        if (allocator_pool_.contains(pool_key(key, size)))
        {
            return; // already registered
        }
        // Built whole (an allocator is not movable); moving the vector into the
        // map moves its buffer, not the allocators.
        std::vector<TCTypeCAllocator<T>> arr(static_cast<std::size_t>(max_thr));
        constexpr int                    v = std::max(static_cast<int>(8 / sizeof(T)), 1);
        for (int thr = 0; thr < max_thr; ++thr)
        {
            auto& allocator = arr[static_cast<std::size_t>(thr)];
            allocator.set_rank(thr); // stamped into every slab as owner_rank
            allocator.set_key(key);  // ... and owner_key
            allocator.init(static_cast<int>(Derived::storage_units(size)), block_size / v, -1);
        }
        allocator_pool_.emplace(pool_key(key, size), std::move(arr));
    }

    // Serial (non-OMP) counterpart of allocator_pool_ above: one allocator per
    // chunk size, WITHOUT the per-thread dimension (serial builds have a single
    // rank). TCTypeAllocator is polymorphic and non-movable (raw slab pointers +
    // a user destructor), so it is held by unique_ptr, not by value. Populated by
    // register_chunk_size_serial() from the sequential set_chunk_size path;
    // get_chunk_allocator() will route serial fast-alloc lookups here by the
    // active size (wired in a later step). This is what enables runtime re-chunk
    // in non-OMP fast-alloc builds: old and new sizes coexist during
    // rechunk_refill (read *this at old, write tmp at new), so a per-size pool
    // is required rather than a re-init of the single allocator_data.
    static inline std::map<unsigned int, std::unique_ptr<TCTypeAllocator<T>>> allocator_pool_serial_;

    /// @brief Register (create + init) the single allocator for `size`, once.
    /// @details Mirrors the serial init_allocator convention:
    ///          init(element_count, block_size, -1), where the first arg is
    ///          the chunk's element count (= size in a dynamic build, cf.
    ///          alloc_chunk_count()) and -1 = never munmap a value slab
    ///          (op/value slab aliasing fix, same as
    ///          init_allocator/init_allocator_thr).
    /// @param size Chunk element count this allocator serves.
    /// @param block_size Slab byte budget.
    static void register_chunk_size_serial(unsigned int size, int block_size)
    {
        if (allocator_pool_serial_.contains(size))
        {
            return; // already registered
        }
        auto allocator = std::make_unique<TCTypeAllocator<T>>();
        // FIX (op/value slab aliasing, non-OMP dynamic): tag this pool's slabs as
        // Value so they draw from value_arena() instead of silently defaulting to
        // arena_kind::Op (TCAllocator's default) -- which put value-chunk slabs and
        // the deferred-op pool's slabs in the SAME arena, letting a freed op slab's
        // address be reobtained here (or vice versa) and corrupting either pool's
        // block/block_full linked list with the other's slab header (observed:
        // sl_remove SIGSEGV on a node whose arena_kind read back as Op, with
        // ElementSize/BlockSize matching op-pool geometry, not this T's).
        allocator->set_arena(fast_alloc::ArenaKind::Value);
        allocator->init(static_cast<size_t>(Derived::storage_units(size)), static_cast<uintptr_t>(block_size), -1);
        allocator_pool_serial_.emplace(size, std::move(allocator));
    }

    // ── Routing ──
    // Lives here so get_chunk_allocator() can select between dynamic/static
    // routing with if constexpr: allocator_pool_ only needs to exist in THIS
    // specialization, where it always does -- a branch inside
    // get_chunk_allocator's own body would still phase-1-look-up
    // allocator_pool_ even when discarded, if it were referenced unqualified
    // there (see DeferredReadGate's own comment on this exact trap, above).

    /// @brief Route to the allocator for this thread's active chunk size. The
    ///        size must have been registered (set_chunk_size) beforehand.
    /// @param thr_rank Calling thread's rank.
    /// @return Reference to the per-thread allocator for the active size, or
    ///         to the legacy single allocator (Derived::allocator_data) if
    ///         the active size was never registered.
    static auto& route_omp_dynamic_alloc(int thr_rank, unsigned int key = 0) noexcept
    {
        const unsigned int sz = thread_chunk_size();
        if (const auto it = allocator_pool_.find(pool_key(key, sz)); it != allocator_pool_.end())
            return it->second[thr_rank];
        // Unregistered KEY at a registered size: the shared set of that size is
        // the right fallback -- allocator_data below is the legacy single pool,
        // whose geometry has nothing to do with `sz`.
        if (key != 0)
            if (const auto it0 = allocator_pool_.find(pool_key(0u, sz)); it0 != allocator_pool_.end())
                return it0->second[thr_rank];
        // Fallback: not registered (shouldn't happen on the alloc path
        // once set_chunk_size ran) — use the legacy single pool so the
        // free path and single-size use keep working.
        return Derived::allocator_data[thr_rank];
    }

    /// @brief Serial mirror of route_omp_dynamic_alloc(); no per-thread
    ///        dimension. Registered by register_chunk_size_serial() via
    ///        set_chunk_size before any alloc.
    /// @return Reference to the allocator for the active size, or to the
    ///         legacy single allocator (Derived::allocator_data) if
    ///         unregistered.
    static auto& route_serial_dynamic_alloc() noexcept
    {
        const unsigned int sz = thread_chunk_size();
        if (const auto it = allocator_pool_serial_.find(sz); it != allocator_pool_serial_.end())
            return *it->second; // deref unique_ptr
        // Fallback: unregistered size -> the legacy single allocator, so
        // single-size use and the free path keep working.
        return Derived::allocator_data;
    }
};

// Static (non-dynamic) builds: a single fixed geometry, so there is no
// per-size pool to keep -- allocator_data (on ChunkBase) is the only
// allocator that ever exists. Both routing methods just return it directly.
template <typename T, typename Derived>
struct ChunkDynamicAllocPolicy<T, Derived, false>
{
    /// @brief Static builds have no per-size pool: always route to the
    ///        single legacy allocator.
    /// @param thr_rank Calling thread's rank.
    /// @return Reference to Derived::allocator_data[thr_rank].
    static auto& route_omp_dynamic_alloc(int thr_rank, [[maybe_unused]] unsigned int key = 0) noexcept
    {
        return Derived::allocator_data[thr_rank];
    }

    /// @brief Serial mirror of the above; no per-thread dimension.
    /// @return Reference to Derived::allocator_data.
    static auto& route_serial_dynamic_alloc() noexcept
    {
        return Derived::allocator_data;
    }
};

// ============================================================================
// ChunkBase: Base class common to all chunk types (Chunk, Chunk2, Chunk2A)
// ============================================================================

template <typename T, typename Derived>
class ChunkBase : public CRTPSupport<Derived>,
                  public DeferredReadGate<Derived>,
                  public ChunkAllocPolicy<T, Derived>,
                  public ChunkDeferredOpsPolicy<T, Derived>,
                  public ChunkDynamicAllocPolicy<T, Derived>
{
    static_assert(ArrayChunkElement<T>,
                  "shrink_array element type: trivially copyable, ==, default constructible "
                  "(see concept_array_chunk_element.h)");

  public:
    // Type traits
    // bool and Bit (bit.h) share chunk2's bool path: one second value, true.
    static constexpr bool is_bool     = is_bool_like_v<T>;
    static constexpr bool is_not_bool = !is_bool;
    /// The buffer is words of bits (Bit), not an array of T.
    static constexpr bool is_packed = is_packed_bit_v<T>;

    static constexpr bool use_mm_protect = build_config::use_mm_protect;
    constexpr static T    zero{};

    // Atomic pointer type conditional on OpenMP availability
    using atomic_ptr_t = std::conditional_t<build_config::use_openmp, std::atomic<T*>, T*>;

    using ProxyType = ProxyTypeTrait<Derived>::Type;

    using CRTPSupport<Derived>::derived;
    using CRTPSupport<Derived>::self;

    // std-alloc placeholder for `allocator_data`. It satisfies the fast
    // allocator's call interface with no-op stubs so that manager code shared
    // by both modes (init/destroy/alloc/free paths) compiles unchanged in std
    // builds. It never does real work: std-alloc routes allocation through
    // svalloc, and every site that would call these stubs is guarded by
    // `if constexpr (build_config::use_fast_alloc)` — discarded at compile time
    // in std builds, so those guards are plain shared code.
    struct NullValueAllocator
    {
        /// @brief No pool to set up.
        void init(int /*element_size*/, int /*block_size*/, int /*munmap_flag*/) noexcept {}
        /// @brief No pool to tear down.
        void destroy() noexcept {}
        /// @brief Never allocates: std-alloc buffers come from svalloc.
        [[nodiscard]] T* alloc() noexcept
        {
            return nullptr;
        }
        /// @brief Nothing to return to a pool.
        void free_elt_pow2(const T* /*ptr*/) noexcept {}
        /// @brief No-op counterpart to TCTypeAllocator::set_arena, so call
        ///        sites that set the arena unconditionally (before init())
        ///        compile in std-alloc builds too.
        void set_arena(fast_alloc::ArenaKind) noexcept {}
    };

    using AllocatorType = std::conditional_t<
        build_config::use_fast_alloc,
        std::conditional_t<build_config::use_openmp, std::vector<TCTypeCAllocator<T>>, TCTypeAllocator<T>>,
        NullValueAllocator>;

    static inline AllocatorType allocator_data;

    // allocator_pool_/allocator_pool_serial_/register_chunk_size/
    // register_chunk_size_serial moved to ChunkDynamicAllocPolicy<T, Derived,
    // true> above (empty in the false specialization -- static builds have no
    // per-size pool).

    // ChunkDeferredOpsPolicy<T, Derived, true> above; ChunkBase now inherits
    // them (empty in the false specialization -- nothing exists in serial).

    /// @brief Get the allocator to use for the current thread and active
    ///        chunk size.
    /// @details Only meaningful when build_config::use_fast_alloc is set;
    ///          otherwise aborts (this build never uses a fast allocator, so
    ///          this path should never be reached). Dynamic-vs-static
    ///          routing (per-size pool vs the single legacy allocator) is
    ///          delegated to Derived::route_omp_dynamic_alloc()/
    ///          route_serial_dynamic_alloc(), selected on
    ///          build_config::use_openmp.
    /// @param thr_rank Calling thread's rank; only meaningful in OpenMP
    ///                 builds (ignored in serial builds).
    /// @return Reference to the allocator to use for this thread/build mode.
    /// @param key Affinity key of the array being served (0 = shared rank
    ///            pool). Only the dynamic OpenMP registry honours it so far.
    static auto& get_chunk_allocator([[maybe_unused]] int          thr_rank,
                                     [[maybe_unused]] unsigned int key = thread_alloc_key()) noexcept
    {
        if constexpr (build_config::use_fast_alloc)
        {
            // Dynamic-vs-static routing lives on ChunkDynamicAllocPolicy (both
            // specializations declare route_omp_dynamic_alloc/
            // route_serial_dynamic_alloc -- the true specialization consults
            // allocator_pool_/allocator_pool_serial_, the false one always falls
            // through to allocator_data).
            if constexpr (build_config::use_openmp)
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
            std::abort();
        }
    }

    ChunkBase()  = default;
    ~ChunkBase() = default;

    /// @brief Whether this chunk is currently materialised (not compressed
    ///        to a constant), at the granularity Derived supports.
    /// @param j Element offset; used only when Derived distinguishes
    ///          writtenness per-offset (e.g. Chunk2/Chunk2A's second-value
    ///          slot); ignored otherwise (falls back to derived().written()).
    /// @return true if the chunk (or offset j) holds real, materialised data.
    [[nodiscard]] bool is_written_at(int j) const noexcept
    {
        if constexpr (requires { derived().written(j); })
        {
            // Use specialized logic from Derived (e.g., ArrayChunk2)
            return derived().written(j);
        }
        else
        {
            // Default fallback for ArrayChunk and ArrayChunk2A
            return derived().written();
        }
    }

    /// @brief Debug-only assertion: abort if offset j is not currently
    ///        written (a clear on an unwritten offset is a bug). Compiled
    ///        out entirely when build_config::any_check is off.
    /// @param j Element offset expected to be written.
    void debug_check_written(int j) const noexcept
    {
        if constexpr (build_config::any_check)
        {
            if (!is_written_at(j))
            {
                std::cerr << "Error: trying to clear a chunk that is not written\n";
                std::abort();
            }
        }
    }

    /// @brief Debug-only assertion: abort if this chunk is not currently
    ///        written. Compiled out entirely when build_config::any_check is
    ///        off.
    void debug_check_written() const noexcept
    {
        if constexpr (build_config::any_check)
        {
            if (!derived().written())
            {
                std::cerr << "Error: trying to clear a chunk that is not written\n";
                std::abort();
            }
        }
    }
    // ========================================================================
    // Atomic / Non-atomic pointer management
    // ========================================================================

    /**
     * @brief The element at @p i, whatever the chunk's state.
     *
     * @details The single point every element READ goes through. Today it is
     *          one dereference: a compressed chunk points at a shared array
     *          filled with its constant, so a read needs no test. That is the
     *          property that keeps the header at 16 bytes, and it is bought
     *          with one filled array per distinct second value -- 32 KiB each
     *          in dyn, where the stride follows chunk_dyn_max_size rather than
     *          the geometry in use.
     *
     *          Dropping those arrays means reading the constant from the value
     *          registry instead, i.e. branching on is_cst_val2(). This exists
     *          so that branch has one place to live rather than seven.
     *
     * @note NOT used by print_adr() or operator()(int): those hand out an
     *       address and a mutable reference into the buffer, which a registry
     *       lookup cannot stand in for.
     */
    /// @tparam Relaxed read the element through a relaxed atomic_ref: for an
    ///         element another thread may be writing (relaxed_element.h).
    template <bool Relaxed = false>
    [[nodiscard]] T element_at(int i) const noexcept
    {
        if constexpr (build_config::cst_val2_no_arrays)
        {
            // The one branch this flag buys: under it a compressed-to-second-
            // constant chunk points into a region that was never written, so
            // the constant comes from the value table instead. The test is a
            // pointer range test and it is very well predicted -- 83% of reads
            // on the reference run go to materialised chunks.
            //
            // ONE load. Asking derived().is_cst_val2() and then dereferencing
            // values_load() loads the pointer twice, and a compressor claiming
            // the chunk in between swaps it for a region identifier: the second
            // load then indexes that identifier as a buffer, in a page that was
            // never mapped. SIGSEGV at 24 threads, in element_at, on a valid
            // index. Same defect, same fix as read_values().
            // Through the CRTP hooks, so chunk_base keeps knowing nothing of
            // the registry -- but taking the pointer, so the test and the read
            // are about the same load.
            //
            // Gated on the variant, NOT on a `requires` probe for the hooks. A
            // probe fails OPEN: where it evaluates false the code falls through
            // to values_load()[i], which on a compressed chunk reads the
            // unwritten region and returns 0 -- a wrong number, not a build
            // error. That happened twice: once with the hooks genuinely
            // missing, and once under nvhpc, whose concept evaluation on the
            // still-incomplete CRTP Derived did not see them while gcc and
            // clang did. std::is_same_v against the forward-declared Chunk2A
            // (chunk_traits.h) is a plain type comparison: no concepts, no
            // member lookup, nothing a compiler can decide differently. It also
            // works for the variants that have no such member at all.
            if constexpr (std::is_same_v<Derived, Chunk2A<T>>)
            {
                static_assert(
                    requires(const Derived& obj, const T* ptr) {
                        obj.is_cst_val2_ptr_of(ptr);
                        obj.second_value_from_ptr(ptr);
                    }, "Chunk2A must provide is_cst_val2_ptr_of() and second_value_from_ptr() "
                       "when second values carry identifiers (see CHUNK_CST_VAL2_FILLED_ARRAYS)");
                const T* const vals = values_load();
                if (derived().is_cst_val2_ptr_of(vals)) [[unlikely]]
                    return derived().second_value_from_ptr(vals);
                return load_at<Relaxed>(vals, i);
            }
        }
        return load_at<Relaxed>(values_load(), i);
    }

    template <bool Relaxed>
    [[nodiscard]] static T load_at(const T* vals, int i) noexcept
    {
        if constexpr (is_packed)
            return elem_get(vals, static_cast<std::size_t>(i)); // relaxed word load under OpenMP
        else if constexpr (Relaxed)
            return internal::load_element_relaxed(vals + i);
        else
            return vals[i];
    }

    /// @brief Load the values pointer with appropriate memory ordering
    /// @return Pointer to the chunk data
    T* values_load() const noexcept
    {
        if constexpr (build_config::use_openmp)
        {
            return values.load(std::memory_order_acquire);
        }
        else
        {
            return values;
        }
    }

    /// @brief Store a pointer to the values with appropriate memory ordering
    /// @param new_values Pointer to store
    void values_store(T* new_values) noexcept
    {
        CHUNK_CHECK_VALUES_PTR_C(new_values, "values_store", this);
        if constexpr (build_config::use_openmp)
        {
            values.store(new_values, std::memory_order_release);
        }
        else
        {
            values = new_values;
        }
    }

    // ========================================================================
    // Constant value testing (first value)
    // ========================================================================
    /// @brief The (read-only) buffer every compressed-to-first-value chunk
    ///        points at.
    /// @return Pointer to the shared first-constant buffer.
    static T* get_cst_val1() noexcept
    {
        if constexpr (use_mm_protect)
        {
            return m_ptr_cst_val1;
        }
        else
        {
            return m_ptr_cst_val1.data();
        }
    }

    /// @brief Check if chunk points to the first constant value
    /// @return true if chunk is compressed to first constant value
    [[nodiscard]] bool is_cst_val1() const noexcept
    {
        const T* current_value = values_load();
        return current_value == ChunkBase::get_cst_val1();
    }

    /// @brief Check if chunk points to the first constant value
    /// @return true if chunk is compressed to first constant value
    [[nodiscard]] bool is_cst_val1(const T* current_value) const noexcept
    {
        return current_value == ChunkBase::get_cst_val1();
    }

    /// @brief Check if chunk does not point to the first constant value
    /// @return true if chunk is not compressed to first constant value
    [[nodiscard]] bool is_not_cst_val1() const noexcept
    {
        return !is_cst_val1();
    }

    // ========================================================================
    // Value reading operations
    // ========================================================================

    /// @brief Element read performed by an INTERNAL scan (compression
    ///        candidate detection), not by the application.
    /// @details Counted separately: these scans read the whole chunk twice
    ///          per element, which would otherwise dominate `read` and make
    ///          the read hit-rate meaningless.
    /// @param i Index in the chunk.
    /// @return Value at index i.
    [[nodiscard]] T read_value_scan(int i) const noexcept
    {
        if constexpr (chunk_config::stats)
        {
            increm_chunk_scan_read<Derived>();
        }
        return element_at(i);
    }

    /// @brief read_value_scan() of an element another thread may be writing:
    ///        a relaxed atomic load (relaxed_element.h). The marking filter's.
    /// @pre internal::relaxed_element_access_v<T>.
    [[nodiscard]] T read_value_scan_relaxed(int i) const noexcept
    {
        if constexpr (chunk_config::stats)
        {
            increm_chunk_scan_read<Derived>();
        }
        return element_at<true>(i);
    }

    /// @brief Outcome of scan_constants(): which hypothesis survived the walk.
    struct ConstantScan
    {
        bool first  = false; ///< every element equals the first constant
        bool second = false; ///< every element equals the second constant -- never together with `first`
    };

    /**
     * @brief One walk over the materialised buffer against two constants.
     *
     * @details Tracks "all @p first_v" and "all @p second_v" together and stops
     *          as soon as both are disproved. When the two constants coincide a
     *          constant run satisfies both; it is reported as `first` ONLY --
     *          which is what every caller wants (fill_list_to_compress files it
     *          once, compress_int must not clear it twice). Loads the values
     *          pointer once; the reads are counted as scan work, not as
     *          application reads (see read_value_scan).
     *
     *          This is THE constant test of the library: the compression
     *          candidates, the end-of-run audits and the size predictor all
     *          used to carry their own copy of this loop.
     *
     * @param first_v  first hypothesis, usually first_value
     * @param second_v second hypothesis
     * @param count        elements to scan -- the chunk's OWN geometry, never the
     *                 ambient size (a candidate is scanned by whichever thread
     *                 drains it, whose ambient size may differ after a re-chunk)
     * @pre The chunk is materialised: `values` points at a real buffer.
     */
    [[nodiscard]] ConstantScan scan_constants(in_t<T> first_v, in_t<T> second_v, unsigned int count) const noexcept
    {
        // Block-wise, branch-free inside a block: an early exit per element
        // kept the compiler from vectorising the walk, and on the perf bench
        // 98.7% of the scans run to the end anyway (the chunk IS constant --
        // 4.4 G scanned elements per reference run, all of them compared
        // twice, one at a time). Inside a block the two verdicts are OR
        // reductions of comparisons, which vectorise (AVX2 / NEON); the early
        // exit survives between blocks, so a chunk disproved at its first
        // element still costs one block, not the whole chunk.
        const T* vals = values_load();
        if constexpr (is_packed)
        {
            // One compare per 64 elements (bit_words.h).
            const bits::Uniform u        = bits::scan_words(internal::as_words(vals), count);
            const auto          all_of_v = [u](bool v) noexcept
            { return u == (v ? bits::Uniform::all_true : bits::Uniform::all_false); };
            if constexpr (chunk_config::stats)
                add_chunk_scan_read<Derived>(static_cast<long>(count));
            ConstantScan packed{all_of_v(first_v), all_of_v(second_v)};
            if (packed.first)
                packed.second = false;
            return packed;
        }
        ConstantScan scan{true, true};
        unsigned int i = 0;
        while (i < count)
        {
            const unsigned int block_end  = std::min(count, i + kScanBlock);
            unsigned int       not_first  = 0;
            unsigned int       not_second = 0;
            for (unsigned int j = i; j < block_end; ++j)
            {
                not_first |= static_cast<unsigned int>(vals[j] != first_v);
                not_second |= static_cast<unsigned int>(vals[j] != second_v);
            }
            i           = block_end;
            scan.first  = scan.first && not_first == 0;
            scan.second = scan.second && not_second == 0;
            if (!scan.first && !scan.second)
                break;
        }
        if constexpr (chunk_config::stats)
            add_chunk_scan_read<Derived>(static_cast<long>(i)); // block granularity
        if (scan.first)
            scan.second = false;
        return scan;
    }

    /// @brief Elements per branch-free block of the constant scans.
    ///        A quarter of a 256-element chunk: four early-exit points, and a
    ///        block long enough for the vector loop to matter.
    static constexpr unsigned int kScanBlock = 64;

    /// @brief Is the materialised buffer all @p v over its first @p count elements?
    /// @pre The chunk is materialised.
    [[nodiscard]] bool is_constant_to(in_t<T> v, unsigned int count) const noexcept
    {
        if constexpr (is_packed)
        {
            if constexpr (chunk_config::stats)
                add_chunk_scan_read<Derived>(static_cast<long>(count));
            return bits::scan_words(internal::as_words(values_load()), count) ==
                   (static_cast<bool>(v) ? bits::Uniform::all_true : bits::Uniform::all_false);
        }
        // Same shape as scan_constants(): branch-free blocks, exit between them.
        const T*     vals = values_load();
        unsigned int i    = 0;
        while (i < count)
        {
            const unsigned int block_end = std::min(count, i + kScanBlock);
            unsigned int       bad       = 0;
            for (unsigned int j = i; j < block_end; ++j)
                bad |= static_cast<unsigned int>(vals[j] != v);
            i = block_end;
            if (bad != 0)
            {
                if constexpr (chunk_config::stats)
                    add_chunk_scan_read<Derived>(static_cast<long>(i));
                return false;
            }
        }
        if constexpr (chunk_config::stats)
            add_chunk_scan_read<Derived>(static_cast<long>(count));
        return true;
    }

    /// @brief The one value the materialised buffer holds over @p count elements, if any.
    /// @return that value, or nullopt when at least two elements differ
    /// @pre The chunk is materialised and @p count > 0.
    [[nodiscard]] std::optional<T> constant_value(unsigned int count) const noexcept
    {
        if (const in_t<T> v0 = element_at(0); is_constant_to(v0, count))
            return v0;
        return std::nullopt;
    }

    /// @brief Read a value at given index with statistics tracking
    /// @param i Index in the chunk
    /// @return Value at index i
    [[nodiscard]] T read_value(int i) const noexcept
    {
        if constexpr (chunk_config::stats)
        {
            // Split the read by what actually served it: a materialised chunk
            // (real memory access) vs a compressed one (values points at the
            // constant marker, so the indexing returns it with no data touched).
            // read_cst / (read + read_cst) is the compression hit rate on reads.
            // is_cst_val1() is used rather than written(): it is index-agnostic,
            // whereas Chunk2::written() (no arg) deliberately aborts on non-bool
            // types -- it needs the second-value index, which a read does not have.
            if (is_cst_val1())
                increm_chunk_read_cst<Derived>();
            else
                increm_chunk_read<Derived>();
        }
        return element_at(i);
    }

    // reader_sync(i) — the read-side visibility gate used by the pure-read proxy
    // entry points under build_config::sync_on_read — is inherited from
    // DeferredReadGate<Derived> above (None/Coarse/Precise per build).

    /// @brief Get the address of a value at given index
    /// @param i Index in the chunk
    /// @return Pointer to the value at index i
    [[nodiscard]] T* print_adr(int i) const noexcept
    {
        // An ADDRESS into the buffer, which the registry cannot stand in for:
        // under CHUNK_CST_VAL2_NO_ARRAYS there is no buffer. Materialise first,
        // so the address is real and the semantics are exactly what they were.
        // Its only caller is ArrayChunkBase's re-export, itself never called
        // internally -- this is a diagnostic path.
        if constexpr (build_config::cst_val2_no_arrays && requires {
                          derived().is_cst_val2();
                          derived().materialise_for_address();
                      })
        {
            if (derived().is_cst_val2()) [[unlikely]]
                derived().materialise_for_address();
        }
        static_assert(!is_packed, "print_adr: a Bit has no address (bit.h)");
        return &values_load()[i];
    }

    // ========================================================================
    // Read-only operator[]
    // ========================================================================

    /// @brief Read-only access to chunk elements with optional NaN checking
    /// @param i Index in the chunk
    /// @return Value at index i
    T operator[](int i) const noexcept
    {
        if constexpr (chunk_config::stats)
        {
            // Split the read by what actually served it: a materialised chunk
            // (real memory access) vs a compressed one (values points at the
            // constant marker, so the indexing returns it with no data touched).
            // read_cst / (read + read_cst) is the compression hit rate on reads.
            // is_cst_val1() is used rather than written(): it is index-agnostic,
            // whereas Chunk2::written() (no arg) deliberately aborts on non-bool
            // types -- it needs the second-value index, which a read does not have.
            if (is_cst_val1())
                increm_chunk_read_cst<Derived>();
            else
                increm_chunk_read<Derived>();
        }

        chunk_config::CheckNaN<chunk_config::do_check>::maybe(element_at(i), static_cast<long>(i));

        return element_at(i);
    }

    /// @brief Proxy access for read-write operations
    /// @param i Index in the chunk
    /// @return Proxy object for lazy write
    ProxyType operator[](int i) noexcept
    {
        return ProxyType(values, i, derived());
    }

    // ========================================================================
    // operator() - Direct access without checks
    // ========================================================================

    /// @brief Direct read-write access to chunk elements without safety checks
    /// @param i Index in the chunk
    /// @return Reference to value at index i
    T& operator()(int i) const noexcept
    {
        if constexpr (chunk_config::stats)
        {
            increm_chunk_read<Derived>();
        }
        // Contract, now executable: this hands out a mutable reference into
        // the buffer and is documented as "bypasses the write proxy: valid
        // only where the chunk is known to be materialised". On a compressed
        // chunk the reference points into the SHARED constant array, so a
        // write through it would corrupt the constant for every chunk holding
        // it -- true today, and a dereference of an unbacked page under
        // CHUNK_CST_VAL2_NO_ARRAYS. Checked builds say so; release is
        // unchanged.
        if constexpr (build_config::any_check && requires { derived().written(); })
        {
            if (!derived().written()) [[unlikely]]
                INVARIANT_FAILED("operator()(int) on a chunk that is not materialised");
        }
        static_assert(!is_packed, "operator()(int) returns T&: a Bit has no address (bit.h)");
        return values_load()[i];
    }

    // ========================================================================
    // Block initialization
    // ========================================================================

    /// @brief Set @p n_elems elements of @p values_loc to the first value,
    ///        T{}: memset for arithmetic types, std::fill_n otherwise.
    /// @param values_loc block to initialise
    /// @param n_elems    the block's OWN element count -- callers that know the
    ///                   buffer's geometry MUST use this overload; the no-count
    ///                   form falls back to the ambient size, which overflows a
    ///                   buffer allocated at a different geometry.
    static void init_block(T* values_loc, unsigned int n_elems) noexcept
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

    /**
     * @brief Units of T to allocate for a chunk of @p n_elems elements.
     * @details One per element: element i is values[i]. A packed chunk
     *          (ChunkBits: 8 bools per byte) hides this with fewer, and every
     *          allocation, allocator geometry and byte count goes through
     *          Derived::storage_units(), never through the element count alone.
     */
    [[nodiscard]] static constexpr std::size_t storage_units(std::size_t n_elems) noexcept
    {
        if constexpr (is_packed)
            return bits::bytes_for(n_elems) / sizeof(T); // whole 64-bit words
        else
            return n_elems;
    }

    /// @brief init_block() over the active geometry's storage.
    static void init_block(T* values_loc) noexcept
    {
        const std::size_t units = Derived::storage_units(alloc_chunk_count());
        if constexpr ((std::is_arithmetic_v<T> && std::is_trivially_constructible_v<T>) || is_packed)
        {
            // For trivially constructible scalar types: use memset (fastest option)
            std::memset(static_cast<void*>(values_loc), 0, units * sizeof(T)); // Bit: raw words, not Bit objects
        }
        else
        {
            // For all other cases (composite_arithmetic, non-trivial types, etc.)
            std::fill_n(values_loc, units, T{});
        }
    }

    /// @brief Initialize the current chunk's memory block
    void init_block() noexcept
    {
        T* val = values_load();
        init_block(val);
    }

    // ========================================================================
    // Compare and Swap (OpenMP only)
    // ========================================================================

    // per_thread_map_type_id/compare_and_swap_gen/compare_and_swap1/recup_value
    // moved to ChunkDeferredOpsPolicy<T, Derived, true> above.
    // alloc_values(bool, int) lived here: v1's private allocation path, a
    // second copy of what ChunkAllocPolicy::allocate_chunk_internal already
    // did. Keeping the two in step never worked -- the bounds-checked reuse
    // lookup had to be fixed in each of them separately -- so create_value()
    // now goes through allocate_chunk_internal like v2/v2A, and this went with
    // it. Its `init == false` mode (svalloc_align, uninitialized) had no caller
    // left at all.
    // ========================================================================
    // Chunk allocation with free block cache management
    // ========================================================================

    /// @brief Return a values buffer to the underlying allocator -- a real
    ///        free, not a park onto the thread-local reuse list (see
    ///        free_block_to_pool for that).
    /// @param ptr Buffer to free.
    /// @param thr_rank Calling thread's rank (selects the allocator pool in
    ///                 fast-alloc/OMP builds; ignored otherwise).
    void free_values(T* ptr, [[maybe_unused]] int thr_rank = 0)
    {
        // SEGV hunt: record value-buffer frees (aliasing investigation).
        if constexpr (chunk_config::ring_log_debug || chunk_config::ring_log_op_only)
            ring_log_op(get_tid(), EV_VALUE_FREE, 0, static_cast<const void*>(ptr));
        // fw15 (ledger balance): free_values returns the block for real (no park
        // into a freelist), so notify the ledger LIVE -> FREE. Without this the
        // POOL_LEDGER (any_check builds) tracked the alloc but not this free: in
        // std-alloc, svfree() hands the page back to malloc, which may hand the
        // SAME address to a later allocate_chunk_internal -> on_block_alloc sees
        // it still LIVE -> spurious INVARIANT_FAILED. (fast-alloc never surfaced
        // it: freed blocks stay in the immortal arena, never re-entering via
        // on_block_alloc on a LIVE state.) Must precede the actual free, while
        // the pointer is still ours. This is instrumentation-only; production
        // (no any_check) compiles it out.
        CHUNK_LEDGER_RELEASE(static_cast<const void*>(ptr), "free_values");
        if constexpr (build_config::use_fast_alloc)
        {
            release_to_owner(thr_rank, ptr);
        }
        else
        {
            svfree(ptr);
        }
    }
    /// @brief Return @p ptr to the allocator instance that carved it, found
    ///        through its slab (TCAllocator::owner / owner_rank), never
    ///        through the caller's rank.
    /// @details A slab chain may only be mutated by the rank that owns it while
    ///          a region is open. So the block is freed directly when it is
    ///          ours, unranked (serial pools), or when no region is open; a
    ///          foreign block inside a region is parked on OUR list under ITS
    ///          affinity key, where this rank may reuse it for that array and
    ///          the next drain routes it home. Before the owner fields existed
    ///          this freed into get_chunk_allocator(thr_rank) -- the caller's
    ///          instance -- which for a foreign block mutated another rank's
    ///          chains from the wrong thread.
    /// @pre fast-alloc build.
    void release_to_owner(int thr_rank, T* ptr)
    {
        // Callers such as block_clear()/clear_st() pass the default rank 0 from
        // inside a region: the rank that matters here is the one running this
        // code, both for "is it mine" and for which list a foreign block is
        // parked on (a rank only ever touches its own list inside a region).
        if (CHUNK_PMP_IN_PARALLEL_REGION)
            thr_rank = CHUNK_PMP_THREAD_RANK;
        auto& mine = get_chunk_allocator(thr_rank); // same slab geometry for every key
        auto* slab = mine.slab_of(ptr);
        if (slab == nullptr || slab->owner == nullptr)
        {
            // No slab behind this pointer (bypass_slabs, or an instance never
            // initialised): the legacy route is the only one there is.
            mine.free_elt_pow2(ptr);
            return;
        }
        auto* owner = static_cast<std::remove_reference_t<decltype(mine)>*>(const_cast<void*>(slab->owner));
        if (slab->owner_rank < 0 || slab->owner_rank == thr_rank || !CHUNK_PMP_IN_PARALLEL_REGION)
        {
            CHUNK_FREEPATH_INC(freed);
            owner->free_elt_pow2(ptr, slab);
            return;
        }
        CHUNK_FREEPATH_INC(park_foreign);
        CHUNK_LEDGER_PARK(ptr, "release_to_owner.park_foreign");
        derived().get_manager().get_list_block_free_thr(thr_rank, slab->owner_key).push_back(ptr);
    }

    /// @brief Free this chunk's current values buffer.
    /// @param thr_rank Calling thread's rank; forwarded to free_values(T*, int).
    void free_values([[maybe_unused]] int thr_rank = 0)
    {
        T* ptr = values_load();
        free_values(ptr, thr_rank);
    }

    // allocate_chunk_internal (both OMP and serial bodies) lives on
    // ChunkAllocPolicy<T, Derived, UseOpenMP> above -- ChunkBase inherits it,
    // selected by build_config::use_openmp. Every call site uses
    // Base::allocate_chunk_internal(...) (Base = ChunkBase<T,Derived> or a
    // class inheriting it), which resolves through inheritance identically
    // to a member declared directly here.

    // ========================================================================
    // Chunk deallocation
    // ========================================================================

    /// @brief Return a block to the free-pool, preferring the local thread's
    ///        free list over a real free -- and never pushing a foreign
    ///        block onto ANOTHER thread's free-list (see the out-of-line
    ///        definition for the full cross-thread-migration rationale).
    /// @param thr_rank Calling thread's rank.
    /// @param val Block to return.
    void free_block_to_pool([[maybe_unused]] int thr_rank, T* val);

    // ensure_deferred_array_initialized/get_deferred_ops/
    // to_be_queued_for_deferred_processing/wait_for_availability/
    // is_not_available moved to ChunkDeferredOpsPolicy<T, Derived, true>
    // above (bodies of the last 3 still out-of-line, in chunk_base_inline.h).


    // ========================================================================
    // Member variables
    // ========================================================================

    /// Pointer to chunk data (atomic if OpenMP enabled)
    alignas(std::atomic<T*>) atomic_ptr_t values;

#if !defined(SHRINK_SMALL_CHUNK) && defined(_OPENMP)
    /// Per-chunk count of deferred ops currently pending (enqueued but not yet
    /// drained). Present only when the deferred-op machinery exists, i.e. NOT
    /// under -DSHRINK_SMALL_CHUNK (which trades it away for 16-byte chunks) and
    /// only in OpenMP builds — the deferred-op enqueue/flush paths are all
    /// OpenMP-only, so in a serial build this counter is pure dead weight (8
    /// bytes with alignment). +1 (release) in enqueue BEFORE the map insert so
    /// pending_ops_>0 spans a pending op's whole lifetime (closes a reader
    /// stale-read window); -flushed (release) in flush_deferred /
    /// flush_chunk_all_ranks. The read path (sync_chunk) acquire-loads it: 0
    /// means this chunk has nothing pending -> skip wait_for_availability.
    ///
    /// IMPORTANT: the member guard and its ONLY reader (ArrayChunkBase::
    /// sync_chunk's precise gate) share this exact #if condition, so both appear
    /// or vanish together textually. Guarding the member alone while the reader
    /// stays under `if constexpr` fails: gcc still phase-1-looks-up
    /// chunk_t.pending_ops_ in the discarded branch and errors on the missing
    /// member. In non-OpenMP the reader falls back to the coarse is_not_available
    /// gate, which has no cost in a serial build (no concurrent flush).
    std::atomic<int32_t> pending_ops_{0};

    /// @brief "A compressor is about to scan and may steal this chunk's
    ///        buffer": the per-chunk half of the write/compress handshake.
    ///
    /// Set (relaxed) by a compressor on every chunk of its candidate list
    /// BEFORE its start() -- start()'s heavy barrier is what publishes the
    /// marks to the writers, exactly as it publishes hard_inflight_ -- and
    /// cleared (release) per chunk as the pass goes: right after the scan
    /// for a chunk that stays materialised, right after the CAS-steal for
    /// one that is compressed. A writer reads it (relaxed, after its own
    /// announce_write) on the line it loads `values` from anyway: it waits,
    /// then defers, ONLY when its chunk is marked. Every other writer goes on
    /// writing while the compressor scans -- the scan used to be a global
    /// exclusion window, ~1 ms per pass, paid by every other thread (63 of
    /// them on the 64-thread AArch64 node: +1.2 s on the init phases).
    ///
    /// Lives in the 4 bytes of padding after pending_ops_ (ChunkBase is
    /// 8-byte aligned on `values`), so it costs nothing: sizeof(Chunk2A<double>)
    /// stays 24. Same #if as pending_ops_, for the same gcc phase-1 lookup
    /// reason; SHRINK_SMALL_CHUNK builds keep the global gate
    /// (chunk_config::use_chunk_steal_mark).
    std::atomic<uint32_t> steal_mark_{0};

    /// @brief Is a compressor's pass in flight on this chunk?
    ///
    /// @details ACQUIRE, not relaxed. The old comment justified relaxed with
    ///          "the ordering against the compressor is the heavy barrier of
    ///          start()", which holds in one direction only: start()'s
    ///          membarrier() stands in for the fence the writers no longer
    ///          execute, so it publishes the WRITERS' announces to the
    ///          compressor. The mark travels the other way -- compressor to
    ///          writer -- and a membarrier() run by the compressor does not
    ///          push its own store into a writer already in flight. A writer
    ///          could read the mark clear after it was set, take the fast path
    ///          and write while the scan read: TSan, 8 threads,
    ///          store_value() against read_value_scan().
    ///
    ///          Read once per write, so this is an ldar in place of an ldr.
    [[nodiscard]] bool steal_marked() const noexcept
    {
        return steal_mark_.load(std::memory_order_acquire) != 0;
    }
    /// @brief Compressor side, before start(): claim this chunk for the pass.
    /// @details RELEASE, pairing with steal_marked()'s acquire: a writer that
    ///          sees the mark sees everything the compressor did before
    ///          claiming. Written once per pass.
    void steal_mark() noexcept
    {
        steal_mark_.store(1, std::memory_order_release);
    }
    /// @brief Compressor side, AFTER start() returned true (check builds
    ///        only): the window is shut, no writer holds a flag on a marked
    ///        chunk -- commit the claim. A writer's store_value() on a chunk
    ///        in this state is a violation of the handshake; on a chunk merely
    ///        marked (1) it is not: the compressor may have marked it after the
    ///        writer's check #2, and is then waiting on the writer's flag.
    void steal_commit() noexcept
    {
        steal_mark_.store(2, std::memory_order_seq_cst);
    }
    /// @brief The claim was committed: the compressor is past its window.
    [[nodiscard]] bool steal_committed() const noexcept
    {
        return steal_mark_.load(std::memory_order_seq_cst) == 2;
    }
    /// @brief Compressor side: the pass is done with this chunk (scanned and
    ///        left alone, or stolen). Release: a writer that then reads 0 sees
    ///        the chunk's new state (constant pointer, metadata).
    void steal_unmark() noexcept
    {
        steal_mark_.store(0, std::memory_order_release);
    }
#else
    // No mark without OpenMP (no concurrent compressor) or under
    // SHRINK_SMALL_CHUNK (no room): the write gate is the global one.
    /// @brief No per-chunk mark in this configuration: never marked.
    [[nodiscard]] static constexpr bool steal_marked() noexcept
    {
        return false;
    }
    /// @brief No per-chunk mark in this configuration: no-op.
    static void steal_mark() noexcept {}
    /// @copydoc steal_mark()
    static void steal_commit() noexcept {}
    /// @brief No per-chunk mark in this configuration: never committed.
    [[nodiscard]] static constexpr bool steal_committed() noexcept
    {
        return false;
    }
    /// @copydoc steal_mark()
    static void steal_unmark() noexcept {}
#endif

    /// First constant value for compression: the value-initialised T. Spelled
    /// T{} and not 0 everywhere in the library, so that an aggregate (a struct
    /// of doubles, a std::array) is a first-class element type -- `T{0}` would
    /// initialise its first member only, `T(0)` not compile.
    static inline T first_value{};

    using cst_val1_type = std::conditional_t<build_config::use_mm_protect, T*, std::array<T, cst_buffer_size>>;

    // alignas: a Bit chunk compressed to the first constant points here, and
    // its reads load 64-bit words (element_access.h).
    alignas(bits::word_t) static inline cst_val1_type m_ptr_cst_val1 = []
    {
        if constexpr (build_config::use_mm_protect)
        {
            // --- branch mmap ---
            const size_t bytes     = cst_buffer_size * sizeof(T);
            const size_t pagesize  = sysconf(_SC_PAGESIZE);
            const size_t allocsize = ((bytes + pagesize - 1) / pagesize) * pagesize;

            void* ptr = mmap(nullptr, allocsize, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

            if (ptr == MAP_FAILED)
            {
                perror("mmap failed");
                std::abort();
            }

            T* arr = static_cast<T*>(ptr);
            for (size_t i = 0; i < cst_buffer_size; ++i)
                arr[i] = first_value;

            mprotect(ptr, allocsize, PROT_READ);
            return arr;
        }
        else
        {
            // --- branch std::array ---
            std::array<T, cst_buffer_size> arr{};
            arr.fill(first_value);
            return arr;
        }
    }();
};

} // namespace shrink_array

#include "chunk/internal/chunk_base_inline.h"
