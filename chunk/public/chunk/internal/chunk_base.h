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
#include <vector>      // std::vector
#include <unistd.h>    // sysconf(_SC_PAGESIZE)
#include <algorithm>   // std::min, the constant scans
#include <map>         // std::map
#include <cstdint>     // uint32_t, int32_t, std::uint64_t
#include <type_traits> // std::conditional_t, std::is_arithmetic_v
#include <atomic>      // std::atomic, std::memory_order
#include <array>       // std::array
#include <cstddef>     // std::size_t
#include <cerrno>      // errno (use_mm_protect: the cst_val1 buffer's mmap)
#include <cstdio>      // stderr
#include <cstdlib>     // std::abort
#include <cstring>     // std::memset, std::strerror
#include <memory>      // std::unique_ptr, std::make_unique
#include <optional>    // std::optional (constant_value)
#include <print>       // std::println
#include <sys/mman.h>  // mmap, mprotect, PROT_READ

#include "chunk/chunk_traits.h"                         // Chunk2A fwd decl (element_at variant gate)
#include "chunk/internal/concept_array_chunk_element.h" // ArrayChunkElement, asserted on ChunkBase
#include "chunk/internal/churn_counters.h"              // churn::note_materialisation
#include "chunk/internal/chunk_size_policy.h"           // cst_buffer_size, thread_chunk_size, alloc_chunk_count
#include "chunk/internal/in_t.h"                        // in_t<T>: the scans' constants, by value or by reference
#include "chunk/internal/bit_words.h"                   // bits::scan_words / word_t: the Bit scans
#include "chunk/internal/element_access.h"              // elem_get: Bit buffers are words of bits
#include "chunk/internal/relaxed_element.h"             // element_at<true>: the marking filter's reads
#include "fast_allocators/internal/svalloc.h"           // svcalloc_align, svfree
#include "chunk/config.h"                               // build_config::use_openmp, chunk_config::stats
#include "chunk/PMP_CHUNK.h"                            // CHUNK_PMP_THREAD_RANK, CHUNK_PMP_IN_PARALLEL_REGION
#include "chunk/ring_log_debug.h"                       // EV_VALUE_ALLOC/FREE (debug ring log)
#include "chunk/internal/global_check.h"                // global_check_nan (DO_CHECK: config.h's CheckNaN<true>)
#include "chunk/internal/invariant_check.h"             // CHUNK_INVARIANT_FAILED
#include "chunk/internal/deferred_op_pool.h"            // DeferredOpPtr
#include "chunk/internal/pool_ledger.h"                 // CHUNK_LEDGER_* block-lifecycle checks (check builds)
#include "chunk/internal/chunk_stats.h"                 // increm_chunk_* / add_chunk_* (CHUNK_STATS)
#include "chunk/internal/bad_ptr_check.h"               // CHUNK_CHECK_VALUES_PTR (check builds)

#include "chunk/internal/freepath_counters.h" // [FREEPATH] counters, needed in std-alloc builds too
#include "chunk/internal/test_hooks.h"        // TEST KNOB: CHUNK_TEST_READ_DELAY_EVERY (element_at)
#include "chunk/internal/read_reclaim.h"      // ReadSection (element_at), defer_release (free_block_to_pool)

#if CHUNK_OPT_FAST_ALLOC
    #include "fast_allocators/slab_arena.h"   // the slab header slab_of() returns (owner, owner_key)
    #include "fast_allocators/tc_allocator.h" // TCTypeAllocator, TCTypeCAllocator
#else
namespace fast_alloc
{
template <typename T>
class TCTypeCAllocator; // forward declaration

template <typename T>
class TCTypeAllocator;
} // namespace fast_alloc
#endif

#include "chunk/vendor/flat_hash_map.hpp"        // ska::flat_hash_map
#include "chunk/vendor/ankerl/unordered_dense.h" // ankerl::unordered_dense::map

// The vendored unordered_dense.h keeps upstream's include guard: in a
// translation unit that includes a user's own copy first, that copy is the
// one compiled here. Any 5.x has what chunk uses; another major is refused
// here, by name, rather than deep inside a template.
static_assert(ANKERL_UNORDERED_DENSE_VERSION_MAJOR == 5,
              "chunk needs ankerl::unordered_dense 5.x: another major version of "
              "ankerl/unordered_dense.h was included before chunk's headers");

// deferred_ops_storage.h does not depend on _OPENMP and costs nothing when
// never instantiated (serial builds): always included, so ChunkDeferredOpsPolicy
// below can name it unconditionally.
#include "chunk/internal/deferred_ops_storage.h" // DeferredOpsStorage
#include "chunk/internal/immortal.h"             // Immortal / LazyImmortal: statics safe for global arrays
namespace shrink_array
{
// Forward declaration: only named here, in contexts that depend on T. The
// definition (chunk/internal/compression_manager.h) comes with the _inline tails.
template <typename T, typename ChunkType>
struct CompressionManager;

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

// STAYS a directive. Not a twin of the declaration at the top of this file:
// that one is in fast_alloc, this one is inside `namespace shrink_array`.
//
// Load-bearing: in a SERIAL build this incomplete declaration deliberately
// SHADOWS the real (OpenMP per-rank) fast_alloc::TCTypeCAllocator for the rest
// of the namespace, so the OpenMP-only allocator pool below (allocator_pool_,
// alloc_chunk_allocator_thr) parses against a type that is never completed nor
// instantiated. Removing the guard or the declaration would bind those uses to
// the real type; no `if constexpr` can "declare a different type".
#ifndef _OPENMP
template <typename T>
class TCTypeCAllocator;
#else
using fast_alloc::TCTypeCAllocator;
#endif
// The serial allocator, by its short name in this file (both builds).
using fast_alloc::TCTypeAllocator;


// ============================================================================
// DeferredReadGate: read-side visibility gate, selected by compile-time mode
// ----------------------------------------------------------------------------
// The read gate (reader_sync + ArrayChunkBase::sync_chunk) as a CRTP policy, in
// the same style as Chunk2CounterPolicy / Proxy2CounterPolicy. Not `if
// constexpr`: pending_ops_ / wait_for_availability exist in
// some builds only, and a discarded branch would still phase-1-look-up a
// missing member. Each specialization names ONLY the members of its config.
//
//   None    (serial, !use_openmp)            : nothing can be pending -> no-op
//   Precise (OpenMP + per-chunk pending_ops_) : precise pending_ops_ load
// ============================================================================
enum class DeferReadGate : std::uint8_t
{
    None,
    Precise
};

inline constexpr DeferReadGate defer_read_gate_mode =
    build_config::use_openmp ? DeferReadGate::Precise : DeferReadGate::None;

template <typename Derived, DeferReadGate Mode = defer_read_gate_mode>
class DeferredReadGate; // primary left undefined: only the 2 specializations exist

// ── Pending-visibility protocol: THE invariants, in one place ────────────────
// This policy is the single owner of every access to the per-chunk pending
// counter. Nothing outside it may touch pending_ops_ directly. The verbs:
//
//   pending_publish()   enqueue side. +1 RELEASE, called BEFORE the map insert
//                       (to_be_queued), so pending_any()>0 spans a pending op's
//                       ENTIRE lifetime: no window for a reader's stale read.
//                       The op itself is published to flushers via the Dekker
//                       write flag (begin_write/end_write), not this counter.
//   pending_retire(n)   drain side. -n RELEASE, called AFTER the ops' write
//                       bodies ran (flush_deferred / flush_chunk_all_ranks /
//                       erase_own_rank_op), so a reader that acquire-observes
//                       the decrement also observes the drained results.
//   pending_any()       gate side. ACQUIRE load > 0; pairs with the publish.
//                       False can only mean "nothing pending on this chunk".
//   read_gate_should_wait(i) / reader_sync(i)
//                       read-side gates built on the above.
//   precise             constexpr: true iff the per-chunk counter exists, for
//                       `if constexpr`-selecting counter-only code (e.g.
//                       the check-build asserts of array_chunk_base.h).
//
// Storage: the counter stays declared in ChunkBase, so the object layout (the
// cache-line placement of `values`) does not depend on this policy. The policy
// owns the ACCESS, not the storage.
// ─────────────────────────────────────────────────────────────────────────────

// Serial build: the deferred-op machinery does not exist, nothing is ever
// pending, so the gate is a pure no-op (and names no OpenMP-only member).
template <typename Derived>
class DeferredReadGate<Derived, DeferReadGate::None>
{
  public:
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
    [[nodiscard]] static bool pending_any() noexcept
    {
        return false;
    }

    /// @brief Always false: nothing can ever be pending in a serial build.
    /// @param thr Calling thread's rank; unused.
    /// @param i Element offset; unused.
    /// @return Always false.
    [[nodiscard]] bool static foreign_pending([[maybe_unused]] int thr, [[maybe_unused]] int i) noexcept
    {
        return false;
    }
};

// OpenMP + per-chunk counter: precise gate. A flat acquire load pairs with the
// release fetch_add at enqueue; >0 means this chunk has a pending op.
template <typename Derived>
class DeferredReadGate<Derived, DeferReadGate::Precise>
{
  public:
    static constexpr bool precise = true;

    /// @brief Publish a newly-enqueued deferred op: increment the per-chunk
    ///        pending count and record this thread as the owner (owner_multi
    ///        once a second rank publishes). Encoding: see the pending-word
    ///        note in the private section.
    /// @param thr Calling (publishing) thread's rank.
    void pending_publish(this Derived& self, [[maybe_unused]] int thr) noexcept
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
    /// @brief Retire `count` deferred ops that have now been applied: decrement
    ///        the per-chunk pending count (clearing the owner once it
    ///        reaches <= 0).
    /// @param count Number of operations being retired.
    void pending_retire(this Derived& self, int32_t count) noexcept
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
                                                   static_cast<const void*>(&self), cnt, ncnt, count);
                        }
                    }
                    return;
                }
                cur = static_cast<uint32_t>(expected);
            }
        }
    }
    /// @brief Whether anything is currently pending on this chunk (either
    ///        OFF or ON encoding, decoded correctly either way).
    /// @return true if the (sign-extended) pending count is > 0.
    [[nodiscard]] bool pending_any(this const Derived& self) noexcept
    {
        const auto raw = self.pending_ops_.load(std::memory_order_acquire);
        return sign_extend_count(static_cast<uint32_t>(raw)) > 0;
    }

    /// @brief Is a FOREIGN thread's op pending on this chunk right now?
    /// @param thr Calling (writing) thread's rank.
    /// @param i Element offset being written; unused (the check is per-chunk).
    /// @return true iff something is pending and it was not published by `thr`.
    [[nodiscard]] bool
    foreign_pending(this const Derived& self, [[maybe_unused]] int thr, [[maybe_unused]] int i) noexcept
    {
        {
            const auto w = static_cast<uint32_t>(self.pending_ops_.load(std::memory_order_acquire));
            return sign_extend_count(w) > 0 && (w >> count_bits) != static_cast<uint32_t>(thr) + 1u;
        }
    }
    /// @brief Read-side gate check: does a reader need to wait/drain before
    ///        reading element `i`?
    /// @details Through pending_any(), never a raw `pending_ops_.load() > 0`:
    ///          once a second rank publishes, the owner field is owner_multi
    ///          (0xFFF), bit 31 of the word is set, the raw int32 is NEGATIVE,
    ///          and a raw `> 0` gate opens while ops are pending -- the reader
    ///          then consumes the stale pre-op value.
    /// @return pending_any(). (`i` unused: the check is per-chunk, not per-offset.)
    [[nodiscard]] bool read_gate_should_wait(this const Derived& self, int /*i*/) noexcept
    {
        return self.pending_any();
    }

    /// @brief Drain this chunk across all ranks if anything is pending on it.
    /// @param i Element offset being read.
    void reader_sync(this const Derived& self, int i) noexcept
    {
        if (self.pending_any()) [[unlikely]]
        {
            self.wait_for_availability(i);
        }
    }

  private:
    // ── Pending word ────────────────────────────────────────────────────────
    // [owner:12|count:20]. owner = publishing rank + 1, owner_multi once a
    // second rank publishes while count > 0, cleared when count drops to <= 0;
    // foreign_pending() reads it. count is two's complement in 20 bits, read
    // through sign_extend_count(): an over-retire goes negative and self-heals
    // at the next publish, as a plain signed counter would. Never let it wrap
    // as an unsigned count: the gates would stay poisoned for good.
    static constexpr uint32_t count_bits  = 20;
    static constexpr uint32_t count_mask  = (uint32_t{1} << count_bits) - 1;
    static constexpr uint32_t owner_multi = 0xFFFu;

    /// @brief Sign-extend the low 20 bits of the pending word (the count is
    ///        stored as two's-complement inside those 20 bits).
    /// @param word Raw pending word.
    /// @return The count, sign-extended to a full int32_t.
    [[nodiscard]] static constexpr int32_t sign_extend_count(uint32_t word) noexcept
    {
        return static_cast<int32_t>(word << (32 - count_bits)) >> (32 - count_bits);
    }
};

// ============================================================================
// ChunkAllocPolicy: OpenMP vs serial chunk allocation, selected by
// build_config::use_openmp. A genuine two-implementation split: BOTH modes
// allocate, initialise and keep stats/ledger in sync; only OpenMP tries the
// per-thread free list first. static_cast<Derived&>(*this), not an explicit
// object parameter: allocate_chunk_internal is called as
// Base::allocate_chunk_internal(...) from sibling policies (Chunk2OmpPolicy...),
// whose *this is not a Derived and would not bind to `this Derived&`.
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
    ///       hot path (most materialisations) for nothing. v2/v2A compress to
    ///       an arbitrary constant and must initialise both branches alike: they
    ///       call the single-functor overload below.
    template <typename InitReused, typename InitFresh>
    T* allocate_chunk_internal(int thr_rank, InitReused init_reused, InitFresh init_fresh)
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
            ring_log_op(get_tid(), EV_VALUE_ALLOC, 0, static_cast<const void*>(values_loc));
        }

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
// enqueue/flush/read-gate methods -- in ONE policy, because they are tightly
// coupled (bodies out-of-line in chunk_base_inline.h). Every name they use is
// declared in every build; the true specialization is never instantiated in a
// serial build.
// Storage: `values` and `pending_ops_` stay on ChunkBase for layout stability
// (as for DeferredReadGate); the policy owns the access, through
// static_cast<Derived&>(*this), not an explicit object parameter: as at
// ChunkAllocPolicy, compare_and_swap_gen / recup_value are called as Base::...
// from sibling policies whose *this is not a Derived.
// ----------------------------------------------------------------------------
template <typename T, typename Derived, bool UseOpenMP = build_config::use_openmp>
class ChunkDeferredOpsPolicy; // primary left undefined: only the 2 specializations exist

template <typename T, typename Derived>
class ChunkDeferredOpsPolicy<T, Derived, true>
{
  public:
    using unordered_map_deferred_operations = ankerl::unordered_dense::map<int, DeferredOpPtr>;
    // per-thread map type: maps Derived* (chunk pointer) -> unique_ptr<map<num, op>>
    using per_thread_map = ska::flat_hash_map<Derived*, std::unique_ptr<unordered_map_deferred_operations>>;

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
    void recup_value(T* values_loc, int thr_rank) const
    {
        const auto& self = static_cast<const Derived&>(*this);
        // Ledger: LIVE -> PARKED. This block was just allocated by create_value
        // and lost the install CAS; parking a block that is not LIVE (e.g. one
        // another thread already owns) trips here.
        CHUNK_LEDGER_PARK(values_loc, "recup_value");
        // Back to the list it was taken from / would have been taken from: the
        // array's, on this rank.
        auto& ll_block_free_thr = self.get_manager().get_list_block_free_thr(thr_rank, thread_alloc_key());
        ll_block_free_thr.push_back(values_loc);
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
    ///            the offset in hand and the DeferredReadGate modes must agree
    ///            on one signature.
    void wait_for_availability(int num) const;

    /// @brief Current per-thread deferred-ops array, or nullptr if it has not
    ///        been allocated yet (see ensure_deferred_array_initialized()).
    /// @return Pointer to the per-thread map array, or nullptr.
    static per_thread_map* get_deferred_ops() noexcept
    {
        return DeferredOpsStorage<per_thread_map>::instance().get();
    }

  private:
    // ── Deferred-op enqueue / flush / read gate ────────────────────────────
    /// @brief Allocate the global deferred-op array on first use.
    void ensure_deferred_array_initialized() const
    {
        if (get_deferred_ops() != nullptr) [[likely]]
        {
            return;
        }
        DeferredOpsStorage<per_thread_map>::instance().allocate();
    }
};

// Serial: none of the above exists -- no deferred writes, no CAS
// (nothing concurrent to protect against), no thread-local free list
// (nothing to contend over).
template <typename T, typename Derived>
class ChunkDeferredOpsPolicy<T, Derived, false>
{
};

// ============================================================================
// ChunkDynamicAllocPolicy: the CHUNK_DYNAMIC-only per-size allocator pools
// (allocator_pool_/allocator_pool_serial_) and their registration methods.
// Static members, so moving them off ChunkBase has no object-layout cost.
// Selected by build_config::chunk_dynamic. get_chunk_allocator() (on
// ChunkBase) reaches them through route_omp_dynamic_alloc() /
// route_serial_dynamic_alloc().
// ----------------------------------------------------------------------------
template <typename T, typename Derived, bool IsDynamic = build_config::chunk_dynamic>
class ChunkDynamicAllocPolicy; // primary left undefined: only the 2 specializations exist

template <typename T, typename Derived>
class ChunkDynamicAllocPolicy<T, Derived, true>
{
  public:
    /// @brief Register (create + init) the per-thread allocator array for
    ///        `size`, once.
    /// @details Sequential context only.
    /// @param size Chunk element count this allocator serves.
    /// @param max_thr Number of per-thread allocator slots to create.
    /// @param block_size Per-thread slab byte budget (mirrored from
    ///                   init_allocators()).
    /// @param key Affinity key (0 = shared rank pool). See pool_key().
    static void register_chunk_size(unsigned int size, int max_thr, int block_size, unsigned int key = 0)
    {
        if (allocator_pool_->contains(pool_key(key, size)))
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
        allocator_pool_->emplace(pool_key(key, size), std::move(arr));
    }

    /// @brief Register (create + init) the single allocator for `size`, once.
    /// @details Mirrors the serial init_allocator() convention:
    ///          init(storage_units(size), block_size, -1), where size is the
    ///          chunk's element count (alloc_chunk_count() in a dynamic build)
    ///          and -1 = never munmap a value slab (as init_allocator() /
    ///          init_allocators()).
    /// @param size Chunk element count this allocator serves.
    /// @param block_size Slab byte budget.
    static void register_chunk_size_serial(unsigned int size, int block_size)
    {
        if (allocator_pool_serial_->contains(size))
        {
            return; // already registered
        }
        auto allocator = std::make_unique<TCTypeAllocator<T>>();
        // Value arena, explicitly: TCAllocator defaults to the Op arena, and a
        // value pool sharing it with the deferred-op pool lets a freed slab of
        // one be reobtained by the other, corrupting both pools' slab lists.
        allocator->set_arena(fast_alloc::ArenaKind::Value);
        allocator->init(static_cast<size_t>(Derived::storage_units(size)), static_cast<uintptr_t>(block_size), -1);
        allocator_pool_serial_->emplace(size, std::move(allocator));
    }

    // ── Routing ──
    // Lives here, not in get_chunk_allocator()'s body: allocator_pool_ exists
    // only in THIS specialization, and a discarded `if constexpr` branch naming
    // it there would still be phase-1-looked-up (same trap as DeferredReadGate).

    /// @brief Route to the allocator for this thread's active chunk size. The
    ///        size must have been registered (set_chunk_size) beforehand.
    /// @param thr_rank Calling thread's rank.
    /// @param key      Affinity key of the array (0: the rank's shared pool).
    /// @return Reference to the per-thread allocator for the active size, or
    ///         to the legacy single allocator (Derived::allocator_data) if
    ///         the active size was never registered.
    static auto& route_omp_dynamic_alloc(int thr_rank, unsigned int key = 0) noexcept
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

    /// @brief Serial mirror of route_omp_dynamic_alloc(); no per-thread
    ///        dimension. Registered by register_chunk_size_serial() via
    ///        set_chunk_size before any alloc.
    /// @return Reference to the allocator for the active size, or to the
    ///         legacy single allocator (Derived::allocator_data) if
    ///         unregistered.
    static auto& route_serial_dynamic_alloc() noexcept
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

  private:
    // ---------------------------------------------------------------------
    // Multi-size support (dynamic builds): per-thread allocator arrays, keyed by
    // chunk size. register_chunk_size() (from the sequential
    // ArrayChunkBase::set_chunk_size, before any parallel alloc) is the ONLY
    // writer; get_chunk_allocator() only reads, so parallel regions never look
    // up a map being mutated. std::map: references stay valid across inserts,
    // as a new size may be registered while old entries are in use.
    // ---------------------------------------------------------------------
    /// @brief Registry key: (affinity key, chunk size). Key 0 is the rank's
    ///        shared pool; a non-zero key is one array's own allocator set, so that its
    ///        blocks sit in slabs holding nothing else and a re-chunk or a
    ///        destruction of that array empties whole slabs.
    static constexpr std::uint64_t pool_key(unsigned int key, unsigned int size) noexcept
    {
        return (static_cast<std::uint64_t>(key) << 32U) | size;
    }

    static inline constinit internal::LazyImmortal<std::map<std::uint64_t, std::vector<TCTypeCAllocator<T>>>>
        allocator_pool_;

    // Serial counterpart of allocator_pool_: one allocator per chunk size, no
    // per-thread dimension. Held by unique_ptr: TCTypeAllocator is polymorphic
    // and non-movable. Written by register_chunk_size_serial() (sequential
    // set_chunk_size path), read by route_serial_dynamic_alloc(). Required for a
    // re-chunk in serial fast-alloc builds: old and new sizes coexist during
    // rechunk_refill, so re-initialising the single allocator_data would not do.
    static inline constinit internal::LazyImmortal<std::map<unsigned int, std::unique_ptr<TCTypeAllocator<T>>>>
        allocator_pool_serial_;
};

// Static (non-dynamic) builds: a single fixed geometry, so there is no
// per-size pool to keep -- allocator_data (on ChunkBase) is the only
// allocator that ever exists. Both routing methods just return it directly.
template <typename T, typename Derived>
class ChunkDynamicAllocPolicy<T, Derived, false>
{
  public:
    /// @brief Static builds have no per-size pool: always route to the
    ///        single legacy allocator.
    /// @param thr_rank Calling thread's rank.
    /// @param key      Unused: static builds have a single pool per rank.
    /// @return Reference to (*Derived::allocator_data)[thr_rank].
    static auto& route_omp_dynamic_alloc(int thr_rank, [[maybe_unused]] unsigned int key = 0) noexcept
    {
        return (*Derived::allocator_data)[thr_rank];
    }

    /// @brief Serial mirror of the above; no per-thread dimension.
    /// @return Reference to Derived::allocator_data.
    static auto& route_serial_dynamic_alloc() noexcept
    {
        return *Derived::allocator_data;
    }
};

// ============================================================================
// ChunkBase
// ============================================================================

/// @brief CRTP base common to all chunk types (Chunk, Chunk2, Chunk2A).
/// @tparam T       Element type.
/// @tparam Derived The chunk type.
template <typename T, typename Derived>
class ChunkBase : public DeferredReadGate<Derived>,
                  public ChunkAllocPolicy<T, Derived>,
                  public ChunkDeferredOpsPolicy<T, Derived>,
                  public ChunkDynamicAllocPolicy<T, Derived>
{
    static_assert(ArrayChunkElement<T>,
                  "shrink_array element type: trivially copyable, ==, default constructible "
                  "(see concept_array_chunk_element.h)");

  public:
    // Atomic pointer type conditional on OpenMP availability
    using atomic_ptr_t = std::conditional_t<build_config::use_openmp, std::atomic<T*>, T*>;

    using ProxyType = ProxyTypeTrait<Derived>::Type;

    // std-alloc placeholder for `allocator_data`: no-op stubs of the fast
    // allocator's interface, so manager code shared by both modes compiles in
    // std builds. Never does real work: std-alloc allocates through svalloc, and
    // every call site is behind `if constexpr (build_config::use_fast_alloc)`.
    struct NullValueAllocator
    {
        /// @brief No pool to set up.
        static void init(int /*element_size*/, int /*block_size*/, int /*munmap_flag*/) noexcept {}
        /// @brief No pool to tear down.
        static void destroy() noexcept {}
        /// @brief Never allocates: std-alloc buffers come from svalloc.
        [[nodiscard]] static T* alloc() noexcept
        {
            return nullptr;
        }
        /// @brief Nothing to return to a pool.
        static void free_elt_pow2(const T* /*ptr*/) noexcept {}
        /// @brief No-op counterpart to TCTypeAllocator::set_arena, so call
        ///        sites that set the arena unconditionally (before init())
        ///        compile in std-alloc builds too.
        static void set_arena(fast_alloc::ArenaKind /*k*/) noexcept {}
    };

    using AllocatorType = std::conditional_t<
        build_config::use_fast_alloc,
        std::conditional_t<build_config::use_openmp, std::vector<TCTypeCAllocator<T>>, TCTypeAllocator<T>>,
        NullValueAllocator>;

    /// @brief Outcome of scan_constants(): which hypothesis survived the walk.
    struct ConstantScan
    {
        bool first  = false; ///< every element equals the first constant
        bool second = false; ///< every element equals the second constant -- never together with `first`
    };

    using cst_val1_type = std::conditional_t<build_config::use_mm_protect, T*, std::array<T, cst_buffer_size>>;

    ChunkBase()                            = default;
    ~ChunkBase()                           = default;
    ChunkBase(const ChunkBase&)            = default;
    ChunkBase& operator=(const ChunkBase&) = default;
    ChunkBase(ChunkBase&&)                 = default;
    ChunkBase& operator=(ChunkBase&&)      = default;

    // ========================================================================
    // Atomic / Non-atomic pointer management
    // ========================================================================

    /// @brief Load the values pointer with appropriate memory ordering
    /// @return Pointer to the chunk data
    [[nodiscard]] T* values_load() const noexcept
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
    [[nodiscard]] T read_value_scan(this const Derived& self, int i) noexcept
    {
        if constexpr (chunk_config::stats)
        {
            increm_chunk_scan_read<Derived>();
        }
        return self.element_at(i);
    }

    /// @brief read_value_scan() of an element another thread may be writing:
    ///        a relaxed atomic load (relaxed_element.h). The marking filter's.
    /// @pre internal::relaxed_element_access_v<T>.
    [[nodiscard]] T read_value_scan_relaxed(this const Derived& self, int i) noexcept
    {
        if constexpr (chunk_config::stats)
        {
            increm_chunk_scan_read<Derived>();
        }
        return self.template element_at<true>(i);
    }

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
     *          candidates, the end-of-run audits and the size predictor all go
     *          through it.
     *
     * @param first_v  first hypothesis, usually first_value
     * @param second_v second hypothesis
     * @param count        elements to scan -- the chunk's OWN geometry, never the
     *                 ambient size (a candidate is scanned by whichever thread
     *                 drains it, whose ambient size may differ after a re-chunk)
     * @pre The chunk is materialised: `values` points at a real buffer.
     */
    [[nodiscard]] ConstantScan
    scan_constants(this const Derived& self, in_t<T> first_v, in_t<T> second_v, unsigned int count) noexcept
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
            const bits::Uniform u        = bits::scan_words(internal::as_words(vals), count);
            const auto          all_of_v = [u](bool v) noexcept
            { return u == (v ? bits::Uniform::all_true : bits::Uniform::all_false); };
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
        // Coinciding constants (frequent in 2A: second_v is values[0], and a
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

    /// @brief Is the materialised buffer all @p v over its first @p count elements?
    /// @pre The chunk is materialised.
    [[nodiscard]] bool is_constant_to(this const Derived& self, in_t<T> v, unsigned int count) noexcept
    {
        if constexpr (is_packed)
        {
            if constexpr (chunk_config::stats)
            {
                add_chunk_scan_read<Derived>(static_cast<long>(count));
            }
            return bits::scan_words(internal::as_words(self.values_load()), count) ==
                   (static_cast<bool>(v) ? bits::Uniform::all_true : bits::Uniform::all_false);
        }
        // Same shape as scan_constants(): branch-free blocks, exit between them.
        const T*     vals = self.values_load();
        unsigned int i    = 0;
        while (i < count)
        {
            const unsigned int block_end = std::min(count, i + scan_block);
            unsigned int       bad       = 0;
            for (unsigned int j = i; j < block_end; ++j)
            {
                // raw-read: compressor scan of a chunk this pass claimed (steal mark), or a sequential caller
                bad |= static_cast<unsigned int>(vals[j] != v);
            }
            i = block_end;
            if (bad != 0)
            {
                if constexpr (chunk_config::stats)
                {
                    add_chunk_scan_read<Derived>(static_cast<long>(i));
                }
                return false;
            }
        }
        if constexpr (chunk_config::stats)
        {
            add_chunk_scan_read<Derived>(static_cast<long>(count));
        }
        return true;
    }

    /// @brief The one value the materialised buffer holds over @p count elements, if any.
    /// @return that value, or nullopt when at least two elements differ
    /// @pre The chunk is materialised and @p count > 0.
    [[nodiscard]] std::optional<T> constant_value(this const Derived& self, unsigned int count) noexcept
    {
        if (const in_t<T> v0 = self.element_at(0); self.is_constant_to(v0, count))
        {
            return v0;
        }
        return std::nullopt;
    }

    /// @brief Read a value at given index with statistics tracking
    /// @param i Index in the chunk
    /// @return Value at index i
    [[nodiscard]] T read_value(this const Derived& self, int i) noexcept
    {
        if constexpr (chunk_config::stats)
        {
            // Split the read by what served it: a materialised chunk vs a
            // compressed one (values points at the constant marker). read_cst /
            // (read + read_cst) is the compression hit rate on reads.
            // is_cst_val1(), not written(): Chunk2::written() (no arg) aborts on
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
        return self.element_at(i);
    }

    // reader_sync(i) -- the read-side visibility gate of the pure-read proxy
    // entry points -- is inherited from DeferredReadGate<Derived> above
    // (None/Precise per build).

    // ========================================================================
    // Read-only operator[]
    // ========================================================================

    /// @brief Read-only access to chunk elements with optional NaN checking
    /// @param i Index in the chunk
    /// @return Value at index i
    T operator[](this const Derived& self, int i) noexcept
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
        const T v = self.element_at(i);
        chunk_config::CheckNaN<chunk_config::do_check>::maybe(v, static_cast<long>(i));
        return v;
    }

    /**
     * @brief Copy elements [@p first, @p first + @p count) into @p out under ONE
     *        read section.
     *
     * @details The reads of @p count element_at() calls for the price of one
     *          read_reclaim::ReadSection: `values` is loaded once, the elements
     *          are copied, the section closes. A compression pass running in
     *          the region waits for the copy (bounded, no caller code inside)
     *          before releasing the buffer, exactly as it waits for one
     *          element_at(). The block is what the chunk held at that load.
     *
     *          Raw like element_at(): no stats, no NaN check. The bulk reads
     *          (ArrayChunkBase::for_each_value / read_range) use it only in
     *          builds without those.
     * @param first first offset in the chunk
     * @param count number of elements, first + count <= chunk size
     * @param out   destination, @p count elements
     */
    void read_block(this const Derived& self, int first, int count, T* out) noexcept
    {
        const read_reclaim::ReadSection section;
        const T* const                  vals = self.values_load();
        test_hooks::maybe_read_delay(); // TEST KNOB, as in element_at_unguarded
        if constexpr (std::is_same_v<Derived, Chunk2A<T>>)
        {
            // Same single load as element_at_unguarded: a chunk compressed to a
            // second constant points at an identifier, not at values.
            if (self.is_cst_val2_ptr_of(vals)) [[unlikely]]
            {
                const T v = self.second_value_from_ptr(vals);
                for (int j = 0; j < count; ++j)
                {
                    out[j] = v;
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

    /**
     * @brief read_block(), unless the chunk is compressed: then its constant.
     *
     * @details The same single load of `values` under one read section. A
     *          chunk compressed to the first constant -- or, Chunk2A, to its
     *          second -- sets @p cst and leaves @p out untouched, so the caller
     *          takes the rest of the chunk at once
     *          (ArrayChunkBase::for_each_span). A Chunk2 on its second constant
     *          points at a filled buffer and is read as a block.
     * @return true when @p cst was set, false when @p out was filled
     */
    bool read_block_or_constant(this const Derived& self, int first, int count, T* out, T& cst) noexcept
    {
        const read_reclaim::ReadSection section;
        const T* const                  vals = self.values_load();
        test_hooks::maybe_read_delay(); // TEST KNOB, as in element_at_unguarded
        if (self.is_cst_val1(vals))
        {
            // raw-read: read_block_or_constant -- the shared first-constant buffer
            cst = load_at<false>(vals, 0);
            return true;
        }
        if constexpr (std::is_same_v<Derived, Chunk2A<T>>)
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

    /// @brief Proxy access for read-write operations
    /// @param i Index in the chunk
    /// @return Proxy object for lazy write
    ProxyType operator[](this Derived& self, int i) noexcept
    {
        return ProxyType(self.values, i, self);
    }

    // ========================================================================
    // operator() - Direct access without checks
    // ========================================================================

    /// @brief Direct read-write access to chunk elements without safety checks
    /// @param i Index in the chunk
    /// @return Reference to value at index i
    T& operator()(this const Derived& self, int i) noexcept
    {
        if constexpr (chunk_config::stats)
        {
            increm_chunk_read<Derived>();
        }
        // Contract: a mutable reference that bypasses the write proxy, valid only
        // on a materialised chunk. On a compressed one it points into a shared
        // constant (Chunk/Chunk2) or an unbacked identifier slot (Chunk2A): a
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
        return self.values_load()[i];
    }

    // ========================================================================
    // Block initialization
    // ========================================================================

    /// @brief Initialize the current chunk's memory block
    void init_block() noexcept
    {
        T* val = values_load();
        init_block(val);
    }

    // ========================================================================
    // Chunk allocation with free block cache management
    // ========================================================================

    /// @brief Return a values buffer to the underlying allocator -- a real
    ///        free, not a park onto the thread-local reuse list (see
    ///        free_block_to_pool for that).
    /// @param ptr Buffer to free.
    /// @param thr_rank Calling thread's rank (selects the allocator pool in
    ///                 fast-alloc/OMP builds; ignored otherwise).
    void free_values(this Derived& self, T* ptr, [[maybe_unused]] int thr_rank = 0)
    {
        // Debug ring log: value-buffer frees.
        if constexpr (chunk_config::ring_log_debug || chunk_config::ring_log_op_only)
        {
            ring_log_op(get_tid(), EV_VALUE_FREE, 0, static_cast<const void*>(ptr));
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

    /// @brief Free this chunk's current values buffer.
    /// @param thr_rank Calling thread's rank; forwarded to free_values(T*, int).
    void free_values(this Derived& self, [[maybe_unused]] int thr_rank = 0)
    {
        T* ptr = self.values_load();
        self.free_values(ptr, thr_rank);
    }

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
            // Dynamic-vs-static routing lives on ChunkDynamicAllocPolicy.
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
            CHUNK_INVARIANT_FAILED("get_chunk_allocator() in a build without the fast allocator");
        }
    }

    /// @brief The (read-only) buffer every compressed-to-first-value chunk
    ///        points at.
    /// @return Pointer to the shared first-constant buffer.
    static T* get_cst_val1() noexcept
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
        {
            return bits::bytes_for(n_elems) / sizeof(T); // whole 64-bit words
        }
        else
        {
            return n_elems;
        }
    }

    /// @brief init_block() over the active geometry's storage.
    static void init_block(T* values_loc) noexcept
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

    static inline constinit internal::LazyImmortal<AllocatorType> allocator_data;

    /// First constant value for compression: the value-initialised T. Spelled
    /// T{} and not 0 everywhere in the library, so that an aggregate (a struct
    /// of doubles, a std::array) is a first-class element type -- `T{0}` would
    /// initialise its first member only, `T(0)` not compile.
    static inline T first_value{};

    // MM_PROTECT: the read-only mapping that get_cst_val1() returns, mapped once.
    static T* map_cst_val1()
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

    // alignas: a Bit chunk compressed to the first constant points here, and
    // its reads load 64-bit words (element_access.h). constinit, every element
    // T{} (= first_value): a global array reaches it before any dynamic
    // initialisation. Unused under MM_PROTECT (null; see get_cst_val1()).
    alignas(bits::word_t) static inline constinit cst_val1_type m_ptr_cst_val1{};

  protected:
    /**
     * @brief The rank sweep of the compression passes (compress_chunk_to_compress
     *        of every chunk type): each rank drains ITS OWN candidate list.
     *
     * @details Inside a parallel region: @p in_region(rank) for the calling rank
     *          only -- opening the region below from inside one would make every
     *          thread sweep every rank's list at once. Outside: one region where
     *          each rank runs @p sweep(rank), then, when @p mop_up, @p sweep for
     *          the ranks [team size, CHUNK_PMP_RANK_BOUND) serially -- the lists
     *          are sized by the thread capacity, and a team smaller than an
     *          earlier one leaves the upper ranks' lists (and their candidate
     *          flags) behind otherwise; empty lists are cheap. The OMPT callbacks
     *          are disarmed for the duration, so nothing re-enters the
     *          compression machinery underneath.
     * @param in_region the calling rank's pass inside a region (int rank)
     * @param sweep     one rank's pass outside a region (int rank)
     * @param mop_up    sweep the ranks beyond the team too
     */
    template <typename InRegion, typename Sweep>
    static void sweep_every_rank(InRegion&& in_region, Sweep&& sweep, bool mop_up = true)
    {
        chunk_config::OMPTcc<chunk_config::has_ompt>::update(false);

        if (CHUNK_PMP_IN_PARALLEL_REGION)
        {
            in_region(CHUNK_PMP_THREAD_RANK);
        }
        else
        {
            int team = 0;
#pragma omp parallel
            {
                const int thr = CHUNK_PMP_THREAD_RANK;
                if (thr == 0) // one writer; `master` is deprecated since OpenMP 5.1
                {
                    team = CHUNK_PMP_NUM_THREADS;
                }
                sweep(thr);
            }
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

    // Type traits
    // bool and Bit (bit.h) share chunk2's bool path: one second value, true.
    static constexpr bool is_bool     = is_bool_like_v<T>;
    static constexpr bool is_not_bool = !is_bool;

    constexpr static T zero{};

    /// @brief Debug-only assertion: abort if offset j is not currently
    ///        written (a clear on an unwritten offset is a bug). Compiled
    ///        out entirely when build_config::any_check is off.
    /// @param j Element offset expected to be written.
    void debug_check_written(this const Derived& self, int j) noexcept
    {
        if constexpr (build_config::any_check)
        {
            if (!self.is_written_at(j))
            {
                CHUNK_INVARIANT_FAILED("trying to clear a chunk that is not written (offset {})", j);
            }
        }
    }

    /// @brief Debug-only assertion: abort if this chunk is not currently
    ///        written. Compiled out entirely when build_config::any_check is
    ///        off.
    void debug_check_written(this const Derived& self) noexcept
    {
        if constexpr (build_config::any_check)
        {
            if (!self.written())
            {
                CHUNK_INVARIANT_FAILED("trying to clear a chunk that is not written");
            }
        }
    }

    /// @brief Check if chunk does not point to the first constant value
    /// @return true if chunk is not compressed to first constant value
    [[nodiscard]] bool is_not_cst_val1() const noexcept
    {
        return !is_cst_val1();
    }

    // ========================================================================
    // Chunk deallocation
    // ========================================================================

    /// @brief Return a block to the free-pool, preferring the local thread's
    ///        free list over a real free -- and never pushing a foreign
    ///        block onto ANOTHER thread's free-list (see the out-of-line
    ///        definition for the full cross-thread-migration rationale).
    /// @param thr_rank Calling thread's rank.
    /// @param val Block to return.
    void free_block_to_pool(this Derived& self, [[maybe_unused]] int thr_rank, T* val);

    /// @brief free_block_to_pool() for a buffer a scoped pass collected
    ///        (read_reclaim::PassScope): released after the grace period.
    static void release_deferred_block(void* self, int thr_rank, void* val) noexcept
    {
        static_cast<Derived*>(self)->free_block_to_pool(thr_rank, static_cast<T*>(val));
    }

    // ========================================================================
    // Member variables
    // ========================================================================

    /// Pointer to chunk data (atomic if OpenMP enabled)
    alignas(std::atomic<T*>) atomic_ptr_t values;

  public:
#if defined(_OPENMP)
    /// Per-chunk count of deferred ops currently pending (enqueued but not yet
    /// drained), plus its owner tag: see DeferredReadGate<Derived, Precise>.
    /// Present only where the deferred-op machinery exists: OpenMP builds.
    /// Protocol: see DeferredReadGate (publish before the map insert, retire
    /// after the drain, acquire-loaded by the read gates).
    ///
    /// A `#if`, not `if constexpr`: gcc still phase-1-looks-up a member named in
    /// a discarded branch. Its readers live in DeferredReadGate<Precise>, which
    /// is selected under the same condition (use_openmp).
    std::atomic<int32_t> pending_ops_{0};

  private:
    /// @brief "A compressor is about to scan and may steal this chunk's
    ///        buffer": the per-chunk half of the write/compress handshake.
    ///
    /// Set (release) by a compressor on every chunk of its candidate list
    /// BEFORE its start(), and cleared (release) per chunk as the pass goes:
    /// right after the scan for a chunk that stays materialised, right after
    /// the CAS-steal for one that is compressed. A writer reads it (acquire,
    /// after its own announce_write) on the line it loads `values` from
    /// anyway: it waits, then defers, ONLY when its chunk is marked. Every
    /// other writer goes on writing while the compressor scans, instead of the
    /// whole team paying a global exclusion window per pass.
    ///
    /// Lives in the 4 bytes of padding after pending_ops_ (8 + 4 + 4: an OpenMP
    /// chunk stays at 16 bytes). Same \#if as pending_ops_, for the same gcc
    /// phase-1 lookup reason (chunk_config::use_chunk_steal_mark).
    std::atomic<uint32_t> steal_mark_{0};

  public:
    /// @brief Is a compressor's pass in flight on this chunk?
    ///
    /// @details ACQUIRE, not relaxed. start()'s heavy barrier publishes the
    ///          WRITERS' announces to the compressor; it does not push the
    ///          compressor's own mark store into a writer already in flight.
    ///          The mark travels compressor -> writer and needs its own
    ///          release/acquire pair, or a writer can read it clear after it
    ///          was set and write while the scan reads.
    ///
    ///          Read once per write: an ldar in place of an ldr on AArch64.
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
    // No mark without OpenMP (no concurrent compressor): the write gate is
    // the global one.
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

  private:
    /// The buffer is words of bits (Bit), not an array of T.
    static constexpr bool is_packed = is_packed_bit_v<T>;

    static constexpr bool use_mm_protect = build_config::use_mm_protect;

    /// @brief Elements per branch-free block of the constant scans.
    ///        A quarter of a 256-element chunk: four early-exit points, and a
    ///        block long enough for the vector loop to matter.
    static constexpr unsigned int scan_block = 64;

    /// @brief Whether this chunk is currently materialised (not compressed
    ///        to a constant), at the granularity Derived supports.
    /// @param j Element offset; used only when Derived distinguishes
    ///          writtenness per-offset (e.g. Chunk2/Chunk2A's second-value
    ///          slot); ignored otherwise (falls back to self.written()).
    /// @return true if the chunk (or offset j) holds real, materialised data.
    [[nodiscard]] bool is_written_at(this const Derived& self, int j) noexcept
    {
        if constexpr (requires { self.written(j); })
        {
            return self.written(j);
        }
        else
        {
            return self.written();
        }
    }

    /**
     * @brief The element at @p i, whatever the chunk's state.
     *
     * @details The single point every element READ goes through. A Chunk2A
     *          chunk compressed to a second constant points at an identifier
     *          slot that holds nothing (tab_cst_val2_holder.h), so its value
     *          comes from the registry: one pointer range test on the read
     *          path, in this one place rather than seven.
     *
     * @note NOT used by operator()(int): it hands out a mutable reference
     *       into the buffer, which a registry lookup cannot stand in for.
     *
     * @tparam Relaxed read the element through a relaxed atomic_ref: for an
     *         element another thread may be writing (relaxed_element.h).
     */
    template <bool Relaxed = false>
    [[nodiscard]] T element_at(this const Derived& self, int i) noexcept
    {
        // A pure read is bracketed by a ReadSection (read_reclaim.h): a
        // compression pass running in the same region waits for it before it
        // releases the buffer this read may have loaded. The Relaxed reads are
        // the compressor's own scans and the marking filter: not bracketed.
        if constexpr (Relaxed)
        {
            return self.template element_at_unguarded<true>(i);
        }
        else
        {
            const read_reclaim::ReadSection section;
            return self.template element_at_unguarded<false>(i);
        }
    }

    /// @brief element_at() without the read section: the load of `values` and
    ///        the element, nothing else.
    template <bool Relaxed>
    [[nodiscard]] T element_at_unguarded(this const Derived& self, int i) noexcept
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
        // Gated on std::is_same_v, NOT on a `requires` probe for the hooks: a
        // probe fails OPEN (where false, the read returns 0 from the unwritten
        // region -- a wrong number, not a build error), and nvhpc evaluates it
        // differently on the still-incomplete CRTP Derived. A type comparison
        // against the forward-declared Chunk2A (chunk_traits.h) cannot differ.
        if constexpr (std::is_same_v<Derived, Chunk2A<T>>)
        {
            static_assert(
                requires(const Derived& obj, const T* ptr) {
                    obj.is_cst_val2_ptr_of(ptr);
                    obj.second_value_from_ptr(ptr);
                }, "Chunk2A must provide is_cst_val2_ptr_of() and second_value_from_ptr(): "
                   "its second values are identifiers, not filled arrays");
            const T* const vals = self.values_load();
            if constexpr (!Relaxed)
            {
                test_hooks::maybe_read_delay(); // TEST KNOB, compiled out by default
            }
            if (self.is_cst_val2_ptr_of(vals)) [[unlikely]]
            {
                return self.second_value_from_ptr(vals);
            }
            // raw-read: element_at_unguarded -- under element_at's ReadSection, or a scan of a claimed chunk
            return load_at<Relaxed>(vals, i);
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
        return load_at<Relaxed>(vals, i);
    }

    /// @brief Return @p ptr to the allocator instance that carved it, found
    ///        through its slab (TCAllocator::owner / owner_rank), never
    ///        through the caller's rank.
    /// @details A slab chain may only be mutated by the rank that owns it while
    ///          a region is open. So the block is freed directly when it is
    ///          ours, unranked (serial pools), or when no region is open; a
    ///          foreign block inside a region is parked on OUR list under ITS
    ///          affinity key, where this rank may reuse it for that array and
    ///          the next drain routes it home. Never free through the caller's
    ///          instance (get_chunk_allocator(thr_rank)): for a foreign block
    ///          that mutates another rank's chains from the wrong thread.
    /// @pre fast-alloc build.
    void release_to_owner(this Derived& self, int thr_rank, T* ptr)
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
    template <bool Relaxed>
    [[nodiscard]] static T load_at(const T* vals, int i) noexcept
    {
        if constexpr (is_packed)
        {
            // raw-read: load_at, called by element_at_unguarded only
            return elem_get(vals, static_cast<std::size_t>(i)); // relaxed word load under OpenMP
        }
        else if constexpr (Relaxed)
        {
            return internal::load_element_relaxed(vals + i);
        }
        else
        {
            // raw-read: load_at, called by element_at_unguarded only
            return vals[i];
        }
    }
};

} // namespace shrink_array

#include "chunk/internal/chunk_base_inline.h" // the template bodies
