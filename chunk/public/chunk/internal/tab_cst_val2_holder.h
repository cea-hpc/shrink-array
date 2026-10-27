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
 * @file tab_cst_val2_holder.h
 * @brief The registry of second-value slots behind "second constant value" compression.
 *
 * A zero_any chunk compressed to a constant other than zero does not own a buffer: it
 * points at an identifier slot in a per-type region, one slot per DISTINCT
 * second value for the whole process. Nothing is stored at the slot -- the
 * value comes from GlobalVecHolder::value_by_id[] -- so a thousand chunks
 * holding the same constant cost one slot and one table entry between them.
 *
 * Three pieces:
 *  - Segment is a block of 64 slots. The registry is a chain of them rather
 *    than a growable array, so registering a new value never invalidates a
 *    pointer another thread is holding.
 *  - GlobalVecHolder owns that chain, one instance per element type, plus the
 *    hash index that makes a lookup O(1) and the lock that serialises appends.
 *  - TabCstVal2Holder is the surface the chunks talk to: stateless, it owns
 *    the allocation strategy.
 *
 * @par Two paths, and only one of them takes a lock
 * Registering a value that is ALREADY known -- every type-2 chunk after the
 * first one per value -- is a lock-free probe of GlobalVecHolder::index_tab.
 * Only a genuinely new value, or an index miss, takes
 * GlobalVecHolder::insert_mutex, under which the whole find-or-append is
 * plain sequential code.
 *
 * @warning One ordering is load-bearing and not checked at run time:
 *          value_by_id[] must be written before the slot pointer is published
 *          (index_publish's release store). A chunk adopts a slot by storing it
 *          as its own values pointer, nothing else.
 *
 * @warning Tearing the registry down requires that NO thread is scanning it.
 *          @see GlobalVecHolder::clear_if_last for the contract.
 *
 * @par Reclaiming
 * A value stays registered until the last array of the type is destroyed,
 * unless zero_any::Array::reclaim_second_values() runs at a sequential point:
 * it marks the identifiers still referenced and
 * GlobalVecHolder::reclaim_unmarked() frees the others. alloc_chunk() hands
 * a freed identifier out again before carving a new one, so a workload
 * whose constants come and go stays within the identifiers it has carved.
 */

#include <sys/mman.h>  // mmap, mprotect, madvise
#include <cerrno>      // errno (the region's mmap)
#include <cstring>     // std::memcpy, std::strerror
#include <cstddef>     // std::byte (the region)
#include <cstdint>     // std::uint32_t, uint64_t
#include <atomic>      // std::atomic, std::memory_order
#include <array>       // std::array
#include <memory>      // std::unique_ptr, std::make_unique
#include <mutex>       // std::mutex, std::scoped_lock
#include <type_traits> // std::conditional_t
#include <vector>      // index_generations
// <iostream> directly: param_chunk.h would be a circular include (it reaches this header).
#include <iostream> // std::cout

#include "fast_allocators/internal/raw_memory.h" // address_bits
#include "chunk/internal/chunk_size_policy.h"    // cst_buffer_size
#include "chunk/internal/env_parse.h"            // env_number (CHUNK_CST_VAL2_ID_LIMIT)
#include "chunk/internal/in_t.h"                 // InT<T>: the value parameters
#include "chunk/config.h"                        // build_config::use_mm_protect
#include "chunk/internal/invariant_check.h"      // CHUNK_INVARIANT_FAILED
#include "chunk/internal/immortal.h"             // Immortal / LazyImmortal: statics safe for global arrays
#include "chunk/log.h"                           // CHUNK_LOG (the identifier and index warnings)
#if CHUNK_OPT_FAST_ALLOC
    #include "fast_allocators/internal/alloc_config.h" // alloc_config::thp_advise (CHUNK_THP)
#endif

namespace shrink_array
{

/// @brief CHUNK_THP for the cst2 region: alloc_config::thp_advise() in a
///        fast-alloc build, always false in a std-alloc build.
///
/// alloc_config.h belongs to fast_allocators, which a std-alloc build neither
/// builds nor puts on the include path. The name cannot be written unguarded
/// in region_init() either: it is not dependent, so it is looked up when the
/// template is defined, `if constexpr` or not (see build_config's
/// reclaim_empty_slabs mirror in config.h).
inline bool cst2_thp_advise() noexcept
{
#if CHUNK_OPT_FAST_ALLOC
    return fast_alloc::alloc_config::thp_advise();
#else
    return false;
#endif
}

/**
 * @brief One block of the registry chain: 64 slots and a link to the next.
 *
 * @details Nothing here is atomic, deliberately. Segments are mutated and
 *          scanned only under GlobalVecHolder::insert_mutex, or on the
 *          single-threaded clear paths; the lock-free readers reach values
 *          through GlobalVecHolder::index_tab and never touch a Segment at all.
 *
 * @note The chain is append-only between clears and reclaims
 *       (GlobalVecHolder::reclaim_unmarked() rebuilds it packed), so
 *       occupancy is contiguous:
 *       `slots[0 .. used)` are set and `slots[used .. slots_per_segment)` are null. Several
 *       loops here rely on that rather than null-checking every slot.
 */
template <typename T>
struct Segment
{
    static constexpr int slots_per_segment = 64; ///< Slots per segment.

    std::array<T*, slots_per_segment> slots{};        ///< The slots, one per registered value.
    int                               used = 0;       ///< How many leading slots are occupied.
    Segment*                          next = nullptr; ///< Next block, or null at the tail.
};

template <typename T>
class GlobalVecHolder;

template <typename T>
class TabCstVal2Holder;

/**
 * @brief How a chunk's second value is stored in the per-chunk table.
 *
 * @details For an element type wider than 4 bytes the slot holds the
 *          IDENTIFIER registered for the value (its index in the type's
 *          region, 1-based; 0 = no second value) rather than the value: 4
 *          bytes per chunk instead of 8 (double, long) or 16 (complex<double>).
 *          Reading it back is one load of GlobalVecHolder::value_by_id[id].
 *          Narrower types keep the value itself: an identifier would not be
 *          smaller, and bool has no registry.
 */
//
// OPT-IN (-DCHUNK_SECOND_VALUE_INDEXED, CMake option of the same name): one extra dependent
// load on every write of a materialised chunk (a few % CPU) for a few % less RSS. For a use
// bound by memory rather than by write throughput.
template <typename T>
struct SecondValueTraits
{
    static constexpr bool indexed = build_config::second_value_indexed && sizeof(T) > 4;
    using SlotT                   = std::conditional_t<indexed, std::uint32_t, T>;
};

/**
 * @brief Handle on one chunk's second-value slot.
 *
 * @details A pointer into the per-chunk table behind a get() / set()
 *          surface, so the slot can be an identifier (SecondValueTraits). A
 *          default-constructed handle is "none".
 */
template <typename T>
struct SecondValueRef
{
    using SlotT = SecondValueTraits<T>::SlotT;

    /// @brief The chunk's second value; T{} (== first_value, 0) when none.
    [[nodiscard]] T get() const noexcept;
    /// @brief Record @p value as the chunk's second value. For an indexed type this
    ///        registers the value's slot if it is new (find_or_set).
    void set(T value) const;
    /// @brief Record the value of a slot the caller already has.
    void set_from_array(const T* src) const noexcept;
    /// @brief No second value (the `= 0` of the value table).
    void reset() const noexcept
    {
        *slot = SlotT{};
    }
    /// @brief reset() as an atomic release store: the slot of a member with the motif
    ///        state, which a reader of the chunk's motif may load meanwhile (F5).
    void reset_release() const noexcept
    {
        std::atomic_ref<SlotT>(*slot).store(SlotT{}, std::memory_order_release);
    }
    /// @brief set_from_array() as an atomic release store (see reset_release()).
    void set_from_array_release(const T* src) const noexcept;
    [[nodiscard]] explicit operator bool() const noexcept
    {
        return slot != nullptr;
    }

    SlotT* slot = nullptr;
};

/**
 * @brief Frees the whole registry at program exit, and nothing else.
 * @details A static instance lives inside GlobalVecHolder::init(), so the
 *          destructor runs at static teardown. That is the only reason this
 *          type exists: the registry outlives every array that uses it, so
 *          there is no owner whose destructor could do the job.
 */
template <typename T>
struct Sentinel
{
    Sentinel() = default;
    /// @brief Call GlobalVecHolder<T>::cleanup_all().
    ~Sentinel()
    {
        GlobalVecHolder<T>::cleanup_all();
    }
    // One per type, a function-local static: neither copied nor moved.
    Sentinel(const Sentinel&)            = delete;
    Sentinel& operator=(const Sentinel&) = delete;
    Sentinel(Sentinel&&)                 = delete;
    Sentinel& operator=(Sentinel&&)      = delete;
};

/**
 * @brief Where a type's second-value identifiers stand: what
 *        zero_any::Array<T>::second_value_stats() returns.
 *
 * @details Running out of identifiers stops the run; this tells it coming.
 *          in_use() against @c capacity, sampled over time, is the trend; a
 *          reclaim refused (@c refused_*) frees nothing although it returns 0
 *          as one with nothing to free does, and a degraded registry refuses
 *          every reclaim until its last array goes. Every field is read
 *          relaxed, callable at any time (exact at a sequential point).
 *          handed_out, free and degraded follow the registry, and start over
 *          when the last array of the type is destroyed; the reclaim counts
 *          are for the whole process.
 */
struct SecondValueStats
{
    std::size_t handed_out       = 0;     ///< Identifiers carved so far (the high-water mark).
    std::size_t free             = 0;     ///< Of which freed by a reclaim, not handed out again.
    std::size_t capacity         = 0;     ///< Identifiers the type can hand out; past it the run stops.
    bool        degraded         = false; ///< Index degraded: reclaims refused, the identifiers only grow.
    std::size_t reclaims         = 0;     ///< reclaim_second_values() calls that ran (not refused).
    std::size_t freed            = 0;     ///< Identifiers those calls freed, in total.
    std::size_t refused_parallel = 0;     ///< Calls refused: inside a parallel region.
    std::size_t refused_rechunk  = 0;     ///< Calls refused: during a re-chunk rebuild.
    std::size_t refused_degraded = 0;     ///< Calls refused: the index is degraded.

    /// @brief Identifiers some chunk or slot may still hold.
    [[nodiscard]] std::size_t in_use() const noexcept
    {
        return handed_out - free;
    }
    /// @brief Calls refused, every cause.
    [[nodiscard]] std::size_t refused() const noexcept
    {
        return refused_parallel + refused_rechunk + refused_degraded;
    }
};

/**
 * @brief The process-wide registry of second-value slots for one type.
 *
 * @tparam T Element type. One instance per type, all members static.
 *
 * @details Owns the Segment chain, the hash index over it, and the lock that
 *          serialises appends. Lives for the whole run: a slot registered here
 *          is shared by every zero_any chunk compressed to that value, so the
 *          registry cannot be freed while any array of the type is alive.
 *          #ref_count tracks that. A value nothing refers to any more can
 *          go earlier, at a sequential point: reclaim_unmarked(), driven
 *          by zero_any::Array::reclaim_second_values().
 */
template <typename T>
class GlobalVecHolder
{
  private:
    // ── The region every second-value slot of this type is carved from ─────
    //
    // One contiguous reservation per T (virtual address space, MAP_NORESERVE), so that "is this
    // chunk compressed to a second constant" is a range test on its values pointer -- no
    // per-chunk cached pointer (8 bytes per header, and a store-ordering hazard against
    // `values`). The allocator is a bump pointer plus the free list of reclaimed identifiers
    // (reused first); region_reset() gives the pages back at the last clear.
    //
    // Nothing is ever STORED in the region: a slot is an identifier. is_cst_val2() is a range
    // test and id_of_ptr() is arithmetic -- neither dereferences it -- and the value comes from
    // value_by_id[]. The read path pays one well-predicted range test (ChunkBase::element_at).
    // Shared arrays FILLED with the value would avoid the test but cost tens of KiB per value
    // and more instructions overall.
    //
    // WARNING: the failure mode here is a WRONG VALUE, not a crash: a site that reads a value
    // out of the slot, or loads `values` twice around a test, returns something plausible.
    // Re-run rechunk_validation_*, a TSan pass and a --mode check run whenever it is touched.
    //
    // Capacity: ONE bound, region_usable = min(CHUNK_CST_VAL2_REGION_BYTES, max_ids x id_stride),
    // used by the reservation, the range test (region_end) and region_alloc()'s abort. Running
    // out aborts: a fallback array outside the region would break the range test.
#ifndef CHUNK_CST_VAL2_REGION_BYTES
    #define CHUNK_CST_VAL2_REGION_BYTES (std::size_t{2} << 30U)
#endif

    static constexpr std::size_t region_bytes = CHUNK_CST_VAL2_REGION_BYTES;
    static constexpr std::size_t id_stride    = 64; ///< One cache line per identifier (see region_init()).

  public:
    // ── Value->pointer index (lock-free readers, single writer under lock) ──

    /**
     * @brief One generation of the open-addressed value -> slot table.
     *
     * @details Makes a registration an O(1) probe rather than an O(N) chain scan
     *          under the lock. A hit is revalidated (value_of_ptr(p) == v): open
     *          addressing probes across collisions. A miss (capacity / probe budget)
     *          falls back to the cold path. Entries hold the segment slots' pointers,
     *          so they MUST be cleared wherever the registry is freed. @see index_clear
     *
     *          The table doubles under insert_mutex past 50% load (@see index_grow).
     *          Do not let it fill further: at high load the probe budget runs out and
     *          every new value falls back to the O(N) scan under the mutex.
     *
     * @note `used` is maintained by the single writer only; readers never
     *       touch it.
     */
    struct IndexTable
    {
        // A value-initialised atomic pointer is null (C++20): no clearing loop.
        explicit IndexTable(std::size_t capacity) : cap(capacity), slots(capacity) {}

        std::size_t                  cap  = 0; ///< Slot count, a power of two.
        std::size_t                  used = 0; ///< Published entries (single writer).
        std::vector<std::atomic<T*>> slots;    ///< `cap` slots, null = empty.
    };

    /// @brief Largest identifier the region can hand out.
    ///
    /// @details One cache line per identifier (region_init()'s stride), so the region
    ///          alone would allow 33 M. Capped by CHUNK_CST_VAL2_MAX_IDS because
    ///          value_by_id[max_ids + 1] is then the binding cost (256 MiB of address
    ///          space per type for a double at 33 M). The default leaves ample headroom
    ///          over real workloads.
#ifndef CHUNK_CST_VAL2_MAX_IDS
    #define CHUNK_CST_VAL2_MAX_IDS (std::size_t{4} << 20U)
#endif
    static constexpr std::size_t max_ids =
        (region_bytes / id_stride) < CHUNK_CST_VAL2_MAX_IDS ? (region_bytes / id_stride) : CHUNK_CST_VAL2_MAX_IDS;
    static_assert(max_ids < std::size_t{0xFFFFFFFFu}, "an identifier must fit id_of_ptr()'s uint32_t");

    /// @brief The registered users (#ref_count): the live zero_any::Array<T>
    ///        holding a reference. For checks at a quiescent point.
    [[nodiscard]] static int users() noexcept
    {
        return ref_count.load(std::memory_order_acquire);
    }

    /**
     * @brief How many distinct second values are registered.
     * @return The sum of Segment::used over the chain.
     * @warning Walks the chain without the lock: call it at a quiescent point,
     *          not while a compression pass may be appending.
     */
    [[nodiscard]] static std::size_t count_entries() noexcept;

    /// @brief Identifiers handed out so far: the region's high-water mark,
    ///        freed identifiers included (they are reused before it moves).
    [[nodiscard]] static std::size_t ids_handed_out() noexcept
    {
        return region_used.load(std::memory_order_relaxed) / region_stride; // 0 before region_init()
    }

    /// @brief Identifiers freed by reclaim_unmarked() and not handed out again.
    ///        Callable at any time (#free_ids_count); exact at a quiescent point.
    [[nodiscard]] static std::size_t free_id_count() noexcept
    {
        return free_ids_count.load(std::memory_order_relaxed);
    }

    /// @brief Identifiers this type can hand out: region_alloc()'s bound
    ///        (max_ids, the region, or CHUNK_CST_VAL2_ID_LIMIT once read).
    [[nodiscard]] static std::size_t id_capacity() noexcept
    {
        const std::size_t region_ids = region_usable / region_stride;
        return id_limit < region_ids ? id_limit : region_ids;
    }

    // ── Reclaim counts (zero_any::Array::reclaim_second_values), whole process ─
    static inline std::atomic<std::size_t> reclaim_runs{0};     ///< Calls that ran.
    static inline std::atomic<std::size_t> reclaim_freed{0};    ///< Identifiers they freed.
    static inline std::atomic<std::size_t> refused_parallel{0}; ///< Refused: in a parallel region.
    static inline std::atomic<std::size_t> refused_rechunk{0};  ///< Refused: during a re-chunk rebuild.
    static inline std::atomic<std::size_t> refused_degraded{0}; ///< Refused: index degraded.

    /// @brief Where this type's identifiers stand. Relaxed reads, any time.
    [[nodiscard]] static SecondValueStats stats() noexcept;

    /// @brief Identifier (1-based) of the slot @p ptr.
    /// @pre region_contains(ptr).
    [[nodiscard]] static std::uint32_t slot_id(const T* ptr) noexcept
    {
        const std::byte* base = region_begin.load(std::memory_order_relaxed);
        return static_cast<std::uint32_t>((::fast_alloc::address_bits(ptr) - ::fast_alloc::address_bits(base)) /
                                          region_stride) +
               1u;
    }

    // ── Reclaim contract, checked in check builds ──────────────────────────
    //
    // A reclaim (mark + reclaim_unmarked()) has no epoch and no quiescence
    // detection: it relies on its caller's sequential point. Two failures if a
    // thread looks up or registers meanwhile, both silent in release:
    //  - a slot found BEFORE the mark and recorded in the chunk's second-value
    //    table AFTER it is unmarked, so freed; take_free_slot() hands its
    //    identifier to another value and the chunk reads that one. The
    //    revalidation (value_of_ptr(p) == v) does not catch it: value_by_id[]
    //    keeps the freed identifier's value until it is handed out again.
    //  - index_clear() destroys every index generation: a lookup holding one
    //    reads freed memory.
    // In check builds (build_config::any_check) LookupGuard counts the
    // lookups and registrations in flight, from the lookup to the record in
    // the second-value table, and ReclaimScope fails an invariant if one is
    // in flight when a reclaim starts or ends, or starts while it runs.
    // Release builds pay nothing.

    /// @brief Lookups/registrations in flight (check builds only).
    static inline std::atomic<int> lookups_in_flight{0};
    /// @brief Reclaim scopes open, nested ones included (check builds only).
    static inline std::atomic<int> reclaims_running{0};

    /// @brief Brackets a lookup or registration up to the record of its
    ///        result in the chunk's second-value table. No-op in release.
    struct LookupGuard
    {
        LookupGuard() noexcept;
        ~LookupGuard()
        {
            if constexpr (build_config::any_check)
            {
                lookups_in_flight.fetch_sub(1, std::memory_order_acq_rel);
            }
        }
        LookupGuard(const LookupGuard&)            = delete;
        LookupGuard& operator=(const LookupGuard&) = delete;
        LookupGuard(LookupGuard&&)                 = delete;
        LookupGuard& operator=(LookupGuard&&)      = delete;
    };

    /// @brief Brackets a whole reclaim, mark included. No-op in release.
    struct ReclaimScope
    {
        ReclaimScope() noexcept
        {
            if constexpr (build_config::any_check)
            {
                reclaims_running.fetch_add(1, std::memory_order_seq_cst);
                check_no_lookup("starts");
            }
        }
        ~ReclaimScope()
        {
            if constexpr (build_config::any_check)
            {
                check_no_lookup("ends");
                reclaims_running.fetch_sub(1, std::memory_order_release);
            }
        }
        ReclaimScope(const ReclaimScope&)            = delete;
        ReclaimScope& operator=(const ReclaimScope&) = delete;
        ReclaimScope(ReclaimScope&&)                 = delete;
        ReclaimScope& operator=(ReclaimScope&&)      = delete;

        /// @brief Fail if a lookup or registration is in flight.
        static void check_no_lookup(const char* when) noexcept;
    };

    /**
     * @brief Forget every registered value whose identifier is not marked in
     *        @p referenced, and make those identifiers reusable.
     *
     * @param referenced `referenced[id] != 0` for every identifier something still refers
     *        to. An identifier past its end counts as live.
     * @return How many identifiers were freed.
     *
     * @details The surviving slots keep their addresses and identifiers: no
     *          chunk is touched. The chain is rebuilt packed, the index
     *          rebuilt from it (a fresh generation, the superseded ones
     *          dropped), and the freed identifiers go to #free_ids, which
     *          take_free_slot() drains before the region is carved again.
     *          value_by_id[] of a freed identifier keeps its old value until
     *          the identifier is handed out again.
     *
     * @warning clear_if_last()'s contract -- no thread registering, looking
     *          up or scanning -- AND nothing may still hold a freed
     *          identifier. That second half is the caller's mark
     *          (zero_any::Array::reclaim_second_values()): an identifier it
     *          misses goes to another value, and the chunk that kept it
     *          reads that value instead of its own, silently.
     * @warning "No thread looking up" includes a lookup already DONE whose
     *          result is not yet recorded in the chunk's second-value table:
     *          the mark does not see that slot, frees it, and its identifier
     *          goes to another value although the revalidation passed
     *          (value_by_id[] keeps the old value until reuse). And
     *          index_clear() below frees the index generation a lookup in
     *          flight may hold. No epoch guards either case; check builds
     *          detect them (LookupGuard / ReclaimScope).
     */
    static std::size_t reclaim_unmarked(const std::vector<std::uint8_t>& referenced);

    // ── The members of the zero_any family that share this registry ──────────
    //
    // Every array of the family with element type T draws its second values from this one
    // registry: zero_any::Array<T> and pattern::Array<T> alike. A reclaim must mark what ALL
    // of them still name: each member registers its mark here on its first use of the
    // registry (zero_any::ArrayImpl::acquire_cst2_registry()), and
    // zero_any::ArrayImpl::reclaim_second_values(), on any member, runs them all.

    /// @brief A member's mark: sets referenced[id] for each identifier its live arrays name.
    using MarkFn = void (*)(std::vector<std::uint8_t>& referenced);

    /// @brief How many members can register a mark (zero_any, pattern, and room to spare).
    static constexpr std::size_t max_markers = 4;

    /**
     * @brief Register @p mark, once: a mark already there is not added again.
     * @details Lock-free (a CAS per slot), callable from any thread. More members than
     *          max_markers is an invariant failure: a member left out would have its
     *          values freed under it.
     * @param mark The member's mark function.
     */
    static void register_marker(MarkFn mark) noexcept;

    /// @brief Run every registered mark over @p referenced (reclaim_second_values()).
    /// @param referenced One flag per identifier, set for those still named.
    static void mark_all(std::vector<std::uint8_t>& referenced);

    /// @brief The registered marks (register_marker()); null = free slot. Never
    ///        unregistered: a mark only reads its member's registry of arrays.
    static inline std::array<std::atomic<MarkFn>, max_markers> markers{};

    /// @brief A freed identifier's slot, or null when none is free.
    /// @pre Under #insert_mutex (alloc_chunk(), from find_or_register()).
    [[nodiscard]] static T* take_free_slot() noexcept;

    /// @brief Is @p ptr a second-value slot of this type?
    /// @note THE constant-2 test of zero_any: `is_cst_val2()` is
    ///       `region_contains(values)`. Relaxed load of a pointer that is
    ///       written once, before any chunk exists.
    [[nodiscard]] static bool region_contains(const void* ptr) noexcept
    {
        const std::byte* base = region_begin.load(std::memory_order_relaxed);
        return base != nullptr && static_cast<const std::byte*>(ptr) >= base &&
               static_cast<const std::byte*>(ptr) < region_end;
    }

    /// @brief Carve @p bytes from the region.
    /// @pre init() ran. Called under #insert_mutex (single appender); the
    ///      fetch_add is belt and braces.
    static std::byte* region_alloc(std::size_t bytes);

    /**
     * @brief The value the slot @p ptr stands for.
     *
     * @details The one way to recover a value from a slot (probe revalidations,
     *          rehash, degraded scans). Never `p[0]`: nothing is stored at a slot, so
     *          that reads a zero page (without MM_PROTECT), no value is ever found
     *          again and every registration takes the lock.
     *
     * @note Not a dereference: id_of_ptr() is arithmetic on the
     *       region base and stride, and value_by_id[] is written by
     *       alloc_chunk() before the pointer can be published.
     */
    [[nodiscard]] static T value_of_ptr(const T* ptr) noexcept
    {
        const std::byte* base = region_begin.load(std::memory_order_relaxed);
        const auto id = ((::fast_alloc::address_bits(ptr) - ::fast_alloc::address_bits(base)) / region_stride) + 1u;
        return value_by_id[id];
    }

    /**
     * @brief Look @p value up in the index, lock-free. This is the hot path.
     * @return Its slot, or null -- meaning "not found HERE", never
     *         "not registered". The caller must fall back.
     * @details A hit is revalidated (value_of_ptr(p) == v) whatever the
     *          concurrency: open addressing probes across collisions, so the
     *          slot reached may belong to another value.
     */
    static T* index_lookup(InT<T> value) noexcept;

    /**
     * @brief Record that @p value is served by @p ptr. No-op if already present.
     * @pre Called under #insert_mutex -- this is the single writer.
     * @details A plain release store suffices: it pairs with the acquire load
     *          in index_lookup(), so a lock-free reader that sees the pointer
     *          also sees the value_by_id[] entry written before publication.
     * @note Giving up after #index_max_probe sets #index_degraded rather than
     *       failing: the value IS registered in the chain, just not indexed.
     */
    static void index_publish(InT<T> value, T* ptr) noexcept;

    /**
     * @brief Register one more user of this type's registry, creating it if new.
     * @details The user is counted in #ref_count; the registry itself is made
     *          valid by install_head().
     */
    static void init()
    {
        // One more zero_any::Array<T> is using the shared list.
        ref_count.fetch_add(1, std::memory_order_relaxed);
        install_head();
    }

    /**
     * @brief Make the registry usable -- the region reserved, a first segment
     *        in #head -- WITHOUT registering a user.
     * @details Idempotent: the first segment is installed once, by whichever
     *          caller wins the CAS. Also what arranges teardown, through the
     *          function-local Sentinel. Split from init() so that
     *          clear_if_last() can reinstall an empty chain without registering
     *          a user: a reference taken with no array alive would keep
     *          #ref_count above 0, and the registry would never be freed again.
     */
    static void install_head();

    /**
     * @brief Chain a fresh segment after @p last and return it.
     * @pre Called under #insert_mutex -- single appender, hence a plain store
     *      with no CAS and no lost-race branch.
     */
    static Segment<T>* append_segment(Segment<T>* last)
    {
        auto* new_seg = new Segment<T>();
        last->next    = new_seg;
        tail          = new_seg; // keep the cached tail in sync (under insert_mutex)
        return new_seg;
    }

    /**
     * @brief Drop one user; free the whole registry when it was the last.
     *
     * @param nb_val2 Incremented by the number of registered values freed.
     *
     * @details Decrements #ref_count and returns immediately while other
     *          zero_any::Array<T> instances are alive. The last one out frees every
     *          segment, clears the index, gives the region back, then calls
     *          install_head() so #head is valid again for any later pass --
     *          without registering a user: #ref_count stays at 0, and the
     *          next generation of arrays is freed by its own last one out.
     *
     * @warning No thread may be SCANNING the chain while this runs --
     *          find_or_set_ind_cst_val2(), reached from make_block_compression
     *          on the write slow path. Call it serially, or from an
     *          `omp single` PRECEDED BY A BARRIER that ends every concurrent
     *          scan (a `nowait` loop before the single lets a late writer's
     *          scan dereference a freed Segment).
     *
     * @note No assertion guards this: there is no reliable OpenMP predicate for
     *       "I am the lone thread of a single". omp_in_parallel() is the wrong
     *       test, since the single legitimately runs inside a parallel region.
     */
    static void clear_if_last(int& nb_val2);

    /**
     * @brief Free everything unconditionally, ignoring #ref_count.
     * @warning Teardown only -- the Sentinel calls it at static destruction. It
     *          does not reinstall an empty segment, so #head is left null and
     *          the registry is unusable afterwards.
     */
    static void cleanup_all();

    /// @brief First segment. NEVER null after init() -- clear_if_last()
    ///        reinstalls an empty one (install_head(), which registers no
    ///        user) so no caller has to test for it.
    static inline std::atomic<Segment<T>*> head{nullptr};
    // Cached tail segment: O(1) cold-path append. Mutated only under insert_mutex or
    // single-threaded (init/clear); index_clear nulls it and the next append re-seeds it.
    static inline Segment<T>* tail{nullptr};

    /**
     * @brief Serialises the COLD path only: registering a genuinely new value.
     * @details Every mutation of the chain and every index publication happens
     *          under it, which is what makes the index a single-writer
     *          structure. The hot path never takes it.
     * @note A std::mutex rather than a spinlock on purpose: the critical
     *       section may mmap/mprotect under MM_PROTECT builds.
     */
    static inline std::mutex insert_mutex;

    /**
     * @brief Set once if index_publish() ever exhausts its probe budget.
     * @details While it is false -- the expected regime, a few thousand values
     *          in a table at under 1% load -- the cold path trusts the index as
     *          authoritative and skips the O(N) chain scan entirely. Once true,
     *          a registered value may be unfindable by index_lookup(), so that
     *          scan comes back. Sticky until index_clear(); written only under
     *          #insert_mutex, read relaxed.
     */
    static inline std::atomic<bool> index_degraded{false};

    static inline std::atomic<std::byte*> region_begin{nullptr}; ///< Reservation start; null before init().
    /// @brief Bytes per slot: every slot is carved at a multiple of it.
    /// @note constexpr, not set by region_init(): a pointer <-> identifier
    ///       conversion (id_of_ptr, slot_id, on zero_any's constant-2 reads)
    ///       is then a shift by 6, not a 64-bit `div` by a global.
    static constexpr std::size_t region_stride = id_stride;

    /// @brief The value registered under each identifier, contiguous: what
    ///        a chunk's second value is read through on every write of a
    ///        wide type (SecondValueRef::get). 8 bytes per value, L1-resident
    ///        for the values a run actually uses. Zero-initialised static
    ///        storage: the pages of identifiers never handed out are never
    ///        touched. Index 0 is "no second value" and stays T{}.
    static inline std::array<T, max_ids + 1> value_by_id{};

  private:
    /// @brief What is reserved and carved: exactly the identifiers
    ///        value_by_id can hold. 256 MiB per type with the defaults.
    static constexpr std::size_t region_usable =
        (max_ids * id_stride) < region_bytes ? (max_ids * id_stride) : region_bytes;

#ifndef CHUNK_CST_VAL2_INDEX_LOG2
    static constexpr int chunk_cst_val2_index_loc = 18;
    // 2^18 = 262144 slots, 2 Mio/type; <11% load at 30k values
#else
    static constexpr int chunk_cst_val2_index_loc = CHUNK_CST_VAL2_INDEX_LOG2;
#endif

    static constexpr size_t index_cap       = size_t{1} << chunk_cst_val2_index_loc; ///< Initial capacity.
    static constexpr int    index_max_probe = 64; ///< Bounded; giving up means fallback.

    /// @brief Reserve the region. Idempotent; called by init().
    static void region_init();

    /// @brief Give the carved pages back and start over. Last clear only.
    static void region_reset() noexcept;

    /// @brief Value hash (a 64-bit finaliser over the object's bytes).
    /// @note Hashes the REPRESENTATION, so two values that compare equal but
    ///       differ bitwise would land in different slots.
    static size_t index_hash(InT<T> value) noexcept;

    /**
     * @brief Publish a table twice the size of the live one (or the initial
     *        one), carrying every entry over.
     * @pre Called under #insert_mutex -- the single writer, so the live table
     *      is not being mutated while it is copied.
     * @details Built completely, then published with one release store, so a
     *          reader sees either the old generation (complete as of its
     *          publication) or the new one (a superset). Rehashing reads each
     *          slot's value through value_of_ptr().
     * @return The new live table.
     */
    static IndexTable* index_grow() noexcept;

    /**
     * @brief Empty the index and drop the cached tail.
     * @warning Single-threaded clear paths only, and at the very point the
     *          slots are given back -- an entry outliving its slot is a
     *          dangling pointer a lock-free reader will follow.
     */
    static void index_clear() noexcept;

    /// @brief Live zero_any::Array<T> instances. The registry is freed when it
    ///        reaches zero, not before. @see clear_if_last
    static inline std::atomic<int> ref_count{0};

    static inline std::byte*               region_end = nullptr; ///< One past the reservation.
    static inline std::atomic<std::size_t> region_used{0};       ///< Bump offset, in bytes.

    /// @brief The identifiers region_alloc() hands out: max_ids, or fewer
    ///        under CHUNK_CST_VAL2_ID_LIMIT (read once in region_init(); it
    ///        can only lower the bound -- the way a test reaches it).
    static inline std::size_t id_limit = max_ids;

    /// @brief Identifiers freed by reclaim_unmarked(), handed out again by
    ///        take_free_slot() before the region's bump pointer moves.
    ///        Emptied with the region (region_reset()). Immortal, like
    ///        index_generations: a global array registers values during
    ///        static teardown too.
    static inline constinit internal::Immortal<std::vector<std::uint32_t>> free_ids_storage_{};
    static inline std::vector<std::uint32_t>&                              free_ids = free_ids_storage_.value;
    /// @brief free_ids.size(), readable without the lock (free_id_count());
    ///        written with free_ids, under #insert_mutex or at the last clear.
    static inline std::atomic<std::size_t> free_ids_count{0};

    /// @brief The live table. Null until the first publish; readers that see
    ///        null miss and take the cold path, which is always correct.
    static inline std::atomic<IndexTable*> index_tab{nullptr};
    /// @brief Every table ever published, the live one last. Old generations
    ///        are kept until index_clear(): a lock-free reader may still hold
    ///        their pointer, and index_clear() is the one point where no
    ///        reader is in flight (its contract). They sum to less than the
    ///        live table, so this is bounded by the same factor of two.
    static inline constinit internal::Immortal<std::vector<std::unique_ptr<IndexTable>>> index_generations_storage_{};
    static inline std::vector<std::unique_ptr<IndexTable>>& index_generations = index_generations_storage_.value;
};

/**
 * @brief The shared second-value slots, as a chunk sees them.
 *
 * @tparam T Element type.
 *
 * @details Stateless: everything is static and delegates to
 *          GlobalVecHolder. This type is the surface the compression path
 *          talks to, and the place the allocation strategy lives.
 *          "Compressed to a second constant" is a range test on the
 *          chunk's values pointer: no per-chunk state.
 *
 * @par Invariants the rest of the code relies on
 *  - GlobalVecHolder::head is never null after create() or clear(): clear()
 *    reinstalls an empty segment, so a later compression pass finds a valid
 *    head without testing.
 *  - Every slot handed out by find_or_set_ind_cst_val2() lies in the region,
 *    and nothing else does: is_cst_val2_ptr() is exact.
 *
 * @see TabCstVal2Holder<bool>, which has none of this machinery.
 */
template <typename T>
class TabCstVal2Holder
{
  public:
    using VecHolder = GlobalVecHolder<T>;

    /**
     * @brief The slot for @p second_v, registering it if new.
     *
     * @param second_v The constant to compress to.
     * @param ptr_second_value Out-parameter: the caller's own copy of the value,
     *        written before anything else.
     * @return The value's slot (an identifier, never dereferenced). Never null.
     *
     * @note The chunk adopts the slot by storing it as its values pointer
     *       (the compressor's CAS); nothing else is recorded per chunk.
     *
     * @details Hot path (value already registered): a lock-free index probe.
     *          Cold path: under GlobalVecHolder::insert_mutex (see file comment).
     *
     * @note The O(N) chain scan runs only behind GlobalVecHolder::index_degraded.
     */
    [[nodiscard]] T* find_or_set_ind_cst_val2(T second_v, SecondValueRef<T> ptr_second_value);

    /// @brief find_or_set_ind_cst_val2() recording the slot by a release store
    ///        (SecondValueRef::set_from_array_release()): a member with the motif state.
    [[nodiscard]] T* find_or_set_ind_cst_val2_release(T second_v, SecondValueRef<T> ptr_second_value);

    /// @brief Is @p ptr one of this type's second-value slots?
    /// @note `is_cst_val2()` of zero_any: one relaxed load of the region base
    ///       and two compares, no per-chunk state.
    [[nodiscard]] static bool is_cst_val2_ptr(const T* ptr) noexcept
    {
        return GlobalVecHolder<T>::region_contains(ptr);
    }

    /// @brief The slot registered for @p second_v, or null if none.
    /// @details Lock-free index first; if the index has ever overflowed its
    ///          probe budget (index_degraded) a registered value may be
    ///          unindexed, and the authoritative slot scan runs under the
    ///          insert lock. Used where a chunk must be pointed back at its
    ///          constant from the per-chunk tables alone
    ///          (init_values_cst_val): re-sync after a lost CAS, re-chunk,
    ///          teardown -- never on the element write path.
    [[nodiscard]] static T* find_cst_val2_ptr(InT<T> second_v);

    // ── Lifecycle ─────────────────────────────────────────────────────────

    /// @brief Register a user of this type's registry. Idempotent.
    /// @warning Must run before any compression pass on this type; it is what
    ///          guarantees GlobalVecHolder::head is non-null.
    static void create()
    {
        VecHolder::init();
    }

    // ── Identifiers: a slot's index in the region, 1-based; 0 = none ──────
    // The per-chunk second-value table of a wide type holds these instead of
    // values (SecondValueTraits). id_of_ptr() is arithmetic on the region
    // base and stride; value_of_id() reads value_by_id[].

    /// @brief Identifier of the slot @p ptr (must lie in the region).
    [[nodiscard]] static std::uint32_t id_of_ptr(const T* ptr) noexcept
    {
        const std::byte* base = VecHolder::region_begin.load(std::memory_order_relaxed);
        return static_cast<std::uint32_t>((::fast_alloc::address_bits(ptr) - ::fast_alloc::address_bits(base)) /
                                          VecHolder::region_stride) +
               1u;
    }
    /**
     * @brief value_of_ptr() with the identifier range checked.
     *
     * @details Same conversion, but returns T{} rather than indexing out of
     *          bounds if @p ptr is not where the caller thought. A pointer that
     *          passes is_cst_val2_ptr() can only yield an id in [1, max_ids],
     *          so this never fires in a correct call -- it guards a pointer
     *          that was not range-tested, where a compare is cheaper than a
     *          crash.
     */
    [[nodiscard]] static T value_of_ptr_checked(const T* ptr) noexcept;

    /// @brief The value registered under @p id; T{} for 0.
    [[nodiscard]] static T value_of_id(std::uint32_t id) noexcept
    {
        return VecHolder::value_by_id[id]; // [0] is T{}
    }

    /**
     * @brief Drop this user of the registry; the last one frees it.
     * @param nb_val2 Incremented by the number of registered values freed.
     * @warning Single thread only, and with no concurrent scan in flight.
     *          @see GlobalVecHolder::clear_if_last for the full contract.
     */
    static void clear(int& nb_val2)
    {
        VecHolder::clear_if_last(nb_val2);
    }

    /// @brief The identifier for @p value, registering its slot if it is new;
    ///        0 for T{} (the "no second value" the value table wrote as 0).
    [[nodiscard]] static std::uint32_t id_for_value(InT<T> value)
    {
        if (value == T{})
        {
            return 0;
        }
        return id_of_ptr(find_or_register(value));
    }

  private:
    using SegmentT = Segment<T>;

    /**
     * @brief Carve one identifier slot for @p val and record the value.
     * @return The slot pointer: never dereferenced, only range-tested and
     *         converted to an identifier.
     * @warning value_by_id[] is written BEFORE the caller publishes the
     *          pointer. That ordering is what index_publish()'s release store
     *          makes visible.
     */
    static T* alloc_chunk(T val);

    /**
     * @brief The slot for @p second_v, registering it if new.
     * @return The value's slot (an identifier, never dereferenced). Never null.
     */
    [[nodiscard]] static T* find_or_register(InT<T> second_v);
};

/**
 * @brief TabCstVal2Holder for bool, where the whole registry collapses.
 *
 * @details A bool has exactly one value that is not the first constant, so
 *          there is nothing to register, look up, share or free: a single
 *          static array of `true` serves every chunk in the process. Every verb
 *          of the primary template is kept so call sites compile uniformly, and
 *          all of them are trivial.
 *
 * @warning find_or_set_ind_cst_val2() takes NO arguments here, unlike the
 *          primary template's two. A call site that serves both must dispatch,
 *          not just forward.
 */
template <>
class TabCstVal2Holder<bool>
{
  public:
    /// @brief Nothing to create. @see TabCstVal2Holder::create
    static void create() {}
    /// @brief Nothing to free, and no user count to keep.
    static void clear(int& /*nb_val2*/) {}

    /// @brief The shared array. There is only ever one, so nothing is searched.
    static bool* find_or_set_ind_cst_val2()
    {
        return m_ptr.data();
    }

    /// @brief Is @p ptr the shared array? There is only one.
    [[nodiscard]] static bool is_cst_val2_ptr(const bool* ptr) noexcept
    {
        return ptr == m_ptr.data();
    }
    /// @brief The shared array, whatever the value: it is `true` or nothing.
    [[nodiscard]] static bool* find_cst_val2_ptr(bool /*second_v*/) noexcept
    {
        return m_ptr.data();
    }

    /// @brief The one shared array, all `true`. constinit: a global array can
    ///        reach it before any dynamic initialisation has run.
    static inline constinit std::array<bool, cst_buffer_size> m_ptr = []
    {
        std::array<bool, cst_buffer_size> arr{};
        arr.fill(true);
        return arr;
    }();
};

} // namespace shrink_array

#include "chunk/internal/tab_cst_val2_holder_inline.h" // the template bodies
