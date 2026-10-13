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
 * A Chunk2A compressed to a constant other than zero does not own a buffer: it
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
 * first one per value -- is a lock-free probe of GlobalVecHolder::index_.
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
 * unless ArrayChunk2A::reclaim_second_values() runs at a sequential point:
 * it marks the identifiers still referenced and
 * GlobalVecHolder::reclaim_unmarked() frees the others. alloc_chunk() hands
 * a freed identifier out again before carving a new one, so a workload
 * whose constants come and go stays within the identifiers it has carved.
 */

#include <sys/mman.h>  // mmap, mprotect, madvise
#include <cerrno>      // errno (the region's mmap)
#include <cstring>     // std::memcpy, std::strerror
#include <cstdint>     // std::uint32_t, uint64_t
#include <atomic>      // std::atomic, std::memory_order
#include <array>       // std::array
#include <memory>      // std::unique_ptr, std::make_unique
#include <mutex>       // std::mutex, std::scoped_lock
#include <type_traits> // std::conditional_t
#include <vector>      // index_generations
// <iostream> directly: param_chunk.h would be a circular include (it reaches this header).
#include <iostream> // std::cerr, std::cout

#include "chunk/internal/chunk_size_policy.h" // cst_buffer_size
#include "chunk/internal/env_parse.h"         // env_number (CHUNK_CST_VAL2_ID_LIMIT)
#include "chunk/internal/in_t.h"              // in_t<T>: the value parameters
#include "chunk/config.h"                     // build_config::use_mm_protect
#include "chunk/internal/invariant_check.h" // CHUNK_INVARIANT_FAILED
#include "chunk/internal/immortal.h"          // Immortal / LazyImmortal: statics safe for global arrays
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
 *          through GlobalVecHolder::index_ and never touch a Segment at all.
 *
 * @note The chain is append-only between clears and reclaims
 *       (GlobalVecHolder::reclaim_unmarked() rebuilds it packed), so
 *       occupancy is contiguous:
 *       `slots[0 .. used)` are set and `slots[used .. SIZE)` are null. Several
 *       loops here rely on that rather than null-checking every slot.
 */
template <typename T>
struct Segment
{
    static constexpr int SIZE = 64; ///< Slots per segment.

    std::array<T*, SIZE> slots{};        ///< The slots, one per registered value.
    int                  used = 0;       ///< How many leading slots are occupied.
    Segment*             next = nullptr; ///< Next block, or null at the tail.
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
    using slot_t                  = std::conditional_t<indexed, std::uint32_t, T>;
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
    using slot_t = SecondValueTraits<T>::slot_t;

    /// @brief The chunk's second value; T{} (== first_value, 0) when none.
    [[nodiscard]] T get() const noexcept
    {
        if constexpr (SecondValueTraits<T>::indexed)
        {
            return TabCstVal2Holder<T>::value_of_id(*slot);
        }
        else
        {
            return *slot;
        }
    }
    /// @brief Record @p v as the chunk's second value. For an indexed type this
    ///        registers the value's slot if it is new (find_or_set).
    void set(T v) const
    {
        if constexpr (SecondValueTraits<T>::indexed)
        {
            // Up to the record: a reclaim must not run in between (LookupGuard).
            const typename GlobalVecHolder<T>::LookupGuard in_flight;
            *slot = TabCstVal2Holder<T>::id_for_value(v);
        }
        else
        {
            *slot = v;
        }
    }
    /// @brief Record the value of a slot the caller already has.
    void set_from_array(const T* src) const noexcept
    {
        // bool has no registry (TabCstVal2Holder<bool>): value_of_ptr would index
        // value_by_id with an address-sized id. Never called for bool today.
        static_assert(!std::is_same_v<T, bool>, "SecondValueRef<bool>::set_from_array: bool has no registry");
        if constexpr (SecondValueTraits<T>::indexed)
        {
            *slot = TabCstVal2Holder<T>::id_of_ptr(src);
        }
        else
        {
            // NOT *src: nothing is stored at a slot, so that would record 0 and mark the
            // chunk constant at the wrong value, silently.
            *slot = GlobalVecHolder<T>::value_of_ptr(src);
        }
    }
    /// @brief No second value (the `= 0` of the value table).
    void reset() const noexcept
    {
        *slot = slot_t{};
    }
    [[nodiscard]] explicit operator bool() const noexcept
    {
        return slot != nullptr;
    }

    slot_t* slot = nullptr;
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
 * @brief The process-wide registry of second-value slots for one type.
 *
 * @tparam T Element type. One instance per type, all members static.
 *
 * @details Owns the Segment chain, the hash index over it, and the lock that
 *          serialises appends. Lives for the whole run: a slot registered here
 *          is shared by every Chunk2A compressed to that value, so the
 *          registry cannot be freed while any array of the type is alive.
 *          #ref_count tracks that. A value nothing refers to any more can
 *          go earlier, at a sequential point: reclaim_unmarked(), driven
 *          by ArrayChunk2A::reclaim_second_values().
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
    // Capacity: ONE bound, kRegionUsable = min(CHUNK_CST_VAL2_REGION_BYTES, kMaxIds x kIdStride),
    // used by the reservation, the range test (region_end) and region_alloc()'s abort. Running
    // out aborts: a fallback array outside the region would break the range test.
#ifndef CHUNK_CST_VAL2_REGION_BYTES
    #define CHUNK_CST_VAL2_REGION_BYTES (std::size_t{2} << 30U)
#endif

    static constexpr std::size_t kRegionBytes = CHUNK_CST_VAL2_REGION_BYTES;
    static constexpr std::size_t kIdStride    = 64; ///< One cache line per identifier (see region_init()).

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
    ///          value_by_id[kMaxIds + 1] is then the binding cost (256 MiB of address
    ///          space per type for a double at 33 M). The default leaves ample headroom
    ///          over real workloads.
#ifndef CHUNK_CST_VAL2_MAX_IDS
    #define CHUNK_CST_VAL2_MAX_IDS (std::size_t{4} << 20U)
#endif
    static constexpr std::size_t kMaxIds =
        (kRegionBytes / kIdStride) < CHUNK_CST_VAL2_MAX_IDS ? (kRegionBytes / kIdStride) : CHUNK_CST_VAL2_MAX_IDS;
    static_assert(kMaxIds < std::size_t{0xFFFFFFFFu}, "an identifier must fit id_of_ptr()'s uint32_t");

    /// @brief The registered users (#ref_count): the live ArrayChunk2A<T>
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
    [[nodiscard]] static std::size_t count_entries() noexcept
    {
        std::size_t count = 0;
        for (Segment<T>* s = head.load(std::memory_order_acquire); s != nullptr; s = s->next)
        {
            count += static_cast<std::size_t>(s->used);
        }
        return count;
    }

    /// @brief Identifiers handed out so far: the region's high-water mark,
    ///        freed identifiers included (they are reused before it moves).
    [[nodiscard]] static std::size_t ids_handed_out() noexcept
    {
        return region_used.load(std::memory_order_relaxed) / region_stride; // 0 before region_init()
    }

    /// @brief Identifiers freed by reclaim_unmarked() and not handed out again.
    /// @warning Quiescent point only, like count_entries().
    [[nodiscard]] static std::size_t free_id_count() noexcept
    {
        return free_ids.size();
    }

    /// @brief Identifier (1-based) of the slot @p ptr.
    /// @pre region_contains(ptr).
    [[nodiscard]] static std::uint32_t slot_id(const T* ptr) noexcept
    {
        const char* base = region_begin.load(std::memory_order_relaxed);
        return static_cast<std::uint32_t>((reinterpret_cast<const char*>(ptr) - base) / region_stride) + 1u;
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
        LookupGuard() noexcept
        {
            if constexpr (build_config::any_check)
            {
                // seq_cst on both sides (ReclaimScope): each increments its own
                // counter then reads the other's, so one of the two always sees
                // the other.
                lookups_in_flight.fetch_add(1, std::memory_order_seq_cst);
                if (reclaims_running.load(std::memory_order_seq_cst) != 0) [[unlikely]]
                {
                    CHUNK_INVARIANT_FAILED("cst_val2: second-value lookup or registration during a reclaim "
                                           "(reclaim_second_values() needs a sequential point)");
                }
            }
        }
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
        static void check_no_lookup(const char* when) noexcept
        {
            if (const int n = lookups_in_flight.load(std::memory_order_seq_cst); n != 0) [[unlikely]]
            {
                CHUNK_INVARIANT_FAILED("cst_val2: reclaim {} with {} second-value lookup(s) or registration(s) "
                                       "in flight (reclaim_second_values() needs a sequential point)",
                                       when, n);
            }
        }
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
     *          (ArrayChunk2A::reclaim_second_values()): an identifier it
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
    static std::size_t reclaim_unmarked(const std::vector<std::uint8_t>& referenced)
    {
        const ReclaimScope     scope; // check builds: no lookup in flight (nests in the caller's)
        const std::scoped_lock lk(insert_mutex); // the contract excludes appenders: uncontended

        std::vector<T*> kept;
        std::size_t     freed = 0;
        for (const Segment<T>* s = head.load(std::memory_order_acquire); s != nullptr; s = s->next)
        {
            for (int i = 0; i < s->used; ++i)
            {
                T* const            slot = s->slots[i];
                const std::uint32_t id   = slot_id(slot);
                if (id < referenced.size() && referenced[id] == 0)
                {
                    free_ids.push_back(id);
                    ++freed;
                }
                else
                {
                    kept.push_back(slot);
                }
            }
        }
        if (freed == 0)
        {
            return 0;
        }

        // The chain again, packed: occupancy stays contiguous (Segment's note).
        auto*       first = new Segment<T>();
        Segment<T>* last  = first;
        for (T* slot : kept)
        {
            if (last->used == Segment<T>::SIZE)
            {
                last->next = new Segment<T>();
                last       = last->next;
            }
            last->slots[last->used] = slot;
            ++last->used;
        }
        const Segment<T>* old = head.exchange(first, std::memory_order_acq_rel);
        while (old != nullptr)
        {
            const Segment<T>* next = old->next;
            delete old;
            old = next;
        }

        // The index again, from the survivors only: an entry left for a freed
        // slot would answer a lookup of the old value with an identifier that
        // has gone to another one.
        index_clear(); // also drops index_degraded and the cached tail
        for (T* slot : kept)
        {
            index_publish(value_of_ptr(slot), slot);
        }
        tail = last;
        return freed;
    }

    /// @brief A freed identifier's slot, or null when none is free.
    /// @pre Under #insert_mutex (alloc_chunk(), from find_or_register()).
    [[nodiscard]] static T* take_free_slot() noexcept
    {
        if (free_ids.empty())
        {
            return nullptr;
        }
        const std::uint32_t id = free_ids.back();
        free_ids.pop_back();
        return reinterpret_cast<T*>(region_begin.load(std::memory_order_relaxed) +
                                    ((static_cast<std::size_t>(id) - 1u) * region_stride));
    }

    /// @brief Is @p ptr a second-value slot of this type?
    /// @note THE constant-2 test of Chunk2A: `is_cst_val2()` is
    ///       `region_contains(values)`. Relaxed load of a pointer that is
    ///       written once, before any chunk exists.
    [[nodiscard]] static bool region_contains(const void* ptr) noexcept
    {
        const char* base = region_begin.load(std::memory_order_relaxed);
        return base != nullptr && static_cast<const char*>(ptr) >= base && static_cast<const char*>(ptr) < region_end;
    }

    /// @brief Carve @p bytes from the region.
    /// @pre init() ran. Called under #insert_mutex (single appender); the
    ///      fetch_add is belt and braces.
    static char* region_alloc(std::size_t bytes)
    {
        const std::size_t off = region_used.fetch_add(bytes, std::memory_order_relaxed);
        // ONE capacity: what value_by_id can hold (kRegionUsable, lowered by
        // CHUNK_CST_VAL2_ID_LIMIT); alloc_chunk() writes value_by_id[id] for every slot
        // carved here, so a slot past it would be a write past the table. Warn once at half.
        const std::size_t cap_bytes =
            (bytes != 0 && id_limit * bytes < kRegionUsable) ? id_limit * bytes : kRegionUsable;
        const char* limiting =
            (kMaxIds * kIdStride) < kRegionBytes ? "CHUNK_CST_VAL2_MAX_IDS" : "CHUNK_CST_VAL2_REGION_BYTES";
        if (id_limit < kMaxIds)
        {
            limiting = "CHUNK_CST_VAL2_ID_LIMIT";
        }
        if (bytes != 0 && off + bytes > cap_bytes / 2) [[unlikely]]
        {
            static std::atomic warned{false};
            bool               expected = false;
            if (warned.compare_exchange_strong(expected, true, std::memory_order_relaxed))
            {
                std::cerr << "[cst_val2] second-value identifiers past half used: " << (off / bytes) << " of "
                          << (cap_bytes / bytes) << " (bound by " << limiting
                          << "). Raise it if the count keeps growing; running out aborts.\n";
            }
        }
        if (off + bytes > cap_bytes) [[unlikely]]
        {
            CHUNK_INVARIANT_FAILED("[cst_val2] second-value identifiers exhausted: {} of {}, bound by {} "
                                   "(CHUNK_CST_VAL2_MAX_IDS={}, CHUNK_CST_VAL2_REGION_BYTES={} MiB). Raise it, or "
                                   "reduce the number of DISTINCT second values the workload registers.",
                                   off / bytes, cap_bytes / (bytes != 0 ? bytes : 1), limiting, kMaxIds,
                                   kRegionBytes >> 20U);
        }
        return region_begin.load(std::memory_order_relaxed) + off;
    }

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
        const char* base = region_begin.load(std::memory_order_relaxed);
        const auto  id   = ((reinterpret_cast<const char*>(ptr) - base) / region_stride) + 1u;
        return value_by_id[id];
    }

    /**
     * @brief Look @p v up in the index, lock-free. This is the hot path.
     * @return Its slot, or null -- meaning "not found HERE", never
     *         "not registered". The caller must fall back.
     * @details A hit is revalidated (value_of_ptr(p) == v) whatever the
     *          concurrency: open addressing probes across collisions, so the
     *          slot reached may belong to another value.
     */
    static T* index_lookup(in_t<T> v) noexcept
    {
        // The table named here is never freed while a reader can hold it (see
        // index_generations). A superseded generation misses newer values: a miss
        // only means "not found here".
        const IndexTable* table = index_tab.load(std::memory_order_acquire);
        if (table == nullptr)
        {
            return nullptr;
        }
        const size_t mask = table->cap - 1;
        const size_t slot = index_hash(v) & mask;
        for (int probe = 0; probe < INDEX_MAX_PROBE; ++probe)
        {
            T* entry = table->slots[(slot + probe) & mask].load(std::memory_order_acquire);
            if (entry == nullptr)
            {
                return nullptr; // empty slot => v is not indexed here
            }
            if (value_of_ptr(entry) == v)
            {
                return entry; // validated hit
            }
        }
        return nullptr; // probe budget exhausted => treat as miss
    }

    /**
     * @brief Record that @p v is served by @p ptr. No-op if already present.
     * @pre Called under #insert_mutex -- this is the single writer.
     * @details A plain release store suffices: it pairs with the acquire load
     *          in index_lookup(), so a lock-free reader that sees the pointer
     *          also sees the value_by_id[] entry written before publication.
     * @note Giving up after #INDEX_MAX_PROBE sets #index_degraded rather than
     *       failing: the value IS registered in the chain, just not indexed.
     */
    static void index_publish(in_t<T> v, T* ptr) noexcept
    {
        IndexTable* table = index_tab.load(std::memory_order_relaxed);
        // Grow before the table passes half full: linear probing degrades
        // sharply beyond that.
        if (table == nullptr || (table->used + 1) * 2 > table->cap)
        {
            table = index_grow();
        }
        for (int attempt = 0; attempt < 2; ++attempt)
        {
            const size_t mask = table->cap - 1;
            const size_t slot = index_hash(v) & mask;
            for (int probe = 0; probe < INDEX_MAX_PROBE; ++probe)
            {
                const size_t idx = (slot + probe) & mask;
                const T*     cur = table->slots[idx].load(std::memory_order_acquire);
                if (cur != nullptr)
                {
                    if (value_of_ptr(cur) == v)
                    {
                        return; // already indexed (equivalent pointer)
                    }
                    continue; // occupied by another value -> probe on
                }
                // Empty slot: publish (single writer under insert_mutex).
                table->slots[idx].store(ptr, std::memory_order_release);
                ++table->used;
                return;
            }
            // Budget exhausted at <= 50% load: pathological clustering. One
            // more doubling before giving up.
            if (attempt == 0)
            {
                table = index_grow();
            }
        }
        // Still exhausted: a registered value may now be unfindable by
        // index_lookup, so the cold path must re-enable its authoritative O(N)
        // segment scan. Sticky flag (never reset until index_clear). Said once:
        // from here on reclaim_second_values() is refused, so the identifiers
        // only grow, and the exhaustion abort alone would not name the cause.
        if (!index_degraded.exchange(true, std::memory_order_relaxed))
        {
            std::cerr << "[cst_val2] second-value index degraded (probe budget exhausted after a regrowth): "
                         "lookups fall back to the O(N) chain scan, and reclaim_second_values() is refused "
                         "until the last array of this type is destroyed -- the identifiers can only grow.\n";
        }
    }

    /**
     * @brief Register one more user of this type's registry, creating it if new.
     * @details The user is counted in #ref_count; the registry itself is made
     *          valid by install_head().
     */
    static void init()
    {
        // One more ArrayChunk2A<T> is using the shared list.
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
    static void install_head()
    {
        static const Sentinel<T> sentinel;

        region_init();

        // Install the initial segment, once.
        if (head.load(std::memory_order_relaxed) == nullptr)
        {
            auto        seg      = std::make_unique<Segment<T>>();
            Segment<T>* null_ptr = nullptr;
            if (head.compare_exchange_strong(null_ptr, seg.get(), std::memory_order_release, std::memory_order_relaxed))
            {
                seg.release(); // CAS won: seg is now owned by head
            }
            // If CAS lost: another thread installed a segment first — seg destroyed
        }
    }

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
     *          ArrayChunk2A<T> instances are alive. The last one out frees every
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
    static void clear_if_last(int& nb_val2)
    {
        const int remaining = ref_count.fetch_sub(1, std::memory_order_acq_rel) - 1;
        if (remaining > 0)
        {
            return; // Other ArrayChunk2A<T> instances still alive — do not free
        }

        // Last instance: free the segments and count what they held, for ONE
        // summary line. The slots themselves are region memory:
        // region_reset() below gives them back all at once.
        int         nb_segs = 0;
        Segment<T>* seg     = head.exchange(nullptr, std::memory_order_acq_rel);
        while (seg != nullptr)
        {
            ++nb_segs;
            const int used = seg->used;
            for (int i = 0; i < used; ++i)
            {
                if (seg->slots[i] != nullptr)
                {
                    ++nb_val2;
                }
            }
            auto owned = std::unique_ptr<Segment<T>>(seg);
            seg        = owned->next;
        }

        if (nb_segs > 1)
        {
            std::cout << "[cst_val2] " << nb_val2 << " distinct second values across " << nb_segs
                      << " segments (capacity " << nb_segs * Segment<T>::SIZE
                      << "), index_degraded=" << index_degraded.load(std::memory_order_relaxed) << "\n";
        }

        // Registry gone: drop the index so no stale pointer survives.
        index_clear();
        region_reset(); // the slots' pages, all at once

        // Reinstall an empty segment for any later pass. NOT init(): that
        // takes a reference, and one taken here, with no array alive, would
        // keep #ref_count above 0 for good.
        install_head();
    }

    /**
     * @brief Free everything unconditionally, ignoring #ref_count.
     * @warning Teardown only -- the Sentinel calls it at static destruction. It
     *          does not reinstall an empty segment, so #head is left null and
     *          the registry is unusable afterwards.
     */
    static void cleanup_all()
    {
        const Segment<T>* seg = head.exchange(nullptr, std::memory_order_relaxed);
        while (seg != nullptr)
        {
            // The slots are region memory, freed by region_reset() below.
            const Segment<T>* next = seg->next;
            delete seg;
            seg = next;
        }
        index_clear(); // registry gone: drop stale advisory index entries
        region_reset();
    }

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

    static inline std::atomic<char*> region_begin{nullptr}; ///< Reservation start; null before init().
    /// @brief Bytes per slot: every slot is carved at a multiple of it.
    /// @note constexpr, not set by region_init(): a pointer <-> identifier
    ///       conversion (id_of_ptr, slot_id, on Chunk2A's constant-2 reads)
    ///       is then a shift by 6, not a 64-bit `div` by a global.
    static constexpr std::size_t region_stride = kIdStride;

    /// @brief The value registered under each identifier, contiguous: what
    ///        a chunk's second value is read through on every write of a
    ///        wide type (SecondValueRef::get). 8 bytes per value, L1-resident
    ///        for the values a run actually uses. Zero-initialised static
    ///        storage: the pages of identifiers never handed out are never
    ///        touched. Index 0 is "no second value" and stays T{}.
    static inline std::array<T, kMaxIds + 1> value_by_id{};

  private:
    /// @brief What is reserved and carved: exactly the identifiers
    ///        value_by_id can hold. 256 MiB per type with the defaults.
    static constexpr std::size_t kRegionUsable =
        (kMaxIds * kIdStride) < kRegionBytes ? (kMaxIds * kIdStride) : kRegionBytes;

#ifndef CHUNK_CST_VAL2_INDEX_LOG2
    static constexpr int chunk_cst_val2_index_loc = 18;
    // 2^18 = 262144 slots, 2 Mio/type; <11% load at 30k values
#else
    static constexpr int chunk_cst_val2_index_loc = CHUNK_CST_VAL2_INDEX_LOG2;
#endif

    static constexpr size_t INDEX_CAP       = size_t{1} << chunk_cst_val2_index_loc; ///< Initial capacity.
    static constexpr int    INDEX_MAX_PROBE = 64; ///< Bounded; giving up means fallback.

    /// @brief Reserve the region. Idempotent; called by init().
    static void region_init()
    {
        if (region_begin.load(std::memory_order_acquire) != nullptr)
        {
            return;
        }
        // MM_PROTECT: PROT_NONE for good -- nothing is ever stored in the
        // region, so any dereference of a second-value pointer faults instead
        // of reading a plausible zero. Otherwise RW (never touched either).
        const int prot = build_config::use_mm_protect ? PROT_NONE : (PROT_READ | PROT_WRITE);
        void*     raw  = mmap(nullptr, kRegionUsable, prot, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
        if (raw == MAP_FAILED) [[unlikely]]
        {
            CHUNK_INVARIANT_FAILED("[cst_val2] cannot reserve the second-value region ({} MiB): {}",
                                   kRegionUsable >> 20U, std::strerror(errno));
        }
        // CHUNK_THP (experimental, off by default): advice only. Not under
        // MM_PROTECT, where the region stays PROT_NONE.
        if constexpr (!build_config::use_mm_protect)
        {
            if (cst2_thp_advise())
            {
                (void)madvise(raw, kRegionUsable, MADV_HUGEPAGE);
            }
        }
        // Every slot is carved at a multiple of region_stride (kIdStride),
        // which is what turns a slot pointer into a small identifier and back
        // (TabCstVal2Holder::id_of_ptr / value_of_id). Nothing is ever stored
        // in the region, so a slot only has to give each identifier a distinct
        // address: one cache line does, and no page is ever backed.
        if (char* expected = nullptr;
            region_begin.compare_exchange_strong(expected, static_cast<char*>(raw), std::memory_order_acq_rel))
        {
            region_end           = static_cast<char*>(raw) + kRegionUsable;
            const auto lim        = env_number<std::size_t>("CHUNK_CST_VAL2_ID_LIMIT", kMaxIds);
            id_limit              = (lim >= 1 && lim < kMaxIds) ? lim : kMaxIds; // lowers only
        }
        else
        {
            munmap(raw, kRegionUsable); // lost the race: the winner's region stands
        }
    }

    /// @brief Give the carved pages back and start over. Last clear only.
    static void region_reset() noexcept
    {
        free_ids.clear(); // the identifiers it lists go with the pages
        char* base = region_begin.load(std::memory_order_relaxed);
        if (base == nullptr)
        {
            return;
        }
        const std::size_t used = region_used.exchange(0, std::memory_order_relaxed);
        if (used == 0)
        {
            return;
        }
        if constexpr (build_config::use_mm_protect)
        {
            mprotect(base, used, PROT_READ | PROT_WRITE); // madvise needs a writable mapping to drop
        }
        madvise(base, used, MADV_DONTNEED);
        if constexpr (build_config::use_mm_protect)
        {
            mprotect(base, used, PROT_NONE);
        }
    }

    /// @brief Value hash (a 64-bit finaliser over the object's bytes).
    /// @note Hashes the REPRESENTATION, so two values that compare equal but
    ///       differ bitwise would land in different slots.
    static size_t index_hash(in_t<T> v) noexcept
    {
        uint64_t bits = 0;
        std::memcpy(&bits, &v, sizeof(T) < sizeof(uint64_t) ? sizeof(T) : sizeof(uint64_t));
        bits ^= bits >> 33U;
        bits *= 0xff51afd7ed558ccdULL;
        bits ^= bits >> 33U;
        bits *= 0xc4ceb9fe1a85ec53ULL;
        bits ^= bits >> 33U;
        return static_cast<size_t>(bits);
    }

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
    static IndexTable* index_grow() noexcept
    {
        const IndexTable* old = index_tab.load(std::memory_order_relaxed);
        const std::size_t cap = old == nullptr ? INDEX_CAP : old->cap * 2;
        auto              nt  = std::make_unique<IndexTable>(cap);
        if (old != nullptr)
        {
            const std::size_t mask = cap - 1;
            for (std::size_t i = 0; i < old->cap; ++i)
            {
                T* entry = old->slots[i].load(std::memory_order_relaxed);
                if (entry == nullptr)
                {
                    continue;
                }
                std::size_t idx = index_hash(value_of_ptr(entry)) & mask;
                while (nt->slots[idx].load(std::memory_order_relaxed) != nullptr)
                {
                    idx = (idx + 1) & mask;
                }
                nt->slots[idx].store(entry, std::memory_order_relaxed);
            }
            nt->used = old->used;
        }
        IndexTable* raw = nt.get();
        index_generations.push_back(std::move(nt));
        index_tab.store(raw, std::memory_order_release);
        return raw;
    }

    /**
     * @brief Empty the index and drop the cached tail.
     * @warning Single-threaded clear paths only, and at the very point the
     *          slots are given back -- an entry outliving its slot is a
     *          dangling pointer a lock-free reader will follow.
     */
    static void index_clear() noexcept
    {
        index_degraded.store(false, std::memory_order_relaxed);
        tail = nullptr; // drop cached tail; next cold-path append re-seeds from head
        // Single-threaded by contract, so this is the one point where the
        // superseded generations can go: no reader holds any of them. The
        // live table goes too; the next publish starts again at INDEX_CAP.
        index_tab.store(nullptr, std::memory_order_release);
        index_generations.clear();
    }

    /// @brief Live ArrayChunk2A<T> instances. The registry is freed when it
    ///        reaches zero, not before. @see clear_if_last
    static inline std::atomic<int> ref_count{0};

    static inline char*                    region_end = nullptr; ///< One past the reservation.
    static inline std::atomic<std::size_t> region_used{0};       ///< Bump offset, in bytes.

    /// @brief The identifiers region_alloc() hands out: kMaxIds, or fewer
    ///        under CHUNK_CST_VAL2_ID_LIMIT (read once in region_init(); it
    ///        can only lower the bound -- the way a test reaches it).
    static inline std::size_t id_limit = kMaxIds;

    /// @brief Identifiers freed by reclaim_unmarked(), handed out again by
    ///        take_free_slot() before the region's bump pointer moves.
    ///        Emptied with the region (region_reset()). Immortal, like
    ///        index_generations: a global array registers values during
    ///        static teardown too.
    static inline constinit internal::Immortal<std::vector<std::uint32_t>> free_ids_storage_{};
    static inline std::vector<std::uint32_t>&                              free_ids = free_ids_storage_.value;

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
    using global_vec_holder = GlobalVecHolder<T>;

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
    [[nodiscard]] T* find_or_set_ind_cst_val2(T second_v, SecondValueRef<T> ptr_second_value)
    {
        // Up to the record in the second-value table, which the reclaim's mark
        // scans: a reclaim must not run in between (LookupGuard).
        const typename global_vec_holder::LookupGuard in_flight;
        T* buf = find_or_register(second_v);
        ptr_second_value.set_from_array(buf);
        return buf;
    }

    /// @brief Is @p ptr one of this type's second-value slots?
    /// @note `is_cst_val2()` of Chunk2A: one relaxed load of the region base
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
    [[nodiscard]] static T* find_cst_val2_ptr(in_t<T> second_v)
    {
        using gvh = GlobalVecHolder<T>;
        const typename gvh::LookupGuard in_flight; // holds an index generation / the chain
        if (T* hit = gvh::index_lookup(second_v)) [[likely]]
        {
            return hit;
        }
        if (!gvh::index_degraded.load(std::memory_order_relaxed))
        {
            return nullptr;
        }
        const std::scoped_lock lk(gvh::insert_mutex);
        for (const Segment<T>* seg = gvh::head.load(std::memory_order_acquire); seg != nullptr; seg = seg->next)
        {
            for (int i = 0; i < seg->used; ++i)
            {
                if (gvh::value_of_ptr(seg->slots[i]) == second_v)
                {
                    return seg->slots[i];
                }
            }
        }
        return nullptr;
    }

    // ── Lifecycle ─────────────────────────────────────────────────────────

    /// @brief Register a user of this type's registry. Idempotent.
    /// @warning Must run before any compression pass on this type; it is what
    ///          guarantees GlobalVecHolder::head is non-null.
    static void create()
    {
        global_vec_holder::init();
    }

    // ── Identifiers: a slot's index in the region, 1-based; 0 = none ──────
    // The per-chunk second-value table of a wide type holds these instead of
    // values (SecondValueTraits). id_of_ptr() is arithmetic on the region
    // base and stride; value_of_id() reads value_by_id[].

    /// @brief Identifier of the slot @p ptr (must lie in the region).
    [[nodiscard]] static std::uint32_t id_of_ptr(const T* ptr) noexcept
    {
        const char* base = global_vec_holder::region_begin.load(std::memory_order_relaxed);
        return static_cast<std::uint32_t>((reinterpret_cast<const char*>(ptr) - base) /
                                          global_vec_holder::region_stride) +
               1u;
    }
    /**
     * @brief value_of_ptr() with the identifier range checked.
     *
     * @details Same conversion, but returns T{} rather than indexing out of
     *          bounds if @p ptr is not where the caller thought. A pointer that
     *          passes is_cst_val2_ptr() can only yield an id in [1, kMaxIds],
     *          so this never fires in a correct call -- it guards a pointer
     *          that was not range-tested, where a compare is cheaper than a
     *          crash.
     */
    [[nodiscard]] static T value_of_ptr_checked(const T* ptr) noexcept
    {
        if (!is_cst_val2_ptr(ptr))
        {
            return T{};
        }
        const std::uint32_t id = id_of_ptr(ptr);
        if (id == 0u || id > global_vec_holder::kMaxIds)
        {
            return T{};
        }
        return global_vec_holder::value_by_id[id];
    }

    /// @brief The value registered under @p id; T{} for 0.
    [[nodiscard]] static T value_of_id(std::uint32_t id) noexcept
    {
        return global_vec_holder::value_by_id[id]; // [0] is T{}
    }

    /**
     * @brief Drop this user of the registry; the last one frees it.
     * @param nb_val2 Incremented by the number of registered values freed.
     * @warning Single thread only, and with no concurrent scan in flight.
     *          @see GlobalVecHolder::clear_if_last for the full contract.
     */
    static void clear(int& nb_val2)
    {
        global_vec_holder::clear_if_last(nb_val2);
    }

    /// @brief The identifier for @p v, registering its slot if it is new;
    ///        0 for T{} (the "no second value" the value table wrote as 0).
    [[nodiscard]] static std::uint32_t id_for_value(in_t<T> v)
    {
        if (v == T{})
        {
            return 0;
        }
        return id_of_ptr(find_or_register(v));
    }

  private:
    using segment_t = Segment<T>;

    /**
     * @brief Carve one identifier slot for @p val and record the value.
     * @return The slot pointer: never dereferenced, only range-tested and
     *         converted to an identifier.
     * @warning value_by_id[] is written BEFORE the caller publishes the
     *          pointer. That ordering is what index_publish()'s release store
     *          makes visible.
     */
    static T* alloc_chunk(T val)
    {
        // A reclaimed identifier first; the region only when none is free.
        T* buf = global_vec_holder::take_free_slot();
        if (buf == nullptr)
        {
            buf = reinterpret_cast<T*>(global_vec_holder::region_alloc(global_vec_holder::region_stride));
        }
        // Before the slot can be published: a reader gets the identifier only from a
        // slot written after this returns.
        global_vec_holder::value_by_id[id_of_ptr(buf)] = val;
        return buf;
    }

    /**
     * @brief The slot for @p second_v, registering it if new.
     * @return The value's slot (an identifier, never dereferenced). Never null.
     */
    [[nodiscard]] static T* find_or_register(in_t<T> second_v)
    {
        const typename global_vec_holder::LookupGuard in_flight; // check builds; nests in the callers'
        // ── Hot path, lock-free ───────────────────────────────────────────
        if (T* hit = global_vec_holder::index_lookup(second_v)) [[likely]]
        {
            return hit;
        }

        // ── Cold path, under the insert lock ──────────────────────────────
        const std::scoped_lock lk(global_vec_holder::insert_mutex);

        // Double-check: another thread may have inserted second_v since our miss. The
        // index is single-writer under this lock, so it is authoritative here.
        if (T* hit = global_vec_holder::index_lookup(second_v)) [[likely]]
        {
            return hit;
        }

        // Only if the index ever overflowed its probe budget (index_degraded) can a
        // registered value be unindexed: then, and only then, scan the chain to avoid a duplicate.
        const bool degraded = global_vec_holder::index_degraded.load(std::memory_order_relaxed);

        segment_t* last_seg = global_vec_holder::tail;
        if (last_seg == nullptr) [[unlikely]]
        {
            // First append after a clear: re-seed the cached tail from head.
            last_seg = global_vec_holder::head.load(std::memory_order_acquire);
            while (last_seg->next != nullptr)
            {
                last_seg = last_seg->next;
            }
            global_vec_holder::tail = last_seg;
        }

        if (degraded) [[unlikely]]
        {
            for (const segment_t* seg = global_vec_holder::head.load(std::memory_order_acquire); seg != nullptr;
                 seg                  = seg->next)
            {
                for (int i = 0; i < seg->used; ++i)
                {
                    if (global_vec_holder::value_of_ptr(seg->slots[i]) == second_v)
                    {
                        global_vec_holder::index_publish(second_v, seg->slots[i]);
                        return seg->slots[i];
                    }
                }
            }
        }

        // Not present: append. The list is append-only between clears, so the
        // first free slot is last_seg->used (or slot 0 of a fresh segment).
        // create() guarantees head is never nullptr here.
        segment_t* seg = last_seg;
        if (seg->used == segment_t::SIZE)
        {
            seg = global_vec_holder::append_segment(last_seg);
        }

        T* buf                = alloc_chunk(second_v); // value_by_id[] BEFORE publication
        seg->slots[seg->used] = buf;
        ++seg->used;
        global_vec_holder::index_publish(second_v, buf); // release: readers see the value
        return buf;
    }
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
