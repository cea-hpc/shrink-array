
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

/**
 * @file tab_cst_val2_holder.h
 * @brief The registry of shared arrays behind "second constant value" compression.
 *
 * A Chunk2A compressed to a constant other than zero does not own a buffer: it
 * points at a shared, read-only array filled with that value. One array exists
 * per DISTINCT second value, for the whole process and for the life of the run,
 * so a thousand chunks holding the same constant cost one array between them.
 *
 * Three pieces:
 *  - Segment is a block of 64 slots. The registry is a chain of them rather
 *    than a growable array, so registering a new value never invalidates a
 *    pointer another thread is holding.
 *  - GlobalVecHolder owns that chain, one instance per element type, plus the
 *    hash index that makes a lookup O(1) and the lock that serialises appends.
 *  - TabCstVal2Holder is the per-array handle: it caches the pointer its
 *    chunks currently use and owns the allocation strategy.
 *
 * @par Two paths, and only one of them takes a lock
 * Registering a value that is ALREADY known -- every type-2 chunk after the
 * first one per value -- is a lock-free probe of GlobalVecHolder::index_.
 * Only a genuinely new value, or an index miss, takes
 * GlobalVecHolder::insert_mutex, under which the whole find-or-append is
 * plain sequential code.
 *
 * @warning One ordering is load-bearing and not checked at run time: an
 *          array must be filled before its pointer is published
 *          (index_publish's release store). A chunk adopts an array by
 *          storing it as its own values pointer, nothing else -- there is no
 *          per-chunk cache to keep in step since the region test
 *          (GlobalVecHolder::region_contains) replaced it.
 *
 * @warning Tearing the registry down requires that NO thread is scanning it.
 *          @see GlobalVecHolder::clear_if_last, which tells the story of the
 *          missing barrier that made this a SIGSEGV.
 */

#include <sys/mman.h>
#include <cstring>
#include <cstdint>
#include <cstdlib> // std::abort
#include <unistd.h>
#include <atomic>
#include <array>
#include <memory>
#include <mutex>
#include <type_traits>
#include <vector> // index_generations
// <iostream> is included directly: it used to come through param_chunk.h,
// whose include created a heavy circular dependency (ParamChunk ->
// AllTypeChunk -> ... -> this header) while nothing here uses a ParamChunk
// symbol.
#include <iostream>

#include "chunk/internal/chunk_size_policy.h"
#include "chunk/internal/env_parse.h" // env_number (CHUNK_CST_VAL2_ID_LIMIT)
#include "chunk/internal/in_t.h" // in_t<T>: the value parameters
#include "chunk/config.h"
#ifdef FAST_ALLOC
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
#ifdef FAST_ALLOC
    return alloc_config::thp_advise();
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
 * @note The chain is append-only between clears, so occupancy is contiguous:
 *       `slots[0 .. used)` are set and `slots[used .. SIZE)` are null. Several
 *       loops here rely on that rather than null-checking every slot.
 */
template <typename T>
struct Segment
{
    static constexpr int SIZE = 64; ///< Slots per segment.

    std::array<T*, SIZE> slots{};        ///< The shared arrays, one per registered value.
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
 *          IDENTIFIER of the shared array registered for the value (its
 *          index in the type's region, 1-based; 0 = no second value) rather
 *          than the value: 4 bytes per chunk instead of 8 (double, long) or
 *          16 (complex<double>) -- ~150 MB on the 50 M-chunk reference run.
 *          Reading it back is one load of the array's first element, which
 *          is the value (the arrays are filled with it), at a fixed stride
 *          from the region base. Narrower types keep the value itself: an
 *          identifier would not be smaller, and bool has no registry.
 */
//
// OPT-IN (-DCHUNK_SECOND_VALUE_INDEXED, CMake option of the same name). The
// identifier is read back through a dictionary on every write of a
// materialised chunk, and that one extra dependent load is +3% CPU on the
// perf bench at 6 x86 cores (handle_no_bool_assignment, mark_for_compression)
// for -2% RSS there, -3.8% on the 50 M-chunk reference run. The default
// keeps the value in the slot; the switch is for a use that is bound by
// memory rather than by write throughput.
template <typename T>
struct SecondValueTraits
{
    static constexpr bool indexed = build_config::second_value_indexed && sizeof(T) > 4;
    using slot_t                  = std::conditional_t<indexed, std::uint32_t, T>;
};

/**
 * @brief Handle on one chunk's second-value slot.
 *
 * @details Replaces the `T*` into the per-chunk table that the proxies, the
 *          compressor and the clear paths used to pass around: one pointer,
 *          but a get() / set() surface, so the slot can be an identifier
 *          (SecondValueTraits). A default-constructed handle is "none", as
 *          the null pointer was.
 */
template <typename T>
struct SecondValueRef
{
    using slot_t = SecondValueTraits<T>::slot_t;

    /// @brief The chunk's second value; T{} (== first_value, 0) when none.
    [[nodiscard]] T get() const noexcept
    {
        if constexpr (SecondValueTraits<T>::indexed)
            return TabCstVal2Holder<T>::value_of_id(*slot);
        else
            return *slot;
    }
    /// @brief Record @p v as the chunk's second value. For an indexed type this
    ///        registers the value's shared array if it is new (find_or_set).
    void set(T v) const
    {
        if constexpr (SecondValueTraits<T>::indexed)
            *slot = TabCstVal2Holder<T>::id_for_value(v);
        else
            *slot = v;
    }
    /// @brief Record the value of a shared array the caller already has.
    void set_from_array(const T* src) const noexcept
    {
        if constexpr (SecondValueTraits<T>::indexed)
            *slot = TabCstVal2Holder<T>::id_of_ptr(src);
        else
            // NOT *p: the shared array of a second value is never written, so
            // this read would record 0 and the chunk would be
            // marked constant at the wrong value -- silently, since everything
            // downstream then agrees with the slot.
            *slot = GlobalVecHolder<T>::value_of_ptr(src);
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
    /// @brief Call GlobalVecHolder<T>::cleanup_all().
    ~Sentinel()
    {
        GlobalVecHolder<T>::cleanup_all();
    }
};

/**
 * @brief The process-wide registry of second-constant arrays for one type.
 *
 * @tparam T Element type. One instance per type, all members static.
 *
 * @details Owns the Segment chain, the hash index over it, and the lock that
 *          serialises appends. Lives for the whole run: arrays registered here
 *          are shared by every Chunk2A that compressed to that value, so they
 *          cannot be freed while any of them is alive. #ref_count tracks that.
 */
template <typename T>
class GlobalVecHolder
{
  private:
    // ── The region every shared array of this type is carved from ─────────
    //
    // One contiguous reservation per T (virtual address space, MAP_NORESERVE:
    // pages are backed as arrays are carved, the rest costs nothing). Its
    // point is what it makes of "is this chunk compressed to a second
    // constant": a range test on the chunk's values pointer, and nothing
    // else. Before it, every Chunk2A<T> carried an 8-byte cached_ptr -- the
    // array it read from -- for `values == cached_ptr`; that was the
    // difference between a 16- and a 24-byte chunk header, 400 MB on the
    // 50 M-chunk reference run, and a store-ordering subtlety (the cache had
    // to be published after values, and a reader in between saw a constant
    // chunk as materialised). The registry is append-only between clears, so
    // a bump pointer is the whole allocator: nothing is freed per slot, and
    // region_reset() gives the pages back at the last clear.
    //
    // Nothing is ever STORED in the region: a slot is an identifier, not an
    // array. A chunk compressed to a second constant points at its slot,
    // is_cst_val2() is a range test on that pointer and id_of_ptr() is
    // arithmetic -- neither dereferences it -- and the value itself comes from
    // value_by_id[]. The read path pays one well-predicted range test
    // (ChunkBase::element_at) and a value costs 8 bytes.
    //
    // The representation this replaced gave each distinct value a shared
    // array FILLED with it, so `values[i]` returned the constant with no test
    // at all. It cost 32 KiB per value in dyn -- the 2 GiB region held 65 536
    // of them and a quantised field exhausted it and aborted -- and a fill of
    // cst_buffer_size elements per registration, with the page faults behind
    // it. Measured against it (chunk2a static 256, 2000 iterations, 64
    // threads, one seed): 17 503 G instructions against 18 990 G, -7.8%, IPC
    // 5.75 against 5.61, system time 199 s against 233 s. The filled arrays
    // were removed in Oct '26; compare against an earlier commit if a
    // regression is ever suspected there.
    //
    // WARNING: the failure mode of this code is a WRONG VALUE, not a crash: a
    // site that reads a value out of the slot, or loads `values` twice around
    // a test, returns something plausible. Re-run rechunk_validation_*, a
    // TSan pass and a --mode check run whenever it is touched.
    //
    // Capacity: ONE bound, kRegionUsable = min(CHUNK_CST_VAL2_REGION_BYTES,
    // kMaxIds x kIdStride) -- one cache line per identifier, kMaxIds capped
    // by CHUNK_CST_VAL2_MAX_IDS, which sizes value_by_id. The reservation,
    // the range test (region_end) and region_alloc()'s abort all use it.
    // Until Oct '26 the abort used the region's 33 M while value_by_id held
    // 4 M: every identifier in between was written past the table. Running
    // out aborts with the count, it is not a silent fallback -- a fallback
    // array would break the range test.
#ifndef CHUNK_CST_VAL2_REGION_BYTES
    #define CHUNK_CST_VAL2_REGION_BYTES (std::size_t{2} << 30U)
#endif

    static constexpr std::size_t kRegionBytes = CHUNK_CST_VAL2_REGION_BYTES;
    static constexpr std::size_t kIdStride = 64; ///< One cache line per identifier (see region_init()).

  public:
    // ── Value->pointer index (lock-free readers, single writer under lock) ──
    // Flat open-addressing table turning the O(N) dedup search in
    // find_or_set_ind_cst_val2 into an O(1) probe. Publications happen ONLY
    // under insert_mutex (single writer): the former "advisory, racy
    // last-wins" semantics are gone. A lookup hit is still revalidated
    // (p[0] == v) — required regardless of concurrency, because open
    // addressing probes across hash collisions. A miss (capacity / probe
    // budget) falls back to the authoritative under-lock scan. Entries hold
    // the same pointers as the list slots; must be cleared wherever those
    // arrays are freed (single-threaded clear paths) so a lookup never
    // dereferences a freed array.

    /**
     * @brief One generation of the open-addressed value -> shared array table.
     *
     * @details Turns what used to be an O(N) scan of the chain on EVERY
     *          registration -- which serialised the whole compression phase as
     *          the append-only registry grew -- into an O(1) probe. Entries
     *          hold the same pointers as the segment slots, so they MUST be
     *          cleared wherever those arrays are freed, or a lookup returns a
     *          dangling pointer. @see index_clear
     *
     *          The table used to be a fixed 2^18 array sized for ~30k values.
     *          shrink_bench's unsigned long array reached 214 300 distinct
     *          values (82% load): linear probing at that load exhausts the
     *          64-probe budget constantly, index_degraded flipped, and every
     *          new value fell back to the O(N) segment scan under the mutex --
     *          29 s at 2000 iterations, 113 s at 4000. The table now doubles
     *          under insert_mutex past 50% load (@see index_grow).
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
    /// @details Same stride as the carve in region_init(): an identifier is one
    ///          cache line, so the region holds 33 M of them whatever the
    ///          geometry -- against 65 536 in dyn when each was charged the
    ///          32 KiB of a filled array, which is the wall that aborted a run
    ///          and started this work.
    ///
    ///          Capped by CHUNK_CST_VAL2_MAX_IDS, because the binding cost is
    ///          then value_by_id[kMaxIds + 1]: at 33 M identifiers that is
    ///          256 MiB of address space per element type for a double. Its
    ///          pages are touched only for identifiers actually handed out, so
    ///          the resident cost follows use -- but reserving a quarter of a
    ///          gigabyte per type to serve a few thousand values is the wrong
    ///          shape. The default leaves four thousand times the headroom a
    ///          real workload was measured to need.
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
            count += static_cast<std::size_t>(s->used);
        return count;
    }

    /// @brief Is @p ptr the start of a shared array of this type?
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
        // Warn once on the way up. The capacity is not a design limit, it is a
        // division: kRegionBytes / stride. In dyn the stride is sized for
        // chunk_dyn_max_size rather than the geometry in use -- 32 KiB per
        // value for a double -- so the region holds 65 536 values, and a
        // quantised field can reach that without anyone intending it. A run
        // that is going to die deserves to say so while there is still room to
        // raise CHUNK_CST_VAL2_REGION_BYTES.
        // ONE capacity: the identifiers value_by_id can hold (kRegionUsable,
        // lowered by CHUNK_CST_VAL2_ID_LIMIT). alloc_chunk() writes
        // value_by_id[id] for every slot carved here, so a slot past it would
        // be a write past the table -- the bound used to be the region's
        // 33 M identifiers against value_by_id's 4 M.
        const std::size_t cap_bytes =
            (bytes != 0 && id_limit * bytes < kRegionUsable) ? id_limit * bytes : kRegionUsable;
        const char* limiting = (kMaxIds * kIdStride) < kRegionBytes ? "CHUNK_CST_VAL2_MAX_IDS" : "CHUNK_CST_VAL2_REGION_BYTES";
        if (id_limit < kMaxIds)
            limiting = "CHUNK_CST_VAL2_ID_LIMIT";
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
            std::cerr << "[cst_val2] second-value identifiers exhausted: " << (off / bytes) << " of "
                      << (cap_bytes / (bytes != 0 ? bytes : 1)) << ", bound by " << limiting
                      << " (CHUNK_CST_VAL2_MAX_IDS=" << kMaxIds << ", CHUNK_CST_VAL2_REGION_BYTES="
                      << (kRegionBytes >> 20U) << " MiB). Raise it, or reduce the number of DISTINCT second "
                      << "values the workload registers.\n";
            std::abort();
        }
        return region_begin.load(std::memory_order_relaxed) + off;
    }

    /**
     * @brief The value the shared array @p ptr stands for.
     *
     * @details The registry recovers a value from an array in five places --
     *          the two probe revalidations, the rehash, and the two degraded
     *          segment scans -- and each did it with `p[0]`. There is nothing
     *          at p[0]: the region is never written, so without MM_PROTECT it
     *          reads a zero page and
     *          every comparison against a non-zero value silently fails. No
     *          value is ever found again, every registration takes the lock,
     *          and the run stalls in find_or_register.
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
     * @return Its shared array, or null -- meaning "not found HERE", never
     *         "not registered". The caller must fall back.
     * @details A hit is revalidated with `p[0] == v` whatever the concurrency:
     *          open addressing probes across collisions, so the slot reached
     *          may belong to another value.
     */
    static T* index_lookup(in_t<T> v) noexcept
    {
        // One acquire load of the table pointer; the table it names is never
        // freed while a reader can hold it (see index_generations). A reader
        // on a superseded generation misses values added since -- and a miss
        // was never "not registered", only "not found here".
        const IndexTable* table = index_tab.load(std::memory_order_acquire);
        if (table == nullptr)
            return nullptr;
        const size_t mask = table->cap - 1;
        const size_t slot = index_hash(v) & mask;
        for (int probe = 0; probe < INDEX_MAX_PROBE; ++probe)
        {
            T* entry = table->slots[(slot + probe) & mask].load(std::memory_order_acquire);
            if (entry == nullptr)
                return nullptr; // empty slot => v is not indexed here
            if (value_of_ptr(entry) == v)
                return entry; // validated hit
        }
        return nullptr; // probe budget exhausted => treat as miss
    }

    // Publish (v -> p). No-op if already present or probe budget exhausted.
    // Called ONLY under insert_mutex (single writer): a plain release store
    // suffices — it pairs with the acquire load in index_lookup so a lock-free
    // reader that sees the pointer also sees the value_by_id[] entry written
    // before publication.
    /**
     * @brief Record that @p v is served by @p ptr. No-op if already present.
     * @pre Called under #insert_mutex -- this is the single writer.
     * @details A plain release store suffices: it pairs with the acquire load
     *          in index_lookup(), so a lock-free reader that sees the pointer
     *          also sees the fill that populated the array before publication.
     * @note Giving up after #INDEX_MAX_PROBE sets #index_degraded rather than
     *       failing: the value IS registered in the chain, just not indexed.
     */
    static void index_publish(in_t<T> v, T* ptr) noexcept
    {
        IndexTable* table = index_tab.load(std::memory_order_relaxed);
        // Grow before the table passes half full: linear probing degrades
        // sharply beyond that, and this is where the old fixed table went
        // from O(1) to "every new value rescans the chain".
        if (table == nullptr || (table->used + 1) * 2 > table->cap)
            table = index_grow();
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
                        return; // already indexed (equivalent pointer)
                    continue;   // occupied by another value -> probe on
                }
                // Empty slot: publish (single writer under insert_mutex).
                table->slots[idx].store(ptr, std::memory_order_release);
                ++table->used;
                return;
            }
            // Budget exhausted at <= 50% load: pathological clustering. One
            // more doubling before giving up.
            if (attempt == 0)
                table = index_grow();
        }
        // Still exhausted: a registered value may now be unfindable by
        // index_lookup, so the cold path must re-enable its authoritative O(N)
        // segment scan. Sticky flag (never reset until index_clear).
        index_degraded.store(true, std::memory_order_relaxed);
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
     *          clear_if_last() can reinstall an empty chain without the
     *          phantom reference that init() used to leave behind: #ref_count
     *          went 0 -> 1 with no array alive, the next last-one-out found 1
     *          left, and the registry was never freed again before exit.
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
                seg.release(); // CAS won: seg is now owned by head
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
     * @param nb_val2 Incremented by the number of arrays actually freed.
     *
     * @details Decrements #ref_count and returns immediately while other
     *          ArrayChunk2A<T> instances are alive. The last one out frees every
     *          array and every segment, clears the index, then calls
     *          install_head() so #head is valid again for any later pass --
     *          without registering a user: #ref_count stays at 0, and the
     *          next generation of arrays is freed by its own last one out.
     *
     * @warning No thread may be SCANNING the chain while this runs --
     *          find_or_set_ind_cst_val2(), reached from make_block_compression
     *          on the write slow path. Call it serially, or from an
     *          `omp single` PRECEDED BY A BARRIER that ends every concurrent
     *          scan. At the call site in array_chunk2A_inline.h the preceding
     *          `for` was `nowait`, so a late writer's scan raced this teardown
     *          and dereferenced a freed Segment -- a SIGSEGV that only appeared
     *          at large CHUNK_SIZE.
     *
     * @note No assertion guards this: there is no reliable OpenMP predicate for
     *       "I am the lone thread of a single". omp_in_parallel() is the wrong
     *       test, since the single legitimately runs inside a parallel region.
     */
    static void clear_if_last(int& nb_val2)
    {
        // CONTRACT: this tears down the shared segment list (head.exchange(nullptr)
        // + delete of every Segment). No thread may be SCANNING the list
        // (find_or_set_ind_cst_val2, reached from make_block_compression on the
        // write slow path) while this runs. Callers must therefore invoke it
        // either serially OR from an `#pragma omp single` that is preceded by a
        // barrier ending every concurrent scan — see the call site in
        // array_chunk2A_inline.h, where a missing barrier before the single
        // (the preceding `for` is `nowait`) let a late writer's scan race this
        // teardown and dereference a freed Segment -> SIGSEGV at large
        // CHUNK_SIZE. (No assert here: there is no reliable OpenMP predicate for
        // "I am the lone thread of a single"; omp_in_parallel() is wrong because
        // the single legitimately runs inside a parallel region.)
        const int remaining = ref_count.fetch_sub(1, std::memory_order_acq_rel) - 1;
        if (remaining > 0)
            return; // Other ArrayChunk2A<T> instances still alive — do not free

        // Last instance: free the segments and count what they held, so we can
        // emit ONE summary line (replaces the per-append spam that used to print
        // inside find_or_set_ind_cst_val2). The slots themselves are region
        // memory: region_reset() below gives them back all at once.
        int         nb_segs = 0;
        Segment<T>* seg     = head.exchange(nullptr, std::memory_order_acq_rel);
        while (seg != nullptr)
        {
            ++nb_segs;
            const int used = seg->used;
            for (int i = 0; i < used; ++i)
            {
                if (seg->slots[i] != nullptr)
                    ++nb_val2;
            }
            auto owned = std::unique_ptr<Segment<T>>(seg);
            seg        = owned->next;
        }

        if (nb_segs > 1)
            std::cout << "[cst_val2] " << nb_val2 << " distinct second values across " << nb_segs
                      << " segments (capacity " << nb_segs * Segment<T>::SIZE
                      << "), index_degraded=" << index_degraded.load(std::memory_order_relaxed) << "\n";

        // Arrays are gone: drop the advisory index so no stale pointer survives.
        index_clear();
        region_reset(); // the arrays' pages, all at once

        // Reinstall an empty segment for any later pass. NOT init(): that
        // takes a reference, and one taken here, with no array alive, kept
        // #ref_count above 0 for the rest of the run -- after the first
        // extinction nothing was ever freed again.
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
        Segment<T>* seg = head.exchange(nullptr, std::memory_order_relaxed);
        while (seg != nullptr)
        {
            // The slots are region memory, freed by region_reset() below.
            Segment<T>* next = seg->next;
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
    // Cached tail segment: makes the cold-path append O(1) instead of walking
    // head->...->tail (O(N/64)) on every registration. Mutated only under
    // insert_mutex (appends) or single-threaded (init/clear); invalidated to
    // nullptr by index_clear so the next append re-seeds it from head.
    static inline Segment<T>* tail{nullptr};

    // ── Insert lock ──────────────────────────────────────────────────────
    // Serializes the COLD path only: inserting a genuinely new second value
    // (or servicing an index probe-budget miss). Every mutation of the segment
    // list and every index publication happens under this mutex. The HOT path
    // — find_or_set on a value already indexed, i.e. every type-2 chunk after
    // the first one per value — stays lock-free through index_lookup below.
    // A std::mutex (not a spinlock): the critical section may mmap/mprotect
    // in alloc_chunk under MM_PROTECT builds.
    /**
     * @brief Serialises the COLD path only: registering a genuinely new value.
     * @details Every mutation of the chain and every index publication happens
     *          under it, which is what makes the index a single-writer
     *          structure. The hot path never takes it.
     * @note A std::mutex rather than a spinlock on purpose: the critical
     *       section may mmap/mprotect under MM_PROTECT builds.
     */
    static inline std::mutex insert_mutex;

    // Set true the first (and only) time index_publish exhausts its probe budget,
    // i.e. a registered value could become unfindable by index_lookup. While it
    // stays false (the expected regime: N in the thousands, index at <1% load),
    // the cold path can trust index_lookup as authoritative and skip the O(N)
    // segment re-scan entirely. Only flipped under insert_mutex; read relaxed.
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
    static inline std::size_t region_stride = 0; ///< Bytes per array: every array is carved at a multiple of it.

    /// @brief The value registered under each identifier, contiguous: what
    ///        a chunk's second value is read through on every write of a
    ///        wide type (SecondValueRef::get). Reading the arrays' first
    ///        elements instead cost a scattered cache line per distinct
    ///        value (+3.6% CPU on the perf bench); this is 8 bytes per value,
    ///        L1-resident for the values a run actually uses. Zero-initialised
    ///        static storage: the pages of identifiers never handed out are
    ///        never touched. Index 0 is "no second value" and stays T{}.
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
            return;
        // MM_PROTECT: PROT_NONE for good -- nothing is ever stored in the
        // region, so any dereference of a second-value pointer faults instead
        // of reading a plausible zero. Otherwise RW (never touched either).
        const int prot = build_config::use_mm_protect ? PROT_NONE : (PROT_READ | PROT_WRITE);
        void*     raw  = mmap(nullptr, kRegionUsable, prot, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
        if (raw == MAP_FAILED) [[unlikely]]
        {
            std::cerr << "[cst_val2] cannot reserve the second-value region (" << (kRegionUsable >> 20U) << " MiB)\n";
            std::abort();
        }
        // CHUNK_THP (experimental, off by default): read by every thread that
        // writes a second value. Not under MM_PROTECT, whose per-array
        // mprotect splits huge pages anyway. Advice only.
        if constexpr (!build_config::use_mm_protect)
        {
            if (cst2_thp_advise())
                (void)madvise(raw, kRegionUsable, MADV_HUGEPAGE);
        }
        // Every slot is carved at a multiple of the stride, which is what turns
        // a slot pointer into a small identifier and back
        // (TabCstVal2Holder::id_of_ptr / value_of_id). Nothing is ever stored
        // in the region, so a slot only has to give each identifier a distinct
        // address: one cache line does, and no page is ever backed.
        const std::size_t stride = kIdStride;

        if (char* expected = nullptr;
            region_begin.compare_exchange_strong(expected, static_cast<char*>(raw), std::memory_order_acq_rel))
        {
            region_stride = stride;
            region_end    = static_cast<char*>(raw) + kRegionUsable;
            const std::size_t lim = env_number<std::size_t>("CHUNK_CST_VAL2_ID_LIMIT", kMaxIds);
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
        char* base = region_begin.load(std::memory_order_relaxed);
        if (base == nullptr)
            return;
        const std::size_t used = region_used.exchange(0, std::memory_order_relaxed);
        if (used == 0)
            return;
        if constexpr (build_config::use_mm_protect)
            mprotect(base, used, PROT_READ | PROT_WRITE); // madvise needs a writable mapping to drop
        madvise(base, used, MADV_DONTNEED);
        if constexpr (build_config::use_mm_protect)
            mprotect(base, used, PROT_NONE);
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
     *          array's first element, which is its value; the arrays outlive
     *          the index by the same contract that lets the index hold them.
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
                    continue;
                std::size_t idx = index_hash(value_of_ptr(entry)) & mask;
                while (nt->slots[idx].load(std::memory_order_relaxed) != nullptr)
                    idx = (idx + 1) & mask;
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
     *          backing arrays are freed -- an entry outliving its array is a
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

    /// @brief The live table. Null until the first publish; readers that see
    ///        null miss and take the cold path, which is always correct.
    static inline std::atomic<IndexTable*> index_tab{nullptr};
    /// @brief Every table ever published, the live one last. Old generations
    ///        are kept until index_clear(): a lock-free reader may still hold
    ///        their pointer, and index_clear() is the one point where no
    ///        reader is in flight (its contract). They sum to less than the
    ///        live table, so this is bounded by the same factor of two.
    static inline std::vector<std::unique_ptr<IndexTable>> index_generations;
};

/**
 * @brief The shared second-value arrays, as a chunk sees them.
 *
 * @tparam T Element type.
 *
 * @details Stateless: everything is static and delegates to
 *          GlobalVecHolder. This type is the surface the compression path
 *          talks to, and the place the allocation strategy lives. It used to
 *          carry one pointer per chunk (the array the chunk read from, for
 *          `values == cached_ptr`); the arrays now live in one region per T
 *          and "compressed to a second constant" is a range test on the
 *          chunk's values pointer -- 8 bytes off every chunk header.
 *
 * @par Invariants the rest of the code relies on
 *  - GlobalVecHolder::head is never null after create() or clear(): clear()
 *    reinstalls an empty segment, so a later compression pass finds a valid
 *    head without testing.
 *  - Every array handed out by find_or_set_ind_cst_val2() lies in the region,
 *    and nothing else does: is_cst_val2_ptr() is exact.
 *
 * @see TabCstVal2Holder<bool>, which has none of this machinery.
 */
template <typename T>
class TabCstVal2Holder
{
  public:
    using global_vec_holder = GlobalVecHolder<T>;

    // No instance part: the holder is stateless. Whether a chunk reads from
    // a shared array is a range test on its values pointer
    // (is_cst_val2_ptr), and the array for a given second value is found in
    // the registry (find_cst_val2_ptr) -- see GlobalVecHolder::region_*.

    /**
     * @brief The shared array for @p second_v, registering it if new.
     *
     * @param second_v The constant to compress to.
     * @param ptr_second_value Out-parameter: the caller's own copy of the value,
     *        written before anything else.
     * @return The value's slot (an identifier, never dereferenced). Never null.
     *
     * @note The chunk adopts the array by storing it as its values pointer
     *       (the compressor's CAS); nothing else is recorded per chunk.
     *
     * @details Hot path -- @p second_v already registered, which is every type-2
     *          chunk after the first one per value -- is a lock-free index
     *          probe and nothing else. Cold path takes
     *          GlobalVecHolder::insert_mutex and is then plain sequential code:
     *          no speculative allocate-then-discard, no slot CAS, no
     *          segment-append CAS, no recursive retry with a resume index.
     *
     * @note The O(N) chain scan survives only behind
     *       GlobalVecHolder::index_degraded, i.e. the case where the index can
     *       no longer answer authoritatively.
     */
    [[nodiscard]] T* find_or_set_ind_cst_val2(T second_v, SecondValueRef<T> ptr_second_value)
    {
        T* buf = find_or_register(second_v);
        ptr_second_value.set_from_array(buf);
        return buf;
    }

    /// @brief Is @p ptr one of this type's shared second-value arrays?
    /// @note `is_cst_val2()` of Chunk2A: one relaxed load of the region base
    ///       and two compares, no per-chunk state.
    [[nodiscard]] static bool is_cst_val2_ptr(const T* ptr) noexcept
    {
        return GlobalVecHolder<T>::region_contains(ptr);
    }

    /// @brief The shared array registered for @p second_v, or null if none.
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
        if (T* hit = gvh::index_lookup(second_v)) [[likely]]
            return hit;
        if (!gvh::index_degraded.load(std::memory_order_relaxed))
            return nullptr;
        const std::scoped_lock lk(gvh::insert_mutex);
        for (const Segment<T>* seg = gvh::head.load(std::memory_order_acquire); seg != nullptr; seg = seg->next)
        {
            for (int i = 0; i < seg->used; ++i)
            {
                if (gvh::value_of_ptr(seg->slots[i]) == second_v)
                    return seg->slots[i];
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

    // ── Identifiers: an array's index in the region, 1-based; 0 = none ────
    // The per-chunk second-value table of a wide type holds these instead of
    // values (SecondValueTraits). Both directions are arithmetic on the
    // region base and stride; value_of_id() reads the array's first element,
    // which is the value the array was filled with.

    /// @brief Identifier of the shared array @p ptr (must lie in the region).
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
     *          so this never fires in a correct call -- it is here because the
     *          write path reached it once with a pointer that had not been
     *          range-tested, and a compare is cheaper than the crash.
     */
    [[nodiscard]] static T value_of_ptr_checked(const T* ptr) noexcept
    {
        if (!is_cst_val2_ptr(ptr))
            return T{};
        const std::uint32_t id = id_of_ptr(ptr);
        if (id == 0u || id > global_vec_holder::kMaxIds)
            return T{};
        return global_vec_holder::value_by_id[id];
    }

    /// @brief The value registered under @p id; T{} for 0.
    [[nodiscard]] static T value_of_id(std::uint32_t id) noexcept
    {
        return global_vec_holder::value_by_id[id]; // [0] is T{}
    }

    /**
     * @brief Drop this user of the registry; the last one frees it.
     * @param nb_val2 Incremented by the number of arrays freed.
     * @warning Single thread only, and with no concurrent scan in flight.
     *          @see GlobalVecHolder::clear_if_last for the full contract and
     *          the bug that established it.
     */
    static void clear(int& nb_val2)
    { // Delegate to GlobalVecHolder which tracks how many instances
        // are still alive and frees the shared segment list only when
        // the last one calls clear().
        global_vec_holder::clear_if_last(nb_val2);
    }

    /// @brief The identifier for @p v, registering its array if it is new;
    ///        0 for T{} (the "no second value" the value table wrote as 0).
    [[nodiscard]] static std::uint32_t id_for_value(in_t<T> v)
    {
        if (v == T{})
            return 0;
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
        T* buf = reinterpret_cast<T*>(global_vec_holder::region_alloc(global_vec_holder::region_stride));
        // The dictionary entry, before the array can be published: a reader
        // that obtains the identifier obtains it from a slot written after
        // this returns.
        global_vec_holder::value_by_id[id_of_ptr(buf)] = val;
        return buf;
    }

    /**
     * @brief The slot for @p second_v, registering it if new.
     * @return The value's slot (an identifier, never dereferenced). Never null.
     */
    [[nodiscard]] static T* find_or_register(in_t<T> second_v)
    {
        // ── Hot path, lock-free ───────────────────────────────────────────
        if (T* hit = global_vec_holder::index_lookup(second_v)) [[likely]]
            return hit;

        // ── Cold path, under the insert lock ──────────────────────────────
        const std::scoped_lock lk(global_vec_holder::insert_mutex);

        // Double-check, O(1): another thread may have inserted second_v between
        // our lock-free miss and acquiring the lock. Publications are single-writer
        // under this lock, so the index is authoritative — this replaces the old
        // O(N) segment slot-scan that ran on EVERY registration and serialised the
        // whole compression phase as the append-only registry grew.
        if (T* hit = global_vec_holder::index_lookup(second_v)) [[likely]]
            return hit;

        // Walk is now O(1): the cached tail points at the last segment. Only when
        // the index has ever overflowed its probe budget (index_degraded) can a
        // registered value be unindexed — in that (unexpected) case, and only
        // then, pay the authoritative O(N) slot-scan to avoid a duplicate.
        const bool degraded = global_vec_holder::index_degraded.load(std::memory_order_relaxed);

        segment_t* last_seg = global_vec_holder::tail;
        if (last_seg == nullptr) [[unlikely]]
        {
            // First append after a clear: re-seed the cached tail from head.
            last_seg = global_vec_holder::head.load(std::memory_order_acquire);
            while (last_seg->next != nullptr)
                last_seg = last_seg->next;
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
            seg = global_vec_holder::append_segment(last_seg);

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
  private:
    /// @brief All `true`, evaluated at compile time so #m_ptr is constant-initialised.
    [[nodiscard]] static constexpr std::array<bool, cst_buffer_size> all_true() noexcept
    {
        std::array<bool, cst_buffer_size> arr{};
        arr.fill(true);
        return arr;
    }

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

    /// @brief The one shared array, all `true`.
    static inline std::array<bool, cst_buffer_size> m_ptr = all_true();
};

} // namespace shrink_array
