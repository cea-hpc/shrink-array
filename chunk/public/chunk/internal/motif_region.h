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
 * @file motif_region.h
 * @brief pattern::MotifRegion: the motifs of pattern::Array<T>, in one reserved
 *        virtual range (docs/pattern_design.md, sections 4 and 7).
 */

#include <sys/mman.h> // mmap, MAP_NORESERVE

#include <array>       // std::array (the probation table, the segments)
#include <atomic>      // std::atomic (the range bounds, the index, the probation counters)
#include <cstddef>     // std::byte, std::size_t
#include <cstdint>     // std::uint8_t, std::uint16_t, std::uint32_t, std::uint64_t, std::uintptr_t
#include <cstring>     // std::memcmp, std::memcpy
#include <memory>      // std::unique_ptr (the index generations)
#include <mutex>       // std::mutex, std::scoped_lock (the registrations, the reclaim)
#include <type_traits> // std::make_unsigned_t (the hash)
#include <vector>      // std::vector (free identifiers, live flags, entries, size classes, generations)

#include "fast_allocators/internal/raw_memory.h" // address_bits
#include "chunk/internal/immortal.h"             // Immortal: statics safe for global arrays
#include "chunk/internal/invariant_check.h"      // CHUNK_INVARIANT_FAILED

namespace shrink_array
{

namespace pattern
{

/**
 * @brief What MotifRegion<T> holds and did, for the application's checkpoints:
 *        relaxed reads, exact at a sequential point.
 */
struct MotifStats
{
    std::size_t entries    = 0; ///< Motifs registered now.
    std::size_t registered = 0; ///< Registrations since the start (reused identifiers included).
    std::size_t first_seen = 0; ///< Admissions refused: a motif's first sighting (probation).
    std::size_t reclaims   = 0; ///< reclaim_unmarked() calls that ran.
    std::size_t freed      = 0; ///< Motifs those calls freed.
    std::size_t full       = 0; ///< Registrations refused: no segment left for the entry's size.
    std::size_t sizes      = 0; ///< Entry sizes (chunk sizes) the region holds segments for.
};

/**
 * @brief The motifs of pattern::Array<T>: chunk-sized arrays of T, normalised
 *        (motif[0] = 0), in ONE reserved virtual range, so that "this chunk's
 *        values pointer is a motif" is a range test on the pointer.
 *
 * @details An entry is written before any chunk points at it and is not written
 *          again while one does. Registered motifs are found by a 64-bit hash of
 *          the normalised motif, CONFIRMED by comparing the arrays (a hash alone
 *          never decides), in an open-addressed index read without a lock; only
 *          a registration takes the mutex. A grown index keeps its superseded
 *          generations until the next reclaim, so a lock-free reader never holds
 *          a freed table (the technique of zero_any's GlobalVecHolder).
 *
 *          Admission (section 7): registering every distinct motif would make
 *          random data cost more. find_or_admit() registers a motif the SECOND
 *          time it is seen, BY ANOTHER CHUNK; the first sighting only records,
 *          at its hash in a probation table of 64 Ki entries of 16 bits, 8 more
 *          bits of that hash and an 8-bit tag of the chunk that made it
 *          (sighting_tag()). A sighting counts as the second when both hash
 *          bytes match and the tag differs. The same chunk seen again with the
 *          same contents -- a random chunk rewritten unchanged -- stays on
 *          probation: a motif one chunk uses saves nothing. A motif that lands
 *          on another's entry takes it over (that one's admission is delayed);
 *          two motifs agreeing on 24 bits of hash admit each other one sighting
 *          early, and two chunks sharing a tag delay an admission: a missed or
 *          a useless entry at worst, never a wrong value.
 *
 *          Reclaim (section 7): at a sequential point, the caller marks the
 *          identifiers its chunks still point at (pattern::Array's
 *          reclaim_second_values() walks its arrays); reclaim_unmarked() frees
 *          the others, which later registrations reuse, rebuilds the index from
 *          the survivors and resets the probation table.
 *
 *          The range is reserved at the first registration (MAP_NORESERVE):
 *          only the pages of the entries written are touched. A motif has its
 *          chunk's element count, and a dynamic build has several chunk sizes
 *          (an array's own, a rechunk's new one): the range is cut into
 *          SEGMENTS of segment_bytes, each owned by ONE entry size, handed out
 *          on demand -- a static build may use them all, the sizes of a dynamic
 *          one share them. The range test stays one test; an entry's size is its
 *          segment's. Identifiers are dense across the segments (one per entry
 *          slot carved, 1-based), so the reclaim's flags are one vector. With no
 *          segment left for a size, a registration is refused (MotifStats::full):
 *          the chunk stays materialised, a missed saving, never a wrong value.
 *
 *          A motif chunk's offset is not here: it lives in the chunk's
 *          second-value slot (ProxySpec's table).
 *
 * @tparam T Element type: an integer type other than bool.
 */
template <typename T>
class MotifRegion
{
  public:
#ifndef CHUNK_MOTIF_REGION_BYTES
    #define CHUNK_MOTIF_REGION_BYTES (std::size_t{1} << 30U)
#endif
    /// @brief Size of the reservation: CHUNK_MOTIF_REGION_BYTES (1 GiB by default).
    static constexpr std::size_t region_bytes = CHUNK_MOTIF_REGION_BYTES;

#ifndef CHUNK_MOTIF_SEGMENT_BYTES
    #define CHUNK_MOTIF_SEGMENT_BYTES (std::size_t{1} << 26U)
#endif
    /// @brief Size of a segment, the unit a chunk size owns: CHUNK_MOTIF_SEGMENT_BYTES (64 MiB).
    static constexpr std::size_t segment_bytes = CHUNK_MOTIF_SEGMENT_BYTES;

    /// @brief How many segments the reservation holds.
    static constexpr std::size_t segment_count = region_bytes / segment_bytes;
    static_assert(segment_count >= 1 && region_bytes % segment_bytes == 0,
                  "CHUNK_MOTIF_REGION_BYTES must be a multiple of CHUNK_MOTIF_SEGMENT_BYTES");

    /// @brief Alignment of an entry: one cache line.
    static constexpr std::size_t entry_align = 64;

    /// @brief Probation entries (8 bits of hash, 8 of sighter tag; 0: empty): 2^probation_bits of them.
    static constexpr unsigned probation_bits = 16;

    /// @brief Is @p ptr inside the region (a motif)? False before the first registration.
    /// @details One acquire load and a subtract-compare: on the read path of
    ///          every pattern::Array element, materialised chunks included.
    [[nodiscard]] static bool contains(const void* ptr) noexcept
    {
        const std::byte* const base = begin_.load(std::memory_order_acquire);
        return base != nullptr && ::fast_alloc::address_bits(ptr) - ::fast_alloc::address_bits(base) < region_bytes;
    }

    /// @brief The 64-bit hash of the normalised @p motif, @p count elements.
    [[nodiscard]] static std::uint64_t hash(const T* motif, std::size_t count) noexcept;

    /**
     * @brief The registered motif equal to @p motif, or null. Lock-free.
     * @param motif The normalised motif, @p count elements.
     * @param count Elements per entry: the chunk size.
     * @param h     hash(motif, count).
     */
    [[nodiscard]] static const T* find(const T* motif, std::size_t count, std::uint64_t h) noexcept;

    /**
     * @brief The admission rule: the registered motif equal to @p motif; on its
     *        first sighting, null (8 bits of its hash and the sighter's tag
     *        recorded in the probation table); seen again by the same sighter,
     *        null; by another, registered and returned.
     * @param motif The normalised motif, @p count elements.
     * @param count Elements per entry: the chunk size.
     * @param h     hash(motif, count).
     * @param tag   sighting_tag() of the chunk that holds it.
     */
    [[nodiscard]] static const T* find_or_admit(const T* motif, std::size_t count, std::uint64_t h, std::uint8_t tag);

    /// @brief The probation tag of the chunk at @p who: 8 bits of its address,
    ///        never 0 (the empty probation byte).
    [[nodiscard]] static std::uint8_t sighting_tag(const void* who) noexcept
    {
        const auto mixed = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(who)) * 0x9E3779B97F4A7C15ULL;
        const auto tag   = static_cast<std::uint8_t>(mixed >> 56U);
        return tag == 0 ? std::uint8_t{1} : tag;
    }

    /**
     * @brief The registered motif equal to @p motif, registered first if new,
     *        whatever the probation says (the test entry, a forced admission).
     * @param motif The normalised motif (motif[0] = 0), @p count elements.
     * @param count Elements per entry: the chunk size.
     * @param h     hash(motif, count).
     * @return The entry, @p count elements; null when no segment is left for
     *         that size (MotifStats::full).
     */
    [[nodiscard]] static const T* find_or_register(const T* motif, std::size_t count, std::uint64_t h);

    /// @brief find_or_register() computing the hash.
    [[nodiscard]] static const T* find_or_register(const T* motif, std::size_t count)
    {
        return find_or_register(motif, count, hash(motif, count));
    }

    /// @brief The identifier of the entry @p ptr points into: 1 + its index; 0 for
    ///        a pointer outside the region.
    [[nodiscard]] static std::uint32_t id_of(const void* ptr) noexcept;

    /// @brief How many identifiers have been carved: the size reclaim_unmarked()'s flags need, minus one.
    [[nodiscard]] static std::size_t ids_carved() noexcept
    {
        return next_id_.load(std::memory_order_relaxed) - 1;
    }

    /// @brief The element count of the entry @p entry (its segment's size).
    /// @pre @p entry is a registered entry (contains()).
    [[nodiscard]] static std::size_t entry_count_of(const T* entry) noexcept
    {
        return segments_[segment_of(entry)].count.load(std::memory_order_relaxed);
    }

    /**
     * @brief Free the entries whose identifier is not set in @p referenced, rebuild
     *        the index from the others, reset the probation table.
     * @param referenced One flag per identifier (index 0 unused), set by the caller
     *        for every entry a chunk still points at.
     * @return How many entries were freed.
     * @pre A sequential point: no pass, writer or reader of any pattern::Array<T>
     *      runs, and every chunk pointing at an entry is marked. An entry missed by
     *      the mark is handed to another motif, and its chunk reads that one.
     */
    static std::size_t reclaim_unmarked(const std::vector<std::uint8_t>& referenced);

    /// @brief How many motifs are registered.
    [[nodiscard]] static std::size_t entries() noexcept
    {
        return entries_.load(std::memory_order_relaxed);
    }

    /// @brief Where the region stands (MotifStats): relaxed reads.
    [[nodiscard]] static MotifStats stats() noexcept
    {
        constexpr auto rx = std::memory_order_relaxed;
        return MotifStats{entries_.load(rx), registered_.load(rx), first_seen_.load(rx), reclaims_.load(rx),
                          freed_.load(rx),   full_.load(rx),       sizes_.load(rx)};
    }

  private:
    /// @brief One generation of the hash -> entry index.
    struct IndexTable
    {
        struct Slot
        {
            std::atomic<std::uint64_t> hash{0};        ///< The entry's hash, stored before the entry.
            std::atomic<const T*>      entry{nullptr}; ///< Null: free slot (probes stop).
        };
        explicit IndexTable(std::size_t capacity) : cap(capacity), slots(capacity) {}

        std::size_t       cap  = 0; ///< Slot count, a power of two.
        std::size_t       used = 0; ///< Published entries (the single writer's count).
        std::vector<Slot> slots;    ///< `cap` slots.
    };

    /// @brief One segment: the entry size that owns it (the identifier of each slot
    ///        carved is in segment_ids()).
    struct Segment
    {
        std::atomic<std::size_t> count{0};   ///< Elements per entry; 0: not handed out. Read lock-free (find()).
        std::size_t              bytes  = 0; ///< Bytes per entry (cache-line rounded).
        std::size_t              carved = 0; ///< Slots carved.
    };

    /// @brief One entry size: its open segment and its reclaimed identifiers. Under #mutex_.
    struct SizeClass
    {
        std::size_t                count = 0;             ///< Elements per entry.
        std::size_t                open  = segment_count; ///< The segment being carved; segment_count: none.
        std::vector<std::uint32_t> free_ids;              ///< Reclaimed identifiers of this size.
    };

    /// @brief The segment @p ptr (inside the region) lies in.
    [[nodiscard]] static std::size_t segment_of(const void* ptr) noexcept
    {
        const std::byte* const base = begin_.load(std::memory_order_relaxed);
        return (::fast_alloc::address_bits(ptr) - ::fast_alloc::address_bits(base)) / segment_bytes;
    }

    /// @brief Reserve the range. Under #mutex_.
    static void reserve();

    /// @brief A free entry of @p count elements: a reclaimed identifier of that size
    ///        first, else carved from its open segment or a new one; null if none is
    ///        left. Under #mutex_.
    static T* take_slot(std::size_t count);

    /// @brief The size class of @p count elements, created if new. Under #mutex_.
    static SizeClass& size_class(std::size_t count);

    /// @brief Publish @p entry under @p h in the index, growing it past half load. Under #mutex_.
    static void index_publish(const T* entry, std::uint64_t h);

    /// @brief A table twice as large holding @p old's entries, published; @p old
    ///        stays alive (a reader may hold it) until the next rebuild. Under #mutex_.
    static IndexTable* index_grow(const IndexTable* old);

    /// @brief The index again, from the live entries only; the old generations
    ///        freed (sequential point: no reader). Under #mutex_.
    static void index_rebuild();

    static std::vector<std::uint8_t>& live() noexcept
    {
        return live_storage_.value;
    }
    static std::vector<const T*>& entry_by_id() noexcept
    {
        return entry_by_id_storage_.value;
    }
    static std::vector<SizeClass>& size_classes() noexcept
    {
        return size_classes_storage_.value;
    }
    static std::vector<std::vector<std::uint32_t>>& segment_ids() noexcept
    {
        return segment_ids_storage_.value;
    }
    static std::vector<std::unique_ptr<IndexTable>>& generations() noexcept
    {
        return generations_storage_.value;
    }

    static inline std::mutex               mutex_;             ///< Serialises registrations and reclaims.
    static inline std::atomic<std::byte*>  begin_{nullptr};    ///< Start of the reservation; null before the first.
    static inline std::atomic<std::size_t> next_id_{1};        ///< The next identifier carved.
    static inline std::size_t              segments_used_ = 0; ///< Segments handed out. Under #mutex_.
    static inline constinit std::array<Segment, segment_count> segments_{};     ///< The segments.
    static inline std::atomic<IndexTable*>                     index_{nullptr}; ///< The current index generation.
    static inline std::atomic<std::size_t>                     entries_{0};     ///< Live entries.
    static inline std::atomic<std::size_t>                     registered_{0};  ///< MotifStats::registered.
    static inline std::atomic<std::size_t>                     first_seen_{0};  ///< MotifStats::first_seen.
    static inline std::atomic<std::size_t>                     reclaims_{0};    ///< MotifStats::reclaims.
    static inline std::atomic<std::size_t>                     freed_{0};       ///< MotifStats::freed.
    static inline std::atomic<std::size_t>                     full_{0};        ///< MotifStats::full.
    static inline std::atomic<std::size_t>                     sizes_{0};       ///< MotifStats::sizes.
    static inline std::array<std::atomic<std::uint16_t>, std::size_t{1} << probation_bits>
        probation_{}; ///< First sightings.
    static inline constinit internal::Immortal<std::vector<std::uint8_t>>
        live_storage_{}; ///< Live flag per identifier.
    static inline constinit internal::Immortal<std::vector<const T*>>
        entry_by_id_storage_{}; ///< Entry of each identifier.
    static inline constinit internal::Immortal<std::vector<SizeClass>> size_classes_storage_{}; ///< One per entry size.
    static inline constinit internal::Immortal<std::vector<std::vector<std::uint32_t>>>
        segment_ids_storage_{}; ///< Per segment, the identifier of each slot carved.
    static inline constinit internal::Immortal<std::vector<std::unique_ptr<IndexTable>>>
        generations_storage_{}; ///< Every index generation since the last rebuild.
};

} // namespace pattern

} // namespace shrink_array

#include "chunk/internal/motif_region_inline.h" // the template bodies
