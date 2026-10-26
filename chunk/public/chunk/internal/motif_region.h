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

#include <atomic>  // std::atomic (the range bounds the readers test)
#include <cstddef> // std::byte, std::size_t
#include <cstdint> // std::uintptr_t
#include <cstring> // std::memcmp, std::memcpy
#include <mutex>   // std::mutex, std::scoped_lock (the registrations)

#include "fast_allocators/internal/raw_memory.h" // address_bits
#include "chunk/internal/invariant_check.h"      // CHUNK_INVARIANT_FAILED

namespace shrink_array
{

namespace pattern
{

/**
 * @brief The motifs of pattern::Array<T>: chunk-sized arrays of T, normalised
 *        (motif[0] = 0), in ONE reserved virtual range, so that "this chunk's
 *        values pointer is a motif" is a range test on the pointer.
 *
 * @details Entries are written before any chunk points at them and never
 *          change afterwards. Stage 1 of the design note: registration by value
 *          under a mutex with a linear search (for the tests), entries never
 *          freed. Stage 2 brings the hash, the admission rule and the reclaim.
 *          The range is reserved at the first registration (MAP_NORESERVE):
 *          only the pages of the entries written are touched.
 *
 *          A motif chunk's offset is not here: it lives in the chunk's
 *          second-value slot (ProxySpec's table), free since a motif chunk has
 *          no second value.
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

    /// @brief Alignment of an entry: one cache line.
    static constexpr std::size_t entry_align = 64;

    /// @brief Is @p ptr inside the region (a motif)? False before the first registration.
    /// @details One acquire load and a subtract-compare: on the read path of
    ///          every pattern::Array element, materialised chunks included.
    [[nodiscard]] static bool contains(const void* ptr) noexcept
    {
        const std::byte* const base = begin_.load(std::memory_order_acquire);
        return base != nullptr && ::fast_alloc::address_bits(ptr) - ::fast_alloc::address_bits(base) < region_bytes;
    }

    /**
     * @brief The registered motif equal to @p motif, registered first if new.
     * @param motif The normalised motif (motif[0] = 0), @p count elements.
     * @param count Elements per entry: the chunk size. The same for every entry
     *        (static chunk sizes only for now).
     * @return The entry, @p count elements, never written again.
     */
    [[nodiscard]] static const T* find_or_register(const T* motif, std::size_t count)
    {
        const std::scoped_lock lock(mutex_);
        if (begin_.load(std::memory_order_relaxed) == nullptr)
        {
            reserve(count);
        }
        else if (count != entry_count_)
        {
            CHUNK_INVARIANT_FAILED("pattern::MotifRegion: an entry of {} elements, the region holds {} per entry",
                                   count, entry_count_);
        }
        std::byte* const base = begin_.load(std::memory_order_relaxed);
        for (std::size_t off = 0; off < used_; off += entry_bytes_)
        {
            const T* const entry = reinterpret_cast<const T*>(base + off);
            if (std::memcmp(entry, motif, count * sizeof(T)) == 0)
            {
                return entry;
            }
        }
        if (used_ + entry_bytes_ > region_bytes)
        {
            CHUNK_INVARIANT_FAILED("pattern::MotifRegion: the region is full ({} entries of {} bytes, "
                                   "CHUNK_MOTIF_REGION_BYTES = {})",
                                   entries_, entry_bytes_, region_bytes);
        }
        T* const entry = reinterpret_cast<T*>(base + used_);
        std::memcpy(entry, motif, count * sizeof(T));
        used_ += entry_bytes_;
        ++entries_;
        return entry;
    }

    /// @brief How many motifs are registered.
    [[nodiscard]] static std::size_t entries() noexcept
    {
        const std::scoped_lock lock(mutex_);
        return entries_;
    }

  private:
    /// @brief Reserve the range for entries of @p count elements. Under #mutex_.
    static void reserve(std::size_t count)
    {
        void* const raw =
            ::mmap(nullptr, region_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
        if (raw == MAP_FAILED)
        {
            CHUNK_INVARIANT_FAILED("pattern::MotifRegion: cannot reserve {} bytes of address space", region_bytes);
        }
        entry_count_ = count;
        entry_bytes_ = (count * sizeof(T) + entry_align - 1) / entry_align * entry_align;
        begin_.store(static_cast<std::byte*>(raw), std::memory_order_release);
    }

    static inline std::mutex              mutex_;           ///< Serialises the registrations.
    static inline std::atomic<std::byte*> begin_{nullptr};  ///< Start of the reservation; null before the first.
    static inline std::size_t             used_        = 0; ///< Bytes of the region in use.
    static inline std::size_t             entries_     = 0; ///< Registered motifs.
    static inline std::size_t             entry_count_ = 0; ///< Elements per entry.
    static inline std::size_t             entry_bytes_ = 0; ///< Bytes per entry (cache-line rounded).
};

} // namespace pattern

} // namespace shrink_array
