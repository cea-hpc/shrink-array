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

// A table indexed by num_tab whose rows never move.
//
// The outer per-array tables (one row per array: its candidate flags, its
// constant slots, its second values) were std::vector<Row>. A proxy indexes
// them on every write; a constructor grows them. A vector moves its rows when
// it grows, so an array built by one thread of a region freed the storage
// another thread's proxy was reading. StableTable grows by adding a segment and
// never touches the existing ones: operator[] is safe against a concurrent
// resize(), with no lock.
//
// Layout: segment s holds first_size << s rows; the row of index i is found
// from the highest bit of i + first_size. The first first_size rows -- the
// arrays of most programs -- are read as a vector's are: one load of the
// segment's address, then the index. Beyond, a bit scan on top of that.
//
// Contract:
//   - resize() only grows, and its callers hold the cold-path lock
//     (cold_path_lock.h);
//   - operator[] may run concurrently with resize() for any index below a
//     size() the caller has observed (its own array's num_tab, always);
//   - a row is written by its array's owner only: the table protects the
//     rows' addresses, not their contents.

#include <atomic>  // std::atomic
#include <bit>     // std::bit_width
#include <cstddef> // std::size_t

namespace shrink_array::internal
{

/// @brief Growable table with address-stable rows and a lock-free operator[].
/// @tparam Row Default-constructible row type.
template <typename Row>
class StableTable
{
  public:
    using value_type = Row;

    constexpr StableTable() noexcept = default;
    ~StableTable()                   = default; // lives in an Immortal: never destroyed

    StableTable(const StableTable&)            = delete;
    StableTable& operator=(const StableTable&) = delete;
    StableTable(StableTable&&)                 = delete;
    StableTable& operator=(StableTable&&)      = delete;

    /// @brief Row @p index. Safe against a concurrent resize().
    [[nodiscard]] Row& operator[](std::size_t index) noexcept;
    /// @copydoc operator[](std::size_t)
    [[nodiscard]] const Row& operator[](std::size_t index) const noexcept;

    /// @brief How many rows there are.
    [[nodiscard]] std::size_t size() const noexcept
    {
        return size_.load(std::memory_order_acquire);
    }

    /// @brief Grow to @p count rows, value-initialised. Never shrinks.
    /// @pre The caller holds the cold-path lock.
    void resize(std::size_t count);

  private:
    static constexpr std::size_t first_bits  = 8;                            ///< log2 of the first segment's size
    static constexpr std::size_t first_size  = std::size_t{1} << first_bits; ///< rows in segment 0
    static constexpr std::size_t nb_segments = 32; ///< a power of two; an int num_tab needs 24

    /// @brief Segment of the row whose index plus first_size is @p biased.
    /// @details Masked: in bounds whatever the index, so an index that is not
    ///          one (a negative num_tab) cannot address past the segment table.
    [[nodiscard]] static constexpr std::size_t segment_of(std::size_t biased) noexcept
    {
        return (static_cast<std::size_t>(std::bit_width(biased)) - (first_bits + 1)) & (nb_segments - 1);
    }

    std::atomic<Row*> segments_[nb_segments]{}; // NOLINT(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
    std::atomic<std::size_t> size_{0};
};

} // namespace shrink_array::internal

#include "chunk/internal/stable_table_inline.h" // the template bodies
