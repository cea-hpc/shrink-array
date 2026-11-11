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

// Which ranks hold something of an array in their own per-rank structures.
//
// A rank holds a pointer to an array's chunks in two places: its candidate
// list (a chunk it queued for compression) and its queue of deferred writes.
// An array's teardown must leave none behind. At a sequential point it walks
// every rank's structures. Inside a region it cannot: the other ranks are
// appending to theirs. But an array built by a thread of the region and used
// by that thread alone -- a work array -- is in that thread's structures only,
// and its teardown needs no other.
//
// This table records, per array (num_tab), whether that is the case: no rank
// yet, one rank, or several. A rank notes itself when it queues a chunk of the
// array (the first time only: a relaxed load afterwards) and when it defers a
// write to it. An exchange of identities (swap, re-chunk) marks both arrays
// shared: their candidates may carry the other's num_tab.
//
// The teardown reads it (only()): inside a region, an array that only the
// calling rank has used takes the own-rank path; any other keeps the
// sequential-point path, as before.

#include <atomic>  // std::atomic
#include <cstddef> // std::size_t

#include "chunk/internal/immortal.h"     // Immortal: statics safe for global arrays
#include "chunk/internal/stable_table.h" // StableTable: read while another thread registers an array

namespace shrink_array::internal
{

/// @brief Per array of chunk type @p ChunkT: the ranks that queued one of its
///        chunks or deferred a write to it.
/// @tparam ChunkT The chunk type (one num_tab sequence per chunk type).
template <typename ChunkT>
class TabUsers
{
  public:
    /// @brief Make room for @p nb_tab arrays.
    /// @pre The caller holds the cold-path lock.
    static void resize(int nb_tab)
    {
        table.resize(static_cast<std::size_t>(nb_tab));
    }

    /// @brief Rank @p rank queues a chunk of array @p num_tab, or defers a write to it.
    static void note(int num_tab, int rank) noexcept
    {
        if (num_tab < 0) [[unlikely]]
        {
            return;
        }
        auto&     users = table[static_cast<std::size_t>(num_tab)];
        const int mine  = rank + 1;
        int       seen  = users.load(std::memory_order_relaxed);
        if (seen == mine || seen == several) [[likely]]
        {
            return;
        }
        if (seen == none && users.compare_exchange_strong(seen, mine, std::memory_order_relaxed))
        {
            return;
        }
        if (seen != mine)
        {
            users.store(several, std::memory_order_relaxed);
        }
    }

    /// @brief Mark array @p num_tab as used by several ranks, whatever was noted.
    static void mark_shared(int num_tab) noexcept
    {
        if (num_tab >= 0 && static_cast<std::size_t>(num_tab) < table.size())
        {
            table[static_cast<std::size_t>(num_tab)].store(several, std::memory_order_relaxed);
        }
    }

    /// @brief Has no rank other than @p rank queued a chunk of array @p num_tab
    ///        or deferred a write to it?
    [[nodiscard]] static bool only(int num_tab, int rank) noexcept
    {
        if (num_tab < 0 || static_cast<std::size_t>(num_tab) >= table.size())
        {
            return false;
        }
        const int seen = table[static_cast<std::size_t>(num_tab)].load(std::memory_order_relaxed);
        return seen == none || seen == rank + 1;
    }

  private:
    static constexpr int none    = 0;  ///< no rank yet (a row's initial value)
    static constexpr int several = -1; ///< more than one rank; a single rank r is r + 1

    static inline constinit Immortal<StableTable<std::atomic<int>>> table_storage_{};
    static inline StableTable<std::atomic<int>>&                    table = table_storage_.value;
};

} // namespace shrink_array::internal
