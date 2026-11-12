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
 * @file flat_address_set.h
 * @brief A set of non-null addresses in one flat array (SlabArena's live slabs).
 *
 * @details SlabArena keeps the address of every slab it has handed out, to
 *          refuse handing one out twice (its corruption check). The set was a
 *          std::unordered_set: one heap node per slab, so every insert was a
 *          malloc, every erase a free and a cache miss -- 430 ns a slab when a
 *          re-chunk boundary recycles hundreds of thousands of them under the
 *          arena's lock.
 *
 *          Here the addresses sit in one power-of-two array: linear probing
 *          from a full 64-bit mix of the address (slab addresses are aligned
 *          and come in arithmetic runs: a weaker hash clusters them), at most
 *          half full, erase by backward shift (no tombstone:
 *          the probe chains stay as short after a million erases as after
 *          none). 0 marks an empty slot: the null address is never stored.
 *
 * @note Not thread-safe: SlabArena calls it under its mutex.
 */

#include <cstddef> // std::size_t
#include <cstdint> // std::uintptr_t, std::uint64_t
#include <vector>  // std::vector (the slots)

namespace fast_alloc::detail
{

class FlatAddressSet
{
  public:
    /// @brief Insert @p key (non-zero). @return false if it was already in.
    /// @throws std::bad_alloc when the table must grow and cannot.
    bool insert(std::uintptr_t key);

    /// @brief Is @p key in the set?
    [[nodiscard]] bool contains(std::uintptr_t key) const noexcept;

    /// @brief Remove @p key. @return 1 if it was in, 0 otherwise (as std::unordered_set::erase).
    std::size_t erase(std::uintptr_t key) noexcept;

    /// @brief Number of addresses in the set.
    [[nodiscard]] std::size_t size() const noexcept
    {
        return size_;
    }

    /// @brief Is the set empty?
    [[nodiscard]] bool empty() const noexcept
    {
        return size_ == 0;
    }

  private:
    /// @brief The home slot of @p key: the top bits of murmur3's fmix64 of it.
    /// @details A full mix, not a single multiplication: slab addresses form
    ///          arithmetic runs (a region carved back to back), and Fibonacci
    ///          hashing or half of fmix64 clustered them -- 12 to 55 probes on
    ///          average at some table sizes, against 0.3-0.5 here at any size.
    [[nodiscard]] std::size_t home(std::uintptr_t key) const noexcept
    {
        auto h = static_cast<std::uint64_t>(key);
        h      = (h ^ (h >> 33U)) * 0xFF51AFD7ED558CCDULL;
        h      = (h ^ (h >> 33U)) * 0xC4CEB9FE1A85EC53ULL;
        return static_cast<std::size_t>((h ^ (h >> 33U)) >> shift_);
    }

    /// @brief The slot holding @p key, or the empty slot that ends its probe chain.
    [[nodiscard]] std::size_t find_slot(std::uintptr_t key) const noexcept;

    /// @brief Double the table (16 slots when empty) and re-insert every key.
    void grow();

    std::vector<std::uintptr_t> slots_;      ///< 0 = empty; size a power of two, or 0.
    std::size_t                 size_  = 0;  ///< Keys in the table.
    std::size_t                 mask_  = 0;  ///< slots_.size() - 1.
    unsigned                    shift_ = 64; ///< 64 - log2(slots_.size()): the hash's top bits.
};

} // namespace fast_alloc::detail

#include "fast_allocators/internal/flat_address_set_inline.h" // the member bodies
