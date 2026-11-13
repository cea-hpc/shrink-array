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

// The lock of the cold path: building and tearing down an array.
//
// An array's constructor and destructor write state shared by every array of
// its element type -- the identity registry (share_info_array), the next
// num_tab, the outer per-array tables, the manager's totals and thresholds.
// Two threads of a parallel region each building a work array of the same type
// used to write all of that at once. The cold path now takes this lock; the hot
// path (operator[], the proxies, the passes) never does: what it reads of that
// state is either address-stable (StableTable) or atomic.
//
// One lock for the whole library, not one per type: a constructor can register
// a type (AllTypeChunk::ensure_manager) and a destructor can run a terminal
// pass over every type. Recursive: rechunk_to() builds and destroys a temporary
// inside the array it re-chunks, and a region opened under it may build one too.

#include <array>        // std::array
#include <atomic>       // std::atomic
#include <bit>          // std::bit_cast
#include <cstddef>      // std::size_t
#include <cstdint>      // std::uint64_t, std::uintptr_t
#include <mutex>        // std::recursive_mutex, std::scoped_lock, std::unique_lock
#include <shared_mutex> // std::shared_mutex, std::shared_lock
#include <utility>      // std::forward

namespace shrink_array::internal
{

/// @brief The cold-path lock. Never destroyed: a global array is torn down
///        after main returns.
[[nodiscard]] inline std::recursive_mutex& cold_path_mutex() noexcept
{
    static std::recursive_mutex* const mutex = new std::recursive_mutex; // NOLINT(cppcoreguidelines-owning-memory)
    return *mutex;
}

/// @brief Holds the cold-path lock for a scope.
using ColdPathLock = std::scoped_lock<std::recursive_mutex>;

// The identity registries (share_info_array: array address -> num_tab, and for
// zero_one_value its second value) are written on the cold path, under the lock
// above, and looked up outside it: by the replay of a deferred write, by a proxy
// that was not armed by operator[], and by the cross-checks of the check builds
// on every write. A lookup takes this lock shared, a change takes it exclusive.
// A leaf: nothing is called while it is held, so it nests under any other lock.
// Not taken by for_each_key() (it runs a callback): sequential points only.

/// @brief The identity registries' lock. Never destroyed.
[[nodiscard]] inline std::shared_mutex& registry_mutex() noexcept
{
    static std::shared_mutex* const mutex = new std::shared_mutex; // NOLINT(cppcoreguidelines-owning-memory)
    return *mutex;
}

/// @brief Holds the registries' lock for a lookup.
using RegistryReadLock = std::shared_lock<std::shared_mutex>;

/// @brief The registries' generation: bumped by every RegistryWriteLock as it
///        releases. Two loads that return the same value bracket no change of
///        any registry. Read by verify_once_per_generation().
[[nodiscard]] inline std::atomic<std::uint64_t>& registry_generation() noexcept
{
    static std::atomic<std::uint64_t> generation{0};
    return generation;
}

/// @brief Holds the registries' lock for a registration, a removal or an
///        exchange, and bumps registry_generation() before releasing it.
class RegistryWriteLock
{
  public:
    explicit RegistryWriteLock(std::shared_mutex& mutex) : lock_(mutex) {}
    // The body runs before lock_ is destroyed: the bump is still under the lock.
    ~RegistryWriteLock()
    {
        registry_generation().fetch_add(1, std::memory_order_release);
    }

    RegistryWriteLock(const RegistryWriteLock&)            = delete;
    RegistryWriteLock& operator=(const RegistryWriteLock&) = delete;
    RegistryWriteLock(RegistryWriteLock&&)                 = delete;
    RegistryWriteLock& operator=(RegistryWriteLock&&)      = delete;

  private:
    std::unique_lock<std::shared_mutex> lock_;
};

// The check builds compare, on every write through a proxy, what the proxy was
// armed with against the registry. Done as is, that is a RegistryReadLock per
// write: every thread increments and decrements the one reader count of
// registry_mutex(), and the cache line it sits on bounces between the cores
// (pthread_rwlock_rdlock + unlock were ~91 % of the omp-gnu-check bench). The
// answer cannot change while neither the key nor any registry does: a thread
// remembers the (generation, key) pairs it verified and skips those. The hot
// path is then one atomic load of a line that is only written by a registry
// change, plus thread-local memory.

/// @brief Runs @p verify -- a cross-check that reads a registry under
///        RegistryReadLock and reports a mismatch itself -- unless this thread
///        already ran it on @p key since the last registry change.
/// @details A memo per thread and per call site (a lambda has a type of its
///          own, so does each template instantiation of the enclosing
///          function): 16 sets of 4 ways, the set picked by a multiplicative
///          hash of @p slot_hint. Not direct-mapped: with 16 slots indexed by
///          a few address bits, two arrays a loop writes in turn could share a
///          slot, depending on where ASLR put them, and evict each other on
///          every write -- the lock on every write again, in one run out of
///          two. The generation is loaded BEFORE @p verify: a change during it
///          leaves an older generation in the memo, and the next call checks
///          again.
/// @param slot_hint the array's address: picks the memo set.
/// @param key       everything the answer depends on besides the registries
///                  (the array, the values the proxy was armed with).
template <typename Key>
struct GenerationMemo
{
    std::uint64_t generation = 0;
    Key           key{};
    bool          valid = false;
};

inline constexpr std::size_t generation_memo_ways     = 4;
inline constexpr int         generation_memo_set_bits = 4; // 16 sets

/// @brief One set of the memo (namespace scope: GCC 14 crashes on a local
///        struct of the function template that uses its local constants).
template <typename Key>
struct GenerationMemoSet
{
    std::array<GenerationMemo<Key>, generation_memo_ways> ways{};
    std::size_t                                           next_victim = 0; ///< round robin
};

template <typename Key, typename Verify>
void verify_once_per_generation(const void* slot_hint, const Key& key, Verify&& verify)
{
    using Memo                                                             = GenerationMemo<Key>;
    using MemoSet                                                          = GenerationMemoSet<Key>;
    constexpr std::size_t                                        memo_ways = generation_memo_ways;
    constexpr int                                                set_bits  = generation_memo_set_bits;
    thread_local std::array<MemoSet, std::size_t{1} << set_bits> memos{};

    // Fibonacci hashing: the top bits of the product depend on every bit of
    // the address.
    constexpr std::uint64_t golden = 0x9E3779B97F4A7C15ULL;
    const auto              addr   = static_cast<std::uint64_t>(std::bit_cast<std::uintptr_t>(slot_hint));
    MemoSet&                set    = memos[static_cast<std::size_t>((addr * golden) >> (64 - set_bits))];

    const auto generation = registry_generation().load(std::memory_order_acquire);
    for (const Memo& memo : set.ways)
    {
        if (memo.valid && memo.generation == generation && memo.key == key) [[likely]]
        {
            return;
        }
    }
    std::forward<Verify>(verify)();

    // The way that holds this key under an older generation, or the next
    // victim of the round robin.
    std::size_t victim = set.next_victim;
    for (std::size_t w = 0; w < memo_ways; ++w)
    {
        if (set.ways[w].valid && set.ways[w].key == key)
        {
            victim = w;
            break;
        }
    }
    if (victim == set.next_victim)
    {
        set.next_victim = (set.next_victim + 1) % memo_ways;
    }
    set.ways[victim] = Memo{generation, key, true};
}

} // namespace shrink_array::internal
