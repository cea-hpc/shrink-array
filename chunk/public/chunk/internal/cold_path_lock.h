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

#include <mutex>        // std::recursive_mutex, std::scoped_lock, std::unique_lock
#include <shared_mutex> // std::shared_mutex, std::shared_lock

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
/// @brief Holds the registries' lock for a registration, a removal or an exchange.
using RegistryWriteLock = std::unique_lock<std::shared_mutex>;

} // namespace shrink_array::internal
