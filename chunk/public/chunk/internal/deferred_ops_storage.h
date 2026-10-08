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

#include <vector>  // std::vector
#include <atomic>  // std::atomic, std::memory_order
#include <cstddef> // std::size_t
#include <mutex>   // std::mutex, std::scoped_lock

#include "chunk/PMP_CHUNK.h"         // CHUNK_PMP_* macros — used HERE, not via includers
#include "chunk/internal/immortal.h" // Immortal / LazyImmortal: statics safe for global arrays

namespace shrink_array
{

/**
 * @brief Global storage for deferred operations.
 * Optimized to remove initialization overhead in high-frequency loops.
 */
template <typename per_thread_map>
class DeferredOpsStorage
{
  public:
    /**
     * @brief Singleton access.
     * The instance is constant-initialised (deferred_ops_storage_instance,
     * below): no "Magic Static" lock check, usable before any dynamic
     * initialisation.
     */
    static DeferredOpsStorage& instance() noexcept;

    /**
     * @brief Returns the raw pointer to the thread maps.
     * Uses acquire memory order to ensure visibility of allocated data.
     */
    per_thread_map* get() noexcept
    {
        return arr_raw.load(std::memory_order_acquire);
    }

    /**
     * @brief Allocates the per-thread map array (idempotent, thread-safe).
     *
     * Double-checked locking on arr_raw, under alloc_mutex. NOT std::call_once
     * with a unique_ptr<once_flag> member: a DYNAMIC default-member-initializer
     * (make_unique) on a static-storage singleton (the instance) is a
     * static-initialization-order hazard -- allocate() running before that init
     * dereferences a null unique_ptr (SIGABRT under nvc++). std::mutex has a
     * constexpr constructor, so alloc_mutex is CONSTANT-initialized and
     * the instance needs no dynamic init at all.
     */
    void allocate()
    {
        // Fast path: already allocated (lock-free).
        if (arr_raw.load(std::memory_order_acquire) != nullptr)
        {
            return;
        }

        const std::scoped_lock lk(alloc_mutex);
        // Double-check under the lock.
        if (arr_raw.load(std::memory_order_relaxed) != nullptr)
        {
            return;
        }

        arr_owner = std::vector<per_thread_map>(static_cast<std::size_t>(CHUNK_PMP_THREAD_CAPACITY));
        // Publish last, with release, so a reader that acquire-observes the
        // pointer also sees the constructed maps.
        arr_raw.store(arr_owner.data(), std::memory_order_release);
    }

    /**
     * @brief Destroys the maps and invalidates the raw pointer.
     * Can be called safely during execution. Serialized against allocate() by
     * alloc_mutex so a concurrent (re)allocation cannot race the teardown. A
     * later allocate() simply re-allocates (no once_flag to re-arm).
     */
    void destroy()
    {
        const std::scoped_lock lk(alloc_mutex);
        if (arr_raw.load(std::memory_order_relaxed) == nullptr)
        {
            return;
        }

        // Invalidate the global pointer so threads see it as null immediately,
        // then free.
        arr_raw.store(nullptr, std::memory_order_release);
        std::vector<per_thread_map>().swap(arr_owner); // frees the maps and their storage
    }

  private:
    constexpr DeferredOpsStorage() = default;

    // The singleton's storage (deferred_ops_storage_instance below) builds it.
    friend union internal::Immortal<DeferredOpsStorage>;

    std::vector<per_thread_map> arr_owner;

    // Atomic pointer used by worker threads for O(1) access
    alignas(64) std::atomic<per_thread_map*> arr_raw{nullptr};

    // Guards allocate()/destroy(). constexpr ctor -> constant-initialized.
    std::mutex alloc_mutex;
};

/**
 * @brief The one DeferredOpsStorage of each map type.
 * @details Every member is constant-initialisable (constexpr mutex and atomic
 *          constructors, an empty std::vector), so it needs no dynamic
 *          initialisation -- no static-init-order hazard -- and, as an
 *          Immortal (chunk/internal/immortal.h), it is never destroyed: a
 *          global array still reaches it after main returns. Outside the class:
 *          an Immortal of the class needs the class complete.
 */
template <typename per_thread_map>
inline constinit internal::Immortal<DeferredOpsStorage<per_thread_map>> deferred_ops_storage_instance{};

template <typename per_thread_map>
DeferredOpsStorage<per_thread_map>& DeferredOpsStorage<per_thread_map>::instance() noexcept
{
    return deferred_ops_storage_instance<per_thread_map>.value;
}

} // namespace shrink_array