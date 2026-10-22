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
template <typename PerThreadMap>
class DeferredOpsStorage
{
  public:
    /**
     * @brief Singleton access. Constant-initialised (deferred_ops_storage_instance): no
     * magic-static guard, usable before any dynamic initialisation.
     */
    static DeferredOpsStorage& instance() noexcept;

    /**
     * @brief Raw pointer to the thread maps (acquire: pairs with allocate()'s publication).
     */
    PerThreadMap* get() noexcept
    {
        return arr_raw_.load(std::memory_order_acquire);
    }

    /**
     * @brief Allocates the per-thread map array (idempotent, thread-safe).
     *
     * Double-checked locking on arr_raw, under alloc_mutex. Do not use std::call_once with a
     * dynamically initialised member (e.g. unique_ptr<once_flag>): allocate() can run before
     * that init (static-init order). std::mutex is constant-initialized, so the instance
     * needs no dynamic init at all.
     */
    void allocate()
    {
        // Fast path: already allocated (lock-free).
        if (arr_raw_.load(std::memory_order_acquire) != nullptr)
        {
            return;
        }

        const std::scoped_lock lk(alloc_mutex_);
        // Double-check under the lock.
        if (arr_raw_.load(std::memory_order_relaxed) != nullptr)
        {
            return;
        }

        arr_owner_ = std::vector<PerThreadMap>(static_cast<std::size_t>(CHUNK_PMP_THREAD_CAPACITY));
        // Publish last, with release, so a reader that acquire-observes the
        // pointer also sees the constructed maps.
        arr_raw_.store(arr_owner_.data(), std::memory_order_release);
    }

    /**
     * @brief Destroys the maps and invalidates the raw pointer. Serialized against
     * allocate() by alloc_mutex; a later allocate() re-allocates.
     */
    void destroy()
    {
        const std::scoped_lock lk(alloc_mutex_);
        if (arr_raw_.load(std::memory_order_relaxed) == nullptr)
        {
            return;
        }

        // Invalidate the pointer first, then free.
        arr_raw_.store(nullptr, std::memory_order_release);
        std::vector<PerThreadMap>().swap(arr_owner_); // frees the maps and their storage
    }

  private:
    constexpr DeferredOpsStorage() = default;

    // The singleton's storage (deferred_ops_storage_instance below) builds it.
    friend union internal::Immortal<DeferredOpsStorage>;

    std::vector<PerThreadMap> arr_owner_;

    // Atomic pointer used by worker threads for O(1) access
    alignas(64) std::atomic<PerThreadMap*> arr_raw_{nullptr};

    // Guards allocate()/destroy(). constexpr ctor -> constant-initialized.
    std::mutex alloc_mutex_;
};

/**
 * @brief The one DeferredOpsStorage of each map type.
 * @details Constant-initialisable (no static-init-order hazard) and, as an Immortal
 *          (chunk/internal/immortal.h), never destroyed: a global array still reaches it
 *          after main returns. Outside the class: an Immortal needs the class complete.
 */
template <typename PerThreadMap>
inline constinit internal::Immortal<DeferredOpsStorage<PerThreadMap>> deferred_ops_storage_instance{};

template <typename PerThreadMap>
DeferredOpsStorage<PerThreadMap>& DeferredOpsStorage<PerThreadMap>::instance() noexcept
{
    return deferred_ops_storage_instance<PerThreadMap>.value;
}

} // namespace shrink_array