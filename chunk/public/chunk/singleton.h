/*
 * Copyright 2024-2025 CEA (www.cea.fr)
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

#include <atomic>
#include <mutex>

// See fast_allocators/singleton.h: mark the intentionally-immortal instance so
// LeakSanitizer neither reports it nor lets it mask real leaks. No-op without a
// sanitizer actively enabled by compiler flags.
#if defined(__has_feature)
    #if __has_feature(address_sanitizer) || __has_feature(leak_sanitizer)
        #include <sanitizer/lsan_interface.h>
        #define SA_LSAN_IGNORE(p) __lsan_ignore_object(p)
    #endif
#elif defined(__SANITIZE_ADDRESS__)
    #include <sanitizer/lsan_interface.h>
    #define SA_LSAN_IGNORE(p) __lsan_ignore_object(p)
#endif

#ifndef SA_LSAN_IGNORE
    #define SA_LSAN_IGNORE(p) ((void)0)
#endif

namespace shrink_array
{

/// @brief Lazily created, never deleted, process-wide instance of T, built with `new T()` on first use.
/// @tparam T Type of the instance.
template <typename T>
class SingletonChk
{
  public:
    SingletonChk(const SingletonChk&)            = delete;
    SingletonChk& operator=(const SingletonChk&) = delete;
    SingletonChk(SingletonChk&&)                 = delete;
    SingletonChk& operator=(SingletonChk&&)      = delete;
    ~SingletonChk()                              = default;

    /// @brief getInstance(), as a reference.
    static T& getRef()
    {
        return *getInstance();
    }

  private:
    /// @brief The instance, created under the mutex on first call (double-checked locking).
    /// @return Pointer to the instance.
    static T* getInstance()
    {
        // Fast path: one acquire load, no lock. Every call used to take the
        // mutex, and this is reached once per thread per barrier (64 threads
        // serialised on it at each one). The acquire pairs with the release
        // store below: seeing the pointer implies seeing the constructed T.
        if (T* instance = unique_instance.load(std::memory_order_acquire)) [[likely]]
            return instance;

        const std::scoped_lock lock(unique_instance_mutex);
        T*                     instance = unique_instance.load(std::memory_order_relaxed); // under the lock
        if (instance == nullptr)
        {
            instance = new T();       // immortal: never deleted
            SA_LSAN_IGNORE(instance); // intentional leak, not a real one
            unique_instance.store(instance, std::memory_order_release);
        }
        return instance;
    }

    /// @brief Delete the instance. Retained for API completeness; unused
    ///        under the immortal design (getInstance() never deletes).
    static void deleteInstance()
    {
        const std::scoped_lock lock(unique_instance_mutex);
        delete unique_instance.exchange(nullptr, std::memory_order_acq_rel);
    }

  protected:
    // IMMORTAL raw pointer (was std::unique_ptr<T>): created once with `new`
    // under the mutex and NEVER deleted. Rationale identical to the
    // fast_allocators Singleton: a global unique_ptr's exit-time destructor
    // races worker-thread TLS destructors that still touch the singleton at
    // shutdown (static/TLS destruction-order data race). Leaking avoids the
    // exit-time destructor; the OS reclaims the memory at process end.
    //
    // Atomic so that getInstance() can read it without the mutex (double-
    // checked locking, see there).
    static inline std::atomic<T*> unique_instance{nullptr};

    // Guards lazy construction. The previous version had NO lock, so two threads
    // hitting getInstance() on first use could both new the instance and the
    // second assignment free the first → double-free. Held only while the
    // instance does not exist yet: getInstance() IS on a hot path (per-barrier
    // bookkeeping, every thread), so its steady state must not lock.
    static inline std::mutex unique_instance_mutex;
};

/// @brief Free-function form of SingletonChk<T>::deleteInstance().
template <typename T>
void deleteInstance()
{
    SingletonChk<T>::deleteInstance();
}

} // namespace shrink_array