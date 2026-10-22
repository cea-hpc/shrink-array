/*
 * Copyright 2024-2026 CEA (www.cea.fr)
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

#include <atomic> // std::atomic, std::memory_order
#include <mutex>  // std::mutex, std::scoped_lock

#ifdef __has_feature
    #if __has_feature(address_sanitizer) || __has_feature(leak_sanitizer)
        #include <sanitizer/lsan_interface.h> // __lsan_ignore_object
        #define FAST_ALLOC_LSAN_IGNORE(p) __lsan_ignore_object(p)
    #endif
#elif defined(__SANITIZE_ADDRESS__)
    #include <sanitizer/lsan_interface.h> // __lsan_ignore_object
    #define FAST_ALLOC_LSAN_IGNORE(p) __lsan_ignore_object(p)
#endif

#ifndef FAST_ALLOC_LSAN_IGNORE
    #define FAST_ALLOC_LSAN_IGNORE(p) ((void)0)
#endif

namespace fast_alloc
{

/**
 * @brief Process-wide lazily-constructed instance of T, shared by every thread.
 * @details Created on the first get_instance() and never destroyed unless
 *          delete_instance() is called: an intentional leak, marked for LSan.
 *          The chunk library reaches it as shrink_array::SingletonChk
 *          (chunk/singleton.h). delete_instance() is for outside code only.
 * @tparam T The type to host; default-constructible.
 */
template <typename T>
class Singleton
{
  public:
    // Never an object: everything is static. Neither copied nor moved.
    Singleton(const Singleton&)            = delete;
    Singleton& operator=(const Singleton&) = delete;
    Singleton(Singleton&&)                 = delete;
    Singleton& operator=(Singleton&&)      = delete;
    ~Singleton()                           = default;

    /// @brief get_instance(), as a reference.
    /// @return The instance.
    static T& get_ref()
    {
        return *get_instance();
    }

    /**
     * @brief Destroy the instance, if any; the next get_instance() recreates it.
     * @warning Only once NO thread can touch the instance any more, thread_local
     *          destructors included. Never from an atexit handler or a static
     *          destructor: exit-time destruction races those TLS destructors,
     *          which is why the instance is immortal. Earlier get_instance()
     *          pointers dangle afterwards.
     */
    static void delete_instance()
    {
        const std::scoped_lock lock(unique_instance_mutex);
        delete unique_instance.exchange(nullptr, std::memory_order_acq_rel);
    }

  private:
    /// IMMORTAL raw pointer, created once under the mutex and NEVER deleted.
    /// Do not make it a std::unique_ptr: its exit-time destructor would race
    /// worker-thread TLS destructors that still touch the singleton. Atomic so
    /// get_instance() can read it without the mutex (double-checked locking).
    static inline std::atomic<T*> unique_instance{nullptr};

    /// Guards lazy construction (without it two first users could both create
    /// the instance). Taken only while the instance does not exist yet:
    /// get_instance() is on a hot path, its steady state must not lock.
    static inline std::mutex unique_instance_mutex;

    /// @brief The instance, created on first use (double-checked locking).
    /// @return Never null.
    static T* get_instance()
    {
        // Fast path: one acquire load, no lock (reached by every thread at each
        // barrier). The acquire pairs with the release store below: seeing the
        // pointer implies seeing the constructed T.
        if (T* instance = unique_instance.load(std::memory_order_acquire)) [[likely]]
        {
            return instance;
        }

        const std::scoped_lock lock(unique_instance_mutex);
        T*                     instance = unique_instance.load(std::memory_order_relaxed); // under the lock
        if (instance == nullptr)
        {
            instance = new T();               // immortal: never deleted
            FAST_ALLOC_LSAN_IGNORE(instance); // intentional leak, not a real one
            unique_instance.store(instance, std::memory_order_release);
        }
        return instance;
    }
};

} // namespace fast_alloc
