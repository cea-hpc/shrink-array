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

template <typename T>
class SingletonChk
{
  protected:
    // IMMORTAL raw pointer (was std::unique_ptr<T>): created once with `new`
    // under the mutex and NEVER deleted. Rationale identical to the
    // fast_allocators Singleton: a global unique_ptr's exit-time destructor
    // races worker-thread TLS destructors that still touch the singleton at
    // shutdown (static/TLS destruction-order data race). Leaking avoids the
    // exit-time destructor; the OS reclaims the memory at process end.
    static inline T* unique_instance{nullptr};

    // Guards lazy construction. The previous version had NO lock, so two threads
    // hitting getInstance() on first use could both new the instance and the
    // second assignment free the first → double-free. getInstance is only hit
    // once per type at first use, never on a hot path, so the lock is free.
    static inline std::mutex unique_instance_mutex;

  public:
    SingletonChk(const SingletonChk&)            = delete;
    SingletonChk& operator=(const SingletonChk&) = delete;

    static T* getInstance()
    {
        std::lock_guard lock(unique_instance_mutex);
        if (unique_instance == nullptr)
        {
            unique_instance = new T();       // immortal: never deleted
            SA_LSAN_IGNORE(unique_instance); // intentional leak, not a real one
        }

        return unique_instance;
    }

    /// @brief Delete the instance. Retained for API completeness; unused
    ///        under the immortal design (getInstance() never deletes).
    static void deleteInstance()
    {
        std::lock_guard lock(unique_instance_mutex);
        delete unique_instance;
        unique_instance = nullptr;
    }

    /// @brief getInstance(), as a reference.
    static T& getRef()
    {
        return *getInstance();
    }
};

template <typename T>
void deleteInstance()
{
    SingletonChk<T>::deleteInstance();
}

} // namespace shrink_array