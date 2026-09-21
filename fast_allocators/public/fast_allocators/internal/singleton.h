#pragma once

#include <cstdlib>
//#include <memory>
#include <atomic>
#include <mutex>

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

template <typename T>
class Singleton
{
  protected:
    // Atomic: getInstance() reads it without the mutex (double-checked locking).
    static inline std::atomic<T*> unique_instance{nullptr};
    static inline std::mutex unique_instance_mutex;

  public:
    Singleton(const Singleton&)            = delete;
    Singleton& operator=(const Singleton&) = delete;

    static T* getInstance()
    {
        // Fast path: one acquire load, no lock. Every call used to take the
        // mutex, and this is reached once per thread per barrier (64 threads
        // serialised on it at each one). The acquire pairs with the release
        // store below: seeing the pointer implies seeing the constructed T.
        if (T* p = unique_instance.load(std::memory_order_acquire)) [[likely]]
            return p;

        std::lock_guard lock(unique_instance_mutex);
        T* p = unique_instance.load(std::memory_order_relaxed); // under the lock
        if (p == nullptr)
        {
            p = new T();       // immortal: never deleted
            SA_LSAN_IGNORE(p); // intentional leak, not a real one
            unique_instance.store(p, std::memory_order_release);
        }
        return p;
    }

    static void deleteInstance()
    {
        std::lock_guard lock(unique_instance_mutex);
        delete unique_instance.exchange(nullptr, std::memory_order_acq_rel);
    }

    static T& getRef()
    {
        return *getInstance();
    }
};