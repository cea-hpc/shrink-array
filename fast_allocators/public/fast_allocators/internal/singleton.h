#pragma once

#include <atomic>
#include <mutex>

#if defined(__has_feature)
    #if __has_feature(address_sanitizer) || __has_feature(leak_sanitizer)
        #include <sanitizer/lsan_interface.h>
        #define SA_LSAN_IGNORE(p) __lsan_ignore_object(p)
    #endif
#elifdef __SANITIZE_ADDRESS__
    #include <sanitizer/lsan_interface.h>
    #define SA_LSAN_IGNORE(p) __lsan_ignore_object(p)
#endif

#ifndef SA_LSAN_IGNORE
    #define SA_LSAN_IGNORE(p) ((void)0)
#endif

/**
 * @brief Process-wide lazily-constructed instance of T, shared by every thread.
 * @details Created on the first getInstance() and never destroyed unless
 *          deleteInstance() is called: an intentional leak, marked for LSan.
 * @tparam T The type to host; default-constructible.
 */
template <typename T>
class Singleton
{
  public:
    Singleton(const Singleton&)            = delete;
    Singleton& operator=(const Singleton&) = delete;

    /// @brief getInstance(), as a reference.
    /// @return The instance.
    static T& getRef()
    {
        return *getInstance();
    }

  private:
    /// @brief The instance, created on first use (double-checked locking).
    /// @return Never null.
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

    /// @brief Destroy the instance, if any; the next getInstance() recreates it.
    static void deleteInstance()
    {
        const std::scoped_lock lock(unique_instance_mutex);
        delete unique_instance.exchange(nullptr, std::memory_order_acq_rel);
    }

  protected:
    // Atomic: getInstance() reads it without the mutex (double-checked locking).
    static inline std::atomic<T*> unique_instance{nullptr};
    static inline std::mutex      unique_instance_mutex;
};