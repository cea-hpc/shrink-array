#pragma once

#include <atomic> // std::atomic, std::memory_order
#include <mutex>  // std::mutex, std::scoped_lock

#if defined(__has_feature)
    #if __has_feature(address_sanitizer) || __has_feature(leak_sanitizer)
        #include <sanitizer/lsan_interface.h> // __lsan_ignore_object
        #define SA_LSAN_IGNORE(p) __lsan_ignore_object(p)
    #endif
#elifdef __SANITIZE_ADDRESS__
    #include <sanitizer/lsan_interface.h> // __lsan_ignore_object
    #define SA_LSAN_IGNORE(p) __lsan_ignore_object(p)
#endif

#ifndef SA_LSAN_IGNORE
    #define SA_LSAN_IGNORE(p) ((void)0)
#endif

/**
 * @brief Process-wide lazily-constructed instance of T, shared by every thread.
 * @details Created on the first getInstance() and never destroyed unless
 *          deleteInstance() is called: an intentional leak, marked for LSan.
 *          The chunk library reaches it as shrink_array::SingletonChk
 *          (chunk/singleton.h), an alias of this one.
 *          Nothing in shrink-array calls deleteInstance(); it is there for code
 *          outside it that wants an explicit teardown (see its contract).
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

    /// @brief getInstance(), as a reference.
    /// @return The instance.
    static T& getRef()
    {
        return *getInstance();
    }

    /**
     * @brief Destroy the instance, if any; the next getInstance() recreates it.
     * @warning Only once NO thread can touch the instance any more -- that
     *          includes thread_local destructors, which run at thread or
     *          process exit and may still reach it. Never from an atexit
     *          handler or a static destructor: the instance is immortal by
     *          design precisely because exit-time destruction raced those TLS
     *          destructors. A pointer returned by an earlier getInstance() is
     *          dangling afterwards.
     */
    static void deleteInstance()
    {
        const std::scoped_lock lock(unique_instance_mutex);
        delete unique_instance.exchange(nullptr, std::memory_order_acq_rel);
    }

  private:
    /// IMMORTAL raw pointer (was std::unique_ptr<T>): created once with `new`
    /// under the mutex and NEVER deleted. A global unique_ptr's exit-time
    /// destructor races worker-thread TLS destructors that still touch the
    /// singleton at shutdown (static/TLS destruction-order data race). Leaking
    /// avoids the exit-time destructor; the OS reclaims the memory at process
    /// end. Atomic so that getInstance() can read it without the mutex
    /// (double-checked locking, see there).
    static inline std::atomic<T*> unique_instance{nullptr};

    /// Guards lazy construction. A version with NO lock let two threads hitting
    /// getInstance() on first use both new the instance and the second
    /// assignment free the first -> double-free. Held only while the instance
    /// does not exist yet: getInstance() IS on a hot path (per-barrier
    /// bookkeeping, every thread), so its steady state must not lock.
    static inline std::mutex unique_instance_mutex;

    /// @brief The instance, created on first use (double-checked locking).
    /// @return Never null.
    static T* getInstance()
    {
        // Fast path: one acquire load, no lock: this is reached once per thread
        // per barrier, and a mutex here would serialise 64 threads at each one.
        // The acquire pairs with the release store below: seeing the pointer
        // implies seeing the constructed T.
        if (T* instance = unique_instance.load(std::memory_order_acquire)) [[likely]]
        {
            return instance;
        }

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
};