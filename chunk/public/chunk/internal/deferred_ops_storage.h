#pragma once

#include <vector>
#include <atomic>
#include <memory>
#include <mutex>
#include <utility>           // std::move
#include "chunk/PMP_CHUNK.h" // CHUNK_PMP_* macros — used HERE, not via includers

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
     * Uses 'static inline' (C++17) to ensure the instance is initialized
     * at program load time, avoiding the "Magic Static" lock check.
     */
    static DeferredOpsStorage& instance() noexcept
    {
        return global_instance;
    }

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
     * Double-checked locking on arr_raw. The previous implementation used
     * std::call_once with a std::unique_ptr<std::once_flag>; because that
     * unique_ptr had a DYNAMIC default-member-initializer (make_unique) on a
     * static-storage singleton (global_instance), a static-initialization-order
     * fiasco was possible: if allocate() ran before global_instance's dynamic
     * init constructed `flag`, `*flag` dereferenced a null unique_ptr (observed as
     * a SIGABRT <tt>unique_ptr\<once_flag\>\::operator\* Assertion get()!=pointer()</tt>
     * under nvc++). std::mutex has a constexpr constructor, so alloc_mutex is
     * CONSTANT-initialized — global_instance no longer needs any dynamic init,
     * removing the ordering hazard on every compiler.
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
    DeferredOpsStorage() = default;

    // Static inline ensures the singleton object itself exists for the entire
    // duration of the program. All members are constant-initializable
    // (constexpr mutex/atomic ctors, empty std::vector -- constexpr in C++20), so global_instance needs
    // NO dynamic initialization — no static-init-order hazard.
    static inline DeferredOpsStorage global_instance;

    std::vector<per_thread_map> arr_owner;

    // Atomic pointer used by worker threads for O(1) access
    alignas(64) std::atomic<per_thread_map*> arr_raw{nullptr};

    // Guards allocate()/destroy(). constexpr ctor -> constant-initialized.
    std::mutex alloc_mutex;
};

} // namespace shrink_array