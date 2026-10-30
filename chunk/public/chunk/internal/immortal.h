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

// Storage for the library's static objects that an array can reach from its
// constructor or its destructor: usable before any dynamic initialisation has
// run, and never destroyed.
//
// Why. A static data member of a class template (ManagerChunk<..., T>::...,
// zero_any::Proxy<T>::..., ChunkBase<T, ...>::...) has UNORDERED dynamic
// initialisation, and is destroyed at exit in the reverse order of the end of
// its initialisation. A global array is built during static initialisation and
// destroyed after main returns: with plain statics it can reach a member not yet
// constructed (in its constructor) or already destroyed (in its destructor).
//
//   Immortal<T>      for a T whose default constructor is constexpr
//                    (std::vector, std::array...): constant-initialised, so it
//                    exists before any code runs; never destroyed. Declared
//                    with a reference of the old name next to it, the access
//                    is unchanged and costs nothing:
//                        static inline constinit Immortal<V> x_storage_{};
//                        static inline V& x = x_storage_.value;
//   LazyImmortal<T>  for the others (std::map, std::unordered_map, the pointer
//                    maps...): constant-initialised empty storage, the T built
//                    in place at the first access, never destroyed. Reached
//                    through -> or *: one acquire load and a predicted branch.
//
// Never destroyed is the existing discipline of the allocator arenas
// (slab_arena.h, static_const_array.h): the memory lives until the process
// exits anyway.

#include <atomic> // std::atomic (LazyImmortal's ready flag)
#include <memory> // std::construct_at
#include <mutex>  // std::mutex, std::scoped_lock (LazyImmortal's construction)
#include <new>    // std::launder

namespace shrink_array::internal
{

/// @brief Constant-initialised, never-destroyed storage for a T with a
///        constexpr default constructor (see the top of this file).
template <typename T>
union Immortal
{
    constexpr Immortal() : value() {}
    // Never destroys value: the point of the type.
    constexpr ~Immortal() {} // NOLINT(modernize-use-equals-default): = default would destroy value

    Immortal(const Immortal&) = delete ("never-destroyed static storage: copy its value member, not the storage");
    Immortal&
    operator=(const Immortal&)      = delete ("never-destroyed static storage: copy its value member, not the storage");
    Immortal(Immortal&&)            = delete ("never-destroyed static storage: copy its value member, not the storage");
    Immortal& operator=(Immortal&&) = delete ("never-destroyed static storage: copy its value member, not the storage");

    T value; ///< The object.
};

/// @brief Constant-initialised storage, the T built at the first access and
///        never destroyed (see the top of this file).
template <typename T>
class LazyImmortal
{
  public:
    constexpr LazyImmortal() noexcept = default;
    ~LazyImmortal()                   = default; // never destroys the T (see the top of this file)

    LazyImmortal(const LazyImmortal&) =
        delete ("never-destroyed storage built at the first access: copy get(), not the storage");
    LazyImmortal& operator=(const LazyImmortal&) =
        delete ("never-destroyed storage built at the first access: copy get(), not the storage");
    LazyImmortal(LazyImmortal&&) =
        delete ("never-destroyed storage built at the first access: copy get(), not the storage");
    LazyImmortal& operator=(LazyImmortal&&) =
        delete ("never-destroyed storage built at the first access: copy get(), not the storage");

    /// @return The object, built now if this is the first access.
    [[nodiscard]] T& get()
    {
        if (!ready_.load(std::memory_order_acquire)) [[unlikely]]
        {
            return construct();
        }
        return *object();
    }
    T* operator->()
    {
        return &get();
    }
    T& operator*()
    {
        return get();
    }

  private:
    T* object() noexcept
    {
        return std::launder(reinterpret_cast<T*>(storage_)); // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
    }

    T& construct()
    {
        const std::scoped_lock lock(mutex_);
        if (!ready_.load(std::memory_order_relaxed))
        {
            std::construct_at(reinterpret_cast<T*>(storage_)); // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
            ready_.store(true, std::memory_order_release);
        }
        return *object();
    }

    alignas(T) unsigned char storage_[sizeof(T)]{}; // NOLINT(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
    std::atomic<bool> ready_{false};
    std::mutex        mutex_;
};

} // namespace shrink_array::internal
