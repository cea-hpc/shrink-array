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

/**
 * @file baseline_vector.h
 * @brief The std::vector of the baseline build, zeroed in parallel.
 *
 * `std::vector<T> v(n)` value-initialises its n elements: one thread writes
 * every byte. That costs the creation time, and it places every page on that
 * thread's NUMA node, so the parallel phases afterwards read remote memory on
 * a multi-node machine -- a cost the chunked arrays, which allocate nothing at
 * creation, do not pay. BaselineVector allocates without writing
 * (default_init_allocator), then zeroes the elements in a parallel loop with
 * the static schedule of the bench's loops: each page is first touched by the
 * thread that works on it.
 *
 * The elements read T{} after construction and after a growing resize(n),
 * exactly as with std::vector: only who writes the zeros changes.
 */

#pragma once

#include <cstddef>     // std::size_t, std::ptrdiff_t
#include <memory>      // std::allocator, std::allocator_traits
#include <new>         // placement new
#include <type_traits> // std::is_trivially_default_constructible_v, std::is_same_v, std::is_nothrow_default_constructible_v
#include <utility>     // std::forward
#include <vector>      // std::vector: BaselineVector's base

namespace shrink_array
{

/**
 * @brief std::allocator, except that a construct() without arguments
 *        default-initialises: a trivial element is left unwritten.
 *
 * Every other construct() (a value, a copy, emplace arguments) goes to the
 * underlying allocator unchanged.
 * @warning A container using it holds indeterminate values until written:
 *          reading one is undefined behaviour. BaselineVector writes them.
 */
template <class T, class A = std::allocator<T>>
class default_init_allocator : public A
{
    using traits = std::allocator_traits<A>;

  public:
    template <class U>
    struct rebind
    {
        using other = default_init_allocator<U, typename traits::template rebind_alloc<U>>;
    };

    using A::A;

    /// @brief Default-initialise *p: no write for a trivial U.
    template <class U>
    void construct(U* p) noexcept(std::is_nothrow_default_constructible_v<U>)
    {
        ::new (static_cast<void*>(p)) U;
    }

    /// @brief Construct *p from @p args, as the underlying allocator does.
    template <class U, class... Args>
    void construct(U* p, Args&&... args)
    {
        traits::construct(static_cast<A&>(*this), p, std::forward<Args>(args)...);
    }
};

/**
 * @brief std::vector whose sized construction and growing resize zero the
 *        new elements in parallel, each page first touched by the thread of
 *        the static schedule that works on it.
 *
 * Only for a trivially default-constructible T; any other T (std::complex,
 * whose default constructor writes the zero itself) is initialised by its
 * constructor, sequentially, as std::vector does. bool keeps
 * std::vector<bool>, packed bits that it zeroes itself and that two threads
 * cannot write side by side.
 */
template <class T>
class BaselineVector : public std::vector<T, default_init_allocator<T>>
{
    using base = std::vector<T, default_init_allocator<T>>;

  public:
    using base::base;
    using base::resize;
    using typename base::size_type;

    BaselineVector() = default;

    /// @brief @p n elements reading T{}, zeroed in parallel.
    explicit BaselineVector(size_type n) : base(n)
    {
        zero_from(0);
    }

    /// @brief Resize to @p n; new elements read T{}, zeroed in parallel.
    void resize(size_type n)
    {
        const size_type old = this->size();
        base::resize(n);
        if (n > old)
        {
            zero_from(old);
        }
    }

  private:
    /// Below this many elements a parallel region costs more than it saves.
    static constexpr std::size_t parallel_min = std::size_t{1} << 16;

    /// @brief Write T{} into [first, size()).
    void zero_from(size_type first)
    {
        if constexpr (std::is_trivially_default_constructible_v<T> && !std::is_same_v<T, bool>)
        {
            T* const             p = this->data();
            const std::ptrdiff_t b = static_cast<std::ptrdiff_t>(first);
            const std::ptrdiff_t e = static_cast<std::ptrdiff_t>(this->size());
#pragma omp parallel for schedule(static) if (static_cast<std::size_t>(e - b) >= parallel_min)
            for (std::ptrdiff_t i = b; i < e; ++i)
            {
                p[i] = T{};
            }
        }
    }
};

} // namespace shrink_array
