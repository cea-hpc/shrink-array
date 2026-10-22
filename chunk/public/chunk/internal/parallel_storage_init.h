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
/**
 * @file parallel_storage_init.h
 * @brief The two parallel pieces of an array's storage (re)build:
 *        the construction of its per-chunk vectors, and the per-chunk
 *        initialisation loop.
 *
 * @details rebuild_storage() of the three variants sizes per-chunk vectors
 *          (chunk_list; for Chunk2A also the second-value and type-of-constant
 *          rows), then runs one init per chunk. Sizing a fresh vector serially
 *          costs as much as the init loop itself, mostly page faults, and puts
 *          every page on one NUMA node.
 *
 *          So the per-chunk vectors are NoInitVector: NoInitAllocator leaves
 *          a default-inserted element unconstructed, and make_filled() /
 *          resize_filled() construct the new elements from a team, static
 *          schedule (pmp::parallel_for), where the per-chunk loops will use them. The atomic rows
 *          are built the same way (an atomic is not movable). No extra byte per
 *          chunk nor in the vector.
 *
 *          for_each_chunk() is the init loop of the three variants: a team of
 *          its own outside a region for a large array, the calling thread
 *          otherwise. Never an orphaned `omp for`: an array built by each
 *          thread of a team would get 1/N of its chunks initialised, and one
 *          built under `omp single` would block on a barrier. use_own_team() is
 *          the library's only threshold rule; chunk_array_new::init_chunks
 *          wraps for_each_chunk(), and for_each_chunk_in() is its form with a
 *          per-thread state (the release of a shrinking resize). Do not pass
 *          local literals instead.
 */

#include <cstddef>     // std::size_t
#include <memory>      // std::allocator, std::construct_at
#include <type_traits> // std::is_trivially_destructible_v
#include <utility>     // std::forward
#include <vector>      // std::vector

#include "chunk/PMP_CHUNK.h" // CHUNK_PMP_MAX_THREADS / CHUNK_PMP_IN_PARALLEL_REGION, pmp loops
#include "chunk/config.h"    // build_config::threaded

namespace shrink_array
{
/**
 * @brief std::allocator whose argument-less construct() does nothing.
 * @details A default-inserted element is left unconstructed, for the thread
 *          that will use it (ArrayChunkCore::construct_chunk, make_filled,
 *          resize_filled). Copies and moves still construct.
 *
 *          Contract: every element a growing resize() or a vector(n) adds is
 *          constructed before any other use.
 * @tparam T the element type
 */
template <typename T>
struct NoInitAllocator : std::allocator<T>
{
    using value_type = T; ///< Element type.

    NoInitAllocator() noexcept = default;
    /// @brief Rebinding copy (allocator requirements).
    template <typename U>
    NoInitAllocator(const NoInitAllocator<U>& /*other*/) noexcept // NOLINT(google-explicit-constructor)
    {
    }

    /// @brief Default construction: left to the caller.
    template <typename U>
    void construct(U* /*p*/) noexcept
    {
    }

    /// @brief Any other construction (copy, move): as std::allocator.
    template <typename U, typename A0, typename... Args>
    void construct(U* p, A0&& a0, Args&&... args)
    {
        std::construct_at(p, std::forward<A0>(a0), std::forward<Args>(args)...);
    }
};
} // namespace shrink_array

namespace shrink_array::parallel_storage_init
{

/// @brief A per-chunk vector whose new elements the helpers below construct.
template <typename T>
using NoInitVector = std::vector<T, NoInitAllocator<T>>;

// The allocator has no state: the vector is exactly a std::vector's size.
static_assert(sizeof(NoInitVector<char>) == sizeof(std::vector<char>));

/// @brief Below this many bytes a vector's elements are constructed by the
///        calling thread: a team costs more than the faults it would share.
inline constexpr std::size_t first_touch_min_bytes = std::size_t{2} << 20U;

/// @brief Chunk count from which a per-chunk init takes a team of its own:
///        below it, a team's start-up costs more. THE library threshold.
inline constexpr int parallel_init_min_chunks = 500;

/// @brief May this call open a team of its own? Outside a parallel region,
///        with at least two threads available.
inline bool can_open_team() noexcept
{
    if constexpr (build_config::threaded)
    {
        return CHUNK_PMP_MAX_THREADS >= 2 && !CHUNK_PMP_IN_PARALLEL_REGION;
    }
    return false;
}

/// @brief Does a loop over @p n chunks take a team of its own? At least
///        parallel_init_min_chunks of them, and can_open_team().
inline bool use_own_team(long n) noexcept
{
    return n >= parallel_init_min_chunks && can_open_team();
}

/**
 * @brief Run @p init(i) for every chunk index i in [0, @p n): on a team of its
 *        own when use_own_team(@p n), on the calling thread otherwise.
 * @param n    chunk count
 * @param init callable `void(int)`; iterations must be independent
 */
template <typename Init>
void for_each_chunk(int n, Init&& init)
{
    if constexpr (build_config::threaded)
    {
        if (use_own_team(n))
        {
            pmp::parallel_for(0, n, pmp::Options{}, init);
            return;
        }
    }
    for (int i = 0; i < n; ++i)
    {
        init(i);
    }
}

/**
 * @brief Run @p body(j, ctx) for every chunk index j in [@p first, @p last), with
 *        ctx = @p make_ctx() taken once per thread: on a team of its own when
 *        use_own_team(last - first), on the calling thread otherwise.
 * @details For the loops whose threads each need their own state -- the
 *          release of a shrink parks every buffer into the reuse list of the
 *          rank that releases it, as clear_core()'s team does.
 * @param first    first chunk index
 * @param last     one past the last chunk index
 * @param make_ctx callable `Ctx&()` (or returning a value), called once per thread
 * @param body     callable `void(int, Ctx&)`; iterations must be independent
 */
template <typename MakeCtx, typename Body>
void for_each_chunk_in(int first, int last, MakeCtx&& make_ctx, Body&& body)
{
    if (first >= last)
    {
        return;
    }
    if constexpr (build_config::threaded)
    {
        if (use_own_team(static_cast<long>(last) - first))
        {
            pmp::parallel_for_local(first, last, pmp::Options{}, make_ctx, body);
            return;
        }
    }
    decltype(auto) ctx = make_ctx();
    for (int j = first; j < last; ++j)
    {
        body(j, ctx);
    }
}

/**
 * @brief Construct elements [@p first, @p last) of @p base with @p make(p): on a
 *        team of its own, static schedule, for first_touch_min_bytes or more
 *        outside a region; on the calling thread otherwise.
 * @details First touch of fresh storage: the pages land with the threads of
 *          the static per-chunk loops.
 */
template <typename T, typename Make>
void construct_range(T* base, std::size_t first, std::size_t last, Make&& make)
{
    if constexpr (build_config::threaded)
    {
        if ((last - first) * sizeof(T) >= first_touch_min_bytes && can_open_team())
        {
            pmp::parallel_for(static_cast<long>(first), static_cast<long>(last), pmp::Options{},
                              [base, &make](long i) { make(base + i); });
            return;
        }
    }
    for (std::size_t i = first; i < last; ++i)
    {
        make(base + i);
    }
}

/**
 * @brief A vector of @p n elements, each constructed from @p args (none: value-
 *        initialised), by a team when it is large (construct_range).
 * @details For the atomic rows too: an atomic is constructed in place, never
 *          moved (the returned vector is moved whole, its buffer kept).
 */
template <typename T, typename... Args>
[[nodiscard]] NoInitVector<T> make_filled(std::size_t n, const Args&... args)
{
    static_assert(std::is_trivially_destructible_v<T>, "an unconstructed element must be safe to destroy");
    NoInitVector<T> v(n); // allocates only (NoInitAllocator)
    construct_range(v.data(), 0, n, [&](T* p) { std::construct_at(p, args...); });
    return v;
}

/**
 * @brief Resize @p v to @p n; the elements it gains are copies of @p value,
 *        constructed by a team when there are many (construct_range).
 * @details Keeps [0, min(old size, n)), as std::vector::resize(n, value). A
 *          reallocation copies the kept elements on the calling thread; a row
 *          filled from empty (every storage build) has none.
 */
template <typename T>
void resize_filled(NoInitVector<T>& v, std::size_t n, const T& value)
{
    static_assert(std::is_trivially_destructible_v<T>, "an unconstructed element must be safe to destroy");
    const std::size_t held = v.size();
    v.resize(n); // the new tail is allocated, not constructed (NoInitAllocator)
    if (n > held)
    {
        construct_range(v.data(), held, n, [&](T* p) { std::construct_at(p, value); });
    }
}

} // namespace shrink_array::parallel_storage_init
