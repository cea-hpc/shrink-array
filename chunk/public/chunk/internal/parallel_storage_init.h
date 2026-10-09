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
 * @details rebuild_storage() of the three variants sizes a handful of
 *          per-chunk vectors (chunk_list, and for Chunk2A the second-value
 *          and type-of-constant rows), then runs one init per chunk. The init
 *          loop was parallel; the sizing was not, and it was most of the time:
 *          measured on ArrayChunk2A<double> with 4.2 M chunks, 70 ms serial
 *          (chunk_list 40, the Proxy2A rows 30) against an 84 ms init loop
 *          that a team divides. Most of those 70 ms are page faults: a fresh
 *          vector of tens of MB is faulted in page by page by the one thread
 *          that value-initialises it -- which also puts every page on that
 *          thread's NUMA node.
 *
 *          The per-chunk vectors are no_init_vector: their allocator
 *          (NoInitAllocator) leaves a default-inserted element unconstructed,
 *          so a resize() or a vector(n) only allocates, and make_filled() /
 *          resize_filled() construct the new elements from a team, static
 *          schedule -- page faults, stores and NUMA placement all shared out
 *          where the static per-chunk loops will use them. The atomic rows
 *          (candidate flags, dirty bitset) are built the same way: an atomic
 *          is not movable, so a reserve-then-resize could not even compile for
 *          them. (The first version only faulted the pages in from a team and
 *          left the value-initialising resize -- a std::fill -- serial.)
 *          Same element types, and an allocator without state: not one byte
 *          more per chunk, nor in the vector itself.
 *
 *          for_each_chunk() is the init loop of the three variants, in one
 *          place: a team of its own outside a region for a large array, the
 *          calling thread otherwise -- never an orphaned `omp for`, which is
 *          wrong for an array built inside a region (an array created by each
 *          thread of a team gets 1/N of its chunks initialised; one created
 *          under `omp single` blocks on a barrier the others never reach).
 *          The rule -- use_own_team(): kParallelInitMinChunks or more chunks,
 *          outside a region, two threads or more -- is the library's only one:
 *          chunk_array_new::init_chunks (the chunks' operator new[]) is a
 *          wrapper of for_each_chunk(). It was a copy once, and the two had
 *          drifted: `>=` against `>`, and the array paths passing their own
 *          literals (500, and 16384 for Chunk2A, undocumented) rather than
 *          the constant.
 */

#include <cstddef>     // std::size_t
#include <memory>      // std::allocator, std::construct_at
#include <type_traits> // std::is_trivially_destructible_v
#include <utility>     // std::forward
#include <vector>      // std::vector

#include "chunk/PMP_CHUNK.h" // CHUNK_PMP_MAX_THREADS / CHUNK_PMP_IN_PARALLEL_REGION
#include "chunk/config.h"    // build_config::use_openmp

namespace shrink_array
{
/**
 * @brief std::allocator whose argument-less construct() does nothing.
 * @details A default-inserted element (resize(n), vector(n)) is left
 *          unconstructed: the caller constructs it, on the thread that will
 *          use it -- chunk_list's chunks in the init loops
 *          (ArrayChunkCore::construct_chunk), the per-chunk rows in
 *          parallel_storage_init::make_filled / resize_filled. Copies and moves
 *          (a reallocation) still construct, through the forwarding overload.
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
using no_init_vector = std::vector<T, NoInitAllocator<T>>;

// The allocator has no state: the vector is exactly a std::vector's size.
static_assert(sizeof(no_init_vector<char>) == sizeof(std::vector<char>));

/// @brief Below this many bytes a vector's elements are constructed by the
///        calling thread: a team costs more than the faults it would share.
inline constexpr std::size_t kFirstTouchMinBytes = std::size_t{2} << 20U;

/// @brief Chunk count from which a per-chunk initialisation takes a team of
///        its own (outside a region): below it, the calling thread is faster
///        than a team's start-up. Every per-chunk init loop of the library
///        goes through for_each_chunk(), so this is THE threshold.
inline constexpr int kParallelInitMinChunks = 500;

/// @brief May this call open a team of its own? Outside a parallel region,
///        with at least two threads available.
inline bool can_open_team() noexcept
{
    if constexpr (build_config::use_openmp)
    {
        return CHUNK_PMP_MAX_THREADS >= 2 && !CHUNK_PMP_IN_PARALLEL_REGION;
    }
    return false;
}

/// @brief Does a loop over @p n chunks take a team of its own? At least
///        kParallelInitMinChunks of them, and can_open_team().
inline bool use_own_team(long n) noexcept
{
    return n >= kParallelInitMinChunks && can_open_team();
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
    if constexpr (build_config::use_openmp)
    {
        if (use_own_team(n))
        {
#pragma omp parallel for schedule(static)
            for (int i = 0; i < n; ++i)
            {
                init(i);
            }
            return;
        }
    }
    for (int i = 0; i < n; ++i)
    {
        init(i);
    }
}

/**
 * @brief Construct elements [@p first, @p last) of @p base with @p make(p): on a
 *        team of its own, static schedule, for kFirstTouchMinBytes or more
 *        outside a region; on the calling thread otherwise.
 * @details The elements' storage is fresh (no_init_vector): this loop is its
 *          first touch, so the pages land with the threads of the static
 *          per-chunk loops.
 */
template <typename T, typename Make>
void construct_range(T* base, std::size_t first, std::size_t last, Make&& make)
{
    if constexpr (build_config::use_openmp)
    {
        if ((last - first) * sizeof(T) >= kFirstTouchMinBytes && can_open_team())
        {
            const long lo = static_cast<long>(first);
            const long hi = static_cast<long>(last);
#pragma omp parallel for schedule(static)
            for (long i = lo; i < hi; ++i)
            {
                make(base + i);
            }
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
[[nodiscard]] no_init_vector<T> make_filled(std::size_t n, const Args&... args)
{
    static_assert(std::is_trivially_destructible_v<T>, "an unconstructed element must be safe to destroy");
    no_init_vector<T> v(n); // allocates only (NoInitAllocator)
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
void resize_filled(no_init_vector<T>& v, std::size_t n, const T& value)
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
