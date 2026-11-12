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
 *          (chunk_list_; for zero_any also the second-value and type-of-constant
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
 *          Freeing them is shared out the same way: a large mapped buffer
 *          gives its pages back from a team (release_mapped(), at the end of
 *          this file) before the munmap, which then drops an empty mapping.
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
#include <new>         // std::bad_alloc
#include <type_traits> // std::is_trivially_destructible_v
#include <utility>     // std::forward
#include <vector>      // std::vector
#include <sys/mman.h>  // mmap, munmap, madvise: the large per-chunk vectors
#include <unistd.h>    // sysconf(_SC_PAGESIZE)

#include "chunk/internal/PMP_CHUNK.h" // CHUNK_PMP_MAX_THREADS / CHUNK_PMP_IN_PARALLEL_REGION, pmp loops
#include "chunk/config.h"             // build_config::threaded

namespace shrink_array
{
namespace no_init_alloc_detail
{
/// @brief From this many bytes, NoInitAllocator maps the storage itself.
/// @details glibc serves an allocation below its mmap threshold from its heap,
///          and that threshold grows (up to 32 MiB) each time a mapped block is
///          freed: after a re-chunk freed one list, the next lists of up to
///          32 MiB went to the heap, and the heap kept them resident once freed
///          (zero_any_dyn at --divisor 10: ~580 MB still resident at the end of
///          the run that a fixed glibc threshold gave back). Mapped here, a
///          per-chunk vector's pages always return to the system when it is
///          freed. Below this size the heap's retention is negligible and a
///          mapping per vector not worth it.
inline constexpr std::size_t mapped_min_bytes = std::size_t{1} << 20U;

/// @brief The system page size, read once.
[[nodiscard]] inline std::size_t page_bytes() noexcept
{
    static const std::size_t page = []
    {
        const long size = ::sysconf(_SC_PAGESIZE);
        return size > 0 ? static_cast<std::size_t>(size) : std::size_t{4096};
    }();
    return page;
}

/// @brief @p bytes rounded up to whole pages: the length mapped and unmapped.
[[nodiscard]] inline std::size_t mapped_length(std::size_t bytes) noexcept
{
    const std::size_t page = page_bytes();
    return (bytes + page - 1) / page * page;
}

/// @brief Unmap a NoInitAllocator mapping of @p length bytes (whole pages); a
///        large one gives its pages back from a team first (defined at the end
///        of this file, after can_open_team()).
inline void release_mapped(void* ptr, std::size_t length) noexcept;
} // namespace no_init_alloc_detail

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

    /// @brief Whether @p n elements are mapped (mmap) rather than taken from the heap.
    /// @details Decided by the size alone, so deallocate() takes the same path as
    ///          allocate() did. The mapping is page-aligned; an element type
    ///          aligned beyond a page stays on the heap.
    [[nodiscard]] static bool mapped(std::size_t n) noexcept
    {
        return alignof(T) <= no_init_alloc_detail::page_bytes() &&
               n >= no_init_alloc_detail::mapped_min_bytes / sizeof(T);
    }

    /// @brief Storage for @p n elements: mapped from mapped_min_bytes on (see
    ///        no_init_alloc_detail::mapped_min_bytes), from the heap below.
    /// @details The pages are not touched here: the first write -- the caller's
    ///          init loop -- places them (NUMA), as glibc's own large blocks.
    /// @throws std::bad_alloc when the mapping fails.
    [[nodiscard]] T* allocate(std::size_t n)
    {
        if (!mapped(n))
        {
            return std::allocator<T>::allocate(n);
        }
        void* const ptr = ::mmap(nullptr, no_init_alloc_detail::mapped_length(n * sizeof(T)), PROT_READ | PROT_WRITE,
                                 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (ptr == MAP_FAILED)
        {
            throw std::bad_alloc{};
        }
        return static_cast<T*>(ptr);
    }

#if defined(__cpp_lib_allocate_at_least)
    /// @brief C++23: std::allocator's allocate_at_least() would take the heap
    ///        path whatever the size, and deallocate() then unmap a heap block.
    ///        Exactly @p n, through allocate().
    [[nodiscard]] std::allocation_result<T*, std::size_t> allocate_at_least(std::size_t n)
    {
        return {allocate(n), n};
    }
#endif

    /// @brief Release what allocate(@p n) returned: unmapped, its pages back to
    ///        the system at once, or returned to the heap.
    void deallocate(T* ptr, std::size_t n) noexcept
    {
        if (!mapped(n))
        {
            std::allocator<T>::deallocate(ptr, n);
            return;
        }
        no_init_alloc_detail::release_mapped(ptr, no_init_alloc_detail::mapped_length(n * sizeof(T)));
    }

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
    void construct(U* ptr, A0&& a0, Args&&... args)
    {
        std::construct_at(ptr, std::forward<A0>(a0), std::forward<Args>(args)...);
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

/// @brief Does a loop over @p count chunks take a team of its own? At least
///        parallel_init_min_chunks of them, and can_open_team().
inline bool use_own_team(long count) noexcept
{
    return count >= parallel_init_min_chunks && can_open_team();
}

/**
 * @brief Run @p init(i) for every chunk index i in [0, @p count): on a team of its
 *        own when use_own_team(@p count), on the calling thread otherwise.
 * @param count    chunk count
 * @param init callable `void(int)`; iterations must be independent
 */
template <typename Init>
void for_each_chunk(int count, Init&& init)
{
    if constexpr (build_config::threaded)
    {
        if (use_own_team(count))
        {
            pmp::parallel_for(0, count, pmp::Options{}, init);
            return;
        }
    }
    for (int i = 0; i < count; ++i)
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
 * @brief A vector of @p count elements, each constructed from @p args (none: value-
 *        initialised), by a team when it is large (construct_range).
 * @details For the atomic rows too: an atomic is constructed in place, never
 *          moved (the returned vector is moved whole, its buffer kept).
 */
template <typename T, typename... Args>
[[nodiscard]] NoInitVector<T> make_filled(std::size_t count, const Args&... args)
{
    static_assert(std::is_trivially_destructible_v<T>, "an unconstructed element must be safe to destroy");
    NoInitVector<T> vec(count); // allocates only (NoInitAllocator)
    construct_range(vec.data(), 0, count, [&](T* ptr) { std::construct_at(ptr, args...); });
    return vec;
}

/**
 * @brief Resize @p vec to @p count; the elements it gains are copies of @p value,
 *        constructed by a team when there are many (construct_range).
 * @details Keeps [0, min(old size, count)), as std::vector::resize(count, value). A
 *          reallocation copies the kept elements on the calling thread; a row
 *          filled from empty (every storage build) has none.
 */
template <typename T>
void resize_filled(NoInitVector<T>& vec, std::size_t count, const T& value)
{
    static_assert(std::is_trivially_destructible_v<T>, "an unconstructed element must be safe to destroy");
    const std::size_t held = vec.size();
    vec.resize(count); // the new tail is allocated, not constructed (NoInitAllocator)
    if (count > held)
    {
        construct_range(vec.data(), held, count, [&](T* ptr) { std::construct_at(ptr, value); });
    }
}

} // namespace shrink_array::parallel_storage_init

namespace shrink_array::no_init_alloc_detail
{
/// @brief From this many bytes a mapping gives its pages back from a team: the
///        list of a large array (a 10^9-element array at chunk size 64: about
///        15 M chunks, 125-250 MB). Below, one munmap costs less than a team.
inline constexpr std::size_t parallel_release_min_bytes = std::size_t{32} << 20U;

/// @brief The smallest slice a thread releases: a madvise call per few pages
///        would cost more than the pages.
inline constexpr std::size_t release_slice_min_bytes = std::size_t{2} << 20U;

/**
 * @details The munmap of a large mapping frees every page from the calling
 *          thread: the re-chunk's "list unmap" (0.87 s of 1.66 s of tmp
 *          teardown, 32 re-chunks at 160 threads). madvise(MADV_DONTNEED) on
 *          disjoint slices runs in parallel in the kernel; with the static
 *          schedule each thread releases the slice its rank touched first
 *          (construct_range), on its NUMA node. The munmap then drops an empty
 *          mapping. A team that cannot start (an exception from the backend):
 *          the munmap alone, which frees everything anyway.
 */
inline void release_mapped(void* ptr, std::size_t length) noexcept
{
    if constexpr (build_config::threaded)
    {
        if (length >= parallel_release_min_bytes && parallel_storage_init::can_open_team())
        {
            const auto        threads    = static_cast<std::size_t>(CHUNK_PMP_MAX_THREADS);
            const std::size_t max_slices = length / release_slice_min_bytes;
            const std::size_t slices     = threads < max_slices ? threads : max_slices;
            const std::size_t page       = page_bytes();
            const std::size_t per        = (length / slices + page - 1) / page * page;
            auto* const       base       = static_cast<char*>(ptr);
            try
            {
                pmp::parallel_for(0L, static_cast<long>(slices), pmp::Options{},
                                  [base, per, length](long s)
                                  {
                                      const std::size_t off = static_cast<std::size_t>(s) * per;
                                      if (off < length)
                                      {
                                          const std::size_t len = length - off < per ? length - off : per;
                                          (void)::madvise(base + off, len, MADV_DONTNEED);
                                      }
                                  });
            }
            catch (...) // NOLINT(bugprone-empty-catch): the munmap below releases it all
            {
            }
        }
    }
    (void)::munmap(ptr, length);
}
} // namespace shrink_array::no_init_alloc_detail
