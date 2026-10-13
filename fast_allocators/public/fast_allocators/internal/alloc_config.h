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
 * @file alloc_config.h
 * @brief `constexpr` mirrors of the macros that configure the fast allocators.
 *
 * The macros are read HERE and nowhere else, so the allocator bodies branch with
 * `if constexpr` instead of <tt>\#ifdef</tt>: every branch type-checks in every build.
 *
 * Mirrors chunk/config.h, deliberately separate: fast_allocators must not include chunk/.
 */

#include <charconv>     // std::from_chars (CHUNK_RECLAIM_MIN_SLABS)
#include <cstddef>      // std::size_t
#include <cstdio>       // std::fprintf
#include <cstdlib>      // std::getenv
#include <cstring>      // std::strlen
#include <system_error> // std::errc

#include <atomic>     // populate_scope_depth, populate_unsupported (CHUNK_POPULATE_SLABS)
#include <cerrno>     // errno == EINVAL: MADV_POPULATE_WRITE unknown to the kernel
#include <sys/mman.h> // madvise, MADV_HUGEPAGE / MADV_NOHUGEPAGE (advise_thp_region)

// glibc < 2.35 (RHEL 9) does not define it; the kernel has it since 5.14.
#if defined(__linux__) && !defined(MADV_POPULATE_WRITE)
    #define MADV_POPULATE_WRITE 23
#endif

#include "fast_allocators/build_options.h" // FAST_ALLOC_OPT_*: the options the library was built with

namespace fast_alloc::alloc_config
{

/// @brief Built against `new[]`/`delete[]` rather than the slab allocator
///        (-DFAST_ALLOC_STD_ALLOC: a header-only use without the library).
inline constexpr bool use_std_alloc =
#ifdef FAST_ALLOC_STD_ALLOC
    true;
#else
    false;
#endif

/// @brief Built with the slab machinery. The complement of #use_std_alloc.
inline constexpr bool use_slabs = !use_std_alloc;

/// @brief Built with OpenMP (build_options.h refuses a file compiled otherwise).
inline constexpr bool use_openmp = FAST_ALLOC_OPT_OPENMP != 0;

/// @brief Debug build (CMake: CHUNK_DEBUG, on with CMAKE_BUILD_TYPE=Debug).
inline constexpr bool is_debug = FAST_ALLOC_OPT_DEBUG != 0;

/// @brief The reinforced debug level of the slab lists: a debug build with
///        FAST_ALLOC_CHECK_LISTS (CMake option of the same name) walks a whole
///        list on every insert and remove (FAST_ALLOC_SL_LIST_CHECK in tc_allocator.h).
///        Linear per operation: a plain Debug build does not.
inline constexpr bool check_slab_lists = FAST_ALLOC_OPT_DEBUG != 0 && FAST_ALLOC_OPT_CHECK_LISTS != 0;

/**
 * @brief An emptied slab gets MADV_DONTNEED instead of staying resident.
 * @details On by default (CMake `CHUNK_RECLAIM_EMPTY_SLABS=OFF` turns it off), recorded in
 *          the generated build_options.h. Useful only once slabs reach nn_ref == 0 -- see
 *          ManagerChunk::drain_reuse_list_for_size (empties the old geometry's slabs).
 */
inline constexpr bool reclaim_empty_slabs = FAST_ALLOC_OPT_RECLAIM_EMPTY_SLABS != 0;

/**
 * @brief Fewest distinct empty slabs an EmptySlabSweep must hold for its
 *        flush() to advise them, read once from `CHUNK_RECLAIM_MIN_SLABS`. A re-chunk's
 *        sweep spans every rank, so the bar is the whole abandoned set's.
 *
 * @details One madvise per slab: the trade is decided by how many slabs a set empties at
 *          once (a re-chunk returns gigabytes for its syscalls, a teardown of a few slabs
 *          a few megabytes). The threshold keeps the first and drops the second. With
 *          sweeping off, this allocator is a few percent faster than the standard one at
 *          equal memory; sweeping trades that speed for pages.
 *
 *          0 (default when unset) sweeps everything. Read once, at a point that runs a
 *          handful of times per re-chunk. The free path must never madvise: a slab whose
 *          nn_ref briefly reads zero can still hold a live block.
 */
inline std::size_t reclaim_min_empty_slabs() noexcept
{
    static const std::size_t min_slabs = []() -> std::size_t
    {
        // The whole value must be a non-negative integer, else warning and 0 (the rule of
        // chunk/internal/env_parse.h, which this library cannot include).
        const char* env = std::getenv("CHUNK_RECLAIM_MIN_SLABS");
        if (env == nullptr || env[0] == '\0')
        {
            return 0;
        }
        std::size_t v   = 0;
        const char* end = env + std::strlen(env);
        const auto  res = std::from_chars(env, end, v);
        if (res.ec != std::errc{} || res.ptr != end)
        {
            std::fprintf(stderr, "[env] CHUNK_RECLAIM_MIN_SLABS=\"%s\" is not a valid value: ignored, 0 kept\n", env);
            return 0;
        }
        return v;
    }();
    return min_slabs;
}

/**
 * @brief CHUNK_THP: advise transparent huge pages on the slab arenas and the
 *        cst2 region? EXPERIMENTAL, off by default.
 *
 * @details Read once. `1` (any value but empty, `0` or `never`) asks for
 *          madvise(MADV_HUGEPAGE) on each arena reservation (SlabArena::grow) and the cst2
 *          region. Honoured only when the PMD huge page is no larger than a slab (2 MiB);
 *          with 64 KiB base pages it is 512 MiB, and the switch is ignored with a message.
 *          The kernel's THP mode must be `madvise` or `always`.
 *
 * @warning A slab becomes resident whole at its first touch. Measure the exact RSS
 *          (smaps_rollup) along with the time.
 */
inline bool thp_advise() noexcept
{
    static const bool on = []() -> bool
    {
        const char* env = std::getenv("CHUNK_THP");
        if (env == nullptr || env[0] == '\0' || env[0] == '0' || std::strcmp(env, "never") == 0)
        {
            return false; // `never`: thp_never(), below
        }
        long pmd = -1;
        if (std::FILE* f = std::fopen("/sys/kernel/mm/transparent_hugepage/hpage_pmd_size", "r"))
        {
            if (std::fscanf(f, "%ld", &pmd) != 1)
            {
                pmd = -1;
            }
            std::fclose(f);
        }
        constexpr long slab_bytes = 2L << 20;
        if (pmd <= 0 || pmd > slab_bytes)
        {
            if (pmd <= 0)
            {
                std::fprintf(stderr, "[env] CHUNK_THP=%s ignored: the huge page size is unknown here\n", env);
            }
            else
            {
                std::fprintf(stderr,
                             "[env] CHUNK_THP=%s ignored: the huge page is %ld MiB here, not a 2 MiB slab's size\n",
                             env, pmd >> 20);
            }
            return false;
        }
        std::fprintf(stderr, "[env] CHUNK_THP: MADV_HUGEPAGE on the slab arenas and the cst2 region (%ld KiB)\n",
                     pmd >> 10);
        return true;
    }();
    return on;
}

/**
 * @brief CHUNK_THP=never: madvise(MADV_NOHUGEPAGE) on the slab arenas and the
 *        ladder's mappings. Read once.
 *
 * @details Without advice the kernel's THP mode decides; under `always` (RHEL / AlmaLinux
 *          default) a 2 MiB-aligned slab is resident WHOLE at first touch (a 16-byte element
 *          slab uses only 512 KiB of it, the side-link cap), and khugepaged collapses partly
 *          touched ranges. `never` keeps base pages. The default stays the kernel's until
 *          measured (RSS, AnonHugePages in smaps_rollup, time).
 */
inline bool thp_never() noexcept
{
    static const bool on = []() -> bool
    {
        const char* env = std::getenv("CHUNK_THP");
        if (env == nullptr || std::strcmp(env, "never") != 0)
        {
            return false;
        }
#ifdef MADV_NOHUGEPAGE
        std::fprintf(stderr, "[env] CHUNK_THP=never: MADV_NOHUGEPAGE on the slab arenas\n");
        return true;
#else
        std::fprintf(stderr, "[env] CHUNK_THP=never ignored: no MADV_NOHUGEPAGE on this system\n");
        return false;
#endif
    }();
    return on;
}

/**
 * @brief Apply the CHUNK_THP advice to a fresh slab mapping: MADV_HUGEPAGE
 *        (`1`), MADV_NOHUGEPAGE (`never`), or nothing (unset: the kernel's
 *        mode). Advice only: a failure is ignored.
 */
inline void advise_thp_region([[maybe_unused]] void* base, [[maybe_unused]] std::size_t len) noexcept
{
#ifdef MADV_HUGEPAGE
    if (thp_advise())
    {
        (void)::madvise(base, len, MADV_HUGEPAGE);
        return;
    }
#endif
#ifdef MADV_NOHUGEPAGE
    if (thp_never())
    {
        (void)::madvise(base, len, MADV_NOHUGEPAGE);
    }
#endif
}

/**
 * @brief CHUNK_POPULATE_SLABS: pre-fault a fresh value slab with ONE
 *        madvise(MADV_POPULATE_WRITE) instead of one page fault per page. Read once.
 * @return 0 = off; 1 (default) = only inside a PopulateScope (the re-chunk
 *         refill, which fills the slabs it takes anyway); 2 = every fresh value slab.
 *
 * @details Every page fault takes the process' mmap_lock. With threads on both
 *          sockets of a Grace node its cache line bounces between them: at 128
 *          threads (64 KiB pages, kernel 5.14) a fault costs ~50 us of system time
 *          instead of ~22 us at 64 -- two processes x 64 threads stay at ~22 us, so
 *          the structure is per process. One populate per 1 MiB slab takes the lock
 *          once for its 16 pages and stays at ~20 us per page at 128 threads
 *          (fault_probe_slab.sh). Cost: the slab is resident whole at once -- with
 *          `1`, only the last, partly filled slab of each rank and size of the
 *          array being rebuilt; with `2`, that of every (rank, array, size).
 *          A kernel without MADV_POPULATE_WRITE (EINVAL) turns it off, once.
 */
inline int populate_slabs_mode() noexcept
{
    static const int mode = []() -> int
    {
        const char* env = std::getenv("CHUNK_POPULATE_SLABS");
        if (env == nullptr || env[0] == '\0')
        {
            return 1;
        }
        const int m = (env[0] == '0') ? 0 : (env[0] == '2') ? 2 : 1;
        std::fprintf(stderr, "[env] CHUNK_POPULATE_SLABS=%d (%s)\n", m,
                     m == 0 ? "off" : m == 1 ? "re-chunk refill only" : "every fresh value slab");
        return m;
    }();
    return mode;
}

/// @brief Number of open PopulateScope (mode 1 populates only while it is > 0).
///        Opened and closed by the serial code around a parallel region: the
///        region's fork and join order it with the slabs' allocations.
inline std::atomic<int> populate_scope_depth{0};

/// @brief Set once the kernel refused MADV_POPULATE_WRITE: no further attempt.
inline std::atomic<bool> populate_unsupported{false};

/// @brief RAII: the fresh value slabs taken while it lives are pre-faulted
///        (CHUNK_POPULATE_SLABS=1, the default). Serial code only.
struct PopulateScope
{
    PopulateScope() noexcept
    {
        populate_scope_depth.fetch_add(1, std::memory_order_relaxed);
    }
    ~PopulateScope()
    {
        populate_scope_depth.fetch_sub(1, std::memory_order_relaxed);
    }
    PopulateScope(const PopulateScope&)            = delete;
    PopulateScope& operator=(const PopulateScope&) = delete;
};

/**
 * @brief Pre-fault a fresh slab [@p base, @p base + @p len) if CHUNK_POPULATE_SLABS
 *        asks for it here. Advice only: any failure is ignored (the pages then
 *        fault in one by one, as without it).
 * @pre Called OUTSIDE the arena mutex: the populate is the expensive part, and
 *      every thread of a refill populates its own slabs at once.
 */
inline void populate_fresh_slab([[maybe_unused]] void* base, [[maybe_unused]] std::size_t len) noexcept
{
#ifdef MADV_POPULATE_WRITE
    const int mode = populate_slabs_mode();
    if (mode == 0 || (mode == 1 && populate_scope_depth.load(std::memory_order_relaxed) == 0) ||
        populate_unsupported.load(std::memory_order_relaxed))
    {
        return;
    }
    if (::madvise(base, len, MADV_POPULATE_WRITE) != 0 && errno == EINVAL)
    {
        if (!populate_unsupported.exchange(true, std::memory_order_relaxed))
        {
            std::fprintf(stderr, "[env] CHUNK_POPULATE_SLABS: MADV_POPULATE_WRITE refused by the kernel "
                                 "(older than 5.14?) -- off\n");
        }
    }
#endif
}

/// @brief int_alloc / int_free go through the aligned allocation (-DFAST_ALLOC_ALIGN).
inline constexpr bool alloc_align =
#ifdef FAST_ALLOC_ALIGN
    true;
#else
    false;
#endif

} // namespace fast_alloc::alloc_config
