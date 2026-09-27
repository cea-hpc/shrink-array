#pragma once

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
 * @file alloc_config.h
 * @brief `constexpr` mirrors of the macros that configure the fast allocators.
 *
 * The macros are read HERE and nowhere else, so the allocator bodies can branch
 * with `if constexpr` instead of `#ifdef`. Both arms then have to compile in
 * every configuration, which is the point: a build that does not exercise a
 * branch still type-checks it.
 *
 * This mirrors chunk/config.h for the chunk layer. It is deliberately separate:
 * fast_allocators does not include anything from chunk/, and must not start to.
 */

#include <charconv>     // std::from_chars (CHUNK_RECLAIM_MIN_SLABS)
#include <cstdio>       // std::fprintf
#include <cstdlib>      // std::getenv
#include <cstring>      // std::strlen
#include <system_error> // std::errc
#include <cstddef>      // std::size_t
namespace alloc_config
{

/// @brief Built against `new[]`/`delete[]` rather than the slab allocator.
inline constexpr bool use_std_alloc =
#ifdef STD_ALLOC
    true;
#else
    false;
#endif

/// @brief Built with the slab machinery. The complement of #use_std_alloc.
inline constexpr bool use_slabs = !use_std_alloc;

/// @brief Compiled with OpenMP.
inline constexpr bool use_openmp =
#ifdef _OPENMP
    true;
#else
    false;
#endif

/// @brief Debug build.
inline constexpr bool is_debug =
#ifdef DEBUG
    true;
#else
    false;
#endif

/**
 * @brief An emptied slab gets MADV_DONTNEED instead of staying resident.
 * @details CMake option `CHUNK_RECLAIM_EMPTY_SLABS`, off by default. Only worth
 *          anything once slabs actually reach nn_ref == 0 -- see
 *          ManagerChunk::drain_reuse_list_for_size, which is what empties the
 *          old geometry's slabs after a re-chunk.
 */
inline constexpr bool reclaim_empty_slabs =
#if defined(CHUNK_RECLAIM_EMPTY_SLABS)
    true;
#else
    false;
#endif

/**
 * @brief Fewest empty slabs a set must hold for reclaim_empty_slabs_now() to
 *        sweep it, read once from `CHUNK_RECLAIM_MIN_SLABS`.
 *
 * @details A sweep costs one madvise per slab -- a 2000-iteration bench run
 *          made 82708 of them -- and returns that slab's pages. The two scale
 *          together per slab, so the trade is decided by how many slabs a set
 *          empties at once: a re-chunk that empties 5887 returns gigabytes for
 *          its syscalls, a teardown that empties three returns three megabytes
 *          for the same price each. The threshold keeps the first and drops the
 *          second.
 *
 *          Measured (chunk2a static 256, 2000 iterations, 64 threads, one seed):
 *          the standard allocator runs 12.59 s at 4350 MB, this one 11.71 s at
 *          4392 MB with sweeping off -- 7% faster at equal memory, and it is the
 *          sweeping that trades that speed back for pages. Before this the
 *          choice needed a rebuild.
 *
 *          0 sweeps everything (what CHUNK_RECLAIM_EMPTY_SLABS=ON always did),
 *          which is the default when unset, so existing builds do not change.
 *          Read once, at a point that runs a handful of times per re-chunk:
 *          the free path is deliberately not a consumer of this -- it stopped
 *          madvising anything after a slab whose nn_ref briefly read zero had
 *          a live block's page dropped under it.
 */
inline std::size_t reclaim_min_empty_slabs() noexcept
{
    static const std::size_t min_slabs = []() -> std::size_t
    {
        // The whole value must be a non-negative integer (std::from_chars:
        // a '-', trailing text or an overflow is refused, with a warning, and
        // 0 kept) -- the rule of chunk/internal/env_parse.h, which this
        // library cannot include. strtol's result used to be clamped silently.
        const char* env = std::getenv("CHUNK_RECLAIM_MIN_SLABS");
        if (env == nullptr || env[0] == '\0')
            return 0;
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
 * @brief Is the `__debug_mem` global consulted at run time?
 *
 * @details `__debug_mem` routes every allocation through plain `new[]`/`free`
 *          so a memory checker sees individual objects rather than one slab.
 *          Reading it costs a branch on the hot path, so release serial builds
 *          do not read it at all and behave as if it were false.
 *
 * @warning Nothing in this repository ever assigns `__debug_mem`; it is
 *          declared `extern` and initialised to false, so it is reachable only
 *          by a translation unit outside the library. StaticConstArray honours
 *          it symmetrically on both ladders (getBlockAlloc / freeBlock and the
 *          aligned pair; getBlockAlloc ignored it until Sept. '26, so a free
 *          could int_free() a slab block), but
 *          TCTypeCAllocator does NOT: its free_elt() honours it while its
 *          alloc() ignores it, so setting the flag on an already-populated
 *          allocator would `delete[]` a pointer that was carved out of a slab.
 *          Treat the flag as unusable in its current state.
 */
inline constexpr bool debug_mem_observable = is_debug || use_openmp;

/// @brief int_alloc / int_free go through the aligned allocation (ALLOC_ALIGN).
inline constexpr bool alloc_align =
#ifdef ALLOC_ALIGN
    true;
#else
    false;
#endif

} // namespace alloc_config
