// Copyright 2026 CEA
// SPDX-License-Identifier: Apache-2.0
/**
 * @file freepath_counters.h
 * @brief Where a released chunk buffer actually went, counted -- or not counted
 *        at all when there is no slab allocator to count for.
 *
 * @details A slab returns its pages only once every one of its blocks is free,
 *          so "the drain released millions of blocks" and "no block ever
 *          reached its allocator" look identical from outside: both leave
 *          madvise_calls at 0. The counters (defined in slab_arena.h, printed
 *          as [FREEPATH] next to [RECLAIM]) separate them.
 *
 *          The chunk layer compiles its release path in std-alloc builds too,
 *          where fast_allocators is neither linked nor on the include path --
 *          hence the macro: it is the counter in a fast-alloc build and nothing
 *          at all otherwise, so the call sites need no \#ifdef of their own.
 */
#pragma once

#include "chunk/chunk_define.h" // FAST_ALLOC

#ifdef FAST_ALLOC
    #include <atomic>
    #include "fast_allocators/slab_arena.h"
    // Counted only while the report that prints them is on (CHUNK_ARENA_REPORT
    // in the environment, read once): these sit on the materialisation and
    // release paths of every thread, and a shared counter there is a store to
    // a shared cache line per chunk -- see fast_alloc::IsolatedCounter.
    #define CHUNK_FREEPATH_INC(which)                                                                    \
        (fast_alloc::arena_report_enabled()                                                              \
             ? static_cast<void>(fast_alloc::freepath_##which().fetch_add(1, std::memory_order_relaxed)) \
             : static_cast<void>(0))
    /// Count one materialisation, by whether it carries an affinity key.
    #define CHUNK_FREEPATH_ALLOC(key) ((key) != 0u ? CHUNK_FREEPATH_INC(alloc_keyed) : CHUNK_FREEPATH_INC(alloc_key0))
#else
    #define CHUNK_FREEPATH_INC(which) ((void)0)
    #define CHUNK_FREEPATH_ALLOC(key) ((void)(key))
#endif
