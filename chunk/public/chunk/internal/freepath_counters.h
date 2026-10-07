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

#include "chunk/build_options.h" // CHUNK_OPT_FAST_ALLOC

#if CHUNK_OPT_FAST_ALLOC
    #include <atomic> // std::memory_order_relaxed

    #include "fast_allocators/slab_arena.h" // fast_alloc::arena_report_enabled, fast_alloc::freepath_*
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
