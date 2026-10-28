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

#include <array>   // g_chunk_stats (CHUNK_STATS only)
#include <cstddef> // std::size_t
#include <cstdio>  // std::fprintf (teardown_outside_region)

#include "chunk/all_type_chunk.h"           // AllTypeChunk
#include "chunk/zero_chunk.h"               // the declarations defined here (ChunkPoolParallelInterface)
#include "chunk/config.h"                   // build_config::threaded, use_fast_alloc
#include "chunk/internal/chunk_stats.h"     // ChunkStatsThr (CHUNK_STATS only)
#include "chunk/internal/invariant_check.h" // CHUNK_INVARIANT_FAILED (teardown_outside_region)
#include "chunk/internal/live_arrays.h"     // live_array_count: the teardown waits for the last array
#include "chunk/internal/process_verbs.h"   // compress_now_and_trim_pools, defined here
#include "chunk/singleton.h"                // SingletonChk
#include "chunk/PMP_CHUNK.h"                // CHUNK_PMP_IN_PARALLEL_REGION

namespace shrink_array
{

#if CHUNK_OPT_STATS
// Per-variant (zero/zero_one_value/zero_any) x per-thread counter table. Each thread
// bumps its own cache-line-aligned slot on the hot path (no atomics); the
// end-of-run report reduces over threads. See chunk_stats.h.
std::array<std::array<ChunkStatsThr, CHUNK_STATS_MAX_THR>, chunk_stats_n_variant> g_chunk_stats;
#endif

// No #ifdef _OPENMP here. The bodies below are members of
// ChunkPoolParallelInterface (zero_chunk.h), so they are TEMPLATES: a build
// instantiates only its own specialisation and never compiles the other's. They
// reach the facade through SingletonChk<AllT>, never SingletonChk<AllTypeChunk>:
// the calls must stay dependent, or phase-1 lookup resolves them against the
// concrete type at definition time and the other model fails to compile.

// --- OpenMP: the pools are per-rank -------------------------------------

template <typename AllT>
bool ChunkPoolParallelInterface<true, AllT>::need_load_balance_list_block_free()
{
    const auto& ens_type_chunk = SingletonChk<AllT>::get_ref();

    return ens_type_chunk.need_load_balance_list_block_free();
}

template <typename AllT>
void ChunkPoolParallelInterface<true, AllT>::load_balance_list_block_free()
{
    const auto& ens_type_chunk = SingletonChk<AllT>::get_ref();
    ens_type_chunk.load_balance_list_block_free();
}

template <typename AllT>
void ChunkPoolParallelInterface<true, AllT>::clear_all_list_block_free_thr()
{
    auto& ens_type_chunk = SingletonChk<AllT>::get_ref();
    ens_type_chunk.clear_list_block_free_thr();
}

template <typename AllT>
void ChunkPoolParallelInterface<true, AllT>::delete_list_block_chunk_free_itself()
{
    const auto& ens_type_chunk = SingletonChk<AllT>::get_ref();
    ens_type_chunk.delete_list_block_chunk_free_itself();
}

template <typename AllT>
void ChunkPoolParallelInterface<true, AllT>::release_chunk_pools()
{
    // Both halves, in this order: the lists must be emptied before the
    // containers holding them are released.
    clear_all_list_block_free_thr();
    delete_list_block_chunk_free_itself();
}

// --- Serial: one allocator, nothing per-rank ----------------------------

template <typename AllT>
void ChunkPoolParallelInterface<false, AllT>::delete_chunk_allocator()
{
    if constexpr (build_config::use_fast_alloc)
    {
        const auto& ens_type_chunk = SingletonChk<AllT>::get_ref();
        ens_type_chunk.delete_chunk_allocator();
    }
}

template <typename AllT>
void ChunkPoolParallelInterface<false, AllT>::release_chunk_pools()
{
    delete_chunk_allocator();
}

// The one instantiation of the pair: the other thread model's bodies are left
// uninstantiated rather than #ifdef'd out. Same device as param_chunk.cpp.
template struct ChunkPoolParallelInterface<build_config::threaded, AllTypeChunk>;

// --- Build-agnostic free functions over the policy -----------------------
// Thin forwarders, defined in EVERY build, so delete_chunk_allocator() and
// need_load_balance_list_block_free() link in both models -- doing nothing
// where there is nothing to do.

using ChunkPools = ChunkPoolParallelInterface<>;

namespace
{
/// @brief The teardown verbs, and shrink_to_fit()'s pool trim, are sequential
///        points: called from a parallel region, every thread of the team would
///        empty the same pools at once (double frees, broken lists). Check
///        builds abort; the others say so on stderr and skip the verb, every
///        time (a cold path: one in_parallel() per call).
/// @return true when the verb may run.
[[nodiscard]] bool teardown_outside_region(const char* verb)
{
    if (!CHUNK_PMP_IN_PARALLEL_REGION) [[likely]]
    {
        return true;
    }
    if constexpr (build_config::any_check)
    {
        CHUNK_INVARIANT_FAILED("{}() called inside a parallel region: call it once, after the last region", verb);
    }
    std::fprintf(stderr,
                 "shrink_array: %s() called inside a parallel region, skipped: call it once, after the last region\n",
                 verb);
    return false;
}
} // namespace

bool need_load_balance_list_block_free()
{
    return ChunkPools::need_load_balance_list_block_free();
}

void load_balance_list_block_free()
{
    ChunkPools::load_balance_list_block_free();
}

void clear_all_list_block_free_thr()
{
    if (!teardown_outside_region("clear_all_list_block_free_thr"))
    {
        return;
    }
    ChunkPools::clear_all_list_block_free_thr();
}

// The teardown verbs that pull structures from under the arrays -- the
// per-rank list containers, the serial allocator, the blocks -- do nothing
// while an array is alive: a global or static array is destroyed after main
// returns, i.e. after the teardown main ends with, and parks its buffers in
// those structures. They then stay reachable until the process exits.
// Emptying the free lists is always safe, and still done.

void delete_list_block_chunk_free_itself()
{
    if (!teardown_outside_region("delete_list_block_chunk_free_itself"))
    {
        return;
    }
    if (internal::live_array_count() > 0)
    {
        return;
    }
    ChunkPools::delete_list_block_chunk_free_itself();
}

void delete_chunk_allocator()
{
    if (!teardown_outside_region("delete_chunk_allocator"))
    {
        return;
    }
    if (internal::live_array_count() > 0)
    {
        return;
    }
    ChunkPools::delete_chunk_allocator();
}

/// @copydoc shrink_array::release_chunk_pools
void release_chunk_pools()
{
    if (!teardown_outside_region("release_chunk_pools"))
    {
        return;
    }
    if (internal::live_array_count() > 0)
    {
        ChunkPools::clear_all_list_block_free_thr(); // the parked buffers only
        return;
    }
    ChunkPools::release_chunk_pools();
}

/// @copydoc shrink_array::compress_now_and_trim_pools
std::size_t compress_now_and_trim_pools()
{
    if (!teardown_outside_region("compress_now_and_trim_pools"))
    {
        return 0;
    }
    const auto& ens_type_chunk = SingletonChk<AllTypeChunk>::get_ref();
    // Every rank's candidates (the threaded builds; outside a region, one region
    // in which each rank sweeps its own list): the pass that ends a phase, which
    // CHUNK_AGGRESSIVE_COMPRESSION skips -- force_chunk_to_compress() alone
    // folded the calling rank's list only. A no-op in a serial build.
    if (ens_type_chunk.have_chunk_to_compress_potentially())
    {
        ens_type_chunk.compress_chunk_to_compress_potentially(true);
    }
    // Then the calling rank's sweep, in every build: a serial zero_any queues
    // its candidates too, and folds them in batches of half the list
    // (perform_compression); the last, partial batch waited here for good
    // (nothing for a serial zero or zero_one_value: their counters compress on
    // the write).
    ens_type_chunk.force_chunk_to_compress();
    return ens_type_chunk.trim_reuse_pools(0);
}

void destroy_block_chunk()
{
    if (!teardown_outside_region("destroy_block_chunk"))
    {
        return;
    }
    // Not part of the thread-model split: destroy_block_chunk() is a verb of
    // ManagerChunkInt, present in every build, so it stays a plain function.
    if constexpr (build_config::use_fast_alloc)
    {
        if (internal::live_array_count() > 0)
        {
            return; // see release_chunk_pools()
        }
        const auto& ens_type_chunk = SingletonChk<AllTypeChunk>::get_ref();
        ens_type_chunk.destroy_block_chunk();
    }
}

} // namespace shrink_array