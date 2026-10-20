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

#include <array>   // __chunk_stats (CHUNK_STATS only)
#include <cstddef> // std::size_t

#include "chunk/all_type_chunk.h"           // AllTypeChunk
#include "chunk/chunk_std.h"                // the declarations defined here (ChunkPoolParallelInterface)
#include "chunk/config.h"                   // build_config::use_openmp, use_fast_alloc
#include "chunk/internal/chunk_stats.h"     // ChunkStatsThr (CHUNK_STATS only)
#include "chunk/internal/invariant_check.h" // CHUNK_INVARIANT_FAILED (teardown_outside_region)
#include "chunk/internal/live_arrays.h"     // live_array_count: the teardown waits for the last array
#include "chunk/internal/process_verbs.h"   // compress_now_and_trim_pools, defined here
#include "chunk/singleton.h"                // SingletonChk
#include "chunk/PMP_CHUNK.h"                // CHUNK_PMP_IN_PARALLEL_REGION

namespace shrink_array
{

#if CHUNK_OPT_STATS
// Per-variant (chunk/chunk2/chunk2a) x per-thread counter table. Each thread
// bumps its own cache-line-aligned slot on the hot path (no atomics); the
// end-of-run report reduces over threads. See chunk_stats.h.
std::array<std::array<ChunkStatsThr, CHUNK_STATS_MAX_THR>, chunk_stats_n_variant> __chunk_stats;
#endif

// No #ifdef _OPENMP here. The bodies below are members of
// ChunkPoolParallelInterface (chunk_std.h), so they are TEMPLATES: a build
// instantiates only its own specialisation and never compiles the other's. They
// reach the facade through SingletonChk<AllT>, never SingletonChk<AllTypeChunk>:
// the calls must stay dependent, or phase-1 lookup resolves them against the
// concrete type at definition time and the other model fails to compile.

// --- OpenMP: the pools are per-rank -------------------------------------

template <typename AllT>
bool ChunkPoolParallelInterface<true, AllT>::need_load_balance_list_block_free()
{
    const auto& ens_type_chunk = SingletonChk<AllT>::getRef();

    return ens_type_chunk.need_load_balance_list_block_free();
}

template <typename AllT>
void ChunkPoolParallelInterface<true, AllT>::load_balance_list_block_free()
{
    const auto& ens_type_chunk = SingletonChk<AllT>::getRef();
    ens_type_chunk.load_balance_list_block_free();
}

template <typename AllT>
void ChunkPoolParallelInterface<true, AllT>::clear_all_list_block_free_thr()
{
    auto& ens_type_chunk = SingletonChk<AllT>::getRef();
    ens_type_chunk.clear_list_block_free_thr();
}

template <typename AllT>
void ChunkPoolParallelInterface<true, AllT>::delete_list_block_chunk_free_itself()
{
    const auto& ens_type_chunk = SingletonChk<AllT>::getRef();
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
        const auto& ens_type_chunk = SingletonChk<AllT>::getRef();
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
template struct ChunkPoolParallelInterface<build_config::use_openmp, AllTypeChunk>;

// --- Build-agnostic free functions over the policy -----------------------
// Thin forwarders, defined in EVERY build, so delete_chunk_allocator() and
// need_load_balance_list_block_free() link in both models -- doing nothing
// where there is nothing to do.

using chunk_pools = ChunkPoolParallelInterface<>;

bool need_load_balance_list_block_free()
{
    return chunk_pools::need_load_balance_list_block_free();
}

void load_balance_list_block_free()
{
    chunk_pools::load_balance_list_block_free();
}

void clear_all_list_block_free_thr()
{
    chunk_pools::clear_all_list_block_free_thr();
}

// The teardown verbs that pull structures from under the arrays -- the
// per-rank list containers, the serial allocator, the blocks -- do nothing
// while an array is alive: a global or static array is destroyed after main
// returns, i.e. after the teardown main ends with, and parks its buffers in
// those structures. They then stay reachable until the process exits.
// Emptying the free lists is always safe, and still done.

void delete_list_block_chunk_free_itself()
{
    if (internal::live_array_count() > 0)
    {
        return;
    }
    chunk_pools::delete_list_block_chunk_free_itself();
}

void delete_chunk_allocator()
{
    if (internal::live_array_count() > 0)
    {
        return;
    }
    chunk_pools::delete_chunk_allocator();
}

namespace
{
/// @brief Check builds: the teardown verbs are sequential points; called from
///        a parallel region, every thread of the team would empty the same
///        pools at once.
void teardown_outside_region(const char* verb)
{
    if constexpr (build_config::any_check)
    {
        if (CHUNK_PMP_IN_PARALLEL_REGION) [[unlikely]]
        {
            CHUNK_INVARIANT_FAILED("{}() called inside a parallel region: call it once, after the last region", verb);
        }
    }
}
} // namespace

/// @copydoc shrink_array::release_chunk_pools
void release_chunk_pools()
{
    teardown_outside_region("release_chunk_pools");
    if (internal::live_array_count() > 0)
    {
        chunk_pools::clear_all_list_block_free_thr(); // the parked buffers only
        return;
    }
    chunk_pools::release_chunk_pools();
}

/// @copydoc shrink_array::compress_now_and_trim_pools
std::size_t compress_now_and_trim_pools()
{
    const auto& ens_type_chunk = SingletonChk<AllTypeChunk>::getRef();
    if constexpr (chunk_config::use_fast_block_compression)
    {
        ens_type_chunk.force_chunk_to_compress();
    }
    return ens_type_chunk.trim_reuse_pools(0);
}

void destroy_block_chunk()
{
    teardown_outside_region("destroy_block_chunk");
    // Not part of the thread-model split: destroy_block_chunk() is a verb of
    // ManagerChunkInt, present in every build, so it stays a plain function.
    if constexpr (build_config::use_fast_alloc)
    {
        if (internal::live_array_count() > 0)
        {
            return; // see release_chunk_pools()
        }
        const auto& ens_type_chunk = SingletonChk<AllTypeChunk>::getRef();
        ens_type_chunk.destroy_block_chunk();
    }
}

} // namespace shrink_array