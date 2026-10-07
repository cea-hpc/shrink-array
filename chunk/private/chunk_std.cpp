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

#include "chunk/all_type_chunk.h"         // AllTypeChunk
#include "chunk/chunk_std.h"              // the declarations defined here (ChunkPoolParallelInterface)
#include "chunk/config.h"                 // build_config::use_openmp, use_fast_alloc
#include "chunk/internal/chunk_stats.h"   // ChunkStatsThr (CHUNK_STATS only)
#include "chunk/internal/process_verbs.h" // compress_now_and_trim_pools, defined here
#include "chunk/singleton.h"              // SingletonChk

namespace shrink_array
{

#ifdef CHUNK_STATS
// Per-variant (chunk/chunk2/chunk2a) x per-thread counter table. Each thread
// bumps its own cache-line-aligned slot on the hot path (no atomics); the
// end-of-run report reduces over threads. See chunk_stats.h.
std::array<std::array<ChunkStatsThr, CHUNK_STATS_MAX_THR>, CHUNK_STATS_N_VARIANT> __chunk_stats;
#endif

// No #ifdef _OPENMP left here. The bodies below are members of
// ChunkPoolParallelInterface (chunk_std.h), so they are TEMPLATES: a build
// instantiates only its own specialisation and never compiles the other's. They
// reach the facade through SingletonChk<AllT>, never SingletonChk<AllTypeChunk>
// -- the calls must stay dependent, or phase-1 lookup resolves them against the
// concrete type at definition time and we are back to needing the directive.

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

void delete_list_block_chunk_free_itself()
{
    chunk_pools::delete_list_block_chunk_free_itself();
}

void delete_chunk_allocator()
{
    chunk_pools::delete_chunk_allocator();
}

/// @copydoc shrink_array::release_chunk_pools
void release_chunk_pools()
{
    chunk_pools::release_chunk_pools();
}

/// @copydoc shrink_array::compress_now_and_trim_pools
std::size_t compress_now_and_trim_pools()
{
    auto& ens_type_chunk = SingletonChk<AllTypeChunk>::getRef();
    if constexpr (chunk_config::use_fast_block_compression)
    {
        ens_type_chunk.force_chunk_to_compress();
    }
    return ens_type_chunk.trim_reuse_pools(0);
}

void destroy_block_chunk()
{
    // Not part of the thread-model split: destroy_block_chunk() is a verb of
    // ManagerChunkInt, present in every build, so it stays a plain function.
    if constexpr (build_config::use_fast_alloc)
    {
        const auto& ens_type_chunk = SingletonChk<AllTypeChunk>::getRef();
        ens_type_chunk.destroy_block_chunk();
    }
}

} // namespace shrink_array