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

#pragma once

/**
 * @file manager_chunk_policies.h
 * @brief The build-dependent implementation policies of ManagerChunk:
 *        ManagerChunkDynImpl (dynamic geometry) and ManagerChunkParallelImpl
 *        (the per-rank allocators, the deferred flush, the teardown of the
 *        reuse lists).
 *
 * They are bases of ManagerChunk and reach its state through
 * ManagerChunk<ChunkT, T>:: (it befriends them). Their bodies are in
 * manager_chunk_policies_inline.h, which manager_chunk.h includes once
 * ManagerChunk is complete.
 */

#include "chunk/config.h"                           // build_config::chunk_dynamic, build_config::threaded
#include "chunk/internal/manager_chunk_interface.h" // ManagerChunkDynInterface, ManagerChunkParallelInterface

namespace shrink_array
{
// Forward declarations breaking the include cycle chunk_func.h -> zero_chunk.h -> zero_proxy.h ->
// THIS file -> chunk_func.h: when chunk_func.h is the entry point, its
// templates are not yet declared here (#pragma once skips it), and the explicit
// load_balance_list_block_free_thr<T>() calls below need a declaration. Defined in chunk_func.h.
template <typename T>
bool need_load_balance_list_block_free_thr();
template <typename T>
void load_balance_list_block_free_thr();

/**
 * @brief The CHUNK_DYNAMIC-only half of the manager IMPLEMENTATION.
 *
 * @details Carries the override, because an `override` of a virtual that exists
 *          in one build only cannot be written conditionally in ManagerChunk.
 *          Inherits the interface VIRTUALLY, to share the one ManagerChunkInt
 *          brings in. The body reaches ManagerChunk's members (max_thr is
 *          per-instance) through a CRTP down-cast.
 */
template <template <typename> typename ChunkT, typename T, bool IsDynamic = build_config::chunk_dynamic>
struct ManagerChunkDynImpl; // primary left undefined: only the 2 specializations exist

/// @brief Dynamic-geometry builds: implements register_chunk_size().
template <template <typename> typename ChunkT, typename T>
struct ManagerChunkDynImpl<ChunkT, T, true> : public virtual ManagerChunkDynInterface<true>
{
    void register_chunk_size(unsigned int size, int block_size) override;
};

/// @brief Static-geometry builds: nothing to implement.
template <template <typename> typename ChunkT, typename T>
struct ManagerChunkDynImpl<ChunkT, T, false> : public virtual ManagerChunkDynInterface<false>
{
};

/**
 * @brief The allocator / free-list half of the manager IMPLEMENTATION.
 *
 * @details Carries the mode-specific overrides; @see ManagerChunkDynImpl. The
 *          bodies reach ManagerChunk's state through `mgr` (statics) and a CRTP
 *          down-cast (max_thr); ManagerChunk befriends the policies for that.
 */
template <template <typename> class ChunkT, typename T, bool Threaded = build_config::threaded>
class ManagerChunkParallelImpl; // primary left undefined: only the 2 specializations exist

/// @brief OpenMP builds: the per-rank allocator and free-list overrides.
template <template <typename> class ChunkT, typename T>
class ManagerChunkParallelImpl<ChunkT, T, true> : public virtual ManagerChunkParallelInterface<true>
{
  private:
    void init_allocators(int block_size) const override;
    void alloc_chunk_allocator_thr() override;

    void resize_all_list_block_free_thr() override;
    /// @brief Redistribute released buffers across the per-thread free-lists.
    /// @details A thread that frees far more than it allocates would otherwise hoard
    ///          buffers its neighbours need. Delegates to the free function that owns the policy.
    void load_balance_list_block_free() override
    {
        load_balance_list_block_free_thr<T>();
    }
    void flush_all_deferred(bool in_parallel, bool& have_flushed) override;
    void destroy_after_flush(bool in_parallel) override;

    void delete_list_block_chunk_free_itself() override;

    /// @brief Is the imbalance between per-thread free-lists worth a redistribution pass?
    /// @return True when load_balance_list_block_free() would be worth calling.
    [[nodiscard]] bool need_load_balance_list_block_free() const override
    {
        return need_load_balance_list_block_free_thr<T>();
    }
    std::size_t clear_list_block_free_prepare() override;
    void        clear_list_block_free_screen(int bucket) override;
    void        clear_list_block_free_sort(int bucket) override;
    void        clear_list_block_free_dedup(int slice, int nslices) override;
    void        clear_list_block_free_thr(int bucket) override;
};

/// @brief Serial builds: a single allocator, and nothing per-thread to manage.
template <template <typename> class ChunkT, typename T>
class ManagerChunkParallelImpl<ChunkT, T, false> : public virtual ManagerChunkParallelInterface<false>
{
  private:
    void init_allocator(int block_size) const override;
    /// @brief Destroy this type's single chunk allocator (serial build).
    void delete_chunk_allocator() override
    {
        // std-alloc: allocator_data is NullValueAllocator, destroy() is a no-op.
        ChunkT<T>::allocator_data->destroy();
    }
};

} // namespace shrink_array
