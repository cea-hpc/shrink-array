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

// Template method definitions for AllTypeChunk: the registration of a manager
// (register_manager / ensure_manager, called from the arrays' create) and the
// up-front registration of a type list (initialize<Types>).

#include <memory> // std::make_unique
#include <mutex>  // std::scoped_lock (registry_mutex_)

#include "chunk/all_type_chunk.h"           // the declarations these bodies belong to
#include "chunk/config.h"                   // build_config::chunk_dynamic, chunk_config::use_chunk*
#include "chunk/internal/invariant_check.h" // CHUNK_INVARIANT_FAILED
#include "chunk/internal/for_each_type.h"   // for_each_type
#include "chunk/internal/manager_chunk.h"   // ManagerChunk<CHUNK_T, T>
#include "chunk/singleton.h"                // SingletonChk<AllTypeChunk>, for ensure_manager
#include "chunk/chunk_traits.h"             // the Chunk / Chunk2 / Chunk2A declarations initialize<Types> names
#include "chunk/internal/link_check.h"      // CHUNK_LINK_CHECK: the program's settings, the library's
// Not the chunk definitions: register_manager<> is instantiated in the TU that
// builds the array (which has them), and including them here would be a cycle.

namespace shrink_array
{

template <typename ManagerT>
void AllTypeChunk::register_manager()
{
    const std::scoped_lock lock(registry_mutex_);

    // Idempotent under the lock (initialize<Types>() and a first array may both
    // register). A flag per manager type: no RTTI needed.
    if (registered_<ManagerT>)
    {
        return;
    }

    // ensure_manager() runs ensure_ready() first; initialize<Types>() runs
    // after init_indexes(). Reaching here unready is a library bug.
    if (!ready_)
    {
        CHUNK_INVARIANT_FAILED("a manager registered before ParamChunk::init_indexes() ran");
    }

    ManagerT::init_static_state(); // the per-rank structures, before anyone can reach them
    auto manager = std::make_unique<ManagerT>();

    // The whole per-type setup, before the manager is published.
    manager->resize_chunk_to_compress_potentially();
    init_manager(*manager, block_size_);
    if constexpr (build_config::chunk_dynamic)
    {
        for (const unsigned int size : registered_sizes_)
        {
            manager->register_chunk_size(size, block_size_);
        }
    }

    publish_manager(std::move(manager));
    registered_<ManagerT> = true;
}

template <typename ManagerT>
void AllTypeChunk::ensure_manager()
{
    // A magic static per manager type: concurrent first arrays wait on it.
    [[maybe_unused]] static const bool registered = []
    {
        link_check::CHUNK_LINK_CHECK(); // links only against a library built with this TU's settings
        ensure_ready();                 // init_indexes(true) if the program has not run it
        SingletonChk<AllTypeChunk>::getRef().register_manager<ManagerT>();
        return true;
    }();
}

template <template <typename> class CHUNK_T, typename Types>
void AllTypeChunk::fillManagerChunk()
{
    // One ManagerChunk per T of Types (a std::tuple).
    for_each_type<Types>([this]<typename T> { this->template register_manager<ManagerChunk<CHUNK_T, T>>(); });
}

template <typename Types>
void AllTypeChunk::initialize()
{
    // Only the variants this build compiles arrays for: a manager is not free
    // (init_manager primes a per-thread allocator, register_chunk_size a set
    // per geometry, and the arena never releases), so registering unused
    // variants multiplies the primed memory. Three independent `if constexpr`:
    // a build may select more than one (chunk_config::use_chunk*).
    if constexpr (chunk_config::use_chunk)
    {
        fillManagerChunk<Chunk, Types>();
    }
    if constexpr (chunk_config::use_chunk2)
    {
        fillManagerChunk<Chunk2, Types>();
    }
    if constexpr (chunk_config::use_chunk2a)
    {
        fillManagerChunk<Chunk2A, Types>();
    }
}

} // namespace shrink_array
