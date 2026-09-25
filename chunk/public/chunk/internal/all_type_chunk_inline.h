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

#include <cstdio>  // the refusal messages of register_manager
#include <cstdlib> // std::abort
#include <memory>
#include <mutex>
#include <tuple> // std::tuple (initialize<Types>)

#include "chunk/internal/for_each_type.h"
#include "chunk/internal/manager_chunk.h"
#include "chunk/singleton.h"    // SingletonChk<AllTypeChunk>, for ensure_manager
#include "chunk/chunk_traits.h" // the Chunk / Chunk2 / Chunk2A declarations initialize<Types> names
// Not the chunk definitions: a ManagerChunk<CHUNK_T, T> is only built at the
// instantiation of register_manager<>, in the TU that constructs the array (and
// so has the chunk), and including them here from the arrays' own headers
// (create_common calls ensure_manager) would be a cycle.

namespace shrink_array
{

template <typename ManagerT>
void AllTypeChunk::register_manager()
{
    const std::lock_guard<std::mutex> lock(registry_mutex_);

    // Idempotent under the lock: initialize<Types>() may name a type whose
    // first array already registered it, or the other way round.
    for (const auto& m : getListTypeChunk())
        if (dynamic_cast<const ManagerT*>(m.get()) != nullptr)
            return;

    if (!ready_)
    {
        std::fputs("shrink_array: an array was constructed before ParamChunk::init_indexes() ran -- "
                   "call it once at startup, before the first array.\n",
                   stderr);
        std::abort();
    }
    const int count = nb_managers_.load(std::memory_order_relaxed);
    if (count >= kMaxManagers)
    {
        std::fprintf(stderr, "shrink_array: more than %d (chunk template, element type) pairs in one process.\n",
                     kMaxManagers);
        std::abort();
    }

    auto manager = std::make_unique<ManagerT>();

    // The whole per-type setup, before anyone can see the manager: the
    // compression-candidate lists, the per-rank structures and allocators
    // (init_manager, by thread model), the runtime geometries declared so far.
    manager->resize_chunk_to_compress_potentially();
    init_manager(*manager, block_size_);
    if constexpr (build_config::chunk_dynamic)
    {
        for (const unsigned int size : registered_sizes_)
            manager->register_chunk_size(size, block_size_);
    }

    managers_[static_cast<std::size_t>(count)] = std::move(manager);
    nb_managers_.store(count + 1, std::memory_order_release); // publishes the slot
}

template <typename ManagerT>
void AllTypeChunk::ensure_manager()
{
    // A magic static per manager type: the first array of the type pays the
    // registration, every later one a guard load; concurrent first arrays of
    // the same type wait on the initialisation, as the language requires.
    [[maybe_unused]] static const bool registered = []
    {
        SingletonChk<AllTypeChunk>::getRef().register_manager<ManagerT>();
        return true;
    }();
}

template <template <typename> class CHUNK_T, typename Types>
void AllTypeChunk::fillManagerChunk()
{
    // For each T listed in Types (a std::tuple), register one ManagerChunk
    // for the (CHUNK_T, T) pair.
    for_each_type<Types>([this]<typename T>() { this->template register_manager<ManagerChunk<CHUNK_T, T>>(); });
}

template <typename Types>
void AllTypeChunk::initialize()
{
    // Only the variant this build actually compiles arrays for. Registering all
    // three cost a manager per (variant, type) that no array will ever use --
    // and a manager is not free: init_manager primes a per-thread allocator
    // for each entry, and register_chunk_size a set per geometry.
    //
    // Measured, 1e6 doubles, OpenMP, 4 threads, fast-alloc, before this change:
    // the value arena handed out 25 slabs of 4 MB (100 MB) of which ONE held
    // data -- 12 came from init_allocator_thr and 12 from register_chunk_size,
    // three variants x four threads each, for a program using ArrayChunk<double>
    // alone. A re-chunk to a second geometry primed twelve more (+48 MB), and
    // the arena never releases (deallocs = 0).
    //
    // The variant is a compile-time choice (the CHUNK / CHUNK2 / CHUNK2A
    // macros, mirrored by chunk_config::use_chunk*), so this costs nothing at
    // run time. Written as three independent `if constexpr` rather than an
    // else-chain: nothing forbids a build from selecting more than one.
    if constexpr (chunk_config::use_chunk)
        fillManagerChunk<Chunk, Types>();
    if constexpr (chunk_config::use_chunk2)
        fillManagerChunk<Chunk2, Types>();
    if constexpr (chunk_config::use_chunk2a)
        fillManagerChunk<Chunk2A, Types>();
}

} // namespace shrink_array
